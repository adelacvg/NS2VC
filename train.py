import copy
import json
import os
from pathlib import Path
from datetime import datetime
from matplotlib import pyplot as plt
from modules.diffusers import SpacedDiffusion, get_named_beta_schedule, space_timesteps
from tortoise_model import DiffusionTts
from modules.utils import plot_spectrogram_to_numpy
from vocos import Vocos
from torch import expm1, nn
import torchaudio
from dataset import NS2VCDataset, TextAudioCollate, TestDataset
import modules.commons as commons
from accelerate import Accelerator
import math
from multiprocessing import cpu_count
from pathlib import Path
import random
from functools import partial
from collections import namedtuple
from torch.utils.tensorboard import SummaryWriter
import logging
import torch
import torch.nn.functional as F
from torch import nn, einsum
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader



import modules.utils as utils

from tqdm.auto import tqdm

TACOTRON_MEL_MAX = 5.5451774444795624753378569716654
TACOTRON_MEL_MIN = -11.512925464970228420089957273422

CONTENT_MAX = 5.5451774444795624753378569716654
CONTENT_MIN = -5.5451774444795624753378569716654

def denormalize_tacotron_mel(norm_mel):
    return ((norm_mel+1)/2)*(TACOTRON_MEL_MAX-TACOTRON_MEL_MIN)+TACOTRON_MEL_MIN

def normalize_tacotron_mel(mel):
    return 2 * ((mel - TACOTRON_MEL_MIN) / (TACOTRON_MEL_MAX - TACOTRON_MEL_MIN)) - 1

def denormalize_content(norm_content):
    return ((norm_content+1)/2)*(CONTENT_MAX-CONTENT_MIN)+CONTENT_MIN

def normalize_content(content):
    return 2 * ((content - CONTENT_MIN) / (CONTENT_MAX - CONTENT_MIN)) - 1


def exists(x):
    return x is not None

def cycle(dl):
    while True:
        for data in dl:
            yield data

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def do_spectrogram_diffusion(diffusion_model, diffuser, latents, conditioning_latents, temperature=1, verbose=True):
    """
    Uses the specified diffusion model to convert discrete codes into a spectrogram.
    """
    with torch.no_grad():
        output_seq_len = latents.shape[-1] # This diffusion model converts from 22kHz spectrogram codes to a 24kHz spectrogram signal.
        output_shape = (latents.shape[0], 100, output_seq_len)
        precomputed_embeddings = diffusion_model.timestep_independent(latents, conditioning_latents, output_seq_len, False)

        noise = torch.randn(output_shape, device=latents.device) * temperature
        mel = diffuser.sample_loop(diffusion_model, output_shape, noise=noise,
                                    model_kwargs={'precomputed_aligned_embeddings': precomputed_embeddings,
                                                  "conditioning_latent": conditioning_latents},
                                    progress=verbose)
        return denormalize_tacotron_mel(mel)[:,:,:output_seq_len]


def get_grad_norm(model):
    total_norm = 0
    for name,p in model.named_parameters():
        try:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        except:
            print(name)
    total_norm = total_norm ** (1. / 2) 
    return total_norm
def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val
def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('numba').setLevel(logging.WARNING)
class Trainer(object):
    def __init__(
        self,
        cfg_path = './config.json',
    ):
        super().__init__()

        self.cfg = json.load(open(cfg_path))
        # ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        # self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
        self.accelerator = Accelerator()

        device = self.accelerator.device
        self.unconditioned_percentage = 0.1

        trained_diffusion_steps = 4000
        self.trained_diffusion_steps = 4000
        desired_diffusion_steps = 200
        self.desired_diffusion_steps = 200
        cond_free_k = 2.
        # model
        self.vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")
        self.diffuser= SpacedDiffusion(use_timesteps=space_timesteps(trained_diffusion_steps, [desired_diffusion_steps]), model_mean_type='epsilon',
                           model_var_type='learned_range', loss_type='mse', betas=get_named_beta_schedule('linear', trained_diffusion_steps),
                           conditioning_free=False, conditioning_free_k=cond_free_k)
        self.infer_diffuser = SpacedDiffusion(use_timesteps=space_timesteps(trained_diffusion_steps, [30]), model_mean_type='epsilon',
                           model_var_type='learned_range', loss_type='mse', betas=get_named_beta_schedule('linear', trained_diffusion_steps),
                           conditioning_free=True, conditioning_free_k=cond_free_k,sampler='dpm++2m')
        self.model = DiffusionTts(self.cfg,**self.cfg['tortoise_diffusion'])
        print("All params:", count_parameters(self.model))
        # sampling and training hyperparameters

        self.save_and_sample_every = self.cfg['train']['save_and_sample_every']

        self.batch_size = self.cfg['train']['train_batch_size']
        self.gradient_accumulate_every = self.cfg['train']['gradient_accumulate_every']

        self.train_num_steps = self.cfg['train']['train_num_steps']

        # dataset and dataloader
        collate_fn = TextAudioCollate()
        ds = NS2VCDataset(self.cfg['data']['training_files'], self.cfg)
        self.ds = ds
        dl = DataLoader(ds, batch_size = self.cfg['train']['train_batch_size'], shuffle = True, pin_memory = True, num_workers = self.cfg['train']['num_workers'], collate_fn = collate_fn)

        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)
        # print(1)
        # optimizer

        self.opt = AdamW(self.model.parameters(), lr = self.cfg['train']['train_lr'], betas = self.cfg['train']['adam_betas'])

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            # self.ema_model = self._get_target_encoder(self.model).to(self.accelerator.device)
            eval_ds = TestDataset(self.cfg['data']['val_files'], self.cfg, self.vocos)
            self.eval_dl = DataLoader(eval_ds, batch_size = 1, shuffle = False, num_workers = 1)
            self.eval_dl = iter(cycle(self.eval_dl))

            now = datetime.now()
            self.logs_folder = Path(self.cfg['train']['logs_folder']+'/'+now.strftime("%Y-%m-%d-%H-%M-%S"))
            self.logs_folder.mkdir(exist_ok = True, parents=True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator
        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)
    def _get_target_encoder(self, model):
        target_encoder = copy.deepcopy(model)
        set_requires_grad(target_encoder, False)
        for p in target_encoder.parameters():
            p.DO_NOT_TRAIN = True
        return target_encoder
    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
        }

        torch.save(data, str(self.logs_folder / f'model-{milestone}.pt'))

    def load(self, model_path):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(model_path, map_location=device)

        self.step = data['step']

        saved_state_dict = data['model']
        model = accelerator.unwrap_model(self.model)
        model.load_state_dict(saved_state_dict)

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        if accelerator.is_main_process:
            logger = utils.get_logger(self.logs_folder)
            writer = SummaryWriter(log_dir=self.logs_folder)
            writer_eval = SummaryWriter(log_dir=os.path.join(self.logs_folder, "eval"))

        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl)
                    # 'cvec':c_padded,
                    # 'refer':refer_padded,
                    # 'spec':spec_padded,
                    # 'wav':wav_padded
                    cvec, refer, spec, wav = data['cvec'].to(device), data['refer'].to(device), data['spec'].to(device), data['wav'].to(device)
                    x_start = normalize_tacotron_mel(spec)
                    aligned_conditioning = normalize_content(cvec)
                    conditioning_latent = normalize_tacotron_mel(refer)
                    t = torch.randint(0, self.desired_diffusion_steps, (x_start.shape[0],), device=device).long().to(device)
                    with self.accelerator.autocast():
                        loss = self.diffuser.training_losses( 
                            model = self.model,
                            x_start = x_start,
                            t = t,
                            model_kwargs = {
                                "aligned_conditioning": aligned_conditioning,
                                "conditioning_latent": conditioning_latent
                            },
                            )["loss"].mean()
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                grad_norm = get_grad_norm(self.model)
                accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                pbar.set_description(f'loss: {total_loss:.4f}')

                accelerator.wait_for_everyone()

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()
                # print(loss_diff, loss_f0, ce_loss)
############################logging#############################################
                if accelerator.is_main_process and self.step % 100 == 0:
                    logger.info('Train Epoch: {} [{:.0f}%]'.format(
                        self.step//len(self.ds),
                        100. * self.step / self.train_num_steps))
                    logger.info(f"Losse: {loss}, step: {self.step}")

                    scalar_dict = {"loss/diff": loss, "loss/grad": grad_norm}
                
                    utils.summarize(
                        writer=writer,
                        global_step=self.step,
                        scalars=scalar_dict
                    )
                # if accelerator.is_main_process:
                #     update_moving_average(self.ema_updater,self.ema_model,self.model)
                if accelerator.is_main_process:

                    if self.step % self.save_and_sample_every == 0:
                        # c_padded, refer_padded, f0_padded, spec_padded, wav_padded, lengths, refer_lengths, uv_padded = next(iter(self.eval_dl))
                        data = next(self.eval_dl)
                        c, spec, audio, \
                        c_refer, spec_refer, audio_refer = data
                        c, refer = c.to(device), spec_refer.to(device)
                        c = normalize_content(c)
                        refer = normalize_tacotron_mel(refer)

                        milestone = self.step // self.save_and_sample_every
                        with torch.no_grad():
                            infer_model = self.accelerator.unwrap_model(self.model)
                            infer_model.eval()
                            mel = do_spectrogram_diffusion(infer_model, self.infer_diffuser,c,refer,temperature=0.8)
                            mel = mel.detach().cpu()
                            infer_model.train()
                        wav_out = self.vocos.decode(mel)
                        torchaudio.save(str(self.logs_folder / f'sample-{milestone}.wav'), wav_out, 24000)
                        audio_dict = {}
                        audio_dict.update({
                                f"gen/audio": wav_out,
                                f"input/audio": audio[0],
                                f"refer/audio": audio_refer[0]
                            })
                        image_dict = {
                                f"val/input_mel" : plot_spectrogram_to_numpy(spec[0, :, :].detach().unsqueeze(-1).cpu()),
                                f"val/output_mel": plot_spectrogram_to_numpy(mel[0, :, :].detach().unsqueeze(-1).cpu()),
                        }
                        utils.summarize(
                            writer=writer_eval,
                            global_step=self.step,
                            audios=audio_dict,
                            images=image_dict,
                            audio_sampling_rate=24000
                        )
                        keep_ckpts = self.cfg['train']['keep_ckpts']
                        if keep_ckpts > 0:
                            utils.clean_checkpoints(path_to_models=self.logs_folder, n_ckpts_to_keep=keep_ckpts, sort_by_time=True)
                        self.save(milestone)

                self.step += 1
                pbar.update(1)

        accelerator.print('training complete')
if __name__ == '__main__':
    trainer = Trainer()
    # trainer.load('/home/hyc/NS2VC/logs/vc/2023-11-16-01-48-17/model-131.pt')
    trainer.train()