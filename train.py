from omegaconf import OmegaConf
import torchaudio
from aa_model import AA_diffusion, denormalize_tacotron_mel, normalize_tacotron_mel
from dataset import NS2VCDataset, TestDataset, TextAudioCollate
from diffusion import SpacedDiffusion, space_timesteps, get_named_beta_schedule
import torch
import copy
from datetime import datetime
import json
from vocos import Vocos
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch
import os
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import AdamW
from accelerate import Accelerator
import functools
import random
import utils

import torch
from torch.cuda.amp import autocast

from diffusion import get_named_beta_schedule
from diffusion import space_timesteps, SpacedDiffusion

def do_spectrogram_diffusion(diffusion_model, diffuser, latents, conditioning_latents, temperature=1, verbose=True):
    """
    Uses the specified diffusion model to convert discrete codes into a spectrogram.
    """
    with torch.no_grad():
        output_seq_len = latents.shape[2]
        output_shape = (latents.shape[0], 100, output_seq_len)

        noise = torch.randn(output_shape, device=latents.device) * temperature
        mel = diffuser.p_sample_loop(diffusion_model, output_shape, noise=noise,
                                    model_kwargs= {
                                    "hint": latents,
                                    "refer": conditioning_latents
                                    },
                                    progress=verbose)
        return denormalize_tacotron_mel(mel)[:,:,:output_seq_len]

def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val
def get_grad_norm(model):
    total_norm = 0
    for name,p in model.named_parameters():
        try:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        except:
            print(name)
            pass
    total_norm = total_norm ** (1. / 2) 
    return total_norm
def cycle(dl):
    while True:
        for data in dl:
            yield data
def warmup(step):
    if step<1000:
        return float(step/1000)
    else:
        return 1
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
class Trainer(object):
    def __init__(self, cfg_path='config.yaml'):
        # ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        # self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
        self.accelerator = Accelerator()
        self.cfg = OmegaConf.load(cfg_path)
        # self.cfg = json.load(open(cfg_path))
        trained_diffusion_steps = 1000
        self.trained_diffusion_steps = 1000
        desired_diffusion_steps = 1000
        self.desired_diffusion_steps = 1000
        cond_free_k = 2.

        self.diffuser= SpacedDiffusion(use_timesteps=space_timesteps(trained_diffusion_steps, [desired_diffusion_steps]), model_mean_type='epsilon',
                           model_var_type='learned_range', loss_type='mse', betas=get_named_beta_schedule('linear', trained_diffusion_steps),
                           conditioning_free=False, conditioning_free_k=cond_free_k)
        self.infer_diffuser = SpacedDiffusion(use_timesteps=space_timesteps(trained_diffusion_steps, [50]), model_mean_type='epsilon',
                           model_var_type='learned_range', loss_type='mse', betas=get_named_beta_schedule('linear', trained_diffusion_steps),
                           conditioning_free=True, conditioning_free_k=cond_free_k, sampler='dpm++2m')
        self.diffusion = AA_diffusion(self.cfg)
        print("model params:", count_parameters(self.diffusion))

        # dataset and dataloader
        collate_fn = TextAudioCollate()
        ds = NS2VCDataset(self.cfg)
        self.ds = ds
        dl = DataLoader(ds, **self.cfg['dataloader'], collate_fn = collate_fn)

        dl = self.accelerator.prepare(dl)
        self.dataloader = cycle(dl)
        self.vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")
        self.train_steps = self.cfg['train']['train_steps']
        self.log_freq = self.cfg['train']['log_freq']
        if self.accelerator.is_main_process:
            eval_ds = TestDataset(self.cfg['data']['val_files'], self.cfg, self.vocos)
            self.eval_dl = DataLoader(eval_ds, batch_size = 1, shuffle = False, num_workers = self.cfg['train']['num_workers'])
            self.eval_dataloader = iter(cycle(self.eval_dl))

            now = datetime.now()
            self.logs_folder = Path(self.cfg['train']['logs_folder']+'/'+now.strftime("%Y-%m-%d-%H-%M-%S"))
            self.logs_folder.mkdir(exist_ok = True, parents=True)
        self.optimizer = AdamW(self.diffusion.parameters(),lr=self.cfg['train']['train_lr'], betas=(0.9, 0.999), weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=warmup)
        self.diffusion, self.dataloader, self.optimizer, self.scheduler = self.accelerator.prepare(self.diffusion, self.dataloader, self.optimizer, self.scheduler)
        self.dataloader = cycle(self.dataloader)
        self.step=0
        self.gradient_accumulate_every=self.cfg['train']['gradient_accumulate_every']
        self.unconditioned_percentage = self.cfg['train']['unconditioned_percentage']
    def _get_target_encoder(self, model):
        target_encoder = copy.deepcopy(model)
        set_requires_grad(target_encoder, False)
        for p in target_encoder.parameters():
            p.DO_NOT_TRAIN = True
        return target_encoder
    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return
        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.diffusion),
        }
        torch.save(data, str(self.logs_folder / f'model-{milestone}.pt'))

    def load(self, model_path):
        accelerator = self.accelerator
        device = accelerator.device
        data = torch.load(model_path, map_location=device)
        state_dict = data['model']
        self.step = data['step']
        model = self.accelerator.unwrap_model(self.diffusion)
        model.load_state_dict(state_dict)
    def train(self):
        accelerator = self.accelerator
        device = accelerator.device
        if accelerator.is_main_process:
            writer = SummaryWriter(log_dir=self.logs_folder)
            writer_eval = SummaryWriter(log_dir=os.path.join(self.logs_folder, 'eval'))
        with tqdm(initial = self.step, total = self.train_steps, disable = not accelerator.is_main_process) as pbar:
            while self.step < self.train_steps:
                total_loss = 0.
                # with torch.autograd.detect_anomaly():
                for _ in range(self.gradient_accumulate_every):
                    # spec refer cvec wav
                    data = next(self.dataloader)
                    if data==None:
                        continue
                    latent = data['cvec']    
                    # mel_recon_padded, mel_padded, mel_lengths, refer_padded, refer_lengths
                    x_start = normalize_tacotron_mel(data['spec'].to(device))
                    aligned_conditioning = latent 
                    conditioning_latent = normalize_tacotron_mel(data['refer'].to(device))
                    t = torch.randint(0, self.desired_diffusion_steps, (x_start.shape[0],), device=device).long().to(device)
                    with self.accelerator.autocast():
                        loss = self.diffuser.training_losses( 
                            model = self.diffusion, 
                            x_start = x_start,
                            t = t,
                            model_kwargs = {
                                "hint": aligned_conditioning,
                                "refer": conditioning_latent
                            },
                            )["loss"].mean()
                        unused_params =[]
                        model = self.accelerator.unwrap_model(self.diffusion)
                        unused_params.extend(list(model.refer_model.blocks.parameters()))
                        unused_params.extend(list(model.refer_model.out.parameters()))
                        unused_params.extend(list(model.refer_model.hint_converter.parameters()))
                        unused_params.extend(list(model.refer_enc.visual.proj))
                        extraneous_addition = 0
                        for p in unused_params:
                            extraneous_addition = extraneous_addition + p.mean()
                        loss = loss + 0*extraneous_addition
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)
                grad_norm = get_grad_norm(self.diffusion)
                accelerator.clip_grad_norm_(self.diffusion.parameters(), 1.0)
                pbar.set_description(f'loss: {total_loss:.4f}')
                accelerator.wait_for_everyone()
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler.step()
                accelerator.wait_for_everyone()
                # if accelerator.is_main_process:
                #     update_moving_average(self.ema_updater,self.ema_model,self.diffusion)
                if accelerator.is_main_process and self.step % self.log_freq == 0:
                    scalar_dict = {"loss": total_loss, "loss/grad": grad_norm, "lr":self.scheduler.get_last_lr()[0]}
                    utils.summarize(
                        writer=writer,
                        global_step=self.step,
                        scalars=scalar_dict
                    )
                if accelerator.is_main_process and self.step % self.cfg['train']['save_freq'] == 0:
                    data = next(self.eval_dataloader)
                    refer_audio = data['wav_refer']
                    gt_audio = data['wav']
                    cvec, refer = data['cvec'].to(device), data['refer'].to(device)
                    refer_padded = normalize_tacotron_mel(refer)
                    with torch.no_grad():
                        diffusion = self.accelerator.unwrap_model(self.diffusion)
                        mel = do_spectrogram_diffusion(diffusion, self.infer_diffuser,cvec,refer_padded,temperature=0.8)
                        mel = mel.detach().cpu()
                    
                    milestone = self.step // self.cfg['train']['save_freq'] 
                    gen = self.vocos.decode(mel)
                    torchaudio.save(str(self.logs_folder / f'sample-{milestone}.wav'), gen, 24000)
                    audio_dict = {}
                    audio_dict.update({
                            f"gen/audio": gen,
                            f"gt/audio": gt_audio,
                            f"refer/audio": refer_audio
                        })
                    image_dict = {
                        f"gen/mel": utils.plot_spectrogram_to_numpy(mel[0, :, :].detach().unsqueeze(-1).cpu()),
                    }
                    utils.summarize(
                        writer=writer_eval,
                        audios=audio_dict,
                        global_step=self.step,
                        images=image_dict,
                    )

                    keep_ckpts = self.cfg['train']['keep_ckpts']
                    if keep_ckpts > 0:
                        utils.clean_checkpoints(path_to_models=self.logs_folder, n_ckpts_to_keep=keep_ckpts, sort_by_time=True)
                    self.save(self.step//1000)
                self.step += 1
                pbar.update(1)
        accelerator.print('training complete')


if __name__ == '__main__':
    trainer = Trainer()
    # trainer.load('/home/hyc/tortoise_plus_zh/ttts/diffusion/logs/2023-11-06-18-18-28/model-79.pt')
    trainer.train()
