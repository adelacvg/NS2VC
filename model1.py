import math
import copy
from torch import autograd
import os
from pathlib import Path
from omegaconf import OmegaConf
import torch
import torchaudio
from tqdm.auto import tqdm
from cldm.cldm import denormalize_tacotron_mel
from dataset import NS2VCDataset, TestDataset, TextAudioCollate
from gaussian import GaussianDiffusion
from ldm.util import instantiate_from_config
from model import count_parameters, get_grad_norm
from ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock
import json
from accelerate import Accelerator
from vocos import Vocos
from torch.utils.data import DataLoader
from torch.optim import AdamW
from datetime import datetime
import utils
from torch.utils.tensorboard import SummaryWriter


def cycle(dl):
    while True:
        for data in dl:
            yield data
def create_model(config_path):
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model).cpu()
    print(f'Loaded model config from [{config_path}]')
    return model
def get_state_dict(d):
    return d.get('state_dict', d)
def load_state_dict(ckpt_path, location='cpu'):
    _, extension = os.path.splitext(ckpt_path)
    if extension.lower() == ".safetensors":
        import safetensors.torch
        state_dict = safetensors.torch.load_file(ckpt_path, device=location)
    else:
        state_dict = get_state_dict(torch.load(ckpt_path, map_location=torch.device(location)))
    state_dict = get_state_dict(state_dict)
    print(f'Loaded state_dict from [{ckpt_path}]')
    return state_dict
class Trainer(object):
    def __init__(
        self,
        cfg_path = './config.yaml',
    ):
        super().__init__()

        self.cfg = OmegaConf.load(cfg_path)
        self.accelerator = Accelerator()
        # model
        self.vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")
        self.model = create_model(cfg_path)
        print("model params:", count_parameters(self.model))
        # sampling and training hyperparameters
        self.save_and_sample_every = self.cfg['train']['save_and_sample_every']
        self.gradient_accumulate_every = self.cfg['train']['gradient_accumulate_every']
        self.train_num_steps = self.cfg['train']['train_num_steps']

        # dataset and dataloader
        collate_fn = TextAudioCollate()
        ds = NS2VCDataset(self.cfg)
        self.ds = ds
        dl = DataLoader(ds, **self.cfg['dataloader'], collate_fn = collate_fn)

        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)
        # optimizer

        self.opt = AdamW(self.model.parameters(), lr = self.cfg['train']['train_lr'], betas = self.cfg['train']['adam_betas'])

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            eval_ds = TestDataset(self.cfg['data']['val_files'], self.cfg, self.vocos)
            self.eval_dl = DataLoader(eval_ds, batch_size = 1, shuffle = False, num_workers = self.cfg['train']['num_workers'])
            self.eval_dl = iter(cycle(self.eval_dl))

            now = datetime.now()
            self.logs_folder = Path(self.cfg['train']['logs_folder']+'/'+now.strftime("%Y-%m-%d-%H-%M-%S"))
            self.logs_folder.mkdir(exist_ok = True, parents=True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator
        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

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
        # saved_state_dict['unconditioned_embedding'] = torch.nn.Parameter(torch.randn(1,100,1))
        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(saved_state_dict)

    def train(self):
        # print(1)
        accelerator = self.accelerator
        device = accelerator.device

        if accelerator.is_main_process:
            logger = utils.get_logger(self.logs_folder)
            writer = SummaryWriter(log_dir=self.logs_folder)
            writer_eval = SummaryWriter(log_dir=os.path.join(self.logs_folder, "eval"))

        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:

                # with autograd.detect_anomaly():
                data = next(self.dl)
                data = {k: v.to(self.device) for k, v in data.items()}
                with self.accelerator.autocast():
                    loss = accelerator.unwrap_model(self.model).training_step(data)

                    model = accelerator.unwrap_model(self.model)
                    unused_params =[]
                    unused_params.extend(list(model.refer_model.out.parameters()))
                    unused_params.extend(list(model.cond_stage_model.visual.proj))
                    unused_params.extend(list(model.refer_model.output_blocks.parameters()))
                    unused_params.extend(list(model.refer_model.output_blocks.parameters()))
                    unused_params.extend(list(model.unconditioned_embedding))
                    unused_params.extend(list(model.unconditioned_cat_embedding))
                    extraneous_addition = 0
                    for p in unused_params:
                        extraneous_addition = extraneous_addition + p.mean()
                    loss = loss + 0*extraneous_addition
                self.accelerator.backward(loss)

                grad_norm = get_grad_norm(self.model)
                accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                pbar.set_description(f'loss: {loss:.4f}')

                accelerator.wait_for_everyone()
                if (self.step+1)%self.gradient_accumulate_every==0:
                    self.opt.step()
                    self.opt.zero_grad()

                accelerator.wait_for_everyone()
############################logging#############################################
                if accelerator.is_main_process and self.step % 100 == 0:
                    logger.info('Train Epoch: {} [{:.0f}%]'.format(
                        self.step//len(self.ds),
                        100. * self.step / self.train_num_steps))
                    logger.info(f"Losses: {[loss]}, step: {self.step}")

                    scalar_dict = {"loss/diff": loss, "loss/grad": grad_norm}

                    utils.summarize(
                        writer=writer,
                        global_step=self.step,
                        scalars=scalar_dict
                    )

                if accelerator.is_main_process:

                    if self.step % self.save_and_sample_every == 0:
                        data = next(self.eval_dl)
                        data = {k: v.to(self.device) for k, v in data.items()}

                        with torch.no_grad():
                            model = accelerator.unwrap_model(self.model)
                            model.eval()
                            milestone = self.step // self.save_and_sample_every
                            log = model.log_images(data)
                            mel = log['samples'].detach().cpu()
                            mel = denormalize_tacotron_mel(mel)
                            model.train()
                        gen = self.vocos.decode(mel)
                        torchaudio.save(str(self.logs_folder / f'sample-{milestone}.wav'), gen, 24000)
                        audio_dict = {}
                        audio_dict.update({
                                f"gen/audio": gen,
                                f"gt/audio": data['wav'][0],
                                f"refer/audio": data['wav_refer'][0],
                            })
                        image_dict = {
                                f"gen/mel": utils.plot_spectrogram_to_numpy(mel[0, :, :].detach().unsqueeze(-1).cpu()),
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
# example

if __name__ == '__main__':
    trainer = Trainer()
    trainer.load('/home/hyc/NS2VC/logs/vc/2023-12-16-18-02-48/model-140.pt')
    trainer.train()
