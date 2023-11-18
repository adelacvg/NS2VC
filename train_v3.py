import json
import os
from pathlib import Path
from datetime import datetime
from matplotlib import pyplot as plt
from tortoise_model import DiffusionTts
from unet1d.embeddings import TextTimeEmbedding
from unet1d.unet_1d_condition import UNet1DConditionModel
from tqdm import tqdm
from modules.utils import get_logger, plot_spectrogram_to_numpy
from vocos import Vocos
from torch import expm1, nn
import torchaudio
from dataset import NS2VCDataset, TextAudioCollate, TestDataset
import modules.commons as commons
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
import math
from multiprocessing import cpu_count
from pathlib import Path
from random import random
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

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

def exists(x):
    return x is not None

def cycle(dl):
    while True:
        for data in dl:
            yield data

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Diffusion_Encoder(nn.Module):
  def __init__(self,
    cfg,
    ):
    super().__init__()
    self.cfg = cfg
    self.model = DiffusionTts(self.cfg,**self.cfg['tortoise_diffusion'])

  def forward(self, x, data, t):
    assert torch.isnan(x).any() == False
    cvec, refer, spec, wav = data
    # x, timesteps, aligned_conditioning=None, conditioning_latent=None, precomputed_aligned_embeddings=None, conditioning_free=False, return_code_pred=False
    x = self.model(x, t, cvec, refer)
    return x

# tensor helper functions

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))
def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)
def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d
ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])
class NaturalSpeech2(nn.Module):
    def __init__(self,
        cfg,
        rvq_cross_entropy_loss_weight = 0.1,
        diff_loss_weight = 1.0,
        f0_loss_weight = 1.0,
        duration_loss_weight = 1.0,
        ddim_sampling_eta = 0,
        min_snr_loss_weight = False,
        min_snr_gamma = 5
        ):
        super().__init__()
        self.diff_model = Diffusion_Encoder(cfg)
        print("diff params: ", count_parameters(self.diff_model))
        self.dim = cfg['tortoise_diffusion']['in_channels']
        timesteps = cfg['train']['timesteps']

        beta_schedule_fn = linear_beta_schedule
        betas = beta_schedule_fn(timesteps)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim = 0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = timesteps

        self.sampling_timesteps = cfg['train']['sampling_timesteps']
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        register_buffer('posterior_variance', posterior_variance)

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))
        self.rvq_cross_entropy_loss_weight = rvq_cross_entropy_loss_weight
        self.diff_loss_weight = diff_loss_weight
        self.f0_loss_weight = f0_loss_weight
        self.duration_loss_weight = duration_loss_weight
        snr = alphas_cumprod / (1 - alphas_cumprod)

        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max = min_snr_gamma)

        register_buffer('loss_weight', maybe_clipped_snr)
    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )
    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, data = None):
        model_output = self.diff_model(x,data, t)
        t = t.type(torch.int64) 
        x_start = model_output
        pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)
    def sample_fun(self, x, t, data = None):
        model_output = self.diff_model(x,data, t)
        t = t.type(torch.int64) 
        x_start = model_output
        pred_noise = self.predict_noise_from_start(x, t, x_start)

        return x_start

    def p_mean_variance(self, x, t, data):
        preds = self.model_predictions(x, t, data)
        x_start = preds.pred_x_start

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, t: int, data):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((b,), t, device = device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_times, data=data)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, content, refer, lengths, refer_lengths, f0, uv, auto_predict_f0 = True):
        data = (content, refer, f0, 0, 0, lengths, refer_lengths, uv)
        # content, refer = self.pre_model.infer(data)
        shape = (content.shape[1], self.dim, content.shape[0])
        batch, device = shape[0], refer.device

        img = torch.randn(shape, device = device)
        imgs = [img]

        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            img, x_start = self.p_sample(img, t, (content,refer,lengths,refer_lengths))
            imgs.append(img)

        ret = img
        return ret

    @torch.no_grad()
    def ddim_sample(self, content, refer):
        data = (content, refer, 0, 0)
        # content, refer = self.pre_model.infer(data,auto_predict_f0=auto_predict_f0)
        shape = (content.shape[1], self.dim, content.shape[0])
        batch, device, total_timesteps, sampling_timesteps, eta = shape[0], refer.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta

        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device = device)
        imgs = [img]

        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, (content,refer))

            if time_next < 0:
                img = x_start
                imgs.append(img)
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

            imgs.append(img)

        ret = img
        return ret

    @torch.no_grad()
    def sample(self,
        c, refer, vocos,
        auto_predict_f0=True, sampling_timesteps=100, sample_method='unipc'
        ):
        if refer.shape[0]==2:
            refer = refer[0].unsqueeze(0)
        self.sampling_timesteps = sampling_timesteps
        # sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        if sample_method == 'ddpm':
            sample_fn = self.p_sample_loop
            audio = sample_fn(c, refer)
        elif sample_method == 'ddim':
            sample_fn = self.ddim_sample
            audio = sample_fn(c, refer)
        elif sample_method == 'dpmsolver':
            from sampler.dpm_solver import NoiseScheduleVP, model_wrapper, DPM_Solver
            noise_schedule = NoiseScheduleVP(schedule='discrete', betas=self.betas)
            def my_wrapper(fn):
                def wrapped(x, t, **kwargs):
                    ret = fn(x, t, **kwargs)
                    self.bar.update(1)
                    return ret

                return wrapped

            data = (c, refer, 0, 0)
            # content, refer = self.pre_model.infer(data,auto_predict_f0=auto_predict_f0)
            shape = (c.shape[0], self.dim, c.shape[2])
            batch, device, total_timesteps, sampling_timesteps, eta = shape[0], refer.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta
            audio = torch.randn(shape, device = device)
            model_fn = model_wrapper(
                my_wrapper(self.sample_fun),
                noise_schedule,
                model_type="x_start",  #"noise" or "x_start" or "v" or "score"
                model_kwargs={"data":data}
            )
            dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver++")

            steps = 40
            self.bar = tqdm(desc="sample time step", total=steps)
            audio = dpm_solver.sample(
                audio,
                steps=steps,
                order=2,
                skip_type="time_uniform",
                method="multistep",
            )
            self.bar.close()
        elif sample_method =='unipc':
            from sampler.uni_pc import NoiseScheduleVP, model_wrapper, UniPC
            noise_schedule = NoiseScheduleVP(schedule='discrete', betas=self.betas)

            def my_wrapper(fn):
                def wrapped(x, t, **kwargs):
                    ret = fn(x, t, **kwargs)
                    self.bar.update(1)
                    return ret

                return wrapped

            data = (c, refer, 0, 0)
            # content, refer = self.pre_model.infer(data,auto_predict_f0=auto_predict_f0)
            shape = (c.shape[0], self.dim, c.shape[2])
            batch, device, total_timesteps, sampling_timesteps, eta = shape[0], refer.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta
            audio = torch.randn(shape, device = device)*0.3
            model_fn = model_wrapper(
                my_wrapper(self.sample_fun),
                noise_schedule,
                model_type="x_start",  #"noise" or "x_start" or "v" or "score"
                model_kwargs={"data":data}
            )
            uni_pc = UniPC(model_fn, noise_schedule, variant='bh2')
            steps = 30
            self.bar = tqdm(desc="sample time step", total=steps)
            audio = uni_pc.sample(
                audio,
                steps=steps,
                order=2,
                skip_type="time_uniform",
                method="multistep",
            )
            self.bar.close()

        mel = audio
        vocos.to(audio.device)
        audio = vocos.decode(audio)

        if audio.ndim == 3:
            audio = rearrange(audio, 'b 1 n -> b n')

        return audio,mel 

    def q_sample(self, x_start, t, noise = None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def forward(self, data, vocos):
        cvec, refer, spec, wav = data
        b, d, n, device = *refer.shape, refer.device
        x_start = spec
        # get pre model outputs
        # content, refer, lf0, lf0_pred = self.pre_model(data)
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        noise = torch.randn_like(x_start)*0.4
        # noise sample
        x = self.q_sample(x_start = x_start, t = t, noise = noise)
        # predict and take gradient step
        model_out = self.diff_model(x,data, t)
        target = x_start

        loss = F.mse_loss(model_out, target, reduction = 'none')
        loss_diff = reduce(loss, 'b ... -> b (...)', 'mean')
        loss_diff = loss_diff * extract(self.loss_weight, t, loss.shape)
        loss_diff = loss_diff.mean()

        loss = loss_diff

        return loss, loss_diff, model_out, target

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
        self.accelerator = Accelerator()

        device = self.accelerator.device

        # model
        self.vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")
        self.model = NaturalSpeech2(cfg=self.cfg).to(device)
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
        model = accelerator.unwrap_model(self.model)
        model.load_state_dict(saved_state_dict)

    def train(self):
        # print(1)
        accelerator = self.accelerator
        device = accelerator.device

        if accelerator.is_main_process:
            logger = get_logger(self.logs_folder)
            writer = SummaryWriter(log_dir=self.logs_folder)
            writer_eval = SummaryWriter(log_dir=os.path.join(self.logs_folder, "eval"))

        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl)
                    cvec = data['cvec'].to(device)
                    refer = data['refer'].to(device)
                    spec = data['spec'].to(device)
                    wav = data['wav'].to(device)
                    data = [cvec,refer,spec,wav]

                    with self.accelerator.autocast():
                        loss, loss_diff, \
                        pred, target\
                         = self.model(data, self.vocos)
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
                    logger.info(f"Losses: {[loss_diff]}, step: {self.step}")

                    scalar_dict = {"loss/diff": loss_diff, "loss/all": total_loss,
                                 "loss/grad": grad_norm}
                    image_dict = {
                        "all/spec": plot_spectrogram_to_numpy(target[0, :, :].detach().unsqueeze(-1).cpu()),
                        "all/spec_pred": plot_spectrogram_to_numpy(pred[0, :, :].detach().unsqueeze(-1).cpu()),
                    }

                    utils.summarize(
                        writer=writer,
                        global_step=self.step,
                        images=image_dict,
                        scalars=scalar_dict
                    )

                if accelerator.is_main_process:

                    if self.step % self.save_and_sample_every == 0:
                        # print(1)
                        # c_padded, refer_padded, f0_padded, spec_padded, wav_padded, lengths, refer_lengths, uv_padded = next(iter(self.eval_dl))
                        data = next(self.eval_dl)
                        cvec, spec, audio, cvec_refer, spec_refer, audio_refer = data
                        cvec = cvec.to(device)
                        spec_refer = spec_refer.to(device)
                        with torch.no_grad():
                            self.model.eval()
                            milestone = self.step // self.save_and_sample_every
                            samples,mel = self.model.sample(cvec, spec_refer, self.vocos)
                            samples = samples.detach().cpu()
                            self.model.train()

                        torchaudio.save(str(self.logs_folder / f'sample-{milestone}.wav'), samples, 24000)
                        audio_dict = {}
                        audio_dict.update({
                                f"gen/audio": samples,
                                f"gt/audio": audio[0],
                                f"refer/audio": audio_refer[0],
                            })
                        image_dict = {
                                f"gen/mel": plot_spectrogram_to_numpy(mel[0, :, :].detach().unsqueeze(-1).cpu()),
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
if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()