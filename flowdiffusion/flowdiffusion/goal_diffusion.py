import math
import copy
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

from torch.optim import Adam

from torchvision import transforms as T, utils

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from PIL import Image
from tqdm.auto import tqdm
from ema_pytorch import EMA

from accelerate import Accelerator

import matplotlib.pyplot as plt
import numpy as np


# from denoising_diffusion_pytorch.version import __version__
__version__ = "0.0"
# from flow_viz import flow_to_image, flow_tensor_to_image
import os

from pynvml import *

def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


# constants
ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# helpers functions
def tensors2vectors(tensors):
    def tensor2vector(tensor):
        flo = (tensor.permute(1, 2, 0).numpy()-0.5)*1000
        r = 8
        plt.quiver(flo[::-r, ::r, 0], -flo[::-r, ::r, 1], color='r', scale=r*20)
        plt.savefig('temp.jpg')
        plt.clf()
        return plt.imread('temp.jpg').transpose(2, 0, 1)
    return torch.from_numpy(np.array([tensor2vector(tensor) for tensor in tensors])) / 255

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# small helper modules

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample(dim, dim_out = None):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)
    )

class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, 'o ... -> o 1 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1 1', partial(torch.var, unbiased = False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) * (var + eps).rsqrt() * self.g

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

# building block modules

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale
        v = v / (h * w)

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)

# model


# gaussian diffusion trainer class

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

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


   
class GoalGaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        channels=3,
        timesteps = 1000,
        sampling_timesteps = 100,
        loss_type = 'l1',
        objective = 'pred_noise',
        beta_schedule = 'sigmoid',
        schedule_fn_kwargs = dict(),
        ddim_sampling_eta = 0.,
        auto_normalize = True,
        min_snr_loss_weight = False, # https://arxiv.org/abs/2303.09556
        min_snr_gamma = 5
    ):
        super().__init__()
        # assert not (type(self) == GoalGaussianDiffusion and model.channels != model.out_dim)
        # assert not model.random_or_learned_sinusoidal_cond

        self.model = model

        self.channels = channels

        self.image_size = image_size

        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == 'cosine':
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # derive loss weight
        # snr - signal noise ratio

        snr = alphas_cumprod / (1 - alphas_cumprod)

        # https://arxiv.org/abs/2303.09556

        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max = min_snr_gamma)

        if objective == 'pred_noise':
            register_buffer('loss_weight', maybe_clipped_snr / snr)
        elif objective == 'pred_x0':
            register_buffer('loss_weight', maybe_clipped_snr)
        elif objective == 'pred_v':
            register_buffer('loss_weight', maybe_clipped_snr / (snr + 1))

        # auto-normalization of data [0, 1] -> [-1, 1] - can turn off by setting it to be False

        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, x_cond, task_embed,  clip_x_start=False, rederive_pred_noise=False):
        # task_embed = self.text_encoder(goal).last_hidden_state
        model_output = self.model(torch.cat([x, x_cond], dim=1), t, task_embed) #modeloutput是一张噪声图，可以尝试把这里的输入改成batch
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':  #this one
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, x_cond, task_embed,  clip_denoised = False):
        preds = self.model_predictions(x, t, x_cond, task_embed)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, t: int, x_cond, task_embed):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((b,), t, device = x.device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x, batched_times, x_cond, task_embed, clip_denoised = True)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, shape, x_cond, task_embed, return_all_timesteps=False):
        batch, device = shape[0], self.betas.device

        img = torch.randn(shape, device=device)
        imgs = [img]

        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            # self_cond = x_start if self.self_condition else None
            img, _ = self.p_sample(img, t, x_cond, task_embed)
            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        ret = self.unnormalize(ret)
        return ret

    @torch.no_grad()
    def ddim_sample(self, img, shape, x_cond, task_embed, target_timestep, return_all_timesteps=False):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        if target_timestep is None:
            target_timestep = 100
            times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)
        else:
            times = torch.linspace(-1, target_timestep - 1, steps = sampling_timesteps + 1)     #100       10
        #times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        #img = torch.randn(shape, device=device)
        img = img
        imgs = [img]
        draft_tokens = [img]
        t_list = []

        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'): # [(99, 89), (89, 79), (79, 69), (69, 59), (59, 49), (49, 39), (39, 29), (29, 19), (19, 9), (9, -1)]
            time_cond = torch.full((batch,), time, device = device, dtype = torch.long) # 99 89 79....9 time_cond: tensor([49], device='cuda:0')
            t_list.append(time_cond)
            # self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, x_cond, task_embed, clip_x_start = False, rederive_pred_noise = True)

            if time_next < 0:
                img = x_start
                imgs.append(img)
                draft_tokens.append(img)
                continue #直接结束for循环

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)
            # img就是预测的x_89，x_start是预测的x_0
            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

            imgs.append(img)
            draft_tokens.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        ret = self.unnormalize(ret)
        #采样10步时，draft_tokens应为包含11个tensor的list[img(x99), x89, ..., x9, x_start]  img(x99),
        #t_list应为包含10个tensor的list[99, 89, ..., 9]
        return ret, draft_tokens, t_list, target_timestep
    
    @torch.no_grad()
    def target_sample(self, t_list, draft_tokens, shape, x_cond, task_embed, target_timestep, return_all_timesteps=False):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        if target_timestep is None:
            target_timestep = 100
            times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)
        else:
            times = torch.linspace(-1, target_timestep - 1, steps = sampling_timesteps + 1)

        #times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        # = torch.randn(shape, device=device)
        target_tokens = [[token] for token in draft_tokens[:-1]]  # [[img(x99)], [x89], ..., [x9]] len = 10
                                                                  # 
        draft_tokens_batchshape = torch.cat(draft_tokens, dim=0)  # batch化draft_tokens  [10,21,256,256]
        batch = draft_tokens_batchshape.shape[0]  # batch size
        x_cond_batch = x_cond.repeat(batch, 1, 1, 1)  # batch化x_cond
        t_batch = torch.cat(t_list, dim=0)  # batch化t_list  长度暂时也是10 
        task_batch = task_embed.repeat(batch, 1)  # batch化task_embed

        x_start = None
#[99, 98, 97, 96, 95, 94, 93, 92, 91, 90, 89]
        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'): # [(99, 89), (89, 79), (79, 69), (69, 59), (59, 49), (49, 39), (39, 29), (29, 19), (19, 9), (9, -1)]
            #time_cond = torch.full((batch,), time, device = device, dtype = torch.long) # 99 89 79....9
            # self_cond = x_start if self.self_condition else None
            # 先batch化再传入model

            draft_tokens_batch = torch.cat([tokens[-1] for tokens in target_tokens], dim=0)  # 取每组的最后一个token

            pred_noise, x_start, *_ = self.model_predictions(draft_tokens_batch, t_batch, x_cond_batch, task_batch, clip_x_start = False, rederive_pred_noise = True)
            # pred_noise和x_start都应该是一个batch [10, 21, 256, 256]
            if time_next < 0:
                for i in range(batch):
                    target_tokens[i].append(x_start[i:i+1])
                break

            alpha = self.alphas_cumprod[t_batch]
            alpha_next = self.alphas_cumprod[t_batch - 1]

            alpha = alpha.view(-1, 1, 1, 1)
            alpha_next = alpha_next.view(-1, 1, 1, 1)

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(x_start)
            # img就是预测的x_t-1，x_start是预测的x_0
            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

            for i in range(batch):
                target_tokens[i].append(img[i:i+1]) #要改为对应位置加上img这个list

            t_batch = t_batch - 1  # 更新t_batch，下一次循环时t_batch-1
        #ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)
        #ret = self.unnormalize(ret)
        target_tokens = [tokens[-1] for tokens in target_tokens]  # [x89, x79, x69, x59, x49, x39, x29, x19, x9, x0] len = 10
        return target_tokens

    @torch.no_grad()
    def speculative_decoding(self, shape, x_cond, text, draft_model, target_timestep=100, return_all_timesteps=False):
        device = self.betas.device
        img = torch.randn(shape, device=device) #初始化高斯噪声
        imgs = []
        good_tokens = [] 
        current_timestep = target_timestep  # 当前要从哪个时间步开始采样
        max_retries = 3  # 防止无限循环
        retry_count = 0

        while True:  # 保持原始注释中的while True逻辑
            print(f"[Speculative Loop] Starting from timestep: {current_timestep} (retry: {retry_count})")
            
            # 第一步可以加上T相关的参数target_timestpe方便后面代码
            # 下一个循环img就要改成good_tokens[-1], 并且target_timestep要改成reject_index相关的作为下一次开始draft采样的开始时间步
            if len(good_tokens) > 0:
                # 从最后一个good token开始继续采样  
                start_img = good_tokens[-1]
                print(f"[Speculative Loop] Continue from good_tokens[-1], total good: {len(good_tokens)}")
            else:
                # 第一次循环，从初始噪声开始
                start_img = img
                print("[Speculative Loop] Starting from initial noise")
            
            # 检查时间步是否有效
            if current_timestep <= 0:
                print("[Speculative Loop] Timestep <= 0, ending generation")
                if good_tokens:
                    imgs.append(good_tokens[-1])
                else:
                    imgs.append(start_img)
                break
            
            # Draft采样：从current_timestep开始快速采样
            ret, draft_tokens, t_list, target_timestep = self.ddim_sample(
                start_img, shape, x_cond, text, current_timestep, return_all_timesteps=True
            ) 

            # Target验证：在draft基础上精确采样
            target_tokens = self.target_sample(
                t_list, draft_tokens, shape, x_cond, text, current_timestep, return_all_timesteps=True
            )
        
            # 比较draft_tokens和target_tokens, 对应位置dt去头tt去尾
            # draft_tokens = [img(x99), x89, ..., x9, x_start] len = 11
            # target_tokens[x89, x79, x69, x59, x49, x39, x29, x19, x9, x0] len = 10
            print(f"[Speculative Loop] Draft tokens: {len(draft_tokens)}, Target tokens: {len(target_tokens)}")
            
            is_all_accept, reject_index = self.compare_tokens(draft_tokens, target_tokens, sim_threshold=0.75)

            if is_all_accept:
                # if all tokens are the same, break
                good_tokens.extend(target_tokens)
                print(f"[Speculative Loop] All tokens accepted, breaking the loop. Total: {len(good_tokens)}")
                imgs.append(ret)  # 使用原始的ret作为最终结果
                break
            else:
                # 等等，我为什么要验证？因为target是在draft的基础上精细生成的，已经有了我直接用不就好了
                # 但问题是误差累积，假设draft中x59已经误差比较大了，即x59和x59'不接受
                # 那么在x59的基础上精细生成的x49'也没法用啊
                # 所以需要从x59'开始重新用draft开始采样，因为x59'是从x69采样得到的，这里假设x69和x69'是接受的
                
                if reject_index > 0:
                    # reject_index后面的丢掉，保留前面accept的部分
                    accepted_tokens = target_tokens[:reject_index]
                    good_tokens.extend(accepted_tokens)
                    print(f"[Speculative Loop] Accepted {reject_index}/{len(target_tokens)} tokens")
                    
                    # 关键修正：从t_list中获取reject位置对应的真实时间步
                    # 这样确保下次采样从正确的时间步开始
                    if reject_index < len(t_list):
                        current_timestep = t_list[reject_index].item()
                        print(f"[Speculative Loop] Next timestep from t_list[{reject_index}]: {current_timestep}")
                        retry_count = 0  # 重置重试计数
                    else:
                        # 如果reject_index超出t_list范围，说明采样已经结束
                        current_timestep = 0
                        print("[Speculative Loop] Reject index exceeds t_list, ending generation")
                        if good_tokens:
                            imgs.append(good_tokens[-1])
                        else:
                            imgs.append(start_img)
                        break
                else:
                    print("[Speculative Loop] No tokens accepted in this round, retrying")
                    retry_count += 1
                    if retry_count >= max_retries:
                        print(f"[Speculative Loop] Max retries ({max_retries}) reached, ending generation")
                        if good_tokens:
                            imgs.append(good_tokens[-1])
                        else:
                            imgs.append(start_img)
                        break
                    # 如果没有token被接受，保持相同的时间步重试
                    
        return good_tokens, imgs 

    def compare_tokens(self, draft_tokens, target_tokens, sim_threshold=0.75):
        """
        draft_tokens: [x99, x89, ..., x0]  → length = N
        target_tokens: [x98', x88', ..., x0']  → length = N

        比较 draft_tokens[i] 和 target_tokens[i]，一一对应
        """
        assert len(draft_tokens) - 1 == len(target_tokens), "Token list mismatch"

        is_all_accept = True
        reject_index = -1

        # 从 x89 对比 x88' 开始
        for i in range(len(target_tokens)):
            dt = draft_tokens[i + 1]   # draft_tokens 从 x99 开始, draft_tokens[1] == x89
            tt = target_tokens[i]      # target_tokens 从 x89' 开始

            # flatten to compute cosine similarity or L2
            dt_flat = dt.flatten(start_dim=1)   # [B, C*H*W]
            tt_flat = tt.flatten(start_dim=1)

            # cosine similarity: higher is better
            cos_sim = F.cosine_similarity(dt_flat, tt_flat, dim=1)  # [B]
            avg_sim = cos_sim.mean().item()

            print(f"[Step {i}] Cosine Similarity: {avg_sim:.4f}")

            if avg_sim < sim_threshold:
                is_all_accept = False
                reject_index = i
                break

        return is_all_accept, reject_index

    @torch.no_grad()   #sample-speculative_decoding-sample_fn
    def sample(self, x_cond, task_embed, draft_model=True, target_timestep=99, batch_size = 16, return_all_timesteps = False):
        image_size, channels = self.image_size, self.channels
        
        # 如果提供了draft_model，使用speculative decoding
        if draft_model is True:
            good_tokens, imgs = self.speculative_decoding(
                (batch_size, channels, image_size[0], image_size[1]), 
                x_cond, task_embed, draft_model, target_timestep, 
                return_all_timesteps = return_all_timesteps
            )
            # 返回最终生成的图像
            if imgs:
                return imgs[-1]
            #elif good_tokens:
                #return good_tokens[-1]
        
        # 标准采样流程（保持向后兼容）
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        
        if sample_fn == self.ddim_sample:
            # ddim_sample的调用方式：ddim_sample(img, shape, x_cond, task_embed, target_timestep, return_all_timesteps)
            shape = (batch_size, channels, image_size[0], image_size[1])
            # 使用 .cuda() 来避免device类型问题
            img = torch.randn(shape).cuda() if torch.cuda.is_available() else torch.randn(shape)
            ret, _, _ = sample_fn(img, shape, x_cond, task_embed, target_timestep, return_all_timesteps)
        else:
            # p_sample_loop的调用方式：p_sample_loop(shape, x_cond, task_embed, return_all_timesteps)
            ret = sample_fn((batch_size, channels, image_size[0], image_size[1]), x_cond, task_embed, return_all_timesteps = return_all_timesteps)
        
        return ret

    @torch.no_grad()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.full((b,), t, device = device)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2

        x_start = None

        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, i, self_cond)

        return img

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def p_losses(self, x_start, t, x_cond, task_embed, noise=None):
        b, c, h, w = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        # noise sample

        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        # predict and take gradient step

        model_out = self.model(torch.cat([x, x_cond], dim=1), t, task_embed)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = self.loss_fn(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')

        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, img, img_cond, task_embed):
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size[0] and w == img_size[1], f'height and width of image must be {img_size}, got({h}, {w})'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        img = self.normalize(img)
        return self.p_losses(img, t, img_cond, task_embed)

# dataset classes

# class Dataset(Dataset):
#     def __init__(
#         self,
#         folder,
#         image_size,
#         exts = ['jpg', 'jpeg', 'png', 'tiff'],
#         augment_horizontal_flip = False,
#         convert_image_to = None
#     ):
#         super().__init__()
#         self.folder = folder
#         self.image_size = image_size
#         self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

#         maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) if exists(convert_image_to) else nn.Identity()

    #     self.transform = T.Compose([
    #         T.Lambda(maybe_convert_fn),
    #         T.Resize(image_size),
    #         T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
    #         T.CenterCrop(image_size),
    #         T.ToTensor()
    #     ])

    # def __len__(self):
    #     return len(self.paths)

    # def __getitem__(self, index):
    #     path = self.paths[index]
    #     img = Image.open(path)
    #     return self.transform(img)

# trainer class

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        tokenizer, 
        text_encoder, 
        train_set,
        valid_set,
        channels = 3,
        *,
        train_batch_size = 1,
        valid_batch_size = 1,
        gradient_accumulate_every = 1,
        augment_horizontal_flip = True,
        train_lr = 1e-4,
        train_num_steps = 100000,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 1000,
        num_samples = 3,
        results_folder = './results',
        amp = True,
        fp16 = True,
        split_batches = True,
        convert_image_to = None,
        cond_drop_chance=0.1,
    ):
        super().__init__()

        self.cond_drop_chance = cond_drop_chance

        self.tokenizer = tokenizer
        self.text_encoder = text_encoder

        # accelerator

        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = 'fp16' if fp16 else 'no'
        )

        self.accelerator.native_amp = amp

        # model

        self.model = diffusion_model

        self.channels = channels


        # sampling and training hyperparameters

        # assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size

        # dataset and dataloader

        
        valid_ind = [i for i in range(len(valid_set))][:num_samples]

        train_set = train_set
        valid_set = Subset(valid_set, valid_ind)

        self.ds = train_set
        self.valid_ds = valid_set
        dl = DataLoader(self.ds, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = 4)
        # dl = dataloader
        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)
        self.valid_dl = DataLoader(self.valid_ds, batch_size = valid_batch_size, shuffle = False, pin_memory = True, num_workers = 4)


        # optimizer

        self.opt = Adam(diffusion_model.parameters(), lr = train_lr, betas = adam_betas)

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)
            self.ema.to(self.device)

        self.text_encoder.to(self.device)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt = \
            self.accelerator.prepare(self.model, self.opt)

    @property
    def device(self):
        return self.accelerator.device
        # return "cuda:0"

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            'version': __version__
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    # @torch.no_grad()
    # def calculate_activation_statistics(self, samples):
    #     assert exists(self.inception_v3)

    #     features = self.inception_v3(samples)[0]
    #     features = rearrange(features, '... 1 1 -> ...')

    #     mu = torch.mean(features, dim = 0).cpu()
    #     sigma = torch.cov(features).cpu()
    #     return mu, sigma

    # def fid_score(self, real_samples, fake_samples):

    #     if self.channels == 1:
    #         real_samples, fake_samples = map(lambda t: repeat(t, 'b 1 ... -> b c ...', c = 3), (real_samples, fake_samples))

    #     min_batch = min(real_samples.shape[0], fake_samples.shape[0])
    #     real_samples, fake_samples = map(lambda t: t[:min_batch], (real_samples, fake_samples))

    #     m1, s1 = self.calculate_activation_statistics(real_samples)
    #     m2, s2 = self.calculate_activation_statistics(fake_samples)

    #     fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    #     return fid_value
    def encode_batch_text(self, batch_text):
        batch_text_ids = self.tokenizer(batch_text, return_tensors = 'pt', padding = True, truncation = True, max_length = 128).to(self.device)
        batch_text_embed = self.text_encoder(**batch_text_ids).last_hidden_state
        return batch_text_embed
    
    def sample(self, x_conds, tasks):
        assert x_conds.shape[0] == len(tasks)
        
        device = self.device
        bs = x_conds.shape[0]
        x_conds = x_conds.to(device)
        tasks = self.encode_batch_text(tasks).to(device)

        with self.accelerator.autocast():
            output = self.ema.ema_model.sample(batch_size=bs, x_cond=x_conds, task_embed=tasks)
        return output
        

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    x, x_cond, goal = next(self.dl)
                    x, x_cond = x.to(device), x_cond.to(device)

                    goal_embed = self.encode_batch_text(goal)
                    ### zero whole goal_embed if p < self.cond_drop_chance
                    goal_embed = goal_embed * (torch.rand(goal_embed.shape[0], 1, 1, device = goal_embed.device) > self.cond_drop_chance).float()


                    with self.accelerator.autocast():
                        loss = self.model(x, x_cond, goal_embed)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                        self.accelerator.backward(loss)

                accelerator.clip_grad_norm_(self.model.parameters(), 1.0)

                ### get maximum gradient from model
                # max_grad = max([torch.max(torch.abs(p.grad)) for p in self.model.parameters() if p.grad is not None])

                scale = self.accelerator.scaler.get_scale()
                
                pbar.set_description(f'loss: {total_loss:.4E}, loss scale: {scale:.1E}')

                accelerator.wait_for_everyone()

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update()

                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        self.ema.ema_model.eval()

                        with torch.no_grad():
                            milestone = self.step // self.save_and_sample_every
                            batches = num_to_groups(self.num_samples, self.valid_batch_size)
                            ### get val_imgs from self.valid_dl
                            x_conds = []
                            xs = []
                            task_embeds = []
                            for i, (x, x_cond, label) in enumerate(self.valid_dl):
                                xs.append(x)
                                x_conds.append(x_cond.to(device))
                                task_embeds.append(self.encode_batch_text(label))
                            
                            with self.accelerator.autocast():
                                all_xs_list = list(map(lambda n, c, e: self.ema.ema_model.sample(batch_size=n, x_cond=c, task_embed=e), batches, x_conds, task_embeds))
                        
                        print_gpu_utilization()
                        
                        gt_xs = torch.cat(xs, dim = 0) # [batch_size, 3*n, 120, 160]
                        # make it [batchsize*n, 3, 120, 160]
                        n_rows = gt_xs.shape[1] // 3
                        gt_xs = rearrange(gt_xs, 'b (n c) h w -> b n c h w', n=n_rows)
                        ### save images
                        x_conds = torch.cat(x_conds, dim = 0).detach().cpu()
                        # x_conds = rearrange(x_conds, 'b (n c) h w -> b n c h w', n=1)
                        all_xs = torch.cat(all_xs_list, dim = 0).detach().cpu()
                        all_xs = rearrange(all_xs, 'b (n c) h w -> b n c h w', n=n_rows)

                        # gt_xs = rearrange(torch.cat([x_conds, gt_xs], dim=1), 'b n c h w -> (b n) c h w', n=n_rows+1)
                        # pred_xs = rearrange(torch.cat([x_conds, all_xs], dim=1), 'b n c h w -> (b n) c h w', n=n_rows+1)
                        
                        ### bbox visualization
                        # label_start, label_end = get_bound_box_labels()
                        # x_starts = []
                        # for i, x_start in enumerate(x_conds):
                        #     x_starts.append(draw_bbox(x_start, label_start[i]))
                        # x_starts = torch.stack(x_starts).unsqueeze(1)
                        # x_conds = x_conds.unsqueeze(1)

                        # x_ends = []
                        # for i, x_end in enumerate(gt_xs[:, -1]):
                        #     x_ends.append(draw_bbox(x_end, label_end[i]))
                        # x_ends = torch.stack(x_ends).unsqueeze(1)



                        if self.step == self.save_and_sample_every:
                            os.makedirs(str(self.results_folder / f'imgs'), exist_ok = True)
                            gt_img = torch.cat([gt_xs[:, :]], dim=1)
                            gt_img = rearrange(gt_img, 'b n c h w -> (b n) c h w', n=n_rows)
                            utils.save_image(gt_img, str(self.results_folder / f'imgs/gt_img.png'), nrow=n_rows)

                        os.makedirs(str(self.results_folder / f'imgs/outputs'), exist_ok = True)
                        pred_img = torch.cat([all_xs[:, :]], dim=1)
                        pred_img = rearrange(pred_img, 'b n c h w -> (b n) c h w', n=n_rows)
                        utils.save_image(pred_img, str(self.results_folder / f'imgs/outputs/sample-{milestone}.png'), nrow=n_rows)

                        #     os.makedirs(str(self.results_folder / f'imgs'), exist_ok = True)
                        #     gt_flow_imgs = [torch.from_numpy(flow_tensor_to_image((gt_flow-0.5)*1000))/255 for gt_flow in gt_flows]
                        #     print(max(gt_flow_imgs[0].numpy().flatten()))
                        #     utils.save_image(gt_flow_imgs, str(self.results_folder / f'imgs/gt_flow.png'), nrow = int(math.sqrt(self.num_samples)))
                        #     utils.save_image(imgs, str(self.results_folder / f'imgs/gt_img.png'), nrow = int(math.sqrt(self.num_samples)))
                        # all_flow_imgs = [torch.from_numpy(flow_tensor_to_image((flow-0.5)*1000))/255 for flow in all_flows]
                        # os.makedirs(str(self.results_folder / f'imgs/outputs'), exist_ok = True)
                        # utils.save_image(all_flow_imgs, str(self.results_folder / f'imgs/outputs/sample-{milestone}.png'), nrow = int(math.sqrt(self.num_samples)))
                        self.save(milestone)

                        # whether to calculate fid

                        # data = 
                        # if exists(self.inception_v3):
                        #     fid_score = self.fid_score(real_samples = data, fake_samples = all_images)
                        #     accelerator.print(f'fid_score: {fid_score}')

                pbar.update(1)

        accelerator.print('training complete')
