import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import einsum
from typing import Union

import jax
import jax.numpy as jnp

class Scheduler(nn.Module):
    """
    We only care about masked diffusion models

    Train 
        1. t, samples -> sigma (alphas_comprod)  - (sample_transition) -> noisy_samples
        2. pred_score = model(samples, t)
        3. score = get_score(samples, noisy_samples)
        4. loss_weight = get_loss_weight(t)
        5. loss = loss_weight * comp_loss(pred_score, score)
        
    Sampling
    """
    def __init__(
        self, args
    ):  
        super().__init__()

        # basic configs
        self.mask_idx = -1
        self.num_vocabs = args.num_vocabs + 1 # "absorb"
        self.length = args.length
        self.eps = args.eps
        self.model_name = args.model_name
        
        # init noise schedule (similar to alphas_cumprod)
        if args.noise_schedule == "loglinear":
            self.sigma_bar = lambda t: -torch.log1p(-(1 - self.eps) * t) # page 15
            self.sigma = lambda t: (1 - self.eps) / (1 - (1 - self.eps) * t) # sigma_bar / dt
        if args.noise_schedule == "arccos":
            assert self.model_name == 'maskgit'
            self.sigma_bar = lambda t: -torch.log1p(-2*torch.arccos((1-(1-self.eps)*t))/torch.pi)
            self.sigma = lambda t: 1 / (1 - ((1-(1-self.eps)*t))**2).sqrt() / (torch.pi/2 - torch.arccos((1-(1-self.eps)*t)))
        if args.noise_schedule == "cosine":
            assert self.model_name == 'maskgit'
            self.sigma_bar = lambda t: -torch.log1p(-torch.cos(torch.pi*(1-(1-self.eps)*t)/2))
            self.sigma = lambda t: torch.sin(torch.pi*(1-(1-self.eps)*t)/2) * torch.pi/2 / (1 - torch.cos(torch.pi*(1-(1-self.eps)*t)/2))
        
    def add_noise(
        self, samples: torch.LongTensor, t: Union[int, torch.LongTensor], generator=None, 
    ):
        '''x0 -> xt'''
        # snr
        sigma_bar = self.sigma_bar(t)
        
        # perturb samples (absorb)
        perturb_prob = 1 - (-sigma_bar).exp()
        perturbed_samples = torch.where(
            torch.rand(*samples.shape, device=samples.device, generator=generator) < perturb_prob[:, None],
            self.mask_idx, samples
        )
        return perturbed_samples
    
    def output_to_score(self, output, t=None):
        if self.model_name == 'sedd':
            score = output.exp()
        elif self.model_name == 'd3pm':
            pass
        elif self.model_name == 'maskgit':
            sigma_bar = self.sigma_bar(t)
            perturb_prob = 1 - (-sigma_bar).exp()[:, None, None]
            score = (1 - perturb_prob) / perturb_prob * output # https://arxiv.org/abs/2407.21243, eq 3
        elif self.model_name == 'ctmc':
            pass
        else:
            raise ValueError(f'invalid model_name: {self.model_name}')
        return score
    
    def sample_latent(self, num_samples):
        return self.mask_idx * torch.ones([num_samples, self.length]).long()

    def step(self, output, xt, t, step_size):
        pass


class SchedulerOutput:
    def __init__(self, xt, rev_rate=None, tau=None):
        self.xt = xt
        self.tau = tau
        self.rev_rate = rev_rate


class EulerScheduler(Scheduler):
    def Q_tok(self, i):
        '''Q_tok = Q[i, :] (Eq.16 from SEDD paper)'''
        edge = -F.one_hot(i, num_classes=self.num_vocabs)
        edge[i == self.num_vocabs-1] += 1
        return edge
    
    def Q_tilde(self, xt, score):
        normalized_rate = self.Q_tok(xt) * score
        # ref: https://github.com/louaaron/Score-Entropy-Discrete-Diffusion/blob/main/graph_lib.py
        # to ensure that maintain the rate matrix property (sum_j R_ij = 0)
        normalized_rate.scatter_(-1, xt[..., None], torch.zeros_like(normalized_rate))
        normalized_rate.scatter_(-1, xt[..., None], -normalized_rate.sum(dim=-1, keepdim=True))
        return normalized_rate

    def step(self, output, xt, t, step_size, rev_rate=None, generator=None, if_last=False):
        x0_prob = output.exp()

        xt[xt == self.mask_idx] = self.num_vocabs-1

        
        
        q_xs = p_x0 * (move_chance_t - move_chance_s)
        q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
        _x = _sample_categorical(q_xs)
        
        copy_flag = (x != self.mask_index).to(x.dtype)
        return p_x0, copy_flag * x + (1 - copy_flag) * _x


        if rev_rate is None:
            sigma = self.sigma(t)
            score = self.output_to_score(output, t)
            rev_rate = sigma[..., None, None] * self.Q_tilde(xt, score)
        
        identity = F.one_hot(xt, num_classes=self.num_vocabs).to(rev_rate)
        xt_prob = identity + step_size * rev_rate
        xt_prob = xt_prob[..., :-1] if if_last else xt_prob
        xt = sample_categorical(xt_prob, generator=generator)
        xt[xt == self.num_vocabs-1] = self.mask_idx
        return SchedulerOutput(xt, rev_rate)
    
class GillespieScheduler(EulerScheduler):
    def add_noise(
        self, samples: torch.FloatTensor, k: Union[int, torch.LongTensor], generator=None, 
    ):
        # 1. token prob
        token_prob = torch.rand(*samples.shape, device=samples.device, generator=generator)
        if samples.shape[1] - k != 0:
            values,idx = torch.topk(token_prob, samples.shape[1] - k, dim=-1, largest=False)
            t = (values.max(dim=-1).values / (1-self.eps)).clamp(min=0, max=1)
            perturbed_samples = torch.scatter(samples, -1, idx, self.mask_idx)
        else:
            t = torch.zeros(samples.shape[0], device=samples.device) + 1e-5
            perturbed_samples = samples
        return perturbed_samples, t
    
    def step(self, output, xt, t, dk, rev_rate=None, generator=None, if_last=False):
        '''Algorithm 1 from https://arxiv.org/abs/2407.21243'''
        xt[xt == self.mask_idx] = self.num_vocabs-1
        if rev_rate is None:
            sigma = self.sigma(t)
            score = self.output_to_score(output, t)
            rev_rate = sigma[..., None, None] * self.Q_tilde(xt, score)

        # sample holding time
        r = rev_rate[..., :-1]
        tau = sample_exponential(r.sum(dim=-1), generator=generator)

        # sample token 
        tau, idx = torch.topk(tau, dk, dim=-1, largest=False)
        r = torch.gather(r, 1, idx[..., None].repeat(1,1,r.size(-1)))
        r = r / r.sum(dim=-1, keepdim=True)
        xt = torch.scatter(xt, -1, idx, sample_categorical(r, generator=generator))
        xt[xt == self.num_vocabs-1] = self.mask_idx
        return SchedulerOutput(xt, rev_rate=rev_rate, tau=tau.max(dim=-1).values)
    
def sample_exponential(lambda_, eps=1e-6, generator=None):
    if generator is None:
        exp_noise = torch.rand_like(lambda_)
    else:
        exp_noise = torch.rand(lambda_.shape, generator=generator, device=generator.device).to(lambda_)
    return -1 / (lambda_ + eps) * torch.log(eps + (1 - eps) * exp_noise)

def sample_categorical(categorical_probs, eps=1e-6, generator=None):
    '''use gumbel-max trick, but given probability'''
    if generator is None:
        gumbel_noise = torch.rand_like(categorical_probs)
    else:
        gumbel_noise = torch.rand(categorical_probs.shape, generator=generator, device=generator.device).to(categorical_probs)
    gumbel_noise = (eps - torch.log(eps + (1 - eps) * gumbel_noise))
    return torch.argmax(categorical_probs / gumbel_noise, dim=-1)
