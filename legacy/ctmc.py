# ref: https://github.com/andrew-cr/tauLDR/blob/main/lib/models/models.py
from tkinter import Image
from typing import Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.autograd.profiler as profiler
import math
from torch.nn.parallel import DistributedDataParallel as DDP


class Scheduler(nn.Module):
    """
    We only care about uniform / Gaussian transition kernel 
    """
    def __init__(
        self, args, model,
    ):  
        super().__init__()

        # basic configs
        self.initial_dist = args.initial_dist
        self.num_vocabs = args.num_vocabs
        self.length = args.length
        self.eps = args.eps
        self.model_name = args.model_name
        
        # init noise schedule (similar to alphas_cumprod)
        if args.dataset_name == "cifar":
            self.initial_dist_std = args.initial_dist_std
        if args.dataset_name == "piano":
            pass

        self.sigma_bar = lambda x: x
        self.qt0 = model.transition
        self.rate = model.rate
        
    def output_to_rev_rate(self, output, t=None, xt=None, is_corrector=False):
        B, L, S = output.shape # |batch_size, length, num_vocabs|
        device = output.device
        
        if isinstance(t, torch.Tensor):
            assert t.shape[0] == B
        if isinstance(t, float):
            t = t * torch.ones(B).to(output)

        if self.model_name == 'ctmc':
            # compute 
            qt0 = self.qt0(t) 
            qt0_denom = qt0[
                torch.arange(B, device=device).repeat_interleave(L*S),
                torch.arange(S, device=device).repeat(B*L),
                xt.long().flatten().repeat_interleave(S)
            ].view(B, L, S) + 1e-6
            
            # predictor
            rate = self.rate(t) 
            forward_rates = rate[
                torch.arange(B, device=device).repeat_interleave(L*S),
                torch.arange(S, device=device).repeat(B*L),
                xt.long().flatten().repeat_interleave(S)
            ].view(B, L, S)
            score = (output / qt0_denom) @ qt0
            rev_rate = forward_rates * score

            rev_rate[
                torch.arange(B, device=device).repeat_interleave(L),
                torch.arange(L, device=device).repeat(B),
                xt.long().flatten()
            ] = 0.0 # for Poisson sampling

            # corrector
            # if is_corrector:
            #     rate_T = rate[
            #         torch.arange(B, device=device).repeat_interleave(L*S),
            #         xt.long().flatten().repeat_interleave(S),
            #         torch.arange(S, device=device).repeat(B*L)
            #     ].view(B, L, S)
            #     rev_rate = rate_T + rev_rate
            #     rev_rate[
            #         torch.arange(B, device=device).repeat_interleave(L),
            #         torch.arange(L, device=device).repeat(B),
            #         xt.long().flatten()
            #     ] = 0.0 # for Poisson sampling

            # normalize rate matrix (my modification)
            # rev_rate[
            #     torch.arange(B, device=device).repeat_interleave(L),
            #     torch.arange(L, device=device).repeat(B),
            #     xt.long().flatten()
            # ] = 0.0
            # rev_rate[
            #     torch.arange(B, device=device).repeat_interleave(L),
            #     torch.arange(L, device=device).repeat(B),
            #     xt.long().flatten()
            # ] = -rev_rate[
            #     torch.arange(B, device=device).repeat_interleave(L), 
            #     torch.arange(L, device=device).repeat(B),
            #     ...
            # ].sum(dim=-1)
        else:
            raise ValueError(f'invalid model_name: {self.model_name}')
        return rev_rate
    
    def sample_latent(self, num_samples):
        if self.initial_dist == 'uniform':
            return torch.randint(low=0, high=self.num_vocabs, size=(num_samples, self.length))
        elif self.initial_dist == 'gaussian':
            target = np.exp(
                - ((np.arange(1, self.num_vocabs+1) - self.num_vocabs//2)**2) / (2 * self.initial_dist_std**2)
            )
            target = target / np.sum(target)
            cat = torch.distributions.categorical.Categorical(torch.from_numpy(target))
            return cat.sample((num_samples*self.length,)).view(num_samples,self.length)

    def add_noise(
        self, samples: torch.LongTensor, t, generator=None, 
    ):
        '''x0 -> xt'''
        B, _ = samples.shape
        
        # forward matrix 
        if isinstance(t, torch.Tensor):
            assert t.shape[0] == B
        if isinstance(t, float):
            t = t * torch.ones(B).to(samples)
        qt0 = self.qt0(t)
        perturbed_samples = F.one_hot(samples, num_classes=self.num_vocabs).float() @ qt0
        perturbed_samples = sample_categorical(perturbed_samples, generator=generator)
        return perturbed_samples
    
    def step(self, output, xt, t, step_size):
        pass


class EulerScheduler(Scheduler):
    def step(self, output, xt, t, step_size, rev_rate=None, generator=None, if_last=False):
        if rev_rate is None:
            rev_rate = self.output_to_rev_rate(output, t=t, xt=xt)
        
        if if_last:
            xt = torch.max(output, dim=2)[1] # argmax
            rev_rate = None
        else:
            diffs = torch.arange(self.num_vocabs, device=xt.device).view(1,1,self.num_vocabs) - xt[..., None]
            jump_nums = torch.distributions.poisson.Poisson(rev_rate * step_size ).sample()
            adj_diffs = jump_nums * diffs
            overall_jump = torch.sum(adj_diffs, dim=2)
            xt = xt + overall_jump
            xt = torch.clamp(xt, min=0, max=self.num_vocabs-1)
        return SchedulerOutput(xt, xt_prob=None, rev_rate=rev_rate)


class PCScheduler(Scheduler):
    '''
    predictor-corrector sampler
    ref: https://github.com/andrew-cr/tauLDR/blob/main/lib/sampling/sampling.py
    '''
    def step(self, output, xt, t, step_size, rev_rate=None, generator=None, if_last=False, is_corrector=False):
        if rev_rate is None:
            rev_rate = self.output_to_rev_rate(output, t=t, xt=xt, is_corrector=is_corrector)
        
        if if_last:
            xt = torch.max(output, dim=2)[1] # argmax
            return SchedulerOutput(xt, xt_prob=None, rev_rate=None)
        
        else:
            diffs = torch.arange(self.num_vocabs, device=xt.device).view(1,1,self.num_vocabs) - xt[..., None]
            jump_nums = torch.distributions.poisson.Poisson(rev_rate * step_size).sample()
            adj_diffs = jump_nums * diffs
            overall_jump = torch.sum(adj_diffs, dim=2)
            xt = xt + overall_jump
            xt = torch.clamp(xt, min=0, max=self.num_vocabs-1)

            return SchedulerOutput(xt, xt_prob=None, rev_rate=rev_rate)


class SchedulerOutput:
    def __init__(self, xt, xt_prob=None, rev_rate=None, tau=None):
        self.xt = xt
        self.tau = tau
        self.xt_prob = xt_prob
        self.rev_rate = rev_rate

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

#############
# CTMC repo #
#############
class ImageX0PredBase(nn.Module):
    def __init__(self, cfg, device, rank=None):
        super().__init__()

        self.fix_logistic = cfg.model.fix_logistic
        ch = cfg.model.ch
        num_res_blocks = cfg.model.num_res_blocks
        num_scales = cfg.model.num_scales
        ch_mult = cfg.model.ch_mult
        input_channels = cfg.model.input_channels
        output_channels = cfg.model.input_channels * cfg.data.S
        scale_count_to_put_attn = cfg.model.scale_count_to_put_attn
        data_min_max = cfg.model.data_min_max
        dropout = cfg.model.dropout
        skip_rescale = cfg.model.skip_rescale
        do_time_embed = True
        time_scale_factor = cfg.model.time_scale_factor
        time_embed_dim = cfg.model.time_embed_dim

        tmp_net = UNet(
                ch, num_res_blocks, num_scales, ch_mult, input_channels,
                output_channels, scale_count_to_put_attn, data_min_max,
                dropout, skip_rescale, do_time_embed, time_scale_factor,
                time_embed_dim
        ).to(device)
        if cfg.distributed:
            self.net = DDP(tmp_net, device_ids=[rank])
        else:
            self.net = tmp_net

        self.S = cfg.data.S
        self.data_shape = cfg.data.shape

    def forward(self,
        x, # ["B", "L"]
        t, # ["B"]
        *args,
        **kwargs,
    ):
        """
            Returns logits over state space for each pixel 
        """
        B, L = x.shape
        C,H,W = self.data_shape
        S = self.S
        x = x.view(B, C, H, W)


        net_out = self.net(x, t) # (B, 2*C, H, W)

        # Truncated logistic output from https://arxiv.org/pdf/2107.03006.pdf

        mu = net_out[:, 0:C, :, :].unsqueeze(-1)
        log_scale = net_out[:, C:, :, :].unsqueeze(-1)

        inv_scale = torch.exp(- (log_scale - 2))

        bin_width = 2. / self.S
        bin_centers = torch.linspace(start=-1. + bin_width/2,
            end=1. - bin_width/2,
            steps=self.S,
            device=x.device).view(1, 1, 1, 1, self.S)

        sig_in_left = (bin_centers - bin_width/2 - mu) * inv_scale
        bin_left_logcdf = F.logsigmoid(sig_in_left)
        sig_in_right = (bin_centers + bin_width/2 - mu) * inv_scale
        bin_right_logcdf = F.logsigmoid(sig_in_right)

        logits_1 = self._log_minus_exp(bin_right_logcdf, bin_left_logcdf)
        logits_2 = self._log_minus_exp(-sig_in_left + bin_left_logcdf, -sig_in_right + bin_right_logcdf)
        if self.fix_logistic:
            logits = torch.min(logits_1, logits_2)
        else:
            logits = logits_1

        logits = logits.view(B,L,S)

        return logits.softmax(dim=-1)

    def _log_minus_exp(self, a, b, eps=1e-6):
        """ 
            Compute log (exp(a) - exp(b)) for (b<a)
            From https://arxiv.org/pdf/2107.03006.pdf
        """
        return a + torch.log1p(-torch.exp(b-a) + eps)

class BirthDeathForwardBase():
    def __init__(self, cfg, device):
        self.S = S = cfg.data.S
        self.sigma_min, self.sigma_max = cfg.model.sigma_min, cfg.model.sigma_max
        self.device = device

        base_rate = np.diag(np.ones((S-1,)), 1)
        base_rate += np.diag(np.ones((S-1,)), -1)
        base_rate -= np.diag(np.sum(base_rate, axis=1))
        eigvals, eigvecs = np.linalg.eigh(base_rate)

        self.base_rate = torch.from_numpy(base_rate).float().to(self.device)
        self.base_eigvals = torch.from_numpy(eigvals).float().to(self.device)
        self.base_eigvecs = torch.from_numpy(eigvecs).float().to(self.device)

    def _rate_scalar(self, t, #["B"]
    ):
        return self.sigma_min**2 * (self.sigma_max / self.sigma_min) ** (2 * t) *\
            math.log(self.sigma_max / self.sigma_min)

    def _integral_rate_scalar(self, t, # ["B"]
    ):
        return 0.5 * self.sigma_min**2 * (self.sigma_max / self.sigma_min) ** (2 * t) -\
            0.5 * self.sigma_min**2

    def rate(self, t, #["B"]
    ):
        B = t.shape[0]
        S = self.S
        rate_scalars = self._rate_scalar(t)

        return self.base_rate.view(1, S, S) * rate_scalars.view(B, 1, 1)

    def transition(self, t, # ["B"]
    ):
        B = t.shape[0]
        S = self.S

        integral_rate_scalars = self._integral_rate_scalar(t)

        adj_eigvals = integral_rate_scalars.view(B, 1) * self.base_eigvals.view(1, S)

        transitions = self.base_eigvecs.view(1, S, S) @ \
            torch.diag_embed(torch.exp(adj_eigvals)) @ \
            self.base_eigvecs.T.view(1, S, S)

        # Some entries that are supposed to be very close to zero might be negative
        if torch.min(transitions) < -1e-6:
            print(f"[Warning] BirthDeathForwardBase, large negative transition values {torch.min(transitions)}")

        # Clamping at 1e-8 because at float level accuracy anything lower than that
        # is probably inaccurate and should be zero anyway
        transitions[transitions < 1e-8] = 0.0

        return transitions

class UniformRate():
    def __init__(self, cfg, device):
        self.S = S = cfg.data.S
        self.rate_const = cfg.model.rate_const
        self.device = device

        rate = self.rate_const * np.ones((S,S))
        rate = rate - np.diag(np.diag(rate))
        rate = rate - np.diag(np.sum(rate, axis=1))
        eigvals, eigvecs = np.linalg.eigh(rate)

        self.rate_matrix = torch.from_numpy(rate).float().to(self.device)
        self.eigvals = torch.from_numpy(eigvals).float().to(self.device)
        self.eigvecs = torch.from_numpy(eigvecs).float().to(self.device)

    def rate(self, t, #["B"]
    ):
        B = t.shape[0]
        S = self.S

        return torch.tile(self.rate_matrix.view(1,S,S), (B, 1, 1))

    def transition(self, t, # ["B"]
    ):
        B = t.shape[0]
        S = self.S
        transitions = self.eigvecs.view(1, S, S) @ \
            torch.diag_embed(torch.exp(self.eigvals.view(1, S) * t.view(B,1))) @ \
            self.eigvecs.T.view(1, S, S)

        if torch.min(transitions) < -1e-6:
            print(f"[Warning] UniformRate, large negative transition values {torch.min(transitions)}")

        transitions[transitions < 1e-8] = 0.0

        return transitions

class GaussianTargetRate():
    def __init__(self, cfg, device):
        self.S = S = cfg.data.S
        self.rate_sigma = cfg.model.rate_sigma
        self.Q_sigma = cfg.model.Q_sigma
        self.time_exponential = cfg.model.time_exponential
        self.time_base = cfg.model.time_base
        self.device = device

        rate = np.zeros((S,S))

        vals = np.exp(-np.arange(0, S)**2/(self.rate_sigma**2))
        for i in range(S):
            for j in range(S):
                if i < S//2:
                    if j > i and j < S-i:
                        rate[i, j] = vals[j-i-1]
                elif i > S//2:
                    if j < i and j > -i+S-1:
                        rate[i, j] = vals[i-j-1]
        for i in range(S):
            for j in range(S):
                if rate[j, i] > 0.0:
                    rate[i, j] = rate[j, i] * np.exp(- ( (j+1)**2 - (i+1)**2 + S*(i+1) - S*(j+1) ) / (2 * self.Q_sigma**2)  )

        rate = rate - np.diag(np.diag(rate))
        rate = rate - np.diag(np.sum(rate, axis=1))

        eigvals, eigvecs = np.linalg.eig(rate)
        inv_eigvecs = np.linalg.inv(eigvecs)

        # self.register_buffer('base_rate', torch.from_numpy(rate).float().to(self.device))
        # self.register_buffer('eigvals', torch.from_numpy(eigvals).float().to(self.device))
        # self.register_buffer('eigvecs', torch.from_numpy(eigvecs).float().to(self.device))
        # self.register_buffer('inv_eigvecs', torch.from_numpy(inv_eigvecs).float().to(self.device))

        self.base_rate = torch.from_numpy(rate).float().to(self.device)
        self.eigvals = torch.from_numpy(eigvals).float().to(self.device)
        self.eigvecs = torch.from_numpy(eigvecs).float().to(self.device)
        self.inv_eigvecs = torch.from_numpy(inv_eigvecs).float().to(self.device)

    def _integral_rate_scalar(self, t, # ["B"]
    ):
        return self.time_base * (self.time_exponential ** t) - \
            self.time_base
    
    def _rate_scalar(self, t, # ["B"]
    ):
        return self.time_base * math.log(self.time_exponential) * \
            (self.time_exponential ** t)

    def rate(self, t, # ["B"]
    ):
        B = t.shape[0]
        S = self.S
        rate_scalars = self._rate_scalar(t)

        return self.base_rate.view(1, S, S) * rate_scalars.view(B, 1, 1)

    def transition(self, t, # ["B"]
    ):
        B = t.shape[0]
        S = self.S

        integral_rate_scalars = self._integral_rate_scalar(t)

        adj_eigvals = integral_rate_scalars.view(B, 1) * self.eigvals.view(1, S)

        transitions = self.eigvecs.view(1, S, S) @ \
            torch.diag_embed(torch.exp(adj_eigvals)) @ \
            self.inv_eigvecs.view(1, S, S)

        # Some entries that are supposed to be very close to zero might be negative
        if torch.min(integral_rate_scalars) < -1e-6:
            print(f"[Warning] GaussianTargetRate, large negative integral_rate_scalars values {torch.min(integral_rate_scalars)}, t: {t}")
        if torch.min(transitions) < -1e-6:
            print(f"[Warning] GaussianTargetRate, large negative transition values {torch.min(transitions)}, t: {t}")

        # Clamping at 1e-8 because at float level accuracy anything lower than that
        # is probably inaccurate and should be zero anyway
        transitions[transitions < 1e-8] = 0.0

        return transitions

class SequenceTransformer(nn.Module):
    def __init__(self, cfg, device, rank=None):
        super().__init__()

        num_layers = cfg.model.num_layers
        d_model = cfg.model.d_model
        num_heads = cfg.model.num_heads
        dim_feedforward = cfg.model.dim_feedforward
        dropout = cfg.model.dropout
        num_output_FFresiduals = cfg.model.num_output_FFresiduals
        time_scale_factor = cfg.model.time_scale_factor
        temb_dim = cfg.model.temb_dim
        use_one_hot_input = cfg.model.use_one_hot_input
        self.S = cfg.data.S

        assert len(cfg.data.shape) == 1
        max_len = cfg.data.shape[0]

        tmp_net = TransformerEncoder(
            num_layers, d_model, num_heads, dim_feedforward, dropout,
            num_output_FFresiduals, time_scale_factor, self.S, max_len,
            temb_dim, use_one_hot_input, device
        ).to(device)
        if cfg.distributed:
            self.net = DDP(tmp_net, device_ids=[rank])
        else:
            self.net = tmp_net

        self.data_shape = cfg.data.shape

    def forward(self,
        x, # ["B", "L"]
        t, # ["B"]
        **kwargs,
    ):
        """
            Returns logits over state space
        """
        B, L = x.shape
        S = self.S

        logits = self.net(x.long(), t.long()) # (B, L, S)

        return logits.softmax(dim=-1)

class ResidualMLP(nn.Module):
    def __init__(self, cfg, device, rank=None):
        super().__init__()

        self.S = cfg.data.S
        num_layers = cfg.model.num_layers
        d_model = cfg.model.d_model
        hidden_dim = cfg.model.hidden_dim
        time_scale_factor = cfg.model.time_scale_factor
        temb_dim = cfg.model.temb_dim

        assert len(cfg.data.shape) == 1
        L = cfg.data.shape[0]

        tmp_net = ResidualMLP(
            num_layers, d_model, hidden_dim, L, self.S,
            time_scale_factor, temb_dim
        ).to(device)

        if cfg.distributed:
            self.net = DDP(tmp_net, device_ids=[rank])
        else:
            self.net = tmp_net

        self.data_shape = cfg.data.shape

    def forward(self,
        x, # ["B", "L"]
        t, # ["B"]
        **kwargs,
    ):
        """
            Returns logits over state space
        """

        logits = self.net(x, t) # (B, L, S)

        return logits.softmax(dim=-1)

# Based on https://github.com/yang-song/score_sde_pytorch/blob/ef5cb679a4897a40d20e94d8d0e2124c3a48fb8c/models/ema.py
class EMA():
    def __init__(self, cfg):
        self.decay = cfg.model.ema_decay
        if self.decay < 0.0 or self.decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')
        self.shadow_params = []
        self.collected_params = []
        self.num_updates = 0

    def init_ema(self):
        self.shadow_params = [p.clone().detach()
                            for p in self.parameters() if p.requires_grad]

    def update_ema(self):

        if len(self.shadow_params) == 0:
            raise ValueError("Shadow params not initialized before first ema update!")

        decay = self.decay
        self.num_updates += 1
        decay = min(decay, (1 + self.num_updates) / (10 + self.num_updates))
        one_minus_decay = 1.0 - decay
        with torch.no_grad():
            parameters = [p for p in self.parameters() if p.requires_grad]
            for s_param, param in zip(self.shadow_params, parameters):
                s_param.sub_(one_minus_decay * (s_param - param))

    def state_dict(self):
        sd = nn.Module.state_dict(self)
        sd['ema_decay'] = self.decay
        sd['ema_num_updates'] = self.num_updates
        sd['ema_shadow_params'] = self.shadow_params

        return sd

    def move_shadow_params_to_model_params(self):
        parameters = [p for p in self.parameters() if p.requires_grad]
        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                param.data.copy_(s_param.data)

    def move_model_params_to_collected_params(self):
        self.collected_params = [param.clone() for param in self.parameters()]

    def move_collected_params_to_model_params(self):
        for c_param, param in zip(self.collected_params, self.parameters()):
            param.data.copy_(c_param.data)

    def load_state_dict(self, state_dict):
        missing_keys, unexpected_keys = nn.Module.load_state_dict(self, state_dict, strict=False)

        # print("state dict keys")
        # for key in state_dict.keys():
        #     print(key)

        if len(missing_keys) > 0:
            print("Missing keys: ", missing_keys)

        if not (len(unexpected_keys) == 3 and \
            'ema_decay' in unexpected_keys and \
            'ema_num_updates' in unexpected_keys and \
            'ema_shadow_params' in unexpected_keys):
            print("Unexpected keys: ", unexpected_keys)
            raise ValueError

        self.decay = state_dict['ema_decay']
        self.num_updates = state_dict['ema_num_updates']
        self.shadow_params = state_dict['ema_shadow_params']

    def train(self, mode=True):
        if self.training == mode:
            print("Dont call model.train() with the same mode twice! Otherwise EMA parameters may overwrite original parameters")
            print("Current model training mode: ", self.training)
            print("Requested training mode: ", mode)
            raise ValueError

        nn.Module.train(self, mode)
        if mode:
            if len(self.collected_params) > 0:
                self.move_collected_params_to_model_params()
            else:
                print("model.train(True) called but no ema collected parameters!")
        else:
            self.move_model_params_to_collected_params()
            self.move_shadow_params_to_model_params()

# make sure EMA inherited first so it can override the state dict functions
class GaussianTargetRateImageX0PredEMA(EMA, ImageX0PredBase, GaussianTargetRate):
    def __init__(self, cfg, device, rank=None):
        EMA.__init__(self, cfg)
        ImageX0PredBase.__init__(self, cfg, device, rank)
        GaussianTargetRate.__init__(self, cfg, device)

        self.init_ema()

class UniformRateSequenceTransformerEMA(EMA, SequenceTransformer, UniformRate):
    def __init__(self, cfg, device, rank=None):
        EMA.__init__(self, cfg)
        SequenceTransformer.__init__(self, cfg, device, rank)
        UniformRate.__init__(self, cfg, device)

        self.init_ema()

############
# networks #
############
# https://github.com/andrew-cr/tauLDR/blob/main/lib/networks/py
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np


# Code modified from https://github.com/yang-song/score_sde_pytorch
def variance_scaling(scale, mode, distribution,
                     in_axis=1, out_axis=0,
                     dtype=torch.float32,
                     device='cpu'):
  """Ported from JAX. """

  def _compute_fans(shape, in_axis=1, out_axis=0):
    receptive_field_size = np.prod(shape) / shape[in_axis] / shape[out_axis]
    fan_in = shape[in_axis] * receptive_field_size
    fan_out = shape[out_axis] * receptive_field_size
    return fan_in, fan_out

  def init(shape, dtype=dtype, device=device):
    fan_in, fan_out = _compute_fans(shape, in_axis, out_axis)
    if mode == "fan_in":
      denominator = fan_in
    elif mode == "fan_out":
      denominator = fan_out
    elif mode == "fan_avg":
      denominator = (fan_in + fan_out) / 2
    else:
      raise ValueError(
        "invalid mode for variance scaling initializer: {}".format(mode))
    variance = scale / denominator
    if distribution == "normal":
      return torch.randn(*shape, dtype=dtype, device=device) * np.sqrt(variance)
    elif distribution == "uniform":
      return (torch.rand(*shape, dtype=dtype, device=device) * 2. - 1.) * np.sqrt(3 * variance)
    else:
      raise ValueError("invalid distribution for variance scaling initializer")

  return init


def default_init(scale=1.):
  """The same initialization used in DDPM."""
  scale = 1e-10 if scale == 0 else scale
  return variance_scaling(scale, 'fan_avg', 'uniform')

class NiN(nn.Module):
  def __init__(self, in_ch, out_ch, init_scale=0.1):
    super().__init__()
    self.W = nn.Parameter(default_init(scale=init_scale)((in_ch, out_ch)), requires_grad=True)
    self.b = nn.Parameter(torch.zeros(out_ch), requires_grad=True)

  def forward(self, x, #  ["batch", "in_ch", "H", "W"]
    ):

    x = x.permute(0, 2, 3, 1)
    # x (batch, H, W, in_ch)
    y = torch.einsum('bhwi,ik->bhwk', x, self.W) + self.b
    # y (batch, H, W, out_ch)
    return y.permute(0, 3, 1, 2)

class AttnBlock(nn.Module):
  """Channel-wise self-attention block."""
  def __init__(self, channels, skip_rescale=True):
    super().__init__()
    self.skip_rescale = skip_rescale
    self.GroupNorm_0 = nn.GroupNorm(num_groups=min(channels//4, 32),
        num_channels=channels, eps=1e-6)
    self.NIN_0 = NiN(channels, channels)
    self.NIN_1 = NiN(channels, channels)
    self.NIN_2 = NiN(channels, channels)
    self.NIN_3 = NiN(channels, channels, init_scale=0.)

  def forward(self, x, # ["batch", "channels", "H", "W"]
    ):

    B, C, H, W = x.shape
    h = self.GroupNorm_0(x)
    q = self.NIN_0(h)
    k = self.NIN_1(h)
    v = self.NIN_2(h)

    w = torch.einsum('bchw,bcij->bhwij', q, k) * (int(C) ** (-0.5))
    w = torch.reshape(w, (B, H, W, H * W))
    w = F.softmax(w, dim=-1)
    w = torch.reshape(w, (B, H, W, H, W))
    h = torch.einsum('bhwij,bcij->bchw', w, v)
    h = self.NIN_3(h)

    if self.skip_rescale:
        return (x + h) / np.sqrt(2.)
    else:
        return x + h


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, temb_dim=None, dropout=0.1, skip_rescale=True):
        super().__init__()

        self.in_ch = in_ch
        self.out_ch = out_ch

        self.skip_rescale = skip_rescale

        self.act = nn.functional.silu
        self.groupnorm0 = nn.GroupNorm(
            num_groups=min(in_ch // 4, 32),
            num_channels=in_ch, eps=1e-6
        )
        self.conv0 = nn.Conv2d(
            in_ch, out_ch, kernel_size=3, padding=1
        )

        if temb_dim is not None:
            self.dense0 = nn.Linear(temb_dim, out_ch)
            nn.init.zeros_(self.dense0.bias)


        self.groupnorm1 = nn.GroupNorm(
            num_groups=min(out_ch // 4, 32),
            num_channels=out_ch, eps=1e-6
        )
        self.dropout0 = nn.Dropout(dropout)

        self.conv1 = nn.Conv2d(
            out_ch, out_ch, kernel_size=3, padding=1
        )
        if out_ch != in_ch:
            self.nin = NiN(in_ch, out_ch)

    def forward(self, x, # ["batch", "in_ch", "H", "W"]
                temb=None, #  ["batch", "temb_dim"]
        ):

        assert x.shape[1] == self.in_ch

        h = self.groupnorm0(x)
        h = self.act(h)
        h = self.conv0(h)

        if temb is not None:
            h += self.dense0(self.act(temb))[:, :, None, None]

        h = self.groupnorm1(h)
        h = self.act(h)
        h = self.dropout0(h)
        h = self.conv1(h)
        if h.shape[1] != self.in_ch:
            x = self.nin(x)

        assert x.shape == h.shape

        if self.skip_rescale:
            return (x + h) / np.sqrt(2.)
        else:
            return x + h

class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, 
            stride=2, padding=0)

    def forward(self, x, # ["batch", "ch", "inH", "inW"]
        ):
        B, C, H, W = x.shape
        x = nn.functional.pad(x, (0, 1, 0, 1))
        x= self.conv(x)

        assert x.shape == (B, C, H // 2, W // 2)
        return x

class Upsample(nn.Module):
  def __init__(self, channels):
    super().__init__()
    self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

  def forward(self, x, # ["batch", "ch", "inH", "inW"]
    ):
    B, C, H, W = x.shape
    h = F.interpolate(x, (H*2, W*2), mode='nearest')
    h = self.conv(h)

    assert h.shape == (B, C, H*2, W*2)
    return h

class UNet(nn.Module):
    def __init__(self, ch, num_res_blocks, num_scales, ch_mult, input_channels,
        output_channels, scale_count_to_put_attn, data_min_max, dropout,
        skip_rescale, do_time_embed, time_scale_factor=None, time_embed_dim=None):
        super().__init__()
        assert num_scales == len(ch_mult)

        self.ch = ch
        self.num_res_blocks = num_res_blocks
        self.num_scales = num_scales
        self.ch_mult = ch_mult
        self.input_channels = input_channels
        self.output_channels = 2 * input_channels
        self.scale_count_to_put_attn = scale_count_to_put_attn
        self.data_min_max = data_min_max # tuple of min and max value of input so it can be rescaled to [-1, 1]
        self.dropout = dropout
        self.skip_rescale = skip_rescale
        self.do_time_embed = do_time_embed # Whether to add in time embeddings
        self.time_scale_factor = time_scale_factor # scale to make the range of t be 0 to 1000
        self.time_embed_dim = time_embed_dim

        self.act = nn.functional.silu

        if self.do_time_embed:
            self.temb_modules = []
            self.temb_modules.append(nn.Linear(self.time_embed_dim, self.time_embed_dim*4))
            nn.init.zeros_(self.temb_modules[-1].bias)
            self.temb_modules.append(nn.Linear(self.time_embed_dim*4, self.time_embed_dim*4))
            nn.init.zeros_(self.temb_modules[-1].bias)
            self.temb_modules = nn.ModuleList(self.temb_modules)

        self.expanded_time_dim = 4 * self.time_embed_dim if self.do_time_embed else None

        self.input_conv = nn.Conv2d(
            in_channels=input_channels, out_channels=self.ch,
            kernel_size=3, padding=1
        )

        h_cs = [self.ch]
        in_ch = self.ch


        # Downsampling
        self.downsampling_modules = []

        for scale_count in range(self.num_scales):
            for res_count in range(self.num_res_blocks):
                out_ch = self.ch * self.ch_mult[scale_count]
                self.downsampling_modules.append(
                    ResBlock(in_ch, out_ch, temb_dim=self.expanded_time_dim,
                        dropout=dropout, skip_rescale=self.skip_rescale)
                )
                in_ch = out_ch
                h_cs.append(in_ch)
                if scale_count == self.scale_count_to_put_attn:
                    self.downsampling_modules.append(
                        AttnBlock(in_ch, skip_rescale=self.skip_rescale)
                    )

            if scale_count != self.num_scales - 1:
                self.downsampling_modules.append(Downsample(in_ch))
                h_cs.append(in_ch)

        self.downsampling_modules = nn.ModuleList(self.downsampling_modules)

        # Middle
        self.middle_modules = []

        self.middle_modules.append(
            ResBlock(in_ch, in_ch, temb_dim=self.expanded_time_dim,
                dropout=dropout, skip_rescale=self.skip_rescale)
        )
        self.middle_modules.append(
            AttnBlock(in_ch, skip_rescale=self.skip_rescale)
        )
        self.middle_modules.append(
            ResBlock(in_ch, in_ch, temb_dim=self.expanded_time_dim,
                dropout=dropout, skip_rescale=self.skip_rescale)
        )
        self.middle_modules = nn.ModuleList(self.middle_modules)

        # Upsampling
        self.upsampling_modules = []

        for scale_count in reversed(range(self.num_scales)):
            for res_count in range(self.num_res_blocks+1):
                out_ch = self.ch * self.ch_mult[scale_count]
                self.upsampling_modules.append(
                    ResBlock(in_ch + h_cs.pop(), 
                        out_ch,
                        temb_dim=self.expanded_time_dim,
                        dropout=dropout,
                        skip_rescale=self.skip_rescale
                    )
                )
                in_ch = out_ch

                if scale_count == self.scale_count_to_put_attn:
                    self.upsampling_modules.append(
                        AttnBlock(in_ch, skip_rescale=self.skip_rescale)
                    )
            if scale_count != 0:
                self.upsampling_modules.append(Upsample(in_ch))

        self.upsampling_modules = nn.ModuleList(self.upsampling_modules)

        assert len(h_cs) == 0

        # output
        self.output_modules = []
        
        self.output_modules.append(
            nn.GroupNorm(min(in_ch//4, 32), in_ch, eps=1e-6)
        )

        self.output_modules.append(
            nn.Conv2d(in_ch, self.output_channels, kernel_size=3, padding=1)
        )
        self.output_modules = nn.ModuleList(self.output_modules)


    def _center_data(self, x):
        out = (x - self.data_min_max[0]) / (self.data_min_max[1] - self.data_min_max[0]) # [0, 1]
        return 2 * out - 1 # to put it in [-1, 1]

    def _time_embedding(self, timesteps):
        if self.do_time_embed:
            temb = transformer_timestep_embedding(
                timesteps * self.time_scale_factor, self.time_embed_dim
            )
            temb = self.temb_modules[0](temb)
            temb = self.temb_modules[1](self.act(temb))
        else:
            temb = None

        return temb

    def _do_input_conv(self, h):
        h = self.input_conv(h)
        hs = [h]
        return h, hs

    def _do_downsampling(self, h, hs, temb):
        m_idx = 0
        for scale_count in range(self.num_scales):
            for res_count in range(self.num_res_blocks):
                h = self.downsampling_modules[m_idx](h, temb)
                m_idx += 1
                if scale_count == self.scale_count_to_put_attn:
                    h = self.downsampling_modules[m_idx](h)
                    m_idx += 1
                hs.append(h)

            if scale_count != self.num_scales - 1:
                h = self.downsampling_modules[m_idx](h)
                hs.append(h)
                m_idx += 1

        assert m_idx == len(self.downsampling_modules)

        return h, hs

    def _do_middle(self, h, temb):
        m_idx = 0
        h = self.middle_modules[m_idx](h, temb)
        m_idx += 1
        h = self.middle_modules[m_idx](h)
        m_idx += 1
        h = self.middle_modules[m_idx](h, temb)
        m_idx += 1

        assert m_idx == len(self.middle_modules)

        return h

    def _do_upsampling(self, h, hs, temb):
        m_idx = 0
        for scale_count in reversed(range(self.num_scales)):
            for res_count in range(self.num_res_blocks+1):
                h = self.upsampling_modules[m_idx](torch.cat([h, hs.pop()], dim=1), temb)
                m_idx += 1

                if scale_count == self.scale_count_to_put_attn:
                    h = self.upsampling_modules[m_idx](h)
                    m_idx += 1

            if scale_count != 0:
                h = self.upsampling_modules[m_idx](h)
                m_idx += 1

        assert len(hs) == 0
        assert m_idx == len(self.upsampling_modules)

        return h

    def _do_output(self, h):

        h = self.output_modules[0](h)
        h = self.act(h)
        h = self.output_modules[1](h)

        return h

    def _logistic_output_res(self,
        h, #  ["B", "twoC", "H", "W"]
        centered_x_in, # ["B", "C", "H", "W"]
    ):
        B, twoC, H, W = h.shape
        C = twoC//2
        h[:, 0:C, :, :] = torch.tanh(centered_x_in + h[:, 0:C, :, :])
        return h

    def forward(self,
        x, # ["B", "C", "H", "W"]
        timesteps=None, # ["B"]
    ):

        h = self._center_data(x)
        centered_x_in = h

        temb = self._time_embedding(timesteps)

        h, hs = self._do_input_conv(h)

        h, hs = self._do_downsampling(h, hs, temb)

        h = self._do_middle(h, temb)

        h = self._do_upsampling(h, hs, temb)

        h = self._do_output(h)

        # h (B, 2*C, H, W)
        h = self._logistic_output_res(h, centered_x_in)

        return h


#Based on https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):

    def __init__(self, device, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model, device=device)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.pe = pe

    def forward(self, x, # ["B", "L", "K"]
    ):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, 0:x.size(1), :]
        return self.dropout(x)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward, dropout, temb_dim):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads,
            dropout=dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

        self.film_from_temb = nn.Linear(temb_dim, 2*d_model)

    def forward(self,
        x, # ["B", "L", "K"],
        temb, # ["B", "temb_dim"]
    ):
        B, L, K = x.shape

        film_params = self.film_from_temb(temb)

        x = self.norm1(x + self._sa_block(x))
        x = film_params[:, None, 0:K] * x + film_params[:, None, K:]
        x = self.norm2(x + self._ff_block(x))
        x = film_params[:, None, 0:K] * x + film_params[:, None, K:]

        return x

    def _sa_block(self, x):
        x = self.self_attn(x,x,x)[0]
        return self.dropout1(x)

    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

class FFResidual(nn.Module):
    def __init__(self, d_model, hidden, temb_dim):
        super().__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.norm = nn.LayerNorm(d_model, eps=1e-5)
        self.activation = nn.ReLU()

        self.film_from_temb = nn.Linear(temb_dim, 2*d_model)

    def forward(self, x, temb):
        B, L, K = x.shape

        film_params = self.film_from_temb(temb)

        x = self.norm(x + self.linear2(self.activation(self.linear1(x))))
        x = film_params[:, None, 0:K] * x + film_params[:, None, K:]
        return x
    
class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dim_feedforward,
        dropout, num_output_FFresiduals, time_scale_factor, S, max_len,
        temb_dim, use_one_hot_input, device):
        super().__init__()

        self.temb_dim = temb_dim
        self.use_one_hot_input = use_one_hot_input

        self.S = S

        self.pos_embed = PositionalEncoding(device, d_model, dropout, max_len)

        self.encoder_layers = []
        for i in range(num_layers):
            self.encoder_layers.append(
                TransformerEncoderLayer(d_model, num_heads, dim_feedforward,
                    dropout, 4*temb_dim)
            )
        self.encoder_layers = nn.ModuleList(self.encoder_layers)

        self.output_resid_layers = []
        for i in range(num_output_FFresiduals):
            self.output_resid_layers.append(
                FFResidual(d_model, dim_feedforward, 4*temb_dim)
            )
        self.output_resid_layers = nn.ModuleList(self.output_resid_layers)

        self.output_linear = nn.Linear(d_model, self.S)
        
        if use_one_hot_input:
            self.input_embedding = nn.Linear(S, d_model)
        else:
            self.input_embedding = nn.Linear(1, d_model)

        self.temb_net = nn.Sequential(
            nn.Linear(temb_dim, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, 4*temb_dim)
        )

        self.time_scale_factor = time_scale_factor

    def forward(self, x, # ["B", "L"],
        t #["B"]
    ):
        B, L = x.shape

        temb = self.temb_net(
            transformer_timestep_embedding(
                t*self.time_scale_factor, self.temb_dim
            )
        )
        one_hot_x = nn.functional.one_hot(x, num_classes=self.S) # (B, L, S)

        if self.use_one_hot_input:
            x = self.input_embedding(one_hot_x.float()) # (B, L, K)
        else:
            x = self.normalize_input(x)
            x = x.view(B, L, 1)
            x = self.input_embedding(x) # (B, L, K)

        x = self.pos_embed(x)

        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, temb)

        # x (B, L, K)
        for resid_layer in self.output_resid_layers:
            x = resid_layer(x, temb)

        x = self.output_linear(x) # (B, L, S)

        x = x + one_hot_x

        return x

    def normalize_input(self, x):
        x = x/self.S # (0, 1)
        x = x*2 - 1 # (-1, 1)
        return x

class ResidualMLP(nn.Module):
    def __init__(self, num_layers, d_model, hidden_dim, L, S,
        time_scale_factor, temb_dim):
        super().__init__()

        self.time_scale_factor = time_scale_factor
        self.d_model = d_model
        self.num_layers = num_layers
        self.S = S
        self.temb_dim = temb_dim

        self.activation = nn.ReLU()

        self.input_layer = nn.Linear(L, d_model)

        self.layers1 = []
        self.layers2 = []
        self.norm_layers = []
        self.temb_layers = []
        for n in range(num_layers):
            self.layers1.append(
                nn.Linear(d_model, hidden_dim)
            )
            self.layers2.append(
                nn.Linear(hidden_dim, d_model)
            )
            self.norm_layers.append(
                nn.LayerNorm(d_model)
            )
            self.temb_layers.append(
                nn.Linear(4*temb_dim, 2*d_model)
            )

        self.layers1 = nn.ModuleList(self.layers1)
        self.layers2 = nn.ModuleList(self.layers2)
        self.norm_layers = nn.ModuleList(self.norm_layers)
        self.temb_layers = nn.ModuleList(self.temb_layers)

        self.output_layer = nn.Linear(d_model, L*S)

        self.temb_net = nn.Sequential(
            nn.Linear(temb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4*temb_dim)
        )

    def forward(self,
        x, # ["B", "L"],
        t, # ["B"]
    ):
        B, L= x.shape
        S = self.S

        temb = self.temb_net(
            transformer_timestep_embedding(
                t*self.time_scale_factor, self.temb_dim
            )
        )

        one_hot_x = nn.functional.one_hot(x, num_classes=self.S) # (B, L, S)

        h = self.normalize_input(x)

        h = self.input_layer(h) # (B, d_model)

        for n in range(self.num_layers):
            h = self.norm_layers[n](h + self.layers2[n](self.activation(self.layers1[n](h))))
            film_params = self.temb_layers[n](temb)
            h = film_params[:, 0:self.d_model] * h + film_params[:, self.d_model:]

        h = self.output_layer(h) # (B, L*S)

        h = h.reshape(B, L, S)

        logits = h + one_hot_x

        return logits

    def normalize_input(self, x):
        x = x/self.S # (0, 1)
        x = x*2 - 1 # (-1, 1)
        return x

import torch
import torch.nn.functional as F
import math 

# From https://github.com/yang-song/score_sde_pytorch/ which is from
#  https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
def transformer_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32
    half_dim = embedding_dim // 2
    # magic number 10000 is from transformers
    emb = math.log(max_positions) / (half_dim - 1)
    # emb = math.log(2.) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    # emb = tf.range(num_embeddings, dtype=jnp.float32)[:, None] * emb[None, :]
    # emb = tf.cast(timesteps, dtype=jnp.float32)[:, None] * emb[None, :]
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb