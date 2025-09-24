import torch
import os
import gc
import numpy as np
from functools import partial
from tqdm import tqdm
from pathlib import Path

def load_pretrained_model(args):
    # text generation
    if args.dataset_name == "text" or args.dataset_name == "webtext":
        if args.model_name == "sedd":
            from legacy.sedd import SEDD
            # load model
            model = SEDD.from_pretrained(args.pretrained_model_path)
            print("")
            # load config
            args.num_vocabs = model.config.tokens
            args.length = model.config.model.length
            args.noise_schedule = model.config.noise.type
            args.graph = 'absorb'
        
    if args.dataset_name == "count":
        from LSD.model import DDiT
        if args.model_name == "sedd":
            # load model
            model = DDiT(
                num_vocabs=args.num_vocabs,
                cond_dim=args.cond_dim,
                hidden_size=args.hidden_size,
                n_heads=args.n_heads,
                dropout=args.dropout,
                n_blocks=args.n_blocks,
            )
            model.load_state_dict(torch.load(f'{args.pretrained_model_path}'))
            args.graph = 'absorb'

    if args.dataset_name == "piano":
        if args.model_name == 'ctmc':
            # load config 
            args.valid_cfg = get_config_piano(args)
            args.train_cfg = load_ml_collections(Path(args.valid_cfg.train_config_path))
            for item in args.valid_cfg.train_config_overrides:
                set_in_nested_dict(args.train_cfg, item[0], item[1])

            # load model
            from legacy.ctmc import UniformRateSequenceTransformerEMA as CTMC
            model = CTMC(args.train_cfg, args.device)
            loaded_state = torch.load(Path(args.valid_cfg.checkpoint_path), map_location=args.device)
            model.load_state_dict(remove_module_from_keys(loaded_state['model']))

            args.eps = 1e-2
            args.initial_dist = args.valid_cfg.sampler.initial_dist
            args.num_vocabs = args.valid_cfg.data.S
            args.length = np.cumprod(args.valid_cfg.data.shape)[-1].item()
            args.graph = 'uniform'

    if args.dataset_name == "cifar":
        if args.model_name == 'ctmc':
            # load config
            args.valid_cfg = get_config_cifar(args)
            args.train_cfg = load_ml_collections(Path(args.valid_cfg.train_config_path))
            for item in args.valid_cfg.train_config_overrides:
                set_in_nested_dict(args.train_cfg, item[0], item[1])

            # load model
            from legacy.ctmc import GaussianTargetRateImageX0PredEMA as CTMC
            model = CTMC(args.train_cfg, args.device)
            loaded_state = torch.load(Path(args.valid_cfg.checkpoint_path), map_location=args.device, weights_only=False)
            model.load_state_dict(remove_module_from_keys(loaded_state['model']))
            
            args.eps = 1e-2
            args.initial_dist_std = model.Q_sigma
            args.initial_dist = args.valid_cfg.sampler.initial_dist
            args.num_vocabs = args.valid_cfg.data.S
            args.length = np.cumprod(args.valid_cfg.data.shape)[-1].item()
            args.graph = 'gaussian'

    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    return model, args

from ddms import d3pm, sedd
from legacy.ctmc import EulerScheduler
def load_noise_scheduler(args, model):
    if args.model_name == "d3pm":
        if args.scheduler_name == "euler":
            scheduler = d3pm.EulerScheduler(args)
        if args.scheduler_name == "tweedie":
            scheduler = d3pm.AnalyticScheduler(args)
        if args.scheduler_name == "gillespie":
            scheduler = d3pm.GillespieScheduler(args)
    if args.model_name == "sedd":
        if args.scheduler_name == "euler":
            scheduler = sedd.EulerScheduler(args)
        if args.scheduler_name == "tweedie":
            scheduler = sedd.AnalyticScheduler(args)
        if args.scheduler_name == "gillespie":
            scheduler = sedd.GillespieScheduler(args)
    if args.model_name == "ctmc":
        if args.scheduler_name == "euler":
            scheduler = EulerScheduler(args, model)
    return scheduler

def load_sampling_fn(args):
    if args.scheduler_name == "euler":
        return tau_leaping_sampling_fn
    elif args.scheduler_name == "tweedie":
        return tau_leaping_sampling_fn
    elif args.scheduler_name == "gillespie":
        return k_gillespie_sampling_fn

def preset(args):
    """configuration w.r.t. dataset"""
    if args.dataset_name == "count":
        # dependent to dataset
        args.length = 256
        args.max_length = 256
        args.num_vocabs = 32
        args.num_samples = 16384 if args.num_samples == -1 else args.num_samples
        
        # independent to dataset
        args.cond_dim = 128
        args.hidden_size = 128
        args.n_heads = 4
        args.dropout = 0.1
        args.n_blocks = 4
        args.noise_schedule = "loglinear"
        args.mlp_ratio = 4
        
    elif args.dataset_name == "text":
        args.length = 1024
        pass
    
    elif args.dataset_name == "webtext":
        args.length = 1024
        pass
    
    elif args.dataset_name == "piano":
        # dependent to dataset
        args.split = 'train'
        args.length = 256
        args.num_vocabs = 129
        args.noise_schedule = "loglinear"

        # independent to dataset
        args.cond_dim = 256
        args.hidden_size = 256
        args.n_heads = 8
        args.dropout = 0.1
        args.n_blocks = 6
        args.mlp_ratio = 8
        pass
    
    elif args.dataset_name == "imagenet256" or args.dataset_name == "imagenet512":
        pass
    
    elif args.dataset_name == "mnist":
        raise NotImplementedError()
        
    elif args.dataset_name == "cifar":
        args.length = 3*32*32
        args.max_length = args.length
    
    else:
        raise ValueError()

    if args.scheduler_name == 'gillespie':
        args.src_num_function_eval = args.length
        args.src_nfe = args.length
    
    if args.max_length is None:
        args.max_length = args.length
        
    return args

######################
# sampling functions #
######################
@torch.no_grad()
def pc_tau_leaping_sampling_fn(
    model, scheduler, device, tgt_nfe, src_nfe, seed=None,
    num_samples=16, sample_eps=1e-3, sampling_schedule=None, 
    fix_length=0, max_length=None, x0=None, 
    corrector_entry_time=0.1, num_corrector_steps=10, corrector_step_size_multiplier=1.5, **kwargs,
):
    if sampling_schedule is None:
        timesteps = torch.linspace(1, sample_eps, tgt_nfe+1, device=device)
    else:
        timesteps = torch.linspace(1, sample_eps, src_nfe+1)
        timesteps = torch.tensor([timesteps[i].item() for i in sampling_schedule] + [timesteps[-1].item()]).to(device)
    generator = seed if seed is None else torch.Generator(device).manual_seed(seed)
    
    if fix_length > 0 and x0 is not None:
        xt = scheduler.sample_latent(num_samples).to(device).repeat(x0.size(0), 1)
    else:
        xt = scheduler.sample_latent(num_samples).to(device)
    
    if max_length is not None:
        xt = xt[:, :max_length]

    xt_traj = []
    for i in range(tgt_nfe):
        if fix_length > 0 and x0 is not None:
            xt[:, :fix_length] = x0[:, :fix_length].repeat(num_samples, 1)

        dt = timesteps[i] - timesteps[i+1]
        t = timesteps[i] * torch.ones(xt.shape[0], device=device)

        # predictor
        sigma_bar = scheduler.sigma_bar(t)
        output = model(xt, sigma_bar)
        output = scheduler.step(output, xt, t, dt, generator=generator, is_corrector=False)

        # corrector
        if timesteps[i] <= corrector_entry_time:
            for _ in range(num_corrector_steps):
                sigma_bar = scheduler.sigma_bar(t - dt)
                output = model(xt, sigma_bar)
                output = scheduler.step(output, xt, t, corrector_step_size_multiplier * dt, generator=generator, is_corrector=True)
        xt = output.xt
        xt_traj.append(xt.cpu())
    
    output = model(xt, sigma_bar)
    xt = scheduler.step(output, xt, t, dt, rev_rate=None, generator=generator, if_last=True).xt
    if fix_length > 0 and x0 is not None:
        xt[:, :fix_length] = x0[:, :fix_length].repeat(num_samples, 1)
    xt_traj.append(xt.cpu())
    
    return xt, torch.stack(xt_traj, dim=1) # |B, T, L|

@torch.no_grad()
def tau_leaping_sampling_fn(
    model, scheduler, device, tgt_nfe, src_nfe, seed=None,
    num_samples=16, sample_eps=1e-3, sampling_schedule=None, 
    fix_length=0, max_length=None, x0=None, hyper=None, **kwargs,
):
    if sampling_schedule is None:
        # new_coeffs = [0.1250, 0.1188, 0.1183, 0.1182, 0.1181, 0.1181, 0.1182, 0.1254, 0.1250]
        
        # new_coeffs = [0.0625, 0.0610, 0.0609, 0.0620, 0.0625, 0.0610, 0.0609, 0.0620, 0.0625, 0.0610, 0.0609, 0.0620, 0.0625, 0.0610, 0.0609, 0.0620, 0.0625]
        # new_coeffs = [0.03125, 0.03020,0.03015, 0.03010, 0.03055, 
        #               0.03125, 0.03020,0.03015, 0.03010, 0.03055, 
        #               0.03125, 0.03020,0.03015, 0.03010, 0.03055, 
        #               0.03125, 0.03020,0.03015, 0.03010, 0.03055, 
        #               0.03125, 0.03020,0.03015, 0.03010, 0.03055, 
        #               0.03125, 0.03020,0.03015, 0.03010, 0.03055, 
        #               0.03125, 0.03020,0.03125]
        # new_coeffs = [0.015625, 0.015625,0.015625, 0.015625,0.015625, 0.015625, 0.015625, 0.015625,
        #       0.015625, 0.015625,0.015625, 0.015625,0.015625, 0.015625, 0.015625, 0.015625,
        #       0.015625, 0.015625,0.015625, 0.015625,0.015625, 0.015625, 0.015625, 0.015625,
        #      0.015625, 0.015625,0.015625, 0.015625,0.015625, 0.015625, 0.015625, 0.015625,
        #       0.015625, 0.015625,0.015625, 0.015625,0.015625, 0.015625, 0.015625, 0.015625,
        #       0.015625, 0.015625,0.015625, 0.015625,0.015625, 0.015625, 0.015625, 0.015625,
        #       0.015625, 0.015625,0.015625, 0.015625,0.015625, 0.015625, 0.015625, 0.015625,
        # 0.015425, 0.015425,0.015425, 0.015425,0.015425, 0.015425, 0.015425, 0.015425, 0.015625,
        # ]
        
        timesteps = torch.linspace(1, sample_eps, tgt_nfe+1, device=device)
     
    else:
        timesteps = torch.linspace(1, sample_eps, src_nfe+1)
        timesteps = torch.tensor([timesteps[i].item() for i in sampling_schedule] + [timesteps[-1].item()]).to(device)
    generator = seed if seed is None else torch.Generator(device).manual_seed(seed)
    # x0 = x0['input_ids'].to(device)
    # print("x0", x0.size(0))
    if fix_length > 0 and x0 is not None:
        xt = scheduler.sample_latent(num_samples).to(device).repeat(x0.size(0), 1)
    else:
        xt = scheduler.sample_latent(num_samples).to(device)
    
    if max_length is not None:
        xt = xt[:, :max_length]

    xt_traj = []
    for i in range(tgt_nfe):
        if fix_length > 0 and x0 is not None:
            xt[:, :fix_length] = x0[:, :fix_length].repeat(num_samples, 1)

        dt = timesteps[i] - timesteps[i+1]
        t = timesteps[i] * torch.ones(xt.shape[0], device=device)
        # t_curr = 1.0 - i*new_coeffs[i]
        # dt = t_curr - timesteps[i+1]
        # t = t_curr * torch.ones(xt.shape[0], device=device)

        sigma_bar = scheduler.sigma_bar(t)
        output = model(xt, sigma_bar)
        # print("output", output)
        
        output = output 
        # output = output * hyper
        # print("output_new", output)
        # print("xt", xt.shape)
        output = scheduler.step(output, xt, t, dt, generator=generator)
        xt = output.xt
        xt_traj.append(xt.cpu())
    
    output = model(xt, sigma_bar)
    output = output 
    
    xt = scheduler.step(output, xt, t, dt, rev_rate=None, generator=generator, if_last=True).xt
    if fix_length > 0 and x0 is not None:
        xt[:, :fix_length] = x0[:, :fix_length].repeat(num_samples, 1)
    xt_traj.append(xt.cpu())
    
    return xt, torch.stack(xt_traj, dim=1) 

@torch.no_grad()
def k_gillespie_sampling_fn(
    model, scheduler, device, tgt_nfe, src_nfe, seed=None,
    num_samples=16, sampling_schedule=None, 
    fix_length=0, x0=None, **kwargs,
):
    generator = seed if seed is None else torch.Generator(device).manual_seed(seed)
    if fix_length > 0 and x0 is not None:
        xt = scheduler.sample_latent(num_samples).to(device).repeat(x0.size(0), 1)
    else:
        xt = scheduler.sample_latent(num_samples).to(device)

    length = xt.size(1)
    if sampling_schedule is None:
        tokensteps = torch.linspace(0, length, tgt_nfe+1, device=device).long()
        tokensteps[-1] = length
    else:
        tokensteps = torch.linspace(0, length, src_nfe+1, device=device).long()
        tokensteps[-1] = length
        tokensteps = torch.tensor([tokensteps[i].item() for i in sampling_schedule] + [tokensteps[-1].item()]).to(device).long()
    
    xt_traj = []
    t = torch.ones(xt.shape[0], device=device)
    for i in range(tgt_nfe):
        if fix_length > 0 and x0 is not None:
            xt[:, :fix_length] = x0[:, :fix_length].repeat(num_samples, 1)

        dk = tokensteps[i+1] - tokensteps[i]
        sigma_bar = scheduler.sigma_bar(t)
        output = model(xt, sigma_bar)

        output = scheduler.step(output, xt, t, dk, generator=generator)
        xt = output.xt
        t -= output.tau
        xt_traj.append(xt.cpu())
    return xt, torch.stack(xt_traj, dim=1) # |B, T, L|

def flush():
    gc.collect()
    torch.cuda.empty_cache()

##############
# CTMC utils #
##############
import ml_collections
def get_config_cifar(args):

    datasets_folder = 'path/to/datasets'
    model_location = f'{args.pretrained_model_path}/ckpt.pt'
    model_config_location = f'{args.pretrained_model_path}/config.yaml'

    config = ml_collections.ConfigDict()
    config.eval_name = 'CIFAR10'
    config.train_config_overrides = [
        [['device'], args.device],
        [['data', 'root'], datasets_folder],
        [['distributed'], False]
    ]
    config.train_config_path = model_config_location
    config.checkpoint_path = model_location

    config.device = 'cuda'

    config.data = data = ml_collections.ConfigDict()
    data.name = 'DiscreteCIFAR10'
    data.root = datasets_folder
    data.train = True
    data.download = True
    data.S = 256
    data.batch_size = 16
    data.shuffle = True
    data.shape = [3,32,32]
    data.random_flips = False

    config.sampler = sampler = ml_collections.ConfigDict()
    sampler.name = 'PCTauLeaping' # TauLeaping or PCTauLeaping
    sampler.num_steps = 500
    sampler.min_t = 0.01
    sampler.eps_ratio = 1e-9
    sampler.initial_dist = 'gaussian'
    sampler.num_corrector_steps = 10
    sampler.corrector_step_size_multiplier = 1.5
    sampler.corrector_entry_time = 0.1

    return config

def get_config_piano(args):

    pianoroll_dataset_path = 'path/to/pianoroll_dataset'
    model_location = f'{args.pretrained_model_path}/ckpt.pt'
    model_config_location = f'{args.pretrained_model_path}/config.yaml'

    config = ml_collections.ConfigDict()
    config.eval_name = 'piano'
    config.train_config_overrides = [
        [['device'], args.device],
        [['data', 'path'], pianoroll_dataset_path + '/train.npy'],
        [['distributed'], False]
    ]
    config.train_config_path = model_config_location
    config.checkpoint_path = model_location
    config.pianoroll_dataset_path = pianoroll_dataset_path

    config.device = args.device

    config.data = data = ml_collections.ConfigDict()
    data.name = 'LakhPianoroll'
    data.path = pianoroll_dataset_path + '/train.npy'
    data.S = 129
    data.batch_size = 64 #128
    data.shuffle = True
    data.shape = [256]


    config.sampler = sampler = ml_collections.ConfigDict()
    sampler.name = 'ConditionalTauLeaping' # ConditionalTauLeaping or ConditionalPCTauLeaping
    sampler.num_steps = 1000
    sampler.min_t = 0.01
    sampler.eps_ratio = 1e-9
    sampler.initial_dist = 'uniform'
    sampler.test_dataset = pianoroll_dataset_path + '/test.npy'
    sampler.condition_dim = 32
    sampler.num_corrector_steps = 2
    sampler.corrector_step_size_multiplier = 0.1
    sampler.corrector_entry_time = 0.9
    sampler.reject_multiple_jumps = True

    return config

def load_ml_collections(path):
    import yaml
    with open(path, 'r') as f:
        raw_dict = yaml.safe_load(f)
    return ml_collections.ConfigDict(raw_dict)

def remove_module_from_keys(dict):
    # dict has keys of the form a.b.module.c.d
    # changes to a.b.c.d
    new_dict = {}
    for key in dict.keys():
        if '.module.' in key:
            new_key = key.replace('.module.', '.')
            new_dict[new_key] = dict[key]
        else:
            new_dict[key] = dict[key]

    return new_dict

def set_in_nested_dict(nested_dict, keys, new_val):
    """
        Sets a value in a nested dictionary (or ml_collections config)
        e.g.
        nested_dict = \
        {
            'outer1': {
                'inner1': 4,
                'inner2': 5
            },
            'outer2': {
                'inner3': 314,
                'inner4': 654
            }
        } 
        keys = ['outer2', 'inner3']
        new_val = 315
    """
    if len(keys) == 1:
        nested_dict[keys[-1]] = new_val
        return
    return set_in_nested_dict(nested_dict[keys[0]], keys[1:], new_val)

def sample_categorical(categorical_probs, eps=1e-6, generator=None):
    '''use gumbel-max trick, but given probability'''
    if generator is None:
        gumbel_noise = torch.rand_like(categorical_probs)
    else:
        gumbel_noise = torch.rand(categorical_probs.shape, generator=generator, device=generator.device).to(categorical_probs)
    gumbel_noise = (eps - torch.log(eps + (1 - eps) * gumbel_noise))
    
    
    return torch.argmax(categorical_probs / gumbel_noise, dim=-1)


def sample_categorical_with_gradient(categorical_probs, eps=1e-6, generator=None):
    '''use gumbel-max trick, but given probability'''
    if generator is None:
        gumbel_noise = torch.rand_like(categorical_probs)
    else:
        gumbel_noise = torch.rand(categorical_probs.shape, generator=generator, device=generator.device).to(categorical_probs)
    gumbel_noise = (eps - torch.log(eps + (1 - eps) * gumbel_noise))
    
    soft_sample = torch.nn.functional.gumbel_softmax(categorical_probs, tau=0.5, hard=True, dim=-1)
    sample_ids = torch.argmax(categorical_probs / gumbel_noise, dim=-1)
    sample_ids = soft_sample.sum(dim=-1) + (sample_ids - soft_sample.sum(dim=-1)).detach()
    
    return sample_ids 

