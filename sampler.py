import torch
import torch.nn as nn
from tqdm import tqdm
from utils import sample_categorical
import torch.nn.functional as F

# euler
def teacher_sampler(score_model, scheduler, x_T, nfe=256 ,student_nfe=8, device="cuda"):
    xt = x_T
    timesteps = torch.linspace(1.0, 1e-3, nfe + 1, device=device)
    model_output_list = []

    for i in tqdm(range(nfe), desc="Teacher Sampling", leave=False):
        t_curr = timesteps[i]     
        t_next = timesteps[i+1]   
        step_size = t_curr - t_next 
        t_curr_tensor = t_curr.repeat(xt.shape[0]).to(device)
        sigma_bar = scheduler.sigma_bar(t_curr_tensor)
        model_output = score_model(xt, sigma_bar)

        output_1 = scheduler.step(model_output, xt, t_curr_tensor, step_size, if_last=(i == nfe-1))

        if i  % ((nfe // student_nfe)) == 0:
            model_output_list.append(output_1.xt_prob)
        xt = output_1.xt 

    return xt, model_output_list

# pc
def teacher_sampler(
    model, scheduler, device, tgt_nfe, src_nfe, seed=None,
    num_samples=1, sample_eps=1e-3, sampling_schedule=None, 
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
        output = output * 1.15
        
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
    output = output * 1.15
    xt = scheduler.step(output, xt, t, dt, rev_rate=None, generator=generator, if_last=True).xt
    if fix_length > 0 and x0 is not None:
        xt[:, :fix_length] = x0[:, :fix_length].repeat(num_samples, 1)
    xt_traj.append(xt.cpu())
    
    return xt, torch.stack(xt_traj, dim=1) # |B, T, L|

# Coefficients
class StudentSolverDiscrete(nn.Module):
    def __init__(self,  scheduler, nfe=8, device="cuda:0"):
        super().__init__()
        self.scheduler = scheduler
        self.nfe = nfe
        self.coeffs = nn.Parameter(torch.ones(self.nfe)).requires_grad_(True)
        self.num_vocabs = scheduler.num_vocabs
        self.device = device
        self.model_output_list = []

    def forward(self, xt, score_model, generator=None):
        generator = torch.Generator(self.device).manual_seed(42)
        timesteps = torch.linspace(1, 1e-5, self.nfe + 1)


        for i in range(self.nfe):
            t_curr = timesteps[i]
            t_next = timesteps[i+1]
            step_size = t_curr - t_next
            t_curr_tensor = t_curr.repeat(xt.shape[0]).to(self.device)
            sigma_bar = self.scheduler.sigma_bar(t_curr_tensor)

            model_output = score_model(xt, sigma_bar)
            model_output = model_output * self.coeffs[i]

            self.model_output_list.append(model_output)
      
            score = self.scheduler.output_to_score(model_output, t=t_curr_tensor) # Pass t if needed by conversion
            rev_rate = self.scheduler.sigma(t_curr_tensor)[..., None, None] * self.scheduler.Q_tilde(xt, score)

            
            identity = F.one_hot(xt, num_classes=self.num_vocabs).to(rev_rate).requires_grad_(True)
            

            xt_prob =  identity +  step_size * rev_rate 
            xt_prob = xt_prob.requires_grad_(True)

            if_last = (i == self.nfe - 1)
            xt_prob_final = xt_prob[..., :-1] if if_last else xt_prob
            if i== self.nfe - 1:
                xt = sample_categorical(xt_prob_final, generator = generator)
            else:
                xt = sample_categorical(xt_prob_final, generator=generator)
                
            output_1 = self.scheduler.step(model_output, xt, t_curr_tensor, step_size, if_last=(i == (self.nfe-1)), generator=generator)
            xt = output_1.xt
        return xt, self.model_output_list

# Time steps
class StudentSolverDiscrete(nn.Module):
    def __init__(self,  scheduler, nfe=8, device="cuda:0"):
        super().__init__()
        self.scheduler = scheduler
        self.nfe = nfe
        self.coeffs = nn.Parameter(torch.full((self.nfe+1,), (1.0-1e-5)/(self.nfe))).requires_grad_(True)

        self.num_vocabs = scheduler.num_vocabs
        self.device = device
        self.T = 1.0

    def forward(self, xt, score_model, generator=None):

        timesteps = torch.linspace(1, 1e-5, self.nfe + 1).to(self.device)
        model_output_list = []

        for i in range(self.nfe):
            
            
            t_curr = self.T - torch.cumsum(self.coeffs, dim=0)

            t_next = timesteps[i+1]
            step_size = t_curr - t_next
            t_curr_tensor = t_curr.repeat(xt.shape[0]).to(self.device)

            
            sigma_bar = self.scheduler.sigma_bar(t_curr_tensor)
            model_output = score_model(xt, sigma_bar)
            model_output_list.append(step_size * model_output)
            
            score = self.scheduler.output_to_score(model_output, t=t_curr_tensor)
            rev_rate = self.scheduler.sigma(t_curr_tensor)[..., None, None] * self.scheduler.Q_tilde(xt, score)
            
            identity = F.one_hot(xt, num_classes=self.num_vocabs).to(rev_rate).requires_grad_(True) 
            xt_prob = identity + step_size * rev_rate
            
            xt_prob = xt_prob.requires_grad_(True)

            if_last = (i == self.nfe - 1)
            xt_prob_final = xt_prob[..., :-1] if if_last else xt_prob
            if i== self.nfe - 1:
                xt = sample_categorical(xt_prob_final, generator=generator)
            else:
                xt = sample_categorical(xt_prob_final, generator=generator)
                
        return xt, model_output_list
    