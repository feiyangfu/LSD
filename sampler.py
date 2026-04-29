import torch
import torch.nn as nn
from tqdm import tqdm
from utils import sample_categorical
import torch.nn.functional as F

def teacher_sampler(score_model, scheduler, x_T, nfe=256 ,student_nfe=8, device="cuda"):
    xt = x_T
    timesteps = torch.linspace(1.0, 1e-4, nfe + 1, device=device)
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
            print(i, "\n")
        xt = output_1.xt 

    return xt, model_output_list

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
            model_output *= self.coeffs[i]

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


