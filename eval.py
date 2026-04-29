import os
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
from ddms import d3pm, sedd
from sampler import StudentSolverDiscrete
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from legacy.sedd import SEDD
import local_utils
import numpy as np


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_model_local(root_dir, device, args):
    cfg = local_utils.load_hydra_config_from_run(root_dir)
    score_model = SEDD(cfg).to(device)
    args.num_vocabs = score_model.config.tokens
    args.length = score_model.config.model.length
    args.noise_schedule = score_model.config.noise.type
    args.graph = 'absorb'
    
    ckpt_dir = os.path.join(root_dir, "pytorch_model.bin")
    print("Loading checkpoint from:", ckpt_dir)
    assert os.path.exists(ckpt_dir), f"Checkpoint file {ckpt_dir} does not exist!"
    
    loaded_state = torch.load(ckpt_dir, map_location=device)
    score_model.load_state_dict(loaded_state)
    
    return score_model, args


def load_student_solver(checkpoint_path, scheduler, nfe, device):
    student_solver = StudentSolverDiscrete(scheduler, nfe=nfe, device=device)
    
    print(f"Loading student solver from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'state_dict' in checkpoint:
        student_solver.load_state_dict(checkpoint['state_dict'])
        print(f"Loaded from epoch {checkpoint.get('epoch', 'unknown')}")
        print(f"Training loss: {checkpoint.get('loss', 'unknown')}")
    else:
        student_solver.load_state_dict(checkpoint)
    
    print(f"Student solver coefficients: {student_solver.coeffs.data.cpu().numpy()}")
    return student_solver


@torch.no_grad()
def generate_samples(student_solver, score_model, scheduler, num_samples, batch_size, device):

    all_samples = []
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    for i in tqdm(range(num_batches), desc='Generating samples'):
        current_batch_size = min(batch_size, num_samples - i * batch_size)
        
        x_T = scheduler.sample_latent(current_batch_size).to(device)

        with torch.no_grad():
            samples, _ = student_solver(x_T, score_model)
        
        all_samples.append(samples.cpu())
    
    return torch.cat(all_samples, dim=0)


@torch.no_grad()
def eval_ppl(samples, eval_model, eval_batch_size=1):
    ppls = []
    num_batches = (samples.shape[0] + eval_batch_size - 1) // eval_batch_size
    
    for i in tqdm(range(num_batches), desc='Evaluating PPL', leave=False):
        s = samples[i*eval_batch_size:(i+1)*eval_batch_size]
        
        outputs = eval_model(s, labels=s)
        loss = outputs.loss
        logits = outputs.logits

        logits = logits.transpose(-1, -2)  # [B, vocab_size, seq_len]
        ppl = F.cross_entropy(
            logits[..., :-1],  
            s[..., 1:],        
            reduction="none"
        ).mean(dim=-1).exp().mean()
        
        ppls.append(ppl.item())
    
    return ppls


def main():
    parser = argparse.ArgumentParser(description='Evaluate Student Solver PPL')
    
    parser.add_argument("--model_name", type=str, default="sedd", choices=["d3pm", "sedd"])
    parser.add_argument("--scheduler_name", type=str, default="euler", choices=["euler", "tweedie"])
    parser.add_argument("--pretrained_model_path", type=str, 
                        default="/data0/fufeiyang/src/Text/SEDD/ckpt/",
                       )
    
    parser.add_argument("--student_checkpoint", type=str, required=True,
                        )
    parser.add_argument("--student_nfe", type=int, default=8,
                        )
    
    parser.add_argument("--num_samples", type=int, default=1000,
                        )
    parser.add_argument("--batch_size", type=int, default=8,
                        )
    
    parser.add_argument("--eval_model_name", type=str, default="gpt2-large",
                        )
    parser.add_argument("--eval_batch_size", type=int, default=4,
                        )
    
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--output_file", type=str, default=None,
                        )
    parser.add_argument("--save_samples", type=str, default=None,
                        )
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    device = args.device
    dtype = torch.float32
    
    
    model, args = load_model_local(args.pretrained_model_path, device=device, args=args)
    model.to(device, dtype)
    model.eval()

    print(f"\nInitializing {args.scheduler_name} scheduler...")
    if args.model_name == "d3pm":
        if args.scheduler_name == "euler":
            scheduler = d3pm.EulerScheduler(args)
        else:
            scheduler = d3pm.AnalyticScheduler(args)
    elif args.model_name == "sedd":
        if args.scheduler_name == "euler":
            scheduler = sedd.EulerScheduler(args)
        else:
            scheduler = sedd.AnalyticScheduler(args)
    
    scheduler.to(device, dtype)
    print("Scheduler initialized!")
    
    print("\nLoading Student solver...")
    student_solver = load_student_solver(
        args.student_checkpoint, 
        scheduler, 
        args.student_nfe, 
        device
    )
    student_solver.to(device, dtype)
    student_solver.eval()
    print("Student solver loaded successfully!")
    
    print(f"\nGenerating {args.num_samples} samples...")
    samples = generate_samples(
        student_solver, 
        model, 
        scheduler, 
        args.num_samples, 
        args.batch_size, 
        device
    )
    print(f"Generated samples shape: {samples.shape}")
    
    if args.save_samples:
        torch.save(samples, args.save_samples)
        print(f"Samples saved to {args.save_samples}")

    print(f"\nLoading evaluation model: {args.eval_model_name}...")
    eval_model = GPT2LMHeadModel.from_pretrained(args.eval_model_name).to(device)
    eval_model.eval()
    print("Evaluation model loaded!")
    
    print("\nEvaluating PPL...")
    samples = samples.to(device)
    ppls = eval_ppl(samples, eval_model, args.eval_batch_size)
    
    print(f"\n{'='*60}")
    print("Evaluation Results")
    print(f"{'='*60}")
    print(f"Number of samples: {len(samples)}")
    print(f"Number of batches: {len(ppls)}")
    print(f"PPL per batch: {ppls}")
    print(f"Mean PPL: {np.mean(ppls):.4f}")
    print(f"Std PPL: {np.std(ppls):.4f}")
    print(f"Min PPL: {np.min(ppls):.4f}")
    print(f"Max PPL: {np.max(ppls):.4f}")
    print(f"{'='*60}\n")
    
    if args.output_file:
        results = {
            'ppls': ppls,
            'mean_ppl': np.mean(ppls),
            'std_ppl': np.std(ppls),
            'min_ppl': np.min(ppls),
            'max_ppl': np.max(ppls),
            'num_samples': len(samples),
            'config': vars(args)
        }
        torch.save(results, args.output_file)
        print(f"Results saved to {args.output_file}")
    
    return ppls


if __name__ == "__main__":
    main()

