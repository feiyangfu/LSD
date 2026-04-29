import os 
import argparse
import torch
from tqdm import tqdm 
from ddms import d3pm, sedd
from sampler import teacher_sampler
import functools
from transformers import GPT2TokenizerFast
import math
from legacy.sedd import SEDD
import local_utils
import numpy as np

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def perturb_discrete_hamming(x_T, num_vocabs=50257, perturb_ratio=0.05):
    if not (0 <= perturb_ratio <= 1):
        raise ValueError("perturb_ratio must be between 0 and 1")

    if num_vocabs <= 1:
        print("Warning: num_vocabs <= 1, cannot perform perturbation.")
        return x_T.clone() 

    num_samples, length = x_T.shape
    device = x_T.device 

    num_to_perturb = math.ceil(length * perturb_ratio)
    num_to_perturb = max(0, min(length, num_to_perturb))

    if num_to_perturb == 0:
        return x_T.clone() 
    x_prime_T = x_T.clone()

    for i in range(num_samples):
        indices_to_perturb = torch.randperm(length, device=device)[:num_to_perturb]
        original_values = x_T[i, indices_to_perturb]
        new_values_candidate = torch.randint(0, num_vocabs - 1,
                                             size=(num_to_perturb,),
                                             device=device,
                                             dtype=torch.long)
        adjustment = (new_values_candidate >= original_values).long()
        new_values = new_values_candidate + adjustment
        x_prime_T[i, indices_to_perturb] = new_values

    return x_prime_T

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="text", type=str, choices=["moons", "count", "cifar", "piano", "text"])
    parser.add_argument("--model_name", type=str, default="sedd", choices=["d3pm", "sedd"])
    parser.add_argument("--noise_schedule", type=str, default="loglinear")
    parser.add_argument("--scheduler_name", type=str, default="tweedie", choices=["euler", "tweedie"])
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--pretrained_model_path", type=str, default="/data0/fufeiyang/src/Text/SEDD/ckpt/")
    
    parser.add_argument("--lsd_num_samples", type=int, default=1)
    parser.add_argument("--lsd_teacher_nfe", type=int, default=256)
    parser.add_argument("--student_nfe", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_vocabs", type=int, default=50257)
    parser.add_argument("--length", type=int, default=1024)
    
    parser.add_argument("--offline_data_dir", type=str, default="./offline_lsd_data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    print(f"Random seed set to: {args.seed}")
    
    os.makedirs(args.offline_data_dir, exist_ok=True)
    
    device = args.device
    dtype = torch.float32

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
            print("Using Tweedie scheduler")
    
    dtype = torch.float
    model, args = load_model_local(args.pretrained_model_path, device=args.device, args=args)
    model.to(args.device, dtype)
    scheduler.to(device, dtype)
    model.eval()  
    
    print(f"Generating teacher data with {args.lsd_num_samples} samples for {args.epochs} epochs...")
    

    teacher_solver_prob_fn = functools.partial(
        teacher_sampler, 
        model, 
        scheduler, 
        nfe=args.lsd_teacher_nfe, 
        student_nfe=args.student_nfe, 
        device=args.device
    )
    
    config_save_path = os.path.join(args.offline_data_dir, "config.pt")
    torch.save({
        'seed': args.seed,
        'num_vocabs': args.num_vocabs,
        'length': args.length,
        'lsd_num_samples': args.lsd_num_samples,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lsd_teacher_nfe': args.lsd_teacher_nfe,
        'student_nfe': args.student_nfe,
        'noise_schedule': args.noise_schedule,
        'model_name': args.model_name,
        'scheduler_name': args.scheduler_name,
    }, config_save_path)
    print(f"Config saved to {config_save_path}")
    
    print("Generating initial noise samples...")
    lsd_dataset = []
    with torch.no_grad():
        for id in tqdm(range(args.lsd_num_samples), desc="Initial sampling"):
            x_T = scheduler.sample_latent(args.batch_size).to(device)
            lsd_dataset.append(x_T.cpu())
    
    for epoch in range(args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*60}")
        
        epoch_data = []
        
        with torch.no_grad():
            for id in tqdm(range(args.lsd_num_samples), desc=f"Generating teacher data (Epoch {epoch})"):
                x_T = lsd_dataset[id].to(device)
                
             
                y_teacher, model_output_list = teacher_solver_prob_fn(x_T=x_T)
                
                sample_data = {
                    'x_T': x_T.cpu(),
                    'y_teacher': y_teacher.cpu(),
                    'model_output_list': [out.cpu() for out in model_output_list],
                }
                epoch_data.append(sample_data)
                
                
                x_T_perturbed = perturb_discrete_hamming(x_T, num_vocabs=args.num_vocabs)
                lsd_dataset[id] = x_T_perturbed.cpu()
        
        
        epoch_save_path = os.path.join(args.offline_data_dir, f"epoch_{epoch}.pt")
        torch.save(epoch_data, epoch_save_path)
        print(f"Epoch {epoch} data saved to {epoch_save_path}")
        
        if epoch == 0:
            tokenizer = GPT2TokenizerFast.from_pretrained('gpt2-large')
            print("\n" + "="*60)
            print("Sample teacher outputs (Epoch 0):")
            print("="*60)
            for i, sample in enumerate(epoch_data[:min(2, len(epoch_data))]):
                y_teacher = sample['y_teacher']
                text_samples = tokenizer.batch_decode(y_teacher)
                for j, text in enumerate(text_samples):
                    print(f"Sample {i}-{j}: {text[:200]}...")
                print("-"*60)
    
    print(f"\n{'='*60}")
    print("Teacher data generation completed!")
    print(f"All data saved to: {args.offline_data_dir}")
    print(f"Total epochs: {args.epochs}")
    print(f"Samples per epoch: {args.lsd_num_samples}")
    print(f"{'='*60}")

