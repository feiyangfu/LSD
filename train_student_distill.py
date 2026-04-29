import os 
import argparse
import torch
from tqdm import tqdm 
from ddms import d3pm, sedd
from sampler import StudentSolverDiscrete
from transformers import GPT2TokenizerFast
from legacy.sedd import SEDD
import local_utils
import numpy as np

class EMA:
    def __init__(self, model, ema_decay):
        self.ema_decay = ema_decay
        self.ema_param = [p.clone().detach().requires_grad_(False) for p in model.parameters() if p.requires_grad]
        self.num_updates = 0

    def to(self, device, dtype):
        self.ema_param = [p.to(device, dtype) for p in self.ema_param]

    def update_ema(self, model):
        if len(self.ema_param) == 0:
            raise ValueError("Shadow params not initialized before first ema update!")

        self.num_updates += 1
        ema_decay = min(self.ema_decay, (1 + self.num_updates) / (10 + self.num_updates))
        one_minus_decay = 1.0 - ema_decay
        with torch.no_grad():
            parameters = [p for p in model.parameters() if p.requires_grad]
            for ema_param, param in zip(self.ema_param, parameters):
                ema_param.sub_(one_minus_decay * (ema_param - param))

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

def compute_loss(yt_prob, student_output):
    length = len(student_output)
    assert len(student_output) == len(yt_prob)
    eps = 1e-3
    final_loss = 0.0
    for i in range(length):
        yt_prob[i] = torch.softmax(yt_prob[i], dim=-1)
        student_output[i] = torch.softmax(student_output[i], dim=-1)        
        loss = -torch.mean(torch.sum(student_output[i] * torch.log(student_output[i] + eps) / (yt_prob[i] + eps)), dim = -1)
        final_loss += loss
    return final_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="text", type=str)
    parser.add_argument("--model_name", type=str, default="sedd", choices=["d3pm", "sedd"])
    parser.add_argument("--noise_schedule", type=str, default="loglinear")
    parser.add_argument("--scheduler_name", type=str, default="tweedie", choices=["euler", "tweedie"])
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--pretrained_model_path", type=str, default="/data0/fufeiyang/src/Text/SEDD/ckpt/")
    
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--student_nfe", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=1)
    
    parser.add_argument("--offline_data_dir", type=str, default="./offline_data")
    parser.add_argument("--output_dir", type=str, default="./runs/text-sedd")
    parser.add_argument("--load_epoch_start", type=int, default=0, help="Start from this epoch")
    parser.add_argument("--load_epoch_end", type=int, default=-1, help="End at this epoch (-1 means all epochs)")
    
    args = parser.parse_args()
    

    config_path = os.path.join(args.offline_data_dir, "config.pt")
    assert os.path.exists(config_path), f"Config file {config_path} does not exist!"
    
    saved_config = torch.load(config_path)
    print(f"Loaded config from {config_path}")
    print(f"Teacher config: {saved_config}")
    
    seed = saved_config['seed']
    set_seed(seed)
    
    args.num_vocabs = saved_config['num_vocabs']
    args.length = saved_config['length']
    args.lsd_num_samples = saved_config['lsd_num_samples']
    epochs = saved_config['epochs']
    args.student_nfe = saved_config['student_nfe']
    
    if args.load_epoch_end == -1:
        args.load_epoch_end = epochs
    
    os.makedirs(args.output_dir, exist_ok=True)
    args.exp = f"{args.lr}-{args.batch_size}"
    
    device = args.device
    dtype = torch.float32
    eps = args.eps

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
    model.eval() 
    
    student_solver = StudentSolverDiscrete(scheduler, nfe=args.student_nfe, device=args.device)
    student_solver.to(device, dtype)
    student_solver.train()
    
    lsd_optimizer = torch.optim.Adam([student_solver.coeffs], lr=args.lr)
    
    ema = EMA(model, args.ema_decay)
    ema.to(device, dtype)

    scheduler.to(device, dtype)
    
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2-large')
    
    print(f"\n{'='*60}")
    print(f"Training student solver on offline data")
    print(f"Epochs: {args.load_epoch_start} to {args.load_epoch_end}")
    print(f"Samples per epoch: {args.lsd_num_samples}")
    print(f"Learning rate: {args.lr}")
    print(f"Student NFE: {args.student_nfe}")
    print(f"{'='*60}\n")
    
    global_step = 0
    all_losses = []
    
    for epoch in range(args.load_epoch_start, args.load_epoch_end):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{args.load_epoch_end}")
        print(f"{'='*60}")

        epoch_data_path = os.path.join(args.offline_data_dir, f"epoch_{epoch}.pt")
        if not os.path.exists(epoch_data_path):
            print(f"Warning: {epoch_data_path} does not exist, skipping...")
            continue
        
        epoch_data = torch.load(epoch_data_path, map_location='cpu')
        print(f"Loaded epoch {epoch} data from {epoch_data_path}")
        
        epoch_loss = 0.0
        
        for id in tqdm(range(len(epoch_data)), desc=f"Training (Epoch {epoch})"):
            sample_data = epoch_data[id]
            
            x_T = sample_data['x_T'].to(device)
            model_output_list = [out.to(device) for out in sample_data['model_output_list']]
            
            with torch.set_grad_enabled(True):
                x_t, x_prob_list = student_solver(x_T, model)
            
            loss = compute_loss(model_output_list, x_prob_list)
            
            lsd_optimizer.zero_grad()
            loss.backward()
            lsd_optimizer.step()
            
            epoch_loss += loss.item()
            global_step += 1
            
            if (id + 1) % max(1, len(epoch_data) // 5) == 0:
                avg_loss = epoch_loss / (id + 1)
                print(f"  Step {id + 1}/{len(epoch_data)}, Avg Loss: {avg_loss:.6f}, "
                      f"Coeffs: {student_solver.coeffs.data.cpu().numpy()}")
        
        avg_epoch_loss = epoch_loss / len(epoch_data)
        all_losses.append(avg_epoch_loss)
        print(f"Epoch {epoch} completed. Average Loss: {avg_epoch_loss:.6f}")
        
        if (epoch + 1) % 5 == 0 or epoch == args.load_epoch_end - 1:
            checkpoint_path = f"{args.output_dir}/student_solver_epoch_{epoch}-{args.exp}.pt"
            torch.save({
                'epoch': epoch,
                'state_dict': student_solver.state_dict(),
                'optimizer': lsd_optimizer.state_dict(),
                'loss': avg_epoch_loss,
                'coeffs': student_solver.coeffs.data.cpu(),
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
    
    print(f"\n{'='*60}")
    print("Training completed!")
    print(f"{'='*60}")
    
    print("\nFinal Student Solver State Dict:")
    state_dict = student_solver.state_dict()
    for name, param in state_dict.items():
        print(f"{name}: shape={param.shape}, mean={param.mean().item():.6f}, std={param.std().item():.6f}")
        print(f"  Sample values: {param.flatten()[:5].tolist()}")
    
    final_save_path = f"{args.output_dir}/student_solver_final-{args.exp}.pt"
    torch.save(student_solver.state_dict(), final_save_path)
    print(f"\nFinal model saved to {final_save_path}")
