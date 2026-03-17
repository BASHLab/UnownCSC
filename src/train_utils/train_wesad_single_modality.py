import sys
import os
import json
import argparse
import numpy as np
import copy
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

# ----------------------------------------------------------------------
# 1. PATH ROUTING
# ----------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(src_dir)

if src_dir not in sys.path:
    sys.path.append(src_dir)

from models import MultiscaleShapeletCSC
from data_utils import get_stamp_wesad_loaders
from losses import SoftDTW
from utils.dataset_cfg import WESAD

# ----------------------------------------------------------------------
# 2. THE ISOLATED TRAINING LOOP
# ----------------------------------------------------------------------
def train_single_modality(target_mod, args, device):
    print(f"\n{'='*60}")
    print(f"🚀 INITIALIZING ISOLATED TRAINING: {target_mod}")
    print(f"{'='*60}\n")

    # 1. Dynamically modify the config to trick the model into a single-modality state
    base_cfg = WESAD()
    single_cfg = copy.deepcopy(base_cfg)
    single_cfg.modalities = [target_mod]
    single_cfg.variates = {target_mod: base_cfg.variates[target_mod]}
    single_cfg.sampling_rates = {target_mod: base_cfg.sampling_rates[target_mod]}

    data_root = os.path.join(project_root, 'data', 'WESAD_Processed')
    # Note: Dataloader will still load all data, but we will filter it in the batch loop
    train_loader, val_loader = get_stamp_wesad_loaders(data_root, cfg=base_cfg, batch_size=args.batch_size)

    # 2. Initialize the model exclusively for this target modality
    model = MultiscaleShapeletCSC(
        cfg=single_cfg, 
        n_atoms=64, 
        atom_len=32, 
        scales=[4, 2, 1], 
        quantizer=args.quantizer
    ).to(device)

    optimizer = optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': args.lr},
        {'params': model.dictionary.parameters(), 'lr': args.lr * 5},
        {'params': model.decoders.parameters(), 'lr': args.lr},
        {'params': model.quantizer.parameters(), 'lr': args.lr}
    ], weight_decay=1e-4)

    warmup_epochs = 10
    scheduler_warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - warmup_epochs, eta_min=1e-6)
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_cosine], milestones=[warmup_epochs])

    if args.loss == "hybrid":
        sdtw_criterion = SoftDTW(use_cuda=True, gamma=0.1)
    else:
        args.loss = "mse"

    # 3. Create a modality-specific save directory to prevent overwriting
    save_dir = os.path.join(project_root, 'saved_chk_dir_single', target_mod)
    os.makedirs(save_dir, exist_ok=True)
    
    batch_telemetry_log = []

    # 4. Epoch Loop
    for epoch in range(args.epochs):
        model.train()
        epoch_recon_loss = 0.0
        epoch_vq_loss = 0.0
        epoch_l1_loss = 0.0

        pbar = tqdm(train_loader, desc=f"[{target_mod}] Ep {epoch+1:03d}/{args.epochs}")
        
        for batch_idx, (x_dict, _) in enumerate(pbar):
            # FILTER: Extract only the target modality from the dictionary
            target_tensor = x_dict[target_mod].to(device).float()
            x_dict_single = {target_mod: target_tensor}

            # Telemetry for the isolated modality
            if batch_idx == 0:
                epoch_stats = {
                    'epoch': epoch + 1, 
                    'shape': list(target_tensor.shape),
                    'mean': float(target_tensor.mean().item()),
                    'variance': float(target_tensor.var().item())
                }
                batch_telemetry_log.append(epoch_stats)

            optimizer.zero_grad()
            recon_dict, vq_loss, l1_loss = model(x_dict_single)

            # Isolated Loss Calculation
            recon = recon_dict[target_mod]
            mse = F.mse_loss(recon, target_tensor)
            
            if args.loss == "hybrid":
                dtw = sdtw_criterion(recon.permute(0, 2, 1), target_tensor.permute(0, 2, 1))
                batch_recon_loss = mse + 0.1 * dtw.mean()
            else:
                batch_recon_loss = mse

            total_loss = batch_recon_loss + vq_loss + l1_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            model.normalize_dictionary()

            epoch_recon_loss += batch_recon_loss.item()
            epoch_vq_loss += (vq_loss.item() if isinstance(vq_loss, torch.Tensor) else vq_loss)
            epoch_l1_loss += l1_loss.item()
            
            pbar.set_postfix({"Recon": f"{batch_recon_loss.item():.3f}", "LR": f"{scheduler.get_last_lr()[0]:.2e}"})

        scheduler.step()
        
        avg_recon = epoch_recon_loss / len(train_loader)
        print(f"🏁 [{target_mod}] Ep {epoch+1} | Recon Loss: {avg_recon:.4f} | VQ Loss: {epoch_vq_loss/len(train_loader):.4f}")
        
        # Periodic Checkpointing
        if (epoch + 1) % 10 == 0:
            chkpt_path = os.path.join(save_dir, f'stamp_{target_mod}_ep{epoch+1:03d}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_recon
            }, chkpt_path)
            
            with open(os.path.join(save_dir, 'batch_telemetry.json'), 'w') as f:
                json.dump(batch_telemetry_log, f, indent=4)

    # Final Save
    save_path = os.path.join(save_dir, f'stamp_{target_mod}_final.pth')
    torch.save(model.state_dict(), save_path)
    print(f"✅ Training complete for {target_mod}. Saved to {save_path}")

# ----------------------------------------------------------------------
# 3. EXECUTION ORCHESTRATOR
# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train STAMP on Isolated WESAD Modalities")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--quantizer", type=str, default="fsq", choices=["fsq", "ema", "none"])
    parser.add_argument("--loss", type=str, default="mse", choices=["mse", "hybrid"])
    # NEW: Allow targeting a specific modality or looping all
    parser.add_argument("--modality", type=str, default="all", help="Target modality (e.g., 'chest_ECG') or 'all'")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_cfg = WESAD()

    if args.modality == "all":
        print("Executing sequential training loop for ALL 10 modalities.")
        for mod in base_cfg.modalities:
            train_single_modality(mod, args, device)
    else:
        if args.modality not in base_cfg.modalities:
            raise ValueError(f"Modality '{args.modality}' not found in WESAD config. Choose from: {base_cfg.modalities}")
        train_single_modality(args.modality, args, device)