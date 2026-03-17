import sys
import os
import json
import argparse
import numpy as np
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
# 2. THE TRAINING LOOP
# ----------------------------------------------------------------------
def train():
    parser = argparse.ArgumentParser(description="Train STAMP on WESAD")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--quantizer", type=str, default="fsq", choices=["fsq", "ema", "none"])
    parser.add_argument("--loss", type=str, default="mse", choices=["mse", "hybrid"])
    parser.add_argument("--resume_path", type=str, default=None, help="Path to pre-trained checkpoint for fine-tuning")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Initializing STAMP Training on {device}")

    data_root = os.path.join(project_root, 'data', 'WESAD_Processed')
    cfg = WESAD()
    train_loader, val_loader = get_stamp_wesad_loaders(data_root, cfg=cfg, batch_size=args.batch_size)

    model = MultiscaleShapeletCSC(
        cfg=cfg, 
        n_atoms=64, 
        atom_len=32, 
        scales=[4, 2, 1], 
        quantizer=args.quantizer
    ).to(device)
    if args.resume_path and os.path.exists(args.resume_path):
        print(f"🔄 Warm-starting model: Loading weights from {args.resume_path}")
        checkpoint = torch.load(args.resume_path, map_location=device)
        
        # Handle both full checkpoint dicts and raw state_dicts
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

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

    # --- Tracking Initialization ---
    save_dir = os.path.join(project_root, 'saved_chk_dir_dtw')
    os.makedirs(save_dir, exist_ok=True)
    dict_history_list = []
    vq_history_list = []
    batch_telemetry_log = []

    # --- Epoch Loop ---
    for epoch in range(args.epochs):
        model.train()
        epoch_recon_loss = 0.0
        epoch_vq_loss = 0.0
        epoch_l1_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:03d}/{args.epochs}")
        
        for batch_idx, (x_dict, _) in enumerate(pbar):
            x_dict = {mod: tensor.to(device).float() for mod, tensor in x_dict.items()}

            # ---------------------------------------------------------
            # 📡 BATCH TELEMETRY (Track sensitivity on the first batch of epoch)
            # ---------------------------------------------------------
            if batch_idx == 0:
                epoch_stats = {'epoch': epoch + 1, 'modalities': {}}
                for mod, tensor in x_dict.items():
                    epoch_stats['modalities'][mod] = {
                        'shape': list(tensor.shape),
                        'mean': float(tensor.mean().item()),
                        'variance': float(tensor.var().item()),
                        'max': float(tensor.max().item()),
                        'min': float(tensor.min().item())
                    }
                batch_telemetry_log.append(epoch_stats)

            optimizer.zero_grad()
            recon_dict, vq_loss, l1_loss = model(x_dict)

            batch_recon_loss = 0.0
            for mod in x_dict.keys():
                mse = F.mse_loss(recon_dict[mod], x_dict[mod])
                if args.loss == "hybrid":
                    dtw = sdtw_criterion(recon_dict[mod].permute(0, 2, 1), x_dict[mod].permute(0, 2, 1))
                    batch_recon_loss += (mse + 0.1 * dtw.mean())
                else:
                    batch_recon_loss += mse

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
        
        # ---------------------------------------------------------
        # 🎬 COMPONENT EVOLUTION TRACKING (End of Epoch)
        # ---------------------------------------------------------
        # 1. Track the Universal Dictionary
        current_shapelets = model.dictionary.shapelets.detach().cpu().numpy()
        dict_history_list.append(current_shapelets)
        
        # 2. Track the VQ Codebook (if applicable)
        if args.quantizer == "ema" and hasattr(model.quantizer, 'codebook'):
            # EMA codebooks update their weights dynamically
            vq_history_list.append(model.quantizer.codebook.weight.detach().cpu().numpy())
        elif args.quantizer == "fsq":
            # FSQ grids are static mathematically, so we track a dummy array to keep the sweep logic intact,
            # or you can track the latent variance right before the bottleneck in future iterations.
            vq_history_list.append(np.array([-1])) # Placeholder for static FSQ

        # Periodically dump tracking files to disk so RAM doesn't explode
        if (epoch + 1) % 5 == 0:
            np.save(os.path.join(save_dir, 'dict_history.npy'), np.stack(dict_history_list))
            if args.quantizer == "ema":
                np.save(os.path.join(save_dir, 'vq_history.npy'), np.stack(vq_history_list))
            with open(os.path.join(save_dir, 'batch_telemetry.json'), 'w') as f:
                json.dump(batch_telemetry_log, f, indent=4)

        avg_recon = epoch_recon_loss / len(train_loader)
        print(f"🏁 Epoch {epoch+1} Summary | Recon Loss: {avg_recon:.4f} | VQ Loss: {epoch_vq_loss/len(train_loader):.4f}")
        
        # Periodic Checkpointing
        if (epoch + 1) % 10 == 0:
            chkpt_path = os.path.join(save_dir, f'stamp_wesad_ep{epoch+1:03d}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_recon
            }, chkpt_path)
            print(f"💾 Checkpoint safely secured: {chkpt_path}")

    # Final Save
    save_path = os.path.join(save_dir, 'stamp_wesad_final.pth')
    torch.save(model.state_dict(), save_path)
    print(f"✅ Training complete. Model saved to {save_path}")

if __name__ == "__main__":
    train()