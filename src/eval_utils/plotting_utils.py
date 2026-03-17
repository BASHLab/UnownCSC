import sys
import os
import torch
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg') # Headless-safe
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------------------------------
# 1. PATH ROUTING
# -----------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(src_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from models.multiscale_csc_models import MultiscaleShapeletCSC
from data_utils import get_stamp_wesad_loaders
from utils.dataset_cfg import WESAD

sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)

# -----------------------------------------------------------------------------
# 2. PLOTTING UTILITIES 
# -----------------------------------------------------------------------------
def plot_long_horizon_reconstruction(original_signal, recon_signal, modality_name, duration_sec, out_dir):
    print(f"⏱️ Generating Long-Horizon Plot for {modality_name}...")
    plt.figure(figsize=(18, 4))
    time_axis = np.arange(len(original_signal))
    
    plt.plot(time_axis, original_signal, color='black', alpha=0.5, linewidth=1.2, label='Ground Truth')
    plt.plot(time_axis, recon_signal, color='teal', alpha=0.8, linewidth=1.2, linestyle='--', label='STAMP Recon')
    
    plt.title(f'Continuous Long-Horizon Reconstruction - {modality_name}', fontsize=16)
    plt.xlabel('Time Steps')
    plt.ylabel('Amplitude')
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    save_path = os.path.join(out_dir, f'long_horizon_{modality_name}.pdf')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def plot_multimodal_multiresolution_grid(original_dict, full_recon_dict, scale_recon_dict, target_modalities, out_dir, sample_idx=0):
    print("📈 Generating Multimodal & Multiresolution Master Grid...")
    
    num_mods = len(target_modalities)
    scales = [4, 2, 1] 
    num_cols = 1 + len(scales) 
    
    fig, axes = plt.subplots(num_mods, num_cols, figsize=(4 * num_cols, 2.5 * num_mods), sharex=True, squeeze=False)
    fig.suptitle('Cross-Modal and Multiresolution Signal Reconstruction', fontsize=18, fontweight='bold', y=1.02)
    
    for row_idx, mod in enumerate(target_modalities):
        # Extract and squeeze to guarantee a 1D (Time,) array
        orig_signal = original_dict[mod][sample_idx].squeeze()
        full_recon = full_recon_dict[mod][sample_idx].squeeze()
        time_axis = np.arange(len(orig_signal))
        
        # Column 0: Full Reconstruction
        ax_main = axes[row_idx, 0]
        ax_main.plot(time_axis, orig_signal, color='black', alpha=0.4, linewidth=1.5, label='Original')
        ax_main.plot(time_axis, full_recon, color='teal', alpha=0.9, linewidth=1.5, linestyle='--', label='Full Recon')
        ax_main.set_ylabel(f'{mod}\nAmplitude', fontweight='bold')
        if row_idx == 0:
            ax_main.set_title('Combined Reconstruction', fontweight='bold')
            ax_main.legend(loc='upper right', fontsize=8)
            
        # Columns 1-N: Individual Scales
        for col_idx, scale in enumerate(scales, start=1):
            ax_scale = axes[row_idx, col_idx]
            
            # Extract the specific scale if it exists, otherwise plot flat line
            if scale in scale_recon_dict[mod] and len(scale_recon_dict[mod][scale]) > 0:
                scale_signal = scale_recon_dict[mod][scale][sample_idx].squeeze()
            else:
                scale_signal = np.zeros_like(orig_signal)
            
            color = 'tab:blue' if scale == 4 else ('tab:orange' if scale == 2 else 'tab:red')
            scale_label = 'Coarse' if scale == 4 else ('Mid' if scale == 2 else 'Fine')
            
            ax_scale.plot(time_axis, orig_signal, color='black', alpha=0.2, linewidth=1.0)
            ax_scale.plot(time_axis, scale_signal, color=color, alpha=0.9, linewidth=1.5, label=f'Scale {scale}')
            
            if row_idx == 0:
                ax_scale.set_title(f'Resolution: {scale_label}', fontweight='bold')
                ax_scale.legend(loc='upper right', fontsize=8)

    for ax in axes[-1, :]:
        ax.set_xlabel('Time Steps', fontweight='bold')
        
    plt.tight_layout()
    save_path = os.path.join(out_dir, 'multimodal_multiresolution_master_grid.pdf')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✅ Grid saved to {save_path}")

# -----------------------------------------------------------------------------
# 3. EVALUATION ENGINE
# -----------------------------------------------------------------------------
def evaluate_real_reconstructions(epoch):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔍 Loading Real WESAD Data and Epoch {epoch} Checkpoint on {device}...")

    data_root = os.path.join(project_root, 'data', 'WESAD_Processed')
    cfg = WESAD()
    
    _, val_loader = get_stamp_wesad_loaders(data_root, cfg=cfg, batch_size=1)
    
    model = MultiscaleShapeletCSC(
        cfg=cfg, n_atoms=64, atom_len=32, scales=[4, 2, 1], quantizer="fsq"
    ).to(device)

    chkpt_name = f'stamp_wesad_ep{epoch:03d}.pth'
    chkpt_path = os.path.join(project_root, 'saved_chk_dir_dtw', chkpt_name)
    
    if not os.path.exists(chkpt_path):
        print(f"❌ Checkpoint {chkpt_path} not found!")
        return

    checkpoint = torch.load(chkpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    out_dir = os.path.join(project_root, 'eval_outputs', f'epoch_{epoch:03d}', 'reconstructions')
    os.makedirs(out_dir, exist_ok=True)

    x_dict, _ = next(iter(val_loader))
    x_dict = {mod: tensor.to(device).float() for mod, tensor in x_dict.items()}

    print("🧠 Running forward passes...")
    with torch.no_grad():
        full_recon_dict, _, _ = model(x_dict)
        
        scale_recon_dict = {mod: {} for mod in cfg.modalities}
        for scale in model.scales:
            try:
                isolated_recon = model.reconstruct_at_scale(x_dict, target_scale=scale)
                for mod in cfg.modalities:
                    scale_recon_dict[mod][scale] = isolated_recon[mod].cpu().numpy()
            except AttributeError:
                if scale == 4:
                    print("⚠️ 'reconstruct_at_scale' missing in model. Scale columns will be blank.")

    # Retain the batch dimension so .squeeze() works properly in the plotter
    formatted_orig = {mod: x_dict[mod].cpu().numpy() for mod in cfg.modalities}
    formatted_full = {mod: full_recon_dict[mod].cpu().numpy() for mod in cfg.modalities}

    target_mods = ['chest_ECG', 'chest_EDA', 'wrist_BVP'] 
    
    plot_multimodal_multiresolution_grid(
        original_dict=formatted_orig,
        full_recon_dict=formatted_full,
        scale_recon_dict=scale_recon_dict,
        target_modalities=target_mods,
        out_dir=out_dir,
        sample_idx=0
    )
    
    plot_long_horizon_reconstruction(
        original_signal=formatted_orig['chest_ECG'][0].squeeze(),
        recon_signal=formatted_full['chest_ECG'][0].squeeze(),
        modality_name='chest_ECG',
        duration_sec=8, 
        out_dir=out_dir
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=30)
    args = parser.parse_args()
    evaluate_real_reconstructions(args.epoch)