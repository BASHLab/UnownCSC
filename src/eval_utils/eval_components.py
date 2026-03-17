import sys
import os
import json
import torch
import argparse
import numpy as np
import matplotlib
# Force matplotlib to use a non-interactive backend for headless cluster rendering
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns

# Path routing to keep imports clean
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(src_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from models.multiscale_csc_models import MultiscaleShapeletCSC
from data_utils import get_stamp_wesad_loaders
from utils.dataset_cfg import WESAD

def generate_dictionary_animation(history_path, out_dir):
    """
    Turns the dict_history.npy into a 5fps GIF showing shapelet evolution for all 64 atoms.
    """
    if not os.path.exists(history_path):
        print(f"⚠️ No dictionary history found at {history_path}. Skipping animation.")
        return

    print("🎬 Rendering Full 64-Atom Dictionary Evolution Animation...")
    os.makedirs(out_dir, exist_ok=True)
    history = np.load(history_path) # Shape: (Epochs, Atoms, Channels, Length)
    num_epochs, num_atoms = history.shape[0], history.shape[1]
    
    # Update to an 8x8 grid for 64 atoms
    fig, axes = plt.subplots(8, 8, figsize=(16, 16))
    fig.suptitle('Evolution of Universal Shapelets (All 64 Atoms)', fontsize=20, y=0.98)
    
    lines = []
    for i, ax in enumerate(axes.flatten()):
        # Stop if we somehow have fewer than 64 atoms in the array
        if i >= num_atoms:
            ax.axis('off')
            continue
            
        line, = ax.plot([], [], color='black', lw=1.0) # Slightly thinner line for density
        ax.set_xlim(0, history.shape[-1])
        ax.set_ylim(-1.5, 1.5) # Normalized range
        ax.axis('off')
        ax.set_title(f"Atom {i}", fontsize=8) # Smaller font to fit the grid
        lines.append(line)

    # Adjust vertical space so the 64 titles don't overlap with the plots above them
    plt.subplots_adjust(hspace=0.5, wspace=0.1)

    def init():
        for line in lines:
            line.set_data([], [])
        return lines

    def update(epoch):
        for i, line in enumerate(lines):
            # Plot the i-th atom's weights for that epoch
            data = history[epoch, i, 0, :] 
            line.set_data(np.arange(len(data)), data)
            
        # Dynamically update the title with the correct epoch number
        # Assuming we save history every 5 epochs based on the training script
        fig.suptitle(f'Universal Shapelets Evolution - Epoch {epoch * 5}', fontsize=20, y=0.98)
        return lines

    ani = animation.FuncAnimation(fig, update, frames=num_epochs, init_func=init, blit=True)
    
    # Save as GIF
    writer = animation.PillowWriter(fps=5)
    anim_path = os.path.join(out_dir, 'dictionary_evolution_64.gif')
    ani.save(anim_path, writer=writer)
    plt.close()
    print(f"✅ Animation saved to {anim_path}")

def analyze_dictionary_components(epoch):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔍 Initializing STAMP Component Analysis on {device} (Headless Mode) for Epoch {epoch}")

    # 1. Setup & Load Model
    data_root = os.path.join(project_root, 'data', 'WESAD_Processed')
    cfg = WESAD()
    
    # We only need a small validation batch for activation analysis
    _, val_loader = get_stamp_wesad_loaders(data_root, cfg=cfg, batch_size=32)

    model = MultiscaleShapeletCSC(
        cfg=cfg, 
        n_atoms=64, 
        atom_len=32, 
        scales=[4, 2, 1], 
        quantizer="fsq"
    ).to(device)

    chkpt_name = f'stamp_wesad_ep{epoch:03d}.pth'
    chkpt_path = os.path.join(project_root, 'saved_chk_dir_dtw', chkpt_name)
    
    if not os.path.exists(chkpt_path):
        print(f"❌ Cannot find checkpoint at {chkpt_path}")
        return

    checkpoint = torch.load(chkpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Create epoch-specific output directory
    out_dir = os.path.join(project_root, 'eval_outputs', f'epoch_{epoch:03d}')
    os.makedirs(out_dir, exist_ok=True)
    
    # Dictionary to hold all quantifiable metrics for JSON export
    eval_metrics = {}

    # ==========================================
    # ANALYSIS 1: The Learned Shapelet Dictionary
    # ==========================================
    print("🎨 Plotting Universal Shapelet Atoms...")
    model.dictionary.normalize_parameters()
    shapelets = model.dictionary.shapelets.detach().cpu().numpy()
    
    fig, axes = plt.subplots(4, 4, figsize=(12, 8))
    fig.suptitle('Learned Universal Physiological Shapelets (First 16 Atoms)', fontsize=16)
    
    for i, ax in enumerate(axes.flatten()):
        ax.plot(shapelets[i, i, :], color='black')
        ax.set_title(f"Atom {i}")
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'learned_shapelets.pdf'))
    plt.close()

    # ==========================================
    # ANALYSIS 2: Cross-Modal Activation Heatmap
    # ==========================================
    print("🔥 Calculating Modality Latent Overlap...")
    x_dict, _ = next(iter(val_loader))
    x_dict = {mod: tensor.to(device).float() for mod, tensor in x_dict.items()}

    with torch.no_grad():
        z_raw_dict = model.encoder(x_dict)
    
    activation_matrix = np.zeros((len(cfg.modalities), 64))
    
    for i, mod in enumerate(cfg.modalities):
        avg_activation = torch.mean(torch.abs(z_raw_dict[mod]), dim=(0, 2)).cpu().numpy()
        activation_matrix[i, :] = avg_activation

    row_sums = activation_matrix.sum(axis=1, keepdims=True) + 1e-8
    activation_matrix_norm = activation_matrix / row_sums

    plt.figure(figsize=(14, 6))
    sns.heatmap(activation_matrix_norm, 
                yticklabels=cfg.modalities, 
                cmap="viridis", 
                cbar_kws={'label': 'Relative Activation Strength'})
    
    plt.title('Latent Codebook Utilization Across Modalities', fontsize=14)
    plt.xlabel('Shapelet Atom Index (0 - 63)')
    plt.ylabel('Sensor Modality')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'modality_activation_heatmap.pdf'))
    plt.close()

    # ==========================================
    # ANALYSIS 3: Codebook Usage Bar Chart & Metrics
    # ==========================================
    print("📊 Calculating Codebook Utilization Metrics...")
    
    global_usage = np.sum(activation_matrix, axis=0)
    total_activations = np.sum(global_usage) + 1e-8
    usage_percentages = (global_usage / total_activations) * 100
    
    # Calculate active codes (threshold > 0.1% usage)
    active_codes = int(np.sum(usage_percentages > 0.1))
    utilization_rate = (active_codes / 64.0) * 100
    
    # Store metrics
    eval_metrics['total_atoms'] = 64
    eval_metrics['active_atoms'] = active_codes
    eval_metrics['utilization_rate_percent'] = round(utilization_rate, 2)
    eval_metrics['per_atom_usage_percent'] = [round(float(u), 4) for u in usage_percentages]
    
    plt.figure(figsize=(12, 5))
    plt.bar(range(64), usage_percentages, color='teal', alpha=0.8)
    plt.axhline(y=0.1, color='red', linestyle='--', label='Dead Code Threshold (0.1%)')
    
    plt.title(f'Universal Dictionary Utilization (Active: {utilization_rate:.1f}%)', fontsize=14)
    plt.xlabel('Shapelet Atom Index (0 - 63)')
    plt.ylabel('Global Activation Frequency (%)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'atom_utilization_bar.pdf'))
    plt.close()

    # ==========================================
    # ANALYSIS 4: Export Metrics to JSON
    # ==========================================
    json_path = os.path.join(out_dir, 'component_metrics.json')
    with open(json_path, 'w') as f:
        json.dump(eval_metrics, f, indent=4)
        
    print(f"✅ Component analysis complete.")
    print(f"📁 Plots saved to: {out_dir}")
    print(f"📄 Metrics saved to: {json_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze STAMP dictionary and latents")
    parser.add_argument("--epoch", type=int, default=30, help="Epoch checkpoint to analyze (e.g., 30)")
    parser.add_argument("--animate", action="store_true", help="Generate dictionary evolution MP4")
    args = parser.parse_args()

    # Always run the static component analysis for the requested epoch
    analyze_dictionary_components(args.epoch)
    
    # If the user passed --animate, run the video generator
    if args.animate:
        hist_path = os.path.join(project_root, 'saved_chk_dir', 'dict_history.npy')
        anim_out_dir = os.path.join(project_root, 'eval_outputs', 'animations')
        generate_dictionary_animation(hist_path, anim_out_dir)