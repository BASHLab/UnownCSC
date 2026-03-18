"""
generate_embeddings.py

Runs the LIMU-BERT encoder (model.pt) over raw data_20_120.npy files
and saves per-sample embeddings as sensor_image_{idx}.npy.

These embeddings are what llasa_classification_4_limubert.py expects.

Usage:
    # Baseline (raw data):
    python generate_embeddings.py --mode baseline

    # Reconstructed data (post compression):
    python generate_embeddings.py --mode reconstructed --recon_dir ./llasa/recon_data

Directory structure produced:
    llasa/embeddings/baseline/{dataset}/sensor_image_{idx}.npy
    llasa/embeddings/reconstructed/{dataset}/sensor_image_{idx}.npy
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


# ---------------------------------------------------------------------------
# LIMU-BERT model definition (mirrors the weights in model.pt)
# ---------------------------------------------------------------------------

class LIMUBertEmbedding(nn.Module):
    def __init__(self, hidden=72, seq_len=120, in_dim=6):
        super().__init__()
        self.lin = nn.Linear(in_dim, hidden)
        self.pos_embed = nn.Embedding(seq_len, hidden)
        self.norm = nn.LayerNorm(hidden, elementwise_affine=True)
        # LayerNorm in LIMU-BERT uses gamma/beta naming — we remap below

    def forward(self, x):
        # x: (B, T, 6)
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        out = self.lin(x) + self.pos_embed(positions)
        out = self.norm(out)
        return out


class LIMUBertPWFF(nn.Module):
    """Point-wise feedforward with fc1/fc2 naming to match model.pt keys."""
    def __init__(self, hidden=72):
        super().__init__()
        self.fc1 = nn.Linear(hidden, hidden * 2)
        self.fc2 = nn.Linear(hidden * 2, hidden)

    def forward(self, x):
        return self.fc2(torch.nn.functional.gelu(self.fc1(x)))


class LIMUBertAttention(nn.Module):
    def __init__(self, hidden=72):
        super().__init__()
        self.proj_q = nn.Linear(hidden, hidden)
        self.proj_k = nn.Linear(hidden, hidden)
        self.proj_v = nn.Linear(hidden, hidden)
        self.scale  = hidden ** -0.5

    def forward(self, x):
        Q = self.proj_q(x)
        K = self.proj_k(x)
        V = self.proj_v(x)
        attn = torch.softmax(Q @ K.transpose(-2, -1) * self.scale, dim=-1)
        # output proj (transformer.proj) is applied at the transformer level
        return attn @ V


class LIMUBertTransformerBlock(nn.Module):
    def __init__(self, hidden=72):
        super().__init__()
        self.attn  = LIMUBertAttention(hidden)
        self.norm1 = nn.LayerNorm(hidden, elementwise_affine=True)
        self.norm2 = nn.LayerNorm(hidden, elementwise_affine=True)
        self.pwff  = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.GELU(),
            nn.Linear(hidden * 2, hidden),
        )

    def forward(self, x):
        x = self.norm1(x + self.attn(x))
        x = self.norm2(x + self.pwff(x))
        return x


class LIMUBertEncoder(nn.Module):
    """
    Encodes (B, 120, in_dim) IMU sequences → (B, 72) embeddings.
    Architecture inferred directly from model.pt weight shapes.
    in_dim is inferred at load time from embed.lin.weight shape.
    """
    def __init__(self, hidden=72, seq_len=120, in_dim=6):
        super().__init__()
        self.transformer = nn.ModuleDict({
            'embed': LIMUBertEmbedding(hidden, seq_len, in_dim),
            'attn':  LIMUBertAttention(hidden),
            'proj':  nn.Linear(hidden, hidden),
            'norm1': nn.LayerNorm(hidden, elementwise_affine=True),
            'norm2': nn.LayerNorm(hidden, elementwise_affine=True),
            'pwff':  LIMUBertPWFF(hidden),
        })
        self.fc     = nn.Linear(hidden, hidden)
        self.linear = nn.Linear(hidden, hidden)
        self.norm   = nn.LayerNorm(hidden, elementwise_affine=True)
        # decoder exists in weights but is not used for embedding generation
        self.decoder = nn.Linear(hidden, in_dim)

    def forward(self, x):
        # x: (B, T, 6)
        h = self.transformer['embed'](x)

        # single transformer block (only one set of attn/norm weights in model.pt)
        attn_out = self.transformer['attn'](h)
        h = self.transformer['norm1'](h + attn_out)
        ff_out = self.transformer['pwff'](h)
        h = self.transformer['norm2'](h + ff_out)

        # project transformer output
        h = self.transformer['proj'](h)

        # pool over time → (B, 72)
        h = h.mean(dim=1)

        # final projection layers
        h = self.fc(h)
        h = self.linear(h)
        h = self.norm(h)
        return h  # (B, 72)


# ---------------------------------------------------------------------------
# Weight loading — remaps gamma/beta → weight/bias for LayerNorm
# ---------------------------------------------------------------------------

def remap_state_dict(state_dict):
    """
    model.pt uses gamma/beta for LayerNorm; PyTorch uses weight/bias.
    Also remaps transformer.proj.* -> transformer.attn.proj.* because the
    attention output projection is stored at the transformer level in the
    checkpoint but lives inside LIMUBertAttention in our module.
    """
    new_sd = {}
    for k, v in state_dict.items():
        k = k.replace('.gamma', '.weight').replace('.beta', '.bias')
        new_sd[k] = v
    return new_sd


def load_encoder(model_pt_path, device, data_in_dim=None):
    raw_sd           = torch.load(model_pt_path, map_location='cpu', weights_only=False)
    detected_in_dim  = raw_sd['transformer.embed.lin.weight'].shape[1]  # e.g. 6
    detected_seq_len = raw_sd['transformer.embed.pos_embed.weight'].shape[0]  # 120
    hidden           = raw_sd['transformer.embed.lin.weight'].shape[0]  # 72

    # If data has more channels than model expects (e.g. 9 vs 6),
    # we build the encoder with the data's in_dim but only load
    # the first detected_in_dim columns of the embedding weight.
    use_in_dim = data_in_dim if data_in_dim is not None else detected_in_dim
    print(f"[INFO] Checkpoint in_dim={detected_in_dim}, using in_dim={use_in_dim}, hidden={hidden}, seq_len={detected_seq_len}")

    encoder = LIMUBertEncoder(hidden=hidden, seq_len=detected_seq_len, in_dim=use_in_dim)
    sd = remap_state_dict(raw_sd)

    # If in_dim mismatch, expand the embedding weight with zeros for extra channels
    if use_in_dim != detected_in_dim:
        orig_w = sd['transformer.embed.lin.weight']   # (72, 6)
        extra  = torch.zeros(hidden, use_in_dim - detected_in_dim)
        sd['transformer.embed.lin.weight'] = torch.cat([orig_w, extra], dim=1)
        orig_dec = sd['decoder.weight']               # (6, 72)
        extra_dec = torch.zeros(use_in_dim - detected_in_dim, hidden)
        sd['decoder.weight'] = torch.cat([orig_dec, extra_dec], dim=0)
        orig_dec_b = sd['decoder.bias']
        sd['decoder.bias'] = torch.cat([orig_dec_b, torch.zeros(use_in_dim - detected_in_dim)])

    missing, unexpected = encoder.load_state_dict(sd, strict=False)
    if missing:
        print(f"[WARN] Missing keys: {missing}")
    if unexpected:
        print(f"[WARN] Unexpected keys: {unexpected}")
    encoder.eval()
    encoder.to(device)
    print(f"[INFO] LIMU-BERT encoder loaded from {model_pt_path}")
    return encoder, detected_in_dim


# ---------------------------------------------------------------------------
# Embedding generation
# ---------------------------------------------------------------------------

def generate_for_dataset(encoder, model_in_dim, data_path, out_dir, device, batch_size=256):
    """
    Loads data_20_120.npy, encodes every sample, saves sensor_image_{idx}.npy.
    If data has more channels than model_in_dim, slices to first model_in_dim channels.
    """
    os.makedirs(out_dir, exist_ok=True)

    data = np.load(data_path)   # (N, 120, C)
    N    = len(data)
    C    = data.shape[-1]
    if C != model_in_dim:
        print(f"[INFO] Data has {C} channels, slicing to first {model_in_dim} (acc+gyro)")
        data = data[:, :, :model_in_dim]

    print(f"[INFO] Generating embeddings for {N} samples → {out_dir}")

    already_done = len([f for f in os.listdir(out_dir) if f.endswith('.npy')])
    if already_done == N:
        print(f"[INFO] All {N} embeddings already exist, skipping.")
        return

    with torch.no_grad():
        for start in tqdm(range(0, N, batch_size)):
            batch_np   = data[start:start + batch_size].astype(np.float32)
            batch_t    = torch.from_numpy(batch_np).to(device)  # (B, 120, model_in_dim)
            embeddings = encoder(batch_t)                        # (B, 72)
            embeddings = embeddings.cpu().numpy()

            for i, emb in enumerate(embeddings):
                idx      = start + i
                out_path = os.path.join(out_dir, f"sensor_image_{idx}.npy")
                np.save(out_path, emb)  # shape (72,)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate LIMU-BERT embeddings for LLaSA eval")
    parser.add_argument('--mode', type=str, default='baseline',
                        choices=['baseline', 'reconstructed'],
                        help='baseline = raw data, reconstructed = post-compression data')
    parser.add_argument('--model_pt', type=str, default='./llasa/model.pt',
                        help='Path to LIMU-BERT model.pt')
    parser.add_argument('--data_dir', type=str, default='./llasa/data',
                        help='Root dir containing {dataset}/data_20_120.npy')
    parser.add_argument('--recon_dir', type=str, default='./llasa/recon_data',
                        help='Root dir for reconstructed data (used when mode=reconstructed)')
    parser.add_argument('--out_dir', type=str, default='./llasa/embeddings',
                        help='Root output dir for embeddings')
    parser.add_argument('--datasets', type=str, nargs='+',
                        default=['hhar', 'motion', 'shoaib', 'uci'],
                        help='Datasets to process')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    print(f"[INFO] Mode: {args.mode} | Device: {args.device}")
    encoder, model_in_dim = load_encoder(args.model_pt, args.device)

    for dataset in args.datasets:
        if args.mode == 'baseline':
            data_path = os.path.join(args.data_dir, dataset, 'data_20_120.npy')
        else:
            data_path = os.path.join(args.recon_dir, dataset, 'data_20_120.npy')

        out_dir = os.path.join(args.out_dir, args.mode, dataset)

        if not os.path.exists(data_path):
            print(f"[WARN] Data not found, skipping: {data_path}")
            continue

        generate_for_dataset(encoder, model_in_dim, data_path, out_dir, args.device, args.batch_size)

    print("[INFO] Done. Embeddings saved to:", args.out_dir)


if __name__ == '__main__':
    main() 