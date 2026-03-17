import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------
# 1. VECTOR QUANTIZERS (EMA & FSQ)
# ----------------------------------------------------------------------
class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.5, decay=0.99):
        super().__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._commitment_cost = commitment_cost
        self._decay = decay

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1.0, 1.0)

        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('_ema_w', torch.empty(num_embeddings, embedding_dim))
        self._ema_w.data.uniform_(-1.0, 1.0)

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1).contiguous()
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self._embedding_dim)

        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            + torch.sum(self._embedding.weight**2, dim=1)
            - 2 * torch.matmul(flat_input, self._embedding.weight.t())
        )

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings,
                                device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        if self.training:
            self._ema_cluster_size.mul_(self._decay).add_(
                torch.sum(encodings, 0), alpha=1 - self._decay
            )
            n = torch.sum(self._ema_cluster_size)
            self._ema_cluster_size.copy_(
                (self._ema_cluster_size + 1e-5) / (n + self._num_embeddings * 1e-5) * n
            )

            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w.mul_(self._decay).add_(dw, alpha=1 - self._decay)

            self._embedding.weight.data.copy_(
                self._ema_w / self._ema_cluster_size.unsqueeze(1)
            )

        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        return quantized.permute(0, 2, 1).contiguous(), loss, encoding_indices


class VectorQuantizerFSQ(nn.Module):
    """PyTorch implementation of Finite Scalar Quantization (ICLR 2024)"""
    def __init__(self, levels: list, eps: float = 1e-3):
        super().__init__()
        self.levels = levels
        self.eps = eps
        self.register_buffer('_levels', torch.tensor(levels, dtype=torch.float32))
        
        basis = [1]
        for l in levels[:-1]:
            basis.append(basis[-1] * l)
        self.register_buffer('_basis', torch.tensor(basis, dtype=torch.int32))

    def bound(self, z: torch.Tensor) -> torch.Tensor:
        half_l = (self._levels - 1) * (1 - self.eps) / 2
        offset = torch.where(self._levels % 2 == 1, 0.0, 0.5)
        shift = torch.tan(offset / half_l)
        return torch.tanh(z + shift) * half_l - offset

    def forward(self, z):
        # Transpose to (B, T, C) for clean broadcasting against _levels
        z_t = z.permute(0, 2, 1).contiguous()
        
        bound_z = self.bound(z_t)
        zhat = torch.round(bound_z)
        
        # Straight-Through Estimator (STE)
        quantized = bound_z + (zhat - bound_z).detach()
        
        # Renormalize to [-1, 1]
        half_width = self._levels // 2
        quantized = quantized / half_width
        
        # Back to (B, C, T)
        quantized = quantized.permute(0, 2, 1).contiguous()
        
        # FSQ requires no commitment loss!
        loss = torch.tensor(0.0, device=z.device)
        return quantized, loss, None


class FSQBottleneck(nn.Module):
    """Wraps FSQ with linear layers to bridge n_atoms to FSQ grid levels"""
    def __init__(self, in_channels, levels):
        super().__init__()
        out_channels = len(levels)
        self.project_in = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.fsq = VectorQuantizerFSQ(levels)
        self.project_out = nn.Conv1d(out_channels, in_channels, kernel_size=1)

    def forward(self, z):
        z_proj = self.project_in(z)
        z_q, loss, indices = self.fsq(z_proj)
        z_out = self.project_out(z_q)
        return z_out, loss, indices