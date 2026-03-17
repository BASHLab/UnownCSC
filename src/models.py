import torch
import torch.nn as nn
import torch.nn.functional as F

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
    