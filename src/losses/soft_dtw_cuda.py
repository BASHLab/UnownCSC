# src/soft_dtw_cuda.py
import math
import torch
import torch.cuda
from numba import jit, cuda
from torch.autograd import Function

@cuda.jit
def compute_softdtw_cuda(D, gamma, bandwidth, max_i, max_j, n_passes, R):
    b = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    I = tid
    inv_gamma = 1.0 / gamma

    for p in range(n_passes):
        J = max(0, min(p - tid, max_j - 1))
        i = I + 1
        j = J + 1

        if I + J == p and (I < max_i and J < max_j):
            if not (abs(i - j) > bandwidth > 0):
                r0 = -R[b, i - 1, j - 1] * inv_gamma
                r1 = -R[b, i - 1, j] * inv_gamma
                r2 = -R[b, i, j - 1] * inv_gamma
                rmax = max(max(r0, r1), r2)
                rsum = math.exp(r0 - rmax) + math.exp(r1 - rmax) + math.exp(r2 - rmax)
                softmin = -gamma * (math.log(rsum) + rmax)
                R[b, i, j] = D[b, i - 1, j - 1] + softmin
        cuda.syncthreads()

@cuda.jit
def compute_softdtw_backward_cuda(D, R, inv_gamma, bandwidth, max_i, max_j, n_passes, E):
    k = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    I = tid

    for p in range(n_passes):
        rev_p = n_passes - p - 1
        J = max(0, min(rev_p - tid, max_j - 1))
        i = I + 1
        j = J + 1

        if I + J == rev_p and (I < max_i and J < max_j):
            if math.isinf(R[k, i, j]):
                R[k, i, j] = -math.inf
            if not (abs(i - j) > bandwidth > 0):
                a = math.exp((R[k, i + 1, j] - R[k, i, j] - D[k, i + 1, j]) * inv_gamma)
                b = math.exp((R[k, i, j + 1] - R[k, i, j] - D[k, i, j + 1]) * inv_gamma)
                c = math.exp((R[k, i + 1, j + 1] - R[k, i, j] - D[k, i + 1, j + 1]) * inv_gamma)
                E[k, i, j] = E[k, i + 1, j] * a + E[k, i, j + 1] * b + E[k, i + 1, j + 1] * c
        cuda.syncthreads()

class _SoftDTWCUDA(Function):
    @staticmethod
    def forward(ctx, D, gamma, bandwidth):
        dev = D.device
        dtype = D.dtype
        # 🔁 FIX: use torch.tensor to avoid warning
        gamma = torch.tensor([gamma], dtype=dtype, device=dev)
        bandwidth = torch.tensor([bandwidth], dtype=dtype, device=dev)

        B, N, M = D.shape
        threads_per_block = max(N, M)
        n_passes = 2 * threads_per_block - 1

        R = torch.ones((B, N + 2, M + 2), device=dev, dtype=dtype) * math.inf
        R[:, 0, 0] = 0

        compute_softdtw_cuda[B, threads_per_block](
            cuda.as_cuda_array(D.detach()),
            gamma.item(), bandwidth.item(), N, M, n_passes,
            cuda.as_cuda_array(R)
        )
        ctx.save_for_backward(D, R.clone(), gamma, bandwidth)
        return R[:, -2, -2]

    @staticmethod
    def backward(ctx, grad_output):
        dev = grad_output.device
        dtype = grad_output.dtype
        D, R, gamma, bandwidth = ctx.saved_tensors
        B, N, M = D.shape
        threads_per_block = max(N, M)
        n_passes = 2 * threads_per_block - 1

        D_ = torch.zeros((B, N + 2, M + 2), dtype=dtype, device=dev)
        D_[:, 1:N + 1, 1:M + 1] = D

        R[:, :, -1] = -math.inf
        R[:, -1, :] = -math.inf
        R[:, -1, -1] = R[:, -2, -2]

        E = torch.zeros((B, N + 2, M + 2), dtype=dtype, device=dev)
        E[:, -1, -1] = 1

        compute_softdtw_backward_cuda[B, threads_per_block](
            cuda.as_cuda_array(D_),
            cuda.as_cuda_array(R),
            1.0 / gamma.item(), bandwidth.item(), N, M, n_passes,
            cuda.as_cuda_array(E)
        )
        E = E[:, 1:N + 1, 1:M + 1]
        return grad_output.view(-1, 1, 1).expand_as(E) * E, None, None

class SoftDTW(torch.nn.Module):
    def __init__(self, use_cuda, gamma=1.0, normalize=False, bandwidth=None, dist_func=None):
        super(SoftDTW, self).__init__()
        self.normalize = normalize
        self.gamma = gamma
        self.bandwidth = 0 if bandwidth is None else float(bandwidth)
        self.use_cuda = use_cuda
        self.dist_func = dist_func if dist_func is not None else SoftDTW._euclidean_dist_func

    @staticmethod
    def _euclidean_dist_func(x, y):
        n = x.size(1)
        m = y.size(1)
        d = x.size(2)
        x = x.unsqueeze(2).expand(-1, n, m, d)
        y = y.unsqueeze(1).expand(-1, n, m, d)
        return torch.pow(x - y, 2).sum(3)

    def forward(self, X, Y):
        if self.use_cuda:
            func_dtw = _SoftDTWCUDA.apply
        else:
            raise NotImplementedError("CPU mode not implemented.")

        if self.normalize:
            x = torch.cat([X, X, Y])
            y = torch.cat([Y, X, Y])
            D = self.dist_func(x, y)
            out = func_dtw(D, self.gamma, self.bandwidth)
            out_xy, out_xx, out_yy = torch.split(out, X.shape[0])
            return out_xy - 1 / 2 * (out_xx + out_yy)
        else:
            D_xy = self.dist_func(X, Y)
            return func_dtw(D_xy, self.gamma, self.bandwidth)