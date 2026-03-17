# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from .vq_models import VectorQuantizerEMA, FSQBottleneck

# class ShapeletDictionary(nn.Module):
#     def __init__(self, n_shapelets, n_channels, shapelet_len):
#         super().__init__()
#         self.shapelets = nn.Parameter(torch.randn(n_shapelets, n_channels, shapelet_len)*0.2)
#     def forward(self):
#         norm = self.shapelets.norm(dim=-1, keepdim=True).clamp_min(1e-8)
#         return self.shapelets / norm
#     def normalize_parameters(self):
#         with torch.no_grad():
#             norm = self.shapelets.norm(dim=-1, keepdim=True).clamp_min(1e-8)
#             self.shapelets.data = self.shapelets.data / norm

# class ShapeletDecoder(nn.Module):
#     def __init__(self, dictionary: ShapeletDictionary):
#         super().__init__()
#         self.dictionary = dictionary
#         self.n_shapelets, self.n_channels, self.shapelet_len = dictionary.shapelets.shape

#     def forward(self, z_q, stride, output_length):
#         shapelets = self.dictionary()
#         recon = F.conv_transpose1d(
#             z_q, shapelets,
#             stride=stride,
#             padding=self.shapelet_len // 2,
#             output_padding=0
#         )
#         if recon.shape[-1] > output_length:
#             recon = recon[..., :output_length]
#         elif recon.shape[-1] < output_length:
#             pad = output_length - recon.shape[-1]
#             recon = F.pad(recon, (0, pad))
#         return recon

# class ModalityStemEncoder(nn.Module):
#     def __init__(self, cfg, n_atoms, atom_len):
#         super().__init__()
#         self.stems = nn.ModuleDict()
#         self.scales = nn.ParameterDict()
        
#         # Dynamically create an independent stem for each physical sensor
#         for mod in cfg.modalities:
#             in_c = cfg.variates[mod]
#             self.stems[mod] = nn.Sequential(
#                 nn.Conv1d(in_c, 64, kernel_size=atom_len, stride=atom_len // 2, padding=atom_len // 2),
#                 nn.ReLU(),
#                 nn.Conv1d(64, n_atoms, kernel_size=1)
#             )
#             self.scales[mod] = nn.Parameter(torch.ones(1) * 1.0)

#     def forward(self, x_dict):
#         # Returns a dict of latent representations, cleanly isolating modalities
#         return {mod: self.stems[mod](tensor) * self.scales[mod] 
#                 for mod, tensor in x_dict.items()}

# # ----------------------------------------------------------------------
# # 4. MULTISCALE CSC 
# # ----------------------------------------------------------------------
# class MultiscaleShapeletCSC(nn.Module):
#     def __init__(self, cfg, atom_len=64, vocab_size=256, n_atoms=128, 
#                  scales=[4, 2, 1], quantizer="fsq", commitment_cost=0.25, 
#                  sparsity_weight=0.01, fsq_levels=[8, 5, 5, 5]):
#         super().__init__()
#         self.cfg = cfg
#         self.scales = sorted(scales, reverse=True)
#         self.atom_len = atom_len
#         self.n_atoms = n_atoms
#         self.sparsity_weight = sparsity_weight

#         # 1. Independent Stems
#         self.encoder = ModalityStemEncoder(cfg, n_atoms, atom_len)

#         # 2. Shared Latent Dictionary
#         self.dictionary = ShapeletDictionary(n_atoms, n_atoms, atom_len)
#         self.decoder = ShapeletDecoder(self.dictionary)

#         # 3. Modality Specific Decoders (Latent geometry -> Physical units)
#         self.decoders = nn.ModuleDict({
#             mod: nn.ConvTranspose1d(n_atoms, cfg.variates[mod], 
#                                     kernel_size=atom_len, 
#                                     stride=atom_len // 2, 
#                                     padding=atom_len // 2)
#             for mod in cfg.modalities
#         })
#         for mod in cfg.modalities:
#             nn.init.normal_(self.decoders[mod].weight, std=0.01)
#             if self.decoders[mod].bias is not None:
#                 nn.init.constant_(self.decoders[mod].bias, 0.0)

#         # 4. The Quantization Bottleneck
#         if quantizer == 'ema':
#             self.quantizer = VectorQuantizerEMA(vocab_size, n_atoms, commitment_cost=commitment_cost)
#         elif quantizer == 'fsq':
#             self.quantizer = FSQBottleneck(n_atoms, fsq_levels)
#         else:
#             self.quantizer = None

#     def normalize_dictionary(self):
#         self.dictionary.normalize_parameters()

#     def forward(self, x_dict):
#         # 1. Map physical sensors to latent activations
#         z_raw_dict = self.encoder(x_dict)
        
#         recon_dict = {mod: torch.zeros_like(tensor) for mod, tensor in x_dict.items()}
#         total_vq_loss = 0.0
#         total_l1_loss = 0.0

#         # 2. Process each modality through the universal shared dictionary
#         for mod, z_raw in z_raw_dict.items():
#             current_residual = z_raw
#             total_latent_recon = torch.zeros_like(z_raw)

#             for scale in self.scales:
#                 if scale > 1:
#                     z_scaled = F.avg_pool1d(current_residual, kernel_size=scale, stride=scale, ceil_mode=True)
#                 else:
#                     z_scaled = current_residual

#                 if self.quantizer is not None:
#                     z_q, vq_loss, _ = self.quantizer(z_scaled)
#                     total_vq_loss += vq_loss
#                 else:
#                     z_q = z_scaled

#                 l1_loss = torch.mean(torch.abs(z_q))
#                 total_l1_loss += l1_loss

#                 # Reconstruct latents via shared dictionary
#                 stride = scale  # Keeps the convolution locked to the latent dimension
#                 rec_latent = self.decoder(z_q, stride=stride, output_length=z_raw.shape[-1])
                
#                 total_latent_recon = total_latent_recon + rec_latent
#                 current_residual = z_raw - total_latent_recon

#             # 3. Map final latent reconstruction back to physical sensor units
#             recon_dict[mod] = self.decoders[mod](total_latent_recon)

#         return recon_dict, total_vq_loss, total_l1_loss * self.sparsity_weight

#     def reconstruct_at_scale(self, x_dict, target_scale):
#         """
#         Runs the exact residual forward pass, but ONLY applies the latent 
#         reconstruction from the specified target_scale before decoding to physical units.
#         """
#         # 1. Map physical sensors to latent activations
#         z_raw_dict = self.encoder(x_dict)
        
#         # We will store the final isolated physical reconstructions here
#         isolated_recon_dict = {}

#         # 2. Process each modality through the universal shared dictionary
#         for mod, z_raw in z_raw_dict.items():
#             current_residual = z_raw
#             total_latent_recon = torch.zeros_like(z_raw)
            
#             # This variable will hold ONLY the latent reconstruction from our target scale
#             target_latent_recon = torch.zeros_like(z_raw)

#             for scale in self.scales:
#                 # Same pooling logic as forward pass
#                 if scale > 1:
#                     z_scaled = F.avg_pool1d(current_residual, kernel_size=scale, stride=scale, ceil_mode=True)
#                 else:
#                     z_scaled = current_residual

#                 # Same quantization logic
#                 if self.quantizer is not None:
#                     z_q, _, _ = self.quantizer(z_scaled)
#                 else:
#                     z_q = z_scaled

#                 # Same latent reconstruction via shared dictionary
#                 rec_latent = self.decoder(z_q, stride=scale, output_length=z_raw.shape[-1])
                
#                 # Update the residual loop for the NEXT scale so the math stays mathematically identical
#                 total_latent_recon = total_latent_recon + rec_latent
#                 current_residual = z_raw - total_latent_recon
                
#                 # ---> THE ISOLATION TRICK <---
#                 # If this is the scale we want to visualize, we grab its latent representation
#                 if scale == target_scale:
#                     target_latent_recon = rec_latent

#             # 3. Map ONLY the target scale's latent reconstruction back to physical sensor units
#             isolated_recon_dict[mod] = self.decoders[mod](target_latent_recon)

#         return isolated_recon_dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from .vq_models import VectorQuantizerEMA, FSQBottleneck

class ShapeletDictionary(nn.Module):
    def __init__(self, n_shapelets, n_channels, shapelet_len):
        super().__init__()
        self.shapelets = nn.Parameter(torch.randn(n_shapelets, n_channels, shapelet_len)*0.2)
    def forward(self):
        norm = self.shapelets.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return self.shapelets / norm
    def normalize_parameters(self):
        with torch.no_grad():
            norm = self.shapelets.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            self.shapelets.data = self.shapelets.data / norm

class ShapeletDecoder(nn.Module):
    def __init__(self, dictionary: ShapeletDictionary):
        super().__init__()
        self.dictionary = dictionary
        self.n_shapelets, self.n_channels, self.shapelet_len = dictionary.shapelets.shape

    def forward(self, z_q, stride, output_length):
        shapelets = self.dictionary()
        
        # 1. Zero-parameter nearest neighbor upsampling (Prevents Checkerboard)
        if stride > 1:
            z_q_up = F.interpolate(z_q, scale_factor=float(stride), mode='nearest')
        else:
            z_q_up = z_q
            
        # 2. Standard stride=1 convolution to smooth the interpolated steps
        recon = F.conv1d(z_q_up, shapelets, padding='same')
        
        # 3. Trim or pad to exact output length
        if recon.shape[-1] > output_length:
            recon = recon[..., :output_length]
        elif recon.shape[-1] < output_length:
            pad = output_length - recon.shape[-1]
            recon = F.pad(recon, (0, pad))
        return recon

class ModalityStemEncoder(nn.Module):
    def __init__(self, cfg, n_atoms, atom_len):
        super().__init__()
        self.stems = nn.ModuleDict()
        self.scales = nn.ParameterDict()
        
        for mod in cfg.modalities:
            in_c = cfg.variates[mod]
            self.stems[mod] = nn.Sequential(
                nn.Conv1d(in_c, 64, kernel_size=atom_len, stride=atom_len // 2, padding=atom_len // 2),
                nn.ReLU(),
                nn.Conv1d(64, n_atoms, kernel_size=1)
            )
            self.scales[mod] = nn.Parameter(torch.ones(1) * 1.0)

    def forward(self, x_dict):
        return {mod: self.stems[mod](tensor) * self.scales[mod] 
                for mod, tensor in x_dict.items()}

# ----------------------------------------------------------------------
# 4. MULTISCALE CSC 
# ----------------------------------------------------------------------
class MultiscaleShapeletCSC(nn.Module):
    def __init__(self, cfg, atom_len=64, vocab_size=256, n_atoms=128, 
                 scales=[4, 2, 1], quantizer="fsq", commitment_cost=0.25, 
                 sparsity_weight=0.01, fsq_levels=[8, 5, 5, 5]):
        super().__init__()
        self.cfg = cfg
        self.scales = sorted(scales, reverse=True)
        self.atom_len = atom_len
        self.n_atoms = n_atoms
        self.sparsity_weight = sparsity_weight

        self.encoder = ModalityStemEncoder(cfg, n_atoms, atom_len)
        self.dictionary = ShapeletDictionary(n_atoms, n_atoms, atom_len)
        self.decoder = ShapeletDecoder(self.dictionary)

        # Swapped ConvTranspose1d for Upsample + Conv1d to preserve parameters but fix aliasing
        self.decoders = nn.ModuleDict({
            mod: nn.Sequential(
                nn.Upsample(scale_factor=atom_len // 2, mode='nearest'),
                nn.Conv1d(n_atoms, cfg.variates[mod], kernel_size=atom_len, padding='same')
            )
            for mod in cfg.modalities
        })
        
        # Initialize the Conv1d (index 1 in the Sequential block)
        for mod in cfg.modalities:
            nn.init.normal_(self.decoders[mod][1].weight, std=0.01)
            if self.decoders[mod][1].bias is not None:
                nn.init.constant_(self.decoders[mod][1].bias, 0.0)

        if quantizer == 'ema':
            self.quantizer = VectorQuantizerEMA(vocab_size, n_atoms, commitment_cost=commitment_cost)
        elif quantizer == 'fsq':
            self.quantizer = FSQBottleneck(n_atoms, fsq_levels)
        else:
            self.quantizer = None

    def normalize_dictionary(self):
        self.dictionary.normalize_parameters()

    def forward(self, x_dict):
        # --- RevIN: Dynamic Normalization ---
        means, stdevs, x_norm_dict = {}, {}, {}
        for mod, tensor in x_dict.items():
            means[mod] = torch.mean(tensor, dim=-1, keepdim=True)
            stdevs[mod] = torch.sqrt(torch.var(tensor, dim=-1, keepdim=True, unbiased=False) + 1e-5)
            x_norm_dict[mod] = (tensor - means[mod]) / stdevs[mod]

        # 1. Map normalized sensors to latent activations
        z_raw_dict = self.encoder(x_norm_dict)
        
        recon_dict = {mod: torch.zeros_like(tensor) for mod, tensor in x_dict.items()}
        total_vq_loss = 0.0
        total_l1_loss = 0.0

        # 2. Process through universal shared dictionary
        for mod, z_raw in z_raw_dict.items():
            current_residual = z_raw
            total_latent_recon = torch.zeros_like(z_raw)

            for scale in self.scales:
                if scale > 1:
                    z_scaled = F.avg_pool1d(current_residual, kernel_size=scale, stride=scale, ceil_mode=True)
                else:
                    z_scaled = current_residual

                if self.quantizer is not None:
                    z_q, vq_loss, _ = self.quantizer(z_scaled)
                    total_vq_loss += vq_loss
                else:
                    z_q = z_scaled

                l1_loss = torch.mean(torch.abs(z_q))
                total_l1_loss += l1_loss

                rec_latent = self.decoder(z_q, stride=scale, output_length=z_raw.shape[-1])
                total_latent_recon = total_latent_recon + rec_latent
                current_residual = z_raw - total_latent_recon

            # 3. Decode latents back to 1D
            recon_norm = self.decoders[mod](total_latent_recon)
            
            # ---> THE FIX: Exact Length Truncation <---
            target_len = x_norm_dict[mod].shape[-1]
            if recon_norm.shape[-1] > target_len:
                recon_norm = recon_norm[..., :target_len]
            elif recon_norm.shape[-1] < target_len:
                pad = target_len - recon_norm.shape[-1]
                recon_norm = F.pad(recon_norm, (0, pad))
            
            # --- RevIN: Dynamic Denormalization ---
            recon_dict[mod] = (recon_norm * stdevs[mod]) + means[mod]

        return recon_dict, total_vq_loss, total_l1_loss * self.sparsity_weight

    def reconstruct_at_scale(self, x_dict, target_scale):
        # --- RevIN: Dynamic Normalization ---
        means, stdevs, x_norm_dict = {}, {}, {}
        for mod, tensor in x_dict.items():
            means[mod] = torch.mean(tensor, dim=-1, keepdim=True)
            stdevs[mod] = torch.sqrt(torch.var(tensor, dim=-1, keepdim=True, unbiased=False) + 1e-5)
            x_norm_dict[mod] = (tensor - means[mod]) / stdevs[mod]

        z_raw_dict = self.encoder(x_norm_dict)
        isolated_recon_dict = {}

        for mod, z_raw in z_raw_dict.items():
            current_residual = z_raw
            total_latent_recon = torch.zeros_like(z_raw)
            target_latent_recon = torch.zeros_like(z_raw)

            for scale in self.scales:
                if scale > 1:
                    z_scaled = F.avg_pool1d(current_residual, kernel_size=scale, stride=scale, ceil_mode=True)
                else:
                    z_scaled = current_residual

                if self.quantizer is not None:
                    z_q, _, _ = self.quantizer(z_scaled)
                else:
                    z_q = z_scaled

                rec_latent = self.decoder(z_q, stride=scale, output_length=z_raw.shape[-1])
                
                total_latent_recon = total_latent_recon + rec_latent
                current_residual = z_raw - total_latent_recon
                
                if scale == target_scale:
                    target_latent_recon = rec_latent

            # Decode target latents
            recon_norm = self.decoders[mod](target_latent_recon)
            
            # ---> THE FIX: Exact Length Truncation <---
            target_len = x_norm_dict[mod].shape[-1]
            if recon_norm.shape[-1] > target_len:
                recon_norm = recon_norm[..., :target_len]
            elif recon_norm.shape[-1] < target_len:
                pad = target_len - recon_norm.shape[-1]
                recon_norm = F.pad(recon_norm, (0, pad))
            
            # --- RevIN: Dynamic Denormalization ---
            isolated_recon_dict[mod] = (recon_norm * stdevs[mod]) + means[mod]

        return isolated_recon_dict