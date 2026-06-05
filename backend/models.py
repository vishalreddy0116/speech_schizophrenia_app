import os
import numpy as np
import logging

logger = logging.getLogger("Models")

# Try to import PyTorch and Transformers
TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    from transformers import WavLMModel, AutoFeatureExtractor
    TORCH_AVAILABLE = True
except ImportError:
    logger.warning("PyTorch or Transformers not installed. Running in Simulation Mode.")
    class nn:
        class Module: pass

# Set device
device = "cpu"
if TORCH_AVAILABLE:
    device = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────────────────────────────
# 1. VECTOR QUANTIZER (for VQ-VAE)
# ─────────────────────────────────────────────────────────────
if TORCH_AVAILABLE:
    class VectorQuantizer(nn.Module):
        def __init__(self, num_embeddings=512, embedding_dim=128, commitment_cost=0.25):
            super().__init__()
            self.embedding_dim = embedding_dim
            self.num_embeddings = num_embeddings
            self.commitment_cost = commitment_cost
            self.embedding = nn.Embedding(num_embeddings, embedding_dim)
            self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

        def forward(self, z_e):
            # Calculate distances between encoder outputs and embedding vectors
            distances = (
                torch.sum(z_e ** 2, dim=1, keepdim=True)
                + torch.sum(self.embedding.weight ** 2, dim=1)
                - 2 * torch.matmul(z_e, self.embedding.weight.t())
            )
            # Find nearest embedding indices
            indices = torch.argmin(distances, dim=1).unsqueeze(1)
            one_hot = torch.zeros(indices.shape[0], self.num_embeddings, device=z_e.device)
            one_hot.scatter_(1, indices, 1)

            # Quantize
            quantized = torch.matmul(one_hot, self.embedding.weight)

            # Commitment loss
            e_latent_loss = nn.functional.mse_loss(quantized.detach(), z_e)
            q_latent_loss = nn.functional.mse_loss(quantized, z_e.detach())
            vq_loss = q_latent_loss + self.commitment_cost * e_latent_loss

            # Straight-through estimator
            quantized = z_e + (quantized - z_e).detach()
            return vq_loss, quantized
else:
    class VectorQuantizer:
        pass

# ─────────────────────────────────────────────────────────────
# 2. VQ-VAE MODEL
# ─────────────────────────────────────────────────────────────
if TORCH_AVAILABLE:
    class VQVAE(nn.Module):
        def __init__(self, input_dim=4, hidden_dim=128, num_embeddings=512):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, hidden_dim),
            )
            self.vq = VectorQuantizer(num_embeddings, hidden_dim)
            self.decoder = nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.ReLU(),
                nn.Linear(64, input_dim),
            )

        def forward(self, x, return_recon=False):
            z_e = self.encoder(x)
            vq_loss, quantized = self.vq(z_e)
            if return_recon:
                x_recon = self.decoder(quantized)
                return quantized, vq_loss, x_recon
            return quantized, vq_loss
else:
    class VQVAE:
        pass

# ─────────────────────────────────────────────────────────────
# 3. SDVAE (Speech-Driven VAE) MODEL
# ─────────────────────────────────────────────────────────────
if TORCH_AVAILABLE:
    class SDVAE(nn.Module):
        """Speech-Driven Variational Autoencoder (SDVAE)"""
        def __init__(self, input_dim=4, latent_dim=128):
            super().__init__()
            self.encoder_fc = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
            )
            self.fc_mu = nn.Linear(64, latent_dim)
            self.fc_var = nn.Linear(64, latent_dim)
            
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 64),
                nn.ReLU(),
                nn.Linear(64, input_dim),
            )

        def encode(self, x):
            h = self.encoder_fc(x)
            return self.fc_mu(h), self.fc_var(h)

        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std

        def forward(self, x):
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            x_recon = self.decoder(z)
            return z, mu, logvar, x_recon
else:
    class SDVAE:
        pass

# ─────────────────────────────────────────────────────────────
# 4. ATTENTION FUSION & MLP REGRESSION HEAD
# ─────────────────────────────────────────────────────────────
if TORCH_AVAILABLE:
    class FusionModel(nn.Module):
        def __init__(self, wavlm_dim=768, latent_dim=128):
            super().__init__()
            # Multi-Head Attention Fusion
            self.mha = nn.MultiheadAttention(
                embed_dim=wavlm_dim, 
                num_heads=8, 
                batch_first=True
            )
            
            # Dimension matching projection for latent reps to match WavLM dimension
            self.proj_latent = nn.Linear(latent_dim, wavlm_dim)
            
            # MLP Regression Head
            # Input: Fused vector (WavLM 768 + Fused latent 768 = 1536)
            self.mlp = nn.Sequential(
                nn.Linear(wavlm_dim * 2, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 1),
            )

        def forward(self, wavlm_embeddings, latent_reps):
            # wavlm_embeddings: (B, seq_len, 768)
            # latent_reps: (B, 128)
            
            # Project latent representations to match WavLM embed_dim
            # (B, 128) -> (B, 768) -> (B, 1, 768)
            latent_proj = self.proj_latent(latent_reps).unsqueeze(1)
            
            # Perform multi-head attention: query=latent_proj, key=wavlm, value=wavlm
            # Focuses attention on WavLM context matching the latent articulatory representations
            attn_out, attn_weights = self.mha(latent_proj, wavlm_embeddings, wavlm_embeddings)
            
            # Mean pool / Squeeze attention output: (B, 1, 768) -> (B, 768)
            fused_attn = attn_out.squeeze(1)
            
            # Pool WavLM across time: (B, seq_len, 768) -> (B, 768)
            wavlm_pooled = wavlm_embeddings.mean(dim=1)
            
            # Concatenate Attention-fused features and pooled WavLM features: (B, 1536)
            fused_vector = torch.cat([fused_attn, wavlm_pooled], dim=1)
            
            # MLP Regression Head
            ssi_score = self.mlp(fused_vector)
            return ssi_score, attn_weights
else:
    class FusionModel:
        pass

# ─────────────────────────────────────────────────────────────
# 5. RUNTIME MODEL MANAGER & WORKSPACE FALLBACKS
# ─────────────────────────────────────────────────────────────
class ModelManager:
    def __init__(self):
        self.is_simulation = not TORCH_AVAILABLE
        self.vqvae = None
        self.sdvae = None
        self.wavlm = None
        self.wavlm_extractor = None
        self.fusion_model = None
        
        if TORCH_AVAILABLE:
            try:
                self._init_models()
            except Exception as e:
                logger.error(f"Error loading models: {e}. Falling back to simulation mode.")
                self.is_simulation = True

    def _init_models(self):
        # 1. Load VQ-VAE
        self.vqvae = VQVAE(input_dim=4, hidden_dim=128).to(device)
        self.vqvae.eval()
        
        # 2. Load SDVAE
        self.sdvae = SDVAE(input_dim=4, latent_dim=128).to(device)
        self.sdvae.eval()
        
        # 3. Load WavLM Transformer
        # To avoid blocking, we try to load WavLM. If it fails (e.g. no internet/memory), we catch it.
        try:
            logger.warning("Railway deployment mode - WavLM disabled")
            self.wavlm = None
            self.wavlm_extractor = None
            #logger.info("Attempting to load WavLM model from HuggingFace...")
            #self.wavlm_extractor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base")
            #self.wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base").to(device)
            #self.wavlm.eval()
            #logger.info("WavLM loaded successfully.")
        except Exception as e:
            logger.warning(f"Could not download WavLM transformer: {e}. Will simulate WavLM branch.")
            self.wavlm = None

        # 4. Load Fusion Model
        self.fusion_model = FusionModel(wavlm_dim=768, latent_dim=128).to(device)
        self.fusion_model.eval()

    def run_latent_reps(self, articulatory_features):
        """
        Runs VQ-VAE and SDVAE on 4D articulatory features.
        Returns: vq_z (128D list), sdvae_z (128D list)
        """
        if self.is_simulation:
            # Simulate 128D representations based on input characteristics
            np.random.seed(int(np.sum(articulatory_features) * 1000) % 12345)
            vq_z = (np.random.randn(128) * 0.1 + articulatory_features.mean()).tolist()
            sdvae_z = (np.random.randn(128) * 0.1 + (1.0 - articulatory_features.mean())).tolist()
            return vq_z, sdvae_z

        # Torch mode
        try:
            x_tensor = torch.tensor(articulatory_features, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                vq_z, _ = self.vqvae(x_tensor)
                sdvae_z, _, _, _ = self.sdvae(x_tensor)
                
            return vq_z.squeeze(0).cpu().numpy().tolist(), sdvae_z.squeeze(0).cpu().numpy().tolist()
        except Exception as e:
            logger.error(f"Error in Torch Latent reps: {e}. Falling back to simulation.")
            return (np.random.randn(128) * 0.05).tolist(), (np.random.randn(128) * 0.05).tolist()

    def run_wavlm(self, audio_segment):
        """
        Extracts sequence embeddings using WavLM model.
        Returns: sequence of 768D arrays
        """
        if self.is_simulation or self.wavlm is None:
            # Simulate WavLM sequence embeddings.
            # Typical segment has ~149 frames of size 768
            seq_len = 150
            np.random.seed(int(np.sum(audio_segment[:100]) * 1000) % 54321)
            sim_embed = np.random.randn(1, seq_len, 768) * 0.02
            return sim_embed.tolist()

        # Torch mode with active WavLM
        try:
            inputs = self.wavlm_extractor(audio_segment, sampling_rate=16000, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.wavlm(**inputs)
            return outputs.last_hidden_state.cpu().numpy().tolist()
        except Exception as e:
            logger.error(f"Error in WavLM extraction: {e}. Simulating embeddings.")
            return (np.random.randn(1, 150, 768) * 0.02).tolist()

    def run_fusion_and_prediction(self, wavlm_embed, latent_rep):
        """
        Fuses WavLM features and VQ-VAE/SDVAE representations and predicts SSI.
        """
        if self.is_simulation:
            # Calculate mock SSI score based on input feature stats
            # SSI will be bounded in 0-1
            np.random.seed(int(np.sum(latent_rep) * 1000) % 9999)
            base_score = float(np.clip(0.3 + np.mean(latent_rep) * 0.5 + np.random.randn() * 0.1, 0.0, 1.0))
            # Mock attention weights (1, 1, 150)
            attn_weights = np.abs(np.random.randn(1, 1, 150))
            attn_weights = (attn_weights / attn_weights.sum()).tolist()
            return base_score, attn_weights

        # Torch mode
        try:
            wavlm_tensor = torch.tensor(wavlm_embed, dtype=torch.float32).to(device)
            latent_tensor = torch.tensor(latent_rep, dtype=torch.float32).unsqueeze(0).to(device)
            
            with torch.no_grad():
                ssi_pred, attn_weights = self.fusion_model(wavlm_tensor, latent_tensor)
                
            score = float(torch.clamp(ssi_pred, 0.0, 1.0).item())
            return score, attn_weights.cpu().numpy().tolist()
        except Exception as e:
            logger.error(f"Error in Fusion/MLP prediction: {e}. Simulating prediction.")
            return 0.5, (np.ones((1, 1, 150)) / 150.0).tolist()

# Instantiate global manager
model_manager = ModelManager()
