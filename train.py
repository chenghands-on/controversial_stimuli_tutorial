import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from transformers import ViTForImageClassification, ViTFeatureExtractor
from torchvision import models, transforms
from diffusers import StableDiffusionPipeline, DiffusionPipeline
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from scipy.stats import spearmanr
from typing import Tuple
import os
from tqdm import tqdm
from torchvision.transforms import ToPILImage
import torchvision.transforms.functional as TF




# ======================
# 1. DataLoader Module
# ======================
class DataLoaderModule:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_models()

    def _load_models(self):
        # Load pre-trained ResNet-50
        self.resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).to(self.device).eval()
        print(self.resnet50)
        # model = torchvision.models.resnet50()
        
        # Load pre-trained Vision Transformer
        vit_model_name = "google/vit-base-patch16-224"
        self.vit = ViTForImageClassification.from_pretrained(vit_model_name).to(self.device).eval()
        print(self.vit)
        # pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-3.5-large-turbo")
        # pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-3.5-medium")
        self.diffusion_pipeline = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1",torch_dtype=torch.float16).to(self.device)
        self.diffusion_pipeline.enable_xformers_memory_efficient_attention()
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode an image using the VAE part of the diffusion model (SD 3.5 large turbo)."""
        latents = self.diffusion_pipeline.vae.encode(image).latent_dist.mean  # Use mean instead of sampling
        latents = latents * 0.13025  # Correct scaling factor for SD 3.5 large turbo
        return latents


    
    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode a latent vector back into an image and normalize it for ResNet."""
        latent = latent / 0.13025
        decoded_output = self.diffusion_pipeline.vae.decode(latent)
        
        # 取出 sample 数据（实际的 Tensor）
        decoded_image = decoded_output.sample
        
        # # 计算 min/max 并标准化到 [0,1]
        # min_val, max_val = decoded_image.min(), decoded_image.max()
        # print(f"Decoded image range before normalization: min={min_val}, max={max_val}")

        # decoded_image = (decoded_image - min_val) / (max_val - min_val)

        # 确保数据类型为 float32（ResNet 期望 float32）
        decoded_image = decoded_image.to(dtype=torch.float32)

        # 确保 3 通道（如果 VAE 不是 3 通道）
        if decoded_image.shape[1] != 3:
            decoded_image = decoded_image[:, :3, :, :]

        transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize([0.5], [0.5])  # Standard normalization
    ])

        # **添加 Resize，缩放到 224x224**
        decoded_image = transform(decoded_image)
        
        return decoded_image

# ==============================
# 2. LatentSpaceOptimizer Module
# ==============================
class LatentSpaceOptimizer:
    def __init__(self, model_evaluator, learning_rate=10, latent_dim=512):
        self.model_evaluator = model_evaluator
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def optimize_latent(self, latent_vector: torch.Tensor, num_steps: int = 100) -> torch.Tensor:
        """Optimize the latent vector to maximize dissimilarity between ResNet-50 and ViT."""
        latent_vector = latent_vector.clone().detach().to(self.device).requires_grad_(True)
        optimizer = torch.optim.SGD([latent_vector], lr=self.learning_rate)

        for step in tqdm(range(num_steps)):
            optimizer.zero_grad()
            # Decode the latent vector to get the image
            generated_image = self.model_evaluator.data_loader.decode_latent(latent_vector)
            utility = self.model_evaluator.compute_utility(generated_image)
            # latent_reg = torch.norm(latent_vector) * 0.01  # Prevent extreme updates
        
        # Compute loss
            loss = -torch.log(utility + 1e-6)*100
  # 让 loss 保持可微分
            loss.backward()
            grad_norm = latent_vector.grad.norm().item()
            print(f"Step {step} - Gradient Norm: {grad_norm}")
            optimizer.step()

        return latent_vector.detach()

# ===========================
# 3. ModelEvaluator Module
# ===========================
class ModelEvaluator:
    def __init__(self, data_loader: DataLoaderModule):
        self.data_loader = data_loader

    def extract_features(self, image: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
        """Extract the last hidden layer features from a given model."""
        with torch.no_grad():
            if isinstance(model, models.ResNet):  # ResNet50
                x = model.conv1(image)
                x = model.bn1(x)
                x = model.relu(x)
                x = model.maxpool(x)
                
                x = model.layer1(x)
                x = model.layer2(x)
                x = model.layer3(x)
                x = model.layer4(x)
                
                x = model.avgpool(x)  # 提取 avgpool 层
                features = torch.flatten(x, 1)  # 变成 (batch_size, 2048)
            
            elif isinstance(model, ViTForImageClassification):  # ViT
                outputs = model(image, output_hidden_states=True)  # 获取隐藏层
                features = outputs.hidden_states[-1][:, 0, :]  # 取最后一层的 CLS token (batch_size, 768)
            
            else:
                raise ValueError("Unsupported model type!")

        return features


    def compute_utility(self, image: torch.Tensor) -> float:
        """Compute the dissimilarity between ResNet-50 and ViT representations."""
        # Extract features from ResNet-50 and ViT
        resnet_features = self.extract_features(image, self.data_loader.resnet50)
        vit_features = self.extract_features(image, self.data_loader.vit)

        # Compute RSA dissimilarity
        rsa_score = self._rsa_metric(resnet_features, vit_features)
        return torch.tensor(rsa_score, dtype=torch.float16, device=image.device, requires_grad=True)

    def _rsa_metric(self, features_model1: torch.Tensor, features_model2: torch.Tensor) -> float:
        """Compute RSA (Representational Similarity Analysis) score."""
        pairwise_dissim1 = torch.cdist(features_model1, features_model1, p=2)
        pairwise_dissim2 = torch.cdist(features_model2, features_model2, p=2)
        flat1, flat2 = pairwise_dissim1.flatten(), pairwise_dissim2.flatten()
        correlation, _ = spearmanr(flat1.cpu().numpy(), flat2.cpu().numpy())
        return correlation

# =============================
# 4. SimilarityMetrics Module
# =============================
class SimilarityMetrics:
    @staticmethod
    def canonical_correlation_analysis(features1: torch.Tensor, features2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply CCA to align features from two models."""
        cca = CCA(n_components=min(features1.shape[1], features2.shape[1]))
        aligned1, aligned2 = cca.fit_transform(features1, features2)
        return aligned1, aligned2

    @staticmethod
    def pca_reduction(latent_vectors: torch.Tensor, components: int = 100) -> torch.Tensor:
        """Reduce dimensionality of latent vectors using PCA."""
        pca = PCA(n_components=components)
        reduced = pca.fit_transform(latent_vectors)
        return torch.tensor(reduced, dtype=torch.float16)

# ============================
# 5. ExperimentRunner Module
# ============================
class ExperimentRunner(pl.LightningModule):
    def __init__(self, data_loader: DataLoaderModule, optimizer: LatentSpaceOptimizer):
        super(ExperimentRunner, self).__init__()
        self.data_loader = data_loader
        self.optimizer = optimizer


    def run_experiment(self, num_trials: int = 5, num_steps: int = 100,batch_size: int = 3):
        for trial in tqdm(range(num_trials)):
            # Generate a random initial latent vector
            batch_id = trial  # 设定 batch ID
            save_dir = f"generated_images/batch_{batch_id}"
            os.makedirs(save_dir, exist_ok=True)
            latent_vector = torch.randn(batch_size, 4, 64, 64, dtype=torch.float16).to(self.device)
            latent_vector = latent_vector * 0.13025  # Apply correct scaling
            latent_vector = latent_vector.to(torch.float16)  # 将输入转换为 FP16
            # Optimize the latent vector
            optimized_latent = self.optimizer.optimize_latent(latent_vector, num_steps=num_steps)
            # Decode and save the final image
            final_image = self.data_loader.decode_latent(optimized_latent)
            # final_image = self.decode_latent(final_image)
            for idx, img in enumerate(final_image):
                img_pil = ToPILImage()(img.cpu())
                img_pil.save(os.path.join(save_dir, f"batch_{batch_id}_img_{idx}.png"))

            print(f"Saved {len(final_image)} images in {save_dir}")

# ===========================
# 6. Main Execution Script
# ===========================
if __name__ == "__main__":
    # Initialize the data loader and models
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    data_loader = DataLoaderModule()
    model_evaluator = ModelEvaluator(data_loader)
    optimizer = LatentSpaceOptimizer(model_evaluator, learning_rate=0.01, latent_dim=512)

    # Run the experiment
    experiment_runner = ExperimentRunner(data_loader, optimizer)
    experiment_runner.run_experiment(num_trials=5, num_steps=100)
