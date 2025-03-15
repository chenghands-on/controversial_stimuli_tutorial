import os
import torch
import numpy as np
from PIL import Image
from models import Resnet50, InceptionV3
from optimize_functions import optimize_controversial_stimuli_with_diffusion_noise

def main():
    # 创建输出目录
    output_dir = "diffusion_noise_test"
    os.makedirs(output_dir, exist_ok=True)
    
    # 移除任何存在的标记文件
    for f in os.listdir(output_dir):
        if f.endswith(".in_process_flag"):
            os.remove(os.path.join(output_dir, f))
    
    # 加载模型
    print("Loading models...")
    model_1 = Resnet50()
    model_2 = InceptionV3()
    
    # 确保模型正确加载并放置在GPU上
    model_1.load()
    model_2.load()
    
    # 打印可用的类别
    print(f"Model 1 ({model_1.model_name}) has {len(model_1.idx_to_class_name)} classes")
    print(f"Model 2 ({model_2.model_name}) has {len(model_2.idx_to_class_name)} classes")
    
    # 设置类别名称 - 使用ImageNet中实际存在的类别名称
    class_1 = "Persian cat"  # ImageNet中猫的一个类别
    class_2 = "Weimaraner"   # ImageNet中狗的一个类别
    
    print(f"Using classes: '{class_1}' (cat) and '{class_2}' (dog)")
    
    # 验证类别存在于两个模型中
    for model, name in [(model_1, "Model 1"), (model_2, "Model 2")]:
        for cls in [class_1, class_2]:
            if cls not in model.class_name_to_idx:
                raise ValueError(f"{cls} not found in {name} classes")
    
    print(f"Optimizing for classes: {class_1} vs {class_2}")
    
    # 运行噪声空间优化
    print("Starting noise-space optimization...")
    _, PIL_list, controversiality_score = optimize_controversial_stimuli_with_diffusion_noise(
        model_1=model_1,
        model_2=model_2,
        class_1=class_1,
        class_2=class_2,
        random_seed=1,
        max_steps=300,
        num_inference_steps=10,  # 进一步降低步数以减少内存使用
        verbose=True
    )
    
    # 保存图像
    img = PIL_list[0]
    output_path = os.path.join(output_dir, f"Resnet50-{class_1}_vs_InceptionV3-{class_2}_noise_optimized.png")
    img.save(output_path)
    print(f"Saved image to {output_path}")
    print(f"Adversarial score: {controversiality_score[0]:.4f}")

if __name__ == "__main__":
    main() 