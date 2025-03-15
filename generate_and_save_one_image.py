#!/usr/bin/env python
import os
import torch
import argparse
from PIL import Image
import pathlib

import models
from optimize import optimize_controversial_stimuli_with_diffusion

# 直接设置所有参数，不使用命令行解析
model1_name = "Resnet50"
model2_name = "InceptionV3"
class1 = "Egyptian cat"  # 改为埃及猫
class2 = "golden retriever"  # 改为金毛猎犬
output_dir = "diffusion_results"
seed = 101
max_steps = 40  # 减少步骤数以加快运行速度

# 创建输出目录
pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

# 加载模型
print(f"加载模型 {model1_name} 和 {model2_name}...")
model_1 = getattr(models, model1_name)()
model_2 = getattr(models, model2_name)()
model_1.load("cuda:0")
model_2.load("cuda:1")

# 设置优化参数
optimization_kwargs = {
    'latent_size': (1, 4, 64, 64),
    'im_size': (1, 3, 224, 224),
    'optimizer_kwargs': {'lr': 1e-1, 'betas': (0.9, 0.999), 'eps': 1e-8},  # 提高学习率以加速收敛
    'hf_cache_dir': '/mnt/data/chenghan/huggingface_cache',
    'model_id': 'OFA-Sys/small-stable-diffusion-v0',
    'random_seed': seed,
    'max_steps': max_steps,
    'readout_type': 'logsoftmax',
    'verbose': True,
    'return_PIL_images': True
}

print(f"开始优化: {model1_name} 识别 {class1} vs {model2_name} 识别 {class2}...")

try:
    # 运行优化
    _, PIL_images, controversiality_scores = optimize_controversial_stimuli_with_diffusion(
        model_1, model_2, class1, class2, **optimization_kwargs
    )

    # 保存结果
    for i, (image, score) in enumerate(zip(PIL_images, controversiality_scores)):
        short_class_1_name = class1.split(',', 1)[0].replace(' ', '_')
        short_class_2_name = class2.split(',', 1)[0].replace(' ', '_')
        
        filename = f"{model1_name}-{short_class_1_name}_vs_{model2_name}-{short_class_2_name}_seed{seed}_score{score:.4f}.png"
        filepath = os.path.join(output_dir, filename)
        
        print(f"保存图像 {i+1}/{len(PIL_images)} 到 {filepath}，对抗性分数: {score:.4f}")
        image.save(filepath)
        
    print(f"优化完成。结果保存在 {output_dir} 目录中。")

except Exception as e:
    print(f"优化过程出错: {e}")
    raise 