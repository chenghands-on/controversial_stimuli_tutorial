#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成缺失的对抗性图像：ResNet50识别为波斯猫，InceptionV3识别为魏玛猎犬
"""

import os
import torch
from optimize_functions import optimize_controversial_stimuli_with_diffusion_latent
from models import Resnet50, InceptionV3

def main():
    # 确保输出目录存在
    os.makedirs('optimization_results/diffusion_latent_optim_cat_vs_dog_v2', exist_ok=True)

    # 移除可能存在的in_process_flag文件
    flag_file = 'optimization_results/diffusion_latent_optim_cat_vs_dog_v2/Resnet50-Persian_cat_vs_InceptionV3-Weimaraner_seed1.png.in_process_flag'
    if os.path.exists(flag_file):
        os.remove(flag_file)
        print(f"移除了未完成的标志文件: {flag_file}")

    # 加载模型
    print("加载模型...")
    model_1 = Resnet50()
    model_2 = InceptionV3()
    
    # 重要：调用load()方法初始化模型和设备
    print("初始化模型并加载到设备...")
    model_1.load()  # 如果有GPU，默认加载到cuda:0
    model_2.load()  # 如果有GPU，默认加载到cuda:0

    # 打印可用的类名和索引，帮助诊断问题
    print("\n=== 检查Resnet50模型的类名 ===")
    resnet_classes = []
    for key in model_1.class_name_to_idx.keys():
        if "cat" in key.lower() or "persian" in key.lower():
            resnet_classes.append(key)
        if "weimaraner" in key.lower() or "dog" in key.lower():
            resnet_classes.append(key)
    print(f"Resnet50 猫/狗相关类名: {sorted(resnet_classes)}")
    
    print("\n=== 检查InceptionV3模型的类名 ===")
    inception_classes = []
    for key in model_2.class_name_to_idx.keys():
        if "cat" in key.lower() or "persian" in key.lower():
            inception_classes.append(key)
        if "weimaraner" in key.lower() or "dog" in key.lower():
            inception_classes.append(key)
    print(f"InceptionV3 猫/狗相关类名: {sorted(inception_classes)}")

    # 设置分类 - 使用正确的类名
    class_1 = 'Persian cat'  # 修正类名格式
    class_2 = 'Weimaraner'   # 这个是正确的
    
    print(f"\n将使用以下类名进行优化:")
    print(f"- class_1 (应该被Resnet50识别): {class_1}")
    print(f"- class_2 (应该被InceptionV3识别): {class_2}")
    
    # 检查类名是否有效
    if class_1 not in model_1.class_name_to_idx:
        print(f"警告: '{class_1}' 不在Resnet50的类名列表中!")
    if class_2 not in model_2.class_name_to_idx:
        print(f"警告: '{class_2}' 不在InceptionV3的类名列表中!")

    # 运行优化
    print(f"开始对抗性优化: {model_1.model_name} 识别为 {class_1}, {model_2.model_name} 识别为 {class_2}")
    
    images, pil_images, score = optimize_controversial_stimuli_with_diffusion_latent(
        model_1=model_1,
        model_2=model_2,
        class_1=class_1,
        class_2=class_2,
        random_seed=1,
        max_steps=300,
        verbose=True
    )

    # 保存图像
    # 使用原始命名格式，以便与现有图像保持一致
    short_class_1_name = 'Persian_cat'  # 保持与现有图像相同的命名
    short_class_2_name = 'Weimaraner'   # 保持与现有图像相同的命名
    
    output_file = f'optimization_results/diffusion_latent_optim_cat_vs_dog_v2/Resnet50-{short_class_1_name}_vs_InceptionV3-{short_class_2_name}_seed1.png'
    pil_images[0].save(output_file)
    print(f'图像已保存至: {output_file}')
    print(f'对抗性分数: {score[0]:.4f}')

if __name__ == "__main__":
    # 设置随机种子确保可重复性
    torch.manual_seed(1)
    main() 