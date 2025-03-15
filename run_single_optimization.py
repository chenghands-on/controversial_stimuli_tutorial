#!/usr/bin/env python
import os
import torch
import argparse
from PIL import Image
import pathlib

import models
from optimize import optimize_controversial_stimuli_with_diffusion

def main():
    parser = argparse.ArgumentParser(description='运行单个对抗性刺激优化并保存结果')
    parser.add_argument('--model1', type=str, default='Resnet50', help='第一个模型名称')
    parser.add_argument('--model2', type=str, default='Resnet_50_l2_eps5', help='第二个模型名称')
    parser.add_argument('--class1', type=str, default='Persian cat', help='第一个类别名称')
    parser.add_argument('--class2', type=str, default='Weimaraner', help='第二个类别名称')
    parser.add_argument('--output_dir', type=str, default='diffusion_results', help='输出目录')
    parser.add_argument('--seed', type=int, default=0, help='随机种子')
    parser.add_argument('--max_steps', type=int, default=300, help='最大优化步数')
    parser.add_argument('--min_controversiality', type=float, default=0.5, help='最低对抗性分数阈值')
    parser.add_argument('--device1', type=str, default='cuda:0', help='第一个模型的设备')
    parser.add_argument('--device2', type=str, default='cuda:1', help='第二个模型的设备')
    parser.add_argument('--model_id', type=str, default='OFA-Sys/small-stable-diffusion-v0', help='扩散模型ID')
    parser.add_argument('--hf_cache_dir', type=str, default='/mnt/data/chenghan/huggingface_cache', help='HuggingFace缓存目录')
    args = parser.parse_args()

    # 创建输出目录
    pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # 加载模型
    print(f"加载模型 {args.model1} 和 {args.model2}...")
    model_1 = getattr(models, args.model1)()
    model_2 = getattr(models, args.model2)()
    model_1.load(args.device1)
    model_2.load(args.device2)

    # 设置优化参数
    optimization_kwargs = {
        'latent_size': (1, 4, 64, 64),
        'im_size': (1, 3, 224, 224),
        'optimizer_kwargs': {'lr': 5e-2, 'betas': (0.9, 0.999), 'eps': 1e-8},
        'hf_cache_dir': args.hf_cache_dir,
        'model_id': args.model_id,
        'random_seed': args.seed,
        'max_steps': args.max_steps,
        'readout_type': 'logsoftmax',
        'verbose': True,
        'return_PIL_images': True
    }

    print(f"开始优化: {args.model1} 识别 {args.class1} vs {args.model2} 识别 {args.class2}...")
    try:
        # 运行优化
        _, PIL_images, controversiality_scores = optimize_controversial_stimuli_with_diffusion(
            model_1, model_2, args.class1, args.class2, **optimization_kwargs
        )

        # 保存结果
        for i, (image, score) in enumerate(zip(PIL_images, controversiality_scores)):
            short_class_1_name = args.class1.split(',', 1)[0].replace(' ', '_')
            short_class_2_name = args.class2.split(',', 1)[0].replace(' ', '_')
            
            filename = f"{args.model1}-{short_class_1_name}_vs_{args.model2}-{short_class_2_name}_seed{args.seed}_score{score:.4f}.png"
            filepath = os.path.join(args.output_dir, filename)
            
            print(f"保存图像 {i+1}/{len(PIL_images)} 到 {filepath}，对抗性分数: {score:.4f}")
            image.save(filepath)
            
            # 如果分数低于阈值，警告用户
            if score < args.min_controversiality:
                print(f"警告: 对抗性分数 {score:.4f} 低于阈值 {args.min_controversiality}")
            else:
                print(f"成功: 对抗性分数 {score:.4f} 达到或超过阈值 {args.min_controversiality}")
                
        print(f"优化完成。结果保存在 {args.output_dir} 目录中。")
    
    except Exception as e:
        print(f"优化过程出错: {e}")
        raise

if __name__ == "__main__":
    main() 