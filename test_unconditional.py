import torch
import os
from PIL import Image
import models
from optimize_functions import optimize_controversial_stimuli_with_diffusion_noise

def test_unconditional_diffusion():
    """测试无条件提示词的对抗性图像生成"""
    print("测试无条件提示词的对抗性图像生成...")
    
    # 创建输出目录
    output_dir = "diffusion_results/unconditional_test"
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载两个模型
    model_1 = models.Resnet50()
    model_2 = models.InceptionV3()
    model_1.load("cuda:0")
    model_2.load("cuda:0")
    
    # 获取目标类别
    class_1 = "Persian cat"  # 对应ImageNet索引283
    class_2 = "Labrador retriever"  # 对应ImageNet索引208
    
    # 使用服务器上已有的Hugging Face缓存目录
    hf_cache_dir = "/mnt/data/chenghan/huggingface_cache"
    
    # 方法1：使用空字符串作为提示词
    # 运行优化，但通过修改扩散模型内部，使用空提示词
    print("\n测试方法1：使用空字符串作为提示词")
    
    # 自定义提示词处理函数，在优化函数内部使用
    def custom_prompt_processor(tokenizer, text_encoder, device):
        # 使用空字符串作为提示词
        empty_prompt = ""
        
        # 对文本提示进行编码
        text_inputs = tokenizer(
            [empty_prompt],
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        text_embeddings = text_encoder(text_inputs.input_ids)[0]
        
        # 准备无条件嵌入（也是空提示词）
        uncond_input = tokenizer(
            [""], padding="max_length", max_length=tokenizer.model_max_length,
            return_tensors="pt"
        ).to(device)
        uncond_embeddings = text_encoder(uncond_input.input_ids)[0]
        
        # 拼接条件和无条件嵌入用于分类器引导
        # 这里我们将条件嵌入和无条件嵌入设为相同，相当于无条件生成
        text_embeddings_cfg = torch.cat([uncond_embeddings, uncond_embeddings])
        
        return text_embeddings_cfg
    
    # 本次运行不使用提示词
    try:
        _, images_1, score_1 = optimize_controversial_stimuli_with_diffusion_noise(
            model_1=model_1,
            model_2=model_2,
            class_1=class_1,
            class_2=class_2,
            num_inference_steps=10,  # 减少步数以加快推理
            max_steps=100,  # 减少优化步数以快速得到结果
            target_score=0.9,  # 设置一个较高的目标分数提前停止
            random_seed=42,
            guidance_scale=7.5,  # 保持正常的引导强度
            hf_cache_dir=hf_cache_dir,  # 使用指定的缓存目录
            verbose=True,
            custom_prompt_processor=custom_prompt_processor  # 自定义处理器
        )
        
        # 保存图像
        image_path_1 = os.path.join(output_dir, "unconditional_prompt.png")
        images_1[0].save(image_path_1)
        print(f"保存图像到 {image_path_1}, 对抗性分数: {score_1[0]:.4f}")
    except Exception as e:
        print(f"方法1失败: {e}")
        score_1 = [0.0]
        image_path_1 = None
    
    # 方法2：将guidance_scale设为0
    print("\n测试方法2：将guidance_scale设为0")
    
    try:
        # 运行优化，但将分类器引导强度设为0
        _, images_2, score_2 = optimize_controversial_stimuli_with_diffusion_noise(
            model_1=model_1,
            model_2=model_2,
            class_1=class_1,
            class_2=class_2,
            num_inference_steps=10,  # 减少步数以加快推理
            max_steps=100,  # 减少优化步数以快速得到结果
            target_score=0.9,  # 设置一个较高的目标分数提前停止
            random_seed=42,
            guidance_scale=0.0,  # 设置为0，完全消除条件影响
            hf_cache_dir=hf_cache_dir,  # 使用指定的缓存目录
            verbose=True
        )
        
        # 保存图像
        image_path_2 = os.path.join(output_dir, "zero_guidance.png")
        images_2[0].save(image_path_2)
        print(f"保存图像到 {image_path_2}, 对抗性分数: {score_2[0]:.4f}")
    except Exception as e:
        print(f"方法2失败: {e}")
        score_2 = [0.0]
        image_path_2 = None
    
    # 方法3：对比：使用正常提示词
    print("\n对比：使用正常提示词")
    
    try:
        # 运行优化，使用正常提示词
        _, images_3, score_3 = optimize_controversial_stimuli_with_diffusion_noise(
            model_1=model_1,
            model_2=model_2,
            class_1=class_1,
            class_2=class_2,
            num_inference_steps=10,  # 减少步数以加快推理
            max_steps=100,  # 减少优化步数以快速得到结果
            target_score=0.9,  # 设置一个较高的目标分数提前停止
            random_seed=42,
            guidance_scale=7.5,  # 正常的引导强度
            hf_cache_dir=hf_cache_dir,  # 使用指定的缓存目录
            verbose=True
        )
        
        # 保存图像
        image_path_3 = os.path.join(output_dir, "normal_prompt.png")
        images_3[0].save(image_path_3)
        print(f"保存图像到 {image_path_3}, 对抗性分数: {score_3[0]:.4f}")
    except Exception as e:
        print(f"方法3失败: {e}")
        score_3 = [0.0]
        image_path_3 = None
    
    # 打印对比结果
    print("\n结果对比:")
    print(f"空提示词的对抗性分数: {score_1[0]:.4f}")
    print(f"零引导强度的对抗性分数: {score_2[0]:.4f}")
    if image_path_3:
        print(f"正常提示词的对抗性分数: {score_3[0]:.4f}")
    
    return {
        "empty_prompt": {"image_path": image_path_1, "score": score_1[0]},
        "zero_guidance": {"image_path": image_path_2, "score": score_2[0]},
        "normal_prompt": {"image_path": image_path_3, "score": score_3[0] if image_path_3 else 0.0}
    }

if __name__ == "__main__":
    test_unconditional_diffusion() 