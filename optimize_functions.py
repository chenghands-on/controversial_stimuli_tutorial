import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
import colored
from diffusers import StableDiffusionPipeline, DiffusionPipeline, DDPMPipeline, DDIMPipeline, LDMPipeline

# 直接定义controversiality_score函数，而不是从optimize.py导入
def smooth_minimum(input,dim=-1,alpha=1.0):
  """ A smooth minimum based on inverted logsumexp with sharpness parameter alpha. alpha-->inf approaches hard minimum"""
  return -1/alpha*torch.logsumexp(-alpha*input,dim=dim) # https://en.wikipedia.org/wiki/LogSumExp

def controversiality_score(im,model_1,model_2,class_1_name,class_2_name,alpha=1.0,readout_type='logsoftmax',verbose=True):
  """ Evaluate the smooth and hard controversiality scores of an NCHWC image tensor according to two models and two classes

  Args:
    im (torch.Tensor): image tensor to evaluate (a 3d (chw) or 4d (nhcw)).
    model1, model2 (tuple): model objects, e.g., TVPretrainedModel, see above.
    class_1_name, class_2_name (str): Target classes. The controversiality score is high when model 1 detects class 1 (but not class 2) and model 2 detects class 2 (but not class 1).
    alpha (float): smooth controversiality score sharpness.
    readout_type (str): 'logits' for models with sigmoid readout, 'logsoftmax' for models with softmax readout.
    verbose (boolean): if True (default), shows image probabilities (averaged across images if a batch is provided).

  Returns:
    (tuple): tuple containing:
        smooth_controversiality_score (torch.Tensor): a smooth controversiality score (for optimization).
        hard_controversiality_score (torch.Tensor) (str): a hard score (for evaluation).
        info (dict): class probabilities.
  """

  # get class indecis (this relatively cumbersome implementation allows for models with mismatching class orders)
  m1_class1_ind=model_1.class_name_to_idx[class_1_name]
  m1_class2_ind=model_1.class_name_to_idx[class_2_name]
  m2_class1_ind=model_2.class_name_to_idx[class_1_name]
  m2_class2_ind=model_2.class_name_to_idx[class_2_name]

  model_1_logits,model_1_probabilities=model_1(im)
  model_2_logits,model_2_probabilities=model_2(im)

  # in case we are using two GPUs, we need to bring the logits and probabilities to the same device.
  if model_2.device != model_1.device:
    model_2_logits=model_2_logits.to(model_1.device)
    model_2_probabilities=model_2_probabilities.to(model_1.device)
  #   print("logits moved to gpu:",time.perf_counter()-t0)
  if readout_type=='logits':
    if class_1_name != class_2_name:
      # smooth minimum of logits - this is the score we optimized in Golan et al., 2020 (Eq. 4).
      input=torch.stack([model_1_logits[:,m1_class1_ind],-model_2_logits[:,m2_class1_ind],
                        -model_1_logits[:,m1_class2_ind],model_2_logits[:,m2_class2_ind]],dim=-1)
    else: # simple activation maximization of logits for the non-controversial case
      input=torch.stack([model_1_logits[:,m1_class1_ind],model_2_logits[:,m2_class2_ind]],dim=-1)
  elif readout_type=='logsoftmax':
    # However, for softmax readout (unlike sigmoid readout), manipulating class-specific logits doesn't fully control output probabilities
    # (since all of the logits contribute to the resulting probabilities). Therefore, for softmax models, we target the logsoftmax scores.
    # This is essentially a smooth variant of Eq. 1 in Golan et al., 2020.
    logsoftmax=torch.nn.LogSoftmax(dim=-1)
    model_1_logsoftmax=logsoftmax(model_1_logits)
    model_2_logsoftmax=logsoftmax(model_2_logits)
    input=torch.stack([model_1_logsoftmax[:,m1_class1_ind],model_2_logsoftmax[:,m2_class2_ind]],dim=-1)
  else:
     raise ValueError("readout_type must be logits or logsoftmax")

  smooth_controversiality_score = smooth_minimum(input,alpha=alpha,dim=-1)

  # A hard minimum of probabilities. The maximum score that can be achieved by this controversiality measure is 1.0.
  # This score is used for evaluating the controversiality of the resulting images once the optimization is done.
  input=torch.stack([model_1_probabilities[:,m1_class1_ind],1-model_2_probabilities[:,m2_class1_ind],
                     1-model_1_probabilities[:,m1_class2_ind],model_2_probabilities[:,m2_class2_ind]],dim=-1)
  hard_controversiality_score, _ = torch.min(input,dim=-1)

  # save some class probabilities for display
  info={'p(class_1|model_1)':model_1_probabilities[:,m1_class1_ind],
        'p(class_2|model_1)':model_1_probabilities[:,m1_class2_ind],
        'p(class_1|model_2)':model_2_probabilities[:,m2_class1_ind],
        'p(class_2|model_2)':model_2_probabilities[:,m2_class2_ind]}

  return smooth_controversiality_score, hard_controversiality_score, info

# 原始像素空间优化函数
def optimize_controversial_stimuli_with_diffusion_pixel(model_1, model_2, class_1, class_2,
                                   latent_size=(1, 4, 64, 64),  # Default latent shape for SD models
                                   model_id="OFA-Sys/small-stable-diffusion-v0",
                                   hf_cache_dir=None,
                                   im_size=(1, 3, 224, 224),
                                   pytorch_optimizer='Adam',
                                   optimizer_kwargs={'lr':1e-1,'betas':(0.9, 0.999),'weight_decay':0,'eps':1e-8},
                                   readout_type='logsoftmax',
                                   random_seed=0,
                                   max_steps=1000,
                                   max_consecutive_steps_without_pixel_change=10,
                                   return_PIL_images=True,
                                   verbose=True):
  """Optimize controversial stimuli with respect to two models and two classes, starting from a diffusion model init.

  此函数在像素空间(x)优化对抗性刺激图像，以diffusion模型生成的图像作为起点。
  
  Args:
    model_1, model2 (object): model objects.
    class_1, class_2 (str): target class names.
    latent_size (tuple): Size of the latent variable for initialization.
    model_id (str): Hugging Face model ID for the diffusion model.
    hf_cache_dir (str): Directory to use for Hugging Face cache.
    im_size (tuple): Size of the image to optimize (B, C, H, W).
    pytorch_optimizer (str or class): either the name of a torch.optim class or an optimizer class.
    optimizer_kwargs (dict): keywords passed to the optimizer
    readout_type (str): 'logits' for models with sigmoid readout, 'logsoftmax' for models with softmax readout.
    random_seed (int): sets the random seed for PyTorch.
    max_steps (int): maximal number of optimization steps
    max_consecutive_steps_without_pixel_change (int): if the image hasn't changed for this number of steps, stop
    return_PIL_images (boolean): if True (default), return also a list of PIL images.
    verbose (boolean): if True (default), shows image probabilities and other diagnostic messages.

  Returns:
    (tuple): tuple containing:
      im_tensor (torch.Tensor): Controversial stimuli image tensor in nchw format.
      PIL_controversial_stimuli (list): Controversial stimuli as list of PIL.Image images.
      hard_controversiality_score: (list): Controversiality score for each image as float.
    or (if return_PIL_images == False):
      im_tensor (torch.Tensor): Controversial stimuli image tensor in nchw format.
      hard_controversiality_score: (list): Controversiality score for each image as float.
  """
  import torch.nn.functional as F

  verboseprint = print if verbose else lambda *a, **k: None # https://stackoverflow.com/a/5980173

  # used for display:
  short_class_1_name = class_1.split(',', 1)[0]
  short_class_2_name = class_2.split(',', 1)[0]
  BOLD = colored.attr('bold')
  normal = colored.attr('reset')

  # Set random seed
  torch.manual_seed(random_seed)
  np.random.seed(random_seed)

  # Load diffusion model to generate initial image
  verboseprint(f"Using Hugging Face cache directory: {hf_cache_dir}")
  verboseprint(f"Loading diffusion model: {model_id}")
  
  # 检测模型类型
  is_ddpm = "ddpm" in model_id.lower()
  is_ldm = "ldm" in model_id.lower() or "latent" in model_id.lower()
  
  try:
    # 根据模型类型尝试加载不同的管道
    if is_ddpm:
      try:
        diffusion_model = DDPMPipeline.from_pretrained(
            model_id,
            cache_dir=hf_cache_dir,
            safety_checker=None,
        )
      except Exception as e:
        verboseprint(f"Error loading DDPM, falling back to DDIM: {e}")
        diffusion_model = DDIMPipeline.from_pretrained(
            model_id,
            cache_dir=hf_cache_dir,
            safety_checker=None,
        )
    elif is_ldm:
      try:
        diffusion_model = LDMPipeline.from_pretrained(
            model_id,
            cache_dir=hf_cache_dir,
            safety_checker=None,
        )
      except Exception as e:
        verboseprint(f"Error loading LDM, falling back to DiffusionPipeline: {e}")
        diffusion_model = DiffusionPipeline.from_pretrained(
            model_id,
            cache_dir=hf_cache_dir,
            safety_checker=None,
        )
    else:
      try:
        diffusion_model = DiffusionPipeline.from_pretrained(
            model_id,
            cache_dir=hf_cache_dir,
            safety_checker=None,
        )
      except Exception as e:
        verboseprint(f"Error loading DiffusionPipeline, falling back to StableDiffusionPipeline: {e}")
        diffusion_model = StableDiffusionPipeline.from_pretrained(
            model_id,
            cache_dir=hf_cache_dir,
            safety_checker=None,
        )
    
    diffusion_model = diffusion_model.to("cuda")
  except Exception as e:
    verboseprint(f"Error loading model: {e}")
    raise

  # 更新：检测模型类型
  is_ddpm = is_ddpm or isinstance(diffusion_model, (DDPMPipeline, DDIMPipeline))
  is_ldm = is_ldm or isinstance(diffusion_model, LDMPipeline)

  # Generate an initial image from the diffusion model
  verboseprint("Generating initial image from diffusion model...")
  
  # 确定种子
  if random_seed is not None:
    generator = torch.Generator(device="cuda").manual_seed(random_seed)
  else:
    generator = None

  # 生成初始图像
  with torch.no_grad():
    # 对于LDM模型，使用prompt
    if is_ldm:
      prompt = f"{short_class_1_name} and {short_class_2_name}"
      output = diffusion_model(
          prompt=prompt,
          num_inference_steps=30,  # Fewer steps for initialization
          generator=generator
      )
    # 对于DDPM模型，不使用prompt
    elif is_ddpm:
      output = diffusion_model(
          num_inference_steps=30,
          generator=generator
      )
    # 对于其他模型（如Stable Diffusion）
    else:
      prompt = f"{short_class_1_name} and {short_class_2_name}"
      output = diffusion_model(
          prompt=prompt,
          num_inference_steps=30,
          generator=generator
      )
    
    # 处理输出
    if hasattr(output, 'images'):
      images = output.images
      initial_image = torch.stack([torch.from_numpy(np.array(img)) for img in images]).permute(0, 3, 1, 2) / 255.0
      initial_image = initial_image.to("cuda")
    else:
      initial_image = output.sample
      initial_image = (initial_image + 1) / 2.0  # 如果模型输出范围是[-1, 1]
      
    # 调整大小到目标尺寸
    initial_image = F.interpolate(initial_image, size=(im_size[2], im_size[3]), mode='bilinear', align_corners=False)
  
  # Now use this initial image and optimize directly (similar to optimize_controversial_stimuli)
  verboseprint("Starting direct optimization from the diffusion-generated image...")
  
  # Make the image a parameter that requires gradient
  x = initial_image.clone().to("cuda")
  x.requires_grad = True
  
  # Initialize optimizer
  if isinstance(pytorch_optimizer, str):
    OptimClass = getattr(torch.optim, pytorch_optimizer)
  else:
    OptimClass = pytorch_optimizer
  optimizer = OptimClass(params=[x], **optimizer_kwargs)

  previous_im = None
  alpha = 100.0
  converged = False
  consecutive_steps_without_pixel_change = 0
  
  for i_step in range(max_steps):
    optimizer.zero_grad()
    
    # Calculate controversiality score
    smooth_controversiality_score, hard_controversiality_score, info = controversiality_score(
        x, model_1, model_2, class_1, class_2, alpha=alpha,
        readout_type=readout_type, verbose=verbose
    )
    
    # We want to maximize controversiality (minimize negative controversiality)
    loss = -smooth_controversiality_score
    loss = loss.sum()  # Sum across batch
    loss.backward(retain_graph=True)
    
    # Update the image
    optimizer.step()
    
    # Clamp values to valid image range [0, 1]
    with torch.no_grad():
      x.data.clamp_(0, 1)
    
    verboseprint('{}: {} {:>7.2%}, {} {:>7.2%} │ {}: {} {:>7.2%}, {} {:>7.2%} │ {}:loss={:3.2e}'.format(
          BOLD+model_1.model_name+normal, short_class_1_name, info['p(class_1|model_1)'].mean(), short_class_2_name, info['p(class_2|model_1)'].mean(),
          BOLD+model_2.model_name+normal, short_class_1_name, info['p(class_1|model_2)'].mean(), short_class_2_name, info['p(class_2|model_2)'].mean(),
          i_step, loss.item()))
    
    # Monitor image change magnitude
    if previous_im is not None:
      abs_change = (x - previous_im).abs() * 255.0  # Change on 0-255 scale
      max_abs_change = abs_change.max().item()
      if max_abs_change < 0.5:  # Less than half an intensity level
        consecutive_steps_without_pixel_change += 1
        if consecutive_steps_without_pixel_change > max_consecutive_steps_without_pixel_change:
          converged = True
          break
      else:
        consecutive_steps_without_pixel_change = 0
    
    previous_im = x.detach().clone()
  
  if converged:
    verboseprint(f'converged (n_steps={i_step+1})')
  else:
    verboseprint(f'max steps achieved (n_steps={i_step+1})')
  
  # Quantize intensity
  x = (x.detach() * 255.0).round() / 255.0
  
  # Evaluate final controversiality score with quantized image
  _, hard_controversiality_score, _ = controversiality_score(x, model_1, model_2, class_1, class_2, verbose=False)
  hard_controversiality_score = hard_controversiality_score.detach().cpu().numpy().tolist()
  
  verboseprint('controversiality score: ' + ', '.join('{:0.2f}'.format(f) for f in hard_controversiality_score))
  
  if return_PIL_images:
    numpy_controversial_stimuli = x.detach().cpu().numpy().transpose([0, 2, 3, 1])  # NCHW -> NHWC
    numpy_controversial_stimuli = (numpy_controversial_stimuli * 255.0).astype(np.uint8)
    PIL_controversial_stimuli = []
    for i in range(len(numpy_controversial_stimuli)):
      PIL_controversial_stimuli.append(Image.fromarray(numpy_controversial_stimuli[i]))
    return x.detach(), PIL_controversial_stimuli, hard_controversiality_score
  else:
    return x.detach(), hard_controversiality_score


# 修改后的潜在空间优化函数
def optimize_controversial_stimuli_with_diffusion_latent(model_1, model_2, class_1, class_2,
                                   latent_size=(1, 4, 64, 64),  # Default latent shape for SD models
                                   model_id="OFA-Sys/small-stable-diffusion-v0",
                                   hf_cache_dir=None,
                                   im_size=(1, 3, 224, 224),
                                   pytorch_optimizer='Adam',
                                   optimizer_kwargs={'lr':1e-1,'betas':(0.9, 0.999),'weight_decay':0,'eps':1e-8},
                                   readout_type='logsoftmax',
                                   random_seed=0,
                                   max_steps=1000,
                                   max_consecutive_steps_without_pixel_change=10,
                                   return_PIL_images=True,
                                   verbose=True):
  """Optimize controversial stimuli with respect to two models and two classes, in diffusion latent space.

  此函数在潜在空间(z)优化对抗性刺激图像，直接优化diffusion模型的潜在向量。
  与像素空间优化的区别：
  1. 优化对象是潜在向量而不是像素
  2. 每次迭代中，需要将潜在向量通过diffusion模型解码为图像
  3. 优化在更低维度的空间中进行，可能产生更自然的图像
  
  Args:
    model_1, model2 (object): model objects.
    class_1, class_2 (str): target class names.
    latent_size (tuple): Size of the latent variable for initialization.
    model_id (str): Hugging Face model ID for the diffusion model.
    hf_cache_dir (str): Directory to use for Hugging Face cache.
    im_size (tuple): Size of the image to optimize (B, C, H, W).
    pytorch_optimizer (str or class): either the name of a torch.optim class or an optimizer class.
    optimizer_kwargs (dict): keywords passed to the optimizer
    readout_type (str): 'logits' for models with sigmoid readout, 'logsoftmax' for models with softmax readout.
    random_seed (int): sets the random seed for PyTorch.
    max_steps (int): maximal number of optimization steps
    max_consecutive_steps_without_pixel_change (int): if the image hasn't changed for this number of steps, stop
    return_PIL_images (boolean): if True (default), return also a list of PIL images.
    verbose (boolean): if True (default), shows image probabilities and other diagnostic messages.

  Returns:
    (tuple): tuple containing:
      im_tensor (torch.Tensor): Controversial stimuli image tensor in nchw format.
      PIL_controversial_stimuli (list): Controversial stimuli as list of PIL.Image images.
      hard_controversiality_score: (list): Controversiality score for each image as float.
    or (if return_PIL_images == False):
      im_tensor (torch.Tensor): Controversial stimuli image tensor in nchw format.
      hard_controversiality_score: (list): Controversiality score for each image as float.
  """
  import torch.nn.functional as F

  verboseprint = print if verbose else lambda *a, **k: None # https://stackoverflow.com/a/5980173

  # used for display:
  short_class_1_name = class_1.split(',', 1)[0]
  short_class_2_name = class_2.split(',', 1)[0]
  BOLD = colored.attr('bold')
  normal = colored.attr('reset')

  # Set random seed
  torch.manual_seed(random_seed)
  np.random.seed(random_seed)

  # Load diffusion model to generate initial image
  verboseprint(f"Using Hugging Face cache directory: {hf_cache_dir}")
  verboseprint(f"Loading diffusion model: {model_id}")
  
  # 检测模型类型
  is_ddpm = "ddpm" in model_id.lower()
  is_ldm = "ldm" in model_id.lower() or "latent" in model_id.lower()
  is_stable_diffusion = "stable" in model_id.lower()
  
  try:
    # 根据模型类型尝试加载不同的管道
    if is_ddpm:
      try:
        diffusion_model = DDPMPipeline.from_pretrained(
            model_id,
            cache_dir=hf_cache_dir,
            safety_checker=None,
        )
      except Exception as e:
        verboseprint(f"Error loading DDPM, falling back to DDIM: {e}")
        diffusion_model = DDIMPipeline.from_pretrained(
            model_id,
            cache_dir=hf_cache_dir,
            safety_checker=None,
        )
    elif is_ldm:
      try:
        diffusion_model = LDMPipeline.from_pretrained(
            model_id,
            cache_dir=hf_cache_dir,
            safety_checker=None,
        )
      except Exception as e:
        verboseprint(f"Error loading LDM, falling back to DiffusionPipeline: {e}")
        diffusion_model = DiffusionPipeline.from_pretrained(
            model_id,
            cache_dir=hf_cache_dir,
            safety_checker=None,
        )
    elif is_stable_diffusion:
      try:
        diffusion_model = StableDiffusionPipeline.from_pretrained(
            model_id,
            cache_dir=hf_cache_dir,
            safety_checker=None,
        )
      except Exception as e:
        verboseprint(f"Error loading StableDiffusionPipeline: {e}")
        raise
    else:
      try:
        diffusion_model = DiffusionPipeline.from_pretrained(
            model_id,
            cache_dir=hf_cache_dir,
            safety_checker=None,
        )
      except Exception as e:
        verboseprint(f"Error loading DiffusionPipeline, falling back to StableDiffusionPipeline: {e}")
        diffusion_model = StableDiffusionPipeline.from_pretrained(
            model_id,
            cache_dir=hf_cache_dir,
            safety_checker=None,
        )
    
    diffusion_model = diffusion_model.to("cuda")
  except Exception as e:
    verboseprint(f"Error loading model: {e}")
    raise

  # 更新：检测模型类型
  is_ddpm = is_ddpm or isinstance(diffusion_model, (DDPMPipeline, DDIMPipeline))
  is_ldm = is_ldm or isinstance(diffusion_model, LDMPipeline)
  is_stable_diffusion = is_stable_diffusion or isinstance(diffusion_model, StableDiffusionPipeline)

  # 确定种子
  if random_seed is not None:
    generator = torch.Generator(device="cuda").manual_seed(random_seed)
  else:
    generator = None

  # 初始化潜在向量
  verboseprint("Initializing latent vector...")
  
  if is_stable_diffusion:
    # 对于Stable Diffusion，特别处理以获取潜在向量
    prompt = f"{short_class_1_name} and {short_class_2_name}"
    
    # 获取文本嵌入
    text_inputs = diffusion_model.tokenizer(
        [prompt],
        padding="max_length",
        max_length=diffusion_model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).to("cuda")
    text_embeddings = diffusion_model.text_encoder(text_inputs.input_ids)[0]
    
    # 随机初始化latent，这将是我们优化的对象
    latents = torch.randn(
        latent_size,
        generator=generator,
        device="cuda"
    )
    
    # 保存其他组件，优化过程会用到
    vae = diffusion_model.vae
    unet = diffusion_model.unet
    scheduler = diffusion_model.scheduler
    
    # 我们还需要一个中性文本嵌入作为无条件向量（用于分类器引导）
    uncond_input = diffusion_model.tokenizer(
        [""], padding="max_length", max_length=diffusion_model.tokenizer.model_max_length,
        return_tensors="pt"
    ).to("cuda")
    uncond_embeddings = diffusion_model.text_encoder(uncond_input.input_ids)[0]
    
    # 拼接条件和无条件嵌入用于分类器引导
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
    
    # 设置一些分类器引导的参数
    guidance_scale = 7.5  # 标准Stable Diffusion默认值
  else:
    # 对于其他类型的模型，使用随机初始化的潜在向量
    latents = torch.randn(latent_size, device="cuda", generator=generator)
    
    if is_ldm:
      prompt = f"{short_class_1_name} and {short_class_2_name}"
  
  # 将潜在向量设为需要梯度的参数
  latents = latents.clone().detach().requires_grad_(True)
  
  # 初始化优化器，优化潜在向量
  if isinstance(pytorch_optimizer, str):
    OptimClass = getattr(torch.optim, pytorch_optimizer)
  else:
    OptimClass = pytorch_optimizer
  optimizer = OptimClass(params=[latents], **optimizer_kwargs)

  # 为了在潜在空间中优化，我们需要一个函数将潜在向量转换为图像
  def latents_to_image(latents):
    # 根据模型类型使用不同的解码方法 - 移除torch.no_grad()以允许梯度传递
    if is_stable_diffusion:
      # 使用VAE将潜在向量解码为图像
      latents_input = 1 / 0.18215 * latents  # Stable Diffusion特有的缩放因子
      image = vae.decode(latents_input).sample
      image = (image / 2 + 0.5).clamp(0, 1)  # 归一化到[0, 1]
    elif is_ldm:
      # LDM模型的解码
      image = diffusion_model.vqvae.decode(latents)
      image = (image / 2 + 0.5).clamp(0, 1)  # 归一化到[0, 1]
    else:
      # 其他模型类型的解码
      # 这里使用一个简单的线性投影作为后备，但实际应根据具体模型调整
      image = F.interpolate(latents, size=(im_size[2], im_size[3]), mode='bilinear')
      image = image.clamp(0, 1)
      
    # 调整大小到目标尺寸
    if image.shape[-2:] != (im_size[2], im_size[3]):
      image = F.interpolate(image, size=(im_size[2], im_size[3]), mode='bilinear', align_corners=False)
    
    return image
  
  previous_latents = None
  alpha = 100.0
  converged = False
  consecutive_steps_without_latent_change = 0
  
  verboseprint("Starting optimization in latent space...")
  
  for i_step in range(max_steps):
    optimizer.zero_grad()
    
    # 将潜在向量转换为图像
    x = latents_to_image(latents)
    
    # 计算对抗性分数
    smooth_controversiality_score, hard_controversiality_score, info = controversiality_score(
        x, model_1, model_2, class_1, class_2, alpha=alpha,
        readout_type=readout_type, verbose=verbose
    )
    
    # 我们想要最大化对抗性（最小化负对抗性）
    loss = -smooth_controversiality_score
    loss = loss.sum()  # 对批次求和
    loss.backward(retain_graph=True)
    
    # 检查梯度并打印统计信息
    if latents.grad is not None:
      grad_abs = latents.grad.abs()
      grad_mean = grad_abs.mean().item()
      grad_max = grad_abs.max().item()
      grad_min = grad_abs.min().item()
      verboseprint(f"Gradients stats - Mean: {grad_mean:.6e}, Max: {grad_max:.6e}, Min: {grad_min:.6e}")
      
      # 如果梯度非常小，提供警告
      if grad_mean < 1e-8:
        verboseprint(f"WARNING: Very small gradients detected! Mean gradient: {grad_mean:.6e}")
      
      # 应用梯度裁剪，避免梯度爆炸
      if grad_max > 1.0:
        torch.nn.utils.clip_grad_norm_([latents], 1.0)
        verboseprint(f"Applied gradient clipping")
    else:
      verboseprint("WARNING: No gradients computed!")
    
    # 更新潜在向量
    optimizer.step()
    
    # 打印进度信息
    verboseprint('{}: {} {:>7.2%}, {} {:>7.2%} │ {}: {} {:>7.2%}, {} {:>7.2%} │ {}:loss={:3.2e}'.format(
          BOLD+model_1.model_name+normal, short_class_1_name, info['p(class_1|model_1)'].mean(), short_class_2_name, info['p(class_2|model_1)'].mean(),
          BOLD+model_2.model_name+normal, short_class_1_name, info['p(class_1|model_2)'].mean(), short_class_2_name, info['p(class_2|model_2)'].mean(),
          i_step, loss.item()))
    
    # 监控潜在向量变化幅度
    if previous_latents is not None:
      abs_change = (latents - previous_latents).abs()
      max_abs_change = abs_change.max().item()
      if max_abs_change < 1e-4:  # 变化幅度小于阈值
        consecutive_steps_without_latent_change += 1
        if consecutive_steps_without_latent_change > max_consecutive_steps_without_pixel_change:
          converged = True
          break
      else:
        consecutive_steps_without_latent_change = 0
    
    previous_latents = latents.detach().clone()
  
  if converged:
    verboseprint(f'converged (n_steps={i_step+1})')
  else:
    verboseprint(f'max steps achieved (n_steps={i_step+1})')
  
  # 获取最终图像
  final_image = latents_to_image(latents)
  
  # 量化强度（虽然在潜在空间优化，但最终图像仍需量化）
  final_image = (final_image.detach() * 255.0).round() / 255.0
  
  # 用量化后的图像评估最终对抗性分数
  _, hard_controversiality_score, _ = controversiality_score(
      final_image, model_1, model_2, class_1, class_2, verbose=False
  )
  hard_controversiality_score = hard_controversiality_score.detach().cpu().numpy().tolist()
  
  verboseprint('controversiality score: ' + ', '.join('{:0.2f}'.format(f) for f in hard_controversiality_score))
  
  if return_PIL_images:
    numpy_controversial_stimuli = final_image.detach().cpu().numpy().transpose([0, 2, 3, 1])  # NCHW -> NHWC
    numpy_controversial_stimuli = (numpy_controversial_stimuli * 255.0).astype(np.uint8)
    PIL_controversial_stimuli = []
    for i in range(len(numpy_controversial_stimuli)):
      PIL_controversial_stimuli.append(Image.fromarray(numpy_controversial_stimuli[i]))
    return final_image.detach(), PIL_controversial_stimuli, hard_controversiality_score
  else:
    return final_image.detach(), hard_controversiality_score

# 新增：噪声空间优化函数
def optimize_controversial_stimuli_with_diffusion_noise(model_1, model_2, class_1, class_2,
                                   noise_size=(1, 4, 64, 64),  # 噪声张量的大小
                                   model_id="OFA-Sys/small-stable-diffusion-v0",
                                   hf_cache_dir=None,
                                   im_size=(1, 3, 224, 224),
                                   num_inference_steps=50,     # 去噪步骤数
                                   guidance_scale=7.5,         # 分类器引导强度
                                   pytorch_optimizer='Adam',
                                   optimizer_kwargs={'lr':5e-2,'betas':(0.9, 0.999),'weight_decay':0,'eps':1e-8},
                                   readout_type='logsoftmax',
                                   random_seed=0,
                                   max_steps=1000,
                                   max_consecutive_steps_without_change=10,
                                   target_score=0.95,          # 目标对抗性分数，达到后提前停止
                                   custom_prompt_processor=None,  # 自定义提示词处理器
                                   return_PIL_images=True,
                                   verbose=True):
  """优化扩散模型初始噪声以生成对抗性刺激。
  
  与潜在空间优化不同，此函数优化的是扩散模型去噪过程的初始噪声输入，
  而不是直接优化VAE的潜在变量。这允许充分利用扩散模型的去噪能力和文本引导。
  
  Args:
    model_1, model2 (object): 目标分类模型。
    class_1, class_2 (str): 目标类名称。
    noise_size (tuple): 初始噪声张量的大小。
    model_id (str): Hugging Face上的扩散模型ID。
    hf_cache_dir (str): Hugging Face缓存目录。
    im_size (tuple): 最终图像的大小 (B, C, H, W)。
    num_inference_steps (int): 扩散模型的去噪步骤数。
    guidance_scale (float): 分类器引导的强度。
    pytorch_optimizer (str or class): torch.optim优化器类名或类对象。
    optimizer_kwargs (dict): 传递给优化器的参数。
    readout_type (str): 'logits' 或 'logsoftmax'，取决于分类模型的输出层。
    random_seed (int): 随机种子。
    max_steps (int): 最大优化步骤数。
    max_consecutive_steps_without_change (int): 如果图像在这么多步内没有显著变化，则停止优化。
    target_score (float): 目标对抗性分数，达到该分数后提前停止优化。
    custom_prompt_processor (function): 自定义提示词处理函数，接收tokenizer、text_encoder、device参数，返回text_embeddings_cfg。
    return_PIL_images (boolean): 如果为True，返回PIL格式的图像列表。
    verbose (boolean): 是否打印详细信息。

  Returns:
    (tuple): 包含:
      im_tensor (torch.Tensor): 对抗性刺激图像张量，格式为nchw。
      PIL_controversial_stimuli (list): PIL格式的对抗性刺激图像列表。
      hard_controversiality_score: (list): 每个图像的对抗性分数。
    或 (如果return_PIL_images == False):
      im_tensor (torch.Tensor): 对抗性刺激图像张量，格式为nchw。
      hard_controversiality_score: (list): 每个图像的对抗性分数。
  """
  import torch.nn.functional as F
  from tqdm.auto import tqdm
  import time  # 添加时间模块用于衡量优化时间

  verboseprint = print if verbose else lambda *a, **k: None

  # 用于显示的简短类名
  short_class_1_name = class_1.split(',', 1)[0]
  short_class_2_name = class_2.split(',', 1)[0]
  BOLD = colored.attr('bold')
  normal = colored.attr('reset')

  # 设置随机种子
  torch.manual_seed(random_seed)
  np.random.seed(random_seed)

  # 加载扩散模型
  verboseprint(f"使用Hugging Face缓存目录: {hf_cache_dir}")
  verboseprint(f"加载扩散模型: {model_id}")
  
  # 仅支持Stable Diffusion模型，因为我们需要访问其内部组件
  try:
    diffusion_model = StableDiffusionPipeline.from_pretrained(
        model_id,
        cache_dir=hf_cache_dir,
        safety_checker=None,
        # 使用torch_dtype=torch.float16可以减少一半内存使用，但可能影响质量
        # torch_dtype=torch.float16,
    )
    diffusion_model = diffusion_model.to("cuda")
    
    # 启用UNet的梯度检查点可以显著减少内存使用，但会增加计算时间
    diffusion_model.unet.enable_gradient_checkpointing()
    
    # 打印内存使用提示
    verboseprint(f"提示: 使用的推理步骤数为{num_inference_steps}。如果内存不足，尝试进一步减少此值。")
  except Exception as e:
    verboseprint(f"加载模型时出错: {e}")
    raise ValueError(f"此功能仅支持Stable Diffusion模型，加载 {model_id} 时出错: {e}")

  # 确认是Stable Diffusion模型
  if not hasattr(diffusion_model, "unet") or not hasattr(diffusion_model, "vae"):
    raise ValueError("此函数需要完整的Stable Diffusion模型，包含UNet和VAE组件")
    
  # 提取模型组件
  vae = diffusion_model.vae
  unet = diffusion_model.unet
  text_encoder = diffusion_model.text_encoder
  tokenizer = diffusion_model.tokenizer
  scheduler = diffusion_model.scheduler
  
  # 设置生成器
  generator = None
  if random_seed is not None:
    generator = torch.Generator(device="cuda").manual_seed(random_seed)

  # 准备文本嵌入
  if custom_prompt_processor:
    # 使用自定义处理器
    verboseprint("使用自定义提示词处理器")
    text_embeddings_cfg = custom_prompt_processor(tokenizer, text_encoder, "cuda")
  else:
    # 使用默认提示词
    prompt = f"{short_class_1_name} and {short_class_2_name}"
    verboseprint(f"使用文本提示: '{prompt}'")
    
    # 对文本提示进行编码
    text_inputs = tokenizer(
        [prompt],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).to("cuda")
    text_embeddings = text_encoder(text_inputs.input_ids)[0]
    
    # 准备无条件嵌入（用于分类器引导）
    uncond_input = tokenizer(
        [""], padding="max_length", max_length=tokenizer.model_max_length,
        return_tensors="pt"
    ).to("cuda")
    uncond_embeddings = text_encoder(uncond_input.input_ids)[0]
    
    # 拼接条件和无条件嵌入用于分类器引导
    text_embeddings_cfg = torch.cat([uncond_embeddings, text_embeddings])
  
  # 初始化噪声 - 这是我们要优化的对象
  latents_shape = noise_size
  noise = torch.randn(latents_shape, generator=generator, device="cuda")
  noise = noise.clone().detach().requires_grad_(True)
  
  # 初始化优化器
  if isinstance(pytorch_optimizer, str):
    OptimClass = getattr(torch.optim, pytorch_optimizer)
  else:
    OptimClass = pytorch_optimizer
  optimizer = OptimClass(params=[noise], **optimizer_kwargs)
  
  # 设置扩散模型的训练模式（确保梯度可传播）
  unet.train()
  
  # 定义从噪声到图像的过程（整个推理过程）
  def noise_to_image(noise, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale):
    # 复制噪声以避免修改原始噪声
    latents = noise.clone()
    
    # 设置scheduler初始状态
    scheduler.set_timesteps(num_inference_steps)
    
    # 扩散模型噪声到图像的推理过程
    for i, t in enumerate(scheduler.timesteps):
      # 复制latents，用于分类器引导 (一个无条件，一个有条件)
      latent_model_input = torch.cat([latents] * 2)
      
      # 添加噪声缩放
      latent_model_input = scheduler.scale_model_input(latent_model_input, t)
      
      # 预测噪声
      with torch.set_grad_enabled(True):  # 确保计算梯度
        noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings_cfg).sample
      
      # 进行分类器引导
      noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
      noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
      
      # 去噪步骤
      latents = scheduler.step(noise_pred, t, latents).prev_sample
    
    # 使用VAE将潜在空间转换为像素空间
    latents = 1 / 0.18215 * latents  # Stable Diffusion的缩放因子
    with torch.set_grad_enabled(True):  # 确保计算梯度
      image = vae.decode(latents).sample
    
    # 调整到[0,1]范围
    image = (image / 2 + 0.5).clamp(0, 1)
    
    # 如果需要，调整图像大小
    if image.shape[-2:] != (im_size[2], im_size[3]):
      image = F.interpolate(image, size=(im_size[2], im_size[3]), mode='bilinear', align_corners=False)
    
    return image
  
  # 优化循环
  verboseprint("开始在噪声空间优化...")
  verboseprint(f"目标对抗性分数设置为: {target_score:.2f}")
  
  previous_noise = None
  alpha = 100.0
  converged = False
  consecutive_steps_without_change = 0
  best_score = -float('inf')
  best_image = None
  best_noise = None
  
  # 记录开始时间
  start_time = time.time()
  
  for i_step in range(max_steps):
    optimizer.zero_grad()
    
    # 将噪声转换为图像，通过整个扩散模型和VAE
    x = noise_to_image(noise, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale)
    
    # 计算对抗性分数
    smooth_controversiality_score, hard_controversiality_score, info = controversiality_score(
        x, model_1, model_2, class_1, class_2, alpha=alpha,
        readout_type=readout_type, verbose=False
    )
    
    # 最大化对抗性（最小化负对抗性）
    loss = -smooth_controversiality_score
    loss = loss.sum()  # 对批次求和
    loss.backward(retain_graph=True)
    
    # 检查梯度并打印统计信息
    if noise.grad is not None:
      grad_abs = noise.grad.abs()
      grad_mean = grad_abs.mean().item()
      grad_max = grad_abs.max().item()
      grad_min = grad_abs.min().item()
      verboseprint(f"梯度统计 - 平均: {grad_mean:.6e}, 最大: {grad_max:.6e}, 最小: {grad_min:.6e}")
      
      # 应用梯度裁剪，避免梯度爆炸
      if grad_max > 1.0:
        torch.nn.utils.clip_grad_norm_([noise], 1.0)
        verboseprint(f"已应用梯度裁剪")
    else:
      verboseprint("警告: 未计算梯度!")
    
    # 更新噪声
    optimizer.step()
    
    # 记录当前最佳结果
    current_score = hard_controversiality_score.mean().item()
    if current_score > best_score:
      best_score = current_score
      best_image = x.detach().clone()
      best_noise = noise.detach().clone()
      verboseprint(f"新的最佳分数: {best_score:.4f}")
      
      # 新增：检查是否达到目标分数
      if best_score >= target_score:
        verboseprint(f"已达到目标分数 {target_score:.2f}! 提前停止优化。")
        converged = True
        break
    
    # 打印进度信息
    verboseprint('{}: {} {:>7.2%}, {} {:>7.2%} │ {}: {} {:>7.2%}, {} {:>7.2%} │ {}:loss={:3.2e}'.format(
          BOLD+model_1.model_name+normal, short_class_1_name, info['p(class_1|model_1)'].mean(), short_class_2_name, info['p(class_2|model_1)'].mean(),
          BOLD+model_2.model_name+normal, short_class_1_name, info['p(class_1|model_2)'].mean(), short_class_2_name, info['p(class_2|model_2)'].mean(),
          i_step, loss.item()))
    
    # 监控噪声变化幅度
    if previous_noise is not None:
      abs_change = (noise - previous_noise).abs()
      max_abs_change = abs_change.max().item()
      if max_abs_change < 1e-4:  # 变化幅度小于阈值
        consecutive_steps_without_change += 1
        if consecutive_steps_without_change > max_consecutive_steps_without_change:
          converged = True
          break
      else:
        consecutive_steps_without_change = 0
    
    previous_noise = noise.detach().clone()
  
  # 打印最终状态信息
  total_time = time.time() - start_time
  if converged:
    if best_score >= target_score:
      verboseprint(f'成功达到目标分数 {target_score:.2f}，在步骤 {i_step+1} 停止 (用时 {total_time:.1f}秒)')
    else:
      verboseprint(f'收敛 (步数={i_step+1}, 用时 {total_time:.1f}秒)')
  else:
    verboseprint(f'达到最大步数 (步数={i_step+1}, 用时 {total_time:.1f}秒)')
  
  # 使用最佳噪声生成最终图像
  verboseprint(f"使用最佳噪声（分数={best_score:.4f}）生成最终图像...")
  final_image = noise_to_image(best_noise, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale)
  
  # 量化强度
  final_image = (final_image.detach() * 255.0).round() / 255.0
  
  # 用量化后的图像评估最终对抗性分数
  _, hard_controversiality_score, final_info = controversiality_score(
      final_image, model_1, model_2, class_1, class_2, verbose=False
  )
  hard_controversiality_score = hard_controversiality_score.detach().cpu().numpy().tolist()
  
  # 打印最终结果
  verboseprint('\n最终结果:')
  verboseprint('{}: {} {:>7.2%}, {} {:>7.2%}'.format(
        BOLD+model_1.model_name+normal, short_class_1_name, final_info['p(class_1|model_1)'].mean(), 
        short_class_2_name, final_info['p(class_2|model_1)'].mean()))
  verboseprint('{}: {} {:>7.2%}, {} {:>7.2%}'.format(
        BOLD+model_2.model_name+normal, short_class_1_name, final_info['p(class_1|model_2)'].mean(), 
        short_class_2_name, final_info['p(class_2|model_2)'].mean()))
  verboseprint('对抗性分数: ' + ', '.join('{:0.2f}'.format(f) for f in hard_controversiality_score))
  
  if return_PIL_images:
    numpy_controversial_stimuli = final_image.detach().cpu().numpy().transpose([0, 2, 3, 1])  # NCHW -> NHWC
    numpy_controversial_stimuli = (numpy_controversial_stimuli * 255.0).astype(np.uint8)
    PIL_controversial_stimuli = []
    for i in range(len(numpy_controversial_stimuli)):
      PIL_controversial_stimuli.append(Image.fromarray(numpy_controversial_stimuli[i]))
    return final_image.detach(), PIL_controversial_stimuli, hard_controversiality_score
  else:
    return final_image.detach(), hard_controversiality_score 