import requests
from io import BytesIO
import math
import time
import os

import numpy as np
import torch # tested with PyTorch 1.6
import torchvision as tv # we use torchvision for some pre-trained deep nets. tested with torchvision 0.7
from PIL import Image # to load images
from IPython.display import display # to display images
import colored

import lucent.optvis as ov
import lucent.optvis.transform
import lucent.optvis.param

from diffusers import StableDiffusionPipeline, DiffusionPipeline, DDPMPipeline, DDIMPipeline, LDMPipeline

# 添加版本号，确保每次运行使用新的文件夹
version = "v2"

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

def optimize_controversial_stimuli(model_1,model_2,class_1,class_2,im_size=(4,3,256,256),
                                   pytorch_optimizer='Adam',optimizer_kwargs={'lr':5e-2,'betas':(0.9, 0.999),'weight_decay':0,'eps':1e-8},
                                   readout_type='logsoftmax',random_seed=0,
                                   max_steps=1000,max_consecutive_steps_without_pixel_change=10,
                                   return_PIL_images=True,verbose=True):
  """Optimize controversial stimuli with respect to two models and two classes.

  This function synthesizes controversial stimuli in pixel space such that model 1 detects class 1 (but not class 2) with high-confidence and model 2 detects class 2 (but not class 1) with high-confidence.

  Args:
  model_1, model2 (object): model objects, such as TVPretrainedModel (see above). Note that unlike standard torchvision model objects, we assume that the models receive unnormalized images.
  class_1, class_2 (str): target class names.
  im_size (tuple): Specify the optimized image tensor size as (N,C,H,W). If N is greater than 1, multiple images are optimized in parallel. Note that optimizing batches might result in different convergence-based stopping compared to optimizing one image at a time.
  pytorch_optimizer (str or class): either the name of a torch.optim class or an optimizer class.
  optimizer_kwargs (dict): keywords passed to the optimizer
  readout_type (str): 'logits' for models with sigmoid readout, 'logsoftmax' for models with softmax readout.
  random_seed (int): sets the random seed for PyTorch.
  max_steps (int): maximal number of optimization steps
  max_consecutive_steps_without_pixel_change (int): if the image hasn't changed for this number of steps, stop
  return_PIL_images (boolean): if True (default), return also a list of PIL images.
  verbose (boolean): if True (default), shows image probabilities (averaged across images if a batch is provided) and other diagnostic messages.

  Returns:
  (tuple): tuple containing:
    im_tensor (torch.Tensor): Controversial stimuli image tensor in nchw format.
    PIL_controversial_stimuli (list): Controversial stimuli as list of PIL.Image images.
    hard_controversiality_score: (list): Controversiality score for each image as float.
  or (if return_PIL_images == False):
    im_tensor (torch.Tensor): Controversial stimuli image tensor in nchw format.
    hard_controversiality_score: (list): Controversiality score for each image as float.
  """

  verboseprint = print if verbose else lambda *a, **k: None # https://stackoverflow.com/a/5980173

  # used for display:
  short_class_1_name=class_1.split(',',1)[0]
  short_class_2_name=class_2.split(',',1)[0]
  BOLD=colored.attr('bold')
  normal=colored.attr('reset')

  # define initial image(s)
  torch.manual_seed(random_seed)
  np.random.seed(random_seed)

  initial_im=torch.rand(im_size) # a standard float NHWC image (intensity in [0,1]). The magnitude of the noise can be reduced.

  # To smoothly enforce intensity limit constraints (i.e., keep the optimized image pixel intensities in [0,1]),
  # we use a constraining through parametrization approach. The optimized variable (x) is an image of unbounded intensity values ((-inf,+inf)).
  # It is squeezed to the [0,1] range by a sigmoid before being fed to the model.

  # First, we use the *inverse sigmoid* to stretch the initial image:
  inverse_sigmoid=lambda p : torch.log(p/(1-p))
  z=inverse_sigmoid(initial_im)
  z.requires_grad=True

  # initialize image optimizer
  if isinstance(pytorch_optimizer, str):
    OptimClass=getattr(torch.optim,pytorch_optimizer)
  else:
    OptimClass=pytorch_optimizer
  optimizer = OptimClass(params=[z], **optimizer_kwargs)

  previous_im=None

  alpha=100.0

  converged=False
  consecutive_steps_without_pixel_change=0

  for i_step in range(max_steps):

    optimizer.zero_grad()
    x=torch.sigmoid(z) # compress x back to [0,1] so it's a real image.
    smooth_controversiality_score, hard_controversiality_score, info=controversiality_score(
        x,model_1,model_2,class_1,class_2,alpha=alpha,
        readout_type=readout_type,verbose=verbose)

    loss=-smooth_controversiality_score # we would like to MAXIMIZE controversiality, therefore the minus.
    loss=loss.sum() # when multiple stimuli are optimized, make the loss scalar by summation
    loss.backward()
    optimizer.step()

    verboseprint('{}: {} {:>7.2%}, {} {:>7.2%} │ {}: {} {:>7.2%}, {} {:>7.2%} │ {}:loss={:3.2e}'.format(
          BOLD+model_1.model_name+normal,short_class_1_name,info['p(class_1|model_1)'].mean(),short_class_2_name,info['p(class_2|model_1)'].mean(),
          BOLD+model_2.model_name+normal,short_class_1_name,info['p(class_1|model_2)'].mean(),short_class_2_name,info['p(class_2|model_2)'].mean(),
          i_step,loss.item()))

    # monitor the magnitude of image change.
    if previous_im is not None:
        abs_change=(x-previous_im).abs()*255.0 # change on a 0-255 intesity scale
        max_abs_change=abs_change.max().item()
        if (max_abs_change)<0.5: # check if the maximal absolute change across pixels is less than half an intesity level.
          consecutive_steps_without_pixel_change+=1
          if consecutive_steps_without_pixel_change>max_consecutive_steps_without_pixel_change:
            converged=True
            break
        else:
          consecutive_steps_without_pixel_change=0

    previous_im=x.detach().clone()

  if converged:
    verboseprint('converged (n_steps={})'.format(i_step+1))
  else:
    verboseprint('max steps achieved (n_steps={})'.format(i_step+1))

  # Quantize intesity. Since we plan to show these images to humans, we don't want to take into account intensity levels that cannot be displayed.
  x=(x.detach()*255.0).round()/255.0

  # Evaluate final controversiality score, using the quantized image.
  _,hard_controversiality_score,_=controversiality_score(x,model_1,model_2,class_1,class_2,verbose=False)
  hard_controversiality_score=hard_controversiality_score.detach().cpu().numpy().tolist() # convert a vector tensor to list of floats

  verboseprint('controversiality score: '+', '.join('{:0.2f}'.format(f) for f in hard_controversiality_score))

  if return_PIL_images:
    numpy_controversial_stimuli=x.detach().cpu().numpy().transpose([0,2,3,1]) # NCHW -> NHWC
    numpy_controversial_stimuli=(numpy_controversial_stimuli*255.0).astype(np.uint8)
    PIL_controversial_stimuli=[]
    for i in range(len(numpy_controversial_stimuli)):
      PIL_controversial_stimuli.append(Image.fromarray(numpy_controversial_stimuli[i]))
    return x.detach(), PIL_controversial_stimuli, hard_controversiality_score
  else:
    return x.detach(), hard_controversiality_score

def optimize_controversial_stimuli_with_lucent(model_1,model_2,class_1,class_2,transforms,param_f,
                                               pytorch_optimizer='Adam',optimizer_kwargs={'lr':5e-2,'betas':(0.9, 0.999),'weight_decay':0,'eps':1e-8},
                                               readout_type='logsoftmax',random_seed=0,
                                               max_steps=1000,max_consecutive_steps_without_pixel_change=10,
                                               return_PIL_images=True,verbose=True):
  """Optimize controversial stimuli with respect to two models and two classes using Lucent image parameterizations and transformations.

  This function synthesizes controversial stimuli such that model 1 detects class 1 (but not class 2) with high-confidence and model 2 detects class 2 (but not class 1) with high-confidence.
  It parametrizes the image according to param_f, and stochastically transform it according to transforms.  This function is essentially a crossover between optimize_controversial_stimuli()
  and lucent.optviz.renderer(). The latter is built for a single model, so it less appropriate for controversial stimuli.

  Args:
  model_1, model2 (object): model objects, such as TVPretrainedModel (see above). Note that unlike standard torchvision model objects, we assume that the models receive unnormalized images.
  class_1, class_2 (str): target class names.
  param_f (function): a function with no arguments that returns a tuple with two elements:
    params - parameters to update (these are passed to the optimizer)
    image_f - a function withh no arguments that returns an image as a tensor (using an enclosing-function scope access to params).
  transforms (list): a list of lucent.optvis.transform transformation (pass [] for no transformations).
  pytorch_optimizer (str or class): either the name of a torch.optim class or an optimizer class.
  optimizer_kwargs (dict): keywords passed to the optimizer
  readout_type (str): 'logits' for models with sigmoid readout, 'logsoftmax' for models with softmax readout.
  random_seed (int): sets the random seed for PyTorch.
  max_steps (int): maximal number of optimization steps
  max_consecutive_steps_without_pixel_change (int): if the image hasn't changed for this number of steps, stop
  return_PIL_images (boolean): if True (default), return also a list of PIL images.
  verbose (boolean): if True (default), shows image probabilities (averaged across images if a batch is provided) and other diagnostic messages.

  There is no image size argument since the image size is determined by param_f.

  Returns:
  (tuple): tuple containing:
    im_tensor (torch.Tensor): Controversial stimuli image tensor in nchw format.
    PIL_controversial_stimuli (list): Controversial stimuli as list of PIL.Image images.
    hard_controversiality_score: (list): Controversiality score for each image as float.
  or (if return_PIL_images == False):
    im_tensor (torch.Tensor): Controversial stimuli image tensor in nchw format.
    hard_controversiality_score: (list): Controversiality score for each image as float.
  """

  verboseprint = print if verbose else lambda *a, **k: None # https://stackoverflow.com/a/5980173

  # used for display:
  short_class_1_name=class_1.split(',',1)[0]
  short_class_2_name=class_2.split(',',1)[0]
  BOLD=colored.attr('bold')
  normal=colored.attr('reset')

  torch.manual_seed(random_seed)
  np.random.seed(random_seed)

  lucent_transform_device=ov.transform.device

  # adapted from lucent.optviz.render
  params, image_f = param_f()
  params=list(params)

  # compose the list of transforms into a single transform.
  transform_f = ov.transform.compose(transforms)

  # initialize image optimizer
  if isinstance(pytorch_optimizer, str):
    OptimClass=getattr(torch.optim,pytorch_optimizer)
  else:
    OptimClass=pytorch_optimizer
  optimizer = OptimClass(params=params, **optimizer_kwargs)

  previous_im=None

  alpha=100.0

  converged=False
  consecutive_steps_without_pixel_change=0

  for i_step in range(max_steps):

    optimizer.zero_grad()
    x=image_f() # convert params to an image tensor

    # apply stochastic transformations
    transformed_im=transform_f(x.to(lucent_transform_device)) # apply stochastic transformations

    # calculate controversiality score using the transformed image
    smooth_controversiality_score, hard_controversiality_score,info=controversiality_score(
        transformed_im,model_1,model_2,class_1,class_2,alpha=alpha,readout_type=readout_type,verbose=verbose)

    loss=-smooth_controversiality_score # we would like to MAXIMIZE controversiality.
    loss=loss.sum() # when multiple stimuli are optimized, make the loss scalar by summation
    loss.backward()
    optimizer.step()

    verboseprint('{}: {} {:>7.2%}, {} {:>7.2%} │ {}: {} {:>7.2%}, {} {:>7.2%} │ {}:loss={:3.2e}'.format(
          BOLD+model_1.model_name+normal,short_class_1_name,info['p(class_1|model_1)'].mean(),short_class_2_name,info['p(class_2|model_1)'].mean(),
          BOLD+model_2.model_name+normal,short_class_1_name,info['p(class_1|model_2)'].mean(),short_class_2_name,info['p(class_2|model_2)'].mean(),
          i_step,loss.item()))

    # monitor the magnitude of image change.
    if previous_im is not None:
      abs_change=(x-previous_im).abs()*255.0 # change on a 0-255 intesity scale
      max_abs_change=abs_change.max().item()
      if (max_abs_change)<0.5: # check if the maximal absolute change across pixels is less than half an intesity level.
        consecutive_steps_without_pixel_change+=1
        if consecutive_steps_without_pixel_change>max_consecutive_steps_without_pixel_change:
          converged=True
          break
      else:
        consecutive_steps_without_pixel_change=0

    previous_im=x.detach().clone()

  if converged:
    verboseprint('converged (n_steps={})'.format(i_step+1))
  else:
    verboseprint('max steps achieved (n_steps={})'.format(i_step+1))

  # Quantize intesity. Since we plan to show these images to humans, we don't want to take into account intensity levels that cannot be displayed.
  x=(x.detach()*255.0).round()/255.0

  # Evaluate final controversiality score, using the quantized image. Note that this image is not stochastically transformed.
  _,hard_controversiality_score,_=controversiality_score(x,model_1,model_2,class_1,class_2,verbose=False)
  hard_controversiality_score=hard_controversiality_score.detach().cpu().numpy().tolist() # convert a vector tensor to list of floats

  verboseprint('controversiality score: '+', '.join('{:0.2f}'.format(f) for f in hard_controversiality_score))

  if return_PIL_images:
    numpy_controversial_stimuli=x.detach().cpu().numpy().transpose([0,2,3,1]) # NCHW -> NHWC
    numpy_controversial_stimuli=(numpy_controversial_stimuli*255.0).astype(np.uint8)
    PIL_controversial_stimuli=[]
    for i in range(len(numpy_controversial_stimuli)):
      PIL_controversial_stimuli.append(Image.fromarray(numpy_controversial_stimuli[i]))
    return x.detach(), PIL_controversial_stimuli, hard_controversiality_score
  else:
    return x.detach(), hard_controversiality_score

def optimize_controversial_stimuli_with_diffusion(model_1, model_2, class_1, class_2,
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

  This function synthesizes controversial stimuli by optimizing in the latent space of a diffusion model.
  
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
    # 根据模型类型使用不同的解码方法
    with torch.no_grad():
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
    loss.backward()
    
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
  with torch.no_grad():
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

def make_GANparam(batch=1, sd=1,latent_layer='pool5',gan_device=None):
  """ generate param_f for an alexnet-latent-representation inverting GAN.
      Code adapted from:
      https://github.com/greentfrapp/lucent/blob/master/lucent/optvis/param/gan.py
      https://github.com/greentfrapp/lucent/blob/ea2cc9e17cf88caa9cfb84b7124090bf365256fd/tests/optvis/param/test_gan.py (Binxu Wang)
   """
  def GANparam():
      device=gan_device
      if device is None:
          device = "cuda:0" if torch.cuda.is_available() else "cpu"
      device = torch.device(device)
      G = ov.param.upconvGAN(latent_layer).to(device)
      if latent_layer=='pool5':
        code = (torch.randn((batch, G.codelen, 6, 6)) * sd).to(device).requires_grad_(True)
      else:
        code = (torch.randn((batch, G.codelen)) * sd).to(device).requires_grad_(True)
      imagef = lambda: G.visualize(code)
      return [code], imagef
  return GANparam