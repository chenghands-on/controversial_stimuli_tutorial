import argparse

import itertools
import random
import requests
import pathlib, os
import time
import shutil
import traceback

import torch
import lucent.optvis as ov
import lucent.optvis.transform
import lucent.optvis.param

import models
from optimize import optimize_controversial_stimuli, optimize_controversial_stimuli_with_lucent, make_GANparam
# 只从optimize_functions.py导入我们需要的函数
from optimize_functions import optimize_controversial_stimuli_with_diffusion_pixel, optimize_controversial_stimuli_with_diffusion_latent, optimize_controversial_stimuli_with_diffusion_noise

# 添加无条件生成的自定义提示词处理器
def empty_prompt_processor(tokenizer, text_encoder, device):
    """创建空提示词的处理器，实现无条件生成"""
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

def cluster_check_if_exists(target_fpath,max_synthesis_time=60*10,verbose=True):
    """ returns true if target_fpath does NOT exist and there are no associated flag files.
    This is a simple way to coordinate multiple processes/nodes through a shared filesystem (e.g., as in SLURM).

    Either one of two flag files may cause this function to return True:
        [target_fpath].in_process_flag, which communicates that another worker is now working on this file.
        [target_fpath].failed_flag, which communicates that previous attempt at this file has failed.
    Args:
        target_fpath (str): the filename to be produced.
        max_synthesis_time (int/float): delete in-process file if it is older than this number of seconds.
        verbose (boolean)
    """

    file=pathlib.Path(target_fpath)
    verboseprint = print if verbose else lambda *a, **k: None # https://stackoverflow.com/a/5980173

    # make sure target folder is there
    assert file.parent.exists(), str(file.parent) + " must exist and be reachable."

    if file.exists():
        verboseprint(target_fpath+" found, skipping.")
        return True

    in_process_flag=pathlib.Path(target_fpath + '.in_process_flag')

    if in_process_flag.exists(): # check if other cluster worker is currently optimizing this file
        # but is it too old?
        flag_age=time.time()-in_process_flag.stat().st_mtime
        if flag_age < max_synthesis_time: # no, it's fresh.
            verboseprint('a fresh '+ str(in_process_flag)+ " found, skipping.")
            return True
        else: # old flag. might be have been left by killed workers.
            verboseprint(str(in_process_flag)+ " found, but it is old.")
            try:
                in_process_flag.unlink()
            except:
                pass
            # now, try again.
            return cluster_check_if_exists(target_fpath,max_synthesis_time=max_synthesis_time,verbose=verbose)
    else: # no in-process flag.
        # wait a little bit to avoid racing with other cluster workers that might have started in the same time.
        random.seed()
        time.sleep(random.uniform(0,1))
        if in_process_flag.exists():
            verboseprint(str(in_process_flag) + " appeared after double checking, skipping.")
            return True
        try:
            in_process_flag.touch(mode=0o777, exist_ok=False)
        except:
            return True
        return False

def cluster_check_if_failed(target_fpath):
    failure_flag=pathlib.Path(target_fpath + '.failed_flag')
    return failure_flag.exists()

def remove_in_process_flag(target_fpath):
    in_process_flag=pathlib.Path(target_fpath + '.in_process_flag')
    in_process_flag.unlink()

def leave_failure_flag(target_fpath):
    failure_flag=pathlib.Path(target_fpath + '.failed_flag')
    failure_flag.touch(mode=0o777, exist_ok=False)

def prepare_optimization_parameters(optim_method):
    """ return preset optimization parameters"""

    optimization_kwd={'readout_type':'logsoftmax',
               'pytorch_optimizer':'Adam',
               'optimizer_kwargs':{'lr':5e-2,'betas':(0.9, 0.999),'weight_decay':0,'eps':1e-8},
               'return_PIL_images':True,
               'verbose':True}

    if optim_method=='direct':
        optim_func=optimize_controversial_stimuli
        optimization_kwd.update({'im_size':(1,3,256,256)})
    elif optim_method=='diffusion':
        # 使用像素空间优化方法代替原始函数
        optim_func=optimize_controversial_stimuli_with_diffusion_pixel
        optimization_kwd.update({
            'latent_size':(1, 4, 64, 64),
            'im_size':(1, 3, 224, 224),
            'optimizer_kwargs': {'lr': 5e-2, 'betas': (0.9, 0.999), 'eps': 1e-8},
            'hf_cache_dir': '/mnt/data/chenghan/huggingface_cache',
            'model_id': 'OFA-Sys/small-stable-diffusion-v0'  # 使用小型Stable Diffusion模型，约1GB
        })
    elif optim_method=='diffusion_pixel':
        optim_func=optimize_controversial_stimuli_with_diffusion_pixel
        optimization_kwd.update({
            'latent_size':(1, 4, 64, 64),
            'im_size':(1, 3, 224, 224),
            'optimizer_kwargs': {'lr': 5e-2, 'betas': (0.9, 0.999), 'eps': 1e-8},
            'hf_cache_dir': '/mnt/data/chenghan/huggingface_cache',
            'model_id': 'OFA-Sys/small-stable-diffusion-v0'  # 使用小型Stable Diffusion模型，约1GB
        })
    elif optim_method=='diffusion_latent':
        optim_func=optimize_controversial_stimuli_with_diffusion_latent
        optimization_kwd.update({
            'latent_size':(1, 4, 64, 64),
            'im_size':(1, 3, 224, 224),
            'optimizer_kwargs': {'lr': 1e-1, 'betas': (0.9, 0.999), 'eps': 1e-8},
            'hf_cache_dir': '/mnt/data/chenghan/huggingface_cache',
            'model_id': 'OFA-Sys/small-stable-diffusion-v0'  # 使用小型Stable Diffusion模型，约1GB
        })
    elif optim_method=='diffusion_noise':
        optim_func=optimize_controversial_stimuli_with_diffusion_noise
        optimization_kwd.update({
            'noise_size':(1, 4, 64, 64),
            'im_size':(1, 3, 224, 224),
            'optimizer_kwargs': {'lr': 1e-1, 'betas': (0.9, 0.999), 'eps': 1e-8},
            'hf_cache_dir': '/mnt/data/chenghan/huggingface_cache',
            'model_id': 'OFA-Sys/small-stable-diffusion-v0',  # 使用小型Stable Diffusion模型，约1GB
            'num_inference_steps': 30,  # 减少推理步骤数以降低内存使用
        })
    elif optim_method=='diffusion_noise_unconditional':
        # 添加无条件生成的方法
        optim_func=optimize_controversial_stimuli_with_diffusion_noise
        optimization_kwd.update({
            'noise_size':(1, 4, 64, 64),
            'im_size':(1, 3, 224, 224),
            'optimizer_kwargs': {'lr': 1e-1, 'betas': (0.9, 0.999), 'eps': 1e-8},
            'hf_cache_dir': '/mnt/data/chenghan/huggingface_cache',
            'model_id': 'OFA-Sys/small-stable-diffusion-v0',  # 使用小型Stable Diffusion模型，约1GB
            'num_inference_steps': 50,  # 减少推理步骤数以降低内存使用
            'custom_prompt_processor': empty_prompt_processor,  # 使用自定义提示词处理器
        })
    else: # indirect optimization, define param_f and transforms
        optim_func=optimize_controversial_stimuli_with_lucent
        if optim_method=='jittered':
            param_f= lambda: ov.param.image(w=256, h=256, batch=1, decorrelate=False,fft=False, sd=1) # this creates a pixel representation and an initial image that are both similar to those we have used in part 1.
            transforms=[ov.transform.jitter(25)] # a considerable spatial jitter. use transforms=[] for no transforms.
        elif optim_method=='decorrelated':
            param_f= lambda: ov.param.image(w=256, h=256, batch=1, decorrelate=True,fft=True)
            transforms=[ov.transform.jitter(25)]
        elif optim_method=='CPPN':
            param_f = lambda: ov.param.cppn(256)
            transforms=[ov.transform.jitter(5)]
            optimization_kwd['optimizer_kwargs'].update({'lr':5e-3,'eps':1e-3}) # CPPN specific Adam parameters.

        elif optim_method in ['GAN-pool5','GAN-fc6','GAN-fc7','GAN-fc8']:
            GAN_latent_layer=optim_method.split('-')[-1]
            param_f=make_GANparam(batch=1, sd=1,latent_layer=GAN_latent_layer)
            transforms=[ov.transform.jitter(5)]
        else:
            raise ValueError('Unknown optim_method '+optim_method)
        optimization_kwd.update({'param_f':param_f,'transforms':transforms})
    return optim_func, optimization_kwd

def design_synthesis_experiment(exp_name):
        # build particular model and class combinations for the tutorial figures.

        # get imagenet categories
        imagenet_dict_url='https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt'
        class_dict=eval(requests.get(imagenet_dict_url).text)

        if exp_name=='8_random_classes':
            # 修改此处，使用ViTL16和ResNeXt101_32x32d作为模型对
            model_pairs=[['ViTL16','ResNeXt101_32x32d']]
            n_classes=8
            all_class_idx=list(class_dict.keys())
            random.seed(1)
            class_idx=list(random.sample(all_class_idx,n_classes))
            
            # 生成所有类别的排列（而非组合），但排除相同类别的情况
            class_idx_pairs=[]
            for i in range(n_classes):
                for j in range(n_classes):
                    if i != j:  # 排除相同类别对
                        class_idx_pairs.append([class_idx[i], class_idx[j]])
            
            print(f"共生成了 {len(class_idx_pairs)} 个类别对（排列，排除相同类别）")
            print(f"理论数量: 8*7 = {8*7}")
            for idx, pair in enumerate(class_idx_pairs[:5]):
                print(f"类别对{idx+1}: {class_dict[pair[0]]} vs {class_dict[pair[1]]}")
            if len(class_idx_pairs) > 5:
                print("... 仅展示前5个类别对 ...")
                
        elif exp_name=='cat_vs_dog':
            model_names=['Resnet50','InceptionV3','Resnet_50_l2_eps5','Wide_Resnet50_2_l2_eps5', 'ViTB16', 'ViTL16', 'ResNeXt101_32x32d']
            model_pairs=itertools.product(model_names,repeat=2)
            model_pairs=[pair for pair in model_pairs if pair[0]!=pair[1]]
            class_idx_pairs=[[283,178]]

        #indecis to classes
        class_pairs=[[class_dict[idx] for idx in idx_pair] for idx_pair in class_idx_pairs]
        return model_pairs, class_pairs

def batch_optimize(target_folder,model_pairs,class_pairs,optim_method,min_controversiality=0.85,max_seeds_to_try=5,max_steps=1000, verbose=True):
    """ Synthesize a batch of controversial stimuli.
    For each model pair, synthesizes a controversial stimuli for all class pairs.

    Args:
    target_folder (str): Where the image are saved.
    model_pairs (list): A list of tuples, each tuple containing two model names.
    class_pairs (list): A list of tuples, each tuple containing two class names.
    optim_method (str): direct/jittered/decorrelated/CPPN/'GAN-pool5'/'GAN-fc6'/'GAN-fc7'/'GAN-fc8'
    min_controversiality (float): minimum controversiality required for saving an image (e.g., 0.85).
    max_seeds_to_try (int): 最大尝试种子数量
    max_steps (int): 最大优化步数
    verbose (boolean): 是否打印详细信息

    returns True if one or more synthesized images was not save due to insufficient controversiality.
    """

    verboseprint = print if verbose else lambda *a, **k: None # https://stackoverflow.com/a/5980173

    if torch.cuda.device_count()>1:
        model_1_device='cuda:0'
        model_2_device='cuda:1'
        print('using two GPUs, one per model.')
    elif torch.cuda.device_count()==1:
        model_1_device=model_2_device='cuda:0'
        print('using one GPU.')
    else:
        model_1_device=model_2_device='cpu'
        print('using CPU')

    pathlib.Path(target_folder).mkdir(parents=True, exist_ok=True)
    at_least_one_synthesis_failed=False

    # 只使用一个种子：seed=1
    fixed_seeds = [1]
    
    for model_1_name, model_2_name in model_pairs:
        models_loaded=False
        for class_pair in class_pairs:
            class_1_name, class_2_name = class_pair
            # build filename
            short_class_1_name=class_1_name.split(',',1)[0].replace(' ','_')
            short_class_2_name=class_2_name.split(',',1)[0].replace(' ','_')

            # 对每个固定的种子值进行尝试
            for seed in fixed_seeds:
                target_fname='{}-{}_vs_{}-{}_seed{}.png'.format(model_1_name,short_class_1_name,model_2_name,short_class_2_name,seed)
                target_fpath=os.path.join(target_folder,target_fname)
                
                # 删除可能存在的失败标志，以确保每次运行都会尝试所有种子
                failure_flag=pathlib.Path(target_fpath + '.failed_flag')
                if failure_flag.exists():
                    try:
                        failure_flag.unlink()
                        print(f"删除了失败标志: {failure_flag}")
                    except:
                        pass

                # check if png file already exists and leave 'in-process' flag
                if cluster_check_if_exists(target_fpath):
                    verboseprint("Skipping "+target_fpath)
                    continue

                print('Synthesizing '+target_fpath)
                if not models_loaded:
                    model_1=getattr(models,model_1_name)()
                    model_2=getattr(models,model_2_name)()
                    model_1.load(model_1_device)
                    model_2.load(model_2_device)
                    models_loaded=True

                optim_func, optimization_kwd=prepare_optimization_parameters(optim_method)
                optimization_kwd['random_seed']=seed
                optimization_kwd['max_steps']=max_steps
                # run optimization
                _,PIL_ims,controversiality_scores=optim_func(model_1,model_2,class_1_name,class_2_name,**optimization_kwd)
                remove_in_process_flag(target_fpath)

                # 无论对抗性分数多低都保存图片
                PIL_ims[0].save(target_fpath)
                print('saved '+target_fpath)
                print(f'对抗性分数: {controversiality_scores[0]:.4f}')
                
                # 只在日志中记录分数是否达到了阈值，但不影响保存
                if class_1_name != class_2_name and controversiality_scores[0] < min_controversiality:
                    print(f'注意: 对抗性分数 ({controversiality_scores[0]:.4f}) 低于阈值 ({min_controversiality:.4f})')
                    at_least_one_synthesis_failed=True
                    # 不再使用失败标志，因为我们总是会保存图片
                    # leave_failure_flag(target_fpath)
    return at_least_one_synthesis_failed

def cleanup_home_hf_cache():
    """Cleanup Hugging Face cache from home directory to free up space"""
    home_cache_dir = os.path.expanduser("~/.cache/huggingface")
    if os.path.exists(home_cache_dir):
        print(f"Cleaning up Hugging Face cache from {home_cache_dir}")
        shutil.rmtree(home_cache_dir, ignore_errors=True)
        print("Cache cleanup completed")

def grand_batch(experiments,optimization_methods,target_folder='optimization_results',min_controversiality=0.85,max_steps=1000):
    # Clean up home directory HF cache if any diffusion method is in the optimization methods
    diffusion_methods = ['diffusion', 'diffusion_pixel', 'diffusion_latent', 'diffusion_noise']
    if any(method in optimization_methods for method in diffusion_methods):
        cleanup_home_hf_cache()
    
    # 添加版本号，确保每次运行使用新的文件夹
    version = "v4"
    
    task_list=[]
    for exp_name in experiments:
        for optim_method in optimization_methods:
            # 添加版本号到文件夹名称
            target_subfolder=os.path.join(target_folder,optim_method+'_optim_'+exp_name+'_'+version)
            task_list.append({'target_subfolder':target_subfolder,
                             'exp_name':exp_name,
                             'optim_method':optim_method})
    while len(task_list)>0:
        cur_task=task_list.pop(0)
        model_pairs, class_pairs=design_synthesis_experiment(cur_task['exp_name'])
        at_least_one_synthesis_failed=batch_optimize(cur_task['target_subfolder'],
            model_pairs,class_pairs,cur_task['optim_method'],
            min_controversiality=min_controversiality,max_seeds_to_try=5,max_steps=max_steps)
        if at_least_one_synthesis_failed:
            task_list.append(cur_task)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='batch_optimize',
        epilog='To prepare images for the first controversial stimuli matrix in the tutorial, run "batch_optimize --experiments cat_vs_dog --optimization_methods direct". To prepare all images for all figures, run without --experiments and --optimization_methods arguments. This requires a cluster, or a lot of patience. This program can be run in parallel by multiple nodes sharing a filesystem (as in SLURM).')
    parser.add_argument(
        "--experiments", nargs='+',
        choices= ['cat_vs_dog', '8_random_classes'],
        default= ['cat_vs_dog', '8_random_classes'],
        required=False)
    parser.add_argument(
        "--optimization_methods",nargs='+',
        choices= ['direct','jittered','decorrelated','CPPN','GAN-pool5','GAN-fc6','GAN-fc7','GAN-fc8','diffusion','diffusion_pixel','diffusion_latent','diffusion_noise','diffusion_noise_unconditional'],
        default= ['direct','jittered','decorrelated','CPPN','GAN-pool5','GAN-fc6','GAN-fc7','GAN-fc8'])
    parser.add_argument(
        "--target_folder", type=str, default="optimization_results")
    parser.add_argument(
        "--max_steps", type=int, default=1000)
    parser.add_argument(
        "--min_controversiality", type=float, default=0.85)
    args=parser.parse_args()
    
    # 处理参数
    experiments = args.experiments
    optimization_methods = args.optimization_methods
    target_folder = args.target_folder
    max_steps = args.max_steps
    min_controversiality = args.min_controversiality
    
    grand_batch(
        experiments=experiments,
        optimization_methods=optimization_methods,
        target_folder=target_folder,
        min_controversiality=min_controversiality,
        max_steps=max_steps
    )