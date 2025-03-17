#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 10 09:45:06 2018

@author: tal
"""

import sys,os,pathlib,re
import glob
import warnings
import argparse

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
import PIL

import seaborn as sns
import pandas as pd
from tqdm import tqdm
from attrdict import AttrDict

from plotting_utils import get_png_file_info
from third_party.curlyBrace import curlyBrace

# map ugly model names to nice model names
model_name_dict={'InceptionV3':'Inception-v3',
'Resnet50':'ResNet-50',
'Resnet_50_l2_eps5':'$\ell_2$-adv-trained (${\epsilon=5}$)\nResNet-50\n',
'Wide_Resnet50_2_l2_eps5':'$\ell_2$-adv-trained (${\epsilon=5}$)\nWRN-50-2',
'ViTB16':'ViT-B/16',
'ViTL16':'ViT-L/16',
'ResNeXt101_32x32d':'ResNeXt-101 (32x32d)'}

def plot_im_matrix(im_matrix,x_class,y_class,c,subplot_spec=None,cmap_matrix=None,upscale=None,exp_type='imagenet'):

    n_models=len(models_to_plot)
    # c is a configuration dict
    if subplot_spec is None:
        #   start figure
        figsize=[np.sum([c.left_margin,c.inch_per_image*n_models,c.right_margin]),
                  np.sum([c.top_margin,c.inch_per_image*n_models,c.bottom_margin])]
        print('figure size=',figsize,'inches')
        fig=plt.figure(figsize=figsize)

        gs0 = gridspec.GridSpec(nrows=3, ncols=3,
                        height_ratios=[c.top_margin,c.inch_per_image*n_models,c.bottom_margin],
                        width_ratios=[c.left_margin,c.inch_per_image*n_models,c.right_margin],
                        figure=fig,wspace=0,hspace=0,left=0,right=1,bottom=0,top=1)
        
        # 添加总标题，减小字体大小并修改内容
        # fig.suptitle('Controversial Stimuli from Unconditional Diffusion Model', fontsize=12, y=0.98)
    else:
        fig=plt.gcf()
        gs0 = gridspec.GridSpecFromSubplotSpec(nrows=3, ncols=3,
                        height_ratios=[c.top_margin,c.inch_per_image*n_models,c.bottom_margin],
                        width_ratios=[c.left_margin,c.inch_per_image*n_models,c.right_margin],
                        subplot_spec=subplot_spec,wspace=0,hspace=0)

    if c.do_plot_images:
        gs00 = gridspec.GridSpecFromSubplotSpec(nrows=n_models, ncols=n_models, subplot_spec=gs0[1,1],hspace=0.0,wspace=0.0)
        for i_row,row_model_name in enumerate(models_to_plot):
           for i_col,col_model_name in enumerate(models_to_plot):
              cur_im=im_matrix[i_row, i_col]
              if cmap_matrix is not None:
                  cmap=cmap_matrix[i_row, i_col]
              else:
                  cmap='gray'
              ax = plt.subplot(gs00[i_row, i_col])
              if type(cur_im) is np.ndarray or isinstance(cur_im,PIL.Image.Image):
                  if exp_type=='MNIST':
                      ax.imshow(1.0-cur_im,cmap=cmap, interpolation='nearest',extent=[0,1,0,1])
                  elif exp_type in ['CIFAR-10','imagenet']:
                      if upscale is not None:
                          cur_im = cur_im.resize((upscale,upscale), resample=PIL.Image.LANCZOS)
                      ax.imshow(cur_im,extent=[0,1,0,1])
                  else:
                      raise NotImplementedError

              elif i_row==i_col: # diagonal line

                  ax.imshow(np.ones([1,1,3]),extent=[0,1,0,1])
                  ax.plot([1, 0], [0, 1], 'k-',linewidth=c.im_matrix_line_width,transform=ax.transAxes)
              elif cur_im == 'blank':
                  ax.imshow(np.ones([1,1,3]),extent=[0,1,0,1])
                  ax.plot([1, 0], [0, 1], 'k-',linewidth=c.im_matrix_line_width,transform=ax.transAxes)
              else: # a missing plot
                  ax.imshow(np.asarray([1.0,0.5,0.5]).reshape([1,1,3]),extent=[0,1,0,1])
                  plt.xlim([0,1])
                  plt.ylim([0,1])

              for spine in ax.spines.values():
                  spine.set_linewidth(c.spine_line_width)
              plt.tick_params(
                   axis='x',          # changes apply to the x-axis
                   which='both',      # both major and minor ticks are affected
                   bottom=False,      # ticks along the bottom edge are off
                   top=False,         # ticks along the top edge are off
                   labelbottom=False) # labels along the bottom edge are off
              plt.tick_params(
                   axis='y',          # changes apply to the x-axis
                   which='both',      # both major and minor ticks are affected
                   left=False,      # ticks along the bottom edge are off
                   right=False,         # ticks along the top edge are off
                   labelleft=False) # labels along the bottom edge are off

              if i_row==0: # is it the top row?
                 if not (hasattr(c,'omit_top_model_labels') and c.omit_top_model_labels):
                     ax.text(0.25,1.05,model_name_to_title(col_model_name),ha='left',clip_on=False,va='bottom',rotation=45,fontdict={'fontsize':c.model_name_font_size})

              if i_col==0: # is it the leftmost column?
                  if not (hasattr(c,'omit_left_model_labels') and c.omit_left_model_labels):
                      ax.text(-0.15,0.5,model_name_to_title(row_model_name),ha='right',clip_on=False,va='center',fontsize=c.model_name_font_size)

              plt.setp(ax.spines.values(), color='black')

              bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
              width, height = bbox.width, bbox.height
              dpi=28/0.04676587926509179

    if c.do_plot_horizontal_brace:
        x_major_label_ax=plt.Subplot(fig,gs0[0,1])
        fig.add_subplot(x_major_label_ax)
        x_major_label_ax.set_axis_off()
        plt.ylim([0,1])
        plt.xlim([0,n_models])
        curlyBrace(fig=plt.gcf(), ax=x_major_label_ax,
                   p2=[n_models-0.25+c.top_right_curly_brace_horizontal_shift,c.h_curly_pad],
                   p1=[0.25+c.top_left_curly_brace_horizontal_shift,c.h_curly_pad],
                   k_r=0.05, bool_auto=True,
                   str_text='models targeted\n to recognize '+r"$\bf{" + str(x_class.replace('_','\>'))+"}$", int_line_num=3,color='black',fontdict={'fontsize':c.major_label_font_size},linewidth=c.curly_brace_line_width,clip_on=False,)

    if c.do_plot_vertical_brace:
        y_major_label_ax=plt.Subplot(fig,gs0[1,0])
        fig.add_subplot(y_major_label_ax)
        y_major_label_ax.set_axis_off()
        plt.ylim([0,n_models])
        plt.xlim([0,1])
      #  y_major_label_ax.text(0.25,0.5,'model targeted\n to recognize {}'.format(y_class),fontdict={'fontsize':c.major_label_font_size},verticalalignment='center',horizontalalignment='left',rotation=90)
        curlyBrace(fig=plt.gcf(), ax=y_major_label_ax, p2=[c.v_curly_pad,n_models-0.5], p1=[c.v_curly_pad,0.5], k_r=0.05, bool_auto=True,
                   str_text='models targeted\n to recognize '+r"$\bf{" + str(y_class.replace('_','\>'))+"}$", int_line_num=3,color='black',fontdict={'fontsize':c.major_label_font_size},linewidth=c.curly_brace_line_width,clip_on=False,nudge_label_x=c.nudge_v_label_x, nudge_label_y=c.nudge_v_label_y)


def model_name_to_title(model_name):
    if model_name in model_name_dict.keys():
        return model_name_dict[model_name]
    else:
        return model_name.replace('_',' ')

def plot_single_model_pair_multiple_class_pairs_controversial_stimuli_matrix(subfolder,rows_model,columns_model,c,stimuli_path,subplot_spec=None,panel_label=None):

    # plot a figure like Figure 3 in Golan, Raju & Kriegeskorte, 2020 PNAS
    png_files=glob.glob(os.path.join(stimuli_path,subfolder,'*.png'))
    print("found {} png files.".format(len(png_files)))
    png_files_info=get_png_file_info(png_files)

    print(png_files_info)

    # some sanity checks
    if len(png_files_info)==0:
       warnings.warn('folder {} is empty'.format(subfolder))
       return
    # make sure the folder doesn't have mixed files (the first model is always one model, and the second is always the other)
    assert len(png_files_info.model_1_name.unique())==1 and len(png_files_info.model_2_name.unique())==1
    model_1_name=png_files_info.model_1_name.unique()[0]
    model_2_name=png_files_info.model_2_name.unique()[0]
    assert ((model_1_name == rows_model) and (model_2_name == columns_model) or
            (model_1_name == columns_model) and (model_2_name == rows_model))

    class_names=np.unique(list(png_files_info.model_1_target)+list(png_files_info.model_2_target))
    n_classes=len(class_names)
    im_matrix=np.empty((n_classes,n_classes), dtype=object)

    for png_file,model_1_target,model_2_target in tqdm(zip(png_files_info.filename,png_files_info.model_1_target,png_files_info.model_2_target)):
        #check response statistics - is it a successful crafting?
        cur_im=plt.imread(png_file)

        model_1_target_idx=class_names.index(model_1_target)
        model_2_target_idx=class_names.index(model_2_target)

        if model_1_name==rows_model and model_2_name==columns_model:
            im_matrix[model_1_target_idx, model_2_target_idx]=cur_im
        else:
            im_matrix[model_2_target_idx, model_1_target_idx]=cur_im

    plot_im_matrix(im_matrix,rows_model,columns_model,c,subplot_spec=subplot_spec,panel_label=panel_label)

def get_subfolders_properties(subfolders):
    list_of_dicts=[]
    import pandas as pd
    for subfolder in subfolders:
        subfolder_m1, subfolder_m2=re.findall(r'([^/]+)_vs_(.+)',subfolder)[0]
        list_of_dicts.append(
                {'name':subfolder,
                 'm1':subfolder_m1,
                 'm2':subfolder_m2,
                 })
    return pd.DataFrame(list_of_dicts)


def plot_single_class_pair_multiple_model_pairs_controversial_stimuli_matrix(x_class,y_class,c,stimuli_path=None,subplot_spec=None,png_files_info=None,upscale=None,seed=0):
    print(models_to_plot)
    if png_files_info is None:
        n_models=len(models_to_plot)
        png_files=glob.glob(os.path.join(stimuli_path,'**/*.png'))+glob.glob(os.path.join(stimuli_path,'*.png'))
        assert len(png_files)>0, "no png files found in "+stimuli_path
        print("found {} png files.".format(len(png_files)))
        png_files_info=get_png_file_info(png_files)

    # filter pngs to match required classes
    cur_subplot_files_mask=np.logical_and(
                np.logical_or(
                        np.logical_and(png_files_info.model_1_target==x_class,png_files_info.model_2_target==y_class),
                        np.logical_and(png_files_info.model_2_target==x_class,png_files_info.model_1_target==y_class)),
                np.logical_and(
                        [model_name in models_to_plot for model_name in png_files_info.model_1_name],
                        [model_name in models_to_plot for model_name in png_files_info.model_2_name]
                        )
                )

    selected_png_files=png_files_info[cur_subplot_files_mask]
    
    # 处理多个种子版本的图像，优先选择指定的seed值
    # 创建一个字典来跟踪每个模型对的最佳图像
    model_pair_to_best_image = {}
    # 使用传入的seed值作为目标种子
    target_seed = seed
    print(f"使用种子值: {target_seed} 的图像进行可视化")
    
    for f in selected_png_files.itertuples():
        # 确定模型对的键
        if f.model_1_target==x_class and f.model_2_target==y_class:
            model_pair_key = (f.model_1_name, f.model_2_name)
        else:
            model_pair_key = (f.model_2_name, f.model_1_name)
        
        # 提取seed值
        match = re.search(r'seed(\d+)', f.filename)
        img_seed = int(match.group(1)) if match else 999
        
        # 如果当前图像的seed是目标seed值，优先选择它
        if img_seed == target_seed:
            model_pair_to_best_image[model_pair_key] = (f, img_seed)
        # 如果这个模型对还没有图像，或者当前的seed不是目标seed但可以作为备选
        elif model_pair_key not in model_pair_to_best_image:
            model_pair_to_best_image[model_pair_key] = (f, img_seed)
    
    # 只保留每个模型对的最佳图像
    best_images = [pair_info[0] for pair_info in model_pair_to_best_image.values()]
    
    im_matrix=np.empty((n_models,n_models), dtype=object)

    for f in best_images:
        if f.model_1_target==x_class and f.model_2_target==y_class:
            col=models_to_plot.index(f.model_1_name)
            row=models_to_plot.index(f.model_2_name)
        else:
            col=models_to_plot.index(f.model_2_name)
            row=models_to_plot.index(f.model_1_name)

        cur_im=PIL.Image.open(f.filename)

        im_matrix[row, col]=cur_im
    plot_im_matrix(im_matrix,x_class,y_class,c,subplot_spec=subplot_spec,upscale=upscale)


def plot_cat_vs_dog_figure(optim_method='direct',target_parent_folder='figures',image_folder='optimization_results',seed=0):
    global models_to_plot
    models_to_plot=['InceptionV3','Resnet50','Resnet_50_l2_eps5','Wide_Resnet50_2_l2_eps5','ViTB16','ViTL16','ResNeXt101_32x32d']

    x_class='Persian_cat'
    y_class='Weimaraner'

    # figure configuration
    c=AttrDict()
    c.top_margin=1.5
    c.bottom_margin=0.025
    c.left_margin=1.5
    c.right_margin=0.33

    c.model_name_font_size=6
    c.major_label_font_size=8
    c.inch_per_minor_title_space=1.4
    c.inch_per_image=(3.42-c.left_margin-c.right_margin)/3.42 # 5.0 inch is the total with of the figure. There are 4 models.
    #between_subplot_margin=16/72
    c.curly_brace_line_width=0.5
    c.nudge_h_curly_label=0
    c.nudge_v_curly_label=0

    c.top_left_curly_brace_horizontal_shift=2.02-0.25-1.0
    c.top_right_curly_brace_horizontal_shift=1.09+0.25-0.5
    c.im_matrix_line_width=0.75
    c.spine_line_width=0.8
    c.h_curly_pad=0.55
    c.v_curly_pad=0.32

    c.nudge_v_label_x=0
    c.nudge_v_label_y=0

    c.do_plot_images=True
    #major_margin=16/72
    #inch_per_major_title_space=32/72

    c.do_plot_vertical_brace=True
    c.do_plot_horizontal_brace=True

    upscale=256
    
    # 简化文件夹处理逻辑：直接使用传入的image_folder作为stimuli_path
    stimuli_path = image_folder
    
    # 提取方法名称用于输出文件命名
    method_name = os.path.basename(stimuli_path) if os.path.isdir(stimuli_path) else 'direct'
    if '_optim_cat_vs_dog' in method_name:
        method_name = method_name.split('_optim_cat_vs_dog')[0]
    
    print(f"正在使用图像文件夹: {stimuli_path}")
    print(f"使用方法名称: {method_name}")

    plt.close('all')
    plot_single_class_pair_multiple_model_pairs_controversial_stimuli_matrix(x_class=x_class,y_class=y_class,c=c,stimuli_path=stimuli_path,upscale=upscale,seed=seed)

    # 构建基本文件名
    base_fname = method_name
    
    # 创建完整的文件名
    fig_fname = f"{base_fname}_optim_{len(models_to_plot)}_models_{x_class}_by_{y_class}_seed{seed}"
    
    # 确保目标文件夹存在
    pathlib.Path(target_parent_folder).mkdir(parents=True, exist_ok=True)
    
    # 分别保存PDF和PNG文件，使用更高的DPI值确保PNG质量良好
    pdf_path = os.path.join(target_parent_folder, fig_fname + '.pdf')
    png_path = os.path.join(target_parent_folder, fig_fname + '.png')
    
    plt.savefig(pdf_path, dpi=upscale/c.inch_per_image, bbox_inches='tight')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')  # 使用300dpi确保PNG质量

    print(f'保存PDF文件至: {pdf_path}')
    print(f'保存PNG文件至: {png_path}')


def plot_batch(optimization_methods=None,image_folder='optimization_results',figure_folder='figures',seed=0):
  if optimization_methods is None:
    optimization_methods=['direct','jittered','decorrelated','CPPN','GAN-pool5','GAN-fc6','GAN-fc7','GAN-fc8','diffusion']
  for optim_method in optimization_methods:
    plot_cat_vs_dog_figure(optim_method=optim_method,image_folder=image_folder,target_parent_folder=figure_folder,seed=seed)

if __name__ == '__main__':
    # 使用ArgumentParser，但设置所有参数都有默认值
    parser = argparse.ArgumentParser()
    parser.add_argument('--stimuli_path', type=str, default='.', help='Path to the directory containing the stimuli')
    parser.add_argument('--figpath', type=str, default='figures', help='Path to save the figure')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()

    # 确保figures目录存在
    os.makedirs(args.figpath, exist_ok=True)
    
    # 设置输出文件路径
    output_fig_path = os.path.join(args.figpath, 'controversial_stimuli_matrix_cat_vs_dog_latent_space.pdf')
    
    # 创建图表
    fig = plt.figure(figsize=(12, 9), facecolor='white')
    # 添加标题，减小字体大小并修改内容
    # fig.suptitle("Controversial Stimuli from Unconditional Diffusion Model", fontsize=12)
    
    # 使用正确的文件夹名称，不需要在主函数再拼接路径
    folder_name = 'diffusion_latent_optim_cat_vs_dog_v2'
    
    # 设置配置参数
    c=AttrDict(
            spine_line_width=1.0,
            im_matrix_line_width=4.0,
            curly_brace_line_width=1.0,
            inch_per_image=2.0,
            model_name_font_size=9,
            top_margin=1,
            left_margin=2.0,
            right_margin=0.05,
            bottom_margin=0.7,
            hspace=0.3,
            wspace=0.3,
            cmap_min=0.0,
            cmap_max=1.0,
            figsize=None,
            row_labels=['wide_Resnet50_2_l2_eps5', 'resnet50_l2_eps5', 'resnet50', 'inception_v3'],
            col_labels=['Persian cat', 'Weimaraner'],
            n_rows=4,
            n_cols=2,
            row_label_ipadding=0.05,
            row_label_rotation=0,
            col_label_padding=0.2,
            row_label_padding=0.5,
            cmap='viridis',
            label_fontsize=10,
            exp_name='cat_vs_dog',
            interpolation='none'
            )
            
    # 创建并绘制图表
    plot_cat_vs_dog_figure(optim_method=folder_name, 
                          target_parent_folder=args.figpath, 
                          image_folder=args.stimuli_path,
                          seed=args.random_seed)

    # 保存图像
    plt.savefig(output_fig_path, bbox_inches='tight')
    print(f"图像已保存至: {output_fig_path}")
    plt.show()