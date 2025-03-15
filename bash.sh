# 实验1: conditional (使用正常提示词)
nohup bash -c 'python batch_optimize.py --experiments cat_vs_dog --optimization_methods diffusion_noise --max_steps 300 --min_controversiality 0.8' > conditional_exp.log 2>&1 &

# 实验2: unconditional (使用无条件提示词)
nohup bash -c 'python batch_optimize.py --experiments cat_vs_dog --optimization_methods diffusion_noise_unconditional --max_steps 300 --min_controversiality 0.8' > unconditional_exp.log 2>&1 &