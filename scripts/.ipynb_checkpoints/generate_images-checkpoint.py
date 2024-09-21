import torch
import os
import yaml
import sys
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path)
from models.cvae import CVAE
from torchvision.utils import save_image

# 加载配置文件
with open('./config/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# 从配置文件中获取生成图像相关的参数
generate_config = config['generate']
vae_config = config['vae_model']

# 加载模型
cvae = CVAE(input_dim=vae_config['input_dim'], hidden_dim=vae_config['hidden_dim'], latent_dim=vae_config['latent_dim'])
cvae.load_state_dict(torch.load(generate_config['model_path']))
cvae.eval()

# 生成图像
with torch.no_grad():
    z = torch.randn(generate_config['num_images'], vae_config['latent_dim'])  # 生成随机潜在变量
    generated_images = cvae.decode(z).view(-1, 1, 96, 96)

    # 保存生成的图像
    os.makedirs(generate_config['output_dir'], exist_ok=True)
    save_image(generated_images, os.path.join(generate_config['output_dir'], 'sample.png'))
