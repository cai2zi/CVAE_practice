import torch
import os
import yaml
import sys
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path)
import numpy as np
from models.cvae import CVAE
from torchvision.utils import save_image
from PIL import Image
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 加载配置文件
with open('./config/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# 从配置文件中获取生成图像相关的参数
generate_config = config['generate']
vae_config = config['vae_model']
train_config = config['train']
# 加载模型
cvae = CVAE(input_dim=vae_config['input_dim'], 
            latent_dim=vae_config['latent_dim'],
            encoder_layers=vae_config['encoder_layers'],
            decoder_layers=vae_config['decoder_layers'],
            activation=vae_config['activation_function'],
            cond_dim=train_config['cond_dim'],
            ).to(device)
cvae.load_state_dict(torch.load(generate_config['model_path'],map_location=device))
cvae.eval()

# 生成图像
with torch.no_grad():
    all_images = []
    for label in range(26):  # 生成26个字母
        z = torch.randn(generate_config['num_images'], vae_config['latent_dim'])  # 为每个字母生成16个潜在变量
        labels=(torch.ones(generate_config['num_images'])*label).int()
        c = torch.eye(train_config['cond_dim'])[labels].to(device)
        generated_images = cvae.decode(z, c).view(-1, 1, 96, 96)

        # 将生成的图像转换为列表并添加到all_images
        all_images.append(generated_images)

    # 拼接生成的图像
    all_images = torch.cat(all_images, dim=0)  # 形状为 (26*num, 1, 96, 96)
    grid_image = save_image(all_images, os.path.join(generate_config['output_dir'], generate_config['name']), nrow=generate_config['num_images'])