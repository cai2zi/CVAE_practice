import torch
import os
import yaml
import sys
current_path = os.path.dirname(os.path.abspath(__file__))
print(current_path)
sys.path.append(current_path)
from torchvision.transforms import ToTensor
from dataset.LetterDataset import LetterDataset
from models.cvae import CVAE
from utils.train_utils import train_model
from torchvision.transforms import ToTensor, Resize, Compose
import wandb



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载配置文件
with open('./config/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# 从配置文件中获取训练相关的参数
train_config = config['train']
vae_config = config['vae_model']

# 加载自定义数据集
train_data_path = train_config['data_path']
transform = Compose([
    Resize((96, 96)),
    ToTensor() 
])
train_dataset = LetterDataset(root=train_data_path, transforms=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_config['batch_size'], shuffle=True)

# 初始化模型和优化器
cvae = CVAE(input_dim=vae_config['input_dim'], 
            latent_dim=vae_config['latent_dim'],
            encoder_layers=vae_config['encoder_layers'],
            decoder_layers=vae_config['decoder_layers'],
            activation=vae_config['activation_function'],
            cond_dim=train_config['cond_dim'],
            ).to(device)
optimizer = torch.optim.Adam(cvae.parameters(), lr=train_config['lr'])

# 训练模型

wandb.init(project="CVAE_practice", name=train_config['exp_name'], config=config)
train_model(cvae=cvae, train_loader=train_loader, 
            optimizer=optimizer,num_epochs=train_config['epochs'],
            cond_dim=train_config['cond_dim'],
            save_dir=train_config['save_model_path'],
            device=device)
wandb.finish()
# 保存模型
path=train_config['save_model_path']+'cvae.pth'
os.makedirs(os.path.dirname(path), exist_ok=True)
torch.save(cvae.state_dict(), path)
