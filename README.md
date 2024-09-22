# CVAE 字母生成项目

## 项目简介

本项目基于 **条件变分自编码器（Conditional Variational Autoencoder, CVAE）**，对字母数据集进行重构和生成。通过引入条件变量（字母标签），模型能够生成指定的字母，实现对生成结果的控制。

## 特点

- **数据集制作**：包含字母 A 到 Z，每个字母具有多个样本，经过预处理后，图像为黑底白字，尺寸为 96×96。
- **模型架构**：编码器和解码器均采用多层感知器（MLP），引入条件变量，潜在空间维度可调。
- **生成效果**：模型能够生成清晰的字母图像，支持指定字母的生成。

## 目录结构

```
├── configs
│   └── config.yaml          # 模型和训练的配置文件
├── dataset
│   ├── raw                  # 原始数据集
│   └── processed            # 预处理后的数据集
├── models
│   └── cvae.py              # CVAE 模型定义
├── outputs
│   ├── latent_dim64         # 潜在空间维度为64的输出
│   └── latent_dim128        # 潜在空间维度为128的输出
├── utils
│   └── train_utils.py       # 训练相关的工具函数
├── train.py                 # 模型训练脚本
├── generate.py              # 图像生成脚本
├── requirements.txt         # 所需的 Python 包
└── README.md                # 项目说明文件
```

## 环境依赖

- Python 3.7+
- PyTorch 1.7+
- torchvision
- numpy
- matplotlib
- opencv-python
- PyYAML
- wandb

可以通过以下命令安装所需依赖：

```bash
pip install -r requirements.txt
```

## 数据集准备

1. 从 [Kaggle](https://www.kaggle.com/competitions/english-letter-recognition/data) 下载字母数据集，并解压到 `dataset/raw/` 目录下。
2. 运行数据预处理脚本，将数据集处理成模型需要的格式：

```bash
python data_preprocess.py
```

该脚本将图像转换为黑底白字，调整尺寸，并按照指定的命名方式保存到 `dataset/processed/` 目录下。

## 模型训练

1. 配置模型和训练参数。在 `configs/config.yaml` 文件中，可以调整以下参数：

   - `train`：训练相关参数，如数据路径、批量大小、学习率、训练轮数等。
   - `vae_model`：模型结构参数，如编码器和解码器层数、潜在空间维度、激活函数等。

2. 运行训练脚本：

```bash
python train.py
```

训练过程中，模型会自动保存到指定的输出目录，并使用 `wandb` 记录训练日志和损失曲线。

## 图像生成

训练完成后，可以使用生成脚本生成字母图像：

```bash
python generate.py
```

生成的图像将保存在 `outputs/generated_images/` 目录下。

## 模型架构

- **编码器**：

  - 输入：展平的图像数据（9216维）和条件变量（26维）。
  - 多层全连接层：层数和单元数可在配置文件中调整。
  - 输出：潜在空间的均值（mu）和对数方差（logvar）。

- **解码器**：

  - 输入：采样得到的潜在变量（z）和条件变量（26维）。
  - 多层全连接层：层数和单元数可在配置文件中调整。
  - 输出：重构的图像数据（9216维）。

## 结果展示

- **训练损失曲线**：训练过程中，损失函数逐渐下降，模型收敛良好。

- **生成的字母图像**：模型能够生成清晰的字母图像，支持生成指定的字母。

（此处可以插入训练损失曲线和生成的字母图像示例）

## 改进方向

- **编码方式优化**：尝试使用嵌入层（Embedding）替代 one-hot 编码，降低条件变量的维度，提高模型性能。

- **模型结构改进**：使用卷积神经网络（CNN）替代全连接层，捕获图像的空间特征，提升生成质量。

- **超参数调整**：进一步探索潜在空间维度、网络层数等超参数对模型性能的影响。

## 参考文献

- [Kingma, D. P., & Welling, M. (2013). Auto-Encoding Variational Bayes. arXiv preprint arXiv:1312.6114.](https://arxiv.org/abs/1312.6114)
- [Sohn, K., Lee, H., & Yan, X. (2015). Learning Structured Output Representation using Deep Conditional Generative Models. Advances in Neural Information Processing Systems, 28.](https://papers.nips.cc/paper/2015/hash/8d55a249e6baa5c06772297520da2051-Abstract.html)

## 项目地址

本项目已上传至 GitHub，欢迎 Star 和 Fork：

[https://github.com/yourusername/CVAE-Letter-Generation](https://github.com/yourusername/CVAE-Letter-Generation)
