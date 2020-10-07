---
layout: post
title: "DCGAN Tutorial"
subtitle: 'The concept and implementation of DCGAN'
author: "Yuzec"
header-style: text
tags:
  - DCGAN
  - PyTorch
---

# DCGAN教程

> 原文链接：https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

## 介绍

​		本教程将通过一个示例对DCGAN进行介绍。 在向其展示许多真实名人的照片后，我们将训练一个生成对抗网络(GAN)来产生新名人。 此处的大多数代码来自pytorch/examples中的dcgan实现，并且本文档将对该实现进行详尽的解释，并阐明此模型的工作方式和原因。 但请放心，不需要GAN的先验知识，但这可能需要新手花一些时间来推理幕后实际情况。 另外，为了节省时间，安装一两个GPU也将有所帮助。 让我们从头开始。

## 生成对抗网络

### 什么是GAN？

​		GAN是用于教授DL模型以捕获培训数据分布的框架，因此我们可以从同一分布中生成新数据。 GAN是由Ian Goodfellow于2014年发明的，并在论文《生成对抗网络》中首次描述。它们由两个不同的模型组成，分别是生成器和鉴别器。生成器的工作是生成看起来像训练图像的“假”图像。鉴别器的工作是查看图像并从生成器输出它是真实的训练图像还是伪图像。在训练过程中，生成器不断尝试通过生成越来越好的伪造品而使鉴别器的性能超过智者，而鉴别器正在努力成为更好的侦探并正确地对真实和伪造图像进行分类。博弈的平衡点是当生成器生成的伪造品看起来好像直接来自训练数据时，而鉴别器则总是猜测生成器输出是真实的还是伪造品的50％置信度。

​		现在，让我们从判别器开始定义一些在整个教程中使用的符号。 令x为代表图像的数据。 D(x)是鉴别器网络，它输出x来自训练数据而不是生成器的(标量)概率。 在这里，由于我们要处理图像，因此D(x)的输入是CHW大小为3x64x64的图像。 直观地，当x来自训练数据时，D(x)应该为高；当x来自生成器时，D(x)应该为低。 D(x)也可以认为是传统的二进制分类器。

​		对于生成器的表示法，令z为从标准正态分布采样的潜在空间矢量。 G(z)表示将潜在向量z映射到数据空间的生成器函数。 G的目标是估计训练数据来自的分布(pdata)，以便它可以从估计的分布(pg)生成假样本。

​		因此，D(G(z))是发生器G的输出是真实图像的概率(标量)。 如Goodfellow的论文所述，D和G玩一个minimax游戏，其中D尝试使D正确分类实数和假数(logD(x))的概率最大化，而G尝试使D预测其输出为假的概率最小化( log(1-D(G(x))))。 从本文来看，GAN损失函数为
$$
min_Gmax_DV(D,G)=E_{x~P_{data}(x)}[logD(x)]+E_{z~P_z(z)[log(1-D(G(z)))]}
$$
​		从理论上讲，此minimax游戏的解决方案是pg = pdata，鉴别器随机猜测输入是真实的还是假的。 但是，GAN的收敛理论仍在积极研究中，实际上，模型并不总是能达到此目的。

### 什么是DCGAN？

​		DCGAN是上述GAN的直接扩展，不同之处在于DCGAN分别在鉴别器和生成器中分别使用卷积和卷积转置层。它最早由Radford等人描述。等深度卷积生成对抗网络中的无监督表示学习。鉴别器由跨步卷积层，批范数层和LeakyReLU激活组成。输入是3x64x64的输入图像，输出是输入来自真实数据分布的标量概率。生成器由卷积转置层，批处理规范层和ReLU激活组成。输入是从标准正态分布中提取的潜矢量z，输出是3x64x64 RGB图像。跨步的转置图层使潜矢量可以转换为具有与图像相同形状的体积。在本文中，作者还提供了有关如何设置优化器，如何计算损失函数以及如何初始化模型权重的一些技巧，所有这些将在接下来的部分中进行解释。

```python
from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
```

输出：

```text
Random Seed:  999
```

## 输入

​		让我们为运行定义一些输入：

- dataroot-数据集文件夹根目录的路径。我们将在下一节中进一步讨论数据集

- worker-使用DataLoader加载数据的工作线程数

- batch_size-训练中使用的批次大小。 DCGAN纸使用的批处理大小为128

- image_size-用于训练的图像的空间大小。此实现默认为64x64。如果需要其他尺寸，则必须更改D和G的结构。详情请看这里

- nc-输入图像中的颜色通道数。对于彩色图像，这是3

- nz-潜在向量的长度

- ngf-与生成器承载的特征图的深度有关

- ndf-设置通过鉴别器传播的特征图的深度

- num_epochs-要运行的训练时期数。训练更长的时间可能会导致更好的结果，但也会花费更长的时间

- lr-培训的学习率。如DCGAN文件中所述，此数字应为0.0002

- 用于Adam优化器的beta1-beta1超参数。如论文所述，该数字应为0.5

- ngpu-可用的GPU数量。如果为0，则代码将在CPU模式下运行。如果此数字大于0，它将在该数量的GPU上运行

```python
# Root directory for dataset
dataroot = "data/celeba"

# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 128

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 5

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1
```

## 数据

​		在本教程中，我们将使用Celeb-A Faces数据集，该数据集可以在链接的网站或Google云端硬盘中下载。 数据集将下载为名为img_align_celeba.zip的文件。 下载后，创建一个名为celeba的目录，并将zip文件解压缩到该目录中。 然后，将此笔记本的dataroot输入设置为刚创建的celeba目录。 结果目录结构应为：

```text
/path/to/celeba
    -> img_align_celeba
        -> 188242.jpg
        -> 173822.jpg
        -> 284702.jpg
        -> 537394.jpg
           ...
```

​		这是重要的一步，因为我们将使用ImageFolder数据集类，该类要求数据集的根文件夹中有子目录。 现在，我们可以创建数据集，创建数据加载器，将设备设置为可以运行，并最终可视化一些训练数据。

```python
# We can use an image folder dataset the way we have it setup.
# Create the dataset
dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# Plot some training images
real_batch = next(iter(dataloader))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
```

![https://pytorch.org/tutorials/_images/sphx_glr_dcgan_faces_tutorial_001.png](https://pytorch.org/tutorials/_images/sphx_glr_dcgan_faces_tutorial_001.png)

## 实现

​		设置好输入参数并准备好数据集后，我们现在可以进入实现了。 我们将从Weigth初始化策略开始，然后详细讨论生成器，鉴别器，损失函数和训练循环。

### 权重初始化

​		从DCGAN论文中，作者指定所有模型权重均应从均值= 0，stdev = 0.02的正态分布中随机初始化。 weights_init函数将初始化的模型作为输入，并重新初始化所有卷积，卷积转置和批处理归一化层，以符合此条件。 初始化后立即将此功能应用于模型。

```python
# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
```

### 生成器

​		生成器G用于将潜在空间矢量(z)映射到数据空间。 由于我们的数据是图像，因此将z转换为数据空间意味着最终创建与训练图像大小相同的RGB图像(即3x64x64)。 实际上，这是通过一系列跨步的二维卷积转置层来完成的，每个层都与2d批处理规范层和relu激活配对。 发生器的输出通过tanh函数馈送，以使其返回到[-1,1]的输入数据范围。 值得注意的是，在卷积转置层之后存在批处理规范函数，因为这是DCGAN论文的关键贡献。 这些层有助于训练过程中的梯度流动。 DCGAN纸生成的图像如下所示。

![https://pytorch.org/tutorials/_images/dcgan_generator.png](https://pytorch.org/tutorials/_images/dcgan_generator.png)

​		注意，我们在输入部分中设置的输入(nz，ngf和nc)如何影响代码中的生成器体系结构。 nz是z输入向量的长度，ngf与通过生成器传播的特征图的大小有关，nc是输出图像中的通道数(对于RGB图像，设置为3)。 以下是生成器的代码。

```python
# Generator Code

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)
```

​		现在，我们可以实例化生成器并应用weights_init函数。 签出打印的模型以查看生成器对象的结构。

```python
# Create the generator
netG = Generator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netG.apply(weights_init)

# Print the model
print(netG)
```

输出：

```text
Generator(
  (main): Sequential(
    (0): ConvTranspose2d(100, 512, kernel_size=(4, 4), stride=(1, 1), bias=False)
    (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU(inplace=True)
    (6): ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (7): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (8): ReLU(inplace=True)
    (9): ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (10): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (11): ReLU(inplace=True)
    (12): ConvTranspose2d(64, 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (13): Tanh()
  )
)
```

### 鉴别器

​		如上所述，鉴别器D是一个二进制分类网络，该二进制分类网络将图像作为输入并输出输入图像是真实的(与假的相对)的标量概率。 在这里，D拍摄3x64x64的输入图像，通过一系列的Conv2d，BatchNorm2d和LeakyReLU层对其进行处理，然后通过Sigmoid激活函数输出最终概率。 如果需要解决此问题，可以用更多层扩展此体系结构，但是使用跨步卷积，BatchNorm和LeakyReLU仍然很重要。 DCGAN论文提到，使用跨步卷积而不是合并以进行下采样是一个好习惯，因为它可以让网络学习其自身的合并功能。 批处理规范和泄漏的relu函数还可以促进健康的梯度流动，这对于G和D的学习过程都至关重要。

鉴别器代码

```python
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
```

​		现在，与生成器一样，我们可以创建鉴别器，应用weights_init函数，并打印模型的结构。

```python
# Create the Discriminator
netD = Discriminator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netD.apply(weights_init)

# Print the model
print(netD)
```

输出：

```text
Discriminator(
  (main): Sequential(
    (0): Conv2d(3, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (1): LeakyReLU(negative_slope=0.2, inplace=True)
    (2): Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (4): LeakyReLU(negative_slope=0.2, inplace=True)
    (5): Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (6): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (7): LeakyReLU(negative_slope=0.2, inplace=True)
    (8): Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (9): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (10): LeakyReLU(negative_slope=0.2, inplace=True)
    (11): Conv2d(512, 1, kernel_size=(4, 4), stride=(1, 1), bias=False)
    (12): Sigmoid()
  )
)
```

### 损失函数和优化器

​		通过D和G设置，我们可以指定它们如何通过损失函数和优化器学习。 我们将使用在PyTorch中定义的二进制交叉熵损失(BCELoss)函数：
$$
\ell(x,y)=L=\{l_1,...,l_N\}^T,l_n=-[y_n·logx_n+(1-y_n)·log(1-x_n)]
$$
​		请注意，此函数如何提供目标函数中两个对数成分的计算(即log(D(x))和log(1-D(G(z))))。 我们可以指定BCE方程的y输入部分。 这是在即将到来的训练循环中完成的，但重要的是要了解我们如何仅通过更改y(即GT标签)就可以选择希望计算的分量。

​		接下来，我们将实际标签定义为1，将假标签定义为0。这些标签将在计算D和G的损失时使用，这也是原始GAN论文中使用的惯例。 最后，我们设置了两个单独的优化器，一个用于D，一个用于G。如DCGAN文件中所述，这两个都是Adam学习器，其学习率分别为0.0002和Beta1 = 0.5。 为了跟踪生成器的学习进度，我们将生成一批固定的潜在矢量，这些矢量是根据高斯分布(即fixed_noise)得出的。 在训练循环中，我们将定期将此fixed_noise输入到G中，并且在迭代过程中，我们将看到图像形成于噪声之外。

```python
# Initialize BCELoss function
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
```

### 训练

​		最后，既然我们已经定义了GAN框架的所有部分，我们就可以对其进行培训。 请注意，训练GAN有点像是一种艺术形式，因为不正确的超参数设置会导致模式崩溃，而对失败的原因几乎没有解释。 在这里，我们将严格遵循Goodfellow论文中的算法1，同时遵守ganhacks中显示的一些最佳做法。 即，我们将“为真实和伪造构建不同的小批量”图像，并调整G的目标函数以最大化logD(G(z))。 培训分为两个主要部分。 第1部分更新了鉴别器，第2部分更新了生成器。

#### 第1部分-训练鉴别器

​		回想一下，训练鉴别器的目的是最大程度地提高将给定输入正确分类为真实或伪造的可能性。 关于古德费罗(Goodfellow)，我们希望“通过提高随机梯度来更新鉴别器”。 实际上，我们要最大化log(D(x))+ log(1-D(G(z)))。 由于ganhacks提出了单独的小批量建议，因此我们将分两步进行计算。 首先，我们将从训练集中构造一批真实样本，向前通过D，计算损失(log(D(x)))，然后在向后通过中计算梯度。 其次，我们将用电流发生器构造一批假样本，将这批样本向前通过D，计算损耗(log(1-D(G(z))))，并通过向后累积梯度。 现在，利用全批次和全批次的累积梯度，我们称之为鉴别器优化程序的一个步骤。

#### 第2部分-训练生成器

​		如原始论文所述，我们希望通过最小化log(1-D(G(z)))来训练Generator，以产生更好的伪造品。 如前所述，Goodfellow证明这不能提供足够的梯度，尤其是在学习过程的早期。 作为解决方法，我们改为希望最大化log(D(G(z)))。 在代码中，我们通过以下方式实现此目的：将第1部分的Generator输出与Discriminator进行分类，使用真实标签作为GT计算G的损失，向后计算G的梯度，最后通过优化器步骤更新G的参数。 将真实标签用作损失函数的GT标签似乎违反直觉，但是这允许我们使用BCELoss的log(x)部分(而不是log(1-x)部分)，这恰恰是 我们想要。

​		最后，我们将进行一些统计报告，并在每个时期结束时，将我们的fixed_noise批次推入生成器，以直观地跟踪G的训练进度。 报告的培训统计数据是：

- Loss_D-鉴别器损失，计算为所有真实批次和所有假批次的损失总和(log(D(x))+ log(D(G(z)))
  )。
- Loss_G-生成器损耗计算为log(D(G(z)))
- D(x)-所有实际批次的鉴别器的平均输出(整个批次)。 这应该从接近1开始，然后在G变得更好时理论上收敛至0.5。 考虑一下为什么会这样。
- D(G(z))-所有假批次的平均鉴别器输出。 第一个数字在D更新之前，第二个数字在D更新之后。 这些数字应从0开始，并随着G的提高收敛到0.5。 考虑一下为什么会这样。

注意：此步骤可能需要一段时间，具体取决于您运行了多少个时期以及是否从数据集中删除了一些数据。

```python
# Training Loop

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        # Forward pass real batch through D
        output = netD(real_cpu).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        # Generate fake image batch with G
        fake = netG(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1
```

输出：

```text
Starting Training Loop...
[0/5][0/1583]   Loss_D: 2.0937  Loss_G: 5.2060  D(x): 0.5704    D(G(z)): 0.6680 / 0.0090
[0/5][50/1583]  Loss_D: 2.0243  Loss_G: 29.0899 D(x): 0.9875    D(G(z)): 0.8004 / 0.0000
[0/5][100/1583] Loss_D: 0.0517  Loss_G: 40.2129 D(x): 0.9902    D(G(z)): 0.0000 / 0.0000
[0/5][150/1583] Loss_D: 0.0007  Loss_G: 39.0751 D(x): 0.9994    D(G(z)): 0.0000 / 0.0000
[0/5][200/1583] Loss_D: 0.2650  Loss_G: 13.2042 D(x): 0.9025    D(G(z)): 0.0001 / 0.0000
[0/5][250/1583] Loss_D: 0.8816  Loss_G: 4.9778  D(x): 0.6012    D(G(z)): 0.0589 / 0.0195
[0/5][300/1583] Loss_D: 0.8936  Loss_G: 8.1565  D(x): 0.9322    D(G(z)): 0.5054 / 0.0008
[0/5][350/1583] Loss_D: 0.6441  Loss_G: 3.9163  D(x): 0.6654    D(G(z)): 0.1024 / 0.0333
[0/5][400/1583] Loss_D: 1.0931  Loss_G: 9.7798  D(x): 0.9604    D(G(z)): 0.5811 / 0.0005
[0/5][450/1583] Loss_D: 0.4749  Loss_G: 5.1082  D(x): 0.8783    D(G(z)): 0.2383 / 0.0102
[0/5][500/1583] Loss_D: 0.5570  Loss_G: 4.0530  D(x): 0.6934    D(G(z)): 0.0491 / 0.0354
[0/5][550/1583] Loss_D: 0.6550  Loss_G: 4.4540  D(x): 0.7340    D(G(z)): 0.1964 / 0.0282
[0/5][600/1583] Loss_D: 1.5417  Loss_G: 5.3091  D(x): 0.3124    D(G(z)): 0.0025 / 0.0096
[0/5][650/1583] Loss_D: 0.6672  Loss_G: 4.4616  D(x): 0.7900    D(G(z)): 0.2370 / 0.0238
[0/5][700/1583] Loss_D: 0.5403  Loss_G: 4.2616  D(x): 0.7856    D(G(z)): 0.2003 / 0.0287
[0/5][750/1583] Loss_D: 1.7091  Loss_G: 2.1415  D(x): 0.3123    D(G(z)): 0.0033 / 0.2257
[0/5][800/1583] Loss_D: 0.3774  Loss_G: 4.3136  D(x): 0.8457    D(G(z)): 0.1439 / 0.0204
[0/5][850/1583] Loss_D: 0.5051  Loss_G: 3.8821  D(x): 0.7569    D(G(z)): 0.1394 / 0.0362
[0/5][900/1583] Loss_D: 0.8397  Loss_G: 3.2884  D(x): 0.5392    D(G(z)): 0.0215 / 0.0690
[0/5][950/1583] Loss_D: 1.1208  Loss_G: 5.1048  D(x): 0.4460    D(G(z)): 0.0033 / 0.0162
[0/5][1000/1583]        Loss_D: 0.5095  Loss_G: 4.8398  D(x): 0.8694    D(G(z)): 0.2505 / 0.0141
[0/5][1050/1583]        Loss_D: 0.6523  Loss_G: 4.8777  D(x): 0.8292    D(G(z)): 0.2897 / 0.0160
[0/5][1100/1583]        Loss_D: 0.2932  Loss_G: 4.2959  D(x): 0.8265    D(G(z)): 0.0488 / 0.0237
[0/5][1150/1583]        Loss_D: 0.4451  Loss_G: 4.5365  D(x): 0.8408    D(G(z)): 0.1912 / 0.0179
[0/5][1200/1583]        Loss_D: 0.7657  Loss_G: 6.3654  D(x): 0.9689    D(G(z)): 0.4643 / 0.0057
[0/5][1250/1583]        Loss_D: 0.4300  Loss_G: 4.0337  D(x): 0.8810    D(G(z)): 0.2210 / 0.0299
[0/5][1300/1583]        Loss_D: 0.5113  Loss_G: 4.6566  D(x): 0.8479    D(G(z)): 0.2266 / 0.0163
[0/5][1350/1583]        Loss_D: 0.4628  Loss_G: 4.1022  D(x): 0.8987    D(G(z)): 0.2345 / 0.0332
[0/5][1400/1583]        Loss_D: 0.5381  Loss_G: 3.5408  D(x): 0.6951    D(G(z)): 0.0725 / 0.0538
[0/5][1450/1583]        Loss_D: 1.3048  Loss_G: 8.0072  D(x): 0.9612    D(G(z)): 0.6183 / 0.0028
[0/5][1500/1583]        Loss_D: 0.3733  Loss_G: 3.6582  D(x): 0.8224    D(G(z)): 0.1294 / 0.0397
[0/5][1550/1583]        Loss_D: 0.4300  Loss_G: 2.8753  D(x): 0.7624    D(G(z)): 0.0813 / 0.0905
[1/5][0/1583]   Loss_D: 0.4071  Loss_G: 3.7229  D(x): 0.8248    D(G(z)): 0.1471 / 0.0411
[1/5][50/1583]  Loss_D: 0.3431  Loss_G: 4.7965  D(x): 0.9057    D(G(z)): 0.1915 / 0.0142
[1/5][100/1583] Loss_D: 0.3428  Loss_G: 3.3041  D(x): 0.8458    D(G(z)): 0.1328 / 0.0618
[1/5][150/1583] Loss_D: 1.2300  Loss_G: 1.5985  D(x): 0.4151    D(G(z)): 0.0022 / 0.3130
[1/5][200/1583] Loss_D: 1.0345  Loss_G: 7.5999  D(x): 0.9438    D(G(z)): 0.5655 / 0.0011
[1/5][250/1583] Loss_D: 0.4836  Loss_G: 4.1048  D(x): 0.8615    D(G(z)): 0.2356 / 0.0283
[1/5][300/1583] Loss_D: 0.3732  Loss_G: 3.4245  D(x): 0.8102    D(G(z)): 0.1158 / 0.0558
[1/5][350/1583] Loss_D: 0.8251  Loss_G: 2.3069  D(x): 0.5820    D(G(z)): 0.1162 / 0.1449
[1/5][400/1583] Loss_D: 0.6552  Loss_G: 3.3054  D(x): 0.7471    D(G(z)): 0.2155 / 0.0624
[1/5][450/1583] Loss_D: 0.4571  Loss_G: 2.9486  D(x): 0.7557    D(G(z)): 0.0904 / 0.0805
[1/5][500/1583] Loss_D: 0.8473  Loss_G: 5.7569  D(x): 0.8700    D(G(z)): 0.4415 / 0.0054
[1/5][550/1583] Loss_D: 0.8673  Loss_G: 6.2354  D(x): 0.8680    D(G(z)): 0.4411 / 0.0076
[1/5][600/1583] Loss_D: 0.3334  Loss_G: 3.5483  D(x): 0.8287    D(G(z)): 0.0918 / 0.0507
[1/5][650/1583] Loss_D: 0.3941  Loss_G: 3.0172  D(x): 0.7888    D(G(z)): 0.1062 / 0.0711
[1/5][700/1583] Loss_D: 0.5443  Loss_G: 2.6461  D(x): 0.7017    D(G(z)): 0.0837 / 0.1145
[1/5][750/1583] Loss_D: 1.5301  Loss_G: 2.1856  D(x): 0.3339    D(G(z)): 0.0106 / 0.1828
[1/5][800/1583] Loss_D: 0.4848  Loss_G: 4.8203  D(x): 0.9399    D(G(z)): 0.3211 / 0.0116
[1/5][850/1583] Loss_D: 0.6039  Loss_G: 3.2201  D(x): 0.7867    D(G(z)): 0.2461 / 0.0601
[1/5][900/1583] Loss_D: 0.6834  Loss_G: 1.7309  D(x): 0.5775    D(G(z)): 0.0279 / 0.2345
[1/5][950/1583] Loss_D: 0.5107  Loss_G: 4.4513  D(x): 0.8882    D(G(z)): 0.2842 / 0.0187
[1/5][1000/1583]        Loss_D: 0.7641  Loss_G: 5.5313  D(x): 0.9486    D(G(z)): 0.4523 / 0.0064
[1/5][1050/1583]        Loss_D: 0.7497  Loss_G: 0.8408  D(x): 0.5743    D(G(z)): 0.0381 / 0.4977
[1/5][1100/1583]        Loss_D: 0.5799  Loss_G: 3.7267  D(x): 0.8680    D(G(z)): 0.3184 / 0.0328
[1/5][1150/1583]        Loss_D: 0.5499  Loss_G: 3.3622  D(x): 0.8456    D(G(z)): 0.2721 / 0.0507
[1/5][1200/1583]        Loss_D: 0.7263  Loss_G: 3.5577  D(x): 0.8121    D(G(z)): 0.3327 / 0.0469
[1/5][1250/1583]        Loss_D: 0.3859  Loss_G: 3.2090  D(x): 0.8359    D(G(z)): 0.1575 / 0.0559
[1/5][1300/1583]        Loss_D: 0.9260  Loss_G: 4.6833  D(x): 0.9676    D(G(z)): 0.5290 / 0.0163
[1/5][1350/1583]        Loss_D: 0.4084  Loss_G: 2.8062  D(x): 0.7985    D(G(z)): 0.1326 / 0.0890
[1/5][1400/1583]        Loss_D: 0.6256  Loss_G: 2.9316  D(x): 0.7637    D(G(z)): 0.2349 / 0.0781
[1/5][1450/1583]        Loss_D: 0.8438  Loss_G: 5.1306  D(x): 0.9038    D(G(z)): 0.4726 / 0.0093
[1/5][1500/1583]        Loss_D: 0.4638  Loss_G: 2.7462  D(x): 0.8051    D(G(z)): 0.1838 / 0.0888
[1/5][1550/1583]        Loss_D: 0.4140  Loss_G: 2.7286  D(x): 0.7754    D(G(z)): 0.1161 / 0.0881
[2/5][0/1583]   Loss_D: 2.8202  Loss_G: 6.9097  D(x): 0.9650    D(G(z)): 0.9008 / 0.0020
[2/5][50/1583]  Loss_D: 0.4126  Loss_G: 2.8698  D(x): 0.7827    D(G(z)): 0.1181 / 0.0807
[2/5][100/1583] Loss_D: 1.4539  Loss_G: 0.3044  D(x): 0.2997    D(G(z)): 0.0213 / 0.7701
[2/5][150/1583] Loss_D: 1.6619  Loss_G: 1.2936  D(x): 0.2708    D(G(z)): 0.0153 / 0.3351
[2/5][200/1583] Loss_D: 0.6329  Loss_G: 2.0706  D(x): 0.6902    D(G(z)): 0.1803 / 0.1642
[2/5][250/1583] Loss_D: 0.5985  Loss_G: 2.4406  D(x): 0.7445    D(G(z)): 0.2055 / 0.1157
[2/5][300/1583] Loss_D: 0.4124  Loss_G: 2.5239  D(x): 0.7844    D(G(z)): 0.1232 / 0.1005
[2/5][350/1583] Loss_D: 0.8838  Loss_G: 4.4928  D(x): 0.9129    D(G(z)): 0.4744 / 0.0183
[2/5][400/1583] Loss_D: 0.5059  Loss_G: 3.3841  D(x): 0.8686    D(G(z)): 0.2598 / 0.0517
[2/5][450/1583] Loss_D: 0.9095  Loss_G: 1.4349  D(x): 0.4967    D(G(z)): 0.0644 / 0.3030
[2/5][500/1583] Loss_D: 1.2050  Loss_G: 0.9456  D(x): 0.3722    D(G(z)): 0.0259 / 0.4592
[2/5][550/1583] Loss_D: 0.6433  Loss_G: 1.5030  D(x): 0.6380    D(G(z)): 0.1067 / 0.2673
[2/5][600/1583] Loss_D: 0.6854  Loss_G: 1.3461  D(x): 0.5929    D(G(z)): 0.0653 / 0.3307
[2/5][650/1583] Loss_D: 0.5465  Loss_G: 1.5042  D(x): 0.7232    D(G(z)): 0.1597 / 0.2662
[2/5][700/1583] Loss_D: 0.8002  Loss_G: 3.2153  D(x): 0.8084    D(G(z)): 0.3826 / 0.0569
[2/5][750/1583] Loss_D: 0.4121  Loss_G: 2.8521  D(x): 0.8029    D(G(z)): 0.1397 / 0.0830
[2/5][800/1583] Loss_D: 0.7926  Loss_G: 3.1577  D(x): 0.8954    D(G(z)): 0.4491 / 0.0623
[2/5][850/1583] Loss_D: 1.0068  Loss_G: 0.9440  D(x): 0.4825    D(G(z)): 0.1071 / 0.4590
[2/5][900/1583] Loss_D: 0.9756  Loss_G: 1.1884  D(x): 0.4638    D(G(z)): 0.0555 / 0.3731
[2/5][950/1583] Loss_D: 1.0754  Loss_G: 1.0662  D(x): 0.4680    D(G(z)): 0.1376 / 0.3905
[2/5][1000/1583]        Loss_D: 0.6894  Loss_G: 2.6803  D(x): 0.8152    D(G(z)): 0.3428 / 0.0925
[2/5][1050/1583]        Loss_D: 0.9542  Loss_G: 1.9602  D(x): 0.6734    D(G(z)): 0.3467 / 0.1872
[2/5][1100/1583]        Loss_D: 1.7125  Loss_G: 0.6530  D(x): 0.3542    D(G(z)): 0.1934 / 0.5932
[2/5][1150/1583]        Loss_D: 0.7149  Loss_G: 1.7581  D(x): 0.6330    D(G(z)): 0.1652 / 0.2119
[2/5][1200/1583]        Loss_D: 0.6314  Loss_G: 3.0262  D(x): 0.8479    D(G(z)): 0.3375 / 0.0674
[2/5][1250/1583]        Loss_D: 0.5698  Loss_G: 2.9108  D(x): 0.8584    D(G(z)): 0.3021 / 0.0709
[2/5][1300/1583]        Loss_D: 1.1338  Loss_G: 0.7756  D(x): 0.4089    D(G(z)): 0.0476 / 0.5185
[2/5][1350/1583]        Loss_D: 0.5039  Loss_G: 1.7990  D(x): 0.7397    D(G(z)): 0.1469 / 0.1993
[2/5][1400/1583]        Loss_D: 0.6637  Loss_G: 2.0220  D(x): 0.6848    D(G(z)): 0.2039 / 0.1652
[2/5][1450/1583]        Loss_D: 0.7179  Loss_G: 3.6909  D(x): 0.9471    D(G(z)): 0.4396 / 0.0358
[2/5][1500/1583]        Loss_D: 0.7201  Loss_G: 1.4389  D(x): 0.6028    D(G(z)): 0.1023 / 0.2890
[2/5][1550/1583]        Loss_D: 0.8756  Loss_G: 4.0837  D(x): 0.8766    D(G(z)): 0.4713 / 0.0256
[3/5][0/1583]   Loss_D: 0.7427  Loss_G: 2.7553  D(x): 0.8043    D(G(z)): 0.3667 / 0.0845
[3/5][50/1583]  Loss_D: 0.5782  Loss_G: 3.0668  D(x): 0.8093    D(G(z)): 0.2651 / 0.0661
[3/5][100/1583] Loss_D: 1.1054  Loss_G: 0.8398  D(x): 0.4068    D(G(z)): 0.0367 / 0.4824
[3/5][150/1583] Loss_D: 0.6533  Loss_G: 3.4840  D(x): 0.8452    D(G(z)): 0.3526 / 0.0397
[3/5][200/1583] Loss_D: 1.0329  Loss_G: 5.1552  D(x): 0.9382    D(G(z)): 0.5595 / 0.0099
[3/5][250/1583] Loss_D: 0.5983  Loss_G: 2.1147  D(x): 0.6997    D(G(z)): 0.1575 / 0.1502
[3/5][300/1583] Loss_D: 0.5192  Loss_G: 2.5194  D(x): 0.7469    D(G(z)): 0.1680 / 0.1075
[3/5][350/1583] Loss_D: 0.6673  Loss_G: 3.6843  D(x): 0.8825    D(G(z)): 0.3780 / 0.0327
[3/5][400/1583] Loss_D: 0.6196  Loss_G: 2.9657  D(x): 0.8158    D(G(z)): 0.3037 / 0.0703
[3/5][450/1583] Loss_D: 0.6374  Loss_G: 2.1525  D(x): 0.7148    D(G(z)): 0.2192 / 0.1411
[3/5][500/1583] Loss_D: 1.0816  Loss_G: 4.4455  D(x): 0.9409    D(G(z)): 0.5828 / 0.0175
[3/5][550/1583] Loss_D: 0.5194  Loss_G: 2.2967  D(x): 0.8249    D(G(z)): 0.2538 / 0.1231
[3/5][600/1583] Loss_D: 0.4931  Loss_G: 2.5791  D(x): 0.7754    D(G(z)): 0.1792 / 0.0962
[3/5][650/1583] Loss_D: 1.7023  Loss_G: 5.8788  D(x): 0.9778    D(G(z)): 0.7488 / 0.0055
[3/5][700/1583] Loss_D: 0.4088  Loss_G: 2.5401  D(x): 0.8516    D(G(z)): 0.2010 / 0.0983
[3/5][750/1583] Loss_D: 0.7777  Loss_G: 1.2286  D(x): 0.5526    D(G(z)): 0.0612 / 0.3411
[3/5][800/1583] Loss_D: 1.2380  Loss_G: 0.4731  D(x): 0.3678    D(G(z)): 0.0373 / 0.6713
[3/5][850/1583] Loss_D: 0.7076  Loss_G: 1.8892  D(x): 0.6053    D(G(z)): 0.1102 / 0.1949
[3/5][900/1583] Loss_D: 0.7834  Loss_G: 3.6351  D(x): 0.9072    D(G(z)): 0.4594 / 0.0357
[3/5][950/1583] Loss_D: 0.4372  Loss_G: 2.8191  D(x): 0.7896    D(G(z)): 0.1591 / 0.0788
[3/5][1000/1583]        Loss_D: 0.8234  Loss_G: 3.7278  D(x): 0.9086    D(G(z)): 0.4615 / 0.0356
[3/5][1050/1583]        Loss_D: 0.7024  Loss_G: 2.0746  D(x): 0.7294    D(G(z)): 0.2606 / 0.1758
[3/5][1100/1583]        Loss_D: 0.5980  Loss_G: 1.8437  D(x): 0.7056    D(G(z)): 0.1843 / 0.1828
[3/5][1150/1583]        Loss_D: 0.4636  Loss_G: 3.5820  D(x): 0.8524    D(G(z)): 0.2313 / 0.0393
[3/5][1200/1583]        Loss_D: 1.0462  Loss_G: 4.5011  D(x): 0.9469    D(G(z)): 0.5803 / 0.0189
[3/5][1250/1583]        Loss_D: 0.6496  Loss_G: 2.9418  D(x): 0.8381    D(G(z)): 0.3441 / 0.0713
[3/5][1300/1583]        Loss_D: 0.4992  Loss_G: 1.8486  D(x): 0.7334    D(G(z)): 0.1265 / 0.1983
[3/5][1350/1583]        Loss_D: 0.7589  Loss_G: 1.4079  D(x): 0.5544    D(G(z)): 0.0740 / 0.3005
[3/5][1400/1583]        Loss_D: 1.2879  Loss_G: 1.1620  D(x): 0.3675    D(G(z)): 0.0784 / 0.3714
[3/5][1450/1583]        Loss_D: 0.6486  Loss_G: 1.9432  D(x): 0.6267    D(G(z)): 0.1112 / 0.1758
[3/5][1500/1583]        Loss_D: 0.9362  Loss_G: 3.9391  D(x): 0.9319    D(G(z)): 0.5191 / 0.0294
[3/5][1550/1583]        Loss_D: 0.7615  Loss_G: 3.8745  D(x): 0.8706    D(G(z)): 0.4283 / 0.0273
[4/5][0/1583]   Loss_D: 0.4990  Loss_G: 2.3274  D(x): 0.7407    D(G(z)): 0.1403 / 0.1228
[4/5][50/1583]  Loss_D: 0.4574  Loss_G: 2.9452  D(x): 0.8308    D(G(z)): 0.2130 / 0.0683
[4/5][100/1583] Loss_D: 0.9612  Loss_G: 1.7213  D(x): 0.4762    D(G(z)): 0.0650 / 0.2600
[4/5][150/1583] Loss_D: 0.5733  Loss_G: 2.0781  D(x): 0.6665    D(G(z)): 0.1000 / 0.1622
[4/5][200/1583] Loss_D: 0.5222  Loss_G: 2.1516  D(x): 0.7097    D(G(z)): 0.1199 / 0.1584
[4/5][250/1583] Loss_D: 0.7995  Loss_G: 1.7267  D(x): 0.5191    D(G(z)): 0.0408 / 0.2250
[4/5][300/1583] Loss_D: 0.5755  Loss_G: 2.8015  D(x): 0.7940    D(G(z)): 0.2633 / 0.0752
[4/5][350/1583] Loss_D: 0.4982  Loss_G: 2.4598  D(x): 0.7435    D(G(z)): 0.1446 / 0.1120
[4/5][400/1583] Loss_D: 0.7593  Loss_G: 1.6643  D(x): 0.5490    D(G(z)): 0.0580 / 0.2414
[4/5][450/1583] Loss_D: 0.4171  Loss_G: 2.2444  D(x): 0.8038    D(G(z)): 0.1538 / 0.1303
[4/5][500/1583] Loss_D: 0.5522  Loss_G: 2.0570  D(x): 0.7097    D(G(z)): 0.1333 / 0.1635
[4/5][550/1583] Loss_D: 0.9769  Loss_G: 4.4822  D(x): 0.9380    D(G(z)): 0.5568 / 0.0182
[4/5][600/1583] Loss_D: 0.5255  Loss_G: 2.8293  D(x): 0.8841    D(G(z)): 0.3013 / 0.0772
[4/5][650/1583] Loss_D: 0.7883  Loss_G: 4.1798  D(x): 0.9028    D(G(z)): 0.4440 / 0.0208
[4/5][700/1583] Loss_D: 0.9985  Loss_G: 1.3710  D(x): 0.4736    D(G(z)): 0.0912 / 0.2977
[4/5][750/1583] Loss_D: 0.5262  Loss_G: 2.5816  D(x): 0.7667    D(G(z)): 0.1927 / 0.1021
[4/5][800/1583] Loss_D: 1.2481  Loss_G: 0.3564  D(x): 0.3712    D(G(z)): 0.0419 / 0.7345
[4/5][850/1583] Loss_D: 0.7563  Loss_G: 4.1063  D(x): 0.8220    D(G(z)): 0.3856 / 0.0234
[4/5][900/1583] Loss_D: 1.6011  Loss_G: 1.1111  D(x): 0.2831    D(G(z)): 0.0573 / 0.3948
[4/5][950/1583] Loss_D: 0.7608  Loss_G: 3.3165  D(x): 0.8678    D(G(z)): 0.4050 / 0.0481
[4/5][1000/1583]        Loss_D: 0.5701  Loss_G: 2.0130  D(x): 0.6802    D(G(z)): 0.1260 / 0.1736
[4/5][1050/1583]        Loss_D: 0.7008  Loss_G: 1.1508  D(x): 0.6182    D(G(z)): 0.1357 / 0.3745
[4/5][1100/1583]        Loss_D: 0.5857  Loss_G: 3.7023  D(x): 0.9155    D(G(z)): 0.3475 / 0.0347
[4/5][1150/1583]        Loss_D: 2.4099  Loss_G: 0.5524  D(x): 0.1386    D(G(z)): 0.0063 / 0.6615
[4/5][1200/1583]        Loss_D: 0.7979  Loss_G: 3.9139  D(x): 0.9122    D(G(z)): 0.4529 / 0.0301
[4/5][1250/1583]        Loss_D: 0.6491  Loss_G: 1.6700  D(x): 0.6393    D(G(z)): 0.1188 / 0.2301
[4/5][1300/1583]        Loss_D: 0.5841  Loss_G: 3.6641  D(x): 0.9360    D(G(z)): 0.3676 / 0.0338
[4/5][1350/1583]        Loss_D: 3.3397  Loss_G: 0.1247  D(x): 0.0653    D(G(z)): 0.0168 / 0.8904
[4/5][1400/1583]        Loss_D: 0.7262  Loss_G: 1.7608  D(x): 0.5516    D(G(z)): 0.0429 / 0.2258
[4/5][1450/1583]        Loss_D: 1.3583  Loss_G: 3.6583  D(x): 0.9339    D(G(z)): 0.6417 / 0.0441
[4/5][1500/1583]        Loss_D: 0.8368  Loss_G: 1.2318  D(x): 0.5016    D(G(z)): 0.0459 / 0.3515
[4/5][1550/1583]        Loss_D: 0.6479  Loss_G: 4.0197  D(x): 0.9018    D(G(z)): 0.3793 / 0.0262
```

## 结果

​		最后，让我们看看我们是如何做到的。 在这里，我们将看三个不同的结果。 首先，我们将了解D和G的损失在训练过程中如何变化。 其次，我们将在每个时期将G的输出显示为fixed_noise批次。 第三，我们将查看一批真实数据以及来自G的一批伪数据。

### 损失与训练迭代

​		以下是D＆G的损失与训练迭代的关系图。

```python
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()
```

![https://pytorch.org/tutorials/_images/sphx_glr_dcgan_faces_tutorial_002.png](https://pytorch.org/tutorials/_images/sphx_glr_dcgan_faces_tutorial_002.png)

### 可视化G的进度

​		请记住，在每次训练之后，我们如何将生成器的输出保存为fixed_noise批次。 现在，我们可以用动画可视化G的训练进度。 按下播放按钮开始动画。

```python
#%%capture
fig = plt.figure(figsize=(8,8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

HTML(ani.to_jshtml())
```

![https://pytorch.org/tutorials/_images/sphx_glr_dcgan_faces_tutorial_003.png](https://pytorch.org/tutorials/_images/sphx_glr_dcgan_faces_tutorial_003.png)

### 实像与假像

​		最后，让我们并排查看一些真实图像和伪图像。

```python
# Grab a batch of real images from the dataloader
real_batch = next(iter(dataloader))

# Plot the real images
plt.figure(figsize=(15,15))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

# Plot the fake images from the last epoch
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1],(1,2,0)))
plt.show()
```

![https://pytorch.org/tutorials/_images/sphx_glr_dcgan_faces_tutorial_004.png](https://pytorch.org/tutorials/_images/sphx_glr_dcgan_faces_tutorial_004.png)