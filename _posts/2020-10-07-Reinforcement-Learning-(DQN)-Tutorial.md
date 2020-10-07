---
layout: post
title: "Reinforcement Learning (DQN) Tutorial"
subtitle: 'The concept and implementation of DQN'
author: "Yuzec"
header-style: text
tags:
  - DQN
  - RL
  - PyTorch
---

# 强化学习（DQN）教程

> 原文链接：https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

​		本教程显示了如何使用PyTorch在OpenAI Gym的CartPole-v0任务上训练深度Q学习（DQN）智能体。

**任务**

​		智能体必须在两个动作之间做出决定-向左或向右移动推车-以便使与之相连的电线杆保持直立。 您可以在Gym网站上找到具有各种算法和可视化效果的官方排行榜。

![https://pytorch.org/tutorials/_images/cartpole.gif](https://pytorch.org/tutorials/_images/cartpole.gif)

​		当智能体观察环境的当前状态并选择一个动作时，环境会转换为新状态，并返回指示该动作后果的奖励。 在此任务中，每增加一个时间步长，奖励为+1，并且如果杆子掉落得太远或手推车离中心的距离超过2.4个单位，则环境终止。 这意味着性能更好的方案将持续更长的时间，从而积累更大的回报。

​		对CartPole任务进行了设计，以便对智能体的输入是代表环境状态（位置，速度等）的4个实际值。 但是，神经网络可以完全通过查看场景来解决任务，因此我们将以购物车为中心的一部分屏幕作为输入。 因此，我们的结果无法直接与官方排行榜上的结果进行比较-我们的任务更加艰巨。 不幸的是，这确实减慢了训练速度，因为我们必须渲染所有框架。

​		严格来说，我们将状态显示为当前屏幕补丁与前一个屏幕补丁之间的差异。 这将允许智能体从一张图像考虑极点的速度。

**包**

​		首先，让我们导入所需的软件包。 首先，我们需要适用于环境的gym（使用pip install gym安装）。 我们还将使用PyTorch中的以下内容：

- 神经网络（torch.nn）
- 优化（torch.optim）
- 自动区分（torch.autograd）
- 视觉任务的实用程序（torchvision-单独的软件包）。

```python
import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


env = gym.make('CartPole-v0').unwrapped

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

## 重播记忆

​		我们将使用经验重播记忆来训练我们的DQN。 它存储智能体观察到的转换，使我们以后可以重用此数据。 通过从中随机抽样，可以建立批处理的转换相关。 已经表明，这极大地稳定并改善了DQN训练程序。

​		为此，我们将需要两个类：

- Transition-一个命名的元组，表示我们环境中的单个过渡。 它本质上将（状态，动作）对映射到其（next_state，奖励）结果，状态是屏幕差异图像，如下所述。
- ReplayMemory-有限大小的循环缓冲区，用于保存最近观察到的过渡。 它还实现了一个.sample()方法，用于选择随机批次的过渡进行训练。

```python
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
```

​		现在，让我们定义我们的模型。 但是首先，让我们快速回顾一下DQN是什么。

## DQN算法

​		我们的环境是确定性的，因此为简单起见，此处介绍的所有方程式也都确定性地制定。 在强化学习文献中，它们还将包含对环境中随机转变的期望。

​		我们的目标是训练一种试图最大化折现累积奖励$R_{t_0} = ∑^∞_{t =t_0}γ^{t-t_0}r_t$的策略，其中$R_{t_0}$也称为回报。 折扣γ应该是0到1之间的常数，以确保总和收敛。 它使不确定的远期回报对于我们的智能体而言不如对它相当有信心的近期回报重要。

​		Q学习的主要思想是，如果我们有一个函数$Q^∗：State×Action→\R$，那可以告诉我们返回的值是什么，如果我们要在给定的状态下执行一个动作，那么我们可以轻松地构造最大化我们回报的政策：
$$
\pi^*(s)=\mathop{argmax}_aQ^*(s,a)
$$
​		但是，我们对世界一无所知，因此无法访问$Q^∗$。 但是，由于神经网络是通用函数逼近器，因此我们可以简单地创建一个神经网络并将其训练为类似于$Q^∗$。

​		对于我们的训练更新规则，我们将使用一个事实，即某些策略的每个Q函数都遵循Bellman方程：
$$
Q^\pi(s,a)=r+\gamma Q^\pi(s',\pi(s'))
$$
​		等式两侧的差异称为时间差异误差δ：
$$
\delta=Q(s,a)-(r+\gamma\mathop{max}_aQ(s',a))
$$
​		为了最小化此错误，我们将使用Huber损耗。 当误差较小时，Huber损耗的作用类似于均方误差，而当误差较大时，则表现为平均绝对误差-当Q的估计值非常嘈杂时，这使它对异常值的鲁棒性更高。 我们通过从重播内存中采样的一批过渡B计算此值：
$$
\mathcal{L}=\frac{1}{B}\mathop{\sum}_{s,a,s',r\in{B}}\mathcal{L}(\delta)\\
where\ \mathcal{L}(\delta)=\begin{cases}\frac{1}{2}\delta^2&for\ |\delta|\le1,\\
|\delta|-\frac{1}{2}&otherwise.\end{cases}
$$

### Q网络

​		我们的模型将是一个卷积神经网络，该卷积神经网络将吸收当前屏幕补丁和先前屏幕补丁之间的差异。 它有两个输出，分别代表Q(s，left)和Q(s，right)（其中s是网络的输入）。 实际上，网络正在尝试预测在给定当前输入的情况下采取每个操作的预期收益。

```python
class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))
```

### 输入提取

​		以下代码是用于从环境中提取和处理渲染图像的实用程序。 它使用了torchvision软件包，可轻松组成图像变换。 一旦运行单元，它将显示它提取的示例补丁。

```python
resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])


def get_cart_location(screen_width):
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART

def get_screen():
    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW).
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    # Cart is in the lower half, so strip off the top and bottom of the screen
    _, screen_height, screen_width = screen.shape
    screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8)]
    view_width = int(screen_width * 0.6)
    cart_location = get_cart_location(screen_width)
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)
    # Strip off the edges, so that we have a square image centered on a cart
    screen = screen[:, :, slice_range]
    # Convert to float, rescale, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0).to(device)


env.reset()
plt.figure()
plt.imshow(get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy(),
           interpolation='none')
plt.title('Example extracted screen')
plt.show()
```

## 训练

### 超参数和实用程序

​		该单元实例化我们的模型及其优化器，并定义一些实用程序：

- select_action-将根据epsilon贪婪策略选择一个操作。 简而言之，我们有时会使用我们的模型来选择操作，有时我们会统一采样一次。 选择随机动作的可能性将从EPS_START开始，并朝EPS_END呈指数衰减。 EPS_DECAY控制衰减率。
- plot_durations-绘制情节持续时间以及最近100个情节的平均值（官方评估中使用的度量）的助手。 该情节将在包含主要训练循环的单元格下方，并且将在每个情节之后更新。

```python
BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

# Get screen size so that we can initialize layers correctly based on shape
# returned from AI gym. Typical dimensions at this point are close to 3x40x90
# which is the result of a clamped and down-scaled render buffer in get_screen()
init_screen = get_screen()
_, _, screen_height, screen_width = init_screen.shape

# Get number of actions from gym action space
n_actions = env.action_space.n

policy_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)


steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


episode_durations = []


def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())
```

### 训练循环

​		最后，是训练模型的代码。

​		在这里，您可以找到一个执行优化步骤的optimize_model函数。 它首先对一批进行采样，将所有张量连接成一个张量，然后计算$Q(s_t，a_t)$
和$V(s_{t + 1)}= max_aQ(s_{t + 1}，a)$，并将它们合并到我们的损失中。 通过定义，如果s为终端状态，我们将V(s)= 0设置为。 我们还使用目标网络来计算$V(s_{t + 1})$，以增加稳定性。 目标网络的权重大部分时间保持冻结状态，但经常更新以政策网络的权重。 通常这是一组固定的步骤，但是为了简单起见，我们将使用情节。

```python
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
```

​		在下面，您可以找到主要的训练循环。 首先，我们重置环境并初始化状态Tensor。 然后，我们对一个动作进行采样，执行它，观察下一个屏幕和奖励（总是1），并一次优化我们的模型。 当情节结束时（我们的模型失败），我们重新开始循环。

​		下面，将num_episodes设置为较小。 您应该下载笔记并运行更多episode，例如300多个episode，以实现有意义的持续时间改进。

```python
num_episodes = 50
for i_episode in range(num_episodes):
    # Initialize the environment and state
    env.reset()
    last_screen = get_screen()
    current_screen = get_screen()
    state = current_screen - last_screen
    for t in count():
        # Select and perform an action
        action = select_action(state)
        _, reward, done, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)

        # Observe new state
        last_screen = current_screen
        current_screen = get_screen()
        if not done:
            next_state = current_screen - last_screen
        else:
            next_state = None

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        optimize_model()
        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Complete')
env.render()
env.close()
plt.ioff()
plt.show()
```

​		这是说明总体结果数据流的图。

![https://pytorch.org/tutorials/_images/reinforcement_learning_diagram.jpg](https://pytorch.org/tutorials/_images/reinforcement_learning_diagram.jpg)

​		可以随机选择或根据策略选择动作，从gym环境中获取下一步样本。 我们将结果记录在重播内存中，并在每次迭代时运行优化步骤。 优化会从重播内存中随机抽取一批来训练新策略。 优化中还使用了“较旧”的target_net来计算期望的Q值； 有时会对其进行更新以使其保持最新状态。