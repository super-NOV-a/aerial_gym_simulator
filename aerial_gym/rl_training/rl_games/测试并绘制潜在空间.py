import os
from aerial_gym.utils.vae.VAE import VAE
from aerial_gym.registry.task_registry import task_registry
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.cm as cm  # 引入色谱模块
import matplotlib.font_manager as fm

# --- 字体设置 ---
FONT_PATHS = [
    '/usr/share/fonts/MyFonts/simhei.ttf',
    'C:/Windows/Fonts/simhei.ttf',
    '/System/Library/Fonts/STHeiti Light.ttc',
    'simhei.ttf'
]
LEGEND_FONTSIZE = 15
font_prop = None
font_options = {}

for path in FONT_PATHS:
    if os.path.exists(path):
        try:
            font_prop = fm.FontProperties(fname=path, size=LEGEND_FONTSIZE)
            plt.rcParams['font.family'] = font_prop.get_name()
            plt.rcParams['axes.unicode_minus'] = False
            font_options = {'fontproperties': font_prop}
            
            # --- [全局字体大小设置] ---
            plt.rcParams['font.size'] = 18          # 基础字体大小
            plt.rcParams['axes.titlesize'] = 24     # 标题字体大小
            plt.rcParams['axes.labelsize'] = 20     # X/Y轴标签字体大小
            plt.rcParams['xtick.labelsize'] = 16    # X轴刻度字体大小
            plt.rcParams['ytick.labelsize'] = 16    # Y轴刻度字体大小
            plt.rcParams['legend.fontsize'] = 18    # 图例字体大小
            # ---------------------------
            
            break
        except:
            continue
        
# --- 保持 ActorCriticModel 和 load_rl_model 等定义不变 ---

class ActorCriticModel(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(ActorCriticModel, self).__init__()
        self.running_mean = nn.Parameter(torch.zeros(input_dim))
        self.running_var = nn.Parameter(torch.ones(input_dim))
        self.running_count = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.value_running_mean = nn.Parameter(torch.zeros(1))
        self.value_running_var = nn.Parameter(torch.ones(1))
        self.value_running_count = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.actor_mlp = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 64), nn.ELU()
        )
        self.mu = nn.Linear(64, action_dim)
        self.sigma = nn.Parameter(torch.zeros(action_dim))
        self.value = nn.Linear(64, 1)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=2)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = (x - self.running_mean) / torch.sqrt(self.running_var + 1e-8)
        x = self.actor_mlp(x)
        mu = torch.tanh(self.mu(x))
        sigma = self.sigma.expand_as(mu)
        value = self.value(x)
        return mu, sigma, value

def load_rl_model(model_path, input_dim, output_dim):
    model = ActorCriticModel(input_dim, output_dim)
    checkpoint = torch.load(model_path, map_location=torch.device('cuda:0'))
    state_dict = checkpoint['model']
    model_state_dict = model.state_dict()
    key_mapping = {
        'value_mean_std.running_mean': 'value_running_mean',
        'value_mean_std.running_var': 'value_running_var',
        'value_mean_std.count': 'value_running_count',
        'running_mean_std.running_mean': 'running_mean',
        'running_mean_std.running_var': 'running_var',
        'running_mean_std.count': 'running_count',
        'a2c_network.sigma': 'sigma',
        'a2c_network.actor_mlp.0.weight': 'actor_mlp.0.weight',
        'a2c_network.actor_mlp.0.bias': 'actor_mlp.0.bias',
        'a2c_network.actor_mlp.2.weight': 'actor_mlp.2.weight',
        'a2c_network.actor_mlp.2.bias': 'actor_mlp.2.bias',
        'a2c_network.actor_mlp.4.weight': 'actor_mlp.4.weight',
        'a2c_network.actor_mlp.4.bias': 'actor_mlp.4.bias',
        'a2c_network.value.weight': 'value.weight',
        'a2c_network.value.bias': 'value.bias',
        'a2c_network.mu.weight': 'mu.weight',
        'a2c_network.mu.bias': 'mu.bias'
    }
    for checkpoint_key, model_key in key_mapping.items():
        if checkpoint_key in state_dict:
            model_state_dict[model_key] = state_dict[checkpoint_key]
    model.load_state_dict(model_state_dict, strict=False)
    model.eval()
    return model

def plot_results(episodes_data):
    """
    绘制结果：
    1. t-SNE 潜在空间轨迹 (前3次使用红绿蓝渐变散点，其余灰色背景)
    2. 位置和姿态变化曲线
    """
    labels = ['Episode 1', 'Episode 2', 'Episode 3']
    # 定义使用的色谱名称
    cmap_names = ['Reds', 'Greens', 'Blues']
    line_colors = ['red', 'green', 'blue'] # 用于第二张图的线条颜色
    
    # --- 1. 处理 t-SNE ---
    print(f"正在对 {len(episodes_data)} 个 Episode 的数据进行 t-SNE 降维处理...")
    all_latents = np.concatenate([d['latents'] for d in episodes_data], axis=0)
    
    # 使用 init='pca' 以获得更稳定的轨迹结果
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    all_latents_2d = tsne.fit_transform(all_latents)
    
    # 将数据拆分回各个 episode
    episodes_2d = []
    start_idx = 0
    for d in episodes_data:
        length = len(d['latents'])
        episodes_2d.append(all_latents_2d[start_idx : start_idx+length])
        start_idx += length

    # --- 2. 绘制 t-SNE 轨迹 (渐变散点图) ---
    plt.figure(figsize=(12, 10))
    
    # [第一步] 绘制背景噪声 (Episode 4 - 10)
    print("绘制背景特征点...")
    for i in range(3, len(episodes_2d)):
        ep_2d = episodes_2d[i]
        plt.scatter(ep_2d[:, 0], ep_2d[:, 1], c='gray', s=30, alpha=0.25, label='Background' if i == 3 else "")

    # [第二步] 绘制前景轨迹 (Episode 1 - 3, 渐变色)
    print("绘制主要轨迹...")
    for i in range(3): # 只画前3个
        if i >= len(episodes_2d): break
        
        ep_2d = episodes_2d[i]
        x = ep_2d[:, 0]
        y = ep_2d[:, 1]
        num_steps = len(x)
        
        # 生成渐变颜色: 从浅色 (0.3) 到 深色 (1.0)
        # 使用对应的色谱 (Reds, Greens, Blues)
        cmap = plt.get_cmap(cmap_names[i])
        color_indices = np.linspace(0.25, 1.0, num_steps)
        colors = cmap(color_indices)
        
        # 绘制散点
        # s=40 设置点的大小，alpha=0.8 设置不透明度
        scatter = plt.scatter(x, y, c=colors, s=40, alpha=0.8, label=labels[i])
        
        # 额外标记起点和终点，增加辨识度
        # 起点用空心圆，终点用深色叉号
        plt.scatter(x[0], y[0], s=150, facecolors='none', edgecolors=colors[0], linewidth=2, marker='o', label=f'{labels[i]} Start')
        plt.scatter(x[-1], y[-1], c=[colors[-1]], s=150, marker='x', linewidth=3, label=f'{labels[i]} End')

    plt.title("DC-VAE 潜在空间轨迹 (t-SNE)")
    plt.xlabel("第1个维度")
    plt.ylabel("第2个维度")
    
    # 重新组织图例
    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', alpha=0.6, markersize=15, label='Episode 1 (Red)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', alpha=0.6, markersize=15, label='Episode 2 (Green)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', alpha=0.6, markersize=15, label='Episode 3 (Blue)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', alpha=0.5, markersize=15, label='其他 Episodes (Gray)'),
    ]
    plt.legend(handles=custom_lines, loc='best')
    
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig('vae_tsne_gradient_dots.png')
    plt.show()

    # --- 3. 绘制位置和姿态 (仅前3个) ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    state_labels = [['Pos X', 'Pos Y', 'Pos Z'], ['Roll', 'Pitch', 'Yaw']]
    
    for i in range(3): # 限制只画前3个
        if i >= len(episodes_data): break
        
        data = episodes_data[i]
        pos = data['pos']
        ori = data['ori']
        steps = np.arange(len(pos))
        
        # 绘制位置 (X, Y, Z)
        for j in range(3):
            axes[0, j].plot(steps, pos[:, j], color=line_colors[i], label=labels[i])
            axes[0, j].set_title(state_labels[0][j])
            axes[0, j].set_xlabel('Step')
            axes[0, j].set_ylabel('Position (m)')
            axes[0, j].grid(True, alpha=0.3)
        
        # 绘制姿态 (Roll, Pitch, Yaw)
        for j in range(3):
            axes[1, j].plot(steps, ori[:, j], color=line_colors[i], label=labels[i])
            axes[1, j].set_title(state_labels[1][j])
            axes[1, j].set_xlabel('Step')
            axes[1, j].set_ylabel('Angle (rad)')
            axes[1, j].grid(True, alpha=0.3)

    # 仅在第一个子图中显示图例
    axes[0, 0].legend()
    plt.tight_layout()
    plt.savefig('navigation_states_v3.png')
    plt.show()


def test_navigation_task_with_plots(rl_model_path, input_dim, output_dim):
    # 加载RL模型
    rl_model = load_rl_model(rl_model_path, input_dim, output_dim)
    
    # 创建环境
    print("Initializing environment...")
    rl_task_env = task_registry.make_task("navigation_task", headless=True, num_envs=2, use_warp=False)
    rl_task_env.reset()

    # 确保模型在GPU上
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rl_model.to(device)

    # 存储数据
    episodes_data = []
    target_episodes = 3  # 收集 10 个 episodes
    
    current_ep_latents = []
    current_ep_pos = []
    current_ep_ori = []

    print(f"Starting collection of {target_episodes} episodes...")
    print(f"Note: Only the first 60 steps of each episode will be recorded.")

    with torch.no_grad():
        # 获取初始观测
        obs, reward, terminated, truncated, info = rl_task_env.step(
            torch.zeros((rl_task_env.task_config.num_envs, rl_task_env.task_config.action_space_dim)).to(device))
        
        step_count = 0
        while len(episodes_data) < target_episodes:
            # 1. 获取观测数据
            observations = obs['observations'].to(device)
            
            # 2. 收集当前帧的数据 [修改点: 仅收集前60步]
            if len(current_ep_pos) < 90:
                latent = observations[0, 17:].cpu().numpy().copy()
                pos = rl_task_env.obs_dict["robot_position"][0].cpu().numpy().copy()
                ori = rl_task_env.obs_dict["robot_euler_angles"][0].cpu().numpy().copy()
                
                current_ep_latents.append(latent)
                current_ep_pos.append(pos)
                current_ep_ori.append(ori)
            
            # 3. RL 推理
            actions, _, _ = rl_model(observations)
            
            # 4. 执行动作
            obs, reward, terminated, truncated, info = rl_task_env.step(actions.to(device))
            
            step_count += 1
            if step_count % 100 == 0:
                print(f"Current Environment Step: {step_count}, Collected Episodes: {len(episodes_data)}/{target_episodes}")

            # 5. 检查 Episode 是否结束 [逻辑: 等待环境自然结束以重置，但只保存已收集的前60步]
            if terminated[0] or truncated[0]:
                print(f"Episode {len(episodes_data) + 1} finished in environment. (Recorded steps: {len(current_ep_pos)})")
                
                # 只有当收集到了数据才保存
                if len(current_ep_pos) > 0:
                    episodes_data.append({
                        'latents': np.array(current_ep_latents),
                        'pos': np.array(current_ep_pos),
                        'ori': np.array(current_ep_ori)
                    })
                
                # 重置缓冲区，准备下一个 Episode
                current_ep_latents = []
                current_ep_pos = []
                current_ep_ori = []
                
                if len(episodes_data) >= target_episodes:
                    break

    print("Data collection complete. Generating plots...")
    plot_results(episodes_data)

if __name__ == "__main__":
    rl_model_path = "/home/niu/workspaces/aerial_gym_simulator/runs/dc_100.0/nn/gen_ppo.pth"
    input_dim = 81  # 原始观测维度加上VAE编码后的维度
    output_dim = 4  # 根据你的环境设置输出维度
    
    test_navigation_task_with_plots(rl_model_path, input_dim, output_dim)