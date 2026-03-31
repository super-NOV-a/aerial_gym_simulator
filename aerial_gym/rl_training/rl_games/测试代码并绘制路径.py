import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from aerial_gym.utils.vae.VAE import VAE
from aerial_gym.registry.task_registry import task_registry
import torch
import torch.nn as nn
from matplotlib import font_manager
# 手动指定字体路径
font_path = '/usr/share/fonts/MyFonts/simhei.ttf'  # 替换为实际路径
font_prop = font_manager.FontProperties(fname=font_path)

# 设置字体
plt.rcParams['font.family'] = font_prop.get_name()
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# ==========================================
# 1. 模型定义 (保持不变)
# ==========================================

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

    def forward(self, x):
        x = (x - self.running_mean) / torch.sqrt(self.running_var + 1e-8)
        x = self.actor_mlp(x)
        mu = torch.tanh(self.mu(x))
        sigma = self.sigma.expand_as(mu)
        value = self.value(x)
        return mu, sigma, value

def load_rl_model(model_path, input_dim, output_dim, device):
    model = ActorCriticModel(input_dim, output_dim)
    checkpoint = torch.load(model_path, map_location=device)
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
    model.to(device)
    model.eval()
    return model

def load_vae_model(model_path, device, input_channels=1, latent_dims=64):
    vae_model = VAE(input_channels, latent_dims)
    checkpoint = torch.load(model_path, map_location=device)
    vae_model.load_state_dict(checkpoint)
    vae_model.to(device)
    vae_model.eval()
    return vae_model

# ==========================================
# 2. 改进后的绘图功能 (保持不变)
# ==========================================

def plot_trajectory(trajectory, obstacles, bounds, target_pos, episode_idx, save_dir="test_results8"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    fig, ax = plt.subplots(figsize=(12, 12))
    
    traj_x = [p[0] for p in trajectory]
    traj_y = [p[1] for p in trajectory]

    ax.scatter([], [], c='gray', s=200, alpha=0.5, label='障碍物', marker='o')
    # 绘制障碍物 (如果有)
    # if obstacles is not None:
    #     obs_positions = obstacles['positions']
    #     # ... (省略原有障碍物绘制代码，保持您原本的注释状态) ...

    # 绘制轨迹
    ax.plot(traj_x, traj_y, color='blue', linewidth=4.0, alpha=0.5, label='轨迹')
    ax.scatter(traj_x, traj_y, s=80, c='blue', marker='.', alpha=0.8, label='位置') 

    # 绘制起点和终点
    if len(traj_x) > 0:
        ax.scatter(traj_x[0], traj_y[0], c='green', s=600, label='起始点', marker='o', edgecolors='black', zorder=5)
        ax.scatter(traj_x[-1], traj_y[-1], c='red', s=600, label='结束点', marker='X', edgecolors='black', zorder=5)

    # 绘制目标点
    if target_pos is not None:
        ax.scatter(target_pos[0], target_pos[1], c='gold', s=800, label='目标点', marker='*', edgecolors='black', zorder=5)

    # 设置边界
    if bounds is not None:
        margin = 1.0
        ax.set_xlim(bounds[0] - margin, bounds[2] + margin)
        ax.set_ylim(bounds[1] - margin, bounds[3] + margin)
    else:
        ax.axis('equal')
    
    # === 修改部分开始 ===
    
    # 标题
    ax.set_title(f'导航过程示意', fontsize=32, pad=25, fontweight='bold')
    
    # 标签 (labelpad 从 15 改为 5，使其更靠近图像)
    ax.set_xlabel('X 位置 (m)', fontsize=28, labelpad=5)
    ax.set_ylabel('Y 位置 (m)', fontsize=28, labelpad=5)
    
    # 刻度设置
    ax.tick_params(axis='both', which='major', labelsize=24, width=2, length=8)
    
    # 图例 (loc 改为 'upper left' 左上角)
    ax.legend(loc='upper left', fontsize=24, framealpha=0.9, edgecolor='black')
    
    # === 修改部分结束 ===

    ax.grid(True, linestyle='--', alpha=0.6, linewidth=1.5)
    for spine in ax.spines.values():
        spine.set_linewidth(2)

    save_path = os.path.join(save_dir, f"episode_{episode_idx}_traj.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

# ==========================================
# 3. 主测试逻辑 (关键修改: 显式Seed控制)
# ==========================================

def run_test_and_visualize(vae_path, rl_path, base_seed=10):
    input_dim = 81
    output_dim = 4
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 初始 Seed 设置
    torch.manual_seed(base_seed)
    np.random.seed(base_seed)

    print("Loading models...")
    vae_model = load_vae_model(vae_path, device)
    rl_model = load_rl_model(rl_path, input_dim, output_dim, device)

    print("Creating environment...")
    # [修改点 1] num_envs 设为 1，确保我们可以精准控制每一轮的随机数
    env = task_registry.make_task("navigation_task", headless=True, num_envs=2, seed=base_seed, use_warp=False)
    
    # === 初始化 ===
    env.reset()
    
    start_pos = env.obs_dict["robot_position"][0].cpu().numpy()
    current_trajectory = [(start_pos[0], start_pos[1])]
    
    episode_count = 0
    max_episodes = 10
    
    # 获取环境边界
    env_bounds = None 
    if "env_bounds_min" in env.obs_dict:
        min_b = env.obs_dict["env_bounds_min"][0].cpu().numpy()
        max_b = env.obs_dict["env_bounds_max"][0].cpu().numpy()
        env_bounds = (min_b[0], min_b[1], max_b[0], max_b[1])

    print(f"Starting testing loop for {max_episodes} episodes...")

    with torch.no_grad():
        step_idx = 0
        while episode_count < max_episodes:
            current_obs_tensor = env.task_obs["observations"].to(device)
            actions, _, _ = rl_model(current_obs_tensor)
            
            # === 执行动作 ===
            task_obs, rewards, terminations, truncations, infos = env.step(actions)
            
            # 只有1个环境，直接取 index 0
            done = terminations[0] > 0 or truncations[0] > 0
            robot_pos = env.obs_dict["robot_position"][0].cpu().numpy()
            
            # === 记录路径 (无论是否结束) ===
            # 注意：如果done了，step后robot_pos可能已经被auto-reset重置到原点，也可能还在终点
            # 这里的处理取决于Aerial Gym的内部机制。
            # 通常 step 返回后的 robot_position 如果发生 reset 已经是新位置了。
            # 为了画图准确，如果是 done 的这一帧，通常不再 append 新位置，或者只 append 撞击前一刻的位置。
            # 这里简化处理：如果在 done 之前记录即可。
            
            if not done:
                current_trajectory.append((robot_pos[0], robot_pos[1]))
            
            step_idx += 1
            if step_idx % 100 == 0:
                print(f"Sim Step {step_idx}...")

            # === 关键修改逻辑: 处理回合结束 ===
            if done:
                print(f"Episode {episode_count} finished. Steps: {len(current_trajectory)}.")
                
                # 1. 保存当前回合的数据和图片
                obstacles_data = {}
                if "obstacle_positions" in env.obs_dict:
                    obstacles_data['positions'] = env.obs_dict["obstacle_positions"][0].cpu().numpy()
                elif "obstacle_position" in env.obs_dict:
                    obstacles_data['positions'] = env.obs_dict["obstacle_position"][0].cpu().numpy()
                
                if "obstacle_size" in env.obs_dict:
                    obstacles_data['sizes'] = env.obs_dict["obstacle_size"][0].cpu().numpy()
                
                target_pos = env.target_position[0].cpu().numpy()
                
                # 为了区分不同Seed，把 seed 传入 plot 函数（可选）
                current_seed = base_seed + episode_count 
                plot_trajectory(
                    current_trajectory, 
                    obstacles_data if 'positions' in obstacles_data else None, 
                    env_bounds,
                    target_pos,
                    episode_idx=current_seed  # 用Seed作为文件名的一部分，方便对应
                )
                
                # === 准备下一回合 ===
                episode_count += 1
                if episode_count >= max_episodes:
                    break

                # [修改点 2] 显式设置下一个回合的随机种子
                # 这样下一次 reset 生成的地图完全取决于这个数字，而与上一次跑了多少步无关
                next_seed = base_seed + episode_count
                print(f"Reseting environment with Seed: {next_seed}")
                
                torch.manual_seed(next_seed)
                np.random.seed(next_seed)
                
                # [修改点 3] 显式调用 reset() 重置环境状态
                # 这会触发环境重新生成地形/障碍物，利用刚才设置的 manual_seed
                env.reset()
                
                # 重置轨迹记录列表，记录新的起点
                start_pos = env.obs_dict["robot_position"][0].cpu().numpy()
                current_trajectory = [(start_pos[0], start_pos[1])]
                
                # 重置步数计数（可选，用于调试）
                step_idx = 0

    print("Testing complete.")

if __name__ == "__main__":
    # 请根据实际路径修改
    VAE_PATH = "/home/niu/workspaces/VAE_ws/agent_encoder/weights/dc_vae_beta300.0_LD_64_epoch_30.pth"
    # VAE_PATH = "/home/niu/workspaces/VAE_ws/agent_encoder/weights/ae_beta3.0_LD_64_epoch_30.pth"
    RL_PATH = "/home/niu/workspaces/aerial_gym_simulator/runs/dc_300.0/nn/gen_ppo.pth"
    
    if not os.path.exists(VAE_PATH) or not os.path.exists(RL_PATH):
        print("Error: Checkpoint paths are incorrect.")
    else:
        # 只要这里的 seed=15 不变，无论你换什么 RL 模型，
        # Episode 0 的地形永远是 Seed 15，Episode 1 永远是 Seed 16...
        run_test_and_visualize(VAE_PATH, RL_PATH, base_seed=15)