from aerial_gym.utils.vae.VAE import VAE
from aerial_gym.registry.task_registry import task_registry
import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCriticModel(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(ActorCriticModel, self).__init__()

        # 输入标准化层
        self.running_mean = nn.Parameter(torch.zeros(input_dim))
        self.running_var = nn.Parameter(torch.ones(input_dim))
        self.running_count = nn.Parameter(torch.tensor(1.0), requires_grad=False)

        self.value_running_mean = nn.Parameter(torch.zeros(1))
        self.value_running_var = nn.Parameter(torch.ones(1))
        self.value_running_count = nn.Parameter(torch.tensor(1.0), requires_grad=False)

        # 策略网络和价值网络共享的MLP
        self.actor_mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU()
        )

        # 策略网络（动作均值）
        self.mu = nn.Linear(64, action_dim)

        # 动作标准差（固定）
        self.sigma = nn.Parameter(torch.zeros(action_dim))

        # 价值网络
        self.value = nn.Linear(64, 1)

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        # 根据配置文件中的初始化方式
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=2)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # 输入标准化
        x = (x - self.running_mean) / torch.sqrt(self.running_var + 1e-8)
        # MLP主体
        x = self.actor_mlp(x)
        # 动作均值，应用tanh激活函数限制在-1到1之间
        mu = torch.tanh(self.mu(x))
        # 动作标准差（固定）
        sigma = self.sigma.expand_as(mu)
        # 价值
        value = self.value(x)
        return mu, sigma, value

def load_rl_model(model_path, input_dim, output_dim):
    model = ActorCriticModel(input_dim, output_dim)
    checkpoint = torch.load(model_path, map_location=torch.device('cuda:0'))

    # 创建一个映射字典，将检查点文件中的键名映射到模型的键名
    state_dict = checkpoint['model']
    model_state_dict = model.state_dict()

    # 手动映射键名
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

    # 应用映射
    for checkpoint_key, model_key in key_mapping.items():
        if checkpoint_key in state_dict:
            model_state_dict[model_key] = state_dict[checkpoint_key]

    # 加载权重
    model.load_state_dict(model_state_dict, strict=False)
    model.eval()
    return model

def load_vae_model(model_path, input_channels=1, latent_dims=64):
    vae_model = VAE(input_channels, latent_dims)
    checkpoint = torch.load(model_path, map_location=torch.device('cuda:0'))
    vae_model.load_state_dict(checkpoint)
    vae_model.eval()
    return vae_model

def test_navigation_task(vae_model_path, rl_model_path, input_dim, output_dim):
    # 加载VAE模型
    # vae_model = load_vae_model(vae_model_path, input_channels=1, latent_dims=64)
    # 加载RL模型
    rl_model = load_rl_model(rl_model_path, input_dim, output_dim)
    # 创建环境
    rl_task_env = task_registry.make_task("navigation_task", headless=False, num_envs=2, use_warp=False)
    rl_task_env.reset()

    # 确保模型在GPU上
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rl_model.to(device)
    # vae_model.to(device)

    # 测试模型
    with torch.no_grad():
        # 获取观测
        obs, reward, terminated, truncated, info = rl_task_env.step(
            torch.zeros((rl_task_env.task_config.num_envs, rl_task_env.task_config.action_space_dim)).to(device))

        for i in range(10000):
            # 确保输入数据在GPU上
            observations = obs['observations'].to(device)
            # 前向传播
            actions, _, _ = rl_model(observations)
            # actions = actions.zero_()   # 动作置零方便截图
            # 执行动作
            obs, reward, terminated, truncated, info = rl_task_env.step(actions.to(device))
            # 打印信息
            if i % 100 == 0:
                print(f"Step {i}, Reward: {reward.mean().item()}")

def analyze_checkpoint(model_path):
    checkpoint = torch.load(model_path, map_location=torch.device('cuda:0'))
    print("Keys in checkpoint:", checkpoint.keys())
    print("Keys in model state dict:", checkpoint['model'].keys())


if __name__ == "__main__":
    # VAE模型与RL模型应该保持一致，VAE模型在config文件中修改
    vae_model_path = "/home/niu/workspaces/VAE_ws/agent_encoder/weights/dc_vae_beta100.0_LD_64_epoch_30.pth"
    rl_model_path = "/home/niu/workspaces/aerial_gym_simulator/runs/dc_100.0/nn/gen_ppo.pth"
    input_dim = 81  # 原始观测维度加上VAE编码后的维度
    output_dim = 4  # 根据你的环境设置输出维度
    # analyze_checkpoint(rl_model_path)
    test_navigation_task(vae_model_path, rl_model_path, input_dim, output_dim)