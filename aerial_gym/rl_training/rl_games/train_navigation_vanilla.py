# train_navigation_vanilla.py
import os
import yaml
import numpy as np
import isaacgym
from aerial_gym.registry.task_registry import task_registry
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner
import gym
from gym import spaces
import torch

# 注册环境
env_configurations.register(
    "navigation_vanilla_task",
    {
        "env_creator": lambda **kwargs: task_registry.make_task("navigation_vanilla_task", **kwargs),
        "vecenv_type": "AERIAL-RLGPU",
    },
)

class MultiModalExtractObsWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Dict({
            'image': gym.spaces.Box(
                low=-1.0, high=1.0, 
                shape=(1, 135, 240),
                dtype=np.float32
            ),
            'state': gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(17,),
                dtype=np.float32
            )
        })

    def reset(self, **kwargs):
        observations, *_ = super().reset(**kwargs)
        # 确保数据类型正确
        image_obs = observations['image_obs'].float()
        state_obs = observations['observations'][:, :17].float()
        return {
            'image': image_obs,
            'state': state_obs
        }

    def step(self, action):
        observations, rewards, terminated, truncated, infos = super().step(action)
        dones = torch.where(
            terminated | truncated,
            torch.ones_like(terminated),
            torch.zeros_like(terminated),
        )
        image_obs = observations['image_obs'].float()
        state_obs = observations['observations'][:, :17].float()
        return {
            'image': image_obs,
            'state': state_obs
        }, rewards, dones, infos

class EndToEndAERIALRLGPUEnv(vecenv.IVecEnv):
    def __init__(self, config_name, num_actors, **kwargs):
        self.env = env_configurations.configurations[config_name]["env_creator"](**kwargs)
        self.env = MultiModalExtractObsWrapper(self.env)

    def step(self, actions):
        return self.env.step(actions)

    def reset(self):
        return self.env.reset()

    def reset_done(self):
        return self.env.reset()

    def get_number_of_agents(self):
        return 1

    def get_env_info(self):
        info = {}
        info["action_space"] = gym.spaces.Box(
            -np.ones(4), np.ones(4), dtype=np.float32
        )
        info["observation_space"] = self.env.observation_space
        return info

vecenv.register(
    "AERIAL-RLGPU",
    lambda config_name, num_actors, **kwargs: EndToEndAERIALRLGPUEnv(config_name, num_actors, **kwargs),
)

def load_config(config_path, args):
    with open(config_path, "r") as stream:
        config = yaml.safe_load(stream)
    
    # 更新配置
    config["params"]["config"]["env_name"] = "navigation_vanilla_task"
    config["params"]["config"]["env_config"]["headless"] = args["headless"]
    config["params"]["config"]["env_config"]["num_envs"] = args["num_envs"]
    config["params"]["config"]["env_config"]["use_warp"] = args["use_warp"]
    
    # 关键：指定使用自定义网络构建器
    if 'network' not in config["params"]:
        config["params"]["network"] = {}
    config["params"]["network"]["name"] = "end_to_end_network"
    
    if args["num_envs"] > 0:
        config["params"]["config"]["num_actors"] = args["num_envs"]
        config["params"]["config"]["env_config"]["num_envs"] = args["num_envs"]
    
    if args["seed"] > 0:
        config["params"]["seed"] = args["seed"]
        config["params"]["config"]["env_config"]["seed"] = args["seed"]
    
    return config

def train_navigation(config_path, args):
    # 先注册自定义网络
    from custom_network_builder import register_custom_network
    register_custom_network()
    
    # 加载配置
    config = load_config(config_path, args)

    # 初始化Runner
    runner = Runner()
    
    try:
        runner.load(config)
    except Exception as e:
        print(f"Error loading config: {e}")
        import traceback
        traceback.print_exc()
        return

    # 启动训练
    runner.run(args)

if __name__ == "__main__":
    args = {
        "file": "aerial_gym/rl_training/rl_games/ppo_vanilla_navigation.yaml",
        "num_envs": 64,  # 先使用较小的环境数量进行测试
        "headless": True,
        "use_warp": True,
        "seed": 10,
        "train": True,
        "play": False,
        "track": False,
    }

    os.makedirs("runs", exist_ok=True)
    train_navigation(args["file"], args)