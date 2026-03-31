import os
import yaml
import numpy as np
from aerial_gym.registry.task_registry import task_registry
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner
from rl_games.common.algo_observer import IsaacAlgoObserver
import gym
from gym import spaces
import torch
# 这将运行注册代码 (register_builder)
import aerial_gym.rl_training.rl_games.custom_network_builder
# --- 导入结束 ---

# 注册环境
env_configurations.register(
    "depth_navigation_task",
    {
        "env_creator": lambda **kwargs: task_registry.make_task("depth_navigation_task", **kwargs),
        "vecenv_type": "AERIAL-RLGPU",
    },
)

vecenv.register(
    "AERIAL-RLGPU",
    lambda config_name, num_actors, **kwargs: AERIALRLGPUEnv(config_name, num_actors, **kwargs),
)


class AERIALRLGPUEnv(vecenv.IVecEnv):
    def __init__(self, config_name, num_actors, **kwargs):
        self.env = env_configurations.configurations[config_name]["env_creator"](**kwargs)
        self.env = ExtractObsWrapper(self.env)

    def step(self, actions):
        return self.env.step(actions)

    def reset(self):
        return self.env.reset()

    def reset_done(self):
        return self.env.reset_done()

    def get_number_of_agents(self):
        return self.env.get_number_of_agents()

    def get_env_info(self):
        info = {}
        info["action_space"] = spaces.Box(
            -np.ones(self.env.task_config.action_space_dim),
            np.ones(self.env.task_config.action_space_dim),
        )
        info["observation_space"] = self.env.observation_space
        # info["observation_space"] = spaces.Box(
        #     np.ones(self.env.task_config.observation_space_dim) * -np.Inf,
        #     np.ones(self.env.task_config.observation_space_dim) * np.Inf,
        # )
        return info


class ExtractObsWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        observations, *_ = super().reset(**kwargs)
        return observations

    def step(self, action):
        observations, rewards, terminated, truncated, infos = super().step(action)
        dones = torch.where(
            terminated | truncated,
            torch.ones_like(terminated),
            torch.zeros_like(terminated),
        )
        # 关键：rl_games 需要这两个键来区分“为什么”回合结束了
        infos["terminated"] = terminated.float() 
        infos["truncated"] = truncated.float()
        # infos 字典现在同时包含 "episode":{...}, "terminated":..., "truncated":...
        return observations, rewards, dones, infos


def load_config(config_path, args):
    with open(config_path, "r") as stream:
        config = yaml.safe_load(stream)
    # 更新配置
    config["params"]["config"]["env_name"] = "depth_navigation_task"
    config["params"]["config"]["env_config"]["headless"] = args["headless"]
    config["params"]["config"]["env_config"]["num_envs"] = args["num_envs"]
    config["params"]["config"]["env_config"]["use_warp"] = args["use_warp"]
    if args["num_envs"] > 0:
        config["params"]["config"]["num_actors"] = args["num_envs"]
        config["params"]["config"]["env_config"]["num_envs"] = args["num_envs"]
    if args["seed"] > 0:
        config["params"]["seed"] = args["seed"]
        config["params"]["config"]["env_config"]["seed"] = args["seed"]
    return config


def train_navigation(config_path, args):
    # 加载配置
    config = load_config(config_path, args)
    # --- 在这里添加调试代码 ---
    print("="*50)
    print(f"DEBUG: 正在加载的配置文件: {config_path}")
    if "network" in config["params"]:
        print("DEBUG: 找到 'network' 配置:")
        print(yaml.dump(config["params"]["network"])) # 打印出加载的网络配置
    else:
        print("DEBUG: 错误! 在 'config[\"params\"]' 中未找到 'network' 键。")
    print("="*50)
    # --- 调试代码结束 ---
    # 初始化Runner
    observer = IsaacAlgoObserver()  # <-- 1. 实例化观察者
    runner = Runner(observer)      # <-- 2. 将观察者传递给 Runner
    try:
        runner.load(config)
    except yaml.YAMLError as exc:
        print(exc)
        return

    # 启动训练
    runner.run(args)


if __name__ == "__main__":
    # 设置参数
    args = {
        "file": "aerial_gym/rl_training/rl_games/ppo_depth_navigation.yaml",
        "num_envs": 256,    # 256, test:64
        # "checkpoint": "/home/niu/workspaces/aerial_gym_simulator/aerial_gym/rl_training/rl_games/runs/520_3_10/nn/gen_ppo.pth",
        "headless": True,
        "use_warp": True,
        "seed": 10,
        "train": True,
        "play": False,
        "track": False,     # 是否使用 WandB
        "wandb_project_name": "rl_games",
        "wandb_entity": None,
    }

    # 创建必要的目录
    os.makedirs("runs", exist_ok=True)

    # 启动训练
    train_navigation(args["file"], args)