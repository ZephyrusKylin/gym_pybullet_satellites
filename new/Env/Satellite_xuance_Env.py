import time
import numpy as np
from gymnasium.spaces import Box
from xuance.environment import RawMultiAgentEnv
from satellite_rl_env import BaseSatelliteEnv  # 假设上一步生成的文件名为 satellite_rl_env.py

# 场景类：多对多防御-打击对抗
class MultiDefenseAviary(BaseSatelliteEnv):
    """
    红方使用护卫卫星防御蓝方的打击卫星，保护高价值卫星不被击中。
    配置文件包含红/蓝方卫星属性：类型、初始状态、物理参数。
    """
    def __init__(self,
                 config_file: str,
                 gui: bool = False,
                 timestep: float = 10.0,
                 max_steps: int = 1000):
        # 直接调用基础环境
        super().__init__(config_file=config_file,
                         timestep=timestep,
                         max_steps=max_steps)
        self.GUI = gui

    # 渲染占位：可以集成 Unity 或 PyBullet
    def render(self, mode='human'):
        if self.GUI:
            # TODO: 实现 PyBullet 或 Unity 渲染
            pass
        else:
            # 文本渲染
            print(f"Step {self.current_step}")
            for i, sat in enumerate(self.sats):
                st = self.state[i]
                print(f"Sat {i} ({sat['type']}): pos={st[0:3]}, vel={st[3:6]}, m={st[6]:.2f}")

# 可注册场景字典
REGISTRY = {
    "MultiDefenseAviary": MultiDefenseAviary,
}

# Xuance 多智能体环境封装
class Satellite_MultiAgentEnv(RawMultiAgentEnv):
    def __init__(self, config):
        super().__init__()
        self.env_id = config.env_id
        self.gui    = config.render
        self.env    = None
        # 构造子环境
        if self.env_id in REGISTRY:
            # 传递 YAML 配置路径
            kwargs = {
                'config_file': config.config_file,
                'gui': self.gui,
                'timestep': getattr(config, 'timestep', 10.0),
                'max_steps': getattr(config, 'max_steps', 1000)
            }
            self.env = REGISTRY[self.env_id](**kwargs)
        else:
            raise ValueError(f"Unknown env_id {self.env_id}")
        # 初始化底层环境
        self.env.reset()
        # 智能体列表
        self.num_agents = self.env.n_red + self.env.n_blue
        self.agents = [f"agent_{i}" for i in range(self.num_agents)]
        # 定义空间
        act_dim = self.env.action_space.shape[-1]
        obs_dim = self.env.observation_space.shape[-1]
        self.action_space = {a: Box(-1, 1, (act_dim,)) for a in self.agents}
        self.observation_space = {a: Box(-np.inf, np.inf, (obs_dim,)) for a in self.agents}
        self.max_episode_steps = config.max_episode_steps
        self._episode_step = 0

    def reset(self):
        obs, info = self.env.reset()
        self._episode_step = 0
        obs_dict = {agent: obs[i] for i, agent in enumerate(self.agents)}
        return obs_dict, info

    def step(self, actions):
        # 收集多智能体动作
        act_array = np.vstack([actions[a] for a in self.agents])
        obs, rewards, terminated, truncated, info = self.env.step(act_array)
        obs_dict = {agent: obs[i] for i, agent in enumerate(self.agents)}
        reward_dict = {agent: rewards[i] for i, agent in enumerate(self.agents)}
        terminated_dict = {agent: terminated for agent in self.agents}
        self._episode_step += 1
        truncated = self._episode_step >= self.max_episode_steps
        info['episode_step'] = self._episode_step
        if self.gui:
            time.sleep(config.sleep)
        return obs_dict, reward_dict, terminated_dict, truncated, info

    def close(self):
        self.env.close()

    def agent_mask(self):
        return {agent: True for agent in self.agents}

    def state(self):
        # 可以返回全局观测
        return self.env.state.flatten()
