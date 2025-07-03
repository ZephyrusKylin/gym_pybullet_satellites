# """
# 太空多智能体强化学习环境主类
# 实现基于GEO轨道的卫星编队作战环境
# """

# import numpy as np
# import gym
# from gym import spaces
# from typing import Dict, List, Tuple, Optional, Any
# import logging
# from dataclasses import dataclass
# from enum import Enum

# from .orbital_dynamics import OrbitalDynamics
# from .sensor_models import SensorNetwork
# from .action_parser import ActionParser


# class SatelliteType(Enum):
#     """卫星类型枚举"""
#     OBSERVER = "observer"          # 观测卫星
#     DEFENDER = "defender"          # 防卫卫星
#     HIGH_VALUE = "high_value"      # 高价值卫星
#     ENEMY_ATTACKER = "enemy_attacker"  # 敌方攻击卫星


# @dataclass
# class SatelliteState:
#     """卫星状态数据结构"""
#     sat_id: int
#     sat_type: SatelliteType
#     position: np.ndarray  # [x, y, z] in km
#     velocity: np.ndarray  # [vx, vy, vz] in km/s
#     fuel_remaining: float  # 0-1 normalized
#     health_status: float   # 0-1 normalized
#     capabilities: Dict[str, float]  # 能力参数
#     last_action: Optional[Dict] = None


# class SpaceEnvironment(gym.Env):
#     """太空多智能体强化学习环境"""
    
#     def __init__(self, config: Dict[str, Any]):
#         super().__init__()
        
#         # 环境配置
#         self.config = config
#         self.max_steps = config.get('max_steps', 1000)
#         self.dt = config.get('time_step', 60.0)  # 时间步长(秒)
#         self.geo_altitude = 35786.0  # GEO轨道高度(km)
#         self.area_radius = 2000.0    # 作战区域半径(km)
        
#         # 场景配置 (3观测+1防卫+1高价值 vs 1敌方)
#         self.n_observers = config.get('n_observers', 3)
#         self.n_defenders = config.get('n_defenders', 1)
#         self.n_high_value = config.get('n_high_value', 1)
#         self.n_enemies = config.get('n_enemies', 1)
#         self.n_agents = self.n_observers + self.n_defenders + self.n_high_value
        
#         # 初始化子系统
#         self.orbital_dynamics = OrbitalDynamics(self.geo_altitude)
#         self.sensor_network = SensorNetwork(config.get('sensor_config', {}))
#         self.action_parser = ActionParser(config.get('action_config', {}))
        
#         # 环境状态
#         self.satellites: Dict[int, SatelliteState] = {}
#         self.current_step = 0
#         self.simulation_time = 0.0
#         self.episode_rewards = {}
        
#         # 定义观测和动作空间
#         self._setup_spaces()
        
#         # 日志配置
#         logging.basicConfig(level=logging.INFO)
#         self.logger = logging.getLogger(__name__)
        
#     def _setup_spaces(self):
#         """设置观测空间和动作空间"""
#         # 单个卫星的观测维度
#         # 位置(3) + 速度(3) + 燃料(1) + 健康度(1) + 能力(3) = 11
#         single_sat_obs_dim = 11
        
#         # 全局观测维度: 所有卫星状态 + 威胁评估 + 任务状态 + 时间信息
#         total_sats = self.n_agents + self.n_enemies
#         global_obs_dim = (total_sats * single_sat_obs_dim + 
#                          5 +  # 威胁评估维度
#                          3 +  # 任务状态维度
#                          2)   # 时间信息维度
        
#         # 观测空间 - 每个智能体都观测到全局状态
#         self.observation_space = spaces.Dict({
#             f'agent_{i}': spaces.Box(
#                 low=-np.inf, high=np.inf, 
#                 shape=(global_obs_dim,), dtype=np.float32
#             ) for i in range(self.n_agents)
#         })
        
#         # 动作空间 - 每个智能体的动作空间
#         # 机动指令(6) + 观测指令(4) + 协调指令(3) = 13维连续动作
#         action_dim = 13
#         self.action_space = spaces.Dict({
#             f'agent_{i}': spaces.Box(
#                 low=-1.0, high=1.0,
#                 shape=(action_dim,), dtype=np.float32
#             ) for i in range(self.n_agents)
#         })
    
#     def reset(self, seed: Optional[int] = None) -> Dict[str, np.ndarray]:
#         """重置环境到初始状态"""
#         if seed is not None:
#             np.random.seed(seed)
        
#         self.current_step = 0
#         self.simulation_time = 0.0
#         self.satellites.clear()
#         self.episode_rewards = {f'agent_{i}': 0.0 for i in range(self.n_agents)}
        
#         # 初始化卫星状态
#         self._initialize_satellites()
        
#         # 获取初始观测
#         observations = self._get_observations()
        
#         self.logger.info(f"Environment reset with {len(self.satellites)} satellites")
#         return observations
    
#     def _initialize_satellites(self):
#         """初始化所有卫星的状态"""
#         sat_id = 0
        
#         # 初始化观测卫星 - 分布在GEO轨道不同位置
#         for i in range(self.n_observers):
#             angle = 2 * np.pi * i / self.n_observers
#             position = self._generate_geo_position(angle, offset_radius=100.0)
#             velocity = self.orbital_dynamics.compute_orbital_velocity(position)
            
#             self.satellites[sat_id] = SatelliteState(
#                 sat_id=sat_id,
#                 sat_type=SatelliteType.OBSERVER,
#                 position=position,
#                 velocity=velocity,
#                 fuel_remaining=1.0,
#                 health_status=1.0,
#                 capabilities={
#                     'observation_range': 1000.0,  # km
#                     'maneuver_capability': 0.7,
#                     'data_processing': 0.9
#                 }
#             )
#             sat_id += 1
        
#         # 初始化防卫卫星
#         for i in range(self.n_defenders):
#             angle = np.pi + 2 * np.pi * i / self.n_defenders
#             position = self._generate_geo_position(angle, offset_radius=200.0)
#             velocity = self.orbital_dynamics.compute_orbital_velocity(position)
            
#             self.satellites[sat_id] = SatelliteState(
#                 sat_id=sat_id,
#                 sat_type=SatelliteType.DEFENDER,
#                 position=position,
#                 velocity=velocity,
#                 fuel_remaining=1.0,
#                 health_status=1.0,
#                 capabilities={
#                     'intercept_range': 500.0,  # km
#                     'maneuver_capability': 1.0,
#                     'weapon_systems': 0.8
#                 }
#             )
#             sat_id += 1
        
#         # 初始化高价值卫星
#         for i in range(self.n_high_value):
#             position = self._generate_geo_position(0.0, offset_radius=0.0)  # 中心位置
#             velocity = self.orbital_dynamics.compute_orbital_velocity(position)
            
#             self.satellites[sat_id] = SatelliteState(
#                 sat_id=sat_id,
#                 sat_type=SatelliteType.HIGH_VALUE,
#                 position=position,
#                 velocity=velocity,
#                 fuel_remaining=0.8,
#                 health_status=1.0,
#                 capabilities={
#                     'strategic_value': 1.0,
#                     'maneuver_capability': 0.3,
#                     'defensive_systems': 0.6
#                 }
#             )
#             sat_id += 1
        
#         # 初始化敌方卫星
#         for i in range(self.n_enemies):
#             angle = np.pi / 2 + 2 * np.pi * i / self.n_enemies
#             position = self._generate_geo_position(angle, offset_radius=800.0)
#             velocity = self.orbital_dynamics.compute_orbital_velocity(position)
            
#             self.satellites[sat_id] = SatelliteState(
#                 sat_id=sat_id,
#                 sat_type=SatelliteType.ENEMY_ATTACKER,
#                 position=position,
#                 velocity=velocity,
#                 fuel_remaining=1.0,
#                 health_status=1.0,
#                 capabilities={
#                     'attack_range': 600.0,  # km
#                     'maneuver_capability': 0.9,
#                     'stealth_capability': 0.7
#                 }
#             )
#             sat_id += 1
    
#     def _generate_geo_position(self, angle: float, offset_radius: float = 0.0) -> np.ndarray:
#         """在GEO轨道附近生成位置"""
#         r = self.geo_altitude + offset_radius
#         x = r * np.cos(angle)
#         y = r * np.sin(angle)
#         z = np.random.uniform(-100, 100)  # 轻微的轨道倾斜
#         return np.array([x, y, z])
    
#     def step(self, actions: Dict[str, np.ndarray]) -> Tuple[Dict, Dict, Dict, Dict]:
#         """执行一步仿真"""
#         # 解析和执行动作
#         parsed_actions = self._parse_actions(actions)
#         self._execute_actions(parsed_actions)
        
#         # 更新物理状态
#         self._update_dynamics()
        
#         # 计算奖励
#         rewards = self._compute_rewards()
        
#         # 检查终止条件
#         dones = self._check_termination()
        
#         # 获取新观测
#         observations = self._get_observations()
        
#         # 更新环境状态
#         self.current_step += 1
#         self.simulation_time += self.dt
        
#         # 更新累积奖励
#         for agent_id, reward in rewards.items():
#             self.episode_rewards[agent_id] += reward
        
#         # 构建信息字典
#         infos = self._get_infos()
        
#         return observations, rewards, dones, infos
    
#     def _parse_actions(self, actions: Dict[str, np.ndarray]) -> Dict[int, Dict]:
#         """解析智能体动作"""
#         parsed_actions = {}
        
#         for agent_str, action in actions.items():
#             agent_id = int(agent_str.split('_')[1])
#             if agent_id < len(self.satellites):
#                 satellite = self.satellites[agent_id]
#                 parsed_action = self.action_parser.parse_action(
#                     action, satellite.sat_type
#                 )
#                 parsed_actions[agent_id] = parsed_action
        
#         return parsed_actions
    
#     def _execute_actions(self, actions: Dict[int, Dict]):
#         """执行解析后的动作"""
#         for sat_id, action in actions.items():
#             if sat_id in self.satellites:
#                 satellite = self.satellites[sat_id]
                
#                 # 执行机动指令
#                 if 'maneuver' in action:
#                     self.orbital_dynamics.apply_maneuver(
#                         satellite, action['maneuver']
#                     )
                
#                 # 执行观测指令
#                 if 'observation' in action:
#                     self.sensor_network.update_observation_task(
#                         sat_id, action['observation']
#                     )
                
#                 # 记录动作
#                 satellite.last_action = action
    
#     def _update_dynamics(self):
#         """更新轨道动力学"""
#         for satellite in self.satellites.values():
#             # 更新轨道位置和速度
#             new_position, new_velocity = self.orbital_dynamics.propagate(
#                 satellite.position, satellite.velocity, self.dt
#             )
#             satellite.position = new_position
#             satellite.velocity = new_velocity
            
#             # 更新资源消耗
#             if satellite.last_action:
#                 fuel_consumption = self._calculate_fuel_consumption(
#                     satellite, satellite.last_action
#                 )
#                 satellite.fuel_remaining = max(0.0, 
#                     satellite.fuel_remaining - fuel_consumption)
    
#     def _calculate_fuel_consumption(self, satellite: SatelliteState, 
#                                   action: Dict) -> float:
#         """计算燃料消耗"""
#         consumption = 0.0
        
#         if 'maneuver' in action:
#             # 基于机动强度计算燃料消耗
#             maneuver_intensity = np.linalg.norm(action['maneuver'].get('delta_v', [0, 0, 0]))
#             consumption += maneuver_intensity * 0.001  # 简化的燃料消耗模型
        
#         return consumption
    
#     def _compute_rewards(self) -> Dict[str, float]:
#         """计算奖励函数"""
#         rewards = {}
        
#         for i in range(self.n_agents):
#             agent_id = f'agent_{i}'
#             satellite = self.satellites[i]
            
#             reward = 0.0
            
#             # 基础存活奖励
#             if satellite.health_status > 0:
#                 reward += 0.1
            
#             # 任务完成奖励
#             if satellite.sat_type == SatelliteType.OBSERVER:
#                 reward += self._compute_observation_reward(satellite)
#             elif satellite.sat_type == SatelliteType.DEFENDER:
#                 reward += self._compute_defense_reward(satellite)
#             elif satellite.sat_type == SatelliteType.HIGH_VALUE:
#                 reward += self._compute_survival_reward(satellite)
            
#             # 协作奖励
#             reward += self._compute_cooperation_reward(satellite)
            
#             # 燃料效率奖励
#             reward += satellite.fuel_remaining * 0.05
            
#             rewards[agent_id] = reward
        
#         return rewards
    
#     def _compute_observation_reward(self, satellite: SatelliteState) -> float:
#         """计算观测任务奖励"""
#         reward = 0.0
        
#         # 检查是否在观测敌方卫星
#         for enemy_id, enemy_sat in self.satellites.items():
#             if enemy_sat.sat_type == SatelliteType.ENEMY_ATTACKER:
#                 distance = np.linalg.norm(satellite.position - enemy_sat.position)
#                 observation_range = satellite.capabilities.get('observation_range', 1000.0)
                
#                 if distance <= observation_range:
#                     reward += 1.0  # 成功观测奖励
#                     # 距离越近奖励越高
#                     reward += max(0, (observation_range - distance) / observation_range)
        
#         return reward
    
#     def _compute_defense_reward(self, satellite: SatelliteState) -> float:
#         """计算防御任务奖励"""
#         reward = 0.0
        
#         # 检查是否在保护高价值卫星
#         for target_id, target_sat in self.satellites.items():
#             if target_sat.sat_type == SatelliteType.HIGH_VALUE:
#                 # 计算防卫卫星与高价值卫星的距离
#                 distance_to_target = np.linalg.norm(
#                     satellite.position - target_sat.position
#                 )
                
#                 # 检查是否拦截了敌方威胁
#                 for enemy_id, enemy_sat in self.satellites.items():
#                     if enemy_sat.sat_type == SatelliteType.ENEMY_ATTACKER:
#                         distance_to_enemy = np.linalg.norm(
#                             satellite.position - enemy_sat.position
#                         )
#                         distance_enemy_to_target = np.linalg.norm(
#                             enemy_sat.position - target_sat.position
#                         )
                        
#                         # 如果防卫卫星在敌方和目标之间
#                         if (distance_to_enemy < 300.0 and 
#                             distance_to_target < distance_enemy_to_target):
#                             reward += 2.0  # 成功拦截位置奖励
        
#         return reward
    
#     def _compute_survival_reward(self, satellite: SatelliteState) -> float:
#         """计算高价值卫星生存奖励"""
#         reward = 0.0
        
#         # 基础生存奖励
#         reward += 1.0
        
#         # 威胁规避奖励
#         min_threat_distance = float('inf')
#         for enemy_id, enemy_sat in self.satellites.items():
#             if enemy_sat.sat_type == SatelliteType.ENEMY_ATTACKER:
#                 distance = np.linalg.norm(satellite.position - enemy_sat.position)
#                 min_threat_distance = min(min_threat_distance, distance)
        
#         if min_threat_distance < float('inf'):
#             # 距离威胁越远奖励越高
#             safe_distance = 800.0  # 安全距离阈值
#             if min_threat_distance > safe_distance:
#                 reward += 0.5
#             else:
#                 reward += min_threat_distance / safe_distance * 0.5
        
#         return reward
    
#     def _compute_cooperation_reward(self, satellite: SatelliteState) -> float:
#         """计算协作奖励"""
#         # 简化的协作奖励 - 基于编队保持
#         reward = 0.0
        
#         # 检查与友方卫星的协调性
#         friendly_count = 0
#         total_distance = 0.0
        
#         for other_id, other_sat in self.satellites.items():
#             if (other_sat.sat_type != SatelliteType.ENEMY_ATTACKER and 
#                 other_sat.sat_id != satellite.sat_id):
#                 distance = np.linalg.norm(satellite.position - other_sat.position)
#                 total_distance += distance
#                 friendly_count += 1
        
#         if friendly_count > 0:
#             avg_distance = total_distance / friendly_count
#             # 适中的编队距离获得奖励
#             optimal_distance = 500.0
#             distance_penalty = abs(avg_distance - optimal_distance) / optimal_distance
#             reward += max(0, 0.2 * (1.0 - distance_penalty))
        
#         return reward
    
#     def _check_termination(self) -> Dict[str, bool]:
#         """检查终止条件"""
#         dones = {}
        
#         # 检查每个智能体的终止条件
#         for i in range(self.n_agents):
#             agent_id = f'agent_{i}'
#             satellite = self.satellites[i]
            
#             done = False
            
#             # 卫星损坏或燃料耗尽
#             if satellite.health_status <= 0 or satellite.fuel_remaining <= 0:
#                 done = True
            
#             dones[agent_id] = done
        
#         # 全局终止条件
#         global_done = False
        
#         # 达到最大步数
#         if self.current_step >= self.max_steps:
#             global_done = True
        
#         # 任务完成条件 - 高价值卫星被摧毁
#         high_value_destroyed = all(
#             sat.health_status <= 0 
#             for sat in self.satellites.values() 
#             if sat.sat_type == SatelliteType.HIGH_VALUE
#         )
#         if high_value_destroyed:
#             global_done = True
        
#         # 敌方卫星被消除
#         enemies_eliminated = all(
#             sat.health_status <= 0 
#             for sat in self.satellites.values() 
#             if sat.sat_type == SatelliteType.ENEMY_ATTACKER
#         )
#         if enemies_eliminated:
#             global_done = True
        
#         # 如果全局终止，所有智能体都终止
#         if global_done:
#             for agent_id in dones:
#                 dones[agent_id] = True
        
#         return dones
    
#     def _get_observations(self) -> Dict[str, np.ndarray]:
#         """获取环境观测"""
#         observations = {}
        
#         for i in range(self.n_agents):
#             agent_id = f'agent_{i}'
#             obs = self._build_observation_vector()
#             observations[agent_id] = obs
        
#         return observations
    
#     def _build_observation_vector(self) -> np.ndarray:
#         """构建观测向量"""
#         obs_list = []
        
#         # 所有卫星状态
#         for sat_id in sorted(self.satellites.keys()):
#             satellite = self.satellites[sat_id]
            
#             # 位置和速度（归一化）
#             pos_norm = satellite.position / 50000.0  # 归一化到合理范围
#             vel_norm = satellite.velocity / 10.0
            
#             sat_obs = np.concatenate([
#                 pos_norm,
#                 vel_norm,
#                 [satellite.fuel_remaining],
#                 [satellite.health_status],
#                 [satellite.capabilities.get('observation_range', 0) / 1000.0,
#                  satellite.capabilities.get('maneuver_capability', 0),
#                  satellite.capabilities.get('attack_range', 0) / 1000.0]
#             ])
#             obs_list.extend(sat_obs)
        
#         # 威胁评估
#         threat_assessment = self._compute_threat_assessment()
#         obs_list.extend(threat_assessment)
        
#         # 任务状态
#         mission_status = self._compute_mission_status()
#         obs_list.extend(mission_status)
        
#         # 时间信息
#         time_info = [
#             self.current_step / self.max_steps,  # 归一化时间进度
#             self.simulation_time / (self.max_steps * self.dt)  # 归一化仿真时间
#         ]
#         obs_list.extend(time_info)
        
#         return np.array(obs_list, dtype=np.float32)
    
#     def _compute_threat_assessment(self) -> List[float]:
#         """计算威胁评估"""
#         threat_level = 0.0
#         min_threat_distance = float('inf')
#         threat_approach_speed = 0.0
        
#         high_value_sats = [s for s in self.satellites.values() 
#                           if s.sat_type == SatelliteType.HIGH_VALUE]
#         enemy_sats = [s for s in self.satellites.values() 
#                      if s.sat_type == SatelliteType.ENEMY_ATTACKER]
        
#         if high_value_sats and enemy_sats:
#             for hv_sat in high_value_sats:
#                 for enemy_sat in enemy_sats:
#                     distance = np.linalg.norm(hv_sat.position - enemy_sat.position)
#                     relative_vel = np.linalg.norm(
#                         enemy_sat.velocity - hv_sat.velocity
#                     )
                    
#                     if distance < min_threat_distance:
#                         min_threat_distance = distance
#                         threat_approach_speed = relative_vel
            
#             # 威胁等级基于距离和相对速度
#             if min_threat_distance < 1000.0:
#                 threat_level = 1.0 - (min_threat_distance / 1000.0)
        
#         return [
#             threat_level,
#             min_threat_distance / 2000.0,  # 归一化距离
#             threat_approach_speed / 10.0,  # 归一化速度
#             len([s for s in enemy_sats if s.health_status > 0]) / self.n_enemies,  # 存活敌方比例
#             len([s for s in high_value_sats if s.health_status > 0]) / self.n_high_value  # 存活高价值目标比例
#         ]
    
#     def _compute_mission_status(self) -> List[float]:
#         """计算任务状态"""
#         observation_coverage = 0.0
#         defense_effectiveness = 0.0
#         formation_integrity = 0.0
        
#         # 观测覆盖率
#         observer_sats = [s for s in self.satellites.values() 
#                         if s.sat_type == SatelliteType.OBSERVER and s.health_status > 0]
#         if observer_sats:
#             coverage_count = 0
#             for obs_sat in observer_sats:
#                 for enemy_sat in self.satellites.values():
#                     if enemy_sat.sat_type == SatelliteType.ENEMY_ATTACKER:
#                         distance = np.linalg.norm(obs_sat.position - enemy_sat.position)
#                         obs_range = obs_sat.capabilities.get('observation_range', 1000.0)
#                         if distance <= obs_range:
#                             coverage_count += 1
#                             break
#             observation_coverage = coverage_count / len(observer_sats)
        
#         # 防御有效性
#         defender_sats = [s for s in self.satellites.values() 
#                         if s.sat_type == SatelliteType.DEFENDER and s.health_status > 0]
#         if defender_sats:
#             defense_effectiveness = len(defender_sats) / self.n_defenders
        
#         # 编队完整性
#         alive_agents = len([s for s in self.satellites.values() 
#                            if s.sat_type != SatelliteType.ENEMY_ATTACKER and s.health_status > 0])
#         formation_integrity = alive_agents / self.n_agents
        
#         return [observation_coverage, defense_effectiveness, formation_integrity]
    
#     def _get_infos(self) -> Dict[str, Dict]:
#         """获取额外信息"""
#         infos = {}
        
#         for i in range(self.n_agents):
#             agent_id = f'agent_{i}'
#             satellite = self.satellites[i]
            
#             infos[agent_id] = {
#                 'satellite_id': satellite.sat_id,
#                 'satellite_type': satellite.sat_type.value,
#                 'fuel_remaining': satellite.fuel_remaining,
#                 'health_status': satellite.health_status,
#                 'position': satellite.position.tolist(),
#                 'velocity': satellite.velocity.tolist(),
#                 'episode_reward': self.episode_rewards[agent_id],
#                 'simulation_time': self.simulation_time
#             }
        
#         return infos
    
#     def render(self, mode='human'):
#         """渲染环境（可选实现）"""
#         if mode == 'human':
#             print(f"Step: {self.current_step}, Time: {self.simulation_time:.1f}s")
#             for sat_id, satellite in self.satellites.items():
#                 pos = satellite.position
#                 print(f"  Satellite {sat_id} ({satellite.sat_type.value}): "
#                       f"Pos=({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}), "
#                       f"Fuel={satellite.fuel_remaining:.2f}, "
#                       f"Health={satellite.health_status:.2f}")
    
#     def close(self):
#         """关闭环境"""
#         self.logger.info("Environment closed")

"""
太空多智能体强化学习环境主类 (重构优化版)
实现基于GEO轨道的卫星编队作战环境
- 优化状态空间为相对坐标和字典结构
- 优化动作空间为参数化混合动作空间
"""

import numpy as np
import gym
from gym import spaces
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass, field
from enum import Enum
import json
# 假设这些子模块也已存在
# from .orbital_dynamics import OrbitalDynamics
# from .sensor_models import SensorNetwork
# from .action_parser import ActionParser
# --- 为了独立运行，创建一些虚拟类 ---
class OrbitalDynamics:
    def __init__(self, alt): pass
    def compute_orbital_velocity(self, pos): return np.zeros(3)
    def apply_maneuver(self, sat, maneuver): pass
    def propagate(self, pos, vel, dt): return pos + vel * dt, vel
class SensorNetwork:
    def __init__(self, config): pass
    def update_observation_task(self, sat_id, obs_task): pass
class ActionParser: # 这个类的功能现在部分被环境自己处理了
    def __init__(self, config): pass
    def parse_action(self, action, sat_type): return {} # 旧的解析方式，新版不再这样使用
# --- 虚拟类结束 ---


class SatelliteType(Enum):
    """卫星类型枚举"""
    OBSERVER = "observer"
    DEFENDER = "defender"
    HIGH_VALUE = "high_value"
    ENEMY_ATTACKER = "enemy_attacker"


## --- REFACTORED ---
## 使用 dataclass 统一管理常量，避免魔法数字
@dataclass
class SpaceEnvConstants:
    """环境常量配置"""
    # 物理常量
    GEO_ALTITUDE: float = 35786.0
    # 归一化常量
    MAX_POS: float = GEO_ALTITUDE + 5000.0  # 最大位置范围 (km)
    MAX_VEL: float = 10.0                   # 最大速度范围 (km/s)
    MAX_RANGE: float = 2000.0               # 最大传感器/武器范围 (km)
    # 动作空间参数
    MAX_DELTA_V: float = 0.1                # 最大速度增量 (km/s)
    # 奖励设计
    REWARD_SURVIVAL: float = 0.01
    REWARD_OBS_SUCCESS: float = 1.0
    REWARD_INTERCEPT_POS: float = 2.0
    REWARD_HVA_SURVIVAL: float = 1.0
    PENALTY_FUEL: float = -0.1
    PENALTY_DONE: float = -10.0
    

@dataclass
class SatelliteState:
    """卫星状态数据结构"""
    sat_id: int
    sat_type: SatelliteType
    position: np.ndarray  # [x, y, z] in km
    velocity: np.ndarray  # [vx, vy, vz] in km/s
    fuel_remaining: float = 1.0  # 0-1 normalized
    health_status: float = 1.0   # 0-1 normalized
    is_alive: bool = True
    capabilities: Dict[str, float] = field(default_factory=dict)
    last_action: Optional[Dict] = None


class SpaceEnvironmentV2(gym.Env):
    """太空多智能体强化学习环境 (优化版)"""
    
    def __init__(self, config: Dict[str, Any], initial_conditions_path: str):
        super().__init__()
        
        self.config = config
        self.const = SpaceEnvConstants()
        self.max_steps = config.get('max_steps', 1000)
        self.dt = config.get('time_step', 60.0)
        
        # --- MODIFIED: 加载配置文件 ---
        try:
            with open(initial_conditions_path, 'r') as f:
                self.initial_conditions = json.load(f)
                self.logger.info(f"Successfully loaded scenario: {self.initial_conditions.get('scenario_name', 'Unnamed')}")
        except FileNotFoundError:
            self.logger.error(f"Initialization file not found at: {initial_conditions_path}")
            raise
        except json.JSONDecodeError:
            self.logger.error(f"Error decoding JSON from: {initial_conditions_path}")
            raise
            
        # 从配置文件动态确定卫星数量
        self.all_sat_configs = self.initial_conditions['satellites']
        self.friendly_sats_configs = [s for s in self.all_sat_configs if s['sat_type'] != 'enemy_attacker']
        self.enemy_sats_configs = [s for s in self.all_sat_configs if s['sat_type'] == 'enemy_attacker']
        
        self.friendly_agent_ids = [s['sat_id'] for s in self.friendly_sats_configs]
        self.n_agents = len(self.friendly_agent_ids)
        self.n_enemies = len(self.enemy_sats_configs)
        self.total_sats = self.n_agents + self.n_enemies

        self.orbital_dynamics = OrbitalDynamics(self.const.GEO_ALTITUDE)
        self.sensor_network = SensorNetwork(config.get('sensor_config', {}))
        
        self.satellites: Dict[int, SatelliteState] = {}
        self.current_step = 0
        self.simulation_time = 0.0
        
        self._setup_spaces() # setup_spaces现在可以动态使用n_agents等参数
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        
    def _setup_spaces(self):
        """
        ## --- REFACTORED: 设置优化的观测和动作空间 ---
        使用字典结构 (Dict) 和相对坐标来定义观测空间。
        使用参数化的混合动作空间 (Dict) 来定义动作。
        """
        # --- 1. 优化的观测空间 (Observation Space) ---
        # 每个智能体接收一个字典，包含自身状态、友方相对状态、敌方相对状态和任务信息
        self_obs_dim = 6 # pos(3), vel(3) - 归一化
        partner_obs_dim = 7 # relative_pos(3), relative_vel(3), type(1) - 归一化
        enemy_obs_dim = 7   # 同上
        mission_info_dim = 3 # time_progress, hva_health, num_enemies_alive
        
        self.observation_space = spaces.Dict({
            f'agent_{i}': spaces.Dict({
                "self_state": spaces.Box(low=-1, high=1, shape=(self_obs_dim,), dtype=np.float32),
                # 使用固定长度的列表，用0填充死掉或不存在的卫星
                "partner_states": spaces.Box(low=-1, high=1, shape=(self.n_agents - 1, partner_obs_dim), dtype=np.float32),
                "enemy_states": spaces.Box(low=-1, high=1, shape=(self.n_enemies, enemy_obs_dim), dtype=np.float32),
                "mission_info": spaces.Box(low=0, high=1, shape=(mission_info_dim,), dtype=np.float32)
            }) for i in self.friendly_agent_ids
        })

        # --- 2. 优化的动作空间 (Action Space) ---
        # 参数化动作空间: agent先选择一个离散动作类型，再为该动作选择目标和连续参数
        self.action_space = spaces.Dict({
            f'agent_{i}': spaces.Dict({
                # 动作类型: 0=待命, 1=机动, 2=观测/攻击
                "action_type": spaces.Discrete(3),
                # 目标ID: 0..n_agents-1为友方, n_agents..total_sats-1为敌方
                "target_id": spaces.Discrete(self.total_sats),
                # 动作参数: [dx, dy, dz] 速度增量
                "parameters": spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
            }) for i in self.friendly_agent_ids
        })

    def reset(self, seed: Optional[int] = None) -> Dict[str, np.ndarray]:
        if seed is not None: np.random.seed(seed)
        self.current_step = 0
        self.simulation_time = 0.0
        self._initialize_satellites()
        self.logger.info(f"Environment reset with {len(self.satellites)} satellites.")
        return self._get_observations()

    def step(self, actions: Dict[str, Dict]) -> Tuple[Dict, Dict, Dict, Dict]:
        # 解析和执行动作
        self._execute_actions(actions)
        # 更新物理状态
        self._update_dynamics()
        # 计算奖励
        rewards = self._compute_rewards()
        # 检查终止条件
        dones = self._check_termination()
        # 获取新观测
        observations = self._get_observations()
        # 更新环境状态
        self.current_step += 1
        self.simulation_time += self.dt
        infos = self._get_infos(actions)
        return observations, rewards, dones, infos

    ## --- REFACTORED: 动作执行逻辑 ---
    def _execute_actions(self, actions: Dict[str, Dict]):
        """根据新的字典式动作执行指令"""
        for agent_str, action_dict in actions.items():
            agent_id = int(agent_str.split('_')[1])
            satellite = self.satellites.get(agent_id)
            if not satellite or not satellite.is_alive: continue
            
            satellite.last_action = action_dict
            action_type = action_dict['action_type']
            
            # 1: 机动
            if action_type == 1:
                # 参数是归一化[-1, 1]的速度增量, 需反归一化
                delta_v_normalized = action_dict['parameters']
                delta_v = delta_v_normalized * self.const.MAX_DELTA_V
                self.orbital_dynamics.apply_maneuver(satellite, {'delta_v': delta_v})
                
            # 2: 观测/攻击
            elif action_type == 2:
                target_id = action_dict['target_id']
                if satellite.sat_type == SatelliteType.OBSERVER and self.satellites.get(target_id):
                    self.sensor_network.update_observation_task(agent_id, {'target_id': target_id})
                # 此处可以添加攻击逻辑

    ## --- REFACTORED: 观测构建逻辑 ---
    def _get_observations(self) -> Dict[str, Dict]:
        """为每个智能体构建其独立的、以自我为中心的观测字典"""
        all_obs = {}
        # 缓存所有卫星的当前状态，避免重复计算
        alive_friendly = {i: self.satellites[i] for i in self.friendly_agent_ids if self.satellites[i].is_alive}
        alive_enemies = {i: sat for i, sat in self.satellites.items() if sat.sat_type == SatelliteType.ENEMY_ATTACKER and sat.is_alive}
        
        for agent_id in self.friendly_agent_ids:
            if self.satellites[agent_id].is_alive:
                all_obs[f'agent_{agent_id}'] = self._build_observation_for_agent(
                    agent_id, alive_friendly, alive_enemies
                )
            else: # 如果智能体死亡，提供一个零向量观测
                all_obs[f'agent_{agent_id}'] = {k: np.zeros(v.shape) for k, v in self.observation_space[f'agent_{agent_id}'].spaces.items()}
        return all_obs

    def _build_observation_for_agent(self, agent_id: int, alive_friendly: Dict, alive_enemies: Dict) -> Dict:
        """构建单个智能体的观测字典"""
        agent_sat = self.satellites[agent_id]
        
        # 1. 自身状态
        self_state = np.concatenate([
            agent_sat.position / self.const.MAX_POS,
            agent_sat.velocity / self.const.MAX_VEL,
            # [agent_sat.fuel_remaining], # 这些可以放入info中，不作为决策关键输入
            # [agent_sat.health_status],
        ])
        
        # 2. 友方状态 (相对)
        partner_states = np.zeros((self.n_agents - 1, 7))
        partner_idx = 0
        for other_id, other_sat in alive_friendly.items():
            if agent_id == other_id: continue
            relative_pos = (other_sat.position - agent_sat.position) / self.const.MAX_POS
            relative_vel = (other_sat.velocity - agent_sat.velocity) / self.const.MAX_VEL
            type_one_hot = 1.0 if other_sat.sat_type == SatelliteType.DEFENDER else 0.5
            partner_states[partner_idx] = np.concatenate([relative_pos, relative_vel, [type_one_hot]])
            partner_idx += 1
            
        # 3. 敌方状态 (相对)
        enemy_states = np.zeros((self.n_enemies, 7))
        enemy_idx = 0
        for other_id, other_sat in alive_enemies.items():
            relative_pos = (other_sat.position - agent_sat.position) / self.const.MAX_POS
            relative_vel = (other_sat.velocity - agent_sat.velocity) / self.const.MAX_VEL
            enemy_states[enemy_idx] = np.concatenate([relative_pos, relative_vel, [1.0]]) # type=enemy
            enemy_idx += 1
            
        # 4. 任务信息
        hva_health = 0.0
        for sat in self.satellites.values():
            if sat.sat_type == SatelliteType.HIGH_VALUE:
                hva_health = sat.health_status
                break
        mission_info = np.array([
            self.current_step / self.max_steps,
            hva_health,
            len(alive_enemies) / self.n_enemies
        ])
        
        return {
            "self_state": self_state.astype(np.float32),
            "partner_states": partner_states.astype(np.float32),
            "enemy_states": enemy_states.astype(np.float32),
            "mission_info": mission_info.astype(np.float32),
        }

    ## --- 以下为环境的其他方法，为保持完整性而保留，可根据新结构微调 ---
    
    def _initialize_satellites(self):
        """从加载的 initial_conditions 配置中初始化所有卫星的状态"""
        self.satellites.clear()
        
        for sat_config in self.all_sat_configs:
            sat_id = sat_config['sat_id']
            sat_type = SatelliteType(sat_config['sat_type']) # 从字符串转换为枚举
            
            # --- MODIFIED: 直接读取笛卡尔坐标 ---
            # 不再需要进行极坐标到笛卡尔坐标的转换
            position = np.array(sat_config['position'])
            
            # 创建 SatelliteState 对象
            self.satellites[sat_id] = SatelliteState(
                sat_id=sat_id,
                sat_type=sat_type,
                position=position,
                velocity=np.zeros(3), # 速度先初始化为0，下面再计算
                fuel_remaining=sat_config.get('fuel_remaining', 1.0),
                health_status=sat_config.get('health_status', 1.0),
                is_alive=True,
                capabilities=sat_config.get('capabilities', {})
            )
            
        # 为所有卫星计算初始轨道速度
        for satellite in self.satellites.values():
            satellite.velocity = self.orbital_dynamics.compute_orbital_velocity(satellite.position)


    def _generate_geo_position(self, angle: float, offset_radius: float = 0.0) -> np.ndarray:
        r = self.const.GEO_ALTITUDE + offset_radius
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        z = np.random.uniform(-100, 100)
        return np.array([x, y, z])

    def _update_dynamics(self):
        """更新轨道动力学 (逻辑基本不变)"""
        for satellite in self.satellites.values():
            if not satellite.is_alive: continue
            new_pos, new_vel = self.orbital_dynamics.propagate(satellite.position, satellite.velocity, self.dt)
            satellite.position = new_pos
            satellite.velocity = new_vel
            if satellite.last_action:
                fuel_consumption = self._calculate_fuel_consumption(satellite.last_action)
                satellite.fuel_remaining = max(0.0, satellite.fuel_remaining - fuel_consumption)

    def _calculate_fuel_consumption(self, action: Dict) -> float:
        if action.get('action_type') == 1:
            delta_v_norm = np.linalg.norm(action['parameters'])
            return delta_v_norm * 0.005 # 简化的燃料消耗模型
        return 0.0

    def _compute_rewards(self) -> Dict[str, float]:
        """计算奖励函数 (可基于新结构进一步优化)"""
        # ... 奖励函数逻辑与原版类似，但可以利用新结构做得更精细 ...
        # 例如，协作奖励可以基于智能体是否对同一目标采取了互补行动
        rewards = {f'agent_{i}': 0.0 for i in self.friendly_agent_ids}
        # 简单示例
        for i in self.friendly_agent_ids:
            if self.satellites[i].is_alive:
                rewards[f'agent_{i}'] += self.const.REWARD_SURVIVAL
                rewards[f'agent_{i}'] += self.satellites[i].fuel_remaining * self.const.PENALTY_FUEL
        return rewards

    def _check_termination(self) -> Dict[str, bool]:
        """检查终止条件 (逻辑基本不变，但更新卫星状态)"""
        dones = {f'agent_{i}': False for i in self.friendly_agent_ids}
        
        for i in self.friendly_agent_ids:
            sat = self.satellites[i]
            if sat.is_alive and (sat.health_status <= 0 or sat.fuel_remaining <= 0):
                sat.is_alive = False
                dones[f'agent_{i}'] = True
        
        hva = next((s for s in self.satellites.values() if s.sat_type == SatelliteType.HIGH_VALUE), None)
        global_done = (
            self.current_step >= self.max_steps or 
            (hva and not hva.is_alive) or
            all(not s.is_alive for s in self.satellites.values() if s.sat_type == SatelliteType.ENEMY_ATTACKER)
        )
        
        if global_done:
            for i in self.friendly_agent_ids:
                if not dones[f'agent_{i}']: # 如果智能体还没因为自身原因结束
                    dones[f'agent_{i}'] = True
        
        # __all__ key for RLlib
        dones["__all__"] = global_done
        return dones

    def _get_infos(self, actions: Dict) -> Dict:
        """获取额外信息 (逻辑基本不变)"""
        infos = {}
        for i in self.friendly_agent_ids:
            infos[f'agent_{i}'] = {
                'is_alive': self.satellites[i].is_alive,
                'fuel': self.satellites[i].fuel_remaining,
                'last_action': actions.get(f'agent_{i}', {})
            }
        return infos

    def render(self, mode='human'):
        """渲染环境（可选实现）"""
        if mode == 'human':
            print(f"Step: {self.current_step}, Time: {self.simulation_time:.1f}s")
            for sat_id, satellite in self.satellites.items():
                pos = satellite.position
                print(f"  Satellite {sat_id} ({satellite.sat_type.value}): "
                      f"Pos=({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}), "
                      f"Fuel={satellite.fuel_remaining:.2f}, "
                      f"Health={satellite.health_status:.2f}")
    
    def close(self):
        self.logger.info("Environment V2 closed")



