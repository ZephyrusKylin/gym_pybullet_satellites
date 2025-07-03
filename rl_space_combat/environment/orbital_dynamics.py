"""
轨道动力学计算模块 - 精简版
仅实现强化学习环境所需的核心功能
"""

import numpy as np
from typing import Tuple, Dict
import logging


class OrbitalDynamics:
    """轨道动力学计算引擎 - 精简版"""
    
    # 物理常数
    MU_EARTH = 398600.4418  # 地球引力参数 (km³/s²)
    EARTH_RADIUS = 6371.0   # 地球半径 (km)
    
    def __init__(self, reference_altitude: float = 35786.0):
        """
        初始化轨道动力学引擎
        
        Args:
            reference_altitude: 参考轨道高度 (km)，默认为GEO轨道
        """
        self.reference_altitude = reference_altitude
        self.reference_radius = self.EARTH_RADIUS + reference_altitude
        self.logger = logging.getLogger(__name__)
        
    def compute_orbital_velocity(self, position: np.ndarray) -> np.ndarray:
        """
        计算给定位置的轨道速度（圆轨道近似）
        
        Args:
            position: 位置向量 [x, y, z] (km)
            
        Returns:
            velocity: 速度向量 [vx, vy, vz] (km/s)
        """
        r = np.linalg.norm(position)
        
        if r < self.EARTH_RADIUS:
            self.logger.warning(f"Position below Earth surface: r={r:.1f}km")
            r = self.EARTH_RADIUS + 100.0  # 最小高度100km
        
        # 圆轨道速度
        v_magnitude = np.sqrt(self.MU_EARTH / r)
        
        # 速度方向：垂直于位置向量，在xy平面内
        pos_xy = position[:2]
        r_xy = np.linalg.norm(pos_xy)
        
        if r_xy > 1e-6:  # 避免除零
            # 切向单位向量
            tangent_xy = np.array([-pos_xy[1], pos_xy[0]]) / r_xy
            velocity = np.zeros(3)
            velocity[:2] = tangent_xy * v_magnitude
            velocity[2] = 0.0  # 赤道轨道近似
        else:
            # 如果在z轴上，设置默认切向速度
            velocity = np.array([v_magnitude, 0.0, 0.0])
        
        return velocity
    
    def propagate(self, position: np.ndarray, velocity: np.ndarray, 
                  dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        轨道传播计算 - 使用简化的二体问题
        
        Args:
            position: 当前位置 [x, y, z] (km)
            velocity: 当前速度 [vx, vy, vz] (km/s)
            dt: 时间步长 (s)
            
        Returns:
            new_position: 新位置 [x, y, z] (km)
            new_velocity: 新速度 [vx, vy, vz] (km/s)
        """
        # 使用简化的欧拉方法进行快速计算
        # 对于强化学习环境，精度要求不高，速度更重要
        
        r = np.linalg.norm(position)
        
        # 防止除零和低轨道
        if r < self.EARTH_RADIUS:
            r = self.EARTH_RADIUS
            position = position / np.linalg.norm(position) * self.EARTH_RADIUS
        
        # 引力加速度
        a_gravity = -self.MU_EARTH * position / r**3
        
        # 欧拉积分
        new_velocity = velocity + a_gravity * dt
        new_position = position + new_velocity * dt
        
        return new_position, new_velocity
    
    def apply_maneuver(self, satellite, maneuver_command: Dict):
        """
        应用机动指令到卫星
        
        Args:
            satellite: 卫星状态对象
            maneuver_command: 机动指令字典，包含'delta_v'键
        """
        if not maneuver_command or 'delta_v' not in maneuver_command:
            return
        
        delta_v_command = np.array(maneuver_command['delta_v'])
        
        # 将归一化的动作转换为实际的delta_v
        max_delta_v = satellite.capabilities.get('maneuver_capability', 0.5) * 0.1  # km/s
        actual_delta_v = delta_v_command * max_delta_v
        
        # 检查燃料限制
        fuel_required = np.linalg.norm(actual_delta_v) * 0.01  # 简化燃料消耗模型
        if satellite.fuel_remaining < fuel_required:
            # 燃料不足，按比例缩减机动
            fuel_ratio = satellite.fuel_remaining / fuel_required
            actual_delta_v *= fuel_ratio
            self.logger.debug(f"Fuel limited maneuver for satellite {satellite.sat_id}")
        
        # 应用速度变化
        satellite.velocity += actual_delta_v
        
        self.logger.debug(f"Applied maneuver to satellite {satellite.sat_id}: "
                         f"delta_v={actual_delta_v}")