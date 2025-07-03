"""
轨道动力学计算模块
实现卫星轨道传播、机动计算等功能
"""

import numpy as np
from typing import Tuple, Dict, Optional
from dataclasses import dataclass
import logging


@dataclass
class OrbitalElements:
    """轨道要素"""
    semi_major_axis: float      # 半长轴 (km)
    eccentricity: float         # 偏心率
    inclination: float          # 轨道倾角 (rad)
    raan: float                 # 升交点赤经 (rad)
    argument_of_perigee: float  # 近地点幅角 (rad)
    true_anomaly: float         # 真近点角 (rad)


@dataclass
class ManeuverCommand:
    """机动指令"""
    delta_v: np.ndarray         # 速度变化 [km/s]
    burn_time: float            # 推进时间 (s)
    direction: str              # 机动方向 ('prograde', 'retrograde', 'radial', 'normal')
    target_position: Optional[np.ndarray] = None  # 目标位置 [km]
    target_orbit: Optional[OrbitalElements] = None  # 目标轨道


class OrbitalDynamics:
    """轨道动力学计算引擎"""
    
    # 物理常数
    MU_EARTH = 398600.4418  # 地球引力参数 (km³/s²)
    EARTH_RADIUS = 6371.0   # 地球半径 (km)
    J2 = 1.08262668e-3      # J2摄动项系数
    
    def __init__(self, reference_altitude: float = 35786.0):
        """
        初始化轨道动力学引擎
        
        Args:
            reference_altitude: 参考轨道高度 (km)，默认为GEO轨道
        """
        self.reference_altitude = reference_altitude
        self.reference_radius = self.EARTH_RADIUS + reference_altitude
        self.reference_angular_velocity = np.sqrt(self.MU_EARTH / self.reference_radius**3)
        
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
        
        # 速度方向：垂直于位置向量，假设在赤道面内
        # 简化处理：速度在xy平面内，垂直于径向
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
        轨道传播计算
        
        Args:
            position: 当前位置 [x, y, z] (km)
            velocity: 当前速度 [vx, vy, vz] (km/s)
            dt: 时间步长 (s)
            
        Returns:
            new_position: 新位置 [x, y, z] (km)
            new_velocity: 新速度 [vx, vy, vz] (km/s)
        """
        # 使用Runge-Kutta 4阶方法进行轨道传播
        return self._rk4_propagate(position, velocity, dt)
    
    def _rk4_propagate(self, r0: np.ndarray, v0: np.ndarray, 
                       dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用RK4方法进行轨道传播
        """
        # 状态向量 [x, y, z, vx, vy, vz]
        state0 = np.concatenate([r0, v0])
        
        # RK4积分
        k1 = self._orbital_derivatives(state0) * dt
        k2 = self._orbital_derivatives(state0 + k1/2) * dt  
        k3 = self._orbital_derivatives(state0 + k2/2) * dt
        k4 = self._orbital_derivatives(state0 + k3) * dt
        
        state_new = state0 + (k1 + 2*k2 + 2*k3 + k4) / 6
        
        return state_new[:3], state_new[3:]
    
    def _orbital_derivatives(self, state: np.ndarray) -> np.ndarray:
        """
        计算轨道微分方程的导数
        
        Args:
            state: 状态向量 [x, y, z, vx, vy, vz]
            
        Returns:
            derivatives: 导数向量 [vx, vy, vz, ax, ay, az]
        """
        r = state[:3]  # 位置
        v = state[3:]  # 速度
        
        r_mag = np.linalg.norm(r)
        
        # 防止除零
        if r_mag < self.EARTH_RADIUS:
            r_mag = self.EARTH_RADIUS
            r = r / np.linalg.norm(r) * self.EARTH_RADIUS
        
        # 主引力加速度
        a_gravity = -self.MU_EARTH * r / r_mag**3
        
        # J2摄动（简化）
        a_j2 = self._compute_j2_perturbation(r)
        
        # 总加速度
        a_total = a_gravity + a_j2
        
        # 返回导数：[位置导数=速度, 速度导数=加速度]
        return np.concatenate([v, a_total])
    
    def _compute_j2_perturbation(self, r: np.ndarray) -> np.ndarray:
        """
        计算J2摄动加速度
        
        Args:
            r: 位置向量 [x, y, z] (km)
            
        Returns:
            a_j2: J2摄动加速度 [ax, ay, az] (km/s²)
        """
        x, y, z = r
        r_mag = np.linalg.norm(r)
        
        if r_mag < self.EARTH_RADIUS:
            return np.zeros(3)
        
        # J2摄动系数
        factor = 1.5 * self.J2 * self.MU_EARTH * self.EARTH_RADIUS**2 / r_mag**5
        
        # J2摄动加速度分量
        a_x = factor * x * (5 * z**2 / r_mag**2 - 1)
        a_y = factor * y * (5 * z**2 / r_mag**2 - 1)  
        a_z = factor * z * (5 * z**2 / r_mag**2 - 3)
        
        return np.array([a_x, a_y, a_z])
    
    def apply_maneuver(self, satellite, maneuver_command: Dict):
        """
        应用机动指令到卫星
        
        Args:
            satellite: 卫星状态对象
            maneuver_command: 机动指令字典
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
        
        # 应用速度变化
        satellite.velocity += actual_delta_v
        
        self.logger.debug(f"Applied maneuver to satellite {satellite.sat_id}: "
                         f"delta_v={actual_delta_v}")
    
    def compute_intercept_trajectory(self, interceptor_pos: np.ndarray,
                                   interceptor_vel: np.ndarray,
                                   target_pos: np.ndarray,
                                   target_vel: np.ndarray,
                                   max_delta_v: float = 0.5) -> Optional[ManeuverCommand]:
        """
        计算拦截轨道
        
        Args:
            interceptor_pos: 拦截器位置 (km)
            interceptor_vel: 拦截器速度 (km/s) 
            target_pos: 目标位置 (km)
            target_vel: 目标速度 (km/s)
            max_delta_v: 最大可用delta_v (km/s)
            
        Returns:
            maneuver_command: 机动指令，如果无法拦截则返回None
        """
        # 简化的拦截计算：Lambert问题的近似解
        
        # 相对位置和速度
        rel_pos = target_pos - interceptor_pos
        rel_vel = target_vel - interceptor_vel
        
        # 估算拦截时间（基于当前相对运动）
        rel_distance = np.linalg.norm(rel_pos)
        if rel_distance < 1.0:  # 已经很近了
            return None
        
        # 估算飞行时间
        approach_speed = np.dot(rel_pos, rel_vel) / rel_distance
        if approach_speed <= 0:  # 目标远离
            # 计算追赶所需时间
            intercept_time = rel_distance / 5.0  # 假设平均追赶速度5km/s
        else:
            intercept_time = rel_distance / max(approach_speed, 1.0)
        
        # 限制拦截时间在合理范围内
        intercept_time = np.clip(intercept_time, 300.0, 3600.0)  # 5分钟到1小时
        
        # 预测目标未来位置
        future_target_pos, future_target_vel = self.propagate(
            target_pos, target_vel, intercept_time
        )
        
        # 计算所需的初始速度
        required_velocity = (future_target_pos - interceptor_pos) / intercept_time
        required_delta_v = required_velocity - interceptor_vel
        
        # 检查是否在能力范围内
        delta_v_magnitude = np.linalg.norm(required_delta_v)
        if delta_v_magnitude > max_delta_v:
            # 缩放到最大能力
            required_delta_v = required_delta_v * (max_delta_v / delta_v_magnitude)
        
        return ManeuverCommand(
            delta_v=required_delta_v,
            burn_time=10.0,  # 假设10秒推进时间
            direction='direct',
            target_position=future_target_pos
        )
    
    def compute_hohmann_transfer(self, current_pos: np.ndarray,
                               target_orbit_radius: float) -> ManeuverCommand:
        """
        计算霍曼转移轨道
        
        Args:
            current_pos: 当前位置 (km)
            target_orbit_radius: 目标轨道半径 (km)
            
        Returns:
            maneuver_command: 第一次机动指令
        """
        current_radius = np.linalg.norm(current_pos)
        
        # 霍曼转移的第一次delta_v
        v_current = np.sqrt(self.MU_EARTH / current_radius)
        v_transfer = np.sqrt(self.MU_EARTH * (2/current_radius - 2/(current_radius + target_orbit_radius)))
        
        delta_v_magnitude = v_transfer - v_current
        
        # 计算切向方向
        pos_unit = current_pos / current_radius
        # 假设在xy平面内，切向为垂直于径向
        tangent = np.array([-pos_unit[1], pos_unit[0], 0.0])
        if np.linalg.norm(tangent) < 1e-6:
            tangent = np.array([1.0, 0.0, 0.0])
        else:
            tangent = tangent / np.linalg.norm(tangent)
        
        delta_v = tangent * delta_v_magnitude
        
        return ManeuverCommand(
            delta_v=delta_v,
            burn_time=5.0,
            direction='prograde'
        )
    
    def cartesian_to_orbital_elements(self, position: np.ndarray, 
                                    velocity: np.ndarray) -> OrbitalElements:
        """
        笛卡尔坐标转换为轨道要素
        
        Args:
            position: 位置向量 [x, y, z] (km)
            velocity: 速度向量 [vx, vy, vz] (km/s)
            
        Returns:
            orbital_elements: 轨道要素
        """
        r = position
        v = velocity
        r_mag = np.linalg.norm(r)
        v_mag = np.linalg.norm(v)
        
        # 角动量向量
        h = np.cross(r, v)
        h_mag = np.linalg.norm(h)
        
        # 半长轴
        energy = v_mag**2 / 2 - self.MU_EARTH / r_mag
        if abs(energy) < 1e-12:  # 抛物线轨道
            semi_major_axis = float('inf')
        else:
            semi_major_axis = -self.MU_EARTH / (2 * energy)
        
        # 偏心率向量
        e_vec = ((v_mag**2 - self.MU_EARTH/r_mag) * r - np.dot(r, v) * v) / self.MU_EARTH
        eccentricity = np.linalg.norm(e_vec)
        
        # 轨道倾角
        inclination = np.arccos(np.clip(h[2] / h_mag, -1, 1))
        
        # 升交点向量
        n = np.cross([0, 0, 1], h)
        n_mag = np.linalg.norm(n)
        
        # 升交点赤经
        if n_mag > 1e-12:
            raan = np.arccos(np.clip(n[0] / n_mag, -1, 1))
            if n[1] < 0:
                raan = 2 * np.pi - raan
        else:
            raan = 0.0
        
        # 近地点幅角
        if n_mag > 1e-12 and eccentricity > 1e-12:
            argument_of_perigee = np.arccos(np.clip(np.dot(n, e_vec) / (n_mag * eccentricity), -1, 1))
            if e_vec[2] < 0:
                argument_of_perigee = 2 * np.pi - argument_of_perigee
        else:
            argument_of_perigee = 0.0
        
        # 真近点角
        if eccentricity > 1e-12:
            true_anomaly = np.arccos(np.clip(np.dot(e_vec, r) / (eccentricity * r_mag), -1, 1))
            if np.dot(r, v) < 0:
                true_anomaly = 2 * np.pi - true_anomaly
        else:
            # 圆轨道情况
            if n_mag > 1e-12:
                true_anomaly = np.arccos(np.clip(np.dot(n, r) / (n_mag * r_mag), -1, 1))
                if r[2] < 0:
                    true_anomaly = 2 * np.pi - true_anomaly
            else:
                true_anomaly = np.arccos(np.clip(r[0] / r_mag, -1, 1))
                if r[1] < 0:
                    true_anomaly = 2 * np.pi - true_anomaly
        
        return OrbitalElements(
            semi_major_axis=semi_major_axis,
            eccentricity=eccentricity,
            inclination=inclination,
            raan=raan,
            argument_of_perigee=argument_of_perigee,
            true_anomaly=true_anomaly
        )
    
    def orbital_elements_to_cartesian(self, elements: OrbitalElements) -> Tuple[np.ndarray, np.ndarray]:
        """
        轨道要素转换为笛卡尔坐标
        
        Args:
            elements: 轨道要素
            
        Returns:
            position: 位置向量 [x, y, z] (km)
            velocity: 速度向量 [vx, vy, vz] (km/s)
        """
        a = elements.semi_major_axis
        e = elements.eccentricity
        i = elements.inclination
        raan = elements.raan
        arg_pe = elements.argument_of_perigee
        nu = elements.true_anomaly
        
        # 轨道面内的位置和速度
        p = a * (1 - e**2)  # 半通径
        r_mag = p / (1 + e * np.cos(nu))
        
        # 轨道面内坐标
        r_orb = r_mag * np.array([np.cos(nu), np.sin(nu), 0])
        
        # 轨道面内速度
        h = np.sqrt(self.MU_EARTH * p)
        v_orb = (self.MU_EARTH / h) * np.array([-np.sin(nu), e + np.cos(nu), 0])
        
        # 转换矩阵：轨道面 -> 地心惯性系
        cos_raan = np.cos(raan)
        sin_raan = np.sin(raan)
        cos_i = np.cos(i)
        sin_i = np.sin(i)
        cos_arg = np.cos(arg_pe)
        sin_arg = np.sin(arg_pe)
        
        R11 = cos_raan * cos_arg - sin_raan * sin_arg * cos_i
        R12 = -cos_raan * sin_arg - sin_raan * cos_arg * cos_i
        R13 = sin_raan * sin_i
        
        R21 = sin_raan * cos_arg + cos_raan * sin_arg * cos_i
        R22 = -sin_raan * sin_arg + cos_raan * cos_arg * cos_i
        R23 = -cos_raan * sin_i
        
        R31 = sin_arg * sin_i
        R32 = cos_arg * sin_i
        R33 = cos_i
        
        R = np.array([[R11, R12, R13],
                      [R21, R22, R23], 
                      [R31, R32, R33]])
        
        # 转换到地心惯性系
        position = R @ r_orb
        velocity = R @ v_orb
        
        return position, velocity
    
    def compute_relative_motion(self, chief_pos: np.ndarray, chief_vel: np.ndarray,
                              deputy_pos: np.ndarray, deputy_vel: np.ndarray) -> Dict[str, np.ndarray]:
        """
        计算相对运动参数（Hill坐标系）
        
        Args:
            chief_pos: 主星位置 (km)
            chief_vel: 主星速度 (km/s)
            deputy_pos: 从星位置 (km)  
            deputy_vel: 从星速度 (km/s)
            
        Returns:
            relative_state: 相对运动状态字典
        """
        # 相对位置和速度
        rel_pos = deputy_pos - chief_pos
        rel_vel = deputy_vel - chief_vel
        
        # 主星轨道半径和角速度
        r_chief = np.linalg.norm(chief_pos)
        n = np.sqrt(self.MU_EARTH / r_chief**3)  # 平均运动
        
        # Hill坐标系基向量
        # 径向：指向地心方向
        e_r = -chief_pos / r_chief
        # 法向：角动量方向
        h_chief = np.cross(chief_pos, chief_vel)
        e_h = h_chief / np.linalg.norm(h_chief)
        # 切向：完成右手坐标系
        e_theta = np.cross(e_h, e_r)
        
        # Hill坐标系转换矩阵
        R_hill = np.array([e_r, e_theta, e_h])
        
        # 转换到Hill坐标系
        rel_pos_hill = R_hill @ rel_pos
        rel_vel_hill = R_hill @ rel_vel
        
        return {
            'relative_position': rel_pos_hill,
            'relative_velocity': rel_vel_hill,
            'radial': rel_pos_hill[0],
            'along_track': rel_pos_hill[1], 
            'cross_track': rel_pos_hill[2],
            'mean_motion': n,
            'hill_frame': R_hill
        }

    def compute_orbit_period(self, semi_major_axis: float) -> float:
        """
        计算轨道周期
        
        Args:
            semi_major_axis: 半长轴 (km)
            
        Returns:
            period: 轨道周期 (s)
        """
        return 2 * np.pi * np.sqrt(semi_major_axis**3 / self.MU_EARTH)
    
    def compute_escape_velocity(self, position: np.ndarray) -> float:
        """
        计算逃逸速度
        
        Args:
            position: 位置向量 (km)
            
        Returns:
            escape_velocity: 逃逸速度 (km/s)
        """
        r = np.linalg.norm(position)
        return np.sqrt(2 * self.MU_EARTH / r)
    
    def compute_closest_approach(self, pos1: np.ndarray, vel1: np.ndarray,
                               pos2: np.ndarray, vel2: np.ndarray) -> Tuple[float, float]:
        """
        计算两个轨道物体的最近接近距离和时间
        
        Args:
            pos1, vel1: 物体1的位置和速度
            pos2, vel2: 物体2的位置和速度
            
        Returns:
            min_distance: 最小距离 (km)
            time_to_closest: 到达最近点的时间 (s)
        """
        # 相对位置和速度
        rel_pos = pos1 - pos2
        rel_vel = vel1 - vel2
        
        # 如果相对速度为零，距离保持不变
        rel_speed_sq = np.dot(rel_vel, rel_vel)
        if rel_speed_sq < 1e-12:
            return np.linalg.norm(rel_pos), 0.0
        
        # 计算最近接近时间
        t_closest = -np.dot(rel_pos, rel_vel) / rel_speed_sq
        
        # 如果时间为负，表示最近点在过去
        if t_closest < 0:
            t_closest = 0.0
        
        # 计算最近距离
        closest_rel_pos = rel_pos + rel_vel * t_closest
        min_distance = np.linalg.norm(closest_rel_pos)
        
        return min_distance, t_closest
    
    def compute_phase_angle(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        """
        计算两个位置向量之间的相位角
        
        Args:
            pos1: 位置向量1 (km)
            pos2: 位置向量2 (km)
            
        Returns:
            phase_angle: 相位角 (rad)
        """
        # 投影到xy平面计算相位角
        angle1 = np.arctan2(pos1[1], pos1[0])
        angle2 = np.arctan2(pos2[1], pos2[0])
        
        phase_diff = angle2 - angle1
        
        # 标准化到[0, 2π]范围
        while phase_diff < 0:
            phase_diff += 2 * np.pi
        while phase_diff > 2 * np.pi:
            phase_diff -= 2 * np.pi
            
        return phase_diff
    
    def compute_synodic_period(self, orbit1_period: float, orbit2_period: float) -> float:
        """
        计算两个轨道的会合周期
        
        Args:
            orbit1_period: 轨道1周期 (s)
            orbit2_period: 轨道2周期 (s)
            
        Returns:
            synodic_period: 会合周期 (s)
        """
        if abs(orbit1_period - orbit2_period) < 1e-6:
            return float('inf')  # 同步轨道
        
        return abs(orbit1_period * orbit2_period / (orbit1_period - orbit2_period))
    
    def compute_station_keeping_delta_v(self, current_elements: OrbitalElements,
                                      target_elements: OrbitalElements) -> float:
        """
        计算轨道保持所需的delta_v
        
        Args:
            current_elements: 当前轨道要素
            target_elements: 目标轨道要素
            
        Returns:
            delta_v_required: 所需delta_v (km/s)
        """
        # 简化计算：基于轨道要素差异估算
        da = abs(target_elements.semi_major_axis - current_elements.semi_major_axis)
        de = abs(target_elements.eccentricity - current_elements.eccentricity)
        di = abs(target_elements.inclination - current_elements.inclination)
        
        # 经验公式估算
        delta_v = 0.0
        
        # 半长轴变化
        if da > 0:
            v_current = np.sqrt(self.MU_EARTH / current_elements.semi_major_axis)
            delta_v += abs(da / current_elements.semi_major_axis) * v_current * 0.5
        
        # 偏心率变化
        if de > 0:
            v_current = np.sqrt(self.MU_EARTH / current_elements.semi_major_axis)
            delta_v += de * v_current
        
        # 倾角变化
        if di > 0:
            v_current = np.sqrt(self.MU_EARTH / current_elements.semi_major_axis)
            delta_v += 2 * v_current * np.sin(di / 2)
        
        return delta_v
    
    def compute_collision_probability(self, pos1: np.ndarray, vel1: np.ndarray,
                                    pos2: np.ndarray, vel2: np.ndarray,
                                    radius1: float = 10.0, radius2: float = 10.0,
                                    time_window: float = 3600.0) -> float:
        """
        计算两个物体的碰撞概率
        
        Args:
            pos1, vel1: 物体1的位置和速度
            pos2, vel2: 物体2的位置和速度
            radius1, radius2: 物体半径 (km)
            time_window: 时间窗口 (s)
            
        Returns:
            collision_probability: 碰撞概率 [0, 1]
        """
        min_distance, time_to_closest = self.compute_closest_approach(
            pos1, vel1, pos2, vel2
        )
        
        # 如果最近接近时间超出时间窗口，概率为0
        if time_to_closest > time_window:
            return 0.0
        
        # 碰撞半径
        collision_radius = radius1 + radius2
        
        # 简化的概率计算
        if min_distance <= collision_radius:
            return 1.0  # 确定碰撞
        elif min_distance > collision_radius * 5:
            return 0.0  # 距离太远
        else:
            # 基于距离的概率衰减
            prob = np.exp(-(min_distance - collision_radius) / collision_radius)
            return min(prob, 1.0)
    
    def compute_ground_track(self, position: np.ndarray, velocity: np.ndarray,
                           duration: float, dt: float = 60.0) -> List[Tuple[float, float]]:
        """
        计算卫星地面轨迹
        
        Args:
            position: 初始位置 (km)
            velocity: 初始速度 (km/s)
            duration: 计算持续时间 (s)
            dt: 时间步长 (s)
            
        Returns:
            ground_track: 地面轨迹点列表 [(longitude, latitude)]
        """
        ground_track = []
        current_pos = position.copy()
        current_vel = velocity.copy()
        
        num_steps = int(duration / dt)
        
        for i in range(num_steps):
            # 转换为经纬度
            x, y, z = current_pos
            r = np.linalg.norm(current_pos)
            
            # 纬度
            latitude = np.arcsin(z / r) * 180.0 / np.pi
            
            # 经度（考虑地球自转）
            earth_rotation_rate = 7.2921159e-5  # rad/s
            time_elapsed = i * dt
            longitude = (np.arctan2(y, x) - earth_rotation_rate * time_elapsed) * 180.0 / np.pi
            
            # 标准化经度到[-180, 180]
            while longitude > 180:
                longitude -= 360
            while longitude < -180:
                longitude += 360
            
            ground_track.append((longitude, latitude))
            
            # 传播轨道
            current_pos, current_vel = self.propagate(current_pos, current_vel, dt)
        
        return ground_track
    
    def compute_sun_synchronous_inclination(self, altitude: float) -> float:
        """
        计算太阳同步轨道倾角
        
        Args:
            altitude: 轨道高度 (km)
            
        Returns:
            inclination: 轨道倾角 (rad)
        """
        a = self.EARTH_RADIUS + altitude
        
        # 太阳同步轨道的节点进动率
        precession_rate = 2 * np.pi / (365.25 * 24 * 3600)  # rad/s (每年一圈)
        
        # 计算所需倾角
        factor = -1.5 * self.J2 * (self.EARTH_RADIUS / a)**2 * np.sqrt(self.MU_EARTH / a**3)
        
        if abs(factor) > abs(precession_rate):
            cos_i = precession_rate / factor
            if abs(cos_i) <= 1:
                inclination = np.arccos(abs(cos_i))
                # 太阳同步轨道通常是逆行的
                if cos_i < 0:
                    inclination = np.pi - inclination
            else:
                # 无法实现太阳同步，返回极轨道倾角
                inclination = np.pi / 2
        else:
            inclination = np.pi / 2
        
        return inclination
    
    def validate_orbital_elements(self, elements: OrbitalElements) -> bool:
        """
        验证轨道要素的有效性
        
        Args:
            elements: 轨道要素
            
        Returns:
            is_valid: 是否有效
        """
        # 检查半长轴
        if elements.semi_major_axis <= self.EARTH_RADIUS:
            return False
        
        # 检查偏心率
        if elements.eccentricity < 0 or elements.eccentricity >= 1:
            return False
        
        # 检查倾角
        if elements.inclination < 0 or elements.inclination > np.pi:
            return False
        
        # 检查近地点高度
        perigee_altitude = elements.semi_major_axis * (1 - elements.eccentricity) - self.EARTH_RADIUS
        if perigee_altitude < 100:  # 最小高度100km
            return False
        
        return True
    
    def compute_transfer_window(self, departure_pos: np.ndarray, departure_vel: np.ndarray,
                              target_pos: np.ndarray, target_vel: np.ndarray,
                              max_duration: float = 86400.0) -> Optional[Tuple[float, float]]:
        """
        计算转移窗口
        
        Args:
            departure_pos: 出发位置 (km)
            departure_vel: 出发速度 (km/s)
            target_pos: 目标位置 (km)
            target_vel: 目标速度 (km/s)
            max_duration: 最大转移时间 (s)
            
        Returns:
            transfer_window: (最佳出发时间, 转移时间) 或 None
        """
        best_delta_v = float('inf')
        best_departure_time = None
        best_transfer_time = None
        
        # 搜索转移窗口
        for departure_delay in np.arange(0, max_duration, 3600):  # 每小时搜索一次
            # 传播出发点轨道
            dep_pos, dep_vel = self.propagate(departure_pos, departure_vel, departure_delay)
            
            for transfer_time in np.arange(1800, max_duration, 1800):  # 30分钟间隔
                # 传播目标轨道
                tgt_pos, tgt_vel = self.propagate(target_pos, target_vel, 
                                                departure_delay + transfer_time)
                
                # 计算Lambert问题（简化）
                delta_v = self._estimate_lambert_delta_v(dep_pos, tgt_pos, transfer_time)
                
                if delta_v < best_delta_v:
                    best_delta_v = delta_v
                    best_departure_time = departure_delay
                    best_transfer_time = transfer_time
        
        if best_departure_time is not None:
            return (best_departure_time, best_transfer_time)
        else:
            return None
    
    def _estimate_lambert_delta_v(self, r1: np.ndarray, r2: np.ndarray, 
                                 tof: float) -> float:
        """
        估算Lambert问题的delta_v需求
        
        Args:
            r1: 初始位置 (km)
            r2: 终端位置 (km)
            tof: 飞行时间 (s)
            
        Returns:
            delta_v_estimate: 估算的delta_v (km/s)
        """
        # 简化的Lambert问题求解
        r1_mag = np.linalg.norm(r1)
        r2_mag = np.linalg.norm(r2)
        cos_dnu = np.dot(r1, r2) / (r1_mag * r2_mag)
        
        # 转移角
        dnu = np.arccos(np.clip(cos_dnu, -1, 1))
        
        # 半弦长
        c = np.linalg.norm(r2 - r1)
        s = (r1_mag + r2_mag + c) / 2
        
        # 最小能量转移的半长轴
        a_min = s / 2
        
        # 最小能量转移时间
        if a_min > 0:
            tof_min = np.pi * np.sqrt(a_min**3 / self.MU_EARTH)
        else:
            return float('inf')
        
        # 根据时间比率估算delta_v
        time_ratio = tof / tof_min
        
        if time_ratio < 0.5 or time_ratio > 2.0:
            # 时间不合理，delta_v很大
            delta_v_factor = 5.0
        else:
            delta_v_factor = 1.0 + abs(time_ratio - 1.0)
        
        # 估算速度需求
        v1_circ = np.sqrt(self.MU_EARTH / r1_mag)
        v2_circ = np.sqrt(self.MU_EARTH / r2_mag)
        
        # 简化的delta_v估算
        delta_v_estimate = (abs(v1_circ - v2_circ) + 
                           0.5 * (v1_circ + v2_circ) * (dnu / np.pi)) * delta_v_factor
        
        return delta_v_estimate
    
    def compute_orbit_maintenance_requirements(self, target_elements: OrbitalElements,
                                             perturbation_time: float = 86400.0) -> Dict[str, float]:
        """
        计算轨道维持需求
        
        Args:
            target_elements: 目标轨道要素
            perturbation_time: 摄动分析时间 (s)
            
        Returns:
            maintenance_requirements: 维持需求字典
        """
        # 转换为笛卡尔坐标
        pos, vel = self.orbital_elements_to_cartesian(target_elements)
        
        # 传播轨道以计算摄动影响
        perturbed_pos, perturbed_vel = self.propagate(pos, vel, perturbation_time)
        perturbed_elements = self.cartesian_to_orbital_elements(perturbed_pos, perturbed_vel)
        
        # 计算轨道要素变化
        da = abs(perturbed_elements.semi_major_axis - target_elements.semi_major_axis)
        de = abs(perturbed_elements.eccentricity - target_elements.eccentricity)
        di = abs(perturbed_elements.inclination - target_elements.inclination)
        draan = abs(perturbed_elements.raan - target_elements.raan)
        darg_pe = abs(perturbed_elements.argument_of_perigee - target_elements.argument_of_perigee)
        
        # 计算维持所需delta_v
        maintenance_delta_v = self.compute_station_keeping_delta_v(
            perturbed_elements, target_elements
        )
        
        # 维持频率估算（基于摄动程度）
        perturbation_severity = da / target_elements.semi_major_axis + de + di
        maintenance_frequency = max(1, int(perturbation_time / (perturbation_severity * 86400)))
        
        return {
            'delta_v_per_maintenance': maintenance_delta_v,
            'maintenance_frequency_days': maintenance_frequency,
            'annual_delta_v_budget': maintenance_delta_v * (365.25 / maintenance_frequency),
            'semi_major_axis_drift': da,
            'eccentricity_drift': de,
            'inclination_drift': di,
            'raan_drift': draan,
            'arg_perigee_drift': darg_pe
        }