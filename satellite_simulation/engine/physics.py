# engine/physics.py
# 该模块需要 astropy: pip install astropy

import numpy as np
import astropy.units as u
from astropy.units import Quantity

class RelativePhysics:
    """
    一个与 astropy.units 兼容的、无状态的物理计算工具类。

    该类的所有方法均为静态方法，并且被设计为接收和返回 astropy.units.Quantity 对象。
    这确保了在整个引擎中的物理计算都是量纲安全的。
    """

    @staticmethod
    def propagate(position: Quantity, velocity: Quantity, dt: Quantity) -> Quantity:
        """
        根据当前速度和时间增量，计算物体因惯性运动达到的新位置。

        Args:
            position (Quantity): 当前位置矢量 (e.g., in u.m)。
            velocity (Quantity): 当前速度矢量 (e.g., in u.m/u.s)。
            dt (Quantity): 时间增量 (e.g., in u.s)。

        Returns:
            Quantity: 更新后的新位置矢量，单位与输入位置相同。
        """
        # astropy 会自动处理单位转换和计算
        # (m) = (m) + (m/s) * (s)
        return position + velocity * dt

    @staticmethod
    def apply_thrust(velocity: Quantity, mass: Quantity, thrust_vector: Quantity, dt: Quantity) -> Quantity:
        """
        根据推力、质量和时间增量，计算速度的变化。

        Args:
            velocity (Quantity): 施加推力前的速度矢量 (e.g., in u.m/u.s)。
            mass (Quantity): 物体质量 (e.g., in u.kg)。
            thrust_vector (Quantity): 施加的推力向量 (e.g., in u.N)。
            dt (Quantity): 推力作用的时间 (e.g., in u.s)。

        Returns:
            Quantity: 施加推力后得到的新速度矢量。
        """
        # a = F/m -> (m/s^2) = (N)/(kg) = (kg*m/s^2)/(kg)
        acceleration = (thrust_vector / mass).to(u.m / u.s**2)
        
        # v = v_0 + a*t -> (m/s) = (m/s) + (m/s^2)*(s)
        return velocity + acceleration * dt

    @staticmethod
    def estimate_maneuver(
        current_position: Quantity,
        target_position: Quantity,
        current_velocity: Quantity,
        max_thrust: Quantity,
        mass: Quantity
    ) -> tuple[Quantity, Quantity]:
        """
        粗略估算一个机动任务所需的时间和燃料（以delta-V形式表示）。
        所有输入和输出都带有物理单位。

        Args:
            current_position (Quantity): 当前位置向量 (e.g., in u.m)。
            target_position (Quantity): 目标位置向量 (e.g., in u.m)。
            current_velocity (Quantity): 当前速度向量 (e.g., in u.m/u.s)。
            max_thrust (Quantity): 卫星能提供的最大推力 (e.g., in u.N)。
            mass (Quantity): 卫星质量 (e.g., in u.kg)。

        Returns:
            tuple[Quantity, Quantity]: 一个包含两个元素的元组：
                - estimated_time (Quantity): 预估的总机动时间 (in u.s)。
                - estimated_delta_v (Quantity): 预估的总速度增量 (in u.m/u.s)。
        """
        if max_thrust <= 0 * u.N:
            return float('inf') * u.s, float('inf') * u.m / u.s

        # 确保所有计算都在 SI 单位下进行，避免浮点精度问题
        max_acceleration = (max_thrust / mass).to(u.m / u.s**2)
        
        # --- 阶段一：刹车 ---
        velocity_norm = np.linalg.norm(current_velocity.to_value(u.m / u.s)) * u.m / u.s
        delta_v_brake = velocity_norm
        time_brake = (velocity_norm / max_acceleration).to(u.s)

        drift_distance = (velocity_norm * time_brake - 0.5 * max_acceleration * time_brake**2).to(u.m)
        
        velocity_direction = current_velocity / velocity_norm if velocity_norm > 0 * u.m/u.s else np.array([0,0,0]) * u.one
        position_after_brake = current_position + velocity_direction * drift_distance

        # --- 阶段二：点对点转移 ---
        transfer_vector = target_position - position_after_brake
        transfer_distance = np.linalg.norm(transfer_vector.to_value(u.m)) * u.m

        if transfer_distance > 1e-6 * u.m:
            time_transfer = (2 * np.sqrt(transfer_distance / max_acceleration)).to(u.s)
            delta_v_transfer = (max_acceleration * time_transfer).to(u.m / u.s)
        else:
            time_transfer = 0.0 * u.s
            delta_v_transfer = 0.0 * u.m / u.s

        # --- 汇总 ---
        total_time = time_brake + time_transfer
        total_delta_v = delta_v_brake + delta_v_transfer

        return total_time, total_delta_v