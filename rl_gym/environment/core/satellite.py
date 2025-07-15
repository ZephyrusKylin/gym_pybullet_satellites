# -*- coding: utf-8 -*-

"""
environment/core/satellite.py

本文件定义了 `Satellite` 类，它是物理引擎（第一层）的核心对象。
该类封装了卫星的所有基础物理属性，是仿真世界中所有卫星个体的“数字真身”。
"""

from typing import Dict, Any
import numpy as np

from astropy import units as u
from astropy.time import Time

from poliastro.twobody import Orbit
from astropy.constants import g0 # 这是更稳定和推荐的导入方式



class Satellite:
    """
    代表单个卫星的物理状态和属性。

    这个类作为仿真环境中的基础物理单元，其职责是精确地持有和管理卫星的
    轨道状态、质量属性和动力学参数。它通过严格的单位管理（使用 astropy.units）
    来确保所有物理计算的准确性。

    Attributes:
        sat_id (str): 卫星的唯一标识符。
        orbit (poliastro.twobody.Orbit): 描述卫星当前轨道的 Poliastro 对象。
        mass (astropy.units.Quantity): 卫星的当前总质量（湿重）。
        dry_mass (astropy.units.Quantity): 卫星的干重（不含燃料）。
        isp (astropy.units.Quantity): 发动机的比冲，决定了燃料效率。
        attributes (Dict[str, Any]): 一个存储非物理属性的字典，用于游戏逻辑层（如角色、能力等）。
    """

    def __init__(
        self,
        sat_id: str,
        initial_orbit: Orbit,
        mass_wet: u.Quantity,
        mass_dry: u.Quantity,
        isp: u.Quantity,
        attributes: Dict[str, Any] = None,
    ):
        """
        初始化一个 Satellite 对象。

        Args:
            sat_id (str): 卫星的唯一标识符，例如 'ally_01' 或 'enemy_interceptor'。
            initial_orbit (Orbit): 初始的 Poliastro 轨道对象。
            mass_wet (u.Quantity): 初始总质量（湿重），必须是 astropy 的质量单位 (e.g., 1000 * u.kg)。
            mass_dry (u.Quantity): 卫星的干重，必须是 astropy 的质量单位 (e.g., 500 * u.kg)。
            isp (u.Quantity): 发动机的比冲，必须是 astropy 的时间单位 (e.g., 300 * u.s)。
            attributes (Dict[str, Any], optional): 卫星的非物理属性。默认为 None。
        """
        self.sat_id = sat_id
        self.orbit = initial_orbit

        # --- 强制单位，确保物理计算的严谨性 ---
        self.mass = mass_wet.to(u.kg)
        self.dry_mass = mass_dry.to(u.kg)
        self.isp = isp.to(u.s)

        self.attributes = attributes if attributes is not None else {}

        if self.mass < self.dry_mass:
            raise ValueError(f"卫星 '{self.sat_id}': 湿重 ({self.mass}) 不能小于干重 ({self.dry_mass})。")

    def __repr__(self) -> str:
        return (
            f"<Satellite id='{self.sat_id}' "
            f"mass={self.mass:.2f}, "
            f"fuel={self.fuel_mass:.2f}, "
            f"orbit=a:{self.orbit.a:.0f}, ecc:{self.orbit.ecc:.4f}, inc:{self.orbit.inc:.2f}>"
        )

    # ----------------------------------------------------
    # Properties: 便于上层逻辑访问核心状态的只读属性
    # ----------------------------------------------------
    @property
    def r(self) -> u.Quantity:
        """当前的位置矢量 (IJK)"""
        return self.orbit.r

    @property
    def v(self) -> u.Quantity:
        """当前的速度矢量 (IJK)"""
        return self.orbit.v

    @property
    def epoch(self) -> Time:
        """当前的轨道历元"""
        return self.orbit.epoch

    @property
    def fuel_mass(self) -> u.Quantity:
        """计算并返回当前的剩余燃料质量"""
        return self.mass - self.dry_mass

    @property
    def can_maneuver(self) -> bool:
        """判断卫星是否还有燃料进行机动"""
        return self.mass > self.dry_mass

    # ----------------------------------------------------
    # Methods: 更新自身物理状态的核心方法
    # ----------------------------------------------------
    def update_orbit(self, new_orbit: Orbit) -> None:
        """
        直接更新卫星的轨道状态。

        这个方法通常由 propagator (轨道传播器) 或 maneuver_planner (机动规划器) 调用。
        """
        if not isinstance(new_orbit, Orbit):
            raise TypeError("提供的 new_orbit 必须是 poliastro.twobody.Orbit 类型。")
        self.orbit = new_orbit

    def consume_fuel(self, delta_v_magnitude: u.Quantity) -> u.Quantity:
        """
        根据给定的速度增量大小，计算并消耗燃料。

        此方法应用齐奥尔科夫斯基火箭方程来更新卫星的总质量。

        Args:
            delta_v_magnitude (u.Quantity): 本次机动所需的速度增量大小 (标量)。

        Returns:
            u.Quantity: 本次机动实际消耗的燃料质量。如果燃料不足，则返回所有剩余燃料。
        """
        if not self.can_maneuver:
            return 0 * u.kg

        # 确保传入的是一个标量速度
        dv_scalar = delta_v_magnitude.to_value(u.m / u.s) * u.m / u.s
        
        # 计算理论上需要的燃料
        # m_final = m_initial * exp(-delta_v / (isp * g0))
        m_initial = self.mass
        m_final_ideal = m_initial * np.exp(-dv_scalar / (self.isp * g0))
        
        fuel_needed = m_initial - m_final_ideal

        # 检查燃料是否充足
        if fuel_needed > self.fuel_mass:
            fuel_consumed = self.fuel_mass
            self.mass = self.dry_mass
        else:
            fuel_consumed = fuel_needed
            self.mass = m_final_ideal
        
        return fuel_consumed.to(u.kg)