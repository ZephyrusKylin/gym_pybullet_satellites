# satellite_combat_rl/engine/unit.py
# 首先，你需要安装 astropy: pip install astropy

from dataclasses import dataclass, field
from typing import Dict, Any

import astropy.units as u
from astropy.units import Quantity

@dataclass
class Satellite:
    """
    一个纯粹的、带有物理单位的数据容器，用于表征空间对抗引擎中的一颗卫星。

    通过集成 `astropy.units`，该类中的所有物理量都与它们的单位绑定，
    确保了在整个引擎中的量纲安全和单位转换的正确性。
    """

    # === 身份属性 ===
    id: int
    team_id: int

    # === 核心物理状态 (使用 astropy.units.Quantity) ===
    position: Quantity
    """卫星的三维位置坐标矢量，必须是带有长度单位的 Quantity 对象 (e.g., u.m, u.km)。"""

    velocity: Quantity
    """卫星的三维速度矢量，必须是带有速度单位的 Quantity 对象 (e.g., u.m/u.s)。"""

    mass: Quantity
    """卫星当前的总质量，必须是带有质量单位的 Quantity 对象 (e.g., u.kg)。"""

    # === 资源与性能属性 ===
    fuel_mass: Quantity
    """卫星当前剩余的燃料质量，必须是带有质量单位的 Quantity 对象 (e.g., u.kg)。"""

    engine_isp: Quantity
    """发动机的比冲，必须是带有时间单位的 Quantity 对象 (e.g., u.s)。"""

    # === 任务状态属性 (由引擎动态修改) ===
    current_task: str = "IDLE"
    task_params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """
        在对象创建后进行物理单位和数值的校验。
        """
        # 校验单位的物理类型是否正确
        if not self.position.unit.is_equivalent(u.m):
            raise u.UnitConversionError(f"Position unit {self.position.unit} is not a length unit.")
        if not self.velocity.unit.is_equivalent(u.m / u.s):
            raise u.UnitConversionError(f"Velocity unit {self.velocity.unit} is not a velocity unit.")
        if not self.mass.unit.is_equivalent(u.kg):
            raise u.UnitConversionError(f"Mass unit {self.mass.unit} is not a mass unit.")
        if not self.fuel_mass.unit.is_equivalent(u.kg):
            raise u.UnitConversionError(f"Fuel mass unit {self.fuel_mass.unit} is not a mass unit.")
        if not self.engine_isp.unit.is_equivalent(u.s):
            raise u.UnitConversionError(f"ISP unit {self.engine_isp.unit} is not a time unit.")

        # 校验数值逻辑
        if self.fuel_mass > self.mass:
            raise ValueError(f"Satellite {self.id}: fuel_mass ({self.fuel_mass}) cannot be greater than total mass ({self.mass}).")
        if self.fuel_mass < 0 * u.kg:
            raise ValueError(f"Satellite {self.id}: fuel_mass ({self.fuel_mass}) cannot be negative.")


if __name__ == '__main__':
    # --- 演示如何创建和使用带有单位的 Satellite 实例 ---

    # 1. 初始化卫星，可以自由使用方便的单位
    sat_1 = Satellite(
        id=1,
        team_id=0,
        position=[42164, 0, 0] * u.km,  # 使用千米
        velocity=[0, 3.074, 0] * u.km / u.s,  # 使用千米/秒
        mass=1 * u.t,  # 使用吨
        fuel_mass=200 * u.kg,  # 使用千克
        engine_isp=300 * u.s,
    )

    print("--- 初始状态 ---")
    print(sat_1)

    # 2. 演示单位的自动转换和访问
    print("\n--- 访问不同单位的数据 ---")
    # 无论初始单位是什么，都可以轻松转换为国际标准单位 (SI) 或任何其他兼容单位
    print(f"位置 (米): {sat_1.position.to(u.m)}")
    print(f"速度 (米/秒): {sat_1.velocity.to(u.m / u.s)}")
    print(f"总质量 (千克): {sat_1.mass.to(u.kg)}")

    # 3. 演示带有物理意义的计算
    print("\n--- 物理计算演示 ---")
    # 假设引擎控制器计算出需要消耗 5kg 燃料
    fuel_consumed = 5 * u.kg
    sat_1.fuel_mass -= fuel_consumed
    sat_1.mass -= fuel_consumed # 总质量也随之减少

    print(f"消耗 {fuel_consumed} 燃料后，剩余燃料: {sat_1.fuel_mass}")
    print(f"消耗燃料后，总质量: {sat_1.mass.to(u.kg)}")

    # 4. 演示单位安全检查
    print("\n--- 单位安全检查 ---")
    try:
        # 尝试进行一次非法的物理运算：将位置和质量相加
        invalid_op = sat_1.position + sat_1.mass
    except u.UnitConversionError as e:
        print(f"成功捕获到非法运算错误: {e}")