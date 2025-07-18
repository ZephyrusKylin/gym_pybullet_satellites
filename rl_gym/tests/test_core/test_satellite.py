# tests/core/test_satellite.py

import unittest
import numpy as np
from astropy import units as u
from astropy.time import Time
from astropy.constants import g0
# 导入核心模块和待测类
from environment.core.constants import EARTH_MU
from environment.core.satellite import Satellite

class TestPoliastroSatellite(unittest.TestCase):
    """针对最终版 satellite.py 的单元测试集"""

    def setUp(self):
        """
        在每个测试方法运行前，使用 from_classical 工厂方法
        设置一个标准的卫星实例。
        """
        self.sat_id = "test_sat_01"
        self.epoch = Time("2025-07-16 00:00:00", scale="utc")
        self.mu_with_unit = EARTH_MU * (u.km**3 / u.s**2)

        self.mass_wet = 1000 * u.kg
        self.dry_mass = 500 * u.kg
        self.isp = 300 * u.s
        
        # 使用 from_classical 直接创建实例
        self.sat = Satellite.from_classical(
            sat_id=self.sat_id,
            a=7000 * u.km, ecc=0.001 * u.one, inc=15.0 * u.deg,
            raan=0.0 * u.deg, argp=0.0 * u.deg, nu=0.0 * u.deg,
            mass_wet=self.mass_wet,
            dry_mass=self.dry_mass,
            isp=self.isp,
            epoch=self.epoch,
            mu=self.mu_with_unit, # 传递带单位的引力常数
            attributes={"role": "tester"}
        )

    def test_creation_and_attributes(self):
        """测试：通过 from_classical 创建的实例，其属性是否正确。"""
        self.assertIsInstance(self.sat, Satellite)
        self.assertEqual(self.sat.sat_id, self.sat_id)
        self.assertEqual(self.sat.attributes["role"], "tester")
        
        # 验证质量
        self.assertEqual(self.sat.mass, self.mass_wet)
        self.assertEqual(self.sat.dry_mass, self.dry_mass)
        
        # 验证轨道根数被正确设置
        self.assertAlmostEqual(self.sat.orbit.a.to_value(u.km), 7000)
        self.assertAlmostEqual(self.sat.orbit.inc.to_value(u.deg), 15.0)

    def test_properties(self):
        """测试：所有派生属性是否计算正确。"""
        self.assertAlmostEqual(self.sat.fuel_mass.to_value(u.kg), 500.0)
        self.assertTrue(self.sat.can_maneuver)
        
        # 测试燃料耗尽的情况
        self.sat.mass = self.sat.dry_mass
        self.assertFalse(self.sat.can_maneuver)

    def test_update_orbit(self):
        """测试：轨道更新方法。"""
        from poliastro.twobody import Orbit
        # 创建一个新轨道
        new_orbit = Orbit.from_classical(
            attractor=self.sat.orbit.attractor, a=8000 * u.km, 
            ecc=0.05 * u.one, inc=20 * u.deg, raan=10 * u.deg, 
            argp=10 * u.deg, nu=10 * u.deg, epoch=self.epoch
        )
        self.sat.update_orbit(new_orbit)
        # 验证半长轴是否已更新
        self.assertAlmostEqual(self.sat.orbit.a.to_value(u.km), 8000)

    def test_consume_fuel_scenarios(self):
        """测试：全面测试燃料消耗逻辑。"""
        # 场景1：燃料充足
        delta_v_small = 50 * u.m / u.s
        
        # 修正点 2：不再使用硬编码的魔法数字，而是在测试中重新计算期望值
        # 这样可以保证测试逻辑与函数实现逻辑的一致性
        expected_mass = self.mass_wet * np.exp(-delta_v_small / (self.isp * g0))
        
        fuel_consumed = self.sat.consume_fuel(delta_v_small)
        self.assertGreater(fuel_consumed, 0 * u.kg)
        # 使用动态计算的期望值进行断言，并将精度提高
        self.assertAlmostEqual(self.sat.mass.to_value(u.kg), expected_mass.to_value(u.kg), places=5)
        
        # 场景2：燃料不足
        self.setUp() # 重置卫星状态
        delta_v_large = 5000 * u.m / u.s
        fuel_consumed_insufficient = self.sat.consume_fuel(delta_v_large)
        self.assertAlmostEqual(self.sat.mass.to_value(u.kg), self.dry_mass.to_value(u.kg))
        self.assertAlmostEqual(fuel_consumed_insufficient.to_value(u.kg), 500.0)

        # 场景3：燃料耗尽
        self.setUp()
        self.sat.mass = self.dry_mass
        fuel_consumed_none = self.sat.consume_fuel(delta_v_small)
        self.assertEqual(fuel_consumed_none.to_value(u.kg), 0.0)

    def test_initialization_failure(self):
        """测试：当湿重小于干重时，初始化应失败。"""
        with self.assertRaises(ValueError):
            Satellite.from_classical(
                sat_id="fail_sat",
                a=7000 * u.km, ecc=0.001 * u.one, inc=15.0 * u.deg,
                raan=0.0 * u.deg, argp=0.0 * u.deg, nu=0.0 * u.deg,
                mass_wet=400 * u.kg, # 湿重 < 干重
                dry_mass=500 * u.kg,
                isp=300 * u.s,
                epoch=self.epoch,
                mu=self.mu_with_unit
            )

if __name__ == '__main__':
    unittest.main()