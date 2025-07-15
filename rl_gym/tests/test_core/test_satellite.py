# tests/core/test_satellite.py

import unittest
from astropy import units as u
from astropy.time import Time
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
import numpy as np
# 假设你的项目结构正确，可以从 environment 包中导入 Satellite 类
from environment.core.satellite import Satellite


class TestSatellite(unittest.TestCase):
    """针对 Satellite 类的单元测试集"""

    def setUp(self):
        """在每个测试方法运行前，设置一个标准的轨道和卫星实例。"""
        self.initial_r = [10000.0, 0.0, 0.0] * u.km
        self.initial_v = [0.0, 7.5, 0.0] * u.km / u.s
        self.initial_epoch = Time("2025-01-01 12:00:00", scale="utc")
        self.orbit = Orbit.from_vectors(Earth, self.initial_r, self.initial_v, epoch=self.initial_epoch)

        self.sat = Satellite(
            sat_id="test_sat_01",
            initial_orbit=self.orbit,
            mass_wet=1000 * u.kg,
            mass_dry=500 * u.kg,
            isp=300 * u.s,
            attributes={"role": "observer"}
        )

    def test_initialization_success(self):
        """测试卫星能否成功初始化。"""
        self.assertEqual(self.sat.sat_id, "test_sat_01")
        self.assertEqual(self.sat.mass, 1000 * u.kg)
        self.assertEqual(self.sat.dry_mass, 500 * u.kg)
        self.assertEqual(self.sat.isp, 300 * u.s)
        self.assertEqual(self.sat.attributes["role"], "observer")
        self.assertTrue(repr(self.sat).startswith("<Satellite id='test_sat_01'"))

    def test_initialization_failure_invalid_mass(self):
        """测试当湿重小于干重时，是否抛出 ValueError。"""
        with self.assertRaises(ValueError):
            Satellite(
                sat_id="fail_sat",
                initial_orbit=self.orbit,
                mass_wet=400 * u.kg,
                mass_dry=500 * u.kg,
                isp=300 * u.s
            )

    def test_properties(self):
        """测试所有属性是否返回正确的值。"""
        self.assertTrue(np.allclose(self.sat.r.to_value(u.km), self.initial_r.to_value(u.km)))
        self.assertTrue(np.allclose(self.sat.v.to_value(u.km / u.s), self.initial_v.to_value(u.km / u.s)))
        self.assertEqual(self.sat.epoch, self.initial_epoch)
        self.assertAlmostEqual(self.sat.fuel_mass.to_value(u.kg), 500.0)
        self.assertTrue(self.sat.can_maneuver)

    def test_update_orbit(self):
        """测试轨道更新方法。"""
        new_r = [11000.0, 0.0, 0.0] * u.km
        new_v = [0.0, 7.0, 0.0] * u.km / u.s
        new_orbit = Orbit.from_vectors(Earth, new_r, new_v, epoch=self.initial_epoch)
        
        self.sat.update_orbit(new_orbit)
        self.assertEqual(self.sat.orbit, new_orbit)
        
        with self.assertRaises(TypeError):
            self.sat.update_orbit("not_an_orbit")

    def test_consume_fuel_sufficient_fuel(self):
        """测试燃料充足情况下的消耗。"""
        delta_v = 50 * u.m / u.s
        fuel_consumed = self.sat.consume_fuel(delta_v)

        # 理论计算
        g0 = 9.80665 * u.m / u.s**2
        expected_final_mass = (1000 * u.kg) * np.exp(-delta_v / (300 * u.s * g0))
        expected_fuel_consumed = (1000 * u.kg) - expected_final_mass
        
        self.assertAlmostEqual(self.sat.mass.to_value(u.kg), expected_final_mass.to_value(u.kg), places=5)
        self.assertAlmostEqual(fuel_consumed.to_value(u.kg), expected_fuel_consumed.to_value(u.kg), places=5)

    def test_consume_fuel_insufficient_fuel(self):
        """测试燃料不足时，消耗所有剩余燃料。"""
        # 一个巨大的机动，理论上会耗尽燃料
        delta_v = 10000 * u.m / u.s
        fuel_consumed = self.sat.consume_fuel(delta_v)
        
        # 最终质量应等于干重
        self.assertAlmostEqual(self.sat.mass.to_value(u.kg), self.sat.dry_mass.to_value(u.kg))
        # 消耗的燃料应等于初始燃料
        self.assertAlmostEqual(fuel_consumed.to_value(u.kg), 500.0)
        # 此时应无法再机动
        self.assertFalse(self.sat.can_maneuver)
        
    def test_consume_fuel_no_fuel(self):
        """测试燃料耗尽后尝试机动。"""
        # 先耗尽燃料
        self.sat.mass = self.sat.dry_mass
        
        delta_v = 50 * u.m / u.s
        fuel_consumed = self.sat.consume_fuel(delta_v)
        
        # 质量不应变化，消耗为0
        self.assertEqual(self.sat.mass, self.sat.dry_mass)
        self.assertEqual(fuel_consumed.to_value(u.kg), 0.0)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)