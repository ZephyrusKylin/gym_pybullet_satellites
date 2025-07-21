# tests/logic/test_maneuver_planner_v2.py

import unittest
import numpy as np
from astropy import units as u
from astropy.time import Time, TimeDelta

# 导入核心模块
from environment.core.satellite import Satellite
from environment.core.constants import EARTH_MU
from environment.core import propagator
# 注意：在实际项目中，这里应该导入项目自己的 propagator
# 为了让这个例子可以独立运行，我们直接使用 poliastro 的基本传播功能

# 导入待测试的模块和数据结构
from environment.logic import maneuver_planner as planner
from environment.logic.maneuver_planner import ManeuverPlan, FORMATION_CATALOG

class TestManeuverPlannerV2(unittest.TestCase):
    """
    针对 maneuver_planner.py 的“后果驱动”的整合测试集。
    V2 版本核心思想：不仅验证计划本身，更要验证计划执行后的物理后果。
    """

    def setUp(self):
        """设置两个初始轨道略有不同的卫星，用于测试。"""
        self.epoch = Time("2025-07-16 00:00:00", scale="utc")
        
        # 我方卫星A (修改为中地球轨道 MEO)
        self.satellite_A = Satellite.from_classical(
            sat_id="sat_A",
            a=20000 * u.km,  # MEO
            ecc=0.001 * u.one,
            inc=55 * u.deg,
            raan=10 * u.deg, argp=20 * u.deg, nu=30 * u.deg,
            mass_wet=1000 * u.kg, dry_mass=500 * u.kg, isp=300 * u.s,
            epoch=self.epoch, mu=EARTH_MU * (u.km**3 / u.s**2)
        )
        
        # ----------------------------------------------------
        
        # 目标卫星B (保持地球同步轨道 GEO)
        self.satellite_B = Satellite.from_classical(
            sat_id="sat_B",
            a=42164 * u.km,  # GEO
            ecc=0.001 * u.one,
            inc=0.1 * u.deg,
            raan=10 * u.deg, argp=20 * u.deg, nu=31 * u.deg,
            mass_wet=2000 * u.kg, dry_mass=1500 * u.kg, isp=300 * u.s,
            epoch=self.epoch, mu=EARTH_MU * (u.km**3 / u.s**2)
        )
        self.hohmann_sat = Satellite.from_classical(
            sat_id="hohmann_sat",
            a=42164 * u.km,
            ecc=0.0 * u.one,  # <-- 关键修正：严格遵守函数契约，使用正圆轨道
            inc=0.1 * u.deg,
            raan=10 * u.deg,
            argp=0 * u.deg,   # 对于圆轨道，argp没有意义，设为0
            nu=50 * u.deg,    # nu + argp = arglat
            mass_wet=1000 * u.kg, dry_mass=500 * u.kg, isp=300 * u.s,
            epoch=self.epoch, mu=EARTH_MU * (u.km**3 / u.s**2)
        )
        self.satellites = {"sat_A": self.satellite_A, "sat_B": self.satellite_B}
    def _apply_maneuver_plan(self, satellite: Satellite, plan: ManeuverPlan) -> Satellite:
        """
        辅助方法：执行一个完整的机动计划，返回执行后的卫星状态。
        这模拟了物理后果。
        """
        current_orbit = satellite.orbit
        for maneuver in plan.maneuvers:
            # 1. 传播到机动时刻
            time_to_maneuver = maneuver.execution_time - current_orbit.epoch
            propagated_orbit = current_orbit.propagate(time_to_maneuver)
            
            # 2. 施加脉冲
            # poliastro的apply_maneuver需要一个(time_delta, dv)的列表
            final_orbit = propagated_orbit.apply_maneuver([(0 * u.s, maneuver.delta_v)])
            current_orbit = final_orbit
        
        satellite.orbit = current_orbit # 更新卫星的最终状态
        return satellite

    def test_plan_lambert_intercept_consequence(self):
        """测试后果：兰伯特拦截后，是否真的到达目标位置。"""
        
        # 修正：使用一个物理上更合理的飞行时间，比如8小时
        # 这比最低能量的霍曼转移(约7.5小时)略长，确保 delta_v 不会过大
        tof = TimeDelta(8 * u.h)
        
        plan = planner.plan_lambert_intercept(self.satellite_A, self.satellite_B, tof)
        
        # 修正：在执行计划前，必须先断言 plan 不是 None
        self.assertIsNotNone(plan, f"兰伯特拦截计划生成失败，tof={tof} 可能不合适或超出了合理Delta_V限制。")

        # 执行计划
        final_satellite_A = self._apply_maneuver_plan(self.satellite_A, plan)
        
        # 预测目标卫星在同样时间后的位置
        # final_satellite_B_orbit = self.satellite_B.orbit.propagate(tof)
        final_satellite_B_orbit = propagator.propagate_orbit_with_j2(self.satellite_B.orbit, tof)
        
        # 断言：最终两者距离是否小于1km (物理后果)
        final_distance = np.linalg.norm(
            (final_satellite_A.orbit.r - final_satellite_B_orbit.r).to_value(u.km)
        )
        self.assertLess(final_distance, 1.0, "兰伯特拦截后未到达目标附近")

    def test_plan_lambert_intercept_failure(self):
        """测试边界：对于不可能完成的兰伯特任务，是否返回None。"""
        tof = TimeDelta(1 * u.min) # 给定一个极短、不可能完成的时间
        plan = planner.plan_lambert_intercept(self.satellite_A, self.satellite_B, tof)
        self.assertIsNone(plan, "对于不可能的兰伯特任务，规划器应返回None")

    # def test_plan_hohmann_transfer_consequence(self):
    #     """测试后果：霍曼转移后，轨道根数是否正确。"""
    #     final_radius = 45000 * u.km
    #     plan = planner.plan_hohmann_transfer(self.satellite_A, final_radius)
        
    #     # 执行计划
    #     final_satellite_A = self._apply_maneuver_plan(self.satellite_A, plan)
        
    #     # 断言：最终轨道半长轴和偏心率是否符合预期 (物理后果)
    #     final_a = final_satellite_A.orbit.a.to_value(u.km)
    #     final_ecc = final_satellite_A.orbit.ecc.value
        
    #     self.assertAlmostEqual(final_a, final_radius.to_value(u.km), delta=1.0)
    #     self.assertAlmostEqual(final_ecc, 0, places=3)
    def test_plan_hohmann_transfer_consequence(self):
        """测试后果：霍曼转移后，轨道根数是否正确。"""
        
        # -- 修正：创建一个满足函数前提条件（ecc=0）的、全新的测试卫星 --
        # 我们不能使用 self.satellite_A，因为它不满足“圆轨道”的契约。
        # hohmann_test_sat = Satellite.from_classical(
        #     sat_id="hohmann_sat",
        #     a=42164 * u.km,
        #     ecc=0.0 * u.one,  # <-- 关键修正：严格遵守函数契约，使用正圆轨道
        #     inc=0.1 * u.deg,
        #     raan=10 * u.deg,
        #     argp=0 * u.deg,   # 对于圆轨道，argp没有意义，设为0
        #     nu=50 * u.deg,    # nu + argp = arglat
        #     mass_wet=1000 * u.kg, dry_mass=500 * u.kg, isp=300 * u.s,
        #     epoch=self.epoch, mu=EARTH_MU * (u.km**3 / u.s**2)
        # )
        
        final_radius = 45000 * u.km
        # 使用满足条件的卫星进行规划
        plan = planner.plan_hohmann_transfer(self.hohmann_sat, final_radius)
        
        # 执行计划
        final_satellite = self._apply_maneuver_plan(self.hohmann_sat, plan)
        
        # 断言：最终轨道半长轴和偏心率是否符合预期 (物理后果)
        final_a = final_satellite.orbit.a.to(u.km)
        final_ecc = final_satellite.orbit.ecc.value
        
        # 现在，误差应该在我们的容忍范围之内
        self.assertAlmostEqual(final_a.value, final_radius.value, delta=1.0)
        self.assertAlmostEqual(final_ecc, 0, places=3)
    def test_plan_relative_velocity_null_burn_consequence(self):
        """测试后果：速度调零后，相对速度是否真的为零。"""
        plan = planner.plan_relative_velocity_null_burn(self.satellite_A, self.satellite_B)

        # 执行计划（这是一个瞬时机动）
        final_satellite_A = self._apply_maneuver_plan(self.satellite_A, plan)

        # 断言：最终相对速度大小是否接近于零 (物理后果)
        relative_v = final_satellite_A.orbit.v - self.satellite_B.orbit.v
        relative_v_mag = np.linalg.norm(relative_v.to_value(u.km/u.s))
        self.assertAlmostEqual(relative_v_mag, 0.0, places=6)

    def test_plan_formation_injection_consequence(self):
        """测试后果：构型注入后，相对状态是否符合预期。"""
        
        # 修正1：为兰伯特问题提供一个更合理的、更容易求解的飞行时间
        tof = TimeDelta(6 * u.h)  # 将1小时改为6小时
        
        formation_id = "teardrop_01"
        plan = planner.plan_formation_injection(self.satellite_A, self.satellite_B, formation_id, tof)

        # 修正2：在尝试使用 plan 之前，必须先检查它是否生成成功！
        self.assertIsNotNone(plan, f"构型注入计划生成失败，可能是因为 tof={tof} 对于GEO轨道来说太短。")

        # 只有当 plan 不是 None 时，后续代码才有意义
        final_satellite_A = self._apply_maneuver_plan(self.satellite_A, plan)
        # ... 后续的断言代码不变 ...
        # final_target_orbit = self.satellite_B.orbit.propagate(tof)
        final_target_orbit = propagator.propagate_orbit_with_j2(self.satellite_B.orbit, tof)
        
        r_target = final_target_orbit.r
        v_target = final_target_orbit.v
        
        r_hat = r_target / np.linalg.norm(r_target)
        h_vec = np.cross(r_target, v_target)
        h_hat = h_vec / np.linalg.norm(h_vec)
        y_hat = np.cross(h_hat, r_hat)
        
        dcm = np.array([r_hat.value, y_hat.value, h_hat.value])
        
        dr_icrf = final_satellite_A.orbit.r - r_target
        dr_lvlh_actual = (dcm @ dr_icrf.to_value(u.km)) * u.km

        expected_dr_lvlh = FORMATION_CATALOG[formation_id].dr_lvlh
        
        self.assertTrue(np.allclose(
            dr_lvlh_actual.value,
            expected_dr_lvlh.to_value(u.km),
            atol=1.0 
        ), "构型注入后的相对位置与期望不符")


if __name__ == '__main__':
    unittest.main()