# environment/logic/tests/test_task_manager.py

import unittest
import numpy as np

from astropy import units as u
from astropy.time import Time, TimeDelta

from environment.core.constants import EARTH_MU
from environment.core.satellite import Satellite
import environment.logic.maneuver_planner as planner
from environment.logic.maneuver_planner import ManeuverPlan, ManeuverExecution, FORMATION_CATALOG
from environment.logic.TaskManager import TaskManager, TaskStatus
import environment.core.propagator as propagator
class TestTaskManager(unittest.TestCase):
    """
    针对 TaskManager 的单元测试套件。
    """

    def setUp(self):
        """
        在每个测试用例开始前，设置一个干净的环境。
        (修改：增加了一个目标卫星，供规划函数使用)
        """
        self.task_manager = TaskManager()
        
        self.epoch = Time("2025-07-20 00:00:00", scale="utc")
        
        # 我方卫星
        self.sat_agent = Satellite.from_classical(
            sat_id="agent_1",
            a=7000 * u.km, ecc=0.01 * u.one, inc=20 * u.deg,
            raan=0 * u.deg, argp=0 * u.deg, nu=0 * u.deg,
            mass_wet=1000 * u.kg, dry_mass=500 * u.kg, isp=300 * u.s,
            epoch=self.epoch,
            mu=EARTH_MU
        )
        
        # 目标卫星 (轨道略有不同)
        self.sat_target = Satellite.from_classical(
            sat_id="target_1",
            a=7100 * u.km, ecc=0.01 * u.one, inc=20.1 * u.deg,
            raan=0 * u.deg, argp=0 * u.deg, nu=20 * u.deg,
            mass_wet=1000 * u.kg, dry_mass=500 * u.kg, isp=300 * u.s,
            epoch=self.epoch,
            mu=EARTH_MU
        )
        
        self.satellites = {
            "agent_1": self.sat_agent,
            "target_1": self.sat_target
        }

    def test_01_add_cancel_and_status_check(self):
        """
        测试：任务的添加、查询、中断和状态检查基本流程。
        含义：验证任务生命周期中最基本的操作是否符合预期。
        """
        sat_id = "agent_1"
        
        # 1. 初始状态：卫星应该是空闲的
        self.assertFalse(self.task_manager.is_busy(sat_id))
        self.assertIsNone(self.task_manager.get_task_status(sat_id))

        # 2. 添加任务：创建一个简单的机动计划并添加
        plan = ManeuverPlan(
            maneuvers=[ManeuverExecution(delta_v=np.array([1,0,0])*u.km/u.s, execution_time=self.epoch)],
            total_delta_v=1*u.km/u.s, total_time=0*u.s
        )
        self.assertTrue(self.task_manager.add_task(sat_id, plan))
        
        # 3. 忙碌状态：添加后，卫星应为忙碌，状态为 PENDING
        self.assertTrue(self.task_manager.is_busy(sat_id))
        self.assertEqual(self.task_manager.get_task_status(sat_id), TaskStatus.PENDING)

        # 4. 中断任务：取消该任务
        self.assertTrue(self.task_manager.cancel_task(sat_id))
        
        # 5. 最终状态：取消后，卫星应再次变为空闲
        self.assertFalse(self.task_manager.is_busy(sat_id))
        self.assertIsNone(self.task_manager.get_task_status(sat_id))
        
        # 6. 边缘情况：取消一个不存在的任务应失败
        self.assertFalse(self.task_manager.cancel_task(sat_id))

    def test_02_add_task_force_overwrite(self):
        """
        测试：使用 force=True 强制覆盖一个已存在的任务。
        含义：验证任务抢占机制是否能正确替换旧任务。
        """
        sat_id = "agent_1"
        
        # --- 修改部分 START ---
        # 创建一个合法的、非空的机动，作为 plan1
        dummy_maneuver_1 = ManeuverExecution(
            delta_v=np.array([0.01, 0, 0]) * u.km / u.s,
            execution_time=self.epoch + TimeDelta(10, format='sec')
        )
        plan1 = ManeuverPlan([dummy_maneuver_1], total_delta_v=0.01*u.km/u.s, total_time=10*u.s)
        
        # 我们现在应该断言任务是成功添加的
        self.assertTrue(self.task_manager.add_task(sat_id, plan1), "添加第一个任务失败")
        # --- 修改部分 END ---
        
        # 确认旧任务已存在
        self.assertEqual(self.task_manager.active_tasks[sat_id].plan.total_delta_v, 0.01*u.km/u.s)
        
        # --- 修改部分 START ---
        # 同样，为 plan2 创建一个合法的机动
        dummy_maneuver_2 = ManeuverExecution(
            delta_v=np.array([0.05, 0, 0]) * u.km / u.s,
            execution_time=self.epoch + TimeDelta(20, format='sec')
        )
        plan2 = ManeuverPlan([dummy_maneuver_2], total_delta_v=0.05*u.km/u.s, total_time=20*u.s)
        # --- 修改部分 END ---
        
        # 默认添加会失败
        self.assertFalse(self.task_manager.add_task(sat_id, plan2, force=False))
        
        # 强制添加会成功
        self.assertTrue(self.task_manager.add_task(sat_id, plan2, force=True))
        
        # 验证新任务已成功覆盖旧任务
        self.assertTrue(self.task_manager.is_busy(sat_id))
        self.assertEqual(self.task_manager.active_tasks[sat_id].plan.total_delta_v, 0.05*u.km/u.s)
    
    def test_03_update_single_maneuver_execution(self):
        """
        测试：update 方法能否在正确的时间点执行一次单脉冲机动。
        含义：验证 TaskManager 的核心驱动逻辑，即时间判断与物理状态改变的联动。
        """
        sat_id = "agent_1"
        burn_time = self.epoch + TimeDelta(100, format='sec')
        delta_v_vec = np.array([0.1, 0, 0]) * u.km / u.s
        
        plan = ManeuverPlan(
            maneuvers=[ManeuverExecution(delta_v=delta_v_vec, execution_time=burn_time)],
            total_delta_v=0.1*u.km/u.s, total_time=0*u.s
        )
        self.task_manager.add_task(sat_id, plan)
        
        # 记录机动前的状态
        initial_mass = self.sat_agent.mass
        initial_v = self.sat_agent.orbit.v.copy()
        
        # 模拟时间流逝，直到机动执行时间点
        current_time = self.epoch
        dt = TimeDelta(60, format='sec')
        
        # 在机动前，卫星状态不应改变
        self.task_manager.update(self.satellites, current_time, dt)
        self.assertEqual(self.sat_agent.mass, initial_mass)

        # 到达机动时间点，执行 update
        current_time = self.epoch + TimeDelta(60, format='sec') # current_time = 60s, burn_time = 100s
        self.task_manager.update(self.satellites, current_time, dt) # 此时 60 <= 100 < 120，应执行
        
        # 验证机动后的状态
        self.assertTrue(self.sat_agent.mass < initial_mass) # 质量应减少
        self.assertFalse(np.array_equal(self.sat_agent.orbit.v, initial_v)) # 速度应改变
        self.assertFalse(self.task_manager.is_busy(sat_id)) # 任务完成后应变为空闲

    def test_04_update_multi_maneuver_and_status_flow(self):
        """
        测试：一个包含两次机动的任务（如霍曼转移）的状态流转。
        含义：验证任务状态机 (PENDING -> IN_PROGRESS -> COMPLETED) 是否正常工作。
        """
        sat_id = "agent_1"
        t1 = self.epoch + TimeDelta(100, format='sec')
        t2 = self.epoch + TimeDelta(500, format='sec')
        dv1 = np.array([0.1, 0, 0]) * u.km / u.s
        dv2 = np.array([0.1, 0, 0]) * u.km / u.s
        
        plan = ManeuverPlan(
            maneuvers=[
                ManeuverExecution(delta_v=dv1, execution_time=t1),
                ManeuverExecution(delta_v=dv2, execution_time=t2)
            ],
            total_delta_v=0.2*u.km/u.s, total_time=500*u.s
        )
        self.task_manager.add_task(sat_id, plan)
        self.assertEqual(self.task_manager.get_task_status(sat_id), TaskStatus.PENDING)

        # 模拟时间，执行第一次机动
        current_time = self.epoch + TimeDelta(60, format='sec')
        dt = TimeDelta(60, format='sec')
        self.task_manager.update(self.satellites, current_time, dt)
        
        # 验证第一次机动后
        self.assertEqual(self.task_manager.get_task_status(sat_id), TaskStatus.IN_PROGRESS)
        self.assertTrue(self.task_manager.is_busy(sat_id)) # 仍然忙碌
        mass_after_burn1 = self.sat_agent.mass

        # 模拟时间，执行第二次机动
        current_time = self.epoch + TimeDelta(480, format='sec')
        self.task_manager.update(self.satellites, current_time, dt) # 480 <= 500 < 540

        # 验证第二次机动后
        self.assertFalse(self.task_manager.is_busy(sat_id)) # 任务完成，变为空闲
        self.assertTrue(self.sat_agent.mass < mass_after_burn1)

    def test_05_update_task_failure_no_fuel(self):
        """
        测试：当卫星燃料不足时，任务是否会失败。
        含义：验证系统的容错能力和失败路径的处理。
        """
        sat_id = "agent_1"
        
        # 耗尽卫星燃料
        self.sat_agent.mass = self.sat_agent.dry_mass
        self.assertFalse(self.sat_agent.can_maneuver)

        burn_time = self.epoch + TimeDelta(100, format='sec')
        plan = ManeuverPlan(
            maneuvers=[ManeuverExecution(np.array([0.1,0,0])*u.km/u.s, burn_time)],
            total_delta_v=0.1*u.km/u.s, total_time=0*u.s
        )
        self.task_manager.add_task(sat_id, plan)

        initial_orbit = self.sat_agent.orbit

        # 模拟到执行时间
        current_time = self.epoch + TimeDelta(60, format='sec')
        dt = TimeDelta(60, format='sec')
        self.task_manager.update(self.satellites, current_time, dt)

        # 验证任务失败
        self.assertFalse(self.task_manager.is_busy(sat_id)) # 失败后应变为空闲
        # 轨道不应发生任何变化
        np.testing.assert_array_equal(self.sat_agent.orbit.r, initial_orbit.r)
        np.testing.assert_array_equal(self.sat_agent.orbit.v, initial_orbit.v)
    def _run_simulation_until_task_is_done(self, sat_id: str):
        """
        一个辅助函数，完整地模拟时间流逝和物理演化，直到任务完成。
        (V2 - 修正版: 推进世界中的所有物体)
        """
        current_time = self.epoch
        dt = TimeDelta(60, format='sec')
        max_steps = 2000 
        step = 0
        
        while self.task_manager.is_busy(sat_id) and step < max_steps:
            # 步骤 1: 更新任务逻辑
            self.task_manager.update(self.satellites, current_time, dt)
            
            # 步骤 2: 演化物理世界
            # 核心修正：遍历 self.satellites 字典中的每一个卫星，并推进其状态
            # 这确保了目标卫星和拦截卫星都在同一个物理世界中运动。
            for sat in self.satellites.values():
                # 为了验证 TaskManager 逻辑，我们使用与计划本身一致的开普勒传播
                new_orbit = sat.orbit.propagate(dt)
                sat.update_orbit(new_orbit)
            
            # 步骤 3: 推进时间
            current_time += dt
            step += 1
            
        self.assertFalse(self.task_manager.is_busy(sat_id), "任务未在预期的步数内完成")

    def test_06_execute_hohmann_transfer_plan(self):
        """
        测试：执行一次由 plan_hohmann_transfer 生成的霍曼转移计划。
        含义：验证 TaskManager 与经典的双脉冲、长周期转移计划的集成。
        """
        sat_id = "agent_1"
        initial_mass = self.sat_agent.mass
        
        # 使用规划器生成计划
        target_radius = self.sat_agent.orbit.a + 200 * u.km
        hohmann_plan = planner.plan_hohmann_transfer(self.sat_agent, target_radius)
        self.assertIsNotNone(hohmann_plan, "霍曼转移计划生成失败")
        
        # 添加并执行任务
        self.task_manager.add_task(sat_id, hohmann_plan)
        self._run_simulation_until_task_is_done(sat_id)
        
        # 验证
        self.assertTrue(self.sat_agent.mass < initial_mass)
        self.assertAlmostEqual(self.sat_agent.orbit.a.to_value(u.km), 
                               target_radius.to_value(u.km), 
                               places=0)

    def test_07_execute_lambert_intercept_plan(self):
        """
        测试：执行一次由 plan_lambert_intercept 生成的兰伯特拦截计划。
        含义：验证与另一个核心的双脉冲拦截算法的集成。
        """
        sat_id = "agent_1"
        initial_mass = self.sat_agent.mass
        initial_orbit_v = self.sat_agent.orbit.v.copy()

        # 生成计划
        tof = TimeDelta(30 * 60, format='sec') # 30分钟飞行时间
        lambert_plan = planner.plan_lambert_intercept(self.sat_agent, self.sat_target, tof)
        self.assertIsNotNone(lambert_plan, "兰伯特拦截计划生成失败")
        
        # 添加并执行
        self.task_manager.add_task(sat_id, lambert_plan)
        self._run_simulation_until_task_is_done(sat_id)
        
        # 验证
        self.assertTrue(self.sat_agent.mass < initial_mass)
        self.assertFalse(np.array_equal(self.sat_agent.orbit.v, initial_orbit_v))

    def test_08_execute_velocity_null_plan(self):
        """
        测试：执行一次 plan_relative_velocity_null_burn 生成的速度置零计划。
        含义：验证与近距离交会中关键的瞬时机动算法的集成。
        """
        sat_id = "agent_1"
        initial_mass = self.sat_agent.mass

        # 生成计划
        # 确保历元相同
        self.sat_agent.update_orbit(self.sat_agent.orbit.propagate(TimeDelta(1, 'sec')))
        self.sat_target.update_orbit(self.sat_target.orbit.propagate(TimeDelta(1, 'sec')))
        null_burn_plan = planner.plan_relative_velocity_null_burn(self.sat_agent, self.sat_target)
        self.assertIsNotNone(null_burn_plan)

        # 添加并执行 (这是一个瞬时机动，一次 update 就应该完成)
        self.task_manager.add_task(sat_id, null_burn_plan)
        self._run_simulation_until_task_is_done(sat_id)

        # 验证
        self.assertTrue(self.sat_agent.mass < initial_mass)
        # 验证相对速度是否接近于零
        relative_v = self.sat_agent.orbit.v - self.sat_target.orbit.v
        np.testing.assert_allclose(relative_v.to_value(u.km / u.s), 0, atol=1e-9)

    def test_09_execute_formation_injection_plan(self):
        """
        测试：执行一次 plan_formation_injection 生成的编队注入计划。
        (V4 - 最终修正版: 修正状态管理)
        """
        # --- 为本测试创建独立的、全新的卫星对象 ---
        agent_sat = Satellite.from_classical(
            sat_id="agent_1", a=20000 * u.km, ecc=0.001 * u.one, inc=55 * u.deg,
            raan=10 * u.deg, argp=20 * u.deg, nu=30 * u.deg,
            mass_wet=1000 * u.kg, dry_mass=500 * u.kg, isp=300 * u.s,
            epoch=self.epoch, mu=EARTH_MU
        )
        target_sat = Satellite.from_classical(
            sat_id="target_1", a=42164 * u.km, ecc=0.001 * u.one, inc=0.1 * u.deg,
            raan=10 * u.deg, argp=20 * u.deg, nu=31 * u.deg,
            mass_wet=1000 * u.kg, dry_mass=500 * u.kg, isp=300 * u.s,
            epoch=self.epoch, mu=EARTH_MU
        )
        
        # --- 核心修正：在模拟开始前，备份目标的【初始轨道】 ---
        initial_target_orbit_for_verification = target_sat.orbit
        # --- 修正结束 ---

        current_sats = {"agent_1": agent_sat, "target_1": target_sat}

        # 1. 生成J2感知的计划
        tof = TimeDelta(6 * u.h)
        plan = planner.plan_formation_injection(agent_sat, target_sat, "teardrop_01", tof)
        self.assertIsNotNone(plan, "编队注入计划生成失败")
        
        # 2. 在理想开普勒世界中执行计划
        # 这个模拟循环会修改 current_sats 中卫星的状态
        self.task_manager.add_task(agent_sat.sat_id, plan)
        current_time = self.epoch
        dt = TimeDelta(60, format='sec')
        while self.task_manager.is_busy(agent_sat.sat_id):
            self.task_manager.update(current_sats, current_time, dt)
            for sat in current_sats.values():
                sat.update_orbit(sat.orbit.propagate(dt))
            current_time += dt

        # 3. 验证
        final_agent_r = current_sats["agent_1"].orbit.r
        
        # --- 核心修正：使用【初始轨道备份】来进行最终的高保真传播 ---
        true_target_orbit = propagator.propagate_orbit_with_j2(initial_target_orbit_for_verification, tof)
        # --- 修正结束 ---

        # (后续的验证逻辑不变)
        r_target_vec = true_target_orbit.r
        v_target_vec = true_target_orbit.v
        # ... (计算 true_injection_point_r 的代码)
        r_hat = r_target_vec / np.linalg.norm(r_target_vec)
        h_vec = np.cross(r_target_vec, v_target_vec)
        h_hat = h_vec / np.linalg.norm(h_vec)
        y_hat = np.cross(h_hat, r_hat)
        rotation_matrix = np.array([r_hat.value, y_hat.value, h_hat.value]).T
        dr_icrf = (rotation_matrix @ FORMATION_CATALOG["teardrop_01"].dr_lvlh.to_value(u.km)) * u.km
        true_injection_point_r = r_target_vec + dr_icrf

        final_dist = np.linalg.norm((final_agent_r - true_injection_point_r).to_value(u.km))
        
        # 理想计划在现实执行后的偏差，应该在一个可控范围内
        self.assertLess(final_dist, 50, "理想计划在现实执行后的偏差过大")

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)