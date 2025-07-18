# tests/core/test_propagator.py

import unittest
import numpy as np
from astropy import units as u
from astropy.time import Time, TimeDelta

from poliastro.bodies import Earth
from poliastro.twobody import Orbit

# 从我们项目的正确路径导入待测试的函数
# 这份测试代码假定你的项目结构是正确的
from environment.core.propagator import propagate_orbit_with_j2


class TestPoliastroPropagator(unittest.TestCase):
    """针对基于 Poliastro 的 propagator.py 的单元测试集"""

    def setUp(self):
        """设置一个标准的近地轨道（LEO）用于测试，J2摄动在近地轨道更显著。"""
        self.initial_epoch = Time("2025-01-01 12:00:00", scale="utc")
        # 国际空间站轨道高度近似值
        self.leo_orbit = Orbit.from_classical(
            attractor=Earth,
            a=Earth.R + 420 * u.km,
            ecc=0.0001 * u.one,
            inc=51.6 * u.deg,
            raan=10 * u.deg,
            argp=20 * u.deg,
            nu=30 * u.deg,
            epoch=self.initial_epoch,
        )

    def test_basic_propagation(self):
        """测试：轨道被正确推进，且状态发生改变。"""
        time_delta = TimeDelta(3600 * u.s)  # 传播 1 小时
        new_orbit = propagate_orbit_with_j2(self.leo_orbit, time_delta)

        # 验证返回类型
        self.assertIsInstance(new_orbit, Orbit)
        # 验证历元更新
        self.assertEqual(new_orbit.epoch, self.leo_orbit.epoch + time_delta)
        # 验证位置矢量已改变
        self.assertFalse(np.allclose(
            new_orbit.r.to_value(u.km), 
            self.leo_orbit.r.to_value(u.km)
        ))

    def test_j2_effect_is_present(self):
        """关键测试：对比纯二体传播，验证J2摄动效果。"""
        time_delta = TimeDelta(1 * u.day)
        
        # 方法一：我们带J2摄动的函数
        orbit_with_j2 = propagate_orbit_with_j2(self.leo_orbit, time_delta)
        
        # 方法二：Poliastro原生的纯二体模型传播
        orbit_two_body = self.leo_orbit.propagate(time_delta)

        # 获取最终位置矢量
        r_j2 = orbit_with_j2.r.to_value(u.km)
        r_two_body = orbit_two_body.r.to_value(u.km)

        # 断言两个结果不相等
        self.assertFalse(np.allclose(r_j2, r_two_body))

        # 计算两个位置矢量之间的距离，应该显著大于一个较小的阈值（如1km）
        distance = np.linalg.norm(r_j2 - r_two_body)
        self.assertGreater(distance, 1.0, "J2摄动效果不明显，最终位置与二体模型过于接近！")

    def test_zero_time_propagation(self):
        """边界测试：传播时间为零，轨道状态不变。"""
        time_delta = TimeDelta(0 * u.s)
        new_orbit = propagate_orbit_with_j2(self.leo_orbit, time_delta)

        # 验证位置和速度矢量完全没变
        self.assertTrue(np.allclose(
            new_orbit.r.to_value(u.km), 
            self.leo_orbit.r.to_value(u.km)
        ))
        self.assertTrue(np.allclose(
            new_orbit.v.to_value(u.km/u.s), 
            self.leo_orbit.v.to_value(u.km/u.s)
        ))

    def test_long_term_semi_major_axis_stability(self):
        """健康检查：长期传播后半长轴（能量）保持基本稳定。"""
        # 传播约10个轨道周期
        ten_periods = self.leo_orbit.period.to(u.s) * 10
        time_delta = TimeDelta(ten_periods)
        
        new_orbit = propagate_orbit_with_j2(self.leo_orbit, time_delta)

        initial_a = self.leo_orbit.a.to_value(u.km)
        final_a = new_orbit.a.to_value(u.km)
        
        # 半长轴的变化应该非常小，这里我们设置容忍误差为1公里
        self.assertAlmostEqual(
            initial_a, 
            final_a, 
            delta=1.0, 
            msg="半长轴在长期传播后变化过大，积分可能不稳定！"
        )


if __name__ == '__main__':
    # 在项目根目录使用 'python -m unittest tests/core/test_propagator.py' 来运行
    unittest.main()