# tests/engine/test_physics.py

import pytest
import numpy as np
import astropy.units as u
from astropy.units import Quantity

from engine.physics import RelativePhysics

# 辅助函数，用于比较带单位的 numpy 数组
def assert_quantity_allclose(q1, q2, **kwargs):
    """
    比较两个 Quantity 对象的值和单位是否都接近。
    """
    assert q1.unit.is_equivalent(q2.unit), f"Unit mismatch: {q1.unit} vs {q2.unit}"
    np.testing.assert_allclose(q1.to_value(q2.unit), q2.to_value(q2.unit), **kwargs)


class TestRelativePhysics:
    """
    针对 RelativePhysics 类的单元测试套件。
    """

    # === 测试 propagate 方法 ===
    
    def test_propagate_normal_case(self):
        """
        测试：标准情况下的惯性传播。
        """
        pos = np.array([0., 100., 0.]) * u.km
        vel = np.array([10., 0., -5.]) * u.m / u.s
        dt = 100 * u.s
        
        new_pos = RelativePhysics.propagate(pos, vel, dt)
        
        # 10 m/s * 100 s = 1000 m = 1 km
        # -5 m/s * 100 s = -500 m = -0.5 km
        expected_pos = np.array([1., 100., -0.5]) * u.km
        
        assert_quantity_allclose(new_pos, expected_pos)

    def test_propagate_zero_dt(self):
        """
        测试：时间增量为零，位置应不变。
        """
        pos = np.array([10., 20., 30.]) * u.m
        vel = np.array([1., 1., 1.]) * u.m / u.s
        dt = 0 * u.s
        
        new_pos = RelativePhysics.propagate(pos, vel, dt)
        assert_quantity_allclose(new_pos, pos)

    def test_propagate_zero_velocity(self):
        """
        测试：速度为零，位置应不变。
        """
        pos = np.array([10., 20., 30.]) * u.m
        vel = np.array([0., 0., 0.]) * u.m / u.s
        dt = 1000 * u.s

        new_pos = RelativePhysics.propagate(pos, vel, dt)
        assert_quantity_allclose(new_pos, pos)


    # === 测试 apply_thrust 方法 ===

    def test_apply_thrust_normal_case(self):
        """
        测试：标准情况下的施加推力。
        """
        vel = np.array([10., 0., 0.]) * u.m / u.s
        mass = 1000 * u.kg
        thrust = np.array([0., 100., 0.]) * u.N # 100牛顿的Y轴推力
        dt = 10 * u.s
        
        new_vel = RelativePhysics.apply_thrust(vel, mass, thrust, dt)
        
        # a = F/m = 100 N / 1000 kg = 0.1 m/s^2
        # delta_v = a * t = 0.1 m/s^2 * 10 s = 1 m/s
        expected_vel = np.array([10., 1., 0.]) * u.m / u.s
        
        assert_quantity_allclose(new_vel, expected_vel, rtol=1e-6)
    
    def test_apply_thrust_error_on_invalid_mass(self):
        """
        测试：当质量为零或负数时，应抛出异常。
        """
        vel = np.array([0., 0., 0.]) * u.m / u.s
        thrust = np.array([10., 0., 0.]) * u.N
        dt = 1 * u.s
        
        # astropy 在除以零质量时会抛出 ZeroDivisionError
        with pytest.raises(ZeroDivisionError):
            RelativePhysics.apply_thrust(vel, 0 * u.kg, thrust, dt)
        
        # 在我们的实现中，质量为负数在 Quantity 运算中可能不会直接报错，
        # 但在物理上是无意义的。引擎上层或单元构造函数应阻止这种情况。


    # === 测试 estimate_maneuver 方法 ===

    def test_estimate_maneuver_from_rest_to_target(self):
        """
        测试：从静止到目标的机动估算。
        """
        pos = np.array([0., 0., 0.]) * u.m
        target_pos = np.array([1000., 0., 0.]) * u.m
        vel = np.array([0., 0., 0.]) * u.m / u.s
        mass = 500 * u.kg
        max_thrust = 100 * u.N # a_max = 0.2 m/s^2
        
        time, dv = RelativePhysics.estimate_maneuver(pos, target_pos, vel, max_thrust, mass)

        # 理论计算
        # Braking phase: time=0, dv=0
        # Transfer phase: dist=1000m, a=0.2m/s^2
        # time_transfer = 2 * sqrt(1000m / 0.2m/s^2) = 2 * sqrt(5000)s ~= 141.42s
        # dv_transfer = 0.2m/s^2 * 141.42s = 28.28 m/s
        expected_time = (2 * np.sqrt(1000 / 0.2)) * u.s
        expected_dv = (0.2 * expected_time.value) * u.m/u.s

        assert_quantity_allclose(time, expected_time)
        assert_quantity_allclose(dv, expected_dv)

    def test_estimate_maneuver_already_at_target_and_rest(self):
        """
        测试：当已经静止在目标位置时，成本应为零。
        """
        pos = np.array([100., 100., 100.]) * u.m
        vel = np.array([0., 0., 0.]) * u.m / u.s
        mass = 1000 * u.kg
        max_thrust = 100 * u.N

        time, dv = RelativePhysics.estimate_maneuver(pos, pos, vel, max_thrust, mass)

        assert_quantity_allclose(time, 0 * u.s, atol=1e-9 * u.s)
        assert_quantity_allclose(dv, 0 * u.m / u.s, atol=1e-9 * u.m/u.s)

    def test_estimate_maneuver_no_thrust_is_impossible(self):
        """
        测试：当没有推力时，机动时间应为无穷大。
        """
        pos = np.array([0., 0., 0.]) * u.m
        target_pos = np.array([1000., 0., 0.]) * u.m
        vel = np.array([0., 0., 0.]) * u.m / u.s
        mass = 500 * u.kg
        max_thrust = 0 * u.N

        time, dv = RelativePhysics.estimate_maneuver(pos, target_pos, vel, max_thrust, mass)

        assert time.value == np.inf
        assert dv.value == np.inf

    def test_estimate_maneuver_only_braking_needed(self):
        """
        测试：当已经在目标位置但有速度时，只需计算刹车成本。
        """
        pos = np.array([100., 100., 100.]) * u.m
        vel = np.array([10., -20., 0.]) * u.m / u.s
        mass = 1000 * u.kg
        max_thrust = 50 * u.N # a_max = 0.05 m/s^2

        time, dv = RelativePhysics.estimate_maneuver(pos, pos, vel, max_thrust, mass)
        
        vel_norm = np.linalg.norm(vel.value) * u.m / u.s
        expected_dv = vel_norm
        expected_time = (vel_norm / (max_thrust / mass)).to(u.s)

        assert_quantity_allclose(dv, expected_dv)
        assert_quantity_allclose(time, expected_time)


    # === 参数化测试，验证单位安全 ===
    
    @pytest.mark.parametrize("method_name, kwargs", [
        ("propagate", {"position": 1 * u.kg, "velocity": 1 * u.m/u.s, "dt": 1*u.s}),
        ("propagate", {"position": 1 * u.m, "velocity": 1 * u.kg, "dt": 1*u.s}),
        ("apply_thrust", {"velocity": 1 * u.m/u.s, "mass": 1 * u.s, "thrust_vector": 1*u.N, "dt": 1*u.s}),
        ("apply_thrust", {"velocity": 1 * u.m/u.s, "mass": 1 * u.kg, "thrust_vector": 1*u.m, "dt": 1*u.s}),
    ])
    def test_methods_raise_unit_conversion_error(self, method_name, kwargs):
        """
        测试：当向方法传入错误的物理单位时，应抛出 UnitConversionError。
        """
        method = getattr(RelativePhysics, method_name)
        with pytest.raises(u.UnitConversionError):
            method(**kwargs)