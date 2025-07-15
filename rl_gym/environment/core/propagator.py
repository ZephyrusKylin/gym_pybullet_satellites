# # -*- coding: utf-8 -*-

# """
# environment/core/propagator.py

# 本文件定义了轨道传播器功能。

# 这是物理引擎（第一层）的一部分，专门负责计算卫星在给定的时间段后
# 的轨道状态。它将复杂的数值积分和摄动模型封装起来，为上层逻辑提供
# 一个简洁的接口，核心是加入了 J2 摄动以提高仿真的真实性。
# """

# import numpy as np
# from astropy import units as u
# from astropy.time import TimeDelta

# from poliastro.bodies import Earth
# from poliastro.twobody import Orbit
# from poliastro.core.perturbations import J2_perturbation as j2_core_accel


# def _total_perturbation_accel(t, state, k):
#     """
#     计算总加速度（二体引力 + J2摄动）。
    
#     这是一个内部辅助函数，将被传递给 Cowell 数值积分器。

#     Args:
#         t (float): 当前时间（从积分开始时算起），单位是秒。
#         state (np.ndarray): 卫星的六维状态向量 [x, y, z, vx, vy, vz]，单位是 km 和 km/s。
#         k (float): 中心天体的标准引力参数 (GM)，单位是 km^3/s^2。

#     Returns:
#         np.ndarray: 六维的状态向量导数 [vx, vy, vz, ax, ay, az]。
#     """
#     # 初始化状态向量的导数
#     du_dt = np.zeros_like(state)
    
#     # 速度部分：dx/dt = vx, dy/dt = vy, dz/dt = vz
#     du_dt[:3] = state[3:]
    
#     # --- 计算加速度部分 ---
#     r_vec = state[:3]
#     norm_r = np.linalg.norm(r_vec)
    
#     # 1. 中心天体引力（二体模型）
#     a_grav = -k * r_vec / norm_r**3
    
#     # 2. J2 摄动加速度
#     # Poliastro 的 j2_core_accel 需要地球的 J2 系数和半径作为额外参数
#     a_j2 = j2_core_accel(t, state, k, J2=Earth.J2.value, R=Earth.R.to(u.km).value)
    
#     # 3. 将所有加速度相加
#     total_accel = a_grav + a_j2
#     du_dt[3:] = total_accel
    
#     return du_dt


# def propagate_orbit_with_j2(orbit: Orbit, time_delta: TimeDelta) -> Orbit:
#     """
#     使用考虑 J2 摄动的 Cowell 方法传播轨道。

#     这是本模块对外提供的核心接口。

#     Args:
#         orbit (Orbit): 需要被传播的初始轨道。
#         time_delta (TimeDelta): 需要传播的时间长度。

#     Returns:
#         Orbit: 传播之后的新轨道对象。
#     """
#     if not isinstance(orbit, Orbit):
#         raise TypeError("输入 'orbit' 必须是 poliastro.twobody.Orbit 类型。")
#     if not isinstance(time_delta, TimeDelta):
#         raise TypeError("输入 'time_delta' 必须是 astropy.time.TimeDelta 类型。")

#     # 使用 Cowell 方法进行高精度数值积分
#     # f 参数接收一个自定义的加速度函数，我们将我们定义的包含 J2 摄动的函数传入
#     new_orbit = orbit.propagate(
#         time_delta,
#         method="cowell",
#         f=_total_perturbation_accel
#     )
    
#     return new_orbit

# environment/core/propagator.py (最终修正版 II for poliastro==0.17.0)

import numpy as np
from astropy import units as u
from astropy.time import TimeDelta

from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.core.perturbations import J2_perturbation

# 修正点 1: 导入底层的 cowell 积分器
from poliastro.core.propagation import cowell


def _total_perturbation_accel(t, state, k):
    """
    计算总加速度（二体引力 + J2摄动）。
    这个函数保持不变，它的逻辑是正确的。
    """
    du_dt = np.zeros_like(state)
    du_dt[:3] = state[3:]
    
    r_vec = state[:3]
    norm_r = np.linalg.norm(r_vec)
    
    a_grav = -k * r_vec / norm_r**3
    a_j2 = J2_perturbation(t, state, k, J2=Earth.J2.value, R=Earth.R.to(u.km).value)
    
    total_accel = a_grav + a_j2
    du_dt[3:] = total_accel
    
    return du_dt


def propagate_orbit_with_j2(orbit: Orbit, time_delta: TimeDelta) -> Orbit:
    """
    使用考虑 J2 摄动的 Cowell 方法传播轨道。
    修正点 2: 重写整个函数的实现逻辑，以适配 v0.17.0。
    """
    if not isinstance(orbit, Orbit):
        raise TypeError("输入 'orbit' 必须是 poliastro.twobody.Orbit 类型。")
    if not isinstance(time_delta, TimeDelta):
        raise TypeError("输入 'time_delta' 必须是 astropy.time.TimeDelta 类型。")

    # 步骤 1: 拆解轨道对象，获取底层计算所需的参数
    k = orbit.attractor.k.to_value(u.km**3 / u.s**2)
    r0 = orbit.r.to_value(u.km)
    v0 = orbit.v.to_value(u.km / u.s)
    
    # 步骤 2: 直接调用 cowell 积分器
    # tofs 是 time of flight a-rray，我们需要的是从0到最终时间的积分结果
    tofs = np.array([0.0, time_delta.to_value(u.s)]) * u.s
    
    # cowell 函数会返回所有时间点的位置和速度
    rr, vv = cowell(
        k=k,
        r0=r0,
        v0=v0,
        tofs=tofs,
        f=_total_perturbation_accel
    )

    # 步骤 3: 重组轨道对象
    # 我们需要的是最后一个时间点的结果
    final_r = rr[-1] * u.km
    final_v = vv[-1] * u.km / u.s
    
    # 使用 from_vectors 方法，根据最终状态向量创建一个新的 Orbit 对象
    new_orbit = Orbit.from_vectors(
        attractor=orbit.attractor,
        r=final_r,
        v=final_v,
        epoch=orbit.epoch + time_delta # 更新历元
    )
    
    return new_orbit