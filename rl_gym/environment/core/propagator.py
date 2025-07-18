# environment/core/propagator.py (基于您提供的源代码的最终版)

import numpy as np
from astropy import units as u
from astropy.time import TimeDelta

from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.core.perturbations import J2_perturbation
from poliastro.core.propagation import cowell

def _total_perturbation_accel(t, state, k):
    """计算总加速度（二体引力 + J2摄动）。这个辅助函数的逻辑是正确的。"""
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
    """使用考虑 J2 摄动的 Cowell 方法传播轨道。"""
    # 步骤 1: 拆解轨道对象
    k = orbit.attractor.k.to_value(u.km**3 / u.s**2)
    r0 = orbit.r.to_value(u.km)
    v0 = orbit.v.to_value(u.km / u.s)
    
    tofs = np.array([0.0, time_delta.to_value(u.s)])
    
    # 步骤 2: 严格按照源代码的函数签名进行调用
    # 接收返回的两个列表 rrs 和 vvs
    rrs, vvs = cowell(
        k,
        r0,
        v0,
        tofs,
        f=_total_perturbation_accel
    )

    # 步骤 3: 从返回的列表中获取最后一个时间点的状态
    final_r = rrs[-1] * u.km
    final_v = vvs[-1] * u.km / u.s
    
    # 步骤 4: 重组为新的 Orbit 对象
    new_orbit = Orbit.from_vectors(
        attractor=orbit.attractor, r=final_r, v=final_v, epoch=orbit.epoch + time_delta
    )
    return new_orbit