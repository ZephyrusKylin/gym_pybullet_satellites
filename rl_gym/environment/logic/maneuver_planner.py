
from dataclasses import dataclass
from typing import List, Dict

import numpy as np

from astropy import units as u
from astropy.time import Time, TimeDelta

from poliastro.maneuver import Maneuver
from poliastro.iod import lambert
from poliastro.twobody import Orbit  # 导入Orbit用于创建理想轨道

from environment.core.satellite import Satellite
from environment.core import propagator
from environment.core.constants import EARTH_RADIUS, MAX_REASONABLE_DELTA_V
# --- 输出数据结构定义 (无变化) ---
@dataclass
class ManeuverExecution:
    delta_v: u.Quantity
    execution_time: Time

@dataclass
class ManeuverPlan:
    maneuvers: List[ManeuverExecution]
    total_delta_v: u.Quantity
    total_time: TimeDelta

@dataclass
class StandardFormation:
    id: str
    description: str
    dr_lvlh: u.Quantity
    dv_lvlh: u.Quantity

# --- 标准构型目录 (无变化) ---
FORMATION_CATALOG: Dict[str, StandardFormation] = {
    "teardrop_01": StandardFormation(
        id="teardrop_01",
        description="一个沿着V-bar漂移的小型水滴构型",
        dr_lvlh=np.array([0, -10, 0]) * u.km,
        dv_lvlh=np.array([0.00001, 0.00002, 0]) * u.km / u.s,
    ),
    "figure8_01": StandardFormation(
        id="figure8_01",
        description="一个垂直于轨道平面的小型8字形悬停构型",
        dr_lvlh=np.array([0, -5, 0]) * u.km,
        dv_lvlh=np.array([0, 0, 0.00005]) * u.km / u.s,
    ),
}

def plan_lambert_intercept(
    interceptor: Satellite,
    target: Satellite,
    time_of_flight: TimeDelta
) -> ManeuverPlan:
    """
    规划一次基于兰伯特问题的拦截机动。
    """
    initial_orbit_interceptor = interceptor.orbit
    initial_orbit_target = target.orbit
    k = initial_orbit_interceptor.attractor.k

    final_orbit_target = propagator.propagate_orbit_with_j2(target.orbit, time_of_flight)
    r_final = final_orbit_target.r

    try:
        # -- FIX START --
        # 原代码: (v_initial_transfer, v_final_transfer), = lambert(...)
        # 修正: 使用直接的元组解包，以匹配 lambert 函数 `return v0, v` 的签名。
        v_initial_transfer, v_final_transfer = lambert(k, initial_orbit_interceptor.r, r_final, time_of_flight)
        # -- FIX END --

    except ValueError as e:
        # 这个捕获块现在用于处理 lambert 内部真正的无解情况。
        print(f"兰伯特问题无解: {e}")
        return None

    # 计算并封装机动方案 (这部分逻辑不变)
    dv1 = v_initial_transfer - initial_orbit_interceptor.v
    t1 = initial_orbit_interceptor.epoch
    maneuver1 = ManeuverExecution(delta_v=dv1, execution_time=t1)

    dv2 = final_orbit_target.v - v_final_transfer
    t2 = t1 + time_of_flight
    maneuver2 = ManeuverExecution(delta_v=dv2, execution_time=t2)

    # -- FIX START: 调整计算顺序 --
    # 1. 计算出总速度增量的纯数值
    total_dv_value = np.linalg.norm(dv1.to_value(u.km/u.s)) + np.linalg.norm(dv2.to_value(u.km/u.s))
    
    # 2. 立刻为它赋予单位，让它变回一个真正的物理量
    total_dv = total_dv_value * (u.km / u.s)
    
    # 3. 现在，在两个单位一致的物理量之间进行比较
    if total_dv > MAX_REASONABLE_DELTA_V:
        # print(f"规划结果被拒绝...")
        return None
    plan = ManeuverPlan(
        maneuvers=[maneuver1, maneuver2],
        total_delta_v=total_dv * (u.km/u.s),
        total_time=time_of_flight
    )
    return plan



def plan_hohmann_transfer(satellite: Satellite, final_radius: u.Quantity) -> ManeuverPlan | None:
    """
    规划一次经典的霍曼转移。
    """
    initial_orbit = satellite.orbit
    
    # 使用 try...except 来捕获 poliastro 可能抛出的明确错误
    try:
        maneuver = Maneuver.hohmann(initial_orbit, final_radius)
        (t_drift, dv1), (t_transfer, dv2) = maneuver.impulses
    except ValueError as e:
        # Poliastro 在无法执行时通常会抛出 ValueError，例如轨道不相交
        print(f"警告 (Hohmann): 无法规划霍曼转移, {e}")
        return None

    # 核心修正：增加对 poliastro 返回值的 NaN 检查
    # 这能捕获 poliastro 未作为 ValueError 抛出的数值不稳定问题
    if np.isnan(t_drift.value) or np.isnan(t_transfer.value):
        print(f"警告 (Hohmann): Poliastro 计算返回了 NaN 时间值，规划失败。")
        return None
    
    # 修正：确保第二次机动的执行时间逻辑正确
    exec_t1 = initial_orbit.epoch + t_drift
    exec_t2 = exec_t1 + t_transfer
    
    plan = ManeuverPlan(
        maneuvers=[
            ManeuverExecution(delta_v=dv1, execution_time=exec_t1),
            ManeuverExecution(delta_v=dv2, execution_time=exec_t2)
        ],
        total_delta_v=maneuver.get_total_cost(),
        total_time=t_drift + t_transfer
    )

    # 最终的“出厂质检”，确保万无一失
    if not np.isfinite(plan.maneuvers[0].execution_time.jd) or \
       not np.isfinite(plan.maneuvers[1].execution_time.jd):
        print(f"警告 (Hohmann): 最终计划中包含无效的执行时间，拒绝该计划。")
        return None
        
    return plan

def plan_relative_velocity_null_burn(satellite: Satellite, target: Satellite) -> ManeuverPlan:
    if satellite.epoch != target.epoch:
        raise ValueError("卫星和目标的历元必须相同才能计算相对速度。")
    relative_v = satellite.orbit.v - target.orbit.v
    delta_v = -relative_v
    return ManeuverPlan(
        maneuvers=[ManeuverExecution(delta_v=delta_v, execution_time=satellite.epoch)],
        total_delta_v=np.linalg.norm(delta_v.to_value(u.km/u.s)) * (u.km/u.s),
        total_time=0 * u.s
    )

# ... (其他函数 plan_circular_fly_around 和 plan_vbar_rendezvous 未修改) ...
def plan_circular_fly_around(satellite: Satellite, target: Satellite, radius: u.Quantity) -> ManeuverPlan:
    if satellite.epoch != target.epoch:
        raise ValueError("卫星和目标的历元必须相同。")
    orbit_target = target.orbit
    n = orbit_target.n.to_value(u.rad / u.s)
    delta_v_magnitude = n * radius.to_value(u.m) * (u.m / u.s)
    r_hat = orbit_target.r / np.linalg.norm(orbit_target.r)
    delta_v_vec = delta_v_magnitude.to(u.km / u.s) * r_hat.value
    return ManeuverPlan(
        maneuvers=[ManeuverExecution(delta_v=delta_v_vec, execution_time=satellite.epoch)],
        total_delta_v=np.linalg.norm(delta_v_vec.to_value(u.km / u.s)) * (u.km / u.s),
        total_time=0 * u.s
    )

def plan_vbar_rendezvous(satellite: Satellite, target: Satellite, distance: u.Quantity) -> ManeuverPlan:
    """
    规划一次 V-bar 自然漂移抵近（霍普曼交会）。
    """
    orbit_target = target.orbit
    n = orbit_target.n.to_value(u.rad / u.s)
    a = orbit_target.a.to_value(u.km)
    
    # 1. 计算转移轨道的半长轴所需的中间值
    base_for_a_transfer = a**(3/2) - (3 * n * distance.to_value(u.km)) / (8 * np.pi)
    
    # --- 新增防御性检查 ---
    if base_for_a_transfer < 0:
        print(f"警告 (V-bar Rendezvous): 无法规划，所需转移轨道无物理意义。")
        return None
    # --- 检查结束 ---
    
    a_transfer = base_for_a_transfer**(2/3)
    
    # --- 新增对计算结果的检查 ---
    # 即使基底为正，后续计算也可能因浮点误差产生问题
    if np.isnan(a_transfer):
        print(f"警告 (V-bar Rendezvous): 转移轨道半长轴计算结果为 NaN。")
        return None
    # --- 检查结束 ---

    # 2. 计算第一次脉冲 (进入转移轨道)
    # ... 后续代码不变 ...
    v_target = np.sqrt(target.orbit.attractor.k.to_value(u.km**2/u.s**2) / a)
    # 保护性地检查除数是否为零或接近零
    if abs(a + a_transfer) < 1e-6:
        return None
    v_transfer_start = np.sqrt(target.orbit.attractor.k.to_value(u.km**2/u.s**2) * (2/a - 2/(a + a_transfer)))
    
    # 同样，检查计算结果
    if np.isnan(v_transfer_start):
        print(f"警告 (V-bar Rendezvous): 转移轨道速度计算结果为 NaN。")
        return None
    
    dv1_mag = v_transfer_start - v_target
    v_hat = orbit_target.v / np.linalg.norm(orbit_target.v)
    dv1_vec = dv1_mag * v_hat.value * (u.km / u.s)
    
    dv2_vec = -dv1_vec
    
    # 3. 计算飞行时间
    # 原始公式 time_of_flight = TimeDelta(np.pi / n, format='sec') 依赖于简化模型
    # 一个更稳健的方式是基于轨道周期
    k_val = target.orbit.attractor.k.to_value(u.km**3 / u.s**2)
    T_transfer = np.pi * np.sqrt(a_transfer**3 / k_val)
    if np.isnan(T_transfer):
        print(f"警告 (V-bar Rendezvous): 飞行时间计算结果为 NaN。")
        return None

    time_of_flight = TimeDelta(T_transfer, format='sec')
    
    exec_t1 = satellite.epoch
    exec_t2 = satellite.epoch + time_of_flight
    
    plan = ManeuverPlan(
        maneuvers=[
            ManeuverExecution(delta_v=dv1_vec, execution_time=exec_t1),
            ManeuverExecution(delta_v=dv2_vec, execution_time=exec_t2)
        ],
        total_delta_v=(abs(dv1_mag) * 2) * (u.km/u.s),
        total_time=time_of_flight
    )
    return plan

def plan_formation_injection(
    satellite: Satellite,
    target: Satellite,
    formation_id: str,
    time_of_flight: TimeDelta
) -> ManeuverPlan | None:
    """
    规划一次机动，将卫星注入到一个预设的标准构型中。(V3 - 最终修正版)
    """
    if formation_id not in FORMATION_CATALOG:
        raise ValueError(f"构型ID '{formation_id}' 不在目录中。")

    formation = FORMATION_CATALOG[formation_id]
    k = satellite.orbit.attractor.k

    # 1. 获取目标卫星在注入时刻的状态 (逻辑正确)
    # target_orbit_at_injection = target.orbit.propagate(time_of_flight)
    target_orbit_at_injection = propagator.propagate_orbit_with_j2(target.orbit, time_of_flight)
    # --- 升级结束 ---
    r_target_vec = target_orbit_at_injection.r
    v_target_vec = target_orbit_at_injection.v

    # 2. 正确构建 LVLH 坐标系的正交基 (逻辑正确)
    r_hat = r_target_vec / np.linalg.norm(r_target_vec)
    h_vec = np.cross(r_target_vec, v_target_vec)
    h_hat = h_vec / np.linalg.norm(h_vec)
    y_hat = np.cross(h_hat, r_hat)
    rotation_matrix = np.array([r_hat.value, y_hat.value, h_hat.value]).T

    # 3. 将相对位置从 LVLH 变换到 ICRF (逻辑正确)
    dr_icrf = (rotation_matrix @ formation.dr_lvlh.to_value(u.km)) * u.km
    r_injection = r_target_vec + dr_icrf

    # 4. 正确计算注入点的绝对速度 (逻辑正确)
    omega_vec = h_vec / (np.linalg.norm(r_target_vec)**2)
    v_transport = np.cross(omega_vec, dr_icrf)
    dv_rotated = (rotation_matrix @ formation.dv_lvlh.to_value(u.km / u.s)) * u.km / u.s
    v_injection = v_target_vec + v_transport + dv_rotated

    # 5. 使用【正确语法】调用兰伯特求解器
    try:
        # 修正：使用直接的元组解包，这才是正确的语法！
        v_initial_transfer, v_final_transfer = lambert(
            k, satellite.orbit.r, r_injection, time_of_flight
        )
    except ValueError as e:
        # 现在，这里只捕获来自物理计算本身的、真正的无解异常
        # print(f"无法规划注入轨道，兰伯特问题无解: {e}")
        return None

    # 6. 计算注入机动的两次脉冲
    dv1 = v_initial_transfer - satellite.orbit.v
    dv2 = v_injection - v_final_transfer

    # 7. 封装一个包含两次完整机动的计划
    plan = ManeuverPlan(
        maneuvers=[
            ManeuverExecution(delta_v=dv1, execution_time=satellite.epoch),
            ManeuverExecution(delta_v=dv2, execution_time=satellite.epoch + time_of_flight)
        ],
        total_delta_v=np.linalg.norm(dv1) + np.linalg.norm(dv2),
        total_time=time_of_flight
    )

    # 8. 进行最终的“出厂质检”，确保没有 NaN 值
    for m in plan.maneuvers:
        if not np.isfinite(m.execution_time.jd) or np.any(np.isnan(m.delta_v.value)):
            return None
            
    return plan