# # environment/logic/maneuver_planner.py

# """
# 轨道机动规划器。

# 本模块是 logic 层的“轨道计算器”，提供一系列静态函数，
# 用于根据不同的战术需求，计算出具体的轨道机动方案。
# 它大量使用 poliastro 库进行底层计算。
# """

# from dataclasses import dataclass
# from typing import List, Dict

# import numpy as np

# from astropy import units as u
# from astropy.time import Time, TimeDelta

# from poliastro.maneuver import Maneuver
# from poliastro.iod import lambert
# from poliastro.core.elements import rv2coe

# from environment.core.satellite import Satellite
# # 注意：我们这里不直接导入 Orbit，而是通过 Satellite 对象来获取轨道
# # 这是一个更上层的设计，规划器应该基于某个卫星的当前状态进行规划


# # --- 输出数据结构定义 ---

# @dataclass
# class ManeuverExecution:
#     """定义单次脉冲机动。"""
#     delta_v: u.Quantity  # 3D 速度增量矢量 (km/s)
#     execution_time: Time # 执行机动的绝对时间 (UTC)

# @dataclass
# class ManeuverPlan:
#     """定义一个完整的机动方案。"""
#     maneuvers: List[ManeuverExecution]
#     total_delta_v: u.Quantity  # 总速度增量大小 (km/s)
#     total_time: TimeDelta      # 总耗时

# @dataclass
# class StandardFormation:
#     """定义一个标准相对轨道构型的注入参数。"""
#     id: str
#     description: str
#     # 注入点在目标 LVLH 坐标系下的相对位置矢量 (dr)
#     dr_lvlh: u.Quantity 
#     # 注入点在目标 LVLH 坐标系下的相对速度矢量 (dv)
#     dv_lvlh: u.Quantity

# # --- 新增：标准构型目录 ---
# # 这些是预先计算好的、用于注入不同构型的相对状态矢量。
# # LVLH坐标系：x轴沿径向向外，y轴在轨道平面内、与速度方向同向，z轴为轨道法向。
# # 注意：这些数值是为演示目的而设置的示例值。
# FORMATION_CATALOG: Dict[str, StandardFormation] = {
#     "teardrop_01": StandardFormation(
#         id="teardrop_01",
#         description="一个沿着V-bar漂移的小型水滴构型",
#         dr_lvlh=np.array([0, -10, 0]) * u.km,
#         dv_lvlh=np.array([0.00001, 0.00002, 0]) * u.km / u.s,
#     ),
#     "figure8_01": StandardFormation(
#         id="figure8_01",
#         description="一个垂直于轨道平面的小型8字形悬停构型",
#         dr_lvlh=np.array([0, -5, 0]) * u.km,
#         dv_lvlh=np.array([0, 0, 0.00005]) * u.km / u.s,
#     ),
# }
# # --- 规划函数 ---

# def plan_lambert_intercept(
#     interceptor: Satellite, 
#     target: Satellite, 
#     time_of_flight: TimeDelta
# ) -> ManeuverPlan:
#     """
#     规划一次基于兰伯特问题的拦截机动。
#     在指定的时间后，让拦截器到达目标卫星的未来位置。
    
#     Args:
#         interceptor (Satellite): 执行拦截的卫星。
#         target (Satellite): 目标卫星。
#         time_of_flight (TimeDelta): 从开始机动到抵达目标位置的期望飞行时间。

#     Returns:
#         ManeuverPlan: 包含两次脉冲机动的拦截方案。
#     """
#     # 1. 获取拦截器和目标的初始轨道状态
#     initial_orbit_interceptor = interceptor.orbit
#     initial_orbit_target = target.orbit
#     k = initial_orbit_interceptor.attractor.k

#     # 2. 预测目标在 time_of_flight 之后的位置
#     final_orbit_target = initial_orbit_target.propagate(time_of_flight)
#     r_final = final_orbit_target.r

#     # 3. 求解兰伯特问题，得到转移轨道所需的初末速度
#     try:
#         # lambert 求解器返回的是速度矢量列表
#         (v_initial_transfer, v_final_transfer), = lambert(k, initial_orbit_interceptor.r, r_final, time_of_flight)
#     except ValueError as e:
#         # 如果在给定时间内无法到达，兰伯特问题无解
#         print(f"兰伯特问题无解: {e}")
#         return None

#     # 4. 计算第一次脉冲 (出发脉冲)
#     dv1 = v_initial_transfer - initial_orbit_interceptor.v
#     t1 = initial_orbit_interceptor.epoch
#     maneuver1 = ManeuverExecution(delta_v=dv1, execution_time=t1)

#     # 5. 计算第二次脉冲 (抵达时匹配目标速度，用于交会/伴飞)
#     dv2 = final_orbit_target.v - v_final_transfer
#     t2 = t1 + time_of_flight
#     maneuver2 = ManeuverExecution(delta_v=dv2, execution_time=t2)
    
#     # 6. 封装并返回机动方案
#     total_dv = np.linalg.norm(dv1.to_value(u.km/u.s)) + np.linalg.norm(dv2.to_value(u.km/u.s))
#     plan = ManeuverPlan(
#         maneuvers=[maneuver1, maneuver2],
#         total_delta_v=total_dv * (u.km/u.s),
#         total_time=time_of_flight
#     )
#     return plan

# def plan_hohmann_transfer(satellite: Satellite, final_radius: u.Quantity) -> ManeuverPlan:
#     """
#     规划一次经典的霍曼转移。
#     用于在两个共面圆轨道之间进行转移。
    
#     Args:
#         satellite (Satellite): 执行机动的卫星 (假设其当前为圆轨道)。
#         final_radius (u.Quantity): 目标圆轨道的半径。

#     Returns:
#         ManeuverPlan: 包含两次脉冲机动的霍曼转移方案。
#     """
#     initial_orbit = satellite.orbit
    
#     # 1. 使用 poliastro 直接计算霍曼转移
#     # 注意：poliastro 的 hohmann 方法假设初始轨道是圆或椭圆的近/远地点
#     maneuver = Maneuver.hohmann(initial_orbit, final_radius)
    
#     # 2. 提取机动信息
#     (t1, dv1), (t2, dv2) = maneuver.impulses
    
#     # 3. 封装并返回机动方案
#     exec_t1 = initial_orbit.epoch + t1
#     exec_t2 = initial_orbit.epoch + t2
    
#     plan = ManeuverPlan(
#         maneuvers=[
#             ManeuverExecution(delta_v=dv1, execution_time=exec_t1),
#             ManeuverExecution(delta_v=dv2, execution_time=exec_t2)
#         ],
#         total_delta_v=maneuver.get_total_cost(),
#         total_time=maneuver.get_total_time()
#     )
#     return plan

# def plan_relative_velocity_null_burn(satellite: Satellite, target: Satellite) -> ManeuverPlan:
#     """
#     规划一次速度调零机动，用于近距离交会。
#     计算一个瞬时脉冲，使得卫星相对目标的速度变为零。
    
#     Args:
#         satellite (Satellite): 执行机动的卫星。
#         target (Satellite): 目标卫星 (两者历元必须相同)。

#     Returns:
#         ManeuverPlan: 包含一次脉冲机动的方案。
#     """
#     if satellite.epoch != target.epoch:
#         raise ValueError("卫星和目标的历元必须相同才能计算相对速度。")
        
#     # 1. 计算相对速度
#     relative_v = satellite.orbit.v - target.orbit.v
    
#     # 2. 抵消相对速度所需的 delta_v 就是其反方向
#     delta_v = -relative_v
    
#     # 3. 封装并返回机动方案
#     plan = ManeuverPlan(
#         maneuvers=[
#             ManeuverExecution(delta_v=delta_v, execution_time=satellite.epoch)
#         ],
#         total_delta_v=np.linalg.norm(delta_v.to_value(u.km/u.s)) * (u.km/u.s),
#         total_time=0 * u.s
#     )
#     return plan

# def plan_circular_fly_around(satellite: Satellite, target: Satellite, radius: u.Quantity) -> ManeuverPlan:
#     """
#     规划一次机动，进入以目标为中心的圆形相对绕飞轨道。
#     这是一个简化的绕飞模型，假设在近距离执行。
    
#     Args:
#         satellite (Satellite): 执行绕飞的卫星。
#         target (Satellite): 被绕飞的目标。
#         radius (u.Quantity): 期望的绕飞半径。

#     Returns:
#         ManeuverPlan: 包含一次脉冲机动的绕飞方案。
#     """
#     if satellite.epoch != target.epoch:
#         raise ValueError("卫星和目标的历元必须相同。")

#     orbit_target = target.orbit
#     n = orbit_target.n.to_value(u.rad / u.s)  # 目标轨道的平均角速度

#     # 简化模型：假设在目标的正后方（-V bar）注入一个径向速度（R bar）来形成近似圆形绕飞
#     # 注入的速度大小 v = n * r
#     delta_v_magnitude = n * radius.to_value(u.m) * (u.m / u.s)
    
#     # 假设注入方向为径向向外
#     r_hat = orbit_target.r / np.linalg.norm(orbit_target.r)
#     delta_v_vec = delta_v_magnitude.to(u.km / u.s) * r_hat.value

#     # 这只是一个非常简化的注入脉冲，实际情况复杂得多
#     # 它假设我方卫星已经与目标近乎共速共轨
#     plan = ManeuverPlan(
#         maneuvers=[
#             ManeuverExecution(delta_v=delta_v_vec, execution_time=satellite.epoch)
#         ],
#         total_delta_v=np.linalg.norm(delta_v_vec.to_value(u.km / u.s)) * (u.km / u.s),
#         total_time=0 * u.s
#     )
#     return plan

# def plan_vbar_rendezvous(satellite: Satellite, target: Satellite, distance: u.Quantity) -> ManeuverPlan:
#     """
#     规划一次 V-bar 自然漂移抵近（霍普曼交会）。
#     通过一次变轨进入一个稍低的轨道，利用更快的速度追上目标，最后再进行一次机动完成交会。
    
#     Args:
#         satellite (Satellite): 执行交会的卫星。
#         target (Satellite): 目标卫星。
#         distance (u.Quantity): 初始时，卫星在目标后方的距离。

#     Returns:
#         ManeuverPlan: 包含两次脉冲机动的交会方案。
#     """
#     # 这是一个经典的 V-bar 抵近问题简化解
#     orbit_target = target.orbit
#     n = orbit_target.n.to_value(u.rad / u.s)  # 目标轨道的平均角速度
#     a = orbit_target.a.to_value(u.km) # 目标轨道的半长轴
    
#     # 1. 计算转移轨道的半长轴
#     # 转移时间设置为半个周期
#     time_of_flight = TimeDelta(np.pi / n, format='sec')
#     a_transfer = (a** (3 / 2) - (3 * n * distance.to_value(u.km)) / (8 * np.pi)) ** (2 / 3)

#     # 2. 计算第一次脉冲 (进入转移轨道)
#     v_target = np.sqrt(target.orbit.attractor.k.to_value(u.km**2/u.s**2) / a)
#     v_transfer_start = np.sqrt(target.orbit.attractor.k.to_value(u.km**2/u.s**2) * (2/a - 2/(a + a_transfer)))
#     dv1_mag = v_transfer_start - v_target
    
#     # 假设沿速度方向
#     v_hat = orbit_target.v / np.linalg.norm(orbit_target.v)
#     dv1_vec = dv1_mag * v_hat.value * (u.km / u.s)
    
#     # 3. 计算第二次脉冲 (返回目标轨道)
#     # 大小相等，方向相反
#     dv2_vec = -dv1_vec
    
#     # 4. 封装机动方案
#     exec_t1 = satellite.epoch
#     exec_t2 = satellite.epoch + time_of_flight
    
#     plan = ManeuverPlan(
#         maneuvers=[
#             ManeuverExecution(delta_v=dv1_vec, execution_time=exec_t1),
#             ManeuverExecution(delta_v=dv2_vec, execution_time=exec_t2)
#         ],
#         total_delta_v=(abs(dv1_mag) * 2) * (u.km/u.s),
#         total_time=time_of_flight
#     )
#     return plan

# def plan_formation_injection(
#     satellite: Satellite, 
#     target: Satellite, 
#     formation_id: str,
#     time_of_flight: TimeDelta
# ) -> ManeuverPlan:
#     """
#     规划一次机动，将卫星注入到一个预设的标准构型中。
    
#     Args:
#         satellite (Satellite): 执行机动的卫星。
#         target (Satellite): 构型所围绕的目标。
#         formation_id (str): 来自构型目录的ID。
#         time_of_flight (TimeDelta): 到达构型注入点的期望飞行时间。

#     Returns:
#         ManeuverPlan: 注入机动方案。
#     """
#     if formation_id not in FORMATION_CATALOG:
#         raise ValueError(f"构型ID '{formation_id}' 不在目录中。")

#     formation = FORMATION_CATALOG[formation_id]
    
#     # 1. 获取目标卫星在注入时刻的状态
#     target_orbit_at_injection = target.orbit.propagate(time_of_flight)
#     r_target_final = target_orbit_at_injection.r
#     v_target_final = target_orbit_at_injection.v
    
#     # 2. 将相对状态(LVLH)转换为绝对状态(ICRF)
#     # Poliastro 没有直接的 LVLH 转换，此处用简化方法（仅适用于近圆轨道）
#     # 实际项目中可能需要更精确的坐标转换
#     r_hat = target_orbit_at_injection.r / np.linalg.norm(target_orbit_at_injection.r)
#     v_hat = target_orbit_at_injection.v / np.linalg.norm(target_orbit_at_injection.v)
#     h_hat = np.cross(r_hat, v_hat) # Z轴 (法向)
    
#     # LVLH to ICRF 旋转矩阵的近似
#     rotation_matrix = np.array([r_hat.value, v_hat.value, h_hat.value]).T
    
#     dr_icrf = rotation_matrix @ formation.dr_lvlh.to_value(u.km) * u.km
#     # 对于速度，还需要考虑科里奥利项，此处简化
#     dv_icrf = rotation_matrix @ formation.dv_lvlh.to_value(u.km / u.s) * u.km / u.s
    
#     # 3. 计算注入点的绝对状态（位置和速度）
#     r_injection = r_target_final + dr_icrf
#     v_injection = v_target_final + dv_icrf

#     # 4. 现在问题转化为一个兰伯特问题：从当前位置，在给定时间到达注入点
#     try:
#         (v_initial_transfer, v_final_transfer), = lambert(
#             satellite.orbit.attractor.k, satellite.orbit.r, r_injection, time_of_flight
#         )
#     except ValueError as e:
#         print(f"无法规划注入轨道，兰伯特问题无解: {e}")
#         return None

#     # 5. 计算注入机动的两次脉冲
#     dv1 = v_initial_transfer - satellite.orbit.v # 出发脉冲
#     dv2 = v_injection - v_final_transfer       # 抵达注入点并匹配速度的脉冲
    
#     t1 = satellite.epoch
#     t2 = satellite.epoch + time_of_flight
    
#     total_dv = np.linalg.norm(dv1.to_value(u.km/u.s)) + np.linalg.norm(dv2.to_value(u.km/u.s))
#     plan = ManeuverPlan(
#         maneuvers=[
#             ManeuverExecution(delta_v=dv1, execution_time=t1),
#             ManeuverExecution(delta_v=dv2, execution_time=t2)
#         ],
#         total_delta_v=total_dv * (u.km/u.s),
#         total_time=time_of_flight
#     )
#     return plan

# environment/logic/maneuver_planner.py (FIXED)

from dataclasses import dataclass
from typing import List, Dict

import numpy as np

from astropy import units as u
from astropy.time import Time, TimeDelta

from poliastro.maneuver import Maneuver
from poliastro.iod import lambert
from poliastro.twobody import Orbit  # 导入Orbit用于创建理想轨道

from environment.core.satellite import Satellite

from environment.core.constants import EARTH_RADIUS 
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

    final_orbit_target = initial_orbit_target.propagate(time_of_flight)
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

    total_dv = np.linalg.norm(dv1.to_value(u.km/u.s)) + np.linalg.norm(dv2.to_value(u.km/u.s))
    plan = ManeuverPlan(
        maneuvers=[maneuver1, maneuver2],
        total_delta_v=total_dv * (u.km/u.s),
        total_time=time_of_flight
    )
    return plan



def plan_hohmann_transfer(satellite: Satellite, final_radius: u.Quantity) -> ManeuverPlan:
    """
    规划一次经典的霍曼转移。
    V4 版本：基于对 poliastro 源码的正确理解重构。
    """
    initial_orbit = satellite.orbit
    
    # 1. 直接调用 poliastro 的 hohmann 方法。
    #    它内部会自动处理漂移到近地点以及后续计算。
    maneuver = Maneuver.hohmann(initial_orbit, final_radius)
    
    # 2. 正确地解读返回的时间序列
    #    impulses[0] 是 (漂移时间, 第一次脉冲)
    #    impulses[1] 是 (转移时间, 第二次脉冲)
    (t_drift, dv1), (t_transfer, dv2) = maneuver.impulses
    
    # 3. 计算两次机动的绝对执行时间
    #    第一次机动的执行时间 = 初始时刻 + 漂移到近地点的时间
    exec_t1 = initial_orbit.epoch + t_drift
    #    第二次机动的执行时间 = 第一次机动执行时间 + 霍曼转移的飞行时间
    exec_t2 = exec_t1 + t_transfer
    
    # 4. 组装机动计划
    plan = ManeuverPlan(
        maneuvers=[
            ManeuverExecution(delta_v=dv1, execution_time=exec_t1),
            ManeuverExecution(delta_v=dv2, execution_time=exec_t2)
        ],
        total_delta_v=maneuver.get_total_cost(),
        total_time=t_drift + t_transfer  # 总时间是漂移时间+转移时间
    )

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
    orbit_target = target.orbit
    n = orbit_target.n.to_value(u.rad / u.s)
    a = orbit_target.a.to_value(u.km)
    time_of_flight = TimeDelta(np.pi / n, format='sec')
    a_transfer = (a ** (3 / 2) - (3 * n * distance.to_value(u.km)) / (8 * np.pi)) ** (2 / 3)
    v_target = np.sqrt(target.orbit.attractor.k.to_value(u.km**2/u.s**2) / a)
    v_transfer_start = np.sqrt(target.orbit.attractor.k.to_value(u.km**2/u.s**2) * (2/a - 2/(a + a_transfer)))
    dv1_mag = v_transfer_start - v_target
    v_hat = orbit_target.v / np.linalg.norm(orbit_target.v)
    dv1_vec = dv1_mag * v_hat.value * (u.km / u.s)
    dv2_vec = -dv1_vec
    exec_t1 = satellite.epoch
    exec_t2 = satellite.epoch + time_of_flight
    return ManeuverPlan(
        maneuvers=[
            ManeuverExecution(delta_v=dv1_vec, execution_time=exec_t1),
            ManeuverExecution(delta_v=dv2_vec, execution_time=exec_t2)
        ],
        total_delta_v=(abs(dv1_mag) * 2) * (u.km/u.s),
        total_time=time_of_flight
    )

def plan_formation_injection(
    satellite: Satellite,
    target: Satellite,
    formation_id: str,
    time_of_flight: TimeDelta
) -> ManeuverPlan:
    """
    规划一次机动，将卫星注入到一个预设的标准构型中。
    """
    if formation_id not in FORMATION_CATALOG:
        raise ValueError(f"构型ID '{formation_id}' 不在目录中。")

    formation = FORMATION_CATALOG[formation_id]

    # ... (计算注入点状态的代码不变) ...
    target_orbit_at_injection = target.orbit.propagate(time_of_flight)
    r_target_final = target_orbit_at_injection.r
    v_target_final = target_orbit_at_injection.v

    r_hat = r_target_final / np.linalg.norm(r_target_final)
    v_hat = v_target_final / np.linalg.norm(v_target_final)
    h_hat = np.cross(r_hat, v_hat)

    rotation_matrix = np.array([r_hat.value, v_hat.value, h_hat.value]).T
    dr_icrf = rotation_matrix @ formation.dr_lvlh.to_value(u.km) * u.km
    dv_icrf = rotation_matrix @ formation.dv_lvlh.to_value(u.km / u.s) * u.km / u.s

    r_injection = r_target_final + dr_icrf
    v_injection = v_target_final + dv_icrf

    try:
        # -- FIX START --
        # 同样，修正这里的赋值语句以匹配 lambert 函数的返回结构。
        v_initial_transfer, v_final_transfer = lambert(
            satellite.orbit.attractor.k, satellite.orbit.r, r_injection, time_of_flight
        )
        # -- FIX END --

    except ValueError as e:
        print(f"无法规划注入轨道，兰伯特问题无解: {e}")
        return None

    # 计算并封装机动方案 (这部分逻辑不变)
    dv1 = v_initial_transfer - satellite.orbit.v
    dv2 = v_injection - v_final_transfer

    t1 = satellite.epoch
    t2 = satellite.epoch + time_of_flight

    total_dv = np.linalg.norm(dv1.to_value(u.km/u.s)) + np.linalg.norm(dv2.to_value(u.km/u.s))
    plan = ManeuverPlan(
        maneuvers=[
            ManeuverExecution(delta_v=dv1, execution_time=t1),
            ManeuverExecution(delta_v=dv2, execution_time=t2)
        ],
        total_delta_v=total_dv * (u.km/u.s),
        total_time=time_of_flight
    )
    return plan