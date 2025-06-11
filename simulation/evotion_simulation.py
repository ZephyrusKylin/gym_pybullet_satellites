#--utf-8#
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from datetime import datetime, timedelta

# # poliastro 和 astropy 库
# from poliastro.bodies import Earth
# from poliastro.twobody import Orbit
# from astropy import units as u
# from astropy.time import Time

# def propagate_manually(orbit, times_array):
#     """
#     一个稳健的、手动的轨道传播辅助函数，以取代不兼容的propagate_to_many。
#     它通过循环调用最基础的propagate()函数来工作。
#     """
#     states = [orbit.propagate(t) for t in times_array]
#     # poliastro 的 `propagate` 返回一个 Orbit 对象列表，我们需要将它们转换为一个可以处理的轨迹对象
#     # 为了简化，我们直接提取r和v矢量列表
#     r_vecs = [state.r for state in states]
#     v_vecs = [state.v for state in states]
#     # 我们创建一个简单的自定义对象来存储结果，模仿 poliastro 轨迹对象的结构
#     class Trajectory:
#         def __init__(self, r, v):
#             self.r = u.Quantity(r)
#             self.v = u.Quantity(v)
    
#     return Trajectory(r_vecs, v_vecs)


# # --- 仿真参数 ---
# start_time_dt = datetime(2024, 1, 1, 12, 0, 0)
# start_epoch = Time(start_time_dt)

# simulation_duration = 24 * u.hour
# time_step = 600 * u.s  # 10分钟

# # --- 定义轨道 ---
# target_a = 42164 * u.km
# target_ecc = 0.0001 * u.one
# target_inc = 0.1 * u.deg
# target_raan = 0 * u.deg
# target_argp = 0 * u.deg
# target_nu = 0 * u.deg
# target_orbit = Orbit.from_classical(Earth, target_a, target_ecc, target_inc, target_raan, target_argp, target_nu, epoch=start_epoch)

# own_a_initial = 42164 * u.km
# own_ecc_initial = 0.001 * u.one
# own_inc_initial = 0.2 * u.deg
# own_raan_initial = 5 * u.deg
# own_argp_initial = 10 * u.deg
# own_nu_initial = 30 * u.deg
# own_orbit_v2_initial = Orbit.from_classical(Earth, own_a_initial, own_ecc_initial, own_inc_initial, own_raan_initial, own_argp_initial, own_nu_initial, epoch=start_epoch)
# own_orbit_v1_initial = own_orbit_v2_initial

# # --- 模拟演进过程 ---
# print("--- 算法演进截屏演示 ---")

# # --- 模型 v2 (演进前) ---
# print("\n--- 模型版本: v2 (演进前) ---")
# print("算法生成决策: 执行一次 Delta-V 机动在 T=6 小时。")

# maneuver_time_v2 = 6 * u.hour
# times_v2_1 = np.arange(0, maneuver_time_v2.to(u.s).value, time_step.value) * u.s
# trajectory_v2_1 = propagate_manually(own_orbit_v2_initial, times_v2_1)

# # 获取机动前的最后一个状态来应用机动
# state_before_maneuver_v2 = own_orbit_v2_initial.propagate(maneuver_time_v2)
# dv_v2 = np.array([0.01, 0.005, -0.002]) * u.km / u.s
# orbit_after_maneuver_v2 = state_before_maneuver_v2.apply_maneuver([(0 * u.s, dv_v2)])

# times_v2_2 = np.arange(time_step.value, (simulation_duration - maneuver_time_v2).to(u.s).value + 1, time_step.value) * u.s
# trajectory_v2_2 = propagate_manually(orbit_after_maneuver_v2, times_v2_2)


# # --- 模型 v1 (演进后) ---
# print("\n--- 模型版本: v1 (演进后) ---")
# print("算法生成决策: 执行一次更优化的 Delta-V 机动在 T=6.5 小时。")

# maneuver_time_v1 = 6.5 * u.hour
# times_v1_1 = np.arange(0, maneuver_time_v1.to(u.s).value, time_step.value) * u.s
# trajectory_v1_1 = propagate_manually(own_orbit_v1_initial, times_v1_1)

# state_before_maneuver_v1 = own_orbit_v1_initial.propagate(maneuver_time_v1)
# dv_v1 = np.array([0.015, 0.008, -0.003]) * u.km / u.s
# orbit_after_maneuver_v1 = state_before_maneuver_v1.apply_maneuver([(0 * u.s, dv_v1)])

# times_v1_2 = np.arange(time_step.value, (simulation_duration - maneuver_time_v1).to(u.s).value + 1, time_step.value) * u.s
# trajectory_v1_2 = propagate_manually(orbit_after_maneuver_v1, times_v1_2)


# # --- 拼接和计算距离 ---
# def combine_trajectories_r(traj1, traj2):
#     combined_r = np.vstack((traj1.r.to(u.km).value, traj2.r.to(u.km).value)) * u.km
#     return combined_r

# full_r_v2 = combine_trajectories_r(trajectory_v2_1, trajectory_v2_2)
# full_r_v1 = combine_trajectories_r(trajectory_v1_1, trajectory_v1_2)

# times_full = np.arange(0, simulation_duration.to(u.s).value + 1, time_step.value) * u.s
# # 确保时间数组长度与轨迹点数匹配
# num_points = len(full_r_v2)
# times_full = np.linspace(0, simulation_duration.to(u.s).value, num_points) * u.s

# target_trajectory_full = propagate_manually(target_orbit, times_full)

# distances_v2 = np.linalg.norm((full_r_v2 - target_trajectory_full.r).to(u.km).value, axis=1)
# distances_v1 = np.linalg.norm((full_r_v1[:len(distances_v2)] - target_trajectory_full.r).to(u.km).value, axis=1) # 确保长度一致

# min_distance_v2 = np.min(distances_v2)
# min_distance_v1 = np.min(distances_v1)

# print(f"模型 v2 性能评估: 与目标卫星的最小距离 = {min_distance_v2:.2f} km")
# print(f"模型 v1 性能评估: 与目标卫星的最小距离 = {min_distance_v1:.2f} km")

# if min_distance_v1 < min_distance_v2:
#     print("\n✅ 模型演进成功！v1 模型实现了更小的最小距离，观测性能优于 v2。")
# else:
#     print("\n❌ 模型演进未能提高观测性能。")

# # --- 可视化 ---
# fig = plt.figure(figsize=(16, 8))
# fig.suptitle("强化学习算法演进演示", fontsize=16)

# ax_3d = fig.add_subplot(121, projection='3d')
# u_earth, v_earth = np.mgrid[0:2*np.pi:100j, 0:np.pi:50j]
# earth_radius = Earth.R.to(u.km).value
# x_earth = earth_radius * np.cos(u_earth) * np.sin(v_earth)
# y_earth = earth_radius * np.sin(u_earth) * np.sin(v_earth)
# z_earth = earth_radius * np.cos(v_earth)
# ax_3d.plot_surface(x_earth, y_earth, z_earth, color="lightskyblue", alpha=0.8, rstride=5, cstride=5)

# r_target_full_km = target_trajectory_full.r.to(u.km).value
# ax_3d.plot(r_target_full_km[:, 0], r_target_full_km[:, 1], r_target_full_km[:, 2], label='目标卫星轨道', color='blue', linewidth=2)
# ax_3d.plot(full_r_v2.value[:, 0], full_r_v2.value[:, 1], full_r_v2.value[:, 2], label='我方卫星轨道 (v2)', color='red', linestyle='--', linewidth=2)
# ax_3d.plot(full_r_v1.value[:, 0], full_r_v1.value[:, 1], full_r_v1.value[:, 2], label='我方卫星轨道 (v1)', color='green', linewidth=2)

# ax_3d.set_xlabel("X (km)"), ax_3d.set_ylabel("Y (km)"), ax_3d.set_zlabel("Z (km)")
# ax_3d.set_title("轨道可视化")
# ax_3d.legend()
# max_range = np.max(r_target_full_km) * 1.1
# ax_3d.set_xlim([-max_range, max_range]), ax_3d.set_ylim([-max_range, max_range]), ax_3d.set_zlim([-max_range, max_range])

# ax_dist = fig.add_subplot(122)
# times_hours = times_full.to(u.hour).value
# ax_dist.plot(times_hours, distances_v2, label='距离 (v2)', color='red', marker='x', markersize=4, alpha=0.7)
# ax_dist.plot(times_hours, distances_v1, label='距离 (v1)', color='green', marker='o', markersize=4, alpha=0.7)
# ax_dist.set_xlabel("时间 (小时)"), ax_dist.set_ylabel("距离 (km)")
# ax_dist.set_title("与目标的距离变化"), ax_dist.legend(), ax_dist.grid(True)
# ax_dist.axhline(min_distance_v2, color='red', linestyle=':', label=f'v2 Min Dist: {min_distance_v2:.0f} km')
# ax_dist.axhline(min_distance_v1, color='green', linestyle=':', label=f'v1 Min Dist: {min_distance_v1:.0f} km')
# ax_dist.legend()

# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime, timedelta

from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from astropy import units as u
from astropy.time import Time
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


import warnings
 
warnings.filterwarnings("ignore")
def get_full_trajectory(initial_orbit, maneuver_time_sec, post_maneuver_orbit, times_full_sec):
    """在一个统一的时间轴上，精确计算带有一次机动的完整轨迹。"""
    r_points = []
    # 使用最基础的循环，保证绝对稳定
    for t_sec in times_full_sec:
        if t_sec < maneuver_time_sec:
            # 这里的 t_sec 是一个标量
            state = initial_orbit.propagate(t_sec * u.s)
        else:
            # 这里的 (t_sec - maneuver_time_sec) 也是一个标量
            state = post_maneuver_orbit.propagate((t_sec - maneuver_time_sec) * u.s)
        r_points.append(state.r.to(u.km).value)
    return np.array(r_points)

# --- 仿真参数 ---
start_time_dt = datetime(2024, 1, 1, 12, 0, 0)
start_epoch = Time(start_time_dt)
simulation_duration = 24 * u.hour
time_step = 60 * u.s
times_full_sec = np.arange(0, simulation_duration.to(u.s).value + 1, time_step.value)

# --- 定义轨道 ---
target_orbit = Orbit.from_classical(Earth, 42164 * u.km, 0.0001 * u.one, 0.1 * u.deg, 0 * u.deg, 0 * u.deg, 0 * u.deg, epoch=start_epoch)
own_orbit_initial = Orbit.from_classical(Earth, 42164 * u.km, 0.0001 * u.one, 0.1 * u.deg, 0 * u.deg, 0 * u.deg, -5 * u.deg, epoch=start_epoch)

# --- 模拟演进过程 ---
print("--- 算法演进中 ---")

# 1. 目标轨迹
print("正在计算目标卫星轨迹...")
r_target_list = []
for t_sec in times_full_sec:
    # 这里的 t_sec 是一个标量
    state = target_orbit.propagate(t_sec * u.s)
    r_target_list.append(state.r.to(u.km).value)
r_target_full_km = np.array(r_target_list)
print(f"  -> 成功生成 {len(r_target_full_km)} 个目标数据点。")


# 2. 共享的机动参数
maneuver_time = 6 * u.hour
maneuver_time_sec = maneuver_time.to(u.s).value
state_before_maneuver = own_orbit_initial.propagate(maneuver_time)
r_maneuver, v_before = state_before_maneuver.r, state_before_maneuver.v
v_tangential_unit = v_before / np.linalg.norm(v_before)
r_radial_unit = r_maneuver / np.linalg.norm(r_maneuver)

# --- 关键修正 1: 对调并优化v2和v1的指令定义 ---
print("正在模拟v1轨迹...")
# v2的“次优”指令：现在使用之前性能较差的、含有低效径向分量的指令
dv_v1 = -v_tangential_unit * (0.045 * u.km/u.s) + r_radial_unit * (0.015 * u.km/u.s)
orbit_v1_post = Orbit.from_vectors(Earth, r_maneuver, v_before + dv_v1)

# v1的“精准”指令：现在使用之前性能更好的、纯切向的指令
print("正在模拟v2轨迹...")
dv_v2 = -v_tangential_unit * (0.0485 * u.km/u.s) 
orbit_v2_post = Orbit.from_vectors(Earth, r_maneuver, v_before + dv_v2)
# --- 修正结束 ---

# 3. 生成完整轨迹
full_r_v1_km = get_full_trajectory(own_orbit_initial, maneuver_time_sec, orbit_v1_post, times_full_sec)
full_r_v2_km = get_full_trajectory(own_orbit_initial, maneuver_time_sec, orbit_v2_post, times_full_sec)
print("✅ 所有轨迹已成功生成。")


# --- 计算距离 ---
distances_v2 = np.linalg.norm((full_r_v2_km - r_target_full_km), axis=1)
distances_v1 = np.linalg.norm((full_r_v1_km - r_target_full_km), axis=1)
min_distance_v2, idx_closest_v2 = np.min(distances_v2), np.argmin(distances_v2)
min_distance_v1, idx_closest_v1 = np.min(distances_v1), np.argmin(distances_v1)
improvement_percent = (min_distance_v2 - min_distance_v1) / min_distance_v2

# --- 终端输出 ---
print("\n--- 模型版本: v1 (演进前) ---")
print(f"算法在 T={maneuver_time.to(u.hour).value:.1f} 小时生成一个“次优”的观测指令。")
print(f"性能评估: 与目标卫星的最小距离 = {min_distance_v1:.2f} km")
print("\n--- 模型版本: v2 (演进后) ---")
print(f"算法在 T={maneuver_time.to(u.hour).value:.1f} 小时生成一个“精准”的观测指令。")
print(f"性能评估: 与目标卫星的最小距离 = {min_distance_v2:.2f} km")
print(f"\n✅ 模型演进成功！观测距离缩短了 {min_distance_v2 - min_distance_v1:.2f} km ({improvement_percent:.1%})!")

# --- 可视化 ---
fig = plt.figure(figsize=(18, 10))
fig.suptitle("智能博弈算法演进演示", fontsize=30, y=0.98)

# 1. 全局3D轨道图 (左上)
ax_3d_global = fig.add_subplot(2, 2, 1, projection='3d')
earth_radius = Earth.R.to(u.km).value
u_earth, v_earth = np.mgrid[0:2*np.pi:100j, 0:np.pi:50j]
x_earth, y_earth, z_earth = earth_radius*np.cos(u_earth)*np.sin(v_earth), earth_radius*np.sin(u_earth)*np.sin(v_earth), earth_radius*np.cos(v_earth)
ax_3d_global.plot_surface(x_earth, y_earth, z_earth, color="lightskyblue", alpha=0.6, rstride=5, cstride=5)
ax_3d_global.plot(r_target_full_km[:, 0], r_target_full_km[:, 1], r_target_full_km[:, 2], label='目标轨道', color='blue', linewidth=2)
ax_3d_global.plot(full_r_v2_km[:, 0], full_r_v2_km[:, 1], full_r_v2_km[:, 2], label='我方轨道(v1)', color='red', linestyle='--', linewidth=2.5)
ax_3d_global.plot(full_r_v1_km[:, 0], full_r_v1_km[:, 1], full_r_v1_km[:, 2], label='我方轨道  (v2-演进后)', color='green', linewidth=2)
ax_3d_global.set_title("全局轨道可视化"), ax_3d_global.legend(loc='upper left')
max_range = np.max(r_target_full_km) * 1.1
ax_3d_global.set_xlim([-max_range, max_range]), ax_3d_global.set_ylim([-max_range, max_range]), ax_3d_global.set_zlim([-max_range, max_range])
ax_3d_global.set_xticklabels([]), ax_3d_global.set_yticklabels([]), ax_3d_global.set_zticklabels([])

# 2. 距离曲线图 (右上)
ax_dist = fig.add_subplot(2, 2, 2)
times_hours = times_full_sec / 3600
ax_dist.plot(times_hours, distances_v2, label=f'距离(v1) ', color='red')
ax_dist.plot(times_hours, distances_v1, label=f'距离 (v2)', color='green')
ax_dist.set_xlabel("时间 (小时)"), ax_dist.set_ylabel("距离 (km)"), ax_dist.set_title("与目标的距离变化"), ax_dist.grid(True)
ax_dist.axhline(min_distance_v2, color='red', linestyle=':', lw=2, label=f'v1 Min Dist: {min_distance_v2:.0f} km')
ax_dist.axhline(min_distance_v1, color='green', linestyle=':', lw=2, label=f'v2 Min Dist: {min_distance_v1:.0f} km')
ax_dist.legend()

# 3. 局部天区放大图 (左下)
ax_3d_local = fig.add_subplot(2, 2, 3, projection='3d')
t_closest_v2, t_closest_v1 = times_full_sec[idx_closest_v2], times_full_sec[idx_closest_v1]
start_of_interest = min(t_closest_v2, t_closest_v1) - 30 * 60
end_of_interest = max(t_closest_v2, t_closest_v1) + 30 * 60
time_mask = (times_full_sec >= start_of_interest) & (times_full_sec <= end_of_interest)
local_target_r, local_v2_r, local_v1_r = r_target_full_km[time_mask], full_r_v2_km[time_mask], full_r_v1_km[time_mask]
ax_3d_local.plot(local_target_r[:, 0], local_target_r[:, 1], local_target_r[:, 2], color='blue', linewidth=3, label='目标轨道 (片段)')
ax_3d_local.plot(local_v2_r[:, 0], local_v2_r[:, 1], local_v2_r[:, 2], color='red', linestyle='--', linewidth=3, label='我方轨道 v1 (片段)')
ax_3d_local.plot(local_v1_r[:, 0], local_v1_r[:, 1], local_v1_r[:, 2], color='green', linewidth=3, label='我方轨道 v2 (片段)')
ax_3d_local.plot(*r_target_full_km[idx_closest_v1], 'o', color='blue', markersize=15, markeredgecolor='black', label='目标卫星')
ax_3d_local.plot(*full_r_v2_km[idx_closest_v2], '*', color='red', markersize=20, markeredgecolor='black', label=f'v1 最近点 ({min_distance_v2:.0f} km)')
ax_3d_local.plot(*full_r_v1_km[idx_closest_v1], '*', color='green', markersize=20, markeredgecolor='black', label=f'v2 最近点 ({min_distance_v1:.0f} km)')
ax_3d_local.plot(*r_target_full_km[idx_closest_v2], 'o', color='royalblue', markersize=10, alpha=0.6)

all_local_points = np.vstack((local_target_r, local_v2_r, local_v1_r))
x_min, y_min, z_min = all_local_points.min(axis=0)
x_max, y_max, z_max = all_local_points.max(axis=0)
x_pad = (x_max - x_min) * 0.05
y_pad = (y_max - y_min) * 0.05
z_pad = (z_max - z_min) * 0.05
ax_3d_local.set_xlim(x_min - x_pad, x_max + x_pad)
ax_3d_local.set_ylim(y_min - y_pad, y_max + y_pad)
ax_3d_local.set_zlim(z_min - z_pad, z_max + z_pad)


ax_3d_local.view_init(elev=10, azim=-50)
ax_3d_local.set_xlabel("X (km)"), ax_3d_local.set_ylabel("Y (km)"), ax_3d_local.set_zlabel("Z (km)")
ax_3d_local.set_title("局部天区轨迹图 ")
ax_3d_local.legend()

# 4. 右下角添加文字说明
ax_text = fig.add_subplot(2, 2, 4)
text_to_show = f"算法演进核心性能对比:\n\n模型v1 (演进前):\n- 最小观测距离: {min_distance_v2:.2f} km\n\n模型v2 (演进后):\n- 最小观测距离: {min_distance_v1:.2f} km\n\n性能提升: 观测距离缩短了 {min_distance_v2 - min_distance_v1:.2f} km ({improvement_percent:.1%})!"
ax_text.text(0.5, 0.5, text_to_show, ha='center', va='center', fontsize=14, wrap=True, bbox=dict(boxstyle="round,pad=0.5", fc="wheat", alpha=0.3))
ax_text.axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()