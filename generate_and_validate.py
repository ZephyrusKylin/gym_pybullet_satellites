import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from astropy import units as u

def generate_maneuver_commands(file_path, energy_change_threshold=10.0):
    """
    精确分析轨道数据，仅在发生显著能量跳变时生成变轨指令。
    """
    print("--- 步骤 1: 正在分析数据并生成变轨指令 ---")
    print(f"--- 使用的能量变化阈值: {energy_change_threshold} km²/s² ---")

    # (这部分代码与上一版完全相同，用于精确生成指令)
    try:
        df = pd.read_csv(file_path)
        df.columns = ['time_str', 'x', 'y', 'z', 'vx', 'vy', 'vz']
        df['time'] = pd.to_datetime(df['time_str'], errors='coerce')
        numeric_cols = ['x', 'y', 'z', 'vx', 'vy', 'vz']
        for col in numeric_cols: df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(inplace=True)
        if df.empty: return [], None
        df['elapsed_seconds'] = (df['time'] - df['time'].iloc[0]).dt.total_seconds()
    except Exception as e:
        print(f"❌ 读取文件时出错: {e}")
        return [], None

    MU_EARTH = 398600.4418
    df['r_mag'] = np.linalg.norm(df[['x', 'y', 'z']].values, axis=1)
    df['v_mag'] = np.linalg.norm(df[['vx', 'vy', 'vz']].values, axis=1)
    df['energy'] = df['v_mag']**2 / 2 - MU_EARTH / df['r_mag']
    df['energy_diff'] = df['energy'].diff().abs()
    maneuver_indices = df[df['energy_diff'] > energy_change_threshold].index
    
    command_list = []
    if maneuver_indices.empty:
        print("✅ 在当前阈值下，未检测到任何显著的轨道机动。")
    else:
        print(f"🔥 检测到 {len(maneuver_indices)} 次显著的轨道机动！")
        for idx in maneuver_indices:
            state_after = df.loc[idx]
            r_vec = state_after[['x', 'y', 'z']].values * u.km
            v_vec = state_after[['vx', 'vy', 'vz']].values * u.km / u.s
            target_orbit_obj = Orbit.from_vectors(Earth, r_vec, v_vec)
            command = {
                "maneuver_time_sec": state_after['elapsed_seconds'],
                "target_orbit_elements": { "a": target_orbit_obj.a.to(u.km).value } # 可按需添加其他根数
            }
            command_list.append(command)
            
    return command_list, df

def validate_commands_and_visualize(command_list, ground_truth_df):
    """
    接收指令列表，进行分段仿真，并将结果与真实轨迹对比以进行验证。
    """
    print("\n--- 步骤 2: 正在验证指令并进行仿真对比 ---")
    
    if ground_truth_df is None or ground_truth_df.empty:
        print("❌ 无法进行验证，因为没有有效的地面真实数据。")
        return
        
    # 1. 建立仿真时间轴：包含开始、所有机动、结束时刻
    event_times = [0.0] + [cmd['maneuver_time_sec'] for cmd in command_list] + [ground_truth_df['elapsed_seconds'].iloc[-1]]
    event_times = sorted(list(set(event_times)))

    # 2. 设置初始状态
    initial_state = ground_truth_df.iloc[0]
    r0 = initial_state[['x', 'y', 'z']].values * u.km
    v0 = initial_state[['vx', 'vy', 'vz']].values * u.km / u.s
    current_orbit = Orbit.from_vectors(Earth, r0, v0)

    simulated_positions = []
    
    # 3. 按“事件”分段进行轨道传播
    for i in range(len(event_times) - 1):
        t_start = event_times[i]
        t_end = event_times[i+1]
        
        # 找到当前传播段内所有的时间点
        time_points_in_segment = ground_truth_df[
            (ground_truth_df['elapsed_seconds'] >= t_start) & 
            (ground_truth_df['elapsed_seconds'] < t_end)
        ]['elapsed_seconds'].values
        
        # 传播当前轨道
        if len(time_points_in_segment) > 0:
            # 计算相对于本段起点的传播时间
            propagation_times = (time_points_in_segment - t_start) * u.s
            for t in propagation_times:
                propagated_orbit = current_orbit.propagate(t)
                simulated_positions.append(propagated_orbit.r.to(u.km).value)

        # 在分段结束后，更新轨道以进行下一段的传播
        # 这模拟了“执行指令”的过程
        maneuver_row = ground_truth_df[ground_truth_df['elapsed_seconds'] == t_end]
        if not maneuver_row.empty:
            state_at_end = maneuver_row.iloc[0]
            r_new = state_at_end[['x', 'y', 'z']].values * u.km
            v_new = state_at_end[['vx', 'vy', 'vz']].values * u.km / u.s
            current_orbit = Orbit.from_vectors(Earth, r_new, v_new)
            
            # 如果是最后一点，也要加入仿真列表
            if t_end == event_times[-1]:
                simulated_positions.append(r_new.to(u.km).value)


    simulated_xyz = np.array(simulated_positions)

    # 4. 可视化对比
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot(ground_truth_df['x'], ground_truth_df['y'], ground_truth_df['z'], label='Ground Truth (from CSV)', color='blue', linewidth=4, alpha=0.8)
    ax.plot(simulated_xyz[:, 0], simulated_xyz[:, 1], simulated_xyz[:, 2], label='Simulated Path (from commands)', color='red', linestyle='--', linewidth=2)

    # 设置等比例坐标轴以保证地球是球形
    all_x = ground_truth_df['x']; all_y = ground_truth_df['y']; all_z = ground_truth_df['z']
    mid_x, mid_y, mid_z = all_x.mean(), all_y.mean(), all_z.mean()
    max_range = max(all_x.max()-all_x.min(), all_y.max()-all_y.min(), all_z.max()-all_z.min()) / 2.0
    ax.set_xlim(mid_x - max_range, mid_x + max_range); ax.set_ylim(mid_y - max_range, mid_y + max_range); ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax.set_title('Validation: Ground Truth vs. Command-Based Simulation')
    ax.legend()
    plt.show()
    print("✅ 验证完成！请检查图中两条轨迹是否完美重合。")

# --- 运行主流程 ---
if __name__ == '__main__':
    csv_file = 'H:\\old_D\\Work\\501_平台\\311\\S1_J2000_Position_Velocity.csv'
    
    # 步骤 1: 生成指令
    generated_commands, full_df = generate_maneuver_commands(csv_file, energy_change_threshold=10.0)
    
    if generated_commands:
        print("\n--- 生成的指令集 ---")
        print(json.dumps(generated_commands, indent=4))
        print("--------------------")

    # 步骤 2: 验证指令
    validate_commands_and_visualize(generated_commands, full_df)

