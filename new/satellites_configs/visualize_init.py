import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import yaml



from Satellite_Rl_Env import BaseSatelliteEnv

EARTH_RADIUS = 6_371e3  # 米



def visualize_satellite_config(config_file):
    """
    可视化卫星初始位置及属性（局部区域展示，包含地球）
    :param config_file: YAML 配置文件路径
    """
    # 地球半径

    # 读取配置
    with open(config_file, 'r',encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 提取所有卫星位置
    red_positions = np.array([sat['initial_position'] for sat in config.get('red', [])])
    blue_positions = np.array([sat['initial_position'] for sat in config.get('blue', [])])
    all_positions = np.vstack((red_positions, blue_positions))

    # 计算显示区域范围：以卫星群中心为中心，添加缓冲，同时包含地球原点
    center = all_positions.mean(axis=0)
    buffer = 5e6  # 5000 km 缓冲
    mins = all_positions.min(axis=0) - buffer
    maxs = all_positions.max(axis=0) + buffer
    # 确保包含原点（地球中心）
    mins = np.minimum(mins, np.zeros(3))
    maxs = np.maximum(maxs, np.zeros(3))

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制完整地球球体
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 50)
    x = EARTH_RADIUS * np.outer(np.cos(u), np.sin(v))
    y = EARTH_RADIUS * np.outer(np.sin(u), np.sin(v))
    z = EARTH_RADIUS * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color='cyan', alpha=0.3, linewidth=0)

    # 类型对应颜色和大小
    marker_info = {
        'observer': {'color': 'green',  'size': 60},
        'strike':   {'color': 'red',    'size': 60},
        'defense':  {'color': 'orange', 'size': 80},
        'high_value': {'color': 'purple','size':100}
    }

    def plot_satellites(sats):
        for sat in sats:
            pos = np.array(sat['initial_position'], dtype=float)
            info = marker_info.get(sat['type'], {'color':'gray','size':50})
            ax.scatter(*pos, color=info['color'], s=info['size'], depthshade=True)
            ax.text(*(pos * 1.001), f"{sat['type'][0].upper()}{sat['id']}",
                    color=info['color'], fontsize=8)

    # 绘制卫星
    plot_satellites(config.get('red', []))
    plot_satellites(config.get('blue', []))

    # 设置显示范围
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])
    ax.set_box_aspect([1,1,1])
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    plt.title('Satellite Configuration Visualization (Local Region with Earth)')
    plt.show()



def simulate_no_thrust(config_file, duration_sec=7200, timestep=10.0):
    """
    模拟无推力下的卫星轨迹，并将整个轨迹绘制成平滑曲线。
    """
    # 1. 初始化环境
    env = BaseSatelliteEnv(config_file, timestep=timestep,
                           max_steps=int(duration_sec / timestep))
    obs, _ = env.reset()
    n = env.n
    steps = int(duration_sec / timestep)

    # 2. 准备矩阵存储位置：shape = (n, steps+1, 3)
    trajectories = np.zeros((n, steps+1, 3), dtype=float)
    trajectories[:, 0, :] = obs.reshape(n, 9)[:, 0:3]

    # 3. 定义“零推力”动作
    zero_action = np.zeros((n, 4), dtype=np.float32)
    # special_action 已经在 step 中被映射为 0 (no-op)

    # 4. 逐步仿真并记录每一步
    for t in range(steps):
        obs, _, done, _, _ = env.step(zero_action)
        positions = obs.reshape(n, 9)[:, 0:3]
        trajectories[:, t+1, :] = positions
        if done:
            print(f"Episode terminated at step {t}")
            trajectories = trajectories[:, :t+2, :]
            break

    # 5. 绘图：先画地球，再画每条轨迹
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制地球表面（与之前相同）
    u = np.linspace(0, 2*np.pi, 100)
    v = np.linspace(0, np.pi, 50)
    x = EARTH_RADIUS * np.outer(np.cos(u), np.sin(v))
    y = EARTH_RADIUS * np.outer(np.sin(u), np.sin(v))
    z = EARTH_RADIUS * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color='cyan', alpha=0.3, linewidth=0)

    # 绘制各卫星轨迹（与之前相同）
    cmap = plt.get_cmap('tab10')
    for i in range(n):
        traj = trajectories[i]
        ax.plot(traj[:,0], traj[:,1], traj[:,2],
                color=cmap(i), linewidth=1.5, label=f'Sat {i}')
        ax.scatter(*traj[0], color=cmap(i), marker='o', s=50)
        ax.scatter(*traj[-1], color=cmap(i), marker='X', s=50)

    # —— 以下为关键修正 —— 

    # 方法一：手动屏蔽非有限值后设限
    all_pts = trajectories.reshape(-1,3)
    finite_mask = np.isfinite(all_pts).all(axis=1)
    finite_pts = all_pts[finite_mask]
    if finite_pts.size > 0:
        mins = finite_pts.min(axis=0)
        maxs = finite_pts.max(axis=0)
        pad = (maxs - mins) * 0.1
        ax.set_xlim(mins[0]-pad[0], maxs[0]+pad[0])
        ax.set_ylim(mins[1]-pad[1], maxs[1]+pad[1])
        ax.set_zlim(mins[2]-pad[2], maxs[2]+pad[2])
    else:
        # 若所有点都非有限，则退回自动缩放
        ax.relim()
        ax.autoscale_view()

    # 或者，方法二：完全依赖自动缩放
    # ax.relim()
    # ax.autoscale_view()

    ax.set_box_aspect([1,1,1])
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    plt.title('No-Thrust Trajectories Over Time')
    plt.legend()
    plt.show()
# 调用示例
# visualize_satellite_config('scenario_5v3_longrange.yaml')

# 调用示例
# visualize_satellite_config("scenario_5v3.yaml")

if __name__ == "__main__":
    
    S = BaseSatelliteEnv('5vs3.yaml', timestep=10.0, max_steps=720)
    print(S.n_red, S.n_blue)
    
    
    # visualize_satellite_config("5vs3.yaml")


    simulate_no_thrust('5vs3.yaml', duration_sec=72000, timestep=10.0)