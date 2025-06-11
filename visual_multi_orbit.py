import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm # 用于生成多彩的颜色
import os
def animate_multiple_orbits(file_paths, root, animation_interval=50):
    """
    通过动画可视化多个卫星的轨道，并实时显示它们的位置和同步时间。

    Args:
        file_paths (list): 包含多个CSV文件路径的列表。
        animation_interval (int): 动画帧之间的间隔（毫秒）。
    """
    print(f"🛰️ 发现 {len(file_paths)} 个卫星文件，开始处理...")

    all_dfs = []
    # 1. 加载所有文件并进行预处理
    for i, file_path in enumerate(file_paths):
        fp = os.path.join(root, file_path)
        try:
            df = pd.read_csv(fp)
            # 为列重命名以避免合并时冲突，但时间列除外
            df.columns = ['time_str'] + [f'{col}_{i}' for col in ['x', 'y', 'z', 'vx', 'vy', 'vz']]
            df['time'] = pd.to_datetime(df['time_str'], errors='coerce')
            
            numeric_cols = [f'x_{i}', f'y_{i}', f'z_{i}', f'vx_{i}', f'vy_{i}', f'vz_{i}']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # 保留必要的列，准备合并
            all_dfs.append(df[['time'] + numeric_cols])
        except Exception as e:
            print(f"❌ 读取或处理文件 {file_path} 时出错: {e}")
            continue
    
    if not all_dfs:
        print("❌ 未能成功加载任何卫星数据。")
        return

    # 2. 将所有数据按时间合并，确保同步
    # 从第一个DataFrame开始，通过内连接(inner join)确保只保留所有文件共有的时间点
    merged_df = all_dfs[0]
    for i in range(1, len(all_dfs)):
        merged_df = pd.merge(merged_df, all_dfs[i], on='time', how='inner')
    
    merged_df.dropna(inplace=True)
    if merged_df.empty:
        print("❌ 错误: 卫星文件之间没有共同的时间戳，无法进行同步动画。")
        return
    
    print(f"数据合并完成，找到 {len(merged_df)} 个同步的时间点。")

    # 3. 设置绘图和颜色
    print("正在生成3D轨道图...")
    fig = plt.figure(figsize=(16, 14))
    ax = fig.add_subplot(111, projection='3d')
    # 使用 'viridis' 颜色图为每个卫星生成不同的颜色
    colors = cm.get_cmap('viridis', len(file_paths))

    # 4. 绘制所有卫星的静态完整轨迹
    for i in range(len(file_paths)):
        ax.plot(merged_df[f'x_{i}'], merged_df[f'y_{i}'], merged_df[f'z_{i}'], 
                color=colors(i), alpha=0.5, label=f'Satellite {i+1} Path')

    # 绘制地球
    u_sphere, v_sphere = np.mgrid[0:2*np.pi:100j, 0:np.pi:50j]
    R_EARTH = 6371.0 # 地球半径 in km
    x_sphere = R_EARTH * np.cos(u_sphere) * np.sin(v_sphere)
    y_sphere = R_EARTH * np.sin(u_sphere) * np.sin(v_sphere)
    z_sphere = R_EARTH * np.cos(v_sphere)
    ax.plot_surface(x_sphere, y_sphere, z_sphere, color='c', alpha=0.3)
    
    # 5. 创建动画元素
    satellite_markers = []
    for i in range(len(file_paths)):
        marker, = ax.plot([], [], [], 'o', color=colors(i), markersize=10, markeredgecolor='black')
        satellite_markers.append(marker)
    # --- 新增的修正部分：设置等比例坐标轴 ---
    print("正在计算坐标轴范围以确保地球为球形...")
    # 收集所有卫星的所有X, Y, Z坐标
    all_x = pd.concat([merged_df[f'x_{i}'] for i in range(len(file_paths))])
    all_y = pd.concat([merged_df[f'y_{i}'] for i in range(len(file_paths))])
    all_z = pd.concat([merged_df[f'z_{i}'] for i in range(len(file_paths))])

    # 计算中心点和最大范围
    x_range, y_range, z_range = all_x.max() - all_x.min(), all_y.max() - all_y.min(), all_z.max() - all_z.min()
    mid_x, mid_y, mid_z = all_x.mean(), all_y.mean(), all_z.mean()
    max_range = max(x_range, y_range, z_range) / 2.0

    # 将计算出的最大范围应用到所有轴
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    # --- 修正结束 ---    
    time_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes, color='white', 
                          bbox=dict(facecolor='black', alpha=0.5))

    # 设置图表样式
    ax.set_xlabel('X (km)'), ax.set_ylabel('Y (km)'), ax.set_zlabel('Z (km)')
    ax.set_title('Multi-Satellite Animated Orbits')
    ax.legend(loc='upper right')
    
    # 6. 定义动画更新函数
    def update(frame):
        current_data = merged_df.iloc[frame]
        
        # 遍历每一颗卫星，更新其标记位置
        for i in range(len(file_paths)):
            satellite_markers[i].set_data_3d(
                [current_data[f'x_{i}']], 
                [current_data[f'y_{i}']], 
                [current_data[f'z_{i}']]
            )
        
        # 更新共享的时间文本
        time_text.set_text(f"Time: {current_data['time'].strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
        
        # 返回所有已更新的艺术家对象
        return (*satellite_markers, time_text)

    # 7. 创建并启动动画
    ani = FuncAnimation(fig, update, frames=len(merged_df), interval=animation_interval, blit=True, repeat=True)

    plt.show()
    print("\n✅ 动画窗口已弹出。")

# --- 运行主函数 ---
if __name__ == '__main__':
    # --- 请在这里修改您的CSV文件名列表 ---
    files_to_visualize = [
        'S1_J2000_Position_Velocity.csv',
        'S2_J2000_Position_Velocity.csv', # 取消注释并添加更多文件
        'S3_J2000_Position_Velocity.csv',
        'S4_J2000_Position_Velocity.csv',
        'T1_J2000_Position_Velocity.csv'
    ]
    # ------------------------------------
    root = 'H:\\old_D\\Work\\501_平台\\311'
    animate_multiple_orbits(files_to_visualize, root=root, animation_interval=50)