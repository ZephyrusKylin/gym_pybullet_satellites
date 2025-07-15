# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.animation import FuncAnimation
# import pandas as pd
# import numpy as np
# import io
# import os
# from datetime import datetime, timedelta

# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False

# def visualize_animated_trajectory(file_path, time_scale_factor=10, interval=50):
#     """
#     读取卫星轨迹数据文件，并以动画形式进行3D可视化。

#     参数:
#     file_path (str): 数据文件的路径。
#     time_scale_factor (int): 时间缩放系数。动画的每一帧代表多少个数据点（时间步）。
#     interval (int): 动画帧之间的延迟（毫秒）。
#     """
#     # 检查文件是否存在
#     if not os.path.exists(file_path):
#         print(f"错误：找不到文件 '{file_path}'。请确保文件名正确并且文件在正确的目录中。")
#         # 创建一个占位符内容
#         file_content = """
#        Time (UTCG)              x (m)            y (m)           z (m)       vx (m/sec)    vy (m/sec)     vz (m/sec)
# ------------------------    -------------    ------------    ------------    ----------    ----------    -----------
#  9 Jan 2025 08:00:00.000    -14671441.137    39493696.095    -1389053.745     -2.363709     -3.170823       3.791076
#  9 Jan 2025 08:01:00.000    -14671582.351    39493506.712    -1388815.001     -2.387704     -3.159980       4.234639
#         """
#         print("将使用内置的样本数据进行演示。")
#     else:
#         with open(file_path, 'r') as f:
#             file_content = f.read()

#     # 找到数据开始的行
#     lines = file_content.strip().split('\n')
#     data_start_line_index = 0
#     for i, line in enumerate(lines):
#         if '-----------' in line:
#             data_start_line_index = i + 1
#             break
    
#     if data_start_line_index == 0:
#         print("错误：无法在文件中找到数据分隔符。")
#         return

#     data_lines = lines[data_start_line_index:]
#     if not data_lines:
#         print("错误：在分隔符后没有找到数据行。")
#         return

#     # 准备要解析的数据字符串，并创建时间列
#     data_io = io.StringIO()
#     # 写入列名，以便解析日期
#     data_io.write("day month year time x y z vx vy vz\n")
#     data_io.write('\n'.join(data_lines))
#     data_io.seek(0)

#     try:
#         df = pd.read_csv(
#             data_io,
#             delim_whitespace=True
#         )
#         # 将日期和时间合并为datetime对象
#         df['datetime'] = pd.to_datetime(df['day'].astype(str) + ' ' + df['month'] + ' ' + df['year'].astype(str) + ' ' + df['time'])
#         x_coords, y_coords, z_coords = df['x'], df['y'], df['z']
#     except (IndexError, ValueError, KeyError) as e:
#         print(f"处理数据时出错: {e}")
#         print("请检查文件格式是否与预期一致。")
#         return

#     # --- 可视化设置 ---
#     plt.style.use('dark_background')
#     fig = plt.figure(figsize=(14, 10))
#     ax = fig.add_subplot(111, projection='3d')

#     # 初始化空的轨迹线、卫星点和时间文本
#     line, = ax.plot([], [], [], label='卫星轨道', color='cyan', lw=2)
#     point, = ax.plot([], [], [], 'o', color='red', markersize=8, label='卫星当前位置')
#     time_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes, color='white', fontsize=12)

#     # 在原点绘制一个示意性的地球
#     u = np.linspace(0, 2 * np.pi, 100)
#     v = np.linspace(0, np.pi, 100)
#     earth_radius = 6.371e6  # 地球平均半径（米）
#     earth_x = earth_radius * np.outer(np.cos(u), np.sin(v))
#     earth_y = earth_radius * np.outer(np.sin(u), np.sin(v))
#     earth_z = earth_radius * np.outer(np.ones(np.size(u)), np.cos(v))
#     ax.plot_surface(earth_x, earth_y, earth_z, color='blue', alpha=0.4, rstride=5, cstride=5)

#     # 预设坐标轴范围，避免动画过程中跳动
#     max_range = max(x_coords.max()-x_coords.min(), y_coords.max()-y_coords.min(), z_coords.max()-z_coords.min())
#     mid_x, mid_y, mid_z = (x_coords.max()+x_coords.min())*0.5, (y_coords.max()+y_coords.min())*0.5, (z_coords.max()+z_coords.min())*0.5
#     ax.set_xlim(mid_x - max_range * 0.5, mid_x + max_range * 0.5)
#     ax.set_ylim(mid_y - max_range * 0.5, mid_y + max_range * 0.5)
#     ax.set_zlim(mid_z - max_range * 0.5, mid_z + max_range * 0.5)

#     # 设置坐标轴标签和标题
#     ax.set_xlabel('X 坐标 (m)', fontsize=12)
#     ax.set_ylabel('Y 坐标 (m)', fontsize=12)
#     ax.set_zlabel('Z 坐标 (m)', fontsize=12)
#     ax.set_title('卫星三维轨迹动态可视化', fontsize=16)
#     ax.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
#     ax.legend(loc='upper right')
#     ax.grid(True, linestyle='--', alpha=0.5)

#     # 动画更新函数
#     def update(num):
#         # 计算当前动画帧对应的数据点索引
#         current_index = num * time_scale_factor
#         if current_index >= len(x_coords):
#             current_index = len(x_coords) - 1

#         # 更新轨迹线
#         line.set_data(x_coords[:current_index+1], y_coords[:current_index+1])
#         line.set_3d_properties(z_coords[:current_index+1])

#         # 更新卫星当前位置
#         point.set_data([x_coords[current_index]], [y_coords[current_index]])
#         point.set_3d_properties([z_coords[current_index]])
        
#         # 更新时间显示
#         current_time = df['datetime'].iloc[current_index]
#         time_text.set_text(f"时间: {current_time.strftime('%Y-%m-%d %H:%M:%S')} UTCG")

#         return line, point, time_text

#     # 计算动画的总帧数
#     frames = len(x_coords) // time_scale_factor + 1

#     # 创建动画
#     ani = FuncAnimation(fig, update, frames=frames, blit=False, interval=interval)

#     plt.tight_layout()
#     plt.show()

# # --- 主程序入口 ---
# if __name__ == '__main__':
#     file_to_visualize = 'H:\\old_D\Work\\501_平台\\数据-星图-第一版\\501\\scene2\\antisatellite1_Fixed_Position_Velocity.txt'
#     # 您可以调整 time_scale_factor 来控制动画速度
#     # 较小的值 = 较慢的动画；较大的值 = 较快的动画
#     visualize_animated_trajectory(file_to_visualize, time_scale_factor=5, interval=20)

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.animation import FuncAnimation
# import pandas as pd
# import numpy as np
# import io
# import os
# import glob
# from datetime import datetime

# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False

# def parse_trajectory_file(file_path):
#     """
#     解析单个卫星轨迹数据文件。

#     参数:
#     file_path (str): 数据文件的路径。
    
#     返回:
#     pandas.DataFrame or None: 包含解析后数据的DataFrame，如果文件无效则返回None。
#     """
#     try:
#         with open(file_path, 'r') as f:
#             file_content = f.read()

#         lines = file_content.strip().split('\n')
#         data_start_line_index = 0
#         for i, line in enumerate(lines):
#             if '-----------' in line:
#                 data_start_line_index = i + 1
#                 break
        
#         if data_start_line_index == 0:
#             print(f"警告: 在文件 {os.path.basename(file_path)} 中找不到数据分隔符。")
#             return None

#         data_lines = lines[data_start_line_index:]
#         if not data_lines:
#             print(f"警告: 在文件 {os.path.basename(file_path)} 中没有找到数据行。")
#             return None

#         data_io = io.StringIO()
#         data_io.write("day month year time x y z vx vy vz\n")
#         data_io.write('\n'.join(data_lines))
#         data_io.seek(0)

#         df = pd.read_csv(data_io, delim_whitespace=True)
#         df['datetime'] = pd.to_datetime(df['day'].astype(str) + ' ' + df['month'] + ' ' + df['year'].astype(str) + ' ' + df['time'])
        
#         # 从文件名中提取卫星名称
#         satellite_name = os.path.basename(file_path).split('.')[0]
#         df['satellite'] = satellite_name
        
#         return df
#     except Exception as e:
#         print(f"读取或解析文件 {os.path.basename(file_path)} 时出错: {e}")
#         return None

# def visualize_multiple_animated_trajectories(directory_path, time_scale_factor=10, interval=50):
#     """
#     读取一个文件夹中的所有轨迹文件，并以动画形式进行3D可视化。

#     参数:
#     directory_path (str): 包含轨迹数据文件的文件夹路径。
#     time_scale_factor (int): 时间缩放系数。
#     interval (int): 动画帧之间的延迟（毫秒）。
#     """
#     # 查找目录中所有的 .txt 文件
#     file_paths = glob.glob(os.path.join(directory_path, '*.txt'))
    
#     if not file_paths:
#         print(f"错误: 在文件夹 '{directory_path}' 中没有找到 .txt 格式的轨迹文件。")
#         return

#     # 解析所有文件并存储在列表中
#     all_data = [parse_trajectory_file(fp) for fp in file_paths]
#     all_data = [df for df in all_data if df is not None] # 过滤掉解析失败的文件

#     if not all_data:
#         print("错误: 未能成功加载任何轨迹数据。")
#         return

#     # --- 可视化设置 ---
#     plt.style.use('dark_background')
#     fig = plt.figure(figsize=(16, 12))
#     ax = fig.add_subplot(111, projection='3d')

#     # 确定所有轨道的统一坐标范围
#     full_df = pd.concat(all_data, ignore_index=True)
#     x_coords, y_coords, z_coords = full_df['x'], full_df['y'], full_df['z']
#     max_range = max(x_coords.max()-x_coords.min(), y_coords.max()-y_coords.min(), z_coords.max()-z_coords.min())
#     mid_x, mid_y, mid_z = (x_coords.max()+x_coords.min())*0.5, (y_coords.max()+y_coords.min())*0.5, (z_coords.max()+z_coords.min())*0.5
#     ax.set_xlim(mid_x - max_range * 0.5, mid_x + max_range * 0.5)
#     ax.set_ylim(mid_y - max_range * 0.5, mid_y + max_range * 0.5)
#     ax.set_zlim(mid_z - max_range * 0.5, mid_z + max_range * 0.5)
    
#     # 绘制地球
#     u = np.linspace(0, 2 * np.pi, 100)
#     v = np.linspace(0, np.pi, 100)
#     earth_radius = 6.371e6
#     earth_x = earth_radius * np.outer(np.cos(u), np.sin(v))
#     earth_y = earth_radius * np.outer(np.sin(u), np.sin(v))
#     earth_z = earth_radius * np.outer(np.ones(np.size(u)), np.cos(v))
#     ax.plot_surface(earth_x, earth_y, earth_z, color='blue', alpha=0.4, rstride=5, cstride=5)

#     # 为每条轨道创建绘图对象
#     colors = plt.cm.get_cmap('gist_rainbow', len(all_data))
#     lines, points = [], []
#     for i, df in enumerate(all_data):
#         line, = ax.plot([], [], [], label=df['satellite'].iloc[0], color=colors(i), lw=2)
#         point, = ax.plot([], [], [], 'o', color=colors(i), markersize=8)
#         lines.append(line)
#         points.append(point)

#     time_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes, color='white', fontsize=12)

#     # 设置坐标轴标签和标题
#     ax.set_xlabel('X 坐标 (m)', fontsize=12)
#     ax.set_ylabel('Y 坐标 (m)', fontsize=12)
#     ax.set_zlabel('Z 坐标 (m)', fontsize=12)
#     ax.set_title('多卫星三维轨迹动态可视化', fontsize=16)
#     ax.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
#     ax.legend(loc='upper right')
#     ax.grid(True, linestyle='--', alpha=0.5)

#     # 动画更新函数
#     def update(num):
#         # 计算当前帧对应的通用索引
#         current_index = num * time_scale_factor
        
#         # 遍历每条轨道并更新
#         for i, df in enumerate(all_data):
#             # 确定该轨道的当前索引，防止超出范围
#             idx = min(current_index, len(df) - 1)
            
#             x_data, y_data, z_data = df['x'], df['y'], df['z']
            
#             lines[i].set_data(x_data[:idx+1], y_data[:idx+1])
#             lines[i].set_3d_properties(z_data[:idx+1])
            
#             points[i].set_data([x_data[idx]], [y_data[idx]])
#             points[i].set_3d_properties([z_data[idx]])
        
#         # 更新时间（以最长的轨道为准）
#         longest_df = max(all_data, key=len)
#         time_idx = min(current_index, len(longest_df) - 1)
#         current_time = longest_df['datetime'].iloc[time_idx]
#         time_text.set_text(f"时间: {current_time.strftime('%Y-%m-%d %H:%M:%S')} UTCG")

#         return lines + points + [time_text]

#     # 计算动画总帧数（基于最长的轨道）
#     max_len = max(len(df) for df in all_data)
#     frames = max_len // time_scale_factor + 1
    
#     # 创建动画
#     ani = FuncAnimation(fig, update, frames=frames, blit=False, interval=interval)

#     plt.tight_layout()
#     plt.show()

# if __name__ == '__main__':
#     # 定义包含轨迹文件的文件夹路径
#     directory_to_visualize = 'H:\\old_D\Work\\501_平台\\数据-星图-第一版\\501\\scene2\\show'
    
    
#     # 运行可视化
#     visualize_multiple_animated_trajectories(directory_to_visualize, time_scale_factor=2, interval=20)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import pandas as pd
import numpy as np
import io
import os
import glob
from datetime import datetime

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def parse_trajectory_file(file_path):
    """
    解析单个卫星轨迹数据文件。
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            file_content = f.read()

        lines = file_content.strip().split('\n')
        data_start_line_index = 0
        for i, line in enumerate(lines):
            if '-----------' in line:
                data_start_line_index = i + 1
                break
        
        if data_start_line_index == 0:
            print(f"警告: 在文件 {os.path.basename(file_path)} 中找不到数据分隔符。")
            return None

        data_lines = lines[data_start_line_index:]
        if not data_lines:
            print(f"警告: 在文件 {os.path.basename(file_path)} 中没有找到数据行。")
            return None

        data_io = io.StringIO()
        data_io.write("day month year time x y z vx vy vz\n")
        data_io.write('\n'.join(data_lines))
        data_io.seek(0)

        df = pd.read_csv(data_io, delim_whitespace=True)
        df['datetime'] = pd.to_datetime(df['day'].astype(str) + ' ' + df['month'] + ' ' + df['year'].astype(str) + ' ' + df['time'])
        
        satellite_name = os.path.basename(file_path).split('.')[0]
        df['satellite'] = satellite_name
        
        return df
    except Exception as e:
        print(f"读取或解析文件 {os.path.basename(file_path)} 时出错: {e}")
        return None

def visualize_multiple_animated_trajectories(directory_path, group_map=None, time_scale_factor=10, interval=50):
    """
    读取一个文件夹中的所有轨迹文件，并以动画形式进行3D可视化。
    图例在窗口右上角，坐标信息在左上角显示。
    """
    group_map = group_map or {}
    file_paths = glob.glob(os.path.join(directory_path, '*.txt'))
    
    if not file_paths:
        print(f"错误: 在文件夹 '{directory_path}' 中没有找到 .txt 格式的轨迹文件。")
        return

    all_data_raw = [parse_trajectory_file(fp) for fp in file_paths]
    all_data_raw = [df for df in all_data_raw if df is not None]

    if not all_data_raw:
        print("错误: 未能成功加载任何轨迹数据。")
        return

    # --- 单位转换 (m -> km, m/s -> km/s) ---
    all_data = []
    for df in all_data_raw:
        df_new = df.copy()
        for col in ['x', 'y', 'z']:
            df_new[col] = df_new[col] / 1000
        for col in ['vx', 'vy', 'vz']:
            df_new[col] = df_new[col] / 1000
        all_data.append(df_new)
        
    # --- 可视化设置 ---
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(20, 10)) 
    ax = fig.add_subplot(111, projection='3d')
    fig.subplots_adjust(left=0.25, right=0.8, top=1.0, bottom=0.0)

    # 确定所有轨道的统一坐标范围
    full_df = pd.concat(all_data, ignore_index=True)
    x_coords, y_coords, z_coords = full_df['x'], full_df['y'], full_df['z']
    max_range = max(x_coords.max()-x_coords.min(), y_coords.max()-y_coords.min(), z_coords.max()-z_coords.min())
    mid_x, mid_y, mid_z = (x_coords.max()+x_coords.min())*0.5, (y_coords.max()+y_coords.min())*0.5, (z_coords.max()+z_coords.min())*0.5
    ax.set_xlim(mid_x - max_range * 0.5, mid_x + max_range * 0.5)
    ax.set_ylim(mid_y - max_range * 0.5, mid_y + max_range * 0.5)
    ax.set_zlim(mid_z - max_range * 0.5, mid_z + max_range * 0.5)
    
    # 绘制地球 (使用km单位)
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    earth_radius = 6.371e3 # 单位: km
    earth_x = earth_radius * np.cos(u) * np.sin(v)
    earth_y = earth_radius * np.sin(u) * np.sin(v)
    earth_z = earth_radius * np.cos(v)
    ax.plot_surface(earth_x, earth_y, earth_z, color='royalblue', alpha=0.6, rstride=1, cstride=1)

    # --- 分组、分配颜色并创建绘图对象 ---
    group1_sats = {name for name, group in group_map.items() if group == 'group1'}
    group2_sats = {name for name, group in group_map.items() if group == 'group2'}
    
    # 为每个组生成颜色
    colors_g1 = plt.cm.get_cmap('Reds', len(group1_sats) + 2)
    colors_g2 = plt.cm.get_cmap('Blues', len(group2_sats) + 2)
    
    lines, points = [], []
    handles1, labels1 = [], []
    handles2, labels2 = [], []
    
    g1_idx, g2_idx = 0, 0
    
    for df in all_data:
        satellite_name = df['satellite'].iloc[0]
        
        if satellite_name in group1_sats:
            color = colors_g1(0.6 + g1_idx * 0.3 / (len(group1_sats) or 1)) 
            g1_idx += 1
        elif satellite_name in group2_sats:
            color = colors_g2(0.6 + g2_idx * 0.3 / (len(group2_sats) or 1))
            g2_idx += 1
        else:
            color = 'grey'

        line, = ax.plot([], [], [], color=color, lw=2, label=satellite_name)
        point, = ax.plot([], [], [], 'o', color=color, markersize=8)
        lines.append(line)
        points.append(point)

        # 收集图例句柄
        if satellite_name in group1_sats:
            handles1.append(line)
            labels1.append(satellite_name)
        elif satellite_name in group2_sats:
            handles2.append(line)
            labels2.append(satellite_name)
        # 未分组的不创建图例

    # 初始化左上角的信息文本框
    info_text = fig.text(0.01, 0.98, '', color='white', fontsize=15,
                          verticalalignment='top', bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))

    # 设置坐标轴标签和标题
    ax.set_xlabel('X 坐标 (km)', fontsize=20, labelpad=20)
    ax.set_ylabel('Y 坐标 (km)', fontsize=20, labelpad=20)
    ax.set_zlabel('Z 坐标 (km)', fontsize=20, labelpad=20)
    ax.set_title('天基态势生成与智能博弈对抗', fontsize=30, y=0.99)
    ax.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
    ax.grid(True, linestyle='--', alpha=0.5)

    # ax.view_init(elev=30, azim=-60)  # 这是一个类似默认视角的好选择
    # ax.view_init(elev=30, azim=60)  # 这是一个类似默认视角的好选择
    ax.view_init(elev=90, azim=0)      # 示例：正上方向下的俯视角
    # ax.view_init(elev=0, azim=-90)     # 示例：从Y轴正方向看的侧视角
    # ax.view_init(elev=45, azim=45)      # 示例：正上方向下的俯视角

    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.tick_params(axis='z', labelsize=20)
    ax.xaxis.offsetText.set_fontsize(20)
    ax.yaxis.offsetText.set_fontsize(20)
    ax.zaxis.offsetText.set_fontsize(20)
    ax.grid(True, linestyle='--', alpha=0.5)
    # --- 创建两个独立的图例 ---
    legend1 = fig.legend(handles=handles1, labels=labels1, loc='upper right', 
                         bbox_to_anchor=(0.98, 0.95), fontsize=20, title="我方卫星", title_fontsize=30)
    legend2 = fig.legend(handles=handles2, labels=labels2, loc='upper right',
                         bbox_to_anchor=(0.98, 0.35), fontsize=20, title="敌方卫星",title_fontsize=30)


    # 动画更新函数
    def update(num):
        current_index = num * time_scale_factor
        info_strings = []
        
        for i, df in enumerate(all_data):
            idx = min(current_index, len(df) - 1)
            row = df.iloc[idx]
            
            x_data, y_data, z_data = df['x'], df['y'], df['z']
            
            lines[i].set_data(x_data[:idx+1], y_data[:idx+1])
            lines[i].set_3d_properties(z_data[:idx+1])
            
            points[i].set_data([row['x']], [row['y']])
            points[i].set_3d_properties([row['z']])

            info_strings.append(
                f"{row['satellite']}:\n"
                f"  Pos: ({row['x']:.2e}, {row['y']:.2e}, {row['z']:.2e}) km\n"
                f"  Vel: ({row['vx']:.2f}, {row['vy']:.2f}, {row['vz']:.2f}) km/s"
            )

        longest_df = max(all_data, key=len)
        time_idx = min(current_index, len(longest_df) - 1)
        current_time_str = longest_df['datetime'].iloc[time_idx].strftime('%Y-%m-%d %H:%M:%S')
        
        full_info_text = f"时间: {current_time_str} UTCG\n\n" + "\n\n".join(info_strings)
        info_text.set_text(full_info_text)
        
        return lines + points + [info_text]

    max_len = max(len(df) for df in all_data)
    frames = max_len // time_scale_factor + 1
    
    ani = FuncAnimation(fig, update, frames=frames, blit=False, interval=interval, repeat=False)

    plt.show()




if __name__ == '__main__':
    # directory_to_visualize = 'H:\\old_D\Work\\501_平台\\数据-星图-第一版\\501\\scene2\\show1'
    directory_to_visualize = 'H:\\old_D\Work\\501_平台\\数据-星图-第一版\\501\\quene'
    original_file_for_demo = 'antisatellite1_Fixed_Position_Velocity.txt'
    # custom_color_map = {
    #     '自主决策生成轨迹': {'color': 'red', 'label': '自主决策生成轨迹'},
    #     '人工干预计算轨迹': {'color': 'pink', 'label': '人工干预计算轨迹'},
    #     '我方观测星1': {'color': 'orange', 'label': '我方观测星'}, # 与satellite_2共享图例
    #     '我方观测星2': {'color': 'orange', 'label': '我方观测星'},
    #     '我方观测星3': {'color': 'orange', 'label': '我方观测星'},
    #     '敌方卫星轨迹': {'color': 'blue', 'label': '敌方卫星轨迹'},
    #     '敌方观测星': {'color': 'blue', 'label': '敌方卫星轨迹'},
    #     '敌方预计袭扰目标': {'color': 'purple', 'label': '敌方预计袭扰目标'},
    # }
    # group_assignments = {
    #     '防御卫星': 'group1',
    #     # '人工干预计算轨迹': 'group1',
    #     '我方观测星1': 'group1',
    #     '我方观测星2': 'group1',
    #     '我方观测星3': 'group1',
    #     '我方观测星4': 'group1',
    #     '我方观测星5': 'group1',
    #     '我方观测星6': 'group1',
    #     '我方观测星7': 'group1',
    #     '我方观测星8': 'group1',
    #     '我方观测星9': 'group1',
    #     '我方观测星10': 'group1',

    #     '我方观测星11': 'group1',
    #     '我方悬停卫星': 'group1',
    #     '敌方悬停星': 'group2',
    #     '敌方观测星1': 'group2',
    #     '敌方观测星2': 'group2',
    #     '敌方攻击星': 'group2',
    # }
    # group_assignments = {
    #     # 'blue_approach_Velocity': 'group1',
    #     # '人工干预计算轨迹': 'group1',
    #     'Blue_strike': 'group2',
    #     # 'Blue_strike_om1': 'group1',
    #     'Observation_backy': 'group2',
    #     '我方观测星4': 'group1',
    #     '我方观测星5': 'group1',
    #     '我方观测星6': 'group1',
    #     '我方观测星7': 'group1',
    #     '我方观测星8': 'group1',
    #     '我方观测星9': 'group1',
    #     '我方观测星10': 'group1',

    #     '我方悬停卫星': 'group1',
    #     'Observation1': 'group1',
    #     'Observation2': 'group1',
    #     'Observation3': 'group1',
    #     '敌方攻击星': 'group2',
    # }
    # group_assignments = {
    #     # 'blue_approach_Velocity': 'group1',
    #     # '人工干预计算轨迹': 'group1',
    #     # 'antisatellite1': 'group2',
    #     # # 'Blue_strike_om1': 'group1',
    #     # 'antisatellite2': 'group2',
    #     # 'antisatellite3': 'group2',
    #     '防御卫星': 'group1',
    #     # '我方观测星6': 'group1',
    #     # '我方观测星7': 'group1',
    #     '我方观测星9': 'group1',
    #     '我方观测星11': 'group1',
    #     '我方观测星10': 'group1',

    #     # '我方悬停卫星': 'group1',
    #     # 'Observation1': 'group1',
    #     # 'Observation2': 'group1',
    #     '敌方攻击星2': 'group2',
    #     '敌方攻击星': 'group2',
    # }
    group_assignments = {
        # 'blue_approach_Velocity': 'group1',
        # '人工干预计算轨迹': 'group1',
        # 'antisatellite1': 'group2',
        # # 'Blue_strike_om1': 'group1',
        # 'antisatellite2': 'group2',
        # 'antisatellite3': 'group2',
        # '防御卫星': 'group1',
        # '我方观测星6': 'group1',
        # '我方观测星7': 'group1',
        '1': 'group1',
        '2': 'group1',
        # '3': 'group1',
        '6': 'group1',
        '11': 'group2',
        # '20': 'group2',

        # '我方悬停卫星': 'group1',
        # 'Observation1': 'group1',
        # 'Observation2': 'group1',
        # '敌方攻击星2': 'group2',
        # '敌方攻击星': 'group2',
    }
    # 运行可视化
    visualize_multiple_animated_trajectories(
            directory_to_visualize,
            group_map=group_assignments, # 传递颜色地图
            time_scale_factor=10, 
            interval=2
        )
