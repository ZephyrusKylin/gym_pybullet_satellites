import numpy as np
import pybullet as p
import pybullet_data
import time
import math
from collections import deque

class SatelliteFormationSimulator:
    """
    一个用于在PyBullet中模拟卫星编队保持的仿真器。

    该仿真器模拟了三颗我方卫星组成一个三角形编队，
    用于观测一颗敌方卫星。当敌方卫星变轨时，我方
    卫星编队能够协同机动，以保持队形并持续跟踪。
    """

    def __init__(self, time_step=1.0/60.0, scale=0.000001):
        """
        初始化仿真器。

        参数:
            time_step (float): 仿真时间步长。
            scale (float): 真实世界到仿真世界的缩放比例。
        """
        # --- 物理和环境常量 ---
        self.G = 6.67430e-11  # 万有引力常数 (m^3 kg^-1 s^-2)
        self.M_EARTH = 5.972e24  # 地球质量 (kg)
        self.R_EARTH = 6371 * 1000  # 地球半径 (m)
        self.GEO_ALTITUDE = 35786 * 1000  # 地球同步轨道高度 (m)
        self.GEO_RADIUS = self.R_EARTH + self.GEO_ALTITUDE
        self.MU = self.G * self.M_EARTH # 地球标准引力常数

        # --- 仿真设置 ---
        self.scale = scale  # 真实世界到仿真的缩放比例
        self.time_step = time_step
        self.sim_time = 0.0

        # --- PyBullet 初始化 ---
        self.client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, 0)
        p.setRealTimeSimulation(0)

        # --- 创建场景 ---
        self._create_environment()

        # --- 编队控制参数 ---
        self.formation_side_length = 600 * 1000  # 编队三角形的期望边长 (m)，已增大
        self.kp_formation = 0.0005 # 编队保持比例增益
        self.kd_formation = 0.5    # 编队保持微分增益
        
        self.kp_tracking = 0.001 # 目标跟踪比例增益
        self.kd_tracking = 1.0   # 目标跟踪微分增益

        # --- 敌方机动参数 ---
        self.enemy_maneuver_start_time = 2000 # 敌方卫星开始机动的时间 (s)
        self.enemy_maneuver_duration = 200   # 敌方卫星机动持续时间 (s)
        self.enemy_maneuver_thrust = 15.0    # 敌方卫星机动推力 (m/s^2)
        self.enemy_is_maneuvering = False
        
        # --- 卫星初始化 ---
        self.num_friendly = 3
        self.friendly_ids = []
        self.friendly_positions = np.zeros((self.num_friendly, 3), dtype=float)
        self.friendly_velocities = np.zeros((self.num_friendly, 3), dtype=float)
        
        self.enemy_id = None
        self.enemy_position = np.zeros(3, dtype=float)
        self.enemy_velocity = np.zeros(3, dtype=float)

        # --- 轨迹/调试线 记录 ---
        self.trajectory_length = 500 # 轨迹点的最大数量
        self.friendly_trajectories = [deque(maxlen=self.trajectory_length) for _ in range(self.num_friendly)]
        self.enemy_trajectory = deque(maxlen=self.trajectory_length)

        # 用于存储调试线的ID，以便更新或删除它们
        self.formation_line_ids = []
        self.tracking_line_id = -1
        self.friendly_trajectory_line_ids = [deque(maxlen=self.trajectory_length) for _ in range(self.num_friendly)]
        self.enemy_trajectory_line_ids = deque(maxlen=self.trajectory_length)

        self._initialize_satellites()


    def _create_environment(self):
        """创建仿真环境，包括地球和相机设置。"""
        # 创建地球
        earth_col_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=self.R_EARTH * self.scale)
        self.earth_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=earth_col_shape, basePosition=[0, 0, 0])
        p.changeVisualShape(self.earth_id, -1, rgbaColor=[0.2, 0.4, 0.8, 1.0], specularColor=[0,0,0])

        # 设置初始相机，后续会在run()中动态更新
        p.resetDebugVisualizerCamera(
            cameraDistance=self.GEO_RADIUS * self.scale * 3.5,
            cameraYaw=45,
            cameraPitch=-30,
            cameraTargetPosition=[0, 0, 0]
        )

    def _initialize_satellites(self):
        """初始化我方和敌方卫星在GEO轨道附近一个局部区域内的位置、速度和PyBullet对象。"""
        # --- 将所有卫星放置在GEO轨道附近的一个局部区域 ---

        # 1. 设置参考中心（敌方卫星）的轨道
        enemy_radius = self.GEO_RADIUS
        enemy_speed = math.sqrt(self.MU / enemy_radius)
        self.enemy_position = np.array([enemy_radius, 0, 0], dtype=float)
        self.enemy_velocity = np.array([0, enemy_speed, 0], dtype=float)

        # 2. 定义我方卫星相对于敌方卫星的局部位置
        # 它们将位于一个平面上，该平面的法线垂直于地心与敌方的连线。
        # 敌方与地心连线是 X 轴。我们选择平面法线为 Y 轴，因此平面是 X-Z 平面。
        L = self.formation_side_length
        h = L * math.sqrt(3) / 2.0

        # 将敌方卫星置于我方三角形编队的中心
        # 调整了顶点位置，使其不那么“正”
        rel_pos = np.array([
            [0,      0, 0.7 * h],    # 顶点1 (上方，稍微压低)
            [-L/2.1, 0, -0.3 * h],  # 顶点2 (左下，稍微调整)
            [L/2.0,  0, -0.4 * h]   # 顶点3 (右下，稍微调整)
        ], dtype=float)

        # 3. 设置我方卫星的绝对位置和速度
        for i in range(self.num_friendly):
            self.friendly_positions[i] = self.enemy_position + rel_pos[i]
            # 初始速度与敌方卫星相同，控制系统将进行修正
            self.friendly_velocities[i] = self.enemy_velocity.copy()

        # 4. 创建 PyBullet 对象
        friendly_col_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=100 * 1000 * self.scale)
        for i in range(self.num_friendly):
            body_id = p.createMultiBody(baseMass=100, baseCollisionShapeIndex=friendly_col_shape,
                                        basePosition=self.friendly_positions[i] * self.scale)
            p.changeVisualShape(body_id, -1, rgbaColor=[0, 1, 0, 1]) # 绿色
            self.friendly_ids.append(body_id)

        enemy_col_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=100 * 1000 * self.scale)
        self.enemy_id = p.createMultiBody(baseMass=100, baseCollisionShapeIndex=enemy_col_shape,
                                          basePosition=self.enemy_position * self.scale)
        p.changeVisualShape(self.enemy_id, -1, rgbaColor=[1, 0, 0, 1]) # 红色

    def _calculate_gravity(self, position):
        """计算给定位置的引力加速度。"""
        r_norm = np.linalg.norm(position)
        if r_norm == 0:
            return np.zeros(3)
        return -self.MU * position / (r_norm ** 3)

    def _formation_control(self):
        """计算保持编队所需的控制推力。"""
        thrusts = np.zeros((self.num_friendly, 3))
        
        # 计算当前编队的质心和质心速度
        centroid_pos = np.mean(self.friendly_positions, axis=0)
        centroid_vel = np.mean(self.friendly_velocities, axis=0)

        # --- 目标跟踪推力 (作用于整个编队) ---
        # 计算从编队质心到敌方卫星的向量
        tracking_error_pos = self.enemy_position - centroid_pos
        tracking_error_vel = self.enemy_velocity - centroid_vel
        
        # PD 控制器计算跟踪推力
        tracking_thrust = self.kp_tracking * tracking_error_pos + self.kd_tracking * tracking_error_vel
        
        # 将跟踪推力均匀分配给每颗卫星
        for i in range(self.num_friendly):
            thrusts[i] += tracking_thrust / self.num_friendly

        # --- 编队保持推力 (内部力) ---
        for i in range(self.num_friendly):
            for j in range(i + 1, self.num_friendly):
                # 计算卫星i和j之间的相对位置和速度
                d_pos = self.friendly_positions[j] - self.friendly_positions[i]
                d_vel = self.friendly_velocities[j] - self.friendly_velocities[i]
                
                # 计算距离误差
                dist = np.linalg.norm(d_pos)
                dist_error = dist - self.formation_side_length
                
                # 计算指向卫星j的单位向量
                d_hat = d_pos / dist if dist != 0 else np.zeros(3)
                
                # PD 控制器计算编队保持力
                force_magnitude = self.kp_formation * dist_error + self.kd_formation * np.dot(d_vel, d_hat)
                force_vec = force_magnitude * d_hat
                
                # 将力作用于卫星i和j
                thrusts[i] += force_vec
                thrusts[j] -= force_vec
                
        return thrusts

    def _enemy_maneuver(self):
        """控制敌方卫星的机动。"""
        if self.sim_time > self.enemy_maneuver_start_time and \
           self.sim_time < (self.enemy_maneuver_start_time + self.enemy_maneuver_duration):
            if not self.enemy_is_maneuvering:
                print(f"INFO: Enemy satellite starting maneuver at t={self.sim_time:.2f}s")
                self.enemy_is_maneuvering = True
            
            # 施加一个平面外 (cross-track) 的推力
            # 在我们的初始设置中，这是世界坐标系的Z轴方向
            cross_track_dir = np.array([0, 0, 1.0])
            return cross_track_dir * self.enemy_maneuver_thrust
        
        if self.enemy_is_maneuvering and self.sim_time >= (self.enemy_maneuver_start_time + self.enemy_maneuver_duration):
            print(f"INFO: Enemy satellite finished maneuver at t={self.sim_time:.2f}s")
            self.enemy_is_maneuvering = False

        return np.zeros(3)

    def _update_physics(self):
        """更新所有卫星的物理状态。"""
        # --- 更新我方卫星 ---
        formation_thrusts = self._formation_control()
        for i in range(self.num_friendly):
            gravity = self._calculate_gravity(self.friendly_positions[i])
            total_accel = gravity + formation_thrusts[i] / 100.0 # a = F/m
            
            self.friendly_velocities[i] += total_accel * self.time_step
            self.friendly_positions[i] += self.friendly_velocities[i] * self.time_step
            self.friendly_trajectories[i].append(self.friendly_positions[i].copy())

        # --- 更新敌方卫星 ---
        enemy_thrust = self._enemy_maneuver()
        enemy_gravity = self._calculate_gravity(self.enemy_position)
        enemy_accel = enemy_gravity + enemy_thrust / 100.0 # a = F/m

        self.enemy_velocity += enemy_accel * self.time_step
        self.enemy_position += self.enemy_velocity * self.time_step
        self.enemy_trajectory.append(self.enemy_position.copy())

    def _update_visuals(self):
        """更新PyBullet中的视觉对象。"""
        # 更新卫星模型位置
        for i in range(self.num_friendly):
            p.resetBasePositionAndOrientation(self.friendly_ids[i], self.friendly_positions[i] * self.scale, [0, 0, 0, 1])
        p.resetBasePositionAndOrientation(self.enemy_id, self.enemy_position * self.scale, [0, 0, 0, 1])

        # --- 更新或绘制动态调试线 ---
        # 1. 删除旧的编队和跟踪线
        for line_id in self.formation_line_ids:
            p.removeUserDebugItem(line_id)
        self.formation_line_ids.clear()
        if self.tracking_line_id != -1:
            p.removeUserDebugItem(self.tracking_line_id)
            self.tracking_line_id = -1

        # 2. 绘制新的编队线并记录ID
        for i in range(self.num_friendly):
            for j in range(i + 1, self.num_friendly):
                line_id = p.addUserDebugLine(self.friendly_positions[i] * self.scale,
                                       self.friendly_positions[j] * self.scale,
                                       lineColorRGB=[0, 1, 0], lineWidth=2)
                self.formation_line_ids.append(line_id)
        
        # 3. 绘制新的跟踪线并记录ID
        centroid_pos = np.mean(self.friendly_positions, axis=0)
        self.tracking_line_id = p.addUserDebugLine(centroid_pos * self.scale,
                                                   self.enemy_position * self.scale,
                                                   lineColorRGB=[1, 1, 0], lineWidth=3)

        # --- 添加新的轨迹线段 ---
        # 4. 我方卫星轨迹
        for i in range(self.num_friendly):
            if len(self.friendly_trajectories[i]) > 1:
                # 如果轨迹线超长，则删除最旧的线段
                if len(self.friendly_trajectory_line_ids[i]) >= self.trajectory_length:
                    old_line_id = self.friendly_trajectory_line_ids[i].popleft()
                    p.removeUserDebugItem(old_line_id)
                
                # 绘制新的线段并存储其ID
                new_line_id = p.addUserDebugLine(self.friendly_trajectories[i][-2] * self.scale,
                                               self.friendly_trajectories[i][-1] * self.scale,
                                               lineColorRGB=[0.5, 1, 0.5], lineWidth=1)
                self.friendly_trajectory_line_ids[i].append(new_line_id)

        # 5. 敌方卫星轨迹
        if len(self.enemy_trajectory) > 1:
            if len(self.enemy_trajectory_line_ids) >= self.trajectory_length:
                old_line_id = self.enemy_trajectory_line_ids.popleft()
                p.removeUserDebugItem(old_line_id)
            
            new_line_id = p.addUserDebugLine(self.enemy_trajectory[-2] * self.scale,
                                           self.enemy_trajectory[-1] * self.scale,
                                           lineColorRGB=[1, 0.5, 0], lineWidth=1)
            self.enemy_trajectory_line_ids.append(new_line_id)


    def run(self):
        """运行主仿真循环。"""
        try:
            # 为局部视角设置相机参数
            cam_dist = self.formation_side_length * self.scale * 5.0 # 5倍编队边长
            cam_pitch = -35
            cam_yaw = 50

            while True:
                self._update_physics()
                self._update_visuals()

                # 更新相机以跟随编队中心
                centroid_pos = np.mean(self.friendly_positions, axis=0)
                # 平滑地更新yaw角，使其看起来像在环绕
                cam_yaw += 0.05 
                p.resetDebugVisualizerCamera(
                    cameraDistance=cam_dist,
                    cameraYaw=cam_yaw,
                    cameraPitch=cam_pitch,
                    cameraTargetPosition=centroid_pos * self.scale
                )
                
                p.stepSimulation()
                time.sleep(self.time_step)
                self.sim_time += self.time_step

        except p.error as e:
            print(f"PyBullet error: {e}")
        finally:
            p.disconnect()

if __name__ == "__main__":
    simulator = SatelliteFormationSimulator()
    simulator.run()
