import numpy as np
import pybullet as p
import pybullet_data
import time
import math


class BaseOrbitSimulator:
    """Base class for simulating multi-satellite systems in PyBullet."""

    def __init__(self, num_satellites=1, earth_radius=6371, gravitational_constant=398600, scale=0.01, satellite_scale=1.0, track_visualization=False):
        """
        Initialize the orbit simulator.

        Parameters:
        - num_satellites: int, number of satellites to simulate.
        - earth_radius: float, radius of the Earth in kilometers.
        - gravitational_constant: float, standard gravitational parameter (km^3/s^2).
        - scale: float, scaling factor for visualization (e.g., 0.01 for km to meters).
        - satellite_scale: float, scaling factor for satellite visualization size.
        - track_visualization: bool, whether to visualize satellite tracks.
        """
        self.num_satellites = num_satellites + 1  # Add one GEO satellite
        self.earth_radius = earth_radius
        self.gravitational_constant = gravitational_constant
        self.scale = scale
        self.satellite_scale = satellite_scale
        self.track_visualization = track_visualization

        # PyBullet setup
        self.client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, 0)
        p.loadURDF("plane.urdf")

        # Create Earth as a fixed sphere at the origin
        self.earth_id = p.createCollisionShape(p.GEOM_SPHERE, radius=self.earth_radius * self.scale)
        earth_body_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=self.earth_id, basePosition=[0, 0, 0])

        # Change Earth's visual appearance to blue
        p.changeVisualShape(earth_body_id, -1, rgbaColor=[0, 0, 1, 1])

        # Initialize satellite states: position, velocity
        self.sat_positions = np.zeros((self.num_satellites, 3))  # (x, y, z)
        self.sat_velocities = np.zeros((self.num_satellites, 3))  # (vx, vy, vz)
        self.satellite_ids = []
        self.track_lines = [] if track_visualization else None

        # Initialize default circular orbits
        for i in range(self.num_satellites):
            if i < self.num_satellites - 1:
                altitude = 3000 + i * 3000  # Increase spacing between red satellites
                orbit_radius = self.earth_radius + altitude

                if i == 1:
                    inclination = 15  # Satellite 2 inclination 15 degrees
                elif i == 2:
                    inclination = 45  # Satellite 3 inclination 45 degrees
                else:
                    inclination = 0  # Satellite 1 remains at 0 degrees

                inclination_rad = np.radians(inclination)
                self.sat_positions[i] = [
                    orbit_radius * np.cos(inclination_rad),
                    0,
                    orbit_radius * np.sin(inclination_rad),
                ]
                orbital_velocity = np.sqrt(self.gravitational_constant / orbit_radius)  # km/s
                self.sat_velocities[i] = [
                    0,
                    orbital_velocity * np.cos(inclination_rad),
                    -orbital_velocity * np.sin(inclination_rad),
                ]

                # Create satellite visuals in PyBullet
                satellite_id = p.createCollisionShape(p.GEOM_SPHERE, radius=10 * self.scale * self.satellite_scale)
                body_id = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=satellite_id, basePosition=self.sat_positions[i] * self.scale)
                p.changeVisualShape(body_id, -1, rgbaColor=[1, 0, 0, 1])  # Satellites are red
                self.satellite_ids.append(body_id)
            else:
                # GEO satellite
                geo_altitude = 35786  # GEO altitude in kilometers
                geo_orbit_radius = self.earth_radius + geo_altitude
                self.sat_positions[i] = [geo_orbit_radius, 0, 0]
                geo_velocity = np.sqrt(self.gravitational_constant / geo_orbit_radius)  # GEO orbital velocity
                self.sat_velocities[i] = [0, geo_velocity, 0]

                # Create GEO satellite visuals in PyBullet
                satellite_id = p.createCollisionShape(p.GEOM_SPHERE, radius=15 * self.scale * self.satellite_scale)
                body_id = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=satellite_id, basePosition=self.sat_positions[i] * self.scale)
                p.changeVisualShape(body_id, -1, rgbaColor=[1, 1, 0, 1])  # GEO satellite is yellow
                self.satellite_ids.append(body_id)

            # Initialize track visualization if enabled
            if self.track_visualization:
                line_id = p.addUserDebugLine(lineFromXYZ=self.sat_positions[i] * self.scale,
                                             lineToXYZ=self.sat_positions[i] * self.scale,
                                             lineColorRGB=[1, 1, 0],
                                             lineWidth=1.0)
                self.track_lines.append(line_id)

        # Adjust the camera to view the entire system
        max_distance = (self.earth_radius + 36000) * self.scale
        p.resetDebugVisualizerCamera(cameraDistance=max_distance, 
                                     cameraYaw=0, 
                                     cameraPitch=-30, 
                                     cameraTargetPosition=[0, 0, 0])

    def step(self, thrusts, timestep=1):
        """
        Advance the simulation by one timestep.

        Parameters:
        - thrusts: ndarray of shape (num_satellites, 3), thrust vectors for each satellite.
        - timestep: float, time increment in seconds.
        """
        for i in range(self.num_satellites):
            position = self.sat_positions[i]
            velocity = self.sat_velocities[i]

            # Compute gravitational force
            distance = np.linalg.norm(position)
            print(f"Satellite {i} Distance: {distance:.2f} km")
            if distance <= self.earth_radius:
                print(f"Error: Satellite {i} has crashed into the Earth. Distance: {distance:.2f} km")
                continue

            gravity_force = -self.gravitational_constant * position / distance**3  # km/s^2

            # Update velocity with gravity and thrust
            acceleration = gravity_force + thrusts[i]
            velocity += acceleration * timestep

            # Update position with velocity
            position += velocity * timestep

            # Update state
            self.sat_positions[i] = position
            self.sat_velocities[i] = velocity

            # Update PyBullet position
            p.resetBasePositionAndOrientation(self.satellite_ids[i], position * self.scale, [0, 0, 0, 1])

            # Update track visualization if enabled
            if self.track_visualization:
                p.addUserDebugLine(lineFromXYZ=position * self.scale,
                                   lineToXYZ=position * self.scale,
                                   lineColorRGB=[1, 1, 0],
                                   lineWidth=1.0,
                                   replaceItemUniqueId=self.track_lines[i])

    def visualize_with_transfer(self, steps=100, timestep=10, target_orbit=35786):
        """
        Visualize the simulation with one satellite transferring to GEO orbit.

        Parameters:
        - steps: int, number of simulation steps.
        - timestep: float, time increment per step in seconds.
        - target_orbit: float, target orbit radius in kilometers.
        """
        thrusts = np.zeros((self.num_satellites, 3))
        target_radius = self.earth_radius + target_orbit
        target_velocity = np.sqrt(self.gravitational_constant / target_radius)  # GEO orbital velocity
        epsilon_radius = 0.1  # Tolerance for radius
        epsilon_velocity = 0.01  # Tolerance for velocity

        for _ in range(steps):
            # Compute thrust for the first satellite
            current_radius = np.linalg.norm(self.sat_positions[0])
            current_velocity = np.linalg.norm(self.sat_velocities[0])

            # Safety check: prevent crashing into Earth
            if current_radius <= self.earth_radius:
                print("Warning: Satellite is too close to Earth. Stopping simulation.")
                break

            # Check if radius or velocity conditions are met
            radius_condition = abs(current_radius - target_radius) < epsilon_radius
            velocity_condition = abs(current_velocity - target_velocity) < epsilon_velocity

            if radius_condition and velocity_condition:
                print(f"Satellite has reached GEO orbit: Radius = {current_radius:.2f} km, Velocity = {current_velocity:.2f} km/s")
                thrusts[0] = np.array([0, 0, 0])  # Stop thrust
                continue

            # Calculate orbital angular momentum direction (h = r x v)
            angular_momentum = np.cross(self.sat_positions[0], self.sat_velocities[0])
            orbital_plane_normal = angular_momentum / np.linalg.norm(angular_momentum)

            # Calculate the ideal tangential velocity direction (perpendicular to r and normal)
            ideal_tangential_direction = np.cross(orbital_plane_normal, self.sat_positions[0])
            ideal_tangential_direction /= np.linalg.norm(ideal_tangential_direction)

                        # Calculate velocity correction to align with ideal tangential direction
            velocity_projection = np.dot(self.sat_velocities[0], ideal_tangential_direction)
            correction_vector = velocity_projection * ideal_tangential_direction - self.sat_velocities[0]

            # Calculate thrust for velocity correction and speed adjustment
            delta_v = target_velocity - velocity_projection

            # Dead-zone adjustment for small delta_v
            if abs(delta_v) < epsilon_velocity:
                delta_v = np.sign(delta_v) * 0.01  # Apply a small constant thrust near the target velocity

            thrust_direction = correction_vector + delta_v * ideal_tangential_direction
            thrust_magnitude = np.linalg.norm(thrust_direction)

            if thrust_magnitude > 0:
                thrust_direction /= thrust_magnitude  # Normalize thrust direction

            # Apply proportional control and limit thrust magnitude
            thrusts[0] = thrust_direction * min(0.01, thrust_magnitude)  # Limit thrust


            self.step(thrusts, timestep)
            time.sleep(0.01)
    def visualize_with_inclination_change(self, steps=100, timestep=10, target_orbit=35786, inclination=20, transfer_time=10):
        """
        Visualize the simulation with one satellite transferring to a new orbit with a specified inclination.

        Parameters:
        - steps: int, number of simulation steps.
        - timestep: float, time increment per step in seconds.
        - target_orbit: float, target orbit radius in kilometers.
        - inclination: float, target orbit inclination in degrees.
        - transfer_time: float, time for the satellite to transfer to the new orbit in seconds.
        """
        # Initialize thrusts
        thrusts = np.zeros((self.num_satellites, 3))

        # Calculate target orbit parameters
        target_radius = self.earth_radius + target_orbit
        target_velocity = np.sqrt(self.gravitational_constant / target_radius)  # km/s
        inclination_rad = np.radians(inclination)

        # Calculate transfer time steps
        transfer_steps = int(transfer_time / timestep)

        # Simulate the transfer
        for i in range(steps):
            # Calculate current radius and velocity
            current_radius = np.linalg.norm(self.sat_positions[0])
            current_velocity = np.linalg.norm(self.sat_velocities[0])

            # Check if transfer is complete
            if i < transfer_steps:
                # Calculate the inclination change
                inclination_change = inclination_rad / transfer_steps

                # Calculate the new velocity direction
                new_velocity_direction = np.array([
                    np.cos(inclination_change) * self.sat_velocities[0][0] - np.sin(inclination_change) * self.sat_velocities[0][2],
                    self.sat_velocities[0][1],
                    np.sin(inclination_change) * self.sat_velocities[0][0] + np.cos(inclination_change) * self.sat_velocities[0][2]
                ])

                # Calculate the thrust direction
                thrust_direction = new_velocity_direction - self.sat_velocities[0]
                thrust_magnitude = np.linalg.norm(thrust_direction)

                if thrust_magnitude > 0:
                    thrust_direction /= thrust_magnitude  # Normalize thrust direction

                # Apply proportional control and limit thrust magnitude
                thrusts[0] = thrust_direction * min(0.01, thrust_magnitude)  # Limit thrust
            else:
                # No thrust after transfer is complete
                thrusts[0] = np.zeros(3)

            # Update satellite positions and velocities
            self.step(thrusts, timestep)

            # Sleep for a short period of time
            time.sleep(0.01)
def pos_simulation():
# 启动物理引擎
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    
    # 地球的实际半径（单位：千米）
    earth_radius = 6371  
    # 地球模型urdf的实际半径（单位：厘米）
    earth_model_radius = 0.5  
    # 缩放后的地球半径（单位：仿真中的尺寸）
    scaled_earth_radius = 6.371  # 这里设置为100单位，调整为合适的值

    # 计算urdf模型缩放因子
    scaling_factor = scaled_earth_radius / earth_model_radius  # 需要缩放的因子

    # 加载地球模型并进行缩放
    earth_id = p.loadURDF("sphere2.urdf", [0, 0, 0], globalScaling=scaling_factor)


    collision_shape_data = p.getCollisionShapeData(earth_id, -1)
    print(collision_shape_data)



    if earth_id != -1:
        print("地球模型加载成功！ID:", earth_id)
    else:
        print("地球模型加载失败！")
    p.resetBasePositionAndOrientation(earth_id, [0, 0, 0], [0, 0, 0, 1])
    p.setGravity(0, 0, 0)
    p.setRealTimeSimulation(0)


    # 设置相机视角，调整视距、角度和目标位置
    p.resetDebugVisualizerCamera(
        cameraDistance=50,  # 调整视距，根据场景大小适配
        cameraYaw=90,        # 水平旋转角度
        cameraPitch=-30,     # 俯仰角度
        cameraTargetPosition=[0, 0, 0]  # 相机对准地球的中心
    )

    # 模拟其他操作（例如加载卫星等）
    geo_altitude = 35786  # GEO轨道高度，单位：千米
    scaled_geo_altitude = 35.786

    geo_ratio = geo_altitude / scaled_geo_altitude

    orbital_radius = scaled_earth_radius + scaled_geo_altitude  # GEO轨道半径（仿真单位）

    satellite_start_pos = [orbital_radius, 0, 0]
    satellite_start_orientation = p.getQuaternionFromEuler([0, 0, 0])

    # 加载卫星模型并缩放
    satellite_scaling = 1  # 卫星的缩放因子
    satellite = p.loadURDF("cube_small.urdf", satellite_start_pos, satellite_start_orientation, globalScaling=satellite_scaling)

    # collision_shape_data_sat = p.getCollisionShapeData(satellite, -1)
    # print(collision_shape_data_sat)
    # pos, ori = p.getBasePositionAndOrientation(satellite)
    # print(pos, ori)

    if satellite != -1:
        print("卫星模型加载成功！ID:", satellite)
    else:
        print("卫星模型加载失败！")

    cone_mesh_path = "./mesh/LedCone_1.stl"
    cone_radius = 5
    cone_height = 10
    cone_mass = 0.5
    cone_scale = [0.5, 0.1, 0.1] 
    # cone_collision_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=cone_radius, height=cone_height)
    # # cone_visual_shape = p.createVisualShape(p.GEOM_CYLINDER, radius=cone_radius, height=cone_height, rgbaColor=[1, 0, 0, 1])
    # # cone_collision_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=cone_radius, length=cone_height)
    # cone_visual_shape = p.createVisualShape(p.GEOM_CYLINDER, radius=cone_radius, length=cone_height, rgbaColor=[1, 0, 0, 1])
    cone_collision_shape = p.createCollisionShape(p.GEOM_MESH, fileName=cone_mesh_path)
    cone_visual_shape = p.createVisualShape(p.GEOM_MESH, rgbaColor=[0, 0, 1, 1], specularColor=[1, 1, 1],
                                         meshScale=cone_scale,fileName=cone_mesh_path)


    # 计算锥体姿态，使其法线指向原点
    # 例如，如果锥体的位置在(1, 0, 0)，则法线应该朝向原点(0, 0, 0)
    robot_position = p.getBasePositionAndOrientation(satellite)[0]
    print("robot_position", robot_position)
    direction_to_origin = [0 - robot_position[0], 0 - robot_position[1], 0 - robot_position[2]]
    norm = math.sqrt(direction_to_origin[0]**2 + direction_to_origin[1]**2 + direction_to_origin[2]**2)
    direction_to_origin = [x / norm for x in direction_to_origin]  # 单位化向量

    # 锥体的朝向
    yaw = math.atan2(direction_to_origin[1], direction_to_origin[0])
    pitch = math.atan2(direction_to_origin[2], math.sqrt(direction_to_origin[0]**2 + direction_to_origin[1]**2))

    # 计算四元数来表示旋转
    cone_orientation = p.getQuaternionFromEuler([pitch, 0, yaw])

    # 创建锥体并将其附加到URDF模型
    cone_id = p.createMultiBody(baseMass=cone_mass, baseCollisionShapeIndex=cone_collision_shape, 
                                baseVisualShapeIndex=cone_visual_shape, basePosition=robot_position, 
                                baseOrientation=cone_orientation)
















    # 设置敌方卫星初始轨道参数
    initial_orbit_altitude = 20000  # 初始轨道高度（单位：km）
    initial_orbit_inclination = 15  # 初始轨道倾角（单位：°）
    initial_orbit_longitude = 0  # 初始轨道经度（单位：°）

    # 计算敌方卫星初始位置和速度
    initial_position = [0, 0, initial_orbit_altitude + earth_radius]
    initial_velocity = [0, math.sqrt(398600.5 / (initial_orbit_altitude + earth_radius)), 0]

    # 设置缩放后敌方卫星初始轨道参数
    initial_orbit_altitude = 20000  # 初始轨道高度（单位：km）
    initial_orbit_inclination = 15  # 初始轨道倾角（单位：°）
    initial_orbit_longitude = 0  # 初始轨道经度（单位：°）

    # 计算缩放后敌方卫星初始位置和速度
    initial_position = [0, 0, initial_orbit_altitude + earth_radius]
    initial_velocity = [0, math.sqrt(398600.5 / (initial_orbit_altitude + earth_radius)), 0]

    satellite_scaling = 1  # 卫星的缩放因子
    satellite_enemy = p.loadURDF("cube_small.urdf", satellite_start_pos, satellite_start_orientation, globalScaling=satellite_scaling)

    # collision_shape_data_sat = p.getCollisionShapeData(satellite, -1)
    # print(collision_shape_data_sat)
    # pos, ori = p.getBasePositionAndOrientation(satellite)
    # print(pos, ori)

    if satellite_enemy != -1:
        print("敌方卫星模型加载成功！ID:", satellite_enemy)
    else:
        print("敌方卫星模型加载失败！")
    # 模拟主循环
    time_step = 1 / 30  # 时间步长，30fps
    orbital_speed = math.sqrt(398600.5 / (orbital_radius * geo_ratio))  # GEO轨道速度，km/s
    speed_factor = 1000  # 调整卫星速度
    angular_speed = orbital_speed / (orbital_radius * geo_ratio) * speed_factor  # 角速度，rad/s
    angle = 0

    while True:
        # 更新卫星位置
        angle += angular_speed * time_step
        new_x = orbital_radius * math.cos(angle)
        new_y = orbital_radius * math.sin(angle)
        p.resetBasePositionAndOrientation(cone_id, [new_x, new_y, 0], satellite_start_orientation)

        # 绘制卫星轨迹
        if angle > 0:
            p.addUserDebugLine([new_x, new_y, 0], [orbital_radius * math.cos(angle - angular_speed * time_step), orbital_radius * math.sin(angle - angular_speed * time_step), 0], [1, 0, 0], 200)  # 红色轨迹
            # print("卫星位置:", new_x, new_y, 0)
            # print("卫星角度:", angle)

        
        # 获取地球的当前位置和方向
        position, orientation = p.getBasePositionAndOrientation(earth_id)
        # 重置地球的位置和方向为固定值
        p.resetBasePositionAndOrientation(earth_id, [0, 0, 0], [0, 0, 0, 1])


        # 步进仿真和暂停
        p.stepSimulation()
        time.sleep(time_step)
# Example usage
if __name__ == "__main__":
    # simulator = BaseOrbitSimulator(num_satellites=3, scale=0.001, satellite_scale=50.0, track_visualization=True)
    # simulator.visualize(steps=20000, timestep=10)

    # simulator = BaseOrbitSimulator(num_satellites=3, scale=0.001, satellite_scale=50.0, track_visualization=True)
    # simulator.visualize_with_transfer(steps=20000, timestep=10, target_orbit=35786)

    # simulator = BaseOrbitSimulator(num_satellites=3, scale=0.001, satellite_scale=50.0, track_visualization=True)
    # simulator.visualize_with_inclination_change(steps=20000, timestep=10, target_orbit=35786)

    pos_simulation()