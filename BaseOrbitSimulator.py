import numpy as np
import pybullet as p
import pybullet_data
import time

'''
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
                altitude = 1000 + i * 1000  # Increase spacing between red satellites
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

    # def step(self, thrusts, timestep=1):
    #     """
    #     Advance the simulation by one timestep.

    #     Parameters:
    #     - thrusts: ndarray of shape (num_satellites, 3), thrust vectors for each satellite.
    #     - timestep: float, time increment in seconds.
    #     """
    #     for i in range(self.num_satellites):
    #         position = self.sat_positions[i]
    #         velocity = self.sat_velocities[i]

    #         # Compute gravitational force
    #         distance = np.linalg.norm(position)
    #         gravity_force = -self.gravitational_constant * position / distance**3  # km/s^2

    #         # Update velocity with gravity and thrust
    #         acceleration = gravity_force + thrusts[i] / (self.earth_radius + 1)  # Ensure thrust scaling is consistent
    #         velocity += acceleration * timestep

    #         # Update position with velocity
    #         position += velocity * timestep

    #         # Update state
    #         self.sat_positions[i] = position
    #         self.sat_velocities[i] = velocity

    #         # Update PyBullet position
    #         p.resetBasePositionAndOrientation(self.satellite_ids[i], position * self.scale, [0, 0, 0, 1])

    #         # Update track visualization if enabled
    #         if self.track_visualization:
    #             p.addUserDebugLine(lineFromXYZ=position * self.scale,
    #                                lineToXYZ=position * self.scale,
    #                                lineColorRGB=[1, 1, 0],
    #                                lineWidth=1.0,
    #                                replaceItemUniqueId=self.track_lines[i])

    def visualize(self, steps=100, timestep=1):
        """
        Visualize the simulation in PyBullet.

        Parameters:
        - steps: int, number of simulation steps.
        - timestep: float, time increment per step in seconds.
        """
        for _ in range(steps):
            thrusts = np.zeros((self.num_satellites, 3))  # No thrust for this example
            self.step(thrusts, timestep)
            time.sleep(0.01)
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
            acceleration = gravity_force + thrusts[i] / 1.0  # Avoid unnecessary scaling in denominator
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
        epsilon = 0.1  # Tolerance for radius and velocity matching

        for _ in range(steps):
            # Compute thrust for the first satellite
            current_radius = np.linalg.norm(self.sat_positions[0])
            current_velocity = np.linalg.norm(self.sat_velocities[0])

            if abs(current_radius - target_radius) < epsilon and abs(current_velocity - target_velocity) < epsilon:
                print(f"Satellite has reached GEO orbit: Radius = {current_radius:.2f} km, Velocity = {current_velocity:.2f} km/s")
                thrusts[0] = np.array([0, 0, 0])  # Stop thrust
                continue

            delta_v = target_velocity - current_velocity

            # Use a proportional controller for smooth adjustment
            thrust_magnitude = 0.001 * delta_v  # Proportional constant
            thrust_magnitude = np.clip(thrust_magnitude, -0.01, 0.01)  # Limit thrust

            tangential_direction = np.cross(self.sat_positions[0], [0, 0, 1])
            tangential_direction /= np.linalg.norm(tangential_direction)
            thrusts[0] = tangential_direction * thrust_magnitude

            self.step(thrusts, timestep)
            time.sleep(0.01)

'''

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
                if i%2 == 0:
                    p.changeVisualShape(body_id, -1, rgbaColor=[1, 0, 0, 1])  # Satellites are red
                else:
                    p.changeVisualShape(body_id, -1, rgbaColor=[0.5, 1, 0.5, 1])  # Satellites are red
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
# Example usage
if __name__ == "__main__":
    # simulator = BaseOrbitSimulator(num_satellites=3, scale=0.001, satellite_scale=50.0, track_visualization=True)
    # simulator.visualize(steps=20000, timestep=10)

    simulator = BaseOrbitSimulator(num_satellites=6, scale=0.001, satellite_scale=50.0, track_visualization=True)
    simulator.visualize_with_transfer(steps=20000, timestep=10, target_orbit=35786)
