import numpy as np
import gym
from gym import spaces
import yaml
import matplotlib.pyplot as plt
import csv
import os
from torch.utils.tensorboard import SummaryWriter

class SimpleOrbitalMechanics:
    def __init__(self, mu=3.986e14):
        self.mu = mu

    def update_position(self, position, velocity, dt):
        acceleration = -self.mu * position / (np.linalg.norm(position) ** 3)
        new_velocity = velocity + acceleration * dt
        new_position = position + new_velocity * dt
        return new_position, new_velocity

class Satellite:
    def __init__(self, sat_id, position, velocity, role):
        self.sat_id = sat_id
        self.position = np.array(position, dtype=np.float64)
        self.velocity = np.array(velocity, dtype=np.float64)
        self.role = role
        self.active = True
        self.trajectory = [np.array(self.position.copy())]

class SpaceCombatEnv(gym.Env):
    def __init__(self, config_path="config/satellites.yaml", dt=10, log_dir="logs", centralized=True):
        super(SpaceCombatEnv, self).__init__()
        self.dt = dt
        self.orbit_engine = SimpleOrbitalMechanics()
        self.config_path = config_path
        self.centralized = centralized

        self.sat_config = self._load_config(config_path)
        self.agents = []
        self._init_from_config()
        self.total_agents = len(self.agents)

        self.observation_space = spaces.Box(
            low=-1e8, high=1e8, shape=(self.total_agents, 9), dtype=np.float32
        )
        low = np.tile(np.array([0, -1e8, -1e8, -1e8]), (self.total_agents, 1))
        high = np.tile(np.array([3, 1e8, 1e8, 1e8]), (self.total_agents, 1))
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.csv_file = open(os.path.join(self.log_dir, "trajectory_log.csv"), mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["step", "sat_id", "role", "x", "y", "z", "vx", "vy", "vz"])
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.step_count = 0

        self.strategy_registry = {}
        self.default_strategy = self.rule_based_action

    def register_strategy(self, name, strategy_fn):
        self.strategy_registry[name] = strategy_fn

    def set_strategy(self, name):
        self.default_strategy = self.strategy_registry.get(name, self.rule_based_action)

    def register_ddpg_strategy(self):
        def ddpg_policy(env):
            obs = env._get_obs()
            actions = []
            for agent_obs in obs:
                role = agent_obs[-3:]
                if np.array_equal(role, [0, 0, 0]):
                    actions.append([0, 0, 0, 0])
                else:
                    dummy_target = agent_obs[:3] + np.array([1000.0, 1000.0, 0])
                    actions.append([1, *dummy_target])
            return np.array(actions, dtype=np.float32)
        self.register_strategy("ddpg", ddpg_policy)

    def _load_config(self, path):
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    def _init_from_config(self):
        self.agents.clear()
        for entry in self.sat_config['satellites']:
            pos = [float(x) for x in entry['position']]
            vel = [float(x) for x in entry['velocity']]
            role = entry['role']
            sat_id = entry['id']
            self.agents.append(Satellite(sat_id, pos, vel, role))

    def reset(self):
        self._init_from_config()
        self.step_count = 0
        return self._get_obs()

    def _get_obs(self):
        obs = []
        for agent in self.agents:
            role_vec = self._role_onehot(agent.role)
            obs.append(np.concatenate([agent.position, agent.velocity, role_vec]))
        return np.array(obs, dtype=np.float32)

    def _role_onehot(self, role):
        return {
            "observer": [1, 0, 0],
            "defender": [0, 1, 0],
            "enemy": [0, 0, 1],
            "high_value": [0, 0, 0],
        }.get(role, [0, 0, 0])

    def step(self, actions=None):
        if self.centralized:
            if actions is None:
                actions = self.default_strategy(self)
        else:
            actions = []
            for i in range(self.total_agents):
                single_obs = self._get_obs()[i:i+1]
                single_action = self.default_strategy(self)[i]
                actions.append(single_action)
            actions = np.array(actions, dtype=np.float32)

        for i, agent in enumerate(self.agents):
            action_type, tx, ty, tz = actions[i]
            target_pos = np.array([tx, ty, tz])

            if agent.role == "high_value":
                continue

            if action_type == 1:
                delta_v = target_pos - agent.position
                agent.velocity += delta_v * 0.0001
            elif action_type == 2:
                pass
            elif action_type == 3:
                delta_v = (target_pos - agent.position)
                agent.velocity += delta_v * 0.00005

            agent.position, agent.velocity = self.orbit_engine.update_position(
                agent.position, agent.velocity, self.dt
            )
            agent.trajectory.append(agent.position.copy())

            self.csv_writer.writerow([
                self.step_count, agent.sat_id, agent.role,
                *agent.position.tolist(), *agent.velocity.tolist()
            ])

        obs = self._get_obs()
        reward = self._compute_reward()
        done = False
        info = {}
        self.writer.add_scalar("reward/step", reward, self.step_count)
        self.step_count += 1
        return obs, reward, done, info

    def _compute_reward(self):
        reward = 0
        reward_details = {
            'interception': 0,
            'observation': 0,
            'protection': 0
        }

        defenders = [a for a in self.agents if a.role == "defender"]
        enemies = [a for a in self.agents if a.role == "enemy"]
        observers = [a for a in self.agents if a.role == "observer"]
        high_value = next((a for a in self.agents if a.role == "high_value"), None)

        for d in defenders:
            for e in enemies:
                dist = np.linalg.norm(d.position - e.position)
                if dist < 1e4:
                    reward += 2
                    reward_details['interception'] += 2

        for o in observers:
            for e in enemies:
                dist = np.linalg.norm(o.position - e.position)
                if dist < 2e4:
                    reward += 1
                    reward_details['observation'] += 1

        if high_value:
            for e in enemies:
                dist = np.linalg.norm(high_value.position - e.position)
                if dist > 3e4:
                    reward += 1
                    reward_details['protection'] += 1

        self.writer.add_scalar("reward/interception", reward_details['interception'], self.step_count)
        self.writer.add_scalar("reward/observation", reward_details['observation'], self.step_count)
        self.writer.add_scalar("reward/protection", reward_details['protection'], self.step_count)

        return reward
    def rule_based_action(self):
        actions = []
        high_value = next((a for a in self.agents if a.role == "high_value"), None)
        enemy = next((a for a in self.agents if a.role == "enemy"), None)
        for agent in self.agents:
            if agent.role == "observer":
                actions.append([2, *enemy.position])
            elif agent.role == "defender":
                actions.append([3, *enemy.position])
            else:
                actions.append([0, 0, 0, 0])
        return np.array(actions, dtype=np.float32)

    def get_instructions(self):
        """返回所有卫星的当前指令（标准化格式）。"""
        instructions = []
        obs = self._get_obs()
        actions = self.default_strategy(self)
        for i, agent in enumerate(self.agents):
            act_type, tx, ty, tz = actions[i]
            if act_type == 0:
                instr = {
                    "sat_id": agent.sat_id,
                    "type": "idle",
                    "params": {}
                }
            elif act_type == 1:
                instr = {
                    "sat_id": agent.sat_id,
                    "type": "maneuver",
                    "target": [float(tx), float(ty), float(tz)]
                }
            elif act_type == 2:
                instr = {
                    "sat_id": agent.sat_id,
                    "type": "observe",
                    "target": [float(tx), float(ty), float(tz)]
                }
            elif act_type == 3:
                instr = {
                    "sat_id": agent.sat_id,
                    "type": "intercept",
                    "target": [float(tx), float(ty), float(tz)]
                }
            else:
                instr = {
                    "sat_id": agent.sat_id,
                    "type": "unknown",
                    "params": {}
                }
            instructions.append(instr)
        return instructions

    def render_instructions(self):
        """打印标准化多智能体指令。"""
        instructions = self.get_instructions()
        print("\n[Current Step Instructions]")
        for instr in instructions:
            print(f"{instr['sat_id']}: {instr['type']} -> {instr.get('target', instr.get('params', {}))}")
    def export_simulation_to_json(self, output_path="logs/simulation_record.json"):
        import json
        data = []
        for agent in self.agents:
            traj = np.array(agent.trajectory)
            steps = []
            for i in range(len(traj)):
                steps.append({
                    "step": i,
                    "position": traj[i].tolist(),
                    "role": agent.role
                })
            data.append({
                "sat_id": agent.sat_id,
                "role": agent.role,
                "trajectory": steps
            })
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"[Saved] Simulation trajectory exported to {output_path}")

    def render_2d(self):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 8))
        for agent in self.agents:
            traj = np.array(agent.trajectory)
            if agent.role == "observer":
                style = '--'
                label = f"{agent.sat_id} (obs)"
            elif agent.role == "defender":
                style = '-'
                label = f"{agent.sat_id} (def)"
            elif agent.role == "enemy":
                style = '-.'
                label = f"{agent.sat_id} (enemy)"
            elif agent.role == "high_value":
                style = ':'
                label = f"{agent.sat_id} (high)"
            else:
                style = '.'
                label = f"{agent.sat_id}"
            plt.plot(traj[:, 0], traj[:, 1], style, label=label)
            plt.scatter(traj[-1, 0], traj[-1, 1], s=40, marker='x')
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.title("Satellite Trajectories (2D View)")
        plt.grid(True)
        plt.legend()
        plt.axis('equal')
        plt.tight_layout()
        plt.show()

    def close(self):
        self.csv_file.close()
        self.writer.close()
