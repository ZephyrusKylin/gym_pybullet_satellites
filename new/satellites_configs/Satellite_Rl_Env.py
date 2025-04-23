# -*- coding: utf-8 -*-
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import yaml

class BaseSatelliteEnv(gym.Env):
    """
    Base environment for multi-agent satellite orbital dynamics with thrust, gravity, collision detection,
    displacement reporting, and role-specific actions.
    """
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, config_file: str, timestep: float = 10.0, max_steps: int = 1000):
        # Load configuration
        with open(config_file, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        # Physical constants
        self.G    = 6.67430e-11           # m^3 kg^-1 s^-2
        self.M_e  = 5.972e24              # kg
        self.g0   = 9.80665               # m/s^2
        # Simulation params
        self.dt        = timestep
        self.max_steps = max_steps
        # Load satellites
        self.red_satellites  = cfg['red']
        self.blue_satellites = cfg['blue']
        self.sats = self.red_satellites + self.blue_satellites
        self.n_red  = len(self.red_satellites)
        self.n_blue = len(self.blue_satellites)
        self.n      = self.n_red + self.n_blue
        # State: [pos(3), vel(3), mass, obs_flag, strike_flag]
        self.state = np.zeros((self.n,9), dtype=np.float64)
        # Action: [thrust_x,y,z normalized in [-1,1], special in {0,1,2,3}]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n,4), dtype=np.float32)
        # Observation: flat state
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.n*8,), dtype=np.float32)
        self.current_step = 0

    def reset(self, seed=None, options=None):
        self.current_step = 0
        for i, sat in enumerate(self.sats):
            pos = np.array(sat['initial_position'], dtype=np.float64)
            vel = np.array(sat['initial_velocity'], dtype=np.float64)
            m   = float(sat['mass'])
            self.state[i] = np.hstack([pos, vel, m, 0.0, 0.0])
        return self.state.flatten().astype(np.float32), {}

    def _acceleration(self, r, thrust, m):
        # Two-body gravity + thrust
        norm_r3 = np.linalg.norm(r)**3 + 1e-10
        a_grav   = - self.G * self.M_e * r / norm_r3
        a_thrust = thrust / m
        return a_grav + a_thrust

    def _rk4_step(self, r, v, thrust, m):
        # RK4 integration for orbital motion
        dt = self.dt
        k1_v = self._acceleration(r, thrust, m) * dt
        k1_r = v * dt
        k2_v = self._acceleration(r + 0.5*k1_r, v + 0.5*k1_v, m) * dt
        k2_r = (v + 0.5*k1_v) * dt
        k3_v = self._acceleration(r + 0.5*k2_r, v + 0.5*k2_v, m) * dt
        k3_r = (v + 0.5*k2_v) * dt
        k4_v = self._acceleration(r + k3_r, v + k3_v, m) * dt
        k4_r = (v + k3_v) * dt
        r_new = r + (k1_r + 2*k2_r + 2*k3_r + k4_r) / 6.0
        v_new = v + (k1_v + 2*k2_v + 2*k3_v + k4_v) / 6.0
        return r_new, v_new
    
    def step(self, action):
        old_pos = self.state[:,0:3].copy()
        thrust_cmd = np.clip(action[:,:3], -1,1) * np.array([sat['max_thrust'] for sat in self.sats])[:,None]
        special   = np.round(np.clip(action[:,3],0,1)*3).astype(int)
        new_state = np.zeros_like(self.state)
        # event flags
        obs_events    = np.zeros(self.n)
        strike_events = np.zeros(self.n)
        defend_events = np.zeros(self.n)
        # dynamics and role actions
        for i in range(self.n):
            r, v, m = self.state[i,0:3], self.state[i,3:6], self.state[i,6]
            r_new, v_new = self._rk4_step(r, v, thrust_cmd[i], m)
            dm = np.linalg.norm(thrust_cmd[i]) * self.dt / (float(self.sats[i]['isp']) * self.g0)
            m_new = max(m - dm, 0.0)
            new_state[i,0:3] = r_new
            new_state[i,3:6] = v_new
            new_state[i,6]   = m_new
            stype = self.sats[i]['type']
            if special[i]==1 and stype=='observer':
                maxd = self.sats[i]['obs_range']; dir_v = np.array(self.sats[i]['obs_dir'])
                for j in range(self.n_red, self.n):
                    rel = new_state[j,0:3] - r_new
                    if np.linalg.norm(rel)<=maxd and np.dot(rel,dir_v)>0:
                        obs_events[i] = 1
            if special[i]==2 and stype=='strike':
                sr = self.sats[i]['strike_range']
                for j in range(self.n_red, self.n):
                    if np.linalg.norm(new_state[j,0:3]-r_new)<=sr:
                        strike_events[i] = 1
            if special[i]==3 and stype=='defense':
                dr = self.sats[i]['defense_range']
                for j in range(self.n_red, self.n):
                    for k in range(self.n_red):
                        if self.sats[k]['type']=='high_value':
                            tgt=new_state[k,0:3]
                            if np.linalg.norm(new_state[j,0:3]-tgt)<dr and np.linalg.norm(new_state[j,0:3]-r_new)<dr:
                                defend_events[i] = 1
        # collision detection
        collisions = []
        for i in range(self.n):
            for j in range(i+1, self.n):
                d = np.linalg.norm(new_state[i,0:3] - new_state[j,0:3])
                if d <= (self.sats[i]['radius'] + self.sats[j]['radius']):
                    collisions.append((i,j))
        self.state = new_state
        self.current_step += 1
        obs = self.state.flatten().astype(np.float32)
        rewards, done, truncated = self._compute_reward(obs_events, strike_events, defend_events, collisions)
        delta_pos = self.state[:,0:3] - old_pos
        info = {
            'collisions': collisions,
            'delta_pos': delta_pos,
            'obs_events': obs_events,
            'strike_events': strike_events,
            'defend_events': defend_events
        }
        return obs, rewards, done, truncated, info
    # def step(self, action):
    #     # Pre-step: store old positions
    #     old_pos = self.state[:,0:3].copy()
    #     # Parse thrust vector
    #     thrust_cmd = np.clip(action[:,:3], -1,1) * np.array([sat['max_thrust'] for sat in self.sats])[:,None]
    #     special   = np.round(np.clip(action[:,3],0,1)*3).astype(int)
    #     new_state = np.zeros_like(self.state)
    #     obs_events = np.zeros(self.n)
    #     strike_events = np.zeros(self.n)
    #     defend_events = np.zeros(self.n)
    #     # Dynamics and role actions
    #     for i in range(self.n):
    #         r, v, m = self.state[i,0:3], self.state[i,3:6], self.state[i,6]
    #         # Integrate motion by RK4
    #         r_new, v_new = self._rk4_step(r, v, thrust_cmd[i], m)
    #         # Mass update via rocket equation
    #         thrust_mag = np.linalg.norm(thrust_cmd[i])
    #         isp = float(self.sats[i]['isp'])
    #         dm = thrust_mag * self.dt / (isp * self.g0)
    #         m_new = max(m - dm, 0.0)
    #         new_state[i,0:3] = r_new
    #         new_state[i,3:6] = v_new
    #         new_state[i,6]   = m_new
    #         # Role-specific
    #         sat_t = self.sats[i]['type']
    #         if special[i]==1 and sat_t=='observer':
    #             cfg=self.sats[i]
    #             maxd=cfg['obs_range']
    #             dir_v=np.array(cfg['obs_dir'])
    #             for j in range(self.n_red,self.n):
    #                 rel=new_state[j,0:3]-r_new
    #                 if np.linalg.norm(rel)<=maxd and np.dot(rel,dir_v)>0: obs_events[i]=1
    #         if special[i]==2 and sat_t=='strike':
    #             sr=self.sats[i]['strike_range']
    #             for j in range(self.n_red,self.n):
    #                 if np.linalg.norm(new_state[j,0:3]-r_new)<=sr: strike_events[i]=1
    #         if special[i]==3 and sat_t=='defense':
    #             dr=self.sats[i]['defense_range']
    #             for j in range(self.n_red,self.n):
    #                 for k in range(self.n_red):
    #                     if self.sats[k]['type']=='high_value':
    #                         tgt=new_state[k,0:3]
    #                         if np.linalg.norm(new_state[j,0:3]-tgt)<dr and np.linalg.norm(new_state[j,0:3]-r_new)<dr:
    #                             defend_events[i]=1
    #     # Collision detection
    #     collisions=[]
    #     for i in range(self.n):
    #         for j in range(i+1,self.n):
    #             d=np.linalg.norm(new_state[i,0:3]-new_state[j,0:3])
    #             if d<=self.sats[i]['radius']+self.sats[j]['radius']:
    #                 collisions.append((i,j))
    #     # Update state
    #     self.state = new_state
    #     self.current_step+=1
    #     # Observation and reward
    #     obs=self.state.flatten().astype(np.float32)
    #     rewards=self._compute_reward(obs_events,strike_events,defend_events,collisions)
    #     done,trunc=self._compute_done(collisions)
    #     # Displacement info
    #     delta_pos = self.state[:,0:3] - old_pos
    #     info={'collisions':collisions,'delta_pos':delta_pos,'obs_events':obs_events,
    #           'strike_events':strike_events,'defend_events':defend_events}
    #     return obs,rewards,done,trunc,info

    def _get_obs(self):
        return self.state.flatten().astype(np.float32)
    
    def _compute_reward(self, obs_events, strike_events, defend_events, collisions):
        # initialize
        rewards = np.zeros(self.n, dtype=float)
        # red side rewards
        for i in range(self.n_red):
            rewards[i] += obs_events[i] * 0.1
            rewards[i] += defend_events[i] * 0.5
        # blue side rewards
        for i in range(self.n_red, self.n):
            rewards[i] += strike_events[i] * 1.0
        # collision penalty
        for (i,j) in collisions:
            rewards[i] -= 1.0
            rewards[j] -= 1.0
        # episode termination due to high-value satellite hit
        done = False
        for (i,j) in collisions:
            # check high-value (red) vs strike (blue)
            if (i < self.n_red and self.sats[i]['type']=='high_value' and j>=self.n_red and self.sats[j]['type']=='strike') or \
               (j < self.n_red and self.sats[j]['type']=='high_value' and i>=self.n_red and self.sats[i]['type']=='strike'):
                done = True
                break
        truncated = (self.current_step >= self.max_steps)
        return rewards, done, truncated


    # def _compute_reward(self, obs_events, strike_events, defend_events, collisions):
    #     # Base distance metric as before
    #     red_pos = self.state[:self.n_red,0:3]
    #     blue_pos = self.state[self.n_red:,0:3]
    #     dists = np.linalg.norm(red_pos[:,None,:] - blue_pos[None,:,:], axis=-1)
    #     min_dists = np.min(dists, axis=1)
    #     red_base = - np.mean(min_dists)
    #     print(red_base)
    #     blue_base = np.mean(min_dists)
    #     # Role bonuses
    #     r_bonus = obs_events * 0.1 + strike_events * 1.0 + defend_events * 0.5
    #     print(r_bonus)
    #     # Collision penalty: negative if any red collision with blue
    #     col_penalty = 0.0
    #     for (i,j) in collisions:
    #         if i < self.n_red <= j:
    #             col_penalty -= 1.0
    #     # Assemble
    #     rewards = np.zeros(self.n)
    #     rewards[:self.n_red] = red_base + r_bonus + col_penalty
    #     rewards[self.n_red:] = blue_base - col_penalty  # blue benefit from collisions
    #     return rewards

    # def _compute_done(self, collisions):
    #     if self.current_step >= self.max_steps:
    #         return True, False
    #     # high-value destroyed?
    #     for (i,j) in collisions:
    #         # if blue hits red high-value
    #         if i < self.n_red and self.sats[i]['type']=='high_value' and j >= self.n_red:
    #             return True, False
    #     # fuel exhausted
    #     if np.any(self.state[:,6] <= 0):
    #         return True, False
    #     return False, False

    def render(self, mode='human'):
        print(f"Step {self.current_step}")
        for i in range(self.n):
            t = self.sats[i]['type']
            print(f"Sat {i} ({t}): pos={self.state[i,0:3]}, vel={self.state[i,3:6]}, m={self.state[i,6]:.2f}")


class SatelliteRLEnv(BaseSatelliteEnv):
    def __init__(self, config_file: str, timestep: float = 10.0, max_steps: int = 1000):
        super().__init__(config_file, timestep, max_steps)
    # Extend or override reward / done as needed
