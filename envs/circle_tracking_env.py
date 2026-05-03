import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CircleTracking3DEnv(gym.Env):
    """
    Simple 3D quadrotor-like trajectory tracking environment.

    State:
        position: x, y, z
        velocity: vx, vy, vz

    Observation:
        position
        velocity
        moving target position
        tracking error
        phase information

    Action:
        acceleration in x, y, z

    Goal:
        Track a circular 3D trajectory.
    """

    metadata = {"render_modes": []}

    def __init__(self, max_steps=500):
        super().__init__()

        self.dt = 0.05
        self.max_steps = max_steps
        self.step_count = 0

        self.radius = 1.2
        self.base_altitude = 1.2
        self.z_amplitude = 0.35
        self.phase_speed = 0.04

        self.state = None

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(14,),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=np.array([-3.0, -3.0, -3.0], dtype=np.float32),
            high=np.array([3.0, 3.0, 3.0], dtype=np.float32),
            dtype=np.float32,
        )

    def get_reference(self):
        phase = self.step_count * self.phase_speed

        target_x = self.radius * np.cos(phase)
        target_y = self.radius * np.sin(phase)
        target_z = self.base_altitude + self.z_amplitude * np.sin(2 * phase)

        target = np.array([target_x, target_y, target_z], dtype=np.float32)

        return target, phase

    def get_obs(self):
        x, y, z, vx, vy, vz = self.state

        target, phase = self.get_reference()

        position = np.array([x, y, z], dtype=np.float32)
        velocity = np.array([vx, vy, vz], dtype=np.float32)
        error = target - position

        obs = np.concatenate(
            [
                position,
                velocity,
                target,
                error,
                np.array([np.sin(phase), np.cos(phase)], dtype=np.float32),
            ]
        ).astype(np.float32)

        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.step_count = 0

        target, _ = self.get_reference()

        position = target + self.np_random.uniform(
            low=np.array([-0.2, -0.2, -0.1]),
            high=np.array([0.2, 0.2, 0.1]),
        )

        velocity = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        self.state = np.concatenate([position, velocity]).astype(np.float32)

        obs = self.get_obs()

        info = {
            "target": target,
            "is_safe": True,
            "distance_to_target": float(np.linalg.norm(target - position)),
        }

        return obs, info

    def step(self, action):
        self.step_count += 1

        action = np.clip(action, self.action_space.low, self.action_space.high)

        x, y, z, vx, vy, vz = self.state
        ax, ay, az = action

        vx = vx + ax * self.dt
        vy = vy + ay * self.dt
        vz = vz + az * self.dt

        x = x + vx * self.dt
        y = y + vy * self.dt
        z = z + vz * self.dt

        self.state = np.array([x, y, z, vx, vy, vz], dtype=np.float32)

        target, phase = self.get_reference()

        position = np.array([x, y, z], dtype=np.float32)
        velocity = np.array([vx, vy, vz], dtype=np.float32)

        distance = np.linalg.norm(position - target)
        speed = np.linalg.norm(velocity)
        action_magnitude = np.linalg.norm(action)

        reward = 0.0
        reward -= 4.0 * distance
        reward -= 0.05 * speed
        reward -= 0.01 * action_magnitude

        if distance < 0.10:
            reward += 2.0

        if distance < 0.05:
            reward += 5.0

        safe_x = -2.0 <= x <= 2.0
        safe_y = -2.0 <= y <= 2.0
        safe_z = 0.1 <= z <= 2.5

        is_safe = safe_x and safe_y and safe_z

        terminated = False

        if not is_safe:
            reward -= 50.0
            terminated = True

        truncated = self.step_count >= self.max_steps

        obs = self.get_obs()

        info = {
            "target": target,
            "phase": float(phase),
            "distance_to_target": float(distance),
            "speed": float(speed),
            "is_safe": bool(is_safe),
            "constraint_violation": not bool(is_safe),
        }

        return obs, float(reward), terminated, truncated, info

    def render(self):
        x, y, z, vx, vy, vz = self.state
        target, _ = self.get_reference()

        print(
            f"pos=({x:.2f}, {y:.2f}, {z:.2f}), "
            f"target=({target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f})"
        )

    def close(self):
        pass