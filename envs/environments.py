from typing import Tuple

import gymnasium as gym
import pickle

from gymnasium.spaces import Box, Space
import numpy as np


class DMCEnv:
    def __init__(self,
                 env_name: str,
                 seed=0,
                 obs_type='states',
                 frame_stack=3,
                 action_repeat=1,
                 im_size=84):

        import envs.dmc as dmc 
        self.env = dmc.make(env_name,
                            obs_type=obs_type,
                            seed=seed,
                            frame_stack=frame_stack,
                            action_repeat=action_repeat)

        self.env_name = env_name
        self.obs_type = obs_type
        self.frame_stack = frame_stack

    def reset(self) -> np.ndarray:
        state, _ = self.env.reset()
        state = np.expand_dims(state, axis=0)  # Add batch dimension to state
        return state

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        action = np.clip(
            action, a_min=self.env.action_space.low, a_max=self.env.action_space.high
        )  # Clip actions
        state, reward, terminal, truncation, _ = self.env.step(
            np.array(action[0])
        )  # Remove batch dimension from action
        
        state = np.expand_dims(state, axis=0)  # Add batch dimension to state

        return state, reward, False, truncation

    @property
    def ref_max_score(self):
        return 1000

    @property
    def ref_min_score(self):
        return 0

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()

    @property
    def observation_space(self) -> Space:
        return self.env.observation_space

    @property
    def action_space(self) -> Space:
        return self.env.action_space

    @property
    def max_episode_steps(self) -> int:
        return self.env.max_episode_steps


    def get_expert_traj(self,
                        seed: int = 0):

        data_dir = "/home/mila/a/arnav-kumar.jain/scratch/IL/expert-trajectories"

        with open(f'{data_dir}/{self.env_name}/trajectory-{seed}.pkl', 'rb') as f:
            states, actions = pickle.load(f)
    
        return np.array(states), np.array(actions)
