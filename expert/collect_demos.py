import os

import functools
from typing import Any

import numpy as np
import pickle
import numpy.typing as npt
import shimmy
import tensorflow as tf
import tyro
from jax.experimental import jax2tf
from sbx import TD3

from env import dmc
from dataclasses import dataclass


@dataclass
class Args:
    seed: int = 1
    """seed of the experiment"""
    env: str = "cheetah_run"
    """Name of the task"""
    steps: int = 2_500_000
    """Number of training steps"""
    data_path: str = "expert"
    """Path to save the expert demonstrations"""


if __name__ == "__main__":

    cfg = tyro.cli(Args)
    
    env = dmc.make(cfg.env, seed=cfg.seed, obs_type='states', convert_to_gym=False)
    env = shimmy.DmControlCompatibilityV0(env)

    model = TD3("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=cfg.steps, progress_bar=True)

    @functools.partial(
        tf.function,
        input_signature=[
            tf.TensorSpec(env.observation_space.shape, env.observation_space.dtype),  # type: ignore
        ],
    )
    @functools.partial(
        jax2tf.convert,
        enable_xla=True,
        with_gradient=False,
        native_serialization=False,
    )
    def policy_apply(obs: npt.NDArray[Any]) -> npt.NDArray[Any]:
        return model.policy.forward(obs[None, ...])

    policy = tf.Module()
    policy.__call__ = policy_apply
    
    # Saves a single trajectory for this seed
    done = False
    obs, _ = env.reset()
    E_s, E_a = [], []
    while not done:
        action = model.policy.forward(np.expand_dims(obs, axis=0))[0]
        E_s.append(obs)
        E_a.append(action)
        obs_next, _, term, trunc, _ = env.step(action)
        done = trunc or term
    E_s.append(obs_next)
    
    if not os.path.exists(f"{cfg.data_path}/{cfg.env}"):
        os.makedirs(f"{cfg.data_path}/{cfg.env}")

    with open(f"{cfg.data_path}/{cfg.env}/trajectory-{cfg.seed}.pkl", "wb") as f:
        pickle.dump((np.array(E_s), np.array(E_a)), f)
        