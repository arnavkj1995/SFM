import sys

import tyro
import wandb
from tqdm import tqdm
from typing import Any
from termcolor import cprint
from dataclasses import dataclass

import chex
import flax
import jax
import optax
import numpy as np
import jax.numpy as jnp
import flax.linen as nn

from envs.environments import DMCEnv
from common.utils import TrainState
from common.buffer import ReplayBuffer

Params = flax.core.FrozenDict[str, Any]


@dataclass
class Args:
    exp_name: str = "bc"
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project: str = "SFM"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    wandb_suffix: str = "final"
    """Experiment name for wandb"""

    # Training specific arguments
    env: str = "walker_walk"
    """the name of the environment"""
    steps: int = 20000
    """total environment steps"""
    eval_episodes: int = 10
    """the number of episodes to evaluate the agent"""
    

    # Agent parameters
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    update_actor_frequency: int = 1
    """the frequency of training policy (delayed)"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    
    # Actor parameters
    actor_lr: float = 3e-4
    """the learning rate of the optimizer"""
    actor_h_dim: int = 256
    """the hidden dimension of the actor network"""


class ActorNetwork(nn.Module):
    hdim: int
    action_dim: int

    @nn.compact
    def __call__(self,
                 s: jnp.ndarray):
        x = nn.Dense(self.hdim, kernel_init=nn.initializers.orthogonal())(s)
        x = nn.relu(x)
        x = nn.Dense(self.hdim, kernel_init=nn.initializers.orthogonal())(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim, kernel_init=nn.initializers.orthogonal())(x)
        x = nn.tanh(x)
        return x


class BC:
    def __init__(
        self,
        rng_key: jax.random.PRNGKey,
        state_size: int,
        action_size: int,
        cfg
    ):
        (
            self.rng_key,
            actor_rng_key,
        ) = jax.random.split(rng_key, 2)
        
        sample_state = jnp.zeros((state_size))
        # Define actor
        actor_net = ActorNetwork(cfg.actor_h_dim, action_size)
        self.actor = TrainState.create(
            apply_fn=actor_net.apply,
            params=actor_net.init(actor_rng_key, sample_state[None, ...]),
            target_params=actor_net.init(actor_rng_key, sample_state[None, ...]),
            checkpoint=actor_net.init(actor_rng_key, sample_state[None, ...]),
            tx=optax.chain(
                optax.adam(learning_rate=cfg.actor_lr)
            ),
        )

        @chex.assert_max_traces(10)
        def _get_action(actor: TrainState,
                        s: jnp.ndarray):
            a = actor.apply_fn(actor.params, s)
            return a

        self._get_action = jax.jit(_get_action)


        @chex.assert_max_traces(1)
        def _update_actor(
            rng_key: jax.random.PRNGKey,
            actor: TrainState,
            s: jnp.ndarray,
            a: jnp.ndarray,
        ):
            def _loss_fn(params: Params):
                # DPG loss from the Psi_network
                a_pred = actor.apply_fn(params, s)

                loss = ((a - a_pred)**2).mean()

                return loss, {
                    "actor/loss": loss,
                }

            info, grads = jax.value_and_grad(_loss_fn, has_aux=True)(actor.params)
            new_actor = actor.apply_gradients(grads=grads)
            return rng_key, new_actor, info

        self.update_actor = jax.jit(_update_actor)

    def get_action(self, state, train=False, rng_key=None):
        action = self._get_action(self.actor, state)
        action = jax.device_get(action)
        if train:
            action += jax.random.normal(rng_key, action.shape) * self.action_noise
        # FIXME: Add the part of max_action
        return np.clip(action, -1, 1)

    def update(
        self,
        transitions,
    ):
        train_metrics = {}

        s, a, _, _, _, _ = transitions
        
        self.rng_key, self.actor, actor_info = self.update_actor(
                self.rng_key,
                self.actor,
                s,
                a
            )
        
        train_metrics.update(actor_info[1])
        return train_metrics


def evaluate_agent(
    agent,
    env,
    num_episodes: int,
    rng_key: jax.random.PRNGKey = None,
):
    returns = []
    
    for _ in range(num_episodes):
        rewards = []
        state, terminal, truncation = env.reset(), False, False
        while not (terminal or truncation):
            rng_key, subkey = jax.random.split(rng_key)
            action = agent.get_action(state, rng_key=subkey)
            next_state, reward, terminal, truncation = env.step(np.array(action))

            rewards.append(reward)
            state = next_state
        returns.append(sum(rewards))

    return returns


if __name__ == "__main__":
    
    cfg = tyro.cli(Args)
    
    # General setup
    np.random.seed(cfg.seed)

    random_state = np.random.RandomState(cfg.seed)
    rng_key = jax.random.PRNGKey(
        random_state.randint(-sys.maxsize - 1, sys.maxsize + 1, dtype=np.int64)
    )

    if cfg.track:
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            sync_tensorboard=True,
            group=f"BC/%s/%s"
            % (
                cfg.env,
                cfg.wandb_suffix
            ),
            name=str(cfg.seed),
            monitor_gym=True,
            save_code=True,
        )

    # Set up environment
    env = DMCEnv(cfg.env)
    normalization_max, normalization_min = 1000, 0

    state_size, action_size = env.observation_space.shape[0], env.action_space.shape[0]

    # Set up agent
    rng_key, network_rngkey = jax.random.split(rng_key)

    agent = BC(
        network_rngkey,
        state_size,
        action_size,
        cfg
    )

    expert_buffer = ReplayBuffer(env.observation_space.shape[0],
                                env.action_space.shape[0],
                                max_size=1e4,
                                batch_size=cfg.batch_size,
                                normalize_actions=True)

    # Training
    t, state, terminal, train_return = 0, env.reset(), False, 0

    pbar = tqdm(range(0, cfg.steps + 1), unit_scale=1, smoothing=0)

    expert_s, expert_a = env.get_expert_traj(cfg.seed)
    for i in range(len(expert_s) - 1):
        expert_buffer.add(expert_s[i], expert_a[i], expert_s[i+1], np.array([0]), np.array([False]), False)

    train_metrics = {}

    cprint (f"Training BC on {cfg.env} for {cfg.steps} environment steps", "green")
    for step in pbar:

        transitions = expert_buffer.sample()
        train_metrics = agent.update(transitions)
        
        if step % 5000 == 0:
            metrics = {'step': step}
            metrics.update(jax.device_get(train_metrics))
            if cfg.track:
                wandb.log(metrics)

        # Evaluate agent and plot metrics
    rng_key, sample_key = jax.random.split(rng_key)
    test_returns = evaluate_agent(
        agent, env, cfg.eval_episodes, rng_key=sample_key
    )
    log_metrics = {"step": 0,
                    "test_returns": np.mean(test_returns)}

    if cfg.track:
        wandb.log(log_metrics)
        log_metrics["step"] = 1000000
        wandb.log(log_metrics)
