import sys

import tyro
import wandb
import functools
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
    exp_name: str = "gaifo"
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project: str = "SFM"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    wandb_suffix: str = "check"
    """Experiment name for wandb"""

    # Training specific arguments
    env: str = "walker_walk"
    """the name of the environment"""
    steps: int = 1000000
    """total environment steps"""
    training_start: int = 10000
    """the step to start training"""
    eval_episodes: int = 10
    """the number of episodes to evaluate the agent"""
    eval_interval: int = 25000
    """the interval to evaluate the agent"""

    # Agent parameters
    discount: float = 0.99
    """the discount factor gamma"""
    polyak_factor: float = 0.995
    """Polyak factor for averaging"""
    batch_size: int = 1024
    """the batch size of sample from the reply memory"""
    target_noise: float = 0.2
    """the scale of policy noise"""
    target_noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""
    action_noise: float = 0.1
    """the scale of action noise"""
    exploration_noise: float = 0.1
    """the scale of exploration noise"""
    update_actor_frequency: int = 2
    """the frequency of training policy (delayed)"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    sample_future_prob: float = .99
    """ the probability of sampling the future state (for Hilp features only)"""
    target_update_rate: int = 250
    """the frequency of updating the max / min of SFs"""
    
    # TD7 Encoder parameters
    encoder_lr: float = 1e-4
    """the learning rate of the optimizer"""
    encoder_h_dim: int = 256
    """the hidden dimension of the encoder of TD7 network"""
    
    # Actor parameters
    actor_lr: float = 3e-4
    """the learning rate of the optimizer"""
    actor_h_dim: int = 256
    """the hidden dimension of the actor network"""
    
    # Critic parameters
    critic_lr: float = 3e-4
    """the learning rate of the optimizer"""
    critic_h_dim: int = 256
    """the hidden dimension of the Critic"""
    
    # Discriminators parameters
    disc_lr: float = 8e-4
    """the learning rate of the optimizer"""
    disc_h_dim: int = 256
    """the hidden dimension of the Discriminator"""
    update_disc_every: int = 10000
    """Steps to update discriminator"""
    update_disc_batch_size: int = 4096
    """batch size for the discriminator"""
    disc_num_episodes: int = 4
    """Number of episodes for the discriminator"""
    disc_update_steps: int = 1
    """Number of updates steps for the discriminator"""
    

class AvgL1Norm(nn.Module):
    eps: float = 1e-6

    def __call__(self, x: jnp.ndarray):
        return x / jnp.clip(jnp.abs(x).mean(axis=-1, keepdims=True), min=self.eps)


class Encoder(nn.Module):
    hdim: int = 256
    zs_dim: int = 256

    def setup(self):

        self.zs_net = nn.Sequential([
            nn.Dense(self.hdim),
            nn.elu,
            nn.Dense(self.hdim),
            nn.elu,
            nn.Dense(self.zs_dim),
            AvgL1Norm()]
        )
        self.zsa_net = nn.Sequential([
            nn.Dense(self.hdim),
            nn.elu,
            nn.Dense(self.hdim),
            nn.elu,
            nn.Dense(self.zs_dim)]
        )

    def zs(self,
           s: jnp.ndarray):
        return self.zs_net(s)

    def zsa(self,
            zs: jnp.ndarray,
            a: jnp.ndarray):
        return self.zsa_net(jnp.concatenate([zs, a], -1))

    @nn.compact
    def __call__(self,
                 state: jnp.ndarray,
                 action: jnp.ndarray):
        zs = self.zs(state)
        zsa = self.zsa(zs, action)
        return zs, zsa


class QNetwork(nn.Module):
    hdim: int

    @nn.compact
    def __call__(self,
                 s: jnp.ndarray,
                 a: jnp.ndarray,
                 zs: jnp.ndarray,
                 zsa: jnp.ndarray):

        sa = jnp.concatenate([s, a], -1)
        x = nn.Dense(self.hdim)(sa)
        x = AvgL1Norm()(x)
        x = jnp.concatenate([x, zs, zsa], -1)
        x = nn.Dense(self.hdim)(x)
        x = nn.elu(x)
        x = nn.Dense(self.hdim)(x)
        x = nn.elu(x)
        x = nn.Dense(1)(x)
        return x


TwinQNetworks = nn.vmap(
    QNetwork,
    variable_axes={"params": 0},
    split_rngs={"params": True},
    in_axes=None,
    out_axes=0,
    axis_size=2,
)


class ActorNetwork(nn.Module):
    hdim: int
    action_dim: int

    @nn.compact
    def __call__(self,
                 s: jnp.ndarray,
                 zs: jnp.ndarray):
        x = nn.Dense(self.hdim)(s)
        x = AvgL1Norm()(x)
        x = jnp.concatenate([x, zs], -1)
        x = nn.Dense(self.hdim)(x)
        x = nn.elu(x)
        x = nn.Dense(self.hdim)(x)
        x = nn.elu(x)
        x = nn.Dense(self.action_dim)(x)
        x = nn.tanh(x)
        return x


class Discriminator(nn.Module):
    hdim: int

    @nn.compact
    def __call__(self,
                 s: jnp.ndarray):
        x = nn.Dense(self.hdim)(s)
        x = nn.relu(x)
        x = nn.Dense(self.hdim)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x


class GAIfO_TD7:
    def __init__(
        self,
        rng_key: jax.random.PRNGKey,
        state_size: int,
        action_size: int,
        cfg
    ):
        (
            self.rng_key,
            disc_rng_key,
            encoder_rng_key,
            actor_rng_key,
            critic_rng_key,
        ) = jax.random.split(rng_key, 5)

        self.polyak_factor = cfg.polyak_factor
        self.step = 0
        self.target_noise = cfg.target_noise
        self.target_noise_clip = cfg.target_noise_clip
        self.action_noise = cfg.action_noise
        self.update_actor_frequency = cfg.update_actor_frequency
        self.target_update_rate = cfg.target_update_rate

        sample_state = jnp.zeros((state_size,))
        sample_action = jnp.zeros((action_size,))
        
        # Define encoder
        encoder_net = Encoder(hdim=cfg.encoder_h_dim)

        self.encoder = TrainState.create(
            apply_fn=encoder_net.apply,
            params=encoder_net.init(encoder_rng_key, sample_state, sample_action),
            target_params=encoder_net.init(encoder_rng_key, sample_state, sample_action),
            target_params2=encoder_net.init(encoder_rng_key, sample_state, sample_action),
            tx=optax.chain(
                optax.adam(learning_rate=cfg.encoder_lr)
            )
        )
        sample_zs, sample_zsa = encoder_net.apply(self.encoder.params, sample_state[None,...], sample_action[None,...])

        # Define actor
        actor_net = ActorNetwork(cfg.actor_h_dim, action_size)
        self.actor = TrainState.create(
            apply_fn=actor_net.apply,
            params=actor_net.init(actor_rng_key, sample_state[None, ...], sample_zs),
            target_params=actor_net.init(actor_rng_key, sample_state[None, ...], sample_zs),
            tx=optax.chain(
                optax.adam(learning_rate=cfg.actor_lr)
            ),
        )

        # Define critic
        critic_net = TwinQNetworks(cfg.critic_h_dim)
        self.critic = TrainState.create(
            apply_fn=critic_net.apply,
            params=critic_net.init(critic_rng_key, sample_state[None, ...], sample_action[None, ...], sample_zs, sample_zsa),
            target_params=critic_net.init(critic_rng_key, sample_state[None, ...], sample_action[None, ...], sample_zs, sample_zsa),
            tx=optax.chain(
                optax.adam(learning_rate=cfg.critic_lr)
            ),
        )
        
        # Define Discrminator
        disc_net = Discriminator(cfg.disc_h_dim)
        disc_lr_scheduler = optax.linear_schedule(
                                init_value=cfg.disc_lr,
                                end_value=1e-7,
                                transition_steps=int(cfg.steps / cfg.update_disc_every),
                            )
        self.disc = TrainState.create(
            apply_fn=disc_net.apply,
            params=disc_net.init(disc_rng_key, jnp.concatenate([sample_state[None, ...], sample_state[None, ...]], axis=-1)),
            target_params=disc_net.init(disc_rng_key, jnp.concatenate([sample_state[None, ...], sample_state[None, ...]], axis=-1)),
            tx=optax.adam(learning_rate=disc_lr_scheduler)
        )

        self.Q_max = -1e8
        self.Q_min = 1e8
        self.Q_max_target = 0.
        self.Q_min_target = 0.

        @chex.assert_max_traces(10)
        def _get_action(encoder: TrainState,
                        actor: TrainState,
                        s: jnp.ndarray):
            zs = encoder.apply_fn(encoder.target_params, s, method='zs')
            a = actor.apply_fn(actor.params,
                               s,
                               zs)
            return a

        self._get_action = jax.jit(_get_action)

        @chex.assert_max_traces(1)
        def _update_encoder(encoder: TrainState,
                            s: jnp.ndarray,
                            a: jnp.ndarray,
                            s_next: jnp.ndarray):
            next_zs = encoder.apply_fn(encoder.params, s_next, method='zs')

            def _loss_fn(params):
                _, zsa = encoder.apply_fn(params, s, a)
                loss = jnp.mean((zsa - next_zs) ** 2)
                return loss, {"encoder/loss": loss}

            info, grads = jax.value_and_grad(_loss_fn, has_aux=True)(encoder.params)
            new_encoder = encoder.apply_gradients(grads=grads)
            return new_encoder, info
        self.update_encoder = jax.jit(_update_encoder)
        
        @chex.assert_max_traces(1)
        def _update_actor(
            rng_key: jax.random.PRNGKey,
            encoder: TrainState,
            actor: TrainState,
            critic: TrainState,
            s: jnp.ndarray,
        ):
            def _loss_fn(params: Params):
                fixed_zs = encoder.apply_fn(encoder.target_params,
                                            s,
                                            method="zs")
                a = actor.apply_fn(params,
                                   s,
                                   fixed_zs)
                fixed_zsa = encoder.apply_fn(encoder.target_params,
                                             fixed_zs,
                                             a,
                                             method="zsa")
                q = critic.apply_fn(critic.params,
                                    s,
                                    a,
                                    fixed_zs,
                                    fixed_zsa)[0]

                loss = -q.mean()

                return loss, {"actor/loss": loss}

            info, grads = jax.value_and_grad(_loss_fn, has_aux=True)(actor.params)
            new_actor = actor.apply_gradients(grads=grads)
            return rng_key, new_actor, info

        self.update_actor = jax.jit(_update_actor)

        @chex.assert_max_traces(1)
        def _update_critic(
            rng_key: jax.random.PRNGKey,
            encoder: TrainState,
            actor: TrainState,
            critic: TrainState,
            disc: TrainState,
            s: jnp.ndarray,
            a: jnp.ndarray,
            s_next: jnp.ndarray,
            gamma: jnp.float32,
            min_target: jnp.float32,
            max_target: jnp.float32
        ):
            rng_key, noise_key = jax.random.split(rng_key)
            # FIXME: Can get rid of self below
            noise = (
                jax.random.normal(noise_key, shape=a.shape) * self.target_noise
            ).clip(-self.target_noise_clip, self.target_noise_clip)

            # Make sure the noisy action is within the valid bounds.
            fixed_target_zs = encoder.apply_fn(encoder.target_params2,
                                               s_next,
                                               method="zs")
            
            # Make sure the noisy action is within the valid bounds.
            a_next = (
                actor.apply_fn(actor.target_params,
                               s_next,
                               fixed_target_zs
                               ) + noise
            ).clip(-1, 1)

            fixed_target_zsa = encoder.apply_fn(encoder.target_params2,
                                                fixed_target_zs,
                                                a_next,
                                                method="zsa")
            
            target_qs = critic.apply_fn(
                critic.target_params,
                s_next,
                a_next,
                fixed_target_zs,
                fixed_target_zsa
            )

            r = -disc.apply_fn(disc.params, jnp.concatenate([s, s_next], axis=-1))
            target_q = r + gamma * jnp.clip(jnp.min(target_qs, axis=0), min=min_target, max=max_target)

            fixed_zs, fixed_zsa = encoder.apply_fn(encoder.target_params,
                                                   s,
                                                   a)

            def _critic_loss_fn(
                params: Params,
            ):
                q1, q2 = critic.apply_fn(params,
                                         s,
                                         a,
                                         fixed_zs,
                                         fixed_zsa)

                loss = jnp.mean(((q1 - target_q) ** 2) + ((q2 - target_q) ** 2))

                return loss, {
                    "critic/loss": loss,
                    "critic/Q1": q1.mean(),
                    "critic/Q2": q2.mean(),
                    "critic/target_Q": target_q.mean(),
                    "critic/Q1_abs": jnp.abs(q1).mean(),
                    "critic/Q2_abs": jnp.abs(q2).mean(),
                    "critic/target_Q_abs": jnp.abs(target_q).mean(),
                }

            info, grads = jax.value_and_grad(_critic_loss_fn, has_aux=True)(critic.params)
            new_critic = critic.apply_gradients(grads=grads)
            return rng_key, new_critic, info, (target_q.min(), target_q.max())

        self.update_critic = jax.jit(_update_critic)

        @chex.assert_max_traces(1)
        def _update_disc(disc: TrainState,
                         rng_key: jax.random.PRNGKey,
                         Pi_s: jnp.ndarray,
                         E_s: jnp.ndarray):

            rng_key, sample_key = jax.random.split(rng_key)
            
            @functools.partial(jax.vmap, in_axes=(None, 0))
            @functools.partial(jax.grad, argnums=1)
            def _gp_function(params,
                             inp):
                return disc.apply_fn(params,
                                     inp)[0]

            def _loss_fn(params):
                Pi_r = disc.apply_fn(
                    params, 
                    Pi_s
                )

                E_r = disc.apply_fn(
                    params, 
                    E_s
                )
                loss_gap = E_r.mean() - Pi_r.mean()
                
                # Gradient Penalty 
                alpha = jax.random.uniform(sample_key, (E_s.shape[0], 1))
                interpolated_s = alpha * Pi_s + (1 - alpha) * E_s
                
                grads = _gp_function(params, interpolated_s)
                grads_norm = jnp.sqrt(jnp.sum(grads ** 2, axis=1) + 1e-12)
                
                loss_gp = jnp.mean((grads_norm - .4) ** 2)
                
                loss = loss_gap + 10 * loss_gp

                return loss, {"disc/loss": loss,
                              "disc/Pi_r": Pi_r.mean(),
                              "disc/E_r": E_r.mean(),
                              "disc/gp": loss_gp,
                              "disc/gap": loss_gap}

            info, grads = jax.value_and_grad(_loss_fn, has_aux=True)(disc.params)
            new_disc = disc.apply_gradients(grads=grads)
            return rng_key, new_disc, info
        self._update_disc = jax.jit(_update_disc)

        @chex.assert_max_traces(1)
        def _update_targets(
            encoder_state: TrainState,
            actor_state: TrainState,
            critic_state: TrainState,
        ):
            actor_state = actor_state.replace(target_params=actor_state.params)
            critic_state = critic_state.replace(target_params=critic_state.params)

            encoder_state = encoder_state.replace(target_params2=encoder_state.target_params)
            encoder_state = encoder_state.replace(target_params=encoder_state.params)
            return encoder_state, actor_state, critic_state

        self.update_targets = jax.jit(_update_targets)

    
    def get_action(self, state, train=False, rng_key=None):
        action = self._get_action(self.encoder, self.actor, state)
        action = jax.device_get(action)
        if train:
            action += jax.random.normal(rng_key, action.shape) * self.action_noise
        return np.clip(action, -1, 1)

    def update_disc(self,
                    learner_transitions,
                    expert_transitions):
        Pi_s = jnp.concatenate([learner_transitions[0], learner_transitions[2]], axis=-1) 
        E_s  = jnp.concatenate([expert_transitions[0], expert_transitions[2]], axis=-1)
        self.rng_key, self.disc, disc_info = self._update_disc(
            self.disc,
            self.rng_key,
            Pi_s,
            E_s
        )
        return disc_info[1]
        
    def update(
        self,
        transitions,
        gamma,
    ):
        train_metrics = {}

        self.step += 1
        s, a, s_next, _, _, _ = transitions

        self.encoder, encoder_info = self.update_encoder(
            self.encoder,
            s,
            a,
            s_next
        )
        train_metrics.update(encoder_info[1])
        
        self.rng_key, self.critic, critic_info, (Q_min, Q_max) = self.update_critic(
            self.rng_key,
            self.encoder,
            self.actor,
            self.critic,
            self.disc,
            s,
            a,
            s_next,
            gamma,
            self.Q_min_target,
            self.Q_max_target
        )
        self.Q_min = min(self.Q_min, float(Q_min))
        self.Q_max = max(self.Q_max, float(Q_max))
        train_metrics.update(critic_info[1])

        if self.step % self.update_actor_frequency == 0:
            self.rng_key, self.actor, actor_info = self.update_actor(
                self.rng_key,
                self.encoder,
                self.actor,
                self.critic,
                s
            )
            actor_info[1]['Q_min'] = self.Q_min_target
            actor_info[1]['Q_max'] = self.Q_max_target
            train_metrics.update(actor_info[1])

        if self.step % self.target_update_rate == 0:
            self.encoder, self.actor, self.critic = self.update_targets(
                self.encoder,
                self.actor,
                self.critic
            )

            self.Q_min_target = self.Q_min
            self.Q_max_target = self.Q_max

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
            group=f"GAIfO_TD7/%s/%s"
            % (
                cfg.env,
                cfg.wandb_suffix
            ),
            name=str(cfg.seed),
            monitor_gym=True,
            save_code=True,
        )

    # Set up environment
    env, eval_env, disc_env = DMCEnv(cfg.env), DMCEnv(cfg.env), DMCEnv(cfg.env)
    normalization_max, normalization_min = 1000, 0

    state_size, action_size = env.observation_space.shape[0], env.action_space.shape[0]

    # Set up agent
    rng_key, network_rngkey = jax.random.split(rng_key)

    agent = GAIfO_TD7(
        network_rngkey,
        state_size,
        action_size,
        cfg
    )

    buffer = ReplayBuffer(env.observation_space.shape[0],
                          env.action_space.shape[0],
                          max_size=1e6,
                          batch_size=cfg.batch_size,
                          normalize_actions=True)

    
    policy_buffer = ReplayBuffer(env.observation_space.shape[0],
                                env.action_space.shape[0],
                                max_size=1e4,
                                batch_size=cfg.batch_size,
                                normalize_actions=True)
    
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

    cprint (f"Training state-only GAIfO (TD7) on {cfg.env} for {cfg.steps} environment steps.", "green")
    for step in pbar:

        # Collect set of transitions by running policy π in the environment
        rng_key, sample_key = jax.random.split(rng_key)
        if step < cfg.training_start:
            action = np.expand_dims(env.action_space.sample(), axis=0)
        else:
            action = agent.get_action(state, train=True, rng_key=sample_key)
        
        next_state, reward, terminal, truncation = env.step(action)
        t += 1
        done = terminal or truncation
        train_return += reward
        
        buffer.add(state[0], action[0], next_state, np.array([reward]), np.array([terminal]), done)
        
        if step > cfg.training_start:
            transitions = buffer.sample()

            train_metrics = agent.update(
                transitions,
                cfg.discount,
            )
            if done:
                metrics = {'step': step,
                           'train_return': train_return}
                metrics.update(jax.device_get(train_metrics))

                if cfg.track:
                    wandb.log(metrics)

        if done:  # If terminal (or timed out)
            pbar.set_description(
                f"Step: {step} | Return: {train_return} "
            )
            t, next_state, train_return = 0, env.reset(), 0

        state = next_state
        
        # The part to update the discriminator
        if step % cfg.update_disc_every == 0:
            expert_transitions = expert_buffer.sample(batch_size=cfg.update_disc_batch_size)
            
            ## Create a new buffer with disc_num_episodes on-policy episodes and sample from it !!!
            policy_buffer.reset()
            
            steps_collected = 0
            for i in range(cfg.disc_num_episodes):
                s = disc_env.reset()
                d = False
                while not d:
                    steps_collected += 1
                    rng_key, sample_key = jax.random.split(rng_key, 2)
                    a = agent.get_action(s, train=True, rng_key=sample_key)
                    
                    s_next, _, term, trunc = disc_env.step(a)
                    d = term or trunc
                    
                    policy_buffer.add(s[0], a[0], s_next, np.array([0]), np.array([term]), d)
                    s = s_next
            
            for _ in range(cfg.disc_update_steps):
                expert_transitions = expert_buffer.sample(batch_size=cfg.update_disc_batch_size)            
                policy_transitions = policy_buffer.sample(batch_size=cfg.update_disc_batch_size)
                disc_metrics = agent.update_disc(policy_transitions,
                                                 expert_transitions)
                
            print ("Disc updates, ", disc_metrics)
            if cfg.track:
                disc_metrics['step'] = step
                wandb.log(disc_metrics)

        # Evaluate agent and plot metrics
        if step % cfg.eval_interval == 0:
            rng_key, sample_key = jax.random.split(rng_key)
            test_returns = evaluate_agent(
                agent, eval_env, cfg.eval_episodes, rng_key=sample_key
            )

            log_metrics = {"step": step,
                            "test_returns": np.mean(test_returns)}

            cprint(f"Step: {step} | Test Returns: {np.mean(test_returns)}", "green")
            if cfg.track:
                wandb.log(log_metrics)
