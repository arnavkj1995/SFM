# Inspired from the Jax implementation of TD3 in CleanRL (https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/td3_continuous_action_jax.py)
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
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

from envs.environments import DMCEnv
from common.utils import get_discounted_sum, update_ema, TrainState
from common.buffer import ReplayBuffer
from common.base_features import PHI_FUNCTIONS

Params = flax.core.FrozenDict[str, Any]


@dataclass
class Args:
    exp_name: str = "sfm_sac"
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
    steps: int = 1000000
    """total environment steps"""
    training_start: int = 10000
    """the step to start training"""
    step_reset_checkpoint_eps: int = 75000
    """the step to reset the checkpoint"""
    max_eps_since_update: int = 3
    """the maximum number of episodes since the last update"""
    eval_episodes: int = 10
    """the number of episodes to evaluate the agent"""
    eval_interval: int = 25000
    """the interval to evaluate the agent"""

    # Agent parameters
    discount: float = 0.995
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
    learning_starts: int = 25e3
    """timestep to start learning"""
    update_actor_frequency: int = 2
    """the frequency of training policy (delayed)"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    sample_future_prob: float = .99
    """ the probability of sampling the future state (for Hilp features only)"""
    target_update_rate: int = 250
    """the frequency of updating the max / min of SFs"""
    num_a_samples: int = 3
    """Number of action samples to take for estimating SFs"""

    # Actor parameters
    actor_lr: float = 3e-4
    """the learning rate of the optimizer"""
    actor_h_dim: int = 256
    """the hidden dimension of the actor network"""
    
    # SF Network parameters
    psi_lr: float = 3e-4
    """the learning rate of the optimizer"""
    psi_h_dim: int = 256
    """the hidden dimension of the psi network"""
    
    # Featurizer parameters
    phi_name: str = "fdm"
    """the name of the featurizer"""
    phi_lr: float = 3e-4
    """the learning rate of the optimizer"""
    phi_z_dim: int = 128
    """the latent dimension of the featurizer"""
    phi_h_dim: int = 1024
    """the hidden dimension of the featurizer"""
    

class Temperature(nn.Module):
    initial_temperature: float = 0.2

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_temp = self.param('log_temp',
                              init_fn=lambda key: jnp.full(
                                  (), jnp.log(self.initial_temperature)))
        return jnp.exp(log_temp)


class PsiNetwork(nn.Module):
    hdim: int
    output_size: int

    @nn.compact
    def __call__(self,
                 s: jnp.ndarray,
                 a: jnp.ndarray):

        sa = jnp.concatenate([s, a], -1)
        x = nn.Dense(self.hdim, kernel_init=nn.initializers.orthogonal())(sa)
        x = nn.relu(x)
        x = nn.Dense(self.hdim, kernel_init=nn.initializers.orthogonal())(x)
        x = nn.relu(x)
        x = nn.Dense(self.output_size + 1, kernel_init=nn.initializers.orthogonal())(x)
        return x


TwinPsiNetworks = nn.vmap(
    PsiNetwork,
    variable_axes={"params": 0},
    split_rngs={"params": True},
    in_axes=None,
    out_axes=0,
    axis_size=2,
)


from typing import Optional
def default_init(scale: Optional[float] = jnp.sqrt(2)):
    return nn.initializers.orthogonal(scale)


class ActorNetwork(nn.Module):
    hdim: int
    action_dim: int
    log_std_min: float = -2.5
    log_std_max: float = 2.5
    final_fc_init_scale: float = 1.0 

    @nn.compact
    def __call__(self,
                 s: jnp.ndarray,
                 rng_key: jax.random.PRNGKey,
                 temperature: float = 1.0):
        x = nn.Dense(self.hdim, kernel_init=nn.initializers.orthogonal())(s)
        x = nn.relu(x)
        x = nn.Dense(self.hdim, kernel_init=nn.initializers.orthogonal())(x)
        x = nn.relu(x)
        x = nn.Dense(2 * self.action_dim,
                     kernel_init=default_init(self.final_fc_init_scale)
                    )(x)
        
        mean, log_std = jnp.split(x, 2, axis=-1)
        log_std = nn.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)  # From SpinUp / Denis Yarats
        std = jnp.exp(log_std)

        dist = tfd.Normal(loc=mean, scale=std)
        x_t = dist.sample(seed=rng_key)
        action = jnp.tanh(x_t)
        log_prob = dist.log_prob(x_t)

        # log_prob -= jnp.log((1 - action ** 2) + 1e-6)
        log_prob = log_prob.sum(axis=-1, keepdims=True)

        return action, log_prob, jnp.tanh(mean)

class SFM_SAC:
    def __init__(
        self,
        rng_key: jax.random.PRNGKey,
        state_size: int,
        action_size: int,
        cfg
    ):
        (
            self.rng_key,
            featurizer_rng_key,
            actor_rng_key,
            init_actor_rng_key,
            psi_rng_key,
            temp_rng_key
        ) = jax.random.split(rng_key, 6)

        self.polyak_factor = cfg.polyak_factor
        self.step = 0
        self.target_noise = cfg.target_noise
        self.target_noise_clip = cfg.target_noise_clip
        self.action_noise = cfg.action_noise
        self.update_actor_frequency = cfg.update_actor_frequency
        self.target_update_rate = cfg.target_update_rate
        self.target_entropy = -action_size
        self.num_a_samples = cfg.num_a_samples

        sample_state = jnp.zeros((state_size,))
        sample_action = jnp.zeros((action_size,))

        self.featurizer = PHI_FUNCTIONS[cfg.phi_name](
            featurizer_rng_key,
            obs_shape=(state_size,),
            action_dim=action_size,
            z_dim=cfg.phi_z_dim,
            hidden_dim=cfg.phi_h_dim,
            learning_rate=cfg.phi_lr
        )

        phi = self.featurizer.encode(self.featurizer.state, sample_state[None, ...])
        self.phi_dim = phi.shape[-1]

        # Define actor
        actor_net = ActorNetwork(cfg.actor_h_dim, action_size)
        self.actor = TrainState.create(
            apply_fn=actor_net.apply,
            params=actor_net.init(actor_rng_key, sample_state[None, ...], init_actor_rng_key),
            target_params=actor_net.init(actor_rng_key, sample_state[None, ...], init_actor_rng_key),
            checkpoint=actor_net.init(actor_rng_key, sample_state[None, ...], init_actor_rng_key),
            tx=optax.chain(
                optax.adam(learning_rate=cfg.actor_lr)
            ),
        )

        # Define psi / critic
        psi_net = TwinPsiNetworks(cfg.psi_h_dim, self.phi_dim)
        self.psi = TrainState.create(
            apply_fn=psi_net.apply,
            params=psi_net.init(psi_rng_key, sample_state[None, ...], sample_action[None, ...]),
            target_params=psi_net.init(psi_rng_key, sample_state[None, ...], sample_action[None, ...]),
            tx=optax.chain(
                optax.adam(learning_rate=cfg.psi_lr)
            ),
        )
        
        temp_net = Temperature()
        self.temp_state = TrainState.create(
            apply_fn=temp_net.apply,
            params= temp_net.init(temp_rng_key),
            tx=optax.chain(
                optax.adam(learning_rate=cfg.psi_lr)
            ),
        )

        self.Pi_SF_ema = jnp.zeros((1, self.phi_dim))
        self.E_SF_ema = jnp.zeros((1, self.phi_dim))

        self.psi_max = -1e8
        self.psi_min = 1e8
        self.psi_max_target = 0.
        self.psi_min_target = 0.

        @chex.assert_max_traces(10)
        def _get_action(actor: TrainState,
                        s: jnp.ndarray,
                        rng_key: jax.random.PRNGKey):
            action, _, mean = actor.apply_fn(actor.params, s, rng_key)
            return action, mean

        self._get_action = jax.jit(_get_action)

        @chex.assert_max_traces(10)
        def _get_action_chkpt(actor: TrainState,
                              s: jnp.ndarray,
                              rng_key: jax.random.PRNGKey):
            action, _, mean = actor.apply_fn(actor.checkpoint, s, rng_key)
            return action, mean

        self._get_action_chkpt = jax.jit(_get_action_chkpt)

        @chex.assert_max_traces(10)
        def _featurize_states(featurizer_state: TrainState,
                              traj_s: jnp.ndarray):
            phi = self.featurizer.encode(featurizer_state,
                                         traj_s)
            return phi

        self._featurize_states = jax.jit(_featurize_states)
        self._featurize_states_batched = jax.vmap(
            _featurize_states, in_axes=(None, 1), out_axes=1
        )

        def get_traj_SF(
            featurizer: TrainState,
            s: jnp.ndarray,
            traj_gamma: jnp.ndarray
        ):
            phi = self._featurize_states_batched(featurizer, s)
            return get_discounted_sum(phi, traj_gamma)
        self._get_traj_SF = jax.jit(get_traj_SF)

        @chex.assert_max_traces(10)
        def _compute_episodic_gap(featurizer: TrainState,
                                  E_SF: jnp.ndarray,
                                  Pi_traj_s: jnp.ndarray,
                                  traj_gamma: jnp.ndarray):
            Pi_SF = get_traj_SF(featurizer, Pi_traj_s, traj_gamma)
            return ((E_SF - Pi_SF) ** 2).mean()

        self._compute_episodic_gap = jax.jit(_compute_episodic_gap)

        @chex.assert_max_traces(10)
        def _get_SF_samples(actor: TrainState,
                            actor_params: Params,
                            psi_net: TrainState,
                            s,
                            rng_key: jax.random.PRNGKey):
            print (rng_key)
            a, log_pi, _ = actor.apply_fn(actor_params,
                                     s,
                                     rng_key)

            psi_Q = psi_net.apply_fn(psi_net.target_params,
                                     s,
                                     a)
            return a, psi_Q, log_pi
            
        self.get_SF_samples = jax.vmap(_get_SF_samples,
                                       in_axes = (None, None, None, None, 0),
                                       out_axes = (0))
        
        @chex.assert_max_traces(10)
        def _update_temperature(
            rng_key: jax.random.PRNGKey,
            temp_state: TrainState,
            entropy: jnp.ndarray,
            target_entropy: float
        ):
            
            def _loss_fn(
                params: Params
            ):
                temp = temp_state.apply_fn(params)
                temp_loss = -temp * (entropy + target_entropy).mean()
                return temp_loss, {'temperature': temp,
                                   'temp_loss': temp_loss}
            
            info, grads = jax.value_and_grad(_loss_fn, has_aux=True)(temp_state.params)
            new_temp_net = temp_state.apply_gradients(grads=grads)
            return rng_key, new_temp_net, info
        self.update_temperature = jax.jit(_update_temperature)
    
        @chex.assert_max_traces(5)
        def _update_actor(
            rng_key: jax.random.PRNGKey,
            featurizer: TrainState,
            actor: TrainState,
            psi_net: TrainState,
            temp_state: TrainState,
            s: jnp.ndarray,
            s_next: jnp.ndarray,
            E_traj_s: jnp.ndarray,
            E_traj_gamma: jnp.ndarray,
            E_SF_ema: jnp.ndarray,
            Pi_SF_ema: jnp.ndarray,
            gamma: jnp.float32,
            ema_counter: jnp.int32,
            ema_decay: jnp.float32,
        ):
            rng_key, a_next_key = jax.random.split(rng_key)
            # print (len(a_next_keys), a_next_keys)
            _, next_psi_Qs, _ = self.get_SF_samples(actor,
                                              actor.params,
                                              psi_net,
                                              s_next,
                                              jax.random.split(a_next_key, self.num_a_samples))
            next_psi = next_psi_Qs.mean(axis=0)[0, :, :-1]
            
            E_phi = self._featurize_states(featurizer, E_traj_s)
            
            ind = -2
            E_end_s = E_traj_s[ind:ind+1]
            rng_key, E_a_key = jax.random.split(rng_key)
            E_end_a, _, _ = actor.apply_fn(actor.params,
                                          E_end_s,
                                          E_a_key)
            
            E_end_SF = psi_net.apply_fn(psi_net.params, E_end_s, E_end_a)[0][:, :-1]

            E_traj_SF = get_discounted_sum(jnp.expand_dims(E_phi[:-2], axis=0), E_traj_gamma[:-2]) + E_traj_gamma[-2] * E_end_SF

            E_SF_ema, E_SF_ema_debiased = update_ema(E_traj_SF,
                                                     E_SF_ema,
                                                     ema_decay,
                                                     ema_counter)

            alpha = temp_state.apply_fn(temp_state.params)
            rng_key, a_key = jax.random.split(rng_key)

            def _loss_fn(params: Params,
                         Pi_SF_ema: jnp.ndarray):
                # DPG loss from the Psi_network
                _, psi_Qs, a_log_probs = self.get_SF_samples(actor,
                                             params,
                                             psi_net,
                                             s,
                                             jax.random.split(a_key, self.num_a_samples))
                psi = psi_Qs.mean(axis=0)[0][:, :-1]
                

                Pi_SF = jnp.mean(psi - gamma * next_psi, axis=0, keepdims=True) / (1. - gamma)

                Pi_SF_ema, _ = update_ema(Pi_SF,
                                          Pi_SF_ema,
                                          ema_decay,
                                          ema_counter)

                L_im_gap_batch = ((E_SF_ema_debiased[0] - jax.lax.stop_gradient(Pi_SF_ema[0])) * (E_SF_ema_debiased[0] - Pi_SF[0])) * ((1. - gamma)**2)
                # L_im_gap_batch = (((E_SF_ema_debiased[0] - Pi_SF[0]) * (1. - gamma))**2)
                loss_gap = L_im_gap_batch.sum()
                
                a_log_prob = a_log_probs #[0]
                # Q = psi_Qs[0].min(axis=0)[:, -1:]
                loss_ent = (alpha * a_log_prob).mean()
                
                loss = loss_gap + loss_ent

                return loss, {
                    "actor/loss": loss,
                    "actor/loss_gap": loss_gap,
                    "actor/loss_ent": loss_ent,
                    "actor/SF_pi_mean": jnp.abs(Pi_SF_ema).mean(),
                    "actor/SF_E_mean": jnp.abs(E_SF_ema).mean(),
                    "actor/Max_L_im_gap": L_im_gap_batch.max(),
                    "actor/Min_L_im_gap": L_im_gap_batch.min(),
                    "Pi_SF_ema": Pi_SF_ema,
                    "E_SF_ema": E_SF_ema,
                    "entropy": a_log_prob
                }

            info, grads = jax.value_and_grad(_loss_fn, has_aux=True)(actor.params, Pi_SF_ema)
            new_actor = actor.apply_gradients(grads=grads)
            return rng_key, new_actor, info

        self.update_actor = jax.jit(_update_actor)

        @chex.assert_max_traces(5)
        def _update_psi(
            rng_key: jax.random.PRNGKey,
            actor: TrainState,
            psi_net: TrainState,
            featurizer: TrainState,
            temp_state: TrainState,
            s: jnp.ndarray,
            a: jnp.ndarray,
            s_next: jnp.ndarray,
            gamma: jnp.float32,
        ):
            rng_key, a_next_key = jax.random.split(rng_key)
            a_next, a_next_log_prob, _ = actor.apply_fn(actor.params, s_next, a_next_key)

            target_psi_Qs = psi_net.apply_fn(
                psi_net.target_params,
                s_next,
                a_next
            )
            
            target_psis = target_psi_Qs[:,:,:-1]
            target_Qs = target_psi_Qs[:,:,-1:]

            phi_state = self._featurize_states(featurizer, s)
            
            alpha = temp_state.apply_fn(temp_state.params)

            target_psi = phi_state + gamma * jnp.mean(target_psis, axis=0)
            target_Q = -alpha * a_next_log_prob + gamma * jnp.min(target_Qs, axis=0)

            def _psi_loss_fn(
                params: Params,
            ):
                psi_Q1, psi_Q2 = psi_net.apply_fn(params,
                                                  s,
                                                  a)

                loss_psi = (((psi_Q1[:,:-1] - target_psi) ** 2) + ((psi_Q2[:, :-1] - target_psi) ** 2)).mean()
                loss_Q = 0 #(((psi_Q1[:,-1:] - target_Q) ** 2) + ((psi_Q2[:, -1:] - target_Q) ** 2)).mean()
                loss = loss_psi + loss_Q

                return loss, {
                    "critic/SF_loss": loss_psi,
                    "critic/Q_loss": loss_Q,
                    "critic/loss": loss,
                    "critic/SF1_mean": psi_Q1[:,:-1].mean(),
                    "critic/SF2_mean": psi_Q2[:,:-1].mean(),
                    "critic/SF1_norm": jnp.abs(psi_Q1[:,:-1]).sum(axis=-1).mean(),
                    "critic/SF2_norm": jnp.abs(psi_Q2[:,:-1]).sum(axis=-1).mean(),
                }

            info, grads = jax.value_and_grad(_psi_loss_fn, has_aux=True)(psi_net.params)
            new_psi_net = psi_net.apply_gradients(grads=grads)
            return rng_key, new_psi_net, info

        self.update_psi = jax.jit(_update_psi)

        @chex.assert_max_traces(1)
        def _update_targets(
            psi_state: TrainState,
        ):
            new_psi_state = psi_state.replace(target_params=optax.incremental_update(
                psi_state.params,
                psi_state.target_params,
                1. - self.polyak_factor
            ))

            return new_psi_state

        self.update_targets = jax.jit(_update_targets)

        @chex.assert_max_traces(1)
        def _update_checkpoint(
            actor_state: TrainState,
        ):
            actor_state = actor_state.replace(checkpoint=actor_state.params)
            return actor_state

        self._update_checkpoint = jax.jit(_update_checkpoint)

    def update_checkpoint(self):
        self.actor = self._update_checkpoint(self.actor)

    def get_action(self, state, train=False, rng_key=None, use_checkpoint=False):
        if use_checkpoint:
            action, action_mean = self._get_action_chkpt(self.actor, state, rng_key)
        else:
            action, action_mean = self._get_action(self.actor, state, rng_key)

        # FIXME: Add the part of max_action
        if train:
            action = jax.device_get(action)
        else:
            action = jax.device_get(action_mean)
        return np.clip(action, -1, 1)

    def get_episodic_gap(self,
                         expert_states: jnp.ndarray,
                         policy_states: jnp.ndarray,
                         discount_function: jnp.ndarray,
                         discount: jnp.float32):
        return self._compute_episodic_gap(self.featurizer.state,
                                          self.E_SF_ema,
                                          policy_states,
                                          discount_function)

    def update(
        self,
        transitions,
        E_traj_s: np.ndarray,
        E_traj_gamma: np.ndarray,
        gamma,
    ):
        train_metrics = {}

        self.step += 1
        s, a, s_next, _, not_d, g = transitions
        d = 1. - not_d

        self.rng_key, self.psi, psi_info = self.update_psi(
            self.rng_key,
            self.actor,
            self.psi,
            self.featurizer.state,
            self.temp_state,
            s,
            a,
            s_next,
            gamma,
        )
        train_metrics.update(psi_info[1])

        # if self.step % self.update_actor_frequency == 0:
        self.rng_key, self.actor, actor_info = self.update_actor(
            self.rng_key,
            self.featurizer.state,
            self.actor,
            self.psi,
            self.temp_state,
            s,
            s_next,
            E_traj_s,
            E_traj_gamma,
            self.E_SF_ema,
            self.Pi_SF_ema,
            gamma,
            ema_counter=self.step,
            ema_decay=.95,
        )
        self.Pi_SF_ema = actor_info[1]['Pi_SF_ema']
        self.E_SF_ema = actor_info[1]['E_SF_ema']
        del actor_info[1]['Pi_SF_ema']
        del actor_info[1]['E_SF_ema']
        actor_info[1]['phi_min'] = self.psi_min_target
        actor_info[1]['phi_max'] = self.psi_max_target
        train_metrics.update(actor_info[1])
        
        # self.rng_key, self.temp_state, temp_info = self.update_temperature(
        #     self.rng_key,
        #     self.temp_state,
        #     actor_info[1]['entropy'],
        #     self.target_entropy
        # )
        # train_metrics.update(temp_info[1])
        
        self.psi = self.update_targets(self.psi)

        featurizer_info = self.featurizer.update(
            s=s,
            s_next=s_next,
            goals=g,
            a=a
        )
        train_metrics.update(featurizer_info[1])

        return train_metrics


def evaluate_agent(
    agent,
    env,
    num_episodes: int,
    use_chkpt: bool = False,
    rng_key: jax.random.PRNGKey = None,
):
    returns = []
    
    for _ in range(num_episodes):
        rewards = []
        state, terminal, truncation = env.reset(), False, False
        while not (terminal or truncation):
            rng_key, subkey = jax.random.split(rng_key)
            action = agent.get_action(state, use_checkpoint=use_chkpt, rng_key=subkey)
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
            group=f"SFM_SAC_ds_minQ_K%s_npalpha_v11/%s/%s_%s"
            % (
                cfg.num_a_samples,
                cfg.env,
                cfg.phi_name,
                cfg.wandb_suffix
            ),
            name=str(cfg.seed),
            monitor_gym=True,
            save_code=True,
        )

    # Set up environment
    env, eval_env = DMCEnv(cfg.env), DMCEnv(cfg.env)
    normalization_max, normalization_min = 1000, 0

    state_size, action_size = env.observation_space.shape[0], env.action_space.shape[0]

    # Set up agent
    rng_key, network_rngkey = jax.random.split(rng_key)

    agent = SFM_SAC(
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

    # Training
    t, state, terminal, train_return = 0, env.reset(), False, 0

    pbar = tqdm(range(0, cfg.steps + 1), unit_scale=1, smoothing=0)

    expert_trajectory, _ = env.get_expert_traj(cfg.seed)
    expert_trajectory = expert_trajectory[:1000]
    expert_discount_function = np.array(
        [np.power(cfg.discount, i) for i in range(len(expert_trajectory))]
    )

    current_traj_states, ep_timesteps = np.zeros((1, 1000, state_size)), 0
    eps_since_update, steps_since_update = 0, 0
    min_return, best_min_return = 1e8, -1e3

    max_eps_since_update = 1
    train_metrics = {}

    cprint (f"Training SFM (TD7) on {cfg.env} for {cfg.steps} environment steps", "green")
    for step in pbar:

        if step == cfg.step_reset_checkpoint_eps:
            max_eps_since_update = cfg.max_eps_since_update
            best_min_return *= 1.1

        # Collect set of transitions by running policy π in the environment
        rng_key, sample_key = jax.random.split(rng_key)
        if step < cfg.training_start:
            action = np.expand_dims(env.action_space.sample(), axis=0)
        else:
            action = agent.get_action(state, train=True, rng_key=sample_key)

        current_traj_states[0, ep_timesteps] = state
        ep_timesteps += 1
        
        next_state, reward, terminal, truncation = env.step(action)
        t += 1
        done = terminal or truncation
        train_return += reward
        
        buffer.add(state[0], action[0], next_state, np.array([reward]), np.array([terminal]), done)
            
        # Reset environment and track metrics on episode termination
        if done:  # If terminal (or timed out)
            eps_since_update += 1
            steps_since_update += ep_timesteps

            # Apply negatve as gap is distance
            ep_return = -agent.get_episodic_gap(np.expand_dims(expert_trajectory, axis=0),
                                                current_traj_states,
                                                expert_discount_function,
                                                cfg.discount)
            min_return = min(min_return, ep_return)

            # Reset episode stats
            current_traj_states, ep_timesteps = np.zeros((1, 1000, state_size)), 0

            if not step >= cfg.training_start:
                steps_since_update = 0
                eps_since_update = 0
                min_return = 1e8

            elif min_return < best_min_return or eps_since_update == max_eps_since_update:
                if (not min_return < best_min_return and eps_since_update == max_eps_since_update) or (step < cfg.step_reset_checkpoint_eps):
                    agent.update_checkpoint()
                    best_min_return = min_return

                for _ in range(steps_since_update):
                    rng_key, sample_key = jax.random.split(rng_key)
                    transitions = buffer.sample()

                    train_metrics = agent.update(
                        transitions,
                        expert_trajectory,
                        expert_discount_function,
                        cfg.discount,
                    )

                metrics = {'step': step,
                            'train_return': train_return,
                            'ep_return': ep_return,
                            'min_return': min_return,
                            'best_min_return': best_min_return}
                metrics.update(jax.device_get(train_metrics))

                steps_since_update = 0
                eps_since_update = 0
                min_return = 1e8

                if cfg.track:
                    wandb.log(metrics)

            # Store metrics and reset environment
            pbar.set_description(
                f"Step: {step} | Return: {train_return} | IM Gap: {ep_return} | Min Gap: {min_return} | Best Min Gap: {best_min_return} | Steps-update: {steps_since_update} | Eps-update: {eps_since_update} |"
            )

            t, next_state, train_return = 0, env.reset(), 0

        state = next_state

        # Evaluate agent and plot metrics
        if step % cfg.eval_interval == 0:
            rng_key, sample_key = jax.random.split(rng_key)
            test_returns_chkpt = evaluate_agent(
                agent, eval_env, cfg.eval_episodes, use_chkpt=True, rng_key=sample_key
            )

            rng_key, sample_key = jax.random.split(rng_key)
            test_returns = evaluate_agent(
                agent, eval_env, cfg.eval_episodes, use_chkpt=False, rng_key=sample_key
            )

            log_metrics = {"step": step,
                            "test_returns_chkpt": np.mean(test_returns_chkpt),
                            "test_returns": np.mean(test_returns)}

            cprint(f"Step: {step} | Test Returns: {np.mean(test_returns)} | Test Returns Chkpt: {np.mean(test_returns_chkpt)}", "green")
            if cfg.track:
                wandb.log(log_metrics)
