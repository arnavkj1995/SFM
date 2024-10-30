import sys
import typing as tp

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from einops import rearrange
from flax.training import train_state


from typing import Union, Tuple
from .utils import TrainState

Params = flax.core.FrozenDict[str, tp.Any]


class _L2(nn.Module):
    dim: int

    def __call__(self, x):
        y = jnp.sqrt(self.dim) * (x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-6))
        return y


class AvgL1Norm(nn.Module):
    eps: float = 1e-6

    def __call__(self, x):
        return x / jnp.clip(jnp.abs(x).mean(axis=-1, keepdims=True), a_min=self.eps)


class Mish(nn.Module):
    def __call__(self, x):
        return x * jnp.tanh(jnp.log(1 + jnp.exp(x)))


class Simnorm(nn.Module):
    simplex_dim: int = 8

    def __call__(self, x):
        x = rearrange(x, '...(L V) -> ... L V', V=self.simplex_dim)
        x = jax.nn.softmax(x, axis=-1)
        return rearrange(x, '... L V -> ... (L V)')


def _nl(name: str, dim: int=None) -> tp.List[nn.Module]:
    """Returns a non-linearity given name and dimension"""
    # if name == "irelu":
    #     return [nn.relu(inplace=True)]
    if name == "relu":
        return [nn.activation.relu]
    if name == "elu":
        return [nn.activation.elu]
    if name == "ntanh":
        return [nn.LayerNorm(), nn.activation.tanh]
    if name == "atanh":
        return [AvgL1Norm, nn.activation.tanh]
    if name == "layernorm":
        return [nn.LayerNorm()]
    if name == "tanh":
        return [nn.activation.tanh]
    if name == "L2":
        return [_L2(dim)]
    if name == "AvgL1Norm":
        return [AvgL1Norm()]
    if name == "softplus":
        return [nn.softplus]
    if name == "mish":
        return [Mish()]
    if name == "simnorm":
        return [Simnorm()]
    if name == "sigmoid":
        return [nn.activation.sigmoid]
    raise ValueError(f"Unknown non-linearity {name}")


class MLP(nn.Module):
    layers: tp.Sequence[tp.Union[int, str]] = None

    def setup(self):
        assert len(self.layers) >= 2
        sequence: tp.List[nn.Module] = []
        assert isinstance(self.layers[0], int), "First input must provide the dimension"
        prev_dim: int = self.layers[0]
        for layer in self.layers[1:]:
            if isinstance(layer, str):
                sequence.extend(_nl(layer, prev_dim))
            else:
                assert isinstance(layer, int)
                sequence.append(nn.Dense(layer, kernel_init=nn.initializers.orthogonal()))
                prev_dim = layer
        self.network = nn.Sequential(sequence)

    def __call__(self, x):
        return self.network(x)


class TwinMLPNetworks(nn.Module):
    layers: tp.Sequence[tp.Union[int, str]] = None

    @nn.compact
    def __call__(self, x):
        phi1 = MLP(self.layers)(x)
        phi2 = MLP(self.layers)(x)
        return phi1, phi2


# Inspired from https://github.com/seohongpark/HILP/blob/master/hilp_zsrl/url_benchmark/agent/sf.py#L92
class FDMNetwork(nn.Module):
    obs_shape: tp.Tuple[int]
    action_dim: int
    hidden_dim: int
    z_dim: int

    def setup(self):
        self.obs_dim = self.obs_shape[0]

        self.phi_net = MLP([self.obs_dim, self.hidden_dim, "ntanh", self.hidden_dim, "relu", self.hidden_dim, "relu", self.z_dim, "L2"])
        self.fdm_network = MLP([self.z_dim + self.action_dim, self.hidden_dim, 'relu', self.hidden_dim, 'relu', self.obs_dim])

    def encode(self,
               s: jnp.ndarray):
        s = self.phi_net(s)
        return s

    @nn.compact
    def __call__(self,
                 s: jnp.ndarray,
                 a: jnp.ndarray):
        s = self.encode(s)
        predicted_s_next = self.fdm_network(jnp.concatenate([s, a], axis=-1))
        return predicted_s_next

class FDM:
    def __init__(
        self,
        rng_key: jax.random.PRNGKey,
        obs_shape: Union[int, Tuple[int]],
        action_dim: int,
        z_dim: int,
        hidden_dim: int,
        learning_rate: float = 1e-4,
        polyak_factor: float = .995,
        **kwargs,
    ):
        self.rng_key, network_key = jax.random.split(rng_key)
        self.z_dim = z_dim
        self.tau = 1. - polyak_factor

        sample_obs = jnp.zeros(obs_shape)
        sample_action = jnp.zeros((action_dim))

        network = FDMNetwork(obs_shape, action_dim, hidden_dim, z_dim)
        self.state = TrainState.create(
            apply_fn=network.apply,
            params=network.init(network_key, sample_obs[None,...], sample_action[None,...]),
            tx=optax.chain(
                optax.adam(learning_rate=learning_rate)
            )
        )

        def _encode(state: TrainState, obs: jnp.ndarray):
            feat = state.apply_fn(state.params, obs, method="encode")
            return feat

        self.encode = jax.jit(_encode)

        def _update_fn(
            rng_key: jax.random.PRNGKey,
            state: TrainState,
            s: jnp.ndarray,
            a: jnp.ndarray,
            s_next: jnp.ndarray,
        ):
            def _loss_fn(
                params: Params
            ):
                predicted_s_next = state.apply_fn(params, s, a)
                loss = jnp.power((s_next - predicted_s_next), 2).mean()

                return loss, {"fdm/loss": loss}

            info, grads = jax.value_and_grad(_loss_fn, has_aux=True)(state.params)
            state = state.apply_gradients(grads=grads)
            return rng_key, state, info

        self.update_fn = jax.jit(_update_fn)

    def update(
        self,
        s: jnp.ndarray,
        a: jnp.ndarray,
        s_next: jnp.ndarray,
        **kwargs
    ):
        self.rng_key, self.state, info = self.update_fn(
            self.rng_key, self.state, s, a, s_next
        )
        return info


class IDFNetwork(nn.Module):
    obs_shape: tp.Tuple[int]
    action_dim: int
    hidden_dim: int
    z_dim: int

    def setup(self):
        self.obs_dim = self.obs_shape[0]

        self.phi_net = MLP([self.obs_dim, self.hidden_dim, "ntanh", self.hidden_dim, "relu", self.hidden_dim, "relu", self.z_dim, "L2"])
        
        self.idf_network = MLP(
            [2 * self.z_dim, self.hidden_dim, "relu", self.hidden_dim, "relu", self.action_dim]
        )

    def encode(self,
               x: jnp.ndarray):
        return self.phi_net(x)

    @nn.compact
    def __call__(self, s, s_next):
        phi_s = self.encode(s)
        phi_s_next = self.encode(s_next)
        return self.idf_network(jnp.concatenate([phi_s, phi_s_next], axis=-1))


class IDF:
    def __init__(
        self,
        rng_key: jax.random.PRNGKey,
        obs_shape: Union[int, Tuple[int]],
        action_dim: int,
        z_dim: int,
        hidden_dim: int,
        learning_rate: float = 1e-4,
        polyak_factor: float = .995,
        **kwargs,
    ):
        self.rng_key, network_key = jax.random.split(rng_key)
        self.z_dim = z_dim
        self.tau = 1. - polyak_factor

        sample_obs = jnp.zeros(obs_shape)

        network = IDFNetwork(obs_shape, action_dim, hidden_dim, z_dim)

        self.state = TrainState.create(
            apply_fn=network.apply,
            params=network.init(network_key, sample_obs[None,...], sample_obs[None,...]),
            tx=optax.chain(
                optax.adam(learning_rate=learning_rate)
            )
        )

        def _encode(state: TrainState, obs: jnp.ndarray):
            feat = state.apply_fn(state.params,
                                  obs,
                                  method="encode")
            return feat

        self.encode = jax.jit(_encode)

        def _update_fn(
            rng_key: jax.random.PRNGKey,
            state: TrainState,
            s: jnp.ndarray,
            a: jnp.ndarray,
            s_next: jnp.ndarray,
        ):
            def _loss_fn(
                params: Params,
            ):
                pred_a = state.apply_fn(params,
                                              s,
                                              s_next)
                loss = jnp.power(pred_a - a, 2).mean()

                return loss, {"idf/loss": loss}

            info, grads = jax.value_and_grad(_loss_fn, has_aux=True)(state.params)
            state = state.apply_gradients(grads=grads)
            return rng_key, state, info

        self.update_fn = jax.jit(_update_fn)

    def update(
        self,
        s: jnp.ndarray,
        a: jnp.ndarray,
        s_next: jnp.ndarray,
        **kwargs
    ):
        self.rng_key, self.state, info = self.update_fn(
            self.rng_key, self.state, s, a, s_next
        )
        return info


class HILPNetwork(nn.Module):
    layers: tp.Sequence[tp.Union[int, str]] = None
    momentum: float = 0.995

    def setup(self):
        self.net = TwinMLPNetworks(self.layers)

    @nn.compact
    def __call__(self,
                 s: jnp.ndarray, 
                 g: jnp.ndarray):
        phi1_s, phi2_s = self.net(s)
        phi1_g, phi2_g = self.net(g)

        squared_dist1 = ((phi1_s - phi1_g) ** 2).sum(axis=-1)
        v1 = -jnp.sqrt(jnp.clip(squared_dist1, a_min=1e-6))
        squared_dist2 = ((phi2_s - phi2_g) ** 2).sum(axis=-1)
        v2 = -jnp.sqrt(jnp.clip(squared_dist2, a_min=1e-6))

        is_initialized = self.has_variable('batch_stats', 'mean')
        mean = self.variable('batch_stats', 'mean', jnp.zeros, phi1_s.shape[-1])

        if is_initialized:
            if self.is_mutable_collection('batch_stats'):
                mean.value = (self.momentum * mean.value +
                    (1.0 - self.momentum) * jnp.mean(phi1_s, axis=0))

        return v1, v2, phi1_s - mean.value


class HILP:
    def __init__(self,
                 rng_key: jax.random.PRNGKey,
                 obs_shape: Union[int, Tuple[int]],
                 z_dim: int,
                 hidden_dim: int,
                 hilp_discount: float = .98,
                 hilp_expectile: float = 0.5,
                 polyak_factor: float = 0.995,
                 **kwargs):

        obs_dim = obs_shape[0]
        self.rng_key, network_key = jax.random.split(rng_key)
        self.z_dim = z_dim
        self.feature_type: str = 'state'  # 'state', 'diff', 'concat'
        self.hilp_discount = hilp_discount
        self.hilp_expectile = hilp_expectile
        self.tau = 1. - polyak_factor

        if self.feature_type != 'concat':
            feature_dim = z_dim
        else:
            assert z_dim % 2 == 0
            feature_dim = z_dim // 2

        feature_dim = z_dim

        layers = [obs_dim, hidden_dim, "ntanh", hidden_dim, "relu", hidden_dim, "relu", feature_dim]
        sample_obs = jnp.zeros((obs_dim))

        network = HILPNetwork(layers)
        variables = network.init(network_key, sample_obs[None,...], sample_obs[None,...]) #, mutable=['batch_stats'])

        params = variables['params']
        batch_stats = variables['batch_stats']

        self.state = TrainState.create(
            apply_fn=network.apply,
            params=params,
            target_params=params,
            batch_stats=batch_stats,
            tx=optax.chain(
                optax.adam(learning_rate=1e-4)
            )
        )

        def encode(state: TrainState, 
                   s: jnp.ndarray):
            _, _, feat = state.apply_fn({'params': state.params,
                                         'batch_stats': state.batch_stats},
                                          s,
                                          s)
            return feat

        self.encode = jax.jit(encode)

        def _update_fn(rng_key: jax.random.PRNGKey,
                       state: TrainState,
                       s: jnp.ndarray,
                       s_next: jnp.ndarray,
                       g: jnp.ndarray):

            rewards = (jnp.linalg.norm(s - g, axis=-1) < 1e-6)
            masks = 1.0 - rewards
            rewards = rewards - 1.0

            next_v1, next_v2, _ = state.apply_fn({'params': state.target_params, 
                                                    'batch_stats': state.batch_stats},
                                                    s_next,
                                                    g)
            next_v = jnp.minimum(next_v1, next_v2)
            q = rewards + self.hilp_discount * masks * next_v

            v1_t, v2_t, _ = state.apply_fn({'params': state.target_params, 
                                            'batch_stats': state.batch_stats},
                                            s,
                                            g)
            v_t = (v1_t + v2_t) / 2
            adv = q - v_t

            q1 = rewards + self.hilp_discount * masks * next_v1
            q2 = rewards + self.hilp_discount * masks * next_v2

            def expectile_loss(adv, diff, expectile=0.7):
                weight = jnp.where(adv >= 0, expectile, (1 - expectile))
                return weight * (diff ** 2)

            def _loss_fn(params: Params):
                (v1, v2, feat), updates = state.apply_fn({'params': params,
                                                          'batch_stats': state.batch_stats},
                                                          s,
                                                          g,
                                                          mutable=['batch_stats'])
                v = (v1 + v2) / 2

                value_loss1 = expectile_loss(adv, q1 - v1, self.hilp_expectile).mean()
                value_loss2 = expectile_loss(adv, q2 - v2, self.hilp_expectile).mean()
                value_loss = value_loss1 + value_loss2

                return value_loss, ({'hilp/loss': value_loss,
                                        'hilp/v_mean': v.mean(),
                                        'hilp/v_max': v.max(),
                                        'hilp/v_min': v.min(),
                                        'hilp/abs_adv_mean': jnp.abs(adv).mean(),
                                        'hilp/adv_mean': adv.mean(),
                                        'hilp/adv_max': adv.max(),
                                        'hilp/adv_min': adv.min(),
                                        'hilp/feature_min': feat.min(),
                                        'hilp/feature_max': feat.max(),
                                        'hilp/feature_norm': jnp.sum(jnp.abs(feat), axis=-1).mean(),
                                        'hilp/accept_prob': (adv >= 0).mean()},
                                    updates)

            (loss, (info, updates)), grads = jax.value_and_grad(_loss_fn, has_aux=True)(state.params)
            state = state.apply_gradients(grads=grads, batch_stats=updates['batch_stats'])
            state = state.replace(
                target_params=optax.incremental_update(state.params, state.target_params, self.tau)
            )

            return rng_key, state, (loss, info)

        self.update_fn = jax.jit(_update_fn)

    def update(self,
               s: jnp.ndarray,
               s_next: jnp.ndarray,
               g: jnp.ndarray,
               **kwargs):

        self.rng_key, self.state, info = self.update_fn(self.rng_key,
                                                        self.state,
                                                        s,
                                                        s_next,
                                                        g)

        return info


class RandomNetwork(nn.Module):
    obs_shape: tp.Tuple[int]
    action_dim: int
    hidden_dim: int
    z_dim: int
    im_input: bool = False

    def setup(self):
        self.phi_net = MLP(
            [self.obs_shape[0], self.hidden_dim, "ntanh", self.hidden_dim, "relu", self.hidden_dim, "relu", self.z_dim, "L2"]
        )

    def encode(self,
               x: jnp.ndarray):
        return self.phi_net(x)

    @nn.compact
    def __call__(self,
                 s: jnp.ndarray):
        return self.encode(s)


class Random:
    def __init__(
        self,
        rng_key: jax.random.PRNGKey,
        obs_shape: Union[int, Tuple[int]],
        action_dim: int,
        z_dim: int,
        hidden_dim: int,
        learning_rate: float = 1e-4,
        polyak_factor: float = .995,
        **kwargs,
    ):
        self.rng_key, network_key = jax.random.split(rng_key)
        self.z_dim = z_dim
        self.tau = 1. - polyak_factor

        sample_obs = jnp.zeros(obs_shape)

        network = RandomNetwork(obs_shape, action_dim, hidden_dim, z_dim)

        self.state = TrainState.create(
            apply_fn=network.apply,
            params=network.init(network_key, sample_obs[None,...]),
            tx=optax.chain(
                optax.adam(learning_rate=learning_rate)
            )
        )

        def _encode(state: TrainState, s: jnp.ndarray):
            feat = state.apply_fn(state.params,
                                  s,
                                  method="encode")
            return feat

        self.encode = jax.jit(_encode)

    def update(
        self,
        **kwargs
    ):
        return (0, {})


PHI_FUNCTIONS = {"hilp": HILP,
                 "fdm": FDM,
                 "idm": IDF,
                 'random': Random
                }
