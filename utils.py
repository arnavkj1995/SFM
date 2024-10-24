import jax
import chex
import flax
import jax.numpy as jnp
from flax.training import train_state


class TrainState(train_state.TrainState):
    target_params: flax.core.FrozenDict = None
    target_params2: flax.core.FrozenDict = None
    checkpoint: flax.core.FrozenDict = None
    batch_stats: flax.core.FrozenDict = None


@jax.jit
@chex.assert_max_traces(10)
def get_discounted_sum(x: jnp.ndarray, 
                       gamma: jnp.ndarray):
    return jnp.einsum("ijk,j->ik", x, gamma)


@jax.jit
@chex.assert_max_traces(10)
def onehot_slicing(array, 
                   indices,
                   num_classes=1000):
    """_summary_

    Args:
        array (_type_): _description_
        indices (_type_): _description_

    Returns:
        _type_: _description_
    """
    one_hot_indices = jax.nn.one_hot(indices, num_classes=num_classes - 2)
    return jnp.einsum("jk,j->k", array, one_hot_indices)


onehot_slicing_batched = jax.vmap(onehot_slicing, in_axes=(0, 0), out_axes=0)


@jax.jit
@chex.assert_max_traces(10)
def update_ema(value: jnp.ndarray,
               ema: jnp.ndarray,
               decay: jnp.float32,
               step: jnp.int32):
    ema = ema * decay + value * (1 - decay)
    ema_debiased = ema / (1 - decay ** step)
    return ema, ema_debiased
