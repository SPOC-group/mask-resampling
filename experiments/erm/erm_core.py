"""JAX full-batch Adam for masked or full-input rank-one reconstruction."""
import jax
import jax.numpy as jnp
import numpy as np


def activate(field, activation):
    if activation in ('linear', 'identity'):
        return field
    if activation == 'relu':
        return jax.nn.relu(field)
    if activation == 'elu':
        return jax.nn.elu(field)
    if activation == 'tanh':
        return jnp.tanh(field)
    raise ValueError(f'Unsupported activation: {activation}')


def numerator(w, x, hidden, activation, reconstruction_mode='masked'):
    """Centered SSE, with the parameter-independent target norm removed.

    Full reconstruction takes hidden=None and computes
        sum_mu [Q sigma(h_mu)^2 - 2 h_mu sigma(h_mu)],
    h_mu=x_mu@w/sqrt(d), Q=||w||^2/d. No rho or masks enter.
    """
    d = w.size
    if reconstruction_mode == 'full':
        if hidden is not None:
            raise ValueError('Full reconstruction requires hidden=None, not an all-ones mask.')
        field = jnp.dot(x, w, precision='highest') / jnp.sqrt(float(d))
        a = activate(field, activation)
        q = jnp.dot(w, w, precision='highest') / d
        return jnp.sum(q*a*a - 2*field*a)
    if reconstruction_mode != 'masked':
        raise ValueError(f'Unknown reconstruction mode: {reconstruction_mode}')
    if hidden is None:
        raise ValueError('Masked reconstruction requires a hidden-mask array.')
    hp = jnp.dot(jnp.where(hidden, x, 0), w, precision='highest')
    field = (jnp.dot(x, w, precision='highest') - hp) / jnp.sqrt(float(d))
    a = activate(field, activation)
    return jnp.sum(-2 * a * hp / jnp.sqrt(float(d))
                   + a**2 * jnp.dot(hidden.astype(x.dtype), w**2, precision='highest') / d)


def make_trainer(x, masks, w, *, activation, rho=None, dynamic=False,
                 reconstruction_mode='masked', chunk_size, learning_rate,
                 lambda_r, seed, device):
    """One Adam update after accumulating every row/view gradient.

    Full mode requires masks=None, accepts no dynamic resampling, ignores rho,
    and allocates neither a mask bank nor mask RNG. Its normalized objective is
      (sum_mu ||x_mu-w sigma(w@x_mu/sqrt(d))/sqrt(d)||^2
       + lambda_r/2 ||w||^2) / (n*d).
    The target-only constant is omitted during optimization. Zero-padded rows
    contribute zero for every supported activation (all satisfy sigma(0)=0).
    """
    if reconstruction_mode not in ('masked', 'full'):
        raise ValueError(f'Unknown reconstruction mode: {reconstruction_mode}')
    if activation not in ('linear', 'identity', 'relu', 'elu', 'tanh'):
        raise ValueError(f'Unsupported activation: {activation}')
    full = reconstruction_mode == 'full'
    if full and (masks is not None or dynamic):
        raise ValueError('Full reconstruction requires masks=None and dynamic=False.')
    if not full and dynamic and (rho is None or not 0 < rho < 1):
        raise ValueError('Dynamic masked reconstruction requires rho in (0,1).')
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or min(x.shape) < 1:
        raise ValueError('x must be a nonempty matrix.')
    n, d = x.shape
    chunk_size = min(int(chunk_size), n)
    if chunk_size < 1:
        raise ValueError('chunk_size must be positive.')
    if np.asarray(w).shape != (d,):
        raise ValueError('w must have one entry per input coordinate.')
    padding = (-n) % chunk_size
    xc = jax.device_put(np.pad(x, ((0, padding), (0, 0))).reshape(
        -1, chunk_size, d), device)
    if full:
        bank = valid = key = None
    else:
        masks = np.asarray(masks, dtype=bool)
        if masks.ndim != 3 or masks.shape[0] < 1 or masks.shape[1:] != x.shape:
            raise ValueError('Invalid mask-bank shape.')
        bank = jax.device_put(np.pad(masks, ((0, 0), (0, padding), (0, 0))).reshape(
            masks.shape[0], -1, chunk_size, d), device)
        valid = jax.device_put((np.arange(n + padding) < n).reshape(
            1, -1, chunk_size, 1), device)
        key = jax.device_put(jax.random.PRNGKey(seed), device)
    w = jax.device_put(np.asarray(w, dtype=np.float32), device)
    state = (w, jnp.zeros_like(w), jnp.zeros_like(w), jnp.array(0, dtype=jnp.int32), key)
    vg = jax.value_and_grad(lambda w, x, b: numerator(w, x, b, activation, reconstruction_mode))

    def gradient(w, masks, xc):
        def body(carry, index):
            if full:
                value, grad = vg(w, xc[index], None)
            else:
                value, grad = vg(w, xc[index % xc.shape[0]],
                                 masks[index // xc.shape[0], index % xc.shape[0]])
            return (carry[0] + value, carry[1] + grad), None
        steps = xc.shape[0] if full else masks.shape[0] * xc.shape[0]
        (value, grad), _ = jax.lax.scan(body, (jnp.array(0., w.dtype), jnp.zeros_like(w)),
                                       jnp.arange(steps))
        count = float(n*d) if full else jnp.maximum(jnp.sum(masks, dtype=w.dtype), 1.)
        ridge = lambda_r if full else lambda_r * masks.shape[0]
        return (value + .5 * ridge * jnp.sum(w*w)) / count, (grad + ridge*w) / count

    def draw(key, xc, valid):
        key, subkey = jax.random.split(key)
        fresh = jax.random.bernoulli(subkey, rho, (1,) + xc.shape) & valid
        return key, fresh

    @jax.jit
    def compiled_step(state, xc, bank, valid):
        w, m, v, t, key = state
        if dynamic:
            key, masks = draw(key, xc, valid)
        else:
            masks = bank
        _, g = gradient(w, masks, xc)
        t = t + 1
        m = .9*m + .1*g
        v = .999*v + .001*g*g
        corrected_m = m / (1 - .9**t)
        corrected_v = v / (1 - .999**t)
        new_w = w - learning_rate * corrected_m / (jnp.sqrt(corrected_v) + 1e-8)
        delta = jnp.linalg.norm(new_w-w)
        return (new_w, m, v, t, key), (delta, delta/jnp.maximum(jnp.linalg.norm(new_w), 1e-30))

    @jax.jit
    def compiled_final_gradient(state, xc, bank, valid):
        w, _, _, _, key = state
        masks = draw(key, xc, valid)[1] if dynamic else bank
        return jnp.linalg.norm(gradient(w, masks, xc)[1])

    return (state, lambda state: compiled_step(state, xc, bank, valid),
            lambda state: compiled_final_gradient(state, xc, bank, valid))
