
import json, pickle5, PIL.Image as Image
import numpy as onp
from tqdm import tqdm
import gym
from functools import partial

import jax
from jax import numpy as jnp, jit, nn, grad, vmap, value_and_grad as vgrad
from jax._src.prng import PRNGKeyArray

import haiku as hk
from haiku.initializers import VarianceScaling as Vscale
from haiku.data_structures import to_mutable_dict, to_haiku_dict
import optax

from rl.buffer import Buffer, TransitionBatch
# from rl.fill_buffer import fill_buffer
from rl.gym_carla_full.envs.carla_env_full import MAX_STEPS, CarlaEnv

# TODO Add offline memory (fill buffer etc)

class UnetSimple(hk.Module):
    """Encoder Part of the UNet Module    
    """
    def __init__(self, num_classes: int, name: str=None):
        super().__init__(name)
        self.num_classes = num_classes
    
    def __call__(self, inputs: jnp.ndarray):

        m1 = hk.Conv2D(64, (3,3), 1, w_init=Vscale(2.), padding="SAME")(inputs)
        m1 = nn.relu(m1)

        m2 = hk.MaxPool(window_shape=2, strides=2, padding="SAME")(m1)
        m2 = hk.Conv2D(128, (3,3), 1, w_init=Vscale(2.), padding="SAME")(m2)
        m2 = nn.relu(m2)

        m3 = hk.MaxPool(window_shape=2, strides=2, padding="SAME")(m2)
        m3 = hk.Conv2D(256, (3,3), 1, w_init=Vscale(2.), padding="SAME")(m3)
        m3 = nn.relu(m3)

        m4 = hk.MaxPool(window_shape=2, strides=2, padding="SAME")(m3)
        m4 = hk.Conv2D(512, (3,3), 1, w_init=Vscale(2.), padding="SAME")(m4)
        m4 = nn.relu(m4)

        m5 = hk.MaxPool(window_shape=2, strides=2, padding="SAME")(m4)
        m5 = hk.Conv2D(1024, (3,3), 1, w_init=Vscale(2.), padding="SAME")(m5)
        m5 = nn.relu(m5)

        return m5

class UnetFull(hk.Module):

    def __init__(self, num_classes, name: str=None):
        super().__init__(name)
        self.num_classes = num_classes
    
    def __call__(self, inputs):

        m1 = hk.Conv2D(64, (3,3), 1, w_init=Vscale(2.), padding="SAME")(inputs)
        m1 = nn.relu(m1)

        m2 = hk.MaxPool(window_shape=2, strides=2, padding="SAME")(m1)
        m2 = hk.Conv2D(128, (3,3), 1, w_init=Vscale(2.), padding="SAME")(m2)
        m2 = nn.relu(m2)

        m3 = hk.MaxPool(window_shape=2, strides=2, padding="SAME")(m2)
        m3 = hk.Conv2D(256, (3,3), 1, w_init=Vscale(2.), padding="SAME")(m3)
        m3 = nn.relu(m3)

        m4 = hk.MaxPool(window_shape=2, strides=2, padding="SAME")(m3)
        m4 = hk.Conv2D(512, (3,3), 1, w_init=Vscale(2.), padding="SAME")(m4)
        m4 = nn.relu(m4)

        m5 = hk.MaxPool(window_shape=2, strides=2, padding="SAME")(m4)
        m5 = hk.Conv2D(1024, (3,3), 1, w_init=Vscale(2.), padding="SAME")(m5)
        m5 = nn.relu(m5)

        m6 = hk.Conv2DTranspose(512, (3,3), 2, w_init=Vscale(2.), padding="SAME")(m5)
        m6 = jnp.concatenate((m4, m6), axis=-1)
        m6 = hk.Conv2D(512, (3,3), 1, w_init=Vscale(2.), padding="SAME")(m6)
        m6 = nn.relu(m6)

        m7 = hk.Conv2DTranspose(256, (3,3), 2, w_init=Vscale(2.), padding="SAME")(m6)
        m7 = jnp.concatenate((m3, m7), axis=-1)
        m7 = hk.Conv2D(256, (3,3), 1, w_init=Vscale(2.), padding="SAME")(m7)
        m7 = nn.relu(m7)

        m8 = hk.Conv2DTranspose(128, (3,3), 2, w_init=Vscale(2.), padding="SAME")(m7)
        m8 = jnp.concatenate((m2, m8), axis=-1)
        m8 = hk.Conv2D(128, (3,3), 1, w_init=Vscale(2.), padding="SAME")(m8)
        m8 = nn.relu(m8)

        m9 = hk.Conv2DTranspose(64, (3,3), 2, w_init=Vscale(2.), padding="SAME")(m8)
        m9 = jnp.concatenate((m1, m9), axis=-1)
        m9 = hk.Conv2D(64, (3,3), 1, w_init=Vscale(2.), padding="SAME")(m9)
        m9 = nn.relu(m9)

        m10 = hk.Conv2D(self.num_classes*3, (3,3), 1, w_init=Vscale(2.), padding="SAME")(m9)

        return m10


@hk.without_apply_rng
@hk.transform
def unet_func(inputs: jnp.ndarray):

    net = UnetSimple(13, "unet")

    return net(inputs)

@hk.without_apply_rng
@hk.transform
def _unet_full(inputs: jnp.ndarray):

    net = UnetFull(13, "unet")

    return net(inputs)

log2pi = onp.log(2*onp.pi)


class SquashedNormal():

    def __init__(self, 
                 action_space: gym.spaces.Box,
                 log_std_min: float, 
                 log_std_max: float):

        self.log_std_min = jnp.array(log_std_min)
        self.log_std_max = jnp.array(log_std_max)

        self.high = jnp.array(action_space.high)
        self.low = jnp.array(action_space.low)


    @partial(jit, static_argnums=(0))
    def sample(self, 
               rng: PRNGKeyArray, 
               Mu: jnp.ndarray, 
               logStd: jnp.ndarray):
        
        _Mu = (jnp.tanh(Mu) - 1) * (self.high - self.low) / 2 + self.high
        _logStd = (jnp.tanh(logStd) - 1) * (self.log_std_max - self.log_std_min) / 2 + self.log_std_max
        _Std = jnp.exp(_logStd) + 1e-6

        U = jax.random.normal(rng, shape=Mu.shape) * _Std + _Mu
        A = jnp.tanh(U)

        logP = -0.5 * _Mu.shape[-1] * log2pi - jnp.sum(_logStd, axis=-1) - 0.5 * jnp.sum(jnp.square((U - _Mu) / _Std), axis=-1) \
                - jnp.sum(jnp.log(1 - jnp.square(A) + 1e-9), axis=-1)

        return A, logP


class SAC():
    """Our implementation of Soft Actor Critic (SAC) + Conservative Q Learning (CQL)

    Attributes:
        pi_func: A function that calls the policy nn
        q_func: A function that calls the q value nn
        env: A gym environment
        buffer_capacity: A int that indicates the capacity of the replay buffer
        batch_size: A int that  indicates the number of transitions called at each training
            step
        alpha: A float for the entropy term of SAC
            '\alpha * log(pi(s))'
        polyak: A float that is the factor of soft update for target network

        pi_params: A FlatMap that contains the params of the policy nn
        pi_opt: A coax GradientTransformation for the policy

        q_params: A FlatMap that contains the params of the q value nn
        q_opt: A coax GradientTransformation for the q value

        buffer: A coax ReplayBuffer for the expert demonstrations
        buffer_online: A coax ReplayBuffer for the online demonstrations
    """

    def __init__(self, 
                env: gym.Env, 
                seed: int = 0,
                pi_lr: float = 1e-4,
                q_lr: float = 1e-4,
                buffer_capacity: int = 4000,
                batch_size: int = 16,
                gamma: float = 0.99,
                alpha: float = 0.2, 
                tau: float = 0.005):
        """Init SAC-CQL
        """
        self.rng, rng1, rng2, rng3 = jax.random.split(jax.random.PRNGKey(seed), 4)

        
        observation_space = env.observation_space
        action_space = env.action_space

        dummy_S = jnp.expand_dims(jnp.zeros(observation_space.shape, dtype=jnp.float32), axis=0)
        dummy_A = jnp.expand_dims(jnp.zeros(action_space.shape, dtype=jnp.float32), axis=0)

        self.squashed_normal = SquashedNormal(action_space, -4., 0.)

        def func_pi(S):
            """
                outputs scaled actions
            """
            seq = hk.Sequential((
                hk.Conv2D(512, 3, 2, padding="SAME"),
                hk.Conv2D(256, 3, 1, padding="VALID"),
                hk.Conv2D(128, 3, 1, padding="VALID"),
                hk.Flatten(),
                hk.Linear(128), jax.nn.relu,
                hk.Linear(128), jax.nn.relu,
                hk.Linear(2 * action_space.shape[0], jnp.zeros),
                hk.Reshape((action_space.shape[0], 2))
            ))
            Dist = seq(S)
            Mu = Dist[..., 0]
            logStd = Dist[..., 1]

            return {"Mu": Mu, "logStd": logStd}
        
        def func_q(S, A):
            seq_s = hk.Sequential((
                hk.Conv2D(512, 3, 2, padding="SAME"),
                hk.Conv2D(256, 3, 1, padding="VALID"),
                hk.Conv2D(128, 3, 1, padding="VALID"),
                hk.Flatten()))
            seq_tot = hk.Sequential((
                hk.Linear(128), jax.nn.relu,
                hk.Linear(128), jax.nn.relu,
                hk.Linear(1)
            ))
            Q = seq_tot(jnp.concatenate((seq_s(S), A), axis=-1))

            return Q
        
        # Policy
        pi_transformed = hk.without_apply_rng(hk.transform(func_pi))
        self.pi_params = pi_transformed.init(rng1, dummy_S)
        self._pi = jit(pi_transformed.apply)

        self.pi_opt = optax.adam(pi_lr)
        self.pi_opt_state = self.pi_opt.init(self.pi_params)

        # Q
        q_transformed = hk.without_apply_rng(hk.transform(func_q))
        self.q_params_1 = q_transformed.init(rng2, dummy_S, dummy_A)
        self.q_params_2 = q_transformed.init(rng3, dummy_S, dummy_A)
        self.q_params_tar_1 = to_haiku_dict(to_mutable_dict(self.q_params_1))
        self.q_params_tar_2 = to_haiku_dict(to_mutable_dict(self.q_params_2))
        self._q = jit(q_transformed.apply)

        self.q_opt = optax.adam(q_lr)
        self.q_opt_state_1 = self.q_opt.init(self.q_params_1)
        self.q_opt_state_2 = self.q_opt.init(self.q_params_2)

        # Buffer
        self.buffer = Buffer(buffer_capacity)
        self.batch_size = batch_size

        # Hyperparameters
        self.gamma = gamma
        self.alpha = alpha # to make differentiable in v2
        self.tau = tau

        def apply_gradients(grads, opt: optax.GradientTransformation, opt_state, params):
            updates, opt_state = opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state
        self.apply_gradients = partial(jit, static_argnums=(1))(apply_gradients)

    def select_action(self, rng: PRNGKeyArray, s: jnp.ndarray):

        S = jnp.expand_dims(s, axis=0)
        Dist = self._pi(self.pi_params, S)
        Mu, logStd = Dist["Mu"], Dist["logStd"]

        A, logP = self.squashed_normal.sample(rng, Mu, logStd)

        return A[0], logP[0]
    
    def improve(self):

        transitions = self.buffer.sample(self.batch_size)
        S = jnp.array(transitions.S, dtype=jnp.float32)
        A = jnp.array(transitions.A, dtype=jnp.float32) 
        R = jnp.array(transitions.R, dtype=jnp.float32)
        Done = jnp.array(transitions.Done)
        S_next = jnp.array(transitions.S_next, dtype=jnp.float32)

        self.rng, rng1, rng2 = jax.random.split(self.rng, 3)

        loss_q = self.update_q(rng1, S, A, R, Done, S_next)
        loss_pi = self.update_pi(rng2, S)
        self.q_params_tar_1 = self.soft_update(self.q_params_tar_1, self.q_params_1)
        self.q_params_tar_2 = self.soft_update(self.q_params_tar_2, self.q_params_2)

        return loss_q, loss_pi


    def update_q(self, rng, S, A, R, Done, S_next):
        
        Tar = jax.lax.stop_gradient(self.compute_Tar(rng, self.pi_params, self.q_params_tar_1, self.q_params_tar_2, R, Done, S_next))

        loss_q_1, grads_1 = vgrad(self.loss_q, has_aux=False)(self.q_params_1, self.pi_params, S, A, S_next, Tar)
        self.q_params_1, self.q_opt_state_1 = self.apply_gradients(grads_1, self.q_opt, 
                                                                   self.q_opt_state_1, self.q_params_1)

        loss_q_2, grads_2 = vgrad(self.loss_q, has_aux=False)(self.q_params_2, self.pi_params, S, A, S_next, Tar)
        self.q_params_2, self.q_opt_state_2 = self.apply_gradients(grads_2, self.q_opt, 
                                                                   self.q_opt_state_2, self.q_params_2)

        return (loss_q_1 + loss_q_2) / 2

    
    @partial(jit, static_argnums=(0))
    def compute_Tar(self, rng, pi_params, q_params_tar_1, q_params_tar_2, R, Done, S_next):
        nDone = jnp.where(Done, 0., 1.)

        Dist_next = self._pi(pi_params, S_next)
        Mu, logStd = Dist_next["Mu"], Dist_next["logStd"]
        A_next, logP_next = self.squashed_normal.sample(rng, Mu, logStd)

        Q_mini_next = jnp.minimum(
            self._q(q_params_tar_1, S_next, A_next),
            self._q(q_params_tar_2, S_next, A_next)
        )

        Tar = jnp.expand_dims(R, axis=-1) + \
            self.gamma * jnp.expand_dims(nDone, axis=-1) * \
                (Q_mini_next - self.alpha * jnp.expand_dims(logP_next, axis=-1))

        return Tar
     

    @partial(jit, static_argnums=(0))
    def loss_q(self, q_params, pi_params, S, A, S_next, Tar):
        loss = jnp.square(self._q(q_params, S, A) - Tar)
        return jnp.mean(loss) + self.cql_loss(q_params, pi_params, S, A, S_next)


    def cql_loss(self, q_params, pi_params, observations, actions, next_observations):
        """Computes the supplementary loss for CQL

        Args:
            q_params: A FlatMap that contains the params of the action value
            observations: A jnp.ndarray that corresponds to the transtions observations
            actions: A jnp.ndarray that corresponds to the transitions actions
            next_observations: A jnp.ndarray that corresponds to the transitions next_observations        
        """

        self.rng, _rng1 = jax.random.split(self.rng, 2)

        rand_actions = jax.random.uniform(_rng1, shape=actions.shape, minval=-1, maxval=1)
        new_actions = self._pi(pi_params, observations)
        new_next_actions = self._pi(pi_params, next_observations)
        
        q_rd = self._q(q_params, observations, rand_actions, True)
        q_cur = self._q(q_params, observations, new_actions, True)
        q_next = self._q(q_params, next_observations, new_next_actions, True)
        q_pred = self._q(q_params, observations, actions, True)

        q_concat = jnp.concatenate([q_rd, q_cur, q_next, q_pred], axis=-1)

        loss = jnp.mean(jnp.log(jnp.sum(jnp.exp(q_concat)))) - jnp.mean(q_pred)

        return loss


    def update_pi(self, rng, S):
        loss_pi, grads = vgrad(self.loss_pi)(self.pi_params, self.q_params_1, self.q_params_2, rng, S)
        self.pi_params, self.pi_opt_state = self.apply_gradients(grads, self.pi_opt,
                                                                 self.pi_opt_state, self.pi_params)
        return loss_pi

    
    @partial(jit, static_argnums=(0))
    def loss_pi(self, pi_params, q_params_1, q_params_2, rng: PRNGKeyArray, S):

        Dist = self._pi(pi_params, S)
        Mu, logStd = Dist["Mu"], Dist["logStd"] 
        A, logP = self.squashed_normal.sample(rng, Mu, logStd)
    
        Q_mini = jnp.minimum(
            self._q(q_params_1, S, A),
            self._q(q_params_2, S, A)
        )

        loss_pi = - Q_mini + self.alpha * jnp.expand_dims(logP, axis=-1)
        
        return jnp.mean(loss_pi)


    @partial(jit, static_argnums=(0))
    def soft_update(self, tar_params, params):
        return jax.tree_multimap(lambda x, y: (1 - self.tau) * x + self.tau * y, tar_params, params)
        

    def save_agent(self, i):
        pickle5.dump(self.pi_params, open(f"./sac_params/pi_params_{i}", "wb"))
        pickle5.dump(self.q_params_1, open(f"./sac_params/q_params_1_{i}", "wb"))
        pickle5.dump(self.q_params_2, open(f"./sac_params/q_params_2_{i}", "wb"))
        pickle5.dump(self.q_params_tar_1, open(f"./sac_params/q_params_tar_1_{i}", "wb"))
        pickle5.dump(self.q_params_tar_2, open(f"./sac_params/q_params_tar_2_{i}", "wb"))


    def load_agent(self, i):
        pickle5.load(open(f"./sac_params/pi_params_{i}", "rb"))
        pickle5.load(open(f"./sac_params/q_params_1_{i}", "rb"))
        pickle5.load(open(f"./sac_params/q_params_2_{i}", "rb"))
        pickle5.load(open(f"./sac_params/q_params_tar_1_{i}", "rb"))
        pickle5.load(open(f"./sac_params/q_params_tar_2_{i}"    , "rb"))


def train(offline_steps: int = 5000,
          online_steps: int = 100000):

    rng = jax.random.PRNGKey(1)
    id_to_trans = json.load(open("./dataset_hub/id_to_dict.json", "r"))
    encoder_params = pickle5.load(open("./params_hub/params_at_92", "rb"))

    env = CarlaEnv(unet_func, encoder_params)
    agent = SAC(env, buffer_capacity=4000, batch_size=24)
    

    for _ in tqdm(range(offline_steps)):
        
        agent.improve()

    agent.save_agent(0)
        
    print("Offline Done")

    MAX_STEPS = 500
    offline_split = 0.9
    i = 1
    while i <= 1000000:
        offline_split *= 0.999
        env.clean()
        s = env.reset()
        env.set_autopilot(False)

        eps = 0
        ep_r = 0.
        for t in range(MAX_STEPS):
            rng = jax.random.split(rng)[0]
            a, logp = agent.select_action(s, rng)
            s_next, r, done, info = env.step(a)
            ep_r += r

            agent.buffer.add(TransitionBatch._from_singles(s, a, r, done, s_next, logp))

            if len(agent.buffer) > agent.batch_size and i % 4 == 0:
                loss_q, loss_pi = agent.improve()
        
            if done:
                s = env.reset()
                if len(loss_q) > 0:
                    print(f"episode {eps} | ep rwd {ep_r:.2f} | loss q {sum(loss_q)/len(loss_q):.2f} | loss pi {sum(loss_pi)/len(loss_pi):.2f} ")

                eps += 1
                ep_r = 0

        else:
            s = s_next

        agent.save_agent(i)
        i += 1


def image_from_pred(pred):
    img = vmap(vmap(lambda x: jnp.array(jnp.argmax(x) *255/13, dtype=jnp.uint8)))(pred)
    return img


if __name__ == "__main__":

    train()
        