
import json, pickle5, PIL.Image as Image
import numpy as onp
from tqdm import tqdm
import gym
from functools import partial

import jax
from jax import numpy as jnp, jit, nn, grad, vmap
from jax._src.prng import PRNGKeyArray

import haiku as hk
from haiku.initializers import VarianceScaling as Vscale
import optax

from coax.experience_replay import SimpleReplayBuffer
from coax.reward_tracing import TransitionBatch, NStep

from rl.fill_buffer import fill_buffer
from rl.gym_carla_full.envs.carla_env_full import MAX_STEPS, CarlaEnv

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
                pi_func, q_func,
                env: gym.Env, 
                pi_lr: float = 1e-4,
                q_lr: float = 1e-4,
                buffer_capacity: int = 4000,
                batch_size: int = 16,
                alpha: float = 0.2, polyak: float = 0.995):
        """Init SAC-CQL
        """
        key = jax.random.PRNGKey(0)
        self.rng, _rng1, _rng2, _rng3, _rng4, _rng5 = jax.random.split(key, 6)


        # Env
        self.env = env
        self.dummy_obs = dummy_obs = jnp.expand_dims(
                                        jnp.array(env.reset(), dtype=jnp.float32), axis=0)
        self.dummy_action = dummy_action = jnp.expand_dims(
                                            jnp.array(env.action_space.sample(), dtype=jnp.float32), axis=0)


        # Init policy
        pi_transformed = hk.without_apply_rng(hk.transform(pi_func))
        self.pi_params = pi_transformed.init(_rng1, dummy_obs, True)
        self._pi = jit(pi_transformed.apply)

        self.pi_opt = optax.adam(pi_lr)
        self.pi_opt_state = self.pi_opt.init(self.pi_params)


        # Init q
        q_transformed = hk.without_apply_rng(hk.transform(q_func))
        self.q_params_1 = q_transformed.init(_rng2, dummy_obs, dummy_action, True)
        self.q_params_2 = q_transformed.init(_rng3, dummy_obs, dummy_action, True)
        self.q_params_tar_1 = q_transformed.init(_rng4, dummy_obs, dummy_action, True)
        self.q_params_tar_2 = q_transformed.init(_rng5, dummy_obs, dummy_action, True)
        self._q = jit(q_transformed.apply)

        self.q_opt_1 = optax.adam(q_lr)
        self.q_opt_1_state = self.q_opt_1.init(self.q_params_1)
        self.q_opt_2 = optax.adam(q_lr)
        self.q_opt_2_state = self.q_opt_2.init(self.q_params_2)


        # ReplayBuffer
        self.buffer = SimpleReplayBuffer(buffer_capacity)
        self.buffer_online = SimpleReplayBuffer(buffer_capacity)
        self.batch_size = batch_size


        # Hyperparameters
        self.alpha = alpha
        self.polyak = polyak

    def select_action(self, obs_batched: jnp.ndarray, rng: PRNGKeyArray = None):
        """Selects an action given a batched observation
        
        Args:
            obs_batched: A batched jnp.ndarray with first dimension equals to 1
            rng: A PRNGKeyArray
        """
        if rng is not None:
            _rng1 = rng
        else:
            self.rng, _rng1 = jax.random.split(self.rng)

        dists = self._pi(self.pi_params, obs_batched, False)
        
        action = nn.tanh(jax.random.normal(_rng1, shape=(2,)) * dists[(1,3)] + dists[(0,2)])
        
        action = jnp.clip(action, self.env.action_space.low, self.env.action_space.high)

        return action
    
    def act(self, obs, rng=None):
        """Selects an action given an observation
        
        Args:
            obs: A jnp.ndarray
            rng: A PRNGKeyArray
        """

        obs_batched = jnp.expand_dims(obs, axis=0)

        return self.select_action(obs_batched, rng)
    
    def improve(self, offline_split: float = 1.):
        """Improves the policy and the action value

        Args:
            offline_split: A float that corresponds to the part of expert demonstrations used        
        """

        if offline_split < 1.:
            num_offline = int(self.batch_size * offline_split)
            num_online = int(self.batch_size * (1 - offline_split))
            transitions_off: TransitionBatch = self.buffer.sample(batch_size=num_offline)
            transitions_on: TransitionBatch = self.buffer.sample(batch_size=num_online)
            transitions = jax.tree_multimap(lambda x, y: jnp.concatenate([x, y], axis=0), 
                                            transitions_off, transitions_on)
        else:
            transitions: TransitionBatch = self.buffer.sample(batch_size=self.batch_size)

        targets = self.compute_targets(transitions)
        self.update_q(transitions, targets)
        self.update_pi(transitions)

    def compute_targets(self, transitions: TransitionBatch):
        """Computes the target rewards

        with entropy term from SAC
        
        Args:
            transitions: A coax TransitionBatch to sample transitions        
        """
        
        dists = self._pi(self.pi_params, transitions.S_next, False)

        means = jnp.transpose(jnp.stack([dists[:,0], dists[:,2]], axis=0))
        stds = jnp.transpose(jnp.stack([dists[:,1], dists[:,3]], axis=0))

        pis = jnp.exp(-0.5 * jnp.power(((transitions.A - means) / stds), 2)) / (stds * jnp.sqrt(2 * jnp.pi))


        targets = jnp.minimum(self._q(self.q_params_1, transitions.S_next, transitions.A, False), 
                              self._q(self.q_params_2, transitions.S_next, transitions.A, False))
        targets -= self.alpha * jnp.log(pis)

        targets = jnp.expand_dims(transitions.Rn, axis=-1) + jnp.expand_dims(transitions.In, axis=-1) * targets

        return targets

    def update_q(self, transitions: TransitionBatch, targets: jnp.ndarray):
        """Performs a single step of gradient descent on the action value

        Args:
            transitions: A coax TransitionBatch
            targets: A jnp.ndarray that corresponds to the target reward        
        """

        grads1 = grad(self.q_loss)(self.q_params_1, transitions.S, transitions.A, transitions.S_next, targets)
        updates, self.q_opt_1_state = self.q_opt_1.update(grads1, self.q_opt_1_state, self.q_params_1)
        self.q_params_1 = optax.apply_updates(self.q_params_1, updates)

        grads2 = grad(self.q_loss)(self.q_params_2, transitions.S, transitions.A, transitions.S_next, targets)
        updates, self.q_opt_2_state = self.q_opt_2.update(grads2, self.q_opt_2_state, self.q_params_2)
        self.q_params_2 = optax.apply_updates(self.q_params_2, updates)
    
    @partial(jit, static_argnums=(0))
    def q_loss(self, q_params, observations, actions, next_observations, targets):
        """Computes the loss for the action value

        Args:
            q_params: A FlatMap that contains the params of the action value
            observations: A jnp.ndarray that corresponds to the transtions observations
            actions: A jnp.ndarray that corresponds to the transitions actions
            next_observations: A jnp.ndarray that corresponds to the transitions next_observations    
            targets: A jnp.ndarray that corresponds to the target reward        
        """
        return jnp.mean(jnp.square(self._q(q_params, observations, actions, True) - targets)) \
            + self.cql_loss(q_params, observations, actions, next_observations)

    def cql_loss(self, q_params, observations, actions, next_observations):
        """Computes the supplementary loss for CQL

        Args:
            q_params: A FlatMap that contains the params of the action value
            observations: A jnp.ndarray that corresponds to the transtions observations
            actions: A jnp.ndarray that corresponds to the transitions actions
            next_observations: A jnp.ndarray that corresponds to the transitions next_observations        
        """

        self.rng, _rng1 = jax.random.split(self.rng)

        rand_actions = jax.random.uniform(_rng1, shape=actions.shape, minval=-1, maxval=1)
        new_actions = vmap(self.act)(observations)
        new_next_actions = vmap(self.act)(next_observations)
        
        q_rd = self._q(q_params, observations, rand_actions, True)
        q_cur = self._q(q_params, observations, new_actions, True)
        q_next = self._q(q_params, next_observations, new_next_actions, True)
        q_pred = self._q(q_params, observations, actions, True)

        q_concat = jnp.concatenate([q_rd, q_cur, q_next, q_pred], axis=-1)

        loss = jnp.mean(jnp.log(jnp.sum(jnp.exp(q_concat)))) - jnp.mean(q_pred)

        return loss

    def update_pi(self, transitions):
        """Performs a single step of gradient descent on the policy

        Args:
            transitions: A coax TransitionBatch   
        """

        actions = vmap(self.act)(transitions.S)

        qs = jnp.minimum(self._q(self.q_params_1, transitions.S, actions, False), self._q(self.q_params_2, transitions.S, actions, False))

        grads = grad(self.pi_loss)(self.pi_params, transitions.S, actions, qs)
        updates, self.pi_opt_state = self.pi_opt.update(grads, self.pi_opt_state, self.pi_params)
        self.pi_params = optax.apply_updates(self.pi_params, updates)
    
    @partial(jit, static_argnums=(0))
    def pi_loss(self, pi_params, observations, actions, qs):
        """Computes the loss for the policy:

        Args:
            pi_params: A FlatMap that contains the params of the policy
            observations: A jnp.ndarray that corresponds to the transtions observations
            actions: A jnp.ndarray that corresponds to the transitions actions
            qs: A jnp.ndarray that corresponds to the action values of the transitions
        """

        dists = self._pi(pi_params, observations), True

        means = jnp.transpose(jnp.stack([dists[:,0], dists[:,2]], axis=0))
        stds = jnp.transpose(jnp.stack([dists[:,1], dists[:,3]], axis=0))

        pis = jnp.exp(-0.5 * jnp.power(((actions - means) / stds), 2)) / (stds * jnp.sqrt(2 * jnp.pi))

        return jnp.mean(qs - self.alpha * jnp.log(pis))

    def update_network(self):
        """Computes the soft update of the target action values
        """

        self.q_params_tar_1 = jax.tree_multimap(lambda x, y: self.polyak*x + (1-self.polyak)*y, self.q_params_tar_1, self.q_params_1)
        self.q_params_tar_2 = jax.tree_multimap(lambda x, y: self.polyak*x + (1-self.polyak)*y, self.q_params_tar_2, self.q_params_2)

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


def func_pi(S, is_training):
    seq = hk.Sequential((
        hk.Conv2D(512, 3, 2, padding="SAME"),
        hk.Conv2D(256, 3, 1, padding="VALID"),
        hk.Conv2D(128, 3, 1, padding="VALID"),
        hk.Flatten(),
        hk.Linear(8), jax.nn.relu,
        hk.Linear(8), jax.nn.relu,
        hk.Linear(8), jax.nn.relu,
        hk.Linear(4, Vscale(2.))
    ))
    x = seq(S)
    return x

def func_q(S, A, is_training):
    seq_s = hk.Sequential((
        hk.Conv2D(512, 3, 2, padding="SAME"),
        hk.Conv2D(256, 3, 1, padding="VALID"),
        hk.Conv2D(128, 3, 1, padding="VALID"),
        hk.Flatten()))
    seq_tot = hk.Sequential((
        hk.Linear(8), jax.nn.relu,
        hk.Linear(8), jax.nn.relu,
        hk.Linear(8), jax.nn.relu,
        hk.Linear(1, w_init=jnp.zeros)
    ))
    X = jnp.concatenate((seq_s(S), A), axis=-1)
    return seq_tot(X)

def train(offline_steps: int = 5000,
          online_steps: int = 100000):

    rng = jax.random.PRNGKey(1)
    id_to_trans = json.load(open("./dataset_hub/id_to_dict.json", "r"))
    encoder_params = pickle5.load(open("./params_hub/params_at_92", "rb"))

    env = CarlaEnv(unet_func, encoder_params)
    agent = SAC(func_pi, func_q, env, buffer_capacity=4000, batch_size=24)
    tracer = NStep(n=5, gamma=0.9)

    
    agent.buffer, tracer = fill_buffer(agent.buffer, tracer, id_to_trans, unet_func, encoder_params)

    for _ in tqdm(range(offline_steps)):
        
        agent.improve()
        agent.update_network()

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

        for t in range(MAX_STEPS):
            rng = jax.random.split(rng)[0]
            a = agent.act(s, rng)
            s_next, r, done, info = env.step(a)
            print('rwd ', r, ' offline split ', offline_split)

            tracer.add(s, a, r, done)
            while tracer:
                agent.buffer_online.add(tracer.pop())

            if len(agent.buffer_online) > 16 and i % 4 == 0:
                agent.improve(offline_split)
                agent.update_network()
        
            if done:
                break
        agent.save_agent(i)
        i += 1


def image_from_pred(pred):
    img = vmap(vmap(lambda x: jnp.array(jnp.argmax(x) *255/13, dtype=jnp.uint8)))(pred)
    return img

def beauty(agent_to_load: int = None, num_steps: int = 100):
    assert agent_to_load is not None, "Cannot launch Beauty without agent id"

    rng = jax.random.PRNGKey(1)
    encoder_params = pickle5.load(open("./params_hub/params_at_92", "rb"))
    _encoder_params = pickle5.load(open("./params_hub/params_at_92", "rb"))

    env = CarlaEnv(unet_func, encoder_params)
    agent = SAC(func_pi, func_q, env, buffer_capacity=4000, batch_size=24)

    agent.load_agent(282)

    MAX_STEPS = 500
    offline_split = 0.9
    T = 0
    while T < num_steps:
        offline_split *= 0.999
        env.clean()
        s = env.reset()
        env.set_autopilot(False)

        for t in range(MAX_STEPS):

            rng = jax.random.split(rng)[0]
            a = agent.act(s, rng)
            s_next, r, done, info = env.step(a)
            print('rwd ', r, ' offline split ', offline_split)

            rgb1 = env.obs["RGB"][:,:,0:3].astype(onp.uint8)
            im = Image.fromarray(rgb1)
            im.save(f"./beauty/rgb1_{T}.jpeg")
            rgb2 = env.obs["RGB"][:,:,3:6].astype(onp.uint8)
            im = Image.fromarray(rgb2)
            im.save(f"./beauty/rgb2_{T}.jpeg")
            rgb3 = env.obs["RGB"][:,:,6:9].astype(onp.uint8)
            im = Image.fromarray(rgb3)
            im.save(f"./beauty/rgb3_{T}.jpeg")

            rgb = jnp.expand_dims(jnp.concatenate([rgb1.astype(onp.float32), rgb2.astype(onp.float32), rgb3.astype(onp.float32)], axis=-1), axis=0)
            pred = _unet_full(_encoder_params, rgb)[0]
            pred1 = onp.array(image_from_pred(pred[:,:,0:13]))
            im = Image.fromarray(pred1)
            im.save(f"./beauty/pred1_{T}.jpeg")
            pred2 = onp.array(image_from_pred(pred[:,:,13:26]))
            im = Image.fromarray(pred2)
            im.save(f"./beauty/pred2_{T}.jpeg")
            pred3 = onp.array(image_from_pred(pred[:,:,26:39]))
            im = Image.fromarray(pred3)
            im.save(f"./beauty/pred3_{T}.jpeg")

            seg1 = (onp.repeat(onp.expand_dims(env.obs["Segmentation"][:,:,0], axis=-1), 3, axis=-1)*255/13).astype(onp.uint8)
            im = Image.fromarray(seg1)
            im.save(f"./beauty/seg1_{T}.jpeg")
            seg2 = (onp.repeat(onp.expand_dims(env.obs["Segmentation"][:,:,1], axis=-1), 3, axis=-1)*255/13).astype(onp.uint8)
            im = Image.fromarray(seg2)
            im.save(f"./beauty/seg2_{T}.jpeg")
            seg3 = (onp.repeat(onp.expand_dims(env.obs["Segmentation"][:,:,2], axis=-1), 3, axis=-1)*255/13).astype(onp.uint8)
            im = Image.fromarray(seg3)
            im.save(f"./beauty/seg3_{T}.jpeg")
        
            if done:
                break



if __name__ == "__main__":
    train()
        