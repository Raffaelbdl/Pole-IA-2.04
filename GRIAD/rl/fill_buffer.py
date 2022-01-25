from tqdm import tqdm
import os
import cv2
import haiku as hk
import numpy as onp
import jax.numpy as jnp

from coax.experience_replay import SimpleReplayBuffer
from coax.reward_tracing import NStep

def fill_buffer(buffer:SimpleReplayBuffer, tracer:NStep, id_to_trans:dict, encoder_func: hk.Transformed, encoder_params):

    n = min(len(os.listdir("./dataset_hub/rgb_img/")) // 3, 5000)
    # n = 32
    for i in tqdm(range(n)):
        trans = str(id_to_trans[str(i)])
        npz = onp.load("./dataset_hub/dataset.npz")
        rgb0 = cv2.imread("./dataset_hub/rgb_img/test0_" + trans + ".jpeg")
        rgb1 = cv2.imread("./dataset_hub/rgb_img/test1_" + trans + ".jpeg")
        rgb2 = cv2.imread("./dataset_hub/rgb_img/test2_" + trans + ".jpeg")
        
        rgb = jnp.concatenate([rgb0, rgb1, rgb2], axis=-1)
        rgb = jnp.expand_dims(rgb, axis=0)
        rgb = rgb.astype(jnp.float32)

        s = onp.array(encoder_func.apply(encoder_params, rgb)[0], dtype=onp.float32)

        r = onp.array(npz['arr_3'][i], dtype=onp.float32)

        a1 = npz['arr_1'][i]
        a2 = npz['arr_2'][i]
        a = onp.array([a1, a2], dtype=onp.float32)

        done = npz['arr_4'][i]

        tracer.add(s, a, r, done)
        while tracer:
            buffer.add(tracer.pop())


    return buffer, tracer





