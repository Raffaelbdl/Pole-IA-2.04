import os
import numpy as np

print(len(os.listdir("./dataset/rgb_img/")))
print(len(os.listdir("./dataset/seg_img/")))
ds = np.load("./dataset/dataset.npz")
print(len(ds['arr_0']))

mem_observation_traffic = []
mem_action_throttle = []
mem_action_steer = []
mem_reward = []
mem_done = []
mem_next_observation_traffic = []

DATASET_LENGTH = 2500

for i in range(DATASET_LENGTH):

    if (os.path.isfile(f"./dataset/rgb_img/test0_{i+1}.jpeg") and 
        os.path.isfile(f"./dataset/rgb_img/test1_{i+1}.jpeg") and
        os.path.isfile(f"./dataset/rgb_img/test2_{i+1}.jpeg") and
        os.path.isfile(f"./dataset/seg_img/test0_{i+1}.jpeg") and
        os.path.isfile(f"./dataset/seg_img/test1_{i+1}.jpeg") and
        os.path.isfile(f"./dataset/seg_img/test2_{i+1}.jpeg")):
        
        mem_observation_traffic.append(ds['arr_0'][i])
        mem_action_throttle.append(ds['arr_1'][i])
        mem_action_steer.append(ds['arr_2'][i])
        mem_reward.append(ds['arr_3'][i])
        mem_done.append(ds['arr_4'][i])
        mem_next_observation_traffic.append(ds['arr_5'][i])
    
    else:
        try:
            os.remove(f"./dataset/rgb_img/test0_{i+1}.jpeg")
        except:
            pass
        try:
            os.remove(f"./dataset/rgb_img/test1_{i+1}.jpeg")
        except:
            pass
        try:
            os.remove(f"./dataset/rgb_img/test2_{i+1}.jpeg")
        except:
            pass
        try:
            os.remove(f"./dataset/seg_img/test0_{i+1}.jpeg")
        except:
            pass
        try:
            os.remove(f"./dataset/seg_img/test1_{i+1}.jpeg")
        except:
            pass
        try:
            os.remove(f"./dataset/seg_img/test2_{i+1}.jpeg")
        except:
            pass

mem_observation_traffic = np.array(mem_observation_traffic)
mem_action_throttle = np.array(mem_action_throttle)
mem_action_steer = np.array(mem_action_steer)
mem_reward = np.array(mem_reward)
mem_done = np.array(mem_done)
mem_next_observation_traffic = np.array(mem_next_observation_traffic)

np.savez("./dataset/dataset.npz", *[mem_observation_traffic, mem_action_throttle, mem_action_steer, mem_reward, mem_done, mem_next_observation_traffic])