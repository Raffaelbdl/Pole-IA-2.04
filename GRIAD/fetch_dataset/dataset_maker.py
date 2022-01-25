import numpy as np

from gym_carla_full.envs.carla_env_full import CarlaEnv, MAX_STEPS

path = "./dataset_maker/"

env = CarlaEnv()
env.clean()
obs = env.reset()
env.set_autopilot(True)
env.save_individual_img(path_folder=path) #It must finish with a slash


memory_len = 10000

mem_observation_traffic = np.zeros((memory_len,), dtype=bool)
mem_action_throttle = np.zeros((memory_len, ), dtype=np.float32)
mem_action_steer = np.zeros((memory_len, ), dtype=np.float32)
mem_reward = np.zeros((memory_len,), dtype=np.float32)
mem_done = np.zeros((memory_len,), dtype=bool)

mem_next_observation_traffic = np.zeros((memory_len,), dtype=bool)
step = 0
done = False
reset_count = 0
while step<memory_len:
    mem_observation_traffic[step] = obs["traffic_light"][0]

    action = np.array([1,1])
    obs, reward, done, info = env.step(action)
    mem_action_throttle[step] = info["throttle"]
    mem_action_steer[step] = info["steer"]
    mem_reward[step] = reward
    mem_done[step] = done
    
    if done:
        reset_count += 1

        env = CarlaEnv(reset_count*MAX_STEPS)
        env.clean()
        obs = env.reset()
        env.set_autopilot(True)
        env.save_individual_img(path_folder=path)
        
        mem_next_observation_traffic[step] = False
    else:
        mem_next_observation_traffic[step] = obs["traffic_light"][0]

    step +=1
    
np.savez(path + "dataset.npz", *[mem_observation_traffic, mem_action_throttle, mem_action_steer, mem_reward, mem_done, mem_next_observation_traffic])
