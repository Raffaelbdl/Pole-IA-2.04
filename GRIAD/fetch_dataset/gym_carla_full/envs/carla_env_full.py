import gym

import glob
import os
import sys
import random
import time
import numpy as np
import cv2
from PIL import Image
import math
from collections import deque
# import tensorflow as tf

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla


MAX_STEPS = 500


SHOW_CAM = False
CONVERTTAGTOIMAGE = False


IM_WIDTH = 320
IM_HEIGHT = 224
FOV = 100

DISCOUNT = 0.99
epsilon = 1
EPSILON_DECAY = 0.95 ## 0.9975 99975
MIN_EPSILON = 0.001

SAVEIMGPATH = {"RGB":"rgb_img/", "seg":"seg_img/"}
EXTENSION = ".jpeg"

class CarlaEnv(gym.Env):
    STEER_AMT = 1.0
    obs = np.ndarray(2, dtype=np.float32)
    obs[0] = 40 #La distance
    obs[1] = 0 #La vitesse
    metadata = {'render.modes': ['human']}
    control = False
    saveImageToFile = False
    path = "./"
    ImgObtained = False

    def __init__(self, img_incr: int=0):
        #On créé les variables
        self.img_incr = img_incr
        self.vehicle = None
        self.rgb = None
        self.seg = None
        self.sensor_list = []
        self.is_traffic_light_red = False
        self.obs = {"traffic_light": np.array([self.is_traffic_light_red]), "RGB": self.rgb, "Segmentation": self.seg}
        #Créer les actions gym
        self.observation_space = gym.spaces.Dict({"traffic_light": gym.spaces.Discrete(1), "RGB": gym.spaces.Box(low=0, high=255, shape=(IM_HEIGHT, IM_WIDTH, 9), dtype=np.uint8), "Segmentation": 		gym.spaces.Box(low=0, high=22, shape=(IM_HEIGHT, IM_WIDTH, 3), dtype=np.uint8)})
        self.action_space = gym.spaces.Discrete(4*27)
        #Initialisé l'environnement Carla
        self.client = carla.Client("127.0.0.1", 3000)
        self.client.set_timeout(10.0)
        self.client.reload_world()
        time.sleep(2)
        self.world = self.client.get_world()
        


        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter("model3")[0]
        print("Environnement Initialisé")
        print("Voiture créée")

    def reset(self):
        print("RESET")
        self.step_count = 0
        self.collision_hist = []     

        for sensor in self.sensor_list:
            sensor.stop()

        if self.vehicle is not None:
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.sensor_list + [self.vehicle]])
        
        self.world.wait_for_tick()
        self.world.wait_for_tick()
        #self.transform = carla.Transform(carla.Location(x=-75, y= 35, z= 2), carla.Rotation(yaw=-90))
        self.transform = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(self.model_3, self.transform)


        transform = carla.Transform(carla.Location(x=2.5, z=0.7), carla.Rotation(yaw= 180, pitch= 180))

        colsensor = self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(colsensor, transform, attach_to=self.vehicle)
        self.colsensor.listen(lambda event: self.collision_data(event))
        self.sensor_list.append(self.colsensor)
        #===================================================================================================================================
        #--RGB cameras----------------------------------------------------------------------------------------------------------------------
        #===================================================================================================================================
        #On commence par la caméra orienté vers la gauche

        self.rgb_cam_G = self.blueprint_library.find('sensor.camera.rgb')
        self.rgb_cam_G.set_attribute("image_size_x", f"{IM_WIDTH}")
        self.rgb_cam_G.set_attribute("image_size_y", f"{IM_HEIGHT}")
        self.rgb_cam_G.set_attribute("fov", f"{FOV}")

        transform = carla.Transform(carla.Location(x=2.5, y=-0.8, z=1.2), carla.Rotation(yaw=-70))
        sensor = self.world.spawn_actor(self.rgb_cam_G, transform, attach_to=self.vehicle)
        sensor.listen(lambda data: self.process_img(data, "RGB", 0))
        self.sensor_list.append(sensor)

        #Camera du milieu

        self.rgb_cam_M = self.blueprint_library.find('sensor.camera.rgb')
        self.rgb_cam_M.set_attribute("image_size_x", f"{IM_WIDTH}")
        self.rgb_cam_M.set_attribute("image_size_y", f"{IM_HEIGHT}")
        self.rgb_cam_M.set_attribute("fov", f"{FOV}")

        transform = carla.Transform(carla.Location(x=2.5, y=0, z=1.2))
        sensor = self.world.spawn_actor(self.rgb_cam_M, transform, attach_to=self.vehicle)
        sensor.listen(lambda data: self.process_img(data, "RGB", 1))
        self.sensor_list.append(sensor)

        
        #Camera de droite 

        self.rgb_cam_D = self.blueprint_library.find('sensor.camera.rgb')
        self.rgb_cam_D.set_attribute("image_size_x", f"{IM_WIDTH}")
        self.rgb_cam_D.set_attribute("image_size_y", f"{IM_HEIGHT}")
        self.rgb_cam_D.set_attribute("fov", f"{FOV}")

        transform = carla.Transform(carla.Location(x=2.5, y=0.8, z=1.2), carla.Rotation(yaw=70))
        sensor = self.world.spawn_actor(self.rgb_cam_D, transform, attach_to=self.vehicle)
        sensor.listen(lambda data: self.process_img(data, "RGB", 2))
        self.sensor_list.append(sensor)


        #===================================================================================================================================
        #--Semantic Segmentation Cameras----------------------------------------------------------------------------------------------------
        #===================================================================================================================================
        #On commence de même par la camera de Gauche

        self.seg_cam_G = self.blueprint_library.find('sensor.camera.semantic_segmentation')
        self.seg_cam_G.set_attribute("image_size_x", f"{IM_WIDTH}")
        self.seg_cam_G.set_attribute("image_size_y", f"{IM_HEIGHT}")
        self.seg_cam_G.set_attribute("fov", f"{FOV}")

        transform = carla.Transform(carla.Location(x=2.5, y=-0.8, z=1.2), carla.Rotation(yaw=-70))
        sensor = self.world.spawn_actor(self.seg_cam_G, transform, attach_to=self.vehicle)
        sensor.listen(lambda data: self.process_img(data, "seg", 0))
        self.sensor_list.append(sensor)


        #Camera du milieu
        self.seg_cam_M = self.blueprint_library.find('sensor.camera.semantic_segmentation')
        self.seg_cam_M.set_attribute("image_size_x", f"{IM_WIDTH}")
        self.seg_cam_M.set_attribute("image_size_y", f"{IM_HEIGHT}")
        self.seg_cam_M.set_attribute("fov", f"{FOV}")

        transform = carla.Transform(carla.Location(x=2.5, y=0, z=1.2))
        sensor = self.world.spawn_actor(self.seg_cam_M, transform, attach_to=self.vehicle)
        sensor.listen(lambda data: self.process_img(data, "seg", 1))
        self.sensor_list.append(sensor)


        #Camera de droite

        self.seg_cam_D = self.blueprint_library.find('sensor.camera.semantic_segmentation')
        self.seg_cam_D.set_attribute("image_size_x", f"{IM_WIDTH}")
        self.seg_cam_D.set_attribute("image_size_y", f"{IM_HEIGHT}")
        self.seg_cam_D.set_attribute("fov", f"{FOV}")

        transform = carla.Transform(carla.Location(x=2.5, y=0.8, z=1.2), carla.Rotation(yaw=70))
        sensor = self.world.spawn_actor(self.seg_cam_D, transform, attach_to=self.vehicle)
        sensor.listen(lambda data: self.process_img(data, "seg", 2))
        self.sensor_list.append(sensor)


        self.episode_start = time.time()
        
        self.world.wait_for_tick()
        while self.rgb.any() == None or self.seg.any() == None:
            self.world.wait_for_tick()

        if(self.vehicle.is_at_traffic_light()):
            traffic_light = self.vehicle.get_traffic_light()
            self.is_traffic_light_red = True
            self.is_traffic_light_red = False
            
        assert self.obs["traffic_light"] == self.is_traffic_light_red

        assert self.obs["RGB"].shape == (IM_HEIGHT, IM_WIDTH, 9) and self.obs["Segmentation"].shape == (IM_HEIGHT, IM_WIDTH, 3) and self.obs["traffic_light"].shape == (1,)

        return np.array(self.obs, dtype=np.float32)



    def process_img(self, image, cam_type:str, num:int):
        assert 0<= num <= 3
        if cam_type == "seg" and not CONVERTTAGTOIMAGE:
            if self.seg is None:
                self.seg = np.zeros((IM_HEIGHT, IM_WIDTH, 3))

            i = np.array(image.raw_data)
            i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
            
            self.seg[:, :, num] = i2[:, :, 2]
            self.obs["Segmentation"] = self.seg

            if self.saveImageToFile:
                im = Image.fromarray(i2[:, :, 2])
                im.save(self.path + SAVEIMGPATH["seg"] + "test" + str(num) + "_" + str(self.img_incr) + EXTENSION)
            return None  
      
        if cam_type == "seg":
            image.convert(carla.ColorConverter.CityScapesPalette)

        i = np.array(image.raw_data)
        i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
        i3 = i2[:, :, :3]

        if SHOW_CAM:
            cv2.imshow("", i3)
            cv2.waitKey(0)
        
        if cam_type == "RGB":
            if self.rgb is None:
                self.rgb = np.zeros((IM_HEIGHT, IM_WIDTH, 9))
            self.rgb[:, :, num*3:(num+1)*3] = i3
            self.obs["RGB"] = self.rgb

        elif cam_type == "seg":
            if self.seg is None:
                self.seg = np.zeros((IM_HEIGHT, IM_WIDTH, 9))
            self.seg[:, :, num*3:(num+1)*3] = i3
            self.obs["Segmentation"] = self.seg


        else:
            raise Exception("Le capteur n'est ni de type rgb ni de type seg, il ne faut pas qu'il appelle process_img")

        if self.saveImageToFile:
            im = Image.fromarray(i3)
            im.save(self.path + SAVEIMGPATH[cam_type] + "test" + str(num) + "_" + str(self.img_incr) + EXTENSION)

    def collision_data(self, event):
        self.collision_hist.append(event)

    def step(self, action: int):
        world_snapshot = self.world.wait_for_tick()
        if self.control:
            try:
                action = int(input())
            except:
                action = 0
        """
        if action == 0:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer= 0))
        elif action == 1:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.5, brake=0.0))
        elif action == 2:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, brake=0.0))
        elif action == 3:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=1*self.STEER_AMT))
        elif action == 4:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=-1*self.STEER_AMT))
        """

        self.img_incr +=1

        #kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
        reward = 0
        done = False 
        self.step_count +=1

        if len(self.collision_hist) != 0:
            done = True
            reward = -10

        elif self.step_count >= MAX_STEPS:
            print("MAX_STEPS REACHED")
            done = True
            self.step_count = 0

        action_auto = self.vehicle.get_control()
        info = {"throttle":action_auto.throttle, "steer":action_auto.steer}

        return np.array(self.obs, dtype=np.float32), reward, done, info

    def clean(self):
        for actor in self.world.get_actors().filter('vehicle.*'):
            actor.destroy()

    def heuristic_mode(self):
        self.control = True
    
    def set_autopilot(self, activate=True):
        self.vehicle.set_autopilot(activate)

    def get_action(self):
        return self.vehicle.get_control()

    def save_individual_img(self, path_folder:str):
        # self.img_incr = 0
        self.saveImageToFile = True
        self.path = path_folder

    def tick(self):
        try:
            self.world.tick(0.1)
        except:
            print("Tick not received")
