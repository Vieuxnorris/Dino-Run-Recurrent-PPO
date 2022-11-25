import numpy as np
import pyautogui
import cv2
import time
import pydirectinput
import gym
import os

from gym import spaces
from selenium.webdriver import Chrome
from stable_baselines3.common.callbacks import BaseCallback
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv


CHECKPOINT_DIR = './train/'
LOG_DIR = './logs/'

action_map = {
    0:'space',
    1:'down',
    2:'no_op'
}

driver = Chrome()
driver.set_window_position(x=-10,y=0)
driver.set_window_size(1920,1080)

class Dino(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self):
        super(Dino, self).__init__()
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=255, shape=(200, 200, 1), dtype=np.uint8)
    

  def step(self, action):
        if driver.execute_script('return Runner.instance_.playing'):
            pydirectinput.press(action_map[action])
        else:
            self.done = True
            
        new_observation = self.get_observation()
        
        reward = 1
        info = {}
        
        return new_observation, reward, self.done, info

  def get_observation(self):
        obs = np.array(pyautogui.screenshot())
        crop_region = obs[300:730,:700, :]
        Gray = cv2.cvtColor(crop_region, cv2.COLOR_BGR2GRAY)
        Resize = cv2.resize(Gray, (200,200), interpolation=cv2.INTER_CUBIC)
        channel = np.reshape(Resize, (200,200,1))
        return channel

  def reset(self):
        time.sleep(1)
        self.done = False
        pydirectinput.click(x=150, y=150)
        pydirectinput.press('space')
        return self.get_observation()
            
  def close (self):
        cv2.destroyAllWindows()

class TrainAndLoggingCallback(BaseCallback):

    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)

        return True

if __name__ == '__main__':
    #while True:
    #    screen = pyautogui.screenshot()
    #    screen_array = np.array(screen)
    #    crop_region = screen_array[300:730,:700, :]
    #    resize = cv2.resize(crop_region, (100,100))
        
        
    #    gray = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)
    #    cv2.imshow('GameCap', gray)
    #    if cv2.waitKey(1) & 0xFF == ord('q'):
    #        cv2.destroyAllWindows()
    #        break
        
    #env.close()
    #del env
    env = Dino()
    #env.reset()
    
    callback = TrainAndLoggingCallback(check_freq=1000, save_path=CHECKPOINT_DIR)
    model = RecurrentPPO('CnnLstmPolicy', env,tensorboard_log=LOG_DIR, verbose=1)
    model.learn(total_timesteps=1000000, callback=callback)