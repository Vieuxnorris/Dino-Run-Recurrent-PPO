import numpy as np
import pyautogui
import cv2
import time
import pydirectinput
import gym
import os

import optuna
from matplotlib import pyplot as plt


from gym import spaces
from selenium.webdriver import Chrome
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.options import Options
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env 
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common import monitor


CHECKPOINT_DIR = 'D:\\programmation\\python\\IA\\baseline\\RL\\Training'
LOG_DIR = 'D:\\programmation\\python\\IA\\baseline\\RL\\Training\\Saved logs Dino\\'
OPT_DIR = 'D:\\programmation\\python\\IA\\baseline\\RL\\Training\\Saved OPT Dino\\'

action_map = {
    0:Keys.ARROW_UP,
    1:Keys.ARROW_DOWN,
    2:Keys.NULL
}

driver = Chrome()
options = Options()
driver.set_window_position(x=-10,y=0)
driver.set_window_size(1920,1080)
try:
    options.add_argument('--mute-audio')
    options.add_argument("disable-infobars")
    driver.get(url="chrome://dino")
except WebDriverException:
    pass

elem = driver.find_element(By.ID, 't')

class Dino(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self):
        super(Dino, self).__init__()
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=255, shape=(200, 200, 1), dtype=np.uint8)
        
  def step(self, action):
        done = False
        if driver.execute_script('return Runner.instance_.playing'):
            reward = 0.1
            elem.send_keys(action_map[action])
        else:
            done = True
            reward = -1    
        new_observation = self.get_observation()
        
        info = {}
        
        return new_observation, reward, done, info

  def get_observation(self):
        obs = np.array(pyautogui.screenshot())
        crop_region = obs[300:730,:700, :]
        Gray = cv2.cvtColor(crop_region, cv2.COLOR_BGR2GRAY)
        Resize = cv2.resize(Gray, (200,200), interpolation=cv2.INTER_CUBIC)
        channel = np.reshape(Resize, (200,200,1))
        return channel

  def reset(self):
        time.sleep(1)
        driver.execute_script("Runner.instance_.restart();")
        elem.send_keys(Keys.SPACE)
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

def optimize_ppo(trial):
    return {
        'n_steps':trial.suggest_int('n_steps', 2048, 8192),
        'gamma':trial.suggest_float('gamma', 0.8, 0.9999),
        'learning_rate':trial.suggest_float('learning_rate', 1e-5, 1e-4),
        'clip_range':trial.suggest_float('clip_range', 0.1, 0.4),
        'gae_lambda':trial.suggest_float('gae_lambda', 0.8, 0.99),
    }

def optimize_agent(trial):
    model_params = optimize_ppo(trial) 

        # Create environment 
    env = Dino()
    env = Monitor(env, LOG_DIR)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, 4, channels_order='last')
        # Create algo 
    model = PPO('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=1, **model_params)
    model.learn(total_timesteps=100000)

        # Evaluate model 
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=5)
    env.close()

    SAVE_PATH = os.path.join(OPT_DIR, 'trial_{}_best_model'.format(trial.number))
    model.save(SAVE_PATH)

    return mean_reward


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
    env = Dino()
    #env.reset()
    callback = TrainAndLoggingCallback(check_freq=1000, save_path=CHECKPOINT_DIR)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(optimize_agent, n_trials=50, n_jobs=1)
    
    #model = RecurrentPPO("CnnLstmPolicy", env, tensorboard_log=LOG_DIR, verbose=1, device="cuda", learning_rate=0.00001)
    
    #model = RecurrentPPO.load('./RL/Training/Saved train Dino/best_model_83000.zip')
    #model = RecurrentPPO('CnnLstmPolicy', env,tensorboard_log=LOG_DIR, verbose=1, learning_rate=0.00003, n_steps=2048, batch_size=64)
    #model.learn(total_timesteps=1000000, callback=callback)
    #model = RecurrentPPO.load('RL\\Training\\Saved train Dino\\best_model_56000.zip')
    #model.set_env(env)
    #model.learn(total_timesteps=1000000, callback=callback)
    env.close()
