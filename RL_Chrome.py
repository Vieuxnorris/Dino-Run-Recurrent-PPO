from numba import jit, cuda
import numpy as np
import mss
import cv2
import gym
import os
import optuna

import torch as th

from gym import spaces
from selenium.webdriver import Chrome
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env 
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor


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

class Dino(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}
  

  def __init__(self):
        super(Dino, self).__init__()
        try:
            driver.get(url="chrome://dino")
        except WebDriverException:
            pass
        
        self.elem = driver.find_element(By.ID, 't')
        
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=255, shape=(200, 200, 1), dtype=np.uint8)
        self.monitor = {'top':200,'left':0, 'width':1895, "height":580}
  
  def step(self, action):
        reward = 0.1
        done = False
        if driver.execute_script('return Runner.instance_.playing'):
            self.elem.send_keys(action_map[action])
        else:
            done = True
            reward = -1    
        new_observation = self.get_observation()
        info = {}
        
        return new_observation, reward, done, info
  
  @jit(forceobj=True)
  def get_observation(self):
        obs = np.asarray(mss.mss().grab(self.monitor))
        Gray = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
        Resize = cv2.resize(Gray, (200,200))
        channel = np.reshape(Resize, (200,200,1))
        return channel
    
  def reset(self):
        driver.execute_script("Runner.instance_.restart();")
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
        'n_epochs':trial.suggest_int('n_epochs', 10, 35),
        'gamma':trial.suggest_float('gamma', 0.8, 0.9999),
        'learning_rate':trial.suggest_float('learning_rate', 1e-5, 1e-4),
        'clip_range':trial.suggest_float('clip_range', 0.1, 0.4),
        'gae_lambda':trial.suggest_float('gae_lambda', 0.8, 0.99),
        'batch_size':trial.suggest_int('batch_size', 64, 1024),
        'n_steps':trial.suggest_int('n_steps', 2048, 8192)
    }

def optimize_agent(trial):
    model_params = optimize_ppo(trial) 
    model_params['n_steps'] = int((model_params['n_steps'] / model_params['batch_size'])) * model_params['batch_size']
    # Create environment 
    env = Dino()
    env = Monitor(env, LOG_DIR)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, 4, channels_order='last')

        # Create algo 
    model = PPO('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=3, **model_params, device="cuda")
    model.learn(total_timesteps=250000)
        #model.learn(total_timesteps=100000)

        # Evaluate model 
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=5)
    env.close()

    SAVE_PATH = os.path.join(OPT_DIR, 'trial_{}_best_model'.format(trial.number))
    model.save(SAVE_PATH)

    return mean_reward

if __name__ == '__main__':
    env = Dino()
    env = Monitor(env, LOG_DIR)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, 4, channels_order='last')
    
    study = optuna.create_study(direction='maximize')
    study.optimize(optimize_agent, n_trials=20, n_jobs=1)

    #callback = TrainAndLoggingCallback(check_freq=100000, save_path=CHECKPOINT_DIR)
    #model = PPO("CnnPolicy",env, n_epochs=32, learning_rate=1e-5, vf_coef=1., ent_coef=0.01, batch_size=1024, n_steps=4096, device="cuda", tensorboard_log=LOG_DIR, verbose=3)
    #model.set_env(env)
    #print("{} - {} - {} - {} - {}".format(model.n_steps, model.gamma, model.learning_rate, model.clip_range, model.gae_lambda))
    #model.learn(total_timesteps=5000000, callback=callback)
    env.close()
