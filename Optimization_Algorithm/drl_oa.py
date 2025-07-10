import os
from stable_baselines3 import DQN
from stable_baselines3 import PPO
from stable_baselines3 import DDPG
from stable_baselines3 import SAC 
import torch
import pandas as pd
from .OA_basic import Optimization_Algorithm_Basic
from .dc_env import DataCenterEnv
from Optimization_Algorithm.drl_utils import BDQ_Agent, train_BDQ, PrioritizedReplayBuffer, load_model, infer_action

class DQN_Optimization(Optimization_Algorithm_Basic):
    def __init__(self):
        super().__init__()  # 调用父类的初始化方法
        self.env = DataCenterEnv()
        # 创建一个DQN模型
        self.model = DQN(policy='MlpPolicy',env=self.env)

    def train(self,save_pth): # 需要设定模型保存路径
        try:
            self.model.learn(total_timesteps=1e5, # total_timesteps需要跟据实际情况设定
                        progress_bar=True)
        except KeyboardInterrupt:
            print("Training interrupted. Saving model.")
            self.model.save(save_pth + '_undertrained')
        else:
            self.model.save(save_pth)
            print('SAVED TO:', save_pth)

    def output(self, load_pth):
        self.model = DQN.load(load_pth)
        obs = self.env.reset() 
        action = self.model.predict(observation = obs)
        return action 
    
class BDQ_Optimization(Optimization_Algorithm_Basic):
    def __init__(self, args):
        super().__init__()  # 调用父类的初始化方法
        self.env = DataCenterEnv()
        # 创建一个DQN模型
        self.model = BDQ_Agent(args.state_dim,
                 args.hidden_dims, args.action_dims,
                 args.learning_rate,
                 args.gamma,
                 args.epsilon,
                 args.target_update,
                 args.device,
                 dqn_type='DuelingDQN')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def run(self, args, save_pth):
        self.reply_buffer = PrioritizedReplayBuffer(args.buffer_size)
        if args.mode == 'train':# 训练后直接输出结果
            best_action = train_BDQ(self.model, self.env, args.num_episodes, self.reply_buffer, args.minimal_size, args.batch_size, save_pth, args.save_interval)
        else: # 使用加载的模型进行推断
            load_model(self.model, save_pth, self.device)
            state = self.env.reset()
            best_action = infer_action(self.model, state)
        return best_action

    
class PPO_Optimization(Optimization_Algorithm_Basic):
    def __init__(self):
        super().__init__()  # 调用父类的初始化方法
        self.env = DataCenterEnv()
        # 创建一个DQN模型
        self.model = PPO(policy='MlpPolicy',env=self.env)

    def train(self,save_pth): # 需要设定模型保存路径
        try:
            self.model.learn(total_timesteps=1e5, # total_timesteps需要跟据实际情况设定
                        progress_bar=True)
        except KeyboardInterrupt:
            print("Training interrupted. Saving model.")
            self.model.save(save_pth + '_undertrained')
        else:
            self.model.save(save_pth)
            print('SAVED TO:', save_pth)

    def output(self, load_pth):
        self.model = PPO.load(load_pth)
        obs = self.env.reset() 
        action = self.model.predict(observation = obs)
        return action 
    
class DDPG_Optimization(Optimization_Algorithm_Basic):
    def __init__(self):
        super().__init__()  # 调用父类的初始化方法
        self.env = DataCenterEnv()
        # 创建一个DQN模型
        self.model = DDPG(policy='MlpPolicy',env=self.env)

    def train(self,save_pth): # 需要设定模型保存路径
        try:
            self.model.learn(total_timesteps=1e5, # total_timesteps需要跟据实际情况设定
                        progress_bar=True)
        except KeyboardInterrupt:
            print("Training interrupted. Saving model.")
            self.model.save(save_pth + '_undertrained')
        else:
            self.model.save(save_pth)
            print('SAVED TO:', save_pth)

    def output(self, load_pth):
        self.model = DDPG.load(load_pth)
        obs = self.env.reset() 
        action = self.model.predict(observation = obs)
        return action

class SAC_Optimization(Optimization_Algorithm_Basic):
    def __init__(self):
        super().__init__()  # 调用父类的初始化方法
        self.env = DataCenterEnv()
        # 创建一个DQN模型
        self.model = SAC(policy='MlpPolicy',env=self.env)

    def train(self,save_pth): # 需要设定模型保存路径
        try:
            self.model.learn(total_timesteps=1e5, # total_timesteps需要跟据实际情况设定
                        progress_bar=True)
        except KeyboardInterrupt:
            print("Training interrupted. Saving model.")
            self.model.save(save_pth + '_undertrained')
        else:
            self.model.save(save_pth)
            print('SAVED TO:', save_pth)

    def output(self, load_pth):
        self.model = SAC.load(load_pth)
        obs = self.env.reset() 
        action = self.model.predict(observation = obs)
        return action  
    