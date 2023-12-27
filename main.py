import os
import re
import argparse
from stable_baselines3 import PPO, DQN
from train import TrainAndLoggingCallback
from train import env

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train the model with specified total timesteps.')
    parser.add_argument('--timesteps', type=int, help='Total timesteps for training')
    return parser.parse_args()


if __name__ == '__main__':
  CHECKPOINT_DIR = './train/'
  LOG_DIR = './logs/'

  args = parse_arguments()
  total_timesteps = args.timesteps
  assert isinstance(total_timesteps, int) and total_timesteps > 0, "Error: total_timesteps must be a positive integer."
  callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOINT_DIR)

  model = PPO('MultiInputPolicy', env,  verbose=1, tensorboard_log=LOG_DIR, learning_rate=0.001, )

  model.learn(total_timesteps=total_timesteps, callback=callback)
  model.save("./all_models/ModelV2--PPO_V1.0_test_old_env")

