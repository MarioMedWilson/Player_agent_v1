# from env import Player
from old_env import Player
from stable_baselines3 import PPO, DQN
import time

if __name__ == '__main__':
    ALL_MODELS_DIR = './all_models/'
    env = Player()
    state, info = env.reset()
    print(state)
    done = False
    model = PPO('MultiInputPolicy', env, verbose=1, learning_rate=0.001)
    model.load("./all_models/ModelV2--PPO_V1.0_test_old_env")
    while not done:
        action, _ = model.predict(state)
        state, reward, done, clipped, info = env.step(action)
        env.render()
        print("reward", reward)
        time.sleep(0.5)
    print(state)
