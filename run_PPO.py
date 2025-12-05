import os
import sys
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure

# Chemin vers boptestGym_copy
THIS_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(THIS_DIR, "boptestGym"))
from boptestGymEnv import BoptestGymEnv, NormalizedActionWrapper, NormalizedObservationWrapper
from gymnasium import spaces


if __name__ == "__main__":
    url = "http://127.0.0.1:3000"
    log_dir = os.path.join(THIS_DIR, "agents", "PPO_vectorized_simple")
    os.makedirs(log_dir, exist_ok=True)

    DAY = 24 * 3600

    excluding_periods = [
        (59 * DAY, 243 * DAY),  # exclut printemps + été
        (14*DAY, 28*DAY)  #Periode de validation
        ]

    env = BoptestGymEnv(
        url=url,
        testcase="bestest_hydronic_heat_pump",
        actions=["oveTSet_u" ],
        observations={
            "time": (0, 604800),
            "reaTZon_y": (280.0, 310.0),
            "TDryBul": (265, 303),
            "HDirNor": (0, 862),
            "InternalGainsRad[1]": (0, 219),
            "PriceElectricPowerHighlyDynamic": (-0.4, 0.4),
            "LowerSetp[1]": (280.0, 310.0),
            "UpperSetp[1]": (280.0, 310.0),
        },
        predictive_period=24 * 3600,
        regressive_period=6 * 3600,
        random_start_time=True,
        excluding_periods=excluding_periods,
        max_episode_length=24 * 3600 * 7,
        warmup_period=24 * 3600 * 0.5,
        step_period=3600,
        render_episodes=True,
    )
    print("Action space:", env.action_space.low, env.action_space.high)

    #env = ClippedActionWrapper(env, low=[273.15 + 15], high=[273.15 + 30])

    low  = np.array([273.15 + 15], dtype=np.float32)
    high = np.array([273.15 + 30], dtype=np.float32)

    env.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
    print("Action space:", env.action_space.low, env.action_space.high)
    env = NormalizedObservationWrapper(env)
    env = NormalizedActionWrapper(env)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        gamma=0.99,
        seed=123456,
        device="cpu",
        tensorboard_log=log_dir,
        n_steps=48,
        #learning_rate=7e-4,
        #clip_range=0.3,
        #ent_coef=0.02,
        #n_epochs=20,
        #gae_lambda=0.95,
        #vf_coef=0.5,
        #max_grad_norm=0.5,
    )



    model.set_logger(configure(log_dir, ["stdout", "csv", "tensorboard"]))
    model.learn(total_timesteps=300_0)
    model.save(os.path.join(log_dir, "PPO_parallel"))

    env.close()
