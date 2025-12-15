import random

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from gymRC5 import MyMinimalEnv, NormalizeAction, ResidualActionWrapper
from Utils.rc5_multi_theta import KModelWrapper, build_k_models

CFG = dict(
    reload=False,
    n_envs=8,
    seed=0,
    fixed_model_idx=None,
    total_timesteps=20_000_000,
    model_path="Pre_ppo_rc5_1FA.zip",
    vecnorm_path="vecnormalize_stats_1FA.pkl",
)

ENV_CFG = dict(
    step_period=3600,
    past_steps=2 * 24,
    future_steps=2 * 24,
    warmup_steps=2 * 24,
    base_setpoint=273.15 + 22.0,
    w_u=0.0,
    w_tz=0.0,
    render_episodes=False,
    max_episode_length=24 * 7,
    excluding_periods=[(28 * 24 * 3600, 39 * 24 * 3600)],
)

VECNORM_CFG = dict(norm_obs=True, norm_reward=True, clip_obs=10.0)

PPO_CFG = dict(
    learning_rate=1e-4,
    device="cpu",
    verbose=1,
    tensorboard_log="tensorboard_logs",
)

KS = [
    {"k_size": 1.0, "k_U": 1.0, "k_inf": 1.0, "k_win": 1.0, "k_mass": 1.0},
    {"k_size": 0.9, "k_U": 0.9, "k_inf": 0.9, "k_win": 1.0, "k_mass": 0.95},
    {"k_size": 1.1, "k_U": 1.0, "k_inf": 1.0, "k_win": 1.0, "k_mass": 1.05},
    {"k_size": 1.0, "k_U": 0.85, "k_inf": 1.0, "k_win": 1.0, "k_mass": 1.0},
    {"k_size": 1.0, "k_U": 1.15, "k_inf": 1.0, "k_win": 1.0, "k_mass": 1.0},
    {"k_size": 1.0, "k_U": 1.0, "k_inf": 0.85, "k_win": 1.0, "k_mass": 1.0},
    {"k_size": 1.0, "k_U": 1.0, "k_inf": 1.15, "k_win": 1.0, "k_mass": 1.0},
    {"k_size": 1.0, "k_U": 1.0, "k_inf": 1.0, "k_win": 1.1, "k_mass": 1.0},
    {"k_size": 1.0, "k_U": 1.0, "k_inf": 1.0, "k_win": 0.9, "k_mass": 1.0},
    {"k_size": 1.0, "k_U": 1.0, "k_inf": 1.0, "k_win": 1.0, "k_mass": 1.1},
]


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_env(rank: int, thetas):
    def _init():
        env = MyMinimalEnv(**ENV_CFG)
        env = KModelWrapper(
            env,
            thetas=thetas,
            ks=KS,
            seed=CFG["seed"] + rank,
            fixed_model_idx=CFG["fixed_model_idx"],
        )
        env = ResidualActionWrapper(env, base_action=ENV_CFG["base_setpoint"], max_dev=5.0)
        env = NormalizeAction(env)
        env = Monitor(env)
        return env

    return _init


def build_model(env):
    return PPO(
        "MlpPolicy",
        env,
        verbose=PPO_CFG["verbose"],
        learning_rate=PPO_CFG["learning_rate"],
        device=PPO_CFG["device"],
        tensorboard_log=PPO_CFG["tensorboard_log"],
        seed=CFG["seed"],
    )


if __name__ == "__main__":
    set_global_seed(CFG["seed"])
    thetas = build_k_models(KS)

    venv = DummyVecEnv([make_env(i, thetas) for i in range(CFG["n_envs"])])

    if CFG["reload"]:
        venv = VecNormalize.load(CFG["vecnorm_path"], venv)
        venv.training = True
        venv.norm_obs = VECNORM_CFG["norm_obs"]
        venv.norm_reward = VECNORM_CFG["norm_reward"]
        model = PPO.load(CFG["model_path"], env=venv, device=PPO_CFG["device"])

        model.learn(
            total_timesteps=CFG["total_timesteps"],
            tb_log_name="PPO_RC5_1FA_continue",
            reset_num_timesteps=False,
        )
        model.save(CFG["model_path"])
        venv.save(CFG["vecnorm_path"])
    else:
        venv = VecNormalize(venv, **VECNORM_CFG)
        model = build_model(venv)

        with torch.no_grad():
            actor_net = model.policy.action_net
            actor_net.weight.fill_(0.0)
            actor_net.bias.fill_(0.0)
            if hasattr(model.policy, "log_std"):
                model.policy.log_std.data.fill_(-2.0)

        model.learn(total_timesteps=CFG["total_timesteps"], tb_log_name="PPO_RC5_1FA")
        model.save(CFG["model_path"])
        venv.save(CFG["vecnorm_path"])

    venv.close()
