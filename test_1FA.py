from __future__ import annotations

from pathlib import Path

import numpy as np

from gymRC5 import MyMinimalEnv, NormalizeAction, ResidualActionWrapper
from Utils.rc5_multi_theta import KModelWrapper, build_k_models


MODEL_PATHS = [
    Path("Pre_ppo_rc5_1FA.zip"),
]
VECNORM_PATH = Path("vecnormalize_stats_1FA.pkl")  # optionnel (si absent: pas de normalisation)

KS_PRESETS = [
    {"k_size": 1.0, "k_U": 1.0, "k_inf": 1.0, "k_win": 1.0, "k_mass": 1.0},
]
KS_IDX: int | None = 0  # None = random à chaque reset

EPISODE_START_TIME_S = 28 * 24 * 3600  # 1er février si t=0 = 1er janvier
N_EPISODES = 1
DETERMINISTIC = True
DEVICE = "cpu"
MAX_STEPS: int | None = 2 * 24  # None = épisode complet
KEEP_PLOTS_OPEN = True

# IMPORTANT: doit matcher l'entraînement/VecNormalize
STEP_PERIOD = 3600
PAST_STEPS = 2 * 24
FUTURE_STEPS = 2 * 24
WARMUP_STEPS = 2 * 24
MAX_EPISODE_LENGTH = 24 * 7
EXCLUDING_PERIODS = [(28 * 24 * 3600, 39 * 24 * 3600)]


def _unwrap_to(env, target_type):
    cur = env
    seen = set()
    while cur is not None and id(cur) not in seen:
        if isinstance(cur, target_type):
            return cur
        seen.add(id(cur))
        cur = getattr(cur, "env", None)
    return None


def _make_venv(*, fixed_model_idx: int | None, seed: int = 0):
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    base_sp = 273.15 + 22.0
    thetas = build_k_models(KS_PRESETS)

    def _make_one(rank: int = 0):
        env = MyMinimalEnv(
            step_period=STEP_PERIOD,
            past_steps=PAST_STEPS,
            future_steps=FUTURE_STEPS,
            warmup_steps=WARMUP_STEPS,
            base_setpoint=base_sp,
            render_episodes=False,
            max_episode_length=MAX_EPISODE_LENGTH,
            excluding_periods=EXCLUDING_PERIODS,
        )
        env = KModelWrapper(
            env,
            thetas=thetas,
            ks=KS_PRESETS,
            seed=seed + rank,
            fixed_model_idx=fixed_model_idx,
        )
        env = ResidualActionWrapper(env, base_action=base_sp, max_dev=5.0)
        env = NormalizeAction(env)
        env = Monitor(env)
        return env

    return DummyVecEnv([lambda: _make_one(0)])


def _maybe_load_vecnormalize(venv, vecnorm_path: Path):
    if not vecnorm_path.exists():
        print(f"[warn] VecNormalize introuvable: {vecnorm_path} (obs non normalisées)")
        return None, venv
    from stable_baselines3.common.vec_env import VecNormalize

    venv = VecNormalize.load(str(vecnorm_path), venv)
    venv.training = False
    venv.norm_reward = False
    return venv, venv


def _load_model(model_path: Path, *, venv, device: str):
    from stable_baselines3 import PPO

    return PPO.load(str(model_path), env=venv, device=device)


def _rollout_one_episode_single_env(
    *,
    model,
    env,
    vecnorm,
    deterministic: bool,
    max_steps: int | None,
    start_time_s: float,
):
    obs, _info = env.reset(seed=0, options={"start_time_s": float(start_time_s)})
    ep_return = 0.0
    ep_len = 0

    while True:
        if max_steps is not None and ep_len >= max_steps:
            return {"return": ep_return, "len": ep_len, "monitor": {}}

        obs_in = vecnorm.normalize_obs(obs) if vecnorm is not None else obs
        action, _ = model.predict(obs_in, deterministic=deterministic)

        obs, reward, terminated, truncated, info = env.step(action)
        ep_return += float(reward)
        ep_len += 1
        if bool(terminated or truncated):
            ep_info = dict(info.get("episode", {})) if isinstance(info, dict) else {}
            return {"return": ep_return, "len": ep_len, "monitor": ep_info}


if __name__ == "__main__":
    model_paths = [p for p in MODEL_PATHS if p.exists()]
    if not model_paths:
        raise SystemExit("Aucun modèle trouvé (edite `MODEL_PATHS`).")

    for model_path in model_paths:
        print(f"\n=== {model_path} ===")
        venv = _make_venv(fixed_model_idx=KS_IDX, seed=0)
        vecnorm, venv_for_model = _maybe_load_vecnormalize(venv, VECNORM_PATH)
        model = _load_model(model_path, venv=venv_for_model, device=DEVICE)

        env = venv_for_model.venv.envs[0] if vecnorm is not None else venv.envs[0]

        for ep in range(N_EPISODES):
            out = _rollout_one_episode_single_env(
                model=model,
                env=env,
                vecnorm=vecnorm,
                deterministic=DETERMINISTIC,
                max_steps=MAX_STEPS,
                start_time_s=EPISODE_START_TIME_S,
            )
            mon = out.get("monitor", {}) or {}
            r = mon.get("r", out["return"])
            l = mon.get("l", out["len"])
            print(f"ep={ep+1}/{N_EPISODES} return={r:.3f} len={int(l)}")

            base = _unwrap_to(env, MyMinimalEnv)
            if base is None:
                raise RuntimeError("Impossible de retrouver MyMinimalEnv (wrappers inattendus).")
            base._plot_episode()

        env.close()
        venv_for_model.close() if vecnorm is not None else venv.close()

    if KEEP_PLOTS_OPEN:
        import matplotlib.pyplot as plt

        plt.ioff()
        plt.show(block=True)
