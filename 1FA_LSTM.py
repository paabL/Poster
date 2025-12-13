from gymRC5 import MyMinimalEnv, ResidualActionWrapper, NormalizeAction
from rc5_multi_theta import build_thetas, RandomThetaWrapper
import torch


if __name__ == "__main__":
    from sb3_contrib import RecurrentPPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from stable_baselines3.common.monitor import Monitor

    reload = False
    test = True  # Si True : petite simu de test (plot)

    # Paramètres
    N_BUILDINGS = 10
    N_ENVS = 4
    TOTAL_TIMESTEPS = 10_000_000
    MODEL_PATH = "Pre_ppo_rc5_1FA_LSTM_2000000_steps.zip"
    VECNORM_PATH = "vecnormalize_stats_1FA_LSTM.pkl"

    # 1) Construire 10 modèles RC5 légèrement différents
    thetas = build_thetas(n_buildings=N_BUILDINGS, noise_level=0.1, seed=0)

    # 2) Créer un env MyMinimalEnv avec randomisation de theta à chaque épisode
    def make_env():
        def _init():
            base_sp = 273.15 + 22.0
            env = MyMinimalEnv(
                step_period=3600,
                past_steps=1,
                future_steps=12,
                warmup_steps=2 * 24,
                base_setpoint=base_sp,
                render_episodes=False,
                max_episode_length=24 * 7,
            )
            # Randomisation de theta à chaque reset (phase de généralisation)
            env = RandomThetaWrapper(env, thetas=thetas, seed=0, fixed_building_idx=None)
            # Politique résiduelle : delta autour d'une consigne de base
            env = ResidualActionWrapper(env, base_action=base_sp, max_dev=5.0)
            env = NormalizeAction(env)
            env = Monitor(env)
            return env

        return _init

    if test:
        import numpy as np
        import gymRC5

<<<<<<< ours
        def make_env_test():
            def _init():
                base_sp = 273.15 + 22.0
                env = MyMinimalEnv(
                    step_period=3600,
                    past_steps=1,
                    future_steps=12,
                    warmup_steps=2 * 24,
                    base_setpoint=base_sp,
                    render_episodes=True,   # plot auto
                    max_episode_length=24 * 7,
                    start_week=5,           # 5e semaine
                )
                env = RandomThetaWrapper(env, thetas=thetas, seed=0, fixed_building_idx=None)
                env = ResidualActionWrapper(env, base_action=base_sp, max_dev=5.0)
                env = NormalizeAction(env)
                env = Monitor(env)
                return env

            return _init

        venv = DummyVecEnv([make_env_test() for _ in range(N_BUILDINGS)])
=======
        # Un venv de test avec les N_BUILDINGS
        venv = DummyVecEnv([make_env() for _ in range(N_BUILDINGS)])
>>>>>>> theirs
        venv = VecNormalize.load(VECNORM_PATH, venv)
        venv.training = False
        venv.norm_reward = False

        model = RecurrentPPO.load(
            MODEL_PATH,
            env=venv,
            device="cpu",
        )

        obs = venv.reset()
<<<<<<< ours
=======

        # Forcer (approx) un départ sur la 5e semaine pour le premier env
        env0 = venv.envs[0]
        core_env = env0
        while hasattr(core_env, "env"):
            core_env = core_env.env
        week = np.asarray(gymRC5.dataset_short.d["week_idx"], dtype=float)
        idx_week = np.where(week == 4.0)[0]
        if idx_week.size:
            idx0 = int(idx_week[0])
            aligned = idx0 - ((idx0 - core_env.idx_min) % core_env.step_n)
            if core_env.idx_min <= aligned <= core_env.idx_max_start:
                core_env.idx = aligned

        lstm_state = None
>>>>>>> theirs
        dones = np.zeros((venv.num_envs,), dtype=bool)

        while not dones.all():
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = venv.step(action)

<<<<<<< ours
        # Plot simple sur le premier bâtiment
        env0 = venv.envs[0]
        core_env = env0
        while hasattr(core_env, "env"):
            core_env = core_env.env
=======
        # Plot simple via le helper déjà existant
>>>>>>> theirs
        core_env._plot_episode()
        venv.close()

    else:
        # 3) Enveloppe SB3 avec N_ENVS environnements en parallèle
        venv = DummyVecEnv([make_env() for _ in range(N_ENVS)])

        if reload:
            # Rechargement des stats de normalisation
            venv = VecNormalize.load(VECNORM_PATH, venv)
            venv.training = True
            venv.norm_reward = True

            # 4a) Recharger le modèle RecurrentPPO sauvegardé
            model = RecurrentPPO.load(
                MODEL_PATH,
                env=venv,
                device="cpu",
            )

            # 5a) Continuer l'entraînement sans réinitialiser le compteur de steps
            model.learn(
                total_timesteps=TOTAL_TIMESTEPS,
                tb_log_name="PPO_RC5_1FA_LSTM_continue",
                reset_num_timesteps=False,
            )
        else:
            # 4b) RecurrentPPO (LSTM) au lieu de PPO classique
            venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)
            model = RecurrentPPO(
                "MlpLstmPolicy",
                venv,
                verbose=1,
                learning_rate=2e-4,
                device="cpu",
                tensorboard_log="tensorboard_logs",
            )

            # Init de la tête d'action pour partir de delta ≈ 0
            with torch.no_grad():
                actor_net = model.policy.action_net
                actor_net.weight.fill_(0.0)
                actor_net.bias.fill_(0.0)
                if hasattr(model.policy, "log_std"):
                    model.policy.log_std.data.fill_(-2.0)

            # 5b) Entraînement : l'agent voit 10 configurations de bâtiment au fil des épisodes
            model.learn(total_timesteps=TOTAL_TIMESTEPS, tb_log_name="PPO_RC5_1FA_LSTM")

        # Sauvegarde modèle + stats de normalisation
        model.save(f"Pre_ppo_rc5_1FA_LSTM_{model._total_timesteps}_steps")
        venv.save(VECNORM_PATH)
        venv.close()
