from gymRC5 import MyMinimalEnv, ResidualActionWrapper, NormalizeAction


if __name__ == "__main__":
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from stable_baselines3.common.monitor import Monitor

    import torch  # gardé pour cohérence avec le script principal

    # Paramètres simples à modifier ici si besoin
    MODEL_PATH = "Pre_ppo_rc5_model_10000000_steps.zip"
    TOTAL_TIMESTEPS = 15_000_000
    N_ENVS = 8
    DEVICE = "cpu"

    def make_env():
        def _init():
            base_sp = 273.15 + 22.0
            env = MyMinimalEnv(
                step_period=3600,
                past_steps=2 * 24,
                future_steps=24,
                warmup_steps=3 * 24,
                base_setpoint=base_sp,
                render_episodes=True,
                max_episode_length=24 * 14,
            )
            env = ResidualActionWrapper(env, base_action=base_sp, max_dev=5.0)
            env = NormalizeAction(env)
            env = Monitor(env)
            return env

        return _init

    # 1) recréer le VecEnv de base
    venv = DummyVecEnv([make_env() for _ in range(N_ENVS)])

    # 2) recharger les stats de normalisation
    venv = VecNormalize.load("vecnormalize_stats.pkl", venv)
    venv.training = True
    venv.norm_reward = True

    # 3) recharger le modèle PPO sauvegardé
    model = PPO.load(
        MODEL_PATH,
        env=venv,
        device="cpu",
    )

    # 4) continuer l'entraînement (ne pas réinitialiser la tête d'action)
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        tb_log_name="PPO_RC5_continue",
        reset_num_timesteps=False,
    )

    # 5) resauvegarder modèle + stats
    model.save(f"Pre_ppo_rc5_model_{model._total_timesteps}_steps")
    venv.save("vecnormalize_stats.pkl")

    venv.close()
