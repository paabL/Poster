from gymRC5 import MyMinimalEnv, ResidualActionWrapper, NormalizeAction
from rc5_multi_theta import KModelWrapper, build_k_models
import torch


if __name__ == "__main__":
    from sb3_contrib import RecurrentPPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from stable_baselines3.common.monitor import Monitor

    reload = False

    # Paramètres
    N_ENVS = 4
    FIXED_MODEL_IDX = None  # ex: 0 pour forcer un modèle
    TOTAL_TIMESTEPS = 10_000_000
    MODEL_PATH = "Pre_ppo_rc5_1FA_LSTM_2000000_steps.zip"
    VECNORM_PATH = "vecnormalize_stats_1FA_LSTM.pkl"

    # Modèles (k) typiques, en clair (pas d'aléatoire)
    KS = [
        {"k_size": 1.0, "k_U": 1.0, "k_inf": 1.0, "k_win": 1.0, "k_mass": 1.0},
        {"k_size": 0.8, "k_U": 0.8, "k_inf": 0.8, "k_win": 1.0, "k_mass": 0.9},
        {"k_size": 1.2, "k_U": 1.0, "k_inf": 1.0, "k_win": 1.0, "k_mass": 1.1},
        {"k_size": 1.0, "k_U": 0.7, "k_inf": 1.0, "k_win": 1.0, "k_mass": 1.0},
        {"k_size": 1.0, "k_U": 1.3, "k_inf": 1.0, "k_win": 1.0, "k_mass": 1.0},
        {"k_size": 1.0, "k_U": 1.0, "k_inf": 0.7, "k_win": 1.0, "k_mass": 1.0},
        {"k_size": 1.0, "k_U": 1.0, "k_inf": 1.3, "k_win": 1.0, "k_mass": 1.0},
        {"k_size": 1.0, "k_U": 1.0, "k_inf": 1.0, "k_win": 1.2, "k_mass": 1.0},
        {"k_size": 1.0, "k_U": 1.0, "k_inf": 1.0, "k_win": 0.8, "k_mass": 1.0},
        {"k_size": 1.0, "k_U": 1.0, "k_inf": 1.0, "k_win": 1.0, "k_mass": 1.2},
    ]
    thetas = build_k_models(KS)

    def make_env(rank: int):
        def _init():
            base_sp = 273.15 + 22.0
            env = MyMinimalEnv(
                step_period=3600,
                past_steps=0,
                future_steps=12,
                warmup_steps=4 * 24,
                base_setpoint=base_sp,
                render_episodes=True,
                max_episode_length=24 * 7,
                excluding_periods=[(28 * 24 * 3600, 60 * 24 * 3600)],
            )
            env = KModelWrapper(env, thetas=thetas, ks=KS, seed=rank, fixed_model_idx=FIXED_MODEL_IDX)
            # Politique résiduelle : delta autour d'une consigne de base
            env = ResidualActionWrapper(env, base_action=base_sp, max_dev=5.0)
            env = NormalizeAction(env)
            env = Monitor(env)
            return env

        return _init

    # 3) Enveloppe SB3 avec N_ENVS environnements en parallèle
    venv = DummyVecEnv([make_env(i) for i in range(N_ENVS)])

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
    #model.save(f"Pre_ppo_rc5_1FA_LSTM_{model._total_timesteps}_steps")
    #venv.save(VECNORM_PATH)

    venv.close()
