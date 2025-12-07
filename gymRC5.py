import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
from SIMAX.Simulation import SimulationDataset
from SIMAX.Controller import Controller_PID
from utils import RC5_steady_state_sys
import jax.numpy as jnp
import equinox as eqx

# Colonnes utilisées dans le dataset
CONTROL_COLS = ()
DISTURBANCE_COLS = (
    "InternalGainsCon[1]",
    "InternalGainsRad[1]",
    "weaSta_reaWeaHGloHor_y",
    "weaSta_reaWeaTDryBul_y",
    "reaQHeaPumCon_y",
    "LowerSetp[1]",
    "UpperSetp[1]",
)

# ------------------------------------------------------------------
#  Charger la simulation
# ------------------------------------------------------------------
sim_path = Path("Models/sim_opti.pkl")
with sim_path.open("rb") as f:
    sim_opti_loaded = pickle.load(f)

# ------------------------------------------------------------------
#  Charger le dataset et en prendre au max N points
# ------------------------------------------------------------------
dataset = SimulationDataset.from_csv(
    "datas/train_df.csv",
    control_cols=CONTROL_COLS,
    disturbance_cols=DISTURBANCE_COLS,
)

# Ajout d'un signal de prix d'électricité (€/kWh) en créneau :
# 0.4 entre 18h et 22h, 0.2 le reste du temps.
time_seconds = np.asarray(dataset.time, dtype=float)
hours = (time_seconds / 3600.0) % 24.0
electricity_price = np.where(
    (hours >= 18.0) & (hours < 22.0),
    0.4,
    0.2,
).astype(np.float64)

dataset = SimulationDataset(
    time=dataset.time,
    u=dataset.u,
    d={**dataset.d, "electricity_price": jnp.asarray(electricity_price, dtype=jnp.float64)},
)



N = 280_000 #280_000 datas max
n_total = dataset.time.shape[0]
gamma = min(1.0, N / n_total)      # fraction dans ]0, 1]
dataset_short = dataset.take_fraction(gamma)

# ------------------------------------------------------------------
#  Exemple : état "équilibre" autour du temps 16 jours
# ------------------------------------------------------------------
target_time = 16 * 24 * 3600  # secondes
t_np = np.asarray(dataset_short.time)
idx_eq = int(np.argmin(np.abs(t_np - target_time)))  # index le + proche

ta0   = float(dataset_short.d["weaSta_reaWeaTDryBul_y"][idx_eq])
qocc0 = float(dataset_short.d["InternalGainsCon[1]"][idx_eq])
qocr0 = float(dataset_short.d["InternalGainsRad[1]"][idx_eq])
qcd0  = float(dataset_short.d["reaQHeaPumCon_y"][idx_eq])
qsol0 = float(dataset_short.d["weaSta_reaWeaHGloHor_y"][idx_eq])
theta = sim_opti_loaded.model.theta

x0 = RC5_steady_state_sys(ta0, qsol0, qocc0, qocr0, qcd0, theta)

sim = sim_opti_loaded.copy(
    x0=x0,
    time_grid=dataset_short.time[idx_eq:],
    d=dataset_short.d,
)


class MyMinimalEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        *,
        state_dim=5,          # dimension de l'état de ta simu SIMAX
        step_period=900,         # durée d'un pas en secondes
        predictive_period=None,  # horizon prédictif (dataset) en secondes
        regressive_period=None,  # horizon régressif (dataset) en secondes
        state_hist_steps=10,     # nb d'états passés à garder
        warmup_steps=50,         # nb de pas de warmup PID
        render_mode=None,
        tz_min=273.15 + 15.0,
        tz_max=273.15 + 30.0,
        render_episodes=False,   # plot auto en fin d'épisode
        max_episode_length=None, # nombre max de pas par épisode (None = limité par idx_max)
    ):
        super().__init__()

        # ---------- temps / horizons ----------
        self.step_period = float(step_period)
        # Pas de base du dataset (supposé régulier)
        dt_arr = np.diff(np.asarray(dataset_short.time, dtype=float))
        if dt_arr.size == 0:
            raise ValueError("dataset_short.time doit contenir au moins deux points.")
        base_dt = float(dt_arr[0])
        if not np.allclose(dt_arr, base_dt):
            raise ValueError("dataset_short.time doit être régulièrement échantillonné.")
        self.dataset_dt = base_dt
        self.step_n = max(1, int(round(self.step_period / self.dataset_dt)))  # nombre d'intervalles dataset par pas RL
        self.predictive_period = predictive_period
        self.regressive_period = regressive_period

        self.predictive_horizon = self._to_steps(predictive_period)
        self.regressive_horizon = self._to_steps(regressive_period)
        self.state_dim = int(state_dim)
        self.state_hist_steps = int(state_hist_steps)
        self.warmup_steps = int(warmup_steps)
        self.tz_min = float(tz_min)
        self.tz_max = float(tz_max)
        self.render_episodes = bool(render_episodes)
        self.max_episode_length = None if max_episode_length is None else int(max_episode_length)

        # Warmup doit produire assez d'états / commandes pour l'historique
        if self.warmup_steps < self.state_hist_steps + 1:
            raise ValueError("warmup_steps doit être >= state_hist_steps+1.")

        # Longueur du dataset de perturbations (météo, gains internes, etc.)
        self.n = dataset_short.time.shape[0]

        # Cache numpy pour éviter les __getitem__ JAX coûteux dans l'observation
        self._dist_matrix = np.stack(
            [
                np.asarray(dataset_short.d["weaSta_reaWeaTDryBul_y"], dtype=np.float32),
                np.asarray(dataset_short.d["weaSta_reaWeaHGloHor_y"], dtype=np.float32),
                np.asarray(dataset_short.d["InternalGainsCon[1]"], dtype=np.float32),
                np.asarray(dataset_short.d["InternalGainsRad[1]"], dtype=np.float32),
                np.asarray(dataset_short.d["reaQHeaPumCon_y"], dtype=np.float32),
                np.asarray(dataset_short.d["LowerSetp[1]"], dtype=np.float32),
                np.asarray(dataset_short.d["UpperSetp[1]"], dtype=np.float32),
            ],
            axis=1,
        )
        self._time_np = np.asarray(dataset_short.time, dtype=np.float64)

        # Contraintes sur l'indice de début d'épisode :
        # - avoir warmup_steps pas de marge avant
        # - avoir regressive_horizon pas dans le passé
        # - avoir predictive_horizon pas dans le futur
        self.warmup_steps_dataset = self.warmup_steps * self.step_n
        self.idx_min = max(self.regressive_horizon, self.warmup_steps_dataset)
        # idx_max garantit : fenêtre future dispo + un pas complet simulable
        self.idx_max = self.n - 1 - self.predictive_horizon - self.step_n
        if self.idx_max <= self.idx_min:
            raise ValueError(
                f"Horizons/warmup trop grands pour la taille du dataset : "
                f"n={self.n}, regressive={self.regressive_horizon}, "
                f"predictive={self.predictive_horizon}, warmup_steps={self.warmup_steps_dataset}"
            )

        # ---------- dimension de l'observation ----------
        # Fenêtre sur les perturbations du dataset : Ta, qsol, qocc, qocr, qcd, lower_sp, upper_sp
        self.n_features = 7
        self.n_past = self.regressive_horizon + 1     # t-h, ..., t
        self.n_fut = self.predictive_horizon          # t+1, ..., t+H
        self.n_hist_total = self.n_past + self.n_fut
        dist_dim = self.n_features * self.n_hist_total

        # Historique des états : on ne garde que Tz (x[0])
        tz_hist_dim = (self.state_hist_steps + 1) * 1

        # Historique des setpoints (même longueur que l'historique de Tz)
        sp_hist_dim = (self.state_hist_steps + 1) * 1

        obs_dim = dist_dim + tz_hist_dim + sp_hist_dim

        # Action = setpoint Tz [K]
        self.action_space = spaces.Box(low=self.tz_min, high=self.tz_max, shape=(1,), dtype=np.float32)

        # Observation = [fenêtre perturbations][hist états][hist commandes]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.render_mode = render_mode

        self.t_set = 273.15 + 21.0

        # Buffers d'historique (remplis au reset via warmup)
        self.state_hist = np.zeros(
            (self.state_hist_steps + 1, self.state_dim), dtype=np.float32
        )

        # Historique des setpoints
        self.sp_hist = np.full((self.state_hist_steps + 1,), self.t_set, dtype=np.float32)

        # État / commande courants de la simu
        self.x = np.zeros(self.state_dim, dtype=np.float32)
        self.current_setpoint = self.t_set

        # Index temporel courant dans le dataset
        self.idx = self.idx_min
        self.ep_steps = 0

        # Logs épisode (remplis durant step, pas RL)
        self.ep_idx = []
        self.ep_time = []
        self.ep_tz = []
        self.ep_setpoint = []
        self.ep_control = []
        self.ep_rewards = []
        self.ep_indiv_rewards = []
        # Logs fins pour le plot (≈30 s)
        self.ep_idx_30s = []
        self.ep_tz_30s = []
        self.ep_u_30s = []
        self._episode_plotted = False
        self._episode_count = 0

        # Cache de PID pour éviter de recréer des contrôleurs (même structure = pas de recompilation)
        self._pid_cache: dict[int, Controller_PID] = {}

    # ---------------------- helpers internes ---------------------- #
    def _to_steps(self, period):
        if period is None:
            return 0
        steps = int(round(float(period) / self.dataset_dt))
        return max(0, steps)

    def _get_features_at(self, idx: int) -> np.ndarray:
        """Retourne [Ta, qsol, qocc, qocr, qcd, lower_sp, upper_sp] au pas idx."""
        return self._dist_matrix[idx]

    def _get_band_at(self, idx: int) -> tuple[float, float]:
        """Limites de confort [bas, haut] en Kelvin au pas idx."""
        row = self._dist_matrix[idx]
        return float(row[5]), float(row[6])

    def _build_disturbance_window(self, idx: int) -> np.ndarray:
        """Fenêtre [passé, présent, futur] sur les perturbations du dataset."""
        start = idx - self.regressive_horizon
        end = idx + 1 + self.predictive_horizon
        window = self._dist_matrix[start:end]
        return window.reshape(-1)

    def _build_observation(self) -> np.ndarray:
        """Concatène : [fenêtre perturbations][hist Tz][hist setpoints]."""
        dist = self._build_disturbance_window(self.idx)
        tz_hist = self.state_hist[:, 0:1]
        x_hist_flat = tz_hist.reshape(-1)
        sp_hist_flat = self.sp_hist.reshape(-1)
        obs = np.concatenate([dist, x_hist_flat, sp_hist_flat], axis=0)
        return obs.astype(np.float32)

    def _sample_initial_index(self, rng: np.random.Generator) -> int:
        """Choisit un idx de départ compatible avec warmup + horizons."""
        raw = int(rng.integers(self.idx_min, self.idx_max + 1))
        # Aligner sur les multiples de step_n pour des pas complets
        aligned = raw - ((raw - self.idx_min) % self.step_n)
        if aligned < self.idx_min:
            aligned += self.step_n
        return min(aligned, self.idx_max)

    # ---------- Hooks à remplir avec SIMAX / PID ---------- #
    def _init_state(self, idx: int) -> np.ndarray:
        """Construit un état initial pour la simu au temps `idx`."""
        feats = self._get_features_at(idx)
        ta, qsol, qocc, qocr, qcd = map(float, feats[:5])
        x_init = RC5_steady_state_sys(ta, qsol, qocc, qocr, qcd, sim.model.theta)
        return np.asarray(x_init, dtype=np.float32)

    def _make_pid(self, setpoint: float, horizon_len: int) -> Controller_PID:
        sp = jnp.full((horizon_len,), float(setpoint), dtype=jnp.float64)
        cached = self._pid_cache.get(horizon_len)
        if cached is None:
            pid = Controller_PID(k_p=0.6, k_i=0.6 / 800.0, k_d=0.0, n=1, verbose=False, SetPoints=sp)
            self._pid_cache[horizon_len] = pid
            return pid
        # On réutilise la structure existante et on ne change que la consigne
        pid = eqx.tree_at(lambda c: c.SetPoints, cached, sp)
        self._pid_cache[horizon_len] = pid
        return pid

    def _reset_episode_logs(self):
        self.ep_idx.clear()
        self.ep_time.clear()
        self.ep_tz.clear()
        self.ep_setpoint.clear()
        self.ep_control.clear()
        self.ep_rewards.clear()
        self.ep_indiv_rewards.clear()
        self.ep_idx_30s.clear()
        self.ep_tz_30s.clear()
        self.ep_u_30s.clear()

    def _log_step(self, idx: int, tz: float, setpoint: float, u: float, reward: float, indiv_reward):
        self.ep_idx.append(idx)
        self.ep_time.append(float(self._time_np[idx]))
        self.ep_tz.append(float(tz))
        self.ep_setpoint.append(float(setpoint))
        self.ep_control.append(float(u))
        self.ep_rewards.append(float(reward))
        # indiv_reward = (comfort_penalty, energy_penalty, sat_penalty)
        self.ep_indiv_rewards.append(
            [
                float(indiv_reward[0]),
                float(indiv_reward[1]),
                float(indiv_reward[2]),
            ]
        )

    def _plot_episode(self):
        if not self.ep_idx:
            return

        plt.ion()
        fig = plt.gcf()
        fig.clear()
        fig.set_dpi(150)
        axs = fig.subplots(6, 1, sharex=True)
        fig.set_size_inches(10, 10, forward=True)
        # Grille fine (≈30 s) pour Tz/u si dispo, sinon pas RL
        if self.ep_idx_30s:
            idx_main = np.asarray(self.ep_idx_30s, dtype=int)
            t_days_main = self._time_np[idx_main] / 86400.0
            tz_c = np.asarray(self.ep_tz_30s, dtype=float) - 273.15
            u_arr = np.asarray(self.ep_u_30s, dtype=float)
        else:
            idx_main = np.asarray(self.ep_idx, dtype=int)
            t_days_main = np.asarray(self.ep_time, dtype=float) / 86400.0
            tz_c = np.asarray(self.ep_tz, dtype=float) - 273.15
            u_arr = np.asarray(self.ep_control, dtype=float)

        lower_c = np.asarray([self._dist_matrix[i, 5] for i in idx_main], dtype=float) - 273.15
        upper_c = np.asarray([self._dist_matrix[i, 6] for i in idx_main], dtype=float) - 273.15
        ta = np.asarray([dataset_short.d["weaSta_reaWeaTDryBul_y"][i] for i in idx_main], dtype=float) - 273.15
        qsol = np.asarray([dataset_short.d["weaSta_reaWeaHGloHor_y"][i] for i in idx_main], dtype=float)
        qocc = np.asarray([dataset_short.d["InternalGainsCon[1]"][i] for i in idx_main], dtype=float)
        qocr = np.asarray([dataset_short.d["InternalGainsRad[1]"][i] for i in idx_main], dtype=float)

        # Setpoint / rewards restent au pas RL
        t_days_rl = np.asarray(self.ep_time, dtype=float) / 86400.0
        sp_rl = np.asarray(self.ep_setpoint, dtype=float) - 273.15
        sp_c = np.interp(t_days_main, t_days_rl, sp_rl)
        rewards = np.asarray(self.ep_rewards, dtype=float)
        indiv = np.asarray(self.ep_indiv_rewards, dtype=float) if self.ep_indiv_rewards else None

        # Légendes avec parts en pourcentage des sous-récompenses
        reward_label = "reward"
        comfort_label = "comfort"
        energy_label = "energy"
        sat_label = "saturation"
        if indiv is not None and indiv.shape[1] >= 3:
            total_reward = float(rewards.sum())
            total_reward_safe = total_reward if abs(total_reward) > 1e-6 else 1.0
            total_comfort = float(indiv[:, 0].sum())
            total_energy = float(indiv[:, 1].sum())
            total_sat = float(indiv[:, 2].sum())
            comfort_pct = 100.0 * total_comfort / total_reward_safe
            energy_pct = 100.0 * total_energy / total_reward_safe
            sat_pct = 100.0 * total_sat / total_reward_safe
            reward_label = f"reward (sum={total_reward:.2f})"
            comfort_label = f"comfort ({comfort_pct:.0f}%)"
            energy_label = f"energy ({energy_pct:.0f}%)"
            sat_label = f"sat ({sat_pct:.0f}%)"

        axs[0].fill_between(t_days_main, lower_c, upper_c, color="palegreen", alpha=0.3, label="Bande confort")
        axs[0].plot(t_days_main, sp_c, "--", color="gray", linewidth=1, label="Setpoint")
        axs[0].plot(t_days_main, tz_c, "-", color="darkorange", linewidth=1, label="Tz")
        axs[0].set_ylabel("Tz / setpoint\n(°C)")
        axs[0].legend(fontsize=7)

        axs[1].plot(t_days_main, u_arr, "-", color="slateblue", linewidth=1)
        axs[1].set_ylabel("Commande\n(-)")

        axs[2].plot(t_days_rl, rewards, "b", linewidth=1, label=reward_label)
        if indiv is not None:
            axs[2].plot(t_days_rl, indiv[:, 0], "r", linewidth=1, label=comfort_label)
            axs[2].plot(t_days_rl, indiv[:, 1], "g", linewidth=1, label=energy_label)
            if indiv.shape[1] >= 3:
                axs[2].plot(t_days_rl, indiv[:, 2], "m", linewidth=1, label=sat_label)
        axs[2].set_ylabel("Rewards")
        axs[2].legend(loc="lower left", fontsize=7)

        axs[3].plot(t_days_main, ta, color="royalblue", linewidth=1, label="Ta")
        axq3 = axs[3].twinx()
        axq3.plot(t_days_main, qsol, color="gold", linewidth=1, label="Qsol")
        axs[3].set_ylabel("Ta (°C)")
        axq3.set_ylabel("Qsol (W)")
        axs[3].legend(loc="upper left", fontsize=7)
        axq3.legend(loc="upper right", fontsize=7)

        # Nouveau subplot pour les gains internes
        axs[4].plot(t_days_main, qocc, color="firebrick", linewidth=1, label="Qocc")
        axs[4].plot(t_days_main, qocr, color="darkred", linewidth=1, label="Qocr")
        axs[4].set_ylabel("Internal gains (W)")
        axs[4].legend(loc="upper right", fontsize=7)

        # Subplot minimaliste pour le prix de l'électricité
        price = np.asarray([dataset_short.d["electricity_price"][i] for i in idx_main], dtype=float)
        axs[5].plot(t_days_main, price, color="black", linewidth=1)
        axs[5].set_ylabel("Prix\n(€/kWh)")
        axs[5].set_xlabel("Temps (jours)")

        # Titre minimaliste avec un indice de l'épisode
        fig.suptitle(f"Episode: steps={len(self.ep_idx)}, last_idx={self.ep_idx[-1]}")

        plt.tight_layout()

        # Sauvegarde du rollout au lieu d'afficher
        rollout_dir = Path("rollout")
        rollout_dir.mkdir(parents=True, exist_ok=True)
        filename = rollout_dir / f"episode_{self._episode_count:04d}_idx_{self.ep_idx[-1]}.png"
        fig.savefig(filename)
        plt.close(fig)

    def _run_warmup(self, start_idx: int, end_idx: int):
        """Warmup PID entre start_idx et end_idx (inclus),
        et remplissage des historiques d'états et de commandes.
        """
        x_init = self._init_state(start_idx)
        time_slice = dataset_short.time[start_idx : end_idx + 1]
        pid_warm = self._make_pid(self.t_set, len(time_slice))

        # Un run complet fournit états et commandes PID sur la fenêtre de warmup
        _, _, states, controls = sim.run(
            time_grid=time_slice,
            controller=pid_warm,
            x0=x_init,
        )

        state_traj = np.asarray(states, dtype=np.float32)
        state_traj_step = state_traj[::self.step_n]
        if state_traj_step.size == 0 or not np.allclose(state_traj_step[-1], state_traj[-1]):
            state_traj_step = np.vstack([state_traj_step, state_traj[-1:]])

        self.state_hist = state_traj_step[-(self.state_hist_steps + 1):]
        self.x = self.state_hist[-1]

    # -------------------------- API Gym --------------------------- #
    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        rng = np.random.default_rng(seed)

        # Choix d'un instant de début d'épisode
        self.idx = self._sample_initial_index(rng)

        # Warmup PID sur `warmup_steps` pas avant idx
        warmup_start = self.idx - self.warmup_steps_dataset
        self._run_warmup(warmup_start, self.idx)

        # reset état interne PID pour la boucle contrôle RL
        self.current_setpoint = self.t_set
        # Si on enchaîne les épisodes, on peut afficher celui qui vient de finir
        if self.render_episodes and self.ep_idx and not self._episode_plotted and (self._episode_count % 10 == 0):
            self._plot_episode()
        self._episode_plotted = False
        self._episode_count += 1
        self.sp_hist.fill(self.t_set)
        self._reset_episode_logs()
        self.ep_steps = 0

        obs = self._build_observation()
        return obs, {}

    def step(self, action):
        # 1) Action RL -> setpoint
        tz_set = float(np.clip(np.asarray(action).reshape(-1)[0], self.tz_min, self.tz_max))
        self.current_setpoint = tz_set

        # Mettre à jour l'historique des setpoints
        self.sp_hist[:-1] = self.sp_hist[1:]
        self.sp_hist[-1] = tz_set

        # 2) PID -> commande physique via SIMAX sur un pas complet (plusieurs sous-intervales dataset)
        time_slice = dataset_short.time[self.idx : self.idx + self.step_n + 1]
        pid_step = self._make_pid(tz_set, len(time_slice))
        _, _, states, controls = sim.run(
            time_grid=time_slice,
            controller=pid_step,
            x0=self.x,
        )
        states_arr = np.asarray(states, dtype=np.float32)
        u_hist = np.asarray(
            controls.get("oveHeaPumY_u", np.zeros((len(time_slice),), dtype=np.float64)),
            dtype=np.float32,
        )
        delta_sat_hist = np.asarray(
            controls.get("delta_sat", np.zeros((len(time_slice),), dtype=np.float64)),
            dtype=np.float32,
        )
        x_next = states_arr[-1]

        # 3) Logs fins (≈30 s) pour le plot uniquement
        tz_traj = states_arr[:, 0]
        n_inner = min(len(time_slice) - 1, len(u_hist))
        for k in range(1, n_inner + 1):
            idx_k = self.idx + k
            self.ep_idx_30s.append(idx_k)
            self.ep_tz_30s.append(float(tz_traj[k]))
            self.ep_u_30s.append(float(u_hist[k - 1]))

        # commande représentative pour le pas RL (dernière)
        u_rl = u_hist[-1:]
        delta_rl = delta_sat_hist[-1:]

        # 4) Mise à jour des historiques
        self.state_hist[:-1] = self.state_hist[1:]
        self.state_hist[-1] = x_next

        self.x = x_next

        # 4) Avancer dans le dataset pour les perturbations
        self.idx += self.step_n
        self.ep_steps = getattr(self, "ep_steps", 0) + 1

        terminated = False
        truncated = False

        if self.idx > self.idx_max:
            self.idx = self.idx_max
            terminated = True
            truncated = True

        if (self.max_episode_length is not None) and (self.ep_steps >= self.max_episode_length):
            terminated = True
            truncated = True

        obs = self._build_observation()
        lower_sp, upper_sp = self._get_band_at(self.idx)

        tz_curr = float(self.x[0])
        comfort_below = max(lower_sp - tz_curr, 0.0)
        comfort_above = max(tz_curr - upper_sp, 0.0)
        comfort_penalty = -(comfort_below + comfort_above)
        energy_penalty = -1.0 * float(np.clip(u_rl, 0.0, 1.0).mean())
        sat_penalty = -0.005 * float(np.abs(delta_rl).mean())
        reward = comfort_penalty + energy_penalty + sat_penalty

        self._log_step(self.idx, tz_curr, tz_set, float(u_rl[-1]), reward, (comfort_penalty, energy_penalty, sat_penalty))
        info = {
            "idx": self.idx,
            "predictive_horizon": self.predictive_horizon,
            "regressive_horizon": self.regressive_horizon,
            "Tz": tz_curr,
            "setpoint": tz_set,
            "ep_steps": self.ep_steps,
            "lower_band": lower_sp,
            "upper_band": upper_sp,
        }

        if self.render_episodes and (terminated or truncated):
            if not self._episode_plotted and (self._episode_count % 100 == 0):
                self._plot_episode()
                self._episode_plotted = True

        return obs, reward, terminated, truncated, info

    def render(self, mode: str = "episodes"):
        if mode != "episodes":
            raise NotImplementedError()
        if self.render_episodes:
            self._plot_episode()

    def close(self):
        pass



class NormalizeAction(gym.ActionWrapper):
    """
    L'agent voit des actions dans [-1, 1].
    On les rescales vers [low, high] pour l'env interne.
    """

    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.action_space, spaces.Box)
        self._low = env.action_space.low.astype(np.float32)
        self._high = env.action_space.high.astype(np.float32)

        # Vue "agent" : actions normalisées
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=env.action_space.shape,
            dtype=np.float32,
        )

    def action(self, action):
        # action in [-1, 1] -> [low, high]
        action = np.asarray(action, dtype=np.float32)
        return self._low + (action + 1.0) * 0.5 * (self._high - self._low)




if __name__ == "__main__":
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import VecNormalize

    def make_env():
        def _init():
            env = MyMinimalEnv(
                step_period=3600,
                predictive_period=24 * 3600,
                regressive_period=24 * 3600,
                state_hist_steps=5,
                warmup_steps=24,
                render_episodes=True,
                max_episode_length=24 * 14,
            )
            env = NormalizeAction(env)
            env = Monitor(env)
            return env
        return _init

    n_envs = 8  # nombre d'environnements parallèles

    # Vecteur d'envs multi-process
    venv = SubprocVecEnv([make_env() for _ in range(n_envs)])

    # Normalisation obs + rewards
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = PPO(
        "MlpPolicy",
        venv,
        verbose=1,
        device="cpu",
        learning_rate=3e-4,
        tensorboard_log="tensorboard_logs",
    )

    model.learn(total_timesteps=3_000_000, tb_log_name="PPO_RC5")

    model.save(f"ppo_rc5_model_{model._total_timesteps}_steps")
    venv.save("vecnormalize_stats.pkl")

    venv.close()



if __name__ == "__main__":
    # Petit smoke-test : plusieurs épisodes aléatoires,
    # avec affichage automatique à la fin de chaque épisode
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import VecNormalize
    

    def make_env():
        def _init():
            env = MyMinimalEnv(
                step_period=3600,
                predictive_period=24 * 3600,
                regressive_period=24 * 3600,
                state_hist_steps=5,
                warmup_steps=24,
                render_episodes=True,      # important pour que _plot_episode soit appelé
                max_episode_length=24 * 14, # Pas max par episode
            )
            env = NormalizeAction(env)
            env = Monitor(env)
            return env

        return _init

    n_envs = 8  # nombre d'environnements parallèles
    venv = DummyVecEnv([make_env() for _ in range(n_envs)])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # Logging minimaliste TensorBoard (SB3 gère les scalaires de base)
    model = PPO(
        "MlpPolicy",
        venv,
        verbose=1,
        device = 'cpu',
        tensorboard_log="tensorboard_logs",  # dossier où TensorBoard ira lire
    )
    model.learn(total_timesteps=30_000, tb_log_name="PPO_RC5")
    model.save(f"ppo_rc5_model_{model._total_timesteps}_steps")

    venv.close()
