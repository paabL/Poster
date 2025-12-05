import pickle
from dataclasses import replace
from pathlib import Path

import jax64  # noqa: F401
import jax.numpy as jnp
import pandas as pd

from SIMAX.Controller import Controller_MPC
from SIMAX.Simulation import SimulationDataset
from utils import RC5_steady_state_sys

CONTROL_COLS = ("oveHeaPumY_u",)
DISTURBANCE_COLS = (
    "InternalGainsCon[1]",
    "InternalGainsRad[1]",
    "weaSta_reaWeaHGloHor_y",
    "weaSta_reaWeaTDryBul_y",
    "reaQHeaPumCon_y",
    "reaQHeaPumEva_y",
)


sim_path = Path("Models/sim_opti.pkl")

with sim_path.open("rb") as src:
    sim_opti_loaded = pickle.load(src)

#Dataset pour les disturbances, consignes et temps
df = pd.read_csv("datas/train_df.csv", skipinitialspace=True)
df.columns = df.columns.str.strip()  # normalize names
N = 60_000

dataset = SimulationDataset.from_csv("datas/train_df.csv", control_cols=CONTROL_COLS, disturbance_cols=DISTURBANCE_COLS)
dataset_short = dataset.take_fraction(N / len(dataset.time))




#On choisit un état initial à l'équilibre pour la simulation MPC
ta0 = dataset_short.d["weaSta_reaWeaTDryBul_y"][0]
qocc0 = dataset_short.d["InternalGainsCon[1]"][0]
qocr0 = dataset_short.d["InternalGainsRad[1]"][0]
qcd0 = dataset_short.d["reaQHeaPumCon_y"][0]
qsol0 = dataset_short.d["weaSta_reaWeaHGloHor_y"][0] 
theta = sim_opti_loaded.model.theta


x0 = RC5_steady_state_sys(ta0, qsol0, qocc0, qocr0, qcd0, theta)
print(x0)




#Simu jumelle numérique avec état initial à l'équilibre, time_grid et d synchronisés
sim = replace(sim_opti_loaded, x0=x0, time_grid=dataset_short.time, d=dataset_short.d)
sim_for_mpc = sim.copy()


# MPC : consignes et contrôleur
SetPoints = jnp.array(df["oveTSet_u"].values[:N], dtype=jnp.float64)
p = 0.2  # €/kWh
c = 5.0  # €/K/h
nZOH = 100
controllerMPC = Controller_MPC(sim=sim_for_mpc, window_size=5_000, W=jnp.array([p/(3600*1000), c/(3600)]), n=nZOH, SetPoints=SetPoints) 
# Simulation avec MPC (plante contrôlée)
sim_controlled = replace(sim_for_mpc, controller=controllerMPC)


t, y, x, u, ctrl_states, mpc_logs = sim_controlled.run_numpy()




x_i_traj_MPC = [cs["x_i"] for cs in ctrl_states]
x_i_traj = x

Tz_MPC = jnp.array([x_i[0] for x_i in x_i_traj_MPC])
Tz_sim = jnp.array([x_i[0] for x_i in x_i_traj])
#print(x_i_traj)


import matplotlib.pyplot as plt

n_dist = len(DISTURBANCE_COLS)
fig, axes = plt.subplots(2 + n_dist, 1, figsize=(12, 20), dpi=400, sharex=True)

if (2 + n_dist) == 1:
    axes = [axes]

# --- Subplot 1 ---

for pm in ctrl_states[::nZOH]:
    y_plan_MPC = pm["latest_forecast"]["y_plan_window"][:, 0]
    t_plan_MPC = pm["latest_forecast"]["time"]/(24*3600)
    i = pm["latest_forecast"]["decision_idx"]
    axes[0].plot(t_plan_MPC, y_plan_MPC, linestyle="--", color="gray", alpha=0.2)

axes[0].plot(t/(24*3600), Tz_sim, label="Tz sim")
axes[0].plot(t/(24*3600), Tz_MPC, label="Tz MPC")
axes[0].plot(t/(24*3600), SetPoints, label="Setpoint")
axes[0].set_xlabel("Time [days]")
axes[0].set_ylabel("Zone Temperature [°C]")
axes[0].set_title("Zone Temperature under MPC Control")




axes[0].legend()
axes[0].grid(True)



u_sim = u["oveHeaPumY_u"]
u_mpc = [cs["u_prev"] for cs in ctrl_states]

# --- Subplot 2 ---
axes[1].plot(t/(24*3600), u_sim, label="U_sim")
axes[1].plot(t/(24*3600), u_mpc, label="U_mpc")
axes[1].set_xlabel("Time [days]")
axes[1].set_title("Commande HP(time in days)")
axes[1].legend()
axes[1].grid(True)

# --- Subplots 3 --- evolution du cout


# --- Subplots disturbances ---
for i, col in enumerate(DISTURBANCE_COLS):
    ax_d = axes[2 + i]
    ax_d.plot(t/(24*3600), dataset_short.d[col], label=col)
    ax_d.set_ylabel(col)
    ax_d.legend()
    ax_d.grid(True)

axes[-1].set_xlabel("Time [days]")

plt.tight_layout()
plt.savefig(f"MPC/figures/mpc_rc5_rc5_results_{N}.png")














