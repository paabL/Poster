import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from SIMAX.Simulation import SimulationDataset
from utils import RC5_steady_state_sys


# Colonnes utilisées, cohérentes avec gymRC5.py
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


def main():
    # Charger la simulation pour récupérer theta
    sim_path = Path("Models/sim_opti.pkl")
    with sim_path.open("rb") as f:
        sim_opti_loaded = pickle.load(f)
    theta = sim_opti_loaded.model.theta

    # Charger le dataset complet (comme dans gymRC5.py)
    dataset = SimulationDataset.from_csv(
        "datas/train_df.csv",
        control_cols=CONTROL_COLS,
        disturbance_cols=DISTURBANCE_COLS,
    )

    time = np.asarray(dataset.time, dtype=float)
    n = time.shape[0]

    # Pour ne pas exploser le temps de calcul, on sous‑échantillonne
    step = max(1, n // 5000)  # ≈ 5000 points max
    idxs = np.arange(0, n, step, dtype=int)

    tz_init = []
    for i in idxs:
        ta = float(dataset.d["weaSta_reaWeaTDryBul_y"][i])
        qocc = float(dataset.d["InternalGainsCon[1]"][i])
        qocr = float(dataset.d["InternalGainsRad[1]"][i])
        qcd = float(dataset.d["reaQHeaPumCon_y"][i])
        qsol = float(dataset.d["weaSta_reaWeaHGloHor_y"][i])
        x = RC5_steady_state_sys(ta, qsol, qocc, qocr, qcd, theta)
        tz_init.append(float(x[0]))

    tz_init = np.asarray(tz_init, dtype=float)
    t_days = time[idxs] / 86400.0

    plt.figure(dpi=150)
    plt.plot(t_days, tz_init - 273.15, "-", linewidth=1)
    plt.xlabel("Temps (jours)")
    plt.ylabel("Tz initiale (°C)")
    plt.title("Première température de x via RC5_steady_state_sys")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

