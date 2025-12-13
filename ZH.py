"""
Tuning PI (Ziegler–Nichols) sur la simulation thermique RC5 utilisée dans gymRC5 (`Models/sim_opti.pkl`), réglé sur Tz.
Simple : on augmente Kp jusqu'à une oscillation entretenue, on calcule Ku, Pu puis Kp/Ki PI.
"""

from __future__ import annotations

from pathlib import Path
import pickle
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

from SIMAX.Controller import Controller_PID
from SIMAX.Simulation import Simulation_JAX
from utils import RC5_steady_state_sys

SIM_PATH = Path("Models/sim_opti.pkl")
SETPOINT = 273.15 + 22.0
HORIZON_S = 5 * 24 * 3600  # horizon max (s) pour détecter l'oscillation
# Warmup : consigne basse pendant 24 h avant le step pour observer les oscillations
WARMUP_DURATION_S = 24 * 3600
WARMUP_SP = SETPOINT - 2.0


def _load_sim() -> Simulation_JAX:
    with SIM_PATH.open("rb") as f:
        return pickle.load(f)


def _trim_sim(sim: Simulation_JAX, n_steps: int, ctrl) -> Simulation_JAX:
    time_grid = sim.time_grid[:n_steps]
    d_trim = {k: v[:n_steps] for k, v in sim.d.items()}

    ta0   = float(d_trim["weaSta_reaWeaTDryBul_y"][0])
    qocc0 = float(d_trim["InternalGainsCon[1]"][0])
    qocr0 = float(d_trim["InternalGainsRad[1]"][0])
    qcd0  = float(d_trim["reaQHeaPumCon_y"][0])
    qsol0 = float(d_trim["weaSta_reaWeaHGloHor_y"][0])
    theta = sim.model.theta
    x0 = RC5_steady_state_sys(ta0, qsol0, qocc0, qocr0, qcd0, theta)
    sim = sim.copy(time_grid=time_grid, d=d_trim, controller=ctrl, x0=x0)
    #sim.plot()
    return sim


def run_closed_loop(sim_base: Simulation_JAX, kp: float, ki: float) -> tuple[np.ndarray, np.ndarray]:
    dt = float(sim_base.time_grid[1] - sim_base.time_grid[0])
    n_steps = min(int(HORIZON_S // dt), int(sim_base.time_grid.shape[0]))
    warmup_n = min(int(WARMUP_DURATION_S // dt), max(n_steps - 1, 1))
    setpoints = jnp.full((n_steps,), float(SETPOINT), dtype=jnp.float64)
    setpoints = setpoints.at[:warmup_n].set(float(WARMUP_SP))
    ctrl = Controller_PID(
        k_p=kp,
        k_i=ki,
        k_d=0.0,
        n=1,
        verbose=False,
        SetPoints=setpoints,
        TSet=float(SETPOINT),
    )
    sim = _trim_sim(sim_base, n_steps, ctrl)
    t, y_seq, _, _ = sim.run()
    return np.asarray(t, dtype=float), np.asarray(y_seq[:, 0], dtype=float)  # Tz en sortie 0


def detect_oscillation(t: np.ndarray, y: np.ndarray, setpoint: float) -> tuple[bool, float]:
    """Détecte une oscillation entretenue (zéros de signe + stabilité période/amplitude).
    Sensibilité accrue : tolérance amplitude ±35%, période prise sur les 8 derniers demi-cycles."""
    if t.shape[0] < 4:
        return False, np.nan

    err = setpoint - y
    dt = t[1] - t[0] if t.shape[0] > 1 else 1.0
    warm_start = int(WARMUP_DURATION_S // dt)
    start = max(warm_start, len(err) // 3)
    start = min(start, len(err) - 2)

    warm = err[start:]
    t_warm = t[start:]

    # Zéros de signe -> estimation période
    zero_idx = np.where(np.diff(np.signbit(warm)))[0]
    if zero_idx.size < 6:
        return False, np.nan
    times = t_warm[zero_idx]
    periods = np.diff(times)[-8:] * 2.0
    if periods.size == 0:
        return False, np.nan
    pu = periods.mean()
    if pu <= 0:
        return False, np.nan

    # Pics absolus par segment entre zéros pour vérifier la stabilité d'amplitude
    peaks = []
    last = 0
    for z in zero_idx:
        seg = warm[last : z + 1]
        peaks.append(np.max(np.abs(seg)))
        last = z + 1
    if len(peaks) < 6:
        return False, np.nan

    recent = np.asarray(peaks[-6:], dtype=float)
    if np.any(recent == 0):
        return False, np.nan
    amp_var = recent.std() / recent.mean()
    sustained = amp_var < 0.35  # amplitude à +/-35% stable

    return sustained, pu if sustained else (False, np.nan)


def find_ultimate_gain(sim_base: Simulation_JAX, kp_max: float = 1e3):
    kp = 1e-3
    last_t, last_y = None, None

    while kp <= kp_max:
        t, y = run_closed_loop(sim_base, kp=kp, ki=0.0)
        osc, pu = detect_oscillation(t, y, SETPOINT)
        if osc:
            return {"found": True, "ku": kp, "pu": pu, "t": t, "y": y}
        last_t, last_y = t, y
        kp *= 1.5

    return {"found": False, "ku": kp / 1.5, "pu": np.nan, "t": last_t, "y": last_y}


def tune_pi():
    sim = _load_sim()
    res = find_ultimate_gain(sim)
    if not res["found"] or not np.isfinite(res["pu"]):
        print("⚠️ Oscillation ultime introuvable : augmente kp_max ou HORIZON_S si besoin.")
        return res

    ku, pu = res["ku"], res["pu"]
    kp = 0.45 * ku
    ki = 1.2 * ku / pu
    res.update({"Ku": ku, "Pu": pu, "Kp_PI": kp, "Ki_PI": ki})
    return res


if __name__ == "__main__":
    sim = _load_sim()
    kp = 1e-3
    kp_max = 1e3
    factor = 1.5

    while kp <= kp_max:
        print(f"\nEssai Kp (P seul) = {kp:.4g}")
        t, y = run_closed_loop(sim, kp=kp, ki=0.0)
        osc, pu = detect_oscillation(t, y, SETPOINT)

        plt.figure(figsize=(8, 3))
        plt.plot(t / 3600.0, y - 273.15, label="Tz (°C)")
        plt.axhline(SETPOINT - 273.15, color="k", linestyle="--", label="Consigne (°C)")
        plt.axvspan(0, WARMUP_DURATION_S / 3600.0, color="grey", alpha=0.15, label="Warmup")
        if osc:
            plt.title(f"Oscillation détectée, Pu≈{pu:.2g} s (Kp={kp:.4g})")
        else:
            plt.title(f"Aucune oscillation détectée (Kp={kp:.4g})")
        plt.xlabel("Temps (h)")
        plt.ylabel("Température (°C)")
        plt.legend()
        plt.tight_layout()
        plt.show()  # si tu fermes, la boucle continue

        resp = input("Tape 'yes' pour arrêter et sauvegarder ce Kp comme Ku, sinon entrée pour continuer: ").strip().lower()
        if resp == "yes":
            ku = kp
            if osc and np.isfinite(pu):
                kp_pi = 0.45 * ku
                ki_pi = 1.2 * ku / pu
            else:
                pu = np.nan
                kp_pi = np.nan
                ki_pi = np.nan
            with open("ZH_gains.txt", "w", encoding="utf-8") as f:
                f.write(f"Ku={ku}\nPu={pu}\nKp_PI={kp_pi}\nKi_PI={ki_pi}\n")
            print(f"Gains sauvegardés dans ZH_gains.txt\nKu={ku:.4g}, Pu={pu:.4g}, Kp_PI={kp_pi:.4g}, Ki_PI={ki_pi:.4g}")
            break

        kp *= factor
    else:
        print("Fin de boucle sans arrêt utilisateur : augmenter kp_max ou adapter le facteur.")
