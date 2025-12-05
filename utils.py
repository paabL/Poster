#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 19:28:00 2025

@author: pablo

"""

import jax.numpy as jnp

def RC5_steady_state_sys(Ta, Q_solar, Q_con, Q_rad, Qc_dot_val, theta):
    """Résout le steady-state du modèle RC5 (dérivées nulles)."""
    th = theta["th"]
    R_inf, R_w1, R_w2, R_i, R_f, R_c, gA = (
        th[k] for k in ["R_inf", "R_w1", "R_w2", "R_i", "R_f", "R_c", "gA"]
    )
    Q_occ = Q_con + Q_rad

    A = jnp.array([
        [-(1/R_inf+1/R_w2+1/R_f+1/R_i),  1/R_w2,  1/R_i,  1/R_f, 0],
        [ 1/R_w2, -(1/R_w1+1/R_w2),      0,       0,      0],
        [ 1/R_i,  0,                    -1/R_i,   0,      0],
        [ 1/R_f,  0,                     0,     -(1/R_f+1/R_c), 1/R_c],
        [ 0,      0,                     0,       1/R_c, -1/R_c],
    ], dtype=jnp.float64)

    b = jnp.array([
        -Ta/R_inf - gA*Q_solar - Q_occ,
        -Ta/R_w1,
         0.0,
         0.0,
        -Qc_dot_val,
    ], dtype=jnp.float64)

    return jnp.linalg.solve(A, b)  # [Tz, Tw, Ti, Tf, Tc]