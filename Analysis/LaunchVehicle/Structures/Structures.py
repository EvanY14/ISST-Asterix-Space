import numpy as np
from copy import deepcopy

# Step 1: Set Stage Parameters as lists of numbers:
#   Delta-V's (Dv)
#   Structural Fractions (sigma),
#   Specific impulses (Isp)
#
# Example (Numbers from Ariane 5 ECA Core & Second Stage:
# Dv    = [3600.0,  2050.0]
# sigma = [0.08,      0.23]
# Isp   = [310.0,    446.0]

def size_propellant_mass(mu, sigma, m_pl):
    return m_pl * (mu - 1) * (1. - sigma) / (1 - mu * sigma)

def size_structural_mass(m_p, sigma):
    return m_p * sigma/(1 - sigma)

def size_tank(V_p, d_i, AR = np.sqrt(2)):

    r_i = d_i / 2.

    if V_p < 4/3 * np.i * r_i**3:
        h_cyl = 0.
        dome_AR = 1.

    else:
        h_cyl = (V_p - (2/AR * 2/3* np.pi * r_i**3))/(np.pi * r_i**2)
        dome_AR = AR

    return h_cyl, dome_AR

def size_accessories(dome_ARs, stage_diameters):

    h_domes         = (stage_diameters/2.) / dome_ARs
    h_plf           = stage_diameters[-1] * 2.
    h_skirts        = 1 / (3. * stage_diameters) + h_domes
    h_intertank     = 1 / (4. * stage_diameters) * 2 * h_domes
    h_interstage    = deepcopy(stage_diameters)
    for ii, stage in stage_diameters:
        if np.isclose(dome_ARs[ii], 1.):
            h_interstage[ii] = 1.25 * stage_diameters[ii]

    return dome_ARs, stage_diameters, h_domes, h_plf, h_skirts, h_intertank

def wind_envelope(h):

    if h <= 9.6:
        v_w = 6.9228 * h + 9.144
    elif h <= 14.:
        v_w = 76.2
    elif h <= 20.:
        v_w = 76.2 - 8.9474 * (h - 14.)
    else:
        v_w = 24.384

    return v_w