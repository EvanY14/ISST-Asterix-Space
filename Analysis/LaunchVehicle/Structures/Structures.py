import numpy as np
from copy import deepcopy

def size_propellant_mass(mu,
                         sigma,
                         m_pl):
    """
    Determines the size of the propellant mass for a given structural ratio,

    :param mu:      Mass ratio, mu = m_0/m_f
    :param sigma:   Structural fraction, sigma = m_s/(m_s + m_p)
    :param m_pl:    Payload mass, m_pl = m_0 - m_p - m_s
    :return:        m_p:    Payload mass
    """
    return m_pl * (mu - 1) * (1. - sigma) / (1 - mu * sigma)

def size_structural_mass(m_p,
                         sigma):
    """
    Determines the structural mass for a given propellant mass and
    structural fraction.

    :param m_p:     Propellant mass, m_p = m_0 - m_pl - m_s
    :param sigma:   Structural fraction, sigma = m_s/(m_s + m_p)
    :return:        m_s:    Structural mass
    """
    return m_p * sigma/(1 - sigma)

def size_tank(V_p,
              d_i,
              AR = np.sqrt(2)):
    """
    Sizes propellant tanks based on the volume of propellant and stage diameter.

    :param V_p: Volume of propellant, V_p = m_p/rho_p
    :param d_i: Stage diameter, d_i = 2 * r_i
    :param AR:  Tank aspect ratio, AR = sqrt(2) for liquids, AR = 1 for solids
    :return:    h_cyl: Cylinder height, dome_AR = 1 if spherical sizing is found
    """

    r_i = d_i / 2.

    if V_p < 4/3 * np.i * r_i**3:
        h_cyl = 0.
        dome_AR = 1.

    else:
        h_cyl = (V_p - (2/AR * 2/3* np.pi * r_i**3))/(np.pi * r_i**2)
        dome_AR = AR

    return h_cyl, dome_AR

def size_accessories(dome_ARs:np.ndarray,
                     stage_diameters:np.ndarray):
    """
    Determines the size of booster accessories based on the staged dome ARs and
    stage diameters.

    :param dome_ARs:        NP Array of dome aspect ratios
    :param stage_diameters: NP Array of stage diameters
    :return:                dome_ARs:    Array of dome ARs
                            stage_diameters: Array of stage diameters
                            h_domes:         Array of dome heights
                            h_plf:           Payload fairing height
                            h_skirts:        Array of skirt heights
                            h_intertank:     Array of intertank heights
                            h_interstage:    Array of interstage heights
    """

    h_domes         = (stage_diameters/2.) / dome_ARs
    h_plf           = stage_diameters[-1] * 2.
    h_skirts        = 1 / (3. * stage_diameters) + h_domes
    h_intertank     = 1 / (4. * stage_diameters) * 2 * h_domes
    h_interstage    = deepcopy(stage_diameters)
    for ii, stage in stage_diameters:
        if np.isclose(dome_ARs[ii], 1.):
            h_interstage[ii] = 1.25 * stage_diameters[ii]

    return dome_ARs, stage_diameters, h_domes, h_plf, h_skirts, h_intertank, h_interstage

def wind_envelope(h):
    """
    Determines the AMR 95% maximum wind speed for a given height.

    :param h:   Height/Altitude in kms
    :return:    AMR 95% Wind Speed in m/s
    """

    if h <= 9.6:
        v_w = 6.9228 * h + 9.144
    elif h <= 14.:
        v_w = 76.2
    elif h <= 20.:
        v_w = 76.2 - 8.9474 * (h - 14.)
    else:
        v_w = 24.384

    return v_w