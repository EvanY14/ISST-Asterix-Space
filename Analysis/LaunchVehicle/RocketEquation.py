import numpy as np

def solve_rocket_equation(dV  =   None,
                          Isp =   None,
                          m0  =   None,
                          mf  =   None):
    """
    Returns complete set of rocket equation parameters when given all but one

    :param dV:      Delta-V             [m/s, array-like]
    :param Isp:     Specific Impulse    [s, array-like]
    :param m0:      Total Mass          [kg, array-like]
    :param mf:      Dry Mass            [kg, array-like]
    :return:
    """
    dV_arr      = np.asarray(dV)
    Isp_arr     = np.asarray(Isp)
    m0_arr      = np.asarray(m0)
    mf_arr      = np.asarray(mf)

    try:
        if dV_arr == None:
            dV_arr = Isp_arr * 9.81 * np.log(m0_arr/af_arr)

        elif Isp_arr == None:
            Isp_arr = (dV_arr/9.81) / np.log(m0_arr/af_arr)

        elif m0_arr == None:
            m0_arr = np.exp(dV_arr / (9.81 * Isp_arr)) * mf_arr

        elif mf_arr == None:
            mf_arr = m0_arr / np.exp(dV_arr / (9.81 * Isp_arr))
    except:
        raise RocketError('Cannot solve rocket equation with more than one unknown.')

    return dV_arr.tolist(), Isp_arr.tolist(), m0_arr.tolist(), mf_arr.tolist()