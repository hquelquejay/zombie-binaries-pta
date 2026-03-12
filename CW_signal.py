import astropy.constants as const
import astropy.units as u
import numpy as np
import numba as nb

import smbhbs

T_sun = const.M_sun * const.G / const.c ** 3

@nb.njit
def create_gw_antenna_pattern_numba(pos, 
                                    gwtheta, gwphi, 
                                    psi=0.0):
    """
    Compute pulsar antenna pattern functions for gravitational waves.
    
    :param pos: Unit vector from Earth to pulsar (array of shape (3,))
    :param gwtheta: GW polar angle in radians (scalar)
    :param gwphi: GW azimuthal angle in radians (scalar)
    :param psi: GW polarization angle (scalar, default 0)
    
    :return: (fplus, fcross, cosMu)
    """
    sin_gwphi = np.sin(gwphi)
    cos_gwphi = np.cos(gwphi)

    sin_gwtheta = np.sin(gwtheta)
    cos_gwtheta = np.cos(gwtheta)
    
    sin_2psi = np.sin(2 * psi)
    cos_2psi = np.cos(2 * psi)

    # Define GW unit vectors (explicit components)
    m_x = sin_gwphi
    m_y = -cos_gwphi
    
    n_x = -cos_gwtheta * cos_gwphi
    n_y = -cos_gwtheta * sin_gwphi
    n_z = sin_gwtheta
    
    omhat_x = -sin_gwtheta * cos_gwphi
    omhat_y = -sin_gwtheta * sin_gwphi
    omhat_z = -cos_gwtheta

    # Compute dot products explicitly (component-wise dot product)
    dot_omhat_pos = omhat_x * pos[0] + omhat_y * pos[1] + omhat_z * pos[2]
    dot_m_pos = m_x * pos[0] + m_y * pos[1] # + 0
    dot_n_pos = n_x * pos[0] + n_y * pos[1] + n_z * pos[2]

    # Compute fplus and fcross
    denominator = 0.5 / (1.0 + dot_omhat_pos)
    
    # Pre-compute standard responses
    plus_resp = (dot_m_pos ** 2 - dot_n_pos ** 2)
    cross_resp = 2.0 * dot_m_pos * dot_n_pos
    
    # Compute antenna response
    fplus = denominator * (cos_2psi * plus_resp + sin_2psi * cross_resp)
    fcross = denominator * (-sin_2psi * plus_resp + cos_2psi * cross_resp)

    return fplus, fcross, -dot_omhat_pos

def cw_delay_zombie(toas,
                    pos,
                    cos_gwtheta=0,
                    gwphi=0,
                    cos_inc=0,
                    log10_mc=9,
                    q=1, 
                    log10_fgw=-8,
                    log10_dist=None,
                    phase_c=0,
                    psi=0,
                    evolve=False,
                    tref=0):
    """
    Compute timing residuals from a continuous gravitational wave (CW) signal 
    from an zombie (PT only) binary system in a pulsar timing array.
    
    Parameters
    ----------
    toas : array-like
        Times of arrival of pulsar signals (in seconds).
    pos : array-like, shape (3,)
        Unit vector from Earth to pulsar.
    cos_gwtheta : float, optional
        Cosine of the binary polar angle (default: 0).
    gwphi : float, optional
        Binary azimuthal angle in radians (default: 0).
    cos_inc : float, optional
        Cosine of the binary orbital inclination angle (default: 0).
    log10_mc : float, optional
        Log10 of the binary chirp mass in solar masses (default: 9).
    q : float, optional
        Binary mass ratio (default: 1).
    log10_fgw : float, optional
        Log10 of the GW frequency in Hz (default: -8).
    log10_dist : float, optional
        Log10 of the distance to the source in Mpc. Either this or log10_h must be specified.
    phase_c : float, optional
        Reference phase of the GW at coalescence (default: 0).
    psi : float, optional
        GW polarization angle in radians (default: 0).
    evolve : bool, optional
        If True, include frequency evolution due to GW emission. 
        If False, treat as monochromatic source (default: False).
    tref : float, optional
        Reference time for phase calculation in seconds (default: 0).
    
    Returns
    -------
    res_P : array-like
        Timing residuals due to the GW signal at each pulsar arrival time.
    """

    # convert units to time
    mc_sun = 10 ** log10_mc
    mc = mc_sun * T_sun
    fgw = 10 ** log10_fgw * u.Hz
    gwtheta = np.arccos(cos_gwtheta)
    inc = np.arccos(cos_inc)

    # Get total binary mass for ISCO (all in observer frame)
    Mtot = smbhbs.getMtot(Mc=mc_sun * const.M_sun,
                          q=q)
    worb_ISCO = 2 * np.pi * smbhbs.getforb_ISCO(Mtot=Mtot)

    dist = (10 ** log10_dist * u.Mpc / const.c).to(u.s)

    # get antenna pattern funcs and cosMu, write function to get pos from theta,phi
    fplus, fcross, _ = create_gw_antenna_pattern_numba(pos, gwtheta, gwphi, psi=psi)

    # get pulsar time 
    # NOTE: no need for Delta tau_a since no Earth term (reference frequency is forb^(P,a))
    tp = toas - tref

    # Orbital frequency at tref - Delta tau_a
    w0_P = np.pi * fgw

    # Compute characteristic coalescence time at tref
    Tc = 5 / 256 * mc ** (-5/3) * w0_P ** (-8/3)

    # Compute time of coalescence (relative to tref)
    tp_coal = Tc * (1 - (worb_ISCO / w0_P) ** (-8/3))

    # evolution
    if evolve:
        # Compute the evolution frequency factor
        evol_fact = 1 - tp / Tc

        # If coalescence happended, simulate merger by putting forb_P to ~1e20 Hz
        # This yields to 0 amplitude in residuals
        w_P = np.where(tp < tp_coal, 
                       w0_P * evol_fact ** (-3 / 8),
                       1e20 * u.Hz)

        # Calculate time dependent phase
        # If coalescence happened, fix to phase_c
        phase_p = np.where(
                    tp < tp_coal,
                    phase_c - 
                    (
                        1 / 32 / mc ** (5 / 3) * w_P ** (-5 / 3) *
                        (1 - (worb_ISCO / w_P) ** (-5 / 3))
                    ).to(u.dimensionless_unscaled).value,
                    phase_c
        )

    else:
        # Assume the source to be monochromatic 
        w_P = w0_P
        
        phase_p = phase_c - (w_P * (tp_coal - tp)).to(u.dimensionless_unscaled).value

    # define time dependent coefficients
    At_p = -0.5 * np.sin(2 * phase_p) * (3 + np.cos(2 * inc))
    Bt_p = 2 * np.cos(2 * phase_p) * np.cos(inc)

    # now define time dependent amplitudes
    alpha_p = mc ** (5.0 / 3.0) / (dist * w_P ** (1.0 / 3.0))

    # define rplus and rcross
    rplus_p = alpha_p * (-At_p * np.cos(2 * psi) + Bt_p * np.sin(2 * psi))
    rcross_p = alpha_p * (At_p * np.sin(2 * psi) + Bt_p * np.cos(2 * psi))

    # residuals
    res_P = fplus * rplus_p + fcross * rcross_p

    return res_P
