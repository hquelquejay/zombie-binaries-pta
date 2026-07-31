import astropy.units as u
import jax
import jax.numpy as jnp
import numba as nb
import numpy as np

import smbhbs
from utils import (M_sun, M_sun_val, Mpc_val, T_sun, T_sun_val, c, c_val,
                   kpc2s_val, yr_val)

# ===================== JAX VERSION of zombie delay ================================


@jax.jit
def jax_create_gw_antenna_pattern(pos: jax.Array, gwtheta: jax.Array,
                                   gwphi: jax.Array, psi: jax.Array = jnp.array(0.0)):
    """Construct the GW antenna-pattern response for one or many sky positions.

    Parameters
    ----------
    pos : jax.Array
        Unit vector(s) pointing to the pulsar location.
    gwtheta : jax.Array
        Polar angle of the GW source direction.
    gwphi : jax.Array
        Azimuthal angle of the GW source direction.
    psi : jax.Array, optional
        GW polarization angle.

    Returns
    -------
    tuple
        Plus response, cross response, and line-of-sight projection term.
    """
    sin_gwphi = jnp.sin(gwphi)
    cos_gwphi = jnp.cos(gwphi)
    sin_gwtheta = jnp.sin(gwtheta)
    cos_gwtheta = jnp.cos(gwtheta)
    sin_2psi = jnp.sin(2 * psi)
    cos_2psi = jnp.cos(2 * psi)

    # GW unit vectors
    m_x = sin_gwphi
    m_y = -cos_gwphi

    n_x = -cos_gwtheta * cos_gwphi
    n_y = -cos_gwtheta * sin_gwphi
    n_z = sin_gwtheta

    omhat_x = -sin_gwtheta * cos_gwphi
    omhat_y = -sin_gwtheta * sin_gwphi
    omhat_z = -cos_gwtheta

    # Works for both (3,) and (..., 3) — no conditional needed
    dot_omhat_pos = omhat_x * pos[..., 0] + omhat_y * pos[..., 1] + omhat_z * pos[..., 2]
    dot_m_pos     = m_x     * pos[..., 0] + m_y     * pos[..., 1]
    dot_n_pos     = n_x     * pos[..., 0] + n_y     * pos[..., 1] + n_z * pos[..., 2]

    denominator = 0.5 / (1.0 + dot_omhat_pos)

    plus_resp  = dot_m_pos ** 2 - dot_n_pos ** 2
    cross_resp = 2.0 * dot_m_pos * dot_n_pos

    fplus  = denominator * ( cos_2psi * plus_resp + sin_2psi * cross_resp)
    fcross = denominator * (-sin_2psi * plus_resp + cos_2psi * cross_resp)

    return fplus, fcross, -dot_omhat_pos

def cw_delay_zombie_jax(toas: jnp.ndarray, 
                        pos: jnp.ndarray,
                        log10_mc: jnp.ndarray,
                        q: jnp.ndarray,
                        log10_dist: jnp.ndarray,
                        log10_fgw: jnp.ndarray,
                        cos_gwtheta: jnp.ndarray,
                        gwphi: jnp.ndarray,
                        cos_inc: jnp.ndarray,
                        phase_c: jnp.ndarray,
                        psi: jnp.ndarray,
                        tref: jnp.ndarray) -> jnp.ndarray:
    """
    Compute CW timing residuals for zombie binaries (JAX version)
    
    Args:
        toas: Times of arrival (shape: (N_toa,))
        pos: Pulsar position unit vector (shape: (3,))
        cos_gwtheta: Cosine of GW polar angle
        gwphi: GW azimuthal angle (radians)
        cos_inc: Cosine of orbital inclination
        log10_mc: Log10 chirp mass in solar masses
        q: Mass ratio
        log10_fgw: Log10 GW frequency (Hz)
        log10_dist: Log10 distance (Mpc)
        phase_c: Reference phase at coalescence
        psi: GW polarization angle
        evolve: Whether to include frequency evolution
        tref: Reference time
    
    Returns:
        Timing residuals (shape: (N_toa,))
    """
    # Convert to SI units
    mc_sun = 10 ** log10_mc
    Gmc = mc_sun * T_sun_val # G Mc / c^3
    fgw = 10 ** log10_fgw
    gwtheta = jnp.arccos(cos_gwtheta)
    inc = jnp.arccos(cos_inc)
    dist = 10 ** log10_dist * Mpc_val / c_val

    # Get total mass for ISCO
    Mtot = smbhbs.getMtot_jax(Mc=mc_sun * M_sun_val, q=q)
    worb_ISCO = 2 * jnp.pi * smbhbs.getforb_ISCO_jax(Mtot=Mtot)

    # Get antenna pattern
    fplus, fcross, _ = jax_create_gw_antenna_pattern(pos=pos,
                                                     gwtheta=gwtheta,
                                                     gwphi=gwphi, 
                                                     psi=psi)

    # Pulsar time relative to reference
    tp = toas - tref

    # Orbital frequency at tref
    w0_P = jnp.pi * fgw

    # Characteristic coalescence time
    Tc = 5 / 256 * Gmc ** (-5. / 3) * w0_P ** (-8. / 3)

    # Time of coalescence relative to tref
    tp_coal = Tc * (1 - (worb_ISCO / w0_P) ** (-8. / 3))

    # NOTE: in jax, evolve version only
    evol_fact = 1 - tp / Tc
    safe_evol = jnp.where(evol_fact > 0, evol_fact, jnp.ones_like(evol_fact))

    # NOTE: if tp < tp_coal, evol_fact > 0
    w_P = jnp.where(tp < tp_coal, 
                    w0_P * safe_evol ** (-3./8), 
                    1e20 # After merger, proxy for killing signal
    )

    
    phase_p = jnp.where(
        tp < tp_coal,
        phase_c - (
            1 / 32 *
            (Gmc * w_P) ** (-5. / 3) *
            (1 - (worb_ISCO / w_P) ** (-5. / 3))
        ),
        phase_c
    )

    # Time-dependent coefficients
    rplus_p = 0.5 * jnp.sin(2 * phase_p) * (3 + jnp.cos(2 * inc))
    rcross_p = 2 * jnp.cos(2 * phase_p) * cos_inc

    # Time-dependent amplitudes
    alpha_p = Gmc ** (5. / 3) / (dist * w_P ** (1. / 3))

    # Residuals
    return alpha_p * (fplus * rplus_p + fcross * rcross_p)

@jax.jit()
def cw_delay_zombie_cache_jax(tp: jnp.ndarray, 
                              Gmc: jnp.ndarray,
                              worb_ISCO: jnp.ndarray,
                              fplus: jnp.ndarray,
                              fcross: jnp.ndarray,
                              At_inc_pref: jnp.ndarray,
                              Bt_inc_pref: jnp.ndarray,
                              alpha_pref: jnp.ndarray,
                              log10_fgw: jnp.ndarray,
                              phase_c: jnp.ndarray) -> jnp.ndarray:
    """
    Compute CW timing residuals for zombie binaries (JAX version)
    
    Args:
        toas: Times of arrival (shape: (N_toa,))
        pos: Pulsar position unit vector (shape: (3,))
        cos_gwtheta: Cosine of GW polar angle
        gwphi: GW azimuthal angle (radians)
        cos_inc: Cosine of orbital inclination
        log10_mc: Log10 chirp mass in solar masses
        q: Mass ratio
        log10_fgw: Log10 GW frequency (Hz)
        log10_dist: Log10 distance (Mpc)
        phase_c: Reference phase at coalescence
        psi: GW polarization angle
        evolve: Whether to include frequency evolution
        tref: Reference time
    
    Returns:
        Timing residuals (shape: (N_toa,))
    """
    # Orbital frequency at tref
    w0_P = jnp.pi * 10 ** log10_fgw

    # Characteristic coalescence time
    Tc = 5 / 256 * Gmc ** (-5. / 3) * w0_P ** (-8. / 3)

    # Time of coalescence relative to tref
    # tp_coal = Tc * (1 - (worb_ISCO / w0_P) ** (-8. / 3))

    # NOTE: in jax, evolve version only
    evol_fact = 1 - tp / Tc

    # Compute the threshold over evol factor once
    evol_thresh = (worb_ISCO / w0_P) ** (-8. / 3)

    w_P = jnp.where(
        evol_fact > evol_thresh,
        # tp < tp_coal,
        w0_P * evol_fact ** (-3. / 8),
        1e20  # After merger, proxy for killing signal
    )

    phase_p = jnp.where(
        evol_fact > evol_thresh,
        # tp < tp_coal,
        phase_c - (
            1 / 32 / Gmc ** (5. / 3) * 
            (w_P ** (-5. / 3) - worb_ISCO ** (-5. / 3))
        ),
        phase_c
    )

    # Time-dependent coefficients
    At_p = At_inc_pref * jnp.sin(2 * phase_p)
    Bt_p = Bt_inc_pref * jnp.cos(2 * phase_p)

    # Time-dependent amplitudes
    alpha_p = alpha_pref / (w_P ** (1. / 3))

    # Residuals
    return alpha_p * (fplus * (-At_p) + fcross * (Bt_p))

@jax.jit
def zombie_cw_delay(toas, pos, dist_est, dist_err, 
                    log10_mc, q, log10_dist, log10_tauc, 
                    cos_theta, lon, cos_inc, phase_c, psi, 
                    Nsig_D=0.0,
                    tref=0.0):
    """Compute the timing residual induced by a zombie binary signal in a pulsar.

    Parameters
    ----------
    toas : jax.Array
        Pulse times of arrival.
    pos : jax.Array
        Unit vector of the pulsar position.
    dist_est : float
        Estimated distance to the pulsar.
    dist_err : float
        Relative distance uncertainty.
    log10_mc : jax.Array
        Chirp mass of the binary in solar masses, as log10(M_c / M_sun).
    q : jax.Array
        Mass ratio of the binary.
    log10_dist : jax.Array
        Source luminosity distance in Mpc, as log10(d_L / Mpc).
    log10_tauc : jax.Array
        Time-to-coalescence in years, as log10(tauc / yr).
    cos_theta : jax.Array
        Cosine of the binary polar angle.
    lon : jax.Array
        Binary azimuthal angle.
    cos_inc : jax.Array
        Cosine of the orbital inclination.
    phase_c : jax.Array
        Phase at coalescence.
    psi : jax.Array
        GW polarization angle.
    Nsig_D : float, optional
        Number of standard deviations of the distance uncertainty.
    tref : float, optional
        Reference time used for the signal evolution.

    Returns
    -------
    jax.Array
        Timing residuals for the specified pulsar.
    """

    # Compute binary position vector
    sin_theta = jnp.sqrt(1 - cos_theta ** 2)
    binary_pos = jnp.array([sin_theta * jnp.cos(lon),
                            sin_theta * jnp.sin(lon),
                            cos_theta])

    # Compute the GW freauency of the pulsar term using the time to coalescence
    Mc_val = jnp.power(10.0, log10_mc) * M_sun_val
    psr_T = (1. + dist_err * Nsig_D) * dist_est * kpc2s_val
    forbP = smbhbs.getforbPa_zombie_jax(Mc=Mc_val, 
                                        binary_pos=binary_pos, 
                                        psr_pos=pos, 
                                        psr_T=psr_T,
                                        Delta_tauc=jnp.power(10.0, log10_tauc) * yr_val,
                                        q=q)
    
    safe_forbP = jnp.where(jnp.isnan(forbP), jnp.ones_like(forbP), forbP)  # to avoid nan gradients
    signal = cw_delay_zombie_jax(toas=toas,
                                 pos=pos,
                                 log10_mc=log10_mc,
                                 q=q,
                                 log10_dist=log10_dist,
                                 log10_fgw=jnp.log10(2. * safe_forbP), # Under the assumption of quasi-circular orbit
                                 cos_gwtheta=cos_theta,
                                 gwphi=lon,
                                 cos_inc=cos_inc,
                                 phase_c=phase_c,
                                 psi=psi,
                                 tref=tref)
    return jnp.where(jnp.isnan(forbP),
                     jnp.zeros_like(toas),
                     signal
    )

########## JAX VMAPPED RESIDUAL COMPUTATION ##########

# Inner vmap: over Npsr (for one zombie)
_residuals_one_zombie = jax.jit(jax.vmap(
    cw_delay_zombie_jax,
    in_axes=(0, 0, None, None, None, 0, None, None, None, None, None, None),
))
# → returns (Npsr, N_ToA)

# Outer vmap: over N_zombies
_residuals_all = jax.jit(jax.vmap(
    _residuals_one_zombie,
    in_axes=(None, None, 0, 0, 0, 0, 0, 0, 0, 0, 0, None),
))
# → returns (N_zombies, Npsr, N_ToA)

@jax.jit
def compute_residuals_vmap(
    psr_toas,      # (Npsr, N_ToA)  — shared
    psr_pos_s,     # (Npsr, 3)      — shared
    log10_Mc,      # (N_zombies,)
    q,             # (N_zombies,)
    log10_dL,      # (N_zombies,)
    log10_fgwP_s,  # (N_zombies, Npsr)
    cos_theta,     # (N_zombies,)
    phi,           # (N_zombies,)
    cos_inc,       # (N_zombies,)
    phase_c,       # (N_zombies,)
    psi,           # (N_zombies,)
    t0,            # scalar [s]
    PT_valid_pair, # (N_zombies, Npsr)
) -> jax.Array:
    """
    Fully vmapped, jittable residual computation.
    
    Separates the pure JAX computation from random number generation,
    allowing JIT compilation and avoiding redefinition of vmapped functions.
    """

    CW_residuals = _residuals_all(
        psr_toas,      # (Npsr, N_ToA)
        psr_pos_s,     # (Npsr, 3)
        log10_Mc,      # (N_zombies,)
        q,             # (N_zombies,)
        log10_dL,      # (N_zombies,)
        log10_fgwP_s,  # (N_zombies, Npsr)
        cos_theta,     # (N_zombies,)
        phi,           # (N_zombies,)
        cos_inc,       # (N_zombies,)
        phase_c,       # (N_zombies,)
        psi,           # (N_zombies,)
        t0,            # scalar [s]
    )   # (N_zombies, Npsr, N_ToA)

    # Zero out residuals for invalid (zombie, pulsar) pairs
    return jnp.where(PT_valid_pair[:, :, None], 
                     CW_residuals, 
                     0.0)

# Inner vmap: over Npsr (for one zombie)
_residuals_one_zb_cache = jax.jit(jax.vmap(
    cw_delay_zombie_cache_jax,
    in_axes=(0, None, None, 0, 0, None, None, None, 0, None),
))
# → returns (Npsr, N_ToA)

# Outer vmap: over N_zombies
# in_axes: (tp, Gmc, worb_ISCO, fplus, fcross, At_inc_pref, Bt_inc_pref, alpha_pref, log10_fgw, phase_c)
_residuals_all_cache = jax.jit(jax.vmap(
    _residuals_one_zb_cache,
    in_axes=(None, None, None, 0, 0, 0, 0, None, 0, 0),
))
# → returns (N_zombies, Npsr, N_ToA)

@jax.jit
def compute_residuals_cache_vmap(
    tp: jnp.ndarray, 
    Gmc: float,
    worb_ISCO: float,
    fplus: jnp.ndarray,
    fcross: jnp.ndarray,
    At_inc_pref: jnp.ndarray,
    Bt_inc_pref: jnp.ndarray,
    alpha_pref: float,
    log10_fgw: jnp.ndarray,
    phase_c: jnp.ndarray
) -> jax.Array:
    """
    Fully vmapped, jittable residual computation using cached parameters.
    
    This function computes CW residuals for a batch of zombies using pre-cached 
    extrinsic parameters. The vmap handles the outer loop over N_zombies and 
    inner loop over N_psr for antenna pattern application.
    
    Args:
        tp (jnp.ndarray):           Retarded times per pulsar, shape (N_psr, N_toas). Vmapped
        Gmc (jnp.ndarray):          G*Mc scalar. Broadcast.
        worb_ISCO (jnp.ndarray):    Angular ISCO frequency scalar. Broadcast.
        fplus (jnp.ndarray):        + antenna response, shape (N_zombies, N_psr). Vmapped.
        fcross (jnp.ndarray):       x antenna response, shape (N_zombies, N_psr). Vmapped.
        At_inc_pref (jnp.ndarray):  Inclination amplitude prefactor, shape (N_zombies,). Vmapped.
        Bt_inc_pref (jnp.ndarray):  Inclination prefactor, shape (N_zombies,). Vmapped.
        alpha_pref (jnp.ndarray):   Residuals amplitude prefactor scalar. Broadcast.
        log10_fgw (jnp.ndarray):    log10 GW frequency, shape (N_zombies, N_psr). Vmapped.
        phase_c (jnp.ndarray):      Phase at coalescence, shape (N_zombies,). Vmapped.
    
    Returns:
        jax.Array: CW residuals, shape (N_zombies, N_psr, N_toas)
    
    Note:
        The vmap axes specification (None, None, None, 0, 0, 0, 0, None, 0, 0) means:
        - tp, Gmc, worb_ISCO, alpha_pref are broadcast across all zombies
        - fplus, fcross, At_inc_pref, Bt_inc_pref, log10_fgw, phase_c are vmapped over zombie axis
    """
    
    # Call the fully vmapped residual computation
    # Outer vmap handles N_zombies, inner vmap (in _residuals_one_zb_cache) handles N_psr
    return _residuals_all_cache(
        tp, 
        Gmc,
        worb_ISCO,
        fplus,
        fcross,
        At_inc_pref,
        Bt_inc_pref,
        alpha_pref,
        log10_fgw,
        phase_c
    )   # → (N_zombies, Npsr, N_ToA)


# --- NUMPY version -----------

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
    Mtot = smbhbs.getMtot(Mc=mc_sun * M_sun,
                          q=q)
    worb_ISCO = 2 * np.pi * smbhbs.getforb_ISCO(Mtot=Mtot)

    dist = (10 ** log10_dist * u.Mpc / c).to(u.s)

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
