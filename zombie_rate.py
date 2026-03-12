import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import astropy.constants as const
import astropy.units as u
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import scipy.integrate as integrate
from astropy.cosmology import Planck18
from tqdm.auto import tqdm

import CW_signal as cw
import smbhbs
import utils
from binary_pop_model import smbhb_density


def getDelta_tauc_min(log10_Mcr: float, z: float,
                      fth: float,
                      q: float = 1) -> np.ndarray:
    """
    Compute the minimum time delay between coalescence (t_c) and starting time of the PTA (t_0)
    such that forb_E > fth.
    NOTE: If forb_^{(ISCO)} < fth, it is sufficient that Delta_tauc > 0 which 
          ensures that no Earth term is present in PTA data

    Args:
        log10_Mcr (float): Chirp mass of the binary in the rest frame 
        z (float): Redshift of the binary
        fth (float): Threshold frequency for the orbital frequency of the Earth term (observer frame)
        q (float): Mass ratio of the binary

    Returns:
        np.ndarray: Minimum time delay t_0 - t_c
    """

    # Compute observer frame chirp mass
    Mc = (1 + z) * 10 ** log10_Mcr * const.M_sun

    # Compute associated total mass
    Mtot = smbhbs.getMtot(Mc=Mc, q=q)

    # Compute ISCO frequency in observer frame
    forb_ISCO = smbhbs.getforb_ISCO(Mtot=Mtot)

    # Compute associated Tc
    Tc = smbhbs.getCoalescenceTime(Mc=Mc, forb=forb_ISCO)

    return np.where(
        forb_ISCO > fth,
        Tc * (
            1 - (fth / forb_ISCO) ** (-8/3)
        ),
        0. * Tc # To have correct units and shape
    )

def computeZombieResiduals(z: float,
                           log10_Mcr: float,
                           q: float,
                           PTA: dict,
                           fth: float,
                           N_zombies: float = 10_000,
                           evolve: bool = True,
                           get_pars: bool = False,
                           rng=None):
    
    if rng is None:
        rng = np.random.default_rng(seed=42)
    
    # Be sure you are an array
    z, log10_Mcr, q = np.atleast_1d(z), np.atleast_1d(log10_Mcr), np.atleast_1d(q)

    # If only one intrinsic parameter, copy N_zombies times
    # This is important to have coherent valid_bin selection below
    if z.shape[0] == 1:
        z = z * np.ones(N_zombies)
    if log10_Mcr.shape[0] == 1:
        log10_Mcr = log10_Mcr * np.ones(N_zombies)
    if q.shape[0] == 1:
        q = q * np.ones(N_zombies)
    
    # Luminosity distance
    log10_dL = np.log10(Planck18.luminosity_distance(z).to(u.Mpc).value)
    
    # Get observer frame chirp mass (N_zombies,)
    log10_Mc = log10_Mcr + np.log10(1 + z)
    Mc = 10 ** log10_Mc * const.M_sun

    # Generate a time of coalescence Delta tauc for each zombie (uniform over Delta tauc)
    Delta_tauc_min = getDelta_tauc_min(log10_Mcr=log10_Mcr,
                                       z=z, 
                                       q=q,
                                       fth=fth)
    # Draw coalescence times uniformly
    Delta_tauc = (rng.uniform(Delta_tauc_min.to(u.yr).value, 
                              2 * PTA['Ta_max'].to(u.yr).value,
                              size=N_zombies) * u.yr).to(u.s)

    # Generate extrinsic parameters of each zombie
    cos_theta = rng.uniform(-1, 1, size=N_zombies)
    phi = rng.uniform(0, 2 * np.pi, size=N_zombies)
    psi = rng.uniform(0, np.pi, size=N_zombies)
    cos_inc = rng.uniform(-1, 1, size=N_zombies)
    phase_c = rng.uniform(0, 2 * np.pi, size=N_zombies)

    # Compute unit vector positions of each binary
    binary_pos = utils.getUnitPos(cos_theta, phi).T # (N_zombie, 3)

    # Compute the GW frequency at the pulsar term of each pulsar (N_zombie, Npsr)
    # NOTE: This is a nan if the zombie has already merged for this pulsar
    forbP_s = smbhbs.getforbPa_zombie(Mc=Mc[:,None], binary_pos=binary_pos[:,None], 
                                      psr_pos=PTA['psr_pos_s'][None,:], 
                                      psr_T=PTA['psr_T_s'][None,:],
                                      Delta_tauc=Delta_tauc[:,None],
                                      q=q[:,None])
    
    # Identify psr indices that have a Pulsar term for each zombie CW
    PT_valid_pair = ~np.isnan(forbP_s) # (N_zombies, Npsrs)

    # Compute corresponding GW frequency, assuming circular orbit
    log10_fgwP_s = np.where(PT_valid_pair, np.log10(2 * forbP_s.to(u.Hz).value), np.nan)

    ### Compute the CW residuals for each zombie and each pulsar of the PTA
    # NOTE: this will only work if SAME number of toas among pulsars
    CW_residuals = np.full((N_zombies, PTA["Npsr"], PTA['N_ToA']),
                           0., # We initialise to 0 since if not valid pair, no residuals
                           dtype=float,
                    )
    
    # Add dimension to t0
    t0 = PTA['t0'] * u.s
    
    for ip in range(PTA['Npsr']):
        # Extract the simulated zombie CWs which yield a signal in this pulsar
        valid_bin = PT_valid_pair[:, ip]  # (N_zombies,)

        if not np.any(valid_bin):
            # Skip this pulsar if no valid zombie
            continue
        
        # Compute residuals ONLY for valid binaries
        CW_residuals[valid_bin, ip, :] = cw.cw_delay_zombie(
            PTA['psr_toas'][ip][None,:] * u.s,
            PTA['psr_pos_s'][ip],
            cos_gwtheta=cos_theta[valid_bin][:,None],
            gwphi=phi[valid_bin][:,None],
            cos_inc=cos_inc[valid_bin][:,None],
            log10_mc=log10_Mc[valid_bin][:,None],
            q=q[valid_bin][:,None],
            log10_dist=log10_dL[valid_bin][:,None],
            log10_fgw=log10_fgwP_s[valid_bin, ip][:,None],
            phase_c=phase_c[valid_bin][:,None],
            psi=psi[valid_bin][:,None],
            evolve=evolve,
            tref=t0 # Use the start time of the PTA
        ).to(u.s).value # NOTE: no units for jax or CW_SNR_obj

    if get_pars:
        pars = {
            'tauc': Delta_tauc,
            'cos_theta': cos_theta,
            'phi': phi,
            'cos_inc': cos_inc, 
            'log10_fgwP_s': log10_fgwP_s
        }

        return CW_residuals, pars
    
    return CW_residuals




@jax.jit
def single_pulsar_contribution(Sig_cf, F, Ninv, res):
    # rNr term: dot product weighted by diagonal noise
    # This assumes Ninv is a vector of the inverse variance
    rNr = jnp.dot(res, Ninv * res)
    
    # TNr part: F.T @ (Ninv * res)
    # Ninv * res is element-wise multiplication
    TNr = jnp.matmul(F.T, Ninv * res)
    
    # Solve the linear system using the Cholesky decomposition
    # NOTE: we are only storing the Cholesky factor in Sig_cf, not the boolean
    SigTNr = jsp.linalg.cho_solve((Sig_cf, False), TNr)
    
    # Return the contribution to SNR^2
    return rNr - jnp.dot(TNr, SigTNr)

@jax.jit
def compute_SNR_jax(Sig_cf_s, F_s, Ninv_s, CW_residuals):
    """
    Vectorized SNR calculation using jax.vmap.
    """

    # Vectorize over the first axis (0) for all inputs
    # Use vmap to compute the contribution for every pulsar in parallel
    vmapped_contribution = jax.jit(jax.vmap(single_pulsar_contribution, in_axes=(0, 0, 0, 0)))
    
    # Compute the SNR² in each pulsar
    snr2_values = vmapped_contribution(Sig_cf_s, F_s, Ninv_s, CW_residuals)
    
    # Sum the individual SNR² values and take the square root
    # NOTE: this must be changed if HD correlations are included in the GWB noise
    return jnp.sqrt(jnp.sum(snr2_values))

@jax.jit
def compute_all_zombies_snr_jax(Sig_cf_s, F_s, Ninv_s, all_CW_residuals):
    """
    Computes SNRs for all zombies at once.
    all_CW_residuals shape: (N_zombies, N_pulsars, N_toas)
    """
    
    # Sig_cf_s, F_s, and Ninv_s are 'None' because they are the same for every zombie
    # all_CW_residuals is '0' because we want to slice along the zombie dimension
    zombie_vmap = jax.jit(jax.vmap(compute_SNR_jax, in_axes=(None, None, None, 0)))
    
    return zombie_vmap(Sig_cf_s, F_s, Ninv_s, all_CW_residuals)

def computeZombieSNRs_jax(z: float,
                          log10_Mcr: float,
                          q: float,
                          PTA: dict,
                          fth: float,
                          N_zombies: float = 10_000,
                          rng=None):
    
    # TODO: we could also jaxify this
    # This is also very demanding in terms of memory, should change that for larger PTA datasets
    CW_residuals = computeZombieResiduals(z=z,
                                          log10_Mcr=log10_Mcr,
                                          q=q,
                                          PTA=PTA,
                                          fth=fth,
                                          N_zombies=N_zombies,
                                          rng=rng)
    
    return compute_all_zombies_snr_jax(PTA['Sig_cf_s'], PTA['F_s'], PTA['Ninv_s'], CW_residuals)

def compute_efficiency_2d_parallel_jax(log10_Mcr_mids: np.ndarray,
                                       z_mids: np.ndarray,
                                       N_bin: int,
                                       SNR_thresh: float,
                                       PTA: dict,
                                       fth: float,
                                       N_zombies_per_bin: int = 10_000,
                                       n_jobs: int = -1,
                                       seed: int = 42) -> np.ndarray:
    """
    Compute the 2D efficiency function P(SNR > SNR_thresh | log10_Mcr, z).

    Parallel implementation using ThreadPoolExecutor
    """

    # Resolve number of workers
    if n_jobs == -1:
        max_workers = os.cpu_count() or 1
    else:
        max_workers = max(1, int(n_jobs))

    print(max_workers, "CPUs used for parallelisation.", flush=True)

    # Prepare grid points
    grid_points = [
        (i, log10_Mcr, j, z)
        for i, log10_Mcr in enumerate(log10_Mcr_mids)
        for j, z in enumerate(z_mids)
    ]

    # Worker function (defined as nested function; threads don't need pickling)
    def worker(task):
        i, log10_Mcr, j, z = task
        
        # Create a worker-local RNG to ensure reproducibility
        worker_rng = np.random.default_rng(seed=i * N_bin + j + seed)
        
        # NOTE: Speed up the parameter space where it is zero for sure
        if (SNR_thresh > 1) and (log10_Mcr < 7.5):
            return i, j, 0.

        # Compute SNRs marginalizing over extrinsic parameters
        SNRs = computeZombieSNRs_jax(
                                z=z,
                                log10_Mcr=log10_Mcr,
                                q=1.0, # NOTE: we neglect dependence over q here
                                PTA=PTA,
                                fth=fth,
                                N_zombies=N_zombies_per_bin,
                                rng=worker_rng
                            )

        eff_val = np.sum(SNRs > SNR_thresh) / float(N_zombies_per_bin)
        
        return i, j, eff_val

    # Initialize efficiency grid
    efficiency = np.zeros((N_bin, N_bin))

    # Run threaded map with progress
    with ThreadPoolExecutor(max_workers=max_workers) as exc:
        futures = [exc.submit(worker, gp) for gp in grid_points]

        for fut in tqdm(as_completed(futures), total=len(futures)):
            i, j, eff_val = fut.result()
            efficiency[i, j] = eff_val

    return efficiency


def sampleZombiePars_jax(z: float,
                         log10_Mcr: float,
                         q: float,
                         PTA: dict,
                         fth: float,
                         N_zombies: float = 10_000,
                         rng=None):
    
    # TODO: we could also jaxify this
    CW_residuals, pars = computeZombieResiduals(z=z,
                                          log10_Mcr=log10_Mcr,
                                          q=q,
                                          PTA=PTA,
                                          fth=fth,
                                          N_zombies=N_zombies,
                                          get_pars=True,
                                          rng=rng)
    
    return compute_all_zombies_snr_jax(PTA['Sig_cf_s'], PTA['F_s'], PTA['Ninv_s'], CW_residuals), pars



def getNbarZombies(pop_model, 
                   efficiency: np.ndarray,
                   log10_Mcr_mids: np.ndarray,
                   z_mids: np.ndarray,
                   Ta_max: float,
                   fth: float,
                   cosmo = Planck18,
                   Dlogz: float = None) -> float:
    """
    Get the average number of zombie CW verifying SNR > SNR_thresh (given by efficiency)

    Args:
        pop_model (dict): Dictionary of comoving merger density parameters
        efficiency (np.ndarray): Array containing the efficiency of detection given intrinsic parameters
        log10_Mcr_mids (np.ndarray): Grid of rest frame chirp mass used to compute efficiency
        z_mids (np.ndarray): Grid of redshift used to compute efficiency
        Ta_max (float): Photon time of flight associated with the more distant pulsar of the PTA
        fth (float): Threshold frequency for the Earth term of zombie binaries
        cosmo (object, optional): Astropy cosmology object. Defaults to Planck18.
        Dlogz (float, optional): Width of the logz bins if a log10-spaced redshift basis has been used

    Returns:
        float: Average number of zombie CWs
    """
    
    # Compute prefactor
    prefactor = 4 * np.pi * const.c

    # (Mcr, z)
    density_factor = (
        cosmo.luminosity_distance(z_mids[None,:]).to(u.Mpc).value ** 2 # in Mpc units
        / (1 + z_mids[None,:]) ** 2
        * (
            2 * Ta_max - getDelta_tauc_min(log10_Mcr=log10_Mcr_mids[:,None], 
                                           z=z_mids[None,:],
                                           fth=fth, 
                                           q=1) # NOTE: incorporating q != 1 would change nothing
        ).to(u.kyr).value
        * smbhb_density(log10_Mcr_mids[:,None], z_mids[None,:], **pop_model)
    )
    
    # Rescale with correct units
    density_factor *= (
        u.Mpc ** (-3 + 2) # volume of smbhb_density + dL
        * u.kyr # Delta tau_c 
    )

    # Widths checking uniform grid first
    assert np.allclose(np.diff(log10_Mcr_mids), np.diff(log10_Mcr_mids)[0])
    log10_Mcr_bin = np.diff(log10_Mcr_mids)[0] * np.ones_like(log10_Mcr_mids)
    if Dlogz is None:
        # Linear spacing used
        assert np.allclose(np.diff(z_mids), np.diff(z_mids)[0])
        z_bin = np.diff(z_mids)[0] * np.ones_like(z_mids)
    else:
        # Log spacing used
        assert np.allclose(np.diff(np.log10(z_mids)), np.diff(np.log10(z_mids))[0])
        z_bin = Dlogz * np.log(10) * z_mids # Include the conversion of dn/dz into dn/dlogz here

    return np.sum((
        prefactor * 
        log10_Mcr_bin *
        z_bin *
        density_factor *
        efficiency
    ).to(u.dimensionless_unscaled))