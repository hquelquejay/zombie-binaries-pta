import functools
import os
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import astropy.constants as const
import astropy.units as u
import jax
import jax.numpy as jnp
import numpy as np
import scipy.integrate as integrate
from astropy.cosmology import Planck18
from tqdm.auto import tqdm
from zombie_snr import (compute_all_zombies_snr_jax,
                        compute_all_zombies_snr_per_psr_jax)
from ZombieParameterCache import ZombieParameterCache

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

def getNavgZombies(pop_model, 
                   Ta_max: float, 
                   fth: float,
                   xi_I_bounds: dict,
                   cosmo=Planck18) -> float:
    """
    Get the average number of zombie CW verifying forbE > fth 

    Args:
        pop_model (_type_): _description_
        Ta_max (float): Photon time of flight associated with the more distant pulsar of the PTA
        fth (float): _description_
        xi_I_bounds (dict): _description_
        Mcr_scaling (float, optional): _description_. Defaults to 1e9.
        cosmo (_type_, optional): _description_. Defaults to Planck18.

    Returns:
        float: Average number of zombie CWs
    """
    
    # Compute prefactor
    prefactor = 4 * np.pi * const.c

    def integrand_num(log10_Mcr: float, z: float) -> float:
        """Return the differential contribution to the zombie population integral.

        Parameters
        ----------
        log10_Mcr : float
            Chirp mass of the binary in log10(M_sun).
        z : float
            Redshift of the binary.

        Returns
        -------
        float
            Integrand value for the expected zombie count.
        """
        return (
            cosmo.luminosity_distance(z).to(u.Mpc).value ** 2 # in Mpc units
            / (1 + z) ** 2
            * (
                2 * Ta_max - getDelta_tauc_min(log10_Mcr=log10_Mcr, z=z,
                                               fth=fth, 
                                               q=1) # NOTE: incorporating q != 1 would change nothing
            ).to(u.kyr).value
            * smbhb_density(log10_Mcr, z, **pop_model)
        )

    n_factor = integrate.dblquad(
        lambda log10_Mcr, z: integrand_num(log10_Mcr, z),
        xi_I_bounds['z_min'], xi_I_bounds['z_max'],
        xi_I_bounds['log10_Mcr_min'], xi_I_bounds['log10_Mcr_max'],
    )[0] 
    
    # Rescale with correct units
    n_factor *= (
        u.Mpc ** (-3 + 2) # volume of smbhb_density + dL
        * u.kyr # Delta tau_c 
    )

    return (prefactor * n_factor).to(u.dimensionless_unscaled)


def unormalizedZombieDensity(z: float, 
                             log10_Mcr: float, 
                             q: float,
                             pop_model: dict,
                             Ta_max: float, fth: float, 
                             xi_I_bounds: dict,
                             cosmo=Planck18) -> float:
    """
    Compute the un-normalized number density of zombie CWs at a given parameter space point.

    Args:
        z (float): Redshift of zombie
        log10_Mcr (float): Log10 chirp mass of the zombie (rest frame)
        pop_model (dict): Binary population model parameters
        Ta_max (float): Maximum time of flight of photon associated to more distant pulsar (observer frame)
        fth (float): Threshold orbital frequency for forb^E (observer frame)
        xi_I_bounds (dict): Bounds of integration of binaries considered
        cosmo (object, optional): Astropy cosmology object. Defaults to Planck18.

    Returns:
        float: _description_
    """
    
    ## 1 or 0 depending if z is in expected bounds
    z_cond = ((z >= xi_I_bounds['z_min']) & (z <= xi_I_bounds['z_max'])).astype(int)

    ## Same for Mcr
    log10_Mcr_cond = (
        (log10_Mcr >= xi_I_bounds['log10_Mcr_min']) & (log10_Mcr <= xi_I_bounds['log10_Mcr_max'])
    ).astype(int)

    ## Same for q
    q_cond = (
        np.abs(q) <= 1
    ).astype(int)

    
    return (
        z_cond * log10_Mcr_cond * q_cond *
        (
            smbhb_density(log10_M=log10_Mcr, z=z, **pop_model)
            * cosmo.luminosity_distance(z).to(u.Mpc).value ** 2 
            / (1 + z) ** 2
            * (
                2 * Ta_max - getDelta_tauc_min(log10_Mcr=log10_Mcr, z=z,
                                               fth=fth, q=q)
            ).to(u.kyr).value
        )
    )

@jax.jit()
def getValidPsrZombiePairs(Tc, 
                           Delta_taua, Delta_tauc, 
                           forb_ISCO):
    """
    Compute the log10 of the GW frequency for the pulsar-zombie pairs.
    If the zombie has already merged at the epoch of the pulsar term, it is set to 20.
    The function also returns the number of pulsars contributing to the zombie detection.

    Args:
        Tc (_type_): _description_
        Delta_taua (_type_): _description_
        Delta_tauc (_type_): _description_
        forb_ISCO (_type_): _description_

    Returns:
        _type_: _description_
    """
    forbP_s = smbhbs.getforbPa_zombie_cache(
                                Tc=Tc,
                                Delta_taua=Delta_taua,  # (N_zombies, N_psr)
                                Delta_tauc=Delta_tauc,  # (N_zombies, 1)
                                forb_ISCO=forb_ISCO)  # scalar
    
    # Identify valid (zombie, pulsar) pairs where forbP_s is not NaN (i.e., binary has not merged for that pulsar)
    valid_pairs = ~jnp.isnan(forbP_s)  # (N_zombies, N_psr)

    # Compute GW frequencies in log space (already 1e20 for merged binaries)
    log10_fgwP_s = jnp.where(valid_pairs,
                             jnp.log10(2.0 * forbP_s),
                             20.0 # dummy value for merged binaries, residuals will be zero anyway
    )  # (N_zombies, N_psr)

    return jnp.sum(valid_pairs, axis=1), log10_fgwP_s
            
@functools.partial(jax.jit, static_argnames=('N_zombies', 'chunk_size', 'N_psr'))
def computeZombieSNRs_chunk_cache_jax_clean(args,
                                            cache_args,
                                            N_zombies,
                                            chunk_size,
                                            N_psr):
    """
    Compute SNRs for zombies in chunks using cached parameters to reduce memory usage.
    
    This implementation reuses cached extrinsic parameters (position, antenna patterns, etc.)
    that are independent of the (z, log_Mcr) grid cell, only recomputing frequency-dependent
    quantities (forbP_s, log10_fgwP_s) for the specific grid cell.
    
    Args:
        i_log10_Mcr (int): Index into log10_Mcr grid
        j_z (int): Index into z grid
        zb_cache (ZombieParameterCache): Cached zombie parameters
        N_zombies (int): Total number of zombies. Defaults to 10_000.
        chunk_size (int): Number of zombies to process per chunk. Defaults to 1000.
    
    Returns:
        np.ndarray: SNRs for all zombies, shape (N_zombies,)
    """
    # Process in chunks
    n_chunks = (N_zombies + chunk_size - 1) // chunk_size

    # ================================================================
    # Compute orbital frequencies at pulsar term for this (z, log_Mcr) cell
    # forbP_s: (N_zombies, N_psr)
    # ================================================================
    N_contrib, log10_fgwP_s =  getValidPsrZombiePairs(
                        Tc=args['Tc'],
                        Delta_taua=cache_args['Delta_tauas'],  # (N_zombies, N_psr)
                        Delta_tauc=args['Delta_taucs'],  # (N_zombies, 1)
                        forb_ISCO=args['forb_ISCOs'])

    # Pre-stack all chunk inputs along axis 0 — shapes: (n_chunks, chunk_size, ...)
    inputs = (
              cache_args['fplus_s'],
              cache_args['fcross_s'],
              cache_args['At_inc_prefs'],
              cache_args['Bt_inc_prefs'],
              log10_fgwP_s.reshape(n_chunks, chunk_size, N_psr),
              cache_args['phase_cs']
    )

    def process_chunk(chunk):
        """Compute SNRs for a single chunk of zombie candidates.

        Parameters
        ----------
        chunk : tuple
            Cached antenna-pattern and phase arrays for one chunk.

        Returns
        -------
        jax.Array
            SNR values for all zombies in the chunk.
        """
        fplus, fcross, At_inc, Bt_inc, log10_fgw, phase_c = chunk

        CW_residuals = cw.compute_residuals_cache_vmap(
            tp=cache_args['tps'],
            Gmc=args['Gmc'],
            worb_ISCO=args['worb_ISCO'],
            fplus=fplus,
            fcross=fcross,
            At_inc_pref=At_inc,
            Bt_inc_pref=Bt_inc,
            alpha_pref=args['alpha_pref'],
            log10_fgw=log10_fgw,
            phase_c=phase_c,
        )

        snrs = compute_all_zombies_snr_jax(
            cache_args['Sig_cf_s'],
            cache_args['FT_s'],
            cache_args['Ninv_s'],
            CW_residuals,
        )

        return jnp.nan_to_num(snrs, nan=0.0)

    # Output shape: (n_chunks, chunk_size) — then flatten
    chunk_snrs = jax.lax.map(process_chunk, inputs)
    
    return chunk_snrs.reshape(-1), N_contrib

def computeZombieSNRs_chunk_cache_jax(i_log10_Mcr: float,
                                      j_z: float,
                                      zb_cache: ZombieParameterCache,
                                      N_zombies: int = 10_000,
                                      chunk_size: int = 1_000):
    """
    Compute SNRs for zombies in chunks using cached parameters to reduce memory usage.
    
    This implementation reuses cached extrinsic parameters (position, antenna patterns, etc.)
    that are independent of the (z, log_Mcr) grid cell, only recomputing frequency-dependent
    quantities (forbP_s, log10_fgwP_s) for the specific grid cell.
    
    Args:
        i_log10_Mcr (int): Index into log10_Mcr grid
        j_z (int): Index into z grid
        zb_cache (ZombieParameterCache): Cached zombie parameters
        N_zombies (int): Total number of zombies. Defaults to 10_000.
        chunk_size (int): Number of zombies to process per chunk. Defaults to 1000.
    
    Returns:
        np.ndarray: SNRs for all zombies, shape (N_zombies,)
    """
    
    # Process in chunks
    n_chunks = (N_zombies + chunk_size - 1) // chunk_size

    # ================================================================
    # Compute orbital frequencies at pulsar term for this (z, log_Mcr) cell
    # forbP_s: (N_zombies, N_psr)
    # ================================================================
    N_contrib, log10_fgwP_s =  getValidPsrZombiePairs(
                        Tc=jnp.asarray(zb_cache.Tcs[i_log10_Mcr, j_z]),
                        Delta_taua=jnp.asarray(zb_cache.Delta_tauas),  # (N_zombies, N_psr)
                        Delta_tauc=jnp.asarray(zb_cache.Delta_taucs[:, i_log10_Mcr, j_z][:,None]),  # (N_zombies, 1)
                        forb_ISCO=jnp.asarray(zb_cache.forb_ISCOs[i_log10_Mcr, j_z])
    )
    
    # Force all scalars to JAX arrays before passing to JIT functions
    Gmc        = zb_cache.GMcs[i_log10_Mcr, j_z]
    worb_ISCO  = zb_cache.worb_ISCOs[i_log10_Mcr, j_z]
    alpha_pref = zb_cache.alpha_prefs[i_log10_Mcr, j_z]

    # Initialize array to store SNRs
    chunk_snrs_list = []
    
    for chunk_idx in range(n_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = start_idx + chunk_size  # Always exactly chunk_size (fixed shape)

        # Define slice once
        sl = slice(start_idx, end_idx)
        
        # Slice all zombie-dependent arrays for this chunk
        # Shapes after slicing: (chunk_size, ...)
        fplus_chunk = zb_cache.fplus_s[sl]              # (chunk_size, N_psr)
        fcross_chunk = zb_cache.fcross_s[sl]            # (chunk_size, N_psr)
        At_inc_chunk = zb_cache.At_inc_prefs[sl]        # (chunk_size,) — 1D, no mask needed
        Bt_inc_chunk = zb_cache.Bt_inc_prefs[sl]        # (chunk_size,) — 1D, no mask needed
        log10_fgw_chunk = log10_fgwP_s[sl]              # (chunk_size, N_psr)
        phase_c_chunk = zb_cache.phase_cs[sl]           # (chunk_size,) — 1D, no mask needed
        
        CW_residuals_chunk = cw.compute_residuals_cache_vmap(
                                        tp=zb_cache.tps,       # (N_psr, N_toas) — broadcast
                                        Gmc=Gmc,               # scalar — broadcast
                                        worb_ISCO=worb_ISCO,   # scalar — broadcast
                                        fplus=fplus_chunk,         # (chunk_size, N_psr) — vmapped
                                        fcross=fcross_chunk,       # (chunk_size, N_psr) — vmapped
                                        At_inc_pref=At_inc_chunk,  # (chunk_size,) — vmapped
                                        Bt_inc_pref=Bt_inc_chunk,  # (chunk_size,) — vmapped
                                        alpha_pref=alpha_pref,     # scalar — broadcast
                                        log10_fgw=log10_fgw_chunk, # (chunk_size, N_psr) — vmapped
                                        phase_c=phase_c_chunk      # (chunk_size,) — vmapped
                                    )
        
        # Store SNRs for this chunk
        snrs = compute_all_zombies_snr_jax(
            zb_cache.Sig_cf_s, 
            zb_cache.FT_s, 
            zb_cache.Ninv_s, 
            CW_residuals_chunk
        )
        snrs = jnp.where(jnp.isnan(snrs), 0.0, snrs)  # Replace NaN SNRs (from merged binaries) with 0
        chunk_snrs_list.append(snrs)

    return jnp.concatenate(chunk_snrs_list), N_contrib  # Each shape (N_zombies,)

def worker_cache(grid_point, ZombieCache):
        """Compute the detection efficiency and SNRs for a single grid cell.

        Parameters
        ----------
        grid_point : tuple
            Grid indices for the mass-redshift cell.
        ZombieCache : ZombieParameterCache
            Cached extrinsic and intrinsic zombie parameters.

        Returns
        -------
        tuple
            Grid indices, efficiency value, per-cell SNRs, and contributing-pulsar counts.
        """
        i, _, j, _ = grid_point

        # if log10_Mcr < 7.5:
        #     return i, j, 0., {}

        # Compute SNRs marginalizing over extrinsic parameters
        SNRs, N_contrib = computeZombieSNRs_chunk_cache_jax_clean(
                                args={
                                    'Tc': jnp.asarray(ZombieCache.Tcs[i, j]),
                                    'Delta_taucs': jnp.asarray(ZombieCache.Delta_taucs[:, i, j][:,None]),  # (N_zombies, N_psr)
                                    'forb_ISCOs': jnp.asarray(ZombieCache.forb_ISCOs[i, j]),
                                    'Gmc': jnp.asarray(ZombieCache.GMcs[i, j]),
                                    'worb_ISCO': jnp.asarray(ZombieCache.worb_ISCOs[i, j]),
                                    'alpha_pref': jnp.asarray(ZombieCache.alpha_prefs[i, j])
                                },
                                cache_args=ZombieCache.args,
                                N_psr=ZombieCache.N_psr,
                                N_zombies=ZombieCache.N_zombies,
                                chunk_size=ZombieCache.N_chunk
                            )
        
        # Rescale accounting for the thinning factor
        SNRs *= jnp.sqrt(ZombieCache.thin_factor)

        # Compute the detection efficiency
        eff_val = (
            jnp.sum(
                # Sufficient SNR    
                (SNRs > ZombieCache.SNR_thresh)
                # Sufficient number of contributing pulsars
              & (N_contrib >= ZombieCache.Npsr_min)
            ) / ZombieCache.N_zombies
        )
        
        return i, j, eff_val, SNRs, N_contrib

def compute_efficiency_2d_cached_jax(log10_Mcr_mids: np.ndarray,
                                              z_mids: np.ndarray,
                                              N_bin: int,
                                              SNR_thresh: float,
                                              PTA: dict,
                                              fth: float,
                                              N_zombies_per_bin: int = 10_000,
                                              N_chunk: int = 1_000,
                                              n_jobs: int = -1,
                                              thin_factor: int = 1,
                                              seed: int = 42) -> tuple[dict, np.ndarray]:
    """
    Compute the 2D efficiency function P(SNR > SNR_thresh | log10_Mcr, z).
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

    # Initialize the Zombies cache object that generates once the random parameters for all grid points and stores them in memory
    ZombieCache = ZombieParameterCache(log10_Mcr_mids=log10_Mcr_mids,
                                       z_mids=z_mids,
                                       PTA=PTA,
                                       fth=fth,
                                       N_zombies=N_zombies_per_bin,
                                       N_chunk=N_chunk,
                                       SNR_thresh=SNR_thresh,
                                       thin_factor=thin_factor,
                                       seed=seed,
                                       q=1.0 # NOTE: we neglect dependence over q here
                            )

    # Initialize efficiency grid
    efficiency = np.zeros((N_bin, N_bin))

    # Initialize a dictionary to store SNRs for each grid point
    SNRs_dict = {}

    # Worker function (defined as nested function; threads don't need pickling)
    for i, log10_Mcr, j, z in tqdm(grid_points):

        _, _, efficiency[i,j], SNRs = worker_cache(
                                        (i, log10_Mcr, j, z),
                                        ZombieCache=ZombieCache)

        if efficiency[i,j] > 0:
            # Only store SNRs if there are any detections
            SNRs_dict[(i, j)] = SNRs

    return SNRs_dict, efficiency

def compute_efficiency_2d_cached_jax_threaded(log10_Mcr_mids: np.ndarray,
                                              z_mids: np.ndarray,
                                              N_bin: int,
                                              SNR_thresh: float,
                                              PTA: dict,
                                              fth: float,
                                              N_zombies_per_bin: int = 10_000,
                                              N_chunk: int = 1_000,
                                              n_jobs: int = -1,
                                              thin_factor: int = 1,
                                              seed: int = 42,
                                              Npsr_min: int = 2,
                                              outdir: str = './') -> tuple[np.ndarray, dict, dict]:
    """
    Compute the 2D efficiency function P(SNR > SNR_thresh | log10_Mcr, z).
    
    Intermediate SNR and N_contrib arrays are stored to temporary files to minimize
    memory usage during computation.
    """

    # Resolve number of workers
    if n_jobs == -1:
        max_workers = os.cpu_count() or 1
    else:
        max_workers = max(1, int(n_jobs))

    print(max_workers, "CPUs used for parallelisation.", flush=True)

    # Create temporary directory for storing intermediate results
    temp_dir = tempfile.mkdtemp(prefix='zombie_snr_', dir=outdir)
    print(f"Using temporary directory: {temp_dir}", flush=True)

    try:
        # Prepare grid points
        grid_points = [
            (i, log10_Mcr, j, z)
            for i, log10_Mcr in enumerate(log10_Mcr_mids)
            for j, z in enumerate(z_mids)
        ]

        # Initialize the Zombies cache object that generates once the random parameters for all grid points and stores them in memory
        ZombieCache = ZombieParameterCache(log10_Mcr_mids=log10_Mcr_mids,
                                           z_mids=z_mids,
                                           PTA=PTA,
                                           fth=fth,
                                           N_zombies=N_zombies_per_bin,
                                           N_chunk=N_chunk,
                                           SNR_thresh=SNR_thresh,
                                           thin_factor=thin_factor,
                                           seed=seed,
                                           q=1.0, # NOTE: we neglect dependence over q here
                                           Npsr_min=Npsr_min
                                )

        # Initialize efficiency grid
        efficiency = np.zeros((N_bin, N_bin))
        
        # Run threaded map with progress bar
        # Leaving headroom for JAX's own internal threading
        with ThreadPoolExecutor(max_workers=max(1, max_workers // 2)) as exc:
            futures = [exc.submit(worker_cache, gp, ZombieCache) for gp in grid_points]

            for fut in tqdm(as_completed(futures), total=len(futures)):
                i, j, eff_val, bin_SNRs, N_contrib = fut.result()
                efficiency[i, j] = eff_val
                
                if eff_val > 0:
                    # Save SNRs and N_contrib to temporary files instead of memory
                    snr_file = os.path.join(temp_dir, f'snr_{i}_{j}.npy')
                    ncontrib_file = os.path.join(temp_dir, f'ncontrib_{i}_{j}.npy')
                    np.save(snr_file, bin_SNRs)
                    np.save(ncontrib_file, N_contrib)

        # Reconstruct dictionaries from temporary files
        SNRs_dict = {}
        N_contrib_dict = {}
        
        for fname in os.listdir(temp_dir):
            if fname.startswith('snr_') and fname.endswith('.npy'):
                # Parse i, j from filename (format: snr_i_j.npy)
                parts = fname[4:-4].split('_')  # Remove 'snr_' prefix and '.npy' suffix
                i, j = int(parts[0]), int(parts[1])
                snr_file = os.path.join(temp_dir, fname)
                ncontrib_file = os.path.join(temp_dir, f'ncontrib_{i}_{j}.npy')
                
                SNRs_dict[(i, j)] = np.load(snr_file)
                N_contrib_dict[(i, j)] = np.load(ncontrib_file)

    finally:
        # Clean up temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"Cleaned up temporary directory: {temp_dir}", flush=True)

    return efficiency, SNRs_dict, N_contrib_dict

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


# ---- Numpy version  ---------------

def generate_ZombiesPars(
    z: np.ndarray,
    log10_Mcr: np.ndarray,
    q: np.ndarray,
    PTA: dict,
    fth: float,
    N_zombies: int = 10_000,
    rng=None,
):
    """
    
    Args:
        z            : Redshift, shape (N_zombies,) or scalar.
        log10_Mcr    : Log10 rest-frame chirp mass [M_sun], same shape.
        q            : Mass ratio, same shape.
        PTA          : Dict with keys:
                         'Ta_max'     – max PTA time (astropy Quantity, seconds)
                         'psr_pos_s'  – pulsar unit vectors (Npsr, 3)
                         'psr_T_s'    – pulsar light-travel times (Npsr,) [s]
                         'psr_toas'   – ToAs (Npsr, N_ToA) [s]
                         'Npsr'       – number of pulsars (int)
                         'N_ToA'      – number of ToAs per pulsar (int)
                         't0'         – reference time [s]
        fth          : GW frequency threshold [Hz].
        N_zombies    : Number of zombie binaries to draw.
        rng          : numpy RNG (only used for the initial random draw, before jit).

    Returns:
        CW_residuals : jnp.ndarray, shape (N_zombies, Npsr, N_ToA)
    """
    if rng is None:
        rng = np.random.default_rng(seed=42)

    # ------------------------------------------------------------------ #
    #  1.  Intrinsic parameters — broadcast to (N_zombies,)               #
    # ------------------------------------------------------------------ #
    z          = np.broadcast_to(np.atleast_1d(z),          (N_zombies,))
    log10_Mcr  = np.broadcast_to(np.atleast_1d(log10_Mcr), (N_zombies,))
    q          = np.broadcast_to(np.atleast_1d(q),          (N_zombies,))

    log10_dL   = np.log10(
        Planck18.luminosity_distance(z).to(u.Mpc).value
    )

    log10_Mc   = log10_Mcr + np.log10(1.0 + z)           # observer-frame
    Mc         = 10.0 ** log10_Mc * const.M_sun

    # ------------------------------------------------------------------ #
    #  2.  Draw coalescence times (NumPy, outside jit boundary)          #
    # ------------------------------------------------------------------ #
    Delta_tauc_min = getDelta_tauc_min(
        log10_Mcr=log10_Mcr,
        z=z,
        q=q,
        fth=fth,
    ).to(u.yr).value                                        # (N_zombies,)

    Ta_max_s = PTA['Ta_max'].to(u.yr).value
    Delta_tauc = np.array(
        rng.uniform(Delta_tauc_min, 2.0 * Ta_max_s, 
                    size=N_zombies)
    ) * u.yr               
    
    # ------------------------------------------------------------------ #
    #  3.  Draw extrinsic parameters                                      #
    # ------------------------------------------------------------------ #
    cos_theta = rng.uniform(-1,        1, size=N_zombies)
    phi       = rng.uniform( 0,  2*np.pi, size=N_zombies)
    psi       = rng.uniform( 0,    np.pi, size=N_zombies)
    cos_inc   = rng.uniform(-1,        1, size=N_zombies)
    phase_c   = rng.uniform( 0,  2*np.pi, size=N_zombies)

    binary_pos = utils.getUnitPos(cos_theta, phi).T  # (N_zombies, 3)

    # ------------------------------------------------------------------ #
    #  4.  Pulsar-term orbital frequencies  (N_zombies, Npsr)             #
    # ------------------------------------------------------------------ #

    # forbP_s: (N_zombies, Npsr) — NaN where binary already merged
    forbP_s = smbhbs.getforbPa_zombie(
        Mc=Mc[:, None],                    # (N_zombies, 1)
        binary_pos=binary_pos[:, None],    # (N_zombies, 1, 3)
        psr_pos=PTA['psr_pos_s'][None, :],        # (1, Npsr, 3)
        psr_T=PTA['psr_T_s'][None, :],            # (1, Npsr)
        Delta_tauc=Delta_tauc[:, None],    # (N_zombies, 1)
        q=q[:, None],                      # (N_zombies, 1)
    )                                      # → (N_zombies, Npsr)

    PT_valid_pair = ~np.isnan(forbP_s)                        # (N_zombies, Npsr)
    log10_fgwP_s  = np.where(
        PT_valid_pair,
        np.log10(2.0 * forbP_s.to(u.Hz).value),  # Hz
        np.nan, # mark bad pairs with nan
    ) # NOTE: same as numpy

    return (log10_Mc, q, log10_dL, log10_fgwP_s, 
            cos_theta, phi, cos_inc, phase_c, psi, 
            Delta_tauc, PT_valid_pair)

def computeZombieResiduals(z: float,
                           log10_Mcr: float,
                           q: float,
                           PTA: dict,
                           fth: float,
                           N_zombies: float = 10_000,
                           evolve: bool = True,
                           get_pars: bool = False,
                           rng=None):
    """Generate and evaluate timing residuals for a batch of zombie binaries.

    Parameters
    ----------
    z : float
        Redshift of the zombie population.
    log10_Mcr : float
        Rest-frame chirp mass in log10(M_sun).
    q : float
        Mass ratio.
    PTA : dict
        PTA configuration and precomputed matrices.
    fth : float
        Earth-term detection threshold frequency.
    N_zombies : float, optional
        Number of zombie realizations to sample.
    evolve : bool, optional
        If True, include GW frequency evolution in the residual model.
    get_pars : bool, optional
        If True, return the sampled intrinsic parameters alongside residuals.
    rng : numpy.random.Generator, optional
        Random number generator used for sampling.

    Returns
    -------
    ndarray or tuple
        Residuals for the zombie population, optionally with sampled parameters.
    """
    (log10_Mc, q, log10_dL, log10_fgwP_s, 
     cos_theta, phi, cos_inc, phase_c, psi, 
     Delta_tauc, PT_valid_pair) = generate_ZombiesPars(z=z,
                        log10_Mcr=log10_Mcr,
                        q=q,
                        PTA=PTA,
                        fth=fth,
                        N_zombies=N_zombies,
                        rng=rng,
                    )
    
    ### Compute the CW residuals for each zombie and each pulsar of the PTA
    # NOTE: this will only work if SAME number of toas among pulsars
    CW_residuals = np.zeros((N_zombies, PTA["Npsr"], PTA['N_ToA']),
                           dtype=float)
    
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
        ).to(u.s).value # NOTE: no units for jax

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

def computeZombieSNRs_jax(z: float,
                          log10_Mcr: float,
                          q: float,
                          PTA: dict,
                          fth: float,
                          N_zombies: float = 10_000,
                          rng=None):
    """Compute total SNRs for a sample of zombies using the JAX PTA model.

    Parameters
    ----------
    z : float
        Redshift of the zombie population.
    log10_Mcr : float
        Rest-frame chirp mass in log10(M_sun).
    q : float
        Mass ratio.
    PTA : dict
        PTA configuration and precomputed matrices.
    fth : float
        Earth-term detection threshold frequency.
    N_zombies : float, optional
        Number of zombie realizations to sample.
    rng : numpy.random.Generator, optional
        Random number generator.

    Returns
    -------
    jax.Array
        Total SNR for each sampled zombie.
    """
    
    # This is also very demanding in terms of memory, should change that for larger PTA datasets
    CW_residuals = computeZombieResiduals(z=z,
                                          log10_Mcr=log10_Mcr,
                                          q=q,
                                          PTA=PTA,
                                          fth=fth,
                                          N_zombies=N_zombies,
                                          rng=rng)
    
    return compute_all_zombies_snr_jax(PTA['Sig_cf_s'], PTA['FT_s'], PTA['Ninv_s'], CW_residuals)

def sampleZombiePars_jax(z: float,
                         log10_Mcr: float,
                         q: float,
                         PTA: dict,
                         fth: float,
                         N_zombies: float = 10_000,
                         rng=None):
    """Sample zombie parameters and return per-pulsar SNR-squared values.

    Parameters
    ----------
    z : float
        Redshift of the zombie population.
    log10_Mcr : float
        Rest-frame chirp mass in log10(M_sun).
    q : float
        Binary mass ratio.
    PTA : dict
        PTA configuration and precomputed matrices.
    fth : float
        Earth-term detection threshold frequency.
    N_zombies : float, optional
        Number of zombie realizations to sample.
    rng : numpy.random.Generator, optional
        Random number generator.

    Returns
    -------
    tuple
        Per-pulsar SNR-squared values and the sampled parameter dictionary.
    """
    
    CW_residuals, pars = computeZombieResiduals(z=z,
                                          log10_Mcr=log10_Mcr,
                                          q=q,
                                          PTA=PTA,
                                          fth=fth,
                                          N_zombies=N_zombies,
                                          get_pars=True,
                                          rng=rng)
    
    snr2_per_psrs = compute_all_zombies_snr_per_psr_jax(PTA['Sig_cf_s'], PTA['FT_s'], PTA['Ninv_s'], CW_residuals)
    
    return snr2_per_psrs, pars