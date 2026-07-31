import astropy.constants as const
import astropy.units as u
import jax.numpy as jnp
import numpy as np
from astropy.cosmology import Planck18

import CW_signal as cw
import smbhbs
import utils
import zombie_rate as rate


class ZombieParameterCache:
    def __init__(self, 
                 log10_Mcr_mids, z_mids, 
                 PTA, fth,
                 N_zombies, N_chunk, 
                 SNR_thresh,
                 thin_factor,
                 seed,
                 q=1.0,
                 Npsr_min=2,
                 cosmo=Planck18):
        """Precompute and cache zombie signal parameters for a mass-redshift grid.

        Parameters
        ----------
        log10_Mcr_mids : array-like
            Chirp-mass grid in log10(M_c / M_sun).
        z_mids : array-like
            Redshift grid values.
        PTA : dict
            PTA configuration and precomputed matrices.
        fth : float
            Minimum earth-term orbital frequency.
        N_zombies : int
            Number of zombie realizations per grid cell.
        N_chunk : int
            Chunk size used for batched SNR evaluation.
        SNR_thresh : float
            Detection threshold for the SNR criterion.
        thin_factor : int
            Time-thinning factor applied to the PTA data.
        seed : int
            Random seed for reproducible parameter generation.
        q : float, optional
            Binary mass ratio used in the cache.
        Npsr_min : int, optional
            Minimum number of contributing pulsars for a detection.
        cosmo : astropy.cosmology.Cosmology, optional
            Cosmology model used for distance calculations.
        """
        
        # Store number of zombies, chunk size, SNR threshold and thin factor
        self.N_zombies = N_zombies
        self.N_chunk = N_chunk
        self.SNR_thresh = SNR_thresh
        self.thin_factor = thin_factor

        # Store minimum number of pulsars that must contribute to the zombie SNR
        self.Npsr_min = Npsr_min

        # Initialize random number generator with the provided seed
        self.rng = np.random.default_rng(seed)

        # Store the pulsars positions
        self.psr_pos_s = jnp.asarray(PTA['psr_pos_s'])

        # Store number of pulsars and toas
        self.N_psr = int(len(self.psr_pos_s))
        self.N_toas = PTA['psr_toas'].shape[1]

        # Store the useful quantities for SNR computations
        self.Sig_cf_s = jnp.asarray(PTA['Sig_cf_s'])
        self.FT_s = jnp.asarray(PTA['FT_s'])
        self.Ninv_s = jnp.asarray(PTA['Ninv_s'])

        # Generate and cache extrinsic parameters for the zombies
        self.generate_extrinsic_parameters()

        # Store the log_Mcr and z grids
        self.log10_Mcr_mids = log10_Mcr_mids
        self.z_mids = z_mids

        # Store q
        self.q = q

        # Compute all luminosity distances
        dLs   = cosmo.luminosity_distance(self.z_mids).to(u.Mpc)
        dists       = dLs / const.c

        # Compute observer frame chirp masses
        self.log10_Mcs   = (self.log10_Mcr_mids[:,None] 
                            + np.log10(1.0 + self.z_mids[None,:])
        )    # observer-frame
        Mcs         = 10.0 ** self.log10_Mcs * const.M_sun  # kg
        Mtots       = smbhbs.getMtot(Mc=Mcs, q=q)  # kg
        GMcs         = const.G * Mcs / const.c ** 3 # s

        # Get associated ISCO frequencies
        forb_ISCOs = smbhbs.getforb_ISCO(Mtot=Mtots) # Hz
        worb_ISCOs = 2 * np.pi * forb_ISCOs # rad/s

        # Compute associated characteristic coalescence times
        Tcs = smbhbs.getCoalescenceTime(Mc=Mcs,
                                        forb=forb_ISCOs) # s

        ### Draw coalescence times for the zombies
        Delta_taucs = self.generateCoalescenceTimes(fth, Ta_max=PTA['Ta_max']) # yr

        # NOTE: forb_P would be too large to store in memory since also pulsar dependent
        # Will be computed on the fly in the worker functions

        # Compute the geometric delays associated to the pulsar term of each pulsar for each zombie
        # Shape: (N_zombies, N_psr)
        Delta_tauas = smbhbs.getETPT_delay(psr_T=PTA['psr_T_s'][None,:], # s
                                           psr_pos=self.psr_pos_s[None,:], 
                                           binary_pos=self.binary_poss[:, None, :]) # (N_zombies, N_psr)

        ### Compute quantities to speed up residuals computation
        # Retarded times relative to t0
        tps = (PTA['psr_toas'] - PTA['t0']) * u.s  # (N_psr, N_toas)

        # Antenna pattern responses accounting for polarization angle 
        # (shape: N_zombies x N_psr)
        self.fplus_s, self.fcross_s, _ = cw.jax_create_gw_antenna_pattern(
                                                pos=self.psr_pos_s[None,:],
                                                gwtheta=self.thetas[:,None],
                                                gwphi=self.phis[:,None],
                                                # To match cw_delay_zombie convention 
                                                psi=-1 * self.psis[:,None])
        
        # Useful cos_inc prefactors
        incs = jnp.arccos(self.cos_incs)
        self.At_inc_prefs = -0.5 * (3 + jnp.cos(2 * incs))
        self.Bt_inc_prefs = 2 * self.cos_incs

        # Residuals amplitude prefactor (shape: N_zombies)
        alpha_prefs = GMcs ** (5. / 3) / dists

        # Convert all quantities used in the code to convenient units (avoid 1e39 kg for example)
        self.Tcs        = Tcs.to(u.yr).value
        self.Delta_tauas = jnp.asarray(Delta_tauas.to(u.yr).value)
        self.Delta_taucs = Delta_taucs.to(u.yr).value
        
        self.tps        = jnp.asarray(tps.to(u.s).value)
        self.forb_ISCOs = forb_ISCOs.to(u.Hz).value
        self.GMcs       = GMcs.to(u.s).value
        self.worb_ISCOs = worb_ISCOs.to(u.Hz).value
        self.alpha_prefs = alpha_prefs.to(u.s ** (2/3)).value

        # Number of chunks
        nb_chunk = N_zombies // N_chunk
        assert N_zombies % N_chunk == 0, "N_zombies must be a multiple of N_chunk"

        self.args = {
            'Delta_tauas': self.Delta_tauas,
            'fplus_s': self.fplus_s.reshape(nb_chunk, N_chunk, self.N_psr),
            'fcross_s': self.fcross_s.reshape(nb_chunk, N_chunk, self.N_psr),
            'At_inc_prefs': self.At_inc_prefs.reshape(nb_chunk, N_chunk),
            'Bt_inc_prefs': self.Bt_inc_prefs.reshape(nb_chunk, N_chunk),
            'phase_cs': self.phase_cs.reshape(nb_chunk, N_chunk),
            'tps': self.tps,
            'Sig_cf_s': self.Sig_cf_s,
            'FT_s': self.FT_s,
            'Ninv_s': self.Ninv_s
        }


    def generate_extrinsic_parameters(self):
        """
        Generate extrinsic parameters for the zombies and store them in the cache.
         - cos_theta: cosine of the polar angle (uniform in [-1, 1])
         - phi: azimuthal angle (uniform in [0, 2pi])
         - psi: polarization angle (uniform in [0, pi])
         - cos_inc: cosine of the inclination angle (uniform in [-1, 1])
         - phase_c: initial phase at coalescence (uniform in [0, 2pi])
         - binary_pos: unit vector pointing to the binary (shape: N_zombies x 3)
        
        These parameters are generated once and stored in the cache to avoid redundant computations during the efficiency grid calculation.
        """

        self.cos_thetas = jnp.asarray(self.rng.uniform(-1,        1, size=self.N_zombies))
        self.thetas     = jnp.arccos(self.cos_thetas)
        self.phis       = jnp.asarray(self.rng.uniform( 0,  2*np.pi, size=self.N_zombies))

        self.binary_poss = utils.getUnitPos(self.cos_thetas, self.phis).T  # (N_zombies, 3)

        self.psis       = jnp.asarray(self.rng.uniform( 0,    np.pi, size=self.N_zombies))
        self.cos_incs   = jnp.asarray(self.rng.uniform(-1,        1, size=self.N_zombies))
        self.phase_cs   = jnp.asarray(self.rng.uniform( 0,  2*np.pi, size=self.N_zombies))

    def generateCoalescenceTimes(self, fth, Ta_max):
        """Draw coalescence times for all zombies over the mass-redshift grid.

        Parameters
        ----------
        fth : float
            Earth-term frequency threshold used to set the minimum coalescence delay.
        Ta_max : astropy.units.Quantity
            Maximum PTA time span used as the upper draw bound.

        Returns
        -------
        astropy.units.Quantity
            Coalescence-time offsets for each zombie and grid cell.
        """
        # Define the minimum tauc for each grid point
        Delta_tauc_mins = rate.getDelta_tauc_min(
            log10_Mcr=self.log10_Mcr_mids[:,None],
            z=self.z_mids[None,:],
            q=self.q,
            fth=fth,
        ).to(u.yr).value

        # Convert upper bound to years as well
        Ta_max = Ta_max.to(u.yr).value

        # Draw coalescence times uniformly between the minimum and 2*Ta_max
        # Size: (N_zombies, N_log10_Mcr, N_z)
        return (self.rng.uniform(
                    low=Delta_tauc_mins[None, :, :],
                    high=2.0 * Ta_max,
                    size=(self.N_zombies, *Delta_tauc_mins.shape))
                * u.yr # convert to years
        )
        