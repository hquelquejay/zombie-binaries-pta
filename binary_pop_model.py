from collections.abc import Callable

SevenFloatFn = Callable[[float, float, float, float, float, float, float], float]

import numpy as np
import scipy.integrate as integrate
from astropy import constants as const
from astropy import units as u
from astropy.cosmology import Planck18


def smbhb_density(log10_M: float, z: float, alpha: float, log10_M_star: float,
                  beta: float, z0: float, n0_dot: float, **kwargs) -> float:
    """
        Schechter-like SMBHB number density per unit comoving volume as a function of
        log10 chirp mass and redshift
        [Source: https://github.com/astrolamb/pop_synth/blob/main/scripts/pop_synth.py]

        Parameters
        ----------
        z : float
            Redshift
        log10_M : float
            log10 of the chirp mass
        alpha : float
            Slope of the mass function
        log10_M_star : float
            Characteristic mass of the mass function
        beta : float
            Slope of the redshift function
        z0 : float
            Characteristic redshift of the redshift function
        n0_dot : float
            Normalization of the number density in Mpc^{-3} / Gyr

        Returns
        -------
        float
            log10 SMBHB number density per unit log10_M and unit redshift
    """

    # transform number density normalisation to the right units
    n0_dot = n0_dot.to(1/(u.s * u.Mpc**3)).value

    # mass function - compute in base 10 then raise to the power of 10 to
    # avoid numerical issues
    log10M_dist = 10**-(alpha*(log10_M - 7.) +
                        10**(log10_M - log10_M_star) * np.log10(np.e))

    z_dist = (1+z)**beta * np.exp(-z/z0)  # redshift function

    # change in age of binary per unit redshift
    dt_dz = 1/((1+z) * Planck18.H(z).to('1 / s').value)

    return n0_dot * log10M_dist * z_dist * dt_dz

def getdndlogMcr(log10_Mcr, pop_model, 
                 z_min, z_max):
    """
    Compute the comoving merger density integrated over redshift
    """

    log10_Mcr_s = np.atleast_1d(log10_Mcr)

    arr = np.zeros(log10_Mcr_s.shape[0])

    for k, log10_Mcr in enumerate(log10_Mcr_s):
        def integrand_num(z):
            return smbhb_density(log10_Mcr, z, **pop_model)
        
        arr[k] = integrate.quad(integrand_num,
                                z_min, z_max)[0]
    
    return arr

def getdndz(z, pop_model, 
            log10_Mcr_min, log10_Mcr_max):
    """
    Compute the comoving merger density integrated over log chirp mass
    """

    z_s = np.atleast_1d(z)

    arr = np.zeros(z_s.shape[0])

    for k, z in enumerate(z_s):
        def integrand_num(log10_Mcr):
            return smbhb_density(log10_Mcr, z, **pop_model)
        
        arr[k] = integrate.quad(integrand_num,
                                log10_Mcr_min, log10_Mcr_max)[0]
    
    return arr

def analytic_hc2(f: float, 
                 smbhb_density: SevenFloatFn, 
                 model: dict, 
                 xi_I_bounds: dict,
                 Mcr_scaling: float = 1e8) -> float:
    """
    Compute the characteristic strain squared at a given frequency f for a given 
    SMBHB population model.

    Args:
        f (float): Observer frequency.
        smbhb_density (SevenFloatFn): Schechter-like comoving merger density function.
        model (dict): SMBHB population model parameters.
        xi_I_bounds (dict): Integral bounds for chirp mass and redshift.
        Mcr_scaling (float, optional): Numerical scaling factor. Defaults to 1e8.

    Returns:
        float: h_c²(f)
    """
    
    # Define the (1+z)^{-1/3} (GMc,r)^{5/3} integrand
    def integrand_num(log10_Mcr: float, z: float) -> float:
        Mcr = 10 ** log10_Mcr
        
        # Scale Mcr units to avoid numerical issues
        Mcr /= Mcr_scaling
        
        return (
            Mcr ** (5. / 3) 
            * (1 + z) ** (-1. / 3) 
            * smbhb_density(log10_Mcr, z, **model)
        )

    # Perform the double integral over log10_Mcr and z
    n_prefactor = integrate.dblquad(
        lambda z, log10_M: integrand_num(log10_M, z),
        xi_I_bounds['log10_Mcr_min'], xi_I_bounds['log10_Mcr_max'],
        xi_I_bounds['z_min'], xi_I_bounds['z_max']  
    )[0] 

    # Rescale to proper units
    n_prefactor *= (u.Mpc ** (-3) * (const.G * const.M_sun * Mcr_scaling) ** (5/3))

    # Make the dirac approximation
    f_factor = (np.pi * f) ** (-4/3)

    return (
        4 * np.pi / (3 * const.c ** 2) *
        n_prefactor *
        f_factor
    )