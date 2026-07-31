from collections.abc import Callable

SevenFloatFn = Callable[[float, float, float, float, float, float, float], float]

import numpy as np
import scipy.integrate as integrate
from astropy import constants as const
from astropy import units as u
from astropy.cosmology import Planck18

from utils import get_mids


def smbhb_density(log10_M: float, z: float, alpha: float, log10_M_star: float,
                  beta: float, z0: float, n0_dot: float, **kwargs) -> float:
    """Return the comoving SMBHB number density as a function of chirp mass and redshift.

    Parameters
    ----------
    log10_M : float
        Rest-frame chirp mass in solar masses, expressed as log10(M_c / M_sun).
    z : float
        Redshift of the binary population element.
    alpha : float
        Slope of the SMBHB mass function.
    log10_M_star : float
        Characteristic mass scale of the mass function.
    beta : float
        Redshift evolution slope.
    z0 : float
        Characteristic redshift of the population model.
    n0_dot : float
        Normalization of the comoving number density.

    Returns
    -------
    float
        Number density per unit comoving volume, per unit logarithmic chirp mass,
        and per unit redshift.
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
    """Return the redshift-integrated merger density for a chirp-mass grid.

    Parameters
    ----------
    log10_Mcr : array-like
        Chirp masses in solar masses, expressed as log10(M_c / M_sun).
    pop_model : dict
        SMBHB population parameters used by smbhb_density.
    z_min : float
        Lower redshift bound of the integration.
    z_max : float
        Upper redshift bound of the integration.

    Returns
    -------
    ndarray
        Merger density integrated over the specified redshift interval.
    """

    log10_Mcr_s = np.atleast_1d(log10_Mcr)

    arr = np.zeros(log10_Mcr_s.shape[0])

    for k, log10_Mcr in enumerate(log10_Mcr_s):
        def integrand_num(z):
            """Return the integrand for the redshift integral at fixed chirp mass.

            Parameters
            ----------
            z : float
                Redshift.

            Returns
            -------
            float
                SMBHB density at the selected chirp mass and redshift.
            """
            return smbhb_density(log10_Mcr, z, **pop_model)
        
        arr[k] = integrate.quad(integrand_num,
                                z_min, z_max)[0]
    
    return arr

def getdndz(z, pop_model, 
            log10_Mcr_min, log10_Mcr_max):
    """Return the chirp-mass-integrated merger density for a redshift grid.

    Parameters
    ----------
    z : array-like
        Redshift values.
    pop_model : dict
        SMBHB population parameters used by smbhb_density.
    log10_Mcr_min : float
        Lower bound of the chirp-mass integration range.
    log10_Mcr_max : float
        Upper bound of the chirp-mass integration range.

    Returns
    -------
    ndarray
        Merger density integrated over the specified chirp-mass interval.
    """

    z_s = np.atleast_1d(z)

    arr = np.zeros(z_s.shape[0])

    for k, z in enumerate(z_s):
        def integrand_num(log10_Mcr):
            """Return the integrand for the chirp-mass integral at fixed redshift.

            Parameters
            ----------
            log10_Mcr : float
                Chirp mass in log10(M_sun).

            Returns
            -------
            float
                SMBHB density at the selected mass and redshift.
            """
            return smbhb_density(log10_Mcr, z, **pop_model)
        
        arr[k] = integrate.quad(integrand_num,
                                log10_Mcr_min, log10_Mcr_max)[0]
    
    return arr

def analytic_hc2(f: float, 
                 smbhb_density: SevenFloatFn, 
                 model: dict, 
                 xi_I_bounds: dict,
                 Mcr_scaling: float = 1e8) -> float:
    """Return the analytic characteristic strain contribution from the SMBHB population.

    Parameters
    ----------
    f : float
        Gravitational-wave frequency in Hz.
    smbhb_density : callable
        SMBHB number-density model as a function of chirp mass and redshift.
    model : dict
        Population parameters passed to the density model.
    xi_I_bounds : dict
        Integration bounds for chirp mass and redshift.
    Mcr_scaling : float, optional
        Mass scaling used to improve numerical stability during integration.

    Returns
    -------
    float
        Characteristic strain amplitude squared, h_c^2(f), for the population.
    """
    
    # Define the (1+z)^{-1/3} (GMc,r)^{5/3} integrand
    def integrand_num(log10_Mcr: float, z: float) -> float:
        """Return the mass-redshift integrand entering the characteristic-strain calculation.

        Parameters
        ----------
        log10_Mcr : float
            Chirp mass in log10(M_sun).
        z : float
            Redshift.

        Returns
        -------
        float
            Population contribution to the h_c^2 integral at the specified point.
        """
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