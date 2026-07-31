import astropy.constants as const
import astropy.units as u
import jax
import jax.numpy as jnp
import numpy as np

from utils import G, G_val, M_sun_val, T_sun_val, c, c_val


def getMtot(Mc: float, q: float) -> float:
    """Compute the total mass of a binary from its chirp mass and mass ratio

    Args:
        Mc (float): Chirp mass
        q (float): Mass ratio (m2/m1)
    Returns:
        float: Total mass of the binary
    """
    return Mc * (1 + q) ** (6 / 5) / (q ** (3 / 5))

def getCoalescenceTime(Mc: float, forb: float) -> float:
    """Compute the characteristic coalescence time in the observer frame

    Args:
        Mc (float): Chirp mass in observer frame
        forb (float): Binary frequency in observer frame

    Returns:
        float: Coalescence time in seconds
    """
    return (
            5 / 256 *
            (2 * np.pi * forb) ** (- 8. / 3) *
            (G * Mc) ** (- 5. / 3)
            * c ** 5
    )

def getTrueCoalescenceTime(Mc: float, forb: float, q:float = 1) -> float:
    """Compute the coalescence time (up to the ISCO frequency) in the observer frame

    Args:
        Mc (float): Chirp mass in observer frame
        forb (float): Binary frequency in observer frame
        q (float): Mass ratio (m2/m1)

    Returns:
        float: Coalescence time in seconds
    """
    # Compute ISCO frequency
    forb_ISCO = getforb_ISCO(Mtot=getMtot(Mc=Mc, q=q))  # Assuming equal mass binaries
    
    return (
            5 / 256 *
            (G * Mc) ** (- 5. / 3)
            * c ** 5 *
            ( 
                (2 * np.pi * forb) ** (- 8. / 3) 
                - 
                (2 * np.pi * forb_ISCO) ** (- 8. / 3) 
            )
    )

def getforb_ISCO(Mtot: float) -> float:
    """
    Compute the approximate ISCO frequency of a binary

    Args:
        Mtot (float): Total mass of the binary
    Returns:
        float: ISCO frequency
    """
    return (
        const.c ** 3 / (2 * np.pi * 6 ** (3 / 2)) 
        / (const.G * Mtot)
    )


def getETPT_delay(psr_T: float, psr_pos: np.ndarray, 
                  binary_pos: np.ndarray) -> float:
    """
    Compute the time delay between the Earth and the Pulsar term GW wavefronts. 

    Args:
        psr_T (float): Time of flight of photon to the pulsar
        psr_pos (np.ndarray): Unit position vector of the pulsar
        binary_pos (np.ndarray): Unit position vector of the binary

    Returns:
        float: Time delay
    """

    cos_thetas = np.sum(psr_pos * binary_pos, axis=-1)

    return (
        psr_T * (1 - cos_thetas)
    )

def getforbPa_zombie(Mc: float, binary_pos: np.ndarray, 
                     psr_pos: np.ndarray, psr_T: float,
                     Delta_tauc: float,
                     q: float = 1) -> np.ndarray:
    """
    Compute the orbital frequency at the pulsar term of pulsar a located at p = c / psr_T * psr_pos
    of a binary located at binary_pos.

    Args:
        Mc (float): Chirp mass of the binary (observer frame)
        binary_pos (np.ndarray): Unit position vector of the binary
        psr_pos (np.ndarray): Unit position vector of the pulsar
        psr_T (float): Time of flight of photon to the pulsar
        Delta_tauc (float): Delay between coalescence and starting time of PTA
        q (float): Mass ratio of the binary

    Returns:
        array: Orbital frequency of pulsar a pulsar term
    """

    # Compute total mass of the binary
    Mtot = getMtot(Mc=Mc, q=q)

    # Compute ISCO frequency
    forb_ISCO = getforb_ISCO(Mtot=Mtot)

    # Compute characteristic coalescence time
    Tc = getCoalescenceTime(Mc=Mc, forb=forb_ISCO)

    # Compute ET-PT delay
    Delta_taua = getETPT_delay(psr_T=psr_T, psr_pos=psr_pos, binary_pos=binary_pos)

    return np.where(Delta_taua > Delta_tauc, 
        forb_ISCO * (
            1 + ((Delta_taua - Delta_tauc) / Tc).to(u.dimensionless_unscaled)
        ) ** (-3 / 8),
        np.nan # If the binary has already merged also at the pulsar term
    )

def getMaxGamma(psr_D,
                Mc, forb):
    """Return the maximum pulsar-term growth factor for a binary at a given distance.

    Parameters
    ----------
    psr_D : astropy.units.Quantity
        Distance to the pulsar.
    Mc : float
        Chirp mass of the binary.
    forb : float
        Orbital frequency of the binary.

    Returns
    -------
    astropy.units.Quantity
        Maximum GW amplitude factor for the pulsar term.
    """

    PT_delay = (2 * psr_D / c).to(u.s)

    # Get binary characteristic coalescence time
    Tc = getCoalescenceTime(Mc, forb).to(u.s)

    return (
        (1 + PT_delay / Tc) ** (-3. / 8)
    )

def getGamma(psr_pos, psr_D,
             cos_theta, phi,
             Mc, forb):
    """Return the geometric pulsar-term factor for a binary sky position.

    Parameters
    ----------
    psr_pos : np.ndarray
        Unit vector of the pulsar position.
    psr_D : astropy.units.Quantity
        Distance to the pulsar.
    cos_theta : float
        Cosine of the binary polar angle.
    phi : float
        Binary azimuthal angle.
    Mc : float
        Chirp mass of the binary.
    forb : float
        Orbital frequency of the binary.

    Returns
    -------
    float
        Geometric factor encoding the pulsar-term delay and amplitude.
    """
    
    sin_theta = np.sin(np.arccos(cos_theta))
    
    # Compute Pulsar term delay
    dot_prod = (
         psr_pos[0] * sin_theta * np.cos(phi) +
         psr_pos[1] * sin_theta * np.sin(phi) +
         psr_pos[2] * cos_theta
    )

    PT_delay = (1 - dot_prod) * psr_D / c

    # Get binary characteristic coalescence time
    Tc = getCoalescenceTime(Mc, forb)

    return (
        (1 + PT_delay / Tc) ** (-3. / 8)
    )

# ---- JAX version ------------

@jax.jit
def getMtot_jax(Mc, q):
        """Compute total mass from chirp mass and mass ratio"""        
        return Mc * jnp.power(1. + q, 1.2) / jnp.power(q, 0.6)

@jax.jit
def getCoalescenceTime_jax(Mc, forb):
        """Compute characteristic coalescence time"""
        return (5 / 256 * 
                (2 * jnp.pi * forb) ** (-8. / 3) * 
                (Mc / M_sun_val * T_sun_val) ** (-5. / 3)
        )

@jax.jit
def getforb_ISCO_jax(Mtot):
        """Compute ISCO frequency from total mass"""
        return c_val ** 3 / (2 * jnp.pi * 6 ** (3 / 2)) / (G_val * Mtot)

@jax.jit
def getETPT_delay_jax(psr_T, psr_pos, binary_pos):
    """Compute ET-PT time delay"""

    cos_thetas = jnp.sum(psr_pos * binary_pos, axis=-1)
    
    return psr_T * (1 - cos_thetas)

@jax.jit
def getforbPa_zombie_cache(Tc, 
                           Delta_taua,
                           Delta_tauc,
                           forb_ISCO):
    """
    Compute the orbital frequency at the pulsar term of pulsar a located at p = c / psr_T * psr_pos
    of a binary located at binary_pos.

    Args:
        Mc (float): Chirp mass of the binary (observer frame)
        binary_pos (np.ndarray): Unit position vector of the binary
        psr_pos (np.ndarray): Unit position vector of the pulsar
        psr_T (float): Time of flight of photon to the pulsar
        Delta_tauc (float): Delay between coalescence and starting time of PTA
        q (float): Mass ratio of the binary

    Returns:
        array: Orbital frequency of pulsar a pulsar term
    """

    return jnp.where(Delta_taua > Delta_tauc, 
        forb_ISCO * (
            1 + (Delta_taua - Delta_tauc) / Tc
        ) ** (-3 / 8),
        jnp.nan # If the binary has already merged also at the pulsar term
    )

@jax.jit
def getforbPa_zombie_jax(Mc: jax.Array, 
                         binary_pos: jax.Array,
                         psr_pos: jax.Array, 
                         psr_T: jax.Array,
                         Delta_tauc: jax.Array,
                         q: jax.Array) -> jax.Array:
    """
    JAX-compatible version: Compute the orbital frequency at the pulsar term of pulsar a.
    
    This computes the orbital frequency at the pulsar term for a zombie binary system.
    All inputs are assumed to be in SI units (mass in kg, time in seconds).
    
    Args:
        Mc (jax.Array): Chirp mass of the binary (observer frame) in kg
        binary_pos (jax.Array): Unit position vector of the binary (shape: (3,) or (..., 3))
        psr_pos (jax.Array): Unit position vector of the pulsar (shape: (3,) or (..., 3))
        psr_T (jax.Array): Time of flight of photon to the pulsar in seconds
        Delta_tauc (jax.Array): Delay between coalescence and starting time of PTA in seconds
        q (jax.Array): Mass ratio of the binary (default: 1.0)
    
    Returns:
        jax.Array: Orbital frequency of pulsar a at pulsar term in Hz
    """
    
    # Compute total mass of the binary
    Mtot = getMtot_jax(Mc, q)
    
    # Compute ISCO frequency
    forb_ISCO = getforb_ISCO_jax(Mtot)
    
    # Compute characteristic coalescence time
    Tc = getCoalescenceTime_jax(Mc, forb_ISCO)
    
    # Compute ET-PT delay
    Delta_taua = getETPT_delay_jax(psr_T, psr_pos, binary_pos)
    
    # Return orbital frequency at pulsar term
    # If binary has already merged, return NaN (no pulsar term contribution)
    safe_arg = jnp.where(Delta_taua > Delta_tauc,
                         (Delta_taua - Delta_tauc) / Tc,
                         jnp.zeros_like(Tc))  # safe dummy, gradient is 0

    forbP = forb_ISCO * jnp.power(1 + safe_arg, -3.0/8.0)

    return jnp.where(Delta_taua > Delta_tauc, 
                     forbP,
                     jnp.nan)