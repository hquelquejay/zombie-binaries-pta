import astropy.constants as const
import astropy.units as u
import numpy as np

c = const.c
G = const.G
M_sun = const.M_sun


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

def getETPT_delay(psr_T: float, 
                  psr_pos: np.ndarray, 
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
        np.ndarray: Orbital frequency of pulsar a pulsar term
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