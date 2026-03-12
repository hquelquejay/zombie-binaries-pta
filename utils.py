import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import yaml


def get_mids(edges):
    """ Compute bin midpoints from edges """

    return (edges[:-1] + edges[1:]) / 2

def getUnitPos(cos_theta: float, phi: float) -> np.ndarray:
    """
    Compute the unit vector pointing toward a sky position

    Args:
        cos_theta (float): Cosine of the polar angle
        phi (float): Azimuthal angle

    Returns:
        np.ndarray: Unit vector
    """

    sin_theta = np.sin(np.arccos(cos_theta))

    return np.array([sin_theta * np.cos(phi),
                     sin_theta * np.sin(phi),
                     cos_theta
                     ])

def sample_from_grid(mesh, unn_density,
                     N_samples=20000,
                     rng=None):
    """
    Generic grid-based sampler for unnormalized densities.

    Args:
        mesh (list or tuple of np.ndarray): Meshgrid arrays defining the grid points in each dimension.
        unn_density (np.ndarray): Unnormalized probability density on the grid, matching the shape of the mesh.
        N_samples (int, optional): Number of samples to draw. Defaults to 20000.
        rng (np.random.Generator, optional): Random number generator. If None, uses np.random.default_rng(seed=42).

    Returns:
        np.ndarray: Array of sampled points with shape (N_samples, ndim), where ndim is the number of dimensions.
    """

    if rng is None:
        rng = np.random.default_rng(seed=42)

    # Flatten density
    unn_density_flat = unn_density.ravel()

    # Normalize the probabilities
    p = unn_density_flat / np.sum(unn_density_flat)

    # Sample
    pts = np.stack([m.ravel() for m in mesh], axis=1)
    idx = rng.choice(len(p), size=N_samples, replace=True, p=p)

    return pts[idx]

def getAvgOverValids(all_arr: np.ndarray, 
                     N_contrib_psrs: np.ndarray, 
                     PT_valid_pair: np.ndarray) -> np.ndarray:
    """
    Compute the average of a quantity over valid pulsar-binary pairs.
    NOTE: We only consider the pulsars which see the binary via their Pulsar term

    Args:
        all_arr (np.ndarray): Quantity we want to average (N_binary, N_pulsar)
        N_contrib_psrs (np.ndarray): Number of pulsars seeing each binary (N_binary,)
        PT_valid_pair (np.ndarray): Indices of the valid pulsar-binary pairs

    Returns:
        np.ndarray: Array of the averaged quantity per binary (N_binary,)
    """


    # Put element to 0 if not valid pair (binary - pulsar)
    all_arr[~PT_valid_pair] = 0 * all_arr.unit

    # Sum the contribution of valid pulsars
    sum_arr = np.sum(all_arr, axis=-1)

    # Initialize at zero the avg
    avg = np.zeros_like(sum_arr)

    # Divide the sum by N_contrib only if N_contrib > 0
    np.divide(
        sum_arr,
        N_contrib_psrs,
        out=avg,
        where=N_contrib_psrs > 0
    )

    return avg


def load_configs(run_config_file: str) -> tuple:
    """
    Load configuration files for population models and PTA.
    
    Parameters
    ----------
    run_config_file : str
        Path to the run configuration file
        
    Returns
    -------
    tuple
        (run_config, PTA)
    """
    with open(run_config_file, 'r') as file:
        run_config = yaml.safe_load(file)

    config_file = f"./config/PTA/{run_config['PTA_config']}.yaml"
    with open(config_file, 'r') as file:
        PTA = yaml.safe_load(file)
        
    return run_config, PTA

def setup_population_models(pop_config: dict) -> list:
    """
    Setup and validate population models.
    
    Parameters
    ----------
    pop_config : dict
        Population configuration dictionary
        
    Returns
    -------
    list
        List of population models
    """
    pop_models = pop_config['models']
    
    for pop_model in pop_models:
        # Convert n0_dot to correct dimensions
        pop_model['n0_dot'] /= (u.Gyr * u.Mpc ** 3)
    
    return pop_models

def setup_parameter_grids(run_config: dict,
                          cosmo: object,
                          logz: bool = False) -> dict:
    """
    Setup parameter grids for Mcr, z, and forbr.
    
    Parameters
    ----------
    run_config : dict
        Parent configuration dictionary of the run
    cosmo: object
        Astropy cosmology object to compute luminosity distances
    logz: bool
        Boolean indicating if the redshift grid is in log-scale
        
    Returns
    -------
    dict
        Dictionary with grid edges and midpoints
    """
    bounds = run_config["bounds"]
    
    # Chirp mass grid
    log10_Mcr_edges = np.linspace(bounds['log10_Mcr_min'], bounds['log10_Mcr_max'], 
                                  run_config['N_bin'] + 1)
    log10_Mcr_mids = get_mids(log10_Mcr_edges)
    
    # Redshift grid
    if logz:
        print("Log10-spaced bins are used for redshift.", flush=True)
        log10_z_edges = np.linspace(np.log10(bounds['z_min']), np.log10(bounds['z_max']), 
                                    run_config['N_bin'] + 1)
        z_edges = 10 ** log10_z_edges
        z_mids = 10 ** get_mids(log10_z_edges)
    else:
        z_edges = np.linspace(bounds['z_min'], bounds['z_max'], 
                              run_config['N_bin'] + 1)
        z_mids = get_mids(z_edges)

    # Precompute luminosity distances
    dL_mids = cosmo.luminosity_distance(z_mids).to(u.Mpc)

    return {
        'log10_Mcr_edges': log10_Mcr_edges,
        'log10_Mcr_mids': log10_Mcr_mids,
        'z_edges': z_edges,
        'z_mids': z_mids,
        'dL_mids': dL_mids,
        'logz': logz
    }


def save_configs(run_config: dict, PTA: dict, output_dir: str):
    """
    Save the run configuration and PTA configuration to a YAML file for reproducibility.
    
    Parameters
    ----------
    run_config : dict
        Run configuration dictionary
    PTA : dict
        PTA configuration dictionary
    output_dir : str
        Directory where the configuration file will be saved
    """
    
    output_path = f"{output_dir}/config.yaml"
    with open(output_path, 'w') as file:
        yaml.dump(run_config, file)

    output_path = f"{output_dir}/PTA.yaml"
    with open(output_path, 'w') as file:
        yaml.dump(PTA, file)


# ================= Plots =========================

def make_efficiency_plot(efficiency_grid, grids, 
                         SNR_thresh,
                         output_path, fname,
                         log_eff=False,
                         title=None):

    log10_Mcr_mids = grids['log10_Mcr_mids']
    z_mids = grids['z_mids']
    # If logz is not there it is False
    logz = grids.get('logz', False)
    
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # use the true bin boundaries and a log y-scale
    X, Y = np.meshgrid(log10_Mcr_mids, z_mids)

    if log_eff:
        efficiency_grid = np.where(efficiency_grid > 0, 
                                   np.log10(efficiency_grid), 
                                   np.nan)

    pcm = ax.pcolormesh(X, Y, efficiency_grid.T, cmap='cubehelix')
    if logz:
        ax.set_yscale('log')

    add_str = r"$\log$ " if log_eff else ""
    cbar = fig.colorbar(pcm, ax=ax, label=add_str + fr'P$(\mathrm{{SNR}} > {SNR_thresh})$')
    ax.set_xlabel(r'$\log \mathcal{M}_{\mathrm{c,r}} / M_{\odot}$')
    ax.set_ylabel(r'$z$')
    if title is not None:
        plt.title(title)
    fig.tight_layout()

    try:
        output_file = output_path / fname
    except:
        output_file = output_path + fname
    fig.savefig(output_file, bbox_inches='tight')
    plt.show()

    return output_file