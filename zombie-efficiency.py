"""
Compute detection efficiency of zombie binaries for a given PTA configuration over a (Mcr, z) grid.

Author: Hippolyte Quelquejay Leclere
Date: March 2026
"""

import argparse
import sys
from pathlib import Path

# Add current directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent))

import astropy.units as u
import numpy as np
from astropy.cosmology import Planck18

import PTA as pta
import utils
import zombie_rate as rate


def clean_PTA_dict(PTA: dict) -> dict:
    """
    Clean the PTA dictionary before saving it to yaml file

    Args:
        PTA (dict): Initial PTA dictionary

    Returns:
        dict: Cleaned PTA dictionary
    """

    PTA_copy = PTA.copy()
    PTA_copy.pop('pta', None)
    PTA_copy.pop('pulsars', None)
    PTA_copy.pop('psr_pos_s', None)
    PTA_copy.pop('psr_toas', None)
    PTA_copy.pop('psr_T_s', None)
    PTA_copy.pop('Sig_cf_s', None)
    PTA_copy.pop('Ninv_s', None)
    PTA_copy.pop('F_s', None)
    # Store t0 as float
    PTA_copy['t0'] = float(PTA_copy['t0'])
    # Update Ta_max
    Ta_max = float(np.round(PTA['Ta_max'].to(u.yr).value, 2))
    PTA_copy['Ta_max'] = Ta_max # convert to value for yaml

    return PTA_copy

def save_results(efficiency_grid: np.ndarray,
                 grids: dict,
                 output_dir: str):
    """
    Save efficiency and grid results to npz file.
    
    Parameters
    ----------
    efficiency_grid : np.ndarray
        Efficiency grid
    grids : dict
        Parameter grids dictionary
    output_dir : str
        Output directory path
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    npz_file = output_path / f"efficiency-grid.npz"
    
    np.savez(npz_file,
             efficiency=efficiency_grid,
             log10_Mcr_mids=grids['log10_Mcr_mids'],
             z_mids=grids['z_mids'],
             z_edges=grids['z_edges'],
             logz=grids['logz'])
    
    print(f"Results saved to {npz_file}", flush=True)


def plot_efficiency_grid(efficiency_grid: np.ndarray,
                         grids: dict,
                         output_dir: str,
                         run_config: dict):
    """
    Create and save 2D plot of efficiency grid.

    Parameters
    ----------
    efficiency_grid : np.ndarray
        Efficiency grid
    grids : dict
        Parameter grids dictionary
    output_dir : str
        Output directory path
    run_config: dict
        Run parameters dictionary
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # linear efficiency
    output_file = utils.make_efficiency_plot(efficiency_grid, grids, 
                                             SNR_thresh=run_config['SNR_thresh'],
                                             output_path=output_path, 
                                             fname="efficiency-grid.pdf",
                                             log_eff=False)
    
    # log10 efficiency plot
    output_file = utils.make_efficiency_plot(efficiency_grid, grids, 
                                             SNR_thresh=run_config['SNR_thresh'],
                                             output_path=output_path, 
                                             fname="log-efficiency-grid.pdf",
                                             log_eff=True)

    print(f"Plots saved to {output_file}", flush=True)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Compute efficiency over a log_10 Mcr / (log_10) z grid for zombie SMBHBs"
    )
    parser.add_argument("config_files", nargs='+', type=str,
                       help="Run configuration file(s)")
    parser.add_argument("--outdir", type=str, required=True,
                       help="Directory to store results")
    parser.add_argument("--ncpu", type=int, default=1,
                       help="Number of CPUs to use for parallelization [default: 1]")
    
    args = parser.parse_args()
    
    # Process each config file
    for config_file in args.config_files:
        print(f"Processing: {config_file}\n", flush=True)
        
        # Load configurations
        run_config, PTA = utils.load_configs(config_file)

        print("=" * 50, flush=True)
        print("Run Configuration\n", run_config, flush=True)
        print("=" * 50, flush=True)
        print("PTA Configuration\n", PTA, flush=True)
        print("=" * 50, flush=True)
        
        # Setup intrinsic parameter grids
        grids = utils.setup_parameter_grids(run_config=run_config,
                                            cosmo=Planck18,
                                            logz=run_config['bounds']['logz'])
        
        # Convert fth to Hz
        fth = run_config['fth'] * u.Hz
        
        # Setup pta object and associated params
        PTA['pta'] = pta.setup_pta(run_config, PTA)
        
        # Compute efficiency grid
        efficiency_grid = rate.compute_efficiency_2d_parallel_jax(
                                                    log10_Mcr_mids=grids['log10_Mcr_mids'],
                                                    z_mids=grids['z_mids'],
                                                    N_bin=run_config['N_bin'],
                                                    SNR_thresh=run_config['SNR_thresh'],
                                                    PTA=PTA,
                                                    fth=fth,
                                                    N_zombies_per_bin=run_config['N_zombies_per_bin'],
                                                    n_jobs=args.ncpu,
                                                    seed=run_config['seed'])
        
        # Save results
        save_results(efficiency_grid, grids, 
                     output_dir=args.outdir)
        
        # Create plot
        plot_efficiency_grid(efficiency_grid, grids,
                             output_dir=args.outdir + "/plots",
                             run_config=run_config)

        
        ### Save the config and PTA used in the run for reproducibility (in yaml format)
        # Clean the PTA dict first
        PTA_copy = clean_PTA_dict(PTA)
        
        utils.save_configs(run_config, PTA_copy, output_dir=args.outdir)

        print(f"\n{config_file} has been successfully processed", flush=True)


if __name__ == "__main__":
    main()
