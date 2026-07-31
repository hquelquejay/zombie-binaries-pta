#!/usr/bin/env python3
"""
Compute efficiency and Mcr/z grids for zombie SMBHB populations.

Author: Hippolyte Quelquejay Leclere
"""

import argparse
import pickle
import sys
from pathlib import Path

# Add current directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent))

import astropy.units as u
import jax
import numpy as np
from astropy.cosmology import Planck18

# Ensure float64 are used to handle large masses
jax.config.update("jax_enable_x64", True)

import PTA as pta
import utils
import zombie_rate as rate


def save_results(SNRs_dict: dict,
                 Nc_dict: dict,
                 efficiency_grid: np.ndarray,
                 grids: dict,
                 output_dir: str):
    """Save the efficiency data products and parameter grids to disk.

    Parameters
    ----------
    SNRs_dict : dict
        SNR values recorded for each grid point.
    Nc_dict : dict
        Number of contributing pulsars recorded for each grid point.
    efficiency_grid : np.ndarray
        Detection-efficiency grid as a function of chirp mass and redshift.
    grids : dict
        Grid metadata for mass, redshift, and related coordinates.
    output_dir : str
        Directory where the output files are written.

    Returns
    -------
    None
        Files are saved to the requested output directory.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save efficiency grid and associated parameter grids to npz file
    npz_file = output_path / "efficiency-grid.npz"
    np.savez(npz_file,
             efficiency=efficiency_grid,
             log10_Mcr_mids=grids['log10_Mcr_mids'],
             z_mids=grids['z_mids'],
             z_edges=grids['z_edges'],
             logz=grids['logz'])
    
    # Save the SNRs dictionary
    SNRs_file = output_path / "SNRs_dict.pkl"
    with open(SNRs_file, 'wb') as f:
        pickle.dump(SNRs_dict, f)

    # Save the number of contributing pulsars
    Nc_file = output_path / "Nc_dict.pkl"
    with open(Nc_file, 'wb') as f:
        pickle.dump(Nc_dict, f)
    
    print(f"Results saved to {output_path}", flush=True)


def plot_efficiency_grid(efficiency_grid: np.ndarray,
                         grids: dict,
                         output_dir: str,
                         run_config: dict):
    """Create and save the 2D efficiency plots for the computed grid.

    Parameters
    ----------
    efficiency_grid : np.ndarray
        Detection-efficiency values over the mass-redshift grid.
    grids : dict
        Parameter-grid metadata used for axis formatting.
    output_dir : str
        Directory where the plots are saved.
    run_config : dict
        Run configuration containing the detection threshold.

    Returns
    -------
    None
        Plot files are written to disk.
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
    """Run the full zombie-efficiency pipeline for one or more configuration files.

    Returns
    -------
    None
        The script computes efficiency grids, saves results, and writes plots.
    """
    parser = argparse.ArgumentParser(
        description="Compute efficiency over log_10 Mcr / (log_10) z grids for zombie SMBHBs"
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
        PTA['pta'] = pta.setup_pta(run_config, PTA, 
                                   thin_factor=run_config["toas_thin_factor"])
        
        # Create the outdir directory
        output_path = Path(args.outdir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Compute efficiency grid
        eff_grid, SNRs_dict, Nc_dict = rate.compute_efficiency_2d_cached_jax_threaded(
                                                log10_Mcr_mids=grids['log10_Mcr_mids'],
                                                z_mids=grids['z_mids'],
                                                N_bin=run_config['N_bin'],
                                                SNR_thresh=run_config['SNR_thresh'],
                                                PTA=PTA,
                                                fth=fth,
                                                N_zombies_per_bin=run_config['N_zombies_per_bin'],
                                                N_chunk=run_config['chunk_size'],
                                                n_jobs=args.ncpu,
                                                thin_factor=run_config["toas_thin_factor"],
                                                seed=run_config['seed'],
                                                Npsr_min=run_config['Npsr_min'],
                                                outdir=args.outdir)
        
        # Save results
        save_results(SNRs_dict,
                     Nc_dict, 
                     eff_grid,
                     grids,
                     output_dir=args.outdir)
        
        # Create plot
        plot_efficiency_grid(eff_grid, grids,
                             output_dir=args.outdir + "/plots",
                             run_config=run_config)

        
        ### Save the config and PTA used in the run for reproducibility (in yaml format)
        # Clean the PTA dict first
        PTA_copy = PTA.copy()
        
        # Save configs
        utils.save_configs(run_config, PTA_copy, output_dir=args.outdir)
        print(f"\n{config_file} has been successfully processed", flush=True)


if __name__ == "__main__":
    main()
