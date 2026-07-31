import astropy.constants as const
import astropy.units as u
import numpy as np
import scipy.linalg as sl
from enterprise.signals import gp_signals, signal_base
from enterprise_extensions import model_utils
from enterprise_extensions.blocks import (common_red_noise_block,
                                          red_noise_block, white_noise_block)
from fakepta.fake_pta import make_fake_array


def get_N_ToAs_for_exact_Tobs(Tobs: float, fsamp_0: float) -> float:
    """Return the number of timing samples for a given observing span and cadence.

    Parameters
    ----------
    Tobs : float
        Total observing duration.
    fsamp_0 : float
        Sampling frequency of the PTA data.

    Returns
    -------
    float
        Integer number of ToAs implied by the requested sampling.
    """
    # Compute closest f_samp to get integer number of ToAs
    N_tilde = (Tobs * fsamp_0).value

    # NOTE: Don't need realistic f_samp as we concentrate on low frequency candidates
    N_ToAs = np.rint(N_tilde) + 1

    return int(N_ToAs)

def get_nbin_from_Tobs(Tobs: float, fmax: float) -> float:
    """Return the number of Fourier bins needed to reach a given maximum frequency.

    Parameters
    ----------
    Tobs : float
        Observing time baseline.
    fmax : float
        Maximum search frequency.

    Returns
    -------
    float
        Number of frequency bins in the corresponding linear grid.
    """
    return int(np.rint(Tobs * fmax))

def get_pta_noisedict(psrs: list) -> dict:
    """Return the combined noise dictionary for all pulsars in a PTA.

    Parameters
    ----------
    psrs : list
        List of pulsar objects containing noise metadata.

    Returns
    -------
    dict
        Noise dictionary for the full PTA.
    """
    pta_noisedict = {}
    for psr in psrs:
        pta_noisedict.update(psr.noisedict)
    return pta_noisedict

def init_GWB_pta(psrs, 
                 GWB_psd='powerlaw', 
                 noisedict=None, 
                 white_vary=False,
                 components=30, 
                 tnequad=False,
                 GWB_Tspan=None,
                 select='backend', 
                 tm_marg=False,
                 tm_svd=True,
                 frac_RN=0.,
                 gamma_RN=None,
                 orf=None):
    """Construct a PTA model including a common gravitational-wave background and pulsar noise.

    Parameters
    ----------
    psrs : list
        Pulsar objects included in the PTA.
    GWB_psd : str, optional
        Power spectral density model for the common red-noise process.
    noisedict : dict, optional
        Default pulsar-noise parameters to apply to the PTA.
    white_vary : bool, optional
        If True, allow white-noise amplitudes to vary.
    components : int, optional
        Number of Fourier components used in the red-noise basis.
    tnequad : bool, optional
        If True, include a time-correlated noise term.
    GWB_Tspan : float, optional
        Time baseline used to define the common-noise frequency grid.
    select : str, optional
        Backend selection used in white-noise blocks.
    tm_marg : bool, optional
        If True, use a marginalized timing model.
    tm_svd : bool, optional
        If True, stabilize the timing-model design matrix with SVD.
    frac_RN : float, optional
        Fraction of pulsars assigned an individual red-noise component.
    gamma_RN : float, optional
        Power-law slope for the per-pulsar red-noise process.
    orf : callable, optional
        Overlap reduction function for the common background.

    Returns
    -------
    enterprise.signals.signal_base.PTA
        PTA object ready for likelihood evaluation.
    """

    # For common red noise, if not specified find the maximum Tspan to set GW frequency sampling
    if GWB_Tspan is None:
        GWB_Tspan = model_utils.get_tspan(psrs)

    # CRN block
    s_com = common_red_noise_block(psd=GWB_psd, prior='log-uniform',
                                   Tspan=GWB_Tspan, components=components, 
                                   combine=True,
                                   orf=orf,
                                   name='gw', coefficients=False)

    models, Npsr = [], len(psrs)
    for ip, psr in enumerate(psrs):
        # timing model
        if tm_marg:
            s_sgl = gp_signals.MarginalizingTimingModel(use_svd=tm_svd)
        else:
            s_sgl = gp_signals.TimingModel(use_svd=tm_svd)

        # adding white-noise, and acting on psr objects
        s_sgl = s_sgl + white_noise_block(vary=white_vary, inc_ecorr=False,
                                          tnequad=tnequad, select=select)
        
        if (ip + 1) / Npsr <= frac_RN:
            s_sgl = s_sgl + red_noise_block(psd='powerlaw', prior='log-uniform', 
                                            Tspan=GWB_Tspan, # NOTE: same Tspan as the GWB 
                                            components=components, # NOTE: same number of components as the GWB 
                                            gamma_val=gamma_RN)

        s = s_sgl + s_com

        models.append(s(psr))

    # set up PTA
    pta = signal_base.PTA(models)

    # set white noise parameters
    if not white_vary:
        if noisedict is None:
            try:
                noisedict = {}
                for psr in psrs:
                    noisedict.update(psr.noisedict)
                pta.set_default_params(noisedict)
            except:
                print('No noise dictionary provided!...')
        else:
            pta.set_default_params(noisedict)

    return pta


def build_CholFactors(pta, pta_params):
    """Build the Cholesky factors of the PTA covariance matrices for each pulsar.

    Parameters
    ----------
    pta : object
        Enterprise PTA object.
    pta_params : dict
        PTA parameter dictionary used to evaluate the model.

    Returns
    -------
    ndarray
        Cholesky factors for each pulsar covariance matrix.
    """
    # Initialize the list to store Cholesky Factorizations
    Sig_cf_s = []

    # Extract the design matrix for each pulsar
    TNT_s = pta.get_TNT(pta_params)

    # Extract all phi_invs
    phi_inv_s = pta.get_phiinv(pta_params)

    for ip in range(len(pta.pulsars)):
        # Get the Fourier design matrix for this pulsar (include both RN and Mmat)
        TNT = TNT_s[ip]

        # Get phi matrix
        phiinv = phi_inv_s[ip]

        # Compute Sigma^{-1} F^T N^{-1}
        Sigma = TNT + (np.diag(phiinv) if phiinv.ndim == 1 else phiinv)
        Sig_cf_s.append(sl.cho_factor(Sigma)[0]) # Forget the boolean

    return np.array(Sig_cf_s) # (Npsr, N_basis, N_basis)


def setup_pta(run_config: dict, 
              PTA: dict,
              keep_DM_columns: bool = False,
              thin_factor: int = 1) -> tuple:
    """Assemble a synthetic PTA configuration and initialize the corresponding model.

    Parameters
    ----------
    run_config : dict
        Configuration describing the run setup and observational assumptions.
    PTA : dict
        PTA configuration parameters.
    keep_DM_columns : bool, optional
        If True, retain the full dispersion-measure design matrix.
    thin_factor : int, optional
        Factor by which the PTA sampling is reduced to speed up calculations.

    Returns
    -------
    tuple
        PTA object and parameter dictionary used for SNR evaluation.
    """

    # Compute Tspan of the PTA
    PTA_Tobs = (PTA['Nyear'] * u.year).to(u.s)
    
    # Compute N_ToA of the PTA
    PTA_f_samp = (1 / (PTA['cadence'] * u.day)).to(u.Hz)
    PTA['N_ToA'] = get_N_ToAs_for_exact_Tobs(PTA_Tobs.to(u.s), PTA_f_samp)
    print(f"The PTA should have N_ToAs: {PTA['N_ToA']} ToAs", flush=True)
    # Apply the thinning factor to reduce the number of ToAs (and speed up computations)
    PTA['N_ToA'] = PTA['N_ToA'] // thin_factor
    if thin_factor > 1:
        print(f"Applying a thinning factor of {thin_factor}, the PTA will have N_ToAs: {PTA['N_ToA']} ToAs", flush=True)
    
    # Define the noise properties of the PTA pulsars
    nbin_RN = get_nbin_from_Tobs(PTA_Tobs.to(u.s),
                                 # NOTE: you might want to increase fmax is noise is flatter
                                 fmax=7e-8 * u.Hz)
    print(f"Using {nbin_RN} Fourier bins for the red noise.", flush=True)
    
    # Add RN only
    RN = PTA['frac_RN'] > 0
    if not RN:
        print("No red noise in the PTA pulsars.", flush=True)
        custom_model = {'RN': None, 'DM': None, 'Sv': None}
    else:
        print(f"Red noise in {PTA['frac_RN']*100:.1f}% of the PTA pulsars with log10_A_RN = {PTA['log10_A_RN']} \
                and gamma_RN = {PTA['gamma_RN']}.", flush=True)
        custom_model = {'RN': nbin_RN, 'DM': None, 'Sv': None}
    
    # Noise parameters
    noisedict = {
        "efac": 1.0,
        "log10_tnequad": -9.,
        "red_noise_log10_A": PTA['log10_A_RN'],
        "red_noise_gamma": PTA['gamma_RN']
    }
    
    # Generate a fake array of pulsars
    PTA['pulsars'] = make_fake_array(npsrs=PTA['Npsr'],
                                     Tobs=PTA_Tobs.to(u.year).value,
                                     ntoas=PTA['N_ToA'],
                                     gaps=PTA['gaps'],
                                     toaerr=PTA['RMS'],
                                     isotropic=PTA['isotropic'],
                                     noisedict=noisedict,
                                     custom_model=custom_model)
    
    # Since only one frequency, remove freq-dependent columns of design matrix
    if not keep_DM_columns:
        # NOTE: this affects the SNR at the % level (minor impact)
        for psr in PTA['pulsars']:
            psr.Mmat = np.delete(psr.Mmat, [3, 4, 5], axis=1)
    
    # Define t0
    PTA['t0'] = min([min(psr.toas) for psr in PTA['pulsars']])

    # Store toas and Apply the thinning factor
    PTA['psr_toas'] = np.array([psr.toas for psr in PTA['pulsars']])

    # Initialize the pta object
    pta_noisedict = get_pta_noisedict(PTA['pulsars'])
    
    pta = init_GWB_pta(PTA['pulsars'], 
                       GWB_psd='powerlaw', 
                       noisedict=pta_noisedict, 
                       white_vary=True,
                       components=nbin_RN,
                       tnequad=True,
                       GWB_Tspan=PTA_Tobs.to(u.s).value,
                       select='backend',
                       tm_marg=False,
                       tm_svd=True,
                       frac_RN=PTA['frac_RN'],
                       gamma_RN=PTA['gamma_RN'],
                       orf=PTA['orf']
    )
    
    # Fix the pta parameters for WN and RN
    pta_params = {}
    for ip, psr in enumerate(PTA['pulsars']):
        pta_params.update({f"{psr.name}_NUPPI.1400_efac": 1.0})
        pta_params.update({f"{psr.name}_NUPPI.1400_log10_tnequad": -9.})
        
        if (ip + 1) / PTA['Npsr'] <= PTA['frac_RN']:
            pta_params.update({f"{psr.name}_red_noise_log10_A": PTA['log10_A_RN']})
            pta_params.update({f"{psr.name}_red_noise_gamma": PTA['gamma_RN']})
    
    pta_params.update({"gw_log10_A": float(np.log10(run_config['hc_ref']))})
    pta_params.update({"gw_gamma": 13. / 3}) # NOTE: fixing to the fiducial 13/3 value
    print("PTA parameter vector used:", pta_params, flush=True)

    # Store the parameter vector
    pta.setup_pars = pta_params

    # Overwrite signal collections to avoid caching
    # NOTE: this can be removed when using jax
    # for ip, psr in enumerate(PTA['pulsars']):
    #     pta._signalcollections[ip] = cw.sigCollection_no_cache(pta._signalcollections[ip])(psr)

    ### Build the array of design matrices, Ninv and Cholesky Decomposition
    # Store the Cholesky decompositions
    PTA['Sig_cf_s'] = build_CholFactors(pta, pta_params)

    # Get Ninv vector of each pulsar (NOTE: requires N to be diagonal)
    assert np.all([N.ndim == 1 for N in pta.get_ndiag(pta_params)]), "N is not diagonal, check the pta model"
    PTA['Ninv_s'] = np.array([1. / N for N in pta.get_ndiag(pta_params)])

    # Get the design matrices transposes
    PTA['FT_s'] = np.array([F.T for F in pta.get_basis(pta_params)])

    ### Set the positions and distances of the pulsars to the desired PTA
    if PTA['psrs_pos'] is not None:
        print(f"Fixing the positions and distances of the pulsars to the {PTA['psrs_pos']} ones.", flush=True)
        temp = np.load(f'./data/PTA/{PTA["psrs_pos"]}.npy')
        assert temp.shape == (PTA['Npsr'], 4) # check we do have positions and distances for each pulsar

        # Store psr_pos
        PTA['psr_pos_s'] = temp[:,:3]

        # Store photon time of flight to each pulsar
        PTA['psr_T_s'] = (temp[:,-1] * const.kpc / const.c).to(u.s)
    else:
        PTA['psr_pos_s'] = np.array([psr.pos for psr in PTA['pulsars']])
        PTA['psr_T_s'] = np.array([psr.pdist[0] for psr in PTA['pulsars']]) * (const.kpc / const.c).to(u.s)

    # Store the maximum photon time of flight of the PTA
    PTA['Ta_max'] = max(PTA['psr_T_s'])
    print("Maximal pulsar's photon time of flight, Ta_max =", PTA['Ta_max'].to(u.yr), flush=True)
    
    return pta