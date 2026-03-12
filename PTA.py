import astropy.constants as const
import astropy.units as u
import numpy as np
import scipy.linalg as sl
from enterprise.signals import gp_signals, signal_base
from enterprise_extensions import model_utils
from enterprise_extensions.blocks import (common_red_noise_block,
                                          red_noise_block, white_noise_block)
from fakepta.fake_pta import Pulsar


def get_N_ToAs_for_exact_Tobs(Tobs: float, fsamp_0: float) -> float:
    """
    Compute the number of ToAs associated to a given 
    observing duration and sampling frequency

    Args:
        Tobs (float): Observing duration
        fsamp_0 (float): Sampling frequency

    Returns:
        float: Number of ToAs
    """
    # Compute closest f_samp to get integer number of ToAs
    N_tilde = (Tobs * fsamp_0).value

    N_ToAs = np.rint(N_tilde) + 1

    return int(N_ToAs)


def get_nbin_from_Tobs(Tobs: float, fmax: float) -> float:
    """
    Get the number of Fourier bin to reach fmax starting from Tobs 
    with a linear frequency grid

    Args:
        Tobs (float): Observing time of the PTA
        fmax (float): Target maximum frequency 

    Returns:
        float: Number of Fourier bins
    """
    return int(np.rint(Tobs * fmax))

def get_pta_noisedict(psrs: list) -> dict:
    """
    Update the pta object noise dictionary

    Args:
        psrs (list): List of pulsar objects

    Returns:
        dict: PTA noisedict dictionary
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
    """
    Initialize an enterprise PTA object with GWB noise and individual pulsar noise models.

    Args:
        psrs (list): List of pulsar objects from fakepta.
        GWB_psd (str, optional): Power spectral density model for the GWB. Default is 'powerlaw'.
        noisedict (dict, optional): Dictionary of noise parameters. If None, uses default from pulsars.
        white_vary (bool, optional): Whether to vary white noise parameters. Default is False.
        components (int, optional): Number of frequency components for red noise. Default is 30.
        tnequad (bool, optional): Whether to include TNEQUAD in white noise. Default is False.
        GWB_Tspan (float, optional): Time span for GWB frequency sampling. If None, uses maximum Tspan from pulsars.
        select (str, optional): Backend selection for white noise. Default is 'backend'.
        tm_marg (bool, optional): Whether to use marginalizing timing model. Default is False.
        tm_svd (bool, optional): Whether to use SVD in timing model. Default is True.
        frac_RN (float, optional): Fraction of pulsars with red noise. Default is 0.0.
        gamma_RN (float, optional): Spectral index for red noise. Required if frac_RN > 0.
        orf (str or callable, optional): Overlap reduction function for GWB. Default is None.

    Returns:
        PTA: Enterprise PTA object with the specified models.
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
    """
    Construct and save the cholesky decomposition used in the SNR computations

    Args:
        pta (object): Enterprise PTA object
        pta_params (dict): PTA noise dictionary

    Returns:
        np.ndarray: Array of the Cholesky factorization of each pulsar
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


def make_fake_array(npsrs=25, Tobs=None, ntoas=None, gaps=True, toaerr=None, pdist=None, 
                    freqs=[1400], isotropic=False, backends=None, noisedict=None, 
                    custom_model=None, ephem=None):
    """
    Create a fake array of pulsar objects with specified observational properties.

    This function is adapted from the fake_pta package.

    Args:
        npsrs (int, optional): Number of pulsars to generate. Defaults to 25.
        Tobs (float or array-like, optional): Observation time for each pulsar in years. If float, uses the same for all. If None, random between 10-20 years. Defaults to None.
        ntoas (int or array-like, optional): Number of time-of-arrival (TOA) measurements per pulsar. If int, uses the same for all. If None, computed based on cadence. Defaults to None.
        gaps (bool, optional): Whether to introduce gaps in TOA sampling. Defaults to True.
        toaerr (float or array-like, optional): TOA error in seconds. If float, uses the same for all. If None, random between 1e-7 and 1e-5 s. Defaults to None.
        pdist (float or list, optional): Pulsar distances in kpc. If float, uses the same for all. If None, random between 0.5-1.5 kpc. Defaults to None.
        freqs (list, optional): Observing frequencies in MHz. Defaults to [1400].
        isotropic (bool, optional): Whether to place pulsars isotropically on the sky. Defaults to False.
        backends (str or list, optional): Backend names for each pulsar. If str, uses for all. If None, generates random backends. Defaults to None.
        noisedict (dict, optional): Noise parameter dictionary. Defaults to None.
        custom_model (dict, optional): Custom noise model parameters. Defaults to None.
        ephem (str, optional): Ephemeris to use. Defaults to None.

    Returns:
        list: List of fake pulsar objects.
    """

    if isotropic:
        # Fibonacci sequence on sphere
        i = np.arange(0, npsrs, dtype=float) + 0.5
        golden_ratio = (1 + 5**0.5)/2
        costhetas = 1 - 2*i/npsrs
        phis = np.mod(2 * np.pi * i / golden_ratio, 2*np.pi)
    else:
        costhetas = np.random.uniform(-1., 1., size=npsrs)
        phis = np.random.uniform(0., 2*np.pi, size=npsrs)

    # Observation time for each pulsar
    if Tobs is None:
        Tobs = np.random.uniform(10, 20, size=npsrs)
    elif isinstance(Tobs, float) or isinstance(Tobs, int):
        Tobs = Tobs * np.ones(npsrs)

    # Number of TOAs for each pulsar
    yr = 365.25*24*3600
    if ntoas is None:
        cadence = 7 * 24*3600 # days
        # draw F0 and correct cadence wrt F0
        F0 = np.random.uniform(200, 300, size=npsrs)
        d_cadence = (F0 * cadence - np.floor(F0 * cadence )) / F0
        cadence = cadence - d_cadence
        ntoas = np.int32(Tobs * 365.25 * 24 * 3600 / cadence)
    elif isinstance(ntoas, float) or isinstance(ntoas, int):
        F0 = 200 * np.ones(npsrs)
        ntoas = np.int32(ntoas * np.ones(npsrs))
        cadence = Tobs * yr / (ntoas - 1)

    # Init TOAs from latest observation time
    Tmax = np.amax(Tobs)

    # Make unevenly sampled TOAs if gaps is True
    if gaps:
        gap_odds = [True, True, True, False] # one out of five
        keep = [np.random.choice(gap_odds, size=ntoa) for ntoa in ntoas]
        toas = [(Tmax - Tobs[i])*yr + np.arange(1, ntoas[i]+1)*cadence[i] for i in range(npsrs)]
        toas = [toas[i][keep[i]] for i in range(npsrs)]
    else:
        toas = [(Tmax - Tobs[i])*yr + np.arange(1, ntoas[i]+1)*cadence[i] for i in range(npsrs)]
    if toaerr is None:
        toaerr = np.power(10, np.random.uniform(-7., -5., size=npsrs))
    elif isinstance(toaerr, float):
        toaerr = toaerr * np.ones(npsrs)

    # Init pulsar distances
    if pdist is None:
        dists = np.random.uniform(0.5, 1.5, size=npsrs)
        pdist = [[dist, 0.2*dist] for dist in dists]
    elif isinstance(pdist, float):
        pdist = [[pdist, 0.2*pdist]] * npsrs

    # Init backends
    if backends is None:
        backends = []
        for _ in range(npsrs):
            n_backends = np.random.randint(1, 3)
            backends.append(['backend_'+str(k) for k in range(n_backends)])
    elif isinstance(backends, str):
        backends = [[backends]] * npsrs
    elif isinstance(backends, list):
        if not isinstance(backends[0], list):
            backends = [backends] * npsrs
    
    # Init noise properties


    assert (len(Tobs) == npsrs), '"Tobs" must be same size as "npsrs"'
    assert (len(ntoas) == npsrs), '"ntoas" must be same size as "npsrs"'
    assert (len(toaerr) == npsrs), '"toaerr" must be same size as "npsrs"'
    assert (len(pdist) == npsrs), '"pdist" must be same size as "npsrs"'
    assert (len(backends) == npsrs), '"backends" must be same size as "npsrs"'

    # Create pulsars and add noises
    psrs = []
    for i in range(npsrs):
        if custom_model is None:
            custom_model = None
        psr = Pulsar(toas[i], toaerr[i], np.arccos(costhetas[i]), phis[i], pdist[i], freqs=freqs, backends=backends[i], custom_noisedict=noisedict, custom_model=custom_model, tm_params={'F0':(F0[i], np.random.uniform(1e-13, 1e-12))}, ephem=ephem)
        psr.add_white_noise()
        try:
            psr.add_red_noise(spectrum='powerlaw', log10_A=psr.noisedict[psr.name+'_red_noise_log10_A'], gamma=psr.noisedict[psr.name+'_red_noise_gamma'])
        except:
            psr.add_red_noise(spectrum='powerlaw', log10_A=np.random.uniform(-17., -13), gamma=np.random.uniform(1, 5))
        
        try:
            psr.add_dm_noise(spectrum='powerlaw', log10_A=psr.noisedict[psr.name+'_dm_gp_log10_A'], gamma=psr.noisedict[psr.name+'_dm_gp_gamma'])
        except:
            psr.add_dm_noise(spectrum='powerlaw', log10_A=np.random.uniform(-17., -13), gamma=np.random.uniform(1, 5))
        
        try:
            psr.add_chromatic_noise(spectrum='powerlaw', log10_A=psr.noisedict[psr.name+'_chrom_gp_log10_A'], gamma=psr.noisedict[psr.name+'_chrom_gp_gamma'])
        except:
            psr.add_chromatic_noise(spectrum='powerlaw', log10_A=np.random.uniform(-17., -13), gamma=np.random.uniform(1, 5))
        psrs.append(psr)

    return psrs



def setup_pta(run_config: dict, 
              PTA: dict,
              keep_DM_columns: bool = False) -> object:
    """
    Setup pta object and fill in PTA dictionary
    
    Parameters
    ----------
    run_config: dict
        Run configuration
    PTA: dict
        PTA configuration
    keep_DM_columns: bool
        Indicate whether to keep or not DM columns of the design matrix
        
    Returns
    -------
    pta: object
        Enterprise pta object
    """

    # Compute Tspan of the PTA
    PTA_Tobs = (PTA['Nyear'] * u.year).to(u.s)
    
    # Compute N_ToA of the PTA
    PTA_f_samp = (1 / (PTA['cadence'] * u.day)).to(u.Hz)
    PTA['N_ToA'] = get_N_ToAs_for_exact_Tobs(PTA_Tobs.to(u.s), PTA_f_samp)
    
    # Define the noise properties of the PTA pulsars
    nbin_RN = get_nbin_from_Tobs(PTA_Tobs.to(u.s), fmax=1e-7 * u.Hz)
    
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
                                     freqs=[1400],
                                     isotropic=PTA['isotropic'],
                                     backends=['NUPPI.1400'],
                                     noisedict=noisedict,
                                     custom_model=custom_model)
    
    # Since only one frequency, remove freq-dependent columns of design matrix
    if not keep_DM_columns:
        # NOTE: this affects the SNR at the % level (minor impact)
        for psr in PTA['pulsars']:
            psr.Mmat = np.delete(psr.Mmat, [3, 4, 5], axis=1)
    
    # Define t0
    PTA['t0'] = min([min(psr.toas) for psr in PTA['pulsars']])

    # Store toas
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
    
    # Fix the GWB parameters to values consistent with PTA observations
    pta_params.update({"gw_log10_A": float(np.log10(run_config['hc_ref']))})
    pta_params.update({"gw_gamma": 13. / 3}) # NOTE: fixing to the fiducial 13/3 value
    
    print("PTA parameter vector used:", pta_params, flush=True)

    ### Build the array of design matrices, Ninv and Cholesky Decomposition
    # Store the Cholesky decompositions
    PTA['Sig_cf_s'] = build_CholFactors(pta, pta_params)

    # Get Ninv vector of each pulsar (NOTE: requires N to be diagonal)
    assert np.all([N.dim == 1 for N in pta.get_ndiag(pta_params)]), "N is not diagonal, check the pta model"
    PTA['Ninv_s'] = np.array([1. / N for N in pta.get_ndiag(pta_params)])

    # Get the design matrices
    PTA['F_s'] = np.array(pta.get_basis(pta_params))

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