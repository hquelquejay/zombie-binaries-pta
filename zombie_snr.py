import jax
import jax.numpy as jnp
import jax.scipy as jsp

########## SNR computation from enterprise matrices using JAX ##########

@jax.jit
def single_pulsar_snr2(Sig_cf, F_T, Ninv, res):
    """Return the matched-filter SNR-squared contribution from a single pulsar.

    Parameters
    ----------
    Sig_cf : jax.Array
        Cholesky factor of the PTA covariance matrix.
    F_T : jax.Array
        Transposed response matrix for the pulsar.
    Ninv : jax.Array
        Diagonal inverse-noise weights.
    res : jax.Array
        Timing residual vector for the candidate signal.

    Returns
    -------
    jax.Array
        Contribution to the per-pulsar SNR squared.
    """
    # 0. Compute Ninv res
    # This assumes Ninv is a vector of the inverse variance
    NinvR = Ninv * res

    # 1. rNr term: dot product weighted by diagonal noise
    rNr = jnp.dot(res, NinvR)
    
    # 2. TNr part: F.T @ (Ninv * res)
    TNr = jnp.matmul(F_T, NinvR)
    
    # 3. Solve the linear system using the Cholesky decomposition
    # NOTE: we are only storing the Cholesky factor in Sig_cf, not the boolean
    SigTNr = jsp.linalg.cho_solve((Sig_cf, False), TNr)
    
    # 4. Return the contribution to SNR^2
    return rNr - jnp.dot(TNr, SigTNr)


@jax.jit
def compute_SNR2_per_psr_jax(Sig_cf_s, FT_s, Ninv_s, CW_residuals):
    """Compute the SNR-squared contribution for each pulsar across a set of signals.

    Parameters
    ----------
    Sig_cf_s : jax.Array
        Cholesky factors for all pulsars.
    FT_s : jax.Array
        Transposed response matrices for all pulsars.
    Ninv_s : jax.Array
        Inverse-noise weights for all pulsars.
    CW_residuals : jax.Array
        Timing residuals for each candidate signal and pulsar.

    Returns
    -------
    jax.Array
        Per-pulsar SNR-squared values.
    """

    # Vectorize over the first axis (0) for all inputs
    # Use vmap to compute the contribution for every pulsar in parallel
    vmapped_contribution = jax.vmap(single_pulsar_snr2, in_axes=(0, 0, 0, 0))
    
    # Compute the SNR² in each pulsar
    return vmapped_contribution(Sig_cf_s, FT_s, Ninv_s, CW_residuals)
    

@jax.jit
def compute_SNR_jax(Sig_cf_s, FT_s, Ninv_s, CW_residuals):
    """Return the total SNR for one or more candidate signals by summing pulsar contributions.

    Parameters
    ----------
    Sig_cf_s : jax.Array
        Cholesky factors for all pulsars.
    FT_s : jax.Array
        Transposed response matrices for all pulsars.
    Ninv_s : jax.Array
        Inverse-noise weights for all pulsars.
    CW_residuals : jax.Array
        Timing residuals for each candidate signal and pulsar.

    Returns
    -------
    jax.Array
        Total SNR for each candidate signal.
    """
    # Compute the SNR² in each pulsar
    snr2_values = compute_SNR2_per_psr_jax(Sig_cf_s, FT_s, Ninv_s, CW_residuals)
    
    # Sum the individual SNR² values and take the square root
    return jnp.sqrt(jnp.sum(snr2_values))


@jax.jit
def compute_all_zombies_snr_jax(Sig_cf_s, FT_s, Ninv_s, all_CW_residuals):
    """Evaluate the total SNR for every candidate zombie signal in a batch.

    Parameters
    ----------
    Sig_cf_s : jax.Array
        Cholesky factors for all pulsars.
    FT_s : jax.Array
        Transposed response matrices for all pulsars.
    Ninv_s : jax.Array
        Inverse-noise weights for all pulsars.
    all_CW_residuals : jax.Array
        Residual arrays with shape (N_zombies, N_pulsars, N_toas).

    Returns
    -------
    jax.Array
        Total SNR for each zombie candidate.
    """
    # We vmap the existing function. 
    # Sig_cf_s, F_s, and Ninv_s are 'None' because they are the same for every zombie.
    # all_CW_residuals is '0' because we want to slice along the zombie dimension.
    zombie_vmap = jax.vmap(compute_SNR_jax, in_axes=(None, None, None, 0))
    
    return zombie_vmap(Sig_cf_s, FT_s, Ninv_s, all_CW_residuals)

@jax.jit
def compute_all_zombies_snr_per_psr_jax(Sig_cf_s, FT_s, Ninv_s, all_CW_residuals):
    """Evaluate the SNR-squared contribution from each pulsar for every zombie signal.

    Parameters
    ----------
    Sig_cf_s : jax.Array
        Cholesky factors for all pulsars.
    FT_s : jax.Array
        Transposed response matrices for all pulsars.
    Ninv_s : jax.Array
        Inverse-noise weights for all pulsars.
    all_CW_residuals : jax.Array
        Residual arrays with shape (N_zombies, N_pulsars, N_toas).

    Returns
    -------
    jax.Array
        Per-pulsar SNR-squared values for each zombie candidate.
    """
    # We vmap the existing function. 
    # Sig_cf_s, F_s, and Ninv_s are 'None' because they are the same for every zombie.
    # all_CW_residuals is '0' because we want to slice along the zombie dimension.
    zombie_vmap = jax.vmap(compute_SNR2_per_psr_jax, in_axes=(None, None, None, 0))
    
    return zombie_vmap(Sig_cf_s, FT_s, Ninv_s, all_CW_residuals)