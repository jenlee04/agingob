# PARAMETERS FOR NEURONS #
def default_pars(**kwargs):
    """Default parameters shared by all neuron types."""
    pars = {}
    pars['V_rest']      = -65.   # resting potential [mV]
    pars['tau_m']       = 1e-3   # membrane time constant [s]
    pars['V_init']      = -65.   # initial potential [mV]
    pars['E_L']         = -65.   # leak reversal potential [mV]
    pars['tref']        = 2e-3   # refractory time [s]
    pars['theta_min']   = -65.   # lower bound of spike-prob range [mV]
    pars['theta_max']   = -55.   # upper bound of spike-prob range [mV]
    pars['r_max']       = 50.    # max firing rate [Hz]
    pars['noise_sigma'] = 0.01   # noise std dev [mV]; scaled by age_pars noise_scale
    pars['refv']        = -75.
    pars.update(kwargs)
    return pars


def specific_pars():
    """
    Return parameter dicts for OSN, PG, ET, MC, GC.
    """
    pars_OSN = default_pars(
        tau_m=1e-3, V_rest=-65.0, E_L=-65.0,
        theta_min=-65.0, theta_max=-55.0, r_max=50.0
    )
    pars_PG = default_pars(
        tau_m=2e-3, V_rest=-65.0, E_L=-65.0,
        theta_min=-65.0, theta_max=-60.0, r_max=30.0, noise_sigma = 0.001
    )
    pars_MC = default_pars(
        tau_m=5e-3, V_rest=-65.0, E_L=-65.0,
        theta_min=-65., theta_max=-57.0, r_max=30.0, noise_sigma=0.0
    )

    return pars_OSN, pars_PG, pars_MC


# ── SYNAPTIC PARAMETERS ───────────────────────────────────────────────────────

def synapse_pars():
    syn_pars = {}

    # OSN -> PG  
    syn_pars['OSN_PG'] = {'w': 0.03, 'E_syn': 5.0, 'tau_rise': 1e-3, 'tau_fall': 2e-3}

    # OSN -> MC dend
    syn_pars['OSN_MC'] = {'w': 0.03, 'E_syn': 5.0, 'tau_rise': 1e-3, 'tau_fall': 2e-3}

    # PG -> MC dend — inhibitory
    syn_pars['PG_MC']  = {'w': 0.0, 'E_syn': -75.0, 'tau_rise': 2e-3, 'tau_fall': 4e-3}


    return syn_pars


# aging pars

def apply_age_pars(ap: dict):
    """
    scale the existing pars using ratios 
    
    Parameters
    ----------
    ap : dict
        Output of age_pars() or age_pars_custom().

    Returns
    -------
    pars_OSN, pars_PG, pars_ET, pars_MC, pars_GC : dict
        Age-adjusted neuron parameter dicts.
    syn : dict
        Age-adjusted synapse parameter dict (weights scaled in-place copies).

    Usage
    -----
        ap = age_pars('old')
        pars_OSN, pars_PG, pars_ET, pars_MC, pars_GC, syn = apply_age_pars(ap)
    """
    import copy
    pars_OSN, pars_PG, pars_MC = specific_pars()
    syn = copy.deepcopy(synapse_pars())

    base_noise = 0.01  

    # things to alter #
    # OSN
    pars_OSN['r_max']       *= ap['osn_rmax_scale']
    pars_OSN['theta_max']   += ap['osn_theta_max_shift']
    pars_OSN['noise_sigma']  = base_noise * ap['noise_scale']

    # PG (no dedicated scale in age_pars; inherits noise)
    pars_PG['noise_sigma']   = base_noise * ap['noise_scale']

    # MC
    pars_MC['tau_m']        *= ap['mc_tau_scale']
    pars_MC['noise_sigma']   = base_noise * ap['noise_scale']

    # store the weights here and scale them
    syn['OSN_PG']['w']     *= ap['w_OSN_PG_scale']
    syn['OSN_MC']['w']      *= ap['w_OSN_MC_scale']
    syn['PG_MC']['w']       *= ap['w_PG_MC_scale']

    return pars_OSN, pars_PG, pars_MC, syn


######################### AGING PARS ##########################################

def age_pars(age_group: str = 'young') -> dict:
    """
   dict of age pars with diff groups
    """
    if age_group == 'young':
        return _age_young()
    elif age_group == 'old':
        return _age_old()
    else:
        raise ValueError(
            f"Unknown age_group '{age_group}'. Choose 'young' or 'old'."
        )


def _age_young() -> dict:
    return {
        # pop sizes #
        'n_OSNs_per_glom':  5,
        'n_PG_per_glom':    1,
        'n_MC_per_glom':    1,
        # synptic weight scaling #
        'w_OSN_PG_scale':   1.0,
        'w_OSN_MC_scale':      1.0,
        'w_PG_MC_scale':       1.0,
        # synaptic densities scaling #
        'p_OSN_PG':         1.0,
        'p_OSN_MC':         1.0,
        'p_PG_MC':          1.0,
       
        # changing neuronal biophys # 
        'osn_rmax_scale':       1.0,
        'osn_theta_max_shift':  0.0,
        'mc_tau_scale':         1.0,
        'noise_scale':          1.0,
        # changing wiring #
        'w_min_fraction':    0.05,
        'osn_sigma_scale':   1.0,
        # descriptions for bookeeping #
        'description': 'Young adult — baseline parameters',
        'age_group':   'young',
    }

def _age_old() -> dict:
    """
    old group
    """
    return {
        # pop sizes # 
        'n_glomeruli':      70,
        'n_OSNs_per_glom':  3,
        'n_PG_per_glom':    1,
        'n_MC_per_glom':    1,
        # synptic weight scaling #
        'w_OSN_PG_scale':   0.65,
        'w_OSN_MC_scale':      0.65,
        'w_PG_MC_scale':       0.55,
        # synaptic densities scaling #
        'p_OSN_PG':         0.70,
        'p_OSN_MC':         0.70,
        'p_PG_MC':          0.70,
        # changing neuronal biophys # 
        'osn_rmax_scale':       0.75,
        'osn_theta_max_shift':  +3.0,
        'mc_tau_scale':         1.50,
        'noise_scale':          1.50,
        # wiring # 
        'w_min_fraction':    0.12,
        'osn_sigma_scale':   1.50,
        # descriptions #
        'descriptio=n': (
            'Old — ~40% OSN loss, major ET/PG/GC functional loss, '
            'severe inhibitory failure, 50% broader tuning, '
            'GC weight rule shifts from lateral (no_self) to self-driven (original)'
        ),
        'age_group': 'old',
    }

