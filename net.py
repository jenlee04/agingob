# simple network

import numpy as np
from onecomptneuron import Neuron
from twocomptneuron import TCNeuron
from parstouse import specific_pars, synapse_pars


def Network(n_glomeruli=100, n_OSNs_per_glom=1,
            n_odor_dims=100, osn_sigma=4.0,
            pars_OSN=None, pars_PG=None, pars_MC=None, syn=None):

    """
    Build network 

    Returns
    -------
    dict with keys:
        glomeruli, OSNs, PGs, MCs,
        glom_to_preferred_mol, pars, syn_pars
    """
    if pars_OSN is None or pars_PG is None or pars_MC is None:
        pars_OSN, pars_PG, pars_MC = specific_pars()
    if syn is None:
        syn = synapse_pars()

    # anatomical map
    mol_map = np.arange(n_odor_dims)
    glom_to_preferred_mol = {}

    glomeruli, all_OSNs, all_PGs, all_MCs = [], [], [], []
    neuron_id = 0

    for g_idx in range(n_glomeruli):
        pref_mol = int(mol_map[g_idx % n_odor_dims])
        glom_to_preferred_mol[g_idx] = pref_mol

        # Gaussian tuning vector over molecule space
        mu = np.zeros(n_odor_dims)
        for mol in range(n_odor_dims):
            lin_d  = abs(pref_mol - mol)
            circ_d = min(lin_d, n_odor_dims - lin_d)
            mu[mol] = np.exp(-circ_d**2 / (2.0 * osn_sigma**2))

        glom = {'id': g_idx, 'OSNs': [], 'PGs': [], 'MCs': []}

        # OSNs
        for _ in range(n_OSNs_per_glom):
            osn = Neuron(neuron_id, pars_OSN, is_OSN=True,
                         mu=mu.copy(), sigma=osn_sigma)
            osn.glomerulus_id = g_idx
            glom['OSNs'].append(osn)
            all_OSNs.append(osn)
            neuron_id += 1

        pg = Neuron(neuron_id, pars_PG)
        pg.glomerulus_id = g_idx
        glom['PGs'].append(pg)
        all_PGs.append(pg)
        neuron_id += 1

        mc = TCNeuron(neuron_id, pars_MC, g_c=9.0)
        mc.glomerulus_id = g_idx
        mc.I_baseline = 0.00 # maybe add noise,,,,
        glom['MCs'].append(mc)
        all_MCs.append(mc)
        neuron_id += 1

        glomeruli.append(glom)

    print(f"  Created {len(all_OSNs)} OSNs, {len(all_PGs)} PGs, {len(all_MCs)} MCs")

    # within column wiring
    for glom in glomeruli:
        OSNs = glom['OSNs']
        PGs  = glom['PGs']
        MCs  = glom['MCs']

        # OSN -> PG  (excitatory)
        for osn in OSNs:
            for pg in PGs:
                pg.add_synapse(osn,
                               weight   = syn['OSN_PG']['w'],
                               E_syn    = syn['OSN_PG']['E_syn'],
                               tau_rise = syn['OSN_PG']['tau_rise'],
                               tau_fall = syn['OSN_PG']['tau_fall'])

        # OSN -> MC (apical)
        for mc in MCs:
            for osn in OSNs:
                mc.add_dend_synapses(osn,
                                     weight   = syn['OSN_MC']['w'],
                                     E_syn    = syn['OSN_MC']['E_syn'],
                                     tau_rise = syn['OSN_MC']['tau_rise'],
                                     tau_fall = syn['OSN_MC']['tau_fall'])

        # PG -> MC (apical)
        for mc in MCs:
            for pg in PGs:
                mc.add_dend_synapses(pg,
                                     weight   = syn['PG_MC']['w'],
                                     E_syn    = syn['PG_MC']['E_syn'],
                                     tau_rise = syn['PG_MC']['tau_rise'],
                                     tau_fall = syn['PG_MC']['tau_fall'])

    print(" Network made.\n" + "=" * 50)

    return {
        'glomeruli':             glomeruli,
        'OSNs':                  all_OSNs,
        'PGs':                   all_PGs,
        'MCs':                   all_MCs,
        'glom_to_preferred_mol': glom_to_preferred_mol,
        'pars':   {'OSN': pars_OSN, 'PG': pars_PG, 'MC': pars_MC},
        'syn_pars': syn
    }