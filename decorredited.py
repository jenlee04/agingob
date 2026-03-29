import numpy as np
from odorcreation import continuous_odor


def normalized_overlap(v1: np.ndarray, v2: np.ndarray) -> float:
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    # account for very small case to not break
    if norm < 1e-12:
        return 0.0
    return float(np.dot(v1, v2) / norm)


def decorrelation_index(osn_v1: np.ndarray, osn_v2: np.ndarray,
                        mc_v1:  np.ndarray, mc_v2:  np.ndarray) -> float:
     # taken from 2012 paper
    osn_ov = normalized_overlap(osn_v1, osn_v2)
    mc_ov  = normalized_overlap(mc_v1,  mc_v2)
    if osn_ov < 1e-12:
        return 0.0
    return 1.0 - mc_ov / osn_ov


# pop vec helper, ignores things that are very small 
def _pop_vector(avg_rates, neurons, n_glom, rate_floor=1.0):
    by_glom = {g: [] for g in range(n_glom)}
    for rate, neuron in zip(avg_rates, neurons):
        r = rate if rate >= rate_floor else 0.0
        by_glom[neuron.glomerulus_id].append(r)
    return np.array([
        np.mean(by_glom[g]) if by_glom[g] else 0.0
        for g in range(n_glom)
    ])


# single odor simulatin helper

def _simulate_one_odor(odor_vec, model, T=1.0, dt=1e-3,
                        stim_s=0.07, stim_dur=0.5, n_trials=1,
                        compute_lfp=False):
    OSNs   = model['OSNs']
    PGs    = model['PGs']
    MCs    = model['MCs']
    n_glom = len(model['glomeruli'])
    stim_e = stim_s + stim_dur
    time   = np.arange(0, T, dt)

    all_mc_rates = []
    exc_current_trace = np.zeros(len(time))
    inh_current_trace = np.zeros(len(time))

    for _ in range(n_trials):
        # reset
        for n in OSNs + PGs + MCs:
            n.v              = getattr(n, 'V_init', n.E_L)
            n.spiked         = False
            n.lastspiketime  = -np.inf
            n.somaspiketimes = []
            n.v_trace        = []
            n.avg_rate       = 0.0
            for _, syn in getattr(n, 'input_synapses', []):
                syn.g = syn.g_rise = syn.g_fall = 0.0
            if hasattr(n, 'vsoma'):
                n.vsoma = n.E_L
                n.vdend = n.E_L
            for _, syn in getattr(n, 'dend_synapses', []):
                syn.g = syn.g_rise = syn.g_fall = 0.0
            for _, syn in getattr(n, 'soma_synapses', []):
                syn.g = syn.g_rise = syn.g_fall = 0.0

        # simulate — currents accumulated here, inside the loop
        for ti, t in enumerate(time):
            odor_now = continuous_odor(odor_vec, t, stim_s, stim_e)
            for osn in OSNs:
                osn.step(t, dt, odor_input=odor_now)
            for pg in PGs:
                pg.step(t, dt)
            for mc in MCs:
                mc.step(t, dt)
                if compute_lfp:
                    for pre, syn in mc.dend_synapses:
                        I = syn.w * syn.g * (syn.E_syn - mc.v)
                        if syn.E_syn > 0:
                            exc_current_trace[ti] += I
                        else:
                            inh_current_trace[ti] += I

        all_mc_rates.append([
            sum(stim_s <= s <= stim_e for s in mc.somaspiketimes) / stim_dur
            for mc in MCs])

    mc_rates = np.mean(all_mc_rates, axis=0)
    mc_vec   = _pop_vector(mc_rates, MCs, n_glom)

    if compute_lfp:
        delay_steps = int(0.006 / dt)
        exc_delayed = np.roll(exc_current_trace, delay_steps)
        exc_delayed[:delay_steps] = 0
        raw_lfp = exc_delayed - 1.65 * inh_current_trace
        std = raw_lfp.std()
        lfp = (raw_lfp - raw_lfp.mean()) / std if std > 1e-12 else raw_lfp
        return mc_vec, lfp

    return mc_vec