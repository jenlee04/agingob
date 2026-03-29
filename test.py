import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from parstouse import age_pars, apply_age_pars, synapse_pars


from odorcreation import create_specific_odors, continuous_odor, sniffing
from compare import comparisons
from net import Network
from decorredited import (
    normalized_overlap,
    decorrelation_index,
    _simulate_one_odor
)

os.makedirs('./figures', exist_ok=True)
savepath = './figures'


# HELPERS #

def _reset_neurons(all_neurons):
    for n in all_neurons:
        n.v              = getattr(n, 'V_init', n.E_L)
        n.spiked         = False
        n._just_spiked   = False
        n.spiked = False
        n.prev_spiked = False  
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

def prune_pg_mc_synapses(model, p_keep=1.0, seed=None):
    """
    Randomly remove PG->MC dendrodendritic synapses in-place.
    p_keep=1.0 keeps all (young), p_keep=0.70 removes 30% (old).
    """
    if p_keep >= 1.0:
        return
    if seed is not None:
        np.random.seed(seed)
    pg_ids = {id(pg) for pg in model['PGs']}
    for mc in model['MCs']:
        mc.dend_synapses = [
            (pre, syn) for pre, syn in mc.dend_synapses
            if id(pre) not in pg_ids or np.random.rand() < p_keep
        ]
# sim #

def run_simulation(odors, odor_names, model,
                   num_trials=20, include_sniffing=False,
                   T=1.0, dt=1e-3,
                   stim_s=0.07, stim_dur=0.5,
                   sniff_freq=8.0):
    """
    Simulate each odor for num_trials trials.

    """
    OSNs = model['OSNs']
    PGs  = model['PGs']
    MCs  = model['MCs']
    all_neurons = OSNs + PGs + MCs

    n_odors = len(odors)
    stim_e  = stim_s + stim_dur
    time    = np.arange(0, T, dt)

    OSN_counts = [[[] for _ in OSNs] for _ in range(n_odors)]
    PG_counts  = [[[] for _ in PGs]  for _ in range(n_odors)]
    MC_counts  = [[[] for _ in MCs]  for _ in range(n_odors)]

    gtm = model['glom_to_preferred_mol']

    trace_glom_ids = []
    for odor_base in odors:
        peak_mol  = int(np.argmax(odor_base))
        best_glom = min(gtm, key=lambda g: abs(gtm[g] - peak_mol))
        trace_glom_ids.append(best_glom)

    trace_OSN = [None] * n_odors
    trace_PG  = [None] * n_odors
    trace_MC  = [None] * n_odors

    trace_osn_neurons = [next(n for n in OSNs if n.glomerulus_id == gid)
                         for gid in trace_glom_ids]
    trace_pg_neurons  = [next(n for n in PGs  if n.glomerulus_id == gid)
                         for gid in trace_glom_ids]
    trace_mc_neurons  = [next(n for n in MCs  if n.glomerulus_id == gid)
                         for gid in trace_glom_ids]

    for trial in range(num_trials):
        print(f'  Trial {trial + 1}/{num_trials}')
        for oi, (odor_base, name) in enumerate(zip(odors, odor_names)):
            _reset_neurons(all_neurons)
          
            for t in time:
                odor_input = (sniffing(odor_base, t, stim_s, stim_e, sniff_freq)
                              if include_sniffing
                              else continuous_odor(odor_base, t, stim_s, stim_e))
                for osn in OSNs:
                    osn.step(t, dt, odor_input=odor_input)
                for pg in PGs:
                    pg.step(t, dt)
                for mc in MCs:
                    mc.step(t, dt)

            for i, osn in enumerate(OSNs):
                OSN_counts[oi][i].append(
                    sum(stim_s <= s <= stim_e for s in osn.somaspiketimes) / stim_dur)
            for i, pg in enumerate(PGs):
                PG_counts[oi][i].append(
                    sum(stim_s <= s <= stim_e for s in pg.somaspiketimes) / stim_dur)
            for i, mc in enumerate(MCs):
                MC_counts[oi][i].append(
                    sum(stim_s <= s <= stim_e for s in mc.somaspiketimes) / stim_dur)
            if trial == num_trials - 1:
                trace_OSN[oi] = list(trace_osn_neurons[oi].v_trace)
                trace_PG[oi]  = list(trace_pg_neurons[oi].v_trace)
                trace_MC[oi]  = list(trace_mc_neurons[oi].v_trace)
                
    avg_OSN = [[np.mean(OSN_counts[oi][i]) for i in range(len(OSNs))] for oi in range(n_odors)]
    avg_PG  = [[np.mean(PG_counts[oi][i])  for i in range(len(PGs))]  for oi in range(n_odors)]
    avg_MC  = [[np.mean(MC_counts[oi][i])  for i in range(len(MCs))]  for oi in range(n_odors)]

    return avg_OSN, avg_PG, avg_MC, trace_OSN, trace_PG, trace_MC, time, MC_counts, trace_glom_ids


# PLOTTING #

def plot_avg_firing_rates(avg_OSN, avg_PG, avg_MC, model, odor_names):
    gtm    = model['glom_to_preferred_mol']
    OSNs   = model['OSNs']
    PGs    = model['PGs']
    MCs    = model['MCs']
    colors = plt.cm.tab10(np.linspace(0, 0.6, len(odor_names)))

    sorted_gloms = sorted(gtm.items(), key=lambda kv: kv[1])
    mol_order    = [mol for _, mol in sorted_gloms]
    glom_order   = [g   for g,  _ in sorted_gloms]

    def glom_avg(avg_rates, neurons):
        rates_by_glom = {g: [] for g in glom_order}
        for i, n in enumerate(neurons):
            rates_by_glom[n.glomerulus_id].append(avg_rates[i])
        return [np.mean(rates_by_glom[g]) if rates_by_glom[g] else 0.0
                for g in glom_order]

    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    fig.suptitle('Average firing rates across all trials', fontsize=12)

    for ax, (label, neurons, avg_all) in zip(axes, [
            ('OSN', OSNs, avg_OSN),
            ('PG',  PGs,  avg_PG),
            ('MC',  MCs,  avg_MC),
    ]):
        for oi, name in enumerate(odor_names):
            ax.plot(mol_order, glom_avg(avg_all[oi], neurons),
                    color=colors[oi], lw=1.5, alpha=0.85, label=name)
        ax.set_ylabel('Mean rate (Hz)')
        ax.set_title(label)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(alpha=0.25)

    axes[-1].set_xlabel('Glomerulus preferred molecule')
    plt.tight_layout()
    fname = f'{savepath}/avg_firing_rates.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'Saved: {fname}')


# plotting funcs #

def plot_voltage_traces(trace_OSN, trace_PG, trace_MC,
                        t_vec, odor_names, model,
                        stim_s=0.07, stim_dur=0.5):
    from parstouse import specific_pars
    pars_OSN, pars_PG, pars_MC= specific_pars()

    n_odors = len(odor_names)
    stim_e  = stim_s + stim_dur

    plot_pars_osn = {'V_th': pars_OSN['theta_max'], 'dt': 1e-3,
                     'range_t': t_vec * 1e3}
    plot_pars_pg  = {'V_th': pars_PG['theta_max'],  'dt': 1e-3,
                     'range_t': t_vec * 1e3}
    plot_pars_mc  = {'V_th': pars_MC['theta_max'],  'dt': 1e-3,
                     'range_t': t_vec * 1e3}

    def plot_volt_trace(pars, v, sp):
        V_th = pars['V_th']
        dt_p, range_t = pars['dt'], pars['range_t']
        sp = np.array(sp)
        if sp.size:
            sp_num = (sp / dt_p).astype(int) - 1
            v[sp_num] += 20
        plt.plot(range_t, v, 'b')
        plt.axhline(V_th, 0, 1, color='k', ls='--')
        plt.xlabel('Time (ms)')
        plt.ylabel('V (mV)')
        plt.legend(['Membrane\npotential', r'Threshold V$_{\mathrm{th}}$'],
                   loc=[1.05, 0.75])
        plt.ylim([-80, +50])

    fig = plt.figure(figsize=(6 * n_odors, 10))
    fig.suptitle('Voltage traces (last trial)', fontsize=12)

    for oi, name in enumerate(odor_names):
        # get spike times from the traced neurons
        gtm       = model['glom_to_preferred_mol']
        peak_mol  = int(np.argmax([1.0]))   # placeholder — pass odors in if needed
        OSNs, PGs, MCs = model['OSNs'], model['PGs'], model['MCs']

        osn_st = np.array(next(n for n in OSNs
                               if n.glomerulus_id == trace_glom_ids[oi]).somaspiketimes)
        pg_st  = np.array(next(n for n in PGs
                               if n.glomerulus_id == trace_glom_ids[oi]).somaspiketimes)
        mc_st  = np.array(next(n for n in MCs
                               if n.glomerulus_id == trace_glom_ids[oi]).somaspiketimes)

        for row, (tr, st, pars, label) in enumerate(zip(
                [trace_OSN[oi], trace_PG[oi], trace_MC[oi]],
                [osn_st, pg_st, mc_st],
                [plot_pars_osn, plot_pars_pg, plot_pars_mc],
                ['OSN', 'PG', 'MC'])):
            plt.subplot(3 * n_odors, n_odors, row * n_odors + oi + 1)
            plot_volt_trace(pars, np.array(tr).copy(), st.copy())
            plt.axvspan(stim_s * 1e3, stim_e * 1e3, alpha=0.12, color='gold')
            if oi == 0:
                plt.ylabel('V (mV)')
            if row == 0:
                plt.title(name)
            plt.annotate(label, xy=(0.02, 0.88), xycoords='axes fraction',
                         fontsize=9, color=['steelblue','seagreen','firebrick'][row],
                         fontweight='bold')

    plt.tight_layout()
    fname = f'{savepath}/voltage_traces.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'Saved: {fname}')

def plot_raster(model, odors, odor_names,
                T=1.0, dt=1e-3, stim_s=0.07, stim_dur=0.5):
    stim_e  = stim_s + stim_dur
    MCs     = model['MCs']
    gtm     = model['glom_to_preferred_mol']
    n_odors = len(odors)

    sorted_mc_idx = sorted(range(len(MCs)),
                           key=lambda i: gtm[MCs[i].glomerulus_id])

    fig, axes = plt.subplots(1, n_odors, figsize=(6 * n_odors, 4), sharey=True)
    if n_odors == 1:
        axes = [axes]

    for oi, (name, odor) in enumerate(zip(odor_names, odors)):
        _reset_neurons(model['OSNs'] + model['PGs'] + MCs)
        for t in np.arange(0, T, dt):
            odor_input = continuous_odor(odor, t, stim_s, stim_e)
            for osn in model['OSNs']:
                osn.step(t, dt, odor_input=odor_input)
            for pg in model['PGs']:
                pg.step(t, dt)
            for mc in MCs:
                mc.step(t, dt)

        rx, ry = [], []
        for y_pos, mc_i in enumerate(sorted_mc_idx):
            for s in MCs[mc_i].somaspiketimes:
                if s <= T:
                    rx.append(s * 1e3)
                    ry.append(y_pos)

        ax = axes[oi]
        ax.scatter(rx, ry, s=3, c='steelblue', alpha=0.6, linewidths=0)
        ax.axvspan(stim_s * 1e3, stim_e * 1e3, alpha=0.10, color='gold')
        ax.set_xlim(0, T * 1e3)
        ax.set_ylim(-1, len(MCs))
        ax.set_xlabel('Time (ms)')
        ax.set_title(name)
        if oi == 0:
            ax.set_ylabel('MC index (sorted by pref mol)')

    fig.suptitle('MC raster', fontsize=12)
    plt.tight_layout()
    fname = f'{savepath}/raster.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'Saved: {fname}')


# metric printing #

def compute_and_print_metrics(avg_OSN, avg_PG, avg_MC, model, odor_names):
    gtm    = model['glom_to_preferred_mol']
    OSNs   = model['OSNs']
    PGs    = model['PGs']
    MCs    = model['MCs']
    n_glom = len(gtm)

    def pop_vector(avg_rates, neurons):
        by_glom = {g: [] for g in range(n_glom)}
        for i, n in enumerate(neurons):
            by_glom[n.glomerulus_id].append(avg_rates[i])
        return np.array([np.mean(by_glom[g]) if by_glom[g] else 0.0
                         for g in range(n_glom)])

    OSN_vecs = [pop_vector(avg_OSN[oi], OSNs) for oi in range(len(odor_names))]
    PG_vecs  = [pop_vector(avg_PG[oi],  PGs)  for oi in range(len(odor_names))]
    MC_vecs  = [pop_vector(avg_MC[oi],  MCs)  for oi in range(len(odor_names))]
    pairs    = list(combinations(range(len(odor_names)), 2))

    print('\n' + '=' * 65)
    print('PAIRWISE SIMILARITY METRICS')
    print('=' * 65)

    for cell_label, vecs in [('OSN', OSN_vecs), ('PG', PG_vecs), ('MC', MC_vecs)]:
        print(f'\n  [{cell_label}]')
        print(f"  {'Pair':<25} {'Euclidean':>12} {'Correlation':>14} {'Dot product':>13}")
        print(f"  {'-' * 64}")
        for i, j in pairs:
            m    = comparisons(vecs[i], vecs[j])
            pair = f'{odor_names[i]} vs {odor_names[j]}'
            print(f'  {pair:<25} '
                  f"{m['euclidean_distance']:>12.4f} "
                  f"{m['correlation']:>14.4f} "
                  f"{m['dot_product']:>13.4f}")

    print('=' * 65 + '\n')


def compute_sweep_metrics(all_mc_avg, weights, model, odor_names):
    gtm    = model['glom_to_preferred_mol']
    MCs    = model['MCs']
    n_glom = len(gtm)

    def pop_vector(avg_mc_oi):
        by_glom = {g: [] for g in range(n_glom)}
        for i, mc in enumerate(MCs):
            by_glom[mc.glomerulus_id].append(avg_mc_oi[i])
        return np.array([np.mean(by_glom[g]) if by_glom[g] else 0.0
                         for g in range(n_glom)])

    pairs = list(combinations(range(len(odor_names)), 2))

    print('\n' + '=' * 70)
    print('WEIGHT SWEEP METRICS')
    print(f"  {'weight':<10} {'pair':<25} {'eucl':>12} {'corr':>14}")
    print('=' * 70)

    for w, avg_mc in zip(weights, all_mc_avg):
        vecs = [pop_vector(avg_mc[oi]) for oi in range(len(odor_names))]
        for i, j in pairs:
            m    = comparisons(vecs[i], vecs[j])
            pair = f'{odor_names[i]} vs {odor_names[j]}'
            print(f'  {w:<10.3f} {pair:<25} '
                  f"{m['euclidean_distance']:>12.4f} "
                  f"{m['correlation']:>14.4f}")

    print('=' * 70 + '\n')


# compare dist = 2 and 5 to increasing inhibition using 2012 paper framework #
def run_distance_sweep(distances, pg_mc_weights,
                       n_glom, n_osn, n_odor, osn_sigma,
                       T, dt, stim_s, stim_dur):
    results = {}

    for dist in distances:
        model = Network(n_glomeruli=n_glom, n_OSNs_per_glom=n_osn,
                        n_odor_dims=n_odor, osn_sigma=osn_sigma)
        pg_ids = {id(pg) for pg in model['PGs']}
        gtm = model['glom_to_preferred_mol']
        n_glom_actual = len(gtm)

        for wi, w in enumerate(pg_mc_weights):
            print(f'  dist={dist:>2}  PG->MC w={w:.3f}', flush=True)
            for mc in model['MCs']:
                for pre, syn in mc.dend_synapses:
                    if id(pre) in pg_ids:
                        syn.w = w

            n_pairs = 60 
            osn_A_all, osn_B_all, mc_A_all, mc_B_all = [], [], [], []

            for pair_idx in range(n_pairs):
                anchor = np.random.randint(0, n_odor)
                mol_A  = anchor
                mol_B  = (anchor + dist) % n_odor
                odor_A = np.zeros(n_odor); odor_A[mol_A] = 1.0
                odor_B = np.zeros(n_odor); odor_B[mol_B] = 1.0

                # analytic OSN vectors for this pair
                osn_A = np.zeros(n_glom_actual)
                osn_B = np.zeros(n_glom_actual)
                for g, pref in gtm.items():
                    dA = min(abs(pref - mol_A), n_odor - abs(pref - mol_A))
                    dB = min(abs(pref - mol_B), n_odor - abs(pref - mol_B))
                    osn_A[g] = np.exp(-dA**2 / (2 * osn_sigma**2))
                    osn_B[g] = np.exp(-dB**2 / (2 * osn_sigma**2))

                mv_A = _simulate_one_odor(odor_A, model, T=T, dt=dt,
                                             stim_s=stim_s, stim_dur=stim_dur)
                mv_B = _simulate_one_odor(odor_B, model, T=T, dt=dt,
                                             stim_s=stim_s, stim_dur=stim_dur)

                osn_A_all.append(osn_A); osn_B_all.append(osn_B)
                mc_A_all.append(mv_A);   mc_B_all.append(mv_B)
                print(f'    pair {pair_idx+1}/{n_pairs} ', flush=True)

            mean_osn_ov = float(np.mean([normalized_overlap(a, b)
                               for a, b in zip(osn_A_all, osn_B_all)]))
            mean_mc_ov  = float(np.mean([normalized_overlap(a, b)
                                           for a, b in zip(mc_A_all, mc_B_all)]))
            mean_d = 1.0 - mean_mc_ov / mean_osn_ov if mean_osn_ov > 1e-12 else 0.0
            print(f'  OSN_ov={mean_osn_ov:.3f}  MC_ov={mean_mc_ov:.3f}  D={mean_d:+.2%}')

            results[(dist, w)] = {
                'decorrelation_index': mean_d,
                'osn_overlap':  mean_osn_ov,
                'mc_overlap':   mean_mc_ov,
                'osn_vec_A': np.mean(osn_A_all, axis=0),
                'osn_vec_B': np.mean(osn_B_all, axis=0),
                'mc_vec_A':  np.mean(mc_A_all,  axis=0),
                'mc_vec_B':  np.mean(mc_B_all,  axis=0),
                'mol_A': None, 'mol_B': None,
            }
        baseline_d = results[(dist, pg_mc_weights[0])]['decorrelation_index']
        for w in pg_mc_weights:
            results[(dist, w)]['delta_DI'] = (
                results[(dist, w)]['decorrelation_index'] - baseline_d
            )

          
    return results

def measure_detection(model, odor, concentration=1.0, n_trials=30,
                      T=1.5, dt=1e-3,
                      stim_s=0.5, stim_dur=0.5):
    OSNs   = model['OSNs']
    PGs    = model['PGs']
    MCs    = model['MCs']
    stim_e = stim_s + stim_dur
    time   = np.arange(0, T, dt)
    scaled_odor = odor * concentration

    odor_norms = []
    spont_norms = []

    for _ in range(n_trials):
        _reset_neurons(OSNs + PGs + MCs)
        for t in time:
            odor_input = continuous_odor(scaled_odor, t, stim_s, stim_e)
            for osn in OSNs: osn.step(t, dt, odor_input=odor_input)
            for pg in PGs:   pg.step(t, dt)
            for mc in MCs:   mc.step(t, dt)

        # use same duration for both windows to avoid normalization artifact
        pre_rates = np.array([
            sum(0.0 <= s < stim_s for s in mc.somaspiketimes) / stim_s
            for mc in MCs])
        stim_rates = np.array([
            sum(stim_s <= s <= stim_e for s in mc.somaspiketimes) / stim_dur
            for mc in MCs])

        odor_norms.append(np.linalg.norm(stim_rates))
        spont_norms.append(np.linalg.norm(pre_rates))

    odor_norms  = np.array(odor_norms)
    spont_norms = np.array(spont_norms)

    pooled_std = np.sqrt((np.var(odor_norms) + np.var(spont_norms)) / 2.0)
    print(f"  conc={concentration:.4f}  odor_mean={odor_norms.mean():.3f}"
          f"  spont_mean={spont_norms.mean():.3f}")

    if pooled_std < 1e-9:
        return 0.0
    return (odor_norms.mean() - spont_norms.mean()) / pooled_std

from scipy import stats

def detection_threshold_sweep(model, base_odor, concentrations,
                               n_trials=30, T=2.0, dt=1e-3,
                               stim_s=0.5, stim_dur=0.5):
    """
    For each concentration, runs n_trials and projects each trial's MC
    population response onto the mean odor response axis (Option 3).
    This suppresses spontaneous noise from non-responding glomeruli and
    gives a more sensitive detection statistic than the L2 norm.

    Uses a paired t-test (stim vs pre window, same trial) rather than
    independent samples, since both windows come from the same simulation.
    """
    OSNs, PGs, MCs = model['OSNs'], model['PGs'], model['MCs']
    stim_e = stim_s + stim_dur
    time   = np.arange(0, T, dt)

    p_values  = []
    threshold = None
    consec    = 0  # consecutive significant concentrations

    for conc in concentrations:
        odor_scaled = base_odor * conc

        stim_rates_list = []
        pre_rates_list  = []

        for _ in range(n_trials):
            _reset_neurons(OSNs + PGs + MCs)
            for t in time:
                odor_input = continuous_odor(odor_scaled, t, stim_s, stim_e)
                for osn in OSNs: osn.step(t, dt, odor_input=odor_input)
                for pg in PGs:   pg.step(t, dt)
                for mc in MCs:   mc.step(t, dt)

            pre_rates = np.array([
                sum(0.0 <= s < stim_s for s in mc.somaspiketimes) / stim_s
                for mc in MCs])
            stim_rates = np.array([
                sum(stim_s <= s <= stim_e for s in mc.somaspiketimes) / stim_dur
                for mc in MCs])

            stim_rates_list.append(stim_rates)
            pre_rates_list.append(pre_rates)

        # compute mean odor response axis from stim window across trials
        mean_stim = np.mean(stim_rates_list, axis=0)
        axis_norm = np.linalg.norm(mean_stim)
        if axis_norm < 1e-12:
            # no response at all — skip projection, p=1
            p_values.append(1.0)
            print(f"  conc={conc:.4f}  no response  p=1.0000")
            consec = 0
            continue

        axis = mean_stim / axis_norm  # unit vector along mean odor response

        # project each trial's stim and pre windows onto this axis
        stim_proj = np.array([r @ axis for r in stim_rates_list])
        pre_proj  = np.array([r @ axis for r in pre_rates_list])

        # paired t-test: within each trial, did stim exceed pre?
        t_stat, p_val = stats.ttest_rel(stim_proj, pre_proj)
        p_values.append(p_val)

        print(f"  conc={conc:.4f}  odor_proj={stim_proj.mean():.3f}"
              f"  spont_proj={pre_proj.mean():.3f}  p={p_val:.4f}")

        # require two consecutive significant concentrations to declare threshold
        for ci, conc in enumerate(concentrations):
            if p_val < 0.05:
                consec += 1
                if consec >= 2 and threshold is None:
                    threshold = concentrations[ci - 1]  # first of the two consecutive
            else:
                consec = 0

    return p_values, threshold


def plot_detection_threshold(concentrations, young_pvals, old_pvals,
                              young_thresh, old_thresh):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(concentrations, young_pvals, 'o-', color='steelblue', lw=2, label='Young')
    ax.plot(concentrations, old_pvals,   'o-', color='firebrick', lw=2, label='Old')
    ax.axhline(0.05, color='gray', ls='--', lw=1.2, label='p=0.05')
    if young_thresh:
        ax.axvline(young_thresh, color='steelblue', ls=':', lw=1.5,
                   label=f'Young thresh={young_thresh}')
    if old_thresh:
        ax.axvline(old_thresh, color='firebrick', ls=':', lw=1.5,
                   label=f'Old thresh={old_thresh}')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Odor concentration (scale factor)')
    ax.set_ylabel('p-value (t-test vs blank)')
    ax.set_title('Detection threshold: Young vs Old')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    fname = f'{savepath}/detection_threshold.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'Saved: {fname}')

def measure_discrimination(model, odor_A, odor_B, n_trials=30,
                           T=1.5, dt=1e-3,
                           stim_s=0.5, stim_dur=0.5):
    """
    d' between MC population responses to odor_A vs odor_B.
    Uses projection onto the mean difference vector.
    """
    OSNs = model['OSNs']
    PGs  = model['PGs']
    MCs  = model['MCs']
    stim_e = stim_s + stim_dur
    time   = np.arange(0, T, dt)

    def run_trials(odor_vec):
        pop_vecs = []
        for _ in range(n_trials):
            _reset_neurons(OSNs + PGs + MCs)
            for t in time:
                odor_input = continuous_odor(odor_vec, t, stim_s, stim_e)
                for osn in OSNs: osn.step(t, dt, odor_input=odor_input)
                for pg in PGs:   pg.step(t, dt)
                for mc in MCs:   mc.step(t, dt)
            mc_rates = np.array([
                sum(stim_s <= s <= stim_e for s in mc.somaspiketimes) / stim_dur
                for mc in MCs])
            pop_vecs.append(mc_rates)
        return np.array(pop_vecs)  # shape (n_trials, n_MCs)

    vecs_A = run_trials(odor_A)
    vecs_B = run_trials(odor_B)

    mean_diff = np.mean(vecs_A, axis=0) - np.mean(vecs_B, axis=0)
    norm = np.linalg.norm(mean_diff)
    if norm < 1e-9:
        return 0.0
    axis = mean_diff / norm

    proj_A = vecs_A @ axis
    proj_B = vecs_B @ axis

    pooled_std = np.sqrt((np.var(proj_A) + np.var(proj_B)) / 2.0)
    if pooled_std < 1e-9:
        return 0.0

    return (np.mean(proj_A) - np.mean(proj_B)) / pooled_std


def run_detection_discrimination(model, distances, pg_mc_weight,
                                  n_odor=100, n_trials=30,
                                  T=1.0, dt=1e-3,
                                  stim_s=0.07, stim_dur=0.5):
    """
    For a fixed PG->MC weight, measure detection and discrimination
    across odor distances. Sets the weight on the model in-place.
    """
    pg_ids = {id(pg) for pg in model['PGs']}
    for mc in model['MCs']:
        for pre, syn in mc.dend_synapses:
            if id(pre) in pg_ids:
                syn.w = pg_mc_weight

    anchor = n_odor // 2  # use center molecule as reference
    odor_A = np.zeros(n_odor)
    odor_A[anchor] = 1.0

    det = measure_detection(model, odor_A,
                        concentration=1.0,
                        n_trials=n_trials,
                        T=1.5,
                        stim_s=0.5,
                        stim_dur=0.5)
    print(f"Detection d' = {det:.3f}")

    discrim_results = {}
    for dist in distances:
        odor_B = np.zeros(n_odor)
        odor_B[(anchor + dist) % n_odor] = 1.0
        d = measure_discrimination(model, odor_A, odor_B,
                                    n_trials=n_trials, T=T, dt=dt,
                                    stim_s=stim_s, stim_dur=stim_dur)
        discrim_results[dist] = d
        print(f"Discrimination d' dist={dist}: {d:.3f}")

    return det, discrim_results

# plots 
def plot_distance_sweep(results, distances, pg_mc_weights):
    """
    three plots for prev func
    1. decorr index % vs PG -> MC weight
    2. OSN overlap vs MC overlap (points below diag = good, decorr)
    3. OSN overlap and MC overlap vs PG -> MC weight
    """
    dist_colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(distances)))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f'distance and PG->MC inhibition sweep \n'
        f'D = 1 - (MC overlap / OSN overlap)',
        fontsize=12
    )

    # first #
    ax = axes[0]
    for ci, dist in enumerate(distances):
        d_vals = [results[(dist, w)]['decorrelation_index'] * 100
                  for w in pg_mc_weights]
        ax.plot(pg_mc_weights, d_vals, 'o-', lw=2, ms=6,
                color=dist_colors[ci], label=f'dist={dist}')
    ax.axhline(0, color='gray', lw=0.8, ls=':')
    ax.set_xlabel('PG->MC weight')
    ax.set_ylabel('Decorrelation index (%)')
    ax.set_title('Decorrelation index')
    ax.legend(title='Odor distance', fontsize=8)
    ax.grid(alpha=0.25)

    # second#
    ax = axes[1]
    all_ov = ([results[k]['osn_overlap'] for k in results] +
              [results[k]['mc_overlap']  for k in results])
    lim = max(all_ov) * 1.08

    for ci, dist in enumerate(distances):
        osn_ovs = [results[(dist, w)]['osn_overlap'] for w in pg_mc_weights]
        mc_ovs  = [results[(dist, w)]['mc_overlap']  for w in pg_mc_weights]
        sizes   = [20 + 60 * i / max(len(pg_mc_weights) - 1, 1)
                   for i in range(len(pg_mc_weights))]
        ax.scatter(osn_ovs, mc_ovs,
                   c=[dist_colors[ci]] * len(pg_mc_weights),
                   s=sizes, alpha=0.85, label=f'dist={dist}',
                   edgecolors='white', linewidths=0.4)

    ax.plot([0, lim], [0, lim], 'k--', lw=0.8, alpha=0.4, label='no change')
    ax.set_xlabel('OSN overlap (input)')
    ax.set_ylabel('MC overlap (output)')
    ax.set_title('Below diagonal = decorrelated\n')
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    # third# 
    ax = axes[2]
    for ci, dist in enumerate(distances):
        osn_ovs = [results[(dist, w)]['osn_overlap'] for w in pg_mc_weights]
        mc_ovs  = [results[(dist, w)]['mc_overlap']  for w in pg_mc_weights]
        ax.plot(pg_mc_weights, osn_ovs, '--', lw=1.5, color=dist_colors[ci],
                alpha=0.6)
        ax.plot(pg_mc_weights, mc_ovs,  '-',  lw=2,   color=dist_colors[ci],
                label=f'dist={dist}')
    ax.set_xlabel('PG->MC weight')
    ax.set_ylabel('Overlap')
    ax.set_title('Overlap vs weight\nsolid=MC output, dashed=OSN input')
    ax.legend(title='Odor distance', fontsize=8)
    ax.grid(alpha=0.25)

    plt.tight_layout()
    fname = f'{savepath}/distance_pgmc_sweep.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'Saved: {fname}')


def plot_tuning_curves_grid(results, distances, pg_mc_weights, 
                            n_odor, zoom_radius=15):
    """
    Since mc_vec_A/B are now averaged across random pairs with different
    anchors, individual tuning curves are not meaningful here.
    Instead plot D index and MC overlap as a grid summary.
    """
    n_dist = len(distances)
    n_w    = len(pg_mc_weights)
    dist_colors = plt.cm.viridis(np.linspace(0.1, 0.9, n_dist))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f'Decorrelation sweep summary\n'
        f'60 random pairs per condition, dist fixed per block',
        fontsize=12
    )

    # Left: D index vs weight
    ax = axes[0]
    for ci, dist in enumerate(distances):
        d_vals = [results[(dist, w)]['decorrelation_index'] * 100
                  for w in pg_mc_weights]
        ax.plot(pg_mc_weights, d_vals, 'o-', lw=2, ms=6,
                color=dist_colors[ci], label=f'dist={dist}')
    ax.axhline(0, color='gray', lw=0.8, ls=':')
    ax.set_xlabel('PG->MC weight')
    ax.set_ylabel('Mean decorrelation index (%)')
    ax.set_title('D = 1 - (MC overlap / OSN overlap)')
    ax.legend(title='Odor distance', fontsize=8)
    ax.grid(alpha=0.25)

    # Right: MC overlap vs weight
    ax = axes[1]
    for ci, dist in enumerate(distances):
        osn_ovs = [results[(dist, w)]['osn_overlap'] for w in pg_mc_weights]
        mc_ovs  = [results[(dist, w)]['mc_overlap']  for w in pg_mc_weights]
        ax.plot(pg_mc_weights, osn_ovs, '--', lw=1.5,
                color=dist_colors[ci], alpha=0.6)
        ax.plot(pg_mc_weights, mc_ovs, '-', lw=2,
                color=dist_colors[ci], label=f'dist={dist}')
    ax.set_xlabel('PG->MC weight')
    ax.set_ylabel('Overlap')
    ax.set_title('Solid=MC overlap, dashed=OSN overlap')
    ax.legend(title='Odor distance', fontsize=8)
    ax.grid(alpha=0.25)

    plt.tight_layout()
    fname = f'{savepath}/tuning_curves_grid.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'Saved: {fname}')

def plot_aging_complexity(results, complexity_names, complexity_labels):
    """
    Grouped bar chart: young vs old, one bar per odor, colored by complexity.
    """
    x      = np.arange(len(complexity_names))
    width  = 0.35
    colors = {'low': 'steelblue', 'medium': 'darkorange', 'high': 'firebrick'}

    fig, ax = plt.subplots(figsize=(max(8, len(complexity_names) * 0.9), 5))

    young_vals = [results['young'][n] for n in complexity_names]
    old_vals   = [results['old'][n]   for n in complexity_names]
    bar_colors = [colors[c] for c in complexity_labels]

    ax.bar(x - width/2, young_vals, width, label='Young',
           color=bar_colors, alpha=0.9, edgecolor='white')
    ax.bar(x + width/2, old_vals,   width, label='Old',
           color=bar_colors, alpha=0.45, edgecolor='gray', linestyle='--',
           linewidth=1.0)

    ax.set_xticks(x)
    ax.set_xticklabels(complexity_names, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel("Detection d'")
    ax.set_title("Detection d' by odor complexity: Young vs Old\n"
                 "(solid=young, faded=old; color=complexity)")
    ax.legend()

    # add a complexity legend patch
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[c], label=c.capitalize())
                       for c in ['low', 'medium', 'high']]
    ax.legend(handles=legend_elements + [
        Patch(facecolor='gray', alpha=0.9, label='Young'),
        Patch(facecolor='gray', alpha=0.4, label='Old')
    ], fontsize=8)

    ax.grid(alpha=0.2, axis='y')
    plt.tight_layout()
    fname = f'{savepath}/aging_complexity.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'Saved: {fname}')

# diag to see if inhib will work or not #

def diagnose_flank_firing(model, center_mol=50, sweep_radius=10,
                          n_odor=100, n_trials=10,
                          T=1.0, dt=1e-3,
                          stim_s=0.07, stim_dur=0.5):
    """
    Run a single pure odor at center_mol for n_trials and print average
    PG vs MC firing rates at each glomerulus within sweep_radius.

    Target pattern for the Mexican-hat mechanism to work:
        center glom : PG/MC ratio ~1 (both firing similarly)
        flank gloms : PG/MC ratio > 1 (PG dominates over weak MC)
    """
    odor   = np.zeros(n_odor)
    odor[center_mol] = 1.0

    OSNs   = model['OSNs']
    PGs    = model['PGs']
    MCs    = model['MCs']
    gtm    = model['glom_to_preferred_mol']
    stim_e = stim_s + stim_dur
    time   = np.arange(0, T, dt)

    # accumulate by glomerulus index
    pg_rates = {g: [] for g in gtm}
    mc_rates = {g: [] for g in gtm}

    for trial in range(n_trials):
        _reset_neurons(OSNs + PGs + MCs)                                
        for t in time:
            odor_input = continuous_odor(odor, t, stim_s, stim_e)
            for osn in OSNs:
                osn.step(t, dt, odor_input=odor_input)
            for pg in PGs:
                pg.step(t, dt)
            for mc in MCs:
                mc.step(t, dt)
    
        for g in gtm:
            pg_n = next(n for n in PGs if n.glomerulus_id == g)
            mc_n = next(n for n in MCs if n.glomerulus_id == g)
            pg_rates[g].append(
                sum(stim_s <= s <= stim_e for s in pg_n.somaspiketimes) / stim_dur)
            mc_rates[g].append(
                sum(stim_s <= s <= stim_e for s in mc_n.somaspiketimes) / stim_dur)

    # print sorted by molecule index within sweep_radius
    lo = max(0, center_mol - sweep_radius)
    hi = min(n_odor - 1, center_mol + sweep_radius)

    print(f"\n{'Mol':>5} {'PG (Hz)':>10} {'MC (Hz)':>10} {'PG/MC':>8}")
    print('-' * 38)

    for mol in range(lo, hi + 1):
        g = next((g for g, m in gtm.items() if m == mol), None)
        if g is None:
            continue
        pg_r = np.mean(pg_rates[g])
        mc_r = np.mean(mc_rates[g])
        ratio  = pg_r / mc_r if mc_r > 0 else float('inf')
        dist   = abs(mol - center_mol)
        marker = ' <-- flank' if 1 <= dist <= 3 else (
                 ' <-- CENTER' if dist == 0 else '')
        print(f"{mol:>5} {pg_r:>10.1f} {mc_r:>10.1f} {ratio:>8.2f}{marker}")

# isolation sweep helpers

def run_discrimination(model, distances=[4, 6, 8, 12],
                       anchor_mol=50, n_odor=100, n_trials=30,
                       T=1.0, dt=1e-3, stim_s=0.07, stim_dur=0.5):
    """
    Run discrimination d' for each distance using the current model.
    Wraps the existing measure_discrimination function.
    Returns dict {dist: d_prime}
    """
    odor_A = np.zeros(n_odor)
    odor_A[anchor_mol] = 1.0

    d_scores = {}
    for dist in distances:
        odor_B = np.zeros(n_odor)
        odor_B[(anchor_mol + dist) % n_odor] = 1.0
        d = measure_discrimination(model, odor_A, odor_B,
                                   n_trials=n_trials, T=T, dt=dt,
                                   stim_s=stim_s, stim_dur=stim_dur)
        d_scores[dist] = d
        print(f"  Discrimination d' dist={dist}: {d:.3f}")
    return d_scores


def plot_isolation_sweep(isolation_results, distances, young_results=None):
    """
    Bar chart: one group of bars per aging parameter, one bar per distance.
    If young_results provided, draws a horizontal dashed line for young baseline.

    Parameters
    ----------
    isolation_results : dict  {label: {dist: d_prime}}
    distances         : list of int
    young_results     : dict  {dist: d_prime}  -- young baseline (optional)
    """
    labels  = list(isolation_results.keys())
    n_conds = len(labels)
    n_dist  = len(distances)
    x       = np.arange(n_conds)
    width   = 0.8 / n_dist
    colors  = plt.cm.viridis(np.linspace(0.15, 0.85, n_dist))

    fig, ax = plt.subplots(figsize=(max(10, n_conds * 1.4), 5))

    for i, dist in enumerate(distances):
        vals = [isolation_results[lbl][dist] for lbl in labels]
        bars = ax.bar(x + i * width, vals, width,
                      label=f'dist={dist}', color=colors[i], alpha=0.85,
                      edgecolor='white', linewidth=0.5)

    # young baseline horizontal lines
    if young_results is not None:
        for i, dist in enumerate(distances):
            ax.axhline(young_results[dist], color=colors[i],
                       lw=1.2, ls='--', alpha=0.6)

    ax.set_xticks(x + width * (n_dist - 1) / 2)
    ax.set_xticklabels(labels, rotation=40, ha='right', fontsize=9)
    ax.set_ylabel("Discrimination d'")
    ax.set_title("Contribution of each aging parameter to discrimination loss\n"
                 "(dashed lines = young baseline)")
    ax.legend(title='Odor distance', fontsize=8)
    ax.grid(alpha=0.2, axis='y')
    plt.tight_layout()
    fname = f'{savepath}/isolation_sweep.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'Saved: {fname}')


# aging stuff

def run_aging_comparison(distances, pg_mc_weight,
                         n_glom=100, n_osn_young=5, n_osn_old=3,
                         n_odor=100, osn_sigma=4.0,
                         n_trials=30, T=1.0, dt=1e-3,
                         stim_s=0.07, stim_dur=0.5):

    results = {}

    for age_group in ['young', 'old']:
        print(f'\n── {age_group} ──')
        ap = age_pars(age_group)
        pars_OSN, pars_PG, pars_MC, syn = apply_age_pars(ap)

        # override PG->MC with your chosen baseline weight, then scale it
        syn['PG_MC']['w'] = pg_mc_weight * ap['w_PG_MC_scale']

        n_osn = ap['n_OSNs_per_glom']
        osn_sigma_scaled = osn_sigma * ap['osn_sigma_scale']

        model = Network(
            n_glomeruli=n_glom,
            n_OSNs_per_glom=n_osn,
            n_odor_dims=n_odor,
            osn_sigma=osn_sigma_scaled,
            pars_OSN=pars_OSN, pars_PG=pars_PG, pars_MC=pars_MC,
            syn=syn
        )
        prune_pg_mc_synapses(model, p_keep=ap['p_PG_MC'])


        # set PG->MC weight on synapses in-place
        pg_ids = {id(pg) for pg in model['PGs']}
        for mc in model['MCs']:
            for pre, s in mc.dend_synapses:
                if id(pre) in pg_ids:
                    s.w = syn['PG_MC']['w']

        # detection
        anchor = n_odor // 2
        odor_A = np.zeros(n_odor); odor_A[anchor] = 1.0
        det = measure_detection(model, odor_A,
                        n_trials=n_trials,
                        T=1.5,
                        stim_s=0.5,
                        stim_dur=0.5)
        print(f"  Detection d' = {det:.3f}")

        # discrimination across distances
        discrim = {}
        for dist in distances:
            odor_B = np.zeros(n_odor)
            odor_B[(anchor + dist) % n_odor] = 1.0
            d = measure_discrimination(model, odor_A, odor_B,
                                       n_trials=n_trials, T=T, dt=dt,
                                       stim_s=stim_s, stim_dur=stim_dur)
            discrim[dist] = d
            print(f"  Discrimination d' dist={dist}: {d:.3f}")

        results[age_group] = {'detection': det, 'discrimination': discrim}

    return results

def run_aging_complexity(complexity_odors, complexity_names, complexity_labels,
                         pg_mc_weight=0.4,
                         n_glom=100, n_odor=100, osn_sigma=4.0,
                         n_trials=30, T=1.5, dt=1e-3,
                         stim_s=0.5, stim_dur=0.5):
    """
    Run detection d' for each odor under young and old conditions.
    complexity_odors  : list of odor vectors
    complexity_names  : list of string labels per odor
    complexity_labels : list of 'low'/'medium'/'high' per odor
    """
    results = {}   # {age_group: {odor_name: d_prime}}

    for age_group in ['young', 'old']:
        print(f'\n── {age_group} ──')
        ap = age_pars(age_group)
        pars_OSN, pars_PG, pars_MC, syn = apply_age_pars(ap)
        syn['PG_MC']['w'] = pg_mc_weight * ap['w_PG_MC_scale']

        n_osn = ap['n_OSNs_per_glom']
        osn_sigma_scaled = osn_sigma * ap['osn_sigma_scale']

        model = Network(
            n_glomeruli=n_glom,
            n_OSNs_per_glom=n_osn,
            n_odor_dims=n_odor,
            osn_sigma=osn_sigma_scaled,
            pars_OSN=pars_OSN, pars_PG=pars_PG, pars_MC=pars_MC,
            syn=syn
        )
        # apply PG->MC weight in-place
        pg_ids = {id(pg) for pg in model['PGs']}
        for mc in model['MCs']:
            for pre, s in mc.dend_synapses:
                if id(pre) in pg_ids:
                    s.w = syn['PG_MC']['w']

        results[age_group] = {}
        for odor, name in zip(complexity_odors, complexity_names):
            det = measure_detection(model, odor, n_trials=n_trials,
                                    T=T, dt=dt, stim_s=stim_s, stim_dur=stim_dur)
            results[age_group][name] = det
            print(f"  {name:>12}  detection d' = {det:.3f}")

    return results

####################################### MAIN #####################################################

if __name__ == '__main__':
    np.random.seed(10)

    N_GLOM     = 100
    N_ODOR     = 100
    N_OSN      = 5
    OSN_SIGMA  = 4.0
    NUM_TRIALS = 20
    
    # toggle these values to run different evaluations of the model 
    RUN_DIAG           = False # this is used to determine the best value for the inhibitory weight
    RUN_MAIN_SIM       = False
    RUN_DISTANCE_SWEEP = False
    RUN_AGING          = False
    RUN_ISOLATION_SWEEP = False
    RUN_AGING_COMPLEXITY = False
    RUN_DETECTION_THRESHOLD = False
    RUN_DETECTION_THRESHOLD_ONEBYONE = True

    print('Building network...')
    model = Network(n_glomeruli=N_GLOM, n_OSNs_per_glom=N_OSN,
                    n_odor_dims=N_ODOR, osn_sigma=OSN_SIGMA)

    odors, complexity, names = create_specific_odors(
        n_odor_dims=N_ODOR, n_low=1, n_med=0, n_high=0)
    print(f'Odors: {names}')
    syn = synapse_pars()
    print(f"OSN_PG w={syn['OSN_PG']['w']} tau_fall={syn['OSN_PG']['tau_fall']}")
    print(f"OSN_MC w={syn['OSN_MC']['w']} tau_fall={syn['OSN_MC']['tau_fall']}")



    if RUN_DIAG:
        print('\n── Flank firing diagnostic ──')
        print(model['syn_pars']['PG_MC']['w'])
        diagnose_flank_firing(model, center_mol=50, n_trials=50,
                              n_odor=N_ODOR)

    if RUN_MAIN_SIM:
        print(f'\nRunning {NUM_TRIALS} trials x {len(odors)} odors...')
        avg_OSN, avg_PG, avg_MC, \
        trace_OSN, trace_PG, trace_MC, t_vec, _, trace_glom_ids = run_simulation(
            odors, names, model,
            num_trials=NUM_TRIALS,
            T=1.0, dt=1e-3, stim_s=0.07, stim_dur=0.5,
        )
        plot_avg_firing_rates(avg_OSN, avg_PG, avg_MC, model, names)
        plot_voltage_traces(trace_OSN, trace_PG, trace_MC, t_vec, names, model)
        plot_raster(model, odors, names,
            T=1.0, dt=1e-3, stim_s=0.07, stim_dur=0.5)
        compute_and_print_metrics(avg_OSN, avg_PG, avg_MC, model, names)

    if RUN_DISTANCE_SWEEP:
        DISTANCES     = [4, 6, 8, 12]
        PG_MC_WEIGHTS = [0.0, 0.01, 0.05, 0.1, 0.2, 0.4, 0.5, 0.8]

        print('\nRunning distance vs PG->MC weight sweep...')
        results = run_distance_sweep(
                distances     = DISTANCES,
                pg_mc_weights = PG_MC_WEIGHTS,
                n_glom=N_GLOM, n_osn=N_OSN, n_odor=N_ODOR, osn_sigma=OSN_SIGMA,
                T=1.0, dt=1e-3, stim_s=0.07, stim_dur=0.5,
            )

        with open(f'{savepath}/distance_pgmc_sweep.pkl', 'wb') as f:
            pickle.dump({'results':       results,
                         'distances':     DISTANCES,
                         'pg_mc_weights': PG_MC_WEIGHTS
                        }, f)

        plot_distance_sweep(results, DISTANCES, PG_MC_WEIGHTS)
        plot_tuning_curves_grid(results, DISTANCES, PG_MC_WEIGHTS,
                                 N_ODOR)
    if RUN_AGING: 
        aging_results = run_aging_comparison(
        distances=[4, 6, 8, 12],
        pg_mc_weight=0.4,       # chosen young baseline weight
        n_glom=N_GLOM,
        n_odor=N_ODOR,
        osn_sigma=OSN_SIGMA,
        n_trials=30
    )
    # ── Single-parameter isolation sweep ──────────────────────────────────────────
    if RUN_ISOLATION_SWEEP:
        DISTANCES_IS = [4, 6, 8, 12]
        PG_MC_W      = 0.4
        N_TRIALS_IS  = 30

        old = age_pars('old')

        isolations = [
        ('OSN loss',           {'n_OSNs_per_glom':       3}),
        ('w_OSN_PG',           {'w_OSN_PG_scale':        old['w_OSN_PG_scale']}),
        ('w_OSN_MC',           {'w_OSN_MC_scale':        old['w_OSN_MC_scale']}),
        ('w_PG_MC',            {'w_PG_MC_scale':         old['w_PG_MC_scale']}),
        ('p_PG_MC (synap den. red)',   {'p_PG_MC':       old['p_PG_MC']}),   
        ('OSN rmax',           {'osn_rmax_scale':        old['osn_rmax_scale']}),
        ('OSN threshold',      {'osn_theta_max_shift':   old['osn_theta_max_shift']}),
        ('MC tau',             {'mc_tau_scale':          old['mc_tau_scale']}),
        ('noise',              {'noise_scale':           old['noise_scale']}),
    ]

        # young baseline
        print('\n── young baseline ──')
        ap_y = age_pars('young')
        pars_OSN_y, pars_PG_y, pars_MC_y, syn_y = apply_age_pars(ap_y)
        syn_y['PG_MC']['w'] = PG_MC_W
        model_y = Network(n_glomeruli=N_GLOM, n_OSNs_per_glom=5,
                          n_odor_dims=N_ODOR, osn_sigma=OSN_SIGMA,
                          pars_OSN=pars_OSN_y, pars_PG=pars_PG_y,
                          pars_MC=pars_MC_y, syn=syn_y)
        young_discrim = run_discrimination(model_y, distances=DISTANCES_IS,
                                           n_odor=N_ODOR, n_trials=N_TRIALS_IS)

        # one parameter changed at a time
        isolation_results = {}
        for label, override in isolations:
            print(f'\n── {label} ──')
            ap = age_pars('young')
            for k, v in override.items():
                if k in ap:
                    ap[k] = v
            pars_OSN, pars_PG, pars_MC, syn = apply_age_pars(ap)
            syn['PG_MC']['w'] = PG_MC_W * ap['w_PG_MC_scale']
            n_osn = override.get('n_OSNs_per_glom', ap['n_OSNs_per_glom'])
            model_iso = Network(n_glomeruli=N_GLOM, n_OSNs_per_glom=n_osn,
                                n_odor_dims=N_ODOR, osn_sigma=OSN_SIGMA,
                                pars_OSN=pars_OSN, pars_PG=pars_PG,
                                pars_MC=pars_MC, syn=syn)
        
            # apply synapse pruning if specified
            p_keep = override.get('p_PG_MC', 1.0)
            prune_pg_mc_synapses(model_iso, p_keep=p_keep)
        
            isolation_results[label] = run_discrimination(
                model_iso, distances=DISTANCES_IS,
                n_odor=N_ODOR, n_trials=N_TRIALS_IS)

        plot_isolation_sweep(isolation_results, DISTANCES_IS,
                             young_results=young_discrim)
    
    if RUN_AGING_COMPLEXITY:
        # create a balanced set: 2 low, 2 medium, 2 high
        cx_odors, cx_complexity, cx_names = create_specific_odors(
            n_odor_dims=N_ODOR, n_low=2, n_med=2, n_high=2, seed=42)
    
        aging_cx_results = run_aging_complexity(
            cx_odors, cx_names, cx_complexity,
            pg_mc_weight=0.4,
            n_glom=N_GLOM, n_odor=N_ODOR, osn_sigma=OSN_SIGMA,
            n_trials=30
        )
        plot_aging_complexity(aging_cx_results, cx_names, cx_complexity)
        
    if RUN_DETECTION_THRESHOLD:
        CONCENTRATIONS = [0.001, 0.003, 0.005, 0.008, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
        base_odor = np.zeros(N_ODOR)
        base_odor[35] = 1.0

    
        thresh_results = {}
        for age_group in ['young', 'old']:
            print(f'\n── {age_group} ──')
            ap = age_pars(age_group)
            pars_OSN, pars_PG, pars_MC, syn = apply_age_pars(ap)
            syn['PG_MC']['w'] = 0.4 * ap['w_PG_MC_scale']
            model = Network(
                n_glomeruli=N_GLOM,
                n_OSNs_per_glom=ap['n_OSNs_per_glom'],
                n_odor_dims=N_ODOR,
                osn_sigma=OSN_SIGMA * ap['osn_sigma_scale'],
                pars_OSN=pars_OSN, pars_PG=pars_PG, pars_MC=pars_MC, syn=syn
            )
            pvals, thresh = detection_threshold_sweep(model, base_odor, CONCENTRATIONS)
            thresh_results[age_group] = {'p_values': pvals, 'threshold': thresh}
            print(f"  Threshold concentration: {thresh}")
    
        plot_detection_threshold(
            CONCENTRATIONS,
            thresh_results['young']['p_values'],
            thresh_results['old']['p_values'],
            thresh_results['young']['threshold'],
            thresh_results['old']['threshold']
        )
    if RUN_DETECTION_THRESHOLD_ONEBYONE:
        CONCENTRATIONS = [0.001, 0.003, 0.005, 0.008, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
        base_odor = np.zeros(N_ODOR)
        base_odor[20] = 1.0
    
        old = age_pars('old')
    
        isolations = [
            ('young baseline',       {}),
            ('OSN loss',             {'n_OSNs_per_glom':  3}),
            ('glom loss',            {'n_glomeruli':      70}),   # 30% reduction
            ('w_OSN_MC',             {'w_OSN_MC_scale':   old['w_OSN_MC_scale']}),
            ('w_PG_MC weight',       {'w_PG_MC_scale':    old['w_PG_MC_scale']}),
            ('p_PG_MC synapses',     {'p_PG_MC':          old['p_PG_MC']}),
            ('w+p PG_MC combined',   {'w_PG_MC_scale':    old['w_PG_MC_scale'],
                                       'p_PG_MC':          old['p_PG_MC']}),
            ('no PG',                {'w_OSN_PG_scale':   0.0}),
            ('OSN threshold',        {'osn_theta_max_shift': old['osn_theta_max_shift']}),
            ('full old',             {k: old[k] for k in old
                                      if k not in ('description', 'age_group', 'descriptio=n')}),
        ]
        thresh_results = {}
        for label, override in isolations:
            ap = age_pars('young')
            for k, v in override.items():
                if k in ap:
                    ap[k] = v
            pars_OSN, pars_PG, pars_MC, syn = apply_age_pars(ap)
            syn['PG_MC']['w'] = 0.4 * ap['w_PG_MC_scale']
            n_osn  = override.get('n_OSNs_per_glom', ap['n_OSNs_per_glom'])
            n_glom = override.get('n_glomeruli', N_GLOM)   
            model  = Network(
                n_glomeruli=n_glom,                         
                n_OSNs_per_glom=n_osn,
                n_odor_dims=N_ODOR,
                osn_sigma=OSN_SIGMA * ap['osn_sigma_scale'],
                pars_OSN=pars_OSN, pars_PG=pars_PG, pars_MC=pars_MC, syn=syn
            )
            p_keep = override.get('p_PG_MC', 1.0)
            prune_pg_mc_synapses(model, p_keep=p_keep)
        
            pvals, thresh = detection_threshold_sweep(
                model, base_odor, CONCENTRATIONS, n_trials=15, T=1.5)
            thresh_results[label] = {'p_values': pvals, 'threshold': thresh}
            print(f"  Threshold: {thresh}")
            # plot threshold comparison across conditions
            fig, ax = plt.subplots(figsize=(10, 5))
            colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(CONCENTRATIONS)))
            for label, res in thresh_results.items():
                ax.plot(CONCENTRATIONS, res['p_values'], 'o-', lw=1.5, label=label)
            ax.axhline(0.05, color='gray', ls='--', lw=1.2, label='p=0.05')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('Odor concentration')
            ax.set_ylabel('p-value')
            ax.set_title('Detection threshold — one parameter at a time')
            ax.legend(fontsize=7, loc='upper right')
            ax.grid(alpha=0.25)
            plt.tight_layout()
            fname = f'{savepath}/detection_threshold_onebyone.png'
            plt.savefig(fname, dpi=150, bbox_inches='tight')
            plt.show()
            print(f'Saved: {fname}')