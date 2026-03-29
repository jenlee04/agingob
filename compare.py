import numpy as np
from scipy.stats import pearsonr
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def comparisons(vec1, vec2):
    """
    computes the similarity metrics between two vecs
    Parameters
    ----------
    vec1 : vector 
        DESCRIPTION.
    vec2 : vector
        DESCRIPTION.

    Returns
    -------
    dict with keys: dot_product, correlation, euclidean distance 

    """
    dot_prod = float(np.dot(vec1, vec2))
    corr = float(pearsonr(vec1, vec2)[0]) if (np.std(vec1)) > 0 and np.std(vec2) > 0 else 0.0
    euclidean = float(np.sqrt(np.sum((vec1 - vec2) ** 2)))
    
    return {'dot_product': dot_prod, 'correlation': corr, 'euclidean_distance': euclidean}

def popvectors(OSNs, PGs, MCs, GCs, ETs, avg_OSN_responses, avg_PG_responses, 
                              avg_MC_responses, avg_GC_responses, avg_ET_responses, n_odors, n_glomeruli):
    """
    Create 100D population vectors for each odor and cell type.
    
    For OSNs: Average the 5 OSNs of the same type (same glomerulus)
    For PGs and MCs: Direct spike counts (1 per glomerulus)
    
    Parameters:
    -----------
    OSNs, PGs, MCs : lists
        Lists of neuron objects
    avg_OSN_responses, avg_PG_responses, avg_MC_responses : lists
        Average firing rates [odor][neuron]
    n_odors : int
        Number of odors
    n_glomeruli : int
        Number of glomeruli (should be 100)
        
    Returns:
    --------
    dict : {
        'OSN': [n_odors x n_glomeruli],
        'PG': [n_odors x n_glomeruli],
        'MC': [n_odors x n_glomeruli]
    }
    """
    
    # Initialize population vectors
    OSN_pop_vectors = np.zeros((n_odors, n_glomeruli))
    PG_pop_vectors = np.zeros((n_odors, n_glomeruli))
    MC_pop_vectors = np.zeros((n_odors, n_glomeruli))
    GC_pop_vectors = np.zeros((n_odors, n_glomeruli))
    ET_pop_vectors = np.zeros((n_odors, n_glomeruli))

    # For OSNs: average the 5 OSNs in each glomerulus
    for odor_idx in range(n_odors):
        for glom_idx in range(n_glomeruli):
            # Find OSNs in this glomerulus
            osn_indices = [i for i, osn in enumerate(OSNs) 
                          if osn.glomerulus_id == glom_idx]
            
            # Average their responses
            osn_rates = [avg_OSN_responses[odor_idx][i] for i in osn_indices]
            OSN_pop_vectors[odor_idx, glom_idx] = np.mean(osn_rates)
    
    # For PGs: direct mapping (1 PG per glomerulus)
    for odor_idx in range(n_odors):
        for glom_idx in range(n_glomeruli):
            pg_idx = [i for i, pg in enumerate(PGs) 
                     if pg.glomerulus_id == glom_idx][0]
            PG_pop_vectors[odor_idx, glom_idx] = avg_PG_responses[odor_idx][pg_idx]
    
    # For MCs: direct mapping (1 MC per glomerulus)
    for odor_idx in range(n_odors):
        for glom_idx in range(n_glomeruli):
            mc_idx = [i for i, mc in enumerate(MCs) 
                     if mc.glomerulus_id == glom_idx][0]
            MC_pop_vectors[odor_idx, glom_idx] = avg_MC_responses[odor_idx][mc_idx]
            
    # For GCs: direct mapping (1 GC per glomerulus)
    for odor_idx in range(n_odors):
        for glom_idx in range(n_glomeruli):
            gc_idx = [i for i, gc in enumerate(GCs) 
                          if gc.glomerulus_id == glom_idx][0]
            GC_pop_vectors[odor_idx, glom_idx] = avg_GC_responses[odor_idx][gc_idx]
        
    # for ETs: direct mapping 
    for odor_idx in range(n_odors):
        for glom_idx in range(n_glomeruli):
            et_idx = [i for i, et in enumerate(ETs) 
                          if et.glomerulus_id == glom_idx][0]
            ET_pop_vectors[odor_idx, glom_idx] = avg_ET_responses[odor_idx][et_idx]

            

    

    return {
        'OSN': OSN_pop_vectors,
        'PG': PG_pop_vectors,
        'MC': MC_pop_vectors,
        'GC': GC_pop_vectors, 
        'ET': ET_pop_vectors
    }

# smoothing 
def smooth_vector(vec, sigma=1.5):
    """
    Gaussian-smooth a 1-D population vector along the glomerular axis.
    sigma=1.5 removes trial-to-trial jitter while preserving peaks.
    """
    return gaussian_filter1d(vec, sigma=sigma)

# PLOTTING COMPARISONS # 
def plot_population_vectors(pop_vectors_dict, odor_names,
                             zoom_range=(45, 56), smooth_sigma=1.5,
                             savepath=None):
    """
    Three-panel comparison plot (one column per cell type: OSN, PG, MC).
    Each panel shows the full 100-MC vector (smoothed) plus a zoomed inset
    around the expected peak region.

    Parameters
    ----------
    pop_vectors_dict : dict  {'OSN': n_odors×n_glom, 'PG': ..., 'MC': ...}
    odor_names       : list of str
    zoom_range       : (start, end) inclusive glomerulus indices
    smooth_sigma     : Gaussian sigma for smoothing (set 0 to disable)
    savepath         : directory to save PNG, or None
    """
    cell_types = ['OSN', 'PG', 'MC']
    colors = plt.cm.tab10(np.linspace(0, 0.5, len(odor_names)))
    z0, z1 = zoom_range
    n_glom = pop_vectors_dict['MC'].shape[1]
    glom_idx = np.arange(n_glom)
    zoom_idx = np.arange(z0, z1 + 1)

    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    for col, ct in enumerate(cell_types):
        vecs = pop_vectors_dict[ct]   # shape (n_odors, n_glom)

        # ── Full view ───────────────────────────────────────────────────────
        ax_full = fig.add_subplot(gs[0, col])
        for oi, name in enumerate(odor_names):
            v = smooth_vector(vecs[oi], sigma=smooth_sigma) if smooth_sigma > 0 else vecs[oi]
            ax_full.plot(glom_idx, v, color=colors[oi], linewidth=1.8, label=name, alpha=0.9)
        ax_full.axvspan(z0, z1, alpha=0.12, color='gray')
        ax_full.set_title(f'{ct} — Full', fontsize=12, fontweight='bold')
        ax_full.set_xlabel('Glomerulus index')
        ax_full.set_ylabel('Firing rate (Hz)')
        ax_full.legend(fontsize=8)
        ax_full.grid(True, alpha=0.25)

        # ── Zoomed view ─────────────────────────────────────────────────────
        ax_zoom = fig.add_subplot(gs[1, col])
        for oi, name in enumerate(odor_names):
            v_raw = vecs[oi, z0:z1 + 1]
            v = smooth_vector(v_raw, sigma=smooth_sigma) if smooth_sigma > 0 else v_raw
            ax_zoom.plot(zoom_idx, v, 'o-', color=colors[oi], linewidth=2.5,
                         markersize=7, label=name, alpha=0.9)
            # annotate peak
            peak_local = np.argmax(v)
            ax_zoom.annotate(f'{v[peak_local]:.1f}',
                             xy=(zoom_idx[peak_local], v[peak_local]),
                             xytext=(0, 8), textcoords='offset points',
                             ha='center', fontsize=8, color=colors[oi],
                             fontweight='bold')
        ax_zoom.set_title(f'{ct} — Zoomed ({z0}–{z1})', fontsize=12, fontweight='bold')
        ax_zoom.set_xlabel('Glomerulus index')
        ax_zoom.set_ylabel('Firing rate (Hz)')
        ax_zoom.legend(fontsize=8)
        ax_zoom.grid(True, alpha=0.25)

        # metrics between first two odors
        if len(odor_names) >= 2:
            v1 = smooth_vector(vecs[0], sigma=smooth_sigma) if smooth_sigma > 0 else vecs[0]
            v2 = smooth_vector(vecs[1], sigma=smooth_sigma) if smooth_sigma > 0 else vecs[1]
            m = comparisons(v1, v2)
            ax_zoom.set_title(
                f'{ct} — Zoomed\nr={m["correlation"]:.3f}, Δ={m["euclidean_distance"]:.1f}',
                fontsize=11, fontweight='bold')

    fig.suptitle('Population Vector Comparison — OSN → PG → MC', fontsize=15, fontweight='bold')

    if savepath:
        fig.savefig(f'{savepath}/population_vector_comparison.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {savepath}/population_vector_comparison.png")
    plt.show()
    plt.close()


def plot_population_vectors_sorted(pop_vectors_dict, odor_names, model,
                                    smooth_sigma=1.5, savepath=None):
    """
    Plot population vectors sorted by preferred molecule rather than glomerulus index.
    With a scrambled glomerular map, glomerulus index is meaningless for visualization.
    Sorting by preferred molecule restores the smooth Gaussian response profile.

    Parameters
    ----------
    pop_vectors_dict : dict  {'OSN': n_odors×n_glom, 'PG': ..., 'MC': ..., 'GC': ...}
    odor_names       : list of str
    model            : dict returned by Network(), must contain 'glom_to_preferred_mol'
    smooth_sigma     : float
    savepath         : str or None
    """
    glom_to_mol = model.get('glom_to_preferred_mol', {i: i for i in range(100)})
    n_glom      = pop_vectors_dict['MC'].shape[1]

    # Sort glomerulus indices by their preferred molecule
    sort_order  = sorted(range(n_glom), key=lambda g: glom_to_mol.get(g, g))
    mol_x       = np.array([glom_to_mol[g] for g in sort_order])

    cell_types  = ['OSN', 'PG', 'MC', 'GC']
    colors      = plt.cm.tab10(np.linspace(0, 0.5, len(odor_names)))

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Population Vectors — sorted by preferred molecule\n'
                 '(scrambled glomerular map restored to chemical order)',
                 fontsize=13, fontweight='bold')

    for ax, ct in zip(axes.flat, cell_types):
        vecs = pop_vectors_dict[ct]
        for oi, name in enumerate(odor_names):
            v_sorted = vecs[oi][sort_order]
            v_smooth = smooth_vector(v_sorted, sigma=smooth_sigma)
            ax.plot(mol_x, v_smooth, color=colors[oi], lw=1.8, label=name, alpha=0.9)
        ax.set_title(ct, fontweight='bold')
        ax.set_xlabel('Preferred molecule index')
        ax.set_ylabel('Firing rate (Hz)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    if savepath:
        fname = f'{savepath}/population_vectors_sorted.png'
        fig.savefig(fname, dpi=150, bbox_inches='tight')
        print(f"Saved: {fname}")
    plt.close()


def plot_discrimination_vs_distance(pop_vectors_dict, odors, odor_names,
                                     model, smooth_sigma=1.5, savepath=None):
    """
    Plot correlation, Euclidean distance, and dot product vs molecular distance
    between odors, for each cell type (OSN, PG, MC, GC) on the same axes.

    With a scrambled glomerular map, 'distance' is defined in molecule space
    (circular distance between preferred molecules), NOT glomerulus index space.
    This is the correct measure regardless of glomerular arrangement.

    Parameters
    ----------
    pop_vectors_dict : dict  {'OSN': ..., 'PG': ..., 'MC': ..., 'GC': ...}
    odors            : list of odor vectors (100-dim)
    odor_names       : list of str
    model            : dict from Network()
    smooth_sigma     : float
    savepath         : str or None
    """
    n_odor_dims = odors[0].shape[0]

    # Identify pure single-molecule odors and their active molecule
    pure_indices, active_mols = [], []
    for i, odor in enumerate(odors):
        dims = np.where(odor > 0)[0]
        if len(dims) == 1:
            pure_indices.append(i)
            active_mols.append(int(dims[0]))

    if len(pure_indices) < 2:
        print("Need ≥2 pure odors for discrimination vs distance plot.")
        return

    ref_idx = pure_indices[0]
    ref_mol = active_mols[0]

    cell_types = ['OSN', 'PG', 'MC', 'GC']
    ct_colors  = {'OSN': 'steelblue', 'PG': 'darkorange', 'MC': 'forestgreen', 'GC': 'crimson'}

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle(f'Discrimination across cell types vs molecular distance\n'
                 f'(reference: {odor_names[ref_idx]}, mol {ref_mol})',
                 fontsize=13, fontweight='bold')

    for ct in cell_types:
        vecs    = pop_vectors_dict[ct]
        ref_vec = smooth_vector(vecs[ref_idx], sigma=smooth_sigma)
        dists, corrs, euclids, dots = [], [], [], []

        for i, oi in enumerate(pure_indices):
            if oi == ref_idx:
                continue
            # Circular distance in molecule space
            linear = abs(ref_mol - active_mols[i])
            circ   = min(linear, n_odor_dims - linear)
            dists.append(circ)
            tv = smooth_vector(vecs[oi], sigma=smooth_sigma)
            m  = comparisons(ref_vec, tv)
            corrs.append(m['correlation'])
            euclids.append(m['euclidean_distance'])
            dots.append(m['dot_product'])

        # Sort by distance for clean line plot
        order   = np.argsort(dists)
        dists   = np.array(dists)[order]
        corrs   = np.array(corrs)[order]
        euclids = np.array(euclids)[order]
        dots    = np.array(dots)[order]

        axes[0].plot(dists, corrs,   'o-', lw=2, ms=7, label=ct,
                     color=ct_colors[ct], alpha=0.85)
        axes[1].plot(dists, euclids, 'o-', lw=2, ms=7, label=ct,
                     color=ct_colors[ct], alpha=0.85)
        axes[2].plot(dists, dots,    'o-', lw=2, ms=7, label=ct,
                     color=ct_colors[ct], alpha=0.85)

    axes[0].set_title('Correlation  (↓ = better discrimination)', fontweight='bold')
    axes[0].set_ylabel('Pearson r')
    axes[0].set_ylim([-0.1, 1.05])
    axes[0].axhline(0, color='gray', ls=':', alpha=0.5)

    axes[1].set_title('Euclidean Distance  (↑ = better)', fontweight='bold')
    axes[1].set_ylabel('Euclidean distance')

    axes[2].set_title('Dot Product  (↓ = better)', fontweight='bold')
    axes[2].set_ylabel('Dot product')

    for ax in axes:
        ax.set_xlabel('Circular distance (molecules)')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if savepath:
        fname = f'{savepath}/discrimination_vs_mol_distance.png'
        fig.savefig(fname, dpi=150, bbox_inches='tight')
        print(f"Saved: {fname}")
    plt.close()



def plot_between_odors(pop_vectors_dict, odors, odor_names,
                       smooth_sigma=1.5, savepath=None):
    """
    Correlation and Euclidean distance vs molecular distance (pure odors only).
    """
    pure_indices, active_mols = [], []
    for i, odor in enumerate(odors):
        dims = np.where(odor > 0)[0]
        if len(dims) == 1:
            pure_indices.append(i)
            active_mols.append(dims[0])

    if len(pure_indices) < 2:
        print("Not enough pure molecular odors for distance analysis")
        return

    ref_idx = pure_indices[0]
    ref_mol = active_mols[0]
    n_glom  = pop_vectors_dict['MC'].shape[1]

    for ct in ['OSN', 'PG', 'MC']:
        vecs = pop_vectors_dict[ct]
        ref_vec = smooth_vector(vecs[ref_idx], sigma=smooth_sigma)
        dists, corrs, euclids = [], [], []

        for i, oi in enumerate(pure_indices):
            if oi == ref_idx:
                continue
            linear = abs(ref_mol - active_mols[i])
            circ = min(linear, n_glom - linear)
            dists.append(circ)
            tv = smooth_vector(vecs[oi], sigma=smooth_sigma)
            m = comparisons(ref_vec, tv)
            corrs.append(m['correlation'])
            euclids.append(m['euclidean_distance'])

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        fig.suptitle(f'{ct}: Similarity vs Molecular Distance\n(ref: {odor_names[ref_idx]})',
                     fontsize=13, fontweight='bold')

        axes[0].plot(dists, corrs, 'o-', linewidth=2, markersize=8)
        axes[0].set_xlabel('Circular distance (molecules)')
        axes[0].set_ylabel('Pearson r (lower = better discrimination)')
        axes[0].set_title('Correlation')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim([-0.1, 1.05])

        axes[1].plot(dists, euclids, 'o-', linewidth=2, markersize=8, color='green')
        axes[1].set_xlabel('Circular distance (molecules)')
        axes[1].set_ylabel('Euclidean distance (higher = better)')
        axes[1].set_title('Euclidean Distance')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        if savepath:
            fig.savefig(f'{savepath}/{ct}_similarity_vs_distance.png', dpi=300, bbox_inches='tight')
            print(f"Saved: {savepath}/{ct}_similarity_vs_distance.png")
        plt.show()
        plt.close()


# ============================================================================
# CONSOLIDATED FROM fullmodelcomparisonfns.py
# (that file is now retired — all useful functions live here)
# ============================================================================

def compute_pairwise_similarities(pop_vectors, odor_names=None):
    """
    Compute all pairwise correlation / euclidean matrices between odors.

    Parameters
    ----------
    pop_vectors : ndarray, shape (n_odors, n_glom)
    odor_names  : list of str, optional

    Returns
    -------
    dict with keys 'correlation' and 'euclidean_distance', each an (n x n) matrix
    """
    n_odors = pop_vectors.shape[0]
    corr_mat   = np.zeros((n_odors, n_odors))
    euclid_mat = np.zeros((n_odors, n_odors))

    for i in range(n_odors):
        for j in range(n_odors):
            m = comparisons(pop_vectors[i], pop_vectors[j])
            corr_mat[i, j]   = m['correlation']
            euclid_mat[i, j] = m['euclidean_distance']

    return {'correlation': corr_mat, 'euclidean_distance': euclid_mat}


def compare_discrimination_across_cell_types(pop_vectors_dict, odors, odor_names,
                                              smooth_sigma=1.5, savepath=None):
    """
    Plot correlation and Euclidean distance vs odor distance for OSN, PG, MC,
    and GC on the same axes so you can directly compare how each layer
    transforms discriminability.

    Parameters
    ----------
    pop_vectors_dict : dict  {'OSN': ndarray, 'PG': ndarray, 'MC': ndarray, 'GC': ndarray}
    odors            : list of odor vectors
    odor_names       : list of str
    smooth_sigma     : float, Gaussian smoothing applied before metrics
    savepath         : str or None
    """
    # identify pure single-molecule odors
    pure_indices, active_mols = [], []
    for i, odor in enumerate(odors):
        dims = np.where(odor > 0)[0]
        if len(dims) == 1:
            pure_indices.append(i)
            active_mols.append(dims[0])

    if len(pure_indices) < 2:
        print("Not enough pure odors for cross-cell-type comparison")
        return

    ref_idx = pure_indices[0]
    ref_mol = active_mols[0]
    n_glom  = pop_vectors_dict['MC'].shape[1]

    cell_types = ['OSN', 'PG', 'MC', 'GC']
    colors_ct  = {'OSN': 'steelblue', 'PG': 'darkorange', 'MC': 'green', 'GC': 'crimson'}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f'Discrimination across cell types\n(ref: {odor_names[ref_idx]})',
                 fontsize=13, fontweight='bold')

    for ct in cell_types:
        vecs    = pop_vectors_dict[ct]
        ref_vec = smooth_vector(vecs[ref_idx], sigma=smooth_sigma)
        dists, corrs, euclids = [], [], []

        for i, oi in enumerate(pure_indices):
            if oi == ref_idx:
                continue
            linear = abs(ref_mol - active_mols[i])
            circ   = min(linear, n_glom - linear)
            dists.append(circ)
            tv = smooth_vector(vecs[oi], sigma=smooth_sigma)
            m  = comparisons(ref_vec, tv)
            corrs.append(m['correlation'])
            euclids.append(m['euclidean_distance'])

        axes[0].plot(dists, corrs, 'o-', lw=2, ms=7,
                     label=ct, color=colors_ct[ct], alpha=0.85)
        axes[1].plot(dists, euclids, 'o-', lw=2, ms=7,
                     label=ct, color=colors_ct[ct], alpha=0.85)

    axes[0].set_xlabel('Circular distance (molecules)')
    axes[0].set_ylabel('Pearson r  (↓ = better)')
    axes[0].set_title('Correlation vs distance')
    axes[0].set_ylim([-0.1, 1.05])
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Circular distance (molecules)')
    axes[1].set_ylabel('Euclidean distance  (↑ = better)')
    axes[1].set_title('Euclidean distance vs distance')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if savepath:
        fig.savefig(f'{savepath}/discrimination_across_cell_types.png',
                    dpi=300, bbox_inches='tight')
        print(f"Saved: {savepath}/discrimination_across_cell_types.png")
    plt.show()
    plt.close()


def create_similarity_table(pop_vectors_dict, odors, odor_names, smooth_sigma=1.5):
    """
    Print and return a DataFrame of pairwise discrimination metrics between the
    first two odors for each cell type (OSN, PG, MC, GC).

    Returns
    -------
    pandas DataFrame
    """
    import pandas as pd

    if len(odors) < 2:
        print("Need at least 2 odors for similarity table")
        return None

    rows = []
    for ct in ['OSN', 'PG', 'MC', 'GC']:
        vecs = pop_vectors_dict[ct]
        v1   = smooth_vector(vecs[0], sigma=smooth_sigma)
        v2   = smooth_vector(vecs[1], sigma=smooth_sigma)
        m    = comparisons(v1, v2)
        rows.append({
            'Cell type':          ct,
            'Odor A':             odor_names[0],
            'Odor B':             odor_names[1],
            'Correlation':        round(m['correlation'], 4),
            'Euclidean distance': round(m['euclidean_distance'], 3),
            'Dot product':        round(m['dot_product'], 3),
        })

    import pandas as pd
    df = pd.DataFrame(rows)
    print("\n" + "=" * 65)
    print("SIMILARITY TABLE  (lower corr / higher euclid = better discrim)")
    print("=" * 65)
    print(df.to_string(index=False))
    print("=" * 65 + "\n")
    return df


def analyze_population_vectors(model, avg_OSN, avg_MC, avg_PG, avg_GC,
                                odors, odor_names, savepath):
    """
    Orchestrator: build population vectors, run all comparison plots,
    and save a CSV similarity table.

    Parameters
    ----------
    model       : dict returned by Network()
    avg_OSN/MC/PG/GC : lists [odor][neuron] of average firing rates
    odors       : list of odor vectors
    odor_names  : list of str
    savepath    : str, directory to save figures and CSV
    """
    OSNs = model['OSNs'];  PGs = model['PGs']
    MCs  = model['MCs'];   GCs = model['GCs']
    n_odors     = len(odors)
    n_glomeruli = len(model['glomeruli'])

    print("\n" + "=" * 60)
    print("POPULATION VECTOR ANALYSIS")
    print("=" * 60)

    pop = popvectors(OSNs, PGs, MCs, GCs,
                     avg_OSN, avg_PG, avg_MC, avg_GC,
                     n_odors, n_glomeruli)

    plot_population_vectors(pop, odor_names, savepath=savepath)
    compare_discrimination_across_cell_types(pop, odors, odor_names, savepath=savepath)

    df = create_similarity_table(pop, odors, odor_names)
    if df is not None and savepath:
        import os
        df.to_csv(os.path.join(savepath, 'similarity_table.csv'), index=False)
        print(f"Saved: {savepath}/similarity_table.csv")

    print("=" * 60)
    print("POPULATION VECTOR ANALYSIS COMPLETE")
    print("=" * 60 + "\n")
    return pop