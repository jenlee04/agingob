import numpy as np 
import matplotlib.pyplot as plt

def create_specific_odors(n_odor_dims=100,
                          n_low=3, n_med=3, n_high=3,
                          seed=0):
    np.random.seed(seed)

    odors = []
    complexity = []
    names = []

    # ── LOW complexity (single molecules) ─────────────────────
    low_mols = [20, 50, 52, 80, 90]
    for mol in low_mols[:n_low]:
        odor = np.zeros(n_odor_dims)
        odor[mol] = 1.0
        odors.append(odor)
        complexity.append('low')
        names.append(f'Pure_{mol}')

    # ── MEDIUM complexity (5–10 components) ───────────────────
    for i in range(n_med):
        odor = np.zeros(n_odor_dims)
        n_comp = np.random.randint(5, 11)
        mols = np.random.choice(n_odor_dims, n_comp, replace=False)
        odor[mols] = np.random.uniform(0.5, 1.0, size=n_comp)
        odors.append(odor)
        complexity.append('medium')
        names.append(f'Med_{i}')

    # ── HIGH complexity (15–25 components) ─────────────────────
    for i in range(n_high):
        odor = np.zeros(n_odor_dims)
        n_comp = np.random.randint(15, 26)
        mols = np.random.choice(n_odor_dims, n_comp, replace=False)
        odor[mols] = np.random.uniform(0.2, 1.0, size=n_comp)
        odors.append(odor)
        complexity.append('high')
        names.append(f'High_{i}')

    return odors, complexity, names


def sniffing(odor_vector, t, stim_start, stim_end, sniff_freq=8.0):
    """
    Modulate odor presentation at sniffing frequency (8 Hz in rats)
    
    Parameters:
    - odor_vector: base odor concentration vector
    - t: current time (seconds)
    - stim_start, stim_end: stimulation window
    - sniff_freq: sniffing frequency in Hz (default 8 Hz)
    
    Returns modulated odor vector
    """
    if stim_start <= t <= stim_end:
        # Create sinusoidal modulation for sniffing
        # Sniff cycle: inhale (positive) and exhale (near zero)
        phase = 2 * np.pi * sniff_freq * (t - stim_start)
        modulation = (np.sin(phase) + 1) / 2  # Ranges from 0 to 1
        
        # Apply modulation to odor vector
        return odor_vector * modulation
    else:
        return np.zeros_like(odor_vector)


def continuous_odor(odor_vector, t, stim_start, stim_end):
    """
    Present odor continuously (no sniffing modulation)
    """
    if stim_start <= t <= stim_end:
        return odor_vector
    else:
        return np.zeros_like(odor_vector)


def plot_receptor_sensitivities(OSN_types, n_odor_dims=10):
    """
    Plot the sensitivity profile of each receptor type
    Each type has maximal (1.0) sensitivity to ONE molecule
    """
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    fig.suptitle('Receptor Type Sensitivity Profiles (Gaussian width=2, max=1 molecule)', fontsize=14)
    
    molecule_range = np.arange(n_odor_dims)
    
    for type_idx, type_info in enumerate(OSN_types):
        sensitivities = np.zeros(n_odor_dims)
        pref_mol = type_info['preferred_molecule']
        
        for mol_idx in range(n_odor_dims):
            # Distance to preferred molecule (in discrete space)
            dist = abs(mol_idx - pref_mol)
            
            # Gaussian sensitivity centered at preferred molecule
            sensitivities[mol_idx] = np.exp(-(dist ** 2) / (2 * type_info['sigma'] ** 2))
        
        # Plot
        axes[type_idx].bar(molecule_range, sensitivities, color=f'C{type_idx}', alpha=0.7)
        axes[type_idx].set_title(f'Type {type_idx}\n(max: molecule {pref_mol})')
        axes[type_idx].set_xlabel('Molecule #')
        axes[type_idx].set_ylabel('Sensitivity')
        axes[type_idx].set_ylim([0, 1.1])
        axes[type_idx].axhline(y=1.0, color='red', linestyle='--', alpha=0.3, label='Max')
        axes[type_idx].axvline(x=pref_mol, color='red', linestyle='--', alpha=0.5)
        axes[type_idx].set_xticks(molecule_range)
        
    plt.tight_layout()
    plt.show()