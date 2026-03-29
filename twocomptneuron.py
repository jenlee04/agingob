import numpy as np
from onecomptneuron import Neuron
from synapsecalc import Synapse


class TCNeuron(Neuron):
    def __init__(self, n_id, pars, g_c=1.0):
        """
        Parameters
        ----------
        n_id : int   -- neuron ID
        pars : dict  -- parameter dictionary 
        g_c  : float -- axial conductance between compartments
        """
        super().__init__(n_id, pars, is_OSN=False)
        self.g_c = g_c

        self.vsoma = self.E_L
        self.vdend = self.E_L


        self.dend_synapses = []
        self.soma_synapses = []

        self.name = None

    def add_dend_synapses(self, pre_neuron, weight, E_syn=0.0, tau_rise=1e-3,
                          tau_fall=2e-3):
        synapse = Synapse(w=weight, E_syn=E_syn, tau_rise=tau_rise,
                          tau_fall=tau_fall)
        self.dend_synapses.append((pre_neuron, synapse))

    def add_soma_synapses(self, pre_neuron, weight, E_syn=0.0, tau_rise=1e-3,
                          tau_fall=2e-3):
        synapse = Synapse(w=weight, E_syn=E_syn, tau_rise=tau_rise,
                          tau_fall=tau_fall)
        self.soma_synapses.append((pre_neuron, synapse))

    def computecompartmentvext(self, compartment='dend'):
        """change in the voltage of each compartment"""
        if compartment == 'dend':
            synapses = self.dend_synapses
            v_comp   = self.vdend
        else:
            synapses = self.soma_synapses
            v_comp   = self.vsoma

        V_ext = 0.0
        for pre_neuron, synapse in synapses:
            V_ext += synapse.Vext(v_comp)
        return V_ext

    def step(self, t, dt, odor_input=None):
        # update all synapses
        for pre_neuron, synapse in self.dend_synapses:
            synapse.update(pre_neuron, dt, current_t=t)
        for pre_neuron, synapse in self.soma_synapses:
            synapse.update(pre_neuron, dt, current_t=t)
    
        if t - self.lastspiketime <= self.tref:
            self.spiked = False
            self.v = self.refv
            self.v_trace.append(self.v)
            return
    
        # sum all inputs from both compartments
        V_ext = 0.0
        for pre_neuron, synapse in self.dend_synapses:
            V_ext += synapse.Vext(self.v)
        for pre_neuron, synapse in self.soma_synapses:
            V_ext += synapse.Vext(self.v)
    
        self.update_voltage(V_ext + self.I_baseline, dt)
        self.v = self.v  # vsoma and vdend no longer separate
    
        p_spike = self.spike_prob(dt)
        if np.random.rand() < p_spike:
            self.spiked = True
            self.v = 40.0
            self.somaspiketimes.append(t)
            self.lastspiketime = t
        else:
            self.spiked = False
    
        self.v_trace.append(self.v)
        self.prev_spiked = self.spiked