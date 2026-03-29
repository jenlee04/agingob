import numpy as np

class Synapse:
    def __init__(self, w=None, E_syn=None, tau_rise=None, tau_fall=None):
        """
        Parameters
        ----------
        w        : float  -- synaptic weight
        E_syn    : float  -- reversal potential (mV)
        tau_rise : float  -- rise time constant (s)
        tau_fall : float  -- decay time constant (s)
        delay    : float  -- axonal delay (s); currently unused (all = 0)
        """
        self.w        = w
        self.E_syn    = E_syn
        self.tau_rise = tau_rise
        self.tau_fall = tau_fall

        self.g_max  = 1.0
        self.g_rise = 0.0
        self.g_fall = 0.0
        self.g      = 0.0           # output gi(t)

    def update(self, pre_neuron, dt, current_t=None):
        """
        update the conductance change IF the p

        Parameters
        ----------
        pre_neuron   : pre synaptic neuron
        dt           : float -- timestep (s)
        current_t    : float -- current time t

        """
        self.t = current_t
        drive = float(pre_neuron.prev_spiked) 
        self.g_rise = self.g_rise * np.exp(-dt / self.tau_rise) + drive
        self.g_fall = self.g_fall * np.exp(-dt / self.tau_fall) + drive
        self.g = self.g_max * (self.g_fall - self.g_rise)
        if self.g < 0.0:
            self.g = 0.0


    def Vext(self, V_i):
        """
        Synaptic current: V_ext = w · g · (E_syn − V_i).
        (Paper: E_{N,ij} − v_i(t) term in I_i,ext.)
        """
        return self.w * self.g * (self.E_syn - V_i)