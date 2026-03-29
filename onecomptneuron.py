import numpy as np
from synapsecalc import Synapse

class Neuron:
    def __init__(self, n_id, pars, is_OSN = False, n_odors = None, 
                 mu = None, sigma = None):
        """

        Parameters
        ----------
        n_id : int
            Neuron ID
        pars : dict
            dictionary for parameters, keys are params, their values are stored 
            in each key
        is_OSN : bool
            whether it's OSN or not. The default is False.
        n_odors : int
            number of odorants being used. The default is None.
        mu : np.ndarray
            OSN preferred odor vector. The default is None.
        sigma : float
            OSN tuning width. The default is None.

        Returns
        -------
        None.

        """
        # establish identity of neuron
        self.n_id = n_id
        self.is_OSN = is_OSN
        self.n_odors = n_odors

        # LIF parameters
        self.V_rest = pars['V_rest']        # resting potential (mV)
        self.tau_m = pars['tau_m']          # membrane time constant (sec)
        self.V_init = pars['V_init']        # intitial potential (mV)
        self.E_L = pars['E_L']              # leak reversal (mV)
        self.tref = pars['tref']            # refractory period (sec)
        self.v = self.V_init                # set voltage to be initial potential
        self.v_odorfiltered = 0.0   # use for the OSNs
        self.refv = pars['refv']            # refractory period voltage (hyperpolarizes)
        self.noise_sigma = pars.get('noise_sigma', 0.01)
        
        
        # spike prob. params
        self.theta_min = pars['theta_min']  # lower threshold for spiking (mV)
        self.theta_max = pars['theta_max']  # upper threshold for spiking (mV)
        

        # state vars
        self.lastspiketime = -np.inf
        self.spiked = False
        self.prev_spiked = False
        self.somaspiketimes = []
        self.v_trace = []
        self.spike_peak = 40.0
        
        # synaptic inputs
        self.input_synapses = []
        
        
        # OSN specific: preferred odor vector mu, Gaussian tuning in odor space (Euclidean distance)
        if self.is_OSN:
            if mu is None:
                raise ValueError("OSN must be initialized with a pref mu")
            self.mu = np.asarray(mu, dtype=float)
            self.preferred_molecule = int(np.argmax(self.mu))
            self.sigma = sigma if sigma is not None else 2.0
            self.I_odor_max  = 0.22
            self.I_baseline  = 0.005   # no spontaneous firing 
            
            
    def add_synapse(self, pre_neuron, weight, E_syn = 0.0, tau_rise = 1e-3, 
                    tau_fall = 2e-3):
        """

        Parameters
        ----------
        pre_neuron : TYPE
            DESCRIPTION.
        weight : TYPE
            DESCRIPTION.
        E_syn : TYPE, optional
            DESCRIPTION. The default is 0.0.
        tau_rise : TYPE, optional
            DESCRIPTION. The default is 1e-3.
        tau_fall : TYPE, optional
            DESCRIPTION. The default is 2e-3.


        Returns
        -------
        None.

        """
        # add a synaptic input to the neuron 
        synapse = Synapse(w = weight, E_syn = E_syn, tau_rise = tau_rise, 
                          tau_fall = tau_fall)
        # add the new synapse to input_synapses 
        self.input_synapses.append((pre_neuron, synapse))


    def computeVext(self, current_t = None):
        """
        compute the external voltage drive from the synaptic inputs 
        eqn used: 
            V_ext = sum all the inputs to the neuron using w * g * (E_syn - v)
        synapse.current uses this eqn already

        Returns
        -------
        V_ext : float
            voltage change in neuron i (post synapti )

        """
        V_ext = 0.0
        for pre_neuron, synapse in self.input_synapses:
            V_ext += synapse.Vext(self.v)
        return V_ext
    
    
    def update_voltage(self, V_ext, dt):
        """
        update membrane voltage using exp. euler
        eqn 1 from Linster et al., 2020: 
            tau * dv/dt + v(t) = V_ext
        solve instead of approximate because tau val (esp. OSN) close to 
        time step being used (1 msec)
        v(t+dt) = v(t) * exp(-dt/tau) + V_ext(1-exp(-dt/tau))
            

        Parameters
        ----------
        V_ext : TYPE
            DESCRIPTION.
        dt : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        # add some noise 
        # I_noise = np.random.normal(0,self.noise_sigma) 
        exp_factor = np.exp(-dt/self.tau_m)
        # V_drive = self.E_L + V_ext + I_noise 
        self.v = self.v * exp_factor +(self.E_L + V_ext) * (1.0 - exp_factor)
        
        
    def spike_prob(self, dt):
        """
        spike probability in each time step 
        linear between theta_min and theta max, then clamped so rate <= max Hz
        """
        v = self.v
        if v <= self.theta_min: 
            p = 0.0
        elif v >= self.theta_max: 
            p = 1.0
        else: 
            p = (v - self.theta_min) / (self.theta_max - self.theta_min)

        return p
            
    
    def step(self, t, dt, odor_input=None):
        """
        advance by a time step, handles both the OSN and non OSN case

        Parameters
        ----------
        t : TYPE
            DESCRIPTION.
        dt : TYPE
            DESCRIPTION.
        odor_input : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        self.spiked = False
        
        
        if self.is_OSN and odor_input is not None:
            # odor_input is a 100D vector of molecule concentrations
            if np.any(odor_input > 0):
                
                active_molecules = np.where(odor_input > 0)[0]
                if len(active_molecules) == 0:
                    response = 0.0
                else:
                    # Calculate response to each active molecule and take maximum
                    max_response = 0.0
                    n_molecules = len(self.mu)
                    
                    for mol_idx in active_molecules:
                        concentration = odor_input[mol_idx]
                        
                        # Circular distance between preferred_molecule and mol_idx
                        # on a ring of n_molecules
                        linear_dist = abs(self.preferred_molecule - mol_idx)
                        circular_dist = min(linear_dist, n_molecules - linear_dist)
                        
                        # Gaussian response based on circular distance
                        mol_response = np.exp(-circular_dist**2 / (2.0 * self.sigma ** 2))
                        
                        # Weight by concentration
                        weighted_response = mol_response * concentration
                        
                        if weighted_response > max_response:
                            max_response = weighted_response
                    
                    response = max_response
                # refractory check
                if t - self.lastspiketime <= self.tref:
                    self.v      = self.refv
                    self.spiked = False
                else:
                    # only drive voltage when not refractory
                    I_odor = response * 0.22 + self.I_baseline
                    self.v = self.E_L + I_odor
            
                    p_spike = self.spike_prob(dt)
                    if np.random.rand() < p_spike: # random num generator 
                        self.spiked        = True
                        self.v             = 40.0
                        self.lastspiketime = t
                        self.somaspiketimes.append(t)
                    else:
                        self.spiked = False
            
            else:
                # odor off — drift toward baseline
                if t - self.lastspiketime <= self.tref:
                    self.v = self.refv
                    self.spiked = False
                else:
                    self.update_voltage(self.I_baseline, dt)
                    p_spike = self.spike_prob(dt)
                    if np.random.rand() < p_spike:
                        self.spiked = True
                        self.v = 40.0
                        self.lastspiketime = t
                        self.somaspiketimes.append(t)
                    else:
                        self.spiked = False
            self.v_trace.append(self.v)
            self.prev_spiked = self.spiked
            return
        
        # non-OSN case 
        for pre_neuron, synapse in self.input_synapses:
            synapse.update(pre_neuron, dt, current_t=t)
            
        # check if refractory period 
        if t - self.lastspiketime <= self.tref:
            self.v = self.refv
            self.spiked = False
        else:
            V_drive = self.computeVext(current_t = t)
            self.update_voltage(V_drive, dt)
            
            p_spike = self.spike_prob(dt)
            if np.random.rand() < p_spike: 
                self.spiked = True
                self.v = 40.0
                self.lastspiketime = t
                self.somaspiketimes.append(t)
            else: 
                self.spiked = False
            
        self.v_trace.append(self.v)  
        self.prev_spiked = self.spiked
        
        
        
        