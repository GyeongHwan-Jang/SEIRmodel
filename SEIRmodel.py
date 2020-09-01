#!/usr/bin/env python
# coding: utf-8

# In[61]:


import networkx as nx
import numpy as np
import scipy as sp
import scipy.integrate

import matplotlib.pyplot as plt

class SEIRmodel():
    """
    A class to simulate the Deterministic SEIRS Model
    ===================================================
    Params: beta    Rate of transmission (exposure)
            sigma   Rate of infection (upon exposure)
            gamma   Rate of recovery (upon infection)


            initE   Init number of exposed individuals
            initI   Init number of infectious individuals
            initR   Init number of recovered individuals
    """

    def __init__(self, initN, beta, sigma, gamma, initE=0, initI = 0, initR=0):

        # -----------------------------------------------------------------------------------------------------------
        # Model Parameters:

        self.beta   = beta
        self.sigma  = sigma
        self.gamma  = gamma

        # -----------------------------------------------------------------------------------------------------------
        # Initialize Timekeeping:

        self.t       = 0
        self.tmax    = 0 # will be set when run() is called
        self.tseries = np.array([0])

        # -----------------------------------------------------------------------------------------------------------
        # Initialize Counts of inidividuals with each state:

        self.N          = np.array([int(initN)])
        self.numE       = np.array([int(initE)])
        self.numI       = np.array([int(initI)])
        self.numR       = np.array([int(initR)])
        self.numS       = np.array([self.N[-1] - self.numE[-1] - self.numI[-1] - self.numR[-1]])
        assert(self.numS[0] >= 0), "The specified initial population size N must be greater than or equal to the initial compartment counts."


    def system_dfes(t, variables, beta, sigma, gamma):

        S, E, I, R = variables    # varibles is a list with compartment counts as elements

        N   = S + E + I + R
        dS  = - (beta*S*I)/N
        dE  = (beta*S*I)/N - sigma*E
        dI  = sigma*E - gamma*I
        dR  = gamma*I

        return [dS, dE, dI, dR]


    def run_epoch(self, runtime, dt=0.1):

        # -----------------------------------------------------------------------------------------------------------
        # Create a list of times at which the ODE solver should output system values.
        # Append this list of times as the model's timeseries
        t_eval = np.arange(start=self.t, stop=self.t+runtime, step=dt)

        # Define the range of time values for the integration:
        t_span = (self.t, self.t+runtime)

        # Define the initial conditions as the system's current state:
        # (which will be the t=0 condition if this is the first run of this model,
        # else where the last sim left off)

        init_cond = [self.numS[-1], self.numE[-1], self.numI[-1], self.numR[-1]]

        # -----------------------------------------------------------------------------------------------------------
        # Solve the system of differential eqns:

        solution = scipy.integrate.solve_ivp(lambda t, X: SEIRmodel.system_dfes(t, X, self.beta, self.sigma, self.gamma),
                                             t_span=[self.t, self.tmax], y0=init_cond, t_eval=t_eval)

        # -----------------------------------------------------------------------------------------------------------
        # Store the solution output as the model's time series and data series:

        self.tseries    = np.append(self.tseries, solution['t'])
        self.numS       = np.append(self.numS, solution['y'][0])
        self.numE       = np.append(self.numE, solution['y'][1])
        self.numI       = np.append(self.numI, solution['y'][2])
        self.numR       = np.append(self.numR, solution['y'][3])

        self.t = self.tseries[-1]


    def run(self, T, dt=0.1, verbose=False):

        assert(T > 0), "Total simulation time T must be larger than 0"
        self.tmax += T

        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Run the simulation loop:
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        self.run_epoch(runtime=self.tmax, dt=dt)

        # -----------------------------------------------------------------------------------------------------------

        print("t = %.2f" % self.t)
        if(verbose):
            print("\t S   = " + str(self.numS[-1]))
            print("\t E   = " + str(self.numE[-1]))
            print("\t I   = " + str(self.numI[-1]))
            print("\t R   = " + str(self.numR[-1]))

        return True

    def plot(self, title, plot_style = "line", color_S='tab:green', color_E='orange', color_I='crimson', color_R='tab:blue', xlim = None, ylim = None, plot_percentages = True, legend = True, figsize = (12,8)):

        # -----------------------------------------------------------------------------------------------------------
        # Create an Axes object:

        fig, ax = plt.subplots(figsize = figsize)

        # -----------------------------------------------------------------------------------------------------------
        # Prepare data series to be plotted:

        Eseries     = self.numE/self.N if plot_percentages else self.numE
        Iseries     = self.numI/self.N if plot_percentages else self.numI
        Rseries     = self.numR/self.N if plot_percentages else self.numR
        Sseries     = self.numS/self.N if plot_percentages else self.numS

        # -----------------------------------------------------------------------------------------------------------
        # Draw the line plot:

        if(any(Eseries) and plot_style =='line'):
            ax.plot(np.ma.masked_where(Eseries<=0, self.tseries), np.ma.masked_where(Eseries<=0, Eseries), color=color_E, label='$E$', zorder=6)
        if(any(Iseries) and plot_style =='line'):
            ax.plot(np.ma.masked_where(Iseries<=0, self.tseries), np.ma.masked_where(Iseries<=0, Iseries), color=color_I, label='$I$', zorder=6)
        if(any(Sseries) and plot_style =='line'):
            ax.plot(np.ma.masked_where(Sseries<=0, self.tseries), np.ma.masked_where(Sseries<=0, Sseries), color=color_S, label='$S$', zorder=6)
        if(any(Rseries) and plot_style =='line'):
            ax.plot(np.ma.masked_where(Rseries<=0, self.tseries), np.ma.masked_where(Rseries<=0, Rseries), color=color_R, label='$R$', zorder=6)

        # -----------------------------------------------------------------------------------------------------------
        # Draw the stacked plot:

        topstack = np.zeros_like(self.tseries)

        if(any(Eseries) and plot_style=='stacked'):
            ax.fill_between(np.ma.masked_where(Eseries<=0, self.tseries), np.ma.masked_where(Eseries<=0, topstack+Eseries), topstack, color=color_E, alpha=0.5, label='$E$', zorder=2)
            ax.plot(np.ma.masked_where(Eseries<=0, self.tseries), np.ma.masked_where(Eseries<=0, topstack+Eseries), color=color_E, zorder=3)
            topstack = topstack+Eseries
        if(any(Iseries) and plot_style=='stacked'):
            ax.fill_between(np.ma.masked_where(Iseries<=0, self.tseries), np.ma.masked_where(Iseries<=0, topstack+Iseries), topstack, color=color_I, alpha=0.5, label='$I$', zorder=2)
            ax.plot(np.ma.masked_where(Iseries<=0, self.tseries), np.ma.masked_where(Iseries<=0, topstack+Iseries), color=color_I, zorder=3)
            topstack = topstack+Iseries
        if(any(Rseries) and plot_style=='stacked'):
            ax.fill_between(np.ma.masked_where(Rseries<=0, self.tseries), np.ma.masked_where(Rseries<=0, topstack+Rseries), topstack, color=color_R, alpha=0.5, label='$R$', zorder=2)
            ax.plot(np.ma.masked_where(Rseries<=0, self.tseries), np.ma.masked_where(Rseries<=0, topstack+Rseries), color=color_R, zorder=3)
            topstack = topstack+Rseries
        if(any(Sseries) and plot_style=='stacked'):
            ax.fill_between(np.ma.masked_where(Sseries<=0, self.tseries), np.ma.masked_where(Sseries<=0, topstack+Sseries), topstack, color=color_S, alpha=0.5, label='$S$', zorder=2)
            ax.plot(np.ma.masked_where(Sseries<=0, self.tseries), np.ma.masked_where(Sseries<=0, topstack+Sseries), color=color_S, zorder=3)
            topstack = topstack+Sseries


        # -----------------------------------------------------------------------------------------------------------
        # Draw the plot labels:

        ax.set_xlabel('days', fontsize = 30)
        ax.set_ylabel('percent of porpulation' if plot_percentages else 'number of individuals', fontsize = 30)
        ax.set_xlim(0, (max(self.tseries) if not xlim else xlim))
        if (plot_style == "stacked"):
            ax.set_ylim(0, (max(topstack) if not ylim else ylim))
        else:
            ax.set_ylim(0, ylim)
        ax.set_xticklabels(['{:.0f}'.format(x) for x in ax.get_xticks()],fontsize = 25)
        ax.set_yticklabels(ax.get_yticks(),fontsize = 25)
        if(plot_percentages):
            ax.set_yticklabels(['{:,.0%}'.format(y) if y != 0 else "" for y in ax.get_yticks()])
        if(legend):
            legend_handles, legend_labels = ax.get_legend_handles_labels()
            ax.legend(legend_handles[::-1], legend_labels[::-1], loc='upper right', facecolor='white', edgecolor='none', framealpha=0.9, prop={'size': 15})
        if(title):
            ax.set_title(title, size=40)

        plt.show()

# -----------------------------------------------------------------------------------------------------------
# Editing :

class NetworkSEIRmodel():

    """
    A class to simulate the SEIR Stochastic Network Model
    ===================================================
    Params: G       Network adjacency matrix (numpy array) or Networkx graph object.
            beta    Rate of transmission (exposure)
            sigma   Rate of infection (upon exposure)
            gamma   Rate of recovery (upon infection)
            initS   Init number of suceptible individuals
            initE   Init number of exposed individuals
            initI   Init number of infectious individuals
            initR   Init number of recovered individuals
            store_Xserise = Store a current state

    """

    def __init__(self, G, beta, sigma, gamma, initS = 0, initE = 0, initI = 0, initR = 0, store_Xseries = False):

        # -----------------------------------------------------------------------------------------------------------
        # Setup Adjacency matrix :

        self.update_G(G)

        # -----------------------------------------------------------------------------------------------------------
        # Model Parameters :

        self.parameters = {"beta" : beta, "sigma" : sigma, "gamma" : gamma, "initS" : initS,"initE" : initE,
                           "initI" : initI, "initR" : initR}
        self.update_parameters()

        # -----------------------------------------------------------------------------------------------------------
        # Model state :

        self.tseries = np.zeros(4*self.numNodes)
        self.numS    = np.zeros(4*self.numNodes)
        self.numE    = np.zeros(4*self.numNodes)
        self.numI    = np.zeros(4*self.numNodes)
        self.numR    = np.zeros(4*self.numNodes)
        self.N       = np.zeros(4*self.numNodes)

        # -----------------------------------------------------------------------------------------------------------
        # Initialize Timekeeping :

        self.t          = 0
        self.tmax       = 0  # will be set when run() is called
        self.tidx       = 0
        self.tseries[0] = 0

        # -----------------------------------------------------------------------------------------------------------
        # Initialize Counts of inidividuals with each state:

        self.numS[0] = int(initS)
        self.numE[0] = int(initE)
        self.numI[0] = int(initI)
        self.numR[0] = int(initR)
        self.N[0]    = self.numS[0] + self.numE[0] + self.numI[0] + self.numR[0]

        # -----------------------------------------------------------------------------------------------------------
        # Node states :

        self.S = 1
        self.E = 2
        self.I = 3
        self.R = 4

        self.X = np.array([self.S]*int(self.numS[0]) + [self.E]*int(self.numE[0]) + [self.I]*int(self.numI[0]) + [self.R]*int(self.numR[0])).reshape((self.numNodes,1))
        np.random.shuffle(self.X)

        self.store_Xseries = store_Xseries
        if(store_Xseries):
            self.Xseries = np.zeros(shape=(4*self.numNodes, self.numNodes), dtype='uint8')
            self.Xseries[0,:] = self.X.T

        self.transitions =  {'StoE': {'currentState':self.S, 'newState':self.E},
                             'EtoI': {'currentState':self.E, 'newState':self.I},
                             'ItoR': {'currentState':self.I, 'newState':self.R}}

    def update_G(self, new_G):
        self.G = new_G

        # -----------------------------------------------------------------------------------------------------------
        # Adjacency matrix:

        if type(new_G)==np.ndarray:
            self.A = scipy.sparse.csr_matrix(new_G)
        elif type(new_G)==nx.classes.graph.Graph:
            self.A = nx.adj_matrix(new_G) # adj_matrix gives scipy.sparse csr_matrix
        else:
            raise BaseException("Input an adjacency matrix or networkx object only.")

        self.numNodes   = int(self.A.shape[1])
        self.degree     = np.asarray(self.node_degrees(self.A)).astype(float)

    def update_parameters(self):
        import time
        updatestart = time.time()

        # -----------------------------------------------------------------------------------------------------------
        # Model parameters:

        self.beta  = np.array(self.parameters['beta']).reshape((self.numNodes, 1))  if isinstance(self.parameters['beta'], (list, np.ndarray)) else np.full(fill_value=self.parameters['beta'], shape=(self.numNodes,1))
        self.sigma = np.array(self.parameters['sigma']).reshape((self.numNodes, 1)) if isinstance(self.parameters['sigma'], (list, np.ndarray)) else np.full(fill_value=self.parameters['sigma'], shape=(self.numNodes,1))
        self.gamma = np.array(self.parameters['gamma']).reshape((self.numNodes, 1)) if isinstance(self.parameters['gamma'], (list, np.ndarray)) else np.full(fill_value=self.parameters['gamma'], shape=(self.numNodes,1))

    def node_degrees(self, Amat):
        return Amat.sum(axis=0).reshape(self.numNodes,1)   # sums of adj matrix cols

    def calc_propensities(self):

        # -----------------------------------------------------------------------------------------------------------
        # Pre-calculate matrix multiplication terms that may be used in multiple propensity calculations,
        # and check to see if their computation is necessary before doing the multiplication

        numContacts_I       = np.zeros(shape=(self.numNodes,1))
        transmissionTerms_I = np.zeros(shape=(self.numNodes,1))

        if(np.any(self.numI[self.tidx]) and np.any(self.beta!=0)):
            transmissionTerms_I = np.asarray(scipy.sparse.csr_matrix.dot(self.beta, self.X==self.I))

        propensities_StoE   = np.divide(transmissionTerms_I, self.degree, out=np.zeros_like(self.degree), where=self.degree!=0)*(self.X==self.S)
        propensities_EtoI   = self.sigma*(self.X==self.E)
        propensities_ItoR   = self.gamma*(self.X==self.I)

        propensities = np.hstack([propensities_StoE, propensities_EtoI, propensities_ItoR])

        columns = ['StoE', 'EtoI', 'ItoR']

        return propensities, columns

    def increase_data_series_length(self):
        self.tseries = np.pad(self.tseries, [(0, 4*self.numNodes)], mode='constant', constant_values=0)
        self.numS = np.pad(self.numS, [(0, 4*self.numNodes)], mode='constant', constant_values=0)
        self.numE = np.pad(self.numE, [(0, 4*self.numNodes)], mode='constant', constant_values=0)
        self.numI = np.pad(self.numI, [(0, 4*self.numNodes)], mode='constant', constant_values=0)
        self.numR = np.pad(self.numR, [(0, 4*self.numNodes)], mode='constant', constant_values=0)
        self.N    = np.pad(self.N, [(0, 4*self.numNodes)], mode='constant', constant_values=0)

        if(self.store_Xseries):
            self.Xseries = np.pad(self.Xseries, [(0, 4*self.numNodes), (0,0)], mode=constant, constant_values=0)

    def finalize_data_series(self):
        self.tseries= np.array(self.tseries, dtype=float)[:self.tidx+1]
        self.numS   = np.array(self.numS, dtype=float)[:self.tidx+1]
        self.numE   = np.array(self.numE, dtype=float)[:self.tidx+1]
        self.numI   = np.array(self.numI, dtype=float)[:self.tidx+1]
        self.numR   = np.array(self.numR, dtype=float)[:self.tidx+1]
        self.N      = np.array(self.N, dtype=float)[:self.tidx+1]

        if(self.store_Xseries):
            self.Xseries = self.Xseries[:self.tidx+1, :]


    def run_iteration(self):

        if(self.tidx >= len(self.tseries)-1):
            # Room has run out in the timeseries storage arrays; double the size of these arrays:
            self.increase_data_series_length()

        # -----------------------------------------------------------------------------------------------------------
        # 1. Generate 2 random numbers uniformly distributed in (0,1)

        r1 = np.random.rand()
        r2 = np.random.rand()

        # -----------------------------------------------------------------------------------------------------------
        # 2. Calculate propensities

        propensities, transitionTypes = self.calc_propensities()

        # Terminate when probability of all events is 0:
        if(propensities.sum() <= 0.0):
            self.finalize_data_series()
            return False

        # -----------------------------------------------------------------------------------------------------------
        # 3. Calculate alpha

        propensities_flat   = propensities.ravel(order='F')
        cumsum              = propensities_flat.cumsum()
        alpha               = propensities_flat.sum()

        # -----------------------------------------------------------------------------------------------------------
        # 4. Compute the time until the next event takes place

        tau = (1/alpha)*np.log(float(1/r1))
        self.t += tau

        # -----------------------------------------------------------------------------------------------------------
        # 5. Compute which event takes place

        transitionIdx   = np.searchsorted(cumsum,r2*alpha)
        transitionNode  = transitionIdx % self.numNodes
        transitionType  = transitionTypes[ int(transitionIdx/self.numNodes) ]

        # -----------------------------------------------------------------------------------------------------------
        # 6. Update node states and data series

        assert(self.X[transitionNode] == self.transitions[transitionType]['currentState']), "Assertion error: Node "+str(transitionNode)+" has unexpected current state "+str(self.X[transitionNode])+" given the intended transition of "+str(transitionType)+"."
        self.X[transitionNode] = self.transitions[transitionType]['newState']

        self.tidx += 1

        self.tseries[self.tidx]  = self.t
        self.numS[self.tidx]     = np.clip(np.count_nonzero(self.X==self.S), a_min=0, a_max=self.numNodes)
        self.numE[self.tidx]     = np.clip(np.count_nonzero(self.X==self.E), a_min=0, a_max=self.numNodes)
        self.numI[self.tidx]     = np.clip(np.count_nonzero(self.X==self.I), a_min=0, a_max=self.numNodes)
        self.numR[self.tidx]     = np.clip(np.count_nonzero(self.X==self.R), a_min=0, a_max=self.numNodes)
        self.N[self.tidx]        = np.clip((self.numS[self.tidx] + self.numE[self.tidx] + self.numI[self.tidx] + self.numR[self.tidx]), a_min=0, a_max=self.numNodes)

        if(self.store_Xseries):
            self.Xseries[self.tidx,:] = self.X.T

        # -----------------------------------------------------------------------------------------------------------
        # Terminate if tmax reached or num infectious and num exposed is 0:

        if(self.t >= self.tmax or (self.numI[self.tidx]<1 and self.numE[self.tidx]<1)):
            self.finalize_data_series()
            return False

        # -----------------------------------------------------------------------------------------------------------

        return True


    def run(self, T, checkpoints=None, print_interval=10, verbose='t'):
        if(T>0):
            self.tmax += T
        else:
            return False

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Pre-process checkpoint values:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if(checkpoints):
            numCheckpoints = len(checkpoints['t'])
            for chkpt_param, chkpt_values in checkpoints.items():
                assert(isinstance(chkpt_values, (list, np.ndarray)) and len(chkpt_values)==numCheckpoints), "Expecting a list of values with length equal to number of checkpoint times ("+str(numCheckpoints)+") for each checkpoint parameter."
            checkpointIdx  = np.searchsorted(checkpoints['t'], self.t) # Finds 1st index in list greater than given val
            if(checkpointIdx >= numCheckpoints):
                # We are out of checkpoints, stop checking them:
                checkpoints = None
            else:
                checkpointTime = checkpoints['t'][checkpointIdx]

        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Run the simulation loop:
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        print_reset = True
        running     = True
        while running:

            running = self.run_iteration()

            # -----------------------------------------------------------------------------------------------------------
            # Handle checkpoints if applicable:
            if(checkpoints):
                if(self.t >= checkpointTime):
                    if(verbose is not False):
                        print("[Checkpoint: Updating parameters]")
                    # A checkpoint has been reached, update param values:
                    if('G' in list(checkpoints.keys())):
                        self.update_G(checkpoints['G'][checkpointIdx])
                    if('Q' in list(checkpoints.keys())):
                        self.update_Q(checkpoints['Q'][checkpointIdx])
                    for param in list(self.parameters.keys()):
                        if(param in list(checkpoints.keys())):
                            self.parameters.update({param: checkpoints[param][checkpointIdx]})
                    # Update parameter data structures and scenario flags:
                    self.update_parameters()
                    # Update the next checkpoint time:
                    checkpointIdx  = np.searchsorted(checkpoints['t'], self.t) # Finds 1st index in list greater than given val
                    if(checkpointIdx >= numCheckpoints):
                        # We are out of checkpoints, stop checking them:
                        checkpoints = None
                    else:
                        checkpointTime = checkpoints['t'][checkpointIdx]
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            if(print_interval):
                if(print_reset and (int(self.t) % print_interval == 0)):
                    if(verbose=="t"):
                        print("t = %.2f" % self.t)
                    if(verbose==True):
                        print("t = %.2f" % self.t)
                        print("\t S   = " + str(self.numS[self.tidx]))
                        print("\t E   = " + str(self.numE[self.tidx]))
                        print("\t I   = " + str(self.numI[self.tidx]))
                        print("\t R   = " + str(self.numR[self.tidx]))
                    print_reset = False
                elif(not print_reset and (int(self.t) % 10 != 0)):
                    print_reset = True

        return True
