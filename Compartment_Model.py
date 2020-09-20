#!/usr/bin/env python
# coding: utf-8

# In[61]:


import networkx as nx
import numpy as np
import scipy as sp
import scipy.integrate

import matplotlib.pyplot as plt

class SImodel():
    """
    A class to simulate the Deterministic SI Model
    ===================================================
    Params: beta    Rate of transmission

            initN   The number of total subject
            initI   Init number of infectious individuals
    """

    def __init__(self, initN, beta, initI = 0):

        # -----------------------------------------------------------------------------------------------------------
        # Model Parameters:

        self.beta   = beta

        # -----------------------------------------------------------------------------------------------------------
        # Initialize Timekeeping:

        self.t       = 0
        self.tmax    = 0 # will be set when run() is called
        self.tseries = np.array([0])

        # -----------------------------------------------------------------------------------------------------------
        # Initialize Counts of inidividuals with each state:

        self.N          = np.array([int(initN)])
        self.numI       = np.array([int(initI)])
        self.numS       = np.array([self.N[-1] - self.numI[-1]])
        assert(self.numS[0] >= 0), "The specified initial population size N must be greater than or equal to the initial compartment counts."


    def system_dfes(t, variables, beta):

        S, I = variables    # varibles is a list with compartment counts as elements

        N   = S + I
        dS  = - (beta*S*I)/N
        dI  = (beta*S*I)/N

        return [dS, dI]


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

        init_cond = [self.numS[-1], self.numI[-1]]

        # -----------------------------------------------------------------------------------------------------------
        # Solve the system of differential eqns:

        solution = scipy.integrate.solve_ivp(lambda t, X: SImodel.system_dfes(t, X, self.beta),
                                             t_span=[self.t, self.tmax], y0=init_cond, t_eval=t_eval)

        # -----------------------------------------------------------------------------------------------------------
        # Store the solution output as the model's time series and data series:

        self.tseries    = np.append(self.tseries, solution['t'])
        self.numS       = np.append(self.numS, solution['y'][0])
        self.numI       = np.append(self.numI, solution['y'][1])

        self.t = self.tseries[-1]


    def run(self, T, dt=0.1, verbose=False):

        assert(T > 0), "Total simulation time T must be larger than 0"
        self.tmax += T

        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Run the simulation loop:
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        self.run_epoch(runtime=self.tmax, dt=dt)

        # -----------------------------------------------------------------------------------------------------------

        if(verbose):
            print("t = %.2f" % self.t)
            print("\t S   = " + str(self.numS[-1]))
            print("\t I   = " + str(self.numI[-1]))

        return True

    def plot(self, title, S_plot = "line", I_plot = "line", color_S='tab:green', color_I='crimson', xlim = None, ylim = None, plot_percentages = True, legend = True, figsize = (12,8), show = True):

        # -----------------------------------------------------------------------------------------------------------
        # Create an Axes object:

        fig, ax = plt.subplots(figsize = figsize)

        # -----------------------------------------------------------------------------------------------------------
        # Prepare data series to be plotted:

        Iseries     = self.numI/self.N if plot_percentages else self.numI
        Sseries     = self.numS/self.N if plot_percentages else self.numS

        # -----------------------------------------------------------------------------------------------------------
        # Draw the line plot:

        if(I_plot =='line'):
            ax.plot(np.ma.masked_where(Iseries<=0, self.tseries), np.ma.masked_where(Iseries<=0, Iseries), color=color_I, label='$I$', zorder=6)
        if(S_plot =='line'):
            ax.plot(np.ma.masked_where(Sseries<=0, self.tseries), np.ma.masked_where(Sseries<=0, Sseries), color=color_S, label='$S$', zorder=6)

        # -----------------------------------------------------------------------------------------------------------
        # Draw the stacked plot:

        topstack = np.zeros_like(self.tseries)

        if(I_plot=='stacked'):
            ax.fill_between(np.ma.masked_where(Iseries<=0, self.tseries), np.ma.masked_where(Iseries<=0, topstack+Iseries), topstack, color=color_I, alpha=0.5, label='$I$', zorder=2)
            ax.plot(np.ma.masked_where(Iseries<=0, self.tseries), np.ma.masked_where(Iseries<=0, topstack+Iseries), color=color_I, zorder=3)
            topstack = topstack+Iseries
        if(S_plot=='stacked'):
            ax.fill_between(np.ma.masked_where(Sseries<=0, self.tseries), np.ma.masked_where(Sseries<=0, topstack+Sseries), topstack, color=color_S, alpha=0.5, label='$S$', zorder=2)
            ax.plot(np.ma.masked_where(Sseries<=0, self.tseries), np.ma.masked_where(Sseries<=0, topstack+Sseries), color=color_S, zorder=3)
            topstack = topstack+Sseries

        # -----------------------------------------------------------------------------------------------------------
        # Draw the plot labels:

        ax.set_xlabel('simulation time', fontsize = 30)
        ax.set_ylabel('percent of population' if plot_percentages else 'number of individuals', fontsize = 30)
        ax.set_xlim(0, (max(self.tseries) if not xlim else xlim))
        if ((S_plot == "stacked") or (I_plot == "stacked")):
            ax.set_ylim(0, (max(topstack) if not ylim else ylim))
        else:
            ax.set_ylim(0, ylim)
        ax.set_xticklabels(['{:.0f}'.format(x) for x in ax.get_xticks()],fontsize = 25)
        if(plot_percentages):
            ax.set_yticklabels(['{:,.0%}'.format(y) if y != 0 else "" for y in ax.get_yticks()], fontsize = 25)
        else :
            ax.set_yticklabels(ax.get_yticks(),fontsize = 25)
        if(legend):
            legend_handles, legend_labels = ax.get_legend_handles_labels()
            ax.legend(legend_handles[::-1], legend_labels[::-1], loc='upper right', facecolor='white', edgecolor='none', framealpha=0.9, prop={'size': 15})
        if(title):
            ax.set_title(title, size=40)
        if(show):
            plt.show()

class SISmodel():
    """
    A class to simulate the Deterministic SIS Model
    ===================================================
    Params: beta    Rate of transmission
            gamma    Rate of recover

            initN   The number of total subject
            initI   Init number of infectious individuals
    """

    def __init__(self, initN, beta, gamma, initI = 0):

        # -----------------------------------------------------------------------------------------------------------
        # Model Parameters:

        self.beta   = beta
        self.gamma   = gamma


        # -----------------------------------------------------------------------------------------------------------
        # Initialize Timekeeping:

        self.t       = 0
        self.tmax    = 0 # will be set when run() is called
        self.tseries = np.array([0])

        # -----------------------------------------------------------------------------------------------------------
        # Initialize Counts of inidividuals with each state:

        self.N          = np.array([int(initN)])
        self.numI       = np.array([int(initI)])
        self.numS       = np.array([self.N[-1] - self.numI[-1]])
        assert(self.numS[0] >= 0), "The specified initial population size N must be greater than or equal to the initial compartment counts."


    def system_dfes(t, variables, beta, gamma):

        S, I = variables    # varibles is a list with compartment counts as elements

        N   = S + I
        dS  = - (beta*S*I)/N + (gamma*I)
        dI  = (beta*S*I)/N - (gamma*I)

        return [dS, dI]


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

        init_cond = [self.numS[-1], self.numI[-1]]

        # -----------------------------------------------------------------------------------------------------------
        # Solve the system of differential eqns:

        solution = scipy.integrate.solve_ivp(lambda t, X: SISmodel.system_dfes(t, X, self.beta, self.gamma),
                                             t_span=[self.t, self.tmax], y0=init_cond, t_eval=t_eval)

        # -----------------------------------------------------------------------------------------------------------
        # Store the solution output as the model's time series and data series:

        self.tseries    = np.append(self.tseries, solution['t'])
        self.numS       = np.append(self.numS, solution['y'][0])
        self.numI       = np.append(self.numI, solution['y'][1])

        self.t = self.tseries[-1]


    def run(self, T, dt=0.1, verbose=False):

        assert(T > 0), "Total simulation time T must be larger than 0"
        self.tmax += T

        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Run the simulation loop:
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        self.run_epoch(runtime=self.tmax, dt=dt)

        # -----------------------------------------------------------------------------------------------------------

        if(verbose):
            print("t = %.2f" % self.t)
            print("\t S   = " + str(self.numS[-1]))
            print("\t I   = " + str(self.numI[-1]))

        return True

    def plot(self, title, S_plot = "line", I_plot = "line", color_S='tab:green', color_I='crimson', xlim = None, ylim = None, plot_percentages = True, legend = True, figsize = (12,8), show = True):

        # -----------------------------------------------------------------------------------------------------------
        # Create an Axes object:

        fig, ax = plt.subplots(figsize = figsize)

        # -----------------------------------------------------------------------------------------------------------
        # Prepare data series to be plotted:

        Iseries     = self.numI/self.N if plot_percentages else self.numI
        Sseries     = self.numS/self.N if plot_percentages else self.numS

        # -----------------------------------------------------------------------------------------------------------
        # Draw the line plot:

        if(I_plot =='line'):
            ax.plot(np.ma.masked_where(Iseries<=0, self.tseries), np.ma.masked_where(Iseries<=0, Iseries), color=color_I, label='$I$', zorder=6)
        if(S_plot =='line'):
            ax.plot(np.ma.masked_where(Sseries<=0, self.tseries), np.ma.masked_where(Sseries<=0, Sseries), color=color_S, label='$S$', zorder=6)

        # -----------------------------------------------------------------------------------------------------------
        # Draw the stacked plot:

        topstack = np.zeros_like(self.tseries)

        if(I_plot=='stacked'):
            ax.fill_between(np.ma.masked_where(Iseries<=0, self.tseries), np.ma.masked_where(Iseries<=0, topstack+Iseries), topstack, color=color_I, alpha=0.5, label='$I$', zorder=2)
            ax.plot(np.ma.masked_where(Iseries<=0, self.tseries), np.ma.masked_where(Iseries<=0, topstack+Iseries), color=color_I, zorder=3)
            topstack = topstack+Iseries
        if(S_plot=='stacked'):
            ax.fill_between(np.ma.masked_where(Sseries<=0, self.tseries), np.ma.masked_where(Sseries<=0, topstack+Sseries), topstack, color=color_S, alpha=0.5, label='$S$', zorder=2)
            ax.plot(np.ma.masked_where(Sseries<=0, self.tseries), np.ma.masked_where(Sseries<=0, topstack+Sseries), color=color_S, zorder=3)
            topstack = topstack+Sseries


        # -----------------------------------------------------------------------------------------------------------
        # Draw the plot labels:

        ax.set_xlabel('simulation time', fontsize = 30)
        ax.set_ylabel('percent of population' if plot_percentages else 'number of individuals', fontsize = 30)
        ax.set_xlim(0, (max(self.tseries) if not xlim else xlim))
        if ((S_plot == "stacked") or (I_plot == "stacked")):
            ax.set_ylim(0, (max(topstack) if not ylim else ylim))
        else:
            ax.set_ylim(0, ylim)
        ax.set_xticklabels(['{:.0f}'.format(x) for x in ax.get_xticks()],fontsize = 25)
        if(plot_percentages):
            ax.set_yticklabels(['{:,.0%}'.format(y) if y != 0 else "" for y in ax.get_yticks()], fontsize = 25)
        else :
            ax.set_yticklabels(ax.get_yticks(),fontsize = 25)
        if(legend):
            legend_handles, legend_labels = ax.get_legend_handles_labels()
            ax.legend(legend_handles[::-1], legend_labels[::-1], loc='upper right', facecolor='white', edgecolor='none', framealpha=0.9, prop={'size': 15})
        if(title):
            ax.set_title(title, size=40)
        if(show):
            plt.show()
