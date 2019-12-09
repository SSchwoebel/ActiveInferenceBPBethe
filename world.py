"""This module contains the World class that defines interactions between 
the environment and the agent. It also keeps track of all observations and 
actions generated during a single experiment. To initiate it one needs to 
provide the environment class and the agent class that will be used for the 
experiment.
"""
import numpy as np
from misc import ln

class World(object):
    
    def __init__(self, environment, agent, trials = 1, T = 10):
        #set inital elements of the world to None        
        self.environment = environment
        self.agent = agent

        self.trials = trials # number of trials in the experiment
        self.T = T # number of time steps in each trial
        
        self.free_parameters = {}
        
        #container for observations
        self.observations = np.zeros((self.trials, self.T), dtype = int)
                
        #container for agents actions
        self.actions = np.zeros((self.trials, self.T), dtype = int)
        
    def simulate_experiment(self):
        """This methods evolves all the states of the world by iterating 
        through all the trials and time steps of each trial.
        """
        
        for tau in range(self.trials):
            for t in range(self.T):
                self.__update_world(tau, t)
    
    #this is a private method do not call it outside of the class    
    def __update_world(self, tau, t):
        """This private method performs a signel time step update of the 
        whole world. Here we update the hidden state(s) of the environment, 
        the perceptual and planning states of the agent, and in parallel we 
        generate observations and actions.
        """
        
        if t==0:
            self.environment.set_initial_states(tau)
            response = None
        else:
            response = self.actions[tau, t-1]
            self.environment.update_hidden_states(tau, t, response)
                                                      
        self.observations[tau, t] = \
            self.environment.generate_observations(tau, t)
            
        observation = self.observations[tau, t]
        
    
        self.agent.update_beliefs(tau, t, observation, response)
        
        
        if t < self.T-1:
            self.actions[tau, t] = self.agent.generate_response(tau, t)
        else:
            self.actions[tau, t] = -1

        

class FakeWorld(object):
    
    def __init__(self, environment, agent, observations, actions, trials = 1, T = 10):
        #set inital elements of the world to None        
        self.environment = environment
        self.agent = agent

        self.trials = trials # number of trials in the experiment
        self.T = T # number of time steps in each trial
        
        self.free_parameters = {}
        
        #container for observations
        self.observations = np.zeros((self.trials, self.T), dtype = int)
        self.observations[:] = np.array([observations for i in range(self.trials)])
                
        #container for agents actions
        self.actions = np.zeros((self.trials, self.T), dtype = int)
        self.actions[:] = np.array([actions for i in range(self.trials)])
        
    def simulate_experiment(self):
        """This methods evolves all the states of the world by iterating 
        through all the trials and time steps of each trial.
        """
        
        for tau in range(self.trials):
            for t in range(self.T):
                self.__update_world(tau, t)

    
    #this is a private method do not call it outside of the class    
    def __update_world(self, tau, t):
        """This private method performs a signel time step update of the 
        whole world. Here we update the hidden state(s) of the environment, 
        the perceptual and planning states of the agent, and in parallel we 
        generate observations and actions.
        """
        #print(tau, t)
        if t==0:
            self.environment.set_initial_states(tau)
            response = None
        else:
            response = self.actions[tau, t-1]
                                                      
            
        observation = self.observations[tau, t]
        
    
        self.agent.update_beliefs(tau, t, observation, response)
        
        self.agent.plan_behavior(tau, t)
        