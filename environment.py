"""This module contains various experimental environments used for testing 
human behavior."""
import numpy as np
            

class GridWorld(object):
    
    def __init__(self, Omega, Theta,
                 trials = 1, T = 10):
        
        #set probability distribution used for generating observations
        self.Omega = Omega.copy()
        
        #set probability distribution used for generating state transitions
        self.Theta = Theta.copy()
    
        #set container that keeps track the evolution of the hidden states
        self.hidden_states = np.zeros((trials, T), dtype = int)
    
    def set_initial_states(self, tau):
        #start in lower corner
        self.hidden_states[tau, 0] = 1
        
        if tau%100==0:
            print("trial:", tau)
        
    
    def generate_observations(self, tau, t):
        #generate one sample from multinomial distribution
        o = np.random.multinomial(1, self.Omega[:, self.hidden_states[tau, t]]).argmax()
        return o

    
    def update_hidden_states(self, tau, t, response):
        
        current_state = self.hidden_states[tau, t-1]        
        
        self.hidden_states[tau, t] = np.random.choice(self.Theta.shape[0], 
                          p = self.Theta[:, current_state, int(response)])
      
"""
test: please ignore
"""
class FakeGridWorld(object):
    
    def __init__(self, Omega, Theta,
                 hidden_states, trials = 1, T = 10):
        
        #set probability distribution used for generating observations
        self.Omega = Omega.copy()
        
        #set probability distribution used for generating state transitions
        self.Theta = Theta.copy()
    
        #set container that keeps track the evolution of the hidden states
        self.hidden_states = np.zeros((trials, T), dtype = int)
        self.hidden_states[:] = np.array([hidden_states for i in range(trials)])
    
    def set_initial_states(self, tau):
        #start in lower corner
        self.hidden_states[tau, 0] = 1
        
        #print("trial:", tau)
        
    
    def generate_observations(self, tau, t):
        #generate one sample from multinomial distribution
        o = np.random.multinomial(1, self.Omega[:, self.hidden_states[tau, t]]).argmax()
        return o

    
    def update_hidden_states(self, tau, t, response):
        
        current_state = self.hidden_states[tau, t-1]        
        
        self.hidden_states[tau, t] = np.random.choice(self.Theta.shape[0], 
                          p = self.Theta[:, current_state, int(response)])