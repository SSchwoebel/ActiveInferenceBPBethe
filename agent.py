"""This module contains the class that defines the interaction between
different modules that govern agent's behavior.
"""
import numpy as np
from perception import BethePerception
from misc import ln, softmax

        
class BayesianPlanner(object):
    
    def __init__(self, perception, action_selection, policies,
                 prior_states = None, prior_policies = None, 
                 trials = 1, T = 10, number_of_states = 6, 
                 number_of_policies = 10):
        
        #set the modules of the agent
        self.perception = perception
        self.action_selection = action_selection
        
        #set parameters of the agent
        self.nh = number_of_states #number of states
        self.npi = number_of_policies #number of policies
        
        if policies is not None:
            self.policies = policies
        else:
            #make action sequences for each policy
            self.policies = np.eye(self.npi, dtype = int)
        
        self.actions = np.unique(self.policies)
        self.na = len(self.actions)
        
        if prior_states is not None:
            self.prior_states = prior_states
        else:
            self.prior_states = np.ones(self.nh)
            self.prior_states /= self.prior_states.sum()
            
        if prior_policies is not None:
            self.prior_policies = prior_policies
        else:
            self.prior_policies = np.ones(self.npi)/self.npi
        
        #set various data structures
        self.actions = np.zeros((trials, T), dtype = int)
        self.posterior_states = np.zeros((trials, T, self.nh, T, self.npi))
        self.posterior_policies = np.zeros((trials, T, self.npi))
        self.observations = np.zeros((trials, T), dtype = int)
        

    def reset_beliefs(self, actions):
        self.actions[:,:] = actions 
        self.posterior_states[:,:,:] = 0.
        self.posterior_policies[:,:,:] = 0.
        
        self.perception.reset_beliefs()
        self.planning.reset_beliefs()
        self.action_selection.reset_beliefs()
        
        
    def update_beliefs(self, tau, t, observation, response):
        self.observations[tau,t] = observation
        #update beliefs about hidden states
        if t == 0:
            control = None
            prior_states = self.prior_states.copy()
            prior_policies = self.prior_policies
        else:
            control = self.actions[tau, t-1]
            prior_states = self.posterior_states[tau, t-1]
            prior_policies = self.posterior_policies[tau, t-1]

        #update beliefs about states   
        if isinstance(self.perception, BethePerception):
            self.posterior_states[tau, t] = self.perception.update_beliefs_states(
                                             tau, t,
                                             observation,
                                             self.policies,
                                             prior_policies)
            
        else:
            self.posterior_states[tau, t] = self.perception.update_beliefs_states(
                                             observation,
                                             control,
                                             prior_states)
        
        #update beliefs about policies
        self.posterior_policies[tau, t] = self.perception.update_beliefs_policies()
            

    
    def generate_response(self, tau, t):
        
        #get response probability
        posterior_policies = self.posterior_policies[tau, t]
        non_zero = posterior_policies > 0
        controls = self.policies[:, t][non_zero]
        posterior_policies = posterior_policies[non_zero]

        self.actions[tau, t] = self.action_selection.select_desired_action(tau, 
                                        t, posterior_policies, controls)
            
        
        return self.actions[tau, t]
    


class BayesianMFPlanner(object):
    
    def __init__(self, perception, planning, action_selection,
                 prior_beliefs = None, prior_states = None, 
                 prior_policies = None, policies = None,
                 trials = 1, T = 10, number_of_states = 6, 
                 number_of_policies = 10):
        
        #set the modules of the agent
        self.perception = perception
        self.planning = planning
        self.action_selection = action_selection
        
        #set parameters of the agent
        self.nh = number_of_states #number of states
        self.npi = number_of_policies #number of policies
        
        if policies is not None:
            self.policies = policies
        else:
            #make action sequences for each policy
            self.policies = np.eye(self.npi, dtype = int)
        
        self.actions = np.unique(self.policies)
        self.na = len(self.actions)
        
        if prior_beliefs is not None:
            self.prior_beliefs = prior_beliefs
        else:
            self.prior_beliefs = np.zeros((self.nh, T, self.npi))
            if prior_states is not None:
                self.prior_beliefs[:, 0, :] = 1/self.nh 
#                self.prior_beliefs[:, 0, :] = prior_states[:, np.newaxis]
            else:
                self.prior_beliefs[:,0,:] = 1./self.nh
            
            for pi, cstates in enumerate(policies):
                for t, u in enumerate(cstates):
                    p = self.perception.generative_model_states[:,:,u]\
                        .dot(self.prior_beliefs[:,t,pi])
                    p[:] = 1.
#                    p[p>1e-10] = 1.
                    p /= p.sum()
                    self.prior_beliefs[:, t+1, pi] = p[:]
            
        if prior_policies is not None:
            self.prior_policies = prior_policies
        else:
            self.prior_policies = np.ones(self.npi)/self.npi
        
        #set various data structures
        self.actions = np.zeros((trials, T), dtype = int)
        self.posterior_states = np.zeros((trials, T, self.nh, T, self.npi))
        self.posterior_policies = np.zeros((trials, T, self.npi))
        self.observations = np.zeros((trials, T), dtype = int)
        
    def update_beliefs(self, tau, t, observation, response):
        self.observations[tau,t] = observation
        #update beliefs about hidden states
        if t == 0:
#            control = None
            prior_states = self.prior_beliefs.copy()
            prior_policies = self.prior_policies.copy()
        else:
#            control = self.actions[tau, t-1]
            prior_states = self.prior_beliefs.copy() #self.posterior_states[tau, t-1].copy()
            prior_policies = self.posterior_policies[tau, t-1].copy()

        #update beliefs about states      
        self.posterior_states[tau, t], neg_fe_pi = self.perception.update_beliefs_states(
                                             tau, t,
                                             observation,
                                             self.policies,
                                             prior_states,
                                             prior_policies,)
        
        #update beliefs about policies
        self.posterior_policies[tau, t] = \
            self.perception.update_beliefs_policies()

    
    def generate_response(self, tau, t):
        
        #get response probability
        posterior_policies = self.posterior_policies[tau, t]
        non_zero = posterior_policies >= 1e-6
        controls = self.policies[:, t][non_zero]
        posterior_policies = posterior_policies[non_zero]
        posterior_policies /= posterior_policies.sum()

        self.actions[tau, t] = self.action_selection.select_desired_action(tau, 
                                        t, posterior_policies, controls)
        
        return self.actions[tau, t]

