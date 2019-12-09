#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

def evolve_environment(env):
    trials = env.hidden_states.shape[0]
    T = env.hidden_states.shape[1]
    
    for tau in range(trials):
        for t in range(T):
            if t == 0:
                env.set_initial_states(tau)
            else:
                if t < T/2:
                    env.update_hidden_states(tau, t, 0)
                else:
                    env.update_hidden_states(tau, t, 1)
                    
                    
def compute_performance(rewards):
    return rewards.mean(), rewards.var()


def ln(x):
    with np.errstate(divide='ignore'):
        return np.nan_to_num(np.log(x))
    
def logit(x):
    with np.errstate(divide = 'ignore'):
        return np.nan_to_num(np.log(x/(1-x)))
    
def logistic(x):
    return 1/(1+np.exp(-x))

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis = 0))
    return e_x / e_x.sum(axis = 0)

def lognormal(x, mu, sigma):
    return -.5*(x-mu)*(x-mu)/(2*sigma) - .5*ln(2*np.pi*sigma)