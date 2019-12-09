from misc import ln, softmax
import numpy as np
    
class BethePerception(object):
    def __init__(self,
                 generative_model_observations, 
                 generative_model_states,
                 prior_states,
                 prior_observations,
                 T=5):

        self.generative_model_observations = generative_model_observations
        self.generative_model_states = generative_model_states
        self.prior_observations = prior_observations
        self.prior_states = prior_states
        self.T = T
        self.nh = prior_states.shape[0]
        
    def instantiate_messages(self, policies):
        npi = policies.shape[0]
        
        self.bwd_messages = np.zeros((self.nh, self.T, npi))
        self.bwd_messages[:,-1,:] = 1./self.nh
        self.fwd_messages = np.zeros((self.nh, self.T, npi))
        self.fwd_messages[:, 0, :] = self.prior_states[:, np.newaxis]
        
        self.fwd_norms = np.zeros((self.T+1, npi))
        self.fwd_norms[0,:] = 1.
        
        self.obs_messages = self.prior_observations.dot(self.generative_model_observations)
        self.obs_messages = np.tile(self.obs_messages,(self.T,1)).T

        for pi, cstates in enumerate(policies):
            for t, u in enumerate(np.flip(cstates, axis = 0)):
                tp = self.T - 2 - t
                self.bwd_messages[:,tp,pi] = self.bwd_messages[:,tp+1,pi]*\
                                            self.obs_messages[:, tp+1]
                self.bwd_messages[:,tp,pi] = self.bwd_messages[:,tp,pi]\
                    .dot(self.generative_model_states[:,:,u])
                self.bwd_messages[:,tp, pi] /= self.bwd_messages[:,tp,pi].sum()
                
    def update_messages(self, t, pi, cs):
        if t > 0:
            for i, u in enumerate(np.flip(cs[:t], axis = 0)):
                self.bwd_messages[:,t-1-i,pi] = self.bwd_messages[:,t-i,pi]*\
                                                self.obs_messages[:,t-i]
                self.bwd_messages[:,t-1-i,pi] = self.bwd_messages[:,t-1-i,pi]\
                    .dot(self.generative_model_states[:,:,u])
                
                norm = self.bwd_messages[:,t-1-i,pi].sum()
                if norm > 0:
                    self.bwd_messages[:,t-1-i, pi] /= norm
        
        if len(cs[t:]) > 0:
           for i, u in enumerate(cs[t:]):
               self.fwd_messages[:, t+1+i, pi] = self.fwd_messages[:,t+i, pi]*\
                                                self.obs_messages[:, t+i]
               self.fwd_messages[:, t+1+i, pi] = \
                                                self.generative_model_states[:,:,u].\
                                                dot(self.fwd_messages[:, t+1+i, pi])
               self.fwd_norms[t+1+i,pi] = self.fwd_messages[:,t+1+i,pi].sum()
               if self.fwd_norms[t+1+i, pi] > 0: #???? Shouldn't this not happen?
                   self.fwd_messages[:,t+1+i, pi] /= self.fwd_norms[t+1+i,pi]

    def update_beliefs_states(self, tau, t, observation, policies, prior_pi):
        #estimate expected state distribution
        if t == 0:
            self.instantiate_messages(policies)
        
        self.obs_messages[:,t] = self.generative_model_observations[observation]
        
        for pi, cs in enumerate(policies):
            if prior_pi[pi] > 1e-10:
                self.update_messages(t, pi, cs)
            else:
                self.fwd_messages[:,:,pi] = 0
        
        #estimate posterior state distribution
        posterior = self.fwd_messages*self.bwd_messages*self.obs_messages[:,:,np.newaxis]
        norm = posterior.sum(axis = 0)
        self.fwd_norms[-1] = norm[-1]
        posterior /= norm
        return np.nan_to_num(posterior)
        
    def update_beliefs_policies(self):
        
        posterior = softmax(ln(self.fwd_norms).sum(axis = 0))
        
        return posterior
    
class MFPerception(object):
    def __init__(self, generative_model_observations, 
                       generative_model_states, 
                       prior_states,
                       prior_observations,
                       T=5):

        self.generative_model_observations = generative_model_observations + 1e-16
        self.generative_model_observations /= self.generative_model_observations.sum(axis = 0)
        self.generative_model_states = generative_model_states + 1e-16
        self.generative_model_states /= generative_model_states.sum(axis = 0)
        self.prior_observations = prior_observations + 1e-16
        self.prior_observations /= self.prior_observations.sum()
        self.prior_states = prior_states
        self.T = T
        self.nh = prior_states.shape[0]
        
        self.zs = self.prior_observations.dot(self.generative_model_observations)
        
    def reset_beliefs(self):
        pass
    
    def set_params(self, x):
        pass

    
    def update_beliefs_states(self, tau, t, observation, policies,
                              prior, prior_pi):
        if t==0:
            self.logzs = np.tile(ln(self.zs), (self.T,1)).T
        self.logzs[:,t] = ln(self.generative_model_observations[int(observation),:])
        
        #estimate expected state distribution
        lforw = np.zeros((self.nh, self.T))
        lforw[:,0] = ln(self.prior_states)
        lback = np.zeros((self.nh, self.T))
        posterior = np.zeros((self.nh, self.T, policies.shape[0]))
        neg_fe = np.zeros(policies.shape[0])
        eps = 0.01
        for pi, ppi in enumerate(prior_pi):
            if ppi > 1e-6:
                logtm = ln(self.generative_model_states[:,:,policies[pi]])
                #SARAH: check the following before publishing!
                post = prior[:,:,pi]
                not_close = True
                while not_close:
                    lforw[:,1:] = np.einsum('ijk, jk-> ik',logtm,post[:,:-1])
                    lback[:,:-1] = np.einsum('ijk, ik->jk',logtm,post[:,1:])
                    logpost = lforw + self.logzs 
                    lp = ln(post)
                    lp = (1-eps)*lp + eps*(logpost + lback)
                    new_post = softmax(lp)
                    not_close = not np.allclose(post, new_post, atol = 1e-3)
                    post[:]=new_post
                
                posterior[:,:,pi] = post  
                neg_fe[pi] = (logpost*post).sum() - np.sum(post*ln(post))
            else:
                posterior[:,:,pi] = prior[:,:,pi]
                neg_fe[pi] = -1e10
        
        self.fe_pi = neg_fe
        
        return posterior, neg_fe
        
    
    def update_beliefs_policies(self):
        posterior = softmax(self.fe_pi)
                
        return posterior
