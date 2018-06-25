"""
Template for implementing QLearner  (c) 2015 Tucker Balch

Student: Si Yan
userID: syan62

"""

import numpy as np
import random as rand

class QLearner(object):

    def __init__(self, \
        num_states=100, \
        num_actions = 4, \
        alpha = 0.2, \
        gamma = 0.9, \
        rar = 0.5, \
        radr = 0.99, \
        dyna = 0, \
        verbose = False):

        self.verbose = verbose
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna
        self.s = 0
        self.a = 0
        
        self.Q_table = np.zeros((self.num_states, self.num_actions))
        #self.Tc = np.ones((self.num_states, self.num_actions, self.num_states))*0.00001
        self.Tc = np.zeros((self.num_states, self.num_actions, self.num_states))
        self.T = np.zeros((self.num_states, self.num_actions, self.num_states))
        self.R = np.zeros((self.num_states, self.num_actions))
        
    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """
        
        #action = rand.randint(0, self.num_actions-1)
        #self.a = action
        if rand.uniform(0.0, 1.0) <= self.rar:
            action = rand.randint(0, self.num_actions-1)
        else:    
            action = np.argmax(self.Q_table[s])
            
        self.s = s
        self.a = action            
        if self.verbose: print "s =", s,"a =",action
        return action

    def query(self,s_prime,r):
        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: The ne state
        @returns: The selected action
        """
        #action = rand.randint(0, self.num_actions-1)
        if rand.uniform(0.0, 1.0) <= self.rar:
            action = rand.randint(0, self.num_actions-1)
            #self.rar *= self.radr
        else:    
            action = np.argmax(self.Q_table[s_prime])
            
        self.Q_table[self.s, self.a] = (1-self.alpha)* self.Q_table[self.s, self.a] \
        + self.alpha * (r + self.gamma * self.Q_table[s_prime, action]) 
        self.rar *= self.radr
        
        #for dyna
        if self.dyna > 0:
            self.Tc[self.s, self.a, s_prime] += 1
            #self.T[self.s, self.a, s_prime] = self.Tc[self.s, self.a, s_prime]/(self.Tc[self.s, self.a].sum())
            self.R[self.s, self.a] = (1-self.alpha) * self.R[self.s, self.a] + self.alpha * r
            
            
            count = 0
            while (count < self.dyna):
                s_rand = rand.randint(0, self.num_states-1)
                a_rand = rand.randint(0, self.num_actions-1)
                s_p = np.argmax(self.Tc[s_rand, a_rand])
                #s_p = rand.randint(0, self.num_states-1)
                r_p = self.R[s_rand, a_rand]
                
                self.Q_table[s_rand, a_rand] = (1-self.alpha)* self.Q_table[s_rand, a_rand] \
                    + self.alpha * (r_p + self.gamma * max(self.Q_table[s_p]))
                count += 1
                
                #print self.Tc[s_rand, a_rand]
                #print s_p
                #print self.Q_table.sum()
               
        self.s = s_prime
        self.a = action    
        
        #self.rar *= self.radr
        
        if self.verbose: print "s =", s_prime,"a =",action,"r =",r
        return action
    
    
    
    def author(self):
        return 'syan62'
    
if __name__=="__main__":
    print "Remember Q from Star Trek? Well, this isn't him"
