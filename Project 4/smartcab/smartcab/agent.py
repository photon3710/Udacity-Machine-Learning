import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import numpy as np

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env, alpha, gamma):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.alpha0 = alpha
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = 1 # small possibility to choose feasible actions randomly        
        self.Q = np.zeros((12,4))
        # Metric parameters
        self.count = 0 # randomly choosing the actions before using the Q values
        self.timeStep = 0 # time cost
        self.trialNum = 0 # number of trials
        
        
    def reset(self, destination=None):
        # record the previous settings
        # print "deadline step = {}, time step = {}, penalty number = {}".format(self.env.get_deadline(self), self.timeStep, self.penaltyNum) # [debug]
                   
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required 
        if self.count >= 300:
            self.trialNum += 1  
            self.epsilon = 1. / self.trialNum # the training strategy is to shift the training from exploration to exploitation.
            # self.alpha = self.alpha0 / self.trialNum        
        self.timeStep = 0
        

    def update(self, t):
        # Gather inputs
        deadline = self.env.get_deadline(self)
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)       
        # TODO: Update state
        self.state = (self.next_waypoint, inputs['light'], inputs['left'], inputs['oncoming'])        
        s = self.getStateIndex(self.state)
		       		
        # TODO: Select action according to your policy 
        if self.count < 300:    # fully randomly choosing actions at the beginneing of the training   
            a = random.choice( [0, 1, 2, 3] )
        else:
            if random.uniform(0, 1) < self.epsilon:  # With the probability epsilon, the car will choose randomly to explore
                a = random.choice( [0, 1, 2, 3] )           
            else:          
                a = random.choice( np.argwhere( self.Q[s,:] == np.amax( self.Q[s,:] ) ).flatten() )
        
        if a == 0:
            action = None
        elif a == 1:
            action = 'right'
        elif a == 2:
            action = 'forward'
        else:
		    action = 'left'

        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        self.state = (self.next_waypoint, inputs['light'], inputs['left'], inputs['oncoming']) 
        sNext = self.getStateIndex(self.state)
		
        self.Q[s,a] = (1-self.alpha) * self.Q[s,a] + self.alpha * ( reward + self.gamma * np.amax(self.Q[sNext,:]) )
        
        # update epsilon, which gradually changes the agent from exploration to exploitation.       
        self.count += 1
        self.timeStep += 1

        # print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]

    def getStateIndex(self, state):
        if state[0] == 'right':
            n = 0
        elif state[0] == 'forward':
            n = 1
        elif state[0] == 'left':
            n = 2
            
        if state[1] == 'red' and state[2] == 'forward':
            s = 0
        elif state[1] == 'red' and state[2] != 'forward':
            s = 1
        elif state[1] == 'green' and (state[3] == 'forward' or state[3] == 'right'):
            s = 2
        else:
            s = 3
            
        return s*3+n

def run():
    """Run the agent for a finite number of trials."""
    np.set_printoptions(suppress=True)
    alpha = np.arange(0.1,1.01,0.1)
    gamma = np.arange(0.1,1.01,0.1)
    maxQOptimalActionNumber = 0
    minAlpha = 0
    minGamma = 0
    minQ = []
    
    for al in alpha:
        for ga in gamma: 
            print "Alpha = {}, Gamma = {}".format(al, ga)   
            # Set up environment and agent
            e = Environment()  # create environment (also adds some dummy traffic)
            a = e.create_agent(LearningAgent, al, ga)  # create agent
            e.set_primary_agent(a, enforce_deadline=True)  # set agent to track
            # Now simulate it
            sim = Simulator(e, update_delay=0.01)  # reduce update_delay to speed up simulation
            sim.run(n_trials=101)  # press Esc or close pygame window to quit
            
            # Metric check: the Q learning matrix has the maximum number of optimal acitons which are represend by the maximum value of each state(row). 
            qOptimal = checkQError(a.Q)
            if qOptimal > maxQOptimalActionNumber:
                maxQOptimalActionNumber = qOptimal
                minAlpha = al
                minGamma = ga
                minQ = a.Q
    
    print "minAlpha = {}, minGamma = {}, qOptimal = {}".format(minAlpha, minGamma, maxQOptimalActionNumber)    
    print minQ
    
def checkQError(q):
    qOptimalActionNumber = 0
    optimal = [0.,0.,0.,1.,0.,0.,1.,2.,0.,1.,2.,3.] #optimal actions
    
    for s in range(4):
        for n in range(3):                                                                              
            state = s*3 + n
            oa = np.argwhere( q[state,:] == np.amax( q[state,:] ) ).flatten()
            if len(oa) == 1 and oa[0] == optimal[state]:
                qOptimalActionNumber += 1
                
    return qOptimalActionNumber

if __name__ == '__main__':
    run()
