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
        self.totalTimeStep = 0
        self.penaltyNum = 0 # number of violation of traffic laws during the 10 testing routes from 101 to 110
        self.maxDeadline = 0
        self.trialNum = 0 # number of trials
        self.alwaysSafelyReachDestination = True # whether does the car reach the destination successfully and safely during the 10 testing routes from 101 to 110

    def reset(self, destination=None):
        # record the previous settings
        # print "deadline step = {}, time step = {}, penalty number = {}".format(self.env.get_deadline(self), self.timeStep, self.penaltyNum) # [debug]
        if self.trialNum > 100 and self.trialNum <= 150:            
            if self.timeStep <= self.maxDeadline  and self.penaltyNum == 0:
                self.totalTimeStep += self.timeStep               
            else:
                self.alwaysSafelyReachDestination = False
                   
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required 
        self.trialNum += 1         
        self.timeStep = 0
        self.penaltyNum = 0
        self.maxDeadline = self.env.get_deadline(self)

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        self.state = (self.next_waypoint, inputs['light'], inputs['left'], inputs['oncoming'])        
        s = self.getStateIndex(self.state)
		       		
        # TODO: Select action according to your policy 
        if self.count < 300:       
            a = random.choice( [0, 1, 2, 3] )
        else:
            if self.trialNum <= 100 and random.uniform(0, 1) < self.epsilon: 
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
        # Record the number of violation of traffic laws
        if reward < 0:
            self.penaltyNum += 1

        # TODO: Learn policy based on state, action, reward
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        sNext = self.getStateIndex(self.state)
		
        self.Q[s,a] = (1-self.alpha) * self.Q[s,a] + self.alpha * ( reward + self.gamma * np.amax(self.Q[sNext,:]) )
        
        # update epsilon, which gradually changes the agent from exploration to exploitation.
               
        self.count += 1
        self.timeStep += 1
        
        self.epsilon = 1. / self.trialNum
        # self.alpha = self.alpha0 / self.trialNum

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

    alpha = np.arange(0.1,1.01,0.1)
    gamma = np.arange(0.1,1.01,0.1)
    minTimeStep = 10000
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
            sim.run(n_trials=111)  # press Esc or close pygame window to quit
            
            # Metric check: the smart car does not violate the traffic law and always reaches the destination in the 10 testing routines. 
            if a.alwaysSafelyReachDestination:
                if minTimeStep > a.totalTimeStep:
                    minTimeStep = a.totalTimeStep
                    minAlpha = al
                    minGamma = ga
                    minQ = a.Q
    
    print "minTimeStep = {}, minAlpha = {}, minGamma = {}".format(minTimeStep, minAlpha, minGamma)
    print round(a.Q,2)

if __name__ == '__main__':
    run()
