import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import numpy as np
import multiprocessing
import time

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
        self.count = 0 # total steps
        self.count0 = 300 # the number of random choices from the beginning of the training
        self.timeStep = 0 # time cost
        self.trialNum = 0 # number of trial routes
        self.Q_action_freq = np.zeros((12,4))
        self.penalty_num = 0
        
        
    def reset(self, destination=None):
        # record the previous settings
        # print "alpha = {}, gamma = {}, qOptimal = {}".format(self.alpha, self.gamma, self.checkQError()) # [debug]
                   
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required 
        if self.count >= self.count0:
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
        s = self.getStateIndex()
		       		
        # TODO: Select action according to your policy 
        if self.count < self.count0:    # fully randomly choosing actions at the beginneing of the training
            q0EntryNum = np.argwhere( self.Q[s,:]==0 ).flatten()
            if len(q0EntryNum) > 0:
                a = random.choice( q0EntryNum ) # choose with priority from the actions that have not been implemented before
            else:
                a = random.choice( [0, 1, 2, 3] ) 
        else:
            if random.uniform(0, 1) < self.epsilon:  
                q0EntryNum = np.argwhere( self.Q[s,:]==0 ).flatten()
                if len(q0EntryNum) > 0:
                    a = random.choice( q0EntryNum )  # choose with priority from the actions that have not been implemented before
                else:
                    a = random.choice( [0, 1, 2, 3] )  # With the probability epsilon, the car will choose randomly to explore         
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
        # if self.count >= self.count0:
        self.Q_action_freq[s,a] += 1       
        if reward < 0:
            self.penalty_num += 1
        
        # TODO: Learn policy based on state, action, reward
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        self.state = (self.next_waypoint, inputs['light'], inputs['left'], inputs['oncoming']) 
        sNext = self.getStateIndex()
		
        self.Q[s,a] = (1-self.alpha) * self.Q[s,a] + self.alpha * ( reward + self.gamma * np.amax(self.Q[sNext,:]) )
                
        # update epsilon, which gradually changes the agent from exploration to exploitation.       
        self.count += 1
        self.timeStep += 1

        # print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]

    def getStateIndex(self):
        state = self.state
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
        
    def checkQError(self, q):
        qOptimalActionNumber = 0
        optimal = [0.,0.,0.,1.,0.,0.,1.,2.,0.,1.,2.,3.] #optimal actions
        
        for s in range(4):
            for n in range(3):                                                                              
                state = s*3 + n
                oa = np.argwhere( q[state,:] == np.amax( q[state,:] ) ).flatten()
                if len(oa) == 1 and oa[0] == optimal[state]:
                    qOptimalActionNumber += 1
                    
        return qOptimalActionNumber    

class Task(object):
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.result = ()
    def __call__(self):
        self.result = self.run( self.a, self.b )
        return self.result
    def __str__(self):
        return str(self.result)        
        
    def run(self,al,ga):
        """Run the agent for a finite number of trials."""
        np.set_printoptions(suppress=True)
               
        minAlpha = 0
        minGamma = 0
        minQ = []
                        
        # Set up environment and agent
        e = Environment()  # create environment (also adds some dummy traffic)
        a = e.create_agent(LearningAgent, al, ga)  # create agent
        e.set_primary_agent(a, enforce_deadline=True)  # set agent to track
        # Now simulate it
        sim = Simulator(e, update_delay=0.001)  # reduce update_delay to speed up simulation
        sim.run(n_trials=101)  # press Esc or close pygame window to quit   
        # Metric check: the Q learning matrix has the maximum number of optimal acitons which are represend by the maximum value of each state(row). 
        qOptimal_1 = a.checkQError(a.Q)
        qOptimal_2 = a.checkQError(a.Q_action_freq)
        # print "Alpha = {}, Gamma = {}, optimal choices = {}".format(al, ga, qOptimal) 
        
        return (qOptimal_1, a.Q, a.Q_action_freq, al, ga, a.penalty_num, a.count ,qOptimal_2)


class Consumer(multiprocessing.Process):
    
    def __init__(self, task_queue, result_queue):
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
        self.result_queue = result_queue

    def run(self):
        proc_name = self.name
        while True:
            next_task = self.task_queue.get() 
            if next_task is None:
                # Poison pill means shutdown
                # print '%s: Exiting' % proc_name
                self.task_queue.task_done()
                break            
            self.task_queue.task_done() # run the task and fetch the result                           
            answer = next_task()
            print '%s: %s' % (proc_name, answer)
            self.result_queue.put(answer)
        return    

if __name__ == '__main__':
    # Enqueue jobs
    alpha = np.arange(0.1,1.01,0.1)
    gamma = np.arange(0.1,1.01,0.1)
    targetOptimalActionNumber = 10
    maxOptimalNumber = 0
    maxOptimalNumber2 = 0
    num_consumers = multiprocessing.cpu_count() * 2
    while maxOptimalNumber <= targetOptimalActionNumber or maxOptimalNumber != maxOptimalNumber2:  
        
        # Establish communication queues
        tasks = multiprocessing.JoinableQueue()
        results = multiprocessing.Queue()
        
        # Start consumers       
        print 'Creating %d consumers' % num_consumers
        consumers = [ Consumer(tasks, results)
                      for i in xrange(num_consumers) ]
        for w in consumers:
            w.start()
        
        for al in alpha:
            for ga in gamma:    
                tasks.put(Task(al,ga))
                
        # Add a poison pill for each consumer
        for i in xrange(num_consumers):
            tasks.put(None)
            
        # Wait for all of the tasks to finish
        tasks.join()
               
        maxQValueNumber = []
        minQ = np.zeros((1,12,4))
        minQ_action_freq = np.zeros((1,12,4))
        minAlpha = []
        minGamma = []
        minPenalty_num = []
        stepCount = []
        count = 0
        maxQActionNumber = []
        for al in alpha:
            for ga in gamma: 
                count += 1
                result = results.get()
                # qOptimal, a.Q, a.Q_action_freq, al, ga, a.penalty_num, a.count
                maxQValueNumber.append(result[0])
                re1 = np.reshape(result[1],(1,12,4))
                re2 = np.reshape(result[2],(1,12,4))
                if count == 1:
                    minQ = re1
                    minQ_action_freq = re2
                else:                    
                    minQ = np.concatenate((minQ, re1), axis=0)
                    minQ_action_freq = np.concatenate((minQ_action_freq, re2), axis=0)
                minAlpha.append(result[3])
                minGamma.append(result[4])
                minPenalty_num.append(result[5])
                stepCount.append(result[6])
                maxQActionNumber.append(result[7])
        maxOptimalNumber = np.amax(maxQValueNumber)
        maxOptimalNumber2 = np.amax(maxQActionNumber)        
        
        #######
        for w in consumers:
            w.terminate()
        del consumers
        del tasks
        del results
        ###########
    
    m = np.argwhere( maxQValueNumber == maxOptimalNumber ).flatten()[0]    
    print '***************************************************'
    print 'optimal values are: '
    print maxQValueNumber[m], minQ[m], minQ_action_freq[m], minAlpha[m], minGamma[m], minPenalty_num[m], stepCount[m], 


