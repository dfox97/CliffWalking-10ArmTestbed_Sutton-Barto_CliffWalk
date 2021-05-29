'''
    Question 5 - Cliff walking problem comparing Q-learning and Sarsa 

    Program created by: Daniel Fox, Beatrice Carroll, Sophie Hook, Devon Motte
    
    Created with the help of online resources which including:
    Understanding Q-Learning, the Cliff Walking problem,Lucas Vazquez https://medium.com/@lgvaz/understanding-q-learning-the-cliff-walking-problem-80198921abbc 
    QLearning-and-Sarsa-for-Cliff-Walking by TissueC,https://github.com/TissueC/QLearning-and-Sarsa-for-Cliff-Walking/tree/master/code 
    cliffwalking example created by jassiay on github https://github.com/jassiay/CliffWalking 
    Also a video tutorial was very helpful Q-Learning - path planning problem - cliff walking: #https://www.youtube.com/watch?v=aXBU1PaTFmA

'''
import numpy as np
import matplotlib.pyplot as plt

#***************************************************************
#                   Parameters
#***************************************************************
#Main parameters set as global variables
#could have passed them through functions but easier to test here 
epsilon=0.1
runs=10#int
episodes=500
alpha=0.1#floats
gamma=1.0
num_actions=4
#***************************************************************
#                   Grid/env setup
#***************************************************************

#setting up the grid environment , grid created and actions tracked
class GridWorld(object):
    def __init__(self):
        self.cols = 12 
        self.rows = 4  
        # up,down,left,right
        self.actions = [[0, 1], [0, -1], [-1, 0], [1, 0]] 
        self.restart()
    
    def restart(self):
        self.x = 0#x pos
        self.y = 0#y poss
        # defining/finding the goal
        self.end_x = 11 
        self.end_y = 0
        self.finish = False

    def status(self):
        #check the status of agent, determine if it has reached the goal or fell of the cliff
        return tuple((self.x, self.y)), self.finish

    def checkInstance(self, x, y):
        x = max(x, 0)
        y = max(y, 0)
        x = min(x, self.cols - 1)#stops x moving out of grid
        y = min(y, self.rows - 1)#stops y moving out of grid
        return x, y

    def movement(self, action):
        self.x += self.actions[action][0]#tracks x direction
        self.y += self.actions[action][1]#tracks y direction
        self.x, self.y = self.checkInstance(self.x, self.y)#check where the agent is
        self.finish = False #check if goal reached
        if self.x >= 1 and self.x <= 10 and self.y == 0:  # fall off cliff
            reward = -100#reward = -100
            self.restart()#reset instance
        elif self.x == self.cols - 1 and self.y == 0:  # the goal
            reward = 0
            self.finish = True#set finished to true
        else:  # safe place and keep moving
            reward = -1 # normal movement
        return ((self.x, self.y)), reward, self.finish

#-----------------------------------------------------
#  defaultdict function as a fix to iterating through dictionary
#---------------------------------------------------------
#added in to replace importing defaultdictionary , this is to stop the error occuring dictionary is not iterable,
#also it resolves the problem with missing keys by automatically creating one.
def defaultdict(default_type):
    class DefaultDict(dict):
        def __getitem__(self, key):
            if key not in self:
                dict.__setitem__(self, key, default_type())
            return dict.__getitem__(self, key)
    return DefaultDict()

    
#-----------------------------------------------------
#  Egreedy 
#---------------------------------------------------------
#finding egreedy 
def egreedyPolicy(Q, state):
    action = np.argmax(Q[state])
    #sometimes need np.float32 to covert from 64 not sure why
    A = np.ones(num_actions) * epsilon / num_actions
    A[action] += 1 - epsilon
    return A
#-----------------------------------------------------
#  Qlearning algorithm
#---------------------------------------------------------
def Qlearning(env):
    Q=defaultdict(lambda: np.zeros(num_actions))
    rewards = []
    for i in range(episodes):#loop through episodes
        env.restart()
        currentState, finish = env.status()
        sumReward = 0.0

        while 1:
            probability = egreedyPolicy(Q, currentState)
            action = np.random.choice(np.arange(num_actions), p=probability)  #new action per round
            newState, reward, finish = env.movement(action)  # apply action to find get next state by finding max action
            newAction = np.argmax(Q[newState])
            #qlearning calculation from book
            Q[currentState][action] = Q[currentState][action] + alpha * (reward + gamma * Q[newState][newAction] - Q[currentState][action])
            currentState = newState
            if finish:
                break
            sumReward += reward
        rewards.append(sumReward)
    return Q, rewards
#-----------------------------------------------------
#  SARSA Algorithm
#---------------------------------------------------------
def sarsa(env):
    Q=defaultdict(lambda: np.zeros(num_actions))
    rewards = []
    for episode in range(episodes): 
        env.restart()
        currentState, finish = env.status()
        probability = egreedyPolicy(Q, currentState)
        action = np.random.choice(np.arange(num_actions), p=probability)  # action prob
        sumReward = 0.0
        while 1:
            newState, reward, finish = env.movement(action)  # exploration
            probability = egreedyPolicy(Q, newState)  # get action probability distribution for next state
            newAction = np.random.choice(np.arange(num_actions),
                                            p=probability)  # get next action, use [next_state][next_action]  to update Q[state][action]
            #calculation from book for sarsa algorithm
            Q[currentState][action] = Q[currentState][action] + alpha * (
                    reward + gamma * Q[newState][newAction] - Q[currentState][action])
            if finish:
                break    
            currentState = newState
            action = newAction
            sumReward += reward
        rewards.append(sumReward)
    return Q, rewards


def plot(len_epi,avg,labels):
    length = len(len_epi)
    len_epi = [len_epi[i] for i in range(length) if i % runs == 0]
    avg = [avg[i] for i in range(length) if i % runs == 0]
    plt.plot(len_epi, avg,label=labels)
    
#-----------------------------------------------------
#  RUNNING 
#---------------------------------------------------------
def main():
    qlearn_env=GridWorld()#create new object instance for q learning
    Q, rewards = Qlearning(env=qlearn_env) # pass in obect into the q learning algorithm and get the two return values , state and rewards
    sarsa_env = GridWorld()#create new object instance for sarsa learning
    Q2, rewards = sarsa(env=sarsa_env)# pass in obect into the sarsa algorithm and get the two return values , state and rewards

    qlearn_avg = []#array for averages
    sarsa_avg=[]
    for i in range(runs):

        Q, Qlearn_rward = Qlearning(env=qlearn_env)#get new states and rewards
        Q2, Sarsa_rward = sarsa(env=sarsa_env)#get new states and rewards
    
        qlearn_avg=np.array(Qlearn_rward) if len(qlearn_avg) == 0 else qlearn_avg + np.array(Qlearn_rward) #find average for qlearning
        sarsa_avg= np.array(Sarsa_rward) if len(sarsa_avg) == 0 else sarsa_avg + np.array(Sarsa_rward)#finding average for sarsa


    qlearn_avg /= runs#final average calcs
    sarsa_avg/= runs#final average calcs
#-----------------------------------------------------
#  Plotting
#---------------------------------------------------------
    plot(range(episodes), qlearn_avg,labels='Q-learning='+str(epsilon))
    plot(range(episodes), sarsa_avg,labels='Sarsa='+str(epsilon))
    plt.title("Cliff walking")
    plt.ylabel('Sum of rewards during episode')
    plt.xlabel('Episode')
    plt.ylim(-500,0)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()

    