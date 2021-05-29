'''
    10 Armed Testbed from the Sutton & Barto book comparing a greedy method with two e-greedy methods(𝜀=0.01 and 𝜀=0.1)

    Program created by: Daniel Fox, Beatrice Carroll, Sophie Hook, Devon Motte
    
    Created with the help of online resources which include:
    Solving the Multi-Armed Bandit Problem by Anson Wong (article in Sep 25, 2017),https://towardsdatascience.com/solving-the-multi-armed-bandit-problem-b72de40db97c
    10 armed bandit example created by Jet-Tsyn Lee on github https://github.com/jettdlee/10_armed_bandit/blob/master/10_bandit_testbed.py
    Also a video tutorial was very helpful in helping implimenting the correct reward structure by The Data Incubator: #https://www.youtube.com/watch?v=aAdD2XRC044&t=889s

'''
import numpy as np
import matplotlib.pyplot as plt

#*********************************************************************
#***********************Testbed setup from book***********************
#*********************************************************************
class Testbed():
    #2.3 in the RL book , set 10 armed testbed chapter 2.3
    # Initializer
    def __init__(self, nArms):
        # Number of arms
        self.nArms = nArms
        self.Qt = np.zeros(self.nArms) # True optimal action rewards

        self.optimalAction = 0 #True optimal action value
        self.reset()
        # Reset testbed for next iteration
    #After each iteration the program needs to call the reset function to clear values for each iteration
    def reset(self):
        # Set random gaussian number of arms between 0 and 1 
        self.Qt = np.random.normal(0, 1, self.nArms)
        # Find max value for optimal action
        self.optimalAction = np.argmax(self.Qt)#will be used to find optimal action

#*********************************************************************
#***********************Bandit Class for e-greedy and greedy**********
#*********************************************************************
class Bandit():
    # Initializer
    def __init__(self,nArms, epsilon):
        self.nArms = nArms     
        self.epsilon = epsilon#value of epsilon probability [0,0.01,0.1]

        self.Na = np.zeros(nArms)   # count number of actions taken for each arm
        self.Qa = np.zeros(nArms)   # estimated action values


    # String for graph plots and testing class values
    def __str__(self):
        if self.epsilon == 0:
            return "Epsilon = %s (Greedy)"%(str(self.epsilon))
        return "Epsilon = " + str(self.epsilon)


    #Get action function determins what the action will do, if the epsilon equals zero the greedy method is performed.
    def getActions(self):
        #e-epsilon method
        if np.random.random()  < self.epsilon:
            a = np.random.randint(self.nArms)#explore
        #greedy method
        else:
            #exploting, choosing the arm with the largest estimated value
            action = np.where(self.Qa == np.argmax(self.Qa))[0]
            maxAction = np.argmax(self.Qa)
            #if the length of the action is zero then select the max action as the reutrn action value. If the actions contrain the same value then return a random choice 
            if len(action) == 0:
                a = maxAction
            else:
                a = np.random.choice(action) 
        return a

    #updates values based of last action
    def getUpdate(self, reward,action):
        self.Na[action] += 1       # Updates counter relation to arm pull
        #From the book equation to update Qa+1=Qa+a*[reward-Qa] working 
        #a=alpha
        alpha=1/self.Na[action]
        self.Qa[action]+=alpha*(reward-self.Qa[action])#Qa+1=Qa+a*[reward-Qa]
       

    # Reset the estimates for count and actions for each epsilon
    def reset(self):
        self.Na = np.zeros(self.nArms)
        self.Qa = np.zeros(self.nArms) 

      
#*********************************************************************
#***********************Creating the Environment***********************
#*********************************************************************
class Env():
    def __init__(self, testbed, bandits, runs, iterations):
        #passing in object classes 
        self.testbed = testbed #Testbed() class passed in
        self.bandits = bandits#Bandit() class passed in
        self.runs = runs #Num of runs
        self.iterations = iterations#num of iterations 
    
    def getReward(self,action):
        #random guassian reward based on the testbad Qt value at index [action] 
        reward=np.random.normal(self.testbed.Qt[action])
        return reward


    def run(self):
    #loop through iterations and bandits to work out final values
        optimlArr = np.zeros((self.runs, len(self.bandits)))#optimal array
        avgReward = np.zeros((self.runs,len(self.bandits)))#average rewards
        for i in range(self.iterations):
            #calling testbet reset function for each iteration 
            self.testbed.reset()
            for epsilon in self.bandits:
                #calling the testbed reset function for each epsilon value
                epsilon.reset()

            #helps track program looping through iterations 
            if (i%200) == 0:
                print("Iteration %d"%(i))
            #main part of program which will compute the required results
            #loop through and play through the length of runs number of times 
            for play in range(self.runs):
                count = 0#counter for each epsilon
            #go through the object class and iterate through each epsilon
                for eps in self.bandits:
                    tempAction = eps.getActions()#get action value and store in temp
                    temp_Reward = self.getReward(tempAction)#work out reward for the given action and epsilon
                    eps.getUpdate(temp_Reward,tempAction)#finally update the epsilon
                    #work out average reward by taking the length value of self.runs at play and the number of times the program has counted the epsilon 
                    avgReward[play,count] += temp_Reward #sum of rewards
                    #if the tempaction is the same as the true optimal action then increase the score by 1 to give the final optimal action
                    if tempAction == self.testbed.optimalAction:
                        optimlArr[play,count] += 1
                    count += 1

        #return final values 
        avgReward/=self.iterations #working out averages
        optimlArr/=self.iterations
        return avgReward, optimlArr

#*********************************************************************
#***********************Ploting both graphs***************************
#*********************************************************************
def getPlot(avg,opt,epsilon):

    plot1 = plt.figure(1)
    plt.title("10-Armed bandit problem - Average Rewards")
    plt.ylabel('Average Reward')
    plt.xlabel('Steps')
    plt.plot(avg)
    plt.ylim(0, 1.5)#y limits 0 - 1.5 as in the book
    plt.legend(epsilon, loc=8)

    plot2 = plt.figure(2)
    plt.title("10-Armed bandit problem - Optimal Action")
    plt.ylabel('% Optimal Action')
    plt.xlabel('Steps')
    plt.plot(opt)
    plt.ylim(0, 100)#0-100 y axis limit
    plt.legend(epsilon,loc=8)
    plt.show()



def main():
    #Parameters 
    eps=[0,0.01,0.1] #values for e-greedy and greedy
    nArms = 10 #num of arms for armed bandit which is 10    
    iterations = 2000    #how many times to loop through (2000 in book) 
    runs = 1000 #play/run the 1000 times for each bandit.
             
#*********************************************************************
#*************Create Class Object Instances**************************
#*********************************************************************

    #Create instance for testbed and set out basic rules following the book (chapter 2.3)
    testbed = Testbed(nArms)
    #Create another instance for bandits, these are the greedy and e-greedy epsilon values = [0,0.01,0.1]
    bandits = [Bandit(nArms,eps[0]),Bandit(nArms,eps[1]),Bandit(nArms,eps[2])]

    #Environment will run through iterations and compute the results for average rewards and optimal action. Bandits relate to each epsilon value for the egreedy method and the testbed has the requirments and rules off the testbed hold the true optimal values to compare.
    environment = Env(testbed,bandits,runs,iterations)
    #create average reward and optimal action variables to store once the environment has completed its run.  
    avg_reward, opt_action = environment.run()


    #Turning optimal action into percentage 
    opt_action*=100
    #call function to plot graphs 
    getPlot(avg_reward,opt_action,bandits)



if __name__ == "__main__":
    main()

    