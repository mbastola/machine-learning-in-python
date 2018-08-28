import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class BanditArm:
    def __init__(self, p, d, p_init = 0., n=0. ):
        # p: the win rate
        self.p = p
        self.dist = d
        self.p_estimate = p_init #initial p estimate
        self.N = n # num samples collected so far

    def pull(self):
        # draw a 1 with probability p
        sample = 0.
        if self.dist == "norm":
            sample = np.random.randn()
        elif self.dist == "beta":
            sample = np.random.beta()
        else:
            sample = np.random.random()
        return sample < self.p

    def update(self, x):
        self.N += 1.
        self.p_estimate = ((self.N - 1)*self.p_estimate + x) / self.N
        

class EpsilonGreedy:
    def __init__(self, bandit_probabilities):
        self.bandits =  [BanditArm(p,d) for p,d in bandit_probabilities]
        self.rewards = None

    def run(self, num_trials, eps ):

        self.rewards = np.zeros(num_trials)
        self.num_trials = num_trials
        
        num_times_explored = 0
        num_times_exploited = 0
        num_optimal = 0
        optimal_j = np.argmax([b.p for b in self.bandits])
        print("optimal j:", optimal_j)

        for i in range(num_trials):
              
            # use epsilon-greedy to select the next bandit
            if np.random.random() < eps:
                num_times_explored += 1
                j = np.random.randint(len(self.bandits))
            else:
                num_times_exploited += 1
                j = np.argmax([b.p_estimate for b in self.bandits])

            if j == optimal_j:
                num_optimal += 1

            # pull the arm for the bandit with the largest sample
            x = self.bandits[j].pull()
            
            # update rewards log
            self.rewards[i] = x

            # update the distribution for the bandit whose arm we just pulled
            self.bandits[j].update(x)

        # print mean estimates for each bandit
        for b in self.bandits:
            print("mean estimate:", b.p_estimate)

        # print total reward
        print("total reward earned:", self.rewards.sum())
        print("overall win rate:", self.rewards.sum() / num_trials)
        print("num_times_explored:", num_times_explored)
        print("num_times_exploited:", num_times_exploited)
        print("num times selected each bandit:", [b.N for b in self.bandits])
        print("\n")

    def plot( self,  fig = None, save_path = None ):
        if not fig:
            fig, _ = plt.subplots()

        # plot the results
        cumulative_rewards = np.cumsum(self.rewards)
        win_rates = cumulative_rewards / (np.arange(self.num_trials) + 1)
        plt.plot(win_rates)
        if save_path:
            plt.savefig(save_path)


class OptimisticInitialValue:
    def __init__(self, bandit_probabilities):
        self.bandits =  [BanditArm(p,d, 5.0, 1) for p,d in bandit_probabilities]
        self.rewards = None
        
    def run(self, num_trials ):

        self.rewards = np.zeros(num_trials)
        self.num_trials = num_trials
        
        for i in range(num_trials):
            # use optimistic initial values to select the next bandit
            j = np.argmax([b.p_estimate for b in self.bandits])
            
            # pull the arm for the bandit with the largest sample
            x = self.bandits[j].pull()

            # update rewards log
            self.rewards[i] = x
            
            # update the distribution for the bandit whose arm we just pulled
            self.bandits[j].update(x)

        # print mean estimates for each bandit
        for b in self.bandits:
            print("mean estimate:", b.p_estimate)

        # print total reward
        print("total reward earned:", self.rewards.sum())
        print("overall win rate:", self.rewards.sum() / num_trials)
        print("num times selected each bandit:", [b.N for b in self.bandits])
        print("\n")
        
    def plot( self,  fig = None, save_path = None ):
        if not fig:
            fig, _ = plt.subplots()

        # plot the results
        cumulative_rewards = np.cumsum(self.rewards)
        win_rates = cumulative_rewards / (np.arange(self.num_trials) + 1)
        plt.plot(win_rates)
        if save_path:
            plt.savefig(save_path)


class UCB1:
    def __init__(self, bandit_probabilities):
        self.bandits =  [BanditArm(p,d) for p,d in bandit_probabilities]
        self.rewards = None

    def ucb(self, mean, n, nj):
        return mean + np.sqrt(2*np.log(n) / nj)
        
    def run(self, num_trials ):

        self.rewards = np.zeros(num_trials)
        self.num_trials = num_trials
        total_plays = 0

        # initialization: play each bandit once to avoid initital infinities
        for j in range(len(self.bandits)):
            x = self.bandits[j].pull()
            total_plays += 1
            self.bandits[j].update(x)
        
        for i in range(num_trials):
            j = np.argmax([self.ucb(b.p_estimate, total_plays, b.N) for b in self.bandits])
            x = self.bandits[j].pull()
            total_plays += 1
            self.bandits[j].update(x)

            # for the plot
            self.rewards[i] = x
            
        # print mean estimates for each bandit
        for b in self.bandits:
            print("mean estimate:", b.p_estimate)

        # print total reward
        print("total reward earned:", self.rewards.sum())
        print("overall win rate:", self.rewards.sum() / num_trials)
        print("num times selected each bandit:", [b.N for b in self.bandits])
        print("\n")

    def plot( self,  fig = None, save_path = None ):
        if not fig:
            fig, _ = plt.subplots()

        # plot the results
        cumulative_rewards = np.cumsum(self.rewards)
        win_rates = cumulative_rewards / (np.arange(self.num_trials) + 1)
        plt.plot(win_rates)
        if save_path:
            plt.savefig(save_path)


class BayesianSampling:
    #Using bernoulli/beta distribution prior/posterior
    
    def __init__(self, bandit_probabilities):
        self.bandits =  [BanditArm(p,d) for p,d in bandit_probabilities]
        self.posterior = [ [1,1] for i in self.bandits ] #alpha/beta values
        self.rewards = None

    def updatePosterior(self, idx, x):
        self.posterior[idx][0] += x
        self.posterior[idx][1] += 1 - x

    def sample(self, idx):
        return np.random.beta(self.posterior[idx][0], self.posterior[idx][1])     
    def run(self, num_trials ):

        self.rewards = np.zeros(num_trials)
        self.num_trials = num_trials

        sample_points = [ 2^i for i in range(2, int(np.log(num_trials)),1 )]
        self.rewards = np.zeros(num_trials)
        
        for i in range(num_trials):
            # Thompson sampling
            j = np.argmax([self.sample(idx) for idx, b in enumerate(self.bandits)])

            # plot the posteriors
            #if i in sample_points:
            #    self.plot(self.bandits, i)

            # pull the arm for the bandit with the largest sample
            x = self.bandits[j].pull()

            # update rewards
            self.rewards[i] = x
                
            # update the distribution for the bandit whose arm we just pulled
            self.bandits[j].update(x)
            self.updatePosterior(j, x)
        
        # print mean estimates for each bandit
        for i, b in enumerate(self.bandits):
            print("mean estimate:", self.posterior[i][0]*1.0/num_trials)

        # print total reward
        print("total reward earned:", self.rewards.sum())
        print("overall win rate:", self.rewards.sum() / num_trials)
        print("num times selected each bandit:", [b.N for b in self.bandits])
        print("\n")

    def plot( self,  fig = None, save_path = None ):
        if not fig:
            fig, _ = plt.subplots()

        # plot the results
        cumulative_rewards = np.cumsum(self.rewards)
        win_rates = cumulative_rewards / (np.arange(self.num_trials) + 1)
        plt.plot(win_rates)
        if save_path:
            plt.savefig(save_path)

            
def run_experiments():

    np.random.seed(100)

    num_trials = 10000
    
    bandit_ps = [0.2, 0.5, 0.75]
    bandit_dists = [None for i in bandit_ps]
    bandits = np.array([ (i,j) for i,j in zip(bandit_ps, bandit_dists)])

    labels = []
    rewards = []
    probs = []
    
    fig, _ = plt.subplots()
    plt.plot(np.ones(num_trials)*np.max(bandit_ps))
    labels.append("Optimal")
    
    expt = EpsilonGreedy(bandits)
    expt.run(num_trials, eps = 0.1)
    expt.plot(fig, None)
    labels.append("{}".format(expt.__class__.__name__))
    rewards.append(expt.rewards.sum())
    probs.append([ b.p_estimate for b in expt.bandits])
    
    expt = OptimisticInitialValue(bandits)
    expt.run(num_trials)
    expt.plot(fig, None)
    labels.append("{}".format(expt.__class__.__name__))
    rewards.append(expt.rewards.sum())
    probs.append([ b.p_estimate for b in expt.bandits])

    expt = UCB1(bandits)
    expt.run(num_trials)
    expt.plot(fig, None)
    labels.append("{}".format(expt.__class__.__name__))
    rewards.append(expt.rewards.sum())
    probs.append([ b.p_estimate for b in expt.bandits])

    expt = BayesianSampling(bandits)
    expt.run(num_trials)
    expt.plot(fig, None)
    labels.append("{}".format(expt.__class__.__name__))
    rewards.append(expt.rewards.sum())
    probs.append([ b.p_estimate for b in expt.bandits])
    
    plt.legend(labels=labels, loc='lower right')
    #plt.show()


    plt.clf()
    plt.bar(labels[1:], rewards)
    plt.plot([ num_trials*np.max(bandit_ps) for i in range(4)])
    #plt.show()
    

    plt.clf()
    probs2 = {}
    for i, item in enumerate(probs):
        for j,k in enumerate(item):
            if not probs2.get( "Bandit {}".format(j) ):
                probs2[ "Bandit {}".format(j) ] = []
            probs2[ "Bandit {}".format(j) ].append(k)
            #probs2.append( (labels[i+1], "Bandit {}".format(j), k) )
    probs2 = pd.DataFrame(probs2, index = labels[1:])
    #gp = probs2.groupby("bandits")
    #print(gp)
    ax = probs2.plot.bar(rot=0 )
    ax.legend(loc=3)  
    plt.show()
    print(probs2)
    
if __name__ == "__main__":
    run_experiments()
