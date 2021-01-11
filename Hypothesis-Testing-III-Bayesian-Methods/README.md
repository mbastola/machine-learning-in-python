
[This project](https://github.com/mbastola/machine-learning-in-python/tree/master/Hypothesis-Testing-III-Bayesian-Methods) is third in the series of Hypothesis testing project whereby we use Bayesian Methods. Here we reformulate the AB testing problem into MultiArm Bandit and implement 4 widely used algorithms: Epsilon Greedy, Optimistic Initial Conditions, UCB1, and Bayesian (Thompson) Sampling for the AB testing. We then compare and contrast the performance of the algorithms wrt the Bandit problem.

The Multi Arm Bandit (MAB) is a problem widely used in introductory Reinforcement Learning. In MAB, a fixed limited set of resources must be allocated between competing choices in a way that maximizes their expected gain, when each choice's properties are unknown initially, and becomes better understood as we gather more samples by allocating resources to the choice. In essence, we would like to know which resource maximizes the final expected return without us investing too much into resources that may eventually be suboptimal. 

While Frequentist statistics would have us invest in all possible resources to collect enough data and compute the expected return for classical Hypothesis testing, we use Bayesian method to converge to the optimal resource while also minimizing any sub-optimal investments. The method is online and converges quickly to optimal set of resources without us having to collect all the data and run AB testing on it. One can quickly see the benefits of Bayesian Bandit methods over Frequentist AB testing where the cost of investing into sub-optimal resources to collect data is high, for eg. beating slot machines, investigating stock returns,etc.


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from classes import *
```


```python
np.random.seed(100)

num_trials = 10000

bandit_ps = [0.2, 0.5, 0.75] #bandit reward probabilities
bandit_dists = [None for i in bandit_ps] #distributions
bandits = np.array([ (i,j) for i,j in zip(bandit_ps, bandit_dists)])
```

Our experiment is setup as such, we have 3 bandit arms with reward of 1 unit with probabilites of reward in 0.2, 0.5 and 0.75, which are unknown to us. The distribution of the reward is uniform (None) but can be others if specified. Our task is to figure out which bandit arm provides the best value in 10K trials while also maximizing our final reward.   

### Epsilon Greedy

The classic dilemma in Reinforcement learning is Explore/Exploit. In out Bandit problem we are faced with such in finding the expected return while maximizing our returns. Epsilon greedy is the simplest way to tackle the dilemma which uses a parameter 0 < epsilon < 1 to specify exploration/exploitation.


```python
fig, _ = plt.subplots()
plt.plot(np.ones(num_trials)*np.max(bandit_ps))
labels = []
labels.append("Optimal")

expt = EpsilonGreedy(bandits)
expt.run(num_trials, eps = 0.1)
expt.plot(fig, None)
labels.append("{}".format(expt.__class__.__name__))
plt.legend(labels=labels, loc='lower right')
```

    optimal j: 2
    mean estimate: 0.21727019498607242
    mean estimate: 0.4744318181818182
    mean estimate: 0.7559478953601032
    total reward earned: 7267.0
    overall win rate: 0.7267
    num_times_explored: 1070
    num_times_exploited: 8930
    num times selected each bandit: [359.0, 352.0, 9289.0]
    

![png](https://github.com/mbastola/machine-learning-in-python/blob/master/Hypothesis-Testing-III-Bayesian-Methods/imgs/output_5_2.png)


We note that the epsilon greedy method is able to estimate the expected reward for all the 3 bandits while also maximizing the overall win rate. The optimal win rate for us is to choose arm = 3 with reward probability 0.75. Our return is 0.727.

### Optimistic Initial Values

While Epsilon greedy converges close to the optimal mean, it does so investing too much time on non optimal rewards. One way to counteract this is to use Optimistic Initial Values algorithm where we initial the initial probability estimate to a higher value. We note below that this method converges way faster into the optimal reward increasing total rewards earned. However, since this method ignores suboptimal returns, the final mean for the non-optimal bandits is highly overestimated.


```python
fig, _ = plt.subplots()
plt.plot(np.ones(num_trials)*np.max(bandit_ps))
labels = []
labels.append("Optimal")

expt = OptimisticInitialValue(bandits)
expt.run(num_trials)
expt.plot(fig, None)
labels.append("{}".format(expt.__class__.__name__))
plt.legend(labels=labels, loc='lower right')
```

    mean estimate: 0.7272727272727273
    mean estimate: 0.7083333333333334
    mean estimate: 0.7461878009630832
    total reward earned: 7448.0
    overall win rate: 0.7448
    num times selected each bandit: [11.0, 24.0, 9968.0]


![png](https://github.com/mbastola/machine-learning-in-python/blob/master/Hypothesis-Testing-III-Bayesian-Methods/imgs/output_8_2.png)


### UCB1

UCB1 (Auer et al.(2002)Auer, Cesa-Bianchi, and Fischer) utilizes a heuristic for upper confidence bound. One can derive the heuristic from Chernov-Heoffding inequality. The goal is to select the bandit arm that maximizes ucb return : x_i + sqrt( 2 Log (N)/n_i ) , at each time step i for total experiments N. We note that UCB1 heuristic converges to the actual reward extimate faster than the Epsilon greedy while also providing close enough estimate for non-optimal reward. The total rewards earned is comparable to Optimistic Initial Values method.


```python
fig, _ = plt.subplots()
plt.plot(np.ones(num_trials)*np.max(bandit_ps))
labels = []
labels.append("Optimal")

expt = UCB1(bandits)
expt.run(num_trials)
expt.plot(fig, None)
labels.append("{}".format(expt.__class__.__name__))
plt.legend(labels=labels, loc='lower right')
```

    mean estimate: 0.2333333333333333
    mean estimate: 0.478494623655914
    mean estimate: 0.7501281131495338
    total reward earned: 7421.0
    overall win rate: 0.7421
    num times selected each bandit: [60.0, 186.0, 9757.0]
    

![png](https://github.com/mbastola/machine-learning-in-python/blob/master/Hypothesis-Testing-III-Bayesian-Methods/imgs/output_10_2.png)


### Bayesian Sampling

Bayesian (Thomson) Sampling is a bayesian belief propagation method that assumes prior and posterior distributions for the reward update. For successful convergence, the prior and posterior distributions are selected as [conjugate pairs](https://en.wikipedia.org/wiki/Conjugate_prior). In our example, we use Bernoulli prior and Beta posterior update. We note that the Bayesian sampling method outperforms all the above algorithms in convergence and maximizes the final reward. However, like the Optimistic Initial Values, Bayesian Sampling mostly ignores the suboptimal rewards and hence, we are left with unknown estimates for the suboptimal options.


```python
fig, _ = plt.subplots()
plt.plot(np.ones(num_trials)*np.max(bandit_ps))
labels = []
labels.append("Optimal")

expt = BayesianSampling(bandits)
expt.run(num_trials)
expt.plot(fig, None)
labels.append("{}".format(expt.__class__.__name__))
plt.legend(labels=labels, loc='lower right')
```

    mean estimate: 0.0001
    mean estimate: 0.0004
    mean estimate: 0.7463
    total reward earned: 7465.0
    overall win rate: 0.7465
    num times selected each bandit: [6.0, 11.0, 9983.0]
    

![png](https://github.com/mbastola/machine-learning-in-python/blob/master/Hypothesis-Testing-III-Bayesian-Methods/imgs/output_12_2.png)


Below is the comparision of the convergence of the 4 methods above:

Convergence:
![png](https://github.com/mbastola/machine-learning-in-python/blob/master/Hypothesis-Testing-III-Bayesian-Methods/imgs/compare.png)

Estimated Probabilities:
![png](https://github.com/mbastola/machine-learning-in-python/blob/master/Hypothesis-Testing-III-Bayesian-Methods/imgs/estimated_probs_dist.png)

Total rewards:
![png](https://github.com/mbastola/machine-learning-in-python/blob/master/Hypothesis-Testing-III-Bayesian-Methods/imgs/total_rewards.png)
