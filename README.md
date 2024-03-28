# AlphaYun

Guess what's the best strategy for Bor-Bor Zan

“运”是一种起源于民间的口头游戏。方便简洁，只需要一双手，随时随地都能进行。非常锻炼玩家的反应力、记忆力和创造力，是休闲娱乐、放松心情的好方法。

# Quick Start
Play against the best strategy and see how you like it:
```
python play.py
```

# The Best Strategy
See the report below if you are interested in what the best strategy is and how it is found.
## Formulation
The game of [Bor-Bor-Zan](http://www.wikibin.org/articles/bor-bor-zan-2.html) is a Markov game when the number of energy units accumulated and attack level is bounded, and hence is defined by a Markov decision process(MDP) with finite state and action sets $(S,A)$ and a one-step dynamics $P(s',r|s,a)$, the joint probability of next state and reward conditioned on state and action, denoted as
$$ P(s',r|s,a) = Pr\{S_{t+1}=s',R_{t+1}=r|S_t=s,A_t=a\} $$
A behavioral strategy $\pi\in\Pi:S\times A\rightarrow\mathbb{R}$ is a probability distribution over all available actions at a given state $$\pi(a|s)=Pr\{A_t=a|S_t=s\}, a\in A, s\in S$$ 
It should be noted that the dynamics of one player is stochastic and dependent on the behavioral strategy of his opponent, and therefore is precisely $P_{\phi}(s',r|s,a)$, where $\phi$ is the opponent's behavioral strategy. However, The dynamics is deterministic given the states and actions of both player, denoted $(s,\Phi)$, with $s = (s_1,s_2)\in S^2$ and $\Phi=(a_1,a_2)\in A^2$. For a rule with maximum energy level $S_{max}$ and attack/defense level $A_{max}$, we have:
\begin{aligned}
S&=\{0,1,...,S_{max}\}\\
A&=\{0,1,...,2A_{max}+1\}
\end{aligned}
Here the values of $s_1$ and $s_2$ represent the numbers of energy units the agent and its opponent have respectively. We further separate the action space into 3 subspaces $\mathrm{pow}(a) = \{0,1\}$ = powerup (1) or not (0), 
$\mathrm{atk}(a) = \{0,...,A_{max}\}$ = attack level, and $\mathrm{def}(a) = \{0,...,A_{max}\}$ = defense level. The values of these functions are subject to a condition that two of them must be zero. Assume a zero-sum reward $\rho=(r_1,r_2)\in\{(1,-1),(0,0),(-1,1)\}$, the dynamic is then defined by
$$ 
\begin{aligned}
s_1' &= \mathrm{clamp}(s_1+\mathrm{pow}(a_1)-\mathrm{atk}(a_1),0,S_{max})\\
s_2' &= \mathrm{clamp}(s_2+\mathrm{pow}(a_2)-\mathrm{atk}(a_2),0,S_{max})\\
\rho(s,\Phi) &= 
\begin{cases}
(1,-1) & \text{if $\mathrm{atk}(a_1)>0$ and $\mathrm{atk}(a_1)>\mathrm{atk}(a_2)$ and $\mathrm{atk}(a_1)\neq \mathrm{def}(a_2)$,}\\
(-1,1) & \text{if $\mathrm{atk}(a_2)>0$ and $\mathrm{atk}(a_2)>\mathrm{atk}(a_1)$ and $\mathrm{atk}(a_2)\neq \mathrm{def}(a_1)$,} \\
(0,0) & \text{otherwise} \\
\end{cases}
\end{aligned}
$$
The above rules define an **imperfect information symmetric zero-sum two-player extensive-form** game. The expected return of a player with behavioral strategy $\pi$ against another with strategy $\phi$ is denoted as 
$$R(\pi,\phi) = \mathbb{E}_{(\pi,\phi)}\Bigl[{P_{(\pi,\phi)}(r|s,a)\pi(a|s)P_{(\pi,\phi)}(s)}\Bigr]$$ where $P_{(\pi,\phi)}(s)$ is the realizaion probability of visiting state $s$ over all possible game trajectories with strategy profile$(\pi,\phi)$, and the $P_{(\pi,\phi)}(r|s,a)$ is the distribution over reward given to $\pi$ conditioned on its state-action pair $(s,a)$. The **best response** strategy against an arbitrary strategy $\phi$ is defined as $b(\phi) = \arg\mathop{\max}\limits_\pi R(\pi,\phi)$.
Our goal is to find a **Nash equilibrium** (NE), defined as a strategy $\pi$ such that $\pi = b(\pi)$. For a symmetric zero-sum game the Nash equilibria are strategies that beat or tie all other strategies.

## Method
We approach this problem by multi-agent reinforcement learning (MARL) with self-play (SP)[[4](#4),[5](#5)], where an RL agent learns by playing against copies of itself. It optimizes its strategy profile and saves a copy after a certain steps, which then becomes an opponent of the next iterations. At each step, the agent observes the both players energy level, i.e. $s=(s_1,s_2)$ to make decision. It is well established that SP-based method is guaranteed to converge to an approximate global Nash equilibrium (NE) with probability 1 for two-player zero-sum games [[1]](#1). However, as both players play simultaneuous move, bor-bor-zan is by definition an imperfect information game similar to rock-paper-scissors. Therefore, optimizing against some agents might results in losing against others. Formally, bor-bor-zan is **cyclic**[[6]](#6), defined as: 
$$\int_\phi R(\pi,\phi) \,d\phi=0,\,\, \forall \pi\in\Pi$$
This property can be verified by obtaining a strategy profile $\mathcal{B}=\{\pi_1,\pi_2,...,\pi_n\}$ by running a selfplay for several generations and compute its evaluation matrix $R_\mathcal{B}=\{R(\pi,\phi):(\pi,\phi)\in\mathcal{B}\times\mathcal{B}\}$. The resulting evaluation matrix and the top-2 principal components of its Schur decomposition are shown below, which obviously manifest the cyclic property.

<img src = "imgs/evaluation.png" width ="30%" /><img src = "imgs/schur.png" width ="30%" />

We use a variation of Policy-Space Response Oracles (PSRO) algorithm as described in [[3](#3),[6](#6)]. The algorithm starts with a random initialized strategy profile $\mathcal{B}$ with size $N$. For each iteration, an empirical Nash equilibrium of the profile is computed, which is a mixed strategy weighted by the points in the positive quadrant that intersects the polytope formed by the convex combination of rows of $R_\mathcal{B}$.
Specifically, the weight $p\in\mathbb{R}^N$is calculated using a linear programming solver with the following objective:
\begin{aligned}
\mathop{\min}\limits_p & \,\,p, \\
\text{s.t.    } & R_\mathcal{B}^\intercal p\geq 0,\\
&\sum\limits_{i=1}^{N} p_i = 1,\\
& p_i\geq 0, \forall i
\end{aligned}
Then for each agent $\pi_i$ in the profile, a training step is taken to maximize the expected return against the weighted mixed strategy 
$$\frac{1}{Z}\sum\limits_{\pi_j\in\mathcal{B}\setminus\pi_i, R(\pi_i,\pi_j)\geq 0} p[j]\cdot\pi_j$$ where $Z$ is the normalizing factor that ensures $\sum_a\pi(a|s)=1$. The mixture constitutes only agents that it beats or ties, in order to grow the strategy space encapsulated by the profile.[[6]](#6)

As the state and action space of this game is relatively small, we use tabular Q-learning[[2]](#2), a value-based off-policy method to optimize the agents. Q-learning searches for an optimal state-action value function $Q^*(S,A)$ by utilizing a temporal-difference update rule:
$$
Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha \Bigl[R_{t+1}+\gamma \mathop{\max}_a Q(S_{t+1},a) - Q(S,A) \Bigr]
$$
where $\alpha$ is a hyperparameter controlling the learning rate, and $\gamma$ is the discount factor which in our experiments is set to $\gamma=0.9$. To facilitate the sufficient exploration of the MDP, we initialize each episode with a random initial state $(s_1,s_2)\in S^2$. The behavioral strategy for the opponent during training is $\epsilon\text{-greedy}(Q)$. After Q-learning converges, the agent's best response pure strategy is obtained by $\pi_\beta(s) = \arg\mathop{\max}\limits_a Q^*(s,a)$. It is then updated towards this best response by 
$$\pi_{t+1}\leftarrow \pi_t+\eta(\pi_\beta-\pi_t) $$ with $\eta$ being the stepsize of the update.

## Results
The experiments are conducted for 10 runs, with $S_{max}=5$, $A_{max}=3$, and $N=20$. Each run is run for 20 iterations and is evaluated using the exploitability of the NE, defined as $exploitability(\pi^*) = \frac{1}{N}\sum\limits_{\pi_i\in\mathcal{B}}\max\{R(\pi_i,\pi^*),0\}$. The exploitability resulted from the experiments is $0.051 \pm 0.021$, meaning an $\epsilon$-equilibrium is achieved with $\epsilon\approx 0.05$. The strategy at this approximate equilibium is summurized as below.

<img src = "imgs/Q6x6.png" width ="70%" />

In the figure, each subplot contains $\pi^*(A|S)$ at one specific value of $s=(s_1,s_2)$, e.g. at $s=(2,0)$, the agent observes it has 2 units of energy whereas its opponent has none. On x-axis the actions are abbreviated as "C" for charging energy, "A1" for level 1 attack, "D2" for level 2 defense, and so on. It can be seen that at $s=(1,1)$, attacking is counterintuitively not the most profitable move despite that it directly leads to victory should the opponent is charging. Shown below is a zoomed version of the figure that only covers $s\in [0,3]^2$ for a closer look.

<img src = "imgs/Q3x3.png" width ="45%" />

Using Monte-Carlo method, we can also approximate the state visit frequency $P_{\pi^*}(S)$ and winning state frequency (the last state the winner observes before episode terminates) $P_{\pi^*}(S_{T-1}|R_t=1)$ at NE. The results are shown below.

<img src = "imgs/ps.png" width ="30%" /><img src = "imgs/wins.png" width ="30%" />

The plot shows that at equilibrium, the game trajectory concentrates within a range of $[0,2]^2$, and most of the time the winner ends the game at $s=(1,1)$ or $s=(2,1)$. Using the same method we also approximated the state value function V(S), as shown below.

<img src = "imgs/vs.png" width ="30%" />

The plot is symmetric and 0 on the diagnal as expected. However, it's worth noting that when both player have 3 or more than 3 units of energy, i.e. when $s\in[A_{max},S_{max}]^2$, the value of $V(s)$ plunges to nearly zero as more energy no longer grants the player any advantage.

Remember that the best response strategy is invincible in the sense that, regardless of how your opponent plays the game, they can't beat you more times than they lose on average. However, in fact, it is not necessarily "the" best response if your opponent doesn't follow the best response as you do. You could easily screw someone over by always picking the dominant move against their highest probability move at every state. But then, they would have to adjust their strategy to counter yours, while you try to counter their counter at the same time. Repeating this process finally results in returning to the Nash equilibrium with both of you following the same strategy. At any rate, I wish you have fun and good luck with your future endeavors beating your 3rd grader nephew on Bor-Bor-Zan;)

## References
<a id="1">[1]</a> 
Hofbauer, J. and Sandholm, W.H. (2002), On the Global Convergence of Stochastic Fictitious Play. Econometrica, 70: 2265-2294.

<a id="2">[2]</a> 
Sutton, R.S. & Barto, A.G., (2018). Reinforcement learning: An introduction, MIT press.

<a id="3">[3]</a> 
Johannes Heinrich and David Silver. Deep reinforcement learning from self-play in imperfect-information
games. CoRR, abs/1603.01121, 2016

<a id="4">[4]</a> 
Johannes Heinrich, Marc Lanctot, and David Silver. 2015. Fictitious self-play in extensive-form games. In Proceedings of the 32nd International Conference on International Conference on Machine Learning - Volume 37 (ICML'15). JMLR.org, 805–813.

<a id="5">[5]</a> 
Lanctot, Marc, et al. "A unified game-theoretic approach to multiagent reinforcement learning." Advances in neural information processing systems 30 (2017).

<a id="6">[6]</a> 
Balduzzi, David, et al. "Open-ended learning in symmetric zero-sum games." International Conference on Machine Learning. PMLR, 2019.