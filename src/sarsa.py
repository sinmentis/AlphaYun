"""
An implementation of tabular Sarsa and Sarsa lambda algorithm
rlsn 2024
"""
import numpy as np
import itertools

class SarsaAgent(object):
    def __init__(self, Q, T=1, eps=0.1, mode='softmax', thresh=-1e7, name="p1"):
        self.Q = Q
        self.T = T
        self.eps = eps
        self.mode = mode
        self.name = name
        self.thresh = thresh
    def step(self, state, nA=None):
        if nA is None:
            nA = self.Q[state].shape[-1]
        if self.Q is None:
            A = np.random.randint(0,nA)
        else:
            if self.mode=='proportional':
                A = proportional_policy(self.Q, state, nA, thresh=self.thresh)
            elif self.mode=='softmax':
                A = softmax_sampling_policy(self.Q, self.T, state, nA, thresh=self.thresh)
            elif self.mode=='argmax':
                A = epsilon_greedy_policy(self.Q, 0, state, nA)
            elif self.mode=='eps_greedy':
                A = epsilon_greedy_policy(self.Q, self.eps, state, nA)
            elif self.mode=='threshed_uniform':
                A = threshed_uniform_policy(self.Q, state, nA, thresh=self.thresh)
            else:
                raise NotImplementedError(f"{self.mode} not implemented")
        return A

def epsilon_greedy_policy(Q, epsilon, state, nA):
    if np.random.rand()<epsilon:
        A = np.random.randint(0,nA)
    else:
        A = np.argmax(Q[state][:nA])
    return A

def proportional_policy(Q, state, nA, thresh=0.02, min=-1, max=1, output_p=False):
    q = Q[state][:nA]
    p = (q-min)/(max-min)
    p[p<thresh]=0
    p/=p.sum()
    if output_p:
        return p
    A = np.random.choice(nA, p=p)
    return A

def threshed_uniform_policy(Q, state, nA, thresh=0.02, output_p=False):
    q = Q[state][:nA]
    p=np.ones_like(q)
    p[q<thresh]=0
    p/=p.sum()
    A = np.random.choice(nA, p=p)
    if output_p:
        return p
    return A

def softmax(logp,T=1):
    ex = np.exp(logp/T+1e-7)
    prob = ex/np.sum(ex,-1)
    return prob

def softmax_sampling_policy(Q, T, state, nA, thresh=-0.98, output_p=False):
    q = np.copy(Q[state][:nA])
    q[q<thresh]=-np.inf
    p = softmax(q ,T)
    if output_p:
        return p
    A = np.random.choice(nA, p=p)
    return A

def tabular_sarsa(env, num_steps, Q=None, discount=1, epsilon=0.1, T=1, sampling_method='proportional',eta=1, alpha=0.5, eval_interval=1000, n_ternimal=1):
    if Q is None:
        # Q = np.zeros([env.observation_space.n,env.action_space.n])
        Q = np.random.randn(env.observation_space.n,env.action_space.n)*1e-2
        Q[-n_ternimal:] = 0 # terminal states to 0
    
    i_steps = 0
    acc_return = 0
    acc_length = 0
    for i_episode in itertools.count():
        if i_steps > num_steps:
            break

        if eval_interval>0 and i_episode % eval_interval == 0:
            print("Step {}/{}, Episode {}, avg return = {}, avg length = {}".format(i_steps, num_steps, i_episode, acc_return/eval_interval, acc_length/eval_interval))

            acc_return = 0
            acc_length = 0

        G = 0
        state, info = env.reset()
        
        if np.random.rand()<eta:
            agent = SarsaAgent(Q, T=T, eps=epsilon, mode="eps_greedy")
        else:
            agent = SarsaAgent(Q, T=T, eps=epsilon, mode=sampling_method)
        action = agent.step(state, env.action_space.n)
        # action=epsilon_greedy_policy(Q, epsilon, state, env.action_space.n)
        for t in itertools.count():
            i_steps += 1
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_action = agent.step(next_state, env.action_space.n)
            # next_action=epsilon_greedy_policy(Q, epsilon, next_state, env.action_space.n)

            Q[state, action] += alpha*(reward + discount*Q[next_state,next_action]-Q[state, action])
            state = next_state
            action = next_action

            G = discount*G + reward
            if terminated or truncated:
                acc_length+=t
                acc_return+=G
                break
                
    return Q    

def tabular_sarsa_lambda(env, num_steps, Q=None, discount=1, epsilon=0.1, alpha=0.5, lbda=0.8, eval_interval=1000):
    if Q is None:
        # Q = np.zeros([env.observation_space.n,env.action_space.n])
        Q = np.random.randn(env.observation_space.n,env.action_space.n)*1e-2
        Q[-1] = 0
    Q = np.random.randn(env.observation_space.n,env.action_space.n)*1e-2
    Q[-1] = 0

    i_steps = 0
    acc_return = 0
    acc_length = 0
    for i_episode in itertools.count():
        
        if i_steps > num_steps:
            break        


        if eval_interval>0 and i_episode % eval_interval == 0:
            print("Step {}/{}, Episode {}, avg return = {}, avg length = {}".format(i_steps, num_steps, i_episode, acc_return/eval_interval, acc_length/eval_interval))

            acc_return = 0
            acc_length = 0

        G = 0
        E = np.zeros([env.observation_space.n,env.action_space.n])
        state, info = env.reset()

        action = epsilon_greedy_policy(Q, epsilon, state, env.action_space.n)
        
        for t in itertools.count():
            i_steps += 1
            next_state, reward, terminated, truncated, _ = env.step(action)

            next_action = epsilon_greedy_policy(Q, epsilon, next_state, env.action_space.n)
            
            delta = reward + discount*Q[next_state,next_action] - Q[state,action]
            E[state,action] += 1 # accumulating traces
            G = discount*G + reward 
            for s in range(Q.shape[0]):
                for a in range(Q.shape[1]):
                    Q[s,a] += alpha*delta*E[s,a]
                    E[s,a] = discount*lbda*E[s,a]
                    
            state = next_state
            action = next_action

            if terminated or truncated:
                acc_length+=t
                acc_return+=G
                break
                            
    return Q

def test():
    import gymnasium as gym
    # env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False, max_episode_steps=50)
    env = gym.make('CliffWalking-v0', max_episode_steps=50)


    # train
    train = 1
    if train:
        num_steps = 2000000
        Q = tabular_sarsa(env, num_steps, discount=1.0, epsilon=0.1, alpha=0.5, eval_interval=10000,n_ternimal=1)
        # Q = tabular_sarsa_lambda(env, num_steps, discount=1.0, epsilon=0.1, alpha=0.5, lbda=0.8, eval_interval=10000)
        np.save('Q.npy', Q)
            
    else:
        Q = np.load('Q.npy')

    print(Q)
    # eval
    # env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False, render_mode="human")
    env = gym.make('CliffWalking-v0', max_episode_steps=50, render_mode="human")

    observation, info = env.reset(seed=42)
    env.render()
    while(True):
        action = np.argmax(Q[observation])
        # action = env.action_space.sample()

        observation, reward, terminated, truncated, info = env.step(action)
        
        env.render()
        
        if terminated or truncated:
            break

if __name__=="__main__":
    test()