"""
An implementation of tabular Sarsa and Sarsa lambda algorithm
rlsn 2024
"""
import numpy as np
import itertools

def epsilon_greedy_policy(Q, epsilon, state, nA):
    if np.random.rand()<epsilon:
        A = np.random.randint(0,nA)
    else:
        A = np.argmax(Q[state])
    return A

def Tsoftmax_policy(Q, T, state, nA):
    ex = np.exp(Q[state]/T+1e-7)
    prob = ex/np.sum(ex)
    A = np.random.choice(nA, p=prob)
    return A


def tabular_sarsa(env, num_episodes, discount=1, epsilon=0.1, alpha=0.5, eval_interval=1000):
    # Q = np.zeros([env.observation_space.n,env.action_space.n])
    Q = np.random.randn(env.observation_space.n,env.action_space.n)*1e-2
    Q[-1] = 0

    acc_return = 0
    acc_length = 0
    for i_episode in range(1, num_episodes+1):
        
        if i_episode % eval_interval == 0:
            print("Episode {}/{}. avg return = {}, avg length = {}".format(i_episode, num_episodes, acc_return/eval_interval, acc_length/eval_interval))

            acc_return = 0
            acc_length = 0
                
        G = 0
        state, info = env.reset()
        action = epsilon_greedy_policy(Q, epsilon, state, env.action_space.n)

        for t in itertools.count():
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_action = epsilon_greedy_policy(Q, epsilon, next_state, env.action_space.n)

            Q[state, action] += alpha*(reward + discount*Q[next_state,next_action]-Q[state, action])
            state = next_state
            action = next_action

            G = discount*G + reward
            if terminated or truncated:
                acc_length+=t
                acc_return+=G
                break
                
    return Q    

def tabular_sarsa_lambda(env, num_episodes, discount=1, epsilon=0.1, alpha=0.5, lbda=0.8, eval_interval=1000):

    Q = np.random.randn(env.observation_space.n,env.action_space.n)*1e-2
    Q[-1] = 0

    acc_return = 0
    acc_length = 0
    for i_episode in range(1, num_episodes+1):
        
        if i_episode % eval_interval == 0:
            print("Episode {}/{}. avg return = {}, avg length = {}".format(i_episode, num_episodes, acc_return/eval_interval, acc_length/eval_interval))

            acc_return = 0
            acc_length = 0
                
        G = 0
        E = np.zeros([env.observation_space.n,env.action_space.n])
        state, info = env.reset()

        action = epsilon_greedy_policy(Q, epsilon, state, env.action_space.n)
        
        for t in itertools.count():
            next_state, reward, terminated, truncated, _ = env.step(action)

            next_action = epsilon_greedy_policy(Q, epsilon, next_state, env.action_space.n)
            
            delta = reward + discount*Q[next_state,next_action] - Q[state,action]
            E[state,action] += 1 # accumulating traces
            G = discount*G + reward 
            for s in range(Q.shape[0]):
                for a in range(Q.shape[1]):
                    Q[s,a] += alpha*delta*E[s,a]
                    E[s,a] = discount*lbda*E[s,a]
                    
            if terminated or truncated:
                acc_length+=t
                acc_return+=G
                break
                
            state = next_state
            action = next_action
            
    return Q

def test():
    import gymnasium as gym
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False, max_episode_steps=50)

    # train
    train = 1
    if train:
        num_episodes = 100000
        Q = tabular_sarsa(env, num_episodes, discount=1.0, epsilon=0.1, alpha=0.5, eval_interval=10000)
        # Q = tabular_sarsa_lambda(env, num_episodes, discount=1.0, epsilon=0.1, alpha=0.5, lbda=0.8, eval_interval=10000)
        np.save('Q.npy', Q)
            
    else:
        Q = np.load('Q.npy')

    print(Q)
    # eval
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False, render_mode="human")
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