"""
Script for model training
rlsn 2024
"""
from env import YunEnv,RPSEnv
from agent import Agent, tabular_Q
import numpy as np
import argparse, random, time, itertools
from tqdm import tqdm

def generate_data(env, MSL, pi, beta, n, m, eta):
    sigma = (1-eta)*pi+eta*beta
    P_sigma = [Agent(sigma[i]) for i in range(pi.shape[0])]
    D = np.zeros([1, env.observation_space.n,env.action_space.n])
    for i in range(n):
        p1 = np.random.choice(P_sigma)
        p2 = np.random.choice(P_sigma)
        state, info = env.reset(opponent=p2, train=True)
        for t in itertools.count():
            action = p1.step(state)
            D[0, state, action]+=1
            state, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
    P_beta = [Agent(beta[i]) for i in range(pi.shape[0])]
    for i in range(len(P_beta)):
        p1 = P_beta[i]
        ps = list(range(len(P_sigma)))
        ps.remove(i)
        for j in range(m):
            p2 = P_sigma[np.random.choice(ps)]
            state, info = env.reset(opponent=p2, train=True)
            for t in itertools.count():
                action = p1.step(state)
                MSL[i, state, action]+=1
                state, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break
    return MSL+D, P_sigma

def exploitability(beta,pi,Ne=200):
    b1 = Agent(beta[0])
    b2 = Agent(beta[1])
    pi1 = Agent(pi[0])
    pi2 = Agent(pi[1])
    R = 0
    for i in range(Ne):
        state, info = env.reset(opponent=pi2, train=True)
        for t in itertools.count():
            action = b1.step(state)
            state, r, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                R+=r
                break
        state, info = env.reset(opponent=pi1, train=True)
        for t in itertools.count():
            action = b2.step(state)
            state, r, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                R+=r
                break
    return R/Ne/2

def selfplay(env, num_iters=10, num_steps_per_iter = 20000, eps=0.1, alpha=0.01, num_players=2):
    Q = np.random.randn(num_players, env.observation_space.n,env.action_space.n)*1e-2
    Q[:,-env.n_ternimal:] = 0 # terminal states to 0
    beta = np.zeros([num_players, env.observation_space.n,env.action_space.n])
    pi = np.zeros([num_players, env.observation_space.n,env.action_space.n])
    MSL = np.ones([num_players, env.observation_space.n,env.action_space.n])
    num_swaps = 20
    expl = 1
    pbar = tqdm(range(1,num_iters+1), desc="Iter", position=0)
    for niter in pbar:
        eta = 1/niter
        MSL, P_sigma = generate_data(env, MSL, pi, beta, n=100, m=100, eta=eta)
        for n_pl in tqdm(range(num_players), desc="player", position=1, leave=False):
            for n_sw in range(num_swaps):
                ps = list(range(len(P_sigma)))
                ps.remove(n_pl)
                opponent = P_sigma[np.random.choice(ps)]
                env.reset(opponent=opponent, train=True)
                Q[n_pl] = tabular_Q(env, num_steps_per_iter//num_swaps, Q=Q[n_pl], epsilon=eps, alpha=alpha, eval_interval=-1)
                beta[n_pl]*=0
            for s in range(beta.shape[1]):
                beta[n_pl,s,Q[n_pl,s].argmax()]=1

        pi = MSL/np.sum(MSL,axis=-1,keepdims=True)
        expl = exploitability(beta,pi)
        pbar.set_description(f"expl={round(expl,2)}|Iter")
        pbar.refresh() # to show immediately the update
    return Q, pi

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, help="set seed", default=None)
    parser.add_argument('--model_file', type=str, help="filename of the model to be saved", default="Qh.npy")
    parser.add_argument('--num_steps', type=int, help="number of total training steps", default=1e7)
    parser.add_argument('--save_steps', type=int, help="number of training steps for each saved model", default=2e4)

    parser.add_argument('--step_size', type=int, help="learning rate alpha", default=0.1)
    parser.add_argument('--memory_size', type=float, help="hyperparameter controls action memory size", default=3000)
    parser.add_argument('--eps', type=float, help="hyperparameter epsilon for epsilon greedy policy", default=0.1)
    parser.add_argument('--eta', type=float, help="hyperparameter anticipatory parameter", default=0.3)

    args = parser.parse_args()
    
    if not args.seed:
        args.seed = int(time.time())
    np.random.seed(args.seed)
    print("running with seed", args.seed)
    # env = YunEnv()
    env = RPSEnv()

    print("args:",args)

    print("Training...")
    start = time.time()
    Q,pi = selfplay(env, num_iters=100, num_steps_per_iter = 20000, eps=0.1, alpha=0.01, num_players=2)
    np.save(args.model_file, {"Q":Q,"PI":pi})
    print(pi[0])
    print(Q[0])
    print("Training complete, model saved at {}, elapsed {}s".format(args.model_file,round(time.time()-start,2)))