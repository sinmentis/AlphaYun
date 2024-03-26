"""
Script for model training
rlsn 2024
"""
from env import YunEnv,RPSEnv
from agent import Agent, tabular_Q, softmax
import numpy as np
import argparse, random, time, itertools
from tqdm import tqdm


def exploitability(beta,pi,Ne=300):
    b1 = Agent(beta[0])
    b2 = Agent(beta[1])
    pi1 = Agent(pi[0])
    pi2 = Agent(pi[1])
    R1 = R2 = 0
    for i in range(Ne):
        state, info = env.reset(opponent=pi2, train=True)
        for t in itertools.count():
            action = b1.step(state)
            state, r, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                R1+=r
                break
        state, info = env.reset(opponent=pi1, train=True)
        for t in itertools.count():
            action = b2.step(state)
            state, r, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                R2+=r
                break
    return R1/Ne, R2/Ne

def fictitious_selfplay(env, num_iters=1000, num_steps_per_iter = 20000, eps=0.1, alpha=0.1):
    # Q = np.random.randn(env.observation_space.n,env.action_space.n)*1e-2
    # Q[-env.n_ternimal:] = 0 # terminal states to 0
    pi = np.ones([2,env.observation_space.n,env.action_space.n])
    pi = np.random.rand(2,env.observation_space.n,env.action_space.n)

    pi = pi/pi.sum(-1,keepdims=True)

    # pi[0,0,0]=0.25
    # pi[0,0,1]=0.25
    # pi[0,0,2]=0.5

    beta = np.copy(pi)
    expl = 1
    pbar = tqdm(range(1,num_iters+1), desc="Iter", position=0)
    for niter in pbar:
        # rl beta strategy
        
        # reset Q
        Q = np.random.randn(2, env.observation_space.n,env.action_space.n)*1e-2
        Q[:,-env.n_ternimal:] = 0 # terminal states to 0
        for i,j in [(0,1),(1,0)]:
            avg_policy = np.copy(pi.mean(0))
            env.reset(opponent=Agent(pi[j]), train=True)
            Q[i] = tabular_Q(env, num_steps_per_iter, Q=Q[i], epsilon=eps, alpha=alpha, eval_interval=-1)
            beta[i] = np.eye(env.action_space.n)[Q[i].argmax(-1)]

            # eta = 1/niter
            # pi[i] += eta*(beta[i]-pi[i])

        Ne=300
        r1,r2 = exploitability(beta, pi, Ne=Ne)
        expl=r1+r2
        eta = max(0.2/np.sqrt(niter),0.0001)
        pi += eta*(beta-pi)

        pbar.set_description(f"eta={round(eta,4)}, expl={round(expl,2)} {pi.mean(0)[0,0]}|Iter")
        pbar.refresh()
    return Q, pi

def selfplay(env, num_iters=1000, num_steps_per_iter = 20000, eps=0.1, alpha=0.1):
    # Q = np.random.randn(env.observation_space.n,env.action_space.n)*1e-2
    # Q[-env.n_ternimal:] = 0 # terminal states to 0
    pi = np.ones([env.observation_space.n,env.action_space.n])
    pi = pi/pi.sum(-1,keepdims=True)

    # pi[0,0]=0.25
    # pi[0,1]=0.25
    # pi[0,2]=0.5

    beta = np.copy(pi)
    expl = 1
    pbar = tqdm(range(1,num_iters+1), desc="Iter", position=0)
    for niter in pbar:
        # rl beta strategy
        
        # reset Q
        Q = np.random.randn(env.observation_space.n,env.action_space.n)*1e-2
        Q[-env.n_ternimal:] = 0 # terminal states to 0

        env.reset(opponent=Agent(pi), train=True)
        Q = tabular_Q(env, num_steps_per_iter, Q=Q, epsilon=eps, alpha=alpha, eval_interval=-1)
        beta = np.eye(env.action_space.n)[Q.argmax(-1)]
        # beta = softmax(Q, T=1.02)
        # print(pi)
        # print(Q)
        # exit()

        Ne = 1000
        R=0
        beta_agent = Agent(beta)
        for i in range(Ne):
            state, info = env.reset(opponent=Agent(pi), train=True)
            for t in itertools.count():
                action = beta_agent.step(state)
                state, r, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    R+=r
                    break

        expl = R/Ne
        
        # eta = max(0.5/np.sqrt(niter),0.01)
        # a = np.maximum(expl-1/np.sqrt(Ne),0)
        # pi += eta*(beta-pi)*a

        eta = 1/niter
        pi += eta*(beta-pi)


        pbar.set_description(f"eta={round(eta,4)}, expl={round(expl,2)} {pi[0],Q[0].argmax()} |Iter")
        pbar.refresh()
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
    env = YunEnv()
    # env = RPSEnv()

    print("args:",args)

    print("Training...")
    start = time.time()
    # Q,pi = selfplay(env, num_iters=1000, num_steps_per_iter = 20000, eps=0.1, alpha=0.01)
    Q,pi = fictitious_selfplay(env, num_iters=1000, num_steps_per_iter = 20000, eps=0.1, alpha=0.1)

    np.save(args.model_file, {"Q":Q,"PI":pi})

    print("Training complete, model saved at {}, elapsed {}s".format(args.model_file,round(time.time()-start,2)))