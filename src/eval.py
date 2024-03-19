"""
Script for model evaluation and analysis
rlsn 2024
"""
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from yunenv import YunEnv
from sarsa import SarsaAgent, softmax
import argparse, time


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', type=str, help="filename of the model", default="Qh.npy")
    parser.add_argument('--seed', type=int, help="set seed", default=None)
    parser.add_argument('--run', action='store_true', help="run example match with lastest moddel")
    parser.add_argument('-r', action='store_true', help="start match with random initial states")
    parser.add_argument('-T', type=float, help="set softmax temperature coeeficient T", default=0.3)
    parser.add_argument('--stats', action='store_true', help="run state analysis")
    parser.add_argument('--tour', action='store_true', help="run tournament")
    parser.add_argument('--tsize', type=int, help="number of models in tournament")

    args = parser.parse_args()

    if args.seed:
        seed = args.seed
    else:
        seed = int(time.time())
    np.random.seed(seed)
    print("running with seed", seed)

    env = YunEnv()

    Qh = np.load(args.model_file)[::-1]
    print("model loaded from {}, size {}".format(args.model_file, Qh.shape))

    if args.run:
        T = args.T
        Q1, Q2 = Qh[0], Qh[1]
        agent = SarsaAgent(Q1, T=T, mode='softmax', name='p1')
        opponent = SarsaAgent(Q2, T=T, mode='softmax', name='p2')
        
        observation, info = env.reset(seed=None, opponent=opponent, train=args.r)
        print("Example match:")
        print(0, info)
        for i in range(1,100):
            action = agent.step(observation, env.action_space.n)
            observation, reward, terminated, truncated, info = env.step(action)
            print(i, info)
            if terminated or truncated:
                break

    if args.stats:
        # some analysis at particular states
        T = args.T
        grid_size = 4
        
        # instance stats
        fig, axs = plt.subplots(grid_size,grid_size,figsize=(10,10))
        fig.suptitle("Model state-action value at varying states & softmax @ T={}".format(T))
        Q = Qh[0]
        for S1 in range(grid_size):
            for S2 in range(grid_size):
                S = S1 * (env.rule.n_max_energy+1) + S2
                ax2 = axs[S1,S2].twinx()
                ax2.bar(np.arange(Qh.shape[-1]),softmax(Q[S],T=T),color='tab:red',alpha=0.3)
                ax2.tick_params(axis='y', labelcolor='tab:red')
                ax2.set_ylabel("softmax", color='tab:red')
                axs[S1,S2].plot(np.arange(Qh.shape[-1]),Q[S],color='tab:blue')
                axs[S1,S2].set_title("S=({},{})".format(S1,S2))
                axs[S1,S2].set_xticks(np.arange(Qh.shape[-1])+1,np.arange(Qh.shape[-1]))
                axs[S1,S2].set_xlabel("A")
                axs[S1,S2].tick_params(axis='y', labelcolor='tab:blue')
                axs[S1,S2].set_ylabel("Q(S,A)", color='tab:blue')
                axs[S1,S2].grid()

        fig.tight_layout()

        # state-action value grid
        
        fig, axs = plt.subplots(grid_size,grid_size,figsize=(10,10))
        fig.suptitle("Bank state-action value at varying states")
        for S1 in range(grid_size):
            for S2 in range(grid_size):
                S = S1 * (env.rule.n_max_energy+1) + S2
                axs[S1,S2].boxplot(Qh[:,S])
                axs[S1,S2].set_title("S=({},{})".format(S1,S2))
                axs[S1,S2].set_xticks(np.arange(Qh.shape[-1])+1,np.arange(Qh.shape[-1]))
                axs[S1,S2].set_xlabel("A")
                axs[S1,S2].set_ylabel("Q(S,A)")
                axs[S1,S2].grid()
        fig.tight_layout()
        plt.show()

    if args.tour:
        T = args.T
        num_matches_per_pair = 200
        max_steps = 30
        random_start = args.r

        Qi = Qh[::-1]

        NP = Qi.shape[0]
        R = np.zeros([NP,NP])
        ns = env.rule.n_max_energy+1
        state_freq = np.zeros(env.observation_space.n)
        win_last_state_freq = np.zeros(env.observation_space.n)

        print("running tournament, total matches: {}".format(num_matches_per_pair*(1+NP)*NP/2))

        logs = []
        for i in tqdm(range(NP), position=0):
            Lj = []
            for j in tqdm(range(NP), position=1, leave=False):
                if j<i:
                    R[i,j]=-R[j,i]
                    continue
                Lk = []
                for k in range(num_matches_per_pair):
                    P1 = SarsaAgent(Qi[i], T=T, mode='softmax')
                    P2 = SarsaAgent(Qi[j], T=T, mode='softmax')

                    observation, info = env.reset(opponent=P2, train=random_start)
                    Lt = [info]
                    for t in range(max_steps):
                        action = P1.step(observation, env.action_space.n)
                        observation, reward, terminated, truncated, info = env.step(action)
                        Lt.append(info)
                        state_freq[observation]+=1
                        if terminated:
                            R[i,j]+=reward
                            if reward==1:
                                win_last_state_freq[Lt[-2]["observation"]]+=1
                            break
                        if truncated:
                            break
                    # Lk.append(Lt)
                # Lj.append(Lk)
            # logs.append(Lj)
        R/=num_matches_per_pair
        tot=R.sum(1,keepdims=True)/NP
        R = np.concatenate([R,tot],1)
        fig, ax = plt.subplots(figsize=(8,8))
        im = ax.imshow(-R, cmap = "RdBu_r")
        ax.set_title("Tounament result")
        ax.set_xticks(np.arange(NP+1),list(range(NP))+["Total"])
        ax.set_xlabel("P2 (Generation)")
        ax.set_yticks(np.arange(NP),np.arange(NP))
        ax.set_ylabel("P1 (Generation)")
        for i in range(NP):
            for j in range(NP+1):
                text = ax.text(j, i, round(R[i, j],2),
                            ha="center", va="center", color="w")
        fig.tight_layout()

        # state visit
        state_freq = state_freq[:ns**2]/state_freq.sum()
        state_freq = state_freq.reshape(ns,ns)

        fig, ax = plt.subplots(figsize=(6,6))
        im = ax.imshow(state_freq, cmap = "Reds")
        ax.set_title("State visit frequency")
        ax.set_xticks(np.arange(ns),np.arange(ns))
        ax.set_xlabel("S2")
        ax.set_yticks(np.arange(ns),np.arange(ns))
        ax.set_ylabel("S1")
        for i in range(ns):
            for j in range(ns):
                text = ax.text(j, i, round(state_freq[i, j],2),
                            ha="center", va="center", color="w")
        fig.tight_layout()

        # winner last state
        win_last_state_freq = win_last_state_freq[:ns**2]/win_last_state_freq.sum()
        win_last_state_freq = win_last_state_freq.reshape(ns,ns)

        fig, ax = plt.subplots(figsize=(6,6))
        im = ax.imshow(win_last_state_freq, cmap = "Reds")
        ax.set_title("Winner last state frequency")
        ax.set_xticks(np.arange(ns),np.arange(ns))
        ax.set_xlabel("S2")
        ax.set_yticks(np.arange(ns),np.arange(ns))
        ax.set_ylabel("S1")
        for i in range(ns):
            for j in range(ns):
                text = ax.text(j, i, round(win_last_state_freq[i, j],2),
                            ha="center", va="center", color="w")
        fig.tight_layout()

        # stats
        print("total match finished: {}".format(env.rule.num_matches))
        print("violation loss: {}".format(env.rule.violation))
        print("stupidity loss: {}".format(env.rule.stupidity))
        print("busted loss: {}".format(env.rule.busted))
        print("outpowered loss: {}".format(env.rule.outpowered))
        print("mismatch loss: {}".format(env.rule.mismatch))
        plt.show()