"""
Script for model evaluation and analysis
rlsn 2024
"""
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from yunenv import YunEnv
from sarsa import SarsaAgent, softmax_sampling_policy, proportional_policy, threshed_uniform_policy
import argparse, time, itertools

def thresh(method):
    match method:
        case "proportional":
            return 0.02
        case "softmax":
            return -0.98
        case "threshed_uniform":
            return -0.98
        case _:
            return -1e7


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', type=str, help="filename of the model", default="Qh.npy")
    parser.add_argument('--seed', type=int, help="set seed", default=None)
    parser.add_argument('--run', action='store_true', help="run example match with lastest moddel")
    parser.add_argument('-r', action='store_true', help="start match with random initial states")
    parser.add_argument('-T', type=float, help="set softmax temperature coeeficient T", default=1.)
    parser.add_argument('--stats', action='store_true', help="run state analysis")
    parser.add_argument('-s', type=int, help="grid size of the stats to display", default=3)
    parser.add_argument('--tour', action='store_true', help="run tournament")
    parser.add_argument('--tsize', type=int, help="number of models in tournament")
    parser.add_argument('--sampling_method', type=str, help="sampling method for agent", default='proportional')
    parser.add_argument('--selfplay', action='store_true', help="run self play")
    parser.add_argument('--selfplay_sampling_method', type=str, help="sampling method for opponent in self play", default='threshed_uniform')

    args = parser.parse_args()
    if args.seed:
        seed = args.seed
    else:
        seed = int(time.time())
    np.random.seed(seed)
    print("running with seed", seed)

    env = YunEnv()

    Qh = np.load(args.model_file)
    print("model loaded from {}, size {}".format(args.model_file, Qh.shape))

    if args.run:
        T = args.T
        P1 = SarsaAgent(Qh[0], T=T, name='p1', mode=args.sampling_method, thresh=thresh(args.sampling_method))
        P2 = SarsaAgent(Qh[0], T=T, name='p2', mode=args.selfplay_sampling_method, thresh=thresh(args.selfplay_sampling_method))
        
        observation, info = env.reset(seed=None, opponent=P2, train=args.r)
        print("Example match:")
        print(0, info)
        for i in range(1,100):
            action = P1.step(observation, env.action_space.n)
            observation, reward, terminated, truncated, info = env.step(action)
            print(i, info)
            if terminated or truncated:
                break

    if args.stats:
        # some analysis at particular states
        T = args.T
        grid_size = args.s
        band_size = 20
        action_labels = ["C","A1","A2","A3","D1","D2","D3"]
        # instance stats
        fig, axs = plt.subplots(grid_size,grid_size,figsize=(10,10))
        fig.suptitle(f"Model state-action value (dot) & sampling prob (bar) by proportional(blue) & softmax(red) @ T={T}")
        Q = Qh[0]
        Na = Qh.shape[-1]
        for S1 in range(grid_size):
            for S2 in range(grid_size):
                S = S1 * (env.rule.n_max_energy+1) + S2
                ax2 = axs[S1,S2].twinx()
                uniform_p = threshed_uniform_policy(Q, S, Q[S].shape[0], thresh=thresh('threshed_uniform'), output_p=True)
                # ax2.plot(np.arange(Na),Na*[uniform_p.max()], color="black",alpha=0.25)
                ax2.bar(np.arange(Na),softmax_sampling_policy(Q, T, S, Q[S].shape[0], thresh=thresh('softmax'), output_p=True),
                        color='tab:red',label="sm",alpha=0.25)
                ax2.bar(np.arange(Na),proportional_policy(Q, S, Q[S].shape[0], thresh=thresh('proportional'), output_p=True),
                        color='tab:blue',label="p",alpha=0.25)
                # ax2.bar(np.arange(Na),threshed_uniform_policy(Q, S, Q[S].shape[0], thresh=thresh('threshed_uniform'), output_p=True),
                #         color='tab:green',label="p",alpha=0.25)
                axs[S1,S2].plot(np.arange(Na),Q[S],color='C1',linestyle='None',marker='o')
                # axs[S1,S2].bar(np.arange(Na),Q[S],color='C1', width=0.1)
                
                axs[S1,S2].text(3, 0, f"({S1},{S2})",
                            ha="center", va="center", color="black", alpha=0.15, fontsize=20, weight='bold')
                axs[S1,S2].set_xticks(np.arange(Na),action_labels)
                axs[S1,S2].tick_params(axis='y', labelcolor='C1')
                axs[S1,S2].set_ylim(-1, 1)
                ax2.set_ylim(0, 1)
                axs[S1,S2].grid()
                if S1==grid_size-1:
                    axs[S1,S2].set_xlabel("A")
                else:
                    axs[S1,S2].xaxis.set_ticklabels([])
                if S2==0:
                    axs[S1,S2].set_ylabel("Q(S,A)", color='C1')
                    ax2.yaxis.set_ticklabels([])
                elif S2==grid_size-1:
                    ax2.tick_params(axis='y', labelcolor='tab:purple')
                    ax2.set_ylabel("Prob", color='tab:purple')
                    axs[S1,S2].yaxis.set_ticklabels([])
                else:
                    axs[S1,S2].yaxis.set_ticklabels([])
                    ax2.yaxis.set_ticklabels([])
                
        fig.tight_layout()

        # state-action value grid
        
        fig, axs = plt.subplots(grid_size,grid_size,figsize=(10,10))
        fig.suptitle("Bank state-action value Q(s,a) @ varying s")
        for S1 in range(grid_size):
            for S2 in range(grid_size):
                S = S1 * (env.rule.n_max_energy+1) + S2
                axs[S1,S2].boxplot(Qh[:band_size,S])
                axs[S1,S2].text(4, 0, f"({S1},{S2})",
                            ha="center", va="center", color="black", alpha=0.15, fontsize=20, weight='bold')
                axs[S1,S2].set_xticks(np.arange(Qh.shape[-1])+1,action_labels)
                axs[S1,S2].set_ylim(-1, 1) 
                axs[S1,S2].grid()
                if S1==grid_size-1:
                    axs[S1,S2].set_xlabel("A")
                else:
                    axs[S1,S2].xaxis.set_ticklabels([])
                if S2==0:
                    axs[S1,S2].set_ylabel("Q(S,A)")
                else:
                    axs[S1,S2].yaxis.set_ticklabels([])
        fig.tight_layout()
        plt.show()

    if args.tour:
        T = args.T
        num_matches_per_pair = 200
        max_steps = 30
        num_models = 20
        random_start = args.r

        Qi = Qh[::Qh.shape[0]/num_models][::-1]

        NP = Qi.shape[0]
        R = np.zeros([NP,NP])
        ns = env.rule.n_max_energy + 1
        state_freq = np.zeros(env.observation_space.n)
        state_value = [list() for i in range(env.observation_space.n)]
        win_last_state_freq = np.zeros(env.observation_space.n)
        tot_matches = num_matches_per_pair*(1+NP)*NP/2
        print("running tournament, total matches: {}".format(tot_matches))

        logs = []
        for i in tqdm(range(NP), position=0):
            Lj = []
            for j in tqdm(range(NP), position=1, leave=False):
                if j<i:
                    R[i,j]=-R[j,i]
                    continue
                Lk = []
                for k in range(num_matches_per_pair):
                    
                    P1 = SarsaAgent(Qi[i], T=T, mode=args.sampling_method, thresh=thresh(args.sampling_method))
                    P2 = SarsaAgent(Qi[j], T=T, mode=args.sampling_method, thresh=thresh(args.sampling_method))

                    observation, info = env.reset(opponent=P2, train=random_start)
                    Lt = [info]
                    for t in range(max_steps):
                        action = P1.step(observation, env.action_space.n)
                        observation, reward, terminated, truncated, info = env.step(action)
                        Lt.append(info)
                        state_freq[observation]+=1
                        if terminated:
                            R[i,j]+=reward
                            obs_set = set([inf["observation"] for inf in Lt])
                            for obs in obs_set:
                                state_value[obs]+=[reward]
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

        # V(s)
        state_value = np.array([np.average(r) for r in state_value])
        state_value = state_value[:ns**2]
        state_value = state_value.reshape(ns,ns)
        fig, ax = plt.subplots(figsize=(6,6))
        im = ax.imshow(-state_value, cmap = "RdBu_r")
        ax.set_title("State-value V(s) @ varying states")
        ax.set_xticks(np.arange(ns),np.arange(ns))
        ax.set_xlabel("S2")
        ax.set_yticks(np.arange(ns),np.arange(ns))
        ax.set_ylabel("S1")
        for i in range(ns):
            for j in range(ns):
                text = ax.text(j, i, round(state_value[i, j],2),
                            ha="center", va="center", color="w")
        fig.tight_layout()

        # stats
        print("total match finished: {}".format(env.rule.num_matches))
        print("violation end: {}".format(env.rule.violation))
        print("stupidity end: {}".format(env.rule.stupidity))
        print("busted end: {}".format(env.rule.busted))
        print("outpowered end: {}".format(env.rule.outpowered))
        print("mismatch end: {}".format(env.rule.mismatch))
        plt.show()

    if args.selfplay:
        N = 1e5
        print(f"running selfplay {int(N)} times")
        print(f"sampling methods: P1: {args.sampling_method}, P2: {args.selfplay_sampling_method}")
        P1 = SarsaAgent(Qh[0], T=args.T, mode=args.sampling_method, thresh=thresh(args.sampling_method))
        P2 = SarsaAgent(Qh[0], T=args.T, mode=args.selfplay_sampling_method, thresh=thresh(args.selfplay_sampling_method))
        rewards = []
        length = []
        ns = env.rule.n_max_energy + 1
        state_freq = np.zeros(env.observation_space.n)
        state_value = [list() for i in range(env.observation_space.n)]
        win_last_state_freq = np.zeros(env.observation_space.n)
        for i in tqdm(range(int(N))):
            observation, info = env.reset(opponent=P2, train=args.r)
            Lt = [info]
            for t in itertools.count():
                action = P1.step(observation, env.action_space.n)
                observation, reward, terminated, truncated, info = env.step(action)
                Lt.append(info)
                state_freq[observation]+=1
                if terminated:
                    rewards+=[reward]
                    length+=[env._i_step]
                    obs_set = set([inf["observation"] for inf in Lt])
                    for obs in obs_set:
                        state_value[obs]+=[reward]
                    if reward==1:
                        win_last_state_freq[Lt[-2]["observation"]]+=1
                    break
                if truncated:
                    length+=[env._i_step]
                    break
        

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
        ax.set_title("P1 Winning state frequency")
        ax.set_xticks(np.arange(ns),np.arange(ns))
        ax.set_xlabel("S2")
        ax.set_yticks(np.arange(ns),np.arange(ns))
        ax.set_ylabel("S1")
        for i in range(ns):
            for j in range(ns):
                text = ax.text(j, i, round(win_last_state_freq[i, j],2),
                            ha="center", va="center", color="w")
        fig.tight_layout()

        # V(s)
        state_value = np.array([np.average(r) for r in state_value])
        state_value = state_value[:ns**2]
        state_value = state_value.reshape(ns,ns)
        fig, ax = plt.subplots(figsize=(6,6))
        im = ax.imshow(-state_value, cmap = "RdBu_r")
        ax.set_title("State-value V(s) @ varying states")
        ax.set_xticks(np.arange(ns),np.arange(ns))
        ax.set_xlabel("S2")
        ax.set_yticks(np.arange(ns),np.arange(ns))
        ax.set_ylabel("S1")
        for i in range(ns):
            for j in range(ns):
                text = ax.text(j, i, round(state_value[i, j],2),
                            ha="center", va="center", color="w")
        fig.tight_layout()

        rewards = np.array(rewards)
        win = rewards[rewards>0].shape[0]
        print(f"total match finished within {env.max_episode_steps} steps: {len(rewards)}")
        print(f"win/loss={win}/{len(rewards)-win}")
        print(f"violation end: {env.rule.violation}")
        print(f"stupidity end: {env.rule.stupidity}")
        print(f"busted end: {env.rule.busted}")
        print(f"outpowered end: {env.rule.outpowered}")
        print(f"mismatch end: {env.rule.mismatch}")
        print(f"mean length = {np.average(length)} +- {np.std(length)/np.sqrt(len(length))}")
        print(f"mean reward = {np.average(rewards)} +- {np.std(rewards)/np.sqrt(len(rewards))}")
        plt.show()