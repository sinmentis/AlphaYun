"""
Script for model training
rlsn 2024
"""
from yunenv import YunEnv,RPSEnv
from sarsa import SarsaAgent, tabular_sarsa
import numpy as np
import argparse, random, time
from tqdm import tqdm

def selfplay(env, num_steps = 1000000, swap_steps=1000, bank_size=10, 
             play_against_latest_ratio=0.7, save_steps=50000, eps=0.1, T=0.3, eta=1, alpha=0.01, sampling_method='softmax'):
    Qs = [None]
    Q_history = []
    Q = None
    env.reset(opponent = None, train=True)

    n_saves = int(num_steps/save_steps)
    n_swaps_per_save = int(save_steps/swap_steps)
    for i_saves in tqdm(range(n_saves), desc="Saves", position=0):
        for i_swaps in tqdm(range(n_swaps_per_save), desc="Swaps", position=1, leave=False):
            if len(Qs)==1 or np.random.randn() < play_against_latest_ratio:
                opponent_Q = Qs[0]
            else:
                opponent_Q = random.choice(Qs[1:])
            env.reset(opponent = SarsaAgent(opponent_Q, T=T, mode=sampling_method))
            Q = tabular_sarsa(env, swap_steps, Q, discount=1.0, epsilon=eps, 
            alpha=alpha, eval_interval=-1,eta=eta,T=T,sampling_method=sampling_method,n_ternimal=env.n_ternimal)
        Qs = [np.copy(Q)] + Qs
        Qs = Qs[:bank_size]
        Q_history.append(np.copy(Q))
    return np.array(Q_history)[::-1]

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, help="set seed", default=None)
    parser.add_argument('--model_file', type=str, help="filename of the model to be saved", default="Qh.npy")
    parser.add_argument('--bank_size', type=int, help="bank size of the model", default=20)
    parser.add_argument('--num_steps', type=int, help="number of total training steps", default=1e7)
    parser.add_argument('--save_steps', type=int, help="number of training steps for each saved model", default=1e5)
    parser.add_argument('--swap_steps', type=int, help="number of training steps against each opponent", default=1e3)
    parser.add_argument('--play_against_latest_ratio', type=int, help="play against latest model ratio", default=0.05)

    parser.add_argument('--step_size', type=int, help="learning rate alpha", default=1e-2)
    parser.add_argument('--sampling_method', type=str, help="proportional or softmax", default='softmax')
    parser.add_argument('--T', type=float, help="hyperparameter T for softmax sampling method", default=0.2)
    parser.add_argument('--eps', type=float, help="hyperparameter epsilon for epsilon greedy policy", default=0.1)
    parser.add_argument('--eta', type=float, help="hyperparameter anticipatory parameter", default=0.1)

    args = parser.parse_args()
    
    if not args.seed:
        args.seed = int(time.time())
    np.random.seed(args.seed)
    print("running with seed", args.seed)
    env = YunEnv()
    # env = RPSEnv()

    args.play_against_latest_ratio = args.play_against_latest_ratio
    print("args:",args)

    print("Training...")
    start = time.time()
    Qh = selfplay(env, args.num_steps, args.swap_steps, args.bank_size, 
                  args.play_against_latest_ratio, args.save_steps,
                  eps=args.eps, T=args.T, alpha=args.step_size)
    np.save(args.model_file, Qh)
    print("Training complete, model saved at {}, elapsed {}s".format(args.model_file,round(time.time()-start,2)))