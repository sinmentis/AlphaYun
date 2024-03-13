"""
Environment wrapper for agent training
rlsn 2024
"""
import numpy as np
import gymnasium as gym ### pip install gymnasium
from gymnasium import spaces
from sarsa import SarsaAgent

class YunEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, render_mode=None, rule=None, max_episode_steps=100):
        self.rule = rule  # The rule or anything informative of the game

        # Observation is a Cartesian space of the agent's and the opponent's energy,
        # as well as the current game state (ongoing 0/lose 1/win 2)
        self.observation_space = spaces.MultiDiscrete([rule.n_max_energy,rule.n_max_energy,3])
        self.observation_space.n = rule.n_max_energy**2+2

        # Action space is the maximum number of actions possible
        self.action_space = spaces.Discrete(rule.n_max_actions)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.max_episode_steps = max_episode_steps
        
    def _get_obs(self):
        # agent's observation as a int
        if self._game_state==0:
            return self._agent_state * self.rule.n_max_energy + self._opponent_state
        elif self._game_state==1:
            return self.rule.n_max_energy**2
        else:
            return self.rule.n_max_energy**2+1

    def _oppo_obs(self):
        # opponent's observation as a int
        if self._game_state==0:
            return self._opponent_state * self.rule.n_max_energy + self._agent_state
        elif self._game_state==1:
            return self.rule.n_max_energy**2+1
        else:
            return self.rule.n_max_energy**2

    def _get_info(self):
        return {
            "agent_action":self._agent_action,
            "opponent_action":self._opponent_action,
            "agent_state":self._agent_state,
            "opponent_state":self._opponent_state,
            "game_state":self._game_state
            }
    
    def reset(self, seed=None, opponent=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.opponent = opponent
        # game start
        self._game_state=0

        # initialize players' energy
        self._agent_state = self.rule.init_energy
        self._opponent_state = self.rule.init_energy


        # value init
        self._agent_action = None
        self._opponent_action = None

        observation = self._get_obs()
        info = self._get_info()
        self._i_step = 0
        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    
    def step(self, action):
        self._agent_action = action
        if self.opponent is not None:
            self._opponent_action = self.opponent.step(self._oppo_obs(), self.action_space.n)
        else:
            self._opponent_action = self.action_space.sample()
        self._agent_state, self._opponent_state, self._game_state = self.rule.step(agent_state=self._agent_state,
                                                                           opponent_state=self._opponent_state,
                                                                           agent_action=action,
                                                                           opponent_action=self._opponent_action)

        # An episode is done iff one agent has won
        terminated = self._game_state>0
        reward = 1 if self._game_state==2 else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        self._i_step+=1
        truncated = self._i_step>=self.max_episode_steps


        return observation, reward, terminated, truncated, info

    def close(self):
        # clean memory if needed
        pass

    def render(self):
        if self.render_mode == "human":
            return self._render_frame()

    def _render_frame(self):
        #TODO: render something on screen to monitor wtf is going on
        pass

def test():
    #TODO: replace the ""test_rule"" object to pass the following test, 
    # see to that the printed game log following random policy is correct
    class AttrDict(dict):
        def __init__(self, *args, **kwargs):
            super(AttrDict, self).__init__(*args, **kwargs)
            self.__dict__ = self

    # the object must implement the following attr/methods
    test_rule = AttrDict({"n_max_energy":5,"n_max_actions":5,"init_energy":1})
    def step(agent_state:int,
             opponent_state:int,
             agent_action:int,
             opponent_action:int):
        # return agent_next_state, opponent_next_state, game_next_state
        return 0,0,0
    test_rule.step = step

    ################################ test starts here ####################################
    env = YunEnv(rule=test_rule)

    print(env.observation_space.n)
    print(env.action_space.n)

    opponent = SarsaAgent(np.random.randn(env.observation_space.n,env.action_space.n))
    agent = SarsaAgent(np.random.randn(env.observation_space.n,env.action_space.n))
    observation, info = env.reset(seed=None, opponent=None)
    print(0, info)
    for i in range(1,10):
        action = agent.step(observation, env.action_space.n)
        observation, reward, terminated, truncated, info = env.step(action)
        print(i, info)
        if terminated or truncated:
            break

    print("pass")

if __name__=="__main__":
    test()