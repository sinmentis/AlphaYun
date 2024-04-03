from src.game.action import *
from src.game.player import Player
from src.model.agent import Agent
from src.model.env import YunEnv, Rule
import numpy as np

class Game:
    def __init__(self, player_list: list[Player], bot_model_file="bot.npy"):
        self.player_list = player_list
        self.bot_mdp = Rule()
        self.bot = Agent(np.load(bot_model_file), mode="prob")
        self.init_game()

    def init_game(self):
        self.num_round = 0
        self.game_stopped = False
        [player.init_player(index) for index, player in enumerate(self.player_list)]

    def print_player_name(self):
        print("ID\tName")
        [print(str(player)) for player in self.player_list]

    def get_player_by_id(self, id: int):
        for player in self.player_list:
            if player.id == id:
                return player
        raise Exception("{id} player can't be found")

    def _request_user_action(self, player: Player, action: BaseAction | None = None):
        if "COM" in player.name:
            action = self._handle_com_player_action(player)
        elif not action:
            action = self._handle_human_player_action(player)
        player.add_action(action)

    def _handle_com_player_action(self, player: Player):
        opponent = self.player_list[(player.id + 1) % len(self.player_list)]  # Cycle through opponents
        observation = YunEnv.convert_obs(player.num_yun, opponent.num_yun, self.bot_mdp)
        action_id = self.bot.step(observation)
        yun, atk, defs = self.bot_mdp.decode_action(action_id)

        if atk and ActionType.ATTACK in player.get_available_action_list():
            action_type, level = ActionType.ATTACK, atk
        elif defs and ActionType.DEFENCE in player.get_available_action_list():
            action_type, level = ActionType.DEFENCE, defs
        else:
            action_type, level = ActionType.YUN, 1

        target_list = [target.id for target in self.player_list if
                       target.id != player.id] if action_type == ActionType.ATTACK else [player.id]
        print(f"[{player.id}]{player.name}: {action_type.name}! (lvl {level})")
        return BaseAction(action_type, player.id, target_list, level)

    def _handle_human_player_action(self, player: Player):
        print(f"[{player.id}]{player.name}! Choose your next move:")
        for action in player.get_available_action_list():
            print(f"\t[{action.value}]\t{action.name}")

        next_action_type = self._prompt_for_action(player)
        level = self._prompt_for_level(player, next_action_type)

        target_list = [player.id]  # Default to self
        if next_action_type == ActionType.ATTACK:
            target_list = [target.id for target in self.player_list if target.id != player.id]

        return BaseAction(next_action_type, player.id, target_list, level)

    def _prompt_for_action(self, player: Player):
        while True:
            try:
                next_action_value = int(input("Enter the number of your action.\n=> "))
                next_action_type = ActionType(next_action_value)
                if next_action_type not in player.get_available_action_list():
                    raise ValueError
                return next_action_type
            except (ValueError, KeyError):
                print("Invalid action. Please try again.")

    def _prompt_for_level(self, player: Player, next_action_type: ActionType):
        available_action_level_list = player.get_available_action_level(next_action_type)
        if len(available_action_level_list) > 1:
            while True:
                try:
                    level = int(input(f"\t\tPlease select level from: {available_action_level_list}\n=> "))
                    if level in available_action_level_list:
                        return level
                    else:
                        raise ValueError
                except ValueError:
                    print("Invalid level. Please try again.")
        else:
            return available_action_level_list[0]

    def _process_round(self):
        for player in self.player_list:
            if player.dead:
                continue

            action = player.next_action
            if action.action_type == ActionType.ATTACK:
                player.action_attack(action.level)
                for target_player in map(self.get_player_by_id, action.target_id_list):
                    if target_player.next_action.level == action.level and target_player.next_action.action_type in [
                        ActionType.ATTACK, ActionType.DEFENCE]:
                        print(f"Player [{player.id}]'s attack been canceled out.")
                    else:
                        print(f"Player [{player.id}]'s attack killed player ID [{target_player.id}]")
                        target_player.set_death(True)

            elif action.action_type == ActionType.YUN:
                player.action_yun()

            elif action.action_type == ActionType.DEFENCE:
                player.action_defense()

        # Check game status by count up living player
        living_player = [1 for player in self.player_list if not player.dead]
        if sum(living_player) == 1:
            self.game_stopped = True
            print(f"Game over - {living_player[0]} win!")

    def run(self):
        while not self.game_stopped:
            self.run_once()

    def run_once(self):
        if self.game_stopped:
            raise Exception("Game already stopped")

        self.num_round += 1
        print(f"\nRound {self.num_round}\n")
        [self._request_user_action(player) for player in self.player_list]
        self._process_round()
