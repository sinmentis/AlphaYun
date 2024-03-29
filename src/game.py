from action import *
from src.player import Player


class Game:
    def __init__(self, player_list: list[Player]):
        self.player_list = player_list
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

    def request_user_action(self, player: Player, action: BaseAction | None = None):
        # TODO: Need input error handler just in case user being idoit
        if not action:
            print(f"[{player.id}]{player.name}! Choose your next move:")
            for action in player.get_available_action_list():
                print(f"\t[{action.value}]\t{action.name}")

            next_action_value = int(input("Enter the number of your action.\n=> "))
            next_action_type = ActionType(next_action_value)
            available_action_level_list = player.get_available_action_level(next_action_type)
            if len(available_action_level_list) > 1:
                level = int(input(f"\t\tPlease select level from: {available_action_level_list}\n=> "))
            else:
                level = 1
            target_list = [player.id]  # Default to self
            if next_action_type == ActionType.ATTACK:
                target_list = [target.id for target in self.player_list if target.id != player.id]

            action = BaseAction(next_action_type, player.id, target_list, level)
        player.add_action(action)

    def process_round(self):
        for player in self.player_list:
            if player.dead:
                continue

            action = player.next_action
            if action.action_type == ActionType.ATTACK:
                player.action_attack(action.level)
                for target_player in map(self.get_player_by_id, action.target_id_list):
                    if target_player.next_action.level == action.level and target_player.next_action.action_type in [ActionType.ATTACK, ActionType.DEFENCE]:
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
        [self.request_user_action(player) for player in self.player_list]
        self.process_round()