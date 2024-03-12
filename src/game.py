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

    @staticmethod
    def get_initiator_action(player: Player) -> ActionType:
        # TODO: Not all player can be listed as target
        print(f"[{player.id}]{player.name}! Choose your next move:")
        for action in ActionType:
            print(f"\t[{action.value}]\t{action.name}")

        next_action_value = int(input("Enter the number of your action: "))
        while not any(next_action_value == item.value for item in ActionType):
            next_action_value = int(input("Invalid input: "))

        return ActionType(next_action_value)

    def get_action_target(self) -> list[int]:
        # TODO: Not all Action are available all the time
        self.print_player_name()
        target = int(input("What player ID do you wanna target? :"))
        while target not in [player.id for player in self.player_list]:
            target = int(input("Invalid id :"))

        return [target for player in self.player_list if player.id == target]

    def request_user_action(self, player: Player, action: BaseAction | None = None):
        if not action:
            action_type = self.get_initiator_action(player)
            target_list = [player.id]  # Default self
            if action_type == ActionType.ATTACK:
                target_list = self.get_action_target()
            action = BaseAction(action_type, player.id, target_list, 1)

        player.add_action(action)

    def process_round(self):
        """
        TODO:
        If player this round action is ActionType.YUN. no one attack target's player, They gain self.num_yun += 1
        If player this round action is ActionType.DEFENCE, and no one attack him, they gain self.num_defense += 1
        If player this round action is ActionType.ATTACK, and target is also attack with same level, nothing happens
        If player this round action is ActionType.ATTACK, and target is DEFENCE with same level, nothing happens
        If player this round action is ActionType.ATTACK, and target is ActionType.YUN, or less level attack or less level defence, game over player win
        """
        for player in self.player_list:
            print(repr(player))

    def run(self):
        while not self.game_stopped:
            self.run_once()

    def run_once(self):
        if self.game_stopped:
            raise Exception("Game already stopped")

        self.num_round += 1
        print(f"Round {self.num_round}")
        [self.request_user_action(player) for player in self.player_list]
        self.process_round()