from src.game.action import BaseAction, ActionType

MAX_DEFENSE_NUM = 3
MAX_ATTACK_LEVEL = 3


class Player:
    def __init__(self, name):
        self.name = name
        self.init_player(-1)

    def init_player(self, id):
        self.id = id
        self.num_yun = 0
        self.num_defense = 0
        self.action_history = []
        self.next_action = None
        self.set_death(False)

    def action_attack(self, level: int):
        self.num_yun -= level
        self.num_defense = 0  # Reset defense counter

    def action_defense(self, num: int = 1):
        self.num_defense += num

    def action_yun(self, num: int = 1):
        self.num_yun += num
        self.num_defense = 0  # Reset defense counter

    def set_death(self, is_dead: bool):
        self.dead = is_dead

    def add_action(self, action: BaseAction):
        self.next_action = action
        self.action_history.append(self.next_action)

    def get_available_action_level(self, action_type: ActionType) -> list[int]:
        match action_type:
            case ActionType.YUN:
                return [1]
            case ActionType.DEFENCE:
                return list(range(1, MAX_ATTACK_LEVEL+1))
            case ActionType.ATTACK:
                return list(range(1, min(self.num_yun, MAX_ATTACK_LEVEL)+1))

        raise Exception("The fuck happened?")

    def get_available_action_list(self):
        available_action_list = [ActionType.YUN]
        if self.num_yun > 0:
            available_action_list.append(ActionType.ATTACK)
        if 0 <= self.num_defense <= MAX_DEFENSE_NUM:
            available_action_list.append(ActionType.DEFENCE)

        return sorted(available_action_list)

    def __str__(self):
        return f"[{self.id}]: {self.name}"

    def __repr__(self):
        template = f"id: {self.id}\t name: {self.name}\n# YUN: {self.num_yun}\t #DEF: {self.num_defense}\nACTION_HIS:"
        try:
            template += "\n".join(self.action_history)
        except:
            pass
        return template
