from action import BaseAction


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

    def add_action(self, action: BaseAction):
        self.next_action = action
        self.action_history.append(self.next_action)

    def __str__(self):
        return f"[{self.id}]: {self.name}"

    def __repr__(self):
        template = f"id: {self.id}\t name: {self.name}\n# YUN: {self.num_yun}\t #DEF: {self.num_defense}\nACTION_HIS:"
        try:
            template += "\n".join(self.action_history)
        except:
            pass
        return template