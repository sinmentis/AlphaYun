from dataclasses import dataclass
import enum


class ActionType(enum.Enum):
    NONE = 0
    YUN = 1
    DEFENCE = 2
    ATTACK = 3
    # ABSORB = 4


@dataclass
class BaseAction:
    action_type: ActionType
    initiator_id: int
    target_id_list: list[int]  # Can apply to multiple user
    level: int

    def __str__(self):
        return self.action_type

    def __repr__(self):
        return f"initiator_id: {str(self.initiator_id)}\t" \
               f"target: {' '.join([str(target) for target in self.target_id_list])}\t" \
               f"action_type: {str(self.action_type)}\tlevel: {self.level}"
