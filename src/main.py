from player import Player
from game import Game


def main():
    player_list = [Player("BIG MONKEY"), Player("COM")]
    game = Game(player_list)
    game.run()


if __name__ == "__main__":
    main()
