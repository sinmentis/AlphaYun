import sys,os
sys.path.append(os.path.abspath('src'))
from player import Player
from game import Game


def main():
    player_list = [Player("BIG MONKEY"), Player("COM")]
    game = Game(player_list,bot_model_file="src/bot.npy")
    game.run()


if __name__ == "__main__":
    main()
