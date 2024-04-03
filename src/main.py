import sys,os
sys.path.append(os.path.abspath('..')) # run without installation
from src.game.player import Player
from src.game.game import Game


def main():
    player_list = [Player("BIG MONKEY"), Player("COM")]
    game = Game(player_list, bot_model_file="model/ckpts/bot.npy")
    game.run()


if __name__ == "__main__":
    main()
