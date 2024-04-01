from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)


# 游戏的初始化逻辑
# 这里假设你的 Game 类和其他相关的类都定义在 game.py 文件中
# from game import Game, Player, etc

# 创建一个全局游戏实例
# game = Game(player_list=[Player(name="Player1"), Player(name="COM")])

@app.route('/')
def index():
    # 渲染一个初始页面，可以包含游戏开始的按钮等
    return render_template('index.html')


@app.route('/game', methods=['GET', 'POST'])
def game():
    if request.method == 'POST':
        # 这里处理玩家的动作，例如:
        # action_type = request.form.get('action_type')
        # game.process_action(action_type)
        return redirect(url_for('game'))
    # 渲染游戏的状态，例如玩家信息等
    return render_template('game.html')  # , players=game.players, current_player=game.current_player


@app.route('/rank')
def rank():
    # 示例数据
    users = [
        {"id": 1, "name": "Alice", "win_rate": 75, "win_number": 30},
        {"id": 2, "name": "Bob", "win_rate": 60, "win_number": 45},
        {"id": 3, "name": "Charlie", "win_rate": 85, "win_number": 22},
        {"id": 4, "name": "Dana", "win_rate": 90, "win_number": 10},
        # 更多用户...
    ]

    # 从请求中获取排序依据，默认按 win_rate 排序
    sort_by = request.args.get('sort_by', 'win_rate')
    reverse_sort = False if sort_by == 'id' else True  # 默认逆序，除了 id

    # 根据用户选择进行排序
    users.sort(key=lambda x: x[sort_by], reverse=reverse_sort)

    return render_template('rank.html', users=users)


@app.route('/about')
def about():
    return render_template('about.html')


@app.errorhandler(404)
def page_not_found(error):
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_server_error(error):
    return render_template('500.html'), 500


if __name__ == '__main__':
    app.run(debug=True)
