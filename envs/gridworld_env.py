import gym
import turtle
import numpy as np

def GridWorld(gridmap = None, is_slippery = False):
    if gridmap == None:
        gridmap = ['SFFF', 'FHFH', 'FFFH', 'HFFG']
    env = gym.make("FrozenLake-v0", desc = gridmap, is_slippery = False)
    env = FrozenLakeWapper(env)
    return env

class FrozenLakeWapper(gym.Wrapper): # Wrapper的功能其是就是对环境相关功能的再扩展（再修改），对未修改的内容保持不变
    def __init__(self, env):
        self.max_y = env.desc.shape[0]
        self.max_x = env.desc.shape[1]

        self.t = None # 海龟绘图
        self.unit = 50 # 指一个方格的边长大小
        self.env = env # 此处是要定义self.env的不然无法调用内部函数

    def draw_box(self, x, y, fillcolor = '', line_color = 'gray'):
        self.t.up() # 画笔抬起，移动到目标位置后再放下
        self.t.goto(x * self.unit, y * self.unit)
        self.t.color(line_color)
        self.t.fillcolor(fillcolor)
        self.t.setheading(90) # 设置turtle的朝向，0为东，90为北，180为西，270为南
        self.t.down() # 画笔放下准备画图
        self.t.begin_fill() # 开始填充
        for _ in range(4):
            self.t.forward(self.unit)
            self.t.right(90)
        self.t.end_fill()
    
    def move_player(self, x, y):
        self.t.up()
        self.t.setheading(90)
        self.t.fillcolor('red')
        self.t.goto((x + 0.5) * self.unit, (y + 0.5) * self.unit) # 使得agent在起止位置的正中间

    def render(self):
        if self.t == None: # 第一步时需要渲染一下环境
            self.t = turtle.Turtle()
            self.wn = turtle.Screen()
            self.wn.setup(self.unit * self.max_x + 300, self.unit * self.max_y + 300) # 设置屏幕为300x300的像素点，数值越大方格越大，并且居于显示器中间
            self.wn.setworldcoordinates(0 , 0, self.unit * self.max_x, self.unit * self.max_y ) # 设置世界坐标系左下和右上
            # self.t.shape('turtle')
            self.t.width(2) # 线条粗细
            self.t.speed(6) # 设置海龟速度
            self.t.color('gray')
            for i in range(self.desc.shape[0]):
                for j in range(self.desc.shape[1]):
                    x = j
                    y = self.max_y - 1- i
                    if self.desc[i][j] == b'S':  # Start
                        self.draw_box(x, y, 'white')
                    elif self.desc[i][j] == b'F':  # Frozen ice
                        self.draw_box(x, y, 'white')
                    elif self.desc[i][j] == b'G':  # Goal
                        self.draw_box(x, y, 'yellow')
                    elif self.desc[i][j] == b'H':  # Hole
                        self.draw_box(x, y, 'black')
                    else:
                        self.draw_box(x, y, 'white')

        x_pos = self.s % self.max_x # self.s表示当前的状态
        y_pos = self.max_y - 1 - int(self.s / self.max_x)
        self.move_player(x_pos, y_pos) # 移动到起始位置

class CliffWalkingWapper(gym.Wrapper):
    def __init__(self, env):
        self.env = env
        self.t = None
        self.unit = 50
        self.max_x = 12
        self.max_y = 4

    def draw_x_line(self, y, x0, x1, color = 'gray'): # 画每一行的线
        assert x1 > x0
        self.t.color(color)
        self.t.setheading(0)
        self.t.up()
        self.t.goto(x0, y)
        self.t.down()
        self.t.forward(x1 - x0)

    def draw_y_line(self, x, y0, y1, color = 'gray'): # 画每一列的线
        assert y1 > y0
        self.t.color(color)
        self.t.setheading(90)
        self.t.up()
        self.t.goto(x, y0)
        self.t.down()
        self.t.forward(y1 - y0)

    def draw_box(self, x, y, fillcolor = '', line_color = 'gray'):
        self.t.up()
        self.t.goto(x * self.unit, y * self.unit)
        self.t.color(line_color)
        self.t.fillcolor(fillcolor)
        self.t.setheading(90)
        self.t.down()
        self.t.begin_fill()
        for _ in range(4):
            self.t.forward(self.unit)
            self.t.right(90)
        self.t.end_fill()

    def move_player(self, x, y):
        self.t.up()
        self.t.setheading(90)
        self.t.fillcolor('red')
        self.t.goto((x + 0.5) * self.unit, (y + 0.5) * self.unit)

    def render(self):
        if self.t == None:
            self.t = turtle.Turtle()
            self.wn = turtle.Screen() # 初始化时方向朝东
            self.wn.setup(self.unit * self.max_x + 300, self.unit * self.max_y + 300)
            self.wn.setworldcoordinates(0, 0, self.max_x * self.unit, self.max_y * self.unit)
            self.t.shape('turtle')
            self.t.width(2)
            self.t.speed(10)
            self.t.color('blue')
            for _ in range(2): # 先画出环境的框架
                self.t.forward(self.max_x * self.unit)
                self.t.left(90)
                self.t.forward(self.max_y * self.unit)
                self.t.left(90)

            for i in range(1, self.max_y): # 画每一行
                self.draw_x_line(y = i * self.unit, x0 = 0, x1 = self.max_x * self.unit)

            for i in range(1, self.max_x): # 画每一列
                self.draw_y_line(x = i * self.unit, y0 = 0, y1 = self.max_y * self.unit)

            for i in range(1, self.max_x - 1): # 填充障碍物格子
                self.draw_box(i, 0, 'black')

            self.draw_box(self.max_x - 1, 0, 'yellow') # 填充目的地格子
        
        x_pos = self.s % self.max_x
        y_pos = self.max_y - 1 - int(self.s / self.max_x)
        self.move_player(x_pos, y_pos)

if __name__ == '__main__':
    # # 环境1：可以配置冰面是否是滑的
    # # 0 left, 1 down, 2 right, 3 up
    # env  = gym.make("FrozenLake-v0", is_slippery = False)
    # env = FrozenLakeWapper(env)

    # 环境2：CliffWalking, 悬崖环境
    env = gym.make("CliffWalking-v0")  # 0 up, 1 right, 2 down, 3 left
    env = CliffWalkingWapper(env)

    # 环境3：自定义格子世界，可以配置地图, S为出发点Start, F为平地Floor, H为洞Hole, G为出口目标Goal
    # gridmap = [
    #         'SFFF',
    #         'FHFF',
    #         'FFFF',
    #         'HFGF' ]
    # env = GridWorld(gridmap)
    env.reset()
    for step in range(10):
        action = np.random.randint(0, 4) # 随机产生动作
        obs, reward, done, info = env.step(action)
        print('step {}: action {}: obs {}: reward {}: done {}: info {}:'.format(step, action, obs, reward, done, info))
        env.render()
