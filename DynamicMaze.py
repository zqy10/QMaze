import copy
import random

from Maze import Maze

class DynamicMaze(Maze):
    def __init__(self, maze_size=5, dynamic_prob=0.2):
        """
        初始化动态迷宫
        :param maze_size: 迷宫的宽度
        :param dynamic_prob: 基础迷宫中原本畅通的道路，在每一步被临时墙壁阻断的概率
        """
        super(DynamicMaze, self).__init__(maze_size)
        self.dynamic_prob = dynamic_prob
        
        # 保存由 Prim 算法生成的、确保有解的基础迷宫拓扑
        # 在随后的动态演化中，所有的临时墙壁都基于这个拓扑生成
        self.base_maze_data = copy.deepcopy(self.maze_data)

    def update_dynamic_walls(self):
        """
        在基础迷宫的基础上，按照概率随机生成临时墙壁。
        """
        # 每步刷新前，先恢复为保证连通性的基础迷宫
        self.maze_data = copy.deepcopy(self.base_maze_data)
        
        r, c, _ = self.maze_data.shape
        for i in range(r):
            for j in range(c):
                # 遍历基础迷宫中本来是通路的右侧(z=1)和下侧(z=2)
                # 按照 dynamic_prob 概率临时封闭，并严格保持相邻格子的墙壁对称性
                
                # 检查右侧边界
                if j < c - 1 and self.base_maze_data[i, j, 1] == 1:
                    if random.random() < self.dynamic_prob:
                        self.maze_data[i, j, 1] = 0        # 封闭当前格子的右墙
                        self.maze_data[i, j+1, 3] = 0      # 封闭右侧格子的左墙
                
                # 检查下侧边界
                if i < r - 1 and self.base_maze_data[i, j, 2] == 1:
                    if random.random() < self.dynamic_prob:
                        self.maze_data[i, j, 2] = 0        # 封闭当前格子的下墙
                        self.maze_data[i+1, j, 0] = 0      # 封闭下方格子的上墙

    def move_robot(self, direction):
        """
        重写移动逻辑：在真正执行碰撞检测前，刷新一次环境的动态墙壁。
        这使得环境对于智能体而言呈现为纯粹的随机转移概率分布。
        """
        # 环境在智能体采取动作的瞬间发生动态随机变化
        self.update_dynamic_walls()
        
        # 调用父类现成的碰撞检测和位置更新逻辑
        return super(DynamicMaze, self).move_robot(direction)
    
if __name__ == "__main__":
    # 创建动态迷宫，畅通道路有 30% 的概率会临时封死
    dynamic_maze = DynamicMaze(maze_size=5, dynamic_prob=0.3)
    
    # 你可以循环打印迷宫，观察墙壁的动态闪烁现象
    for _ in range(3):
        dynamic_maze.update_dynamic_walls()
        print(dynamic_maze)