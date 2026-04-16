import copy
import os
import random
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from Maze import Maze

class DynamicMaze(Maze):
    def __init__(self, maze_size=5, prob_close=0.3, num_close=4, prob_open=0.3, num_open=4):
        """
        初始化多路径动态迷宫
        :param maze_size: 迷宫尺寸
        :param prob_close: 选定的动态通路在每一步被临时关闭的概率
        :param num_close: 迷宫中随机选定的动态通路（红色）总数量
        :param prob_open: 选定的动态墙壁在每一步被临时打开的概率
        :param num_open: 迷宫中随机选定的动态墙壁（蓝色）总数量
        """
        super(DynamicMaze, self).__init__(maze_size)
        # 增加原地不动动作
        if 's' not in self.valid_actions:
            self.valid_actions.append('s')
        self.move_map['s'] = (0, 0)

        self.prob_close = prob_close
        self.prob_open = prob_open
        
        # 1. 保存基础迷宫拓扑 (Prim算法生成的单连通图)
        self.base_maze_data = copy.deepcopy(self.maze_data)
        
        # 2. 扫描并分类候选门
        close_candidates = []
        open_candidates = []
        r, c, _ = self.maze_data.shape
        
        for i in range(r):
            for j in range(c):
                # 检查右侧 (z=1)，排除迷宫最外侧边界
                if j < c - 1:
                    if self.base_maze_data[i, j, 1] == 1:
                        close_candidates.append((i, j, 1))
                    else:
                        open_candidates.append((i, j, 1))
                
                # 检查下侧 (z=2)，排除迷宫最外侧边界
                if i < r - 1:
                    if self.base_maze_data[i, j, 2] == 1:
                        close_candidates.append((i, j, 2))
                    else:
                        open_candidates.append((i, j, 2))
                        
        # 3. 随机选定固定数量的动态对象
        self.dynamic_close_doors = random.sample(close_candidates, min(num_close, len(close_candidates)))
        self.dynamic_open_doors = random.sample(open_candidates, min(num_open, len(open_candidates)))

    def update_dynamic_walls(self):
        """
        每步刷新环境：执行随机关闭通路和随机打开墙壁的操作。
        """
        # 每次刷新前，先完全恢复为基础迷宫
        self.maze_data = copy.deepcopy(self.base_maze_data)
        
        # 随机关闭原本的通路 (产生红色临时墙)
        for (i, j, z) in self.dynamic_close_doors:
            if random.random() < self.prob_close:
                if z == 1:
                    self.maze_data[i, j, 1] = 0
                    self.maze_data[i, j+1, 3] = 0
                elif z == 2:
                    self.maze_data[i, j, 2] = 0
                    self.maze_data[i+1, j, 0] = 0

        # 随机打开原本的墙壁 (使得蓝色墙壁消失，成为捷径)
        for (i, j, z) in self.dynamic_open_doors:
            if random.random() < self.prob_open:
                if z == 1:
                    self.maze_data[i, j, 1] = 1
                    self.maze_data[i, j+1, 3] = 1
                elif z == 2:
                    self.maze_data[i, j, 2] = 1
                    self.maze_data[i+1, j, 0] = 1

    def move_robot(self, direction):
        self.update_dynamic_walls()

        if direction == 's':
            self.robot['dir'] = direction
            if self.robot['loc'] == self.destination:
                return self.reward['destination']
            return self.reward['default']

        return super(DynamicMaze, self).move_robot(direction)

    def can_move_actions(self, position):
        """在基础四方向可行动作上，始终允许原地不动。"""
        actions = super(DynamicMaze, self).can_move_actions(position)
        if 's' not in actions:
            actions.append('s')
        return actions

    def _is_dynamic_open_door(self, i, j, z):
        """判断当前坐标和方向是否属于被选定的动态墙壁候选集"""
        if z == 1: return (i, j, 1) in self.dynamic_open_doors
        elif z == 2: return (i, j, 2) in self.dynamic_open_doors
        elif z == 3: return (i, j-1, 1) in self.dynamic_open_doors
        elif z == 0: return (i-1, j, 2) in self.dynamic_open_doors
        return False

    def draw_maze(self):
        grid_size = 1
        r, c, _ = self.maze_data.shape

        ax = plt.gca()
        ax.xaxis.set_ticks_position('top')
        ax.invert_yaxis()
        plt.axis('off')

        for i in range(r):
            for j in range(c):
                walls = self.maze_data[i, j]
                start_x = j * grid_size
                start_y = i * grid_size

                for z in range(4):
                    if walls[z] == 0:  # 当前存在墙壁
                        line_color = "black"
                        
                        if self.base_maze_data[i, j, z] == 1:
                            # 基础迷宫可通行，但当前有墙，说明是被动态关闭的通路
                            line_color = "red"
                        elif self._is_dynamic_open_door(i, j, z):
                            # 属于有概率打开的动态墙壁，且当前未打开
                            line_color = "blue"

                        linewidth = 2 if line_color != "black" else 1

                        if z == 0:
                            plt.hlines(start_y, start_x, start_x + grid_size, color=line_color, linewidth=linewidth)
                        elif z == 1:
                            plt.vlines(start_x + grid_size, start_y, start_y + grid_size, color=line_color, linewidth=linewidth)
                        elif z == 2:
                            plt.hlines(start_y + grid_size, start_x, start_x + grid_size, color=line_color, linewidth=linewidth)
                        elif z == 3:
                            plt.vlines(start_x, start_y, start_y + grid_size, color=line_color, linewidth=linewidth)

        rect_2 = plt.Rectangle(self.destination, grid_size, grid_size, edgecolor=None, color="green")
        ax.add_patch(rect_2)

    def save_dynamic_candidates_image(self, save_path, dpi=150):
        """
        绘制并保存动态候选示意图：
        - 所有可能关闭的通路（dynamic_close_doors）标红
        - 所有可能打开的墙壁（dynamic_open_doors）标蓝
        """
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        fig = plt.figure(figsize=(6, 6))

        # 先画基础迷宫（黑色静态墙）
        maze_backup = self.maze_data
        self.maze_data = copy.deepcopy(self.base_maze_data)
        try:
            Maze.draw_maze(self)

            def draw_edge(i, j, z, color, linewidth=2.6):
                start_x = j
                start_y = i
                if z == 1:
                    plt.vlines(start_x + 1, start_y, start_y + 1, color=color, linewidth=linewidth)
                elif z == 2:
                    plt.hlines(start_y + 1, start_x, start_x + 1, color=color, linewidth=linewidth)

            for (i, j, z) in self.dynamic_close_doors:
                draw_edge(i, j, z, color="red")

            for (i, j, z) in self.dynamic_open_doors:
                draw_edge(i, j, z, color="blue")

            legend_handles = [
                Line2D([0], [0], color="red", lw=2.6, label="May Close (Dynamic Path)"),
                Line2D([0], [0], color="blue", lw=2.6, label="May Open (Dynamic Wall)"),
            ]
            plt.legend(handles=legend_handles, loc="lower right")

            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        finally:
            self.maze_data = maze_backup
            plt.close(fig)

    def save_inferred_probabilities_image(self, probability_items, save_path, decimals=2, dpi=150):
        """
        基于动态候选图叠加反推概率标注并保存图片。
        """
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        fig = plt.figure(figsize=(6, 6))
        maze_backup = self.maze_data
        self.maze_data = copy.deepcopy(self.base_maze_data)

        probability_lookup = {}
        for item in probability_items:
            probability_lookup[item['edge']] = item

        try:
            Maze.draw_maze(self)
            ax = plt.gca()

            def draw_edge(i, j, z, color, linewidth=2.6):
                start_x = j
                start_y = i
                if z == 1:
                    plt.vlines(start_x + 1, start_y, start_y + 1, color=color, linewidth=linewidth)
                elif z == 2:
                    plt.hlines(start_y + 1, start_x, start_x + 1, color=color, linewidth=linewidth)

            def edge_text_position(i, j, z):
                if z == 1:
                    return j + 1.05, i + 0.5, 'left', 'center'
                return j + 0.5, i + 1.05, 'center', 'top'

            def draw_probability_text(i, j, z, color):
                item = probability_lookup.get((i, j, z), None)
                if item is None:
                    label = "p_hat=N/A"
                elif item.get('valid', False):
                    label = "p_hat={:.{}f}".format(float(item['p_hat']), decimals)
                else:
                    label = "p_hat=NaN"

                tx, ty, ha, va = edge_text_position(i, j, z)
                ax.text(
                    tx,
                    ty,
                    label,
                    ha=ha,
                    va=va,
                    fontsize=7,
                    color=color,
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0.6)
                )

            for (i, j, z) in self.dynamic_close_doors:
                draw_edge(i, j, z, color="red")
                draw_probability_text(i, j, z, color="red")

            for (i, j, z) in self.dynamic_open_doors:
                draw_edge(i, j, z, color="blue")
                draw_probability_text(i, j, z, color="blue")

            legend_handles = [
                Line2D([0], [0], color="red", lw=2.6, label="May Close + inferred p_hat"),
                Line2D([0], [0], color="blue", lw=2.6, label="May Open + inferred p_hat"),
            ]
            plt.legend(handles=legend_handles, loc="lower right")

            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        finally:
            self.maze_data = maze_backup
            plt.close(fig)

if __name__ == "__main__":
    # 实例化迷宫：10x10，设置 5条可能关闭的通路(30%概率)，5面可能打开的墙壁(40%概率)
    dynamic_maze = DynamicMaze(maze_size=10, prob_close=0.3, num_close=5, prob_open=0.4, num_open=5)
    
    for step in range(3):
        dynamic_maze.update_dynamic_walls()
        print(f"\n--- Step {step + 1} ---")
        print(dynamic_maze)