import heapq
from Maze import Maze

def my_search(maze):
    """
    实现A*算法
    :param maze: 迷宫对象
    :return :到达目标点的路径 如：["u","u","r",...]
    """
    path = []
    # -----------------请实现你的算法代码--------------------------------------
    start_loc = maze.sense_robot()
    destination = maze.destination
    move_map = {
        'u': (-1, 0),
        'r': (0, +1),
        'd': (+1, 0),
        'l': (0, -1),
    }
    def heuristic(loc, dest):
        # 曼哈顿距离作为启发式函数
        return abs(loc[0] - dest[0]) + abs(loc[1] - dest[1])
    # 优先队列，元素结构为：(f值, g值, 当前位置, 路径)
    start_h = heuristic(start_loc, destination)
    heap = [(start_h, 0, start_loc, [])]
    # 记录访问过的节点及其最小g值，用于剪枝
    visited = {start_loc: 0}
    while heap:
        f, g, current_loc, current_path = heapq.heappop(heap)
        # 判断是否到达终点
        if current_loc == destination:
            path = current_path
            break
        # 如果当前路径的代价比已记录的代价大，则跳过
        if g > visited.get(current_loc, float('inf')):
            continue
        # 拓展当前节点的邻居
        can_move = maze.can_move_actions(current_loc)
        for action in can_move:
            new_loc = tuple(current_loc[i] + move_map[action][i] for i in range(2))
            new_g = g + 1
            new_f = new_g + heuristic(new_loc, destination)
            new_path = current_path + [action]
            # 如果新位置未访问过，或者找到了更短的路径，则加入队列
            if new_loc not in visited or new_g < visited[new_loc]:
                visited[new_loc] = new_g
                heapq.heappush(heap, (new_f, new_g, new_loc, new_path))
    # -----------------------------------------------------------------------
    return path

if __name__ == "__main__":

    maze = Maze(maze_size=10) # 从文件生成迷宫

    path_2 = my_search(maze)
    print("搜索出的路径：", path_2)

    for action in path_2:
        maze.move_robot(action)

    if maze.sense_robot() == maze.destination:
        print("恭喜你，到达了目标点")
