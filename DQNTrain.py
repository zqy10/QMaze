import matplotlib.pyplot as plt
from DrawStatistics import plot_broken_line
from Runner import Runner
from torch_py.MinDQNRobot import MinDQNRobot as TorchRobot
from Robot import Robot
from Maze import Maze
from DynamicMaze import DynamicMaze
import os
from tqdm.auto import tqdm

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 允许重复载入lib文件


def train_by_dqn_robot(times, maze_size=5):
    print("start times:", times)

    # maze = Maze(maze_size=maze_size)
    maze = DynamicMaze(maze_size=maze_size, prob_close=0.7, num_close=maze_size, prob_open=0.3, num_open=maze_size)
    maze.save_dynamic_candidates_image("results/size{}/dynamic_candidates.png".format(maze_size))

    """choose Keras or Torch version"""
    # robot = TorchRobot(maze=maze)
    robot = Robot(maze=maze, algorithm='qtable')
    if getattr(robot, 'algorithm', 'dqn') == 'dqn' and hasattr(robot, "get_state_feature") and robot.memory is not None:
        robot.memory.build_full_view(maze=maze, state_extractor=robot.get_state_feature)
    elif getattr(robot, 'algorithm', 'dqn') == 'dqn' and robot.memory is not None:
        robot.memory.build_full_view(maze=maze)

    """training by runner"""
    runner = Runner(robot=robot)
    # runner.run_training(100, 100, "results/size{}/train/".format(maze_size))、
    runner.run_training(
        training_epoch=10000,
        training_per_epoch=200,
        epoch_image_dir=None,
    )
    runner.save_max_q_image("results/size{}/max_q_map.png".format(maze_size), use_target_model=True)
    runner.plot_results()

    """Test Robot"""
    runner.run_testing(
        testing_epoch=1,
        testing_per_epoch=100,
        epoch_image_dir="results/size{}/test/".format(maze_size)
    )
    probability_items = runner.save_dynamic_probability_image(
        "results/size{}/dynamic_inferred_probs.png".format(maze_size),
        use_target_model=True,
        decimals=2,
    )
    runner.generate_gif("results/size{}/test/test_process.gif".format(maze_size))

    close_values = [item['p_hat'] for item in probability_items if item['edge_type'] == 'close' and item['valid']]
    open_values = [item['p_hat'] for item in probability_items if item['edge_type'] == 'open' and item['valid']]
    close_invalid = sum(1 for item in probability_items if item['edge_type'] == 'close' and not item['valid'])
    open_invalid = sum(1 for item in probability_items if item['edge_type'] == 'open' and not item['valid'])
    if close_values:
        print("[Infer] close mean p_hat={:.3f}, target prob_close={:.3f}".format(float(sum(close_values) / len(close_values)), maze.prob_close))
    print("[Infer] close invalid count={}".format(close_invalid))
    if open_values:
        print("[Infer] open mean p_hat={:.3f}, target prob_open={:.3f}".format(float(sum(open_values) / len(open_values)), maze.prob_open))
    print("[Infer] open invalid count={}".format(open_invalid))


if __name__ == "__main__":
    # tf 2.1
    generate_times = 1  # 测试次数，每次测试都会重新生成迷宫，并从零开始训练机器人
    for time in range(generate_times):
        train_by_dqn_robot(time, maze_size=5)
