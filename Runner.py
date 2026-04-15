import numpy as np
from collections import defaultdict
import torch

from tqdm.auto import tqdm
from copy import deepcopy

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import colors as mcolors


class Runner(object):
    def __init__(self, robot):
        self.maze = robot.maze
        self.robot = robot

        self.train_robot_record = []
        self.test_robot_record = []
        self.train_robot_statics = {
            'success': [],
            'reward': [],
            'times': [],
        }
        self.test_robot_statics = {
            'success': [],
            'reward': [],
            'times': [],
        }

        self.display_direction = False

    def _state_q_values(self, state, use_target_model=True):
        """
        获取给定状态的 Q 值向量，兼容 DQN 与 Q-table。
        """
        normalized_state = np.array(state, dtype=np.float32)

        # DQN: 优先使用 target_model（测试更稳定）
        if use_target_model and hasattr(self.robot, 'target_model'):
            model = self.robot.target_model
            device = getattr(self.robot, 'device', 'cpu')
            state_tensor = torch.from_numpy(normalized_state).float().to(device)
            with torch.no_grad():
                return model(state_tensor).detach().cpu().numpy()

        if hasattr(self.robot, 'eval_model'):
            model = self.robot.eval_model
            device = getattr(self.robot, 'device', 'cpu')
            state_tensor = torch.from_numpy(normalized_state).float().to(device)
            with torch.no_grad():
                return model(state_tensor).detach().cpu().numpy()

        # 传统 Q-table
        if hasattr(self.robot, 'create_Qtable_line') and hasattr(self.robot, 'q_table'):
            self.robot.create_Qtable_line(state)
            return np.array([
                self.robot.q_table[state][a] for a in self.robot.valid_action
            ], dtype=float)

        raise RuntimeError("Robot does not provide a supported Q-value interface.")

    def save_max_q_image(self, filename, use_target_model=True, decimals=2):
        """
        在每个网格标注 Q 值最大的动作与对应 Q 值。
        """
        fig = plt.figure(figsize=(6, 6))
        self.maze.draw_maze()
        ax = plt.gca()

        action_to_symbol = {
            'u': 'U',
            'r': 'R',
            'd': 'D',
            'l': 'L',
        }

        for row in range(self.maze.maze_size):
            for col in range(self.maze.maze_size):
                state = (row, col)
                q_values = self._state_q_values(state, use_target_model=use_target_model)

                # valid_actions = self.maze.can_move_actions(state)
                # if not valid_actions:
                #     continue

                # best_action = max(
                #     valid_actions,
                #     key=lambda a: q_values[self.robot.valid_action.index(a)]
                # )
                best_action_index = np.argmax(q_values)
                best_action = self.robot.valid_action[best_action_index]
                best_q = float(q_values[self.robot.valid_action.index(best_action)])

                label = "{}\n{:.{}f}".format(
                    action_to_symbol.get(best_action, best_action),
                    best_q,
                    decimals
                )
                ax.text(
                    col + 0.5,
                    row + 0.5,
                    label,
                    ha='center',
                    va='center',
                    fontsize=8,
                    color='black',
                    bbox=dict(facecolor='white', alpha=0.65, edgecolor='none', pad=1.0)
                )

        ax.set_title("Max-Q Action and Value per Cell")
        fig.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)

    def _get_epoch_records(self, epoch_id):
        return [
            record for record in self.train_robot_record
            if record['id'][0] == epoch_id
        ]

    def _get_test_epoch_records(self, epoch_id):
        return [
            record for record in self.test_robot_record
            if record['id'][0] == epoch_id
        ]

    def save_epoch_image(self, epoch_id, filename):
        """
        保存单个 epoch 的轨迹图。
        """
        epoch_records = self._get_epoch_records(epoch_id)
        if not epoch_records:
            raise ValueError("No training record found for epoch {}".format(epoch_id))

        self._save_records_image(epoch_records, filename, epoch_id, phase_name="Train")

    def save_test_epoch_image(self, epoch_id, filename):
        """
        保存单个 testing epoch 的轨迹图。
        """
        epoch_records = self._get_test_epoch_records(epoch_id)
        if not epoch_records:
            raise ValueError("No testing record found for epoch {}".format(epoch_id))

        self._save_records_image(epoch_records, filename, epoch_id, phase_name="Test")

    def _save_records_image(self, epoch_records, filename, epoch_id, phase_name):

        fig = plt.figure(figsize=(6, 6))
        self.maze.draw_maze()
        ax = plt.gca()

        # 提取该 epoch 的状态轨迹
        trajectory = [record['state'] for record in epoch_records]
        x_values, y_values = self._build_compact_display_points(trajectory)

        # 使用细实线+时间渐变绘制轨迹（从早到晚颜色渐变）
        if len(x_values) > 1:
            points = np.array([x_values, y_values]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            line = LineCollection(
                segments,
                cmap='viridis',
                norm=mcolors.Normalize(vmin=0, vmax=max(len(segments) - 1, 1)),
                linewidths=1.2,
                linestyles='solid',
                alpha=0.95,
            )
            line.set_array(np.arange(len(segments)))
            ax.add_collection(line)

        ax.scatter(x_values[0], y_values[0], color='orange', s=80, label='start', zorder=3)
        ax.scatter(x_values[-1], y_values[-1], color='red', s=80, label='end', zorder=3)

        # 叠加当前终止位置的机器人
        robot = plt.Circle((x_values[-1], y_values[-1]), 0.35, color='red', alpha=0.5)
        ax.add_patch(robot)

        success = int(self.maze.destination == trajectory[-1])
        gamma = float(getattr(self.robot, 'gamma', 1.0))
        discounted_return = 0.0
        for t, record in enumerate(epoch_records):
            discounted_return += (gamma ** t) * float(record.get('reward', 0.0))
        ax.set_title(
            "{} Epoch {} | steps={} | success={} | return={:.2f}".format(
                phase_name,
                epoch_id,
                max(len(epoch_records) - 1, 0),
                success,
                discounted_return,
            )
        )

        ax.text(
            0.02,
            0.02,
            "Return(gamma={:.3f}): {:.2f}".format(gamma, discounted_return),
            transform=ax.transAxes,
            fontsize=10,
            color='black',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
        )
        ax.legend(loc='lower right')

        fig.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)

    def _build_compact_display_points(self, trajectory):
        """
        对重复经过的格子做小范围偏移，减少路径重叠。
        偏移限制在单元格内部，避免越过网格线。
        """
        visit_counter = defaultdict(int)
        compact_points = []

        # 8 个方向循环偏移，半径逐步增加且有上限（<= 0.22）
        directions = [
            (1.0, 0.0), (0.707, 0.707), (0.0, 1.0), (-0.707, 0.707),
            (-1.0, 0.0), (-0.707, -0.707), (0.0, -1.0), (0.707, -0.707),
        ]
        radius_levels = [0.0, 0.06, 0.12, 0.18, 0.22]

        for state in trajectory:
            visit_id = visit_counter[state]
            visit_counter[state] += 1

            direction = directions[visit_id % len(directions)]
            radius = radius_levels[min(visit_id // len(directions), len(radius_levels) - 1)]
            dx = direction[0] * radius
            dy = direction[1] * radius

            # 坐标系与绘图保持一致: x 对应列, y 对应行
            x = state[1] + 0.5 + dx
            y = state[0] + 0.5 + dy
            compact_points.append((x, y))

        x_values = [p[0] for p in compact_points]
        y_values = [p[1] for p in compact_points]
        return x_values, y_values

    def add_statics(self, accumulated_reward, run_times):

        self.train_robot_statics['reward'].append(accumulated_reward)
        self.train_robot_statics['times'].append(run_times)

        if self.maze.robot['loc'] == self.maze.destination:
            self.train_robot_statics['success'].append(1)
        else:
            self.train_robot_statics['success'].append(0)

    def add_test_statics(self, accumulated_reward, run_times):

        self.test_robot_statics['reward'].append(accumulated_reward)
        self.test_robot_statics['times'].append(run_times)

        if self.maze.robot['loc'] == self.maze.destination:
            self.test_robot_statics['success'].append(1)
        else:
            self.test_robot_statics['success'].append(0)

    def run_training(self, training_epoch, training_per_epoch=150, epoch_image_dir=None):
        epoch_iter = tqdm(range(training_epoch), desc="Training", leave=True)
        for e in epoch_iter:
            total_return = 0
            run_times = 0
            for i in range(training_per_epoch):

                current_record = {
                    'id': [e, i],
                    'success': False,
                    'state': self.maze.sense_robot(),
                }

                if current_record['state'] == self.maze.destination:
                    current_record['success'] = True
                    self.train_robot_record.append(current_record)
                    break

                action, reward = self.robot.train_update()
                current_record['action'] = action
                current_record['reward'] = reward
                self.train_robot_record.append(current_record)

                run_times += 1
                total_return += reward * (self.robot.gamma ** i)

            self.add_statics(total_return, run_times)
            epoch_iter.set_postfix({
                "steps": run_times,
                "return": round(float(total_return), 2),
                "success": int(self.maze.robot['loc'] == self.maze.destination),
            })

            if epoch_image_dir is not None:
                import os
                os.makedirs(epoch_image_dir, exist_ok=True)
                self.save_epoch_image(
                    epoch_id=e,
                    filename=os.path.join(epoch_image_dir, "epoch_{:04d}.png".format(e))
                )

            self.robot.reset()

    def run_testing(self, testing_epoch=1, testing_per_epoch=None, epoch_image_dir=None):
        height, width, _ = self.maze.maze_data.shape
        if testing_per_epoch is None:
            testing_per_epoch = int(height * width * 0.85)

        existing_test_epochs = [
            record['id'][0] for record in self.test_robot_record
            if 'id' in record and isinstance(record['id'], list) and len(record['id']) == 2
        ]
        epoch_base = max(existing_test_epochs) + 1 if existing_test_epochs else 0

        for e in range(testing_epoch):
            self.robot.reset()
            epoch_id = epoch_base + e

            accumulated_reward = 0.
            run_times = 0

            for i in range(testing_per_epoch):
                current_record = {
                    'id': [epoch_id, i],
                    'success': False,
                    'state': self.maze.sense_robot(),
                }

                if current_record['state'] == self.maze.destination:
                    current_record['success'] = True
                    self.test_robot_record.append(current_record)
                    break

                action, reward = self.robot.test_update()
                current_record['action'] = action
                current_record['reward'] = reward
                self.test_robot_record.append(current_record)

                run_times += 1
                accumulated_reward += reward

            self.add_test_statics(accumulated_reward, run_times)

            if epoch_image_dir is not None:
                import os
                os.makedirs(epoch_image_dir, exist_ok=True)
                self.save_test_epoch_image(
                    epoch_id=epoch_id,
                    filename=os.path.join(epoch_image_dir, "test_epoch_{:04d}.png".format(epoch_id))
                )

    def __init_gif(self):
        self.maze.draw_maze()
        fig = plt.gcf()
        ax = plt.gca()
        robot = plt.Circle((0, 0), 0.5, color="red")
        x, y = self.maze.robot['loc'][0] + 0.5, self.maze.robot['loc'][1] + 0.5
        robot.center = (y, x)
        ax.add_patch(robot)

        text_epoch = ax.text(
            0, -0.1,
            '',
            fontsize=20,
            horizontalalignment='left',
            verticalalignment="bottom"
        )

        text_step = ax.text(
            self.maze.maze_size, -0.1,
            '',
            fontsize=20,
            horizontalalignment='right',
            verticalalignment="bottom",
        )
        return fig, ax, robot, text_epoch, text_step

    def generate_gif(self, filename):
        fig, ax, robot, text_epoch, text_step = self.__init_gif()
        p_bar = tqdm(
            total=len(self.train_robot_record),
            desc="正在将训练过程转换为gif图, 请耐心等候...",
        )

        def update(record):
            x, y = record['state'][0] + 0.5, record['state'][1] + 0.5
            robot.center = (y, x)

            text_epoch.set_text("epoch:" + str(record['id'][0]))
            text_step.set_text("step:" + str(record['id'][1]))

            p_bar.update(1)
            return robot,

        def init(): pass  # do nothing


        import matplotlib.animation as animation
        ani = animation.FuncAnimation(
            fig,
            update,
            init_func=init,
            frames=self.train_robot_record,
            interval=200,
            blit=False,
        )

        # To save the animation, use e.g.
        ani.save(filename, writer='pillow')
        plt.close()

    def plot_results(self):
        plt.figure(figsize=(12, 4))
        plt.subplot(131)
        plt.title("Success Times")
        plt.plot(np.cumsum(self.train_robot_statics['success']))
        plt.subplot(132)
        plt.title("Accumulated Rewards")
        plt.plot(np.array(self.train_robot_statics['reward']))
        plt.subplot(133)
        plt.title("Runing Times per Epoch")
        plt.plot(np.array(self.train_robot_statics['times']))
        plt.show()
