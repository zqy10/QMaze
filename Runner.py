import numpy as np
from collections import defaultdict, deque
import torch

from tqdm.auto import tqdm
from copy import deepcopy

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import colors as mcolors


class Runner(object):
    def __init__(
            self,
            robot,
            max_train_records=100,
            max_test_records=100,
            max_stat_points=100):
        self.maze = robot.maze
        self.robot = robot

        self.train_robot_record = deque(maxlen=max(1, int(max_train_records)))
        self.test_robot_record = deque(maxlen=max(1, int(max_test_records)))
        self.train_robot_statics = {
            'success': deque(maxlen=max(1, int(max_stat_points))),
            'reward': deque(maxlen=max(1, int(max_stat_points))),
            'times': deque(maxlen=max(1, int(max_stat_points))),
        }
        self.test_robot_statics = {
            'success': deque(maxlen=max(1, int(max_stat_points))),
            'reward': deque(maxlen=max(1, int(max_stat_points))),
            'times': deque(maxlen=max(1, int(max_stat_points))),
        }

        self._next_train_epoch_id = 0
        self._next_test_epoch_id = 0

        self.display_direction = False

    def _build_step_record(self, epoch_id, step_id, state, store_snapshot=False):
        record = {
            'id': [epoch_id, step_id],
            'success': False,
            'state': state,
        }
        if store_snapshot:
            record['maze_data'] = np.array(self.maze.maze_data, copy=True)
        return record

    def _state_q_values(self, state, use_target_model=True):
        """
        获取给定状态的 Q 值向量，兼容 DQN 与 Q-table。
        """
        # 如果 robot 有 get_state_feature 方法，则将其转换为扩展状态特征
        if hasattr(self.robot, 'get_state_feature'):
            feature_state = self.robot.get_state_feature(state)
            normalized_state = np.array(feature_state, dtype=np.float32)
        else:
            normalized_state = np.array(state, dtype=np.float32)

        # DQN: 优先使用 target_model（测试更稳定）
        target_model = getattr(self.robot, 'target_model', None)
        eval_model = getattr(self.robot, 'eval_model', None)

        if use_target_model and target_model is not None:
            model = target_model
            device = getattr(self.robot, 'device', 'cpu')
            state_tensor = torch.from_numpy(normalized_state).float().to(device)
            with torch.no_grad():
                return model(state_tensor).detach().cpu().numpy()

        if eval_model is not None:
            model = eval_model
            device = getattr(self.robot, 'device', 'cpu')
            state_tensor = torch.from_numpy(normalized_state).float().to(device)
            with torch.no_grad():
                return model(state_tensor).detach().cpu().numpy()

        # 传统 Q-table
        if hasattr(self.robot, 'create_Qtable_line') and hasattr(self.robot, 'q_table'):
            if getattr(self.robot, 'algorithm', '') == 'qtable' and hasattr(self.robot, 'get_state_feature'):
                state_key = tuple(self.robot.get_state_feature(state))
            else:
                state_key = state

            self.robot.create_Qtable_line(state_key)
            return np.array([
                self.robot.q_table[state_key][a] for a in self.robot.valid_action
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
            's': 'S',
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
        fig.clf()
        plt.close(fig)
        plt.close('all')
        import gc
        gc.collect()

    def _distance_penalty(self, loc):
        """
        与 Robot.train_update 保持一致的距离惩罚项。
        """
        distance_metric = getattr(self.robot, 'distance_metric', 'euclidean')
        distance_weight = float(getattr(self.robot, 'distance_weight', 0.0))

        dest_x, dest_y = self.maze.destination
        curr_x, curr_y = loc

        if distance_metric == "euclidean":
            dist = np.sqrt((curr_x - dest_x) ** 2 + (curr_y - dest_y) ** 2)
            max_dist = np.sqrt(2) * max(1, (self.maze.maze_size - 1))
        else:
            dist = abs(curr_x - dest_x) + abs(curr_y - dest_y)
            max_dist = max(1, 2 * (self.maze.maze_size - 1))

        normalized_dist = dist / max_dist
        return normalized_dist * distance_weight

    def _reward_with_shaping(self, next_loc, is_hit_wall):
        """
        依据 Robot.train_update 的逻辑构造单步奖励。
        """
        if is_hit_wall:
            base_reward = float(self.maze.reward['hit_wall'])
        elif next_loc == self.maze.destination:
            base_reward = float(self.maze.reward['destination'])
        else:
            base_reward = float(self.maze.reward['default'])

        # 与训练逻辑一致：仅在非终点时叠加距离惩罚
        if next_loc != self.maze.destination:
            base_reward -= self._distance_penalty(next_loc)
        return base_reward

    def _edge_action_and_states(self, edge):
        """
        将候选边 (i, j, z) 映射到动作与通过后的目标状态。
        """
        i, j, z = edge
        if z == 1:
            return 'r', (i, j), (i, j + 1)
        if z == 2:
            return 'd', (i, j), (i + 1, j)
        raise ValueError("Unsupported edge direction z={}".format(z))

    def _q_values_from_feature(self, feature_state, use_target_model=True):
        """
        直接使用特征向量推理 Q 值，用于局部墙状态穷举评估。
        """
        feature_arr = np.array(feature_state, dtype=np.float32)

        if getattr(self.robot, 'algorithm', '') == 'qtable':
            state_key = tuple(float(x) for x in feature_arr.tolist())
            self.robot.create_Qtable_line(state_key)
            return np.array([
                self.robot.q_table[state_key][a] for a in self.robot.valid_action
            ], dtype=float)

        target_model = getattr(self.robot, 'target_model', None)
        eval_model = getattr(self.robot, 'eval_model', None)

        if use_target_model and target_model is not None:
            model = target_model
        elif eval_model is not None:
            model = eval_model
        else:
            raise RuntimeError("Robot does not provide a DQN model for feature inference.")

        device = getattr(self.robot, 'device', 'cpu')
        state_tensor = torch.from_numpy(feature_arr).float().to(device)
        with torch.no_grad():
            return model(state_tensor).detach().cpu().numpy()

    def infer_dynamic_edge_probabilities(self, use_target_model=True):
        """
        反推动态关闭/打开候选边的概率。
        基于局部墙组合穷举计算 E[V|hit] 与 E[V|pass]，采用迭代式估计（EM思想）避免使用环境真实概率。
        """
        if not hasattr(self.maze, 'dynamic_close_doors') or not hasattr(self.maze, 'dynamic_open_doors'):
            return []

        if not hasattr(self.maze, 'base_maze_data'):
            raise RuntimeError("Dynamic maze is required for probability inference.")

        gamma = float(getattr(self.robot, 'gamma', 0.0))
        close_set = set(self.maze.dynamic_close_doors)
        open_set = set(self.maze.dynamic_open_doors)
        rows, cols, _ = self.maze.base_maze_data.shape

        dir_to_index = {'u': 0, 'r': 1, 'd': 2, 'l': 3}
        direction_order = ['u', 'r', 'd', 'l']

        # 初始化估计的边开通概率
        estimated_open_probs = {}
        for edge in close_set:
            estimated_open_probs[edge] = 0.5
        for edge in open_set:
            estimated_open_probs[edge] = 0.5

        def canonical_edge(loc, direction):
            i, j = loc
            if direction == 'u':
                if i == 0:
                    return None
                return (i - 1, j, 2)
            if direction == 'r':
                if j == cols - 1:
                    return None
                return (i, j, 1)
            if direction == 'd':
                if i == rows - 1:
                    return None
                return (i, j, 2)
            if direction == 'l':
                if j == 0:
                    return None
                return (i, j - 1, 1)
            raise ValueError("Unsupported direction: {}".format(direction))

        def edge_base_open(edge):
            if edge is None:
                return 0.0
            ei, ej, ez = edge
            return float(self.maze.base_maze_data[ei, ej, ez])

        def edge_open_probability(edge):
            if edge in estimated_open_probs:
                return estimated_open_probs[edge]
            return edge_base_open(edge)

        def build_feature(loc, open_map):
            feature = [float(loc[0]), float(loc[1])]
            for action in self.robot.valid_action:
                if action in open_map:
                    feature.append(float(open_map[action]))
                elif action == 's':
                    feature.append(1.0)
                else:
                    feature.append(0.0)
            return feature

        def expected_v_with_target_fixed(loc, target_edge, target_open_value):
            deterministic = {}
            uncertain = []

            for direction in direction_order:
                edge = canonical_edge(loc, direction)
                if edge is None:
                    deterministic[direction] = 0.0
                    continue

                if edge == target_edge:
                    deterministic[direction] = float(target_open_value)
                    continue

                p_open = edge_open_probability(edge)
                if p_open <= 0.0:
                    deterministic[direction] = 0.0
                elif p_open >= 1.0:
                    deterministic[direction] = 1.0
                else:
                    uncertain.append((direction, p_open))

            expected_v = 0.0
            num_uncertain = len(uncertain)
            for mask in range(1 << num_uncertain):
                open_map = dict(deterministic)
                prob = 1.0

                for bit in range(num_uncertain):
                    direction, p_open = uncertain[bit]
                    open_flag = (mask >> bit) & 1
                    open_map[direction] = float(open_flag)
                    prob *= p_open if open_flag else (1.0 - p_open)

                feature = build_feature(loc, open_map)
                q_values = self._q_values_from_feature(feature, use_target_model=use_target_model)
                expected_v += prob * float(np.max(q_values))

            return expected_v

        def q_sa_on_base_state(loc, action):
            open_map = {}
            for direction in direction_order:
                edge = canonical_edge(loc, direction)
                open_map[direction] = edge_base_open(edge)

            feature = build_feature(loc, open_map)
            q_values = self._q_values_from_feature(feature, use_target_model=use_target_model)
            action_index = self.robot.valid_action.index(action)
            return float(q_values[action_index]), feature

        def infer_one(edge, edge_type):
            action, s_state, next_state = self._edge_action_and_states(edge)

            q_sa, base_feature = q_sa_on_base_state(s_state, action)

            reward_hit = self._reward_with_shaping(next_loc=s_state, is_hit_wall=True)
            reward_pass = self._reward_with_shaping(next_loc=next_state, is_hit_wall=False)

            # 条件期望：hit 分支固定目标边关闭，pass 分支固定目标边打开
            expected_v_hit = expected_v_with_target_fixed(s_state, target_edge=edge, target_open_value=0.0)
            expected_v_pass = expected_v_with_target_fixed(next_state, target_edge=edge, target_open_value=1.0)

            g_hit = reward_hit + gamma * expected_v_hit
            g_pass = reward_pass + gamma * expected_v_pass

            eps = 1e-8
            if edge_type == 'close':
                # Q = p_close * G_hit + (1-p_close) * G_pass
                denom = g_hit - g_pass
                p_raw = (q_sa - g_pass) / denom if abs(denom) >= eps else np.nan
            else:
                # Q = p_open * G_pass + (1-p_open) * G_hit
                denom = g_pass - g_hit
                p_raw = (q_sa - g_hit) / denom if abs(denom) >= eps else np.nan

            valid = bool(np.isfinite(p_raw) and abs(denom) >= eps)
            p_hat = float(np.clip(p_raw, 0.0, 1.0)) if valid else np.nan

            return {
                'edge': edge,
                'edge_type': edge_type,
                'action': action,
                'state': s_state,
                'next_state': next_state,
                'base_feature': tuple(base_feature),
                'q_sa': q_sa,
                'reward_hit': reward_hit,
                'reward_pass': reward_pass,
                'expected_v_hit': expected_v_hit,
                'expected_v_pass': expected_v_pass,
                'g_hit': g_hit,
                'g_pass': g_pass,
                'denom': float(denom),
                'p_raw': float(p_raw) if np.isfinite(p_raw) else np.nan,
                'p_hat': p_hat,
                'valid': bool(valid),
            }

        max_iters = 10
        results = []
        for iteration in range(max_iters):
            results = []
            
            for edge in self.maze.dynamic_close_doors:
                res = infer_one(edge=edge, edge_type='close')
                results.append(res)
                if res['valid']:
                    new_p_open = 1.0 - res['p_hat']
                    # 平滑更新
                    estimated_open_probs[edge] = 0.5 * estimated_open_probs[edge] + 0.5 * new_p_open

            for edge in self.maze.dynamic_open_doors:
                res = infer_one(edge=edge, edge_type='open')
                results.append(res)
                if res['valid']:
                    new_p_open = res['p_hat']
                    # 平滑更新
                    estimated_open_probs[edge] = 0.5 * estimated_open_probs[edge] + 0.5 * new_p_open

        return results

    def save_dynamic_probability_image(self, filename, use_target_model=True, decimals=2):
        """
        反推候选边概率并保存带标注图片。
        """
        if not hasattr(self.maze, 'save_inferred_probabilities_image'):
            raise RuntimeError("Current maze does not support inferred probability rendering.")

        probability_items = self.infer_dynamic_edge_probabilities(use_target_model=use_target_model)
        self.maze.save_inferred_probabilities_image(
            probability_items=probability_items,
            save_path=filename,
            decimals=decimals,
            dpi=150,
        )
        return probability_items

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
        fig.clf()
        plt.close(fig)
        plt.close('all')
        import gc
        gc.collect()

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
            epoch_id = self._next_train_epoch_id + e
            total_return = 0
            run_times = 0
            for i in range(training_per_epoch):

                current_record = self._build_step_record(
                    epoch_id=epoch_id,
                    step_id=i,
                    state=self.maze.sense_robot(),
                    store_snapshot=False
                )

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
                    epoch_id=epoch_id,
                    filename=os.path.join(epoch_image_dir, "epoch_{:04d}.png".format(e))
                )

            self.robot.reset()

        self._next_train_epoch_id += int(training_epoch)

    def run_testing(self, testing_epoch=1, testing_per_epoch=None, epoch_image_dir=None):
        height, width, _ = self.maze.maze_data.shape
        if testing_per_epoch is None:
            testing_per_epoch = int(height * width * 0.85)

        for e in range(testing_epoch):
            self.robot.reset()
            epoch_id = self._next_test_epoch_id + e

            accumulated_reward = 0.
            run_times = 0

            for i in range(testing_per_epoch):
                current_record = self._build_step_record(
                    epoch_id=epoch_id,
                    step_id=i,
                    state=self.maze.sense_robot(),
                    store_snapshot=True
                )

                if current_record['state'] == self.maze.destination:
                    current_record['success'] = True
                    self.test_robot_record.append(current_record)
                    break

                action, reward = self.robot.test_update()
                current_record['action'] = action
                current_record['reward'] = reward
                current_record['maze_data'] = np.array(self.maze.maze_data, copy=True)
                self.test_robot_record.append(current_record)

                run_times += 1
                accumulated_reward += reward * (self.robot.gamma ** i)

            self.add_test_statics(accumulated_reward, run_times)

            if epoch_image_dir is not None:
                import os
                os.makedirs(epoch_image_dir, exist_ok=True)
                self.save_test_epoch_image(
                    epoch_id=epoch_id,
                    filename=os.path.join(epoch_image_dir, "test_epoch_{:04d}.png".format(epoch_id))
                )

        self._next_test_epoch_id += int(testing_epoch)

    def __init_gif(self):
        fig, ax = plt.subplots(figsize=(6, 6))
        return fig, ax

    def generate_gif(self, filename):
        fig, ax = self.__init_gif()
        p_bar = tqdm(
            total=len(self.test_robot_record),
            desc="正在将测试过程转换为gif图, 请耐心等候...",
        )

        maze_backup = np.array(self.maze.maze_data, copy=True)

        def update(record):
            ax.clear()

            if 'maze_data' in record:
                self.maze.maze_data = np.array(record['maze_data'], copy=True)

            self.maze.draw_maze()

            x, y = record['state'][0] + 0.5, record['state'][1] + 0.5
            robot = plt.Circle((y, x), 0.4, color="red")
            ax.add_patch(robot)

            ax.text(
                0,
                -0.1,
                "epoch:" + str(record['id'][0]),
                fontsize=20,
                horizontalalignment='left',
                verticalalignment="bottom"
            )
            ax.text(
                self.maze.maze_size,
                -0.1,
                "step:" + str(record['id'][1]),
                fontsize=20,
                horizontalalignment='right',
                verticalalignment="bottom",
            )

            p_bar.update(1)
            return []

        def init(): pass  # do nothing


        import matplotlib.animation as animation
        ani = animation.FuncAnimation(
            fig,
            update,
            init_func=init,
            frames=self.test_robot_record,
            interval=200,
            blit=False,
        )

        # To save the animation, use e.g.
        ani.save(filename, writer='pillow')
        self.maze.maze_data = maze_backup
        fig.clf()
        plt.close(fig)
        plt.close('all')
        import gc
        gc.collect()

    def plot_results(self):
        plt.figure(figsize=(12, 4))
        plt.subplot(131)
        plt.title("Success Times")
        plt.plot(np.cumsum(self.train_robot_statics['success']))
        plt.subplot(132)
        plt.title("Total Return")
        plt.plot(np.array(self.train_robot_statics['reward']))
        plt.subplot(133)
        plt.title("Runing Times per Epoch")
        plt.plot(np.array(self.train_robot_statics['times']))
        plt.show()
