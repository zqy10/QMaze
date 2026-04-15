import random
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from QRobot import QRobot
from ReplayDataSet import ReplayDataSet
from torch_py.QNetwork import QNetwork

class Robot(QRobot):
    valid_action = ['u', 'r', 'd', 'l']

    def __init__(self, maze):
        super(Robot, self).__init__(maze)
        self.maze = maze
        self.valid_action = list(maze.valid_actions)

        self.maze.set_reward(reward={
            "hit_wall": -12.0,
            "destination": 150.0,
            "default": -2.0,
        })

        # 算法参数设置
        epsilon0 = 0.6
        self.epsilon0 = epsilon0
        self.epsilon = epsilon0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.001
        self.gamma = 0.97
        self.learning_rate = 1e-3
        self.batch_size = 64
        self.distance_weight = 4.0
        self.distance_metric = "euclidean"  # options: manhattan, euclidean
        
        self.target_update_interval = self.maze.maze_size * 2 - 3
        self.step_counter = 0

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # 初始化评估网络与目标网络，状态维度=2(坐标)+动作可行性特征维度
        self.state_size = 2 + len(self.valid_action)
        self.eval_model = QNetwork(state_size=self.state_size, action_size=len(self.valid_action), seed=21).to(self.device)
        self.target_model = QNetwork(state_size=self.state_size, action_size=len(self.valid_action), seed=21).to(self.device)
        self.target_model.load_state_dict(self.eval_model.state_dict())
        self.target_model.eval() # 目标网络只做前向传播

        self.optimizer = optim.Adam(self.eval_model.parameters(), lr=self.learning_rate)

        # 初始化经验回放池
        max_size = max(self.maze.maze_size ** 2 * 3, 10000)
        self.memory = ReplayDataSet(max_size=max_size)

    def reset(self):
        """
        重置机器人并清空单回合访问次数统计。
        """
        super(Robot, self).reset()

    def get_state_feature(self, loc):
        """将当前坐标及四周墙壁状态合并为增强状态特征向量。通畅为1，阻挡为0"""
        valid_moves = self.maze.can_move_actions(loc)
        walls = [1.0 if a in valid_moves else 0.0 for a in self.valid_action]
        return tuple(list(loc) + walls)

    def _choose_action(self, loc, state_feature, is_train=True):
        state_arr = np.array(state_feature)
        state_tensor = torch.from_numpy(state_arr).float().to(self.device)
        
        # 训练时使用 epsilon-greedy 策略增加探索
        if is_train:
            if random.random() < self.epsilon:
                # 仅在合法动作中探索，减少无效撞墙行为
                valid_moves = self.maze.can_move_actions(loc)
                # 极端情况下如果被四面墙封死（动态迷宫），退回使用所有动作
                if not valid_moves:
                    valid_moves = self.valid_action
                return random.choice(valid_moves)

            self.eval_model.eval()
            with torch.no_grad():
                q_values = self.eval_model(state_tensor).cpu().data.numpy()
            self.eval_model.train()

        else:
            with torch.no_grad():
                q_values = self.target_model(state_tensor).cpu().data.numpy()

        return self.valid_action[np.argmax(q_values).item()]

    def _learn(self):
        # 只有在 Replay Buffer 里的数据足够时才开始训练
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, is_terminals = self.memory.random_sample(self.batch_size)

        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).long().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        is_terminals = torch.from_numpy(is_terminals).int().to(self.device)

        Q_targets_next = self.target_model(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - is_terminals)

        # 获取当前模型评估的 Q 值
        Q_expected = self.eval_model(states).gather(dim=1, index=actions)

        # 反向传播更新 eval_model
        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 4. 延迟硬更新 Target 网络
        if self.step_counter % self.target_update_interval == 0:
            self.target_model.load_state_dict(self.eval_model.state_dict())

    def train_update(self):
        loc = self.sense_state()
        state_feature = self.get_state_feature(loc)
        action = self._choose_action(loc, state_feature, is_train=True)
        
        # 1. 获取原始代价
        reward = self.maze.move_robot(action)
        next_loc = self.sense_state()
        next_state_feature = self.get_state_feature(next_loc)

        # 关键修复：只有到达终点才终止，撞墙不应终止回合
        is_terminal = 1 if next_loc == self.maze.destination else 0
        
        # 2. 加入归一化后的距离代价（支持曼哈顿/欧氏距离）
        if not is_terminal:
            dest_x, dest_y = self.maze.destination
            curr_x, curr_y = next_loc

            if self.distance_metric == "euclidean":
                dist = np.sqrt((curr_x - dest_x) ** 2 + (curr_y - dest_y) ** 2)
                max_dist = np.sqrt(2) * max(1, (self.maze.maze_size - 1))
            else:
                # 默认使用曼哈顿距离，兼容旧逻辑
                dist = abs(curr_x - dest_x) + abs(curr_y - dest_y)
                max_dist = max(1, 2 * (self.maze.maze_size - 1))
            
            # 归一化距离 (取值范围 0.0 ~ 1.0)
            normalized_dist = dist / max_dist
            
            # 最大化回报时，距离越远应减分
            reward = reward - normalized_dist * self.distance_weight
        
        # 3. 存入经验池
        self.memory.add(state_feature, self.valid_action.index(action), reward, next_state_feature, is_terminal)

        # 触发学习
        self._learn()

        self.step_counter += 1
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return action, reward

    def test_update(self):
        # 测试阶段不再探索，直接根据评估网络选取动作
        loc = self.sense_state()
        state_feature = self.get_state_feature(loc)
        action = self._choose_action(loc, state_feature, is_train=False)
        reward = self.maze.move_robot(action)
        return action, reward