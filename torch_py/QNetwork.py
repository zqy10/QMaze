from abc import ABC
import torch.nn as nn
import torch

class QNetwork(nn.Module, ABC):
    """Actor (Policy) Model."""

    def __init__(self, state_size: int, action_size: int, seed: int):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state (e.g., 2 for x,y coordinates)
            action_size (int): Dimension of each action (e.g., 4 for directions)
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        # 将宽度从 512 缩减到 64，极大提升收敛速度和稳定性
        hidden_size = 256
        
        self.input_hidden = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        self.final_fc = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = self.input_hidden(state)
        return self.final_fc(x)

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = QNetwork(2, 4, 0).to(device)

    # 模拟一个归一化后的状态输入 (例如 5x5 迷宫中的位置 [1, 1])
    # 强烈建议在外部 Agent 喂给网络数据前，加上这步归一化操作
    x = torch.tensor([1.0 / 5.0, 1.0 / 5.0]).float().unsqueeze(0).to(device)
    
    print(net(x))