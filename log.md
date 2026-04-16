# 强化学习动态环境导航与动力学逆向推导开发记录

## 1. 初始设定与算法瓶颈分析

本项目初始阶段基于完全可观测、静态且确定性的网格迷宫环境。在测试 Deep Q-Network (DQN) 算法表现时，发现其探索效率显著低于传统的启发式搜索算法（如 A*）。

**数学原理分析：**
在确定性环境（转移概率 $P(s'|s,a) = 1$）中，A* 算法借助封闭集（Closed Set）机制，保证任意节点最多被扩展一次，时间复杂度上界严格约束为 $O(|S| + |E|)$。而强化学习（RL）在缺乏先验知识的探索阶段退化为简单随机游走（Simple Random Walk）。在 $N$ 个节点的二维网格中，到达目标点的期望步数（Hitting Time）为：
$$E[H(u,v)] = O(N \log N)$$
且根据 PAC-MDP 理论，Q-Learning 在确定性环境中收敛到近似最优策略的样本复杂度下界为 $\tilde{O}\left( \frac{|S||A|H^3}{\epsilon^2} \right)$。这证明了在静态已知环境中，使用 RL 存在极大的算力冗余和效率劣势。

## 2. 环境重构：引入马尔可夫动态不确定性

为突显 RL 拟合环境期望状态的优势，将环境重构为具有随机转移概率的动态迷宫（Dynamic Maze）。

**关键决策：**
1. 废弃随机关闭任意网格的设计（会导致环境方差过大，破坏收敛性）。
2. 在 Prim 算法生成的基础拓扑上，选取固定数量的“通道”赋予概率 $p_{close}$ 临时关闭，选取固定数量的“墙壁”赋予概率 $p_{open}$ 临时打开。
3. 状态空间扩维：将状态从二维坐标 $(x, y)$ 扩充为六维向量 $s = (x, y, w_u, w_r, w_d, w_l)$，并将“原地等待（'s'）”加入动作空间。环境严格符合马尔可夫决策过程（MDP）。

在此环境下，智能体通过最大化贝尔曼最优方程，自发学会了在“高风险短距离”与“低风险长距离”之间进行期望价值的权衡：
$$Q^*(s, a) = \mathbb{E}_{s', r \sim P} \left[ r + \gamma \max_{a'} Q^*(s', a') \right]$$

## 3. 逆向强化学习：环境概率分布的精确推导

在确立智能体能够拟合环境期望后，提出通过收敛的网络权重反推环境隐藏动力学参数（动态墙壁开闭概率 $p$）的逆向推导方法。

**数学推导过程：**
设机器人处于状态 $s = (L, W)$，试图穿过某扇动态门。该门关闭的概率为 $p$，打开概率为 $1-p$。动作 $a$ 对应的转移分支如下：
* **分支 A (门关闭)**：概率 $p$，即时奖励 $R_{hit}$，留在原地 $L$。
* **分支 B (门打开)**：概率 $1-p$，即时奖励 $R_{pass}$，到达新坐标 $L_{next}$。

定义局部状态的最优期望价值：
$$\bar{V}_{hit} = \mathbb{E}_{W'} \left[ \max_{a'} Q((L, W'), a') \right]$$
$$\bar{V}_{pass} = \mathbb{E}_{W'} \left[ \max_{a''} Q((L_{next}, W'), a'') \right]$$
定义两个分支的总期望代价：
$$Q_{hit} = R_{hit} + \gamma \bar{V}_{hit}$$
$$Q_{pass} = R_{pass} + \gamma \bar{V}_{pass}$$

根据贝尔曼方程展开当前动作的 Q 值：
$$Q(s, a) = p \cdot Q_{hit} + (1 - p) \cdot Q_{pass}$$
展开并移项：
$$Q(s, a) = p \cdot Q_{hit} + Q_{pass} - p \cdot Q_{pass}$$
$$p \cdot (Q_{pass} - Q_{hit}) = Q_{pass} - Q(s, a)$$
最终得到逆推概率的严谨代数解析式：
$$p = \frac{Q_{pass} - Q(s, a)}{Q_{pass} - Q_{hit}}$$

## 4. 算法降维与期望最大化 (EM) 的引入

**工程挑战：**
在使用 DQN 验证上述解析式时，由于神经网络存在函数逼近误差（Approximation Error），当 $Q_{pass} - Q_{hit} \approx 0$（即病态计算分母趋零）时，微小的网络误差被极度放大，导致推导出的概率失效。同时，计算 $\bar{V}$ 需要遍历周围墙壁的状态组合，如果直接调用环境底层概率则构成了逻辑上的“作弊”。

**关键决策：**
1.  **降维至 Tabular Q-Learning**：放弃 DQN，使用 Q-table 进行千万步级别的纯随机探索（$\epsilon=1.0$），彻底消除函数逼近误差，获取无偏经验价值 $\hat{Q}$。
2.  **引入 EM 算法消除透视**：
    * **E-step (期望步)**：使用当前迭代的隐变量（环境中其他墙壁的概率分布估计 $\hat{P}^{(t)}$）去计算 $\bar{V}_{hit}$ 和 $\bar{V}_{pass}$。
    * **M-step (最大化步)**：将计算出的 $Q_{hit}$ 和 $Q_{pass}$ 代入逆推解析式，更新所有动态墙壁的概率估计值 $\hat{P}^{(t+1)}$。
    交替迭代直至收敛，实现了纯黑盒条件下的环境动力学解构。

## 5. 逆推概率的统计学特性解析

测试结果表明：单个墙壁推导概率方差较大，但全局概率均值高度逼近真实值。

**误差泰勒展开与方差解构：**
设 Q-table 的采样噪声为 $\delta$，真实的价值差 $\Delta Q = Q^*_{pass} - Q^*_{hit}$。单扇门 $i$ 的经验概率 $\hat{p}_i$ 表达为：
$$\hat{p}_i = \frac{Q^*_{pass} + \delta_{pass} - (Q^*(s, a) + \delta_Q)}{Q^*_{pass} + \delta_{pass} - (Q^*_{hit} + \delta_{hit})}$$
代入 $Q^*(s, a) = p Q^*_{hit} + (1-p) Q^*_{pass}$ 并提取 $\Delta Q$：
$$\hat{p}_i = \frac{p \Delta Q + \delta_{pass} - \delta_Q}{\Delta Q + \delta_{pass} - \delta_{hit}} = \left( p + \frac{\delta_{pass} - \delta_Q}{\Delta Q} \right) \left( 1 + \frac{\delta_{pass} - \delta_{hit}}{\Delta Q} \right)^{-1}$$
利用一阶泰勒展开 $(1+x)^{-1} \approx 1-x$（假设 $\delta \ll \Delta Q$），忽略高阶无穷小：
$$\hat{p}_i \approx \left( p + \frac{\delta_{pass} - \delta_Q}{\Delta Q} \right) \left( 1 - \frac{\delta_{pass} - \delta_{hit}}{\Delta Q} \right)$$
$$\hat{p}_i \approx p + \frac{(1-p)\delta_{pass} - \delta_Q + p\delta_{hit}}{\Delta Q_i}$$
**结论 1（高方差原因）**：单次估计的误差项分母为 $\Delta Q_i$。对于通过与否期望回报接近的网格（$\Delta Q_i \to 0$），任何微小的采样噪声 $\delta$ 都会导致 $\hat{p}_i$ 产生剧烈波动。

**均值收敛性证明：**
由于 Q-learning 在充分探索下满足 Robbins-Monro 条件，采样噪声无偏，即 $\mathbb{E}[\delta] = 0$。因此单点误差期望为零：
$$\mathbb{E}[\epsilon_i] = \frac{(1-p)\mathbb{E}[\delta_{pass}] - \mathbb{E}[\delta_Q] + p\mathbb{E}[\delta_{hit}]}{\Delta Q_i} = 0$$
对于 $N$ 扇动态墙壁的均值估计 $\hat{P}_{mean}$，根据大数定律（LLN）：
$$\text{Var}(\hat{P}_{mean}) = \text{Var} \left( \frac{1}{N} \sum_{i=1}^N \epsilon_i \right) = \frac{\sigma^2}{N}$$
**结论 2（均值准确性）**：随着取样墙壁数量 $N$ 的增加，均值估计的方差以 $O(1/N)$ 的速度严格收敛至 0，无偏噪声在求和中相互抵消，全局宏观概率精准收敛。这从数学上印证了逆向推导系统构成了“高方差、低偏差”的有效统计估计器。

