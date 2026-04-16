1. 添加路径渐变显示
2. 删除Robot类和ReplayDataSet类中is_terminal=(next_state==state)的逻辑错误
3. 修复ReplayDataSet中的经验更新逻辑：允许新经验覆盖旧经验，避免因 `build_full_view` 初始化占位后，后续真实训练的动态奖励和探索数据被直接抛弃。
4. 网络决策逻辑对齐：明确 `MinDQNRobot` 采用的是**代价最小化**（`argmin`和`min`获取目标Q），而 `Robot` 采用的是**奖励最大化**（`argmax`和`max`），对应的墙体惩罚与空地衰减奖励符号需要严格相反。