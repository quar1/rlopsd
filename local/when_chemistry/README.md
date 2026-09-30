# 科学问答数据转换与奖励

目录名保留自早期化学实验，`reward.py` 已支持 chemistry、biology、physics、materials 四科。最后一个完整 `<answer>` 标签中的正确选项奖励1，否则0。

`prepare_data.py --source <原始固定化学划分目录> --output <新输出目录>` 转换既有划分，不重新划分；已有输出目录会拒绝覆盖。

当前任务启动、数据位置及训练流程参见仓库根目录 README.md；默认数据为 `data/{chemistry,biology,physics,materials}`。不再使用 `when_` 数据目录前缀。
