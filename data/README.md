# 数学数据固定划分

这里是现有实验直接使用的准备后数据，不是重新生成的随机划分。发布时 parquet 文件逐一核对 SHA256，保持题目、顺序、标签和参考解不变；manifest 将旧服务器绝对路径改成可移植的来源标识，并记录本次师生及评测 thinking 关闭后的完整输入长度核验。

| 文件 | 题数 | 用途 / 上游 |
|---|---:|---|
| math/train_teacher.parquet | 28,519 | 训练；[siyanzhao/Openthoughts_math_30k_opsd](https://huggingface.co/datasets/siyanzhao/Openthoughts_math_30k_opsd) |
| math/train_plain.parquet | 28,519 | 相同训练 ID，不含教师特权参考解 |
| math/test.parquet | 256 | 固定开发集；默认不运行内置开发评测 |
| math/benchmarks/aime24.parquet | 30 | [HuggingFaceH4/aime_2024](https://huggingface.co/datasets/HuggingFaceH4/aime_2024) |
| math/benchmarks/aime25.parquet | 30 | [yentinglin/aime_2025](https://huggingface.co/datasets/yentinglin/aime_2025) |
| math/benchmarks/hmmt25.parquet | 30 | [MathArena/hmmt_feb_2025](https://huggingface.co/datasets/MathArena/hmmt_feb_2025) |

原始数学池 29,434 条；训练/开发共 28,775 条。此前按空字段、冲突答案、重复题和标签可解析性过滤，具体数量见 `math/manifest.json` 与 `math/rejected.json`。训练与开发 ID 见 `math/split_ids.json`；随机种子42，loader 按存储顺序遍历，不重排。

教师参考解来自原始 `solution` 字段，仅进入教师输入；参考解不作为学生直接 CE 监督。测试标签只用于评测判分，不参与候选选择或学生更新。

重叠检查使用归一化精确匹配/字符5-gram Jaccard 阈值0.8，不代表已排除语义改写污染。可解析标签也不等于已独立验证其数学正确性。上游数据及问题内容仍按各自许可/条款使用，仓库代码许可不替代数据许可。
