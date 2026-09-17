# 计算量压缩的实测后续

正式内核仍是 **918 cycles / 10,811 个静态 VLIW 包**，目标仍为
**<900 cycles / ≤12,000 包**。以下实验没有超过正式版本。
`validation.json` 记录了默认图摘要，以及重新生成的完整程序与实际
`KernelBuilder.instrs` 逐项相等的检查。

## 常量生成转到 LOAD

新增 `force_load_scalars`，将指定常量的 ALU 加减生成改为原 ISA 的
`LOAD const`。`search_constant_loads.py` 按早晚、后继长度、使用次数和固定
随机顺序选择常量，同时冻结已有 hash 的标量/向量选择后再做精确再平衡。
50 个配置先通过参考语义检查，再搜索可分配排程，最好仍为 918。

保存的 `constant_loads_918/candidate_014` 迁移了 48 个常量，净变化是
**−16 ALU、−4 VALU、+48 LOAD**，所以 W 从 54,723 降至 **54,675**。
它保持 918 cycles / 10,811 包；静态 slot 数增至 232,190，压缩载荷为
161,292 字节。W 减少不代表静态体积也减少，因此没有替换正式入口。
该图的资源窗口下界为 **914**，这是下界，不是已找到的 914 排程。

原 918 排程在前 100 cycles 有 56 个 LOAD 空槽。另用
`constant_load_priority` 提前安排新增常量加载，12 个配置仍以 918 最好；
六组按实测空槽迁移 hash 操作的后续也全部停在 918。

## 有界排程修复

`repair_schedule.py` 用 CP-SAT 表达现有依赖、复合 dispatch、各引擎容量
及 dispatch 互斥，并限制每个排程单元相对保存排程的移动窗口。
可选 `--lane-order` 进一步保留各标量 hash 组中的原 lane 顺序。
这是一种受限搜索，不是全局优化性证明。

- 918、窗口 0 的正例在 0.05 秒内通过，重现完整保存排程，分配通过。
  求解器的 `OPTIMAL` 在这个无优化目标的可行性问题中不表示 918 全局最优。
- 917、窗口 ±4、60 秒：`UNKNOWN`。
- 917、窗口 ±8、保留 lane 顺序、120 秒：`UNKNOWN`。
- 916、窗口 ±4、保留 lane 顺序、60 秒：`UNKNOWN`。

超时不证明不可达，也没有产生可晋升的新排程。工具使用现有
`.solver-venv` 中的 OR-Tools 9.15.6755。

## 孙节点缓存与原始树保留的交换

`cache_tradeoff_918/run.py` 试验减少孙节点副本、保留更多原始树块以减少
备份读取，并将原先的 MADD 节点选择转回 FLOW。24 个配置中最好为
942 cycles；两项未保留可分配候选，汇总中的 100000 是哨兵值。

全部取消孙节点副本并保留深度 4–7 原始块的 candidate 017，W 降至
**54,211（−512）**，但冻结执行为 **967 cycles**。其 LOAD 子集有
1,562 条操作，最早发射时间至少 68，必要尾部至少 67：

```text
cycles ≥ 68 + 67 + ceil(1,562 / 2) = 916
```

因此，这张改写图即使理想重排也不能进入 900；保留原始树减少的恢复
读取无法消除这批更晚才能发射的节点读取。这个结论只覆盖该固定图。
部分保留孙节点的 candidate 015 下界为 909，实际验证成绩为 942，
不能据此声称其他缓存布局也不可行。

## 更浅的 dispatch 和两轮地址折叠

新增 `dispatch2_groups` / `dispatch13_groups`，在深度 2 使用三组并行
dispatch，准备深度 3 的两个子节点，再于深度 4 直接读取节点。
新配置沿用既有 STORE 缓冲、真实 ISA 和通用 case 展开器。

`fold_path3_groups` 延迟中间 q3 坐标。在镜像布局下，深度 4 地址直接为：

```text
16 + 4*q2 + 2*b2 + b3
```

两轮 MADD 更新和一次选择可以改为一次 MADD 和三次选择。六组的成对
对照少 48 W 的更新，新增常量准备花费 8 W，净省 **40 W**，增加
**12 FLOW**。相应成绩从 920 变为 923，展示了算术削减被其他端口与
依赖抵消的情况。所有 24 个筛选配置先通过种子 0、1 的语义检查。

## 冻结执行验证

下表每个候选均通过种子 0–9、每种子 20,480 个 hash 阶段检查点、完整
PC 到逻辑周期映射、最终输出、非输出内存保存和独立展开逐项相等检查。

| 保存位置 | cycles | 静态包 | W |
| --- | ---: | ---: | ---: |
| `constant_loads_918/candidate_014` | 918 | 10,811 | 54,675 |
| `cache_tradeoff_918/candidate_015` | 942 | 10,835 | 54,427 |
| `cache_tradeoff_918/candidate_017` | 967 | 10,860 | 54,211 |
| `depth2_918/candidate_016` | 920 | 10,048 | 54,848 |
| `depth2_918/candidate_018` | 923 | 10,051 | 54,808 |
| `depth2_918/candidate_006` | 1,011 | 10,139 | 54,680 |

分配、排程和缓冲区内存顺序的 12 项回归测试也全部通过。
`tests/`、`problem.py`、ISA、周期计数和正式内核均未改动。

## 复现

```sh
python3 search_constant_loads.py results/compact_918 results/constant_loads_918
PYTHONPATH=. python3 results/constant_preload_918/run.py
python3 rebalance_holes.py results/constant_loads_918/candidate_014 results/constant_holes_918 \
  --moves '[[0,8],[0,16],[8,0],[16,0],[8,16],[16,24]]'
PYTHONPATH=. python3 results/cache_tradeoff_918/run.py
python3 search_depth2.py results/compact_918 results/depth2_918
.solver-venv/bin/python repair_schedule.py results/constant_loads_918/candidate_014 \
  results/constant_repair_918/target917_w8_order --cycles 917 --window 8 --seconds 120 --lane-order
python3 export_candidate.py results/constant_loads_918/candidate_014
python3 -m unittest test_lane_allocate test_schedule test_memory_order -v
```

可继续使用减少 48 W 的 918 图作为独立起点。此次证据要求后续改写
同时检查新增 LOAD 的发射窗口及 FLOW 的使用时机；总计数下降本身不足以
承诺周期收益。还没有找到满足 <900 的完整改写。
