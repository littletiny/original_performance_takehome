# 优化尝试总账与手写设计评估

整理日期：2026-09-18。主线基准：`codex/sub900`，提交 `816687e`。本文只整理证据和评估设计，不改动内核，也不启动新一轮搜索。

**当前已验证成绩是 913 cycles、10,537 个静态 VLIW 包，尚未达到 <900。** 如果停止深度数值搜索，合理路线是保留已经验证的 hash 恒等式，重点设计分层取节点、路径表示、跨组流水和 scratch 生命周期，再用少量明确的指令模板实现。现有图仅靠手工重排不能进入 900；必须改变工作量或指令实现，但不必把突破口限定为更复杂的 hash 代数。

## 1. 范围、指标与证据

本次合并了三段历史：

1. 本仓库 980 以前的实现及复盘，见 [RETROSPECTIVE.md](RETROSPECTIVE.md)、[OPT_NOTES.md](OPT_NOTES.md)、[OPTIMIZATION_GUIDE.md](OPTIMIZATION_GUIDE.md)。其中“当前”“不可行”等表述属于当时阶段，本文会纠正被后续实验推翻的结论。
2. 980 → 913 主线及全部现存 `results/` 实验目录，见 [SUB900_WORKLOG.md](SUB900_WORKLOG.md)。附录逐目录登记，包含未晋升、只有筛选记录及没有冻结成绩的尝试。
3. 参考仓库的独立 950 路线，以及本仓库 `codex/flowfold` 旁支。参考仓库见 [完整历史](../original_performance_takehome/docs/OPTIMIZATION_HISTORY.md)；旁支检查到 `58439d4`。这些结果不是本主线的直接父子链。

“所有尝试”按**设计方向和已保存实验批次**归并；不把同一方向的每个随机种子、调度迭代或重复快照分别写成新方法。没有保存完整验证结果的原型会明确保留为原型，不补造实验结论。参考仓库的细分历史已由其总账归档，本文提取与本次决策有关的成功、反例和分支关系，不重复其全部文件清单。

固定负载为树高 10、2,047 个节点、256 条输入、16 轮；8 lane 构成一组，共 32 组。访问深度为 `0…10, 0…4`。树值、输入值、路径均依赖运行时数据。每周期容量是 ALU 12、VALU 6、LOAD 2、STORE 2、FLOW 1，scratch 为 1,536 字。内存操作在此模型中没有额外 cache-miss 延迟；预取的作用是提前消耗有限 LOAD 槽并缩短依赖等待。

| 名称 | 本文口径 | 不能替代的指标 |
| --- | --- | --- |
| cycles | 冻结机器实际执行的非 debug 包数 | 静态程序长度、求解器目标 |
| 静态包数 | 完整展开后的 `len(instrs)`，含未执行 case 和 padding | 压缩源码字节、逻辑排程长度 |
| 静态 slot 数 | 所有静态包内 engine 指令总数 | 静态包数 |
| 动态 slot 数 | 一次实际执行发射的 engine 指令总数 | cycles |
| `W = A + 8V` | ALU/VALU 的加权资源量，每周期最多 60 W | 语义计算量、全算法复杂度；一条 MADD 也按 8 W 计 |
| 计数下界 | 只看总量和端口容量的必要条件 | 可执行排程 |
| 资源窗口下界 | 再计最早就绪与必要尾部，形如 `release + tail + ceil(work/capacity)` | 全局算法下界；当前实现还忽略 scratch 约束 |

用户的静态限制先为 500k，后收紧到 10,000，最后放宽为 **12,000 包**。比较历史结果时须注明当时限制。源码的压缩大小不构成指令包压缩。

证据分五档：**正式提交且验证通过**；**独立冻结执行通过**；**已分配的筛选排程**；**语义图/未分配排程/下界**；**仅原型或设计提议**。求解器的模板 UNSAT 另作局部排除证据；UNKNOWN、超时和搜索失败均不证明不可行。若正文写“验证”，指对应报告的冻结执行，而非仅参考解释器通过。

## 2. 当前基准：后续判断统一从这里出发

完整证据位于 [compact_913_packed](results/compact_913_packed/README.md)，机器结果见 [verification.json](results/compact_913_packed/verification.json)，当前源码绑定见 [validation.json](results/compact_913_packed/validation.json)。

| 项目 | 当前值 |
| --- | ---: |
| 动态 cycles | **913** |
| 静态包 / 上限 | **10,537 / 12,000** |
| 静态包构成 | 553 主体 + 9,600 case + 384 padding |
| 静态 / 动态 slot 数 | 231,523 / 19,955 |
| 压缩载荷 | 162,359 bytes |
| W | **54,406** |
| 计数算术下界 / 固定图窗口下界 | **907 / 909** |
| scratch 地址跨度 / 实际存活峰值 | 1,536 / 1,322 字 |
| 当前排程与窗口下界的差 | 4 cycles；不是已证明的必需开销 |

| 端口 | 动态使用量 | 913 周期利用率 | 899 周期总容量相对现状 |
| --- | ---: | ---: | ---: |
| ALU | 10,870 | 99.22% | 超出 82 槽 |
| VALU | 5,442 | 99.34% | 超出 48 槽 |
| LOAD | 1,778 | 97.37% | 余 20 槽 |
| STORE | 984 | 53.89% | 余 814 槽 |
| FLOW | 881 | 96.50% | 余 18 槽 |

这些余量只代表**全程总容量**，不是任意时刻可用的空槽。FLOW 的图计数为 880，已经包含 pause，未包含 bootstrap；表中采用包含二者的实际执行计数。正式 [port util](results/compact_913_packed/port_utilization.png) 和 [原始统计](results/compact_913_packed/port_utilization.json) 与源码绑定，不能用静态 PC 代替执行周期。

当前已有 9 项提交测试、3 项 native 测试和 38 项研究回归通过；种子 0–9 与实际 builder 的 901、12345 均通过每种子 20,480 个保留 hash 检查点、输出、非输出内存和执行 PC 映射检查。完整实际 builder、独立解码器和验证过的 lowerer 一致。这是既有验证记录，本次文档更新不声称重新运行了这些测试。

源码 SHA-256：`394deefe4521566b2a02b2147b35f8b7536e4da7724c6b8a872deb144cf23c7e`。静态大小采用实际展开数据；某些 `bounds.json` 中不带排程的 padding 估计不能替代它。

## 3. 成绩演进：成功不只有一种

### 3.1 980 以前的本仓库历史

以下是旧复盘记录的代表成绩，验证标准依所属阶段，不倒推为今天的 12-seed、20,480 检查点标准。

| 阶段 | 当时结果 | 成功或教训 |
| --- | ---: | --- |
| 标量 → SIMD/VLIW | 147,734 → 约 1,182 | 大部分收益来自 8 lane、展开和独立输入交错 |
| 标量份额、地址、常数调整 | 1,185 → 1,163 → 1,157 | 可利用 ALU，但固定比例很快遇到局部平台 |
| 虚拟寄存器、分配器 | 1,160 | 单独略慢；为后续节点选择和临时量复用提供实现空间 |
| 浅层 tournament | 1,151 | 比逐叶线性扫描减少条件构造；受其他瓶颈限制，周期收益小于计数收益 |
| 扩展节点预 XOR | 1,169 | 单独变慢，但表示后来有组合价值 |
| 深度 4 部分缓存选择 + 再平衡 | 1,085 → 1,069 | 把 gather 压力转移至 FLOW/VALU，是结构性收益 |
| MADD、常数去重、死路径更新清理 | 动态生成版本约 1,057 | 删除工作，为后续排程提供更好图 |
| 正反向交替排程、离线嵌入 | 986 | 曾有同图约 83 cycles 改善；不能把启发式平台当硬下界 |
| 合并 pause、入口折叠、选择性常数合成 | 985 → **980** | 980 为本次 sub-900 工作的起点 |

原始依据：[旧复盘](RETROSPECTIVE.md)。959 参考文档和旧日志中的排行榜数字是历史外部参考，本文不把它们当作本仓库已复现成绩，也不重新背书旧外部调查的覆盖范围。

### 3.2 本次 980 → 913 主线

| 检查点 | cycles | 静态包 | 主要变化 / 结果性质 |
| --- | ---: | ---: | --- |
| [portfolio_1](results/portfolio_1/verification.json)，`e2c302f` | 932 | 477,092 | 新鲜 SSA、dense dispatch；符合旧 500k，体积仍过大 |
| [compact_971](results/compact_971/verification.json)，`8b4ec3a` | 971 | 9,819 | 压缩配对及查表范围，牺牲部分速度满足 10k |
| [compact_939](results/compact_939/verification.json)，`5954122` | 939 | 9,787 | 已有 handler 中准备深度 5 孙节点，避免新增大表 |
| [compact_936](results/compact_936/verification.json)，`1d667e6` | 936 | 9,784 | 地址链、共享常量等组合 |
| [compact_934](results/compact_934/verification.json)，`188b459` | 934 | 9,782 | 镜像树、及时备份、lane 生命周期分配、浅层共享 |
| [compact_928](results/compact_928/verification.json)，`5b10cbe` | 928 | 9,776 | 邻域改写及 ALU/VALU 再平衡；10k 限制期间的正式检查点 |
| [compact_919](results/compact_919/verification.json)，`9c32b1e` | 919 | 10,159 | 12k 允许更多配对；短链、I/O 备份、地址复用与再平衡组合 |
| [compact_918](results/compact_918/verification.json)，`e9da622` | 918 | 10,811 | 固定 PC 表与已有地址常数共用，W=54,723 |
| [compact_917](results/compact_917/README.md)，`f954339` | 917 | 10,810 | PC 最后一位折叠、header root、raw 节点及常数 LOAD，W=54,652 |
| [compact_915](results/compact_915/README.md)，`4d45354` | 915 | 10,808 | 部分内存广播、深层 lane-offset overfetch，W=54,577 |
| [compact_914](results/compact_914/README.md)，`2dd2238` | 914 | 10,807 | 按已有时序改常数表达式、早期地址前缀、FLOW 路径更新，W=54,536 |
| [compact_914_pc](results/compact_914_pc/README.md)，`2437e53` | 914 | 10,807 | PC offset MADD、额外内存广播，W 降至 54,484；同分减工作 |
| [compact_913](results/compact_913/README.md) | 913 | 11,049 | 交错 PC 常数、短前缀、实测 FLOW/LOAD 空槽，W=54,406 |
| [compact_913_packed](results/compact_913_packed/README.md)，`816687e` | **913** | **10,537** | 同时序和端口计数下填补表洞，减少 512 个 padding 包 |

各行常包含多个交互变换，不能将相邻分数差全部归给某一个开关。932 → 971 是**静态体积成功**；914 → 914_pc 是**工作量成功**；最终 913 的 packing 是**静态布局成功**。这些都应保留，但与动态周期成功分别记录。

此次全目录盘点还检出了未晋升的本地 [exact_balance/candidate_032](results/exact_balance/candidate_032/verification.json)：保存记录为 **926 cycles / 9,774 包**，10-seed 冻结通过。因此，928 是旧 10k 阶段保留的正式入口，不应写成所有现存 ≤10k 候选的最低成绩。本次没有重新执行这个历史候选。

### 3.3 参考仓库的独立证据

参考仓库的有效主线是 `fresh993 → 跨组1000 → depth5 969 → 调度959 → leaf952 → 调度951 → 三组尾部 hash 950`。986、dense992/991、内存偏置991 是旁支，不能串成这一主线。

| 已做工作 | 结果 | 对手写设计的启示 |
| --- | --- | --- |
| 移除轮间与整 tile 屏障，保持指令多重集 | 1,917 → 1,355；后续尾部与 setup 到 1,322 | 独立组跨轮交错是比微调优先级更大的设计自由度 |
| 深层标量化、按位置选 ALU/VALU | 1,322 → 1,210，逐步到 1,111 | 比例随阶段变化；关键向量消费者前不宜随意打散 lane |
| 路径位共享、C6 偏置、节点缓存、双 lane dispatch | 1,110 左右逐步到 999 | 一起考虑“取节点→mix→hash→下一地址” |
| 独立 SSA、跨组共享、深度 5 缓存、leaf MADD | **950**；scratch 1,331 | 手工发现的结构规则与排程/分配配合；不是求解器自动发现整个算法 |
| 950 删除全部 dispatch | **1,101**，静态 1,101 包；另一个 fresh 对照 1,107 | 代码极小，但 LOAD 代价显著 |
| 无 jump 的 FLOW 缓存组合 | **1,017**，静态 1,017 包 | 简洁路线有价值，尚不能据此承诺 <900 |
| 所有标量 hash 恢复 SIMD | 999 → **1,175**，动态 ISA 少 43.8% | 总指令条数更少不代表端口工作更均衡 |
| 扩大所有缓存、全部输入地址转 FLOW、延长两条 PC 链 | 均有变慢对照 | 缓存覆盖、端口迁移和依赖链没有单调收益 |

950 程序静态为 648,161 包，**不满足本次 12k 限制**；其线上超时根因在历史报告中未确认，本地正确性不等于线上接受。以上结果及验证标准见 [参考仓库总账](../original_performance_takehome/docs/OPTIMIZATION_HISTORY.md)、[950 最终报告](../original_performance_takehome/solver_results/goal950/FINAL950.md)、[无 dispatch 对照](../original_performance_takehome/solver_results/no_dispatch/README.md)、[FLOW 缓存对照](../original_performance_takehome/solver_results/flow_lookup/README.md)。

## 4. 成功方法：哪些值得保留

### 4.1 hash 与表示

固定 hash 的基础恒等式已经进入实现。按模 `2^32` 计算：

```text
H1(x) = 4097*x + C1
H3/H4(x) = (33*x + C3 + C4) XOR ((33<<9)*x + (C3<<9))
H5(x) = 9*x + C5
```

H3/H4 用两个并行 MADD 和一次 XOR；H2、H6 的移位与 XOR 仍须执行。这里应复用已验证公式，不需要重新搜索常数。

跨轮携带 `hash(value) XOR C6`，下一节点取 `node XOR C6`，则：

```text
(hash(value) XOR C6) XOR (node XOR C6) = hash(value) XOR node
```

这是真正有效的跨层 XOR 抵消。512 个向量 hash 中，已有 **384 组**递延了 C6；剩余 128 组处于轮 7、8、9、15，合计 1,024 W，其中末轮输出需要正常表示。收益要扣除共享节点 bias、恢复、路径奇偶语义的成本，不能把被删 XOR 都当净收益。依据：[XOR 审计](results/xor_918/README.md)。

### 4.2 分层访问、缓存和内存布局

| 保留的方法 | 为什么有效 | 使用边界 |
| --- | --- | --- |
| 浅层共享 tree VLOAD、bias 和 root | 多组复用运行时节点与常数，删除重复 setup | hash 早期所需值优先就绪 |
| 深度 3 小型 dispatch，顺带准备深度 4 子节点 | 选源和复制融入已经执行的 handler，并填入背景工作 | 当前 15 个双组 + 30 个单组区域；表总计 9,600 包 |
| 11 组深度 5 孙节点预取 | 每组以选择替代 8 次 scalar gather，共替代 88 次读取；沿用深度 3 case | 增加复制、选择和存活期，不能全组无条件扩大 |
| 深度 4–7 镜像并 bias，地址保留为递推坐标 | 中段下一地址简化为 `2*A + bit`，减少每轮 offset 选择 | setup、旧值备份、全部读完成后的恢复必须同时设计 |
| 深度 4/5 原树值留 scratch，6/7 用已读输入区备份 | 减少备份指针、清零与重新构造输出地址 | 输入读须先于覆盖，备份 reload 须先于最终输出 |
| 备份在 bias 计算前完成 | 避免原树值在整个计算期间占 scratch | 曾发现 depth7 值 cycle54 读入、915 才备份；是具体生命周期问题 |
| 子节点复制选择性使用 STORE→VLOAD | 使用有余量的 STORE，汇集运行时词为向量 | 新增 LOAD、buffer 顺序和最终恢复计入总账 |
| 深层部分 overfetch | 保存 `q=A−6`，`VLOAD(q)` 的 lane6 就是目标，`q_next=2*q+bit` | 当前仅 16 个偶数组、深度 8–10；每次实际 8 字写入都必须分配，其他 lane 不可观察 |

主要依据：[镜像与分配历程](SUB900_WORKLOG.md)、[深层 overfetch](results/overfetch_917/README.md)、[当前配置](results/compact_913_packed/config.json)。overfetch 保持 LOAD 指令数，但增加 scratch 写入范围，不能当成免费的 scalar load。

### 4.3 PC、静态代码和端口

| 方法 | 已验证收益 | 不能外推的部分 |
| --- | --- | --- |
| 控制 dispatch 宽度与深度 | 从 477,092 包降到 9,819，随后恢复周期 | 大表越宽越快没有依据 |
| 删除被 handler 替代的不可达 main 位置 | 实际展开更紧凑 | relocation 必须重算 jump/PC 常数并核对执行映射 |
| 仅第一对短 dispatch 链 | 减少 exit/entry 控制开销，参与 919 组合 | 全局长链会耦合各组就绪 |
| 固定 PC 表与树/I/O 地址向量共用 | 919 → 918，少 53 W 的准备工作 | 常数数值相同与生命周期都须满足 |
| 单个 PC bit pool | 与常数 LOAD、header root 等组合到 917 | 四 pool 不是更优；当前 913 已替换该设计 |
| 非关键 uniform 向量通过内存广播 | 有验证的 917 同分、−67 W，成为 915 的起点 | 启动 hash 常数不适合批量搬至内存 |
| 按已保存时序改常数表达式 | 98 个合法改写 + 早期前缀得到 914 | 只最小化常数 DAG 深度反而较慢 |
| PC offset 从已有地址向量 MADD 得到 | 与内存广播合用，914 同分再省 52 W | 不是每个 PC 向量都适合重算 |
| 八个 singleton PC bank 交错 | `64*q + 2311 + 8*bank + lane` 复用现有地址；消除四次向量 offset 加法 | 物理表交错不要求不同组同步执行，不能与运行时组绑定混淆 |
| 短 main 前缀、早期 FLOW 常数 | 表仍从 PC14 起，bootstrap 放在空 FLOW 槽；得到 913 | 初始 pause 必须在任何 STORE 前 |
| 实测 FLOW/LOAD 空槽与一次 rootmix 迁移 | 合计形成 W=54,406 的 913 图 | 依据实际依赖和时间，不是依据平均利用率 |
| 利用已有 4097、766 常数填 PC 表洞 | 两条 offset 指令改写，时序不变，−512 padding 包 | 属于布局收益，不减少 W 或 cycles |

依据：[内存广播](results/memory_vectors_917/README.md)、[按时序改常数](results/scheduled_constants_915/README.md)、[PC MADD](results/compact_914_pc/README.md)、[交错 PC](results/interleaved_pc_914/README.md)、[最终 packing](results/packed_interleaved_913/README.md)。

### 4.4 分配、排程和验证设施

- **lane 生命周期分配**：同一份 993-cycle SSA，整向量分配失败，lane 分配成功并通过冻结执行。关键是允许已消费 lane 复用、考虑真实 8 字访问范围，而非提高机器 scratch 限额。
- **保留合法 warm schedule**：修复负 lag 导致有效初始排程被丢弃的问题，避免工具自己退化。正反向重排、最优排程重启、按消费者安排新 setup 也提高了搜索可用性。
- **按具体 hash pack 迁移 ALU/VALU**：不是再扫全局比例；实测空槽、全依赖审计后保存命名覆盖项。对 W 不变的迁移，收益只能来自资源平衡和就绪时机。
- **完整 lowering 验证**：核对 handler 的背景指令、跨度、lane/case stride、返回 PC、bootstrap、pause、独立展开；末 handler 恰为末逻辑周期时必须跳到程序末尾。
- **内存负例**：延迟旧 buffer 读取并移除复用边，应产生真实错误；padding 写入完整分配、未使用 lane 加毒，防止解释器的隐式零掩盖错误。

这些属于使能或正确性成功，本身不能报作周期收益。源码工具入口：[optimize.py](optimize.py)、[lane_allocate.cpp](lane_allocate.cpp)、[schedule.cpp](schedule.cpp)、[export_candidate.py](export_candidate.py)、[verify_actual_checkpoint.py](verify_actual_checkpoint.py)。

## 5. 没有带来更好提交的尝试

“未改善”只覆盖表中源图和已做实现。有些方向已经正确执行但变慢；有些没有分配成功；有些仅做过局部模板排除。这三类结果不能混写。

### 5.1 hash 压缩与提前分支

| 尝试 | 结果 | 下一步价值 |
| --- | --- | --- |
| 将 H2 常数吸入 H1 MADD，或 H6 常数吸入 H5 MADD；允许固定输入 XOR bias | 指定模板在必要低位投影 UNSAT | 没有新的模板前不应继续重复同类搜索 |
| 两个任意仿射项 XOR、MADD/XOR/MADD 链，继续融合 H3/H4/H5 | 指定 9/12-bit 模板被排除；低位模型不能通过更高投影 | 不是 32-bit 全算法最优证明 |
| 11 类非线性乘积、AND/OR/XOR 操作数、二次式 | 指定族被必要投影排除 | 不支持据此宣称所有乘法改写不可行 |
| 四操作 outer-affine/feedback、18 个 AND/OR 目标组合 | 没有完整 32-bit 新公式 | 不属于宏观手写路线的优先投入 |
| `ahead_bits`，利用 `bit16(x*65537)=bit16(x) XOR bit0(x)` 提前 parity | 提前约两层依赖，但每个选中分支新增 MADD；在 971 源图上未改善；保留的验证对照 980 | 先算额外工作，不能只看提前两周期 |
| 将剩余深层 C6 全部预抵消 | 尚无有利实现；三层完整预 bias 要处理 `256+512+1024=1792` 节点，最多替代对应轮的 768 W | 连 bias 本身就超过这部分 gross savings，尚未计恢复；不能重复计算已实现的 384 组抵消 |

依据：[xor_918](results/xor_918/README.md)、[xor_pair_synthesis](results/xor_pair_synthesis/README.md)、[product_fusion](results/product_fusion_918/README.md)、[affine_outer](results/affine_outer_917/README.md)、[lookahead 验证](results/lookahead/candidate_017/verification.json)。最后一行是预算排除，不冒充已执行实验。

### 5.2 缓存、保留原树与 STORE/VLOAD 交换

| 尝试 / 覆盖 | 已知结果 | 主要限制 |
| --- | --- | --- |
| 保留所有原树层在 scratch | 未找到可分配候选 | 长期存活与连续向量要求 |
| 在保存的 938 排程保留层 6/7 | live peak 1,506，但 84 种 packing 均失败 | 峰值低于 1,536 仍可能碎片化；不是不可能证明 |
| 原始 heap-backup 低工作图 | 计数约 913，SSA 993；整向量分配在 1,508 live、最大空段 4 字时失败；换 lane 分配后真实 993 | 清楚区分图收益、分配与可执行周期 |
| 镜像堆中 conditional LOAD 代替复制，24 个配置 | 32-load 对照冻结 **944 / 9,792**，未胜 934 | 新增 LOAD 压力抵消复制节省 |
| 孙节点四词行宽 STORE，48 配置 | 最好筛选 930；paired 控制验证 922 | 更晚的 VLOAD、地址和行复用等待 |
| 深度 4 dispatch，64 配置 | 最好验证 **948 / 11,865** | case 成本、读窗口与同步 |
| 取消孙节点缓存、保留原树，24 配置 | 部分缓存验证 942；全部取消 W−512，实际 **967** | 后续 gather 形成 LOAD 窗口界 916 |
| 深度 2 dispatch / 两轮路径折叠，24 配置 | 验证 920；折叠版本净 −40 W、+12 FLOW，却 **923** | 取节点边界改变后，FLOW 和依赖同步增加 |
| late child-pair rows：48 布局 + 28 顺序 + 16 平衡 | 最好验证 **919**，未胜 917；扩大到七对为 927/930 | 保存复制却新增读取、有效 lane 布局和行依赖 |
| early pair rows，16 配置 | 验证 914、917，未胜 914 | even/odd lane 顺序解决表示问题，未解决总吞吐/时间窗口 |
| uniform 内存广播，30 广泛 + 18 后用配置 | 广泛最好 929；全部转移 −267 W 却 **994**；后用版本才保住 917 | 早期 hash 常数晚就绪，全转移图的窗口界为 924 |
| STORE span，192 配置、16 个进入排程 | 最好窗口界 901，W=53,844；全部未保留分配结果；诊断 peak 1,604/1,673 | 超 scratch 与晚 LOAD；不把 974/976 SSA 当成绩 |
| STORE span 的 12 个小改对照 | 有验证 **920 / 11,173** | 当前源 915 上没有收益 |
| 早期 sibling 地址 / pair-prefetch，16 配置 | 验证 915、917、968，未胜 915 | 22 字真实写入跨度；VLOAD 保留两个孩子有额外选择与存活成本 |
| 全组或按层扩大 overfetch，30+13、再 24 配置 | 部分组成功到 915；全组 W=54,295 但 **924**；后续低工作验证 **915，W=54,310**，未胜源 914 | 少 FLOW 不等于 scratch 与消费者足够早 |

依据：[cache_followup](results/cache_followup_918/README.md)、[work_reduction_followup](results/work_reduction_followup_918/README.md)、[late rows](results/late_pair_rows_917/README.md)、[early rows](results/early_pair_rows_914/README.md)、[memory vectors](results/memory_vectors_917/README.md)、[store spans](results/store_spans_915/README.md)、[local spans](results/store_spans_local_915/README.md)、[pair prefetch](results/prefetch_pairs_915/README.md)、[overfetch](results/overfetch_917/README.md)、[per-level overfetch](results/overfetch_levels_914/README.md)。

### 5.3 多组、更长 handler、quartet 与联合低工作图

| 尝试 | 可确认结果 | 结论 |
| --- | --- | --- |
| 2–4 包 case、多段 STORE、短 span-chain | 验证单个双包 930/9,834；四个三包 948/10,244；混合 span 链 932/9,892 | 工具支持有效，未改善 919 |
| 单 I/O buffer 复用与早期 gather | 验证 943/10,183、933/10,061 | 省 buffer 容量会加入顺序边；错误组合曾形成依赖环 |
| 全局 dispatch 长链 / 跨轮链 | 个别首对短链成功；更大链未持续获益 | 共享控制减少 FLOW，却耦合就绪和延长 producer 生命周期 |
| joint inventory，240 组合 | 最好固定图窗口界 902 | 当阶段组合仍未过 899 必要条件 |
| cache / 常数 / chain 图原型 | 得到窗口界 899、W=53,825 | 是值得排程的图，尚不是成绩 |
| bound899，60 配置、11 图排程 | 最好冻结 **967 / 11,900**；界 899；gap 68 | 下界足够低不代表已知可实现；68 不是固有开销证明 |
| dense tables + span，36 配置 | 布局单改验证 **914 / 10,167**，W=54,575；span 对照 **959 / 11,444** | 是更小代码的有效备选；低工作版本尚未兑现周期 |
| early/late rows + spans + PC，128 配置 | 最低 W=53,698，计数界 **895**，但同图窗口界 **923**；另一个图最好窗口界 901 | 不能取两个不同图的最佳数字拼成一个方案 |
| 上述图提前 depth5 读取，12 配置 | 最低工作图窗口界 923→921；联合控制冻结 **1,049 / 11,998** | 提前一些读不足以修复结构性就绪问题 |
| quartet：64 字行、两次 VSTORE/lane、base-four dispatch，40 配置 | 最好冻结 **916 / 10,929，W=54,455** | 正确地少复制；新增 PC、VLOAD、同步吃掉收益 |
| quartet natural-PC，16 配置 | 三组控制 **917 / 11,314，W=54,431** | 分离物理 PC 顺序与 producer 顺序正确，但未更快 |
| quartet + early/late rows + chain 联合 | 冻结 **936 / 11,333，W=54,403** | 功能兼容成功，性能失败 |
| 不含 quartet 的 natural-PC child rows，12 配置 | 冻结 **916 / 10,809，W=54,470** | 未胜 914 |
| quartet 联合，192 图、12 图排程 | 最低 W=53,850、计数界898、同图窗口界923；另一图窗口界902；无已分配结果 | 诊断 live peak 1,623/1,666/1,793，均超 1,536 |
| producer-first quartet chain | 语义通过，窗口界变为 903 | 当前形式未改善 902；尚无执行成绩 |

依据：[span 及 I/O 记录](SUB900_WORKLOG.md)、[joint inventory](results/joint_inventory_915/README.md)、[899 图原型](results/cache_constant_trade_915/README.md)、[899 的执行反例](results/bound899_915/README.md)、[dense tables](results/dense_tables_914/README.md)、[pair/store joint](results/pair_store_joint_914/README.md)、[joint prefetch](results/joint_prefetch_914/README.md)、[quartet](results/quad_dispatch_914/README.md)、[quartet joint](results/quad_joint_914/README.md)。

### 5.4 常量、端口迁移、局部重排与求解器

| 尝试 | 结果 | 适用结论 |
| --- | --- | --- |
| 旧版全量 ALU 常数合成 | LOAD−10，周期却 985–988 | 启动关键常数从一拍 LOAD 变成长依赖链；选择性版本才有效 |
| 旧版地址树、input load 抢先、强制填启动 LOAD 洞 | 洞消失，仍 981–984，强制版本约 990；已回退 | 可见空洞不一定在完工关键路径上 |
| 旧版 PF2 两位路径折叠 | 983/1,001；+1 FLOW 版本 981/983，未改善 | 工作转到 FLOW/ALU 后新端口成为限制 |
| 精确 restore 范围、header root 复用 | 14 个 restore/buffer、4 个 root 配置保住 918；后续 root 复用参与 917 | 去掉多余依赖可保留，但并非每次立即省周期 |
| scalar root mix、删 root broadcast，16 配置 | −16 W，但验证 922，未胜 918 | 共享值更少未必使首 hash 更早 |
| scalar 常数改 LOAD：50 变体、12 提前量、6 hole rebalances | 918 同分 −48 W 有组合价值；本轮未独立加速 | 后来参与 917，保留同分低工作候选有用 |
| PC bit pool：两轮各 15 + 12 组合 | 一个 pool 合适；扩大至四个未进一步获益 | 不能按省一条向量操作线性外推 |
| uniform scalar operand sharing，16 配置 | 最好已分配筛选 915；未独立冻结晋升 | 删 broadcast 后强迫整个阶段标量化可能更差 |
| 全局常数 DAG 重合成，6 族 | 最好冻结 **921** | 最小深度目标忽略当前资源时序；后续 schedule-aware 版本才到 914 |
| interleaved PC，24 配置 | 低工作/低界图出现；冻结控制 914/W54,427；更宽 bank 未更快 | 新常数与生命周期抵消扩大 bank 的收益 |
| 前缀常数，22 配置；浅层表初始化，16 配置 | 前者产生 913；后者最好选中冻结控制 **917** | 不是所有 setup 提前都有效 |
| FLOW/LOAD 各 5 个空槽预算、6 次局部再平衡 | 保持 913，减少工作；没得到 912 | 已完成的局部菜单内未改善，不是全局最优 |
| 四个较长固定图搜索，各 80×1,000 迭代 | 最好分别 913、913、914、916 | 重复搜索没有新收益，不支持继续无界加预算 |
| 旧 932 源图 CP-SAT，1,200 秒 | 界 902，SSA 936，分配失败 | 图范围与 scratch 都不同，不能报 936 实际成绩 |
| 934 图 CP-SAT，240 秒 | 保留合法 934，报告固定图界 917 | 仅排程/证明范围内的结果 |
| PC-delta 图四个较长搜索，各 100×1,200 迭代 | 界 910 的图仍为 914 | 不能将 4-cycle 差宣布为不可避免 |
| 常数 LOAD 918 图局部修复 | 917/916 搜索 UNKNOWN；918 零窗控制复现 | 超时不证明目标不可能 |
| 914→913 局部 CP 修复 | 固定 lane 顺序 ±1 窗 UNSAT；±2/4/8 为 UNKNOWN | 只排除那个窄域；之后结构改写仍得到了 913 |

依据：[旧复盘](RETROSPECTIVE.md)、[常数/路径跟进](results/work_reduction_followup_918/README.md)、[uniform operands](results/uniform_operands_915/README.md)、[constant DAG](results/constant_dags_915/README.md)、[interleaved PC](results/interleaved_pc_914/README.md)、[prologue tables](results/prologue_tables_914/README.md)、[长搜索](results/interleaved_refine_913/README.md)、[局部 CP](results/interleaved_repair_914/README.md)、[工作日志](SUB900_WORKLOG.md)。

### 5.5 已写原型或提出，但不能列为成功实验

| 方向 | 证据状态 | 下一次评估需要补什么 |
| --- | --- | --- |
| FLOW 辅助两轮路径合并 | `codex/flowfold` 的 `624d155`：保留旧 q 和 bit，用两次选择组成两位，再用 `4*q+s` | 未找到该新开关对应的独立冻结成绩；检查新增 FLOW、延迟路径消费者和组合正确性 |
| memory spill 分配 | 同分支 `0d37642`、`58439d4`：长寿命向量写内存、使用前 reload，按分配失败反馈选值 | 已有原型源码；未找到独立冻结成绩。需正式保留内存行、处理现有临时区冲突/恢复、审计所有读写并核算 reload |
| mixed-radix selector fusion | 设计建议，尚未实现 | 当前 quartet 是统一 base-four，不能用它宣称 mixed-radix 已试过 |
| 多区域共享同一套 handler | 设计建议，尚未实现 | 固定寄存器角色、背景 bundle、返回 PC、控制开销必须一起解决 |
| 规则化 modulo pipeline / 固定物理角色的新手写内核 | 本文建议的后续路线，尚无新测量 | 需要完整 prologue/steady-state/epilogue、scratch/端口/静态预算及冻结验证 |

旁支源码：[spill_allocator.py](../980-flowfold/spill_allocator.py)。这里的“未找到”限定于此次检查到的提交及工作区持久产物，不把原型存在本身当成已验证可用。

## 6. 需要明确修正的历史判断

1. **“dispatch 被 vselect 严格支配”已被后续实现推翻。** 早期按一个节点一个独占 jump 估价，漏算多组配对、每 lane handler 内的背景计算、子孙节点预取。现在的 dispatch 不是预存随机树答案，而是选择固定 scratch 源并执行运行时计算。
2. **“固定 scratch 放不下”可能是分配器问题。** 993 的 lane 分配成功是直接反例；反过来，live peak 小也不是连续向量可分配证明。
3. **“SMT 已证明 hash 理论最短”说得过强。** 保存的证明只排除了若干给定模板和必要投影，不能覆盖全部 ISA 程序和跨轮表示。
4. **旧“内存私有、只要输出对”不能作为当前实现规范。** 本主线要求非输出内存保持；镜像、临时行和备份都必须恢复。内存长度和写范围同样检查。
5. **计数、利用率、ISA 总条数均不能独立预测周期。** 895 算术界对应 923 窗口界；899 窗口界对应 967 实测；反过来，同图好排程也曾大幅改善。必须找出是哪类成本限制该方案。
6. **“全部方案没有成功”也不准确。** 同分低工作方案后来参与更好组合；静态 packing 直接扩大可用设计空间。应保留少量互不支配的方案，但不把组合收益直接相加。

## 7. 不做深度数值优化，怎样手写才合理

这里将“不做深度数值优化”理解为：停止大规模常数/系数合成、成百上千个参数组合、长期 basin 搜索和大型求解器压拍。仍保留指令计数、时序分析、少量对照、真实机器验证。否则无法判断手写方案是否合理。

“手写”建议落在**几种算法模板、资源分工和物理生命周期**上，用简单生成器展开重复工作。逐条输入一万多个静态包既难审查，也没有性能上的额外优势。生成器可以完全确定、不做搜索；固定 ISA 对 scratch 源地址的限制仍须遵守。

### 7.1 先选架构：小型 dispatch 与分层流水最合适

| 路线 | 已有证据 | 评估 |
| --- | --- | --- |
| 全 gather / 无 jump，配浅层 FLOW 缓存 | 参考路线有 1,101/1,017；本仓库旧 980 也是小代码起点 | 适合简单、易维护基线；没有支持 <900 的新预算 |
| **少量深度 3 dispatch，准备下一层，选择性缓存到深度 5，深层 gather** | 本次 971→913 且 ≤12k | **首选**。能同时利用端口与树结构，静态规模可控 |
| 更宽、更深 dispatch 或长 STORE handler | 多轮实验的低工作图受到 LOAD 窗口、scratch 和体积限制 | 只有新生命周期/共享设计出现时再考虑 |
| 重新搜索 hash 最短表达式 | 已有模板多次否定，无新等价式 | 与本次“不做深度数值优化”不符，优先级最低 |

可以复用当前已经验证的 PC 布局公式和 hash 恒等式；不需要为了“手写”删掉它们。若希望代码更加规则，也可先采用较简单的 dense 表布局，再单独量化放弃精细布局/局部迁移的代价；已有 914/10,167 是布局选择的对照，不是新手写内核的成绩预测。

### 7.2 以层为单位规定数据表示和访问模板

| 深度 / 轮 | 建议主模板 | 写代码前需固定的接口 |
| --- | --- | --- |
| root，轮 0/11 | 一次运行时 root 准备，多组共享；回根已知 | 正常/带 C6 偏置的状态在什么位置转换 |
| 深度 1–2，轮 1–2/12–13 | 紧凑缓存，FLOW 选择或 Boolean MADD | bit 保留多久，差值只准备一次；MADD 条件必须严格 0/1 |
| 深度 3，轮 3/14 | 单/双组小表；handler 同时 mix、准备孩子并穿插独立工作 | PC 构造、八个 lane 的顺序、buffer 所有权、返回点 |
| 深度 4，轮 4/15 | 选已备孩子；末轮没有无用 next-index 工作 | 最后一轮恢复正常 hash，及时交给两个 STORE 槽 |
| 深度 5，轮 5 | 小部分组使用四孙节点缓存，其余直接 gather | 缓存覆盖由 LOAD 窗口及寿命决定，不要求每组相同 |
| 深度 6–7 | 镜像树坐标递推，gather | 原树备份/恢复边界，减少坐标来回转换 |
| 深度 8–10 | 普通 gather 为基线；部分组采用 lane6 overfetch | 每次真实 8 字写入的短寿命空间和标量 mix 消费时机 |

优先统一**表示和接口**，不强求所有组使用完全相同的访问方式。现有收益常来自适量覆盖：全组缓存、全组 overfetch、全量内存广播都出现过变慢。手写规则可以是“少数固定 cohort 使用一种模板”，不必退化成逐操作例外表。

### 7.3 手工设计跨组跨轮流水，不加整轮屏障

每个组的真实链为 `路径就绪 → 取节点 → mix → hash → branch bit → 下一层`。不同组并行，把某组的 LOAD、另一组的 hash、第三组的选择/STORE 放在同一包。一个组准备好下一轮就可以继续，不等待全部 32 组结束当前轮。

有限 scratch 需要滚动复用，但不能把实现写成“一个小 tile 做完全部 16 轮才切下一个”。那会把深层 LOAD 密集区和浅层 FLOW 密集区分隔开，失去端口互补。合理的是保持少量处于不同阶段的 cohort，并按组交接物理状态；具体同时在途数量由存活量和时序表决定，不能仅凭经验指定最佳 tile 大小。

手写排程可以使用固定的简单规则：MADD 留给 VALU；可拆分的二元运算适量交给 ALU；优先发射有后继等待的节点读取；FLOW 先满足 dispatch/关键选择；早期常量保住首 hash；尾部从最终两 STORE 槽倒排。尾部即将供给向量消费者的 8 lane pack 尽量完整，避免最后一个 scalar lane 拉长收尾。

一张稳态资源模板必须包含 handler 的背景工作。把查表写成单独串行子程序，返回后才恢复 hash，不能直接继承现有 913 的计数到周期转换效率。

### 7.4 先画生命周期，再决定预取与 spill

物理空间至少按角色审查：当前 value/hash 临时量、path/bit、少量待选节点、共享常数和树缓存、overfetch 写入范围。对每种值明确 producer、最后一次读取、可覆盖时刻。优先及时消费、提前备份、缩短中间表示寿命；仅在有明确复用收益时长期保留。

当前 peak 看似还有 `1536−1322=214` 字，但地址跨度已是 1,536，不能直接再分配 214 字。手写固定寄存器布局更需要处理连续 8 字和跨 phase 复用。机器同包读旧值再写新值可用；同包读取刚写的新值不可用。

不建议把通用 spill 作为 <900 的主要手段。每次 reload 抢 LOAD，899 的全程 LOAD 余量仅 20 槽，实际窗口更紧。先减少生存期、压缩行布局或重算便宜地址；若少量 spill 能换来更大的 gather 消除，再逐笔算账。spill 可以解决“无法分配”，不自动解决“足够快”。

### 7.5 静态表预算必须在宏观设计时算

对当前每 lane 展开结构，深度 `d`、联合宽度 `w`、每 case `s` 包的区域，case 数成本为：

```text
静态 case 包 = 8 × 2^(d*w) × s
```

| 示例，s=1 | 单区域 case 包 |
| --- | ---: |
| 深度 3、单组 | 64 |
| 深度 3、双组 | 512 |
| 深度 3、三组 | 4,096 |
| 深度 4、双组 | 2,048 |
| 深度 5、双组 | 8,192 |

还须加 main、padding、PC 准备和控制成本。当前 `15×512 + 30×64 = 9600`，剩余静态余量只有 **1,463 包**。它是完整程序相对上限的差，不等于能随意加同样数量的 case 后保持布局和周期。

共享 handler 的潜在价值是减少复制体积，让同一 12k 预算容纳更好的访问策略；它本身不必然减少动态工作。当前表携带不同背景指令、寄存器地址与返回点，不能仅对相同取节点片段去重。要共享，必须先让这些角色一致，并将新增控制成本入账；本文未给它虚构周期收益。

## 8. 对 <900 的具体预算与优先级

### 8.1 现有工作还在哪里

数据来自 [当前工作分类](results/compact_913_packed/analysis.json)：

| 类别 | W | 宏观手写可动的部分 |
| --- | ---: | --- |
| hash 内部 | 41,984 | 复用已证恒等式和 C6 递延；本路线不继续做深度公式搜索 |
| node mix XOR | 4,096 | 仍需混合运行时节点，不能因 XOR 多就假定可消除 |
| path bit mask | 3,584 | 检查条件复用、避免重复物化；保留正确 Boolean 域 |
| path/address 更新 | 2,784 | **优先**：持续使用一种坐标，推迟完整地址，跨相邻层合并 |
| dispatch 地址 | 744 | **优先**：统一 PC/内存地址角色，减少重复 pack 与 offset；须扣准备成本 |
| 条件子/孙节点复制 | 592 | **有条件优先**：减少副本和写入次数，同时提前实际消费者 |
| setup / restoration 算术 | 558 | 共享和及时回收；关键常数优先直接可用 |
| MADD 节点选择 | 64 | 按 FLOW/VALU 时窗选择，体量较小 |
| 合计 | **54,406** | 分类成本不等于都能删除 |

hash 加 mix 已占约 84.7%。剩余部分仍有结构空间，但不能靠只优化 setup 承诺大幅下降：setup/restoration 的算术总共 558 W，连将其全部删光都不足以独立达到 895 的算术预算。

### 8.2 下界要多低

| 算术容量目标 | W 上限 | 相对当前至少减少 | 899 周期目标允许的 `T−L`，若完整下界恰为该行数值 |
| --- | ---: | ---: | ---: |
| 899 | 53,940 | **466 W** | 0 |
| 895 | 53,700 | **706 W** | 4 |
| 890 | 53,400 | **1,006 W** | 9 |

466 W 也等于 ALU 的 82 槽超额加 VALU 的 `48×8` 超额；不是在它之外再加一次。通过 `8 ALU ↔ 1 VALU` 换形式不改变 W；换成 FLOW、LOAD、共享准备或更少工作可能改变 W，但须检查接受工作的端口及窗口。

数学上，只需 `L ≤ T ≤ 899`，不要求 `L` 显著低于 899。工程上以 **完整资源窗口下界 890–895** 为设计目标更合理，因为需要留启动、尾部、端口耦合和分配余量。这里“完整”指检查所有已建模端口窗口，仍不是包含一切约束的精确下界。

不能把当前 gap=4 套给新图：已有界899/实测967的反例；也不能只把算术界降到895，因为已有同图窗口界923的反例。若采用更规则的手写模板，排程差距可能比当前精细搜索图更大，因此 890–895 只是筛选目标，绝非充分条件。详见 [下界余量分析](results/sub900_margin_915/README.md)。

### 8.3 哪些宏观工作值得优先做

| 优先级 | 问题 | 建议做法 | 不再重复的旧形式 |
| --- | --- | --- | --- |
| P0 | 地址/路径是否多次重建 | 为 root、浅层 selector、中深层坐标写清状态转移；只在真实读取/dispatch 前物化所需表示 | 在现有两步上机械加 FLOW，忽略边界和时机 |
| P0 | producer 就绪与 node-copy 是否一起改善 | 设计小 buffer 的准备→读取→释放周期，减少冗余复制并确保节点读提前 | 只延长 STORE handler 或只扩大 quartet 行 |
| P0 | 多阶段端口能否持续互补 | 用固定 cohort 模板跨层交错，给关键 LOAD、MADD、dispatch 预留位置 | 整轮屏障、全局长链、全量预取 |
| P1 | PC 与地址能否共用更规则的角色 | 复用已证 interleaved/dense 布局；必要时研究固定角色的共享表 | 先扩大表，再寄希望于常数搜索压回预算 |
| P1 | 同一节点准备是否被重复执行 | 共享差值、bias、root 和晚用 uniform 常数；保留少量例外 | 全量内存广播或强制整阶段标量化 |
| P2 | 最后几拍和启动是否有具体等待 | 手动处理有限的关键 pack/STORE 冲突；每次只改一个原因 | 为提高图上平均 util 而填无关空洞 |
| 暂停 | hash superoptimization、大范围组合、重复 CP 超时 | 只有出现新等价式或新架构再开启 | 在已排除模板和同一窗口反复加时间 |

对“压缩计算量”的准确表述应是：**需要降低瓶颈端口的工作及其时间窗口需求；不必消除相同数量的数学运算，更不必改动固定 hash 算法。** 地址表示、共享节点准备、复制方式和 PC 构造都可能贡献。宏观布局如果只是把 466 W 从 ALU/VALU 搬到已经紧张的 FLOW/LOAD，同样不能达标。

## 9. 建议的手写实施顺序与停止条件

这是一份可执行设计计划，尚未开始重写，也没有给新方案预测 cycle 分数。

1. **冻结 913 对照，单独写规则化生成器。** 输入契约、hash、机器和验证入口继续复用。先明确要付出多少代码简化代价，避免覆盖唯一可复现的最佳程序。
2. **完成一页设计账本。** 每层的表示、节点来源、A/V/L/S/F 操作数、case 数、值存活区间及恢复范围齐全；先做总量，再检查 LOAD/FLOW 的实际释放窗口。若固定图必要界已 >899，它可以是简洁基线，但不能标为 sub-900 候选。
3. **做两三个有因果区别的宏观对照。** 例如直接 gather 与有限孙节点缓存；普通深层地址与部分 overfetch；一种规则化小表布局与现有布局。不要立即对每个组、轮、指令开放几十个旋钮。
4. **先保证物理可执行，再手调时序。** 分配连续向量、覆盖 padding、保护初始零和临时区；通过语义检查后跑真实 lowering/frozen 检查。没有分配的 890 排程不能替代有执行证据的 913。
5. **按 trace 只修有限的启动/尾部冲突。** 正文应接近模板设计预期；若差几十周期，回到跨组就绪、buffer 所有权和生命周期，不继续盲目压拍。
6. **决定是否仍有 <900 路线。** 若宏观工作无法给出至少 466 W 净削减且端口窗口同时合格，就如实保留为更简洁实现；若能把完整窗口界推到 890–895，再检验实际差距是否足够小。新图差距过大则改图，不套用旧图的 4-cycle 差。

每个候选的最小记录应包含：来源与改写原因、W/各端口、窗口界、实际 cycles、静态包、scratch 峰值与分配跨度、验证状态。使用既有 [export_candidate.py](export_candidate.py) 和 [verify_actual_checkpoint.py](verify_actual_checkpoint.py) 完成真实程序核对，晋升后再运行原始测试并生成绑定源码的 port util。

总体选择是：**以现有成功结构为基础，减少策略种类，明确组间流水和内存所有权，用少量模板生成真实 VLIW。** 这条路线适合降低实现复杂度和搜索依赖；能否突破 900，要由新的工作/窗口预算与执行程序证明。当前证据不支持“只要手写就能省掉最后 14 cycles”。

## 附录 A：全部本地实验目录索引

以下由本次工作区文件盘点生成，按目录名排序。覆盖正文的配置批次，也保留早期没有专门 README 的扫描。目录最短成绩仅在存在带 seeds、cycles、static_bundles 的 `verification.json` 时填写；不把 `search.json`、`result.json` 或 `100000` 哨兵当成冻结成绩。模板证明类的 `verification.json` 不按机器执行结果计数。

配置快照和冻结记录可能在晋升时复制到新目录，数量不代表互不重复的实验数。最短一项不代表该目录其他候选同样验证通过。带“本地”的目录没有已跟踪文件，仍登记其现存证据；本次不批量提交历史临时产物。

盘点结果：**183 个目录，1,805 份 config.json 快照，110 份机器冻结记录，54 份 README**。其中 97 个目录含已跟踪文件，86 个目录只有本地历史产物。

| 实验目录 / 主证据 | 方向 | 配置快照 | 冻结记录 | 目录最短冻结 cycles / 包 |
| --- | --- | ---: | ---: | ---: |
| [affine_feedback_917（本地）](results/affine_feedback_917/summary.json) | hash 模板/证明 | 0 | 0 | — |
| [affine_feedback_free_917](results/affine_feedback_free_917/summary.json) | hash 模板/证明 | 0 | 0 | — |
| [affine_outer_917](results/affine_outer_917/README.md) | hash 模板/证明 | 0 | 0 | — |
| [balance_918](results/balance_918/summary.json) | 排程/分配/再平衡 | 6 | 0 | — |
| [balance_920（本地）](results/balance_920/candidate_000/verification.json) | 排程/分配/再平衡 | 4 | 1 | 919 / 10,159 |
| [balance_921（本地）](results/balance_921/summary.json) | 排程/分配/再平衡 | 8 | 0 | — |
| [balance_929（本地）](results/balance_929/candidate_007/verification.json) | 排程/分配/再平衡 | 8 | 1 | 928 / 9,776 |
| [balance_931（本地）](results/balance_931/candidate_007/verification.json) | 排程/分配/再平衡 | 10 | 1 | 929 / 9,777 |
| [balance_io919（本地）](results/balance_io919/summary.json) | 排程/分配/再平衡 | 8 | 0 | — |
| [bitwise_outer_917](results/bitwise_outer_917/summary.json) | hash 模板/证明 | 0 | 0 | — |
| [bound899_915](results/bound899_915/README.md) | 联合图/下界与排程 | 11 | 1 | 967 / 11,900 |
| [buff_0_2（本地）](results/buff_0_2/config.json) | 复制/临时 buffer | 1 | 0 | — |
| [buff_0_4（本地）](results/buff_0_4/config.json) | 复制/临时 buffer | 1 | 0 | — |
| [buff_0_8（本地）](results/buff_0_8/config.json) | 复制/临时 buffer | 1 | 0 | — |
| [buff_4_2（本地）](results/buff_4_2/config.json) | 复制/临时 buffer | 1 | 0 | — |
| [buff_4_4（本地）](results/buff_4_4/config.json) | 复制/临时 buffer | 1 | 0 | — |
| [buff_8_4（本地）](results/buff_8_4/config.json) | 复制/临时 buffer | 1 | 0 | — |
| [cache_constant_trade_915](results/cache_constant_trade_915/README.md) | 联合图/下界与排程 | 15 | 0 | — |
| [cache_followup_918](results/cache_followup_918/README.md) | 审计/综合报告 | 0 | 0 | — |
| [cache_tradeoff_918](results/cache_tradeoff_918/candidate_015/verification.json) | 树布局/备份/生命周期 | 22 | 2 | 942 / 10,835 |
| [compact_1_4_1（本地）](results/compact_1_4_1/config.json) | 早期 dispatch/压缩/链 | 1 | 0 | — |
| [compact_1_4_2（本地）](results/compact_1_4_2/config.json) | 早期 dispatch/压缩/链 | 1 | 0 | — |
| [compact_2_0_1（本地）](results/compact_2_0_1/config.json) | 早期 dispatch/压缩/链 | 1 | 0 | — |
| [compact_2_0_2（本地）](results/compact_2_0_2/config.json) | 早期 dispatch/压缩/链 | 1 | 0 | — |
| [compact_2_4_1（本地）](results/compact_2_4_1/config.json) | 早期 dispatch/压缩/链 | 1 | 0 | — |
| [compact_913](results/compact_913/README.md) | 保存检查点 | 1 | 1 | 913 / 11,049 |
| [compact_913_packed](results/compact_913_packed/README.md) | 保存检查点 | 1 | 1 | 913 / 10,537 |
| [compact_914](results/compact_914/README.md) | 保存检查点 | 1 | 1 | 914 / 10,807 |
| [compact_914_pc](results/compact_914_pc/README.md) | 保存检查点 | 1 | 1 | 914 / 10,807 |
| [compact_915](results/compact_915/README.md) | 保存检查点 | 1 | 1 | 915 / 10,808 |
| [compact_917](results/compact_917/README.md) | 保存检查点 | 1 | 1 | 917 / 10,810 |
| [compact_918](results/compact_918/verification.json) | 保存检查点 | 1 | 1 | 918 / 10,811 |
| [compact_919](results/compact_919/verification.json) | 保存检查点 | 1 | 1 | 919 / 10,159 |
| [compact_928](results/compact_928/verification.json) | 保存检查点 | 1 | 1 | 928 / 9,776 |
| [compact_931（本地）](results/compact_931/verification.json) | 保存检查点 | 1 | 1 | 931 / 9,779 |
| [compact_934](results/compact_934/verification.json) | 保存检查点 | 1 | 1 | 934 / 9,782 |
| [compact_934_cp（本地）](results/compact_934_cp/config.json) | 排程/分配/再平衡 | 1 | 0 | — |
| [compact_934_cp2（本地）](results/compact_934_cp2/result.json) | 排程/分配/再平衡 | 1 | 0 | — |
| [compact_936](results/compact_936/verification.json) | 保存检查点 | 1 | 1 | 936 / 9,784 |
| [compact_937（本地）](results/compact_937/verification.json) | 保存检查点 | 1 | 1 | 937 / 9,785 |
| [compact_939](results/compact_939/verification.json) | 保存检查点 | 1 | 1 | 939 / 9,787 |
| [compact_971](results/compact_971/verification.json) | 保存检查点 | 1 | 1 | 971 / 9,819 |
| [compact_alias（本地）](results/compact_alias) | 早期 dispatch/压缩/链 | 3 | 0 | — |
| [compact_mix（本地）](results/compact_mix) | 早期 dispatch/压缩/链 | 21 | 0 | — |
| [compact_neighbors（本地）](results/compact_neighbors/candidate_002/verification.json) | 排程/分配/再平衡 | 31 | 1 | 931 / 9,779 |
| [compact_placement（本地）](results/compact_placement) | 早期 dispatch/压缩/链 | 24 | 0 | — |
| [compact_refine（本地）](results/compact_refine/candidate_001/verification.json) | 早期 dispatch/压缩/链 | 6 | 1 | 971 / 9,819 |
| [compact_search（本地）](results/compact_search/candidate_009/verification.json) | 早期 dispatch/压缩/链 | 46 | 1 | 973 / 9,821 |
| [compact_tail（本地）](results/compact_tail) | 早期 dispatch/压缩/链 | 24 | 0 | — |
| [constant_dags_915](results/constant_dags_915/README.md) | 常量/共享/端口迁移 | 6 | 1 | 921 / 10,814 |
| [constant_holes_918](results/constant_holes_918/summary.json) | 常量/共享/端口迁移 | 6 | 0 | — |
| [constant_loads_918](results/constant_loads_918/candidate_014/verification.json) | 常量/共享/端口迁移 | 50 | 1 | 918 / 10,811 |
| [constant_preload_918](results/constant_preload_918/summary.json) | 常量/共享/端口迁移 | 12 | 0 | — |
| [constant_repair_918](results/constant_repair_918/bounds.json) | 常量/共享/端口迁移 | 1 | 0 | — |
| [copy_load（本地）](results/copy_load) | 复制/临时 buffer | 0 | 0 | — |
| [cpsat（本地）](results/cpsat/config.json) | 排程/分配/再平衡 | 1 | 0 | — |
| [dense_tables_914](results/dense_tables_914/README.md) | PC 地址/静态布局 | 9 | 2 | 914 / 10,167 |
| [depth2_918](results/depth2_918/candidate_016/verification.json) | 缓存覆盖/dispatch 深度 | 24 | 3 | 920 / 10,048 |
| [depth4_918](results/depth4_918/candidate_027/verification.json) | 缓存覆盖/dispatch 深度 | 35 | 1 | 948 / 11,865 |
| [dispatch_chains（本地）](results/dispatch_chains) | 早期 dispatch/压缩/链 | 16 | 0 | — |
| [early_pair_rows_914](results/early_pair_rows_914/README.md) | 条件行/STORE span | 16 | 2 | 914 / 10,807 |
| [exact_balance（本地）](results/exact_balance/candidate_032/verification.json) | 排程/分配/再平衡 | 32 | 1 | 926 / 9,774 |
| [flow_holes_913](results/flow_holes_913/README.md) | 常量/共享/端口迁移 | 5 | 0 | — |
| [flow_holes_rebalance_913](results/flow_holes_rebalance_913/README.md) | 常量/共享/端口迁移 | 6 | 0 | — |
| [grand_combine（本地）](results/grand_combine/candidate_007/verification.json) | 缓存覆盖/dispatch 深度 | 9 | 1 | 952 / 9,800 |
| [grand_groups（本地）](results/grand_groups) | 缓存覆盖/dispatch 深度 | 28 | 0 | — |
| [grand_phase（本地）](results/grand_phase/candidate_008/verification.json) | 缓存覆盖/dispatch 深度 | 26 | 2 | 942 / 9,790 |
| [grand_refine（本地）](results/grand_refine/candidate_000/verification.json) | 缓存覆盖/dispatch 深度 | 8 | 2 | 939 / 9,787 |
| [grand_row_pair_probe](results/grand_row_pair_probe/candidate_000/verification.json) | 条件行/STORE span | 1 | 1 | 922 / 10,815 |
| [grand_rows_918](results/grand_rows_918/candidate_025/verification.json) | 条件行/STORE span | 31 | 2 | 930 / 10,823 |
| [grandchildren（本地）](results/grandchildren/candidate_017/verification.json) | 缓存覆盖/dispatch 深度 | 28 | 1 | 947 / 9,795 |
| [header_root_918](results/header_root_918/candidate_003/verification.json) | 常量/共享/端口迁移 | 4 | 1 | 918 / 10,811 |
| [heap_addresses（本地）](results/heap_addresses/candidate_008/verification.json) | 树布局/备份/生命周期 | 6 | 1 | 937 / 9,785 |
| [heap_backup_bound](results/heap_backup_bound/analysis.json) | 树布局/备份/生命周期 | 1 | 0 | — |
| [heap_child_loads](results/heap_child_loads/candidate_001/verification.json) | 树布局/备份/生命周期 | 24 | 1 | 944 / 9,792 |
| [heap_early_backup（本地）](results/heap_early_backup/candidate_007/verification.json) | 树布局/备份/生命周期 | 16 | 1 | 938 / 9,786 |
| [heap_keep_lanes（本地）](results/heap_keep_lanes/candidate_001/verification.json) | 树布局/备份/生命周期 | 12 | 1 | 935 / 9,783 |
| [heap_keep_preserved_6_1（本地）](results/heap_keep_preserved_6_1/config.json) | 树布局/备份/生命周期 | 1 | 0 | — |
| [heap_keep_preserved_7_1（本地）](results/heap_keep_preserved_7_1/config.json) | 树布局/备份/生命周期 | 1 | 0 | — |
| [heap_keep_preserved_7_2（本地）](results/heap_keep_preserved_7_2/config.json) | 树布局/备份/生命周期 | 1 | 0 | — |
| [heap_lanes_993](results/heap_lanes_993/verification.json) | 树布局/备份/生命周期 | 1 | 1 | 993 / 9,841 |
| [heap_reuse](results/heap_reuse/candidate_004/verification.json) | 树布局/备份/生命周期 | 12 | 1 | 950 / 9,798 |
| [heap_screen（本地）](results/heap_screen) | 树布局/备份/生命周期 | 16 | 0 | — |
| [interleaved_pc_914](results/interleaved_pc_914/README.md) | PC 地址/静态布局 | 24 | 1 | 914 / 11,056 |
| [interleaved_refine_913](results/interleaved_refine_913/README.md) | 排程/分配/再平衡 | 4 | 0 | — |
| [interleaved_repair_914](results/interleaved_repair_914/README.md) | 排程/分配/再平衡 | 0 | 0 | — |
| [io_and_early（本地）](results/io_and_early/candidate_008/verification.json) | 树布局/备份/生命周期 | 8 | 1 | 919 / 10,159 |
| [io_early_ordered](results/io_early_ordered/candidate_038/verification.json) | 树布局/备份/生命周期 | 40 | 2 | 933 / 10,061 |
| [joint_inventory_915](results/joint_inventory_915/README.md) | 联合图/下界与排程 | 12 | 0 | — |
| [joint_prefetch_914](results/joint_prefetch_914/README.md) | 联合图/下界与排程 | 12 | 1 | 1,049 / 11,998 |
| [lane_choice（本地）](results/lane_choice) | 排程/分配/再平衡 | 7 | 0 | — |
| [late_pair_chain_order_917](results/late_pair_chain_order_917/candidate_021/verification.json) | 条件行/STORE span | 28 | 1 | 920 / 10,813 |
| [late_pair_local_balance_917](results/late_pair_local_balance_917/summary.json) | 条件行/STORE span | 16 | 0 | — |
| [late_pair_rows_917](results/late_pair_rows_917/README.md) | 条件行/STORE span | 48 | 5 | 919 / 10,812 |
| [load_holes_913](results/load_holes_913/README.md) | 常量/共享/端口迁移 | 5 | 1 | 913 / 11,049 |
| [lookahead（本地）](results/lookahead/candidate_017/verification.json) | 路径/提前分支/预取 | 18 | 1 | 980 / 9,828 |
| [memory_vector_followup_917](results/memory_vector_followup_917/candidate_007/verification.json) | 常量/共享/端口迁移 | 18 | 1 | 917 / 10,810 |
| [memory_vector_holes_917](results/memory_vector_holes_917/summary.json) | 常量/共享/端口迁移 | 6 | 0 | — |
| [memory_vector_pc_917](results/memory_vector_pc_917/summary.json) | 常量/共享/端口迁移 | 15 | 0 | — |
| [memory_vectors_917](results/memory_vectors_917/README.md) | 常量/共享/端口迁移 | 30 | 2 | 929 / 10,822 |
| [neighbors_918](results/neighbors_918/summary.json) | 排程/分配/再平衡 | 46 | 0 | — |
| [neighbors_919（本地）](results/neighbors_919/summary.json) | 排程/分配/再平衡 | 47 | 0 | — |
| [neighbors_921（本地）](results/neighbors_921/candidate_048/verification.json) | 排程/分配/再平衡 | 55 | 1 | 920 / 10,160 |
| [neighbors_926（本地）](results/neighbors_926/candidate_047/verification.json) | 排程/分配/再平衡 | 58 | 1 | 921 / 10,161 |
| [neighbors_931（本地）](results/neighbors_931/summary.json) | 排程/分配/再平衡 | 40 | 0 | — |
| [ones（本地）](results/ones/config.json) | 常量/共享/端口迁移 | 1 | 0 | — |
| [overfetch_917](results/overfetch_917/README.md) | 路径/提前分支/预取 | 26 | 2 | 916 / 10,809 |
| [overfetch_levels_914](results/overfetch_levels_914/README.md) | 路径/提前分支/预取 | 24 | 1 | 915 / 10,808 |
| [overfetch_refine_917](results/overfetch_refine_917/candidate_003/verification.json) | 路径/提前分支/预取 | 13 | 1 | 915 / 10,808 |
| [packed_interleaved_913](results/packed_interleaved_913/README.md) | PC 地址/静态布局 | 1 | 1 | 913 / 10,537 |
| [pair_natural_914](results/pair_natural_914/README.md) | quartet/独立 PC 顺序 | 12 | 1 | 916 / 10,809 |
| [pair_store_joint_914](results/pair_store_joint_914/README.md) | 联合图/下界与排程 | 8 | 0 | — |
| [path_choices_918](results/path_choices_918/candidate_015/verification.json) | 路径/提前分支/预取 | 61 | 4 | 918 / 10,811 |
| [path_refine_918](results/path_refine_918/summary.json) | 路径/提前分支/预取 | 2 | 0 | — |
| [pc_address_pools](results/pc_address_pools/candidate_001/verification.json) | PC 地址/静态布局 | 8 | 1 | 918 / 10,811 |
| [pc_bits_918](results/pc_bits_918/candidate_001/verification.json) | PC 地址/静态布局 | 15 | 1 | 919 / 10,812 |
| [pc_bits_combine_918](results/pc_bits_combine_918/candidate_000/verification.json) | PC 地址/静态布局 | 12 | 2 | 917 / 10,810 |
| [pc_bits_constant_918](results/pc_bits_constant_918/candidate_000/verification.json) | PC 地址/静态布局 | 15 | 2 | 918 / 10,811 |
| [pc_delta_holes_914](results/pc_delta_holes_914/README.md) | PC 地址/静态布局 | 8 | 1 | 914 / 10,807 |
| [pc_delta_memory_914](results/pc_delta_memory_914/README.md) | PC 地址/静态布局 | 14 | 1 | 914 / 10,807 |
| [pc_delta_refine_914](results/pc_delta_refine_914/README.md) | PC 地址/静态布局 | 4 | 0 | — |
| [pc_offset_madd_914](results/pc_offset_madd_914/README.md) | PC 地址/静态布局 | 17 | 2 | 914 / 10,807 |
| [portfolio_0（本地）](results/portfolio_0/config.json) | 早期 dispatch/压缩/链 | 1 | 0 | — |
| [portfolio_1](results/portfolio_1/verification.json) | 保存检查点（旧体积上限） | 1 | 1 | 932 / 477,092 |
| [portfolio_2（本地）](results/portfolio_2/config.json) | 早期 dispatch/压缩/链 | 1 | 0 | — |
| [portfolio_3（本地）](results/portfolio_3/config.json) | 早期 dispatch/压缩/链 | 1 | 0 | — |
| [portfolio_4（本地）](results/portfolio_4/config.json) | 早期 dispatch/压缩/链 | 1 | 0 | — |
| [portfolio_5（本地）](results/portfolio_5/config.json) | 早期 dispatch/压缩/链 | 1 | 0 | — |
| [precise_restore_918](results/precise_restore_918/summary.json) | 树布局/备份/生命周期 | 8 | 0 | — |
| [prefetch3（本地）](results/prefetch3/result.json) | 缓存覆盖/dispatch 深度 | 1 | 0 | — |
| [prefetch_pairs_915](results/prefetch_pairs_915/README.md) | 路径/提前分支/预取 | 16 | 3 | 915 / 10,808 |
| [product_fusion_918](results/product_fusion_918/README.md) | hash 模板/证明 | 0 | 0 | — |
| [prologue_constants_914](results/prologue_constants_914/README.md) | 常量/共享/端口迁移 | 22 | 2 | 913 / 10,795 |
| [prologue_tables_914](results/prologue_tables_914/README.md) | 常量/共享/端口迁移 | 16 | 1 | 917 / 11,047 |
| [prologue_warm_914](results/prologue_warm_914/README.md) | 常量/共享/端口迁移 | 1 | 1 | 914 / 11,043 |
| [quad_dispatch_914](results/quad_dispatch_914/README.md) | quartet/独立 PC 顺序 | 40 | 1 | 916 / 10,929 |
| [quad_integration_914](results/quad_integration_914/README.md) | quartet/独立 PC 顺序 | 1 | 1 | 936 / 11,333 |
| [quad_joint_914](results/quad_joint_914/README.md) | quartet/独立 PC 顺序 | 13 | 0 | — |
| [quad_natural_914](results/quad_natural_914/README.md) | quartet/独立 PC 顺序 | 16 | 1 | 917 / 11,314 |
| [quad_producers_first_914](results/quad_producers_first_914/README.md) | quartet/独立 PC 顺序 | 1 | 0 | — |
| [quad_smoke_914（本地）](results/quad_smoke_914/candidate_000/verification.json) | quartet/独立 PC 顺序 | 1 | 1 | 917 / 10,930 |
| [resource_windows_918](results/resource_windows_918/README.md) | 审计/综合报告 | 0 | 0 | — |
| [retained_928（本地）](results/retained_928) | 树布局/备份/生命周期 | 3 | 0 | — |
| [scalar_roots_918](results/scalar_roots_918/candidate_002/verification.json) | 常量/共享/端口迁移 | 16 | 1 | 922 / 10,815 |
| [scheduled_constants_915](results/scheduled_constants_915/README.md) | 常量/共享/端口迁移 | 12 | 1 | 914 / 10,807 |
| [shallow_sharing（本地）](results/shallow_sharing/candidate_004/verification.json) | 树布局/备份/生命周期 | 8 | 2 | 934 / 9,782 |
| [span_chain_probe](results/span_chain_probe/candidate_000/verification.json) | 条件行/STORE span | 1 | 1 | 932 / 9,892 |
| [span_search](results/span_search/candidate_000/verification.json) | 条件行/STORE span | 34 | 2 | 930 / 9,834 |
| [store2_0（本地）](results/store2_0/config.json) | 复制/临时 buffer | 1 | 0 | — |
| [store2_4（本地）](results/store2_4/config.json) | 复制/临时 buffer | 1 | 0 | — |
| [store2_8（本地）](results/store2_8/config.json) | 复制/临时 buffer | 1 | 0 | — |
| [store_0（本地）](results/store_0/config.json) | 复制/临时 buffer | 1 | 0 | — |
| [store_best（本地）](results/store_best/config.json) | 复制/临时 buffer | 1 | 0 | — |
| [store_spans_915](results/store_spans_915/README.md) | 条件行/STORE span | 16 | 0 | — |
| [store_spans_local_915](results/store_spans_local_915/README.md) | 条件行/STORE span | 12 | 1 | 920 / 11,173 |
| [sub900_margin_915](results/sub900_margin_915/README.md) | 审计/综合报告 | 0 | 0 | — |
| [synth（本地）](results/synth/result.json) | 常量/共享/端口迁移 | 1 | 0 | — |
| [tail_0_0（本地）](results/tail_0_0/config.json) | 早期 dispatch/压缩/链 | 1 | 0 | — |
| [tail_0_32（本地）](results/tail_0_32/config.json) | 早期 dispatch/压缩/链 | 1 | 0 | — |
| [tail_0_64（本地）](results/tail_0_64/config.json) | 早期 dispatch/压缩/链 | 1 | 0 | — |
| [tail_0_96（本地）](results/tail_0_96/config.json) | 早期 dispatch/压缩/链 | 1 | 0 | — |
| [tail_12_0（本地）](results/tail_12_0/config.json) | 早期 dispatch/压缩/链 | 1 | 0 | — |
| [tail_12_32（本地）](results/tail_12_32/config.json) | 早期 dispatch/压缩/链 | 1 | 0 | — |
| [tail_12_64（本地）](results/tail_12_64/config.json) | 早期 dispatch/压缩/链 | 1 | 0 | — |
| [tail_12_96（本地）](results/tail_12_96/config.json) | 早期 dispatch/压缩/链 | 1 | 0 | — |
| [tail_4_0（本地）](results/tail_4_0/config.json) | 早期 dispatch/压缩/链 | 1 | 0 | — |
| [tail_4_32（本地）](results/tail_4_32/config.json) | 早期 dispatch/压缩/链 | 1 | 0 | — |
| [tail_4_64（本地）](results/tail_4_64/config.json) | 早期 dispatch/压缩/链 | 1 | 0 | — |
| [tail_4_96（本地）](results/tail_4_96/config.json) | 早期 dispatch/压缩/链 | 1 | 0 | — |
| [tail_8_0（本地）](results/tail_8_0/config.json) | 早期 dispatch/压缩/链 | 1 | 0 | — |
| [tail_8_32（本地）](results/tail_8_32/config.json) | 早期 dispatch/压缩/链 | 1 | 0 | — |
| [tail_8_64（本地）](results/tail_8_64/config.json) | 早期 dispatch/压缩/链 | 1 | 0 | — |
| [tail_8_96（本地）](results/tail_8_96/config.json) | 早期 dispatch/压缩/链 | 1 | 0 | — |
| [transfer_refine_914](results/transfer_refine_914/README.md) | 排程/分配/再平衡 | 24 | 1 | 914 / 10,807 |
| [uniform_operands_915](results/uniform_operands_915/README.md) | 常量/共享/端口迁移 | 14 | 0 | — |
| [width3_1（本地）](results/width3_1/config.json) | 早期 dispatch/压缩/链 | 1 | 0 | — |
| [width3_2（本地）](results/width3_2/config.json) | 早期 dispatch/压缩/链 | 1 | 0 | — |
| [width3_3（本地）](results/width3_3/config.json) | 早期 dispatch/压缩/链 | 1 | 0 | — |
| [work_inventory_917](results/work_inventory_917/README.md) | 审计/综合报告 | 0 | 0 | — |
| [work_inventory_918](results/work_inventory_918/README.md) | 审计/综合报告 | 0 | 0 | — |
| [work_reduction_followup_918](results/work_reduction_followup_918/README.md) | 审计/综合报告 | 0 | 0 | — |
| [xor_918](results/xor_918/README.md) | hash 模板/证明 | 0 | 0 | — |
| [xor_pair_synthesis](results/xor_pair_synthesis/README.md) | hash 模板/证明 | 0 | 0 | — |

## 附录 B：本次整理的验证边界

- 主内核、`tests/`、`problem.py` 不变；没有运行新的优化或改写算法。
- 当前数字从正式 artifact 交叉核对，检查实际源码摘要、W 分类之和、资源预算、静态构成与文档链接。
- 现有根目录 `utilization_profile.*` 的修改属于本次整理前的工作区状态；本文引用绑定当前源码的 `results/compact_913_packed/` 图，不覆盖它们。
- 历史旧笔记保留原文供追溯；其过强推断以本文第 6 节的证据范围为准。
