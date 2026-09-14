# VLIW Tree-Hash Kernel: 959 Cycles (用户提供的设计文档)

> 来源:用户在 goal-mode 对话中粘贴的完整方案文档(疑似出自另一 AI 会话/公开 writeup)。
> 本文档原样存档,作为把本仓库 1157 -> <1000 的实施规格。以下为文档原文。

本文记录当前 perf_takehome.py 的完整算法、计算图变换、VLIW 调度策略、 已经验证过但无收益的方向,以及离线调度表的生成和提交方式。

当前标准测试参数:

    forest_height = 10
    n_nodes       = 2047
    batch_size    = 256
    rounds        = 16
    VLEN          = 8
    groups        = 256 / 8 = 32

最终结果:

    CYCLES: 959
    Speedup: 147734 / 959 = 154.05x

## 1. 来源与约束

题目来自 Anthropic original performance take-home。

必须遵守:

- 不修改 tests/。
- 不修改 problem.py 中的硬件定义、参考 hash 或核心数量。
- 最终结果必须通过多个随机 seed 的参考实现比较。
- N_CORES = 1 是题目约束,不可通过修改测试或硬件模型获得虚假加速。

## 2. 原始算法

每个输入包含当前节点索引 idx 和值 val。每轮执行:

    node = tree[idx]
    val = myhash(val ^ node)
    idx = 2 * idx + (1 if val is even else 2)
    if idx >= n_nodes:
        idx = 0

树高为 10,共 2047 个节点。16 轮遍历的 depth 是静态的:

    0, 1, 2, ..., 10, 0, 1, 2, 3, 4

因此 kernel 不需要真正执行循环、深度比较或越界分支。所有 16 轮和 32 个 SIMD group 都可以完全展开,调度器再把不同 group 的 load、hash、lookup 和 store 交错执行。

## 3. 硬件模型

每周期各引擎的槽位:

| Engine | Slots/cycle | 主要用途 |
|---|---|---|
| ALU   | 12 | 标量加减、移位、异或、按位与 |
| VALU  | 6  | 8-lane 向量操作、multiply_add |
| Load  | 2  | const、load_offset、vload |
| Store | 2  | store、vstore |
| Flow  | 1  | vselect、add_imm、pause |

其他约束:

    VLEN         = 8
    SCRATCH_SIZE = 1536 words

机器没有 bundle 内转发。周期 t 写出的结果,依赖者最早只能在周期 t+1 使用。 这使得"把一条操作拆成两条、但声称同周期完成"的优化通常无效。

## 4. 当前计算图

### 4.1 8-lane SIMD group

256 个输入按 8 个 lane 分成 32 个 group。每个 group 的 w、路径状态和临时值都是 向量。一个 VALU 指令同时处理该 group 的 8 个输入。

注意:这已经是正确的 lane 打包方式。把 8 个 group 再塞进同一向量不会让指令数除以 8, 因为当前每条向量指令本来就已经处理 8 个独立输入。

### 4.2 不保存完整 idx

实现不在每一轮维护传统的 idx 向量,而是保存 hash 产生的 branch bit,并维护路径描述符:

- tb[r]:第 r 轮的分支位。
- P:累积路径编码。
- A:已经转换为真实树地址的 gather 地址。

进入 gather 阶段后,地址使用递推式更新:

    A_next = 2 * A - (tree_p - 2) - tb

对应代码将前半部分提前为 multiply_add:

    B = 2 * A + neg
    A = B - tb

这样 tb -> address 的依赖链只有一层,而不是先更新 path 再单独转换地址。

### 4.3 P-fold

尾部部分 group 使用 pfold = 29。它省掉 round 13 的一个路径 multiply_add, 在 round 14 用 vselect 与一个 -4 * P 地址式恢复 gather base。

本质是用尾部相对空闲的 Flow 槽换掉拥挤的 VALU 槽,并缩短尾部活跃值的生命周期。

## 5. Hash 压缩

原始 hash 有六个固定 stage:

    HASH_STAGES = [
        ("+", 0x7ED55D16, "+", "<<", 12),
        ("^", 0xC761C23C, "^", ">>", 19),
        ("+", 0x165667B1, "+", "<<", 5),
        ("+", 0xD3A2646C, "^", "<<", 9),
        ("+", 0xFD7046C5, "+", "<<", 3),
        ("^", 0xB55A4F09, "^", ">>", 16),
    ]

### 5.1 加法加左移融合为 multiply_add

恒等式:

    (x + C) + (x << s) = x * (2^s + 1) + C  mod 2^32

因此 stage 0、2、4 可以分别化为乘加。例如 stage 0:

    x * 4097 + C0

一条 multiply_add 代替加法、移位和加法组合。

### 5.2 Stage 2 + 3 跨 stage 融合

设 stage 2 的结果:

    y = 33*x + C2

stage 3:

    (y + C3) ^ (y << 9)

可改写为:

    u = 33*x + (C2 + C3)
    v = (33 << 9)*x + (C2 << 9)
    result = u ^ v

当前 _chain() 用两条 multiply_add 加一条 XOR 完成这两个 stage。

### 5.3 延迟最后一个 XOR 常量

最后 stage 原本包含:

    (x ^ C6) ^ (x >> 16)

除最后一轮外,^ C6 延迟到下一轮的 node XOR:

    (hash_without_C6 ^ C6) ^ node
    = hash_without_C6 ^ (node ^ C6)

这不会改变语义,却能让每个中间 round 少一个 hash-time XOR。因为 C6 是奇数, 内部保存的 branch bit 与参考表示相反,代码中的 tb 和路径公式已经按这个表示处理。

最后一轮必须恢复真实输出,因此会显式应用 C6,并将它和右移并行安排以缩短尾部。

### 5.4 SMT/Superoptimizer 结论

使用 Z3/cvc5 对多个模板做过等价搜索,包括:

- 吸收 XOR 常量到前一条 affine madd。
- 用更少操作表达 XOR + shift stage。
- 用两条 affine madd 分支加 XOR 表达更长的 stage 组合。
- 搜索不超过三条目标 ISA 操作的相邻 stage 表达式。

没有找到比当前表示更短、且对全部 32-bit 输入严格等价的表达式。 XOR 与模 2^32 加法混合后不是单纯的 affine 变换,这是 hash 继续代数压缩的主要障碍。

## 6. Tree Lookup: Blend 与 Gather

### 6.1 Gather

深层节点地址在 8 个 lane 中通常互不相同,因此真正的 gather 是:

    for lane in range(8):
        load_offset(node, addr, lane)

每个向量 gather 消耗 8 个 Load 槽。

vload 不能直接替代它。vload 接收一个标量基址,只能读取连续的 8 个 word; load_offset 则读取 addr[lane] 指向的 8 个独立位置。除非运行时地址刚好连续,否则两者不等价。

### 6.2 Shallow Blend

浅层节点数量少,可以在 setup 阶段一次性读取并广播,然后使用历史 branch bit 选择节点。

一对左右节点可以表示为:

    selected = bit * (left - right) + right

这既可以放到 VALU 的 multiply_add,也可以放到 Flow 的 vselect。当前 vmsel() 把两种实现同时保留为候选,调度器根据当周期资源决定放置位置。

更多层的选择使用 tournament:每层根据一个旧 branch bit 把候选数量减半。

### 6.3 当前 mode 策略

当前关键参数:

    ng3  = 2
    ng4  = 4
    nb15 = 3

- 根节点直接使用缓存广播。
- 浅层优先 blend,避免随机 load。
- 深层优先 gather,避免庞大的选择表。
- 最早几个 group 在 ramp 阶段主动 gather depth 3/4,利用早期 Load 空槽。
- 最后 3 个 group 在 round 15 使用 blend,让收尾阶段用空闲 Flow 换掉 gather 尾链。

这不是"blend 永远优于 gather"。增加 blend 会减少 Load,但增加 VALU/Flow 操作、依赖层级、 广播表和寄存器压力;当前参数是四个引擎接近平衡后的折中点。

## 7. 预处理树节点

depth 4 到 depth 7 的节点在 setup 阶段执行:

    vload -> XOR C6 -> vstore

后续 gather 直接读取已经 ^ C6 的节点,从每次 gather 后删除一个 flex XOR。

代价是 setup 的 30 组 vload/xor/vstore,收益是删除中段大量重复 XOR。setup Load 和 Store 可以与后续 group 的计算重叠,整体比每次访问后再 XOR 更便宜。

depth 3 在启用早期 gather 时也做对应的预处理。浅层 root/blend 表则保留 raw 和 adjusted 两种表示,分别服务第一轮和延迟 C6 的后续轮次。

## 8. ALU/VALU/Flow 联合调度

### 8.1 flex

简单向量操作同时生成两种候选:

- VALU: 一条 8-lane vector op
- ALU:  八条 scalar lane op

调度器可将 XOR、shift、AND、加减法放到 VALU,或拆到 ALU 的 12 个槽中。ALU lane 操作允许分块发射,不要求一个周期一次容纳完整 8 lane。

强制全部 ALU-first 会变慢,因为 ALU 还承担地址计算、广播和标量常量构造。当前策略使用 flex_valu_frac = 0.80 作为动态平衡目标,而不是按 opcode 固定引擎。

### 8.2 mflex

blend 的同一逻辑可选择:

- Flow: vselect
- VALU: multiply_add(bit, left-right, right)

单周期中先安排只能使用单一引擎的指令,再让 flex/mflex 填充剩余容量,避免灵活指令 抢走不可替代指令的槽位。

### 8.3 常量与广播

setup 阶段 Flow 很忙,而 ALU 有空闲,因此小整数常量使用 ALU 加减构造,向量广播也可 展开为 8 条 ALU copy。大常量和第一条 hash 关键路径上的常量仍走快速的 Load/VALU 路径。

输入地址只加载第一个 base,后续 31 个 group 用标量 ALU +8 链生成,省掉 31 个 const Load 槽。

## 9. DAG 与软件流水

### 9.1 IR

每个操作是一个 Op,包含:

- scratch/虚拟寄存器读写集合。
- 可使用的 VALU、ALU、Load、Store 或 Flow 指令模板。
- 前驱和后继依赖。
- 调度位置与资源占用。

last_writer 根据读写位置构造 RAW/WAW/WAR 依赖。树内存预处理的 store 与后续 gather 之间没有天然 scratch 依赖,因此代码额外加入显式 predecessor,保证内存可见顺序。

### 9.2 跨 group 软件流水

源码虽然按 group 生成 IR,最终执行并不是:

    G0 全部完成 -> G1 全部完成 -> ...

调度器会形成跨 group 的软件流水:

    G0 hash | G1 gather | G2 lookup | G3 input prefetch | G4 setup/store

输入 vload 使用较高 priority boost,主动填充 setup Load 结束后的早期空洞。 group priority 形成平滑的对角线,而不是在 group 边界排空所有引擎。

### 9.3 迭代重调度

动态调度流程:

1. 使用 DAG height 做第一次 list scheduling。
2. 记录每个 op 的真实开始周期。
3. 交替执行 backward/forward scheduling。
4. 对整个 (group, phase) 添加小幅一致性 jitter,改变搜索 basin,但不打散软件流水。
5. 每次尝试寄存器绑定,只保留 bind-feasible 的最短程序。

默认参数:

    resched = 900
    jseed   = 17
    jamp    = 0.8
    boost   = 8.6
    pfb     = 300

一次完整动态搜索约需 40 秒,因此最终提交不在服务器上重新搜索。

## 10. 虚拟寄存器与调度后绑定

计算图构造时使用虚拟向量寄存器,不提前固定 scratch 地址。调度完成后,根据每个虚拟值的:

    first_write ... last_use

执行 linear-scan 分配,将生命周期不重叠的向量复用同一 scratch 槽。

这解决了两个问题:

- 调度器可以同时保留更多候选值,不被手工 tmp pool 过早限制。
- 浅层 blend 表、hash 临时值和尾部值在生命周期结束后能及时复用 scratch。

每次候选调度都必须通过 _bind();无法放入 SCRATCH_SIZE=1536 的调度不会成为 best。

## 11. 离线调度表

标准测试的输入 shape 固定,因此最优调度可以离线生成一次,并直接嵌入源码。

当前格式:

    Python instruction bundles
        -> pickle protocol 4
        -> zlib level 9
        -> Base85 string

build_kernel() 命中:

    (10, 2047, 256, 16)

时直接解码 959 个 bundle。其他 shape 仍保留动态 DAG 调度器作为 fallback。

内嵌表最初由同一计算图的 1023-node shape 生成。对比重新生成的 2047-node 版本后发现, 959 个 bundle 中只有输入 values 基址这一条立即数不同:

    1286 -> 2310

因此代码在解码后使用:

    7 + n_nodes + batch_size

修正该基址。该等价性已通过逐 bundle 比较和 8 个随机 seed 的完整测试验证。

### 11.1 曾经的超时问题

离线 shape 一度误写为 (10, 1023, 256, 16),真实测试却是 2047 个节点。服务器未命中 离线表,于是执行 900 次动态重调度,触发:

    timed out after 20 seconds

修正 shape 后:

    JSON program generation: about 0.24 s
    Full 9-test suite:       about 0.54 s

## 12. 当前资源下界

959-cycle 离线程序的实际 slot 总数:

| Engine | Slots  | Capacity | Count lower bound |
|---|---|---|---|
| ALU   | 11392 | 12/cycle | 950 |
| VALU  | 5719  | 6/cycle  | 954 |
| Load  | 1891  | 2/cycle  | 946 |
| Flow  | 946   | 1/cycle  | 946 |
| Store | 63    | 2/cycle  | 32  |

单看资源计数:

    lower_bound >= max(950, 954, 946, 946, 32) = 954 cycles

实际是 959,仅比资源下界多 5 个周期。剩余差距可能来自真实依赖链、启动/收尾以及不同引擎 槽位无法在同一依赖层中任意交换,而不只是普通 list scheduler 的空洞。

当前主要操作计数:

    load_offset       1816
    vload               67
    VALU multiply_add 2644
    Flow vselect        934
    ALU XOR            6864 scalar lanes
    VALU XOR           1865 vector ops

要稳定突破 959,优先级应是删除真实操作或改变 lookup 表示,而不是继续大范围扫调度参数。

## 13. 已尝试但没有形成收益的方向

### 13.1 将 gather 合并为 vload

不可直接做。8 个 lane 的地址是动态且独立的,vload 只能读连续地址。将 lane 排序后再 vload 还需要动态排序、排列并恢复原顺序,而 ISA 没有 vperm,开销通常大于节省。

### 13.2 按 node/path 对输入分桶

理论上同节点输入可以共享一次 load,但需要运行时检测、桶边界、lane permutation、原位置恢复 和额外 store/load。浅层节点已经由 blend 解决;深层节点分散,碰撞率不足以覆盖分桶成本。

### 13.3 将 multiply_add(x, 2, c) 改成 shift + add

会把一条 VALU 指令变成两条有 RAW 依赖的 ALU 指令。机器没有同 bundle 转发,因此依赖链 确实增加一周期;在 ALU 已接近 950-cycle 下界时也没有免费容量。

### 13.4 强制 XOR 全部走 ALU

8-lane XOR 需要 8 个 ALU slot。局部迁移有价值,所以当前保留 flex;全局 ALU-first 会让 ALU 过载,并挤压地址计算、常量生成和广播。

### 13.5 奇偶位旁路

可以利用:

    (x ^ (x >> 16)) & 1 = (x & 1) ^ ((x >> 16) & 1)

但它增加多条 ALU 操作,且由于无 bundle 内转发,并未缩短可见依赖周期。实测资源压力上升。

### 13.6 投机计算左右地址

同时计算 left/right 再 select 会增加 madd、减法和选择操作。Flow/VALU/ALU 都接近满载, 增加的操作无法被隐藏,且 branch bit 仍然需要等待 hash。

### 13.7 更大的浅层或 depth-5 全 blend

可以删除 Load,但会生成大量 pair table、tournament select 和长生命周期向量。结果通常从 Load 瓶颈转成 VALU/Flow/寄存器瓶颈,并恶化关键路径。

### 13.8 完整深层 cache

即使命中率 100%,运行时仍需根据动态 idx 做选择。没有 gather-like scratch lookup 或 vperm, cache hit 并不等于免费读取;选择网络成本可能比内存 load 更高。

### 13.9 修改 hash 函数

不允许。测试与参考实现要求全部 32-bit 语义严格一致,不能用 avalanche 相近的简化 hash。

### 13.10 单纯增加 resched 次数

可能偶然改善 1 到数个 cycle,但 959 已距资源下界 5 cycle。它不会突破指令计数下界, 而且在线生成会触发提交超时。更重搜索应离线运行,只嵌入最优结果。
