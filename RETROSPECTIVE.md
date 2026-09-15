# 优化复盘:147734 → 986 cycles(150x)

> 对 perf_takehome 整个优化过程的整理与反思。定量数据来自各 commit 实测与
> 最终 bundle 计数;定性分析分 top-down(资源模型)与 bottom-up(指令/DAG)两个视角;
> 最后是对优化方向的反思:盲点、误判、以及 986 → 959 → 869 的剩余差距。
> 相关文档:`OPT_NOTES.md`(工作日志)、`SOLUTION_959.md`(用户提供的 959 参考设计)。

## 1. 历程总览(全部实测,测试参数 10/2047/256/16)

| 阶段 | commit | cycles | 相对变化 | 手段 |
|---|---|---|---|---|
| 参考实现 | — | 147734 | 1x | 标量、逐元素、每元素一条 bundle |
| 向量化 + VLIW | (早期) | 1182 | 125x | 8-lane SIMD、全展开、serial SGS 调度器 |
| knob 扫描 | (早期) | 1185→1163 | | SCALAR_MOD、vaddr alu 链、常数派生 |
| C6DEF | 0caccab | 1163 | | hash S6 的 ^c6 递延到下一轮 node xor(7 轮) |
| offload 扫参 | 01170ef | 1157 | -0.4% | Bresenham OFF_P/Q、RUSH;**自评"已饱和"** |
| M0 vreg+绑定 | c89f806 | 1160 | +3 | 虚拟寄存器 + 调度后 linear-scan(使能器) |
| M1 tournament | bf42b20 | 1151 | -9 | d≤3 线性扫描→锦标赛,valu -234 / alu -672 |
| M2 预异或 | a3a4fa6 | 1169 | +18 | d4-7 节点预 ^c6 入 mem 尾部,递延扩到 12 轮 |
| M3 d=4 blend | 6a58907 | 1085 | -84 | d=4 部分 blend + branch bit 直接做 cond |
| M4 再平衡 | 72cdf8b | 1069 | -16 | OFF=1/3、NG4=6、静态 slimming |
| M4.5-4.8 | f825a5f… | 1057(动态) | | madd 折叠、const 去重、死 p-update 消除 |
| **M5 迭代重调度** | 60729b7 | **986** | **-83** | forward/backward 交替 + jitter,800 basins,离线嵌入 |

三个关键观察:

1. **M1:计数大胜,cycles 几乎不动**(valu -234 只换 -6 cycles)——当时瓶颈不在计数。
2. **M2:计数赢,cycles 反而退**——op 删除改变了 DAG 形状,旧调度器打包变差。
3. **M5:同一个 DAG 零改动,-83 cycles**——此前所有"plateau"结论都要打问号。

## 2. Top-down 视角:资源模型( roofline )

### 2.1 机器是个吞吐机器

每 cycle 固定容量:valu 6 槽(×8 lanes)、alu 12 槽、load 2 槽、flow 1 槽、store 2 槽。
优化只有三个杠杆:**删 op(减总量)、移 op(跨引擎平衡)、排 op(调度压实)**。

### 2.2 平衡演进(每引擎 slots → 计数下界)

| 引擎 | 1157 版 | 下界 | 986 版 | 下界 | 利用率 |
|---|---|---|---|---|---|
| valu | 6456 | **1076c ← 瓶颈** | 5800 | 967c | 98.0% |
| alu | 10492 | 874c | 11372 | 948c | 96.1% |
| load | 2104 | 1052c | 1924 | 962c | 97.6% |
| flow | 704 | 704c(42% 空转) | 908 | 908c | 92.1% |
| store | ~60 | ~30c | 62 | 31c | 3.1% |

1157 版的病:**valu 独大、flow 近半空转**。986 版的药:把 valu 的 656 个 op
拆开——~300 删(tournament/pre-xor/madd 折叠)、~230 移去 flow(blend 化)、
~120 移去 alu(flex offload)——四引擎收敛到 908-967c 的 6% 带内。

### 2.3 下界的层次(这是本任务最重要的认知)

- **设计下界 ≠ 问题下界**。我曾推导"load 硬下限 1024c"(8 gather 轮 × 256 元素),
  并据此判断 <1000 不可行。错。那只是"全 gather"这一个设计的下界;
  d=4 blend 化把 load 转移到 flow 后,load 下界降到 962c。
- **计数下界 ≠ 可达值**。986 vs valu 下界 967,slack 19c 来自依赖链 +
  ramp/尾部排空;959 文档同一下界 954 只留 5c slack——调度质量是独立的一维。
- **每换一个 DAG(算法表示),三个数字(总量/平衡/可达 slack)全部重估**。
  对旧设计做的下界分析,对新设计一律作废。

## 3. Bottom-up 视角:指令与 DAG 级技术清单

按贡献排序(增量为实测,受调度交互影响不严格可加):

### 3.1 删除类(减总量)

- **hash 代数压缩(11 vops/轮)**:仿射级融合 `(x+C)+(x<<s) = x*(2^s+1)+C`;
  S3+S4 跨级融合(2 madd + 1 xor,c34/c35/m169 预折叠)。16 轮 × 32 向量,
  相比朴素 18 ops 省 ~3500 vops。这是 125x 之后一切精细优化的地基。
  (SMT 搜索已证:3 条目标 ISA op 内无更短等价式——XOR 与模加法混合
  不是仿射变换,这是 hash 的理论极限。)
- **c6 递延 + 预异或(M2)**:hash 末级 `(s^c6)^(s>>16)` 的 ^c6 是异或常数,
  可推迟到下一轮与 node 的 xor 合并。把 d4-7 节点(240 个)setup 时预 ^c6
  存入 mem 尾部,递延从 7 轮扩到 12 轮:省 ~160 valu + 消掉边界转换 op。
  代价:30 组 vload/xor/vstore(可与 ramp 重叠)。
- **tournament 取代线性扫描(M1/M1.5)**:d≤3 查表从"cond_k=cond_{k-1}^Δ 异或链
  + 每叶一次 vsel"(d=3:7 xor + 7 vsel)改为"branch bit 直接当 cond 的锦标赛"
  (3 个可下放 alu 的 `&` + 7 vsel):valu -234、alu -672。
- **madd 折叠(M4.5)**:gather 入口的 xor+add 两步并一条 multiply_add;
  死 p-update 消除(round 12-14 对 round 15 blend 向量的无效路径更新,-9 valu)。
- **常数瘦身(M4.6)**:sconst 按值去重、小常数全部 alu 派生(0 额外 valu)、
  vaddr 32 个 const load → 1 const + 31 次 alu 链。

### 3.2 转移类(跨引擎平衡)

- **flex offload(Bresenham → 动态)**:`^/>>/&/+` 按比例 P/Q 拆成 8 条标量 alu。
  目标不是固定比例而是 V/6 ≈ A/12(最终 967 vs 948)。
- **d=4 部分 blend(M3)**:round 4 的后 28 个向量、round 15 的后 3 个向量,
  用 16 路锦标赛(预异或 d=4 广播表)替代 gather:每向量 -8 load、+7 flow、
  +8 flex(L1 静态 diff 可 madd 上 valu,也可 vsel 上 flow,L1MADD_N 按向量选引擎)。
  这把 load 从 1052c 压到 962c,同时把 flow 从 704c 喂到 908c——一次移动,
  两个引擎同时受益。**这是 top-down 平衡分析直接指导出的最大单笔结构改动(-84)。**
- **mflex/vmsel**:同一 blend 逻辑保留 flow vselect 与 valu madd 两种候选,
  调度期按当拍引擎余量选边——灵活 op 不许抢占不可替代 op 的槽位。

### 3.3 调度类(压实)

- **serial SGS(Kolisch)**:Kahn 拓扑序 + 逐 op 放最早空位,天然跨 group 软件流水
  (G0 hash | G1 gather | G2 load …),优于 cycle-by-cycle 贪心。
- **vreg + linear-scan 绑定(M0)**:计算图用虚拟寄存器,调度后按
  [first_write, last_use] 区间复用物理槽。本身 -0 cycles,但它解除了
  "scratch 1536 字"对 DAG 形状的钳制——M1/M3 的锦标赛都要 3 个滚动临时向量,
  固定分配下放不下。**基础设施松绑 > 局部 trick。**
- **迭代重调度(M5,单笔最大 -83)**:固定 DAG,交替 forward/backward 调度 +
  按 (group, phase) 加一致性 jitter 换搜索盆地,800 basins 取最短。
  离线跑(分钟级),最优表 pickle→zlib→base85 嵌进源码,提交时 0.24s 解码;
  其他 shape 回退动态调度。**把"搜索预算"和"提交运行时间"解耦,是解锁重搜索的前提。**

## 4. 过程反思:盲点与误判

这是复盘的核心。我在 1157 自评"knob 全扫过、已 plateau、<1000 在我推导范围内不可行",
被 959 文档证伪。逐条解剖当时的盲点:

### 盲点 1:把一个实现的失败当成一个方向的失败
我用**线性扫描**实现过 d=4 flow 化(D4_FLOW),flow 爆炸,结论"doesn't pay, keep 0"。
959 文档用**锦标赛 + branch bit 直接做 cond + L1 放 valu** 实现同一想法,成了最大单笔收益。
教训:否定一个方向前,要区分"想法不行"还是"这个实现不行"——尤其是在
资源配比不同的实现之间(flow 贵 vs flow 闲)。

### 盲点 2:把自己基础设施的限制当成机器的限制
我早就知道锦标赛省 cond,但以"需 4-5 块/vector、scratch 放不下"否决——
那是**固定地址分配**的限制,不是 1536 字本身的限制。vreg+linear-scan 是
通行解法,我直到拿到文档才建。教训:当资源墙挡住一类算法时,
先问"这堵墙是物理的还是我自己砌的"。

### 盲点 3:笔记里躺着答案却没连上线
我的 OPT_NOTES 早就记录了"mem 尾部有 extra_room ≈2607 字可读写",
但 c6 递延我一直用复杂的 pbar 补码簿记在 scratch 内腾挪,从没想过
把预异或节点**写回 mem** 让递延变成结构性事实。资源清单要和问题定期对照,
不是收集起来就算。

### 盲点 4:把参数族饱和当成设计空间饱和
"knob 扫描全部 plateau 在 1157-1163"——那只是 (KA, KB, RUSH, OFF) 这个
低维参数族的 plateau。同一 DAG 上 per-op 优先级的 full search(M5)直接 -83。
教训:报告 plateau 时必须注明搜索族的维度和覆盖度;"扫完了"三个字
只能用于枚举尽了的空间。

### 盲点 5:没想过把搜索时间从提交时间里拆出来
离线嵌入表(pickle/zlib/base85 + shape 命中 + 动态回退)让调度搜索预算
从 20 秒提交时限解放到分钟级。工程上的 time-shift 是优化方法论的一部分,
不只是 trick。

### 盲点 6:在错误的瓶颈上优化计数
M1/M2/M4.5 三次"计数赢、cycles 不动甚至倒退"已经反复提示:
当时的 binding constraint 是调度打包,不是 op 数。我却继续做计数优化,
因为计数可测、调度难测。教训:**先判定瓶颈在三个杠杆(删/移/排)的哪一个,
再选工具**;可测性不代表重要性。

### 做对了什么
- 每一步都有 profile(分引擎计数 + 分窗口利用率),方向判断以数据为准;
- OPT_NOTES.md + 高频 commit,扛住了两次 context compact 和一次 2h 子代理超时,
  状态零丢失;
- 快内环:do_kernel_test ~1.5s + rounds 二分定位错误轮,正确性从不靠祈祷;
- 约束干净:tests/、problem.py 全程零改动,9/9 绿贯穿始终。

## 5. 剩余差距:986 → 959 → 869

- **vs 959 文档**:counts 已对齐(valu 5800 vs 5719、load 1924 vs 1891、
  flow 908 vs 946),差距全在调度 slack(19c vs 5c)。4000-basin 搜索显示
  slack 不随 count 下降——需要的是更强的调度搜索(或更扁的 DAG),不是更多 op 删除。
- **vs 排行榜 869**:比 valu 计数下界 967 还低 98c,即需要把 valu 砍到 ~5200 slots
  (-600)而不撑爆其他引擎。在我和 959 文档都已知的技术集合内不存在这样的删减;
  hash 代数压缩已被 SMT 证否。最大嫌疑是一种**降低 gather+blend 总工作量的
  lookup 表示**(例如跨 lane/跨轮聚合相同地址的访问),但这需要运行时地址碰撞率
  足够高且有便宜的聚合机制——在本 ISA(无 vperm、无 scratch-gather)下形态不明。
  这是留给后续的唯一开放问题。

## 6. 一句话总结

125x 来自向量化和 VLIW(把问题塞进机器模型);1157→986 来自
**先建资源账本(top-down)定方向,再用 DAG 变换(bottom-up)执行,
最后用调度搜索兑现**;而最大的认知税,是两次把"我的实现/搜索族的天花板"
误报成"问题的天花板"。
