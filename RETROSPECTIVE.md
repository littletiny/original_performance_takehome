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
  存入 inp_indices 死区(mem 私有副本、提交只查 val 区),递延从 7 轮扩到
  12 轮:省 ~160 valu + 消掉边界转换 op。代价:30 组 vload/xor/vstore
  (可与 ramp 重叠)。
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
我的笔记里早就记录了 mem 里有可写空间,但 c6 递延我一直用复杂的
pbar 补码簿记在 scratch 内腾挪,从没想过把预异或节点**写回 mem**
(实际可用的是 inp_indices 死区 256 字:mem 是私有副本、提交只查 val 区)
让递延变成结构性事实。资源清单要和问题定期对照,不是收集起来就算。

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

## 7. 附录:控制流维度(indirect jump / 代码段)评估

问题:引入 `jump_indirect`(用 scratch 的值当 pc)和代码段,能不能找到新思路?
结论:**省不了 cycle,但找到了 1 拍的真实微优化(pause 合并,986→985)**。

### 7.1 结构性排除(四条机器事实,problem.py 可验)

1. **计费模型**:cycles = 执行过的非 debug bundle 数(run() 里
   `has_non_debug → cycle += 1`)。跳转只改 pc,不减少任何已执行槽位;
   跳转本身还占 flow 槽(flow 已 92% 利用,是最不闲的引擎)。
2. **单一全局 pc**:N_CORES=1,256 个 lane 共享同一个程序计数器。
   数据依赖的分发(computed goto)无法按 lane 分流——而本工作负载的
   全部动态性恰恰在 lane 维度(每个元素的 branch bit 不同)。
3. **程序不可写**:store 只写 mem,program 列表运行时不可变,
   自修改代码不存在。
4. **slot 操作数是编译期立即数**:valu/alu 的 scratch 地址在 build 时烤死,
   没有寄存器间接的 scratch 访问 → **无法对"第 g 组的数据"循环**
   (循环体引用不了动态的 scratch 块)→ 连"用循环压缩代码段"都做不到,
   直线型代码不是选择,是被 ISA 强制的(只有 load/store 的 mem 地址是动态的)。

### 7.2 经典 indirect jump 用法逐条评估

| 用法 | 判定 | slot 账 |
|---|---|---|
| 循环复用代码段(32 组 × 16 轮) | 不可行 | valu/alu 操作数静态,循环体无法按组索引(事实 4);且循环控制加 flow op |
| dispatch 表代替 vselect/gather | 被支配 | 跳转是标量+全局:每次查表 ≥1 flow 槽/元素 = 4096 槽 ≫ 现 flow 总量 908 |
| 代码嵌入节点值(const 跳转表) | 不可能 | 树值每个 seed 运行时随机生成,编译期(build_kernel)拿不到 |
| 数据依赖跳过工作 | 无可跳 | 深度序列静态(无 wrap 检查可跳);每 lane 都要 16 轮精确 hash;hash 无值相关捷径(SMT 已证最小) |
| 自修改代码 | 不存在 | 事实 3 |
| Duff 式可变长度入口(组间交错) | 负收益 | 替掉 ~32 条 dummy alu(≈0.3c)要花 flow 槽 |

### 7.3 关键洞察:vselect 就是这台机器的 indirect jump

per-lane、数据依赖、1 槽管 8 lane 的控制原语已经存在——就是 flow 引擎的
`vselect`。它在 blend/tournament 里已用到 92% 饱和。jump 族(标量、全局、
占同一引擎)在查表/分发场景下被 vselect 严格支配。这不是思路不够,
而是这个维度在模型里封闭。

### 7.4 唯一榨出的一滴油:pause 合并(986 → 985,已提交 555bbbc)

harness 要求的首个 `pause` 原来独占一个 bundle(白付 1 cycle)。
中间断言只查 inp_values(末尾 vstore 才写),提交 harness 干脆
`enable_pause=False`——所以把 pause 并进首个有空 flow 槽且无 store 的
bundle 即可,9/9 绿,省 1 拍。这是控制流维度在本题的全部正收益。

### 7.5 什么条件下 indirect jump 才会翻盘

- per-lane pc(SIMT 式发散)——即便有,逐 lane 跳转查表的 slot 账仍输给 vselect;
- 可写程序内存(自修改/运行时特化)——但离线嵌入表已捕获全部编译期特化;
- 编译期已知的树值——被测试的随机 seed 机制排除。
三条都在题目约束之外,佐证结论的稳健性。

## 8. 附录:rounds 11-15 还能否压缩计算量(专项评估)

结论:**不能,已在三道地板上;该窗口的查表已是"零 load + 一次 gather"**。
逐轮实测(当前 kernel,按 op tag 分桶):

| round | d | valu | alu | load | flow | 查表 |
|---|---|---|---|---|---|---|
| 11 | 0 | 299 | 680 | 0 | 0 | 根广播 xor |
| 12 | 1 | 327 | 688 | 0 | 32 | 1 vsel/组 |
| 13 | 2 | 328 | 680 | 0 | 96 | 3 vsel/组 |
| 14 | 3 | 328 | 680 | 0 | 224 | 7 vsel/组 |
| 15 | 4 | 351 | 688 | 232 | 21 | 29 组 gather + 3 组 blend |

三道地板:
1. **串行依赖**:round r 的查表地址由 round r-1 哈希的奇偶位决定,子遍历
   不可折叠/预取/合并(预取=load 翻倍,16 路径预算=16 倍哈希)。
   实测:全 DAG 关键路径 201c;组 0 的 r11 最早 op 在 cycle 137、r15 在 184
   (ASAP 无限资源下界)——ramp 段(0-80)里 r11-15 的工作尚不存在,无法用于
   填充;中段 valu 满载后,提前执行等价于挤占其他 op,总拍数不变(吞吐守恒)。
   r11 对 r10 仅"位置/节点"无依赖(wrap 无条件、d=10 不发 p-update),
   "值"仍是 RAW 依赖。
2. **哈希 SMT 最小**(11 vops/轮),每轮需全值,末轮须恢复 ^c6。
3. **选择下界 2^d-1**:锦标赛恰好达到;条件用 `p & 2^j` 或复用上一轮
   p-update 的 bit 向量,已是最便宜形式。

毙掉的方向:
- 更多组 blend(ng4/nb15 右移):valu 966c 是 binding,gather→blend 不降
  valu,load 降了也不是瓶颈 → cycles 不动。当前参数即平衡点。
- 路径打包(16 路径×4 节点连续化):死区只剩 16 字(240 已被预异或表占用),
  且把 r12-14 从 0 load 变成 3 load/元素,反向优化。
- 同向量 lane 地址聚合(d=4 仅 16 节点,碰撞率高):去重需排序/压缩,
  无 vperm,开销 > 节省。

剩余空间不在计算量,而在尾部 ~19c 的调度 ramp-down。

**alu 气泡的定量审计**(可否调度兑现):总空闲 448 槽,但只有 ~244 槽
出现在 valu=100% 的周期(200-400 段 192 + 400-560 段 36 + ramp/drain 少量);
ramp 与 drain 的其余空闲是依赖地板(工作尚不存在)或延迟收尾(链在走),
不可填。把可填气泡全部兑现 = 把 ~18-30 个 flex op 从 valu 移入 alu,
而全局平衡方程给出计数下界仅从 965c → 962c(k=18 时 valu/alu 均 962c):
**气泡填充与再平衡是同一笔账,上限 ~3c**。当前 flex 是构建期静态比率,
感知不到局部气泡;兑现需调度期动态选引擎(dual-form IR)+ 重跑 basin
搜索,ROI  marginal。与 959 的差距主要在计数(对方 valu 少 72 op),
不在排布。

## §9 与 959 方案的逐项定量对比(985 vs 959)

两边嵌入程序的实测操作计数(我方从 `EMBEDDED_SCHEDULES[(10,2047,256,16)]`
逐 bundle 解码统计;对方数字取自 SOLUTION_959.md §12):

| 项 | 我们 (985c) | 他们 (959c) | Δ(我−他) |
|---|---|---|---|
| valu 总数 | 5800 | 5719 | **+81** |
| ├ multiply_add | 2734 | 2644 | +90 |
| ├ xor | 1852 | 1865 | −13 |
| alu 总数 | 11372 | 11392 | −20 |
| load 总数 | 1924 | 1891 | +33 |
| ├ load_offset(gather) | 1816 | 1816 | **0** |
| ├ const | 40 | ~8 | +32 |
| flow 总数 | 908 | 946 | −38 |
| ├ vselect | 907 | 934 | −27 |
| store | 62 | 63 | −1 |
| **资源下界** | **967** | **954** | **+13** |
| 实测 cycles | 985 | 959 | +26 |
| slack(实测−下界) | 18 | 5 | +13 |

解读:

1. **核心计算量完全相同**:load_offset(gather)= 1816 两边一致 —— 树的
   随机访问总量、SIMD 打包方式、hash 表示都一样。大方向无差异。
2. **差异全是"bookkeeping 操作的引擎摆放"**:
   - 他们比我们多 27 个 flow vselect、少 90 个 valu madd:blend 的
     `bit*(l-r)+r` 更多走 Flow 而非 VALU(mflex),加上 **pfold=29**
     (§4.3,我们没有实现)把 29 个组的 round-13 路径 madd 换成
     round-14 的 flow vselect + `-4P` 地址式。这两笔 ≈ 56 个 madd,
     占 valu 差的大头。
   - 我们只有 40 个 const load,他们 ~8 个:小常量/广播他们走 ALU 合成,
     把 Load 槽省到 1891。但我们 load 下界 962 本就不是瓶颈,这笔影响小。
3. **引擎均衡度**:他们 alu/valu/load/flow = 950/954/946/946,四个引擎
   挤在 8 拍内,几乎无免费容量;我们 948/967/962/908,spread 59 拍,
   flow 只用到 92% —— 38 个 flow 槽的闲置正好是对方 vselect 多出来的量。
4. **slack**:他们距下界 5c,我们 18c。对方 900 次带 jitter 的
   backward/forward 重调度搜索比我们的 basin 搜索更充分。

结论:**大思路我们不缺(gather 数相同即 lookup 表示相同),缺的是
一个中等结构技巧(pfold)+ 平衡与调度搜索的极致度**。若补上 pfold
(−29 valu)并把 ~27 个 blend select 移到 flow,valu 下界可降到 ~958,
配合更重的调度搜索,~960 是可及的;再往下(959→869)才需要真正未知的
新 lookup 表示。

## §10 第二轮优化(985 → 目标 960):folds 代数 + flow 定律 + 选择性常量合成

承接 §9 的结论("缺 pfold 类结构技巧 + 平衡极致度"),本轮实际落地的
是等价的自制版本。

### 10.1 入口折叠(FOLD5/FOLDT/FOLD7/FOLDR/FOLD15)

思想与对方的 pfold 相同——**把"位置→地址"转换从 valu madd 挪到 flow
vselect**——但作用点更多:不只是 round 13,而是所有 gather 入口/出口
和尾递归。

代数:地址递推原本是

    addr' = 2*addr + K + bit        (K 为常量,bit ∈ {0,1})

拆成两步是 `t = madd(addr, 2, K)`(valu)+ `addr' = t + bit`(valu)。
折叠后用 vsel 的"按 bit 选常量"能力:

    aux   = vsel(K+1, K, bit)       (flow,1 槽)
    addr' = madd(addr, 2, aux)      (valu,1 槽)

净效果:**每个折叠点 −1 valu、+1 flow**,且依赖链不变(bit 仍是最后
到达的输入)。r7 出口原本 3 个 valu(sub + madd + add)折成
1 madd + 1 vsel。5 个开关合计把 valu 下界从 967 压到 954。

定性教训:§9 把 pfold 当成"对方独有的技巧",其实它只是
"flow 引擎能按 bit 免费做 ±1 常量选择"这一个事实的一个应用。
**先提炼机器能力(free const-select on flow),再枚举所有应用点**,
比照搬对方的具体变换更系统。

### 10.2 C = flow_total + flow_idle(实测定律)

本轮对调度质量做了精确归因。对 flow-bound 的程序实测:

    CYCLES = flow 总操作数 + flow 引擎空闲槽数

且 "idle-with-supply"(有 ready 的 flow 操作却仍空闲的槽)= 0 ——
交替前后向搜索在给定窗口下已经是**完美的 flow 打包器**。这意味着
继续堆调度搜索次数(800 局收敛 982-984)不会突破;要降 cycles 只有
两条路:减 flow 操作数,或减结构性的 flow idle(ramp ~31c 等首个
hash 链、中段 r10-12 wrap dip ~30c、drain ~15-18c 末组收尾)。

这也是判断"还能不能压"的定量标准:当前 folds 后 flow 总数 ~950,
若 idle 不变(~79),预期 ~1029?? —— 实测远好于此,因为 folds
同时缩短了 ramp 关键链(first flow op 提前)。**结构改动既减总数
也减 idle,双重收益**,这是它比纯调度搜索强的原因。

### 10.3 CONST_ALU 的选择性版本(回归与修正)

全量 CONST_ALU(所有小常量走 alu 合成链)把 load 1901→1891(=对方
的 load 数),但**搜索从 981 回归到 985-988**。原因:4097/16896
(hash 第一轮乘子)、forest_p、inp_values_p、c32 这些常量位于
group-0 hash 与输入 vload 的 ramp 关键路径上,const load 只要 1 拍,
alu 合成链要 3-7 拍,ramp 被拉长 = flow idle 增大(§10.2 的 ramp 项)。

修正为选择性版本:只有**晚用常量**(MOD32−1、forest_p+255、extra_p、
17−extra_p)走 alu 派生,ramp 关键常量保留 1 拍 const load。
结果:load 1896(仍比全量前省 5),ramp 链恢复原状。

定性教训:**资源计数(load −10)和关键路径(ramp +N 拍)要分开算账**。
"alu 有空闲所以把常量搬过去"在稳态成立,在 ramp 区不成立——ramp 区
的约束是依赖深度,不是吞吐量。

### 10.4 本轮已证伪/确认的方向(补充 §4)

| 方向 | 结果 | 原因 |
|---|---|---|
| NG3=2(更多 d3 gather) | 992,回归 | load 是墙,加 load 必亏 |
| L1MD 高比例 7/13 | 985/987 | 过度移动 flow→valu,valu 先撞墙 |
| NG4=4 | 无益 | 同上类 |
| KF flow 关键性项 | 984 | 已是完美 flow 打包器,优先级项无空间 |
| 纯拉长搜索 800 局 | 收敛 982-984 | 同上,C=flow+idle 定律 |
| 全量 CONST_ALU | 985-988 | ramp 关键常量链变长(§10.3)|
| **folds(§10.1)** | **valu 967→954** | 唯一被确认的大杠杆 |
| **VB_ALU=28** | valu+flow −28 | broadcast 搬 alu,稳态成立 |

### 10.5 当前计数(folds + VB_ALU + 选择性 CONST_ALU 后)

    valu=5725(954c)  alu=11456(955c)  load=1896(948c)  flow=950(950c)

资源下界 955,四引擎 spread 仅 7 拍(原 59)。与对方 959
(954 下界 + 5 slack)相比,我们下界高 1 拍;若调度 slack 能压到
对方的 ~5,cycles ≈ 960 恰好达标。这就是本轮搜索的目标窗口。
