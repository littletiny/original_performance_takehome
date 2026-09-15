# perf_takehome 优化工作笔记(抗 context compact 用)

> 本文档是优化工作的持久化状态记录。每次有实质进展就更新并 `git commit`。
> 目标:`KernelBuilder.build_kernel` < 1000 cycles。测试固定参数
> forest_height=10, n_nodes=2047, batch=256, rounds=16。
> 约束:不改 `tests/`;正确性由 `tests/submission_tests.py` 验证(也打印 CYCLES
> 和各档阈值 1363/1487/1548/1579/1790)。环境用 `python3`。

## 当前状态(2026-09-15,午后)

- HEAD = 794ffa8,**1069 cycles**(动态调度),9/9 绿。
  历程:01170ef 1157 → M0 vreg+linear-scan(c89f806,1160)→ M1 tournament(bf42b20,1151)
  → M2 pre-xor d4..7 到 inp_indices 区(a3a4fa6,1169)→ M3 d=4 partial blend+M1.5(6a58907,1085)
  → M4 offload 再平衡 OFF=1/3+NG4=6(72cdf8b,1069)→ M4.5 gather 入口 xor+add 折成 madd(f825a5f,1069)。
- **M5 已嵌入:986 cycles 离线表(60729b7),9/9 绿,全套 0.65s**。
  - 迭代重调度:jitter Kahn key 只有 1058;**forward/backward 交替(/tmp/m5basin3.py,
    参数化 CFG/seed/输出)是关键**,800 basin → 985(修后 binder 下可行)。
  - 嵌入:base85(zlib9(pickle4(bundles))),EMBEDDED_SCHEDULES dict 按 shape 索引,
    其他 shape 回退动态调度;CFG["USE_EMBEDDED"]=False 可绕开(开发/扫描时用)。
- **M4.6(19bcbb8)**:sconst 按值去重;小常数/表地址/prexor 地址全部改 alu 派生
  (dead-after-setup vreg),prexor 30 块一条 +8 连续链;gen_vaddr 4 const→1 const+3 alu;
  p 活性分析删掉 round12-14 对 round15-blend 向量的死 p-update(-9 valu)。
  counts: valu 5791(966c) alu 11396(950c) load 1901(951c) flow 907;static 517。
- **M4.7(231f1c5)**:L1MADD_N——d=4 blend 的 level-1 按向量选引擎,超过 N 的
  向量用 8 flow vselect 代替 8 valu madd(-8 valu/+8 flow 每向量)。
  L1MADD_N=22: valu 5759(960c) flow 939。
- **M4.8(057a26f)**:NG3——d==3 节点 pre-xor 到 tail 2294(8 字),前 NG3 个向量
  在 round 3/14 走 gather(u = 2301 - pbar,同 d==4 的 xor+add 折 madd)代替 flow
  tournament。默认 0(关)。NG3=2: flow 880(-28)/load 1934(+33)/valu 5797(+6),
  load 下界 967 超过 valu 966 —— 计数变差,只能靠 ramp 填充回本(搜索验证中)。
- 大规模搜索(4000 basins,交替+jitter):D=默认 counts → 985;E=L1MADD_N=21 → 988;
  F=NG3=2(2000 basins)→ 988(进行中,可能再降)。**count 更低≠绝对值更低,
  slack 19-28 不随 count 下降 —— 搜索质量是瓶颈**。
- 新工具:/tmp/polish.py <iters> <seed> <cfg> <in-pkl> <out-pkl> —— 从 incumbent
  暖启动,加噪 backward/forward 交替精调(smoke 通过)。
- 986 的 profile:[100,900] valu/alu/load 三引擎全 ~100%,ramp [0,100] 和
  尾 [900,986] 各 ~90% —— 计数和调度都已接近当前 DAG 的极限,再降必须删真操作。
- 与 959 文档对照:valu 5759 vs 5719(-40),load 1901 vs 1891(-10:const 17 vs 8,
  可用 c5//c2=1 之类派生剩余小常数,但 load 已非约束),flow 939 vs 946。
- 工具:/tmp/m5basin3.py <iters> <seed> <cfg-json|-> <out-pkl>;
  /tmp/finalize_embed.py <pkl> <cfg-json>(5 seed 机器校验后写入 EMBEDDED_SCHEDULES)。
- 调试陷阱(别再踩):reference_kernel2 会 mutate mem,做对照时必须用 copy;
  debug vcompare 放在 instrs[lu+1] 读到的是 lu 拍结束后的值(含同拍 touch 写入),
  读"reader 视角"要放 instrs[lu];递延 round 携带值是 stage5^c6 不是 hashed_val。
- 调试工具:perf_takehome.py 里 DBG_HOOK[0/1/2] 钩子(hook(h,v,val/bit/p) vreg);
  /tmp/dbg_round.py 按 round 二分;build_mem_image 无 tail 空间(见 M2 注)。

## 机器语义(problem.py)

- 每 cycle:6 valu(8 lanes)、12 alu(标量)、2 load、2 store、1 flow。
- 写后下一 cycle 才可见(RAW/WAW 严格更晚,WAR 可同 cycle)。
- flow:`vselect`(8 lanes)、`select`/`add_imm`(标量)。alu 支持全部标量 op 含 `*`。
- valu:`multiply_add` + 全部 alu op 向量版。SCRATCH_SIZE=1536(紧!)。
- N_CORES=1(多核被官方禁用)。hash 6 级(problem.py:449 myhash),S4 是
  `(a+c4) ^ (a<<9)`(不是 +)。

## 关键技术(当前文件内已实现并验证)

- **hash 12→11 vops/round**:S1=madd(a,4097,c1);S2=3 ops;S3+S4 融合=2 madd+1 xor
  (c34=(c3+c4)%2^32, c35=(c3*512)%2^32, m169=33*512);S5=madd(a,9,c5);S6=3 ops。
- **浅层 d<=3 用 flow vselect 线性扫描**(不是二叉树!):acc=sel(p^k,acc,leaf_k),
  cond_k=cond_{k-1}^gv[tz(k)+1], gv={1,3,7,15}。**关键**:只用每 vector 专用
  TT/CT 两块(各 256 字),消除跨链 WAR 耦合(FIFO pool 会造成所有链锁步 1960)。
  二叉树虽省 cond 但需 4-5 块/vector,scratch 放不下,别回头。
- **gather d>=4**:d>=5 直接维护地址 addr'=2addr+(1-forest_p)+bit(3 ops);
  d==4 用 CT 做临时 addr;d==5 时 pp+=basev[5]。d==height(round 10)wrap 无 p-update;
  round 15(last)无 p-update。
- **表 vload 共用 16 字 staging 缓冲**。
- **调度器**:`schedule_ops_serial`(serial SGS,Kahn key=(h*KA+v*KB,sec,i),
  逐 op 放最早有引擎空位的 cycle)最优;`schedule_ops`(T_des paced list)更差,别用。
- **knobs 全在模块级 CFG dict**(perf_takehome.py:40)。

## 测量事实(1185 版)

- profile:[59,944] valu ~100%;[177,1062] load 100%(load 是 binding floor
  1073c,总 load 2146 slots);[0,59] 启动 ramp;[1121,1185] 尾 ramp-down ~60c。
- op 统计(D4_FLOW=0,MOD=4):valu 6855(1142c)、alu 8544(712c)、
  load 2146(1073c)、flow 704(704c)。pool 总 lanes≈63384→1056c。
- knob 扫描:SCALAR_MOD=4 最优(MOD=3→1259,5→1206);D4_FLOW=0 最优。

## C6DEF 设计(c6 递延)— 当前 debug 中

- c6=0xB55A4F09 是奇数。递延时维护补码位置 pbar(pbar'=2*pbar+(sp&1),
  sp=val^c6),表存 node^c6 且叶索引取反(leaf j = stage + (j^(ntab-1))),
  常数吸收进表 → 递延 round 省 S6 的 ^c6。
- 递延 rounds={0,1,2,10,11,12,13}(next d<=3);h=3/h=14 非递延且要做
  pbar→p 转换(vop("^", p_v, p_v, gv[min(d,4)]))。预计省 ~160 vops。
- 相关代码:表构建 ~426-436(rootc6v)、emit_hash(defer) ~464、
  主循环 defer ~531、d==0 rootc6v ~542、p̄→p 转换 ~566。
- **BUG**:round 1 结果错误。待查:表重索引/递延 round 集合/h=3 转换逻辑。

## 后续计划

1. 修 C6DEF bug → 预计 ~1130-1160。
2. 削 load:vaddr 32 个 const load 改 1 const + 31 次 alu 链(vaddr[k]=vaddr[k-1]+c8,
   32 字保留只省 load slot);hash 小常数(2,3,7,15,16,19,basev,negv)由 alu 从 1 派生。
   目标 load<2000 slots。
3. 重扫 SCALAR_MOD(3/4/5/6)和 KA/KB。
4. 攻 tail ramp(~60c 损失)。
5. 收尾:`python3 tests/submission_tests.py` 全量验证;`git diff origin/main tests/` 应为空。
6. 若仍 >1000:诚实报告 floor 分析(pool ~1020c、load ~1040c 硬下限)+ 实际成绩。

## 工具脚本(/tmp)

- `/tmp/sweep*.py`:knob 扫描(pt.CFG.update 后 do_kernel_test)
- `/tmp/profile2.py "{...cfg...}"`:逐窗口引擎利用率
- `/tmp/counts.py`:各引擎 slot 统计
- do_kernel_test 每次 ~1.5s。
- 快速验证:`python3 -c "import random,io,contextlib,perf_takehome as pt; random.seed(0); ..."`(见 bash history)
