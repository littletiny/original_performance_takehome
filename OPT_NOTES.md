# perf_takehome 优化工作笔记(抗 context compact 用)

> 本文档是优化工作的持久化状态记录。每次有实质进展就更新并 `git commit`。
> 目标:`KernelBuilder.build_kernel` < 1000 cycles。测试固定参数
> forest_height=10, n_nodes=2047, batch=256, rounds=16。
> 约束:不改 `tests/`;正确性由 `tests/submission_tests.py` 验证(也打印 CYCLES
> 和各档阈值 1363/1487/1548/1579/1790)。环境用 `python3`。

## 当前状态(2026-09-14)

- **已验证最好成绩:1185 cycles,正确**。配置:SCHED="serial", D4_FLOW=0,
  KA=2, KB=1, SCALAR_MOD=4, C6DEF=False。
- **当前进行中:C6DEF(c6 递延)debug**。round 1 结果错误,尚未修复。
  见下方「C6DEF 设计」。
- 备份:`/tmp/perf_takehome_backup_1964.py`(老的 1964 版,已过时)。

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
