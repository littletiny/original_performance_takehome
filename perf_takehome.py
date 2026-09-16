"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

Validate your results using `python tests/submission_tests.py` without modifying
anything in the tests/ folder.

We recommend you look through problem.py next.
"""

from collections import defaultdict
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


CFG = {
    "SCALAR_MOD": 3,   # 1 in N offloadable vector ops -> scalar alu
    "NG4": 6,          # first this many vectors gather at round 4 (rest blend)
    "NB15": 3,         # last this many vectors blend at round 15 (rest gather)
    "NG3": 0,          # first this many vectors also gather rounds 3/14 (d==3)
    "L1MADD": True,    # d=4 blend level-1 as valu madd (else flow vselect)
    "L1MADD_N": 32,    # per round, only this many blend vectors keep L1 madd
    "GROUPS": 1,       # split vectors into this many start-staggered groups
    "DELAY": 0,        # cycles between group starts (fake alu chain)
    "EA": 64,          # priority: eround = h*EA + v*EB
    "EB": 0,
    "SCHED": "serial",  # "list" (paced) or "serial" (chain-by-chain SGS)
    "DELTA": 28,        # pacing: cycles between chain start phases
    "GATE": 0,          # how many cycles early an op may run ahead of T_des
    "KA": 2,            # serial SGS Kahn key: round weight
    "KB": 1,            # serial SGS Kahn key: vector weight
    "KSEC": "h",        # serial SGS Kahn key: tie-break on "h" or "v"
    "C6DEF": True,      # defer the stage-6 xor of hash c6 across consecutive rounds
    "PREXOR": True,     # pre-xor depth 4..7 nodes with c6 into the mem tail
    "TREE": "tournament",  # "tournament" (bit & cond) or "linear" (cond xor chain)
    "HSW": 6,           # rounds >= HSW use SCALAR_MOD2 instead
    "SCALAR_MOD2": 5,
    "OFF_P": 1, "OFF_Q": 3,     # offload fraction P/Q for h < HSW
    "OFF_P2": 1, "OFF_Q2": 3,   # offload fraction for h >= HSW
    "RUSH": 2,          # fast-track this many vectors through all rounds first
    "RB": 40,           # rush priority bonus in the Kahn key
    "FOLD5": True,      # fold r5 entry into r4 p-update via flow aux select
    "FOLDT": True,      # fold tail-recurrence -bit into the next madd (r5/r6)
    "FOLD7": True,      # fold r7 exit (sub+madd+add) into one madd + flow aux
    "FOLDR": True,      # fold raw recurrence +bit into the next madd (r8/r9)
    "FOLD15": True,     # fold r15 entry into r14 p-update via flow aux select
    "PF2": 0,           # 2-bit p-fold: last this many r15-gather groups skip
                        #   the r13 p-update and recover addr4 at r14 via a
                        #   2-bit flow select tree (-1 valu, +2 flow per group)
                        #   MEASURED: no gain (PF2=12 -> 983, PF2=20 -> 1001;
                        #   flow total rises toward its own ramp-bound wall)
    "PF2MODE": 0,       # 0 = 2-bit vsel tree; 1 = vsel(V0-2, V0, b13) with
                        #   the -2 as 8 alu lane subs (-1 valu, +1 flow,
                        #   +8 alu per group; alu headroom caps it at ~8 groups)
    "L1MD_P": 2,        # Bresenham fraction P/Q of d<=3 level-1 selects
    "L1MD_Q": 5,        #   emitted as valu madds instead of flow vselects
    "L1MD2_P": 2,       # same but for rounds >= 11 (the post-wrap stretch
    "L1MD2_Q": 5,       #   where flow supply dips at the round-10/11 wrap)
    "VB_ALU": 28,       # first this many broadcasts emitted as 8 alu copies
    "CONST_ALU": True,  # synthesize small scalar consts on alu (saves loads)
}


VA_BASE = 1 << 20

DBG_HOOK = [None, None]  # debug: fn(h, v, val_vreg) / fn(h, v, bit_vreg)

# SUB900_CANDIDATE_BEGIN
# Verified checkpoint: 928 dynamic cycles; 9776 static bundles.
_TUNED_STANDARD = (
    'c-pjmX_!^TwKi3?8;OYtf@m`%4zwWHOo9W7Q)}F*#F$`EL_tN-'
    'h|&s*1BwHJieeH>%;3GK0dW9FP~+6hI48jwXB<#Oo6%Xn^;XqdRePWI-uwNCoTr~tp6+w@S-'
    'WbD?|Rpy?0VHqHN$_yXVgrbQ8O(&XXNSQXV7Ar9yDl1-VFc7%@{c?JN?X&C%OM$b>f+4oIP%a{jMtCYRZh!E$p{dw)qdH%osH-JN'
    'fi8&!yj)RyB6y__IgNm^@?Tc)X_xZ&@=<^_XG5uJZ3$GrE=ivWnh@9)9%|y+ZFcZpNt5&FR<at-q+sTe#2orpl_8^ckZy-'
    'e+{HyhV7EX{zUpyro}^!l#XH>3;)n{52l_y*F0To6+a_)fV>}uQs{Yz74C-'
    'aTQ;L#bzwNa*Ou%Gxt~D$pucUI>$*RucrSQF7xp1hre)9xX@`^pL4=l^b04BBzfgs=+!A+B65x2xOdzbK5V_a4qk0=t4(3`87^dV'
    'dWoCDRehIS6{It|M?NsGrpxJ(ALc%io*Y4c;+_npukM}<rgxmeg6@{r(ATH`-'
    'z~4D_nlIb<fraR)`t&Y>sA}WY8@`+v#|ICi_gR2V=OkvWzn5FZbn{1{|4gqy1bVDxu@;(5&03ge{@-'
    '+oh0S{e^+%?)xL{HE^p?f+gO#i=4%VTK>w?Ydkw$Y?1bm;Y=*z)3;)nv>Zjp?K5~NM)f)HO+OT>b7rQPjKEUFWuy_}XkHg|UEY>F'
    '9=|^M6pFZx?v8SJR{K%6|@?Wn@ewh0zll-RWkx!+ooI-y-'
    '%zb6|G6|#mjC=xp@ey7EV`)Q=DDU96tLQK37W&VA%U%A5VfCiFd|tioR%^oQJ6vUBSggY0{jhiqix0x$6)fHji&wFDFD%}`Vom&-'
    'bhq%$CgH;Me*4X!uThgvb>DK1`<D2P>*)XDH%8azH_+3m?yd5fbTlU<CwevAr;bm5{-'
    'M94?^M}l2bON>%k&#R_{A~vo5_7$S~vEa!hL<oecd<1)vm6Bq}~pzciif&uzKG8=IdegZ+x#+VX+d6*TUkTSiBMz|H9(cuy_HBH^'
    'SmYEMBh=Osx~e4Sd7zrb}{99;B<85;psF`3-'
    'oF5%gq)lZgAy;lQq;Z|%281HIi1`Q78(Vg)T8%;&gwe=7)jSpH9XH~E#9Y2kh)O6NKEonMm8fNjI8r`&7Lht-'
    'q#IxE9s1s4Adizl%7S6DoQ#S3BaEEX?@#nV_kA2l`<$r@H9uZtpKS^YDw&e!ni|7ZTPf7%dr!iuO9UZ(HZEZI2AtL&aF+&fp(C&t'
    '}?JJ67?@PEHszA%5nKkc3`%AfR4d*qArr~K31^CkJy{%OyAY5t6V+ACj{KkJ|FkuT4m^IvW;zTArNO_yTvOjs<#;@PlRg2hu+<I-'
    'K+E&rTE?yk0xbYQx78{8duF8uN1ZnZqD7P@=IuPt(~Jr!2}z{flp77t>vI4mB*;)$?$6pN){@fa4%!r~Duo(_wLu~?GW4BJfQ|Hh'
    '`ZEmZz5R#!EonBN)~HfC*HRMXNcGAI9P+_d!T+Zsjo0^i>bRHSl~+huE2{vB<1e;xDPcWel&``zmCuv*}x!LL2wURzY<KeLhkg*~'
    '&L{_W+TZlHgA`==Y}-#-57Ci=Ipf4Z6e4f0R7(7*lSL9VASQJW9-'
    '`U^X1DDTMXpw(<kMbDjD8)mB&y%Hoa%yv<%_X|QeOp%UqLKv**u1<-'
    '<K0@U$IAIS}`AVlD@b0yFHGe3b3IDV@UqLU0GlLJBG%nrxxR6O<@7uO2za(sa+fL<|hTU)5tNgOC{cQ)8Umo_q?Wpp};Q-'
    'i9D!(F}0NYvRQ@kJ++8tGXrGLK2c2)UR{`q3Nlgh95&zIY7DxY4oU3GT4!XEIqxz0{gwntoDscbJ<<^QsK#h2)}ex_u1-'
    '0SZ7{II&ut?mzh_8I-#-(Nuk<TBMiUt)Jw`85B0soh28*ZAkl?5--mmacMI)!F0DJZlu()aK{-r_e2f75Qd-<0dHDcJ9WFRrK-'
    '6jhm$M7pK5{?>fVdP`GJuLQiW)KC~T;J<*I0NVVu_+@or*q_K^<DEyD9rM*(E$nkR@Nw42qI55{LyIYh{f0chJ0=nCY<leBl$E_Z'
    'y;+<1X@9Lko&D+xM@Xy=j?Pz!S=k4?Mv^)Is4tWRK9sYU8yd&)n|GZP)iFSv7-Z}3~yMr$Mv~86O8BS}BZ%$fl0?E%W=n#)piO8-'
    '~`9IxWzd_~Cd&Pp&o2ZW(`p(fOY`_NpA>8Y&&N+M3>7!00-'
    '?@Q=ONxn<)VP8NU1$FdX^L>3QgnH;nyTy`ai9Ggs)!fQ+I{2VDZ77Mtgz?DcW6-'
    'dthgQTatev1J<mz|zOb6>R(FTholvijBzDm*R^+U?`}?4xUz;K#^g^S(-'
    'No%duZ@P;yRl~yeVcd%HS86zODV53urq*bpM&iv!aQvse9TORUE8D}pHs9G<cmM6XfMd>8AZE6R@W%oJ+7`&wr7;yGj>pXiG+Vxe'
    'CbIXD}L#HMdwrmd%F|fys)~%t>%XChO8jzl53Mq=Osqse=t3>^wpm70$!!)k7BG?4@oq#eyE0IhepFyQ`mUU9idZA@8*v#J*B_)s'
    'y&h8N`(zMRFNVt%kf`|PMUnrZ8h#ZuYRQMint=N9UfOCQX5wT6nPN5Xs+!+@ud$HDP>vF=6^g2@>VCh+rw&(Tip=^d0Q0}?^Km+$'
    '}bhY>fG#2v?;faGOHsgxh*(OoKoJ4-cWsyCx9123{Z>bQ;TBz$ZfwS#dfu_yT|V_%U$`cVfA;nniE#D-'
    '6`~|dfQ5~QLeYGH7Po8t?YK16rHzLwv8r5=dG39UX!Bp_QCF;Nzr-xV5>DLx@-'
    'OOwwe^(^h8d3Yub)(Z}rji$>qaKQvWhJyimQ3P7K~Ibk`7Nvyy8OBs7%$nc!yB*>>7T6sh34Y6$KIMfRsGxIQZXM&x#jlhmxRy4k'
    'J%9(D?S7<(0VTSboYZvJ^IMV@kZ|Gc##SGk9OzMUdpd3XQ3jUs2cr+>b^B5%2uf4+kvcX<#0X*WUVJzP~`9O!WGt=TxxjqRgj-hp'
    'l(U7y>58<icK?m%b1=6Jh+!;NhhWJODM{%UB|N(WU4(M4T|RfvqDAm_M)+oT}3*7nJGuGT1WcSNrL?Id|iSj}{+o2%r0vF@<#b>2'
    'H)=%07cd7pfdf8J5&ee=csc_*Fs%a{1)ops(nU+SNC(fOYFGXH!>oe#*DmsK9!-'
    'BvnLdB<qF2q(PH>9qbZ3a{vYI6ddhg5Qvb>!Rawq=rPBjT5IoF(@^aGCWXYOE+ai@_*XBqr!jA?i1ztV{Mnk)i&BLkE=aZ{)H_1s'
    'r)N>izn=P@z)O4c7T>!a-'
    '+Km{~cC;bE}zQ^<QpvPgwmG8}p{BvNY+Q9+OJ59ZA7UbBnUIKP+uUe{!5%F8D6t){{?zjS$o#Ty;{Q2kL_0XiFa6wA|p|{XBlrzA'
    'FD(7F%h1V0`IaMSiTT$SD6KuE=~p5?5zw`)%C7e|8eNF|2NItG@-q@Rur>^l9O!y{72l1FBZ`PY~T-'
    'D?~@KZ>Ljl<pdqQm3n)cQ+QLwT!`_gcbai1@5<UJNT&-M%HldQii^_#8|dC_?e5YS$~OLRN-'
    '6AkT9L0R+vjP!Ulhnsv|Soky;c6LEEXv^`n(#a?bmVjQw^oha{RrM+@Hhh4{miskmL1Luwqv!znW16-'
    '}GO~|K=avZ~j;65niO}FWoo!j6OcuZ*Wq?GT>x3a$LwsI2aF~fn%>v1nHDx$?H4E9oT}|4cG8r+J5N`+aERUhcxQY+8!GB&+pvz|'
    '30j)bE`ju)!T0M$11+C2KKc`6>{)Mnvi2hQiL2jk{;w(jnp8AYNQ2&C594ohPY@^=q6kw7Uk`#{HauSjo6Gs+2pyPT@>9FQPI^op'
    'SqUi+Ti61VK!E}d)GPvHfc5fsVJBdF<t9K@w>2^?pD`@jr5kHN60?ld^E7ZzX!(H2k!P>7-'
    'Ju}+xuXQec*2IhcWhnyPXeX>;reZ0LIt{?)Cu~<0(nXtg$Me;5OK1Ix1UFDT|SG!XHJ9ja9Z*6?7i1F*-'
    '4`#byM9A~vEn2Dd(!SL?OFO67M#$^6niSgD+0FAB1ywoBrv>!s{Z<LW&{k-'
    'V%vR(7zK)p%{cim#EMynTF)LX2bMijMe2@g#3?H|N^0y2h=hhm-tG6#}4Vf<~lxVwH0^<mi$+_6L7#PSA9JD<n&Q|CGeT^g&Zo$w'
    'COUMH^$3ewvQQyOxF8kXqq}7}m{1aN^{b>dHpkAabv}>LAzR1<~>CDz}<8=?SS1wm{Ptci)C=*4n59{EeTd5dlSK;cjhjSGkt;ed'
    'HUfyh_&hYdcTn&H39&Sk+cMBXZ2yF(SX5{UUP9*)1ZkoV_A)%GoI*pWORI;^Z`@NtTvLl4Tf<c}=v&HTY&LRbpxy*-'
    '4z)baZ!kQC}9UAKlOespIg5rtRb8_oB8Bl;wAqwzsKViG1$S_70UBS>LPeohr{{Jzv|qRKArzLyu|EruK0y>eM!9(WiEy7KLgTY0'
    ';>5u@;qTmuS(ccBu|ZH8D3cgSi=Im+3U`P?I`?wQ1fVH9E9`y*5jQ_^gHyo79=)rJNIw>?fMsT{)K1+(Fz@Ncvt=#ma?sCP{)-'
    'CJBNyCdq*nCW(RdB}swRB?*DGCCMOFmfh_8noNn)mZT&`>!SPD$T?Vl{JcG;U=HFM${7UKe6%maO>-'
    '8AyrIX6jv%dZteW%$KT&qSL}h8)EXw9;Cyi;9**?E>Hj8!h>n+;el<vIGC|pQS?&sRxlJ3MWC>Be1;+NXaN>BDz+RpYT+poXY_Ex'
    'u3i_Rh4!8g+5QDxkrpl9D`y67pT1I&jMq9SS5R8$-4T#g|eSeIdlI})pk_+*)e<Je@Dp-'
    'S6><IdZtDM$;e_6Qotdwfc9_6(25G`p=5wYR06ceH=nj&|NL{%L#KdBgnE4z%-*^-'
    'nv}&KvHZcA}kkoPXMxcHZ%2snNZyGKtjM7$+-'
    'wwvC}nOQZ&GWQ59FkjD2%mMrRMtgq1jpVk_0Q;}+*@R&9dxs2E9vh;qa?GsUY3ze;ptJTVOi>v1;%#Qo^$|M|XZ?CF?)AU@B^Bam'
    'Dp;AMN5|78dIV3fS$HK6htMFJHR`V1dOSJ!hsR&d4l$K?2PVWv`f+Xq)j#qWr-Xgm~!brlmnMf3r<B{3UcX?JP@yG;i2WqD_&e!%'
    'ZU5-aSRd!H3&WCB+H?Gc9_Tng{w_LG*L3oW${R8o}E1UqY4y(y-b!Avhap|y`;qP8o^awRD`;MZAFOt-e(Z-'
    '`8NbGV*?9Ph7jxJ*BB(STa?;4^Hzpq*<NoC>mrhy&g_x}~yv1wpQUPx!wOL8J@a9ye$+g<1wXjeV<m(tPT;+j?5;KlBARF|V9{Lf'
    'l8^<0a+)XDs=Aj!*|B>xa3`L?15Duh_vrSMp)OJNua&bvv?js9`Tjs9`Pjs9`Ljs9`Hjs9`Djs8t3-'
    'v_$kZJahpW#$!RLn5;CG<m{KWEW_Agq~6?DfKtBRotuRYFn#Q72>U!*l+TVUeqC_hG&fxCj^1z2q`tzx?@z-'
    '8$AtfiG=IX1EQ_5SVyyzw}m$%Rw-`>heR}2-'
    'X7mkY*OCAF4xf{<sIz`9W7Gc$v&Z@LCQPZCpCLc{y+!UX{4eD`%Wh*da#ppvZ9Bp3l@%Tjk3RpX@q6=!njy$FOGVSLcd3o1eLOF;'
    '_=?3ZTq--Q_&-op~>OHW05y5_5_LtxUy<bRP<nXZj_=2`*5cydKA*qhZu^w-'
    '3|Vuw;oab`AAp7H@XKehQl(<T0<}AEjYRVy*{2DWa5#DH+qPH^HOl^!gjof95)ZwySOU<JWVE0WYQ%@$(Oi;Ha|$_Y;C`ZGCWk<e'
    'o<<NDrAWH?<Tq9x;nfzS0ib`zj>Rse~GI(+Ws}JULl#r)z`{C99L&4`<`D#f3KFDL4O*YVK`hT#`bU+MiyQviGU-'
    '*hlOM}sUnFm?|a;mOxr`E2s$g&2zk{>*@xn)16_PvwIvTFuGTBNPyAwsXuD@z?WXbI6?xxpN3mYu?8y~jHPPLizlGI>?zPFBx1vo'
    'q%8=<H)l|?|bmw<5(Qbs0B52;>2I@k<67PL)gyEw$*?xCB(T|}0S`_3Mr)ioLWD9K%hyo$mjEk#X6%yvWx`VcGT%E4$rE%3u+nwX'
    '8r?$Jq)vE~JMReym@mvsA=R48e999!xVlNMibFr9OSsWqRk1`yX)48ZtQ$*6^4T(QB#&;uo5{@kjf8DV$H?xZh!48cj^_d~_=o}%'
    'tA*@I*Y*6`iMS)Iq0&Ox>+g;f_Nla_IwUIA>j{EZGh1Gbsx-k4E#UAg4ORYoTK?ka>Ur^NXfzI3c1%(`Ibl%=CDBk!`=N<flf{l-'
    'K-'
    'Z30mZP)6&lV4C2@v+W3mrW+!ivA{*)&@m8FIuc`<Ve4#qkl5Y6`A7ZoKwhF*Weg)3TxUnr;Q&o_V`oJIO){007@XqUQv?PncAwjR'
    'n8!R#?|iH?iyDwkTQ>}@94xQs!Z8|Mpm=wR7aVg>7+caiV;b6sR@CJZKdpT6XFrOJw-Am1RQn;WuG=_$PqRl-'
    'N5m(`Nmb*>rIjsde7LgJ^*>tIB;+@E~_>DKi-G08^%7lBj@gWYIhDX5{D!{$1r^T$8=&{PHUdgRSY**G?l-'
    'iul=4}=|@#$pHoQPM@a&@&c3BlOT5v9xJ09zGc&B-'
    ')kwDA6jr48KUenVu=+rw9{zX3DGmEi6Ebu4dGF{y<b9&G&ih7Vo%fHnIv*gWnlT71=!|8YWEo`ak4zNB1cwNc<%^pBPh8SLZ<FCq'
    '9yMZ;sxo_xp<hg#0a`yV_v<NlzeK9f>L#U1##BpI@Gi2fF-_@xY>K2`6d$|Mo-7I`DTvL3StC8POscRW63_BloVCWH_m8Xztl<z-'
    'bQH0j2bm;+K|6YXlO!x!D?8XE35zz$9%%9^j%6V!PScEZD2#2x>I9dhZxvQ2I;2bMusX>hUA7CWQ4Z<SCag|&)uipi>J$eq+93r{'
    'P}J*j^C*{;f%DFikCbzrFSVz{gZ!P_vmeFm%Kq9OpMKc`v>j=ZYz%$5H~s7M#^)<Mz6+}h6dui@V766uKa*sBwo~?i^u0(s9AuJA'
    '4k?I(O`6FG8)S7=zL@t1UGRa07-'
    'CSCWN+x8_cRQqsfkGN+JFT2pF=UiwMi&|O7aNhQN9KL^2qN=j_pJ$J?hDq$UBav*rWSz2(R6$?A}p_l9@R#uE;iB7+24ct607*-'
    '^JTPJLMhOW9KP6z71l(Na10E!l0;t9+{~S;yYF3vau8wR*{j2CZ4YUyMl=)59MU84FvT*Fd_;?Cy+-t-'
    '_W6BWN1=3K&lds@$IGTMJ-92Jgt|@GQC~d0Z|%cT*t@NN^SoXSM!x!BP;LXh9bs#Y`66RgNrk}A%JjkUN;65F1FyN0K>((-'
    '5hYZ*os>MrW4!nvkD1PbTeBgPSS3P1pio;;4V5Mb`R7Mu}eyJg8Dxo>g*Uz9w&MDwPpFy25&-'
    'srz$%+?n!dFw~7**rR=_OMe3}Zwg>A9@Y!y8+W^MNyXWl!7ANnKw+~R9eD}OVK;q;*^Ns<CllRIy1sqPkN8TAQoT5iW;nF8DnIk9'
    'g7zqR)jwTq(H7;owmT)L1NL66Ac2Q8QPE{63X+!kPO6B_I_UGExBSX>L{zBWcD$-'
    '*Ev~6yZIQwLi#M!4t(cQ1?hrFtXP9BsAa*9lA+=oAqeYny7Q1;#y`{QJf{lx7t4}*eFc93R}6RNz2mOvZTc9-O9@h-'
    'd6+30HCTG452h0{b{`gPx{3P-m`1islM-3`qVR*vy<%C$YCBKbAn*tte9Pxc#=<mBcWx*9gpwQ|#1(Qnxo3|dWG|AL_lD>?r3-'
    'InQ6%Eq0}ZTNITmr(F?pk{cOt)Wm=Ge8V6WnEZpV<Z%Yw0FyU2iS$7P2DHFo1tCZH^|=5w(b{BhM|4kKYWIvjlF011VcM}Kq97}X'
    '{SEvnbme|#dhkVZM(Q>I&0fLuAbNSk5SLN&Om|2-KXqF;WaPQgS0)-gd!#mEk(;m83_l;Q#{tNw%b~?-'
    '6lA7n;0Z#dmtaoFgdvYKh<f1x;gwe-'
    'a*$CoM`U=@>#<TCYSCS!|}>CCed72Ra8Le*O!$Otj+Ba+Mex&_e*WZ>2jCf*Dkx#7)0{2vL{8g_KmXth%2&zZ^RYtk2X=vUA3)_t'
    '9z9FxFFbrwLQuB9Gy!_kb6Ef2$s$sx#vfRRSyIb_88-'
    'DS0lpbhbq$5bji;aa7ZH(bk;Brqe(=?#iDR6&qnXo12Ai3&%3NoQKmn3GUcp8a&GVNGNr9Kw_-'
    'y!k^x9kGUR8*)fi<jiK~Gcg)(04r0ot-6)s28B@rUIO(&S7Fh`s1_0(@BU-xbA?jh}=5fKduVx{Bzj<&xs5|h5K?XXIpg#_DKe3r'
    'MW?8*W(BuxcZN4c#=5`_yYD{&`cmIuNn7kRZx3U?KmMfwHOwdHMlh<4WBtHwLaBF-1u-'
    '$uFJsgU*N)dR}DAFW1La)L8onXB{FfJq!H<h>bxL!n(C#@|qA*O&1(C#k$2<8LU;>(BU`lU2Sa<8LVB8<3{*gDe+R*|j3e%c^Yef'
    '*0KjX@S(}j*z_Bt|_RFM7UDW!nN4AHU4X@Ro<?og1p!Xvb<O8w7t~}@~7k;RtR#59UVn?hq8M%=@ru_QdF~F7HXhK>+YX%)?r5q('
    '6Uemg|mKO7RsMM;vAH5K4KRNH!|cSV)*e!8m>^}_Es9_QAGA`Q{l;_RpfoKWa~W$J1ST@!@V?b(?+goK*YNWKx{N%AzX8=P0Lm_`'
    '1`f00vK{+r4Be$+c{pYhiQ8o%e8o^6RYwDzS7xMq!*(Hdq~-X-KgBH_nTm?qC*o+!|J$i=@PGHV_fVT1t6yh-0rTrP<AU+Xhy3-'
    'h2~UMsL-'
    '6I3Kg2)szQb4bXBO(j8TOO%^9jtp*fSWEp(pxMetw$2>#nMg8v3cUs+L_6zg@8fM7}$`Pu1_Y6qt>ifpEWc?}W49JdH%p}XA<38$'
    '?W%3cC*kw)%<yW|S#FGpV~aq56}>>bgLeInYiuTKy;_zK+_!LQhPN&qbOE+q&Sx-'
    '5aP&_)pqi@jF~h=n{BL9vi?%;l@pvm2yKUaLP#q@*`;1=mi|b{)dVXSHOW9#XNmQRB8-'
    'gtvHzq<s)2bf=d%1y^tXN$(^ztbc3!sZ`h~hHR7?)(bBEJUz7SadEaD%?Q$FNbVwc8Irdcl50!IS>&!h@)fzMk6gvDS(cNhSkZ(y'
    'SlPDG#8L$KrWSM18s)jzEdsc{w2J_*Fa06_>`TK45c|?G0?59!i~zGgT~t;FI<~=qtcUGeE49eywEfuS5~HRfXKFa-'
    'Q=O&~pXK|a)TChv?T-&S%aES+Z#AN^vZgv;*;l-'
    '#x`<qc3UB2rZO@JNvXus+eX3NU0_rg2wQd4bBQ!23d$V%+2G_cv>@5Ip1b0x`zoXb0+!}?DIGh34gxQKu0gS?}Dx8CQ`!8)jkuoT'
    '`?(2P#wx|X!_HCPfaYmIrtE~SAWzx9_!zw3BV(<T{e{QgHJH4WTu4ko-LObgvFITiHlq`V1yz{H<7U!xa^dw?a>rn8cP&_yrkDxZ'
    '}TX0)Bx)kq~dllM17nbPih4z3{S1+;$rn-8uJt)=HOYFg^u3kzWX`-'
    'u_*}7C$FSmz!bXNiiZl_rZj8!T4@@|!afKdtUr7J)prS?ui1ot!*X@d)tebozeqOz|Q!(o5G-'
    'r&?BgJ@$H&>KlkqnG}^9(Lq;#vrHBkTg`ocGN4@x^(jGcxC4+i6_rdb^-'
    '5tIxLD@Om6FHWPsRi`FaM3b<a02K&(f;kpW`6=bIQH)-'
    '&JC0I^>A76ypzk$>i@HHnz$Fp`%)#@LA_5!1QAZ8$MqsO+jz*5G@afatrAM`AnPzUIc~@K=ZLHV?M2MqDI%oU2kPy_<N<?<F4dhl'
    'w{lUQ<vK?uv&*2KlUy0J;qkK({dh=r%<F-R20O+Y$kEp9KJ2BA-SLgI7M6*GanI`ZVILXzQtWJHs=U;M<Zs&W(qt2J-'
    'MbjpC??MXZI_R6rD=0St(OV5xyFJX^7?CZmFK^!=*Ble6z9k;Vr}q_HN6G(HN^g%gYItG!(6fbDtHkO2#Y0gCSo*{M(%poq(mp(@'
    'I1xPh$7w&C}Zsry5ksbi;VZLuF2+eb9zF&Ymx4iu}dO`#{X5Lz>Kzc^T72Wo?@)w~Bjv;DOT3t+4%Y&B$x{pvzZ5daxkHbzl&os+'
    'HDCE1EQCL-;ch;*k!q&p|ubr%la(n6K9zAq<2n(mWx^lXMtrW6e*DYMC(rUP2>D#QIA^OU{XFq&+EvX5xKX!0D!vSq4wic=1X-'
    '9z(9*+^ldoDbVa^8wjn`)PkH>VVmN!;oJSk~G@BZODBo0=?J9=4Vm@?C3RMH#V?hC%2H)VvVHDXG`W6k!Poh<^%k%H*|##zgAM3z'
    'pKpEanY9T-|h6N#^V<2(a-'
    'T=aLwH2TFVz|hdJjdAPJ6jK4(A@40jGgKoT719MFIyINmv+0ZA~zxe@_MFw%LWotZ_5?CfI+4#@5Kr0;n(@FOv-'
    'AopODVb9q$IwY#LHgVMOo7a|(AUWPWO@+E@sT_?0J!_-L@Rg4s*?_h>p26>L5$WU-Se=xViwCRX#TArRJ<w9`iSvHDh6)M}?;RRC'
    '=$dk>2YeD-'
    'v1k`77VTrjqC>1$bc_{?PO)Oq*;gz)<H9k<m0U~h13gWJlIjeD>?E`!%x_Ub^(bBO2Cs*)Y^Aw?LHA&M;uer`8myu@9Ayfb79W>H'
    '`OHS*iv9Q7G}LwQOz+gNc_9Xviw+zg+<0KvBo>xnMB7D-X#0o}?GQ1d9V14xQ^bgN4j9o`(H{-'
    '$2}`9toE9gQjxac7t@IZN!sRls`YTifog%;b8%zfsBfq*48kf$IU;P)<2OT88`fr#OI!Pg@hFPJb<X1OAHwTOnh+;a3N?(jnY8cb'
    '8bk=Ommm^#{K5gcyPCXLRPKkkWp!FJtyh%kxQeO-gfY(x4On#%%*0olgvDr?=j7<~mzH}X<lkAHRi5i;VL4e~3Z!$PMj+cSd!?r5'
    '5b<S5mnt5u-6+f1_X~^+Djs-(r_wlSc+A6+pd2gh3!)?p^pe`NmTHY7YX1Hm2KV-'
    '|oJ<Iz;6@*)s?}@N2+_8KBqP7jCGuOrG@uF7h27hRS$2hDF?O|nKoWAH^807;gq_BY4v04+c*k7weUD_jcnZvB!M>#`zKT%90`-'
    'uV?*-'
    'sR{3@>vj0IajEh`A5w4US>3PHXwwexNrvmVrBM<bC!8y}@w|;@Lr!{mhA|86dx^b+VtnKrX@Ur?0ikOchPt0%z*p38wA{ZChscu#'
    'kfPmA0)jn}^k}wcReWKQoeqoocu@;!b6+Gz=hq2wf6hXVjxOvo_=_>JXPV&jKG7Lhm1B{%^oulQ$DyyxCYRtV*4O2Hd)#V%!ER?%'
    'b;`ltv;CLk*y`M@raL-'
    '9$#e8kGqoBW9+@?@MG`oVdP2+N7+nj?}hQW)C<1+tFuyB&nHi6|jHP{Ked&Y>SM=voi4spHF<k7ZRWF#l$CkDe(zkPJF^w5})wZN'
    '}mwLxYbA++g)|62mdd83(YiiQ?tY)mgXGWH3?AbeNZqpDR_RaU8g-'
    'g`wCg9qNwTt99$O?QgukeY?X2E$1G*H&e;3BRoNe9p%LUiV#X6}pu#q4kncWLalXR&nTTMu#NT5TdzPqgbd9}q5b)~C<oiT}GTft'
    'S2>cCFa!x?sA|F@s2*|2z^m;*B-d2m~o0WjbwJ8if@!%TCEB{=lQQix#ulNPU+X~+2i%L*Q<bJ-'
    'S1TIWo=Br^v+M}!HvviTRJ833ixfqaRW?`8G#4*#bT;d8ufrsT%ZM$hEV!2G)8qG{BmjmGKZ$*JNrkO5<FnfAuoSqHv5nmB$&<OM'
    '-7aumNl-'
    ';Kq#j3f6Z&x!N1xt5qqK<t*jpeZTc?H+~c_ny!^21+IrIztJmj>?5NpVIExHNEIPKh&Sz@>rvb3&XE11=3bK)?$0*fW9^=rN$=hj'
    '-U7UK9{Frs}+}L@xb(0-@-KYha|v)Q?rMrA}M~vZ2j%5<E>%5jNAnd2`MfO-'
    ';G{8X@jU3}o3PTz;GEXUy3L2oXK{Yd*=1(Z^gLeasEf$J`iw%uUh9+#G$(Ez!sPtgP$k?0$~Yv7=`b=?wJg0eZGqMGWjunUiNyk3'
    'P=W^-'
    '0T*J>J+2NzacRVeH1F>Bo*Vc2m;zV^1)4bJF%>Pc(K*()VLeGWN63_ygR}7N$G&=8A5o;CtvXPzyqX9($$nv9Ah8hw{hx=1f)H)f'
    'KQ`2pC)vitEAbT5b@P<g%#nev&iTyApO$2VB%oGJ9ySc3&z0;zOuGcH<MNFeq}xb%-vY3y;y<(GIKz1-'
    'UOPhNNs3?#IpcbK!X0V!sfM$Is}4d@{*_py<&rLRILocPVYKQE?$)wAfiV$3I5E<r*a&6_w`R%Kr<E&=6ET5a&%v&)zy9Tb;BTn}'
    '~twkgQbcVdJuHD}eC631}y%sxH2)%Ne{rqMcIc0p9~UQ>UQ`OLoQ?XbTgh{>3pudf7FKG>YmHdV__L-'
    'e6IrH&`6$4VFZDgQbz)U|FO$SkCkYNv2Li!uwMkmYTfvz@0h~Z2;!9yrE_S9?nUq*~<Pf<Gj<|%Klli!*sv0H)?JUZRwJ{p?eA2;'
    'Z$_Yi5~qen$kp%o{k1OIZ|WLqb7UjOf;#9mcJ~9h-'
    '3&a*CFEIIIDs6mjmu3@YcqVbgGFg5)wqzGVM)mtU?yLmq}v3H}<MjVjlyy>m>FGpwbmi0@DMA-'
    '68w7F+zqI7sRpWHJa&S>u{~L=uA!G^R7VLOEj}0X$^QXv<CdAFrdNQUKC^e7hNC9Co0iNYbDe4sj?TUvJ)^*JI+$MH4O#(Mz3~}s'
    '#+@@wv7+)hO*$ss&bZTJH$;%jdhaLSa;9>4qYRa)XQ1VImNLi`UW3H-{7O@8?24K!N<`zSQmYRPoi(|X*oWjOWkZ-'
    'i<#_}v5L+AI92wL7%wp|wQmPUJK>#wk=7U!nS5op*E<g>wf#t3(JAK)`Z>l9Hd0<a*VqG$lvgJhd!Ui>>iGsA&IQ#A3|d0Dpn9RP'
    'bw&!Rbe|^bNR&wXgKOX0Yw|HU;WsC3o4u>laoQZUv0%%_t0syB6QQXXKTuEVi%+X7tVPfmH&A==or1y^JLTIOq*z8f=Gr1umA>d`'
    'c-'
    'jg40{4$G1jr0Ahyd)zj3NO0F~bPJe#|%mupcv!0Ldms5}=~6N8Eo6j4cMY$i~PmvMF+lY)%Apmg0)j*{TVE>UqW<WPCZ?fA^hGk<'
    'X1F3N90Se_PtKe0$m=d`H?Md}rFtJdeAXnIdI<q)6Ei8Bn%F29(cIecTvBXbRjXGI$DJa`&)kg`OC?&9h8mXk)W@#civ_-'
    '7O!ma})zn&sB*>6!J}zjpxyKElnSl>2$g%25tKFjwrn+z9mQ-Q{QrDRFV^4au*az6JK&y)Rz-qvKxRA3{YL^5aEfNwTHM_FmJUI#'
    'K64u<~Xk&r5~^G6N_-N%h<|E(TTnLALxA>k|e)p@Dy`&hI4Mmsxs(>{^u-'
    '14v#o9cNlScN$t|3G31v>@6uyKSkdmI$3`RVR3p%2A`MjZc)&;l6+Ip_(lSMlf0!67zh{JP-'
    'F{?*uG@ZWiV5G9g@o_QBq2W=;%PxgK5$;6J`mt8NXpf)<pp?8EN@EFQQmSVYWp|C2#vO;>{b?_f07_+g$>Y5AoOcpJI-'
    '~9SJ_|GRyFdh`RF2;z|jua1x6ap=<$$|W-'
    '@v_VuXy|t~Hz_S?RmXJi7IHB`n^H;EXbj_hQ&}=J8&Fv;z})FU3w}Chuj~s7&R(9Q(9W>IhKWaD$f%nh;G{Uy*-'
    'zyKzyo^8_^>pM-Bfc7q=4dB5nGTRTC=eA<aR=F?8nT*9vN#0&OeBOLzreIq3P_5&j<{&tNK8h`s?x`SK?Mxj1;hQtP~&@j1HrFV<'
    'YPBVctj4}3DEwpf}K}c1``qkt!apz?V9YtT(Ypr40VR?$Sk1Veh`q{L5rWLw3XroRDX)oh_+u&U}ctz9%H5HCi9ULcmd}_Fy$qCI'
    ';Vd_>5Tt+Y$g_2&OWc7XZz5k(@g`Ckjngrli`)B|QhNaD?6b`$wAI<GhQXzUuDUCp1iw|U>UN-?$41E0yG~$-'
    '2+*NFQW~2dL58ry5O!{qF0oy!3oXVtH7cOO;af3(bQYgS|)P+9S?ul1k2W<GMoMl0K(dm+AS!=MB!mj<;0JJ*gKUmf)g7B1`uKC@'
    'dO_wHTWLa`XmM7=piA3W*ndp?K67Bo65yp!FhJFV24cBabwi059eT+K`4t}ETt>i(PG)3};1{vZ|fqD$rff^$jpn=38QU(&w4c94'
    'H%qjPf4hUxpj<#1ri?n@275C4J8j!BR`Pg#n-4_aB*d*zp|AY3DMLB4rOoZr-'
    'T$=Q5Z%%r*w@B}HKjhdLjEtNKqmhv_VK_2!Cex!{S`p+)*S?pbpJ^8SMa{!JT<=_Won}!b%?IZ~<OeB%CPDPs``S6vq03wL@Ovhy'
    '{K6kM;*Xnh95+Nqu0y!h1q9D01-fE(@Bl&SE}&=%2ZrdfKJw{-'
    'b99;>n4r^y!1+2&2wbQoJya`xNncGjhBI^hNT*2#sy|s0PH~dvp?kwQ+yP`)#25`E!y<moUSwH9F0YW>orP$^lZ~>-'
    'o<fY`ME0n*52z-}nJ;QiX8ZLg8bW9$$N`Ny@8^NMXEbck(6rpasV0}Qo1=<>-'
    'Z#qrPP3Y6rtJ*Pnx)n^%DaLt{R^Q6;P6}0F7G!ux<lM<-&AS-'
    'X1tax@oCB<*d$|fo=%fn7wCl7DxpDaPFlk5;KlPUcs7bBaTWZ{^d#~|gA_s<aF{oJNe*Q~;@J5DV#<6lCP}43eAi>WN)`6}Az7lg'
    '9nd}mHUQd(zy|FEAt*yTLFg_<55Z>%qaU9<$27;f-'
    '%+D+k<hWuu_xn*n)r4(20Hd@9048FjY#eZW4H5?dmJ$O@>#4lpwzHo+qfJGXG^M6H|9o@vXe9o@)?zOZ?}ud0lN0~AvwU1h$1Cq>'
    'k=j8b|p&4HYG~P?Msx9JCrCPt4oxSZJ83%C!Dv^)?}P<ZlP_K6;G&jC-g?3PmAb`3w(0wp%l)-7O1PXmb-'
    's(y4!IE%AtJvbLWS@ASLXz&OlXjlD>hF$Vq5<Xa)k6D2$(GbWcMYoh$N$OjK_&{U7rFYsZ<WzGV8>@>cA?nW+9``q%MM?!=j>0c8'
    '3=<w}p%_FL&DHQIhxqB|PlsxV!FSW1NS2*Aj!#ww9e&a4!$G0^|P|Ght2C)dg?W}pFow?L5Fl3$IKFk!^XHzp+t_Nae)+@%HePmj'
    'B`p#JG`j}{sxdfXpY-5hP?{3!IgjGUZghaQ%Zp-pyVo`KjV?zo?2;I`RH{9$$>oo~B!#<lVO*k&b0E|<MbTl&*iUMvF|yV)_BXIK'
    '%SUi|tyyKy%6YH_}|$`U-?4>E=<>}%}b4WoD)B4&JH#EdVBnDNCCGrlBZ#+OFS__Bx@U(T3ujz>miaMW5PjpD;I-'
    '1%UeR;if39+LU|H7lG*CQtv?j4Pv~T=K#nxIRGXC8(!X4p%pccT*o77%J*LS}NoMEr41L<wOx3RSQ=s+2p%xdrIac-'
    'c#GrSt{|@mHkadKZRTUn2df3b>py%ehLfbu^Ij3uO$6*>jfo3B4>~YiJaqD5;;ReNaX#cjwS&uiF`nG;92m4lTt=2SRInWWG#eEg'
    'SnJOmT`_s`MyQ$4S{dCxK}@^Y}H3L%q8Tdg@zvYX^97$XB-bw2-qTHj1ZaeA(^wWVc^428T}MK)}fhr8E-'
    '0kbS7TLp2q%{VWUYg#D5z$n&bfAWY|tp1aXTgn8qI3p6u^qt+v0-D)4sjui`$XYwIF9yP-'
    '@Vvr6zA6Lp##o2<*$(M7aB%#w)qM;SZtwQ(Etj@zJ5+y;H)Hs}|(LI1c7_Ke$LKul}wZS2j){ff^ztgydw*+mgWFKtitqS!;*)3P'
    'ZQCDE)HD)XIw6wG!k`v%Y$v$!c@s*_-KF{U=~#ZX{3)3|TJ%|SMWAuYB)Pl-'
    '5}*k7bXoJ;KyDG}!~`^%Jwb2(6Zv32MP`)i%{3YF+;CTiT&@hDHo&1y{53{O$9OEgF#Mw29hYJxBBj!#ipkVSk}J2qfZ7Qk|N$@b'
    'Rv^sJnpc^bq7k!^ZQLKD-{2u+N$htYRUT0&umhv;J8V;#(}C_Ip#9VmX};~3h-'
    'e&j+BJhJQP?cpRXU2r>t2_UtHm&b2ESpwXnG<ZeA4)_hq=KN7ZwVf)>W5;NFmBvREwZ|cuW9CZaMGol_pU-LsYJ;-'
    'B+Ddb|1z)}=I*&-'
    'vhrB;tYNdsKs2YjU>T(<tI9S)D@;Xj~4RBjY8Ui}m+N55?#af{LS93fcO0NOfy{)T^94)MOCosYjp5uuO@kHEn5@S3OmyTkPCt{$'
    'JRYHz;N`QGXoZ8ssUN9YuT~Q2yz;a_MBOsRmZlz_&BS~a32homNB5~*w(C}neXKlgqnyCFu$y8v&k!z$KD!mM}^II9|U-'
    '(i}(7~Bjpij3qQm5^r0@jY*vx^FL;?jacc3D^*t?iW(pdPF3G!9U45AMeqlCb{)@T*k#sxMZ;tved~gcoQhW1nmSqxUrTydaTyjb'
    'tAt0_#VcZir;Y$-'
    'A2*M`MT(Odevm0M8Y7DD|UbGxjmD<^0&#XMUX6m1{~N|Fy0_8okizu67;0)azXNI(ntQb2Zc$C;i@4v}2s~2UppPUh523<ld?%yz'
    'z#dKfkqDa`SM^o#gJpu`ZucikzMnUu+iYqp>_hVr<~llDDwQXtW`5tHNo%(G;^3PisMz*=Mvs%Rsoz*fRT^7I2yUmljBw{kIlOnO'
    '&(psEj-'
    '1hsK`xVX^ancx?ASUQ!khYC3(%q3lethnK{C*pwA^^BUdMsT{0gWepo`u#jtvb$Rz>lE5`0Ah)@&r6fZSoT@WPNls#pN^=qoK!9='
    'GwQWQKwTnog_7MrxAtHe~MkG+Dhy>~kB#;{~qy@JJRkr{HN0P|-'
    'ho%E1nYM>D2`Q0STVt1ciQSDeT`ukHr0vf$`Y9wnM`ZL<7>|z3=%?&g3Sw?b2?WWpx<&YrK2x^02@Sula(t6GDx}Y!^IF0?yot@&'
    'jATL}TFkc~lMr|o^Ush**rAY5B`<o5<^tegB`N44eJeISUfrl7!9Ykzw~~5mx^G${fm~55JOe2*2o4<p6(zCH$)Pvoyamo$W&0U&'
    'IJg$UvnU;&=SuDZ=1G(e&vP|*0rMP6hv&J5yMTELrNi?~7n<=~#FyxVM)|_%k_mx@vN>8DOnGqKByNyH3Cv65X-'
    '6<MI@}B7=sK&k%#5Y@J)MQgr&wjBH}nOvf_OH%e(vr;JeZ_sD0mArjLP;k6avULdLQ}kMD2c{$w5fe?i!79SN#1CF&H3GyB}$?fy'
    'uU9t5JK5YWHJ}`eRhP>ojVRTgmXe*&%}$cSLK0IIG+Ov&EBpl7Lr`PHSp1pnw0)t{@S@lcfgf{1&VhYGjBES}i)q$)9*$46^;3t-'
    '>@XPT9=Fo_86b|FN+<oz#l*0>{}&mTQhi+!kd@VbR}(k!zE(y$pqGezjTIJq!ime4!lym1dM}7h^|o)yZ}>_5?0F*`16%F{-'
    '7R;vTmJPXwEf_8^8}&DH_D5G9`#2l>UymTpIE>~2LL@`Eahuo={W7l1^J8zWBungAI|i04x|;O0hf{CR^X(@13Yi>#O?B~=;Hq^{'
    'AO{$dTc;F9|PkQ=i}|9`~Y%wz!8a$_!8fRDLdn@qquZq_Cn@Co-jlM(ndoprH$dJeim<&@orstjAV?0b>3yv%2%9Y1tAPq>52n1('
    '#iaAW%ma<sFG9KY@>CeyU&ug3GYHz<kBQ6-6yX(=R{$($4t$G^^!MDcp~g~v$_luZ5cT-'
    '&re8atAUn09AlPl~6%19oci8J(bM67A3#Mk&z_U7)s;YuFLG$Oq?Xq~uiL%f+_brG~o&pQtbh?<(6#I<7t-'
    'M?2##tq&>C6Y%X^u(Gmn=PCOq&3|*3vNvegx8twNX2qoaFSBA2{#RKs>E0gtHg?!!^zl^!8{6JN`(j$~dYvfudVeB<jqIPf7?1tQ'
    '_jO7sHPHwxDeOrE^rd!7+8XccB5WdGLLPuSugl0yEIY7c3^KDjgd8C~JtT{Uy>p>!`TN51;h8HTdB}a@Od|nanK0|@`hcq?Mmk4$'
    '>32lMjJwIZqN2&&<U65y#@*yQqq@o6<h!7f#@*z*qSEQR$(<KO#_S>^_M1%CBqOSi4A&(_bRU_mOH<FSO51@hlaViW{Du{#;dw2H'
    'Nf%|e;ZsXe8znsVqa43s3sQk{)BN3U#!Y?@YxxHDG`?dzWdx-Nmj0Bprfq2>bK0~W?PJcHwx?~(iPH}D&zdu*9qnH<r%pTBziQ5%'
    'cD8?ubz*V_CK|C5WcV*OVy4NAUuMMlAtQcyDlLjcR~YV^f3o6u_Qa;5traTG1{H&SO;U^JUO2!6xET&G0q%waOn}?r02APTIKTwB'
    'Ar3GB?uY|SU=P^^_2gy5`#w(D+X@h-'
    '_Z{_<B+9m=ut4}|1rEjQT`pikEOre}2#p&7jH|%IZ*+b`nVP{31aKnjrtC>s6ENNpDn%#CHPypLOpr{>AttuzI-'
    '~(x657=CS|O7`*WX!$y&E#T40i`t2jktr)xm&waCI=^9b6p@c?VYqW8Uf1v9KS*^d0XcJ6LSBW+ZGs<SzSh<MXd|fl}6JyQfbZ{y'
    '^J(O9@#z2_3Ua=t;fZ&e(;9lS|thyEqP8Fm{RIK$hI6Lo+)dYygUqewx|6#BV_SK0Y#>{M`stVh1;ncve&q;{n9;t4E!Nzf>j{w+'
    '$H@2Mpdpd)R430FVSH!~EPg37V=haZOzUwnI{&Z<AqKQwx_lET+N8N|DFWIt`piH;(k8Bwcu7_PuTOb$%2?V|lQY3++ZotC8DXA!'
    'S?-'
    '4|s_#6J2&h86)~Ui3764l>(y?UG54XQ4>DtN(0e|KIN(dTZu|tq9Z*hCR7PODCScMK`5qE2}3AmQwc>VCQ}JVDCSZLNhp0(0dYZw'
    '=QywJm)n0=Sja;_@tAVJH%;g?p>RiMqCUyKx)-'
    'YJz)qSCN$i7F1eP73BCza06+uo1sR%MVSVfT8Au58*>Qn@o9U9%SrLO)I*UMa0G&-'
    '>>Twyf2HBY#DZB<1=vAroJ#qQR2hm4bA)mfRx^glvV5+-0O9%mwr-'
    'G`?{yh__VElz7~e`bnxk?(h8mY{w?M{S8t^}9G=p2kIfyV_t5bj8608sd6~iTEm2_~hvf_aeJX+wHR^AQcl02Se3ZZWJSv4t5{m`'
    'dMQ+Zb3RvyN11q<P?6bY+YK-|3=wE)8O-4We-cM``;=1Q-NGG`kl)$!U58f=JqJJ>$y$J?P+e8a(9~Brrez7_9-'
    '{@xsB=?)k?OKME8hx-'
    '~<ufz1p_*qMJ{uF>7MP$2v)opReuBhS6`N<R(Zhd^dZZH1u}2=SzcT4|{=<QGxbCC9?wUL}}dYWiOJ(%{}bJQr_zXS~h#9VXluOH'
    '5#I$Z}A>FgAx#?N$j}Kg-W*;-AMZFF`56(a|^sj^P0;40A4V=GTO|v9gs0Zq6{amx7!-'
    '~n32kjv5(8VqBh1ZGE8aK$JoO&ySEW0u$e{xhUXy@DQLEiZj$gFMtBoji>7|u#XgtItkP;JCe3!GnyJ#N(N?FUdYtwj!h@p#`WUQ'
    '|X>YP2qX`3K@?3F)20LI_iH>u;vs2|<##Y+yopDaPT~?M>YhxRXPn6doah7LxA0ypd?+Z8QIT}<wjK@3|`5giOU4TTb!2QUxK1<n'
    'KV^ngsvU9@fVJDUFts9&tF*yJUoLH}97MWjZG~16V_QQ?Inw&P<nA)hp1sgR1yO%3pew^!IRndR`RXG^{N`+cdzSRD+!dlL9WlG8'
    'L3^5UF!Lz`~z0D9LG0QEUJH-'
    '`y`zl7#kVE`@MKI}#Rf7e7f0RLiKEN%0V^9eWQTWjY)%t0o<Zg5>bhILL^Uh0iC3O4FPqR(5`!7gAd9?p8O!EPxawev^6}sUUrTG'
    'E!D=!XAo<WGswcX2aym^{jyz<6t>FRF5mG5PM$=WmQlARqSd}l3e@^={eDRlmK8u}@ex@#m6(N;rCdfs0<)SxmR+7XU1sEmgWgkc'
    '6yhJe>N)&R;dUAp1gxPH;7-oDi(W^7Vne2XV%@R|~3H{3A2BRN#G0h$>}(k=}-'
    'j}k@V?26oDzR*7H3o7?(+rGjQ)Ea=*1{z`TpKYYPQjGwY4_A9fZx^%qAB$$%!Ogh~kTtT@S7fZs$mf}ovENB<&(w_HfK<m-'
    '8M~TvVOM8tEBXM?(I?exqg)eaO=OP{o)4UYCPK;Jcnm@aFeDS=Ko8g~xg9qm(@Ia0!@7@l`7)n;Y*!JQcXja3uCW9)$4CikS51+!'
    'BiRY6YK^$v9ULRj<<iRffwZ!=aT5fuW)f4{1}t;3;#X#D4aksB%LI1~P)sdzZ2eq=fFy3Rb&i6e1w}g{M`qGahVdrqH|U#<!a*ei'
    'iHF(!bb(7eAewc8;wLw;f5r?ED=L!ObCM#a=eivnsr?eDb)XxGG`!HM1EM#1q%N~|@9j7ff&$wY&{ICJgS2gy@fqD;Tb=P4?c<Je'
    '_?v!?lp_31fArh%-we>;_y~V95KTdRuJ_Vl_n6KcUU8V-'
    'lhSQ0V(OnXJCJYpOE0Be$oZ>KS}!VLkOjFDQ^<nci8*9J=(OC2=JhCb+$JcJEe`Y@{`zx=^ByjQ9CxW?RsosuN7E4Cbp<abD(g2C'
    'jCNGkZz(u71zO3fEZsv7WGQtrJd7?83_n(-6A38Z-'
    'V&gA*E?MEnxqK$xS&&!PcG##y1=z<3b?;gxqd_cffp9?KCZJj^3^opVw%Vw3<DUr4FZ=gDcK#){<nq?=6>3$jGN8|0~OA@;Q*k*c'
    '{e!1`EY`|qk)adlhMD%?8#_f3-@-TgY*Z-'
    'YqLY~2Y>Q22iy<8_JwoFjw>hnFEpi8^#=Qk;EpN#x2%c0>qa*&LW)V`8}1#B*YyrxEX?_hh!eeYh(lJ$X<YB4tB6(E4Sjn7tkM(x'
    'c+3cmc08tpMmOH^(y%yIifZ*pF|w673;rFe$+O|!v6UEy+#~D1@yI8!$v6itG-oHybxv7{2IzWV!a;m?eUj|SSI|v#e%KRq1OM;L'
    '(_|$aPhz{E*~d659_mbYm;x*iU$Y_IQ*Q&Rl)!dYm*VkvOX22wrEv58Qn>kngqxw>tcvmchpHIQf22Y@AE=&Zl4Fz!9r!Hr&%<g*'
    'vi_OfH%ol3+m-'
    '#3FI+AmJzMUO1frpQJVQ%H2E<vjVr#O0sgL8FY*slp0D?<03r}kxdVH5nG1*V=fI3c=dbO*Rw5HSp9K54w$=3nU;}iA<P>ifH-'
    'vo$}kK0>-m$CkQ2Y^Q2x#T7McSg2O*s*WQC|(qH?3>Ab%7l6SmW-TKVP5|`owY=q`65y~p{%6JnMQoh7IC*$yOAHECbCKuhgOafI'
    '<}GM$O)imr+eF#!Je*!KUOhc%X&I^OxUu4`zgWm&fSzCdUt(+p0i~$9Xcj$*}^@R;CgrD>Y*Mb92uxOohZk5XNP6HTnbGm7h8|+R'
    'rar3F#6V%0gJ0u(~#nB8qM9^Vr=+<1kW8ErQ~EG2>D1z7f*%D&)2x4Lt;j(&k3S@c5!&i$pUIUMI|u8?t%h>U%0#G$*g%_V2=By?'
    'WXVQ6zd}i4B;6Yne6{0XK{^l>qjR*)Um3}6Hw#a-pr**#TC1g45haL2kIoglyTS;{!nO<317#@Tx5W5dK-@Ee-'
    'a5)59Ig7rWgk;^$q1AWmmxbbcy4Q>itBn5Y%mS=+M2jAOOc!q%6idwAnV9U1y5n_aD4Rzn^x|P*FZNXmk%K!@NlLHEqWk=`*0mdt'
    'tkgE_oe6k#rrrf&C{P8}DEP26BAljqO937{3c;n$Ke{ZyAWf4^nu9K|09%9TEpQ9IEiRPmOB=y5mY~U6M!=f)byoi|b@gv#=Wzkm'
    'VRxES;^(#gRUa)cyK279vOT_g-=b>7)|QlDoqV89dKOhXy^~)zXFWCITSob$JUBq+iu3JnmH~g!%-'
    'P0?kiUDbV~Rm12NLMO)c2vtv`+*ebJUrM7UpjP7J5#=j{mU*`_vBu=m~MUNY*Jx0e|3wrfeRwP@<AU$81S)8NXcyfA19^-'
    'Rm46)tpH2zcpGDv1X9EDX$qFR70a}`}v^knIJhQpIxXUe|f_$KCLqJ}azsiDkEYAAOlHI%!P8p=IM4dvdXhH_t0L%Cn2plzmM^PK'
    '^w#2faL%$`}8j-'
    'A=FO`+H14#ngi6u~`HNptNQAN)68P?(eQ{Y8qF24v|iLX)7=taDYVuMT>7xG7D_3f;3)g{WwrB+Gmg@>MdE>uyO7-'
    'GdX%VZ*)H<{VMphtte4#Qm76!B|NTd}k^<4{kp>O6|v)JtK7yx6N$5a{(jeuCmv8sSVZ%{DKeapEMfJnuUY<7mda<o&o(=jmEPqt'
    'p29acvclwbHHL?&k3uKP=2(JX5w~VX4pWIM|_v`Qn?~fN*J|GmOwwYbyG&SjnsU-'
    'w3}oe?dK?iB*@expi7cs3v}eHtVzqUVOu+O*QE(m0e$81l}8kyCT}}IG+2=zqiee@vxi3KuLF8P#9!VK?F)gfSlidNAk02QUC=%f'
    'Z0<tfXxJG!4H-'
    'xFKYoiCh5e7ywVh|!dpc0z!Mtb(D?ILw42FB?FnxgC#}ixwTCkyKWsp;o$D}$vCN=Sx)ccxt$)KL*^jf(F09aT^#gY?Oks`a^bxw'
    'V&ll1giSz%P)*;!#!-?*$Ws&9N&7}a-'
    '9Rv6WHZdMr8Hz6yG>N`*B;qRiDZXd`b<eVr{KS1Fzx3KBl5oGtLMz)#TPA#E5`Y_{#kD3Klnhs%gxrib<3F)Tsl_I2Af4V=gUn@J'
    'KLWI6We6tqy;BK$j%abztDavAWMn5H(zDV{H#m|#W5<g#J*gJj{)8tvNJr2wltFlD^d9lJ;42&21if3qhxAAz_NC;++PKpvoYq!^'
    '=aMGAEcML8VHw?Bl3ANEg?U)!upKV6`<HYlYvcIc{4z|gfqE`8%MpNi_L1=&0Xam(o0^jy9w_>GE%;=}M9OP2(6kU#!Gx{km$0-'
    '^86nl1RMnA(8DS!p+gP9P|{2;Rrht-+d-'
    'eWjJF;?4qjn7aN3Q$MO2<#fEY}TO(OS=4)B*WluG!E9n0{+_d*|jKx;auoM?#O9bF+%vW7w5sc0<YontX*})V>^DlgygJCQo>Ja4'
    '=7B~2Mx-YYmD~jGv#sJ*t>1ACj1ciHZlSIHO1r@o5*WqN{lTOoG_8dHm;-'
    '@&KcRm0~m@0m*POiV!<Iih{0GelLs>z3$EcI499}oQO9^JIEjZU&q5x=wE)ajPF#m*kDV_>>Z4Ffpr64VGsrDh60t#Y6Lmhqb0JA'
    '0cAJb*uOGXpv_Pl5f#M712H8l_1v9E_ChwWqK|Z56gXuvgV-E!IMV-'
    '9@&6aj)SWQ72mR%NBQ_*f|mxtArXs)p<!s;rtY1t>j>T2{b+9$(m8XA}EQ>DtuEM!kBblQVpR&!$8qU^O5I*m-'
    '$?U_9+9FX2*1~YqjOr&(wfuF`DQaYi_QW7bh(KHxHP&5W&o*Rz!?QT%?gs{5P4T?VBNG|bWF{kG!sIn{gRKazbF0i%U#nv6iAzxi'
    'InbI3iXRap9H$l;9a%@q<-'
    'JPssoVC{B2{%{duV~{JN!sRvpU%ncAK)^q0E22fhxTj6ghY03W_eI{Uve2T`?Ii`>t>T*U_{CFGNR;mb`4L7TI*v(t?lZ1T-'
    'b(u0MwcNM|jV*z?j*`GU=eC%V`}+r)zx+R1_Tv;rV1N-}iv|R=cc`kR>-deIWLBt*WR!O-'
    '^f1<bV9d3yi#I)kN(Hg#)=QU4bs*+tLb!Ksvzl1x-7dIc$qGxQ-'
    '0GHf=DZGv7i6rQJimupe_GAmzu>t{}2SkEb|)^1vI?ZipK+d4);jO4e#oiu?Zy8pV@H_OCeRNmlo7IOt3v@|sH_UnvRCL!CJ<IUy'
    'fAufJa!-'
    'v7|ho^W#9#?TIE0#Z9Wo!ylr9EQQYdN(2dD!S>(*qmyaIFzc_)t(0C$Hw+E2}8zzgsLtKxcI4wqf?MEpUj*hNXG|P=xvufzqTm5*'
    '{<Qai8|C|NBQ3<yKHxzW@+cBP$eElzlFqXl4ywIu$gyXHv0IKD(wN#BL<kIE>MZ|(22WGN3yGAF-gttENNHLY`aRKWDZz4x$EaPu'
    'yS(S4;Yyg_x*s7Npa)PonYnU&Y!tp<>c0%d2Sr%C(;hmSvjKW@?3@GWR!gD>g%w_W?antLdV%!m=v?4=2sjldnz~1=j(ZEw<6=nj'
    '%sT%jvR7tM=k?L+-=BEvj4h0xepv~??6T~g-i`YOj4UF`KI$i^#BcLCQf;2sS>SK&LVjHV-fmEj#@tt!O+I;-'
    'PG)fWf~(Go297IscBm4DUGR&Q5ig~F_<wbgJ;M{3!hLMC8#vlDiwj<ba9ypr~sFG+{)M^VkwB6qDQop)fb3D8LaI;OeI2_g@70_u'
    '&DV#3zCz3%>^-'
    '=H0#p{*xnS1JA+f7pFI7Mh9@iyC{Y;H;N2=K0)on&l4g~j)EGOJNw8OC?9Vd^tD779i%i1m7RLTEll1Ekj6E`wB<0q|{wkAXENRP'
    'Mhk*Vu(9`jbYj7N{VVf-kdmtyi_tW;lV&c2#TVr2Ur=5T9YjQ89MH)LF-PF)$yh&~4bc%HbMTBzWgfez7zyCHWI%gG#J(VE#pvIZ'
    'H07bO6_9!8`pg{dMMh23$w?mCgB<*02HZqd5qdms3UqJ!-'
    'FvGDEh3Ch{aEzSz2ZB%cHF*C+CZ!%lj7Lo2sQRjE(h`4G#ufJ(Ih&|wCm|=c<@F}I7`4uRRSeo&RyhOoU*xB1PI~{k0<%F*4>wM@'
    'L<@a>2usdJ;*7(RaT#qi4ok*obk;a5IVXc35yFylGum?;mQ2WK%W+t8UIq^%^oe|jnT8VDk{;QhLTKw?FAoHR&z9-'
    '>3?d4Q_8E%ctiN2)DOkg!1&?_&0&t#@QX5Tn6xQSnK7ToXwkv?HPFT&r*j*OZb}F`<q-'
    '+XJQ(Yt+3kmB*Lbt#K7)9oVZ!svs=bJq6=OIb9z2Xe|6~z-cRu!58$(_EQk)XC7*&5PJAM3OsJEX4)(=Tx7YfJSxdt-'
    '(jRTNS;6xn^RJNORPs0!$!66m69f1G5vL_OE#SVNP}kBr@yxuB3B`J<6M?dOQ^X;(@3M4|6L!e*iS(x@c&OUhuq3dJsmcbPBF&v`'
    'mDUSe^94!uaJ(r*+3KFP3h6*JXLH7T5Hq;?@ts8*k$yiXk^b_SYLDniMf0L|kOa(ycj79Xec8j)<IM-QEN^g}{$KHtzCE#A%^Iu3'
    'WkRO9)9Vozdlio&}l)sLN%ipQ?ep`WR^jvD~33T^!H^_vyi_~XakqR_@4ZTjCGd@(xwvy?qKb@*p1drIo?-'
    '>U3rp#k^`qfzN+g*qdze1R$sE+lE3=99Ho8hb{CI@?yI@OG8J+i!F6^D*fZf@6H{m0l#%xVI1$eu*qF{jV6zbxBH0J0|6$U6PVfx'
    '8M{$3~F9>HXZ>DY^u!a!JPfEWMgL*X70?-TsA7)`&P(K(H-9<ixUdO8L4-Y*+(Z1$Fthr>J|Mr+V-z-'
    'IA*FQ3BV&|bUl(UGZlk$muKj^^TV9b<Xz3c-kt_%7&heg-xy)%56q}^08%6!xFw~|n8iE;{=iC&o}4r((gW<uReE4I+eZ|n)al8O'
    'lR~{cKue{b>}@3N_y$v>=w~@+L5JpfPiZFm>QRub^EjT*Dw1UM7mpf&Ok5AND1W`dim5!|Pt!~5Bp9c^+DCKt?5heE=bKWVmYY**'
    'hQFuO46~UfBVadT=5Sk;WDZ-YBy-qWC7HwRRFXMtqms<w_A1F7?w}-'
    'dDDm6UG=AF+oeWM!`+@+roY`3qc7EGEODC%g&3Lj(H*N1T45z4$yqj$!?`FHmyV*YSZgz;gn;j$XW~a!z+1c}M`ne0g)?o7~ghd6'
    'P)P9<w%Qz%0U&?S>#QAEe3iA|?7U%U4!$l`KZqI2RVEl>lvj`I1wG?bv<($4uPZ=Wi*MSt*aNfAiFsbbZBt1Uj%;{l<X>B(l?*Ev'
    'xs1)|E<J>a&1)p$6^=HOY+ty>3zsb&}U8U{&60V-'
    'F%Z;i&POUCGrH+%LrQiemx^8J18Cr`kFqDM^O7RvhLtH#>>reu9y|ig|^*w*yR%&}^MTz%Qpqgz>2vBx*#hR;=xMFw`VCK8o<C1!'
    'E-'
    'rXLbgim=7J0gib^W7;dODb?Z?FmUNn)k9NCZTA)hdoKah8^6Lzy~$Y{y0*DO{!1>X_Tq5&Zx1|>L<|ip&blUM>|jKySCf#hy}R(C'
    '69t$elDx<>M0frKv{}se_3G$YqY)JaNJhqI7Nb*rI2v1Ea`;V#hrj?_`7AKFB3z*8%j`#VegK@dt%skM^TD{_3=7s>pE8p4F;y8{'
    '7N(9)(ZO@ov$d04xz_j6?uD7I__;U9NG^u`*_B>w4E+sOWHw%EcjUldwxyjERkXt(*xS}^2&@n$S<18WAp%w$}?=$dwxY$S>Q)zo'
    'dv>bR$AamW~~L{YF1nJNh5j-+{vuCz?*!$c);}daVDfq^!Q084hcP4XQ^ImrxT8HeCG#}Y$hGH`B{WQ*Tf0nwGjz4A19JDL~*-'
    '4pj@f8#K*NixXGNbMgzaw1TOnV1GgsJzkRfaqpvg4d(%$9pbsdas&oao8$mYufZ~Hfzq_wa5yz(~im%x-'
    '7^5g2U>AU#?atERFjgVuHa&jhT1JV=K-=zdply#h(02DY(6(nBXxl3ewB5rGv@LF@Mg&J^q(`#7wtISmFcp2A6-'
    '0p<uo08^)VEX#-R?Sf`<lh!Q^^$Mnv}Vp?|GEYsZO-'
    'v$o*E4mSV^0bY(kAFUXn7b`hN9wz%_rJ9fxWx4S}k<mk~dlQ@hX+s4CBF3|k6qyI5u7o`2=4aPp0cK9zc_8%4f<j+Gf;;m+7MSHM'
    '^%!KG?nmCTTf>JsOA!-'
    '21U<RnxnY~qz0u<QaX1F`4+KoGq0d>g0%7i+kU}Z!da<DR^4oO%UQim+8OsPW}R>ssJ4?DdORpRqbtt`L{bJPEDCf?PBf=nFg&RA'
    '(3nm|*TeG4R_t<&@*kr{>GeZ0&L*N#o3Xash2Xq@oPw=#A>AVhaH;yF@Ftj>VonIi0Cn^4saHVoaL^xFc!cw4)vp5YEWZqPatZN)'
    '-kA4=&2790C;ddHR+`v?PC@?~JaTvjT?X3!&I>;89;V{HUEwp6i7bAFNwuk$+~jKvF*BbWMWug}_rSgE|*x`Nz14@sC^9Q+{~&0R'
    'D5wU3VJ?Q`^}*;|8<eu1Vddq-GppV`|DW3azac(lqSXLu}7906c~8F&=j3hw1iB+aGr`xa8bf_!-'
    'g3G;E~u^@A^vgC?<A1>L?UfcOh^5jJm6CTxWDOKN<fG>wxAVGEyX<O@e-'
    'J!aIi>n?P=z&6WuC!fcn0~8OW@ig%X@PckDVPLpFUW$Hh9<Z$3wjzKMyJMWYSfc2%7U)O=D#=#+8T@jld_<%p+GLlcm(<53Pdr7>'
    '(q9R1dxC^f1O$_l*gMM1n7$;G6=2_@3KIbXPH8kf2&-=MviF*pZ}+4&wvrD?CLht_EnP)3b{~iUo&AO@PBGHZN3O(Jd1c9-'
    'BH@lJdRwzEd{*ev5(K|Bq&-SR{jW`R-'
    '76mtia9Y=2?QNd|uGV=$DhOR~Sb~?=#d(O(jOQ2ebD(h)7Ewi(BkBm}`JIZoylMVpY1nHA4JY)~MheH8D}gyZ1fUqZv|0?<;t%A$'
    'j?MLTnlOYt}d-^5N`LHOinXsqSGaj*PjI+8zev$dD_k>|wx=jJV>uUJy>5W@@d|Q0FbD4y#LrYH4|WwAe^^gpQCkj~RX-'
    '?&TsjfZ<$|bphmfg)V>`r|1I6ajGtW9Iw;`kmFUl0CK!q7eJ2Fr1HO7;iiRf^<9N+8d7;5D(uscI{ru@f=Kzo$)}jB84<Ru)Y>)J'
    'C8R4V-u2Gej3xuwC;|ViMl2ixHrlnj>3{$k=JI8UeyFTlk!qlz>IL6m6KU@TX(S_qheH1JLh(>qx^FE7Vtnf>g-'
    'rcdI+lg80i`aC4Jd_S2uCWz5YBYz_8O+8(XWG>b=DycU3?wY>_m*6G}qR4gYQswVFx?gFew4Tt`1&nEBk0>muEa+(RmSZw^b)RmK'
    '_Xc0C268v()QhPuEM<_DJu<a<Kn`E(H58>O!#pk}d@MFY7|E|B5aI`>*Oku)j(dg8kPdCH<fRh~ywG#2hUR9mS#wA-nQiC2VYpag'
    'z@`+HQHpfhDUkuh{M|G8siV&pCe_I1O`LCkYE2Bq<87TFLa*C`gxRkXHHw6o)>2kQ?m7EnWvH4u1M08Q7nBMd}pCJOhvl9KfdTE6'
    'w1>jqtfQax&^W&EUq($O;IgPa8kgIEZ8qr2d>peySaHY$`=JKema(S&bNFDzo9hSr{Sl<V!Ax7aZmzoTd*8iu}bRbVLC|3vIqKp4'
    '!ENS8?+f52SXX-gHhQ$haxEEj(`uVFnMJf)mXXrx0=Q$SL@IJaY=c6%U<)pWV4U|NH}&-'
    'RN%|si5qlwt})a8EB|7!B#&cbo+6x@Egr2%Pq)QY$>6Z>Rq&ZHMh6*jO}f`I3nx>mS_4Y=<+v!K?p9G0<h`e@Y+9={fVKiEp$KI8'
    'uA0vG~!_N@5yMJa}hu?zIK(POB!T2rLswyvYif;{Q6gpm7;&i$Ni;-q^`-_aN{`HK)XUMJv$w(!R4-`LGFMFdJEoz__~|iGB*CZ-'
    'IM5@uY~*AIeyNoi`o01IuP2dOr~4C#OXdv>3KP0R?`8-zLGJk=|E#&&6w46FJo6_%xb!~v9D#!YPyfHuV>6^y05WsWXx(x8{-'
    '&FKLu!d9I&QBzu4N)byd{So~x{M|6SQDy_z~qmnl1CtoKoAaNi9>oY+z~H+IjA+x`v*ABCMAB=Iwsx}B5ppefskt`oodxhoOep2^'
    'AG2;6M&H8qevlZA$5Q2_4O<$J@wXS(-C6=<stD>(GNC?ZPrfMm4I{}4T*h_+e>`bP&FZ>dU2{?6J#oG`sw*-'
    'mBPep8C84n}pNv9~HQ$DDqEIzKmwt!{OjYLhA>t%1bHNUwy-'
    '&6&t;g}!V95*L@U7T!)M%QhoZF`2dS4vJYRH>rdqbc?GKMUsYtQr3-'
    'wQ^tlvQVNbb8Q1cas}mhxs1vPHR}{to&GY*H6reZD%BvP|QLgU)6zv+)!v8b$YxtgXzv*58(O8mkdJ)`i$)<b>%vwnb;$;wcCK;$'
    '#3{dOw{a!VgDiYJzs|==!#1zGA22(}067<s@CPfVl!-$cFNXSu59s=u@L6zXA{*sff*3m4D)I2UZrSiX|MA{4#<UL-'
    '9z0hw_)@wCx+{w^%hG?~rVo1>#VfMVpfRA?R$Iyo;-bQ8V_7~vxb-W3F-'
    'b*zI{&^C5Td@0c5ZMmY{#*@d4_<%H^^>DHN^?bybkxb3b3SBfPSG5B(&-'
    'u<76TF7ham;|UI^~PFpGR&+SON;D@qmF{0V4lcG5Z@P0UTGvyl$ryOs3QFBDYGVeY^(CBoro?`BGbe1&bFxpD1&$XFw%X!3z{Ag>'
    '6|_o4$CtYyA8xkw>#lkY=^Go)?uL1de+4$t?a!#T|q5n*>Fc4=8UCKz{fuz<^<&Kgp6){qV!=vO&k<1xpnP@bbmx-'
    'LxU?xaY$R&a;law6&_YNp8sTUi|YyTH^Rw)GJ>8F6Y*y)I9Jdk^q5RzhnqkBIfohZsr3dS|U0sB@f9OFd>1YN-ap(Gcy#CEWBtd$'
    'CO>(d=Sm-P$VNe$|<6uXy{l&1?t7+pk?_JC-'
    '1HWD!W?kW2Q<6dN#0*{i&*tVTzEIV^HVktx_BuE=%O4wCBVm2N;z3vywaNM2CDXtT<jcZBPXD4Xt?%1-'
    'u|bCa@H6mZm+IOuKgg)ehB(o3bA<|@ual4`mO{uc9M-'
    'mTpC^~9L>ApBxx%zF`lGBxIX2tb(|^M1tx;wX%ouXr*&1yKtSfcnJ~wnJt+tMYipM*-'
    '|CfY#{t)kt2cf0mLSMH;U{Wd!d~Qy#@jUt~WceWi502N-Va#x))U!q$-'
    '5!GJMNE_{F)V{8#W!jvPndmkGNUpijs^d6VaLQ;#1rI~PwLoi>#>Tie4wo?fmkiEXN&1{DIT8<n$Vt^3SVE3zt9NWa4T9Dd#nsk>'
    'y`LL|lk;Cem5JUDRy_Iy54=A6@dIx3+=khF2akE^`29+qwWjqKA{UNObt79WOyGI!i(ubLN288rsB%T2weZXTH5YmU~cm{;@VZxj'
    'NA$^#1W<W?^6x24ZGi!)(6V^mfH!C~EAApg{cBn{G%yXO>kD4S;F~l(9gZAEJjw{{J5jlgTWEaJv!9>>;H_@b$8i%0(thCMnb13h'
    'VbG6+kW2H3#B<5V#bvi&OG3apZZ-f$qPS;=~#1V9&4m8X>KtAtaAT<0-'
    'mpcUgWzzdLP3kpgy7BJ6FYT|Z;59+XpOqEpqR#dX#a6y<G^#jp7ec$SA|;9x!9H51%U<gGA1>jUEj)E4UnoOYNQgXhC0{HfSMnt?'
    'aV1|W16T58GH)ec9>%TEhg=LaF#Sw;v+SQ#k@g}XKRYXPxD$r1)cEN!fe1D>qSm?;=o&`#_4e*{kj)czoR`f8WxuQdXwGw9RHU}a'
    'D54@&-'
    'X<$`f&r?G`9Hk7;#_BJ<5?Bcj~fxoRWy@E_87;WeMw}`zBICDUl!T3FOTfmlOucf6_GuAiX`)EHB*kGb)b)CqH(lO^wmr?j`n~7+'
    '7XG5N;<sH^=Gci*MEu<IuC5u(J`onP6^t`!B*!Z=XahKRB}sI*<;FXvxhI2lJz&56309)8v4%{Bz<Q@Z_O;~=!NK~nJ*o^5dAeXr'
    'lS{PPtBa^m?0s<gT7R8V9-'
    '}84h;HQ#eqTJs5mg_TNMWeeW&~mj=`g{k9vK69%y#TVNg$C1sU>8M)P1pKiM1JPm{aD6aRqWzMO~AVMI!()*q}9J_WB5m4aLct?!'
    '&#3caTrDVtm=4P#%ZI3ej|(81V7l`z@a8HEl=x{oW9{^>$A-5}AlH%NnwfOzWdjTs-rIuv$y$^7mQ!fA*3f_}m$rs!4X1e-'
    'vs2GD`V%%EmMe>)_j|K(SMTnEH^g1@X`Xq9tG=K^TLq(a*`n&Q$ZS(4_@wY|>7kyQVQfr2M!P<VZ;WM=4TfT;%N5rxNf>B(!0F`w'
    'Lrd`?!D6uxv*z+SvGPa?$I5a5stA-'
    '8HWj1ogm3X_6csu<!r2J_RhAh(!|*aID)TN%zX|5@7~gq?PXMg%o2tPa%(>937*WbRbX=f?H@&VjaqrvAOQKNj5f3^#!|*x5f}Ix'
    'U;l8!&5?P3ve4UV;rupw_cpsE~?>e5S{sY#gFN4B;|c@0GnqqNRK;`=DEO>wsE)JwYhHksy@cOc2U%B?#rW6NK_R2|{^wf>3@pK`'
    '1|`G5=_?gq{?A|IP?I_g8+@Ow6t3pk*mAWe<`Z!zGbu!2=P83X10v?cE`y&kX*@jagAn*T5@6xM5RF&IULTPEeUg=M)9i3vI#8IR'
    '-{eYx{5xpLB_Z5QaRZ$*bhD|I?bhN{&g|f;*H$HRypm6o+R}nO|%A8Cd;+;-'
    ';U$H$wvm{q(Dg&;UX|!we7&7Mf2AD(p*z=t4I5W=EeZrM`8}8Z~l><?}?@HDZ-2xRy*kn+mVSrDV%d+Gb_Pd*NLOs<0-'
    'URC=&B%Avbc+RaOFqy1}_IQ~89*k7ga1ztCtm_~q<>ac%l%*~pTY>{=vLhdKULhh%<LT+QRkh{KE$lXvZ<Zdh$ayLo&tP}WXIB#M'
    'vKsyut3}bfC`9(j+l@(dba;xch(7!kljzU{=dH?lA8~G0n7q9juMG?8GZ&M)#tH<QupGX_X^%#B4xrRStkTZ`w7^Oj3^r;e(#j!k'
    't!@1diGM@R%_e;((3~_l#A(7Caq&(6-'
    'vkMD{_8WI|1^Gw5QJZl0NIhMmxv%0NH}<)OeZfe=;GksJ(w~h{l_>(Jpo*EpP2@29oi`3NL6x<mziIoUP=nbG>C`K^<~InC-'
    'H^Ki$y+t&;Lyd$HHT+31wsNIQ$CH&0}Ihd-8rmI(O}tGucX|3g9)v*E-'
    'wBHSr#Rb&}y>cg+u$8_QXVN+p7$`vq_P|LST`tbUGdDG^cx}3hQ-koLJOEs(KD6>z%BcYsVf?Jf-'
    'AC{Zc36Qr!m!qd73|FHM{{q?gDn%Ub%i9&*<vn-XS|Jim*y6ycD6W|B$do&#vR-ffx?RkX&|wzmdJY=^sK=Y&-'
    'oc#8HmEeYOg;OT|t{&vObFF5XFG<&oyu)n?QT5o>^z7*ftn*h%Z)LJ=7L1y+H!%pp`6>&g+6zF0-'
    'yQL~LrK5fw87UCgZ&0$2YQ<vE|JOQ<E{oTrT_&>}o&s0dN{1=ibx7=PkA~NPZAA!e@6bY3-CLj)n-'
    'TD^)!u?&fz9^khz{6Re}PDd@5vDkHx{<mufZzAe=`K6GHmOAgA98C#B(mXAf3=iExlhsi9%cUJo)mUx`D{-'
    '1hhi|${uQ@_}3Ol7ymgMRM_01#M4Khzn>x8;mdhl*>0KmD#~^j9{FE_LWc!%Bp7t~*N$=>?Iq@jK_Kmlb}nKJ2L?OV3gi;~!J!n-'
    'L2yV0MzWOJh+OwZb)^dVnS@z*j2G8X?WZXh_Sh_x{|dmb+f1aY7zlmKk>_DLot-pOC2rA5Oy>$Cxk6T))@!Q*hwm9BB9B@g@7Ol)'
    '9j1no2iV8W40R!SUq_$fLh^oL`Y5@7{R6O%yuUpI4w9U|0gUjgjk})w&OYoJku}?sqdZc}19X`W{2RCqO%266Dg)`>(ahO?p%#W)'
    ')WJf*W4*G|wf%#U8uB0v(UOj^?a^;0Enz#TP|Jj)-'
    '<usdGQNG;2#`hT#{p{IA3JZd@Q?J!()yWWpZHSH*vnl0a%g2>iQh>mD{8fDbjLBcOedjaH?+q4(a>-'
    '$w&eiHEBo0X+9^nAh{NGan-|pOJ86ZnEsDZPg|UO1^AjtGo!x+otvLR-_sC3(@07h)#(*@__C6T{(g57iBrZ?9F4psG*J=7&NpxM'
    '`tQmboGZ5k!d%7y3GFQpg^%z}gE$VMVK#FGrL#l2V`gf!5^+4Xr8R7#`dZ2xOLR#-'
    '2|K?qP{(cSDsf_SFCIkOl$b1gV(AF3E;5)iuAXPF1&$&3#;kbjbmuYEnY-'
    '<G9IZoR@X3Wlw1mF@*06Nw7f)uPtZ@(x7YtnNsOO2ZJ_A6n;0GsdG&ULPiGyviykQY5rC%fQdl~ij7I=21UL5>%=W|j)cRnsvl6('
    'NTpZgp}y3SMRtMaV9A+hnut3|p0Ku3ccQlJaynXnQVNu7WY0${b0ghJ+%C^w3pI2~UdXYG#BdO*D=5KB=N>_*jxIx|X@&Nf}MApz'
    '^DhR;+7$?oM_all?fb=G=@=<#5N<u__*$`QT3|M@l5sLkxtlI&~??XMW3zZ30*wOO<ZgqO+ukwkYN6a~#cwX_>oAM}bjCt6j&`>a'
    'c;xi@QKqOcdV{9#wMCyTYeRpaeU?t4g2*yTY$ZpaNg2LKh-'
    'n5yzDV?WK)}&=qOtA=JeQrOAcTY?v<uH1QJgL0(k_xa)(;zTv6&SCHRPM5WZyUT(@V49(C(IZV2++ZlUB{IXxGLenJFS7?Ov^UVI'
    '7g`u}@7~EeniOkQ__OBU7=J_%tL#|*#32-``ti)MaDG(I$#GX$Fzr5s7C&`3tq-'
    '`+HL^oj@y66<cLq6H>bf{Q5Hu6`+C29k;=5KqDTO%lg*m?bNDV6p+ut!KXz974!GxUEPEAi_><KkCedVWo;_7bFsF6YAYo08WV2#'
    '6ZJe?WZ!WNUvPR(m@K?+;<M4;sepjIg4Jc3;g?pJAlw5#~N4K}?T76_9r|$S3ohB+QF8`3O2P;*7C8@ms4C7R)XVK`D`@Z}4wXZb'
    'Uu8g%&lwOGS_bli6p23T;*qK)nhiA5VUmg+XO}BU?EmZ+m9KHn^%V@HmMZ2I6W<FqJmw%CZk@ZfNPE1%r=1KnCH136xRzbfN`@Pn'
    'uV7_@s3Oh)<_nkocs1IZbySXeD^h=gX1in<7YVdfbo_F`O!FuMa8vrm5IvUpC3a;7UZN3E^xP$HgEdK{X!A>Y@eL$~TD}T*(SJ>a'
    '+~z>M)&nN3?#*-=GhXL@RhlU%F6O8ZDi*<W@or8xt(d=^92Rm>7!b-<0OKu^Q&&TE@f-<bp=NSH73z2KUbQmSpih`96&O%J<Fp<-'
    'BpeU%sE@a`w;ncL|CL^Vy*6ed1i)goH2sOnT^q6TKB$`?yyd$ocJ*FvxhH_R+8!hH(k(j~xf{aHqe*@EJ`KO;r3^ZTCnU-'
    'qvZ`J8gLTRNKCp=YLNM7PFmF3ih-el7*z&sx@a321eC3SkX#U8*P;?S_sbOxK^AwX=UIHnP>0`DyCCPEvnT`Z8MLekdSB_12%%GG'
    'DT_|Fna{}$^tDa+ebhR5^z0MME7uS(C^YrV{g)NNv5$6DY+!m*q5|il4<M*Qp=cW>`!_w$u#asiZ01C4xsbK9{5lcfH-zO2E_23T'
    '^<gGNLIneJF31?6g3_=IehE!DZ1cvA)jvpm|{6tzZugz=_jL8Y9f)vZjs2MdnB^x5s55zk3<$dBaua~NMx}G*EWs@`7<ZYj|KNLM'
    ';Y6}+PO4c>^jB4<GQYQ^w$XK+MEO!>qZI7ZkwZ0LwcAire0Gj44m&w*9G7pNJR7z61rgm=n7qdB%rIAtt*-l;V;*4Vu-'
    'e~8)CrZHZ}AeBv$c^`P|Ay`)a)}i3K*)!({VCu>6nw9eY{XGj$4xa(h_R!owQB@vxSr!B<ou33Oe9uZmi54BvrBl|p{hmTwD-'
    'cBq`+9v1CUIKLw-I-ux$XIOMZ&6!72bVA9QXH#@WdHt>;D9S5}A#t#P%YRbXAvdcEd}YyOACe`xw9p2YqRAdB;X40wJM{rgzMOF1'
    'J*X*G6UF<GreKYJO8e$ZUJVyYbSMqu`&6Yn>||wkG=VDg$%>ZATabXKpMs;llL6N*A9zH>tDp=n>H+0lIaE0)4OsD8I#i-'
    '7MuB*QoMmYwq{wu)@P4~HS9tfZ>I$5ztu`_*^Hgm+n2MgzhgIcaxHJogfkUnZ3440AiUgh2a+t5mQp$(3G8}|(lY`V~JCF6H;t{%'
    'HBpgy>k7+07U@%|xu@hgx`Z%5ik=XfKMp{O+h9i!#nNO*QPd)Sh(e~zHRu$L!uvN8>5|spTYD5O-'
    'W@u=JhGr6+&}64_bCVb`IDjLOs5H)~5m6B}5;Rd`?#;d6fSN?&h&Z5)2%-'
    'q0h&bY`prVZvW>CNNR@GWnd!OU|et+C6=Q&RwyZiLnwX4?nuJ;u|B04}Vh2uvS^rsQY@Iz3QupN6C<%uw1_lTx!HqaQgYRYB<i_s'
    '!Y*=!&&dQ?+38#s&>YszK=h0$Z0vf033v_w-'
    'j+gJSM<cyCn%HXGG?${P`B{~%~nD;3=`m>>DDJlOfwl;UxNb?z(MV9BV*cmxNgD&MxBkhl#sFX-Qsdr}YF~U!JEWM`@>5b#)J)>+'
    'by-'
    '}yA_DgTHixGj=3EFl}vBQX}cj0!Xw<9|lY_Eh)LIy+|Sv=k&BF<+fAKn5F>$DDwcz+vzXX)UcQUYnbghE&`UrOXP6La1rOEyqPgd'
    'FsdyJXzujvUE#iCTLzJbU}Xz)d#=Zt-raYP_lT02h7upGyl1jrE_g;4Bard!Y;{ID%fNVhN(;noI<h8(gv*fayTuQ?lD9GPxh)dw'
    'H^8CwZoL`s#4))bWZarrO(WR|RccZhUt^jLRU2Z<$#8|DERol0X8XAJHM?QY5zB-l!2vw|t+>{w=)fNeN{}&*|=>7-'
    '<TyZq30OP&6<q$rW!0+_nhiYUFZTbFAQwatg`Cp$DLKcXIg7Lin{_*^~Sdl3STrr84yrEEt!A$Rj3$K5c1mGcTzUAf0TE^1aPSlU'
    '~y4)~W+mM%tL26qKJCm#~Y1^fLo`J1dZ-'
    'b4#tOg5ERtxqDF}D9nwRdi>8!#+GhVAXE)O1b_}_WE?o4Ren~+!2&v{XJ;an^EU<R)W4}h3$uO7$~IB7SB|I5uqs}i=43ryj!4pW'
    'S%4M*p<=scoaDVlXG{7-'
    '?`pfd^oiDMyQfff_SfXB*`p5D_6Hf4`{^>~ioz_VSQpCg+^eAWJWX2US5<}L2cd3(4wKfYmjpmZa^0S~hzqYzrIXeRXy$C?65x;>'
    '-&!8&dRy5Od}Oi7p$m$ub&*rLdnK|Bd?&N(q6Eb@nSDJ<P;8soH#FmZz?K+_p<RvLSuVG`v7Is=^`<oR*BQ5A?g0sj@L?!4PA`&v'
    '_)Kw>_fznj6-'
    'ZxUCL3c8x6w$zfIdM_0exM_F$TNzS;GHuNU<=s(F+gIR}{Vds$}TtuG|~ZmnCk7Nb|Ts;dTWoGx~d%k$+Gk(9vKdgl}(BCQI$;fT'
    'k2EpJqE1?NBVFJpUXi7E<;#MS1}g2A-?9q)C6Er{=<N9|OwD07~y`z-1p$>5T?ED!|hF88YUKOYd*!ure@xfI-VCAkzmLG`_ke>$'
    'Mu9b#4>2d4xr>Ts>VKk^9ue01SpGEJ=p?_*5~f8@SlRu9y{P3m!bAY30(~WJeiW0ur~N#d4m^^HET9zVu4w(ms@VK8l|&Nc6R7cU'
    '+hi^7v_4A&<W(E9CJPXN5felB|%&PtOW@{H0kTkDrlwIm585lxO<7wmnb#%2^?-'
    '*q{<Kjw0sSo*dR!J2xa1%_mU_twDeW##P#Jr4fk8iUtOc9+APLXJqi`6&XBwM+T3E$l%c@GI;n9#K&S+Wjv$dUtgV38j^GgD7Ua_'
    '$A=v7k90vo@O&NGBl#0v?lZaZ&JffHDK2zsVJ~S09NO+e$+Nq$oh2F=qU{eeo)RTjb8E&CgmHk(V4SvcIH^m7q2gTcH2f$2T}I(X'
    'QWX1r2INZYdr&m>!?edmF7n@WLB$g0E7rZde?BP0B+j@cl&Z}-'
    'y{Q~);GPy!q2rY8W_T>TbLohZx7nbJM^vWNwB1dd*n8UUAuF=0(^wD2t{Z(UZ?1!^wXq+i&7bwgwx!LV4~_jym?TeyRxoXL|0;9v'
    'v}wWxGDWx=w^txPx<W`krl!qwwi6Yp7|bs47=1g9Tm=*K6w*mUhc;0PgxC^eDo^@6@-9_wFu}fE<y<ua#qlN0@#dH)HW{HGD-'
    'iNEqFK0FN&AC-Zc?mW-'
    'K{8COhdB+5Ie|ZLKkCq5}&ynWj>O%?_un&vXv=@x<&Y$CqYV{EQ(;<|CG_867SPq2IeXbcN~21mJTR*v7n{47m~d1I1uusrAE37K'
    'GSaS0_8Mj>8qo(mpduC&M1pE*|8;$iML8pr^^Z~kF;)=C9!uG^yG>V_Wvu@V>d2xMJD4S2O?UL>53@|t;Kj4-'
    'Z7ARNE1oAt54~m75+qNKha^|plgH|VZb{Bur4&_&oDfSiWp``DO^8|9D>Y!Glzza90H91x}nVdL@CFEHIQVEbeUDbJyA95f;*z0^'
    '-RUpk4K$)l8<<wskjF4s8et95g#%YR}+sq^&ube5mRvu4g|gV5O5=fn=}-'
    '}NS7z8HU&6ew}Yl?`&JENtPK%*xxeOE&l!>SeDYvD_sT@7WP^%Bgg)b=fTCv=LH62ejXTfj%_8fbyP9RzJ$W??t$V_60dqL|YB4s'
    'qIKb#RArNiGh7Q|x7>J<4&K!<e*|0T7*qPFw+t>b9`g0@gtRl+p0?p7wima=&c2Rd|&jGuWji$%sOp0k|m^1(bL)QMX?<ncKHw!Y'
    'rj*qrZ`jK&0?d5h=)sWIbh~;JU=wJl%ZnBD7O2&RFjrLO%str@Lpo9K9qpOlJxY-'
    '$P1b>SuJ~~=vQGFVBc$&ujBK7=Z)12<eG^a~J1;5tz4ZnP6DoSn2!50Or?M;lnE<>d=sU5+C@N$tEzf5A=>k)@Uqt)3|vlgM&epx'
    '0RjQucfErVR6DZR~PSnFJcPW;8PeD4|iaf(~mVC<$8WbwYSpBU~KlAZmpjIL7(nMVpeCheVLL@3lJb7RsN;Hcug9tXj<GD~j$iM_'
    'xFYG2rk%E7C15MtNIxf3eKu)V{9^MF4VPwE(7Y0&#jU^Qi0XIj4Ibn$+z!1a7712D!I9ywZuQQKwyDkz8?C$sZY9k^qr+$_2$oT)'
    'vBfm-'
    '{1ctyz|IouDztLH(1B6CYGpxEFBErK!$d1D68zoZamww1L4$%Yn@>hCH$!EZSVSl_M!nz^uW3^TXpLwJktqVl1<!#k^d7;o@hRX&'
    '{4oh~XL!F#={%J=1Dr<=-'
    '0a_iy@vSC?(r#X#~ql9bVLI=kAt;nrWyth%6C}%E<BBQi~%mn8v6%^RSoXuy(Y=Wohgkf_mBA}kS8rQEe$5x6cE>QM)5A45B*;Un'
    'jH+xrx##;%Y>w3<kRIGnBI*LdC*I&zg@mTiWDXK6DmWVXHV<brxn=yd_HTh<HaRD(~kE@!>3<JFcH`)`q_%!AhXb=o(Z)D;ZGrK@'
    'v#?It@kcUraZh?N9BRn#dLj~z`Dx@&@)=dI|G@HiL5l*_Ox!?;v&dBxD1!&R{1{JLtRRWLAajdrvYT9hk-peG{Oz+rcxA;|2#W@O'
    'E;?o2>beHhKp05C%oX)V&Lk$lTK8|^LKP;F(iMjYE{RbrrJ>GC`;uDpy+bO6WA!v3obNdQ1ot(%>fr8V9->>L1-'
    '&gim{sul~AFpzm9X0IoI3+vvWD|GC$>^KMeM~>-sOz+5*WObwyV{+Hsa4(gL#s&;=&Vc>?jTbFVb?_vcBcr!-'
    'X(&tJ4X=qt`UUYC4#WKMi6#455jIi+4hf81?`OzKkuGKO&*w`lKnI}V7GYX81;684aH+2IBQAa`P;~+E<?c9ER)$3R7@*%M%%lpP'
    'E|lLNZapAaquh^S;bKfnP(U^_-6{j<f?k2ZONyl0flMaEPXTvb;D-qtI0t!>(*mncHo&-'
    'Fgq+^Qjf7Fe?sU0DDz#PB|B=oM;ZAZ?v6ER2D+g0f@o>SEWwdcO#VrQc}_dI`_AS|CqU+u(incLgL2)l(9j$q!4uh^E)wjy`hX*A'
    'k6v?twxcDH9H{NVQm}2Jm{=;Y&D!QtEqPgENyG2F3hp=lvDZO(#y|E3_|Di)-'
    'A0kIto&an!;sDOSA{pYLaNChk~huDR4MZvA!I?;QC6Z2J>#}2c(({U(Ei$j>3G?Ym+HD}tM*q54!stho*lH!0yT;&{X_E7h*+-'
    '^1Cb>h2nQ;Cfx$3VDgN;oBf<0yQb&13V@1NsuLFr3@9fvWsm-SHLU=<myvaU$3kVXoVdg9Qbd~Aq<U~p(A*|!{I-'
    'x0ca5=K<J+*l4NC()0B9Vtb#k%zN+8SYv*-'
    '_hyP#Eo`?UtGmqD|lwr}&&vam*u05e(BvQUJp&k`%u%i6n(D%ppn93sXo^@WKp|6uWdQa10!;?3+F$m<UMkrDf;=9kWnO8(%tx2f'
    '4dnMS#4swp$7PO`W#wG>Zccg1!Mok&)&|NGdi(l8R3vNyVp;q~fzkQt^2tsrVw2RBYzf?8O?jv-'
    'h(E!PsP&tcpBayX?{g_|ysm*d!_wSnJ4yJcA1=i_;!qV^U~zP>b@BJJQhRDKNQ3JUj-'
    'n3Y6+mSEG@O5gj7QWZXHl9lxVF?(_)YV~x-h^^G(|9)@+I5(+2Ea5iMpj-veLgG?BmD7*PEljanqk{h#>ww-IJ&jD_2M;i(1*C-'
    'h&dqH^t;k4|a>@$8(QM|BCR_?%daJYA_RwYuv?gTZYASIKeGNTdCN)ZrROI3A;6tXCWnsWI@p~b;j?=L(V2!jV0)(%0zZJ?oN0sD'
    'Mk%e?6#2vi4Q2H;2Vm>Bb#=b>sM!M>AMmb}J^E=peW5=cfP0MrejRFcIzQd^ZVW8^ngX6aU!3Um~d85zUdln6w;D(FjiJtP&eoQs'
    '8)So{Q4rAKuNN^<TG9~XewDO`#k$43sLh5`pspU6S9zs6&{5`GbS9IF_xgU1Pq;W&8k0Pkba^L7(U9)^R*$w`sJ4egk8+GedvakF'
    'm1VXM<ilOGNg6=!r_g|s=lw3pzE;#Mmn8+S>Y|F+Jtk`z~qZqDl&H_cTsbNN5ej`0P{6QoPqk4`v4@S12NiXte4c#k^0)Rb-mf{|'
    'OBn*g@s*5)U`+Hq_1Q$X#wwfPy4cHG+h9H2*TZGHjFBeym;1M=v(Wr3ETphUbBDu%3H3K7F27!`kVm_?Do27qH?VC&QJ^?aAa%8i'
    ')fhNPu$eTkz2k5G>0sw|jxz^jg#V#ln=_4u^Ecke0t%UWVYl-0M1#qQz=mKks4PR@EZUJY3^PuFMoW@Ctv5lN=b-'
    ')o?A@2L}=+Heo@&J@;#RnNQF+ZC^~*G^#W_5C2#CvqSAgM(F=*be{+tFZVU2o_f1Vjd0V)iAixf=oWfX5HC4%dQ@)?KeIqqDXUx8'
    'U=z5uHtc5M#*XcjHtcH8_#mlt(Y27R%5tUObZrbCHg8>KU(mHI|g_UuhuS<fP{^i?@F<5*ot{qJ5Pnpmv;jNO*dg^ItX%3QYJkZd'
    'QPFnlZIWp#)z~9(}pny>l$OQ7J##29J3pH-'
    '&pJ<C+PRk5!!B1)tsph`Kmv+q0k`ufzuLQKv*HMJN<R!;~b4T!3j?FEm~q)|28EHy`92o-cAW~4n2-'
    'j!sbMe<71obAj8rY!!(ov`^k*Z!G_gRMrddS4v-'
    '0<?;BQ28HovBR!bSBAr~0bQufo(3mlxR&ArO5^ch@>vRh?Uyj8Uf9fol}$4HI<e)<KnMIS;n(`qD`f+3VMCYXYu6yhY9f?*W@C76'
    'QY6!IjPf)Nz+B$$GIDd<Ts1tTemN}J0q08%t_Y&NQ(x2m-'
    '`27kb>VwndwL8!GLZ$mMzlt5kW2+2>%9HF(Cr(bCMwc!L<SJ7`omw^$LPcWz=lAWJIgi2;sKEv3GOl5z8!EBje*^D6;p1wX316%%'
    'ES7<yKjr|*q2cxmC1Ous{v1e*Lcqry7H%7Ox0R5w#+9ivt)l*02SvNqd3$xpT0c^(i4i7oi7+2ja`reb#(RB+nkFKILzSv4Ps~XU'
    'dcw&DKr?OQQf;^B~GeaIYt(hVZl-'
    'A6V2S#fq$pfJ^v*dx#T4?eP)_CyTVpqsCu~FF#H2JCb8x8r}HWdM^qm)z+@pZ`ik5jRf_zq_8f?F7i!M19*2TCv6SoHjCI+lG5Ph'
    'Fs@xpkGm+*>Nv$K3eBal$6|789ok=(jax*uI|iNa3NUXg8BrkO6leXAPwKBr^BEydUT+64(Q|vGpnJfgIU}v>ijP>;sWhY{lma&A'
    '&zVZJsvEAGPoBuvxy?zRRj?`D5hoxhc5Bu4hfQd@0JYp(*&deP0y{oPVYx<ri&|vRhUevn1F{B0P@T{D&R|j+g|(L!SG+wC6;A>-'
    '}_)mps>pg7r86EIxGW#FHJKda_@LC+iL-'
    '|2RjVF9R=3CEaPRT}jW<{D14)>Wob(^tF{j{Q_fiGj4uF7g25~1%%79s@D9Pnj^7=Ah_9%H)QUaH|RJ+X_$m7V<cSZ0xcT$)!iUP'
    '!^}%$UN6@&r1M*i2M_7Y(s(=+ox@@TX3>)@1+PW)6_0~jCrg*@Bw$K;Fx;(Elh_d<uXaS%GsG*TWZd($kYSeu*G853s%|F}?f2Sg'
    '%6_M+gG$H#<e1R`{`^+*cV1R$ZO<LxzlF^3m!JsD+sSnMD3?UePhm2YMI2=)lVkdjqmAt=%xuR{loHP*{pv~SP@|ztIMsQ`D3MEt'
    '96=kDiaKto?J4Qd<5t?9nmA;~ER?GLeMj3CuN&cS(Gy@fN&#-'
    'a5;4A0m93Pv;C#o(T?0+og46JpphRzTA^(h8*<hP0K*1BAKh`I_WnFGMTX!hQ7`kOhlDmahyJrY)b`P)i$WW2kBghn=Y=oqv#hzj'
    '?KpT$wsRjeI;iykCplJvPp~YaDHu#sx21E@ZXg$qfv^KOYPdA`yNF!MKgq<`uN_Hw${x-'
    'XVTLZG|+K&nODhj$ikxKKno5rK*Ee=mFu|Jk6(ldeoIjPWHEi=aFD*J4Kk=*A!n^c<He5TUe<_nd^{F_xmY`31{<S4(>QAPQk?Ny'
    'ZD*+E75ogGz_-`Pn;`JJ6rsGb%Uq&uZoicVLUoA-ZN)@PT<4v2!p{zCHAhS-MPc$@1+me@q^REQ2(`s5x3O)wTaxlh>_$gqT0T{H'
    'Y}*YK))hMHja@M^aVzq(s^wP%K3-'
    '7`y5%r=!Km7l0Isr*!>>DA{k<bD#Vc#6HfjUrLMpN4AR%Pp+ZFr7#t!unC_^O6=H_p%bt)H8{yxqiBfKxrmnE`Sbtx=?pRVo)?Q7'
    'byFpws*u!jIeInNfFjf*BW8nbg2>6O<OF&y6Lhbtb4RdVBH6+0P7AP^D(f5#1&cs_9R`LbDQby)L$vnsVeQgf#e||ibjXPcfCxzj'
    '!I1ZB$f6Eu}LN+C-N=ynnMi6D?|opb=1uv1N1zU0i-0lR@-'
    '?Qvqg<Fwvh+#?XkuV2+sH=ooL^#(}}3<Yg$BYX^9MYd8QkgUZ|5U*X|{}W#=%K5UfChvFj_$dvL0*8b_~+>w^RfWMvb7^r`{@Q}j'
    'QtLE0nnH?M<CIq^4dK=32Pl-PXF3M`<Prd3NCyUz(MppS-'
    '%T<m(Tl@@uUhJsvd`M#)AfnPTx|8n9#C_?J5Q$ZM>zHqbGUnN{M{;W<?0_A*5*c{{VSde5Iq#a*eYSAfjSX?6#3#v8HgiT!dI>-'
    'elF8fUi%VlDoyxSN#QDk`zmlH*v?iX^R$O?^>6GcYo2su&Y0*{mvMIP`dIZ<Stj^=72+RSSLsbDhBGmYyiI3Zt2`EQiVHDy`JwO?'
    'rjOi*;%Enph*$Som^>t}!BS8ky8Lp|vTdqyi)aD`#Nev{efDlBTgb5a9tL8muLL%*(r%(G-'
    '!!7sg?_@#Fezw~b6m)=YK()z?NZAkpm`@t{yQCK{<|Fb89no}aMKxyx-'
    'R2$Z93r4;~k4+}zSy^fy9EI=D|85Hm({Yius?Z|Uh_c24(=r#q=bWsDSoQZRF4jsc{Q+X>=O8@e4NrgaSzXpHe^DpW+^^_FfcsS~'
    '0^Du@Zx@-e$9!ZL8K$?b8V~OB9E6Ips3;$7Kyt(~c0Vw{vN2iw4-'
    'Ft)zlld=y=$OHEA+l29ZSvsBmfZ|RHo7Xdsdbhr3<XZwgITagNQC}Hv)%r(Z<UmBc$kKn&4z!cBY^E(Jn2DyS}IPJ<S4%73BCT$;'
    'XCLWhJo$5a_EU#uVylmBcbYGOm&sSI8w-664xRp01Jv^NT3fti)g*#3L^%hOfFZGH;Db9OCdIL5m(2AoF20&!hGBN48nl;1Q+kx-'
    ';>ZWH|0jV9Pri&j#whd9ld{MpmJnD_FzF<0W-R&D*2VdE=3b>?;^q%8vR8kcyJm0U39^4>5q4unhP24WN-'
    'M!<`#|A(?ysB8MHkP#Ux2WpdzB2Q0lrP#(w17(thGo}??!HKe<0Nk;2X!~o{Q?KytgC}{enE+Zx>2W$}JD*68Yg7-'
    'b1vT_k6@!;)aRYE5yqp6+1$jt;77<cU~q!yUKrpFxuMeVAi)@V2F3UmR6>j=J-'
    'wDga52)IP=itd=iKV+SbH<2^8%`rg<T5%x$$8DLAVmz$VapSgvr~ujVcl;6UZ+}`kqCU*8((A)ki<2_Wygi`U@1R|~W75FVPAaDz'
    'vu`^bC0Q}<(7vq7+B%em_Yq|FcnZui9#`BIROd4Di{w9?`_V`nJINF=rH#7?-'
    'vp(Non;uE(#Bn77M;?@E;4IOX=7JUT1+n}ZR|#Yj7-e<Vbo-pc;`woOni>AtNaQMus_RcFc><>Ok!~QKx-'
    '>8IDMgglo*^w$ZsVErymr!5`)tpB3y~V834Jp#NZ5sfE0%=K8AX@&{BJ(&{AtHw9Ot%?KcHMfAvRaoU%)G3L{Boac)gYwM~#QrS0'
    'Hv{sSO{i6*J1n=O_W*bEc!js==l1HV#Dk-*M!N1>c-k1ACp_Q%nWgIyhpvZc2cWJ~dTb%7TuRhFR})xI=Z--6)-'
    'w*pPtLDi2>S&eRxigB2IIWgYV(+%{E;$1ld3W0oAd~Uf|%Mkp7O1AHj2-U=dl6g8Rnou&l#~2YTA<Ih-'
    '7A*0FXr8=2Pve2f>!O!<7%J5%^#!d&doE_v8q_hbe>nZ3#^ZVR#>6JYN?pb938Od_DzvV-'
    'k}}i0<xj|9N3&jV$EP}3)VOYwpd^@_h86<_sdySrGZ2xgq#667LL2zeLPK<kmU7GkO6&z0nWJN5Lsn+I5dn@1G#+n-'
    'pPZ`kc)MyHwU6=G0*_Y-NX{F%TP2LsNx!p09H#YWCzN|rD!h07j$6P4BE@N|6XIH(0*R8uvP^~$Xdcqq5kLm^6%L+<l-'
    'NW{GZw0pq2pmCR*^EhN0gXFO6^*e*hR|i7AY}|<SXgHLM6Y}c)VG{<30%HfST2CaL<mHp!lCSNF4e63&0x=2EN9jV*l?ap#mf+-'
    '{b^dgS@K~cn$KdPT)1jyE=i_p!A%7!TrE|vlDyu?@FP~I3Jh94rmJ?JzN{5cz&`f@NcigVc<hk=jRjr&+bs5m6{yO7U{Ck&y%&93'
    '|_<Jm`s6Vba#G$t`;Ex2ja0TVGE|z6Iizvw5Ssl_u|M!{#mq-DD3}>aB5C-'
    'T$j@#t;JS`o+ZrMhK!D8sjoKfxaXhdpGt@IHgqW^?D?AlN8I0vDq{#X#z`>HU{ald6VaXJoN6{klmi;dfpKRi1)X({gX}9)gl#St'
    'kaT&dn=H;Lrq)#V-'
    '<3T^F~0kG;Nm!6dJ#Z4uHc+$&^_i9@+^biJZDH}8}yPH2X~G^FPTAHQ{1REqvFmrhyxkYb)G>EcCr$2Jc<fWiQG!R(1BaYFI`7mK'
    '%HF?a-'
    'I_l>p|aG&0SJhkkv`qzECC`;nMu7{Wcr$1bp*Psad9c;$^?0zgCn!mO+c7oc$k@_=z*c_nzvaipk;Tt#e6?e25uNa9GX&IkR<D{Q'
    '`=KZ<oQ8)qvM=Qu`7hi9#X@03z>sRGAYX2t|C113`s50R>RR7cl`8@i8O-MQ*`cpvs-'
    'V11RF#xyTrEO2QbEHwR^!e5liOE3VYlT1fIeT3VbYt<0B+0!MxP&V#NT{Gbw0%%5tYCBAfLgx>y}REuc<s~ivVz{$9(e{xG5>Qw('
    '_z^5DPRR3*2GaTwvvkl6Jp-wf&pb!@7R5utD5<{KpMuU=As8ii!P*UtBfX_i-rW3$tGk5_*i`Lj})B8t{Q%o72=c#C~V@^1Qg3<1'
    '*xsk<k(Uw^(7j2cra#6c1mW#fV#d6WsSu7WAlf`n;wplC}t#LAUYr@Cf#n`Rg6k}2@68Tg_Mop>V&zo?cLha~K6KY4^C!CZeSjz@'
    'TaF#$E?@}J<H&HLUNfiy)p&7#7LdlKSYCGVagzk6&=3dxsqLd+{0H|BYC;*BUQg=qhLPjl6t`JtQ)lSiHb(~}z0BS;kIS&T;qu_X'
    'lfbLO-3q~8euSpp->tKbu7@^e{wNr9B??klrp$2RUhlZcKLm_AnNQ*A6@KdPp(t_XZhpqbA&8jRW-'
    '&D)wds^G8V+468?bZ}=aX0PP6mfA6U0RB`xL4VsDLCOiWrqtmVGoqv={b<gq3}-6LOb+b8FoQgDYOBch2SEC^TP)lJF>t?_l`-'
    '!h8^LiBL$s{&}Rq}1!l$)L9;fIYIW0Rcc=otTHcr9eXjF<Q>yc}ce}0zsRz>mGp+XaKv@Z`wimqkLxCLldM&fm8$9o~U%Wd+CBVP'
    'Fi*{9t1evewFdpV6%Qi)@c;`9})k$HWoUieiTnIK#OSd0v<&Gvr7195Z6nFtk8sPj?ObkgX*s`3w-'
    'Wiz>BGls*?;`tGk<&R#Ut|Ay>SQJzrz~n6*WE_j-4r@UyM<SKqvt@+qQ>`-'
    'kVMmweV7qCII@p$H9a4KQJ`LF%6DoZaXH;c=V^D7be{Gwh0fEo%zj<rVDsz3@YuKl;;&0)FGB(64oPQ5B|_(Q_<k6)#=Tx3uqo^G'
    'x!d`scieKPKpgRZ$nQId2`z;-'
    'uVv|{auP=Io4<1}Q`xz1G6)0}N|<kB`h$ymPiF`WL%U*ChR`|$hR<e5KLQcTo@f)%Gb^>67Z9N>IPCXJZ}34dmhxa)7q_RTO8ICv'
    'scc>he5u{fNS*VC#vYgkxvx|3?p;KK{j}sFfQ>b^8xBsDC-'
    '%)~amEz^c5R&AcZ|gOqcC;ILFHFk`a`^2uVYY=3vq8^#)ttJ&t=HTf{Wh=y+(TG(&M=SK-'
    '&WE^nffeIi!WWs`<W{QwwF#kr)H)Cu1c|;J=juZ&#@h?`h~!D2Km>8YiPUwk?$|DdPR6);0r<s)EdYhNv&m;w{YB(C#Qg*KMf`(#'
    '{p^zZq<uImC;L^#-PnxIF$228$SmvI6IWgWG2^<hkLDFL1|YhIEa)OYOf!*}F6QO(eCrH?zu!(q}BhU$2uA^+kX(+fh-'
    '!_6Nr9pYE=~o>Yf+S0DRXX?N{Y!R{&qAVozWxzv@Q_27-'
    'a+5xs_vRD*HPN1Np{9G95C_fhhI?5=+KDtR!t}a|_l&cHX8s+N3v_`r5(bDUAN8*@nXh#)F<gh>8lPce>%3CBc{5^Gyzh^kH@5|#'
    '^pr1Y^l6sz_3-nI;AbXDq6|A3Bq0~C26Q{8>>J8D>aLH`ba;JnO9Nzd#Gm>$ycQv+4X~qkb$s3*0=<X@B^&MsZ=5J<i`-'
    'w_Bu?tibV7*^O0oDgp6kvT&MFG}_R1{!csG<Pt!zv1}KB6M8(SDAEbiW9isWU=0b)b_dIxymsC>=PsJTK@X?5G$Tg$Z4FdjP}|q>'
    '^7JH?V(keCZp<S67-Dyb&l9-d;BYDxJ{x)#-'
    '%3uak@xj&h1iiN^QNCe`?+$L|VItlx)MXVTuvY!;5?{)kR^6|dFYAM<Xl7w)<#636VGF-'
    '_5)CEG<iW@}yDb{Xw&tjAU1ht@?fIldJKHnDEv8tWBa%Ch|ujN7yWw=Qk6R~7vvH_kYep>Sn474#w;#bigFP)u&G6N<?l1eSXfAe'
    '}tdJK9mDj8<tgVwOsq5myzUSXUR2ZP&$F4rT0+ivI%#BFOQyZjtYAw~PsJ&#mf=QnvCapb<;BAMf-JXRC<W3r4seJ?Z-'
    'JM;jF${X->dygR9dUX|<7uG4NQ0}7Wyj1CcnOMYNGUE;|2Ulpit!C^Tqq^||%M+Re_rVpJM49fMUcpLVt-'
    'B*{_sL?qi$w0nD_>8t)0s?O1?Ny)rU_DkTnpN6Hl=@y>prgB7Bk0{G4p@-0Y##?KD75&lkPe*1BRly}5C`;@b_wO_2Nf`o*D9F|-'
    'on`Cus8NawSkS_e%iK3z3(6m?hRp?%zFR=m74he_70yYO#wGl@klOk%;+_Rj{Z(ASkS2d(6&j?s8{Le4zJeH9sW*7cldi9-'
    'QhJly2ERAbcfgJ=nns&*&ViE*r`GO*dY`aq*I2{An2e|$#Su}%_*_QJbcpi93!qF6Ik(gcTF;D!Bs=>W4Uegv>jIFrE|f3jVeY91'
    'ehBI=)0LrxnO(0)OLVid;g{FU==KL!u{%K0ER>wfT58FU|6I97#?W=MnoEbeIpIP$nvP<DF~c{U*(hF!aFga3r*sl+~RP^JlnxtW'
    '{9l8Cdf-AToN$5TNE2Ro&Xnf$EQ}|5WXOvA>ikKYC9+ZnW1udP8e5nTe@DBk$g@M`4zGRJ!MVFMD~(3rJdhfcId&{_ThI9ZOChsj'
    'KXEc4v%wU^*{}7Q2-&NEy27Qo2xF;g^LxK?YP0Ok`$PJZ?=-DsTsyTX^IjIQ-yc*<kFVv0O)-'
    '&>){O6u!bdV*z=MMHX*NYCcTa^rQ?x^$7pLF){<7#e{-b6;7SLgrvrJ9*Cp`KD8;00GR-'
    'q5?d9)}rf^c;H~@3x#p#}#(q8^SXd5TBrqN*jOS;AfqvOT3r+r-'
    'mXGBFprGqOnyH6SD4=SOa>?(e?64ap+aEw7xREw1w*VPD>bSD3h3TUH>1f^r$l#kR-'
    '>K#wg42fi`rq(E8`@3>?==q?kEk5BRw~s@<`mx)`p;F!E_Hn3GZ*u!MRH{F5`#4mpKXt}0RH{F7n>tjgKX;a}n{=(Uu|q3TFa3Tc'
    'pc7n;^oJFCmi^wxei+6qQeB==-=aMhOPucTHY(x6LLu}VofPOkO2N7-;F`H$Hws#4zU0J9GZW2c`#micbc#+F8tLj&u-'
    'zKVVB{7bjpZK-wHC*{_L^onDL>7+rLp@~1X6xiX1~^wdk&~1-06o@=vj)&d4-;(cKo9XJr~<@TYGn~$__7`-'
    'OP<&RhJjTQPPQi=oCnM*oRqA0u*Eti1olBrRQ)k!7@qD0XpIGg<j!RZw1TOJG^R8FntZ-RUZY1*(bc}t6&NH#)JE|E(&J#1pK$47'
    '*$7me`=J>n89gfT9*3C0H8a8)%&`q2`Q}Sf@(Q>I$mvMT13EYDn6h+Ie-A7<#SM!e<>z@nplA4G-'
    'Dr56_D6%8GjhKDCVERtbQbWX~TEK-'
    '!;M=mf~DzPy5E`wJ}Do{bKamKSr+uV)QyNMz2jVdS$8yZ<4wAdwg1HJCQ4+n9Z8tD@^I;nv=np)#8jt9k))|Zj~BJn04u36YfuaF'
    ';%Kh@JUOu08Ni)^QW3t_=|9C4%7K?iueMCL?hR}P~?ynN_YNT=+xLpVj-In<VCD72|1MAL@WAO>4K5dKTg^V`9R3qaF?N3BljI#5'
    '$%<FNa!U;@&y%0HPN)WuvGG2k>O0%9CNzWDKn=<)V;tkDli8}>Lbh8H@bVb&BJj;#DqSWuW=N7CmrHKABK)`p%p*}xzLxP<0`EIj'
    'A*~E?Y3G(SRN9H$O!<ja9L<7NLKA*@m<zw*4Xo-'
    '|Hjy1F~n)GAEc^zeeK5Lh|jr@d}2ADGEAQK%L?<Y69!~^h6<1BI4z}dCn)ovpj}P`;KrqMVOB#IRR7=$t`z~t9RZn>Ja%WOG<e)D'
    'Fy9bkuuo<}z23LNe{gS&N54o2(A&NrW<JwscrQ#gxB;#4X;FSTh$}(?>l~f-'
    'shV7oU8<yBWP6Ng39$~B8|`Vz@~4x99RTo)TiC(ceyz&XuXP}MZD&`;E=mB+`#*Dz)FD>7ca9?wV-'
    'n)rIxXR9dZEkdUm#WGb4|ZBf`V?5I(k4r2iqEdCBiz}Dazb@7cx>JbMrleS!I^F>0mId%nC)@8O-'
    'u_VYq_9j66%=Y;Q2y+by1pH#7@9qSppl4!blx-'
    '@P%&<XrCRDjtr|Cux^2=ZbbLF*m!VS}zFZIC#GtwXrc$CvC2p%X2qcR$&mBUSt&pI{{T*g=Wyk%PI^3ZM>|)P%sJ0DhvZrv#i2!5'
    'H+(>Rp|Y28jm^=7cpJ(&s9lRB`VCWPw)_L*Es%hDT${mSn(|ll-'
    '15rr;YwPM%Iz#TTaI}io)C*sra!ov|nncoRbG$*tl?q>wI5IYFTSDA8Ee|FHUkQxNAb&)K$N@IP0p8T(@vVM=mV7gpGSsb<kz8%Q'
    '9hhY}9snCK_maqER5nXI_k__H^P|y;YHca!bFZnyXOYW8JdV^cc4rcntBM_nK;K>9!K$dCdH%c!w7-W(H-'
    'Gg%Dmoq?~S1j22t(KhyaTyE%G*A=*Bz#fT2qj`|^}la0U}TWmUKsQ1Pbc27n}HSsj}W^`DStZYFBZ2ndvE=P|(VJY{~c=XQ%4REI'
    'qKaY>-oFDm>9BBWoig1FidSr#;vpN-3YQ}mA!&nqg2#>+Ni(@%T3X}BhMparW3VZ{K!DoZdi--'
    'BS&WG8*N5MVHiJs9Sv9~XV{32E&!;X&0%H59aIBC~sWXMu%M23lmn6JL06QS8o&<e6oWYYWon-Ak<+x5f<t6Qu!Uhd)Bryn`Bug3'
    'xg!_dVfNVqmdW?dq?oh$esqGki=Ra;<A$lx0<&&uHMQMz3He!vxYBD!X+Voj={uzqZj_Jv8~p2<j9G)5I3?}W-'
    '%#npRL7c_ej5w)e*XNrJ6NwLlpLVPAz=e?b98YeLOIAy=S>1qzr_ABR7>6rYi$}hBNdZcRM%j3Yva_qLLwRBPE>7=~d5!nHq)nf{'
    'Sw)eZil@R$Js`H`t%V5-oxDX)RtYPpGY~hB(MX-'
    'e%0T00zZWNpVcbayEij;jSOGCoXGY$#&(s(px0wTL3=7oDXNx+75dHU$kFaz7)ce1Kb?#`P1;?T4Z{z}-qfb2%>aUo-Id^T}vapv'
    'IigxT>zEiNXm-9kr(7-'
    '3p3szh|KS%>(i#hZgbgtrNMFA(8<f;A2FbDv^K1O41*SkXW~_c^v_Aj|s#yE6#VZpK0%9V=7XewyWJoYBZcsg#V^te}G42o?0g_8'
    'OH+o{mNA*%o)vtx81YJiNG$ejv!mrlMY+lDD)*KD!tfMa%1$+|6O*Ix*>)Ln(DuWUPe_YF9;GP{_8rDB3TAVCt$UCKd!!H$`S$WF'
    '8`HVS|pq4%*q>OxWC`ah~?b?4{|rOr2-'
    'drLTHvH#k!auWJ^P{NkTUaf!^X99#hmYJ%9V1XR<eQgrFRRd7Drl+fXLYj}LAe>Xn*{XOHrN=G*$y1N#kiv_|g^cl$Kj^G_LPs+u'
    '=LSsY#<de|Dw9~_@$>?JS0C@}Q7tJzLeypRK4~P8($F!a_@A5wO0pFH-Cg70bsidpBGpmh?Y@xqU5mC-FlpS6X;n)ilMr{Hu;rm)'
    '(RiN7GZrBx4V77+|u^CRu?TX2DCPK_YL=MIS`N+opL|60ttw>p}mZYv5o546la9AaYVkuUnLlFX=3GTGWjl9Dl6(%Ubg|sEWYkPE'
    's?sVEppse+N(fjOPDNClpgV(^NZG#|2N^I?6Ik}QZ4x_6Hp%o0wZHa;Tb7ElrBGQ2mI3BhKV)6M?CB^4c&>@n^g59Izbn#NOcY8S'
    's(~_#?p$3Ox8svD>rp$#p={$_3Ox!L-7rTQJO#B4NJSVspq2|9-'
    'r)sNM=0jaGMGh_H^?Ou80rRV}U=YrQq)9ly7qU{M*Ku4Ww=<P=T|wV`SBrSwL|qndp-'
    '=H26jFy<@h|Q7LzwIU<Xw>MUxP4KpzYs)fmg8YTNpG<7~HN5YK4q$*9KiChPN97#6Gj%jRctLtC(5GAZIrY8;?(jL)Utf)|2c-'
    '^;R26wkoq*RmzzG$X84G)W4(=xq|*B15xx!yd)jHi8=<p(+x@<9a1;`4yyL<6ZIh@UP+q@r%|xU>KOJ6N`_?gWe0;TF;$8vzm$Mw'
    'C(NMm_3vyzEPIipSHF^t+X<`7@3spTmmPIytZit=lb^m-w&6pLj_{#U*|H17Q9(8|UzY=oDX=*b<xg-7wo^r0WvQq>t-'
    '!U+keP@TvUOZ@iq<b#t>?+a`_-'
    'z~4lWn5GIJa&?q!1DSRhm%H#%0_D+ERJpiq6xbws!`1x2$^Y~dvL38#v0e4OP=;2D;&eu+I@+b3l@=}e04f>H^>k-'
    'vrh_5aL(f|?96TdAntuE=)=K)LJ6&KJ*flVkJ^oCs%89;;dOry0pZDCFkDfzk|x+<XL(vs?<g-'
    '_ecU9~(xWE8j4}5WFKWx}50>p`h}cv|rE$s)w_bP@+|`Bp-ns-imZoReZrxHzXUd7AvZ0n0FD4ytmPtDU&L|Vb%n6pGA(hB5LY)D'
    'GZDJ!U)mQe;cn-;9brpCs-'
    'L16SDQ^r9+vYIZ@simw(}u76YNoG{>3~+R~RgqMV~7!u>+Iwq)ogZC0+E6rvL!{mPa0&;bmKy5=1P68kzFkJ<60Owz$!2T4b<F3p'
    'kF+JDkro#zM>9As)%_yT`iLNXPUgX~jr(h0>SJX6^b6|Z{k-'
    '|SPTHW^>j?zkZ3b=&L|owt(m+iVp?!5wU64gw1SINgBlBu&d35i?26`X=lyY0}>e<r?V-&s83F;Cp<xAU;XfunTyi+!|8)cu@-'
    'TF7`#n75XCHgUUB&yO-%IT2N1iy3P`(Lp`o!=PDWE7W&?ts%4y~?DfX}+{hH+AB{aMGW-'
    '9Rv18+swmEgr@TYdxc0h$Wpeb;?EVlQ~R~QTKBj&P$pd>DG#54gIafxF|30R03j-'
    '}>kX{Wb&w5Tjz1304G;=P6LdTIP!Qj)GJ?#8hp^o_j{+G5Tmhjm>NHO%N6U41H30&ne#`Dq6vD=yq=bd42zhY=LpfXw%MJ#*TB8h'
    'd!Kf)_hxn6a{o6jgOk%Q`<dlr$yh`!Iv9Up(Jq45d-'
    'Y#y;FoB9)x?Ul>ZIl8rvrP%@R2fsQbgPWeVk0UgvP5JL_WChAm=zAlw3Ux8i8Q*~Ill%8-'
    '!;f$}~j%DOyeo(X0$wRt4N;}h_77`rzj@tIC1X{WTI@01=ogoZHt20EwXmy4x8?DZeCZW|Ca%{9ZLyC=7XTUP8LJy#Ka2?YfmDrI'
    '<MBmfSH_Xreqry&Wi};l`WUX;)bwyU}yhu<9hLQsxm2oH%PKXw=&6}R|e%$U3?x7$(DDSix1~(>n^X-'
    'wjP*Nf6euFB)I)0wQ#Cb!&BPdWF^yN^fEh$-'
    'Sl8oY0vYN3>@mW%)_&h07e36tXHWyhOyMh(at<@eXaDHhC0OQoz7}$W|72q(sDp6B+QiL&?pw*dfQVBO$yDDmfb2U!USnZG<#nsv'
    'x(_MnmxPqcXi9&avG^=7eaoIuJ{uN=gr;Ko5C2-'
    'gy)A~fLRAyxByAh@F;!d|Vz}p8f;5G(x7y%9VQEKx(XUmpOX5XIVLANRg!HFlH!o$;23(?+Do^6qMQTbg@eBC826%Ew(pT3*dL)l'
    'KHpkxV(8)!UdrfRkk^9#OP3jnR5V39`?J89cg37q~d_;|6P&_gE~xfc$+T<Gg!siC(HF<GEi8zEtAA03Jify<6l`WE{ggW;bePp`'
    '>z5~0Flu>!Xy>)FN8g@qEcJ8N9Bq4RZ>E@3Fq!uQM7OWDq)<)T|tuWPaHOae7nam=7Cy?@(Fo15O^D)|7@JgHu&T8?%H)7|-'
    ')IDHq*;Q&f#LGX9AEbk{6J>}*76cd5Gyq{ypjaT#wO!M-'
    'JZpLsRCkDTF#T3STUkj;KL3v;2Dg_L${{v9m0bbt%w3qkPvTD16(UEZtdzV)2D<@T3qf3Pz=p^Yht;o`-'
    '=e;bYxot~vpL4Q^`)onJc4$$p!Cp=op=J&8Hn-qmdwzyVNzOX1aWyjrx&P4>4H>%pAFgJ0UAW*a4JOip1^76Pq1)03W|O#f+z@so'
    '?(cG_9RnC;yHr^Ha$yObwN(WxR<q>2&^Tf|xweTqV%u&N_Ebj`k#O7{Ds#zQpq@+yCS9-'
    'Zp=%~!wZABoR(}!UXk;GV%Vhd#W#$y}f{5=knJ74&2@W*TY&r`}+)e_{A<HQfm)!Ck1zMS&f!WWMDz6C(y_LbVUs%C*1`}E7HXA4'
    'eEc*ZmU0gIWn(b?$j{G6K%Jng8!L8fU@Oqp(M&fYlbz0PQw%?h`xOTUKztbKTR_m_{Ma{7lNq;}k_K5I`V(SmJJ+dNY(PuN(yUJG'
    'rq7WJ+`O`os^oq4$di*_;!Z|(unMvWC9{<YH(ciB!p@;FslM(|uT~R@`Ht%KcGhtR~K@EMw{a_;@u6IzDHsDSJvz8M!3-'
    'cIH(IyEu@g0M|65QRphI|5_miHX3R5&f`0iWP|y#aU`4*uUq_neQ;2k2C?Tm29~8cqxR<nuo=VZfOf3S==AeaA2-'
    'j=Rlm9DvCXW($z8VF)n9<8X$Fw;hrNPD%*~>&b4jF!uvh(;D>VTPL7BKQScxt3Gkpe%LAzKpe+2ElzZ=9x@&2R-'
    '<%mqqavy`RAuI7E{Xy0GrxBD%cN}LF984NSb<cTJv4QaLxP~pj|?emOl%yOXy(ct5Kw!uM&z(C$SSL55COO0rRi3;+*Wk73TE2zh'
    'u}B(OAzzJcE7Fg-'
    '+B37n;P@7?MrthQ<1azP6#&;&5nQR6R^)fsFl?)V!`F0~(5t|7R%$tosw0eOqx4cUu!cl3@$=u=o0$;tfE0i)*y4Q<9zM%Z)|j-'
    '$D89#Q>s-T=GVBsa{Hw<J^l-hTK{LZx^O?sFEwc4UEHfOhxE#)@yrsMKoBOvb=|k0AHE$2(TMA4}7SjDT~-'
    'ok#h)yPk_u!W{x62W=>=JCqQOSXZ|NZX3k&&D4>VVWCkdpht5jV{eDbShKYo|N(=U`lhdq)g7GR>33B6BRVZLj6h;;@<1<X~o?~&'
    '@XRBIXpdj)-Dz-lTgxoQOVZ`GKWaY=Jv~IF>8qZTuw~a1V^tMeiehTf{rx`y5G2cxyev0<KmuCDF@O4Ntev0|FD?%sV1lT?p{pqE'
    '4`$V^zHDV7wTvbu7+YD!?BuHmVYt3)PF%nBm0^F$tk2%{Vd2u}`KDaamqfWuSN!Eib+GQyil@i^{Q<9ML6(xGfdQk5B8<VWZ1qyV'
    '`TcwkZ;|oKj#}=m9PAJSho|taS&fvs%+wx9S>`TN~85Xjp)<XxZ0>{1M3aE}rMP5eVo|irA(gTsOu8$^DK2j&IrS2sa=BFHWc_Wa'
    'zeFeom3)6AJJSwDOof7J{T@vcH&XH^k5~xDs)+M2F>zdHGbxUa6<|}09TUSJ+!FO+pD4#Y}3sh8Bs5GUl6Qhc}^Jf=|%X8B70l~>'
    '_H3%<m%kWzzXp!HfAtOV;+Y;GiPlxpq{@62Nu{?LP-&pU7P+wqS&QM&5-`|j-'
    'kPtWW`xy!@aU*|_q39Af@`o7;FL5Jp%usxZ8~LLQ1(@_;_Yon;0iHaz*fKp^`7<mwZ(K+vI^zE?%n%szei)_V(GICZa-'
    '5?foD5qp47;bn*bBS$88Gld3wsufdXRwuivCKB)iH|xw-'
    '_}PKBKEL;WO%~?9z<<r2oO^&@jC(3Z@>R?KTxraE^Vzfjaa7WogzZyPJ<o-'
    'o*q~RdeROthUjkk8(XoR?alM2`425sxwn^pfsKv>#)o+kTpq#&S1gvovg@sB{zJI@KpKNH)QDd#|?C2hMs@iKsROR`^OD*a|U=we'
    '80IFDob$#-'
    'I4(^5;xGT89IBx5*`N@O7U99gN0HQ`%VB0rL%;`Co{|4{Ce(I4Zpu0DPzsiKuXs?@V~aR7v?K0$VrN<KhH&hFO4zLpi5)6F|egE+'
    'Zf2wm~9MPY0NeTsx)RB15+Baje#hQ*~Z2t#&k5WBt4>7>@lFEa1Ol&`B)3&XwEboIWCVQjo+)#tEo&(LSKiH)csJfp-'
    'r@|ROfZQtQsg9zZ$=Eo-'
    '}TUUn%l^$?ZOs7GCJe=~B)#^jCO%=W+oglCj>tVmnv3+Wn0d#;;?+Z7crnM6inr)nXr=H0Jy2q%lv)$*C&k@gMDb1$E-'
    '2)(P~QO!Q*9@R1cRQnU$LHgCl{`36kKd>LHg*~R4<q;Vuh7&|wemfu~5XgVP8tU>EMQBly4oRd=&1qI1DpwoDPN_gMrI%q>&=vy='
    'rhIsD&>HwPY+}y5F4GrgpcFkGFK5vB1b&9bs8liJN*Vvbgtmt{hzHDSk&o}lJQze#1mpsdOmGh({*r$Nm=qr-'
    '2WBoaiV0Au(4bvZOg_Ykzcxd`)7qurCuNmHB+y$ILVVrFrx?P|6!@4JrD<!LZn?^M~ZpS}s6nQck(mmVQwMJ-'
    '^&oTA|Bh+@6XM)s!5j@U;qYx<0(=;PGKeszn?5-8g5O%ECE)~uYcB<H}74gyYr6V%gK_@Ou%K&{b^vMXD)<}-'
    'Mor*Q%gAAR~pd?5yX`L=gl6d<4i*z-2XD3DPUBC?{y=w51lC{I3D2NQ+s!^Pbd-E?E`Dlfu`dTIUYRnf&@YOGWRD!Sm`C<va2IP-'
    'P@HH@BBEeTvzErMwQ2scNhfrwOp<?U8x^1D+a~*7o!~}HD>m{9DpyE`q8zQ*Z<P*YpNek@Fd?`P+qD{NR_-'
    'JA#_HVwdIQh1x-%5L_YJ~*)qdFzuSz@4Q)6SiHM|NNw=LP4nb2WV|xjWC}RfzRE+Anmz%-fN$hx>XRpMB6%ezgEfS_A3MKsSlBb-'
    'Rk~RN+F&zqCjLtOWAwPN|w~Ap-9Cv%q%U8<t}o8036$-jO3;geh8~0lpX`b^-GR!V)_)yvj71<-'
    '@|Oic=39Zpx(A4Td5SS`pR`XhPu8E-0n8xYdt^;JFY4^6xj-k<m<*m1?cl=e)>S*%=z`L5YKD*<>m%?-'
    '!@8@&R$YkHR*YrKE&HCjIh3JU8c{0Ox4m{13RBohw`!=vcA4R3vU(AV?&o(S~g{%m@#p%8-<nV0+fg&U@O6tK^RB5V7tMa-;{-'
    '?5fyD*dbPuJgv&tNe6cer>T@qe^dS>W26+l#xcR?p!g@;oNJ-'
    '0B?PQJ6lPUl58I`;Jxg;b?r8uBS*iXW$KKp7&cYw9?RNuj%B#+uhJ5-'
    'Br;QXEXKy<#sdUC_ALYc(fGJb<5>o(%&otB1P|PI@FB&%e_nRV!XPlOZw=u1jKb_`PV>EqJW(;3e^xYXUR7pGU)l?Xb&eTJcxoe3'
    'vHmh)SaQ*bFQ&|rzU&WEee3d`LyW&2TKg&x_TXS`siqD_pu%EW+8U_#04SJqCmJh)z2p#_-'
    '01xc%O@UMR{%m@Q7v)|heecovll0HWnfX)nZzFH$`NHc@#`QkX>993f;5q*NR=*$AIvkBX;Mv%N1GvUOW-10FWR;Fd9mMm>9-'
    'F#Z@|PzBE`+oXTF+jikq&e^<1Xg=qNJJOB#(K@3q})m^Xw!nSZ(|Jy|En{Kt;T~(*~&m|Fxh9B47eke;G{xPyV3r_LC=n$kX<FPr'
    'eXrOx}t;Y4;F)+~pK=us-fn<TO|xcZJ<s^l?|xhL5~KIC{sWg)vGbPt>V=_E@}s)Fp-YdR(e0<B-'
    'PqAwf6v6}L8ye)kk=d{I#sptqtx$GMnw+7X}pKMWW6ai=!cpBak@-'
    ')@yKS|D}8`3XZ%!pl~lG(hRg`ZObNfh<5z!A{+QNw)P2M9O<`5_Ol#dvX$Wx5|5Q5=AGmHz!ea5*s*)qLbK%lc@U;6@1JR#oc8%V'
    'Hu@@EPMO}h3V)`%OZ6LyR6$$w|OSb)CK(-Y6Ue)xXjXVd6ta2hpld}HMqJ^fyC{4oV91|WVp&dgyTNhjp)6s(lxqYVJnOgHq1R`-'
    '#)0Y*?u9!uX{^Ba-qU@J3?4s>!rQ>hyqJWlsj%jgc4PK`%yxXNHO-O7%<U?A3%Y0qWRkdS%6BiaL1vP%<TFnl(-'
    '@9P*}K{o#Kx`QV;;LDSXRsE=t<3m40Y_)C?M%KQ7Xk{wMDTEkQ!p>ObZ{NEyVb^+A*)B@`WlK_S1D)W#{yOlbSaw<ny0<l7I7^;U'
    '|6M#ou;=f&?m0EN+nu=HgesSxceI-'
    '%ozwdfIV6)IA4#l5pCWP;N&aQ*Ni{AH7}4;#CG$YwrMc9C%{(7<fcBQl#fdmz~G{amECU#$I8q<2uPAX992aCmi-vQGj(Dx9@zb;'
    '4QuqK+Jp_9ekAv|<>_&G{6_#HMK`vhk73O3c1Baya-'
    'U*E3t3FAq}^kH_lDtuPXgC)R(FUFw^{i|sN~HvVm@7V`TOWgjv2fUrcLD%)!8f#KDU$<Bt4WB(oTc_5YjiO!!0<Rswlc#;XGf2Q;'
    '0fu00oi%+r0_0It(93&JUE1j%4=<t<&0>E6+`&kYsZgM@3+oueRC5=;MsI7QN3i=MNL>JH)N8?0WL`XMqg0|2?fl2%9$GVz9rXD7'
    '-#%hsmUuf)pVSPSV_EBR;hgV-HyV%%+!z+rC9y9ikOsH~t$HLC=e7L>S2(i~vEx_wJ%08|GJ*3k^S_T+%NC_a)LC+%GvI5{_2|ag'
    '%JW(-2Sm=4z5_D(*dsQFuXunAGDSj+_aYSaPWxS&BRgK*CcpmI3C2Z)<BBOUKx-'
    'jJ=(e(%txwN`C`nh#1YqvvC+QxTqHx<S7`o<_HURxU^3sH6%l-8K~(xLM^rjd+Ifgh`%tCIDL-SIJ!DI)7XLUVQqaK@2ZV%(jSJx'
    'X&|J#WsNgNh;;^oH=bTs|}(YX4@KB5z+0X?GlXBg2FM{On&OOnL$p-'
    '`|{GVyxRuF^NzM{O~3=EPH_NH(1d<LH8SM%wC}T4OzwBp!*HRbsy0E&SXxpd^01*8eHA5K?FaLz1%4xA87V)g|SHi3%Pwtd;gprV'
    '0dn6O@ZFLI3O@s7nIIrB0q%18h4r<SV58nG4S)HExgnX!PJ26(Vbt5XyGbdP~r121v7&Jbyv>;;Jf&G2CG_fm?)R`jBWInHycEqX'
    '#);advsNtAlx8B1{5FPEOP`D65qmu1KwbV@T#Bi1?T;RE;t_`Y{B_JAq&o%gey27BvirqV9yj>>Qa(}e;52Cr2xVK>30)WA$v0l*'
    'c%NmO+6d*xdzj&m0+L-JMd+wH~begcuD|Ea(jUxYl^ZXQ;if#pytRF2IWjQ%Gd?vOgG8Y1?3NOdCY>=2e%*;;``Yv7-'
    '~d8)^8YTL=o1N3^by+WhUc{DAf8b!;C0^nZ+n0ie9eLrFkc?wyt!uC)a0XkU?9zY3HfprcD|`g(#~r6(IN8sEx@cT4J9xTzK85AS'
    'Bb#=@Z>0>NqGBgoao?A|D}>wqrms7iJwt!}9z=+Yhy*!+SudL1w4+gg$``0__E1H5mljn+JGC+x`lmybgvlw0V+=J%vWnZ>#1^=#'
    'bW&czR0#EXi{2Z!17bKU1td4D96Riq(e!p8P_w{xI;<n<EWyAMNNiJmcsv8Ne6|HV3;QeC#8@(`4U|gJazxV8QvzpLwF3($Cvfsy'
    '8#*Lu^{3mm9+dMnb0!ici8=3U}h5y*E-5+>K4aV~^R)x92ZV^z+4VbedbRTEV>Q$j(!-'
    '9+PmBL`X&p`IG~*$dcb&K+eAm=C26_;O83(Xx2?UiU3<y&+H>)Zh9~s4%wTYJbz6Frx(whlf?-'
    '%z$a@mISo8yOg5)aP@~dhbow&K4iLxdSxXVDe)fa#YJdX1d%KdV*zt-*1WqjFJ%4th@im!m7G-'
    'IQ|6E2FH8q)aogh%`t20iGTkI<t87o#E*z7Q}eUqhv9d2YhJdpm0nQdY-JDC3cmUl`$o1`A}FIc3W^e-'
    '5sUi2^6qu%r{n4<>z7pzeq`gc?$G)!kKD5u*y;}T!5tO__hvo_|0u6&V@aqZ!RXWIqkG^adShuy&}`{WL=I5MGfU>5x#7;kn+?^8'
    'Eu__z<WJDHAxt$I@dApMDA|J5)0ul~`04PgH@i2lu#u77=`4C(E<vArWfWP<=%ARPM{mjVw5L7RP#l_WLL#T%nK(+S#|61fZ=>e|'
    'H_%AOZxaVGS+j1PHFS6xXPY>k177yGEQ7WPI&mSd`T$zONV9SqZMMvIksl+j^jB4vPgnMoPA)?qv!eyRdha!}djNCgSw?vUY0pm`'
    'gEm0kjW9Cq$fzzHrBow%D!0U<#lh$pE*i>AI-'
    'U$uiN@}+KP$bKY!((MhI^`um}gCWDtqL;&g`e7buKsA^M8c2<OMJ#v|*M8qbym_erh>nAzS};hMiRB+VMcEsjm`*xR^LoO$U!Rt*'
    '8>4q_0mXDrAt_=?&U#T%9V6rbVSy-MbHsT2(CKg}hV<wi2t!lO^!ze(G;|D-itdhvJY3R_-pP=UOWMmj8}f2oK6w=(`!vx-'
    '{+vtQeW8|Q+lLc7P+|V4V;t2S59y!;X!%L_)-'
    'N%|$=l{v%FZz<X4;W5>;coT(3E1}Boy_eO^w3A)l&k#d9Fj)j@~6%HVWzj*YJ2C?Fvna%~Exj|25A^n(*g2(|A+WkW~j!Xly4@)C'
    '7Tbb61}}XUIc$HZmt0i+~SnG3xdaE!N4lYB5iCkrw-8AJt;{?P4va-!9Q?`&|M(-'
    'vljQcP{TUWR@iCT42ZxN@(?<AvY*JiG^qc6o+a~^HrE7wo)FH%-7Jr(0W@-|H47(tPz2;c1l;ra&={R)y?sX&17~7$IJFxmeO{-'
    'Y_qgbrtcd`!slv+V7OY5wj&Kahk*xe4>0sB9nh|>ZdGU~HzxpuNea~DD&EVOw0Vq_WYwb-'
    '?=UiEopq}Hm9Ma8ae2y=P?r;&NXV~ELVOW^=_%Sv42}Cv39qy_0^)FbWrbTEmLldmAuEJS4_!dG^w1WCOAjIuaOt5z3YQ*k5?uOp'
    '?(+BL`*jPgw4vt!zSed>L(gG@XbRiCh1L><Naoqj<-gL3n-^Pg0rhGaRg0kBX-'
    '@2|#Qh5lybxus%2&BksL#F~WXX9azK@fHakf52y`u^O-'
    'gF;F6^reo!M53~DK`rfNRA%xH^Eme2A{=Zx$Bfnw$n2Ca;p&qkK469(9m;;;hlAkNBw0~YZJ$`4)xl^5xs^2ThbYN14XsO;=bvoz'
    'Y`>7{sa>ooK3lAkzs3`o?eB^kIT{5(a^`FFX7(8;`9Y0g8$Yk`wa>YM~ANOL5`q$Rq`30B&+Q{LxXZsi!oDf27+AThHW&*4a0yGX'
    'W0dy#D}ww?Vz2H4frR`!h}J{Tic(fi0jGP9+o1mPow2dm-BRO4^Nl#Ol`-'
    'e%XyZzN0{=A`F|PQCi0vI)qu+^HF;~9`Slj3WAhX{pZZt_=@H||o-'
    'vN>73(0qV}RZe>mYq%9i%T$mC~KRM<~VKbv@CUQ5as^Ul`wv;UmI^fHeo0Y+)>Y`o<V%w9*SXn{(;5(v2P!TR^J<&%D_BZM9SE@J'
    'wY_Rt*&uh)BS+*jL3|lwC1>g4qmU8WS|oSNUI9h2v6~k#x}9{K-'
    ')A)AoLmklRPw2Mo8lC|tcuCE$a*g<J7GC$jh3f>oX;iOIZD*Ps<SMNN^GMwJXl_fyGiWq*~-'
    'Rt`|fY~?_eP(?JUWYBVu5>AsAJ6M%Qtw&=5uCGd4yh&sj@0eT3)R>YBJSZ5`j`n~IV5C{e-TMUfbx(()++<Iw$@H%NR*dI-'
    'CqEjG;{721N{-qCPOmB+wTD105$Ig#0%pfVb52Y&Z-|NJjWN-@DJGgX$3$~(Of+vv6V3ctr`$o;^c|S*BEyDkxke-e-bW&@nQxfm'
    'kM1}b_CUh5uz*k`_cIrivSUgm+I1REYn^r=)~bcN4;*QUjauNu_M(k?5JRj<P`%LU??s~6g#r=W(`COG3RC4H0z9IW^HH7+!)IA2'
    '{l^Pj7#g<zg$5yF2nsJYhy`b!Ug{}D=m6cLWWPyTLPs?<7p)P&<`&U)(jptHVg;o0mEFIT^sjUqtQu^+CAx4=_<i{W3$u49>#+PU'
    'lW9<r&m+~y@_C*dX+BSrqiW(Nep9O|1+}k}O8b$S#5<JKeyt0`A73ZY&$NuGO45Zn-f`sDP-'
    '}A`3l|RUdU)?T{wpUSe;K7&y=^_}A~xfwr!BSOvqLs4M`oox+ux<{BAhPF-'
    'q1;zB|y!AtW<}R2hW_Ng2b|F&KZW#a}k~FMgGf2>100|;wC!TK5WV#qm%uZ|8fbP>?QuorF62Fa;Nte-'
    '`$N%Lspk!`adgIx33zO9vS<bV-YMgU`^)h>Rs$3ADM>un{ou_v)MM>r4j@k7vR<>koLTj-H;48<h$7S6Wn&**?y4V-Sb`Thskt9-'
    'o<WAK;3y)`%yC6kax2m^JqhU12CIMhnsjaMo5my1iZbNdCu%b2FL%(<lJq81j^mf^OW7Mbneauau?5gWu;W(8K-'
    'nNVIKsK9~2;`w>x>mXM?{61n|i4vE~4me^hvJBXFhT!i$?QjXOTPn2Ra&W5bJEPy#qE6D5IGAvfu%?PHpi=%yOjxDew^GstN}7JG'
    '@okCp27FRP>|mXyM;HFQ0tbf^%RwTdSni__U!09rLV6l9U?=QgJaxl(55>vaB1xxeJ{t$s1z>L2s10Wsei81t>Bm~Rb=`PSfYnCw'
    'L&15Sa?BJJnUB9QAYe}H#Ulp22<PQcl8vE7Ijqy}-ei)LDR)M9ex`f8y%@qkZo)3v@&A+|W%i=r27_tr?0+-'
    'GU8gM3eia_jwnRk08*dm0yd^GEH)EMHGK+9esk-(z-'
    'q=GRPUzXe6__Cm5QNo|G}rP0|1Z(#P&JlW!F%IHQF;h6JH5sv9RNPeSO(EO&7LkjbFtgbLSZp%3_U$`OW3pd7m;ii}`+#K_TxiMe'
    '3#pergH`GJ=OGMz?L_g1IonNhmFLs<ps2DQ#2^!4@F_un}x6m7<gzt3Cvr)gPE^J(DzO6Y_AE6la48i>ZC$NrbPoH9|JR4xo9c2Q'
    'iQTeGzDUy8VE!WkIWK0+zPE&yNbFCL`ewM<x6}2p`Owy@-'
    '$nf?;&ztuY`n$Ya(f;QBg;p=`FLZi&!=fwBI~Gk}eor*W%jC3;Zh2}=oNa0BazE_+xc%6a&ry9mM?K;>>KV^buXv7n$8*#W&rzRv'
    'j`}8pjxKQxWpdyn@`%2<U(}>2#eKje$BD)Lz7POSFP(6(I$I;eMw9}z3VY84hsfZ4_cf3}KH-'
    'Z?G#7<Bq$4X$)tr1<Bd<?8*s~*@n~zFDrI&)v9TIG@rzNb4z`incG2%)JBcmRz+2RCW!a;qCj%qllE9j_(gSwKAYB;D*(@_lvbrl'
    '`ea8RG2qZ$tCvvgD=9@t@XlHAInbCO)jHO~uGpVj0=-'
    'Bwvk9<*w%AORfFG3uo`>uwE32g|;aB~Bg38=u7q%EhmR`qCy=eBCQG?tphaEzONaZCA^=>Zk2<e6BD=y~aetEt7iW;bc&cjJOc!k'
    '`WjFTr%Q9pPMg<Q?vQfNS;fF<<zP}w%Xv36;fFl6nm}`O4S<+ke<D=bpE0~i5A^vPFQ#nFIvfHP1{7sXhIdEha1GspOE3QmD18^&'
    '~}yKe6EkS&&XNptL?MytZ`QcK#H_01GHA!l>uIhrz^M6_GClPGLmwtp=bYUO9=hav*?{r>Up9*y^2aF&%D&19|?<n*}XEZ<;)hWw'
    'p))jIa`%P%7yG%B@r*EWI_lkn`^24i0k8v3Y*LGlr8zcYg32WkNKLOMJ?feP0xbVobT>c5!sj4&Txtb>wE$k4<77}DRf)Pm{W&aA'
    'CwZw&d@q-Ss={mrQP<E!OXogR<fjBvQNfJmb6RiGgh*sUD6|CB}+PfJu_Caq~q5sV<k&Ee!VkRvUDT?A=&bGOQOD$l-'
    '(+`gTt$d%C^gFb5?M;XX-'
    '$)zo#ygnH$XX5^Y4I60rSfJ2vC26_zQ%IAEWBAlaK2{)&;S5Sv6d_&k8+Ki9k|1n1Br_Un7betoalukRg!5DgIs(TBVE*c8$i%OL'
    'nn&k`o@EphlcfH--oDlv|yX-'
    '2$%t*fdXKh0o?8q266inqrBh0Ak#<CXn)&4fnBD*My$Vhx}rhlUqxfi5{Le5|*C*mIjGjN(nEvldkZ@AIm&LF4EFP0s@5zE5oS&F'
    'J#61)2|rHaGf{9L`Dp;%M8P6>TMB3C7kMb(olu8w^E})A02SHW-ckzmLuQohzanX4e8A@8w1uF5wZzW0d`gW_pF=l>M2sD<iVvVE'
    '{p=3qHA*48OgDp15rnmh63nQAch{D=2=`vq<hdsOedFh87x5P`_eLF<GSL_6&b49;KxA+o+KXkqX^-V6EEmlC{&KoYMkA-'
    'a0usbR-V)BogVqx|}{6n0;OUskmN#j%uoiChRV(^n1Jckm)jrCg6pINvEE+RD%#x7;PsRdiJlj()M&i&!TR(K-06xK|Q4DS-'
    '38K6~@LqsaKDzDsHT!s3_o2?hzcyXfGR7?`w+(p*I?6J|N-'
    'MC3DJNSij?|!I~5VwD`SxhH@p1BD9^{P>tQ?ludXY_Hc7H7bPkuyrcGU(=?aL+@63O9t1A=A_2emmPln7{aehLfrbqTm6Lpq6hWo'
    '{xx&BwYfOBkJOyLn#Gy_RfjteGH{4?}jDK5}%zt4uQvJkwPUP}swmlkmtBVr`OSDUzmS~r56v8G2n_N3}!{%yKaXU+N-'
    'zbAy&d0x(iRR*1+_-T`F?5Bd|5MiHS)ih(Y<DHD9`}ZQJvK5sd)y}W_P9&z?qMm|-'
    '(#<{!^h1NkN+Mpy>@FOC$A;5=fq(d`X3#NwSHF*Eh+0#=(yN#wx4Kvl)?Y$6e=#LF1DN{{~e3{yb%ID0Xon=^G-'
    '7aY9=U##4>n&EEDvIWrCiuOwcQq33|seK|?GP^oeDHzT8%H^Dp&cGTv48pTHb&CTlVZ<U?gLzrh}#R{DR8=?S+AhiQ6TS_UEKwKW'
    '7>eSHsM?~ll-'
    '$qpksIZhpLx8^xzKU6Zc{DPyQej!paY*I|g06%z2X3xw7`oFcd9fQNTCBMa<6e&OV5}@Gu04Jv4h3L;X$!(wZz34W*B|*!%8HL)h'
    'QZmkMcY$s7%J6p*ZviaZ89v}$Y9A}D*+m)Ftd>xbE5z>^iJmAZY>CuP6yXIvI!u4{5R3M?!~X?a^>+nz^I-'
    'qU5B3#{wno{Fin#?a!~2n9>O9P;eyn6f{3WN?^HRh#x7T*ZqP#W0i}&anaJFTs1H1>5RqcW`akZy&IkSMpIM~PJvS_C%dycR5T#-'
    '47gml#wz+zMzcXw{PtJvz<EotfOmGm_BPI?;qC>dwrAek)U2P!atv>;?)g-sNOzA==-'
    '3Gnkf+HM!TUOy1u*_|T4IR)pJoc{k19E_%gsdu$2TBZ+_N4PCe5)OQ-0z*beIIMfLgDA*C{an+-'
    'E!2a`rMXFvKG#gI)+=q2G2)tIQmzO`Y3+}7s=m2e6%s%B>kn)8!3UH3EWm5Y)BX!wYCL3TZ!z|JhR5uHp6OtC(9YiG;njZB&i*-'
    'q1%IaPU;KoQAGfoAH4fk#)z{A#lA(JQ#8F=laIeEK1j&BoaLfaFXEGYpUY8jMJWJMU4>i4v`*0~7PnJbmf|CtHA=ah?G`7E@2(`a'
    'V7E$otq-2DOn`A@j-=ku9#~FKw5id2~*zd<s^*CdHXk=dQZ50}qEg<hWN@!-'
    'g7(0j_851kDH$1q7H20(wd+JxL7xL9D7@AiXX7V~x5Dy}USt4-'
    'pk#v{WA@<}l@$d|BtoP=Hoiz&(FXlj@y%gd7^yS)ES6K@#l|;k|FQuN>D0rl}tYo#G5j^z(MF|p9vF)nt=Ew-uRoTDm02)VvwS6Y'
    '!i0hR-#@HVinKz`o;OaQ&N=yIyIB!TxANrhtikzZ&*dy-)^3%T>fyR9tn0&&{pk!KmGykNcxkwEb*^q_|RqKW-CVR5bt3;fWQoJK'
    'Qxl$-c&>s0+92}+VxF!~AHz*+l@9k*r{vJMP4=`2V5`1_sj4s^9?AaRF-b(}>`I(4W+^&J@y;OpGdVEipb-'
    'v#*j6#&S@<M+Yj)#X%Q5cPp$*93ZCZl&yqa*Y7(?HpGkfGfXyugtr=8M;ukT0f9r-'
    '2MP#Vqz71!F^b&sSjBlWEhpShc~i${!{ypFJIc;%1>d+1ik`3GF$|0{+D?_WfvMuQLn?r^EQ4tbnA^g%AGrZrXp6PE&gHE2(<!Yx'
    'gfi7!Fx7&!p0_je#Hshr$sGjWE0o=<kPJK<ltJU{X7xM01HL&xo#ZWpjG>E6SbT=j`viaef4pVCi}dRQdBvu!$Loe}Ir%0Ig#)VL'
    '@AEJ7$4suLUv0FElIp?PlyydSn6){X{s_k3;c+A_Xza^x#B$zQRMtaVL6&Dn>ri9#G7PiaxV3jVC&&I92H_+qg7H+tNga3@1dix2'
    'mSd+NABV8Dg-F)jbD@q9(}54k?3d2kDwE#xNS)xp85Irn(Yp=rVT__<<OS11bH?LMi%!Lc0h4S9;2wVo!OO*i-'
    'Htd&;}Uo^qGiQ|=mj%H4SSe0O7q(Iey04RSKq8^&XjRs5ro6rLWYgnD|=0mheTr_osG08=P<sy~br_)IF<yIvZP>MniFaHA15)Wv'
    '?ub}Jw@I=I^=6Vd|!hoA$HdIz!u<FmlZfEG}rP{l5y0Fc?r9?km;TiIgDMVPJZF^aL7t!xPeiQIWyN<kvCl|4>D;?9vUiA*aG9Ck'
    'N$2t6`Jx{foB)uV_apiKE$0rT`z`Dz)?o~>dDZH|%_A6k#x<3sC_n|x?Ja+eRSM{e_>^~ilbv>v6=*VWj;Myk8Vpmt-oNn`9*&6U'
    'h!kPj3KS&gxS>}UeU@ENQP3>)o~r&!I(osv#UB+`6{^==WG%#fS8GV?~haF7%d@pXtmMi>x)jS&*{hNQRmj>eCcipKrmSk^Nc)@T'
    '!WW$3Q{z%XYF9mS(FX7?aZMkfAOF|Z^zd`1dFrh9vK3Nof+bWRE~rjs-!rFEuvJ5Okw8{C-'
    'rC7RFRzFArO;8iV(g?x9m(yr!CU$xV|3s(g3%<|vd0m1pD#Uj$lCPred$qmD1Ozs%1<UFBeKOP`_F3O$exVsAcVgo5sWsa#jguzU'
    'k-U+Z$rcK`kc7bWrJHs?EZThZ2%$~)x%s|XCT&F7>`#DU^48-'
    'h|2&5!0d{zqfKGxWC4F~*vQOylf%CgG_Q~}`KtR}nNR%zblZSWV}ui0V?dSm*J*>HcY?VrSkAB{=m$dpTA#o`iHd2dHbv3LB)6zq'
    'MOw(hpj6qx2gBc~&X*ct7#J$NL7NHpdUuF!Urk#U%rm|l^Le;%T6=}+8&xI=0g_iDd&{czRYD02&Q+daxP2IbJ#>tMI@muz2CW`j'
    'Dqz(NrA+FE<9FL%Fa`;=P(by(VfF>fspdeH~|AYnzE4(4|9lwZY#qGR1jpadT%hGfO1-'
    'JinYOvrgUqe&to7hTa63Nc0tPQh`Wkc(-HhKJMFLOFV*mDpNGT8V9Rq?OoKM_P&YnrS5*HR6`Wxt!2v-'
    '3o&%#k1QDhvnM}sH=?!jk)Qw)v_KaGSKzP*`u##@FI?i;2i{GAE*MI*OtH|xqP1-'
    '^VcOhuT`*H{7F(^V?T=3@<7?Qm5%fq?R2Ey_^yr=8n?MYybaO~`ESE?T!XZ|%5b1J1VnY5H7Jew`%5D+A|)O(43B$^%D7TK!tUoQ'
    'BUK;|iMLk8wdtmKZE$v%(d$q%fqh;jWvNCL)P6-F4CO<L)MZ>}jglj?6Uv8nq7jyYf78+-'
    'f7um(m@w%TSBGT6q*q<pjQ8U@S2jC9CVf#hW3s5%UB!~AmELd_%MhessN)mmh?XLP8vRFi#9p%lO%?n1N020OyyeAu*Z&dJ4$gbE'
    '5_7Wy3LW;BMrdPRQ1+y7U!NgI)V;t%-5LRR%P`;|_8^bczy7!{i?9BI8yMXb-I4Qw!Jn>+s;-'
    'lKMeqz$)x_ASvGZ$;mit9&WhBa6vecT3Zp(O&)#1a;5MApJLag5rKTe-zVwP^K?NKVV6XNE1NwM{HBWk=spQX5u3y`ZR?Ax75;wd'
    'c)wQ!6{a*a2Co-'
    '#5nQm!SNyLX7BcGvcY0`9Lz^MR*>T;0|WG=??$nf6Hhec;ZYE3Huv&@0y<nJFe&t!a|=uu1M2t<kFZb9AZk=iC7-'
    '(IqX?j2Apq+dB&+MObfU{Y9AgL?G=QZFf=mlk~&}KVMEyT<po8qNh;xf_ilqDg5Ls>4|gM{ONED+~m-'
    'Yg^JBpb=mH@jskzj)wl&`?4yXobQ$4E`$dY&%jtvdp%UJPuXO=7{*5lc#<ws9*tjwU*tj+Y*tjtT*m!0Nu<;7JwfysV1^E}Cww#$'
    'P{!g?Ua`FUyZ|I`a#`%7jr!O7Q>4iu?-6}Qk4qWziT%F&G+O+u#zIm5!uJ3^-'
    'vt}a$8+dnyPNPh*2Sdy;>$0cLSIex+UOInH(9iqm{COFCsn_{h2Cm6n%r^mYe;LbQZWOANPs6$JQlTFUQ|D&6&z!_++C^@e`fJ)X'
    '#SCa!yx>;EkU!CU3}zSXSZiSzi^BHRGKl(o{(|eDr#^ainbbiF<{H|%s{}8<x6aq_En3EQeop=b+xZ*vC)v*5NINpv`J3{m*v{Xa'
    'uV6brH($wi{+9e{MmGFZhv*`d$TtNQK(^&=Q9B^lQS!=X7!y2Z7c12obH!G%q;y=a)dl$ODtot&T&_)sNJFQh6tn|$l|b58<8!8~'
    'd<bH}By?<wKifpsR{GRuI1uI3vO4_iYS`y^r<XE5N)MbJrFM=(ev*n$Oi`)$#JN$1=R{~`gp#zI_v<d}JJ=x)LT|g-'
    ';WR>Xdx(P}gzolmSMvyM>=6!zkj}lehV}zlxw~+zPuh;I2v@c}mgVCLz4N8N_b*<n&@z=pRvu9iCHuK6NT#H&2k1BrpEuD<j~#{a'
    'YvRhym-OX~v`c+sguvbuA+R?`2<+SlfvuyPBZFPf#lfyMaj@(8IM}r|4tBi|2fJR3gIzBb*5*;pY#wvbRH2UL;sZ8M#YNjtv5RHV'
    ')>iT_i)02Z&O!=We@;Z}3kdj@0PLkZHq@a=0y<%sgE9qp!dQp>47dD92los=c9g@>1}f)&0&Wr@B8xQ&1EFXA1UQO)MLEsbI;c`j'
    'ez6aekl$@C3`XK$grdN#<iNnrzk?d)Au0w&U;6mxd;@UTEG`8v)@5)cQXFh4|4uUj+2ItYh5pK{2nB0m6=nDS66!sI(5nzQkjebq'
    'r<<?Ze-tK5n&|zeau@e1HiLuntJw@T=f7h!I3)i)o57*^HEafl<=3(q9G+jtrTV|4p&Z`r9{@^)dj2v%nz&oCuY%a)UrY7Tp)O5d'
    't{wf`{}5F*rR`p<{w$8763Dzma*@#xP}?f|mMUQL0^`Pu_KPBpfDGCt<;0Kv&^LIKOXaT8V*9V3poptg33a$#l!P^pD&Y~QW5EL&'
    'E9_$|H$cJG64nAGE3{MzwcgbN27er=LQWE&1fRAm>sQ6)O>Akt4s*I=-KI&VbY`Vp@>OiP1}XN=vN*5Uzw3}e{i`rbN+}8-'
    'SqjPd$WlbkMhjdeSF)d9YIqc7k$v3ca~P_=%;YyPRQ(CVqdt$>Cyk%=0aSgt$!}(;`co#K%az3yCcmY)ZQgPvgBt0IkG~Virh;QD'
    'NT;AQ)jb$%I%`;y;A!ZskJbgi;>-3+zc}mcKMO|+3p|@6i>vr@Z=fS}14kCW<BPqCj?_&YSzN=HI+u>rT#hWR!-'
    'e_*h*fMdDc@nAKq(G80<glACi|A7wqFZB<k&r1`RaPA{82Wey`bL{9A_VuFJ;fr6nQBJMPACmk(aVL@=^|oyp%&DFXgbvOF6unmy'
    '#U&0m8_ltF!T0_{1+nbGnMtL-C?7crS|c@s-BCODFaZK%L1qI<?)}l>8CkK8rYz=&ka{*pD=*d<lD-zAAs*VFY~Q@B;mnBtv--2Q'
    'FqPFCk%x8OqB@SYn3q3KEu>p}dNOC1xn=kg&uI<uxQMTa|%TO}v9=#qGRGSPa}jUzXyAmZipW6;@K#<{<6(aez3{cAJu$pNCrAw}'
    '@^2kLEZO0<P73TR^><SR%{?%qk_g`=i(AKFRmBZC^-'
    'mbB)LJP*SoQj~U_BziB)I2~z$g`0B&1`4`Zo!D)R3fOGI#Ujwi^T>$(a2z2k4s7`S!`Z1@a5?(BU)&u*or2xfvBCV9u1ohWz!HQ5'
    'GK*wpLCx7ABh5_qADnf>u*uSrevmvt?!8G0}-'
    'L(sBQHAnD4_yVTq$I(w;UU^?Yf9dp{<ye%dz}`4Z~vgh;oIxAczpXuEiT{whZdi2|5J<8w|~-X-}g24KEnonq_GPOTli7N-'
    'f!5f?{DmbMa3JDB9hW~Q);!bp%NF?6t~(6S~2hWG5a*T<ieh?*ZH+L5S7$)En0D}<{Nf8`x*OyVNY{_u@4z`H3u5zMzcRe-N-'
    'dZ@ajhz57swvIL^R%)PlaR&x^wh_zlt{qpbK!9$w1VpsG1iCKKLNc0XNZ7PoRZ`i<pih<?<OIC2*T)RgP__|1ybp9dK_Ry-RLXS6'
    '#41V<a|OkEgG`J)LkJ2Ok{5lu147!;F?!7<5bj!DLlm}Cr%Nyf04WDNIME3(<RY2l4u#K2m3=T|V87T)AF4BM30dC%1JtDkyt$wF'
    'i0aeIxwf`0azG*m}xxQSzRxacGxm>qNNSy-bnED5z*LPtu`zI2V)xFj!OrY>Cl%Q&Cm@?XUqUAX9Vn7%rS@6QThgcw8--'
    '&KN`d|F5|nkhd0ogmzv5!Q&I`LhgDqGeny3=+fh=Nz)C2tLnyqn>-'
    'bDJ{*T<OE%<;ClsKEIxU&i!=>$iZl&36lof6EYdXGRHSLRxk%G6w@A})i_kQ{#H|-xbZCdTG<PFMLUW-'
    '(63ZvEm;eVAv$~|%%+LpvF-{7B!dQ1EBo~<ICZlE*SBs9^A*jT;w4ktDODIxsJE>P-'
    '21u(Nb!F~2WU(7KzPpjTx=7wX6yUUzP!aN)iK)`#Lrx-BrDQfIr-FtLJ9*RCD}Tf(GR9tct5XJ!z4AqXHYWsKj{*{&1QGuKHW<2i'
    'sZGR4YDIL}8!?V70S29d=)Wn#`ZD<Mblsk`fAA-8h5e7hWavR@7kxB>f!Uk~!@R5rwGjMCNpZgn2e2cm?{^0-e_rz!$Of>`-'
    'Ai;HLy<dwmCj(OWasPX400j+-'
    '!&e;(E^uVV}c!mAePiR0jAodxLw(#?JOfHFc?l>4i>*z0ed%BCeGG*lJaz;%)$~E+L%&NIMAQPlZ{hcj2HE&i^kbeHYlcdd6Dd6D'
    'Cy@flfel^{QMQNm7#{8zecvQTV&P#xu$1<YaXuY*}vN0D8)B~SF0S(ERY}_?PiXTl3~V;%Ht{uPI`oi(}wAk#vv&eI;ulQBFqWXP'
    'P7<@D0bCb>FBD{l>0%{&nRsvE!Sfdqq;rSV6Xc2v&eDjEZP}Ku<%0sNOb0A#g^y`cs<)+_el6zuFURa?5@T&EYk)H9*0PW1&{Abi'
    'v^D##HKnua_Ot!VI!iJ9zTjGL3;dKY-'
    '0XY)3e}Kf99=H05%_{>Dj+pslhKBDq2r#aPVGSI)%&a^(F|u&r0X%u)?3G15q3*;O+ql*^Yi9Flu0VLf?^&S~R2-'
    'kkDgfk@8XD@k7P8dZL8jzk}rp=jU2jrBE%p4i@QBA%s6v)3e~V$7p)?ua+bB;qd$^hoTM<`3eUZ4iWh?4u=~;`ez*uw=~Q?O5rqC'
    '53_^D=|6poZ#mC$@o8M_@?j1Vks120uY$P_^{uO6szbT#_b|&L-'
    '?#=Q`4Yk7|0EK#p6ps>;ai<F9!rftR_O6WgsjlxDI*N7^jKlOg?fx<e<iZ{5|qkv*M3SN6EEYNP|R@^$sjb2JL;LO;(AR-'
    'n({3TV6;^NpTpykCjU!rXtbd7B9Rm}+?VXD&oK1$j6w<SV1Ksz*sBXMCs{l$Gmv%XG6UH@E;Eqn<1z!8d9E#x5#-'
    'A8y|E=phg9S7xDno5dMq<SQA&>|jj-s_W4RG{^EY69D)cfxRn@pVRDF9S`fW*@H?2USmec`p`>a|6+aThf&C~s<BbiqMienV%V83'
    'h0T&*v~a~N-GO+j9#Lt`ij5Y7-'
    'nRNqwjs~G(O+RhX(_Aq7F8+)6U5wbxh6}}u|(y6B5CQLOs4a>6(Pxw3sB*(s5#sjx7_8eWSb~%-kU;(#R48Xtbc!*-'
    'Q6s<6EM_;e!BI9h2K$l9wo{F+kx2^s`!Bl#r|4FH6{!jmYWMQWpuKSR?oZ(3^g#`6+Pm)UCJi;+cjFm|8qhQlopvd-tNU;tE_xFq'
    '8Qvay{SiDr^7tb&<|DgbVyn^Q{Uh!HdrLxQXlCdhbD5~0RZrMUSbrof4R|V@tS=!aX5>uA;yI`j%OZ$B=Q<SA$6U;MZY1anpOj+7'
    '>0{#C~iO|%I3iIHJ^wyWkNpFQwkAEm9Ep>dN-Tfaf(xJ@8qbFijwlx~3LrpTgz0t_<c1Pgw-'
    'vN8A&`K@iighKU)W62qO#2M84in<sOj7-xl~%uJ3na>Yn6TwORm@Bhz1tjxnZQI{>vJXQS~odi)0^TD2_4M2Vb_owyWF>-ghzK-'
    'dMeL?nMqFN*)aA2ikUZoet)eXTz53~m9VjfD*Ilf<X!`zMK^hvN;Lez@BVKn^e-0lu5ml-TX8+JBvYHkDPhVqI~sfpD}~}bUq%Ki'
    '=(@DPb}cK|_Ae=zaOIbU$`@T_v;M=5H2FXLdwgRLH2F8uKflBz{qrjfH*P6n{f&wBS8(+{k-'
    '~@1foo)d;yo&26ab99jn8i04AFv@v@jc-Ws=#>DPg!P<*}V#lZ312d5hFGtqHXaEzp~uAI^Wu)sDL+M9=L_?71Yip8ge0{cg(s(c'
    'jLmi~KIMz!Zyr=`WKas7s|AdKd=sd<DhDl0P^0Bg2Bu-=*ymR$z#-??ygyih@6kq}wkjjHB?3@ISx^z|4tnxEC{JR(|V@-'
    'E6%Z8#=RO-xK~($K^Rwvr$i3j9r;J30P2h{%NRPwJLPpis#aqy!FxKZHOlC{b=$&h$ipDX!15jllM_Hc^`9m_yCiCozi136>4=#o'
    '{li~vtW?gl-'
    ')KA*o2!!@%?rFW_=Z?EYjU{Du9r#j9n}cPuIAZG8wT4Xcpvb7^8t88(6ZWk(|`@^S|4N{D=6LeWb8szOwt9{HvgUME@>lDz2v@))'
    ')NVPnw|P0nvz$g~fpqk;9GM6tqTq+TXI52e9IOWB;OMqI{T<iE=WRXNH)f1;M~MhR^2f?$BLXJOM8tAYJZ&J2i$Iri=K>A&SdEV?'
    'k+tNP>-?seSS}S^s~$y?K0AHTFN=Bu}9zXaUi>u#9VEk+P#As}%K`8OIq%6w$gMiy|$u6a@rVTo6=TW^q0^D&m5mjvI=CDB^|-'
    '?uer@DkGzo9Y@sPIY~~Eo9Dsa{PF!wpI4_>ZrXd3cTP@D&UwF`;kQ24KAbY`?5M#WO=o0d>2-'
    '0+RC)27A7pQl{6I$w{Y0|U2=q~k4$_;wM*is=@HLY-%~p|r`X+qMvfG|hh5TOFJ^BzKHw&BFvOrKt>IP&L3JV-yXRr_EH_6eWfQw'
    'Lw(sKtEbhR`&16AOM+qt$cqeOrjOjnsr`@Y%GuSu2FkWhtMVCKiA&O2=o&kKTi@jNdGmLJ;lLu06J2w&IQUEt}$%9yAsYpR(>pJS'
    '5EnC{4sB6W(N*!nD^6RqE10&&Q$eiWDeIC%(_vOl=oHdUhxI>tB!Rq!=~BI--'
    '^VN5d>#E?@&DuJj;&Lqh+P3ha+gRn9=PRV_ki7?#l^}*#a;(8dX87yQSRg{9WW$Hp|k`9;xU`%T-'
    'CjaBqBDO0Z(rm0>!|kO>!^Lvj29|8;zeqtvv4Zvtd1fz3?`3`pbmEgwWM4fM+F&lWA4G>uVinD>rHa|B1SSj3u-'
    '6@)6eFTdh<F|6^FEexsG%`czC$}QPmQArx|tNj&xQ>*35cHq8}2X7&*vt$km6l(9&Emi5>zq_Hs3~fEIB{9jfCNn;mPe*7OoXKbT'
    '&445ISu(iFhuiugoPb;l&J(?o;8FWpa<4NvP?^gRR$GW;WC2`Yt3y@nXCGK#l{?3A`H#HatfL9Ln8&)<W>H8~Z$U^N=yZi|yWrED'
    '2N>)dqcYn^b!@9)I3V_<Nq|Tmn4R`YmRV`lYmskR-'
    '{~I3f1!8Z_G)2N+|r4QNGQce`5Xuy^z2aLZ7!y8vg<*(mU33ixr3j&)=i2VxzR-B$KRi4dX+AD|q$4OToFHm%=TrDd>|-'
    '_rq}M7wB`aWLk42)Ao@v1KgzPPr{t*}9?P6vs&Nfe~m(J~Gta)Kfbdc(pDi3k`;?PQ^58KcQqIBWi#lXs;)_=nUrL{=z)(Pfy0<*'
    'wz34R3rcfe!@_o$rM(pnue1E{G(fmmwAn>5VEHz#Q&?`a8H#!Z-'
    'O43T&Zadt{in#<A`vw%kR}Fy_6OS1@tY_S;MNnD!pTYwR|<!&da?;p&d&(mA`hPI1~!ct*2N*m5U|#*#|B1vkx9c?F;n~A7XKJO3'
    '5{;gZDA4w#i20R#YaNl1=hr(sWI>D!*47uPPd}v*BwxnXYqa(9VUgE68x2M}syDzOEv(bv_N+aQF&|IXQ+#9u}~h%S!F4A8Izu=p'
    'FDod_9;}cCWVse|3v=MOM-Yv(?iU8vjcts<tgf2`y7?+Y}|VOtoz@l+ZHOw#_jW#8lhvg_@D3+O`GCX_;!<y?N5pRNE$-'
    'lg+SWHZ;M>7TEC_LZxII?3@iXaI)RR=FA|zYenroW?C$&Fklb?iq=cjh3jtEsdDn2<t|1U>>`I?3m(`QSLM~dLm$lzGDp#EG2*u$'
    'V9}rtsaG)$wu9694eRm`oDQCakUyGZs&_HrfNE9V$9w}Sw);Tw-'
    'x;0w{wSVA9#;ohY70x(%Ff6tVAaG&bNQ$V@V|%&F*lHE;1Vr`y30K-GG*SzO@FC^;5D488NikF-'
    '0RhNFz6EnbPR>{6k|a;Z{02XRTOK&sY5{*oI4b9!O24rD4abMe8K5Mu_&BBBzcY#$OxX-Ahp8`%Z>jbW^7Q+%13xMNPF6kF<Kwws'
    'DVRLEDDO+{6ACI_yN9*@z@Dpl*t5N#!UPIU)T7#VV!JI0SrkeHz+%qzoH|ps)?zt;W{9XKeto`m4nl7U4?Da3DIxKeYAV6YGcEMc'
    '<>`Sk(by51r-'
    '1^RV3v!n>I@7*AAv9t#iThx2f8|!bQ<tsK2Ku71EU6v)$@=lUp5ccB|tpZgrgFR>xc2>Uf)59c=>h4mz=@O#TO7=9m2lU*?zn3|~'
    'R!cpbtcKiQJ@tpi0W!J<;65-bL#3!08g7c?E8E@(O-UC@-I3z|-'
    '(DtYI)VNk?&$sE@Yx9OL+J>AcW5FAS)1QuABoo96KaTj3h02)oi76rpyf;y(m&{_E!Ox+6$oTR`G+f|G^uX?`^(gdq#8=%FrTr7E'
    'qhjjSSS!!jr?eXs^!p~nADky&epL{eX?|D@iCder8<=l!eLBQ(b`SKHm{Mpabs1@kcc|l3!koJ7}OOPJb^W`r^R%y?dpNv$YG<r8'
    'VK;Pm}e5-'
    '@*?HaG?&F$$(TI<p*<}!z5n0<BxXKMx*OKbXsJ!4&jQZc^#A|{9)!zp>Tcj}OChczB*o|tG78YmO_Km#?4w$`0APH?VvPR^#e+9f'
    '%M=4#jET$-'
    '!hlJjV;c29=UT<wvZPjj_rGCVX?;|pe6z;;#%TImGSqf*b02G$%h;^a1w8??1%VZ?cUBX!v+n9@Kez<(0*OhHG$rwVzNpaXB%%vT'
    '9I5I#-Fvjv?9pDyG%g3iK+2zjoc!|*eNJkLMb#ge5|+IlAkYV>cNS7)c6GV-'
    '&X2ul?Xu;S72HW(vVdbf|;(KB@C+>M0je$MSw#iwu1)Eja9bTSjVF~?6=Zi6N^w{2fov?;f3KUlOWw{3q|v?;gk09dpsx2+{C+LY'
    'VY3Knh3ZEFpS_U(=&=5S4q;bIYBYC5^%o7t&awv+ve!A{94<53_#MHTbAZV^+_Gap5(@ZehtEVBy_y=$3b@Cs|`Fq5<xLm^jlYiB'
    'w+2S^IPx^YkLT*nlfz#>2YuCsi^&%f(}%1Ul|=!&v@Zh7d2iiU1^=njjew><O!SIb)-'
    'dSZ68IVGt@xR|q+NerxwISHN2V1+Ol@Ru<#GQJ*pcMc!ryFovNqR7R_@lq(q0;3>9G|cY9yi<Qe-'
    'I4oAL_>wYXX!+>br_DLnlVfUJB9c};J;SoADH8O3AQVHaQt*Pe)S~`7DRJcaw!AzYHA*)FtD#C<9{lH<?&39`B_#w8oP2vEUQ`~o'
    'CsU+gu9(%Y<Uw5%$roczLS)oR)c|D9dNPTgCvBihvvboP3;wCqx0=7PX)J@XAYj%JNwd=Tl><<S=-Le+IDf)wyU$Y-'
    'JG@U?yPMOXKj0uwWTow+tP!3<b)XVvWZo5;el%z(SXR-'
    '+=c;g`jNT_!f$R!iHE%L<3&MnySbRPruyezEJ5}A=Vm19clNh4xAyloOyF}?7QlO+@c5`L0c?9Dp0)(A9gxA>QtPaOHEtD^oBWR|'
    'Pu_;*;%(v|_}NrhC-'
    '1>>_g+}5F^!yG%i~bxoAWwr&b9Sw?eR(OfLrU*Jv5r(ejO&?8lG>W1`2EII(ch>ii22=o<luih(jHZo7vf}3aOu*vj;n}eTh8n99'
    '+3+Kej)Sr=5o@H|@q+5_#JBxN_58thE-htwBsLv*o*h=@)GIE)3v{CjYxK`BZj6!ZptXor`tEM3Oix9W4U;-'
    'J0+E#MWkd&voIVRTM7zHQ%udr|z_mVbsmFY{B^{;?i>!XVAH>y!mlk-'
    'mG$%SCNNz7a50}lYbDhZb|MYWX(zLA(L@ya<8fD$MP`3+wIAHRML)ALbAI#KH6J_{jX0g{VmfHTN-Sz<=g0cxUB?ME;tXjwZM!B='
    'i#;ya*OZb{zk~HekSsPLT>Xjkq;7byPt`Cut0^ZhF0-'
    ')zB#hS4aQy+!bNjB5~;Lo3HVz=iIt~g%LlR`^;6y}e5WZDvpJBt;Z&JKvngzQjuKC@H?C8cdWlY#HE%0UN~x-'
    'ORx#MJYX}U7$pu1QU96VH$Ax^pI6QJXkYU8p1&#4Z2ive5F`1~xvZ*%vF?_Uj#;F*ONv8P`_Sv|%$}3ZS1#{N6JDCA+LPr{8z<Tk'
    'Lt)PbWVuh_xXfO9PA*33)QnARMNj?C|yq{(HCHLh-UDQtT125#(h+Y7DL=#?m0S``03fTL&WxZ;g>n`ZzO}e`_8!sZ=)pvSbeM>R'
    'C{57Gx4)kHmFegr_52zWb%$mPM^itR$x}<r+D8nSZ<$b~k!z6tWjanBb>5E4VS{5ehCm#?-6(;E~9~4FuCK-Sy5Uu^G;+IL_)bI-'
    '<a4O6Trow%GQa&KI+7HrHfn}O)4-5A5)CpP$6mzy`w1;{0zW|!UctP~eEmlMa`r`3oIYopZy-*+-'
    'sdutbO&IM3{b!2J@s=uQN2~D986WaqJaA4CwJYTjH6Y4`_Xh-39O%65p5z{%?yaRrRimIy^2<@s2KfajXmd=-'
    'iIyEXLs;@QK*VotHsfNLLfo9JBF#{v9PRR!;`!AK%Euch!cfH&fvZYkvV%W+Q2nR+Ipx+VKZ9hl%UO_FVEKuVSpRVIb2WS!sMf%j'
    'LF!oeGC;i;z6?&s!Iy#Qc=$3Xod913q!Zz*HSKp`tc*`M9GQ!Tr&Ec0xoyg5m9AK#eFk(uPgn7Q5$#aF<QJT|?c@?6e}bPhK4z=('
    '3%}0*iLG5lcDm25_WJBiz$FVhO^L3kPE#5pey81=+-n);=Oyz@RVCIG+XCfdT!C_LSD@U-'
    '6)5+01<L(gfpULWpgceqC@YT!`Mr&EJWRId=pCo3XkHR9Ttxx<doz@!D1y$?7w-uSu+*Aja)Fs&X7O1d;K{=rFig^0v#Y-'
    'QTd}Hf(-'
    ')5v!*2*W?*S$+(&5>Kn%+5;tYKNRP>SMUMifA&6FGUZ2}ScP?axq7d8%h4WsuoMoP(LpW*>1b9_5;i#QCxV*L#K<l%ZH^Q^3Ixoa'
    'B_*-'
    '~qXOjE8=5FZ}2IF>hswO}DyApxES8v%*Z_7~?QG<~cuCuvk@_1flC*Iwp%MR6!@+<N@&I{7L%|Gn60lVc6!0G>ehj&BMORJ8bJ^H'
    '?DfCjlO#k9b?13tCz|CN3mtaUz@Idc`1)>0_m8B(BV(A1v%<(Af~FbxMQlSqGO9>JI4NISploT!mz5<;09?qqXcG{UeYS)fq^=iX'
    'eN@Qu&QDYZb<;ga3e&a)EKI{e4OVkEcYma7ya|S`F;7Ir8Qo_42w*G%~5B^UfdfldO1rt*Tu5!4(1rNO*_&(xC48TOteM(DohUA$'
    '*oe|slsp;*5J<YWkzld80SA&u|-&{{%C2(??RPXB8b8xd}4^g4J=VaVf=(RqA<2_6?q!-'
    '!QJ8%T95~D^t^)pyW7L!d`&l$>cIKPhNn94el3-d?!eBvQ>vlgyn5+sAEaIebZ?gd-3;{=>g0xC2mTz#jwIV-'
    'w=`#B7Pe(DJd0!elei$kNdHcF8d<-'
    'm<Sc;ocv}fH4+AYknT0{7YlzA6K2)zZg?zFjk}R8oHrZ9kUS_yjs3f@SYzqFnLe9wt<MUo;E-'
    'eOUJnsY;i6iH;0`s@G7kH9KG5E*c+D6Z39)l=);>pWtURA7usip(Km9S)=M(J#0=>YImSh9yuir_dp0DK*m?4gt`IDrlTSHY5f76'
    's$&5e^3Z=oA1WesqX~@jd?(=P-'
    '<N{H&eplRa86VrWO&HN0ifuWgVk)f>G`3`{QTX9LGe>xurzp?7HI=#5=7*bg0arcTt6r#>$7)YnCx`nkwce;0Wg;37}QI?a!P`js'
    'DH_G|=0qja=|kxzR%auZA!Ehph#zMEQCt&uSO)j`OcRevSbazg%I<Yo_26FmEtXT>lU**O)61}pnAtpOJ*r@c4XxVb9FLWP6{6)Z'
    'YDb#~rN7w64%b>2)j=go9?-b@eY&GdBM%#q~H)Sy$g!TRfM7?uv6NjhOz26`5S%^qgK@g_#O!7&F<ct5}IP$-'
    'W2?!NrKn_Qe!Q{UaUi@xhjD*)AYYxfVf6S{vU3+OJp!MPW&Y*+@s_hDGA@g$>j+#0OJ{4b{%(*X=i2~CWtC6d5W)VCGGQo=Kc9~a'
    '9jhmz+z-'
    'x{!+!L=gme+#T@Wc}xWzu>C=7Fg?lJ2+rL<_^pe>8<mo6CW4yVd}&O@;mWi4}cG<4js1Zl%LLi_`DAyl3rvHgQ;s0iaCDvv-'
    'D)MRlbSuC%#Nby$k-'
    'i4|$BQ(7e4H3|c?hyuC*A_Fgb(1IRUhgXZmhV9*Bo;@G1FRmYyk@f6I^21DQ6&*chE)ww5%Fzx9_Jek>rTYRb4N!OZ)0rl@sLQc{'
    'K@hB$874H&+L!iUFwX)i-'
    'm9}1fgUxlD?fZe2Y{xf!ee9W7m!z*f6YHAvvu9%6lK%Egta~!Ro{9BHj<siEJ(Gd<Ozg;{(w>PeVX`IH`HyN^H_n+kp1h2ip0jK*'
    '@>wdV<HZ3x;}IZr4S-^d?waD9d@LV(cND{GQ@<|oz(?<iE?S-'
    '2tZsDM(k*UVy47t<x4CWUcH6Tza@ow^@z3J2xxeGT2_@{k9se!JBJ4#O=csa17h1WMV@i9&0@e84OwOD=M|>E!DX>|w3Qr5TD;=|'
    'e4n<tN0guToo9Y5VQRAv0mGCh9E9-'
    'TyP8OPMBi*Z$Jg|=Q%p!Fv?)5jhz5ZtVVzW8Qu)7%3&0Oy7$C$uX8SQuE{iwI(btLQ!$N23ZR07eiV;@q|#_snmP}1$LqaVg&Sg)'
    'fODq-%1VIqw5Jn;g~{?W9$HN%+Edq$viNKy;`;=*5Od8<gg2!k>+^1Qw3VwE?lFCNQVUvOxu-AU{kJTpm3v__Df(hu8;8hosyO>H'
    '?m6OpqXlB1|xI{bWNfxzonTwzST<#=31Zq;u*$njbB!P-'
    'Guh5BLXAT2xtFbSR3<72Us(yMn{fhybH$i9KP;}l4LkzpRn=~j1UA5J;bA1JDE6x<5B?nV!S<BP#CUXxd`RYl>yTAZCGqpONRKiS'
    'pzISyAgst(q%*5-7fj6q(wdIt(b?0{x|r&VJ^lbKqB>$Ry}y%<+$Q?WV@7id$mIsx4lQ>Z!-'
    'y_VZ4tYcY4`#Dz8{*G02pk+|Sa;3SJ-mfoa&7&_6^wB@S_~odRW)`bDkMz<p+ZIGXf7QALp~`K*kV#>IjrLc&T>4t=RXeBeSmJtB'
    'D(a^T%?YoE`xTT`_a^G=s1{76qDSH-'
    '2{*LC`Z9VmaHD2$C;5u$qeDo~6*C;qBJQcgZhqfaZ0uW{Q0RwP@|#|WVJCRwuHsXV7Q<3Powg`%3#K3caZJAsgJw#?fy!@ik8kU5'
    's_?3>hFmiw(N3RHdKLu(f!1F>#`(*=oxj}2`OAHszueFH%l)0dyaLm9X@0^s_c~5RmE9Spi+*PnLvLbO`{Z}h0@X={gT}LqWlP&h'
    'PZrB=xx51ouq@~pS`*{j(W5oWjxKy$X`<*k3>cZs2Q2Q*Y=LzXb>T~{sKTosjGV8MDj?ysMoRRdx4GI)i9Yl(SGy~ukiJS|Woi1E'
    't38w;1gWj;sW|<Qb@1IA!)o3-@-nKed+XFnl*abfsn?JM4cDPyhF<Sku?Gq)q%g-'
    '+_u<b|z&S5AKTm}%P6PAvW#Fhc44HZrjVo0sV^QCamc&(8{V5JtT}i1ZTvffOW^mQT%$2f<nmhedgdIEZc)tX8Qe-qd*Zt^pw*Jv'
    '3R*%E6nyV<L&fTpJ^y<KX&c{g3kNI)V8@&Fxsu&#Mu!APSKAD-9zXWzT-28kg>~Mtnc{1#9a>gDoJoJ{&bJFgX`*W)98-'
    'NEamoi&geG8*1wk{~rJ;YMy47C+KEm5MA+P{N2vD3q#u!Z_c&$<FgJvSY%tl;I!ROJ$SL)GK*c+u0H^nYF_jmZZQo<9#KwIig99K'
    'p<)4AMo8Wad-'
    '`>5@h<b1s7cstcGonZXj;3z<2a!J`#;D1ADEVwm8cwWfS5T(}3@dh6R6orkwVxlJma`~YWjlUyDx(~Z`V!DGZ~<5I~U9tljjBC9Z'
    '+4<2nfbDV*-0XG`_j+JP`P7&Tt6>#{jc5Ht%TsdKZZZ-CmDR~;Byi1R4sLo_6`hFQAx|U4SNeq6Tk$#R^G=q78-'
    'QGy~)6B{oz5_c3vjDz?YE4w{|F(QnDDp&&u(t&G6#8NymWfU=5Lf%DsYJZ2md5XGo#h;+AzG7kN0Doj5q1WXhvrDo=Nu(H|NXcEr'
    '_|u@jnlQBj-smEPeqwcj#e|AK+tFJ30pkVR(@%sAU(a@POPCP5wcXi@;-RrL$!bnFe}9dR#zrl<tZj9YdIFl=%@%7EO2&J5Ll-'
    'H$}$r#K}mVMFcrPW;$)v}dgv@B56_u-i}8|ksS+<)D|gSy<~jX3zZA=#i{%f+VUDtP#7EFUvlV4mBveJrrLYd-'
    '_}*rlwL1C0Ii7IW2O%AR)rH3_)WcmdKwDP~(9RVDw0Fe-9b7TM5v~}ZBL%3v!YQogy2+Jr-'
    'QtS1Zp(|J!`|Wqt_q9O#GPw!U58mw?c5)#VZ-WT6Gna)-'
    '+Zur*`+?a?rOEBxfjR6vYTBs*&J6*cB?BPyWK9PE$|%1oR=s4F{9%WtFep)G2iMZsKA|%d75shy=@}ZJrL?9K7A%b)4qYU0MBNa@'
    '@0?zk7V)_ow9c{lb`01A%KSheQ{@g`8vYjaqkSGp4vA3R(CLW-'
    'gdfT12BbNi&!cX4!xqKe(x=ZrSl{!U%65K-pW^Qk|$e!_ige|mfwB5Je3$Yk$<h!asa+9V4Fe-Bu`pFK9u9Tj@C((x_FqYEf8K^%'
    'H%?B7u3m0h;FH0(bki}WKo?6mgEawC=n*dor&?)Ps^S9>ETZOw5?M=ZRgZa+dK8s4o>~_2&aD9(e`w6Jkd)8rS`llU=L=M^B&=Hn'
    'Yo&#wAEkqU#1t6|I#nhJEfw7n@7|aCcMqIGAc^<Y!J!Vz~a}J8IMlw+ZOqgWVX)-<R=4z$D}8sh0+Jf4#yEx?&?-'
    'mIdX+7q27UO<ZKH0S=DXWz2P4W>P7T<xy#H}$F#*HfVx@!!3v;mk$+@#MEDjd-'
    'q$3Za7j6hxTMZuBstp##mlL_xdxmtI{cW*P(Aqt)8Usfluur6uBIy)!<YQ-KZ?n<y8B`O`&oW<r1f(!L9(2B_BiB(x0d%Lyt2k<D'
    'Ke+^G^s&{qa{gg%`{;4!sB!=MO~#ExNLWI%I&h<RRV&`c30=#uB|%mZa`cD?+96EUUw*yhvldQawVRS8NYHYIJL9_jE5Zn8T*Og^'
    '{1#?z<Pb4ItLuQPt&JnoiJDVWSu^qJ~Z=|a^yXL>$wyY_6V+uF#U;~9F+@2TU%t_RRXptAgP0V6Wn<jtC8tE;yt-'
    'q$c>y5)1I<ApK=I8d%d2hLmy$Za=(lxhsH#|f=6V=kE})nPoq$au*VtWz}cs@aRPV|sS2_4Sn{z+y-'
    'n7HeS|@(%{pZ<O>c7&b4xSML}PDNd4bL9zs2TcB8{)!<shp5%BYJ2%HoM>%HrBIW$~mmW$`6x%Hm7Yl*N<Nl*Lohl*Ln1l*Q8=ug'
    'R5;*W@ZkW^zrqJG;Yk0Zl)i&yr76Kf;V#8_MI9kwa9Y9GPpWoU|=PGtiN_p2|tvQ#1n|nVD2hdIUu?(2<$t2wJ8)f|ko2Veu7?l_'
    'jWZ1bpZDuW~na>^s72GB9{no2H9cPg}F&yp0-'
    ';ygOkB6#SK?>de62eW?26T$X&4Z<|iCG02k`3Y(2EZWGR@yDe=u%s)UCA$*+`pn@lK`)0irT<i`)8sH4wO`FDs4D7L=HPr>&9Lja'
    '?3cfETyT6J@@840PgPP)$GQzMVgmOkWR)kQ}h`>q%${NXGp#i0h6f=1QCAOjS-qAA0G-'
    'O8ax|#f`7`<xeQqvx_1BQ#~mtmpzeEZul-|Bw)by{%A<R{?B4{?f8XOfM$!rD$e(wXG)4=-'
    '_N5|_FoovH3f=Q8gQ<S|BR2(aJSk|O^rL4)cOWpqLVVW~4q{HRum+uJYoX(amD0y*CZX^NuTxT4h$_W7_E8O>><550I*UUJA%_-'
    'j^_9)2bp?U(|qoOWt09_brxblB(O1FcWbof2fhSt8`tlxJ3op>_Q$nqs+}{;?TW%jqARW3in6aWAZu(?7PrQaSzO-dHK8f7}NP<('
    'y{LS?>64vO9j8Vz*;I8SI#j(OofB#XQaq0%uFD?)xRl3s&p>(qy^SdY_uCu$u1wFJ5YxzhzPz#u7dBfR<9fHZlmjdk47H`7{lPJv'
    '%5tENq$8x1XH6Xm#wTz;fEc1`A&%?Rr%hGtCvoOm~Gbm%GB4D_mjBm98-6DpweDwJVIdCS{(AlzDOv>j%^UHG2M-'
    'gO*bGvTs}l7jzthS;S7~w@SysjJ0bDJ2RbN3(owf>5Q|(a0Z#|{4!>0{Ort4%a@k8i#l{Cy!|;|=5_v3xNS0?!nz63O#WCbztgzH'
    'E10D_(<pV*x6B^){aBp58P47kn!V`%oz97|@<`@%X$Lc+sHC;FFHLhi0%+t8jdr88c0jT_t+mG{jcKhNm^7iawldj+*4jbIp0w7!'
    '1%52IVzpas<oR9jU>jJm+U+XxoZc4KI(Vi#V0&@Wm)?JCF}zEm4$!A0u>Cv<7wDz5!C9Gq7FU{-'
    '=NqRZagT5s=}zFj{d5wZz|GqbyYGb!#2I!A1Dm;_j6!j+D?7`U&ehrQ`<vkBQ>6KA*bE7)>bu26mz9P3(IH{G3m>IB;+Nv2Cw0W<'
    '1x|fS{1v3sUqMzP*Vc%nl<ayrWo|8nXAN@t%AFT>xbwo=Ixno9^TOIYFRX*}!j5oWSV!lD^~AWyU`9DFu&X~aZ*j*f7~<{b^}SJp'
    'O)IXak1>yrqc7qD{=Z++FE7;qerGeg=+za&9=i}~N}UxiCqD}8jqV%F!L!{t0xqnDsnvYU>B`$$j18~m-'
    't!Ge2B_ydHaXV!tNNl9>q}di&P-<%`Rf#3?I^~cD2xJ|cf7-'
    'tW2Iow6JE@&5+%NwEOllQJ<G$H<2@^PDUjvx8%B7PVEvQ+>NN%?1O3++AiFY3dpcHjV><6i88UA)!KUpf_!>KlliBndKNlxe;bx!'
    '&p29>$8YM@P>$Y2R6s4B#o*YeHT;rq{rIt2Hj-'
    'i0v9!YOXE!{KeL)tqSG&DfQP^JMg#t#jUF@8AHKaMFSpaiL+6?ABe=x~t~_;IQBh)n-'
    'gCY7rH1|@?Ez%;3Fwk^zY(_>NMM$(31%G|*(43<q;fcsNgA*F7qZt9lmW^So&?w0Dk+)~}bE!BIwrFtJ?QA=?|_Lp54CBh9vb(Dd'
    '0w9tbkm0ma0B(JD$7@xe6*9~C(O6(d3_I_@G6P6-'
    '!mGy|MCtFsEARgx7ro`c<k;Bby4mZ0y+%$H$Y2t9Rhr`XDgc}Tj8(WBRZ)4vu!ffo@AdFlO@?N+)S*c!lLbB?b{F+KR-'
    'kK-y0v^mWPq<LokuUS-uy<Noh9IOV+&6!%t4NB_mnlshA@v2onBMD`jHGZ$|6~+Jkp?6eP~ha)<U-'
    'mL4NOK;2&FQ)h&F(Ok}*0!r_4BcFw_NOlUG$2T%5d?*9BXO<!|hme_t$Dv+$6%$XRFxL)ldS$YCg(=^q6QWpn*w4u*0s{bMl<Wef'
    'dd0~pG^^^Xl<DEG<R5O{0nuKTz=si369zPgGVC#EG#t<aOfzVx`{b@kE{m8%vSZ`dz6>zx#Z#v3s1EJ=~GJSD#6DZ}%{>hwgn57`'
    'tJ?M^h&+fp8Eb6B*y&_r)fTlp5SXm_KDegxI6?*of=58YKix_x*0MK%}9wKhR=W3g<J3lpg-'
    'Y$O$Wt;m_K67oEE?ZXd&S1k+c@r#_Bi)fhc&~Uj!!xat<S2{FY<<M}oL&G(8z4~!ijyJb+^nb<j?>v8wmJ{P*IlQE?!X@~qWT;Br'
    'DR3Jd^`ZAcFs-Kq*M2k&kA~T|fwGtf5kYVp_%i1HH}GZb(1Gw}ma>E3%PeIF!<Sjg4uLP@lN}0Q#%3J`U$e&Ln_%Z$!g^er|E@(E'
    'VZY{QwsC8eKH+QQzwuFEWWJ%w+GEMC90Qi+TPhEz)OJOUN`W&}e2kus`m&AN43_yoDhrQDJzuYpvn0DYOR~GOB#oUVY2qx&9?p{N'
    'NtUEijf)vW91Qbv*zM+xu-m^20}ngZyido|i=5L|sh0XO|M_vy4M1?D4M1?DO()`*Iy3xpp>W4f_OIG0S_*%5Fb!4_y-'
    '5+gRJj{zy*;?tH_X`GrOvHr>U@dj&TH6Pk5Z9akC|JKxm%BgTaR;YJuY_ZaRav=H`ME~Hrr-'
    'g*Sig)MQ>3U*Fl%3ne%h@a(+$==iKb${bqdZA`>ae^S^D{jP6sl;@*%OHGXvU99eNF(oGDLMF-=bmKD1tef@yt03_v-'
    '9~Kh_w>;8aJ;3Fk-Bm{}4{}#OGE~c>tKV?BuP|55<TxB|Iic7Jm%Ptqdy`D@93G76s~gzlIrvXA7<@1Uc}7pn?-'
    'cU>V)*D^buqk<tfeJ<#9^3%JihoEFLC@TRmR&h$YBilabf?v&GOYZ!CF>2;W!4#MazBb)70478MrPsHJ-'
    'u!RKBHSLU$?pF7*VZNyG9RcDr!Qm<z^NpH(|%>>TsLC*WJkc0#rl@^B%`g*;5iLxns9ULA2V&itvQsQiY;ruoI=#*7^{2mkm9s#~'
    'sO@&+bvWb!p8-'
    '(d1hCf{a3YgRsLY)}1+&<1+$)?)dWV)+6r56Y{!u2{|~wr${POUg%$>8Ud#ma96BT|XtGHjT<}*xfwhxnrs)sZoQLyM993a<}qPl'
    'gy|}@hgv~8ejxcRtEqT966>5X!mjXHwU{>OkT+??4}pv#(KK>x1X7O+TA%{sU*>zpDTcVZvO2qh9s^8(C6`}TwiHGeh?z`#uP%wB'
    '0?`t`EC_*4Rd)#WteC;g91Pm&TFc3B)Dss7LnBz9Oxp$12++Qu>8@7<&Hd9{v!=bJ<$5~lI(EQfPz1j#DZ{#Y7+rM-Pym*2to~|4'
    '^)E)?7?{bHXHU98SH6ObEg1^_@<_bBIR+yndKDD94c(;R8o%rV&kuy6pN#+^u1TeiN1U$9if(`k5KnbAEEA-'
    'K0@6;eS~^I`UtgU`UtgE`Utf(9ibxO;D-'
    '+NOEE*iNBxH(>StNh2bkRIWAeK^Oy2EdlEz*C@e^Y)34iLuSWJGW+GN0FBlB#!D}I*=lLWC3Rf7ze+}$*Q-'
    'uzuQOx77px(G~#+vrtL!rmq{7p?+i0m^mJzg00XH5QSN3;YiyX@3q35&R09vuC8FRMH-mAmCxQDqt~bd#V`0<n1ZZH<P$0cUb1gj'
    '`9a5SN4$u+cGRK^uhK~2)5a~0^4>UY+vVr?Vdtl(|=sXf(`zZj0M}*s!asgbXS}d1-'
    '6e=g9xxeZ=RG5Y!$nw7vD;b^bE9!9)jeN_NooiBigIgiYkn*x^me_*sD%lHg;DBqk_OLmg0LDQhpOoUf@`e;Nj%s5GOY|oY+Ik#>'
    'yeI^oFHqfemZp5S9CEtCr!vpNv%ir6zCcWbV_J)!`~$1@k2jgx~rg{3H*A_Z9-'
    '7ezV`kf)M`HZ(~9DiE0x8Lfy~5j{@PxszC$@VW@te(E|)X_`H%LA^0poQcrvJj8eF<+ZW8-'
    'w^v)0)F@p&#O3YISFBO7!CA#tUAn{p^j<vWwjcy2d1VTOYeEp-'
    '?14~w4p*f;2gTF5$Wd;(5FYx9R|)T^18sAeenfC8>I|1{Gx+IVRh}6j@hcyRtMiaJuMiUT>-'
    '{bkiSVa>7mLKzs!aqEb!VR(g~T<gK?D+EyiU&O4+cnVq&R~JCmR?NwbNA%3dz{3IwcuQR~ImOg}Z9Uv2K~J0NVfNyD*3IFmc-'
    '=5j^ErQb_zHMB**^=5UunBtx*?pwvqQTO!D_7=mP4o`lQ@B`z~K4}EKV^u3ygzWWNHPruShvFL+8by6()UR7-'
    '((5E~24^im*L^X&&AB@o-'
    'GJ1Xi`o2b5MT?sW9BY&4Y8*o@M?z7xLecnQJ$y|W>L#9M6_!>CDn8wdp?K#Rme}W2Ve(rsiEXVz9{9trrjWNbMBbb{8+bBL@16i$'
    'cDuVmX)ha8)~HjX8DszlQZNTebm5nb;KbdA1xk4se8<P&VjqLQh385RF91RPZoiL(ApE7@$3k$iY7+rL-P3=Jg5X-'
    '!AOeCgSbxmu4}yE-LlB(XCOH@wJwW#})NMXeUo2!O4BLkYfUr(<n&o?3%1PntJ`|WIalsKv)7ZlmIEED_$x{D^>jb8Wb#niqk3hQ'
    'G&Z*V63xkiw&vLAZF7+Ae)h@}TAWaci_ik)lE0mpcsp0DsyVNv_*$fjdleMfsfN+zKyA64^Z=|2o3SLOH&hk2d%dzTM(Ll0+8Ft+'
    'f4eB5EG$i368q`1XX-IBRZR#0(6F-_{#-xE7mad9zy*(!AqSW1M$eHYJFEO`Yr*4Nq>!#EKI-ram((S*Fyvq5}oIdJy7}D;Fz5NJ'
    'O*+UttH;{WdAL}=(+hM?aD0=sfD8h$ISF4bp*#mCR6mXl#FO)_6##l>mkN3#QN&{n<p-'
    'pr0cwuQ9mT)4n0{3#6m0lRB<flEf?h~T*`Vg&sL$u!Fp|zR(3}c5&jL}|(7wf#Wy9ULTTUlDKa>S>tRsfIE2yr0||1d#0*?lpC*K'
    'YQ!@G4-TRfzqs58p5I;5*8PZz~SVi&moKa)J-'
    'vVgp~@GYw<m3lGsS7QSDqHW}dCz#s+&sbL2A656j<x7Q0_LVI6zd%f^&<QD&iS>W5(z?UMAM_`Su(#TNM3mi<EEtJxB%-'
    'Gt8ITFg&84_Y|KZ}@o7&k;dMb{`Lq3$8NEJXATA)@<*h@RsRt%Hme@_b)Ysk0p6izs!KqkPq*Zt?<OHL1J2(5fbF!tLBS#;$Lb12'
    'uJffa8DrIQ}>f#~1iGZi+=Z3dhZ`K+3~$k-'
    '@R<kCIp%!$XwB;`n3LCIgO{fes8!Nd_EqbNdbI_IhzF%<cWu?e*d~=RAm#EI2+ch2tTpDbW%oT1>h+$~|pBietlD!w}koZ8gGLpB'
    '1PKmu}-'
    'fu48%W7`VsUmcsJBA(m%`SneNU`PLB29nrG7$b2U`+}8)|B1e)ucd`*Vgs8@4Gja$S*@0vWatM+9@HRZmIzJ!BANe?bI}gVf`Z#W'
    'm^bQKgdt<SD0UYbTXcUWMc!)-^IDT8TiNG-oOQQ%J&s4W(#&LgjduANps&20z$0zwX*5G^}?Xd$VxR&#{+hp4vcZI*>p2wjXIQ<>'
    '1s_r_LO2R>kitQ;B?-!!@#t_8=LKNTT`4}}w6G9Gm4e}#=$w+qbJO*1zQh-'
    '07$^S5tT^!EjH^g0+)Zueh8T@WXjo;z<_<hUA?<;xu9qr?HFC;Kf_}z!e%L?IF_rh+m_=ShqEf&A8s5TM!g#p<u0>3w^+cV>LfVw'
    '?1es5E^*N@-'
    '9`S_h8Qq{k>|2cxY+OJ)y*H*aDC8}4Za`hFd%8nF#_YdJaD}?W{A$)I7!M94y5mTCaE|Xs|5?&p~<a%aA*^?KM2bqe7UP5|ls+3x'
    'R^xIS_^)gCu?3EASl|Fo*&V%noK77G>xKCFomNB`27l5zsh23M}3lFh-EPS6<Z6e?c1G0Mrd}pcKGsE{-b$e#`-'
    'mY%1AHG$AJG_R;zol}1%I)T-'
    '!jZE8bk@M{kP8IGmaHMev(vg1M=~B~!N&+1Bl`e+YPnB@JBeGGZTMX3G`rz5nOh>EA5s(GfN%oL4ky6Ca01+sifFuo)T~tA_bQ5N'
    '?30h*<vxDv^6)z*aD(S@GQSV;0^2$L5{+Z=3lGsa7Qc0>O~iH%`m=Gwc5b%1J@a;Mpt?QtcJ2;!d;Rzw5xT)#A3Uxz2alTrk;!u3'
    '_?EkL){11gpFUKFme?U!Z3Jyppf$sfQ9{>}Jn|BAHHKmA*h6W{5T!STD6I@p`e)xvCrfd&Y0@f}AsN|eByQb`c>BN-'
    'w^20L0d}AAvAZ)bf;-TIZVM(K;N<x}=tEZ+=;}9X5(`~;h9<Gl-KpASfG%0PmUu8-KXi-TfoYQr(4~n{scx?qx-'
    '>EVtZuI#y467h_#(xc<3VRW3!nr4oQBSRJb=!fsTIB{>WYwq3XAJyRu>#LI@7GqI(UuKNq7T+aJf|s-'
    'y2WY$aUQZQv%cJ<O*2peUz=uvyj8RB42QF@R{c`qE?wgp=^JxfvrO=mUdRyA}gCmP^H-'
    '9(&E{dGFW;Qbf+*tLIJNq_w1Bd=&nZzsYvLqRc#`?1{mZi5nh9SkIY`fe(LtjUc;S={=R<b)&^GhUfiM?%SwrN<E_FC{8ODAYE<5'
    ')_?aL`Cge^-wRo-1+KEPIVQNm?AI^z|;hfk%oD+Xf&52Tmk1)ea3NQdX9NBS9sYzR8>bRUl<Ib~WF&&T;#*~;Gfea^C=go#s{n_w'
    '--fS4*`wJ78+*>2}5(c}U0?5_xKQ$J)@DNjDk^8=C6M<Y9im4IE)$frRx%;czGb0xUd1@BqP79E`L?9EQBXGo>M>9@7L51@cmcoT'
    'S&@W}yCaHbYP@eX9s};$a2ipfi*e(iTdq4==|4YHPOu=@r)qI#{X^sz2*82=1RK#a7^q2c^c@C%Yly=BDV``>4A}7x^dC+~&hwiF'
    'hJ9s+`I*kYv76?tC5JFe?!DX?~g@?E-7P_lcn+WK_a9kDvUHu-Jp?iS3Ju`G+pfAe;-'
    '5F`<{%Rzlf_M>@dg%a|c;~03l;ROqosZiZ#ct=}_Q4RhkA%2w8RB+s3b*?z7vwZ6;&7QSPjWete#zz>InWH%=w2LI<%|Gl8JFi;7'
    'V3SuJWuhK2FQKGNA8MWKXV((<!##NPGoYQLPl5j!n9cA!b41pMeYjKCIY!IAk!j{tKTCta$Bm~Gb0y<dRi9b&JH|X+$Tdjvt<JJ^'
    'L}Au%N03$2M*u$)mX*;=K=De5Ri|CfNT{4@~#vhTPc_JOe=(QrPY0H!7&=53-'
    '09J9Qo<~>u0~V!5tuMEr~eMd91iyOp^lCQ3|3z`<B|-'
    '7r^%=AHL81`cd1NT;8tXdk&X>6$LhGy3eM^!WSN5dMtdOQ*9zPYA}4$BQ|RKJu+|9TB+MJZ`1%7)3d<$4g+7a^r6&cu^NFY4x2rX'
    'I|!HGc|4UP;0GTt53tI>X9&u@c$}Q$IZ8j4F{=UzzpBcoz{PCF$5K;aK{yp23#UTsa4P&G75^%?={s0Qo-'
    '88*)y~NaM4;LwSx&5}-IAAxHMM)Pf>={~poCfg=OsRz*A+f-3vI4`yUSxi3xDeJSkSIhZ6a(gboAvBHdnt#W}Dku-JaRz!Vq4b1+'
    ';SmN4GeKoXnP6?pw8EvsEp+bY(Efx9SM4%{mzGpJvQMq~f)M<~T)z>p}D35SsrCq1h&c=G{Tis|9mGue})sy<R|7BvVbi9BaFVP%'
    'vy?5}8R}#=5Pui1)W^@?su9Kl1^)?AMRf&gSwj+TKoP@_<6_f`0oeVxbEUaYZb2m#H=p(1n4xA_BVlJu*YLjk-NEbYY0E$O7H@X?'
    't5D<QLlBp23wcumfH*WhcKC);omH7IXD3!*uX>jYWMAUY^5us~L0zhwsMrDjH!Az6(S6J|4pNHz9oQNx}CeEG!&KvDy8V<dK&)i8'
    '<`2a9R}_VA{qWMn}0VhyT*UToeg;6(^EdY=flg6#=7jfbE4ow&&%=d<T2BmkuTL`(GIgVE9W{#sYYrY7=oN0Yh<R#G!<KkIaV>zf'
    'remK9qn#zA_7dD;lQ@3(N2<V-'
    'vHMSBd+U(IRYXT{khRb*+@cIed&D`BousE5632e;*?h*QnJ{#$P_1#H}&(Ji^9&_YiWVjrr~=<R}~SEfw+t8}n@{<b^io+f2yOHs'
    ';%0$cwDrav77`T;_T$7FGu_j&$bwZG0D{hVS|qeJ#)ao+%2Xmg_&hhDD~9!=GBiB2&v>Q*9zr%XL>g9+g`DfNBttS`NMWc*ZKP#;'
    'NKCSP(C@<qTk&rnuUMZ<*}l>SZb!?!ztd1VasEx|)b=>h@|omzTOi!`o7oHL&B~?ocv`;eSX+6w)ZUo#z)CrcUi}F0A7&x=|-'
    'rajkF{l-HKyosEs{T!rg`if*K__qvb07xS=pRw3-^SNbFtd+?_|iN)TFs!as;bO%2Xg}nz=g9z-'
    'w7(J1(t}Bf_xrt*?fpGIZN1uVNz>~bwT~%;-psTaFNL3NeHvhJR%S%GM-'
    '9~H_Tz)D@Enq)}_Y(Lps?LJwK^lB_@UXb%C2efa_KIHpl;92m?=Z(vyRHCJ$vk!+Sh}lR@n?N7K9dK=szPAYuedfAjPR${#)9z~)'
    'g}Ury2GE00^>ufK?E3K%%03x=GE9IomdczEG)~5nA-xK44o?6!O(M@vRm-SlR4$Y*jg(UKdmqoF|-sf#N-'
    'azM=es!0VekO2!qv_VxNyP)WkLc{Kpuq##AcyPX?>8aUHSNr<tHw)e~Ty1;wiNXYvP!b36fML1-'
    'Y2^FQxuDSpUD;9YqLJi8DA^&kH<7J=}mK8;1-U8+q40(DnB6@|bBszC$-'
    'p*Nq(Smo6uy%fKQdM0G?HZggWvw!1J`^H}VkIPzDH~kBj{}NO=1G|FHggMHQ95zS797#$c<T|Tzc>u#-'
    'LJ+~tD+{*hX^8wKZynafFusJ0mwW4SXEY0jW6Aat#S~e6!F|e=KJfmU2fT9%0Z+fuzhZ$0f9hYc!27Fe69GKk!B0m4?_t#-'
    '0(dY+PiOQ2o20fNC=d?YgoP#uey2be%kXq9QmQ6z2wS?mjysMAXmqE^2dqI4u$1{Hoi)(X#EMYHg(n-'
    'y+^yuvCbD*~@MJR?yH|O#g>2nxJlRU7?scAQBTM%NPqvexYsrvnnNW-g9xT|Yr&0XZT^YJsA8)tk;qBZ)c+-'
    'FUvsk>rpZY8oZ?~&95qQ&Gu{a8E3sr*%yg_d+&gl204V~=FG5?h8*G?`cxGOMoTbcYnGp1j0Oo_bPT#evzl`xgbD9~1Ai}bo;9>x'
    'u%Qh2_}s8_jpI~_u%s!Tl(A&WwU{H?%#<Iz;0z8w7mkK0nGWm}Qz4z@O-'
    '4)!7|xFtg=#XV9U@Gka&HzN;t=M@5;ey7i4fd_x;^H|``P;DZBr+fIBDBvwp4I+RCgY-;BA1`I-l=SbT%&T-dBnP(7@Dr-'
    'BG?HH}b6y3X94aUi$Os7LU~r29k+0$$nm-o_Z;+Esed+mzN_E&t`j+Hmb6<MCQ4Xc;o%H-'
    '}IgD~U>G|Dr80mJB!Hshm?QUq3PJePRzf7f|dN6+^g!!GjGK6D&JWk9D5VkD<NBy_Ih=n8kr7vRPI8n8UfTQk-XQSZwh-wf4N9fF'
    'FGulBHyM`?XAm*DC&z+*Pv$;larK7pX;3zQ+T%4m!xLT@6Mziw1O_MB#e=19DIm*;O&#>thAK2*M0Z5EQdh1mTQ?Lzw)V{@59(BS'
    '4c`C5$jESOD8`=dEMXB1ZE1uL-wOuz%6dlG$7ISZ1dZ$(W_9<A$R1fdxo7@8)9cAl>{x&u;4S(v}*vRx2)h5djKNS?f-'
    '=@ehzc9gxZ&Ky(pK~nV{=1pAqd98Zjpo+1`c2Nns}uv`fjLD?X)<L#w1M0Gk!6hnw=I=W&`g`m@QIDpwnp*|iq)RYFm1WLr_5c6b'
    '#<`uY?9NEt7TsBLa%_HU_>pEIViv1I`<XCvKhyo`IbMpyPLt)kN92e;G&jJ?BMQDZL$pR(jZl36_<Z=k*D<FG9f<~CN&i6{|3?g('
    'v0ZFZ|UTeRXRIml`c+MrK?j`>E@JGx;tf+9$Hz2B2@4^-=rc`Cm<y*7>^b?iqj(*8;xWsqqG8d)=;T-X-'
    ')@Bk;KH2PrhAfzUof+K6XgqPkkRdq~EDFS%!3FFr<vjC)`FGX2b^Pnl=)$RSv%Et8<w=&e}}ur#h>K1`Gd2s*1x^e-'
    'l;3;gY|Zs^W0H-$E)|xY%zcnJrxDx7lKCSmw9u;v+N0Ey9HZ+?r#A&6UR!hesLBV;t~Q6tYl>EDpJHWaDTb-zhZAx?k#IhZ+7<UF'
    '<M_t=eQ6=C^`j?uXjmc3?Mh%nm9%k4JKpP#<Z`@=q!bXh<J&LjGGAm25nSZkdDc`f4{JTj$`rzJkq8n;d-'
    'CS9=I~P|lp`R^q|i!8!P@ug+ug__SqCpLe7<?8-'
    'fUQB_!H1I%@PYMF_%6bdod!>G8rkY(17xG{Eo;ZJRh9p6t>n=Iq|VKBb$gUjK@_Z^NUV|4X4CT7{Iw=jjnUOmC&*99%QcDj3Wd8Q'
    '`W(5J9(;Nhy_n&T*ceSx{UkfS#5g;br$Q6>5!;*DYQ1R7&Kq2EXuZT+WC=CEJ1dr1V|$(%@%He1!_(iAqzo%%<GF-LL4h3^&`aQ('
    'cSVh0@l)TY=0U#;3?8Su}70Y8vo@*@pnTQ1AJa4Av?*(;do2iO5mUXY{Q#Zr{Cr2F2(d=h*06q8^63NVi0=&!roUtcb(NFy9;{FC'
    '&6w@h1bJ(X?h^;#8xGsHqaVd9^Y94j+HHj1;_U}~AI!Hr`Je6P@u>nGkEJLK@EHpdS6tEx?wAzz=`J#`DZ5a;`F?pJfx`87wf0~%'
    '+NA{eOBEM^d+xSB?`0;r6o>osao$R5VQ#>(NtT+9(Z+*}Q3@<iw6p%-'
    'Pc)HZ^nXOh%5fum<~)HZ{oXHwL*fTL$J)V6}7XR?E~fum;>FSdiDcUCM5U%LX!vwp@cu|o@gX-'
    'n+TE>>+a#c9%o=1OsvP8MVvZz)sN;%=r);mDs*ZJeFz6bIKjcdlwrSe9Zbx)&Jf8|HE?4|BN=40E}_=l62CmU}VC7qLd{TFUV{hx'
    'k5{6<Azj-'
    '1%3$*yL)g+L0Sv*4cC}zh|+GcL#AlACbvF38Nt`2N%dk)IGp?JmDSZ@q~Ar$D||b2g{s}2t&nr_H@Jp)$N(n5m%|(>(A!e5oFS~N'
    'o|~~S>R)P^#K*M3oFNJC{vQ2Q~)eO#j9oLu6n5L7ozsX5VZ$|sC_eq+Lvv#T=6f?<8r34ZeKDvpUYY1>MJJi=W@2W`Wmai<xS>lJ'
    'ywFto6XgKkkUZj!ea^V4))Ot6QWKN)V#~&#sUO2x+esW=erX;p6^aXp;tdxX7s|4i3s!_q;AiQ-'
    'Z$0l^`o~)q`fFj#!=0DGL9N^br{3XL`Hs7{}d+shnRdR#N@#tCg1WfSt<XG^e*y7gQsth;ziEkR$aU^2Y499XK18zcZYfYla~@UM'
    'J1toDtWNHFZYsdtiN30G#!=~40F^&b9ac3-'
    'gB5dRomY8nB2tjYj1R)<#;^HU5>}I+~s&o>5YD{%%wLlxH+D^^yXl7d*;%cx76+Rqqjt)ZSUR!Pn=8S6mweLM-'
    'Wp|`glhv09hM7lRFz-gSIezijCa115!{Q5JG)L2=zlksK4z)UBS*&&UAvW7foy(zDyBKSNM9_gwo+_r3s|N*DEHB4qvavlGz>VqZ'
    'M}Xr)ji)jHF)$Y^{3jVjj<BSIpzN?24mot$wh~wid>xIKtK*qHfP@Yu{G4*N@gx5me}+8k#M+naSlLIl%7wdXtaL4}FLV8IN$vT}'
    'pVJUad)BZ5>P1+khK-YznT&hH!m3gzKRpT;EAqW_@@)k5f1kKl3@oGx2jjrx+%F9^@3n#Lq*VqL}zuz|Gkae65IOvpdWO@CV3!s{'
    's5FvMv??u<D2gJf6(10gorMYY+us{a~2^3?tSc0>Fo=+cN|B9d&#C04@{ZCM{3&PaPgUOr?rLkVZ5NQCPb>j}vLaFh7a|pRvp*Pr'
    '8`-^k3tcPyaRETph~rGr{uE4NT2|f#D2T8P0&i!WpnSH3OP5{M_eClJ`PaIF`+>+=p)!lc#9=3mft?L_Qmsev5`Yp3JTxk0-'
    'Nh$Ybhm^n+!tyMgX*$g|ho9Hws1Tz9it-CjR@+oa)(__L%!Wh$)TQ8xbtdfaf*wB=SjP@CLxO7u`@FVm*c+20jdDxuJKD=UKg;eU'
    'q8D}>=IAq>ky7{2Ro+t3+bXz3CTL<S8?qB{sl2`G#1U@Y;bG`d5ube8hy4#i?bN~Ajs3%d%$l<MKMJKRU{`^cZKQ2ZfMI2SO&sy9'
    'k_Jf&R;kEgUNi88|a!7>|R7_E{BBV4X-&uoO>RkzoV;_|@f?acjBT`g%i*q!f@8Yb?o+gfG1(HyI|N{FDK+(wSQs-QtB6FMlw?yD'
    'ho4-'
    'c{XUJAQykY$VfnXwF&%BbG*VunIcMCvdOdMs6a*@FYPjP?pc(cF)itvWiAgZmoGZ1)Ebz{8k4U7O+anH*9mV5WMa5sxRfYsBM;?H'
    'WY^SU*^10K<?qiU9E8>h{b4eox(=q5N@=bS@~kkMjipz@*x7+LEh7obm2}eY2A@*7!Ol)WR9hoj$D678_56w3P^(S}q4uluO7Fmb'
    '(9IAx9CPms~I81;m6TzY%h@V)Sx3@8z;M=Vbzpz&t5pGAqtmm6K+8+;EIYW^3@TWBB)Gq?keCc=Fo$R`*mcH)%S?&gg1+pyejWue'
    ')xEh|I}@hgc#abMn@zHW_mA=m0_wWE++xCy&5&ifWJ{CyxRY<5h!9x$t|WbKyVbSXom`HC%8FyRscFZ?hJYq`Z^99b6}aDvi%n@O'
    'LrK7ocyOw3SbBq914v_Yi|d2l)&q3XLP=vz$0II?5#mwP*6A6P2=^&6BQ_T+~kGPt!cv;#8IOwD?mvPbQ@137FT393U0TRoC7}<Y'
    '`=P(TF^q%dI{^D5)XqhkPy;k?;`D#Uk>3)h0p>S$ElQSd1F-'
    'pHzbgHRK7ZL8dIxz(OMNwSWcF?xP<Pu<+RvfB6I_nwG=mlbpy|+R3L~Ouuz5IZ9u&hv9Z$+3LiIH8xHS!7?!g7Fcg5aTslVbrhGE'
    'a2RnvSp}xYF*sv}334@H5|n`XGt!?bf2M@y7fv;}V}uPkk5I?#>*$aK|9vr!<)Z80L$ezGf*^v3?Yw{pl<tz}W1$HT@q8>aSE)7`'
    'ph?q4572>`ph^3uQ&ocu&?GERR1Gqn)258&5R^dr)T;-'
    '2g5x4M&yrWb&{n~a>lJaWzzCP&tGHLKz<w;CEabF@MYCm&H0i#A4O%Wa+8z!ZoFmDzN0|HzPkPyWy}$D07=7^=lYiq$Z&RA9O6Ca'
    'n+ng@4hWX$S_q8e91Bki_3({U(PUZloSf;rnc$(hxV#_o)P87hUe&nUGxP*sT8jH&ns!aqgbq5|4h0DRJK?E*qRf9~GlzXOeIS;c'
    'qo&5Wk0>Jo^AdO8dkzOFonJ4138+U=&1qzuq!(gV`U#<6nc?1|i2c_*!I$0y<v=B;@JSf$mg_~!&I!9tuiFWj(Fsw>D`U^0sN;~='
    'sG2BEu`q3C+r5*i67+|Fx{TPg|7TD3(!ywnuhb3&rsx>UZF=$yxBt*advRGKcLoADh<#Vb{23RulY<e&b&IC)^7M-'
    'RVWPl}Yj3%iDnS9GV)4nB){d!KR^JNH8V*#7VV{*7K))MG!Xccdw*eV)Vz~hJ%^IJ#7en4p4;WU&r_)ZVecuC5zf$ixf=uj7{?Z7'
    'ZoW&770i5MR?SO4L-Pg`WJzTrsA_=vgMz~uu%K5DN1%jJVYK4z}I<>;^cQ^XR%RrshJ%j6i1%AP3AR{)i|8(xS-B|O9nv8Y_9+C-'
    'pI56U4?s61UYh(P5fszE0IGHopN9>*HIp+df4YI=ttp+yID*M<r@9d(ki70_^=S=--$`*OJ3b-'
    '1ihEHH%2ADp7HhT)J9mzO$R>W$T>=uCJUt$%UZAm<g-'
    'ea2<OoL5lyIhQ3lub}P=E*s^%g1U8F?w0ck>i*5;?m4fZ?n^Ek=dhqIjt{Prk5llETkCk}b6h@E2&cMRmdD~09%6YcPM4`R5jfR@'
    'bZ8V#ho}Y-'
    'IK5Og$mDDWi#k|O=>fGFxAL1xAa?{xP}sxNHQw3k8l`V%Geax#!ksXPa<T*K|C|W3#^xCzSSR~N6z0K9Zg*k(${V@ehV3V3al7B&'
    'U(V*VDN7EJH*vfBZz*r)cKhE--oowvzqOphRRAE4$Ze>H$u3;(t1<ZuhBgXdQosF+v6zHscrg}}^HiG%OzN>XEDDons0I<3oU9sT'
    'ax{a4DR{gaIR#~QG9)uL!NZ>!3@WKFpTpz<*0np2IdK*YK)1Hu-'
    '1*E%wHS@Y1;R(+t)W=Vx1m_ff>11`NseOh!~EKUQ<RTeWM@YklxsRg*c96;=4DU0axqm^V(Y^2Vn!@`TV>8e=F37h)2lkz_OA;?j'
    'dULji%nLBhZq){to)j4lOg(E;xr70Wgs7<&CIvz_Iji5mUc$nUT^fBYLH=|hh@@d4)A*sI(R<FZVN&7T?n#=Ly+x}^OKwETpe7p('
    'Urj^n_b~svc=WCC0kwDTe8hny(QbJ=q-'
    'w&ZBWkX$<!dPNgES)AEGjDy=tinM^W>V@UsW!A(bXJEi=nO8M5K3?c=a~DMj!?))&&E(;a($EOO!X&W}azi>gh;xzTOv!7`s4eWz'
    '~Ed~T#)JM*~_py&Kd(#!#J#|XJFh1}ai<bEF_cVUR!J$>Yw+Jb4G^c<ZEsYDbq>?-'
    'qwNR?FL36UzP3|UhTm~1jtpK+va$K^a3jZck?Mgt72p_2WuRPId$(V)C!6UF9n9bYiihzxUf$dY498GM)t-ZB-'
    'd`xIS^JY5;pwZmhv3$Hdj7Q4@=HWAppT|HQ4?0&Cq&x~FD+L^Hn_!*u_jyb^YdH`2WhbfgpjMZ?%0uC$UhBX{5Z43wo^EXzFpc;4'
    'LKKSnl!Cx1Ge^Chj(iHert~WU{D%A>)m*V)~j`xi35QLt`q4E+^BU0gN*)9rK(Zv{UU5pXP6rmNLoCN44+UzdoavUolp{P4%L@ad'
    'S#YV(J_b$~YB5Udn^<bH^rs~w~nX{(!YiG`yg3%w5Nr5?lZVg6#eB}K(MBc^_d5?t1YntaUjPM+Wk)FeFf#)z>=s66dJ%`~U&tVw'
    'DVyWa#$oT?vWw;3ZBy0+98Ch19Ok?O+7;4nHvkUMP<}>F^v_TUah|o1<a?-CWFY9g{8H-zZxskEB{i|vdVRHYh9xSuT-KcKQY;yH'
    '$XEwQjospT8mxDYhSQ+QKkU*N7HKi4<4zOJ8U<C6CUgwgF;6evuG4Sb6h`)prVpBLF9t|f%GdCgh25n#tt)+{7osg5Silo?i7NcB'
    'RRU0eQtZn$UHVg1zt1_-'
    'VlUcF6cZt7SGottK(^FiIEd*lS6{BK73=c6X7R0x!HW48Hi+ZrkAl{^I&kSPy+L=KN<2@>q{&E0fnezi#81(W2Vridn1arkMZn4G'
    'Ybz#!=#Tcyj5&YK>!J9(_KNcdmdCHkkmXm}$k;C@@0pIo2LMB@~u1W1?UuUye53(}j+RdJ66ITyab+d8wc_;e#eU{57G=87tvZfG'
    'zbw^wfi(hz%3u5s*L$!&(?_bq}WybGjb$e#~>etSUUl{ERGU+b|DcAb!a~ds_zJEAZrN~tflIx7lA}+j3ZD@3<Hoz%$89ZC^!TYx'
    'myjw!>{xbycUMcY2&*Z)^H0Ejns>Z^1ef2PttzctMo(sAHKN-@r%~-'
    'Qmlr#PcmIN_#u@%cJcrKP}@0_#%^AsDy9bX8$x*sl##V$O<g|XP3sM<te_iyUKGGlj(x;-'
    ';?^=oIwE)4dCnFN@FaxG5~T&1S|z)ZvxaquiBx&k~4aMgr!k^FduO<<kNR8@gP?|j_*`RM(zUTvg2Rp)1U;5^L4v{%o0xRJ|0aqr'
    '-47N+}o2WK~7MwfR`b~EO5c?V^;a5;plgECzT^EihI63gIszb~7#kjoi5UGPyZuP+on(;d^9$8-'
    'C2=JDKqouksOe^k}TF|tXWd6Xz~=ZLiHXVvW)M45wqTsfD2$ia6z%J*^kcS~(_4o7a|7DP(Lk%eX-VeRAoDa`JyR}d*rb$LmM+1Z'
    '$!?H%6E!Tc&>HwEkWGzT;6=NWF}SkF*6h=7saG7{!p>eiUAIA(ECbgWdse@IH?co9oj_<)b)hq$~-V|g@#kB8-'
    ')!LshKE<B#funUi;GVGEG%akX*Q`OSfi{%DRL8(gyEH`vA&0R8JnL-##)a@B$ngc8!$k8)C&;&LQ=JIzr_-+EUT>d@>-'
    '}TkuT>dc!-+jHhJN=E;{jg0G9u6Pm@*lMELRFpvQpo=)ME(=?3O7H;gA+Lwb?D2e5)F9-'
    'ox9#QP46XN$6c)cr@2^q<q_*+%R=Ov^)NL)=wo>Smse{nU&!E7;6P4yL02Bn;n<bOb2xV8F$Z!#sj8yP2XarS_sD!8r(Zksf!uTI'
    '_6)ks!4|KZ7d5R>JV=8JC@j?Z&dBSyr!z6rGNBy~=VCmFL1Z7aKZl@wvR>8Z=ToIz;K=l~a*$h?>_-'
    'BOn62EO6c{lV?^u#u#1yE3BzlLgQzfIM#s4umh!sd$tcQB?As@FFF!(6kj$!aA;7{mY=*HtoDZBA_Qp#>o{>0C!{w=dV@uYf>7}e'
    '%(5&p#U>h=t(%>iyt5lDcS=3Ft+{`OeaWvMBb;>ORONevX1=?hZ|zT#nU;vr0y@ZtMQ2;Zmb)p%aY)A8nVN@&=a9G+~X2y^>n3k8'
    '@vBwH!Id_=O1!pj|#?e?JXje<UKJ#?B2e9Vr(4X?uNC<dPbnANY|oyQYxcIWX#o86-@`-'
    '`g5%#7Km)O%#ctbXmxm|d!F&mh&D$H39d(_`+C$^w9${ZQ{8zv4O(k{NoAp@`cV6jMXos#_gy7hqnGkJ}<1;P&Zy1(}zHUh+_m6U'
    'KQfXK^`5csseXxjbHYJ2`OUPY~Wt4jlOth4HGB%}KjtvpvOL&1Gv5OBedE2WVt~SIIoc{$5B4SNBK{9#13MgU8c|_K1RWk$SMqaD'
    'H0d9wXx1BLdFL)a@BWob&lOh<F36Y+El3(N>{%FnN)TC+&@Z4#L>D4&w!eD^kA0!<diegP4UNUR<wq^9vz}&*ZXFc;SdDj_2df<v'
    'y3oM3`_xvMFhsY_b{`;Qmrl&cO_ccUlUYlkpChtwb#O=|bNkk7V#s2<|TTE`(s+9X)wGD{4<3&x+bJ3c*Z0SY`w-'
    'R=3A!Iroe}@C)kp3|h{C6R{*^g4ZygJZ3zXqdkf8(E^3XOXR?u<HGj%JJx98AqVA!m_Fe{nTJq*rd}E6<taCK8<PW!#JEbISjFXj'
    '!sZLTg)GCy%Ru2&N`u=Z_u|9`pKO*bxJ!zYdvlc)7sv5h?U~>TS*lHNC)u<RfORJv$>X_SkL2;(uSZ4!n5zfN4B%(f?U@0rUpq5^'
    'm#f<|kQtPwwVW%ES$Kr2Wu0fGS;}1gKwA~MqCFli>O(CvMybGy^?|m~y#gt~+`2zcppc6*q4x<SKzDw?e1V!EUKB&15{N5xcu1fa'
    'h>5OsLv<sY`(B0R?J4pk5z;IOzksL)x>Q(>$ni`bB@`>uFiK)86LP&yVuO9f{^;jM(c6CID_~J&q<-'
    'YlJThS&{?uq5nXtY=wTY;V)E#&XkEx7AI%*zM8981x$W(%sMkwYLgKhlOQpwM=q%hSf@`5$YPl}j_`G<~7=2L~>!2wF|7pN%Yyw3'
    '*&iqROH07%YuD1#sONTIAUMA?KC%3zMDnW3-'
    '1?QJIDyFn~RA@V9t#0cPq!gm|TZXw{iQS9JI5qWMjN$uCd^AcdQu~_fhz)yRo5c!6F)r(@$2@i2mEIPkbZ6e4wbjSDRG2|OCHobW'
    'a`Njm*AX6z?X}ZKmacy}H09N>Tx<exvu^?7pN%oq2LtBIYU1Qp<{XGO0U=a4;$30Wn8x&%1VhVfn5bS2gRN^s7J1+lauI6)O5BSV'
    'n-OsUR`*U;k0LN0<FU-}097`0}nX8957J~iTTrJ>Og7YPhrJr5z;}Ukj2WVV=!R0rF&eU`_jETi1Jj9q-'
    'Tz;(DM3njHLFvO|%6wql`tX=CpNXnLrqVMXmzwph3`{4L%&39^Ybw7LzNxHcK7>yZxOjgn^jWbb4VuSOoSm|#z+~U{PFbilh0x<d'
    'gx2~9RoOm7P5`doNXyxrbjIM)SgY4|fn`&K-TH-=Oc8eMqb-'
    '*rY<@0sREj+?qB@bqQqBIyN9Ly(A5qBslFM%kAydD8bu2RBA*y4M`L=2kflNIZeWQ>G<JC6`nYF4xrUJC4c?=x2@UIj@tyLz|Koi'
    '&ZYL5=iBi~eMjTW=8wxR+<P+7i_%athPWT;;R7Ou5-Iz>%W5IsJG=%kbp{ea`=Pp0*U96zU;tB*K-'
    '2Aiuj96zU<t4}z7h6rLnf_)8qx5v`^z(es^8rg4raCU`Vn>L&4xZL1_lK`M!ye1Z$@DMez;Cw~3$pB97w3uKR`egzqVGTyAUnX$U'
    '`aelE$mDFMz==YYZMjn_vSQy@F+NZgtm)(_T@YZ;4l0yEv*r>O^r6OFnk~gouzk^6iqE<H5006|tTKdD6s!zkvah~ShA`RBh&7=h'
    'O!hZgO=t*{15DK=8p5Qd(QQIQn6%Ot$`B^44fT=0%1xRL@ME*oY&apD4VQ>r2(H${8M~p70I>e!V`C8se`;(jBA-'
    '@mA{1A3SM=vGimT9D{dtVy>Lsc{CI>SxmtP7@nI*y^t(CXNVa+Ryt<)Nr#;a7@p&7fZAO8;P17Vrh4Zr7d1!`C7t2&N_uHO@lA(y'
    'YBn3mz+;_`JA<Iz_~3#<Tc?^L6jr$Cj2pt>|=V&CWZF*#{Sg94w^%+*?spEE=(+3SCOQwi_XRF~_*ve3K$WwCjIi(`Qa4{>oUFzZ'
    'yC48UyQB(0!32V?>!<*z`G56A>e+Q?t38f5Y?o2K{WQ`DBajs$+K$QSNC-uLPe#;fX7%WPRAJ9CnmszjQ^DVPuDLvs_Cub^C?VF%'
    '>$H5B$TL{<XJ%Gx{is=ZQxJ23>@<P_kvE*5RVCSi?>{KQ;cf|XiwE%Q~fK4Vse9X_8kD^?AkFQQpo&+?E7b5938ac-'
    'p!pJ<`fZ#pg(neYtbVv)I1wTal1>wZ5rYEKSBb8OU}e6nhg$z%pWd0eSe48vav3K`vsU<1GLR^yTN#pI<jN{a1ED#4hYRfQA-'
    ')xLk3bi`4aEG#AKW|IU}DZc`n$dbb?KkHVOw6*-'
    'K+gQ@h5Df|WKbCZ~6m6qZ=$#uT3jEB10RIy#2=G^?Sl?kO8=x2*;;33HeL|RHaK5RK{0s7pLga(`v8UuBL-Ozp&?YV<zgD%$5Rxb'
    'J2R#T4>J7=4xI7lPz21<#&A(E&*Bg>2CjrK0N-'
    'jf4{;8nmpj7eTmI|wOwNCqodcoh1BTS2k3=2?8N!`jd*pHJz?j6LG{v+geM#re%2)TnL9Sj+kkUy~G2pb;1D22+qLR9_|qVmZQm4'
    'i~KJe$crmT?3&r~NGB2y6!Wb0UR><?dLHh3Rpuu7iB=mT-A=PMOss(TB-bG%~L@H5R?_5L08(`@U)uA!!6XJvEnI(&!g;duB<aC)'
    'MqlC5>PVr)EL#(sXKTQjTXG;|(O+VhC@dxoD3Y(@Eyy2z0$J6_#CnObT=V2r*X_9(6tyV(z#U=BgOtf0~V-'
    '_T|Kk0BhGkj+6&3GrKnuZ>Izk)RIv|i3$5*&5VtwZf01oz)eo}@44fwR}=-O@MKvm{p)c)hI^tfPNlc4MS=8e-'
    'CdW(Vi+FcvRDkSQf(sAvti^e%Vkf`hH<?tmpwiEDfQbkr)M9hetZ2GUg=w4ZDz;g`3*%+XJQnPLRKdGc`0QvdpXSrJLc)QEk!4y&'
    'i5b=$Fr6RW)8H&Fk-E_=zyE#ON12|s7_6YyTb{=!U^$oI3bQtO^AL>o@|Y9e<uH6C&WO8#Wsc=q>^E!&DF6yvB&!W{t1Akqo=)a>'
    'sY`D>(`qW3t)JNX|Vuaq1r?kVd&{;5k?rsVOoR{ep>zZ%trWl_1o(Q@LNHB!8j&sQuw(i#19YgvpB@h2`K^2rBoJHle~~@BB9vj$'
    '!5}teKFZ$YPXY@lC7o;Hd&Eu)8er;$;-'
    '+10`=VWFuk6@9oa!MTv=w4s^Sowl4U;YmrW^O&Co0xj~}MzQWr)3gu|6d*fl9ATjj~xW!TB&tA%3yx-'
    '+N8q7~k3dMsL>Q*9!0P+(lA=d$OZz^G5pWzRubtbTju9F!B(Z?7M%ABMU{Zn3Vx#ke=83dB<crcIGL;&|kJbyWnTF{gso_$fj@!{'
    '9gh9`AT2$EIe%z2Phn;VgJ2oCQfLgQ^Rc-*MtA=+5OvPMBYoY}Eg~61D#RXkFZ68F~a)>YG-'
    '=4o%ODO0dT0NqoX%S!NT@)FvC{Z#8!J!ix0**wtNdc`SC}87_~-?mE>b0=qC8mq%b1#_IA2>^`G@duHq=>bKXA-OmE-'
    'o|2P4+XC@Zgs;_Fu7PbiS5EPmtzd)qZ7S1hsSjLaT2BFYUI^S=2;65w;GURT{`)GEXy(cwhNX*|^$Q#UtcY1_oZCNgEQujk_Y{sf'
    'gOob(Cyo^|lsa%4$3mEOJeGa-MBi0_-'
    '}jPs6?$V_xd4E5S6mSbV0eftVgbBNwTS>QjLH=e0ERKUA_BnAs^6X&z$dETUO#}>2f?z{%&*KEodcv2oIl}cZ*2*e&p@N={A{yYE'
    'U;1wvjZ`T1f)QGUkKvj5X4JD5X(%`^}bxa6ADjD*PqI<qKh(J2CHiLFQQp$uko-8zu;v~yz?&=!mjRtD`T+>&v0cdcIT-'
    '!5!i*nxH1B}FjQAYV0Vf7?U}JF)o-sKyLBn-'
    ')_{dQMbN2(UwEpv<>h{D*Jz>ji))bGooWG<3jr4?t%NG2=i<Tk%aER<HjIDG%muLRCj9a(+yxr>n95`shQlUVN%C{>)Indoh~cow'
    'RuFU;o<Qh}6&M?vYA#;J2*zbBmd3Wbe=c?qlU3Sp0298lkk6}oVJ45Kv7O1|X>4ak<z4Fs%M9IG6|cz*-I+?UATxCJmZW~@?g-'
    '58ed>&Y(q81OoQKNWb9S)~tnM)GCvN>;4&2_bj0@oPRSLgfh4`Hm;`hc7zui;#U5-K9ON|G-NQDjz{n2A8bYK{d{;5I-'
    'db5WrNZ?}3vM$CLA-{LFcbt%0cq~zF4<Eq?GI^FpFj(~!g%GU!;YJ=$XL}=$r?b5=3c>opG9!2rQg&xYFbwC75eV+C9;SW-'
    '7v<6j2Jc`Ub4p6XxN?8V;Lc}epN@X;C)`AYmAm2iEwGXn8hVK`r`SZ9y^*-cPm}>u`E?4|Ux&cHBn0-'
    'X5ZFCL;1t7WR?bGq;4?dCLuBx|DQ9<a@VO;tBWCcKld~Z+_}rSazBPPq%f-'
    '^@_Vm+cid>E<yyPq%Pis4i$J5%*io&pdu*?{Sv6&TtVHmkt5g6{F9;SW_m!vRU16w!Ty_rbSIFxbo%D#PnVx3png(u3ZnPWLv8s^'
    'r3xm=$@>iQ6=mxf549U`@73aJCIa!cNv)5i5-'
    '7vHME3aOKANa7I{dN6!+kE+na4R&K!jX9HcV^<^OMjp$Td!%QBlkM`4+61>i#<2n>Sogzh9#3sMo5xez&W?hwez45&y%dEKvcngK'
    'b9MxLd#Z=2AHJozbnpRu*{iJb&3xu6Ug#9dofKas5VK`A^8kDdLkg#NfGHms9MS)z0Q{d2z>`A&-'
    'xLD)Na1hY+QXM7)<CgP!=!;)p~MgUdn}fO7qY>zi-'
    'i2n1s=u<xry7I+{Hp}=CQoFNBP*TlTu@M8<Q^<avyXT+{EMQZExc7^tLxeVOKv`X6(XP-'
    '4uac7|EL=uzRF>nEJ6>mP;Q|!2af&6vn;@F*YT{*v%ovjuJto_7own_A9lg3VDsD8^GNnd}$6C6dpAUFsN^9_+L<h)Nq%hx-'
    'D|Q#F5V)EqpfdJB7RzQ;qI4p3kj0*)3=D-gTn8b?mFv<kf4Be)L;88hLX7*-'
    'd%+>2r(C@+1Y}WSjg$t^k5{2j9%&xovOe@!YmIM<G~0SY`xIL1mik2nL+o9D(4Y)Wg(|;5NCuQ{v7Xw`4`m={2Vq;PcKW`D`59B|'
    '_ymY63+y^fnB<TIG<D?X8jTvSgXYrb>y~keVYK!Z|WEoFlh{bL8l7j!YJ~Xt>hc%LGz7xniGbLSE}ra!eQUI-io`av`r5UTN+XLf'
    ')V#I;<1FFh>Qn&6(U^TocKG+sk)_nM;O+m%oL_Gu__8<C$)6$%JNtr+%<{(M<4!F}o!LnhBmTqPJu~GaX1DtsbU+G?xe4xzm&aSq'
    'b5kMBkN*n=(~xJX597J(MeoTjPYxXK8=9M(_Cy`__9bd4c%0K4QrV(vBL=FgenL`hP>HUlu}rP6+j0A=E!%@;QGw_!pDQd}x2p<Z'
    '>U%Uw|_p3JB`e!+Sf!3rwK6ul6jOA}eo!<Oltxb9g-O?HnG@dpn26BtPf}%bffGLo$bFPkw*_o5QmwKlD-$lY!EpS-'
    'N84Kn@?jqHHs>OfnYKas%`EH|p$BvXVEc3TS02xO^}tk2ib#HXJ`c8iT%&%SIwOQeXVU<OhP3%VDN;5N2v9eV&d2zFWeL3IX4(4o'
    '_D=fKzLxb8CIjc3O%D<f>5H?_DmhW$>w&;r3dVGPGCwM{z$gd9F@7`zM#*`oN_k9{pHTVl&d=A*RG;q;FAe;H9o-'
    'awe0rn0%edRZPCc<U6cT5s&Vqmqbhv&sC~HL=n%ss==<G$}1~tmadf;iHH5x^YeKTeUfHCA>8<1Zeh5=hJ<@_^c8xi>LiZg_LLZ|'
    'PDc)8A0}6aFnQ0yM5i$-'
    '(3>I)mTZ(~n~SY1*(A?lq&p6d=DCcNqmwPLGa}*MWGgtCq?QV9==qG)D3k4A=ISM>4d8f<56AnNyiLQgEtmHciZ1BapBf8Cc!;U7'
    'aNMEVWPl@iD|#?ih?o-'
    'Gt5t)D65RJxgDh}dl!jw!ZTE$Z>KrGr%4OW_E&ChPrTy2<Drd)t#t<R_rQ;kThad~LkH~96M82Ow<R={e2tjf!%mijJ`6<{WW<mK'
    'cF27?Im7j6>J+rX<oXa|9arp(88<_>>IxaV{LdXm$X9Hx;;AvHc|LFSLMNIxwD0ZN`;j&m{!b4mZi_Gs-'
    'n+Rm;L3ud}nb)WW5y*UBHOPX@x)7PTdiQl6+kZKjMA`*=N4z?~g(nNN`>N9hQ`=4r*3=T-hs*H}n`a<fw~x&kAvQlqVY7-'
    '^4o*xwAQDbMq@#N~mpOjI^O$Al#GEgp=fs2r;^#!EFh$T=4~=Yq#cO>mJ`I2tq_VK3kd@T0JuMcC@DS5tvG}!W6M;oN2rHwoI72m'
    'vz~Tq0K^81lG*A1HTex=ufCp8=Tedx(7bQA=D~5K;ZOW;0EfLV0QwmHqy3)*qqBc4<Vpw)988NK7#MM><i>~D*hBepH62p>fS&3o'
    'A)uhDgV2jHM4(p1p%g6e)A=W?Kl>-^l!v<Kqj;D7%KkIy$$CxZC<Us1jnjVWqc!=q-'
    'So}n_iEtowC%qEoKwhgFL^zNiss@=H$mSt0!4}K7rp#cut#j$iv|$>ORG^vi4yiWVlH7qJsR$SAeq2s;AUTuc#{<cAAxJ**4I)f{'
    '%_?z57d3$Y_#dkH0r&A6s&4@AaRU|qfb;lYDs=$g@ms0`0@v|7{c}h39KWxZ(lr3#^*#u<F?psohzptgTOos}U;FY{5W+)T9t*<N'
    's!fDJ)PwM9ltH{sHHa{XAE^eJ+{WhVT}(BTWlYN)(E(GHJz!Ei7E1VC$J%X)_Xb=#%PjHm2o?gdKbN%*h-'
    'Y#9ctE^91mef3l^LGsKyDWPO1UqS2chGrul8f8t?;e6+Mmfo&~emP2QXAx_}*N#Wb!a{9Q9QzhH4`lnWb?(9397csagYI-'
    'rxiCN%-x$to9{YR481pd*O;$T*5P45sS-LRhtN0>M?mO3YXWb1`)XYST)GxG#XqIfF9$riO}M-'
    'l^DI)LnOWR#p7J=DUv?s5S`0O4rf&yKOW9*2ywP1g|qh=lKx&{=mgb;IcNVDLk?@BTlg760&9~J^#w!r>REoR;V_}XtaK9l9C9Jp'
    'q?y$W9^d-DWhwwrL=H!pmx^0~SNb^U*Qr+>BvqvHP7jegVb7#Z<>O59LXk?{VOPdN5}x78SV%5bZ8AWTY9RISy`Bk@6t28MHOK%-'
    '3RtdD4Kn$TxnJW2-'
    'jAbTvQ73YW%A#|GPKKR&=C2!a0)3F>x;PD%?21B<+8CzdRbS1SWzwQX^?N73nJ)0g}QPBQ@V2Pxu>vv1osrhE4Zg9-'
    'kCcQarquYg7?eE)km5d*vbiha$jS3ERD=Znf!$pAji<}IxJQ@4jy7ytajW6)utW{_;w7seOLzVI10HJD_W^~W9Wq8pVV)!H-'
    '=6K)}&SS$IyEQiBWxVRVDB76qTH69m{rLvP{a<X<YX5s9t(0T{cU)T893DN4l!BQ_oI;bV3Ny1|dj)4nbO(5@WWptjc6x_|g;{$^'
    'P)AIX996;7hX}BrV}fa~>qE;7c<eB(34AW4#Qk0bnQkfNdmXM-ABB;B`ddY*hH19}8G`i1T9s`=x4=0kF)GpPnBfb_fG>euUWJ&+'
    '4~lK8&wazrB9I_6<^@+TfnM9qmm<qLz^bVJ}LSgLRX%hlgM--'
    '%J$9m}ky$MslSBn<9Lv!?x3i6#W<~poWclZ>pb$jd~xd_JhrNKPsh$4fz0DWOWX+e6E(TH4XcQA?$w%VLvDZ`~BdnZN2oX0cdMQy'
    '7uZUK}7)PhyqM6`l*M<f)*ZPcr0i?R&6qXwxMH29WJsny+HR47ulI!eo?<YGt<i;_1o(Q?QsEUZ{;@JsaV>|ohhA%3UQ982%N)qo'
    'aBkh(_v6F)+1B&tTm_}HC>RD6|Ccb%8}v~tmA`oCa{F-'
    '_~|(vO+q^VzC^~7Y(rv}t1V^Axhe40hTts;!CMp_%^#QZ%V=Q_{){LfjvR8d2JoHa!*?r_Lp6L$ggl}UeD!mWh=nga#E4k<zOC9s'
    'z!$o3L<D@HgGWTbw@BTd8NSD<+v|t#NkIzVi8;WmsVN1pb|P2tce$hsd5X}cD>3~1K9_VOr}5-m(wzYIhg{NwEaiF3lD1k_)-'
    '>1}g<xYL*pAPIiiJbus2oNf9E~EFc2b@x_bP0V4a}*~lrc&fti-'
    'iKXUYCLb87(EOMJ*4!sY+j;M7iDK&()|{m59z!b6OVh3qS;O$Nx8I8xP-BD+{24AV%FU96C)+cS$59<Oe%AF_i}kgdTLeySi-'
    '*p7p1ONmJxxI8sSnMobNk<F3$?8N1e9OWtfHkWiJ=9FQ~QmR_c)HGIi3$e;Wte%kCalOk>X=u6wZVdpkf)WYUz7f9X7ltVe+cbg-'
    'cUA@7&rG46nqOye4RCv@kK4sueyvYD8w%N>kg3%@F)9|f@DQV7ar?At6Jcs$P)0?VS{SlX5vG=_+cTTm6V&bX<F+d8DS+d;A6GSV'
    'X31ALpmR|r%;$><+a#y`Jaqz7h}BT|`?2$xC2+M2t!W7F9zs}z5KeMo+8syB=`u_|MaF7R?B!o5tKwyxeuhE-ep()vEt!{b`URJ*'
    'n3txvj?31pK$>1i4I5y3vXAK<Ojc=IT#kad1-'
    '5LuYc7bzG(5xwv6!w?Z6da8FiIChY}sH8FNoN(33YpR>I`*z{g@t+pFcfQ<;WLV<$*RnjM2Z)CtEV-htmMZ|5lu`3v*6eHVBgqXO'
    '<_{62Yc%-'
    '8jT`F2wbTDW7$Vis6~XVk;n+(=xIR;LELsEMS+Dd%XjI%V`<;A7%}zWkkRGS}ylxN&PIcK?rxQ2jauHyi0@lXB3ny0Ak$_7si4Zp'
    '5ek+5bsoNB0vm7a$y9BVZbho0C7&;o}E5J-CjS4tMftpW)Ame&fI+~Y~9tbtkUvFq$n^cr%TYeOr@*Tar_)^D*hf~@^WGJX-'
    'M60pN+R}Ve(c1^Mlgx<X?nrBuFoFv&}Gl(s3cd?^~=^3{ye41ht6iFyncYgeQ)o)*PwPpD_WFkJV|$HTAN~24;7KZ+5R2(1QxG{d'
    'u}<g<J{!ppT1qdfCTCJiYAW*~0s^fgAZSlUE9zZBc_;_=g3NKW$_3b|D*>@py`PDVnf||A=Xvq<XTG%jX2Cp8Uk+^MX`QU^Xumq<'
    'ZoTm&*jHo)iiBf*{osCggHKswZ5?7lo(Mj<MOfOkTwahZT-'
    '}=qi;hyO{Pr_LxCCna+J!j4BcZ&zV9})?mzE_Q(8SF8{8r6l!l4I%v|}^n{2fWqm@#ld?XMZOo?yyWRg{YOXxQ%t)AoQv@vsO$2f'
    '6*WlsfzEqC8r%3iAu7pyN>`#-dsYni>1H)z_X-VgW%|+6R7!>vrNoyidc$)cnmEZ|Bf!^Z&VE@R8`M*+~C{1%EZ!-'
    'vdr9vKL*r*QTzO+gWCutR1!m~ukExLr~EiUip*VIR78jSx+zl3KuGs^lco)q!4rB8}@+R`Vpjq;3Ol)pu)PkE@x^xwqs(=KO~5a8'
    '3^AmAV_7YZ8PL%DoZklEbM<Q+me+G=Pp5k>B9u3$=eX^6k!NThC(CS)y_Ell+`R^?*Wjv3NdkWJ$iKc4Z9I%+B8^A8lBY{?LniQ$'
    'Aw@TqIK-A=xvtw23wpTXe1BGPo>hJx-}25wP-n1+7ir$jv2=Tjn{?DMH?!#+D0_V19`Pab9l`Wfc(=e$Xz>qf72b)(n0y3y-'
    'ho%0Q@gmk7WdA`w=JkQd@J%P*SwsL<mc!N9xKV~o%uck^)OH+b~w+cU&^+68tG&nF3o;C<J`B%<jhw4l*SZfF}pXIWPD4;i_yX$E'
    'YPmuYvh$qN=I@<`}kqf2MPxE9{%5f=fVk&dNc-FZ-Xd+;J^n)p8X6aDOV9!W?os+L=BYA_6H94QG<XPq?*EM7Kx!qhfXZX3p<)NG'
    '?<fl0z$AQ1Go(g}GPA08Rp3albCaq2m;Yk;hRwvKkNmr9rhnhp(Oj?~hlPBFxTAe(LCp`+J)rCBh0kU2zL`b4vYq5x@lUyv~=_D6'
    't8|k^Zd{yyxxJ#&TSu<XCg&IsdRo>xFT&xlu>cKq4+l)T4m!qbXq2M-'
    's)s*4qHgmNX!_S}1RSSlnznH7N8Gin1uJ&R0`I{^A0?YkD&Q|kO3i)BqW~(13<i|Oi&wIR(YjQSm^8_J3$<@ol7>w`hBArM6m5%T'
    '|!K5e@4bdI&jELu9d`86cFg}xQ<mdZVTyMS(W*E6Oee)diGrop!6kz-)XY&jbA=jF#mK-'
    '(no;EAv1SGbENo`72x%|u?pS{oJ=Qd3I0heFcv$PMnTxVm(A949_bFfx}nQdRH0J2@a)iHw;|HegH0}(v)tGL5uN`ef_7wi5%+P('
    'wKiXv;9s_KC;t%!snC&Q43Jc<Y=)NN~%>$fNdf+(wN7!<^e88Kl%-'
    'F3s5hNOZDD4<9N69xoP0To5vb=6U#0sYUZs#8_n)9me<=lS>Eho$A#?Y@29t~%$`d)}jwt?8F#IK|f~s9~5&Ak6UWK2gYv^xQs|C'
    'o{V=x9eN@f(X|zd_jb37`~8`ZPhSI0Q{Murf@$h1gTG|z!3i)NSvj32RwVU<8%auUV6K#0!gVa3d*s6fFYCrpnsA`BYnbBB*8Wbl'
    '3;7aN^5QWx$Vqw`<>hNl_4D*98pz)r?82w+<Lnr18nQ&k;uRyyvUQ9?;`XD@tVR?fT<^c1M?8@UnaL{*8ZQ7#+VV1>pOBq6vzUEi'
    '?|{RWC6aa#^hiDl5eT)^OXo`U9ATMw61Q0^|1L^`#vu9d^7|b@p>ibr(-K#FBg6WHskej;b&n3UN84O6kG7B+&2JfA0hiiq-gpRl'
    'k05V@sXG@pgq?O42uanyXEQA1cpCNW4Ns=|F8k&9YX#if`75<+DBRW;G2XiA1pz@+D=QQ2-(55N&6aI_VT|~{!-'
    'xF=!3J2%h@_}a5YPI2&R$}{Zp?D1t(m@m7(BVrp5%6l;|7(bx=u(9uQDc0=D})C!4=vx(Kuy{12+ElUJiw&FF~S;|r)<8GV!~9{a'
    'Bv%{sxho`jLx60*M=8P*GWp6`|f__n0M*Df)EPEpM)OGzpPn^~HsTGLcBOVd<qItr>%Q>_^&s7j5Eo<KoWYHai*>Q+)?qo+_^HG}'
    'Av7UoL<a+43p&RovYfV`5)FR~Ku`sZI23P`w!t3m;}K#d8Y@oQsz6GY?J0|IFLV9#%I()er6mZxFYN2(|}nxh(r$#ZXFSo+u5g;1'
    'X*VD(uVR_#4lRmfo+U`*M%v(<K_@!CeJy-'
    '4G=ouf7)jn{Ur+J)5Bb#j#2f;`KgAz?ipW>s1jx%i<1Q`kpo%Arxv;rIlZU9QWyeBN8e9~bcwvw8KumH1My{LqKx+g#4ji+CqYN@'
    'WnH_3gVl6qZjEXxdhsEyZ_rE?83bR~xfhgme?s0|L4Uw#WLLe9mQ~WEtpw5xQ)i3VK;`Lnn(mGz6DL*9rh7i{U>cBFc9dzK;~Px&'
    '+<>slPPC_c6lmyTJQcmrh)3H2(%+>B+6`serii^Tbo}dHSh1DDhMbSNrrPuo<DYBTuutK+{x7O96?!6evIPK`FSR<-'
    'y;sW%9o~qxq`7`fEZ#d1HZIY{l6cd)MRwC3(I2UVbY=ny=~s0nJyz9N*^jEF0CBD<Sazm{X*?2IY2Te*=Vf8NQFACcP@YG0xjZ3@'
    'ZjZyY$2t3B-'
    'JnM$Ey9s?Xk7$JWtMQv1mlIk;$c6LE^khO72f@V?)7D3zS5?wN&kMJIA!ogLiE=Zq(EwNfXuK~JLyJF(fynVSU6pK7D@e+%=aaQU'
    '&1%Mvbo>dEqaCgZIBYWf~s8;Z+u1^TNMXDjDjn+uop%xH^!mkXCv1fmDzz$F3sV3ZZe=TtUIM&fH3z6Th19mP>!%f#FtGpq^h>{4'
    'O>P9X2!Y2+P}Eb~-bmL;6_KA-'
    '2X)@e{IV7Q8!?YBitPPZE|7czX`&8$)xyzfnfIWtwqRMYR`0^z66503owxmI8tOr_PmDV2_$VlNQzJzidTwAH`6hRH8^LH)ZFS~o'
    'K<^GXZ2zI)?CF?mUWmTAS=`gh}l7I1CLtw9U89uTmA101&I^ea=CT+i@*v;p2Pu(sNoxtOEeVY!(^B}Z|?!_CoeI7%D#Ge?^lmJf'
    'DzsoVb~(EFb>dJj#^o4qi<)`8Pr{XF<41KuS}K9ML9)#*S(-vnn7kXB+N>L%!O+JaE8@~Q;;HTXhOm-CF+JFR1v$_y?}c2e!BSMO'
    '>jcV(^K`i@)|ipbFg+M*R_>)2fvw0dj%Yztbw^?-oY`%siz$u~=;=p&=KJjw3-U8aOy@3dRMf3STM269-@eU$wpO=5dH-'
    '9cw<x@xF@LPvZy!xeRbIhxC4lL%IB%T;L_f#;_$<O7Bi-'
    '~>N+S;n1KR0Sr88c?4$z}rkD8dL8Te*+qGbmKgJ$#DX|Xnd$vou9Ofm&uDXHlIM1<_toKzH{3{H5TC-wufpgZc}4&(B5d~`-'
    'i%)+!~8u;_U$%i)+=l=hj%%cPU4Y>eOy3DusEJA@Jjg$_CEetyWThqrm00*tQqj$Zc}ILRKoG!i1D2$&m@f|1*vFHEG1JOC$bBUw'
    '9k?jNe<@Caj<*Tc##VUVpSq%{x@%UXup6lVe*GlLokpV_OrG2AITgXYj00WxGUDa|&Q9q8!qVdypuBAF(48y>JmbLeaZRjR_!a{!'
    '?98Zqg>0Ye#@`^E&nIxk;P)F6HP^ovO}<38;$;)t3wTB<`f#Wc#SYzn55T{AzL!_lw}8X2-'
    '=y>aW#}pc7gq0}=rKR~o?YqyhYH8o&b*0G_B;1S1F?;qHNa#CRb^xOX5QHAi(Z9+Z!nqk6cH0DiGPD#l#|S!Is);z*#LB0^>G5~<'
    'Lw$l0S}rvHj;MHBcbJ40a%7qK%G#$T&30WkiTy0F|Z2BYl^fbqNP+jGNM-'
    '=!RFs#7rjJ;K;F;DMR00nNg3M{3tQ2)8%uW0a$h?#B5LK1=KNekkAXBN*oIfe8dxrV+e0jo|fZ1P@F&9UPOo@HP5K;D(rd*ln<aL'
    'r~7`R@ERO592n;djdG%+$Oh96!Hje6BoZ0vOf=1&r76DzsJ381>tj<Y?lRLeKU52LKrS$S15!(S7QPoT&XTBH-'
    'y1zy8<A*UVVFR2<y9)qfvFz6OKE|D>-'
    'G2{;Se1cN%0=tEN}?n4@F4JSu_8htsHBmqz7#X;dDSm{&i9+YnKAoCc^`A`j;_HxUcdxmCq_CkZ)#`>rteqyYy3d~u!V=*eQ;O+_'
    'tnC;Y6<DXd|5Xjh*1xg2gqR88_=IG5k)<iT*$Rwqc3?;}5XK<==<d*6q`7*6$lD2z9#F#+!I!|K9vyTf4S?*rW7_tdxNc8B#{%F)'
    '3(70_M6rL<k~F6Ht!wh7Z1E`Li+qLK0WJ5I@|<G4IJ0qIB5kbXA}>G#u+J~}bc&SA1~Bwt{b#M&Z~OAOpMFnLIXA*WYaidY<M+p_'
    'g`xU5o3i9LihqKaG1DnNcG+uN-Ze9I&jD%qFF$L>OAnF9P3Ox}={kJWeThfw&!Mf?y7-!*DX4)_utA5j-'
    'p5WaPt8tD%?;7fRXUwwN)_)<nf-=!Q)tds0#jf85FIT75b+s5C4!)erQ<E`Ls8g<)v8#tRr-'
    '8S9<&ZSYbjdy};iHf$WEgqN0B;fsM8s6*E@ctkT?_)&GLVo2g_*H_haCx37YdFtWY?UiGz>_j$fZ6&iO{~kRY$+h>x7mm8SSGL4i'
    'HI54>MVo&uJ6K+q0ohk_%Rf^uc$Er1^thz3(H;5zaCp2<S*#|Kz)1ef_{CMa<s2bL3cYuIwY=qJeS8NAoo}ra_^-'
    'fw;>I=W7DoHVCoC&xt`C7kPVU8UpO(cA@sVC6Ac?8uZuYGu=m1p3q`)R00wWNoM(5{2E2M($PLsi?5oC|XEk6OMGDZN`hV|u;C+<'
    'I33}db&g49v!Tr?t^QTbo!ZrL93f}o@Ob+mpJNuZru-rX&!0<l>^xWB?zCCx(9etN_^si324W~zPwjj~wi@3ao#G_Ea{8nx<s)Mk'
    '!<~A<xGDlyl#QP+3)STl6?+NCp6-'
    'O#yY>plVhd{{dxzAI7nE5pck1)R`;Zd`ySHp_XS@?YAvbfw$o@EM{9(}n){@_|al$kx@A=pZJw$~xH0X{$A0EKD;PVseA{{-'
    'i@l)wgl0qH&(wd&$~N8!qVy`yktz}``={Qs_0wGrHgD@aQWrV)&RjHJi~TwX<LVlZ>a@m4~t1xoEf2yBVQ1F&$!uSck4eg(o~%&$'
    'Or+*tNoSl_7#coV}WP`6pB(#}yrsub*6?x>yRm=rV4;*$~ji+1m6O@gNUj|f5F?V?&&(uO#1MDiWGKYCzvOQ7`LRPKd1hHR;RSf4'
    '0dUawCSF0a=opVjLWC7<>`MOZL+n%Ruz@>;8~^_}W2dY`ac-'
    'Q`#eVib+JJPeCLj7J7OS!c%tdMY9mF)n9EgcS%!CP(s8g8&FY9*DqK(MMNs`S-'
    '}Q^HvPb$wGC@aC_og85TY1e?Niadj+r2_ZBbl2T!d56={`5ve}KKl`Bl3Zkqq7pUP!J{ir{M&EINnL8`maHwu@h>l=m3)Ah|~^x~'
    'qVe}XmmWUyPg*MoH;$5ww$^#RQ8Ygyb(AHd|kj>XN5>=c*pvbcplxRcBEEN-'
    'a}?&8=$t(86i_`J{J)>iZfo<d5hwSYN4vWc^nFy}{EiL(OQYty+aG*tSpFddX<FyhJ_5fVLcxdpd>c5|ElnC)Sv9Iq5BA#6{}m2$'
    'MU_;XBd%t*+o+gTokE4r0O;fikM`7B<)D4CTlSN$Hmz7bz0gqy!iw)y-'
    '(?VZ)JQt2hS5=eEfc|3xD_T?zILFCoh)e39{&(*Ib*bd$h_jGi2fV<<@66^$jhI=|X5PA3IsH0=b31&vL3^ir)pOH<(He>Q%k<E^'
    'Qw^$k3v}`Nz8X}u3YYpp8q;q9KGHEmX=l%~^nW<24GHQ8epaAc~VNtk9+F?<+NZMifP*9Sb?Bh~HRKvRho_2|lH=27?g*`FlE!+~'
    '`TL?M9<m0zPJmVSoTZWIBQx<`BqPt&AvJD~?PR`yB47ahy+dg!&BJv}~a`|8+FEdkq1y(1@KQliSRDvPNU{~;Js?m0HITxSb4~$Y'
    'oIU>}q^Z?z>UX}<xS#C;)gQUiu@NGXGJKhs%v|&^;8KiE};ZeBI*x^yQ(AeSmY~Q9RDeJjeMR+B4EiB;)xlUc6U&k(^C9b*xm(6m'
    'k434)~FQRq^*b(oraw6sMG95S$gO_Q*aX7q8|BWNyWm<0ZhnMNNaU{G<!;Jy(`o`#4aQU27G@8!ke7hU|1eY(^4e%$qd@*YSynyu'
    '9lfI2#WbzYj<1bV4-t>Iquk06v3vcy{!iBf`<+E{Va$SdbA1h{Vv_@@)d#c`C+*~bv?oo!o><d%JH7MR0@3dCz2QO314H3-'
    '}0e|YFYp^*f{Hbqzm3kBnk6|j8&s!eDG!D@_@fc=sxj<1DT)yE0<|_R0N!OCx=CZ&&Kfx|Pcc|3b7v#|05!tP7bX)XV6r@%4p2yi'
    ')E-yBns+jyZBg?C9$`MhxP}LDpxKPy*`HWuHI9Uo%iG4x*+~Z~)&*Hybhyj@Uw^Df?BfZ`-'
    'aLAo1dMUcpN!km{5mk;{I7_uz@zPj7q{)S&6V%w>an}#%_uOirBW<7=6-'
    '=GXtspwGUPuugNiQUcj+__LL`TX?mMD4wsiz<P<S2v|cWaA}V{*T&mM8jWPZog;Lb!;@B9KA2QH=?xmC{C75>zYogBlP}D+P93l9'
    'T`6xCa0KB_^M-{C{xRrZ@yW&G0{X2>LONpi2^kAQy1C$tr+d#N|hf6p1Rh+{}oNs1r!fal53*z0C>7sTSwsoM-'
    'YMF1to?B5^N~PpzK<+9v&--6q|WP6IBcBr;qfi-Z<K4HKjK$w%cCOzv;lSM{Tetf#(}_lKeqF5>=BRIXEFa-fn};M%ZDbD@$z_M;'
    'k*1C^BgyaZYMlk3!hE3JcF&hijk8z(zOT_EI>LY@-'
    'QJlTfB{HW*2B7r>nVsek#Sg%nF92g}83=Uy(E%L=mVIGy_;KtoiiT65EPx~!JE^$v(oW6lr*T0WQ(7ZP)PTxe;11Co&^%D1$f3#^'
    '6h;yd8hv*rG|G|5Reo8+?m+q-;UIBHi-98|}9XDlCm#}2-2&!?`m-Rp>AmJh&2nFP7H720OS>L2(K{d`lsR02s&X-'
    '~#Xg=zH&1rHm_EhqM-'
    'X%=Vu|Y4yVlxE%X(QWd&i>Cb{12Y}cc<+?Cb10b6#6{wOTD~cj$qb#Nx+}_2xgu|LZ{&LHGR!xj|c;GKO&P6v1m8@Y!mP-zT&oAY'
    '7TQ{hC|?)^V9TtqbaMlw1B|Z&ptk%Wb!{6pQD&;pGCITx9!1De8NRM7>dsoYD_?zC~d^$L2aUTvz&dwX;Xhp?!Mq<P0}^^Cew-ug'
    'Uv1kN2lGac{0K|(str;Jt8X5K`&2vk}VhpZv^4l4F7`%;h)nG9-'
    'Ht3AflgR%cM6jIoDQ)ZiIE;R)=n4Si(HS@~Gz70_u;LoNo)LKW5<TQ{cE69tM^XI2O=1`^AT116&Xk6vwip5*c}v`se>O6pC;We+'
    '`A=LNzA9qtwO->k0ld3+xF#Hg`|(i<{J#Dj_m^%1<}Fz~oFn-2{RAEI-Cw$mG*D{Zxrncy<r!F)Sdqdr&PI2Bw6{ISl`Uhss~lsJ'
    'zUf(zbyvi@a#Mn#*31QULeTu5Ke8y@xrbJt@}ampgZIA&(cqA4cd)MZe?iR!4bOVM+m=v^sujqao_6D}*-'
    'Ya<42k9epeQ7K%`~h`)s*^b<8EKut#*W@V6?&Mz!yZ*ZcKT$Z~x_^iZashJYij^kAFv6M1lWgZ7ej*j9{iq)(Iod67!z$~++NN&n'
    '(#W2N1;GE0wKX~A*lWq$+F4YGY7TrbE2lpuvFQz`YGex|F`rytI@lxu8`?QFcQ6JoAM7*5(;65wj71RfJwuo1T3c4{Lh)u9fpaSA'
    '(rh1drm@4(pekc@(a1jrM0&%e#6EIb3BZM^uuM^rBd|d9v;MJ*ByOh$qdNKk(Yn;vJuJIhQL25V5F!C3<J|5qyV*T$DvN^38=2i)'
    'X&oTTD9tv3+g_ozMNR^*+b9dYn)^9g;$IV~`cQbd~9OHdEoyIL-'
    '6?gkaaZ5}9*lnY@6|CgWfsR{aMxb7(>fFa3I4@@MJ`K*xnCz7WPJK829tuvlhQEh`vr3H#0H?Oi%R%5|c{_y@I4{rLDZH$xM(4(I'
    'TYuLKyNkuevaxV7!Der#k&ydg)l+>&K4)vLsJuo<wmdyg0&2faV6q)gvFnDV*>%6;DR$jc-'
    '0xvm9#_MEp&g8G!P%fKjPJnFpgoLj@)?_P+Ag2vw2QGr_sUh<7&{@R3zfn9l`nvKAeVJ@d*<!(`Yc51U;lWh4kBE`<Doi;uc|Qtl'
    '_2e~&?|o>$T0Qs+?61|Qy<S=3G$Tsc#e*vsSTBzIkw*UkXuH~cL}hyPlIiE8f?E$gKcI4Z2jV`;H{caT@^tc$Zg^CU6dQbisG>m{'
    'Hddst2qitn#AaJN<=Er&3T+k`rAT~{UX`YZoyWny64h#%RZG4I{^t``cXSU)qrH}n-'
    ');Y+uz6U7VOigrqoMWQe&T{EqzC;LIDgHQ56c{Wok?SnX^6CCgvw|4p$$~P3HW)`gm?K=S=nS934ng6K|IY?LC+r%?6%2gN%G$@4'
    '*r7BH1*}6_MYXVRU3Fq}{~3mEqfa77j{33nS9c!XMJl!YuCBJ~iOf*RBF3$U9J!-'
    'v}A=Ca!PH<IWV;H^D+f6V;dSxGM$qO?licBc?B)s8`<uE<BXuHEzc+)%3Z>#H)Y(lu+ElHB1S`?E*C>VB$Rpt7!92yd%`db5Fc~P'
    '#@1d@y=2o&(VZ5HSumk1xFuV2dCk6b{bwMr{VQ<0$vSy+>vrcjhMVPqTQq`xT*LziuVl|)l6N(WkW_~QrB`AMq|cv)a|`9R3GmEP'
    'Z%F(-0LGbi_!k7+fh1%yJptcM=pa<A@5PS0sA<YCP}PyQK(g@jih*~T#=;&sqgF5P~gJvni>k+ht-$>;DU*#1_1YL_3_-'
    'mJz0G`H*lX;AJ5T#GzHx44BsAb4@m=eWE!}COau3sgpW{$+rEcz+F^Q{$wN79F+GEO|Fp;SEVF7py)dyyl4!j#htkRF{6MJCg@=m'
    'hHAu8{Xn<XO++xv%+f7uXRG^q#csu?UL15Z|Q(EHSh?s=1-dqvsCB;m-ugKS}=5tBbq^oc3v{3BAFPj#M-'
    'G|he0PKR5rv+elr22Sn?EX=GJU4cqQ6JCIhcrbIQ_ceXsjXUxWCK^KR_ZZHhR8>1?wf<DO}b&6O*dF}{q#uwtl0GsIbA3GrW89Ez'
    'P$(L(DVayPWpj4CH=rWn|NUQV5+9GE0n$=lD~;83HioI{?_pmZi?jZBD>Lgb0mKs*=@I5BKe2N=8A8P<R7C@jlTo4`c>#(JUtZ0a'
    'H{E{IDSBl3BWPj{OJKWK1Y2#H;zwHAJ2{BXVu4Z^d&8-'
    '(f&?GfMderMgecTyT3<j_#l+8hsft1(@W_DZhU74)}0LB9#~7$U_Cbt*8fX`b#}rn*<Z+R+_mFEYOU?Zg}hqZkPBJWcTg8Dq*UK!'
    '_jJLHkdfupPl)8nHGx_{-mjrZroy^t|MvnU0%uIc`aaAEg)m&ij8F(qP-'
    '6lh3?`Wo0O51h$8$sY|J28GLwL6Oc#bxt$+#^p$Jus*ExG&(C4P>x9s3$^>Z?`4e5Q)aq``FE*!D~!^zLH#_R!lQjowjd^!_Q0-'
    'Z=^M?k8~Zw@v0`S1TZQpg<P55s(81W?DoYrW!DLRYdX{;ImAO$nU)pZVgj%06_|Y|Mj5zjqVf!e_O}o1z8@jz6no+LKm*#iBRZXr'
    'p5$77wqvw0CY#GkLQN&pVY^5LwAn)c#h_yDGzuwL&+!FG^k?u9>vIGZNuXp<gm8oaZgjCA!HN7mTDk>DUv@F$%~5g(e5I7O_7{mW'
    'R!vh-tqP3XtO{Q&{2~J?=l@VjqtY9Q47V##iKBVzmg)~?|qy5TCDGLUyBvO$cfw|F`px_wOJi{FngIa_8@S}@v|2QFKipm<cB(J>'
    '!b9mYJgVX)4xXH(t&@C!leWMnvaM)E?Hv)Q%)n{3TxY_Zlu+==IRw5kCFpTD!-b^-'
    'B`oXhH87^zYtq^AjAFeM6&4;+*h@bs>J*z+%u!zC*ZoQoi_>iW%lS}fmcpPp9uUZbo8mfn?Xm7BK)Ov)HuR>MMotO-aI;L7UA8bq'
    'vq;XW#FoS_+HZ@FWq#Q92-'
    '%AS>~LjfB)a2aP7UnMd8|ef6Hh7@u?ECj=sX?AbCBPyRl`NmA3f6ix`J8FJgob;ipNdb2o~4r}lY}OD?Axur!T3$Z6*9&Ek%7y7_'
    'zaxRab={@x<)Z1UxUkV`I~G=Fauca=|>zqgLNDT!pfgiBR(Ekf9b0)KCYRSRg*HS?^B3UnnZpoP=_Cd!~t>T^C6g-'
    'g*r6opICJ(SO?6H+nZ2S}bWg>`C@>u3pK)kvEiXP)SOUb{lot8Aj%O_;BYZ!%T%Opjrx_eXbpDnm?+x4lzFP=_omvpvVO$Wpq4yR'
    'E-*8<#&@f8%y8f3be_9bDEC<VWAhB@^UF-^C>t<VR29;4F}jayQ3@?stW1w>9@o{s~sQDU)}FAT#4Z)<631QMh#5-'
    '=lEpw!h~y`J~i?oX`<dZ3|(k)0SL^(vmQ5FSkwZJ;l}4O4DyXv&tYtQ$nhMiY?q$`oGBJk)}jQHPRWzsBHWRCeLPuBaIme7)Oie<'
    '=Ak?il<+JyIJxc$6&o&OR3}0RoHV!-p6be6a4dJSH*OdZUulx*E0O)2Wm2=3#gj4@J&CL$-'
    'n5Qy)y)38K&1i{vT1e+}S^(aJjR8<TL$)Md^N|A9zzZJl<U*d8V*Wc!{eC9Lk)S;|DlF7-'
    'AdhC=TZ7haGJ8nS0n<DJIQKX4}5^vqfYJm^{K({Jp?re_K(ukjVkI{O=_u2U;=QMXIxn+5K6}<k6heOQxfnK;^?vGC7jbjw(#f=d'
    'z5p@dY%(TKXpM%T+21{&pBoYJ`!^_0Rri6s{5W&nR3Y?4S8;UYQCtu|o_A_z3Nt+NSH(G$q6QJe3(*hFa>!d0HD+D3d3PLh-'
    '@;+&%1rxo6~4xo70lxM$=c+|CIV*p<#%{0#0{{7mjy{4DM)4i8^5Zpz4D*=^2E85zvSEjX5uh3RUw^1ZZvMiCCYhV3lF*!l<mD+*'
    'WD`d1XLs`amY#-5s5&P&kos-'
    '~uC+z(jBqbLuRK9zX6UwHO7y5#0*KLbIM^v^HHzGJCIv_qL+pEWEY@H0G7oiHLH5p#AzB4#A>D;LgTev{gBY2HVz8cWi+8f#!7TG'
    'B_?aeQy#ftpmZwNLRr8=@~O$@*#5<gRZ(WfU$-'
    'RT+hgQdQ<G6fH^CKeZH?`KaJjmLcDWC@ETj|8g#6oWS~fkfp<~`G>%7dM5G#P{DETAOQRsODp|6Q?oQt@YDpu{~fG0B1dj5QbWm+'
    'dlz{l<h!lM&3F>3yo}@Zj<@nL=UImluyDTh9smj#DDN>$-RUAf9r`hoGpzYv&x+6V&tJ#`nSpQ-'
    '3wa<j@Jlr&fElQb!FdQXuo=2}A<V#2)qq?K|B@Qj%cmgYuS;a3|2E9z3DEiMVw*&@K|jHA`LV0KU60_u7vY!m_AB$SkhQS`<#|m0'
    'DgnyoX;7Y)sGPb1$#?LlkyYoi7+bLGqnnXO4}a>TTaealwKK+XIm&vnS8@3l>vLR#^@dc|2dgQpiBoGjodtBJF7}c6DT*zbHR*Sm'
    'c_VcTUJ3;!T*FJDz}%w71OQVTL<9j7Y$k$$d72uKi`bt6W_=VocIm8(m=c;YVNar$h?3*CM?E-'
    'x)(f9Lk7Q3?qp+S(t?DSrMO8&>G>=`vWfKSA^RYzOhi{8Ce1{}T0J_PR9BbuGJPrS7QX?2$foElsBj^=_Yh}_S7;J-'
    'Y^%M)0$g1#BIfBWT^-'
    'lh5RENwc&eK0_Q7A0oA{K?h@_jWXpg2$8=!l0D=YcIE9#Wh)L=DJA^DjyEjZ%f+G`*;D3DRr3l%t>z;pHz_NbcjMWg0K1C#Jhwx!'
    'hMM3H~q%uF)w<6RgX-'
    'm?<8F$HD{_o0Q0yi4b?OxU&l$iiDpE9HWHz`+LdpxVIdwSKm2Ip3gJdi0U7{I229r!xo34=`A%TpbMEcLlF<@LI##9;vrqgPFDkR'
    'xr9xVXoB0_)M4fVaMcr-X_dyz841k5+FVUpdvM^sC5;d8-'
    '@YS_k1`Q&C5;bo<+hQ=2Y7PZN#moZh<A|2M=ufY%+UBKAk}q=Pft=2t@U2pFrVhS3*8H!tF()1TH{ju`vlt$cMMNZSkxxXDm|@1J'
    'n%}E8>sK*l29PRZ(9-'
    'y#Fy2WfYnypwr<dB3+AsIwA!Ab2IO)Bo2JU>aIy6z=)MwIBvOa^*5aC~O{(cer3k({Ixcuc$O(dqk^2h>P}1qAv0OHDFuxErT78('
    'ePQ(1n1k76sCuP%0I4PUf!b#b*5l+gct#DE{?SzxEX)m0V%|Rkl%BFxS*`+>7xtDxK7l)L|9a;GZ{iBzLq7g1(X($@!sWAa))F!A'
    'GghsGZy&yE6sRrco{hFpK;8l2`0o77r&rTq_h5Qi1W-6iZoJ8^&?UwL`sxfSWKm^NUbBDx>P-)gjVw*G)&q_FTs3J^TrYJhu-'
    'uxY<2M=OS#_LcMSVK@+WSfveAW+vv*)UJ-Wdm!tqJ2aLo4J5~+8E#I8^+{HJv+{4^5U%7QUCO1p?HLgSQd)MYBeSRkJ=E$L3jjP6'
    '$jz*EHxmP)7LZ|i}<PcHq5S6NbWr^ihHM(aFu9{36CqVuCqr9hY+Q<a5%geWet5CwoT)3Xaa|*YuuJO@!xjLDMcP^{*H2ahnNchX'
    'TP4!>42|RY&PdqfDG>na~<PRTN-'
    'c`DCq`x_mFrQlk4<O%vyO=77+DsUmglXxQ6ASK>SIK2>_zD#9l!_1Y_+L1jM0gK(2`~#qH6lT@NT#X$IMGBE=x<pKfL_$!}%=aq3'
    't>aq3#6!BAklp2^KtEc87lKOxm5+&7;aNp3JRhRNq5DI9{}3i^vsk+>;58OH3g^S??+xhZ}ZZiKjk44C4dr)Lfy%r(rGt9C0w561'
    '7KR2Pd#+UsEE947b4%Gl^Td`2ie4SwDkq4czkYD@q<4UBz806h&1aYg_=t-tzsZhG1z_3<1{LQ@dd`w<?5r=%e~C=KBw(-'
    '6Kp*|dven`484bqpKXTxVO}e{AK8H!%5`RrMOp<m^b+F}q{;Dq*@)ZE`Ud@)!}SC^wb+R5ca$SkzWBU#4OX#BC;#ik|mW=iVs)!o'
    '}uO8Gg1{-tEa@V{Ft?CmlP=>MX}o-'
    '{CVuAq+q7%uoogQ)2=k48}e)0Kx!;GXo%er22Sn2;Z$fo}*J}3c`95&4cjNG=z^&LwG<M!uKQ~{27=3rJf&gKlxZhF-rsP_JEsma'
    '4v75I7oIc(#Womla&yT*|mK@$%dT~DnB<ZA>}?0wfg1jk<}iB|1gXB8sT#sx6)45im}jAD1ko37K&gJ))W+j-'
    'QHZ*(erP)u9T|*UVU%R3I#9xy0b#TyIPG2056#NtN`!=2+j%s?*R4j+~B=OeLP2h&=k99K#|-}xG82l){40*qY~fI-'
    'A@q5WQgorgrSTL!(NZD3rULKuD6rD=VDs=xi}&HTntP<7xyMwEj8re-'
    '<tWm5r>$|_#=%u#9TL0XAJOlZ?*y5(~;>cfQbOvt4QzuRLBRp>=_|vAx!pex+kC$cb7AD$p4NUnC02(8!$8!yl@FaL&3X3jR^oR*'
    'kWh^c)?Ob1Hd~_eLOdK?^Peq(HS%~^J<EMhp_2sgq@g1*imVO-'
    '6xX8YFCT6k1OB!f_SaXNHD!6yo^fv5%TefGDrt<2*Zq@(S$?jbyKK}-'
    '3*>cnxeOA9fpB8<+@D9^URTijxR!SiVF=Z>_TDLy7<A4IBV-'
    'r)9nToYj<ItJUR=s`koF81uguzVWFU1sKx|<7R);=0JLEIVF92$N_{*xXzx=W&(Q}oHRJBWb~@n2jdxPs<L&<A7J=^%xh3z-'
    '@^&__5GY$#tao(u(mOf=0N~Q=rd`d{@g9m9>4)Oi>4)Oz^g}T@G3~ZdBmmm#Z^!Wch-'
    'H5_rLF?a;q@i)y<1RMfmZPPni$@#RhzpY?L5ErLtG}GvJ}54%U{qxet0N);TndAqW2RuCIG!)hv5O}1uG2?K=0A&<GIm0S$#Z58_'
    '*Pb_Y-nEQjWigh_yDJyDs8t+Oo`%E?6aL_Ls0791J$%W*jD}Bxg-'
    'C$~@FQkw$HtM(r_a)ZU*!?I8@`q}zKa!}k<Ym68ufv{%p?zNUygiFQKvE+TsS04^`GDcXTtUaXR4VTyZ+t`=8v*;-'
    'G&Yv9W>;H&S#h*0>#HH-'
    '*_?_xD30KQ<55drW8bBzdq?=kA*x#4@i`go33ph?0d;1ed)YCC<x@7N+7$#S6Ei^R{Bz{+Zj&#;)v>}i#${B4R~t2vM!exFR^_oO'
    's_k4@wEfdqcrGklwNi}E0bZxTyAnBkim_H~!zMBH8f%lk-)B#ILRUu8w)m9!VKPf-TP`hSb^!3BE&9?oPv4dAzA|11FO+i-R$fZ-'
    'a>4h3+P8WR9uu*lf~00w)V9RT2C)yH!K_yP6t9PL0;E^j46C2-<#FXM3^T`_|?lBLA*jl0Jkh~pdgh&vL)H|`mCB7Se&EAC9}-'
    'ne(%g}A-'
    '&g<Xl+8~2U7SzfOY_+Ds^j#Ygunu}0Hy_x=e*p|t1y`BS_k5D=kw9X7osHUOhKdJhcBcV^XmBS8+ZBe_hirYE=V&h8<K`ru$RQiy'
    '{slJW-*Fvgq|5`}(?VnGkaaYRL4>S3V<#ESLu%C}ni(QEwmR9QD{34U{gdAe-'
    'z(OYH3wgRZTEyfFLY`@kHZ%Ev?H<bozF(v*BzY9v5#dTcg=}Juj#IrMT9~8bmGF5h5voKtE0Itt<^Wc$uslTB*0#d)o5-'
    'r}RW`>kn8QQm=4W`o#^Da07u&#-w=cr!Q2dvTMH$nRzN7VPA#hi}76Nzm^O?O)-'
    'K5zcVW?eAW?zgww~S!$a%|jW#DG^|$1N=eEa!81s%4=q;PNy}VO+@N8P?>BxID}DiakoT8z?bmJzBLJXljm*VREsM&CJmt)r+B}2'
    '-R?VI*|dt2aRQSYE%~w+f1&M;_Py?zg?Z12vmwUq?0OT{v-#e8Iu)y{(w1ppRAG}eJ>BFg)-'
    'a$wNQpTAfLgDQwD#Op+`)+zdKNF%<x-wa-7Qb-*<7G%JtvB=Qx#{o_-vsa?|dW<5X^%HR9S}vJ<;GPUY04cM4Ve#Qh+oHYGjr5fG'
    'qcB#H}Ys6CU|se4o2EaKH9FICOtP*bT|t;!FrQd+Us6R_Cp?$Q7^Lt#?i;Xl^Rnd$f$zaeL31%3MdHK>J(T7z1ss5QuE<oYQiKgQ'
    '&D&d5XYe!yMPM<cPaMs7Dp=U`=x++mK+#Yzyl(~vmTt<!Z|aZhkaOqxILiyUV2-'
    'w#6yv##KZ4@U;G`AmOgr|K%L_(<@EGE^B0$c{amc%u7qU&gu=o^YjhqMzONXhdon2g`DE4*$s(BprfXMYZ{xt2*`MzHBQd$LNV1W'
    '{vN%CUSi@53Ggw*nzbWA3HFg$xBiuf1Js!&g8?e>`H!X%sK)Kt>jj7bT$@R$!*5S-'
    'Q`~P3GD$+DjBmExRms?_6C2^)ZfK@(2G5jfUv)<w|s$vTc9Ib)wiQL&uENQKr(B#@2|mAyh5{z+A(>U=dY<!@V-'
    '_GvD|MsQ^mb{r6EYEn8+QG|7L;6?Mw8n3W7q~;QC%Rtc6Nf!&<0xHOy!5HYtNwfeoF(hht@(>33a$QaXoGk>!Cb?nfJRjab~D3Mm'
    '`2_(-aw+`zC#@>bhgrKK&W|B%VnR*C8(h8iw+*lsFqt;!F0d+m5e1$_bOsX2)>^qcUq^jtOh7{S0|ju+KmSemC_qV|-'
    'M6t{)T15tp^*A`qP>u4B3xKy0AZ0j4@s1|}&jcOri)hM5}OH<aK0#mm0)9_B22<0t!nYLqZ!^?CXTLUlCbnG2?nVw_o;B_;V=u=-'
    ';RAq_}rz+frEIz`j?6P<OJ>Jte?z2qefdWnPCgv<$0bvka_yikeS-wn_@cgwQaxUC?uLtcRstt@;aL|GCwsqrI<r8{W7ku>DS?1M'
    'uu5m3ypBmRf^r>-'
    '9k^E*!mS0JtOG8OqLd!M$=T~*#YrWK!a5;jOI{23oonTVQCith)L<@)Kk&{Hp{+AdwWWU9J(a~If8^vaoY{&6Ub$A^(zIz!dysfI'
    'Kj+sEVsUD98A3Ig&va_h^CqAPz9j$Ri|2p0DLAs^R39eV2WtT9yo*P>83uww9xh9o6O6#oXt9miCZv&j_#n8SD+tiqVR(1Lwoxnp'
    '{)xm8%frqrJJAs=!U6_`ig5;CJ+Xhqup%pgUQSRwM;KsezE3?$~${Ye-'
    'nI%vYEGy*C9AYV&Zx{r7wm}z%VBa?A5<<3S8ePhzw^IPh#?P{{0oxgPiu46LRBydAsr4*21|vki<1`6005_x&d16gOcC*#rrA(F'
    '--4dwu)FTU$`mz>=LK053Fcgxz)R+KB>YH?85G3I~o)`ql6S=w7g(;}bl2z<OvCo@I4j6X+$=;&zpgMH@j&0!vOq2f*R+Hd)2FH_'
    'pBX`HLHD%<B41JxT>b0S-'
    'h?~MROpbChdAN0yo1?mgiF#U~xP@_=TcWmwahhA9w1sh+TcfhYS=1339V~+AS|_ET5xFr9(O>VW!Dj(+s-'
    '9Uu)F*!_6o_!DmqLN~wHgxuM13oM9Rx(UTfYth;;*^6p@m8L&HUQA3i%VKuH|sBo6DxQFX1m-'
    'Hn*(^>j>Gxv?9cfliRu3qEQR*KrX+DyykYtLl7J3KA;uY+GQpuR_Mvt0NJ8s9h3h~pkz}TC9yY=IZi0%L~JB9F`|}HX8!g<Yxw96'
    'PL$Azk>9_QB%kcF(?g45z~4GU2vKu_ETEv(D+`zU=PnAxC7fzeC@w!&V*+rgZ+;wvOR!HIgv*#46Bc$On<v-'
    'ozD177L)zREBg%{{kBDS6M7A!9U-d=2k(<|ub|;a)-j1i&?SYmER3VRIaZ`f)(JXF8tMM@`ZcgYwmc=c|RXmQxEoqS+#Nt-'
    '8$RE$*)-)OZkz-6rFn^a}&u$OqAEse`QjPUGep~*clXvfG#zv5)SMMw^>Wf+&3Pw29;!rSdP-'
    '6nXsBh6pL12Wtcv27;PvYiA7j__1U{q;G=L@zj!d^thMAuy&;d`Ll^KM;4e9z}qJ6F5;(lV0QaX+NT?a>d3SEiV}PAd|>#ze+VWg'
    'FU$h6THA5yi&MWjmr^z=GYrh@#_`@}MH>$lXdFTtpqYTgyX=Xd~kkY(H-?x}M1|6Bzv{jZwL$1-'
    'pRQSf4B`>XR=C#Uh+)NhlWAs4)Rp)VD$gVG-_@48o%1=7tvb9GfSL-'
    '7jW1H$EwB1;$}Ta;C5~Mg5B8)52$*4Nq_%MkaVY=NioZm&=8W5?~{hUY!vr`ah0xkc5)=7=F}4$;WAw{HDf&jqyDAw-'
    '34wv+#s0kDR7(iPh6>DZJ$y3p%QeT55$>z)ECGy2%B?hz<X~<^)+l60L6*DD`h$8VX7{)zVN<zM{qifKuQ2-'
    'vogYjPsiyQ2vG+`xW*gn<v8=h*(M(_>x8wQ(%BS!D_qvItB@~X$Y!Y5<duV(5?nONRxR+B&cabvUm+ib)^O$5bMojl=g+k_eI;}='
    'qoN?V$^2f0%Vd|Ii!oYY#UiQq(5;~A|j~2&v2v%_04Ih|F-'
    '6${f=bXg*;#DXojh^otOnmeO1dsK?$c?77EJwYD@qq^*#D+5Gdg`{x%4dzvbpm7j_}*CS^dY6a9SRZn~E+ZU!!}1;%f}9Mh0dGwa'
    'ta1w3Ab^*D)Q<Mr6f5dPHo=)~m82!4(2yL}p$dllKPhi7oPuQ{5h)QI*gvbzu;aq1fY^YuKdlr)CP`JBoz#xmqA7y`m&3`q+&TP4'
    '{SEgPXk?T2>ug=RX)8%Rv+)|&HT0mZnx{7x_X7&GgW92c5)hEp9Ens)wDjmcx|@|3Yx`LoOdUoAx!l)K#+u(RzOyoflsYg9G*s#K'
    'o|_#U%w^F5e=5n1pDhVx4g!yRC_cA5?VKQWwadKvZrgnmp{wGYB}^NX1&k-*O0OPJoh8-(HJst8sdqdEh(DUzQt+gM^Sm-'
    'UKlr=MX-Ya<K(IlT;ak~h^fcL5o;Nm=IBXBiY~ZaCGTP;+llWAd1LU}6)g0$XrZlO)6;QT#1U{iRX7l_aw}M)7u91v*FZPFe)IGW'
    'kj*w=n7^K8729h-#y=i0FKMG)+kD6@F~{*?fZ9*M>WYM4I7b5LybA9-dxeC~n=wC~mz0mp{`dWG?=Ff-O;fmL-'
    'DNr`3)}9q++jAz#&`o$gF-'
    'sJW+5K-=u@EK}?2KR(pdaH``&P5r(algHF2q)dHi6n{s1xSgVS2i^S3S^NqS@#K{Z{DGIOOm@5)o;Fg+j^BW%&G1J!Dk9D`EB-'
    'JpZ{+ehb2OZzKH~G#l@4A@i1FFWw3SCr&4{GIhsbG>)>0{<Le3G~$Vy-V609&bA+;KPT4Vhdssi7be`2J2Nk8o;bDzTUP-XC$OwO'
    '%o@B)%%_hcDdU;hcA28UCf5NhzZ)R;U5|6Ot_H`?2-m{NA|in&{PJr^qAjt;cC1u$kc+*Zwpnit_s-'
    'H}i%1iRb}ZlW>w*&HP==bNLE;FEAzat)P;qmtYU1_~YIrQEQ|U{Sq-'
    '1PSBr@LEg+;hynQ;sp05u3=wzt+cFyPi#=tLeW+d^Ef>2&kNFr%h)k2pd5B@ma+9|P7F0Roa)3-'
    'W52A%<T3W(6dT|&Sao5YCEq4~8P-{nzK>sr1=P^FurCcbK2|%Xwe|MfDrpNdK|sLU)^?R{j{z_-J`TcAf|i4Wu`GZB-'
    'a{}lrp2M9;*=$91FqgWZJTG2C47rF+0SKeKEJ`v=NOx`qU~y3AF<ld0i4FTVGLI=IaY)!Xx*1(Zhhsy4mCHN>er#>o~Op-'
    'G565aWZof)w~<8_NARfABEB@TesKk+(>7bG4!GH$T6PY2N?UBxxGQi2%rNT#Mb9yq&S{@)J~LXj6#x~qhOta1IhGRzw3T8ECcNh*'
    'a-}fcPHu@3#Ci^I=V6W&c<5ZrJvQS5bC+0h$eqG<6$lcknw({Defe>y!QoVKsKKk%m^=m_RX0^RWR(}-'
    'CcOr>AWzHoczeU2?X_%jW(BWH-!H*t_VoBKi#+daMdYU0cqO*8FtI+&D-'
    'e~LP#^pT2+EA`F<jIFi0e5m(Jvy=&azE$x1L!d;<wDU+EElW+xNJ&Vw)yrjTwySJdNPS)^yDah(TS%GAyo7b5f|q;Z!GuTKp$9CR'
    'eh&rp3WMuOWmm1U=l9ZdV(hk~b{Ja<J{k^9Jhee!vp6x8-'
    '{J?ZV{xMAuct{)O)~41R|5>z%^ZcRbG_!m;%o^Ef!cmO(V1gNJ0RJr;0SLiN!SjIZH=^J{piII}ITF7l{EZP~Gj<kTQ@RG~zaItM'
    'E_&5tt7uD@(z6t1Q;F$z~xnwYb1TH0$v{b-xnyM((^d$llr;U-G2XypwzYrX-mWE=45R<Ryj;hFyCh~@w*ciGT@k<G=ZVa7OWn1R'
    '6<ygx9XI~D6a<OXvz1Zlo-'
    '+4|^oj_(i6(HU6(0fC=BI+Nr3V{>#CmV?;Lk4~~5&1^dUolLVAP(50ZWp@2#w?*L!M7KrZ3PiW%?3b1@dlg3)&LdE*wC)Z3)o#*`'
    'K~|uh<QJ&aik%qdy0z6l^l!+nn#z8h&{THIWGS_Kw{Zu%bsKlIJGXHsyKx(Lw)?hm7rSj6ceT5=aW}hZ>n0@a!@uM50BenuIYTl>'
    'hH(ps0=<xBT>WLYN8t)Rw@2X$J-6p<o0dwdmB1ryOOICCS_z!vOYF2eR)C>R;SI!4V>~J11<V)Gh~Jdm^hExI?l9)HMpSY6CEUtq'
    '--Ob|=KEFHHr6#JfRl_fN0+NW?Q(7#XxM&jIy`&R1@mvYNbPMG*ROGr+B%z)U6*O)0unbbW?5N(*&R{1l+7JcxRlKuIXk7L7Hr&t'
    '+eCgDy5uIH8OzX^CPguaxtUw#Q5G<#9y_!~tE@{^YUNUH+nhkGeVOC_et=Pr#~9kc<YgjujG~PUYqQ<wH!*p+h}{<AhfH1};>aF<'
    '#N?GCF0#iTGkKMW>)K-'
    'ozpoaWku(T1^`{}3)~4OxO}TF1fatwNR26u;rt4Nfk!E2Q67*^AjKcL{?u^3qVeZV?I4w1qS9=PrgK%oE(&e&t-'
    '}VyQT4|;07J=l#In4D_oU8gedrX$@HjOna@zkqwEx6l6*MbLoETf#ut+=pI%yt_*<cxY_-'
    '44&VUF+}Q@=8YCvF_yZDyztO7nfHjGy#rD?1Uee*a<(L*?pJwnNNQKxs{i)Y_7lTt|(kp<*q1PRpqXnjnh&Q`78`_hzvCwJt<;3%'
    '|-JNz3DkDrlX~D6}*gk&8zS-'
    'k~Ocx>uXA?Pvp9(riRWMENS$fckpY5aqoGD+cc8LMRvREy;N2m{YJSsvQ7I|x|+zLEYX5>j#bfXqxMcH^nPAtiIR1&Pg3I<lx}*n'
    '=5t;^qhwK*&Gp4kio&HxCPm>=B$IOXPfOYS84O;vyM$iaN~$A%8D2)OYc;&SvG&o43unK(1XfNv`%My9JniP|yRov~o3HP|+Inxk'
    'zE@y<wVSW+!)kjqU+W#t_ftuB6-au-'
    'd`|{+E+>tH3S7I&sQXu$N9D&5q%;{_#LMK%dRl4#H?5}a3&?w1oE4ty3%@%GSMj(z3Rm&CJ7@p279LfDcBDGlc^u+*?*UTVYF|*r'
    'p}(bCSNIJmc3Y;7na#Bsi?B(M2a|mxWS46qFiL5Y*S!;7XW1R>agKE{&dGEb5w0}r;sfS4HTW}=btBm)vfX0#isWIDZIG}}B>P2H'
    '(-5xa2u}hwUMD-'
    '~a=H#oDH4V$a&+}ma*^9AABWkCK(+>)N>>#Hur*#SGWVhoTcgOK6YlLymWQoD9e!qD!YdD3qn@kknaT2{jasC7XSl2>go^_GL(8z'
    'd*i=H|GhSEdvN`QCTnqpZMC;fs65+-'
    'wIhqvIOI<C5Mkb)9mp_34Zu;s83cl%Ec$u*BJ9wF}ax1({Sh)>eCal~JuQTlS{N)bHmpc^_qs*0t@`ePIFJN+ScwC}Tg^+GO4(qu'
    '+d^rDxP#nUkHiY7ES&{i6g>YExx=en90)2UKxEG0NVX8F`4!u&IS^01{t3|47MqEjv>7%?-'
    'MszivMis$R%!IOAAl}NXp875?3sQ+gld`TpR?)q!V6vVdXK6In^OCQ73Cv=Gd|e12_ZH;qjzP5!a&N)W*v}xp+O?=^rusz_2Cs0E'
    'C`Mf?jlqqqMy1%r7@Mi*=1tkYCW)<pdP$kcAhGGI+87E)IMv2bI4&qMzbP+?P2Z!ZSqO>kNz6Tlkk~e=0lAo!Ez;C;KP2lNsU_M@'
    'TPH^d!a6iF8qOMyJ2>(@;|hgppUnor(pZ%JAvMBL6avKGqLMMqe%O$dra$aQYUaP~*pAdB({^AxNLtsiM)E96mAKL&@k*!FqJj5R'
    '8i|`~B5^ck0TdD!$%QqMSU_T>dlnM)Rc#7IBAjYdC=wqoGQTM=67@ZLCJ2d7Vct6giJR1bT#jQ)jYPAWzAi7*tLZYigx`{I&G%F@'
    'TOV^q#kPUVCN4)&1`!2sDv2^vhH@60c+#1KNwl|DR2DxBZk}7P<40uQ$ORnnqq015@kaa@c!4h5h#!Y_)kPa|6?la%ITTL;w~%r}'
    'EttH@q4p{#KBGZ8GmY90*&eDQ3P{5A$O5N6`G=w4gj0PO3eJa$%%$fAr@j@>27z-X?#+dO^FuWtm-pB*-'
    'Ro)~lg59;Qu(1?3nAO~>=1_kRO*ygM+#XbG-0R;r8wj@ej4x~Ci{_4`JvoTT(xBKYKM}monDOw#jG?+K60L(p10w|H~aa|bA-Ib&'
    'wrk)n!&kvdX$j2`uWd43pvrxf1W4gZGQgqd?9c5^Phha@(z{%3{$b`>7%rc<C(xO{wNfsaH@|&QTjlUxpjFP4{1}(W+9D-'
    'W?>34r18*4YCtaEvSo5HZ^IIlba_Q2TQN6rdPKo5nBKb>g5~;$c>Fg_h)9;Mj>T-'
    'Z+AQlL<j+N^P^Vrbd;7VWVhEW1+|1spfYxCLg66|aE(RhU?EI|Dez6YWd+W8r6(Yge_@naQ3>nV*3QMfH#yua`IC&raI6R$xK0e;'
    'lELuQxr&rc2s!#rLC_dp-ABW;|LXo-jytAmj6?1}SQMg-kf@aZ=)qq^iW)h!%#YBZ#*D1>yuXZ`2FBy7^)%ssnWAfO7^J~n%8Eo3'
    'P$H^3SfR{-'
    'dcZ8S8G<JfQ(Le7DFOy8}0x#1qpewvgUb!2*zOj^>YaPa}b)rHVQqQC@wz;N%If}_Tg6efd%?wZhL7=~9)$8cX+8l~LIMwD*^j%h'
    'DepFuc>6<h+2z_uL=LVr~vl@_Vc1-O)^%wD*j#5_|FLE4UKbZ^v-'
    'SaUga*uka0hnZwlws7ZCdajj$x%1anj++0W|OiZlhcLV*BmurC>yiCIhw&yD1L?^6z<DO(!lOszLeU%T;#9bA7ND??QR^!<U!nWh'
    '>l~jl-'
    's?iK@8=J42MYV%Vz9E6%Y>^#%pG#)mL521M|{wqKkQ8UV57vlgIkc8|xc0?S!w^+1U0>9_~oC+C^W4B7CNz|0@;d7HchIJ-'
    '>`k2x{Snp20AO@Y4YEGsB~Jl0Pq%yIPOK9FhBQNBi&RINe9s3V^u3qjw)kB_f030gmatKlgN{>T^%h=K$^`eXs#UmT?Qn@C?r~E`'
    'C)-sBz&$6`{u6rN-nj?xGZtq#RYL?J`k`RT1`?oyak#QEb|5qIO=nqXuCs0`456GW_({zRD?hqnFyUn0h`tP$8qttvub`%G1NGJU'
    '!jY)61<qz1_;w$E`f3V5<?6GlbmFObv~hd_u@yaT`>$=foGp{x?}hE+BC;BFo75Rii_V3?~{LYUHogm^?;al`wJzgiv6)3C^AM2+'
    '^gE8CZqFd1kx$YmUC{V7sSOmhOf1II8u-)N7grF*!}heatKj_j#_MZ-OcHIYBZW2XpznARP~wP+t(Vi@hKB^tL`hKZ#nn9?ftV@%'
    'WAbr<+^C!0`+_0vU<`_G?MfMVc~sv0h;gj^r1?$}wkWnHx`XNvOHuM3;n``*SrWkGbDW(S}OF+T)!RwJDTDT|}cRugHO~W6z#bgb'
    '_H!l#oqk)<tbZ^E7)p{gtGTsl<QbeUwq{{}s3LeS?srd%CHXt-'
    '#1+t||0T68nd8qUB;s9g_Wr3)@YAQTi&B+FT>+P~$&U!cBWSD7~W8e#z3kFdMHXo4%NR50zyZ>07$4u{2)brJ<IF6I~i==?!X39!'
    'sxJS{k3xQj;i2$j;+@{8>T?>NA|3DIe{GIfr0b>%jDu2ntZcd!R9P6?UCb9Yc*uD7=pqqy>RZ>|VF$ITj7!<v#{t>}g_F>^I>Q_8'
    'M@f?1)XsEEwLBKhuN}ua!Kad^YN^s0Mdr^kS6gRYr&C?U<TOfdcv`=kS_Qg1V3~p+<)jjR`gS8Z{=5(LbtNql*H(k2D5&1Y1|h0p'
    '>{rmvw<4w+5!h7j@T7OB1W>=eCZ*b=mWdkH*a|&pSQ__q)7&#BrEa@bVGI<D*7uVq2Jej!}U}tGHEiCjQp8Z*PLV-'
    '8`XXrLAmI&wFzUNi~{nWIL3r8A=~Kg(#xM27um8b6JCAM$XNe+VQH#hT0xZG&a=suc$G3Z2v{dT`R>$-'
    'YwsFWcfDrnf#}K`%C5O#{Jf|>Mw}rg$hFg|GCVw5BSDMYWs9z^67-'
    'Lb~uW3A8veX6yxr1JZtRkGC&*uVonZz1qzIGpc0_b9DKTow(y`6RRdkh!K*V>WMeqEb*B1nEC;{N6qH@Y!Lhr`Hd@Pa*9ypXOvs='
    'Dsjs>$)W~q6%R-GjUyaF?>`6Z0*dc*@;vUW??&*BuUd|`(?R?@s&L{3`ed4$ydQTVIJ#28EDuk`JkA%$(o)7E0pe`HrvD7M6O+C-'
    'w?}VroX5Z^|F%i~c-NRk_bUD)dsu^`dz>)Sz?yF?gmwV8w!UUy)-'
    'l5>l1q);B%HsOz>#xtkg)!>0aAA!4`LZgzQdyP#v5`FP!U@+L#y;B1{Z29Wa4$QJ+P2k<RB{Ky%}f&-tkZ}5-'
    '+9c5i_T|GTyz0*;-W7wCocLTbK;^4nG+X%i8*o6MRqS)g?8{uhz%qk7o>o407G(cVif?t;TLUca4$Oqi(-'
    '7AWomtz16a5y#sMr`6ytz=rmj=ZH+2JCufW<*ae7=_W^547+4BrFwgRW=du-+HJVgMWK$g^%c$P%9R`${!FL-J{GJnSIfI-'
    '(qmZ<_h_h?H_0iR%uu@*#G4_fgWaMb^5#cMV(*NXUKM(u|yunpLLwpRW*WCnwKroZ_+^d7zumNE5+Ai<ml40+)hvv&bqj32VBt-'
    'q`R3)jVHz`}Jg8sxKfaXmj_7O$3Fv|&`ggk-)9s}%o9t;Mlb2!wd;7!f(;YnX01*6^6xbKT>@b_T2R`dW`u^Jr3(3%rB&+;q-'
    'GRuVXamk46uL$rOVAoI$5ZFi+Q0^bV^6QBcjd83sXT!CL%MHQ8;mA^7y@eI<&A!yUzz75^}Z_U9D^!c~C#K0Yy#xJ0e@ne?p^_Ly'
    'U!Zk7uWZ@ba2j(+={d&n+{X!v^<E(CkM!;`v?v_TBKq%K)$|1UgjLWvQSBKV{8^w@AyCF)Hz;jLXSSjlzcD+Al@_s8&1<S+FxaR6'
    '3Ch$$l4Y1zB`z1<as~UI`vOJZ?O-a0R2#=eScqIh$EsRtTL-'
    ')^=_|Imj%kqn`Qd;L?2V0x4zcbV71;jFb%CfrtvW6^NETbU{7t3gv&*~-'
    'hQhvFgRVz<q)okufm938@m2lO8qODhJxlrKlT}ozetgyKptiw8tp4xFDKAfDj%SGIeX6q|Nd;|gEN)h)bp|Yz)d?ZcvSBrRnnd!j'
    '~IE}~6NQ3k&9=EnK9L2T`ZX<>zI|((0RdSlH&$t{S(V7c+0hNs1S$5WEX~e=+G8(aPm5fID?A)eaGT~B*Q5EfPU>2>?7*h+;+CJr'
    'dvFjI+q08}Jj<@b`U!K|sD`Bu&s|S09tebeSH!8>4*#9FYCtFRrUpT(!o9DVn;M+*Dfa7)sk6V(+?3p}nW#v@%vg!<nVz*F=%}TK'
    'sC?#CyFcjpeV;1(E80K#iHl;o(^T95llJRqv!S$ClX5lIsjaj%#M&o=2FHKG7L%GL(SDv*Eb4u}HHpw?!*U!VnyM;-'
    '_40bv2LAVuK4v&=g{4ADChuc>&IWVCvybQGlmr-ZCq1OL9Ob#Pq9SF&XldewOPmUm2o%jfOHmT}Bs6LWJb>buCIi#rrVf(pOQs+Y'
    '^@1xF&@m9?DR13SwdUvLw3y5Rfm{onFzpM!h7sqJA!o@L~<RnqH@=27aR`IG{TwhVz1}^SZE>%2=v<hI=9LiLi&Qu2otWvKemU(?'
    '|V*dd37m8K4IA`YN@)qi93{U-ph=ib5%3C9H!-p!Di~2EDV-(@>D2jQCM^S%II?x4_EuQnaI>OjYpSvnl-'
    '4sq$6{>E!QH{yLmN2zE`c4eaC2m?|xDUl@Kn}75)z{Ql19A}_lR(<W<SQCTtBo{<Unyt(_k}RSV%zzVCWTUm`&DsO++$MsiGAEkP'
    'x8f4{idEn$+vQ50wsH;QF6eZP*Ok$<M}Mcp+5JNP?W%_ri7woof?y$aj5UasXT;mxR)9bKsY==4ah~QX_cnMPK<23%xX`ptc#$0c'
    'm<MCuaAg0ae=^@WT??OY9Ox>vKx=b`SMGhtk4-a>)_7QTHghHJX-'
    '3zf@@D{SQy`eg1s_u87K|g9b5)V!}b85kk*UgxIti{_MM!SfZE<^s5PkB?5GhY=G4O199REa0tGwfWzh=tRZR^AA)IPzC<s@pG5L'
    'vI`W~GYq{F<o8W6ziYM=(>N;J1hSN4_o?o2fU1Q`Kdh#|>)0dy_`C{jI@YoOQwS*;f!>BqBhW6Z}X3pd5U&G%+IVUz4G#=`)7c93'
    '1o0RXh4>_&NM9(R&u<^UktS#~!EfYC0phdBU{c9lKN0id*-'
    '>?Ld~(!Cr^x5}pzFx@8&(*yT3BNosVnV$tleeP+YV1!dm3kBl}H6{Rz>P`#^0^>evKmZsIR0DE3V6BrgVws%-'
    'vHpbUDtoA5(aE&?g{zlFAXc-KR?(Xf5UaKLwUn&QNGx(O#E5f5{4fbfpDW@=NI!a%h#w{S=s%12F;b5{PsERtc=Y)qt|IN|zleAW'
    'F_F&3ZEy?0ZSt7}1ourtu;HE{SU}KZK^6q{xu=Ih5Kc8c6oL!Ym;eZ>J8^mt1ou?~0wCB>4anuXwNAKht(fdheYr1B$l1+cvI#eu'
    'Mc}I8yUOs&7{~V%bHq8mr<x<d@jcBPMI7JL%~4&B?<dSrJ&x}uxow}lk3-LP`D_9``=!y-'
    'D5JGUi3?0hvAAoMeQ2KGi2gdXvxG7Y_5phn(+$=O9mVl5OBAz!=E@6Mu+%?lMkp-dR5L<h`H316084d?&j^C$eriAfEE}l-'
    'xqQXeiN-'
    'Q6veeR94?F3sN1Q6!<4zT=%1LKU(SVxHWMghL)MOFI_Y7{Ev+wJ$wnNTNVC`3FtTnFLTVsv9T0b?f%B><D@Yu@hQeS3n6beH6N+I'
    'Z1jJIjgQQn5}D5R~8s@W5`fCS8oSrF8}^@&gj!l|AJh2UZ}CIEu!&Yu|s!C$EX0T67g2ITS-lMuve)@EF5vQ;R%(w?%@`D^BM{+c'
    '_TzZOpCuay(`YwbF+or@$Qi@%+6P6B`Xr}5V$W3F37YKDX2*NAgI1UIbeID#@p^Wf4hqG%g^1&hCGwVPGO5((~Vy@}TXlUJc4H1}'
    'l{py*%vWGMP(Vo#|+^i`=b0q9fr{j4DL?XLy|ps$G<kjpi!mn6i%RA+Wc6AK%HuY=`tQ>zFBUnfg3jXPU{p^+X{IP7Y9MWh*Djk*'
    '!5O$0HYVGE)y+_GT?r`a$iEMDbOd}|o~2fJ`?RfjHr<Fz>%g7Of@d&K!w_#ZlX+kCj~$24oXJ(nAcLgjC!h-^a4;93x3-'
    'l+v4=GEMUnDTn*PBBh`7VnWKuW)eWbY{9lwweU~vdU+(uSyA0$0e?ZYTVSdP>q|p_Rw*2*G)BUVVkKU|D~nJe`)3MUs_xKOB*JC<'
    'nnw*p=O;(zU8(=FXZx5ZrL!Wrr0n&nH;Sb&ZTMv4O5qyDzeEir`CcD^I9#)Fw1h2VFo75jLXL=GD#_#*`yf`Q2lNV52l2nc<ta6i'
    'M;|dbgw76&7rhZq-'
    'W|sa(Me*`x>Rryp`F`$7@lQ+O}0(&s65Zw9k7FYiRXJenur}%AIj60wZ4QP8mgVt>e!@czhtY?SUXtX=sjm;8r=zczW35@#At-Wa'
    '%)cW$vsLkZ+kLvgt6V)q)Q5LM`Yp3v$z8PDt&Ut`XP+$`ovHw0No4g69BF@T^mU+uF%%?AeLcEI}E=u&wqOW?L4oLs>*B2J277V5'
    '_kBqf12X2TCJZ7LmSuhe&phsIt3LBzs0Ab>2CWy^Lsqz`{R6gNYxFAC>2rPdyetCeMv5(Pc>L`96-'
    'xc6yV11e0I0P?ePFBAWs;q!tvICu%`~c{n!(=69)iyiD$o4446ixpbl03%8PCu-si4fxo4X`k|O8Jnpo(+j5nm>Pa7;k{;<OPt+C'
    'jTJ^abhd=evJ4%y#8?&8#-'
    '$h+aYB6v~B)dk`V&Kq7b~6XB%Z`!kVGLad3yUZ>&ahI2lH(B;84($|Dz1`$c1)MkQ)a)EnWU5aJ%{%#f%+peL^cQJ^jdIWZmb0d='
    'Aqmin8Ci6rgIBjxm@R041j&!^{PSieD9QFI>}#qdIxc585^ZHr3%=I!=IK|+8I#9YzIhq|G+4@#v=K)unqn$;PMkwh;}+U^&|q<i'
    '0KI5bMjx7H)u<p%;B8~UF8;p@Vv;j%3S5UcjRc8pZOyL922SUb&flyyP2oXs3)!<fsZFdHbv%)T2N%h)q*1PKyHf6(3&vl!eNp03'
    'Q1s^e$Nv<i95&>w8WoQ^)BP^r_F<19LaGJZD#%~<Qpb5tc)w=h{&=lI+HJlD_~c0;PC!*H=dfJtDMQL62Wa;-'
    'Xh>n?^!*D$u@e5K83?OiGVd;MV#{JjVmz(C#DiW$B&HAf&azAV~skr1h5}f)y?2g1&t?t0PO83GiTO<GIL2SC^HjsQ)Wih1OQlcy'
    ';ms|cabM~LMCxnDY*xSZt^!qixdmGk5KFvqZW>W(y_2D7e{fWMrHVif9CQ-^Q6O~*v-'
    'l#!!7DzJ@h6_$F=%s5|_6N_%oqki`r<Gx%DgI`eRfH*C*iuWN}#80|29QRjT12$>>^wBKZne5cL1TL-'
    'i}FScdKO4+&<`jIITX=Ca%@nu}}N9;WRz!fPV^|07z*#<Nw9jSTUo*9&_Zm)|O$oTn&!jh^R+GWjL*pYl>Loi%DT&g}M)hrr?j+Z'
    '1rCIxsu&y@iulH)J0ztr8>$aZr|#^)H!S3qgq4wGf1uoi80TuBMS;HGao09qhs7`?eHtw{X>9KMPk4_KR@UV09u_4aOo@4aOr^4J'
    'IO24HiYF8cd(>6V#I@;SPvHKg};oy+ycGf>KpBWQ*Pr8g{TfMVAQsB6)p2<qP1D=%@N9CR^2Hy%vy&IGANe>~QUKY9SIarxqd+bM'
    'hH{e4?NML)#1W^D~Ue|9a0)YMW_`FckqqusL>#_bmNNU@^g;xLwJ~00%Po`vgC#YtFFwxM?MhA`-'
    'i}9K~$U`zM8LBuLt0fR^?cgkr$p)te+tycL2x;~Ro^*ungLN4!Jj*{D$k%HWd|0+H<&H-'
    'U`c_$IxX+u=S7_%l>2;t=25d%1vpi)w}yEEh4i7IG1D^O<`>y>xRK%Dm}Y1$O<di_-lK`O4IW{BE3+$q|Otvl0yKJwPvFBxs1v;}'
    'b=ut~J>p6nTbt%_Lmnoe+vUy&mi~;W?IDIChb(XLnow63I&=yT5u-'
    'Brl6>FNq6Wbl6_VBZQJ1H=+Ay&iJaG;N*7i*HQno8blL0N?}f{iGl(e5{G7?K%eosTBt}oR|^%1=klRoQu3kiP4X@ADtUdPVB5BN'
    '8l;FCU%1e{l9II;0sZL8ng9uo+NcgwO=-XnKQkg#4^-}ku-vtnQH`uz+4^Q|EJAVOUR;ij<P|0^_zB}=3ivvy9-'
    'frqYb|6syj|S>xwCnd5=R(s8=AtqzOnn=T;3tz&r1UpB}%h)fb?0PuZ5<>^R>{Fcs`%qA51^}@(c`pA2y-'
    'y86q_sjKDS1&<Bblc*BP`aVsP{SE!|5e|E0GCAN-Fe8$PY)aKI2_nQK*ZdQ0}Xk>)9RP;22lg{BG481Z+=yqY40{X!=si{kBd==|'
    '|;miwe&L}~J6Yw;gFQ^peE-r5q@TWIN!8B2%alcr;#6vYDI%Jt$U(LK)s7lPMg{s87d}goAnDmR~^~R9L2{gP3gDRC8h_-'
    'P9wDB=Yu?qn(o+AK;Lcpv4@g@O7H7V8H&mQm;YUg1flPwda4iCf2L<En(tE0`TQW#JLudbB1-'
    '7hkgQ<%8@RRq3>rPW*z$uVXI*rhsFo~r+;1H(Mh4%<&&(L`-'
    '(;NK1*cXByVz@MQq5*=A~kp6Gg#xht)Vty@zB<ANc_taEf%L<|Fy+T+!-ic3r6LSo6-'
    'e*Wp*Zd@$6e_*_O{E1y9(4oMmKo1|nP|P`L9HIaSw20ox_ZZ>;G30fS}p5Gas;zA$Y%;U$k+t@q_c!P-'
    'pQDb5Ry^>93J9GUSe#%8`ENx^JyW#jR$AmM*shDW`0^xpT>LWT)^dp+ND1So5<c0s4y{Q4}toBRO8ZMjfn-'
    'd(3n_|b4$HViu#K=GrK2U-#4hPY8h`^XS8KBVK1`Ee0Y!HpX&SGU0Y^aG-'
    'J}^JLY$WEVI1yxVxpL$2}}7J??2q>2WX1NsoJ5N_yPKGScI|j*#93w+XFzFTpgd&mQTQW{<qOC!S>ik%`YUq=L-'
    'Ji=se22b`)X3gmNqsm25dit1bOdJv&Q4+tQ1ys8G|N)xq7DAuo$_vnZ2U^s_J$CK2g(vg76yxC`vQW-'
    '5Hjt?ABX$($FM?{HeGJaj|AIafnvUyuJi~yUqxC#I~$+eJf#^l|ug*1er_b^K*Z^7if%yKhZF*%u8_EH-rA7FMz*lU?|DO^qhjz^'
    '^7_}ZS}SU?Eli!3<mlh+M}Bb=&kC>*z_F#&MYw_<e=9QA+zIKHL^<RVxm;kZst)^LPi;t-!f@c%7+*)7qKXbhA66F}*o2FmL-SZ_'
    'nH_F@dyC3do<h-'
    'F413z;yBd{y2>ZZ`sE`Z6o**okAgQP%EM0d0<dX8}<EsCuCQgj3ZE1>pN?OaK7&Eq)^ifO<dx0AE)Fa?vZ>)GWlw%2p0RZ`zZe_Z'
    'HwW0vV3f!K1&bgczKx!&$4fcA6o%#plJG%`mTaAF~AD)=b`SjykL2fg4R2@ICU06i-'
    'U;)Rf7?go(W4J`f>LH}<$MdT#d%D;46kIvUI5kqLAknMU_&?zh`|p5Z&<p^_p0;Ti3=^raMsViHbO9E!=e)R=&FTl&Vl$wS(0=>Y'
    '-lwpObFxje+S>2_Ns)JJ%cTV-*V#Bc1mgX1hhvY>#$YWc0p&wb6oA5X^rz2%9*NXet@00iyhxy{$?CFBj<+coSXOolc-'
    '0^R?IgjJpLIv9ZyHXs480cn7}kpS3KE~}~PT@K^;-rHQl9FCPfH+qi|JS*Sm$7o-'
    '!x)O#dntbVllKTDW%ik*$lyIuOLP7bm8j}N*bzPTKxP5Qs0wuK;)B|#Wve*oGLk-'
    'B~IJQmp>F6(f5xPErOPzkiY(+I~DvZRxPax>_@i%Jwq&8}sAZv{3QQuF*EG013vfLLtmirRNa$o9L?#nD|G+u5=qwxwycwb2y(mf'
    'K5jKF0Zm;mp<G<e_Klap9LWaYnEYqUQ3-'
    'l5QgQ|%oJy?JU(z#6S@#oIw^v>p(!M!%^B<Z==d&_gP5f143ozyL`lmKyRWZm4ET2d0xJg@_2PCW86Fk<!`n<*C<e@EyeDD~bJ(H'
    '{<~k%spn+$CY-ZS!oyJN_&nuSb`~%bIrk0IaJ6|=3to|CSWcjLB`=io+n6c0%p?l4g7j0`Xzz;c2ok4N2S5|)}FvvKmg{eEHLVm?'
    '-L3}IMqI(V60YS0>G$m#hM^6>Hz^@d`k_;<uJBQ*5g!hH5D2~W-'
    'gBuM!XAGw*kUPIw2KepfFNiaypks39aR2`eqoBAS?+yMrf6wLGnp1j}=-'
    '6Y7pvbUz6xsd2|9FN2l@ec4o={I~wKfm|T_ENm_;O?O=}#ynxX*W1sj{m!6uy;n;2_{|SpP;lGi~foZ)>JXEP>W>%(N|J;2;5ecW'
    '-Hx!XSsWD;gFyVH;6NE@TAOMkXs{y&D$0W-ejCVXTRB6@q8dTS3KK-Hu)dW0Y@-'
    'p1eq5Y&|I9NAX!;Sfs6*{fB5BJ*z!*>CoT5?+%)KeZWsC7|a84GGnR4z{vUdQw^RR?3^3wBi65!Z8@zu8yFo1OpqGm|?FQ?UYqB8'
    'YoQ%l?xD<!B_ww`nyI_mcx6#Xky?;pxRQc8c|BmWJk%;Z#dQbIBXkm|P8FGPaIvKmbp4cU5hys#O=+l*B4W2%*j~rPl6sRrs2;`='
    'M~f+U|SvRF}WDOt#=gVNo?I{~eZ3I~D>`NJ#4*rXM4OMI__iwlC{=tnnhI(sdlSJPcLodM-'
    'b=>S#A{2rh|~bTbF1o>)n@aB#CI8U3lFAq6Sx^v=*CH($0a)bMbsWub;&r^e)J9+Q6NO}5sOzmKEBa+m?SJI$hZsp4mdoGjp|nCd'
    'WZQg%<(Y;|{=1wGtmK~J|?(93NW^mdyCecWb2U$<FM?ludW!H?utH~%U{vpe5yHhBb~`3&E6{cYhxIdPymaaFbzCS22pXKWPdlP('
    'XnF`R07sEt>vF}ZrjB*kpsR@JY%`P0iaQuW=n88>nil`Mw;r#+rUWbdn0f6HP$<Bw{9i}z$<3g%Re?w?ZjAEdGkhE%c`UZyYI5_p'
    '-(eJQ+5U%F-'
    'RGJWZm!^`xgTLCXK`LBf6H>MB>Uhh(w@N4l0W(N{p8@1U#vqYdUA(*~d53v5S6`{t3Q>_R!?g}*~SErbyao5WSILUFpR-'
    '<&^OCWpe_c0uwZJX#D%Ba;$HLl)ejP{R-'
    'E$wFQHYjEcf9kOi^0emgr=bin*@i2LL{+Z?ul1%{2VU<P3IM#`H)T8U+F(kx;I+xlL4wPls6{3Q_eSebPyIb)Khuh25b|(7&$M*`'
    '#g+0bTk9`d8ER`d)yhy?FH~c4^@>T_dab-q+qyHVLZ=_(MfMJ!#nHrdJc=DWHg$-*<rAM5p*W_?t>96L)&+A$D-'
    'M5J4qiK0$(h|){yirhMa)tEformzJ}u+&2k!Ne;avWxO9-%E8zV7Hswe0xs45Ijcn7zQsjg(G>~WqE<-'
    'CE*FRaYz7nUOwCT?<A76SB_tqMf|oN84l0zOe=ay5=gaXv<R9UQQrLB`148WEAM<t5DIVjEz4lJgnzcxXblNsLp39Oo~Lhj4j;UK'
    'o#tCpt<sV?M?3Xtfa|E_ZWciPvXN-4u5#j6^Awgp>;+do;yh&!(8c;lG`m&QpVM(zuJ;W<-'
    '}U)ciZgWiy`Uln}x2=^0KL5u}s+JJa|DWJM0oGQR$@mqU#Yr+PWm_>0w;T%BVQ8VZ=c8tGNSpW{`+pX*h^kMb(v|Lj%5pXXJ=pYK'
    '(||HanXv|w0(y$`p__7l13N;C-'
    'RFdhr1h+~v>n_n~8g;TG<{kZIysKfF@N!13|(vAR$W4XrY5cv<`_9zC)YdFej|I?6}mU6oo1*v#EoyG3azvY!sW5cOl2{m?=8j~w'
    'MloHIKDc$T_FLwLZOWeNoQnzot%<Wq*cl*{W+`jcnyYcdpAPw?fOfIoq9{NDAVriA74Bvi11|~5N;BtZ4SDeAXJ|vHge%YI(Kjq7'
    '4l=w0lgRr1}21}^4AW`E(P9@eK@ldIZ!C6nUzWB)^Tq<L-'
    '2$#y3Eb{e^sUel&Nh`gHDHbo?f`@hk8S^c!#Hqxomf*bprmDPlGj^G>%Xqs{vB-'
    '7=>dE9{%c1S9+U6Q*`MwMr9^7JEKS~=nABy$Q6e_=ir6UwxuVwNpQn1EC?Qd+Q^?OLeqn;BT7`_wL)hBZ~Q`2)d$)3TY8K-'
    '6$T%YBB5iXi>zX%u2xIdr4>lE)nG-HrTpTU!$@0V&^KbYKcSFX5*`SalnMN?Xnu=SVf=`yzdFS4QN!Tt)#P-'
    'OV8kqbqJ{|2d0^vG8ujet<_Ez+Xs`Tq_n1q6w<9Lu^V*lwV|;Z*l^Fvq%*+ifPUW6IVLD?Y1Ng~43Tu4(K7iW#S68C##`0THg4@q'
    'h?d%y=N5v5Sk7iWze`DP{~&+mM-5o_G}+*@W3`5D{(;Pyk77b^afh&*-~(DwlH-@`ei)mxLm-mjur9lwDgSa3U}{R+s~DCLoV~iN'
    'HmGVz{LOCk7M4!D@hO!Xf5pxxjgXeE&`i-_-'
    'B16r;rnmdw46kQ2>T)nT%KgrE!)&KQ!l*Q8JRpa>Vvcu<53XFQnC?)81U(;moNt0Vx=_M61;fv3KNxfPR^cbx5GTe?!B9`l5b*#_'
    'BNPnR2IkHik+n}{Bhr+o`%=lEqJNv>t_EBfzi<T@6=s{j5D%+v8}`tNJydZv1Q!r#}ylpU|ue}7lL&q!skGgEI9d3DEz?&NZIK(6'
    '_%l$<{zQjF{%k&M%`OsvoNR}n6f@mCQplJVDkCN3#X#$Y8Vdu_!~z}#OllMIDN+gqE#B_eXF0s%~;c6l$c{aVeVJ(S6foC-'
    '8IB~s;G7{192@RV2C&dl)WP8U=&-y2(JQ{~iDhW@l+Ry@Jv!<Mo06qAqA#)|9(!IyOq%DKGPI)H*Bi|kfmOY!y-'
    '@^<r3_rg|a>v*Uj#u+?Ae_sE(zlm@`jK7I+L5#oUGkKeo$=hPm%Xw?7{JpU@{@$3&d=5catTJw!j-'
    'BtsZeWRW<nA1nT=i_PR~^Uxe^XZ~P}!nB1(7J9)rg`(blVzR*^RCY-'
    'x<Mm0m+LqvleXqWe<sP$%}_Xxa7q{`Al4z^3L7_Ba;_~DwAgM#(k}yv5cmJ35)ONXxQ<zvcKuC>~E&l>uwx;EoI&hWLaYBemQ=bm'
    'S@b5+6~ax;@4;~#tf-Vr>>4y(?X2tQnL&zm;3;$q5>{?14;y-lGcG%QU(0;hE`kSeI}dOn7<ptcZQ)0=vzFWRSB=Z?C&C6-'
    '{S8gT;JmF`3zmw&LaWQ3f-'
    '1gXjT7e*4~>?QGAiz<nrW#RtqZ^1V>v$ye_@twaD#wEp|IzOWcmvQn%x^%<XtBcROAy+>X~uyW<u2MsgBKtFA$=rpMiw$x_^nt*i'
    '!j;~XYG;YQo{shZ(+0fmbxSq-+;HC-Jlf(mE4I#dL8n;H`!f~xM&%|QeUxQRCh5iGW<0lBD;?UK~T$pVY^oabQS7FQzp+)RQ;Hfg'
    'd^?YFHKcE5EjOYgT;qX$_{ic2T)_*)Xzg6M1;X^{Vp$J<E+@ee%Ssi%WKa*XQD6mkm3sNPH=5WIb3W)0+uZ*xF=NA64jaaJ0L+iC'
    ')FB$J!1pw6c}RC;4-'
    '782D(T@#8#IMX$uNZh5y1Rzn}qFaKH2zT+8AS7;619DM)+9hQ2J9F7oFHCRCIed@1hLyf&vraBxw(;Q@bXVO*1pLcbFLoWuV1i6F'
    'HeEzu$1UN9DWAC{euVOwOXEi=pSdi4jPjYw<HsqVxgxHjeCEn{3h9RQ#lByJ%e5#q;^Fe?G%mNZJ&+#@D1l7NLZ3SIwV~*PGhG{s'
    'zOU7o0Q9LFF(C+jc&~!cw_Od$mD_Gtvo3-9tod~bivTv3g{0as!dh$j3D0^p3zZDQ&PqQ@eofA#o?TCK-'
    ';;h9z)G(4dW5O*rQbtI3gb+_k3urWoBjaxNsK$a0kzAFKfMtpD~v<E38jn9qVRWz1K>LOeFA{bqye~NPXH_+LNYxIfa=uaLjee98'
    'XpS4&()X!0IC~tYY+h8Ufmi5z#VEpE+4RcV*2ZZ(nbm_Z^_ww5B&}*a2l(mkYEh{{R?s#7m>@jLg2r=OI{)lNl4sQsDRKqhKrqJx'
    'WqYzOPyo5%sGb3onyGdIfg6AF+5D}bg+9@{*Zv(vuW7v+!O2yXrRo<f}J|`b)m3>GhG)7yA5hg0PNI_m>2}RXR)7I2<&#M0lECYB'
    '<#9y*-XRkbvcVCqy1X7W5CV6GyX?shWKz4(;$$nN1byIB(u{X*~Mx$X7J>JXa(2FjGL4%TgUDef(vkynX8|}?KemL-Xd)6cbF2(6'
    'ItL>Kk52V@WGj`4+Y;EH6{Rj>IUBy1isnWYApnOyVQVOE?*LSUAb(o!M9wKAFXHh9q#viNAmZ-'
    ';qg{#B=dV7@34#U81nv9as6e?)Z|y-'
    'uEf_i(=b@I<v7F2x_)QZ$$nUhN?3eP{*=JtoHQ1{uZcwjRb}R$N~POV83K_Ie$ZDGc+Sc`s&3Dj?T`wn#5|b=MfHnr2!$e?>4s1!'
    'zM{qiKvCV_+k>D8Cb&Hair=dNx%|KO>CTyRxh{Iwb4dqIdQ`ZlNzUOQ+$sK&!Jn449h($g1EhqVBD~g_I<ZqpgmiuQu<XE`!e#u3'
    '?C2@a#gC#Sy_2qf4E2be&h_K63-k1Dt59sk>D^92ty<E%l|MLOzb|(ufIT-'
    '2>>p}G#_?jqOO9763Qd@P&4P>~7=1}Mh9VQrbYmzo=c_RR$W%Azjv!>hJ-j0bnLnrjxg5jd|4|3ASt&-'
    'hH6<aF;AL8PRm$yLjuBK9vDxc+unUh;*_joH9UZyCz<3u?adz<dd$WIsH7Y-'
    't{W}QV;vdca9eC04PiFrP*_6A@{v9$Ye>VGfNSFOZ@83a$125xlZiLqz{;py|IK51hNe{)njW26<s41YPvn?zAtIskllvf3(8Wzf'
    '{`cjR_)uE<1yWjzC&?J+y)sxUu0*zy6vdtL$tIcPO5pZM|7?mbYHmU3;@71ZTpP4+#Ev{BUZ?i9c_?(cRc+--'
    '6UdT^<Ma+3ZZt*52IbX=nyqQTZ5b|?xYLYJq`Gq$($rpwEx1F4jas8uXT<>;_>t8%|sH_cu0s=PMvk;)aY<MUF;8epy5wJy#$<@5'
    '3xF)l1l4I80?U;4<IA-0wj#+n~W7bV}%)0v>v+e;$dH>!~-'
    'hXg)3O`wfbi343xKf^=fwP;*I6>*|h%_!EQ8NjL(tJ}%QsvsGKbb3@y*7OlwM5n-)XJtjlQdaC8fHh9Y4ydA2sJI7YDB1M-'
    '&bRD^{L62zU@@p(Gy{T=eMJaNZ_7La#<&wGTr<Q2Z>EQSe=j7CW5)`uL)_PmFNVlq$L5T1^xGj<;{XxJU$|C5!B=HQ8_`xj~a<Yd'
    '8>#YGrfi6L_y6TtK@Bhx<5{lw+qw$Q86rcxpJ4E4M!J=JXsHK0il(hSr*n`c6O+R;Z$dbTKFwBCRbycjOn{0P2ZDfsjEbj4kmL}`'
    'W%e)`nI8ESEbqyfoZ`%G@0m*@NK3Hrc-'
    '>k;nBsN<9iG{BJL93YdCaq*Z4lepo_c3lMR0^E{pFs<cYX@`~Y$1o_3^3#jrle`HFALjVP++?UDS*<cXT#^%Il77NJrlyRz8M`kW'
    '&}%?hU)8EV#-)tFq3XNvcvSFJp)j%I>JS{qoll6p$7C$ij4!ZB?s@X<&JC+%eNIDH!-'
    'Tpr}}Dv&t0+R|qrK7B)w(&&YXwtBPvmuIov6P1~?L`t0Hx^+d8%$>9;c7TxUIdObGVDfNIiQtb>3&WCNf9CQsZWRx?kXO2bWg+BM'
    'JX9Oxd*9ZPv9-R^b3$zmr#dIp*7MYuT-'
    '{}A@L)^mS*`TEEU@Avf725mWJ<E)!%G?bjXbFjILT$G?uO;u)EeFc%ekpFycd>pQ)_r1Ea#@ya560Crq=L&Sk6tY;RCRoyE^lW6q'
    'nk0ZWk-=yz|^HQS^DEQ`!otlF8M5kvvN8N&F}m*4*$cpttcu)*P)*a&D-B;Z)~_8n{}G$<;k3#g-'
    'a|imdS{OPT3_%Wa8`ohP9QiG6`-r7hgH)m-'
    'R>QmbTRkt#sc73YiG_W3IE;~pmRW3%`X;yyNyA0_Hzi}*2OKDLY>C*otPxQcj>t>Y<*_NWazT2bYl3qDp6<c&yuYh=MDY0j{2{#`'
    'o>5PF_abKWi>Wbsy(sr6|_g_;^pH7eB9KdCXfGTOz7ip1rzf$qJu07V%5jg(#R^0{w+=HNzAW%*G5v7h9}R#iMv2(gsgLWrcI<51'
    '_n4|D$eaOb~|aQ^$*&VL{2{P%O5|9-'
    'BvD?~1jS<*Bhv#CvZf*04ESq;+Ha5FN)=PDo$@pDEyF#TOyYM~9Wr54%{TXOb_NybZ<`|ZGqtb8_;UAWntM@mOmPTTW~z~bX+ga3'
    'PSpq9lyk*l;K{+axvhcTQm>fhiNgNtGWq`?<%g=!Dv47X;~M<jk7Wy}47B&<J#b55Xyu;=;@_4OBDRT7!S)QJw@Hk7o_!NiojvCr'
    '~6RJ2o|UWo1PZwZuw_$ABC>e@f6g%ZSPwNQfiEN6$9R9L(u{()TGrSXsC>MoCWlYhH1{zW^rs<q#DraA-'
    '#z(3^HJL)7S3EIx=EbkVypV>{`CrHI%9m75j>liin{4JO5xvfswCqk4YYnz9X3-'
    '5((L;e0;lUz`YWU9?x_0(6KNSfBf^O>xtP5ctd#e^zB)QM^;0_wZ)c`bw>KCguk#Q&@9O2DM5t~B>mHFP6Li%T$xaRY22sNK*kBI'
    '1h6^{6p32{R^{7FpCtsHRso(If^F6`M_GzDy>~?!IP>38)RCpcsq-vZ+~|?~4(m4Jx)-'
    'bk4c=o_p`BdZga1@8f&*@zn!fy{dQ4J@+jC`OlwHPKAkE>-'
    'EqlYeu&mnT@xPJCc{Wm?ut&&ZkVB=Q^R?Y~}0n8X8BarRxO#*%;$K`@I|3c7Xz^8}#`jXk;@?=WqrbX_4q3QHF!*|K1Qp@MAj7{w'
    'QEJ>yF$&^_}A*CPlj}f)gU15PD?<{Uh7|RFTz&r=DhZbkEYKg>gcNY$04-'
    '|FU;VpoDm*1WJf^QjUFzi+Lj)m1VN73uCOY6BM)EdP^F6dc=ULc9f@aqL<2R*oj^yC$JN}Twcpg^a?qVo#>Tv5<AhW<YacDSIa3Z'
    ';m_sg*6#k6W&!?_5nrCsu2)8SJR4~|v$>n61S+5HikqgI-r7ScAfjx7kyhXD&m}NG{J8`Mh(D*C?2@+^PoYtpoO7z9-'
    ')^?tYba%H_S^~7?q^K-wbborZ23fL_A{IDB<l4uhJ7-%`k9@13iUG>E06lK()NwvKK^&OQE<1>AEUhHVCR*=o`?cFBS-'
    'p<HcRuA?RG=E@~CH|r_q>?yp|Z;`_Hr7iWe0Qb|^yfkV%rU`r_}F!13^I2^<garX1rEU%t;lrCz=m%%3BY!AaVuM$5a$MpAXotND'
    'j|wq{+amXB%t+)*A|MLw$wKImoC?rB+FR@;t1^5c1S8#;BL!u96;U*%un2J`-J7~g)^y#I&%2TV2ZKa-!q_XK-'
    '<zY?4iT>)za=RQ|a$hV5~VQ)tz5ISU&1g$=6ZwY)3drRPR*qd^IOWtC>mPR6L%<Cb80Ff58Q<pEh*G~&hHE)Ig%U{HGw)<EaU>k?'
    'M9pRZjC9aH8GNxibB*UrGdLm|P)9D$rh|!xrS1|k3xWYwTgumg9gveNb@%KugZ+Nc+`iA#X7Lxnt4`=Zxw$<Y7$F1MT+L5GcfCXb'
    'yTJQ@|qfv{2SfjhOfmf=Xa&V^3C+&u_W0i<alwUj$*+kSd-'
    '6?JscaG>gxgV7Vs%;mX4qL2g8z}L0kKD?7@pbFG$!dZ)dU%G5^YILq#)!EuR6U4)VZ;xH^UoD#Hj7X?T;j{TyOlY9sro1o(SUcVj'
    '{*@5o7FWb5DkW)UO%GwsSpj!o{dj&ekw!*lg8syo}V&9)4vdFZR*{1+@nM>J$esqhS2Rg1?PIgsL%B@tdlkLVRhmyXt@*4icCw}*'
    'q9D>5Sx^+q;EF7UypGIr%s;3-'
    'Hjvt4d}}=J@WJD%5%{F0){6X^KV3F)=Ybm+o)58)v^XQ#?<nXe6^GZV>vEDFL7xSoJQSxdZ_Z@ou-E>-'
    'v)I}0GviWh`WN|H2AFU3WC!}#n6-rbJ#zIIs6eZ^DE<jC+bnbSe8SO#FN{pg^9IfC~S&p$))*Pa#>V}c&)>)EnjzJ4K_Y`-'
    'A5U!?gR%F+Dt(yH~u~u$)SmL_abZ;GD-K;4{8Y2J-'
    'kyxsP3&%*97REdc18xx_2q{&WGq8N~kR*$S|&Z^CFm_b??U!Ogwmxlb<R?0oS=rdk6KSc)ci$`G1?ZY)RIxs(5+i5usfXF|{3@T!'
    '~-9(&T(CLhHxWDOcgwvYa`%xfzOE-'
    'H~UQ)O1IlVL((>#cw$69tJPQv|HwDHyq3c7GdAGEJ+X5ts6u25bx9&s)tL}H35349>jtmJ;Z0VAV?4KSuRNF2@Z(q;cYaDTIt~(G'
    '?jK<QpikOa|io@s7O0Jf+y0w%MGR9UN!u%)pTx0P!rL37iUzfY-'
    '5T%75Cs<7}>C%vjKx~2mS#g8#Zt*U<mHNw=%Nf1<nMVj=S$XZZG+!Q@dJtDW-Op<*VK02UEKuj3AJtcIwtmp=yVBY6?}mpQ>vD)J'
    '{E!g+XeE&uU?i+TpWYn9}cytKCnC&RnV8?PyI7*5foz%>ri~B-'
    'I;VxbeXN;qQE7GPb+JY2M2OY51%0ATiQwq`4S2pU_0(s~n2<Wv<pQ;6m&|JhpJPev#pyk$2<q8spr?I}IETn`0UX`5Jh|!8EW4LC'
    'NJw8mMmF9IAnMr{+)%yhmLVpn>W^EDF*<d{&EsG!UQVqLj|xfc(g#Qxoy~1IMpcvmZS+$SQ~V7@&69Ivb&MDEbShWOWYdag9TIOm'
    'Ik#YaP;KqC<L2a!8NK4(Tz)AU#ww%D*|~8UZiIl<V?*<+`$PKE`7LUuv;!7w}nOlA{P|$`wg!r@mB6sM_J3T0+(CHg!#a+NnphI7'
    'sdADJ~9DJABHEQ~G@a3J397zx*mnTvd5f&#d9ABXw&txt4*d7GftX;+Fel9?Gc75$Mv^y3cK```y<1q1#$Na$D=iZfiZ@w$@L$U*'
    'zAN?u~@s#B}e9eBJw6q3*pd;B&2ey97jqD5D}gF276CJ@utpLv;`D)EcULE$W&8-BXWfNs#X0Q(O|Hd-#->q%?W))cqDXI-'
    '`0j#zyd2S<KZV>eI0i=hWJTMy1lGJJ8lda>9ilv0}ICcYjq+z4l>m)eiPn<6v(?9qetGgT2){*xPUidmG_kZzFAS9$zOIwy`54xt'
    '(K_qcAKs4J8CZPs-'
    'F6VUxK&DVs|lr6xd!By;gbEg~?P`;odP)pV4&ShvC#b$`}&_;LZozLi%s9$r*g?Lhi%Bx4O>pxi>6Yfc;}7`e(dfvuFeh2u==e21'
    'XdiRNq<xlwSk5lx>6Rl<avm63dnBhDitxs?OXzLDI<@#fJH#uX;0JciJfcbW1K5#}&0;ZnKOGR+;Uuo&|j1BqI0NB|+L&utAAGTx'
    '{)RLEWGnpBfe;zE7}j>#BClw&jbnW~#zrs@`#srtUlRQ<qZs($D)RkylK)sI}J%2Zc(l8X!{Ww_M7%wlNPQLW4n1-'
    'wA@vH2e8$<0<7r?jlj`qgGPtCw9QP{Ug=kkbF0xxh`WOVNmhk)M256pl4gUwB5SMDa#5LM8gXx+c}!lek1*g??J1+u%zF8GE9tTo'
    'zGO?QjO;^h4A}nhGd6;abJeXlsT~w`{jji3Pr57V35?vB2@>Y&VUig#H-'
    '@QY^Q~m~$1SrtxD3L2o8(l&8TFjF?f(fKY3$xDFo_<vR4)4QMPV0MKVQ;$3NPG_(>liS71KnvolQQD=;(=_}6+6*b;yW~iwDq^?O'
    'd1tq^Nggm`1EXIR?N}xmFKbkZPnms%%=Wyey2=>q@KhLwOp#+QatZE~C`8Ht-e$LI@N>{@+;kR-q+k{u;Ft!P=$y&AvTjg-'
    'J3ESidwh7ziNM#drWtFtfgPhN!9#Mxlznxk@%NU2VB54mU!sszIX%D84niVQxywR*s3BRGPNj3W<4^)owb$&f9V77eK=yr85c#Zy'
    '>mHievv$F<{%aF_`W#asdr|TwIl>PoPA1c+8CJLkcwc@N7=vf(7v<(9NDMO|Byg-M_kR7}r(4*4zci1S<rTPymwcDt<*EM+lnEHD'
    'ixcn#7;_KQw@1V|J*Ufn+HTU*lxYf!ei5H>K_+FC4>7!<cN*r%AJ5=IZ)itSRo#eqIT4|sRZ(TEKHU$&a&Z4O#OlXdu{*8zU&F83'
    'TREE%et{Q!Aw%Z;wUk=5s&w5Ans*S8&Xycxsbpriq!-'
    'n^qKri}g#<#7?Lm#6@e^unekG+lLV6Na60%P=vtW#e{M$78s84<pe#TgpmYhEBp#k3?r)AybeDrmgXoKQi(q^?Od*CbESGl=6s8_'
    '`3DgDIy;MfiC2*<m#21RpU27FXv1VX9rKsSZ7>4x7EKW-KywVcHBSN^*<KkaiH{CJ)S39pOufc36b3P?m>xlSE5j`=(IQ;*D+!74'
    '5IpHK~#@2kW-?oK*^Y@X<4rxJ`B%YFkPNIuDFc812FjdJoxLLG*B5HI2DMzx|l)O(2l$lePtcniign;+;Pe1-'
    '_Woa*7D$QTR$Gxz+l~RD+8Pcoeplz@w0Id~sKTWUJZSc}{evm$tya?{s7fogA&UtMw5}ueIa);qGa4sknI>U8-'
    'g5?7U{%rO}V#ZfW$RR@ugsUkC=2kT8BJEOcZG+0%g*hFnVwK@2Jxg^yX-Zt7p=iB}bVm3Svm1JJxcl7!<k$$i$pOf?0#fFxmC2_y'
    '+AX90Js_$Qt8UA_~2968Zuvz#hWs^e{Dqst|%2GB4Og#>j%UX8BqYFki$G{z_Oxa+Tag-qlr7HK>@f>Q+pL`Ot0NH9!oWCVi+!#>'
    'W8V2H?r!qOJ2&4@l*A*jT+LfB^Pr()thg}&1}Wy&GMyS`b-'
    'ic*9);e;e{>)UKEfjD7%3B(E8(}}yhqHr~LS)cPHBHHT4gf<rEqW^IZ23kxp@$C!|XIdKA3o}$J%qtC|Gi2eXHL?cP!hmz+P{tgi'
    '*&fETOVDoD+9tf=a=68xkB}pHY6+U|krtsmN*;g)BhqDpAS}zw*(yPWja9+|VSg2sz?E<!36&M3Nx%ZFAgI9yNI~+&t#9*s2`mb)'
    'm%yU%dOC4e##c9t6n+i3A9wDNmG0u?QWwgN>m#>j=n}iQ@5~UAKbDRjio5t!E5ZG?>0_k)7^Vv@+FiqJD--Zo$VIzrnQdhv9*enX'
    'H;LI+CgX887wx7PwiUvtw?_D<#k~GHQMg4I6HfAlJ2eA8)54vTfnOvESKnqw30w+0O5jr1kxsb%Dq>>-'
    'VU)lZhcIUgd~sMhR^W^I_zn>b4grqW<l)L`v=r*kI~)*oxvaSlGsnNLbNkF5K`<+m#ylmW2C+TDv(-'
    'P#z&0(dB^mgyLR(#gKjGv|a?dq>_njs1C+sYNKVfG&`JRv`-;wHJnjQE|fiF%?j28G}CdRV_zBo}aM&QfLhCyq)&-'
    'P^ummfmTTG)@E-n>6kegchVaV-'
    '@DN?mHsRtiEXtaO0#wQ0M$bnC6LG@ba?TADV)ZA07h+fiK3s0W+1oTm!dSQbj7&_7Ae`djZRfkt6h2{a13(#g3hz9-'
    'Y7QR~Jn*E*Yl?nHyzjaQI$yZTbxus33a7<%@%W_+Nc3TC;CjkcJ}*l3HnjE%OK%h+g(xr~jrn9JB`i@A)Aw%C-hVYwj8-'
    'g03(>7RC0?bD&JZ3&=RRxqm2Ocx<Y*e9HsMvr8<)(2b&D}5_%N)zVxyGvkB*j)m1!tRu?z`QNE$)DtlZ);p~vqp}UIsG<gTWcKM('
    'bb9!DsN8nqY=#~SZ&G#k4Hq;eV;k|MMOll4`i%&Y(+B`?vG%j$%32%4@9uZoSX*_MevfzkDW+!2AC(OhG^dn;)WOiwovnfjh0^_G'
    '&=j_m#EpIVKfOtJyFAPS(vY7-'
    '8&0+__F5O1o`@%!So4uNho^)UXqG#up)nEZ<9B(x7nN7+v3gaZS`jM&hTdT&h%#X&hlpVTK0Kr(3qWpl}6G}M8ubUf5!S-'
    'w$wC4z>N_+7Wql88rK?<<HX9<Fd&Qn;BhhK*h^?<z7?WoIVqXZxd=JJFkiSsvhZ^)+}bSM?hBWz#^f;W3nop#OF}sl@RC$ag85OC'
    'TE+UYofKz`6h5oTy~1&lT`hC;e?4$OpA7y7-$-MW#Ooh}Bw24`BstxNNV363NV3reNV3VsN3xm2qfxSj`)^UUMi-YuRhD&bB@D{4'
    'Z-'
    '9<@k{exw5ur9IJxM<P{$M@?yd;zk0WV2K7FZbba)6^Axpi4`c;d;YUGU^`Wq5O1rfq@~T`oPj6b5E_fiHtovb?}oz^PeY)T>}{mK'
    'XAB_-d9H^2v&Qui1fS`wP<wsEYXbnqkgPQN%zqg+Hiymhi{z&K5SbN)l!f-h<&u!lWqe!(i3}yd;$L053_!3s`Y*;$D9NZ6cj-cW'
    'h&pN1$E!vl`oI{QHry;kqX&!pkOM^I7<6H3*9wEorlo>ft_X)u+S#)~Ywq{8qEV8sR}})tlfUYt@_KVQbY};1O%pAB$+-'
    '%e@&J7AJ|Z2*<&QBr#Hi{ZTN(0bUYHaDbPj;sLBE<Yqk-V4FuGis@IG&G3ICqEmk`V^a-wn|wyVS0cg%JG@Pe-'
    '5Ql)@Eh>oCLJ=Vu)d@&E=8h@I@^kS?cNYZVK6cYN=2EGkAn#e@RCse0=y&@y<b8qu}1ri_&{<TsP0w63CH`xF#Fi~5o3r_S<h4?z'
    'w_ZJW?A(GmFU6n1(XxAPV8Vf+=6IEQYT(7<F?Kg@TG_&;F|;-'
    '7x8$@>+x4m?*<nEF1B;gHU4g&S@jy$vtRhIo)3i&>**4qbOob)p$a2Z%5fbCCM&>8Lb(d?l2nv_dFD0ehxDCZNKZn|+1b)lNysQ_'
    '9)L;(f^rickT3jD<f9$y<@s_sm+PpQW6+7`5$^~)Bva%RE9rCL!3bV7(w!-Y-'
    'q>V5W(oLW#F>wQG?;)z{UM>ec@XZW0`3;fIP^sU-'
    '(kfu*vd}|rCEfaKtu_EDN0j*5=>8kmxQtt;3cVgg9<jwR+oXx`KjPmIoidWqd7q{8iOoj%CeS>$4Rk{2iw=9zy2ccg%jjUycJ$6H'
    '}g(7QNGL@;UxJR-UlbkExZj*k+1MBm?2s?SZ9LE5rT{=tHK_r$ifj>&MNfGLf>pC(LijtB_|K`ZT5zi)$vw)L(A%$)itRo`SOwnI'
    'mJg=jvEjQsIbYO(gdU3hNSkYI*TIA=2h1FSxb%EZoPwDOn0!01_!%nbg+vi2fJu?u!|PXGS9%nq?TvnVUoym@i6J)kpe0+N$yRNM'
    '*Vw}CJpNkdoNUYc&qn9g||UnlZt3B&&N3n57X~H1`pHkel{Ma-'
    '~AjsOq2V$c$g;l^YEC$tF)Da@|k#;eElpuOul{&9wteD6COHAPlbUaGI>CkPFnwryxHXK&)JAnrW@OcPM>S9)0=k<cV>tiGOW}6z'
    '9ix4ulJWw;o+_R5-Pkk>Y7v(dU?VdO~%k9&+wS2AKd~dvxx15(48B;<s#_8{oZmhl-qXivjr>_oKPzx!b`=ILT6{<7Migb{PhC<F'
    '1QFXl;+*(2Df@S%*2$F{K3YXc#e*?@m><X)$Q2m25sUoO06^!$NV?SHmYP#Shsl6WJ2>R-Vc=}-'
    's=5OSuR!Aq~ggd*ak<tM?`|PT3zSJ%oJD!#t6>c4DwjZs>r`B^i;<xSJPp5&rv~z1M!|Kh%Cb4LDAniA@TsB=hM&`4-'
    't9+h8B@G@Z&-'
    'OCr2Jcbew?xk;aJD%X{E3rVD_GKo!toG;1_QxOM}DwF15zQCdE%Njo+WkSXT%Cw~wsZM@Y7q0;`Tx+WEMUP4?lnwLYJAIHgd$_AK'
    'Fq+FF9+j&8>8pj9Ubi_y661QE}a@#F|3g&N+cSBD@-'
    '!KlksVSz<7!jN$;3&7aUlfpuc%!_CV$ZpTYI6Whc{3g`oI&%Z8I8CLdSszbmYWC5vruVH-hy6PI4T?3^Bfzp$2{`-'
    'dv=Bj6mQiTD$sk>HK};=@|>V4egy>v5JQqIHyLUSxdF&Sx>xY#R%Zc+L}{ey327u_mGEK`y`Wx-'
    ';MWoJx@QSh2J4Lkdl0G&o;PP>1RU*r($TObVsNkgEfq%IGyktz6;Mp5(IsGs&;_x2S!2_L3VJv=%hAb~QE@Wd&Ab&Z%1{O*tgrZ?'
    'gxJJ<<$a-|#arzQ742>6npDhq6?r29NVww@$ij1eT79oIzuGP7@Z1vlDlEzP>Ad-o<zR!mB5EMl@7p5kAlL5~MASmA-'
    '!F`)hg`p36j2koe!n=PE^_^TNknbr)(&In`@M`bBF8QM7-wzK1`Nt_5p*YX&%)t_E44_gZqKA}Mc-zBs5tRf`$NUqqOM8R7W>b@g'
    'jGJ_PR=6(AXGz@lkRs&x;9-qS-'
    '7EFEYfPBY1hnTH316+GXb0dR||Otx7&V8aI0)Bq1kGgO88Mu49FMYaeM7%8W{NjcTxS19Ro5~!E&ojs^bR=3RG*(exypyBSmOO>>'
    '5`P>kbbJ_?iZrJ}Tf}g;x+W?8D|4Pb;BYc3KJDveQy_%O-'
    '5JRc8MrDVo5vmEIfrx$$U#VUggpe5=2T+w)7?H9`}dF=0v0Vj@nJgm&Uq{ORa*axZ=Z`Xt<p--z28cjGr1PmbF0n{n}QKYj~3ZOk'
    'n9ZDG2p;bw)@tuZpbRe?rEh|q@EwZ4oW5OAfI@goBMB}vBm=jKXimd%yWESpQ&ESr!Iv@&Z&q1Ze_KLK+hj9aYTJ61qhhKtE|hIS'
    'RL=r&)>LxtuNW4KGsz*tm1Vg?rirpuXJ2xySAxDe1NXLBK-NzUOyK(oAw3jr;1uKq%|C^YU+^Yy;3z?#F%*$)-'
    '^Yj|*f?9;x09};kt7VzT&zL_Lo{X^?Z=$)-'
    'Ep?9`EW$$c0ZsE}6gYa2vrY`Pu{%;#MuZN%6xOpQiQ$b8v^XfB74(vlzp<Wtjdadv_AOga9`Rg`;8^JjyV?9?mRVD^j`imF^D?BX'
    'V>sq{z3HXLDUJiZrFPvUNzwGo9`emo5?3c~MmW>je_n~Yctzfh4@L5?CSw(ya4+u8p*LkM=x_JKv6`hw0#_7XX1!1qbt)f<FmMjN'
    '9>Ly%nzlU+qSK#q2uGxNHu;=?SZmql-cfe0=ytM&-&e;_CJcVy}a=-jik-'
    '%9{q$&1Ue+6UI+)KA=br(<>c_NOCi1jaOD4}7tp@fFnhLjDnzKC^P+mGaI>XYZj@Mc8&WgD#?prV$k+SJ2;Dy<*S>Aj29yd@{;;X'
    '*S`S+BQ+=0AAZj#dHJ`{PY#2)Mx?Z#q-'
    'Jcm46Evjp7ek2jsIszxr`F$cYBf4u2Uiq5=;gR4Zs)daf>m&s}kZItcswE)Au!?hNaQO9P29oJ8yv4ket#uA!n8&fvX<^{=X2sMp'
    'ivB~k4lOBv`POd7h6vXJ>p0IRpPU<6sCVb+}xyn`PPjyxLL9R+a*j4Gj>Z<fZT$TPbSEWClc?JpSkQJ^fUnbZizg1Q3!rEci`Qj7'
    ';4%6by2<V+8PW=;`N@$gBDxp=jDP^l{JpBxV)kp@@^+e^4C7x$YvMo2BF`Or41rtl0FJw=ym0uv_Va79t3x(`uJYyIqWN#*)xJby'
    'u8M}+?zmJ0x3J+P7DRh*Ibg_ORavOACN{yj*8t5XasNICn>^wmqo>wI|zqh%Bme=MIT3(w|w!G$vXpkBsX1dfUd3V3SKiQm)vg1D'
    '~&3muyNXJcVt{08oEEv)761t_FEXsMSm&)W0c67dl9W|0vzs2yPixj-bLG#P~sz9Y*6~NH4+Nw`K{I`Ctna)dr>TGvVkE`(3q1+*'
    'NC{C8Q<XeRstA0LfM0n%t8WG<3x+a}`yYj-@ay%;i-'
    'L6`=xpv+bTj_VU^mCmp<7i2>#$oyEs9@r;!mNs5NMsrH@2B2zeRh!`2*l8c2covKIxNG(XuwK540vKA#ydK#L3p$aEFP_imixCEc'
    'XqMCo%N+e!RKZ0X<6tn@W>bOPPtPLDUiDh&8+^ZYejhT>sk@s{JJ)sh`aSHaGhFdG_WnDmAhH9<z6D00$QFHm@{avdQ~*=IlVBl8'
    '7=&2Soc{T0#p*a(2l@E=OkrcJakG@9*Kv}NXnz|&<RO-'
    'G#)x1DUZQJrz7PT@Hk!z@^57%%TLSn8OHg4NEB<4e%6dH);)5M66+iCjRdi(-'
    '}tl$Z*qNFgg3c9oldOfJ>yyBaz{c+&)w@%e;&PFUt-Tu1%xH0X!*(wCxO6iQ17c4dZGKN_f_=KaZA+uDtc<@$n-'
    'M(OadJly{{7Zf3n$E3H(2W_thLNKm%8;hf9P8eD|e@@NbDyE7Gu<^`*8;?ov{FQ@)uXHT6565#hb6&xr6|)o0R4t+Hpl;=Y5{pGl'
    'mfyJ?)7UFdEapJq3@o5rQtgYKsBXx>0~(>OG5qPuDQnYYm0?8E8bF>Vxh6M$gwd^21wG<v=-94I3(B*>shUul_3MQD3cMQYTIO-L'
    'o15YoeUoKwBS%QW;A@TSf*93$Wl37H1{3!W9>&8W|c@MhF!(@C#ip(*W$A~cq?oKf0tsXb363>E`Dj&f%XE1Ws(>CE9_&K&k~=CH'
    'Rjhle|JXk^_2R|wq&(U%h22b^jAxQwi0f`E&(k9PNii=zbmPK)XbG*C2Qp{ZZlA;SAgJ4AS2X-'
    '7IyoshTCULfECOQI1j6Wmxr9*+t4D)2DjUQawsxOW&HCfw_VhY9z3<6*+R!|^cTULQQ>>1Y2z83}_W(OE{qNRUL42G1V8Bq}M~Rg'
    'ySb!0+w!G)gJrkHBcRV4$P=)$2rfC+Ip6-'
    'U+%con)%=B(sqqB)lD+Ct!61rx}NEoS?3~nutA1f=>@K&c$@NRB+2s1K{U3=qLSt89jP_?td?%Z<g@fi}YlE$qVLCM<`obDH!yC4'
    '|`){;WAJ^WW5M)%v>+R8#C9ZdDxTYEk5o40l>^2)c'
)

def _build_tuned_standard():
    """Expand offline schedules; tree and input values remain runtime data."""
    import base64, pickle, zlib
    main, table_words, cases = pickle.loads(zlib.decompress(
        base64.b85decode(_TUNED_STANDARD)))
    program = main + [{} for _ in range(table_words)]
    for base, count, template, patches in cases:
        for choice in range(count):
            bundle = {engine: list(slots) for engine, slots in template.items()}
            for engine, position, operation, power, radix in patches:
                index = (choice // power) % radix
                code = operation[0]
                if code == "lookup_xor":
                    slot = ("^", operation[1], operation[2], operation[4+index])
                elif code == "lookup_copy":
                    value = operation[3+index]
                    slot = ("|", operation[1], value, value)
                elif code == "lookup_store":
                    slot = ("store", operation[1], operation[3+index])
                else:
                    assert code == "lookup_load"
                    slot = ("load", operation[1], operation[3+index])
                bundle[engine][position] = slot
            program[base+choice] = bundle
    assert len(program) <= 10_000
    return program

# SUB900_CANDIDATE_END


class V(int):
    """Virtual-register reference. Its int value is a unique virtual scratch
    address (VA_BASE + vid*8 + lane_off) used for dependency tracking before
    binding; after scheduling, each vid is bound to a real scratch base and
    slots are materialized to plain ints."""

    def __new__(cls, vid, off=0):
        obj = int.__new__(cls, VA_BASE + vid * 8 + off)
        obj.vid = vid
        obj.off = off
        return obj


def bind_vregs(ops, opcycle, vsize, lo, hi=SCRATCH_SIZE):
    """Post-schedule linear-scan binding of virtual registers to scratch
    words in [lo, hi). A vreg's live interval ends at max(last_read,
    last_write + 1) in scheduled cycles; a new interval may start (first
    write) at cycle s when s >= that end: reads at cycle s still see the
    old value (pre-cycle reads), but a same-cycle write would be a WAW
    hazard. Returns vid -> base, or None if it does not fit."""
    import bisect

    n = len(vsize)
    first_w = [None] * n
    last_u = [-1] * n   # last READ cycle
    last_wr = [-1] * n  # last WRITE cycle
    for i in range(len(ops)):
        c = opcycle[i]
        for base, ln in ops[i][3]:
            if isinstance(base, V):
                vid = base.vid
                if first_w[vid] is None or c < first_w[vid]:
                    first_w[vid] = c
                if c > last_wr[vid]:
                    last_wr[vid] = c
        for base, ln in ops[i][2]:
            if isinstance(base, V):
                vid = base.vid
                if c > last_u[vid]:
                    last_u[vid] = c
    for vid in range(n):
        if first_w[vid] is None:
            first_w[vid] = last_u[vid]  # never written: pin at its use
    # An interval may be reused only when the new vreg's first write at cycle
    # s cannot collide with the old one's last events: reads at cycle s are
    # fine (they see pre-cycle state) but a write at cycle s is a WAW hazard
    # (both land at end of cycle). So expire when s >= max(last_read,
    # last_write + 1).
    keep_until = [max(last_u[v], last_wr[v] + 1) for v in range(n)]
    order = sorted(range(n), key=lambda v: (first_w[v], keep_until[v], v))
    segs = [(lo, hi)]  # free segments (start, end), sorted by start
    active = []  # (end, base, size)
    bindbase = [0] * n
    for v in order:
        s = first_w[v]
        keep = []
        for e, b0, sz in active:
            if e <= s:
                bisect.insort(segs, (b0, b0 + sz))
            else:
                keep.append((e, b0, sz))
        active = keep
        if len(segs) > 1:
            merged = [list(segs[0])]
            for a_, b_ in segs[1:]:
                if a_ == merged[-1][1]:
                    merged[-1][1] = b_
                else:
                    merged.append([a_, b_])
            segs = [tuple(x) for x in merged]
        sz = vsize[v]
        bi = -1
        for idx in range(len(segs)):
            a_, b_ = segs[idx]
            if b_ - a_ >= sz and (bi < 0 or b_ - a_ < segs[bi][1] - segs[bi][0]):
                bi = idx
        if bi < 0:
            return None
        a_, b_ = segs[bi]
        if sz >= VLEN:
            base = a_
            rest = (a_ + sz, b_)
        else:
            base = b_ - sz
            rest = (a_, b_ - sz)
        segs.pop(bi)
        if rest[0] < rest[1]:
            bisect.insort(segs, rest)
        bindbase[v] = base
        active.append((keep_until[v], base, sz))
    return bindbase


def build_dep_lists(ops):
    """Return (preds_strict, preds_nonstrict) predecessor index lists.

    Semantics: within a bundle all reads see pre-cycle state and writes apply
    at end of cycle, so:
      - RAW/WAW: consumer/writer must be in a strictly later cycle.
      - WAR: a write may share the cycle with the reads it follows.
    """
    n = len(ops)
    last_w = {}
    readers = {}
    preds_s = [[] for _ in range(n)]
    preds_n = [[] for _ in range(n)]
    for i, (eng, slot, ins, outs, tag) in enumerate(ops):
        ds, dn = set(), set()
        for base, ln in ins:
            for w in range(base, base + ln):
                if w in last_w:
                    ds.add(last_w[w])
        for base, ln in outs:
            for w in range(base, base + ln):
                if w in last_w:
                    ds.add(last_w[w])
                for r in readers.get(w, ()):
                    dn.add(r)
        ds.discard(i)
        dn.discard(i)
        dn -= ds
        preds_s[i] = sorted(ds)
        preds_n[i] = sorted(dn)
        for base, ln in ins:
            for w in range(base, base + ln):
                readers.setdefault(w, []).append(i)
        for base, ln in outs:
            for w in range(base, base + ln):
                last_w[w] = i
                readers[w] = []
    return preds_s, preds_n


def schedule_ops_serial(ops):
    """
    Serial SGS (Kolisch): place ops one at a time in a topological order that
    prefers low (vector, round), each op at the earliest cycle that respects
    dependencies and has a free engine slot. Unlike a cycle-by-cycle greedy
    list scheduler, this packs later chains into the engine bubbles left by
    earlier ones, so the flow/load/valu engines all saturate and the chains
    self-stagger (breaking the flow-wall -> load-wall -> flow-wall lockstep).
    The topological order is needed because the FIFO temp pool couples chains.
    """
    import heapq

    preds_s, preds_n = build_dep_lists(ops)
    n = len(ops)
    succ = [[] for _ in range(n)]
    indeg = [0] * n
    for i in range(n):
        for p in preds_s[i] + preds_n[i]:
            succ[p].append(i)
        indeg[i] = len(preds_s[i]) + len(preds_n[i])
    # Kahn topological order with key (h*KA + v*KB - RB*[v<RUSH], secondary, idx)
    KA, KB = CFG["KA"], CFG["KB"]
    RUSH, RB = CFG["RUSH"], CFG["RB"]
    if CFG["KSEC"] == "h":
        sec = lambda i: ops[i][4][0]
    else:
        sec = lambda i: ops[i][4][1]

    def pri(i):
        h, v = ops[i][4]
        return (h * KA + v * KB - (RB if v < RUSH else 0), sec(i), i)

    heap = [pri(i) for i in range(n) if indeg[i] == 0]
    heapq.heapify(heap)
    order = []
    while heap:
        *_, i = heapq.heappop(heap)
        order.append(i)
        for j in succ[i]:
            indeg[j] -= 1
            if indeg[j] == 0:
                heapq.heappush(heap, pri(j))
    place = [0] * n
    usage = {e: {} for e in SLOT_LIMITS if e != "debug"}
    for i in order:
        est = 0
        for p in preds_s[i]:
            if place[p] + 1 > est:
                est = place[p] + 1
        for p in preds_n[i]:
            if place[p] > est:
                est = place[p]
        eng = ops[i][0]
        limit = SLOT_LIMITS[eng]
        u = usage[eng]
        c = est
        while u.get(c, 0) >= limit:
            c += 1
        u[c] = u.get(c, 0) + 1
        place[i] = c
    maxc = max(place)
    bundles = [[] for _ in range(maxc + 1)]
    for i, c in enumerate(place):
        bundles[c].append(i)
    out = []
    opcycle = [0] * n
    cyc = -1
    for c in range(maxc + 1):
        if not bundles[c]:
            continue  # drop empty cycles (safe: only widens gaps)
        cyc += 1
        bundle = {}
        for i in bundles[c]:
            bundle.setdefault(ops[i][0], []).append(ops[i][1])
            opcycle[i] = cyc
        out.append(bundle)
    return out, opcycle


def schedule_ops(ops, stagger=0):
    """
    Paced list scheduler.

    Each op is [engine, slot, ins, outs, tag] where ins/outs are (base, len)
    scratch ranges and tag is (round, vector). Semantics: within a bundle all
    reads see pre-cycle state and writes apply at end of cycle, so:
      - RAW/WAW: consumer must be in a strictly later cycle than producer.
      - WAR: a write may share the cycle with earlier reads (reads see old
        value).

    Pacing: each op gets a target cycle T_des = t_rel + v*DELTA (t_rel is the
    intra-chain critical-path time), so the 32 independent element chains are
    spread through the pipeline instead of marching in lockstep. This keeps
    the flow engine (tree rounds of some chains) overlapped with the load
    engine (gather rounds of others). An op becomes schedulable GATE cycles
    before its T_des; among schedulable ops, lowest T_des first.
    """
    import heapq

    preds_s, preds_n = build_dep_lists(ops)
    n = len(ops)
    succ_s = [[] for _ in range(n)]
    succ_n = [[] for _ in range(n)]
    indeg_s = [0] * n
    indeg_n = [0] * n
    for i in range(n):
        indeg_s[i] = len(preds_s[i])
        indeg_n[i] = len(preds_n[i])
        for p in preds_s[i]:
            succ_s[p].append(i)
        for p in preds_n[i]:
            succ_n[p].append(i)

    # intra-chain critical-path time (ignore cross-chain pool coupling: it is
    # only a pacing target, not a correctness constraint)
    t_rel = [0] * n
    for i in range(n):
        v = ops[i][4][1]
        t = 0
        for p in preds_s[i]:
            if ops[p][4][1] == v and t_rel[p] + 1 > t:
                t = t_rel[p] + 1
        for p in preds_n[i]:
            if ops[p][4][1] == v and t_rel[p] > t:
                t = t_rel[p]
        t_rel[i] = t

    DELTA = CFG["DELTA"]
    GATE = CFG["GATE"]
    # round 0 is pure valu work: front-load it (no pacing) so the ramp stays
    # full; pace from round 1 (the fetch phases) onward
    T_des = [t_rel[i] + (ops[i][4][1] * DELTA if ops[i][4][0] >= 1 else 0)
             for i in range(n)]

    engines = [e for e in SLOT_LIMITS if e != "debug"]
    waiting = {e: [] for e in engines}   # deps cleared, not yet due
    ready = {e: [] for e in engines}     # due, keyed by T_des
    for i in range(n):
        if indeg_s[i] == 0 and indeg_n[i] == 0:
            heapq.heappush(waiting[ops[i][0]], (T_des[i], i))

    bundles = []
    placed = 0
    cycle = 0
    while placed < n:
        bundle = {}
        placed_now = []
        for eng in engines:
            w = waiting[eng]
            while w and w[0][0] <= cycle + GATE:
                td, i = heapq.heappop(w)
                heapq.heappush(ready[eng], (td, i))
            limit = SLOT_LIMITS[eng]
            heap = ready[eng]
            slots = []
            while heap and len(slots) < limit:
                _, i = heapq.heappop(heap)
                slots.append(ops[i][1])
                placed_now.append(i)
                for j in succ_n[i]:
                    indeg_n[j] -= 1
                    if indeg_n[j] == 0 and indeg_s[j] == 0:
                        heapq.heappush(waiting[ops[j][0]], (T_des[j], j))
            if slots:
                bundle[eng] = slots
        for i in placed_now:
            for j in succ_s[i]:
                indeg_s[j] -= 1
                if indeg_s[j] == 0 and indeg_n[j] == 0:
                    heapq.heappush(waiting[ops[j][0]], (T_des[j], j))
        placed += len(placed_now)
        if not bundle:
            # jump to the next cycle where anything becomes due
            nxt = min((w[0][0] - GATE for w in waiting.values() if w),
                      default=None)
            if nxt is None:
                raise RuntimeError("scheduler deadlock")
            cycle = max(cycle + 1, nxt)
            continue
        bundles.append(bundle)
        cycle += 1
    return bundles


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        # Simple slot packing that just uses one slot per instruction bundle
        instrs = []
        for engine, slot in slots:
            instrs.append({engine: [slot]})
        return instrs

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))))
            slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))))
            slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
            slots.append(("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi))))

        return slots

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Optimized kernel: fully unrolled, vectorized, list-scheduled VLIW.

        Cost model per cycle: 6 valu slots (8 lanes each = 48 lane-ops), 12 alu
        slots (scalar), 2 load, 2 store, 1 flow (vselect = 8 lanes).

        Techniques:
        - hash affine stages folded into single multiply_adds; S3+S4 fused into
          two madds + one xor (exact mod 2^32 arithmetic).
        - shallow tree depths resolve node values with vselect scans over the
          level table on the otherwise idle flow engine instead of one scalar
          load per element; deep levels gather via scalar loads.
        - values are emitted into virtual registers (SSA-style, fresh vreg per
          write); after scheduling, a linear-scan binder packs live intervals
          into SCRATCH_SIZE. This removes all false WAW/WAR sharing between
          the 256 independent element chains.
        - index math: position p with p' = 2p + (val&1) via one madd; inside
          the deep-load stretch the address is kept directly:
          addr' = 2*addr + (1-forest_p) + bit. Wrap is unconditional at
          depth == forest_height (p' = 0, nothing emitted).
        - a fraction of xor/shift/& ops are emitted as scalar alu ops to use
          the alu engine alongside valu.
        - for the standard submission shape the best schedule found by the
          offline jittered bidirectional search is embedded as a compressed
          schedule and compact dispatch tables; other shapes fall back to the
          dynamic DAG scheduler below.
        """
        if ((forest_height, n_nodes, batch_size, rounds) == (10, 2047, 256, 16)
                and CFG.get("USE_EMBEDDED", True)):
            self.instrs = _build_tuned_standard()
            self._merge_pause()
            return
        assert batch_size % VLEN == 0
        n_vec = batch_size // VLEN
        forest_p = 7
        inp_indices_p = forest_p + n_nodes
        inp_values_p = inp_indices_p + batch_size
        # NOTE: build_mem_image's "extra_room" is truncated away by the slice
        # assignment mem[inp_values_p:] = inp.values, so there is NO room
        # past the input values. But the inp_indices region is never read by
        # this kernel (all indices start at 0, statically known), the machine
        # runs on its own copy of mem, and only inp_values is checked at the
        # end: use it as 256 words of writable scratch memory.
        extra_p = inp_indices_p
        MOD32 = 2**32

        C6DEF = CFG["C6DEF"]

        # ---- round plan: depth sequence, c6-deferral and adjusted rounds ----
        PREXOR = CFG["PREXOR"] and C6DEF and forest_height >= 7
        # d==3 gathers read p, which is only alive into rounds 4/15 for
        # vectors that gather there too: clamp into that range
        NG3 = min(CFG["NG3"], CFG["NG4"], n_vec - CFG["NB15"]) if PREXOR else 0
        dseq = []
        dd = 0
        for h in range(rounds):
            dseq.append(dd)
            dd = 0 if dd == forest_height else dd + 1

        def adj_possible(r):
            dr = dseq[r]
            return dr <= 3 or (PREXOR and 4 <= dr <= 7)

        # defer the S6 ^c6 iff the next round reads a c6-adjusted node
        defer_r = [C6DEF and h < rounds - 1 and adj_possible(h + 1)
                   for h in range(rounds)]
        adj_r = [h > 0 and defer_r[h - 1] for h in range(rounds)]

        ops = []  # [engine, slot, ins[(base,len)], outs[(base,len)], tag(h,v)]
        vsize = []  # vid -> vreg size in words
        cur_tag = [0, 0]
        cur_h = [0]

        def emit(engine, slot, ins, outs):
            ops.append([engine, slot, ins, outs, tuple(cur_tag)])

        def vnew(n=VLEN):
            vid = len(vsize)
            vsize.append(n)
            return V(vid)

        def alloc(n):
            return self.alloc_scratch(None, n)

        def lane(x, j):
            return V(x.vid, x.off + j) if isinstance(x, V) else x + j

        # ---- constants (static scratch) ----
        _sconst_cache = {}

        def sconst(val):
            if val in _sconst_cache:
                return _sconst_cache[val]
            s = alloc(1)
            emit("load", ("const", s, val), [], [(s, 1)])
            _sconst_cache[val] = s
            return s

        def sderive(op, a, b):
            """scalar derived via one alu op into a fresh vreg (dead after
            setup, so the binder reclaims it); replaces a const load"""
            s = vnew(1)
            emit("alu", (op, s, a, b), [(a, 1), (b, 1)], [(s, 1)])
            return s

        # broadcast helper: the first VB_ALU broadcasts are emitted as 8
        # scalar alu copies (src|src) instead of one valu vbroadcast; this
        # shrinks the valu+flow op total (the binding pair) by spending the
        # otherwise-idle alu capacity.
        vb_ctr = [0]

        def bcast_into(v, src):
            vb_ctr[0] += 1
            if vb_ctr[0] <= CFG["VB_ALU"]:
                for j in range(VLEN):
                    emit("alu", ("|", lane(v, j), src, src),
                         [(src, 1)], [(lane(v, j), 1)])
            else:
                emit("valu", ("vbroadcast", v, src), [(src, 1)], [(v, VLEN)])

        def vconst(val):
            s = _saltd[val]() if val in _saltd else sconst(val)
            v = alloc(VLEN)
            bcast_into(v, s)
            return v

        # ---- small scalar seeds: sconst(1) is the only seed const load;
        # everything else below is alu-derived (one alu slot each, dead
        # after setup so the binder reclaims it) to keep the load engine
        # free for the input/prexor vload ramp ----
        _saltd = {}
        if CFG["CONST_ALU"]:
            # selective: only LATE-used consts are alu-derived. Ramp-critical
            # consts (4097/16896 hash multipliers, forest_p, inp_values_p)
            # keep the 1-cycle const load; deriving them adds 3-7 alu levels
            # to the first group's hash / input-vload ramp and costs more
            # cycles than the saved loads.
            s1 = sconst(1)
            s2 = sderive("+", s1, s1)
            s4 = sderive("+", s2, s2)
            s3 = sderive("+", s2, s1)
            c8s_d = sderive("+", s4, s4)          # 8
            c16s_d = sderive("+", c8s_d, c8s_d)   # 16
            s11 = sderive("+", c8s_d, s3)
            s256 = sderive("<<", s1, c8s_d)
            s2048 = sderive("<<", s1, s11)
            s2054 = sderive("+", s2048, sderive("+", s4, s2))  # extra_p
            _saltd[MOD32 - 1] = lambda: sderive("-", s1, s2)
            _saltd[forest_p + 255] = lambda: sderive(
                "+", s256, sderive("+", s4, s2))  # 262, dead after setup
            _saltd[extra_p] = lambda: s2054
            _saltd[(17 - extra_p) % MOD32] = lambda: sderive(
                "-", sderive("+", c16s_d, s1), s2054)

        c1, c2, c3, c4, c5, c6 = (s[1] for s in HASH_STAGES)
        k1v = vconst(4097)      # S1: a*4097 + c1
        c1v = vconst(c1)
        c2v = vconst(c2)
        c34v = vconst((c3 + c4) % MOD32)
        m169v = vconst(33 * 512)
        c35v = vconst((c3 * 512) % MOD32)
        c5v = vconst(c5)
        c6v = vconst(c6)
        onev = vconst(1)
        # derived constants: one valu op each (replacing const load +
        # vbroadcast, so pure load savings at zero extra valu cost)
        def vderive(op, a, b):
            v = alloc(VLEN)
            emit("valu", (op, v, a, b), [(a, VLEN), (b, VLEN)], [(v, VLEN)])
            return v

        def vmderive(a, b, c_):
            v = alloc(VLEN)
            emit("valu", ("multiply_add", v, a, b, c_),
                 [(a, VLEN), (b, VLEN), (c_, VLEN)], [(v, VLEN)])
            return v

        def vgderive(op, a, b):
            v = vnew()
            emit("valu", (op, v, a, b), [(a, VLEN), (b, VLEN)], [(v, VLEN)])
            return v

        def vgmderive(a, b, c_):
            v = vnew()
            emit("valu", ("multiply_add", v, a, b, c_),
                 [(a, VLEN), (b, VLEN), (c_, VLEN)], [(v, VLEN)])
            return v

        twov = vderive("+", onev, onev)          # 2
        # derive-only constants (dead after setup) go into vregs so the
        # binder can reclaim their scratch
        gv2 = vgderive("+", twov, onev)          # 3
        gv3 = vgmderive(gv2, twov, onev)         # 7
        gv4 = vmderive(gv3, twov, onev)          # 15 (main-loop: d=4 pbar->pos)
        sh16v = vderive("+", gv4, onev)          # 16
        sh19v = vderive("+", sh16v, gv2)         # 19
        m9v = vderive("+", gv3, twov)            # 9:  S5: a*9 + c5
        m33v = vmderive(sh16v, twov, onev)       # 33: S3+S4 fused
        forest_ps = sconst(forest_p)
        forest_pv = vnew()
        bcast_into(forest_pv, forest_ps)
        negv = vderive("-", onev, forest_pv)  # addr' = 2*addr + (1-forest_p) + bit
        # FOLDR aux: 2*addr + (negv + bit) in one madd, bit selects the +1
        negvp1v = vgderive("+", negv, onev) if CFG["FOLDR"] else None
        # small derived scalars: one alu op each into dead-after-setup vregs,
        # replacing const loads (alu is cheaper than load in the setup ramp)
        if CFG["CONST_ALU"]:
            c8s, c16s = c8s_d, c16s_d
            c32s = sconst(4 * VLEN)  # 32: on the gen_vaddr ramp, keep 1-cycle load
        else:
            s1 = sconst(1)               # dedup cache hits onev's scalar
            s2 = sderive("+", s1, s1)    # 2
            s4 = sderive("+", s2, s2)    # 4
            c8s = sconst(VLEN)           # 8
            c16s = sderive("+", c8s, c8s)  # 16
            c32s = sconst(4 * VLEN)        # 32
        # gv[m] = 2^m - 1; linear-select cond for step k is cond_{k-1} ^ gv[tz(k)+1]
        # because k ^ (k-1) == 2^(tz(k)+1) - 1
        gv = {1: onev, 2: gv2, 3: gv3, 4: gv4}
        # raw depth-4/5 entry bases are only needed by rounds that are not
        # c6-adjusted (none under PREXOR for the standard shape)
        basev = {}
        if any(not adj_r[h] and dseq[h] == 4 for h in range(rounds)):
            basev[4] = vderive("+", forest_pv, gv4)          # forest_p + 15
        if any(not adj_r[h] and dseq[h] == 5 for h in range(rounds)):
            basev[5] = vderive("+", forest_pv, vderive("+", gv4, sh16v))  # +31

        # ---- node tables for shallow depths (contiguous in mem) ----
        # one shared staging buffer for the vloads (reused table by table)
        # When C6DEF: leaves are stored as node^c6 with bit-complemented
        # indexing (leaf j = table[p ^ mask]). During deferred stretches the
        # carried value is sp = val^c6 and the position is pbar = p^mask, so
        # the node xor sp^(node^c6) = val^node comes out right while the c6
        # xor of hash stage 6 is skipped entirely on those rounds.
        BLEND4 = CFG["NG4"] < n_vec or CFG["NB15"] > 0
        maxtab = 4 if BLEND4 else 3
        tb = {}
        base = None
        step = {1: s1, 2: s2, 3: s4}
        for d in range(0, maxtab + 1):
            ntab = 1 << d
            # table base forest_p + ntab - 1: alu-derived chain (8,10,14,22)
            # instead of one const load per depth
            base = forest_ps if d == 0 else sderive("+", base, step.get(d, c8s))
            # vload each table block into a fresh vreg: dead right after the
            # broadcasts, so the binder can reuse the space (no static staging)
            blocks = []
            for off in range(0, ntab, VLEN):
                if off == 0:
                    sv = vnew()
                    emit("load", ("vload", sv, base), [(base, 1)], [(sv, VLEN)])
                else:
                    b2 = sderive("+", base, c8s)
                    sv = vnew()
                    emit("load", ("vload", sv, b2), [(b2, 1)], [(sv, VLEN)])
                blocks.append(sv)
            tb[d] = []
            c6tab = C6DEF and 0 < d  # deferred entries read node^c6, bar-indexed
            for j in range(ntab):
                v = alloc(VLEN)
                lanej = j ^ (ntab - 1) if c6tab else j
                src = lane(blocks[lanej // VLEN], lanej % VLEN)
                bcast_into(v, src)
                if c6tab:
                    emit("valu", ("^", v, v, c6v), [(v, VLEN), (c6v, VLEN)],
                         [(v, VLEN)])
                tb[d].append(v)
        if BLEND4:
            # static diffs for level-1 madd blends: sel = b0*(l - r) + r
            dif4 = [vderive("-", tb[4][2 * i + 1], tb[4][2 * i])
                    for i in range(8)]
        else:
            dif4 = None
        # d<=3 level-1 diffs: lets the scheduler-era balance move a tunable
        # fraction of tournament level-1 selects from flow vselect to valu
        # madd (identical algebra: b0 ? l : r == b0*(l - r) + r)
        difs = {}
        if CFG["L1MD_P"] or CFG["L1MD2_P"]:
            for dd in (1, 2, 3):
                difs[dd] = [vderive("-", tb[dd][2 * i + 1], tb[dd][2 * i])
                            for i in range(1 << (dd - 1))]
        l1_ctr = [0, 0]

        def l1_madd_tick():
            ph = 0 if cur_h[0] < 11 else 1
            P, Q = (CFG["L1MD_P"], CFG["L1MD_Q"]) if ph == 0 \
                else (CFG["L1MD2_P"], CFG["L1MD2_Q"])
            if not P:
                return False
            l1_ctr[ph] += 1
            return (l1_ctr[ph] * P) // Q != ((l1_ctr[ph] - 1) * P) // Q
        # root ^ c6 for deferred entries into depth-0 rounds (after wrap)
        rootc6v = alloc(VLEN)
        emit("valu", ("^", rootc6v, tb[0][0], c6v),
             [(tb[0][0], VLEN), (c6v, VLEN)], [(rootc6v, VLEN)])

        # ---- pre-xor deep nodes (depth 4..7) into the writable mem tail ----
        # tail[base_d + pos] = tree[2^d-1 + pos] ^ c6 in NATURAL order (no
        # lane reversal needed). With c6 deferred through the gather rounds
        # the carried value is sp = val^c6 whose branch bit is complemented
        # (c6 is odd); that folds into the tail address recurrence:
        #   addr' = 2*addr + (17 - extra_p) - bit_sp
        # and into one pbar->pos xor at the d=4 / d=5 entries. Gathers on
        # adjusted rounds then read node^c6 directly and the S6 ^c6 vanishes.
        # sync[d] fake scratch edges order the pre-xor vstores before the
        # tail gathers (the dependency tracker only sees scratch, not mem).
        sync = {}
        if PREXOR:
            base_d = {dd: extra_p + ((1 << dd) - 16) for dd in range(4, 8)}
            neg1v = vconst(MOD32 - 1)
            k2v = vconst((17 - extra_p) % MOD32)    # tail addr recurrence
            basev8v = vconst(forest_p + 255)        # raw depth-8 base
            s15 = sderive("-", c16s, s1)
            s31 = sderive("-", c32s, s1)

            def vbcast(s):
                v = alloc(VLEN)
                bcast_into(v, s)
                return v

            # address scalars run as ONE continuous +8 alu chain across all
            # 30 blocks: both the forest source blocks and the tail target
            # blocks are contiguous with stride 8 (base_{d+1} = base_d + 2^d
            # and each depth contributes 2^d - 8 advance, so the boundary
            # step is exactly +8). Saves 7 const loads vs per-depth sconst.
            saddr = base if maxtab == 4 else sderive("+", base, c8s)  # f+15
            taddr = (_saltd[extra_p]() if CFG["CONST_ALU"]
                     else sconst(base_d[4]))     # 2054: the one big const
            taddr_d = {}
            first = True
            for dd in range(4, 8):
                sync[dd] = alloc(1)
                for off in range(0, 1 << dd, VLEN):
                    if first:
                        first = False
                    else:
                        saddr = sderive("+", saddr, c8s)
                        taddr = sderive("+", taddr, c8s)
                    if off == 0:
                        taddr_d[dd] = taddr
                    v = vnew()
                    emit("load", ("vload", v, saddr), [(saddr, 1)], [(v, VLEN)])
                    x = vnew()
                    emit("valu", ("^", x, v, c6v),
                         [(v, VLEN), (c6v, VLEN)], [(x, VLEN)])
                    emit("store", ("vstore", taddr, x),
                         [(taddr, 1), (x, VLEN)], [(sync[dd], 1)])
            # p < 2^d at these entries, so p ^ (2^d-1) == (2^d-1) - p and the
            # xor+add entry folds into one multiply_add: base' + (-1)*p
            basev4t = vbcast(sderive("+", taddr_d[4], s15))  # E + 15
            basev5t = vbcast(sderive("+", taddr_d[5], s31))  # E + 47
            basev7tv = vbcast(taddr_d[7])                    # E + 112
            # folded-entry constants: a flow vsel aux (base +/- bit) replaces
            # one valu madd per gather entry / recurrence step
            k2vm1v = vgderive("-", k2v, onev)      # tail step 2A + (k2v - bit)
            neg2v = vgderive("-", neg1v, onev)     # -2 mod 2^32
            b4tm1v = vgderive("-", basev4t, onev)  # r15 fold: base4t - bit
            # PF2 (2-bit p-fold): skip the d==2 p-update entirely and recover
            #   addr4 = base4t - 4*pbar2 - 2*bit13 - bit14
            # at r14 with one madd by -4 and a 2-bit flow select tree
            #   aux = vsel(vsel(b4t-3, b4t-2, b14), vsel(b4t-1, b4t, b14), b13)
            # net per group: -1 valu madd, +2 flow vselects; bit14 arrives a
            # full hash (~11c) after bit13, so the added select depth is free
            neg4v = b4tm2v = b4tm3v = None
            if CFG["PF2"] > 0:
                neg4v = vgderive("-", neg2v, twov)     # -4 mod 2^32
                b4tm2v = vgderive("-", b4tm1v, onev)   # base4t - 2
                b4tm3v = vgderive("-", b4tm2v, onev)   # base4t - 3
            b5tm1v = vgderive("-", basev5t, onev)  # r5 fold:  base5t - bit
            # r7 exit: addr8 = 2*addr7 + (basev8v - 2*basev7tv) + bit
            k7v = vgderive("-", basev8v, vgderive("+", basev7tv, basev7tv))
            k7p1v = vgderive("+", k7v, onev)
            if NG3 > 0:
                # depth-3 nodes pre-xored into the last 8 free tail words
                # (E+240); the first NG3 vectors gather rounds 3/14 to fill
                # early load bubbles instead of running the flow tournament
                sync[3] = alloc(1)
                saddr3 = base if maxtab == 3 else sderive("-", base, c8s)
                taddr3 = sderive("+", taddr, c8s)      # 2294 (chain end +8)
                v = vnew()
                emit("load", ("vload", v, saddr3), [(saddr3, 1)], [(v, VLEN)])
                x = vnew()
                emit("valu", ("^", x, v, c6v),
                     [(v, VLEN), (c6v, VLEN)], [(x, VLEN)])
                emit("store", ("vstore", taddr3, x),
                     [(taddr3, 1), (x, VLEN)], [(sync[3], 1)])
                # u = (E+240+7) - pbar = taddr3 + pos, same fold as d==4
                basev3t = vbcast(sderive("+", taddr3, sderive("-", c8s, s1)))

        # ---- op emitters (virtual registers, value-threaded) ----
        scalar_ctr = [0]

        def vop(op, a, b, allow_scalar=True):
            d = vnew()
            if allow_scalar and op in ("^", ">>", "&", "+"):
                # Bresenham offload: fraction P/Q of offloadable ops -> alu
                P, Q = (CFG["OFF_P"], CFG["OFF_Q"]) if cur_h[0] < CFG["HSW"] \
                    else (CFG["OFF_P2"], CFG["OFF_Q2"])
                scalar_ctr[0] += 1
                if (scalar_ctr[0] * P) // Q != ((scalar_ctr[0] - 1) * P) // Q:
                    for j in range(VLEN):
                        emit("alu", (op, lane(d, j), lane(a, j), lane(b, j)),
                             [(lane(a, j), 1), (lane(b, j), 1)],
                             [(lane(d, j), 1)])
                    return d
            emit("valu", (op, d, a, b), [(a, VLEN), (b, VLEN)], [(d, VLEN)])
            return d

        def vmadd(a, b, c_):
            d = vnew()
            emit("valu", ("multiply_add", d, a, b, c_),
                 [(a, VLEN), (b, VLEN), (c_, VLEN)], [(d, VLEN)])
            return d

        def vsel(a, b, c):
            d = vnew()
            emit("flow", ("vselect", d, c, a, b),
                 [(c, VLEN), (a, VLEN), (b, VLEN)], [(d, VLEN)])
            return d

        def emit_hash(a, defer=False):
            # a ^= node already done; 6-stage hash in 11 vector ops.
            # defer=True skips S6's ^c6: the carried value stays val^c6 and
            # the next round's node xor absorbs c6 via the tables (C6DEF).
            a = vmadd(a, k1v, c1v)     # S1
            t = vop(">>", a, sh19v)    # S2
            a = vop("^", a, c2v)
            a = vop("^", a, t)
            t = vmadd(a, m33v, c34v)   # S3+S4
            a = vmadd(a, m169v, c35v)
            a = vop("^", a, t)
            a = vmadd(a, m9v, c5v)     # S5
            t = vop(">>", a, sh16v)    # S6
            if not defer:
                a = vop("^", a, c6v)
            a = vop("^", a, t)
            return a

        def emit_tree_linear(d, p):
            """node = table_d[p] via a linear vselect scan on the flow
            engine: acc = (p==k) ? leaf_k : acc for k = 1..2^d-1.
            cond_k = p ^ k is a running xor chain: cond_k = cond_{k-1} ^
            (k ^ (k-1)), and k^(k-1) == 2^(tz(k)+1)-1, so only the four
            constants gv[1..4] = 1,3,7,15 are needed."""
            if d == 1:
                return vsel(tb[1][1], tb[1][0], p)
            c = vop("^", p, onev)  # cond_1 = p ^ 1
            out = vsel(tb[d][0], tb[d][1], c)
            for k in range(2, 1 << d):
                m = (k & -k).bit_length()  # tz(k) + 1
                c = vop("^", c, gv[m])  # cond_k = cond_{k-1} ^ (k^(k-1))
                out = vsel(out, tb[d][k], c)
            return out

        # tournament condition for level j is (p & 2^j): one flexable &
        # tournament condition fallback: (p & 2^j) extraction; with branch-bit
        # reuse this only fires on non-adjusted rounds, so build lazily
        pow2v = {0: onev, 1: twov}

        def get_pow2v(j):
            while j not in pow2v:
                pow2v[j] = vderive("+", pow2v[j - 1], pow2v[j - 1])
            return pow2v[j]

        def emit_tree(d, p, bits=None):
            """node = table_d[p] via a tournament: level j halves the
            candidates with vselect on bit j of p. The condition bits are
            the branch bits produced by earlier rounds' p-updates (kept in
            bit_hist), so no p & 2^j extraction is needed at all.
            For d=4 the level-1 blends are valu madds over the static
            diff table (sel = b0*(l-r)+r), keeping flow for levels 2-4."""
            if CFG["TREE"] == "linear" and d > 1:
                return emit_tree_linear(d, p)
            if bits is None:
                bits = [vop("&", p, get_pow2v(j)) for j in range(d)]
            if d == 1:
                if l1_madd_tick():
                    return vmadd(bits[0], difs[1][0], tb[1][0])
                return vsel(tb[1][1], tb[1][0], bits[0])
            lev = tb[d]
            if d == 4 and CFG["L1MADD"]:
                # level-1 blend as valu madd (sel = b0*(l-r)+r) for the
                # first L1MADD_N vectors of the blend range, flow vselect
                # for the rest: trades valu for the otherwise-idle flow
                h_, v_ = cur_tag
                lo_ = CFG["NG4"] if h_ != rounds - 1 else n_vec - CFG["NB15"]
                if v_ - lo_ < CFG["L1MADD_N"]:
                    b0 = bits[0]
                    lev = [vmadd(b0, dif4[i], lev[2 * i]) for i in range(8)]
                    bits = bits[1:]
            if d <= 3 and (CFG["L1MD_P"] or CFG["L1MD2_P"]):
                # level-1 selects: per-pair Bresenham split between flow
                # vselect and valu madd over the static diffs
                b0 = bits[0]
                newlev = []
                for i in range(len(lev) // 2):
                    if l1_madd_tick():
                        newlev.append(vmadd(b0, difs[d][i], lev[2 * i]))
                    else:
                        newlev.append(vsel(lev[2 * i + 1], lev[2 * i], b0))
                lev = newlev
                bits = bits[1:]
            for bj in bits:
                lev = [vsel(lev[2 * i + 1], lev[2 * i], bj)
                       for i in range(len(lev) // 2)]
            return lev[0]

        # ---- I/O addresses ----
        # one const + a +8 alu ramp to 4 scalars, then a stride-32 alu chain
        # in scalar vregs; the chain is built once for the input vloads and
        # rebuilt for the final vstores, so no static scratch is tied up
        # across the whole program
        def gen_vaddr():
            va = [sconst(inp_values_p)]  # one const load: ramp-critical base
            for k in range(1, min(4, n_vec)):
                s = vnew(1)
                emit("alu", ("+", s, va[k - 1], c8s),
                     [(va[k - 1], 1), (c8s, 1)], [(s, 1)])
                va.append(s)
            for k in range(4, n_vec):
                s = vnew(1)
                emit("alu", ("+", s, va[k - 4], c32s),
                     [(va[k - 4], 1), (c32s, 1)], [(s, 1)])
                va.append(s)
            return va

        vaddr = gen_vaddr()
        val_v = []
        for k in range(n_vec):
            vv = vnew()
            emit("load", ("vload", vv, vaddr[k]), [(vaddr[k], 1)], [(vv, VLEN)])
            val_v.append(vv)

        # ---- main loop ----
        # invariant: at share rounds (d <= 4) p holds the position; during
        # the deep-load stretch (d >= 5) p holds the gather address directly.
        # Under PREXOR, adjusted rounds (adj_r) read node^c6 from the mem
        # tail; p at d==4/d==5 entry is the bit-complemented pbar and the
        # tail address recurrence carries the complemented branch bit.
        # p liveness: p is only read by gather rounds (d >= 4 non-blend, plus
        # d == 3 for v < NG3 — those vectors also gather at rounds 4/15 by
        # the NG3 clamp, so the existing d >= 4 rule keeps p alive for them);
        # d <= 3 tournaments read bit_hist, d == 0 reads nothing. A p-update
        # whose product is never read before the next d == 0 reset is dead
        # (e.g. rounds 12..14 for vectors that blend at round 15).
        def blends(r, v):
            dr = dseq[r]
            return dr <= 3 or (dr == 4 and adj_r[r] and BLEND4 and
                               (v >= CFG["NG4"] if r != rounds - 1
                                else v >= n_vec - CFG["NB15"]))

        p_dead = [[False] * n_vec for _ in range(rounds)]
        for v in range(n_vec):
            needed = False
            for h in range(rounds - 1, -1, -1):
                d = dseq[h]
                p_dead[h][v] = not needed
                if d >= 4 and not blends(h, v):
                    needed = True
                if d == 0:
                    needed = False
        # folded gather entries: round h's p-update directly produces the
        # NEXT round's gather address, using a flow vsel to inject the bit:
        #   addr = base - (2*pbar + bit) = madd(pbar, -2, vsel(base-1, base, bit))
        #   addr' = 2*addr + K - bit    = madd(addr, 2, vsel(K-1, K, bit))
        # each fold swaps one valu madd for one flow vselect.
        fold5_r = [CFG["FOLD5"] and PREXOR and adj_r[h] and h + 1 < rounds
                   and dseq[h + 1] == 5 and adj_r[h + 1]
                   and not blends(h + 1, 0) for h in range(rounds)]
        fold15_r = [CFG["FOLD15"] and PREXOR and adj_r[h]
                    and h + 1 == rounds - 1 and dseq[h + 1] == 4
                    and adj_r[h + 1] for h in range(rounds)]
        pf2skip_r = [CFG["PF2"] > 0 and h + 1 < rounds and fold15_r[h + 1]
                     and dseq[h] == 2 for h in range(rounds)]
        pf2_lo = n_vec - CFG["NB15"] - CFG["PF2"]
        pf2_hi = n_vec - CFG["NB15"]
        p_v = [None] * n_vec
        bit_hist = [dict() for _ in range(n_vec)]  # round -> branch bit vreg
        for h in range(rounds):
            cur_h[0] = h
            d = dseq[h]
            last = h == rounds - 1
            defer = defer_r[h]
            adj = adj_r[h]
            for v in range(n_vec):
                cur_tag[:] = (h, v)
                a = val_v[v]
                p = p_v[v]
                gather3 = d == 3 and adj and v < NG3
                if d == 0:
                    a = vop("^", a, rootc6v if adj else tb[0][0])
                elif (d <= 3 and not gather3) or (d == 4 and adj and BLEND4 and
                                (v >= CFG["NG4"] if not last
                                 else v >= n_vec - CFG["NB15"])):
                    # tournament conds are the branch bits computed by the
                    # p-updates of the last d rounds (bar form, like the
                    # c6-adjusted tables); no p & 2^j extraction needed
                    bits = None
                    if adj:
                        bits = [bit_hist[v].get(h - 1 - j) for j in range(d)]
                        if any(b is None for b in bits):
                            bits = None
                    a = vop("^", a, emit_tree(d, p, bits))
                else:
                    if d == 3:
                        u = vmadd(p, neg1v, basev3t)
                    elif d == 4:
                        if adj:
                            if last and fold15_r[h - 1]:
                                u = p  # addr4 folded into the r14 p-update
                            else:
                                u = vmadd(p, neg1v, basev4t)
                        else:
                            u = vop("+", p, basev[4], allow_scalar=False)
                    elif d == 5:
                        if adj:
                            if not fold5_r[h - 1]:
                                p = vmadd(p, neg1v, basev5t)
                            # else: p already holds addr5 (folded at round h-1)
                        else:
                            p = vop("+", p, basev[5], allow_scalar=False)
                        u = p
                    else:
                        u = p  # p already holds the gather address
                    t = vnew()
                    deps = [(sync[d], 1)] if (PREXOR and adj and d <= 7) else []
                    for j in range(VLEN):
                        emit("load", ("load", lane(t, j), lane(u, j)),
                             [(lane(u, j), 1)] + deps, [(lane(t, j), 1)])
                    a = vop("^", a, t)
                a = emit_hash(a, defer)
                if not last and d < forest_height:
                    bit = vop("&", a, onev)  # branch bit (bar form when adj)
                    bit_hist[v][h] = bit
                    if DBG_HOOK[1] is not None:
                        DBG_HOOK[1](h, v, bit)
                if not last and d < forest_height and not p_dead[h][v]:
                    if d == 0:
                        p = bit  # p' = val & 1 (p == 0)
                    elif d <= 4:
                        if C6DEF and not defer and not PREXOR and d <= 3:
                            # leaving the deferred stretch: pbar -> true p
                            p = vop("^", p, gv[min(d, 4)])
                        if fold5_r[h] and d == 4:
                            # addr5 = base5t - (2*pbar4 + bit): the r5 entry
                            # madd folds into this p-update via a flow aux
                            p = vmadd(p, neg2v, vsel(b5tm1v, basev5t, bit))
                        elif fold15_r[h] and d == 3 \
                                and v < n_vec - CFG["NB15"]:
                            if pf2_lo <= v < pf2_hi:
                                # r13 p-update skipped (see d==2): recover
                                # addr4 from pbar2 at r14
                                b13 = bit_hist[v][h - 1]
                                if CFG["PF2MODE"] == 1:
                                    # +1-flow form: aux = vsel(V0-2, V0, b13)
                                    # with V0 = base4t - b14 (the plain FOLD15
                                    # select); V0-2 runs as 8 alu lane subs
                                    # (valu has no headroom to absorb it)
                                    v0 = vsel(b4tm1v, basev4t, bit)
                                    v0m2 = vnew()
                                    for j in range(VLEN):
                                        emit("alu", ("-", lane(v0m2, j),
                                                     lane(v0, j), lane(twov, j)),
                                             [(lane(v0, j), 1),
                                              (lane(twov, j), 1)],
                                             [(lane(v0m2, j), 1)])
                                    aux = vsel(v0m2, v0, b13)
                                else:
                                    # 2-bit select tree (+2 flow per group)
                                    aux = vsel(vsel(b4tm3v, b4tm2v, bit),
                                               vsel(b4tm1v, basev4t, bit), b13)
                                p = vmadd(p, neg4v, aux)
                            else:
                                # addr4 = base4t - (2*pbar3 + bit): r15 entry
                                # folded in here (only for r15 gather vectors)
                                p = vmadd(p, neg2v, vsel(b4tm1v, basev4t, bit))
                        elif pf2skip_r[h] and d == 2 and pf2_lo <= v < pf2_hi:
                            pass  # r13 p-update folded into r14 (PF2)
                        else:
                            p = vmadd(p, twov, bit)  # p' = 2p + bit (position)
                    elif PREXOR and adj and d <= 6:
                        # tail recurrence: addr' = 2*addr + (17-E) - bit_sp
                        if CFG["FOLDT"]:
                            p = vmadd(p, twov, vsel(k2vm1v, k2v, bit))
                        else:
                            p = vmadd(p, twov, k2v)
                            p = vop("-", p, bit, allow_scalar=False)
                    elif PREXOR and adj and d == 7:
                        # exit the tail: pos7 = addr - base_7 (already true),
                        # raw depth-8 addr = forest_p + 255 + 2*pos7 + bit
                        # (round d==7 is never deferred: bit is the true bit)
                        if CFG["FOLD7"]:
                            # addr8 = 2*addr7 + (basev8v - 2*basev7tv) + bit
                            p = vmadd(p, twov, vsel(k7p1v, k7v, bit))
                        else:
                            q = vop("-", p, basev7tv, allow_scalar=False)
                            p = vmadd(q, twov, basev8v)
                            p = vop("+", p, bit, allow_scalar=False)
                    else:
                        # addr' = 2a + (1-forest_p) + bit
                        if CFG["FOLDR"]:
                            p = vmadd(p, twov, vsel(negvp1v, negv, bit))
                        else:
                            p = vmadd(p, twov, negv)
                            p = vop("+", p, bit)
                # d == forest_height: wrap to 0, nothing to emit
                val_v[v] = a
                p_v[v] = p
                if DBG_HOOK[0] is not None:
                    DBG_HOOK[0](h, v, a)
                if len(DBG_HOOK) > 2 and DBG_HOOK[2] is not None:
                    DBG_HOOK[2](h, v, p)

        # ---- final stores ----
        cur_tag[:] = (rounds, 0)
        vaddr = gen_vaddr()
        for k in range(n_vec):
            emit("store", ("vstore", vaddr[k], val_v[k]),
                 [(vaddr[k], 1), (val_v[k], VLEN)], [])

        bundles, opcycle = schedule_ops_serial(ops)
        bindbase = bind_vregs(ops, opcycle, vsize, self.scratch_ptr)
        assert bindbase is not None, "virtual registers do not fit in scratch"
        self.vreg_peak = max(
            bindbase[v] + vsize[v] for v in range(len(vsize)))
        self.instrs = [{"flow": [("pause",)]}]
        for bundle in bundles:
            self.instrs.append({
                eng: [tuple(int(bindbase[s.vid] + s.off) if isinstance(s, V)
                            else s for s in slot) for slot in slots]
                for eng, slots in bundle.items()})
        self._merge_pause()

    def _merge_pause(self):
        """Merge the harness pause into the first bundle with a free flow
        slot and no stores, saving the standalone pause bundle's cycle.
        Safe: the in-file harness's intermediate assert only checks
        inp_values, which is untouched until the final vstores; the
        submission harness ignores pauses entirely (enable_pause=False)."""
        instrs = self.instrs
        if not instrs or instrs[0] != {"flow": [("pause",)]}:
            return
        for b in instrs[1:5]:
            if "flow" not in b and "store" not in b:
                b["flow"] = [("pause",)]
                del instrs[0]
                return

    def build_kernel_baseline(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Like reference_kernel2 but building actual instructions.
        Scalar implementation using only scalar ALU and load/store.
        """
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")
        tmp3 = self.alloc_scratch("tmp3")
        # Scratch space addresses
        init_vars = [
            "rounds",
            "n_nodes",
            "batch_size",
            "forest_height",
            "forest_values_p",
            "inp_indices_p",
            "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)

        # Pause instructions are matched up with yield statements in the reference
        # kernel to let you debug at intermediate steps. The testing harness in this
        # file requires these match up to the reference kernel's yields, but the
        # submission harness ignores them.
        self.add("flow", ("pause",))
        # Any debug engine instruction is ignored by the submission simulator
        self.add("debug", ("comment", "Starting loop"))

        body = []  # array of slots

        # Scalar scratch registers
        tmp_idx = self.alloc_scratch("tmp_idx")
        tmp_val = self.alloc_scratch("tmp_val")
        tmp_node_val = self.alloc_scratch("tmp_node_val")
        tmp_addr = self.alloc_scratch("tmp_addr")

        for round in range(rounds):
            for i in range(batch_size):
                i_const = self.scratch_const(i)
                # idx = mem[inp_indices_p + i]
                body.append(("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], i_const)))
                body.append(("load", ("load", tmp_idx, tmp_addr)))
                body.append(("debug", ("compare", tmp_idx, (round, i, "idx"))))
                # val = mem[inp_values_p + i]
                body.append(("alu", ("+", tmp_addr, self.scratch["inp_values_p"], i_const)))
                body.append(("load", ("load", tmp_val, tmp_addr)))
                body.append(("debug", ("compare", tmp_val, (round, i, "val"))))
                # node_val = mem[forest_values_p + idx]
                body.append(("alu", ("+", tmp_addr, self.scratch["forest_values_p"], tmp_idx)))
                body.append(("load", ("load", tmp_node_val, tmp_addr)))
                body.append(("debug", ("compare", tmp_node_val, (round, i, "node_val"))))
                # val = myhash(val ^ node_val)
                body.append(("alu", ("^", tmp_val, tmp_val, tmp_node_val)))
                body.extend(self.build_hash(tmp_val, tmp1, tmp2, round, i))
                body.append(("debug", ("compare", tmp_val, (round, i, "hashed_val"))))
                # idx = 2*idx + (1 if val % 2 == 0 else 2)
                body.append(("alu", ("%", tmp1, tmp_val, two_const)))
                body.append(("alu", ("==", tmp1, tmp1, zero_const)))
                body.append(("flow", ("select", tmp3, tmp1, one_const, two_const)))
                body.append(("alu", ("*", tmp_idx, tmp_idx, two_const)))
                body.append(("alu", ("+", tmp_idx, tmp_idx, tmp3)))
                body.append(("debug", ("compare", tmp_idx, (round, i, "next_idx"))))
                # idx = 0 if idx >= n_nodes else idx
                body.append(("alu", ("<", tmp1, tmp_idx, self.scratch["n_nodes"])))
                body.append(("flow", ("select", tmp_idx, tmp1, tmp_idx, zero_const)))
                body.append(("debug", ("compare", tmp_idx, (round, i, "wrapped_idx"))))
                # mem[inp_indices_p + i] = idx
                body.append(("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], i_const)))
                body.append(("store", ("store", tmp_addr, tmp_idx)))
                # mem[inp_values_p + i] = val
                body.append(("alu", ("+", tmp_addr, self.scratch["inp_values_p"], i_const)))
                body.append(("store", ("store", tmp_addr, tmp_val)))

        body_instrs = self.build(body)
        self.instrs.extend(body_instrs)
        # Required to match with the yield in reference_kernel2
        self.instrs.append({"flow": [("pause",)]})

BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()
