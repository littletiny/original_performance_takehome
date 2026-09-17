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
# Verified checkpoint: 915 dynamic cycles; 10808 static bundles.
_TUNED_STANDARD = (
    'c-'
    'pkR2b@*awLd<)9L0{<#(o*vg`r$j6gwa)@nkD87BaDb_0mTLB*sM2gji8gi6+MPUS8hNhmM_>V#e6Uj(tRgv77P|G>QJ#+Iz3H_d'
    'XYs-{=4N1ny@(XEVc{d)C=|)$jVQS^9QeC+hHz8QSmMdF8zBJM+pn=t)z@T{I7tH)I!1opQyrd3hJJ>khN$RkisdJ7w>4U-'
    '<6qc^BWHFPSv;O8BW8^n~e?o9CHb%h%2uH~m_?ux~PB+@$IDP2{o1&dclby1Z`Mym3`&-*<-'
    'n6?xaZ&c5yEnGs##O{?3(`(@4JreAUKydSxD+YR5%{;z7cyp-<^Ki@4c=UwopcY&YYamEFg!+Tvg4nJol-'
    ')Hu`aW^RX)uM4;J`{d;muc?Jy4p7@!MDoQhI~i({qP;QH2=upr|RAH-|>3rH23~xc$4$y!7nc5m1*;;2j#=y!_LNwQ6P-'
    'j@E;oC)0FX>7rRf1Lb@(H=2zo>NB5TR0Us8B{qD)H-vNI8-'
    'u74KJHzFU)9?cin3s>uyTa$eZwxh`WqG%}tNUDA@pHNVVEzi%@Il@GJ_`PQNB8fa<Ii_<*Pr6g?-'
    'hNnzh!yJKgH{^x(l9OTsNC<a_7m@CrumQJn38KkGtq1cfZ5(QTC2U+Z%<8G4|HW)sgm&?YHf1bxUq*C%Bg#+y~ngE_Ze}@==!W?Q'
    'bN%0RDMwkUM-XxA+7XTm9l=T>RDF;gN7B?mf*P@ve??*MGpD`P7T*&sko_a{Ckh)Smnd;UVE2jYbj6<BppLAHS53v3EBs@9Lgn75'
    '))_pxTr-!{r$Fn`gRx!7}^{i0;5Aw!Z70?f+$_ZiIihd!!q6CogHYQ%N|xz^(7(-'
    'nkhrcXglkFIm1bNXY&BMey$zuyne5=?sP6m3L!Ny@z-'
    'HiC=t(H+!VJ{s6!6NB>qw!@v9Ab_`tZ?0)2Z{E<Iov2aivWh;(3@JsE(o(rGKUd)HzY%lKR4PS-'
    '7&Hmx<nLPa6;KT3Ye&gkE+3YSafy;~BFYm-#`iuYITe4`YZ^HYd#b#X4kA03G_zS;y3m5PC#oM^}(BJK`aEtDDn~nJI?`P3-'
    'dk3D|4yQG(cF*lySpD{-=6{3(0Gk>_kk85IPU9UuKfl*4@P7F_EwJ$4Y0O($FRZhN!-'
    '{0DbSLcMRo0&L%iyOszz4EIZ6kHk4j*c^f6r>(LcgokoimV0z8*e<-HIju)T-{O;pO<ed*0VwPRJYbe(rK&-'
    'Xm{xmy`0Id4G2~Iq#JZaF<i^-uXaxIW_N-4|3A$K<T{|2|qvDUB8Rh$2w^r2ba70J+L?3u)W`%K?U=MxC|csZtj0fgUid^8@+-L-'
    'apZNEByN<?y<j_<(K<|Fx0$(d!W;QjMwoqz3msT;o<|ocojd?G4A>${KmWf8yoN&$GPj5@f#bHJ>LnBWJ7qqZF##BL<{`&dY}U?$'
    '4r|?C#2$@!zb{w_5RNHNp}Aga3`0!o0<Wa)7<4a)B;}LJ&!m3iod}ZvV0mHd-EcgiF~X3#&dY*ulR2~g&)1cFP_E4Yku)GE?)ABc'
    '3f=mi)V20GT(0XAk$Uk@6EP1Rf3zHW$)Y63D{qJ<9v%F;a|Oe2aD8ip6Mdsra$qwTp!)?Q1fJ#U*Yee3I1EFlUGx|9R7a8U9Nz?P'
    '403f{Qai8Tm^s6ahI#%@44=B4g5XNU9N?{=eyhaJ>Jg8{@xzPMY~@-'
    'fr~Bijn<3&U3ZQ;@PN4YD&eipqT6iAJLtL7ZQAGFiqGBN<G$$$-W5-'
    '<DT{nYtjjNiM?B6w;?LmHKfA~9*=@<_J*#dnHRcQ8jDF2sE`&3By1QHiXY}jtaxt9IGu-'
    '78IHPB}%cXEe&vKW`;EbN_Zo3U{`<X2AQF_la6#=(weV9hLUA*H`&}FQ?szI?eSykgUxG4#gfWQC!qWd?)M1Hvyz!;HV5DQ>B-'
    '1QYs02jmMh3@k2aQT`0#HdlL?h@2fKHgn|dderbOHfbwM0W}5DWBvnK|SS@-'
    '6g1}e2TjS^^{L_xBol5{olt=eYD6wLN#heiTp!&{1dhl=+jy$)Q9KZ>bE!Et(S;Xqt<=f9$m)mXIyddq>JHR@)nRNY$rH6-'
    'h13XmdaR>?_j0UEb_m{Qn?cDf4cjevjkkvaALU>E-!Xl<Wbxr&wC~CTU<QpxAP--V^8_T!?^gJU;G9aulvQXaq+od{0bMZ`o;g^;'
    '%UEFkBi4+G0hbD`_6;wfSdLw|DRYsQxzZ{J|b1rh$N{AeL%z@q@k&h4v5r4qxpp}-xOf$+#dT+Va}yhL;g8P<nK-'
    'h51|m=%xEW~a4T-nLzoLieyCgY6y_q4ALbUlgt=JcL)@Z|Fqew_2)F1f;FS+{i-'
    '7`O`Qc6?528fY2fv`3nJe;(>|>lG^3Q^SHuFXP0qM(*pgO$4g=2(Uh#iEMNJ{?fwfLiSV7zHsMLx|sAP;0FQ(@OMn*S2!oFJM{K{'
    'S`TcWV&l0+Ao$7J~%5+-'
    ')|Ymq6NfJg671Z6|<w@!EDGs28toCxLqL+IBLi7q4xnfO_%Tb}FbB(6$%fkX<ot>g5;P=xMtd?iy_J95H*|gv;=OPlF9VJ2ImkxR'
    'rv{U~5uB<-'
    '!7ieKa~8!_i)9C?^+2gdJsC1fIK@0*<oZQU}E%Tqp9&>;wBm<YQwA{<knqVXOUzFz1EUJHp%=R&NP&M_4^C%=JO?e*?*X>b|i-'
    'm@D}krW4Qd=k7b*g&D^ithX?i_zl(wHUiCra}><|89Zm_vRP?4rPRi1y2%R3+hW-ArPvm;=Bx&Pq{Th)SX%1^9>7>y2glN?llg8|'
    'T3uv*Wh|{N0=7Qi+<oYLeCn>(p^Nf@7fuyDgp#`+FTkdEch?gH81)8sJyC#F@8PZ|2{7wD-SuPvcD<Loo+7}o_jcD)1z7ez?z&OH'
    '3y*hj!>w}mhY2nBSZ}$TP%*&g8S~GLnlrkomNHxIo|-'
    'SfPIxJmVkwnnKF&(1Ugm#`rL;+y+rm!i5a!OXdO?`$!s<C;ZV0Pqgt;lK+J(6}?67~pKKh&6AYY)``@;W}kno~Sr6s&b(`_%|Map'
    'e}fETJgK7uGA>$><2R+1<t?+A32c4rmr@ko0NBk{F3@uix*E!FJZ3&-#Z8`pH?pT-'
    'jV9|3Lx-}x_v`9@gXE6n*}&#lEhcWKRb?;9M2e!)>_430wo;3y0Tj>5p;C=7Cr!o?G&!foK~SiN-xJY3LW_;2RZ49;dyV(uJ{VF6'
    'Yoh%ND?VC%7MS|@^+9G%#0+D3R3u>>2e1nIGjlBmNrxHldv_&FtN0iI+z80vCjjq#f;Q9x+UlBTa@ai1wozpxrD&7iP)Nq|b{)w9'
    'Cf7*@ljIVk+(tHMkRoA(RYy#H{|Zw=~+C%qT5IztQgpI%CJ5*3fP;M+-fp?+kWJ5=VEx-eYicg!w6sLS#WdQKbZImI%qq%!RhcBM'
    'Ao1I_LdmHSF_>W=Zo{JJ##!(Dw%nt@@pQJCAq{of?=w|H@b$lvD0NfO_eRwqicUs#Qh=D@IeU6{+mJH95&6=C(VFf(X1)U3+TR^2'
    'P&0}Z>cHH0qJD0JZzRh3dysq+Ea6aI5S`;C!kBS`zPw5^DraUyn$HZO0KYtO`wJ6nR1<U4p<n0euTZWQ@OUYsD!0rZ1SxTVQQL5o'
    'RF3VJl;rJ#F9ZVGyL<fou>M~(^_Sn^cR&vULyNZjD>3H~dtoH206d2D=~tJa2;rg6%znneAv*s2mWAemO$VsLl6y4)TdlACslx38'
    '7q`>{Y@js)5(R-ao%-t3|WnNN?m=^4^A20@=L&494lEX*xI>^FlS!HYM9xh$yG_rO;HhlRZmaE*T9S1U7omS;)#-'
    ';{(GdDoeQ7dh1Z1iZ-Y>@VPj)+kM<mV<}|n;fU4BGXchEprR}s8eh$f^u8&@ovSZ1s>qmnw^^O70LAvkz9Kx8})jTZ(^xjC-'
    'QfA@ney1<^_2BTX^xJfN%0@m^24VKAtNw)D@$pTP@E}kDTtM2fkMe`L*VMg;^kcgl~Q=%p&0fd-EG%77GmRdB@^@jKZ%D^n+PQ|I'
    '$cXNm?CIBRhegX^f1vE8NOcJGtmx=6iXO^(h3h|FAy*Pw<N*ABrD~emy>vp(3Bj&#x}>S3w=2Bv>D{#QF$b$T(~H@38eK=3fc3Q2'
    '4xx`K>TZgwLUvM}=7`d=|z0PMBo^^C<CA2ld#N`iJFML~|rq7AvD!5{y<j^Uh-'
    'J1{N*kJR8l)V4=dPeN#Zh#QUi=Ll^0n{xmMj&`O^n3!Z+X4d}i``P_w+Ey95A>y*!3OxYp~=+2~k?o!GYVL*3w@x*qxr&Z>3SRrW'
    'Zfc@hh4hI|ftk-YrteTk;{hL^5ePdhvePQl#8IpTt{&}q9eh&61$mbqujtr}xN^^8r-7U>AVf7Pfjti?-'
    'gqa>rt(5R``l=~k2%CX|;6-p4DGFW;XOY6-rLZlWNog=kF)i@xtj1@Y4!hmQz35uZlF{)<4gg3OV|(xMG}{C~*~^-'
    'JIqM?Q3C16<2~*};iN?WPza)Dg^Q~5(>m;c78pmk{2%4hwE8*BhQMy?#(d1P4ah9SJid%s4KpjEfbl^wiBv~2%*JwV+PVH~St!>r'
    '5TYrjHDBqXYpP?6&@5k%Ep&yj*&+EUVZ<!y!>wn-Q%@5@D=jbWr2l4s~jL7qY{r<#H<ntd3K>bnzWC<L8N^mTL<L-'
    'QgD8C@lg^Y#V7z-J!{`ZBo@B4%5OkBvHNpn=#e|Jf9Y*@{g=J>E0De-^P>Ui|+d37u%ba-_P=C^otG$y!sbrfc}cy%P^8F)1m6JE'
    'T6ADiG-'
    'g+0*>n}&8U9LRMtiF#NKG~t0HagM}Txngy_jGQRgCiNi=Rt+9hP1FP1qKYo37jOSqyr48@IaRtr=9k3My<M6WL1wo|vnH&L!-'
    'Na3CJFOj;d_^&@4+h&-'
    '^8#2(M}305c}k?f;XHJR`7tPhSg)zTpNDFGtxZlR^dQ3M+tU3C{d7RRRO04|A{w*ho_(cAk+j02=u^A85|%EhW-pbj}_^FxIWR5?'
    '`Q?uA;IU1CHiM+UJZBgCuv>_t3OKfdRWbqW@T7`Anpw-5bpnk6>P!rK~7Id^Mmlc-'
    '%InOD8!Vgr#L?%&4XdpF3lsUL~%EQOEZf%T&E=6Vs~6ejn9szt0gJlfb?plS&?57_<ZWb$x+as$)6k;8?kGJdDt1TI+b5MJBiV(7'
    'w7J0tMKX(cf(eU8{qV8732?v&0+0p*eX@J1A1spZuhuDnRhqKMgBc^-C$OT{QK^@hgm7|AGqtDW|hc)=&pO2)gu3qyY6k)i2Pc2-'
    'N&pI`E37?Fmva<q{lIH=e^V?FmvZT3iC8(=u#I63_xcHyo;XspJUIQB{?XT<VPZZ*s4{CnQz)LmSh)&w_GB#h>yKYd5{(og|$DkD'
    'E4iovTUm>u!TL}M!7IN#h9@49{H1)jP(BCQ<#iA%jvylK_##zv|ZyVLIt}ufg)6}SraKj1$#A#B2=(dlPN+4J2izORIpJ~i&m$rL'
    'iufbE500W#kNR*l_a!*ph7D?Lbpk?DjcX=rCA+Tpg-1z6==Et2&*T-zJ=A}()=*2UX<oOzjB8Vw44u}-'
    'TDe)_K=YQ2M25~85!&u!t5>4U^DLGeUbmtK6dbkcipk*vzRwCS6ZOZ`OjHgTc~2~u|n?-'
    '3bTtMjeuw0Z<%5C@Ri9AYZGLg!1aPVNT0(TfzKhf{)6LP`UCLIT>|MqX-?g#=JU=Hpk~RS-'
    'j|v7pm3c8APV1IMlR6h!gQ073p7obJ!RzbTq(>x(z-P69y{=sS>X9wW_aXK<r~e2$e+sJHYY^x6a>B_BX0^q-xDKe3WDF0B3}x^-'
    ';=E?70zLu1T9;k=nj`oxw@Y(k^P5I5TjQP4C@uFxM0{OC>0+|y&Vi4I$m`o^h08y&lcuUC-jd+{%CB$AXuvlTOI;w6;=>%Zx>d7k'
    'milBk?)XZd02tb|Ie@j_uv;{1;*@`VYNz_x5H|sFq^^(G}D%_dI~~9xebrYyqK9`yhGMzW`l@bE*p7`AqP7YI!0n<xiwn0I2VjOj'
    '14x`Fq41_T(FIWT(N;>oEtqf(H{^7iy$zFkY!k{7iRmg`l|%*j8~sX^JZAxDZ!=W6?mQh9agJ_*%(&Kh1ncdFGzELSUoSz17Y>7G'
    '!KzGK%fH&s>2#y@Jv6d33kj28Mxde=FSYf&Df;0S?_E#8)tY8l}<UgV=|x2N6G37PV|E%YWL}lp;q9ZfZ$Uh&quc;(I1D!?rlkd3'
    'ZWS7<Jyl6`X%g+e`KaZBq5~?r_O$ckl7uHF4<NQW70wf$(Nk{k2c@~!m7NC(=L6{9D0aYKlB1TMr;6@Mh^!YgqDBMG%kDl=&F1E7'
    '6n%g`9=z?d*p9ZT-`I@L}7KW{2dCudgq&|y4)w<QZ$Gl>*h4&G9(scy#n9M!EO{Nh#Em&D{RumEoM_>!#O6ahkQB--'
    '5igC>yK`Zhrta*KVYEGgMwz^d|f8sMZUuJa4>{}-85lvPRF;&^pN;}yj##q;{Wl!KyPWb6XXo^k@yDZPb-wn{oVv;pX*7eAWM?ME'
    '>~<S@h)B`@cZ@3y(=0ybcY;bdKW}spFe)eMdL3=<O)2055+z_R<mH}KXSe-Y~l-Id6{z*ChmRK9=6Jnirh|!Nfo)81I=-'
    'YT*^V_cttMd2x%rr$`zj^%@j%H@ST-'
    '8Kv5ZU7iA7qlwY2t;N`w!aiI0)SH8q_d|r=;!NT)?G6S!Mp7)k4NpB#M0IVjT28dYZtANMAtO`UlDkRy8<<JKFfnmy~%;P-'
    '@o<$dl((0o89z7zdfx;e@F2kkZq1Eis=QGf&^qK#bfg+7Rb4N(FG~rQip_1Qxg-'
    'drg&PkZ7;A7WQY@Q%@a9xNX%4Vn{SGk`#N>S9%-yEYTQW#*4RW?#^zu-3tY+~e<{7ym4=MpV8(iw`uj$3uJNsw-'
    'SwTDgD?crB@+H~EXezliPknH7Gd)wsQ-qK5W!qFN_)xvMop|M=oOPgx5XzRSCMy!9yOhu7cAIl)#innrKVHQZ(z)r4533HbhK%;q'
    '8Ay&bk@;4NM63(LNq9RFTea&D=yT6|~M3VR#&7qP`LVt6Z45Az&Dd-w#4wvDC93k<91giwTXFKCyF!=%-'
    'j1U4;@N)@ScCP9~jK+ffSkfq-I#~#GQH|jHccqPbBnmW@8~7_NEH?<-'
    'yU5sN4zqz*H7bI`j;tqlqZIpssJPojR3L|+Epsx_N8$851_-'
    'kRHiCa1#|ZONzX==7YYOoqKIVK~Az;Mku{*3E?5^LkPO!V)Xgy$e{kC;~-'
    'SsBx`?~9QtmEshH(RgQU2m~2Z=u8|&{mSmL<yC$hiq37_-'
    'O;NqiUu3Bak;Mq&G4f7KFRLNthjie{;PsJIkmlWp<HKCCcm;R$HZgirTF{l~|bC-LF2A_+PvE)!!t(?H+#hcZu+mJ^kt*66^VU`P'
    'JtVKhEBM^@T(*%04<3M2C#)>NQOhHjiBj5DAFP2v}*uf!3grT3tx(Gs>vsyn6Ch2t;+DvqD`J7o?7^mG2M1%Thiv%=a=+*uc=g$7'
    '5OEYV`y7*Zp^h<$kl(*r76Qc9c;m-|Qr#a-'
    'i8YN=OOD(wk=lgXzt)g3<KmIl*vx^Sod@y?H?}px(SF7*TIt5)M+IpugbL4dY5_$vX9*-'
    'reD7K@(54{;E|9bgDW6B)y{m0rMHg_5R==6Px_cvix4c8#eAxqe;OdXy!fqSq1?}tPxs;GesUMY?>9h%~jK;!0@n_T7t-'
    'gO9)kJ90D^0zc+<%bf5ruo6%D{M3{xrEcD9oP+=BHv&gRw7iOt6Oa1ByVU|g=OmKkjA|fm$F3;bhQsRnyBb5?Y=5JFeaaF#FN{Or'
    'Ucc_%OCf`h@#I^Yr586Doc0Yb!Mghk|E^U$2Zj@QCiYLehZtcNqM?<;+AGwQ0a0i4K&?VM=5^;`*%|J^ORg=&v)*)>1f*EYh-'
    'U7cmOqj*eEcUA*Rs}B+grf5&8LqqIR`(gN2HfgC<JAMVx_51R;#T(=uU@#-$;-'
    'v9PEIav_4E0ZOKXDBeV6!Pj}rcvAD6_V%M=z?=>#rS=Eq^RSefg?szsTb!|G{;|G<H`G7G}$No5v>)f37r3aiHzzAY!Km01#2k16'
    '~Bw1VBeOl>Q;zgDIwvM18*jpwC{uGs|el-uJ{8Y|WCESz$fV}pKgSMixM?yTKf_ZAU}7|zJiPQ++mNyNt|BEC_YtHN!8M6V7jxcl'
    'pZP(e;NgcZo}#;^kU-'
    'V|1_qvtvg#8+W&#nLMSo#r8OkwH+mBc_BQDt90hjH}{zBCN5X9=mGHg4BoTd;=eK62?d;i3FL@5*@L2d*ZA;iXTl0CFu1eiLgc5y'
    'a8N+iaAcyV0m74PnZ+4Pf%iZSR!TEn=@5}%K@btfyvBei9xzNF-TV=PQtXrNtm7(s~L&QaAlH$0iS$=U~t(j2qb0j-'
    'fal+soBW0B*ZMF%^uCn7!~IsN^y+gluz(TW$cj-'
    '#n=G!nct`I!2C9KUUj?RAk{yHW*L`c=s|$8I>tQ+(C)|6ULGDxIiHGwn7`?K3W5P>F&S&Iw3vjT0a{E%JOM2xU;&pF;}Lg2iz-'
    '4AB3y3->w8|rmE0+q--fSVWg)J_?&yMapUX^}jI!EKgsisP=3^AX#J&GCUxt!Uld}l3YB4sAxLeyftAU9`_?J|;Z4wVq>ds2s4@X'
    'KX+}ki(^THjm4VPBhGzyo*o=B4MC3FCMGNKdA7#KV6QZ_^6tSE`G0Sg_lBi4B#)oZ(*rWN{zqfgzbP#y)Rz@oiFdAhqWf=Xf75+S'
    'J;h7BT$iecCwqNy5&4I&wp!w{vv9o56ITV#7z5Oa^?J~W<D(*Az*N1~+tLuf#wGRMQ{lSE|>iJI*=C*wO22;gOWLamI!BdbQWvP8'
    'rX++mvCea2A=H^!`xe+NmD-`I*y1SbYTA&*jc26ciGO;W=e?Gm55s9s^yN>PR90d&5i3eSV+Nko}}N6-Ny2nK^tu>+-'
    'V=s@WgI#3!z2TK3YfifU;pbQKhD1#CgVy?6qKDfFa6A50VBNNRF2JAaQ^Me2IT@^!|BNWz)B8KihCjv^<!k;7P?lU5WKZ7+nI4Ft'
    '*`2zEn<SO|-626kP#ur(|B?v#Wmam{0ejC@jHr70Uh<Rj45I~^N!GZr)Mb&^tTL2xs4R|_X>r#(<UD52_iYXi`*?E|-'
    '@sd3;kt`VeZv|O`Yw+!`0<Y~FURh9QhopAKMdocu{fvvvCP@vAOUyfxIvSUn&62SNU9nq%D>HjY_{vqAy(E0~4k3alt!Gh?=>KJ='
    'o&EAPRFNY~)*eBf3NU|Hi>cp6(xnn5T^gNJ2D>pmQT5WM1`|7m@s>H%1-'
    'BsdE?%%FC4#*MEia4mYH9u}tge#gzg1jmSr?$x;7q;18D{XMUgQijxKb|(*D7o4S-'
    'VU4%1D_#C4BXM970stH&}>xN%q_pA!5~e3L!&@nw;I%61CGdph#eEj&;)N>OvX{#IWe<Lqo8ub3_vUcSG4)3J-'
    'W_Hu{h8tM(bJuPKj%0sT{qsQwdqVJ9bo{jN0s5rp?0Y5p^;ZcmC&w?$Z#CiIw|A@CTwv(FNA5WU&wJzfx-'
    'b{CqBk~(phnJr#$%Med$9a#rM<lbIv;F*-'
    'de!*;lM0sb;L=_LZJ0#hnCQPoWHlQge$$TAq3+2eWb$Qe|m`A1Ky%E#1_IPi?WNeLB`)MrCwn(0%;`S5^zTn&sLqRu02dV+fU9G~'
    '`D;&^;4^jIz3&9&~fW(-3vh`&wt5Z<gSAdSC2hxZYSqeq_V?mZ;(E$Qf3LS`nSZkyMF$gQQ_CPqB^%S~nw)AZQ$@3-'
    'URhEMnnJ;(OtH4#vSGnu8;1A|&gDuKef?Y(3=BvRMs1fMKSfI~D0{u!PP%9D6_}glGiE%Ets_o`z!tJp^gse-aWW(<B=PG*c5PUW'
    '(``qtQW`;`YVW6Yi1!YPWz=o$($wJuglq*>Tc9D`Li(%_iwqyxt9!i%i1;s=8l4YQG-'
    '1Q3Z!1EREdL=mE`AT=a8eHytwYy#e9(TTGT0waEUXkqn5XtV8THk4N7A3I%0zS4%>nY{9APVcKN)iF=QJa@JwN^%rG@9J1>NfgOU'
    '5>c8Wo>sjIX{cdVT>zYVumpMbg4PqYiZjV+RiAazE<aS=T-|$Gjc;D<{7!~0uzng_=lNB?rO$VBX<R3t}*hR6xL+yR>-'
    'gd<3*Y$`tsY-{6fWuiTe^f-'
    '~$Pq=n)y&`_{l|usm=YtO%S2D+8y&s=#TmI&d1S37iHFXqG<cFR}=Dy(zPa_N0t5hoP5a1D=lj<Vpk^wCGgMSyRl#Ocg0c1%s!w-'
    '3K!pq)fNN>Xnz|S8F%tV;T5v^qJSWYGp{)4F{Xd&7CKJlfzv%J1VoEqT=ap${ehy(-T~CaP8-'
    'N%K|Y9SgN1+6+|dts_ybDaM~f#zuT|CWe3akQ@?`P1zd-p`4vPi;5*#oSK!Qp^YC*S!!TQ^-97HT{qlC(k%;JBhxR{eq-'
    'u=}{v7Ff66VT$$241JZAr*nn`C>A1iytlU{3@`kXw^>P-'
    'b66vfN3T{T0cws^CQ^{K*Pl^na!(cu`e)s)Co7GA4*wTjlStvdTB3uCm4;-{Ne+E`0L2djes@>q2S|{k8vp-'
    'L|&KENgb3b_N^_&pY7J_|11}k`}?|MdmDji>%p-'
    'd?V@ORrw~eYyS>Djs(4OuZ#dHD4DnquMPcKN_yNYVGTj+2A|Vgm$ttfaHfkz&J7|4?vrWmu*VKA-Uf~4C3IS_yY5%dSBzdq@7vKc'
    '`25uz5%t&NhwM~7p0Qc@sMy{TvKOV06wyAJ4>C2EX>+7sG?>dZgjrs$<_c|&_KTionuf5;%hpWS5Z!v|ni<+0=NEm<mD(KdQA>4p'
    'pZR2n5TFex8KMLnf69Mk0?s?8JPNhFuDT}WHXht%C%DbvI(xy5E(&fPMl1Bp>SfyTGBzMB@yV8{6>JWFQtJOK?DYB8@vWD!3n&FB'
    'HA?De0a}dmINJ$OTa>!lUVzr}z>`4LWFzxniWUY_v?!RO#laLU38rXiFh#y$$_r>Bgl10jPm$)AB%!Y*N@^GcuiWU|MIjc_FLqVt'
    'AXYfz6}(7eO;GS6g+Ecji_{mW_{W%50`&HiOm9-J%sPJS{gk<%-'
    '(Y`b9^f}PSeZwxVvAAr^^$aWOT3xO5%&?A)OErZ?tL%B91+n^*q*-vt{%?GR%I)fyrU*S$mMp_7Kn1ayiQMA-'
    'rJSkL_r&kj;!;XxRt2hYEr-PEdC|r`xSLjgmbvUx}c=dSCTD6mMOA?NGN4(or`EVbJaIwuKI<{Rb$9p^$(e=0U>iWFl4R<QRXVj#'
    '441~P;GOhETO5={Ew5+TjpCNp|2OuT(bf>F+&LE*Jp(=Y*yebDw8-'
    'cG3)Iw%F2w@pjcD%U#pFK$V@CJKLV+@$TvXlE%ppR{ByC)=2rVka|@Q){Qi^XRxGpm4JgfRSY~sp1EiUUWj41uP@3DZ%;r`HNplC'
    '5+5BFW=1wfL)hLH`Hn&`f<d$zVS2?k5GFOw>&PWa-_(Q+vc4csf9%s%V@P?jX&Y**od6-'
    'Tm?CY|iti=Lk4D$8kh1mJ*3&@L*p2Sy>7h@B&FCi~M5)NNOUWx=mzKFaG`FXa)pdVKuDN*eVjR?j2%<cE(T|yX?z=^4;W<fLo+hh'
    'Zlnxg#$&tj2{0KqEGRbEhMCW2arovx%w!2f%iS(6S_<{|Gd;kN6Pp$p|*8M;v3lc5XceHpq?K9Hdc<wF^|P(G5O3*}?!G<O3spVV'
    '6ltfVZm6>ll-GGnrX5(U96<ia#z>patBY*3+-'
    'CaEm6))t#4LEz&Q`($bUoA%RLiJ0D!rn@%7b;ye=^KE7G;*OLGUVjtJN(HaK3#J2t*XMq<uQK<s=08B02OZ|oz2|$FhmFb%A7Y+6'
    'Dl>e9MTMyS@nbA4xZHaKLgTu`5_GxhPPq;56U`)7kx*)O*EqYqL9vkWdQ7pUd@Dkttv|%_t4iBwTD5)h%V?XOohFYEPFWY)2kSK>'
    '#$0kM+U%tyG_PRuApgHQM5s6lr6h?oM%5;fhAZ#^yjOIB0?Y56q7xO^dtU}RS%H=Ju+^$$cNl{+@`n-'
    '2N)9o`Y2*=OI6^KlMkC}CBixmoV$ks96?@gS3G|`2iFcBy#w2N>Dngb(Vb51wV2#Z*13Z1MwJZK?og^Q-D&xF0E(@0N-'
    '^0Pz1kab7v&;wuE|+)RMk;W$yyte30{80L0`6PlW`LXSjC}{N6ibv3bZ#d(eIxN5;4qi9(_Bq%chTXmBxO=~j~4iTficC>LfOQ0('
    'yr5KwR3Pg-o;vJR0(_^E2dF7>;o*A*mBsvt^5^EUkSH)L3=@%bDK-'
    '@{Y07FNqIFT%4?%E4cd$d%4;2*78Tv)ead`G#q~~Qu2PXAdQh3GRaEqQNSW`bNHIO4%r%PO!4{?Nn;k7R8oWLgOkM}C&jge7!Rrg'
    '3GyDi^sAt9c3ylAI?@c$kJPz^RV9+){XodCG?XItB2xUbVZ@7DvD_5gp<oyPI%3>!*h()@R7{8f_5#pwjHp7A#A6DkOit@<sV#)Y'
    'x++hNW%I`TFT@tO6K+>C+eQf|qZ(gB>1JF;e`Z@uU-n>S=2cWB7_rys^dh-UgA%Na`GZvOb;89(@6r6e6x-wM!YuCatJj|-'
    ';AwqaDI0@0|1aI{g<wM0g=y0BsC^1mUTV)JehR`T&AA@%!pLSd3+(KXL3Vc?Zr0J>6NDnjEEKP51PV%cQ()7{hWL<C;Fu&cMBEf#'
    '3=T4{Wb`OdK`-7f4gL2!wC=wh9dhRUBZ1<s#kCPA=ghw=SzBHbwBFv)R!K<yidMiok++;_+BTX-DPSi|b39o<n!|^Fr=D>z|bT8K'
    '8z>0Z1FV>R5j(I#UVopG}dQFHJCMul_N7x;ePDUW|j%ryaAX<#-C?_Idj7ldbBjP3@RPuB_GizK4o0ySY-'
    's@n$f<m~hd@3<$MCZD=xl~P#>uyE4>YppK4;{tx62X0jg-S3p9_);TU@$Zu(2UhLFznv^Ad-kmM7E=mk|wZ?IcEZDl)M8CHk$0!_'
    'F$&TJGkD`LhdgaxOF3<1Q_CVGvWdg5~?pL-'
    '&rIt!hM|ER<&{M)<QzbGrhGW#ZpT)w!mqryOd2sry&5m^DOXi3G#n_bRwJZcebzsEd}Dzh(fLb&FENZx`+b*j7{potE>CmQNgK-o'
    'kgJr5*SDq1P0QDfq`^UU?5!_7)X}{2GXU0fpnRp^|HAjI78%5WZ)8z_qH8qAQw&8a)|2}U}^nS`DCg2nF>XZv5B67;IfP6wn_^i='
    '7)wg0S>!tAMQ+0h738J6Zv6XWWcotZ^FVAqvqZ2=H<9~x52y;H}Cc^ug1;0J<V%z^KLKmdfdF*+q@As@AffodL9(_)T)6hXhI+bn'
    'jA=hrbHQeh$nY)6%DKe%r101A!^TU3A?Gr6-'
    't=s#<OBT@2ZtEr5zwIYTD?)fAApJVODly2u9vT7Nd+3s)b^dCb7I>Kj!vWqKK=RNRBi<Fd|J1j7XCL_t8|+LtPkAv%u^iIZ0@Cl$'
    '<3rJ4sFxnw=%*3C%8&6NP41$(cg4n@p=#zIvH=hAJ>#+YNYum0;okH<RrSIPr=|$a68%Jxo#vuukTm`1)-'
    'FDttJ$_RRXWcxn!FUGXv5mgA#kfQ=<}W2mTQ5@mpmr4}YQQ`}I4#(a+E<e(7d#~Qwp?_zGy@Rd6!Z`AOWDVZ)u&BctS)9{Qk*W7e'
    'Gr!g`_;^Y9#BO<Zt?3y}p5bmczxEB_p&Y{vb`q^g93fF1YM?qBsy8LDMeweXPs{@jTvc7^5HI}fJF-n#gkMT8@YMRiYNe@%pw-om'
    '$QBRU{56yKNzH&$E^%}mCD`;-'
    'g@RfT~Z`SZN(tVTBK3VBaMcXvH&>+%+y%=L;4YX!sv}N0ZR4+;neO(0kz*>4aAzLhSW!=I!)`x6!)RJrwY;zU1{FGza4i@<8E^9l'
    'z;_j}vi>sAb+%b>2Ae6fnGW-DR&y@UKOqH+ISbwH8=h9HxTjol8Y~O~;2vAWK7OaQG5@ZP62GwJb-pfIH7Z+eOC1!#tqmd7TL~zB'
    'akFsELPe=Od0XGnW%!iOab&PGK<b4Xui}t?~HYxK`{Yt_)X9l%jOW3l^uJ#)VTh^_9Ce3dp=m6$ndsM;(cB`LD^E(L|fj~2D44PS'
    'Go?xKN8uOH7VVb99L7FWwJO$qTAZ;tbTd@(#3+s}&@pCSz0QkuJ>vT3tOTa^cZk$8m7FMM^sJvcA(PEviE0!a@+GJJbm{^eCm3ZI'
    'ZrOY^Hff*0fUj#hRAc40=vu~KBZ+sx>n-'
    'EC)CI*teNr9wqav<rO5=i={7O4pFwDD<030<0GL0YZ#Xq99qh|JXba>Hb77s|}o5j&dh4x-'
    ')Egm9rp75P;l&Wa{kq6az{$#gmMe+UxzaOVF|%^vcB(hSk;As;Nw5!!ml9@hxF@+!ajw!~feZOK4A*oWWob<XjYy5GC$Q!I7=waJ'
    '#Mgag_Q>mHtAP+4pNimO^Er(h~lM-|L+Qj_i?0}vndb-'
    'j$6u{zlOv(<T5V+ng_rfm(M>F?uP63^kcB;z{atX=I_*GN21*AQ(e_LEaZ01oaBijY8XZg*6K1cEcLlOiM#oZ6igA%Wn~?xF|@1j'
    'lMuMMxkxvb(7S3G`iUmbwrk*EY_I4MWw8Rpt|AW@*3I#wSsi!!)<67(M!oit(b)`7~Q6g2?@;cpm}e{s4T4xOvZ{FkB#_xUUa>M+'
    'odp>gQJ{2*ejR`qfB*eM<fP>O_GU?*V>wvOw(dAYDU6xmDIS-'
    ';H&>2!s_Hicc%HWL<2Pi+b@wMU=wX^$b_&v~8Py1Nhws4{;ZVY*DbG{q~zs13l1G*n~4$*7%u*Nn)DizD~@~b%a?%MVjZZ>4srjE'
    'WD2=TM&qNAE+5(Hed141#b=l>VsKRlo^=P9nV;If+_SP>={HPjRZY|aHKyW&ma~VK)^EyL<SP>45E-'
    'h9(ESN&7+_+%8Yb+o&@b&d6d+|*vW8cclH*1oB<Dr;;kqR-e^W4=7K`iQp{$FIfjyeA1HWmKT)j2z<s-0vD)g74e90p$u2R(j-'
    '<rEnFG~$kYt*vVc6)!yLmW5<asxbz*aln%_ksWzJ}yBqO&EE`k>j$`r0Hv`GFHzi*`X8ifmFUvdAs7-'
    '4bypsdcy{`ChU5N6NHT&J@n);vz|FXBk2uRMen19?<MQd{vtVHM<W+n}>9CU}i?S^n5&M!zlcn00tlme<y+kh{E4Vpe&>CcQROnD'
    'EyrQCOQg#r-'
    'CkZ+*R?$>nLnIBuq68+A2S=P9KsKCnxx+Qpg;351GS;kU8uTGKW1w=CD`D9QF>G!#<ojgu5Tkl%x}ow#%nPCL=4C&wflr4zBZKxC'
    'V9Eqpi|0sbp)^V{a-'
    'M)=<Ei_I$n_`0QsfG3Zb`Re<$Wr^Z$UB#6}DO%~#PoivY6DY~v6)PvdPxspajjn#O}wrURV;iJw9`;J;uDwdct#vwF0MmL`dn279'
    'UR31U<GM^rrf<$HKu_xS089Y0G#JdJ|{>b;qO#q+X!-MQ>k=c}k7|Qkiw=w(a>-'
    'g_r^7EaVb`EeN;l|qH{j*h@!!t)hm6;ye@1LnPr*NBjKRl9i*(wIAdDLXBG<mMIhQGzCmVGq^La18y(-'
    'b_S!aqn;FoeeNVC@6QCO3!0XEp5{7N6C$b69@lLoCb>dz`~YOb>gU!zRp{dThgHOnG}e!xqd9Z>b^623a+0Q`M2IU|n&ts%mp6p}'
    '3p3y`laI;K7W}obC^W7mIPMkC;*l-EG5=1EyFHZgd-yDONx!@2@GOL>J=#%^41GF%Hx|C~v~uOk?ceLGT<qcnmzq4&I6<*})s}C_'
    '8u?o@ECQiig?3V{+5%0>7CygPpiWW#%vv*Ja7pN+`an#gYmd$zf5iH1Drr(j7fIk?q2wkirJ*B12LPX-YUNhHM#}7DKWO&Wj;q2`'
    '9slID@lc$X?>oVLmVg<GfCDVEPqp?$aEYo}$hDn#0XgwRu=`xOtj3kLbvA1<QPflMvj^Az86ks5Nn1FOPHI(TGF1{rOQ{${9S~YG'
    'blk!dBt(kB-ts%7rGl6I4QeJ3OTEWcg?PutqiHANC^}^-'
    '$<PnUA@=sQe0I86FpzK`<j?90(ZQ6XK#XnB9?a4<;Di6JwkRi1MTuCj!Dexo96z!?g>dg=0UXpD?F8A&<?>5ZZ=Uq##E@onNS%aM'
    'UPiCY&`&rwfOT(%Hgkqja)x+$iZToHt6E3kQxmaow*`o%@6Jphn&0PtzXKd^01Jwzo9rWiA>debS0Lwk_lV#_G6Tt?ao=yf;)Y*;'
    '7Ic1*J2ps3b<k8xP#YuVvP209Wzr1%?N^gOak#w4%okwvs7c(7?7vT$vg_Bk#YI%PX2j>GWf<l#Q7rS?J;g!K~9f7OM$dHM9d>Gj'
    ')}?#l895U7BxXWIww})0B|`-'
    'BX%#GO`wXNpo&S1($uKIX|O{7;M{@71v*3)4rn2;UUfRsxni;*?vQr&tsv2R+#J3`EbtfBv)*Dnj5!OVo>L4YJ98?g_%Z|LmN$Ls'
    'V>s@zM`uT<$J*fwP3oDwg9>UW#<(IEy$T1RC?UD2WpN|HK^F8GWj3A1Gl4tQ_tP#dB4z%o;QYGbny1M6+IsiI??lip%ERzfZ35*f'
    'De6y+k2oK?;tVFNIh2EUL48|k3zW-Q7Csp6v~Z^Lb(&8Q0}BClsj3_Og1nFl-EK7K-'
    'nye3Mie0>VJv@Z_rF9>g$A)R@G@!GgdJ4HVSUl=()PR>RINJ!KARghq?}DD~l_?GzLn?db%X4H;G+>N%ZVyBFqGWkJm*eJuC1OCc'
    '9MS2He!M)66UU+l&wpy{62m83#nKD|1Q4ChbiRcct)Xole4|`*ac>-LI4I=mDLCM-'
    'S>GJbFkc;nBl736CDpR{f*AW=4Cb$$X+CQiVV2h)3a1nt2qgd-PRB&)OxMH|sdrJTOT%uaR-'
    'G`M4}jHb1K4Wb<}eoNRt7Nj4uRw>7Ul35wFOo9;-'
    'ma%~d@W|H}`(0mNHG0DY);FjD{D1EkqK1h6uZ*ALmb+l7e(m<4mu!I#usg2W)lZmKY=S#<m(JBi7w4v4)3%<dT)*TDJ!BN&D3%<c'
    'IHab6mY*$GTe=Y2%?$|#-Tqg~Ph$a@49*Br0ew3bwh$iNgUf5<poGQHqJrdYMeGnX61K_9>$pp$^j-'
    '19CD9l;pL0nZdit9DP5$gYWtMf>F2vHYjr=4UN8OU0jXvscxEv48<#0{B?G^erg6`)*4Ti-UgSjSrTHn>#BSr0e3RL5H<_uxQsNQ'
    '?^zrtdC7vq4W0fc3osChjpF4P~Y8+QLKRG}Z0OpWw?&@Y_k+f4wprG<<b(X|}@xXL=Z~uMp2PU42LUI;3dkp_wW`xm~y9tP~!~+q'
    '!$0A>R;Y$oB{{<a>r0^1Z?g`QBlMd>@)2?;0{1L}<w9AwrWuZxI?Y`iL5SprIN+baTYWi1$80Qx^L+64BL3nKE&v%oudvXL_hoqZ'
    'zF+*)9IL5khoAn@IDmCn4g@T%u{B0P>38C%I1sc4$fQbdmR@X;3_e<O6AXDxOjDu{3=YBPQE8OJdI{`5yM1lD)M@lROc7G%4h=N3'
    '%Gfp@f1Pqv2Mg)A=)eIdGK>sn_o$f}0}{w>8r{(w@Vq${bF?scN<qp);wI+^OFxGkKKKL0|9@aSFjBlE<yQD@}LBgIV5}ribF8A|'
    'FZ9J2sW0Q&ai1)Ks3Hn#!-'
    'Irt*x`RGyid%Ck~adA2TQ60J@zrJWVe3e)5^;2>enCN1~f#20Ce^U%&4EvF6!TQJ_HZlrC3;1)#y!f<RvFyp9HqVaVhl|v8$X)g7'
    'vHruMcPST_g9H?`@q*)(0;^hHJ(>`!|#DkKCW8n0Nha`>1z|lJohbkYwG}ra;rFrbohtl*?g^h@aHxD5q`3(@}OlKud(dJg#hu75'
    'B>Z)-'
    'QSeLj7tl18PnOQ?6<qY9WDu*{&U|gc)FxfH;BQGD5A_UW1thwHaFU?g<d=aq~zLQ7t+TkD`$!mv$cqFeK4&sr#b~uPf^4j4b9?5H'
    'mgLovb9S-tl6d64y%s1WEn(rtkt!QQ?U_Rt4AkU)HwyZiiaFb>mcea<qV9;tjmAusw*O!axR-'
    'k3Bv7QIn5wHj8v{BEu5PU7P5S$)b2)-'
    'U#2+jyC1ZRd8g0n&k!PzlZe6Wb&utO58_z^M2<pw0kvtBto?D3S1{a#`xxAORx28@+E15BXKc9wN&P6~wVwiQ(11!_g>Ss(A8{p_'
    'UI{WSNu;fs;S)qp%6ACSir0`hobKpsyD$m7WYc|0W`kEfDwT#wPANmtR~q9zmD1*sU_xJ;7MjYu76H<Z!lcG^(iPjpoo8|B073FI'
    'z~{j5@qh$?MsHYLIhDw#{81VQ!3J}AbQ3_x@<<4Xo1^q&wVXh|OnTtrWzsbi+dik+86k^3QbUfNNC92RWH(SaNmY{zjT?nJ00y$6'
    'RRO<ISgXx6PU+UI&?W3*zsTZVe>50dTG68cv?m>{Yqx5c3Q1SM6CP*N5vKRQFcwzx8ay1PWQ?I@b=M8S`Gyn#sEQH}Q?LVncZJ&D'
    'pC1=vebk^yvhZ{q4lL)?eS`b(IM<O)JBOeJzpW^YU+nm7Ea52g`qs9OU&E~@u-0iK7I-'
    'F3)t>1FrB5PCXAm0_2UHOBC%S}`R@L4J2ki4qRSG+;v1yy4GVPfUM8xXAjhyGUcC9wK$-dkTt?+`5-'
    'Yo%`N`oO=gk0K538Vm2r6S)NiUvb!~?QLLg<z&A&PMuyOZV&`I_n^b1-'
    ')M&&26vt`Jtz~L_L6~!$7#Cqa&5Q9zMKLOD^Nj9SV>63yO2Vlt`!ajdYQ(Hlj(ROZ)`=&~0Ym2`z|iXwVCanrF!bh-'
    'J1L1YWNi>>h}uW^Sm~b!^A-'
    'pLnx`#7pO;DJRAbhKZOk5p)3UC%m0+c`91v9t>f);TwMF?!X%@!qPTOM}BegR*xaW_>zi%X~ht{+r>+F}0RECHzz)wmlVucDLMUx'
    '?|Z3_7#9TjVV_;!9Rugx>Ue8X+EuWB=oL^Zo8syb|HiJrB<j)dswZHLOk9i=9-0+~}Ok-rj~CnyoU3i%c(tG@;b7Ks+g;c2f=@U%'
    'B1c-ou%ss~m@m_K;2bzs+s_@K6YncYG+r;7_IojYF7<*zw{#-'
    'B#OxHzbqUrU{_O4>Gb3#B(t(lMP=lgqA?p89lLNQNTvxK;dXBVnCbjtlIt*f0jd&WWjMKkyXC^|7A(aXgN-'
    '0grgJ?|Y!&Q@weENcHB8BGsEW#p77}pjx1Fv%rSNHz_6;gxG^>1xsvvu{6-;LVU4ZTpd_0Egym-'
    'Vt#`p`r`t)yG2D#uYI08f<T}|B8F{_OIvN(bTkPSlTbB7$Ez-H9A~v&66Sn&Jl0A_X2CvA$!gBlklO)uu{`)3phs5-cHv-euM8dt'
    '?Cn*-'
    '+XWrFI=FYR)7J#w7c}l#!EIb4WYorWf{fa@PLxp_*U479ThU#YMEIowg1Z#eKubx+4T~mOAd@JIC|YfDTeLvuG6$(iYL3YB!kpve'
    'w?;a88FqAr*=lHLSR2mPYNV}mp2MAYCy|zLCxyIhM6Z#Xjp#M<vk|>Ujy9s#$kRsj8oAntUL#){(Q9XWd0n4m$^WdGM%cm#Mltd6'
    'aACfp+;49(U(>koAO1{jjB*d75m_&e<zLzoAnOYPWPNb}r!OT$dJ{G{g5P!?r`qq+ma2mRLUyMUBOq?>7GDZ#@>+qZyUJ1btUUyY'
    'pOPSsHO7ENUZph-'
    'H)@dzPt%FKAt^*H45<4>gt~7+z<Np1^X;Y4DLt1GOFYgh?~XMziX^dyMxi9u&?uI~8X5(YSVN;|5^HD_PGSwM0HJPHs2%P_UX)n('
    '-m4e``A_)F5WyG}Zz)o|*7@s*YsEHq`?wR+C=G6-7t@W2n3jazgJrJaz?KR%I`aM`_tEH~pC1}WO40C}?oks-L)0G8BkI-'
    'Z88w&mirOZ6M~x<ZM8VteMC~J;J_{Z6t-?hK5mWBHs<z$QX~J6HtOcjBArf_F3<6z(#W5=>c<(1jQa2??gA)awp)-'
    'ArRbveF4X$LDxo))wR^}-N+7po?QSo{JQ-1F)J$&3(vIoVK-'
    '_4Q+4fwg85@+oOTY06#nYjlu33SO%KmUJe_v|WpjNnX#{1r4E#f@!a&$fd&Dibi1MAr}e+c~lHR!B#I(+%})8%gixB&P3(Xg)%Aq'
    'XCoWTeWZUO!_~Y>_qxmq@_CI`nZJ%^mElgKgm?$1mfU5Tp@7^aWo57;`Cu3-'
    '2F^p?)pc}sLh8RGpT&qk(HZ{vh7$cF^=IGh5yGt{pS^G0&f^zRQP|k(UL9kYkMp;<n45BMafMQvQnFyb~nLR)l8}3<{o+|W!DTf+'
    '2|VbtZgH-BN8L4P-}1+yG!nEXvEwJwQZWg88<YVLo)8+G>2x~-'
    'DnQWxV_N~$+)@G9G<!EPCFbY<Gx4CDzYL)Z>A3qF?>OxDm^#iC|OQj#w}Rf_ENefNqkf!P2i5EGNWn1nt&=o96^D#Sqy5J7Y6rRU'
    'lYRJYe~c^aiYB^V*l-j>5bU-*m0PT$AWagPZZYFe1_yMg*6nPA-'
    'P*&4aIARpDL`jdF^nI!WxQK2tQX?Lz&G}9@x}i6i&9L$vmrYyL#{IIfX*?I{PJsLM4am)<luPjvuE@U*Cz?pfQ2d#5rZ2Tzna_Lv'
    '-hy@<KvWU4iTnJvgVln9x+yksYEJ=aiQcn(9hqhv*Y#3`?8|9{l(j7$~`M1u<D}a0p{8W;r%^!xTJ9BBlYeIt=FwP#9IUF#SUEzX'
    'cy^McWZ({7_d*%U%gr4cPYKci3%-'
    '$Qq=Pnb{cHLrzhoB~B9NR7F~1v_^N#Yl+h}{+NF{U)LsQi`r9TYU~`^1w5b>G?SlD>SR;K!H<z00LS}e&IQ~p5l?k@aY#)&?%Ak3'
    'RSJ{pF`-cwG%V2?hB8l4iTsMLGH}?%fk-'
    'n>bh7OoEW94r8{_)sl(~$AH!tl9vr_;*(cG}x!SgukkVrEr7hqUQoQ`=VZ98ri#rXkcyj^&c`8w8R^SNCGPahA3g~P{FVd1#(09Z'
    'I}ZUU_T2oN$2kJAVcG>wQeG7vVM5NEg`a2gpC(m?2RVoXQ_!P801%rYKh(cNVfUiQKkA_sv7wLv1hJrN<OmQG@^!H#t5zizN|9ki'
    'LFDdSQhY-wlQfPqwy+GqVju-H=?otaIVGpgHXW{c)&nOAG`yylTzA84DPcIu60?&3LX2|H>6hSq-iW<C+KmwxzWJ_&Ooe)eWQ8Dn'
    'lgdNZGbA@|;rfE~+la77KBT%9c<wL}}Q54KuS!{+ZvYIh`&(%IUaMvg0H<3G{_2?Q}!9Ls8BehB<s=q?Y1zl+@E5cs=<cymFRKE%'
    '3D=I@fT@Sw~;AhrCk%s-+6_#-'
    'm^m{MuBU&Zu}$v7|~t*Qv;Yw0s*5`H4>oPj4j7Qea9KW+!L@SGUvLX1%!wiEhVW4itB*w^MDv<l)^FQz*a<pzp!tTyG)9|Vf(>+V'
    'VaU6_lw^`VNAAWA8e1UHb}FPT)zJRn2c#DkJ)rOZQ;d8JTVL@b5UIzIw#n^X0+(vDX5cLv=L0Zmzut6#wm+c+m1@fmjpK4Th;EA5'
    'nPHyC%CD)N@#zs5kT`H8H+{9dfh02P9m^K=Me&X3!TMx`KuucRP>Q&NyXo`M9vnt}vO3KBRq1qqy{$s}}0eD%)a-'
    'BuPUFK$VN!hcD7x<Vw3=i!bhvQF;ro-'
    '0{zMDqkCRtSn|+ubws2Zu}qLo<Rl+fNb!4i@HJO=VRZO^(qan*4?i(PWbj(d0LEh$hd`A(}kbxB1}IF0M}vM*1=Gz79nWA9A=3lR'
    'V(H$sCTO2x7=~oS1O>GvTA%RzMCWHM7HLY3Fs;#!O@MHfmD)Wah8j{MC!f#dj5fX~VM6Trw)-'
    '3<0{ioq=t&946z!gDhqAKbp1U`l>1I)Oi1gNS;L8WR5_xBj<eD;yyF!^%a4`u%8I5dW|CR8TJ=}VQ+v4+=c^%W7~5hFapn^?c-'
    'p`>Rl26_jkqSanH>BS=02jH%ZaoohbS<G$KOJiBbZivgq*jHDAf1!`IK8l0}EF(d1cl`1+f#X3^mrU`!SrzJcb{EINFH%xRgQH)V'
    'z(WeBx0fqn6aHw#f+r!7NVWCJW^HbX=@al26hbiItwb1hL!?ud^Z-'
    'GVQZHS6Np#4a`@_{USu`LeZ^3KLU8(hc808U8dualkvR69mqH4}tDNyNV|+chbHZqcPm6_^$Z5GE4_=S2%9f9_hm5$vL$^4jbTCs'
    'ebhq^`TCH(?5ZVR-0^C$;5i)I4fdlB_cn~5=%sWnx&SA{4`505&3DBULx|-'
    'EWt!nL9i4Pu2jgARu&P7+Et|9ewGC6LnYdF_N2$CX#6Z{%`s@qD>8<xr9^aNbO3%2m41jG32Y)L!--VHdnZsRZ6*rLsbGetQvDKi'
    '!CXm={P_ZNi=;;We4)8jQX_x9$lNBWkw0H-=1FSg&zG3nB{lNrOU)gU8u{~O=1v)7(xvS-m#BN6Nlhwz3V3iaXT&{)-s-'
    'kcS&9=tSabq$ed2C{v|A`UwuzXaYz}7hu<JCTBIzYfmYH`nXLl~prk~>U&L!FmRGfvlRGUFQC_YPLP<I|<Jl@lc)fghohuR~7ns8'
    'LpG@il)o}}rPu`6lz$k>-'
    'Odu0sv@8T*^2~Jq?unE$8VA9&%nT7irPP^^rg{o7ljOngM7pOlQ9B#bLJC6uO&fKV`22292cT7!D^+=k!%yekZ(p;!bqhf5!`<fv'
    '?mN**xR`<IH=YzrcyX(X9!$4=d>%;TI&EEd{i2Mk!dc<|bkYm_zPjGUhAGi(5K`pWRsvyoX_vqpnNgNkcFsES4Uv!C?KFZuo3hv%'
    '7Z&<XG>4N@c=uiV@e%8kc$x28_R%!sa&x3!gO@IG^_mJk>+B~Ou8uZ87g2Zs6d@9U1K|xbXfY`!OWP~l8L<ZQx*<*Yy96g5D!pVD'
    'DP;~J#iH&|VKII;XoqqSE(Lig2BJy)OFny(93wkO2M-'
    'pk`h!QH=Y*5VJ=wfl!Ya2vLRbpW%+*F2qrfzsfkkpaLU2E=BjJN%XVj|Xw$XR8_yZJGuQv{dNbA<^fpDHv8vrHaze5%4UlTW-nB='
    'I}vQW35@Q>kHTo<uqT^a1^E+cIkD6XHo_v9K1KomG+7{XYw;D~89m=`ZydJLBGu3dHhZh!!`71uc)hgP$yn8w_5DNsf9df-'
    's+7K{uK}pQoULBGhM&=agWdU&XkcaGwS>7Xd#{#c-'
    'IApQqtgMd+u+*eG+NqLT|s5wlK9BRl9uD7XPcBglFEuXG?T1SXdT_u;#C#EnNa4Ogab#+|qDn+9Z#slk%9RQPDTi#O<><9m32rxS'
    '_Gp?pj}z`Hwx2wjHok@*Pk%^#VM@z%~R)|b|K|D~+((aqvmzwCmaCf`sphDAS0d*He4cP^{iT74N^uL5og3Hh9ALLyKS)-'
    'Q@!mT%c99Ko6Sww=A<DcwO0HfBt8G1neT5A)h;yDfaO>mLxE(kVbB;@ygQ;D_2F8qGQ=<2MsPl~FuaaFvhR8!<O~cm`i%&$UvBC%'
    'eeddDQ9fjZP(WQUvS(=T-j5U8HXb<^8n8?BYZ;5;22bME{-ar>f-'
    'xVvZ_viPt2zC?pN=8JU5~JQYm3Ly?VAF?8xHD#nzZqGFs+u40_eS5=HKJ55D5;+bD85kI`v#2E`W57B2kW1?uJ_lz?;tymQLSNeh'
    '-tWO+G2G8b*AynX*CmD-h7WdY7?69c=-DmD5%~jew>-'
    'W;$mW1g=N^C3BFXJZolav{laTEN>$_&bE6TI&r`Kj__1)4BDNU0v@47{9Z_MIdTO`VKrM{ivaW#+a)6!C)pF^bE%oy;O_853J|po'
    '}e3iULBbIm!c8EoKT6Vg(WgIGA&4(-kKmTGej5lMs;Vx846F+iqe$(OekR*j08s&VKGU-KvnI;&kpd_?hy2$xY}iaUKjaj}J4X%x'
    'W7>wspaq;*UN7=`YQWVTA8g1S|3&(B+Ipc@u%45jP7JvM5BegE%X}+v%ExK*Lqk%TecAC3cL;S=5;r<HhFGLd`}X_SXvVLIv8%3$'
    'P{0_v|ChHQK!7x6RSWSZ^LsOfLqn$9U>Zg#aRbVH16irrsPvBR$Zm{sB$BIgAE+3;_4~M>O^3a2n?^2vpo`9y&W7F}Ve4jn1gHf^'
    '2E_Fnayeu4%Y6EIMQaR_Vaxil2I1%5W;^HU!m-zEMQyj_j;XO5+q5JujsN#Z*t1jF?i<vFIve8q2O>E-'
    'Qr0om8P4lc85eDx;%F)p9IO>@W{`#ST#nr)e>W;e0GQsv!w$$*p{ZYAH8XXwx7$SF=)sKgAx@Y7HJ0bHKst3CW{<Cn3%`_ULx?t1'
    'G2BA~U;YsoW|U=A@>x8{YS>R&w>ykprRR7Jg~xE6>*(iI1iu>UUCZh%4Ks3d+v+q+sqPaq5r5ZeR1T-'
    '=y;tvd^5sj?t5x{zZ0@<(dgXJGe@azr{s@d}IC|xh-'
    '$!?~~WEi7N*X#(zjY%Vw@2Kp6jV5jX^Q(51$^Fc&Luove)Zmw^RnQGr)F_DtjID%K*}aXwRp^lFFKs|7mHN=@sRo#&P@$H?c{>_C'
    'XP@%ebq_-VZ;E*oI>m+A&K9EaB-l9^i8YSSk?qYuoasHz6e-'
    'Q=jK2L7G*C*AU69GZAAlMZ70wybHz9bGHF#}OxD#dnDUQ|HCEILR(Puh6>vkYTutok!!)D40!?T1%{dGLV76@Z7)7nh0E9BI})kW'
    'kDk3^)rIc;vzr1hDfd<t<t8aj6AEVa)1ST!t*8Q=cxK!LXAefCe4-=q+M@gzD*^aZ-'
    'Y8GO1&N?&4W&JU8=zHd*Lieg!7U@<uRLyMv2=MIYp_v#s&k_TSLFa6$4P!Q<epFJ>?ou*;DBewLRrtP~B5K5A{8T+NkhjfwWr^zA'
    '$5V+khyic|lYGtUJ__`7zpTY=$a-bWvB;)w=l$&5kZz1BP-sITK5hc(D>sPp;~~smKG<Y<tlt(qgW^l>n~jS-Np+al(d38WE!a%I'
    '0+HN&yvf2D`x>j3@`EcO!EsJ%{6b&7qR>VemR!axrd=wuMW-'
    '>E4Sx9~;b5=y{YK6!i)Zf?q8iekZnxGm($WjLIT;EiyZ~z14<bH!rVc$==#5!80Z5*X}qX;cM(jLew}puwK8-'
    'R8VdT_ZjTFZ(;VQzREmCuos(LdfX?s-2J^v@fVLp^KrH{qZBhAZ>L-'
    '(vrfQ)gNST{H?wmU4GVXh3ulTmmxpx@N}&~U7Nc&;5h<QTs9`TBoUsZVTrZr~g5j%#u9ZsMRT<IpMLoa6Y_hpb<%w3yJr-'
    'w5!4jQGNC2=uXLH!}7Bg_Uspe<0%-)Y;RO{MN(%~38iO44%tUdL4hWu^JPF$#n?9OPgU_9M^g(M)-'
    'Wzr1J%;zc+%F_sXv_iQE6fy~Ad9q_%T5k+z6MrAlz5SJWf>L!&$cB*Ai-'
    'w7$UNl@J^`a3XsTZ9fl6uidk<^P$6iL14B$3pMP8PMzM`f~<%nXK5ka)D#R>u_!=g8emr=(8Okrv8O19JtP4?FK(4Q~{kxfavKZL'
    '13uM0s9<D-sDlr!aE)0i*FAqG@?JxanE#d~s~X18?Ola-'
    'b>%3wReCcaI8q4;*(73V0tJcaI79033G@3HT5kcaI482po402>2Kr_w#G&*MTEVCLN!s+~#@&*eDzc39d|MKzNJ}2@`1(0lb-'
    'hk9bilio?;>D^k*zlY+&{L_%*%^PDCLZIKY$P^2*tfk8~aKS-IUxuQTbz43B6v-'
    'TnR1W9HM@$q4<3;gcO1tqw4#u@`<wLmY=5R}#epXtgNH;vW7Do?=qaw3x!<p!t(bx^TlBk00hRm4MrSsIqr1b|N>s+I!4oQ&S8L|'
    'p9({l5@RwR5iR0)f~_n!n|;HGW#L&(<6%<BZHPp~-TZAVyXfHjhv|IN}+cUT+3V${^rG3qkor@YpXClthGZ=86<`j!A~9E}-'
    'KYF<Q`6;;(|J0!cnB;>M?8Ov4?gXHas*+`~>Z3=^Lrk11T8gw8HUb(D;AR7cA=X?JXp$mJqou!TEIC~VDiL8P@VwQ#bcaT-'
    '|qg4Yo~3}3-Kl1LLnCv>f2h;&)jT`<}H|AIGn-'
    '0sm7CiDWz5xlRm4dVA0Ie|f8QFJ(7N)uu836z;%lSDjoZL%n$+he37AzC&likmxQR`m*L)riiAp2sDYaryJ4|L-'
    '_2wMx)|gM$t{B<R3HgAP0_=)fUC2Ob`D;1Ny-x{Sjp&3Vr)n5Dl|)y$imS=-`-'
    '6W?4|xGxS1^+H;k>@SzuJf82MO6ixu5Wszp?qC9?Uk-'
    '(M=Ld8LlPLW%1RVVz(H%^o^ve<8@_PWutJuO!sjxS&yN*&}mrHX<W(m21w(pEW)h5*TjOMmOUt39;cE)t%m9T?Sv<m{{x)jw`H%#'
    '~MP2SZgI(zF9t+h#-'
    '=QTB1eJ0F)9`D;^zLJHBvnJ2Nm|64HjOWbS4k{j_z9e5lQ`8Zp@GCSwK8QiuNb<S!x3H!&P7wj_DH`lFvEH7lZOS_#Sjh0^_;~#5'
    '=8p<xC>(}qTji-nQ^h<~90lTVH@B54Sn>7Zx-W@rI`OzgQ-P!OB5L(^v5npjcidX(qgBxY#kbdvTz%NC(jsH_kM)-'
    'Dt&QOe_<Ecn3;24HVG8(qT1JMgnkAUL@mYe&n~)`#yop(Y$(xiVn7qkZg2|hbC78UaSqvOOd%n=A`<>lvM=QholjC@-'
    'z_IGnHBBeQ5$ztJY%-'
    'c!?E8O&=NPxxV2OZg@9I1z&08v(HBaUP+xWb)vW;H;*jwj(?~JObmn+jP^DcP42}_3;sRSPvW*Wyci!5S1st&ivxMi&pWuo2Kg4('
    '*-HhB4Q-k{rdJGen}R%8z(r}P<x=_G#^-;(ATO&u&p37oopxp$Y37dTD&N-'
    'x?|Gn)JCyQ@UP1rPoe5xTqnNbi)`ip6?%WnLDDs`(ygP0Na@r{Kiwg*|1lXJ%Wu8N5d*ZLC(5ROE@3k=}|1=(a{VOdMLKd`&%^5o'
    'DIuIDB?%+@{jS4|}vY%qvKA4<@2}N17LOw51?mm`QOen%KP^0e~xg5+neN&R`w7Q_UI7Lj#qZ!9Mh$iZd99o>XuK3(<?}&0r#WQ@'
    'I&zgs1eJF3ph4+@uQ$`k`SqMKstZV&6<Ai#p=N(PYb7i)~N!8YNYnd4Gg7k2pznM!1oO5=lL)Fmdjoj;CcjC)^eThbiW=UXU;F4y'
    '!eKzh=7tM(<HcX^c8aDvc)dtO(O~*4EgxMoxs~16{r=ny9Vu6=k(cq2h`SQB=>Uh``%#6GnVcZI-'
    ';)=z$51X5l83MMe{(cLOK57(DWD0gnW}RVTeSqE{&;+2r`0Xxq2iDKk8(uM>86MT8j{(%x0bG@p|3WOv)mOKvb$4bao$+|zF*we`'
    'Iyup0a#&4rWL;I_bT=<OwzrIyLCwOYe8jKRc_MA=9?7~?P^MzZ<GWV8crDlo;ZG7=Wt!H=D=z!SZmg!M?VFuRY99zA|k3M4%B1)5'
    'P=M2z@~K+N9vxj_FLnr4rMz~Aw9gs`Wtg0DQ%Zcnf^GE9S79!5Z|40E4Whmjvi1{J3O#dZEUf+}CAp)lE8)l^XEjD3mD80x|a%tB'
    '9hT}?2VuQh8OsALEht-Q$A7e%(Gb<8}Sj1ou->wU}Z1P&$k7IS-nQ;EG3eJD<KT1n6c9C)-gAOia-_{w0L{tCYGgurfKwq=-'
    '$<=fzwN=!a`Eb}_p)>%}=c9CqSw8>5f@{F=@;3y$zSJYNpVt=e7l>dX!;4Mej#*v3ebHAb?5DdBh-$a3FfrJ6)5IHn9_I-'
    'ocp|P{CiKw3uPvzxznKHX)^kdVM*)t<d$O{6iU@y=lK#8cgMKX*OTjh_jWlScFFr0@{Mnxd-at35BM;uqLGah!;gb{Oxin6G)GA?'
    'af2Quk(?cm}p4fKd+FJ&mTVgLZAYdCYHoJ>;tRU)+)6;|!N);>m<2ZfpD70ngO?2)-'
    '`*N9NK5~Ye?7I}9{i@zfB1_ETgD)Jtb8huUVJt;H#y2yJ`X88?~_a-pbn<DSyuvks##`Lp3Su|Z+qYJ!2aEnk~QtH@tzoeUN?e*^'
    '`JiaA4ml6bxEX-|@jm%e=74}@_tIR4A)vs%~5y4Rk$m~uFsUprlcAAnx2L$&U1T`6NP#)M8bbiU^tBI*WMDV^LIF@a7Kor2U@a;-'
    '75VS_&<#Pg){LlG~dZz@R+<c@Mg#jYiz=eT7XnJg*0RF7$`M^g1KTQt^%*kIgKA|M~7PxWoUSi+%0%cB7l&bzJ>9XG>?*1g~8szD'
    'mzn5kswUp+E;7qy|USA!C<JwkfPSp%j@YR>YkPV;9grLa&sA<Q+PyJQX7HSE@Mw`G0V%g36%*!O!&So=Ubx;=L<VIRXe;MpFs4MT'
    'B`Z$cx2chgh3H32Lf}(V63nu}a1e~(SaqR~0ew|ZhuoG`2vHdm?+gLkBY^xv|dxl>lUp@93hd$1FO=xkq-dCf?9{l_RG+NTo4{)G'
    'H{~`1P9Hh~2^O(gWaM*xPlG*CfpR_yCO<*Fhc<IgaVu=;YAEs$V(D@tufZR5`Wj~&1q@8~f)3;d=sYju3HNZBPf1+LG&mYRwX0<b'
    'xyGl!ilJ3bu{`%1`W2JY^L!-'
    'IVhv#~>1o8|U!Yf4ec2x{Zg%1S(HeVGup7zt|sDvcs{u<rjkc2!~qq}l2E6ys1OrSZ(v+Dw?a{?P>(47<c6oc}d#0DC)=VUfSpgy'
    'Oty9fGnsy`)7$n7!2vOR1gxu7k}2%p<=%6!FcK#ap^|NSo6e@6;)ADB?hwND6`Pp^dr#Fg+ECJ>+jO?iK_n0qS+m?aF{yhWPvnJc'
    '}*>NaU6X54x)Pnt=Y`K~uaizJrHe~_eN5vUP@#t;QmqbQ)FIkI#QrmIedgCkeP{F)lTFk8f}=%fQYUv8E=^Y8)aPA{^yL}bSa^RV'
    'C+ucxKiqKNLHhd^}qH~$%801ZeBkLBuT77D5nf@@vPJiE8KRbu+)BhPRo&E$+B<abCj#l@GFz5ryF*&c&!G5~f22EeMdK%==v+rj'
    'YPI?K;8)A9&d!aO^7p(SctvplU<EboJ|C~)~>4}F5zd@PBg@FR3EX=qh=z8!lin<Kz^z2zJZi95@;G@1}(b7c_^_JlN>Xl_CT(TT'
    'AuO_&TBtjY14F3b>32%j0k9Ih!v-PbJO0dh!FNxAKTW-0lQ`5?0_R>yZ>MuF7n$<o}Z0{lKeoa%L<59U&3PII>OD#V-n9rpVq-'
    '4EjIg0bdMO(doi*V^Qijz(|TNlMk2G?pgpVH`K#BS%SOk4y8mB51)vfiD^`dptIA^7s`++olP5fx5GUt-'
    'uNN{&f!pP9PCLC~yM3+3Kah3G_1Vt-uNNE_fdWPT=|6g4ECInW$NzEgs1@7<((Mly*THQ!k~*iy*^MIszHSm8^28WMQ96i6{28E0'
    'ABX|JK_MPTDI91BgJwb05|lA`|;Cwpc_hb-QcaH{ttxY1DfM@>~x)`HXor>^wOhCTHi#5osyhNGll&d|eePVYgz?=8!5)NNUh0a|'
    'Pw7m}%qBG?eOw?!u3mhv7upXip?TK@WvvZ+}1!4kz7)6xsub!6P`tP>Ec1qhk%*5^LCQ8Z}ukm3wGBMZr|=t??uU$?c=@H2GA)lX'
    'f7)kEjfMzRo}P`%RjLBv=^f$b~QulhJ$#cEM*_PNNFV7j6%v1Tn2FZjM?iDA**UbPZs$s{M>?3;FaSd@>PXgTgVwVaq&%r0w>`e1'
    'NafEFMq*$u!yA3}b@(1x#>bzy$XXnBV~c6Fe|rf(KDis;4wPZA6>IiniKNoYM=cv_0zRoP#~p?6J+4!N9hwK)BoD06qB6_gbwIbv'
    'rl`Mr#}11E<{!>8S#^r^?SI%+_>8>5d4+8b<83Ajs7h3ek6LiD%h2;jo`81>OB)c8m)VINKR6-'
    'g|w@P8PYAP8q@meFbVI^3Bjh<yAO7=Xt+ajcsTzC>loob7@}kA7d~M_2N?ehvr<Jf5^}IPRzg+ryEr9oZ#rdI7$0$X_RwzufZl7F'
    'R-'
    'T*3(yj>ER+kKNp)sq(1?bWmd#c?51onx*TO^+^CdmpWT%6mx#5r+1C}EJqMtVnA@W_)JgaEz{5{x>7*(CxMN)NoMkH0IXGKzVdQK'
    '!&r{_gdb$UT0Ri_t4QgwPsL<eaurkR}&=Y5ioAcl2AFr)Ur_T+T-R5;4p*FBFG<^gcfG<O3wJG0>4|9<41Cq3JYn~$fw+U8JV-'
    'db*ctf?Zk!d$1RkovftazkX{c5;yay@%ptJ;o(gY=&)g?5|bX)3KA!Iw-BtX%!le)NaS;K9deoDITm+rok$FDlHkR-DTfJjKS=}7'
    'D%TrJ(HZ)ii2D6NK5T;T9!?{teL*WQIkYXr-6UTW0xwaMp|b_)78ulb-'
    'k3AsR*BFrYkzMprJ<7PP|D|Y42$}BWN~Jc)@8Vx8_O#XO`f8FAIEqD_GNcm_w#~YcFYfW#p^C)@bA`*hJfGHZh)va1|a2E70#?5!'
    'r9IXA{x^JH$Q8^9{pNQ$3H>-hl1j{-'
    '7d3K^H3gBIiqtX)obAb$6Ln(rK^jHDT{pn;SJD9Uqs>kd2=2PZ;TVxqu^0IN1#tzMGknVo-ctEeXPG`eyXM4i{!%Mj-'
    'CVwj+o@s#ER2^QoeS8E{l>U(MUffrBim)^&7HpGz8(CRnni5L7OW0;gp)9Gf<wJB~?FE?4FPMNmGA--Ul7)qaDf7-'
    '6lsS(AtRpfLS82RzJ<rYDH|i4rMgH>-r-'
    'LWj~^6IP61SB{A}$tXRD)uuo_Qx4|TB0<K~SZD}BJEfT!A&DJ1v9+h~n!Y5h=M%M7Lef>rru0!}H<~zLgZ35?Vxk36(7q^&j}}L;'
    '_9cQU?@YiBW|V2y(e4e#){qE(Pe-'
    'y76Jwoez&=*Qy*^_jo^3cDmM~&g=8{LvwzgZJ(S#@X@lMGsQq8rMb;KSQi3KdPv#WU`i?Ns23#mp6AdM0h*%%Yo>Z!&?+GiSKU<7'
    '?c_kn4;u{8}N!<4CKcD$=_+Af9iPe`VxcWh3+sMm-'
    'Ejzc&Zik9I}F6ChUb4bVpA8xUnYcw$vf{#LP`KG|vHNF_}usx;caTmrtt_y>EE#bCyvhwNbN}f^7)r<||D6SXNk-kWsGAm;L4y--'
    'sVAnvtaS9qiDH`p(HmlcCngay)SlZB7^&f&mL%@M%eFLvu73Xi^wVOZ4A6Volp=-'
    'Q5irtxgeX2V@1l;IXh^?a0*C+DBz>R*5cq<0@<ad5JxY2J=@9>~xM9|gf3R*qCS-'
    'D&!3h1R`;clwJIK>ArBb7OT2@tXQPLt>So<+ZUh^7T;dhgNV4m}Fi;$VIBV_~q=yp6>RnlJ^i=pQtXLwiV?|I<803gXhgXgftJR8'
    '<@$4>2$!_AG7^u|qoPq1NfpcvlMkIUa%*e+bf0sf)5fXYAE}uFNkTzGaIF2+$1)8ABPdf8LQA>4rf?j3<KFrksx($k0j2*6GmWR&'
    'wd%l!qG8u6E2u*kIt*{ZaE=E(mG=q<JpagA&QdX)YH8ZhzH0L9G;(!MaCfu!g7%)*~u|^^D43y`nN$@2Cvc$Ctr)tG%OaZUe4tQ>'
    'T~N%f+oWNttsrp@Y8==3ve@4p(Mhrfoh!nf;lP^J<MlOuc=B;Cw$xe(IChWJj=qVK~;DT!DNsGI{Q;iKL^qHdFjr93ini_&7iF5W'
    'I;dm4@XNob}}OB|Wb)_sd)LNxlvkjplT?ddc?GQE$CXZ*>+Y6~g-!AyuC$^B-=@eWA?%5GeGOqWGpLbDk#r(-'
    'h3xtzmk`J8|ZK`@Oqs%=LKBXD^MjT)bDcx5lhba6B(%Yr6=U#Xr}?WYR}F;v$O<8N;RlJh5-'
    '^QvyYPj&h+oSEUg3!c{4(nRJrZ@QEg9Xp_S%%!<3_xF%Jym6kLBcu!g$NjKbml}K~89hRjzE~>m*6x#%4nl!~j*WkQ(!XJ#sl>Tz'
    '&$h;L9v$>J#F%4>pyJnuyaKvKeK0|SsenFVCgn;`fbh?O1?ViF$Mygtm5cs+(p>MD}7GSo!WU--'
    '+#xD%z&SOh13bNkdKhOqi@k_KPTSP&CcGLt#P%jpe-'
    'l(mT5bCv}{3a_*5cpi+R)MYp^*05Ii@Yz>=wQ(#uw8%)`+E(li@kDqQH11-=H-'
    'c*yfQTP_mJjQ9ZwpO<~4?xv`HV>215bLyb~uC2iuUL<?vgpE`t{eSP|cCps<XGTQuUEj8WWMxDE2o_od_SmyQ|0pV!xmf}5z!H#O'
    'xQU7G++M0AZ4lu~Vzfm{Xm5VNTG)a%vWK{_%X@YO1#q0Fl!&z@*P;^d6^>}Lpb9y4ZLg(+CurxLKE6+$00xv3trLgkqK-'
    'I+RWr~u)#Mxf$trDj?XSyCi}`dHeT5LG)3au8Y8HxkR*j!z|OdaeeiRujQqlkKG;$E1RJsSbY{I}{-dDIObO*@YNO<r<kSrn|2;-'
    '_mi<08K7jh}_#vD$<PKR9quFf+^9Dj)$r(mqu{)9tYJ3I_%ReFWGb-'
    '?%g1~QsC1G&VHMd3MlkbNh)uqLqgGYW%;j{ZWP=wT?@QHkF67aqPCF(WBAsnisIW-RFda{<44IV2jU?P49x8k-$3rzn;3-'
    'WTA&>glFFkH;SSWk9!ecEeiWj!&gUqe+9tJa3immWkG1ce3cINj^S?!e)xOI7*cr{gEAtC7njOh*>aeWh9g@2iN>MaW%4oE@H~TW'
    '?j0%3Hh2@|ruL<HxY3Dy}xxSX@in;~*X$Yx^Uah+Z3(Ck6W==w1W(c##cg?tI1*_Q9s)BeL>QzD97d5Mxy)x=nu`{d3qx~nkcjQ)'
    '{5odqCjoD%=w!bO!UnI7-'
    '3jA4JRA|zvpa))xC%M7KsONzMCGHI?BSwjfcGC6igeY@_!fuVG0Pj{w#ud?QaZ=SxSCmz?9jODf`A+1ZnDZ4Q>zR!T&FtaKJ+^zs'
    '9X|*^qBb5Z5BffyCLEFBrr9UyJ!H<V$^M+9T-SX!G&D@vys?J$qSd!OT4#vkfQk-'
    'h(eAHNG<MB}niYrc^38ak=tzrs9G*(@)d+TrQ__gBinGpkaAig_-'
    '#aj5p(a(=(MQ8;UAGZc@LI4DYmACK=kvl~$L+Oud#vdmq0L5E-'
    '7f3k*TEl8!g!*FaG)4xRaFhc!+FZ~cRGTMM$&t4C)I;7I>r>*98HCjjulNaTi09ep7lJ5x>@`53vOOp@JsTKW{?IykP;k$rwoaQP'
    'QdoOxU_jq$aW9Y-p<(Bt`jtuoDY;{GWywG&BQY~kBNtE9v9|o+V04^s5`RB+$_8|UVxT1E4b}-'
    'aVZZr=c_JF1b6m2IUk#f&IiuO&C)S$heT~Ob}(#4!NexhUqhr7w|MTW<Az2*+c;l>vzg8f4%^8ZqGWdm$HtRH=Ljx4G~sC3i}^b@'
    '3GWDB9O}FI4)%W95IZ?LE|KauhA$Pm=G&zA4qUx0u#?91^Hy?PB42L3^2wrilFhar;cJo4qxgL&4(yEQH&i0ZQ$+l$FlD-xpq@eb'
    'oeZ7>#T_nZYc^5N1Jhn+b4cwE6EteUtu6Ulq!>@OlUWbuhMJ=U$1SkkjzOY0p4wU$Y-'
    '@LLQMCDsrfeR3OrIC{x#tz&M*=tF8u1QL<_4Vx409DGhOo7fBCgGe7~7Qfvq=_}P+4)^ZL9+;==g*(P&`5e3C5%gohkouALjGTTV'
    'cNVL=9iv>L6*RXbW2L)5fQ2_)6o@0zJiD5fjHf+$e7mI`gSt#8zSZGSUXT@kUKX7u>%={17dY-'
    'R!1|@JVnhirp*+DRYx`Rlv1;mQ3!Jq~J1Ep>p$ex_XUTc&kq@4YR{WJ^^7xO+kTCY)%Y5vZDv!z2=B`^Bs#ro#_A`hgANggzVrPn'
    'gix0UeF{kOi@#(e!h~&QKQ>?G`5bCE4VUmq`4=7QRk=3b7JG*{+fIhaC8RfXpT@fJmfptz4kWdkkh@kRpH5YX>ycVXHq;k2FH+6Z'
    'PE@@@Kew1_`^deiJh+wQ?vW#%k#eExUS0kQ}@W~d;qzyYY4dpf$DH=-'
    'cmfjdy<B)j4xn*mr0g#Z;~kB<~tBZ>mE`AZm(g&PD62_8&k~SKhMD)erBGI=x>oW13Pl!p_g-'
    '|ZH<MiTcia=rolYELmbV&oA|@9F~?BbGLmwTpJ|W4IvuFaqkN`?*M(CHo2yO~5o{LALe7kv!Ndiw6X3D(2|7tKRt-taz_&d;HZ!-'
    'YI3Bx0g$A-'
    'RMIkHleiY2D%?A<PE^(yTM9V<!27QrzAr9=#Scpi<2J^`hyjL?VS(~GM?B0ZsvePtyH0Bdark%Kg8BgNPa)$GkguC*#aA<FrG*CL'
    '}@WFj%eOR};5yz*ROmlS9O@pV!!)*7^<^&z*jVxVJM4z0Ha)`LgxWm+Eo_KtuBfbwmY4awvSfUa-Glgs=L^03skW<c&zhCG#gP7d'
    '=W0v4X^C-'
    'zOrmO0`YFlrqP`WDGcpWi0T)ndVFOee4*MerJ=WrKhU!rj0Hkr=zzC%Xyx=*syW1DN6uD03eht6J`jCs1&IY2zFuETk_5VHpAa^('
    'Pk{(#=5z+zbWVDw30!Uu_sY>xd}lyMzRuCM8x><9qZj;J7CXY+9Nv2a#_f+C>RYH&yiX|)E7W7Ji9r}Trit;cKIwBN4nxQm+6Li1'
    '2)wz8hS4O%%Z2#RHEp!yhRo1OU#!c+d8(^@MX!G;yehlvTS!+jF^Z)1Y*F#g+?%sOg0m9Qpg=|<~BA#S=w(%kn1RhoB)zcfk|{?a'
    '5-_)CLC;V;b*g}*dL6#mi_QTXfO8GjSzo0>U&drM5UpA^xxTdZnCtNVf_%77#JD9PfZ*tF%#A&Mtd-'
    '3kab2v)Ze+#Z{9Vkl8ERQ*<&4ppBgX{J50J6l#y@}LeAF7{#I$pi5DC6{K-*YK51_J^A4EH7AJ{>)edZNO-'
    '2Tj&`JUX3X2DsjtGm7JYUN%Ro@0NtTLgVPY%-;jJhE^27-'
    'Wsi(T_ZY3sF<e~I*c1j|eP=Ri$yt<0c%4ut5DC8NaTHiWxtXg`KLuOQ-'
    '7EJXXp&6U$C$;WNe)*E^9YE^3+;1Z&eN8K<l>m_WN11fmDrYTPRU}`y+-'
    '~g{C`84|1Z|e;9rc&Y+oJwnYSv_PgDC`H*z#TshMQ2%~4zsnJPddkWRfD+-'
    'mOI{5km1)T)V7mVYBTWa*UlvAM;&5{WnbB+B=&UPGfOKa$wcKZVOCHao+O`D2+J;YAx0o)t^3Ss{19Os=sG70weR;u!R;N`<a$3<'
    'QTpynJ14i6BOJ{u;}sPC(l1MFRRmA^9~diUPZ4F*kHMZNGsyCz^wY8+-'
    '%1>~lmh1cZS9Rt9wbOq$8W(fMy<*!@r<J;h{pj6Tv);BCP{16)|r0uihi#vlJb-'
    'o67&%i>&{DWAcPC1$NGNLfH}>DW=RByT1miHYmlu&)Y8tRyjsf6h5!?WCJ}NM8|fvEYhr)u<#<L!wa<6SFp41<T3FIl=zV{meYi%'
    ')E;^x&G@qfB3Gw`~ocd%{%kVQ|?kEjGc;-h)6_g__NM&7_NU*0tR4ui>SLbW#YF`O&G|SMFwT+=(%=$zahd!-'
    'B&r);C{L<aCL)M&KDo1!EN`0eG%5+kGmd|8MZ#%pdtD)N4iziZWYH28nQ?igWCGx-RsI7YsCQJ3F^umlZEc(?(uNB-'
    '<;`%{3oqh`hc?<_<+(s{2v8i;=W3t^wtLV8gTp@*pU&CIHMt+^=W4zfoNnMC)?nT(}6yfY)Iv&+-'
    '8j|6Y#sVDbrnnP5I7w%ji($cF9{thbgyfCcQ$WfJ}P@>WI5%;wwZG$kbPeC{W6;JOH>yFv4!Q`*=D3Rsibyu<2&8Na2$L7wo`&KZ'
    '{)quM`5!Tz8eYLEof=5Zvx=R>GEUceg6Eflm_HT(6E}0#X%fw<3S`7Ch;rpnC8Xp?@i7l_%i-'
    'bb5}hlJ^d3a9S6zitrd}WMoW6t{`&p1X32~1;}8~URybUOGr_w7<>O$J<7k&x`V}lPJtTdzeXJt$l3U>JPG@Q;DDdQg!Sm;?>4T%'
    'o{?9G_uvYZQP6KxqD<57R;rBVdpK$SJRcYJZRJj|B*R`P@^@HnmpAKyxQiJoA;s~hQ=!o9Pem4ap=gmwBURVCM?+v+>K?NLH3ahl'
    '`M-;n!dzd2im>85UxEtg{u%d2%Sj>}?)GjlY~XT{pt?n27855i&-'
    'v^sC4uKD7^uCWWhQv1Ukxpz?0(C~*O!7Xa_WZ+ZG+88s{=LSv1)Q#bq2X%##y(34JtNQbj@VK5k6-'
    'hT=Vep|5tMZMvI+!S}_zzk<f7l3OSzAzt=Zh!@2`AKSZ~gbq6tAB_#J@HXnLt7hC@|e9CitnJj?wIm&0P9KPO6p6wu9kI*bQ{AuN'
    'G^I=*Grtl7bMztU^Ua}b?W4$eym?%^_>xe9|HbBx4`9ajiPQpryP2PVFa_mF&04AnizeU!~&f9<28+S!G$SX3%q>c6U8w42~{xLK'
    '}lKfW&s$3SuquRkzPNw#6<v#Xr=RWrDBp>^6)cLU0IvEPy<rCGoc3&j_zKfJ}IA4_t`x@nbZgSDjcKDIDNsZ<|=lo18rWqhq&bS3'
    'g5LPO68p0TBKu*wM2MWSLEy?dh*0#BYQ<T?h*X<B$bJxePpq2Zp{N#85ce9A&r)55tV<QOc_2jVL?z4G#%PnL<gOQqLt@nf0T<kH'
    '^d}Fb|qApkNN8%@arE)(J1m#uA{nT(kxP`>SFU;WB!~Kdag6(@~5~3$#gALd?vKsIyQEWE?wcRaHY&y5mD{PjHUSV;`o&heGHrqI'
    'p#%gbr?)8m?Nu-Ror$9RQWD%vgXejllT&=XV!L5qa32X`-'
    '#<=K1Eegpqtyn58678I`l&{RAOjo%5T3KXaxMscaBS|Ulg}7CHGPAo0D2;Gp_HDRtTh@lYiA1x{k4>kQB%00B%3DcE^J0^afvWBd'
    '(X-sRtHVYc(r{04h<mf3ORrMCY!Vk`IG*m+kh6BRXzYXTc68G2IVLNTd<G^fk$Q-'
    'PA>!>r#w?t%kS`0<I>>f}GnU6`diSaPHodfwxM{5!OAt~k^Aq_7YM!}JYF8@CyMxYvY^D1F%?<99Doh&7R&!?>>SDFtDpspUq1!w'
    '}%f#ydD?Zb3mu-'
    '<e2=L%s{|}JhDB>r;&MuR*w&KtR>5+9mOS@sehWn0`wS+_EB!yfr7(@CP&Bc$dPpB7Tpyf07l(3KUp;olKF_DvI`6BeX1sT%;7Q-'
    'TDtp&Bz*cA7lY-4W6=#!k@aFhCf>jNCm-YypIZsqef^4d<^-'
    'CPe5*xy_M5t!+DJjsnC?CVXE#rkAci%09Q_CN#WOl_=BcpE;GlTQ2z4Iiuo9dJ^^pC4%x`3v=mnmF%t&F&u4b!HQZ*wK|M3}$3+;'
    '0#nY?t@UJuCx1^ai4C??}K0H;5kd?;@k21y;H2;t`@DhQ)2|$-6HQi-AbEsBYf<o*7LD{n5p?R2D!11CAqPWC&{-@WL*w-'
    'MBLn$fQlutW<K?fJC$U54{Lu9)Vnu}1`=AKQe1K+yQ1K82$5`}((V^ac;GD0OQQh8nz^#!y0yE<uv5!yz75_j7VIwN^E7sPhqTsy'
    'La^GA<l>JOOR9s|38Q)$;P97O{CbpPg@?cnUv6>;#SJEFp80=HI_DEvq67JyPkb*^(pd9}$TjteT3ec^VcV^hY9KR%3}z0J<#x!u'
    'iJ03KhP?Tc=!Q(tMniO3*=9?OyFA>Xy8zuAS8dRkqJ^KQgSp9PE337%6A~e1k4gRs^YxAaIS^?zMM@ACzb^u_|01i84_DPoEIvL$'
    'RU4RRK3WBy`8X23UdytF6PQr4q?*WtlIyEUOek4eO=d#LvT6zwN|sksnNV^=HLXDT?+SD?oxZik{fRG0H74Iow=1{5bvtNDhGvTY'
    '7<hmCG$hNO&c067O6;o>dGH@F^FaxGPgSkYfv`*Y8gM@|fy&EWE*4)$E#RxBE0yHBkKl5y!Gz}<?!x*<-'
    '<RjGK)!XSy%S4$|3|tQY|@oE?i5s#fg*4Z>R=|mJ??@hdBJpTA7zHqq{=?uupG>ETPIJ)v47O;2JV$B<Q^ryKq(vBfxI7L&^konb'
    ')C>*7aJjwk8wXVLYf}yvsocck8?jZLNDLTr?x^b-`l6QLNDLPR|-'
    'NeZziwhS?qWMyP%_`3ij36<$1n3TDf<_rt&|}PBpBF52Vu@MUr90<RhaQR#uDnX)`oG*AA8Am)nTTxFE%gL_5#To>NB+BY81L@^#'
    'RW;=}c(LRGq_$?4WnpE$I2544={zn=_B;{E#pGAKDFIt{?4<yao?>?iITff%qPu46#2j!Mv;-'
    'HaR6sNW<V117X?nQ>Qz__Z522yx%p;-DU&-B$cC+kND9hlq=vuV)SwN3yGZK-'
    'Y|%{DDSEe<Dv+rt=b2H({SeshOK`XGHm=Tk)ae${nuoS{6Utz6uCl7HvNmWT0e`n{b$I1~;EA+o)M2T@B*`X$<A=EjhWa3umm0)3'
    's}DWc9`6x);<cS}tF2oWxm0qgbx}wcAn-'
    '0ssNq@_ujk`H(wIl+=C6cev&qa2s%R{%d%*3J_@jHBz|12Jl}aX@{J9KE~7A1La#ht=$C6L?28R=NZm`_fqcohPzqS4OBbWBRPY`'
    '0T{%6`9O>pQ)-d(!`y=NH_0F<oeAOu%I(05v)Pzh%IYh<yH&0o5`?xzJ>e!@s3+X43-'
    'yFsbfKQGQs(wFXmTI1S%W_UOJaxLJry6qbq3}zaR>Vg{7@;W{AoB`WwMsklgM$Utr#_RCj-'
    'k_X)CqKb+rcFTvlpn4Q3MTSE+r*$@k*JJn?_um=X~&UqWIheEhgoLVn_+zuT{KqQ#JWvO1Hrb!t80d%&d(C7N;fQjeHZhs-BqOf`'
    '&APh+d$44}cVdmPuSBN7&BWWqv?idZOL$Brhj4J&7K>q0qWg)WpcZp>!T)w!_5MAILOGI&f4EiEL5u2S`=C7*?6Ce1T!UQ=|lbVR'
    '!H_S0?)xxNrVnk?aGV->oa^<QP5-l17NK45wF72Mx<G-jmjG7mmLIEZ^{m@CCQb)<&--6y5{-;~(*Eeb-b-SH_eH!vJJF?)1JXm_'
    'GGL7ZG`?|0O6EjAxhR|?#;tw*cO9Ye|0ndy*qioqLDL$1fLCyGco+T3FDWSjZ5xKnKI33#f_4a!clxfk~7HuuOo!{+8?XWHawLco'
    'X}c$|a3^M1>7o23TkO|gk~gMm5INm)$tN9O*JGxQjD6UU@2#=MvRWH_&a+nvp<kFY>8FZSnhh~-'
    'eY*ofC>I#UpxXJk2K#b4CPRANJBD)F)wAk20=TfzEm&m#mZ*p57D!3ynUIjZ{w`T(O6FtXdj_+(cEj*npICM=3vl_LX(EB8jI_0U'
    'A(RoSx0Hv0#<1MtTu04^7mfQmIms2C7h89UAwk~37@d9Qg>QarFZcu6O5@b#JH!7G{N!K*rXO$ZzQti*;tJF(%<No@Ezi48wDvEk'
    '<>HvG9o@!~V^jj{D&eRUaEXRG2Kn-'
    '2AJPKqIiyu0;Z1*N@*jm0ZrBMufMFj}sO8L@_T(JaASpCz69B`{@Ywe*#~?1}dpOnlr&W0SOtAg?j=yxzdm?6|_}0T1I+QDRw8gI'
    'cG`7r)!9xrvs|<m*?w-_ySg+2TGX(ekz~R^}k>w&9A4Adu57|BP!yp0pi&o-'
    'rIje+S>l7(gF|Z)8l69)oXWJlem9Z)EImJ`Uf=m}fo#-^i0hBzK0Oce+VpF0bZ#q1#_KaHzS*aqyd!-{#z*A@)8s7-'
    'g@V4{K~%5p-brByCfy$bs5z&GiofXZE+OmwAIP$&ZXH|Gainti-?1L6Tb>Y(Eb#3emsZ-'
    '{*sDJ4$N6hsYRksdh6N+K0b#IK#1E3gtc&chSv0^ZszLHmzC=B7%X%8YMHC%*dE#UOK--nzG;Qgm`x??by>7L56`w57ji8nrkYz+'
    'm5LXx^~y4p6^hd)Mu-w!OTp(VY)7Oiai95Dq^$N!4tSJhDCVQJgvsn!UFLWUWF)+@Tvt|dp2u3<6-'
    ')G_NTVg8`+$8!6DimYja{sU(x4m!jvivr}3sQ!%sKo$xKUT=|ci$gw)3}4H=UUqPtAyGvIH)WNxvqcALjyJuDI3(mVxu)MTEUPnc'
    'x0=_BxZpQ(x-_tjF=U2NPnI#J?rXR&NNT~k^lsgJaQp_HIf{YK0WQEyz@$eoG(JGnK;>bOPJz!vD-'
    '>X_%!vhqV%uJB1nL6kj#1sdNhaRRm*Q%(bWXc{;GeAAlREaEQPkqkgnYSf9Q%YrB+LE|u;QUqU*&u$b}uO#L)z9A@wx3wFYo(<(w'
    'tf({}`$V)9jJr%K?(;Dr<^XwtaaT&yY%V26@e%my@g>8(g3%h*%JJc@s8+ZS#vA1GuxSXo%JI04^gmY2m(akE@of9r!hs-'
    'f>$X*}Q5EyGfQAz|qc6NWrHR*EQpY97m`(_?T6^U@G+3FQ(UR))F-'
    'ZhxY!bm4mqc*(N+LLWClQ?SNd#w~Ac8~o%m(eMpQHiNtzbH`(#>Ze%!&XjtMn&m-'
    '_WkZsEwok4#lzbRc107^I7T!fhza}zc4SNj5q^FQBKT5t;da+O1FdOdX<#>{%HRYY9P_<7r9@Ualwh(pv(62_?U4Hgktwu*i8f)Y'
    'x0DWc~vL!yygK}UY2il;33Tt{^E31SNQU`zye|=b><DuL$zxUHOx%yj^`!Yu~;%+e!nd4FW2~$4W0eCEL2ov7&9s>vf&mL58Mq{V'
    '=~9&-'
    'x@K%ZAWVMinwwOweD5J$yrd9*vCji=H!N(V1=S_T*KW|8)tPVWGS}ze3O+9or21nZ#bo*73(m`GrEI7U&%CByA#49{Q)XBWi;zB='
    'y39(!|J`iSiL2N9d@EtzGkibfK7W#yAei+31^4qD&Ae;Yk)ASi>#8lVGM<LS1UI!bj79Ra#1jC_y6L~*1Z@+Uc$fFt;GKWRH|E?i'
    'yQC~xR96Nq-'
    'L9=1&*?LY5HE*;7lxdJ{z%VHWF75KYCcfwNNTsI6wt7FO_#H*3W0q(Fibl59rc+U^iWQ4eX&aZvnsQJJKlcGUd)nGpZ|<n<H}uf3'
    '^CO_<SW(v*#jIggYYV8SW5&0r?-B)O-'
    ';(SNWYvgk>u;XU(3~r7GoY67^ibh>4Z}o=Yu}{4&L4#BCF@*}Y7Vg3rAQ74{$BmRP?J<Yjbhz;zi$igc<%=qf8FyG(91H{GTu+)f'
    'OxOY5Ely5>os*^|Kkvpy9YY2!{qsr{nQ!MJ#QMFy|ik*VHtX|TUw+z)Klmin~G+ER0Upyl|xrNLO^$LVUg1@yD<yY0x*Z#Mxm7ME'
    '~_c5Pt;?$QVY!5q#E=1Fmco5s}Sy|2{?&r7xPwYsjWNRa;+ubUgfcne{kvJv@du<yc8Gv1JLY--'
    '{B&SW<7zc=g(PTl%WMvX!3m3=XwE2$D|=qkmLdIyrbyh~+G+Y1cSE7p+-'
    '{o~2);IL@CJPysbANiM+{@d(^&zfp|eS;OtH{H%fm!h@dj)K5R>_GfO=S-Q=brR-'
    '&ZL&&eMB}`|Z4SOE$aH!N?@E8=4zn?VqKVg@mU=5;R&yDn!ys3x7>K_Tn4d56nBy98Yk5D(p*<CY@-'
    'ir+xdW|WR&KTKiTG}?L6>Q<mFd7K4L33E$;}#W6yZ8D7F85L^f~S$oW|`^v<6!Wov76cYq?qtVCHmZ-'
    'B#T8t(jecA=qQy81!k>2urnPoCxjI;0SgN;iwlf5a0|egV-'
    '>!{Zp>DJU?T#`y%ap!vxDFcqMXXyqk3!40l^^fCisC*h;76M(ci;gbI#rxE1+bJ4(5Cj0m0Apuery6S#1B*X>$%RWr#KYUA{4>48'
    'lz?D=vFHZnONEnzOMo5(}2=27g^xKTDXd}uDdV<U61eM7t9CUrQbtCZF6%gA%%o?|FRwWNB<=U&Oyi3YCeQrkc?*WI!f{KX~{ptc'
    'Q&tfxEgeGO>w4=qg$a=oihE4R58c|T6`cB|L*ckAj_POqO-EXXf>+V%3d+YPAr5utm$0bBfE2N?ZDTgA)+kIY0(I*zzyIe5{uv>V'
    '2Y@=va&8E#YEPo#roY<cVwDHj$uSd*Ah_mUR5=05Qdoo4i4t(x5Rhcc}Yk&#!`hS<<9(!~&a_xj4+p{tv-'
    '2Xd$OiB<ume#Nj<N9IBK53LY`8deJW?irJ!K0>=hGK8xb_rt&U2#tLA$SeJhe8W$-'
    'Qqo0w8he0ZnZyLNHPgRAUfrb!;HA5W&t5kigjh}Rd(b`*$KAF^ea%+s>c%X6xLsE_arUqiA<sW0zKl^hUD$6gNj7$pf%wHj2GyKq'
    '-33a9In%kl)D8vJ812J1`4x@G)4%>}8UgPrQT|<>`*pPkl4B%1_L>k}24E2gJLYeMUL&2=)`95ra{Av&W=V^4xcOg<8yaFDaF4ux'
    'L+k5zoA&V~77{xWck*bFj66^#F@I64$t)oMmnDtbSK$@AMPLL^fd?V0jp1jWC1-'
    'P^6mC9<zacrwv)a8W{i^4+ds_!`X;FDD(gRFEd~Fd({eH;?>xD%D*JU!VE`|ey{B&*B3}r2!K0_LEX)+#Y5m>b5A{Sm{d?4u0ZC!'
    'g4{HBD11&FBRC=sw7=ev0`M7XRkX1OZ3JFkku%kO{)$-'
    'OX_CNIpO_p)OGNFT`EBAOi?B%MznEDlbs&pj1Fz?N178~g^}CcZn=0?r<v&k@#NT>)quw-'
    ';*65bK&60a94_M8H54!^IfthSgKy#iTUQlr7>SSw#OAn`$tf4A$40xuYRPE@qb>AhuKF?|$2$w{xt_;I2#VtB(R4E4F372OdSN${'
    'q)BQM}5Y0B}(({esL+Y%oU+l-'
    'Oa8z?MI!uHSD6Ff>%8YS!yO|IV72<gs_HPwY*$sr^kcA}sA54R;1lVulyFaTm1B7v$X6I_@x_0270yl>>4|n*8J;qz5H#VdhT8c0'
    'H1iwE-'
    'F}zC!?Gi#^==3M!ZD5_a@*lt`kOvO&P3Ly0(j3~LYLHrm39293YSx3Z!^<L{%bEh`P9sM|Kan`yzVI6B*m=P9>E$V`Sdh@?a-'
    'wr8<oK8$b72q+s>A@c->2VK1uv{_=u!?My^kan7${q4aGAzY=+w*W&!dX%-'
    'hSQ3{oE{?qe8WD3ZW)~t%e_SBr{>~m8Kknfc?2!mAIMRZQa||aw)f}Fotuxg;MsoYQ>>XX4*4-'
    'Ke0D}sw2_pP?W&(Udk(rFB&Ct79%eO;?b&UG6u*Gma;`+44M%`MvH@mfHDf$0gYz9c#4We{EHqt(WVTS+ukJ6jWZClctB17y4cwv'
    '5Ws)jd-'
    '5n&*oG;SW(^f2qoJ7zLIW2&=*`g2^Z3zgeCY^RI;y;m0_QrH=cwZm$StnIU$AV!BKD(RJC<TO4_kWYW*^h=YcJET$xziml~YkR=f'
    'KibGE{q2dwABl%+Rc52`?*>h!*#7)1urU5>Bzpn0=D!X|wgF(^UKYmlS1V4ptFw4W@q($&<!}46cCU-Z^mE$1rG-'
    '%u^>SHRXH;!RwTLf!OHz967X91y=uO?mVB3~V__PC}GPTNz`mLRrQN(4LA;eXg`NlPJ9d|L<Yl1(OD;Q<4gmqvhRoAoh$z;DUH#5'
    'p$8C!Bs^-'
    'HvpQ3f|aGe(G~&$9N*&W(WS!4i!AXTUqL!z?y%ka5n;k@_?4X3%VIDxBEAYtXZbrZ1nzVA_8bkKXTVp0bZ7Q<;;bWp)iBXPPBPia'
    'encSgPRxa2woHip6nHX;e!R6vRd~bdnxD#JXQ56}BS*v?%Yr*<{Px1#RmrW8J;sP8V9thZ=F&S~iX6AzJH3X@9l{#p_GObf}+isz'
    'KWS_gYTt28V<!9iZ(<KO}D9&jH)%za~7!G>+Bo^hULe^)Sz9aJ$hbS}$w2bwhHkSG3zEvTLRhRcoOHc%M=IO3S6_Z?T7xaHJ3N4<'
    '{MBBZ6FCL0HC$&DZ4bE180S|69viq{BJKA5+95N}Q9&5wyB9XP)=cW+Bm%go>Jp5Z<lpH{K2;j}_DQOMoNxU%84}%G3mS-'
    'TN2I%WVe#CV_9h51wIc1cVgSr(f?R*34Xp-KRE4l*jUy=*f-'
    'hI=O24wOl+tAzQEAmW_dWe56yJZMT_<+TejjRPlBjJEn9FnVokBdW~oWhkiRJVA6nub%@X#3#E`0s%WQ1a>I)-'
    'Toh)<5@B|%huz?MCfO{@NH)uvWK+Eg%fVN9m*d6KYf9raEWXn)=JeT?Si&1XH=+gryq`$xPc9pOEPo&A3c%IIbs1N{UN#!qN0C5f'
    'Ao@Nmwy}&mMODKiO5V6rRkddvQyF)fsz$^Sxq&$R$f%iP+!?AG6;+3fJ5yDo!<&2Hqeb3D(QDVgZfDGxJF`*U5G!U)SxjLV(m|8a'
    'S=@K%X!!r9UQuwk&G{5QI1NgYo^@1$Mo;S#4T|o*Dn6!BGa@i(IOVz2USsNB><C*;BJV$4=T44U>M&g=^XJt%nLp2f(H5JLap3>H'
    '!L&a{_bnmr%x#@*Vfe>U=+nxbWkmdQ2koxVZr6Bnu2(3-k5CC$7`0LS^*g)4xcTedsg3G-'
    'DJEalZrg@9_wLYOMM%^;KFRuZ?z#JX(3v5)<`V;&va?IY!o7<i_F8Z=;vLsXt_XF2HW56UvWZwL9$)l%^vPz+fN^${feA~EY}K3?'
    'fWs04aCl+>?kNmF;H8r|*xpGTY<v<26YlRpAkIY@)^@=1)<~|lIlKzFow*I%8486SpV!R_JMIdsYCMtS3l2;$if9Xoc{2>jU7wY`'
    'ugMI&#DgO^&Ct^SfM<9e&`FXg2H)<e!sh5;uvq9;kEyk<t}luoYzz3xaf5Lym=7^FVr%$N@?dE`hUM-ID<izikS)pV<SL{C;uh^+'
    'cP6)J|GG;{J^qq*Tf{$kF+6|C07Zfsh5^x2MI}<lK8$$@S!y$~)PDw6osTxjk6q>j)IA!k1&}!)>POc_&XR(w+pm(|{ki_4Ya|3;'
    'RAi$`JFDw`g~UGDG3nFVZP#Gi5_cY)&PD(CB2avVEpnDdD@4Vi!vvy4$GtY0DABnO!Bxmb=#=_e_QQNDwieU$Q<N_y_z`0|Ex=4<'
    '+lMC^+21QRz5~1Fyg=>2yZbc2PJGYP_9XiyQpDC!#Aap1xf-#pVWleQkSuQ!k>q2Tn?fY{f6Lqa-3)iZ5ol5I@emMPW-'
    'F~^j3)SvPBB()Aq~|TdHh-z1nD%$g}-1L--'
    'L2xi9{stcDHJlDQwNRRbMWk(3hw9L2tJxZrc6r=)NFmfisma-7=@#=S2K=H0c{KuJl0-Oy4`nnclzdNG-yv&?S^OehvD9(JNA;j4'
    '*Ni=V-'
    '{<+W2msh8^2x@o7vX$$~Fsy9Z)%q0jJV<v!V{Y3@J_KacSfi(NHbyJtkOWu$h`i&pV(Ns;PU1`9)8>IBXMpH@O|b<ZfqirwH|Pz('
    'UOPN7b{D}I9I3hfbQ)BQpNXU7!MMKp(CgR%>>+p$3++6mb*UB8DEIcVFEF<Owb4f>*FC!Xo0HE^~p+2tq<NCvVysX|XGL3w^7Egm'
    '3kLV^=hHE)ZPp?u6r8ZAlSa!eAq9Ge6#$0dQwy^_G?-'
    'bvtcd=j|aM+7dbrS3V!$h&7DwEAqmI;#=i1n%KpBX(~*tfY_C#<fI7H~_Fx<{Pa>WQ}Bjj#LCk_XqG+cO;LyCl@XG7Sc-'
    '=0h$*7f3P)Cx8&jh^CQn|h=vQ76w|$g{~l6Q)2BlSn=h>MP?Ji;-'
    'jifT`1&G&W|mP`3Ph+sh=9+Hnhn+r1hd&;%|f7^E!G+e<hSUtwwVI;?J3bnx>d)m?31r&75@I|*+}-D-~!MM<YHW$DaE-OrQJGF`'
    'o2&**NCMi!1j=4JCRG==M_UKY0LJph08WNhz7Oehu)1@o%o@5myUw}q4y9hvHzimN_$~n1<cgG3GC<7zusaZ`;Rp7){xM2i@=9l<'
    'N0h4e7#ErQ`=h3V$WCZzTkF6ikxv}rq)LNNyaf{Mhx;>V;HgPw^v_Y1|DTIf44pY(HF`1>o6GQD|QXZ>$PS4PWK-'
    '=lTvX1t)&6}m~xMZ14p0Yw#9O{PBF6e8A!W6!!m7K>FT-g&-'
    'jhDi?DV5M%y(2iQ`D#L}<M%4}>Q}puR}C2lz6ZHlWustERjg^PY?}*=BrxDki7RR?=?>-'
    'F4APOe?cdi<@japvx~<u1tKi{cuA%B0honNJDbvCrB~yq~xZ1gktG(Op4-'
    's!k@dHHII7^P(W84pmKxAKB0jeuG|z8JelJZ*+djt!g`@~Obc`+dNm7czE;z1QRGc8!ckB16vNNgS`?ctSQk<AXcFt$L{1(KUS2N'
    '?IyI*Re}NPXr$Mlv@mkGE39K5HM69aeZmi(D_7nxR_G&~D3#&#Zt*~m8NQJdmqci=h=OI}BymYv(RwQ5~D|M)HV?!ThHcUavdbpV'
    '!l=G$9x9m>YTxTjY!KDK3r#!kjWo?<MTaQJXrAhW=R~=mbCVT`qJA4~fv>eGVMiV_&FCZmB&hofS_Qd^Fb4Ia?%w&cuP?=V;V(_-'
    'QdO>~`U$L!InZ#X>_McnA_ubl!PE)13OhlH-`awKC>@;W$n}IBvJZ0LiXv56cP|RGNDuNF-'
    'r1R>_@v1=uxJbCTzv)bc!2MmPw)m0U^%rZ%aqb2M48tCV^&vOc)zi{)>tv~NYPns3NvMduOfykrtLS<p=wnOOPtt;sCDWXNNrRuC'
    '%zpdximO>FE=kF6a);5N##c;Y#k7AZtf4bmvv#xxWDwe)B>;q`ZAqeub^Z1JqHWynPF3!6(qj1?DWtY@inN=IJE7t)(hvUJTrF$6'
    'AIZn>m-e%-l%G=!tb`Yg;y-$>#w_Q3R+59eP}3$|D8o?jh#E4HR+Z$YFqLMhe&IYGQ0yfCXXfi@9u0CHBwdeZ#rbB|cX-'
    '<8l2NbX8J|mr+opQf=aNzM;W?j6#=rKw&n2U<@j9PNMzx3+d@dRDf~5*0qm>E2wzWnEvU|y1=;x3($V#lw@>}5U<UWdkFB;W0aaR'
    'MmOwNxzOpzWQrt3OQ*AFZL__annPf*YGr!ums!ST9<73kXusl*9>KmKwp6z!?qvy6<TLDh_GP<4nj+0tEbU9xWVIp3~~jPWU&xVZ'
    'WfXK0wDhflqwMsB#PsHL^CF)<~iw`n)bO5S+~00!mw{9aWEg%q8+)nx!bs8}kxmSPDVsQhbWu-'
    '6r9`m%H4Lv>58YEHn@yi>BfLCL>+x}867+}5H52Y|1Dh<svsnG7I(5V}H^3L`jFCu~B<vX_skY6Bh5@U6Gds2X<>TF4mjkn;)ED('
    '!}+gLa3B906Ik`O5W%btAFhxRnkr4%RS*xO|z-'
    '1dn`SlsqKlg7N9fl@d67Iev}cB$g>aqT;|LY_>kWhxV~_4+YE2ej&xE>P7w>PpIlu{!Vb5Qn_yNKj?M^y?F_~gKUkm$Z}&%jf~k9'
    '!}5*pP?1q=H}%}9r`%67YB~8vplIhO?&n5}&Sq1seHo-'
    '8vOLsh12KE@S7Jv!dwK(Um`u7pyE@xvEjhq1_MXTNG+uJ#*~pnTTGyNc`841q!PcwlWj>H#019|Yxrs(BZGEA1r*<P_a2;o3!Xf*'
    'Jl+bA(-J*kXNN%)6xY3+fa7eMid}4AQuvnr5J6UHi$IlwKMT7Tu0P4j?APub-CgZnINC5s+y~34$UkskCb{z3Iq~XPA9A9hP4uEF'
    '(_xG_eRhQexS4x8!OYvag>$V7biRtPGUJqV~4S_o5sstHlvQx8(G*OLOH-'
    '%=}2pq5f9&yPIEpmd{WTmx$2np1>F=Ro;UWk}?WZi_W94;8SUOvq8G=X>6UZnW?ui~5HFPa>Hx0%fWc<-'
    '}}*6QNS9VmpDxC4a{^LC&RV%iQALd@EMLWoH_PzYaUP+PS=Hy8<`<&+-'
    'X;ttmXggY1Lf_!mU5!b<9wo%+Nh%HH4j<IKw7|jBEM;;<th+VJqz)nh7?!R=7rPDUY(s9=@02;aL7yylIhIAmt2;6!hd2$9Y&Qxx'
    ')aTge2Qj8U6)jNsV6Va?KR!Z`A7_)}tEr<WyAHsUKCe@>S-'
    'Ta4It%~C~v*@<G3%PbKYIip1yziWDvd<W|c`oAlIpem>MLc)4K3T}#?{1+zygP$aCPqsQzbz)Q!@&{CO*JAr(IXZDL6B)5#j*cyx'
    'b#cS`qf4%e8hJy@yYKghX~}?fXXR*(^cjI#;PpVtqtX1A1p-6)$<jiU%h<S9bO-gwg|hjJz%L~L;fiIuxnF`k441i+l;i@+z`@gA'
    '%5rnQ%iNWI*l59X7o#*jL?n$t2lUwEI}NAY%;vFb;I&hdlw6l;spQR%|e?zV-'
    'Z5OEpt8?YTf8U;eg?2x5Ff?IJ&T4iA8n89ba3;wg#qh{(~RI@$HD>><)s?oj6tbPq8Qo9qP=85xtm90E;>h{n3lu#8fv0xN!=Yxq'
    'p$U|LZg)@Hyo}X-'
    'MF&yN8V&S6{F^VkG?ediXb~nDKG(BPPfphZWjfqoE(+=zm4}#RvF5amO&!<Tmm#7#49hd)Zj6lvhBL9iWI8bv!$^_woCWocbSGT&'
    'JTj$rLq?CWFaTQ-n~BC57)VZ36oyh|3tRoZ;aLrAi$PG@B3&a(pnK99&7-'
    't5;LNAUWx+5@~L}rR%VeKV1NRJ}VdD802cYgEC0#{05)`Nsodwv*aLmG+6Vw)^6Z&cueJ1xI;oJ=TFYDyktieqh|<ui&$lk#^sr*'
    '!M8ASpam3#`LT27ui$1jk)c_zBA>Oi+x8*V`Gx<89uV+&Q-'
    'ZS%@OV>$$xwUs5+{3M$+p!P2Mrbd%L&GP+K5o{B*Svy)f}L*<}@T(zQ7;ohLq$N*}Co`=8+4)B6Qt3^&*PwlBUl2JVO0ooq6#^<F'
    ';)uQEI#b!2ZjUr+v@}<;!1^A2Oj;5bnv!NVCOXk{>pq%}FrCY0;F5l<gz38mWkN=QQM2JkN)%hFpsm`YR?S+qauYzfGi-'
    'Ma0@@%Nm$NbaEZd89}WkH+EfQq7_?%z1*O*5v8;kz-HWWhCLB}Syzxj({@o{UQ?9&P<;t6F>WgnZfjAIkA*jTljQ!tiSLiL;`=id'
    '$(X<z{S|hUkye&Z%|2>m656FeS(zHo;}Qi@?JF|B|DB0C+T$?kYPQ*}d$AAnwVl!j&?*cT*Jdh~?SWX%y@S3~(IDmOD}+k?;Ymg)'
    'N6}iQatyug=Bxl)>5d2h<{JSrh&!+<0^LZ3VbE1px>lCVO9`uJFoEAC9+6!Q=E1wBB<2goZPQ?#fHBJbkjIX|7APwomIcG-'
    'KhS~TaOv;6^L?fh<M$Ywjd%C9alBjW9;TMgY*<~c<$iHjMnNOxY{n0#d<K7}wX9IZ%tTk+{;v&4@rIBF-fZr=ez?Bz78vw}nrgu3'
    'g7uYLJm>1v0EFZMR}eA@sa{O;qH29=TURej*SfuWMP|LkzL=q7`2w+?L##Wik!N;Y#2{<QuAo^mh7~;!|0lX6bMjYBm&)8|X9St3'
    '@^WYvMW=*JcNYX`5?N-'
    '&o#u+2HVJusZnX72D@I#H*N(tEHq|^^r+yCPD^h<1Vhdq4HgK66C_3b;%|cH{SU0aB3EmHVuurAxkA1LDY06KH4}Sx0z$9VE1!E5'
    'YSV>+M2eFSKFD+zy#@jKs(C8~HG+LDCu*xosl8?YXYr;O$>V0Iu@PC2fTRBSv`ss<qfUuT$s|9CjQ_7j(zeP{&P~Q_sAwg69#mSs'
    '6Y&g-'
    'FJzPQ1k&!)nYE&u9g_lM)%fd^W3CANLRvixIGp1|#e59kn5?n`GcW#UT<87Kl`>Q*v+a_!2Sj9D)2km&uNc<tZh;P@A7CUY?1JKH'
    'DW)rl-%T!SCNPRYF(F|anWTc2q6HqoqwcQk3Ikh`x80tE(0#6^V%QlJML2o>EuuVon?o-'
    'Lx7Zi!rm{a5s06@ZM551|~leT_^`d!io3DOwJW)<r+LSgQDr4_;qDhDObpw@~r=oHl7j6}of6ESa}L7i%)*>MCw5ey;)aOV&$Iy;'
    'Z5-'
    '~uf)yCc%&YSBIM`Bwbbbgb43K4<93NR^mO({lLK44u0Gh%#)v2I|7OtYxo*b?rbbZ|QCFv10Z1FaSgGe5@bOa~7kzCm8PE`xq`pb'
    'l2tj+DmeM?d3^do3!12>q{oi@qnRpXNb|*q1K(vfJc%=c)>_*yBK9*-r3h~m1d*Dhcbd}z@sG#OPg7QB>}~VV~P;k7Q--zC{WPFq'
    'IDlH+G~s8nVdPn;s7qv+<Hm(58Yz1Fr{`;dQ`0{vxnl~<O?NMI>YtO;6A^<0CI}K!vf_L4IV9EPH{I|;GE*_vH&{8WY7ZX6mKRAs'
    '8ba6N|puf7=$L}_X|nN<M~`D@(u0&C{*?HI3#_dL<SZPN|dP--Sk-*6}B~T49Fc&8lv5G){{>bM1*(-'
    '5|%hxYDWf;r0cKGrOB6N(&W{R?iq~lBjLrDNurUZQSAC!L8|zl`EZ-'
    '6_@5h}<%;5d>N!o*vF9YKGXWhI+JjTk2Z*b4bOAN#NBf?k`c*&AH(GM>jV9<mrOL98IH;=5l1hyqpY=~vK_4T8v&nd-'
    'Qi=PtQwG*&S{{I-SM;c~(|~5RZf-nn|AihtPjmkpLs;&UuW(N*H-'
    '?>Gh!x4O>hlel#?Qug7a4Gi*O^i@;`VT#`OkAPZ=O%)kp{CtD}=cv=sTTVkQ}3{I`Z!cN;HxtSCiddR_bJ*T1%nqZ`{c6a_f{EAN'
    'w0u*{o<f&wy|KO#Hzo>DZy7L!;w(T0X=DfR+XUa_fWY!hdSOv^Fmn$=Ey@(L-'
    'n@Q~Q=S2QAL0iZHyV0@mS*L>0bRyK8JN`xaGL(Li-$(`B@Sj#k%H<E|B=^Vc9a$%v5Y!0Eenak*}C&tiNW=MXk4IIpl-ckBWKcK&'
    'mwECB~Kfa0u4Qnr%|P(Igg;W=?%REI#W(3NY$XS#}ml=OV0sP41wM8l4uA`o;;P3I9Mg00){(pcH&Wh&QlI9yB9*^P^Hp07a0j&H'
    'KB`uHc#cV7n(kt63fwEMHrQoac~5kmyO1%(R6baf*~X<-uzD-'
    '~tCe~_^al&iP&=kO3ojau*sXYpoA?mt)~zkGdJ5;`cyFMS94uWU*$vY2hf0bGt=?!34vLk$2KL*tM8w60zhB^q6AV79~ax_X(<(g'
    '=;$$~g3yTRor2tc>yDQDsf#@woXJO!RkPhnVhtwat1&6Lg$a=kVM7in-'
    'S0R}9LUXfaN9<j~b>Y4B{VUqG^J4X>Q6j}L8!L_tjL?p6cvUEO6v|JdSi{sxE>zg*IKnX000TRXZbFL5{ZPz7yy9?Fi^cs;JzKdF'
    'xk<+G}%xlTU2dWMH&=Ty&f#XP5aj)!FPs&$NJJGXkFCRzKQc+*Z&+)WIbPD5RG{JcjQEYGDP9K;jAxK5MM4-'
    'bjBA}5HNC1`g!Hfk@{>DSnL?z?fF9>KZw$z{~<(TUA^h=L;jAxT{8H0!_0=Gn*iyE)T%T0lLxm`vR4o&)7E5=-'
    '3ktkTl%D}Xe+3`g62FmdjlG_ERivBANkVDH!G(~X5SWhDW?_eHOyxK_E<Y9HK2ns^8i395!qXsf2kSJR4{Zi{_(T(N!xJ!g^=yw)'
    'BDZzSm}x7eh<uf>3O6qo?+3#qlWw{Z=XX)b-%z}I(|d|r2j$>wvPH?W!BT__fRGVa4hXiI-'
    'K&c$~W?%YRBeG}%<OA&!FdQQc;J9({U$5Z7RncaGTM&@LUVQ3SQ)pEHByJ0O`Y{KJd5#!(WnZOa@pijgYY+=@oe?eETM#>5O*o-'
    '#yzZh)le=*k7|6-'
    '`A|HVjC|BHd9{ukp+{jVe|bcun3cLc#5O7426SUHMufzobkNfW^i&|fzyj{aF2dZWr`*I*gy2v?ek|6GKWvafbgYk*esbvmJ#uYp'
    'UPaDp94!vT@1jhjESvw9}5^D>IR557K|Z_3548}7nJ5L0isU3BT;(wEpAPbW6VGl|XdY+`dfm)IQ7CpO2r#O8Q`FO)?Fs}r-'
    'HuNe2bIOlxTxHrT(=bK<^1=EsD27}fpKaeNZoNoF}(6{7*y*V~aGejYt2c(FTEu+*CgETE+wm|-'
    '0oF5j*f9Ija0(B(1%UOx;a(1G-oRjD-'
    'a}wQUZlb%)OLUiWneO5<LE~dN*G*qH?k#a6`G)Zaj1!x08TW3qCdu?bHFO3Cg*hxO&`Cu~aXp6p=fpt%5UO+cy%>(4)zx}VC5LIe'
    '*iB}*#%o=|OtzRpM+t7fMr8qNFCPz7Hj@5=aaXg59H`aT8TYgiX5~}HeZ+{X)Spz1{j^b)W``ZjZF}#{PDgX2ipHok6dI&!3C4wB'
    'cZL}wTB85hMqqXk_RpZL`6(HHxqV{PT&Js7I8{1GlhU?Lm?nlRVg`S*N)xkFR2svcrV{Sybd_*VXDZ=aeMQpL)0N*5h1j~Uq>l{2'
    '^la@`6}YE9==NqV_U|c)Oj1qqJKf33ZXA@Dy6JcoI(ml8qq8%4a)GVu2K+fCh#GGbUdMm?ZgDcWHe;oC_&8a;79D|xqr5n(QMyf^'
    'rQL0cHPc@xq-dKpUd%!qsqu<qvpicEmb|HkCvU1flQ-3f<V`g)c~gx_-c+NbH&w(0ovIQl?~HPg0sK!-'
    '8Q)JScGWL2i*(<PC*fBZp}FtGce70vq)w)o8HZY@P|S>d>r*Lm#CrKOiX5?DdOAgp*hM&#B1b~;j`bI>%JPlR7+Aii;^2$MU1I`~'
    'CzX#I#kh@d%bA6Dv(D@5Y*M>1YbwvgA8V<dNgr(2hCJdO9~SHY!Kl^c`kh+bkJF(Dmm>gXaop#r2@L)ExUU$N^inZ~pKoeo5@7g4'
    'fM{#Dld|oVk^kGNWgo%mW5siZTJ{l~zLz-8P|H4o)5nY747Kc|-'
    'OR@tfcnJK{qY8HK9QMvbORutSmb@O0esfK)zochdGcd;T1vCF$I@&O80kJIEc`^;paQ>1IhHcBMnIiyxDCdRa_#uP=bIA!<d+jmE'
    '%E+X!yJzDjB7U$--<5PaVEm0(Y-j{FdXzk<4!OPeZ9!I6AdG&FE;*J1oUw`#$(Mw?D|9l^PxzN_*4TRs7O*hrU3-'
    'h!g9Sj8tAj<>vejmsMWx-WD4@{t(=5dGhZ~t=!YUR?+74kDX9Q&`xHHF^0OgDmnXcZD4zrIBR^I7On^Vn8OkR_85jFof>NO-Qmhu'
    'og{Fx6kP$oWKZD~L-'
    'i=r8DHZwo?T~8OS$Zj2yIrK0Vzt{%dMWSIZm6ke8U7z^DQ6UC6d}JKU4e{`eyw44p8x;*eQCJ~?jyR`vl=3TtFdx5F$13(9B<qK4'
    'Mt6022c&drY{HhnUVKb1D+U%h~&_Fdc2`8w;7lU@hGP6Fy!{%@pZ+r$o@ONq1biRNA@=ryR7(|`R(||X94R-Pe2~-'
    'v&tP7F5tI(T{b{&#DA=uS9NBQn_UN}-&rF4w7Ye4m8e@yux>LgqD~Vrx)iOeY0&76tqKH?7?0-'
    'R2*lyeA=<sDeG$*z4L;R)ONkN1lS&j+_E8X+xixN*L3CTUW%+QwiS9Uap_RCBQ>>YpaLR5jD(@CN_)QJY+N^&ud2={6N{5?K1F;+'
    'h=Z#{wb<~vtS<@aFL_g0uyYX|xpQsn4iNup{<@wTZ^jT*#s-AqGtyLi?GEDtSAbgpU+RA0<m}&x-'
    'p<}CwT!x}y`7a##CvzFPcQu8}(DBt&E<^Y6$>TyXb$x=hIADv~tcxzsG@us=IPz-'
    'H6G<{6*Rt5%S$Vi`HBqg74IKRzmJ*2m^K6wPMz4bMVAg-'
    'Q8m0{yf^<K$E3^FIKrnf;KfhkP$}kJ)F!D(~HSV+{G_=Xih^KE?14La6X}cK!SmKqoy8*l|DlcpX{$D8l5<z2)$`y<HQbo`OP6j_'
    '#xvu&^++%#M!ke~2BXEe0%XW4hWitd+nXI_!goa1<{xS%XXW5bed>=8{YF*o4C8zNII}IORBN?K)$=6>|;4d1~KQ2)QEw69JQ$3U'
    'XWlv)d3U`?ZkA0~DEB257b8MRCLdUNoGY+G)NYoIp-'
    'Bn<+m*yrVmX*l1JA{qcA12pW!gK0N=zsJNXBzPC-y8lOiJ(K08Ej+r0Si2m(_%Zb56<D_X|hO-lGEY|%s!aM$<s8Jg>1)Q{2j(BR'
    'X4b6NTzEOO+q5|HB0KZn4G7E7%?;vLGjcArf&YFjl6@uJ>Gvl&_LG@XuK|><RZ^i=hW9|doX^pL$w3o;#m#7BZ#xv8tw$nezwi}>'
    '4t3LQQ*!ts!4+THB1S-e7M4k`3`$3ydr?Q+5lIL4Gb8#(6HRwa(6XJ4Q^6`J@YF=e~kOBIB{K8Fi9muxV1hm0xjOn5F6E>)bz{-'
    '5xuWFlX9M0U%vN)(viiW4}e*m^`O7oa7IMYDF6umeuTn{apULsC22@YB$%K-'
    'ZW)Ypks<75_@481=0?=zIz^sriMu@NQj#)JP~t0UWE-'
    'q8dUF(Jt72XD{~^6x1MFc_P4f@dMS^&cWxYuU0gCN0LTuu((!9D5w%_rbYvoBknkb7RuQL3>HnJOtneA~PJ+Q1SdYG+sWB?dQlad'
    'nC$i$x+rSRG;%}Yls0p$Nqq1I&?sV5YMH@7Cu%E>m$d;#L}X+y$=;ivmFjiXX1K94|L=}cX$#dM8Owzt{>BXF}OMWv4I{RZ7#*JX'
    'p&l!U=^Ym>}?Tj~nV+$R@`u&o)<mbD|n{$qjDIn^AI+Jbf!1Dqf-F*Q?-'
    'otB=%MKt_Qr++$Zzxe?!b5})~P#+Mp(qKR3ZnW7|!AOs00Wr0dc^WdG#CpUH(IS}SwJNgbDHwK|a-'
    '2C2D(<xJIyH<dY1_>&Iq~p=fw^HsW)EukV!u=qU(1;75``Ql(Ft=ne}K5<O1Fm%G6|U(U^^-'
    '4$)sRxniPynlY+g{q+stfDHxw71^e)79;Wa*BN25j)R}M7XzjigYfN@Zg0wg-Ntbb-siA#aEEcrIlDie{uTK|?)nwcvFj9Myg-tW'
    'd^P(M;a7eudZIdR$OdHqM2(twV!vlzleaLZU8}jB6`oOu`y|2NXiydv=O=|+`*Bz9UVyD`mq88TSd_rlYo9hBXPo|q|f0&JDzkh)'
    'CIc)a(2V>sC1f}cU6{7UG8vBg9UXLjBkUu3tUEdI8jRTasIdIIdj^`CkHLK7%nVf#um$a{rnVR}294SNT6V~`TcXDpqwcMQ{ZR;Z'
    'xUZawD?i3qAm+m2j0aEO=`Lf-=#@VDuE*Sm&eVHX}qJw-XZUdY}&{`H&ItzRT8E2FrdBhVEe)>f=***|SR5G|(mU~jeA#v#U!K}R'
    'Luu_xWPH@qAy2kA4p=MVe?9MNls|_5v1^yP}UDcF1(##jSTswA3C*w=85k*SWUx3}VC~0=FZeoM!+L!3EL;)E#_iI|>PRRzMkOR%'
    ';q194%TJAV>gF93FD|}oU)esCtK$!o)h+*oVjr(%UxPPh8^o(8AhZQEQpJd6WMTYaheU)2b0{|i$x}e>Kk1?A8uNM@Xa1a5x`Bu2'
    'aSN5dl9e&zn?sS>u`e!q&!7Ts;kP!``M+BpXgW^y7D}@npZ2SINVLlx9)o&D<wa3I%I<=6h$?oeS-'
    'ZMo82WfKlnrLse<jvETM8bi*!wiw2>Nox`vFyFS57yecj^}G+?$8gFtSr!+t(^`J-ropc<P_z?V#Sm~FLZ*ciCzSR^wS9o{agY-u'
    'S)>v7g){sd_!u+;fFs6{f~5oR!_6-NyvL<t2nOr^@a4>h*o!t%|XFLHtPa(V2Bnc`{o>YI=JDpTY=wqWI8il7+(qohE%*-'
    '6iJIE&jjlxht=s&Zo9eyls^}F|LCG((Kb_PdvXGCs~c(|4<GUI4`~{!MKp(h%`2?ZHj|DP#&MB|NFD8G5(xd-ggk#fZwAb^DG{%p'
    'gMnHcrk|nk8gFxAR|_eZwXvvvjt4%9WSJIlR^2QB&vugZ60#o&qm_B@1<IWkmd&r}4;w{?q=P#o7Z*{j!{orFE>rl0;wQ^S8XXy>'
    'A`FZr{B0yJueQ=Lv_b5Y>ZPF1k$0U{Casj$Giy}Yb@;7v|B-'
    'hYe2)CrnpbU6F8W9Ox0>Z|Q6D|7%JR`?*?$+Cn6V@?^>LZOJ|=Ucy&<Q{bEEO-'
    'R?wA&)m%_8(1(x;W4s2o$MA~#*gIm%%zd7nfHC4e_6Gpd;`GXYfH?p#COTU~{DY<u@|WS2Co8-'
    'dNPDKjizl$pDECLpF4aqwTWWo7kTbBXn--'
    'Wu-!-nXn?w=Oth5goafgzuYDjX~BUs2r=`M9BnT3rb5lEW7*UURw%5K1CinZGTFf1Ou|DNRMml-LB&IQ~)6%o7GL=<hX?5e*-'
    '_Ey|3_G!y_Ip2N{o@b)IJ7Bs$G$IQ#%VM%ME57@<b?puRRaCfcu)ba@s3g<~%dF+I(Oyqsz&gYZ5Zj3mag-cgwX?gM=0w@&!|&7*'
    'WvCjLW17#FmPAHU+@Q^uxceQ#-JeO#=?p_4_gjCs-%_r-'
    'h7_W^j8I9=0$f4r)U$z4mgMmqz^sK>G7m6oZVpi7=kDDYC4K!RIBXF|?emy%N?H$h7`%2&9<ncp4;RVQepMHuw`XamLd0vf?x&3b'
    'z(3s8mK5P=)jn73xbIm1F?indUF$yu<4wM2{TXBfOhq4GDPhaZ0kA{R_~!!mmx#)ZU^`5_&b<^TE<PRGM+vs~ZurBv55&`Yo^h)b'
    'pVn3)l62NDiy10QKsr>wFTJ-=WbjCeJ8>L6-'
    '~JN!Zo;Z{TKD7Hs%=D<2EDru&f_jAR_)u?Zw1zRY|uGvu#Q&hJYSbU1xpgB;Q9nASeifu%Mz$yc>)#OAfN*1tohe=R(k7r6{oVvk'
    'ovsqx2Y<~I!RqwGdS5t5kAuT)k#Mdf@5Wi?#ISWvQf9-T|L0Mxvvy*`FycdKd}BRFzR|cARAaD7)_2nl7elA5lszE$4e`rO`Zc-'
    'QTP_;8+W^4qAoP<PQ|A6Gs*vbCU^57Z4x=YXll;3?rrXbTKAH5+c(OrcZOn-iZ#UkNzlk#^C^=sn4$7D4XC3so_*W64_dbPcynMp'
    '^F`dopbJh5_l08N-'
    'VYFItZb=Kv|HlB%uDpFRR*u^<L}*N@Y*T9JI1<`8vd(HVDO43ZWz?u`BA(I^z*xx=b!E#<2H|jzo(V^gXOKhOu1#&r@^&$dZu^0M'
    '%wAK3!zpdtTXu1Y$W>x0&A#Zu;_1>ABCHFXFZ@oZK3U!-=JSCR_*)N-<ZrULfdWUM7TIrvG}o1F-~J4X@K1SF2wpCSkC!tD-y-wX'
    'XJLUf(XSczB}Hyy&61&hK1)o4R>&SHy4U=LDv9yI^~|UL2>JU27s?7+qam0tX8SApNZ0OHIk1uxm#8%QrVLY_aoyb21LP6jGJ7`I'
    'OiL5GV`Uov{<@-w?4g~8A=GTr&;$<-kmwax<|6I39O~xaJL4x-'
    'We<s3=K&CdVjjW)`@j2sS$pSbDU~}JIT7YtPH;H^QShZdhaUtpmE!3QBl9n`igr~gB+eLM<gN@>Z?W7pg6M~*_j;|6~J1{9Bl}wP'
    'KkrfMh*?PI9wMC8tyHwRsT;he(pl;FiVv7R%<B?R~I@JYZYvK7DU}*@&XGf0$**>j8Yz}Pt_(w<98i_BAZ4iCI@x}8=OCKD!N(;v'
    'Dp1m3A5P!N(r^t{aOjP*gd3#T<jiJ!Y+1?D4`d-->7IR=#DdPT*K|3d|a-#?k4W<TlcDUA85o<;Y;<DZ7~)i$~2NPx5|)-'
    'EyyPGX_GjZlB9x+Gf9S_=2>tpG1A+T=w(z_Bk`VwTc?r~L__p;phwaO>ZT#*JDKJEk$;~ln*Jvkw|B!G7|-'
    '`8c!F@7;4DaNMQxyQNRzf|L=l&EP#*qlgSbR|vOdpg4L8JU;DO2m>(^pU3FP{h8r-'
    '2P7)GKj=3aNblpX};2gK$gxVKklMrGa6+cz!S3?)aW9RsjmiCOcJ3XeR2<)NizvpV%u>wax{&bAxPRCd!sR`jpP`E#GN?u3T_8r~'
    'gi+?a;jH?G%-'
    '#*J@qZ>w7GPPFdLcwzZ+er?$}oTHr$7sWIrljY*IYBK86gd3&B``3TQ0FTnoBpTF;)tn_QoYs^U5S7?3V7_>VD4eeUBjwxaQP94f'
    '4mi)cuY&<f9Cj~(gGHQnUk^KxIPP8wUz9lSUIu58IPhNXt2v%{-vE2JlVo=3&)t3T&VE6;zt~I#k)%Rrhwuge%-'
    'TNd2KuQ*B}e*y+ob26Oq+*@Kw7J2gR)4~?&A2{BL8<6!fb1;5MD$yr$XupnlB%+F(U`GD3xM6lRnSRTBNxCQ|DucYiazMcH+080h'
    '4u1z60@{tYh+BsDWf1lkY)iAiL<k52b;uWAX#Y^<)>{ze7KSFTS}7S<(Fo62Cus{v=_9rTlWS>00w^U`TL_ipj=%<FH8vtVOZpVb'
    '?&d@xv6NcSO_7R0!X(%W}9v{EldvBNPI7vJdw;-'
    '>+=8E~i`fu;mVP4|ycc%Ve7xqN{MErdR5_ylUL7>3JNCE)1`auSxr@FDkOl_;cf?2_yTifeGHV<-'
    'B!aX^AJ=0LXq;OFIFPtLMG!6fkvY5<i8-'
    'P$J6%$wfYS34vs&F7e|ueH|J<=oq3mk5SSpn+S<V>6J}}x~=rbrmD!AAd{T0vsz5`3J7%Xqz2ZD#;u5dZk=*ZSvuWuw2MY`wy*b7'
    'kc5pBt39n-Z`}?xgj^5lXy;c(XK5Y|d5Sbe8jR5&CrvF}VaL6g3G*$|QHqn)H@h-'
    'nzVLngNBX7Z7J3Kxu0HZcJ0%6BiF>t7VU{cRfEK{vBuEcRhY`;0@!iqZ9pCU@1+|Em6<6Dq2E+0oetfTSTg2~qnsOVAyE*N~0~N$'
    '`cdsFuFpp?9egU!o<X?lrHkgbKOw92vs{>B9qy%<X5s{e9Mj4pe_&oI<EmNDN%Kaj~J6EBV_Mn!&A<Ti5KrN?1E0blU=*5;tP`e`'
    'Gx5!cIHS8pXxz+u?($Lbef589Yf-=(_L|;fsiZxa=>C#5m#u-8;fPRCC6vlQ^JK=NIM^pa38jGQqW;zRe37fBBE9xf-'
    ';xh<%NK1HmO=7t<n}oC`U7{%F;Fq*jGunWJR}uox;2v#~1K(1wxP8htEb3Mo=>BFTHem5uIC*`a=Dx}<BvQ6Z{0Bc~-LVb-HN5+j'
    'btg9bSHTbC72mZRH@4ySYq0vtaeg5LRO^1>^FA%3k5`OanLfn6Xjy01(*X+d$Gcrn4}<f|I`@iAnxjOGY9qJ^dR(cO(MyB6+$E|{'
    '<s@yqEp$$Dt0c&)iR3;2HLSUhF+Yp$W5my@`+z~P>^@*Bth*2J&1i<_-HubwfB7y%O8!%K;~a2GGt9adt=lfYIDAzzT(tAo)h-'
    'US3=K`uc*U-)ub3RT00ZmD4<q*@W~dYfNtGvaex7Y*;FC7i7^gJ1`Sx}V&27B#ZZjSFc<l(wt-'
    '^G?4c7=S>MGnKysCHL65*x16XB41TZdX7<9IGJ)B63zB*GWn@z%W_cit(=y=2^tIUZ*UYVj~e1o3!bT|aVT3E#i!J(FaBg_;XJvF'
    '7~TDqi}C66#W-9MeOqm@6-8v}5)RkJ9JKBa15+9<kgZ-OE`WTcmh0eM*{0KG%jafD%X!0QYH<`J4*8^e``fUWs_vA=cd^)BPi@dr'
    '&le&L;H9$0d}Xg&Bzi?{hJED$gj))ysy;)dO~1bJ}q-!;H4PZ4v-'
    'mRjv|+x3bK=N+bql?RNM1X$0tKaCy?wjRnAXtcXJT*Fg!+_>2;P1D^?kZLP$O{ysqs%mLV#hkhNTjLO%=IT%s69~-'
    '&e=0G(|(%`uew3BOo9u#|WIfb8>N_V7FWAU|@SMQqoO-uTl9M*jO2~uMky!Fb}H3l|h{7lEfO@bfeIM{IUN81a&iP=2!{(PSKfEb'
    '?a1EF}{6o*@X#qu9J!ulJwMCx>@0uURXvpn)vmET^yxY(q0q7TrPLUxc3$YqOe3kReLCaDYSzNpuhoq<I)P~x?Dfz>Qry{^ou>AK'
    '9RX-VePbbaR3v@~;ST9!FAEzg{qZV;3483an2CB89dS~*)7b-'
    '5oUrGtZj1}J=pcJ)<RgT)AQLP?A~x$|ijhk7UuNm12LDNR<(hv+WZQ;#B^52YnvsjZpMXa<uvTx$a7vP@-'
    'jCiKKJC4BcD#aP8~ab!s6{8TrmQ=Lw0zFYkB7$&uae;~u9wD3=4Pf3#fBN-'
    'uOhJPj_C1LgtWfYej<NQ+@%?V|D2+Z8m@h9STsTKXczl)zF_Wj+e050yT-Lui69BR`soBL`m+HkRQXNIS?f>A03!gR42h%ha@l!C'
    '^fTL2tQuacm*QRQq=WGS`&o<H-'
    'qiI1#5^Q)v$<&XItQULp7ey8Fo3U~}LAd>`FbSNibrsz*jDnrqooD{60Z9FMhjZ{g&YE%+ECj0nW6}GV)Re!A>+BT>0%g(v^&^L0'
    'F#x~Sb+#}bJ{+6ILCZNwGb;d+=>7>G%gkC`G1d;66L#YOth8CXGAb$g|z<r@1Cb+W&pLnWr>y5if)i8;zHt9zSxHbdql4V+|Yy4{'
    'aL(#A0*w(GgCt<u2n(FYusl_;2#Qb*sdI55pif)FKlASsMVK383kneJx1o^JeNs#YKodo%=(n*l-'
    'YMli6uF)*WHxBd2EgGy_UA^Q^u<ni6nDjrzJE}0u_C;Xq*Zb0UVu>xm1>dZ)sYSo2Sx?WJ+IbI+kR`n`SZhvDvYKkhbgJ$;x#9VZ'
    '`lj288WFom@LHY9^eJc*g*H~865fhthzQU97dwa3-'
    'TpuVjrrPXxZJk>?~GO~Ug5vp+tPaS+wNVtCPrHKvUTr|8*h|#uUO*b%lbX0+z3&ES!D(Ul}^2hphB(f>0+R~T651V8fsQu;V%sb)'
    'o{KIM8CY?s#)EA^F52!cH&-Knstlg6|ev*K>)U=dv3Z-'
    'gh!zMfENrCr}$YPS8i1JtT!lXixu1h|L+R%O8lwFr)T#B!d|HvZxInT{7{4S3m=hBSa)2*f0fj(SCA=auMU$8>Spbu9_LB^pBgF7'
    'B|etCrC6wGgmoLN+p&=gh5XRQyY*Z3y|)G7R(47LG`rfwd)Phx_e2y!aUwMZ<xiYEO+}FyCr{H*BTma5g*Xrf510FN?%KjTdM?sE'
    'q0t6ah{XkbVlByQEr?Pf(M6LJc9U==iwtR%x<z&XplCRPRu_zzOOM=>!wtCxEvNDU_hW<1$ZFPW>oM*w)<>l9eYXf?`m1R&Pw9{%'
    'cC%9DI9%H|;qn$6i@)k9`%RV1g9(Jw0UhyZc|=C-'
    'CyPe<Y{Khy$3NJug+)0I`)!me(%@W%h|rsZNUoYaNNohWrhKH=46xE`CvI8ZmR+i7hV8G1UgJx+0?Ii<MJVUDiv@c^!5Z_JEIo_5'
    'e&gUW=U<;?drF1Z=h$mo;dM!TccH@T^YNXp9j9oY_e<IN8SB219Dgpc?(50UMWz05L*d2k%pk${6I%xX2wrW5PBDcHs>SuGSw*-'
    'Z1PA+zdb|$Qxn2dJ4kTaPF17i83lZts49RuU+I@!;;IO8xfXC+ex4}9&x`FEYF+sm=A;S8P5rsGgi;EOqUx+K`bHj@P?3N;Jg=TS'
    '3l%G@mu1p-'
    '7pCY93@WI!oU$E|*$rwkcPWxuSC{v<)`u|`Re67j5*!|H&GhU(BW>PdL<V|fUZ&B|3mPtE(>`?sOVz<#}mu+(>q~l|WO$xpvqJl4'
    ')G=JdeT5WKD;^11n?EcI(X!WZ5i&UL~xyeQbfvd@Ey1<`f|A_BtH^(wP7@`lKwFM;@nN>5wRhMG7-'
    'nCM8HVb=*Kncw@cU|hr9TU51t8=^Z@tam>Fr*HOqq)e&qxrpJ*x!c{BnHv!)re%VM@bW8wkjku`&41KE2EGg+SMFJ5b>wELA$$y%'
    '=G@+pXpqnWo%-5jZ18=@yzQ29B(~Kb1jY5j0DFEzIx$b_t{TFY_R&>Vl7r}FS#FS2I5xh-H)|gYp=SW>o7SA-'
    '^0g{Wk$vjVWpZFkN*8qtJ&nGQm-miM!D?)J*DQd#B5f@%&>_>-'
    'Ag@oU(J@FbH7z5Q2Qh{Bqvh)BsL@`QTrseYNt|@<iv#MVr#zZs_WQFs#;Pl5#yKZtLxbt&9CiJb{6uByG*S7mRHN!wXnLOx`Ax{1'
    'PKP!;ZPs=ANwpG$o!z32WFWufwu}Gq3pIUG$*3&Ue%pUsL-aNg{)1q&QeSY3GdHFXEmFZo`VW0n<39p%ytOxeZKLs$?CazN+3|<d'
    '|?F`<NjI#+SNMX`9wu?w8<FE(=>otLj}q{!ZLJinSvzrpR=Q|pN{}Rt7tZ0xdYn0*Ou+xf6t{Ne8#giouVr=)n#b_--'
    '|9Cmsj@~Y?oMH-D|L2Vu5v^!FGui)-MdYZL!3<-=NzTYpe$hHc~9I9yI8-jZS=0ox?yc*EtN-|6C^Bu2*RS-RAe{dkoP1<L>7B0h'
    '0V}^5nY&Q8e2~AE}1H>7l_`FViy@*K`JlAgRylxnAMocPAdfjNE3rJ+Ll$*LIP#N?92{g$6A>i|O#+%%6q7m@YK#K%F!%y5$%~=0'
    '8qhb)y`@C^LaGSlui~keFxT1hTKVKY&aWQgF7-'
    '0pC~Z9PoXOCct+g#Y;WRyn^~+zPWtKx($6G|43IA8d}W=c+!=eU{Y6dqDfuJNhWnAC!5rjoMKW}a;iyP$!S7<-'
    'oZkBKK6h=V4<ZSuh~<qTcRUM4sge%8V4(FsFoZ^HOmt9W0O$e=>-'
    '(H$%rLwf)fv@*JlvUFxM9RQ8B9TuyhgdV{S<sxP=p(>53)!84fl*?<@rKK-'
    'p8LDqsB3Bk*Uer#Ro+TX;dljBY0jFNjZxRr{I7e~qHD*6l9c@NZg(v&295Eemz1xURlxPb{KI2F{>y`A@d)I?a%})2&;k143#X(5'
    '1^Y14z%b{!Rzjt~iPo(JSw&{kXU`SNp3CWkzR;q>KF;TM_!Lwd#G=#}g`FxG!M;30}SKxO&{Jl_S$0%;&i`Lq6)<d%Onrs=V=bgq'
    'yYh8kZGz-$G>insvL%%%#;}hs0ESf`NZ>OtmMQjDj`AWE8BahACM5e0?U?6LK>D$<C>T{(zP7>#>^S*iUNR?=|$>9kCWo+W-'
    'G=0cKNCr5*Z02XZ&o7g5k1iy!o>7P@0`%S<z{jXs9`w$L<LThlJ+7B@0#WlWE4`1a7Kxioc$av<a&IlnZLt+Dm?)kMzs|8=@@pJY'
    'j!pVkJV@5QO`7X>lJnpM^pnk7D>EnM$*f!oxZPiRk8?qlIx`ZVCKLI&<K^f$Q4eFfSZ+_}FJ4z=9mz8ZZV2B3yfL$eeT6XdQQPP6'
    '9WQdCTmyLto-'
    'o=1pW#}v7%snp=RdNj@FM_*94skyf7)BbFe@WFs7>rnamT&;5n3Y@{$#0$HNuq5@2twy2jt^u_*3bH%GYF?p80VSPawDV&ZliK!Y'
    '#76~dJE?THv+&4?yK+Yh8<u$d-fv-'
    'C^2uCEo~NrpMPW!h+uOlb%YO~;wukAb{~EsujJEvO2tX%T+N{_!`+2d|y2@xZUq#r@aFpdOZcWsASflfcs9$IJ_TXQ#I@@biM=;0'
    'LNS^L2u7GY(34FE@+qw#rto;(I@q8K^?V4sMJlqWqAk$TDb8<l8-'
    'w>ThL2f+N_^(l|+(KLMXbym!pdtQMD|r3=rGHcKr7a^_sZr2X%g8!xY?y$x-nGeK(5E45&*~Tmm}_j(H6R8?ZRq|NjGt-geC~6%s'
    'R^mgrx9C%>YYaesZ=gpKr^XS9$ZLcsZ<_ZM1#0A?zxy|aYNHc>O>>Mo{37_ID@~)!O4yDF{p)!%FPqny~_Q4Q{58C>r#CVX5q8zi'
    'l)X4%1vhp&+JsWrVhv6<#xr2=>LHD6J+0FWO`XUCoV2MurJ|nwy$!(h5De_|D{Xshxw)Vqda~1S-R&o!OvZ-GN%n9IuMCIh(gN%H'
    'e4AUnQ`bK%c5uUfS<rHGlXG95*hEMfQBVoQY_g_T=cw`kH9_8X_1c^4=YY_P9|a%3qE%z*P`miWI0#e93AI^Ld4$EA6gkrZ=(CdQ'
    'DOxVWanb^zZ>#_LEA_g+r{*}wXl9_#@uhMN+nY=Q_XQG!|}&n^Z#f*r8z~pkB3Nmr6T#HBD)3V8`R1fykyrGWqSuUSz9*Hm!sK4>'
    'c}j^)*i_JZAtSgP8t^z*03~q|AWG7Lfl|~Ce70+@!elY&T^XOE@4YjZyb}<8#mP~+AB!|?VY57#wTf@eOMZ(9U*j(4U7tr+x3YBW'
    'o50X+R9bxL_}V!iNr;Vw6xe)Ye{U0!H`dqJm;kN?kUo6oywm_7Ri4KY$lc;22zjphq19Dr(irb6Uz<*op+M>VfF^cTYQ=qwfmrGj'
    'DH$FT6iNH;P>#VXHvm#H41$<ryyjxCTg2~oOApCTBI{|@!a#B7{KDyoAoa+Oe<22Wkty}7|2;xp-'
    'Z9>(eEq~H*F2bHaHWp@#mvEr^JkNe^hu)jO+d<5~iOV*L@74VB~V_jl+6E7H)Fzc6C+(`mD}oD0g^!PUH;4w{s#QG`^i15zO)Jya'
    '=6%Z_kZrdwDKrDEFywFn2fZ8?{2<C|bk3idtS&QOzhM(sSuR_{d63$>ln1254Jg6QC{A{Z4D>Q>@hjllBGquZ4=~Q>=%Cis?(wM}'
    '$RqD$xO>kUBQIuvf{c^<M42MoJe93&q?Q4r+X5-'
    '2wALj?8!ZEh}D#7eh$3K)cNLYa;xUt?R4=$k1oKjT?$Z^8stH_#wB`G%U+HWaACaV=rXAGF2p&ABaE9Y04cPDSyU&waC=gsvIBwN'
    '0pOOpH$2Xk9##^ME^zQtc|;9v>4qZDOX0uEWW|_`xKEG#_*aJ>?zF~F0qo*?SOWn)Vuz5r7p6J4$d;V3yI^oQ0}_Z@+fme%B3fPE'
    't}i*{>A3{cf+*1EvogjBjb6?ausxta_?Ba2Y;#D>y{f0kI?TtCB^n?n-'
    '$x?l#wS1pOG7D4R6pVAjH=)i|4^6`G~BcX=#&@tOtotjG|4b46-{xy9c!US*SFx!8kUbo>YBtxmJ{l(A9N5cS>#bM|EneKc-VF{c'
    '+97!e%PU;Q9a0Z6k`|KFirwI%5E}818?8FmL!&g1~GzZ#mcC@BBpo`W-'
    '%V%aw}!N&HdxI1l26UHIOv&Dp9FRlG^K^PRaklF5ca=(g~2YDaTbrlhM)$Ofo(y0MNo*hW-RNo?0BZc^L=i=|r_V=ywoRPR8J?yf'
    'fZ*51`fyZK?`?vLaMpT%x2_u``(>lTw=e<f}$$uG#ChC>^sora#z=|wj|KJ@}LdWHOoWL8M>$d+>Pi`gp4kD`F=pc`ABhyP694=I'
    'OVx}EC6Q6<XdrcJU6@;NHmL9vw<$VDiWx+C_<$D*#7hz<vM5(08I@0zWEyW9%Sa<$^F(XB8FyOm*L%}8d1Y35%SrOI=$Vd_?R{}V'
    'VXXQkB7;DO8J^8J;2)AGac#n4t_D>e}}d+v8y9De;4|9!rTuQu)rVHe+v+kSvB-fT2@TrV%&&Bgir-?0)Qfs2OC-'
    '1>ZJw2O!qPSQBEk7{u~^#}Edg8mZ-'
    '?omf;CMg=Hn?3h)q30M10T$X^#EnzGXZmWz@MVTuWtSV8H|1p|4>fNoa@l55+~e*TEKoG(1AAG2PxCqPmGKF1HDeH*Sl(3=VEGs`'
    '^@(tN{1QvsJt?z+sjztbG7F&oMIzm!iUc5xNU8=&==gEKK@M|9|AZ%+f6hLXEPy3Q^&6Im9*$(EN0E3vL4ruwzJ@kiV{nvO0gQ?J'
    'X=RS89j@GW;ss%XGM=pCKGxTG9E|6aK7FTQUB5r^cBiM?C0bEYk{CTT%geb0X2oAh=0h);<X*GhBqO2?COO%>Vv-TjtESXX@egWk'
    '-9<DMstf_GdXR|6)f|rBm=TP8@NO+sie#ChG&>^+ZvAJd+d>BJ&*jq=LEI){?&n)Pr#N<7K(m3GY42!KVjrz?`a=}Etya17RJAJE'
    '26RX>tU60@#MRl-v#HJz6md02K*ZJD2!V4~KzA%kw`#oDv)5;0_i6I%ExMh4ztvROH8CMWLk|KZJR$GdbOUxKl6H)-'
    '3?bB_6z@iKt*0j?u>RoT9jeL&VY>*HaY#87iMamDbxshuLgxgLD|Jo@xmpXJ2<i$(Fho#fiJPkjLZUr_w65k!$F@3`Exlli8I#93'
    '?n>i69N*0IVf-6vkpF4@|J+A%9NsA*j}HEAQ)*SwMpI`U-&PX0-'
    '9`Y^7uzh67pzQk;d+$0=3vm~SFlBx3O$?9;O!0dR?dV6w=n#uJ}%EK6nFzK=`x-'
    '3lCIIB!2%f;le3js3TPtnf1XNt^c&iJJSTs@rG3E>vBADry$sStZ25jUnEC8TeU))8zBw1+>7(&qxR(+^7&g-$TvcexjM6ZY__E-'
    '~pig2%Cjn9k2{QY-'
    'b|1_6k#B4F$*lZ^Bh;sI9h!HwJI1m++8auQ#M&V!leD7Vq84l<*S)PoII0~=Gk2>(E&<v;;aEYCqP!5^QxI}oP~K=sMp8;(qGa4P'
    '$Wo&xwTye?X-'
    'fOBiGZn1!Ap|%;q67CIT%vw3$R(rlO57&$t5Xp6}~I2ETilHS}*qDtei{{ivCMQQh95UR1T!<827QEp7xt8Au3<e?hPY|%Jtg4X#'
    '`QZLA$q%-~wOK?w$1ENtKwxGSPCF9r-%(IwBb99cqi8m^~S2%^3DGnNveJg8iOo-B`Ul@>><SMLAac=P)r;m79y-'
    ')^H7c<CjFPXiIS_8G-o{*PEj?u%-'
    ')RRpILjPGRSR^gy&|a=lXMqrGk<y?I%?w~ZhqU)Ap2v@UOI_X!bExlciTYS$v8>sECGhFPhj*_NqvR6R)ysYAp#di}%SsOqAWBxI'
    '6!cz5QD=F_Gv!%-Nf+_!a&%V44y6&Z(g3-'
    '~_pv|E`8#WL+l;7Absz##T=r>{9PR&8OKU&*G&rfhm#%BJ^9+4SBin;xIC>3x!h9>6?CVAA&}DC)7Q$o&eMRg9|lh3wsi2oMEjks'
    'fS*OWal561`a;`ya`9K!jlyJHPw?Zeu^B-kO!V|7qt52(`67bb7Sr2K<=V*G8q++-'
    'wsL*G8PI4p+elFotdg4LRn(|3X22lC^stP*9X(e!(j0yRAy%N%sK@D)sX(wErqnO80C3wLY>Xc`Hroj<kYz=bH_H+OXLG<j&&mI6'
    'Bp=LP1jFz1(~8|93a{7E1@{9#*ABi5Z2yc#qvt{et>}QFK>~s-x&`&G_(lH6!LH*a$I>PMdVJHy`J2RiMv&OA9fXych-'
    '7OqQWC4)4{%nZ6554gXbSZy$u+c@1L)XX>H>-'
    'di*D@aMZZ(}i!&5xto!ai2+4BdkuR7KvNrq60|oJ$||4tm?Mv9)>cns_x~f)$P@N45PiXx}RrOcU2FtJL=ulgN*3-'
    'Bk<!s$w?B$xsCM-1-+lIq(Ss9V2+4(*FEt1%#MiPPG?7w0L;1S^OZV7`<{He0o5KZgeCy($^iy%p3n-'
    '3@v|qjZ0D;4#G+Fyi`#FqM2}anHJL?E@G<QVTrmu1m9|Mfraj3O!-!Z8`+8wyY{U6XcvS3R`b>B<mw-1Z@IAg37wBdM>8-'
    'EFcef~zg}xErtyCZjeNzwMD)yK2fP3>L@(JZOVTt@M?<i8yVWb`&Ig}W@ZH`J>W^Y?-'
    'U&u@EejCjr?CR#~X7Re3M7;f{VyD!fjde1~+|Gg_`LX0XZ^V?(muTc%nsiSzPe!L0;BHU&=d>0F$!KaOU4iM3_28ZSIn*Px8wQpP'
    '`NG{-q-sZ5%&-RJY9dx>laWtrD0gOt(no%XUc4)D_nYb9k-'
    'j~?_cjabx6_)0?v>Q$WLPY7YU8hfXT(?XG~=F={@C^w{H%`)O7dHpiF|>odTg!|voQ%RkhyG9Y!qyzuVG$_Q-'
    'gNE*fxlf<WvJF^og-f_(i_kz&Q!=e{1Edj6Cn|&6s{Z^XOqudVeJINQOui8VYDdvK?k<Ht6UYoDEV2YgW_5=wwUemohrp3Uz>3#P'
    '-R&-EX}Yq_H%_;_%Os!PeIVX}XgIgEwx=pk&aN1Xg8d`@E@N5E3WS(tPTI8VRs$+1{V8m@l2S4lA~e^7U@MVl9P02WV@S$sP-'
    '0U$BMgj6I25nhPs7i#AU#a_29AxSbADKJ5Ps>jU}-'
    'zfeX^+oC0riycKpY(OHup?L}yXwFgoC2*WiN@Ae0mQd6Y92QiM1c!A3o%c_cN3&a_R{`TNhSok%g9yEr#xq@}$ecS(RNssZpiU#!'
    '7V|SkV7FqjT2fB*dsKuI{mu9#mSFSup(wEV3-'
    'Qa04(|@CK<JEDH;PRA6n}wE;V^%9U^=#w^o^iQA)X>gK#*kx3D`=zDOb~d4eE%rL400!i5Xx%w5G)FJy<#;<c{n3%q_fADFIuXEQ'
    'v5CZpvz6(P2YKi!zCE%Wke%71=+ev+^s2HexGZlDDx==|rDsOyo=Qc2+5!>{GbOd^6q^*rDyju@q=7)Sv`D37r|?BZGA-'
    '!Us>F<Hrb&z*B9nCH}Ndd8}VuG`FDuFBoTk2m9Aj!*YB+(Twnsp5zvNTXKtDmE59lPj1n7B)8~0lUwv%$u0VBc8d-'
    'QXga5TjiaHZ8<eIf3vksniR_nf3NsCS#MqvKUf5H+eYQ|Rq0tw<%6+s}eD`Ubt1M3avRI-uN~AP{n!d{DglDUaPB=#Z&u~Bxz%z9'
    '9MWY372!asy8KZx(Hos4Z{EOB2gD&3#h0xl6$08!o;g2uf%sT|f(3|i54r(pTP<i`i5RZjv4SP64?2T+92Ba$Y;jrEP-FZ%J(V9s'
    '?tuu5BXw`mIi&Sewdjv{Ngl{*(D`?5AP<wLUg_c5J!<RNa^-'
    'i}>to@5E41A{OU+gvEGsTrDVBe`?LeHs7M}~B)#^6aNT+3KI%Y<t|PdLN1?1hJ$a4mb|@g`i$cs$^QYuN{ncxV>N{4$*zR<c>w0U'
    'EE7#%Z%oOzL0%Q*y8ERCTw>MU^0l*#Dwr(HoUR3cPH$5_<?=^!e8fi8{{Z5}H|DFQGHZ1rr*R_X$cx#bT7oK&hF~mE@wST2ZZV2U'
    '(Hgy+$$+pZ#A!bNf7J=-I@qV6YGK;q-2)g5wF6>7oBGbIc%ba)L;#b&8LcFHo+ENP~6|M_w`^`0|76Df05(JTHenG<TT-'
    '(;MUS9=@O>zMnfQ<pi@A#-'
    'cp!{cP=8TtfyFi$lACHq*oJE#um(^Mkio=LfIFWa32QXizXK*+;h(#)Z{N_Hcom%~(Y;Zm|WsH;$)BRgG7iI3Ta;Q_`Iogplu#LZ'
    'r`Wy#5;BS&i4<;=A{0y#5{o;pxUbFQ|p%DOW!_5<m@Tj>lz(a50=Z!XZ@dfO7QP!&sk3iqQX6?Up2#>^*K@>n4bw@r%&8V(XfhAy'
    'yX_XoWkA8?wG$oWosLze@AO+ZW@sWob~fJPnF&5XIZ8rJ&qM4fk2uEs3EHtHw~pKV1vVp*fK`s3$TX^Pi0w(T|;2{k4woo9A59(b'
    'ElUWXJ=vjQE`viPwC-'
    'Cx25c*4MRLs)N@9r&b%Yd1f`ty`Jd?Rm0sInHf~ICv3G-yd42!?G$fE!dg4U+flGdPVsiMdsh^OPNBITBV&G|;TZP-Th*+GY)^KU'
    '&^%|$PaoHE<*tnd_IQZSt+iUPrmu-'
    'u$+ZSlA0D>$fK$<}1(mu{)ko+MNx)y~<bb~>uZu9zpr6VAp}Pwf$h~eq>n6ti_X~G`b(1YKyEiH(aa1p&Op6?j>J>B@c1nVt^AwX'
    'usxE5x7`D6F0OeSrh^$lYqPU-Lb?3^kpjDKz(TKe&8t~=OyN;Q{C9ZOzk-?CwX%h*Oo|LKhqIa+gsb65-cV;AU$OCat0c-'
    'Z#V()#6sPs|p*vH!GkbGjIyH7GBiKW_rf{b}>v#F)a>Od#ceDba&#ydx`o_@6i;*_fln*qHF(e|)7H@dSmV-fapS4sjnhlCt$4%5'
    'LrCIs&LmHT`l6BnctGygpbzexjHC*M>!_odsrIV<d1rLaqg7S<a1-'
    'Fv4LT{$$QQ)jjvzRc^)v?iL#7*kqHGqnHx!8K{b7697^nmWv;D~^Sj{Sh5&-0-'
    'y0dl@$(zB}Dt8ODuF8+N=A!%}#p88=!BjDC?}?4_?tZX;9WFcbKdWaamf!1yO*Qu455<qwi8z%eBQ{jUMLa*y^0D5e>Y4uB+d2{k'
    ')@B&<}@s%`hjadi%QYt|VKa|S^lT1{Sn!wFNyc5uR!u^mi2HJ5&{j}_xNcx1}4?D4<)xwE}0TBZ4#>~`{wq?y4T?2>+`|MfCAx$l'
    'bZF}^0dyLvTSi-U2;(puaHIAL*FHuJ7}ERS0UDfff$H@*n-'
    'lXBd)QS%K&DafWaA0@HD)Pw;r$YglkHj*k<SDtv>=zpz}&TMsu%rmPyBUI1-x?2cc0?s1A)O)36c#h8cHFLD|Yub@*`rnCD_-'
    '&|Jdtz$!gZPs`MYq<v)6+?kF`u~oK3}Z^l>1&-txF)dyl~UvI$6M-'
    '8T=!&K5vUE;6u}OHk<5+eQsT~5ds1|l9N}rMTDvUb$i6#`d@d+v+#N8^rV1un{xYVN%B^yK(Px#rMD}Qgdhn+pIu1;c&B2<f6#0c'
    '4`vc=^!sc;Q-4vpY>(&5_Bpuh1OZt~BHi-gUh;wdvxXZ~UFm~f#~<{@Y6)+bo2u*iGv8b-'
    'O*>~9@0^v@^0*Ry3*8VmkuMQ1m%uVlOYferEl(?_^m{1#w2HEi|G`p|5>K!@McbadD2}2i1DK;jEVfnN&v{}%8PuA}pb4oAnwZL<'
    'NvRB)oXVgnsSKK$%Aje2!fD4OQHCGKVT2pI5!B<1f<aGLPrDU5?@Y8ZlQB@T*D^0Q697Q`n#Js9|KDvA;unXySQ1p9e%=uWU(sqK'
    'OHgTT@}A(@eD8c7g7b9bz5ucL_ryxJ<~4Rt;MBz0xSI!Iet#u~FQQb9HPVZFv^7rawcmpi7;5A<;Y0?@`CT|EPOi1zhEWpje(m?+'
    'lsLoIej`SU?gzEsiScpF)mbO`TrJa!c4Sb!+^_eEejlZxh1xjXoB0m<L%vD}3IuzW^%rK{T{M#=j3A)8qy(KlUO%V`b~8ZcD2pv!'
    ')Q+prS%`=7c61ivsk{T7g?KFQLT4d{rB<8KZEd34PPG1bvuuM6)LUZpT7SS<gun(ODP_3UAMtp2r&)i-'
    'kEsb{TP#w(mHs#&gOX$1r3s42w7=X!rW;6EhA0k(aC?M}MkV(g%ruVn6bG)`(0_@M>rQmCVr04-'
    'y_LKF_asyjf&;i3mS2CcMv`nc_|&aFr{nJ5lSLSUWZ8lkjbHQ?$!82-Xa6g+<O)0>p5g-'
    '%d}9%Tn&TKe`m!^?h!{0uPZRt>Qts=2<xR=%8o;I7XNh5XHV(9K8e|C!XGs!(mgi3e-'
    'LKU~GZSjuHf!E4t;DV6urfF+fBfDVoQbRZZ@qCK&9E@n<{xyUCW<R!r*hZVOzNpn%HbpEaOCh2RGc|{WPb~4dqVD@2+ff@2U-'
    'xe6XfwA3vHVOc|6!c-)5+A?vDrZM?@ZmD~4-E={SZNkHDyiu(Fhs|2dCX`$fq4_WtB8&k^U<`f9z;7@}ZZ^|HSKqQY7As!vj-);-'
    'ZMcZNn=@qi*R@)=;7{BIQx?pC0TGPpbn-bKVI2rMHNqfDdV-AAlwP>fatI<1kkT16iYQg)iV=JWG`CZG<4%A6iMU(=}`ZWC3|{=k'
    'MVFeD0(H_xS>DrRJBV6g&Bb&E-'
    '20%#s}THc!T6;lHim;i~FzI<Qi&0z!{pB8_EFSaj<%lb<7iobMa6C7oD=oa38&_HD{ZMqK`_s3j=pfc{s-'
    '1h3j21<l!(>Vh*!nEl=V%$@?Cc#Gy6bh3B^|8Q6<$fAR4Jc!?iBLR9{wigE7&N+Kb@-Rs&|{H~iW0~p`qL7vCd+vCg-'
    '6GKK95IysBA46#4L^{)!1{&J!Y7%ysGe==tFTGu~$CzI$61uDf#}X@kL_>(&EGpE&0u}3GNmuVL`W6QarhB<6qud!Mfq)xMtfbc#'
    'yxE923?lw^?e|UP(RJA}ryFH(T-'
    'xqkY&n;4&==o|p>KM88c$zFUpZyQ!meDYxcCJut1x3uqS1OGCtNIz4H}t$@01w&QNpph%=0cawJK=R59Z?Jmf7+$|uGXFF~sthll'
    'rcPk_rSOo(O@@?WzemD&c;yB?*uw*j$_V(C5(L3Ql^eSp1c!t0MSyJ+JvHQAH#nGQQaIH9NT^wKhX6qCw(L4q3!p0k-'
    '3zh#&bbeSSIzK!Uo!>JPoga~j&X3GQ=SO9t^P{8aJYCf21Vzw)^*mZXY*@6pFR!h8G43!0+7>cBIS=aq<qpyWDhmD;Vb4GAHh`Go'
    'GCfn|Cg>GxNWFrWrQls`s&zR$bbRW-wW=~=O;BaTnxe`G1oE#$9DjE=ft?cGZ2^5-'
    '<}9Jbu&a%%B}nNavbJ6DPn0I4``3|q*>@!%Uwb?hmx`5rS0Nb!3>FE6gyinJvc0o_X>iZh5t^lbc}xCtrP0xQ76UWMpu)$%?pm&`'
    '^QLip!p|!AC>;a{O&2t>amwu=?HV8B-'
    'mfz+l%2RFM?=u<8xLKx7R?TN8nct%&z3M(79j9ea90jM;P4!mX%;XD3Ci}?4YvuR2w5{3SzS(`F}f9=`@g(hcbHbi_1>9#KUgfdV'
    'k1qL{t2QhQkIU2N^&!5I<Cf`2!3?zYcy)A*fBB2mPndesY}Na#ZELaXe=0wSw&zK6_QvG6@TZ<%$fFmuy^?5_w0Rq_JWst@63D8o'
    'H^xv1BS9zUWpF9DAxo-qyl@Gx2Oji_9yRBg5LP}hgu)MxApP+SRel|>*M#eKK|j>$3Mc-B<_Q%u)0jwZt#k<X_eJ-'
    'L_#q~N|RWAe`*-SnAB>_gml(1`>nWP_t|N{)f%aG-StT1rkBkP-'
    'U85yRc18`0b!QX<#s`jWDd6?itrrral0y!Znu3G!%GKINuS(?QGj$YZXDmF?rPX9ZYK5`*esG=z|G=2)b9%$#4RRDYB-DbK%rt&P'
    '`)>6rplsmjhy+Mm@rg&b5`OJ{|-'
    'ZW3pp$!Pz}Owb#6gB@9xkyy8bTg|F&oXV^vE)uiP8+9JC7_g^TAp6J8;nD~t~l2@^(ziSY_!!G_47BPNNIQ=yB<Wk}sFkI&0Mx;1'
    'w%%2@g##_K&W)v1#Pq+Pf+@k&+j^Q+-'
    'jud&$%&YP_ka4naGekPMvuexyc*;om*S~uB|HEFZ8sd5itgLEy%nzU%W7d{1uG#iDZR_?&8#^yO}mq#6awuiz#_G|P__M@6|+{b='
    '{zRCW?iPRVNvEQO^av&M1BVZr<9g2GFWqdg)C(4Zg(4pAEVtBOy+;?)xE#Yj}v(<!($AT6?6u!j<;c8kCTk{DDOS@j@(A=2q=qfp'
    'dZfi`n6x0ox!s?LcF5Hm9zw?mXuC2}<xzdY%R9jE>Ui`ka)OGe}{Jc7H3I}zha1eZr`xxaQ>=OJP7jJ2Vzyk7nE;iec^=(}AV<u#'
    'MJ2ylfhhZol_M&D*`89|2+$<@-;gYgknyoi?(w-_2;jD(Xky7p`S38^l+z=ojkx+>R3o}gA(Q09a$v@sp|K10Sl}=@2a&!*Y-'
    'N}70_WWB*PX?QXKUgxg4GanD-'
    '?8NNcX9EqKHJ2Q>RS`7ndUlG>Tr*teSa8ZbQ;)?V2n;9`!S44eXFcXrW!qUq^FZedu!)mvuZc`E%iS*k*sUgF)8_Idcjs1GOcDX{'
    'Og=SlS-'
    'U&&{m;bo+qAZmNZG1Mf&mhvC^m7$Ee5S$4Z}SU!xt5A1i&TN9X`7I2RoZ@k{n7Ml1D4{y0V}^{4#=+UoC=<%Phu)MwRtfHu3`rY2'
    'H_gu?Jce8TWUbi(jMY{KwEWWw-6T*B}}RKoCcrQyTsVCqTV!g@$bH*UtFDoQuLi^bSV5NL0wqivn_^31Z_;$-'
    'e+lZ}p|Ni|w)(N`;9!4jh<;smwsNzD`AP)Q9lFJSSu7d9&z)_-NpDacAIovJ5rOxY@%T&zc;#?@}twH0HrRB-'
    'z?mOD}{@;g|?Pg&P3STm}}5&ppzTL=4^t%H5t*1@i~b+B*PI@k@i4tArhgWcry?n+G<Z5gY}wtlPqUYB<Q+8Eel9ZoHCGI&{Mb9w'
    '`9I&=^v228cwYqnbMbz3dB-d4-'
    'KVXNgf*lM|rwpwnJs+M!}z17T8Dzs3*9{X@OZSJciPR1J0G&>dlvI<0seN1rD62ld4Mj~xwnW*4uzq#z)aCGVrJhm3;NO7nW<%Io'
    'gZzsqJTiQNOj1zXXhdCim*x2@UBAl?FJ%R$7GOc=~39FEa!3^tBHc#E(?mY)MUq%b6fa?BW#eGAtpUI`<gl(Lqo*QC<R279SOm+`'
    'b*|bYg+hStL6TDps)(bXbg_s_;Ey2EJR00EM)^B$w2mtF`&J*V4&hVb5J7Y}ZHt|&_CYtR<qfbf1Ij*)`Ln6*~io`I5e}SttcX}c'
    '&B&sA`*ILEh(rRX#@!;UxEwoO!TQjT!SW@7rn~q6L&8CX(mU%_=a%;ziAekw7@FQqrv(*bVtioy-'
    'EqDUX$_-r6mZ^syi%cp`OySf-oadClAuq;7RMsX#XtX%rjogM`qspYO5W`tE54lW95k!2|2<<LFTGMY)@zP02up~*MD}R#xg|kJv'
    'W)w{t$4tXz9dqR<#@WvsKVbx&J6=-YOODqt<k{8Z^&c-HUBguXdXy;?&ss?p12u$-'
    'Zs%t+g<4^i!O<D(XwkVPvz_4ej$yi@ZnWne+=Eg92O7undJf*68UND~alTW72o9MpoK&Z`axk41s<&}4pBAaNb1<QHF7M!AM(scK'
    'YYwK=!uU5F%&83?-pRqF>Um#?45{BPbF!K^yz^SJcK`K2cP|zb^`xBYHQcz-'
    '_oH^;;p9?ZM=iSzQV7g=9dBB%<1OoTY_?v<+t%xN$9f&_TCZb^M@}fW3Nn&&w|ItmZC-BcX+2j8bXp-'
    '^Rt>DC*{wI3j#=C&Uds}}TCQ^<o*%){dv;gL{{?R7Du=U8b`3Y0Zn`Rt(Nn9IH`X@m88g>P9=?fkG{J1hms|v!;m`p+**c(KvkvI'
    'jtpoZE>wtdKI-uXO4(PY71Nt4eS109e7es-(HT~`NNKVQoi7rW*ILIx&r}bVAI<-1$6JYHj(NnB3a0-'
    ')JO`*FPoQ2{AwX!u+u#rL*(nOO4k229D!lO(y$?&Kbk|H~b55ZKtp`?pb`)A7EP|$_y-'
    '%!tm3gA%Ag(~1s&4o(fP|QWwz<D0&Exu>*{{KVQK_9h{kxY@AXc8e(6HPKiYUz?fL}s|>i{}&;7l`M}KDbJ|)G}|rMUAaYJFjy(b'
    'z0G`TcsHlFK(f7zz-@z$lW;TXnLX*J}7Q&32wRbRJc)U3Ys`A92+FV#wo;q+|>{V!-'
    'x}aFpT(V#DCm_JaA;#AOc|+@$XTHa7;ph*2TEME~Z#47~mBY3WUYet&srRaVjU7;qUTt)jITIWwECmpbvK<FCgb)ot38=rE|RCez'
    'j5<Y7G9qPLtNk!bqoXA10=OQ*QD!Pe!ar`=kaLt1^7i{VZSfQI;>dzvYV_VELj4TE6H(RC2lm9gigxMg_-XDTPtN@mNM-RB$|&`('
    'M-'
    '*Sp4c)o^APpQ7WQYqz<<x4^gWbIi%<+&EZb070UunSM>p_#P_(LJDq8p`(FoF(`2L8gDXZV^Vu8V2be{41A3r3{In6gPa>ktHler'
    'nxa*3pX09vx5OWRpqs(<aAJ>VBBc<rYDS=D{!x3|B&__E9Lp9l~JZ>BCYdzVN(qSWcLc}95N>rcIf<<*DCzgoy`BYGGiZ@_yVJPT'
    'b(u}*0iG`-F{(fXRCT+k!GqJ-oAUvQ1Pz{65C^wNd38@$yH);y*f*i+bI>Xju>l@8LnmbLIC7DouLOblof?fo<D`ax3LED?{I}1j'
    'sG>fU$Y%<lFxRx2FM74h%OA%1ZEorKr@J|n?E{uDO@aq<g13lm>vK`|_)2A}*we=L(j#JTf8F8TAY{_5pLy9<1m$u|D`7y;79;Yw'
    '$LDkakXBF!Fy1$SAA3{%F*v$#Bl}FWV_+XU{Ijy!e6n&6FY&A8>m6*}TKr8GJf1@d`Hnq>LU@c%Tj77u$*g>a3vpw{d<P-'
    '1f!p`E_px+W}bzy7LC|5~M!-'
    'Gg83)Bk0DVZ)CEDsFw>V}f0i>0LLYAI>DSxTDjroR%#*|5{}wCps!Y@?;KWvd+E)KSe~FVV<lSt5NTpN63md|aoC3!d0%;(~{Edb'
    'r@Zofa;5bf<$0p5AHTf){Z5x8NNPF=?K4R+MFkB(NvVkdaadVI(t!uGP%Vm8)1~opm`}_Hu{BN=YbRVK?HKWTw<l6W4KO<4xV0-'
    '7B4`2eSwL^}SJsPu!JkM-'
    'V=#=H)x<U5WjnR!22A`L9~;Z;x=RQ{pzQ;0DOBtwTQM1K9%s<vKW$q$K>3bhi(+Od;S_9%hMez^&|SiEqHIB>m=pOXpkbH)EXgC8'
    'm?>QoH39E1WYziaU(ZbJT%h*!dLGK+(~taH4j04`N*W?Ej;a7rc~ix|u@~Il)e&w<S=4okpL;h;hG+v1zvoqs}eGtHN-jNllKH8j'
    'Z$zE~+i6Dw?a4NsXDtw`~e?<Eye&a?<X}?(wmu7-ifwzpA$zC~4-hoQ%-'
    'ou4RtPr;+b=F*@JDYN<J~&Uu{svis!Y+@Ia=x{%oBL_YUo(S?)Ky+m~7<aI9>J*Z0>T;`spQAReDX&>Y&EQHqu+t*<6yDs9m7K`6'
    '?k@j_1{I2UcuE*kcU8H>j7QcVT+jrE_zJv94*RjQ7U*`!snI~L|Wh19jOXERA*Mgx;<HL1<t97>-z-<Zj<#x-'
    'KQb46UIJn1~8mHw1v!3wMzu^SAm7-'
    '^IevW+?(xcaN+Z<7{k=@nSHNlOxM3m3Hx=m)sFwZu52K)*B%?wEZR}jw?uDnT|tWhWOE91RG`A^!P75xG{%K4(F)?{H-'
    '($QSwaksdPx(JgR1CwQ9e@<Jk<yfL)pmCsqMvXLSC8Inxm}@p-6doJGH6Jl@*$?HKkr-7PhH=eFjEZE#xn?Csk-'
    '`yNGw!1H<4BI2yMZ4K-xN7O<m)zJ8<ET*%61ZeEdS$<<z{Crlir~R-W+g}c)=sv^<&D(NYM!~Fp7D)E!d_-'
    '9L<ZtHcX<7!o9@Nyc9mwDU?yTpE#PA!y+(^vOW(G`|wIwxn`Jtwml82?A{EMT;DZ{CF3zDZrUy<s!n~cw^yciWi6o(x$FRElwNX1'
    'DeWCuW`H20W+ADG!{j(<en7Rw3s?o~teLbzU|0dD>C2FSZ86UtGF*~Nc=oViI9$rJM+_m@GM+tZNG6x_>@h=qxWer0PGDjs&z>-'
    'iPkR|+Uf9N+Zn`s9GT3So&&64?s%#nUuu7?TzaaMPbY=MEe&nL<b!Qw4y(7yGkZz*NP`JrvvPN7kRvV|Zmgy}YVGakrL;=IQ33HR'
    'PxrDhXV6dl9USvLDZW@?tjk)Pyt*2A6WRb+2Li@wWn&|9z9z}*SXTS3pQtmn7@F$RiOYx_8j%uC%^<ons&kiUXM)+vocZczP8pd|'
    's5?W;b{;x!W%HV{JVxH&alGT&|hg)@7Yy<o@iu>QLt)WBT#WvvewhefFYy;k5wgIoNZNNL+HsGm(12Va|p4+|iLr7fc?42J$9yn+'
    '3{5TfPpquUOisd+x*BXAucHlm82Clugz}A}nCR94^O_c;$PgLeID#k|`UMko&k2Ms{i<vmZ&=oIXqQUe%FJ<CX)AziLiPH>S@p2}'
    '<YiOO%V0bKMrj`6m#W&(>ADc3TI-'
    '7Pxj2jk{CaMMNDuuVix`Z;(mGnAWFqnf*SxecV;j&;y?HOm(s=ZO`f_hs&Bu(f(%xXd(#;hiEUn)Duw$cMOWbbDKX#M;!8$^DKqm'
    'wWRKiNPs0l&xT<iHr&2<nJxH7as)yJeypjofay8!+^}&SYMirPsSvtJzTY-tosqA%}`THZQqjv&I>l4!wQB&?dLUv)a_&sXbn8nP'
    'Y-}gZ0P9N>8Y)8LEg8pnz1ekMg)qCbIa!#AcXHMwjF3q;N2lDvYzK=x}_KDCCDzv2ivX-HmS(oBT+_9$Cdu&UtlW#K`}Wh=&p*1p'
    'ibb9?@s*n0Pu7k5i5m%-'
    'PIJ<_yO0Gj|L>qcPm0w<}9nYZp$*4k<^?l;%~NQh<`XlW9Q`9(Nrs=_VqRy^_><I*<cjonf?wZ_ZF!!>4OFt>Mcxq}K4^8dhsq9u'
    '2KEtd7L4)Xa<nEI;~T=A&PMo#M`NI+B~LA}pX+a?x%N7%I0jT2?vE=CT;M%&^%Z{Est)dwL7&R-'
    '_}a6xgPQsm_R^sTj1`Pa#vd9la)84OmdoP|N+sfiB!{9O%mZR;O;<Z*}U<{Z^+Q+;24K$^BNRUfeVuNMk$LHMT=sV>{Fs+vV7zA2'
    'PWV6=}8f5=^+PGZ$B=Ox|psomb{Af8;{-b!VLS>g{(D;G}OQu7YaGcV(TEVui+enei)8ak@l}re<`?7-'
    'Y|7`S)IJ5IhN^5r@+EzK22Z6m*OG(D%NVLGU#6ko(g2-'
    'q9d<23CaWOdV(9$wWM?zv=c&{5=s*kmvvxlM+%t6a6{rYCXrD{Lxxj!L3Su&M6zt&F*kM;|%BCy}dXSwof>3ZlYmQKeeoo7uOA^a'
    '972+Ns)^wV@g#zHelgEx)Z}RtA*@-s8;Pj_5g+{RtwPsF~31YIxjJCv0;#Xm5Iv?hr(Qj$97xHdo>rgaYx{ctT6U<P-'
    'isOtlMJXk+L(|jP6)zPE~U5$5e3Av9}*<+Db#zCHs$-8wYRgzglb@ytV&nsd4bu-qS+k;H|x{WyZl<`+#Bty<mCnWnQ}meFo>t-'
    'xpIr&X>PGCV`wU`ydq5Q(mw2WNgipbnhJrEYI+i=!4k`zm7G$kE6)Rcilgj@G9;$+sLBKA9Bt5C4Ne+jr;W8X_KC_%Wbw?;zi8s*'
    '~O)X*|-=+@>hrH5=UOhQOiMRV5@Cv|Bs14UT)Ud1q=_;09eTIAZq$W3=d-'
    '1MiMD31~K){H*j$$Rfuyu5}ikXgX56~jiB<QDjTo5B-=-'
    '2Ta)a3Jz6!r^U2XQ8_|hkVp#%ej3DWlL}eLVYT5jXO^!^HqdJ6)0ROd&0ROX$0Pk5wfcGsUzz3EQ;6vZW;(o{Ow~5F;(MgnXfFAn'
    'J_U--uQ77>-6PK7~Fe$)C{V#uJx;!0E{)YW3vIyS&xJvHcx#aAfT_Wj{+s)SGXw-'
    '|wMCEaU>~?+m0L@O2@f=y^5^*IBP$wK9WAM+xtKTh%G4S=tldML}VR`i2g7C>|H=NjHWGP{htx2?d;v^`fAgkF_Q~cx-'
    '42By*@qaS$ebZL-'
    '3KN$aKI>x{67pn;=|Z@Qi(73M!nIu7VfzlKxK4>7*!BW!O(N0@Y)v9MkUxvzM0$alpR=)_%VTrIB3Y~{O2m)JWbIczrvEI6lg*fp'
    '7bhoIR@7s6Bm8~vb6l;3BP9TAXdVBG=uBkF?~1V1xt5m)q}Iv!%gH9u@&nVeI11T3bz8lRRb8BjFDE8g_5DPAB{2c2OB3<c!~~@-'
    'OT>x#Y#zgdv^v%!gP6G8Hh^5m#jiOP1P*mbR)$HE6%+qo4M>p*fG5NvQe@k|w@<5$5kQl7P@Hp4;-O>{4_3!X7o-'
    ';{w!JE>>8Z6fDqU=iN>^K>(#_VWbhkAsJ#39iPg|qXOV_BVRrMDP{1eAHxrxET>#VvrGw@Fw|M^z{GZx048l}JKM~-'
    '>+khoio*27|M;yP~<YG;Yrj9@oGRMBuA;BVS+0I#q$K(z$9X19a3*?#H{QPRnpD9Q89X5)J0I2|`K@H-'
    'r*<L3<AH^+DTB?EVj`de>77hB68-'
    'paB`+DwbvST<SPX>mKtrf5Sg?qJzeZK=htSvF0ZYVjMEP1m+s+{v;TsyV>jTGW=0<|AUB8qLSVg0j&RtteDO!#Po`qYwT-ca_1wR'
    'vo##x|XJkp|9%4v`>fZRYx)H)gcE-f2RF9Y~volv}cFRAOo59?XaPH5R=}W3J?Dv5h5`m;mZ?ol3`G`ZB&>_9FOhFKIMnxoN-'
    'd>wK(#6GbB6oYtWH09{D<EIAK~%paajgn$3G)(`PgIyJ~lI9~JY}=sqqMCgDZ{w#Og1S&k2ApK(yQZEKMO+}V|CGu7E!(fM=a<wl'
    't^Og6zDV>PWyt?1wR$t<MyL+DxYj}^@7?qMOmt|=3DGdxI7Wj@1$WIS59<c`-K-'
    'D?~Vrq5pIcyN8Tf#XroXB#;l?ey6ujz@b}ZJhQf!(=3Xe>d;`HY|y9WVe8iT4_i8fx9C%L)+Qk!r?r_;9h}sUhW>Q7TwQkM!^RhV'
    'pAPiEla8NX;(B-Fe6tm<9u61FJy;p;h`MH66T1-En|*8+)C!?!%cyor8bPH+sBcElo>UZ`<U$uL{`5CiQyj062!NdsT(-'
    'MFw<gf$C-'
    'wh73&5jDxz95vaNu8IkNA<MzL&UUvfwG8E0fax65sdX=G?}eC}#ymUTIDP1X~TQ~{#Nt>1wStT;5$f<ZErLNa?8LQ*jx{QdpxeXt'
    'bj_YblUOxhTuwTn#^1a;|SU-s9TA?_64aC)@A$xLy~IK%UcEm0GY{=f^uHeynoM({20-'
    'Q);v21mJ4a`OW>uxLi`^S$%(&zGEpa5JiaJ63J0i7RN8DO)$}D0-W%6%Fnc3C7{n8~!kXJ-'
    'G^OiytMhDOZtg@#6$`<@40<KS^L)K41O*(*y=x(Q1pIB{1kJ{4K6bU}wI_sMc&X`ptjEE_~A}`!sgqo95WRV^6zmjx|0fcxpe<eW'
    'povLD`c3F}x?OQ5V8^=;8cA<?I~_OHfEl7Kf}x4)`WVt>6k%P}Yof6wTxutn%1_I>Gs#x@&MRw^(y7c0Cib&pd2;CN``2*z>IEtg'
    '?mJ<lK=_d(aWgu>N3rQ>(x?;J@`>VmcOoVR%r(@sr37K#kYge~|uM`}Y4uc4Qq?{{ZRF1A(AAwV|AoxDi`9erHS3K=u1avbREx<r'
    'h6~bd>HHZgGvC`E=LFuX>${bdO^$gZCPz-'
    '%z$DIaGYKEU#rAefcxYqc4A!dGzJ~U><$>bIhYJf1Y{t<u5RgzWhaI`@D+G*r>a)?Vos>7GP_-CTGIgK&jneJ2#F=YH}3;r~Vso`'
    'HoQp9N^oYz{^~#w^Z}AI+@6&+bYNUOF`DJ#$^4vpi@a+Iy#i(rlYTI@>=^ld94FXUaKei+a^nR5ThhHmWm~s5Kz`87-'
    '<C<kgX<10Auie7^_wcGRBgSb6o^C(XmgmVze{STsulXvKMu>%lBZE^6I&YZkc8#*glL_g9)DVapi+~1`NKz?Kau?B**e#*9L&&E1'
    '_yD9yytWT3o25W30BJ_}!mu;(4<<wKvN09o_e&A8j`7<>jN@jHTFa5#V<d9JGHKUI`37(IO?LnUvOtNc2v%r1)I7Fv|SE(tRtV)D'
    'IHwG0Odbi+ekx<PWkMGs^xTlPII~53)lu%KyN2{!SJaCzmAeSxaaM#1_=tAbd_hc>_JBhXmefR)>3{&L$_=#2ii;EEQ<p37@}`ha'
    'zVccP7ouk=#`9S%j6R@+&neI<A@wisia;GAJ?Z>dBzUuq!BoGQ+N-49e@ek}@bf>}txO;O-2Q*a2JqKk_-SY4$&ys5H0g9!vuZ*a'
    '04`U;$g<V}lkI%%Vs-'
    '*tNWD+}}Y4AcFEf@V<K3YdsUJigbwkR0^bUt6frSw*aNA1iN6HT{EV*5qCeXYpOZ#rA@ZudpA~`I^#PR>w=x}orfix&iKy9np9_e'
    '7hnUqGrkM4O4TvKhoa!t=Nv_!o|Lo5Mq*a&31(B$2{r4yhw>gi*&V_M+5G}5d>&JsLdYYhI&}QeeiIw=XtcM(u<Ft7(mSZeE>c@J'
    'sTOthRg1d$szu#>)uJBmGIRurhaEz#IzM1w)6~iT#1tJ=ZGfq5q&kF);YSx%UJ5_DyzesjIo$?UyJB_@P;rI+?#7{50e*<%m?iu%'
    '@eJ8ISW8v&jg*+jIBSao;&Yg6L@6Q5o^0lE2V8DNu)M#_ix6)stsFI~Getd_j;>yCx*w*rj$!0#HR1-'
    'E8o@4R@C8xF)At#yrquFu34<?+dY&$2uryG^*kue%Fm;Sw&S2eDBlUgw(Z#5jz>ls){Q>;w>W<6d$0dZgfXSw^5|Rn()+LjxM)yV'
    's3FP8Oqu2yykd#1H8IckUYM1<=c39*hlRC1$A{fNYu+JlCit%7EXcW77qgaQ_743bF!mQx2`lLVRx<-'
    '>oAm#^YTQm@pgS0OigekBS7y)Ay(|#S&THb>M>(r-pFB7ku)~x$cXoEViVC%Wb=w7W>wasR0)a-1H@Dm-'
    'YhW1=0d*^JRz0qN$LZ->pHciuLWDEc$F8e|&qE}tfnW-FH@PWUJ5*P%<J>lQggV^2QY~d2-'
    '_@(f*l<X&=Bd8_IU8QRKYMf9@BL)#_Sva+g)*{h_)ngcLwXW2M4Z(Yy6je`4@Z=T4JOwMvp_UNo^n@svvVG+?K%eY~(iNNa$s_Rd'
    'vSq^9rA&jC9Wb!(jL4RAb1;LWY@hY1Bw|&+pDX5&sqE3)kfe3OVLXsB>(y6Qpyel*q25%+tir7bY{81QSgd49q#B7$Q5wYmub$Ml'
    '1RJ!#MF6#N;QN@h<P$N9zSo{!?73vfece2Wg0G*$b?RO44{Y8E|Dbaj)auHhhAdtCB&7TNy+nM-iMYE4?O3Y@&Hi#Xu1_9@pMRQ8'
    '@JpD66Go~oWdzO@46H5jfur=m(ae&7lSyBntv9x4n_6Vbx_eaHj6aTNa<R?C@DIqU^f31FV$O^qlk6l@400}`!u5n9hpaAKFEAvG'
    'oo1>;E@G6N9gH<Wx;Zo(h6O*5T4FNz5)ad>$0YGNz3J+cZHJXaZ>v9sl|+}SeFn?Nck<{*%kA)33wA}*>C{X%yKJ65n}%&#Q4LZ4'
    'r}>#ou&Lko5LIO4ir$b_N%CV|_@6Rl)}3lf;x1&W5C8$%+3Yx{gB_q<;|y_#sV#j31MF1C^aN%JDQvkR+hCTNkHB%!5s6R0anW@o'
    'JHT=I_W6~vfems9uArSz4Wo+QCetOV!h~au@=~L>O@xJe19Liid|aJ{RqHfLJH#0JJ`e3(#<yB!tW&d-'
    'Xv28!9P{xOtt>?&4%s^MAr}Mb%xnnOn?0-'
    'nHCcS#?A)h_FPNSCRPjZ#bDt)@WQedH!+4`E!<B97HO!Q0h<%wUH2`nQ)Ea;rV9E+$tyjN)3Lek5eDCfxHCnQXEr3)9af^Y}-'
    'lC4IeQ!THLGy(f#Kks)xWr}<m)Z>CGMhnMZZn80YzA?q6P36X6Aq?g0$dqWG69~9DU<-'
    ';#f8!~saeTv*+(74hZ_m_HOx#_A1T;pT1M;O^13p!S}V12oTLu=FfEMS!%gMuWJfX6jNEzIRy}Ziyl^A}U3-'
    'pIh_)g*=}oTXX!7lWlbfsyn78r0Z$XFLkfh#*bzO#R_jb%wo3{9i8AW`SFz~L)-'
    'Y;jc8=>&e!<Z+aG{vKsfYEM>R5u}q^9^tX$;I55Z8T0LN;93{irSuw&ZE=i7FJ-OnO8%0pjbm-'
    'J1qP%tBfWwQqs~lCw&8BMmt*v5y!B~twWcF#v`kRRq+V@%$;O7Ri7LnD_8W%figg?PY#mtX&bA(19M@9!1Xs+6=kS?@5HJoL!@#!'
    'Q_F7yR<(~a9C#05eukEqy)k}sfje>+A|a<7IoT%|fUR1rGMj`Sf_2v;*VnWBdt*olw&LnLd)>-'
    'MzhO0_H;`uZ&h$<4Dw*J~)fmRoD(Klh7<jtK`%YW4)tV0-`7g)miEL2GOl?rAU?#Px?F>JLQ@1tHb<FBB*$3{XwQ=9Sp3%Pljq-'
    '1FudJR%?Z?b~qlbInD1Q4o(+2nGoK@-X)iMip1~}H+@=k-g+OXC_ZFZ>`47XpZ*sB4sY81X-'
    'vkKqqt*rM(E8xA!2zY~a9*ho<I@tq5gXij#Jux(Ro<7+NLxbn*la3e-ypV-'
    '{Yj#eyjlR9}MUi}mBfPN~h2e#D`X3qtb4FFe_)WH`x2}fqFm;DDaB4oh&dT|p;B8u|+7S1wR2zp|Rw|A&dOC_DVXWmyIK^@#G+2&'
    '=Q!PirX_h15yOty2bjy+Oy|O{LAgiH4xCo4?9)vn?t9>NK-~Xc>Xm~%3f_<|N<mPnB_8<pwzidx(5cdbTY3JvFY;Owt9+-'
    '8cMuvm3eJB8YYBq&7aNo;jkh`u0g{+H-'
    '!)}&?YB(|3!On0xmbMo)wi>dA{`6#=j<U_m6|e@;7x5#Owdw&l(i;E_B2d91h7GVxQEjd+rKSCF@a}#_Lu85%_qD}`oow;pezy2<'
    'e_MQbfGs{e&=wya6p~$z0-'
    'L|iVz62p9=NOa+{U=Rb!T{OGhP5jy?pv(@DrAeqA~a>%f{$2=;ygSu&1}gJ{MbJpYK~@pG(N)=?fm$PicF3Il0JV!BprcKP8()5o'
    'B--NsQ^VY$_=*eK(s%3QVVG({ab;Hpf@l7laXT0a0x|hZ^`Xf72OG%to@d5)!g;VynqZ{R~`iJqky8_5fE^?X%ZV^%-'
    'JsjNRIfu~ejHPU^{<wx(c|<zPBGJCa7@xNHEu_X*h``V6Y=M0dzo)<K)UjAwgj<Ci7bp4#+fX||VMQ?o+0x3+;<yR4&Lma_KQK4o'
    'uzrWkA9{tAw(gnaw{-X6Znbo$GoHsL7v#<Ch~@;ckFyX8-'
    'WF<9rm|1nuVn)hR~qv+coo%JUO9G?wTJ8i^Xy~W;Oi@hNhdqXX$)G$jbHQbU)jj*IrBV}uk|Ndu)qVXrLWM=a{pltT5)W3r7e}y%'
    'P>j((7cAHe^IbG72ud_ADFWH*pf7+Vlmu*e*E4C*2U$!RsRa=w%@3Ig1$J|(R@e`hP)-'
    'N&8GiFuz!0KONXv5X79BsJzH9;G$eofJat6vLf!_}|t(1xpD+oKJqk5eL5PJa;6{UJAQuJ{>8gDanF5ImRvKmN%1AAibaB<;MP-'
    '82E8Ty)$6MxXhmA`Y`j3EKtD%G-jJ!=^&6mPztT9ll(}+@w8RxW?^4!*D^Bsi<QN4(cD_3=XPHP8l3jncR-Sp;om9jIS`mLIN9Lx'
    'JE747;H%yDOVz;%1PXm<-#G)u4R|rgl^p&JTP$BG(=UqEGxiqeppL)pe)-'
    '~zgNTm;7O}2mNN75y=t@S;<bBBoiSs|1v9_bIOE)#<o%oQrs6~{1Q%cBBIDvKTzr{}G4Lb^@$>06RmE%f(ifk1{)}@c+ypmnk`MV'
    'iCjP+0UEqgu@pUe~$;G$Bi&=5<vLW*M;1YO^<pptXK{OY{l7d)V5Q_?8X~A3qPuih4dGQd{J3LC(Bkuee8C|nS@!CE01CF0D^I|!'
    'K>vr{k=E^;blkd|*ylz*ICoh-'
    ')*N77h5~bIj0=~ry*piUK&M8}QWzbUN>d~~=v$_P4sPWsNb+~N(Kn+K5b~YSBRZ5M*!)c8bY+zXFW&|5JR=OGefdp$pjs8H26@^B'
    'BpnxS^Mt`6k*2EEmFvblJQnC*u;zf8p8fnzjnZoBYAz&Bp1h8UT!9i?&h}b1QVkHjbo$t<%R0%e`&)u;At;b6u0IG)JrJMj&FndN'
    '`5P{3Z*ut3$m)D25oWQdWBrHEIh+!REj&CXCUbLOvC)2L1MA>vCQa6!!z}h%R`$+~00|{D*UjxS@rA#5gv7eM$Bsd-'
    '=r4|W}{iW0*!Et~TG$c3<l!Ar?$3ariaFC-TM{qm_^QkTzpAF%-'
    'WM^>P>cCNaQV8+5v=Sci)^lU=2sfG=iAQ{%xlw>r1M$x&Kt3xkhydgg>><tt$ftckb`mEi*?0-ai9FlkmJB`2L?ICy3Ti$-'
    '0LXK5_CDOto~@K6OLVq`$=Ygt_IIoUFlTEx_HEPIkVJ4(9Ee(*1fF9tbL@iWA0c>_TJRWAp(buvq8n6s&sZZ)Sx3r?)bVPI*vFud'
    'ZPut$=CBAz+BH>Q=>zjE7nmQG0drX;VB%}div=d!XkIKZKa|%*08<U}%Td7mhrA#Hm`kylJr^)L?im2Hq+3!`@ixmAy2v~=K}{(_'
    '(bxo46zOaSvPhY;PmnpvoPCVcQReI;q>D0V+mJ8HoNeM_l{u5ruLRA}SjXT(^SKb3%PK%~6?TwPTdt`!^*NH!qgcf}aldUwFLbnQ'
    'D+@yf5{R$4sN7P9%H@?%iElJN7L{<L`LU?nBCm-+r5fK?qEPvqydVOV%aBqbpI^CWuzvr9iG|9qe6Jur^J*-'
    '?20myZRG;9K>pjchL@?9yJ~R=D=~x|>h}3jb4o^hE^aYMcL_5<OKQa;RO=Tu(v=SbV!%`X-'
    'kI#pATwaDpNl|3Vf{J3BEgr$VVk0M(1vH}CE_Oxcht6En(_y{rxFp*u+ZF>nwzznFvJ8(aD&Y~|XhAF<;YJH$@%W^?CIXLYeE$`N'
    '$LHk*5qMmVlpgu~$q<h}=3=A5<9MEJywamWE$MJLPO#IXSwdHFEag{<QY2N8_-`|nM_qA-'
    't5gw(6EDGaR5K2B5l3>O9O^3iapD~6CXV8|Gb8IR`g7fxk@XM*xbDozdWwOZh>3cML0orcNSRy$_VHL2<^ubL5ZEhr=0HB}BJhba'
    '1g@-'
    'vK)m(BSOmh27RDm*33*Kf0@Xmg8il|Y<OLB3T!ADq`5ed)f#BXftq|B+5W}4<jpULlS}~wGSWH7T_EoqD%C0KeOU6kfO3J|1aA!}'
    '>%LBEwHXH;vG|?cyp$P^74ilqO*WJd20EY!;NR&;82yxinw5*}<HR13CtVncm_+p5|l{>@XgDwsqD8pe@B^=_d7scWbZnP*ChY!e'
    'WB5<e%;@?p?d{JHyfy0$ZUX%xiSx0z6=CuN)qBeCno)b8z=mJG!#q7kg3+YlfRXFVMo0dG{4HS~@oVt?_XInWO<_~6L={+3g59(9'
    '#Uk-=)gKdfUABRJI1b;&I5vnbz;<V34iR)|<c?jShOd$sW9D-'
    '@&5`aH2ojd|?2WH4xFF*L$!3pK;5hCs?xBhiK7yGIDSJ<u{AnP177OV7(tG6H3_KbUIghv?F_MH1so;~oAHf9~4jA@vt$AX!7Pcr'
    ';JDqCmYkSYoN6ZZnHFPM$Y7dTVzW2GGyGQEZcTH?q=4Dc>U^ab_u4lt7+V7U|$DZVaM?)xQTpm#y4FW4sU05km|$C|V!Fzb(WGe;'
    '$2P~su?z!16M>Ca4Tt1eb@+q0R>)m*Gax2sXtGhl;LlPLl#NjSOay-'
    'I4RJLvsfh~67R^bQWu`;mv<Imo0##Uj5&st@88`yEm>QH{>;k$Zw_^8SDvAk<g<N93WPa<aRS4@7EB`WPPUV0cJ~;j3K?w{Wd2g?'
    '--pR>H6vgQH_HJQy#D!|)KiBm%?gu_7=GlXY|yh6l?#L}2(Md51h0KA12}SZn0sPZTX`z&-'
    'yz7%{B_PH6Lk6wxzHS<N!6`}_$}L|<SU1Xo}QsLI^Kbc@e>82)^S;h%>X9ui`ByTPz4P`Ne{w<Jz0JFN-'
    'uDb^~9df5slg~Z53_85ao#26GT`|r5R{(sAs{cBtRZ$$Aq3E)j!9OMGHy#}xvfMX&7j1O^4B!cl-'
    'kI4n$YK<55U^yV%L0<rKbxbY@lLIkC-XRBs$${7|?~n(=Jwr1*CdGh@?W&Oegj1<e>#1D>e45<uv?gg9I~y8`CbUj2*}^(Wa;RfK'
    'U<YZZk9_s#m}c4~SU&zE;;TQ;l;p0&dU5SGT*Jf{yl>(Q;WzP%@S7MKeiI)%c3mm7OezENdKbuVmjU@&7sz)c;(W<3B7U8S3lay&'
    '?KF_pQy&`(WO#^UV}bm(ye0>b+iTRQ2g?CuLOu-evAKXu$R8^2kORns{Ey`w@&Ng8AIKQpX)(g!lAeirnVE@+gSW62lbt#{B<0-'
    'ek=Cin1Or04kUlWs;)@>cz8K=}rVw|-Lfn1g;SO`LC9796+3M9yv3fyMt#r*aD;+f5N(ap-L+-y^<i1jd-0NKA-kOMW6moA%#JNc'
    'YA40wN=vd^!LyV3^?kn<|2roiCScDe=Q#3ltix?*F5aC69BJYp~xdTHl;$klDvN@-pbMa>zP`#0h2ke<-aFP-'
    '&THDM9TfzIZ@zv{?_>za&FNK)>Wr*3~A!a}IFgqn%kN>!hax^hl)4mD&(`{b1$<@c5DcfF}X;0e3h+(d18Z~?zKkDN6*)kko?^3-'
    'q!#6Bzvsa1-'
    'Sq1y6M&h_w6vHDN7mMO&<uwuZS3OvS{e?+8F3SE6mv@M;zn{uG<U#R>GW&}Qy^dzJIy@69_t=6M(7DGJXR9X_%?u~|TBda7t@XX0'
    'iGw{P9~>h2<`BsvLL~3-kX$mt?AfI3HA91*y-qQ-so5Kb_&nQSc+azqhW0$$R33v6&%3C7p$xS*xTsw$?op^+ChlhyY_EFj<6}_^'
    'k8pe}YG07oMA%;SU=g+#Cg}Jm+dD$uA;R|Vkax&~+T%miwg7IatiFUCA+3p}ZWC)b5oL&V32cvAh>|2Lnyw{O&{)fY3Y0pl<t7>Q'
    'w0c_~j2M{gKL!bG8tg9aWWN5yZRQK0BM$MtfkVP?;8)={Ff#lGKJy|Py<teDKvaO?S6vMM+re<DgBVkE0K7!ps{p)0JW$E(sz*K{'
    '7Qk@-'
    '6Ji1UZ+T6G*;Nk~VRm73Plz(RBjp_;%<gCM4tW54q7UE(4B1r~4!4r0?NYLA)lJmpf_}2*$8C@x_RTH$gKH8OYHN}JYNa%Q+PDyE'
    'GgHUh;wM!yRjh|fvY4+KE~m++inWiViTQ?U0G?_}S`U|mF#5YVLbAb>2Y~l<f@gb$rgsu|9N&YvWP(IC>r`A)DR`#FU|ekQY)`x-'
    'E_k*VUJ_w?)ni4NUYM+LQKomCydc8#&XgDALGHK9kSmdKg@Fl`hwMrVc~%OZ1aU#NAbn=h7jrV&FpeM$GC9t(gqrg$2jPO9DaPQv'
    'CUMj-alVDWgx|uX@LM<|{1(nJ-$DS`p$=e&g@B!G0gGBs;!;IY-<XQaD+SBcgPsu!*r9ky9AJmxB@v-'
    'A^;i+1G8oY_qC#b7$O|GuW#`BX@&NYqz*i8nQ<H%mEvi#EvPQ9r`&1IdRs=}xY{jV!np>#y+S*+C9T<0EC>lbu#T$l}kVHluy(54'
    'K%5ON}d@=;*<Pe-'
    '^hTuHcGt{u($1OjnBptF@P^g%3aSp}|${dCruEX$hnbSMP12XcdD;7yEe&7N*)j(D+b!M#B01t6ytk>|eye0>f3mP}-'
    '!E!*koxT9Z|IA!aCcorNc|i^+)7N#bydV#hCx%g9mC~|x)NX{oXf*-'
    ')NMR&m;B0nx{4zfPF0t3oJs8(_qmVlU7kHy$JRFJM;J=3Vb~@?h_5qxrK$L^#r$RJO3DG=0MDuwbny+Fq3C3+J>+804dP5nGA8~Q'
    '~a2bxLx*o*mx#T?u9IrzimkKymuQNUt$M6v2V{!bjye0z2>cJv#3?n~23diH+1raztPhOA*$J0X`uVu2xS<FmALLUiZ%RC^3%DLf'
    '_M(#enyOX|0k!e+FLh@8D{^}v}uOT9*hKM{XMCADnBC*xq&~I!qL>t8TMy>v@*;fBIY=8eo+uy&b47v+l=x!>5?lc#=Rf=t|gzaD'
    'Aj!Mq2dhfGhp$iXjRxETk$!j9qU-'
    'e)S?k`NySyArqS@MDi_xF5xK_2K{7(iE&VVJF2t=*~>9bqA6F)T~;i<W5~NxO<H^d+Wo@iz~)e+ywdErjia5VjY1ux+snGp|{Onb'
    '$4D%zDc(v%%8JY$RG4)jPkw47T^Vu-#q;+vzTB3rzf4c@EV`wp{_X>b)n#!WJH4LM&{z%WEQFs~#)@wlGB#qF_5gUJwD>3*-'
    'fPV0)PdTTKC*U`b<VqJ}6Lo<PGRg$`}eq<z;Q1#OGv)bZ;Yk(^|P4UkRes1E5M_~{VA(?bML3=w>x=L*bXvZhO)UC(56l~c``IFF'
    '0lNyUzfi@3N`uO;iTbz)#jgo~eGe69?}D_t1hQwHN1E{quyw@MgiP2yq~#Z04Ez5B#i6vH!2j79N1@|qkdCQzye%YkA7B}~)ATqq'
    '_`PLvnqKry*87s?Crp!jFyC`R!>)7H8U4Q;LC4LDSZMZ_;jmQ$)i){uKV0x;Dmf#?AD?;&t!gup#J1nxy1aIa>v4#q?~wsFys6Eo'
    ';Xrk4D5u^phPw}>0Wr<gI+gizQR^r~SA{fvvt(lP}9%|-CXe$02A6Z6IH-P@J(@S8-'
    '|@03|#H3VnJ;uoIa>{$GMEU$^M!s@{ytT4>h*-'
    '=*bY<WS16~0JbkORM><6cyLPG}pHh?8y3cwi#FX3yXWzirRHm!PXaXP+UHt2qO#46$bbqv7^!Js0;oJ&ZD^agwR8z&~)XUd4m*@Y'
    'NOiZ(6WDXce=`Uxy@H^Sw%mV@G$aN#z4|8%Cl#&vigy6>*)`DuxC>!nU^)Ve1&$WmcgIUBBbFB(4q}ZUmRa^>=r~OCqY!)p(CdVy'
    'e(#O2#BHRp_hb1-bge%HTL45ns2E>z{~k*fW6hH|^O9hE=O{c6uU?wrAf>#4+~lltdhF&tCHiA^w6r-'
    '3E?FN?8sEjx2=ZLk=9RsL9(*c5s}*vlrY3j+u!|k5G@a!nPK<8DoIP39oglAS4^3bk|D<BOCF1g)&eMcR)EJ1SPKoO7*5^#Da1-'
    'UJ?h&5qL=iDAlNcISQ08F<*`X<wNp<Tuowo1uZ&11xu{-'
    'IhzugV|RvUCr*q@+>)SfcpcT6*KkXQF()B14BZ$=Rb{ghRaIsy9bs!yXR5?IaBLLQRpNVa>=e^g;(Kyz71LGXdvWX)(^cYob8HsV'
    'RpK2vc8h5l@O?P8i(P3{XZl&__mdhl4)l2l{fBqfsPm?~@UJU#U6V>^R5w2}5{-'
    'DPGb7=MPjqG!9@S8MB?^x)T3?C6<HPcTTrFc|MiLx>Z&^bKtJ!z#SwkX@v*v9h7x&n+4P4x7&)(qTK701MS1on|M=6s*>f|X|EJi'
    '?qU!U@Tl7v8cq})c%;^G}feZR+fU2fGFO|ornQkBJUkD(Yw9kskkPg}n;ej_>#*GKid!IxBq%okl`t}R1mS_zrzZO6wV6K*v=7MW'
    '}1H4(^EW1mGK6Q(DNLgpj#f?T8ndxw^ii504u@kg*<=-{?@kHS)yM#Hv~9SS>F;yTHCi|-'
    'rmiVWp*i7g;IOk8SOwX?qB2d2F>J6v35dJMB8#O2&5lJ&P>?JxDo9I&P#SRXC(FR_QxNn>CerqfWQ$h4e}qIyrX(GeENO8C;M2-'
    't_^-'
    '(6TfRR+sKC0MHWIx7~IaI3RoVfmE2CIXgfl)oAUOPGkSM#1t?c|k6(a_@i(V^#uqq=exFR|Z6N)#?V7KA5STNe0I@I=X~H7_iceL'
    'iMPetR&_fC0;~_QTbwy$`^N3z7j{}D|J-93XaNGJ4fZKy=h+f8*&L5cn^@W+zz}8A-o^k*`ocBi^0`p7;IMwgX-oJVlfD}nh=Y@)'
    '$*DM4630JQ5b~L5>Xg@OkR-7i40BS&$!s3Oyik6d-Y0>MzNVzGUQ$dmyd`Pmw?AdOo~gv;Ug-=W#I1-'
    'm*R48_lQh!g$e8Q%2tB2XH*#nTF4BP8jTKQ+J%sLyuyyV8GFo|40nE`VPjF|u+|B3)p1*|P;_g^N|J*)fZ?Yu4DT<4Vf#ujR4+6!'
    '7KU)EiLo%eUtSXdLp8i7M!^tf;KV2xJ}xiF<vWHj9LKY#B?|vt5IsCUtcknYkCJ6a^?a*Xy>in!8%n5AiOSjUqOZ>p7ihBL0(V<+'
    'fw@*(V4f8hm~X`e7FcnCg;rc(5gEh3%ZgY+_8_S@>L9y)i0mhJW(-'
    '?h6u#$&E(aDXKv3QJ>{tlGea?=B;Cu3#2ngzNh=L#t)k#qhd_rE3hiTw?3YlvJp2f3wB?3P#hyfmj-'
    '`YgfCzSZ9m%L(%n^&w!tcs2=`d;)1pk^nE1}c!kdnggV<&N~-!-@DEm!fZQ*L@}B)xNr*d6jAMY7qA@Ls2g7Wrm|1m70m9T-'
    '?tLOS$+nGc@HWc}+a!DCSIxM3PaEOOVFUo_1jl2KUtStp2G{No2L&fp|$=wO%z|l4q`O_U5`1#u|UBNXX8{%!13WgW40S*kUwJrz'
    '2_CY&7W6y>?_~rTp|NX4eysK2!>d40qI0MmXXkGAXDge|3WOMz-_xkwpBSJGzIDCE}0V(K&oP5qEK|WQgpcBeB!i{4Gj~xc}Opnf'
    'Rr9^#e@YY_3KE$x4&o28C$XO0%#2t)nAnAMbQ@?DW^;C3&X*H^KC`B;v<v`d=uB?(TL_O4|%|RAF>wqIfBx!frKw^_cRt!#kK~i='
    '7X72+x){5|%@Gw$z!)VLV&r2v`p1*>Y!QM(}KfBM&-'
    '~XDhXqqonrK^W?;GF`H+gXo%+=@v_k}t!MU)SX1ouLvt#@h~w+*R%!Cp10EAQ`EaXaA}1f8=$L%7e^)U3ow=B(X8-'
    'wu_@X!aD1KD0h4WDVHX$k2e<!*uA5V}|mFwkQsu<UDGR=oF(S?(3?#)D3PR6+p6Wusj=fjxj&dEIYWugZs`+PVPJvkZZBbeyLm2j'
    'j}ozRTz7)GSVZU&38zemx0)mm1!KGGo3VzicNZP>LemI$-EJL8F-'
    'aWxjl#?Cn0>e$#B|4?3&XT}!>Gv1nrpDKBG*hrt{vinJitHF-'
    'M)evit?_=T@$^gThBXcgzxndMw>##LH<51;3%PjNhKbWPOe2!U~zvr1{Mtgx-cB&Vd`)RL5wOO}@b!Vb)MHcR86b|aeR*VElspL3'
    '%$sU#FT8+Wz*tv#VjgFn`E%KT?bG;&%>#9`z#FSEv_mUzr9c9na$m}x{izKT#^$Hb@6lBTWI5gW-'
    '%4k_F=8Hs13GGCyD~W$5)bpc2bUza}acS)RkF|@{E9q+WO1fFSlI~Woq=(fj>1p*!dMUjUyeSenS6si_AZ_ef$%d<c?6}xjhFcvM'
    'JIhbXYx2zU0~M;Ys#EaHE2!w%z8UHUYbFEPLTh6wY{M-'
    'wNi>|8&aJsz3`tD4)_g98C8k|#Ar~VO)33D%e6NJ47{o0JrN~ejJt4xIbqRHesfWLZi5p$l|6V43&K;NGK_-'
    '61Y28O^Zd&*CZsl(U(N~3So-2q?%8CQpB1`O5Y39{!j*p#rxYhBoGyjCVCeO@29(><LBx{u~eosO4clK$n@M(jnNewMa1M-'
    'TM=)@`tx+H`SYlX!GEZOt87^>$MRqqqZUZT)_BDec$x@uDZlywKYq}glE<aSN7*PXfTmS*c6cd&b!z2UfnJ<@E0;|}&rvyF~B*el'
    'I85iF3Sz2axTQ-'
    ';D)2T@<FF!SnPIw5xE;XWtC&in)NnmjZAbXZW_VyhNpfP4;VV`HDfI?KI`{r4x6Stz<vea)WYX(#+2RRX;=$@aL?Z&{k@#cXD>pw'
    '{i=u-2_M32NQO+FCa?cWWeRfJMn)IPNJdOrGSpc37M|#j$170_Cq9yGAWi{>H^LNA?{_*=I=I_ZafPE{TH&s&iAX-yBa7*c?w0*c'
    '?^mrXDMz$_?hJIjYJH=CL`d%5AK?L!K(l7eZo`BW3N&U@P9x&$biwmx;AuR<2=yZ7v!pBh4f9PUS5QuB+Z=Z04|q)eLP?z|RHoIU'
    '$I@7J|4-2;x)HkYo+xT&)ymmlB3WO*S+x#ceUQo#4*Mi0$HEPy+y#n-'
    '+Z`^|tu|yCzPd=Ix0%PjUO)nTU&$3LT^B)fdO}11^r|2V5M}G5Skh1y@(8$yglKItr7uII49NCUbFA>*y)+4tbFLYS^bVGexQ%Q}'
    '5VJ<;EB)o5k!Vrsye2*+4`eQA%M`Rb;Mb-O8;L^P3#@YI<Y+HqYkM?)hB~+daKs-omp*6ws7{l`i<t4Z;8Q5d2+3@HcqX%HR$R<z'
    'ST68SL7Jb1=cNJb~R@AYa48_*4S<S|-j)9UvDpkkxA}*$tk+CA+~BxFi>psc}v{SPleJ;~b3ml3WO;#yOagCAknx-'
    '$;YJLmmWg3=v$95>GZtwnm<sEp}sSBmI@5Y>JM;b0DQos*Az%LJWQ*#9+4&gQtdG1S}DgQ%VvPWjKfW{0?Q9GWZKOo0uUSoC>pv8'
    'OFhdFq@bWTs+OqCT1iD_aT-lu)B-'
    'ptC=`6mCn=^X?8$`;(0Y1OLv1QaOrL^1ul(pE7XHUxD_yQOQYNhnAoLJZpEqc4tb#bZV2TnFK=O6MpeI^-'
    'SEhE$%Kl6oZdTWiNn@ADfH{2`TP*g-'
    'we^*Jw)?qA)4zL8h+~&gF7@Sb$sHZn3$3}J}|gMQ&YzW2B&CR>iEFm7EMo`{n8*NW~Ajlaa&Y^JzQj8$HW9>rEg&3?6d;3)zB>44'
    'JyH9yFn$mEDGA{!6Kjyv$!k@+Azz@qM&`6yh9#ne^5?>*+hH0k{b_ga^s<O7!Ng-aWyht-bjjRvMQk2bu-'
    '@K*#fOhuC&ozkY5mj{97T&dxRkWuGe3>1PrmIe(hr_td@aEwlv9o9h&6BO@-A;FwMS%Y99W;o-TN2G4RVJc&}$-Vp^doPrdT;-'
    'Cz)0z8egJ%cGj|)PqGd<-'
    'zPMk7~+;*;^jfl=oeEhdkij5rVfU`Tr<JX%=qjdsK1j3d@q>R^nYjTAV6I=^H~SX;K)LG{q$Xa@c<y${(rE?!x=R5Z>Po;oUQY_v'
    's$In@!jp8IKP&g@yCLbmK-ejzqAR3*D=jI730VN)f`B@ldGOS+N^5f-81|MsP(G4~2TL2p$R;`4v$-'
    '6fiR@qIf7ymv_k1kJpZs>2S#M1mhmhVqTk2D|e|=)^J&dCDtZX7_-'
    '*&n3N<L&@9$AtCd4#W%;Wx@_Lbn;EO^8e<wt6uMolC^ALPDI30Rtnau_FT8C1yMc_Qh<ZFHk!LPXOq)cfSl#uDrCxIvUs&`qr8zh'
    '1&cY{Q5WfXkXgGIm>hJ9red|_HvM#1-'
    'c@(#J$NjrwEDN@&$So(T#s|uvaENi(NgX_xO7+iPm#^8EzHwM>}yCj&sxQR#C+XBFPEv;<DWWoZ#*-'
    '0faH;3t^#r1SroSGzZhmgmjiD)&pVR1?Y;^AGh2o82z`KKaTmIUui_?TS=Oe!^0@3@Lb_U-M7m&En$?S+>_6dJ1WUK=H8RL`{(c-'
    '}%on6kAw#RfYDa_X{z1F6-vB{W~~M{wRbr6hB}oTBM0n{Z}{ivoZ5`tx7HNU{OlU@ezyHDE%}-'
    'Dp6kyy9kZO%BZ#6Hia?Yzgs7VMW7wqHkYD>^*(6<;XUr=-XEik<Ss?%5ox~T~h)ManUy&IV>dlF5u$0N(92{W$%ea-'
    '%z|H4t>M$k{sw`HuMY+_e?JIk(&#1^Gq)EB~~=%nViys9fPKmY8jm=VQxuW<CH!&2+5yh)1)m_4N-'
    ')@vD>LRiQ5s}<i>NGDV}RV`A<w}7*Gz;lF3%oB?RTqT~NXYf)-8-yCwijO%Av{0bI`2@~;U1bDkE4O#qqmb?biu&|IM9V-'
    'vupb|5}W0Gx|riT;N23TZO+a`#4p5N~yFBnt6~?u~+>8tP}GU<kAEY!nP(ww}$YFIcuBcjZ>rPjur}*ALcPE9?i$tyT7e)z(V;X{'
    '~^68d@(Sr=$VeP%VLMl~O{W{lWtpXDa)uX>yp#r8>(o8tfLGZmvjX1q(2K8dq4>vu4wL(u_dv9Lvc!%(aN;aB;M<i05-'
    'Lx)LL`8i4y^!3Ym=Uo03Ome=F}V|yF=mD*8xz)0R6%-BD2fw6-'
    'XZ}~?~QR;pC6&W|#+cbTiX%})b^|0{Pq*QU)Y{RMBn5FU<t*E<{#FVvvUAPspM0Mg}7Nf(oAhK0e2{C$;hf&5<#efzNWGeqpR8GH'
    '7Xmx%={2`&`d4u?4LhJJ;@uvh=WhFT%E<~D}ShBt0E+)_8;#h^ri?}#}RR}Gq7rs9hlkgDt$6|7mye0=GxsBPv<2{!PlSVX!Yhm!'
    'VFqv9emgjPMi2){6iKo#RDd83c)?rP9t{g_fA^eTi6mO7VrnS2X151J6&%dycz`$_0)<L#HDIpAh>A|pysX`XbFUC}viaJY~DpJu'
    'PW8!J_<tX3<=COs?oqn5;i}||vT|z$QMsZhyK4ct~-'
    'Ut_z0PEvbt#1_<`&1%vQm_B#SWv=4{5cks+vPP8tG1eq=c86_n6KxfR&AKi=W{xb0Vrjgv0-'
    'm)<TfXYn;M&R!mK~|j9b;VS}%gSLQe7J$!)8J<f~ZCVjwv}3nN>_lMs?Odyt$X`TBG%*l!c@5zD%0HHt)RXIWRRAe4xYS=LQUQzY'
    'UOmUY*<ZHf4lWj(Y`MIv^vtf$tgNW^C>>lI6jH`3*Wz9kWRsgh*a=<QtzlIryzh=n9P!~?OAyhmOW0ZBC(FGN8S=IezhNWy%+kkg'
    'UeCv+s&bI~mo$0{B}F_+=-tS--;&+vFopDkc`Jg?6dF+5(>WQcEz`J_~mZ4rx%Hpv6nIGYXBZ6HYFXka>0>n2-'
    'Kk`Si9su<VN&SO{61Y0RFCZSzcm93M(lGUa}>s68uGAALLzGfBci6FY0Q1pq)s}&&EFwucm2=1sCeJ~b~@B|OW0`g;dO+<J{4f~5'
    'x;T@Qn7o);EFmo^FbS7DO`W-Et$wtodg~Ss41G99?<fN!cD8sW8rgE<`wcZ0=O{kSq$AkEH-'
    'LAB&G5X#_rL$AB`ScA;&lakURs(XJn-nGn22I`)`OZz^5;;yQWh-~hQ+K4hcCY4ud+AU<#Kq?-'
    'CEC^7&*G7FeS702admwi@sfx%J50kY9+PH=X`01j((HBe4sqJmvv{s}`GKs0>^iuVrtKaUu5k&~Zx%6sLpi@%3On~v9W?7`16*6i'
    'oIlvZ?Y<#yj|_2p<gQe!G4FY|6*rqp;$~B_c{+}O2~ZL@n~^P22A!CDb(_ExrZe&}G<N-'
    '5%wCd;d8!swq?Ps7)tg@xi`n6LNgQTJ;3W~5g(<iy3bQapS4Cm=NO^}it?H}tVfNqvvlt#9mD0Ya7EQ3Wqg`>sGA5yR$fBjfb#%@'
    '~OFI}aH%p}J`es>PcNK&Gn)14<(J-3wx@*xgn)15q8T{9j*WG}&(ZoQ9cppNi@I&YqehB?`E`5#v;upE-pGskqI0C$*TpWKnrEKM'
    'isZV~}&PEjBxH=NY_z+h|0vVt6Y91qb24i?NkC8lsF}|9|NS^hRcZiTYgE_i7ACh~9))zq8;uzmCiKK6$%{>_6u?p6+QcTNNGhzU'
    '9yS!(y@?d-)jQGQUO$GCHxPF_6=}-@r`-'
    'QMPDum@xyHuY>A+mv~A|h9{c;gbk|K+TN`dPpdzs|vmYae4H2lwM=v8;LhT{wR=75ghVpOpH^cWz^na9k4$XLyKfV&S}2UXugPWH'
    'w>!ugL{xGMg|r*W`jTrT34LcgO+fc2>ChntX6RoK<MnR4GnFsZT89Vr+tbMy*j*9gB4%t(<C7*I8-'
    '<#bLfKs7M{=VRruzv;9NN_TQy)HHNbi{74iITR3t5H;C6bI2@YqbUg<*Lo=Md!NF_R?(hZ<K8N;(H*s(}u8gJF8{lI22dP+~T#L('
    '7u_&orcM8X~u^5JjxHcBUPswW{oC}zXYonYCn5%1}oQwYQ4sjCJ*XG0UKo)AFNqb89A5ly!8(+yb6_>~|uVZMOtF>)y;rItFp8n2'
    'o8-'
    '!IrJRrpGfDpR_cBx>kD9>bNmChD0^jA!Ni=)3{!n7&+D<+drz@@+hZQG&uVlrXv(R+!d=^N<cxDjI+631gwzmBiUhCSi9E*8h|5Z'
    'A@xc(uGH0>>~f*G1tN=Iy#D91oCph`=$-`*rznJR-!ga*~@2XDIj<CNxn@(EOUX1-'
    'cy`#}Ubz*BYi#3+E9wfUB^1U<l`dA)E*9LdIGx8fPSagY21!-'
    'yl0a@f&0(Bz{Be!~~U9Z8N~xiEfb9A+_n~X&fBO{Wr+P_Qk2VTVebAsZ>oX!?qfk>tnGE4{?1gw(pnML|_{x>G~*a!z5lGh3$dz4'
    'slY~*XP6b@s<3H*#K^016&Qhabo?4Q7Qh3?sq|+hN45=20z*(sYckZ3-'
    'o(Nde}ZF#P*;N+k<u?Ppz<5Lhr(6m?xowVUV6pMt|4<J)43qhrxL^6}=AA1Cvcd$HSmLn}NQ^AyHg{CAV(%(p1b>kp4<4I#n{)YC'
    'vv?g)}_D4Y81ZPhJxNX_%WEq96@(cS97U2gy4`KpN)$hJ27dk(Kwm)LQq#bQ-oGX}^xf)iONMDY@Ddj~bn59j%I6!~E|kj^BcV=e'
    '41bPh%*zUJUVki|2YA$B^9p2MO_BBTxI~Nw9EWHDT{;GF21y$)-'
    '>VVc%>Tl@K1DO^>4>JT!Hn`#CP&REdq|R7z2mL07%|xOj5Haq;AY<B}NQ!X%7KVu0HyFNgqcQNDNtaBq=!$OGJOhru|p7OgF@@z!'
    '(FziAMwBujwNYHtSja96G27Ph^axGf>IC{$HB+5_=3A&4&yL40Tk;#)lsk7i;tSigj5y<QVtv=<BxY@I0{hZX-wQ_%y9em_&h2K%'
    ')EvQ#aKMX`74Wad_<qPuFMU6V>lz_OAW_1<Uf2EXDN@%)NsBr(leFau{KG0j>R%L^i!wGNeU9?`6ItGq)V1fO1(0$C@4>Tu{69e('
    '|iLjWpma5X2g#TM6Xt_^#_kKy7N54X>Txcz>J+uk8=Z}ajX&2l*)0=m~lS3^LzUUV}AbZ?06hJbE^=wS%xHj18xfNqoM#bS9N`?&'
    'BeGSMZa#B$F{2|V@sXYK|=<e9s{5P4>l1%{b8Gs*&gUtSPlfqTnWkFdbE$vfo1@5Hinh>5Z_h;1o_Ca_4`DcW%vZbysO)Mlz1M2o'
    'a~9!uJ>Tpa6x_a7m6FA2fhCj{^9p3yy?p$67v+H?31@@8q%`X16^=_24uQtT$w<-'
    'y>e_BZ9hm!%l1(G|g$r`WRfomgVa!(0eg6+{mO;p&38t&$6&24VbeP+pGT4a&>$Q4ogt7#{`UOXLL+5bh&iJp#hF%RA(O@bt2vTc'
    'dO$PBz=QdhWJ9Q9KS|S~BcklBbyx((n(oq%MMCg&*x9__+|lmxc&FEJW}f9)h=~L@U@w{1=Jdv^VfSWO>t&eIGg9bPf9lNa&^k`y'
    'sNqeMjf8#an6i&sd_)zAkp_x#*?pCqB=`Ta~b@UjMA!pzu6vHz+*Mio!0;#92|;y;NQhf!)L8t4Cn>4ta+>*uAh)FrucQNuvFpF4'
    'bH!mS{&?DQOK*bu)Oasr7-'
    '0y10;aRDzHNsdH{|4co8dJX}8?;`#?6uKR|#{<U8ggb$#rIVOoW)9fW(U?DcA*~_|KNW7h9ujpzE@lKlkOBWA`chl@uU3ww5q}jh'
    '?sYwq{EA-o_mz=O0)T9%3gPL?g6nbI2Cq$w52l9dl^!Amn9)aFp%RA&j?`14xrd!DHctLYt{wKAPL14_R9uQbFD+mPU%qjw9Z;82'
    '_Y#}W1$9w2}Aw=h8AvzBa(fJ$C>i(LG4Jjb*{7B-'
    '{Bc#1FRZa}0u*?@X8(Ur78s?e;{@J>OdVN}`zo&#|pU5XsYq4m)QZ5eRVvCAG6bs^xN~IKPJSOf2)9S?CU|OA+#FSFN)J#lbN+~X'
    'r7etg&94=oyqLkt{@(y``{Il`^8g|=ieK*<Us%Nc{!)z9JN%yS6sVz9con0T|1TIeS(EMVE=F3Ag9}%MYP7lquaP+oYbqVvWT)dW'
    'o#Z_mwaq)TzzJWTsos0D;SYUN_2N!RoV1w1!Z@Ac)g5_0bcXF{Q1=}0N4m;Hi1b=~x&8mjtU@qRR<W8uOID0pEW6$0V-'
    'q^FF3^Gj0*--}ha(O|7K|VshdW1o~Q{EvLfk91qhuXSCj7h11Ld;G@ed-'
    '7Sx_ub6)=Mr@_c&($8nFn&MRJ+wb&ZGXZ7ir~uj1mj3=UHd2gf8^0SCuaTU-RkG+O}&#|&c37k}Z|VjZv$Px5Su4p@k%c(zmrDa7'
    'A+w!950+ut(Jw%_fN$2<&9uqKaTqP9{fLcM$`wuQU{UJ}<rz6V|c1NVC-{>a3inD`nO-'
    '{9g~TzrRDYPnDoQOjdmF1E=Av51xnn9JH-'
    'J(ZJ{s0j%SW_mr~wU~f#;$*cBZ(@nz4zAHi>a0#2MrQm!!<tJAQz^$HF5zOSK3O1sz{PTXvPfLc#Y(;twd{R89Dfwzn0YuZVu`2e'
    'XzPtdou(DI57x_>{<g!hs8e>d?UK7Tz~x*QmkATy6fV!@;uuze6-K?xq*z=I#!KRGIRr1sfy;sw#D;<Gk_(qqH}(-'
    'KqU6D4d#j+`B_A%YuL--'
    ';V5y%;J=n6YOsjTf%|_)~iI_Vb77kM)Drd~}w43f07jv;#LvF74J{L>$$$W7s7t8d?LU9=vD>Uu#ZHag|p;AWh!5=fM@nHX-NOmf'
    '7-O0o3_7Jn&!z_FWpC)2Q+w`*m7U#KGY|m7@K<c7&<P~_7)k98>?E`{`m>k;&^j~>R4m4I<IcylRuDQ@iuIF~-'
    '_Q``rM?br3J~TdEW+jVWd$?>}anxKqzRjZH+f+C-)IUu|oypJ@-'
    'jbjd)sfv>6Y)?&JG0vp@km02iFYL8(L_}yR*=fx&jZ`XA=nZRY+#VLC9u!a2{4XT`cLaK_zFHrWGA_dz;euXx8qLC0=(xvrM+vXQ'
    'v$oDm<{kc-=(KdJt^!dv8aSwO^HS2+wz(S8>z;&Ta=Cb7-'
    '=kGY$Qxiw|uDVxNl%174Wo)tV_`@Hd|G*6t*HnbWpS?ELqxE+k9xe$TDf`{XKAf5`rsr!6m6ObnNzS+tD-EcJ$1%9X<1Hf7$}u+_'
    'R8c)ug6gh4_C+_8MSpfol^}Mj6;zCibYb=~1scH5PsF5L08(_lmqG!gExE-'
    '#yB6`~(@!VmwEfzwS9b#{hk@eX+=F9c?|M=)S9ArAlcHl}d@ikj;u!#R|AeU2`2sJyZ-fG=9Cs5JTgKtqodtEjvsMW12WV>nnyct'
    '-6*Sj)jtnL_Rx0jATS2U#4n(fQRf)Lu3~WvfE&64G_4{Mc|`cd{KFbPjGQaTA@2vJ=U~X1j0j1i$&nG@|qk7bo8!akb3Z#?%Yq2R'
    'WGJH7sk0qP9M=npbF_o+;#RE6@4nxK5A!DvOc0yn9}SOrYOS+{!^VxV|@-'
    '6N4hlD7jQAarK!FUdt;o+)Qh+nl#;GbnK1T&9_n_4sB7mf*5IRVPbjI6+oz9_&p>BOIo4NR$lIi44e+_h#U}?JO5wA>WUjPKv_QS'
    'U^jLhtLrjmw=L_<h9QbT!g|J~bd*;F?Ew?+6=P?gHtF3Z&&zxT4zM<DROR5AlQcp|?foOs|;EBN+JsVqxO@LBvzdM&zX_qiA<hqU'
    'uX|U_;mpV1n=W%hAOR#=EHWO;Wx^f}wgFGaD79z2|x3bIcshQ$Dy_t>k^kFv6)0bhKM;4=C>(Gie30Dm^OC7{SH(vP|7hh7qWU3B'
    'jLa1P<-e*QEFyRSi!~*l*@|qmLOszaNjBc-'
    '7z@+k~&yaF74=~$VJ?&mO4QE+BR5L@FEXzog5RnBg<g;``4B!{_ld?t=ovdeWbaEV{=%iB;mCBl+)FlXZqmVp^u6GUzep^F=-'
    '+E8D<wz#ZuqsR7BAw;wDowDuN~oRAlqj>N&|Yo{h*z{hryAJIk6a3v)rmMZQH4>7I3cM}*QZ{)HMUE9PrM|qtZy&8qzwpokwuOdM'
    '3nWx{Iv3zvObu<RvuH<S1<37Cy;(~Sz&4m%5_n_^tcpTUP#wPw5CXMtqZHiCPZ~4r7}_l@UCK8irrk*E`zn0gmG{$<QRhXb0KKo3'
    '_&|M^;tQLZY2`DFShj9ODxy>Qp=RR%yPZ2)HQ$D8+aCLgN8Lg?RK}h<$4Cbg5uIj#qpJzThx2M8H?JXcu5>;hv6kTP)q5bkFXRxC'
    'jA2w^k!5a08HSUQGEb|<sI^%_BUmytz(GHv5rYKR6*lhwF^M%sMHP3*EumKvyHdFos6MwFYXuu_wyle-wJ^{!~<?K?(xQ%RL@{Ez'
    'T^W8Vd6}xPQ~u)@l=wE?SB)fCKdaoCQ?x<cAA{c+N5g@VEi%nbp*~)dCc2*<&r)%3U9?Ww7^5W72DAAvb-'
    'h&$lI}$D+b6gVQ)nN87B0tC?F4!cgO?eyUI%XstRa+PcCZ&Z)T>`MUuJEw!WDn0i*E)Xs*D}vCU)X_{d}E*lrm*`Y?>M^|6_p9hR'
    'YE3Z{6xfN1y?d?EY_HiuuqP|xRhlHtL*+0nt#j<-'
    'nx8^HJz7sfX**d|JAJPI79q{6;Sy~gHP7{fzsj)n2V@|qklHZe{tCWsM>gAw1%W5nWMMmF;pvACh~4tZd_unfkf1dp*v+rfONPIY'
    'WTi#m!G&yig27RpmGd+r5JLlAy31mW8u2#0wfT*L6F;wA?GN21UE%J4|J+33O1-'
    's;FL#xKcyMoaKZGO7HP3^pEZGRFo`{?v7ctGJk=c4~KUze0*EvKoxHW1$QW@pdefH_2-'
    'vEHcc?+ff!7=I!k$iwyJnc9caPChw33$}7rTi)|=<ZX2>xI7|ftq{Zz_xq95>!M=`EEu;W*$w~SLrg<9He|X{25YJx<@%&DR=iwf'
    'ni&Fe|n>FZ5(Waa7lVwbNV#`mKGx4c4_A8hQf>HDB*u-'
    'hRogz+6Fa**D?QDSP9o(1MnyEVvnHXKkRaV3CPAsC~A>N5a^mcho1fpSP-ibmq%-lOsh=!ScCkoNS<sI@M`hl`YMQ2Zqs5-'
    '$vFsC&IQHx?5)tqei94<N+;D57>?$ip7Ld9My_;jrP@Pe@+rVkD={cecq5rGC`Hc6b4c)i~ZiPQT%Rh*W1&EDTloM!Lo;`GGEk+T'
    '`%dx<4Dz#!y^N(=`fdu@R1&s<yGgknw-*|QRHbR}e~L3uY8+3*nW#v=P3c})becVOjW46<SV-i<;w%<sEV$Q~i@kO$e1m-'
    'lobl#K#R6UwwJmw6^b;({wwgaXjKhFiKjQ}bHQC8S+I&%m-'
    'gFQyv;`j8ONTS7pO3`IR_Id<yUY)BU_20QEqSWz*=p*O%>_E3l40JDq39D)PP^$d3y4lsl{vJ%BXNPz1`iht}zieF&jbd}_VAv~@'
    'Urq#G?iN!QL!IoG|e=M(w!1QNWUmk;Ln8Gbln1(6d5{2oJ@(y`0{d9R;nPV@l)m9dDTxI0?WUY2jSjX+2un2!KmiX<P%lPeQ;%DJ'
    'vx)}0xG_kfQa%;Sp#*y%&B^LX`k5;7{2tP_xCL07lT4M2Dp1nnV5BKqGGxdP|nP=~mH-`u8xM4*rv!KnENPZbW?|d#^sf1be-'
    'c@NdIWAlhOO6YdL?9L>pel_)Y?FMCIC9*oGzPIX@`5~weW6lo*DM)D?o5zHr$XmDN~sCXX)Tegj{e7zSjCC?c@-'
    '~<#$|72Z;B(evwIH1<0Oy1z(?`jtS!RaeLeTNHZ`R-qDyLOZA9196x)bysj0RR-BVL;BQ|k7?n>KW$_*=DUHmR5HoLi8{IgODM7='
    'Rfqlt1^8cmeT(wGzo3_VL@QXqHB_lQV=94YS*kpkIQUXTaBua?n^*1F2cMOQg_YbwU3#7>TQtxa@rO^b({L|$zYa+5cpuJhDX)z_'
    'fyZUv*Qe&3vJ7Dwsd-'
    '^tz)1N84(vMpkeM(4Q<kCQyw10U~mSzElgojknvNKK8H=$Wb_u{IcT!)jUP$~dIrShdBQlZxYch4`C#1)jz;<nlD0A(y8y@i!Pvp'
    '2o!A=E^sTh`;rdcZi6;b&?n4f$+wPyhufIa9gN|Oxjdp<&cEVV@Wx2mGPTf@b`5tZpWr*wblPBPEAp9UKNa?V(#=*RfhG+-'
    'p$?>12wkJV|bk8@f`SQoyXdu%iYgItLi)Gnz$V^z4TeHv|?RjNQfKO&N|jtbSjAds*u|O1@V4bq3=MwQj*3K;wEW4A#Rcjy#>oUk'
    'fbqcwDaVfM5xgoCGQZSM!TQ9AP;)qEi=HLz-JSJ&#)I1HA5ZdoMFK(ks+Rw5XlRM+FG!w%L4e6R7^^B@uO%+#pG01KcZUsw3Ni{&'
    'u4g?<WU>=pq<a!;>+FN1FdQS>yo(5GTjolA*_4iw$AiS+GNBHi*sEUqM3+94=jieDi!#sH%!xbLfkZsC&W#2VK}wy18FV{6X@s5H'
    '_3tFc9sw~&4uCimZLY#hv5%`4)F$5W1f-'
    'fLfP!?>}}CsW8wma$4MU1fRC~TtSx5S1H!m39JPttTG%5o;eg*JqKiut`nyDQb!kHXn1~*d7PJix+ps8?Ij!P5^_kGE#s%eM;}$s'
    '-4w7Wj^plrtFNenSGGbc1n~Rr{3MA!fEDLEoF>WD^C&n%0!n4CiP)K7)$`{Bti6AK-Any=CQhtEEAP=5*l%+w&rnbGm7L4-'
    'PG;D)VIJAblWT~H<6u*+)$-R}mCHiUjUdZq`$>TQgKZu2_EpFTcy$_-v2S-8NW<LSBIye!x-'
    'zTD*%Z&SnM096PMC~rD5O(U1|CESck{Q>u?ZBF}mvnaP3*y}5%C<Oi!-'
    '8SvwC&LTPK`smG@c2!T^i4X+b(K{2UF56YKON_zDdLmZ=k$G#18L3c|jg9x8wPDcwnGf0|RBdE^Gn3M0ArIRlA7cagxVr;G=jEYl'
    '{W<AP>cL9JiO+-'
    '9c@cE{V8RpB=@)j}W)(v;G|X2yur#8^FQ(i(l)rfgHRF(||UJgVQYTOxmQsbz{T7a%00a1=s~kqr6{1{I62uxEhJ}X*~UH`!t^Zw'
    'tZBj4JM_1RHSW@e3Ll(+xAhBwu9saxmXW6h1z~7pIZ>GrB)@nAl9c=47wmTriQ$rH6`XRF(wf+4P$3BOiF>O*kxS8L2*kG)UeM<-'
    'ENA*5-}RtN3;{0h~pC(1;ew~IUXGyb-'
    '$f|cwMpXbPXd(fUM3+NH>`>Q<QdT!ZOoc65s}=cCnL&hP`d03Q>$p#mq`2LTW6=CXvnM2jV4h&E?g2N&dzRIWM!L+A?73W=FMU%#'
    'p1Vxf<6xg{|e|64R0<d2yMoeQ#!V&*pUQ;w6@qgW+jV_`Vok&H@vIzYZ{83xWAt7nlH`IwJOPvJ?H_XRiA7DEOIIma`ZDKMQu2zS'
    'uQyZGgKaZutY4<mVG&p6go4XsUNTB@%e}45!2*uO2VS-!v%){>G@5Ntl}(qgp0m?rzN4chM=-'
    'fv$CneVd&&Ni4gWg{2I0xH2(wB#n^PgOUg_rU=YRh`uH8juw6BusY~_Jw)H{JhKS5ZGvY}Cs*<8Zn0ROT+Oq&Vu?PvhG+A{QhjnQ'
    '&*qC|`s6yEEfCA~$@M&2C|2l`8+f)ztZbWyHh|+&7mnaP{6oR<i(IzzmcdcId_yc8x5{hcI)Ofs*F@kEr{d>Pc!b&dc@!RDHh-'
    'S8L9J8R!VAl6m04R$PT6jiNznrM@NZ1vIWBRHREQ`OVs7|Kno!b{>;g#Ad<I!p0Foxs$hrZJG_^t29blw2m$Dv!BF(Rn^~CBn#YL'
    'C(0t^w=#!?tSi*!szI7nX~BK`O2PMUZ_8rcAW%UlG)4((ZmK-j2Vom6PzRS$J)ECSz>*Tf<4LwQa9CSEy2zldt#g;D%PR0}VR@-'
    'K2y80=SO5wU{HB^<J}u)4ln7rek<)KO$+7I&G4no&hItXMn&_BRO$KmQ`Z#_?|`Xn%%H2-'
    'Fp>D)|a}7xR@woU19|8^k?aoTIzY`v4%sS1s%?MCQQmjSzN!@C@vFj>pZ3;d#41h5xq64jVvnxeH0yL_Mb<345t)lM1wi>VZ#-'
    'h2(4Unm9;qk=Nvaq-LU#!*^3INRqb)BYaaXNK!ruX5glr?%{rA^laAgE20QZqZm~%6##XBO09h-'
    'no@wIPBm2*5QLh^Wp=8qpv7a2)XrM#HAYf3x9YrN`s~XI;_hqeU%HQr^N8{QwY4uY$XyOYXAy~+SO<|CLPY-'
    '2Lu4^QvB?MY4L5Q;7N%|Tx(0|`;UW_D`&TJM!e0NHN{Cbs{M}eYw#aMZ5c#CMCIXQ-'
    'e7}rBB#iJcqYw!*@XMTTW0{exb8BErX#5&=i!RvDlB|GDj8EAFr)y|=My<PM?dQpg>)9TK_RR^hWoT{i?by*p$)*Q5@*t~hDa^Du'
    '(P9mArw-OOhFH7H+uSTgt;szL)DL#O+GXRv+JWP-QQO5mI3C1quz}<8blaS;0VY?vnCy{=>twkrIJhV93MnS_UZ=-G@@aWZ93-'
    'ER*W~XZm!o`hR0las#LZD1<S<b;=ky%+E2AC)P+nn@n;dMR*0JSCkhqRfy*B)hjiRK{S<6j(ql#6plk(#pBkM3M7!~(xRIG#%u*k'
    'z269;*lLgf7^^*IaRVX6z%y?7PCtw7z2R|DD#)LVNk7kd?`d+|Ch_AXGv;tgEvqtUt#$K&|6Iau9j<Zs<*<kK8{T4^F{6EUk&)s%'
    'YM@5SQrL3vFa9v_g`<iMj&U&*omRW3Zz#ssG4SGn*=(M6cPU*+^8xtGFJU~0rZE?u6dn~ro(6vk4-'
    'KecH&DZy1#E6A_n7+lmLJFK`YxB#0K7Eq513#jLW1=O>>u=pDskHa;DkdLDv?n;~>qgcz}2%3CNq^T$+znpkKQdKuvsRWx!?Ft_t'
    '#tf=i9M9@iuDwi94TDI6dsbqDQ6p4}?GbN-XO+eCtd`j_Wp^TKqF^dZQ9X$;)upJO#M$x=d4lB!@_@4)X#m7o+6L|`r754$uA}O2'
    'rf$@)CgOZ8pwb}j<+y!mFg2&l1M^iOn9m5o{M!)BH+q%Ra&x95rH6^`9B(_pAvGQa^J*gavX4@otYUD`)~#~_hmC<1nbXzY6|P4d'
    'S4WbvUhRg>H;T(tgZL=?HY#9P4a}ri47b6_%Hla$%P<TxG${(hFoTn#Fbp$1DGI|k$~)x2@WEvmo}5h<XKD-'
    '|ASJM$Pfg@n{EL~;wRn}8x>4?7_UaI`XNH*lU5MGAdzkGlx{`81c7*7~DI#zv_CHCjq6tnz&RRn(+v*bC6q%cKtuDad8tR-iDgL#'
    'Nnr)RQV`GV2uW@^{Y6{{w1>i%J1WZ|RpBkmfvHdJ<5xKJ6AaX79L1Drs^O)j3n9#{QrnnC#d@_$I?)$mCLmmkCtmIW(nuw_hk!TI'
    'UZYOcC2DNG|qE737Fxci$y**Z6dPu$|MDq9$$-fVg{0k4shvVaY+pPS6yRF*TB4J%YOiUeXjO}4-'
    'EjMG<YAV0MRj+gWjaJihgZM>v!Z4FG%MLSD<P^p8dadhI4C3NN85hXbi%WRrQq`oeYXVrd8w9XyHwa+M&<v9{B?`?jsZ*lR43j=3'
    '3eCTecgTb0!+C{Xmr(_rRNV$%EVvqRTal*TR!y~1PP>&lv>Fg}VEsJ8SS-=>fPQTV^s_>s{~-'
    'kWO(DhCU*M9|kZC`eivOmD?f5D9pHoBs_19Fqmm2!7zop{+)KD!xor(`qLk9NuRD77m(#c-'
    'udKPz!4l2TxqViUSM3fqfsj*OQi%pj82AgbIF$qk|)Tl;7n6{}=jfOC-'
    'Q==LUZ<2S&1Lc904D;oQn4ub^0hx@Us*$dm{utSaRM4~!lEB@WNRkVkJ{PZOZ{s-'
    'z+kR?_!qQ9+$Jd28o)F^rk0FkK>EZZJE*?zbzwT~tl_Z2Td?K}2+_H*WpE+wa+>d0jvGu0%mN1T&_IlSUAB~0H63ZuGXF`Q&ts0M'
    'Mu~=@4mzM1YFKwC8gsGXvV`PnB>Zb7+StFR*X*@>O=$G;id9XZ!ht;a9QnbWI(O_rQev(z$;9!hXj;soyiXMyIuM*@w*52+~hIU)'
    '0d7ojTLxK7QpJif?0yPXi&&1vZ>au!)iH-'
    '$wI9_C8p91}BSorCED%Xdf%Ea(fxhwosZZ@BasZt$NkmT}Gtn;GsD%Cb<ZExUCm1=uirP{8rYw}vQ8|1ZYH^^(tFbRV;Jr{@xlQ5'
    '{$bAgyJ34=a87l<j3d$YVl9uOa2w%MDMO%i7$hHO(z&_4301eMcP>2-'
    'K?A`UBv`}NsK?EM!HCZ@<qysS+#)?|7RyCH<w*&)RK6hiD*o;6lID<>vi$ikL`uVtQ<t>aGM4I>|G(c{+Nni!MnZb&J*+p1H~`I&'
    'Bi=SGG9!o^}075*z1%UA_dtOjF7EQ;G=x@Eh;bX#VMVP0lLnPQl?8BwMf=5<DtDgKqbLmm{LSjnxZft^!IQRKgIu^}OgxQO9#h~A'
    'G4lsYjQ$%DDrw?H}DixP2Afx2m$vL>Crki;H!rZ*9G@%>C|7kG~Sb+>H;&i5ed#zC*Pue8Z-'
    'U22P6B3e9&re@T~${J1Rtrqjlwz;xaM~kJF(<=NQn^_za?q5;|x)Ty16rBusC8H{cBg(hm7SDsbEuIH=TfW4>w<}Hgfn2<x7~EmV'
    'K4Ri?2`!KF7#?5HHe><A<4fEqT$q`KP3%#+(G=50C#9OW6>EnY3Z{FcKNqJJsJSY8Tb!0=Z>xWxCl@^nlx{tiW5vo{X<)0gcqO_8'
    'CCjl<LQGM|QtF|>Z(v$yJcbw&*(aR|&z6d%Dl{b{ALW~QIVK;)Gsu1v&mj9zzWF}A(tPj7#Xnr0^aHqfUNQRPOeCUXqAJh&pq|DT'
    '6WSuoXLx*(ctoU}l>YvQaj}1a!tgT_R14Oh`2~qMpg>KVBap*MT$mVL6a%0xTy!bm{>APUW-'
    '_<4v{uxlS&L~JEsD;jr>UfwgODe_)?r_1!GzJeB$jXWDR;(Kh!t|i(Z6)J347=X$9LGi8(gc~cY|wnd%h{3SZT`l<>DD7;sE2fgN'
    'cI^s(N@ymfL96rB@TIdeD<U9Q&(Dm}Xu!&!nN3h_9qX-MmbEH6_C4<wB&yGQL@ynr52~#D2lWJeL;fms~86G)Q*7tGOq{=X|=7$I'
    'MeUt4(Q&M=@eQL%rLUioQr2%%p#HC;T1JSB2}gh$AXwPvkg!yc?{fAMXY$>Bspde0u1NB4r00pW{Yp$V^=wj?y~%f{$jTgxyaZ!%'
    '(qZI-'
    '=rOMylBT#b`#V*aO6Ij8w4)isKonVh@5%G^%3jsl5?fKo!&)u+61g0T+e|PQ!Se&40lHUF)?W8jo}ovqDX3%{OESMf}SvVp%)?<}'
    'Tfmeo>louwsmt*CNPK`(!sbJ3rYC&dyKrP4tCj6a6&S578!7Z6_{DjHr~E^~K`LsiIwPkhN5rN#e+?3#knC%es=t&{0`8(irNWbt'
    'j3T0a*`H7#f)MB!QtpSub?d+#FIh#ZaZerp!=FVpIzokW_z^Y0fKB(X}M?sgsm$QQVbJyYo6Rfjy3#S4kjT-'
    'i9p4<<s3@uKaX2m@7ZcH?NoRkP-yRmBiYFzS%Wgm0-dcb(UEPEvQn>?KdI`7i3kOrfS0B?Cdftydrbus@7cmD-jpyV)H&oQ3Xe5?'
    'WHNb^n<xkBr~chM>9M=rx*QW7#^S3XU8%;zF-5@eM~Z_KZ9K_c?b6}acF_cijU^vMwqceEUV++-'
    '8sJ$BhqrtkK*FKq=H+9@33PxI30KF2B+hWd~^P@N~?4yE}m6_HZV{xCs_LWV`EnV5uIHo!uzo!#OZ_nGY8jPd2xNjP^2Lg=Oso;&'
    'UjjDxbulh(RgyDz`Wz}rIZR6vy@aBR)`Z*DH>u{>so}2ZMQ}HJl22-{<0;pAw0WTbK`ll5=p{a;e+25_u*4+?4Xg0DmC-'
    '6j=jPL4Bz~--'
    'C#caY&Vz>Kg(IWwSRe6ltV@}o4Y+yP0V3kK*5zJX82wkFw6OWynP9n7RA*rwS36#5*OTuVH+60VIMa{gG<t>NdEl4A(O-'
    'q*AW#_5XTL@_y2FeC5roQOiXmxM@1MH5O>^h$9)vkVK+O1;ytITPE~b(4bzq9xi{0#JYPp@y1zbeSDiY```+sFl@k6t{KuI*eCel'
    '2wUk(YFjLx{ySEuWHi$6DbSux)6E=vjYI>NiyI&|>cQPD{?b!zmkBjYJ{)pjmsXhCc;W3Rg6_O?^sZgh$VCo<pDTpBwJeQ{tu)@s'
    'umzuV{vYJBkoma+oiFU;yalLxoa7girLi_w%MNCa0n5b36)D)U46lw%0!(|oGq8Y{y=aAzHZuzKO8EcKps#ak6YpOtGJkk`0j1nz'
    '6Y-9e^g;G7@_<kk}b+T1`tw!Owy**mt<Ao`--j+q{g(<YYz&$JAtK7+IVq)hmPP>|iV3s4{j#;XQR0l+|0-hIwQS-'
    '^`c+d)|7UgmJn6LC+EEo^RA#q?Fg+n4FyUgQVBVr1!!2)YUOu@CeLZKdj-'
    'M8D7qNTBCI+l0b|2Nv=j(a4Qb2PZ}>y;R__L4}jGSy1>M<EnfC#zL2i-'
    '6CQ&sA^c;ufb0@{deS_7oe^KTmfC6E5N7MJZg~p2g)wDO^rv>TW7zNfyHX`4p0eNfv8$7~R3D>SVGuNDkG?q?4}XuUFTW=Z4y#{+'
    'rBmgon+yct5t*8ZPyIY_0W5JtjhU*gVs<B1U)^9{XAmBRp)bP^kKU_cWJ79d`S4v{Kd8$*0Mus)x1iWUlu0*np20r?7EH78@6*u<'
    ';7_ERU~o!;_~2E6KxNQV5(eQWRRFelC+Wwkt)QD3gt;WIKcHl-oh4dT45b*6V81gE*G<)>HviZEB_9&%~1TO$jH(Z-'
    'x4^DXNA&U}GLJlzC<!#3B)H;)7TuuFzu&Adx0)W5IQWkVs1|7<pYGB+@);u29H#+#}n;JWir0K!qSTyoNT}@c-'
    '6030QFY;V^#w7+uliBQU~spaxY<y4##!3?DB^!T8QBj4ugc?3p8~lYU8$q@OKCO)8WA=3{lzGpV+P)Jd;ofcZEu>75L;#ePYjWRR'
    '^|P5LGW#WL-kA0lvzP=7QG>D{Gnh$=5!(qH$(SOmgNd>D(sm-Ltj8Eo_L*Nd1Qda%!W5z|9&u29HJEJNUUoMtLqzHJioaunN*;nt'
    'Lu*ZCFsv*a^XX_2%lS*3c~6?rOh*VvYL%rTxWs3*P#lh?Kdup0G=_xIR^kC&#fdsh~_m!`1$K=ObWc}pI28gH{zrBvHQEb-'
    'pd5QwXV;mF$!AJT(){Uzq!KZ*q++{8z*KwPHBM5qm$C%Hky(AvPm-XLOVZOj!4d5C2|oT=Tyu~ap--1pVvMtM2+dzqS4o+#*dZYo'
    '!0o2t8k2u|>ES_)2gXW=w0GZofQO<TWYFs;@7lOgEcnW=E7PLt@VaF|Jl;#4?1*+5g_2wc|9R5&u(G$oS0?t&137jpHw*)D%3GzE'
    'XzRA|2L$FT^6oA@{ufsgAk5mTXg_#309La@)qsHxCgp^#g+N4Xw57H^HWWcp)_oV7z;RgllbMz$uCh#8~vnI5alxyw^Ke{+gZrw9'
    '|etxnFwY}!M<S{ROw!i8Po=qy~^6^<^#RZ_sw)t+6T`zr|6@bR(~tbd<{^<^noW4&F6fa|Fvk0Pmirz&Xp*erFwylMStl6eI{`d&'
    'YY#U$LpC$X6PyB<@#bKX9=O(Ldu9v<!{5z{+wu25)J<XLXpi=}#uIKY$suEQPEGZ;@E>lW^+lpgwcmMJ)E$1??Iu$xX7oK50k!I^'
    '>E`NlEDDzD$rhRMd6FZHz}zYt-H5YxH=CmALT-'
    'F6=CViy5dqRhxjMpV*($O=InA2^S<#&OSu`Xneo_p+HMH*oce$cf~b2evs@;9*Z35?AZd0f$6zGlS(eM{zU5Gu<4;&D^N35W&qnN'
    'nfFm#&uaLtOg6U#?ta!;j_S6Kg#Y@yThz2EbD`=`|%rJ<f_Xq>d9Hij<S%uTuIPW@jVHfKFVC}E-'
    '}Yq>tFVg@U~MCa1Vw`8o>+mo9qk2vM<c6><e>l>V+|0?4;C8K3POE+071B9$?G&zKKV6=jn!f@;A-'
    '5_&S#U6Yk;bSQyvhkcehbc%ENJi9Epb{yIwJVV1r^L^J5Q`U-'
    '_aud^(+s)eJ}wWD@IS`yE?xQo;im+>HK8Niu>>CFNUnDSaIwyOjjNdx?W&_B3TswI;0^VP2Vqb?bsPGOY7AEZuWl(;|0b3i3$V>Q'
    'mh7KN1z<uw9apOnQl&*FM^7T4zmxYp&IzZT_5p0-Hv=7jU{8m?Y2^YLa_IP-'
    'i2bKhHHkqkGnC6>tc8$BkXNEjaB78z3{43Bt=j42YHt*;PKBz&H}LLu4fyhJtX1xuoVI##Fy1=Z{JWGHYE;OFElenb{ObF%oEoSG'
    'Mn7dwT@T2yb7rNRGn4AP3Jjoi^!v`SkGSdE#%Mif^F{fZh`O&X|l41TI;uh=f6pxAB)O1te9ySKrnml7|zj2c35Lp?Ar9%>%p)>t'
    'URO>B*Y^1t+$2%>zj=GG{pe6aS`D5Cs1`U(+5`IGe(3Q1q*LHP`UL3CRg`7KlXyL$IgJZz!t#ZdJk0QM<aV9PA9=VpODg=f?ydvN'
    'uh4CEeqbM=v=8jkarS}WnV>!@BQ;dfX{hjKv{opsvhwMDnfH~{r>5}feijf#M5LOUGGxM`b;V&a<@`WUc__}KUulZ5(&<#@{G!F>'
    '}8XSj)PV&S|=kBRV=!G_;N`O0A9Z=!tVx%vtbzVZ}(g+lVzW#xcdnc7!a-!-'
    'YnfLIfEwY))daBqf+7Xg+}&0@Kf#qzu?me0>(xi4(drRhhdO0f&=)sJC#OtTKbLtF`6*c7&O1pK}=#P4fLhFW|Aw>DsCo#`1`v5W'
    'LLt>?{RIsE2_Q}d@n%`#K--dy#|+x3|T_iZee;U>O~#qx`KOazv}hTleE8EpJ*6qe`dD@0)Ve0_yN!q+@c6|KgrqE!(bg4T~3$L?'
    '3AXVQdjbXW<cMcxC_h$A?4#B2V2hN>CC3vpWZg=n9BA?9aah^f3>I(VgXk`6&VG)n-'
    'CJZC#OM&Bt+t(WlI{HH6J4V1+3{w)Y`yb%^IgJam`{wUAIFyD7uERNwOw#DN3c|9h=#ehe@Ey~3J(`<`!G3M(lM7S7J^%V-'
    'qT$jb)juz-r*an!s3{@=xoSmM<*-lxUEy&{Rf-KG|VVf@lji^3ceN63KNh3Bz9bjkPBe2<>aCZh%w+Z-'
    'd>s=U%vHvQo|KRD_L&o7M>E~-*&P`e?*n$Oai0>oy^i~n9fggo6p53|byfJkA{qrjP%tQSy7R+!5-'
    '^GG?sU8!d#RK;JE=r3B?EhVq7S95Gg$OO43-lEV>09Ud%di9vFOiGd%&@z;g?rWpTW0+Kq|Lzcnj9lFX=)73-'
    ';*NHKf_wZ#;|Oj8W3P@sqsoCH56RjEU+JXv!Kv&J+3*fGV3d;-'
    'ZPyf<1ki}({X^0Wx0%|3|uawDFc_wLP>&m^6Xjxpe#|}?_*fLuU2u&T=sLVB=ts1$+o<cR>uVRv05Ex!jIMJI17H}x|Vp->X-'
    ';WR;%M2_*vlMo!^&gm55`$dn&MNosnC=oGJ_|C~Z)SMeg3cbU9u-JqqnMYQA-bN0TRI;Oao3t~J$5nzA~_y7+KjkkMp%FUV*zy%&'
    'mE_v5l9=unQO{JxU_Cd`o~>!~NJFxYUh^@AQ|s((p+X5DsOo~^zlD!HV;%NL?vkyLw9vj<NevJGQu7*7`3hA|A=jFD=!O#xlSg4$'
    'lqSED-'
    '0SgN?E1A|wi?P{nQX8U^xtQQ^4(%OY?h0<+iHrr}#egix%K{TL)@|pngQM@SQ`R87g@%(cy7PI;nWx8pNVU^crn83K}97uWBHLz1'
    '0?YMR}J90{V*Be!bRJAPpSnovxNz>?1?BsBU2gQ*G^W*{BT&RZd<U!k9sD|_8VcSHAv_^|;6QLT(lf}A;&`@`-'
    'k?`A`P2*}lHmIAk62qgbJ!{YK=q6(++MW?X2%`g6ADS7wPN+w7iX!yaeo4kNw7n$b8QNYdX77zTI(aRg%na5n@Wd{6mTJ&GT6cN@'
    'D#E(Znv`SbbUn%D1Cjxu&A*STxIriH(WqZ~lWYf{;eXmOF#q8gl*lDkfrA+y4{)lOU1Ll0lU2z&3RrxeY@oXQFOrQ^BK>8uiE?h7'
    ';nB(3UNAg5M{}AzD+2HGCQcL%yGZpw4yl>``Y+3PO0$<`Jf+#o#cco2vdwZM{Mfk23hAW6RjUL`+j`JlO1?I<wnUtEV~`PV5=+&a'
    '81wZ6kFccetm}~mh8Q?XZ)W$QpbQ=qXd1=vSY*#&!vV))YxdR2ddj7u{GC)w?JM4!qt@vtyGyx}sgvAR`9*O0C6z3g1=>tWo;9hP'
    'es?wD_4G(mu;)a%wx|BGugG}fuUBL|@z*QGEWWKgEe>-'
    'tscE#r8VI%~k7fmf|A*><>q(5L72KZ79UWGc&Z+yCd{8pTE$$seQUh?YA_XEDtai00okTK3?PgCpi)5(!o;~Rzl3{9hd(u@T!_^-'
    '4q?<@a=roKbkr&BGwO5HUFs<Zxf0nEz60R{^-G~$vQz8klo)2GZBx-'
    'k}Sp)K>bo|1v%6MX|S7kgg)~m&g++JjitTr<|MhjOa7C1Gd1%~+>O!<px`K>_Hb*Htv1zvOF$QjMkvw9u;KaPtB3kwf&)?bD_#kQ'
    'RHNe(W45A*pc4nBUR`TR5oC%>op{0s*#znA&^EC)BgxB2`W2S2}$`TRTwN58M@I;@wj*H|(<y2vR0sLxaJ0ik9X5B=9dJ&_Y1(BJ'
    'Mg8P6W|nv7?UdaanLyX~3nC^?v8kN*F1N%)Hd6+x?x9JbK5ldUs{jk7II>BeD?Y`fZ)AWf3`oYRDWjJTfjp_PX*r-rO-'
    '#I30&lwKT#)VtX~b)gpY-c!cWB;6LgjL8D^t})Qhg?a=H6i~c`G-'
    'K$CdjU2{9mQj?$6>UO?cz#3$x%*^mR7)^tRlcXgwe6RcAarZT(4ah91@|nqYbh=s$>vswmhn2@K!ybP(^<k7LVj=uxrVL`IxA=V4'
    'CO2Fj~U9@?<!rN_XbTNNQGk0Okyf)-'
    '~{lJPrrrd4R(UvpBphz@ctbuc8?WlS_L^b*zjdZ@N8P6ZR46N`u4I*vLFRl}8PPC|j@}#y4PN(~_;X@exgmed>MGI9Jl6=`PBEXr'
    '986u|PZ&hr|K#FdPyA#KESpCI*OLv=^d)c$*$jsCKmsh_!GRlDw9_4C`R~G9QK;OHGt#5hmAiHN*pM2%ZOkyC@5|+fx&tP9dq8wg'
    '9g`!{d5;b_K(ONjF9AF4dS6<2X8|9DY++0h5((%BsZ*@0HfQ;zWyLL?dXiy%n4JP4p>>uj-Bv<e#H{tp@qST+Pej!my9%sMuoaZF'
    ')>xG4%&}O!2-?W3U&a+CIUiFGjU}-mV7}s_!qu;~^ZGul~2Ibh=o$?xdsP!P^QvY~?+N_N?nTY@-'
    '|^hvIpF$cwXxyd#Cks~8@_qV8&jhm@{46BfeJ_N)h2yGnJW%t-6B3)t|mTp1n4&8}Clijp?Vc_{a=nfszA;{FVYYBz(Qsv-'
    '6%Ol*95B-_@VBBvSIJf-i)g7RrSCJvNO=`j(Y)HZu53Y1{hm!d#<haOPKbKI+3LA0H3W&)>PS4kL9)-'
    'XrQ0(QjeVn%87py8w)%Ql=~>)Qm94Os+!y!A5IYXK($^DsOQz<fy-'
    '%y*{1Jb>YGqn%I>VtCvlTnu43Mkrkj;Z+WMUZ>j!M>`AjYZ)Fp$yoNUyTVx#Hkbn>HR3InISF#}O~=Fna=soD2gon=nBqOj=DEKd'
    ')shURc{!>j`A$8cko%YgB+eI`LmEacEJHQ>qn*S|;B?jDHg}v`=>d=}dL{M68yjGH^~f8J=K=CA%_8rv6!H#c*njf1#nT}SgGyVd'
    '_@AklxY}yZ4r9nd{0(;z?N_+^*2l75<LWzmb`8U$y>tPrJ)}BX#&Vb49inokfH_)EtiKX!NnUfFc?w6zqVh34CJvQb^q2^x@8PCZ'
    'CZ;zJ%=SuDZ{A&cKp|gpuX4f0dfHFwRGzPc?7+Pjmek{Eja}47GrvaGc#qD*awAtGJS<1xc>v34Sy<lfVQB)XhiTqp&*23#-'
    'w0E>Va|`xRLBRq^g2;D_J^PMse!bb$QlR2&qpqzgV~fgR<+-'
    'UXxvw*s|*^Oc$&9?&}hE=F|lY|tjENm@ew^H0*xa~5mXEs!CJ3Iq491#ppYlIS2>$a6Xh+#bfn(aP(`oZYkCTY8b%$Xn>F!Qdd*g'
    'wxEkrfI1<kTFkY4g<L|Rztkv|~o?s59&B0-D@*$eB+t(Q#s!5-'
    'HCJ3g>;Z(egX>sBf9RNQcQZw@)O!!6cy{!#neftabJA=vV!IpWLG|%AJSWLd6$HZZBs~!`9No|qWqA&?Ydo2o+zt;l_Ig%n<JuH}'
    '-EaA89beHNxsa3U1J>_w!{$Myu*R8G0>JdzmnYy)BnW<YFRZ@35+VMAay6LA*h-'
    '0{A9@cdO_X?5#hc)*;fn`A<<!HN7NG4k4I$A3Xd5w)UP73zp@G`;en0~=ZOo27aq&bR)E_onKHwbl*@h6*Ex(J@CMD+8U8CxdO6^'
    'F!?iFCsu5!IIFwj!!6!O%0Ksx2q!0TI=ft<tK<wyhglmN8i=P@SUII%%~Wv0=$lTaXMK_P?H)GN@=w@em`XCS~;N+cPy)!f!{Yn|'
    'd97kbNEA%f1fhW?zRjGHjCD!c~7}BMY$O?8%d%G>!M=$w==Il+^Dfi9THc)3+ch)dv}h*_<b$`0J7f0~G&4X!caonRIGiCPjbk>t'
    'a!SI1Y(J@hBV;QD$jwE27L2>~mdIndP~9Kt!448foQY+g1)0IYM>!2tY3tk9QE{hgO>;UnAvuC%L_TyAsy#c_rt2L^{L!AS{?6WP'
    'g~2?E6{Bo|lE}+7x6fxvC_{s-xhka8o#Dor#jVHFPEpEvY*~XW_7tx-'
    ';~A4=<^^T<6mR3{{%WEzbe(nCLEfC_wW9p|0h6PKF6#CCf<x>nC!3Y+?{@;riIb;7UCvLaol+ScF;~81H%+qgHpG9uT2cw^mwd*|'
    'udw%RGxEOdT%O9>ka^Rh?8Pd79xAE|V-'
    'SKgi;8au%2CQn*AZ2`6ZP{PWBf9VUa}za@iduhxerBg_r0;^4>{7V0oHS+CRUwsHaY=U4<vjdY`f9MKHsc?qg<YiY{n=nXZ!+A$1'
    'z$-)50*9q0nI2WT$KYy8Xp&#K5u|S4fxFHtEEA*HM_rly*gnI$jz9GuJn5+jxxEJfBRhDgAK~!YB_>U{8dh{Xgv2n|Kfhi-'
    '3TUO<_r@@Ctzr({)t<1*l5vC57YFE2)o2af4jvo{S&V!fZN7<L-'
    '!|cm3CHr!$PrV$dv%`o)?HjJHW|Z1}R~H4_tVI*UC?0W9EM%w^IYmUW@FfoiMV6C<T4+#QQBu3-'
    'gj~&cy)hQWa0@raqWC2}CIZFg#v)J*X1Os6#Z&Zv2o$fERyMY6@z63S8Qlp>msf{Pbb^3d<!V9AqC5jRRfvir9W~RL@5a!<cvz_v'
    'R#g_OA7!z6eio}6QdsTFRbT3oOBU#Kf&DL=InKGZ$Z;!&Eh6m?zvb$8jJmB>=#~aA`#4xjltmHvC5wWYc&<=ao8;<)JeNJqJcFBJ'
    'F$}kGQ!Iv;=`j&5g}Jc^mjcXnQ<O__z8(<aQf!b`GPZ5e(5y?5e1Sb~CpdM26!LzYMc&6*<W0>YZ(|C1y}9a5;uy(nT?ar~Iy#4C'
    '|Jm0ZwqR5z13S1Y8R^-5$JMnghx25|Tz$#nR5y$B`BC~|1`%T6+}w=KM&@jsz_-'
    '%spN%htB){WI@NX$%Z6Q2Ms4F=2Hk_8X$I}n<=2!^B7rQwY!jJ1Q5xYEdV-dSNu=~wXyS%A-'
    'K*TO@qqI`4ZA*sc?eb=#GC{CQnpC0^PO9jvr;1KhU&8Q~ov6{WPPQHpF|L#fSD*%8>XkS?`$~M0eI+i)z7m^KuS5k81bljN)r%1w'
    'VWX~^AyGipkE!eI$sAS9)D4`7TLv<96N$>880pP%q=8$*NboZFGLobhcgd^U*H7S<SRBI*+!Bl9zw0p(Q?a?Rh^ZKibxYJ#e1RSi'
    'F%@r;Rxh?~(a@|rp^FKq2MgBRfLfY+eiB_sBD4;KrsC&Vx!GvJwF1S5LICTi@Apd7Emo_UN}T*0?A#Awo<RSwfEj%hBlW8d>S!O0'
    'SjW|wVbjV6uFe(|ZCH(>Ch9Gl%6*)vQ%Y)>JzL7uX(cs6x{(iJSr{fkVz>og2)py&7iu5jS_)-'
    'OL$mq1x`H>3Td<}S%`I3{DyAD+9%&6WJ1a!E*~uq;)|)37)R5FurnNN{qSB2YyOk)SbTCs-dYhwA$Dz!#;UDhPR%^IALj-'
    'B6ja*H1eMd%>>I2(?xW7~zTrd8oC|{{wmm$m40O8b<Pb{fL{vy1jDof>NZ-b5R#gHin@cyD%dNqezj@`{y9$boN;~QLxX5$-'
    'N%<w0c?SWB&tOR@Dae^AiTGGwbRle&CIk;&38m)WzR}e8W{739S^t<3Un0lY7H>?Hj7pgBz-'
    'qO`<JjI=a75uYKf`a9ZNh|zmQaW6(#`w2>Emso+$!YB?)jFF-sy0=<s@DGX*~o>g4v{Vz^pI3<`d7YP3HGd7N-'
    'c`O?xpahpTblZ9gR#{;62DGI5OXGNGYDbZb&Jfzivn|<DVwVno6-Szd_w<`-|$Sr)&)7qf_#ojee_$5-nDH0E-'
    'K&G98AvHG7Y#*RAE>(Q;Am*t2`Ekx?BdopR|muKL*j_8mh<yRUGSJWqR!z@Kn3wpv={2%FcOFYp*8UEjp1B{kd|dlBlS2Nf8sgu='
    '9Y2j!0y!gieXV(Oc6EtWR8;9h1y`6JPAnp+=Qif4QqT8d|U8(Pfx6U)N5?_zROy=y1a84P<0FL$Y^T}u>n{T)N$*#F~7s-'
    'E#?`Grt-aI>t}!Q6sGnZG41li&EB-#t;+QY}r%4!&pYJQrE?)nVRb4>9mFad8jSqA~SXVJS14dD!vq^d{>xo$NU0sst60wLD)cx-'
    'Dk3SHf5R4A>p&Vuj>W9l$NC#5PZ6SSg+uZCELu7;RWFt4}R+@0zefysU8;bEM7O=iRFe5fgSB!B(Nx?;1Aql)M8b`afXmO>4}znE'
    'H^Zw}i|4f5g<=_Uw-WH5Yy;T`}IX9!EbH>NZa8QnQe^OC4<g;T(a-q4w-'
    '^t{#@^ZI6~008Bh^7hYlDC<blI4=~um=odbUNbJ?n^dBMTkDep?i!^5q+1`Ac;iY&YvEik7BC+AcY(Gup@PaJGCjA3kKy4)>T~aJ'
    'od(K<rJ?12l>ac+EF0N+)^G7yub&eozI3qf_QiLj^?+Zuo0>>ahy@+a(NWO4v6sMQeNN>r7SWt{9#8Xp6e$?cOD&)3pry4cznjKR'
    '_Rp_GDgf8j~Cz%vSzx7&Z{T@txZF<UK5>k1Ky?JsYO7ZktBTDh~TO*2D|F?P8*K=ogR{;CGxA#Y=B@mccO0{0-'
    'gsv@0GzAQ(Y5JK^$4E2kOM!v0=^@8;wQzsqWd`n>{Ts+Zb22Pe-rk~wTGD<KS1PIArId!a*?&6I*5jG8?FyX}cO^iL!uGkEKh|C1'
    'F2;ts@6k<Pc{tb+ZPkAL0uW4KYJw?M8z|B=ujK8U$2qbTPg^yz6i-'
    '_<vKR=ub<b9FH%kmKwx!E%9^yC&y8?QMD<q!pBGhJEO<T_l8-ZU7W(qux_tiSfQbDVGvL%a+sj}|P`1fp6)z>WPL3ATqSW-'
    '!JBU@S0lc;IFVM#Bdrumj7y@`fo8%z2S4as*b>04enyJK>xSz*~M9JU?#W1$YnD^WCGx{*h=19rqAaqWPea7cu{iZQ^gGNwZD9z7'
    'tULa|W~DAZYy!6G6EYp}gKT?b*M9s?1dYNT4<%v`Fz3geb8IgiT3Jgr8GZB(_AqU+ZQJP48pbM<5jl7GlTa#jkG(*$f>;J24uzLq'
    ')p(lwm=Vh=!$Mrx`#QeqX?Pe>=7%GB%B=OQ^<Gx9|cV$BYTZQv=qV}hixYX7*DGY6S(J1Z882jh@9EFOYGBE)gclfO;Ih~xf24~P'
    ')Qouvm9>L=))?ggw?*9$xdD2H(MR0=46%mQV0s=4`Y*xd*#%bcuLCwZRdU0?#?s=F;a8N%@BFDTa0NFGfxCz+$JVov021E+e~RU$'
    'c4%c(>WYt8Y*T9ywEMSJrYgVfj(suwlD>>lJPT3nn5UdmfZP7)(URmT7Kba_1tCSh^tofEjX4|aAWCh_~tjszt>hS^awq%rgDQ8V'
    'Np^?-;Oa<(2&sIMT4%H$w$E2xO$27$-'
    '%UghNJ6hiOKB6Lm)p?3=PxUj_`$p&?*ROegdZBnO8bpZ>hoj<k?`>_m<Pb^$#C$rU+%+Y?daue?Lh1<=N;s-'
    'MrdG2GdhIctn8mKS26*uFQ`_yTJCQfauPG2jI{%dZyV!zMSf0Xo!y>CgKoad>T2RSDek#G}pViCDskBOKkjal!AnkVno10v?hIeI'
    '`L|8eguBKxU>8IgSrP=_(ws-FykT@WpSHwrumsfTg(ObV%g$|7}c3aPgvYn!ErLUu`Yjr|X>5xb33ee@Kj&Jt=*PE0kZ@H~o#9sJ'
    '<wHcQVY=-'
    'Kly!W^WvP^OIgj$1<;VEB^8AUsK1FpidNUbu3PhreQL{xExy!$fW#+1yx6!cEML#pH*2OoW!WG2optMoaupdO(Dh_*^}pkRMrw$t'
    'rXsEd|3sbvPp$lbZw{$9vBFXHz)*a~6m5QaG$+>UK^QAUzodI&HdPcdoAHYQK^TZeGLH0i{sDub*&PuQgnCEKP|dGMgXHXy<T4TD'
    'lwdJp7<EJq(Qc=%-`o#SmXsj#UF&U{<z<=I^9Qw@#D=Gt5Jq7mGyrLi1vg_=X-6A>V9Fc$bXnmHx9H5Ya0=PY)>MMP`wR@-'
    ';Vm#Es(WxfJ5=%OY-mN`^6D@$3OVwzRnteukKjJ>kbTlJtU~Vdi6R_!(|K_JJQ$1r7V&zSw9I$GWy4BzpRix}e6DR9`tgCA`yQWK'
    'dCj+84tjsv3QSW_$^?t$d{{{%7AAiDzc5&w7mJA<mCQAbhp?u?T!ekBRUYjY;p0@)+;a10p=e`FcPhkFjhn?5_@CMD>74H0r^-'
    'Mc_fmJc6s|Q^>qOi_8TnWKKvXkn4wx(6m!QN@#VpoiAZ8ek-TVZZD?p<dy{-'
    '<q7WQy34OPDO{g^0mXM(#Hl6_ani)zLA!0hs27mc>)^A1f9sjPd9s>FBN|F-'
    'MILE{d1MP>aR}dPK`ahG(PJWJLSw++N6m!y>j4on;Q~FN&`elXRxqBa<+!VCkS@JgD?ND$oQ}_idd9%NPz|227+KOf@KoCqGOcY+'
    '0k=c85)S@zx)QEJWGdkrf=amRWN<QA^$XxU##f2_K&nr}Qj`Q4&4Y-'
    'wk{HzSxu)Z#`qHIE&gW{6lIBm2VcP2vF8tSWb*zEtc&uQ^*<72)@l|XKUuPT=S7FixheQ;`b<npJQ50vsM?_Ivg&q)56xV=qF~wv'
    '{%i!FBhk2zI8Dvhe62?Otcfq!RMXT8C!hK=qE*$?(Dt!vx;nyiVcg*7X>nxsoWbu4q3ePBtS7q@$BpIUm+s{Ljp{m+`9+nJK1MKJ'
    'F$#6B$ejbsGP=oB}k;zDP5Q`(|s|+_i%drU0bWuz7US8<jeEGk`qW4f75{KTya7YAtJL=ntK(F~85$Nrq2SlJ3to)aPYNeu_OFXE'
    'C=w9;eZDEgyPWD$hCQhTkG>#lQbk{O$DQmgflEQAMEOxhKv0ItN?nT*NzfT=&YK3lEnpSet>yJ=;Am}muF$#xJF8g{-'
    'BJrs8Z*I|?!B5yfJA$c+x=SjKiLYlk7az@Z+7CWm!PSv@bFq0Ge~m3shMV|nY>Dy*dQ1eKJLwyXz_a-'
    '#5qPfD10wJYmj7!($<nf7RjgH#E_7ajP1JVAtrw^@-k`eMs;2SGw_SAL=uzy#-'
    'DI3DQr6^tTT_7UoCWmOETDU40ex{e8Dr=Kj+cD|0)_j0MdNW2Geq5;KI8<$l#ORo0kc9)<mHn$P1tcfeZ4|3f5_B-mW-'
    '0LP@m*Y&E`@3Ef&mh6Mu^Z^HX|E1eiPP8;bz5`6dxy?x_bvfEle`P_(pcaz<y^&(00PyB50e&80GTk?>E)TLP5vngGScFef2fsSQ'
    '&)IIrjGn-rY8Wa0cx7S6r0aK0o3=R>(xSN8z^p;VudazM)S2)*jitC*`!DOnLAESQ5^>{5#5{_7n=`A)9R((`iind+Z6Et{wC_gE'
    'yuJ^Vcu$zSR*5lHT$Z!7}I=9@$yxtAUgfn+fF-wW!Mmb)2<WJmvvETqO0wiFUlHIrrudKa|;>b2-'
    'ZK!&d!<xE@k2ClwM!MSS|&fjL?+&c^BOS7a!dL^M5_+XC53wD9%%GIgVdz$j3L+_cqWD|C6(o}PCoPm8pG`|OsG0CfsnL44A&lw0'
    '`^B=KrhMV|DES$IKF%fX?s&6a;&gPp$z`3^`5CP{)Q6;ySWNCR{weQfwW|!qb^s)+Tio=xAc-OR98^}7~1g?>uAsO73!g04Oj<;o'
    'T+$W3UX;~a&!FEXQhP^i-xf}Lhg?uenGqroj*Mc?PG6-'
    'c{!_=N3bK$j2?WKu|;wT9FhTuMuX%*gjoi8~?=2H;DcmHQBxZx)L84K=5^q2^6chff(0dDh6BEa284~PKwG?X(hrej)G{J)8-'
    '?@}n-Ig7&YvMB7EMd4+s%^SGa<6RejN3MS3dZ{~cb&~6)?#$K6uD`zvSEn!v6L;h4G-fp-)y`a<p4W1{BXYr-'
    'FlLNKnjt;aziuM%apRhF+&B<9g)h;RDj}}OG+iIVw2WN6uIw2t^Jxp=>HRAf$MD_$6^rAodQ1e4ch)x+fn)PcB5>SS4~W3=WvJX-'
    'Ov1Db$D=io59Kx9WGJcPX$QM<$_2l}P+;W*H(_Jx33t)}4{~*sJ*#G@X#RcfItLel#kpg-Ud+|8+%+R!jN~(}R^~G7x;1AzF+6hO'
    'XFDPxgh!gzzwE$W#GHcUb+~1_RZwva{EJq<U1-'
    'Ii2~i2aNBE~m`r=X^6V&^rl6{;|Kgt8Hd77JfG_m1k9!+exnHN(xEi?I%TwUi)uBjkZ19z<C3%O|;G%2YLoNAJKYHa+s%TIOTrsc'
    'g3HE_F{_6C!z>c)vur!PY-l<966yhjHOe1reJT<yz)W%WKJV!&lHg0v6bi$DYuY`B8?s+Iia+iK+B0T5XFSoqZH>-knNS0d}Cta-'
    '@gsx7}vBzxu<-aON<c{CT`*F2gF@atlR|G3QXN9h<J8U9>`$A#QG-%Tv&GqE8cRJ~H?3bnT&-79q-'
    'c5IXKm6|NnzJheE)CEEvC|tQwM@}6ul?+xf60w8X;&#)pYOH=gR|oPO=V-_Db}+q#%T)uu$!>Y72xMu-XtICrQ#ShA=z5808A}OI'
    '>e6&C{MqUwvr;WWe=f)F=8<mU@vM1Ucsy&~mSUFwqRjH&*D|hT`GKfQYWM)qhD=?BYYpj_sVPG3C&Ie0{jEh|&Zn8P4yX0I1-'
    '<9BI#+Cc7bHq{z6-J<J9h>7yq$-'
    'FWZrom%8t3<V37^2uIF#ZtskU9D_AIMU4`CNg=O7uRO4N`sTFU%DwyN3jcRxEs(*zVWH~lB4|6MzCz#vH;|b=r7PI-rGMl4vptt$'
    '!_%_<N9hXMOUXt|i{D4WN<FKqtdOE7Wx}=vQIH^l|J2IfUq>rahOZpN$m3gWEhzj-'
    '8ZH`>nth{)?4aPL^k3w75^Dwm6h(`4#*R9x3)19mh|FqEtu=N{m(=ZK^${3xwn_-'
    'o=%qargxEynv=lBheXJY$?$1|~gQ_S4|EHn4fTD_Iby+X?`(W<UJIdv%yTsbw3JEA+-'
    'D_ll$hYX$aje<lA)l_WTF=yaWj^yzFnX}uun&4<Faq+uQQr8ADqr3eIa{B8nN)*u;c)Ecrhh?Aj)Ln~fQw1&Y_$kQE)vzGV3^)jM'
    '^30MjHmJP^@9xK@H|cSv_RD+I%`^R$#}k-'
    '+%i{^mzAa|@ZDpoEhO3*s>EFPmR$UZ&^y*@JHioIQxGB*<$`6>RVHW33Ww8<sXC*us9>bXv8$*?Jdj>9UgA1#}yK#vm!7+GRmrB|'
    ')z&%aVq-'
    '0U7U$A9Kw_X@pU+*(EA`>IM>aDst(FXNbGEcKG276pJze=lx2D0s>q|JB3v1gn0eJN81<RQR3+HE|Z|7#nM=l|MP3<2$B76Hc^ja'
    'R?1Iuo{FLHgi1MvrTytMpx~-glL5>(mES>6UC%ADc>a^SJ+nqOsJ3YFOlJEDmnt>I|uVW2b<-'
    'xti!0^6_>qlh}F@L1}eI&@q~vr9L!!t$SqBDOAa9mPljRU?zA?Rp3g~1D3c-'
    'BuxhhER)XZ({V%_Ew7+#PuAe(Nq)!UiL$=q@kCkQ6*G9ZeX=T7HNjGdoM|=I?=evtPc?!%e5SR$Qgtt3Pj9m$`EQY`1Ix%<AIz~w'
    'z)IME!cau^h!SZYbeF_xunGgBLoH?I1XbzkyiY+%D@%oj+-'
    'L7Om2(UeypNHBaN9|UZRZ)SOlYizXX+`JajK_uOLr3nVD#Jj07h1(M{!Wy7V55bu*Nqx^;K;D-'
    '<~)muK%wC4k=Kt*wNmZc|bM1NWEew8y2eJ_A$0quh`iRFjpwlL$FV|o{*uwF;jLG@%ZVursoyeMdCiK!Ic|~E2-'
    'sJZm^4`!)XZUaIBSKxBH$Xey|p;mrJwWa>uz<tzN*6XWLKuB6d7m{p^>p<JktZUdE1RyHdS^9nW^5dKEjKZD{K??0EiJxI-'
    'q4GS9$eAUQaT(I-7dV`#B%&x4}5_FrO)XW>eJiN#_q4vD~`c@Tr6usE38$5j-'
    'I<_d*sXB`%)aE>J%HDJ!>Y6%PK?h%SL<Ty5PH3^rID)J7OGYd=0#zlQ0_<5X}G$%={{%X$5nTKH(omNOg;y~d~nDFX^I>h$2e87<'
    'K=TLk0F+;|mQTA+KO}akSRQo1ptKOW5*}h}Rm+;FKxt!-A$vqF4LEw-qB%j(IB->!?x+4#h=Hh>i#UxznudxN)-'
    '{>(BpfrzSNE9fCaQncDg3?@}P-'
    'DS9sm21AMC=?!J4d1O9m<<=^?OeB&mS?QjXAtz0CAS;#fh36T$D{*o$iHC$ClJ_c8}HvZoQ;_Y|kb$JP0kqGlCg#49%kD>GDPPVX'
    'mHK+7Fvqf^9@76P;=*s0u_<)#bWbdMuT-ROzl9N9j7#^+wMq!6r=a$)ZTZ?#u(H`L2J91t(nUZ?WL~mmU)VPV?l4MuBrExArLtPI'
    'HAq?&Q8%R+C#xVdK#~CEO|p%;!d^vU?_#l1>8RkeXD@iB&q`*!j1w<tkPC$mBr-=`&H*=NFC)v6ZXwl8O%1OhLQ-'
    'O8C|#^+Owi6sW{U;lvV3pwTorh2cS1m!4+LKzmpg>(7*9eYj91Ls7O-ohYYAvYp+Phe~r-e~(2aT<Y(!sNAH-M4-'
    '|<qhV2~9LDXFEQ(5Vg+e~%zGYIrzBjH8*5rFil(hpla)qRpc<f=S3Tm)n#jf${^t)gRdr~PZAbFdqJ-'
    'm+TI}AHexA|z;UZiSTmz;xYQ@^(Q)^OV;l?W$!{+}iFBjNN;|Er{wJ?jf!t)z~(XXi6K2$v;K+h$-mJd4X`%W-)oQ>#pk-Ujuna$'
    'L4S#ddcdF3ny2BNmr%sei=c@<lx+0+;3)4UfX*aBiPuQCylU6!I_kEmy0ofEm*m^f-'
    'aItRlbh@{%J5L_#JvrEccFPoNTOrvaLcGDIcplz%SN!6g^J-'
    'Nh?Fzs}UIUa|Krh8@)3Sd0!yW>ds|L^7X(@gtK3L@U;VtDltA4}4wpR1e4Yp0m!NZ$uW3&y~aR0_+UX4t9-'
    'D?aJ}k207gC^YCbH>z}cBgiHN17LU*CF%ft)4{1ac9!GHdD2w9JT%nL_S+;r`GJ2Q5{wk!NMXsZjZ4;JR%|D*uqD@+`!n4K_n0$_'
    '9Zo}=#msry_l2lV*4z}r}!O1MjCJjyIPy%UKGMDm4!;^VLhI#?RgTT0xC)6|88<_>;^W|VX3#0dXZd|W^S&qdW5!9X0F9(R`mi`q'
    'BM7YksVu83+kBI=Hc|aqhfH;!d$5<4I<_d+}NSVWsc&)DZIFPizkT<E`?p-'
    'G^Rat`Hp1g@<!Dc11R7dHFoH|P)u%4^BNTRTxr@GqD^ON~-'
    'b<ZlcvvnK~6tHyDyjR1hvMJ=)S?Lsn#|h8aeyi~84^_mkQ>V8sHgeS{9Tg@)Zz@+yjilZRsdg*F-'
    'i}%FcI5&1I$HyqTdrY|RU{p8NL&?3Cmd45$a~6kSuZ+bU%6CVJFbcE$@6NlId(nkR0X-'
    'W7ToH!<8Gp9$Fii?Tt`Z77nI1a@=Ol>Jd;Cz&*V_;nH&aqCWnEZ$zhOZayUp!hFS`yP8KZHOwExbsW6yPO_Ve5Vy>PyGs_zF^K$F'
    '%kaupkJnNck434$#!8jz&x`*J9BGx^i%(@l02wYBnbw$u(Pzq=R)Ywv(YMW6CH-eK&q;rwl2-'
    'a$m?(QK8?(wGE+nnEBQtdglBZ9~C0(Yqc@YrABx)&VJ&St8234Xgi%SOkW)sLxJl4_3nGc{XM%~5xOKA45=|0!V&!+$wWx$85g|B'
    'R`ltR@9k2uzP8kNaMp_0465L|Pv&H6+sfct=Bu*}sp>7VVGKp&QFJrVE-'
    '~gcK(1+7dasV@p0jI<`b=7mc|gfGc(J*FK>{#ScPWfICY{@}QXht`cnRNjZI?+D|46ox6%8>B9NQ%eiGgtHYwEd6J3{4`OP8Gz&q'
    'U+FO9@OrY3Ds4bjCHHI=Oi*hO9+LGGC3iEMO@)bp;wtJrG&1HthnjS7SG}iR%^_U{2AC$LxZ-BcC85wRXDdy6F?XWf$mP)siMOtg'
    '&e79c<aP7_6IZX8}!Ef`(2QW2PQmyD9rshi`xu2`9cHM0MtAuq(HiI+0gvCTZb6O-frWbMy*U#W;68EvkVE<1`hbGJwY6eLyFx-'
    '(*JORhAw4&Dbt$z^9t+nYVVa|JcBxBhgd8Rk_J}lPsaH(OjrvFfnDPsB&vb@2)!spPdD2F}<=J8-&-yC-'
    '4AQa&La7_3u)IDh*=4z(KnEd^%!YER?rJm`X?rp+FrBMYzxVYUfg}PICTdt&r$bOd47<Rn<EuS&$Fsdz`G3+n~ST<wWVGOin#;_$'
    'GWVwuCOMDPg8RJ!kGpdz}qZQjT&&1}B4UaW3Txxi%iQmv;ikNs*-'
    'rKz^wxig$IoYHx5XmO%o`Hk5+4|+Hlhx`Q`}uR&@_lPRf02Blw%N~L1{o=g^*b@X|1M$CpiQ^_PYJ6AZNBxlB`g&@!O_~^T=qtLD'
    'z+*wpHzdcV!&YDE@+UPRHCv6oSWc3nlPUQ;O9&oZ2&k!s%hnKa~n)vd*zwh-'
    '1mrBQ^Tc3#G3jYJ*J4MkI7a81?10~2d$syLF?~%(5gKT+5pdkHqi5+4e~r_2bqtbCZDRWEd)MGK2uxl=T*rnwbjC5B%?|sZ`vv=n'
    'PGAXHE6i*Bzjegm3^)(XPoeX3)l&@W1c0Zoi57{xpV8Wz)@QmruGBMExaRotO@;c%xms_WUP7NIwNDv`-'
    'vV?C@fx9?DY;)uX44Br-Wz8*t5Ka=XeXx^%kB-'
    'Z1G4eJJEGOf(h3J+)VJ|Z?KARby(E@1fw|ksSJ;SPSCrNrL1kMF5T9kHdUWFnuSdm%NZWsIrTeh_H^M0a%(9>YTxL)gi^*eYr^`*'
    '=~2W`XNYoxo4b$h*6AGIt<$Awx6TP=4*7D;fiyM~o6xc$wWu88BpQ|n{opX=Ef1p4jnZ)k`p_QE2ebX_KHtCY3#h1{!XH@E59{yG'
    'sk|2!;5sVrhZXo2ROt&#@Gq(08`j`oQPn>z!mj8aR^eZhIOZ&tcD{Gv>VDI{{d=K~FW+<TkUVKZo`v-_qT6x~jc?02tY}-'
    '#iDmCIctAfzx83>G|5b8@x8PU*tmF!B_v7j;v!c3cy<eKHWq7nxg<5KxE3}96XQ6&)YXnYWc=Wetr!YJQ*f0n<Y9C5#wq&rn&WhH'
    'c(C=)P-WDGq)d*vryG`kIMrkA<MG6B4EG!Mc8Nd}{u3$lh_88S9J*j0B)6kiDrq}l$-I=rP-{{Qwpm1l-'
    'X=TRm&YfU!wb}=TQmAt6Yt;Bs^95DzP7=D5*-'
    'X!$skOu#f*Zbjg}TZsH~vYetEKa_?i1=7TUIngT~9)RIP<}@&_tSl7%Nc^QVdqj$(SDg$Ouz~2k!iFA(<&k!&C(P_AZWE;SFupgp'
    'jRbw40rDasLXl6N_TxIV;cT`p%<UbDsV;T63N%+?q2nJExmsvnuAkem(P&^Fdv2m48>X=<5oa@AazC%+l13dvkS$+d%J&Sz~g<_v'
    'h+t7w$Tc!^WJp*7XbpNIH~UfPS`6o$T3}42jM=mq<khm0GRSfl`B?2sPDFJ;MH>)fIeB5$bV9RQYq&+zgapUK(zo)Ez6ir$@5aoS'
    'kQOeU<1Io%#Ppi_VvYTXd#ociHH0-HPQ1n<eGq@(ZZywB@PTVMQvGj=m2-'
    'p+hQnd;mv+wi8@<be>RMO9UIYhc=7&dn6(1cF|@L|A0t^+&<bY;vbQ^ncGR5Mf?*|5mFxtwP%S`t><N|eI6`5X3f}Js=?*fZiC%r'
    'VxG12Rie9c9{V@Ca<&xi%9&O+vvU+uaQ!rAF{EhkQles(6|VoTpL*T(-}P5-xc)or-J$g90QHvZzZ<CDcKvsQU^74||AW-'
    'KVgKDBT>VNCwbXnypXKbgs~oXg2TNVy0A`7(FuV@8y3#?hiHE*DlDeeJwa-TdQEl%#ZI65u;mJ99_SSbF-Gj6E-'
    '{`@4q;L<;Z_D29<vh(i1}pn$7F5JHxw6NvxPHOiyk6@_BS({>lYh6*V5rN}(Z0?T8M0e<E4eu90-'
    '<&;g{B(odM{pgt3fqGEdA7xFuZFQfI94@Yj%yh=L+=*BY}t^45eDGwEn?-4BM(jlyg-JGOzhWuG(8hJa87uU-KQ&j$Dwl7dMxPZo'
    '_%y-'
    ')O_xTD%Qszij2jc;N%5CTls?$58E4twp8`IvK57=f=65RCkfGHfF_>7MBQ5R+*>&07Kgx95B=#Aj5KVD+R8boP8Tug>I1v>M5GC^'
    '%M;$wXPu|d61mXVIo;bp*vWx-`4Jfv-Z_oJ#J!alMEd~1V=|lo}>6{ZoEN6Hr{l_A#u$P-RzfctLCD;y)*L=+Z7@T-'
    '^rFxm;nW9E;<miGS*QQiw5nNl}tF6tJ8(`=dfTRS?%g2lFwbagjNkT3&Xh#-'
    '*TJN>BzA@j1bd9w^9X&c_53J3sZ<WgsC^UT5Ctcw&N`q@4a8xS;m28B{Jm@1MDMWdfr-uJB6lzbiPRwmDX}9Y{I(W2Dehp)NCuGy'
    '<Xx>saY#x$U{2i!O=XG3uEDUI1Y(}<0$(?i^9=Zh(*EC42Xau+!8CuTVDppBf0vGAc3P<@bjswyhuKC6+6k7#%7&3)^`*9Ds>Jnf'
    'Zu~z_+6BOA8v`TaNH>(zm39g=Imst=5TUqVS8-'
    'WzXHt*NsF7sjT>&|UuNK<4Zkg+vp}4tQ(mVt^)s`vnZa~tVjP`G=Magc9b_KRMUhCvOI;KRMZBYn3Sp7n5@Rl22#X|BWCj$#B7ql'
    'fi5JxJ*e}%w#<AUMm6>p^WID5(H{sy4ah$FZXO0SX1`k46XlKVL&=T+#8WPSjnoaoUEzWy>fT7yCJ$aBJ$ErPfh$X9tl2|QdsHS0'
    '09%jks#3-'
    '*8vE&OPe^raI6U?$@Vpo4p|LVZUhf=T3L)lm7VzHf0ur~Nsy6~JTNptawV_^xGx;Pe=D@yj(i&jaRM<HZPm82OEQ6&ktBnnEv?Vo'
    'iTX9`uLm4HVFaR3(tt->sO(7+8IVwKO*QqX7y3Q-'
    '51BbMPzB1b+M42!&t)+a+?3vQ$J$xt<!xOkFbY6>w2Cc|OdW~24V2sM=`dXkaq0+;vc!m;Ur@CF|jrtr2fi?>TerdC%IZj3U8<G*'
    '9Nt1ONUrfVJ&&23#0i$u88C9z0+sbs%YQ6!p&B%_dM21FncZb=sO9n(m(on5fuaI`tc2~*HftqE2rY^1y5i6jgDiEp;SFk}TL@q$'
    '(GJ>)mkMm`Yrf2Icztv7<#`0<RS2gN)lB$X8MI5X)<5s$NyUKH>+JLygFj)_Sh3U{27^rcGQt{gjX2%5ch(_#S4hqKVUbbC`_8=N'
    '%V@&ITqerYTK;Zm2z0&rQ$-g;30nnzKJ0-zZX0l-'
    'VKYp$5LxPR7LJc_H61d)(D1nVJ@kSvCEj)K%i67XQV?senX<VwcwgNg9a*o(5po|bYM4`u32uGW#j98>S0_O#8Fh-'
    '483QAdhovDv-QtPO8LGxt4b6+~3vXi=yZs>*qWrhug>pq{J8H>oCgTlPl9X_M1tNtfx7R5?56xs2ukO^XF2Txwb@ARjN;$6M58H0'
    'Ej_<uaN95iTR#Qu~4~V-'
    '}FU3dvMc{KS{e{D&{B_^B_f_)lM0@n1fdeuB@XTMX~au`Q5L+gDeY2dG_~MeSw5>gfvaZElUqXewMwZavjJj={8lz4uT7vGu+TV1'
    '+IcNXKho12xG^*ZZQZ@AMeHo>_SfH0GOL77IVP&SkOi`+Lbgr=s(rdG0$!&4*?{#C!<1v{S+Pu&i=xQL+eJe&gcJW&y9xd<O3h6;'
    'Z&3{x(4c%?mNoPSv%Gv2ucz#a;}LVP21|zfhGfYkUV+XG+*6o3p#PI$Nq!?b+|SI!8K&fqG0(ro+A9|37B7x`%=D#iyr=y;_N4qX'
    't|9jMvoq0xw^`hdhj~8Ang2jR7TNO?en+hjc#I$((vFbK|Wdvif6B91>Uk(E*1PG5O79X#*|O6y^<xIYCof1=UhDr~^)W2Fp_Gb`'
    '3uH;CG@&Gh{$H+AEV!MDA^KnTL~yVWVk4)2JK&Y)pC<Q$09U`pjXjN52kpZbV8_hrxMjjjltN;Vz+FG=sA$p(;es=`ld4UV;cx&R'
    '}??&@_tc0!^fD5lGz<#h!D0p5@I&)<jw!ue2uC^tCvoi0SXn>$Pz&08H+V$F9-W%n<`#jUeTZb}T4~#p(-'
    '9dl%YWlO?)Jfg|Bjpmu<zcV!s7KhYN1wJ;=nvLz;%{+~eIm~~j+z1(p>|6HgSTnj~=P%k<z-'
    '(Lvzl4FJbrBE+BR_GIjdesp@|5~WmbZ4I146bDAhh_?SgsHvD6NPOM)Z8F*Tp#ld*2bD0F10q&?083Oi`o6YvXZ`f?g=b)rpS4Vt'
    '^S3tGc{G0yPmo6K~-giB*toQ_G1|;T_0-=y_zBK!XRtf-!SuHyK*qSxVVB&{c4-D)0kOn%-QM8yy@;$?k7|aLCJ#~xVl(sZ^-'
    '<|!CcJ>`+DF8=DPR(Oom6g!wy9;Gz7qDc%mEvZP38nnD^$J%dCrq09<NaECl|g#}t9U{IXde-'
    'r~{eO#0Xotd#d~H@nv`;yBc8mMdJFWd#ndr7l`kC3RzEs^MxdzVm@0hQNCfHed}!g2;e~Xc*FhT0<oZlZ9$wpp|x?#n38-'
    '{4(7rzRt|*;QmInP(4F&*Fi${5dpc;5UyrB=i@AfM+yuNp~8v=!x2bF8_T|OQyv)1W!A@n0WP&Z77Uy8m?B_Ul-HUu9M|fg@d_mu'
    'Lu#mz_;OQQ^>}91Y%{e!#MK=_&%oxNe#HS0RBCW5M~?IHjwc!C`$i|3cQ(U=X8d0X^$Mrfe4HR86$qiFB_>nC2zr$-'
    'N5t~uD$=F`2d*VZ;k7W=A2J5;9fUe{aa-'
    '~gKs#6#Msqd6v|cV@YX4}mlbdDEmc)F`4Y4qQOKpgS!HarK5g07ZgF%hvIkwh$l3}Klh}oZGDEni%n3G1B|Ho32Z4K>HXD4fEpSm'
    'E~r1zfY@BW>qJ1qansn^y{2wcq(mZS-5OeT6(2AuyTR(rWObq+ZHuJ1yn%7CjvU0>L1wz^UWiMF@xw=iXGHc>-'
    'OODZtRU81lGQSe7`irf;Ky^}S&`5qf%%?_8^7;E<D^_U`Nf3D0MFzxU*<5;1aUTn&-'
    'RZ#8T81VbmSn0bLT{n#3I5&|2XQ8=VJ#GT+ow#~bhA~*z+*g<MgC84%9SA=*#Cj0?OsBLI)^Z-'
    'ksfu7DMaNEISZ~?OPyT=5)F*1r{{@?7ny!+KTGm6llBIJzxX{&*SXLN8+i_>V99x?|yD8SzaGg!DwqB~o6bdhuE&H4G?CbeYo4F1'
    'uQrC-qe{2O&eX5e$1Q+XUq%l<>HW=$Uz%e5U;u=QO34>9G!Ez@IMHvRem|$e79OcznTEyrq>o=as)ya}d#bEn*s-'
    'yFUb%uKT@ZHAl&&xfPM-Y97<IiE=&bu&!m>6dVp4Zi}L!OKyS-'
    'G|K1si!ZOGYD)X31zQD6*5WbS3WAu}g>tKAWM@(Y{~z!1r6Go9x|bcOEuR=mj$*E$7LZ>H|Ug6iKSF7TmhW7&v0SSIA^@68t&Sd>'
    'jBjXPJ)=BoC+ud?n8z$e`(zMGr%AI4bwW<9Y7(?@!|H+U*&iV#Mj9&cR|t)wkqXUCMiGcTAjd6w7Pk!8RTqO`9=`N7H7^Dk!bv%x'
    'xh)nW>Xa)hjq=HK9@TEPIA|ApPNIg87IO?%C$!Ao!VRKH}Vaj(NeBA>lJk@K2#>C=~b%T88?po=3w_9Oeb4zGkj{Cz%~q2f%T!%o'
    'e~nC#^dV)}VX*8Z-'
    'uL+3JN)_@OA4gEb<#$>^%Y5oCNn@1@2Yp3URwF=q34dW_ixg>*7@PR`M4&)9n^)fX&$_h1Y4I8y_JYe#m>)GT18<^eOc2$-'
    'p5z)W2Vn5o+b%+xzMX6ib81D|wJx|f*RYAeMJeNw6)6S<8js6Z*-'
    '8fw+3_q<0KNm}MYj}0<xyvD*hirmKWVT{RppYa~&@OU<iIXs>XV@^T&oU-zL*pV$k&3aoI$*{lX58R1k><^0^<sh;97<?S_NS{Wo'
    'SBjZFi&TabGkp$u<fuXKOospROgnK-'
    '<k%W^f(yvN9^zr<`aHoHGXqAfbh~Xx=!Wv`_mn6}Y3cv#g>xC3h}s7u$rUUVzFW?5JLlHyCzv|W3Z8NGC5z?4I6BYlc<*z0JQK!T'
    '9?yg^x1fwpd21GOJBl5K=8TKmBysmj+iWc6vBf#MgeNOAqzhT~^K7O5xeD+92UBWKhP_+&a+ku{kE=g%r^0d|SNB_>9VLS>?Du65'
    'hCN2Q$ZIvMsOloAuD9;@(F`ShZwffog5pa!{wQ5u9gh2CEJMaIq2W7HzYccn@OXZVc|4vUV_rcKot&P$TX|3f81{0s>ba4*r4sAA'
    'nOgy?m@gN_lazBlgsDfF>L0jdt>z=T1#7%pu+}?j>%613-'
    'a88$ytA;;I}4kfbk9CuI}404(naA7RbDF223xKX_RWU<v54ixI5zLi#*577@vIp0c|0q|{DK-erR?5I>rfEMji&$i*+;cp>M#_&^'
    ')9bw_FHa#HCeZmKaMwKX#fF0pQfAw@XoIFA=A$o20^F!UGpC}EGpE#4fBW3w;U;he9O_&#pUYMvC<||Ec)8{RxQEspqf(2@EBlod'
    '@nM!#dek%%ETP<n;r3G+?Ly;gIzj2o-Jbmk7vtRP_RYkfDBo$AxAaAQuEYDbM$k%`_fBSY>C^-<;)p=tn{T5%YD|(i-'
    'K~s#6(#>LP6iq>0g{IR*S9NAil`|L(D}(j25oBlfQ8#95woSt+eC=wIeVwM|60#dW{Ig-D*<W)0<hQe55-'
    'Mi=Hr8EUB6g^FR{2Kx5c5_72<fs29Q9R4t3_lIV;>;<_Zd;E)1!9y{4PGY@D5D^lk{O%I!Oe|3R6kB)XgcTCC@E4(})Q<?rC(|n='
    'zI5()Jscw334_bw%+^cZ40L+u2k5R4M^YvpA2qq6=-fZWKEW)T?1=R-kgWZTN4ek${5nCCIg-yR8y!N8H;DN9cv1P%7U?XxOMF{&'
    '+QSiaqEe<p$2d8lPbrzR<Z%2+~$IKr4<S|B@OCKDIzC&?H9QqE!ArVzZ<}tj^VycY3#&*q^DkJ#ruNOn#H3wu1(3dkOg_mlEz>=I'
    'AmvU|wDySOmS4*~t8w))l29qJ|s2|mnVKPyD9I7-W4<-'
    ')+;BAR#@{oGSbu%Eb&=D3pPcd~RS0AVJ1~>Vs@~>R+FoQWzsNRAoBG>4ZRJuskJ}EK8At^|1$wG3U?J4B6!8x*T9w^Pl4~YaNUh0'
    'rYRN@^S5(P{1DBg&IB|NP+qF@Q%{*7X=d^I~M&f@A3TLr|`St6}8)09sx1OI<ON;`vxuA8p@@*%EPj(>NY2)%RIp8t*$tDO}$q;@'
    'jc@x!quC62k)Z_?mc$HO|i^*j{L+~8vaa8QNup?Y(7yqD%2n!@JREH?Mu9yZ&cM%gb9o95z&#$ppLb!aR$pVDI@uxTE}n^D-'
    '@icRe?*o1HYW-'
    ')AbJ1|G`p^TkSYY<ySL@bx8VODfFX~N0#az{fxp6Lv{n=3m4(#`VU6<o+vFkn0j=zxC10!EjUSilW*G7IQkPGQ6fWr9bZ>*X{S@P'
    'eGq0^*m$bYS6luP8Arg~V^NNZfCGNNj_xW&b=Rnu{M6i$u88VX;X3Qjdv1qInc=MIjNM)>~0Xgm3>=LD%uXtSC;E(=IiH%Z;H>6R'
    'y$YTzQ8{CO5h8-wem>-GpFGr<Kl7@0q%b!%-'
    '<ZT)_Vy(Me*%?*1NwZ=<wCh{12O7~Fq*7;J+>=72m5nu{MEi$S>5;jtLpqQ^vF&^(H_qc8|h>+L8E!nc3BpaXeexiC|ZqO4%y8jN'
    '|Xpioh7Onga5eZX~v?SM7eJ@xxceP-GbK4eCYo%S$#dZqZtVCGatP-'
    '>hMa4^SBk|U@(PMRYqK0l8lC?lAY$x$Pj*U1^B!}!O0S*Q^yIBv_r@qq2Yu?>ow1M}c$E<P$2j&P|_v2c7ukBNYzc@*zN!4aO;J5'
    'g|iZ~sm~$1!JduCj3(Q%7A9NQ~4lrztmD(weNRX^k$4nKqZB--pwt17agqW*Gfxo^+U=uCLTx1wnzkE<K^@yhRMD7)RM)Y>j>&#t'
    '|gdGt_j838rVM>lW+#o~KsZWP|!MSNltseLv8{@?F&231RtN7M2HY50*Qk;JGWWlgC{7h*(U*b&iO|<W@Z<0+Z%Byc>nd?~wQ?29'
    'pQs+Lb~MV=3FoGZFJAOKqM7b>!3&o{aM;0lzIPxw=KF1=3M?ws++CO<49zcHf2LF~gqi%JI0up6$l*xYC~O!SVQ=J=>Gxag9COi{'
    'o)^$ODE>LA@(2h8KxSlWO1vz*40ksZ|q}ZqD=(1pJ5KQG`sc9kAld$cFNyTO32s899_J=4pHtTLjW|$I9F+mmuh?Vxi_e%WRH9=;'
    '~1)!qpa|9zC3lR``<pX9fH=XYVm&O*v9J-'
    'PfbJI@%(_h@?z+Dw*|CHAt%Tnkh9wvMKetPO4e{(ug;jk&*`#<C&s>un@Nn^^2;Q<NQaNmLNFP;;DJ=;Xy=PA}?{6<=hQd`vc9!<'
    '~dk1${Ct>$iy=thjHE9`d?zre)x_>uvsn<&tHm}{eZH>(yiRIxplB<<BF8vS`|LdAb}JTeAehPe-'
    'nGdhL{!U>|s)mE_}1M>i4+1Dy-Yz9jUlX-'
    'F}_wDqz_lC#RcG_qpbqweY|k@2gHyg2m0*_jMQG#gd8yZ;)z^JsGBMlxnUd;a2V^Bba$#lL5CA_HC(sjNzXJ-<8Kw-<-'
    '*F>Vfnn{~DWKTGNKuG|Okc`D-!b_bKHRZR-ivWNPXF1g_AzPIVOO4=#<@NvJ=%e0>+8{^UBPb{6V>m($x#qJI7RZfpOHfZyiqYAx'
    '7$HT7nwo1~g&?K48%EY*BTjr1a;YSSupixn0PO0{p0aL{gb1^GoQrDc5}QrC8_#Td&>TC(JoW7KBpq=MsUWzNcb+x3nAE!O1kwjt'
    'rn@>ym6R?Oss%BJ}<kcI0J`>avLSUqNE>&`;m%c&9zv!n0A-'
    '@56pmUjAoHVE^)fZvXt`C|#c%~^kj#~b#|geBn&sV=ma8KSP2YPNI)yV3~i7y#lOQ1yPYP|1wy9m?4(cNT+bW)hYh&#+}dD_}AZL'
    '_b^+&Q3EiC+?+9@4-&P08?<jzroZDp$_G-Y%XW#0YKmU-(vyrS{rJ~ET7Ef@5KNxqLeSCaz$A1t{t!-'
    'KcQVPo1efi+;akcJA&>HrD@AHWo92qNB4e>mKwc=ISx;x@sy6l6L~!C*;3dq2@~3@Q<IU6h;a8X>SCu#KHfkYUc<O7)XXrljuhZ6'
    '$ZF9-'
    'f`=8k(myTSOaIhq9sx<a_;w|#l8Pf!nV4sPeeeH>wf~cCs1mb$E|q^2v;U~F3E(Vb_Icbd%i5==?cc6ay}*cN8Frb=1pIcCs_#q3'
    '2z;z=vcJs#8Qd^X#y?r4+H=>Lbud$3TUonm^$JtpN}^>RCe+K4V!y+MdPN$d&cHXh72kH!;H<~lzQ(zBU|8LR1D_BHlHT=3p+?(k'
    '^?KKY?(npqP&eu_Snxq>c`RwkIicZg#@>A0f5saA+cxBXSw3gUKZ_aum@-fA9<ElRr?-'
    'oBgj%p}FYx<Zi!O2g)>Bg)`pN;x*LX(sG2Yq3D!I35qbAjBj3~ATGxeT?-'
    '{x$Iu1g(Z>rw|`iH~}Vs{1sH!=?7@7_KHtHO(#>=OK}vc0UZ-aH~T7n8{+blXvfi2=$_*Ov4C)Z7e5*DiA{5Y>J@UVXtNsV@OlKo'
    'bo_@@qfkIzO@a-U6xN6@~?to^kudmEwHW8^Z$f`7r3Z42VQ=Q)?9|YvwerEbX2~<uchNHcfWn4uXA;-'
    'nV^?pM}C+PdXu4}(pk75Im}vJ1;q~fGIfk}dx@3Og}c_nQ?#rg)hLY!$J_QF!lq$&e22pTfo>}_xaiH*5|>#=576Oas8BD3twE&{'
    '{EnjONIuOmzrIGJj3()5l+h#|jm3CbPAD_}Bn*-'
    '`$7mcX+FA9%RFRHp=W=Y3O+Ia{1CZlcs!Q)JuBI5P!$#+*GN!a~o+<6;o++))Go}5)Go}5~Go}5?Go_vAnbK^%&SSbm=o)MLK@5*'
    '~?5zJO3iFYzv8TXv{6=tbB%kGM<Mf}MCF7YmX31zKj#<U*dt#}aiDMEEm^i3!svZnzv%XD+Z_yC3=r`D{@K|al0=q85Re>Di!%}b'
    'a8w_`vorFuff)Cn7xQqiVOx*+pbb2tg(Ru%sOuZ^-lim~lr3~YO-'
    '9;Gt+u#1f&rpJZT!_Q9*I=8w&)|xc;J0f&XkXG&U{Mqt$*S_0vAO1K8Bf45TgDS`%r0i`(@NQv+pkbNG-PrMn1Iv0GCEP}Shng-'
    'G8N|oe3>)MudtK)io5|`4EGnVItmxJhm~h%d-jNK**wjjJ;Cs3Z>@0@R}-YVNQSZ9ef;!Xp~JHFj#gp8FTJv_G4-'
    '7!p8;Ulu4I+<s$sLOWC?x;Acc+g_f5X{3T)_$<labD=h$A~%^VreyfH_{GjGf(X8VbGo41Zm2h*H)#-'
    'KwGR_ls;kbsO~XLL^}n2K#`Z*V82fNBadjLPqA-'
    '@esYeob#*EmNDFNcAC1ZIKj?Tdx0$t<8Ozsc$5zuQ)TkiM@r|OyF#d7G;uK9=Lesc?Lbd@ro27Tn7;n!kINGZ}YZewvFWT92Z`H;'
    'kh!NYh$jA=h~QC%=}aH%-;n=3`9%Qtc8?}QoOnryD8y!m@PTbt4+fZyxnln+x@04Me6LQbvS})9S)k-'
    'KSzBxLd6mKjkR&V7d0y04{ZMqS5uuH@gaPBg!$ex8G}nvJvy3Aq?Zb7m9Le<Gncc`w%1Nf6vIaHMUD%nztub$&#y60#`9~;D`xs>'
    'r96>i*V9w)Ws$}+<G(u}t_4|1s;tGD{H)&rxAJM)rR@<aU{4R}_`uJ9XIcSLj0@P)^V>4=g*SBRWhd|W!_n2RCUDr*B>4h$(Np1G'
    '_&VK9Xl}fK9o+OdHjU)VoXCv6$b1>kr7>T|b7{;kX5!!Gz1>>LD9Fus!XyIib4Oq`CiUas!iUSB`==PyA-'
    '<~QOVnPZH+2QB>okAAfgT*q-}M6ji$&5I0;U9t{Y(__){|t`nXuxP;J1s-'
    '*j8LutyQN>SuYRThgZ_PgIJ~ac}196f*}@+cE)&;sllcr<!K~qnjS-'
    '|@j$<vtgC(q3uHX2#sV47s<EITp?bxR%8^1XhcPIX<2UJzTI(q6_MyhTW-KiVw`Nxh&1#9F4rvl=GjZ7M!5O}_e}ZrAKhu*foaM<'
    'D&h`!e6Me(~In?m4SvD(#bLLtc=cX;|=pZt69u?amvEDJAM#sG3BXjAN*j9>;I3%u>q7x2@DDyCnp-RM*dBBsZ5;0{S6?#CSs8B^'
    '-T`)y;0Bd6@j9x?A_eLaq8Al=YV5x34c^Hhm`}Y@H%r)$uLHfQl>>6PR;cgb}9|(A})uW7%HwU)&1iiWHaYopir<O1R-+Z-'
    '{5&9Ou&co4q^ziWe)RA;rv~|qFugCV_w?lf5PI>S%m;N*seh1@_IQSicLn7d39z(w<_`#Fv7X`l_dO)E<%<Rmkcd;RNEMU!5skcx'
    '-qXrZH*1gf2*g(L9b^{KA6-?b~H5x>^4j4mQmS!4uupH(-'
    'LoK8p<uLgTu>mFyv!A6IdYr@bH_S$@Im~~<DRl)Cp!K+);4lM@bX+o(9)h2FHd2GQPFV<6ZjZ*l4fc`Fc_1_w|17rH4KDRrY_Z#4'
    '^_U14nn%$;3Wo5s`bWXAQV%FpfSG||7(PQEr8y`cs`BdejlPnoCX_gVSy306`MIvim96V^HcJ+|c;yHiHn9!V0=C=1M*d!*_6<2G'
    '|18x0BG|~^FJL=IiMlhH>gnOO%F&Zrbal?ct!GL|0QR~s3-'
    'wGG_nRS9gL!pM&ZSasgDRzqDBo1un_LwOOSsgkSXi#tV+z2MUK#VqstduAnr7jtRu_UL;kc(BP{=RrkyV!fhmXR1)hMEQE>C(`c1'
    'xko<4L7uw-jnJPkNe@xk63hNiTCUPpI>G(%W*?2{n}`eJm@YP#5r|uVYZ_<&m`7ahqD;bjc#AS5_Ja+4IzqCLAZZ8k33h1gWJoCr'
    'M4n559~cu_Dn_OybTR!SUK3iQ{nTnm0|Fuev%Gk8r8gv3UGYkBOKj&4V8hHBEwT21HGhz4U-'
    'W&SH<Ov)Je}3l&bWA*u2W6K0m0gj$I%=!~bdc(=e$ddGK_P#1H{=8<#`9mX1P7&}mPC^(EADVG2aV<)Qj1BbCQmHUCi*hLcU?>yC'
    '&D$BrO>}EfY1cz~F>9}Kid(eOGnPd&dyJkV(dwZ*O8;mmDL_VV{{Jqa(VF;J{JQjv;=rIv6G>>9n6b#{M4UB?eZ#|%p)7T?V^~sR'
    '{qbhmmLspUcA-ame=2<I=_%c>e<~Tl$tVMbT>&es|Jj`a+I5Av6Vm~KL23{lSl7Z7mvSi>hk}4Uvj3i119wTXzfy4M~M^xL#L+}e'
    'vTWb*4EepXu+k;>moHsk?LC{?Mi&zN4rM`%T;5&Lu1O&~a7!(CTcv^#^AlOF_DC98q$nINUzdgwpQy{n+gKdo4X6W6D-'
    'Ox0`$8l6Y(ah5+*JqK9O@nVdRk)E8(3>IYRd2g2p36|n;0hZ@J&&Q7!Ik!G3PS;dtL@oTh6)BIRgMaI9|_8yn&|84!V%i`^&tGx)'
    '7u)%?VJT+-;^6Ufh%5u-%(^d5BJOQ6wQ5n8H-'
    '1_&X=)x{6vq5@D$BcIVj3gga>y}l&9EN4=6MxmNN96eVDr2z~>ZG+UDU?Epbn4bz~<q)VsJ!P#ETnWEMqU&P?V|;N`4jF2!BWPUc'
    'bAWnwblKxvj5EGbPeM-7#f9GItuQ-VJEH2GAm5iUbDQ>H3<F+50Re8_VUupYq7K02sCsR<|$(9`h5?3LI}a-'
    '2{dSR513u6aEB<`$b-WDQSe91>UK(FKPT657ey_!;EU(>8t{IrKss50n_6_$bWtjVH!7-Iifb`eCci>-MBS_Nu;NPpVOD_Dy>-'
    '0DD*8vL^$vq33OTGDv;Q)I0X%AoU4MB*JTZw)*OT>rw`%HM@loZ|4euhPBy(dYLqvnY0e2Y-'
    '?zmZ1zINE3J@m4C~Hrd6qW!_DHOy55*yImOc!J6w=knT6!n0jx_~}E0I<rwDbiGm1wRNuHb$MQ(uTs=E~aIDhR!vlE$&twi+YW+*'
    'V}5+S_UkSc6*%`giOiGGC2w-'
    'F=@YpQ{b_Up>mOx<B2ZRm)LZIFK8x5BGreVT!{G5o(uhELp$V)aysF+I%n1=;r<&jWjx5>d{E6;~hO(%<T7N&0gZ_Xk&JGI}YYym'
    'HQ0sZn&q^M$mo?K0EL0!pP=qsgI$Xe>a5(yOFXsr!5nBFz`~k1w;KnQrv!m`k{2Ihf1VdKGct-TR)VNZUIq0rgqZg9Cf^O6@53zJ'
    'T|{JC}MWcGqm|*kHs1qF7;Tfp`X%Y3W?{G%BebMV3i#yA%l6T86o8D#8)GDY3lt;V_uF)Vf>{RQ+Jxx;WtK+)7#q23wA4Q=LNf&w'
    '(|mQr>nf<+Ruf@bt%9QPwzR@(|Zo{^q#{#z2^u|?`Z?$k27q=%dv4A<STpR+1UKC$75{_mwG(b#$W0&g*0(;i#4crRvU}r%i9fr>'
    'B+C2IQ7BA=}1Xmrhd<9RX8NI{#P<&$*p9L-'
    '0Azs@8k%ajzJD5+mRzN^I;SG{UxS8Y^UBJiFu8WU8~Me=E%&<0?|c=3q;qH9EORh)FpG(FiAz*OBnX|<ruvUwv#>cjBftel31g|r'
    'Iy4ReTyDbNCBsmL%vMhNASwc5EO|B-Pn}c=?vjhs&%SGGnilg|AF{0c?1>KA2FYoqGI||^G_D2k*=a^DZ{4BR43N2Fl?iznbS-cM'
    'I3d=UU{}PmsuKXTe#HHSld3L#}tye$+c~jZ;eTsD8gCem*9H01S7vjVA<m&n)>o^35FSa3=*$}Z<S>|NFGC-'
    '`^U`Z$5H40ar1cz>fA3epPxW&`zNf@l1{yUYM1k8^k!kTJ$qJ17e2RV&v87+Wqe+5CcgA_yticReX3A58J}bqsRozZdq<oci*qP<'
    '?KPf=H8x!5iCANA)nf|9cgpMO!H}+;zByTDS6wW%=s*O!0+k6GsE|Hc=2$2(Z&M-'
    'D)<(uXPm69^As@26E~jCri+bHp$2bGMfs7ma0LzceO?$DGR^X=P%u;+OtuBVMyHq!F(^FT63&WMJ*b6rQ>jX*Q3>L6O#~H<wUNE)'
    'ISR1}shd9!UNAkMj@UA9_XwHjCBAWAJQnAR%J2|EF*feSjtavIC0frAq8<&Nv_A=j0Is>_O(#A%zMWQn|oS`IJi@eH2W@3m8npoH'
    'fx6nf3Butl+)k@rSm0GgUWQ{B9JCdm@1oc(!!_m@qIe(C;eUL2EoWY)PlL!e9l4a@}(*fC>(*n0+o{LBGERA<{u85|&I9Ei|T%23'
    'X(jTWSO(6(dT4@yB8dE(p)B`1a*ArIHy0yBU)+J0eJ(`ANmm28dQv;PgHPF+i2739_KyRNK=;KoZuF=V~PCwvVu2+*2r#rE#1l{@'
    'CWNo^3wgR1F6DMNoG7(E<@mOA4GhXC85zl0Co``3%IIoy}zbNO&*Ojad0T+;qFmkFiwW1>2?V0iOIA>q@hZ#}Yn>9Z^iN9$gb|-'
    'a#469yK1zA*6QWt$m>H@v3M+EU;nw|?wu8u^Q)q*<T*Cp%RBz^!_*SPB2MGU)E-'
    '|@u52O@V=@H#iC!_DsRNv19=w|N^}7LW6sZWz3o$s(T0VzP**vY1@V<{Qi6-q>$Ol)KF-'
    '=>jCKB~hDd6!5!QsBZQFj%4af@_3P?v>%DlCl8~FsimMfS*TUcH0zV~Zr1J0)eTl5N6Tr<aP`xBbDWRu_G&*Y!^LJ8&5x#M;9qA2'
    'thQ72i-'
    'm2z<wdJ<xSkgz!@!RCNp2Z2K#yauSi;K*@%5LSBI3y_rigg*iYdiR{?9CVftsng?Yc`R;XH?{eWm9vcDhE%qu3z+AGF6=>ewbu!6'
    'FOdt~nJ`-'
    'PFm@gfoC`hIj}ihOPBCBpaOd!D+wRbteB@`%pg#12NbTNhRzTi`=Fc9k#QMr1k@9q$x7Xm>$)UeNz+y0cNX}u|yS1c}~X*exvh6J'
    'Ws{>BA%z>{9>lwR%Yt!G^sWbqu-<zD=e)b_$|$XLdB0Eeaug3yY#rQd=ju-S|W%_csNR2B#Wq?WH4pEhGF4|)%i>|y6N{ouC5i-'
    'jQ9{!`y#t@>g^49K~uW1V~p&nYqqY^)!E~HmqrD1TElhLskl5?znf7|-'
    't@jO*saF0Q#_GpZTx0aMLaphR1r^3F}0Yr+qci`x$q@$4J5VFHIV3l*-G1nsGCPE8D+Cpnz-CO*BKJsqVA!0R^fY_didU^O5fYm)'
    'Au&@^1V&HeQ#4A-`mvJ^)|uY`w3y0s8JIonN8(LLn({8DcO{E%O)YECQ-{x6H%-'
    'aujYx@;1w<q@uU+Mh<MV83kp^+R_2!K%;cU#L^a^r)j}MNOx-Hs+)x6dutO;7xEyJpNGTrmO)8k?hwvd$>`2zH3ihX_x?G-'
    '9qof*P=Zba6wf;R$<fPv+wTC^~gru5#@sMEL@;uK+LFA>DV+uUIS<E>1)n+|*6Gav3?1TjqlX#|Jq9M?-'
    'sl(9hRobhFpq)@1Sx&)(xsORavh83`dxsHi2OV%oM7e}{Cd+tCxx^&HJ5an_qCpQRRLEGFE#I-'
    'MLbO$AQjLzYz7ZB#Uxiw)_XJIjEvH3LW*z(Vc9EKG)Zwu=Qk&>qb%k1z;(o3`#e15#Dw&$iNDZ|oQ*#)}E%riPYU0-'
    'gXJ($`*9CWWKC}Bo+*~XmvR&-7dD*e&chuWP!X|wDNeY|iX0dtU_Q-'
    'a3$djQuv7CZ3bM12@@rYMCHx`h!I3%K;%shxEc}zXoxjLpAQBQWE>3=Fxzf+l!4ynYBOuaZ9nyx*pPM%hWN;T9@WY4O@r5bKeo>x'
    'amHBttG3NEx{L=^hCy6WW)-'
    'mpJ>!AUZ7(u9447WnudDYTrIMaxB@Dp&GF@`YLyR`sCF3CTE_cPgb!)<IH^6S=Azj8>s@iR#@V(Ws@l^P2j1%;%vx=gpGlJDwMdO'
    '1RW{k*LHwIxlLTG*A4gsCg2u@Kn@1d66DaC=6a%)*Xn1AQg78hL6WdwTnoasmP$!JHV};JIv67g#`O$8sJC>cbiFOwsN&Km9Cz_i'
    'L|B~DGDDTuwKcN2Z_|{DxN$<L{3-pWFbv|zvIcn<nUj^lSL#QaxG65lX6IB9eY3CN8*2)0{!GH=r7*h6uD!D5!EGcjx-ma9E(P{)'
    'Z|z+{!5REm?X`kcsgp5oUH4nBPPj<^?*XoWY6rXJ)VUEm=%~Jznu11@OdT0{aRfK5pJO@eBu9X%m!~8;s3P3+gZsP3f|66)>80xP'
    'O^@Iw{w&A6ug}WyJs7`oex_p8@ycr9=TnTyXaUxK@2|rX9{9dvJkt3m-'
    'W*eh8!P;Q4JZArpNJ{sIGb7Gq*J*7JP82DY4+&q{l>n&pf1OqQEyr=Q$(5cZnWQ$Yt!A?!~Gz6PyLh8uc4SXj!XHB3};n87Gq~m#'
    'kN(kS7OQj#J5zgN^uU<j2A8<8<f7b=4eqgc<nwUn$I-'
    'pT*3j0cOGgQ==0#X?E#n06LL>!G~4~em4{R$EXG^WzwKmZ?|O~FVxp&dj<QnE#+I*9TTmnZh06qU-SG}48o<(kHz4NdQ1cc%|m}S'
    '3WMkC?0N(SFVzDId5JyCmt(`bNC6L?EF&G*#_fp8X?7U+ho*-WL)A^Veus?*d)Tl95-'
    'EmVFU<%d8$Edhw!joVc@#D<6hC<k_AeAbc^tMc6hT=6yB7+fEQQSr#ZaDry$dC`yJ_}5LOgstA%*y<S;SA<-'
    'a@=%HW;;Y9uCdLr^ezCE;Th4htKOV5jZrD;<+dsPSpmCz~MAKppaYGGmkW))sv7k(JmS%9qSu9UM}YxT0mB!vk%kFXdb57bP`7O;'
    'BGThEmTi?c0X6QkRDAnM$cX)Nw}m+&7hr4(obDM8=GXH`km8|+gaz^j`xgbKTE;vf-KA~OGyvcGvx4S7w+<(LP|R-'
    '#NR89q~?vh&KGls7sO%_u5&>w7MJQV5nB}Vyq=HRqFkV#OvD!DGCiP>Q`kORXW5!2keI=ZABaL;eg)Lhnn8zFquXKtQsE!A23{e?'
    '9A`KMgclfYLaYM9%M5oL|C1rqtb;wu1LO`o2tU>vcnE&1Gq4bTtS|5|{8(3D5&T$BU@`ps+U1z*GyLa=WA%qwh(*$AdK6(!j~u;w'
    'dyR(JdX0`aB(7eg6Ame)M$<l>{i)NAybIp1@xlAGK6t;*2k+PW;Qa<4yx-'
    '`9_nXYRQwP%r?hSwIa`<OB=ICTqvPwO~eS^cp*&dcgoXM;@;XMNIwwzJsa2fTp0PGrxac!IkEAeLtUx{x`TZ?R8hA{_eji*Pk(Nw'
    'bXT#x-m7slH5U>p)>+e2_jA<3Ec*<#98d~coLUcF2jh-IS!A7m3bHO-+@)9jPrn{|vRY-%v}X-'
    '2C!?lf>u?j>Cb^+&F*7f!3<UaoEsR@Zh%ul!34gAG=Z?-'
    'i`jr1F5*7={(5yNZIKFQ<}WUnGHQb+TG53}5Giv7}Gid=wV&O^>91>B-'
    '7jitQyYip>YXr7nui2mMu#DWo>jKCcD~Oy8Y6hLz0rL10?Nzc#Sv?GS{MTH!QcCr5IBo@4w+)E^9)3jDP{aCMz<47Dp6;EFmR-eX'
    'A**8uT8ODd`VL4ClIo~{k%LzeV%MeiT6q_?Yi|Cl9xs6R%1!jitkDUb|O*EqS_o(xIz!W<yX9^VRLHGGQ2@~ia9o5{^pE{?T3T<Y'
    'RjyRX+{3Q5kiFRRVsL2cF;z5cq=$|kgR(EqtI==iJ&G}{m?9e8&BbaYQgnEr?qA|6$L6kgN>b|`-'
    '`^oZtkiTZOut^@m&`;2;pIekLi@2GbE%+*c8_P-+0>6K0gDRc=i1M(!NG+{4>w8i9iPC#8y#kp&^TE&#$v6Lpg^Co<Ap-'
    'W;R0GGNX76KpYF@;2D+Gk3&zzyhN40R~hFo)f3&6?M+f%Xm+C_v>h2hC2n8mlHMT=Jh$g|;sIt6~{73V8dIS!O#>he~3LXu+>xNP'
    'j=W^{`&+B&Bt%jUvxyU^0iscVgi5Q^Kw*gH1D~<#tBGUGm&|GgK8Ii=T}JC#q!6niy)=tcYe0>62%1bD>LPEe@BuG}hv8=rM(4X4'
    '-GJ8zv}|)$QOjxRX1@Mz$)uCc_e3LmYvaL+&7q$b_r>62y2}4U2a&hcW>0YR;qG-Yp_oU<_J^FlwSj{STN)uxt7sF_U25-'
    '+M8WV3+qlVJ5-8%74a8f?etF$Lzs$*G@GtqfG|R!xA&Cw}^77$hNpS`sU5*<|@--4Gfo>7Hi;l^q4|&GTB!eyuXg-'
    'l?H#HQ<>y;jhEaW>?OB{c**UdUUGYwm)suiCAUX-'
    '$!*(7c)dtwQxfP#k<6vU&&@D9)56=AsdF+4Vn`;W@n<P^^@KxiJfOsu^cpAAd7RDO<3t|J4kCwxYDY8|Z{;<>n7g|y*0gY)%VJIY'
    'i5^oZ+}%Da5rw;38{)y;?Ha412tR9GS3|PSN<G2PM%UkvY;tLjKM6HRx)xB_;2dN9?Vd<6M@0{Pu$jzkg!Lpm!|O&JrCS>6vN7%&'
    'YP@Rn<SLCczf_@Ow^cZ*!)o|lB(+|)V$vw=s@x6#=~CO^#MOfG{nrk;BrX&=p1Ao2tvs4bqLoK;NwgM=nw-$SygRiiY-'
    'efmJFBrW45&2XLpD!xGpGG{HL^6c>OxfVY+KdSx@HkxrFp4|U^Q0TZu1oee-'
    'vtzo%)x619rHrit2^TB|1c6-q4{8opDcuh?;ZEyKZ19b&w?Q40gP`O4UO;ziW4?dijdrJ*4U_-Ku_tU!mLU68jV4SS2pXv$^@ZYj'
    '`xP#2OyWDzT=R%}>l-'
    'j%`DPjyiaKr=$uD9l)W{_arxBxyB5;CfBm03dV+1my6ZUoMuNgLjp_MVIRbhu5z2_1bLEcjTT)`hi&RMWQN5rCkgVd`XXbscBqV@'
    'QDKsKuUyUMs=o|$3g>f}VyxyI2K5h+YNK$}L|0~(YWOaVR@sKZpPC*?E^)EQnb6H;*7A6AiM2eQTw-'
    'l8>z`J(1Dly<+V%8(jg%KR3_G<!+`b0STq5Zn6!3t}cbe&REQx<452&9=+xw6#RR1YW>jNqQo*-'
    '=_OY#LaJKpEZ)BYgT%{~EipKy)pOSG50$k`dC2bpET$2D{>srpJsO@EbNGLF$N^@)Cqn5@?tMy9svrSe^O8$1)2<h}3alIwUp&%`'
    '<&&oi;En9V1aE#m#v_0->=3{*D}MNsmf`VUF}|3m+`w=<88>o^a%=RMITb(xkESH7u(<dPIATDK3`vQ5VZ1_S7pN+-'
    '6GCMn~)Y3k6I>=uPpphzJnj;&ly{wk<@DW0-'
    'qnW7{rs8Q5yia=@LbR}7bZAm_1t4W&n`)1~wZ}#ou>J5PeS`fR?zJ0sz``*lV{yw#0oW(=QKDCMlGs@$b%Rc8!_92|*;b<?zJmd3'
    'qJ#yrV$VqB*q#DL~OxvxIY8ppDxQB4yb8&F{MyyEF^0`TXZ|##qTXoPt-'
    '4mw_cU>pg8CW71><l;dJ>y`rvJ_E)4LVCiD0$+Uv^(8=`7=>CdE!hIPM$cEP3n)ANL`P%KSC&{7bCh4Ti{kR_n|*Dxn12Lz(TjK('
    'HsFLx^<1_3b4_wYcx-Qk#1e1`2wtT>l!T(V5Zy7<%I(5^bKyh-glBVco(h}&$(%m1zvRu&VM&hZ*R;&8_Ex754y>K$yo>T9SzGd5'
    'ukr7y}vJjn^>PFY4i2ZM&Y=Lvr#y1;%qiaufMo->f1)@U`2`MeNwd>`&OWkiS9QhD;c)~TTD-'
    'S)1Y1;G>F8;Dq0n0tCu6Y#T!X2rFnY|F5!9%Y&&YTIu3>ab>qAZZUA-ToKVe_l!bku#U*i1fvG?-m`89FdC~eH-'
    '_@`acyd<vIEDcyo=uakImx*w95`_<3I|S{%O>6DllB^hLvlJ`N_`YO$t}lO_bx3HZ{C~hwt)Yr6v=*8*mEz7NNRiJ9@rFZwK>8bO'
    'jt{ok$+j35e;S$?~1^$B99r}D{!*Lxj6Ub<J1h|!Qd!iXJJ2zjh5;X<`?`U^=V;8`*&P()+M^zYK<rL_w+<=cydm+%>-'
    'a4j;F#U_1Aho3dc^oABAHl-'
    'p`oKo|z~EKZ?LMy_6k8SesuJ3*PkSB8qn0F60f=WqOB@bExV6P9f(K>ak78d3NPjcL_P4@PJ<sasjm!-'
    '!0@qGTx5}?5w1mRTsYYswAc+t0Xp#Qhw&9H`H*!S&s5~gC0=<Bln)IiS9QUn*rDfxjCh_$NZF`(8ica<r`2cQ@ImEA(cJmvQ7;u?'
    '16Vr4Jz!}s1L{#^-1dA2?YOX{o4n&-'
    'v)2*PVi7%RCkc2oOiVDL6mJZTZACmwUOVLr#(sPg#`l;oAb=`o6!DfG|z8F`=iM`zXk1&S?2j-'
    'v_G27^IOsWFdbX4{bP$B<@m69l(cy>zV@{F%A_{GFiL4zSrCTYlA?R&#QQ^aZ*lovlgd;Q#{MAP(-+~&Al-'
    'v^Um2u(FX#g@6*taIR(CJKOrENa)oGUyM#H{_XN5U?anP&m(T&vpjKE5fND&qgimA2wRUzLkj#B41O5MOw>PC)IH*u7D7DuU@IZE'
    'BaQEJD#`!4~Gm(#v2_}bIHtCHIH-'
    'O&mg%cDBv))f6SrydTiC4(~!ht`rEsVLuAQlf$SLR=N3f$;9Df;8~E`hZMjjmg#T6aq1|p1daH+2RQni;5<dF7j{;i;7xZHbs8v?'
    'av}6*q*lS3S4KFj~rHBbw`dXuXaTaEU$VZN0wK;kweR?zR0oVRe$8*@@gPry8N#N(h*1tx8iG03$IRU;rB+<!t(GCxh+Kt&8ZKBY'
    '9XBIK&TeJSy8^Bq(lq#g}6FM3*p^Y2WjE=^Z}V{z}kcs?iTVLt%Q4pJmb+La8Jz%k0XE;EuKT*&=RetCW0z#einhEIZ{X~nqq#51'
    'hS%kF1&G*|E?PZIehaA44Qe7`2W=HrW}d?rLM=lj%sGqi>z|n@U^FM*CbW$#iCN%UZJju;qxupN~2WlEpeyz9PSN-oXG8^6Cv$aJ'
    'Q%8)aHfNys`*Am`F0bhn#5nx7wno$s(FdkG<f$lnN+jV9`K?*Ad`7mTUz%$1iPtO1&5Ht(8=4g2TmLGigt8z$*bYZCpPCnAx~>{-'
    'y`IEzQeOlfE`Juz?X&Gt=_ROz9QtS>RneV4nFLA_Qh9)+^f#G9$5LBI%{8iUC4d5+7J8u<?G<1QDql`MM!10<7-'
    'c4uT84#_eWbL%R^Y?j+9l>ocd6xD#Do#g{tCEMft9il2uY)h--sZNqG0QL968V^#Pfz#oDxysjfyShgCg;f}9d!e&B*P=*7ocsk{'
    '*E%v>@uqWx6LWWZ+pE6R8`x=1-'
    '}#Wp0?3p=rms`kM~{1)ZGTPV*l0Ne0?Q?(#M#7Mz*;A>C8u1hM|OUYiqDTvJ<)=4Y4Sq=vBVC2pe#WP=NBvkR>Oe3L+x3{8vn@Ne'
    '{=?ipSkmA9+uM1MVm-GRdjKtc+Og0bkt1DZUab?SLu54Msl`Si|vSk%lwyfsLmNisIs$OO_*@>?`HMu^iCNG!7U%a48F;pK7%{=)'
    'zc~@`MhNeNQU`8#%Sc*_YaPy34y@Q+Oa1gNOR6~l^nO}G~RO{e8heNgQlZx^UB_&#?ulMyqS_kjGK1l0c)(2!V3aip+;_0loL6!H'
    'CC3%76^G>Ajyg)6UM??}S4i4FW=*%0~#J>>5v)7zsd;>eibJ#JS%Z~9pc8uq<W4wSJ<Av-P-zYHJvy5_0m~LvPWE19_?)Ay2rReb'
    'Tkes7wm2~E#YC<bBuF#yvpvsI&JS0=6OVwy-'
    'X>G{o2?dvDGpb34+FY&EAWujsm=Z~AP3;xBheQYPin=+QY`AqPh1oFDww&T@7;0NVfi{e_t)xgB2HRFqs12iSt0~ro;kGrD1jIbx'
    'eLlF9)-jwOQ4{mTi0Ls;gQ$xGsWZpY8ad7RriDiK;Y`y)CA>)EKC|?@l-$2V{#b_sFV5|{66-mFk>*_wQ5Q;U$WqS)-'
    'n`Fwa^8(u9BVeK%Zm!Ga#XQ}GaMY{u8|>SO$;fU#q7IgoS8^792S^Tcbws{Q10Yh(E(ezwHM6ngMzZbm=OMwEAVPeyKyf>ZaJQI>'
    'zWUn9x7Ql)%4Ju&F}RwnOa(^lIixX@&K8I+vGtq3%AQdWESp_BV-o#$m3)i4C>B9i-'
    '!g8X1{&@DmXX8UWMEt`+OfbFeCQ)>)^i}7CuV6sG{OEK|S5C3)7(L6MB4?EqxH{MG-'
    '1uEA1#=Z5VDSU1>?03&Bv{P7|>C>N7$G45ykAD&WubF`0T-'
    's!Cb#Lv)AcXge+1A_rY_5#H!t0db*Z@D5zPA@VXd%l+hKY>~s{W$cuPDcPQ<G(LJHN6FOOC6AG**(={B`?61-'
    'Ap5dko+SHnK%OG|a+YI~I6Zk)m`3I)*qX2-'
    'm@#r9O{V5FGeczxr<xfm)A#i;nVM0OJFY1rsiCDd7gO7TzCm&&`)`}hdO3d+`R@haPS!=QnyS9DXfZyD6tRBJgyl8ulY?#D$oKbx'
    'Z9Pc%_kwK$I1G%+Tzhb00Ik8QPvFC@4!BdlC0sRaD{>Mkv^-'
    'B>_0AWmMhDAu?st%p=G4{t0$`7mX@WN&RvRjKI8|+^;QvJ*lc^`ADis1x#s9{X_7=e!#`+foH;*idDN^62?ukh)oPWMHrmE6)v8g'
    'I`R#z^j3_uf7rW*)m)fmH;Z0)b681`gqe$9$uQ?}Mua}2w(HNIM6*p{vBH9LlVd5zPNU6G0+s*Kzv@G;DwaCp#8(c*X;Vf>6!Y2r'
    '2?Ru?L6I8|M!xDV@NGIgjV?Jl(zi~sy7bX{K5h*$7dTmqKF)XIHN%w`yT!g;@hc!^M3jj#(sZ7ULX4R(?g?A4wY$!>5-'
    'Y(sD)cY*n83uYqO0~U|1R*z&a?2firJ(7K}tlCbaNcO|6W-'
    'DGJIRJuiLhr>n>s}PT4@X3BFC(lu>f(r^^0OTQBY;%#X<GA&`K0<#sl$2dL#4h?ACoCgow8%s_=6pekSuE=8uD;Ncx|Tb*jDQLN2'
    '()@_AIJ$$2Qa)qoW&`{M*Rn-zEl1&mw<#zAaP$lX?M>Wf!7HY`G99e+z~fKq24nSa;Os<R-iLl+iw8-'
    'oXVLh{A9XH>BLm`V@<zU>L-'
    'rC>RE@C}YD)RcSv`j`nMe&Q9a#>~xOK&fw_mOpeaha&)$iqqFs{wR5}+LR>1~K@+lg&Wa`j|D~DP;G_VqV7HM%9d7Z(Ms};Kh4Cu'
    '&I#A`vhsWn+Nn*cHW#pnX9k!J<00c271qq;kz>QHb2I9sj7z1%*#+H=S-'
    'IxYxna$Yi{LXnJ5F++ZG|Q!x|L;7$Do!eaFSpCc1`NdBB8zMg_k%&g!{6nmY3;+pxP3(TzS~XK>cqx%_-'
    'kH06{!g^{57whLEJC=Et#hL0Q}ohGFX`=wOA<EIy3txjQub-'
    'P0;V^8rg92!%a~*`QfIF{U;?mu_*#8n$F^C1#|3Fwc0yD^OgT0vW5J5Bd?AU6A^1LwXMYrzFkj|n0I51<Zc=Rafgtv(sY0}8m5N4'
    '-l`b>npaOEf*1apS5M;<2l#7Vt;aDV@YlS07OCU#mnXb;m5j@N6#nfj9_kVXdzhEDPQRziV#84nH%H;9hnq7tom3Sy1ta6a`jksd'
    'ERtoi2kTS!luWRl=Bod67NwE1C{3J2nZ;R@X3nCta292D$?$SFv7CpuFmj;N0KjborkFD!a|o$-'
    'A@w0moiP7t+lO_j>Qri4x667M$zgt)e9!1!rEreJEm1hf;g*c;B?<XrsG!;ixF#+NE|m;StMNOjy{f3ZR~2<WV$Z5Frg67iBef}_'
    '!qay3g9uD7+o$rw2y8Fgr}Co+j4#`#^5Y1sFWaZ`lL*W&+o$qM1ooHhQ~7BG2G|Cdb~r!)LyTjO2%2xrBVRO5`)lZKhr$~cq^;Ix'
    'bt6MKx?yn?j&4|-'
    'v9ly4BQa#bx!!CvALF2hm<~FDw%Up#41jeM?Z#7ZnZC?u%%)<*B<{v}TH<ld?vB*|xY<46*Z$Ux!|aLF?;={l!TI@pL|Zs`KK~oh'
    '9u97g?KOr?d>~T)7rB54;kk6)X_x?(7!Wy3L`pXRCXC>)Fzr@8ryCc-`3<*5;rxbMGj^Dy?8%<Rh2z>giq^UFxWAHd)-'
    'ehhp$Rkv?%W|S99E-K67VFa<IsQXv?6w}sr`f}#dz02!M$qgkOh%?Gg-'
    '1D97N+NlUHtw!T}DqMd1L4+cIX1>q=R%IYGu2T3DwO!)nnVt~RZa293~)O?Kxa*DKZ;nJzoR@qGv~vyj7&KxP(l_%X=LLJmIxnOR'
    '@<QxMdBB?hp!IeVrOb`DUwa7Y%sst0&cT=Y;QMk&)TOKG@0ZMq#+q74NW=}&+o!fNy<!Vv*g2>A1BL_iUOd4E6+LXEH&@xoD+x)k'
    'cGb8AtJJ}BuQ*M7~~$j&rvlOOmLSTqRNiDW(mRMUNEgKQu!8xEa!W9hLS(NfuP>^wo)aPB-'
    'Y*>LbYA=z;9yd$!8%WkJ#`{hwzyN)KbYuaeqRh9>FN1ApaGG%|DcHIm|glX3;a72K1;m@BIq+R;`0osKFF|%pcKh>2f=rg&=A_g2'
    '!#3N_l#^_zxu2P5X3Vuhc$z9s$d$%AS7)HsNBGe-oB-JaJDAXqyD%CHTE;JwrFr{%gPPvBVF<-'
    'fkC6sIWD9Uvq<iwq6>V+7d;lMJgKUBaWVO3PW#zO)W3@^d-'
    'AO+L!4^S{1tekPAcio?;a8HkSCM7r^kNZk+JfQ?LM46&JV;;F0ABu_SR6OC^3m7w^5^SULX~Dt^F)iBClnQ}P2LhGqKm{BUrc}dt'
    'NPtq|C7ltZRQmk^N`<4@vnkc&`ZN-'
    'H&PgLL{I;}0(IC%)M^ee_kit@0#PG%1Z~X~sczR{8Q`Lj=ZC_R2PN?e4(blUAfHUq&Q!T__9}HBj9Tjj$m};HELjqI_FTu<p)za?'
    '|P%WGZozdK?Pumbj84v7N?sWN;I|Xm8sE%l@@LJBau3100Tc|3>Io>7Icn3A@5o&^i9`_10(GgDjgqq|C$o)c1jwq@!Ak?LVLq^E'
    'CQ?x_!gs*5P5{gzknxb8R+wp}oMMJpyp+H63QvrvBDcW!GkN`!)OHdo6X!`vDiiU%aGg@Bt$&Im#W3ih#7Q2OGv0FJ7yNzS9+c_4'
    'ygJZEfITqV5`<)Vu$dkSjoJ=S|ok*hzI91kaOI`)`1sD9$yBNmskX6-'
    '@e?Q<{$;wjYx&Suh?lk2>%;89&at&0#Az{k38xILkF1)mLLCU4yAD~<~N<E|1m0ZPexv41%DlO65q|5kq_~kbp98~JCp&~~RFx-'
    '7_v@do)(s%$><)+h2$9ygg2v8H;befrJqMJ@rt0uYWG<9mSn@&@Y>E8{GWa+f`usr2!@2Q0L){nM?mBW-&HKwQ-'
    ';xi8is@T6)z#(BO_B%WzK*jL)*9WPXet&?9!Mp1-'
    'u3sfl%r30&N0^~2;cGRO(_ZKrsZxH6;t)=!bYSs8;Q|$`U@q3(4kp+%3VqFiWzmMd=EAaQLtpb?S+t?A`LHb7`5p^kS+w&#7Q(W4'
    'zu729+k;KUs^5*B*_7G=Q4WTpp1?>nW3hfEEH~*bd|R@X$DE==1QrNP!b8Fe1SaDlS!DZCX&wSY_fthleLztxId4a}*p3&f)`jC#'
    'wJ7en_w1J@`H}oc-C>pHWBIYVQ*cT*&zbyPmK`e854f>G4>T3DYi@=Gh_=lwumI7vxC0g!+81}i;=<+IYlKfv#i1@{%-'
    'Y|V*7KBOkf<j`N{_Sos^16|?P5G6OtiP+Az4KGO37_o=}&XyB7j7^C(rR7%q`_P)L0XlQtD!y$5O_5EMuI<a>jYAV4TNF#(At_oX'
    '2X$d8~2l{vWdBx5}-st&lmi4Yn1shqlAE63FmSPw_1~MNN(5=Pip^KIYxB)azot$OD0L#jo%{pkVQvK9Eha|D34RYFEny?#F-'
    'bMjiZ!FCJD-FQp97eiG3fIZPs&3l9GZ$^acC5zPk^dKG1WMo2^pVPjgu9TdyFPKnheU@F&NWxIgebt(Q@(lU>KJLK_ibqPvZdcQm'
    'rLp|+_byKANR@0FYqxM|n304qKR9~bALj?<8;=xeCzNwGNBG?_JT}(K4rBiJb<%_3{Q|emR@*K^;Vy_{8JznmY-'
    '>nd6uUXp3N`VHOEn8nD&|<T^lhp!EHVZsi<CpAxC_hxUTP^rRexlm!^UvgG>TX*}`;bsS=5cU8NBRX}26Ru(QV~d*P|x_XULw@r8'
    'Ch=>>Um$*G`zulfrmn64PWA+P+7mBkI5qI{*s$`slXL_t;@_%wziS8wX--y+srB27MG&MDw|xa*2b>FsxemWu`8Zyj@6f9S5q}RR'
    '`<tlbjD9uMsFa2L^B?I9$6QmM2|%B12*LfoW#~iYY#s0f!BklE=rE#nu5ArsU>$$EFa$x#&=Nb(}psc-'
    '|=v$sNqx(hl+YgACpDY`%;XoN-9i6=Q<~dm+yh$ak>0}{H;8$iEDj~U-'
    '9r2zqG!IQ~I;H5xA54Z@_BBxP#RZ%*;%;#g>?vnPKbkFf%jLY7S;*YHej6W@hZ1`)4sTQ*Sw<LcNk&@?M^n0H5LA%S&VVZLJXJ;i'
    'isKOlWhKM?&Qb-{Fx^`R>)nWRdSkT6?ix*32D|^WAZ*b#tX`ewals00rl#Y|>KpgVb0eIV)CQj9rOjORVmT9ryF6{-)o-'
    'Wgp+*vX2M2?BhW$`*?`UJ|5<>k4L!d<55zg_E>$S`PKnsf~y)m0mI_-yjiTlBm1T&;E1rH`gc4UDq#2mkA@2P6MamkAWO0&Z=;Zt'
    'hz~N^ueZ5DT_$KlM0qadMHTDbW8#KRh8;B~G<1VdA3B%xI5rsDUKM!)J4<bQiad!e<|jRmX(`uxtl%P#RqWcX4r}M{*0&<BQEv+M'
    'm_1!5{BwNcPFjhog}=Z*COl#w7JZ+#%YD-JGS{UtoDfhMP6(*XR?YLv63eU8cY}QP4Vo|T!;jF6`vPx%IO*GNeoeYi1Z`lxf5A#$'
    '#`kCrPLZe0{t4MloBfm4{e@NhDcM8&{?oFT_WfsMAMN|=ZS(?``Df(-'
    'SH&;m@}HGl9=#^8`+t`g+`z2LR}D129$PcKTlXR>R`c_!#=r%r3g-'
    'e;Ws~b~Qn((@*YzS!T9>`{hc^W$dD_)!3TxQx;bVz~8Eoq1zF=M2yMfc*jhyyw;<WcHPJ1_V+Pj6*-'
    'm{rkqxXH)B|L_N+A26|vOJUl(BsQ@8Um=bd~31xHbuVXSB@V89bo(z=m6uh3HW5m-'
    'rE6}zcVp&1A$f5LJxy8+hg99;pG{N5BIF>(6KKTEqF*PcnzNb{!&c5=Vk%MH2kGfC)Zp4r23{qvp=Q&+M(H>R^M`<56xv!n}tUQQ'
    'Coxu0a4q9M*&eggogo9J4GlCpx2l6bfLatWIaQuzetm{`Pmc301B8e22j9+Y_dL=>`_CPb)|5FNM;N5nVS!LZ>&CcK>dS^iRi?;*'
    'WR8_sK0VN{b$tQI9C38^|)i@KdZj&Sos>&qPAIc?Q-G9O{>odGjBSqpTE!BaN96=dPE?Ocaz{<XY_ou+jrLqe`Mzp1P~QBXVqY^-'
    '89Y#_4PDyo9j4n3=DyZV_*nO%qH$nO9GgW33fHxbzsS&<~9j49m|5QI~_Bz!?{YvipdwxH(Vv~K@0aEr$we+eO|Z;<mw9HMv$vN6'
    'J`c^7z?1^zmu;L@X(^YR=^`gwB{#H8UrO@(ikWKld_5S4<&c<YJm@u#A)vSeTTb$-{tP#_qhA_40r#Y<?i2e-'
    '2MB$>;A=EoJPolq3&kLt0}?^kq>LrAMkHqe^yRA=DJRkPoML~Xqy2~&PL3W6_4bS0_DT3Y_AdU(6YT=z#~Pr=Aum=0}^2J7?1#yv'
    '&r`2@k!aP5%?h4-'
    'i*D2#5G!sz10NVxfL6GN9^j_BGjjDX8e6IzW9yf@4({TB&gWMVSh##`*YYaUC_twwB~v{d^&<jpAYKGpV`P_ZrA3;bIzIXq|w@Ka'
    'J=kdSHK&j-VRSYT(3K>PSt)}^OXV~TFh4qc%+EgT)#`lzyP>(3=Duvvxxa?8(wVre~5Rw&j'
)

def _build_tuned_standard():
    """Expand offline schedules; tree and input values remain runtime data."""
    import base64, pickle, zlib
    length, main, cases = pickle.loads(zlib.decompress(
        base64.b85decode(_TUNED_STANDARD)))
    program = [{} for _ in range(length)]
    for position, bundle in main:
        program[position] = bundle
    for base, count, stride, template, patches in cases:
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
                elif code == "lookup_vstore":
                    slot = ("vstore", operation[2+(index&1)], operation[4+index])
                elif code == "lookup_pair_store":
                    slot = ("vstore", operation[1], operation[3+index])
                else:
                    assert code == "lookup_load"
                    slot = ("load", operation[1], operation[3+index])
                bundle[engine][position] = slot
            program[base+choice*stride] = bundle
    assert len(program) <= 12_000
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
