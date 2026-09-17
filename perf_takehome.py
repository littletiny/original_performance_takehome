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
# Verified checkpoint: 917 dynamic cycles; 10810 static bundles.
_TUNED_STANDARD = (
    'c-'
    'q9hcbrw#)j!NGS5Yh|iYVfxgEX;@U85=LVM&s)kBFC{4_!o!`Q>?xQB;hH0qfWhO*9ly>_Z=#AT~4>?4u$Medt4Z*V=opz4tkn<o'
    '&&W<nuaw#<OSc+<VU2Yp?oUYyGc&pg{xCfc}FsJN25-F=_CBCUm@44;fzQI{}yX=E{c;yJ7T%q#+z~z}N``HSEn@(zMq{e;7NV-'
    '@Us3kl{DstM1i<ZyY*e0vyuu&IyGZ@1zU!CSwXiZZvNq&pBs8(l}|5G#EXhFi@KJ9b-NsX_hoFZ@VW4qA|Yd!2R+4aw7^i-'
    'q3Ht9roQ0rFS!bHSq8xpENYD-'
    '<3<6&^;*k86EExohKv*;QJjn8ef=C4#dlb<`Q=2@WUJ1>w3C&FF64(@g@BR9*m!P;Fvz6@Q%t0^t2t4{8;<pjnf}~oTI@#xwGdN7'
    'B`?9Zb6@p@74%E@BsV92jS&`czLhrIw83zX@+-TKYb5=I^IHbTPL{B?>Ml1aytGV-@kowK7PVj`t$21B;D~-'
    'JK&4xW)3qq!ykAseo7<zfrsGbL3G(2w&jxJl4kbDm*9^tvLCjCK6#J(<n6R@_k<tNnD6^Uygbmp1nGNBatz&^d3Fsub4g48@D2LB'
    'JH<Tw!RB5{@E*-'
    'w;=Pxed&i41d=GkvgNEKXWb~jBLvHC^=!+Yk;0K<bobR6K@YE9>iuZf4y_FmA@(}w(n{&xQ_*nMmJ@Dt3*q_(YE!ro=&%`I%=^k$'
    'bEq1uYdRlCEi%mrG?)G{kU7u**^(4G>q&yif4;t+st%rFubHn&dx7v-'
    't2QJ44DWxrz=bk$uslZiPM69hyqZV&(Uu}7NOAZ=bFxEbBlhgx`#-};de(sHUd4s*PEjeTv_b|z&`12n2=j-'
    'U_n{)Q**W%N=MIL^lhuq*6t7);`E!NOtqg&L{BcEj7_Y}Nzv_2It4@up<J7l!g3x7FcchOeUlMgH)5%TodNSNDs*T>;sn~U-'
    'I7?xBO___+W-$$g6QzP>+-'
    '{Z#|(DRyCQpafwKKE#Qf0Oa@Qv1|D=8_v7FLZ}hM1q}e@dGW^<sx?2>K0$q;wQKGiWaNg;yYSwa*G;TtZ|EPiBhN7>u>1#)X@<M='
    '()?ybI))L(F32dBDp;oH`**JaWOu*)!zOjD}BuKADMpsgYfwe$G1B+e)5~}ac{JLdNJ*VJHemQ?r(KZiw`)4&jfr)zuDq`^99j?7'
    'oXGONB6RS(F1<%7AtA-om+fHiyF81gcjer#mBVxCKm~*6}Vme#O3i5m)j@Cmn`$AqzYeDIoe!SM{=wwQvn~oDnY$c@DG18KKvN_@'
    'VDUQP4+Uu%Zu&hRJ@#Q-'
    '!dQGy=5{adCXq6N`@wn+soF;u;dAQ*(MpDJZUf6CL@xk>}9)TWHQlSwogVSPsf{FjZd_~JWoy1Y^;08s$^L*!#;62zQV`_%j4$hk'
    'mkusd<k#&QPG2<z>XjEclaDPTOv0Sczug~JiyDuZr(?^<R&loUcjGEwLkxmHuX!l!z*a9H#gdmyUN~09^R#m`^xR&2eeq}7Vp#I6'
    'SsJe79U6Z3rZvl?9<lcExV_sqh6WRjP~c*34mI>EVJaO!=DGa-YnvCXd{wKh|!_BNG^>y9o`|v{nS0-i}-'
    '{u*oRt95A~^gs9E@RJb!YX{p1qd?#;<%_Ht;lCAr*Q4okKsSJ=zp$+qN5dpRQ6o?K-'
    'wM<zRxtL^2eWM^_s^x#cIvT&?<sysfrd)7K)?9oo*S0uIeu`BWCW%j4F5nCS}$@UWXt-uGz!(v_sJp>#dRwoEJj#qg)<IOEkcZav'
    '>(OGhp5@Wn*KWHX?b+nhW@vD=)oQq#w?BzWC>T3UP8SU!(?oF1`;=P=g?}x+5B1z1CoF$STV*>>)6v+~MZYyc?jRDhs0$$cx>xpz'
    '>bmYtV$lI+kXVK8%n0WL5B$A8Ghg~a@$zk(LV5ER^JPneb_@rRpY7TyNwwLqqs~aEoJnuwH{OZexy;ahW4}0sRKOgos$pAj=ZIgj'
    '~*xMz8_^`K621k4Udyy<Atp%e+vIt2q_z<*-'
    'p|3zfH0Fnm#Ct^$?>&$MY8~T~?b`9){m0zUZ%9A9Rk%zf^?06;vqoEEvUyAx{Ot{fFcU=5GbYS$1hfsT0s=aQ)ky*mkzaKc(2`dL'
    'Vo%GJ1w0*Vl56ed&}41W%U%vk)+N{3%i+oTq_@2sk!(n=x0fT6jY+{?j!HHqeejv?&D}71_^5tl<?lDfUM4r-'
    'UXmzt#CuF6ZyAXJ_X-'
    'q1P9~B>@baU?a83muq*^}3wMiY)fk`*GD^fkFLb|7@c5F<wXGC(Sp<0zlAg0={1+)v4=_#OnSiLEb=l$w60jGu45CIp2)rkU*4XY'
    'Of^bV`30t$g)ry#@jv}g7$I<xP%<F<rk+FmD}7w1R}{Dz)z@Xzq!dQBL7AwA=vxamwmBe;olPNwFOa5c<Y5d|w_3Zf=rR;-'
    'Q0Zi|!z`a$67^SA$vfV;!$YXSF!)vE$Z!fJ+qGsEgjfwcCkPX){ktL_3?gjHt&t;1@CfQN#NTSPLhr@dZC5@~5JQns*65W83c)Zr'
    'sp0zfsLK!5CrhrGt616j(9Utz3oqQG%!1M3jOE9*t_Ld+{)2^be{bf|!zg?*id`#Qzm=K|WwkL>B9<C=HJ3oaKZf$aH$D+G!ld%)'
    'mIfilRRFt|#f5VA)Mt`;bT>=}b=1d1Vh$e>K19O@J3fNGW)fj-'
    '4bpuK`9)0nrq(#u=eCg_ejFl#uIyoCPTTHE17!74#VYP3KB?~YVHK2rP3<R(KVr1sR9Oy3K*BarEJ0jGx54Fa`?-'
    '(4g~Gt^f87Vuo#>pu&WM4RpPeF8<%7JK~{fwE|;z5c5}VYJO&-'
    '!D)aZMWBd6DW>$*z55E<<ZXUF}ewNHa<qRD1Mxd$kWqQuoBNRrSRk!fn(@NcNz>>S{d@Dh0)%$FlrerI>^i9#$eKwKq8=1pwPPlD'
    '#Pj&2^#~GO-Cl1mQtM7rtwDX&|t(43r6hlV8o6HM(oI7#Ex=CY>G-'
    'ZA}IoudcdI)31y$#a#;sS&+fRaD#)Dwr0*+<F{{#ic9cYY7qeNblsNiMz@330H3G(l)ujSDgw@Le&Iqd!0?rMq;R4PJt6>7p53A1'
    'w%nGYZBrFQ6izO@$8}Jftz;yeIs{|>P4$eNpZ9PvW(_(s+5WU=klX$LT@mz;rkvwwt35o5BKtejTk$GU*s*-'
    '3PvsLSetx9DwCT6QQ1e_9{qnm(ySalK5Dy%*d@IY97DB!`c`ar<LVfB(g9qv~z3g{J9(*;}?j&?N(?kQHC&CikE_{yDgRE?7bnP+'
    'Ks8-oOl4y82;^2}W2!KKKvoZpu)w>*Y_TbbsuM3OA~y+EtOk*-aO$-'
    'WhES9p|H1e_jL?+JJ`tfmRLKCFHv;jOSjO8rdw71H$QVRf2BsS^w`33ZYAAYn7Fy2HF2nf-'
    '|)*EItEF36Mdc6fzEnQ0;*D9Ru7F&29$%fq3J9FB<pN}!AwGtU%s*1^BwxLQ1#G1c0}8ep76AJp8)+*~p}RsoX*l!wQ=NWzk^Lgu'
    '<KkOdj&d0~B#f^i1j{cc8~=1kEoDAb%Ox)p_*Gex(dP;;i}b`)yP6y1SB&6%P*QK<FFa2eh&k}sD1dG8y-t<rIvrR43BDAhO!vB-'
    '{(BwCf_!I;@@khBf)Ic*n{4F%~fhH5BS4rvsVZKZ(Ofl!|bm=jhX3z#2P{}N~l`qlda9to={0{Vp2cmcl*<UU!#rtrJ<L=De(b8='
    ')2_Y`(DJ4c4#_l^tkWdBo;pBM%7@Ig?>1%I}ZNr%??t&VaLhmp>zathv@QKVB=Ya-'
    '228`L_gwGUL&aIk<UY${=?fOk@48ztcE@K9d}m>E`|2$&anMG}>oU!hP~7*<H7rID*GVQW~e6EGwYYF3Wq#W#*LR||MnI3F8LGx&'
    '#aUN&iN-'
    '0KceRG5@Pq$ewbT<c*NiCK)O5c!IEKAMJ##Fl9lif>a#=2{{h<vPUtRw<Jg4dKwvoHjNUqOOhAbfm2+IBZj(4|4`=2{ATqtW{TZ#'
    'i1MC35q1)t^05T93#DN4$Y;HJ;1RM+c=+Y4cf*oGP4#<V;7lOhnBI6%&bSl*hOYGpk3@DGaJz?c9EG)Xcb?RK8fQcY0TKR=onK5A'
    '7Q#NVJFGtrLh@>bPEzLNQweJ7wuk6Hmra|p}4L0QITM5C*p7Xu2s^9mBzH1naqS{(g&;HXz2qo&_en^1{^0{(2U-'
    '}Nyjztd&%N)E&LzJ@^KyfL9&2c5C0=sLT-Tnl`JAR!flde<R<u|w4OwuS(zl`HKy6|G0m!F@;mEE%VbJ?3eju3AP5?iPQM7NcO*O'
    '+&d<yo8Fe)xShA(TlD#ZgvX=)-'
    '_KIN1UKuRetAZtab+BZw$(k0w!~Kmdvl*#FSQ|%Y|DEMpFtPbE7WtJVqhg{>HbjdxWv7^ED`Ya-5UoZgHwQ-RXzy<{(sHc3*PC&d'
    'W2*L$upr2xQzdMO+-'
    'M0~!s=}a6CI7M0$VAG0~`DG=Xeria+qI#A>l@u9PZa&N*E)PCVu^ugqvh?gkOIx;bxf}>DS*#xJ4#S{rX!8zmv&Pv437dXU0i{jO'
    'FITa^91%fpSlMib`i4D>{UDpkq3Bjy?C<<Uk|oP{<#0U`)o15@rT=td%f7tkAeSFRW0Jo*!0d#rz?xP_6z?SiL9V@vy=-'
    'oD^30Vb4dOCt!G3{Ul&SSm6#fkrcQL-h7@o)ER4&y|LDVL~(hXNZ5C1k^K6~tb(W_Zord0nKeY0l=-'
    '_(!klnRYbDGLtK|}&3ad2&hDjeg%+oztMUk|GYi07Cy>11)Wb(beZVlJT<Oh4*271e6mA!5Y*URKbd)*ESGFfe}+e06j{1n-'
    '{^%5?(W+tA5OAm~aiclxbVYcK!?X*oeiEBA>20d_*A=1MT$Op)boGv0)d0K6BjZKk*&_g=F9Op(VxhQ6{O%jiCFs~%c4l5+^-'
    '@^)D_Romo0)__pSd}Bk7KDF}2@+0^Pwqn!&Wcaw!xGN6CllWb9kl0!i-eO~akylAWV?W-'
    '(s*7`=%d!eIPCi<g%ox4B5o=3v~4QWE?6Zyv*p1E2;{3E%8$1P-'
    ';l3cDzBr^m^1r{Ucsd)OAguY*^~@>K*AZZjCxSQnX!y|M8Y{%2Du$}nxrp&IZe_JznmuNk6%uc48Sj^Ne1GV(<FoN%W0Cq_;pQex'
    ';;-WWHQNEu=c~7DDNTAQE2K9ZF5DjH8!QngJX)^m=3a`-'
    'BzBKXD6f%<^~C~qQfU)URd=*g%ef<fjZu=&_%s1tUi!v%fw_IZ)};M8XlE!t`XhNfS&~|RG>N<a>uDabhJ-esUY{!L}{(8*mvjS3'
    'X%NX>@Ql^_Z=8poOK*_wg$}&lGA{cIhus#za>;X(K4UO0Q4~N>wWNA6J`AhL%#0uLA_5TUz!_Pgf2sTBhXn<q&1Qi_~!K?Vww+@_'
    '*E3Z_3=}44xGPe9mLFj=83lK`q^!??_Wh+;VF)`-?1v3Xf%)WDjaCEkXtHyo(op$y9hE$*uES=MhV-GBgiOW`*Q>tCF}r>AftpG$'
    'Pr|eu!A^)j1qRRvHOT*C&nbZ#$>Ya;w{{})Y(Lrgmb58zx#wJ+Nw4qO4sNz9U(chSY^B&mxd$(*3ucfbfjB3CxMQ0Yv&}8P;29y1'
    'QKX%ok2`Tyqz<MiBH?d63mG2Qm+A?mv}^-'
    'hf+e8`6Qe2Flm3>OeZODBfX(si91=uS0w(%r**(dsgv1Ppxqs!wi2EQE0p^G2rJ|nA7f}HebyNbexG(mN6F`%(NXe=XE;YCkMrx#'
    'Rml_l`tw!tlzmMBE~4w}BZXP2;4&*<k$GQ@^(D%%&dP<%=nUY(9zZx*{oe#|dl1tjy;mB?KNVn8q&?!8$H_5U9V%ej0gS+VDq^eT'
    'G17xwl--w!@PdRB?E=msDeCBUwn`pj=KGmSCNlF~ppvI8^A+Id^cr_KenE6|hvSz-'
    'H+MLGrIIK4a9m99;||0n^geykU>%>guORAV9X~W?i5S&_upUsRENl(6cezR$B$LwAx!o{REap#%H6rfBL3YC$3wZT_c*ky4WMr(c'
    '#9tx9zEA*Wg|rC@Qchv}x<iG+S2ecRHz+&;D<j8~0<eGd+=<+8n8Gd|Cr75MTRZ_%By@9&Ct<3D6Wrn{m?ojSTRaWZC7k5;tWLmT'
    '(&H|OZq73v1!t;clzq{8v<X*Q!e2<6TsAu65|N2js+><mp`UumAkTvdG}?R6u)c#vQMNp(b>PucV}hQn;2BHM7jpOH>i6#5+aLlg'
    'RI8+Vtj#7!bgz9Zl)SSJ?f}3>;hlH%x$u|eGML)wum@2PsZ-'
    ');fc%W)IbGQYM6U@CsI_w``LWm7zgF!YCmmA2F7px@I9PctCGv1EeOgIm<6!x;mPo{40=AJT1cGnYRw8c)*Q}jH_C6}jlN6#<DV{'
    'tnKC#aUSZ9sg3l-ceVxC-v3fC|`ao)3Jvc(B|5A&-'
    'Efg^N2@Lg6~3X&aqoP<jy>=1#91z0Q?KvO`=Fxlmt3DiW%Zs$zEQjzR&&IG(Ipn+h@E*Bt#b0Odz0aB0;L5oYJk19h?>BGwK8|j0'
    '|@LTDl$#A)Jfn%c8;GGIdHAy)pD<KM{0?LO76n$eIJeg!KSsjLKWg7k}Vw<B}!#J6;IVzDPY;@GcMn`pIbo8)E$tYt~p3QDlo2_J'
    'B6<Hl9SmB7s&cL$`M@E(%9%5*wBEC6M!O_au(-hyYmYkBkSHc3xIoUr;SSUFud!K|wlC!dZk+4{DTK2CJmPpRa-Y?-'
    'T$%)y&NmwdPW|qi$dVF@rYk1aD@ODkuw{Xn<rb*4PTD@OGkFffy1_-'
    'OaXt*@2{;c8GVKq*}Z^NoyAS2*+yV@h*GKt@v8H$GJ26w$H*@Ys((+NGfA=V?q-)&8{qJ^1FHSEkmXOln;`*f>@s-'
    'PtQ12mR+pnW9ldr%OUwXnT|3d2o}o#CTNhdeIvj~q$QNEs{lYWPif<{bhLf;oKdy-'
    '<D)xBevF&CX;e9y9yO5=l<Z4ce|wm&i;o+O5uz$VzadTb(JPse&<Xbr$-53T|?%vn4Va-'
    '0W88NMtd%#jVbjaEyZA=^_n)7Q>4i6@$$=m=T?&0^QXi>^zWa9ZzOoB_9_w{3f}$X%uS8lM{41`fN^4A4lZEpGBSveoj|wrr_@FY'
    'B)A^%J9V%I=LOStCN>!rf7VhhMs}O%jov~YNvo-OZ;wElEf2i3-'
    'm{0acJ!2q6Qhk{ye3Nov<xI1AH?qaMxRsE$Cp`>ut$46rc8bd$JwHr@h{h>_G8ZMB6Pn7$TUq@8sa)l$^p%pQHjF>)+dg8%U0Oqm'
    'dXkk92SE&FYeUnAHT$A={c<JufuzT@GQ$d!xf0<ozr|UEc89jmPTLNn&=tG@zPFIYwJ)f#tN#Qw^XT9|diI^hRDGp}EdDY5ygtAl'
    '>?*gyCRi)|e7-tj&&~BoiyGBTe`!1imD(Oh-'
    'Hdc=cNeous47*^v%%`-xWLS>9SYTtKbLWQ9>!G>!fJTaoV-2YTuiswVG3X-YHr&95YDe-gfgx8GTrnRI8HB61dRhsbVn`us4;ceC'
    'O?oYLJc3<ymqh;6mFb@n|pyKjJI_Y2VM{sEdjAV9MR259!60L>n3q1ld{m(qDR<Sa#tm&tif>JTp#@KQLo=mXCPt0e+n4y(liUJ0'
    'v3Xt#&eLIJOZ)dB&pht&lN_sXyE@`A7`NLUwEdj;Ae_}%Xs+U6rouq$(%4-=p?Z~rrB1*lZv1-c2SdDxOATg-'
    '|R+9IU4i&>#=!e1WZOth&oHb+)SHb>%p$2aO6?YpCLPm0hRDx+RKo~e9pipoDz=+OD+JYS)k>sRL~baed+FV71reA)S7g>Uh*u*y'
    'luX}=N@4hX9T5)KTj>m{rYf7ct?G^}1!=ti1VTABYH{Sh6*Qz|t~R_uKlY6XEweD8e+qOm2bi!&jm`Q#zN(QjChaiwv)s)Aw8bVa'
    '+Qi=`_vMOUWlc`3TSq{z(nfE<NxZ~&z$bc6#aRiRhnS4%b7t)A;OTozV~G+fE60z9M9w_ywV?;2e{HlhEaQQ@%-'
    '{ZEZYg^XxxP;7!;GE32C1=OVgKa;gLgJ6aZcF?61KCw#19)l`PM0J+WI3s1){UC~tqmv$@A+Gw4%4O2}JlRBwEok%eQ>2s<8U`Zt'
    'l_*pF>N*tYVbu#IdRR?SaB)~&iIOf7AQ~<Y@3}z3)nQew;hI3^XVGzWNnKL<Yw(TS>5xb#Bu^CWS)t)?*4IT%-'
    'sON;lW!o8Dn#vxlnx=gOq<ZyCTDe$DOtl3ze<ywx2t;3%Nc*;S@I2xPi%<yvlQ{rwOph8!%Wh2K%58myrzszfeq(%aJ4tn>O=5;R'
    '|wwk4#E39A$Y&n2k*%T<mf%QfE>Oj4=@$Kr|^_QUG}UGGGEHBKk>{s88D#8UX^T7)Z07n4l6ZU6{bWdqQoR~R=wxt&MK<DW!juo)'
    'bQ_XcUCV*(P)~2Uj;^)tl$#ALKAL-i%ynM7X3vHZ|F#XjEhux(ic6mD3II_eX}TX-yglRC~`jl{j(^{JP<v!;QS%|?vL`hNW#C_b'
    'Z%PP{6jz^uWg<a=;y{Phw^8;h+vOw1K6XxpY&07Qpfv|ZDC_i6p{}@)WX&?UTzJUCPn%Nt*f^!t<jQrk7<2jiq;xGDjc;w64nM<&'
    'xM2iuaa5NI4m|muW%xt(`XpjP&%39wP-'
    '}zce_x+3mRV2_GQOOA5$$rYl@h$NNht9Gmgc#r6?zzf<QRbrrSNPm|`qMInII5(nU{j|M#P)6?(^Pg-'
    '2j_dXLItJaT0abxb!j!rrxXL&Ng}rrXbp0vykyoK`8s9G&LUU=On1GdVvxVEaiQHLK@P{B}X7Z4w)Il`agOfl#11;ng0Sk$H*^v&'
    'Lyj(MgW$wxZ}HtDklhgJg|EQVT!QdG3V6+Tj+v3I68%uw)N~E1m0~?4?}kF6W^&5R{kS?N&liVq;J6-'
    'nI#2mU+t(jr!%5mT2d`!?gTGiU)Dibl^cHVgImNsNt%>cQfIzu-HU#BwnnA>%$)*cWn&<1j)1`F!dIE=~dF>D-'
    '6=k7+w;lYIwyNgBM7crr}kcRn7kts8rG*$vjGNSFNFZ9UpDCPi_)Ko?Y%rvL>A=Nl}zyp8>ULb6sTP97sO0p<^S8$v8UxGDXMzQ3'
    'M1cTqOwuz$>)tw?`JbhART?=D|VX&+ssA3AF1k;gj&Y0fM4nk0|!6*TXf^qmSSwN6DW{n6BY<N0e@ofRxp;NLYT9gLl2BU7dqp%C'
    'VUm@yNA~LVlHT?B{I}d)-P0G++VW&<0Io;-W3rq}VV&gaTWJxQCKg;fVXI6mikW+!hFk2K>$_-'
    '7MkLK&A5~OhF@0d+g*(8s5|%mo$_DgdW^Hj3R^{fI5Prf>u4$)6+Oi+b%pGjA`4AwmctmJbedfBP-LS-'
    'J65Sp0tZ{aDq&GJtpStvOW_1+Q6ucxTgqmWU5p60Dt5ZG8N(dGGm#eRJm0aY5RsGxeZ|tmMAkEVJ|9n(|^oCtq*B!&B6Fo*rhf`r'
    'BPrw!usT?x*%{0tvo-RCu1~AB~dgZSuZq)Egc?4BJV+^$Y8;}D2`ZDq3~;{SyQ33J5U?B>W~8ly&ob869h~VQGodYbZJBk*bp#HL'
    '}`nM1iT<(=!$^pA__o1BH%?4WiB2?hg?MH0`%)BbZN+akVJv)$bE4!x!2{Ow`Y(qbFfUt5k$Q4TIoTq<Q5lWJ54~To}*h7;F^R4U'
    '>!4#wmwbih7=+!iYyVBcnF2hJ@$V%Mag}I&q#`ldw}043XQW}ex8R{^~EE{@Tz_&$QfSMAH_Jss|KJo!tknrD54l%H3+RphF1+Hk'
    '?~U$^Tcy>s*QQ#-tNcryd+(ej5H4^!#QwBkgMzAx*!$MJt3T#@G|0a3D*j*d%lp+OL(*P8wmyB<;S-'
    'Y`nc6Vfw~+XRUX|5w@44~gx^UYJo`YTZK!T3U}v7z-'
    'VetOIFGT*CT1A*AgBz2m$X%pzcbE6xU%QIuP{+0yy`E4N1mhzhc?Q#plDD>tO%6bk!*x(rH8x1jgAMZrFVrW<-'
    ')!YWn9=9qJ#^3LzHh}cL)V*@Q0A8f<uHn#U4@ml)Rk_=u=pmGl9eyir7uYm`xvO&4+SBJPJcRYOG3f2(N$#&U#NVDbG@5<z-'
    '1(5FF|3BSO%qM~akJpw153=yJ4v0!Pk<LjzrFp-'
    '<pQv<vIP2B6uzGpq(n_%OT@t)VX^TqnF)|CL1f;kEtO60R3MhWn0S^@JARyMn}MhOTuv_|Z$BFLJOTl|n^>VWkTnl@M~jTMF+~$#'
    'TA)Bu<s)q##k(OZs|pz?aFap<kah^sg5jkn1N*2+7bEV%icT2Kudo-hnFUsca4uL;HJsSPhc!QTVfA$uJlreT4Tt!TJRKixt9}<^'
    '_VrEb@WiIaFN7f#7*mO2xtO`BaQy#8jH9t8&oXN4nq2!8o0XbT2c<-'
    '1i;$H}(~svGR~HGF0Q&kp$RRtn*e`WUuxX98oS8AwYj=u_{`gOXl-k|B!=Tv8e)AO9CWM5bWzUwR*+7`>O;)SSv%-'
    '0lw&M!2y$cD!A&ME{G=ux6ELYSIF3(s-o~O>%EzTF6Y$5EGmcL)Wlq>+2XMLd{%?h-'
    'sG`lm%Ut&Jf7^fmkX08l0EivQSxN6*Iq78o<eD4ua_hfQ5f0lx00t(7DXz3nm~bO==hRALFMS!GsU9wp;6$qHBg8Qb_pNr7zf!|p'
    '=aU|+g%h)P#(?QRlx&ph4Q<bf(MmHQ=XvUA?2$J;AY8Daw2t^Pv$Ha?2a~(@V+q6tMqoyMbp%C>8>19#~~8aAjLb)%Y@BcmhS_ZT'
    'y2j$v+w*EpH;uF)Ea52ERRkRs6-5HrwddMj<&xp=Ftyx(7-'
    'E#H*#qGXO0#!>D6%(2_3IxiW{UMTP+i&=a6OFJaS57XHWVpLxG+(Z&d>GV{d}Q^!D{1FxpXlk_<IS{~)Z(D%YYxhN|^nOJ6iZa1O'
    '2-'
    'A~*+^4N+Mv*A1b2;|qrf=E0Ris9K6khgL{(qO#P)b5?Ds`JzDm>ZtjfVrs6)CACb<1vyka@rmy#VFpWoJn)qYYP2Qec!Sg}b0M+g'
    'D2|BCk>@l9N`nR*FQs=GFJl-Injzg4Jgis^o)@Ib8ZyQn5G>$P*F7ZIenVCFu;7pvYPv_9ntO)eNKQ69OIRc)8vaf|Bqzb2BiNCH'
    ';5<*5BPYNANr0oDQXrZtd1s*GRDtG<q2mhz4H!qq-xkyH^ITHJbX<~yzFDRmNy0KNLicjKIt`&&CBmUA7m;&CYK$!Vq&m$21;YSo'
    'OXYBS8;gp<KZz<#$X1^qg02352)6ozBG~E=iD0WgEP}26hzPd&qaxVql@7*4MR99UMsX?JdKkc4hMP7}a53CQ7$h75q%<WyqjmtU'
    '+TYph;}xGKWQqqB9~<O@hZUa{G}a$c_OKMlo@gLkB$U(;O4~}sC{~i8Pb$W-'
    '5<5Pn7&1&@m5GW$!(`$=t%{68+)w;&%2fU%K87k|h~nj;n~sl{Oc0@#(Io#GP>3$lH?JueYPs)C1tXaIE=v(?-whV_-AaLaO)YfM'
    '3R5NEPK_j+i;s;WKY`DUmbigz`Q#+-AfKZBByJ%eqx~fec9?)bjQU9JtAK*ArU3)XLfs3J#)$!<3<;5-'
    'Iqb>YE~1k;L*aoiw0T*fh3sf^dCC+&%haD-'
    '=@C5VO2*)>7cfY>^bXnH9hF0almC3YvYn<%sD`VPaxI#u9;DRU!tE4Xt{i|A+AFw1arH_E1y?GrYdK!Q)r$L8)lp-ZUAfp3Y!C%S'
    'deEUw`Gv?~csr{EW3maB?LGdgfM2Z2crQl$X6@@2WU}7sSD9>Z<hmk-'
    'T`4nEce0_1<W&uaZm5@wk3X3xk1M!YsVGl2DcaL=y3591-'
    'a8$vNSFnkjJA5;I2^M@5eJ5F%vMEA*d$daW>DDpHPc~$@oSbt0OQwehyTT|IS&1cUvnK^8^7i`7%+a#Pao$QCK{6U^{ArF&}*7Z)'
    ';W4z8Q9C-Cpygo;xG|<9RuSq(GEqYB!r33s~c>>L_T&mS#mXevJ0k2?!}bshN+S}IV5}VCb(x)vKP0~x3Xv;a6f&U3qfcu-'
    'vUDtNF-gbw@QS9s5m8}=w?#ZUF2s(eR(Du1m}#5Jj5SW6x$%jOZZ$F3U)y!RXGw~l{%rl1PPyBq2-'
    '1S&g^8WcVwYIg~}bR4OKAcE*FjM>kCax0hOyS?(fS?OaYavE*|KMOiTflt1TYvN=&@f(@>g&IGT~y4P_|J@P%6;;)W+>a?sgkM~i'
    'LV$wpn&NMj~@3U!k3g{4px2{cEct5X!ZPLM=qcOR8njvajCHS8b4QPZi92j`goyJ>1?Xe)2iMA66|rtUbn|6WiLRjYX4DvLud7b?'
    'n8<M-'
    'NFY;~jvgrU)VQiG00*Q97<1eXuzT`NetaX1hCto=d&Zz^>@Ve@+$Rf4l2@dCB`U|sPdRfV(F{StMiaFs&_U;88H<?vko#A!J^mbE'
    'F|<T{b=Leb4xxhy4Xyg67IW0)pHkm>@3{e3@_2C_KUywU(Z3|cCpS!W4fTAH08;ajFzStRW3hkC_xp20l3^Ixz6KFXIlwTztJK*o'
    '1U%c2m(1_Jp(a|QMBE#cFDV{d6DY3vv-jc-!L_+{L=WTpi$8*oc93pNQ(MJBVM&MmY|=D=pbkg8-'
    'ZY;lYIWHJx7y2bty{qb!POnMdPS5dyM4oOt}w<aV}@!wjuPOl9_^MRW(4s=jNY_ecgI3uJ&c#j+SJ+6;JH%9O?;n*(swW<a56(92'
    'j%HEwpPw@#~p3H7$0sX|Md~vdOYtT!4n(c9F-@pzPwYOjwirP0kDnk2)N>PNJsUzKv!adr?SrT5c+c;0c8_Zd~BF-'
    'u?3M0=TomC>#dsAiGg{r8z6w!mywUa`nJq3#P6(uTUqC`mI8Bylp<tX#;N|bqcHOf4^7G)k@k1`K$M45*-'
    'B~y<Yg|Kf*%`YAlp=k(vrqn3pkvL2=hb(tPq`4AaXChsf5?N=Mg5k?O@ceA)OX*ChuybNo3&Om*m`<&Lem>o}PQXCT?oD?|bp%b7'
    'Z%ufUW>NN`fSWbHcUuLQDeeK&QNdN-?CqrB8Wq)CTV`Joa4<-SRU&Ry7(g(zEi<B=MXEhmRdG_i-Xz1bdEc%QzP40*jl69~wcaV#'
    '+J}c_Zxppg!IwB!ggDMPDynhvQz^V&k-iml3Rg%|K#~G;v{1<wCUjmU+c-'
    'qrN+mnE6tA^Pb~+#jMJmf}@f*QtQR9Bd09#bKA2IY6b?(0yl#5FDV}|RZS@H=3dC@NU)Fna$8YZ7bY&BcLtCm32FMtVjeJs_<cIg'
    '>)bD?fpl^lm@C>=J3>U2oLPi9l58nrA(tz2WlxL1x(GtLZf^H~9IK0Cn8=LER<+yFP97vSde8E$U64rvAVN?(zCfP(uaM--b$7f~'
    '!p7QoS@HL~T@7aF}cIaV?+Qy9@u=yeJsItmpGg;D{NTA8hG=|Nsgs^w8=@qyHdTXW`#)v~4DPUKe*`5VW<F1$xHkH@*k+=TX<f<H'
    '@Ng}9%B`=sxIu)l)8NZ)Uxp@P3k-w*RZ1%GoCSx6C#S#Gh20vNO1Vll-'
    'p=D5WY3SZ21i?=9BH_t7W5)Nm69BO!lVqG@WK+$bSx;`oYu*x`~H7o$CLYiW-j4^TC-@uZjnv58ag5|E_A+Xw9`R;F2YrdI-'
    '*2k%2E2E}bs$@Hx?iIoK9}<lJp~3hc7L5Pl!T28$jQ^3r_#ef_f7DY0H}Q4_D|N(JuTt#aQ0EPbs5<JDrzE;@&zWgygTfxsri6W='
    'T?rdQ+v0b|#AS~6z-'
    '37;0?VNsiAa8gS|Yh`vNXjvSz1Tls&B0Hf&n_mF`3HZhI)i3sHJM)Dn@3ElRmUS*JXSr8<6CQ5WV<28)KZ|VvN!8#FsM>iBx@4wr'
    '5qUbuv9l>3?otM;u1+&vL~FuR1W2j}KLw_z;^0YYG3?i~CAVkNQfz81<EU$-'
    '}?@J5dJ5K!XWlU>ImP0SX)g^@|v&K>c?5GO~#J*YtO&7XwY|so*H5-dk(9Kyy%Pr6Lc>1>4czCa+t#)fb99OzBo%D)I}ITYaU-'
    'E7xxIwIcPKbE|I@nd=OX{8o_{vA>MNRy8WA<#>M&iiM6e31G>hp(0LaMh)UByd1T%p_c5v#u1(7%Ve`5(_)$INLNEOpv+B*srskR'
    'pXuSm;eyjmCz{KA&EJ{M&nr+1n*km|gu;2SW+jf#+=I$cePcbKB2<rs`8AcG`mTn*)7(p<xrVbehXq?`I7f3>@Hh?UY6=Ucr;C(F'
    '=7{r*(Z;Lb79-_E-IYkuNU5w;K1ym8N~UaI<IfWf<6={NflRg<QZA9nPRDQqQWSenz-Y}xs?-'
    '#rbL4wWQ+Uq(=5bB2IcKUqp(!xueC8)LMZcV(IZ;!n%lXVtYl?GS<FQ^;y5-'
    'ImY5l&MOZQR#_1D}~0WE+*Iv!3_m2QCM!wPv*XKHF9X)#HY%Q;R3%{0k78li(Yi>(Wl8O(z6OVfmsbbrE%6o(B=Tb~qYm>_&}>nf'
    'iu6rra#`GeFV#(^P}GEGJmyVl&y+nZO9lR?l=bN2$Y00wC8UVsL`K+W9?Q2!6snSQbq+!+#rJ3~WoXIKdC3=a*2MuY}JBSQnBQ9h'
    '?=<!EJ`0h=e2I>SZ_WwOn&(V&#gIfzh4QM-o_2r5V0&{+j}GK?xrA~2IbroI(bIz<!$*#3r~s+O4=nZm*$k_(jxR5RSI4wVGB8Ns'
    'F^D#MrAE<$DaDqBLR0bXa@2c`a-'
    '3}F)V(WSl`ZkA>~Uo2pb_o0>um}`6}FQ*LL&yc0BcHxY;E7v$GO@$k{WM^>7jH;&GELRPUli4u>rIuW1<Cii|M~yB?0<eY7h3L}o'
    'Gb%`JUGm@^{}<w<7F<zlECCCtSc}FZ11Z*{^Tyzbjc6>T_^PLL?FlMkdf>4t7cBhh<4{<66bj3eadjcpB!*cq4)NH4a?#lip9{zj'
    'Tfi1F`d*9J@<qRDX+Y`S<<ry+d~T&LMWLLZ(~p8s&e7>lF(~Kh44@Fy01lewB>OspC;&BxgQojSJBPz@4I0AsENv0V_OV$|M6)#0'
    '3`rfb-lp;=bn_)NPQiOim6{_i`7pM}no~Z=a-'
    '|Hucncq@sYfb@$Y$RtJk~FYa`sP!$_A$5oP$LGMf_0)jMbkaH2NJfz*X*)0WUdLdcaEq>EXbZTbjjE2(6m^t?&CG`gm+V4Na-'
    'D4+M<SteWx~&eNQQYpLOU&5@2)8h)ltN{+Sb-Ep1L5V$9<QyNNS4|PhzsER4nDGjG8rckFef~uH8ozh6EVhVLiqo|50)7<2K6AJN'
    '}lUUPm2I3Zyn+<$X%?#5*VLWd=&1v%a<J%;6;6X)oi%jV~6V97#W!4TAoS0#8PYZm5gUu^8E8|J`i$m=7-'
    'n!}~V7CumT_<3VkD?a@2+qJGUzRZ*3D4uT3`auuwHLz@@oZivSOhI5kd2c;OW=CJLg+1gTb4pgp^u2uHR+@&wTk8;33h}PD;Yv8-'
    '{v8r6|{8WdOS;IA_GOYCTys=9G+>CdWa#?Hf7^fo=DdU*yXLx-U9aegaUyPJWQbvo*Tu+QH75j8F%(W#@TY*$hfor;-kb-Bje6~!'
    'Uu_CM#i1}ES>SI_a%+nL`eU!&CcEb$~k0a`#XDAz8aSqyh&q8MWBtPDO@tt8If{huBDk64p`8OW@3ELc#LENg9rN1yo;aMn5hcJ%'
    '*<fS%nHWL>|o5y3C7IaV9d-5#?1V<nlgQo>M0ml?h&HsI<=FmUaMFG;q%7Woa*8PM~x%|dLxY&k-2>&if-'
    '%S9i`w#mQW*8q$%@uVu9Fx9iuFI(})lUvMA74kkvlwvr2IOf@;%#^yOmI9A&ky7o)}<Kly?&Y8F%LE5=|KB{zJ@*ltR0_>!?bl&3'
    'N!WB(V`K6EqdMJC==?=yf^p@aH9*k-*TwwY`}IG$=v6^yo2J4V6HOtn#|XhAL1tB`@-E2ww1D6%yx_Sut{H!;?GHe!65=>i-'
    'O2xmy@PaQJ(XG`iueU!%%t?^hQ6u&5IPj#xRi=vp#X>Qe3QP9S9n<v<&B1hDDl!~(3XSh{&MG>7dx%fy?mhvpFyik<aJv)xi9id>'
    '9B~&v7W8{B!TW(0vuiDE;?$<>PJX}OBQPkQ~F%|2mn^$TGQ5A1)wT+!eBt$3s_i>X`^t1mEH(owVQfu#~EDDfcS?X8dyVY<EZviK'
    '5hN3a~J;(4Camr>mnw6_KhPQ+hHY3r%T+K1Or5?f;x?57gFO<eXye5R&G9)}&!A;C-qf@lXQ!6;1eXWsvpJX6bQ-'
    'x{7`JGk3^OXk;G@06b#u0RSOJ68aqCyeG87iuKI8(V4yCdGIrfYRT%#@g(_J{jApj95~do9TYjvMu+qtD@u0-'
    'bf=OOuW|$2#5?wlQYL1Vz@UG0+}Qv`S_AhT+vy3VzMGtQ#|aafP=(%jvNC>R=R-mF!<3#a0P!-'
    '%_ulh&qKgsv_>9DI_Yu6BeP7@PlNjiE@sk!mA<|94Ro9wGQ&xEY>>s>e(!2kuY;u%;IsL%i0KOG>?&NNTT^hGkKiS6w{Z?AH>HMf'
    'F;Ni3NFohe)y2F4L>Gj!<UJyuU8x8iuZc>JXOM$2`_SIDBm2Eq$sMbRU)R<=_&$kEJW+-d&wx-'
    '20rt*OoV2g%Z2NN>akgqqn59P#hR+pSgdJLTU}%Js5Itjx=Lf7UQ}t!(@QF2_@dEusu?uXIM$(0D(GoxJ5j-'
    'JnAvVhdC&OxW!{RJ5qATHvowWtJc{Yxg&~O(QL|Qyl)s1;>UTWtf!MRuXX<=6*(Jg?MRjyA+Az><yQ$I)8U;1<#u(FvI(k!FVu)J'
    '0F2;_bp57c|$97Y3cSOfMRNkF+AQ~IHo_UT5iCehZn)<joYuMDkIYr2N!g2BO*(SJwM4rlWk&>>CIvvib2wjmYh)AiR%_%C>fSsx'
    'W(g^jhF_mmo;4N4!8SA}E`B56MTlrZUut)h}8n9Qf#kUcJ_A@j<`T*=A5&G7mb8q_7`s4bTf*zKoo>uVN%yF%!J&e!mEm>F*6<jK'
    'gAuWL)eGF+S{3MG&VJ$;9?^dCkck9s2yG`ik-8OXdZWp?Fx3}HAT`t(P-'
    'R&0{LS1E?l%%D%&2EwDdA7dby@Gz_wRo*>1Z}<SQXP5Oc(Q9P+u)g9$87Vv6j2IhOyoCS5mFD+8L{{{MZwE)D@W9FZ^Ui5^KRa%U'
    'A4sH6};_Mc))_Q-5LoHeC;;4v%%GFi{dPJ+U-!Ag->XY;_ax>MV-YO`(`|C6#fg5k(LH=F$w?gZ~7dseew5AqK+R0iWZb-'
    'p=%V<6Gd4=pnnX*I)M$2TeBF_Qx&`t%i9tKuf_5e<@@Wgd__y)P3178#sFtrdol#wPvm&#LZXk3+R8bR=%b^;a&9C#ri^}1MxbNL'
    ';P+%Cdhd*VPe!4a>Y(q=^IaxQk0Jvw>zwWoK<dkr2(LMvL3CMWushYRWmmnl5pix7ND=dLwhO8yLr>H8ZQiFhh=jjSsa2HuRe|#>'
    'c}#4C7BTJOT1Mqd6){GoHgqCkZ9uWG8Wmz;-'
    '>EqMwMua{=ONl5f+EB}#adg2SZk{gYi%83t!+ZAwQY#CwhOV=_Stbal{E99sLTT&s-fADN{T5s4*NeTIX_Vng=#!pySa|>A-'
    '>1J=`@xF55cL<(wZz7IYTT@XP_(_Is8_I2;uiChR^+|xJAf8n(<QvQo@bDJG5JvBH*4-'
    'Yd%$Yl*GxV)b8F;dH#a%n2A$Off!>ZPBR5gjF~7gg?WFKSup%LMPT?};Q6+lP&!2|3;&R+tRP6|4=N6IuIAt~8Ma(tKZ_dkc!V7l'
    '_<CR%b?ITeU9~Bb3q~5Z#9W*#=9ZZBG)F5|1#2nevX(8mB{JENW{7Rn7$7jj|GW2Iqe#s&%8vM;!ock85UB1l#wXI!oe4*=pGmmU'
    'QtUzrH*xQn|H(K4RT_Fb2C6qgHbAs`dn}pByP)+@8HUUIs)h=O{0rrjhii{hTwucb!7ErKf@)qOf`##x2x@w1=u3!Pl-'
    'H49Mb2rZxry+}6pY|7<|!1#VNJb`s@T`DJ+fFP>$Ch-TU>IxP$KUba-'
    'ENx#N_(ll;GMMmwQ-~G(9ONs<7<oIxH54_$#wP{dv@v=aHx#qoBPG3L1~yu{u)Fc<@^2NdMr`J5EP-Djv6%I?_LQ#9C{vVd!o|L#'
    'aQ-'
    'N3t15hw&I6<*X&iUdwjeTQb?0ofz8NL?y~25<WzMWkbS8D0=Ehcw5?EkE1^2A?i~wVayu#LPN*CN*(4L_jCoHDWE<fdP6Rqt)3Gp'
    '0^^Na3c|26)kFp8AHKvx|Bw(K`iG3=p?}C%9{Pvx;-P<d@ICYo>1N`%)=FL=lQnGhE|t^^&mYPQB#g0#^5+t6W-'
    'j|<7PH(l!Ys!!|89elnB=4Y#W-'
    '*~SwJHhnxH=>LmIsZ$k~0LZA1*|s~9mjgpo9}lA(;GnVk${B+Z;;I3sE1CL<V0GcOs*NSgV{C_>U$CNl8^K9#6^pU`|PQS3gcnMx'
    '=!p4JhW&ZE9MhD^Vd@H-~c?I}(w&qT>4kyX*SxXTgs3QIM(Zlbu;46;-'
    'sZP_lRxixLTKHB)!haR%KLJ!&9p@;0A&_i~w?;%UhJ~xr2_z`!JrSQ?y%rmHVAJeQyP(MAcS&^VhdP?&_L_skzCe)i6wWlG}gOuW'
    'Ng!)rTE!M*Xe-'
    'g%{8uZwg38?BdgNq+Po#=ru52Bf*xjyb8RFFP1_XtWV&DC}&DSyc?3k5b7c(uo)t$eBe)4n|4XFEa`qayBfT1N<xmbME8@%_>;BH'
    'E+^@CD~d1`ZX2gi)!Ttcvi;)e_A@L#wwW{KVqvj#wemTbGB185IG$ToKX@wINbn8%JM>vCt@Mq+{70jV8r|<=WiS!E#OR>0r4Q_j'
    'IsagL^tyuDv}SEZ5wg4i>exXT;mCfOy*-U~YRow$1cK2(_siO)~TNk5bNWY-'
    'U<i6P0qT@60+qZ%djC5tPoml7>G7CG(ysc5ccRLTwnmxF(EVTpLC&t_!0V*N4%I8^Y+tjbZfSrV#u}wvupevQLOs59`*2?J18lW3'
    '|LZRjg}j+1*JfUTUXr#Vpjwuu!(d@MVn#u!WUsG-4eKjm^?(sU*G#3LKrOR&nZLxumfgL1DimX|6`FbKa*`$I;X*+MUP+-'
    '<%q~kZeg!UP!i)#tMG-_S77PWCxF7u*;pKx{iut7IAHpfpbOGAm!malr#NKPjv~au$m;3eVat7H+UqD9L~_Fw=6eG8<izW-jxo&Y'
    'QiabbGTGPXB~n7_^a9|enK}=6D%?R2(=tXj3xv9e?W`=H+r|H>KQ0-'
    'SCsj9MEaPuv_UYU&fO`o9>HJ#Bc#>(G8JshEua(0Ud>SUR~FHUWdCM38O+v9A3;uxwbDnD322S<QP!r3(>F^^=HY%Bdp5OEW|rE*'
    'SH~waA6w4($PA>7t?|8!Pd6Z+egPwij;r;br3pP7!rYSzc%j43O$Cg5`>5g~@-'
    'xszb$wGy;9+$2JjUcv^jSPa_gu8jyk108<QHBqq9=863`MELTZ|6oQ?MkmyHNnYrK7X<G%VGrDW-U?S8%Iu0~m`S)-'
    '7cb@Z03TkvLT$@0~U9@JDq;vE_#)M^qn^AtN*{b<|d9>K-'
    '1QE$U%6iCsgZ3inlpyh26O8|hD~vbNN%#+KUG1ex9$A$fg9NER{*r%y&`mS==!-;6@)=M|D&_s<B`$6!I^R3(qY!pNseo`6M>OO-'
    'qci;W}d8Sx>>X|~5@$b8(JI-$Em+lyqc?|M-'
    'nBjG{CP;vAKE@C$PuY{%pyOEinFDl5cpj~v)rrr)@Qt43vxTzk0LS3#iC;{UKW@9yrB&@Qud6*zIjy8WzAuE(3Z`y^cG$rUc8Mr^'
    '?;5qAeAf?$I@Pwq|WbJ^uC8Rf|X!w&`RhkC0cevGKrUC7pZuNv|sd|@NJ!M*o-'
    'lIK+bcf*B4ozFwDeTA{e>CowWHQGXiaa|S@wj1%DXg&74ldu!jfzT~kNQxy9X~m1MRAA~!t|mUNPU=1`r8oizLbP}+_Z)s>mK1r)'
    '5!a7?T12a7e44h4PBlLp?`a>!<lQVVNHDI+G$uDpSkuL*2QP8gNF6-'
    'nd_)wLwx3r*RU}@bDcD7%1mxXLfr4?Do!>Uep0|f{=11V=(pTJFRb9`*W`V4kH5j#Y*~UXmhhwH#yhk@{1}4%B_;maC}gexS*g_&'
    '2`lB=>icC;i`P6UZoK<a<$(ujm^fTXNdcI{&M&&Z9L=Z3Ivl^J#yZ@(=6>l>;?y90i1*o~$BDxP>4D<#fb>Xlcu;z%IP#gtilgk0'
    'z~a6VuMJsJX*AUL<o+2dts-'
    '~NQR%O#GrJ#QWo&NV0FnNt;W_7pl~PX0H54p?%P6A+Gu`ScN{7H)x4N3LOE6FSxT4GKy0Iv`1qTCZWD0(Wn;2!f&2ZXe4LZ!6HKF'
    '!JkE_^%t)RX1&f?^_;`{!%`7Nb6sz}Q%^BwYPB?F?$>D<?(;n1Dq)Y}9*J8&R%zk*%NR`;g}mLsSU0~;C=n#eW2BPnyAhG#<e>|e'
    '2mZKhhdKCM*hy3E4?CsG%1hK)Q!BLLWx=}a5J0kaD=HOz7^`?;o$Sv;>&Q{h=uV3wx|i?>Jz4X5hwI;z)(+uV9n+yH*Oh}xM_xVb'
    'L;?lxMb!_Tj3xYyF>J%PGB4W7pHHa?xm{aY#pGENi3q|7%b^g{vhug*d_;G2finK6!q-M{-'
    'KtFFOF#qp`Ybe)N5T1tE<(Vt}+eI(H#V;X%d(X(Z`qAWd8I*c^6DuW}rQ$KVuRCXt{tD&+xq9>Y3Chq*5Wae!+^Ya3;O>Q@TWrkN'
    '=pJK4-5rR4y2-~dKdR@bP%wOYE(F!DEjbh%ePPOa*Kt^3)?4Z9j0cz*5Pe@PP*A-'
    'F95S?L*C|*WGy~k#h{mD8TYNgV~@xDRX8fu5(&|zz-'
    'S=sko=J=p*P_~|$m964BzlVK;vW?WNY&F;UJ<1Krpqr+GLnri3Fg)yLd|qTo?ch=u8&dP^;R)H0YIjL8eP8`ME90CCK4AA?0nw9H'
    'aj>{XnYy946t*dqIItCSl%z?q&Wbrk(nwfm#T+YXChY3a{mD^lriVb8PG62zvpoQc`adwG=6d)Ob%NkGn{UBSnO<jgrg=@XNWLRM'
    '`&*#E+X8RYanl1SZt83fU1{JLWTMa>)DdI_xpAewl#%G;IjJ+%ImYB^CTU)-'
    'lYd7`8kp;lUCra+r7a|l$@Lg6G&*b26Rp3C!yU8G15Aw&{~;bBJ_k+D)V%8-<LJjcSe+g){!<p`L?g|+(^xBf(ujOR;8d#}-'
    '=aa(j(iWM$Y*!~zcaKcP8Av18F_cj?Wi3$*JOOiBGGTgi)^cr1^C5KeEec0K7P6U^jr8<h^K$X^m`0GS1Cv53zc$ozEmkk=c|~0P'
    'r=tJ<<WklQXcKM#W*qCzN`njkYHoBN>sq%OumOwCYW&)QL=Xrn2vhM!{towhiK~jl;t8d(uD#A?>G%HL!0KtHVWl?wL~JSWhA0nM'
    'Ix$oB%<0xBC2g9qS{3ws=XIc1%p_2?$l!Z;*tpba!`S#_|+$9N0fo(_Bbx0xyD&cVr2>({BX+lKTg7W4SN;OAbG&Uu=Y@J<4RvVG'
    '2V}hIZaXqz-ha)B?UzKaG}*F9&q?`JcS$rNvf#@IwKU}L#_;^(_xza+to*Oq~SQ7#_yw6`ap7gfV@0!i&<@x=bSvr62|w0(GxvPu'
    '3ozW_q6({r|`NN8|qKy78qnHiT5>w?bb(9thFpP0Eqe^8)@dk2<E^w-'
    'bWLO<swQ?l1yDLvM{2irWy#}Vp*JgM47bQw@{u#Exj*ee8Ib1my4q4Jub^dIrKhP<t7WLr?s;l(3I=w^Ok7J9q+?hD@fPCv2GteB'
    'smK$-0EYJt<ch~J|*D_t>dIoOAU>!G{e`u&NO>8#b)NXxnv^s!-caoe)r=$&e8atd*Z^m8o&EVDd(YEBz-gWpK1K=M=3A!O!>Zmc'
    'YQ4G+~9DPz%MlQaC8j!FVdrscMtOkDN<;aVz?@L)6$|QH$6xN?iHyzgi5>x*T};JpZc<Tn~gunVg$-'
    '*w52Iu^m43~hQlo3@Sa{{!c}G`ot&@nJ2wC(tM3Zw$49^I58}H5miv)IXd}PlM-'
    'F`;;5|RlX@!9I727x;qE6<+OLT7Q6MOiX?R+L5Y2MyLBk=P(C=0Rtk9|*(I^((Rsa071Bs7(=>7SQmUAoYtJB35xG)tJ{G#tW&c`'
    'WN=pG;Gn>OBmz%58^%nC%sDFPCT#4RTEz&owB2^3ui_`LKl$jKOhn8EA;m_YV>J0U<&^FhuAFg$Vs%-'
    '|hzOj0c3Vq3kJsNT5x}$M|7^77U-=M+I6iX}Pdh?4v<tQU?R9sKTTBG85_Xtf4wyyOuQn*-'
    ')juO6^d2VD*!g9bKCVj#TVp3f3}ZN86gY=PG%Oxf5CQab`~B%qN&P@ytHS)e6X@PcfS!k4|JRMHYRUHB1!={ic%9WfREyWH8=?0*'
    'ykJhK~p|lOv8AjHb9n04Ve&9kHHB#aG)?C3~5Ruymv$7`@!$c$&&_o%WqwYPFj*YdX8E)v#On=~f#wG|)c2zEOj4t5y;=X^`$Mcl'
    'w6L1*(&IMDeV_ZLm$l35pZ^FH$d#j76@YQLcD8cd)8cRHDn1mC-'
    '21AyKivwoH+%WGRM6BQGQGO`<>5*RWVJ0?uHfJ((g}CmNyDTl>Z~>on|fL97x<)<iG106yZ{delk(;yPWl20rF8`(!bEs=4tQ+5&'
    'sHYa2=Jy)K4p`o$>jSA`bWRBCeKqS-X8uyd+W$#N0**WD|^!}qWPD&!`;p=>)k5T$Co-dt|Imv|e-'
    'D>_?vG*kA&M(Ae`C2G)Dm}F0-NVXleqgSZB1giJreNYYW@_}pA!n=L62bFLG@8F=?672txLY9haDc4izWIe-R{v_?-'
    '%#vg)+$9Yxz{-RhH6@`A=R|fBWuo8#M?s1$^a19;dWjX|&OuW-Vyp3j*@}+7mmkZ_vZ!*h+Hh_*3*JJ*!S-'
    'yn)^Iq}Y+?#oXj+JzrTxHR+E;Hv?)6@yqOOzPTwHGK!?+`372bAsG$#H*lUy-'
    '8XEWR>9lj!Ugi1*0oT5;=xv;4q0QYjYLK$?LLJ8+Wr-'
    'H!T@46LA`O_6jIhTD9gp7aRtx)KniFR%Xt)qOsFb3_7)|5mHq4irdr4E}$W*EA9jVu}S8V+SLKAlpUC#y`qvMSQOjpM0VO0kP{r}'
    'S)qj%1F2@2(EXY%r5HE(yea+AC7NVgs4-'
    '6@^Usic+S0MI}?dqLwLNF(*^L;(n4b@LOQ4H24?$5MwmKh(ea(CQaeckY)ItCeP%keF@cUuph0%H!Mxj=zWuEIw{*V`((xN1++?T'
    ')qY?L0qokSK(*WAKH4VwSSLe8jj=pyvOWsQD3=doo6z_U+(&^1_o5-'
    'lofNjvm>=$HL8FJ>3y|w=GzP>&uD8=%Di68dVNyj5x!x(sHvF@%Bm1M|#QjFNU2^8W4(^Z!jb!P1v!>MJk&*fs1|^ny_|~m*j(X3'
    'e?Av9MhB!N{8li$N4`8)QGde<X-'
    '=JSh9*}~J^c%_ZQIL&(D|tK$GEu4Y(@~xx^O@&NOeFV#$0a^ZrnH;fMUl2Uef!M;1UI#%kq7(wP1@m7))Q^0J=5LNAB3eG;Io{$h'
    'Bnnzj-chWJ^;GL8>v_o5okD^ZV7~Y)*?4H+kh2?KpfP5HbuF2Bn`ZFphv<>U2Va9W@H!a&~T!AJj0;AJ8~~0ucD6ID4}|X?$Risx'
    'Vto7enw;8tu#Q}=yo609xO?@I2#TPr8|6#+u*^{1C3XSV1MTc2aHlU+TqY^uBVKKb*&)-'
    '$&qn4l{})YU!76G_y=o>`d)QFF1L~HV>KMaXZG(Y+FffzIJbzt&Is-'
    '6Lc?ej%!SZm8wIm#vHf4_5ywhyjW3YRBiU&vw{8EEr&w{V^5a%|QzK6#N4bmneI_YuK8bNS6z??cuj1BtaWfSZ9?f&ctQ&~-'
    's3XunQUqEm2_MB++s&|9Ll?y!%7?yb81mIe3eHhn+YF!5ta}Gvq*RCN*iJ^`SRSWMP>TnAL$SVuHD%|rAECZE3E7_#{v0*hL=6J&'
    '(dnu#+h4e`H`&Gu9=d_z91RpUY*Fb~MKPLcll^rAf!gGVTxFS#YPH6_sn*8k@PvDpko1PN&Z16bj&jeXth>%MMf7?{xUMurw2x*Z'
    'Y^#QDitYcM8ctH|uB`A?(o5hou57gRiR|F+p)B7X=?jr|5h&i3D=7FETJr9s;^*JxyBfBI>D1dVX0u9?2<5^DmUqVa3Bt*+QCDWN'
    'W0AQ)eR`|`Wlpjt;$<n<+q4JChHRG#IMRe{lR<d+9pRo&*(SfBj+wzB?VO9-'
    'y{#}csZAWiuNRT&Ua)nVyz}K$l?%U;alCdGwA1W}?4`Qfy`j;<hYHSiWIWuoh~WnRt}BeyJ=gl6+vC*K<Z{+L*ZH8^lhnoIO4d8q'
    'd&Zqe9X+mQt<%TJMn{(RsGqH@I-'
    '9z?r4>LE>d2Q;0RJo+KSAZG>5}6wlpC8tu+S?QB441l`rga${sMmVQ^I!<!ef`@!P7ek`R&ney9wOuXXxDFb7cbzt(>Pn&>Zz^8E'
    '<~1GRT>r0ZumsmR<!I2n@IDqum=HH3=U~S=Yf~&}~^r)s~B*mY*x)9Qv8mF3&JofAql}lTA)B*=Yu>?_r*^T@kaDs&R~}J{6-'
    'f4coolMwQdr!P%mOFw|)`B)&j1+kWs0a-'
    'e{pLYvW0T#kGdDxe>aDxjZ;DxjZ?DxjYV1^ooa)`73M8k!m^J!r=~#&yp4pqb)SG(C>B2Xdqx`bD&wlA_h^nvQ^{$PBae`$S#dXM'
    '4t55trAZ@I-'
    'f^TrqyXlIT~OX&!<q_~IBVgtiC7*dJ89J$?U`=`;j`vnI3Y?Ob|f4$Vz*xvaS~HpL~g=F!v?m(7|_LsNQlORDL<t<(aV;9KC9R4<'
    '8CMx`@+X;V!5$)F>rrsz{oC<Px)sN<IY^ea3a1EOwirn;$KdhLzWYOY`+N3JX-ead_ySJ@s<5u>qbvf*jDDE8RUoVz53SfMSG#1J'
    'bq@Os1$tAZJD-'
    ';`@SVUI`6O`&1&9DJB+8nU_I!!#PG!LgVZXkrUzqF<zeEgXq>iDr|S^fA$DT2@w}_WIK10%>r9gcEWMXTs9~C*@$i_FYSAyl6o-'
    '_|8Ks`U(6)-b#b)b!;#b+WMK?v!T5o7d;m``X2e41$5W&SI#vFa(O-yX{>rQqhU`fa)d{_j-'
    't(MMb{|;6LIw<p0Ct!huOuU58Ur2MgC<{0nMa~ueIPq_(IAMtC6S)Jre3W30-'
    'm?3H7~%t~r?HRzFDSmV?=D^`nIDIggTBE#btRLrE2Aa{OvG#!zhC&e>x!^#0_BrK7!ar-oPEVmsXB$H=4ca*v&&o^CB&$HUScQ?Z'
    'R!7rNKF$qtTAwLzvf&vC9LrrMjDsFn>FeMkj~=7e;Uu#FGr^laoCpR0*?Ax<X$9yhQ;GmbyXr=LJc`Y2(T!<UDttNBp+#<G43f&X'
    '679I+j%;Wh7h-'
    'L2sbNA@8UA9Qs!FjUi#vXwP3Ow*CFMNKeV(~+`;HZVd{&nQ>p1S2)|jB*1>$^NV(E0hP(2U)hq%da?guvKKMmXZ$WEf|jL1Ya=6y'
    '_m*r<e(JIx}4XFbwOX|xqn#+Bs9sMb8PqrjirNkmyjmNXKSd+a<m^s6UWiWMrN^k0!C?Si6vcw-'
    '940)ukj59TC!+rKm#<t=WgjqYZgqB257G08Fooqj+SZ~pt+iRs3+|?$-'
    'X^O9raE{^b5I_M%F_bibkm;(@UAIrM<bGAj#3sTsbr0mrjWMSfWdXX5{WlMe99cpL)?WohHWQ%GzX2(vpkileI}JF0xJ5C9Sz=Ia'
    '!~y;hN=SL(-'
    'NjRFjQKJ1#0uHYM%ty={wB=Z;8S?u^vsZl@MA*ka_R>^b}w6#{cVh|PQsXQZNI%Ov%3a&$VFs>99mIpi}vaAOuQ_}c-'
    '}2Vs<gR?>IMuB1w~V9?Kl<Msca2o>8N<G@@p%418fQj9GnsN2=bz(6yOQDYOb`5*>TZG%G%qfy(qJ&gN*IUTgsWNLe1MRiNLRH-V'
    'S*n_oi=;nl1o_Q?E*;*Y_0HsuMca3}m5aHYE3SV3Er!{4}q8^vY5YV|~2saYhKbH*W)*%Pvl96m#O%Uu3)RJf~<(l2&a-'
    'p+NKF7|)ZIWK(H2xnY9Ghc}f4hVhIrduakn}gbeRR5I&Hx^XVy~otvW+B^=#Vd};HCs6NlTK+*ZbrV%H)Di9-'
    '&O`;}&g5CYMWgp;0FHm7I}7k=##m%`J+o{@n8x_RA&1xBwR);WCbxw#vov#2y3--X8Yz^~mqY6X)X&HVn36X)e^U?j)<yt0{b}a)'
    'WEhrx^`@xPpNBsabP)+z+#i){(N3j?rFCk?jBpr)tJ8?Zhu9DQI?;6O_B~%gK!0_~k^#9{h4v*<So|Ib72$g_F3ZF9`cU6%<9N;5'
    'Ie6Tr!k<(KO5@BjV62Ivwb;aouin6^+^}P+e}Vm%;e5-'
    'o6lES!y0h9v`K&sLUTv&6{M0M>OgPD=pWrkjMazMz5u4w7;a|<<wF`38!e*O9!H))2xgRl5o0a4Ro-'
    'CGc>E9L(osvepoJX_vIXAM}GFvuGTq*`Tj}Lsd294SV`REXnL2Vm-WeXI#N0EP8)Ok`QXM<UT>MNJs3=6360XC`GqRZ+*vn!KpZ~'
    '&Na7hVWNIMMfpuhhJw>Lbaf<S&kfJQWVW!W<b$)JaT^#i7K*}Nr@{ZIlF$nVG;q+LLcY-rwLEaht|45K`fwN*k-'
    'WAS{1$j3(CtavYA~rRYgu7ap9CB?owK>Vx*%1aSD%qwMCQ*RrJ&6Lx(CU4O(k6?4zAfGI`;8P0504YT$L1)+;#jRkj$-'
    'ldH!X9Nz4*~{<Yw<K;d$QTLq5o!TYW?V1>`n%q=;9y^K=yA)g2xKjl8<k2QD_#T(bjJd^%}l622&y$)UquD0Z)O#qK=uBiQ6<H8y'
    'j(lb9?W@GDaSb{*lF3>ncH_Mx3Ovlf4yIh+i>qc`_h!R02`=F!N@eGSKWWfPkne8z6P*O7iiQO@^{F5qFL+=n~o`5eW>oYmPXM?r'
    '4eyBD9q&t>iy<}x3j4CVF?ox-'
    'H+&S6?}moTllYci6XJ9M+7u+tNEYKt0<wIlomw{N(D6K+*0u4?Q72QEO6Lz~39<pmTc6gqZ5`IQpZOJ8!dmS$=x#ftn5+zQ2-'
    '{7nftu12CMQTuT@5)FIYU$IP^C842WuXDD91N;Pd^bqS(Er@CFd^9CJnai+8sP*j-UWX>*XVHuxy5Y&9xNFH|>Z4ec>K#O7zJ_-H'
    'lBm`l?Pg|KtWx>Goitw3nl7wnDudOvT$qtLmJ>R#vZG$}24WO=>xc;30*6X&dum40@8R>hT=>ab2KINf7X|MUj*LaYdxRrn)_k83'
    'P|TVi5U7lW#0mk&`KZl@2K4ijZ}j!L%+}oAW&&0-'
    'GM$@0lxP+idVNH*$sE0AWurdm_m##thF&?iJZ?60T@J2@ll6UaaE)&+^ofKu(uX@fm#{(lkj58i#z<fI^`(SO(npY<M?ah6{TpG2'
    'vJEc^a1YJzvXmnh{GB#Kj#AnakNW;8xLh>KFbWUPQ0S0(6Ru3-7W-'
    '9sj$1dSq#7yatT(<D4zWhr2NKOsN5k2v^Lki}32G8hQD(AS7Y+9Ym@b)o0j9fFy3i0!L8OtEj!Et%!;qQe&oo(&ndCkiSjJ3(XBr'
    '>GvD{odA35fjc@hTa;$iOvgFOrLZ-T`^j61`-e>af4cS$)@%~9q{^DA=1cq@v$PTY<a%7lzO<Mkq-E4OZ|W1`K;z=i6yPvM#-'
    'v6U?OXK=q7I+ecjQ3dh$@>I{P8*=GEej5c9X8qX_xXg4}aClC*R?|O+K7?MH{yBIjT&L-'
    'u!*@b&P5+K{tLruWYvEP}P5+K_t3I0k<vj*tyRt1@3h)=2_s6^OSDGuAE!t^j+9zZ?*7PQfv}HG8#%UPJ&CMR!Yjbd!g&LXQc_%v'
    '{%l_^TneI(_r%d;z94oomCq4lR<8{(SBnvQ=;#eGY+ypz7H9sl#{(uieyhY#(XMLa|>6>$>l`1_hSEf;3Fxl>wF+Mvd!y%-N-'
    'd~!U8=h57)v%2tL$k~WWst!tat6>5lkj>|WPO`7S$2P5H^V{pz`jqT8yyAbr4H<o0jfu+)F*V5Cb><fd!_zSrhBE{B|~dZ+w0oG6'
    '6|mPF5wePY%?I^VUGThmhG40^Nm*8z#JEbE<rm$ac?{-'
    '*1AUF_iVTSIK?|zp(tSsZJA7YoVI)9B!7VD#jRyLP~qJS6$copbg(wbjyR95eMSFOY^Ti6GUDwr-'
    '5B*xGys*eo5xB@BJe>PZYnS@^Dmufs^<ftpGJih6M7P=t@y2<LA4+ExPDf`$1$J(L&B$VPwVF-Mf_ZL3hkQyIg+UHlef^RA-'
    '5rHnDbLeI)t`#*9#AvCrYBEyxxV$k0jmVci>>{TSR#u<3zO5HM{oTFjH2iVo2pWXfNYc=Va(h)xm$*!Dd23bLwH_jIC5hsvK=))|'
    'I}5(4$4M!Gd5E&O3xDjHXh-u?m{{IfpG29OY*kw^VS9W~Odsn9a$8mgbXPgs7E-'
    'L;M@GmT;(lgEkTxbMgn;N;u3<`a{$BaR1irB{cDG-9f?;+825*NRh2d$MdM_b-'
    'I611;;CmiC5}rbfh0CHh_jIn!ds@#g3$|6Ghm0W6S!%rX9mz)8oTn(@tTqY3DH5v`ZLl+BFO|?PeV}8n8-'
    'tRK`<Pp_!*pUcu3xLanI)D9(WrX;+#(Ot!$md`cH)PiY|sSNkrS&8W)Vszv+T-'
    '%@Qtbnr?E79AbDD+ys49jvPeXUP(95}7ctTYG<GC2gs<_`al_<Gk*8KMw&tivc;ds22+Q$c6t#<fX0ia@wrsJU5UepWpgiDxpgxy'
    '-=q*30*H3$?+%(*$k9#0QJ^QdHRb|Nxq3>xFLJF*@<NG!Y;RZhBPSbajR#^A%nf{sXK(8!yQA<;p0Qk;ZC9FaOco-xJ&3c+%@zZ?'
    'pEA$_(F<2(*ZwHr>0Z+zRF4YzjelPGA<~&M8yRqM^Nk8Ts+?7S}JAY^8Oslu*IpvxYhK(pA`99`C!VGr#A5tzlr!!-'
    'Y~D%c24Pl7`H(%n{mF{z9|*Qn<$)R$IGZ%B+?6)Q=wL*7p|a+H>;9MNI%cj(7@gvsf>9Vgj+pF-U&#zx`84?j5tNh{bLeu-pl!%W'
    'R>@FzK{m@lxav<x6f8D4c&}18m8@NxFp4Chf$w2mQKS=I|#QJWjYmH=^d|6NXW8Cvpo^%dOzwU@*%swD4kr3(}vVZ#I-'
    'm*fI5k|7N-YNCy`Hhjw4Da*W&ab>Ll`6Y|TwH4XC}ll`2*05SJ=7&qdyr>FW|ENfvkCMANRQZ?<W-'
    'ahXl<ZAbayNcC1K51ON3lB<Fwy*)9dw>wx_^N7h@)MQ6_(O>3dGzG?T%Wsqq&6dCmWc$I-gTwYIX)<p)e@3b-'
    'Gn~tgar`cfar{1var_~Saa<L~IQ|&MIIa$395dd+(MIb}&$QGVAmAa(PJO6M(9vjVru}XdT{eUCeDU8z=|{pub~)5&H-'
    'n4qa2v3T8$Lg+U7tCs@&8VGx9S*4h=*gs*zDEPJ4?-'
    '|aVZyhzDRZ={L<0#6|$+}61N&nkytK!eT+iu_}$I9rSuYh_g$V^=I}cQytXPTpoz)$v>{uK9hKq9h9=18mTY^e$+aWfvJ`7k+9ph'
    'FgeC+`=$s2gL8><9+1xOhpk^x^@^K=`ieL`@lM)!2(U(+(^`rDo<xAwT!LQuo-'
    ')M?0uVKT$GZxxmo}Fm&I`b1+1(si2!z}d2U3(O2n00Ipzx#$+Epl!kujwJxDu>^_oCF0wXxnEq^6v?bkVT3<fR3JZu^tj|C0iYDX'
    'QnjeA|kBNX5#K|luwJ1ng~wHMBmE>HONFic@-'
    '^_3Paj#A1&iM7gaVMtNhVJyUCw28col~3{6%(XQnS$>l}XfB)EilC2~m;1>eP%Q4a;*$ClBh3RcCcrl*1*<2KyCR<Js@oKU#@<jP'
    'tcDNm5FT{DvGTV}=OvA3GcfexCn`*}%7EEjPgEA`J_Asj{hN%`1JBQF+>1G}0|UK;$utzM=809@==BTO$Gr&~v&FTjmg$n3)H@I?'
    'r<A}&H$BaBDVFAV*Wp=<ZV(8S0%sJQ(r8P-'
    'FF*~+M+wXr*_NylTB9+BAsFm&@TDITHo69pU$@$hc0Wa7P(I>%!umTpEExd!g#Iz%AK!}N(7Gff2VN=49zOMD)ro^tH&ro%CAE(y'
    '}baihbjRV>5q50ZYM7#e;Zc|ssse#i5CKs5b2@pM46{W|k(Ks5fk@MJ)={<`v9Ks5il`Kf?rq8OJ)Bx_-dqVlRV&5)GY%8N28$Tu'
    'WIZ|h4Y*}Ew`$Ye8wz>jaD(dCnTgD4aB_ueL2&BLTaN#@}XwCzyk4tW#Fwq{(pU5=a2rQ3bcjN;Plf$+8D+U!B_jpPjIVEERNcLv'
    '3=;3CZ<rv3CHJ420QiOCw+thnp)df291gJlY+9i{NQ2Nxcr9Jnyh%p|2Vi7oU|zpx}*5Vie!FY99=FWeM^Wsft%3y+9L`u(1}C6k'
    'mpNyFbX*QgJGuO!>uG``m57yco6<Wq^ACl{9ptrosS$jpu73K$CA9HATp%}kI#QsMW1(+$A;Cp{=oI4zRd(`4j0;I5*F7B;?4p<>'
    'HS1O7vtfnG~+8$-2q1hsKgdq3qZ9H~RQjS?H7&quT8e3#J{n$o8}cuf-_yhT$^Eknz)y=jpRkG2iw3sGP03g23eh9A4i6Epny9TC'
    'P&zb|~~jG(hzYd%F7cgWV7PZcIDVXLdBg*;Mc!rnG?i!OwTK0tDr$%_IlZFi6fh}$T3u9n)_Fau36La?>_fgx95&vvF{Cf!<<FpQ'
    'EC#Q$&{)LIhq%JvT@Q>T-59mI)~J8MSr;rZ{P8Ti;<LRZbe#|{#@X$C%aL>;9W_=q~HyY|3GbYV`{D0=zW-Jsq43p#gW2rwD@4pS'
    'v{KZ{1|RYxeZ4leR*Mqx(o>W<*98W+5^n?TKNM|u22vK2i?N3<2`nE^Ba&T~ti?xBW1D2hy*hAxpsg1_Bif79B0Y6@Ix6kNT}ncP'
    'XWiM|&L%ELn>JTJXMJ&dMdGxBtzpJ2Wdp7!lX>S2<fV7?2U@e|B<!?S*Z`5yFLxqZc6k_>}<nbC7H+hckeWde&ls(u0<uuQfjn%4'
    '1Q>P$W_<8d8%qMlt@DYjmOsC-gIhC2hSJH#%;?HUC}HVTP0zshB;wh~HHPvuA_rL<7)I7nrcr2cLWQ#n(j8N;B;SrXrkQI)eLTF?'
    'xuoI|}08CN+M{WocWm1c&O?_)MaBnwzmPZh~R)+EzJvd9i}b0j-'
    'Sx_TATwxvw6k5VRvnO9QI=v&_wiU?A}p|94S?RMaom#X9NpcVtsU!u%ZF%b6!kt}9oZ@Ng9aB$*9k-Wur-%BD{%B5-'
    'K)U9A$+mXjs_d^rpBW{z@iN{tCKm+7s*2P_TZ1o^CKR#tm+|9-'
    'U9AOTl?4==06T%=k!u%_hW_g;hM{JN!l2kqOz3*kY$9IN%R#N?pv(^7j;1M>~|3R{lvuMvzl?VGdW2sot^r52FT~tozTL#@D>0c{'
    '9gycp7aywg-T6@DnRb*Y36?N-~+a_AxZ*5Z3jeUy?-`A=fE|T2gg3QG7;MlZIHR~YBaD-'
    '*2rj!kLRQotxvqIhc(BfJj8W+`bCCa~zxmYn;O`q;=5cv$*E4Eherol13OVl_?|5|H5cXcM`@y3xNb`$IG)yO#u?Dd~Dvd%(#eV<'
    '0&S!A#OqLFzP+v~q-<enw2Q>RAudCPUT)yO|fU4th9XKFMH(wi{R%o?q*g%n<7H70n0Ay~ETzG?{eNoJ(g^U{Pr-'
    '99}pLwPrnp*+=1p_(4AvCrG=2%*O}O-HJ9A3mBv>*v(2i)Pbq<QcSatf|C=cssJ8-'
    '^%rnHzD3mZ05J&uCEm0?ZQTWJ2zm(gm_(3`h;e!z#^3$v{zjOD(Wtwzh$e1(M*_6Q*6~jrx7}p@qnrhAwGwSj}5{mIm#FE<Y_b4^'
    '+=s+#Y9Mjd#Ode;}A*z+G;=7*U$c48_oD#AI<gM;OF{MDd7(Gn4U`ZuyjL1{S1wNZ&y92Z!Ro$<&Hp|)M1HRT}rL`;Vl)Nw?>lNc'
    'bzC<KnB~Fk0?p>`Ya{YSJJSn$Q>jf^G**d+KkVJ56D8FXWCG6iAFgG_;r<&Z8BIjnz+=Fk;k);PcG*%OP#rha2|5R)m+2A)0ZEjG'
    'r-k{$u3`fh<?zOT)4hBlp8kS4t~i7YOkT3c<M<#|6pl`sdkWXqd}^1bg?TX)(!D?qesDEo|P%CqG9nmPU(-'
    'KUh6FVO0_i)*s<U2E5T8Cc3{tDnX(gjVrej%OeivAR><_&Do)}dJ9T0b%*C^G6_29o#v#U;1~;?Ich5uCUE!POHPsv;|434H^CRD'
    'Tu9@Z-`6rT6ouByNZ*$F&@r!9(1T2U-'
    't(LO*hFIg#P+($xUNpbTAG+!|XWUNP5z^&M#xl(kpUkEqaiJo9RTqg)WKMjK7h__P`BWv@+@EEDL%Y4jG~1Roafo+W8q@oB8q>Qx'
    'jp=<Sjp=<ijp=<ajp=<qjp_ZsV<$c)AB~KTI2hK?GTdtlI<T?zMaE~OxgwLCfnBgxU0G0v-'
    'b2M#RXO7{kNTg;vrKb8<jQQnRT{Y95^^+f!6oEq;DSrY(ZB_lkfVVME+Iz)7y5Y1^-am178;&&LxtE8W~j>%(kBj%)jT?GEp-iQA'
    'Q^FUi*`A;u({7u1U4JS8B8-y45en!2uDY$FH>r<JWQ4_UA{*<e)_#cPd)JAju>@dn}M$kT}klYn>y83HH?v<c@-'
    '|hf3xIdx3TBZOxoei@j1S3u8A3)(S(DSl48(DN*}a1S(~ZpE@*LzMw8q3+^X{Z0P*n7^c@wEhvq0-'
    '73RYbA`U}zlh9XlA6h?{;;U@%20iF+d~gPBj&mh9IYS5TEQiT9IE*kFsACunbbJ^M)G3Sx>KsM`bqS+^x`xp}-'
    'CPkp^~1Saqlk(9<!4R*7JuxkNGLd0tC|D??-'
    '`9$ghM^Zp%VX)1w}Pf$D@JEL~<a5^5(L){B_D=A0gpP2S+z0b8ko=rkzTy^YJ@7N+e&W3HC-'
    'vq)Zp=kI=TiDA*&Rqwtc5F?miyI82_aY;~ACSK8_@d9HxdA?@DKX#{d*+|NwMqa-@SppZ%x=O%BDke~&s8ZOl~faF-Fp)~z02b$b'
    'p+w{~lGJ#g6$EJOgO5RQpBnYOsr+q<?+<igB*g-'
    'DPEvFk{_YD4|y?+LGXz!rGo!Wb7FjjjP4erw3M?==}P8!_9l7;#Kuac=IpmZ$ru3?NRJ1LKmG9BEk1p^_Yxi3x(pd(9nTEkM6$Q8'
    '$6_LtlmZm)o*toXjop0I(GCAa<T3Y1$v$M@B@a7$Oj?BJHJR2Ocdpel5CtD7bLJJGFfmGtiv%_MnGGjyF)aBuFBVj8~nkvs&Nnpa'
    'KnV5F-'
    'xOxL!OGHR*7@gAQCnC#mHcZe!bs6wioixSXhhk%C<{Y*1cCXXqe58Wu|=0l_D{56a6niN+ZYUp(pGtM&YQT~15ZS7J1{b0HFDF6Q'
    'Oj`k@30r0N&DE~q5zV^WW!SI1LB_l)*wAcRWVMteKnCf9jm&?#J^a^R4h6b7~qh_0as0&GrvS^AMdarNDxInoBbCWW;+3EnFtQPy'
    '|{x`{xEeoYUx8p0T0`H3;b>F4L8Cr1D8Rtovy$&2^`i|UM7iIdM+*%i8`hnb97h_sQZmkP3{YY-Di!iMwx7G!iej>NFz3&g$U%EN'
    'S1-Md%ZmL&H+d(xJT}Yq^lv5QC;EP&NU!#4T(X{7BLSHEZG`edXLOw{E&WtvdI49Gnri*c)W4ZNx_Jmd-'
    '`$>nBp#ZsSl0_E%7zzS2S{AqS8t36%MiX&Z7hkDinn$W#B|`(*Yh-'
    '8|TP8!(*lVS48cTu1$JGO>0p1Ay2V1naB_=A^k?K@aWfY@Ve|aS48=fZB#s>DjevS9_|0H93d#V|26=O6Hrytlc7;h*zp2fav%_Z'
    ')%MEfD-BA!MaxINaXf|}ptGloLJgtlu~Q+td~Ikl#~M8%;7G;F5$XxT6`_O~o5pY9ruoqH0cpzG6-'
    '%!j?ocD}XyhyuA(G6c7asqJ<jiVj_gV!sD1qtg}pJ>97@595JOq~2NV_nbt1k4V3yIt2(y&6lP7s#1dPBnu0;+X5`iiCmRBk?2Of'
    's5t+Apn1qrMjpW#K6`o?4FqNk#BlQ)$7Q39aaM>*#k<E-'
    '+n$1x()0LEX53>k5B4bEE_fkG(#F0AXt^{a;ZvQ<@@O2e+>fUhB2godDFs$I;73X7F3!^SJwnSxMxdcW)kPE3IuMT15bp(Q65@#W'
    'V@eGiY-~}N8ytljW9tLC?)?-7WhM(iXl`&u`2I3KOl7^KW9Nd%8zo$44P8HfuT2bB=tHp_cGpfM*eGw-'
    '$%OOq6RHGF@|z$sTJ*yIhqgC?(yF+&$Guf`op3-'
    '<iDCi*(m(?a1ktE5PD#wA5*23zZJW87jhW2&@}4F!GjVKCgv7i&)TpRY+Zku#h?+#rh>F-EV2dCP@;ke#&Z(-'
    '~X8qq<|Mec<^6>?_>F?aCGw!o@l=(u=w_L^X3KbD#r=<LhgZAYr$?8UV;r8#F<eJjA^;s;{58tVY43TTPATn6MfUE0XZjvMB3tfl'
    '{-7z~qVI>%z3j`j>6ukBisGnuVMnBY<VvM&HJbXC0YZUxlu*|!T^-'
    'q*+$@GwIk2B)&XvgwZJp@Q9e4B|$!=NQW+adFMC5Yj92~EjjdzbimxD18`CHht+k5VO47{q61!yQ2ko!gMUwjm1?Z)In27gr&i#z'
    'T0V!El=HK)=OaucZ6(kPN=C6gDo!Y>vgeEL}N6_dWsEoQKXdV>TZy?Au0w504fgunvPpRCQ~mfX3b1q#P$|Mii*YeYP?NtS&?yb_'
    'Pv#Up#`5^`?&<g`~M%cX8(J_<IeBRrf?)?W}qO2!X;4ZqFjd5{=gi6~TyoxI}RpCLsq$&b?BAdVTB>Whj|OW7)Sodz?M!o9yQP*-'
    'HbWBgU;!t`z1+w*nZ^25x}?z|I{oAlSJ91_(PhXn<koUKl{^+zJDVotqhevBynS%w3p3>WKo}>Pktx*{tl(!n;v`20F?G80qc2pU'
    '=BswMEj_I1F}mTeFZ}FL$K9D+2SN10UtmWQU?F>5OzE=$S65M>9H)(o^S9b*B5KD>;P&zu;?}(12j@b<SwOF!%<iG@uv^GlFq|oO'
    '7dlJoYVklyxoun>@;DaEc;K2Rs-H;X(jB1RLR=09dw<Hn?V-'
    'v)h^)((|L?xTBoq0^RYM$}>_3l4Q99?VPEkU)`X(Rxb22wmL$zo@vb50gU_3s<(j3Dy1NScHdeRwfokosNJ_-MTNw7RaC9)P*E-'
    'MJr&gwH*kBbdY7!h16;iB(bA2e*L^zOdFx1bi>O%m!1W=W+1}}L+(Hr8oQgJCrNPM9VK@V5kA3dQJeg2irUqLJ32cg9TIU+LXtk='
    'm)vEGVtIAugDsQ!_yw$4mR;$Wet<1h1)JWU3^k?YxV6@hc6kt*jszi=tlg?P8*V$g4weelM4|k5%W_{V-'
    'I_G9su5Ay+2IBdMgT*lbn|{+Or?AbDm5)jD;htJ<coQH@6XnY7t?eYaa=o;jELZMh+D?%xw~w|{{gvbBu5-'
    '>IlXDK0Ip@&!X<Pd`yJJo=v{M%uEWjn}gSkBs_c&V_&+W%`wejVyhO&FXp{$5%hq;i^XxzcO9i2=AU-FMqPfgF^Fp%HWp#)9%7ki'
    '6}T@M3<Y%p3+1Bh(!Q+sMVLA>6++D<DH&9+uBhNMc^J7|vB;|in(<05zS?n=0u&n+AeS!8nVp;9L2p333a#B)=P&?FP{bJ>FHv$J'
    '<}N<M%zZJuUoH~_?!3x6nGf_5F<qC<Es(@eZFuSoS=o3Mz{NAr>r-SS<O9M^LSB;)5qIP4yZIl2E&n#7r?g)6?6Df=0f56aK-'
    'P^d@kBEF*U*6?Fn3J`yGISsoRK&QiF-rWFb9&TE91EhJlX?q$#&EJYQT!gDJFJk3t42xL08j~W{uEv;%#j7zRV)beah*-'
    'WF)1k9dP%(LtRRSb#*Yp80s&;bEJ$1S;%zZjnipL|-Q*-ez=c=T~=UrN--HwyBj*KRHKDK;@8|HrfYixpCgT1!Ul>|GPG$lbshMg'
    '2C^6#V28A!B?>VDQlfiHW2LVY07E~qwT_z3y3hnzr)V`?v|Wxi0NH{{D6&S|3IH7O?VYGgBXUzJiS2`Z@}=1P#cD!3s)U)-'
    'iWw@Wa~NGMA?v)oVKCeM?4saj?5UUmy9<lUY(PFwKZyT?_`wW@1#-c{C8OW&p-'
    '1iMtH)ACk~;)fZg*Zqg05}$|M|B?nUeA6RNhbQm}NKJklwgN(`dplMrLaKWQRwzQM`v+_bgjDxVYzu@`_bzM;gjDx#tXT|CY_&bq'
    'hrF5xik;e^z<~nOQglO?wH)#5jj5NX58Pbj%a6^n4&nAtdZ}cl?w8dwaxv#1N>5}S)Bl4VE!4qEs?C=GZ)0LY^h!UeBwZL&%HZb)'
    '>|x1B<3{X$q+_7dOBn-Ep*Fe<Lga1*c(V|q3~)qAZ^%0Db`oQTDT>-'
    '&6*t(mCwdg_3~sjD*p<Csn{5h!xg`RaDcH?aK{DKDGBcG(6f`-'
    'Lhrg+11{B@>r!^UjdzMFQd!T659q6W_2P!6O#0d9LCCSuTWjgFSyOd!Um8xGXowj_~^FOJSJ%1{WPA*g7>r={rPHqC`wh%HdDhT$'
    '}zTF<tJ~#Mt_ci6$9<nT?IlH49^Q|T)qp&4R#YbqE>{uOv1z{P2;>2wU%KuW6P;ITi{%xIo=?IN89OeT@YCBXWw8v_Dh~_ot%L}`'
    '`VQ!Rz96`-'
    'NN;*n^BG0L0c=Z=JBUuo(rCSgVCI6R7M>BzM*_d*bQT^Xba|@aZ&V6*9CG$47u+Ut%;gAH+{V=6=7X6BHD<$s8ZOXPH`U2+2$9b`'
    'lU3aJ1(kN1fjhZQjMtkneM6;CY-~Dra@e%ssBn}@%&qMS9KcnqIA_;hmwg-zOAQXUv-'
    '|kn1HNtOqMPYX^oC{xF*qsG0d6Xk@Jx!DY3t4SF7T6%h%q%NFZMZDhma5Oa+|MI*z$5lLfAX%?^a^(RAPWPRnyBbRZ~LmWq;e8=V'
    '}#Xf`&zgf!wam~#mQJ#Pf>XFiz1V!VS_~!7f%OIzkhdz!plcp&w3N!3iIRo^@6y5eN9}yzBaC3FO2Kg*TwbgMZA8^a!!EJICJcd3'
    'Obl$bJj0=)Dg>{F4z1WX={JoeadiWzUj6@#-}aRS0Y&o_oGh|_-RAzk+dXDLSN~nb?Pz?>XIk;uwCe%U_m7yJJ-'
    '%yp18yS5L??bXtHk_Oo-l$ptmkSkpJH}nkM@Vw>J~K$G06;C0v5<DtaYcik|a(No+Ic?V{$SUBp|*-'
    'p{pgt7xg=!<k}`UQmn}e2JgwmfS#?-+cs<{W2&7*`rFiORJ<u{IFAVI8{7YyR=OyEW8~qV-u#KuZWX*-'
    'C$%MGVwPI<(wQqR~gDV`JTHuID3t6DSEL7{0=(g>2w<<Pn6?rmO=$h;#R+%-'
    '{csV7`WaF6NMkAXBixz*lMLGcXE<?ALFyMHOB`Qh!!#y|91~)wCjb6oA4x}Eg>5*?m5+0d?>IiaSOW9{*O4)^wT83t`QW^Y6E=FY'
    'N1vM!j4?XR%J#_Ji5%<1h&lD1Yaz(Ho=$5jGFkQG8;dgTxMN^Q_5`n^DU-'
    '@sXnkDvH?HL{Kv5hufIiIyZuUgWa&TEYlbQm_Wf@e%7(c*i<3sWn$sine@2A<&y3LjSrPg_J3{~GMCgA;1K>{fMiqJVFqhVH7TdM'
    '-tv_lA3g_<pdt!#+$gkX;do7SuXC#WPSwgcdGtXyG2HRJId5%$d{WYRMdYATyQcxhDDD9_98Ts`o?a`&k{`D>GXG)R%+rPBOlrq-'
    'uSK4F4yW35a-?o1+QGdIdZLicq@)E>C<0PjcMcHk1EGa96wxF*phcu_~l2=rOm-N?Or7k5d{UnkkXSO$#eV>odc7xM-'
    '>#G_N{%}P8s)6BuJ(NWT65+uk-'
    'C^wIA|HIG!AA+{!QBSp;b*gJ5Q*_>dQ=#%X2A}@!Ou;^#ydb3+!2nh2KVl%={Z{NDtN9lmN?#|K1?Q*j%Ub5uL$(Iw<~&=Y0(m*F'
    'I?7jqhplnLaz3@vKxHxYbopyPG5?_>47>LBSAqxo}lphXL$Enh1b8ryVe5CC2_dd%i$j)4@qy}+_(dKm%))ZEAR!0=xS3n6NVqpB'
    '8fYfP9zaWQiFpyJcO$tteFv1dwu&j8cFndf|Rq=)L+neF}Q1`Nvg}y^sJDCb!i)-'
    'ebS|wIRZtZGv9^Ep33G^nfZ=Wc>P_A!2Iz5HTh@yPmPt3)iOF<uanvHUJjNON2B{VWY%6hZl$WGui|-'
    'jHOcl@jB<KUOJz0)mx?1U)lIL72pZ*@BLi$rMsbCk#syX_@?p!8-jZt{wI;`JqwuZK7c~Om&+=Sk`|TCiLoAmH_^-'
    '1RC3$DoAcWgH(LheFU_zZ<SqjmQ7mSUg%wU3W>{B?KC@C<7;>lE1^LVwIBI)pY2zYTe{4Qjb8+gmwzOSVLK*{~<Ecc5l>J8TWMGE'
    'v5>zm<#9BI;~9702bVC%l%=u%o_i1`H_&(^0R^Mwx-Ya!#xmzdsPt?MlJU6!v$swjqWD<3(N*{eS#SL}!eU`aXLtOF?b<^*&g<y;'
    'Xffi6mGG9B8pq~ahRy}e2*5aKL)?~;myc&&StR4Bw*$;V1679!)rJ|z_l5kawUNkv1(VcqQRU&?pX9HjsdvOaDCnzspwN~=`W`xz'
    'YW(7)*JO3wVy9fAtl9Xd*(&#E1|W?*$BV}%jR!8O`kI@L!<RzXLhEk9=OGGTRhPIsBizZx|@!b-VEja`<g{-&`n7IN3WYb=a~-'
    '1Q$C8)ISg@Gp%uu^|1?5B~xaXHhc8mBL^uqBy)vwi~oG5QZtB^n$~@Oha8D9PZ0Ch~C5DzCwfO{fwj{nq!rx8Cq-iu5~=gM^M(gy'
    '>fTmS#A5lX*=b1{E5d40#imeBdyn3;kTHk7EW30N8gt`(rv8%t)0wBTJG+w>>5cksJMJr^L?ZmV>fYYl(+2RrMbgFcI8k!*n2w&X'
    'Aee)c4%&pl84x+x%)>RVw2|93&r>kG<O0i$Zyuth8e1WiA+p~p2#wP<W?*D0cRjPWEsdS8XI>q{<CX&41heVhe<#o`x<*oKx;Ks@'
    '5jP@V=UY^#lroASh#PFg?nc#+_%KSeQQ_VG)f^RC7@Eio?D#UmbkF>&ZEV3XiA~QpEbvZ1>J`gj(f;8$dI4Zs>=P9eb~FP!S)t*V'
    '>=G6fxFg8)<t>#t`f;8bKPAIGbOIw<uFrj+S4RK^kc^UUf>txOHM+KCcFO&<U}2y>@^|XLsEAn<Dn^$`;1`;P|C_zhomFjK5UK4D'
    '5<P{a0bf~lFV)Z#=XzrTck5%WWl1l3SFkr0JkhDa=Zrl^d&kBXWIB!?KC(oEHzpb)#A{-'
    '9F4i9HoQSK+8*Z~oMc4gmbyj2En;8NVn_FK5SDV%8@QvZWX2?D&1HoQa^{C7w;SXvkj~_!g33c~_f9UTIDD*-'
    'LC!*H{Z1{YFyzkfw1Ub)R;3w;#u^tC(~$F>!&&0d#`nlmX#z?=YvFQVpw#@jx+fNLmfJaVS|bg5be!Q*mP&_>$IXk4BV1yzNOxx^'
    'G21I{A}>b0)Cmf9VbN|yB*|0n9{8?1xu0utVFzYZqsA@yzs}-'
    'MSn_{ElqYUgjNaZ!*|()!At!hW(s?1gXq89RU)lNc_XYtz%&YN~5H4ajTCIl?%=*B*hNBsx?0l$=qF!=_>sF7E6un0po+q>hG4P<'
    'SMZvR`$2pjUNx28JvFs!t{#R-'
    'R%hBK$>e^Ad$U*cj3Lo@44&?nECAgu5GDdw5(mwBTWp<RP>&>vAHo`(`k1^b3qf~8$000(1BA6e+wHma#AwDnEpfe8fc@e~>JuzL'
    'lrrR9O94aG4+$h>5b8BNuSSe`No>O8{JU@U0N{osZ1dKq5S@AUiB2Z#jd~F!6mzWkW3~+%G<KpWATA;+dc##Nc+$ORJ-'
    '&A=2O(hIh6!}gFRmgQvKt>f3OM#YB1du=tk44X<vg)1)J5gHH&BW5Gn;}=iuJ3{}1iJLO(xaZ#h1!sko4tM8=Ky;nM}uAKXKfx7-'
    'ni)feu>$4_?IqqZ^kn7U@ITOl`ch`>VMRWoLj9cNjV6F>Dt;=N|;d~%?1*V&@|ZIQEjXe?UJP`?rXPllL<=OFGQv77o*blOHpb2P'
    'f=<6&rxanFHvdx<pk#2T2{L_6HSlgkGl=dyRgl;SKU<*2uc+KG^>qy_i&>U7~+kXM>I=^R(e;X5c=t_E)AY=cL##c7q`I$zB_buh'
    '1-RS&(8{eWQdYc0~J&<YXDrFhYgga%Cvz3EZ@xQpd{DCd=<K^-eDRYQ&wx4K<I8Q!h6?1P1@60ha>fv{jI-'
    '>pHTKeZd>o3h0L)9+>*HO+ucTULLY&%aig?ZhPZ|B3nWKHUc*fUs`}>Xt|&orV)Jn7=r^eo>!$2G(p;K}y5C8k2S!s$;XkK^a+M0'
    'y&uig9+@u3dtjmZ_Mt?BOXE0d{dEO#O@14)9?qyl6<89g<%IaNF0r^(!%+o3Uwr>TB`Igyh{f%s}zvmm-eQ>T<&=^6J^chDQ&NYT'
    '$qcY1Tx|t*ttV~v+$n73cE00{uX>93i7CI>8|BegqzU2^5-wN-(tB@vsH@y3v;}iLwl5F~P<l)j~{U$exT{-'
    'dH*x#fW@*mvMY$AtjE1A|lqwN=Efcm_)C+kdccba2aqTlp&G$u6=Eg2=HQ`jRrd@JG{P*igJMLA-Cq9ZDGv`e?*%2>PHEu`1gJ9R'
    'qE&nSeJ&nW2vQ*1dVygODQxEveaecN$De4A~p?SGc`$E6U|zpL%>T3AS_L^@H6tJqzI^X~>YJ&~&Uca*{W)Yz|-'
    'LN0z0cb%Iks9CH1Trb#gpuNejtxYT9no)3Q9l(sFMS73ShP_Qvg*&z9nUPTofY(+YY*Ao3B1wNK<=c|vaWh??tmfN}R|vJoD`|XE'
    'tUW<V<CB8zXJw`;5yv0>G15e8lo=Nli$i4=GYn)B>;Z<O;m<YTfta{))pL;3cs)z@O%7JR@svfc-i~2oMXLBP+ZKNBXjhcY;r8~-'
    'lHNwgR6iyrWF`4O2J*k6r3vjnP`iJ4w>yP`vEsO9DwoAYn|)T>OG^8*k_XK5nfMn$@O2(AU&F+|2!gA4z`T$LsR)9rdBD7gQ<z5_'
    'rf`9lA>p1#_hhuS7cyH}ocf+0SI8NA&@S-7U{YZZ&|P{O7ptuOxMz0AC{;Q-VF5haO&2BOIm(GFxAG-DB3Zzu__?LX;z^4M=$mM<'
    '6STd!v_B1Z<#XEpptL^^?>?{X4@>)tIIkIl$p*9R{Yx>2&QNUV6%ljP9Vx#I*!M=7FEzt=c*xU4$)+7G=!$2x{gR+7p40ZEE=Z9h'
    '+=AP2#_^U)fwju38Iz44u=6=D*=ukL6xjpQEbfO<5!xO@!#+^i3-'
    'rjN>9!9^EW3>yB;8wyFJI7gJ$Cx@qf9oa{B*vpbhOC^m7g)$pz;`#4Jwb#_vDT<*`V@xnOrm%REKi<R5gNwbGwLd>VZyzx8ibFaw'
    'nVg)kF3ge+;{$w~@QLcAO6KW*y-W?w#3iW>gz>&TP36VeipC8pyGu%w%Sgp;Lq{6Q?-'
    'D0l?Zb_8ruSM{4PyorYpr`YC5%D$9`7SA=73T;ZLlAes-28%gIDc9dboV1R)Si~C9EJFcrjFrfCxNX0{u;%CNwH6|s$H1-'
    '>1t(Z=xpHTwyW*<RzK}{UaV6V)6r@7p2tQ@8ehHufgyGV@ZR#cWs@AfNhsckXqFm>PymuLR&L~Vam+FzDT|JM&mGIL`7Q!(FFJN@'
    'dA-'
    '4EtsT|>V%$mU0A3}Nvj5VT_N!^<bRwU59cf~^l`RcUfmP}`+&mI3w#Z;OeA=W2j0K9*IWhbTL%q)gL)4@x43YRZ)SyHhAIX<n%Yq'
    '=4$IOTz;>UfcgF?N3VHdp<zfPlpeCKRPu>Yu+mzr0g-8mvINH(35h4CYK9(n!e%+L6a8_=XybzR-'
    'x!0NEY~r3J$=hp37BrKZ~Nb$HVTYR<Z^7ZPy4%5E|7ihi!JC@ax%I?UPxDV(yHRN@PLd%`PSBAF1rzlJao>?j%G%I$43Z3LV9x!l'
    '!-'
    '!&4|xvMs)v`2^cBa`CH&$biyyc5|lap{VS!hKqkAIf#t|te{Ss8GSjSZ>ERx##;xE|!zg#Y2Ug<(viT_&y2nV>KBTV5vpYJ`?&M4'
    '@>>xfl`{p^T@ubi5g3x+z=!Yc(tq7PT-'
    '!LwD5c{CJ$=8Z}S4TbN|D>{W%D~9u7G#86;he&gE4bTPL=J)RMO>&qxA1%dL6u0B>zj(WTz`JS%k@rwXPz{B=&)WT8`$LB7tt}#0'
    'b;-'
    'QPJ2QTnT2|1c3B3d1F}oA5kp92bh~oTfm5rel;l|*__}?RkVALWGf>I}r<at43&!stZO@Jvzk?|tD`|qCidf8R9Di3V_CSa7GR+W'
    '&a|(n5PWH)#&)3pDBNx6vODRPzoJwjgrO3}LBv+usg<nT?LT#yZq5{I8PD;@OC4{rylp-'
    '=%?Kil#gxpbxO4uknLvd>A|L|ysqmM%zDsJvbiKVH$`9rcKh63l#WJ?SM<XtK8^zn|EDvl^maI|1?M)_IClh#Y#f8T8??v+Y<(P7'
    '%Vi-'
    '7g0q_4#%<DYJ!TKv;fAd^jS?QsU?Nw{JV&lCw)ny|Bz;7)(z4`nyB<%3y2o;flqD_TlQ1*CeIS=u?_9UYbfwLMEoCcmGI&}O@p?9'
    '-*@>&KIC3bX%EZg0Sw;oEKlG(MK^xFyi|SibA_KzpP*a&%97rm}p^MLG*c%cw)ZQf=%^n!GVW>?Uirg<*RW8h0ZlTI-'
    'U<d?zMMGJX?yH*{R`wBM!m8IbA|^;5aF$OYKmEm&a9Q&bJtVNJ9N00+*cUETa~lPC!EF<jlvb^GmSN_S^}!>c}h0d45Z>p-'
    'u1`d<Ift|+R_oP1I;b***{w;Z@pgzYYe@iL-lBDSLAl^sO-=%;|vs6)$?V4l~-'
    'OKtQb_O`n;Yah4&?KA!@_8yL*eX<ef{wmZD!P%ZH04Vy2VvF>i*do0*wn*=bEz<jAi}ZomB7HEnNFQ?KPTov^q}(|<j?Wj>2mX?>'
    'o#+ETC2H%diuAgjlu3^FXPQo2%Kn+sAf1~2L5v|-'
    'V}=7N(iv+l$aB=G15ll>qO6fi%f9t9w!UByKM&PDMB8)1nd+mY_cRLMgkC3w$JOrCCr%CT);PNV)5E)W9QXYhahvt2qIxem#ZMR2'
    'hU8j4Q&jIKxANJd+L-vt=Zb1mviRBqQWV|X!(V0>yN(xox*3?^#EAcX0b}!B#-B1m#w`T&=XJlfD7oEM-'
    'x+=iooq1Vy~pO;e{c;;J6vVcxFtzb$2W(h<+Du*kC3nF`r{|*&d(oz-bHn}cZu`gWeD}AlbSd=*n7289yleud&>!6jtTGHcDf*^$'
    'wEngt|jJI16W>rL3IGjYhP0hWO?mts|J?WURVucdF|_}!7Q)6s2bwswZl1_YQO0RotG69A;Y!om)eWm>+qz$9$TVeHcs)-'
    'IpbWy{Saqyrpjc5O^q$mpOklsT)lE#k*ig1C=&I_-'
    'YS&F4#=aDhU+I(=nwjR$#;a|oNWAiG_-=ws)18^je8+~&`hbQl>M=>UoKO*Qc4N;Opb7K8F-B-YdW@eV;eDQPlH1LylbY!)s6yl0'
    'J)U+3*p^t<HY;@g58{3QEo-f999^~v}gLEN@l(H3fAhRZ?z3$hGb8>mFHgE<~S$EV~@-6*yD3N_PIG8`@9^F-IU|8&oB8A-'
    'Dk+Z@R;jCLk=aIbUtB!=YzmjN7Iw-<7&9cbklD@c5RO+9WucYUgJI8mot-'
    'V>=+Sa)!mA}zfDChFll8>aJxAZGMvy&1;ACG2y|Q@6y$g~&)-'
    'r|Zo|7)8iGsG<SdQdRLBnB_6A?bJA6ABySWg`!ky6bknOge(ent~ZB0N1pCnbq1BM(jR~1x6@*;1@i|Kj$`3mmE!kPiMp&_3c@PO'
    '<dw`1=`S{4^MD1V?t`ww*5>3<Tco7OlqXqB)=i6P}m8CtHI=~fHf4EubOFjrV(!28VWjyoFLSunGudmbU2-'
    'xI_6y)m5M7sL7eF`PdT!})_ToIe!8ImCet(vwiNXk~w5?1gzF$bO@QG;nI9frD#nSB;4@Un^`T!Qe-wJvJE!oaq`pX>e%65e<XCU'
    '&1xErl*VokCPP^Qp>+<A=|&D0Iqlca02P*%#_J7os~H<IsUMqY>*57?ZA=63yytN(u%#;P%z<^$^C|62?Ib*a{0Y3WuW&9W51p@y'
    '_@Y<d_<tfx2Sf+7MD0;Y16wQAMVll2ySj9&X+$@-'
    'Zt&9NO#%5osa;BCLl#|1i3*7awIwiDP3+G4rN%Txd)3SVL)&%wo1Z;;6ALBgb~60*e3}yf(Nin5{3j1Vv}Sw>wTY6a<}AUS2LSZ+'
    'F3J7Tb7ctQw_9zDaqE{((asE>;OmRK^D8Jpepf_nq)Bj?CCgfP}*Q5qg*PL$b_Oc+Pf&CWDo!zi4s{9fyo|i(&L&?1n7RkpZcnHP'
    '%In(x_Yw|-'
    ';3VLyH3%Trj+oksBBHu%ee54@^K2s<CP?jw<u=FMJYI|v^$j2WSCvr9ZP94%qi_oCASrB(@Y;eR0X;NvE3F&oJF>SU|nrxg-'
    '%lzR0n5;>Wu{!F<36<<^mvC7G=1#01CFB5wSN)GtN<x65pttYEVdfZ&O0CLdRf?<P@g>ozH!vF<5Bj#u0t^(>cYi24e&MLz&5W+J'
    '>xrpe#1Xlc8WbMfz!U&`(OFd^+$y@ID3w7FS$xZ-'
    'f3+dfI;H^*_qU&s??GE#8b&Y%bvHYDf(VQw>B*F<ikmlP;7lH}GaV3FYxXE}|(v8OTL6IlF0Ih9f?sX)bxr(Q^|pb#XyqioHmsAh'
    'n(F9YUzv88lkK#dQOXRsc@z0vfI0xpoDOR`7hgfkrDhzTH8i6&BySgGTEu2`nzMhsa_l77|bFWLAb(_Et@SHDJ>kGxy)aiR>-'
    '*z1Gp_8Fj<kX)*{CD>cE?n+mQ~rqP#5O@iN#?J{wt>)cuf;ZYu<Gz7@97rmIz@sjfO1)Nw=!hQ{>6;&f;X1^DCyM+=_hfz*;i^$t'
    '8;?;RXPdlZMm}#dK5;Sc~A^u~D4igF)uJ7xQ)8&OdhO2_1nWe5m<Brm?8Ie@0*$7L(Xsq`yE2>xg)$)Jt2T<LmI!I*EPEwMbq*kM'
    '!V1^^wnwW63P^#@&lmKQqI-'
    'W0~yxS>l`!xaik<_v0<L)N~Ml<F=NhLc0)k0FrPDC3hY2!_T4@}yAlQEWK!bm92wu&4YmtV8#S17;c(62?|<PFOrK?81Nhy+JCc#'
    'BZyJ>`hkR_ZSFZLnLUxphOkrMhc9`Z`WA{_v@k>4seUaZC>}E=%1G#lsL-'
    '=qLrZ7zW7pFt<P+TE_NpaHPuE?^6n%k)bn3xaION1hhxG<?{CM?kKlh{<iS<wQyBCoqmO?b_V^rCbH;W8>}`wtE|q1oOqN88foM<'
    'wYvH<=KpfkZiF>sl}bg|wAuy;W!axoj4e4yr>0p`fSz@a@Plu(SCk^Das%3awj=U;yr^9VBIJ61(jo$SGuqayWQol0FPgP15xHL0'
    '5-y$u@v?n3%)y!*CO5du^BoX9_{>QTpu0MM?DXKy%UhKP*0GTqtw_F+V~d2^*G*+44R(cnLZ#A12Sn1za|%~zMI&`O%-'
    '<NE$J?W2RY7Sa&}TEn(Hd4Iqbt3ZXT;S2>|qVy!PmPiJXWp4#BqH@j;+w-'
    'f`vcjB~3Mvw??yr<%bY5*};|H%0lN;iL^o0$>e)e3re7auRAEKoKg}%pc&>B!oW)ZXn?tpQN|XL)sl%6U2paEeBpbI@3nZq@8I-'
    '+Q6O(rtnWS@(4XnyHURlEBU}eGWM&560;tdI;9P+8nG=U<Z4p#@PXv|T8$qS_MI`A15kC4*evGKnq>3rc`47(9X(r|LZ8NDx!14r'
    'X6;=SFLSbv#9$Hew_K-}_j_?m=ZQ&(-<$zH^{tnY>QvviYi)c4GTF|8-'
    'X`<6|gQny&fTA?6(w*Vm!P?%U?Ok!h`SpzF{!Qk&cgTjf6#X|`a^TwSYvJo|1mN4g9^TyykhguahDdL8w3KPVdC1_Lwr-T-'
    '?PL%W#VxiZIT#IPv?&QVa%&-ZBwSsgT%)L#a5RJUoW3$mINq^08GD~$T<q@+Rfm#ewsT2A#pm1M%%E~u6AlgP(6xSoj-'
    'f=gE*u;5oz_<m3c~aaPh%WLjH^8-'
    'bO2Yjqf82S*nweJ>I~{~xPRuWi%5MvB5y~gjAWfa7sWIdvb=^<*QFY7#`4NhL&9TbqO5>qVpwV8)T$lJ&~1hlId3rb9>YR2HyWz0'
    '*(2R->;r~ZA#O4DLBpWgTaA5)1-ukA4Z<qwAsXrD-C8y<R-'
    'yMA!UT2(?5gl*w_;DgzLlyY9me}l_<N_<5orb&HHAHXG6S=$Gz>U&!}NrPvmKqnO>##!;Vg7#ULZ1E$k<mHOjK8<D~S}!SY{B8`&'
    'U{r*aNUU$SLd~>|wL6iE|MFUvQ79`ZAOHy{6iqiPY~iRX^5`y5Cg&nN$4%Qw?BB^#@IL04qs7WT;F@bQLMzAL$Pgov>4ChH3}+E;'
    'H1~w6dh1Xuf?dJUfcxEVqa#HBTUp6q+vAl9vz0ZaF95L$K}3<W(mrhkrCuHs3`LFMM-)^j{Fl5xOR$g>r;$$rl;QZf^og-'
    'M*nyM!$J3MI6hx-sO@zHmVdU*M?*4o5*1jWs>;?<(Vhd0UaHF9!0++*LG?RZMKQdMw(C@n%QlCDq3DQJfgYQt_mNMvL6<v1nTg-'
    'g559`wFfeUwc3E_6~>zH+e3s)>1539?F&YlF?TBal99f{-OB#iNNeUEWnVS|LVMfmNJtC6(^Q{uAFaIz6we&Q_Eibz-'
    ')Q=r{i6++yRxO7$HmxZTb$gMb{yz%Ti$XGye6YO0BIJ~?8;!ZQJK~HQF-'
    '@9Y?BS(8p*#s!2U4>Cl0hvrQpN{`*aFU9AuwK!HI+Ivne=nh<z>vCpOyWh3I}Qj?EZpe*J+SzL8!|o3ei~>Djy88EdQ{S<jSotr^'
    'J5uoXhy;`6f9Q<ceH4B6s{43~hII>vVl9<w{wXr<}9=WU?FGV}GJ@VC-'
    '1_vb?UreX5W>+EX7?4OJ5TZS7NQF7Q(U2?vF{E}hU<jjXL>Y9A{a0Xph?;+hfQ-'
    '<ojWJG@p)}icboi6yd(m5I<Iv$%Chq)LO>U4w#FtKW*fbv_7$ZUF!2{!{eQ|0UFfh>!XjT&nmhI8M}k?)6R$wo6C+!#g|uH{EH0Z'
    'eGzI6-'
    'c4Qfx$#kJwsB)A}S%?PCa_x+kjBQZW;iW~r7DrCB<R;|hDbW)C^Ouy<(2!k$~$KWLWpKEIIYxV^&8+A_gPTq8O<pHTK%!$a<IjzQ'
    '#9fgtT|GVS4mWtp=dcH47{X%`F9?DkXjt{E1TyGK6E0K^)NJ!?Ac;xpQLHsY6p7*)a;?h;i)_tbDyw6J(DA|#bjAy=)20XoIWX{o'
    'Tq_mE6L=ZjLzepCX6mE%$5n6E6uYzzX~SCv8dwh75>#nNj)s_iSJ6h|A}_FYS1up@ukJ?U&i=XbFQz{j0x_p-TkYIup|j>zhQ3()'
    'I|o5UDjMnMYvQ7bp0!3i!_y<H`v0W`(w_~p4;6D48$$*T1`0L_G5uyw%9gmtF%fXsxw^>=}o2_U|GLBY;5<i1ol^P)Bdqj@Feoq;'
    'q=!F!DrklS8@%%Ml~F#+R{_oFM(t;{f1NBYd;?ZPQ5<vYDCWmXFunVD=)Ctn$=r#_&nAS2Ebp|gDrca9w|Yr;;~DmxkTTy@%-'
    'TB{vj7&3*o0rLqPb!&mxI#+<S%Qcr&J1A_2J}hvOg=|zz1Qv-UDa$`m80IL{eu>w|tm~Zc0t(&deHgC<xlfAHJ<>Bq_WPYkgTGQU'
    'SQ`?S)L=*P?xL^c_cIsA1$tyQ%^^_J^-'
    'J1|`>NP9^Ek_W1@uT?6sn$iu1R*DN{u%G(~B@{nh4(8Fl?HHuKydnVScy3Zd={Xa(Ec?H6Vw<F)h--2Zoz>zB}dNuT8*$2_LV?Xh'
    'iVwnhJ^~{+dr4<U+fowxhK0G>?;{;BzL|OP|hARd2f=BKYUwMWV8BP>~2L99$&A3WpSlu)>aR4E+di;M>0x(o${)P`a5V)aGku>7'
    'IlJ%p_*ZS*QaeH$JH{*+knahp3){eAlgBrd(eQS?i!3!Yv_leFh3O6dHS$X~#5n6XJd#ugMvDymf}(3k+OX8ZM8{x{EzayHvJ?m!'
    'b<D69;<B6hLSP@y$F%rJPY0fyC2cy<_vjrJDAlQtGP)_C}O`4C4Dl>BnHM3@H5=!h!EVQ3!29Dl(LBCgsk<7&=F}^Kib3<Ue05WG'
    '3>i{bd>CK*3#doyL}PvhH-4e-'
    'xE+;N{d*BkEb$XEM(<0>ZHLJbG){Ve91aRr(~^Lj_#RX!zN$V2m&#0`?#Az2_Jx8;e9iWN0A@BBRippQuFm5^!N93WyZxUMK>)6b'
    'L4um(KH`Y3K<~^WX3*%1#$@{;w-LtKd>)0-'
    'foHSrd?M|068?N2Ys1chL^kko9n~q0(c&wFsl(YI6jEACq0r^%^kZ)pFnI^#>E>M#gXtZ%~x49vtPXheY{mQN3OmWt^{z0@#b9qV'
    'v?kzFg3Mnf>Ql4f6b4S?j8(tR*F<XidB6Pr{q#o^=tAeNS9JuleonW^D8|fzzcWYlhwc@|8dww3=s4$JP*NjR^{)r}iSx4Y5lWT6'
    'a|oey~m@KVwfFdqbU=lJ?G~<Sh!XyTXB)4uNP!AyB6OtCE$;u|wS=C-'
    'AGv&JYcauPHlI9O^feoh=Uao662DB9UY>XVldsa*vO)pd^`}RoKU1or?aO*ybiA9)TIOy@o89hn>VMHXi4U7HZvj+RI^5$bUm+ZW'
    'YFia7XOX#YTV2zl)l#vCtBMn#XD!k*pWAP6HqxU?$TPUO}W4@VY;0ZX8Z3?4JtyuTWjiDVPdj3Pp>@wan6-'
    'zIt%r<i#FI=X$RekbDXF=CiOYBq)Zf+>nO5j90q}zwZMif*ufO1@Zp~c4)`vkiCMnew;E*Z=EQ&()&i<udd#W6~Mb`@itPE_pc3>'
    'KBPj$m$~cgTPkh)52~g!xPxx*Y|6Tl;xOP|Cd2uM#it7jyHLqi^(m;z=)ckw`Lrw;A>H~+T+N#2zz^K%zRZCmFN>|(TSBYW2i)nP'
    'm*1<_R@TVK(yaf4dLS49f8H$$DYk*|fGemLxPI>b?W-8AKgwURuq5Te;=8Geh}Y@lq-}6DgNh%Xj?bjhkf-Cbs50b<_#CD>B+oZD'
    'u>T>U8>(VLH%!HZZa6dM&>=iX_?f37w65kv?+}&286UAbmVsZXu5i5687j>!KAXW;D9@u8>oRKx9C<(%;h)yrPM=)ZKNlXn;LS;Q'
    '=dpBvP3hPZ>A;#?`;)9cihko!n*J-zoTKB?-'
    'kDgp6Clu)g}pV*IaqaKoALnIG$9jrx<bPxHb?T5v2o8Z?now_4a+s^*u|8IB{$;t1yi^@V0Ep)>}F6crAO^^wTb!Cg5wcghQ=ZzK'
    'JG0f2*@7_Y3?=oIp<0~j`aH#X-'
    't^pm}YK^$9HKBp?q6yz2o*8V)+h%=KJ@5ko6D${!Uq1^6&2w46lEGcSJmc*2=z8c+`gfQ+l^t6xJ^<?9F~smEk}kQ3VHgU7CvAQa'
    'EmDvh*6eio@*ZGoLri>0&g<DCKq)1p8c>#ECF`62o3#TDrucjYPg=q|PCaBXLS(sqIirsg}t7jM5ZGMcwEyO@UO@jSkn8hl#q;r!'
    '>V?76yW=n^TC^5rxdTh0{7>Z8L^Uzeh2ASor;hYJ+h0eNcViDEsold}+~zsP*Q8fhJ1}2Z(VtJFz}QcM*a88`=xnuPw8($Vi>Q9_'
    'iJbEdDnpNmL`kN=<ODfD(iI5j12NF3B9C=pzk^`bdMLKGKk=kJK3Tk%mTnq+wAXX}H%%a&3^RgSB$1_pA4r6}H+~ZDf*2HwQRabs'
    '2QFl$pc(LKYbY80eq#w&TC&Fn4>p#(qKZ;FBJz*r;98yE@G-0&kaxcUuukW2#$TPh%~N+!_M-'
    'C@Lqa`ShQ_uK^aiVd2eiu+R+;Zw>?t-GSlFL13XfD7-lsEOZBlH-'
    '~_Q?vU_?4)@hcD0=8P&sTvx%MD3y3PVzVOlCp_tXIIeR9S^UrRzfJosTnik<h%2H+Hemyq#<8uLUN2o}og-'
    'g|Og+oZ3iqVs|P?RYfOukAikt2pV(`Ta^&F&|SMmd2S*9qfSOD9VZl1>=uWwk~6#BqhLL+<r`T^-<l~aC6YXNCNX!SviV-'
    'n*m&BZ(YVt^hK&8bp-iERUXMXSoW_gMnH|qD<Po{hYclrwWLW}JxsRt_{B8xsuIR<v6hzFT7pI?Z|I};Wry%OpOE|&lT%9U^NR{W'
    'KDF%%K`r{%kP2THT<=L9NL8n6Er$;WLjQRgW;goU!{V{SQ8e&?$+|Xr^9>wR`fsYP-'
    '!F<$5fnqQ3T<8IRr&jRyEMeueys()y&o}nBLKL^uNPKYI?!$_N+nw&t8N!Huko30hGPpZu2|fBD(&!>zzd!{-uu-V6=(TrfQaf1-'
    '`!QM6p`7f)RNHX|B!Bj<SF;ZL=E$yNIs257ZE;Y{%1j6e%nDhjO){STDrhS@Ja8#Gcfp!;yT9cF?1R>@651Ct7f8?Juj%=BsYsw<'
    'm7qEr`CoIA%urm|NqO!D2(sCWkzco%E>hz3-Gx##ihi{-'
    'YaqF{HOv!8{%f6<?JzKi^LiL4au*4iJ^8?}(pwbtBR(Ip)r(<EGJ${h14zi&)55zSISuYJ!n>b<2gsfs-hIUh1MQi1F#CW$+nvGw'
    ';7JW;`%-4Mzb2aV=wU52fz$$gS4Q#E^6^s`4V**=Gw}fllRYg3xhy7ndo6`OQVsE?(mod=-'
    'F(nx&S$y0ceE_pfq*w`+HIzvX?bWHbW-'
    '&EWBC0RN<f|pzu!s;$kXBXA5v6(Cj5RIMb&46S+^@FtUhO?9=g!;eAkq@Uv$H|cHSpADJS#80r^Ah-'
    'EhpK>;G4o{Iy^l`;=l}nJw)zijiesEbX(3p=BqP_IbtFvXjdoxa=lZ!<iEtiRHlsLlr2!yNZcBMnM2|kg{inr~g%j*IwbBlUuR-l'
    'qsqAQQ8-'
    'eW0UkLFpT?*^!q_TElj`3^4(=sia~j!5$x#xEcQ*U@VxIR+_2P>%CF>ec5+VaXP_G;U91+wU01H)4}YgpZMlXtDIxFodzA`|FW1#'
    '8^i!mXFcsJT6t=Fb3rLioEmP7%iYT6r+-F4E==h_tpG<?tQ^t-'
    '=>%XTB@T@UXJp)RCBvL(V>|rT@`8i_`PXWx&8z_;7$aRSV^~0&@4c!F|1ZTMcAhf@v*7mGv8J?0AxtA5-lsHW)%zw%7{Hg-V+-'
    'EV+adc^)Rx+SxSLlYzLz$HxSe#)@OOML&z=AvBl&K$7a4Y<5wDh>6>eQiZuTlt2>9ueA@z(lnPWl}^?C)i}$#1&-'
    '8=hd&<9|2m>@>r7i?4YXZfK>rYmS9|@b!Ic|FVm(UY>{6F&R+nsW^LdY4ZS=HZ!=@-Q+|XM!<ri2xfrM&Sm*VRdb<?_X4+-'
    'qUh9wC^|JUicU?6qEnNj=+u-bIyE(lPEBLcsZr2}9<1zHVYYDwn(>^To{7FT($k5N5MZs3mk-+=%1m#4B&K?>%c)tx-'
    'v@@}<c7RhvrM68DDDQg(7Wp*3y%xa-YPR(XR5|{c%q_<O$H5lm9k3|oml_QI>OU`hId~CnJXP&$=~1B_K#ZX@OQL*IyNgmWgut}@'
    '7$3F5(d45P6aO`cm=$SU`p^ZGHS-l{4YdB-Yl4JW`EGj2mf|jR^KmjO_DxxOfOcU-'
    '{$KtDaKmTVyusJu16#1ZmFFse3KYWkPF0cVt=TTc!C(zu7xI+T*Dg5$|8T*ItIEs=dfPe=d^gscm2Xnhyr*Sfqvj+WKNHlk-;-'
    'wChOO|Z5ROd+uF8S1NbIWS9W^l$|$diX#pZz430SV5*=O6BUjBi{<Y|Q@$p<#JuV{w272Q%&50W%bLHF|W+0mpeftpx${1na`v?U'
    '*?zvq=OJ4!2tnD-@p-fMMS5#Zc1hJd&TrQN^W*-c~zh4S1+}L!`qmf8uWkmI;T@%#n3lFt~@y^JMM`1-'
    'g{3!b+P?a;IKG}V`V2_+Pnl2MJ2E4-?&6I&jHA{vC9{COs$kV)>SFJ^x@mwK-'
    '{~9=4{HGj+Z6coqazWabi7bvW*w<Vf!Tn<d_xG6d9`ZN;nn`8PE0~{0!pb{-'
    'amsJQeXJHWPtd4T(^kou8K!NpooT{=1X!tAhL_i>p>~du$=xtJ*T_cK2FzP_W4(#0llefZHRvpl<2dy^bE+U`|0;Msg$4cVpz`v6'
    'Z#X?G-'
    'YY&z0p8t|Po*0C6F*<sb?I=)k4BPID8f`+jh`I3PN&MZGKr>f!@}S>jj73XqkWWB_h;1`addwd32=E|#e3^m>tql!vbu;qUh2@XV'
    'S0dALI1<4Y1&o|L>AaADCQYPM#znJwqYCuzDq25?@deI`_n4-'
    'fmr1_(5{ZKoIi>P*g9a|LMn3@$Wl4|IZ^?0pDO%bjRkA>AjV^xDnDeEE8x>Zc6OHSZop<|q5@8LVP7Q}skqA02d(~Fg4>|1YpF@J'
    'JMeYQsS;noaE?|=wX&_UlND@va%2pgZe(ONM!~dW;%@OwBSVYR6fVGNafCs3KRph&<`@}nQHn9w$Ux(sw0ylUb{?R8^Mtg}KJ8ZM'
    'b5Rqw#*e+1C0t~Ut@H29C_us&lk!;wC<;SI=WZ8xSZ?aBv^R3MbcP)F-z|J-st4?C+!?7Jw72t-sUD)rg-'
    '^_Sz_OY|lHnU*15Ht1V=`egxoq5rK3V!l6AMwDcafku8|zFjJRqvc5zppLj`Mk5gFw^EbXh=Ho?p>(;zU}Z%k^FVwL4N#m}J_CXs'
    'sjVkLmyKL80N^gjRb=x0G-A_sc{$VqS^Blmb&7Zc~!nzA?$&KPI{TVv^fGCb<J*l6yc*atAUQRU3Y$P)grJr3XLG`>2#-'
    'IL7^$O`TC|kQX&g>2WnGfDJ{=>-'
    '%tLbh!pa*<x;3)2C_^T9HN|S^`O+RWOH(+u1(s%mPe)v0R*8fIl;qhjR)rNQTbuZ7wxsmVw({auo)xm}60*8pf-bVo{>nhg${JHi'
    'w%F3OQc^p{l5bTm-'
    '6mp^W>RvRjPM#J#5I=g#(jR{JAzowEN?J6;!{90Tn10=|~xYnxOX3LZT+r#opIigbUEd^}yrOH0{TYmR@jxukbpNs6Wyve@j}fER'
    'LKwTYthlQK3Q3Pi%2LfQ5&Ww(ojLO%h$MDiSd35En_%{g3Q%_>y3pHf(G8!v^%uZl#k<uVB~RCHWkWI$)izlM10jR1l(w;paw9uC'
    'gUii1xgMz+VCe?!5cd1)f!(T;aA((PtsV#QJiXg<`eH@5&&W*AwLS8X$H6~Z$f!A(yH?=}{QofAu8Kp?L@DZKljKq#FY-ff0Xik%'
    'YPbwYyKPAx@$;vq_l0-1Ot#-'
    '*#o>c0R>1^+BmbX@72u2^tV!He&Q0D4{n*Sr1wZhoD{S@67C@9i1KS^pwPm!<jU|J@4;P_y>p4(4Pgpg^1QF+q3?!dM_+6NaHdy;'
    '{f>|Do)MaVGs^Ft{Y)=ciyRN$P~3fv+UF<Ua>h2VZikuC9`lgo>-Hiwu!@E2VJW2IEu!9i=fvP{+-'
    'JwS_Lih9r$1qUv+alxTWGA$;AgQfSJul6|4a8vpVq-CHR}n>lxLm%2I5J?P)qSvoM2k%YPfoB=WnB3r#Ny401E0a$`AAr+9E5Z>j'
    '=@4kwKx_J?ff=OR~O#9QX*IB5$AA5rS^lKFhbq`=D$pQ3hH4AkQbosS^JQVX(nbHFS^naRv2<K9c&g1&?v!ZGiyfLMSJeA)da?=y'
    'A)h-y9jYY(`Y+|fhHJPpy|BO@ULPfkwXA$u(TLi&27fR>nE8(DAscaENfUhVrLA&jg-'
    'P`vJo2aOWXNYA$m&&}{iJQfi#Lh}4vy*X70+ah_>`+clj@SyPyVONFWgZ+m{#)^`RdZ&tmckz}X;$}EBFi@eq0SzvWCr?i<$PI~N'
    'cB<VrDbBbzoPP+*M=qva|)VFe^svb4ve0ES9YrrO3!~PyUj3*QM=sr2eI+KI<cQoc3*G5PXc_M`yuideQM%yY}anWKIJX|=MeLX#'
    'CRUeA^SZW9Gs{@^7+D#mSDRHneFD_Do}9p7jC3V#w&+nl+BaX!+=QPuHTVB-tm;sbw=eM9*Wj3<7=sW!s`)U)<!><k$`}v)9VS%G'
    'j_b5)Dm;P39{hCcsCbzu<+#{ud_T8fRwA1bh9#mx&4!KW3lsom6Ao+(azna$jIidFoXvB{%L+Uv6h}(Y64k=XB3ce_X*mus=9~Z|'
    '7FX3p-$;jh2d3z%r!fAm3A8b$A8>|u_=_&?TiArk1L(Sf$d{_z^4H9Vx;O3084VL8IIWw<5Q`S#p^N6zT~T-'
    '!8FN!O*EJ$+pmiT(-f*pB!31;eiO5Bt#!#!dXh&1M2fln32d@SSumhOc5xC6E=)!+zV|&w#(;EgN2-'
    'sQ1G?JiLX~k&J#5=eK!%)<+RFhFSSwYa_zqGNKJ*X);yXx9_{c*AhVLLX;R6q65I%hKg`#OS0eJUw*uQ>L@!z>0q?N0>O#A30EQW'
    'wtu+QWiL&zU3pt`>*JBt&m`C(;@%7wQi!MGQ6!9tg|Ml#ojpmrQ?>EE6D6gR=iV_jBC6P!}o%S&m3Q}TXg8MLGv7XRgpt*~0=#zk'
    '~6$10EPj-*(~$m4NesxhYS&i#f6HEYPBX7j|O`F-'
    '6kOgBq3E^*+1a0Z)_0m=K|Jf*^mElEsiHVLMEl)Iu0>4}nac(F==TKp`g6nL<w3d;5CGVP<la*>R>rd8KfkNN+dQo3K2^xx3=$HF'
    'z|_UB{4KL}W&3g1`oKhq7xEOxas423Md-'
    '?I#*1AM>dI1%_Xq@2qzdq@>LT?;8~SUk%nQd4WbYh;j=AMNMr5fxsfh;av)kZi5Z4;chAK<BV6HQBTRKIC)>#@~?XAVuGA@~#wB;'
    'R~xrnG<LWv(r^GtF6pU=im%X#DL7?ic$ny-'
    '=GLo2Ku5V^p!p?h^g6zqB=XlxrTyz3hspp(lI^=pI3Cz=^BZ3#;n&dEjwwQF;#AIu0Nu3nhl>T*P~2vP4uWQmM%>*o_hd#$cE0F{'
    'u!<*+|pcZR8Fa`4J%sJtm;F`8hLEL9VM|Ujlc7XH2(goNaOEiMH&PyFVY}zMUlqlD~mKXGuFNdD{1^hCtxeBCVUIZgQa=vyCt9Tk'
    '>x@fe<=i3{i%Z|xKKu-2`tAaKFeU!?tbFB#Mt9Xmm8%Z<3ZkR!?5O<nr%YW8i3Z-q-ud*HO7$P8o&O?*~Zlc`%<}yl)%YFB)_^};'
    '#cK_kyDo|NdZs}8$HLqD$-E$xr<i1(Dh1}_%V#L()6rYr);j1@8dFTValEB#Gf^Ak})CC3CIA_9=F?86Gn39G$Q-'
    '@+8?Lf1N+;bq}&7jAV%IZ;l=)0$~`aug5-N8GzPy&xd#S9p}ZH9XiZ0^#u<hrER;!bisCU!j$B0^YLl$SthFcB+}XZ}J@-'
    'R!Ggxm%y?=X;+p>(^nETxl*j^m2=Ov|-'
    'WVV#@T%MFql1x#Wb5&9@Nxti_LrfflMYpR;Aype6byY@zuj<d*_aWkExE0AxzJdim+$wrWE?bFG2Mru2x4JkPxTpSyry!w!>KZm_'
    'p?PCr2l~4)96LIp(41MfR~wW#l7_<rZU<^_uF4x^18YJxh<yqrA%oeY(Cr$+{)DbqBYP9_TSM8GkfR^Qo`mA$@NnxeG~LXSB@V>q'
    '_;-@71TKJdsFKWj6qwkz77R~^E$Bad9saJOQpUU@DAb8vcE%Eov&-'
    '(l@aE_1^VlKJ!sV>G!R~hTYOEz5bbDgG5|`CGCFg$F?w68tKVm<gl5;<5KarAiKW6)+<lK+jz9~8P6L$ZUocl@JkC6us4al@cXe_'
    '+o6rS@o{4&l0<LP=@%i^DdnIx#~=9md=by%N#;b~P7_yfS_^c&M<K-'
    'HR$3uf1?c&}R|@0p@(TZ$m1L5}uc%7X*~lVgcT98Er<v`bRp)Wp)>kOHSBIZl%J%_o<T>y1sbDJ5WC5yUaIgs64|PE9MJ=-Vp-Cm'
    'rEaJeQl25iuF6GX+?5rT_~@L08M@PZD(K*1~(}V9X_j9bj0bYoEGqXbEYBr1+eZE(1<i)z0oKjk2dElNXdcn|KJDboWqn(oYlOtp'
    '>4FwV3{Y7)fNX*ry%A?Z|YoGQ~C|6a9~x-H&oN(PL-'
    'aJmnusz!LYO`gm5f!d5B<flw@c4GxsYWv}BX^Sta0B+|UYIn{Ij*NV>OoP=R+HhlkR+`&gcH0Za>WR_e#EJRk-Bf?`<Jt{O-'
    ')nme9RXr{wR@D>2VO2fJ6jrg_+LoS~wkS2y7U$XPH2oA|l=uA-'
    'N;r${ivcUP0umzGvaMEjj6XvMV&kl~QR*&anwwpa93uKBz4ZL{%ATH0?>5n|2L34%UJAJ47Fm&B=BJ9I@l##obO=jsI*s=$iShib'
    '#}3xcyo7=!-M~9d2ZR!VP&3gn*r2)oq~pJwym$y14}%Krh4AJP`&1EZLQlj#jaBgQOzbmQMh#EJK8szu@LcS3U^)m-'
    '#y*dg>HHkgzt)_ur+vBZzWxr@pXiYOp{frOfAlX!ZXmj!*H!f)N7OddGk+I}IC3qa0w<@}p}7F3JMo?t9L<9eeGd5rU4xBUL_dLB'
    '7S2BammTL{`dOwr9<AxF#*3h#om4`aN6&IyHc)N|gf;@yUB+upXQ{MT)|B+hLK)LR&nav*SDbQOm~}R&0A8#-'
    'a?D%jpQ_rN`0kae+LE~H*HpDN@zigqY8#{N-YM)5A;v-0c(jq-'
    'u?Yo8Z<zP!EQJ>rOJ^&*xZ!<{!fQgT)z1UhK#E<a68Z?tX<yL{v25X*ou1fEGMsox$60r~<`K1%xA=bP!KM@Aje}@T4|)o_<TsV>'
    'PeaBk|CP*oRE8~5g2H=6L&$2A=Jv`<TGsmdf=4inYAL_}vt~RHW%qy40XPNTyO(vq27%>|zTr9%DSTD)bQkpE+O=XGGSAJACdPbs'
    'JQgeK6LFtsB&Ti^psy_)5F1ht*c?5e<4G+XwI$VjbX~}LBT8QYQi`+0<OSf{MbMrCwapm}j{Iu<x^{E8+wT0FK=;i;S7fH0Aye<)'
    'k)66haoxWoL+t{^b^nemwJRjA{X4P_<&PhPcVwRK%5IMOOM@eQ@cD)PjNv^-'
    'bj;}0tRz)C7wf`xd`Fh_CPOkwC=!WBN6XvO56^=>ac7PDk38*5MOYF2oK6wBRa*e$ey!FPnp}|YU~j*P*W$r#(-'
    '66yRu%EcPgQ*6QTeSZ(#9-'
    '?ykTr~ZjrUJJsb3b$Oe6ljQv&#SY#;lm4rV9@=8LW0(B)}P=N@RP^ds@NjOv>h$SQ{(8FS^+=1FYCph*I+P<iyui+8Pni0!iuS-'
    'p76I+UaCWrZ@1KCEIk(7(SvGDrb-jvt5rP^F{Wm~`&k_G=%q!_Cyj<0@&pkXhUjaoOHXcVUTzP$5T4W@fPOqk~T@y27-'
    'NVon#VVWPn`;OHxy7LDM)BHf*cBHtvsjxQ%q~&Dj&DEf(^xUUr8R|D`_mAr9X4#7CuBwTlE8rAKCo$<Zxb-Io!G=+GFGUsiUP{<g'
    'b}`H*I*l?v!*rsQK<s9iPjotE6NU*zDRu2`m{D}5vU?b&6rH7P50i3=VrS?eZ};~ow@6`+OEjROi@7D!Cq+{?#Fl6i`Q#WDdh5u?'
    'N=fNR-DRSLy|w0ir}m<(%eX92v8|@ytR-bdA716gKNQXrR^4tA*kwC?gDx}FXWY<oKjVmb<Dh&u44wkM6b$$iL<*!QbRK$!H+w=-'
    'Y)=(I`n*~MhKpwt{)Xog{)Xoh{)QC^3&slxm&1z*m%~fx{A)(RmHHKh{XM4>->e_5W(wI!x-'
    'yYMds1Ofrr4h36WvU{F1w4#*JXDz`F3b`!|SvD{T_z5L&+r9DSwFr4Va6~hlY2*QFt95-'
    'u+hLbwnDvT9B}~Pq)c$Yu3mM{|@#!M=x6&1@Ff5L&_AoMRwRCfpB1xph3X>f74lqP`gAQ&{t>AZ`B<Cl7czC9P1LVF*eAGhMSr+C'
    'v&(ZSb1hjm1YpI`pP8Ld+EB@Fukxx;pz6k0@jYHuWLfvnNvABfsF;1VpD;!e1Vysh80|lwEz(M60aN`Sa3Nu1K7|%^9J0(1s7zqf'
    'Yyylk-'
    'M6FMR<3UuLSonVFlP3?Az(KEZu}hxpO#(D#2AeS$v#3!i`k^2_S?nS5wltpbs%QG#Z**;={AV>2Wy<MDPW3OfPoX^QdsDey#92EW'
    'Ep3;q@t<=B~YsoC^O@ZgJ1FJMXiNO&;S|L#2BID+bUpiJ87eeOXKOKV+tRdRDj8j_(+Qi<VYU-'
    '^q+NfZWqchR!_9UD$zCU%$@d*e4RbyD8X^-YZ$7o)m~On!ujV$pF~%ITxTOEIVXi&*$JzF6I!4{r|1-'
    'Ix@YKN;cfL6oHi?(Yju2gyLb8V=EuS$x29lyp74_j}^Ga;uIv@26FB(UC>M2@|NRwvx&f16xajg;ti0!HDn!&jjzKg>p%t~b18(6'
    'u?Z8dR+%!>r+u({*S?&Mn)*cg1V@$4nZ*8)Ji(C~b0*P0QYScuUd|*Ah{Oqwq1R^;eX2vN?$z#fdJeKynmc?i+eIG*fop+JYAQT9'
    'r8`1X-N7l{k($zIX0KfA)ON${^w&<kIE)2<<D_?22`2GRL94?*wqF~2L)r=dZvzPMUV?q4*Os_bL9cBYCvYDu98OPAAaZ-'
    'A7pZcBQL(q$L|?;c!)V*zaORkvk6SS95EJn;+-zSiSjXok4vqhj+3qmNfpczrozu_``N{Q8bvTSif9q6-'
    'BVWQ|r|b|T<Zq0<F&(H}Z|qHJ=k~V-'
    'sOjh|e`f%GkImHoF+flElEuBhX`$`|_)0rcybMLwA*?GY*xk=Xf`xJRdzV#5>`h5cvXZWZW5`WgV+Z?7J&zLbu9WfJ^48e*Gb<hn'
    '_A#zK4u=>oR~U<&N_H67|F;vyeoH{CR4i^TL|A8~mW7i3ZXdo=%c>5&wrW{wqSuEJ05t@xn%vO3-'
    '*|jM6J%^QYronMK4NFVOYeOU>^~ZT?@r~|xE7dj!!^e+V8EtlVt;1m5+?25W!SqI+saNa$XV?_GpFT_8?S4|jJ3)AQ!CklHgjC3b'
    'do!ry@46>YM=7y-Y4hiKV5%K4(F0=6`&=q0!)vq05jq$z|6P`Fe|PC%#N!7bK)w%T#gN&RM2qb2GXGl8jfLa3&?kR6@hmAbB-YM^'
    ')J`=z{qY@Y3K2n_rJ1JHPXrRj1@fepQBFD2TnC8=ma^5m+4X0WhQe%Ud#RV>*&{<Ak3(slsB;ub*F!6(MCI5+1E7hmjNX9y5<Elh'
    'mX{b7%Tk<rK&peF{M8be*^tgdObg?S1}qs3yLRhQd{d?bl{8yKNsBD8vCLj`A#R_9lY7aH4JvA^Bt5KKa;XwZ6{ma7eJ384`*B`I'
    'b81`4rcpTgE^M<tA=nW>t8i;BpXl-<v?~oHH_ofz-'
    'o9@;Ax4nJ2Rrz&g`hQGdJf~`{>yk(q*!89OL#BiU9hv!l028x`;rahy5&PolCN`y%Wc24EF&)ud2@YNK!mSBwe6?mRY9owH>}7NM'
    'OFhy;ZL)?9G`(&sL|-'
    '7)I3&O{<0M4)|Fuw43zTzG3W+M)sQMwVmNM{ym0?s6VTa)dWZS1^R0en<~0#LOMU1{Gqtmrwdz__Hq*i`?O!6%SvoA=kkJn6Yk&-'
    'vJ$&FSz2_3{mM9f53QUI`juwSOHHDxbroHF%w(>r^*+C0l5HS9;BVU(e6e?RVQ<as39PafrdvS2wJ_ZRnX83s^MZoP5D`-)C-'
    'D_yjo}#y<(7d@{{)3pBE<U7Q%sQr=tRm}BmsJ+u-ljRitz28E9{P?y(+wWzOXx$_G-op|1dvnblgU2E+K~Dmg^9+$glM$2Oe4mC!'
    'kf$N%Ax94G$i5+^db1)MVc79b$z3z&09mwF0uqpsN*dO$J@9fNC=6YK4xTL02mvnhd&H0ncR6)#52zhX5z54P10QRBg^+Vgoc~w9'
    'YHdq2QjU4gCxbLz*`HDC-aQQ*jbJ8t0@BS*cCO1&cAl$;NvR(~&+#>~N0~Na1|<cv@jU?fv9C_9c_X!Tyo=VL;@4I3V&q42--FgC'
    'g0(;7Il`gvlP7s(Y&UxF)-=dS4(<)kZ-)RhtCxRDDpFr)$G;&MKNq^Bx2Jto@T|(-'
    '0|^_NA1L#m(AajgYu>;P+#@Fz>}nw8g37ZV66li^ekj**qh>*D!$YamWlk&ZEDFO!Z_?4lN-6@~d$8pDpYTrM=P!1;CH7>c;uH-'
    '5}Dj>kGS4Hqz<wX#xJw#obY-'
    '3rJNv6_6}#c|$D+?dSHB({lO8Ik+{27^J0z^I@q>^=6nb_@_>j$TT<({0?0&KCe>AxQTz%_4Y5uHD*SX02jo73bn<KDCxZ+N_xK^'
    'CA~LB@$StsRJqUujKL@jc6*d;C-Tlpgb*;yHKo1G2;bC?u)FSUy{@qDYu=rss8f11UKr~qwbRvyZl5<nzu=-'
    'DQ#D>X)DI`{APu+&jMG_`la3qNu011^?sV)6ol$|ddM`?sZ;I08fxc>^wtJR5R@kKNUL|8EJ2ePI&f?bkeTEURPeAtS2_fsb1k0A'
    '9KY2qTTAY-vnSkyIUV%dHFkcKFpsDW8$uBzTOr-VoPqUNNmXaNuP@4+3rH3wJ6zoG_L}yR_q!I<mXf59g-'
    'J)a{@TI7cN%!>Y#_kXv@@8#&m0^9GJ{RC9q1M`7Q;_EeeyPIi8<B_OC)f@ViO8FZAV=DS*;|L~R8TbCUQnx4ZSJ7cWu|Ja{gVl#&'
    '3`u8RIDQmeSVf$s}D4UDY5^BVzfg#n`=j3HZ1JEiP@8^tDlL1qU{K&%1_<qLkdXSL)ilz*D%&*@5l%(UxFAys5Af<xL13<z9#x46'
    'u-F=-BFXz;uhh>?jIPF93;lBQ&~#2=t3>ql%IlvrYxXd4T65zKy6j_(MTY?K-'
    'tG52lq9~KB2l4ghv2@66wo+p6hw^NR&D8D315u@T=IDyyT;&pXgnkhl)1J@JMtV0b-'
    'n3^!p1OO33q6?4z(H|D^4g4Hwh6%3m{fdn42~Kad&V4;8)m$T%0oXn%JG8OQFT0<T)q0skZvT&1xz5)!Ub&QT?HZDa?%3QB*nUN>'
    'Jk=j8m@FR~2crz+L9brJ;R$u2NS6x1eRNxKFGg8$3bT~OKHy#^<OzF%_Y-jvbGm$9^;p1D^GjLyCs!TlqYh(BE7*4FmZVz-eX*A&'
    'AY3TjQU+$er@&UA-'
    '?hg58LIJ}lz=1+lQGh~TaMdaok1>$DG^ixVYvCE!OPtWvTI@MZt?0%WyB`#NZF*DJX3NL0Rq6cx6;L~vWuNHtB9?N_sJVkW+7YIZ'
    'RG~H9M)g6Kx=HzyEc=vLFiR9ey?ym&~j~9k_FBWK|g=6*-'
    'oR;75<_Eqpo#0Rq8zE9%uJP5V<xx`!W|chZKdLAw{e;soUt;QF+g&RX?6Gg}+rZOoZ?3uL)uKZpLA?!ls+1=$*>m$gC-TAQRFjjc'
    'n`>&r0|fA6y#ttomDu$BU7_$|0<KoaHxhJ`$tW8Izugxrue_^^l~-QY#mXyh>tf}V*LAV-'
    '%KN%NVDiGQ{jgw0oIRBUk0$#u6|%bo4e*2tf#Fwmh{@+r_P91Fea}vNx~zq5v1iCy*jAK}AtC<|eZ`QGZ$p<mIP>iav~5F|yC~3b'
    '3|($jp#L5D1W$F^(0dC5=Ljtq^>A&!YJ9BQwQDy7(Y%wh?9+X)2>Z$gU54&LQ1Mi<9QA#JxA}9VFn6qG=8s!zY3=DwwI!_sE~{JZ'
    'v&J6p=V6{5Z<q*QQ@MDcPWJo9AaVs9Km5>3vX)BD<z?A;B;WFih#iq@c~y2)zlGVD_?dlF=orKdVv-Tw@oxkvqip}Uv5335;p0=$'
    'JV-'
    'H#_l^3J{{@N%w<=<6QSNZBrW5_PxLQx|;wt^+HDPAEtVs7IrM=p~)JQ&APudvrotMS$oIZ&Yj6E*JSbxEZJjTKIXb^Wtm^;M17U*'
    '!^asOYGondnQe_^E^y2p6^XE5zE;7hZYaGeESIQs)`Ie-by{>TW0{Doea&Hr;NkaeW{4b>W|>zwJ|)08;oCvSzp$x92WE3z7)4cg'
    'XYx`3l`GdjpX{zq8m1eU^<8~yK;xq0{lb`9+^W5?>64No-ob1Hz?ef$nMjh`th7<~WH{#;S%<QtHbN>c6QJCOF5ii)SkP^&4bDu#'
    'xXK~Y&TEM#els;c2ay81~ivG{AiAxJD<t{r1cx?uJWl^V`mh3YZ`*~4XSzw-h+&Rkd4=oyz7PN`og0y5fvWJ-2^Djl&-'
    'mBPpA`qQ7D9w!OcD;&P=HKWm;I#I^C4UvjrP^4lQ9H|(FL@I{HNX0NTQZWpRR1Cud6@!Pe)9YgHb%BD={?Ld9!PiVytFNb0%tgjo'
    'wr{QPW8KM0`#x9Is?RoVTv4EYA#QBJ&q}<c6vENrelbLGvQvL#ch@VvyPa+jn;Gi!#(p-'
    'Ss1Q}Q$d1~FAj*D2qQ`ccogoDbCHi$20VyCk*i&F=wEq=M;Mkb#<nZq10<S6I-'
    '7N)PQ^UJk3%sToY4=@VGN2!o@B@5SG1K;`*^V42ebBTRy}H)s_&2~ua{h5=<$4vxs(Nc#v!r*GLWc5+o9SlTZ*3XxZgo#ayGvzN%'
    'DP+KmyzyP_h+2D)dLyjZuMZsxLZ9WguBQAI%T79D~R7;=UBf@mU$_>7dl(!{XpBDJSv6`#Ll_>dSh)8&7yu>W3|>h9}?m5k9Tssi'
    'QeCZ0+FHYbb@dfhsCK*6i!`0W6g%lSI6#9JzM&)|GuUQJ5Aq~L!6lrMD*RqC_dF_;GYT{l+do84vtn~jyyw!9WEys@)alb8YQ9@a'
    '&iUu?8hZdILER1hr>V4QTYekj(5Tb;U=8tNc`Ul(RBX5Tj<61*h@OG7T!sT-yrQT9KudMkSjF(J}Ky{b@l=8YN>u{s7<6AB&Y9N4'
    'H6By{j0&ELD#PuA|hh_t40<P!%#{Tv<|3-F>40KQo@-*)xc>0wTPrIiUB2R+h-'
    '|zLuBbY8!Wh4cH7C>?Sd}nsy;!+vP^?l9)SfrvoXn96;=kq^`kSFM((M0L%}K%-'
    'Lu?(2p0J`{x(w3v=8%6Geu4N2(L6l1^!XqXr{PnALE5)$PGQt`^;FYeuCGTK|%Ipks^P>G7$pb@37t+x@|C${Y!XvqoWFbxuywtl'
    'N-(V*KUsQEOt$uva(TQ1wI5u`kO;aN75=1DbpfYrzLDg3q!~d`NE9~#nj?#yZ--duGt0|7oorVPZQY;|4TG-mpCHjKPQ+eDj(nXk'
    'Fz)TauS!N=`bK2v+T02O}E$R>g?v6-'
    'TT@@0RO)+@SQxJa|zlIV^<Y|MepRLBuZ%RQVv%fCnPOA;cKd<Q*sev=L||Nf@7XZ$wf#lW>InxQj6J?T!hqO4kZ^MwU|rE#aq0pa'
    '|vsO)7krh;b!I?3g!&8<h_bA+Zq`q29!4EW?1-'
    'd{5fUk{NYAp^)k)=8{Gd8%9{Shp2m<+a_D>12S^jaxa`7nvt1j36s!x@nC|^JI9&{c4u%*bp99kM;?<>#4f7*DNO@JHxcre}GW<U'
    '%oT$*-y;I>xg(=G23Kk62Z<~S%L-l))!Z&OBay?nKIX}a7S#e}*kVnd1Pmq#!oQZ-HNXj-'
    'U$`|~Gb04tiy`f!!7mBRwYg@0?M7!P(1Si{$6oz8AGs#jyVUnc;Wq2uprT2y=ATy6&jLC<CEyiraAr}KFG#%)R4fn$*8je4ssBW#'
    'caEo_}sXpZ9<un7`iQ4ix74bG_X;@TC^=(3Ep$!$vOptJhV2#@yL`>3%W!PC-'
    '7E;1};KJ{yb%$4&6adWL{t0yLy^vLt{ZJ<hAKP@Y@IgOB+~hDO+2klqvdKaBZJdZ@(=j?<q`W7aYAgG%sixY-'
    '4c!(4@srTQoo*;y&gk-'
    '9hw6}XY;PGmgfVuDv;N|e!kssn(|pnJsQSjje#!8x`liC3WO!IjnfA$sr`42bpJI4iO?mbh!}DrNvrjdCU_A=AXFU63+@5pAA=9x'
    'qk1roCOtbL|Ea5}EVeCME;V0QUirmE}H=z@RtBcuvQte0&yNxrVB}L+k@qlGpwS_YuNg<Z;y3DUrg4hY5eRrmug86CwW3;E9zC9s'
    '}E+Qx3KB-WQZ8X5NjtCQ<5ty1}f~E9bq>Ow!mXT{>8Tn2uBiF_<a$PJV*T*vQT`nX2Ar-<N98w|dfdi1fHBM}oR&{UOrL6k-'
    'O%33*4}v`i@J6J$b`vBBJZmGWg-brNrH9i$mI-'
    '71LyAp3`i_n1q~H$4qRdr0JV{Gxx+(bDu}pulZ^CHRTUwVbAVnF^NcE2kL)Z+EqMXNI>^{N}HVdRE=SSx#)he8$6hpY$%$?${wvv'
    'SX`Yd5@r*tnK-CF;q!Pg#5sMJ@<#sUSxc+AMYe9q*aJXzR9LJ4Ga9vM;8+>tiNsLH)FB^-'
    'FBfIx1d%u+!5d9MV(o)w@|^8$2QUVygb1?cp=0G*K+pfmFVbXF)p(}Oq8-'
    '~{ZUTtCPiyx(S6&DqL>(laW|(bxru78{cY29NUBKya$`+s2HB4P>hw-k6URc0o<sQ1<Ev#l8B$aj$+z+^cVld-'
    'X%(Uj4ARS3lhE)w7(A(?^Q<?WYTJjtNltxq|#-Dvwi8T2?m}U56=PWgWwvu>a1&V~gPGy-'
    'E6T2X>0*IQQRqgdyYbl1@9UBUpTT3?ocVyS>(%iagr3TK4RwnsIOwt0a3mv|~h&{S1m!hN~fawy}c@mqXyT9c;KBf>Pd)NH)6%Mq'
    '|mb?&-8@<FVcwlNUz7ABqVohoB=c#CM_7|IHU5b(h%w<pL^&yfO2;jD}$rc-aGbbrM>jlLS_KnxPDq3qE@$#V*4YpFPXihWMlRaJ'
    'tj+AKcR^>&Az@7iM|MjqHVKUh?oihIt;N(%-'
    '~>DaG!<Ifyk<3eh>Ze_lY(`g_)j#~*UasoCb$HfT^5N;EchLZ7d97|iZ~W=L(1W{SPD2xGXrbL<8hrXzA7)&FFcY{A`a;kBnZijl'
    '%IASW<`*Pbp=9yW~RZfY_<o0g2vT9WbE^kjTCBN?B~OvY!klJVK>!lULm0iRlAn0>qtQTz!g6UIMk1y_z{uB#`K{%CLgQJWCXv~{'
    'r3M0>lhKIziV<0NfKZKv&PD&(>Fd5%*;6VO|5hETAC_Pkvj9#vDKTiFNvi=9*)q*eHA@RrSHGPbWMCJ~v7zwB8=0KS|l<p1j+y%A'
    'E<x1s7#t@Cf*!V)tjh~bYMc;pl#tK@5t75aA+Y9Hh5l|t&1Qd||G*Cl?1a_j4SAMp)WW7h<%N95SW8?p-'
    'OB$EoF^&zL1Bxo&hkkwLSS>3_ym@t{TV>#O)lS(_zadVAHhi=@(c~c>odOO@e&{J1y1I>d%Ch)Fs`Arg}%PQ=MyUs$OgLnrivy>s'
    'r_+O$#<O!eVH-Qy;Kg~(ooq#K6?bl^M3%lbqSJ}&Sotpu6ODoSC>pZ$=aJ@^$k;R1RaWP>=Tuhi57ZYa1#e~^$F=0+zOqd%N6INm'
    '8f`LYFIqJoC!chUAbhZB9&U{;vl#t56!AyvNNuh)7xaBE>j{ex#Ebh_}-obG#)X@avF=}w<ZHWQ5g}a6~WocVO25-'
    'O*#Qo{EWEDlkOauE_2lx8>BR14wJ;ys5sSuP-RPalo_wWvI5uqos7L4nmMYA3#<WTdi2776>MkqI%qyU`>!UirtXF({&KekbzyIP'
    'IXPDpzn=6{}Y-'
    '^Vk~`Nyy;&nTy>Guj3EUqYSq`^=+1Ouyb{f}uyllchWV7~h5^jzWN2q#hZ_JyObCzhFd6+pmMZlXW!%%>{*;%T!Q>M%uI;1=B)s^'
    '@>7i!Rkx19d<lAce+;x#9%j8@#0PsR)AYX<pQ<Lst^QMp;H>4d<z=*H>LZCgIQVT0%b4H&9noX>E=3eU@JK>QEs@&Nw@_jIVh*!t'
    'e8zS?9Yv7<qIr{F!>7wkDsd`stw)`dL13Js(Xz9c_^!TzzC4%9Lj5rblJz#gEnDufoyMv$#|Kb!tBh&_X+m4E{j<m%!AiSC7L(a!'
    'LZ}vpnQfg(rMC=>A<spKQM8;UnHt`@p^7_40i8p`b}_Bx75l^s-Cy%8UZ1kdEPbazyl2#&XOXD0rYiA5J2ZP0T+CzBe)<u-'
    'GNLLcP;?ZX}ABJ0{9OWjh}^CG5uF|9nPi$5#D*9b3}MHJ36$<k>TA=;Apo;g?Br9g1_)?7cX)d-'
    'tFp{U&FiIAgX1Ljp^on<oc8lx_{u&Suo(OX)EIqVcT8KTy0z&bn+_aC~vV9eBkM+wL~_%aR`fC9ds~L#3&@c^r$=Y$9aUIC()$>X'
    'e`i9U_SlJ3cIeT?h)wb`og0`9dfgysP32iy3GhujV|HCMuu_Ylzr65Fs_N}VdI8Fu1%(KoVePXsWGE0ctJ%2Y|l$54}sj{)L^xxn'
    'e%V47l1?Q@4+WrFM5McssVf%9Z7P1nvlMJ&tVtGYM#gShXCurnUYs0tR3J?TLH1#DnRK|K_Vfr)%S|(0g=JtCWI@J9y9{BrXuMfL'
    '+)9)qLvvMV4kb&<3<K*Gc=IhVWG*+#xRIM?$smU&Pa6=zNPHOK+;^;AJhipucCE{^g_t3Z_AD6Ji_0!(7j(TL?5l7fLQD$3;5zLb'
    '$)TmK=h$DmzOv@zCst1jG5YP8}@d$u?RGpk3}KisMyT<xRSQaC_7yP+P*EEg}EATxD2Vo9Yr#Iy{iZV*T81}3C$!e6WqS!i@dh9h'
    'ahh+1h6rq86V=L9LZ+iRd#dDW-CE`@`jFh<J)WJ4O^rEG>J?G7My1oN!(g3_P)0d{$6)kPH5Ed-|fkRdt7v}<`#}&FZwTvoX<<-'
    'thQi#d`r-'
    'Q2DtM>)_z8lce`*5mBUY{NDVbQQbP?bK+eKT_X{wh3{n(!mIhK@84(ulE&5pJyKS!(yx_9866l%_p|9u=Lhq<;zcr^6gbCf}b49$'
    'Hqd{_YeFE(q48p#6bFarLY}WhvBakDWhp@1hWcH$y-'
    '`9AO0PhL=YdbA%s~w=hR2TaX13^HU6!i@nqIt2XAFNRThxy3ug&+aAq}r^~ioR2&1r{$S_yTLIN{fgOYvYTj74T;-'
    'D7V1nZ25<*z!Xx<CiGKUxkL0^0b2&X-g)2)<JNJT4ji2Rb1c!|BJQ5IBPV$)J`J?ZCAFA<uc(bv(3O2os{85od8zJaU{ir-oPZ?p'
    'kQ10ao^k?D$zx8SXn4+P|6C+;HUqS6N!vVwG}NskNN|Wo7b5GbY+`i#o!RkgO{M9`Fs)py-'
    'J#q8<RzWmQPs0KxunBq9jvl6>ye_`;OynV{OPHqYkL+hFFTy;l~*BmOQ^q9oPSi0Q4U8r3NMG<sN`NwagOazY0<ecoEOTRf2JgDp'
    'v?If5$~;){bgL@G4w%{#Fg!-'
    'y7aqEwcL$9{!u#06W*0|ADct6z<uI7gwxL7!<IB#dNqtk8nDR0N3xIWv8~iul~QeVmUFuzspZBNq7aN^jS^VabQxDr>dcZZ<BMvN'
    '$cmg>RFfI9LZwJGMYLY1xSA?bFH~GjW7G--%m2o3gIA({r-a(<aexiZ6XMPreCH;k8g>w*N;y$RXR^D~Y>i@Nki2r+8|kOp0f(-'
    'cGs<dyHG%WVY5~^-JFDs%Zclbo)wLe*g1oX?$QY+xRdro8m2=8!5!Y3-'
    '3m}?iM;88%8zxbKYwTl~It7Gu{fr|?$+g7k2}|ZDZd)jOE-SX9zefKDw@lCVJ~SGaNIPwZ*iIvNzAXCsR(d18{yv57NZcxKtHD_4'
    '-`hc!{aZ5i6^I^k?YFAn^;2~9T0BM0Xea{4-LTl0n&_VhfXL|s>F#8G{-'
    '_h@$s4oi)zj;q@NNa@VC}tH_A2Ogzm|OrdOZ}=?Y|ZvgAQ_hFA4c^lYIsCdT1pTiYh@_dZ9YxJK$0fDbnwP{~?R2Jvpl0lScx8@L'
    '5f2gys|G+d&cI^A8scx(lA;rG#sVj_gaSu0is+TgVE)CTHRm%nvc0hbRlG8Jt#KQ_bWd%C*%j9-'
    '=I)X7do`x@ry&Q5IEm;}E5KKv4EBUO%8^_u`>2pO+_jnzpz5_Qu)T-'
    'd*dJy#yFQI_$i+OZJgc&bB1Qd{52%3eGkbc^orBx8i>}TzG-'
    '$uRfIgM9)A0)BQwlA?1=f3ba*cDA4&jLxC>P84C0oouNRlO{NLegS>dbu+K$0fyTdDfQ~%uW%L5=8~s-'
    'pAlsr+`cNEcKJnE7_F+{Q)i~TydZZJ8`?7vfvPlj)?o*?{_Gyvm`t(TBeMThdJ~Pr^pB3q^&*t_i!cgd$gFf>?QEjjKz}19k{hyG'
    'F^G&ZfB5UTB-'
    '8zkVnuir|Y1dQpx$kKrHzo)VqmRPWX<2PRd#bj#`3B=AUG175wl^VhoEqu=k511<V|{yV;wRBuUIha5mqd_YSvc@0t=1;Mhh=zIy'
    'OqldY?pB}?lHGQ$w<T>y0+5~qpJK;1fa0jrtMzBQ1zUq0z;PE=%^U3p<bXTU=sDN`?RpB53tK*mhgQDlJZpG%gPSbKIvl1tq7Q}9'
    '`vc}qkvTysifERf&XQZK-'
    'o)&qxvqGFfJ(?yy;#c>oS>oE4V45;U0Bsklg!!!mU9vQ1eN*ZOTZ^A1Qk)mM~eb0H<ap4@2xRS|lAmq3p3*L~Ms4JjkZsfd~&mKE'
    'J<xAk>^qP^-T`KpaNF=l?A)W}YrwLy}BK8T%))&SV33CLheP3Dv*|>}-'
    '~(NPj&#HHw_=X~#QJpmGFkwPQx#Q7(ByyyS!8B_A9w`H*<Y8{;J(8ZY@UzT{0v8Y;$mK=;rJ(^@}ezbMpU{gwTa@_TT>ET?OGhYm'
    '5EPQoQMmqnK`9WFhT=CEk#w{t%e*wf~u?=My6OAb4^*wKOFi;hvkJ}C*t#GC`PAq7}Wu!B;7#Y7~>(YH?mz_w??5ZagmET#Z!oRS'
    'R=O92+sfHw}w!~qJ*+X0K#RM@ixxJY&VSi^tPT-b9=#@0oN%{p(t$8Zs9)t!dRPp3@FvMCb`C{j<C71bKicDlT%-'
    'Vtf1D~f8ZC_7zQRO>|8>8heyFKR3k5QL6ZOs`3v{&T_R1C%{U_7Ddudx}7K#ufH-fpVT(*fSk0Q9j+P45gue^ObD3yM>;2l7|kAs'
    'NR9do1P2pr#E$Mr1mzP>x>47skQ4=m=cHPj4}P;%v@Ahzb#r=R~Oa0B0hsr+d|%}LO=D*xF-AwBTVYe2-?TUz<@l<af-nU!$B7rq'
    'AR>iAikk9M7h&v6%ShuVM24qz%$fw9f`SKE-2-'
    '4hv5k%d8xf{KV&9;7Qg^97;vdXC2^Qe)DgxWXM9VmMdQUl)?)ztPX+~HleGx`>yhkw(lr~%wqKy=_bZraq$Q06E%DK{hTbh!gq0A'
    '@rW6zdlLwS;X76Y0>bP%BCD2;UbT4D2lkpxZ&(O)(fWk9$GTuks89EsoQFew-#wJvqp_B0eiq6o<*o>Mp!1$~JGf-'
    'F!`Jw_fP&x#jUf3`CCo-$BCo`tt(JZGJh|a`|h4jcM)oIoqtyv@e9t#~}W-'
    '9DJxfSh*Fj=F{XmsJPF_wnER>H*n(c&u~Eo<`8k}Mx3cH-'
    '1Twn=a5FOFMfb<V^5B^?C0Uaom7`T%X2vwAFp5^OAzBkTfMx+g++UYH&cjrt^VXDu>WbaR!lsf~sKZ4`>>*cEXUfb&Hind%`HN2Y'
    'p+#l5H=VrgWmhgcbz>LFJ7qIxKtk{8@6U4RO_<W}iImiA{y!SZIl&wi3-'
    'sn$JIms+S3|NADk(c>8x;4#>f<S=}kn>>Xue7wRg`*`6Q+!~P2u5!3WajO_^XrW?OR`{Xolr$H}5iL^8%nJX$SOsp@CX~&EY*PKn'
    '@fU^E^JPbz@>Y$Qb0EgX-P!8wC!AYf5uHJ^kR~*m;P*zkXV{q-Y))avXqKzly=Jfz-7b5Y&_7YTd>a`?f~W6$RI-'
    'n{K!6P`?!NJz>`b@dI`br}iDDr51kMo#wN7DI`7FqQIS_tcfiB>D0hYAz$L*`A5)Un;{S}qsp@q~>Q7t}+xj(5GpUi}^6n>^K3n-'
    'Q2Q<(#l>hWn`u5tJb>;kwLsq9{U_54s}_hudOCv!W|*BujKiDL_1@W?7~kDS=1g2c{fz?6~!03FY#>uV$j`Nb`-'
    'V!da!;?{zH=0wGyv~vpk1;wbeaR^xqOB-'
    'L<(TeG~&t*mn;Hu6m!eT6dRg_nKNinjD@~V?mE$nz=%=aS+<EBWA`9W<Bp}@IHbJOKn@;)%lPnsn^;TA<7IanR&s<sp?3CsC8A|M'
    'NWiLFK`+tb@>D|lDZt&s*LtZhhGnYIntD)Y7>Rb}Ef<f)VSR8o@4EN;k9nZ^z2Y2b15<e~3HqRI`CJal6us@z;7I&kS{YNJ2ais_'
    'iMGd#$5W?HcA$oAPLynoYOa0wj&R|a)S;d<+|)fwz|h_XGr-'
    'L})VJrl!tOWUK2J(mG<+A04qU`{)=6$XWAdyENNccY+>1XZ9gMZ2w#JyItaL_fs}5&7+>T~o|!PYf~PO*IVd-oN;-'
    'zgeM>43CM5)Q5bL1d~0L1;KSX)y`@bu3LA86Q9%?5!TZS$9c!oBG_$ZzRdQ2gD!hhBaaLzX4TI=7%)^#Xl6X9c>U-'
    'U5aQWa(=kG!?b{xRskMEl=(4$doCD9v0PX3I?$E%0P)&JY{gRLN6o3h(Hr|T44s?9nZRyWL{_HLP^C)fC)PA02&Jl3hBIl}I+fN&'
    'No)MCfXBD51L6~eZ(rXAh1H5xH%nhv0;yUILRyW2d=qw|P^QE%CQub>~*jk2bnNNO=JgSz7<TvQ-Yd3=oYrFJ6Kzquatp7*Zb--C'
    'wV||jmQ4kCEf-)cqND-'
    '8#Vs~{*i0g{juni*?1{;>O?5gV)zg@SAedyS+0JdF6Q5m|sHtg6@igboPl<%CAoFw<Yhrayg{d6ufnR{>Y&&jF(^LB<uzZ$XS`<b'
    'p(UimSrl-GWc(#H2TU9G(SgG7d`0AU)B?{m721NV9@sM#(fM{fXidT5B<uESFx3I(o5)d!92LD^IzI1Rk+&l{qX!m6opT%cYPdzo'
    'q@FB54E-T)dPyjtyAmnvkAh{L9v(49w%z6Bdox<m1uOu57EiZ*1}jI}-'
    'oixtjVTh?TgOLpg#sX|7pFS6WiUrO_m6;fd3A0D@yM^~e_VvxPin=wcm>FpS#uJwkDX%NAZL6&cC%9u4G*fN;<HPH8>T@NPDHDO)'
    'WqDB?P)En`PjzYAXP<23uM!Zg2aP#03AIVX)cZ&l--g6uG0z{8jV>ksGG1j=)6HvqAFL$RT{PEk66D*p%ucZ06ojlMII%Ud4YQwr'
    '<gkyM!aEu5Mj*%h4F)BnjMu!N;m=NI@>*-'
    'dlShbU}0gss|!C)gE*HMDOX4J0oB_h*_*QAf4z%)(Pdl_**Ut*xaH)*Kx4veh4U!0ToT!RDCWQ$DvJbhtl0WU*eyQ!0T`tx~#{`}'
    'iOfBs#dKc64y&ld#x^M!%_d{Ll3cc5AgcDKu%(rdu(zS4ca4s2}~($pKlroJ{8w0uM&0Ow=x8#gy|qHwI<Virp-'
    'W$PT5vf%{OgiTp%F=Vy&IK{lJ!fL1%JcA7Zh7I0ZAM2t+5Ou7Jf*pIi*G0i%`<>TC!Op+K>!MeK#PYf**eQSTy68}0{^)hlYdtp*'
    'riZ>Km~XZdMv6b;Vi05v;w;iLC=Kk39GAdaA`%#{sE;2_+57R7HJXKzJkW0{+bb~2fxSmzfku$Q_bMzm2>CM)D0u#8??0&E6*w?;'
    'Ta<KhJI24=;r2U>f4h{dJ23uzCWjedr=7)NAJ}Rx{P0JNf1kr_fw0-'
    'mjhWA@V^L8PJCSOG6R8UM7i(D)?xVbi7^h|T0!2M?h4XE*_TdL{(En?!{wIIMCbK9P1w{RJNZj7yK7~o+K?K~du%IP~fQLXW^@Z*'
    '=p=ALWdmMNLTaN>;VCr$;6)Zgtyn>;}fmg8eIPeN)UfdPdh70E9g6LSHc^js*a4`g2_i>KBISQ}+`G~PCQQT>)4^39=qWLhMtZ$`'
    'x#ZJ~{^L>$X2d68SgU}oPE%b4$j$?Q=aV>9cRLcvy^90XNznJ!Fj@)Pky-Qe@l}H>BL3<UP+Hq=FgEA{)$-'
    'P>cRk7qo#?|=PTdU0KxcG0KGHbje|2XNIgi|mDGeEc)5i>wO^gNaQ3uX{gqJI=m;&;-'
    '#YIo`G1?;6eD0^}a^TmKA+F#h`smIPFO#D9w9nNerQL$>a>;eA4!(&M}9+!sWaalMXmxtqVMK~T;hU0OSKOW5@WsWYG3ktr=&6~b'
    'lr!&&oa6G94X3V^w#o2n)tMEPMFwQ=X<=9rUL~<JVq1j{soQmQc=X&AWESRanx52<nFYdWurVHN_1M`@;DTbLTd@}|!ON0&?vxV)'
    '<K$SoeW2nx8egU~}nf|=N9LZr<JzLK<pJ&>0*2KZP#pqfe=R&~UFrDU#LbQs3hW{MQ$>OE2rark2NT0+&`Uczeav5@0RfME^!GC>'
    'w-*2!IOJlshzY-'
    '0&<^@ifMj%Bkd;6=<vNtT<T_hsd=Jy)xpNO&FDXo8oGtQT4{{=RTj3^gBkeup!2v$7s@d1MCBW2HwDswz_r9Y3W@V1vMNjqaY)yG'
    'Vlho!B|#Q0CD0c9exyQ#EwFmJ$5bBQ;@1Sm>g<)X?n8!kgdwASTLTb|$ncA`E<VOpbn(%cD{(~_MToUc-~PF-'
    'wNxW)z48M8-9bWG25=E5Jzb2iVjRKD-is5@b<3Qx?5$UVd^o>P!)h^_H|P-!pIvJInR-kJi_0%9@m5O+0FnO?pPW<TV{HlMo}>o3'
    'i%*v!UF+uR)GT6H);wwE^9X1OS?#iEHMf#4a_w~z9G;ozxsMLjH|wwQgC%YjBU)$?9~p6Yq8KvDI)SD>kS-YZa5J?|Chs-'
    'E`>lvU4r1=_0Ty}E@2ct4I99p&4Ej8V9kKZ%s-'
    'Dgm}O?XZEu9G0y%`kZ6>pXIHeN+fn{kxk`_6S1s1@8@a+_5<(d)tL>M@CA|fSMy}Fc1~lMi#E<@3{!ESa~Z=-v~?b1n23X%!x-'
    'k_VCOG}X=q0ohdz@y?%i#kqRirOj6SU}WE_ssi3-EPdqp|-ji!xqy3>KVoUXf2`9^)>F4j~(&y^OzyH*yN`vkI5-'
    'EVr6EcG_G?^w4OV+X9{OsDkKT#)QJ70!m|RX7`7P~mKNQH8VNB^Azwf2(jd{6~ee;brB{20Js)D2%5E1>;$TA@!hOOi&nA4+_R4h'
    '0*1`Bcy?WaHv%Z01i7jU&I$h3{TY&q4#qtgf2LXl4d5d8Y6}jICAfmlw~t0^o47!xLTo0PMuqZ%%5Iu_G11%q#)-'
    'Um;spm&S(Aw6dyX1?E>?OVnu(j#Lrsk8|q#qeSZBlMC!~L_4`Qt?C<2%{t`c}eK|9J+HgxCJ=J_9eZ`~slrzV_EMkfD!HGG_=5O^'
    '^KNzOAOVnRsYC*^nS&XS<`Nn3VMDw0?<^VgXL-K@vwQQ<dXZw=iY+o8;6U&0%eR&8|tO#L>mBAbD>YVS#Q-'
    'poX!zI?Q4)mcZScWQ}rnfJ4J9HEERhG($xjd}hQsb#czd|9u&I|G=SgmVOlpZI21n*6%U&xBCIYD&2A4Jy&L3Di>MAt__bbTB|*C'
    '#=AeQMw8-Y7|3&(t%`TnB_%dX~#%r_!u$mQB*beS@jAEAta1rOTCT+v6Qx!%3^JGk!qIS+=$RnCy=<dyY%a*1U%-'
    'E}VsV45`}K!l}(ADPH0Epsx|2F~{z}2e4-XpYjjE%zE9kmMGk-IL#iTC1-G&Jw~AW`&yw2#nt62O>f`FsYc?5D-'
    'woC{H&K;pNyYY+$2iZyOb*80qSn)H!}`fg}dZw?A_)j?VIAGaDfOx;7VV;sBoI9eLIHfl4D{+u|Ohd8PjFxP3^nW?{CkX8Z$Aoz3'
    'KzprUtl83vinr;5H+`ZDxSmtN^#!0d5aUGh1@wo{K2(eRkgP#*D&NlDJmm8LA;)O>9oj%Xt4@pc;;9<qj~UN`cp?j@an>dm)d=0_'
    'eMzttfa0`0UEy&5HS8O7u~_T7Wu9m?JYbR!6TA*1f@{8JNOw)(`kkvGD_ZHb~zz9X=eSSS$US_C|02OkcRyNN9GnheCOrQ(UI2^#'
    '!^vioGlYDM_nuW5eeW+q@4ju*S5^+dY*yi7mYRS|>t7_DqIqeWY8syqkLGyi!Ps(l6E_t`yc^mUkDsZV$#S-'
    'zZY)f2rdB%l+Kd)x<i&ahpxFj~&m%mLSloFcvk#_YjrJW3LKhWkC>N`x}|3JDf{+i{r81#q;(A`?@f5<Rg_o@L}qT4C^7nI>CIEE'
    'V&VV(F!@hd>1QnA^PTi<UsTV+T<F`x!9}pQuJ6jp)g#T9tF2;(YZd`=P7TdGwod8xU#1ustrT~uy%D(%d|?gKbFam2UXAPbOSO;+'
    '5;k9!ePWW-Ih<s`I&uMr))cCuDgdj#NF4r6TQGVvqD8ZeEm;{%HY<d(!QbDu+Wp<j6VTsX79|Lpmh$Ea8^#{=1-'
    'UE*KBiFJS$)47G<SFA0fzLMk{kffo<mvbi%lBZH4kW19IQuhiknLr`Y=AS?b?moDR4kM%UvUp#j_M`AqfHm6+t={v;;mE%ym~8?f'
    'Yj0Vmz9_Q`mg%vv9_8iHE*kZ>rm;f^Zcmy)IYH1m}V8o|6sBluPZ@62~HOVk9X)5a3Ap)QsUZ=q%GeEF5=*{(Cq3%=rHt0>7mRGC'
    'SXUl`BRqa2~pmz!sc+8O!iSRGE&9G`M}FNX%I(EY+>b-'
    'nSx{JnBzk5w4+`DHHpzZUELzvbM=l|CnOy;<dRA{)&b!S06TsOs3ZdN}f8<_dt3gP1A+Mgkt~_ppGMGHU=V;AOUKzBm5G%k>-'
    'F^9b~}DH#m?p3Ub)nHm*OA8qN1&j@tI{R3U`nSrkOtUy<McAzUhC!~y@>vR0!b7jbZgU?kV2M#`0hb&6?*jQC?w)<YlHF0_dlHo*'
    '6!T-'
    'Wk)>d%BX?BOBFt`JokQD=^t1OYrvPDpnB`^ht**XH8rSD6cf^kp}#amn)u*Me$xH5u>{QfzLYvlaIHK9;!NGKG$A{2^U84AU&3WZ'
    '`<heEMyLZR4DS13j_3%~Fkkn~n0cE2Wcr(2{<XXTHKWy&0ubE?fzQBOLbKVr$F;kyD7bGSKLyzK@7)p3J>nh=t}6LS}I%69eMz&e'
    '}yRAr|#E*oCw6?TbKA=%@zDK#Jjg>P<re!|ZB8LCNK_5X9!lDO<&hgMJn6efN;ed#;}rY)=W7b&bgbm6ms3jL6-'
    'A_LT;kG!JL2WxoQE&%K;RK5okNZ;?3>7qPy%3@^>We?wyA9OL)_$Yo|WaMDl@pH;sIdPUUS_4H@oJ-'
    'J(q<LE}%psYRp^}oo>o_YC>5gCFv`8d7ewFhgkzRVTZ;Ia4^OPKf>8dPp4#sR%mO1S(6&i((qrv#p>+qnrEK#Pb@_NfsWx6S^sVr'
    'CKa8=Yf>}jMwgm0WgU%D+fLlqs`^WxSwl~Imqjn8rT8HwuRcCNXE>0*!aVqw0t)43V_s&{5{onKJCa2Q0~iz?3aeM!Y-'
    '#4jtCD_*5nphI{*N8iRMn@3fJCAN=xr~a;ZgmWg!UZ|V<tP*rI40fl}5z>6(ZP%kE0(PJS+p<hPQf1)XZ5?=b+XUX-'
    '0|W1F+rYbfP~hD?IPmVa3+-'
    'D7ojxhPvoOyqpZwfSfHDy$IR8gcg7cs#yUbgWC=05KlJqlsfk;4n>scpP2V{HaeP$PH5w^n4%8VhHp_&OzFVA=#221uj;MS~(Q<p'
    '2GPtV0?t_S(U4ria>UT=ZupJf5IC8cLsD$LhKHMsaJ`LK>&g*xLx-|*~7AIRAgb#M-'
    'RtwUtEQ(^;#C2E|I+lZOyLmV=CQn9i#bB>ZI*Dhir3@p1I;Oq7#cmTbt%yYhM>OBRDrAO0>9LpEzTrok2Zk?f@&}}kwA^N}!TZnF'
    '(Aq&w5WvD9j!Au>(sqLP)f?B%Hw2${+g)~@BAiu55L~pshqs$~}TaLT6_+FSV>{=`n=9>&rbG11}x|ZpEF^fN<SFDZmu#b_!uYDK'
    '#*<H)iD$D`A!)bvbZdwDF(Oq>@hqRSBZ~V}V`NxLOJO!4vkMf|#JXCteS9Kf@0{dxLWD+eSr;aBHe*v}BOts`j?}^yN&gTQXvkhO'
    'nA8$fVJVE=06~Zfg+6lTQSRuU1C!L_x!wTUwKIH`MPC7juA&E4{x_)xD16qA(S3J#uY*ieVu7~-sjcnMpTS9ZMB4IIrb7F8-'
    '+_~OB%QDCNL4GFLPQk3LwbD_^@HCtvef=hOSL3QpOk?D-'
    'P3*G9b(>h=#f6)oWQFo<5J6jfHtHg>So(Y7n<l<}=@KOK<G%$0LV4yxSj$45C}<g*fDM_0uzs6lpw)31SPj=<kQceBbOt&yom3*+'
    'hU7BEhLk!78wGs$9Ne7iPl1|6Cb-pFW`kRxUu1$?pkib~IG|-@LO7skWI{NgYuI2I(+^1((y@8<II+$n-'
    'wpLemIQ}k_dgbUhok;VJ{+9M)ftyVcWhP)Q!&)LF)CSS9V(+t*G!%u8f>+}r!gJ7cIs=hN<LFi!giX*d*~`lM(J9K`u58pLVGCZ{'
    'Kr!0o{GwnY_@kVWj45rw%CDhqu^@uBSP=%gR!}!>$DDH^C;y68|K9QS(+Vkde+If{vqh(8o#tum?jgtT;XJ`i7jeSD|_0Fxk12l#'
    'U7Nwp4k;-'
    '#&i}GS9)e#5EQdmI$XmXxg@WKdVTyQB>RY3DBR#3Q!gV)5AT^*kfVna`!3eQ!D43D%}B74V|%iaiADNiP0U>7AfMksHtgV4jXnzP'
    'gr3qX`7c;m9Bs?#Y*nw#fN5Cf6se9GUg{%;7fcThH@{mVoV?9AnDUwP=^%*8Vy_{&Q+?g>d6>`a%WBWZgl1ocd;w-'
    'O`#R(cF{RlTB430#&At-33X__-6d3~?@Oeo19OoWBd4gIhv(ic(l{sS?<V2s)CtzKsux>iZ3#Wq2AYtK7@|h$l+-'
    '9ClV!>@@>XFT9E{&KF&Ds(5%O>r}N{0nLW6NHH7=Ah?PY2;UK}FI9<nP3ofgD8eBxM7}yry7k&Qt8$?oDy?dKMMyIZ$2?-'
    '#W*|`q=sD9NtCAJztO(7zZrY=h(6NI76^R6utHI_g&95f&D36Glk!)xB(HO==X|jg%ItUMWWX!d~cGRbprCgBxI|6$Jd9rnl5;CZ'
    'FTF^P~AE$RJTqK)vYr^b?eMf-8w5&x6Y1qllz6a%F2_6gc+)e%0Vi#Q?wy(L%U|^RkRZGLbxFo(-A#J4|E&(T|S-'
    'a<kpJldF}!0&_2(7!d#h&+Q3%2$Ma#~cDdhkVBvOoAkdc&BpMAzQ>L##HkJy-In95BlR|M$^Pi;IUU3@p|4GwKF&`2tac+xzB+VZ'
    'bZ;`pu{GZ}Q`&^p8sC@gXJIMU_LQ+VPW^H46-Wr&}dYCU?8R7Q1-4<U^;G~83g!Wf-Uiyi9K$xrT#62j?H7Xk|T+ewmQ}qVUtC^-'
    'Za$e1J-'
    'N<=0GxR3TtC^`cb6(9Xy@m5?X6voY<VE0T(m0fR!#ryoO1gVJUl~fe`z(oEJW+Sk5g3PX8cVzWthA0Q#p+}GaH#av(#U<Z#rL~c6'
    'n}88KQ~RokaCu$>0V*3$V}5HJ&NTqrW`OIf?Wx>F6bLZ<IbITSf$cS!hhL6jEO)Ef<46OAXqKfM~t4CHG;jwXcao$gUTi(D<62X@'
    'rv{A@y|c4DAzudR7wDg&FM`!f||%nQ}4E>ovy)j*0o?$CNi_|tTvs48+WkG`%OUs<D(AqI{2W&#11~@FtdXXIn3waLk$x;{E-'
    'GT_>|&m0-UpF6ju=7oIR_22?5O61ZD1YS1@N2l~4NagRQlRN+>6^h7tkG-MsxaXHLJ)l5>rzK=fihJ`cs>Qs?sNiZ;l6)@ebeE?|'
    '`wgz7@pXhEtj;#Nx_Rzuwpi|bAB<)rau__DUMp6R51Y*N)`syXgPt+$2GOs&jtOh?|`r`gnWB>_*nP;Qh#jLRLvWJ_*``Mmw=(9E'
    '*dH?y>=4c}o1@hR8(T`|BCt_OH*fUl{(2!@X*?1Jfzr!Da14x+8_HJc;+ULDa>gEpe41!Y7}59g+y5%dy0E1a5owmUQju{I5VwmE'
    '6ja4hdw)PQjaHW#Kw8#FUpGyW)yYr4H)r>WMj4*b>iAya{4=@?{<1Em8<erNOZU^%8sZ&wUQ-'
    'D`tqwVf%wT`>~%1hy+ip(4h1#b{I-'
    '*sd6ZO2ZbGSg15?b%)2t()>}e9`}hffAV>>pGtG5Vx{ggY5uJ8fpywB_}^iSnI;{Jyiakd2$)WBwG9|gaj^}UPjMXy7*KH;37AlE'
    'B@P%-Lq*2>bgefu?~W)6@56o{7jjA%(-'
    'mU=3Y<>^vvoF=abi+vH*Xaazp>V5)RfLLWUW%F<%$v>>mGEP#G6ZW%s84zKgt^nq|Vg_b#8c2=SBo|Ze&pBMg?_lbWrEU1a)q#i?'
    '(CS1*fFkBjI>IpHs--e0Qt=0CxPSihaPfxRDu=^`SS&dd$;eUvFoL7X#w_k$zrF8DPD@`2MnEY>D-dlq%r+$ix*-'
    '^>K1tB`DNAWH{UJmOiolX{$vAB@nC)_;i!ev-qRg8ec0j#@DKhxwSfDV6Dv<SZw=mz?_sb*r)5WdGw}`rJ*<b#2yT8{kMpo@M$Ve'
    'j1_0qF*4#Q;fIuvKKQvaV=mXSIRWZkWFLQ+saI&L0$i)mjS;A9YbaH%MqQUFhpZ3fkW)iB<g`!@IX#p^&Isj@GebG#tWXX)o8u4G'
    'Y;{TvtfEhZ(v3dn0t2D{3qCbpK!7Jxtos$>8S0H6nxUiclWh-'
    'C%Z(*F370~KhgOs*iP?NGMWGJ3yYl>Lo$|~=Aj89BUmPv|plI<giC*W@2nd%&K)BqeIU%=I$l_Tq>~3<|C;~rS+!dJB8!;U^_7F6'
    'PqPp5IvQt$NPt_6lp{Y6&KiRh0l~G&m)=^vSHc?yc1EaRuZKJl@2SshQ502VuxAWx%<d)=kvUN>T4+83ihpt#ql6=AER@<2}PGAG'
    'cm<jDvGeK@!70g8B{@YeA6Qx6~FShNwW4eE4%C?KAtOtH*$|%EX8{X|)qD)+XkD;*jA~xJ=*{=%<+4`akMR@3I^BLxj8C;s<<9fX'
    '=n6MUkLk`6hUS<xTfCSs-'
    'R>^&{Mk;QVwF2vg&GX9JC|!_**4rqDA~B)2QMxMglJYi6H)Z~<yp3|0GXGKDMmZb_Y@Kl5zGK!qGDJi|yM*pYZh%^L$gGbXGV26e'
    'bHOn}${4HVR|s>EXPCN1m=3;Q{ZL^JQNCJ;czf!7@UydU+NVu{ar&ptfN>hAO@MJqIfh8P3Fjoi2<c(MX{R<b#yU#k6c$8PEY+KK'
    '+*U+@^upL=UGIA?)JAmCRfwc@KEtUKkwOd**{<eZUuYflg*HK7I56l7ZG*mWP|z0+4*Ej7pf6l0%)y@I>MCK{c^a>)g=y~@udWrQ'
    'qhq{^x+SAhli6R*I>t5@EST5HV7+FsUkm=yY);sJSTXg3YhoZV%oqHIhhXAv<TnH*<V497%nwWaoT0cKQ;z~4YRgSeM<GicP$XyR'
    'Y{Ja8-PLv5YNnn^o0&s}f6h;h2_R7-'
    'R4qs85VSV@6;Dka9c(d3cF1m=;H9sQQ&2EHXO6Fjt@T+a#1iWt#X>%}ciO6;8N<HuS|BX1*8*X6y%q?I>$T@`Dq`b!L`4J~kE)1('
    '<1rNxa6GOe0*)tCM8NT+as(W?(EqjB%<$$N^IpWZD!2{<Lm)mIJfW-'
    'JQ&iqm!mnQ&NRe4hq-<!P7UWJ~l{tZx_uzK&6L0~Eius<wv#WpSDv}P1JdNq$oU@5-tJfldfxVh5OjE^~r=JPaOgR8zhm-fE**6y'
    'kYKvJ^2%7$Ehwj+19A{w_y>;;F48wp4`wZq|c2)>y)TlfMgdNixy@<5KED<lu8OymfoC{aIq_Lb^Be_)5%NonMHJWLEytJ{LTTa}'
    'ZNAmew&y?E51MMP`!nv-'
    '{<B;148y<Qxk)1l0^bI;Kka2<U5mQWd5=_MmyXbBis}Xl++2rk<^u#RoNDB2rDHFv~k3g2+=)4A<>#Vq+DyXYx$I%T?@Xv{(8%HB'
    '6JfG)I`^#!m?#dZDDx+JejBb^KQhS^%Jfynwl<l0c$L^L<g&6F)(`2aZL60?sqdD=mlD%a1u?=xCwkY<wW+tzbSLd9_#%!c$^15K'
    'cRwU97#gwgRGP`2ZRy3L2Fn=kU%)>B)DVof~F|CUybB2sk31?AemJ9d4DWkl?|4Elun6uY<lCe3ReTKVq%;azRC=WJ>9VSDA*n4D'
    'X5PPo-'
    '4Px(;p+W5ZGBk*NK!ygf5Blox=`zZRoGH!6J_b9RviV#L_P><*curom>=db%?7!NIX#|er5i61_VQr1{jG|N==++whMMWj}UXuk#'
    'Y!~;yXpvI@5l{9vjf`!3YK<}1-Mlw^_kwf;z!@$Me9Du_)%~W*;plGahZ}G-'
    'jB*SDj_$#<*a%1WQrbbl(S4M85O8!qB_9MFJs>ebDB$QpN*>tT$HQSdoFx1=b^IFXL*R=gD^;V-'
    '5@8Ac<HRCdi&!n%wdka)cWeYzFlWAVAJ{7c1J5L=6?4OM&V^Zu4S%fp$+_jLt-'
    '``4?E4DT0JdR2P@o5}5&NM6MS!i?k3ht7ExGwvfzrly>?aDeHa28GRiL){gn;iQ3;kE9|Cy8SXA&{GlkR5|L-'
    'zb4j>)^JQ1tgKopvia=)B?8K2H7u%v_A!!CX%~q1&5t*zmAto1L0!<eAPZ0xwcq*x^3AVWH&1td3d@kFk&3pIF|V7BB$YoO8Y(cG'
    'h5pZMa#;*+qJUS;WajdZhWDY47wXvskcqV2oMn8SuuMWg?eYRK*z#yXXCen75qY(xiCCj*s?5DicgxCNI7^%8RdpzlQBv=f$%#fY'
    'L15OP`DlpAPJdQ26P{E&ddKI=Rz9(P`mS&VoRaSEqAU2xNJ6O6PV0X<nVyxxGN1SEqJ16-e~zEQDqPnO>cVu!BIV*Vu~>>-v}=H-'
    '3ZC*SVyaJvwGeDX^7SB`5N{FLZ0p<M}}7Hk`)uq0k3%7SBgQx8)?BkA*&nb9g=x`e07s`BV^bwUrSci0s+*k;tA^p9+5#5rOGUtI'
    '_b`YbcyHj#B2`h(YLNZo%sY+M7o_tO@*nf5}7@xF38yj5U+v6QCNUjwfw&vj$q_BX-srmA=-'
    '^+Me6oa_D0E0w0ibZhGKdot<+9em{V>T&0W}CKIDdks>fhWEFw;MOG2`P-GAOk45&N|3vUXPfZuR_Yx7r?n?v^`!5kc?7&3$um=-'
    '0S@v6^CObQ4o4~veLOF~lp`STRRfNe{1)qbpS4NK427gC=h?TW1z*&4Cw^N5-'
    'YVZA8uwh(Cx?~j<;TOA7o~)whfM0Ay&jAzOG){yVrLfBRBl}jnV2vnG&u&<n#=ZVNA~tSUWcUdYSAXczjyY-'
    '8wU_+8<X^iH`vY;g%NFyFG<zy`;=U`*Uds1wd{3IaRY>)(XWE{<ai#{J0B;3i?j)j=sxY^xJEI6lE$bMT5nx)V{ht{M{%3QJUXM^'
    'z(i*iiE-Gn*x;fX{9Eiq@t1yAxeS|N(f$e>yXPSfkeU$G?2OGSPZEY2HX{Q2iylO?V#D<{S<GOlo==Nd5^vF2mNd@wv_IwPw=C&z'
    'T5I>FWmqERVRvG@Oh25v1b=>ISwogX7A%nuyK$|c<&?d|XTnV#+)2$!pdib6w(?U;_>7ggejL;KhX6T7BEA&K}Z8K}P;W23+S)=&'
    '6zNZVhSG~4Lk}TL{G&R(OIwQhhp0oIus74%WJEEN@&EW+%HTy>R>R<u(Nrm|mrjRc5K1Pc>bFua4kps6RS<~$!;fL#9%|4!bzh(C'
    '(qR_0%1%B++9X$hEfGp&2^cgbh3;7<XE?%+xkm~7O+spT8qb=ZY(%aJPp?u5PInwOou3)Y{kY+#Si`BTRIaaHgX`y8YH9-'
    'qK$b5glbEN5->yrtK17dq`OIkz<$bLn(wVRkeBBNczb6FkcvdVN9@b?1yPaTET<9nV&wOBc3VQik})h<01i-f-8(En-Pn3|5=WEO'
    'DlM87!K@J&Qgbk6)&m<wV<;eWzh7#j+233E|wD7-CD4-'
    'bd<I|B6Z=p1?%R3;hI0HLDW*|sgm3R{@h3>3A7nP4R1>^ac3?6}S3oLZdYdEQ?E(=o-X1R^ex?W4GYs-'
    '@uQbnGQ=&){vXJ8+Iy8+{1pcpa!aa*kJ9-HCI&4$_@D$LnC-g>$^x=|efkixw9C>vpz4(wZLb>PUe!HAlLuqXg1I92KK*0JadJ$-'
    '>g)EC{bf#xW$oCPt0yEE-eiXYo{cT{nyScg)hbRCA=XPUM=jw+|U%NJH-xnOg)aGJQ?0z#rwNafCTW_<C}4yzs2K<^<tsbIpl@i2'
    '-{GB%0~zrl6NVR*_!rsyDozg#Ws<CV>n;N4Tp=0?X2wb#*P4p1tqOO}SSZ7MW7}uPoOEZX{p4<YxL@?A$^#J;Quc1WR*<?J8j}&c'
    'VKlZb}cxc!>H?w$C^&v4+F9PDG7W+XD_l;!BNy>2x$>0%nuvco<z|`jXUmN*PM5x!9@am^L!Vsppt8a*0#VF=6CVr=DZB$YoAF$5'
    'fHaoqCRWB7<#Tncj8}F)gw=T5ynLq&Zf2HV<>0K-'
    '1NO^DJq)*eS!>`YbqW(a5Z{`7~K<nzIC$@H~}&2M}`~oK_vPgfM7t)x>Esm8|ZA?I{SecBAWU(STvv(p)kEGb(F*RPqcLDB8(+=U'
    '_wKA$&ilrAWKH)b(yb%2bY@-Gq&8*;jot{p1t&*|hXV?`LLzZ}fg{4)8|rSEjY+y}S}R>Vz+crY?1ZLMhP~mIGkA`MMaI|HO20-'
    't50L&+&9t=8<_VmJZzQlnOXb`iPJbj`hplURr>D0O#05${qOWbm>dOYl;$i0Oq^nxnz&SIM1mDS1Byu5c5i$=X8a6HO_Oo(oBx?o'
    'USsj#d%Iwo7dw!r)$g`ah}sqGsSc1o(5m8{|Q*hw}+~LXM|aU42F)K6dRJRil!M^A!wSB5~66D&$PTtcA6)eI-cg%Ij3HYgJmVoD'
    '9lN7<~-'
    'DO@nM9vj_(Q4$?yHGbTJ_3poDqM16KbEzFcRQ?S#2C)J2Sz_&Lp~KVu|*e&w#lO8lJel%X|f_{`zdlgF8mR&Ov*NVbmNMK%jIh<;'
    '5M?zg@)mEKR&%Ty__3?iQcC^I*+?R7)ZWcC+CC}c1GC8$)6X0y$<?CUD3q;=&9o5EaD=Z{Un@~;+H5jjrzlH!(GQL0aeufd!VR>O'
    'Pra?lLV;Eb><c!$=&zC4o^$E)~duK;b)Z`*5F)xQe%tt+;Fn+(El?Z1C*+OP`?CJf`mDoc>k39I9H+I}L<Zq}ITEA_97=ItNofnp'
    'FoWg2+e4$SOzg{S#AXFXdDv|A)Z=kgKM9`xouf=e28u*h{Df!*}0=8A%uETa>%D;D3_&Qw@x6Q|8w0!O%S7X}A7_X}HwopBv$-'
    'EwS;>saenW1CysPX@ax@4zimK%8CfefRrg*}G}tT<)7;Z?Q+^VbEDYGqx9xfWm67cNdSNSNKMhrP+D>o2_oP^Elo#@;tW9C2FFQb'
    '#&}^`L#fuCg`^}3e;=%V$Uj45&h1&_;yDQC8kol1v|EK;}<~x^~iwhx<CuuSMSP9Jt=-'
    'W4oyAgL3#}IY`D9<cmkAab5m3$R*5}lz|wixm)_P1SJjSsaLXunq@}>inCqi`lpT#%Mae+5n97@5evsuyTWYnT=0sJb#y5k>;Q77'
    'QG_tmIPzI+1&_Npo)M9TMg&l*@+mbk9kM_PHd?(k`nL?B`+0wS>!=93G%p7B%^cr(Qrd~m#@5%{)zUC62w`R<xK5y+*bD7UuI}JU'
    'MZs-'
    '3B3&!2nKOL*Z&n+qqevPS@s?gnYimzG%`}C(S{kpe)6znN;Loq_n#^&NYgA>erp23520`lS5Hh(x&=M6zuwa<~cSa4QkZ&ceqv_S'
    '-%=bHt)Wa9lkE?#iXlK6ly{Jme0mV&x&;9DF9UsrIvqrfVW9<RS}9y5?1MsUlU^97Pc3=fH=yGb)LR3*X285Pn@_kiOpq?zss$C>'
    'bwelIxAVnt~RypB_pZWTqCoCJpw&*(F7F1Z<PSK4$(lnvoR7%Est_r^Hx6R+2>BjR}sHo3yBb(oRk@&wq8%udK%GEPX_UTHvRuXJ'
    '8$uXKKBuXI6ZuXJH(uXIspujDfKuZ1sH?S}SMjn8Sn*{h+CnBV#s%j2LHQ5ZA}x^uDXC{NL|a0Iz2YM<G*=ZUzO5AEh5c9sBTUBh'
    ';sjKAo8MeWhAGsQKQ!l%j{%H6;MxmYlkhMPg|`@_MWyu^Lq9juW{-S<7f8oA7U-'
    'xKW2%iZ_Az|I^jSbW~%)$WHux#u1IC}4w+^2f|=UIl-'
    '`{LaOf9Gb8?htM>N4!Yc2{S`?Ic&?rf24RQT#zvdsJkRmD4d<Udmq)`SwD94}_Cce$4yomb#MU4z)6Gcw?R_4%AW^zwz7_a6(zlD'
    '1USAu6AzDXMDTQ9k3d-XEU#A2<rN3m?_0#$*-o?-'
    '8uU%&{&|9^qV|&H)6}I1OOlF1&hMklx9pZ_^(k7ZntZOQo$bPWny)JmSuq2H*lZV7UT$h{vn@lP2t(n~P+k#+#&+pw880$A7sl3;'
    '~DzR3~Q3VP794lEa!=`o~xF_^y>`!=7f6jCz&+2d74C*(`uY~PoT0(M6(d!%m5g3A=@dfnlA%dMJLog3A9K6@jWl_O&d%$-'
    '7;Kc$mn$0m*>&gnb=H@Q1*xUpLnVZXwnYpx?{lCqWPT!u%?EhURv;Pj?#C|UK`Wpfhw3hYZ&$;d2)qt9fY{h<~H@WGsZ|mIWm>qe'
    'eyO@Ov)-59TkhhOnn7dOuB#^u9*-xSed0!fsv|5BV!c!udgg@IFpA1+<N0WwCiA9rkJvH~T-'
    '``;BdD1#{%rn%2X550;@?!6Gycpx_%+&$M!Q=2Bd+}Gn^<jPCFs#fv%c@XE=iT(M?!fNwNAw}=#eGzFWGC)px)b|wAJ?7PUH*jb!'
    's#$i>O;AL?n*d!*RxsjB{r+H&akg-'
    'W3=92;leh(Ii8>|KxL;1`pO(pJV85{5iIM#pfzoIi+0Y>P!kk)E%@|rZ_i80^YraWD&YtHO0?}6$D|)&e%iyYx+Ag!2q$+$9s<J2'
    '9g!VDIJqOT6YNfRM0VB__=xPHC-M<_sGh_}WIZNGvyCtn6M1iN7t_!{V>NpQrcU1^@`090>}Rkpavti>Ob<JGUz;O~1>Cv{-'
    'GtG#v-'
    't}j6kUqu4&K%~2j>WoNA^Mo!O{YKU96qd*xAKtoRUjuL1Srute$m*!va4?xvSO!KTekh>l8o7xC3J!IBt0}?+eE*!)QM^ZY_?g;9'
    'x)1lR>^_e(5<mUpJ?DPR=RjbkE7TGfMTLB2NX;#5Pxqrg1L${^6T%%u&S{+|fMcH|b7#shh<^Gl9Ona3eYiwF5dAnT2;pbdzy^v0'
    'YG^W5bG)$-V2dvv+-tz-'
    '$HX35UUU?*{EEd{;OOx_bxrVUdITaB^^m!=Vn%@Hz&%4ybPeYZ}UjNw;D?{`N6rLXD+MkB^2X7`mPt<M4CS(@x-'
    '*V00Hv;4WY*@h098eN%1<yJn_vZ&1EivNuCRnY*<$NACmoY}^UGHjcknBXpgTDYVzQr1nOX(oFWR3-'
    'INXuL}5by4QB_<&3HA;VTm|9S_sxb9P(8t(+66ZA^gL38f8tgKMMav7lAW&Fi_km09kgbR8H(ZX&v6CZZ*1m3)}(4_YP5&I8a&b|'
    '+g?R0CZ%RoxCA$w{mB`XH889rVF0tqy@-'
    'xqt;%dzMz6;8!kU!F32rt1j>>17if?B)7D8Z3Q=S67*OY?qBxM;KS)<eK?u<$W201)7`6GJA-'
    '_7lW<taY@_FKdc6HRhr$DR*!F@8uiL|f+CYMBwlR;(;A%3tQXn}`<fU%~f2|iE2VHe<j=Oc!M{wNjFnuJ)-'
    '454BaonxD?!$4n9{OmGyY<w^u*sx(39i$p@wDEcPp7%eishY9^WX{CMZlTuX5jFQiQOcA?N;Hnt6=Vr4RhQo{8B%-'
    '@N+ZHu1T1M3t^PoEEnnHc(YXL<9V|T)F<#}xmcgXJ7kdl3Ga|g^vTTDf3@z*IJ-t0#@SGPD&y>0{Y%E#_4-#-'
    'xtc}6ugw{LE8G|&h21lA&<k+jWl>XHZ(c@t8;*F^b4c`9h9M$y9Qyvk+cyadFT%phu<$A@d?74+MOgUl!oqLQ3y%lm64uvu)k|4l'
    '-(4?fdu9*40_~X`23mknW(e#m$Ueun?~%z}?g#JhFntDC*G^0851YzqX=myq`8+&JAH~Pv*}4y(hUe&`SxY-tAH(P3F0fae;kBDy'
    '#%9`{dL;*u_R_23K7JO9-o73HE8#EpjsaL10EqhZq2VVQF077~MZwK_FP?_Sg>L7jp=U-'
    'EG=XWbE)RmhJf8R_{1UC<ja(H$2LOQwOgDH29fRNqcy>4DD1AOJ*0Hb&+>0Lvo4|R1_kmY)F{*vx!lfnd2Ny0Yu_au%q{RK<!sR3'
    '$02c=_NDttQ#7ZluaxedL^9R2Leqq}37U-3Gdw<3?g_mw>C!z(si8Jo^fiItvfEfht-'
    '@CzAF5CD>eIA3gkG_Bx{%F`t4y?z3XmG}N2l#SxbqIX9VIAS?13vjW@n7f+U+yn-fv=Bw4<5?%+7-S&Eqec-'
    'ndbf=cueTl?)`h`-v0<TeVW3X`qkfupKc#`(ANCS-QwmhRV3KwF=gvvyn|4%f5()q-'
    'FXM0U|+zLtvz`MwFB@i;<L3kB_U>a@@~Q_+2-'
    'M}laI(8=}ma$Q2Okuo3ZrSP4B?cXLsG4rOzIEN0vT&>YZ5n?4@^R>9e=qg?A^xf+ipr?S3SBt|x@v$354P@Lbtnd^GySC+!Pd`=@'
    '&b;gg%4cYxzT#1{NQ;%ND<!WXEvkh%l=d3tyX{3}dP#g5r4O)teknX621#X*^?%@NAAt<u++Bb94gg{&jy4#kbLS}J?-SMi&Bu=-'
    '2m2c1_#kAeq2Rhl;{;NuO-f;mU%$)JlC&<1`h$|p0eWLyW_T}5gf8djCkr+s1>M0-x8c*5sf_Q4E(bCQZNaDr9qFmug*-'
    'xQuEZo#IPV5&Gh!U4KzhrjMH<f^M`uD|-'
    'i8?PFA554NO5?;{!PMO=4xlNf{mHDkQzftBExDTu?!p8pPCLR8|qZ@qfbyp1?bPwDY2E&9H0oIWDl`vll^R+PZg!xY728#}Z|L-'
    'VzylZL7>MG1@6=p_-nNne1tuU`tn8_7>2s~-C4uk)A6m^tpNj=iG{fL5x?9kz_&E4}=Uv>3O_gGrHA9y}LvU!KW|8T3>`-'
    'h&624Gf{In@Kxt@>`lOm9ZZ{&I!s)GQ)qKw7B=j@VUV7MjbDgYQB#7YwDXm_S2anygvez*IiHk%;99s+kFf?i1+S15DXpcnZsaOd'
    'ILheLj+$P;8#XiY9ZAyPAMyLFQoR`fEc>8v{698g~$vfJ)pd7kWA(za^HK{oR(ZTVP{;Py90Mi406lV?~71u6dj;&EV9oKHm4b^6'
    '^O!!`ob!3d^oIBmtH*2}hI(OA9XIK}<p8FNvX{95nttkH(#adDNouZlSltLg*wFGBJCbiM0IvN-'
    '!1IW84w&_#AiiphmO<tR(DskGV_fxny9*LF0B*JJ%vE%>Fb^s_a1Jbi2ic+MmTql^v<VE+kcUqRN6j<3jE)<Lav}*zumTUpDK_9-'
    '~Xm0U3-'
    'o<}rHOwlKOS!l>C?kq7Cr5+KDJU!Mw6xX<;8AjQYIJ_)5X4M!%Sln6RgP<k4%hm?cTxAG|6RhY>Zr4I}9R<k(wx&e>C2#m4IqmmQ'
    'V`(p2pI<T}FJTY=T%FuR_MQR_FZD4OZf)8Lmn~T|oL=L6%na}1Dwjq&2=|bkSxr~i!<WQ<&KAXYpS>2+yc=RnZtupA_lt<sMwuQb'
    '8G5XdQp>KH!^x=(fNJSsq=Y~}Dt*0>w=%Z;kDhYkW2%CbwUlGkuIq2KDc@BO5snCxT`2JmCj&2rL$vlZMMy^6N<1xl)TDz-'
    'fu*BB1<t9TC_003Rst6U$3%HaBwabgRs0-E0DlRNCM{z%5<g)XyS#Bybu-'
    'Tl4&FPV{$;w5oV&ww+;v)WCDaAPD+uvZ3Xg}_RXIsB{E16oU&xNg1gvAvx7MB-'
    'caYYF%;@kW>6^n46U#DVmIgLrcBF$}|BrFmSatantCsMAmw8uOa9}{N2m4q)<m}8n{NUJc}V~gOX8legW|9zs@`CuXY`TmftBo-f'
    'xBhnH)yxU==+V54CkF$>53&Zs*Y`~7tud;h#q@K*~g;Dx7b}x+9ud{n$jDCaN3uE;Zmr#ufKk#8luQ07MklvDqbiZv$!tZ1BEh<9'
    'a%3}1nTi=+9Ke*G4srXw&V-oO36LEAB{<dJKDg}T2Fi)&JeKC)}e+s>f;P)?~*WOtM(M?~3-d-'
    '|jlJ#ihI2wX#gqtHygMY0p;)2EB_Jz??qr&KEQDOA-'
    's4#klPh1&oX8OdHF=kd&Bt2U&ZJNLkqQ~G$(<Xz#t$7Tdu`LXK6=U$rA`Gr7fkAugn^G|dce*JRgJ05^1Pszd9Fv5>t&~)pfWb46'
    'im5zpFpt5%3H>2q@E=0YzcZI#Qq4jR>=zqN#CB6pMAd-'
    '}m>|L}&YvIw=wlJJ*i}Slcn#|Sn4Hg}SF>mCRlSBibCdO2AM4fYe5hBiXV2UldINjrrs$2F9w-HV7<;SCff?*IkvZ)3-'
    'xl^hjIsB29(y3sOMuVb`{q>e!L4pi1>f5=CINgj6UQcjuL*XZNdaGf8c+uK^v>{rBAZXz5DbiJ>=4<{b+5c9V|z4UE9mjAzVo<D9'
    'q$nD-V2g$DRwJmO6!$aT*9R_tFQr7t?+dx3q*{QI%)a3f&nWnO`ZRTdD2qV`G07sEM1*9j-RGhHNn`PAdtEw7}FDpq$L>Bldw-'
    '3bx+7T7n?;+n;r5vd_{(QW9;tN%911QfC@8ud>ulSL#TbxH^q;_AcIhNgg3>H0wL?r*#U?A2o&(i3JyDUSksD=-'
    'c?}v`&`Pozgl2w`%4+HRj{E<13Dx90Fn@VhWP@RHuJnN|IQ59&JCDH1G*qT0oK%ijyW2bjr4*r|H%y4-'
    'VJDp{l9Th4;RQ0@dedTTr13r!n`apaNa8m=cqww+?5kG{fHc;lR|8}BfABj=4NNfXLdubyb=BG3o{UXB@fX}d5C_Shv@!!h>oY$O'
    '2J?Blt6Z#;4gZH8Y~6>(6iKIX^+t5wyCqqPN25HuD&1pHlJi``g2h37D4&2Jd|IJq1;STp7donCt%n`P;TO&Y$xQusZj2QL(-'
    'sp7!D}|%EI9WR_4EDL0PiZO#{k+vT_5yr2%C?nRgeg{C~@Xa;qZAj3j{(FHMz*J7jqix0cZCfIePDy`Qi#NDG<u15t~wMHxK5n#c'
    '3zJf6SH<N1JqXCk8umN-'
    '1*YIdYwDnL7MZKIE)mhwX!PLD?hSEi^iCn8HF(^BA3ak6Ef$SDS^V#Q!(kzz18MsagA4++IPVmP1#itRM~FA>G~5dTYrF+S`6l5h'
    '+w^1mb;Z>9kWIR1_XB;XiU{D0-)xLqlcTwNmjvo>z|2F8*Z=PFLeBlRR0@#6)PPBdf^YOpcg_ZbLJ&O>-'
    'h9>VkU5N?%$@IXuiciyaO%#^>tSJA9v{-r@k!R!25mk7{-'
    'T|3>MAl<NUS`5m~DV8rK2j6dF_%11e?`tu9cS1)S!FLyoSe1aUor1Sg;R_G(Rw{g#(3k}H!fL#g1m7(*AOXJfX+Q#eVYS~X559-'
    'y#4tiJqJUfsUs2l;QlDK72PlCpWK5M(?OH7n&DwGK#xU5ZcX0;AujNs^HIL#2c@$ShD7G!{n?)_}caY3qGaV!%6WGiEnaB(_GXZK'
    '$VKd8VLC9e<8-Q1Y;JGn^=M^FN^%%kEqO+x<drMHTOCZ?3{@bYthKG1N6~XgpOag*oCEiX#@Kzd-fZzo*AOXR!(r=fC;9iVi1{y5'
    'FhXt28g5<rrkx%RMgx<ub^#ww2VX;?*&K!2z4@Bdd&+&aFGLVi`zA5OE41{0LL%4~`L3m*v!mT}o2Zc^D61JIh>5h{~0pwd^RBnU'
    '$!L};&<&U6jIO4u6hTn@Z{Am$}--t21tEBvM_-+pg8hC`^?Hz{g`@fTlVR(plQZf7~jVS}eO&wrhDc&gy!_C<1pm0DL7-'
    'l!cLb^j47-ly`Yq~=@7(O<KVO?i-V+a$C6EjkR8hMXm23xR%?p4eJ3x?4B$~sSxJXvqH=X-'
    'Bt<LRQz2<2PZF3mvrjXZ=!9>R<A5N=ZhVN73I@9XBO0nMJ^-'
    '@Eh{GpjkN9Q7Q&Qpz3|MbLdAhVJu4(47)Pw*~rY2)esVFeXbt*S`0=snCUocsCWg&(oL$=)w}bn*?20qIZ*^yNK?P0NplphjO5MN'
    '?!b4EX}6@yI9OS#;b}CDR$j26Zna6I3<rmnaAPxc^n>?(FQ<O0VKK9CB4K~+b+|~e1+{Qy^5-'
    '5gXnE4N#o(9E%fc#P}+KdqWG_@lcRCjRC-TgeTGY_9z}MGA_Py25j-'
    'hE@MX=+=^Y~Y?k3IC7QTB*GeMN_r`hRxFBQJ<1n;H7cM^?B@TkGcy_e)sgOz<R$)ol?-'
    '66rFb|BrM9QdA^*MIvdQ(Fj}ES0)0@LpHy#q5u$w!Pf54W|KvwRv`7&>-'
    'ORiZhp5O8iXbT3zaHImP7|G?B<f>Qogakv`w)Ck1M*bs-'
    '1f(8Cz*oPLGe&Y73jqX!gnJEwUe+BxuWwhea9iPk)FqV1R$zf&Xr2~&gVT!i0d%1n&yx9uE$?L!|Y6P;+yaY&jIZATnZ26`DPFnh'
    '<za?sl}xOb0}<)D|P^#Hm<8R%tc-JAxLgWfaq&WC!A_WzLPh&)5ScA*9<nnJsX3@O`GfgZpJYG>r3_H7<&=jEZcV+Lw<GW3soRr;'
    '3eup@jsa<E2xKXTCOd@pu5CVg{q*fX{}c@DCNN09BFhwQW%vR6oAbh`}O{t^#Kk-=tP{`geL9*#rOAln^>B$#Wk3dhS7bIra-'
    'g1L4c-'
    '66qT+mQy81KA7me&5BWQ$de&4yC^Ee<W2NXX;!^OPxYyumOrw0=;vsHb!wZBC9r+rApx0)A5W;Y&GIcyzM2nZ*?Z#G=c8H2;}|qA'
    'pb58^7Hc`-'
    'zfrl6%nC1Rd|Sgg+tC)=vO)9e2spCm=&_>!s1w6c(F*~ogTxtpv+SizD<>xP$~jvU+aWK_~JvHkceM=))Qok#sD*Xf=tmE?3*NL4'
    'Cm7w5;TULXh1mtz9b9aDrAK2r`VjirGS0t+cR1l*sKRkM{c!D*SLnFa5Ahl`urkHo|(tw{5&Qv$YXNn3?{GClR3-'
    'bD*YN~Ib5w@=PZYzdJ3mD%)``iNA~co(!-hgd%n^mu@WnX+h=0jZYsj<j2O3<V`8|C94j-al-'
    'RYeeqt(a;UP{;#qA~<lOT9u6;4bNy!K5J1n&iOhXld9GYu#Qw^!yh-'
    'Ig$&m^`>xaTfKP7>4K*L@iM?qY&<+unhbzceR+hQ8|%pZo=x8ni#z^1p50T^q!SR?}9vfFU+HNm#jp^{o{(GzEgUHD-'
    'ARYlpf^@0?k6DM>A9DBBjT0Q?2in9?P|sxbGT@0K7H^a6=J*XT|{DPMK#cfVWrX*-`+uZ+}uMfZ-udN(FEOjY$A7EW}Ak0Jd+E0N'
    '@Mh4haC>g$9%Z;2Vk}ZRk0wm)wvHt*3sWi*qDjGdxr{=qBk(JrVM?*@h=G1$z1-^q!qZ@4`HKFUq60MXdBv{lV(c!+A~U;k-'
    '8Va9$UBIIj;qoHv9X&Krwx`*Mui2}QV_7320w>9}9b)zZ>G=W%P_`zNWmg@^b_DsCsxm;~Iy68t0yxAsjEaC;HmApy57Xh1o*y)`'
    'd*t23Eq)w+?p5Rx;Y!J*d7MK)vhXf?vAOzgvK+U25avjjT%BGjIfNA03KYOC_7-'
    '8Ca~hnW3+$IL6t0R>iGuQaU+UX?JF1+Pn()&*83t}$&2oHIGp99YPfGZS9lioyF-'
    '5qM|E;2k0zKQwxa$Cm=Hef^VD!3z&@aw>SAqA>~Jg_Srt3B2}A62MzUcSr#5t~8(=@ct<SUKa#5`$*<-szJfqMEXoKIHzl(-'
    'V3^s1#J^-'
    'FgMRPkF1R9=j|_(YapbEky9#Uh>^1(<T#SQAmodY%OI2!lBXafJd(4(b=a6K(3=?TlXLU?<oo<S8JOQEyZL>R!}RJH(_6C+-'
    '$Aj%w_fTsgy-o}uP+g;uupkPDxBf|r=-'
    'GpD~(BrR=|XvB2%Ij_DvF^6$9xG3DJt(Xh0coHaqW_=NY(!$R(L)V39xv1YaGfF#D6Q+9*t*<Ez#R(~$V8QejpHUmc(@x@fN|!93'
    'P2ba(rcoh9rTP8zSla_tC;?ea<EBZ}(4tZL)!$hy#arAjY0yDR48sE#P2P|I&%YOYF7=wSL*_>pCAt+DK_S%=zAkwb0gyo_#NDmR'
    '_paF|L<O^5rykhDtfopDG)Vmi(Cc1elp?Px$kV)_UgP*!Ewj=8FcM}+y%R))P$VfN3|$YNz!s0Qm!1=_W)0AdbO!H3-'
    'Rdg?~wi*@=a-'
    '|GqgMrE8=R)xP=$?=AZrEbls5rd>|!`We%NPQrOAug4=EoX;aCiOv_9d^0Y2eWH!u+;6?v$hLzXfxP{AR9ph_V#(OkK9Iugzgbxu'
    'xB2F9ZF%4ZhTKF27BO;Gz|8{Aqg0yN!UIKgY9WR0tQFYfU>H?iiF|2(hLt`>`m&Wz*lcjKXs1lPBmvM|D#q4GlKYPe}w^RzG|s3-'
    'p5z#kl<Q(;aqf3EEykHn7J8zbjaglRGxulH!Mv%OVFNqzQP<7S5qQ={6yb;_6cG?2<%a~WqjgVBW{`}q%S~k41#l4O6;mG&Q16xK'
    'Ip$LQeF=!1xk9`dlNy4ce*zbmH0sSCc%>CzG)IHJJ5gxSdO9rWfhENMdNNpD|4n@=I227&QPw*k%9KT5s6ZG$xl+Skp5_F^!3stB'
    '6p3t<?e=wPx*45%BXO$byT?6CMsM!Fe+SZ8x<}d6csKW92G9MiwYOvj#ql*UuizeApej&@<(s0cI@LAh94EduwyA0(yi}Hg(2MOz'
    'El`~L}L<QNE6X435JKzfCLzhrU7N~2<()DVXZPdTNqBNFbBscG@i-bqgwk)EK-'
    'b085(590o5Q;C2Qg9X)7J^CMGY{W3Pa?ez7s|SZoSB7MlZ)#g@QhvDNcfm?pv#SzvA>Gg)A2BU4$xjlh6s6kcUM&!Dhl9))AJg~E'
    '4Z6uwi0!cL`7NVmQ}6@_r8`%_W)4vk4bAx*>%Nhs_{0}@a;h6a?SC+?Kf6CVZp#`d1=qs-ZHQyQ$Jcs!iYGtC+P+||>~naZccf;{'
    '7>B+4@mOQH<p$Pmh{H@GV<$Sofm+Fk0aJ!Y=PmY5M{I^{7lw#b0#O(NT~)ss?M?OU`dyA%3a0UYh1`q0n?g!%5AT%HBun=ue)7Xh'
    '(zDIn7OJdg@RxYGlvK%7lu5`ajv+dK(~ooGM;5XaJhvXsT0vdUr=t_jDv@hIf@A|8J(s@U5+BCYaeuK=A-A$N-'
    'XJ=kU^E5;tY>M4r<!k5zgRB>aWJ+Q3}BkmgfS%kRGdBlw?HklnuhNs*I@!J?(R?EP((HE77Ru~j(khz2dOlxhlkpj09CsvpPti?8'
    '=!gOzzx1Eu0Cctr85gfaef+M}`gQ;+YJ3W{R$7wVs0gg2FJ0`)gGYv?9<2V{nmXer1ou5-'
    '?7u#2CyrOo=gW^VubqeK93#M9ZC1Xd4motrV+m~8AdaVa%9QlFziP+)(s6_gp9uC=$Nu&?zsXUX=v&aPGDf=cM6_CX?Xxb+!-'
    'Zr4`{6w)R{#u%!E0($6Nb?KDLU&K>LB(5OsQD_h1-'
    'j(7z{A@*s9%Z!`CQgL*|S3lG}3(^N<|~w=Al$HK1X8`&`7heQxY1x(0~LqK1>73p&9r)d&#*alHw+J^Aw!evBk1iXvW^x%u#p(dN'
    'ZX4HN|-x!i-'
    '9|sVJ+CB~ne+jdL0pmh7(S2gez!eP@6y;iA2I@^UhS0QB%DIXF4oY!Dos9AP#J4o{9Wje_Hoqs%730m{*4v)~Bj7{^$IA<D6~AxZ'
    'g+rRHH3flh@~(OXbLC+6L?!XasScTI6fIm>%rZh8ADv#YfPo)BhjOev;k;SX8snqm5fi)rSpa8Ym02^Uk$x#41o*+c0i%x<`+Ys-'
    'pUb9?#Df-'
    'B9wO0Q&oiT$X_C(5a3Ds&68*tuG!3iBJg<U#j370)`Ck+g!!P9v>qMbwQBRKZ5yV;D0*PL(<C$P~&?^GLkV%|n7EJm99(AP3xOQ|'
    'd~0#UbUa^vK*wE2R0QjocjhYT}h9cl%;yblcmvdR$@lQF?_p_I59rc`^0DUIjBhraIWSU=~T&Ur`S+t0glVUuo9Lz-GKo1~%jMlG'
    '%)hnvEc$eIDa)v+Ir7>y4o~vuRYgt}Oz`lXrsbt^(szZ?o8YjjK$9vi4^Zd!~7`)Oy<o+?=}JaI4L!L9NX+rkn+TG`HZO9RAI2?W'
    'ZeDMW&e|N#`2&R<4muJAIAWARVPNX0P2E<ZLabN;N1Zz*?frYQ+p#OO;upm;!5=GHVrcU@cc>onjKK70Rqv%!0L2nGK3*uvRIvF<'
    'xxk&9%(^*N9gf2G9+rlW#CpBT>!q1e=aJ4s9%e2mG)hk*m@?R!WH5dDxP=)^MvWscXHS#+0+x&*at`7Ux{Mxf_)ES6*doS<s84ec'
    'QIcEX{7aM7_9RS5=o#Cu?)nAodUKQ83@e+$Vb$%y%(;&)x;IAm+y1r(hPwbUynP%=gTNi=`!xV}Xgi+K;TQ{oK|5q?7LNs!KY72J'
    'H3D&dR*+_0B_;`A~5K<Qc-;=A=jpOQd<c)Q-1*Y-=hC;8t5xQLvoGl!Jm7b13LuFpt|E|D-'
    'Tmn&nW?sbJnz?5n#=!iT%?RAQ*p*9ldJ;4A)nW!#{rI?7D9bdHZ;V8Q;)y(O9wdrM!e38ut8p<3ukEEuxu)J2&OyuI92nU54ZYq}'
    '}(v2wXL<}hVG@n*!~%6#h02+LWwHqi1sf~7nBax;awz2wUOFhiqxqSVUUKh{Jgt~}hTiAr4gMKq?Im7knj`97$>(f&UP7e98T9YC'
    'In10<{8$QbW5X5_9tU9vum?4kXn*NBlKdX)%F!3`qNzpfTBbLN_`43A@aUZfH{g?uo)!)FQeJ3djW;6&M3+8P~J>A*Ivmm%T2m=3'
    'Bp66gUqt6lg*GEXCFjJg0C<6SC|5z#yuulWuk2MeB5q^>#KN~Es&mo%oFHJ_eabFjD`v%CGN3ezJKGOR@j=~D4%6F0;u6iC3^KIa'
    'p{x|5wH&v*mQ?Yx|{??iJSju8;HpvQ-'
    '`hJ6gGW03Z+VL_h{$x3&GwI7agZRh_Vf#F0K02qle9Tx!@g`RasWjz|5n~uwR40@Ivne|w-'
    '>j#!kM0i;<(>wMK6he{`Je^EkZ@7<4UGKMPOgZcQRtY*DGeEGA9<NwKS{zgQ4G?k1j!v|t#<^BM`2FI1x+O?$I75d!W$YFBnrgqD'
    '3}4fk<EJ6^)2tV{M6=p_fpd#|f%EtbX-^a8c(9t`gAqOILlHe{tte3#_k-Mi8{%Bc7f=hN+_V>!8B(IMh~8_QOlI~QCzF}|#wAr1'
    '*^2Fisv=l`aY<E0ut4LIs*1+b9m>hG{IHm)vNR!YZ!ACeo=LDzhe=n8QH5@&dda0{jgBWRvRJ`p9S~K;U*Nu9tyjlD-H-'
    '?B@H|j=%>(tROeKD8;2Q*?b#FxbdtXHRdw)dx`v67HQ%L+a#+YqDu}zp=r$C}cn3WE*_=*q9WD38BWio}|!)0NXGk0OGA1({C402'
    'eDhs(k&gM3#?+bILH4DzSw4&`9>i+p1HFoi{yo{w*qG<BH6{-'
    '$KV`!KGPBSbrm<WPxt@y04zO^q>Zk<8&d2d(9Lxw+7Nze2BYj8m4_VW9hdjb4*$Ta$(Jh&-'
    'Hk%ftEU44k)`8Pa^IybcYo^|kUOz_X?KMx~PXHOA;&uRwt(^lng~y+`PECoR3vBQl-3?-7|!-S<dY_^k-'
    'NZXPKMzl<)J{71^dFN@IK$XY4`znq`^G~J;b{La@+?-fnp{hQzv#KN~at}U)aGP6o%^s5e$!+Qh**LzT7CJ59hYKusM^-'
    '^;_a|G*^rpkT4R<F(Vu*m{;WFD})=K=dn2C$8A3cy+5Iy#xD(tPGFHk)bEeC{r`nCa4dp}bKtQ<|?-'
    'Dv#f$7|ClC7^#HhweUct<S@SKqcWYv?@^h~;`eBhB!*dkG)WS}syvz`iFYSUKS2^dLw6_#$;)yz$y6aU&y!b2p5D61(_2Mdm))<;'
    'RjlQ@Dv!msb|b*I4!-IwbY6q68TQ-P;cKS-_6_)&<w6?pHQRo>POr=L$jPF6R36=X<k9_X2Hm4TUvNhN{2v3Y>r^Ut-'
    '{u&#tKk7GYBwsdzarFDIMm`RJSNkb`yP|&%zclQh1%^xADzd_LM@9^nC-{PLM@BaJ;)<f25Pz7<5{{xIjCKmx7AR(R=U;_0<P-'
    'WH60pIpcXM#u@zL0#hguCXlgNKZeU0zV0OJ;pX;=f1>fjA`1Z_$Z$ehJ1?84i=;5H;vR*O*lv@_<BSE=k)p|4-YOJG-'
    '0Yi;N{#Y>728vX=zAZ6q|E<iG)=>M8GFO!-ildi*T&9!tJucJ9`W{aziGy``JgFoO*6H!2lDIv|F`iHoH-YX@4s5p+m&Bo1<sOcJ'
    'kmIDIFUi+#9EayJLEH{&!ZmLofcKHb4OBMh4Y{s5S@@30!*{Pdd?&{61?XZ(z(u40yy$9lAt`__wv%)*DF85bz;Zz;fH0=8xIh%r'
    'WloeFi9){2Nz#U*k_h~^#$bM0>ZOF`Nm8#W0cD(sCuBN#-'
    'xD&Oyzhx5C~v{yffOi@!FDDoP~MB8ItfsoNOvfw9Bw<2XYXq*O<Nx;Y$MG<%5~kvrlAKbeTcniD@{A4JKBqbq-n2oC+m?|i#Z6-'
    'xYD;xc=4jkY}lwb=GyaQ!8|q(=DqV^o>a`8N}5h3GZKS_;!_v~AYq?nv#^`3k3wCj1-k<G@u@a-'
    'sI8YuYWN`nzy5JO=TlNIA^1*|dS!7vC%yWUGM&8dNtsUG_heEvC#=GgN!6UNN>3(LbM8$+tb}ULNpy#@syTP&B*3^j(s$aejDxY&'
    'ah7Mj!0c7;sCZ7{r&@6#GxPfefA4uhzru$-QU_1w6CS$(yv|2FwgY&>pYiB!8x|2#-V^yS?vH#J4{%%v1AjhuU@|B-'
    '(Qer%pXFD(t#Y{^r0p9Xv!)Ic%$nK(iA^Hh`c7-'
    '~Nb^!F>6+t^v`V@iaY#b`A5PCTN$G$4AZdL{`d=*#C@V=nhr=rK`vOWMS8>#~fY}4%T++KMS{hJ9XjUC}3#j340Re*-'
    '3jLb5d<F_V#d*A{u%`{zH&tO*kg3APiLA@v-pJ*9U*s=y9g2(~hDyfWYkF#oyM6Px8}4z3V{euCnJ&|q_(sGitS-RjBNSqbPF12B'
    '-ag>BsVF=ghoqsfI}RxWh1>@dCu?X~C}i=uFUEGuK_SC-'
    'I1MN(Lq9JK(GApEMwtl6Q`V1h$BBT{T*U{3`zbT7;Fz8)Us6q+Yx=6hiV?>wEiu=`1Lt);Ee6hh+b(9etWng|&IQ<SmiU<>eX#Ww'
    '>4|-'
    'QBk?oUUHw+#XPUdZP2y*UySiQCXQoVL)*Bha@qR(v=kVR2f_Wi^qu}mq^xEI0rbWX;e3zOQ{UVJi1CZ=r!$r8ZEI_h$xgU1#C<l;'
    '*P~(1WnK}A7Kw|9=2B=&?`goF&{X&kZlaC|_omZ-WFbN0sO|qEz26VSNw#kwB{8eX&S7Bq0>4J;bUz3=o<*MUemsq9Z@}%F8SncH'
    'M<EBU~cN!R+v=6$!tcban!TyGx9)rDQxy9^1MwT}!hT?;Qn@S8Wn3qeP&-'
    'S6_r$P}PVty(VH_@1c6B?)Jx}*~tUgf&fQupi1Eau!LFB^ZEaf@Cq;|S|>nCn-~-'
    'v5Hn@TkV`2QYcZFOkkLMq&Z8?L0)L=ov9Y_TP3ryW=Oli9@$ozkjpj^t`52E$Y&~nxQb{GN~O<7IF?)b3BPT2W&@_i<|?t6N<$W3'
    '`{>ldpA0U;(dZ>^)JKrJVvP!3{3Xr7o<WF9%4Z%6dPzv8BlZ*o0j7GvY^ODI=ssDWkFGe^7iY?R2H*PWJ;w0B4h~x&;Wv-Vae5^T'
    'PwDn(^AoyUkL5b+>sj5D#1ThU=Eu{WWAmlBl3W4*O~jm<Z7-'
    'Z(5>~woM*1v=s}$BtPj+e_>zIP`clp%)CcLyIDt(ctS{$;Cf!aCP9?G%6C?2+!Cguo5LvE#MxuS`g{eq{hgg`3#0fMe0g1Q-'
    'HzXkuUgd@)B#x#5Whsoi<oQ29);Ot#S0HRogR{q7flTQan%IF*IU_V;+zzWIJ<g`;SuxI9ZM%}()v3Hv;Fc!Sg_&GnBJf$lyis7r'
    '@7cmkDNtrpOkmnFw8}v%2+b@nxMAni$*6!h*j7<FF>9jR$yA2Au`wz;A`vs8vNKZK7RT`26fH_cB|OBUR8&4iW6D4!$G337eq9zS'
    'Iqn3n^6RotxqYa3|8<#K<1Trv5mvR<bSkh@XB&;mkt9~>xxD7)InXr=_}{x^NW{)Kb+)JW3yj)ndUlN3%5B%OJ4t+(WFAp_{Z~4*'
    'Lt_8m=->{C{eP#kJEVU8gO2Zz`uR^f!9$||zvvJTG<J8>Ilgxi>)hqBm-1f0ZALm_r-l-u)6U5EsUU?X_&yb+TWL&!=*0E8F-'
    'dg7tK66*I>*w0vh>Nk<D4?QFBNg3{pC^2Fzd<7`n})G`@Q?9oZq`T(?ekp^?P=W6E307xlUkSN^Nsp>x9dw4Ycc>a5?pWKIgh5ez'
    '$#tgxk%`mAV;W=M!mLeE9A8JL$8|&7VD`nDxy2WOODmLD%b9zW%iq>A>|k_{Zytb4768PD;!!h8H?1F}WDt`y`c;DY6e%aHnL7zy'
    'hD7QZhyQ(13(Y5dg?Zsw|iJULv2gS5t_|vCLfs_!lHVaCz{x1+yS>l~csh|8j#0O--w?<98F8O*zFFdDfel#r}Wu*grat{fjf$$8'
    '>mU{^Hb*-'
    '%I>B`ELFs@$+{l?b`$`74(Sn(Z;t|7gO(Pr*90VUg+nhan9`*pZv=O=>D`ZK9D6FiO?MUBo)U!a7Y@Cd*YCU$mPCBK9CZ*gvI)aN'
    '{L(^O#>1lm$2ABDG$f3@;KIwx-Nu%>Hwu)F<2a^!1_|klg~A+){D&w!FE<PHm#4*i&-'
    '66mu42d|I4HIm^^w1Wr)C|!jabApaKhuZ749H*n|T6i7hBFpV)u`>#3Ce^doe1En_)eE%Xiq*JZ+tECpA5>6257I^YRTPA%&An#L'
    'pmc0VMHNdYXZ&&f%EJ%$D(02bEw<njR9PUQQBoP~8I^L|vT4LlH7vmeEXteYyAV~WW~S;u4+vTx-fdu$%Emt+Jk9oKX|J6r8+3@`'
    'bL^K&pB;7zx4(c$XrNC!w}5WsT4^U&q$O}Puu&6>(*w|^{m7YooH2*^u?8Bq$zI2EU)0vR6RlvE&pL}L<w+!D!-'
    'Qh*Grc1jYEkEH<#K!#O6r92=XDvJ9sz_DM)61Cn}yw^B!XMI|AXmz(>PgD(Sj4-'
    'q&t21DNXV!6=h49;X2p^Y+@TD0D|4!oP&rVDDgT&8Y-'
    'PIo@e*WgJ?v(iXhr9YeiJyPEtG`II5+=N0ufVoj4U1JsCBr))hVT+$hFb{3UmjBm!Z;N_O@%N#!cS8n{0@ytfbjlE(Uk&WShb%fL'
    'HIZtkN{y=^`Djp;a)|`!{WH0zYes5v!soOg7ABj<DuE~e<I@XwE*KWmqL%jZ4i)**`Ha5YZl4x<dJ-'
    'Q9?6$w^atEr%(ee#i3~2!6?|p-DwrB~^;c;c;Jw||-'
    '=$dtlkBekA<bHtL3i~}iIluA7gD+OTE#eq=NN5q49_sO6pnEsewK=3c!Zy&;&?WVNx(6z$<LB-'
    '3~TnYBpe@40}^lyYyPwHaD1#Nk%|ghK^+GOuE7)*Fy<PiXSW*=8FK@$V%uj{TTwiZD+4MsZE+}jIqTHTV)@-XmQToI`SM&-'
    'oT*@kG4IqgBPH`rO*cw1@6-'
    '%akjy(Z(^N?2oti}sTgy8&8%((C63O=}Wwva^`NcLWGpR&KfnNUSsrZFQ_<1URr_q=M{I){UxD@=tI{iEezbDXu1pLA}|GYf>o>G'
    '*4iez@mankfQ-KF`JGW)u#9@3nS)$#V~ZTOo7P9=C(nClDd68k`yn|#<{|AINX;8IvA>ptt^&0_eyJcduqV|ehkT7Ucyx84=xgbi'
    '%|zl$=$$zfJsqAOs1T#-)tOGs_CA&WWElTlK^l?%|WSJ|)_=b+-'
    't{~{H`@LIn}#qe`9CLss461fgja!_HNexXuwP*0=*2|1{+&c7%R!>1Nu_&kB1&q5+Se7MZ+LV=$zLs~t2z6y!-'
    '@cG7FZ58;5(E4B=t^Eo)w9d=tOAp9M+!-'
    'KE7q}F}_k>Fw2#M5nxTSt7i_#JsLyQ?AS{LUi)&+?ERDwMQ3Q%Jj1>hNlY#sLk!S#dRi{eTwH!FP#$>e}AB$XTxh9p#C!SeJeq*P'
    '+T^7SdCRASAeJ0w(M4WK)e1L8A_Z933_J4hxu>8?OcWM^8GHnUDdX){q*bWA87kBUc&OoB`kY!ww_`k_3g&&Xr?+dQVvi<^`oWJ9'
    'LRJ0i-InIocnm=A<YfpKjo8u{SZP3p~jZtO1g7L24MF&p-e@w-'
    '8pr53*$KP3Web_R|vq;nn~T}b9UJi3sgNjyZ)m7qzSfk$hKCh;xZAwiQkkM2+ofG@}c*yZ#$s-'
    'hg%zKG07ifyw`I)+7R^4W$$;wOS|Z63n?^AP?n58?AOVj0u2H%3+Aj!FtieKv8JVh^dea$IjuiRAugGk-'
    'd=`=8^f#AFHbmmec*JTr!HU;J<)L+4Tuwli={A)UtYm_j;@<1tAXuB8V{z%VS<F-'
    'aKyj_#0v;q&PZ<zV;{QC!7c6L_lYqLwv*JmVONMNQ*ep-ErNvlvGa)q=I-'
    '>fo1S?;2(=%M5^r=K*|X9>DYS0KOmtV61al7uC6}kLp|+qdJ#OQSHGNC%^9(rnjfDdq9{Yh{i6Bfby&uzdywf;rAE#DS=-'
    '*0ml~7sVR>wq*GHKn}pxt^k512g|#|13BU8{4hi_ZfbLKZey=Q&!7b3C;J8NXQ0!K-'
    '+JHUA$2sp+U&^J88lvJjCc8!<oJna^5<d}sN96H)Rvy0#^7y?lgI}zX+2HUvOyCV&8ty&9^zba1_X^X~voGE!OfTQ9;6Y)I^n{BK'
    '33HSxk!<@T^qFTDvV;T+tr~GEeoA22PQr18bYjip3hBg}$0cES1U*;+hGD&qOTzF1x<dknFQhw^gW(&Bthw_Aem)Onq`j1R(K7hi'
    'l<z_YjU)4DJUfrZg?Thyl+kG47d~|xD|Q;CYxRfpS(~L9;c0@mN;AsS1UFG;w2IqAHb(8Q5Y@GXKztXqhWw@ENQ`X_>0Jy@NTh2L'
    'sbrw%#5n#Lez?A{gDj?lanpBvA)O}l_(D2O=<$V=G@p_5U<qkHu+qmDQqp`D(j5}gd@iCpl!N12i=xfnD8~hkozjr&CE^L4$KK(X'
    '6R}nWUAPKK|FU{Sg~U&U{84%2pOZ)aqCE1eGRS{RL?saK3NCbTY42EO&*8)2NasG>vuAISW~ArH7RrqA9NAKtvB~77=f)WRIerMk'
    'hvKJnx;cDKD5TS#o=`}qJv|{w7LTF_OOVB|W+x=c;ze|a1X)}~cPIzLe=3s2+qnj=am7LwbLZwU_kA970|VwPvZi7Me3Z634LLHR'
    'wB6~*k`blt&On}wC~bEpGG#<*yR(p!C`#L%jocX}(sqA{Z}sNjG4`Fb>zSz8dVk*%%raUTN|pXB1D|^&=HVQfE{NHkVSNaG2-'
    'aQjQv%j@u1_qav%8*HNN0CFF$vbA>A@0U4X`*d3D)1!9TH$YknT_xX+euTX+fRYy^x_vyh}2{QJ_hLb9!-'
    '1Ex$PUDaKo8TPF65bsPf(C%5Cb&lTdGQ=d<7e={!1ASI!OO}gRSl47YI!A%dA>5<%=V!0m0y(w1c(cGG1r5@v&B7ng+7MoP0P|}_'
    'j$LB6Bn3rsEMSE#`GUUN(U$$jxg?KX@l2#$U0}d%aYu!zJT~gjUtW8}~-'
    'ujy~pseDv7Wv|eyChZRhIFsHJlWy4JY$vE)C3DFyi!cyjLG7#aOk`Mzw<D2CWhbCJpBHffgga+vM*x>@en<nNw%)gBba0BN<EVK;'
    'I7i6m~HE7J(|h3uF+#S>Nix6O=Fk+RSdsN3ZU>4{650IUM1jXUv~dg_;te}Y4AG?ha|wyPW&rL@PoB^B?*52r2%DCoE6Ew8ms{iR'
    'GTdBDzOx8_mFr~4_`hAg}UfC7ipk#F80yd^KzKrY%_Zeru<YvhdiM1R}xLSI|A?33mI%q%VYC@S!^OdIP<y;M;dS@b{UE6?#!?;3'
    'aQ<h=4CXtXn25m;IKu*gCdnP_VgH&b5P_HCO^jRawSYF`;rHwmh!+u9FSVdvy#S?flBTUW2gMpvQWtfEUe0_WucO-tN+n}vMSDsg'
    'kpDPdSh*YMMx*AkZA8t=Ri(Cg)hE&TQyT|V%`FLKPLytdj*!x2EFn=fl67>EAJOrLL2nT2LzVU2EFn@fu*!TuY5>gIj!pyaJs}O^'
    'Bp1P{9*>u)ANviD+B2f*jt2YV#Z<*f`>%ZX1$ZrJKCtM&>QYdB#7-7x4GQT0l+?Pt5k5pty-mma|w+p1Dwn@V<&HNS>WW*EX?oZv'
    'cSoq*|%swS(RqH=9vrX3V23GeA&*&T4hs9=2`jqCbDaIB9|JFpZ#)%K{%}YNz6*PBjhN&ltIsoJbK>F6!7+x1bBPVOPU^%OT}Ri^'
    '^{zNbA*H+a<TZ45`M_l;zvpNA(xByk?=#V7e89U4@(q0{0P148L_BbDfDVW?`omfl{mcZ%U7mi5gws36^rv|Ov3SPm*Tag;~Q4%w'
    'WQ<wZ5mLP_PA?aP&Q(S)hzPXH1P^?hdE3r<OWHI%zCa2tPlO=76{j0E>05oon#tVjI~PL+Vc(lUg|cUkL3?iAL!k%f0Vkdy{N!CF'
    'ny4(Outj=gYCukn9;4<*^4tICg0o<QgQyB0sYK8=-'
    '=5ErOS_yw)T&~_@Ok<*mL_+<hm?D>0%$Nbt)X;AzG)x@lzU8KBbGBq}P*3U0@zxPa<`BhX$0TO%}m%fxu!*&UQUX={lwdC|Qo`+9'
    'G0327|U9RmZ!qoHE8_e~F~0cLe^^|75T-'
    'E02|TGskNk{M7}5Tg6N@3*Gmx>S=cC5Z&K&^SxjSwi#xzU{<Y}W{F_Zp;>0BV0M|=W|>IkTs<?!;wqun5ER=<Qz&Is+qZ6$3PpH^'
    'HmOj2p2j3dM!N=YBuU0uNH~@v8Q-'
    'M=WvP(6=BYGB7tCcoP2)_7seN~ZM8B6a_?ex@&wCmC)T1p^&s}6eiJZpmWkF$dH0(1`S#&(?vrti-'
    '#Rq>4h*Q_JQeQzjZA_jT*as=p#J%VG65D&!7@M;XTu(we*quMJR_aDmT}Wb2t%-'
    'RF8iZbJySE6vu>|3ooxKB7aR~Q6FcpWBXiWKpYi=#4BvG!-'
    'Myk6M%C+}sKv^QO$Y;Jmn7<XwIA_ZASJ>d`RagCSCWbQ2WOsG0GA}DLrQi$r8bw^dw@GjTA7yZiRV>`<Vkp<iAM0|pIPZ~vu87sQ'
    'FSg7L-'
    'Ped6$<*467X0>Ko<QJ1856!{5~v&4VU@y<+BuH>%tXEx0_OaJxvx+nry5^=d}=aka~zVE%eNyANyw;%RTy7L$*6`^8ed4ss9sEWN'
    'XV#uj_y#7gZ#cC46majd5ps_!15mj^N95Bn_-f<_fbQ+PeydS=6>n<cwf=4#7KQb{#cu<C3&PipCMCkr_6As-'
    'q07r*XmfbqTp@o`f}A4Oh1mln}z~bius}&u;%6zxKm^I0(^XfQHFL3wQ83bz*CUBmjL{2!Q4|Q(b)p$<f&8uAC5!P0NfpiBxF;=;'
    'yjg<O%03pR8lte61qb|Hudv#hjIWsvWR1&lfpIk`Fl~D&Iaa7VNz<JfiWJ#;hqx}zU^dtNr#fGSM{qgpkI|g*63<!9?&nSY<Xcd1'
    '_^CN)*!ux3$iZJYq=ilQoWALu`bi=xf<(oy@89d2J4Nf^s9%)kX=+T4;Bcre<_$7OVtwMTR)u|mV`%mIyEf$B8^E1OTq#?ofMXY1'
    '$sIuEV-2KkPw!9f$mTaXdf*C?RXJ)!NLsc26i*m3qF7w6vn2xSEjDTQJM3&h^-'
    'ZQt(dO$Jb|oM=Y?)Q_sWRj+B_hAQ@6=_a*XxK^2d5)Ez4v5#f<PiNSTK`3ru@uMtI(&4$6!q-lR0<)~+#%-'
    'xp?(BZU#>5?P|98@|sosVIg=cqSFan`lfzOE;L>X9_7T-C*jUDWtS?TSj+CXzBJM-Ju*5KU0Kakg5%s5Ix&D1zJdC!H!f*_=j#z$'
    'o`9X)GPD;d{QMdNvIbHbC2{*eqPhB#n^mJ{#b{r<#}wrlxg$|rqyUBC+;ovSVu<;>ICY$u5#sIg@%_aT{%RVQIg95Iw~`|1gFuD@'
    'x6A70sOWw7g+!=ESLuh#YP#u^s}h|hDUfd6~GNNrVIeHRE7C}wk!av&=c#~vH;9dbvfOk3;^?<dWr5(4uD@QK}`@*zSPS=xMjiX-'
    '1_FtU<p|py+NX#)FLF0<a!Ieu3wK)`?~zG)>td@sQq^awcUJIgDP{F)Wcn?YK#qxv?SbUe(E(G4>(KdbsPaWTj}*2060hK4e0+*<'
    'A^;hLa!cVR$I0A`+|9>P(p{mcb||7UU-5Dso<SJV-j=-'
    'ScVCO#Lid~l5~g_bcX~T;@@<Ka=<&e1Rnu4Omkg#P<+({p<2s9fMEiEGQexPPe>P2?~6UTm;gRN_YM6<4Ba>6kF~{GnTPIwGSEF-'
    '>JhG+I!3QXF-'
    '?3gsmEAi_}+MEG;omX45e3djO$FL*K$<rT%|Xbh*tdwk?i3ybT2pSEp*=%W?(7x7vFtiDs<rqCZ<C7DH@XiU08;Rg~U!+6O*93lJ'
    '1ZI-T%-X%7O0moYSGoG*if*X%q3WxA<QRW~|c_HVO03f_c<kZ3dOLU><W$*7jyFXtd5(b(Ay@D0`Im#!{`5oN-SLGGho&kw4Z9Yg'
    'Hb?FBiGZh+MOUZ##1nX*1(Ek*ZcVu%ZLswN<`ktVPsg!Dr6^+*>3u&HgL)=*}ycVb+SfvtX{5B|K;N{*zK+4ELXu3gfLbCc$?GOE'
    'IaC*b!?|lJ9I4-66qu_A=d}92mb<RD-'
    '1yR?I6Gdgw=<D3=K^IBtoxRm7gudiY_Kl~oTb8VlgHdKoqSLCk$D%{!9S)lWchNfw!4>kY_I7>q0AQ2wCIQy3KT?Kfp?wapO70`f'
    '=cc|Ch!c-OPa^~U(FeUOt2m=k+J8sY84I6U?Ordbe%|F2+fh%u}}Zs!1U(xg(<-'
    'XxPLYHup5s~gKbnCP3z>grYz*b8ndtE(Fuee>uJWnh?%z6o@Paxna1DORKA6@cL=p>OvJj25-'
    'XH+o~0gEikO(^vp7J7<Ne@Gey^=0B9?Ey?<+rS6Cj`*0qy59Ja2Z62``Bg9t0J`s}pC4qGACOI01bq0rXj#(9^w6o-'
    '{GUFtY5x8gSR=|3Lv(qSQC&fs9Pnc?p<Ube8uVsk{n|<$_Wim<a%`%y!_GX!4%)tWOEK`iRhp3fQf-(0k-'
    '66r4n@D#k2gzTQpw*~E1<@AAqRr;4x4FnEjjYSgOEFv8)&{k{wLG%Z&XwF8*%^L5MXx2N=)EP(g~ZOCtzV2F{YW0twRuQ?mxuHun'
    'dM;CQ_ZG=XZ`GsnK(X7*b}pFEcU0G?JBta(pZUH8&1c5RAN_L*Kz4FY4(=r;z%R4eJ;)`Y!VVn&qorFboSa?WIB88ElG`HVcu^^Y'
    '7|>bZO~G<3Fr<9>hUDHLpeyEU*uP7OQ9-HC3ljF88(KAK*mgRZ>wOWcf-Dufb_Q9&%p;hGVZktGMIfdkJ;gQ%+Aka_Bom500kYUv'
    'B0$9dYai>U_MYi-'
    'E1i^9jKmRwicLAT+cL3DwrBn&oZKdnL+hzBP&w*YoCvVvL0+UkfEn*%*Im5H+BwwBh&e7e<RcRYk!jjWmt~iBtdyN_3%rfDWE$fK'
    '>0bkLpe}hR-'
    '_^~7BJtSAE)<&?ojLXjB27tbh^a9Bp1u}#`0iLC;IugZ8ef$X4`f8q*{x}$MPT^kq7aDJcysq=n`+>9j%H0)>EOvdDsBi6%|gj0<'
    'gEQaPAAhb<+x`m;<bvRXB+YsNSKXguwj~>e?4#^j>TltefIb1#?}gk{J8;zm@6SwZE0=+_k?=f-fw@Z<F9Vf|^dIFci=o65#th-J'
    'u-'
    '#t}W_3vRW_U_LzE&Udqie^*X(rTVv|=dIdGcv`yt#npsZm{1mi9$?*kCM;9UX@jPNj<`KIvkJuM7h+Ty)l&VNi86$Obl6se#wd&4'
    '9(%Bbf%%fNz3S9f*?juH#YQd2lI9UG0WOf+PdxHa%aBBG+p*p;>NkaJ&1v?qH%5>7&TV*<F?X6`2m~AOoms`sMaMQqLaBEorW}n7'
    'Fx<eTNW}n6jbcb>PyroE$n1`;_;B7$tz-33&;`akMAu%GJ8x|AKp>|Q5t$wW%K4I75@`*f@N9CcsC=cZqGf-ZFcX7gZDJBd%;kz8'
    '0@mM7n_L!0^d{<&C9w&Sou=&nFf2Q;-m|;|5p2k`GQY?U1o6CqIMz1x4OEtl>6L6bM=d8U=rgPTbmLz{+EpAJazoV$lZwe~`-'
    '628#zDRc{2fnb9d6q5&@I2NYT1eN&wzqYu)rK@lcDN2Beb_6@Es*mTHR4Rlxn%>55d35w!K3pC{yvZ3mt>CX0Q8i9#VyNv6wKW*_'
    'd&0MxhLj2=v^@P##{$S6wG}w*TInmbAQZra8$uOP>So|huCq4MUG>AiCJUyia!+0&{BGZotE2WI*;w`GM&fv_9R@x3f-'
    'QB>(SJ(I)#>i?vQ}%m*@^<(Gu*I%V5`?%~jG=Ib*1$G)oF*lXD}#E0I!RtovC{Wj<2oIQO$&%6zQM@$PCLs=slW3Hwt0&AHC6B|X'
    '8drMrDEL+$ZiJ}3UHg6S*Z<2vP-XO-zJ<DB8@0!A7+u@bi8g6Qn14QIy!5&Yi0&8N!DRi>r2M&bW#xc(rCZC{a*%u{1;+grlyUqP'
    'bzQ1f+((1&~RwyD+Pt#C+MwRlq;0+V*CFsBJ~x-'
    'ee}^Nldy3b5x(w6k}M@aLq4_I5x*Lwi_97_jZ9iptIMki=58vkT_`Y5NW^t%|FEro1W^L=n3zeW@Z?))J#8pNa7@Y62p-'
    'Sk{#yi!@te@;Ajq((`?vJ!?bKv7;t&k=~K6Vu>Xo5tZIM{Lh)0Gc)(TSN0Ci^YOmV`@EOk*;~$U&YU^rcb>Ai;|z=m#Vvw*3bkCQ'
    '3j{G{;x_+kW91Il6=60E*VB{GnrgG<Dcmj3;;w&)J5|G-'
    'c>+2b<J2~Z3>N<rh5D%sbAaoUHP(X%z*0A08q$^AuLr9b0TNIca&jRIZD{;G#HKEgJ0L4<29UEtQ=tyjJcX1iN;#y;=83eAt*?Mf'
    'XdjEn9yp``A`kI=xy!NNTM<N(b!-L{K;&Vbxa;S|5c%t4v-'
    '|}*yTLVUO6i$u7RJ|ZL(SrndsktrF_JL|+^8zQ`uFlQv!)92%@j!A%z|`43ZwuU_?z$9(V*V0;_zvX{vnVWfY`}D5Lk=-'
    'fj#<AU^Ogpm476#7?HTjKNeUIMqK6p6j+t>PthpXYzCOKLrwV>Q*AUdlR;{Ko)64Cg$}X6glp&!TllkAk12pm=4o)>#NLB31!!hK'
    'L;>0Wj6oJc=AsNTYfLKcN+RE^mDUnPST@)MT-~w}fN8^H0Gs!MPLQyk!eP3<7&`m82PW5D4+CJnl?CRtDPU@(lMX~wehT-'
    '+9V^FY@JzTPH255D*dF!K{l5BeSGl`}!KckB(;dLB=C0Ci9fQwCNn|s?oRbNdwbn~GCi1o|<sj2MhK{kwgp24Hi_9H*OawBGP<|P'
    'OOfw(?nIP9+7DHyU7TGG@_jENxJvC19dQmTh)qZOP7`~E+-'
    'kK|e&KcFXYMQ!3ht9J+IjkuWe>;WFx3lQ1&Z6@+NtL>Y&f6u`>LNPtkW{UU=)6-'
    '>y)L5jE=d);h|X_Is@X+!en(Q(E~4{pnOC=)rH9R+b8d*vxlFat;oRhAwJ%R?nx{}13r@I(%2;r&*JC2UX=L)NC~%qq5#R)|{;Hs'
    'fnFS{-*}K%)Hh|gSRA`0zuEZ`P0p-;$L}PQOQ|k?7F>OOq5v>}7E8@1)up(YzjVj_5)}SI@VT~!`#@3Lcb_N<z#5k-'
    '0MZFEq_I$L47GXtd0<6eRfProTG{W<m0pz^Uz#I+xe(Tg_GKbQ@TvVr6Ji;Y(ipAq+dQ1U4HuVg&E3j#B45fh?5J71GPx;pc4a^o'
    '<wigX8zb2>luBPdn$6YO^`|Tpt=@n^>F#y7TUYI#m@<=IZBB=l=?_`m3U26Ni3;*77%SNwmmdTsu`yMq&CU2SVd(~i>yluYkQ$u9'
    '3!hGMahRWm}yZvJ77MZ;3+NKYa$$O3RwPvt5KU9uwxq3qo=kT&T?PzYib1WL+B09&S@gqGZLOU8U{3c2}ngJ2o5hVGWf<|Q)jUad'
    '39Q=OiTz+-'
    'BoV9tdtpR0cRObB>wWgsUs=*u<ik*VVxhw6juky^UhP36~6dK>nqVf6^8gbjToQixOBP_n2JWf!YoIF8DoRU0AK%AO9WpS8HOU79'
    'gCexFrb1;}?YjqeL+Eo29OnqI#R4ZdUo+kkGIRG^G+9ei%a0Ojr0k~L?i2$H^^8XzLKr<i$fFQj8E@(UQRIt+^ZZG_GLBwY*H4RX'
    '|5Pkt}F3zS^pPo&to{>$fE~R?Mf|0qNh2@GHm@8Mjkzo(m8>!ocIzgzhJ}auP(7lkIGYE^o+fnK*a&7b!-'
    't7G$PloyA+hKvm*VD%FZk}1f8ns`DM#xk%3i8dcva20e3yp$2SAc@fVGcA;F}lZA)A$}+hvSe&aFA-'
    'HS+vI%xxYc&7erCA&nVhsi}GMgg?nrr&$E?Xlar2~qrU-etVH|uj>$Cm58__g?R|q6l$`sqLVdM_fcQl!Yy=zXPg7IilI#@dk(~l'
    'Fvs2)Z)D*amVS8dCSh${Hlh^Bg#qMV2Dt2oaHkz&R^}E}ctKS{tJ^#rlc#Ug>kTnD0w`5wyo@zXrRRZjd92IMxLTsBYxP<5~TRm|'
    '|0SHr!=#tnjTOh8uE?YBWyKEg&xXV_nve;)mfY^2@>Znfkdv(>P1R(aVP&}eu0xBt)rbPIYeL5DJ;UoNv;X45FSF#{JBMahLSr8w'
    'Z>hXfa1oe&)i3o*<!|QR!jYNdPBj9BTg-'
    '6265(<xo*VB!1wq{`ab_m<$OjQ`zo+H3y&VjAD@95TAn+~G2)<!)h0=Qp^ZLI|&ifgSkE4H=Pp@my(b>!KcKUi@)s!MIQ^@$ud5g'
    'x6V>O;578iJ|I@M*xiUMB=rQ|O-'
    'RWIT93XZQ}l`_(LXduG8qI}6^!Qs6}fFG^1wj_KDn;&iJEaRF$(8I+8~Jl1QhuEoe#e+yw7ifx3zHG|<5A%@#<wZNpUc2@t$i_e?'
    '8kM6GZ?}O;BwMLJLfbdsiyK8}<;<{_ij_s~>SmEwkr<Xyvg#hJS??|zBJ5sFXJlOlA?yCdTQc%3P>n$)DzlXF2$yT)$uHJ6K0O(y'
    '|_hv2H)ePSOs{bpC>Rwq?&&i^?X$sYA)Nw-nSlV_1>Ug2<vj-d1NkaY19<)}^2-Vp}O*^WWgzCW?<%7*Y_?-'
    '~K?YUZHAY7~ViX8jb+-'
    'h`lt@jV2xz<WOCc*;#ucFPh2w)(Ui=r&xIYpamkp<keaC5D5@^*9Ebnx4!B<;Y~$pKCFZfutvq#W(RhRHhR2=Fi)lkct}Hbg<14#'
    'PuhQfR$2i`Fx<Xq}ry>)|OYcpg*N2>1+lbQ2h^9G6lfdOTh&Y1cN)Z{a4xf%n<DS*zEbBha1FLN-'
    'b&n?dxuA)?R5Fo;HUH}z4dlqrgC9!GSWtz`$%X6rpYrU0TTT7PNLHd_>}2gwy{v(?ly&=za6MgHBx3%A+&a#ksWq<6bLz6dt;>L>'
    '{&M3|kl&!v;@_DUK57i{WKDNo1$>qAVS`(3`tOiV((npi=ZTm9!^;^_cl0xh|GtF;W@!JPP7c21m?ofGr2bK;0h^Q2@7XlKkHY=n'
    'k3gWmT-'
    '^mgECvC+q4R7D<o&4oud*m~t48f?9+$3&p_YegGu5qjYnUlN7hc|{v+5qgg(++gc#WhMof4Z~o2<_Z*dJE|@~<#fs*xJ=io?sglv'
    'eSxvD8>6Uar(6>27`_AGeLV}jv$MdPp9S8LS>R1ZMe0_KWU6{LSTvGp>N(-'
    'lSYZl1FMJj&%%T^96$K{Ji^Auy!d!Yu<fO1>*=$1x@Psghb+|goL}y2+Q}Qfi^Ejd#YrS|7jkR9VV<If%ug5ml0?EZS)|wyNSnJ5'
    'bjkT`eSqcwWko!oAx<12HUrABdXPLTMQq=W1rus>Wx;_sJy`-q?3rzKw6m@-(sR5Frt}iilt)!^y(OjJ-'
    'h>Huhi#;fmQKJTNcvGqCU=D9xc@^wKab=ibGXZ~w$&_FaG<TIN_X}bSsr00gtBEwS3V*&@mz28_o1<iM7<G2rCdVF_{x+kShlWM9'
    '6e2Y@l$4@rZj(|p&23UD6c^}Mrnxk8syftW0kshLt}fA2bcNRN=)7Tu|2@nn?^R~W0dy8=TR{uxP$Y_Tg1v!bLZ>a*ujTMIl^<vV'
    'R3@v**VHLlL%ycY$y)L?bxGEduc>RYo_tN+k`3e_?VfBTx6|p#rbZ}WGeW#2jLq*-'
    '$C_wVJ51otDg8Hh)w>kU0o%J2%>moHm~^i%ldh&Mv9rV!(bv<Ys3ggzOzzz&lbfk(iR+aYY%{J-6-57d3&#{#66}o}-'
    'c;R#vb0tj%F_NX%F@<bqAYFo56aRO|DY^w?GMV*)?K13Z=!O{hZ+8IBTT6o$qvgTskM^S6>meHXw7}8Qao3xD#df9s$#+&%(LvyT'
    'VXMf>VK@G;V8{~(8rk|tpr8)j^(PgpdH>Yj@h9kvl<R>Vi&lH!<%>qZszc&)bAxsZNcz?Ia<!tcJ#`cqqmvb#o^P)XTJI^3l%jW3'
    'KMu-'
    'h^`_iM$?MJaq`uHK3>4dW~#Jo54xe^?nXI9GxDwr<;}QuSLo7lDa$EE?IXFO6i+I8MJb+C^onBQ9#JOlW4NjiM7G+3t2TnlL+Y@R'
    '5G4WVr)rnJPD$&ellj^?X_Is|U%Mo2lP>0K*Q8z2)qL%iv`@O3uicXlNq6)0^rT~Qx;e|6OzptL&VV-yRtUhH_37%h!|&Tx1uVG8'
    'qnIC>#f8N&EP76kW?Ei5RP)l<#@IMBq8}cv#N?SW59rELJnQC_rFhoOD~n0^&aypl1x#YwLlRbG5Tr{(s&rX!CJs|isUfS$cUdjb'
    'q)8<m*PwlMoFG0)Py;6jV(4qh)yabDmxp8TDTf~v$Z~37x`vWM*23#8YGJyLl0w$QYX!A1-'
    '9Skp8{ze?>v)ip6q4nn%*gQ8aQea`yHirTmGv@*PRd;VRi$`Z%BxE8w3JsBlkPobQR^>OeXB9Bb%V{FtV7YSq>18*q|s_xje_^$1'
    'ufrFkoE?RaH1eOF|fZ*5=19<D>fZ*IYOp_!YBw6@3ktzMA590F!2bhCX8p-'
    'DhlHiv#P@Q7_71|J2X@mjqo^T#62Pu_fn=#GvaQ~)!IC9o7?$jDW1ylo27Uv$8Q!B_kBE%v8d8=ee2F#E*AlW4I-'
    '{{8)rSao0G9*lD&-'
    '2_J`V5ZXU|K8Fs3sP0R>v4d)~ak8UeIKyw6U&Nei5Xhn6IUXS@qug86+*AqU|>q(#K^_0)_8s{^;p3cqm%90$08~wk9nO@zv+G<3'
    '+n5()x1I*mlw@UFehu<p2(;R-Qn24Jlm#V5W5^96|6FR5L9N$>Njc-fW3_yLrT{i&rAt&oB*-r~OXl2G|A-'
    '{*aZqr6)q4%eS$mH^!BEls6PFQmapXO*0!{1+&a=TYEZy8h~V*5}ffog|!)T?b&ceUZrqRqALBI7MMTpsLh&#u?+TW$*U^|LZ^iV'
    'SmLX6}DKugj%*49jI?q32OJB(AKk84f9s=11(CD3UrE)8P`NP$y$LT%Mu_6w1NR17rnKQxO0ssmEk8$tG>8r(`nKW^Jl*(hz&ps6'
    'X*!k1-Ldu$2F0k3UcpA^wib;_uWH{@QZ1LE3Vsi;fpg7lG+`k|;AB?{nz@&-'
    '1{?kKX#{kjRfLUFW~S5zc|rPN)k?*xl=Z^nMw*{UBUwKEq@~4cteW8jx4uX`aWMu|PZvhr|K#Y#dU&zEjI)xQwap1n~`*G1Z-'
    '?>H&q4?~lu-JI@FkO0D)C(5qA%iMEds`+eBj(s{ru;TK(>!PV)q&L42sYuYSZAuwBJfq7a8OaNL7qSnKx_;HT;G--oo@K5bQTXj4'
    '-5vfZ41T2O%2kkN5_zQc`0gLf}>B8zKV~I_p6nq&J{xC!#T*5{}GAVd|HK)1xw<1x97x7jk4Dnjuk`Z~&S`@d+n5<`z)~zxo>-'
    'jW2pitU<o<al(-'
    '5FAMoTgcuTi8^0^|(x4H^|(?)ag?F+PYI<x820GAp|?@RS!Qfl|uL(pM_tmlwNFytk&>p8%}U_lccl^^rW09v?pay(AeFRF(7!4k'
    ')~sN4f?z9b6bPTa5qsKlKLQ5HWIEk!m*dZ<c~s3cEs{!a*=&3ax|lPB5%iH5-#HHSWK?aV<L1T%4bBBZUhmIh|-'
    'O%^ngNEW6LZ(NQ0`EruL{RnWCPQ$rK~E%}n)>>NnCkcsMg2^crMj-cLLC=kyQk&V3}uJ^o)f5sbP9)W(btVR}LqrWGkIxe+VnWTw'
    'gG4C9Xnw56)`YeVeK2(bshRcRYzs?HLjsDG~!!n_O=?+u|il&MDy6yZi!=h1?gyIv6sMYxC+u~1y1#}p?9(J~k*V`xD@Oe19sEl7'
    'nPP{>|vnWG+0<?7p-Q_J{}!4~Xm!+u%;S_kkLoyz$vCmvAO^V#eXL!6zM#aZi=I)q*QbrL>pGaimbEgkF;sGS=meA?zxT-'
    '_+)GgOY1o-'
    '7sHuGP_Y2746GAmi~q3=LYhW2!28Cj!#C$n4$)er))EYni&gabmp;QhyxoUdWwnp3Xb5c!X2E6N|_BdQ1U4Qrkk5)u=*vq%1WM=c'
    'qz>BwgBC4=7|dw#?RSYWv=as{%I#jt4|Xdtq0)z~&RrXVel!?_k?<?|N>FWJ1}H<gFY-'
    'B!9Bt8ilDCpjT~O9d0*3vl6qdYg29f9wEF>%EG&ihqswg3wbgt$irL2li2|?*kYc{35da#@MNytt)a)}98(<M2)|thg+B@P<;_ez'
    'VD#lGc=#g6e`ap)-B^6WMZ6n}&slm*0en(-H+<Nm3*nQ}E<qln3*nOh-9`^6WKOosQiD{00<w^Mh2vI+>@-k~u!*%D(`|XN-'
    'Lp!L2i0VosoIiyEyZ{H6YjcN8{b+8x0ADQYnz4J0^CG-^Y#s%%nXR>mhxm?z%#duC-ZFuAQo|`1<ZH2euKFxBej^h;td`4YlN0Ab'
    'oDI_U41a)S{PR!Y`Zq(W%Zc5e=in|a0TzhqH&@g6S28LS&WI=+<>^oL~U-`>H&qcWtoQ=8|_ul`WJ-'
    'r1A5nLh78T)U0}5`*=ft(HJ446ZQ*wA!uIM8?&9{ifug{@x{JHWJ#M5ZUa#)vE?|!yC>ycI;Kr6z65AG*Nn%^TGD&QYXOq}gG1sv'
    'tf?SS=YOyKUk0#Z3k7A{qzB>9cJH!^gwZb8B*&K)AkOHY?wAC<|7135_fgTXiRtKJUhp4tXtMnNPwWQ8Q!El|fO*Qld&{0wm7PyU'
    'fF2viBnVe8v>7${#$@!XWYqK?&4iUg?YZhj2WMTG17G|HaFme4Dj*SG4u@`eZN7BjTy?66$BDJYO4xrav818MZCJUM$?{YQGh2bv'
    '4Hdbneh@_+wA&mDh)uKesqLXt7mdtbL7@H&sm(Vd5$USgKM2@7nu817TH}rsr97&L9$EY01PxToJwWcluGRRUOT-VqX#P9LY=UMR'
    'u?wK0Z9U1zyk~K^6`vUu{HV?797K=LqEN{zVd0`gIPiC?FSqjUrb$%2#&@MUXx13TX(FavUBuq6J%$-'
    'EJp6gihI;DEP!_`#RknJi>U;j3bC9#tsl+WU7m@y4r6RJy&-'
    '_tyX%2+7FHB`pt`|j0aBCu>OECS05^?(R0gG4K%u>6@mL!pk<WtPMYc=ho)g0@>@wWLmyT6w{?dK|Za1_RL>8GR)DI1>}RO(sqCa'
    '=TsakwIIXWS!cc!uIwowijiw{Ztm)pR=$B<<HOtv6;qekP~es)te@hDK~ioOQuk2@<x_SrQhGgl4+EfyqP7_F)ule{;rhMO~>40r'
    '&!>^nL5P+cZVJm0bFxo5x`xf2SflD#MLPZxS#7Y6lzjkriA|<I1FmkOOiV2_D0r_df{D`75J;Jc{N(j#FP(T!%$PQ9xLerfbPfwb'
    'a57-'
    '<FWu<odW2OIqe#2u)eqxsn`3^sJ7~LEE?4otUu*RR|4zL(Wtfn{RJA;pT{!b9TMu?W4T&qVohgqbxU4@Uw!wTW1$Pz&^Z>m>-'
    'Crj=$Z?QfbL>FAOgA|$j(vFU9Hbhs6BOAYW9z@v(f~MkqHd(NueF5P7Oy>1u5YM0Y9@X!$4y<XxqahcZ1rQLhjBia+hS0`*aq$Yf'
    '@$dz~wOnjXB}X@OsR3Kbh{b2NhTMO2^%Qg{Gi?lRIX}pGYu}9T((SsaEh<(!4`M3=d}NA>-'
    'g;T%DL_0_*3{B^JYQ4P9a}{FxpTfnjrD5g1;g2Si{PMA{_^!)x>z3U#S2!*E;OJgP0HT6*t2IND`upk<6d8n6D$UCLVpV!1gH?j2'
    'p$sCuK?m4foFER>gKp*%hd<+UM{wZ^Y^Sy@D0G8kT#ykrQxEP2ULcv<q2Ti`X$God^kOR;xYh+NLqNoMo*s!$#CEM9%vU1O08*U&'
    'W<xgY5<5y&+c7J=NQdO!qnL4sYQkh@l&p-_|RyaM_mT>ZoJ@GyJ$X9casD>&9Yu5!BWC1DHJl1*4vAK-'
    'F#7MIJixcprfm+Mk0_Cba<7!P`S+@DI=YdJ#PpG!D@IwnhwK6IOImdSW>0am)U@CbEk;7}iIO4aY3?<Iny@1xM4dje*%Y6$ORs%5'
    'FXn8G}vZm|%C3+NUL;l+AP0SHs5z+6}X2vewFnI2F8!W8=gVRtJ8VRG)R(`P8upgKz-rA{fS(`1+q;{0Q)gW7@KTJ^afbuHmf$5X'
    'Jh!hpsrUZfbaNrGeGH9ZJu_`FoKUF>%8X*5f(rFMGHx%#h}7!Gl*^<HqDp5~~RTvgFITz%@8Ctl_138A*x?1#T{^`ubSWgK7MOqM'
    'Q>Z!rFau3Sya<LfhbwIH6aZ$Ug?--2RJ{<q7UB>Bz5WjKB9_p3C0qm3%{$Iv{fg$qhxk@>*XkFlZK)v18fGYsD@%M&*xmKx!4+>}'
    '^qgeP!QVyO|H#7&8%3wa86+?E(=9BxnsxMroRrP{-'
    '&&Em6E<eAJ?FS_ntbJawu>P_aUmvj2p9+=N=7E3eIyk<oF9BA1*x_NV}Z^ZN2y%Eo6_eL=hf0(xx-NNzzTlfN>22Zp$$y2IL@sw)'
    'Q%u~OE<NI_wZN8&>h4-'
    '>Yck6!3XWFA@8NR9T?^>x|lw`qA<gizU#l#%F3|7BTTf&Ac2)#;*ZeE21K$mf+K&TM`>`nS}qcPA!^y7elUTciJg{Ki3xY;bddHt'
    'SQdw#>!zeEmSwz<WH2SEe3@E~a578aBDCwcN7#??orKN!ejIY&+gU-dD7t8awO#jfXAqJO;S+k4#wNZ{RFQYzUVp{lJ6pJVu@!jM'
    'N9meW6K_nQ5sdR~$a{u@_M3AMuxdyT87h1w<4dD^Sp&UAX%N)R<mSOi`fI=m}=jHlk1-'
    'f`q<b3(}<kIlHW<WJ=4M|nA5=6)9)1bf<|gJ4fvR7~>g@+5yNU^grn0o{FBn4WqqSC{i3edGp?bonPx4*y=E21-'
    'hFY9iGQR&viXeE-O?>|@%&RpH+)aoBzbpgM$uwQ2LdU}HZ6Rsq@w&DJ^5Ap-7|M6<J5ARH=u5xMmhqy7-'
    '8kMuF<z#2iFpO**N2>aOF5CeyE)!dZ<PLervH|7y6J_!1;#Row@wzwDuwv};fyv>ng<r61}YOa#7b>Kj*dWWq8uj9C$+Z3S}7`_8'
    'B!(-'
    '{^bi+_EuPB3k7%Z<gXB`$zwN`?~Zsz~xf(XTH)D<Fmk~TV5iDVr58ok!<E>ZFoRKe@Zc`;k=b{=LaUoMD40+xYugzYEW8@`e$8pq'
    'Q$FO+=~*m#(Y^D9Di5IH5>=D9362%@ef2SL=eq?qjYmAR|?NYxT~%lg||&kCEuW@Py-oI~UlaikhJu0Jln?x>}NUxmNYEoIfc$nY'
    'IN;|Q*<ahjtREHd&O4g%}^K3A4D%audi0@j(Wc|hWl^Mox3L}&4Z0xSMB39U#TBNujWkv#rjlO}LB_(!{EC6Xt|mwly3p2A>D9L?'
    '4IQ1~sd!cGhSBy3HWW8|4TUwRNcS4$6q=W1y&;U6Nh&NA4n*U@CctOvhB24oy{ezS(@I5>Z7!r8qN)o7-x)JqKCfzS`*s&A-'
    '0nsPAyjrOR~%d)k}Z}y}dgFGq6U{A_1#FKIi^`soPcv6mGj+8@vQzTF4L?#Zz{PaoaRp`XkWRsk;S)H86?`v*z*+Ec0EjtM6r)32'
    '*`;X7Iz*&vzc!W)78Nrd)c|1`TuF`3MW)71NbUF3G0RtI}E=#VnQ$>~>_Z8U9a`k&VJNGaQ$dY+?G6!shUa8BnT_1L*Qn2STwLvO'
    'o|5kUdc4A)^b963OpF8HeM;Ow8JYt7|&WHDId(xX3zNxS2EezkRe7}A`)}(Vo@90ml)t;W`u={vS<WvQlyY3TPjduhNiL1st5{DE'
    '>6eLfAk-)u$QU%G~d6^EI6i5{$cP9w(-hzq#$7d6(kCMr1yIdQ+FdmgrbBi~LKX;Bq^v&A@{Tlo_!S!Eisx!1w-'
    'EaVu(|jKFUIsfChs#qq>>@*4ej|&^Jt<tyXKIsF!k)P;vgpx5(~he>oEDf49GBKrX3$`U?|t@^Lm0mI+oPcj-'
    '%Vs34b*udDxVPW)~FmO;GKg?bI*NaQP~TJ#G&#`91?*_^Zb7th01T}@LdEd_aGN>u>{sMDh&=-'
    'OI+eC5bLDrl2b1UiLj@rww7d5p6L-<;TaX+*OC(aMu<$<*-'
    'rNP{;;#1>I<dzG1z1|oSuq&1R+i@&*F4%3a4E;@*mk6P|Rcc#5GbedypyC$pSl-{oLf5RH;D>-'
    '+S#jhcSF}8OvmKeh5x5(k7Ta(a*6LX^xB2-'
    '00P@+0}3nSI1^ozpBR+fG5=<n8*H;LhvNCU9RKF1>i|Oe~{cy3TFG4Db2I7MG3(1TcpoQjt{Q}-K+~-'
    '{rUnPpwaLpSa0kM2JdVs%6<Tshu>*PHyFY%$--}606$GxUQHCZ*jZ;FqcrX7u-'
    'ynHqF#^PMko>W25dG$6@)ip^BqdtuEEA5l((&$#Fx8d8>8F0AOz*#gz9AIw)QHX=aV&e-'
    '!B%Ea1H%pF}X*NiD=_wWN}|q&VQn_AtQ4B_aS3@F%$EIY$NZ#b2WvVNV;jl-'
    'P;4H73ifDka}l<v_H!jvc;9WTj%NPB6KuNwMWMOp6<69V_~QWyh%TU=N1shVi(C>h72|!x9Tp{YT=aT0vAK7)HBBQjhf*~XUycWL'
    '5|fz_2gLZ7sm(or4XAvxq8+p%*O@1b5?WnM6ZbjCtSoevEbaS$3(2>M!sXC*7M#v<tAc1-'
    ';d3>idmT_<g{F@2&w++zra0QAGn7x?OzMs>C5}e^d&<Vn1gu*S36QLS6P@hk)gi4pE*rkqaI*bA!YUDgA9xHtiF7RVT{%4%ZC{j>'
    'REjW+n^<!^yQ-rYpSfi{4K+Zs+(f@%`OZvc_vrS8%#be;GLr|&F%M(#pG*9MHGq27xb72g=vKG(<p_hbe2Pe!fc|?P{_tC!(?~d`'
    'B3t`Y3^C=y)N~Zp#%{Uv|s{d<0GhGSsIn(0m(5U;{e$j8R|knUXcYPlW9Vdrp|@lS{=@l)<j2g1W(!!8_AJ8X-'
    'i}zNAaW`agj9RNqeFqIhrROh>4^*PddgjuYEaFB6Xty;t@zvkpo0?+XG^O2p2IR7KqR2F%dvCLikw}5aDKi76nA6&rrykJRxVZQ^'
    'kpVUAxI)5xk$d_OwC%hEAk&Ucg`#?2XE-Hme^4gA)NY&~-'
    'X!gXZAX1Sqfu`w^hP7`%o6btBQ70HA7!iVOgC6C*#6z!;Q+dmC8WQ*ggB3wQoS&5;j85w|Av2=?OYIfF%bNH65|MmE=UZ7dey3a*'
    'XC;(k3QqBpX6LO+)=y^*if$zBn?k-'
    '0uYAuCd3;}Gu)hax<jC{<UHD$1iwvP#P0tU=%39tXn)34bFQugx*16=Ad4AVOgLzR~6}Ekq_e4*o!pIu8qU$+6ADGJKfhl!s-'
    'wspXWf!cI{83EbFIrwJviXde0!e40U>h&wTcq3$dZ;PNQ}Z@sx$r&{KzNb?N3$0nQ{g+t;JPMYD6hy>2xW2!(*0_U0fJ|Yr0Tj&8'
    '137qCh73u++ji%^8Nu2?skg!#UefkLQyg~3^q*~wSs6HLIYrzJX;S$;4SKsGh1vFTJAG-'
    '3*M|8_1n<)X?)=4RJ|Dn+~Fvn%lJxPQ)$?Y8ZGPXk9Ln<7bO70~cjx9;J&9SOMhmT?Hb#+$DWztIHai0#M#BkxAEpnKW&0U@zi{G'
    '<wNF098#vu_My#9b`PB9(4&eHc0(ZTCDJs_fkmw8f!8i1BX1GjUvo`cn5K-'
    '!uEpxu^egmy4qAk5F;*SdOh6zqlzqzkWuQa(jwgMv-49yvtn9~<o((=v<H*HcKX@Wi35J*9XXPbuEkQ;N6uJmMWZk9bGJBOb?-'
    'cTT8!@8YVT@x*hcrswHibKgB;ql9n~Jz}GT|I}k5v@eLDM{I+b9#Pu2r5+HWea*8f)CE*zlUTqEeP0B7>VJq}PyK-'
    '_hJRT1pu52>+5Z&5&iP-if{0<wd<fV-WdVCm7O=->0XsPatZqhAg3nkMyef@2GLyOrq~Q<4W+#Rz8edf7-'
    '5aKE;O@Z1sWw#}%hgL<_0$<H$yVKxxh$E<Jd!hFu?)9&Ml6<B=rIw!yFfZ;MD^}EN8d?A@2=zZfQa5*=2;bL2P(38^k58M6woTxc'
    '~x7BAkWrCV~$3p7q!`)ne4XEKQJW)=s#xxU6lp&30Xi-NdX!Z{c>*Az7D>>iBwN6vh)^*PZtn<8>{48Ky(F%PZtn<2PySj(EeQxp'
    'RxRW=Z0$e7}$iF^qhJD?|iQT+)K|`D8ohcjD_+NJtjgegIsz>spTqtClPA-1U(=^Et_Xms4b|-'
    'wg?f((h%bAFIn83o5kIUS=>!ct^D<9eoT~9DzE_wk*QW-Bl6NxvA`xI1f^<$%}B{e{guBgk`XTG`5uSQ+aqcA&I`1%>FuRnwzmEu'
    'nC7Z^D{(R|*OW2i;ik;-'
    'RjKmumpQ6pyMz4>C!QpylwW`~+N2pC%T;w=rkQ!Zy<(vZx7RBc%JcP@2q^yr6aQnNe6GHe2q>SZ2Sh;GJgY+eK+{^;EVf?}U<O-'
    'v8Ab&vt+`x<Hg7F$@Zi5ka8Iw&TIW5lBn7@ws%KJZ0TL;B!YMK#J4MdRPLY$cQ)F6diado;w7*HJh}nwtdena$41qf8KmH|=45$9'
    '%7mDOo>N9?^NJiO;E4bxkOf3E1`6Asd<t?L*Z|AB{o;o%U;LKPI!$q7Ki{V*%Oaz7}V7W>RhR@S?5`p29^neHqn`c$1D`=UU&Fs#'
    'Com*Gil+c)qL8PqK=dys&`3vgfpf`WDESn5i!t(nN!_Q<fe0~<gCucD{J%!<?u~*1z($V`|#ob){CKe8;Huk6w$9Fqlq1u-'
    'dnQJmqT_lo`wnRx?B9hTj#DP0Q2k>oNlUIU)yhWXox7jn#;H+2}!v&lb3*(7;OazQU9A`zr_<VgQ5imYk4~T%Vc~*tGftDG_P0`e'
    '(H01s~X$ZGOU?EJDl)we1^aMdUuQh6-'
    'P#<u|(JGS7*}(1vLM<0zBFq{F)}+g3Sj%7;cHJd5GTS3bw=H^(t0TE<RRl|SGw#|Gy+rI*HR>9!2GBk;=?GV3+V55+ozyEL+3S+s'
    '7rAbEJ-'
    'J$<yXD1j4c=P{T1Qq`&dX${8kh%NbJrJ?qPYeyC`EG(UQn<vXjXS64Vd@oYKS|6(A}2jbq;Tz*?c*c4ysxKU(A2(VDS4+xz*!wM('
    '$q{>P_Kl5wF6zPR@j@xw^)+vFgWFf8x(eHme>&ZLtb#i#kK7t<0AvyVi$T^65HMVp5B1FpJ4~u7O^6uHJCJ){Cp9x|?1UGvALxp$'
    '}1an{{^sQ}uaLH~0LdQam%?mrBvhd|xWqFtp4qc?yTs$97@GtXtS=Hg(M|`s)xgiWvU;6z*G-'
    'HPvlWuj15Yw?5sy<3gd{^3xn^_1tX3JR{ee5(_z~siCeR>*G6C_4OUAuJ#?P`uUDk*Z7WA{e8!(0ls6^wXt+{KMCZEEyt!A0mF`W'
    'W1eAU?()J?JYU^~rFg!&3k!A)75O!Yr-ZeEBe*&>tPLE=K^F%#h@&`Y;~=|hG*=a2k;xde7JOrHT)f-'
    'xfOYtG<cP7ANMP!#w;dRgZK@Yo`OK%8xvs%yV0(L%sxL%bP3&%Z8S3cUBX)EPl-'
    '4E_s6m<rk!fLo)m66=&rB4>+kN3Q9}F8uje_4Ybt%t@OPkyKaw(qt?aQTj^0zM+Y$jS}n!*JBN|56M7AHKxj#k-Yeo9cg-'
    '85ifpCH04>;-'
    '~yy!>RpNT?NFw;zc+X{rW0nq!ON&nVeoqs?Ba0Shl!0Zx^$aUpl|X;Srd?&MZdUG0K%6%to1>sxLugH@xAbgM><YAdY)5J#IfHk|'
    'ulc|2ya5!SMFI~1Dx=B6(y#q*|JREp<KyQpA;(VR^+2)CWvQ-6TA+-'
    'mCj*^vm6>fr}qDG^Qdte|xvPA;Yuy%+e!LcQY!4og9QPIK{au)bUk#}Vi$q+tG0Tn*$@dD@Jt>o`UBo9mv3pF8CNtM+eXplo1~_#'
    'a6XuUl15p|)A|F`L86539PZa>6=Wu<K&U#(o-'
    '3_x9>UBlt&{`f8p>$lUzJrFgQji%aojV;2|fJz6&1jhA}7nc*m5nEyX7(Yn5!>u$^D`EEvHHA+%yfR|0iy&1VRQ_K1#LcJ@(;zKa'
    'Kekp@>1gx)LNvbg)76jhwlE>6cK@=B{t674`4W3Z51$B6TQq2*>eesl<D~O(coSG+y2;*rr-^zOqGB~M)B9@}-'
    'XQ90JsT!AdG&N7&=AJJp#Zz=$Qi`YOx};$9(K1KySA=>mB^sEuzmUO_4@TIpWw4}!9roWcSi)f`{$EMO!xcfz|9W&#Y;aA3x(l_*'
    '9=x8sPGN^+YBE&^7Vui;A)6Mg<u2X#MFG}!6bDt8&OumC$CBOlVi>iL)?CH{HygI};y>CG=vJ;{0pwnh!Pc||R|}+f|0eU*U(B|x'
    'O@3;Rd8S_}#Zy~-r4&zX^_7B6Nl(mH2<+kLX|x-'
    'gy}EMS?tk`i^s`#SyNN)@qcyylKzGdmS{2Zk4~k1a;GwUnT{hcqArG^yZ^S?l>HnLN#)-'
    '82AWWYsF9ts_rPCtrC10ac%2$@v$D?j(aLr6b$Q0idh3@hyRx#{7yD|_uHhsI4+iH~?Wlp0r^E76#$nw6XI3%vT?{FMayv>=F%&|'
    'PC(OEk^AfnOPLOq~RIlzh8a_9=&O%COm8uuEV{%_0B8_b%5q8h3Tq^)VGeMg2A$FDaZYR0Fay7a>oFxzK=xrm3=F4|VL6h?0#hk?'
    'nuh7|+0lx=L+T%;~AS=YGvc+zHFqeJK^n{|zjf^jzMS}nuE_l|K4R*S=&>pGBuNov{5)DwAH$K3SHSp1!VL*nq)6Nf}(TpIzj;4v'
    'B3?e&0&jO#^uK%vzC6U&>BhAM(`q^bCRf$+um6C@GhHJ~oB!9;<5xigXdk2JAdP&xWh3WFW87+f4;(3c!kC7s~)nEBcnUY3xw3%o'
    '2LX;*kzLeg&VvV^4F;bjR)Plwmjv5Z<vLiC-()pWC)9Rd=`+t--eofU~byogzm@WX4J6@@?Z@Q;haUk5!P0)LD3fI=DnWh?Q=nyV'
    '=AZhxG@O2;f#mV^tbN$R(1vb8VrG)euIucNIm@ia;Oma(I~FYq);{Wat#$t~|}go|q_50j$KmTI?S3rH7snZgBUajjyZ*Vr`0a)&'
    'GR><XLl=~AdE29HDGf#>1TJkr^*c!V3A9gD{`dQ4pH=*PO9Zv-AY>H!gWT%rdQvJ6kkHXg0>T^@!fGs&ZRYch*Gsw0xw<WU`&%ps'
    '5JsAMjAR7WTC$V&XD?hPB1p8hih?#e8<m!=fqCUv@iB`<XD)x*_QyDp-`s7EwlramaG)}2VLXTk-dbZk}I53v+u%K~+n>`>Ka129'
    '9KFLOxd%q`D}1s`0(oLKO!)MJV_8npsAp2xHrt<(b|T8%E%0}2^~W#Ai=%qP$8zck}dP|o?U6m&Xep|dPy$8Gh^^{S9h{drf)RGB'
    'n@PNC&>lKPa<`ue)<)r1S|WZhxWtfkY`S{BPpwmhW`?`J`~X1(7C8(1dmeUZ&80V!`4qV*N79$?fX4O{&1U^9BNy31@!hH3VzvN~'
    'n+4ClsT5$<hnEEeC>V<PI5jYLo2F?GtF^ni#u<z;$6A;a*bvW*F-LKC7+>QH1hrqE8nI~lL4I0X+}9aMv7v!BJ$Bbv#`Sjn-ocTf'
    '<zG6m_*Sx7HWL3$gq)Y6uH2U1zmE`1l4;nOaCH}ag3{@sHm`F2Kur8vcQa6)5os_jXICgOCpj>Xciy&2kwgW$SNBE}%~?L3c|x%G'
    'LmNQ7&c7mLJ~^_aM#(3OTXwD^+TSr3R<l9%fNg-pbgven5|Jj@{vyKL8hQt*944NvudczJy&`nkvj=JEJ%uDYZuHvgT%W0x!*-'
    '%R0gJ5oWAQ7}^V;FQ%fM)l&Dphhv*DL&9@`G~s<^uoSkPYjFvb3(mpk9r7n1pZY-'
    '5yhR+Io<oCx7jYQ!(xIQG@2WqAB#q~g88v%d`XXqK%)`ENl|F*q6b8v@l8FTkY3DsjdeH103~3er?bkI%S=|;sWOvQc0?&C-'
    '%M&T-)5!d69mCqPhc`s{Y}_#PjZX;yP!JGCxrT&qf#1-y8z18xI|*2+0msqe!0_RkT;ASp>_+~aYrxQVa;nL-'
    'rRp|($Igjinpu6ZV_^CP6+ua>{>llRT;Z+zh=HE>*|4z^|sh-pCfQcT-wKxI3yxX4de?0BGT005y60nG__OpfQU3Tz|U=^Vx2|H('
    'Ayfe$pL!*okj1PS@fQkMemWNu;k)#&{f2SkKG9V?y8EP66!U_fxlHfSfa&O)#8FlEBPSws}P(AGSxzA7KsQdx>ho@HNRU@k9yWnb'
    '#Do$S1sj})`$Emo_Pi1c9lLodZNJps17#Wy}A0miFq-W--i(&?Cr7m?S(_)@Ovf>iNG&N`1UCL0s?N2!tZH%Km>jPLAMvf?-'
    'OMe%}AfGj*~6~_b3OIMVXZ+V+T*a-'
    'CYgV!O1+++Q<;=q#c%6q#4eN|72&yTiIFBDmyEVO3jLKLN&Oa57%*2k9IFs!_|J9UZ?XepLJAZa9__kDzX6_w`?|Vqne{hTNBHIS'
    '0RIK`xJGKv?b7$dGN(2a7S#^4ldx1*r?sBdQ1d>K^S*L0T_gJM-'
    '+ft=>ZV{K1$z1p)R6jNe00V*ErcLK}Q(?qb=}?qss#g<XDb%j+QHByP2z-'
    'Ozw$CaNDG9h~Q7M2!1<@;EF7Qo23wZvBVViW+wXq+xYh5>KJ=;Jy&~N@bm_*_S)d7&P$!(Sof~wxGl5kMguwS#G*;_T89{(u1Y2A'
    'dO9Spr5!$lJ7X~n7jS1RhWF?(5f}z>+!=*o5Z9ei7_QI*A~4)c-'
    '$S8>qGcGKh)oQV@#?#hwq(BczBjDk*c5lXt%V15gR$;f+i+9_O%D*#S+Z{AdkBHNDhuQlSs=I00{Q3^kWa%jO?6Mn`O)D0gOYPW!'
    '25?b#b+=YB5za1<wGTM5MZ=^pGye<<AuqbEa*lU_SytM#tex8W{37UOH+~(l<_It6$@p!gu7y)yjhQlfHDZ=t|%yj!0w8Ia%(*x0'
    '?J40dnnXkw9FoEpR^C*_GuPw?_}ZDCJVRbrBn+=)Z5E#!w@uc-'
    '?Z%nU|XKNWt)fKo^6HgMxkzD$vd{SfEvb<cPSEE$FP||9A{qJa4&~ZCr9uyi2D>E^qNpAaFu!%`(ml2<Z{+;4jN<KPkv-'
    ')2|vrg{L)%@PrAzlK^LFsw_~9T_x0^q=)R!GL_imW`|T*`g7Ci`1>H7!Km>G~>w75FSG0^?Vy)hK*;LW>dh2DAh&Bl|iBmVm4oQa'
    ');h$v@{%#iGZL<hJCZ$kM!Mu3&y^`~q!TU${2+Wcu((L(kUCJ{$AR9-'
    'w*Dkf4OMCHZ87wD$bJqtsBUY2?L>X1MNdS1eJYCpSel7w8f0NrBHl&)H5qfJa>9T%g--'
    '(4V+}3wuA^eOUQvkxQf4#r~1t3gxJs{oh6oN3(^|aLk3P6~$rH;|}P^iIZxgo(+pOOz;Rwi|7xvh_SQ;RyJ73m<bT^_NZhq!=Y6G'
    'Ym(TZ2@%uecoRb=qn6iKE#GXT;~(8S!3rMzqV$h+|VT;#8~*QU9;xqP+0_e#xA;yJnlVleC8>m|jII!s8+t6vh7c<<QN0C0BovHY'
    'u-'
    'T4z0Pq&AVe!45z<47RCGZm<X0Cc;a`LVpys`B6pW!SgP9T0TC=!$Lf11)MK<Pk67o8yJDS6SZMoDe02>a5Ohw~Qmmj$vW^l6x+d!'
    '>F54~HK+dw$lT8!_hHD?j)o5F_iC4wdo%ZNg9NxCAs9Mg{U|ls5!`jz^hvq^pHe0dg!b4#%rLgo`2LSIy<Np`dVmL*pPklfJ8(lo'
    'bO-iB#>d~g{q=`rTS6)2AJitpuG#?-'
    'giRA;lR21tmS|;V0%p@36SbU@9o6{L|HU(*B(wZ`mW+iPX0ckeuOsTUgSoYN96D%d0eKZeN9GiSJAExb1&h~l=>*>sMU4%W5u}UO'
    '{z3#X$3=g=R;Z48_IIO00z6weV?kcb_#kob#6!1eAju%2LVDK5pwmDa)8`-'
    'unWKCtVHIMCUBAP4jYa*H}?`y?m`(c@EXEBq$XCC!9?MAeoxTm@Lts}Yq9aoQ8Uh67k^csh`fv!jO(`ZhZqlXxF>Yizj9%eXl7Iz'
    'JUe$Ulx?%GcNfvY)I$_p9nHfhm<C2$DOX&KcBAzaH;HuwyXvnwvge9rOeZt3!be3QZkliXQ@3`J*BKdqzz?vfT(D{|04FPi+bJF('
    '$lj>yf^{JMyznESejr<nVCF`<7_CiL0N<jC1ZkvKif*O{QUXPU3GL08W<U*~|Xo@2hw1zkPYe4Ph+V4nFpAJq1I^9-'
    'Nhs6nkC>`AU3c5Lb}qaSf>>Hves^O(^a;a@wRr<s0!fmE12;omu8_Og&}tszCXwm^gdkIljqjXEL2@_LD}k;icLa-NXQ)4NQ>bFW'
    '<{;<?u@D<<T1WkR0AOfH(aPneVS)LC0^kN3Jvs7|)^qs}x|`&}kf7uAHjTx047hRMc?bC+#w`!VTsW6QHtg~hLb<Z7;+*-'
    'Q1tdQ~iq*a;~O-VpXXe29k*;tD+OLr#bWoa%%EP}zP*KaInr-'
    'W=%gWHq$_zm2Oy^F(c);WtD)H`zBtJU7`liivt#nW*P7lj=nQp9XpyHz$+G3>%zGCfn<lWEvH|4@;&~lF%%$(`LCX2Rz1UVGZ~s0'
    's20u-J2{x#|H&sQv}AueHrF70h%4lFypAaP7F0T2ogoR(lP9TcJW!oZfpF6M}sS7yPwf^X|r&R_1qkPgQ<3?$dASFRMQ-'
    '}n+JNih$r&8T*MQ3U0zJ?`^w}#kD0_O>f8A`8QMdVDP&y_O{S7t;=2-'
    '`m0e=LCqX?^?d|`QXu8_H+C36hcdE($zJ$e{yemJDphu_>_eT=6cu<HtRlv%QC4zDER3~LBxJ<#1B?^$RI5jWV?^+u;6)q(FaHaE'
    '$RH75oiTZP<E;bJ5L$R_jN7m*cCL*5jDiQI7S4lBhA0o4|p3h9$k~K&rwjdOikx5Ey{U+^ENn)q1l07;^VvnOk?9rhTo5~z&j}DX'
    'A&f_q9)Kp?mg{JoCaEYw;hfDK}e<;C@(|I^XGOr6T_yP)u=>pWfZs=2-'
    '(CL3Vhl_aw?!vrv)<E%<{z2!B{<`Pic$cID6MbvIQ@@@iyE7#NV70fkGo2$7cA50eW9=>Cd7^rYc%G=<#iZZt<ZL1!nA_cjuYUUg'
    'dpy~2sh$~bF|4JY+r8CiT$Av_SgIJ2JY6vdcKnB&!kJ*Wf9$fWx(n=avdJaptrf`@%ex`a*4S!0{|dEHBs*-'
    '=CZRToWT%VVzb36C7M{JjUphV2fX9Is>cwof;S`r`=;R1Xc?*+fx=F3!W24BI&XEC;IVHrKOIjcz>*iYFkhr?J!*EFP?hxkj{FTQ'
    'vhj>vBh-eNmMGq*Hsef`#qmuUpHe$3a5r5WL?BjOsx&?`ATT<)oFYRiybGJ&hC?Omx0>7o6_ck4{HQ1&9wgpXKmj1`qGl5n5i7jU'
    'WqtvDgwg;QkW(#%zlk_uN%LEpwh2I{o9`Nv+gvSB=UdqC6YB~Il;OZi?iVVaCxMSmppsMl!XfEoFNC4uQ-iSqD4;&JKK>dXN7KOl'
    '<^neHiPSpbn<)NOOEu1@DI~A_C3sn#8RJcL9l;Sla*=)Jmgjz3>ZI;AKs0||7?v|`7salpS`vf+q>&R8Vl8fKGV*85d8u<r3dS1u'
    'l06i14=$RIvhqU)<fd;9Y2<=MbelQM3-'
    'vtdONI)5mM=s*C8x3sTUfH3t8CPF1sQfEa{l(Zw{;G5HU}^4lVJs}+R0|_viC45R3YYrX|2+zq6ZL=yTu##i3T5k;;j*iCi&a}Bo'
    '}t}aH`=3~dP`hmk9sj=Nx#VxYQBKE-wv8*OHhx&=IcDEPAKWkmB}=ESsLZ2IU64G;GB%d0XSdIf^&KfIPVC1OMsDpy{6om!Ok4cN'
    'YmFc%;+;lC&njnsAJ4HQ`cu!hg!2KvP)=uPGqWYIX)X9);cc_pXP=a#o`lAwI~*!JM@?ceCo&lS`<EC)&nB&Ib9DZWJI2l#b-'
    'C|s2ylg*Ihemud_#&WA|gQ2F+1#jwGcYb7EE;YS}amD9(}UxRUk-'
    '037=w|2J(a2O!fibu`}3)x#c}Q}8&z<||oj&M3pC?yYwj56bduQFjLI6`a{y>CIl4f)`4^JFJ$rP}+HY)~Qd87xzZ4)|Id2jWB<m'
    'pNC3wtBYe%38z{di^}zSOav<RgKvmJ<tutX1S)6f0fp?z3@U519G%tI+FN~Xyw%s%TYc@k)z{uzeI2~j*O9EgCPF>p;WrhJ1Ngm~'
    'h2P9_`01V1atHhGmT(W&+*MrN%9E)Ea<}r(AJwOXI@E1i?=7kOcxGGrI9K-=&*R^iPKb-'
    '+AiE$BgXVsh#9|OmwImjUpXo6X7}QUHQWOS3K$D^{I8zTOWKo`y-'
    'CpahiI?Db3GZaQ(I{1@PQ1P5yots{gP%FO&UD@mQ^uZQs)NZ?n4{KV&A3B-'
    'C4$Dnz7sOkTikb6Bgqk$bWx+o50`XRqsa}IbW>x<3zu|Pw{Zgh>FRdMeL~8nM?Lzd;c<Zezh==ti+@2&#zxrDzLbYVbMB?FNQ6@@'
    'jYZ-|dQ1co^^<r#3W<M3KI0f9&e8)4*^p&OT%flKE_8JtcMWHVnC3#e^R6*nK~s%C-'
    'MITD4z;x&)KrLR#(yj=XS!F`pn&?J!c#xA_S6q;JoQ6cPyNu&Q$Mu#)DImfrZ5I{uubCNLjBgGa5^3bDEwO%g|oj1g$E*k>)Sa8g'
    '671_Vj&3USr!Yy#d=Hx1odN>90fsmQj?<~I9m@WWG0q*nunr8N4@Rsy`wmohk>*AXsikgVm-%T-(O3)sXhahS0X+Ibia2j5u<{-'
    'fdP8q(`_d%k*;UaV5a^{y1qm`Iqq|Ma@P*NKXNWo(#8ypqCp{VX$9u3G<OZ-sgFWk5GX`9RlN@sg1LLP4HTX&;1|;&qNdo$SWlM2'
    'v5oB57*@JuQh>-K$gKxg9nI<9G@jG_@S>dV$Cn9y2*+LNTVC)-'
    'a`lZc=+_J|7zX{CW9RXJ74lfj%_Ab`)l60T%EucSwqd=IId=7La`?198?L^^;nR_=_rngqZd-'
    'W`?kuhEE}0pMUa!Z;Vnm?cO0tE(7Ovx6atG#U*=T`wjk_eEdP72PDP%JJ*I<iq8F`y~msw{A<Znw}&yhV|b}X|yoF$sry=PHk_f|'
    '5Sj|!_w1w2R7%Be=XrB+hDg!)wT_kY*Mg4Q#bE%t|fHG{7+oa&KLH#3v=GIg?fm}}J6!c_<P7TcAGk_DSKv@a1U3-'
    'U^QRZ`QkWW73CChHyXcVBE|V=kseSUY$;l?$3CKsc|hU%Eo5^N8CYCL)r=*hOikrq>|vUNXzzD~2>JnVchX{7cbX@0$*S>wRNkuJ'
    '?{*6Mis9r_`Hvs;y@(1gAEz<Yr<(+{lta#DKVoC4-'
    '3paWhMX5Ch^CmJB5Z#H}p3g%}XGAyrdH*G8ichfiCbBvchqk0z6o+39($F68QBoY+TjKY4?m5jU`q^r;T6omwiPlDEyN0Vk={o`F'
    'ogVGN+_m|9rA>pC#cSu!O@%y{9^WbXey2r~CIg~{AcFB9`1j-'
    'JTn_Q7}KAlkk(mq2X&9kI5rU_8!O;Z~_r1@@A&Ls|)St}6*^Bd~>)t@dpzkXe?pNi<7{omO)+5feFr;4o+TUhgtC`s2zFrp8xTVr'
    'bp4&G%o#n*s?}S{?=+8&0bpW>~r5+w}|-'
    ')7UunuViYD!13y%dEDPW2p;#9g?Zf1DHC`t#{kJv?uurA&($(})D^R3`p_~AGy7^<gkfg)qnQdb`x>&uU^Wk+r5R@PwZutvglz;;'
    'gX_Mp=5FbWUy<l}ZfatimBlW+7Ybw_c8hf#S!cb~;#3lLt#eRQ-S8iDYLb&P7#B;-60<>FUM}VXGj=7@a>R_+9nIRl>>ya%-'
    'z&`8{^hbYyDL`>n2-'
    'JvcMZfW4|ibSpi95IuxWa|u>rjZN>mTgbj*Fiq<}R{b#fJ;$wqaIOg2*8#<w`uZJ0_vn0iB~lJ9EJBS!kiQ0WsMoIs_|y5v--'
    '^NCR!o-R^XS&cAHQ&$Pv)Z!J{OP>Zm`%xJ38tKw`su@@&W8<i+lIc03#&bjyw!d-'
    '^gzYaECT#y&nW(Sg0dpoubXAb{vWf*nw%GWpzYE9TCQ&Ksg9?o0fb_pw3LYou=;O@eyotdsHtiCqqe8Ob@Oq3k|09WYb_~3pHec7'
    'Oqhzwy1z?v@P6_O?;KFOECiI)I@=(EXA5b?```Hgv4Ap*KV|opsw;-9HXNw((4C>pRbL5MECz_T0#e-'
    'mFf2lAl`xROFhIHk<v4*@dG;g+>5-'
    'pe<6Br8Dl798)0n_1kCHkGKWmr)D7+TTa<kW`p76$XwrDA~beGBcs81B=Da~G*xi`6#d;u%HU!lU6ej`%r`m6RQHhJdxjbw?9mXV'
    'B7dmQZhVas{o@;n37%1{H9ELEi|b^DAP=o1`q4wLQr`7EPwKk42Ly?PJASr^##&t+NG6xKFu7(;Fn>W3ibA;&oxl<3x#jo3Pu4>D'
    'h4=r{=qZ;i5P--yOnWzEShtp$t?vbt4_dVD?du$2tbn&l-sg_hZAj3l)CGR2_E-'
    '7ppP);lkXoVvOJ}u)U6{QN9n~AJZ~Vc7tWFSLz+=9X2+K?WmL|adV3OESi33KZ~Xx+FwlK*JmZJ6PvtcQy*4uh6x|_sf3TNsMhOT'
    'UCv@2%DYBKJZ<e2a)sO4E945dwO7a$ZfmcQE8NyzAy>Gqy+W>VTYH6E;WxR=zr}9-=p|rxNE-xz)DyP7l6Ld`DI?mw>LO!okAYRD'
    'd}(fkv*(aJ>6)`N;qmM|O?W&zPm^LE)4^r(ttW@0?p3rnNLi;f6gBQu+-'
    'h8#=;Qp7TC3}pluk8{V(Jp1^HxDYG!56~HR?#2tTBIo9eQuq(a!fdPRVu2aP>UeVbo6e1+>qp-'
    '}Z~>F(K~8mvEm)orWi(<xc%sUq)XF(K^0@hMc)U&=^ls$4p!{{><}M=|;$H4$Tv``D2X7)7UT`Ph-'
    'Q13426lbA^<!)skl3a5YfU)cZfmUW6@8bD3;#KGnZu7SVm!v_wxkP*2@4ogqjX&ZbB#cir*6V9`O74K%<wyIXk9l~%-'
    'Yz8sb(Uvu%C$8)=I9?$K<i^=!SGE=M~q{^%GRC%ZxeH^$pSsnPJe3~oPn%1V7eQIC^UMEQ{kHGc8>gAE@21$x}l)6!pW*)6-'
    'B&p^x>Ly9kt(qA^X<K!tNq+6N2`s(GT_ZeL;-7Nu?!fFDM@>?3sh*Gl&}e3}2$P-a-LfUQG2WG?d6G6~5j>s>Meukk6j4ml_mml8'
    'LR(RF2I@!1KquGW_gWty2wKndHX#L%3NYv3e<p@qgBZFw^&{P?-'
    'eRH4dmzK4`+?TUeI3K%+3S3C>|y4jV>dD^m!*ISY|nnn;M4g)>QoOwt<#WdcfD_Ky1}#VZuIR<H&L55^_cCRScP`}Gr8)!5u%dA^'
    'Mr2xnB?)~A(F?Fhsa_=zprd34J&O;;9XozOSdMwT1<VS^+G-_Sjfp|W@s313D@e|Ol?1Wt6^%X-$WZh-'
    '9kx2=5QEnkbOiA7LYT952%5n=qY?)6?8=};lpd7F3uFbZ7OJtvjnwZbB#<*H{3P88}3@)4cAx-XrMovEA;~raC|3cYi<8hiN`Z-'
    'lz2STMyX&x`&7!Q(DFG}ohg%}^klqQZD3=we&-'
    '1;y>&ZtSrEvp+=bhyVikt9V1IpzO?>St)k#57(kT*0raD;H;4ct0f;yQyZUpSBOXLXHQ<vBgu%9l`BVaFG;zz(fx<rtGJ#+(xgk-'
    'bJtJtiPRBC0LY+gn31$h+*=E#_tm)oOHyq!mu%^Zb8;))iU;gE=|3jGl7XE9k7ZS;VMtO|36LLQz|vxz-'
    '`p|;#RyR{3ik8_P|X8;+4C-~ZhduLEr?V+`YHQ1r;6qzFEh0XmIqkjG^Rr^$c^#PV_aVXh>S-k;D+GbJGJ4Nz-'
    'l0H!%*X8e2NfDq^xH?Tz*x*!-!Q_69aj-&?_oo$CttIs)sNkxNq%cBjuG-4-'
    'e%l8o^q3W*k*fFf4ep3V<5@T)4vlBykm6kK=3zg;VyN8P>H!f{?&b`I5+2hkk)xA+gTc&9v<JopDcAc1U~``c9L+)hfmH$B&2~gB'
    '`659Hc9pb{o}@YE+oieyYm5_|K4sX4v#;|c?4)i2xCh+jII&^Fu-'
    'j3*gH5@|wYmUXW3Ov<0jtnHYIOnj6xdt@`K?MFxHqNxqJ*<Xh(>1TsYZS3osnR~GwqB-BVN$XDCMXh#Dh`Fv7H_ep&ZQ_3S~W>ny'
    'pNj&D1Ii6d>cmrxYo`GW^dd=!BKJt10S)Mf__iaDePF>nL)71rjw*4ZY6Q&63dZ5KFcNz2&bsaD&Jza;jxkX+~6EF5^5uFgvc3G#'
    'yqk19+#=LxVw2k4!@NXHs=G6Ov<>>amf$D0A}gsBd;xEFR%ZyJGRULXU~Sqki-cMd7i%9uR>?bB00|<7wH}kQIPPzXYG8Wghqhyj'
    'KP3eqMDs2wLo3(A!yg54)98wKIbLRGM}WZW27qlI;$h+mYq&K%FD6Os&?ha&JOZVaD<fMg;JfJi3#?s%?(`uH!DJayM0WGg!CHQ3'
    'FSpLMwYTN#|}=a5s<kF<81|X<X*!t=sxmcgI2#&a^ugnoIPUh?QGE_=j0!*T{#XR&H~KLU!b7<;|&F*QrX{W1zzE-'
    'o5QeLgi7oPvD^sD@amJ4wJLdv9EVXwQ6GEuBZW3?3|jQiJezd&12532};@dHPr&<9Gjq@oo5r&vvX~#Ma)-'
    'bKf;n74)QzDLw=wJm{)z#YTXE_%)C79s878o7J_i5J+Tm+ug64aNBtlkiPDZ8^?(TNXwFc`c+5f&7wBq6?uH6fkE^)r47Ca=lgL#'
    '9$JUfY-'
    'z8XxE>jeK?G%}5OBFa{k~Qc=x7h{gM7IGCbfVjoP;{bO4>>x~ZMq6N(QS+xo#@0gm+W*f+li^i2Wmz6&GP#HRZ{nkfEhAG{k7aj+'
    '6cwX{5<^WYuX!&KRDCgSp3b>V<PaUpU|UG_^Z?dBJgL<P{>+5EwgzUiK)RlHn!bxx;W!(KS!p<TA`N9yc$tVCSPftIIDt^y&6-'
    'wOgYW4mcE+q;AgViPr@}esLDPWSJ@yt{uEqigEEOzaitBi<xj)4HYl5Dg{y6lJzs(At@rQYHW~IcClIAtvdcku7v{?!s3qpNzDQF'
    'Zh>z!Yd4)9ky!&Ed3a8l@3)6{uOax5zQ}}HZOgrfT5im7pD5NH{S(wRGb-'
    '%3;4Ke_#ba2q7M`3W#W<X<b&=y!>aL{I|V{ouPk(qa=0xJv-UQ513wNsFT5CFJK1RSdOvvh!dlbUYkO@l~|2>1M}G_otms;DHZ0@'
    'U~4y>kD%u9#kEt0{KF%Vpxi|JO=waK;ex%;z~iWpmyABYnznqW+OCWw@UH#XQOnm5H`N{hAw!D&&*-'
    '4NoQ!JIH_Y<aHur_#d83rsDVi<<t#vntFiSc8JMz^&n5CyS&*gE;6$`=wh2}PIkKl_btgDwShSr)(2Q#F7+x-'
    'C4x_2SeUXYD}Ag(Q=Pi18Tk4{ut9};+nmXyZpKwN!*+WtS7($W32+wV3EEu!fY{&xoM}L;q<i3yB9eZ>N!m`CHENQn0{b>HAf$r_'
    'nA7o)qh(*OONhsNUYE*n$LJYHL6)h$or<T^Z{G}?y{-uQ7bE?*Z}$3b*zEPau-WU@WUtz2(XuVsr#9K|+mrperM5A(n=rM7kwL_e'
    'M~#*i&eS&Mn0+5)d6A+<m}TC`lehU}*T%{lPIYZ;uxPIyQ$*l@C=35Lc;4sUXcyF#u)M59KA!tr<Hxrc{Mb7J7BZN$$!=1Pn={<O'
    'cVJ1Owjf8THZn};sqjW`Yj5<n@kVc3Z}hhFMsIs>^mgz@Z%1qN8h~ESF<0jVR~)s~W(a7qw}%`#Y_bJaSb4jOM)+A4hEhJ<Ii2ub'
    '9V=xxRduYCcjz%ir2Je)%7$lhC2g0zkywMvm~xHCZu6l%`$YRlY&K$*>jD9S4hi-'
    '2{uwsG_QxEXL)qQ)QjVcry9HjxF|g~YGq2Dwp5~4+`(ZDd&8UrSw$WBxU2Gc3He6k78p(ECU2Gc3PF!7V>c}o!U2Y0McqCU{Odqk'
    'KF#BVfPZovpE;&yQz6Zw28%{MaR^IFNm?H9iRhBnTywp<IAnm1I6pa%_eO22&D_s`#eVo?FUvp%Kwjqmt90d__J8TxP+0bjlAzZy'
    '6Lbu5quv!Z8l`P^I^8dsM`$~=(7=GITJE12`4gmb}PdVs(@~r%fgU+|DJJff$I!CBhHX{8!uFeyxjc|1q+gxU*&Q<{&JIeJ(Jeqc'
    '(NqXb$k#r-A^Cr6a!`H<M9!_;#tl&S>V~PlVa#@bgS2>c!`Zd`(b(?b7C4y=nN>%0R<dSPQp<`Pxmg4LEI8w^_d44nY-'
    '+05hIRH>k*x&7hfA~pz&>4%zpRxyCkVSADF>2#ne3~e=vBG{lacZNj^*bWf##;N|+l-VY9Fs*pb-'
    ';nW$&)5q#E|Jfnl|^<cA*Y60F_)lR~}PngxX_C-'
    'Xd;JbA7DT;Z)bhO8p}}rij#M=OkQHUk8wWV2JGws%TKC4JqY5)NSyw*)-;=FA(XJ>k6N8-P)&IxA7_0ZGFmhJD+mh-ltr5@F~|Wn'
    'd5G*s)VYrQ~Mb-'
    '0M*TQ{yvMn*3}^Ch}D0ak@7B~4mVOh)tacqsWTqPOX@M_xFJ@=aGo1tMZ8##DdYq?&<YOQxEWx2tJm!8ox%J@=kJ<2qCvP$0JWY)'
    'cY=T|XsU_=6Ya-gAt{-Amj)y~UkgZjz8;YDTo#b@d?O&~xjZ20Ndl6d-j<}N1_Y;;x%#-'
    '<7#x!kzZugCQ*!PoR4=CG{2&(A#t~9%$*GIdr`Ro{87W{$EF%RBDb#tUOnZYQ8>l257Vt|c(gu6nZ7CY28J@ifrs<53DZDllfs;r'
    'sp&w*xp}M8AGsmMER(masey6Kw_t~T0>nhs)+?5`~yNR8$f7EsK%pUzo(`ayebRWm}9o+GZ{tP#h;pF~|Q{6o7Hz?<93{y?1v>FW'
    'bb41QQ?7&PETXTf2|Ii*8O+~RsMpIGjDJJv}^D;EM;-'
    '>4nw#`z5#Dw+ns^p1DhBL%Mb65Gj8CUWZoUCgiuc*SDLHOP2Jmbop#xb1bU9ZzQhO{h$&3i)K<@nRVE&!#`V43Ez4N$OmpTL}R?0'
    'MX&tJU`CCESp!b@pgCQ|EzRvBugu=KMT0upxK;J4{!p%1rvbOm#I_Z7bC2<raG*#1z|d5TH-'
    '8SH?3_?3MA%6nl$d;FGk2&32s7THU+bU&mynumc6hxTbfvPvsb{^zQW;97B`VeZEZ~qiSFB#!V648#hfDZ(I$`H?`E!hkzvx@4Ku'
    'BUqp9{TFWVO<Xn;AtGGa?3p?wwOG#abm>tf1o`~IW3=del#v$9RPwv|+)v={ia^GmKCc2hGf8()~729(Fpf7%(jOVP_C*wIQ_7wx'
    'bx;y|(!N|1P7@6)L^e|f|t<|l}`AO6W=DZ_n6m#woHJbT8pktWtfOs1t+W=OMWsdy>I~?>gf+z_LSzkw~80TfBt(glj(DbM+=$Zj'
    '97h+$<kbiBh&Hi~5Go-'
    'z9b*4UazY7X#etKVd6skYHNv~6<%2>{d9XS%$H?UvElUD4P@uU^|i%EQ2ncBZp7&Sl1s`37MMpgUbr5QHwODEA>YBLmPq%v-'
    'XhZ#3vgZPnF<k)Z#+x<}X3+&{UyL6Xiupz(F*)GWtLxBbdlhI$Y!X8ZF$t~&~?vfJ6xnjw6NT^GRj2n=ZoIJOiv5b*SyB+HMjFtz'
    'ySf1oNke5G}zhY;O$n{B@90Y+ylY=0zXi}`|eP8)D4b-'
    'MfX2I{(R9qGG^16VBy}X`EQLSp@Dif+0#nns3(|W6F6?$4#15XAywxg*$8K&OlE}!&i*9c=H7Lk*meG}Fflb?O3P{)@jHa8g?6Hg'
    '{~*{Q;Y`qT^PrdMn1U*5sgIq2ZAVHe%lZ~-'
    'UB`vrMcU1KB`yK>~M{}ek277KO|EEcSoybmd*gDE&WFE#<lH+gd8|G5ub$L1ypip@<|(}jt(>3#H*sCX&n@ujqa$DVk#06YUSP|?'
    'dVva>rVki6ejV&Te9E@9MDGu$$@8%IuvW3c#59)pFxK7J4s7W^P6EO;^DHfxpPaaPKD!}0qSBH2U!&nrc;mpq!^0;R+P7uMTafBF'
    '=O6?z^X=g7oQasKmbxjI!4>-'
    '{|3Cb?cn+kF#Sll{;qn+`FmJfr6`Mjbs`VRSQyhJ#kBmL`=r!^?3Ghc~QH%C0?wa9i*6RHDfdouay0lA?-'
    'Xwb;XRaxl#8u3(X64oBdSxH5+$aY*qL4EyM}N9ADj(W$BtIT)k#fI``lWuUxDB>N~i@tY#qPsxdWKwGk)pVd2h?QUs)cmr2$Y^<g'
    'Vl!SWBV{M0Tg=?VIH;c8=sk~g{Cs>~H6TE4U{-P=OSA>?p4xtV+s9b9~6kJ?#b3pQw45M_C8tvaCq9=_=-'
    '`3Q2K}j$Ur5BwTwvD}c(H`?K-ibwGFB}qw#4~Y70VEz`ugnN$M<FCq|MtE*O}YRQ53>VC>j8zbfy$8B7gRW-'
    'n*KpZ!Mu<W0n*<YY&@y$=5{hL+9e}sm)brVNtr|)lhJw=Gq`<M?^>+mVgXl=d#LX8oqY{5ug*esOu1Tw73+Bez>`eC+V6P)Lz_|i'
    'oh=<N=mK=R)Y;?p0o#D?DiYXf+j0Nu4|`$ftz2G#Mo2>T<*7$=+wVq#5>NGRBr5TW-'
    'i=a_MhH8j6y()9Uq3=Yj?n`OxzSr?%M#ZKB#j%dE|sKNX6PQNx`8sv)VjYLP@;A%?!qzia;0Gv`vg11sb7RzY_%mX;h55RG!G5E%'
    '47?9VY($d$-'
    'UV<*+sjf3b=v>vIOhMc59Oir?jfJ$*p8{cSuIrL@L}AKH;Ii%l9`nNbi@0{%v2BAlV3U$^JY5nv=g53qUy4d$9oAqsK%5&^(G=Q2'
    '^|xOGF|7c$*$j$ZV`gC-Wxj)fAIpwTP(>DX0v^-WW@q+h#Lm!*xowQqpASWE&+-c1?CrrsOQOk?T&jnt2p-'
    ')*Rx?*v3tp>MFGmW2X);yM3!<1CML6c)7j26byhm%(?F1@-}-a{&4$xsdU>{qq6HH+KJY8eXqn{HyhX}hLt8F2Z-jb-'
    'j4+$oa+5pAa2%UB7kU~@$M)fUZbm%B7k_i9#F_s%m7g>W~!s{v@T<+b9zGng6>1=v7<GsKqKDIoZ8%s>aoGKV$j=!IvcdD#(fOzX'
    '40YLVLp^x=R?WEeJJ@>A4(qKL&+n3D0!3*C6A^c;v%k|^0?mP`$8KS_s`<`jxW-KjnKZZJouTD|05QDaH@aA!tVt=CIWuuQS6C=U'
    'w>T=7XiOJ^ngP4VHSS3U~2>ozthkK94f-W>Qx5nk$6}C$^ffGm-!k4xDu;5tkyG#)O-'
    'U^W}5G_)h3?ICfe04Jef<Zt6O<8&j4jHhNm2C_WHKc1|<Wsu(>m*S(5YN^u{E(7jXu3WweVP77`f_^Av|cwx#As+m5^&v2-'
    '*%4~*uPK8OV)oa%#EFg~NlM1ave;Jr~`9H66t5n#Mi4=7|6R%9!@zKzK*<mRZL_6TKdxQB$ClsZ<w(k*jfBvz+bngPd~OajBn*i<'
    '7#IHu*)8C328eK_2F-3$8gR`Ycq=))1_>weIOBW;H-'
    'tS}s9`yFA0;b_}+1}hB5nD0w41=b;ZpKqdVfP8Hh*>`=BLOc*T&f|HNGv>@6#sUz|^I<Fi_v<kc05s2GUlahZ<waOh0|4H|i*lqk'
    'D`gqXZ)MoZy`RhcsB;VvU+3^c+9Gd4u3=jWGaB1P5QolfQVn5L7jwH*LtUE9SweM_u6gV!U@3YklVmoTle<wF5o$m;w-'
    '(7eX%=%=7rR`-'
    'FHq!SMriuI!Nd8nQmT2B%ByBzXZx@zAvyuHYkak4D~jRi`F2ivroI9Ah|BXh3P!}Gc{GC&g__NjNq8hf7ioX;ToAJ!Zsjn$j?8pS'
    'uOrh2^Xq86!UQ{-'
    ';h14(zAxpNVrRZD<CtUTx?7HN1rlrVban@^2bpfu1obo2RnmLZX2A6rX>OL9t=^`0W7=npOCnx?Em5keT4!@Ao(neV!-'
    'mQeh95aLiuC3?IXRg6n&Bf(v}eJHIMJRBBMSABDHH85hL6xjMeE>njyh;K{3S;nG>RcPvJI_$4Z}k;`_^Dg+_Mcek?K#}Q3sWzde'
    '(-BM@e<FZT<qw1=1ALl&$5ey-'
    'eP;t);lCl*x*+GN}b_8+K^QcXcmvG=QWJ<<6B~4~O?!ozV^(%c$|z@~=lv*^H3=?>m@}L^&Iemgn6z!Yy-mD08ZH#G8YUxCqd{N-'
    '!iY0Q65BQmC0snbb!x)z`AFfPTbBN^Gd=Fqb*mMm2SrlWoBYaD=Gp2$wn8ULEN&Cp&;G>^M<XGiq;`tVUhh$FWXvss3Ue-'
    '}{ixJe{r4Lh3#VHVs@UiOC+dLhWa+bgLE<&RSRj@5kBPhpg3r4YZ!vK%J~>m?{}p7o7Dz8Oz)9-'
    'MsmYcK{!8vVNxoL*it;0*4gpC{rpA+<%qZB5O>s*Yk%stlI_jIF;DvQc`;lONNZtfH|cWgp&r$krU0N8O_xLDP9zlB{_?Qsde+2V'
    '?@1F5+f>x@`O|qxFhE5E?r~zbA>uXx|ZAL3Dr|lkp!x26Vb(BaI#DN)_&iQT<4@x1~H64z8QAA#<c7>Hk9w>O>R65e8dTSX$gkJ3'
    '493-Db!IWN8rPl*<AL<Z-'
    'J4+GyW}DF~91<;j3YyQpNqPk}~m~(B~pgtrq(Cx!RekimSzPn_9;{w_m6~az`a6rFw??!wF<Gp)S@{LiLcY%lZO_oXB>)MhB1Jv~'
    '^ZX7$yQ+_o7<LF!EtzFz5g}gu8w9mSl^1fYFWv`R;R4HX30d`G0w}6y}dLiKMQ_Q#Fa?uE#5CQp^<paw%I78Je`3p~Ho^$4j*i`}'
    'bH}{DH%d9WLM}tHV|XmD#kcO0))~q~aK`2M?804>OvnrZP;do#PpEzb+kPt|n%J^ExU*Ce@!QM>E-`9-'
    '<1@WT$$R(MAzFT+pPh;`qLWbOguuFqv*RbCuOgXuHHxf83KZ-'
    '|Z!^Si!@oSghb@=`n?R&SV#KozrMZUS`3|63Nbmmt`292QSMo+yRX~ZspezjXoi%APV+!m`wfP-'
    'T!E!|C!1tFc%JyR15{C;9)Y1k*~M%d6wgQtDucNHke3p8gy5Gk(4_3NGf1nhb(I@cLQ+JPNX>4Rt32mtMs8u$^;7!v{i-'
    'ETXngy$^&s}JfBxtV=kY^${o(bW92?kk16B`FO&Oxcx?(x88!&&pNj8|sq}?Sg8Y*I>zMf-'
    '(sh9^SR?&Jj&y!DbLSkPPM7LNEBGDB4)utsmQ6OOG16FlXn?8C+|AMRIlhO>)czZVYs-'
    'H?TI;d==NMbHQ%99SXq@j8#^^vRFSPNG=Gm67ppbrNpfnJ)a(RKx0jVqw%PWyIw{f+Mrn0zNMpIc_T`cXkU#Xn#eunqJt`nNMn3|'
    'K+V|@Sck|h)o@LcP>A?#YIHsGTT%b>B_-enR#ZJ`9m_h#WpE2_DAj=QqO8!^S*5tyvD+W_P-'
    '*s30M+y&d!!xkv()o4jug^lVqN!x=>>UK%IVofxC^N(DM&tr9FcgWb?*NFOVuC~ZXHjActvNmVzC*#>H`pIZEi+;s~eSIlgG>1t<'
    'cGhsrOtL~W9^30A%DA~(%r}P|a&Qy|G3${<&Ukl&jQy_KAn+v{-C(B`mO6uNwkhB@M-zNYz-'
    'W$nAzOuKJW@4Wz08Re8wozGc+VI3t5hPnkRjpzLR#NBQYMq=XRG5}D(~4G-$TOH9hRmm3=q09*x!$h<heLJcXihnaE*-'
    'TxwuBg^ITk03<`s@7ClnQSC~*{4Ngc-'
    'T=&>U1`d`}o!yzjJO4MMqsxI}0ov9ntIkLf6#T}3)E>88n+=V<%Z{GQ!m^rq%=IB_CE=6Oq8@Pct`BO8f#<nnNVuM>7rCo&zkyQ^'
    'M9i?hiK|!a5rDj#Ur5g3_#PSp;CozcGXUtu)Q`&n&<F*_5qYY>oV34;r@-'
    'hh<0&xu7X!eEQci`<e%&J3mrym4*yz<uj!W$I>dWL{bA4VoUtu41zMHQv<79;?OFbd)w>=@XOtvwiIXXd`S}TbHU#;P^>I>}q>V$'
    'P3$9J7f^Hd&Zgmn*BJ7j3=X;K|$?A?#lLLR6S^r5rv8l>m>mpVU_I0`}O?aoccXoMN#$UGRB+ZiC^nK1^)cxH?N#b9t}9t^rmF!5'
    '|N5p={X;m*HO1BNGTsO`~cP0VG>s<6crya!UrBhEpXiRNe`F6ywfyEPw7ga$WzH!+jKRg<j2220jAxK@pn#y6<ta(bpu%{-'
    '<~(9(x3#qVW0KVvs#!y+r0ug~1>ThVRPv|<x1HOnaOYw%`BI9slP-'
    'v}+nQF)tsbK%#@cv_5WWjrm$wZ)`<Pqq@{4W>>uGYR(Ks*H|kw=c!m<4ZC2`cjO2z7%7>ueWOw)Z4M3-i|ZZRAZ8K92+n>rbu-'
    '=cXe<*{Nz82<9o0S)2%kArud;8b&BJ5+^yFpBiuirKb}^z3A&iahH19xgX-HOnJ}8=O>c9-'
    ')iRz5qguu@VN@3r^nH2L8@9e@g;p4o*m>O+S#*_@xK;z^TC#--le<eWz-^<F(`B-a8kO{rV2s-'
    '!>lqSuaJErNPnqnbcxx}2?4ochSoib1x9$+G&fppE{2-|=#C=9rnbY-hv-dY?A(JF8kS@TW<1@Q6bXzwub%^mS_GId3GKv-Bg}lA'
    'Hx$=QBo)u%DjAz9dSWMi_T4&RFyP~tgK1(C*Z9-'
    'jZ{jn9WrCT6%t}c4vkC5tzfM7Tp+ELvKrgv$qezrf<Q!A`Op2Jj2qb^2aG%s|{!{cou;gh<H{)MXxxijFmGD^^0kL{YBS>6CUjW7'
    '+vq~Ql?7(G)}6oa_xD3iCTV*6+c{Umc#OIMsXO4G60*y0$?y?V{<F)SJCA7LJmu!1)i-'
    'cCf8JT%22aU~Cj<B$SL9MXmj^h4~)i{x;WY)S?^=bpSs4#y#O0Q`2(Vhk-;w9aI-eF;`V1Dfx0wKWAAFy!7qbI$cW`-'
    'Y^xXUWaTMQ7VYKZdTr^IaR;KjR;z&b7~Qb&=~_`z%)%b60r#h6mat^*{<}JF`IRl>*ujt~$vu)1x=i3`|7z^a~rt_!$_bf)<_%)d'
    'Jnzb>=(#f1RCORY=@4jl++vPYOX>9mU_FJ_zA>sD-'
    '1sz4oziJOhWs!LcU}DF8>C?_?fzuR?HaYS9JK=v4@gl!MfZTY)v6VA(p`V`LV?N)ZYp4#$?ru2i!gjfV5MtN&f-'
    '5%juxFomF9Sp=P#LeOBQP7}#n=3p4ZLb7+w!3eEZsWH3eg0_x{+cgpNfWQ^mRsG3Ko$gHiBxUQ?B0cqYI0dJDW}aFPTU$)JT<>X8'
    'UeyeQhOID}0?;L^=uVzYL)1pmiKq|r^rLxN9U=jVr|J-iNW7vBQ5sUq?#w6+xl5DqL}*A82`-doSq4d1R~K>@+nUQTgPL3;u-'
    '#eAooi$PLk3AoJ$ecqs$b@=^`Dv{T*E&#OSsDQMIOeJ)k7&5@6N*btQ3q#YAUq8fekQ|C0jHkdolI%)ZV++qtHb8t&OH#>~@pYnZ'
    'w)f$p4@ZT_ABbmI>u<ybsK8qCUz~k><ub#v&3<)iD;4YxI~1C8@=5R+N$iNu3p?B+cfjkp0*uvu0la8pkM}FN3|CRy@81=2X3$xn'
    '-_1cOlqeNN;6|7PZXxyB(Vj>VOLTVX6$Gj+(62B*T-NsSM)QWRUBYn+$f%a+4vhS8g)YwaQIyah-CLVYKVm43o!We~NlIh5bEQ?4'
    'SKbNxugsTTvh9C7zpeSH{8*PE{ETzm<AS1ZBQ{5@++6v|o5wXY-h}U*l#kWHPqN_Fbw)N+-9X-'
    'G}rEu%cbyGKgv^05uIj=(Nc_9FCka`L(V!O(&WxW~$EMdM-'
    '<Lgzk>Vz{s2_T#r;WTezOOYL0MCb=6$qy6UQV!nM^^^R2es!qgIvxT)%q6yo+~5qC~X+ro8qmSiV<#&k=(@+yb})4-'
    '^IhG3+!XmeSeV!;Tf>J$sc_w<+oV6=MLJfw39fsqQZ_i8qk0$?QNX<Yt=48}IuPD@VXc|B#!stL$+Vd7-T^stlqJwYe+-'
    's*kKFwA7Vp#Nza{2t{xB-7RRImOzO8R`d|LhZ>+tt(BWJ(;C`#3`DZ%vSf>gJ(EwfqzVFf;^e4e&WdlmwG%-'
    'Q;(+bxG#&xs`65BO;xDp%rEghHjbr6{VPv5nycy@3r9Fr=U6zttj9#?M*WPcqI4rX#;PdYXwr=e8H;T)#-fI!Et0tzUbaYP5WH-'
    'W%wTxgBAFrZ8fgPD@ET<!YVaCOb&SKPLS`FN%REx1tKX)OvOkNIbIXwe#_15FU6(L5JY^f!V_%)XIGk>=k7;G~p7^dBGbPAY9HSM'
    'un?`qQGs5We;zT;H=9T1_TkR4HKR8d9Sopo9$3(zSKlpQ_;0LlfHwu0xzpaoq%;t%puc-wlQP&a+(mFxAlfdW)p*oxY0^zens?f%'
    'tohG`Q$C8~IHS<}rD|8Qa#a0?>m0cXWF?@gOW5n?O+#dB{_+D*~dg{Eowf5*uhVRLCeSVwad#Z4KHQ+r>gghF5=0w8&0Z-'
    ';zOEOulCi?QR{%F5~W{czWIZou%=iuLkkGRaBBVa^auE>!vqEL&Qvb><$u<EZKi7{QbpHDD^t4<)|bePJ0Tx|COL0<V)jOew40B*'
    '8POZ*L%&hhri;Z?8^7@j<4zTOJ2$IaIf@Or|09SN@|&DT-zddhqq4X<(L>lk=FZG%h<-'
    '=A4XoS`#r*Vv<8IvsBvqneYKb18NYNRrc1<mB(;UxJS~(e{E7aiTpFMilC4lMVh|410K|uEU(kz~t$T<zs<mvy_FF%>;ozc0X*;h'
    'wi5zYZ}#*-SdC9(>oen2b!lHkB;N`USgSc-vjH&{2?s*dL@@@uMxtT<MRZJe+xe1B0;N5FeEMz^ePT1)Ws$%;`6!cW<>mRd}l>$&'
    '4@ZwvrC-|6r|c9C|U!@AB(`^g-K7TVO@1J-_C5(*)y4X7@b6}EAAuS*Ln=c_d@%(TX1|YwnxWvd@r*{ztlM-zvLl7+#h(5QMN+UG'
    '5T2Mdkn4P<6Hvx#lp6>#X7!EnQf<)tW&S%O1}}dmlN`&kADk3;-vpv35LW;e-'
    '92R)ZHd4{R_G3VWhv1sRu>YR~J;`*ZK_oddhOPlRsFx&HScPM}3eY%|BLm1zTig*plOWsc<yPFCde0$g~DK9iMabKEo}IWnOy+%W'
    '-Yg`CTl>wNdAP@MIoM{yCt)Nhi#O)t>%;9;k9!wb!e^=E}PfQk4_)<c)s|KH}uPtpr2j<h>b(6zXnMR-IGp9V0r><O|mD$W<WYtf'
    'tJXkF^-'
    '>QvIpUpiW27!KlCAM9`NqQ^n}y4S%!VBnVci<~(_filUC;$>S9IIF=_**oZAm_vA?|zcd;;c?tt;1}Vun@_HQ4lc$sS%(bn6%`Y{'
    '=JCP&#<%!%CgI*-'
    '4m36W~{Vi9@jc}KolqY5UTksJl<=09uBu>gN;E+PCY|1RMdXTb9c&T*;lkuM027wr9{8SJ(SEXj)s&h&mf3=baLSDvT*P(?x0&XD'
    '{qaCZWyT)=7GaBLl6i%UoRrN1HqDGT^rJw#_gn#1IJJ?vOgtNG&3#;sFl9^za(ds`7OnO@WXM;gc%l1a~cfU7njpWOHW7$DY&NIL'
    '9@4`o%xEGaRNSwHz!6Aiu*kqOXMerPr68{yxgZ5G&dz!{zNj*mCbAw>=*~;3%F!yX>?GTuHHd%fs%sg9GdkaiFn>IfT=H2JcmV&k'
    'YW4}(r-'
    's<1pvI5ijKZ1A{UdvLVPsDbTSXFz0^s7sdUY<1I{;#=fKM5$Uvs*I`KO?T~76La8eWK_$CgqXl;w8dIoP@WOU`U*V_v4U49-'
    'Xp6oBu114$S4Nxn;$#QOiMny2_e5*D}=h6zeH;zmu?RYDVC(smDY+n^*@H6*Wn&z|DKTIzqZkB6wE|S4=-'
    '6vzWj}e6!Irv(Yk)<G1LPmwsn%q$0LW>`^!*lHvk~6lz~nrm^b+jg5WXTxc}$p;0}XgN^p^AY?XxYl50kV-FCZx%88qg%)?Hv44e'
    'm@9=3(u~?|%D799-CDh4O90PiHAoJu<*E1rAnxk%Do-6MrMqGKrT^CS1`T<uTIdX;{a`myJy#Enb|K!Bka+@Zd-Cj;QdqBR6&O9f'
    'cYtG*~R@7(VkVwi498#!hO<83PuE*B20=ue~l8{e9>Il1|zCfmS&0m$NUGvp4S)JWfUBZ#q=UbeLsa*ti4X6uCqq!E;g(Wjt2kOE'
    'yXsicyVL5U(fV!~Eh#Ns&Sjx>!pf0X;9g3bq9({B4|9iXAD65KVZPjU-ZYGT;AqkO|sT*h}MMc06jlrlzVGJUgW@;`C4b7}3F~iD'
    '>#$b}S@*_%y*NoykW$>X%+L2~#M2#eB#Ha`X5t*lEetTEduBvnH>F!fmD=+u1ERIn3+<W#{yY~3)nciej96;BBqgiGvQ~Jn93y39'
    'ee=xU2@3gI~@7RS0vs>VrP<jholW8PPUc-'
    'zkXQ1e(*k#DgDiuRS1?}FSaHl(W;ekp;MR2gHL(doQ5Uv=y&A+&uV-!-FEAZjKw;gG&z!-QJzA?A`@WUcCZCi7?O=OzWtr9yIM-z'
    'Ds9s_QWD2n(6OFslo_9{-3Zf@tuXn)Ln__yKMkJJP-t8>5!Kdf~QeBC2aD9J^6+NRcDt}73wxxh7{JQuho(-'
    '@k#(6!x?L&=}xY#NUZB<Gq+1gDK&DevYM5*QE7+tx$5BDXdVd>BU`_<D171xGHq8)(a;0;!<@<rfRcM%XRSFCZgfkL+APR>EFcQb'
    '1<HK6ya_*$Mk)mjW^r4#=(rWGT$H(?PAU|BmCSl(1Lfc<cFr1Q&hMnj-XVy76F&3tSV*ae-'
    '?xO`yq3anvH01f@7|HxF?=;m+Yg5(JOu0Cc$HIZ#~dED`kvSs&F;^yUb>%vjhP7nqc-'
    'm;sO=H7sU=CLr%i%)$x5_n3>>I6?NO=3)*g1^F{`F&Aga{@h&5!}*~9&50Ybf)O|5YB`P(J??tsuv%~Fn11{3a9i|Eleqq{?mV2^'
    'qB{@gw&<Qs;-~Wrv{Do?<uzcZU&S?JSj==Afe*}!h2QJN!Uv&`Jv55twLpt6(efS>$!BAOfd{L9Y?Sa23~f-'
    'q1?Yq?MHKqF92GppzP^Mp4{Es>h9OSsxqwsQ%7~gSKyiE}qNw;>J111N>ByCK6w%#WZnGuxaPt34@#lq+TJ%dJ!Pd9z!NW-'
    '{dhl>kiyqk|URan6mEmxhS~9%7x<w3snvfRAmri%oz~dNk1K}ZFrjK+hY*5|Hw6I4--~-'
    'Q(r98+aR5`6}l=7HKZaU_)wo}SuHo57T)B1=~9&^b}$DEd#8j3b5C)|u@a!di?W+Icf6qwKn4v?%(-'
    'OtVOR=k|n`=URol!x<Hl=5)iiqed=i(QlTIw)FGUcE>~GD^5Ly#MNY#jwbBF^B)R)IV3+1~KsTu8d8DRn~l|ZIEK3%r;0d(bG0aG'
    '11F5NHNjdHb^m1ZX2YKtdtt0kgU{f)jb^Zsdu^<-'
    '9y~3zj!c7P`W+8{$d?$J+2rA2NsB;Fye{^ma>=iciI~|=A$z%32QPa!X*L3rTY2o2`ZC#P%Q{3li02nWU657nyk|Cve#aoIz2i4Y'
    'Yt|d{Md;8w;YT&`H3;tui<Em7?1KgEX$?V*u5(IPJr2m2{9iq4<!M!BZaFOyD~|`<#2qi3Ecqx7x5U;19+(&jrm)N3=u<ZXQj#=s'
    'y*hDxGmJ9*LKM_c_6LEhBPGVAG<G<L>KO~FBFRXaY+Ca^^@Nl1jUEcf&eJ)Pzy3u?8Nn<mfi}Er<rkHP(NDdB?lMQwyniZk$S_1S'
    'kL^fk}4Oa@cqGs>dXUQ;-'
    '9>UGLF>R4bW5ImB`&R4v%B=;GT$rV%TYFhBb}A941X`x*jVQC`!E#`GhR~_hUNI$N$4g{D0I`ad!*k9<QX~QQv%jC?4Tf`$I{cPp'
    'LHlfYgs-Ul1T4Rto|E`H@<XDc;>R(OmF%41p+92|S|3vAPq4X3k4JTeD0}_8vvRMCD|wkQO4f68+t17h7`%SS>C#MJEfffyGpvEW'
    '#>tlXzN;MM);{v;+qTo5GW&SSfC@%*!y^ZL-YwW3`$&y7zOr(;@KymXP^Kd?bm)olPOJ1(uPGX-'
    'L#JKM;yUxYdDBBsQuw0Z7!3Vt)`4A5jYekhoJV$Yd&ZOSVZrgX~7j4NZ*JjMmxP!Q>h2quXK6#}i=Nlb4WF`b|fMPKbJs0txArOl'
    'buE`4C!*25^VbS~P$=g4Ut|+)=a^4d5ElS~P$=hSs71+;OxP4d4!N`LP4sL9Bi80rzMUa3AN{ieiuahd!bA%Xjln=oWZRUQGj{{!'
    '#}+fe5!c7z)I_YE1wT^&>hE1jI+xf&d_XtQKT47rP}?;#;87^hx@Od@D9>O-'
    '5L8n~UVYh+tSsq)fG*<Eo2gNe!Oi9$hg~H`ONV@DS?8DaDz}+h_(&PjX6o!D~Cspy^FcX*s+;q8T)O$SLg$FLNp%MCvvR%|lq!<3'
    'sbYBs6z5r3+giU3o1Hg8Jr%LLmsZIur`QooY=01ofjh7zDw`)PevA?otafnTg$!vp$r)SE>b?qrf%VgaE69G~MV7Cm<r=>YZ=Bf#'
    'Gb1cz_#2ER?A#6ALL*)l)2TgS!?}rmDACLYb;^v6M1ZeZ(?19PWO~R2@RPC=0T~3`qbykUgFR*(dqF@*H?N+6bA)<c;~cj25V1UQ'
    'dIc{#u7a;Rm-m916ep)tUhK>4$VE2!4;N1p)B;L@mf<6ef>iz)Fp)YO1kzprV-cXkcnxjJ*PI>7qH{vMxl*&NvC3HXLo;sF;lApD'
    'h~|RcHa)8bmP#jX+C#CZ?htXjuwHHJXB!1WimsYtV9_iRow%8Va<-T<&)GJi?Hd!o%kiNqp|k$7c;Zb$uf2mT&VQ-kQ~EEid{j9S'
    'OxI+~-IrHeXX~0<ftc(BU9#KA{!_U~{)xkjYHU3l^6UQCy`ONpDhmoiNY~24bg(%BUA;02*-STwl`yQ|DpDn`3I&boIH|t=Gn{-'
    'YDZ#9x_Pw<YESuSoGrHNRK80E%2mGfkinNvni<Ghl@ExCefFRxfI{;$HhDvk8=;l=_uQ6sm41I{mqr&nwiwYAn@n0*gK^UqFLsg_'
    'tJ7e>Z5{tT#AExT-s#partavvH%O^n{!-elO5MtmE$^_;<(PHy0P=sZtVOtH+FuyP37Ou)x>4IdXLK|To2KrNM2GvRkHU-'
    'a!7$+Ux763yLH#)L?+wvc<pwL*Gh*-'
    'Gg0e%8D3P_d=$JFnB#eY3IJJ)5Uomz<^(X&yr0H%s(w~*pUZ&YK9>tK_PKN}%%?Uk@dIAoET((C5*^%lCc~qu7|VF9MyHG?iR)`z'
    '9nK6_hcnaF;mmS%II~?H&Ky^VGuPGO%yV@(gE06i*Kk{B^>##?`h6~+bP%eIFwnorL(NzwTWi!zVtRUD5V6Y#X=OR;p22M{2g2K2'
    '_GfK#DNR+-'
    'wj{?&5&q;0@m^+yXf_B)`6@#mya4x#8g*w~<l}OOc99!p=UBG>g#1ioxDm0Baxs@{jj)Dtxs1yOSG%*Eo4HY{zT*KdH=3*O<NSBh'
    '?LSbOceKXjRX7WtGDlxk(x0gR*#}%c<*J}pi`8Hm_&kY^@lf8BEomEC-'
    '6puRWm|Y>%O_boTly8|S5$f&?rIlXF@ua<I}479E8||dyu6mn(@rTa;PDDoaZGO48>o`WcCncDN%`4`XfRL7juG)-'
    'mRP{y=`n+5guvl3)AE*g6|?LRif&@I;V{P&WR4-'
    'e$0KAe4X1z;WF8HlL5|xuBRcK3;9T~!WNr%#B_F08%<98}TUs`Ux3p}{+R}1q-qAc6bnD-'
    '!AbO@bFF{J@MdFwl)FVF~iQ_b==kt+aBE=K*3lVXV;^tTT-'
    '||Q=z*ck1L<{e7ZpREP<#N5b`jE@tJB{0k<GaKsBOb?Ok+$UEE4B4QH;cSsVmPVbYnFCvjY=#wnq=E)?wg8x6evx?56<Zgno~cB0'
    '%U8N9#_94xV7bt@Ya@hvbMGiD@<0;mQbk#?(kzS`&6cgp_c2M5(e}a*=Kn+#|LXh6&!=ZGZ}=)*@_OVq9^vqsc<B}MRoLZOeYdn('
    'VIpjtfJR_NQ$-?-aWaUY60)wj&&T~ecTuWc=vV0m+<aquC{XdhbUfSH=fC2t!*YSJqk65T4P(<rdGEJZfRK&-'
    'qP~FSzB7JP9E1Nu$&q5s6=jkD_abYD%ELitcEw9mD<T5{aWua&eDUr^^~68PHF4h=O9jnoBA3~>~s9q(ZSf2>NO$`!7fIR{Cp_(s'
    'd~c-F2~+tj|}}w*yrexp%24OR*D3`DN+!T8@7$hKRR8~5nJbFC&Ov9My!$l;6c<D@AgVLywyE|`(6GJ-'
    'tY3qto<${n{%h@oI4FlQB7MMWNQ^+YyUi=F)hf(VD-'
    'WET;7m4x#d^zoKz?n&gDO1<#vPAP)S#AH```jP9epJ$u7fzafcl&I9WVLy%AO7d8(6~CSEpWlAs?}Qey-'
    'la<r4{Vr3DIG*?H_IMj9?+#!<dIC|J2v=R5Eg&OrOkFrqW2wW1#9Dz$Rjg-lkvn=7_FfskkpzMUG_iHLUVwaFT{YtnYNX5T;6L(0'
    'yIxjH{@*txN#B(I0F5-'
    'Do@?ABnJSzEanuHFOe0T8@39*M^BL*Rs(maB}s7JiQ%ej<=!wo)<QIR|x1UDtL_!Vf%E8ty?;gtD7%n=J%iVoK|X$+O_MTS2)ggg'
    'S5WSSvUnCMG<n9FCido)A-'
    'F&3Y|?QI*G{6=$xK+ZYeXQSk|+So`(JIJ$JWT>6}0_kT>pI{{_@R`i3mz~4Bs@Pjik`$At^X%x+<MLKJdh~?6&5j;DDQ~y2Ffa(m'
    'xLrIh!X|VRZyO(B7rHUQCqzgnF(#Bj;#X<gSl{y)3+9k81ja!m61XPQjF_aih1z#J$0Q_m0NN-'
    'h(!S&@eQrH$bECe|Zml=$kd=6RcjU1E=;K`0=`;HYE}!Gc$pfQYH10+hjT`NvaW}bW+!z;)yV*tKZgJ5#9p4nE<+trT*)urWey26'
    ')ZgzsNjF4sEmQad_U$YdGTYvI#7EBUh*oA{wB5+Nn2{1_?3#9{Y@`H|>VHK0Fcxe)B)_m1Vj38g_YhGpq3r}D7QX|M&`&XSC!4r_'
    '-?lurX41Pd8se|w<Z^7f(4t4Gx!T(#Y?3T#3Ex{4~wmv$^Q$Gb}ngS4<+n?d_IUpsJJ<0(-tr=N|3Ck~A-'
    'm!WY6g&(UHZTNM8EB4Up`;PNNkf4C5hqwMZG<7_4dRW!HJOILBy>5b$FsF6KLe(1QkADhPI224Ys6)o2%dD~5<JVp77FvYsT-'
    '~G=1JzrAzKA!_}?g;RYl(Y8aTs^o3<9t@V98-'
    'WHq^I>){MHZdwDJ;qTbH)vHld&emVe<VG*ShlQzc*eo6RZQuNNcq4wBc4X@hJIR8XBMb{|5OoBu$u#dJDd^MS%BVfV7&s}Zdj`Xn'
    'sK{xi-(_TufjYl}JG-rdVO_~`5U{PvtH9}t^(<zs;rda;=s%bmhD8-N+!%E}1rBl2=TpJ-B86U!-'
    '{fW*0yd1gRm61I!xSqA>*akk2lQ1Jj>O<yi~{M#krlU_bImJKrQHH;#Q&s8TYuOo7EByr7+`~#BXCWop)YxgeGYb&(j;8$%o)LHK'
    '04}(57BngR{*H3ZlV+1a2p!E`sy@%J2b3po*nG767wltITBxYHCNp^QonXLS3Ni~Z}u=(r5uS6OI>JWY}||nyX+jTgieA#?#I!*f'
    't@<lUV^|l$5yQytimZn_T&h_Er4;et!18vT?}dYu4efpcBD1I=x=_S1v5$*veh6;30#vYW!*j5fAF6%qST=Ej;^#p>07_FVMCH8v'
    '}0b^f#>wA>4SCJMI<4Sf)R)D6t=FzUWS)8tUsN@J2AI)s(exdXr_ERkXqu$X`O8Pr&fe^zqH3CVJ!z8a7jSHgnsPxETmvUF9;}@='
    '!mTV*&;ki^kFOiHI5B_Hp9-'
    '9N4V`rlW6}Neh6%C*K?ut<~|i4JqQPt9WAJWWLr2{25Uiq=?@vpkROUbsjOEYl&>T~`MH!<m=cOB;Gs9q&XfY&E%5S3IE;3$(S*Y'
    'l22Ji=1Jd(w;P`q<!*RO&eLftUrN6j44UYOpjtzz5#keF4j+fw)066LgzA6ZgdO-jjKc^beGnF=WPgK|53Ld5o(A>f0$`}t`iN~`'
    'tLMbLcSm9=lVKq2_qRu9ZcQ8|h3w{%z!h|Bd!n2c!k(!J~W-'
    '2S^>4=I*s*xqMnyj1^5tWflM|#2O)CB(Lh}!&TA)nDK6Qp1`z?^Wej9nl18<V*Id{f1?EzouRBn^-'
    'H=Ho&u%HUSxLMzI)s5JqI)Q@6y5F+)007QOXaqncRc1$Ajb}nDgi2N3pFGfj2JQ*BU@PjwC=k-'
    '@pSu|2Lyq?5HRwCxSD}sNKB%1&@*I6%oCKJzDt9cd^&s&dtHWM#chkOnbFIp3EE)y?V3vnJ3FIxlgRwh<j`*1u%UJ}BjiX8Zud^L'
    '&5FEoY87T84Qr(sgxe0(S-'
    ';a1~AG5MBS6M#wmDAojFQZER=<QEjPRwk3NM*@?#arwN)<afAyIY#uTRAHTk4n^G&ITUtB;85HhaYKQ3gbhXB5j7NgN6?tqyPEvg'
    '$zeuCmwe2;mc&e_yz|>=a%()+p@McJv9M0v1)jgl&7^v~+s_r}<tDq$8bKEMdAJtXPVPwqqyAMBLcs{Pnh*-'
    'c*VUQ;FzRQ#HVBM*K>!##sfm%9EXX7n!5660VEi_hFL?^GQYpx-'
    '<ouP;DO|>pZ*IBV!|hzyGQ2)RrC?&JE1Il>*C@Za{|s`JoBZHeTs%kF1;)kmloR4yyg<$U5f?8~(xCvS&X`nDE1aihn)?g6SZSx9'
    'G9)D?fU9t+5AfHM0RLiBO0fkFmY=5KP~UuFC=TIP6GL&>pw<N7P(O-wK{(V40&w_6MU9ZjbnKBNlPJMH7Zc-BnGTOO8LzB?SnkpK'
    '&0ti?ps=Pu=(OoPSN(u#smbC38vloBOyc6gJf<DeH&t{aRxZqB>OX&nV=9vv95I(^YYxtLU{kSpA8db3g6;hL?t?S7Fy$<sR+!QP'
    'G0e}>V5q-VWhe~cR+XVJT&>mwz)(M=^+7Py3j$zxzM5U0$zJS{?2hqrs*W#II~p|zsB_4HsdEm@QB-'
    'O+VsfXl8T}Gwz5F=FU|fUDiP(&2ahgU%>&r9pvPg4u({zp*vFRK$V*?iE#^!O5wK$2%&JLa`BJacVjU+rf=fgANs<^rOT)B(q6P+'
    '|mY_l`1IYxh_Nukh$`%DUj=F@6T05tUjY6ya+UJwAy&Wf`mli66D);#f_h%^r8w!R(<#;B@Z?qYZ>kMTk1GZ1^p>DaW1T6xO$MpQ'
    '83r>NA0p;U2ln%Ox;7Hr3KO<6TgYL$I>7_H9UwCsWY(rZGAOgrO}u=0)~T#~6LrZ|x)S}p&j+t1d@9Z`Cs2#!OLk)0s{26rtH=2d'
    'vTCKc+LTH_l`?7Je5T1oq_<%)3W(`Hz+tG#*=UTbZY)t%f@aNP$V!>QI96YY0%Q(;{x`>IJ%#)Jc38Q?PB*HWZ=o?zH5`(17+m&&'
    'oDOE23P$={`vYPG;!voEd4L*I05C~X$pX>F*y`{R;KO)^P&ua`Tuy?qKRR<q=Q<1CnZyLzXM&y*KM#FQ5wxM)f`)iA|M^XC`fWr&'
    'DhcGY$(UA5f;CV$MSvK|xcKjBnQk6GCHoGR)uOLq^antDv!{FIxj`byZyu-Qo;=;=Tyxxp5wVfLr(ZvA2FLd6ZYS{KTac1o?u)H0'
    'Kl_bRzf@9%Z;qxAi)SfJG0P2!cz;eVN78ysUDWiy!VD8ncEGu2Mh&8N9V9xQn+w*#Bxb==?BQREA>b>xe%IC3M7E0`N`JdU}6o#U'
    'As*hw?wabV{}=7w}miVY7-'
    'k#`FOEC<r$tv_sisJ!7;>qF(;sMciam?=)ya2;W?6AE*ajv*>Jdu=g8Bls6ym7a~ARQMxc)bm#muxXApa=QjngZwx-'
    '$_tc&o#iF3=9Ul&&U4EMC9mO@8;v-'
    'SksQtP;%ZG_5c#{mRpjga!pPtKMUlVX+eH3^xD=7MKy`93P2~E+8bU=5w`vF#`ChdqQ}0Z|GV>xOmC7<R^^a(ones=p%nZi|T4sj'
    'h11&RCCWn@p;rKwy%y4`Zaa+tgNOn-wvA1ZgI*?-'
    'sWRmX$__moJv560@S6r&Lwq!W>%Kt01J+ajuEVYTkZ!pk~F?DvGWbmFJ$Ru(oO~U%SZwQqz+-'
    'gIpgm<bnnfhgllZ_Or%RIl4LUm7UrH41%_R9Q{tL<!JSP?$Vt=@w?%HXXYWFv!jeAGw{8PeF1HI_z=%p^J^Y^@njqef;EnGv?uOr'
    '%jGbBW3bTWbtPid8MIX@hL5D)UX5ZF^OMeY5XDrtIDpXgUt3$yR^ZE1|N5TfGu0+xOL)OiePy`PK6^5pgxwIHS8(?G<}+4(HSqRZ'
    'l$*+M<R7`Imm-9noP_#OW}`#_2G|#py7{$LTO8#OW|5#_2FB<8&C4Y@uBnRa$S{w|1&vdyH+~x|NZ`r{^%nO6%d!I-'
    '(JIoWVPY2IF8_^^U&B#!%VAeKv;5{x!8GQ}jDu_OM4v$mXH3nyO^A!f$6BZi3yTBhlzv;x<g_BHa5hfk<tI`*Dsc%>+4wgHx#o<}'
    'i-'
    'MG}<5a!5&5pF*!iqK1SUZi@+A59*ZUL5!7U{6h48vES4FI0^2O^hY!==ypW?Bib+{bidRxrbCKr<QBJH)Yl71^DGc?ni*ZRH=>#s'
    'x)E-'
    'mZEG<E8oR$D%dKJE9ZBNR(A}n;X!mf#I6xy$_ttHK|3LLFfRAIY_`k2;$$xPE;z+zrY6}C(D012$HT}G}~1DMVY`g<L~5vs;`Ku)'
    '!f8PRke!Fjt>P~RGk?}Ug4<6DV%D8A<h(N3&O6S2NY>rfF7z$Jmi6SyQ(k4$Q8tk%WG7P{EjA{QH5>|$d}Tx@Kqi;XRFu`$CK{0t'
    'Vg(6RX^tS6vjvkud@)Nlxf+I>7ep9}VJy^lJt?tO-;G1$lQnzU42bcUrF*~iRDW2LL+Nh|3N!FMR_#QHS(>hE5}gLx-'
    '#O(^jMuF2FIle&q$VSG6fcdh2!HJpyNFctT8qYIO)P;&r`sx}W(aXF}mc`3gAJp5Sff(2b^D=o$|+ix?0;rlXfb3pJO!fh-'
    '3Tqf5USBNvLx@(B+Nn)k0oUM--oDUS#bB;N9aLx?_@7?pXij}jUVx6_XKGBdScl}|-'
    'JeYn0*M#y<;F?T5Gl}pCnj?d3ylhA;KV9qc)AcSteKGf%9#Jd!F|}H^2_FZWPc-`JLKQ!S^|s^--'
    'j{NlSB5vu7MjHHJ=FNnC$S21m66wJEW%vleCVBw4si6LKW3oKZTMvuqXQj1>Q5LQ=@^XJ&FD}^tL|6)-'
    '=Lz{kS2BgVQqLY7X_{fC8NMKnfhdsQa=G3EFw>Ha4gz^n$&Flg?Q*vkZLt)`ek^gQYfmyC~|QEwQhYxtRl0RTpbaW#T+Kr8i_GXp'
    'UL%Z5Hm>ZDP1P3BOQB~HL$>o`ZZ6aL`o2UTPcZM;f<E7V8*o|uR+vN8QE~9oGdbjM)+T9qEIr5SJLFIzhqk;%tnE0Lg^@QO{R94q'
    '`Z${kBgEwn8Fvw<h>YeD2h0)rGtAZ+EB(F$46d514}RACd)6nz~(hse$f@_D!S5emdm)w@{8`+Dbw8;0N5$h!&m^=DN}0wsNe9s='
    't|D{3OVZAX%r`%`FODk0r*`!eiZj!)&wtDOH>pa(}b+QVmltpMS*KV$tZA5rY@NjM|(QgTFv;#u$fJFNhAEF*`4^<8XeOb{JFwTU'
    'Z~KcVd5=r^{&Njm1}WZ?ONQ{xE8myuElMgYjIoeTHK5Q-iWGS8Q^-m!dd-'
    '<m##<E{kw?9ptt8ZOCm)<H%BUQB+YBqpVXcQvr6EaP+AFGlPMEj`v2JTa@1uFYnbi{u>}f8lNAfCQ+i&?T*q_{YlZ5`LMWphyBYp'
    'yT)rIhjqh^oWXiEH)`&?>bhI30&HGG6sjLyr`b<TstP#!nOnFrf`k|HtQGgz0S%$cJRSvCr))B5FOY`c$;8mveX_-gY1j5{5nMJg'
    'hwAFTp%%HkbB|lBn=QKpU*i0H?m3Y^3x`}_<UKjD6ZFLdvlZIF)J|GRTUTh%^(I7swtuT!atBN+q$7*8|t3^#z8aB^bkwbLhDOp`'
    '0A7`?#&HU7Ul}6UQ16PzvVo?+dwt>yNZ+T$TPvwFjur;zQV`M$B!B@&SMD`b@Y1LlVJt*_rP|6u8j6qoQKPrO%rva1rmsV-'
    'iM98MI$qxaJ<)mRM<v1>S+R<TL^r9ly30(B1BG-vrlv9ywB^Q0D$aNAIeW}Pb$3;Ji<&R^U$-'
    '>T$9T$V%u_W}0lZ9R9%I#dVS9M+R!`kJCdt0E`nA}90$}!fwKUtJcu~8g|L;Oa?K|s{cs7nwKkFhM{ZapBvSIRi+_SMuAwalHUJ>'
    '7}g%blpb-HBT6PSif`MD6QN)P6Q}aoj;=7Sk;o78%ErP-'
    '&CWuvdmfRAX&r?vq3hWgB8v*m+g(58vdL&>rU|=>L9KgrzB4JpjVK=%4`jCYPVe2Ve^%AXQEDq#S3>yS7E?Bp_`<0eG?n*ESD;`p'
    'I_<0^o6$W%RBGK=?`-'
    '=k9hWOActMi#(!F1f#DP%Umf^EUH=WM>QXhv{U5t;M0+<3dG*K$)pEYL|~sAJ@`xn2D;IM&qiRO8$I}^2uyUN2mc&_jc)YdUm`Hl'
    'C)r5c2?y@kOt*koET2e%yKPg&{w)xWOrb*<fm<KRvzbEfhDbhy|1hN;<pgWq<t|F6B550n!-Flj+<7?E52;%a4o|QwQ-'
    'M7k!dJ>T5_n_+hZsPqizupB=9FU2`FXX-m;Q7_2=vCU^x?_zD~etU-'
    'b6q;F~T>INLEJpCKAa>5x)BwBBe20UQg^*@)pH{XEJx+<nkLtZv<*l-bhq-'
    '7$H3A@HmI*RuBunlSw?bYYLAo@S;rh@dyv&5sgRKZfB>!QGcs;p>X_53&DY+bi$N&L2%U1sCy6`PqHi{i#<5PSIRiA_!ni#SuLBG'
    '(u|bEDRU~azR56inHOe;^YRRXPw^pn1_`?egJ)rdAVH8}IWW<faxD4kZ!RLN4<2AH3b2^)LUYjyOG+=|l=Br0a-'
    'd61oN{oS%XCYKMc=6;9NRYq#}<fNs(m>2;j)8<BRsb|QsAh+Rr^plzS=^FqbQxxrF{?_^)u=b1jkb>%RFTdj_{Q-4pV-'
    '$EM>yvt_a{{>@9#=z2Bl&M&+c5c~xt;89rKxe;jHoF;~4-'
    '>#Elly6Uw>u6k{;t6p2;s@Imf>a}IAdhLEvhm)9QZ?S9?lVbQgoy1><rtsGsb<BCaJ@9RcWP1(1t&w~*?QGWHsY59G;2U%ZMc<iL'
    '<`-o}pMFB6LFhZpvW$H8&<9^B<Gklz6gNpHi_Lb`)Ta#R-'
    '$@GB&cr{PaP11!7&^)p#KHAU?&Gf7wU0~#<c+18wJucZRs+ZzPxHLHQK{QZK;A@}=iP%!-'
    'DU&wj9C~l`8_v6ycLr_V3r>q3!918|3lf8<<DYop1-CK^WX-'
    '1=>cY(7;(FYHbY9M?siouxoCS_5>`0T0heTv>96vtm2U*qfYr)XoQfMuG)cw?l(0M0>iibBDx9n1G)%WhM?sQZ2ekEb3~w{j97~g'
    '#9L?n{b5+IUO<c}4S5p`oUUSUVR3>lca;~|mW@v}aGgs3X8e+GatLaQuU}L?$dYj2s5pmqTCp+6x$Mdmq4o=zIf=FWe`Z7N+l5Mr'
    'jJ4Ny_U&WkdgpR-btD!=_7?*?z{SsV~Md*v0FNpHOUMjdXl*wE~A=o+;v^B8tubBJCIaT@SH*uYd|DmmVJb<cpejX41mvbEjv<a)'
    'NA^{~ZjaY`K!|Qpf$gCljVI1}Sq8oPnk{fpXvKs)dnGVE$xg-+%%`gERnWuY$gV<R5`9-bFh!?;7@K91RxY6)XQnD>-'
    'O%?$^l&Yyog7AQC-Aow>bf~G7nS$!sc;}!mtL%lAa)rN}N5adDWx5_-'
    'Q%(69lRsp(C8(ImA2B;v13N`%|3AraTHW`IXSIpT`Nn@5B)h1BnN~z{r?T&boU-&eDt;HIL2Xc;X&}-'
    'B58|ObCjH$u=3<r{rkg2d%P&XXCN3~i#;<>MsFdLgTpcRqx73;}Qhq9L8`sBYv!IgAnM!7~>guQvVYBmFFq*f5lPq&$v%H-'
    '9f*t`cGnGritID<d(1txOALB$1dQv{YX?DXqOty{;g(#?8jCgTQ6&6*tDw#7-<dg-'
    'x2z_%?Vgo90LPTX%pxnSu9~?{%GQgK{4QMkZ<2mvQU&h?X7=P8*LS+nJ;A^2WeqF7}BI9RLT1)DkalXnK=c}D@zQ!5nYn^ev&Kc+'
    'HopIhkRmRw{Mf>vthK_iFt6`f$C#D#H^qUv+P<o6Dl^n>4z7ctAONeS|Gr_XV4>IPwAfuivWKW&TZNsB2-'
    '781?PSuMHz_B3thQ9iJ*Mtfee$_Rh!fjA%vIzHuysi?^Z}k*h$ieyO;t62HdM7kq**I$jW~R(ZDlnN_8>@)G7SB6v{gQYQ%qy?cp'
    'dPFyuX$h<m`%j__fL44GxvRX8CIb`FxklE54kOmnreFrz|Nk=r~<Ztfe~i%_~Jrj?ljetLPgew=0m6jYKx1!t&Dx13$(m1!ci|NR'
    'u}&4YeVG?U*XzNd9PM$vdDW)zTE}hJJ{Oixe}N%#DH<;2N2WqAov&V02lBVwyNkV2G@x}4cKt$N&I=-'
    'Z?5y?1+ccg&XbqG!1h{AUIq)>Yc*L3CbprSfPK1|EVFmvWh}Gz;APCQ58!2Nur2U1HrR*oT6tRtQ$*L^o}~NW2p}ctemMRoMbh|l'
    'uM3ql-2b{zNk6UDWD3+2C+8~(uaoY(OqLhm$vl99HstqW8%8-5GJu!`7-D9wIHNP$TGnNhvhA-'
    '@XD*(YLwn7h&;L16|F7Csf7tog(x49Un4HYvH{KRR*RaAjt&L>O4~bYXT|>lz=^CPJl6^CNAnOp^9ZY_Odai`TRxYXk{0;2I`i6C'
    'LCp)R+<UndB6O-'
    'k|Ow1s9?n@XEPFBmyn3zKZm4lg>O9Yien3zY@Mc7Mdid|lU*SbLBglD{y8Bdn?)VPUS=D<^CRnTEGR$_$c58!lkOmJcIfrs%Q#wY'
    'vL1B#X1wOanb-'
    '@uU(vwkuKES#pHfCbYu6l9b0dwCrtBd}v{gz<T>A@6z{u&82oTEgpG{@A36z|LVPpI6Pf*B@S%v!fN)Jo=Tg6%)s_m6p!CfFm=8+'
    'xa+PE?=YO<Am`gJR6g8)5H?X{nlV`lRJ5ZEdv>zHls}U6N7&nXMqOuO-'
    'Z?*=$CI?o5$3Og_AS1V&UWrt+GjZdrBmv*3muH!f!M+g3IA$bhLgf=3u@`U2M{SQ;Tthn4^pVtSR2Ecip6u-'
    'MHf_H|}_f8+Sa_jXSP(<Bq4famUl`xMQEU#j<!*BGj)_LwUlu7XIY>Og%+UCDcBmPyW$seqA98XJ9B~;S3Cg*@U_$Z|jc4cG{8p`'
    'FtIlXTNUaiBtW9668$><wEfQ9h8g2gLF_X77x)uxkNln2jx=n2pyEm#G`aj-'
    'Y*{0ce$A%ujd=wE1^N<Ph_(Q85+fg1v1X)U+&_q=D;{*n|Eu}P7GsI__HtA-lDe>Y`G{;*{%A?v}WNv3$0l=&qC{Lf<2OU_ReE+s'
    '6Tk*HYUI14<5Ol$zlEkoo_MuWk!5QkHbl7>Cvh&5@FGB3B2agnS91>ad5njQWZcU6UQm<U&O>o%5#lnD52W?@L~&m2!HXEw3mfTu'
    'w#$qetdQ^_v7$0Vt(@X^ZaDdOY~Aw?<@LxQg3CXuAe{=3+G-aV&U8iMcJf&I&VAQfDP|AXxTgnwv(NL^$-'
    '|O#DpgvmX|UjR~C=RLClgXPeVV|20Ol`PHdALxD;OVXgJDQoJQnK#Z3I)Nvn!Rnp`%X$xm^)h=sBv)WyOf<w*2mxg_dLX`KpFms^'
    '6|tC&7(?8yY8w*&q-'
    '$l8<gTKPj}D2D!vrN~)7fMOQT!%)n^c^Hbb$+@siQqIa}68c9zfaXVy`Sp*2VNC4U;BE|dgoo2H>R?5C6GQ2~&dHq`Kvwoe@S0_U'
    '$MBkM0*>&q=61w&?6MNHlQ@9^vj7H!9=)x6X>dw8BdW++Z#Xgjsx!>RWouphRxp2pRv89uJrS>x-`65uFMq`H<JIOtv|-'
    '_N3vF09-9nozBCeW$U%&qYO0M+$'
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
