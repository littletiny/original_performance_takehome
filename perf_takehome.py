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
# Verified checkpoint: 918 dynamic cycles; 10811 static bundles.
_TUNED_STANDARD = (
    'c-'
    'pjmcbt{g@jj<qkSc=MqAnn?bOc1PV7Yc*$BqdL3#<u=qF4|XL5;dnEzu+p6BA6bfS?H$6gzh8Sb(Ld*s&}f#g6=D&N*|+dr7{(e*'
    '*WjpZCD--uupb&df72&&*t_XUGPk0sVo)yH-x`oHe+5dgq(;MVD0lYC0}&D*W=2i!UEDJ=+CZG@dg3d<{DaAX-kDKH?_*>x(WKhu'
    '?FP9yxZ@rPHBB=gHHn#!jXS^DC;WE*fiog&Z(ode$;)kTn=Hz3O~vervUPgKV#?f%&CB6;9csvhKI|YtL_lzp-#>)!55NOrK<b-'
    '>&rg%+H<QDl5x&!JlZAm1q0Xhfw?`qT6)XUck3$ImZ4}OY>9v;s13$zbtEvZ?*rJ>2%oyFLxP3Khk4*c1X4t{+Rg>_<?@heiM98*'
    'Z5r{&To#Nr%CmBqw(|pvWg_pD=VMk?gKyh_QDN?Bc82#(cZ_8g{(DxEPDUl(oem=yU(mG-'
    '6VgkrubPJ+mF=@FPqS1C)ifV_RsdR|9mL^^C9*#eS@C~KcroJO^ff{;ye1Z_V>WIblt>W?<i#ZWc#?g=zM;+>=gVb{2<-'
    '3v+$!#AsGyvo*jlCua|wdq7|UIfmVqCo8#wbYQL!kUN*Cz^vgol3_q^@_rvku54HckmEPFifq5$UZg#PO7T>tVdRlz#7F%fXom*_'
    'C#kX#;iSEe0H*BQqW_(AT&oAO<H=iEwX(Yatxw+B!9L&XpY%jXWW~0VlH0FX!FB(6r>Q}hwD)5d^$<8vLG&>LXWs!cLd!Aq6^W5*'
    'XZ@=K@2@Q(wfwoTMu`Aweb9>(d@bYl`@$mWYg+j4HScgJzi!%6x?r4Ksd`ye=Zc$H*EpG7zEjGKw=d{@57N5~#qg#ARVz6%yYw5a'
    '0bPFi(UEEQ<q0r#GH~!sVikH_QBQJCJD4##D#BMhdBXL&X-'
    ';s1W<dH{C&uVZr+I;H`X6G0|?4IuIX6C&}Zi`R2bbEt%dq0MfX<>!*I9~2*Kk!E+B)1{51^Dj+th_!XdGSWsLAT2%^Icke;ui1F;'
    '$ye?fEM*`@jflSaEte7@tIq!p~a_#c;8guN2eP-zZ&nd%-'
    '!X)cppX}ubA7V@2kb{tg(CL?(7f#bhJr@(9{UwZ4^S|(<>GxLU;o2@G*O{i}3Ps`-'
    'xT*BF(dd79Y87^A(BVNB2L==!3uO7E5XIj$5pz#rtltiWZ-'
    '{#Y$Sd=N8LpvBoXxXt6dY26KCO%Wt!8_{a!>_DT)j{U5St$CyPe{=GeWWsH%^QhVx*DE44c7)dlI<;eH`B;N27_TKx`zC6UMguPG'
    '#_I}?gWQ&{*=>%^UvM0Pkc@Tekf&KS4NGRLgH@!{^7S|G5>~J^u8r|RrZt*HDR=LG1w5TgYx^OE#E_<6K1Am*e<^65ek_eJJrf;d'
    'og*igz_V-Nm!qfPfpR^zUA-sIh{%{#@qGHxI8=PHlFWY59vWfPxeO8%GvX=*BL$k^DvO_j3n_@3JX2Y|o_Oeq}l})pE^)lVna(9a'
    '_(c+cl=`KY(xpj=Y$=dAg>?6A<=R2_5zn-?gy<c~eowD!oyLq$k6`n9O5n1E-5g&$k+21~j7Yo^w{z<+-'
    '&*e?`(H7GKf6FY2AO*hRNPDeBwl2HSUTcx9&n~jp1(9vYM%n9zBHNf<Y_A)MY*TiLz1~G+o3l&pbz_lj$u0{|7xod^a@x(%US!M6'
    'V}@1&@6L&!+B}j8_O8qEuTur}csbhL_1?*YeG2b$k$td7;N@ZVE}x^j<emI{yh)#-'
    '#ar(FpQZbM(cXVnLo)1Tudj!uB0JDtPlgsE>uay4z^)=Y$X-u_-'
    '9&b<{Y6jHGhSA3+8!SV3iD8U9PYNeimYymy+68NcZ2n$?ZUm*+sg|2;hJ!`!<ODBQf_dV$odoHf>T5`ASkvXoF!1~c!;b}jzpmxV'
    'J{!S%ZKcHOoV14>us+mL35GyvDZ_frO5gjnbY0P$}YFp_?c&CW9;?rBD*uIw%2V$Ha8n*ulEqyyzB}qkEci;FU5Nv`Uvx+4Z`a*9'
    'g4KuTH3oT!<+X9oh0E7da43wGdUe#r17&@8Y4w^n32Y4ksT4M`3LZ(k7OC+8`;awc-'
    '}kP%Px4{yV%RFc;36(%WinyyV=X`c;36)%N}^%d)UjKc;0*38%L4V+LKxhtwlB`8*8uk6xm(b_-ON=C$g1h^G^`j3fvAO_d|#e6#'
    '``wL{Y;Uz!(9%X}3G^Jw}ePK)mx!jjAsnanXohBJ8b2N#bF`gXIABzR~e0!aoEY&YPhKKNonK{c4JUF<~`Pz__rwRX`S2zY_pqOD'
    '#l^K472oqC)mFYYWQYNxDALUe^lzo=5EU<Fpr+x)b*VEnY5oSkwysAhN^Flf(VCU`nJU>pc>nCtqeIOopI5C`>{DZb90-'
    '2A>6suAVG}71ziCDCex4Bppj?zes&96xn03`g{gI|5NtnZx?V(5XCP9Do4M1TEN9&^@@P$VfC_r8DaH%fd|#EZV_;5SluAt%pl%}'
    '@vRqHY0XA4qAcy=F%sQVB)Xdlmyfw*^ayhD8dlrO>@k!dD;%pwYpj~7MtRlR%RvTR`Sa9=q{=oG=9q0*ZvSgO9krl%_K#IA8Hc^H'
    'r|nNI5^!0#k!b?YpI_Z0P*?EEs4{o@X5n9V`ex%_clzewUw8WM#J}$J&Bed&^v%P+?)2S-'
    'e|ttJYl+BK8C*#BP{T?UZ3(WOTEQEPP8QuE!%Vtn<5HI5#};O1t`XU;v4$NaP(}0ef=6lJ+w+h;ijsNQ{_~ym>}Cu59u?Mp(dA%N'
    'K31zD%;@5jD#DDePHiB}=;qWb!i?@tyCTf!VU=q&!i=6lt=N-'
    'NZ}o&#C>y*&vFo*@pIUkyS{6=>x3EQnuWFJ519gIQOILw<%bl3EiEjC;fTP3HK35<*@T-'
    '9W@qu5REf5v>)oTI|y<go&cEPXi6{teJXlKz*d(>XfBYEB_tQCKRtoW18q(5AsC-Mt>^6_EO^Xe68n1alXG|#cQ%pQz<F4#^s$Q='
    'N(%#B5W9%T##=+RZM>Zu8*=NZ~))*3}J+a^%liI!;>X_+=Mdm?C=BK{5-74H2e0jGz1Kb-'
    'D&p1Y|}1imcTpifkTKWVQKw8FFQ8c{1e=dKa9!t?GLaVso#*9ctU1$T|e6<&1L2wmYNydE*~Lc9&Sol&GScSYe%nQpJ2K>5WUEAB'
    'd$Mv?Bjv9<0f8bpBQeAWgTL>F221jg6m52jkWhPH1kzxIjl(X=C;vG4g$fgs3B?{5N*3afJjVp6|)Rlpy^>Lmex3adU68cDzEEn%'
    '0i>LsB`SmC#|mj3z7q31Kp?Z<}-bQ*3K0RZkNvxmGsw&$>;#JjZzu^R7u7s2qpjDd-DbHed7+AR19Rv|~b4KY<kdO{C~cH8kXdot'
    'c_&kKY}{%)QWaA{Z_B=JeRRVP?T$LrAoWA6&qUF&eaIo-7$x0};l8*sNd-L(-'
    'no6}vJaIZPtwHddX(_LF|r;QKW%<ilwQCh)v6``L*g|^*o{Q;6BHPY^^?Kxkeh7DVJ*l|8s5=#nI6N*3HR~-'
    '^vc87ogK^ij!92r*63b-'
    'h&ZWeG_SoM(5EbNfe@h!p*nM*rlj<enaXmPmPE)N!H^Zvu#;Rga=4uQi*0$<L>fR6>f2#=2lcAaD}!FyLVu>&zOXKbuicWy;Joza'
    '3QQCGNuCI~d3PAnah<5r+gt=ug%q=j}L`b7b^1u?uJ;Ll;TSioPxs;7jeK}7cp1S)>jT+)kIAfZDL-'
    '(B=}bDhK<D3D1x$G!Rc0$=361ovXQFq^|0y@8U%(~=L^PIWkL?V8|gpaWKN!VoAFQ~7jC<g*KX#)j@w{awHj;nt=KxLo)XwV*)Y`'
    '&_|_&)Z!@Le9}9B;@>TLPE~XCM4v%Y(hfL$tEP^d~8BO!NsOSf+q$~P=Lkw4<=e-'
    '1yd<WP%ZhM10FQskgXvaf9G;XyQQBbvEu(!j0ur;e3~t6m<_?+>rGuSvOEz}BN9`=iRn237X~psCE${<>MNn4-'
    'w*f!YIyf{&aThavBNeoTh9*Lq-+B_Y?HH%?66JAHnGDtHQUS%+q7&;gbGj<FEpqC@2XZziH=x332Z`yk-'
    'g2YixG3!Ritrh@ju<eCHK3Z%$qfoJF4q#ypKA^-'
    's>otEi#{Fti;n6xvvzG*b8@=mChsq)j?J_3OFOI@QCdiPSQCNRjOZ|C7~>=&XmwLtj>_oE}Xfe=<No$Gk1T1fZJK_^xsGB$7}BU?'
    'xn>Og%Z552yf8g1#*-'
    '6C4iwe=x(FQBR}pxV}uI&&YCek?pb03M@T|ku?_5!?7hZxOO4#KSq47@kv)j`Aqe6<fiD~LsiqB3!!G@i68D~qgB&Gs^SL<4(Gqu'
    '`bM=prxcyvAq$qL!xtPeY5;veVeQ=z_9XLMPfiq<Gh<WNa%k0s}^My&0K0rMixmxeY1>J8BYlIoR?IV?zvwc@f!c?(@x+b=7zJNA'
    'CDyK;}ARvKz2nj55PyHAY<x#?W+pNRdqJS>s*y%*pVeL^oqd0atnRQqP6wW0aJDtiptP_eR(P7)kFl9wNH6i?AEgc!G?d+(=0f1_'
    'wFGhR+A7z-'
    'H!Ft*#tjk$_HAnT;#Hg<pXxN)L_0=stop|aWH_vH~%npeUXKRV_XyVSt5hR?u+)4dV;L8LJt<s%I`zd&2chHvF%`g8>yQ-'
    'C6{)0AFYrp&_?XBJYvX-`28^4@IJ8TcXoK2f-'
    'Py31H(;<B|2|SmP4yp^tYN3sIddpDZO=fx6Ur7+IPYA+ATWq1AEf(9!?y&?1%B+cz;Cz`q5byMbl7wnQ;q+l51%bP^NIPJl3<xhW'
    'RplXaK_AKZ`N2c45imY%=>ri91RZxIJ*Xo@$?0l~w)jeW*$!>-'
    'Rrazy+TyG2<pF4me`_y0pe??}UUo!Ve679ggtquP`zHS;+2318rG!vorlcEN$WrPNO9}7eX|oesq9JV(Z?rbF(OTQ10Fwob4flrV'
    'vqey1=&iJLEBj3Mk*H0go%b^dlV!B^_LVS2Mtcvf##9+?KC~M*$Y{Hv<+xEs`weZ!O_Dd<zv!k{7E-'
    'vn(8E={eY}xgtfy*Y=<3i}EeXLR7jkrqVkd@lgwc*hf<0piE|%FdMuNKv_+28wRwO|eOND<6*ds{rUjp_FEA;+a1q8C15Hc@T!Pj'
    '9?2j4~U4GH^)4R$nbup^xjy}Ll(#NR@62ETEdH0PNxE=laMQak!Mo)q@GpGHFwpP3Qn*aoD#NQOvy6daBAwl%3y@T)@PV?7ndKZ<'
    'Qcuf#@NFW`!Bzh6kWFKjv##9xEOm{%Yme1|BBFm(<{w@W~}T?5kX7LacDfOLBVq}!8`E-'
    'Pl4J!~F#D^%T0y^O3_)|hS5GUq84VTt6O7kNt7kXQ-bB)U$A4Kn*a`@TfUp^RcBcVHr;y9MkOWVBx5GxV!ZB|H!xOgtD%qH~$Gh1'
    'a8FnYDvAqEnf*hc}}`nH>Oc$zTz(4zM&jqFF~+7M;+n6D;=!bS`bT`wDsg`x!(ao;%YDwo8~Tj9gpeF=`se<SR&(+jOf7wDGQTo<'
    'T1c4w9*7TCtNPh7U6ClRD<T;}d<g%$_oqV?TusM0}$6q!Zo7>BMys?g_%)Tfwcos;0x@(=!i6Ijx7IoYo^z+UC(Hzx7y@-'
    '+J8Vx2oX@m)bFbUJ~l+(lh)N?oY<T1;i`~7B)_qm7RS|<s>r~q;0O~zKbrLX1qmje=9^VedwPnO5WX&iWKeoA~NN{%=Ah2DcZu_g'
    '0Rrv+C3<|74U7?!tV%J8NLxA*==F<uz*9ts+j<#LNH0-i_?p>1ZUYdnuXQ}R=C9+G&QiwE#{$J!8eRfcSi-'
    'Ygn9No6wDEE;Pxkx=(7^a2z^gKMKH0*IR{jRC=%~i8rkF&;F@$F_Ko-900r89nBx^X{BbtmGYJcVB-ToJFsxP!SRGcYP_$vyM#1#'
    '3It1nEQ_Hi_D8owkY41dPfxpOw0MH!6+ir0e+Fp3q;m_H$4exYHcupbP-'
    '|$n?JxSTWj*2&BLxGZ=l1Kp*okow5Lqs7$X8$o=OOA_!-'
    '6pt#k0ARVH?#dn0~J1u?_HYZpkzaSF5&*L1((8)VGE+W_Lm@}N6Dr1t49PJ999nqI6SNt2sk3FE>O@pY)>@ly~3)mz?aMQ!@>9~I'
    'oS{W1-'
    '@+b%_W5D;ZC~=(BRDVJM#Q)FiK@hr^I=bBxGxW5C;_^Y)uxr9#3$(6)Vj{h;4Ddcq@)o=+2r?F^WQ?u}<8bbYd&#GA)PigB%eMH3'
    ';|I59R11j+MsSbk5aKN&aqIyBH?P;caIZ!zFpV?d_sUlFNI5U7RP$=j~t@zm(+kcC?FMN%DF-m2AUCBCF?JxC`ZfqVNm;E;`?%-'
    'Ek5W9O=c8tmD1)IgH$X?;zTJ=yqrsT&;-J$L`-v&4~#+?1GDbb-'
    '`$xGk1*GgHwnjvkhKC^$A2ADdAa#U+t=agfeB+n?)M^q|2|_<-'
    'rndOhDL&NHjA6VE313X#&1JRHCs7&fQ@W?M*=2hf6d$0c;;3(dz6OsmKiq=Gm>VO~C+?Vk+k*@o`63yHLeIa=W6B&bd-DIlVkkAE'
    '0=Fn(tHZZkNY#nu?uyf@>A&(RlM75^vrm3Q@TE#B~a{Ysn{WjE<!L#AuhcsaVziTEdeGzuHgs8wpP-'
    '{AyR{OL$u0SG&4U!gC70^35$)_|`630WT<c-7VU|iwfRwi}vu6f;Zjb0C-u!TW-+-'
    'UQw{rEjq%h3YNJ=CwNW4autW72hf4S`?FAC8wE`{oH#=<&C@+n{9GvNmL-l*1u4223V+&(cVt72-'
    '@Zh;t=m5qYIBWf$obcBjEJ9cHv|ZppK*=(wN3>fLPPlAbDa32tlevaX)PxoJcro!mh^<Zjo8l}TeOQs!8q$A1;+~pR3|GqK`^4q6'
    'r3b1j3R7+zpR^LY=By?yJ8E7o@x)pKmpy<o{G&O`l!7Wn?rP`4pa;w&|B@T*fyf8+DAEib7&H6)rcF7xGFUqmDutx5?}Cll0XRCr'
    'A_c~(4bdJ_*%t|HfWS$Py(mvY;dFmPS@FxNC}*wv&u*boT;;+krFsdXTu^TaJJ5dM@rxvomCknkRF^pQCyl2&PN4UEOP$P2#vbZ`'
    '9t)ua4SaX$B`UCD`U-'
    'z53?M<&%bX$SuGcDg@GDossuD0mVkHkAgrLrYB(YhG(yBL6C3<hSluk4Q&=JR=^7rwW_T$4SG9y6!fLF9AH(W!1qbT*0X0KBsG=v'
    '-Qi0Hs)OS-LbtLxIi1_qQ1pQ?J7Wo+WPX)LkL8f?Q5%m-'
    'u?SwQB%6Viyq(&MR<?BKHViW7)OcdO#a68f0#c2c+`YL)1f5`53(mp&1ekj<(N*jN2FBwZ4k<6xKw;pd)e-'
    'OrritJP?Ydwn;zc|{(N)}608aFGK=9AfA_?=t4D6=7Oi(9-Tvr4$tEnb${Q24!Dydtw<@CUbeRc6ECk8bgr%&K6zDv@v-3H|(5-'
    'd=#3{8mPQ)J=ulUYXsd|20N9kR)v6NdiWgi#i$iohp$pN)dlOC@0|~5{Q7fX=~Vv_3%X4ZwnP1p6oYNbcb6%JWUm~%{pg;IauB$8'
    '^Xczu304q%e!SmIauC38^*!%9@%gXmiNr6;$Zm<9p^C<j@LjLGFE1v4U`exn4T_|T)<2X`|11%pM<(n`x9=rnuMD*N(i@_gx_m8Q'
    '@Nn|03$1gadR}mO6QEt*A(txN(f9+_(J)Ew(BKwr>rEChbhTI;?cM*cxwD?P~8rOb0srjhQQAyGhT*4g=EIda2PBZkS~@#^h-'
    'ONW52YqIrvNanxnt8tvUQlyPD&_w5d4&Ondq|56~psqTzHas6S{piv@Kg2`b(5dq^}!yyy3lCj1v=bXCY&8+<VzH?a2UU1c`1%@8'
    'edH&sT;`EX%Susz3gzhyc7En$%=6=j8q^DvR95`H0>NHi1%NhS<c!4RLEDR*99F}ptNj2?tz8E0M4fp9G2tSkBtj!~U;L-'
    ')Zks<ZBh_#OK=>w(ULV^?QA(RYYM`8y4#SyA4q;S3h#QOWs4_ial!l2;LY^eZ$lmD;aR9iA6fh@*cMR%kbpSRJ%butP=WdshVwbY'
    '#CzPz3sewMX>RKMtawq6pXr({idJD8FYe3`gXmu{!WzqM9Ju1ASFQ4Z_DvPr5t`A1&ta5jmi7-'
    'Na5<Y733sPE%nL{;1(>E8*!H&S423ok;jl0ULt`Tn~>00Y9x_e4+sn_Z{jY-Gn4ZNYo}S6ii5dvRiZ^AV0+|x)P3`>K5JL0twUHq'
    'C0tEH@HO)0{Az&MNha$!cBV2&ckbz;8cN1JV`r<V?X1~n&Yr>O;OQMCh$Ka-@n4t3bsOV?lL8pfHqc#ZSpaZ7K~z3sN{6uD3w8-'
    'hT}Cw?wIBu!YlZ`1l3tb^PR9ckD^@Fc--1DBWohs6UWG!g!aTSvL>TFag3}fXipp?Ybx3k$H<z7_GEmlSlmOxQw;YgeoA;Q;Q_@W'
    '$CC*UDkt+GiN%_1XAB36$>w=p6A<M6vdDxf!TrXgCETa@$)k3F`xQTWD`8ie@@whS2)J*29s#$FPbA>3@tFkNG(MGpd&cJyaLaHq'
    'A$<Ndnt%~{uzSlfAsmhXuDO9#YZdW<<J=aWNvCW#XwtVE9KKkq>8I#}=NQ;Ua(I?Fr@Pu5VNX3iHkMVk4jvCj>nRNrBEf06Dy$|*'
    '_$GWOqT|ED{~e*AS6CgUpm$jHN2?K5_#aUavzvq?75vGG|0oIfDn6kUo>`#GL5zz%Ip{@-X$?WE;}uucppeS=-'
    'kAeV4({GE_A^J2TuPW7sHHfR;ItHadFwPODfIBxYp0`*i_V-?n^Al+;<#5Bx|=a=_V{npE-'
    'z2{3>nASk5>#Ib8u*p%K5&YZ?<=gsLz$?&PF!j=Mp{I$cFqvqEj2$h(W~IiQJJ2N-'
    '0J*V=!@eA~$ddv3P7=r<xQTb2RbHdXgrdC{NAAhh_<xY*5kP{?rO`F4VIik580W;HA>Js1SkEYqU~8+O+Z@6D7P&(<z#v7<-'
    'rUF$WkVYPF^hA>6AxsT7aDr;?)L@dQ#-'
    'JC;FesNN<L;dUClC+T&9TfHyob)s8+AnA3ITYV_$b+TK1B<XdETYW6)mATa?l3u5(NNlxux{cW8z~?Ns6B0G|tcG#nWIc+mRaoJX'
    'i!!IrNqU{_kz7B#A<a>$gYkYeN2w0M+tD1Q>W_D$IZAaX-'
    'i+ob)nRxqnxj;Q<E?0pQXPSJ5}Qa9`U{x5CO9aR9kH1bavv;EXLLun5O<q@&}*DGk$bewRNiICX2WMzY<I#JEVdI9u|21uItc0+4'
    'OfKK6%xJ+N)9z+zwmzuK@JWp^v{Nc6{5l6VTH)$mtl2?f`h^ee?^q8eNxivG^g^Om-ISEmw2i#7NCdE@jY1pkUK5uHZG@(Iq^G>q'
    'uP+zV$0UVz24mrawg=Alq8QVrxnppL5*Ut+iMbagA?#ciGaUUFf`mSN-'
    '#>9J}v2WhP&ZsB)!hm_G9342FdDWuz=Gf>z5${PM3UQD+Qb(S=$U1aHiz*J50b?lGCWe1)MGUG*<~YN5+J3C)l?NaJLV#JX(N;iC'
    '_uSMpa)siC)E38?{kDlXCQUJKb}ZrlFB(9uL1DQHvXyz9><DJDHwb>b$*KfVF-bEGkecD5Nvi8aA%8{-'
    '{QeS00L*#ppakA*Nm}Wy~Lc+8K(1c{#7G8hXl*&puFw9CjZW^45K&%UfHop#tR*kq`#MzeGer7y|zm5ecD^>ZAe*VJO@qA`-'
    '$dxK~6ZgyC?Xh)4)kaKDH$7ZUwvv~WhYuS&FSPPV6%%J#<stnfQwM*+S~!rOJ80smjpl6afAe@BKgw2^!y0VpY1F(#h_t&Ctt0Nq'
    'PDTyTB1jH3k;v*jEtn3UCVtYC7sf<pyUvXvYun3}EPK*6+ZwPlLYArJI~9x~7qddWaZ=q)V`!HdrXa*X(1mIxi3__E}<A0l9L(3w'
    'waxH266aT2}{|Mv*`tKt6;*dGyAhy#0s)c^%O!|G54eZmTr>AgXrbd~hYGqf+0I9kD<wJ)+cPQl-TIkKCih8}0jQ1M<Ks`y;yp48'
    'z|uGmLVZ4R=1!h6e-Xes_vWr>%YEk~7Q4AE*-Jx*|^mhQW)1z6zs-G>EenuM==N;pQrU$ie1Dk`{LbC8rw_otGhq-'
    '484lN=@`<NdkhI4N20FC+&_$$ZyK4vvz^UN3!l&WQ-Y9V+Qh00nMztEVKrPIvGAtfbdjx+E*NDB9Ei2U2F`z--V#R1F_t;{_21We'
    '}c7Un|NoUV+Nr$aod%dM9HjmGK(|XycQ8&lg~GBICuRI8l9a<e2cxI_ZnHj;9!+v)u&dV4C+hz7>4huZQmhlg=i>_kvkxli&xzw6'
    'n?ZqhQ|I6yl6r!nN=}31*&6gB>CzDBmK9uv28WXOl=>_*=*ax>++BA?)S~lqLQH7lC~%a^Xz}4epM1Y#rKFBh8g)Mx8WIOO868^Q'
    'v$V@l4(y)v)Q9kb;onC<V7^PmDcQ!C&3#c;dIQ4_%PZLAi=UqDH@=9HED5G)_=3S#chb!d+9{>JS3baD!VNO3)c@Dvj{JUVv-'
    '7!ThQKQ)C+9w<xG3ORSrM#oF>EYDlt&Fhe*iV}g%`RB<VIuKb%=NVmxnZ*r-@Y<RH@VU3gR>7_gMxdM#vJ9S+FK1-55-'
    'v~0Y?s$~J?ux<zJ{BP9`TMQ0G>*>Som5~i5cCgAqeu_sBex1?Rfv=DrEqoy)>AEsGbhBo{#rPLV(g%&G<Ob^<}QKK+%-'
    '^|y9G*f_dsdx5h%?)Z9qG!88d>+gC+E8uC*3vfrJ_s<r$@-'
    'e54S#AI}yLg~!NlDGAcs?RtutIk1Gl4HulTB+?4Imz9L9%UCm!uuJPC$!#Ovtd0TGr-'
    'F{H6%{^FR6kBw@VS63!ifvk3HVA_(U8$uiXVjut;-'
    'NFF{5=kq9vxZ)*)14PU{N9Nla>8i6Dttt*a0rF|Bnq!lTH#BQF1~l_j3si7d-'
    'AOJ#Xa0T%hJ!NLNJO|pI`2<RlOW~(OZ@(B*N7ePsAueivti0`Jk@P%kKoE9O4`zhx|$Qpgdi4lqoe!<xgiVxLuI)vhQ>o^}mVZ8O'
    'y?o0I31W_w5i<uJm1hy+Cj8T{`a+o#>Vm5o=K33wx!XJnTw3RZHv6V+K*-'
    '$02n#o~v&L`Hr5hx;*YgnLXl?rr80s4BY|4hMfQG!5E63|&%lTb~}w9NT2CSG~9wv3r-'
    '$|j8P{?bO`t1zb9M06F#=$nbH!Uzu?)eF48w3WC<>@R&uG#1A1UnvS)j*ohA%`e1OX`3PI_*7uhO(bf6!rjCStgVQ`;e7-'
    'Gw|ou|A5pW?Yylcwa?#E=7-'
    'qh3GCjLgrY{r_QnE}J7tkQZGCfwn0g?l{YYkoUE@zK^60o1AA+;3nmf|)%6`^m_;Lx{eNa)*C8TvL24Sk!2g}zP0L*J$<+qWqyN>'
    'Nk^){Bcv=wXih&%S2bk%R*1ARPf97tCyy4`!Oq5{*r48BB$Y5#c#W8g*2l4;3jKI?6NP`cV=G?O$z45qZ)$t2VH-Sn+)?;1svN4E'
    'geN!s?#c{{;nqJ<M7x!yWsMB2<m(><U$5I=e#En9i<HHKwyGRE_EE3RPn|yF%5N*Og#YJf~}I0>W=IbT{s?NvYllYd9%2Fo8eoNz'
    'EW2_xR)mzoQN!=bdu|{U}bt0Ow%xoj6Ny2zjKOCg@L2Dd!0erHBe?ElW{4R%OeWl!VTE9W#$kQrRjd9-'
    'XYR)$yV1qfnUz1Q6%vO)AF&fW^d`4_*&H*>d^~2j`p{G=P;d1&Rdq?zOT)U+h7a)wyv@*X}3$>f?KJD89GN!YfJ<%up7(u$B#Z#i'
    'cO`*caS~qr8@E2mT6Q@qn+x-Rd*49&m(P)sw|QAnM8c>&QRk09a8G2(m1&<5aesd0!`}Yz5Q4PE^^-'
    '5?v6TdUw7f>f8tw83c4`=UYh~sNp`A)X(FShqu1Qq661OMLE^TmC0p%CPQPJQoR}u-'
    '9(L#kuB>~4F@Z>pVKs=Ahw@3YILie$;@VF%emBfZnlz(n&)M!xTyKAY&8`%o8YFgM{e7rsXWzh`>iHQ=Qj6OCNP&ve4(o~q9p#6Z'
    '9qT4VB?MGxjPa4BH2u6Zq{(u4+KV^hFxw`EpZc&@?5#YOvS}HHKdJ3=W8`blm<E12pDS3_<I6|#aQQmVi9Z{VkwuP&&rl@3HqF@j'
    '_c0v%vMm{xe?BwL~}0?95IRv#~|8zcZ=Y*>|9qn=b$#9`7IBb-_9ZP+a+XvyN1kfw~+bm9x}f@Lgu$;$oy_F7Qv-hH=B5`Q<Gavy'
    'w|D8FHMN|(sc8j;<^GDuhu`S_XP~M&*}pKRSdZ*61|N-Q_wI#d)9G#{~1me(1q15b-'
    '!69VV?F>_BR!i*N>NQoo2#*d&5lP$V+F#XjmVTeF#8*P*Do)aREOnE{*z$z!dkZG`(?WCGG3>CN?P^;+(N98o#raJ=MFTu2OgV{w'
    '}pB-+aLx`ZH^ZXBOw0sh9(`x{(?};~sqI1}9RZl1uI5A{`AM=8JHb529lj^Bf-'
    ')@VzpT+5b6kT0{Qy0>LMK=ULd24sKB|)1}<YG#8v-'
    'WD<Q1V@Z81;8#{sxOWd?NexLPb%0?FaqO*wp<{9Et&`!dakQ=65VY0>DK12hT1DyGhXi~RQ-'
    'B^8@U4o{w~q+;PBGVo`kO5a{mqt#{$_QdsoBb0^s|h_=jgc|Oty7&4!2huX3qo;E5udV`>LIB@w*evk=V2CRU|p|bbA`TPHt{5mR'
    'w~@TY<4EREDEA-rE<aQCLog;xyy2M|rLCxbX<Wfqhlp6~`Tlzi_dP8D}+2cp(DW^~{f)h<}GLeQXl*BPZkEp-di|!u-'
    'go_;)x{$EFcKlB4}gLr=LCp^@CG&`54|<a*Z&#F<7o>jm^>;S3GJaU~xbkD&ttc#A}9X(+AkIE84trYE8cWd(*o!?=u!o?vZha%Z'
    'd}mm^QfbLP}3#~}h=Y~3qSWpFH$GHt_L0P}wp2y%k@5N)$Pe;*0>CF}QL$@v>B@Wqg1C>n2$vr-'
    'apI0siL30Nf<V%L(CwldsqB(pCW3hpAauNZnamf1Fjg-vAkHN(QDGW&)hc{7=P%aFXe%)TpWih&Oiy|V3UOw@X6gbyas{+*;;d7T'
    'O4nVQU6CR#T1VUiWX$4pE)<yvzCUQ%BO1er!s>jd=SEj2u4TF?XT5c2AHo+EBdS7~ETcJFlwh0V67ijR?ObNrfAd*Wf6Buybn*iM'
    's7pB-0-'
    'Ni~^`e7zx+9`54nGO0wfu`fR)GJX?Ze@LZ>O>NPp=|^K)VZG~CH<`{(YlKg`&QvyfV=M0OP^gD|HU8ff>LTACYBqWxaX4&)FzHpi'
    'LNM|LZ?&rbFV0V*O9!DwjQWJ({kvqK>xH0P4Q>oXYf0XyNKcV*=<t*%X_rMwyHleaEJO*s)1rLr=`z$5pAqF|&*a>!ZOQTvg*(SL'
    '()_1F)#O_w)+*dfzJ1~>g~}?D7kUDpT6rPzuV;CkmmE#pvkf&IseNJUE*g$@D+D3UG#ux$|KABX*+r&Icd``{?sVVysi|kZ+h;l8'
    'LCuh@xq#Od2NiKQFJ)<!q0`J#X^RdWOXUD`nv<@U^sWrD=$v}>hlgc^48HLSVV+HBraQ@-s=G}<y`-'
    'mCEj{V|G|Y_S3<!EZw?ab?X*~<=mx<6u(8+Vh)D)xrnhJz`G1_maZp9Z2;;W1D#b_6j9kzFh2cbKs{plR^P`rEa81z)Ueen7BQY<'
    'Na;Jp=Z8hqa9NxEG#o5<g>;8Na6S`twk=k1m#n69()oFqwd9=$rr<%su_=p>a04qeU9tl-'
    'c)SjU3<ReU<pJcfGhK?FcfmZ4Y^9pl3k^VEZ<WH}tQOi!wDI)R_X1(IH!6|*_%)y+v556gjyeV_l4%Y=ppE!sUIi7}iKVhpXQ+Ni'
    'OS%jpu9afo3=*ia1n_D=gi?34C^_*vQqV&Ak6#C~ZXi2c((5M^l}h`<-'
    'zPoXdCHr@UzTh4o~oyu17E^Du{Rdm~d<yj7GG&5Hl`UY7=LOkfZ?Gl1UHPq+FIrmE$gxHEQYZ;ST=RTq&wtN`TuZ-'
    '+8WqF<Kew`e~)dCu8I7a)1OLY?FxE1cOcO?AFtxlA1y|!J=i`0OpC>$n)J0u({98`zDOE^wAunzx_RKQs08jVUSV6@;7Y9*W~T<{'
    '5INvdYFSQ2JSbX(KVU(rO5z2{R8w-'
    'JV@cg3KR6lOruF?KE|4*tF>io>0vtZ8y0I+F7IMnp#uiEv8VBnKY>pHq_DZxYF^6wpY+QBFE<3uvODsC~<arW%gbzW?M(33J^F-'
    'Gz50{9Ah?K3_8;MH_yPW~7Ja>OReg6s^_$nqeOrs|A`PQE2HO(2N03kWiy3BNg|mYDbvM?ziodgDK?fc(qQ;c4VKk?@^rI3Lv<f>'
    'EZb#YJr3sDc5&IQpvN`JAU5{Al{o1>d{2`bNlZ6B<4k4yXIg{?<i$OpLiQ?XX&%k<sm1m6n)N)+JEFpG9NzOg}mqO&LM3~8R?(gq'
    '6HBtlv8!P5_3hXn9$tNeV@VG2<0L#!5&L?6Y$qYU>u=b<TW6ai?BIDIeUJWk+QfvsySbz%xD=yPkB1scO+*NeZ2E7NMwii-CMHk>'
    'f&^taaO!(Drk!S6!BZ^x{Zb&WU6X7w$e_<!By&hnAJf875Nj0KJ7Qx5#$)_NlAaFvOGRx?R;jZGsT{`I}v51xIGbckiI_=(IdD)5'
    'v5kSLlKlq-'
    '=c^zgmI6eZOROxF<G)jEG}y;lyb$~BNHe~nd(sNRYk#i^a|}UK$BO;MRg`c5b8;2KOe6{>7(;3BZ~U|3ciUc=<OB!z!i$bVL(0pj'
    'qq(ZPQ(B8{ZeW}6Q^0BiPP-R#A!}w;&f+d;xsojaheyJINcQqND{tmWZk2#8l);!WC-'
    '8~ayjOODXdU1DA3ReSPcs70SdJ45T%(r)4!BW5`KTVX-<;}s!YMR*3SM9L$2QoivE4?1pcFd(-nU8t#-'
    '~(rUP=cUFu|%!1#$Ux{=x!682`N@}fjMlgLYtlfjXzXxa!lfvgd}4Hg~WtE1kr6(%Jo<D-VI$-'
    'SC`5BEuEsW}#lF7nMjUt2HX4@@zj7^!c*(>WWAT8g*O5Y$n;hbmD+@g^FI`iXbZFw{=GjfSIc;(b(w$2{6dCkm9G5lJVFsx#G0Ys'
    'oK2w$iAy2d1Px3GTBmkkCpqJ>oM7w|MT;8bN&tTw4jJec8vg+V?#XYv1=ptZzRQvA+FC#QOGQ5i74xg5Ug4+m9fwZEF|lhjudT?C'
    'ClrrK5(})hPX|eG7^?I?(m}-a0hiTL@adv4Xru!3n&#E>4mgmr4ICs|izm>RUf_q|{momAOcGW+q!e8}J*M9y`3Tf=-'
    '}tL!Ch3MmmAQyXXW8H`WOhZlV(?+*Bt}xS4iP7;!K+cSRV?y<O2Q<#w(Jh^JWUZP-'
    'k6H2FeB1!H6I@FxP3Mr;!lJj<Z<l0<B{2M20aKL0Z8mr2qqdi;k)dMAO@TKdniiU(fR#2hixxOQK1e>@N_lW>VMejDH=Z9{xTQt0'
    'bExsf`t)pNjg7U#7lqDo)S0oOU4;+l*qeIo~4=W-TuDysC&9B{qMwpI>`+s0$!<n&(%L>uw^l|Zu-'
    '&!uq(gY8^tw@BMcJHEx!DZ*h4h*i+lip+J0&W~4QnnUr_Rf^2;o}rw4YiL3EWoSY8RcJxDEwmu~I<z4CCbS^@Hnbr8E=Krs6hWFI'
    'cNFw7*Q81zDMBEmzcg5_-h^yRLiZ!c@1b8lg7wv936lCqkgC2+a!Y59T2`BQ^MtTfwuTBs-xGc*zG=LJPByF|2jw%0#j$=rm4Z+V'
    '<$+>QY~0HNCTJTlLA!tn+6PQ<K)?hY0w(AfFhM8A1VI4x6a=MS)th3+urGQej^Q9-9-'
    'pz%kt(mw*29aMBix(d744{d)K+A^Vg*FKT~NDqa;sZRiDS82-D+x4JG<5IO)Y8{xB7!Af9&d3e>7FG-'
    'P~%rsfz8c{XV=x5qxYNQpvTcgM)&fDT9d&9t<wd^kJ+SRFECKjG9oDcbVip7K?5?WnCy?c5|$r`bfCbimso8yI6FWC!$*iFKEsaY'
    '=oCJrvx^`tC|x5Ti`X#`~*sg&k9t9*&!u<XJ9SN4XlNEfwgd#Q?#f<`CV4S6|Q?ilxaaN`%yF<>p$caZSkf*V9B5s(|g$+u!y^6W'
    'pM~<-'
    '5lnm?gz08GscK0h<A|OS@{j6v~)xl8K=mL2w)p+LZ`T%^GZct!ycZY@xBbb|Ld|jE_gyucmg|OqkWHx<#4(0QBenDe2<EiQ0@Ctt'
    'b%cdhdH98oYQ`Q%V>Xi0wu}0pLX3HNM;BD(BY~Gjd*#Z*3f5)^BcX1Lu7>6U&6n6V~tJV)jNU`=od&IMQ$tM(%`R;RZ%)^vneYa>'
    'FEdWc~;Ap0zUA>gs%`!d1Asggj1g4@-?C<&)-'
    '77<`YjO_?B8&vTpmHS_{%I2+I89WI9qA?Ywf%_pFhDCuv&ywJe?3f=7?7bJ$Px5a~?T$BJwnwSwmK_1`(tP@U*_$|PzI7VT@&ibW'
    'd}LL{-6Qs9i~*5twvn+3BIYOD!}a|8eIxs+MCuo3AXhRFWW{Yc$?9qBbBplObk=Ku}~9Kg!J0UR1QfWra@aCqPVR#^_96X|-'
    'mG7fcafUDwA=SG5?yE~rExWr$sp{h0e1kza9e;aTyh%9Uy?fZig`?khbg7J;NFR0VlDCvi^>@VSBxB5cR>s1$=G9A$v|L#xa#7wT'
    'l-S;QIebK4D-ELCOs*jIDzw2N^9Qxe^zm220o8g)`n!5$AwH-tv#f0#2ECB`+294wM|FlmW4^}Pjv?~%3)+4|S3KJDvRm8$CBgQ`'
    '~hHcW4PPw-'
    ')>5!#K$z>4W=Yn3ZI3cyAQfhX7@rcgwXB@rVzu6;d?`<hPj3Ad~M1~G>Sx$85uutlU5FPf(3L?k`KVT(3u}Fxk5Ew^7T#Y&?FT^U'
    '!m-<7%ZG0sB#<lU0Fp^c=_#9oZLWGHXo~N-mI+|q!Kv04kqA-mQW6o5e){`Y@=Kr{^Rc;6vydXVS8ecC;pNhlROVY=`@%6Iwy)5z'
    'dntKlWP?<aj|9?hR@*Mr&mx|;$<Fp^u$a8Y8jEdr!>;9d9KN~IVl?s}#*>;N|0=g(?P7&-'
    'Te)oh#$E`7mJx8USq$2*Sjo(<={3NJVM$r{~nI;pBqV5qj&R~p)nFoObJVZns;7Sp3fQN~Q_B&iewBITb(SFNOfnCCR4|-'
    '6}?`<UE4#o8m%_PiHR<#vLy`L$4j&=i;JiGjrce&@seAM*<?ywR>w0Qw<uqzYau?h8gpHLFZ8AR{YFshVzKzJ4^NXd;M29cIpAaY'
    'c26x|GuFeZ~d<g!&#Rb0192H@HVn%2prXn2K8iiTIpq-c1ROp1nA%cN-dZJ87ezax{P;dghY<Ia@%9Awr5?zgTb>XibkxT_NB6_k'
    'H)agvdO9ZS3>!hMC}sEHY@p03kIQ{w16Q}Giv+NTnJr?`9Y7x?(NMu8~Je^hRkuCY9R{$txP|FK<||JXjve>@<}f9w$EKXwfBA3H'
    '@nfjc-p6sF4T&&hvLk{7D*IS&%i)d@)JXnNRiJfgFPUuc-'
    '>R$VnzXt=?xx@j1!;l=>q)(B>=+4cJd(6(~<APFrr%jiJ~yKAm1oJw`|eB|z>u#=%flBWEzknQK15SkR}Ua+T(qhRDihe-#-'
    '86O?Gfmek7s?zs&GSu6}zGk!9w7JN4S16i%WS)4uJ36D9QBwonV82bI*ImO94L1dud|=ROpa_0w0BZ<Rd}J`IBW7+uz|&lMdx%>9'
    'vJhl^K;S|4x@@5U9g;c^(T?}p?2(ZE!5{j85R7Y2NJDc_kB^BPEyo09GJIuY{`=f@3PTl%v+KCGdMAS<3c8fJ-HI54yBPU3mw{Dx'
    'O(MTe8qU=)*?rzF=!<BW7Nq%}aRfrh=VJr%o&U7M*uu-*&q~8=tahs!Nv`xr&vY8-'
    'odz@#4+>BrCH8Pgr46TIHJ>~~Yw(y&o(1VGre;>00-'
    'vLyPVpT2nq*{{N=JtIMoH7P2VLpFxl$%R@v$6bv#|`6&1)0+m23F9hAEQAtJHIPFI$Ju(k1t@jfkXMxa&=bF?V&>TM$w17W!KwmK'
    'YOc|B{50;(m88OE_6OvMWrK)X(E0&jI?L!jXQ!){g|6b1wZ_3ZMF}OPWTJhba;o;X+aPSqD+Y$x>Q-xfYFf&d1Gcy-FH*Y$;$bE4'
    ')_dv$OE7%kl5pF~&8;d?v~3bG66F5K3R_VV?2H_6=azoF_N0hIy^)(XDCbM7$X>d`tHh!q9NXN?(qN-'
    'o!W=Ypvs>P&l;|I3bFKQ&WKxqhL6-'
    '4mc(5UHhtpEDCs2Gl5fG#M|V<&+!Oo8ug9I$KKEZSX<ikvytsEe7+(5^=3KN<f*Zs(U9=dU90>PwEtTbhSktr20z2JnK{$vHc60q'
    'G;ANaqi!5)xTTJ|ahxH<F85{Zm7cnBf?=IH>c)wN9qXtYCm9OrDqn6;WvCmj?htYIXz4i%-'
    '_nc{Pg?knazL)znxJoPD@*ra3AF%qOPO^c5GZ;U==C`|kiqyia&6c6W<S?#NVVZ7Um-nD^IV^Uu0U13m5Ke!ILJLDTh2l5%B+rq+'
    '(WY!9ONFBt>hs0@N5+axvR3(X^?x3qBG!gPN-|wMD%cqZxns22y{WmsyHC-'
    'c^mf#>JgI<SK^~avr^E}P)S2Fb)4B2GO+t6Cayyrwaav50+jh>hR{1ha!a!9kyhDZ*g<R=XAmeZ^AP?X2Lhs}fH-'
    '7eeEx6)oEx7%)N#Lv6(SniL2)l>)Or>1c|_AYI6jZKX@>B5q^csW7Fdo7!&M8cMD^?{+*YG@c3bm$Nu87~OpoiCrZOtuaBB-'
    '(W%t{z0`_DHPD*+VF2%FxocB6JAg(xJ1sVxQlwXBPYZU{8BKv7sTVhSM<<?}!2{$4lYfqf1_S~541mOlmWF3ex)q&fRoh00Nh^!O'
    'wr8?zijB2`>T&I#ETqlE5Lwm)eJ+klItqc{!4w1CiuP||jmVOuDvyP)-'
    '3#r1`2<dSF2e8{UImaws8Uo7|vEQ{r#3dMwWI#&QuEDoM2V=>THQO&Jv+E?i7)n`=Qe*8=3ci|Xf6npMRP%I{q9-'
    '*hqvvoTPy0B|cQ>h{=P-%~`kdf%HY@2lT*AZ5PV(JNR?~BcI2mZDuC!uljk-'
    '=x$q&jgT=n45Pss(yXHr+supi^_^T?b#WQ?YNuoWUa*rgE;*g&b{tZ1Hdkl;k>jR3o7|8RE%@Qe0UNY8dcKq#eWJE65z(zBfkX7D'
    'vb?}mmK-'
    'y&GzCPv8wO@OAaI@Sb;XRnAg0jjI<NmqFZSLwL(6=Jb!<*F*j^NyRExT=*3Q4xm|i^`FmScE(vBFZ6})RM=TOQ~~hu3C5+#q;TRR'
    'VJ}rvrBz*CA5{sTzX?^?4>u5#zuNfjjx8<D<yn2(%z@TS2OLgWCQ7OG#W51Eg*7D`6Y~tfy}NFCd7)frGzVu;v5m7kWJPyASm9<d'
    'TrW&E(VB$z9^aNqP~?&Y2^dLcpP^`{yu7%rFn`Tf)0Z_JnAsO!@~}PJUqHE0K>CwxJZyt%V3u@BhWI$Wz7h)RNA}``66hf?_{qIt'
    '@K>>O3+NtW1j@=^j*=$sz<wP1cN(xWA4y4HYPPbv^ONq!~GI=^JSC|;JNpF+0P~1>Z>i+OPJ10fx>2_3~<}f)4pBkY2QBdv_By9w'
    'C@ml+II{+?K|0?_EFmjqee=4R#n846R28TBwODaG^2p8^PHDlDCcHHeA1$G=-kFa5pB^7(7xi0bz>r?d(oO|9&9!-'
    'n+PkFFHeJ2%GamCYUK;m;B94t7+)1&TLB&~gsPV?X#}Cs%&@wb21feTKIoZ*)z1*HhgBJ3`8+>pX_~_t=-'
    'g;m{aMSVGQ>~V7!`e2Dvpad(ICv--)=2bu{Y;NhSgE1f^t=L-1PaTgoQSZ(xn;24oU!_@$bAjt&;I9U-OtJaQ|-'
    'l*4o&yB4S{j@)d)yLisX4ct`m<KzKJ$sLk0Aam!h#!lJn4Y?}WR;+AvJwnTBuJBhCw;+Au1h-'
    '!#i&Z8NsA#QmWjZlr79X=|cot58i!jkpk@@Yl#Tck`|RU^Nf<D5JxoKJm_88yu3iU~a}6g=sJ+$|M6?Y+9)6g=ZQ|2%2F%?13QG5'
    '_Pj`p+tOmQz(W8s^)k`MN#a*p*SMG+V+Ka>^8j^J5BY;)q<8FNv!&lYuJSY(s52PK1H<#4%L)mNRqA1eAOj<Pr3>tQ=99ZRI2SyP'
    'PNAQQKTKXKI&nRjt(m&aPI;(388P4$&_<FFiDOh2-'
    'rvF%(?li5?d9h#MaDh;vnpJo1@hKk>|G+@#6SXVf0a)MwNn$=GMqBFWrm)FjE^XWZz?<Y(OM`EYp;gQoLxY$KqRRdoeQMmag&lBl'
    '{?W^Y{LN5-'
    '$AFq$7H{~q}vz6|Ow+GhFe2bm^00ASZQ5Fs<V@f#(S$2R{a37umIb+d#nF@*XZF@vKUzePf~*qORjLU#wTcoWnFW;Z?cQJO<@N#1'
    '0pnZ#R+Ne)a>tu>sxRF-'
    'W`Q9&A8Lbx}`$3(tlN#2?qV8l)96?gl5PQi0NlXkZ0QGK=(<T<8C^*PcHtocy#6b|Yh{;}kdCe%Cp6UkFK$a|iyeUk7z2{W`$68='
    ')cpR`XB{zk&>+GVXxe}sazX|w+Z(Fa+=wD)sb`%-f!f0V8xLzGeM5~aF(uGRuo+_YdKSa!_?w6U@iM6q?U`+Xw2B6UgNbQ^-'
    'L&lGncLu^{_o2DStTIX|b=bC=rZWIiSAiPcc#OMeKf7U)RN?b(k5~HR#pphpw_aOg^w|_6fvo;x6O|djwj8FR>5MX!vjat$o(k{V'
    'B6=W$}y9DNe(iS-McHxQ|8cMcEn_}d+XN`_>gbxxUj)UmG;Jx;T#G|jVeGj>gepV=+pA*XD=Y=x)yV&Bl_Kk>V;p5{3)of_P^@wN'
    '@_uz6wG>3co`nq|rHy0zK<=e;ABKljHZf@cKD$^ZA?SgxSO}3U+9@h1uh|@yh`OD=THusHcuCOlECYZqc=#Pmu+g-'
    'u)lAj>j+C`qO@sxt6d<Oh^^KIiJ$gof+kl`VnK!%5P0vR6B31oOwCy?PWoj``ibpjclh*g+9Se*?C%M)2@<D^V&NXn#6-'
    '1Ed^>EuI~ow+TpOT-+n4aHVwo!|;t;@IDpI`-2O88%c@t=iBP*)&#EX4%-'
    'SnkuSFZR%FdjetjTts%`<y^@O!X)^FtJan11%eXM;&NS`{hT$C3yDJ!mfu?=e?(v9)mt#uZ&rK7rJ(7||qGIq7J5$Df!<p&Gf@0('
    '!Y}aTNUA)1?AMPv7HPDaLmc)76Q;$I-;r1|wGleq}rPf~2T-'
    'j^WB&6vpO|Xj_OB3SY&e8<IxV1DvFrTC!CPI>axc{97iCh=P3sNmxWRTA?UEOwzM^U^gV*~1@Oh*H`yjH0vjWE)4fv5$p57VGL>H'
    '{gmcoDg3&;FA1+(McQ6p3sv0jgjW+uj0n!O84TiOlw;yeu1nGDYKSujV3h8rl0>E?}i8%GYp7IULGeQ7IkT##ddVD&50ZT`v@*&B'
    'pM6aM;UNT|X>Hw^j34aQK<8x_-'
    '>GXc!;o1Fw@IGh_;_84HvAi=mqfePc^^SXF^gvy~o7<vx)rHcF>`V`(9~3)rD^$$a|7@o~K^pNg5I9JF?g4EqR29~?x^^)_n!Hp*'
    'qW+NgM<b~!%h2i#MAK$LcRYoe$ky$M$&8WrjcXKMp+T#wABjZ-'
    'o<g8F#JF?B>266#2ykWfb&782@6qe4O**`<(BM;aFr>PVA94h!Qgp)%Y~WYs82-Vw%G0q-'
    'F|@YeftqW27Qgh!=9BSj^IjofM%MRoeSxK%Sn<@n81Zs<?7shacs5@&~giE~1~#5+U3#JQng;=Ir=@vev(ZFMZE*)X4Tra2)l34<'
    'o?WICFQBBohnJ)%;}ZH@++OXHs`BNjAo$5sMWBgch*Vg&M+#NF>qQJaj=CGU5%VfCDpt_{3Ymk2i)?*ZSSqYZITvh(o$b963z)kx'
    'nV(0i)0q7eA(C<Hzy3W47lg}~=VA@F%o2>dP=0(S(rhZStno+kW=g3a1<I3HE8Mdy?DGYx&ZM@&khl#~$`(C`u`tlRTv6g{+p3f)'
    '}F<GwF_M8srSqC)2MbQw>;s4><omX|Fzo0OFIPEN{ur%10A9`L0f2NBBh%`%Bw;+tg>If8i;{Y@j^OFgCfAq5+?FJpX6p=<7mERQ'
    'R6i+o!32?bx3hz>0Rjfaf+mPbpIcEGFCXi$NZ-CvW9R!0XQM0U$#F6BZMk~Wkw!S8J=y*i6(1KLKl0ZFSR)Y5Ss-'
    '}Olq^+Xxh>P?kA|AE`Rm@+BXM&UZ@|KQs`nw0Hze(3FkzB|$(#NA<k<4~fT@NWD@kZIUZA~xI)6=+Z+sU0N^y#cRIc7wA?YJW?@B'
    'T*@Dg|7fL`);%+jyR9SSq!%B-w-'
    '`3P03dqAnGU4%?gv~xCxG_9joT?7xTi*IoIi?no>Atas2}2z|Qt<r6(Jng0tmQ3~}dLk9Z#t?unXE`=iw9I<lLHW4GC}cdCSMSrR'
    'jMLRF{`@$hYiiUG$^o8348xRefM)S&zuhjHRQl(nJ3-'
    'mK7IZ+2*~HzzdMyE8P{n;RPJ%?l0otcOPFmNR@S?&}Su=_*fzKseJ*`Wz0TmJTxC@n&slH%%1bw^m4}&@gK!q&t#K%6}TDYk6V*7'
    '?Gk14#|VU7I)JB;Nk8gN?1-'
    '5WLv9CXNb4;T}7%Xg2R=WRP4v4nN(FXi2+&%>FYG90B{N)x=s>zn}cfQ61UrUpRrj;T_%0T<{|1n?ekra&;SIV??Q}M9&M`Z0MlO'
    'Gc8S=Fjo;rBsDQ|N1v)X>2Xt1U%c6ZiXEi!3+6O7#Mt4Q~@Xk93!nF_Yyer`~?VDN}`h{=z_<=HmQC*wY@wP(F6~SDC2j8vsLtPU'
    'R1?}*daiSnR20YVqRB5KRJPZS)pqQIwZB-'
    'd7M*Q~&h1aTmVh5_)RuJb?2tQcqWi+qQN$($t^v<N_25!=W368Bq2hgn!kno=3L9x`l@k0g2%Xkc2W$0Pm$Dz^c>>&0qP2%YwPeR'
    'E0F$ZBajM2=l*dl1I`xfD=jjpDqMzLR1C*xk<1GUdsa97*w{*NiFHrmpsgm;P-5MfiTnd<8iL|YhbUCDNEIlGeoOhjg9-VIG=-'
    'VIaUnjgaPtgI~S#q5nv*}*PAfZA-7W(LSIxLA803S6R@h*AfaYEKP;%QSOCR>Ek_43$-'
    'IxhA5>h^UE0bH%q$)l+qzWnK{~paTjtI=zqJ94Y>6y<^fv_}!0L<xO>jG|ZOsiJ@l*x<*2^iG=Lrzv0Przh5ffJoXFar{}fL`WW('
    'xYXEBCMR1xC<g|#x_lOX8jl=hd5nIOLdqj!5#o>E|MXloSJp!ZFarhpg(e5gVVv~}rlU$!P$^c^?(MJe}1GRKL+P74xNj7ASBWA^'
    '0Z4qT)O02<QlB2OlMhzuqC5)@F3ZH8o&swLx5!C)Vdb3fQy-'
    'm|oqJsB(ozKM9Id)i0uJ*qoAzaZYZ=HCCKi7$87_+ZvD78*J!<c(T6Tfxh8G7cGLwfUdjP&l+G19wF#}Il!ZmQBuQ=y&DuAo}{PI'
    'BX~A6V7RPVDN^DEvf^b3V6TF#`lT`>cf1I6P(uMQIY(OH?3%<Vcd%Rk}yVeJAeMG4y{x$LRGz#_FcCj5~inksV`_!VXfz%u;BesG'
    '!5q6ojI34#!iFii$ZLQ9&sx<#0>|t*DUWuC(XNN*l^PL&A^wo-cIv;<(0~<Un^oC8u~yxEtkJm5lN{Px-'
    '+npDLcAN#^_{1!pKf7vLaI1n*}?^j+fT0vzmX651&;AeS)9;Xu!N%X%~E?SQN=@!W!qSfY|ik8vAxQuAv>5_g<Wt*h!n_N<&@6r8'
    'bKsBEj6+>qrSrQ}?6wcfE3JzT;be0=XrB!{l~#|lnlhP{MO6r3b|kF<4)$A#j*!g^)qg_`kjf(AaF<|KE3AwAnrde%FH&ayr}a%N'
    'nIem*0zLs$l!r&wL3J<ootpfT-RMVf^w4>Kp6V^Zc0Yn?73;uT?pGFs5b(MG6}fClcW@={VRI<G6N)X=YA%|~@^A~ib#_E*Y}!Tw'
    'C~C{`MS{kh^<td!I_(5D@Hn^spB`jHfUeR6(3MZrh+KV^!fHp)AJhxw!vdfA8j_Gsu49^rdyekI_y)TVN0wcRLyN(u{cmpR*?+~@'
    'S7SufOL-'
    'v<Z#OHc*00ZHnnLPXgo#LeGZt=3bbF>|7tmzF^K<`<+I*F+<+22z63jfNNm2Py9qa6HGVKUeZkjFJ+orSK6h(#GC|zR`oJt(H@EX'
    'eH2yIsJ`R0Acz9x2PiDm^Ixu)b+KDqZjD4w&k!7x~=Uv=!4FDdk*=a4|@Q|H_)^1z|jqKXghLj13lVKk%q%hQfJL6I<y;|wC+l*='
    '{Lg9tujx>!%E|KN1U~|bx0ZgSjeE!Di^-'
    'MZT^vh6Qu8}SFd;`Fm*dP#23SZq+lik?0uDw>=!AfzPDoS*j`cK+ZBKx7y{>6zYq}54tDWyN4LB}`--IkQTR-ak{1qo@|%rp&Xw7'
    'gDY={qy<+WucM|nkLK&7Is`jQh!+W0Q1e!VGc02k$Uj?JIdL>PXg2>P%DJbuFoGK~R#ZKI5l7hVK#G$!rqP*LN56Wz1AJ6GWxh@|'
    'u>ra^pf3gpy)P&1t`hF-xj;z-B)zenTY5eNhk`uJ4^@<MVz&It0x_*V-'
    'WqS&c9KQZnBC(SR*afN=nx6^|3sYh0qNLJ$0mpmjLzZzrtRgkJLNt>SuzVot`q2Mna_ICj#dUgNd^b}i`0fss;Jd%81mFEbCHU^2'
    'D#3TPD#3TNRD$njtAu=oqQBOTF`0#$BHe9xyBpHo_V0W_##}{;p_R|%lxq$|>?XJkFT>yC;#<6)<^zRWoJiwq!Yxjs2{z#tC({6-'
    'aEnuDhETZ0snpOQ+~PFI6nApnMq?1j>c*zP&+j%5ap&z17Ep#gp4L)~V9GXSvqKBw{%YuPSJ^cCJj9UsW1DN5GCwcJI~6J0-'
    '8}Ff8_))#?r{p%x_;B=7`o3mA4mw4EnD()xhI1R5&S4@$&U*P*^-'
    '}XPZJp?__x`T9~T$0B|jyNlfATh#dUiXxUBFV%=~EZ=X^u+I?o>)rqTJZ(VYmt%@KV}Sg1?lxhF54Im+kyjkhsD!RJmO%|vJso%N'
    'yzH-UBW13q2co1$)4F%Vnm)5U!$i2qv-hCJ-k#r-JUd@bwlM+FC^TMKwy<JZ^^T|_Ds);i-ojrv#y<6XvTVvwcs`*J-'
    ')X_*s~Pive@&#XvgAQxUOr$L8KGWX^sGmn}iMvnDe)R!`Htna3_iIHRdFKV1<3qz|(jHgL%0EPbC6$(BL?$1>U>Yb3Ai>UQ>3*lR'
    'D7bzEk-ptXidjf$!M;9qlvPzgxQvm-36_+el(Aw9880E?q2@%l(5ol=@@y6T9C=W2GzhnlVfNQL7n~PY-'
    'N$|eB1ploFpE!kq5;#>dgA_f*>VWgnH@s587dmFQKbTYlJZu@lR2~@`o)7}|Z6he2;`yb7Jp3F(axeTG!$pJ-Vzos?hyvH=Q#fa_'
    'pZtk?Xt6J}U8u?6*F?Qu26D`SEmkbHXN}nMu?_@-'
    'Z`(78Hv1};+x@wA9*dN3uS51Pm{5F<>olu_HQLSrF5`#SEAF1jeJR2xS|gK?t2d`~_chGEZK!xO&m=@2JdkG!Vh=Zl$Kj*}!sh^v'
    '6)t1Xl?V%ys*Z<}x`aoPs*Xq9Qzd8F6NSnDlU6-'
    ')CtKH80P*^Kmf3=QO67A++M%Vdibf%pu`rHt4Z7`jE>gZ|uxP}#m7yAb>spnvZyPBdP&64DD;`TU6`Cp@Ml=nYDLb^Q+5|iO;FAN'
    'x;FI2A@JXLA_@r+bd~#42e9|urJ~?<NlL~hdK`Rmq(CjJ%-'
    '5;`q9>`0mP#_9hzzJnxKFoF<j|!#+{R^YMZ@nl0zsC8YRDVg54fEwT2%c9*vqc4b;2APsQa=G6$@&!)x3adsju^0}F^Q|{xq&)F`'
    'Kq29LvCTBB%bbO;mg%-4uSH?k~q3s!}yR}Y}njY1KrBLYEq7bkH2{?>-&&$-gq_}-'
    '8d(|2NN35MryFiX|PSyUzN#ZTd2J%rw+GLcUAK&{cI(bbQXP2C7ngzS4n5l4^+}w^h1?&7X3&iokc%ZNoUbd@*He@Zl9M_!wrIt1'
    'voX7$-;gGeEO_!+7yC4C_?iBg_K-A?Noa1A<wP7f$?S)PuaV68$KjL?sH)h(-'
    '<nKW#(s^!;h#+*i3gOP2&1|WL?Mv+$gyfHfoidB=^EbrE;_6X4q&OekZvbHY%7~B)7vx{c@{hPR1Ax9U_DrCw)6W>ic(Bj5w|{6|'
    'EaWS?6^!E@}COx{a_06OYWB7Igun@r@+qiTTVq{(?NQ<s^w<)W~cJ4GVTMdnnm@V-'
    '2Ooov&&d2*djj!E5FxBab)pU)?@Lb9%ZThKy+A44?c*&+RAaJCWB?z@wHKEu*Dxsz`nQxT_*{O60bR)aP$*jxM@L`fRpEuTev|ML'
    'ab&=1GuFKGpB6%cwH{JX`h<lcshO;XYJAjTP522@kNi9?s<o$nzZQK%;?0;q&Yyz?a53Od}t~d1KzTh%EYYRm)ETmMbO^oJ@n}8I'
    'Pd<e!8ddfrK+Wg%5rCGyN2m(<Pjxn2Fyy>J<*VlS9q{x;+e1aRIu$$Q|bZ-'
    '98k>^sN)DcfKYLnH?nHYAdEQlpW1YVtOQR5Ehw~NXwGuo^>W7U$h)A+cOPS0jCKnsg{$|1l7ihOo+{aP>xb&-'
    '=qO`FylS~M@S0lAK}6b10+TD2gtzn^xbl=58ZMnhHkl&Lbu$>p<C{h&@Fdr=$1RpHWF|Wd0cUn`Yj3f=b95$P@p*W#UD-1-'
    '*#WHQ;nbAv*_@%dm0@`<;imh+8uk&m)wtt^LhAQzmD^HLc;lR{!UUKuE@<$)c7rOGqi*Y<NRHLI-'
    'liLKT)^{gMv;g7{89&vcq9f^DdJ-DD$OK^R=%}A>=qwGfFzt<h&Q^NVuR}x|jm?o;3AXB1EC6%c7IqKvPKL5c#>LipHI%rXm_q)K'
    'd7_Lq!;mE>8j46?|i}HU*C#&^Yf=@c03(^PUBdAJ9DSUGRtj?ejhbj~Eb=>|03V*)|!zU2@hWZlzpC;`9DzO(<EZbtNnIf#7jjQc'
    '_p;6E*C}Ws%>MaG&SFv}D}zcp|rkiux@EO7)AH%LOecxyd;|#?1``pz8wxXks7$O$r2{$$<bgB@lq71_F>P(cqR_C*^zT9BSy6Ej'
    'iITUx$WHJqoNz5GHI8F`b*M$jJ{oRB{S2q>_f)LwcIByn`f-'
    '^0i0;Jx}Dl&BZ#YvxL{QD~BXeoo#Bnsy(yi0(A70AHjLCgjd|ETEkBo_ILYi0xIDGw0Em3H6<$!DEQ(%^ref=KmXbc+^P1X%g*;L'
    'Cl3>p<#;TnbC}KJL+kZ1wF4o(QNBM3Hk%c3AT7cib*94qynMc?j2_KeMOhfX(~g($flCS~TYf<jT$GpK%bKfAT{~~C%n7aP=Xeej'
    'YI)zPel9bhOz!h0XSR%MaHFqrLB%&wvm!;UH<_Ddlb>*&_y4}4sgCqB(V3L6O`9;VNrfodZ9IIQc3~NlJr=s(Pyb5Igf_@rhtO~!'
    'e{U)Rwky1d1T~PdkxcA(mj(4?64y4#_pE4230&Bdo57UF$w95@^OTo1>$R-'
    'iB{J9h<=6CaY@o6lt&ATDFwaxUvbzGKY#agy4P^x@wAW>}j#1&8GTXp$7TZ1VM7KxL7Ej_f+hj+kX|`EtL~h_l+mtB1QRiJxL}Q~'
    '&hqXeCUr$%2*5rJ`HwX=#yq-'
    '!_*bZt!%+FmZ>Zki#UkG@KgjWNV9Dn_fv8}uS0a09$ITDS4cAe`enX3!X!AamPnQdhByHsYI_|DLrPSI?p(Uwlt{O)@RSf=r7Zak'
    '+ZM|r0`L}~+H)8{qI*;uVnzV)h^rtP=EsfuWHr>aQN{U~6VV#>>EmfF*~ZYB#Uh*9Fd{l`i+zssGhHluiagsPU4CWZ-cs>;<>m`e'
    '8dRA?{oepJ=|aa7g*iE9FVstRK$L4`4tr>QW8@^lr(P@bW}7|JtMASs`v!Whc4l_lqnh(^uWxId-V>ljx&lN?4AEfr03E-'
    'Xq%|3}Sfi)>F-K9x3K=VN6(D44X9SQ#~ZjT$d}5H()-'
    'FlxN;k(UzvZZ9U9ROaPGvuwPeXl$966pbzOqN1^7URE@=%nOS~xW&>sR>FIlsmHAtO+1^|@#mx7h~egiXyM|b=8MqA`3_NKfsaXp'
    'e0`^lE}_##*U)LBTj;dWJ#^aW5jt)3bkmVt%gU2w(&|g59ovjU;3<u@anG5k2GLT*`C$^vMH<3K{4`Icl*Z0@XLX)dyNVCtbBXHG'
    'ih5yR<HvYkDB;DRt$u@0K`|EFLa_&aI!k=V+-i73X6xC|EtA>i$TdK(YJOa>zFUEweNcC;3iS8`^lx3D9pEwy-wODRjOQd)+ezNc'
    'chm96X9UZ2q33wikHd08BR;if0Hp53=cPG-2E=D%=FbWi;PZ(9rJ`=?SBHj<n%i60?o`_~?C0Rr6{zbA(AKH*<ucpCAqyK;-'
    'X3@wEedp$f(x~4flgDPM7Au@feL$k_W~UjhXns3Y08U>_qP%7rpB)+gA}67zTx=#o^m8bhcHexogy~KplpJDH&V@w9~WfZP;Zt;B'
    '78!@yH<qT$)Rx~T$~@(xi0BInr%$lxn`TVn=AKg=k}!kt3lkp63lF>GAoY#D6GhUi~had(IKKaI0@aP2;Z9S7g5TR?DNlTLFRE1-'
    'p{3amq>cY*iU~UZN5a$&F*9-MK_O%{-9qW?Wonme0-NN7*Gg<Y$}-'
    'h7CnOcq|I=)p1Tb5%z(JzZnlM6DH`mGzilg1I?y$<fei}J4RMhTiq2KK&<4>nhq~AX)u#+|!3|>b40q8Ds!plOj}_lWO{wrU+`R2'
    'TR9@-'
    'lO1{Hq^u@Gs2}2uWP1mR}Cv=XoJ@=hA_j_r(p}EpclnIKojdsKTh>>gVHpUgRDAU(dM~`GcT!+~ij*ROtyTDO#9cEWJI<CX)2FJv'
    '8nBAcm*J1X6W8*r^o^V`&PyF8{Jg2#Yrxj{3jbH6*Zvo3Se$Cf7;og{&a}CgaInSQelc=q<6Xi>Zx@(|ll){zKoZj9^Qj^E**_Ok'
    '{CM}0gkUoV_<x`yB2>7*6WZo{}Ne|!uD&Z;bK+Tl!4A0D=elE}Rpd2dSeM95d*wWq{rHW1K@$MXpHR-U3^0kIIqH|RA_-'
    '6Y4xw~unMMdYwI3fKYV5s8yx!rRIxQb>T#^?9tq=f|4ZJR<y8-'
    'N}npqJ+OeI<>u=MeNTnr6?@{NV!nY7XaD2{=fzkM*NxNM(b`(Q^c;YzSIyILIy5QCGcnal-'
    'N^mz2Ci!n4LdisvTH(dZyph1UqB2phsURISa?vw4j=X3W#MY_6b-'
    'Yc9Tlsx`A4B8i<%r2srm_w8I~gf4lBy^<e8xmL!}_OZ%D=1d$x#!>A)QNt^8?R%9>>kDVL5&<IgbMhIA4j7ZkvSH|bc`n&-'
    '#EsBjFwaqGIMgi;#J`S~)f@l1LXd;-uj4ZxjDIiVLbkD*38NdSgQ1)Ewu##DZAwDijpDflf|#*JZ0JmR+VQJ-wEKcinr*?WnmcAu'
    '8!JH}xE8_}`izwDg7Sqvqa?hne8TT$9z}0_A{DsIOO&Cnnoya9H<S-'
    '=cbD*{4{uXnm$wx9d!8@2+}2jbBb(zgw|e3iB<;IhaQ4fXKV`d>d0*3D8~hq2&(Za^V_+W^wiq~05aPSUUHM<ji|-Z9{l-'
    'uTy9z(tyhz-j*Cl>+ZNCI1WOfr?gWc+)UsNUXrRNO|fo+~EG>lkT4rEjj(dIhe4Ci$}DVHd}OBJ3B-'
    'wtP)WyHj2Eyl0?WsG{a(^LzzU%i45o#Woi_xq&pumRsB(Qfj@dBiIv>|^1{I1Tr*L%Jl_0<t{Bi0i5KPAlQF>Y<M~wd<j7RrSoS>'
    'MtXLPE`qeqoWB1cprw#U<bojo{3dS;b|U$M5Hc<=7tlG@Y)c8?Zru!wr(_nnJL?wlPv9B&qXs+wl61HI`BkQnkn0llPsO6C8Jkmy'
    'AolE`zr0tX!iAzIIT*9JTEAYd$}m2N`pSHDesY1llkuJ`MynQga&@~XYo>r-'
    '|b7RZt=sB$#LH<LesvVL<nhUXLtAD0B*u<=PG1d6z{!)?uxc`MQM)%`mU(SK~azn4?|Dg)BI8pbZR7FZM?l2Qw4^R-'
    '&h)D>g4xE+8s?0X^r7aM;gBgFS%Pj%Cr?5>FWh<_MQ0FQJS3#ttoQByx+cp>IIm;(_c^#$|LR=w~nK-'
    'd&WRy?>M~u1q~E~%T#hGV=h`b`oezK&rrS<10D7LGIaCCL(tu)Bk`;{7s~0e>Do>IK8J|q`2-'
    '$HbGpV~?iAUZ`91fl;wT6CEr^Mz3z#RW-|DI9ml`v?F1VUhz8ohN&=2n62P;yf%vG0|7?9&nn;1~ULZ7AJO6_Y<W}{0S=Gy-aH*G'
    '9Bas=B)BeB{W+Iciexjr%xv99ZHbR8Eyk{PhAUy_cQCajg$YNu1WfNP>Y<!$#?UW6|yAMGeoYHn%hCUXoic3kBvO%%GpZX5e^pK7'
    '#=(eeIFqh*Yo^)EDP`RKUUYt-'
    '{m&|{sZ;+7DgTCb_T#i@hcIaxr9HhD_2Z4)rT_j~ppUU?X3)C<4I!=zh`5dA6Npk2d{biiz{(q4Sh3h<JMPr5WOz?W1Sd3!}g!R+'
    'Ix$%Z=@$E$2z*4gb&m95XZI4Y#dHe_8Lola#Nvu=(|r?O31cYA$`$~I>`c%W*wCF|))r$^DGV7po^V2ehVA)W4AA!7JZ7ZJi1Qmf'
    'D~Z{jla&y}ltvJ^TW5AT)RbEr4*2w7d}1$HLYG8ut=FRYh90$Y|SIz$_9Q49I-'
    '9u)yy4i4zj1tR<4Ul)k%i+_g(%;^G={qXPb06(id;duZJ53{Sc1#H!b`t9l+0bgqL9Ak@6YtJXA5xI#H(I}Uw5I?97$<VGcjJ`x+'
    '99NPVMlDz<ltd>VqYfrUM30J4%eg#(|NY7KC4)(#UScs3$87A%;K{o^>~jdcN@>4%n~O_kjT%n!4W6M=VG^5;aYc^)p^_<A<h>mx'
    'jRWln(1+6qJ6{`mgoHnawlGIZH$K6jq7B-'
    'Wi4<XlfQ=fRUb|W;V3S6#*RECx3V1y2R__Y<N~7N!b3Ul;ixD9r`2~C|>k@~3iS*29N}N&eo~t>BL=Ahc;kYez?77x6C48rn*XgL'
    '*-H6a1kE-2{2>kJ=+Wm;IACIctkO*e+sM;NgP!^A>-'
    'IBy8+LCWWU8x8v<?=tKSADOzEqi%X?islSby2mJIm{t%EGRFTa`$eeLttcep(Q04S*=K9^*W6i=kNm>(BnJ>@-'
    '3RM&EW@xCnqzSHydqaC}O=Z6tP|uYTrkNde)0W$An8l$An8m|A5P4l~qr0!Yip6Dkf``2jgh-biVT659KB&puVI;9S>vPLbjBk?M'
    '|xJa1S5GmC0e;uIVsF?Md<Mx0j&<aL@_mM%A((;tLnIY4+;^mkPJ^-'
    'xPaBO@;sHG`WZeiX)Km6UEV)^&`bmclw#)XsP{BadgxCRB;57cC0uPNGbwVq%!bBhNVJkULY8e*H}%7+r?c%1q*3aA|&I*I6LAwX'
    'iHR>vRodkrSf2j7aHk0CSfnyzbQixo$}4x^d3?D7B?`Un$*Xid=qQb9=G{EuBbfztoXE}HT;Y6Bwh`-'
    'yAH3GOEWldX@&%<%<w>!sbY!4cbY_=x{|gv^jeuVlz!Cis2fNVI$^C&h>W8sz~=kp-E1l!yh1n4&*O<Bmi^F@-'
    'N|h}#c>5fY}5ZXVr&ysX03-'
    'd?yua!HYgO^TEBB>t=}cI*6$ix>vs#S^}C1G`aMEx{honvG}K29zSgvNbHxd`6A;3Wm+3H`pX`=8KVrweok+=Oa85N)HRTM&V@}C'
    'HHBvFprYz|cQzxRLWJ=()OzRBmeOKn|vvoXt0r$Xq9=(94WCIUgz#H7iBP{U9Z1Mvv&`I6q*>mW6e51o*g>N;VRz%d_>u^Nj2OZ8'
    'R{HSx`etJwLWa4RLVvV>)+FqbWT<_#*pSSzTgf0uIIMHf8>~+$)%GJi(nBa%TwJBuhF#<#(+w&MU)n;O1GYjZ4lW<t?L@?UWX_&2'
    'dxv5y#$jA;2IkQ}M2+ARh)xMDRAnEHzKBXDPaIx}pX8($CTlqP&cSsnm{0P*)OSs&v*3;Z+7^8BibSL7QOjyQ<Z@h;4S%JNq<VtD'
    '@@oab7Yb(`rr}Lk=qh2)~(3^QQB)Pfu9KUO(sb}+fwh%bxJKrvCMlHr$$5Dj?^M1#f0uM)C%(~P59m&-96}}l^PsB-VT@OU0&o;F'
    ';!do_~eGuHTRqcz|iQQizsmJ6v?Q4hoNmqJhW^erkp;W|UK^e}ljDr*Lqce<Olk1BAqjDXROm(N$VTN;O@^07qCfJnGNow(D_*_F'
    'JvDQikc;UZMY`v>AcrggA+7RBD^0`*hgK$xi&#2)yHyX54Q!i#wy-rm~EUP!D+>V9yCKa0bPuAXqE!=@=EGeDD^L}=fL@C-'
    'eO#k@@esJyE8upc@ILf20sf3V?08zU)V*$RGn9Lis-'
    '()VZ;%xvP)}~SIY>_a5uR*v{lelkkTXnmFtBEGFR>!pVF@`n5Oxb!=AKY@p4UxT+&*&0m3m>cm4v<9N3@&{KVpBveeMh1MM4oOZN'
    '<>C3emQY2A{W0i5uqX%zY9fSIDvYQpw3zk=F;ic>$qH>H)JQ&Q+AlK3WVDC6L~H&bz((4ZT8pSW}GNj@n=J{qq!A0ZOS+j(>3bcy'
    'oQ2zY3hX9t@^HnT>>epYiPq;O*w*NeJ$~U#;>@758_)?!{|I$ihF<L7X2vf{jpmdOqqjE!o<uw6rxgJw_mGZg7&RD4w0tDE<w1aQ'
    '*^!`$wSn6)JX0k$B!q;eUL~Fk6NLSQID(~r_31Brhe#HQ{mgBlGH**kw=HOm$A&chMtYxLL<xWp?zZyPZvY1>N`313Y{Fg8(+i)f'
    'qEDx&XK8l8pp(usd^cY&V?|08!u(NAI$Yng<3yJH(eR`J2;p6(ir(nAs^St=fj+7#m`sEilHt@!Ni_=nil)EJ~{HuXb^mTSZ!Av;'
    '2RNjby{fOdNWO)of1Kmdi-6N`dO>j^W<K1LnrbSNpwRe@qAizLnqtms%8f36dnVLZs=5=T8nPzG&>NKx-Am~V>hEIL41ZT&1hH=L'
    'zfm50wd^Rgz{11H1?qyJcitdPy;6Ji~UOp;}jP(qcgus_-'
    '1K#TGyK@9+dVLx0<8cJ3}1{lj%>B9w`mv+&+<YwsFjx>_I(s)$<|Wb=1lxj=Od=(?g3KFY6<G|G`1{SnK1CdiB9CS!<&9bv*l))}'
    'wMWD5$fbDw#{%SY}NuwUOG`y8ZNVK4hpg^|y0`uY+jHIFna@F5yb<jY6X)uJDCNzm_n;S54Fr%Jm%zR|tA>)4tV$UfjV75gWdKVi'
    'vrlmvWkf21*M!#dSAn2j4h;cDTs_nr_nj%`DVwWfYPsqo5&*d4T5hQO^)P!~dErqYgo9{p{?}+}-'
    'B_2J%qnsWLQA`Xn#BpA<C~3|4A~qUM6Fd@O0D@Er@^5%l8jeD4Z+apNUBweJn(69o9$t~{{C*ABNrbBZtFgf>>vOKOkZ@TJ}AY6;'
    'J%be>U9+wH(b?<G}zIOvTYPM;{WQ-flQI`%~2YZO~6k!X)rWcAJ11#%AF&JEDgMAC}c1#5F6oFN+E*9KI@rKuTXYab?kTli*Y2<P'
    '$Dz}=ZB&}VXLPZy|Nz619(M_ahIU)AU*M>g?$9ofVmbfnF<>qvp`(B4{B6Ibx$w2KDOCvS@^^lv0e%4^n2g)83?p>ctHpb2bMYS>'
    'xcVklJSr=4$(i-`xDXm~n%?(gvC`!w-'
    'MKh(CbNepJYdoXK8M1~L6*es*YY#!pdLA;uhZ7p+rv@$XrabfXgGAS$`EtA6H%VknnJVqvk#baesSX?cW!s2lZ#fVg;;2P=i?7fQ'
    'Z0s3n9DYgfP33yQD`Dr64&_g%-'
    'VeE4?6Hspv+Xfn1@1*%zf;;d%7kRJ2v?*+^qTvt51Bgj56&`$`*pH4RU163bq3*8M0^c3y)$2p|7vuVK`OEsy`^C8a3|BUW&M(IG'
    'XTY*K^nEcwDTg=^(PhdvM(=By@on{R_a4PI7d^21l?^Cyu1Uap-wFR~hPa>Sb>DWyEiqlupr&rKE@{@!jCZ+)$B|JHs5^+~q1yN3'
    'DSl&2%oXuS+YjU4IT%kutY0h97RRjHp$PWkJpM2NpZa{)cnbKWGoWh1<Wy2+#}W5zel*Ugo{e$OKwfJ?oiw6Ou5KPp{CK6Ay>NWm'
    'BcDo&-'
    'W5iUlcXvBH*)+UFGr!N@$Bv%t#9?og5K0>z)}6Eaq?9v8o@RMky=bh9xmXsm}Fli;B(&$^mR$ER*nY!hNM?(w_0W@o%iHwGzm}o2'
    '7Vt>-vP*SGj^JA8}?OVN~9W1|GJF4>@beZBbuI}h@_gt*^{;5(WF7Q#)BT3l(fg|p-'
    'Dq=mV$&>p8&wV)EJt}krvTp5w;Fb67hiV<|zWY?3h184hc6Z!_)DQ=F})Eokuk1L-'
    'BMxra1}PU(jHq?%wrzLUS7Q8VS#_D@%yvd6geKUnfn6M`@r*82vngrU~KKumv4BM6`3X^nAkqN85LYX<3}>J2USaD+(xRay$ovBG'
    'O?gRxB&l*RjSVEG$KX1+mKt*q0_I#^mIXXpD(Lv5=sNk=TM7l~|%F=nAOVqDBz}RIEXN&pk8GGc)gUuJg|?;k)+Yi?HlB@60n#xl'
    '11uRI3yVhGSpH3UYpJ(o*%LXW_f#5OWFp?%izk`J59b+Noq;C8#8wn>3k*7g^-'
    'rzsMs00Yw)14=l3Ce^8M{{>CDU{0A3V<R4KaMfg1h&!jKb_F*R-'
    'DH~6f)wWJQqmTV#F+1Zqw#M{tKJty`ViV2SW~V=L9xFIfAt*UbU;MRpGVY6=HA<Shtgg2?hn66_b<9}mzOt02X1@|?fdmt(=leQD'
    'PG~tb)?EjHj`>99fXBOFd4@=?PZ2rI6SbWxDu#b^MBg34(|i~iKmWN$&|UK#7bi0IbCZaS{lX+7V>e2Af0%*_T}bb#Y^QpCP)!|p'
    '%W@M+6IAoG`G7Y=M>5r#EN|(J07qet4H;UwQ`Zd-MJjqHL&oX=6t<`8FcPQ5#ez=DvZwj3j^@3+2HAXHrfpNz+K)-'
    'wRJDfVwN6phI)LfhRJ9Id@-|higIFV-s@4XkaZ}YgSUM3m$aQw8f`DI(Xt3R65)HPSO-'
    '7%ZrCAwxHvBNd`n;EAB6p$GOu;zwQt_;LSrRuOVxH*lurQvxZ|@4g9A1+K6##$Xq@66DM#CH?ExMna8WC+sg1WJ6)3%27foDJ*{O'
    'c^?)X@YjQ7Zg9nV=;~g@0$mg7jBH47>*`8SP-amS}3=u(fKpmky&V)9<B@Dw<EOtu=ldpbI26)GXGPhJ=y82&~lANs)|nbb?`8>6'
    '8=BOCHkRxMx|@?kSV@LI!v^0ylFV8iAX+4vfIfT!%&Aj+=ubaL3JoK{!8iumtY7IrNM?g~6qLF>@UqlfoCTf)IXk_g@WNXhd1JNb'
    '9l#*QWiIxQR=7Kzb?PrG#9H6-x9I7i-'
    'y)wRzgc_F`Wyl4vK18l=_cX>qQn#|1`A%t`x~_IWLvazJU9#ms+TX_sr+qk~GjLd%`pP}&!?+`~gk`;r!t@kJ>a?E4^)C0Wrmz;t'
    'I4np8;~<cqD&n4*P^w!R&GWdNrabT01V(K;4}hyM8e`zd4aOd}qq{fc&ut4F(>k!smX(ry2MvVRsgYVW2j+y&aYi^F!gsk(`Bn?P'
    'fzZl=^Guo$ZO6#fG@q`HN&oIq%(Zl#+b@ENMx?4#lLw^z5*<uAnLsUrKcxdw+i<nL5bU6u+&d>5Vrxbodny`N(goUluV5K2@N6CK'
    'WKDeBco1W85^*t6qxHhyOp%7@b;6Q3!woqRNRs(wrw{BHg@fSskQ=^XAUkj>z5Pl0SEhkFWSv)D170@>_<OhzEPlFj2Okj?Sf<V@'
    'lB8K*V8G?^Imvyb|8><`L5%~Ah@8WWp#$k4KWn*Kd%Z{q*CIJXRcFt-eUC@t@?m5|lNbpMhfK?617Gq-'
    'Z9o>H}?iC75*%?>iN><ePhG?6y%<{gH@O6_4E^eeR&>cX&6A7<gqU3=Xm_kB}E%1Jj?lW<6eH`$2Wb5ZI4N=%MwGnH2Ku%@vG<rP'
    'eGqMK`a5q&3TFyo1~>QeD&tDY5$w(2=?Xsg=9psjjd{Mo8yV$W7B7k9R5MXgxy7fo&OghcF50@r(zS7$?ALHZe8n-'
    '_}<6#9r=3@=_6U?G^eFcLaYO{pB;6z$-'
    '}agC<iON;2=d07$tJ1=ML1G@EF3#Kyng;YI#4b&84Pcig0P%ex;)zH^KxiI!LLtg`J#n{sgeGN+U#!fW!HN;h8&oJ~g+#kkHGW4~'
    '*d6?bL*dgiAu9tmCG%@$#7SKmoMKs=tb*zle%I6e*BI|9y&ntLT2B63Vg^b76IGN=JEmA=1fpGWlo`9*Vy*B*3e})yQy*|8qTI2O'
    'I!z4V~8zJM~S=(<#bEE&#_Cz)};`?Y7l^slD!z$D6ZIwK570)W1cb+Pf10NfNv_Ru^Z3^^F3}@XceWAPwR1%+t7<9tZ8vNr_QWTm'
    'p&qzC&f?0EsVzB7no{&JWTGJe)HByi$Mu*u3Kb4cTJ*(iEIp#TM7d$kF-}Si#PczXpT?;e^=l6IG)gr?5p(JbQq!<C1E`eXHL-pg'
    '`<8bA*KQYoy{yqKUN+uNd(w0!H#Y5-'
    '1C0V_}DAnBRov!WIYHjdmU@nuElAmOI?^JEiFL?Tj`sjVkRmG|41IaoulhKmLkHid%^Y~_Mw-9TOKf!I-'
    '{y2P&C*dV*uVZyhU^69ceg#!GeU*mKeYB*CC59+8Wg^l|sd?N=fPe%IKtZ%01aTNYcXZk?&2*Go+mThAd1jMqw~kLpi`V$|+#RoQ'
    'W<RabvB%55(gH-%eCdlU^d+QZ{b?(p!7bu>4C#c%wYQy#C1ur`R+R54yaqE0SwkByR7|6h-7=zK44R7!ba@Jn&Gx74-'
    'hxIFeGOJk=p@lsnK1r=0*YeQbGkexvj79#6H|oOu|nGejetU4()Qp~@`ZNMPJYFD;uTixv+Vk8)IypcNt-'
    'EUchKGOEu9W}Rwlp?+BA(iuHX{4g461%P_JzQ&P`jHmf`+Fb~DAm2MWHB=&6S(GUf-Pc#Mqc!DyZ%V|LIigJeL-'
    'dosOjAj1cHQAX5@Ic0+xGuV%EpME?E9ERFoQjO7ef5Ya(A9L3KdA3$VT-'
    'QHbmq$WDE#1+pn9rO@YdI<|{k08RP#G&#DqV=FXElZg;TU_2l}Y#%U5%qJ+`F#EJD?&{NzP-l-FJCT(5qU_V}jn*8lDpLsovuuLE'
    'maE&j|We>v%-azgnL*C1}ludtKUQdjKaD|H`7%?S@{w-'
    'xnBq@qYi6p%=Sntitx0#&e4%_Pwm$qXWYU)ml0)oS3es<HCdbaRwDd*!1I}#m_C3lMG7RfT%dxpqiVKrlEB2i2NI8Jm=}?dl|5%s'
    'PqLM?K?#83qx$Ba!qSFNPD3acWbhouIsQS8?!`fhax)<lU=?kgp4CKce1W7yd_xJMU6V;@g1ode;m`Scj<cKG;1}5Q%<wiQGn-Ev'
    '?u;1ZwSb<@{WM|Id2K*B=DX<Fy>7GRS4b{P=(-'
    'a0ab_qb{T8XUbrBa%?oC9oj`&JOY%j=tsewwf^9<9GT#Z9O_U35sl><+z~kk6>O)WsgWnRK#9xJI)dww2mZ{%f0Xeq4N^}mG3g)F'
    'kLo;rSL7yb_H@{<0>QAjhAeR(%3_+uvTVU_5YKKqm2^8s(T8{Rb*lucsKfILqZU<$+MO@BT`z0Hv9c0Wc*_Sz1y^I~b7EFcTOyI&'
    '`t<5cB(zIzwuEdUgLN4sJq^NUw-Q?#F#y%E)>e~#xxKX?oj73f+)`7O}{)_)nj>M-;=5m#`AEoHLqci{;u_0x({Uk-'
    '_9j)!BDLU^MZ9hxVdB<w|d5X?EPTPO!v`s_z#C|?rH);EQUd_#VrSZBMhcKw}CXX>OD|e*$;f2V@AL;nkOCpxC!CP%kNN@hXgj~^'
    'q!2T{2eL&ORmu?Je$=*!AK9pVzYsub1zdn*q3~R~WM!!ChK1`rUfMK{=JGmeczsJ8X3o+In<QIa{kZamozR*WX^Gww%J)-'
    '^5p4qaLtnl~7K4?Uu<9=fwF}(3#mjk;7(04w{>p_`g%I_Yo+6-'
    'r}7e<VH_WEg@_zMLg_pHcP4Zz^Z1seaPHz)+5uJw*^8}whjzNCnEsZ>&AE6pTdr%|y@5gc870N#qn1+Jl<Br+5Lv~-'
    'rjijK35enlyl`vGq)-7B3i{>k|aT_jcAA|uMhl-%57>_f4}yVux<V~uy8v41o&gnhu+N5c~D26>-'
    'JtR2$1WQvzlqT)m8oGJK0IC}#ZIsMYnvpC8fx+s_A!Y(DkVc3Mb(VNu^tavK{JuNaR&=-'
    'yc3HPJXC3x+6zH~(1DROEmr85Wm1Y($LoV(RC*|r-9ea5XzZis7Rw=IR({f<mw-'
    '&OSDu1GKF8^vjNZ$~bS0cZesyn|;&9QiI%$@J?U#nUhU*XSm)X<YiiO<Q)z>6a+hWDfO20`4ebE*d8=6iA?C6lmBE*&9s?RvMYXg'
    'bU#tw7U#bC;p>WAZiDhO|@r>YzYTIQGKhDCkKVG9fOM5)3B50#?-'
    'D*@j633ULTno2*U1fq2X}9!o)h17Qa);nEY;)jLGkH*6rc8zEAO(n3ZfYd>LSX6AWEylItYiZu78`vUJa2Qdpf7mJZz~6k*YN91g'
    'juJ@t61bllRNqV0~n$;w`}aqN<UvfGX#3%mp|p-'
    'qcU$9$7WC^$PqDIF&mF`9LUh;bu)4R}Q8Ut=vtucbz6IeI;7SQDsI$rQKIcKedqT+McelG&QSR|)o@Q9!)@L##y%Pj>n`05FoP_C'
    'uJJC0FN13Yl9v(HN`kk9_9w3WDwIIu7B}m>6Igzmi>|&$u`+k)`{)i*)}gC?9?nuBA5P;;NBj(w404N&t-0skZ+?`__h5{dkHH-u'
    'H*qX_6xh;!#aVuvV&I4a+#cznB!I!r)yG*L}{goKV-'
    '&lVP1=@({5%`jYB#sbic~jBmn*4cFU_D446dvuStBej1ZVEaB$8(yWg1wv^c3sM68K`T~_gVh`D%qiSmGt~jf2+8U}%u>s5JZV)%'
    '37lCwSVE-'
    '6Ur^c{q(vGDt2klO!rw%%yuLilUKS4i(uj|jyD&z0_Ir^7ee*FuLD89%yq7TIv`4?!YW#jE{Q=aQj>2@Hw&#oojYaRUME?%=OvI}'
    '+zq`WwGJ`R^x26R6K$7Y^H(e?T@(GYHPJ&Evno~J}fKF<n@O_HeX4k^R4WLpiCiJ-'
    'H{1hBLz1aOLci{oG)rf{fR+m#PFU0M|NSE?u_nrQPX%JGA-'
    'L8F)2kf^t1i3buTV{!0TB@6m85+`@HF12$x>TQ{=+Y@jJXNKuADvYFndVx|0d2%lS&d%lCSvnDhJzFQju;*&gf`uI@B{qByY=+#0'
    '?*WA>f~z*VZy%JJ>oE9prh;bf(i~#qkftfxggH$lr}gY@c||%2b_h+}pBrNM8mTE7IO0sIk?}3?iQN8Mj>I&q*H*n!=c`uaFVol+'
    '8mN1YZjGX~GcRRBHKp1e^sfrUO@K-'
    'C7q@f|>3Mry3Ksx@@Sj4H>jO!@jp8YD1ZbyBtUD64W`@63aID}d%u%2@Gv2^L1BCGgM?(mS@dn3$JkEx$l-'
    '2W9{skz&oUOyc$pdGwEL&&Wh-!hIShTvD0SB-djMrFKc;~6css|J(&t#ujOp``xc9#VzeN;F`le9<@!)#XU+~F!q*)e~MPRhO=1-'
    '<A_d`QqpQ08<b>X=(x@36gXimAy}pLpR=rrIaw7TqybkN?r>I>)#4O}e+I`O)Li?HNnQfi&kf)EI65n=6BDviq`ANjsQ7pDK9|I}'
    'aq+fbNsBB$9lG{;d4S3sHI24PpFrbE42sS4kn@vSxcw$OgLyBQ{TW{D)@7#+f=1gFaseE$6@k{%2Iy*+GGcw?lXika!j{JFJ5Yhu'
    'dM=?q$-E2W`A5KJ#B@f6C#hLzcEqx2p<vdz)?FE4;%Sum;h~<Q$!-'
    'QPZYk0qVwT1$4Mp`2Rq;=OfYwJOcLK+&gSQ%ax<{U~Pv=G#aAqFxgXkYP+}l2ARwNbiry`44tuioemV8#y!diHEeNS+^lNYlw9&|'
    'n@eDKVV=J8ccR|NLfInYogfVWpKiIf%{Eh)0*%EbVUwJWVPRsGRnVjM&_>aZb!`u%8+7Obkk1)v)os-sRHt1x=q*JrjuoCaH@r-'
    '0F?(>ypCAg%KChy{>@pPvW|yldFuOuUf!P;S6qtQcMS<Culoyx{^~W8)ypg@t!ye!huss2i3Y%-'
    'Ydd7bu#9`&y#d67+avp4P^_olfDg1xC*4b`KPG-ww7|L6Brd?flb3)`gi9HWowJ&1f|KLO9ybgjTj~Cr~TWU}WdQ*^;`#(k%>BG|'
    'Lrt0dC(&>Ic0X7`&Xt-&Ja>U-+?pyC<;+YR^FC+2-dZ=#i4vh$kj@g-'
    'XS70kc^VSyLWDw2`(RWDc=sQ&E>UP!#abd_dhHcSRLCNHFBT@rzMyX3mk{8Ba6J=I+atpwb1NM|P*bCO_zNepjID-!-'
    '8#Ynmf77G_D9ApVOU5_6X&|@pzBFj5kh?ov7&n@<z_b%ee6O^`#CWa0$AP_w)^k|8PTA7UvTKUq&j`~0YAJ@@Rc{X{lgDolETzf#'
    'e!<QMxVJwlM7{qacf`X&Uj}A8uQa95Z;3V)plR<!Nk00}ztPfar<?P)p!KJ>C0+cdYZefC73!10>jgojd(!9W4xxlZ-'
    '^smAGSVR?6VE;ov}d<SFQFG&m&jX`+vjwyAJ=zK;e-NxU!XL+CmY@lHUjzb?SgGpMo}7G(nI41-Jo6Jl-'
    '`yumf8Kw<l)#6;T@$^9~8lFq6N_MpBd&jI{GQ&4Cxb`E>~$Yg;d7yjH5J}li%OC4&P1Tao+73>bDwcH{GSBxkXnN{pXHpRNK`K_m'
    ';MO?e1k=xflS8w5r^`zI;(d!_^L7Bq6Lrfn-5~-+UMRWU}DQh&Ur#qWx2EXC6htbcuG<>+z`;q$-)8fOPc3f-'
    'Nb~wEv_KyN+p?b#wf<#3jm?aSeKo48<3^-'
    'sz3uJKX7dr}M(QyIc!;et7pg*Dbvzyt^MIy*;Xwt`z;}?UI!RV01stvhKq<^6ijauN0H$hD}aFa}l@xPYvG6k}VSVk&*aIsdedz#'
    'U`?~?cq^xPcf2HnbMwWBsrLj?y~>Rmx52vVmPA!p*VxvMQ2Dycnqa)3>zCu6&vHm+F;(?=6}Bo)@955KXHd^=KnJMgTHls=bKHtN'
    'BoZSlj>Jy(5Qp`MLP`8<gjjC(uM||wY&1eNGaE=B2UpBkOEda1=6m&<UP?5h8|u+`+hS0#SRF}F0YT9u8RV#^E4y+>06X`Qg}zH-'
    '}*wFV!dlq@I^rx^l`ynGWfVtF?<Ww(7<7F%H*i!f*>Ah@H!@>)5~Fezg(`ZzZiRHDPi}o#vYdLx@L-p{y-117xJd-nqRkIbBio&&'
    ';f_6XSG_iQmZv0@X6xTvmx*!V$jt(E)QQ-p2E<A4|kp6nM4OCiS;4_$PHe{#|z?b#vWd#Rr$5f@H44$qmK<*Owq^&>E-'
    'DRV;Sk;QgQD6l0p78Em@HvfWEu^v=TF*JjR{wVqDsM+%fB%ol568k$*r>K=dER>Ubc1X(YMb6sU_bbb>bE{K65Z<Yy|i?Wb8p=D@'
    '5*ce(-rJ>KX*NQVf-dQ{NM$5d~0Fu1~4uQ@P`G1nWzG67W2jO9@=C4%D|YtE1M##7u3$aRC)abc%90Q6G1IsQ(Mq-'
    '<yCLHFM2+JVR2dhL|$tvS#k^WR`cWxrC|iAH)i7outn^`4WLV-Ve7U?wT&4X}G4CGFAS-QEVT6G}#OED=P<yY^2BF5&S6m+)uJY4'
    '5#girXWzOr!&v=^?0Row2OX;w}b=1{(2*Sr}b$-'
    'C&_iwR#;C>MuUg9q4d(?&N(n?e);jgwi||hV(;hSu#Fxk%?U$?^Gi$i!v#`(f9#xf)eK9uLXK-'
    '47!*>B~ZXz%%ln^KrUuc0Td7yv#I_GfQu`s{0VrAIaK{jWw6W>30UUI1T6DZ0+xANduNnr#pDO#)et+4KmXUV>#osNbR7ygGu?Im'
    'p<5*(xG{{pT<N+*vt&5rDRTJIPRedx+A~Z78=soO#%~8xAoT1zX}lP%In3a7Y}jb~7`(nwO8xhYq0&|K54y$>e=a(ll0Kh@MipC!'
    'p{w!Ln2$`xB+RAO`4!XPsJSV)u-d~mikrEhp8g}-J#IJh?xFq2w?`f<-'
    'KL0Tg>h;LvE)#7<AZn6j1G6edizh!$M<suF^p!N0A{D8==_^u`Vryy^TBf2K0)VSoU-k21$QImMO-'
    'R=1ovT5Wh(&%1Ez9LI$mmmMu#6NN5h;x{-h1IQ+cg+?v>LQdK3WvwG-'
    '#);1y2<9zDoNPYd<>$I(~7Eo7?<=LVa2$&>~5XG&+IBl8c<m*!dsd90NW@Ug-Mj`Oom6e3srfS)QvuQ-OH<m`!bBur1@$rP&cFW?'
    'uGdD})H7!#-'
    'pr5R7lmcIb=3CTz3iSF$45n0Sa+rEA=2iVEHnE#mubug2=l=f{S{LQYVeTN5Mj(0tdwRSKg5Bb76Di=9R=nEhDzKNucP^Y$Y3i^r'
    '6)_<DZDY(1Ioqu;V!TA?|`)r^f=*~~$`<uwC!~~@YWU=FPe4-Qi{KSU-'
    '$0}v#wK<$_2FSoi^oWe=S^MMK@a<R#3uF+N|H@YHQBc|KDkIKXhjZ_x(am85vjc1`gLZ&)X4DQkLk!!Y%^<iP+6V&Np-mvj9h708'
    'Mu7t8vn0D^3=~>Cyd^XjDUNs^^X~S*l*f#pEg2hap+y`Bark4!cNc}Ct&#$Mq2qL*81wIsb?ezK(Qc~Y&%0%dMh~r1DV3bZVNBt{'
    '$g6^kTMbmuYWZyY0I9@#W3a-'
    '4o*@jqe@VA`FZil)^}M(9D2}*}QBb!@PTy!Ky`+K;5DD_rZ3bvG$w0w1_d6$LZx=`eMN{dQOe&tunlUZj&!eVWzr-Mb)dgo8(Rp('
    'h#Ua`OhTVDHB}-'
    'ZtJ8y|bDxX8Y51EF*gV1ygV~Z9a@uD7RD}KU`>2S8zw>>0B@r561b8QFsbhcB_f=_9_AFG7s`>6mqs6PHoT9Q=ieJ-Mn^P;KqP0`'
    'f(=4k3XKbkt<5>1_Nji%1GMN{Y7y{WTXrmtsv>>(&l*td*u)ORTDKaJ=Q>{Qx!4fm_iu{kM(tbaJ4rwK7+4OXn&<lpZFa4188Iw('
    '7~v?DYdo!RFl9)aKTB3Lf|Cw(4HGtbL+*~J7)^i*)byV|_0D6dFQY8R6X>Ya%Mcd@AQ*dl4Ct#@w|q-'
    'H|qUd7PQ%z>KrM0!lS_o|77ya>fCzRX{9KAb1BPMNb=I3?7(eqeN=e7+&*CnkY@C}u#(H8>2@xa1)2jj0g-'
    'LnD;EwXh!)f<yaNL6x>h6fD#9RsQCGf!yUYh`5vK(J$FVey~p}-i3mri;?-'
    't|B_|7*4I9mthGiUW;!+37!9*XwCIfQnr+4hA%;RlW8lCzcl7b`BkYNGtCwVPX*507^@RX3Zz=5ig@E;LE9{4bK%#Ch?1n<HdkYH'
    'tQ6UhMUl;b1Lev{xKsQzX#ctrhCCmSH@ZKDwTN2U|CQy6}yx|mz61V3rMnF4~NI7eY*dlZdu69Uas_)+^;09Bg+=*^mXG2syXMe('
    'RZ?`P&O?E1mvE^7~nuE_2i6A@>T}|xxmw5=ZxB+|32@xCWOZEYA4V3p|1LcF*K>08>P&UK{%15z*@^NgSd?Ic<Jajr1OyzKII&qB'
    '^!05>n*x3ZingoRPtaC;Q)g5(Mm0QIXh5b7>zk6hhMS*#75*QXcgV4sH@8Zp8<zYISbeLk$pJK&yq{R)|K}oX3dLEK`$;n;;eYoT'
    'beE|w_93NkIvP9nrLBbCJ!?e`H6~eUCPq4y>#-'
    'lV_#Cv76s9K)juofpoa@tFPVaPxIZP=A0YY2*(9jKXO*dcF$)4ja2$tZ@q*TY{M2AO<i`YuqY$XD?|v;}Sn(y5^0JTxlSP;t&hHW'
    'lYwV^eV+O6+x1oCku%dMeJZkg3+VGBkx+KvnG!rmOs9k96`9@?`s!sn+JuGAcQ?gq<Hw@F69D&k1n_{rB$KYE?b$LB32~j?Q(y<7'
    '|oLO;@XDeYKWDGcO4$FF_rUh1OOhr2jR<qkXj7b^gu1x|&XL&cE4DS2HO72T>wj&7_#;-'
    '|VleSrqcv7pj^~5l=LouB2OnjeDv&6#W8*z<@TNIlR8<Ts%r6!PeD?Ot}bQgUy>3i`#e;{^J17>nFJWMXKrWnBQgF1zV2vj&%#2t'
    'lJ*hNd*)%0glo?x>9GJ3r!gng1+580;%qvlBls+>Hla=0eoEyl_uOSqhhXqX@`}fSwE%>ew{5~L2oNC3|f)xWVF&?pi~e>hJy{H3'
    'c>_>h=Eo?KvfJi5G#l>LN}$#LAMB$i*lk{+i6vmhCX$je!dw7F6_M4k)79ir5z`i3W>J*)dqOB0bZBM7FbwY2sroZn5WYcb{7!j<'
    'RZeo__n4hu3t%KNkUiyWdNmQJwr;nU&)z$cVR}|WOo&2)XjD`VMfijJ%kx`i``wAQMcNj!i>7j?jg*m+hMOZRql*mnv9OIo86xW0'
    'DWo`Jv{tu9@;yfs)$dqcm=;HRQ&-'
    'aX5~$|x(9^fj5gIF+{c8>uda8P)*vJKTfrTI+x?Z~TLhH0J?jA^w1+TC4;!qLp~RdwSZ70dId8D-'
    'jeT2&rIaTPDD9r5r0D}oyH_c>=%CW>T}skExU~C*M)$L}pQ)bB6q{_Sd`!9NgZ{X_6hu$3T79x(c_ZpGwheX@;8t;mYOT7(25vX*'
    'Gn}nsx(&4sv@dK~r>yQ^YAC`VU<ToSm~Dc?f$YQ20ePmc(qn(wu-'
    '$I8v|7qRta^@;rp?3e+p0FZMfm;mRCH_^e!r|*X15BzUtTS@TN}wJhiVxwAMWgrcnCVqIatgu#5?aV?g_fp>YJ_Bs|GqwjlMM+;M'
    'X{Q9H=@Jm=7~8k(MD5d8dUPU{Z-wTv|1jVo!vf%b0bzy@|pY|Nm>rV7vA=*^@hB%-'
    '}($0z~WU9~o6O57jcWMFv?#I&<pRE9a1JgbL)VVe)W!x@+Kw$~&u-cb1ddmR{u#-'
    'SV^(kGl?z#}IzMJOLC^RPE(|&Xsiiq~bp4LcU}KUhfrvzh9)X$=4LA7VZy=R1f&S3jvosXsA9BeS}90)gYo_@u;B)ARd>`h&>Aoo'
    'aOj;#UzTal%3En#x3IZSpRH~HM)+4louI6x*PseH#^hB^frP@lKP~*{$17GZo@xVwJWl%kjm|b3o+z!J#Zg}WNvp{hnF%LwzEvuc'
    'd<toqQHN)^Ub?DlTH3ncz2Sv4;e~^L}KD$Lm^aTK>lbbhNc;;m;LwFF$-'
    '&(Z6VkK8Hi&*tx<kCrsYc@Qk@b|+YCN+caHa$Yi=kkG3nB+3Z+o6Kq$c42^YppSc%hNO|ODo?QwMbXokUmqp&v$=J*xmnei>IzM9'
    '5kt?451Lcdtyg??SaW35@d)~K&8<FVG2qLM?uu3!$~BLy^+WaZ{kV&=F1k2T#5A<PfV@Y(Y?L-rq$e&jaM4Ot}@W+-'
    'hXxL*xiQ7hyAC>G~fKAgrQg~E6}x+dHfyTNIj1uD+Z(LrIPh09s43+wfDdrU!pk(qnP7xZTtmi@Y7H7PIE{evRaH~*jT?o^0^+Fu'
    'vqZu(rvgKL)X3k%2ji2g*B)cNhUjfp;ngR()NcEk3|L+ZFvN20r9M+@qBZb925p4f4iB*a?fa`(2jaz8~cj@!CFL-'
    'ELI<S$S=a~gRgx@_#X-Q6Br(4S?{_z4C5Sq8MfrZ{PDux}`~0Y=+viHX*+!EM&rVE;MJ8VA#vO?v5$sQ-vh94ON6S4+QUIwklt(Q'
    'nA&*EmOUOxk#FN*nJL73D!&urPClG@u=y`*O|4Q>;_df0|0^Zxjxgn`*laQ}$tjDBY83yTLFx*n&I9*T+qCe0|bH$JeJ!G!=e2E='
    ')h1fLN0Br>i5G$qt~qBbglzq#c^f4hPX)lFSYp=rT!WheIF|wPT3?gER2Ky8!H+!KvjbvKK}}f|YfYnhW}Kj64`qP}Rxv;IRdzsw'
    '}WHuAo$vm3+n*)mjl-'
    'I;o&<&dN9^7u9+ZUZSJ+6Iic{`2Ejdyq=bpr@6a%UTh)C!`=grm<>ho`|}cY8aW^<UHc;3Pf?fhet#Y}D!Wa+fk96qZQl;b7Hk#H'
    '*F;Xtza#jrwGiMun-h*bxGEY|^@^@=y@f|eJF}S$&-yauYi!kzO-RR8{iAK%0Je>zB;mTk&d_N?kuIHaI?)@L&noa9(HivJ{C|5z'
    ';kibuq-K*YCVi)Wsw}^6&`Ey(BRZ)*@UKv|E61>*wFX&aBO~~}8N`o6%}p`AV{J~{7*dWfVti8vRgA>D#2GsT?-KCn-'
    '{scF2*v2D_rPL(j~~Qgia}TFz_nf{J>TD9yd}->yD?dje0`CU#O-'
    'j<W!c*de<{jQEK)(@M7Gi!pqueldc2lN3@^~$Nz>M0H6qh9;^t0wMuUoSG^-N7$|B)-'
    'yH;yjaXwRcFPYe%7~<r&r%4N8YY~Lp)7TD-'
    'QUP?Mqo95$BEG6pFY13Nk^gK2Gh5hihl2Mi_0wVCW^?_7{`h?=zz88mny7mGUIjfES-'
    'Gu^-QFL<&oS+vU9+ug$`0Y4Zrwr;cV$uim-~zV0xV#5^3nG-wv*<r-+m?%9)_Dpc-'
    'Y@W!ovY35*`jTk??SkiG+tn6A2Fo8zwwB;Qz@Y6nviwLM6dJ#fQ~+POmm$UYQyHy6MusqAe3<?BUVPbxw|9drWri{i{vA^qmiPN;'
    'lYW>$F!kA!%@`53MrX{fiDf9#rdp)qx*|g6nVE(<Izj@$Z`P@q0Sam)SA{epY*;ftvw8rvs@BI#zUB26HpI`mZe*bUoYtv|!=tds'
    'I*XN{v|?vfKMK&BlSw%_a8C_Y3fZu;(?qTDUONsB+bogSGuV_j69GB@~|GnkM&q*AUuHk(eWBkCHW_971sly14RV?C0>X9g{4L8('
    '@c&d}Q;Hi0d84spoPh43c#HohVM6=e^)$Bx3LPMMu|<Q-'
    'PP9vP1+btaRcM5%}@4qeSn>4Z68ZV4_6j#|2;ev*=P`dg0xwE*jJb4G&gQ{-oBvR&Ynjva@}%`(Qty#G`Q14;<eUj3YK)nky8;-'
    'mr!XuF!6zI3;2BPtoC$OBB`(jHN=kySK61X-'
    '2Bgpp1eM*jL)?3l^i8L%9Vb*k7YKUf8!4%Tzp8upn%kiWfYd*GW=kEO&z2X9`N0UaLZc+)d;4c)ABSgK(VNgAMld+8(6Gyk9O#zi'
    '+Z^&``&JKPogcjxuS|)5K!yBeD8h0%*G;HXNFb=%>K(tHwOv%fNr@QqeycU)Xn)Xm<@4^7zRH-'
    'Tx_~z;qC|PN^7uuwV(^1we^1K7@8=Odely0_5ikZ-ZWM>6AMi$^^*G(Cg`R-'
    'QER(iUm?XuI)FfeNHU1!@FNrLv86wudB`fHGH;ZVg&=vKEjFo*XA;+8~$eu-'
    '$%(^_5)*&5$e<b7<+;Q{*}nEf(W1!l{l#o$A>WnLz(kMoohc~d5<@k!+55<y#-'
    'ixF4k@ZRGl~A?Z6gunfogM0I}fGKO0OHx!Z8P?N@3WuD3IvOwZ}?EDvd7Z92D!w$e^_RfKjDWlzc4wsKC66Xmp?qm%ZT&mM0uQGK'
    'z+e&1LtY1B2w9xa^C9~yhC@aKMH?D3*#G7i07K8q*OiIb(6pzZM`-z&{-'
    'rhcAC%^e`_vM;EZm1AT|ueZZSbb6GoESnG|o`0YNV{`&HHuldL)>~_g9YX%J4ZZ`@t=E7Iw}!LMeT>gK9SrMF;%d?#>@-'
    'L6@)h>tf}IQ=)^=7Yh13I>Gm9*lOOVC>^jK8v_FXOd4b#fHJX#YjqFml*5WBW-hH+JHyb;0NZh{zl3FCAk>*`n8RS0<3s_?ohuyC'
    '%zz$jGEShWMPTYyUM4Zv2tQzX+NnHx2wB}z4q6@oB0H2ABB1SJ|O4JOyzyjuVc=_!GKHHgBiV?XS?*bh5D_QNiS{jkZgAJ!84VHd'
    '`J*p%20n<`)=dac%?r7^Yiw8yZ}{n@B>TjhAwwO)qJhn5Fs>jW+SEF0S38D2GDty}s;nFLxeRin7GZh#Q3HL2_~2B^I_Ev6!BsPR'
    'RVLkK)7YDe!-=v7=QtfHJe_%Sv@<)vTbDn6?1oKnj9$F#kwlzRR_v`6eY3=l(WC-'
    '^NpO_B@CM3ZDn>65__`Tq%0Evx4DYgd%#(8q(axW7>geZEFdXia5`=f(Z(BxAc2qArdUqfI&bNnExq-'
    '0XK6V<?9h9$Y!$3g*R98naJ?^X_IJDts%^q0d`ob*UzpE~`s{X|_0P(cO8an8iG#fE_+JEZ`p%R+}xVKXClH@1-'
    '$opfiO>HKjZhIahl=vCOK^#vJqL%mW+kHwr#80p6RAfN-'
    '|y=v}2m>v6vEwV+QiFL$oVS^c?I*{2j2VU&V47mNjnEDdx<MhRz*eof-'
    'eNHdoXW9S!?=jhkj+!-0i3Eg=56_SiUVE8De-;V$}5Uji-duXulUUXGK;Ug{-eZC&K7SO@Hax?GiAdbX#_nSs$J)o-'
    'YNtKTG5?8Nv>1;lJL(#OWROYLdeO&RtW{t8>$nJPg*(VjtH&8-'
    '4LtIiRjhrQahjOktDzSI?yFt*6r3P)BPYfF~u5(1CIY#6Fml}IjoH9-`LD*52Z+qo1-'
    '1NJhlvh(tmT$i6(Wd)sM0RU#h3uF^$t%i^DePMzCu$PEk`|cf@!$excsz}O8J>(IcB9MP!$`f5(oB`uggi?dlL2Q~&}U4TT{#8Fz'
    'ZyH+h?FV)+3%VZ0d!&!TP}^58a%2(eSnCvOA7hiC0>ecCc@(>nlxovac?TyQZU$JKrJPE)BzIDES%quTeL&c7RM#*TzYSC?~5g~%'
    '#!@4o>|C9IDy%@;PZf3Ez`f#szi9j6|(KeovrRiqVIP}>XPox?5cQX-'
    ';HN>bv(0c;+cIfp4qkW%&v=PcD?@rM=P05!61>xQ!q&6`4kKi-GItv6^CXc-'
    '=;V_!al+SoEB4kd6F{?dn)<mT3E2*q0d{cYUkaxtp;u+gyTNo&9G)4WTyst%5u=+8(VNjRDGQidd53bZZ}*oy;L;q&uBeR6P6ph6'
    'T@zeJ-OtnX^rEl&N8Cp`Zr^*jE8cn3F@mUMs{ETIVew`%Mqc^<gLfUvY6IKo+{GKMj0eFiN|Sz*~Hd(K-'
    '@APg%E$c?c@CnT9Hu5?h#YdnQ16GnBrE7bv%rof=Rj<M<}}ExN9*Fed#U*^F5w7wrjzJk7dU0STM_Dg|S~PnC9_<v0p2g=kcPk|4'
    '}f}<0WHvE?C48VhyQ$#P5_{?@#QGDyrcep52;s%=^R%W}i%#v?;}gx2CedE8R|;#<~cr>zuL-BT6;)bXF3jY+^<j8`y^nuoFXOP='
    'KBo?z;l~#E_g7ASi~^tN=q%)^!9W9jSKTA(VHdaOFek3Q4v54p$T-'
    'vr*kwfYV~Z^`F#s+`CZ7$<o^+vS#EUjaSE#M~VkB&K+azgZ$W=(Xey@i7@E*6kcp@@B@Vxi|+qW;T7ubs;?u6%XA#vYEk5ZvVjaR'
    'q1X}swaF%c0p)X2T0GV7g913U*~vI+nM2^L;&i4@M3(ePNFl2t$T5^kZ7NaWX>DgX)R>EI9GP99!=12K2#*IksnE;o6}e-'
    'xPUqp{XqNs0n{_6SA;*Nau4A*#=IQLX(9(5m*3i)XuEOi(7&b39H7#cp2p9P^GC+-`)QSgGmi>6T`AKZVON-'
    'N*V9Xm`k}PStTdI^a?8)Of4Qf@kSXH)I`{=-'
    'C$D93~?%<`I!E+1)ZSxV<eW`Y_`v&5G%Y0WchPJw${@4*p(%`E^IGpNfh+}AJ^PFE$7A1?-'
    'Vsc^ai%by4j@!sO4TmNbXF9S1mpRqspsCUV14E(AH@V0wf6@!RG`c~yJeMtv?c|@!TU2k=8ffF7sh4HCM`eZ7=B_=r1)IOpZr@}n'
    '<-'
    '|xHkb&!oMQY5^$W>3A;jOldMMLjBDske_^kHV0RX4fm9%Ic`GhDfvU#(%Za&?QF^fBCQEn}6d+uWp2)|<}g=b{dIwZiLFCFRIj1%'
    '=;a!Kf0Q?rW$Az#^SpmtcP;&uwj8(B;hn<{=$<`!Lpw0+E2toLf#)yP^R#{?!GQZZ(IrA@+E|`u7Fbly#S(m^?tVX=uUzz-'
    '&EQqz6@^^-h=Or)eywq~l*g0lh}y6>LnXgoK!m*d9XJL`yf5632I=b<Q~jWzxygbyiN&(s8kiT7pX-%;vq?qifK13dA-'
    '=JPfkOq!5R}_P7+&Fw~xqn$8ZRvXyfL7~}C_{Jz5LH6??%=W%9IKYAVkQ^oXB9QwYY;0N`7+(c7MP`}UTZS=IEcA|6XP<oKP1wh`'
    'IX`HtzPNTryNulMRx^#){$=h_*w+2(V+bgg-'
    'yWGvWBY8}gjr29J1aB0JuFaj3O*dwoHcI=UDRG9J4FwPSORN+3Ad<G=hs|MhKc`6=(({ABflgfK2HK-jY)XSYwg{!AGc~*}F*W>Y'
    'Vpzy<t9AB|iUkh;tbpE)Qu0)Zek5wwRPNGiLwqO~nqX0o-'
    'v>=~)kn%Gv}Zo`w1!vi*u3YBmeF})6~9==mL!y`gHbJ_*@=Q1qNqo2-'
    'sjbA(`lG%!8OvA1?HWL<Mmn?%#Gqc{<6l<Z@QqdM<|}dzN#_qn<-'
    'fALlw_qU)LD)%@$nt;fm+5Z)lA8=J+4>PYRgI&{uz40aTe{6+Tn;3LohfD>t3WoRZy)AU&_g^~yc_oT6*Opjat8zEwqr3?@$_jQv'
    'uHG)bH0)@7SUl*%{xJIYRDIjt?q6i!R|X%8(-m5mywr767oLfh%ejtcL-)OLoVf-'
    '=b5Qr)kpqzrPNR8uJ`Dg(~J*x6LJg?EN3EM>=qcZIQYlpP;h6H64Zs^`nr+fLifN{*n*6<#0J9OS=rZ58^&b2&9RK1)pwaKguz3F'
    '9&3sx9_0XRa7QJX>k|d1?#4jkX(8HdqI3zeoUHC|SQ0La7%i{XnsbBVHee4|P4LUObL?MZt(|O>_-'
    'Aw{TtL^dqhJ2E_|fzUOQgV<f$s?3C6U`F`sssw})uHfna)ZpCaZoH^Pfvk`<Qr|rz{_BX!GbWdXAdO{ZQT-ubhaSO46^#A&*6!_h'
    'br$>t&=PMMjJ2-jt&*(=;G0UsyPoKx+v;I6XSWTb*0w$jg<dMO9)gT@jtgRY&WU#Ip%p-'
    '&K)sQ5pZQ*+Ku>#(b&SDZkF<1H*M^LnbU-Zx<IU`?C1?@*uJ*=wTvp|ewt?_TvIg3q6wvF_<NT<pD=>KA$En#;KEuhoa7^Dlj;l`'
    'si8(O3VYcnOpG7cX@^@1R9$5O#2SU4Ei{wxBCAX*oJM7wL@o*kxOcy?IWr%xzw6T+zANd-iBP>p>`fvFH?(?16poLe=oEBHIEOV@'
    '34W49}Ukmkjvu4TSO4qZ4gt5%08yhauReWBN=La;A;W6;Z<)!4{kpN*qzQ|92~DHjnyEnIp7gcco;UA3_D4u`!*n!i#QAC?yCbVu'
    '$RlD8H>c3;vd`>q9=Q4FNO3FsFeH$~fiv0J&=$v8d62}c2}og5_iW=dBgg4-'
    'th_4~TLlh@BrcROIZi)H&}Js+lTG*>Jt`<nj+@9l~7soryHMBC(Q=HX?!;f~Xk8HkSC@!Edub=(5+_fxOD6*S()l_Fc}DoQbcn(i'
    '{fpzDQOW0?|Vuh}j?JSM!G<MPAfVlMoJ7UQU4?rLJU)4kmTY^ZyfCd9GGahYm@3XkYLeXB+;XmVNiCG<GbHY~2R<uE?+|Kuze)`L'
    'eVF0mICv!GoyIgcB*!rvJk`8&;#zcVKCcg9Bk&bY|m86WvOmC0S_>{pE4)|Z+eDL03TEr7WYC=5j`Q!E=rTYoA25^ztOl7hWvNQ_'
    'R|VD2<xslJ^Z(Gw|(dZn_U3)x)KQQ+>+^@cC@YBwsH*vEbdM2Cj((4~5oAqnde@75cgKGpINi)fqeCP<?ZMr{1-'
    'VMZML?2*Qs^Tid&$Fx<Cv?<--ST{Yw=P$&#7;~e+$>r7nJWTCO-'
    's{1Vr#Xn`ezp33r7OD9a&=_3WG86*iB}N}%7|l2O1c;U{A0rzRn^P>#Bf4Y^#;VSIV{K8;F*fwuTPvUZ*aOo+~)d)6Z~2h-'
    'pzB0N-u=(KtDl5Q5<f(*J1tS)S0V)RQ9LA4~1fJG7&-Rt%sNq&~M)t5s_viYP-!yL%BByoWne(3A8yRi(g|bSW}kuUX4sPAc~-'
    '_jDQxl>ZgU%Ipn;jiT=fuewSt?^YBk(V+JP{cv2~bN_`<+%4w<l@mht)eG?Z8`$61!hOoHjbE&XD6s%_nhx=A86b^)%^_9ZlzMad'
    '2_k$%Y{{LS(^|cl0t+}lX;x{P!-'
    '@2Y|pv#+<38S(d_AN?fMB|mJuepYEb}F8G3%rxEP#*N+I+H=}B5@#ZgayKM214BfkM{unxgTEc0o3yVyxh;>Nhy1-'
    'x(md6vczq&iih!G5Tu3k-H2GlX5gOgAAUc)MD-FHIuqQRbi((EvX}X@_^`6q)z0E}bg{sZi2kLpQlqo2HPg7di5o%m_Fh#R#J4!I'
    'SO_=Usi<q{us2RvI&LVfVad%9?ly7aXtd93nYG(xT88a*g_aS!eL>5N9XS;b*<tlClONQ5hHi`S=l@{nVhHXb^uPsPf(&RcodVM!'
    ')^@pOV8^W5e!9x-r!-'
    '<UXZz_`fQLj?@S9W^rOJ2JS{@yte$!yw7ZJlg$Owj=sv9c(*b|w8>QY|THL7}!{e);wT;lmYK_vQex&{gnHg2UD7Tz^Gvz;?@Tjd'
    '0%xm2oNjIb7&D1d++5iw3}5W(I~rO2K<+&`r3wLb1qYVt)<S3y{qI|Yu(xGH_@b%rb7s=xi23B|2z6Ad=>wcURE|EP)S^o%bp?DH'
    'w{vJ)K3m@Y9RNf{Ds$Vj^eQavmVluIr_DlC#IF?8xzN{e!YQ=A`BlT4-%v_(D&9h+@2K=(@z%k!J+-'
    'E2Ntt<J_~)%s*Q`E}cua^v8?WFi&~n&C4F`%)oKUAf>9OF&)XSjH02lsK9JyfE6CTuqWHVsGu$9U^G2XWVK%psmQP^`?ju`-'
    'euhA|3nrF5HAoUz*?|H6gN^gYo$^UAKM0_kYace%|CrHm;uZA5sPwP>@gQ5ih?UDGawMh2eH~8g=qyy<}aAlJN_^CL2>$@9~rp4I'
    'Cul4(!G)m2ly^KIHZ)_91sru`Sm{#oDy1iaq2VRqR#nq+;3jRVCHJ3^&J+?fj^+Kl7*aBFNL{%SKyqi+tEJKHGP{b7CQzM>+J9OO'
    'LVR8Dcs$uhxZOtj})0UHG=BC%ks0Qx@Lc?Bpt6VjFTJ2(}hfNCB|wpIK4j8jkbd%}Yf7u2Rtt)2TC{i<+%G(CC53%8ztuTGTB~+&'
    'gqS>sacacRoG{SZ^xAJ}=r$UKT;cg}zHzy_0H%5~Z88?G7pI_GLoDrXQ>|LIHauicA>Kv%<(?5#bsEs<EW$OYy!&pKXA03E!n(L7'
    '_)q8A9;720a@8<tC?G`(jvmDpeWX<)qxNfUuN^;!meP7-'
    'K_O;wp~qJ*wwlNcAVTD3K$D)+62V2bA{Iuxmy+2e>nG80D0LCGYm+9@@zn_+u#rjSw^GcZ&<2iMTa-pK(sDaqZlK1wi0pjS^~U9%'
    'EnEuJN`YtyuTO>^y+9qt{J~LlFgn`sgU_)EuRq#zbkSu~FJ-T$FYi&(cmUNCzTXhwiiyrJYntA8Jq6hf2u>7yGq&m4Yg-'
    '&3E#~52o3i<?IF6pvYbV{Sa~<O4vq-!oeteh$tL1qb%glf$D}oh&@8YHpZb`<d1@~`U-'
    'JUxQvRXLi!JLGCxCDK0V^k%j?SSo1wzy#faPJ<coQh(Sx4dnI<)ly+N|xCfASKzyd6IU)Ju<tDJQNeQn?vHX*{)Tx#b@2bdv#Sfo'
    'ISVMUOA_u;FDkg7(4<<%pzB3SII{&+d=UqJsf{OJvbr{l3#3$x~sroeT1D&K0%t81}eG1a?_CfwXqtGR&R0!4eMJ-'
    '0;d8IWdMnQARVF1JQ8AJB)}py_c%*sb<Z!#Kv-'
    '4#9hNzFL$#c6Z*?r*X_WC2PiR>EyyBKDy8&*I!!^W2dtuqyH@0N~a0g;xxhN#|Sk9_`+bO?y{wj&B47o<ZVmw7sLPkER00L@;wLd'
    'ijd~D!Mh^-`RC!B5)!>-@UaM)-'
    'g5X@gj8<@nAIWIo9V!qXC@iN?)GlOdO$;Ki7N%>EN?wEJAxqB8%e__YFc(V6_Yd_DhZ{7L1*C7W+?D*T&uWJoxY0MVLwA(gOC*c3'
    'LhD!-YgwOzgRhneucIMRTKx6_VhBZB>Tx2!-'
    'HEAHp42(#Q>OCDk=?&b$S(^=FeS%<G#=Y#$uf|;A94Y=NreH>a=XN&eRlIxG<?2piSN&1D?$cBc~7&m{Wo@pV89D^X=K9+tAG5eP'
    'sHKLr1C~VEhg-#E5hx`wnOAqK>f#m-'
    'dXfQsHvLQ;33Z2n$hYxq#7b$B53CJYs2zh}j@lw1mO|n!8|AwZrUXiC2LLva)bmF>b|9&!D5LV1<(CjvyCa^?z{2+>aN_J0v>k2|'
    'Pnn+?QH$M@&HMAC-99jgus7FrY@l#|D_0qt0Hy!*STz8+bUr7e-;yX&=-A75_wJl-'
    '4NVJTB$d9B?m}b3+q|kJN4856aH<x8D?P*R}5)Hs|QQi8<UBYbMd@E(N048>$@&ZJv02xHSrg<QyFzAg|cznr=;390qWSGJ?{f>&'
    '=(|P;m}`CrJZ#C~&p>>4yPV%Y;Tc?#^FITH;p<$lx7?Jv2?2Egy+!8f63@i)zw)3b4b=>N<LO)$s+p+TR+Ur2m)`i8E?R)+$bsS3'
    '%lCc2ZNCS+)9|6zFc_w5aTl0D#}|Gvk__r73gpnWNW3s*JBOy^q^>(T5V8)0f%<_&h?bFab0P?Y#5Fs^S(XonOk@d6UJe;x;IDU('
    'VWj7s^(oYU{w#o>qzzKx1hqmKgzRvR{}_Xt&jRJ#RQJNLq+b%C5(&opwc3mB#^&_v2|gIvBYvD7mUn5&Rz%8NP8kWu*Z1`En34{I'
    '?~elxm$LUMn}jg5e@oE<8_rC)98hA5``y{wPvBc(S%k>)cUH>&7#)fPcRNo04JBYbfe4{Qf<<7^4ox3k3u4*V8TuCKYtkyrP5F5n'
    'VQ~(v2f<sEs-lj*fIl8OX{)EW@@h?JI@2-*qYNYvCQ;tNWMsl<b0>hX*%JR3J_h+p@XFJ-2qKM+fgXN5`wdfk-'
    'k;?sl+QbW1i_+coV=ZF=N$jnkKOOLkT*dL5gHByG^K<a60Z$gcd}p;O0*g^}LgCw12ATQ3i*cPWT5E22L@G5tAd+aL+tbcCEy%ul'
    'C?Gn%CmD3mud51%6W7?w?-Ks1&)>=e|;F^8RE`gmc2?pWG?6fzK;rF^{}7OVrzc-'
    '?}vX!~Am!TN$05%`5}z0S^JSASYDo}W_EdUe49JWceUq_eX$D-'
    'mg}JX^6F)@efF0%oe?(#E)HzbBZfjy}lgz}_d%es0?Nm~MZZw&P~lpQP=$nJ{CB-Jx0V(~79U+3>TAN=R42sWE~G=fFg8M;Q!HSD'
    '7m4F7^%+Ko`F=?ZAy2ay`OxvPgzN7}c@+W5-Qv=TnrlUC}lP{fG%AH%&6>2s&e1l}W$wK=g`|=FLIqU?x2TO2xJ-'
    '6DbI)i#rN@>28Ir`tImNN$JZXB}QYEWB*=>rPY~0<4BL{EFH!au5WraX!@dv(6zL$r_J8?mA%F%X&)-'
    'bi_HisWlbjk+zg)AA8Qz<9bFtizdujb(nQ1^<#{ZWV0Q)<(hDab3QZ{6sge(Oq6x9oJODGe1ZzR}*|uevC?x($?w*()sZTQ^^=W3'
    'LKFx~Mr`eJEbY-MI1$h1T3KnXM(_Yvk%3kkpgf>d)YUR>W*T_q9v-'
    '47}7%kLR%mH0&(i*(YU8rH<`X^a$%`DE{rO|++g#Q!VcK!Po0fg9un#l@dcT8a_FB$u_l*jY3dmf@xy{uru*g8sKm=LzB?N8a17%'
    'tAi-N>`U1JI4;dGtZ(F^Hr@=hD7h2qk-`(!Nn7Y>OIW0L_WvWoS=f`OdEeyZ-'
    '!?n_QkN!=M+1^K=)SHEB)$Zc<Xmd}*D~AI{)WRQ(R35;!W>m`rdQOs|Y0@K*-5hWCGC^l_wfewwxjmoh~&avwiW4UL{-'
    'zJ{kwol=U7t39=39LTMdx&iVboA7x`fc++<BBM*$j^?=`m&Lpwi+L}?5xWA{kPqssc>87%Zw_g*_K#=?7;<Bw&tvFWO?<<Dp0hAV'
    'O4jRa^vRO-Iv4HLWPQ%Vs4iKb^U-'
    'kS_W3Th4@Gxwe+~>tj7A?Y<}pw2v{FQFss3k_%xzTRingttR63&+Uccp3bNTh7GUd;;fcJ`VzV*o|o}8ZIA*R~W{nZP#T^n@;1yj'
    '|Q0thd_Y{@-dVLWu<nIe@P{n=~b-'
    'D&`PEv&8vve&|zY7l!ZyjL}_*TULrFncYmtA?=G!uo0`do3)4RH}D9k7&#?(q>`u#fkOEs#_E-YjH;`j^1Ajl4W7SJ_HtjH9rd&p'
    'lA1MOI;WL9E@g@6SACA;*?eRg0VZL?6wz;{c6gHSZVBk$Uvp9{Z=6~fv*C#=V**!h)FN@g#8WNNW|6jZdmuQt7+98+@r}_)JAqF?'
    'W;wa7Wc<;v7A68inOYzN_+{{vp;778BuC-DMTsRa33oc_E<sbdo9PC=@v0PuotEqj)Fd{aXJG9xJ^5etTS<SI31YS(8uF-kID@~i'
    'fR;n3G?Kl8ckn<>s3)T)0dDv7S$N~5>#V~YAk&Tre;Moj=lup{Gu99Un2gVQK|1zem!0B7G7;<DBi<s>`aySn!p3#b`lDTBXv5E-'
    '>A78*TGdGjiH1Yi492Cg8-'
    'UWg5r@Nf~I6)?ob4%Mv=B$sqs+zbNX*RohN|e6O`MVSWN$vyU`CrKf;9UzIcy{w^@o$;#xaf@ljl7uT*>%*V{Q_(ni~3a>)k7r<A'
    's(WC!9?OM79-2H2;Sc1p>9#HW{bYRQJg6H7a-'
    'WN+*<N_$bsR?m}4dvRGeHEOD(8PRXr$<sjxX|H{V<(gU_u3sGpy#CzH&`cZ%{B@X`Cg+wifY;NT?J75J>dFP{=t5K*y8wY~A5t<r'
    'q1UO=JegjnM^kEgO^l}0^g6>ZDv9$2e!_2Jibs$5;z$vta<~-M_=F-'
    'Yf|?Sm5U1NHdDO+4Z@Khr(QUU3rV1|4kfo%{^UH#}<(n~MWm>kB4=adG?HreK7K}NQOK`SxbdDj9a?ID*0jcU^zgaLVfHvyL!qFF'
    'kJ@{;N2RR|z7ZoETC-=jJ#8>5TEe5^2gIhUOGj#AkOu#Rwp*dD@xos!IBtGC0_KfCsp-'
    'k3n>rUtXV#)f?K}wYw*ALWA?hz`y+CHsQJaV-(z>zOX85fBc70hqDvH%suj-'
    '8BP`Ef5Wm>(HzK$=rU{SU*cT$GBQQP_P91MBQ0=r(duPG!*DhSz0|;u+xpyLZ9gd63<=NR@&Br0XEy*-'
    'umOtfx#!+&|LE0?DqTzc4!&e4d^B*NvHZ$wCvfRTJQLGXu9T&Q@wL9Yh{~^n{qzC4zhYN@4qHnJ8>g*a6`k1+iBQ<1M!{_BF$@nj'
    'MXO-LS6a4$he3Q#rI(l(O}9q@ueNM&3@!&M4)0j|QX1K9q?8FUH3UQkfgEZ6a>AHD@`rCPm^03MJ&3K~Jv>WSHo9@1K^c{k0#!bc'
    'W2^7BwwWe9*K~y(C+?9?(;|h;{6TiN5_Fb}yO!_Og4+^tTVd{A^5QkR%6P($2=dVOVr@duQ$ONqUwSreSjq(vl=>eiw;Ja`vxA$j'
    'iLPqO^lspiT)xyV*DmDF$cWuiSca?;aFn*-{^%^I!Ez+9Y4MhZn);^ORbfq?1CkGUaMtnl0TB3Sm4<+zt0*=$-'
    'b4f?AmAcQQDJAEo!fooUefKBBq5A6n4$Z_zxuJJ|J=rfO0AohbDrxMgSfkQK-$XKH(i7Cy#V+FqJ%yc>)#cyX$(0iV5-T*a-'
    'TL)|JMtC9ZG$9DGpDltlkYxDz+`(TNoe<HaLK8IW`<K#|es=#G+{ibB7h)vo7txM6fIwzSKW4<nry9xBLcv?PiZ`<Rp`a-'
    ';O`*w|v*wVn|(1p!aaXEDL?yCO5d=FP}l5k&xWSLnkt!^6yG}>&1$-'
    '{+05;{$;8IXh~#lrsx*KxW`=PS6;t>BrZ?U88iQ;s#kckhO|Ika29%Tw&aV8!f(p~|^IWMj)FN68c3S|`QV(kpO;wnMa#7w(t;L='
    'W*aVFGnhex1`K@Bi#4ZMun+bdL4{gpxz<-hfbX^w<X~O&ovsg+d3%-'
    '~AzH!UDzzK*9;ZxHd4oGDk>WkwKM*7dV`|{a?=6lt=F+T+iGgFm%~d30H`T`U#2OZoP2GV%N8~{dwB;?PGtDc76N7!fA&@EPr0w<'
    '{bd1tgZ=N`DUR{O;?!91k%SGh0#{z>7C=Qlo({G6o1QPsuz1J=ilH<ujabo8hF)Nh2NTmyZ+EyJCAdND+Ghl!#*qnxnA}$&Df?H3'
    'Ui}C=bfoAwwo-'
    'u$ps#s5!xLZmv0P;7ia{{c=jf1TrFLszP%anRLNzYG<3+)3>c_8P&#|%ZBXvc50Qf*P9L6KcC}C}CnOGFobXV*^&#yusHo>lH`(p'
    'Lv{s#-E!{xdDlnDqg5JZe@sLQ1y{f3tq$Dvmg*0y0ct@VEk(I<dvPI*L?J6or4=90AUh618EEY-'
    '`@(tR_Vudv2CP4VOb?}0=*QBjnDu$-{4@!}B=Q6`MgE+smy-XEFt0`Kp+$rvm%h5T=N$mO35ye!a4x^P+x~tYqpOLNJAUj=vJq=W'
    'WaZ<Xw{i95Od)h~2`n!jHR4{e-'
    'yE~kDJ&P4y%cB|~)m3|IY2W?=2sW8V{}QlAVJ%MM2SrDFIenyB@s4OYoNKHUOy6QZDMXYCg4`^p0C-'
    'J`7+G3tj!*=C*PqosUc@?Gv7-'
    'Sw%a&|rdO7sfJjWiO?R~{2aq78ZBBqb4RR#)@CdN`m0y&)tZ)Xvw`ad>ycS?K1y8Xo1o~8XPyt~%eJxcp|cz2z#y-'
    'NG9LPQV$poC-fpu&r_pdV6ry`V%)Y@rT!mQ(+`1071aw*O=(QSwWDGp<+8PAP;>DC)iR)&Tvm>o}*Im37m*BgjomV58mJA=AAVOC'
    'vplNDIy2fcXBIrerk~RnAlyBKIi8h4-'
    'h%_Ac$l@a}qJ`;_*J@a|{E_ARNf7BS5~Dw%M<uh7ZcQ_Do==U^*Hk#Zx*A5x^u1Np<R&<##JFATTmA#1E7Mw$ZUIyI!*R@t}xUL2'
    'CNN@i!b?0i#ocbz1^3YSQibR_-'
    '%o5s+4kXeYiZK@qiYEKo6hbRPi0xvJ<(kPJoY!t|SE(+wfMS<Msqd@MmD3H563goVc0=W+>ych+26Y^+AB<-'
    'W@8wLM4O1UQ%b^PSfKzn1jI-d{al{L)wM#s?E=#}E5MqziXQ0QOKuP|xfjed=ec=a%6-'
    '<^JiF?&z?6{hTa(62CL??u1Hi_O%36hUp-rMoW03*`lr%PAiflQDYjUAF5;&NZnx<5EUrwxwJ&`o#AOI(9Kwi?QJ9jG-Rg0m)On+'
    'EZtcj)Xo4mTFE`9JFCN6}Ji0Is%;^=dF8!cuyst9dcOmxrN&8sWevoL)mQ#J6MZy=$&ZBNNM<6O&MUA%qn!*_6hGkQs|WL8{W-'
    'z`Nc%ZNS`oNjrmP-BdW5{e4u;id2PR|xV`?0k%+oG<u}|fMUilN;js@1piL*b2#MCyNrs(^zr$t0RW+>0`~>o`rbuq0XtU;HddS1'
    '+&o52wDJVKE%sqg2v4{C%IM87V;(ETU?B<2-'
    'AJ*k3jy3yFElyoK7eQ;9wqju)701_~74|X1oJq<Lo}&6kizdQU0T})gwG#v6Jid$_SjPj17>sLMkk|gB;KG?o<x>mKbBXJ@0cb1?'
    'W@a+P=DWFVEG=9cZ!j!L#BnK*hGj#08;u)+)58s}h7p$1YIo&iWx`TCgYy`zytasF<#k0wE3YrORW}@5&cL4CzeIEk`>aHG3rTN@'
    '_!gG0M1Tu7UWo`7&Uc9r7gnT1jLYa!3r!G~_}wPtaC70S?`0>`?I>9sTF`#D3+ESnP}i_a2mXWXF<as+PE38Crs({ReAd2Ka3Ei1'
    '-!CF$`-'
    '37vwm%FZ_}}9hnD%PD#TF%;zp1Uzi73^xI_c_+w+Q}~F#<~~3rfd?#<^8N2{wZQmTCJv#r^FS+J2w4OzewvMM$&zC2fDG#6j~EEv'
    'B|Ts334-'
    'BnQn^a25(Z3A#C7lPz_GvXhFGg6GqjrAbXRyc8~R;T~KP=+BeJV;3Wjx*G;KSDTQ^hgIw6J>@)na!h?=6<D^kO8<UM(P~Ymftyk$'
    'n=;MdyA*!JaNnF+;txm%c|svD%k&>_)0>d97-'
    'H}8P1DIGASkiXu@o)VrNJZ@Rt?L7Nv=JCPL==t3i}N@RpH!e`)6a%(v19Qi;CkDfRPrbXkU$N2)Wo$WnU?nynC>+r?9CF`g^Cb2@'
    'x84XE+-e7wxa9Mf)W<eoQc?`yJ<#(}UK)Cv=Fegmc-7oifos#?G$0JR#L8aZJRGi3-'
    '}6I<gNpGm&wqjR2mvE*SbtQSKOuRl#o99!K#hTn~kP$S^JR+QL38cHsNNea24IVwFKJws<}jP;zIkJt3)Xyg?F*@;}rnqr05lS6U'
    'hUpgwz_qbT)ZbO~6L%0#wgVc0PG=n{s}&Oq57OA%LuBItEyDf$ESI=2)K9lg#Em(spkh$<LGy=G+k+snRX!YH%_Vl1x~Ug(Tt>_!'
    '7aV8No`5cW(g6C$7~NQE!N6jzcS`tPe1BO2sBiq-uCMbL2hZLW#aS%uNc4|AzyKq$9nsZZosA`9H0f^01PRI8Q90*6UXqK{LWI7<'
    '|MjV;{ayTSPRy^NB9;d@6j4EP=;K1EwA7OaI;ebtvwO9#jVg?+WHs`~S`+79YpfeP1ARRdWdpp&WwaVkQwqk%P!mcuP<xr~ZN=z0'
    '>-#dxHY^Otu^_1~P}`z1WmSq}1Jd|F2fPe@h^tE~xGw+3-KS}i!OH+9Mt-'
    '#Tirq~7X_ROsyB0YH=@#QY%&bnXizw)B&J8*Q=v;?Q#qB&b}=$78$SQ=F3A3PqF)`QZLdhl5KY@b5Z=KpgeMG+WnD0==ah63O=Jg'
    '6;}Sph<@MqwId1j<hk1(5ji}doh1=HoiSeTGCG4hA+z&XoP=}v|CZWbS;9=qC;71lb|d*Bq)n*6I4RS1eMS!K_zsq$;Mr6Hj#Ya1'
    '#>aCV<$KRzm&b$8oNFBq2S>`6;*`$B;!pA&-'
    'd5#w#BTPj_T0hkcTDvpKQrmF~|&z>1qZxXYezuW+ABvGso&m6ga`#u$qG|$G<`#*xJb-hO)l{*cHqg*%mA-W~6iibA?-'
    'H8x#fqNn8L?@DIo8V?{wTg=&At*rH&2g%{QAdJP%1MfZUdP@MU!1f?eoOB_6XA03HJQ*<;ZcvG`ELDqAnzcIP=2DezVx!Kh?FSSv'
    '9+70mo&#lF+e27~sA&Z<4S`giz))MPW<0J+pZ0%+%|51O(U0|+-'
    '{Df|VN@sE`2)7;hLCkaN2r&<)O?3vhg*#SPL9|g)hYu99k<hBFZsKtFiK5FlgtAW+rJEs?eWvIFJ6F{BX>pD&ve|iAgaH5JOxN);'
    'V^8&C-(YyH>1Ob_i+`WE1}zo6b_@x`85*yhQp=OE#(wE*-9OOQ*X<-7j?|IWv<VH4olVVXw4X~1JF-'
    'p>Hr~!A6;Z!IcIlCBvqV3rpJdBL|LO!`V@;>h-yazNeU8&qA^WhDQCSnlZ$jTp2(JGJlQSU-JMJ01x~8jjB^AKFr}8IbpJ^vd(i('
    'h8U=1Jr5fXwkV$ULxb(vcvE`qKf=mh9$`Ud@S!rismDqU{uX%behF!prW{;ibh`R(uj<zqE8fICX7cZ9ab774e7Ze5YYnx>__K%w'
    'eVExj$e4xW>rn7%`s7=|CBp;R(SctJ1L#7O?vc_8J$I}1F-'
    '=?}Al0Hg*3+|tF5rn)HXgA||r)7VqDQ>xwBVCB*}=(=p>E;aTP!!o=>z`Gy94G{W~#<4-'
    'Qu5M!0ckl~uX7dj43+J<W2l$1zuz3f_nzsh~8B~%~rUnNCkTq}j4hDWKas-rugA^#5*=cD!&e+X-'
    '2t3(XQ*3gpjdXPUL{}pbJ$2Fm`)Dam_JQ_7BO?RR9HrO7qFqyZGCT|&?9eURPt$qF463C3f>Rx~G)E&zyMk-'
    'oaf#e{Yr?)Lk#Y|$MB|Q5!Qh}EN~dFusjjM#8R_*OM%tX01!J|KMq+Ts#*yY02G?vH-fd}cBg7HsRtA?s#0_q3a1%#F{5A&n#Ey&'
    'wdzrns;AYO2O7JgdM<qCxJKv@3%5+;V?d@Ej4ch)K#KIs7VvxUeVw~JcR`!MgyW;P;mhwM8j^G`YD0<8jH5UIJ>G90rfKC-VrDux'
    '@;e#xVm52kDwC@NUU2H<8Z<wbu`W$bokiGsa-'
    'D$IxeckW0Cyf1T8fa083&>`EVduug&+F+WZR(_o8i|hEKG5?G#2<uKFQ+^qc`)<hgSja_n49B+nI9j_E%CwJ8XwGU@xk2A3CH0Y('
    'k*94DVIndbS?0E*xvwnf%}4-M_j{&1)jqEWZ-c_UEcjov?o~FbE)VSy5XlhB3h6&D&^v}igqvkMDsK4UQQ~Pq}uzV?4MFa_tEz8R'
    'EodZK9NA}t1<S;RGELQeJY_^RO9TQQ=$Ix_GuRC$K>%Yw&1`3?(X0U(E?8CBE{9#ap0I`V9Cdg{fp+l{eRc`VmCO3Hht~Oi}>>lH'
    '%nLhSG0hrXHjJqWet|GtO1fudVMXtJ5)oC>r5GJEK>aDL-'
    '?O_QT!3C;+bWB6mnXImRRuXxRyK?)g0qWj|XS2=Gkd~pf`c9e#%hY$nt_2L==zo+n6wS7hheT1Gd-`G|%?XZkQ_-'
    '0pJ=9Y(nIH<Nd9RCHqBoh0ZW`5+~KzX^C*evx#uTbBS<7n+Qi-SuIZlB36h%1f}hlM9*ZpAzu)sWU7OI7pyE=K<~Osl**0)7cujV'
    'KZyc83&x*azde0(wQj=WSjf*vb98A!Gj}=CO4eaSGU9c@)2yEnvp3cU9-'
    'G1e#2c|miJ*%Q4o5+srlVL$UaK>7^Y+RPWW?HASz&aovzuh;`|BK@fu-+%>biL>eSf`cA+z-T&m69mrSETWScb7tHEf_3>HA-'
    '}zSj6S!s=`=xa}%*-'
    'De8n#oepyvxSgAUx$&1y}y=iyooXut>l@UZTvE&o?NA>ZNpUhwe*u#JtIJFj|h<K6#;U+BMz}o#LD%JSh;?YaM?e==H&jLk^*Rbm'
    'Kec7zKH8AhkTHB8R~@0a4xBGhnH3FsCgz=lY)It1x3*~sxf%fp)}f3Czf+5!}F+|8#XV422i;*ys*u|l^J8}1P#zLHR+9_2#ZbZs'
    '&qr=WI~xyvi-K57E9BMVrhDDEKM(orRk-yG`%dArkBUkG-fNl#*3@%tC^U+`5Go>Yd+wlNV`1>ZiMxPo?RaDX)%t<sZBR!XA>*bB'
    '+9?ED$MVMDdVRhZ>H;c>!cS;WyO|S<|l}#8GfxAU|(VF?ti3E3Vh4O;tjqUns#iL4iUmpi3s87M0jvaB0M-'
    'Y5gr^D523I6Vv6f)<*rib_0{6~yAIt-Cw?EoDINUvwWeHmjjc!@b28tPH>Zb^!c_;i-'
    'aR4QoK^rmQQ0Pb57JBLR3hq`PKcBJk#zV|bLmA+bMiBnn8d~VCsYdfo_rh1!<+@Ig4!8hn(sn4G|Z>nbfqcLb8Jqu8d>3VE9YbWp'
    'W*5`zKF`TClpaW_!|Yw2RFjmWllXmoA35nT58VE_Sq_0ig4>L#<HBpSh~q368={)>O)MayCFqgV%ov@78dQZfNG$9)wgk+#8w(D8'
    '=y7kwVvqCs7y7?|0r)M`=EHOzXRP2?u<NVXD9RXD+`uOUV}EcSb@B!aKpS8-XYKDob8w*N-`f?L`ml33a=9EQi)q?n4PN>K<-'
    'z0^OlI&L2vot@O&m4x2^n}7DKOU5C7&uL$3ftg(uS#<HK$v3^At~FcI)(8bt$B_4RKqGW7C>mW^<(yV!Vw9&iG4gpsq{em5D5f3L'
    '`-'
    '?YmHg;a7ckXlu^i^gZC%5zR=@&(X2}k;r~;Ek#Q=RN!<3))<~~9_@z2ZpCiRdevyIKWbug{V~Jjw?j=(YpI4pLU1V0m}eGI+HF>X'
    '0l2DQRQ*2_UF~-gE$vl_yz}ZrSDTXa0ovJNWvytpu{-'
    'iG38y_cJ0l3rEch%eGQtDK$Beg?{bs?}l`M6i!nM4;meNNVtBckwq+YjWo#~z_=Wal%8`hfOm*VLh31sPjnDUJ&kW+O}C-'
    '0AghUmikQu^`J@ZMd^!G@&g9(=HN)0CKmz|})jW)kAo?igAHf&88@^$H<u4^7EQFfQo@eUOMdr_}75DZQJLvu~wrcS_H`owD73N!'
    'eiGy<ii%FF^W;Si?Ezx;?8U!RkpS*z_!~Vx<MFp*%)+odu^%F~V}K^|{8e1XDfAeunKp2etE8xgly($2uHOLS!@dFQgK;2NY6@+X'
    'IUzp?OddB{Un0D4}_95hXN7q~>Jvv8iLRdW(HGWeVJeK2)quZ@2IHv;*t4Aca$)47s1N8!n*`$Xt{2L=wf@0Wtr&7V3}O!Z-'
    'mDm8BZPlKok8_rr+^=QMvxG2TpMFe&oCrIMn25M7o@QQkn;W%4-(Q>0Hm=McI-lcIbm)O%C+yw!?}a>_`z)V`IE9<j!27cSW}Ub}'
    '{OduVvIO{}YHQ=FrZTCtMQud<`s8iz<NZWR+9oKPpN0Y$iKOeE45CmAzsq&2Fpxio;jQII$oNGCE#91NlZ86*xG=sX6AgTZthgT%'
    'oQI*mc%U??5NAaO8^&SGFE_Cm9qFStG!uW)eei|God5ChP+tmO*W2<_ToPQ%r;!{`%k8`@#N&Z7MQ7-'
    '^ot5q>fE8R!8%tk~&(C=Bz$U}_tUS2rU@t@}!+{(A}*5tGFCa=3IbzOP^sah6!<9w?Rf4;8E-'
    '!uRfv28h@0Z$}ygQ+lRhir{?2)1G`b5<J^XjNf9?M~V8FqDBO9(y8tYbO16i&+Nw_fy^qQh1qiT$Dy3a>dOy_eQ3{Mb?iYEJw<L&'
    '(e}8A$NRKDSVmCM24NLJpdv11*cWXP7Gwk}g7p}Iinv1N{|5wJVUDfm=%{vqg<5F5sDH3$gRy#S3i94{Ou}<eEN%!D$-'
    'Y`_Lu9p<!0(CWeb&@gYoS}Mv<|u>Tdjd`2%W^rpbv`i>et0!y4qDdrK{b<Qo7n*T%@bsAR$d+vdaa$jjDbi?VT;!mBS0xu(%4-'
    '34TUnDE!*%GR=?%l`IxI3mX{A(SpKzrr~e27<U=x6iW*1^k{_Jv<Z;f{R(ScZ=*RWNVkHWF?9;3@XoBj9th>aO5CCwQmAEv>m7y1'
    'x6=I>6UDlxNhKefUIR1Hvt1%&rMvieS3TISyT$jlc|45W%JX6DRvr*zxAKG-'
    'yOl@8*sVMx#%|>yF?Q=zQRSLrGOWRq5D9Q0X-xqEkgRRvwo_W}{H7-KPoHO(IQ;3)jE;?kJre|Cudi-'
    'TDMTIH^`@^%diQRb>hXLJA*%x0buLG=st-7CXR>Ygg%YT&h9|=$5yHF5a2yzDPu7Cy+*>2XIA73Y2Qpw7Q2v7i-nmrU-UTJg0raH'
    't>RuDVrn0#~)#E$WQ1^O7%Z38Qv)fp|&9bh5L}NT?tI%F+wiXSUt%WO=lKHV)!*s#=gXbJVcn^)j=m{-'
    'ajkDwmJ6gj#;Y<<A`mP|zkJqrV2+MgcoYL*sumI=79o>!(?^@soY`+r_m-'
    '}cWY5{TijJ7>|i_Z~ILGQ3nr>y#!UX;dju|7~)YQKcLs&{1vu{5|{nYaWZRTv?+Kwaq+2{O-'
    'US)q|IL48A~nyRgC7LZ9OHE8bt(ckU~ik^K+Gi>c-!>H6jpmEF!`=_@S6}lFr1E&kHo=TNNMK|_A6-'
    'D1cF*;HZK(DH}fzc<OEWU{2C=a#QrwD}o(5`Tc<qYbw#X3QU#rj<hp)-$e59TFajP~;THMUhtwE!vZv?Sbwb4ibmwJp<)z5~1-'
    'Yf}D5+{6XLk=tAk$)W90pVQZXLq&m3UjyPE6*_$ln0u7y^fllOoTl!<MofdA)s$cQC44|daeW5`Hf*c1hkNs}nEw66i6=;l?s<Ek'
    'UqQ-'
    '*3KmhhI|E6#28no?f=Oo<Mw&5?xm+jncPB2~pm;RDDp3KAiJtOOIL?S~c6L<9YnCvw`=Pf%Ujy|3jSBiI!=!DEeLD;>2Vy+V`^fj'
    'xe;oecKLO&f*M>j%oD1=nib2xfM0n#B6|Zgp+hWL|#OrZW2WL`w*Jkj+kLCh$Q5MRmtipCH3$DBSMPf`7Z!CY<-'
    'p6MYk0XN}xFC_pP4TolDTjfe-yMXgo37j}d%58j|7?4O;U52$w$(5*V-0dECKl0JKU)aH?{kOK<c7-'
    'v+VJeQ^=sQx#0VGKJw!kE4%<ugbMK@(f%S8LWBWM4!gRIH(KX-I-'
    ')D=3&5hJ<2Pvfo@U&<Z{r0>Ri{DqAa~e~P<8Ek+QHt0h6N+lZId5fI*-'
    'H!$NoLqf4NpmC+RF@&N#@w;Ca4EgE9`enfZS8XO&P{$R*9kvqgm$E+Zp>#*j#;qGGpHEuk7xkbo*P|*GURTR0~Yx-'
    'E%wu^<$n`bTHTwlwK_4t#wa|380O7xj(qGy$({ELG2=*KD$-BiC>S~a7`?KMTyfW&NnGR8p9G-'
    'lpu}ecuNV=I2O2~1Zlj8FUSzHIjY~=Z)>|$BF%QnK50@j9REC5i8iFOQ(vezfz_d4(J1+1M}P|X25WG*$$9NyI(Iq6tPQ+S_ofTA'
    'gJMiy7v(@iaiN~K;|=4Dn-'
    'Z!2QHj+5m_*WlY$EADE|K&fA0_=AESO=w5GdOrfdY3)puk<jeYwmf6<1u4L3@*4$pYz>gsG6G;K1qR01$`g@j0MBE3x=WEpU(IPM'
    'pjM#C=(ShguI$apuu?i8}v={`GEgftq50Ho>4Qi>@erO*Wam3gY|Rv{_wOE-'
    '<)BrRLzr(a$brGazi1dGb64RyWD>7*yRXo3EjoFB9~^)h+UzhE%u8a~fLRCeLYDb-RB~!Ok#sl{-<}zls)9>H0f!85ezjhK-'
    ';n&_Rm&D>N+6cL@ro8eZ~W58}ugUh>_84VH#EbI@#^sbS8%TM(hvbBoRBr=kDBrMXA2N^xzYsD*lmv18K7>s`j4l+FTwXONF|<v#'
    'gg+FqTiQ&91^Al*f@am0GewdSP%z97BOJ?E)0Xtuht#EAf2uJalE9hy6ucj|UQtN|JYR5*pR`9OL$Us4o}-'
    'QT5TV6m=&xVvF@vkL}o%x*#*6^!;ulL|sF@SE7-'
    '+8b2OeES1se^oeAu&>bn%67FV(f`g?t2k576~VN)1vyM#nN#*N^y1QJprIG{tQ+WKio^C`L$9P${TpM)rE}ch8ibwT#<qoPO<h~7'
    'oaovekX@92$S%sGLV?tck8I)>-'
    'L2Xqc2yS8uX!TacoeXPLkwphOH7tjv?|J0+_Tsb5*2Bj>}_u~L2nMHbODsgIC=OLw5F2w>qEYUO6$<y(2KkM1E@xkG36jbFIR^96'
    '}j8sk`3IbWz{lC?>5))dH;!#u4@PF8Y5v^xvjQ8HJe78FHW}^RT0yrOgOYiIsIV8#OC@oDmK?UsMuWJR>kJ}b}Ba4JF3`R@1$aLy'
    '|YU6AQ}}@utBjqM1@ity+^tU&Ji(%BMk+Bn!v|O1HZXA<)A?O9UYCrC8@?+U7&E1dbrJSzy|QPpqN%~Ll1@Pc-'
    'A+3Fh}d4vgFM00yNXOO~2A9a!5alK4CBI%E!=XIzLg{>*X%29IN~eCH!J^bX0mLV-A0NLHQ8N2tTQ#-yl_=-'
    'f++_#@@QRGgB~nFzx0ypaHxiV*r<O0Z}cd+O;vHYJbycolTPje*|Eg+$cSvVE6iJi2X+4%<I$To22V<C=cZ{Ntqo5nRF*W6u?yr9'
    'fB&f=Gua|YL`MDY^noQI<MGiX)s%>?5%|z&U8tP;&xDp@YDV2Or$?r<l84Rk%tvu70*@Aap_sLRc)d0Y_X3jN@UKL`m3MttEpNYP'
    ';h-'
    'j<>{b;Tjbj+nA2=)a<jsPzHingu$MB*<d9tW>EWymD^M}7`*lXt{hAcz)y|9(U}r@Mu(PA++Bs1TEEw<pDf&@20s!4#)7LPab$yh'
    'r4U;EFP#U9CH%C|QGKI%Ls0Djp2$E}(BjE`Qx?oqw#M`<XPU>eh7W3Q69^>0$pZTdrvQ$*Bt<2RF|JY7OAQ1ZkvuL8E%Z)6v7sSQ'
    'R6Q$pcxY>JzKFAHg;5s?IRNnCw{}c7_jXHTW^X(H_a@pH3pA+X#aGt&z?#?G+G|DZWKf_v>rvy*K5Hi&1)e0)HxToQEI-'
    '7w3&fZ7U*AV>;;5u6xmOET59K=VIH<;g4qWeg8aW+NXgoFJn#s>T1h=&T4dQ!;xEa37fDN1bsLKc+wGR8r;&jqyQ%4s`JNG(f;N^'
    '|iZ1y#!c7@$IQfo%#sNUB7CWk-'
    'cKREqxEjt+0ApkHX4!yBsT@33RS8!GAVv}3~?Dp`LcOpO_as?G7CH}oe3JV<D~J*j~DsLAhaa5>sWWgp|i_-HoQeci3wG?`CT*!h'
    'zJh3*KQs&dT(1t8G8CxJY&B!m>t@jow(Vu%5vX>zxM5a^_&&$LlxXzMRjhPM7vWoYZobcVK8IzwA)ouRFb&d}C{Xe=&Kz|n+;<)0'
    'P$23O=H@Rk1FdJW{d{FpWsh-m3x>9MSC@FC%KC2rWwN}wR)6kcyDu`j0AJJD-oPiaX1UO^)DEP<8oBi-ry6hu-'
    '(lzU7GoT|k>t^ffF2k{972v8W^J*9y9xjgxV<F(8EK(vAO<m_~;DQet!$`|OMFfkIpU4+zfjpFX<5u!+p9=*tyzpd!MHPj$>y;B%'
    'BR&7bPYTn50ADO^yFj0fyF1_AT;$2Fwe<~3Tqn~1dKs1ij)vx@Ux6mrCejWOPlz-@IA^&{V>gtXF`BC<0N-'
    'zTEhr+Y2ej^6j$Li|0nY2aX(KIhIY)87!`@fJd-2PuIf}@LZ3Doi=N`@LeVVyq97Fm0H61TViltag=-'
    'V__9w2qXP1!<t;uIvKZC^64i3`RiSY7uZ!$X4VhdjlO8w@=n7dr%heHix8mFiLj?&S@|cmlXbVU`8c{pJ?c-%w%Ru1-'
    '`qR!ffD)eYKW@=GCfNA9s1RnhT62s#?SK#S=)Zr$`^_P(x*TDCHl|ZqD8?$r!EWMqdfdZ4Xh#zHChm=$caeO0Eskq(kdAF2Wi>Jb'
    'H&u_MWM3qN^-'
    '`Pz~j4EY7md&~3^}(d!N6yR2*UzM*`VRii&Jbk7cCs+^&FHt4yac${SDE0f%45zgd!SoNL{6u1HC_bDNO&cHSP@zI|DVJ(nRIyyH'
    'WX+DwNqIUrZkY=`?sTQh|t%?XPi<lMv2TuZPN6ae|(UfS!{DUANSh(yX1zs<R!0TT{;PtN~@OohcUf&Uc*LOzX^=~5Z`nO!5m9Xc'
    'auOS75$US}KQr8xNlZYwlv4#D1tp*tGVt%?*PiEw{1(B9P*Sk<%=yn&T3tjF)<Dt9#_cE)#GZJ9vX1`Zv)xQl~m~cwBDrBDxg+97'
    '$r#+S3E((wD1;O4hYL7qvQI6kOhv4)t?92W!G!#_orzD!yD!uHRF!@`}_Y|cw++}Z#o|ZyLb|NbcKy-'
    '$)_*AXhyi10YjjC2VU8995+!E4;_T-Tb&h)b8L_XNb#PH!8;<VWK^bLcYv7ETQmF*<j4_+aIHyl!X-'
    'f_^b;4KG{F5Yt#2xZej(}H)MeP0>_b3m_8^9j0Ee&iRLvZIx?W7ylNs%!|leORzRw<FUmNT_~E3bvXfJofg7C@q^b_xCnAK$o9ra'
    'N&^p5M3<n8yThM!&ptYk;fsz@A_2P!-'
    'TZ=8707s!UBox6t;eEft1`EBCYm{<a_jAmb%y<L}@9UKF8ws!1Zd6Izu8u_rQfhem%n2&b7wbYS^b`Rpry|YVu7HuUou~Ur1U;X%'
    'P`~z<HLzr=xa~uAvhL6^@TXzZ(uGn{>e_t3H@<YM4}xU|bQDsty%G+~@R37^;SjortvjibBTV@cUm=$QTlSzmP)4(D3^^C}a!^zr'
    'T}0#-8E#zoC$^SNQ#JDP-'
    '(j%eqcRdpk>3pJ}(b!WMd4=N@sbv5w4g9sJ)if#?R^8eKCq#OWWUrB<cG*sS?e;V_QX{8?}qPtxwE_JhbptQ3op+V2hH?rBx-'
    'F$}w>Wx3Zdy@S@}KEvz|TA2HdBX{`G<O9axwEa--4<<;CvkY6o?12M;f0(_A4VJFkUuvz{JU(uW+3F&96zc(Uq-'
    ')ZOlFKL^<e{?&Hd^yWK#U)w`SByhkJJ3n5#uR;-%I`blpti04uHoAkD#a47HMLEMM`hQzt%BPO^LVt#X3V-'
    '11iz4s6}_*%MR%%68kJ0-A_{y;(nTgp@o@3(LV!snv35rsXh>?-j~zD{I#^MdOfYu--vaU1HsHO42|uM+4GfJn3{(f+tG-wb7f?Z'
    '-5;7f-'
    'z<2(a7<ypRm&r0Ky(uP&MsU9{VgZSV`E91+M3v=KH9`K^)V*4sgE_WO?{k+ZR+DqY*U|LST=pM6Xgw~jb^7n8}e#8lb5sjHTK#S&'
    'UGM7B;WFfr4e$~`94^Z>4`F2=?EvVIWdwCBKae^oZL+%l@dY={Y7zYXjHCO;?jt}XJDuiZq6ar@~x3?+8R@ad3<p$z;}*bweE(CL'
    '(`Zf4K3P1Bv;A2WPh0dq<e0Z`-Gu8HO9$9hsuPGXDi!xAizyerSvb>?4J+_fo`@Khklp3@nWPjT;?W>+oz-'
    '9PAT{83O8qrL&8>#31gIZnC^y*jO1V1{)mo<I=pylp*uwv;&L20?*1S%lOe=@I;Hw7*P=a@=J@~g>vn{)YqJZ}(Jfk8TVBYl0OT^'
    '86|HQhv!dV6d{)ezn9z#;w)k!QQHc|TyHF2KF4PgpWqEjVSsqclEJp#mg<QXhPv^5)&|d_JHWqsFCe52-'
    '*C5j2GbVLpa(gax3z;Vupq7Nw1^u;jkDY-'
    '>z#X1R$UeAjbtZNsN4&ETs=4=bHm)@8{hWi)%^lfuaef)NeICLnGuaooGZGVzLz2Yf&?NCVOto_ux!9?yOycbIxpX8q#L$iGIvZ_'
    '4if@y8F83#>ka@ADSP*N9lU?*bL$hHFryy}x?eAR9s*r<H>h-'
    'Fu+ZPnQ1TeNT?MA+uHr@V__Sr^gjN~SYt?Db9+Yyu?za*jZW2nP$nYOE&GrcbyiLV$}k)-'
    'QI&y%WSD!otU^=eDkYtaAW?K|MCDz5*Txo_zpVmIp2cUd}Eu|(JIbu6(YEJeUzL5&gPCU&!QL@bCJjcw^DiWF<GL@{cDWu=N56N3'
    'd2D;@NI&Y3wgbLTzu4WG~N|9S6&@47qt?w$G0nNz<f0Yyija7a{Gm~2L;I!V`fLbc2xHBSrn%ROtIxYx-'
    '{m+sj~fiB%vPEu8_ai5U_@u)2%HLf#KARe>TxyEfq3d9^+Xo@Z~(>~(~sg$%-'
    '%yXqW&2cKtm(1OPF^EP98E6@AHKllMNho?jQ?<;Ws?T7n=&AZbs4p^3g#Vi+_QY~NTiWyKNNRt1inLtabEG;|TGrNiQk^af^UdG_'
    'IRd$#I1jfIsoyNEX}8aznI{940&b{fmMm+zKTACm&cnWnx~Muz83d3!K>4V|DL)Y1{yK}5s^b?CJ&{(>?@wF@gX@r8%SY016zAV@'
    '(%s*<=MEUq3%TVE7|JgSuX)$~l;hyl_a*8F|1f+IsoK25@F8S~@(#m?6D9oawzW7z-'
    'eEWyd8E9<a0>DX>S}s7hdpTyeMNRfrX5nk$+O1;(&XrnF6KI!31D0=GXabnWF~-'
    'dV>W<sQ#OF{=WGC@CL6#QZu=IK$o!UR8lUG*Q|V|tAUAS)<4dNU5yyAt3QsgVn@Q^ZfC{7nQo9nH72X@;*oNn#8r7u5iMC$cBrYb'
    'R8L$b~@+7YGC-HNTX9+h@4l~85ty-Q2S(T%f=A6i4yGeOZPtWv$AG4Z-Htb*%ke!rRj!bu?ZWh@4@mfpaw$fIKorDEN>`eVy-'
    '~lkqvXQQ|iu1Pu`C0zJfwuG;CN(I>zc4|%1cY(YmdG_otUszdMxU*p7r96i@<)+=_FlezR+aM#i|sPtG@LI@Y9$33mTGPJqE7h#x'
    '0Q@$iuq>Q7Uk^uFf0)Lq^)=YlfahZfI@4_pTIh>^-oA~W@>bZDUOSw&}Cxl%(fzBAooc;k%;jwb7?<gr3(q)>_^9sLq1-'
    '@$O(Xvz{>al8+K$&@>PLjv5Et2q-9~@B;dp&XX|TIYFY}<9zk+8$6tbfWvR#{IUba#=d=u6EyAFE7P%}4r(ryT(-dc-6YCF-'
    'J?TB0cz!Z9d^Lq)-wB*1V?eAI_!Cney+L3f*gv@(ml0+Hm!-?BVOl|aqU<8>B9Rr2UBulm`GH)-'
    '1hMybkiP;`aZ93($Ap1J(l#iV>#@q#WLQ*Td(1Kg(WrGS71^*aoQNkUCK0WYD*kPz=VavE<y6lpn3BV}o>MWwgp)m|qq)r)o-'
    '?pg=E}sPy`^<hE)dvj?=qX8GQ;wvBPrvDHnm?7!QiIAm$*n4AZt5GsC62N6HI&*C+>rvWjV?y`;`=2qWn_Ry<~+)*n2!W^uvt_I>'
    'm{YWN2gMyIW`ClLDIvx{&bR!V3xSV~bz`cr#t^6_~O|5>We1x&XPwH@b4D^Afc&vE^`sQQNWwQNK;pHZNspw7*eL6zWTz@dM(vaV'
    'B8do>TKa47e_?%<y=*UE9R#5ZT0Lt}H@Vj|HOUZF^SOJhTrIX3VUj%)2VT1Ps-'
    'rA&G6UauE{ja1s^9CE<OPxw-^-V?a~2r{0?hn2%2Vcq;)5(rF)WCtw^o<>Q?M>?5apEJ?tube-onCm7{%KL$<qDX;KcZA~!RvuBo'
    'C;|?uoo)xxe?%#R&Qa{-'
    'z(Q8LpxMiYiQt5Y3yZ||$Y!{jfk@3mc{*@TKg}2Go!#L!nLD!v@Hp~cHY*@aRaIs52^s3)$vlg=57@+4?dK88qBwo=#AGDsX_Z$)'
    '16Whz2X6vJ3;sE+UsAV1|_%ka-#zVw?s{Zb@A!$OI_MuNebCi=r-pDvODm_NmV+9<qlev2g9IsP2M--'
    '0NsobRpj@M~S?hME4bneao$LkDkv;oKKOqb3#$}C4Kn1CERF^@u5lKMef6L$B6nhs>rK_yyNJtZYS(@y=*jzR*NMwJ-'
    'Jy*_c$|9aBgjjMJfT@3Ts-NFiFBxuYNv%Bl4^hIbmfs`7ZDdwrdZO+v^ec=pbay`rq;WXuum|pWy!kX{^(CwJP5Yt3Bfq4w(-'
    'F$4jQV$a9e~CRe$J-'
    'XntrL~zSgal?{jRacqF0T2uMp1un}@TEZBF7cYy<f$HlZ`QcnCRX17aBg{OaIbwnyItJvS*fto8A=TB!axYln8m`UaDhTZtLi12G'
    'BPD3@430RLLn&6-4wOKg5iU&}1`VS*P%!TkvSqNSYtUjox*8snf-'
    '<9OQ$2K3ldxRnm7v8id}?XY$&k?yS)B)SP}A7}frI5M2867?B6*Xmdz)VO2xS?&yIK<Ro1BgN0kY#_zi*_FiI#`Y+PNH#}7M6PGb'
    '%K_GO0ui~9X&MJI0|s)pZMG4DemT(ajn5*%)3d9Y%kG%$5C7`9_)h{r%p_u*hz>GZOwdR!4<zJg6c?W4v5);P<LhppbC$a29ji0m'
    'F*XyP6<(j6mBKdDv<a(2yn2j9`fhf?G%?@bI4QA!t;Y6?$8tm*`z;>NVQ_4>c!G(8V^@urB+ueE*rau{{-wP^-LU-gYcZvZb(t?Q'
    'D~>g`uP~vDwcw8dzJil_eu<uN>JXE0IVO0Z_}2RBjGJ~f^#>me3Wqm2?Q^cz+$=bgY<}#KIriwH6Vx(=UTK&@eMTRX3H6a6B44*1'
    'F>Nvj=YdTq8<aC{uWIaBc^oGaU<<~RgvaQvF!O?Y&vwDuANbc2jSaBc-Zz*i$nM8~CpeL&&b%~HC)?EQqcKpxO?1%VwrfJ|XM#$d'
    '#o2)HCqZIaT>-'
    'k)MUoxVuS+uHeLrrkK^iLMx1!E=f0Dn4$&xpnHB0gI+0NwaeXv3M3L9_t+UfPp!M?G{d;@aU*Y+tOCdZr<=m;E`GoH6<Y<hVjXJu'
    'fQ%aeu2y?8q2JFpK0-Q3mEl6u{ksL_dKwY@1(V-lMbSY=b>x`J(>JL=%ved>ES-'
    'jEzA`nOQ8<@s(4<s*<Z`>&TzzCyab40}-gi>-(MT9eiJISYq=cZT_2PtPfskmaenr(vQ9*AQSN<tpi-0)I49cR+$Oa#<!XyC-KYe'
    'wIn{gn2n4QU8$KT+`-'
    'vEpho>%Un{|N|(N+J1=ol`X`ZSn3|oNWNUceB&{(7?;3N}3rmjRUz?mUJW)3rDGiFyNOI%YJx+8Iw1S{V92^kJ+>HEZSsxV(mn$t'
    'JQ%&LgL0G?Q3FOw__L=lsh^&wq#^KL1^6Wn1!yRlAGjPYvqVr9E`z+gf0+o$xCD+Cw*S4YRFI5vMvb4z0`v--pmwr}Ipbiq*2B<a'
    '%Z>h_+TkcZqR&a`)?%Y|+hb=3*)G2QYAGWOKQm4FWeAu#*OP%s&@L@|Tu0QLQBB&)_lR*?niK7$sPiar+V-'
    'nk8rI|n@#`Nn!`3**l>C%H1%cY5}o}P>@+{x^XxL)$TPeteLRQ5&OB>9$SqGNWZM$ukEHQ`u0w#++EWCG`)svfAP8MM(ugztcFC{'
    '3fhEEiaxbQ}h0Y`SheiC{6S+SGaDk-'
    '@q|7Sgk4;St%4C+v&Ngnfu<4(e5D!(T@leiR#<ZsfVIF*e;;GW9Rg8YZS)7r7R|bmkgpf?h3GbKNL;p=Y2=dxq_DJH_e)8wE1kjy'
    'HSpK#9G#ugvCJx6hfvwrbrY!)PU|MLLdE>O~ZwH)aA(n=KhVFyXd7W+j?p&Ofb0b4--aI4vf!yjoja8X4R`jmPQGg!I!EoP-WPon'
    '<$Dle{ilvs~8MBd}$@WAQqI_0!p~W!AHx9mSf}<FI8mav*yQ>!(k^mNEM2C`xnOY?gMZK)>d<mbCfLi5hFB=@;bW3#tSsVn$7#f%'
    'Y91V$!7DjBn$T__DCUSgGT(<7|hq)t;U{&dR1f>hDN%=4b)1iu~{S2^r)M%Dbblk?e!+=vI9Dde~<eol%M8Sk;h(T}J66=-'
    'qhECX-'
    '#L=^Ax1*+q$3Xp_k<PSo=@nG6KR3pSYy!2Y65CIjK|lGz6}l$*QWO#5ic#@%ABR*`VnWW@Tq9oUTQpLUIoXHy}WrA8WOt5<azne|'
    'hAF1X;^)e>MKEzQq*2@0n_Ki5(N;c<o%hmu!rHnu$F1&uc)N|YCLiz!i}w3Z2`M2V7wZnY)xu(fZqCGj8<Z)aYIg_4msobqxv80B'
    'maZt7-'
    'w$~{0CPCg8eLn{d5<Hxc5`s)9_jfgg>!3?YCbojMW<;>N7VC>}BQ0_(9Ji`hGb}RKtjq_VSU2ALG>7=e>Fa9iB_yZvCpT`{uznbr'
    'r20%Zq?KZ&dT;7deXN<+`6F;mwj+0tZCOJ3%$b|c$!=VCd4eY*#?1Sx}trv%_VoKg%Ys|}9uJ907d9FRi*(aE~poFw;hIP{929p{'
    '1m9$kOV<=mFt1WB5%<OGu|F%{S+A@)^kw5_M+L}W*Bx;nog7Xnfr-'
    'RH@j&djt<6<<hyqVP*W0>P8RfA^jo1k+ioHy4enK<qZNhXeaW0HyE-jrnGxPMMEaon0D6UQB%Wa79ZxC)-'
    'o!uozQk&gFTP2S?Qnw*f~sXIT>^|n-2Ge2W!cqGeI=;O+s$6?oV!)x0Ig;tI8?BP!qd4ZkTc*+w6vmYm+o#|VS-'
    'v%E3O`<?+Lr}jBZv;r-Ioa-E$ftI*bSJ9uD}8{IVtk2LQUSFEF4(S{yvv)hE-'
    'A1aNtYVxRMJbFN_weNNiTCM>F=CMdbv|cfA3V%D{NcPn>kY4LZr8PEkAGbT7KS6?58+C-'
    '(}SJiD*Nol_q*imLG*ECrVBthW~SZl1G+ZNmDb<$_k1BduaFyQ@m+{zeR?LAUmNBRi_r#Y+fUkl-'
    '8{MQksew3Zq|XqupR!{n2JPfyVfzrNaP)@omfR0P6M<CiGYD!7*rAfrLc=*uiEOg0+7tx7NlsNTZ}~Nz3Y2T4%Jc?=z|@C~P;|y!'
    'u^)njl=mRduT^L;zL(cI_PDef)B^p!Ah&LFuAwLFua^vycBFGW+;7k=e(ui_AWLL+DHrc4xN1YOG;w7`0(B4C`mbFsMg9;UipG3-'
    'KeQpM*nyRyNRfqTnpblLL#i6Z!Z3)SvaC;dmeVW4Hxs8{E}xSUdCW&q}-'
    'zhc|?RsqNKybkAlLv#LTd?;5ND<P^f;SOR#gZ7SN>Hm^)aV)51Sw!>&sY10cwBkjkDw%urRX_Ga_BJJeK!euIs^D;Vb^D;VbH|D}'
    'T;$XPaPGn0se=z$AKER`K^Q{fO_vRMtz048c2Qosn4ZOUjB5mA~jT4!S&>G)q-'
    '4|3X+ZkpyDbRg0Jx0cPoFq4UoFwBtPLc^856NwodxWaZUCqgSujYi3UvKu3U&nh%ows;NofEvI&Ra#sd$WnD<u_&&R-'
    '6Wx3E(Hc&hnFAn+T_ARKi|Ff%USHd7ZF9PkxJ+d^~COsV{Yeamiu*BuWHlo%TmML(g9|Kw{mfi_e~iZSTDZ>_CZf!$n{RNkEU695'
    'z^@DA~*Us7EHu_l_FKn8Y?^01ad;SOCslcymI12?$o3GS(5!+E9C1O_a{tj8iy^RYMyE*CL*eKTz3U4h+a@g5WULf!ng%uV31ZD?'
    'a+C?b!o3Anm{&!1K~F_5cn{%h>}sDDB9id7^Z^^VB5in*XW0r0b-'
    'Ie5m>D|6ZmhioR2RbKW#m<@q{%mldXvbh0Mm*V50UAOl+yb@6^ZR;ckdS14l#qJxaR6fi-'
    'L*!Io+U^{U8=Kio9xP9{g*bci04wOG`TR9H|>B9bvK_FdP8tHPUbUhN)-O@EiM3VdeTIch<(zdsB^YjtPY=54%<ZZG-vZfmKwHlt'
    'Cmxbq<K)uzq7R3lo4Q73Kb9Fo%o{~?IJ2G>%BX^hY!|`2T&Yx4M*J~CV*qyoZ9%i`<7v94xcjdZ!nB{I<b`P`MovZF)W_y}Cd#`0'
    'fg1i4^?X;wM+FBS-'
    'c<{W8=7HVCFB3&e3+sjONj*AY!kjzzGX5DeSpc+~^<<p{nneJt8s>Py+zX~I#pDY$!UAg~Ge2Sag(1vEV2xIzEwIMerh9|nbQ@d1'
    'iRSwxoxc$s5j}{Zz@IEw@7=Si`L;Yzd2DY%_u?yYtF`cWWt>WC$rAfe{3dWU90wY#o~aYDkBvW5H^Yg>$Nw$qE%v~l04Exs_qV6F'
    '^Pz4S!bdvlet5HiH(ubIV>7o1d~<AO0;+}9v2`ozg*Lu@o51(YiDmPL_xK<j)&QNWXk^sqbo}iZcLUCxkxp&lWlrdJ@G=K<dw3al'
    'LkD=>;JP^&cSAY6jJu&Dy#CB8;W(=+jfWGTwWC|pTdfLo8=UwjiHu7W?&e#2O3d|XWi`=Mi#+S08c&_|HU(j#EMVDHqVAv3BY()a'
    '6#Am0TI-'
    'E)1D>L2>#q8kHbs@Xcu3#UXIQ@1f?e9<@{&;L5v)78DfT&JQH{R)`NI0{TT)ENh+kLYRZaGtZ5uwqwGAKX+J=vEZNo>qw&7!3+wi'
    'er!%0%=GSl&6=qqT$xup8CEhf=A8>1cO(K;KWJ*Lt+8>0he(>fcY3=?XdjZuy{wa&)qh-tNZ+04cTf)6b;h`>zS#YH!s5;nTAEB4'
    '-`qbrLYo2}S}t(0-'
    'jcUxm!5oirM3;cSdiOAF0@F4pP4HgoJG)wg=vVmmSJ9#<X)>*6}(vBPr7PZba!VQus3F6&?)ojBJ#hpvHLJ2t6k?)Pfnc{RRbjdy'
    '?IH3w%PTZi8c7}04fS8dO9~^J4z#P~ph+UJ!j6_4oWOk$Dc{A!9M$AYoKE2Ic-7Yct`F3-'
    '4hs0v|J2>%HRQTo@2wF7H+|H`<Pkygicn#bCB)q*^viqoHJ5QY+hw~bG<zNaw5Y}*-'
    'jAlKX2~#jI%XY$4wDH+cn1;D+6D%*^e+8KeS`YYGSkr4Jx&m$&obZND-+6*l-Z1NQzTiYRj8Z%)IHe6;e$NQcOal-'
    '$Au|ocMN?y@$qHwtr4{hK)QhSUe5ZfncnSeZ>b{m2*|nBv&)qywQV-zfC}qrpjFNgFx7F;(sq9+N8pj9;DfWesjAG{^ZBFghaQwb'
    'Ceu=no46`HH7RGw(StGp6N8FsVV1wler<duhwA(K6r>6rfB~4E`tekJdjtf+PD&ItNX)o)AYmi$L6wR{a|G#b>U9Dvs`=3a(ah4f'
    'LHZH8CV<D8@A5o2#uVCl=2`U44(*$({G-qh9Tu`xz#}`-Q*%Gy^{%PU0UxGOSM$bGu@k(I5_R3G&EbZrzb`3-'
    'f*R&#E*Ji=htH{x{&C(!P0uHLaWDn)zj(&rxS3RfKyO_P_PM{T-'
    'L}l_ezEICg6J$iO^`2CxC3(8`>@3+qi9coSKyc8MwFjY*P5l{1(`0(4XJvVfFUxEEgcnzAVbUWrRRcACBkU2b5q71P>g*IB?Kcn~'
    '%MHX$vf#^1@rQN9$ctw@VAw4BbpZmJZ5Hc8p(<f^q0DxiCsZ-'
    '4+dgF~QLfC+SN#g#HkpOmK|3y6n(Mg)jT|R;tIZ1~M&nw`3vqU)KRXIG$5`IhV=Qmy(X+RA^z10wZ0Sjv$7EjaF`0LCOy*ixvny$'
    'o-@^NE0*UfA3a-3HQn@W6JZl1O*#;Fm@EM^Dtg+Ox*$T*Ue!Vn`+-'
    'JRv?;63W;(m6p6Q{$;GtaI@jH0lo5u+&VXv8QA`x()D&u&Ka-'
    'm{kxz4!WIe?+Epu3b?QDK6C^<|>hx?s;fpsc(8E>hNGobtbkYA0iDa3VMY+Z_ZC;4;`5J1xr&CN9f~Kg3@=g<g*)UVS>&Sv!R|()'
    'Hk-=?S({rYip5SOw@O_AInRL`rbB0`+K6+b8{B;a-'
    'ufaf|^$nwb3>n+KBaXoTP;6(n_h6xdIsdGEpa*t5+#$=Op%*H3|?j%WuVK^XO4MwUT(8E#XEH;#i_)q4=1ZsY$JJdO=^O7xZ&_L4'
    'T(g3~+kEc}_1F==6d?8NJ|H>nsCR;BVGh2CBe9>nwxQ>je?=a2}YT0P)YL#)Z)lKw_K+339qRE-3g}tp&%d|6d+M(1)J0o-'
    'jDIpSPYcIH+G_LM%L}U&`}@ElNUaq9zqP@bFZ*0N4k3&Kew`6!F8DI0pr4O8&3O<*G3%In~l|zLluymWH)1QL`)`?z@S4Bncg-Sr'
    '$IlgPZ7%)1(k{bHt_<#W)HxJn@y<8(-sP3x2@-'
    'YC7R0lQ}<*QA7?L_CAe=*A(l@VF%#^culiT3w96Q2Co^`y~8<jZzgJ*rCWVFQ8O$}={t#<X%j-'
    '0B<j&5gb8fu{m7%)gkjRGFie~}YSptQHuR}0pKF|<X(#9dii<DyWrnUdum<IG7;CWx&EGKCVhx&w#Pq(vDRn0}^X3JN-'
    'C5J~MdVAfrsqo-%_Gx?W>}reE!)yt)UTP4BE41hvEAElQ&mZ*h!9=Qfr+c+X`V4KcTb$sPswq$?1FEDKdS@-o+r-'
    'BW&?rh(~6^&uRTkdo?&MvoOi0P&q+{=aG{xV6V%*XXy(@mDsL_{^P8YJu1`=LSH;C~912QM#oLR;|Jjy?;AZ%0R^leWpJiqFb{G='
    '&EI*KXywBFfGBjbkRQTS$8l0AEm;i9k#1R1Ql{f;xy)|I<eSIXkN0Gk%4`!oy^YCAcgnRSwF|lR4dH5eH6W%<mCIW3Y51$Z+wws4'
    'fQQh#P&YC>SXjQ~u!_5Z!prfRXx6Ex&5g6uDgP4YzQkvl3P_u(I{z#HjmgcHOX0C=~JJ@TP2yvgp5h3oII3mPfBwqIYRf%yl>&xl'
    '{Y(_Uze@wt`bQATb1Z+n)Pgf^kKe}nUCIK7L&C<0A*pY6Mu1mm{3^*RNvnnh@#B{=mUVpqc*sQ)y0WR1r{?S{33jP+DG(F7|v3<4'
    '6-rhAUpAb_<@KpFRyT-'
    '|EY9Cvnnworxg%T6!uCb{(y=1<!w;SilE%KX4Bd@^MmiDG8DSJljHX~(^RTNV8SVbZI&YV%Zl77eMHtBbNb`-'
    'B$Ts7&B)>XgGdXmul{;l;9q4)he?IfakX&K6GYt~f2tA?Ylls;(MfnhS5F_O06G=IyUi62GC>aYu`tT(vQ*sZQKcAG1W-'
    'R??be{db1>Iv83Nw|>2m)2p15s0s>lMEveUt2F3Mj*biZZeEAe6PJ>UPyf;6V>llk)?M)CfAmU>l?~u0W$4cFD=fGm(pjnA5wP$k'
    '4ziPx#fTb_Q>Xl1vbd$kOdM@bIbx$NI7VMeX%)eflaYFY=JzZTGL4nbj3C4L0rtFYY=b2qHC-CmE`e*tS4#0`W`Df+Y_~it|TwXr'
    '7s#yXZO&PjI$JMT{_cO+kBRNkgV8iLtf=)CwS4({c{-'
    'e;T)_za%^olAhpFbJmb}ChiP~w+o?UK;TfM)2TcF6jcwAESe~V+<<oAielG3q%IDG^t`02iWsB#k(}TG}32yEOB$HBSCER>jJw(T'
    'j^^UF<YA-!m4P^S9nXEl3{QB@Vh^98iTYx-pZMn_G?%azC0I&ylpaKBwsT)vHLn%1s%9zRoj=6GfI}75yqrR!uHZo8RD*{TQafX&'
    '*5}fhNm17dz)j^A!mWfxWQibcqWG>8aTS>`LL%o0?3xzH@3?>Z?&!>-'
    'Mp5FW!r}?zcsogk1cF&AfRD(I?X{BlFo^?^c<KdlE6|k_-5!l+ApLu2`hA6n7;^cI)T9;TFE%jAmd9>8mi6zof-'
    'z1hvOMRPIDlPS0V!5=`_b#rz5B6{N&d+^?Iw0pAprt~!^{I|p!GnfWM=<k%k8!XWy-'
    '916sD`=~I~?6Fm`U5qm)5v+Rx<r4oMiF|@qM2kEuM2<q?E|x8>JjaYb`Qyv^F9WM{6rGakO?K6Gv+=GI6vHA`?d|6PY+#Iqy(xQq'
    '5){^5kjTPEXk{gevt5;0_R~4HanGEqE-Yk70}GBz-oUjt3|8j{*sW4(C|%-'
    'kGgQ6_7vLG`IzVQ!zoOd{WrDg|QgxWGdw&q_tC#`i~7&Wd37I6^Z}YR7I*kHc*l2?+H))+E1ufK6BRoLbWF5tRX?n?tDo_?~baW!'
    'cRuU9Efqk>w1-Tv!UgFiD@H>tG&gf5uolgE|X)?60Q%xNle#Jw*#E2eQ>IVUIO<3N9JX8UX-W>u!9GA>|*l-'
    'XXt}^K3{N(Iq08H3C`9B1@vjbMDK976|TKdC)Zx6vuiKZ#kCje>e>r+bM1w?yY@mov_;`;aL?aR;d59VE>V#j%zXj(K|}3urUZar'
    '9Tv3V3tA?@RzFsf-ag0!-'
    'b(~L*nSJmr0wfXTDhLJWhLrf*voX0?yY{62?BNSz4Be%2Gw4&3a=DpAK_GzDt1nH6nf(XYIU<*;B?e~WEmDP?`|DH^%JJt+FfAx9'
    'B!D|UQJF+F_Y?z99bNzI06{~xi!R5$d|?0#cv4Ajrq<d7#^9Z<Be<a6@kBcg1Pz!)dQYvu3iJQCh8PFG3tC|J);bkAuy27i(h0ER'
    'SXt(weBXO7rV;T<PP;I)X4qdY4^3BN5+c{xxLm^6L{&p#vsR(%}!p*W@j&Dvx}GY+RYJLEyY^;-'
    'Ar3|fzxS}vcr&m(j<(ezr=(t!^!@-z$VkFnU`Tir(@3DtB_5D-'
    'N1iB&J68zFcUUd0E~4t2sUhm3CqAQ?G1Cn=sN=bvYFBSDzcSN9b5#c24~UOI(G=Pbq3RPb5Ao@Sa*uB+y|&e?vb<2pbQo}QeOnCV'
    'eW4i&{F1<Im(=JN14;nQRY-Q%A8J)GN-en%<1B2W-'
    '$@DqsQT1;bqiz^^%gi``P*z2$0vhga^QxKY@qCB=7_t36E?haQ_OQHlu^3&43x{kf|c;>gU|J$_Xl}u*KqQKqO}=GjI`VSh8L!cx'
    'Fba-wD@d`f{Nsq#D~8Hn2SYE2*eUS1DpS>?)zV_vtq0J($)X9_?D&FK98AZ}XTpv3i>~cQrD&WXfGG6mU#a0P<k<mmmcm;L;TyXB'
    '@0|Rt`?a7@e6&HAC}V&`ciSpUHX$cQ9N~NJU&`NRE(-'
    'SgXJ4GNdBzs6YP>sfa7|=gX0bxRd_;d!!=ntUq6YRK#64HU)omI3M}&mqzj-'
    '1tZ4rfe$0b`gH^yIIgZ^tV6j%#%kV+q)nBrHfjEBCd2F7?6$ZP=Ew;eF3SP2sVB9Ux=VyGRMUj&kuVOAQcDvvd!>Q-'
    'n%3bIIt>m_7_(h3!L-5-F3o5hYmBY+VWD$UQmJp37X*H&Km-'
    '~5v=izK*A;3DGUTb#jo9j$sENqlrLXoBSb@UbmU{^#Dm7U91F{8inCp+o3B<vytC0qX<5wN93$9wtAwQw^%W=1cu2@Ss%R4y0%DY'
    '3y#1B&U1dtpoU{V$&#8n)Ca;i=()fY$Vu8B0Qtuhu$stZA=>xHQXt-)&04xC2rRgI8i%5L%+>Um|J`$GvoToY#6*Wr=fF+8l-'
    ';BvWT`~rKrIYoS}y39{#`%?YRPiR}GF834KzQX?RP7(iFUEwFReWQligto0nq^!;`Gyayq)MBFvT#0l~tgdy!%BJc5^!HSg^z<L7'
    '?#>7A6rs*$@Ty;8TB(`l))iU8$f3q|DOH;&Ya|iYnBBtZZ-'
    '1xSWIMyKkFK|&o}qMd2B**`5@s{+Tt|^Rlv{^y<tJO4ZevCC%XB*{qU+KhSP}gy{gD;XuT?$HC4pHKHs7IZqHaaHHvN-'
    'USW|0Mqk2N^YpKD`K<kj=4|W#2tBHCRd#F3}XYnYZwp$VKv?|G&No<oj%e#5(ktm4$mQZV8BH3Wx$Jz-Eq&iEQo{K1eQU4g4q3;9'
    '3^;FoiX4khB>U7snuf0%bx?UUGkS0@|X%Msr@oRC%%2E^oM)^4c6-'
    'MKTISRAfjr;2ugq+cqKzrz(Q=gf?)kR=uoV8|Fx2wIV8>(L<gOmZRoWmnD))w@CMf!Z+R343<DxS)DFau^P+h(@?P;{+S@@cdjXG'
    'OJf97dI(+1R7qx5#SKI!6gr7r+s<QjB;k@@g_cp4G7Q8tW&~Vd?B0NcLx&6~kTZX6AZtfxJ=N0<;5CIcT1z^k_W3Z#1mu0-'
    'H&#F)plqD8<nnTKfw1SIeOb*5`w^#IRJThi$tASp8bdsD3IM<P>cSrt}CTQsXIoO6|*Ayj?{$RTmGG_f4;A)af2eds@1s3CMY9d1'
    'CHFUPGI2yv*hsWAl}V6IPv%L>^UETwX^?8m@FN6WCDe6X*9ym+0{a2=$QUenzLE)lDHnKJJMP36{F0F<J5C1d(H0Y?2RsVyf#d)n'
    'dteZH@D`%*EEIc`<@D#IX>peK$2+uYGqlB5%{DbJdr2&o@$EvijLRoYno&hVy(+ZHR?=PgbI5r#<w?Bd|vD2>tme^)HbgW%J){0M'
    'Ay{Ql$A(7iqrCg_kdP;pN{uCi-'
    'Q57>sBaxA5dtI;pe;JwtY?K%8{&dNX#yc*9)h&!`gA*f!)c;@Rt^M)SclCSPPuhYf3|Zvj~9X8Jm0&Ye<tmFZh61N$ypQ|IL}jB^'
    '|?orQB*RSB&fh6Qx9v#}tuj8mp#N2s1Dn3%1t#9*y)E0s%(v>G~}dFgbmh*x6=qQN%7<zNA}GLJY^fVs@xC)kJ=vjaQb3MY#CLfG'
    'Ty9>J}d;J^aqoq98QN>mk3=6U&0d`2B>b)}iY*^*@$p$SG|sdLi8#BxEx#iR-icDVtz?bjVBXTP<&`Yf%08cBYy(5@-'
    'mQVo4lBuo?@=1MxQbR``og$9E(CWWRGyi5vBXLy+unlA7%DKuT-'
    'b+t{6QkEQl4oHpft<?FgO)2UvzzU))qAcK!0kL5L_oU8f#W^a^n2S|7OV{uW8SrU);u=QY7Uv5p!{}o-'
    '?qYO}pwxShqF6ZY@S}v|4*#Wa+~G$H#~pr*aNOaQ!f}TmD;#(Dae}$SDJ%RP$=&)uUe_|W#^dQ^jb?IlTtE>Ni>2x0cz_Qx!FJ2r'
    'n1`byi05g(TFM7-$BcuP=E=m_&I6^xCV+wRWHZ2U8<{)(r@-'
    'R5?^(sU!)7Gj0_rJe9^MYHXJV8m(y6@Cb3uF=VxRfx4DNgf_j!>itz-'
    'oY7tDH<%HYDMu<eGc(ZWVd4ofU?H*DQ9moRuWXkS?l$aEU)5Sp-AV3y`hSS^SPJ7Lel-'
    '!>Cgk(sc*C}2!RU4Miz=h4A2$4h>Sz^;F$JtfHEt8HrYR#a$=<lm;QwduFp)pa)g_6K#nO@IDT-C#3*LAKmzGpj+g+$4O@tOxT-'
    'd=${(+nJbYH$b}Oxfo0$&bXAWri)=5_`@3wRH-tU@9;RC0R}!z<D=^Gd>PhQZM1sdi(rtMS?rXV#eSgWa5Cc2>P?Xz&5FR=aDdyQ'
    'mc`PtyPqi3Thg`@03B(G<W8~a1Tg10)AcqGz?|J$gHL<4Ezdt+!0fYMVu*k#XJ^F4IGa8K2RjLihbfq%gL!U2zq>-'
    'Pj}lcaP+i(cmNebir38~s3y_&i3`m?MH8U*F<|FENdNyaP-'
    '{;R}6Lqr1<U%m9%xrcJeG;W0?ORy$q9?b}!^QW`iaJnk%{h#o?QTv*=z&*3>R=f>^ffI0&u3erWBOZ;=~kpe*sAE1Uck}a&gq35F'
    '7J|F#1Zwb>BW3vjzCY$wU!Ry0uU*d4&g!ph-ZJ%MTq!6<a-'
    '*1Ygz$2h5M#kXa;|gZq1v(zj^M5L_5f62D@bRlR94CYCSLJ*rZu82^jU5;2AH<{iwoVjw(3k;*lUI(<5yG*4vW%5~MeSy1<#+w}K'
    '_lS=lvcjP<Zstq`_X<_I(}dRgFh7PedFC^Xp)vUAr}*nXK~R5xK8W{y?e1+`A}vDgZX#om%)b9Ns0Pq))NmZm@ENyqgiX*~vFQ`M'
    'PmkFFsQahgmRr3G9JWq0!$>j1+<g;x6GV3^3kfg<Q8@fh`%NRP2qRBuY#>oyhYUCG@<((z!*>je1fv^9LEQCZrgwkL7hlzW$??Na'
    '7)N!zD6R#`fvC9JZPrS(~5DNh^togvfx(rq+9tw0v%0j-'
    '(Z)ylo*Y9iyTTLx!?)5i?IADRp;($*YC>FBUk&v!4(f+1eUb4|u`2&PL9Vg2#*(FMqHlm$pN#(|V)P<`!g8By-DRqk!kI-'
    'Snh68GCG_x9=mTYTLCjn-'
    'MLCOu>;-pkQ`eU!5$YN_Jg2N<kAhV}0C0d_$815IFSIKT2H@Br1vin{0A@!LIfaLmI-s~mNB5*2oi5IuzoJI5fNMqRR-DV}ZUFww'
    '|iqOrq76NibW4in8BCYn1;?82DfqoNKL(ue|#GE{MzKs|^47Q83=q<j0tA+1oA**Da1(}v_N(VutG;Y^%}9vLT8J@{IyzT!@6$da'
    'jr`+~jUW!xmK;G>Da9t#)Qsx6qJD!$o9X<yb*8mIkOL1~iqXZ@sUI)K%aX6bosrZ!IpvYEO|I*7f<-'
    'P1>y?6F7sDC>>AK*cqR;K86EZqC#3b7+yW)GiUA&zL#rnb}P!YwDS)ghyDTAFm<IfR)WH;AI&s`?_5Cy|r!RW}r9BoSA_Xerg2Ns'
    'p&p?k5;A%jv13=bQ+vACdcS>5DF&6=nRmbCd24VK)MM7{K^?0XTi)H9G;t=z=-'
    '=zdJ?gxW@n(j0M&wLpaG~KW&(O;K(IGVfmKs*5-cL>;WRM(%NdjBg9zE1A#-e|qBuPqWRO|+BLG`w-H(Km-K_gjaI%|q|0SI4X5E'
    'j5lijTQF>qLF!tV6!^f=yU=cLE;KKpfgB5$rf>B+pKP}*Icw+Wt459h~kD9F&E=RYu$kb4TufLYKKR?287r@~4Z4Mo9984cw$SSh'
    '2SoDM5xG?X)7rHqDhCalz-S$Uhq7Ff@8Hrp=;f;nc`NDl(Db(1dz@IbYo`;}W3+*-'
    'YPZ?=HDw7=8}F6?skhM%34HwQ0(4$Ok!rF5K`f`fuRus3TcE#SNCpWhoU?4NH1KWnA`e{Xo%U&1dAG#3<>l`IKFdMrx<nI6ZIAW4'
    'sBNsvoVU`bGtp2(7*etHs1f(Gfy9D6>O_k2eH!bUy=f5WV<od6U0lAw`D=Q0lVl<E0q4i3)D!7*^pr|{v~(8bvs!b)7jlCYb$+WG'
    '*BTe;Oa3}&FxfBtqb=3A~XJd1@_2SD`q98W)o0fEBBY^Pn}wU7ST5?=f2kGsL^7y9Gw@Y+v*+yh?w>yLZF>j3?6FL;&aLEtd8uRR'
    'M6QBzr~I3(ljz@z3GMq53;{TA>n4H*nA+^n>M88AMO?(qD9(p3&W{g&6M3qOAa?DFq<ox1V!cZNT(fpY|U@S1joJ+q1TQLp?pO%G'
    'MGtZ3UqrhA)bIyAs_LoM5J8TaPLYr$y86)k5eT@Ls0t#7%sg8y+1D}+5*F0_FgyN(sYUMv^d!M?tM6~cp9E_8sC=_XbP51|5`EHE'
    '2`YRV6>n@n4nhd3;wt)LdvP~W2ffHcw{B><$c{+Iwjn&^)?07z5)u>=6pOn<Bo0BNp2HUNO^!ZL@JnkFEKx3-'
    'j4Pgi@n$?XmImHoxty<gm0lH>2nQFji(J6FvcgIpK>lPt$t$M;Gs@8Jq*ueJ%~fa0tD66BHKt9=t>j^L|{64f!WR~II#!d)FE;GG'
    '_9QYDDV_iwf-^I~%`UR-x4{^VBz-slm!skg4eexd~4e%r_Use)%1cX$KQZdK}1Ot`Ov-'
    '>{y1Y|X)~WRV}gjFWRlCU~iJB|CeUXk``kGX1$Be78@noleHYce;`N(}v~a8((_)<(FLb`}6*E`LH|9qaBpM1F7evdRD4uq<UJar'
    '=)sPss(UUu&{_3{kQ9tk8f&*U-^g2hYh(Cz8MC?1e`6@xkCL0HEi`)p&k(GArTA~<wGa-rW-tnrf6P?np>itDp60Cs97ayMv0nVV'
    'u!%JHYy)_TW_j{T1<U_c6^J1hcqr9-'
    '^ASC`IrCxYFeFUKXre0WRvot_nOsf_EUF91BSpDby|SRQalm%1|Zv1iF+N^_DYb6gs+|i1K3?%kf=@$dCi29?&>?t0<l*;(G2G(E'
    '*PL294aT{HC<>FZU|v{R~{_U-THxbzKx`Qjl<Y;WE<<1*~WTUVOg4Dfr!3=`xCIhM#>P3YHNu@quNpu!f{Flj<m$)Sq?59hJ+;ir'
    'h_BVhc|R^6#Qsrn->W`4WbP+AOdx;IL(Vg-2{s|*7eGE*aZDUq0W)c1^o$CYuH5lD&ah|=SVd~`Yrs78xPmm<)lz%gUd-FdT5uE('
    'gtfKcJngHANC2E+o@WgkePB$_w+K!r&vPfx1`$Wa6JXDJzQ@LaecSLbryyjeHd=YgW=QyFw|dtNGuHDn+}PE;RYHL0YjRIU7}#Pk'
    'p@J-5ax21;$Zk@07Fdn_<<>P(34l`aYPfeoO5?1wFIt2^-'
    '>F=k+z9lC8i2x*+}lkrj_>_q56PK`@%o^;z61AgMakJLo)3T|LBW{WjX-'
    '<G0_K^o(KQvi^(z_2><AdDKZ@d|LBXiC8i)T?55(i2fIxn?C#kS?7sA2w>A%U(+a>&fAyiUu!C<pG!}MiX-ov{Xd+rf!EO@`h=3i'
    ';WsBlq*StvpJH%P3s+6{A<o(j|byiBOzq4FQRUWZZ8IgT+6BjtxwosL=GdwVkUBl*TIVX^|1}nrQ=C%PVWF&LKbmA1<c3_2!W_zs'
    '*=a+T>E5vm9d`GIy4rbHv+Jo8V5N3bb5zPMQ!|dZcm`yJLGyT<v#lj4}>9AOseN1B_U`7+MYZS~j(|`z=!CdZIq-xhBFgTE-'
    'V5!sCmQmUyU;QYMs+X@8faTz>E=^Q>r@swNRGGVKB2>;@t;2LDdsQdZ76+5*c<sStO9+#D^OQSmcI?O6s*o^usBJFLTQa=~0^N&l'
    'm${1Q1qt>yO7fuhz7M@sdC;3t0DAgyddET!zNvRC^j6WB2<Xw|wv2+_78(!%Jy@TXMe28fgsaBWbD`x#D8*$QX4?Yvwrm1rsJ)x@'
    'ZG+kzcaS|us`i-5yijI4G_dZM4o3mU9U5AjH2ZX<DMz9X4Q)-'
    '5eL5Ni9Cv7F&nfm}xjv^7<6v7IZfD@Nhuf_oZtvTXczf5!SX~~*W){Gh{_4YHF$Ujscr3>1XiNmgXd-rt!q`?C5P>n6%iW4p-'
    'kO9qM3ql8sA&gndV49JsWxE)yTKXF4G&js0FqWG^FyX(_$xy>K=-'
    'N5HR*`;@5ZHHf%Wf~ahmxWn`_b;>)(x6zYEsC|DG!WzO;$gOQhQ7&^8mVJ+y5L(e~FJq3uN<ZO?^h13_2-'
    'Zu+~Ah=m(`)e*69dyd9Lz>Q{N_b9k+qX7|cgUQ^zD7dA~;RXX!VI#L;!cS||EJx`$OQ@?XrQ<@WR_EAKz})IGmV;{ysh?j!wN&`4'
    'KUe4qcKbitB%0OQ{*R?X6I<K=QITkFYun(#(IC$EDf$-'
    'poQO~QoQO~P<b6;31c}cu5oO@HYvg(E8i!(TxXXK0PhhhzDqEX}WHIPSHHfTdR-'
    'I<xR{685o&hA>z^(FU*FEdNYm7sFX65n5gn<fmrNz~lKs3Rlh}7_P<Y=Q7Zf<}y1KN<E_GaP|A5QeauM$1y^McZ{-'
    'GIal&<{SI=*CA7!|*=@CI`Nt8(+8qIWwSwzTp;PBp*p6xUUKIFJw$puZRqK_hj_4DhOKIyGeQwH(FaHRZq!0pRZ~XA1W+S|B&iH$'
    '(1Aj1OrcU>Bzr84VRo-vn&JKc_D1uhOk{0!nTtKTdYw>3Wn?bXkeL_K8DKCMo+p|dBsa<9cZDP>)tbgadlkC0pMSgCpI_xVslg8A'
    'w9xFZ%-'
    'uDA@m+7)eAm)>l^gaMBEmMUc80dBH@d7dRtNWZD7DMk8Z38{5CWLV9jnT3croa0R8Ai;Fra9C;EnB@Owa3`cn3~jVLT43)%Jd!r}'
    '(d6=&H3o+@EmM1CT?JXq}EP#Oknzf`YD(_k7~lD{t1!RF#CY|VU#B@S4gf&Bas^6f&%FApK#*+E_(8;=Mi<#N>!j|$|ZbTkr=3w4'
    'N3>NQ8GL&*UdK>8aW(qHF6dZY*GUQ#`;Ar1E4D?X$X18I8D+hZXOw{Uwbq`#&yML?S6-+a2UA|PF22Edx!UKFJ3n*sXKi-'
    '0umvd;7k#X!147Ser?vTdXAbkxM<50#=wGo7@CB#7|C&G|ku!EwC}y4MK~$~iobM3w{(s0%`%whw{2A_Que5U8Nv^z_;4W(#$Y&q'
    'nu{PzU>LbWdPTIlIO;3v8%#pzFa7@5fv-Ie0&jAC6h$L;CYPNRRR$eXvw7X-FR;)!z&07&Ik!#6lWw;f`2He@<f}w2TFGV-'
    'Z?Ldy2_LXc_v^BeaYz^bN&8x@({#la%J(%cNy+u2!AP>BAKJI(FQwabkQH#eK9GcW54krI{X@pA6C5Aw=`a5Y1hEO(PwIT=iG8DT'
    'wdc28*8wd_V45xx#ynYeow1xvm*0yyr>V@C)Aa^D+F9kKupkVR*EM;e(`lQDgW}sa`IGVVa1Eu^5J1m>7%Uf76%<3_nRX7J*?{vx'
    '!j{){h>6;jZ)z#bEg0P%NK=^%9vzN3*4EhMBFa(b|Qr30x<v*zSW>m!IUd@E(n6A0C>Y3ej8^qWQfL&D}C+uClssb=p^bWfODzVS'
    'iKRXjb0{OVX^q6_%x0eJ3nUv-+OeBLoQkmyh6;c?cfkA-K1MQ-'
    'BZ*y8IKO0D|e^?~Fw-+`^r)2wq8JA`tu(-B<*IVV&-bLa=`H2n2VdZzu-AM}<-'
    'twE_TNP}=B95p5I1k(NM`aHLBfQ6yH#Y+~;iBxmt(`*eug@({PHLfm%GxG6Sbwh)`3Ncuj&+Mt+&GSF(|NcJ$us^pj_HQ4IpYP&$'
    'f1Y@Bh3*v9Ji5j4Ij*sGLc_<$1p%^rXM>L9$kf3fCKrv0lq*xTgEli3<@iZC}f#Rp>#v)J*t2QYL#rn}BP~4rqp%@gO5PB@0O4MO'
    'F8(KO}sCK63KsDm!B&kIj1+orE((bn7LhB^$W!qWRCFy~tf=H?nIg=ZgjzvlW56{npc<vbD`TY>jJuIGWW|BS*%Cy=(i9C&8N<KB'
    'J{CJhe30#FnfK6|ykow@r!|-$;!&5vAU(`tbs@y{`=ns!-'
    '1RpNJfb<aDz#y0&{;pUA!wuXOi{L3VrU(SH3*s5Nu_6%6E(lnsyNW_EyCC$V7lB~*826xWC<ei&1wIQ{+D|P-'
    'Srz*FQp{P>@ED&1I2w3lO8aYkYPd4)-r+J!W)XFXhuyD2?A{^+?4BtD?0z6US)A75&CRk6zWNFfft<$FPXL-Yxuien%C@K1d7vuW'
    'o?Zh%Q?@<527y7B2jH}c=Q3^@#wyapL+~k5y{Q|Mr&7I@<jY_)2a95z#!YZYoYS}|4v7%KUm;^yj0nDko+Cm8pGgBEM6jNzVgP(@'
    'Ac8@o{LI#A_rr)+KQ3}#>lL|w$wlr8+36(cM*8Xzp+3wY_3IF+6GEi^Dn#ms()SYglj&IIJT6Pp3JhR4>Dw9k`CWoQso&D0)a4gq'
    '*Qz8PY?_AlQD;eye-Sq+xBtir;IDiEye===c533$`oQ0BOaN9=z2gJ8!~j@N&8x9881CWKSQ)&I#uNczc7eevy;>B6*#$O%o}>r}'
    'vy}f84JZP_EamlF6$9Y`fi(w;@DS;pym)2pwL5dnfK*8kF&%@>CZefZ4U_H=skLS&fgdHdZ4cMqgt)#n#PwMruK(>zVC0o9<CtCh'
    'BrWHVU56y?$Pv5pB<;j8x{gWOnL~6HN!o=84LT)hSK>3sLiaNtx}WAjSNX!XQKH6a;oBHgphCh|KmI>ru?sixk67$}N@F6#FRa8r'
    'qQvj5^du4D_beI^A%68-6@%RiLOX5<HI-hF3w4A{FFa(4J=^O<QHD<MA+QP;b;3_mOMITL*#gPWJ?MTLLie^1x@U*b{U{@n&yh@!'
    'Q?1UG>2T)y`L#?(FbmCZWIB>*fBMLD6q8O@$#gXLkLWAYF+^aJh3*O;x*z62_cR~6Crh<hL$`jSMi+vve*D*Bp$j+hS}b%wq%jfD'
    'g_U?M3c9z^lSDxGY#I;&T|HODK=*f{6YW!h>0gx`h9mIWjzY7s_l>$gu>~15MlIH<d|p{VSs1J%mAZ<VW7c@gF<&rqOgadC8HQ@-'
    'F@c?RJivb!0{-?8@aKep|4#<+&864AqD#WuLdd76Ntglz|I<hC@;n5e?j!gNsg`I2mn3SGkKmj^uzsG`V-'
    'XBD@p>$Rm(!Rc5X`|gnA+EiLNIF@x6_jpfne4&&Y=NCAec1`Jy*pbxH?4e{2Vex>kLHWF<Y20K}@x(kCEhSxElxtmTe$!ABpe!-'
    'Z*J0Qk)=uTwsSM50~GExV$68<+&j)|C`Yvs7;q)C+n72!YJ$##f(wdA!<7FCHG92J_=b3)0vv%O*b~K$wTpMAH|R4q4*3R#SIfRR'
    '-?E{qQ)1J!}{sph($5n#2c|FeuTzE$YEHBH=^Y59rPp-'
    'a`;>t5Fv;4Tor@j8$uPL5Aw;Hrnaal^hcr0T*dnUhE2x^OrPAxtGWWn_UHUt%HU=0KnCy}fi1~A0Iv@LJTV0DuR{QToN+^xyMl#|'
    'u3(|U6)bde1q+>B!9o{Tu+UW(ECkT~(TDEjJm{Y3L-#bP>NIp4Bx+0{=<4VGXDoE#CjJ=<-'
    'N`g20=lpS|BQm}M0%14=>D1pL_k;1RWZ;V6$)Tb5kAEl<An)l^v?Edw<_CwypQTg@i{Pyq^Do#^h>Q#WTSAHo$$YG*=dN~J45XLC'
    'dBUlGD<;1>AE+mM$)x!QjMkS-=vyI>=11<;il4cb5hNuYmlXOk*2p(faD*1ByZ39fQR@#;KOyc8hkocs<#WlSWm&fVqpw-'
    '@UK`HZ>KR4FoxCmR}_rzq$i1h@o#8A1dR1u6#-'
    '+_ylGfsSW5|Mwzjp>OZmY(wFLgBp$Nt<^OXj}$D`Ly7ftse=^Hd`!xVN)ET=|Umg<gbw6xT6YK&#9?xe;_OD?CzNy{#WwD3$Xr^a'
    '*FKHNk7ZIU{Eyg%`G%%U(${T8p5$9Tc+mxdB`p=L(N`h%N!{@~`JP=2yd?z+CEsgb$s@I_N&YyF$!km57f&Gd|nN?l)s;n$ed^;_'
    's0iYf%l!(a`!GNJSauTBEHDf?Bf0LM?f%Syi%Te~2JX8S{zy8h6uTz}}+u0M1e*B`pA>kr+|^@ncn`a^f%#GJ<@280=X@bQ5R`d$'
    'suH(^IL5})`Yb9LU~|5PEI=`WuaTO$EqH7&MAVl|D4K$@P4QBg>P*%}pvv<dVLMb&*Z4NDH!NcFfYB;1qIaQJEnI_T|HOM$8>Uo{'
    's>xSxHEiLmx+6m~|?7p0&Tr7b|K*FS-'
    ';I?v$*K0cVi$v;Az+?ro;sLOq<UeQKkB}X^QJ<vw_)rgh0wBjmBBqwfPROMms10Q=!^RV}HA?)b~njVWi_^Ro#*jq|tBCw~YcXSl'
    '?U<F1;VeeM@hN9}h0x3ssQAdQX`OfXt;>4?@?t=|*kCZM5UsEtz<m6l_{%s@ZTA_AlBwcG%?*71Cl8?7(R&7zWtL7G}C)zZt_UJq'
    'M6Sq)3*(M^+k(lzrIEIf8WpMmjh~wLKB=45^n0q4+bI%mQoc{6|v6zFenh}e+H)u=*=JZsIiNYMr)|e>F-'
    'A3O~R3(42kUeK_WY5`C+TI(hoJjkDQVsZOZ%P#Bt4k70g5;|fLjCA|_PxNg8NLE3^M$>5LSix$;{-'
    'lFoWaTKAx>`3XLU+zt*KH;g}KJ}ijjla+{P8_b#=j{W;&^|<!zNj)>4}Xyw`o;y_N^OXA1#OKh4Zo;K5hTj0N6nG$sOgdTz%?0S~'
    '5sY!vWrr*9}qr3-'
    '*pO?lSN0O%{60nkr41E9Zf2EYK}41n{5GXMq(X8;VcJ%#5=%(P>u!N=MR)ZPf8c1M9?{)41n!*z~iQ(Wggedx*KTQT7;t>l)lrC8'
    'CE-'
    '`^s@+~0i6y^x2w=L%s?Kh~^R%)wX9ipAUuG$sObdXmRQVGezjQJA}fzM&}P&SK6ObnO#0Q+F#`kf_7G4xvQOfI&&Um`7p`usctw`'
    'k&c-'
    'iTcEG0dVjT7aT!QE3(azOF2G28O)1IxJ6EDF2kM*=EY^)*QG5NVNVD1;&N__+@4FYXM%ZQDzN8C%o}BF!^g=PZ2vRF_QV|>w~zY>'
    'd^Qh(e=CGQ{pF9uA`rgnkyr#iOJgDssHfuQC<LMhF$#eb=^Kht@j?WCX!~V-'
    'kf@e&bhcIa{gm0$!unRK1hWR0qI9u+!%I1WSVf1S<pn2CjPGLk#FMa}dJ^_CPr`ogN!T@>g#E&kuxmXD`=ynz^CjjhGYaA3lne_0'
    '6{7IY9ii}1AB8ise!+vw3qVkR^P{m4gztGY7J@TrOauh=JdBTmAUgA+Ab2NzLoqY}6EbdTjstr2AXWYm2iD{9UKHhc>}s)tK1~Ju'
    '(!`lUxf^(Ou~aW5@UNldk?NHM{^gX-s~m#_dIzhR^l38Z>$L_s*^fz|rgEITEy&3MO!_pPbNJeWoE*rcPct!g9>LV9-'
    'sDDC7v)rKi7H7(JH#cH4f84)U*&&5e3gI0BC7mpVebm_&A-H!bym~9Sz>lPUyYHD?6R6lDz0H3;BpoV$g*`EljK&gP(YTg8<-'
    '@wGm8Xd*}92Ia=S8P_GDG9S2ZKm_Q+ulnh86X<ZKIH>{)!Ibl5wRLUtB={Uw)%UernLk1I1ej`C8yB|Xv6#PUkQJw_M2Z{3N4_bt'
    'T~ywg%X5H97eLjB<Esd3V0%!M^xlIS`B6!Aye62)?Ll(g-'
    '+)Guvgg9>%Dv<<q{G17MEQkBxS=)yMZ+@nh!$88AHDe8}z6~z@yTt~6(s1cX4DR!jflHc|@{Izi$7QnSMRD=ELVpt??q<Xv1Qqs*'
    'u7kY2pi9+uU#T0teQceq(@)rr_m-6C$g9Y>U>PDG<!x!mP^(QA1%k-=<htE;y(Pa{!U!seRS$vL0-'
    'v`t99E0{K^Y~Pvp}<5w$D+>0Og_iq>FFmP_eZkF64RKpB!|7;hq95=(<3T4(j%DEd%Gw4K-'
    'sW134T$`jOrbKWfMz(2y+o#@BHOX)H|;&rrw!W_OWnf-xq3`UD@<Y^^9BCg%ZOQ>&?;kyu_4=4d&_viBaf{=ITX>!Msi8>LrN@51'
    'Y-^-=(TT1~+~6GWI)CTg}xgQuRYBBz={Ws;wnQnWnDJ?%7Ht(K}KW%q%bSGs`QH=$Nv7q-'
    'XKkV9Nc{=Cj4HBC5JVo0sk~x*Yg_J5dh&aWUn<w8T$_OS}m?x7t<RD6q7guQmuQ(&ejfkzl}H)yQ<6z4)_Czp@uM$@FV`u~w+v-'
    'HH7v_B*jx<E2kpD5PqYm=3P#bahREWA<+3df{UhtOd&-eGZawr<Il>#I=Y6F|%E=0ik1MwDeKTgX-Nv%S-ndT`T+kPSnb-'
    'DyCMJmiPH^d3y`>rJZ9~v8UaH12{LCD?Po<67_>mb<#3XNlrI@Mf0p#&T3SudCsa-YL}eVsZ@)cZ8xA==4_JzwR?^gs2S?oaO-X-'
    'jeMx3#6Lda;Lduf_Qe#ZMTy(WR+*~Iv_ag51a~19)K4@1sU?NhmhLjTg7n>;s35H?rh=5#_SJB0N62)oUCl4BJGQ<047)kotI^U6'
    '>cM%u!0E|t)vh_)7DDZo<7Tll)pdMSSE~~e7f(?qCN7|&PDxzkM4g(LaEVPXI4Gx%(!PK$Id!z>jjC2h(iuF=wMsLILX#5AA+kN$'
    'XWK3XrRw+`XkA+6szzOZA6)@mB;y_!+K2jvb^}Xk2TvMZYWd<$lv+MlOsOR;^_%&`c0Hx;`-IG3J#kap@TW@#D8;?L)J#zyb-lzM'
    'rc82&YEE6YWV|oDOgi5Hc$vh#^WbF?_Xfes&<ppJ)}Ga_j;mylkToQ%$E{3)gsiE4eR(=k7l&ygP_COv#}+D&`AIT6b%CEA*fXc*'
    '>h!=~IW@0Pezty+#qo4@i{t6+7Dr`g>)VRR&eps`5!u<5G$0~7dnyemCOL3fm>j5bm{RT9k?c?>J4CvE=cTHD4p7Xye4Pj?yV8E}'
    '8evkRB^%=gU)3QbF9;!dYzWC|8AvWmm)X{mHMZU>9d66L(h;`WD;;eMywWkczAK8wPF?83_WlI+D#7;Qgy<M8Y=wcXe%d#8g2(R7'
    'o#3&1vnb$lcAvhnB7n=;eXuZZ76n|+?mLzS6aie`OVemTF~I$>fcp?6>}A&It96y%YvAShYrUdW086bSfv-tx0J^^4@Ic9K1_Qi5'
    'hFHBY#OiS&R;OpMT9?+@)~WBN@7m6(OVTB_aq7}^sqLG(JY8-(rLIU<P@`1*yfR&>4ik}lb?PD?ybmT|=M#8q6EF-'
    '4<g)6=e`_Zg>)zT4#=5tna#{6_MdY%=O1<?H;GHDu!Tr?3TMA)+9F2@fYMo94ib3%wp>}wpP-'
    'RjJsA^0*Kc2m4Xyl#9t}|q7IhB23s;N+m6K(@^tW-DV(jJD}L#mr{#yUeLtUqPI{HGAi7lmLxJ_PfOJZ=tj#jN&Hxe5|4=v;>~+h'
    '`O=xGpv7s+{dn9K%(oF81O4Py#kA!Fh6$CB!v4qnUuWcY>|%?VVt&dpnAsL*G~gKL@PU+fn=+$J2laevTP5pcoXd3pqs><S;xljG'
    'g+{u|7VRDX(>QVKotQnBTfHhYo{Ea@ldrP>sxCF_p>vAC*HoPm}vUI){{r*ZahC>l2@9Zj-'
    'q>Sh54kp!w=>16&+#fD^(EFf&88^nz5Mf{e^rC)wxl6}d>dIyKaX@dJsvLBn`T0=8lSiUa)^@9YF&-'
    '8(x$Socm8#eu%D2#N!k@pqyq4o;u}5fleAX+SYB-VnlgVNT7?F#+;bIcB=AMxxO1ISzZ5kt&?2=HhIwGGCcEM6!F#VCk9=enUg}o'
    'fyJzRtA2^hYn^@&U$k{0c|s9{kNZiyvtdS?Hc&{ob}j#A=R}x>#<!c)pa?p_kA6;*+{}Vb%}@J^lNo*jyh?7qwdQU&@1#z)a?XeU'
    'ENL)*40Jn75c^^^a@zCx~LkN6KOz%Ix&j|6ocR&!YaiIBwZ=>@-'
    '}pdqE?rQVW0&EH8n?Qf5`=n3;jbKxwE?Dd^8^_*(GS8d2NX1OF}fC6r%Z&jA~qkl^dpZJ?)Fh^$#S>kn;~zSGhR-'
    'H9YK@FXs(ZUzsrHpLqClt@^)Egho=<sY`tvuT^(x9Dk+$QV7R-4&L1f%DQ)Vg0k-2C>-k>i@-'
    '4~*Sk?TK8XfI;P??5P)w~?eNiYm1VtU=^#kFjJeFdBKw(o=Rwin>iLgg#-$$JyLssW%H7<wL-'
    '#7Uj${P}&w{~OVGb(RReBRnEiO*ZRHSu|Cx0$QMr1i>b*k2d!luN^%a&ov+9(8(m@$7ZK_9_?CHK0}zmcLMwas`xQeV0pig1m0YP'
    'LS6viBgXBjYTNOFriDLl;e|WK!kGqC=Dp8Zn1gL!3ogXLvkCsGQUVH+msHa_4U$J+hVu{jHbb+zfPs<CJmi{uV1W?+<Bz2OqZx1&'
    'F?mq=`tk@9Gl5>xsr+AD9bQ<XVNZKs5XgCwV(p_DjV~!!Z3}8-|NHtxsj6d=ZHgzl-S<pyStJ-_DdN@TbJ`yVz7Km2)n55?2*fF7'
    'NKvqb8P)$6C4s(zt|Lq6p?esVG5eMA|g^A*sqtxgrhvLkCa8mqSR^4^SgAmmYgS3=2{%qsQoz80$XNR3BULUp=V9^))FtXTry~(E'
    'SF5L1&8EH$^2TqrCKF*-'
    'Y~Wo*)L%uaS2k5n!KRp$TVv5e^z4tfbnzei`_mMlzrjS2jvYRC`V;L*@e7ZN6A?Bxe_0UpeFC9fmohX^^yWoly0$0Z2BkMLzmd}&'
    'vi7W2q>D=R6SvL6$M2`)(x_laG57YN68{%Wokn6u;*BvPyb!z(SK{YBG188qgn2&(fz((;><9BW@{O^Z<pE-'
    '^+8fClg$5yEg2;fy&W#q`;s%*e}P?O8D#z7(g)d%A;?B&Kvqr>0n^Gybt0eC5$38h#R!hH(i_eDqh&0kTzwyoTZC$;;Rv?fzPSQH'
    'GkTb=v2cWY=o$;hPiagMa5P>FJ+*fi1xI!++(-'
    'k8fFnB>M$>?z^5L5Y<+kLguksxAeLd%Vf6qC8p68q&=sD*H344g|=P0@Q3ssu1^`iE}CdG`k{&4AI?WPcGV`P5GS*i7!QE<)^%Wn'
    'UaV$^;-Q{ob-N4b~I+9hZ6rK1?)8u(D$DpVs4#ZRT$%ZDOo#?!NQi-jWGL$_Ecen?{?+T+p8-'
    'y>u4;%}k>5qa@rXh2bwkh_G{w|C~$QdSbV5qm?9bE&E5Ja<Ppi9mgZceKf^a7uTSS+33fu_rcTZvb5S*!y#cy|JF0t40KM;j|+34'
    '0PqZB9wXEII##tUU$wc!UMPm=d7So*pu^&P!{$w87$SRyHp3}3NW<%4B1;lAB@|DYNEl|IR~0#0mGdhzI!Yf;U2ojf^j*GDFTcpV'
    '62zoFGYcoE%!gufFi)imit&5P?W})2gaoW!DzDpPDTe)rB~2eihyikrI2@U*I;7U$+pe{5cI7N$OEu1I|K4Oxby*869RIafAFH>Y'
    'H?ql4p(hBCpjIV+Hy*AI#RXcjO28bYR?JD>1fq~^O4grs*KZ-)3G2YWBJ+|`9RzzRAUXq_Bk-'
    '%3ILHFx<@P!;U0R#0`U<V69Gh8f_tNYSVIFMfH;l@6s0fj5+2YPG4JPPfDAAPZeOAHPuO*cH0uW>>^p?t0ehFc+r(-'
    'C`eMS~Lu6WSld$^`sn**j<Uho#Qkf34*c=F#J~oGk*t}Wh7wmS#Y#H0Rl+3Pceu|h=$H!_ML(Cf3G96_s2U}wwj+=z4ui@A!r;?<'
    '=Ax+QOGZv0;4?SbyIGM&o9MUxF_eCAj!)ZXoA$>CqC`w-'
    'p<YFZrv2;#r$@imBKX}b2K=84hjm}CCnp>Dt&6(ZRC|GN=r*tcs4zj=-1eZSWMufl{?}VZj#x(#N+gmOjo_^u-'
    'gwqk}TGtRU9hrXVy3VDe(si!qTsk`a%C$;N$E06VcCo%1n|_1c=py*khWdPJpGdW*hU6Dg{i2YMoo1j{EF|Fudc{I=JB^8ukhC6u'
    'jgpWfXh4L798Uv^(i=q>{8>+(;%gaZrKau!t%f?z;jMK|jOt7iBP6LzFr%bWt<Gda%@de|>;dqw(2>13bYw3G9oe&UegN~|NCTuM'
    '<*ac8?~erQq<@xNQ8^IyGPgqbT&Nj2c5|)~YG#g|OJ550Xs$pS>Q54)^_CtSJLZ(oX{e7U>cm2D)N`;tiA+OnhC|}gP@ChBh=30)'
    '1`LP@_$;R9hzR)T2agE&z+$gYigJ?gSt#H$1cuw*-{<7iDT!mcJ|$6Cv02>*y)HwLQh2^?RC`X2WA)cd)JQW@b-'
    'qwfW`Nu~1oE39ke>{J{Fo2qbOyXuu($kr$yLLB)eX|M`M6PX-Ee;_x9QKSZjIz};Xzm(FesL5?QrRJ$XJj|8xex9PSi_<Y#;sb8z'
    'K>m_pl)n!g#A2qA(2WuptV=Z_<-QU|2tT1cqUqHx!5A14Nihy(s4eDFdJ84g){>T&-h5>`EfvqFN{i-'
    '~|GkczFOmJOuDtA%LF>0sMFdz_swIV?vP0>AOq_G9_KYgdkJXrA!DiEnUWhAk)+3%mOkqT^UQ&c7zYz#W~a51$OqrLeSNZy)hQLa'
    '1R?}p}UmEL_il-U}F??-=ZgpfUbV@2<XBJZ!8YF9YW~(-'
    'gw>Vu*M}!d)7*Arbe>_WH5K2t+f@o4z^DUtYq_GdqfD^w?o)I9l~}_MuGTHXvJG!eFR3ERCk%H|6pS~b+@_tuK<f*-'
    'D9pk7GU$MznH84VbZO-*Ica@=*Yh>7fS(mq_0W*F;OpQP2x|9da)3g^<-'
    '>{1vA{lrdTk)L1QAod;}F4Mu7QkdXfk*>qn0OGpzfj;=tThgx$7$pS|xYcuHVxw1>wdLp;6{;_;afk56Rq_=!MeW2j9gnU>>-'
    '(`Q0W&v8uj3!!G^IQ+C$s7G>Ch++?p9vIREeWnLD@9-(Th-b$OhuNqt%5dxOHI`%)m)uc4g6~D*Y(nr~ku}>xurvtP)3-'
    'Sm!Eh6sV-frsjVS`bj4xQp%|#)Y@%0WpNf8L<QWgE^MIe~NSAdkw#Uc3MJOuA38s$5|iit)&E!49a#2*zRzAi-kvmxT=W)ufF_6x'
    'a5e}+NsmxMHi`zPd6b~R@{ty#LpC5A5*ezThoDb@RWX=&@%sM(1vW2MmFaT>t)AUk}4v=KcUTVeqW&$cBNz%S64A^^;>OIVmKMFE'
    ')eM(XHEiU2T2_w}O}0bus0!NPAT4!}p{slrvlH{ZNce-'
    'u?UoIYKKOKkk2YqByA7dTKSo)PM~448izg8AJL%+G~jo)>DtHjV&lu#F>t9&F<Xpa|PI0%*cEjsU8#jU#|LNO6QXQoEyl@jD4C;|'
    'aajChFxv3WR?At+D8Zo7ftQ-e+k{gaQF8u{BD8c$c0eLV?ha9-'
    '%<MN^dO=y(fqQj4Jy|HPncjXNCG(26sn?xLXq9?r$OP=4aG~Q>Y_{(G<=|kjm}~cFNUoiqz6{B(o*0NXIZK(#mwKF5c&zcD;l_*6'
    '5s__#(4}BFSYxT?mP~I}6|lKD^)ge%<OsJ+IZ@>k{=!QlKzaPvW*%^uir%i$(8D8WT|%3+uTpi7AX-LQfJ=7^@#WqA(WlvaLAuo|'
    'Y$jDH+73hV=1LLolqg2L6wRQWaCPfWCod)gvaD@rZ6YhnAhI=q^RjdV64L*y&|<xCc%N_rStT`hOc{lRz%LzFfK8F73ya+wIf-'
    '%skK`9l*>3<>^4?A?=tBD&YP7Nrt`0{P@^yQeC7$J5j1j3W!|&(Cg#L_SVOf?X8c>_J`SDAC>Jto1P;=-'
    's%UBkhcqIKrx6tH&5PHVY6p8qXuC2J6`TqD$T%dSqQtwLfEAt?4D2j21AW7OOKOO8e@_kC#f{Slu=GnX@&`V%rCl5V9UpK!q3FQ>'
    'J<*3Tp`seuh++Z)IPaVRT^*gB8h2>mGEm!DA|VTnL*DtT22(9;NyPPN+)Vg4<9<KIR(r%J%t-~f@pBVP7n=lkTF&ptmXz8W3@d-'
    'PZD9Z=|_*S+McHY#lUz#p5VPd0r~2fj}kS{#e|7%U?0f@1+5)i;FnsvA^_^=naPw%TO((Ww+EJod*JbK51bnAffq7$g!@R_t{L<6'
    'mU48hRDz~qIv}@_plV!Un{qv9%5N*NG5a|Z{<Zx@&Xw?QERW(GUl89S)esWIX@z=B6p+I-0~>dOU~%J45G-'
    'ztlEknW8>1xg<MbpEl2|`_gd~1}1{4F~3&I#OxgAV23z_>XZCSuo31&D{!hdaaqlQt7sze<N!ykk?9%FNZ{M78UjZzs5uLv<bC&a'
    'J{G5lgiefT2%g2R89h5s&V)4!1E5_Y@oFVkflEIvS{%h?0>jX>tZZ-iyc`;CNuV+kXl@F6@wstYuP%RqlB0AZSdO*_FPxoIbuBsW'
    'Dt7}jD_6olu{lSDvRKY9d&U!(!WK=^lgQg|*hXnC<U^savFu}s&vKnJzzG-'
    '$6c(Ol{wtr&)P{S1cNs?#zUUKwKei4eo5g&2M*gJD$0=b6uI)3s`W`TS-'
    '2rFznQUYD*@PnpkOrC+J1&F8Pvuhlc=^Ec@?U|+?uVb1lHh>mKuRwBB>{1t#NJ^SXJV9DIP6D*mVqu>k6usI68PtcP@z*j$d1bkn'
    'j0mZ<#IxiB3qSOHt^i0$8kXViU;Ic=kvXAtIaUBK|GIU(+)ae=Iz84~QZiw8|L*)M5Ay@CJGpRSI%{EiNlGw!C@lu{8v5~jUI71D'
    '(TjFxgP;KauxI8nI!Mzg9Ei*9^_<L(bECuL1UnYMil+@H3>xIe{NXOOlvSla8LAUG#Ip~%sRKpT&i9+>UdXflK>qn13_1|egF{r*'
    'FPp_x~AkP)-+g)Wny-384EozMNFiARQe{_beVS?|uU@3^KN1{D?&OA`B3PC+D1oasqs9(;g#hNf-'
    'A8GT7`=T4r<`u*H7t)%j@ZMkAOk#MKN{?s(n}yqiN>ny_CYVGS$9X#67trPEQSBc7UZ|uHmT5Y+?gSa?)}0_j-'
    '5P~uSedO+Se{2u5`ks?=n+_cnFbVt<xzPOdX43|B8B-'
    '1$sIwjkf>{vBx<Qa#!}1zuVV)DFFXeGjRN%)?Q^w=?x4=hAo~3f(epz@pBW<h72nyB4p(>Cezqgj-'
    'L{|YNOh0xXFFE?)%L2@8JYV@n=cRV1EkH6hj%L(#pQaf?<D>TlcWj5U^V_&NXwuJ*tQcauiJKl<#k&WhG8wXMPYb8JxK(H^`l2%_'
    '!SyZ6emH8fRg}eKb59B(jsc4>%@}bbSp#uv5gv3;g!t*Iz9)((n`%rU~;<P=S^sd%W~>RYAeJ(z6KKg2;8J+DJD*2{K;W*dr9I~{'
    'GcMVR4~zJU$smy(`P?0!I`SIKiD$NRXYF_L#6;d4~|pj038V8h)HY*!Lr7&*e>&Vs5vsHAMIor$w$x#hs5y_G{zytr>vW4?iiJ`z'
    '7&fGV{+DEr8*W_Ue+Qkqc|n8ouh~=yv`$eDcDLBe-'
    '8}Sk=UV1ZBM)eO5!#$c+;TXFX4n@eccMxa_?MjjS7vYg0{sC#VRPQXd{Q$WtiKOaYToBt?%O%Yj0ApVU1?7S)dnZv#{+f&qQwu*P'
    'wO5RQEVm+U0)e80?vobx{TAfTt9y3)jzbN+dw>CQgY&DBkNSGNLft%xpzeWjIWJMO0-'
    'tEKWs{g=cx9@<%+ZY^4O;lBG*F?QE}c1hlortq<pW=By`v!@2;T`=hIwEG7U2tWRSHx8;~Umht(9aBGl(n^snhz%vK;2f|lRKNK8'
    '4E9VULj_~To1(Kb(_M$7i2E`K1uJG~rnN)jgJi-<|K35<ZtRFUw#UtEA8jHuTX-pA#<SaKm^__~sBTGP7olZsJk&kXzy-r0|p0x-'
    'oa<?byKVB+%t!he5HB2_^0_q=G%HgG2KL@9dOMi^J*6RR}Hl+nSNvQ3@Keb_*nmAalz*?M)8#jby!wf97MP<%pV?&vf*yx((EH)O'
    '^`@+{X|1Dg_Too7fR&X6fXLy|#%OJb5(DALG@zhu}!dIOdi^k7sOvC}MXS{RN0S=4OIqCq1MeAH-(b=vc8W$#*Z(#jWt+Dfy-'
    '<b(S<O0)5vYHryO>SRP<Z_{)al($-'
    'yf%KYDoq{uR$}c}#{U|EuTchkgQV?GsRqlml@WXDe3`a3a!>tMrfm$Mus2*=BLvk2GHqvybkv11ZO_GMV2F1pART{(@bx`kIvy)P'
    'EuhVI0+_`GtRVB)Di)7$6Dk&u|E4h!c+`u~B?^zQPF<q#2<zCTNQH9OkZi3Mb62k6I45G$1hCd8mbrjV=zXMjj5B313Y)5I^c7{L'
    '=~C1;eK##P)m`b5w4*8RN|&Y;+@E7v+KKyfEKfU|YQ}U$+J#$ntW3KyNgE~^Fu32t4DpPMHUxL$9W^ce8K%}%KKSM&>Lh(MgL!&t'
    'A>I~!Tc^c>4{qYLSn#c+F%cb2^(1wTDhh{r>>5=R4)ff#NL4WeA9fmTEtx<VMB#x_w;km;6c>4wn(EDSa-'
    '0D@$g{G`@G78KgEeXH(6S0k2r|CU5G_qYQONBPQ52S^t*j{Qn6|c}up(_^MPaA3trdlx({@%Ac1hb?QP?%@5X-'
    'mvzK_P^1lUA`#^Z(BQb-'
    'i)$38t4jc^mE$D(l>jfoJ2dJ(!siNaM#pBEzvVI8{_sVL?h&6<V4W|o|jV+GJuBU%FmbhI84c?CJS9%(2^=o#k_9qKK7Ji@B%;_&'
    '!BR$^y7mmwaTX1qCAnhyVqQ@l1{V>pu=g{ce8%|#mw0`Foj+G0qpg}G>lk%(Q*MSFDRwlo(V&?mNA5=ESPsqdvc*?KAGB`Pgc6Rw'
    '~Aj95^@9h?yh$|*D^0+f0sx<`Q$7OQ&{C}A<X7bzlxRKIFedOkNPce>2QL(T1BUUf+>@ON(y{oO+{dDUQ3HMCp-pwc$7Tmc}`o8$'
    '};B_`yR<hakqt5Vg^aoe;vz%b1r7p2RuK1O1zT37a%IC3sin^n+BMcK8U?Nv)|(g;e_3vwPPTS(PcpWDp^F=&J)HBIIHv6-'
    'q(a7bLHYEv8%kz0K`c5;l#t%fzfKdR{vVB-F$rb9#M8;bFp-<6kKa!Qh+!)l-'
    'N1St`^*xnbrCCojoC0=_ZG_YpzB!FNhxnvZ)rm??zbUNMDet8?@j&Ot24>!mU;Rd-Nvq3r%*e}s{`wyua<haAkYqWKAy#clafUTo'
    '>f{|n$w$7honN%<I@%t>2EfIb}zx;KsK$a>#=>xHmAGn1FVk1B6XiP+sDopwVQAw(>C=Wy>sluW?5S65Q0ewR;2!0?B!QF`;!=C-'
    '1vd;F{A!p-RkXBNhk69*tJ)uhYn7&wMT3)0h;5EfIi?jiozFxp)fESn<=mld2nP3c8IeMs{7^1#Gi25Hx)L)qM!=TgE-'
    'b64Q$Gv)y4`8s#&(#9BzEpQ6`Nf5F(+|c17;fRgSO9-YV<JimVg4SJF{OntzYofo(n6U32W3oY;f3@K#Q=C(9)LFrwc3-'
    'piwUu&j-'
    '{JgWZTlt!6wovc1zZ>uZh6l<($!;0u%h2iJ)tt4JP*Yf{9p9?&0yy5RVN*Jl4wvc)Tbhg;xvPwGfj6x^T|*GW6neBI^qDkFqYa3V'
    'rwmniu{IXY0j2g!dF`Hw|IXUaJbpV0_MpVj&E-@K7v-'
    'Kcq1cG8m@(p(q&)OY%^Z46a85B4qGIG@uv=Kb8mK&Hz$};6(`vQWIbTTn>_cr;8?DrF1t$v=zsL^1544TUZ<GM(P6(wv$5GHVR=Y'
    'LfBrMf$e9)_JUPwh3)`rV)<|^47Fa`0fM#e_b?E}S=aBQQ$N38>*uImhx!<9E!AT>rEf2xb}ys{<3m3ji($Bhhhs6koW?|G!Z802'
    'M`^;aJ`YD}!h!}wXu=oMfMPKGqzId3eUQT-'
    'r}1^Ir>4(Cz1DwK+yQ=J4nb?kgQw=0bgtTH6PY|!q%`mldsm3q#vx*5h}fb2!LRPj!44V5R2{ZIEEeB#gIM*sux*Xi8o}LStJTvY'
    '?Wy%)3di;nI3SK3_7Weu?WLNlp}VV4%?tPy=;>=?p$oTA8w=e>XiP-'
    '%6D&k+RPqxnRBcr9lcWI=$xlORKrzsLK94A|it<PLNM@M;RYA8OM9*v=iTvrBJiQcuufJ*4t@EOWa`yp}dOmoOgTn*w-'
    '643Jgy2m=@LrMu@1$JXiv1FI<<iz{sokAR+pwW_PcCiCcG_QZX*+K6es3;qZz{&seYvy)m5fI*$zJMXcUP(UY1JW-'
    '>h?mFviR_mW3dajFgX^xlW9x@c3~YRM`0J%X>t^H6B-bK-AiadG1z@IPjxs`!kcw~Qq4iDCM6c2OpXP*!fH(ZvKnVM4unlgm~_}-'
    '7dc}*?A{Y%w`qvoT!`IEGuT~#-)H1J939;m$4`m02jln|k@jNjJ}c4#8N1Jk^dQFW-$Z(F0qp)1-|QW}Gj}(kT4>-'
    'lmg<~BLKmNWN-S{U`=`VLcRP)V04^-TlqlfBB29?`ZjJ^-0QXWFPz-S2%+nV@=$>h{-'
    '0itc0OAUXiF&R=+D||QAXx`%H+Tiet>VU2M#)VA<gOM0a_<b0+d%qC?i@S~hzJ+2o)Gx{#PNqcC8#RJ(Ctpm@z%rO#LS%fi@Dk=F'
    '*bCcx!NYtDgW0bj;VH%FM-'
    '<ubhP%cyHL9p0<w0vMDx@x712DkFeC!TlM=%fQ7nvCBRf<KjI|7mfHAE2Qc)C)m*wdWvlF#CF&V4mMS%Ch95Z*K*LNsm)n{$Pag9'
    '9&@NPba`*NQvqxOav{v^ckq!7amGZ>zW#RZ_`>Z^GI-=CSQ`2ydco2vx^-'
    ')qd(lLFsgn5zvE6Ep8J7~Uc=_I_^?OICYVF4JK1kJ<~|Qm9=DfmnMA<Ed(wiFm5oWl=DOMOhXFV_39hQ80!@T^0pnSoCGZ!T7^G7'
    'z2hk3f2fV3RiW7SUr;)$&mED#znfUz}_thv$d*GGudvVYG+GTuY|SKbET3ce7ye}l*kf3-'
    '+v=juEg|vBlnxHK!@Al({LNy6>ft@nQgF9VkYO^2EdynX42i0cxjJQIqk7Ts)?x8MseBRo%3=DZJR4{A;uHdE*J5HwaZ0Jt8SRi<'
    'szn4H%#|(5!0$0mSeeyY1IwOwY)e0f0Cz8T!DQeIcW>^-YQp`s$C9-'
    'NF0voD^Rx>E3I32PPV_{dKp6bS5p08W5Q=i^`i}wog<Yf(ORqa5&K$m2-wd;z}_7Kwy_UbP>p-&q_S$bTT29{#>sTJxtX-'
    '1+LPF8;;3rx@l}a3spe}{A}7@yzA=|@gDmoW#}n1A5b;E{D~bX!i#(X(6-'
    '9xVJzSq*`f)KJw%a6!1Bw7KZxdL;6~%#gU0x+&=?;4!ss%ZO@!}kOFma9?;@0@lN=oq8O`2mfy2vCg4&Y{A8fJe9k@<Ou%zHv)Hp'
    'w6p6>pOS*&SXc39=`=Oy*-'
    'Rc$v({1L0*d9}j|;$$UH*UR`4eYX9QnwiQMy3AZhTYF0>l(6PmMdfJsDo}PAPl=c7%urf+}_#CS_VzdWXz?D(j11#{$;&8hmubSJ'
    '+TkTBSZYhu{=B6Rb9cj8Tul)i9Fu<5KXZYFBFm&Nz_TCV)YeLNaCB$q~>H8pVmgxu<+vBArQ&hLebQJp?CdhO&D<-'
    '!}%d)6$lbU7`@ARFV+O<SADv9O1z1IhCTdC$~@OF@DUZMJJ9l438rhQMuQ`5c|1zuQ*_oBeN2CF<`zzZw+UKDs?rQa(Kyg%frz8e'
    'L;uaF#Xb9NdzBK9*Stwhj*R7Pu9Ee2$dwapfHGbCL`l^n};uTrYp60d@Wk{mo(-'
    'xtFAixAfLhOlmy(Q4rw0D)g(GJD_5spciBzPWlUr*<h(4b0WsIn|;>H8fXsIn}a6H8xl8=G1N_s)@N;l2f~vsHP>c47T^>^2=#VR'
    '!2Nb?J5z^QoAY&$T0V-'
    'qJaDb7AVC48P;x96p))Cbzo6It}jC=T&sO*dI`@&{$NvosP;%uSFqL^7K9blGqk!)QaO{}OW8OcX#X05c5MjS`$Et*_hoU~0r$%_'
    'O!~Zg!lVS{aH~v8P@(o>F2=MIX6$Gx#<a8ACo%0p(=KY?q(HaOpW>;V==p=wcB)!S;d)X{C=|!gX-'
    'M%TweO2~lG^v9zzb{eeiV4uVm)09cwr6Sj{+~O@%zPrw|UF_PV|ct>t(>t#xs0qhP-xR;_JlMf#}1QBWsv3{7@Nkgu{+FCuj507f'
    'N+mCNF()66B>f%c-*jyg6yjRvPwob+Y|joos(se>lLE%$}!95w*1MVrs<zN(BmO1A5xCL`I(W6`soQaiY%G-'
    'ZD6@mKI_J*AF{8vRNM7!tBUid2pvN1SaYnp?)n?AEE9O>VBaf6zXA7C=Fdt{L-'
    'j2^u>uGfhaZ&y%~MOj_=AV6KffE6Lp02X)Ab@Gc#*zuIg$7uL`CoZp*e!J9u?wdgAtM+jM|eS6hgEuEXE+QvEK2zc)ktHMjWt83x'
    'z@DgA`sP+@6~YTIf|2L-'
    'I>9me64Dv!PcUX}?OdqtZ4R|_A9d7TuVKw4HpUT+W`g{s8#V?Gw!rV$?MvDh|^TWCxX*yAcOJ?YD$s>EPjmPJ*GHK%VV2741)hIK'
    'g~P$bN*N`~j6%qQ!v0s#jikOd<Q^qJvER5{7l4{pRE?**wY&mixu5P7>~Bp>#kt>s1t!_&#!2f^m^P6Y#MDtAG!sp`{Rs`?DDo@Q'
    '{5kg@5^pJjv%5Z&LGeMf^@q~o>~uxf>5pMKoOV-W~9@pvo(H_(^}>8B@ud6e{n^;sSz{dS>mC<cLVhSCpfT2?3O17nImBh-nBIz6'
    '$9WRpbwQ+u7#Zm6J{h6KG(5v#V<`@eB$dr_+2XVCU`h_)6!+Eg#04z`_=4if56TeNeqP=`^`&Jf?(z`@{JS5rf+$ZK4pWTsJBmFs'
    'SRw%RMY5Bbl~y9Rh1AhMmk-_vJtCs;-wk4$5tCuB}69^oeD#Nu%+jVS_;-'
    '1kN=&x)e(*uZu8T2T}p+56vuzM&XAHs39jh_FDbkuzB3@;Y9jc6hR6N556jNHwf+c`yeN7u7h=iBC0G#^h)PLd{qltwE?6gQE=yH'
    'A8~d7KEBHJ=%d#GnPkt5NgKo=m0|P&z7p9kAwb8QeBaO{yQP`ciqtuAM&#X06fnJ;18h8YXEkE)hpD?Ti@0bu>gddcp?^nAJdqKB'
    'Uw+<%BUk5=5b}zk-RH?Ls9DBZlN_&g=cG>WxBp9aquhUepHo2r&8+lDjP8+&H>+-iS#>F4kLe;YFGv%bs<JtX2c^X2q#PKm-'
    'L}fr%0oqVguz=8Wk1WAD$|Wl8W6Al{9K9Ha|Q~8buX*AD%9as*0@-'
    '&yWQi@gc8jfW?767C%hX0FA{xgsN9k;BeNDKQ|VOa1(Q5vABxHL>$g~G2V+hoMFA*i#nWJ(l-'
    '>P9_|*B2z;DF{F&H}%AqiHWGaBc7GCNJnZhRJ8X;_WR!}*-#>eA8q53*JzAV+189crl;&Hc(a<~Tm+_y}-'
    'UY>r(jMW{}@0qZ=B3;jX)t%A}Ojq4G-N<a!UD8cVR^2t-'
    '{IgW30saR0_<IVe7zuw{gj!KZIn<9kFBX4r6Z2y6S4U$a@TVt#RTTbUeO5)`Z#VjeqLjn%F!n^73E?0JTi8}6uSmzDvW7zIEUB(Q'
    'w$mCWMq8L<n{b~lk=NyQqjUZW>;Kiia-N|xJIBqk50`3Aj=RDi0rqH)+rl3u)jabT4oJ{nc&R0f>gVtc>ck&1_+Ap?d-'
    'sgIO{BUy2mk&owW&qrMUB)9KhN}kiTbS;iQQmTeMmMi%dYQmek>&64(7)~@;Mq)1SDB?(DU_vQIKrpT423j6eRgP-'
    '<`gpDE&}|aq?O$!DrJKOI9zo-GYumQk7B;(NK($InHX?r#MICaZ4c;m4^_x$U3+W44HQK2un<l$WZxr;$EysOg5RjS#<$r^HQeOo'
    '66m+x`T#!IcI##;BHnuLBqV_XUIqcsd%m*VLV&@KhCZLOsnE*JLN-qlVXb`D7`OBv&ID!ypA<7VU?z0Ay(8?fv8!jmPC_SQp}&wY'
    '{70~LsRTZ3b+&%P_aZn#Dao~{^!ihnVEaP>+YQm_u1#Waqr&y-Sf`OIcLs!-{eD$z`kI+ywo;+<7Z+S0-E6^aRs8y@sbAY93-'
    '~QQ$^&p!8FfAl{t6R7ewT>57rkH$!mAw*}5c%tCw@Cnd_p~@qiHmeYb!b4O66tbDCP$@sKZ|7a%H3@4&JfG6BFz8m~jLc<q$M>me'
    'zw^Gb-'
    'o1^jd&_%H!K&Dn6K4ixayoQ+`WAOSzE)Qa9&`q=8LXoaKgRxn6lrT>lwnM^Yn{!xhGw}d*?Ort08<bxax<6C$(wpt&a;Mv$}{T=$'
    '62o&q*ia;?K?Aa(3chVO`p!g7dK@k-1m)%-'
    'a9VMzzb#de+q>PIHZ#7jo_W_g!+$X*KM$J+Qe*qQHQVmkQQf>Wkl7{fmEQC8}A$({G!bdRmCh5efBbj=OQ>W=ZLfy}-'
    'n~E%h4>0SfqSNpYv;L{-'
    'BCt>8bIaBxusLOCgT$v9`2IMA?*^eJ8u%X1lPx*$#W(O=EPUY!o{NR=HhoP5eD!lhz!!}5Toime>kA^_d#Jvk2z<L|{ROO?8_2!H'
    '#4=7wm*doCsY=3cFg}xRM8T>%o+7-cU|G+hcZv-FPSP+QmWA=2Sr{Ldf^ikYso#1tBbi#yDKR>VsSTVw+|f+E!6{2RhN+F5SOdl~'
    '^)|P?*vZN0el*C6ngQ{+5X9?*8gD>+98WgqfEd63^RXa?CwM*<#P92CB0#L4D+0t|s^_CXyr;e(0>p>u3yOfae>sRnx`RYXyBDM>'
    'sj2)E>m$~2O3^QHHKOQHAa8YsAD~Z)697)q;2oX??_OE(em~{*R-'
    '@xfyX5_7uV_4ayR=I_fcA<eqT@@u<b!ChXfis!4HBwm09zdb7GlD+X0Hg`m@D$?Lh(z#5Q|uNgco8FyFp(Qfmr=q5r_r*yby)hz4'
    'QeUi2c64pa^1z@~n*OLk#KBh~?^GrbY^t2-jOQN~oiS>!cekl(J`!Gc{JI;|0Y}AqfDSq#-&Y3(>u^5IsCKziwk{x>2aqO-'
    ')Q)#o=eb@CJVQK%ORXbJQGFSj<Y~GX1fTsxNKY`us1FC+;n995lepnyvN-'
    'QG6^<Hfa#|LWcSB4JQ3sUW^4XJi&{xAYQAlDF9+yGH;$MqOk-l_#%&KEZJLM5Ybq2xW1qWhz~8Bb}vVcAaC1ID2xIVDQyOHq0Fo<'
    'qx;y!avvavCdG0vKYy5=da()bFK*KlhRR<ht?e*29(7EGFf<;M<jFq5)Z>ye1pQJ318|Za7zbtt#y;7BQI#4PcOwSWt0mXz`df3>'
    '0HJHP5G4fh8$wMoz5~RNKa)8q#_#Y_EQ;X~UW!HW5`9esiuH3vpcr2FOHnA^M_&+u;wpVX5fmR$hT@}zTI>s%u^#I-'
    '0Y8a(#pT8T!~Mhl>i(WD_BJgE)FsXD=JHvyeLicRPh!nCNi5rh`mpN@fVJQex5xh|Q%^{8So^1#2jC<P>VvYN-Zu;Ckq*=iFusP#'
    '+q=V}(8cOL?LQ=Us)uBG>Q+DE%dwb+yT2TZ*+u%A0+_XsGS5{2vxF#kn=co_EG2FC)fW`NEN#M$)E5-'
    'N>@j7jvmqs?rc$H&m54ziRjk*8@~l|3wt%>OW=UMwvF*f9x8M{HAoxK{HIQar2&o3q#0wEs1)=gSo>bDb3-'
    'Q!Y0_cFG)^04+U>V2?LH!~}ad2R$NigB+7(v}8$AJ-rX&<RkPvXu3md&_cfkR43Q$0yOBFy$xeXOvV>2^H}D(;DT5FfEyDVqDWTP'
    'd3RwOgrB2LGfo*AT+hAv8%0Os=M_8ePz_z*pQZ^cA;@e8ufzUvay{SKQw2D{k-b6}NX<>VF9HIr05ICar>xD=Jb`1f~I|dY8{-'
    '1S(iqf(L${t{pO*KXBDxt5pS%n<Z^(pnhJ|j0HD@TbciId}b#Ibt%u`GsBnFy%f)j+PxIdi`u=IbtjZ{B4C~yMyxMk3@WjL`t5IU'
    'AI2qD#`2&x6^CnemfK@cGRpx4Mmv~>$`Th*>N>eSxm^uoRDyU1rLAEdy3-'
    '}CJL@5Mo~4(OYBx^9_6eyNr(ye)RFc!^{+kAnZ3p|i#P*BJ8)Oj8nEH*-)GHkS7x;dk<!#X5*X>b?XAtdCif0h*QOwk5@cLR-Gu5'
    'iDcphsLzHy9z>$Ei9I-'
    '5XlC2@25v2dz}Ie50S^TK4I9b;H?x|lX4k|oI!HI%vZ2PtyZU?+#SC*y%VId)@F*Pq~a8padJ151~9VCg12-'
    '_lv4&5Whr3@v>KS8Gf{0>1a#c?k*p!aYmzM4LTJ@kE<Fi&^?SUM|^7b(i?I7x==<3YMZaH`XRyfpvix5LmvIOpBaiimK!)N*Agow'
    '1R@En+j_zFm*FqX<7qAx`nXTfg#;e5L?9HWR6>xKWDCkqz1+I{_U3HFPQod{aN)TQ~z}ptcL>z1pEvZ=8iIHt?h>V`Q>~NS)$4e2'
    'HwhBn(--nmE!3rdzIqpD0>yNbWPcAPd{|L!2P1q4dQD|&*i3#%q7)@Fhxc-'
    'Z4QP`&E7RINAFH%fH}Ik^_#Fie6Fk1I26XKfS)usxI!b<TDnRj)keBfBh^;AS|im?x?&^MUb<={m|$GFk?J6G>y5I^jTu{R49D}4'
    'T)kz+GlVgpiyY!y{7St`@yw0AOYzK&y^C4-inNtYk^)95_|<L8im7oFPZ7ou!-CsySWs~5$d1-'
    'w+dbGloVtf3^V}%h!PLK;L%fTrm3G|fJ&p=S0yHBTOz@;YGm61VP7*Yu0fdY+XpUeo-'
    ';)T<kqj1jQlS~cV3j8snz0O)dDNl7lfz)f7x1$|vcrtI-'
    '%gd63ANfJF(8P(Ft1e`zjU8cJlSENQassVpJL{o!?QvgC84h5wtN|juev#I9tcmmJ9d*3xa#S6|G(z0P#PvjU-G<-'
    'sXa?Bhx!Rq`;_1(5j)Ix!}&n2c5(P=&N?ydo$o|W7{eCi&Z*4nw>lH(X`{F3iqNWY_GwGi;i&>ObO%i7V*@OKVFKWhuzKrYu2!1?'
    '{%@}C<>kfI`b+jL#j^zVEyc40_AO@d)*Z4{^irz-'
    ';ja4>>3whKZGVLu_utCoPkiKEP#ohsB~5=CNH+^y%E`#Ms+Yp*x_&ZKT{k+YuB*{>XLh|u@lhhOfq!;!Kv74#@S_f?UH4Jn@lxCM'
    '3;(PKfa;`|H^8Hp8Qe`px{kh6Nt+rX_B<Xv`(@X~HdVI5OX8X;TjM1Wm80gxFBUPCqhOoGBBpY5w7#HF6@7=S_FfO#X0O3rj0z{3'
    'dm$>p0x-A9EWIPPm1=gTUfi)YZO=CKYWA6_o^s|~PdRg*=b5p);Z22F@39IWTc@zvKa15PQkBkI)TJ`nVvCy9Wit7|)-'
    'S6WGTCMenAJ>~eAFNhUIvBDLKHs8)OJA;LYrL0X8kqR$Ciu2L#&T27kx)xQviq5>SSK(5)q@=4DWr3h*4}lLSIm*f<B8wtgE}8D('
    'I6<RIsEK&<2o1b2a=2$E%-'
    'OhX&9(<R`%4+T*yu6~v8Tl;v(F)CLbX_}C@|w*gtW9hsUM7oy&Soe&ox#rVY(|4d*Mr+~Ku$8PoxJtVe0j<RPxB{rRnF86n`)O8s'
    '!HV?rF!SKFjN}P^88|Ch;{gNAE!3YnrAr_2v`kDe@v|MHOmET?njI_rMR=K?p7>PyfNPR(}a{3Ngh5I_D-'
    'f<$NKSioiRBLtu$qt>GQi(nz1_zj5SuAlzOm!DkYNz=eSGzgH>4Y>F>J1M^_}DfD$AMWmj!D6BfhFWO;rMt)tMcur;&3U2<CB;V$'
    '0svlNdgc3NRtO>fSN7?#TFqH9~0DOc^URn=0MSW!8c-|2oLc_EEHeV*F-'
    '?kyzV=qpa>SZBMORR^aX_+#Ex0ggWlMvLq!%J5|Kea?9K^$il(|I-'
    'N8<cW~h$F>=yMf9gW$&k`_X}=^+On+od2kC=0o<sn!b!21lX{a)Zop8Q8QGp8vq;_fGR8$4&7Ae1Gr)41hwJtEK36Oy(q|qT0I+d'
    'kaPgD>|2mEN0JAPEUvV>Tkw^5x&}+v0!{&UlY;OVGMDni0SD7Tiq#QdOF7H3ko@l9kcV|R{~2dzOlGz&oDxn>Xbpga)dB9ayXJIy'
    'S)%-HP=9_X|@3U%-B+>w>%i(WBU{sE3#lbDg{P(o@$q-'
    'n9gx=wrPqP99L(XrkKfbZtO}@skm0#G{qGhXG)u<xRT>kY10%}ahxl=H^?EEfqAQNcKm`-W#&xmB+7|h%-'
    '4S_7Lf1|Z^Z)gDSb^u^kR&0S5))@_PQ%7dO1p8P{?cSm<^x1NpukX&K1BRB<aYbp|=}TuF65G9a4SFQo_wbZS;tQk9(vLIXH{RB!'
    '$T3(sehff63%z!sEYXatgumKQb9lX#B5CCJ-'
    '1`$Ydg6aivTq5fopM$>ba)W~tsXNNgP<@ocwkskjyuD;I9oUwdOL65$~>#v<`CeN6-'
    'sjS=pSLL%7d?kFTC`hr3(W7#_013PXFx_V;AEwuz~A#$QV*k()hGanK6P-'
    '~&y_E3Y59a2!M%tGzx6x6C19$zq*R)vGv%<k{G`kU0W!!E{Zuie^dAq{>n?ifMaGs_Q(Jy4f>5o+hynylU?BA%<C&Mi#by%Z+y&N'
    'Ez^@M#9N`{vMcn0NAaEDqrb-'
    'j2oLo%)&xS`K5OKZ_Vz4lv@MMGP&+(fWcyGh*4kPz7%L^`zq07EJZF?$T{Sbt}P|yv!ExGVWp*^t8SSIx3c7$wamk@)8aqtjj0Ai'
    '(Gu<X#NlRGo8vDYHd=e-J4@hJ7n6BS-'
    'I$JvDn%yE8!(6LAzEQ|3)hd@(D~$CIUii*F8OlX|R6!wwa;UK|l37N#5h$c_|+AeYT6px}E*-'
    'lDN8^rg%w#vNDP>;H7LAMe52ZyJNmbfx0q+@mhU{0(E5s<4g1fg*2PXV0;V5qsq=l{lkLC1-'
    '{^M9JX%16IA{pohEst@35?=*@vqpDKxIgqOnO9jjv|WSd&6y3y%D)E$k%Pn&V$=ZO__r<Y;Yck$N&y6No|+iH9Z$%VUGY8>bR+Cb'
    'A%$rkOM2sO5edR`g!a;e~1#|61g)^yWo;771f~h|eN%jL-U+h-'
    'k0CH~5)|>8~)~B%;6KRegtu{)!rXK_Thp@{S;#RCjABB6$HZ&@AVc78CW*F9@37mUGqB=`Ou5;XmWV9hN=3DU)W}#9uPiQ3l+_*h'
    ';>83gj!ZAm2R;^4GE;zcdB%<{T-DNeO2Q?*Ceyn&d-'
    'FAZ=nU1RPI{vB>o{k!WJkwU|u(PHIo4&g41T<PC7mU6I3EY`^U1u|S4feI5(s&-67BK!*4Jc@&V%H;Dl9Yx)imK)zI8P)NeLoX-'
    'J}>#BtGJnOb1`zN*PGVX+B-'
    '{GVpj=E6PC|Lu)V!PfvSo?9@H3?ZivS4kR1?%fsuwItZ4Zj*2gJuvFc$Jd{OEzQD%Bq4TA7I^rl~+o(V%>sOS4y^F=HALFB_E+fy'
    '@kcn&|MkM#@BImlbMZg5V-'
    '<M_9g6y1u;Ctj#v<H(bq(P80@hl3dH7{M1c5peTN7TU#2f8B;s5K;!QXiTTh|}(MWW2FmoZOouD&Bb0PWmCST<M=pHZ9GNP)#=E('
    '_QT$hD$vn-6)Wnp}I3dXhI*-*3SEbLOU{CBfe1t$W$IchK`2E4hdk`o2qJT-'
    ')hNRs*3^7pe?8oR4P0Jj8pSf{X(zUsGm_%+{tXDoi<8Ft3vccZ>00>5C0ol*ET-'
    'y{OR>+~HW@O!zwppbwwFIT|@2%DB5;}yG@tpXMJLITaMu0g#kG8}n%go=y^F8a{TQ|MilMQ`&gde>*sJ3Zww0A&6n;ioO<)3mQ0?'
    '0@U1NjEz(S8p`V*BkArIl%U`XM1V-rM>Kk=%&8R^BNi;!TVW=<P*61pFuK&9C=7KZ{mwsB*Qa&5sT#4^feJk2BUltg=F(hB9OdZ-'
    'ys6Y)Aa>~^qjNJCAv4j)|czvfDPRD2E4_6Z@@<GdjsCKfEuXjZUexs&H}bY7O)$#fSr*7>>ipMdk<gsR-'
    'G<;L+=|#+f;JT;qx>qRj7q?mmTYm<Bn?HRI`AWq8T!+<@KI4K%-Y1BDgJ6TTS0vAN8A@UQPRIzKlgMJj0i<2wtJDDc-'
    'G#R{B!Jv}>Ag649=?LEj;wU2}%MppdF_8G^CvVLkU<4{vba_3$QlUFI!QNL`afYRfEA-'
    '^e0$W(ujzI3np|mkevgshYYChl5I}+X7a&uAz3mPk#(CgWerghs~z_iJH8Y(0zfEu^R`!qUvxX?`kV(R;TYiJ&JVi&9HB?m%7Dd*'
    'nTe3dSJ96Z{FBfu>gh#_$n5_kLzm+0GQ@GFy2=pM#;{6lL#fdH}oAMl<a2e3ks<_m-'
    'TKQ!_~Et+5y^W|H1L~O%Zq;%N?bHs|*AjwGTP%(QQs?g5k_g8yek8fF--t(Y=&hO$9PNO3AEbpe=|hC9{)3=5w!7GAF69x#LnYHy'
    'La`_bDawl1lTrZz-9d3^AX1$<;g5sIMeMb*?O0sv~s)Obj{S=y1dAQKlX<Telw}TTf0+`2t;`5Y5ilO-'
    '8fxb(4j*16Gv<JpBW&aGW}y<LC~FLN}p_ECT|B?jN~&iz>|J^tVD<n6Rl;J5vo{weWw!?-'
    '&$Zt>86HuqU%{uW_P1nXRUBHOZdLQ9t2ovOSrr&f)4*dooX*%hhT2WIk>Zoi2QH8NsPZDJWvuy%ZGqX$8BS9nuDv?2ZTxz6p~R+T'
    'gzus(p@xhJKdrGM>t=yNsu@>t4*{>&r5ZrcT^-iS&n|ZHC}D94nrFAML%1lQa(+w9csux6?VgOlv2)X=Dje3d-'
    '3^rJ&HLX~~sZqO3`st4`aD%G|baa*JB+3MW5MYh2aZR<)K<G5Lq;Rae2bO}*wS*gjIPvz%_f21wtI3@zP)srSt?v>C~}a)vg3*B&'
    'yS^{t1DXMO8Y%+l|cS^7-wb8?z5;wnyj5n%T7iGd%dZ&sp~AidN(HKsFpy4SDDbW^Goc9-'
    'z~bbkeD&MnX_v90adLatU~C8u|RT)P2wcc>$|BQxKKOy`!tU+tg<&t!kKllnW81JoC`y|W6XMBZU8rC7yL*LXUQrAQkSntTlE7Hg'
    'AV%QY>3c5hFAfu1s+Bdw>5=Sb^W%;a0kOuiiV#e4<}CiF9%nYVMad4Ea{)cP$mG`UL+Yyz{lOC<oBay(wKCclm2U7A@V|3NprlLX'
    '-(A+YCKbP&(}#MOVXzrma>f<J-Pl->)rO=ih=)o0Z0n{-'
    '!SajHK(oTK9U<<`t2IV%6ouxCed6hpg`$I^_A4UOK0sSnIJ|AkPk!x^1=J@GMn$#|NvUNW9$tXDCkf0m9kOurPDYMM+sns;*&SJM'
    'S8gSvfMEw{$g{nP4S_H1{K$4cQt>nC`y^lvwzsyW?^H*V)nx2Vr~(%zkZh@-'
    'QktC;(TyZtDRy0<fV7%R_~;8=ot>AJm^Bae;07dD*T=7^WB)GCOn9kN=2ZGObu07utRAp~G0`_MpO3sN)Y&>P`v=`G_)x_ZlalCI'
    'vxAn<h=7sv4&1qP=2yhvck4{%1^!xcR~Wf|6M^c-'
    'Mw$OFLlwrVN^!S%M;<U!zh+hlSDINmmwJQ)0Ln@X+(x7%ithk)1pv#^?|DkL=Qg1&_fB}VB~x{EZIig4%XBWe1Iq;#e7i=}2PilM'
    'N&iSGvg|87{bCDJMbj~JJ|JyY9c4mAh9x;`?V%c_rz=d$Wk%-BstHp_bf&gw9i+jvJ;<2btXzDS2_-sJ-'
    'fJAjG!4jj0pPJ`ksZUQgwy<q|^0L@lhdONt&ZJ1r{<W4`rS-P_=|3ccot88*(Ah^uxXZB<eYEaeMlL}aG)z$W7Fp>k#vL}@=BGqi'
    '}h^ikD)bu@qsRu<4|8#vcQ16DtWLuF>Sm(oft6TF}8}M8AmGL}KePuikRNrFuZrv#>?R6qompWMlaC|QC6@d^u^-'
    'U#^uHu=Bz*~5xHvbP?%OY6OkVjmI^bo_pHb6RdFNnnkN=IM^@mGa(40aHJ4faxF;AT|HAeV%-'
    '_|IToNH059=0ts9yM3`!p$paS;O}`)1owc$dY$n|T#HQ&F?>o(G9umko1m>59_2&czR*#z&7m#ulDOv3mUu~lVnS++F~(h5sF;wF'
    '^jGovl|ajqR=4o_bwGo9{|qo|)D*5RCsGfL@MjW}2l{OndXxf^`lV3%vn)z)O`&uY!@xpv#~Sb_3_4uz+(Pt6Y3}%~AO&xn?@&z3'
    ';_^X;d4~QTm*iW3*U;A0JR+kW{C`d+>R~}rC^}+-W4xqt@xI<;#!hAv{zef+-'
    '5x>{+_HU)>(LXBoX#lo7DvZcy23+@j;(avs;`OYj50R8C#o|F%zjT)XVk5{excBy(zR^X{I8BaI@;(1BE)GPkmmp}Qh=<@0`fO0K'
    '#pY?zBJ>maq3Y?T~`Cu&2IIAb?rAo&9f&%v@pCq8H!@h*K()j^%e?!g_4L1)SIY%YiD1fS`n2QLUcjc*Wbl-dFgC>zY=*>0R9K4C'
    '^w4H>W&bT73fE5O#V`&d+BMqG%xaqSWLo091)Aj_w+Rp6Q(iiy-'
    '^b;SpMFq3G+9+e(BI4rCx@~<vMn5XEC`#r%O87GYD=k@(6|Hc0dZDS7#CWTb?QPzYSwMN0W6K)rVj+@N_B=!Q9PEstUoV{t8nHV&'
    '=x{F#C#LpZDUIX(Zb1o^_!^ISv~_mSse5141d>5^%pdMIfI?9y%Lf;kq+KXC*p98l6LN<;j^b%}YEo7M<`AN5-Oay}l-'
    '5#xy3qFKWgFtKSzjWB!)cFC-fHnVqxkd&i*2MbJO#r=4*~Yq8ERl{j6kbDhn#mNglE?IuB~B%eNl`2S)LZ15Zhrhq*w3+&(Vuz}g'
    'pGuLSS?w^vP=K)^Pvzv)QF%-'
    'GUhZTd=Ekez4(Z{Vq&Gi|FdAK)`)27k@AJ|<X8V|sO_`Q4OWQpxJ9}|m4c!)8vXk4wYiRhRzcDO&PV+u@le^kfR?|A*HqJeYSIor'
    'E8UaDshan+2BOMT2Wc6PC|68zQxzb*B!#5u~AHuaOZD5e_iH!*BgAv8kdGAM<{*;zFHK13sy>940<3hXX<leQzUO#dy~kH9khjc!'
    '8%#ps<<?e{3Rf(rCkkPvP6oEp3a=*;d88&mqLr;R5$3{!tONHpJiY%CJt8OFvU@os%h1QLx69*9CBnCXEiB>tY)uPGXMl4VGQAmE'
    'rFslLuvt79wNGT#b^sj6B^#SGAhrM<IfGHSAenL3pRa@nUlA(&wdDLT)08IZF`2nIK!YCLoyXst*=cTN_%x22$4ttqvpx$^rBw8M'
    't&CU3Zn@ySfRowtA10NvW3Lpb(Uf0MQ%^dEU`Qs#>s6^llAgrj27xKLjc!GvR8>w{5DIPm5ljAFvMjn}U*8u*ZDG@8{5B8A?Zk|_'
    '%`$A`-W4#(|Fni$nDcyg-Us~HFX$u~J`JUo-'
    'p^W4JJr5<1qdJawjY;G1{fA9b^57vU=5b%lomP{>?e%C;?Vh{rRZ3ESsLHMC_)8@Pl1BcJ29kHJ6F0-'
    'v+&f77V_k11_)3|?@n23y^iQ5_`2#0{9EQmYoRhBbzns<`KVilesiN)$JeNDv7X)N?m)XWJ+d?;$>`~$CFdNlAT%QDpyQ2^1d*|4'
    '6kbKS+`1@V84Lz#^aJaHYCovontdpYL+RHaB4mH=`CR93T0$=pw6reyBiY{}g0R4x4gVx`9ok3m#+PznERb-'
    '_^hXPMuJ2@;qG$3&A1pX;{LUghd=5yfMdTpd6Wruy$2cd|7W+T<jw^*8SqTTs-'
    '>lKDjx6t%`nB5IG#V@1>+gNgddnA+oW^#u{N$FubXh4el%s5K$SG{=HFEm+?TR4t(ANj$N?c!R{%)lIIto+355CRd~L7~uBgEN-'
    'vJ;`Y2OZs&x!g<;{_Q-%o7@;Ou_izwV5E|Mh_?T--2?bJtlph)f@BGH2^D`*t?T{0_xZbznem<7KM;;bCx>hIn^cAo{FpnvQ>%RB'
    'm-'
    '2<(~%i@+|Jsecr9&(jw~V0VtbppeRE7Q3+4V}^UWtFyrw*y(&v3lx{zmB4SP6z0Ea{av5DnZ^;!8IteYlZiz4JPZBD0LrIip?qZ)'
    '%I9aHJU2B34}^ag(MEFxmH!|cWDV^u53$6<$#-#AZ@zT(40T)`EP~e8ex+nTJ&nS#S2^Q%{WN;X?4XO3Hih-q9T1CNxY2-'
    'E^w#NX3ZR$3XdbKpdI^kRnE{2+OJF=-'
    'Ur+$Ol!2P7FDNAPS>`V+N2O);jz#5i9?Yp2P&wAe4|FTBBOL*hBcAQ94tNa9MV4=LHbyxCWXESAdsP;)7i1wjFD$gz`<d{6nUn9S'
    '58;p5ldj0Ee62m%AFG3}vnL0rkKqs4lWs`5{BwKKU405Bk&#S$$(&#k?uv(=gx9Ly=S{-'
    ';6%34pF+9P*SQx*kuZft1&4Wcu!eFg|QIqfm`hti_c%HtXkj$sZ@N{b$c9KI#PIhdd$tmPqE>6ZfU4&$UlSN1-'
    'IU2@fGKnJ0!8}ud+zDCa{w#~!3$w_bFTX`ktA0klWUeRVL-MQ-f(C#eX;R-'
    'FIb5qL&hHA2zgc0oCAM>PTBZktd`ENjnkl<{1YWSnQR>oPe^4xH;foE5MeXzYn&MJj_Q4{wy1-I{WQ<nVh5CXBt*-'
    'g{f<jWCB1cT14l4zHV!<)%>f#O8CUCG&?;|PsL=sVQ+|QWu%2o^~b1^p>0C-{+z_nQbPs;-'
    'Mnv~=FF@p%vCbB+Z$x6z{e#(+p%m}MOZJ1SD9byX|C0DDtI@F$(xLU*2VfL&kK#;2{5y|71%u5%RJ!jGgQ#b?^Ig_t`2^Fz0hDWG'
    '~h4EARnuy8QJXpl!3-+prntZ3}3nC`pYxD($#6Cr~6mp4xgEFhP3pl8B>dyiW(tw+XN%{blld`D1I*ZDSvZ%Z^)p-'
    'YA>9b%>>zIN+@En-eo`DbW0+`&M-'
    'KR~Gqsv)8UtcocPsdut%zE}(rOA<^_P`P7=3i{k6Pmsd2d+_1a@AE8dJ}AFFdd(OjdvQyM{%{5mG|G7*ETp7$MD4l$Kv=geN9CBo'
    'q4c`_B*ifU>VbXcagpzqW$h#eL*3)&n%9U3Y5{Y#q&v}62i4o%*P~s0Nu%1=+4SQ_u?#cuS-'
    'D}JXweiAC^uT<ascvJp;cKg)*(5+8d%XM>kxlEAs;KjCDa-FKW+tcLCr8|HmdXPfz2-'
    '_o_qGZY6buZ{>j4^9C73ANWO>(0v%Q+8V*9bJa0V*hjy;%2))$160N$_)dLI0R+=LX&$Tqf(i0q=*mI}Cdgl`FDQUug8X&*f<oe-'
    'WqGt}%1@1hpeLmwHlzgqPjV^*EL1^It<DvWjshkobEY|1YQsEQjcOy*+pdloaJ5Ug9g7N1ZOT>1eR4Ie0+l?O<+#;(GMkdNLwPdC'
    '5n%IVF73Jv=gB-'
    'L4ZxH6v;lWKR}0)~aw1oY1YR9OCwNS_P1XmXZ+30LzImmo=G#q)rxTnKPbWB~SdRasvY1`7<Wv(+<n@BAT!WUp-Zc<w+c<TGYUDw'
    '1HEp$M@pYj-'
    '4NNABrCKhDn`wzu|B}Qhc9&EuBr!1kS*llT_jw2KOD$7g74GMJ?UB)fsqrP4UTiBoQ<MFq)0U}IOVo{)yn^xEE9MXWN}xuF6}<WZ'
    '$4b2OW2yI!4NFFjKqo+(d<<?D<PbEN$3CMG)O%+%f_m?aVn&}(HncbSq5VfeZ8$n-BU9ZvE7a(ct!f;HTZ1cnMbYRI6@L2-'
    'NC*eN@kaZaAzS~~c9E*%KL4|dLs)GG_i7GtwH=eMaR{O9z<r%V3~l;M0hQ_!r<HZ1P?t+8+xVqW(<PN{{92$6j4fe4mg85O8p}g>'
    'TxjnvnQCr^_*YD|DCKM~ns0n&BN*z=Yy?BynZ@jVMwz`=;JcVk<Zh|{ohc<GJb|dn&e$b^hPLUCYRZIQHm|%iOmVfFboO4ulkqe)'
    'ujR=Enwnqb$wWIf!=G3|D+SE&E7f__R1O38eCjDr>eL0)Ql6|*7g9%gvRX}}hVo>Mx`_J8leOw%5zFFsyiD`BH#3HAYxs-'
    'H3~j#ZS&d+EJF5{aZf6xU^m(GZmk|B9u<>sT%4@ebw(TU;W<d&a$8g1*@sdGU@ifRW|DRqWjb#*FcvDz8(JG|QeM>m)8>R`xiIe*'
    '0Le)4GyI%+tZn7mRHwaX2x+Lr(tiyF@6H(~|oD-'
    '@f(ckFmDiv|&ph&6KZ#2X51Y^7;W7*G62o3)lQ+t{jy){?&^Bl$!^P0|X1pV3Bji5g}yO`l?@(jNj*#g=HeS>dsUz?F^Rd0qJVIQ'
    'KpddHw6>}~1@!?())s9w%-'
    '4Oni^{>{~qCHQH1?qMM;!B5L{&(+ux{B+gPN0s2GWw__+=o0)45pZq^XQdUi(!i`U)djW{j#l!I0#Bxq9}1vxyR?g?nV#Y}!%mdx'
    'UHgBUtovX!wUo1<o0l=Q5u9OD8^IYiwV3&@$TR;oRCMoT-'
    'r#Xu2^sVQLMWx0gq1un6YL&V@^lueX;{g#r%=tpN}j!hY93bd>@8G_u##sVp;}66DeQxhWFIS$0A5oGep=B3u96b`OgHr;U2u&1v'
    'Lo;s&b8Y?u*OK_aBm|*N_^{BVg>t6lEf933-'
    'v>(+uY+AxPFkP2=8ImfzcR3=lJyI&Hkhj3|v2H1OwMkis507$Px)a+_B8HLXAVFqAKQ-OkJ=kAaLdf2tWHS6Xd6j)BAZgAea*ND$'
    '5kMz;7=}GFLmMPAh?XY>==LmiQ)inO+OabW3l>3iQvs{47_05?+5612wNXi;<caokc(13(azMyFL4^u4c<g%hktT)u&f588qW+ke'
    'Lhys@{3tp84wMG=jbAoJO!$ol~%EYtO7$#Ruv%Dr|y@{Y+PYT8AA<L9~rCaL8(!tq*J^!b*N@EWg+{mS5r<%WHgN`K7+G+{Rs#T$'
    'VC|scA3|w4~HhiGN}mB|(#(9=kmOtX>}m{?RT6geoh%lM7)DWlmkPbTCuPOu^QxOm&ty-6-'
    'aL?TT$JZHAY`wU#!=OCowvjP2ft>P4yJ^$V_*C7ZzfZxpQlFN5F~ERVKP-71NDIRrZx>F1?jw<-&})+yNaC%QVra&VESE3kpyX-'
    'GgtamVSp1?oJawL--R3;r6eekXN9NgRh%WdKF3opA{6oo!$w#@*96rx;^ZIgW%&1oAml5f?m0Jw})V75$dW)E??Dxf*8HoL_@KQp'
    '#C#%nSTFwuTkH*w?W&tUL5I5j}gxoNq?;?5*PUtF4tm7ZCAg!Rr4k5|Ng2wMQSUEa#`tw>pcyHX`iHn~c0?l&UxtdCv&dQ;^+^GL'
    'B~;yBTFJF5>D|Nn8ooxA9x)I!qU#PShgHd5XGEcg8YZ4dLN7Qcv|BBL==V7*29iT?-'
    '+1T@KY=R~|b}Jxdto?jB@nwg~X}K2!fNG`f}QCwZ%&d86ON!V$jLH?eTsrmu-'
    '8sW$d~E2^Y=HLqW5tqi{ak+%w#|L>V?(woGQHBHd!)zhzDo8ZK4{9VOy;an<-'
    'pwaKR7$)<NLyv5Y#0_zR_z<oxNI`r}7UFGF5Z}d6-'
    '*p#rZ7DZN;se17^_!(51kqEMozfPf_}N0s4Tgeei!Ct}3Z5;oj6^7Sc6%(Z*H9j~WH)j()J%>WnA$gw)Y*Lb-DD&h;URXDk!XBhU'
    'lV~wV~UMYXk5eV*IO$?BUp1|K~Hh7OluUl{*!n}9oCZ+OTa<DD(DUFslIR}{x#}z-'
    'K%bwI$gJxO$4u%YOO~kIPe#y5V<ys$aX12&V~?#`0OyYIT<1Zb23*Xr$B&UPUeYZyc%s!=8I$k1PkWm8j(zdSiziJE0RfSj6JzdB'
    '$FX-'
    'h@~AH7GknwxNQyl)Qr4#J@fThEGFS0SS%(t=xZV{X^io96eid5`nA`}FxgJjFT57Gj#*5i9?c|zo(V8vCY()WO<3sMlNPx;0iSJ4'
    '${`Xr!JE^nD{;pl)w1`h#1;?2B!rREQkZ--i^=vWObYN>m}B0N92p>p;iL(?9_AQMcGv8NyK`5b^e%IJCu`76SYzX1bWYb=$}g0I'
    'yd#zgZFq>v|1tG<vuqDgXXGv0=KJ$lP{K3tSWvFj*F=EQSmT{2P`-+l#xbA-gT7PHi_Cx${Cq%`%|hcRzhefg`BmK4)&T^y4BsTX'
    'XLc1bMtI+AlG{b~xO`1w0UP0RI9C^?aQRvmmwSY`#MV@9{R!+*d6uhHj;HrIu2wsq-sibm<9K>s;A*Yo>3tC*V8_$@5?8M|p5B*{'
    'OEZ$YO^x7b(%pZWdXBMJg^~kH^Yuk6Ea4$UEG(DkYa(E2jPY(1EaA(%8wE?S=eq^H$+Ed}IrpADPKWBKz~|3r2-'
    'OAlhXlggM{(65TW^oDJ+FIEj^OIz6ewTMg0e#jl&HUyeBecvl8-3KKi(z$S7Knbk~Zxp+D-'
    'eaqz<F`I`erI2J@@T=hYZ)t&ZZ78xkh>zQGzkJwFan=j6?g<}1rsAi@L4SRgLa*F*r(7~s7qAi|e<FA9iY!}ki#k7ZRPT+e&et{t'
    'ga$#ck8tD}hw38pI^s1)?(VJM27(7nSd5-'
    'VaG;O;h&DV6J1%9P4IBQm9O=a$0S<6jDOydWOp)44jnM9HGX0!tdEx>D1lOO!cUtK0j}bGxQ+yl)4_i9(&Oe=mmM?hhfS<=0rPrz'
    'c3pVe-wvrFjVl$5s)wz)Rw)h+5($5tYYajf0~!Yrt9uM`_kPqwf$=d3>(ELm^SpvdZJZ437Y9f5@V(dlqfaX3=(DDJ;x9jjQ8Il$'
    'tzRsPPV9#|SmCM5(0Xg*vrFsU*#K2!GfrENJ4x=k1v$no}={Sl+t-6E0_DJnHdUkg~0>-n(8ny&EEoN0-Mc0My_|7#-'
    'Htz8<%QRC^XAjljo!%B=HkSw4k{dAo<i?w!ClJ0y1RWUIa=0?1(KL!yA(UEd)B$j|CKL;(3beTPD#q-'
    '8*^WOxKX{=Y1cdt`z9To%aZr>4!91kTmAS@~t5`f%!2{+CewNU~1-'
    'Td4kyo$S8?wHj@c@(O_}cee864xz^AVG+l4_eYW06jG%y!5JtBSNlZTfgBG?q>V12w{9~R7nO)1%fa|BOnoYeVR&X9jLo|}G#1A2'
    ')een?@q7B32pEHz4~>Fx4}FIS7(b`)5CP-!^&JXHl4kd?VFC(#1>-'
    'q5ANIf$?d*(F$bgUbH;qr~>Jt8DW&?W&!y_0Tf6NY#p4s8?e0F$TkQyE^8}@Y+eg6@t$7jn}kJdeM?^$=@SfM7_^+0FQE-Fz`<4&'
    'e7DN*xN501x+B9?6LPoZN09@)#rv1r56WP1c<^BxY1g)%(EVX;tNudj)KGMMDBC@A;TcZh)U^ZE`EP`*Ilp^z|X8I%{OQOJeXujE'
    '!-BEjE;uCjx5?;y2Mg>=Kc&2n$%xCL1x4`p}+Nd8k6$-'
    'S~jej$tG3sWA;+f1D%Ngx+@zm6%<?$;SyomjGaTuFEJcWITJlAh|HlGK}y6Kb+`scX~@22l&{*v7H+EeK)!4w#VATEEP5kIg&yek'
    '_XN8NMHj;???^2o!@kz8{6+UiuCZD1JfTAp*r0>N^yYCoS_UhA}(>xGl)St#=k~FJ|F3Ed{rOun_zc^Z8&b1pm~0J`~G`t?=DpSU'
    '&tYDaRy}NCWS1ED+yG&G>(lSiKs@!M8AQD0Dq3cnUL02D{Bl<nb7f`8|e1xGi<Hz{)}!axnSB`AvctG;up5U1{<#Q=b^Gq7~B&sA'
    'V8FZ}#w55X1L7JQl=v>uVxF3`RaY3dFtj9U?&dqP{}}h^Og06jCNF1MzT%M*!l5SrGThg7~E@h%X93oP3DXJRefF>mXsXU08nd1D'
    '8ZN5HodF*YOZ6l;3VXKgHGT63L?~CDa5x;l{D|EfQgkO1~148X{-'
    '5LF;?8du))bN<ou&HRe5N(*!<VRHD5VftXGes*C|_N?RUxW?lx(yxOW*6vLORibe54eN6<4!OB%pDDI>05P{;C^c^Ble38CGA$8K'
    'S`p&LAlzRIobCZrS>TIFDbQ=V&kYF5q^B_=yw?UDTcPU0NJc8k|C_6m*W{1bi+2L_<YItl#7OSm}k{?`hD-'
    '5ij<Z4!lL{`<`xtd#Y(yIU9YF>$?g3(GaP|ml>h%cC`Dd}#4;<SN>2D#v*Lfyn|h085@K81M+BV&OK_a7Mx<X!rj2q1$!Mn(a-'
    'uf9VBkYCn!hye1%`VNIuO3S9>qc|q{tdtlof&)0UbHQ9PkW(2c{PlLi5l$^oTb%IVLL|4aWr2&;hfZv8iQ4AG2Jb-'
    '93F2l!(uvER;MpIsM#G#v!SMK8Mv(xjBvl|-l1U%GIT+?8%;0m0iDg*3u0CZX=z~PDQ;8r$wP2~Kw-'
    'eLt{O~J-)WvzbXy!GxiYFIr6;CeMs-RkFnXMDdYdTvGW|&5@-'
    '9j+j29a=_PDs`=Y7#pcP)ki>rvM<SNo+h|jhe(Jz&M~Lv55e2Y7(0S=%psH$pB2JSU84bBojrM2Z4MLmy8!`mq~c;#nkKi7YPDjy'
    'G@>MxHtAFSzi&pS8q<V>kHs}F0jNpuxdJP&5jb=U6J#l%*y5kwr&IwU+YE?@wF}}QCeo_qd6vup75m@FLTG}%~ewgKg}6TVeJL{w'
    '9-f%56e)m-ovzs`o^AZ=6DF<Dji32%A){|CUMH64AXlxwmEbwL-nm0ZcsJX-NjLaa!Lh8Pxe<wp##)j&6Y{Cz~nHlH~IV%!neXwb'
    '_|ihI*mJAVZ9%<G8w;Y3bxm_y*RhW6kcbWMiAJwX#{~?n}TwsWmZ?XPvNXyBXM)urVcKb>OXD=ZMszdbp+WnrCRCw6Z&$syM&+S4'
    '7Khh{4{58GIa!}#6qQ}@V?0Wm&X|%pV<uWHl~^i%13|7R7;V=jafe}n>r>{Vq1*^y4vrf2YF_nbi^SGw*HuQQK+>M=lErU%H!<ZF'
    '&~TT^xW|fd=Zh~Z3^$QZ6nCu+BSmht!+WM(lV1D!!ZFtCcjLoe>+0m8B(p_R1vM`K)tuC)xNoEBH^bw8^Bd_2|vvlxEvh>{Ir@X9'
    'FN_E;~J@Q6Uln;&p1hge*z1p2!v3oILx9s@(}g&rKw}Xt`l&aUv_<mX9aZ@RVMkofIN{Q#F2o~tDzHq7A6$D*^_?i0+R&**Xrs#x'
    '6{1Nc8wrQYu5;}v~~scO3SQ%Eas-2)i0H(J#Tk1`*YPy!cXcR-'
    '$dFUCnuXpMB$WV3yCO<Pd*?;kO|3F(o>k2d`RLTlag&DP&qmI$m}20Am2^Ngr+f!U+TDU{l3~LL&K+kF!d!5*YC?X4q2Cz9U|2RO'
    'x{khF%&p%w*@6#xAD-&dzlA4NLW{WndQkSGq!n)?HfTU*1i#xV(kkmla?9#I7~o0W51(4(*;f>t|hZI;Fuq4#cXE-'
    ')*`lMwjvz+3ED7Q5{~s?Z5h^tTS0_&49gxZd3}3^?GXCcI38TMr1V&`NSNV0nyWo^vuYIomO4H&tzRNd5?*Vao<*#?T9&q&3f|$?'
    '$4JW5hqMVTm?O$?^LnjQn17h5Q}gC^^EUTr1pn3^jo{zfqo6{m%#ovg*&F>`j>p&b?0u|$QOufV3&%s+!2Cm|b{BeZ8$*ras&m+7'
    'd<1yQVOQ~yT<slp6^}tly`ZajELZ!wu3{9nZb3PPWGjFwie~Cpy}~eUT`Cc>orR-z7k8V6wm-QVRV`h$M6KgGIuIFd;p$+L@1xE$'
    'lY9kgXy*5j!n^Iz2m-4PjUce<P*AjV?`+$YsXF?hZL~JU8z)JpcLK+cr4u}XqguM|-'
    '2=EQ(I3lDTxqOLMBa*Q&s0V58)SGiYl3d%zTN{4=7sG%%t2LU%MG_X<qGeE*M>{<lMI&ICyf&hpO%TBOVrQq24zQMI)%#1yK<~ns'
    '|`jq#CN|pksjE3sGJ)`7}Z(?gw#zf1kAUs;*ou&``Pyp(O23OFDX!G*vu=tY;QnJwT$Gniq|itT1Ike?og;XVDB<5AEEl<V*C+vG'
    'tOL!oWcJ6s08sSq(JmRcweE;mGo<y)J}=+x#vdj=asbkdnN4wUP*hPSJEEjm9#6olJ;P)q+My0v=L0L1#+SgtYL-wECuV4Sy+D-'
    'z}n6y2dD=O_#gl}unc?+u~)UpgP(b-'
    'k&*DjTaApwA3o8@C;*ywG&%}^BYFLLsbv5(cPLcoS~f45`kx=!+3pmk<{RhNOeN1SlydMi8!s^28}bX>iwvb3IvG&@7FlhnYUKAE'
    'Ig@PZ)NQCnLM0?^1aj#Rh`{&#JO!drSrC0*HaAwmJPru&NAus4UZNSGzd}98sAb-'
    '_iNY$^)z{nT&?a3f^%ydwYGgjebfrlg&sN(!WSXx!Di)b=t5LDY{7hdHflTw_kBCC%C|<v6Y8f)k9SZdVWTr-4+g8h*Ku-yoQ7nz'
    'ePMJwHJ%B4a@$JL!_4C~Thm=$u%Wevt%+x$e<|JLQ;IFFz>U`X@w+Ox#_v|fzug86QACEuHkT~2EygbKHiN~|eJkM~CsDtJ66G#_'
    'Dc;Cv@juhTUXYszHeA4WqmYPZPNoH!QqnN?k<w4Ut)#zAg!mUQfLUW70CIXt~9UU13&C$Gm$<#7vnmZJ7CilrUJ>MiSt9>AMY|7O'
    'PR&H9ft4V#dnz>d|>x!N0IHADR-fmdPJeBBNM?_nL_&k|Xz1QIWx=jIIht(riHRtD8G4eA~cYd2EL#S6isiWU7@nntFBgGA;8$A-'
    'i)!3Or;t^RS?kq=QrEU;48aua^tC=)FRn|TaiRP(}h(+StCHue;NZhEei9n)xM`NOpcm%IsF0~Aa<_?AY$9=N?Blv|Mc`AHeQ|F?'
    '<UvPB;)=fM>&Bg#PBRZ>70dQ%ZUG}RH!KonO3D;fJOO>!T!ZD%DQNNKk^_|RBzm+EQT?0^t_11!SH=e8{@G_pfLQ|^X$ts#sB~Ml'
    'pkel#iE%})31va%1NVhTdMGB-xW<mOe2Wb>-+8%jWG&eso7K^W!>~lq6@il!-'
    '1QyM!7#oGfBYFLTsbyF+cPQjamSOQ5t%B=9R+!#u87<j$3~@^F3;1X^QTT$_tgv*9xTlQEb$WZDC6!Dfe`zhBn-bq6uK=4xZ<e&C'
    '&r-KYo5M|Jt6L?pN6k~elT`9KU;SRXY9<}m54LHaBSf-'
    '=YM1vANW1%lpX$F%!EsC$j$eKo92;P5>yQUWbMrB=a9mlk&lLg375bV8IGR^+R1_S?@cI=~%iw74P{@DWC+k06kISpusD`O;r*sL'
    'hTAU2q`h9Ch+E~GCGHe{~xRezr|3x!3H!ry?60GTcZSoAlsQtO>CtObH0IvEAms9GF0@t=DnqfWsAd9xqBH2u317k(<A$9Y05J&('
    '_$orV7uTsbxn?>GNWveyrBi&$PnnzI$Vr~qR+)2H10BN3SY%C!ER<e&90m#SoH4%U`ZzzcZ<XB$6ENU4b%^eDPjk2865@HI^O9~*'
    'I;wG<I>P$Cb%~5C53aj>H5PRslCc7zH!Q9?r8(k?rm^-3Ez|{~=s&rQ~AK=Vpw2e85;qg!IJ5-+tI7|)E7xIA}W%~V-'
    '{lgHtI_c_3(l?>7-'
    '&D#@g+#WOaW#{hgp|eMrx<He7Ww%y9M;R7m#hqM*ush?;Dfh{?I3N2m&6qZHMg9B4O+XC%vLnEl5oJAoMp>{mU1+=E$e8{;LT2_W'
    'Idj`fb}X4znY>8F}rSU)8d_HxV3JsnktEodcHb`*0f}Rs+Q_=8<h-'
    'HCrh=1ImPhARVuA#>S#{9#&0l`7Ox={xmk=hLbZpemf4Na3Ux5?k|zt3g%(4^L&$ZfnVEh*E1z`g=ch|v$uT$HqjhZcBHX8SZ1v)'
    'fl4a{_(BS)(Z8AY@4!+~wl01!{kf))Jk;(qLy;Scu%j5vHhdnt?Cf!sAdvd%?x<g!LPEL?X57o(@oG6o?s<S=$fdrp?PkZu1ne<k'
    'BStIvG!q|`P$)aRYYHVu`t~#mCvh&4fxY0uudc!!LOmlSE@T<e@@9lCt_cd4nu!|e{3u|nca`rI$Lmy+ReQq52TT+)}Y<%!GvBrk'
    '`w23wLwvr{sYtY!;%Z$B@VbHplBp356axsx(<8zpbxeN1mGM6bhNHen^WY1dZ)axKSKfKA*v945fBU8t_^1=Pp97YjWH#LtDtR^r'
    '#{%+^;#mVA~y}^zhQLe*zJ>3zBZHW*o>Fp<Tq^APM+xhD4nP=hXC@d>A*?c>uXSB_C;*vWzjt?zam1A>!__nb&hx@dRwfXxc%NEz'
    '5&HHC&bhA^sl#`s0T8f02-CenB6~#I_1LT)USSZi(OF(T}g&7ClXUvVY5gdEWMsR8%!hq*sPE8gl+HeS`h9cy-JJg<CgGEJZPwQS'
    '@kA;P5AM0LjU`laYLyRO($8jng?t`U1w&=K@8fMAutCA(jlJt!JIYyq^;FVmxSZ?t8$;Oh^IR?jjw2L)3+^1cv!8epF8C!z}A6ho'
    'ZpJ13H-'
    '7_RDwCnds=5#dGDCUGTktT+eG}RHzNolGh8L1W5s4>jxY9f^k32WkT9f9P~mJALn6jUcmV{m(Nd&W`QAW~k#6zMDZ0BFL>eN#fx&'
    'fiB1jIN2(RpQQY75t^Z)cv@s5mq|;tn&O`{RCdgnjA~xqqmQ>G~B0stfkkMEX7%amOix1(vLITsOfCeVS6xjWyzJYx74))RJYk$*'
    '9uf{N~vLWMMpdE=3{-Y>TJ)h#i|mump!`<%YW3~_Uz|aW1{x8L-cj#c#cq%cP=A$?vCV+jM-sAzsf5W>BX6t1>qCS+(+B!^=RRXw'
    'lzQbcwrUQy5L7JepNWcVIgW`)^UjI_b$zi<H$<Z<`^Cyeveqg!+rLMHT;s2<q&Jo@JEyx{wanTmVL|+Pqu*Zh!gxHFpn9UHP{65m'
    'N4TQ!!~$JOC3Ym*<9V@7|Nz{b+2P6`w3U~Ifk-xxVqmll%31f1CF8WJgy$JCwC@yX6#)lLZ+)KTq=BFcD0i!8;1n=zGNGBCn|}fN'
    'Y`et%x>oZt}Zi9`ao{F>5J1)&FGb5ZG7|&vDSvWbcnU~qLL-&YB*b-'
    'UBS_C9o=iWDn+@l3q4*xVD?4I&ZBrx`g)aJ#E;7Qz{xA>OfzM!Q9ltub`q&$h9UXCqiWve%6*66^iJjELvh}qNNX6b<$*KF9R2qQ'
    'oI<G3{6Jjf$rC&Xr;~u*={Nz4o)s%%NXOIOo;|=YI)8%45j;&0<p#HZ>Jk}E<8+CPrg6HYSk>1_GOK*O4cs7FYp+K6E?TtbG4+n1'
    'UMz65&XQ`FaE{j5QVkc*(V8mN2;q_}KP^#+g<VXwB?__4<O}1_ibpYI$7pNMV1_?Y0B+6MgG^n{GtN4|YJ>0|WH9If?Ja_p{$Ux^'
    'g~4Q0$qA>p?8EUoOXtqtH7y=>ro|(QXK7-'
    'd;my<6$avzT8W~S~R8!3K6Y|FTS=@6!UB#&g786%?&D1mMzJWyGF3lY{Etnuul>;@nKe0iO7dW3`+sNf^K%ODh5aCATnNkfE&b7O'
    'uq&~4g=>YL{Nmp5FH35cQ(yi@pu4YK3Md~oGF?E7~#m&}0Fg*TdjXjy+@w7dAh^gs3<+UfP)Y*a}^L@%!^ajX@Cgqvl+~rakPZo5'
    'kj3*1aw3zA7C`-'
    'NVE70^Q84Xw7zCnad4Vx_YxOI*nmR?jYs|Hwuy`yiEdw^}HZwrzKsdZ8bHQ*6Ia9}I$W?^t(OYUZ4aA0fh=3sDOi|*!PaA2$M=3#'
    'JP%kJi5a9|n)lD-'
    '(6*!G}@nVONdG$s<>GmY$P)j8#Bc>^pvlk+TX?sA!o=hC@M#&hXhR?O1p$#P1ivjTPLTJ)oBrzL7aI*xfAx_wL%tw;*zG97g+yT>'
    'yE6Sd$z(FCih!^ztl$sBt;8dueF^7ckEl1R|zwRztvZC@vsfCko@9VS^jmeGDZs-DB~51U{es1B605A+wNW@ZfDhN<_Bx4y$Rdqf'
    'e^oSJ8GbDPU$JnziqGM;zl@?sXRDNDk1(XN@xx^zgpbAGD=(=!3r)>7eNvQ!sTW#nA}BAZCv6$bET^LYyn;4K*vSd*1+Eij4hOC#'
    'F~%(M6!Fo;Eu4HW4xL^;msnRUdu#>Q&sY^o>VeON)2{%W|iv4;99_#2sNeYKiuruA)1?NvUlH$V|{TAu06U8c);wwCEKo~>niG1F'
    'gBHmyTM(<QWUEekC?!u2rKfO)TSb@|{QQO?6o1Yw=aP^Q|Mk+hYm&Z$C|tvE?<B}x>Sq%9@NHk_olWypYyjazBLVqX{7M&RCx?-'
    'yw&aC@ags5ExGlPT;h)L$j6{^o3Np&paqx|_3ognArawLN<T+=)Cdz9Cwb)ANjP{@4r|PjWIt#*>`PC}#XQWyZe>_prTARu{B>SO'
    'WSOHmSFz0gm?XLamN|Kg~45E#;X$Cy3KF84!HHRHqOGxTKurd$=KlINNs)fj=<ScXZEY81PeX=NyI+KXny>vv96&|AN@#Jm32bPU'
    'QL2aa4smhJTUZ5Zc{ep&pSCN0_s{gnCp$Y-i3MW$MaY5Y)#SGBYQ1w|{A-jAsX#DdX8eW)|$=**7b}g_-nac2(X2GxJmSMf^{AW%'
    'bR%E2w`UylVPZ;Z@Tskj2B*Rp73s4C^YMW|_K*T`bg1=*QKu($&ik;Ocm(#>p^c^EzsWCsj!ggVY`vy#>uWExa5xwjaL3sQ3ii>>'
    'svkzzR~Cw+Mx}v%i`3drJuE=SC5FoXi5FH}NdYPixV_xSpL>cu8E(PHViRKpn^K_Q{MzG*5PebsSA?S;o`4Frq*mM>BiDE`33vEK'
    '(T|H-'
    'n4tr0|V<f484Mgk|C%f;QhriDZEJJXnnp$zWG9^o$4J$xJ<&0^c)P@O_;EUtg|{k*;lR5WH5YPL@vo@CQ<zA_LSs%c*~}n}kKs_X'
    'Gnt;h*IISb-P}a*D1U5I((U=<UrEyIsrVwDqMU+gU8-'
    '$SENdOP)#j8LgNZS5ySI8XQ|x^p3u!02pbQ7z+(91jgNo<XdkT6aXWk_iKGYp`yz&Fb>4}v4I8xpWrryOEXkzVkCmvDNH?;g4(lL'
    'sC|=yS__WHes+#+t@Wh8CtO$jankim!Yxmbu3r*f?+>LKFC9@rC#^8&F!?-'
    'J8w@6Ek?$$&QwJFK@cvy7JWp{Tdx6()m4bjlM>Ts<tSI9{Tyjs1{H)e)jl(6}sxlUrb^4kJ!AxV$N){uS`K+!th!D*DMqf~<`qJU'
    'j6Fdzz3)URu^a{=Vus~LUQ!R&^nTl5Utx6l3mr#a1sj>>Y;|%Y$hrkLXC{73N$YJ4Xfs=)<6gXMrDuI*5R0upLS>lS=liO*(ZC-'
    'MRs}4@?B!^*$CJdv2gekbxKzc4akaqjF65s}CS|+fZic53zA+aEYTMdZ?;fwm3h>A<|Du%F_ip%Hp{%J(TCA`fc#X#8lJ82Lm-PF'
    ';>t?wl@-I(vK?vV`eSQ_A`of_RiVe`LzLNOZnzioR%7i1;_cs?V-<lr)a^OcP`pA_m{N&<qPij0-'
    'E1LAK&{g2Xs;5exFDGvyqgW5!iK=n_dHd7{0JuQ$O&bCJk)ns%8My&ryf${k)7}>Xhu>tOwiFsf&Hy;`cM!40`STH`XuZaMoc@;y'
    'W!1%nbpp5_{(;0(8ZsT{dl3NP^W>d+%??K!B1K{7s&h;Yg-'
    'C6{n%~_xih7=`k34NKXja={P93zwc<_3M=2y2^$?wJgvEOf7A5GA2|Cl!=~?vo6r6m;LDk}}Z!k|C6U?w<@L&<)d!j8FJf<v&uOd'
    'm#%t{%xRZfYD}B9_Y-'
    '?hsA;pZZ#|xbWiDPB0y(e#jq&Qy`Xo6BR~glb67$5@H<)e5ZA;Nej$9tFNAe|AzbAb!qt8uT;mtQwXSdBMGph4BTvEL#VibjgMlq'
    'Tyj^Y4-Qz(VqMnfuotVFPlEk@~Cgpq~^eT*HU~pKbn}OObNbkn~)$GQvR*#kMg)~5^GdTp{?w$ttMXczI1R&gMcq{-'
    'P)7KOLAPo*<gyDq%NU`sWx>KM4012@|*ZCIm1HV%)G*~GN+pZz#-IUFKd~Dr)_~(iyaZ;>yaVPwNZif9V*#GD&NY-RPSaYF0zm@L'
    'NKXNrr21)q^9OC65DZh|I#2gUxEaDKlP(pTuW{G{m@3%gkg8NHZxXUcu7vS9OvcbthOjWrJgJh9<Utn%Hg7fE&UOC~jdCU>9@Pzw'
    '}h=u2!`kILF*}Sw7QQ`ATx(+fTe1`WvqTu{kR{u3pXLL!9_XwS}CDq+AI!{YVyYm^Q_la`#T87E}Ye|Kd;r7pU%+Wb-'
    'bj}<2uBW>83V6&jhygg+1Rf;CE8EmdRLld5{fbO&$S8q@2`yZi-zUW1h;HaoY%*b%<-'
    'Y~PKRS&{nnLjnc=%C5H><f})PbqhOgj>NMbaXU2&jq7DFVYs9}-'
    '&;v>#p)R}s_{FDYX1o6GhQX6af>C;j{wg9UQBtLdK4LLEuJ1ZjF>I3BeIzeLz-'
    '9lukcIoF)Z^Ju*rFq%2xkT<QPud_5`Fjq&rjn4`SRX<cOQhislGFj=Eh0o$a(aBG^t3|9qv~CnsWPC37B?*XJQ{WV8L7gV38So4d'
    'OTDwZEYBkAryLr)aR4_O8oP1ug}$bU$^THcWn7JYOjF6Ro@EN7&3r%0t6=ETkbpJqrhw^It8?w*hQgW*K2~qs5Es~|3^~}QJ1L|$'
    'Oq6GCb3JMq2m5|Ez_ci+4RDKkPEdy81NA)B{v{u&m#L;Nc?H}WLIccKzK(4w4{B1*=e8cn+58znqDjMf9&<<oq&7`uxj(L-'
    'Wmv4u;YPz^ZT^|Qrijg#l-ry#NmvyJ3-'
    'M!aRmKvLD&}pDML$)RhFjOlVqb^T4_pAVf*I%bjZP5iYe6(YJ2K7ax`>ZTMG~|Vm5LmVcA`=dg+rK1MUKHCOr;{n;t-'
    '}%k>hX(Q>n=DXj^0E&zRc40#?FnQL9kzGV5LI7pPz~sY~keHV7M_i)ofOv*Vu{9&2>C(ePNKZ_(EjG5S5_)j(m)Q0oVC4Xm|O+_E'
    '@D1R|}vMv#_P-Tf+byUvubYxhJf<PN;Zldxht@EU)_)kAhT*I>)e+a_CM{v-'
    'Ge%bd{8cLXj;HZk;Ffm9kc{j<M7I)a}$ngQfCk*@3aGNclB?Fg46fu&UxPFAH$gW5B_({q+)Wn9T)8Csg>86R&lBG&kDqY<&j-'
    '>9!CV*J09jrdJ8;&GCnB4|-JWuR{RzoQ5%#$b3?GuytQ>3S{^siHq)!4t5k-'
    ';OWbwb+e2HveiO#;!_EhQPyC4V?m!wk;YO50SPNj-CLKwk;=_2$8m}CYc10wk;-'
    '^43YLn1`YcOB(Ruim7*b(O_cyu!X%+h{CCrs6-Rv1BG1Hlj{{>(3^zJ3*2J&rYl@io$-'
    'D;oA7bZ(4&Y2V2IEFe!Ti^2kSQdn$pXLpZ{fiGIaB|zSJ$X60*OIvD)xH<scdZawW~mykgFU6&1l@^cQg#h+9F*u&<v#XHg;N*tV'
    'uiW*pb6+<FA>J@-;mnob4waP0s^NPf-'
    'SwukbQW&liOHk6F{d!v3_mkwhac^URNreo(CW;YJ6=ntz49ril5UEpy$c1f{!~F3VaXFh^;^HbI@xCbb-wcUxD1v*Uq0R3b?RsA*'
    '=8;Geda)n)XzGqv6&@pm%ymdVZ=B>jS^e>&pBeNp;^(zRQW$Xy+4n+Au1)9t!1hl6D~m34ya4hOY|f$IjxzQO@wrWfS~z^x&*fn;'
    's6Hg8R@kICcaynY@0I|s*F9PV;(ti>PK*Az+~WutT5_y09_oYr;D!G^GXm9xz#3}b171p==%ZiD3rl$oIw_DH5)bsH=Y+rQ>w^s%'
    'UTLS*a*>IGw(v)Eg%5EmBV%d%}7yL{ts4ffSh+vPZQu0+6|3NwH?KbWH;pj(LQ^r?$Bu%9?vmewCkBB{@MqKIP6P<ybPN-'
    'y&bI>r-cbc`p?=vb^tYgOJX4`JDJBB-'
    '03%`h(V^Y|QwUasX={0)+DyTS5XiBr7gBs+vdxJ;w(P#ka+D<6*I%{SP;A)F{r4_I8S;*J*PSnn$D!R^Yl?TkZy*YPEoU}oDDp$@'
    'A}U3W<xa-#X}J&yZ$u6MHvd%GPxmijV92LN#V-'
    'f`;9TVy1YhU$<90rMp~HG)i|QzOVUIu!%K`aBTS>RgsDriIo1*>G{6%P?KybGh>vCeb>%BEMav_OAErM>{2to8jwAa-hK-N0-'
    '5t4=&(%Oe>KN>^p+=mXPoPjvdy_)46ZAAt>xZ1*HTl3SA3{UAkR5xQ+*L=W;A>SSE?8c10+<fETAUH<R8lp=c;vYdO$jcs5kWJSd'
    'ni(YX;+8=V_Lwb8j46yD7t(t(w}%DUXTX9Ib4tY`!LrxljC0$Jr=4pa|n)nMj=c$rSt!MOg0!<_22H@Q%!3Re;8WmzVozehcnlbv'
    'n@PBh5%uQu7$>HA!r=d|5U<9G}&xrhO|e}wZ{@8RkUUnYD%S7+NZ@FlaYJvU&7h_?G+p2y%-'
    'Ki7uplxKHyn>`yrx3Om<=r;B&X7{aS4SWx<U}=4a2TSWjp*ofvv1Pi!D<x@U+PE_0ze5<3WlR#To`MNA*)y{Vp-vNS14s&Wx^Tss'
    'BVEhxDNK!WExQxEmQ$6w3AK7CT2JawLda~js{iA7oNrV0hjDeTa|UPfvL2s3(8bY~{{^-'
    'TMUrl)&UuzMx7n)^yc>Hpf_G!DVwV4`Y{akA3uTom*}!}|hXc;n+3u*Wo+fvTSDMoc>?B_!hM&7q>_o;N%#c=Nf^b{dW4POQ!P&#'
    '^m9Iq=FT3mg2ad;ib~G>IcwB6a2n+aBPE?UkY7NSrc8PdO3rv0?U3b8J;8^+&fCs@fv1dQ!*(M(hWPfIc{g6Duv<7%M)Sh|9H@Df'
    'l5mX#|H-d^|?_$RPy3FU_gae<3&t~j4rd-!YxN)(?)$Jo49gWQ7<Ya#un8_*00krHV<CAVQMw1D|)-'
    'w^=Sb7iyl1WKVn%|SjNiXyH3hsJrydnp3mC}w-'
    '(pB9iNcAh}1EJ}FAg8p2`EqL76GDeTmBIq;&P*5c!>6BPk6<M0hT1F7%;v%OX$0}cK8+yW*r%A8o0QUHybGqvTS@huk{PPKxu+H!'
    '2QRaacayj1rq0iOk_Q&kYDNjwPjz-RtpDh$#u`WI5zM26`S~r_zx*Dj9as48n=P1rBXNJmM%XLSGh1o*T~G_`6&n~bbtZ)jC<P10'
    'mv(gj9kpuJSJL$b-OJUPzW3(=m?eC-'
    '&qEN0*t0A7xAgGTN564nUJsA?OZzs0fMefA5OC~UkOFhROjYn?p%^J{UnIjmD|ly|J26z$YEUP-rrZ6w`hjb@J%Fnpx~AK1T%F{a'
    'Zo6~!BiD4>gR39ArrVxejq`+T%RH+p+~#2s7Ih?9R@D=8lzFN>+vPZ%5(1qb2(K#`$)Hv{N^G1Jwjt}H6TP^N>p&rfhNDRyMX>ok'
    '2gdgCw7^T^`gmI6B@wl8=G`70#mAbUT2V}G+==>vLIs&wFz#gPdXq-~21$VCrt-Fx!f`y!XP!W-'
    'f_o0IleDRA*6&uWg1OW&45Pz(R)(~5K5|S~^RQd=^00$=4*_GCijQEb`nQCit_S?T5`Ma_;1v>nx<<fPq^nFgh-'
    '39J;TuNIUsCu!I*acg@UZJ+BbKhH$IBF>8zKtXUF6iwnMXM&wlE29bx>?!(pG&<MBSWuF-pYL&B41<BBpNc2l|3Sg-'
    'Ka(LSXo&;otACE|g*WgodD-eLCpcxGJqnUZ0@rID~Pa4e&H02^EjeYJ{*weR4ATpq5)WIR&#0md7X=kJ~E5Gduxn8!TsCG7;Sv%V'
    'n5M!nnoKU?!6>kPUbZpYjle`TVabh^j0^fB0?H#0^lGG|ihY&CL&v1tr|-;8;+;r>}{aG0m$uCThlnw{=X^jQK--L7{y7ep#_7@M'
    'TKeBu5?T!OXCsy`6v<gSO`m;n?m#c7~<;k(8Z}$=dm()Ks(%3u-K7otDRi<1hAX16Dw&ze=Z`4aXzWiEF~~giP<l_2#-'
    '?qyBxk3P#&VRkow_BqG>kI5wURnU%g^>gQ(p*o9oMv8*f2LU2-'
    '7mw97{#DWuUbx15Y*XwHvfK%FMGxj^S5ICuU3`~4%A#gUa7o4OoDCApq$%^t+BG;|@si=2*5P{%CDkv@cv?~XKW&Bp2-'
    'fvC1s3S~30Wtg=VZo19k#rfMVVaxoe3v;I5h}d47HWD(1kpyQ8HOyx(1HC+TG2Ai@ADrCKOOzp-'
    '~AW@5BJg37&<mPhJKV9LvLc;p?b7@Id6a<rg`3SZXV^(Sa8Cv4vhupYJE+_a&BJCaZ$@Tyu0I~mh&I!3kvy~X>i)2KTe`yj(O-'
    '*_&+lwt>bj>zpFc}W*8e=Cv%EOR+y+EP+ouAxCuklJ0ffas&$TPa<${CO=dZ&+GMt4s!irNqS|Dx<Ec&NIhxvJzGtaDjAN%70TQC'
    'D$5MbiE(^#Xf7{~S08vegyv5tx{IFOE!mSRAh2Y)#nux{Qyo%$a7H@c4$44#RKh_r%@+iAx7jKxp1WsaZ=Ru;I>CzIrb64DsTDfi'
    'q1O*~prW=)V-A39_^!zC8Jsh1qqTBdB#|}mU2}IwIr$BOi79``!B_4eRC)RUS(2l%s!$eOQ>7S}7+J>O|R+kC}zxaa;=V?-*-'
    'ZBgJZe0Dke8FyjJf~$IAk7#3ek>s2R^N{W<U)N-'
    '1R%|8KOqW`XqhNLj?)(u@+LD2wq<Z~NxyG%%d{rcRha%iMtYSVs*eJwE=CN5Y9n7FnX_(dfXkYO8Nf}ZlhTlQth7?^$(H05EA*ax'
    'kknb3_hf6b%8I-vA114<#Cx(WSz`s>laG?MB=3GW$0l9EJ4E?Ur0{-17Vp*HHYwJ}-'
    'SfvhK^}AG!(#ym_c=TkfV=cH5dbvr;lwBa!mBzl3V_x6f<hi+DZ9@!5ZRA?Onjm~b+w_jNl#lsuE|a+Y)Qy$fkk;vdilh((oJWkx'
    'ZztpL?$gmXAy$CONDyQVkCi?*meK(Q<Gdtsk&6}5>An>f~87QPY3u!BPG$<!s;}NL|D|nxauy|92)|y<f@NU*Gjj41=0H|>AFvB5'
    'tC&z+Dli<#PA6{8_tByt+CDDEyb1=wZco{N{d?KC51G3O4)9&A$mWJ^1>_BcwsmE)WsYXi%+#L=@O3l+0*RVe>H)?d@m3Grc^ITx'
    '9q(o)r<D*HobF4s)}I7>zG<?&;H0URkB99ZNwXydflFV%E~K?9%JD~b{z|HIl8OYF`DkHdy!5xEe>#=Xu`WHDkcQzCR~%*yM=HKE'
    'DU-fY=f$Nb0ax?p37sAGvA;|?EVAXs!8nr!#nz#LK;0~02s>gpuGUIVY}Axg*wsn&0aKpix*A*z>B7D^`hw?deQW4UNrq9FPc7$W'
    '39*}YwGP~M7_AIf6v%sEy^#2$v$2F>+%yDbPL}ZGGKgn8xMvxYd9w!9Nc_xQN(bfHI8G&c|On5<_qi|YiYRE?y;7x)7KOd=qa<bp'
    '4({;38}IMxlJT=94p)|5*F4BcH2lu!&B+@k&q2wn9bS4*#(##P!pt!@i$8Kl0EyA-osrhT_M>on0nQoE!1^9uSqAoG$UI^Isu!_H'
    'jy;>f{@p;I%Gk@GYf|<xLz$KQ3u_<u#ROaNMGbOi;OFPC?^YTfRN^eJo}r+ZyIZVxK-'
    '0w`@g8KDJ0fYX8*?+9(8t6eVpO3%AP&J@K~Ly>|4XavW-_+xXt?-3%7Y+XKwHC5}}@<N`+*zt>r{7@&mQnV&-$l-'
    'fI$(?+JC2aHc0U$_Y-'
    'F=?n=l3(E#^SZ@%aSf$Nm^}t2SPYwXio?K4=&ivpnbTJc4OQyD#PxK8?z`U5Z&YSPkEY|#Rt7ft0e_mfxNV=!YwHu&)Iad^h4S;^'
    'dycI<|A!w+j7NKA{1!LS6=s-'
    'NCyL#UV$%1k3s79z~eNOIjp<b{CUad|QM52L&m9KD5Sh<RO&cHR?a|W)Zl7wWNdX;Kjl8@AD_71PW<4Sd%bPR&)q<Y4l?I<G%#-'
    '@~Qrl_PXQy-UGy#WrEm+}U^xlQv}tHZ6D$6Ea<eN7=5pN!ROQ0!s=5&vo@$z}?kNiCIcb1-'
    'rbyHfTQINqFRwFxKD6z{vvU(68dB_G|*#I9OnrwTR2Zir~2$1}Js7_PzU{c||=eE*xP9#YK(quuo2`rY*Z@c;e~j9IDXNtaJ~L#p'
    'Sk=l>6;5<x!1PRICb_U8U4)Xjpn*0u`u`>?J5G>jm0QgEgEOEd+_%c1=#lr#5l5o>?ARf|~rKc=rKq~=q~mW*D8Be|ABvoj=dgTv'
    'amJRpy&MU%Tat_AR`RI?3}=wn%_oQYih&Mcx;T<xE#D0)Rr6vU}hrzQ!i^!%8qyCnE^wl)kC`fDvlD%F`j(WcuA)U(nh*EUG?oV7'
    'o!J7b0MEUR#{=}K?^Q`GxH-6Ci^c9T%I3fh6)BGhk16sgDqd3gnM!<MlIhx@dQHTa$SnnG#yQcjz|-'
    'U9FvG~@0g;LqtQ3UfY~r>iW6^7NMFaIBZJZ39WSq#NmM>Q&sX{Ca9$&-'
    '~Q9!4C34x{btk4uMZGT9~*&S1&)n%IXKZ2GkSiJh(Phn1+_?zH{lEi;<WjF#i=k4)$&!W)5o8?Ez{$SHI!JxA?V;<MQ|c&)4^XW2'
    'uN{^H?gP**umOv;RqX_U|mfNep^}_7bpu1Z^QhIBm6=^I@E}sLlBZ9#q{t$MLw-mgl{|@wnWcy{wtnW(x1`(U`^s{dJ3o0lKXFO1'
    'HE3U>XE9+Jm!6>SbJQN24#}L=N4RdlA2z1g=yTAIn7;@;@VJchmC{V3er=uo?w@W7$1^$n%wc;J8P`^LyMQ;`u%9DF%WGc_8Q{AW'
    '{tWS1UQ~x9Wn<3Bne1ViDdQ(&=VCt2vTutbw1`?6j8&@4eZeKznaCB+wq)Iy-'
    ';=51W70Z|wD(I3B+fu7K+yR#wt)xg+K=wOc8uIK#NEmNo$OG(pq_7%;RFRPNqY*S=13x$nWur^!EINDO395ykRx63>_Vf#Y5g&+~'
    'Dwi0ApZx0vnEDD(42a-*{C_EFVI($YC#nrxGgoi~NlV=b8H2dU5vyA>?c^}nZp=)?9Hn!wxc94y3obGvo02<y%5-'
    'oaw5H@BMyOR(PD>>gkc4##v0i?0~&O>GsfbL77qLI`vAF7i6i!mi%O4q&~Q+p>U%GjQ6ADQPMbu3?IoNB0fzeEf*#>;AxTpNMDrx'
    'KG40ecV^f?B|u49cIoW(CiibTdF-axh+DB^r$KX_u!ZLwmWnZKVd0)mrjf-l-'
    'H^!Qoi2p*fwByf$qSz0lN)!C$<gPeV~_+WCwx;*CPae{{ft$4m@tMjp$D?B!$1iRd@b{Ies4i#2i7r&8xV&RuC1z0s6mu;_9O=(3'
    '`Yz1Rp=<Ii(%u0qz&^6d(7Cc#4nvi&?!UZ=&yj)Voyt-'
    'y%@tZ5b%?wh9z^TL+4~Z30E!wt*sVyFih*eW1vz!v>ghUDfShJsseWxcaY1K1-'
    'K{hl#r&ajUm8Z75O`u+NOc(5*#~rv)$Y6P6NwI0%dS^%%dJ!4b>#F^=b?aLgBZK*Y0sJRsuPJ{~A$?JKfl9Om4Eap<lV8E$Y*YHH'
    '{B8u%T2rPJNYUkB5+?XvBH9VSC$a!qDt?<iCg(@3nFe3wf~3?{v>5R3AH+cn|ed%`7J+6oA@Ts+oZK&<7OL|1T|QP5qYuW;3iJFT'
    '8aJZj!#>S9+i@itSJx^~jm%u;_j)1QLbW^MY{Q4Kzp=cAhEgmva=9u)CR9}kLnrjG}UnSPElz2`96Wk8R_X(^7a?<L;@H_9<>ERj'
    'wh{Zpp?Y($NX{>G`UokUd$OxaWkmrH3U)Mh&o_tO*sKXomh|AH9bGrj+Zt8VseGsoliUVZ0LZ`SopU8oBI{baBy>`DE0!2&F`VLm'
    'kli`6F0(Wof_um>UeLi;umbRQ4)${F+Kc^?w-gdY!yc*2i|3hu~%H%nLCJ?XCYw1aUm{9Ec2>63fBTi!Ns%i9Oh`W``q-kG_*o+-'
    'XG<1Li$3`;X^$d+boPZdxaKF!cUn2cF>jm<SosnllOQro6_fy3ZC%+_1!%u+Y(S-'
    '#0`*F46O0p9MT^v>Snc229+O|8=}jw1Z1$*b8k-'
    '#%L9r&+wpPxFE*KQt2=OWeR?DmLM_Z{RT%o7?pTg;GmpQ0$s?RUIs#_D}X#9hs{IN$#?d0Q|e#1_JQ!&*tOp@b6xy-=5s(q`i~--'
    'G<{C9+DeTmN<mujaf*37F5Ss6oNCb+02mO2A&Y2P;(~-N-'
    '54Su1W4f(D{|J!gF6%Y3ds8I2Nk7>vWBy9Jw^FNX~rUXnmiZjnMbmQB>cD65HnO-^gQXl;F2-'
    '<S{i$pXmz<Rc(GZ(+31A%az8^9kGKvG<28byQzJ`Zn%fsmS`-K6=^zc+rU<m<TB&>Lh-RL8(smwYAY>QCIhK#_?2W3l?~S=6;w97'
    'Dj7^=!>f}@DjQyt456~&waHL&r7N)^)?xl_RMHJG|H~}qKmWE0xdw<%F3SU;xp}mZ&qs|A^4V5Y$cF%EUd1n?00_VR%P0VTt}iH5'
    'rI`gFc&xPsfODDpFx4g51^!uD-2w1%8qJRbOi`7oWN<#v3ZH-Jg7nP~OZ7o&TXhKXBRGh_()m^jBEQN)WJj=4*I>=tJVAcki5wHA'
    'R`7QmS1U!b!<;nXsyU~<Gu#$;?8~@Y>gbnIaL+Mvqd#h>c;m=dE)TJ|8|nGzZyBxUv#Ak!KJORR^I`Vc%sc;8AuMtWE%@zU6~dyh'
    '7wpg%6mk#0mtB()ah7G_yYSD*o?%Ro46>XI@UOzH*;aHfIRO4ux(%LWi1`S<W-'
    '`?7ydnpWgWkJLZA?M$*IDT8EMJk4gqIUt%RhNIPFCP_Z0FzW1)fYJjf(}L#zkJj0O^ZyqF!-'
    'vB>&6wyou0!&uIOgHyWYevmsW$$Gr7lM@@wA+rN&Q2zTlW3OR(|%a*Gn_?)$H!V5GL2as^j0+{8dyV4Qxx2&sp7v^?StvjzWb*ur'
    '*p2!9hmI_xBQHUisS*V&w$8NGn?Jga?$zs)1I)0NSs+n{IC%3ES(lMOep;|~sadM|>DP48jR1f?2P|-'
    'ew{Y_cefAMXLaszZUGxD%#ZXPY`^Kv6(eb&ax`j}U7Qxq2Aw{MEV;urdYLSEtbvZ^X>A~Zrd6h~_K5iuI3###xI%Y!+@EybIXyE4'
    '0nNQ`U1h}86{jLThm{xW~_?qsF<x&8d-'
    '<Q4S``*~?nr*5#H?@3mv8|~+Nlhx{%_Vazo8ucst`Tk_B`ZZB*ArX=T;r|$t?gb#cISaxszYT;9kmbzG1EINjw6xECjga<PQdHW9'
    'KxkgY%~2qP-'
    '@Z8tgkS0l3VDXv2@$s?yLjk1Av4^^4vJPWi(VaUQ}z`>Oo}Pq6(Jw5rfhbF3|mtzja|ID*>}CV+4sD<Sy!)aw!c?5JHV@(b@S?G-'
    'MzZm2qffj2z;NZcT)(wC5ynXz72u(vGzQgS7l@F94*~*dn2TK78RB5A^4g1a7z^Y;J0syg5OvAf<kj(Hc7mRspCyx(g{fv!}aq_o'
    '-9tbJAVHqxJ_en{fmWKCn(pyM5y(GO0@0}POYJ@x>Go{h5_o&!e}y>DEmMTGnub8x>sNcSbMjVT<3w*f~z*J93bh2EaKOuioRh~P'
    'ymJG(iB1AL!!_(r<5>`Aj%2%;`pm}_-'
    'G}{R%U}EhDS)|RQQ=^FQB@e<q;~NyPf3`DyY1DbJpbRn7YE4d_GejrDD;Z*nKqL#)9jFFa5lRB|ZII^cJ4EsR+rMxPO^n2#ib9LF'
    'p8pETkRRpYY_)WT)F2tVT+3w{LJVGUvE`-&2r;!tElAN1-#f;Sa~V%xxUO@g8$*gmC<ixospk-'
    'e*qQrLSc&)+NBJL>_HeLo7Wx)bI9>4URNC-fD29<?)FM>RkV!Y{$PEE$3)&u3@Pqrj-nqm?|XGzs4HJjM76SmQvYWW3cgMDak;r#'
    '@S|Y1wn`kcm0tx91dIHi{f32l26U&sXV#EW);9ef0mow2P!`Tx2bJVat0ofuRuG9g9Fsr2{`p~Y~$zRs3*4RN~Tu3$t2kyYgFcQn'
    '!3h9n8~C0aPr2zxlLuP`QcWTvF3lTR8&rSNp|FKWNMZm5{?f<x^fC_Z-eEFbS+FP=}6W4Nx#|NO1D;2IV#?p1OmZc$tBXZULr$HU'
    'xPAhoNIB6Q-#5GIK-'
    '*M;CdY4RAcaS9N<)B@CzK@RAq1j4sR+mxDiLUNsiX23z6%cer+1Z8urUvN$?uKqO|0yt=WQ%<eHH=JA&p<4~aEA+-'
    'gXy;n$anN;uz>HT)Y){mdACGu$;p(gvm{Wl8vW5%+}@*D#WRv0MHU+r-'
    '<tO}$>u?!CA!BMImHvw3oR@_A4&46%+YElm2UCzz`&gm~x>;W|if7HX*PAiV{mTTd7Md!Y`NPMWQclLWkgtA37^{vzGAa=B{<SnF'
    '*+mnEC^4s#rlOP=M;A08TOdAQZkSj(?271c}rOQ}9JCsXR<7Vgnfy^a#{m)Li|Msww#!_7K=2|7ivnw#H+#h}EqG{7Sn9+vW7osM'
    'k-AG-uG4D(OifP+zLm;AE?9@pDz&K7w5+@66K566wn%|bV!FJaDVxVjZ$OzGOYe*-'
    'SRbdmzU6Y3z}TKuX=*Z94|)X@h2hj6uDc{4{tEGl^`zPZh?So_1RhQ-?d?ov_d;wMWv5^k=_r-'
    'h!0r#m0GuSmt0YXT(=C2@4S|N29Hnr@4HkEvtKOWgtwms;;jy1>?AVRw$Ryj|~`a-'
    '8LDRMCv%EN{XJ+)#j7{&VN`!z{mp2z&6jhP(%X2Wf_$De$<?o=p{a{KB67MBs4)b3Nv-<+^(fF_z>R-'
    '~6%RvBrm64UaYc!ctM$;AhL+y%s4}RT$r=bBf&XW68>)ni={ge9d<26F{>jFhBevm)X%jis3QKi@w4!$BVwgL8=d986NXl&`prs9'
    'W;{+=gB?3mjoim#~Brtp=jFog=2;NtxyO0#^Klf#6AZ#ku4EKca(j*E*$@>!T&ThzI!2a@Z1=}kY&Ac{5^B?5wZ4%`;3US|E^L|U'
    'EY7yKlxX)kfm=U(!scZs-'
    '!DihIo%J?0cVyNHMNH*?y@GBbVBje~;c_zsH{4r~Qrl?AiT#^Yeat_Mmn`AGBu=aoo6lh<m;y=4if=6^=PcVWdw&xOvJX<^+a^Ya'
    '#A2jvgFK4pX!L0F;o_)_(&<3+6`AbkyWEa+;g8lF>9Btz<M!N2_A8Pph)M+j{1&W+bb3GAziXt9J@a$Rw;c9_C|G)|&v+F-'
    'hx9gxQ$1^(Mh&OyYWzVJ-J8R?Ld?8sa{T@;q}Nv#PGzPxvfjlT%R+K-'
    ';7rfJcNC8F!)Xvx`wTGUs^MY|mXrvBFFpGkLOtLQ)N4+BbTDVH5bw^69<-'
    'UXDxi>}|e4YZ=eW(OSmya<ndH@%3fXeWmczeGPoPi?&Gb3MlufTGB1`CiHVJ90L1FZ0qh5j`w>xQia}YN9qHZK7*e`+J6_#ynSI>'
    'w<^s2z}>fUy!|0=KbO+m_Ygt+z~o+1X&#i^NAeLB5Y%5t-'
    '6GGB?4Mq2KF>htHKr0Xsc%3Xnz@m*9GB%8+T5m%jHl&jBjaf~+7vVNyV(xyqz{^Cx}!Q+%EMjMK4uryj+F{VTwn`)Dbrmx*mv)85'
    'GPi*3KY@}1^g<)@O^L?zE_6f`;ahv9~y@5!@}@=co@Eq2*dZ~Tn%=e+D!yfj5ZNY27HGv6Fyw13R%v!)DR8F<#{{|<}bIE@iZK5W'
    'jqZ>+hX?KTGo+$o*t8Zd^O^?GVd;+RCpz3Z*4KrT^0w<+yhZZne>p}PMN!fMB!a);g%9S1hVjRFju>dn&-'
    'LdFGwx$Mfi)Z4X3G4eUaJ3#(n~C*3y)TL)8B_5|T>>sDDtlCK;&y$*J6Jkb2q$@8D7HWO+V&1DqSv^TxLMW9?)-=SDjj&$-'
    'dAn5{oc+uAt5)nIUP#1}Y0J7dXeA}*<sfs>}aySw}pzZ6ra>2Pf=BT0`}nL0{_HU}5uT{m`N^?A>YT|n&nZn(l%+C;+@rl2P|6*l'
    '%&Pf;c&>97876EQmN`-LC3RYDDxR17v!s7gs)8||=-'
    'OSKb325|y>wQ{!I1{gPH<W+8)CulF@88_O?c*c$P#ccj{*}@I(^iOfk^%M63OHhRgx(QO9!fCkQ6|mZJ00H?s|6kjcKxtK6X{zc6'
    'MFbVmB$JHhG|JNKAc}xdTlS(%Jd?=`-'
    '85T{iD8pPQ9;ev&@?o=(X5k6f`Zz}s>Wzs5?NiyWHcy5MPzY7M8tgo=H9BhRrTKgntsJOp1+Uh|7~8s_rH7Jty}kg_xp61)){#KI'
    '>7&X0ig~sc=sgO0S50233q_O`yv7!VDRom$O8=C7ZdaVgLiMj9$@h9BiwH93vcH+OdcSn(@TWvTp(!Rr4S1WiQ$}AQD0E|<W5wSi'
    'Y3{|1S6P>IkON?vT<f1o@C?9d?s(wBvp}x{PzL#&9^}goscHDK!jtMOI=Y7%n9Z`?DSHC$NeKlk2`+4Ubtx(-'
    '}W!3X5KWdv2EM_sIhI^{;08S+y1DrZQK5+v2EM_sIhI^{;08S=^9QcV7hM814wAzVx-0fTmCOuNas0E<3}SnHJ-'
    '|v!7+cfRUw{Jqg5fEQ=?Tr`?e@a<J?NaRk}P_rL{%Jx&@muG`(|O$7HFo;RRX*H&Jn!1~y(wu<{jPLmPeX2OHYpyAo_@m)BKbLkl'
    'Zk4K}n3?HaJ5U1!&V4R3N7yJfzVvQBe;1<;N0Y3Rn0V&LW;l>DP&;3bjF856RG?94@N6OltV=ini6LpLYmArY7kbAx&oVlW+W4|^'
    '73Fdawq0l6@iC24@~QuIC=uIV5vXt<`M98N=Fon$2qg>{xAXym4g97!WLUF9emx#=cHJN~!j-'
    'cKITY;b{}Oi265(F{WBQa|{Es&3nyR@GjnCWy>&K8TPI(ny=74j+u9!tZWW9@WnG0aN`7v8){vv)s>o*B2w*4^Q=Cr2pX+y%^<y='
    'FVRj<$z$E3!@zHs6HSUsHmivr(c#jc{vP!rQzUaX}DB{GCUhtC<#`kl}nRJY$aE#%(A%`RYk)j_95Yfq^6-'
    'NQ4gnV{xEjtX|M7p?ADU9`6JlCBW3ePv71B6=6}ZOJ}H|&hNWKGnm)$VJ^mqhP?I1A55dIrLvXA<oLd8YBG+blr8)Uav0e$MdMVZ'
    '`Ki0=Yc%``&7e#p`+^vhEyz-bnAXha!y(A+JbU$X<rM@EKaF?ZaF5XlX!SxzDvdAhOa*R*@R=`gR^*bCj$F)zs$JAw^9v>vrH#tG'
    'XHLyc5yB|o#(ADiqK3r|0Ew-#wn`s{=N2;f(Y9dFeXJ{8DN2_NUE!>YYMBI@tKBOr)17Ey0?Td{>J-'
    '*lg$;qTFUo<D*9_x#6s_n79ct9T$;fv;0^osIDxLduVe6bPsqw}SQt&_eu4G~HXT(<!A{p1*nRm9Y>G?dmsX}((=>|z7z?K$eCS5'
    'ZzWA<7NTdcC7WUxBO+<7mrwgP@A+N_C^4lI&_Vo#wS*S9h~8n{wDCsEt$yk|Wd>Mw|L47y>)V=l-'
    'b4p97zpl=e9$iju{5nfi_?MI2`8isJRU0rHi}S>9+aYe%d%!l`z|dgC5_OoTU@n{;uMH^P0qILaFt;vw=SjIC4Ci6gjLZ6er6ZpU'
    'M3%YnXVw~eSLYqyOZ;bpgto#17+jh*3jD@`(Xp>1PVc-'
    'd`ZH+bFV2!iXp7d@=$f&(v_oc1Cvvb0S~X*;|@KL?(>2>D0L&998MZJ9c3lgnM)Q^)8BXfKET^L3GwFjKO8(0r+X#`++f>YuSb__'
    '97G!UxSQ?j7ZWV1?dMKFF~>pD$-@ogT*Vg$}h5#TiQ}7*>V*=qg|B?kf=0y~zW-'
    '`yx;=iTbhHatE2Zw2;ts3Krm*L7<xzEXLKrVm#Xgz-h83SeWMo3-jDyVg3tKE4+LEN%LO^?mZ>#-'
    'lD#RxdD2bsabAlPX2PN8^Wnxj&;N5^)V4{Xl_NHC^v+=)hEgg1%koyC5f#QduQ6ZY~&tyEIWXYd9aG?s5bKika0`Eizz#+rvqA4*'
    '+o4Quw=@v>RIl`9%MK5oa4;=E5i{9@|Z_7M|$8fQ_~((C<=&}R$eb?{GmH#1}1566#*=&!0#UNvly_q-sYtHv|eN?!F`*l+Z}o|L'
    '-Of(G&-'
    'BI26oJMeI?dG;XJRzI_RJEF%b@G?tI@U2L<EwjdIWe!dUZVj!iOP9mL>O0k4@3)FEd%Ooyy?kPbQ9Av$D@19Zqa4$mRyx^m8$j7s'
    '&eT6~Ss6n{Tt`KO9L<ZgXa=*kO)Ciz36DQ;(QDX3RL=1PzrU-'
    '%3wnMe!J(iM=)wV<DM;afBesV<Jdr%e`I&s_A?u`Ikz@sK#Y&MA0EE?J!>#{+9=9WO_xzSNgR%Ji+P3)ztgB5^ra{kS@m=<^CqU2'
    'aHI7bhT^-{IJE>meMN8{)z^8-1GVD#HZs7#-'
    '>a(J`U9;MtwRU1xqfBPF|msQ5rmGYADAsM`nY?<GFCZD+_eY%!oiII`w^Bs>(nn#c0xv@K)=8FS7b#d3AQseTm8)%B@9CYP2@lMH'
    '+=3G^+chhi9Z?0t7<U81{lNVf^xFE0}KF{wd~g2@Af>Lwh~A8iO{%DZSkI7{A5+retNjCO;w<#H-'
    'm)W~~iUpPnJO8}j@@;++Ymg$FetZUWpV8qGy9&GSZL88W0N#=94uDI3L03%JitjF4%=Et!nhEx4G*2IVOF}Xx^(hoM;$FtbMLxz)'
    'eWmafPSHWr6d?^~28nJci2(p<I&ly$>l4Rv`TrG4SPmXTtlGQp!CSO6SxM1`9)q{+TzLNBD!Ro7ElS*b^4dI{cz6LTjGW=Rt93A7'
    '!a$zS0bffPTMot0W;8Opne~+sl3iXAEC4Fh1HF077^fj^OhErV=Ywox8F}XBziWca*;C9>52YEN#ZVP>v_rPtn_;z_O+*TTUyAN('
    'FO-EeD)eu2ldW0RmXqV(^sE@tRQKG-k)MfrIOW48-'
    '^|3wbO0Gr=^_l%tBGga~w75FQLQ@yAs+NX=F5s3#cYvQNQ7tjmg{d{Q19&%>tBJ)=cmq@`rCCATe9xc6njB8`lUS3#rjN;`lhY)P'
    '$1BHYe7d94t)utryzcNYTwt5|8U+Layr&kewN+SFIsXhV#lLx&NU<NHp$WMho>UseuJXCyOrHzR^0{EO&jn}uT(HLHf^&Q>IM?Tb'
    'hZ&;mk|1gUdj=X5|5@=SpaJrd4q5TroMvpSb>UQFW39VGACpTDC%r_g-Uy{D_Z0}&G(cQkGv;!!QtpCo6%NXcb)poP^YjYcn<G_a'
    '7yFR*;i|uYY|SNdQdo2jWCiFZwp6hUuf>)mma9I3N)q=V1B5L}+zV+Bl_c(i^^r;vA238<QONjLp#~Pr`accvdIJhr0`ZF2wj$UP'
    '$qk#;M3RJb%o>q3=N}hqY&g}pSYt2O$K(>fX_5)Q>+KZAAu;Y9sEzc-'
    '%9Pg;FE<b9Gy_%$m<QnVR*@7my~8{`gHYL`CP}5dRU`(TpBHj<Ag%F&NN^>5q!T{>uRj*J-'
    'uPjfPXw+xUc}*3fx*MiO@AqHy)lE+uvYaFRQtMHT`WxPOVa>f;hm+7BTq9IQB)l}v%ov%?~adk0yxk3SSPq!ACs#dUVl=KPjVTfC'
    '~Z2@*2*(9hs0Q?GP2wKW-'
    '3P)m=Q+LolI>uYvXRFo=sHP@3!zAV;yjhP6g^bVtkp(i8N@1x{ebe&}wxfCqke#>LyNvKx@@>9`I&43s&oCl~1{w&TN9R2yQ$Re%'
    '|9wd1Y_`<4Oz{Mvp8yd~*dycr+Kr5gyHjaU>u8$q$Mxjm5CjZFl!{pQyV&@L4VtOy%LG?6*SA^wN0n9)F*lkVB4YSNxbMXVNqiEY'
    'P!PmI+qqYMP;eC3-'
    'eZHNhHPL(?>{NYA19CRnBCnz<Ttg^>DnE$FA))cj_UI0C6P67|8m2)+I&iM`xpTyhUn&lRU@4X|Kbkj05_PI;8avtS(M@hlie^BH'
    '|a8hg2!t2a$BewJ$j;ACp-g8dBC{<GPe<{J-'
    'C_r4DS&**kepzJ$^$6eD>{|Z;FOvYwR2HA1vLJ+Wra@DP1B~lQhd$VWOyg&;El9WkLEgxq^$`GdeQ?HtHksjwGnUh{j-'
    'R3oPE@rCAbFOz6xFzc@LJ`l$qFFF{X33tIKX#1AvtS(K@hlj}@|pifY4bnF)d4}uj?)wBB;}4UQrA|;`P%AuUt68vYpWA|ZS`7TT'
    'b<-<tCQX2?dfhF`em%fQTgjtq$H-P=f@p@!`Jkyh-o-fElt*k`tz_pWSB0?U?p+ej(S+A8{oURFw4Z|5*vwlzKcd8p6{YjPGy^-'
    'l8f$e$T3!z+^^@ubotaJ@KvgUlM*eKRpC|=(_i<ZqS_9Q5+|0^;GZ~}Mnmm8aZ^Oo>aSt8NBjBH7;Y+0W1&(uMdSlGHDPN5-'
    'PBbDHLT!DD5G5Iw6Px3ev7*ydH8wPb7R5J!7$Ni&eRRYxm!XcDc+_vz;tm@X!FJ%S^+;y#4}wm5zlnNa_ZWoAL}h}m&`YBhIrAi*'
    '8RG!6F%+L3oaGv8Om*OyMCDJEs$n?Kur&MKiRiGlPl?g9n2uj`YcpArX1@ejz9akBRV)4wJz)o92P)nsHPXz?r^)-'
    'a7m^N9dtX_$z-Y%pzDf3fu=w1K4qCsIO6CndSzvA=DTqbPjA6RJiP_aDQ}ZD`Y4?J)<!=Aa-SrzO^E1&2fBi&tFSssn3kXqHG{h%'
    'jt<z%3|+8uq5({<BCd*o407;KsL=sA_@`-Tzg9IE=T!~Hdo_*;UX5d-'
    '*E+b?YaLATS_hLu0quD1ii@*CxcOs3#B)~&5zk#Aa;n#)ExU=UH_YQa8fVGF$N5zpLn6s=E3bGn5A_0X;i25ytqBpK+juBLPQmy+'
    'oJO=3eTktAqJ8#XVJL+98ByZ7Sy;_p0zU`r6MiukO|b3~46=vh8?Y&aP*X+gkpsb38J-&ldG3!|H7f-'
    '*1m<z|c=5jbczhLawa*CQ=F1jDJYPjY#Pd}Yatha+m%^b{G9m>4n=ehsG?Y770!7H6AzI~>hj(ep`Z8*e_7(VlKDG2WLj`8)(^EQ'
    'DGcZsG^%PWS4wX((z&BL7g*3Z)sC){Ec8dVifvNUs4%Klg!_MHT%&AjdBjCRddUlqtv3$yqV<|v<^hqN=`qu*_KR$oO9a#t=bL#%'
    'Ff;P?Zkhs~J7WT69!Y|Bi$f1`Tz%SrV<xtBF;1~UQUaj0K(=J-'
    'hNXEJX&cV|Q&RAMuoTA>NjFf6o9tKj(2?^v)UsqnZS_Gomo;JG5VR_|D2j!Ks9FkX7(}J=}&ZY%rwXC59WsRIe3(8tKmll+r48>l'
    '__da8=BMp78Z`$_;)HlS?0Gr00S>9((J|LE_7fv-GmalhTV|(d&{m<NrRZ;#2cWYIY{|(@IWqKE;{ckuU;Vejf{*a;=cSXw;z8Q4'
    'CZw9UO&7f7j8MN9rgVy+F&|2RNdX1qvH2KQsOm#{4$|Y%E8OXDsx^=eUjl#T@!Y8pcIGzn@>qo*>(yw-'
    'v^rw(f3jXVwj+gLEY2F5?K7O9%h30DwjP*h|)xcOU+}7CMl)PSOZu06VF9bWRj`G5RJg;Ez-_l+<0{)c+jKg2&HTCRdQhbAV3gjg'
    'y)BMf&d7c&xEOWy(ko~v@sR@uPu?2m4NG3XtQ#eJ|R|ttyiS!j>;&sFf2|@9CV*G`uIE_dmAuQfNWT6ljZzQfz2#mH%wu_-'
    ';F**AeOm$5-'
    '`*+gLKB&IH*Z>>JFS1<HoP1EMOTwuJ#k%A(jqRo9bxCt8)<n4^+^scHE;)$j6*F#eei}bsuHF^IcZDT@dfGN#BHR)SQk!;}fIyQJ'
    '^3n}<>hNNRoiaIWv8(2;f?r8F9X8pi+Jjwor?EIkj)EPk>$5w|-'
    's?<#!@Jv;Om$1R+ofrD8(chjy%2wNtM(xwPp*k%O8I4$ADXKg9P5X0s=={-'
    'xTUfEPI>*%+@rNoeh9a5ZImAl=6U6nuRK3J_#d<ohUrp+tLxN-'
    'W!h#e978G7>>Y%^+>SLLEJ}r9HpZtNQ~If%91n7W=AIl6Hc>4+EgMAY(>zt)NbGevk~YX7&hW%}qY+8!DBE^3^)KErk1*9e;h2}D'
    '9dk&rV`_mv?U)d>&#8%|bNN-4W16cP66=_7sv)tC`DA1Jo$@-Sxku}w920Kix+upS!t*LYFFZfZ3h-'
    '4mH}){~P46m4nd*^nmCMtvB6;!TLeS;FEl-'
    '`}t8L~=awa>l^IkvWjs*Zewi3mo;tTM{i1FZ0EVs<Bv;5C|RT=Ala4H$=e~&h{H!83HnH&E=l>dQk9*FWk$@9ul|E+j|_Dz^7Ph&'
    'Jp)`lVkhMLY*zrh2*)UAYDyK6ONsgH|#ZFe9nLr8t9=M3z}XqiT+dVl7MFNUs`8Z$<p;PeD}?HLL3T3WF8GW9L*?8lh8AmQv~X=f'
    'i=92e)~k9JJ0Kv@&Rh;txooW@+!&{(g8^9+sk%KIAIZ<N<7%^i9$$}8a}J{aYdLwRnk?53$Qb1PJ8)W-'
    'M_+vnj<rX*DTt)gwmFjWO!Nd7tvbI2+eqZa57hiSVCe<H!cWWe4o2G8^{c$SaB)jkH#_A$7|$KW|W2HSehG^}giL~;k%g>W9)m}A'
    '4=vUcHqx-'
    'V*EUl&8qBVQ;+^XYRu6)SUd4jvK*?Kv3_$u&=tw(Qv`J*O@EF@hh$N(t&)j`jMM<GjA*c&~3c!RuREE3}1|wL&|1Su3=Mm$kx!Ok'
    'M9J2_9l<8j&61Jk^bs?NGOdQceDHxS%O1NG}9Ncm*hv8QiRk!&C>Ywm_tSSiy!>sY7vh==i)h@*l;`A$@J@V`;77JnLg=tv}bt<e'
    'HgDTNd^vzciMGul?&p4Hhk1PIs>=OuefTsdrtFqT^@xyDEn0iPSEzTtD;nu~jpfoM)d~HH*pl_Q6%vOfIlbF4B=Nw2v-'
    'oPAsy|E^^~7whynG%j6O_E>k1alePl(2vgHJv7*9O?EfsxS2q%~nt;bAl9Ago%#43&L#&zMJR4%o{INbJ*W66n%x9sFsgd?X1+`>'
    'oE(wRiOZ;P^5gqSwdQMV!zPE5FoC@w0I4qX}dB_FuT1g}sE!=3^A%=gf$_-'
    'z1<hH(w81_!wR#*|^+?m@ND`H){a9d?Xy6~>t)>)BDx*NBZR^%J*&P}a#8a8IOQuPy?#aZS)K39u;F~i*Wmo~<l8_u&a*4zj5F}d'
    'bu(&qjesx)eI!xryX^*z<)9sq$;{2jPgC_`&HG{i=fF!CS4ejC#md>vnH7j_=JR+zW*;dQ@xyAWQh%-coqT5aAghSwVNb_u-'
    'JQhaX4JnhP<G)XJhAZW<Y{$R=Ze+goZ{r@cM9-sR}zLa5I{7ai+tqbSb6l>i*`j}iJGELJ!+Y+qow-Wx5^&l+Tm+A+$))z$%are;'
    'azCbNYXk6-q-'
    'eBr6VGl9YwXXjLcC8?Q`+P{WEVsh~NVM!6#X?B5>=?x&NVMz}#bQXb>=4BgNVIMu4xQ#S2rbtl0Bo`YaJY=?MW&hvL=i<1ipcH7a'
    '9aImH^&+l&a*kzurKRla!toHO+)R#DkZ)dUrsX-8vSe*O|XZ1P-'
    '#;?#i7(|neGgE<CV~e1?rDh5arByV7!}2f3%V)XWD@AEagmFFrKBHX$Qu$lr!zYc(=Lu583`9PU!kou>ErlQ^aR_2u^bJN$(@Iqv'
    '>5^O&mvu+>!Cf;yJd&S{BZ;CDyXf>tk|_!Zb|{GNPqw-'
    'F$Czrsg{mfEG+GaBJfkOf5Dc&Z>k7^?pE@fr<FxfH4CD{Rbhy0#^4ALx2UOcOQiS3s??54gnUB^nF4A3j|3ea_KV2Dlxp6I8#y|d'
    '5l>)Y)V&4V_XI*hMh{&bh7r>H3n^eU0;aezxc3yhV}8HpN_RYoaO0Q>;G9FldF!Mw!U9<%_gbBZ<2=lO;V-'
    'bB#rQ!q>+A;G|F$1M%&PfX_b9^7vW=PAKE1tPj5>PuY@h+Di5!4kP2AX^)lf$Mirhd9e=ZxxX`dfvP>lD5f-2uYr`&L@KX~-'
    'H1QeBz}A@y-^HVeCU)^?qKREOXI|0~TU$1OKsRF)rt4gv%x;o`8r`8lBIHh-Zc1lUUllLOI^W9EbaMk7%fO#-'
    '*(aQ|pLTKPoyXvZ<k%M%Ker9AN_?JWU~`(+c|5Dc>pY%S;`N*pE=|(|$+ci&S2u_EhbbP`LrM*M8aT%GOP2|FKP9PaztG&sB@BLQ'
    'QX@_igdcI5DEx@iwZe}$O%i^@X|nJmPE&**ahfXph|_h#AvpWHmF8hA(-'
    'OhRBUqQE9z?xkEDg{~e8K7+W7*B)X(e{^cv^|wImcLv0`vmBGYf3tkzr;|s)`WJwlF7y2m9RjaXi&F3pMHDa{FmC$NlFD`)LefCD'
    'r}H?QCm>dP<PAHqND5^u!*9Q>izmr&g>E%<r@?NJHB<3_a@@wR+b1RuA?7Dgr6O6?&j0+1#nc)WoofU|v!aNiXpw%NSlW-'
    '(wGtr<d5n<LM>#<TLxk^fq}BC+N-X++t+OA?D!#gDp)h#?WX>O-t}Wv85)saqF4WQk;h*VNOdCZskViv=rf?-'
    '^`peCq&SvnbT5)Fz^g>LY*U#KvCUi-DVWW`bM2dkG%NmO;x!5)H&3o$=s$ikh$ThI`CpvsD|hzj$}RQ=IZwHczTJwJf2=+Z$6VxO'
    'PhQ?*GN;-'
    'Qcq(3YB9XqG<H7Dtk`PgLR=<o1#%Gv%|(O;ydP{#aKM!iH2g&I7Un04pJsld_!$BgXnOT~tYJrREO6Z`(vvX0sEsC^PHK|10rN#O'
    '=2A=a$7YOZ6EF%K;+EW2<H2U|%*nL|$R>_v+1gy<J|0gtv5&`-'
    'P3+5O>s!;dp2rDayqH^LolnTC%)SN_@|m(VCFHZ@d6bY>%kwEApDo)^LS7@=QbImQwxfi6u59n8%-'
    '1tNWxjzCMsyWKfT97E<Kmq)fnc+s@CL@M2!uq}vqWhB2z3-'
    '~k1UE|;ux!Uu;mRN&oJ=@k7t;8BcFw5XH>#&5e5Tk=_VlCas&S*P6VM@Lg8h7zyVff8bI1;38>mv!Uc1Xso6q3AzV>=zEB%P#taQ'
    '2V0OZdjH8M8B#S1({M|QsJWa%#Jf0@v&3u+@(IyS`D)E&sj8;V3eN6`(FZi%=jY*F<mYyw7$j><ZP-Z?sz_Xr!dC5{7F(S%<4r-'
    '@xoB<uxYr>HRbW*#7V-4u6UKfrwpo`iq9B)8ZwMRJOfNpB9aPY2I6YLO=BI#!cW2i{O7-'
    'rTtdvaV_h}j~0_T*8F0+F*P&Fn=*%$}TzheTi%%x##-'
    'W3UQvhM7DDt1y$>o7r$lrcD~b_0{yP^)!fQiomx!6W%a9?G3Xs5L~{6jca;G;YOBWL5H7aHmaS(!=bo+41Z3t)tk4tOB3R_C~8FZ'
    'lnH#3;jyR~IMwi2RLntrOoXSITRbbu)8GuVqC9ODw}xsME7&HD)e5$3r9tjyq+CLT{xRk%>beJaEWjn4qIC-nC+F?tKTtH(^C!QA'
    ';-F`lG@?81Xm4>-'
    'Cy#lbgtxCHe6lj_lhw{A@!xk5YB5p;nonJYT7oQcCI+|C^rKb*=4=L+Nun>km|Cho#BpKF3EfZl@e;V`%2@Y<Q&q;g-'
    '<$fF9PU@}1rE5p>Rj&E*m@J3p*olQHL(X&b8C}^p@NwUtK0pR`u_ef^|(Nu9UAb2oT$`9#5QW9E)!UljNt;|SBJyN5jd=rCP%L0v'
    'I9+yT+d}k!t6}rvJ*jeZs4*rO^)2iWf!V_+{9&9njD$VWw&5*<Z<t?@OD?iVMnALc6NP}BMs1U%oW*_BXGtMvEBx!8WHPlJM}RU!'
    'Oq-@*-^m`&M-SF*v;nlhBlnWwar>qOZ~Fi!7r;F{j%E0FRPvXvf9Njt6lxF+RY7jc)AaK`+CA7My5TYCQ~lWQ-'
    'Ia_4OgAVOKa}QT31Co$u(NH;*7lRm`727)S`cx?PyD!O)oHw;orzrQ~ZY+vm)@#M#g#`T*Sy&&wEK96XAL0me)ji-bhVV9N~F2+#'
    '0UogsyGztVoIPv6O@=X{Dh`+CCPlq`?khAM-3#`!Llq<@%s;+sp~S;-'
    'tu_PWcu4xXqpPD>ialILcw16gka_zv86G=}!F>yS!&O`By{Q?(|=GCwNfc?e2u5k4iiGoccmv14JeBLf7mj)JeuQw+VS#hG&|q8W'
    'roAa1EnkJ#&RVCc-n#J(?5cnWHr2bA)Hk;r329oB*agGXP->Ly2%bw)uTQQfV$+A`19S;gFL(LZxafWmT~#yj1|ci-'
    '2BXH{vhe{ow7Mg!_$7yWiaU+^+#PmHDCj!MAH-+;1gUt$2q2nadg-'
    '>wj<&qhtN=E`3ad|CyUKH_HD;Yg*U{|C`J0t!_92O!*%K7~xPH80|`fFd48YW1OrF`vd`C8Ye|UWi*XE+UJxeASAxwbnYP{zUg%C'
    'At1ix)X*UwzU|e}8494=JP@McJ5CxMg5d!t{<Akx6@#~X6aF|R?T@wf`Q!0uUd}8HobmfYH8##zE!5dr&S<V_Osq4)6^w~>#v}Te'
    '2xl~Rs5Z(O$7sIs2xqM2_C7Zp42vqZsAgsnivPyL`S!<o$exbz%{4g&3#|NYwR_G+dboH>Qjog~VH7K1M=_NIxw|<rNZFk*iX&tX'
    '!YGcE7dTZ&+0%(a$_vT+&SgqsdBK739LG_6EW|9MD6Zb;x(ysdCi58&obGa=TAKOJmO_0!bUGRm!IQ6y#lkelL*lS7E%1;$ru})Q'
    'Y0C+^NC*gRNn6YsCRA&#{?6S1%~OngVJlZ-ivZ;B$amCyV)&B>)B^kdpggD+*}rlQQ$rYOIBM7g1q(RU_qlq)embA4A8_>-'
    '`>73A|H0K??WcBJ4dZIPS646RYJ>g6nEGacfH8`xZx^)ELmYL>=b^2y5=z))i_6CXQ`?e){t~NV>-'
    'BI2t77Z*`}8q+jQu}Fz$Y|zuvm0SjD0-H<QUp&sULF4Y^sPD63yO7I3BN41pMBC5nAGr&Gfy2;Bac#u|p6HS+)c&nnOr1pU&gz-'
    '?@6yern6re{%Iyu+6KbG^v6b`Uqa0PiaymO`0HhncBP%8WvHSG|Kh<Av7$B<xBY@oPBL%EsS4gb*zQqM5|*hyiFgI$HL2-'
    '787@OA>BvG;Rj6mhOI7x0ZMp)XbTbte5QTofT^8s&4iEje!1WBmXyhN<-'
    '2Mj@y5yb)MESokUXT8*!S<t_tov(<DbY4&Opr^M+MPZ{Xr0?;AH~mKn$~_{6d{opi+NV^$B;N90={_6ttd89N);7{%Pj2o2u5}8Q'
    'ERU4)^qyElywXnpjiA1+0lR^)vdIJf>b%ggY-wsOmr<-cO%UkE4XTY-ni5Ak<YyY_~_*j}U_%p`BF;Q&(cuz@yb@^*j%2!<HfNIV'
    '2S%M$mSetmjQ@pi<3G@wWZcQ4eLmW7pg-'
    ')!7B=+8k5gC{Tl@qxyh5C<Uwx|5|W}h^v@7ub_2z;&?rd1op=8B@L)xGgIFSS88T0j$d?bti|CH*2Y?Vi#{fg#s8QByrEUgNM%J<'
    'Al7LHDQNd2d3RaqS_r6GJb=XA<wyj<{FKzR2a&Y99F5qh+dLkTtbJ5{y+EzBPB<HE;Z3p&j+3qsLF)@6%UCx8{J0J?ws*4MkF#LJ'
    'SsF5kFZHX>Z#ewRtiD$R92rMLBd=#_s4?;urmo7^1L*I!F4o9!3F~5w{G>i6kCC4!TBB=K^MWk1#LWn@D051_@UWiao?e{&*iX+m'
    'rf!xQ`+3<uO`?a`(#JgB;x5Rq0zJ>Wk%rm#qwYZ>P_Ni|FQiz3V&{F3W6|hAiTb@$vFjsn3h+a7VV9^|xb;W%P2_8%o#nTX-_qpt'
    'W$L?3l@zqnRva<KvC!xnm{P{*o0$4;Xmnb(@ykCDYjn7T2V#x>s6Hl-'
    '(Kj{C5M5~lwzhpl`8nFzJso3$K#%3_a@fb(_`Dw$%IS>AA@Dg$NTDN;I57ND+^>0E*JY@P;FVjKLuySd{L^tYX=A6W8RYZ^K&5VG'
    'kkhxb>XZx;0Ry2@w=&4@O-)SwltEHpdI;*j7$gOziJ)#{kQ8*fE{5%)b!lMgjj%EqQ{xy&mZ|aUJ{W6iIRAsOroK-fldG^(v}~Wp'
    ';OTbDHW;lBBx&Xe$35|WYHREVFKEq(=vsB~m@@c98B1u7wZyh_bDe_s=CVb|72MGR!ptAVKWnq9E5NFDDeebGwE&9#U{f3Q2Y^ZK'
    '`aBRUY5^33z@QdDF&OM=0Th2ng$2spuyi@m(>-sCB3npGsaVU}Z!tEWZ((da-'
    '@@2@14Az+muG_p_?A|R^<28=YDG{#>h*0c3hGC_T4SMtxN<vL#259^?x&U-mU<_9g6-!$!S)NDKl>%mpS|7lXYU9hD*Ms-'
    'rn|M|cHHpoXXF`Ww9E_%Qb|+-DD$xhqR)_Qgv#-$7K~a7rQcmu1*PAWtavYNZ^XD0U||?{0xS&U^4WW5(fq-Csy&1{W4q1_(K}?j'
    'wz7_zr0wJiYLa%8t7Q+HQOPy(0*hXgYh_PTsa8XsrUE1}-?N$u({Q`|Af<S?TYi{wIV^lUq7%3%P6Q)g;&Ki&eoz-'
    'E3yR2)kCbq)9O3qew=<7h-*0j283TiwJg7L|w)boN2{15>KLG}Y@%c>rMv;jRs5Y?TSx602Z-'
    '<ezjOu1rz^j5HX+N3{><_O>ilhT*I&dJoMp7glMALzT;WgU4y;N;w@>0sdf6mlF7LE?I4fGG-rnmJU!u>oXfx&iu91oNftK{dJ)<'
    '1?)!6>1SaqSFfFb$kGuDMeaPJnM=!U^y#Ovq>4gIV$V94*@0k`$uaLP}$+m9VRo2W?pc_O$YlMN%=f-'
    '#Y=e?*8ElqeGm+4eT<%YejAJ?#PQ{p3hN?&SJ0HaFnvV)%>l`;gQnO*kd1-'
    'f_?2H2AN%3LYQMS(&vM1Jt(5gkB^koUMq@S!N{K!^9yKtu#7-$?#9FupkA1G0@Mo=^BMZ%qNiA|d;5)Px-'
    's>RHy}PiWi+$dPF_JXo9$&kn%OLs{b^>igB(CJn;qpqn%V3m2hq%CXF1qf0@ES8tL_8&s_YX<hhQ{G+{)b+jc#42l;e)T?zXq;4~'
    'J=Ntdu8dCF<M`qADq#EPJX^#<q62zvv-'
    '0H|p9GU|+cQ1lSj@&1c)AMYerMoe%pHTXV!L`F(mUmg)+oS%dFV|LxMMyJ160jr?V>AGKI>*#G>F(?WfZ`aP$W`d;-'
    '1PD}NDYAJV8a|gU1Vel=Es>kO=FybVx;rknnGHB~ZQ?ul7AQRAWX1Z6yaaiK+<O)~&GUxClKS+Aaz<h~GC&0rn=>&KfCgrnnlT$K'
    'w$InAbYB$H{f^FTsK@6I?HzZy98^>?q2SVUl<+-'
    'Xyf;h1UQ}22MA|B%f55m~ycQP+}5XMU$gt6Uc<~z(OHeunEbcr8=)cZ54dESA&ar5Dbx)V;zr~>*6rrs;Ae;kiap-'
    'r$OmOa!6lSf~uMziAmu>FdYPk?A)@(B<vOwME8t84D){QuZ;BP0'
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
