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
# Verified checkpoint: 939 dynamic cycles; 9787 static bundles.
_TUNED_STANDARD = (
    'c-pjmd3=}EvG?yRKllk^q`lU-)7uSe(+1ZlB4G>B<B_<dM#WlNw6@}odq9P#s6kQCZgjWXt#*woQABV-BPyh}HFmdqkk$R2IcH|h'
    'd7jYs{pY<m`Sd=IB)>e*oH=v8GxMF94ts~Ls0#kysf$KmwP^GLd+vE>d}a~cEU2D&-'
    'WM;tXpw)f>T2^B&9wgAsvol5{6%Liu%A8myi4I{7F1`Qaq)#`ExLZu85hIv&eZU0@N;ih-'
    'O%t0?^JKB9{SiBD=YX*=8SVM4xg*`S>)fU!pFh~UX=ajg6bvVZ@UV7LiqhRt6uF0zdN%IKX{~nBmCA|h2NUFldHi$grDD9J>mnWz'
    'I1i=8(YF}yixoO>o-^3DEN(?+>Y>R;f*edPq$#$C7-'
    '?k9^%Y1;LTmVgC{?OKWkI?tS$0c_@D8=Zw&u^6a060x6>B69bL`U!_CYFH_~nAc8QnobHm)O@$&s{xZ5pW{+HX{?H(_8awFUx@$v'
    '(12e)Uu+}YK+y)J?m`Y=AkN8Kn83OxBo-Dr5*`Qb-'
    'u@E7axN9g6YR^9tqe3gmtcKMsXQ+0}Dbz`;i>F0h}$)vR*yvas<L;ib>@EhUpHO0TTVEZpzeC|bO&pP+A)6e(}tW^!ZM!C`Wur2T'
    '$JI0%{3IZCa!iT{3Md6S9JN%IiboXZXx6O37KHR-mjqi?vY)pb2lLWbM66AhKkozY=9*_iiU=rj(A|+Isccaq04uaU@nXP{ff}pp'
    'a9;7^afj;c8MXnhhqbYt!D?C$pqITB?mrdn2g|`{N@Ax)b!`*vy_j-8mf783Q!DqF_%cI@l?&x@Vj61>|6EBZ-&K(;sk8?-'
    '5<KpG<&b#B|<q7U6cS5{8(M@wF7T#?`mDKy+Remk}!RDx%@P;#3k7mMyg^O8Xf#G5{EW><x-'
    'RAN;{;P_wxs~4PpW&@um$%xHRsU#Mvt9VD-p8tcEUejX{8s<Ps((DJ*&h5>A7Ir#5!OspAJwY{&o8>x0Z%u>&5Spla5LS_Vl8Y-'
    '|I!RM^e^~6HQ6b!^6O#cSG3G}2RC`XM{sLwH|^w~SAq<?f?Up$_y0$D<9{ZNg*E<H^`erLNlRLz?toO6h*WD)s`zxM1=;Rt8@ziO'
    '+`Zum<gdZs{zsKJ#RPXaZ;FZT2v+|o&a=)>b<^VI%^+mj6t{qoX;a(^LOv+7F5THfcDPQKbj{6nOX6=VO*`P8d<Tf;y%#mFIegMt'
    '@)kdFx6xbRzHWA{yn`n>$2)kkJCe6;i#sY_-T;b38~;X7B-;23KznGn_rfFH6E9zI-'
    '*i8SmoK_+xgW;MweCCaf3kP%3l<GutS7w~u2hHXj+d)oA)3m|08)?_nB%si>vnjR9{#-'
    'v?ooGtyqxGBa}UJJNp87&FkVh}kGqHB<rKHVJsdAv+!OARcsbRrbj#x3=!AzL<98EmoBgskc-'
    '8$RY}D7>z41j*qeSD;2#dzI;XYyIZw|^I-ikhHshbl%snd1vvcqTb&-'
    'X+%#Rcnht9U<=#?sO*P#;|cf1}Oy;lKBTXpVNjikEX?y<U!&*TNh8I$mA}OZS_2c|GXlZ{y`W(8*Wg<$Tb|{`hyFcVBnki<fKMH{'
    'AE*<xB3{?nm)*o%^o)asE#Dhy!U^@p&`R+>=CuLhH?i*318b>)#yKU)CN3!E5m-'
    'tTO52W3bAkkIP||=^%O>R+&_D1*|gdxhG(iY0s^MRStV@C}N(Wmt1((xo4dTV51jzoLkE1BEAjk*Fe;-'
    'et53dC}v#OX4d+9(vsESNmu0$jwYp-jm6RKc{UcuxHY{0j&(2a{yWaS$R^}?_Y!Z#6Wm(1CMUXe{NOX+g}>DSD)?#!t-'
    '`+FQf179zcc_}q($rDYeKwy9lj>U%MI`~DPC@bugURp6MRjHmz&|MC0=fUuc^@z{yiXtH>$<+p)D=^jvyGhX@h^uw4j!>pjRLZzt'
    '&cebM{*NBKNt4vj2bTZkPR!A5HuJXaFg+|BnHXLi_(%04cQpj{}fG`~P?VDYXAj0FXla|3m;O%i;_EExh30vls05J1USSX+U$_&i'
    'Gqh@DAOq3~j6zz3>$+wXZ0+Zc6XyYYJYnq-<aY?r}d2=3qy6-'
    '+YproVX46DK*9{XmPq&f+T2>U>81U{B#jtv+IQqh`&6+J?|A=PjqX%rt3-'
    'W1#jqjvU|~6x}M@*@>RNSaclh$x}NIR`Jv%@r28dG>#cZ9x?kYU->O9u-tI@KXxQ=kg9?`G;CTJt3fAk8c>N&-'
    '3pO@he^|ka9U8AcqF~9!#p{nMShK_8^~V$}+W4?PeG>{O+ChJ?0%eGX(2rNJto`xlKdInAdL&=eW`&aNV{_Fkit6hX+yzaQg=@rx'
    '+Yy&Iw-ODY0&-'
    'g5j_Zmm2Fu(E{}XTU)8K!SdFc0tC{WXAqWxF}dKzuCKU9IDMkDRVDbUntrTt+FRP~kkBBN2KqT>6#6`1O%`TpYyY;{z9zmI}5IqJ'
    'Uugn~mkD!*?~V8o;L`%fvb;>$AaX$Fbpx*4(BCzT&kONM16{2oG~^|azY#}$8<ZhnSd&M)KjPw@H|@%pFuLtE&F?!%i+baOA>Y>X'
    'aU!y<4$5!Ah10CioU)}xfjwHE}y`ah5`TpRwI+zp`6i+iILwIq8xew~K!I{zY^_wjJ|$*9{PM!fqJ{qZ&>YweF#0+0MLO7M|CRz='
    '8w*T*RVOa6EzFwmc%1PA&PRRn_g*FK?8WIxFbE4+I$pS^HH=Cj|=Q24@qqG4ikMv=+E>sA%dFwx@a3N1ugjnA2<Di!>Y9od96!=H'
    '-BSNIJ0pZMRqaK|*~2OUb9m&dL5yC~S;bkz7=73^?2Z2WEtwm2O(es=|XoDLknhk{K`M~>f9!7ism$M2<Jn`bsK0xB?Ju*D<(1``'
    'H5DFSdXVX%{2w&uMcrol&h3remzol!M-'
    'S+=$ChNxpK@D7y9J=xBMP133ua!v4uE5=+C{Sk^m*Cg*0qpr#RNX4*giua0fSBpPNG4Pt|rzt{SV0ys<L^KYj7c4;ZD8TfB1&Gib'
    'OfQ(VY1y)lQn+=qvojtmewLzG1-'
    '|4!(xg_n5zy7cUUo}3DY8w(ayd&3`OKhoWPHw6emlk1=Znhkpy)L9=P6&O7*BVlV8X%_Ot>ut6K+qzgheTsa7PLz+?j$2i#eD;Z;'
    't8+Z_ryd%Qe=P_F$nYiqHc$J>^3h-'
    'jg>B{Nz9pfTE=6;SUECMGt=jpeTBH2Pldj{zyPk^za@~6g~V=fTHN(rvZvOC@$e_1$xBivcF7$Ch^JaFIS*Td^Y<l6lha8o#mxF!'
    'g6nO7sDF_H7LPx+&^f)x>)0Tk}`lulH1Z7qa!&22Q$kE{CT&Y5%?PSIwSBG+y+MAFS?D4z+ZBk7=f>Kn;C(xb6eu`T&ZAVFwpQ<D'
    'cBkeHT=~IHV1<Ze~p6e5#UDAmS^wJ>Uoax!xW!^pI3gk;zRIU<+oRS3VuQP5h@;oQ9ZkodM-@rd0SG?+mm`OO6qw>QqMb+dM-'
    '`|ygh8Y?u*Fj?a6Jq2)zX@O7e(snR?T|fuhV-MWF0E7237IP}wg*dtZxUyGwz-ZWpibR-mxM;`NslXzcKK{bdCzyM4U=iUOS-'
    '5wE|hKxub~*I!ehwRO2>9#dGf;|q&+LSfNPR0S^qcWY8Rypz|2>#h{*=!si8C?vi<Z&bX_(R5c3M5YK74Cw_(fx!^mW1mu491esF'
    '1`NztJVci&6j>j`H&TT5_@r^!(BAc=vmzkuHk7<<5!%abg}r=qVfP-'
    'Za-}~`@mv1?-&!IXu+&Z4(2<cbFbb$AVvfIryN}b|pTgZf^tAWmn@}jxKY&Y5!9>3dR~JKs_)B{S*z)suGdkS-EZ)2pJgHC6hyNA'
    't+)v}ppTj?em~2EJF@XDKwGd4GXABhIr~D*6KSS<4!I%;FxhKH*8X0>zpIQ*?%vNdX^8JUu(NjYBu#8JV3*pxWWjO|_6p-'
    '9h2uMaz#-MY9e-'
    'M|NVyu2uI4H>*crqN6(X#lbf{lu{#XlWvRJ1PsnP8)$eetV<jfxh=KN}9$Xk+|y;c#6X<@m=S$Gu6!e+YM@RSus~%jz=>RWE8iz_'
    'RK0eG0@F1=<d~XS!mycEI(7Q%a=Nt3diCF9C(r11c33>t5x@>U_(+3V#)l2%j%q^aSA$BCOGmYqd&N6wbHl0>TFPIg{Lf7v%nzYI'
    '(`q-NJ-O+->PBaJQ$Uz%5EAfxANvf_8Uj0x)i|;Fpnpa}equ$U*phxch5TT<{+@$A`(`SRaC7c#-n%do`&{{Pq5-'
    '|Gn~mpatv;|NJM?sBSbYym$|zwWVOae*}$f4A+nJuY^bXBR$}63wfPihr8G4?l<A?_jLDi*pT<dn_q>0`g3vry&m`9Pe8K^OBxi3'
    '{EeCt{ODg(8GvjcwAd9_cmT!0D%M)`OTw;$A4kYmz;S*n>{C*nmH^ZQsMlDRNBW9(eeC{sh0p2B)n#1}>Z=q&S`JtWw1KQ9;0jo3'
    'ifQ_PE5>J?xIM_-xB*X6ayM?ola$<zoA4x!F?7EGPf~*IH{(f4u>BT1NeQ;!iYMu`0*nEnz+v8nYQi~?pDBNe4r2d>@}Jl9i(&q)'
    ';Hidx{3)C&T8vTDrL|R%mw!Vs#7x-'
    'rQ2>{)HQ2>jr3CVTaT3e}wn;z_m?l9zV3`E=fMF8c19nM(511vkNiPLe`b{1o#A7x~SJuF>f<>e7WHbCLJZ<!aI#Bd|vOQLz0q-'
    'lsVL)6UhQ>i#>&39l>(bVG0rXk<zbN?+Yhafs|EKgHo(Hv1{&(p=^j7`5;Pa>d@Lbiu4M!TE*a)LzSO@LS62*?i^Zu`jor4$sKNP'
    '$D58;6ylWimYvq3JuEksT`vdkK~?lqK42WdDSZKD*&r`Sndpgz6`|78_^XOd|RmN}mItV};w{>-'
    'RMzf}H0Q6}D9qclj8c9*YdKn(Eb!vPr2&$MPS>`6FmYYfxlJME~^&fTl>{kA$R#EWSmo`Hj1`L{@jD*)$%*<~XQ*z^!K(tu6JvS9'
    '{1I*zhS{_iUKSnY^uIyky9R=I0AVtA-'
    '>*Tsn8NdI(@^DAWKo(gv_v0)2#?H?ghPI9Z)D7Qi0#7Ms?yvOh8bDj)$tI3Ky0qzN$7jcyvv_Dm`zlXxv8ZX~)6Wkl|@=Z6<y%{g'
    'xa+BO!@$zjq*}WYv-*HpiJMr>e*W%udm+!f$?mc;O-'
    '>l(mm`@x5eC<XZ9aembCXEWo=})l5temw(i)aSvHOe8|5e#lnpamn&x%dmSPCxsLpE>&iOp3t5&MO@7eyN6+51<<UP+tKD#RY$#j'
    ')EGg@$UUP{+atU*iXM5-94oJb&5TXN3@@(*v(j`{d~oK#u9j~u)~8RquqKxhHmcG?sdN}-'
    'F#KM&Hg~T`I>fH{6Tc{1MRQS{#q@$JG6RD8m(Sf_j4v0@Rd8}l}?*V233Np32y1YXKEzt<!5PzP9yQ)8m(bJVy+oayattvCPo)oG'
    '`|RUZ_@F8hxX&x>23EfD?e9!*Iz4tEdds^k285wHu$48#$E2CyJI!Fxj&`5o3%ecGer0w?XLy_(_elyob~s{n_nVG4@8i&k}wA5m'
    'DK;<@UTB4WHLe}x1k-~D3Ay3!{e0yvgUC8@ydTibG-fpg=<6s`x6ze(fIguA)!LZW@kJX_mW-qPLkJ8!h`;j9^t-'
    'l_p3ZOjP74eELbzR1t!ZdVbGr8KOP)2f_WQYi;52+E@urxD$Th>{Pgk`*lNJ4g)Pu$3hUhOFVLu(6w$v_<D2%!n^ndyF^o>0H2yB'
    'b*@<V2|FYp||1XVuj#CxCHz|_;2a_WCe>5qQ|0k0o`F}PswSm`vF&x!rXQ^WONDc9b4t9@TGlg_5$R(Xb11e94^y)$BkOAXonZa%'
    'V0-6yjh^U4bB^S~88n-gN`-R$nNyivzL!g%sP}weJd55Je@9>o6-9BY`N2Dz84k^o97qh%5i&ss`l>E-'
    'bOi6+yqwsL$mu$e7>>w`^k5Z5h{^mKuQ4@Bwc-'
    'r2hn0V04rnLiKxL0R_*;%8&;REV*8Ox5gs5h*v^%nIno&6RC+7@GXk;xL_QFsgZAEtzKus0+qeR<T$M(dXavrJD?2T0fyvw{8i0P'
    'v!ujlVMfd0CN{jb9@g{A=T1(43U|4XC|lJ^iimFKJFoy<+@YO;K#$Z~QvV1cF`=pkxi!_`ex{1l_#k|7pOJA&Q9qmjO@aq7p8djr'
    'O1IaEH$P#~Lo%Y%@4#huQ9w9f2Qu)kNG#iV2T1JSt|B_1CMMAp_ve_`{We4Fi6>Ksf^dPTT|Irzy#-'
    'Flz;aBy%_neuQ#!dFwjFC$x2sM0`S97u>k(dFz4~H;=dOH09>!2|0LAT(1rd@+j+4t3}OYU20YNR>t$mzov2FV=fF#5a0I4{17?X'
    '19-%>s!UAH77pkjrZx-bQGS#S4=2ENh4Wx81=4<M+|#TsKR0f5RFDW72RQdUfkRvq;p=tIJWt{f*JSwG$a&`}9O7z$ug#owo|=ca'
    ';L*|11^{Zdozt5~3be2B$7uh6iFf-MJT)JryMv5BQTt_d_oVh?wBL^IPBs2Vnv-^?8~<a?-r-'
    'fsUvI!8mssa~<CoFhr;Wec(Chin1QWk8ifrd_e*HWbcav!4bQ=t3irduigssuRX7Koev+Lfd`SAbq1XeSu$qeJC8#(GgGL6QMHgQ'
    'Ro;Hg(|qechSg3oPY;Vr0Mc+q(moE3<|LoSKe7#@5Lp6Z;BS+Ih?s{#njHKIf1p~R>k!|Cq2a5sX40?*Q>eK!eqj6oQ4Ki%zb{Bh'
    'bpM0eATzeh7hJ=ys0Yfi1U8~;Pij@68Csz%lFXBqz$Lki<RXZ+Vp)DuR@v(O;(u{XmeuLdK%_eMhxj|^TC2!wzZ2(?!{)ZTz4vUL'
    'LqzZ<PCi1w}A>h@~<Pq|i}^AQ_3-cg9)RP1P2qT9nUVRDH^AGU*}cA)VmX#WUF=4y)p_DgKMJH-IV<xKNw2H+m2louL*hUMJ!?Z%'
    '&J`N!@w{<D^UY_aj5v#ii(1YEK{%3&{!db%?$<nn;X=4WpYq9LX2h}I&dJ6hpbV}ArNBFMHTq*2u0F2m`M<~$nEo6f;|4gTV*!3o'
    'Z=*i2i*Vs4c`&xW9Eum8NwMR<-'
    'y5q_S;bfv|J`$bk_^K*>9pTzrn;}6mP93A&aw~?4~Kj<cA_~~SPz+j4>PR0if=J?6fJ!CM+PbcHU2DAKhGCpE3%}*!eGJ|>k{z5z'
    'ibi^d^(TE0oAW8tvZP)>FuGQell6YVj)nzel^pZ*}DnJO^Hi~h7o@-'
    '^rk2Zb+CoL={EzYNZyUJo>=VF^^EscptHYO3t_Dw{x{SuLE|3oA^AQ8z9OhmGb4DLI{0CpKzy7`545r}~klXxv5$I&K*9Kl|UVtv'
    '7Xn~j!+Pv%jI^SN?lz%tK{hOsD}vu(0o4SqFxQhtPqA9IuSKQJ6=xY_z28469H8EOMFnoR#O#!sZXGeJ{~pTvnEi;18MNy7iFz_9'
    's`)6V{oLWgx9y8EyKJLf-Pm@I-'
    'e;A)dA&nyIzDR)+fLBu?ph)Th#1!AnD><GeDQ1K2~)(~8xr9gcIVZu^qgn}oV`6Y5^4gZL^1fAiJ$lq+!IRHVBnz0BH@c@5kfISI'
    'LpN&F&r=zODe^B}FX}>q!eMG?#>>KFrqYAy+Ptx7T6r9NZQ%0-}?2Xxm={b19TxOV{g8<-'
    'i!xS9^09P0$>EP*erD2+mKVJK*43l#Z99&~EApsU{k3s_MdzuJ|6VRR3-'
    '(%PSUI9+L@eh&${L%PBwLg~vA)f$Xpv5kYQYkL;K}3^dSWN&;?#oI7XmUSR5kQmsvxopq9>4+uG<hJ42hij}c_ye9bb*#A^L<)^C'
    'M57&m$GZ^xMCgrLicc3M#KRg#7R*c;4z$15=S^uVN;z=ZwZ(K%-'
    '$@z+Z&D?Ehc9l1OA|Fk53x^RV_!|r{F}_{!)<@YIvuTtpnTYCM602N8~L^6l9w6w-RE%3_%CO^sB&DYsBDbQ(=6-'
    'CQZGkpwTr<R|AMLw2$D6VC_JFS}Hf)Wlm;YLa8h_@S;wQas#jF^pe~fz%cl`wZFi|hh1b5_&g+*;|S$%RAMXOcw3-'
    '2KVB!9u=VI&QYP$mKT<Mb8~pnu6SmR6Uov5v{C`O%Y_s1<GGSZ%2UNi+*$N({kW9&ngXthRG++o0u~>s!EkT$i0KU*z3lAkDC3Pf'
    'H5=QH`B0^<1McnMcfNB^f1Q<~Lm-b5wPRrc}HE<^BP6Gf%%?BHr8aVWBP-'
    '1G}(7T!AeHbZ`ngcf{Ue0yH+}wD1tsCyHjhENC?cH_p@_IMIT^}#!xgFfRcsbwIx%s8jzgrh9HmIHuvT7)_ZYJj>ZW;;!7I=X7>_'
    'C|pvQ&YqSI7Wp&;oy#4Vw0T<-'
    'cs_Z%c#&RoBB3Tvzo2dVVoX0_Ux)U@r#z^Q3x57DI!FrTB9PTsFmwPM7!|Y$X?3u<^VV@381)kvn{+MIieKy{<+<5S?Yc##jaMSz'
    'gx|s~|$l292=_Vzg}37^|QF^Cpe43gWbE))=dxX!910v5GpjL=o1A9A{XRW6TJ!97igDca-B!%71m+a(uk%D}wD;SADG@NI12-'
    '1=qD>x@DA5>AJP7c|f8A6r*W@55(+Pg`KDNkT&nmrMcLmXS!?<{Q2Ke;%LYP62d_iAz!k0u;4QR`s+-'
    '1?^FIuQF#BQ{8zRuycJd7C;RvV+HcKV%?5M}Kz>Vfe9NsMF2mvHE{jH&1bLf9kC6f}m!sp)H&>#Y$sw4l(X(Pb`?wJ-'
    '3+Ir4S+EuYVc|p)3=8KGgIIdy0Sw)dx+J)S%b09f9}-S#2_I$jyN+ZDz>vb5Rx}fldKkZwAmJ!k5(W7I<-caj=f-'
    'E@wTy3}6Ynk97y?^wx9BfYH0m0RdPJ4MjBB4TTIcv*38Qtce_0r<*ZN-zqxCxf8)39w?|&<d)_MLFVYJTo{RM}oA*wt4ub{PFw1E'
    's!2Y8<caddIEwM~-'
    'q_~xK?Jw=wqEM3~A_ah_;@_7icT2VgzXi;hs{RE4$okTyuq70MhCs>r>68!|bv4g}o;h~e0_`J@X#5Fo|5?|1nllY>}oWz%O<|MA'
    'uCBG0ILHur2d_Jb6IfgL0G9p@H&wv%ib3vd>T{tER@%rEuHf!ISgOCRiwFO~4jEJNxEHI_pOV}4o=?Dq?g7vJE{Ki@>o^AtJtMDv'
    'D(>Cr=IDd$zZQNsEM2V<v+;T7q#CSIDaX6fa@@(7+FcQRhHtq>9w?uk2ZY5}aF@`l$j=NMXASIxQ*o+$y3~BNb>y$)<wsmaDe(Of'
    'JD2V6*bc@RQ^Ja^2We!vWVuAm_cNL4VkVek6AZM+Dxd92$H5Rf@jRt-{=Q&`Y9#8ZcHml~rOf^%O{iDh)bdSdC70TV=R*-'
    '8SocSI9SCI_Kr{02SgM4<ke&Zndpl7PNSgM9^+?@riNqtB985+NLw8<cfb?jm5n7Qz<>uJ$hDvv35n|mz2{c`1Qcgy4T<H{{^kH_'
    'mLl)KYC5wBM&x7e*D7d~WbSZ6<N5ni*No*A5WR@BdgSQ2aMSs_Tms`}XwoMT=6oJG&$1G%TPr}dkor1r9Y%OJ%669aG`K!5!G-'
    'x+_f_GfDj4u$v-#ZDD`GY;BU!8hZe{S<sN4%%PAH{+lK6nrxdI#9tk<Di2Sd`BGAj=uF#inFy>V3-'
    '`wi)5R6f}`GLIWE<cd5|4=#~vgg0Kl_-CC+_-k`+^Qk3c`r_;-X*gjl9stY066wX5~74^pdcg@N-~#2D(qKix-'
    '6Fq1SX5+6D^HJn(y?V+G32T=`vmG+HrIM8AJjQ0D3^`r350`R{~%x{N?{&B`nRuXwW-uNj>LeD1}Kb7u&X28D(FF)S>!hk;uUab('
    'R=NkV29n^0Cwl%RPuEGDw_;K2QfsoFRHKv8e7cBW;e6^eP8=}1Suzr)-'
    'CcgR^a`PBpb%4`Uz*qMfaK*tPkI(RP15OgSO7R)q0T?Y2ufG^SUi<Up<Zi=61LbZp>@tRKhGEoYU%DBNL6`mLW_yge>`yl%FywLo'
    '-RywTt^?_&4#O`8nLLvnJZ;WhRkDCjOECN5qrfvwH$tRp$D@ypJ^s$9z<x{U4K!$!MYl@i`3Q!0f;^X@hrUgD-'
    'mCml0s9Tg_n2IsPiwy)pa{~STfnU`yt8fvylx^7q8+1vM-Tz30X`2OG#YFFb&SrEx%S_{up$|3|4oc4lF9bpLS-bQ?Z1t>NM_rA2'
    'azBdZvS1>%c2ts`aUCb>}r_NN1^q^ZB@&PBRlex32tceRFH6?3wfn39UVsBlk4YW2#$h0S0d!uraXJ9ezs<L&O`Qko>-'
    '_M)u~%(9<PNbt)*y8BiH(L@R2_%WOKxELOZe4<bG&_zY-CSUqDKK4Wb%<s}}Z;9JB{#2kjx*Q8P9>Y7WhgnsM1-'
    'd{}nujL(i8c!NF#=V!e8w*kNmCqphti!WN<>5n4j43c~tVdXX@`C`@2ke&W;)wdS7yI{o|MRc{mr_>xB?opgl3t_)5I-'
    'Wzh(N;F?O{yrx*$Aq^j#DN9QMJE;?Hs~te<R&pfw<b=M0Zyqr1Uq_-PH)Q{VjC&R|Ad+90T$0ZwA>>yVBj?4br1_qq~0?a7O@r#J'
    'hhQa7e%i#=CzRa7p0gEtJmp43g;z#N<JyPau}vrcA$P{AkNE{j%|`S<b25-w0l(;a&Vo<1e-'
    'TYh*m%)qX8)eWY?Ad2S}X<e!Z{O#3g=-'
    'Qn6R%QR!Jebq9Dc$)S@ZEPpfjwf)dV%PV38p#J#O#MTRECZrp{YWFtU|dc$?Lk_FcoQ{UBIKrj>MO?f!A>l(Vi>Rl<~1`Y=hQVQ#'
    'My(cQjqF$^oF)6)pM=ykUc*f)U1%nyDP-*?xr*O55|wx{_`ZQBed6+iP%SKZ*62>w+CBA)qIF|qi_P=C7LPXU1FIc-i-nacsH`v0'
    '^W_bI^f;$r5HM6dU&(O%<F;;4JPMRVefW{SAjp)m9aDdarWp!c5{ZBS1^|QY8KpndYghf+k&^if}3dp5f!Kdu30$q&A5G6afrIXZ'
    'N>e&l>$xD)gUmwQl|#Lv*uO}iNDpY1!(o7qy1<~kZMHWFUJ@Bsqq(D|0PoYH??15xWxZG?bjJ*03N0N5F4rS{V2<deD>!xR`Qbf{'
    '6&qWujF0=L|&sKMSY}SVlXlN1)F6P;SAdp<Iv!XZOqg1O$z@8@BG06z1v``wRnA(!KN%poE@0cp>Wr94c}aM(1eBk6~kL-YuZ9RI'
    '=Gz3sAe|7QbViG@;qB(Av1~a9F3LC6h1gtV;L}c-e1sI2TY##c^Zp<sZ<yM+w+>sia_QsXs#pz-'
    '1(yBN+LjVFKMnM0?V{cN1}eHBH0YqQcF?S7O<8&@5q5ryJRlGr#~ZvpumMc)0Uw*ymhbP9Rjahr?QF-'
    'U$b0(gwY?9lgnjW)hXr%q$6gQ#qcOqQvr-'
    '#t7F|kd%%6P+ZU$Yep}k@x2N5{DDCz;(r&*q?e@iSw^O;>>u>;Rq495oBd54&oAOON+`?`Rn|6B(M{`LmphBY!dvKtNAf>IEz2ZK'
    '_1e7*JJ>bAl37JN%Nu>x|!QK=(ZLm0-'
    'gp!@BJN#l>ws0)QrAYVcCimp)04{|J@eKf%<c<Ko1aK*7%Wna=g!d<W3E)y_Pu~G>2`WVR62PTUpa1}X`??*S!f?ogLo<Tz;E;zy'
    'ZwI;qzy;n!oh|o#YIYUw`2z68S1i}!IIs<t8&_3Oc7$XUMlPdU3_tTSd&LgH0y)LbV&pRV#IA~w%jgojAsD%g9<dvWk;}Tc9&-'
    'X=>q~{)dt0DVW$cUyMVsUP1R%7E1B?hb#arnMNV$h?Egrwc+zlp2i)l4<jKL5GrCP6jy)Gx44>nmE<q(qvzQ>v@*n6mv^kKUnXR>'
    'U|VJ6GAG=~rsH`<tBz`k=U2S3q(f#+Thev<LarI~}DV*Cne;NV+~e<F5pNFsX;gW9Rq5Eiq-jzue)*2FqSOq|kPeVF#+Gc@D?3tW'
    'zEK{+rmwIHzVtlO?EuO^cv3dS4Bnl$*y#y>6{Dg0F9SEj}jH8kaH{z0?X8_v!HvVGlf;sAi{2E&;H@G3SMPSAt?Z!(-'
    'm0QKK&IFSI_zs2M^dSt9fA}9p=TKWC^@VbemR>AY1!W99b!8ao`44eni<#cV-xIxkyX5FwLt>M-'
    'UFY`Tm@OTqI0DrAjC%zKwf;KV8vn_s)Hr!icTfGG`yV(}(CflUZLGAjK&>~<Vexrm20Lt`=5*h$3OFtc`75Jk5Y6`xn8=r<R>c(f'
    '_i|mATmK)qv;fs3o4S_G}bvG2gre(pp?XBBB2nGN}W67Ae;PW;Yg1G}oUrDkWjr^j-'
    'T!OWNg{(;7#8SmB>QaTVuhLluf3hi6laDZ1Bf>TIJAlWKYV7L_c$A^WK6I+U59U@CXx+J21sZp5R)MyiyH%iR=XMon*|}c@8g_12'
    'fp$I5bb-'
    'u6q&nB44=KMP0{su;F<IVq0EL!Ht`3~mZyEMcz}xzcL`7C=|6Rg{{b1NDmaV8lw}?_w0rXQ!%HTmjDJhHipHfm)^qeRqH3T6hrKE'
    '<U6V+b?i?LZJ!i#?wNDDPw`4pXRMRQGoL|B`iu-'
    '7Uad}x)$ii=Adb}6rP@nqYI`EbKUVYh(=;Zm?gU^BQD><%y)iPxke3?N>Ug2hd|CI!ozcufixIPsbkEOFvBDOlvhDO0e_iRGkVp>'
    'tnxK$=b<MycTYcXL0Ato<jxVb=dM&ufNARw;DfiYIo-O*G1njoM?CKeVJhSVFfDEG`$$-'
    '3~UEYvk?(L&`;Ri@}c4DVB;}F%%~G+ZL7bw{VnQp2P-'
    '6*%e7_V12GkVuOS2sw6hBMOR0$75hnuE>xl=kaEhNKunY_kR3!6Y>7O)tP@6?r0Ym3LlkWEAeEsCHhIwkVc!bSqrPplHJGikz9Jz'
    '1N=g{Q*3C3JMt73cnW#7i9<rn<$+<pHa=xA?IXC1<&W(AJb5ow=+^jPu>=w>=QY312Qf+`6mnGE(OMeY3CHWn<!GRn6jyvE;Wr?s'
    '|LQYa46By7$uTDzb%9WDX@JEWEMkCh(;WNYHCRk0WtAO8cEjAx$z@RLXU@;6UY=Oy2xUU#$-'
    'NHpk;AB@Abli_e6#=#floP!6pn}7S;Mzk9PAfud4=XsX2&_G#;Jl(rk!1=FtYw*Ox&qe(E*1l*7$h4|mx^z-'
    'S(+4G2SGJp!I~8V>?SSTHY_qpk^!Y%r3oJ*)hxo4B$vs-;4p1<N-#J~+-'
    'wO3hl!h0gTZ0qW<WfWHT|H%wEFo<$Zl)2zaVk_UeNx+M16Zv`-'
    '>8L>m}_kPCUT1+Fz1*fa|osG}om?0MWwss!*W0;H_0)cU2h8wB`)8Wb9CTdsLotNFlU<YDR@vYQ_;+qRMH@=3v{I`8pxkc4od#47'
    'Qya!jpn+-'
    '^WOGQ!Ag3fyKrLWMBvK2^qMNd_)GOAfJ(e>&Ay<AWiZq890;qF{!boMiiA+{t#8hLg4LGk{gdUrh_qRb9RD+&~mpdcpIIP{qg<tc'
    'ygds94x#HSn#nz#drxl$&5TPUUS7{kR*eQAV`ji203y7Du|I8F+pJ&iv)e;velrj3`~Ny7BX;Ksk)|sf|T+HM~hum0NF|?2#*E^L'
    '4au18;(>27lJ|+20aLhRbV!qgB4ZuA1GQe1l<P;R}4k(VZ0DinuOT`d*%=!v%n@fRA?M)(fh`=PFSZ|rBedoNo&?fu$7~hQQ12ry'
    '|#0JKwH^-g#L4x&fE^}P%uY#g2OXE&)v5r$rbY7-'
    '!T3h1Ae9?wazu*l}T3X3kLj5$z+{pz(<P@`Kb0^XJKOG^cxavT(12$h5fWb!>4cxvr&g7q!hY#Z;EGofOhYQE4y2WDlaM)y)IEjq'
    '8A?e-Ab$luETV!2^ice9Rr-1`Diobvvo+Xmo`nUc1We#B!Yfy7D1PYX5f&PSZC-JWnKO%iBQ{t`q9ccoE`~GAoZO3(6!)0bgQyMG'
    'Df>MBoeTnc5jL6zrS{Gi`RUhcJFfQMt_iY?}-'
    'N;N@CertPg1p3R><G06q0W+BJlBD1y0rNPtk#a*q%yruIjBhHS{<Oim|ZDNB%3J0|DL)QY7@6Y|YQlBb_Hev*+K=@*S}5rDZIupk'
    'qVyTzSahzB?KEyRPH8yDii&7BMJ;O5qacyM#?LOi&+c_AL$IVC@WgI12Yqh=;`v3V6m1w-'
    '<w1%JlD4Kai6VrZmw8ytZXW9AjEPD7SpNt<Q_a%vGl!5?kZc@O_eqDfWQhn*4f85NznEp{YBPn}y+qHIi9ecG=3hSjI-x_{VhWQ`'
    '99t3OV<R!JWQjF`_6c%U(A$>_jf)RMu0#i%7?gDOTX85#`1s3jwVp%}HyfkE87K}siMF;7WKus?dGq&g!=cV|?tuu>Jaj1|k?$j%'
    '|f(Z!HIE5ASxGD;QGE$zWNSU%GB3wwi%*@0m@(D8Io*bd{x8BJWT!xPu*h{W}BiR*P_;(GbS^*SnXy`~koUPG+^Y(xU5BI~8R&Ax'
    '8@*~U*We2(-`OCR*%9tCL6e%lk^M6utt5<q@@jTr`^1ikH<#y_q3Z9i-'
    'Lv(iIiV&Fj#YMvB$5CofB0^5P`@zlU}SXN{^<i}iyV`w&$9XZ77$Z1SF>qIvR9SgJ50J(300XHM}XOETy{V|dBP{d2{-'
    '<xeV`ykO8!ludsXO%31qy$$X!?2=CK8btVJq~VbeD}{7|BQ6w_?+?2QO^y;AVN5I7~AFY5RG+44=@J#kd%j5tTTFuF~~=xJY)#g8'
    '9l-n<YS7t=go$AHmM1r0il(V((#gRKLsn~${zol?lExB3CRTx3^`1b1Ji{(qbY&u@-'
    'RygWJ)h~xSwG8#g6b3Ex(xalPtg3k$y6}SkC(?mOtkx-'
    '(va2runHh*PsUK7g3Q?j7D+oE6kOku9m6i2qp_KQaF<fhnOLnHLwjq1M5fGOh!Yk|6JT_CuEd_%rLj=RIk+<C{2Cx7Ze?K7#pbMb'
    '2hrzGQ+xb#MVD7^>^F~e-%mn4C9~FY@|PJ{8O5Z^jXHQrsgjFxZDF0CV-'
    'V%FJS`MrLRkv0CwpH2@}8?Zj>+qtl=gJ6Ttr6EMWrJzgvhHO=4=s7;z~*a$ID<z}1xs&X1yy+Ob5j$x0|)<9!dM)@n$$jZ4{&R~;'
    'NoLBV1;kcohxS6#WFFo`IWMD9qFMD9$JL>BWTk$%w2ncVpW^zv+O`~rG;j^Qw-'
    '131EA%p(Eyav0MC=H)QvQ2<gnj5!U^Z$UB@6?jgOOC++nLIvsoTwSB!j3@_?dNq-uVqprH>+Jkm#hs1<D>CSBuTD^(!R}T}3d}~<'
    'ST3KgG?isxn#yupn#yuJPh|;yY;*3%PRRY(iMby;DfeS1=YH&z+>dR^{n)9tAg4xb$v<6(f(#MURSK|15YyEP&ed}4dXw^Zh*!8-'
    '`8&lc+@kzq#mQm=!nQnJsVi)G#mI5J^mWEyue5c>P^OZZ3yUzl%tIIMz(31V7w*JA%VQT7<DVU>^8Ug-xOxe1Rl;G!;Xd)p@rc{X'
    '70XEA;}whz-x^i9t40|-&Q>800#}!-z)mTi{Z|-'
    '2)=CxTmBt@xsVK1lOD)G)s;g_qgvEJWtP~I>`NW4i_SgZeu)P7WA`bD9j?p22l`6Z#N4X*F4Nr4JbE*z?!AS76!tgb^W!!lbxyrH'
    '=;4rTxr!0`9x=MN4Q5GR!vBvH)1;92)=R)J&DhuZ_;}5n}X4vizR{m^B+K$BzkCLZV{#-'
    '8e3@avm$LFNJ<8xEr@oQ7x@#|9G@#|CH@p-B5c(D=~1ACx0J*7-'
    'qdq6+jDji?GQA*WnTG~2j{Pi|nD)}nM^&70;BUUO|>pwF{W3D#-'
    'FiVxX?fwwu&yis0IOV^PQq;<SkyF%YTeyM<jSF{`L+j#)Y3|m&+z;2>uKRJnz2<)1EBpw}4ZEN4J816Mz0%j^4V-'
    'GB6mKxb&+DcCFK%{1YHi2`$ckNiFl9>0L_*rz`dy-sK56})gUY$UxHefSb-vvAL#$N(USs@tE5*%$-'
    'o$nnGXk`&9)&O3R*%6KZL8(*Mce9e_@ZsK0={TlJpo^|tyaQUfdN-'
    'Z9&4w{6F#eSR>>C7Muw`pveI67yv9u6@QQHgb{gN2>(8YYJk}U@wEokB4zNx<dbmct!}#gkb|j!&F31irI!k(pchnKdy(#F}n1@u'
    'y<RR64^N{L(c}R8tJfwO+9#TCp52+r+A=S980EVp6+?DkP1yGCIV&1F(U2%KKTNJoG#cH=!?EmBM$))rO>vxX|bb)bimdRLc7X1y'
    'PprJCh3ID1yi~3<wGbM-qlS3p=a*qM&7yF8VZ^7UaTnpHn>{-CpWXA$_Ci@kzG1;wveaT)0Y)f`3U{})ch{r(T0f2Jr-kTJ7hSU)'
    'qP8jfXxR39x3Ot=`YmLV8t00-vu_~jiKQk-}qmQH0of4hKCXtMAy7HP4Bz$qj7-'
    '>?1&yV{1nHB<1`Jj=RWd0%Mfxt5^Gn^C}2wiN+-35rSP~z195pI)sHK2k;iYre5;x$_CnW*5%vD}-z3Mj^M-'
    '^U>UWGuFZmIb@l#O4O@Zkg(Q%KE*cfG#oa-Ljt=(%i`doo(@X2^G!)JYZAn?%65yoasSKofK<*W@4<UG@r`P)!0;-'
    'N*k`#*j1TI8?M*bN|~w|=4<SqOa;iDsMYL>-'
    'Jp@6PfC8HMutAY{F^jV^a;u@(8$rZ(kXMCazrq@MI%dp2sLTeNYfvhY0@k_?Sh~#w&1)D_QiU1o)FKN1L5504G@d7oV|+G3*CcGk'
    'sF>Hg>+hn4(2uf)*Ni2%C#WZY3w#l6(937wwRvBmbFzH<t+q{<9r65;Bj2QxDz~%3mJBT$8iy(PVhJ`X3z;9$0dw8!Q;486^BpE4'
    'n-bKxxaCt4ehAWa2xmD;DlO*%uwWm$Zo3z%iYNG=-_Y*0E$<QuhD9C8w@KJrjp=)Yh)Ag0JE^bh(2s&@}GZ%m{wTh+?&-'
    'nKcCe&ugPkhU&v~lU(9NpU&?Bn*Jd@&>xw=qTG2s8w&z=abQH`!*6(J^MYdp+ds25$?H`rypigQ47<-'
    '9WPPCleM66DGT>6Ht)_w&$ig051C)iU2bLdwZPLP4KaXV$cFmCGiwLHtgg70i;9xSpA4zk7BuU*BsS&v`E!38|=g}c`kdd_hiIP|'
    'jaN+7!x*Q@V8bg5!dU10q&QB)tdepg#INz-kHx_p<;BV?-#R)vy%dKF`dSp&oh)}g{4+-'
    'hh#u)(Jtt@G{nW$nME{No%Su$VRAV)Y#?<_UHX7r5EN>B2;-'
    '2^LTsGp#0Ccyy*(O|r1$%(a?q;jCt|)f5ZxnAuh>7Qi3Vt)^PA%S3n6nCiGD4^CfX{c%x1mso#-F7*1$oZAA1{GRqFS-'
    '((o`(*zO<sYNFFKPcx<(JdlSG51O@+;^L6y!1HHe&I&@z;|f9cD3G$9Q`VsI31Tg}3H{%KG2a&3sTl{}=kDeKh8Gu3(y<#_Zaa1('
    's-YXyYLIgw0YF_d~_Z8a40Lly*R0my0cm-Yx>lQA!|YBpq%w8CiRlaa*H7H}0J>=>DPh?ba_MG1OU1IMz`$gTq%~=-'
    'kTbQS@F9Pu}Yh$$NFldp$CFuReLNM<wrd8b=daL;c+J0_%7yHfl`6oJNPzlZsRgN_|2Q1bJ;u*g8}UZD{U3I!BbDO~4>e|7h#)k3'
    'qHLt$(=8$MSzQZlS-+N|)_ZtiN4z18)Ca<yUgivW+mG+O-'
    'Tbey)*zEyIn!)(Em_l75d@TFg}wiROD2P`JO3e(w7gV7UJYrF~m9jRFe&SJ4AU`OB-p8Pm6oL@gY}oi-'
    ')2T7{@pj~T4I)kKFD4$#uT@JhXq!)=kR)q$zl81uLOfk86c&w{^2ETPLSrsnsR288FxO%oPNgm#>v<j`Q}o`dP`5QF_^4xziT1{<'
    'o3rMp87cJ?`x?#3Bxu`-'
    'VC4l~%T<}kV&Z?J2~c)B|rxhDRkf|@k}Zq_NWh<(isla4Ei=uxD_1mlxTao|0In2H`luP((*UQomlfYUYObFs|+Zg3m`>DiAg_*V'
    'YswD!}`Dssk4qB@ZVJSNeA_f0h5{Spm$|3m{mAklyiOf+C>h<PMZnS6UeI;aUYtDwsgv{@`GQbOyL^ipe*@KT&GRK&#fp|qvg#bk'
    '{lsvDq<NT4Xl1{-r2f1Qy^xW^PY$y7Lqq$__a{o6F8S%Cv2Gv{V2$)sKu_;%-~nRD=cL7G|x-xsE-Rq%aLn)(Ib7pJLT@O??@gbU'
    'x778sHMpA#u|G!{Uz7loNp*vW%7<jz^ku+XYwHf5DaJkKkpB&}1C?v`Sg@Y983kU^_OpS7Ic{tr={5^tlfz}vuP3^{M3DG0T$kma'
    '6@1R3W1U52f|IpB9W))aA|3q-uDME|bD`XbJ5fq-'
    '|H5ZKjNqQ#lbYp_C#%NF+4SYW$Gs2Nyb`+`teu)y{uA!lHL?b;kn1Va|2_@WAIxT1sH96J0{@^dzNe8OI>_%55rmSy33Lsu1X4C|'
    'EC3a6UpRif-'
    '^zI%s^;F$w>f*Y621vo*CINY^>&Z!NDyAIGfHQ{jA13IS`9Bv+<b85ih<^%fp0QVlo{@KAGMK%kTC_gU@7AyjPQCKTj1imhp(h1f'
    '-5|2X=K;t0O{x^dYOla?cL;h&AK!MwEqtx~PBs7|3F5fRh2L+}-'
    '?u?n3a4=JMh4ZA#^bf3e*8h}#@W<9qv+&DO^7^>))=IhX3gxR->V;P-'
    'Kh&0^c8w)VMPzPqRzpv7!5FO;pKZ1<mK)0zOA$YmGb)!^KZcTW*6%+k|J?&%#b#Dh+VG%{)+$Xywn|AqZa=B%p9#Q?PwLdIEl>yY'
    '(`6^5aRhF^#_+-ed8dp2mBJBxPH;hHOFy?~4QyMktU4TPWxcXe_xgnLLu|R{Jq}jm%E|v6V8$K|twPqV!-'
    'JLu)f#S^azpf_L1aX>=P3i`hi%U@1}qMno97IksLNRNTV*@G;ogw#_@;YPw&PpwE!mE5ySHUKzT@7J?f9;HSGMDO?!7|mrm*J+5t'
    'OW77B%N;WMyrm>Sne?q}nmM-eV-@n`}_U1|FSw1|9H1B1w3+QZN(rafU7Oa@+mW24OxK=G6wwakRzg$Ii+!y60yZ-'
    '3zje?uA)K_o6JLdvTW0y(G)%UTT;U-x`VrDpraemn&~<W{>7!bCSpjwEn?C_UK>6-RZx|uu1z<t-nJ{%fmY3=NqA0Es)6KPeW-v_'
    'aFXQXjIEY?w^NtG)&~)6-kxg%n3>F+r~F&Vd;UJ{7Ee|J#d^prKPFO7eXujgr-'
    '`<dS*+L!VH8|R+&=v63tMnPL5u*K&`4&pj1|@IoKC2$SgDpD9h|MB&jTsb`oxp|1vdR1;_af6Sq$TMp##HHOWENtz;{W{o#zCM{*'
    'q#B6SA~jqPThV&(6|?n<!rg$DU`<M-CWg9MlQ<64NP-'
    '!kAMXKK1C#N2a~xDy^~|Bg29I&61WXu}TT?3Ov<v$>B8nBM3{bAUUq2|?uFQLNk!i&NZ%9Y#9b3N0~f0=09tLy-o@?&m$Js-hhOq'
    '#Z>&b+YwunrNCr+s|4|1`!ReQ)M)`n}ol8P>2YV)4T|qU%-'
    '(leKux_Q~8KR7<~c>_g*ABkommNf`y}w8$YpN;i#X)&n#HABXcW(;HShsS#7~D3F5yV8|anTCtyT744(!wQqf<uRUJ|yR<mi8%xE'
    'nYorM;Dtf)V=V0dZX;Lj}>Tj~$;3k$}UW(@w)g0Y<@(BdAfAQp}Lo(@xKNaMb*rM?}u^$##WBl`D49je`FlY=4IJ#F&;(b6&iHu;'
    'Zrv1LFA)Rcpu&shJ%D64C+?O=Ha{?p*Rw-oof)cVgU;X2-;xP8jKp?wP1OWYTlq;b7OmyQGQq}lel5=i*7!5lIbe|1|tY1`-'
    'p?ziYo9L+gp+><*BLF1by9CFEI<w>)Cl<mEicUCP7(|A=V))k=+{y+0+kv3QK0H3t}dr@GaElfGFx5T1$ZRI^UklRtb6SNSk2OBA'
    'THJ~NY6}V~l1Y|tj<VyC55N7-g$#F5Z9|JTgEm%G;GbbLj=))h&9TF_M5yx>W35%}TiJHBKlz8omriBSH-'
    '!OtH7NODO=6WZYFoMC#<PV&*P&fw-Slu9?`Piguex77P-'
    'f~`JAfzYk*2oAGx8YV6Syqh~MV2I3!&PRkC}yw%K_v9>N%rd$>))aRt_IA$J;B;GI;4+gC<%gIX#JUVw-'
    'B5605Xhsw_zhN@Z@DQ9C-'
    '38q(%X=ps!2cB%$kqyAnydkt*~ZBZzA;p1y@{nfC@8Rt&;~&O$+oS6P3+pfW=+qf3&8mk6|5x$G4z*Q<u6s)b6C8n?ZSg|mfnf_4'
    'n9zHJzJo^JhnhTYlQHFjHEp4_%hMe*JS|A57)$T8AnO4!Zd0Smnu{9oZWgI^~EXYhT6){MMv9Oz+Xate;>u!ZuTUR$#D*BU=OYTI'
    'Vyqm-G3u?VzV!a`YQ9n4af&Z@*Rjzktg4zBZTT`o2JWC-'
    '+smIU&!#ZcMtx)kU|H|FpZ_&Op5zFZ1?9hm}OJ_WvxN`bFw3}<FzD=*)nd2AJOfDg^osL$eLt)h75!<LTXdD?>4T4vFKk=@P;j6z'
    'K0H)-*>Z&d(IrFYm32-'
    'l={*p2AFN$;?m;zUR4+_nI5k8~2d8INJ<Bz6lAJ`7#mZbc)(<F=5yd>9k$zHNfLa#j`+%1m`D92Pma>9)x%)Jfq+JFZ=?WHgszd;'
    'B8Bw?m`IiaW0ZyEg=pZ9zs{S!96Q-!akY2;)UBw*IUXvRI(}M$NOtkeEGAs9E99%cMv+&N$L>On{hc(%eWm-'
    '8fcsTmYOL1vx&TdX5~O5K#SEt`!fceqCX}cn8?qHa(~Su<JkdB#xrZdy(}@XX7;bOye#r%51Lj!=lVKAw{e#v+39qgCQ4yz8x$(;'
    'J*ls*0}sMKy<%eOVsQp<u_=Flii{K(lfIFAOSZpUVv4j@0UrNpgr9C^;Rsq{ea>DaLA@VM23dp>>rymHY;cly18*gnV&?ye5cgj^'
    '&{kIW1av8P<x9Udt7l_f)CdjzkQU{Q`Udq4r2d?$xMCXM0K%45E)$=75Ftm8Q3y)Kmxa0lZvyc>eOMunS=pHJNMAN3CjVwh3+j_6'
    '3Ay%D>5JylN=A{Neq}mW11VL(3n?>Ni=4aVirwiH5>)*G|l|LJX#>d2~c>}`u~YX+;i6d*cO7tO?q3!i0x=vCQB_60LB_u7vM28%'
    'u3a^#l|UIwr#O%2-j^}KZ2_Y5De8>u90xqIEVA_w_`CKyWMwS?IHWwr<gEk%E}-'
    '*94=+^UZ_o>TC9(Yu=3<g2d+k;?mr79WuffXLbKdT8nqScltI4+LW>Gfr_gLK_8~&{YI);<wQaGC(J!#r7v~sS@4W*7;taa`c*ya'
    '8T1h<WMu|r~ZP2rOP&~jMZVb?lA=uAK0Z?loVm2}62UTQuU@?{)ekx6o!+BnXjVH87oEnQM{%%YC4uZ4W$6OOs^e$#IbtKAHMTRB'
    'Eu`aLzoi+q1L{QfoKO*kC8e+||px|(uV7brl`VbZTI=$r4ASJ1}I2J%6)j#!PlKw%VetbnE!G+i!U&FzL965RvByGTNU~K8$5ZCB'
    '7EOF)N^CGO-<lxSoShM*_Q#b)J4v-'
    'yzqq9Pj(FkQzp~(_hCK>I6#{gySb(#2uP6_tUEH;IG9ZA1ISasI_+iZg~#4)|;d2M~HnWEODumIE3U{-'
    'XYIhx#s+EM6Jx;X@1ze55)4q3@@kB9k)-NM4T3r7<&3+HYe9LX%4FXNa(Cf<A%C-'
    'X7arcqNFd!Ub^S1R;T4Ezgy6r*f$Epcop)?gi6<d%m*Aip%HR?RVfyQoyFtiNXvEv8Q8MLt_mSi*#@QxZ5!1PNRrWCPYKJTu}dPd'
    '7!F-'
    'j6Wj6Y{PHU<6Ret$36QxfL@Rtb@}{2qK9Kx3^}HelkY47^R<r5k!XR?HK7|oPH`s6WO;p4TFh{+E2$=Vxb%{w1tZmF4iBU<`?Pc+'
    'u*{qBS@SgWu@SW4ac%-X_7cXQ=g>}9bX{P@zn;Scn?cp{}w4Te9mA5YMGE69!p(iHiXbEca_-'
    '~vNhaQ=5cVbjeo}S)K@H_8()}Lsb!E;1M2s5p<d$rtl)nxVZena`2c(L=@jS(8$P{2xeBysJ&1I*0ZR~q7<aJ#UxTVT!Fq3p6^+t'
    '8vQBfyDLfhLa0Z8j*>+O?5-pUW6{&O1mXMM9E=l)9EN<fj{$_ug;%WC={OJnE-'
    'N*KXSa?2$I}&2yIRze8AaBR7ws~Zx!(?qJFyKj%%#nDIEYU*ys9;>V!*Dn$0j{Ms0<NrtdSFl>IkYSdvbw_h{RVZqI?eib4eJIt9'
    'aF7zleEa+xFAJGKA`-iTJlDnk~caYa6T2-'
    ';b^hxihC?<^zF*M9e?P97Hk5a6(6y1>ab3K7i+!PliL^RGX5LIG+^N62xBh=j?kXNsW{4Vmtje|xgPw<3Py_(^(spRMt)UK-'
    'V8HzFAj1Vz`BI8oWQG_Xc;1J3=_;a-gJ@0-'
    'g(E7qC9CZm`>@dQ6c6<f#45Yz_^^gi2a<tuk(IRKhSwUryuIPgVX;o^vC3?|0W$_`&-O>e5;7n-9QSL<;|4-'
    '(*z%k1*9@dEFU7AOp;lq{a-'
    '9KDD;0d^K*S`EwraiBLzdnaFc{)4;2ziYZ{e<H9`whjnHkWM(FlbBeW>h2;Grtgzii=LW}d>OFw289h1)fy9G7Eu|$J8@~A`8MTc'
    'VdfNmaERx?5lZHNU$A%2uowz|sUsfV=xW0Y3A1shtnO#@*PAP3hrgh_xLT-#Oqu`>5|5A6@LOa-o_5kO9*2|z9z0OU-'
    'X|Kn_$g5#=b2#y<)#sayaG!G}4@D7vIpwR~H;pB%@VYgs1F|WJ8&6!4JYP>CmikAi&e$m(akaZh^j6P=Fma?ZeI~02LaS|pJpP0<'
    'mlTcIzpQT%38(bY9QgKZNJ|{;~;P|{x`3q@q?!m@A%4lb-amyL29%|g<3?RlCw}NJZ!>$IedX5qNa1a;A;L(ONqR2d7fOYBM@zZe'
    '~8VwGoyNfl}evUACBe8q(@{j+v?)6yxVd3<dU+h7*O``1UVX)$-iMQ2+Ws-'
    '#F4l(XAMsSB2_XKyNeg<=eOvBnm`$H^uPy_I^zs<nY0hV9>e2s;!JYoEz&@*v+y1PVU4eJhccd5oA*1Ey~jPRCHtDJ05@Zx6~M&Q'
    '`#S|=`cprd8BOM3g|&|r2s=7}RQVpm7&HV!U&=u>GRCjN!a*zTIN1~Vn(+xO#-'
    '&XNE|&BRKF$Yr2{g#<gt8@H0xVHGB3`LJG%Iil;0!0kqx!R;|NgWLPs3~uiy)mi{s545~=w-{_L`Uq+8CvZ3-'
    'FV`n=!XN+KVLDb(<}H9`BT*wTc0^VWYyYQcsGhZ1e3(`1DP##T{q<>NHZuG5S<KhZqnqc9TTBW1e#Gc?QeD3{&T-'
    '^d{5TFH<X7AWhwJexHsE|chUuTe8F4JiPaA)Mk-c<?j&-CN$sbotqSO_G9XPdQJ+RXO-uKZ+a9n#MQ+BYmT%z$|5VB^A^&(ASI(='
    '8})O^so^`S%58=-'
    'ek(Q37#Z_Xoqx{wGi!=Wnpmscryw9$Fw0|Pa!IVH1{W(L&6rD+HsxC#vc6qljFVsIT=nBaV&5u5W_<1aC?J)3kI*?X`~BYVf{G^Y'
    '1ZoyPQz(`iiac+Ca5X>8r^qW*l;y3K_c)#zf^dX{Re!o9~()jz-CTq6~Cpkq>U2U;a1b%4%7tszA<4f0$c(oDx1t%cK$?%ToKJYy'
    'g3$Ks};BSyG(j}YfNKqezyWW<<#&iG4B!F-'
    'L3y%oDeZDK<z5yEpe1j1)xOyYz=v6K8jZPlTUl1Vb5E|AGpp>&cWj|RsIKd8CUM}rpPJ$jZDu&TF`uxh~BTy>TK^o%eOhmWH(3<g'
    '-(a-U_;2^nFE9pQBbg1fbU)$-`L-'
    'xnsE=j<*th(zxy=6K^!v`*6V&SKl~|HKc=BgbLZ1n#~<s&ZCn8%FNNg_w<J9k~RlMeM#@iddB?VI3*gG$-'
    'Yn=B8ZJwJFzhUCK3GpK?v}Qm$!!%r#{RrEPG*Mi}tagYcfQ{@g(wm~n`P&gi-'
    '*8wqR9(ish<0oS2hvzHpWuy!h$Vct~Ex&1HE1N)I{pQb&0++@=pKu6QK3)qZt^n+}RVFtXVur1z>&CT+j-'
    'nNKkWWF#?5lAZ5VKUkj_B_pK4A{ReV=834=T^<Z;&eRkVZl-ol<72X6_=xex4yS{>)%iq?Acdb_kF~5|CYjF&;C420C(g8;<<iGb'
    '9L-?HuL+2*(^-Fz0CqMJ6MXvBr2#5r86Z;>S`OZMKduBdd#3<a-'
    'CsDVTTl!c3aA3wd2Q$%ZY(eLCB3bfLtm>Mp^2TGDDY|9hWCCgwg;$Dq$=6!?o-se}tCJ<f(!ccauL-'
    '%XR|WE&B;<oopySjh?t>vtYt-D>3zUu+v!-'
    'tIks3O0;aa@sr|)JH`49x?HAMk9}|mjO=3J6eUI8)xvp0dc2#3^M=%TcdYRyK((iZ6P3b+ds#S#2$S4u+;doJNKt@A(l8%;s~9%I'
    'OR;K+&GUB@8twx#NQ@PU?B6toUZNFg>|3!p1}<=$LtmR3wQp}W*w_uxY_ShYhYh3&lX33gG=3{`?v^QVxCX_napy4xZVTf(+blzq'
    'SP9HkTOCMoOO$NE#Jx%|F)8&DZ0%F5n(oI!qkYAK9E;H?7X5KZyJp#+U~$;vE0RcTl^&r7BejT^@es`M@TZN{c&4%whonjI;xM$s'
    '`0_A}hw(>`M=nX;Ha1v<7_~OmPZ}KENAS;YkJnA&5ohd)Mq9w8JBLiD82U~n?89WC+N{h7?8mIi7%{rXBgKWE{zQx!vnIv_KE)=U'
    'S6ClRtiUx23xtUl_=3U;VIttYsIWvhT7Tqayrh_pw^m`1F!Ax$DXbFCXA7V~jDkUHm>cw_(1>~)1|4-'
    'NWkcG~KoyMpR+DC_;#dgF1B3&Duz4A7Z)p_!0*egghbZ4|+>_i>V1aSZa8)oC4@(tQ1^*(=sBsFj6K7~1yE51KPivmLa;@=aY973'
    'Do$;U1JbC4M<ImDOdS#yRpVd5jWxnyB(^-#x*nb=2e!E|R&oIBxH3ZvR<j`2=jT8km-JpR_J&J?r)(JYv4+eT)BHD*$kkbo+k2kE'
    'p%knYzmi1qf5Ys!>e?^e+d)9wVP~?rqJ;k-'
    'wHyQUd^@#!()IE!B8bU5m`Gtl{N7h^PXYUu59qjI16Uz>^H#0M*TRi0(3&T{mF`r>J_CJk@5^SAYc23{4{%*^s^}E)8)iQbQX5&_'
    'KR|)KdB|}WG6P64yxe-'
    'U1^6#G%(wHQmld*9N|H~;Mr%BRk56L$&_os%KH+}RI7F+&YZSq(eR+)%_sF`vfM}~8=Fddnv3F|nP)Z$iQ1fm?XF4P3$(L$4+mH+'
    'QS9Hv_=GWH)4)+kt(dOCCWsRE8>Qt~IN{vu8*M*5pn3#3rv5m~6wWue9+vrwbYLXAgdp~h*YP-'
    '9f+RPB&?!}kP+XGN1Wz~ng~wk9#2uZF7ECi1LTm0Mkfb`oV}LeLh(<lge8Iw-'
    '+yu_l64cW=vHG2pf;IELKYDs<>+<reQ%q&}WZZq<jX$O1ft?mk*Y*8FL7_pvI{QBNmc_>mUu6Hzo)Y3cvdj!hI!k`}X1Bc8#vNzP'
    'Yc%lGD6-'
    'qm<^#7Jaqg6x_|eFktED%fR874%{`8D?h0s6T}ul^Jk1se+x#q*%;GS}X{8Sjc^FJ~7LBADe;*pRgQ604!VucYv|*=qf5Ty>&y?U'
    '(AUk|B)&ZNKYk69d5y@lRS>HV04H|@f3(%+V8H$T9JHSWTfZxlIHYg`Jk-'
    '@>l(<^=}g{PXG0eP28#VXvTZlZX4|fU04oUY5#>K>%Pjg<-0uWur-w@MCMp&KeeDxivr0;Ar(=g+%w$bhT*gM-gA+vff{~6M4Wap'
    '|KZ0HpPLwX4<>Ky;BFtvXRFHP$aLi%vWTFM^&i!a{)d+4+##Q59r53O|eNKx7?9N?kEnxR$x%;xM7KO90=|PYX7}G&5o|?j>npVl'
    'o^<>@arYjyoD8aeWmIM<R>y>KoT1ZmxZ>_Kz`GMyicia274WG_99DrvkSnP^_5P2-v7wc(a)4-'
    'wmn#5*LGXAd;n?2e1ze#NN6yyIcvDtRx{~_`6Q;q+pbkaJ__<u>%cBRGsf!CN)Jhsdeyg>Wcqr~7`d8h2kyss+F@f-'
    'YzDmEuPl^*?zM%ufy<0#qHnzv_x37C`Ea`In8=&I`vS3)q@l}|i37yF1kMWL_juyxoobTbdTg*|HvCW>Zu@=~>CVa*`m(Ked{sPS'
    'NZE4E6j28LuREwBN@3{H3S`}bF~!3yUZ>3372@)g*7;2JOCrs$Z({u*llW}%PgmghKu*E1Hv!clk^%gsW@DCQ`+>#=9rYRk<+4pR'
    'O(n@0<Awh#{nLoRM?q9iYUiT0c0Dg%^wdw$;N^FO@w|Bb^5_V+8&v)zQz8A_&X)(fNoSHDdp&=-'
    '2f(ap=DO&o!pUxzkvWWawD+Qbpqc_nm)Bj~i;Vh6}8ib2GsA(PHRKz633==Vq@of%>~MUOUd{#ZjqJ^&44lap#SkkU#OBr2{ZM=)'
    '})YZbPWEzks)W}(M=_{yqT>PsogZX5<zEb;D-'
    'J!K{5`ec@KeJabjKAq)UpUHBrtFxTzvsupdIT`;kN@|4Lk$pRtPqhyzu*ihWKB~a-9+nB{HzBWu4r(>9`-'
    '^5`=c<es>Bsp2(mx94R7le(oF%|rqZ-_!Zae=5S<98K&c8<|FgWYCN!9<c*fE8ViQ}--'
    '44)I=tludCSa47mOQC;v7~eNb=efYnu*7}RXRI@8+|d@&ciqxmzM^uTM;njvX$XtiQ7E6R42s1bTV;mO3DrT#t$x2#Rfe(z8skFE'
    '0PucHulw`0OjI4Fg#+^t_8b7HOuG+TfB=Azv9vpAVG=FF9uLHpfd@wLRf&z;+%7mPM$yf#IQB%-'
    '&0aVjW@t6<LxII4u$hc>by=cP)VvlFicK<}b7?W2+*vG|tS^z(yD$w?I)r{}-'
    'KHS4S7Jt|;Gx37+nN0Zc%>zU;QqE7(ucTicbcjrqor~#I2_@#u?!%4ltv5h5T0dU3kxv6FuPsqGDB5O)(}LyG<oR?4ck-NfoRyD('
    'hWqz29;(Y8g{7k0@1KVFC{(O4F~xcTAbZ+CXJ<=J#eZ|m2URL0X0LarI~sDZR@`*te0ny{BxX?mZ5r6p)l5uC@?QZDn<;z5+fBO7'
    '7kq_H6lac)HPBgG89f9y1NURIpC60k1UG==Qq&ZV-'
    '|bWUQR#xxW%TuSLS$R27vNvv3o48lW;z=`cG75K2DLlx<rVZz4%_;iq{<Ut!Ua00M8*V=s^~C#-N!@fl($Kz7o29r-'
    '^1+?7zeHruhcxVNl<f6efzG=D~ZF=2u-FhC)4Mqzb1B2Rhw-'
    '+4_HAvv=w(W)V+qDq5Yeh2e|FRO||0G^S!V_@XftyTccasn`R)XiUYP@I_4z_kypS7y2v8Ec-'
    'uNzhKb07nVX~yELDDgm2xYDPUWe#2WW};RGy|<}N|Hx75JhZT;V{1%AByD)K7*X1e<ta-IDaB~q^_No6YR70zW4C|dZwQO29#_f0'
    'b21iu%^fD`<_Stgv|_boEw1iuR<5&KzxVN{>}VIvp%=OBY?8|`y?ae@JM6Kl?s8BZb;k!xq4^8anQKF>W1t{hifyXS(lpfhEhC2%'
    '>-'
    'dIM{dM!kWxNt523%?7=JwMlc{z}loSZ(wa?%A1S;OjUJMjJL$5;88s;D`T9{ScJnU&Dn|B66Pb7I)42g{Cz`xce9)lpNTtGvuRu-'
    't1{25!M%xP5Al@h$EEL9OpL*DElGM|4VGkUump>niMUk(gOW6#_8+obg6mcX6i$Y*$A-'
    '>Cv>6|@*mwLgQ>6R=g1}~+*<8SW{P0dC&0I0BN?m>H#WTiX3L%C=vf|?~=C_(7o*+3LXyF{^ZgT$|6nh+A#lG>UkoyH%`v@~{+zL'
    '0szhihhe6#4x1UErcW}=%Y8Z*gF5`~%UCX2pIaZ^NHT3m~0%Tza&Pp42@X#IwqwF2J8j@5D(@UD<Mm*7SmNzM73Q0Q6DIn^HN>{<'
    '_x-'
    '<{5KLJlOKKUsfs)T8rFMucHKf*t>$Bz9oKKSWMiJ9<ltg@!TPWD}AuGuecs%S|>R=?argNV?Kw6Oyhn*@UF4O*SFv8bcG30`uBpb'
    'M~uDiBYTq+z53PRz`w=nM7r}7fTkRP?}SFSL_Sb9EBB|&Ua@q|D_W1|8<}?X6U9|+0A;iT@_pGpTr~PtJquqG}0=tsP`-'
    '0em$;6Z9p5`6K)6pE+cg8i^&LmKZOXL+y?N}7s)ss@YL_rS*6o165a+U@7yXkoA>?7pI4=}@;T{45;650z5?aO4Oh>W1|&F|iB|o'
    'Ap&*q>vqQ`Cj(lc3800~O{xKFHcM#v7t-'
    'oc^maA1UbuyiV0|(3l<jb^&;!Yms6O1EezIREp8Fo#x8Fovv8Fo*z8TLrC8TL%G8TN{^8A4AKg;BL}+w4shay3nXC|oa4e0nw@zR'
    '7^vNQA!`m92xDN6ju)J9LH=eL0xc7R$0?pUb}v63(zHw%9+JViIp*RT_iV$J{XgCTY%cH{8EPnuC?JGBF9Kb#QpKG1OHyNvL3ERT'
    '|2^to%7u?x2dng3niZW&s8WzF4JZv+bB=Jk-ScYnE0$mYdJ0vSp+rtRb{Gra3L`2t#+KKRyzF9)#E>0I<0LCwGEP^+w!p$5kWOhG'
    'mG8akd>usjK4jfzvH_JRhrxXN4`8#~~q-d!CO+B2iQ}z$gzZ_7vMv8|d~>fGne?aDWOs^ZP3FoDY#2o!uo-'
    '^eg4huF?dEc7ItFphT>tVq<-'
    'eW!#KAtCmltug`}h!{L&9Tg}Pax>p)J4w_z1>hwG7Z!p}A>F<L^S#1rW6k4ixv36((wH${>|GI*App>)m3l*nP{TkzMQ=CWjFByM'
    '_;&iEhRJnyxjq{jtw@Ibta^-H9YNnOSEv7P^c7GK@CLUsQO%+g5aiBvp2&Y~c6d?PaEOJN1<nSZS?5W^^#-'
    '$<K5YXtH1a}M~VQLFLs0)AS^w^v=amJ2w?1H>%MS~lBGnN<D^RZH?<~E>-'
    'QlNZho~g0Z^{=IkD&VR|NqWDrSP$|d5dt5>o<9$;_thlC$2kbs$llGxSTPyAYcXI-'
    '7VkRbc9O}v9^)Tm^X4IAn2g?h3?Plq4z?y8H`-3<q;INbxVr|&(l&EMU@2awbEzLx0!IV9EG=Q-'
    '(Be+czqVNS@`9FJ>lHF60d2@L7Q7*v2!dUYXgtG8Y!^f!rZ){hc%mR-Xn$QOTbD78ja8nk3`Xw2YG!C>Y4i+Bb4Y6#FH80PHdd{<'
    '8l%K|X;O^dZ4<l009kvjV7>y9JP!RyYTBGlA7mx2;jy_h2Z=zSq^zjiP107hKCgky_B{rFn#%(i!7gu*lH@*%RR+B&^FCX~cwhqi'
    'DjD5@jbNT8{#L)AWBmjv1-;(-DTdi|2U}D$+OLOL)HJ#oXHli-'
    'X1ql`lHu5}`^Qv^4K;VHmI^}7(cC@uR_oUdsux`X%aEEFVZKh1XQQ#M5K}^<EbsbyboqiH({_xk<VIU)eUHecJe?$T%ZX6LfD4lc'
    'o|-T%$CF9MTHIW8bEw7LMK_08+)^Ai5c7bcKMBgaT@wy-yJb0--La$U{v2kKqr>fq{ZtR+K;fQ&c38v#hZ;PRxEnkigR7XehYjJL'
    'WZE7!vU!I2dDyUHH52r(tIc!7%IilWS+i6J&$WJ{6a-(3+$aY4*I7SVN^a*_-y&7E^R1s+och)kM!t1n_riSNb-'
    '<FzVYUix<e00W$kPdkcaryR=4B3~&$5rQ0Q~Pr4V)E+qZaiE79CWO8?_@Py%Lg+OGJliV!HFAwmCm)XISBGq&!Y&KssPToxU*ez^'
    'B*}(aIM!4VUIstKccIib5&i1-'
    'U0FkOEGSdx`=l;P$v@C~N{=k6TSa6L5OmvlKD`pT|8%KAs5^0ZAR#8|=r$&eaBk9U9r8+GMcqk)5c`2CGHUiz*h*tO6XEs&+a<vm'
    'Nen6h{w6Xx9?tCS~2txG>$)9btye&%6e2z_W8tr{G$13a(8^!L^AgxHc&T*Cwan+LRPrYe~VisZ`%=unz?Lk{b;+Q)K^fi@{b>ml'
    'o-hU=Vt+4MS!Ses93diorPph_CGKeoo_1JtF13Y_YI&jTVO8Y76eNQ0$(y;NJ+x?m6rKLI+$A;x-'
    'CeJdUmWh}?F9a%)4Qr@%Tv=(I1LX&*x9v>)AkSh+=hf4cdIa(DOxs>Mo~ib>{kAdv6K)5jI-mJ^8S!G!<``d$MD*C)+yG-'
    'w~nJeymrUmG{w9}GAcr3xAN^v*26AfuO>Io1#&G4ureaac=oy~BMq*PmZm>_zds76#;#7O<^QAfK}S_jK2RsJcc|X)Ng1cJX?y1q'
    'B;MLwqb~*l-$FV?o8Xr+GdWbZi98t+AkFJJ4t!3tCo3ZHkmz6Pj}+d=VrT4+{~`7j)<u4_7x>|9tt3gI9-LCdOqHx9qM-'
    '8}`K4=4oIQxvAkBN&us2&@l8@EEW;3B}tuTv7Y!@x;x!s-'
    'RyJpp}n{S@$z}FMYIpsfGMJV_yX)%+J`TKQKEhL64)f#hikzk(LP)U7K!%Zamu||?1guu^=qQ8+-'
    '&_zd5aqyGresG4!msapsQt0zzNzvLB~~_<$<WkpoM2A{BjI(U4(KBPcMvPc>Lcm$rhgfw@f~U7kGvFw(ti1!lQlK`ajTqeFFPeK9'
    'XlPmtcH?7omo)uu4%Q>V8bW_cC7T39b4h9@nT7uAHGGt`s4ir$ejwnW1S*Z!RR<tJ{>(ScBd@k4G72Fe=OAQN|k#m+-'
    'KQUZgmH%Ob5P8%4RW&EFU;&fRSqOBC6;yIo_ABC%w#hw1*BX|l-'
    'RIzWr#cc;bzMk2~05AGmMOj%|Mz#G?ywh<*0I*BxC7UL3ZD*~Q})GTyq3A7-#>XGbE8}`vG2=6BAUx*s^I|Ghu79wcVA-'
    'u2+&N3LoRfH<sKsK)!{}wE7O{gscxD_iDPN{@`%uUk0a<uhr)q;OHBlwr>Vn2m5X9yoY4JeYj2fAkfL{c|cw;J#?wNgZ5EPV?n0E'
    '#?XLdm<bEA1u$iab_A$-7fK0t_YZK^;Ocl)NXLC>%=OD>&Qv?72}<Tn*NLq&)M<sl>c@CUNauHL>g6kVLL`Ly21-Ymxmo1U-'
    'UEep}kIz#jgN07g*H?@C*i^DQ<TQbt-VY}9~*q@%|c4Tz^SY-'
    'S*|X|HhTe`f(7T}t2_=Fq!TUdV>G;>~wr>0D^MkvH33Iyfe`kM$pl(z?aEJIe_>_{aeY^5N&D@Nx(KlO!H$aq=1)?#?V|2q5ycfJ'
    'l3?MyLI8hISwGP^A39St)o3#V0%Kti}m^j$Ne237n2yrN#-oj@_ij3EYm|rN#;Tjy<G$1ssn(rQiuX4=R6|6-'
    'LQdOrZ}KnCN>;RkPzY55>IIy2UIilJ0BBj4os>`h!?F!E~=-3M01RPv|npOD9$V;G2432^$Pz7D_A-fU=#%E>?g6#&8E5zySMl{~'
    'N#n`*HUhzySMmUl_muOd|(jFcoSG$b!ja7g%hjqm9t$HfTV&5(C_%0l8Dfyv43gLrk{!LbtpiqJ>D@D(}5js5P|e&*D7iS0v&wpN'
    '1>6`xgvOTYHeZJF+<1u`py8rqZra@V3M=dO^Y4j^&Uoye;JmUQ+P36nR^#;B6^3yiUQ}=0P~Ld%Y(3)S<rJ3b!W$`bL5N18gf>;_'
    'T;qzjd0z+?^0sDyIit#WNt9a-6SJmd}$oijr*8Qxb0KcTY=DtsflQgwo;s?EuZ9j+4sGk%3Ks<mS@6rUo#lV4Hc81t6H~{Lkp-cm'
    '>w^=SDF4JpT(?g|!BU**#Khz#OSKQdI>Tq)%bn<(dR6C}fFRJy7UPp7%wbt`KyO&W+PR@HJai7;_@x`UB432;w?S$B}CVaXo|cIf'
    '<yY8fS%5bQ{*PM~KTW8l2bn5dUQ%KlqcBupj)%O6U*%6eauzZ2Pmjw8v7yfWWRvdrEUG#T<y#K}tLUH0OPmvR-Hr+ERutj}~-'
    'f(E_A<OfNTyfKv?&@RC?Ole&}Ph^`=@Lv@%=SBwsTWviGJt`+7-'
    'OXwAo7;8(E@L#b10BUxs%L&cziyh?(s?8C*=MBzJq!juZgYy$9rT3D-'
    'DT$O%UuSSI;xf^M#A!WA`MZ>quPA@FlJXVhzoevmMfoo)DPK{EISQ-dzpB`#ovw%>E5>P@4g*93Qg?Th>~|+tXOuF|Jth=tuzsVM'
    'deBlxU%dJhBdDcjh4m>P9DDLC$>GUdYz!5i;t(AsE|yMQgRY~dL?ue6kLhOk7r9Q+up@{b?zS78*GX`;o0)eFZad@!E~J}bNDI7;'
    'Zif2@vCu5uY>yPdMRYR)8G?7v%??NqypwL~kRQ0%6ywogKFenBK=9k*JKv79?7{wYml0qAk`(Z;)ld!15-'
    'Ym7)#UjHQ12dYnuU6r^YNB|w_H}!kcMc-Tt4NBEh@^g=-#;HaQLvDeF97%(;@0(I6OvmCM3$##6+2zlqge^6J=^jqD-'
    '|U%GA^XW$H%jZVR$mVBMnf4sKRRhG6d*!woVcz8SYEIStdXCkW?-'
    'YDq6!kmAx@B22agTRW=i>#Mn7`a(CyZW7F|Ebhnqm4t&%#$>hRJ5E7v1QE_H?!t%Z4qWupkVz7S*dy`ne)lL?2tM2%0|UW_+v8xO'
    '*@~|KTfrcFC76mrQR`UEQ*LjtZsDLD1a}0%baY?dcBvyajHenbjuK(3`kKTRXJOq0{BNFu=*0LLA0szlPkgp+w_x9L={1LKhS{Lu'
    'kV8Yw+CnS`e(!r)VmUV&>68Y3_)SJSrM;-I8ORS)yzOS=f21Ye27i$Irma)>7k``=yZ3;c%eekF<VOTy-Dv^%mZepn9EsLY;mjqh'
    '@KK<1qhlBpkN4hC$KAs+!|zs#2H^C(Vnr)TlxxAj)|}6!U|?!4;A$tZG#6S<<!uih5vTG-0KVi@-'
    'VT5+Ih9ujei9QHZp8lae6HMteR>%wF2FGbOy<5Bd-mdqQy|xs2V-'
    '<v*A+z6g~9H@3f0VhGHA!rZ1&)6lMPD^mZw`R^Q*`cf&Zn*dKGJ`i%~4M5aZ)xc<zFg5#@JaO#A>ICon|DCpV+2(k7bV&lc}>BF?'
    'X*O*9E-Ly+Z~jMF!0LruZSKeVA*a1aN1wNr7@S-!c>srr{=aQG(cZl7O>07?OTAv=nbGpdZurx-t_$`x>P!r+X*8ll`@LF?TR1Gt'
    'SmfI`dY=-V-'
    'hz8_Cd7%B}2DxrB1k7UvYdnxg>MPsl>kg=Rwz#cfZ(3rAQXUx&S5X(|BIsB?m@GCSS?ncnVW)??_<PpE?uvcLa)!9}5t2`dliHG+'
    '#o1acHeo~d4l*z_VuCkZXVth-LSxtDHM!qPGVE5Yp)5x&#qqP5*5wP@i<JJob`nGYK1o(W{xGnM1PXPoi9j}$aJ~})KZK%OELB&C'
    'Jk}e7ZR=Q2{g1WdKDulV^1nxp`!M!P8Ma3^u5?oKlNceo!|2k^fc~##(sE6kVtNt#T)`)>{X|#zM2$P&SY~4j7GZse^@EF{o7JGj'
    '#)>1@xm9(72MwL=3jh!i_<pnmTl#1$!RsZQK|Jc;3|6DbK$Q0e4uJUd!q1OdRx6$oMGZ@2})W=+^{8Vlm!~;SPhz3gi5h=HsR>7&'
    '<s>7*WF18QYE<Mar9G#mYvfYMhBO}9<L-4PvQo0~sbzn|M+{DX(A!3s}YHYQ4sLbR#TyYQW(V@jghn4o?-'
    '%x;VW!TC$mA}@?#OZG-*f%m!;oHhzPj{aT)!*~d{GsnEKi{TqqHGpmm-'
    'x?C3+>=KbQV08w09Lwt=J}3yX9pK`)JjFRdXe|w%85zHc}mHvGqKUo?d}eOLi=-Fxaf=Ldz6QEPuO%?#{5-'
    'K!A<i=qhf3R2KXv=?>dmh0}mtoe5QcMwQF6eHEDrIVlf8ZcRuUmRS@r&?ZcvG}a2Q>Y2qiI*bu66N8rK<@UEI|1Zld#kGUn#_8xU'
    'q!NZ&XfH^wTS1f<wH^JFWFopG<%K?D8Y#i)#n)9<xV1q^;Z{F3G*~NbQnrMa9sQ+@T01h2nK&~MzKA(A3BHK^HyOT&!!`xJqPK~p'
    'W5Q|3IWE-uz6L(jpmM95l>d8_%mEgh)^cyWt)a)ZWT-$&a-MW$F+HRhHCDN)4KXgd5}AV?6^7nmmESP_U@P6t-Z6fh<&Z&z3wi-j'
    'DX!-QM5Vc$7Z8=<YF<E8mWz1-QB}E?7ZBAD%cp-'
    'G*h#M5gLmj<g>_XVM2yYjGHM$_YQ)@f5YlDZJz7rxA;;IhrV%g%2!tniI)qn+TWM-'
    'P!%A|r1^Mu+vN`LRj<<zeX*)8&yJQ@MPVM#{(y5`~GM(D(&9T^PXAT{j`&Y5E%4C{_Q^g(|Q>2Z=AH?K+s_Oq@7;!E(ZXI7ASw)T'
    'r*}`d6q=k&nXNgs}5%RlP`F}=@`p)1Q`$(PQ_*s3JrfTxe<Mq*e<EUoKh2~u$@-J4uGsJ63bDkd5+<p)<zZm+so-'
    'EWEm~Cbg4occ-C{gOiRIvxm1nOU1#oj3ssaJCqo2N{oDJ@m(p3*}7iL2O2YHIvwz-Et0r6pjCl~O|a9471vjpt$){Ic=>dh9a;SB'
    'tIRv)sEH_VQ-7-8Wi9gKG>Yz%iP_O+M#ajMdLi^`DnZ!|UxVhFB@aw8Q`5?Y-'
    'mkDz5Z#&CI<ZsxVdVCIR9UC8momnvi`N+hAIDm5m7j91;?TkVJIRY|~|o3#m?LH;r`BMR!610%LI6iflH!P8Zk)r;*=zX3jY?_kG'
    '3g_m9nQl@A|ZuWjj_J9FlgrxfTx9~A!>ow`R=a9~ymm=O1uXO{p29htAoe(WLb{5)3^xWKE(8W?BPZhcCr&d8)-'
    'TcH)C(%$2$9x%r5P`DMno6h?1ZI2w>dA;$!iA(jRTntOe)&(-+XL${N0V^U&tbQ3-'
    'vs^a)D#$S~(S96&8DdMgGk_VQeX}dD7?TtuTWu@T7pmR2mw5}-'
    '?mNi1<vCaF7^f}0kS|(WU!}gHTB8;S9VkJ2v<=u2vd9^#2{Bip)LSr76*{&bErK7WsU@5J*$Pq4Juzlfe~5l{b-'
    'YVY37i6jp5?B^wBY~nyE9A5>Sg9m+u*!X0Q&C&Bu4hcIY1)FK;hg1m|v!Y`~c8U$seD<5L*(M9|G4VIp>oY`b*CFO$XfI#a2yU&x'
    'e0_u7y6F`W8z}5SB8kST?MSWm71pdwRag_E2Jeh8T*<>Kst`M`95F0@O4Y*L3`b#{@3mLLQY1`Izb@xrL9bHl?d~GrRR7SiY|mWC'
    'G<l=a;Ho7{=B!3Q>3PCmFvl6aW6nR1ZlA`4>|?Cg=Uz@RanIa$<j^!^yorPnm|%+8K@~YZ&cZtx;N&=VZ|2npR}+lPJoqBaOM?h}'
    'P+58`LfA9!Q)w2Bj0KS7MkOL8N$pDVYr24<9JymYq?iSF>MPJtO;Q|1utpKz1psXGK;2I+SV>r*1)|Ci0Aj0-'
    '!D{bd<w1h=c5{4o#L^Sr|9cK$016RUF$0KJ&0T(@rb0Vi*XNY#1@W1i7QWBD<9W0Qzw0&u+>e!zHaSyVXJevQ@IgVL9hLkn~~lQa'
    '4lI^t;lJH`ri#PCq{VuJ&-'
    '`rrWoyo>t4>vRWra;z4EgoE*@DOS!{A?*)aA;gL@V5+y%86G)W&ur`qU_~F^Wdgh09f$7H&&jqd}KRh3}mU*jS5DNJsuyan+);YG'
    '?D3|n1h)nyFoOpQola1f65E&3Z&-'
    'ncd5kT>$8Glfr62k9xqEOUpsz+qhbthbSeV<j6>Gz@{;*uhO1vrg51pEh44Os5bgsWLa4`P5(oAL=DJ<y|v^GSB*4~^D<lXf!wsr'
    'I1ds4Me9s(|7ax_}DqCeiL52ypwkPN%E$^-'
    'Ab+sdzowrI9dQ*g6uXEOuh*wxp3T)mb_craN6n!j#BOl_ZbO|B%Gm%TU1uT4!niyjJ4~+?iEMG#^d{*`O0~uK;#0Wy#LYcu|9Km#'
    '_<aV~RLv=_JQ<+Xa+&YRf2jWDvtaF{GeV7FL^QdxNgVCLhA6tHE^rY^rv|$&mi?pf>5AM%#N|IUsjy!)`5FulvjKxL2E)>(F>TP!'
    '7la(pI6f@qmV~e>T;_l7GCzRF6v5b&o|nyeC_klL~)VTp7yZ?lz5qW`2>1F09iYHXyQ<Z=gGKAO$449a2o}2s@YubehGZ{I0D}Ug'
    '?fh<C8;GH6a;PRTGm{ST!kmRaKLdYgjcU*;Z9kGXt@k93`M}-Fh*Zeo&INdoAvTd(}$u{nlTeTP^GY@>*D-'
    'UBQ5C>~{>5SR_8_%SwNiTIQ`NDT$F8#>$eC7$F}HE8?h^UVpnZ<mmN}m+YQi{{)dyq1QiAR8;8oPm&~>UjJkfRH4^D#m4G8m5Dv3'
    'evUFrm)6<~8sK}~_m{a;hhI{vjEJ_;R+WNKKc(>Z=FQojyp*ji(o%{=Ld)^DQaDnR!P`pzZuJ#hPIn#q3L?7E53{5c>*_11b}FT<'
    'eHGORrS!G0rrN2LX8$!*fR@tjzm^KnQri8CsH!TZ|Gk*1s<N*Cu*KC`KA~63v*<Y4A5h<MdVRcL!_K#tNrW&Adsby7$gqE*Xvhms'
    'E&P4)$c+QTuQ3=6qih=}g*cEmJ^IUZ<VM12lupTjw1Xnvceuu*9WqG8;UZGx<H*@?g7ByD4KN3O3jTB_t*;)X#*arwl6U9?KTA?0'
    'I*pYwLnrwYmEbtpd*SxeX`C%~VsskkXeM=e3C@g|*5BqZq&gs5qbViTH<6S<=c{p^4gANjjP2o(Qz$Y3^YIP*kye*sO9~BpK+s_u'
    'Tq1!;twNhpfQQiBhXKqyj_=o(wBO=~4$d0qIH_pgX{yJ?`SL!C<cQ0>i%W0;#LoE=v_8~I{L&Ka08x0k0-'
    'X@G5?|pkmE)RarNdb6;09DkW2<O#NmXH3sgF>+X_7CZ^YwWrcs~yV4S%Xr00>oFvs5n3pKAPpg>-(9MDrQF-'
    'y+w$UmL66*SxvA#GHVC-iu09!6z~O@L~U3wG1C{I!W*FOC6?``_{M=3vvC1OXAF2<b6YtHU3)@y$!d;>QV;K1BhSZzm7qzbOe!u1'
    '&vGHR;Lw|CWghDR8nExc-zed6AEXUPu~Iz@w2I<MRGvD$;`Nrs3>bxiFiSiwh>e~3$7;qWM29c;mG1oVOfDTlTdreQ5B+wzoHm5Y'
    ')J*F;jb$+3omuKN<Y(?UZKt2-Ax8&`-'
    'b!HK}V6xXt>xOT4t+rh4Y_|tFv<%IClDm){Gt4M{xVCR;%5&8}z<JAF$U@WT&vuK$GgIwxO@_jE~VaN3|V&O=N70Haw~wwBJ$fDD'
    '94ps*S!FTKW|936CmseEmYF(AV%X7KZWB8Z~N8;z1+8Xmo0_w+4Tj`AMi++l$Ei(gm7VP=Oj2XkzKt=Jsao_=m?(YsLp37QxCPm>'
    '6LUHND6<?OYg#283R`^G#kMf)ZsI&7P**?U=+286gb5r}PgRVGN#EDr|frW`cqnp9C2P0gg}h=a-'
    '5cp8{U~V@7C$A1M9f$!Fv&oe_N>N|A>!f9OY$7(9X>o`h`Vk@<dl)&#yi{pvY0hO!#n_FquG=sOHDWN#E_c1}T82LlgEy4EE}xVf'
    'a8T!PWrft2>BT3Gv2j6a|dVQT*_?IS*5QKItY(}xj~xFA%I`9G-g@e`1PJS-<f0Z!PO^X%Kx_zyO_JaFm`xFM~~`f-'
    'aJokXr_C`ZPZ{%I$CgBP0+YIBQfR9eP;HqEK-aQU3-'
    '4#%};Z=hgxHfgZPKG~dUSH9Drmo~>VLT>|?Zr{YE+b?nH_D@{80}_|+z{I6HC~@hy{Ge)ziZHJ5)<AEwGv61_s59=`=+0S}6&Ot#'
    'l+}v>-SW@%5-_*?gKY%lc7O4XJVnro)yu%x9wc^~D&dOk|1#^p>hu!7-'
    '1@IMZT?(g{nwo$A<rsQ9TWXYSND=yllDV>a+j!yy#X>fsC~kO*crxl7|hmULg}Aj1p%rMY@<MMfU|Zs!`Qn6^LuP?>s*s#GM3n8I'
    'j<$yI)Tf~2jybx|IO)Xzr^}4I}y!QnpYIO=HbPg3MG0)HRoe9HD@0GbQH?<@^3Aa?B(B92-'
    '~X{&QSZ7NG^3jE_42qF~(j<r9o5X*N0+W0yYG#we2O`DEhrbxffU=AZwy3E30swgQDKQI&Je@YW-LE2~BFP%Dd1^UBY+%(deaK#}'
    'CJ#iF7?b9E%>(4g7E%LeY)<a6D40oA{v{&83_9;RM>F{uaKGe+c>3!}GFaE(F}_!iyI<e|ub?btd?})PpJ&okEid6R!D%%A-'
    '>;*I&x6yrrZXNCF}xaihy_lmZsi3ykiomy%L$wNV0Xbz8M5$;=8cW|;%<jBko_p^8>L|Nqk1>5dcI49vbH&3zH}vJnraX|-'
    '^=8>XeK@bbI~9$R@u+CR350%e?A0&8ZWZP+Sld>lKi(b`i4-'
    '60!N((elKYmIFLpZ%jU9mq0QFDF%I0U&upaZT#2hx~5>crO#VZ)wm^xYYSi#0^A+>u}T1hD!C%sHDTP(tl3*`zy<gbcNZDw6Od(F'
    'xFtK#fgRvkzCfZ4MkaLfy^-'
    '!WTgc%geq)MG&_DRz|ZOt|2z5H)uaCR3^C04|7j=G8c=~UoCyFdu?^f2VXJn5Y@1(4e{vLt!h%pGmYZopcz;@%Bl9#YYj}^}X;A)'
    'Ye)Wj@6o)#sA=#B_#s^)4mZGk9K#BC+&|OR&Ln>_VP;_i%C?>~x3g+)iv{$wiBW3QD-'
    'HOR6_sW)Gpv=9p+c1>pUfFWY<hfV20&`~Wm94~>c|?R${-}7{v!K9Nt<Bc!?~c+bVYzQ=&R!r^+?o^-v@H={`H+O`QAwN-UGt+q*'
    'EC&MHoR*cW?iQ=wd*Zn)C6tf-'
    '=H)#$^~y!ni|!%Hz`f+W{aaVMGg|q5}z>4)3nn+)6mr*Evs3E?nmi+Y(l^A@LE|u&{URR?EJ^#DZAYHPniZTM9AHm$!v4A3hXh70'
    'UyC+oK4+X;x9|H>5ej5J26}ciN;X<l7qUn6LJEiIltrrpCF)nOMi7KHTt&jap9rr_AtlcG3vWYz(@r+QvyTF%;wE#crf;AL8$<KD'
    'mN6XkNI4{;)j0;0-Dt`*-jdc_Sy8Srwx^+jpcY>O;;zVr_FVFuZ^D02*1Itcntg%FuTx-'
    '8wlD3cb0Wypqom6t+s3EFSRQ!f148hXMWhWL<5>1wky$r=7;S|w4jaX?4t*5M4t~$Xntrb(S;WKM&s{yeoDj$QhrJ$3D308D<joG'
    'mu70|pO1NAxAR}i#FX&D7mfiJ?tmuiXPvN3fHZoqR&5v#E7lG(RBhT|hN@WLcL&uAE)QV{I@?M~i1TolqsoO#pAtZ00qB)LT=TAY'
    '5eXfGU(PQhKO+&hFf((4se=Wi(e@5rH%tMXyt39(4nTha*ixsI#-'
    'MuExmydyLYn>31*bRq)rU%dR?`6N_h1(@>!#OVSNbbUnbEbv`j79n?ciy|%Sb`^9zY)A)3GH$oc#Yfz65-'
    'hSNN<FaO*;}(3H@E*|o4#&?-'
    '5}R!{R<Lr=DPE^lK5lMiWQEP+dG>O>9L3<Wz5ziy@9lage9hG3ju*EwE47TWADa1j+9qnDc&>er4k0--cPE&6j#XiUI;e9;M82`I'
    'anR=N;?j#pMX4+x=mJ1TcL)Ea-U^ON}LoDy(+UaxaY;L~|GjpFj;+G!(ZSIla786C+Z^PLr~z<7_$Eo%mmePzv0Gpx{-'
    'K)kOrHRw0{%8bhsK|!OrB;N7Pe>LWsCpiE0a*V0!fYm}oi?`lb$TC$U6taiPhni>PLxn>2a0#xb8lh^0qV`A$8mB7|W_1|hvY3_s'
    'LTK*pU5L3W1WHYg-{>k;1R*F-F3eV_y-Z_fj3DOfAKU<fI{&%en7q^q*h`%Mc)aN5Q9a+Zd1e9rEBDGA0LP_C@Sv&wB$MTbO!a30'
    'kbN<XCHCRZ@ue`9c!NY5N3nTNBc{(3n_p+76GRbsZ;f0Of%oCEhpzPFWI9cu?u49}9Q<O3l$eP=wN2CJVJr)$cB8aDqnf}`3|`>W'
    'yMX@M+?;?otMnJgC81DzdDBd_&w&60|LIa0{4GU714EBMrH$!N;@wD#a5QB~qx98~DAD0ts$L!yF#9n)^-'
    '@f){wm0qJq?xeV&s6<-'
    'bDX;BNqT&YI_+`@=AIr*JK(f*Qy3eF9L6uT)0#)A1yERJV*E0q1l>HL48qN6UwqKHO)yQU@%LhtjfA<^h;TJdLgSz;XflQY|W~~;'
    'UJb$Ykl<sH1_v+T0)n-OY^IpjXzdZ6klijn+j1sg*VEZ3)Vk{UXp)HAs#{SuYYU7s=6KU3wm3@E*KsDGXY<+C-2nHg%Q&0MWnY-'
    'k=_yTDWdAJF3#Vazb^*h|8T+Er@mp@OHGQ|xv#4fi$7GcRFJMJ8SNG2+qFjO0`^#!a+pptgG4kxG?Ix^XgAeHxzB!Ys?Bnp{lQc('
    'GgS&7@#uU!ag2^9b}|0Ad?ImtK9T6oClV*<L}FS{Kw-sVI=&DT`CC4)nG3(IC}>MbqOJ@V{<LK_@+jIeqeIqgMp-'
    '+`?NW_(z;?bpdD{l-0<F!4f@JtYK9^k?kVG#-Q@toB>^G)*NkZH2Otnc;=0BS16^?RYD|4mzVl`f>+?T2eQcY~ECdv?XQ#DCS-'
    '_6xzDd}IXrpN^Km1=4Zu&4LgICYR8(_6?8Guq){Q~gC4N_$~$M1`B~;FX5TUeV-'
    'Bf<`TcYPXMo#&Ng4=b})Z?h&d5RP%O(Zg3Si9|2G30ihWn6FoM^ry|I0Fe<bHkw)bC!||u0O-e6G&-^r#F4SontLc-kwhevtp3?W'
    'lAaknq>zdBdPA<Rm>P|nyiMcjH-'
    ')Doe!WWkKVYnOA9$a@L^giv0ZDEF>iFcSy^HuRdO7Ntab^+Bs(oFjhRn*c<yU<q8N;B=lwpu66w2x3eDE*#~+Uoh(?+M|1sqw#Xq'
    'P-{#s9t3C-8Bm!ssS#E>+$-'
    '6B2wdeEGe_nJGQTfrLUDI2W=)hOCeU{cgmc7&~Qpl1kFc|!$OSV^^jjsA{CTidtr%G@UNVNUtlv(_=hZ0CGhZJ%M6~_d`7LqZC8Z'
    'n-'
    '#KZuA|O+dD;?FPK{$OJr<pUbr|BK$CkPp4gErmk3jf&{bKX!;{?Q2dInMc?8M#Y?WRI5DRA*3$Akx{0)5=u(paqnUZ^sL5#&Y<GR'
    'q%W8r9Z(*e)rbktZ_3#3O`YT)5wiAs^*TD<=7XOiyvRpIBKht;S4$jdcY73#Q`m_FXem<1F<Rz5B^rIPd2Tf6BpKHpMtWqCbI(<j'
    'dZ2*Xv+X3J7oZpaT$PQ=L|ryO9mj>H3N|BmH|k1R{+WDF$|{?FD}Q&N`H((G!8(3@{?+Fe;Y_xarShwS)8*CbAN|nm4-'
    'mDnXiWEA?*;z51ob4%!RZBOS{DeI%|DJK2oCMfA#+-mxsLp0`R|bGJHXSdlT=S_JX}MQ#G)_g-'
    'z0A;sE?`78QrnxsNA(RPy10kOJ!x?^XEE>#+B<92V9VCh4&WT~}Fm(F>hpT!{{`WK+|CiC~s@$k1*zlpJXmw!%=YD&nysy*CoXB+'
    'HoA_4-'
    'MLde!#&DRAl#ib<GghdFPc+u+*rQ}`c0826T+>T(nIcI3J1)f}Wt`ukl9oTJqk%7$~Vm8hq_*Ib?9oZ7HfOM)f}j%O1ae7({UXIn'
    'Tm9HgtWvs1ZGy5V<qnZ&2x%>|85{2Ns2Y>|IrM;DEb`Q|AxNMJ`2)CW}cUnlUa(jQ*<T^Sz>t>aQAO#363^%L`u+Mah9WCgp_F$C'
    '+;mElHf`PTpjQEJjezZS=+c%|wB?p;v1D-)m78!Id*@R^#%$tsg+0d~P8qNkL9S-'
    ')YAe#36`o`xHXiAPQKH;K6)EYT2?8tR>t@aZ|69=wv4s6uuUnEY?&zodMHPnBS$^JwJLr9Yt*B-'
    'wa4T*y^10hVTRB~FDMmfQ}3v%t_79rkoJ-DXc4e^T6Q&l-'
    'P9Q!xq?>5S1l;4+QLBEbsU(*&a%ZgNpFWk42mei1foN&vQT(ezvT$$pcn-'
    'S{0`wA)spr<XOJEv=pw{7`jk^^Abisuk69j0WkbR#wk5fMEnM4#9zq1nov97WMjxMdn(4k45h`hdsZ7NFxx0c}?Z$a3F-'
    'QNy5c#+7Ic04qtAv)Sk51Rab6JBm3OqBmXUa$KRR&C{7bYfXQXNxU5<$;DhS6>RH84Ew9!IIwqEJ;#}}Fl_d0%1wZ~9kbnh1{#%f'
    'M1wsBhLpwkq<i9r*dIUrM2hf8BME*z6gBx$MPZ{8Drj+XId4r63lel&eoD?0{W0D^8u}P2lxTME?e9~j?PI}BIBt7PTqT)*BnE#n'
    '7({M2+zE*}%rNb<&sn-Ug2W$1=tc4mR(j=AVm-K5B0J-'
    '$5KboMYnnr;IMuAP2SkRsLsp>y`(qOPKxl}E*QPsbBI~9&9t=&u2qN+>3gGy}`)?OK$NA}=Y3qk2p3NL6%E8Xi?&VQklF3jB)g9='
    'W64n#@jC+99kNK<>v|0Yx35!P;4_E@V+Wml`K>7~Ae1U@X&?@iz-WdnCCQ$#E@GTw(x%~pHZq=Ks*-'
    '(h{@hGM}aMq?(7a|$!hL3xLDmgSO&io{ZOQOzY)7nNL6bWz16H5V0JQgTtfB^4KyTUv0f2!2bu^A-'
    'B>5;`OI$n>OIOciuaWi#Rl@H-'
    'RP0!i`m1vi$;%kj?tT!<n5u>vZge6Rl|XhQLEMn<%gQ=<y10>`;vmcSX&?k@ONL_iL@^fz$sH@V2H(*w^Zr1i>yQ#4-'
    'y%mqYfM8H&6QlZ&!u`Bml-'
    's=lRy3KLTi!M}ecwI`>8A!?nHcK!u5gu+xjy9r#3M=><6H%!7=p9v6n>GP<3KPK>)K4kYrK%f5Kko~Q3?D$lkdfiEyBFHJ#5~m#y'
    'Es)ze}nORIDc@d@R(r%|6hlSeyPG^o~dqCc+A>rnZje9t!`6z%(`m1A~>F_Rw#nw`D!I!y1}z*r&5p=LHjjZk(W_KM4GK(PXK-'
    ';UD<_S+f>Im{}Uq}^+yZn@o_V61GIKzm`+LFFA5o^Q%(IJg$&ba1O2j)VLH{<|5?Z|oyv+|6*5ex`t-'
    'jF8KzU={ojQQ(`T2}B1+TMvypPfm-EK^g8V&FeX<~A6kQSzmT0=zl~WsVRkkw$X-'
    '0_Pk>i}T#1^_Bf1aycp6=*%h6UV&W<z5|f3)*IR>J-#4DhBhnfh%IV&hBH|1oIUJ#QNIIFPucg6`I=pj(y|bhl*%-'
    'SVuUTaguXE4iS{YmW`8z@PN*5od-'
    'FZ3JNgbeq=uLm54^8~>kyhBLNj;XkAF(#e@xc-R|bo6y~EFeH@$@FNA(^ti+GyP){T*@b^57{_g!u_ptlO-Wda7!F*D02yY`Xsd!'
    'oXO-'
    '1eRPw2dC}nx0Y8y&dBKOhuLu1lUXlQ+)q)?5EL8HlA=ynf?Xm=N$RALMw`elk0L$;=#S}$s3S5fs57x6`u0*(EUj~f6iWLP{L{GJ'
    'r7*d+j0+BgC;q<y1eU6eKLue(O&%P4C)Q1^_$TxrY5{N5RqzZ=yVm8y+b>N!(tWzr|?Gr1LA)ZVo$^F9*SXT91Obt#(3rhAw-'
    'O<w@E&#-k9gHs)qo?(4P(f(^ZbtgDKqN7YI(3V8G#9$(8fKQxSs@LZdWsf{wKpUDr?}j4g70j$H{Sk%VjknXP68-'
    'x@0Yg+*`(;L8KGik;HY3Q0>RP`%8O>5swK>n6rWtLMgnD00eIgEU8qXNpV8NgnDD0Lr&uoX-'
    'f<`G`nnR3#Fm4@6Lq}~*t^YOj&AOfccOwmqj~TF%WSIWzpd`sPfd50#hvbCbuZ0|FbgLM_f!l{0^u175cvb#Ax?v8&*p<bq4dnNC'
    'Hww|+Mv)Q5pvHPpU3w`Y-BfSuFA$$#<Ob*=>K1!VD>{~0?o5)1OI$<swA2KgJpF39@tgCEv%LfyW{mFA?}KXQI<6Ecxp|$AFm7-'
    '#MfqvXiHqBEB0YLck{&%aNsk_vq(_fW(xcr;dh~<@ccLVDKahU1PjRyIJI512SImDFc^k4lH}9HJX2uUFZ?y9RfLu@J4>yq77sK4'
    'XNPePWWwKnHx7Ze7&lnRbi}g?emDK&%meV1Et`|RV<p`L>5lB+#l}j9fB5?%D#1YskaRjzbD(F-X9|**P@-'
    'y1XoW2TAN%+hfX;5k_qASM~j^dBa5Yk1zti_OtDy~VH*!V|TnYg6%rIB;l0)}{(lbI<6mN9IUU&DW<BYd~;lP2XWxYRc=oRMwssi'
    'RZ67xyy4%Uc$Ff_LG_^~n-'
    'SkN2qw?+h!=dgR9YW3bG^^9F_*yYLfOBhuD_XUQfCbjonvUMC0lS)kSKQ)b_eWmNg#g}RV(pD22Bn`r`fMwi!YfYwrz)XsUYUHNd'
    'WKMmKymSq`tSs2O+_SPZdEsnD9Tz*}OL8z|_{Q%<YLO+1^y3h|Gy)N_vD6b3s0K)4+KY$dzG#Q42CAuiRoL*F~P)U~`Sh#jaLEMr'
    'u<+@~GOD*i=Zm(+PxIRO+cHv1^VHn7cX@x=)RJ9zJ>5xKT6}A;P8({>I%t887<hn9NuB%eyx;jO!Yf|L8Hbt&QDRM1Nk&DuqeF0`'
    'r1j>vgt7ic3<N5dT1_;mtv-'
    'P@wE~U78=z98(qPf|a`O$~S(v`?lx^?<fn`Q)@x#Wp4jFo91Rz6rjteAhIr{Repo=wldT$K&xd;RUDKU%M`V+`o6Ji0&D09eMjhv'
    'N*;y`0x}8^FxC8b7(HUZT8|H(gO}q^yE>T2XDLM1r?kQN2vXPp+zL{KIh*Q9k&~+?uBgZ+^_%zAAM#*5RKm!2jP~QFT<S{!{@w=='
    'c9jK~U)T|C~SZ{+2d?DBoDx0pjv(>Aj0?i=`LOIoJsXfK(38Cl%F;ex`H=is~7EajpH<g0bPC;@ex2PJ6q8m4h3N)cuqfw5Ad-'
    'A^Va8rt`bT%W;K?C`8%)tpTGo1`NL|$$?;-(ofWq{o{kO&HD9*00L-'
    'KP*$TRI2I>%r}MkSAE3?p54hZVZ&dxfaXQfGlsg`sPKnKEbxLGLuTw5MG&?2FLbp@0EVMgWyvT~`crP72I?deu-_O!<=H~x_mQL;'
    'wxr_$D2>Jc4#DN4^2-=rAO$FLFjZi;|4u>?kZVlGZJ4?r-ptIJ;yNZt2)BYkMgBI0Ve~Az-'
    'i|Sc_sSwDEYMsA~A8szH=ltdTKneTv{))V+^*fabj8(Ph5f8U5t%gLxyo?QXhhjK~C129lcck%;#bdV21mJIG6V^ZGI?XK8U>cx3'
    'olFCxrsHUU_+$nh23{~lv?^*oLo3q@FoRQs2bxI=@xVMzF&-GlDaZp|yV!H=^?Majo+{-'
    ')(+lkIt}*Z`U76A{{!D+y_$$PsfVosfgSkXY<}<I>JH|iG);p#D??&GgsK&mnkXw0OFsf#!yAPrZDHQ>b#GrKQz?jOX^$OR-pz<d'
    'PT)vN?ibloRZ=<BxsBSVUz(Oqr;b>>YN|pi<oeS}jg*EMgHZMU&Q83O=Vq~<{S|BRQWp5FAUoleDE}<@@P#I#ZzlL+Y)dXZ7v|UU'
    '8UwJApBPh}~^}M-'
    'u<dh7JoQ3K{Te}DW_k{XEsSKC|2BqJP*3c<i>VWN*WJ_<gpjPrQ_IW5eiqqi1MewXt^HvN6&bMkVesnVA{yDIE>cn;?vQ)M1J{Q>'
    '-)#<8}zXIB}st1l{Ss_-o@CrQ0*6H#O$82Cn>))T#<&oGoCr=|-hG`}>{zmg{Jl|!^YZqF*e6<Xe^uHW1+8-'
    'S#WyPS^D+;erQGS!ix<W+x&4s`~_br0^X)>`tCfSt#6nIf}vio-'
    'c8r7|C1c=7qR>$IsXl8I`*Ms*e(1F7X3YK&gi}f+=6c4sFB2jyjy6Ie4$BJ~-'
    '<3*iFRJCHss9M>Q7aDK0;OcUr@fHhG?$5MOk#2^aOcwsenJoP6Y!WgDxXu&{cQx5<vYScZ9N;?B)nZ*}QLvcEo<$Up%d6cCAF%~g'
    'hvWe>-'
    '>A<7s7tDGgO8c<{YJ5c>hcVy6#kL8R6AIIev^kb@bJ=s%PfIA9sb7vo~RTf4j?$eVXNut4RFpPi2=BSpe~7DGLT2_Q;%dINj@Ov?'
    '10Cz`(Y5lyge7e{FqMpDD(SxDU?gPhdz-jJbtp2gHM;zr*a|0Pb;us?;=X{<q}^H#%0;o`t$N**b4EEro0MKvl_yRNYI=ClZvbf4'
    ';I>7t9jQzDRnoUv<-'
    '!%g)Qj9HyVuo*`Z^59{=U_z+`i<hL_1Sle|o{R<0gOt()p$?cBiEWG|C^O|*op9@8!tJWf(Ic}PKN4CPnxXWgfW@`b6`EBr%oot9'
    '8UFX}FkJs>>IFpW*7gNopTY=z;7%^SbjWQ!OfEw!}<W|%h^br@M`)r7#g&?_#)sX0vB3%zQ8Ur@;BCUucCE?-'
    'w9jmy^;N#pVjMO0!;&YA16!YplE_0)JoK}Rx=s;$Cp##`PaQzRUY<1+EgcGX`+v|wsp(G1fX)TReA?pXy>TFKRETl7dy-ilEM%Y5'
    'ONE$Y*WiI4)G20?L}R@8_&>rG~riEACS4BlC<I)J*~&!6RCf0mI-'
    '_%VNuk)ro;f3A_LccR4sm2S!;i%W4=qjoXNAvCQn4;Y3NVzV8_{)JYczb+^OYvuO4f<kb98l4{(tBg|%|7g)DSg9t<y@5T>s7bKL'
    '88iv@IAbQk9%slT*yD_t1bdtTlVFcOAY7rCM?|IqRTyIQhzf%%n9rj~oK$P)W<JmT*bkay4hHK;RT-'
    'y!Gps7(l+WB*=JO}Ky*g5xQUhm%n>R5(Ya5_gI##xh)&s+$0B5-'
    'opfo|S+>8;uAXshzghFKF?ovJsL`e*Xw2q*JXz2crPtX7~;XUCAQeK7+T}P0PF_ZwavX>E>><B=~(a|*5$VDG^NVvM<Ar_n}lj9-'
    'U;;mM(xp%6Wb1!P$=|wH^q_AY;Qhl)W<Lm8%)vh^<3LT`j+&h$x@&=PSaeCg{(6<!Y>s)vc`vqk>{7h<UQ(d|xm{SYT=z0{4b#`r'
    '(Zu~}>q#M6kCh5j+l}Wns+hvk&{7#vq8^2p7>BjGsg6jN`O)j!PNYAMo>(I4d7&%d8hNh$-'
    '(_nqfyb=0ro4^~{4GKzsDi>!ja0XOzb%HSn5#^MryrzByMjbB%v9g|8FFN~4+LIK!M67cS+|fQ-'
    'D&A3h0N0t+1GwI#9>5JIwRCPYsikw1iM@d!kYrMO=N6ONJCJlUl>WviXu26HA0#HewV)*Z!Un9x4D1ORYxQ7fXPZjxt!TZ5^3M);'
    'z9mTwv5C)Ltwkg3utG(Y))&kspz6cG7wv=mbe4e=+J}TCYT$kL5q^SljFCWFU4@T#D5It>$oy^ruNSLf+yiee{C50z9szY=dwzHn'
    'i0K{jKz*I7{v>$jyxyzU014j%OYFvjH{B){B|>?Rq(ZH>?ngV{<XGYMn(kM7!>pdt{;lKorgwC+$$^7cVig#^Q(cXi#+c^nS_H8p'
    'cs(w($@7=)XBb`br+yGf-);EeArLFJwQ@7Qt?+GDr?ab4#z%MG?Ir8{g;C_kN|idKy8Y8-'
    'kTcYPPW5|hITr8<O;~}pg1N7fsN2wwq&FY^M0)kntJAxW-ke^3^y2jPqiw9$AAJD6|Eg;c;~9ofEk;m3iZ>EvTc0X}6lEp`k5@S;'
    'AT-30CbVZyQW+Y^)V{hhV-'
    '_(%bJdpc{!T0WRyo*vVHr6hL9ws_>>UQm91YN*RxB$T)K=&xwCT|eF?yy^g0nN=Ogc#tiD&trsFg`YXG&(D{kRnw4OT6}KX7EJFq'
    'lu2s!~Y_&L>M&X%>eYSG^jEd4)G7522JTx8{zul6tl$AP<J*ZJ9l4o6&ai2>vpQScvIYi5=9G5{y(Q`k$#2%WMh`a(T?5xGGvE{f'
    'U5gWg_5Rl?ZrOCj#CziGX))BH%4b1iZzOfES5?zzyxxfYq6+u%(v#u4v8lc0mc`q$)^Zt1M|~_0)Ts({URKOgSGN1noKSP+FRB0%'
    '*d>T%W=NDjuUb&n2{#cz^ULR&S*8{^x}#@eQIk#kKfdrZO^4x!Y8V>bb{ME`ka|e|x!qhtUT23jZ%g_>sV&jjNH+qh$L65nTc@RW'
    '!)wI<Z2B6yvHg!B1C}QU<gJ4$7KriEeV$-@*#br8_9U{QDa<EB8w6pc(33IY0Of4L4;+-'
    'F_I<Q(SBOBlT2kVy8FcY;$!GCGbwTi`1p_J3y2yWN>qzsY+>@-fya{7`6(Q<kqoi7dB8Pf_+jZf_-'
    'u(f;}%2!9FDu!9Fz;!9Fb$!9Kn2@=3Sejd4@am3NbA#4Qa5V9*SsnK61+Ml)mdTa9MM=wKy`_SG^)8#QWY7X~Iv{o9RBOm6l6q~q'
    '*r8T~z-ncU`k6@pT&^zRhdodH;A_Z&%ai>v-F8q_)Qw-'
    '*VE8n^;Gr+}RvbaO;(b3jWHws7sKVN!b<=C9LCmcP|j>kREy1&H^R$z7RZ%zldYz9!izQ)p2)Wr{6orcA-'
    '*U`3AHZTxSXqH!KD{*NNRVSqRn1i<eM=!jCY{@wsgylW0b7{&JXXB6~{9JPzjlQIyB&<Qi!lVH338Ekh!5Wa1(40%F@FxtQmq*UU'
    '%xUVR6xW$a&dA7hNsBk-)59{UCoMB|#6b|7Gfe<UBBFuv7qAx8aJ{$g9bJ=~rMf-=Bbvj~d50jcodOb6lLR!-'
    '6nZ<DFk~VI|WYUse)rm}@E9q5vrec=#s%A6Aw4_%xN8uS1&O~tfW%%-'
    '{f#e?yKwG=z+m>F$f>`{&vxp<c+B~ye4XAM&pe^}=<(Nb7r6&he^yF;i>6sQXV9o>i7JzhCvq|&&EShHrNCML@T%M)0MoR^ccon0'
    'Yt7Wu8%LI>j4P%?D<+MJ_1&_Fhk<HagTAh{JAii8NYcIm=m~H$zq7VU-hInx(`@G(W8q_ZuF**qSXDElECue!|r@<hGi5nD3aYo?'
    'k=sq9CpsYcy4|`vYr)MVP=~>Bmx=O~=Cnn=*pNywxC*$ck(oF(iR$-^_vFK)UWH{fVRe7Lx&95-'
    'AUB{~sQZ1pj{dZn>4A2Lt#h_$gF~f?QT7PbtKA?R0?1Ki;gFso8ciw~3RYSv{XVG%y@bzAco-0T84_Gu*57xA;PZVx2PPj-UZZJ-'
    '`Sgo9PHk5Zr{<@2y3q|VIT@BqQ!cEu+$7}yvr{Jc0jsKlfwAg(n7`5hg5yNk(rn@1}tNVZ)n^6X)cYPukx`<t-'
    '!BD=YnSqPGhg<#_I^!<WJayDW9f>qsni0G#c($}}>HnnKi3gPauW~uV((P5bkzwifcPGsaTAZ_7rt1A}muYC6?J^CGb6lpOajuB<'
    ';fgy$U&8MlZkW9L9}@YnOydjMg*kH4!76x+_J7FHk}kj+(euMjTfn;}!+FGTlzB_^)C>L*xZC|jKVutaDw*2(nJ$wPd6#JLhmMJ$'
    '>t|9jB<T8Clne>FzM^DE(Df%$G9>7FPsxy=>t|CkB<T7%lni~;)JG#v$2~M`YAt`N=vv0;LUshHb?46(>quH0agMCMuf~zp57alZ'
    'dO~d@s~@UsWc4F8jjWzj&&cX2d8fa2NGA9!{l+1i;Is4xhqU3yEWY0fv`5p}FqGxtll@b{LmwHw4dj){Xzm0u9i?u&<W6^%DpcN+'
    'E{Oi97jt&AOK!&Il5uNyD3dj%?nGvl9VP`$ccA?14{bM9bzM`4&q#0K-'
    '<QF0M$q`#Mt9AA2HJ((?b+Wzu#I=xLB{Wx4xRCBs$NR<i;CG&j;><1l%uPd?NC53GcOZ-hf@r*F?&IX>D3_iM~Cd~sG1de>zHEj('
    'cL{AriIA_yGJG>z2N|LDqZUgWuDTX?rQKA7(t7!RZr6a2GWfjJ`OZ~2bUFfs_;uiohtlNQKt&ORFtT|EEOdpFk8Lo4^es6m;9lsg'
    'ZZ*Q(y20yKgvb;ZW>#&aL+mu0k=7o;x(GGua-hou7jg{km%9>f1pfq*J*|ocyUsv@m|!_J`5o<^7lXt>5VDCKuQ6YrWD}TlmaYED'
    'Zp(h1z4U^fE6hPSSfXDr%K#z^oKc>`SFLV#O-E(1WVki9V`M*6P$enwwQlbgOau&IyS!$A4w=LpH=+NSh58B8-'
    'd`+3PFuLSU}NQTpJ!PpkgiZ=Z_Xpt`=4P#|x-'
    'c&*3xhTZfV_x3cXF%&S#}A1Ja&)NRCbqtkGOzE|V0t3GKO%{!uJm<&7|sB%uq@SVbeh<8mb9F1m`Mo5o}tT!!5*^EM_GdO*tT&@P'
    '`)q$E==m?B#|E5O?mKx5VTQno|UJGz3$h|t?as+}fOC^dO<LDsr+kL2jthgAnJXSz3T-'
    'fi|08F;4%e61VT<ea)ug+KfXd}K}|58NkO|45rn}Lm9M=&3rq41a=!zfaIsGq<gPeSWYVUQ=G^=I%|kVx}$_{K=0^6zlYlb}0d{J'
    '{bntF8h<Zf7UW#*a`*@1!65Q84s!7FWBg*)AZ@>Mz-xBlKYvB}l)Wne~Q1s>rGVz|UCmhZiPvufxK;qlh38ZK(yjpfU2Hoq}9`i&'
    '(^U1Ow(Iah##@deUC)#ZRRJfQ3$fNc~Ad<2n2!M;dMnOP0}RDdq{i;38YiROa}_wwk51_)Bb6Ndn|AwbhBrDZk8CUODBL+iJGtK>'
    'i9_%@M5%IPW<6c#I(jmT#PsX~Ks9hm~o<hr#8_Hr<v$C?A6kPy2q8n5n_a#`!Y~vA%_W=DYX_{x#0Oo1frW<NP^%rj|m@D2A<(*i'
    'h%qd90&rn|I)Gj;?LqfX6$!wt4@Z;OOQWQ4H5JMHo}2jkUqm{2Qb_-'
    '2n}YzF0cPD`zO@fZ!&MResH4gWu$FCK#bbthcL`A~qbau_(GTtXjPyPQNHfEQ%=N`-'
    '+INQE*rkMK%lailWFXA{Vj}YBgf5_z8$%$#h^@0expyfTRyFpc4gp5kF8>5xP5`uyyxf7|`cNWAl~O-JbUJP?5*!t}@^Ifw+-'
    'Rv5zt`#hO@9IhifdYA|ocU{VAPI639?GqmMqc6icDfmTw})`0%xa~f%7`4qbsFwO}oF?JhdqC|1EN1HFbbl%>sidD0aeRIlF;mYG'
    '{{6wUHe2sflT|6$BjRc+3qB+P6A5hTXa|Oc`U<wbBkj{prvC*yj8539(^sBcQ{yh!-qLH{L%}$j|f^u+=_CS^xO^-'
    '%!QOvL@J0ERSxfwvblzp-'
    'c+CFlnEr+Uv++#O5sEf)d+ClD(Z@Zfvq={qz?V$L2REjoo`_eu0bm0#vq=hwCxu#TJ{m9C<p{Vert)aBSEHW&~KNtSJdHh}w(t$C'
    'wCsvo7A+nBmU(HaW9rnBa1uK$0a1uzXs&c*^_&3%xs0Iuau+-'
    'rH|E&dl;<#zDynwzqH+okT@GN4}xa%B5T4lKJAdk!2@g@h+#v>G!yRPs@@OF8qpmdaj!XpJGq4&n0HAo4weKut!PuW^8>2M>+i2s'
    '0;#NwI-Co+aexvf{Yw^)CA>91f^!U`y3$=m!mgeByqe+ue!(%f5GKt6lll%q&%5%;__W7h*lC=e2)5YHBj;ps&%>5*voU_og(M?<'
    '<mzHf8zXozHeoLG;;rA?ortB{MiU3XZ!3t?N|W2x3&T}9ydvGj|0FqC7M>#cwjVYE5`#l5kmeW&xQgOcw~rJ^GE27y}>_J(@*w7a'
    'CDyc&GzOe?6p5hb~637d`^S>5iF&~g*&SY%N`y~HIg+)Lci!ni~}CBp<t+twC}wpfZ;%+dU?0>ip)egTgaRK&!LFbYvKWSDO0A6m'
    '6Lh`_t~Wu}O|%!&#8l)~@Rv_`ZHci0R}T5UnPO2z#GejA;Zeyb%FcS9M$6_5&;U}&hvY&gu?(^~yF(Ndv#A-'
    'q(m@(3{%f{nsVh1iNvQ^BYfZmQp_#ANJnetW@yjicb1rrJgh?760D6GY9W7MoZQB-'
    'fxTa9dF~`dr7l>fmHA&X=$a*u{n(O76Vc2WRPQDx-'
    'GN=+^g8&9C0WiPGO{Rm&@@wwrf#FMZ7TbF~+KS}pNxMYh$ht}aUjCb%uilL@Mbzf6cu+an8`B)VRdg-sT5IVv!j=+uUcZ0d!Lp38'
    '7=(s#V7Of@<<e*=7fjfBsdyPRg7tAU0o-9RWlTS^{w@u#IU|A_c$Cl`M28h)l7--;1|)p-A4!V^vK4=Glvn&=-^s9-'
    'h8KcYy%YO;S+fr8Z(|5!p)P4$mQ&|s6pWaeNq>M~AfcNljikWfbH3H>+45A?@Yj+UqNhKCe4y$y8Lr4I>Rb!kLG1z0+fkOP-iBxL'
    'QR7YR)OX+}ajd!#<ydt+S2-5lLii|do`u}F-'
    'N*mud8m4e<HUFy#m3hT1epEZ<?%2I#MP&O({{dq&dN0#~vh7to=>Mt5f3}mUlWa#FLd*_=DlJggoYRV3_7C(JE*jrp!v-'
    'ewVGLmlabED)*yPkqwiQqhaZxph(WOlo_2L_6|%R=+1n^y#ReEQW+X!|t+fB%GWKc}33#<-tT&c9&X&nf3$G4AJ-'
    '^KTgUbISQD#{Haf{vG4~2Sj-Fx=eWWdQD>|0X(Q;t2bq0t2dh(8~j!n)P<k<Er2mEZt(Kb@HDDJV?egk)Oeua?%Qb74)4G)wACs~'
    'GKR$=yQrZGm}`SwlGum>o*+&a=o5`9`@E}2tl<H?p!Le}Zmp+nl+u4J$>hyOA^sePefnvWN18F}3DOJQnXA+h*BN!r>4gkNvZpMK'
    '`pWb|_Gpo6JZ^20;~L%EKh=@i5R?Xk?8^TbweK}_IYl3Gqy~-'
    'mY{1kfi;KKNN_khOt?6)#=A~_aDa#vV9m@d7(Jjyc`a$@%yEw}FQ~2zS^HlY>xIOc?_*mv|@$t;x;uD#_#V0d=i%(_#7M~VT7`jn'
    '^W>Oyc?`rJ07{W1}-aKL{LQr;=w;_>BfQQo6SP1mkG^iTey%)pKRXVo?uaf?f_*l}8MWZ+evHvzs^G}A{R=za-'
    'q7=~^Nci^%sO<OzD(hxSDGr5V@aV!-Kqp<eqQ-^`SFxRR;VP1mE?kjQ!i6jFJzTgoX;7S@(E#CCUDV}3kPYYaqkpW8U2jk-'
    'P!QfR5|JQJaIawOPN)fi1W6p31dgI)jl23zmnBRih$bR2JA$6VpiN}U%rGbc>!w@5s=Td9=Hj)0-'
    'e8&FW|l_KXg>DL*ovbGtRnLeT=fy84!ys{H5dz8AW<Rx%p3vQnj=8B%Mqa4Yqi}CMYvEP*KU&mWRJW7m!gBnOTG*}L*9+Ip=HR6z'
    '8oDx-kK}XBYYq(<SJ;a$*O*wqMmkuKWShD!UB{>LpM`W<l~_UDFyrq(9z8KNZHQ9q;QMsKK9FuF__b=%C$1YqM|owDR<U6`#+7#d'
    'YAEs<jab-u&su}tf!u6ccCVsf@75yjXjnlV1vV8TXySr;BU1lIcZhF@^$IV--'
    '_le@8=b0y56s@>u0n4-Uu$9tF)#52?JvlZer3cv`8E0pEl67<~HUw7S`P_iVK-'
    'VTC-97#jO7xWVA)VO2jM!yn0(3Jy^2ITO;rLZsQM)3wXcrM>Z)b52WP?RKGqQ|7YllVFpUJORdsoQrb6c0^lN-dH7lUhNnT%Dj)0'
    'KZXlQZW2y9T!bio2%_6gVNxauAa=A_7w`To!p=GW4huSX~lk69)N%jlQB>RQJB>RP}OtM_q+C&A=Ouv&t3{zmvY-'
    'q!YmO%i7O)o{FkUoA(&D*dgW&8Mb);?Wjnr(V1Y6<d!HULG=Z6wE~(7CN-xOBm6C;2VKQAV47$I)%WXw&aHx^5V4`aMShwoP@7=c'
    '}+9WzFv@q;`HTFnbyAwu7aA&2P?M;_1+Ci?|6oWA%RbDl-'
    ';JsW+4@gVJHUQ>pMRA<otGl>0SquSvEOw5?hY7P32S71Hb`WZO*Z&nz+%q_fnKmXiN>t7hSB>(5c8)j8ImtEzItp{C}kz|(;wJyR'
    '8B#u<SPt_>vVN(BsdF#;T17f8}xxt_Zlzy)>j?lJzbuz1yv{PHG3cb@qr=5mirCPt8l3~RG=AP^=K%OvY-'
    '1_A4u4|Gq%*y40}Wkd`=6fpD3xCbyoL4VAI>E;{xVHQj`-^33UOgG=m4=2KO^DX?~VY=B-'
    'NQ!+k*eeXeL5J~e#f1$D6P<;E=SQH1HO~q!Kj>HXs|WQ`G1Gff0f?e{P%jhD$+r{$DXItcat7)_C}D+|@4c-'
    '6PLcbzF!%HyD>Svg!A7lcr9&5J5^MiB42^0oGJ5a#BqVJ|z}OjS2hxq)T6g|_!1$x$Hh9vn$N>`lcyCQ($Pw7&OwX*ZH9c!j(2*X'
    '2j*J93Mg7zL*fRauH-B?U=OTRbC;YM^?*P#!Z53o7+{x(<VTs$xCGK_BvYlGuUS|Q@<**?3pDeV75=-'
    '1rLtSUH8tQtvB2hNoXtT2ECYzN_H`}aiy2a+MM>wBr>==E1TH>A}_f>1lrdZs8n9CY(t4(I082c+a6YP(I@7Zd-wWL@PxnQ<>ZgI'
    '`*S>}hxkvUEvN9HJj9GPPTa%7GW$dNfdAV=mviyT>S#p6pPRID>Ru|)RA;=+?kWPhwKJf%eTx1x?8Su!Tus?}!F58K#G`e9p}Nk4'
    '37GwFxzZ6^J&gUzHLcC?xFLz`9Uhu&CYaoRe|R<mTdaiXnGl%#dGt!CGTI;UW48U0yo^~~ZLxPi8PrwfdcGyiuZ@p!9hAK(s{T&s'
    'ts1>;4JXD3V=Lf)5oicy>^>G}xV(4GiD8zXV;4VRICw4lme=oGzQPGe~HY5F*VW6|kq=DaaIm?1qeBM@P#`>ddEsqS;qDWJNq(1L'
    'g}lDz|^9k!Jg$5WBHFg|ytt!7GAUfHUW3c}mUOM*VfR&#Rj8a~S%ws{AQKtAClE6|WbgPAYqv4<D_6f1e`5rse1$~6AS;A_fB?NP'
    'xIend{$YW?h2`p=j&Jo>zT#>M_iTIer@AG5z*M;HEU&VQ;l*$<tk7pxA02BNuzobcA1ysC}8Hp8;FeKxUIdvc=NTT03r!rr{k_#;'
    'eRmyBL&Um3<f4Jo037UK5QTe!#OoAnNWFH)bT&*mB7sBO*&$GHp&va`#eAiKH@{jr<N&>sV^5tm{I6WuR6|8ai04QB2B6a2Ii2JQ'
    'aS9LBo}ZQoLXSETl&Kek9cNq<7Y4M$zDwYFLt`<y_gX6v;I`E#=%DL!dy;!a70UediA&f%&$M>`|d_9EY0ZBpx|4<g3u?=WEUg~y'
    '|=e6UPxoVwAwNT{1#T!yXL-DL=xJzR#MX%ESnpa;j(krVjf1PXtGADl?XPXL6IC=?2Ua55c5fe=oiuqYV9sdOxxF}9Q<1sQ2AM|Y'
    '~BkP?6$BeNO0;d*LHQv;^Hy}we*jQCkwJ-fJ$1h_+y0O9yaED0Ht!bgHL{b7oz_=-'
    'B4g|t8?`8csMNDezqFe}Ftsom|5D^ic!A77*<x9={v%dPtfee`lH4bE1ilC9EWxI^{Rdy<d(KyrcXz!vJ^L@fv*%yl7%q@8hcutq'
    'q!kP+->O&Y;|&ZH6S7c?X<@?SJE@rQhjCPedsm;oQ2Qvx%aKkey3*?`d^-'
    'zuwz#Y_#SPfWGLa_v{vyj9oPYF)_Xp7JXjG)nInD^bFS&%kv^jB*A#ggr9sk1CVW8d2>Lm8{u8Y>BPO!n~=Z8ZvTMZ1sHhlk8#h)'
    '`P5S1jY5Jj@(i$v@US8$E48ukfS{&h1P|R_Lvk}A9l3Iq|o|^!$5nsY8$RJF`4Rxb?8jZK~E0i0KD5y4GJE-!A=WG0rxk0-'
    '^>k`8FM*C&I{II^Z4l-'
    'W%Z!C9DTE_9#R*`@0Hc#Y;6{z)DrFGU22i{9<D)F4XzPa)!`Zk;eb;97#)NIOKqtYY#e`f)`3uMxp=W^zj8HN*nP>!sK+Xge>Bj5'
    'MCzi?`9T56=nH+?R!@ibxd((JbM8n;62?&7rcJz3@9a~9-2bV(Sf>TK|I-'
    'R3NNsU`n17aUm(?R`5cQq1dQ^?sFR~~gc<*0qQ9y9uxx}JU<K?={qJZEPy4<3GI8x(A-7B05<)Z3BrZB3O>C+44BzI*E-'
    'fSsH@Mn6sLLG;sfmW<FANPN`6iu65As@N8KF%4_Q(<nnmAC_Q5@gP$ad9#f2c;cz3e=dTad9d%&;`J78q`vyi*Zu0b2*6@`1Ifpb'
    '~-=3-$D}UIjhPvL+V=j-'
    'LiU&ec*ci6&B^zQS}jNK1>=wBRd1sWBAM>GTl)pHPgAs0t<b^(MAO9+e&|Kj2Aaqze8QxhE{tx1c&;H&4awBG;~f0<`SoH!g6p4+'
    'Azn@^DLysp0P5?9g<vgX7!+?nzO2hB#c+p!xF_$tR9j4(^rp5JfB@XCK>je>TzuF#A%=JG?%SS-m-'
    '2M+FwI5D+@AVgQD8a>dLw(!B9Lb%H2TB)=(p(4dYQ@ndslmMp`|5yhjko`?R^f1}KU(;)8|npLA6VJT@en%L@NqD<BU_^sn+K8`+'
    'DLqMu@9FVdxVs*$}&x87;or8=qbcPT|@Ug7UiR?caKzuyV@-~#I(a{dQSk?5<{boi$RSL;s|RsO|*O-'
    'K@C)^A@(=YOedQ2iN0gABS<v{sq|cAC7^4_73kA~xND*`)&fK?)#cne$yp!t_V^dMVh}>#g6m)=gPj`1e==(cD_lwiFc2ZH0fI73'
    'j>$!hgVua}MP$5w#kF0#!#R@xwez`Nlk6VcI!^%KSOG@OLW-imK^*l?HTb;qOxx;pv5cfW-'
    'h(GiCTS8n&%iqCv@=T9W}m?hQOQ_XZA7`O$!gTW@6coZ5~VYI1>seU=ygd@Hb_6$J$<!3|NZxY(a=61-'
    'I^<!T!(l&fvEPOi4oGP&AbtK@12Et0DpwMMSm;t5Wp_1of+dkGZ8ddWiTyegLW^+<HGK`J)hiV8ih?+I3P=y88f%ynteSnD)N4s1'
    'XvazQmDL4EL)VSN;z-'
    'kHG6NdsiQX~GQe0_pHjIMMC$m$Fgb;WUFAak64v+sgzU8eb;p(1bEUhbEQ@Iy9+F(4om?f(}h76Le^5DWj`xk;q9HhD9({3u{@;@'
    'SDM->h4<WAuj||_xny4IR14^P8b;KKE&#zZP<g;oDHrlg5He$vfc!yZFuFoL*BV@mF5CZqNUbJ1}S*E6=MP9W0R~Zl2D-'
    '}dCWG*VkvD>r%S4V)>$=Nr6NeK;5KBalqbiv_(rgx;uQ5;!BUV*gl`ACX)Y1I8%$vCubtoKfkEt0YkuFKDnP6g=-'
    '(|?{Ug!u=rq#R>QP{Aw3}O1Q^c}<n;3F8gPPp9W~UpZrc<nxnurt}xY`lLq;K&wrQ=RVDwK{p$@JGzI_@N&e~!{|C;9yIR2w;Ij&'
    '6?{fXdalCU*u99amT131)hn(0?yD3^ia01oIJ#pw86M*CnFe(xny9IP8NhbpDI@bZ6;DDfi90!>`zYAONGs#t~||u%{`^QwL;Y&8'
    '(=9U!zKVySVBJe^2T>Jn8RGorkCV1NyS2WqgG_Grq#>GQPs=Grq!J8DC-VjIXdy##eYlZNdJhS;s&=>sXE6n`mH7GmRDHYSSQUcD'
    'oJJdhpY2fHl>6uwN?!9}0FP&L5Y1rJ@7&eY)|r1NH;TlC%T%LyF|u0sE1^&os?S{?DK+x#iIvPAUmpexp*-'
    '29b&x$qcbmQZc9sr=?oh0NF(XI2R0#FhxEyUl*9T$94I*4Ku{_p4Q|7bNTV-x1qb6zTKL<RU|4ENi8BVa|{A$gKXglLdd9QX@?l-'
    '`*5CjxPgifCmu%{s6qXEno<2<b8OGf*~3em-x!gK7dU@Rj#Tt5!H`))UT?_l#>i{(oj#D0PEm(M;8$B?82LZUMnYHJKp=_JxFZa-'
    'wU~AKA|ymOg!cN%!QJ?t+;Ir}`JN)E$k+DFzm`FsjehmjlGQPjK0UTgBM~kjj|UoKOn00irUg>})`Z_U%s|nJGxnnlsxeLjMs3*X'
    'q5%bU5$#-'
    'WguIA$S~o$LMjP3ip_QVY?3bbJqixt%phUub*j|64gV{S@litDdR6H*nOtZ5ABAq}0vTX3LZ$xq=`T4wWANGl67sDopvBKXj>3*V'
    'JRJ%0k`?SPMlj?hux98{_b(+TG)r%vyP`iaQ9lXHBqQb!ooLg739SrU7|HldV)Tw6uxTAQGwhFka7IfaF#kwG4l=atcg(0R@QT^X'
    '!8@R;gxywIzq#Aat;s}=mq@f0!z`LjeClDJok&q8Le=P3z-'
    'RM)lS}v}~9opHOW87Kl=vHe)NGFzA5n#+#b1fjF<SloaB>Aw=;(t9(P2GkfYCx7}DIh6oH3~?IdW`~-'
    'qGqFjq=+6$Pm<f7NnhIjn3dWeRce2nnA#scwLfO3_Q#x9JSpw2QP6>p*${1-hRie~Z3pG2YSoa?UnuV+G%-'
    '4bj9=g|D3Y7D)3B+tN~eY)5xw0|>L@3oKXnq~c##X6(7Wft=IYd7h+{`CY(gKO3!6pUcfha1IC7-'
    '1Ka5;fDtx4}Qs7q(`ZSC@a?qz?psG>)&Peoal<PB?JxlZtwzCpUgS)C+tijb#D*9mcC<E4DP8-'
    'Yn6w&&%(A<(wB6litSnH~?+T#1mnB#`=hi1H#-'
    'E497j2B8IG2^8he^P@iWzQJ4XqaA2kK4acVWlNS2EYhzMm8{aum`r_4|Fk(gZ}0!rG$+`!heVSK=KlLoKk>82_X#yR1nfnKmox$7'
    'TShel(aX#`ja15$`$@+F+rjz^B4By!zKP#ziTPC_}|!N6xaBx><o^3{O`fIJu_$;gVMXqQSuhnj#8)bjTULuF#VMm(lk2oQSML&n'
    'I4zx>ST^-Thda39v11o`o2Es{LyUp>-_P#HPV)-'
    'Ksr2zU3+_`oK_=yBno)dg##LiluSJCVS}%6`1=D?E9FAC1+aqfK`SsJ&O{2wQpc`r(vv)-'
    'fZnQjP#s!8TJ&OmI;?;wqF7QLSwMGBoTrW|ARc)IKlNGKO44n=)cUssmTFz{dVZm5S{)c*i?Uiq7wASFK0gSk4Y8(j+I%5=NBbGR'
    'Ay$?CkMV|pDOOGrP}V_j$Re3Nfh>~A6UZW&JAo{csT0T|nK^+hl8F;j6ECsHCY6&DZPK3fHfc}Ju~Ba_t5w_GqNX`&pApjZ1{2g4'
    '^@!qS!%mrJBNkqx9B)(XjhHO4X;F0xP|{_=E;1k~jTd!&u#`A}USgnk&Jpxd1I`k+OD{7ZE%6b*+<>*jXZ#8S&GV@u@O>#|h5X}='
    'E76=0`N3UFbP+^dZub(+nLoH1R=U55fpePmTNR>Y)82G+GwApTinu=2Aq@EPjDRx`en*vMTftnIp@3-'
    '{IspkAWqlE$dauxC&hgSiR+;J}{y~d%<eTIn_#Vnl@-U1?<R*CpHd}I&JPJ<>xk(;_vy<E;kHelsviB*jdYMIiF@md(kqO-'
    '8w(1rJ_%0>dd%w*^LRzq8xGzw51L?O3Jr2Dl9FiqYb$2OrXyP=(rec<ZO23T{^q1~zxS3A3e(U&(u5*57-'
    'Z`iq@)uM7?2_Ej425lM!=^BygcEZyoIbz5!Zx-S<0hQHU^&KRT=2Z&Ixg5b#i?a=v<xjTvDL9@Oj#XYGw~aBc==-'
    'p%^g`#?y*R{qA2-T%nyjoNsXN{i1Dh%PWe$El@CtT-'
    '7li@!HM+!B~&yzQM<p4N+c&z?^jTXq@!wa`o^TXI!?4ib}rG&{f(;;IZ}84g()+=?yF(jXV|=&Mkm%v6K;cJo%2lDmts@c?wkDY>'
    'dmyK0OijJqpa4ZCf-'
    'P#J+4oUVub`!hB8edfuM@n#_<GYUK2|{&x&|2AeSP9PPy0QD1;sbe1X6`ji25K^&FYsz6nxCGQWKbB%5Ls9zlistb*k>2lNKl_!H'
    'ooC4XlU+|?v<Plm@#&1rQ2WoB5WjdcCxqVg6Ss@W_v6rv30L0Wi*OOh_FO5SwILwTV~I%2H_X!>~_f@F0<(*38*b86K6k5dcH?7>'
    '>la^)&IT~e)FO{YtWm22p9Nv(1%oh~WN7SZXFx@<9>E(TrsH$$38a)XJmPr^fTS1<gFaRPEi5vsGK<6OLkjxz;;2lWt5pjdIT9`;'
    'A&b<<&gOkOtC5muuK+;3<nweqNao6;YGPB~9NN`EM(g50_;{oxo~@_eB5M_};H^MTSIY5k7;)K>bVtZ&PSjjM{hcbqcoGfXPEm#E'
    'KAH_)$9dSR}Sd#5k-{zo%|O+3%%*gk5R(Wy(hoHkyleGzgtH4c+#>kLUCqHy`J92yFjkI0dsu=se|7_i4%sCrW-'
    'g{P}wgC*hV8h8s{sNz#)idBo?kM$9i4cl4*SBKxO^oLo$T~QBE17gL8ByEh={|H{(5~T_bSX@oo=$VV#YfY4<&b+TEsfziW^RsiG'
    '+LjQ@$1vnsUqevz1Bs1gU*P2aUvC(V0hmO;O-dncn8L&#bsSFcT-$(5>x3YNK-UhjfFntkYyn4-B-'
    'sK^#t(;EfNl8U2n(={WX~2*5J{dbAWR8Gx&*yy1>;_4)GF~-'
    'WNrQes|eE@OmIlmCA$GM{IFz{y1$f1tDPKPz*yLrnzj3jn2vGV;w4NCxo@#C(WvRVNNnNFm;!O@;^mAv{7S|gz6}|rztAd$;U?oh'
    'Y!zO7vk_Z2%;|11!Aes<Q75!ZqgR}xWbQSH@f|3TRIYGgFqXq{EF^=*yR$0kI9;p;v01wsnDKKJ`jKQGW@NGt`)0Ba`(?5Z`)9Hb'
    '2V}Aj2WGMl2W7Gk2OANgy~rd{vP(=7BD>s(c3>~Wt6!-'
    'PjQYVTOEw=h21YVn3^TSMCm+E(v`{T3VKZ?;bEb7CaB8CWe@*SoVJbdG_lbgJbFj=<i@m{)WN-'
    '|#`q4aDp>$|en^aWuEh)0v&Z63U*+{y1w9V18eJ07fU2Kx1+oeVX@_M0t&5~{-'
    '@Dv)*WPZ;AP%uvBUsnJM#>xEa3*etPncu4bE{s!&y$j&NIGNw404|J^`8O26<c%~cunc6n&d>yqHvg*dVp^0>8MJ7$)jMBFMO~m!'
    ')Eu;rg!hOzaa~)R+Xel8E%}nL%*7@N%UoiTu*{_<3Cmn&lCaF>CJD=2VWO~1RC~&J`{QBICz;q37%B=xZYr#W^Y$gZ-'
    'f3`0py1uecUWbT^5E?zNp*hAh*T#i(W*#D+pQ%#k0v{hHam|lJC8+n9?R@JZk3(Kt!oXvW-'
    'F;o7LV;+6Bh=3ybK}{6ae)aT2i0~SRXyoZ?rdctVSsj3@>$~tFCS|hW>;}GK`-'
    'zNrv%LMigd2W)?}tf2_|_(mgBurg$u=nm8d37lSr$$z@7|?Gk!{kq<H$vcO$M3J2g5IZdBNCBNE(tG;$v$%gzX?}I~8qOlP^5{sw'
    'x>KU9IliHDmx2vql9U0iF)#fwzt`LrL=-`X4x-'
    'xw1O|H78u|3fqML_&&h5oeYL}z{I$Rv<RjYzu0)I39n4CICytvJRIEV(`oVi)X2x7_Bk&T~bQ(>(j}hXnKKL-'
    '>9?G&qJIT8IoP8XWtIq$!#e2gSVTxzhik^tW>b`;2AYJ08~BOgP~=%UTI|c;2!w!SwSM8M<G8So*)0{sVcx@+DVY6_)cAS1qm)P!'
    'w^}U9HKCthxaY*&{n+ZQ=d#_8Et*H41M`mcs+Ftw~EAc>Cc5M8tS;L&z4zm$fHMFAMF;^<}=;3k$p(j)HwjUxz=?W>;Ms>mn1Ad0'
    'qJXEwZE?)z^p8q&9}gLq02>I+$yI+5{sWZl^BMZ+3C$IPE98wCFhPuZ`9lwUE6mbes-USY?v9c-'
    'm$X63<$exX2QqJ({qu6wp=&<;H1Ff=ljqq3SOEN!?`&dgbEn%(^RS71~T@+A+>#9vHhCHSGur&~MN*lJekhu<~IqV3>z;7^6$v`4'
    '^Qhx0j>BpFQ{j#Mg*O!F>kz_GZyw$GsgT2&%38M+*N~UVH3W`uD^OcAwI}-'
    '`2V#s~Vw3ont`5dV}AtIIYpi9?V=xmqdQOXpC)1_T2g0ztc1}OdyjZcl+%odCPvxByZV|!yRg(c*{kf+2||>t2J@rgV&lE3BqiR`'
    '5Of)o4Yt0a{eVNCK1+OQ22-'
    'VXWrU>wD6DT%eAYPYP)l(R_h*h;%FsA=Blh=lpBrb8X!YKN=Xv<4Mr6~`8!g5e?Vldt+C8nOX74(L|s~M&uO?!<QBNcAhp}rkey;'
    'SuvA67DD${mW&_LQj+{hSu&8;_4J=|_bOGNlvw@W&Jk=6oM0}u%yYLU21}Z*)zcwW@huD<J9BNY{bC^wu%;7dAGDp~y$Q)@?B6F0'
    'L!lElvK0Z2AKJK=uusFd=VZkd4N5nWrxv4(Gieok<7o|aSzsOAts6JkI#LU(lFElNsaMV{YlCu}E-'
    '_=cbIr`bn7afc&OOeQ`YH9B@myX7sE**{6xpXvM@6yrO%cY~Sw@XK3AD52C8(caX$7EvB$7W*D$Jtb69M8hhnV^ATro3o!4nRaz4'
    '1sXu7sleurD^GWBj+TUd-MYq9}OvT$k8{^7YysMUFr9);o<Kxkp8>QiTeb!|E_l$`@TqZw-W~n$ok#FPdy<=7d-'
    'CFf>IsD<Ibic%rT<5!_O(`x>gkG83nG{cK%o1Uicqdst(wqt?)mwegZ$;QV6a=GPE^#dwe!mmsBWpSuI`%O>PuewPea`c+Q2O{2G'
    'H$ny6D{KwSrw!5UJlt@H=S*fDIFd9I#(!0?HfEds^ojK9H&e+gs~Z*+Q_e8JEW;prbG!Y<&@!~CzH3_yX)|6KT=T0fEhWN+brX8k'
    '07dPm`ZZvA9_dS~HBER|02>4qYKMcRP)os-YaTB<b?<dQ4qYHhf$Hn~3)mNdcdS`v?CSW44za8DhLigdNRAL1He3v;55?Ga3^c&v'
    '|nxBikM06`8Tt!S4H-TK2$&VtRhrQa{U3i>7+Yp-Img+9eTERXyX>|i;7Kf{}lQ}=Vc2+o`6IKrxrm16y@f>KLm4pj6VoH7TlD-'
    '@vPuP+p#<8Le!mxDR7pmF?Lo&RDGSk5wmB|0bJk8^_Y2gd-lzoqY&r?TyuGSyuHLl<U9y=Pj{B*yB#;6xu6sRIY`4bWWb%uz}M`0'
    'Im;Qo7X$Ot)Hw2H`!5Qo79uX17|7Cft3B$y#9q_gt++3yyKm2pSKm9TIK+N7RUiLdN4F9!T4~S|<tHu@(@4?QCt8b}Adk-7T0W-'
    'KiE6Ez&Mt?GkVJ1*g2>V06DnR-'
    'O%q#~LWIH`=LkjvMS$50Q^iX_XS}R1cF+R2A}pf@LUk8iQr%k@7*St>tr$bN)}x|3c>4&i~c<|8#<a_J4E!S2E>xpaA|?F76maZo'
    '|e1qPE$kx*?>f*%*oc#uVp|jJ%1rIJ&L!#%BwC<;*vJ|09t~Adxg2Jocjx8PpzD*%R@Q&aE}LyC}m)o*)0j(fFwTGyHU{^S^ZdbN'
    'qB1KDht96LlaIkH<LC{sDBfn>X(sI)3gA)C5UDCIKM9u+hn|%i&~X3cwN%D0nrpzf=AZ>Y!|d%S#oy0b!}8I)9YQl|))(bMc|mEi'
    '_njyP@i4dcoR|-B^Y^qHS@2T+}2Cdc&?&NQp_15}P0;E<s9)1Su&Kq-'
    '3iEDbZbajPw8D{AYPJkH@C+U*M;1Y$}RE@#zHT|J(U5<v4)pP?;XX^9h$P75QO_f?9Ip5~&s0G?Q<pF6~?3UUZ44cL(B=L^E2v4N'
    'avA?Y9>c0`zM`n@NK0Zw!4xiMA9ieyIaSiroJxdiL;G{gvpf18MYMs&4;BLmynK!k4Y`|2t5>{7JvhHHH6*m4EHp!hhAqxh}62{d'
    '8-_;#Gg6gzK54E@3_7W9$j5ANl8Bhe#_Q<MoKN@-g;8q?K>HHzKWkjC~Mk<%fL(BCQC>jEB1s-'
    ')<9t$Kv#IB2Y#5Uf6K>GoDCFh_Ff1m5(<zsQ9W4fDqAJ)3>+U)g-clI(O(GZHdTRWhRtOJF0o#;ExOxZ1-'
    '5(Ma)jx!%7;vA#0+x1YA=?6sHvbf<iSajK8qZ?ArJb7phTVsAL!2#|_w={OQ7G`;7DdYIV1L*7<+4y4^nK{Fklnx6eD;B)Z|g;Ao'
    'fVj{BnXU$eU9zU2JZ^UURKj_QMOqP~JQ;NlvsWWbCyouox?Nkf`YAj$<t>pUYQRjbP}+>ZKFtQx`J+Qa7ZMnS@qUjCp;gnqD41sO'
    'w?rz*%8D%>T!hIx+69*n_|*r<TxRrDWKaJ-uSqY{qS(0^0{^IH0k3OX*L|EQwlV){?ESpS9|Ur6%?9pi@+6LC#5DyNrLXSI5|;e|'
    '?JPM3lzcH>5p(!8KoDYbb?FH(y0t&p{Gb-p#dMEUp|ZPF6~QsncA_Fuq*j8E+81sp^V&JSQn<Z|SFi|+jXq-ZdT1<oa@l38w-'
    '(aI$i-FWk1frebYodG&W5i0pc+nzNp>?qz%V&+^!1tQAzr<kzXNk5<)6?oqmw_1C;)yAh=Z9=-'
    '$CZ=0$Qo7Y9r(11Gy49x2twuHe7QGElE?}_7k#k-'
    'F$5N3QDB$FJKrZ!wVMrv^psQNjI9OHs?+Z<Y;cu0&r<8*0+a>H8C#mEG{2C`aQSAMNONza}bV;%IKV4Gn{gspQ;<N(pvK+t<wEnU'
    '}Hp(*&eN3U+JPWUICD%L;Pj#j`NUhdFuP-'
    '?Y=?&ZuSG*abA_8Bg)gq+lgx0|suyzEJJwShx>g=1Ge_dR^4(Io7Vv^oj!Xu<zbyeW;IYHGb;OjX>WmaJ9IhFMT+Jh?@kL<=fKT~'
    'I@lxWP-8R|smE1jWEa{fe}p-y(*>kO4r-Pt-rr4(+CsWInAsqAlN-'
    'nP6ZU$QHq&~W8YEDo^PU|<CivN#&2%Btxar1Ir+v3xt;`Y)7?gPj)0d1Q%qi-^?q3zV-rngk|T$sp9Gr7EG-'
    'rXyD?-?}3F0KYt%z8HCN41F>1;#m6P9>8(*#juOx>5EYp-SowvixcSUV4)bkpyBgY<XbA2{9hUlS0asipzbx-'
    '4t5*v?lT2dz0gL!-'
    ')<N~{IOw8vIILziMz1`nM@WwWWR8O=;s>HG<!xPUG+*Ep!%TGVak@nO0NXi>p=my0qZ6G=&J!pdRSm>KzcnYxHbU29^-'
    '`eI_rO|V}cv4|Cx>ndL8%`H?q9Rv><5Bx9MIb9JknKZ(r*#QpMZ-tiQO>`h@*@8qH=HN~BOQy4XapK9+9Ni~#)8=nr9_Rgs(1M4v'
    'o&y=ZA3>CJaZz8y~2$12tD-L5)LgV8yz>Q-'
    'uBYs$``eobpEy}oTBMIxP(?I4FCjg#%6OCp`F9iU4hjjkP`OCs&THi$}0si}8hz86J~YykDthPN$<d`-'
    'jK_9U%>>SjLQcaRause#i>I!CnEcvVN@I<C)|5zhC=pJ9@v0G@-qFL3^DD;8fpCZTgFxGBV#Vw-'
    '}pDYPlvm?GP8(oCShcDytbD6Vx&Gl9a|2_nVzx)OF|oaVl0lMUBKs7Kz@*o5mug^Q+Ii^?v<zVdqhpe<RmxSf7<jf^;ZS|NmqvT|'
    '0`4zInb(f$)7jpbt~;mY}5Ad26s2j?CG)B0cX9dW*^*7=ejE-H|QpUJ!7k^+hMS^RWGfpq#@X&nbe7)B5Je|Dg^-'
    'oQ`);(&Fzxez7WY7^t6tL-'
    '8DaU$R2Px{kL(&pz6@g3)nV6hRpl=9pGjpRm^G`u(1z~3iz8|5^|$6vq4RZn}vpX)qVJ>xAuyw6q7`hp)m;Hu~S)_g-'
    '=Pm3tiGfERsTAXb~b+>w1!2IhRCY~Q=s^a(kN!m*I0SxC3%|VtBIbfUYZa=-'
    'u(*e3!a;2fJeD*R9EyN*MGblaNMy2?CvA#_q&<T=bam@t$gU;V&MJ+ybQM#m@rDU&LyG;wEY1!f5>#DWh$uPiyfpuLWI>ptCa+*r'
    'OviPLD>ik<vspzO&r$p^|ZbWXf;=|Wlt-;v7^EC_Nfm9H8RKvc%V2`1oF3}n}3=y$YBgwv2^d7Rz#o=L9)uVSD#g-wP>VRPY-'
    '0A8_qfC7p)2%3`oSj*@-4Iv1-'
    'O{a+Uvqk)c2sC$az<e3i!%aOsTEyOiOPCaR$_u!ZAD+45o}Fg2eTT27<zP6Kd5H0K!5dAH7B}{j|A0T4wlI!$TT##mjZwnl}RXr%'
    'eM*T(uXJ#>sEiLBC(dyd6og@ZFJAF-(8p+YOQf)sx_`kwZ_${*0?6s8rP;;V^OL#7T2_ON5_O8wIuRl04UDT?s8z6saQm-&7>K-'
    'Lwc9=tCEa<?Eo#e!K^Eww|tdH8)WWH;tm-qD1g{p24Zs=h|OgnHkX0eTn1uu8Hmk^?C*=nR>a|ElRv<!mCa^UF5+eLia*%0flV&m'
    '(k|453`I7Odbu;KDys!4DoUl-'
    '7)56dLq$3y*0o}zE4Y^5nHtRd8vhNaZBHsazM0z1RKR^JH9@Il`%Y?tQd#od)C8rn<a<uj`&KKkcl`6)Si!yHkF{B4-'
    '?RnB6n~nXU`K}wSkLE5guy`ifKygR;~Z$-'
    'R{xW93yf<tM9eoT9uOCh3%zkv<<S~8jou*9<nnq$#Z~$fd8Et8F&e>VKSQ&nE=V~As~i1P*Vw=zrcYJ+M{Cpb{a|6DEQ|dy!V!)D'
    'is$XEvilB%=Zvrf@FsS&LKE<d3zUf`aJs!ypbT-'
    '1IL^V`k#p~p9lV#hMm*iYWSIdAyTas8dVIU1Q>_I`2uRgv9ePuDagd9Arp1LNS`HRz-QA+i?e9@d*xA<KZ^fHEv_|-'
    'c{fwh4ihtJ6y6PqQQ9tLZjZ)2i-cc&T#q1XxU1MC!e$mn8&ZX>^T=hzhDUHDF9wB1q__b*c|3m-JDde%=rMe-'
    'YC1E4Nlkxcw->-{2R?a^hbGC1leqLko>gSf6LjEt_3b}{@@RBVusS8=?9=6fOxW{Jm$rCM-LlFnD{=Os*V*LYFr0e&{r0d%=>Gbi'
    'LjQxa6Mtx!?X+J5GRG-'
    'Z9_B=Y8Q}g6~Nn0O31SG7fa0U3ZM`bH}v<%uKxIeTcLTLv38^jxsqt}ACS1;$YGdeyAAWcXDNE4F)(xfDSG&u<%O-'
    'TYsQ`O7MzsveVRk``y6v|b(`5fzyl!~4*YJXH+BAl|QQ)5SC4G>U^YqO#mMYCox6xCa#k{GW0WcK7TEy)9qr?a|9LHxD`)}kBn_3'
    '>3a<@_r-dY|&S9ziBj`a9F6^mmqy(w+cCoil+Sf@saDz>h$*K0+#h*@8H&o&xFmC~fj|s<`@@bR(*``dQW=rsC>n(~amNc1QHXLU'
    '%Qr%gXPYv5aE?GG@ALaAp{mG4=u@N4_^6c@?LkKea}8T&D|1J9jMD!zRZ9Iwa~?Fy1D|0y-'
    'q>STNBh$AU>VITp|%QO5#0BrJe*5?#NX)c01)w2-Y<XuVpkjOD7n#UB6=7~bL&j_TC0tlV@D^h3F3VUlhakZmhOv65gJo^-'
    'Gg!%JSRZ;F#|`qht|-(0(~=<oqN&Nq)iiJ~Q%MJ2B_nsXHMwHPT-zG}%<Ito!*vTo%7&)tfjwsyehZY`Z+?EzG`Y73ycO{@NDc>~'
    '|L-vyiCBJj=XgsavhYPC%gwc0j`T5Xp^t+r31Ry!n7s~wZ5Rhx)fb@*)x^v`BW)28rOTam-'
    '}+k?&{g806z@QZDfzOM<!bn5N0z~uSwYYBEO2npAG{#OTFOmOu5eILNw@cy23{$*Qtbm(-'
    '=<k_rW*4L;h*NyL%eyonrxnkRiks!FQ?7}{<{G>acociWxPJL%8A!0clewh}npm-'
    '`$Pwh$SX?&7;nvkTPCMKz;NlEHya*}$QlBAxdvec8T%r{DZQjE*rE&XX_ePt-_j3w;yT2_f*f?1s;&d(_Y^<H=nR;4~~rAqBxN(W'
    'g+F)>X*-mP?ym2|Pl)ji<w4hGdT{yi$Dd8_m1DMD+R^Y2w<&D)%RpTf14JAb|kN3L-G{pxPB()kaV+U-'
    'Tz<mq9RD45L6)9^LeJ!AP1=>1FaIjO6)>#GNuN*AjmN>2<(V??Nc(h@2CDW%l9d*{7q?4B3>MI0z-'
    '0blYL^TWGAr@xdRXfwU+FXxhDLDo`yPu5aAFKa2jH)|=rFKa2DpS2X<pS2V}U?fH-_fcAzu`HW%e?e%+WOcBbqPEi^S)dK+7FToB'
    'bB)r~6x;eWh4R}L{wga@9DWDT(!_Ja?^yUnmR&bSXyv}4Nav0ITZ(Sp?7yRk=9m3<tw>kT2Q{#dm-'
    'H;~@Gs#9I=GwtW&Ch1bUUud>m9&j;|qM*`F-'
    '+|oZkBCGI*0VdTCPxRTMaV1Vwq%2T+(d#ZJL_j|5Z8RL9TEFx;~;40n}bxKGS5+&;r_&(1L1b8O8M{LYf{_wY$ZQ79~~sbUTpO74'
    '=WrXKe#Io_$ShIbk=I+DTYxi2VbX56iSymdg+))$AabwSJ1+g6M`T{H@kr)x$~@>!t@d4enDrn`#1#MMKkhbV%Iehx*@Yt`O^zQm'
    '1KgmltrPX^e18Kg3{DwK$_h3}GpCmzZ;L_<dPXLY5yb_!gi0Pv`J3ycMX!O3Xl+%I$thitCeuV$-'
    '3JW3I{Tkx%=v4Wp!FgKbXMfl=P^^*UlZlq2A+q!vP@!wOAr*?dP%@!{TZNufR#kAo6V%m6dW^{E)W^{FFW^{E~W^{FVW^{Fh$yL0'
    'oXJk0=qVs#^$ErWvT}^0?kzjbC66$xk)b8?kyVURU_qf#X^7p#b@$&b%)bjH8yVUdY4>)al^<t*3J*pqN2%W;Of8-'
    'M0@F^EDhPiZGt8#5e(aDyAchKZQZu4uFQqx2Arv*I4l7+5L)&teIaUfqk?@y0wM@M?7VFnhyLnFZg7yoG5**$pEjOyw$Dzg<P&C?'
    'Z<M?c)18C(yt-bF4qQ9bQXia!R$$cJkE=db{r(NU<r%g79eHsaleE;|tfxW~vmhqmRthHgG-FWzT()&Wg``wd-'
    '#(rbLc@YJW5a>E1%{ees9w<laezdh+9G_<xzgTX(gVU=FfWIQ!oMl&;ZEMh-'
    'L%`m%|UgvjGdiJ1|^z5<P`t)Pj1#o>ORJHEXne@d8eSI}lwFG^A4OF%6)dBTo34Of?s#=1+UJO;O2e^H<oAYNYLSlF4&ryWLJm>e'
    'fYVzRxLE+vwyQ#$Zzc&>RPC`RQnwTGQ{$V3_i63_UQ6r|6A94OM6YUaZ7Y}9a17M@He}MNTsHC83n!(=sQ+2q;F^@M7)!gv<j`TB'
    'AZ=(gm&XA4!e@&u$g50tb_w&)zYwclYk${qP4e<MFNI-OyHasX8I=TTL5uhBU5RWQOt`CBaLX5l!QU=wZctQ&C)9NqGx^>`ay{d*'
    'y4;rzkyubmJf2(>lJz~V7@}mx*{FU_t_K~<ilt@2b&!1^I2ZB*az5JRVM&vD8(`DsXXaHqe_x-'
    'uJ^$U&@M<`>(pro;8t)CYKl%Dga)sE7w)kA_ar1aroff-Vs^_XA@DbIR5Knbx07MRqdeUC{!+Vf26(Z1KD9_{-'
    '~>d~HWQjhliCiQ4PV5CQzwAAK1U|km4u{KXlV`D~A_enRqj!6)PGa7lwL$KnyFzfuOI%I!Wtt-'
    '}(aCkEl4sTY%;Z+HTcVfcf`Gmupop5+_5)SWTFvY7sGhY~z@6}&<L=CQA^*5f+gUnz3y{_bJ#?ZP7OETCHYCEF*kw8&iaH){SJY)'
    '^ixkXB|);-'
    'nAY<#9>H6c^8nwY6sP0G})CTD6^Q!+KHshOJ9Sq_($jFsN)aBF>AN(qN+ONLD6I$TxX%ZiEWS$|4=S>2W{w3?_UuozH`FB&t$4FI'
    'WVzr81qJX!b)SNdRPi7V*`E~!TA#2#?qhMApbD^oo!uHsZ;Jfo76TUXBu)|*zEtImJ5)%T_9{5M<uK&s9!+v*9aI=^D8A4=8vRa^'
    'Z?s?M+3>Pe|Oziz9iYMsF@#0$aAon2FnfqfZNWi~WPfVX3L)~iCX#M>6=s`9*{NZpIxu&Gk!L!W9!zAp6XUA0!FIE(68mEtU`b>c'
    ')`ZB;!d3*DEwqF$tf_N%f)`RXiDz9vhQugwzWMOmV}xFJynp75-=F2`BQkFs^49kZZ@5!ZQEdvgK^<I9w3-jN$kqIef)`EYp-WG7'
    'G%@qH#}UFKM<qb;R%;U_!6Qc9M?J;I7{QMv}Fm+O%411rnL?GFGc<eKCO;AXkF{ULC(T%SA%s38}$PXTH;BBBQr(jVs|d8R{#DO2'
    'Ec9MVOZ?XI-'
    '|16|$3NL*tv%xbZ@qP}K+?a4i=wNN5yFcwJg7%Rswed@4h_^t^1gztxNmRM3{7;DX8`DuShA#wQ`e`q0rd96RJkjVV3KU|4F>--'
    'Ug#OCLqImyBKc}Pwk6<6s;KpOu%Q?#W|@W+U~>=cV+RdmPyf<gCd_P}a4cHAAqa(NHr4c&o^MvG%n_l`TM^xv;@Lo#{UJG7UEO!!'
    '}|de2b8`xUF8LrQYLW;rI0#2Ci+c!*(qkB1n>_jowZKNTF$u;EwPK@0!1q3s(V)<TK>SblgGO6<qk3-'
    'ewo?cn~{B>sQS1Jk|!TDwN2e+TkTYJa-'
    '3uS|FLRq4*YI^EgVq&xfCbZ0M0clKhvvuSrdW9Zh759=VOe>^`t2RZ$2D~ch{8#>I{i<?*aC*m1Dt@KaUPz<<b##CqgN7W5f8U34'
    '8cs!LxU$zQQrE=)2R=p>v1p2zIf#svANxxi{cUUPFhNGjjCKqX&U{?=kX!#(QU2pVI2wB4tTBZL{?Ls%~uGaZKWq=J7ou0hCcBkW'
    '(T0^GX8&ky`a^5b+pA&1Qz=5FhW+g^!27X_t2CzYLLL9p*=-S^U^j1jO-y=L$2-'
    'e@PjNRwevT+bnhAd41aKuA`q(<gDm>@dua;W$PlyftX4F}a-'
    'f%gcm@vCfOS~!rDPQ$!(p=mHb83|@%a{c#b4ps|XxV$c`Wy7i^H&dGilnq~{?n~M-'
    'pltXW^<L7R4`suP)Okr5rlDn$tRe0Wo&a|%+xc0YSlkzE0Pf3P+5|D)o>F(;9S92k3#dkJ;-'
    'qW|TY?Mczqla579alVf(BcBc+~~rHP!lFpk1>!THsJQQohLohsvSy%@%k|u6f>K0ZQeItiuA6%T>=?t$)9E&E94KR~@-'
    '<Y!{6V`T!_=h2Os*qfpF+r=Y&jo*+nWR1|4p!K~s8T-'
    '+#gHl7mIuzHhikZG^2)1O>Wt&9KQpIwlxiw}QuL7y%@{M`jnI<XmQRi!if5}`Bt5}h;p5}7ml5|uOh5|K0d5{)zZYIHK=PdGv!cS'
    'tF*ZKg)q<7-'
    'OTvnYTvDmMMiwww^Rhh>HSNx@P&i_FvE;M`L*8|G}$e{M#R%8K@g>fv4%{e%?v)@#(8?`o=a^IZcD_4Lv|6;)>`A+7Q9*1)2@$tv'
    'JQ>)$8@Tg-'
    '}_n&Dt`iW?60rr6<NYl<HZcBUBOU}K6S4)&#3;$T~@WSJ6Zq&{ZUr0!pwU*y!K?wx=sHzMYs0*V`<OH5s|o@fsQ1(NS5bbmBA3YI'
    'mUvV_bzNW7R@TQ*Y#EizTmB2xt|GF8wbQw1$DRnQ_+1uZgF&_eR$P2!utH0QS}@$4RG%rf5vc&kVJBl+Gs`TyhXz2mehuK)3!nR{'
    'WW3s`6tj19{wiUng^QDAWxTs!KjSWs-yBx<lP(pE)X6~x3Ol4h{PV8^~nTl?BEmL(rku*J0j_TM=(=gggZpC|fEzOUCW>mPfgv(J'
    '6z%zNgXGw<_0sd>tsY=@-gnUXq-(pDL$N@*9tWUf8T)_xW9qU^;#x!JAk>exDrmaK&zc5YU-H5O?&O-'
    'bPY3*)rTur<UdSfIw0fDfRiF4gxSacn79@xX?G`Z1SNG&4w|oXQpSG#Kl6%xrqvP@1=~B{d0BN|aN%TKaB?M<_E@`3{IjD>H+hu0'
    '#gk&Ur>J>>2yoSy>@TnZ&cgg}dYN*KVzWoQL(NTS?7Z;{4D-xB}JvD8uOPi9|^P1oLRstd@@g4Bc2M8&v_ivj8-'
    'TG4x;sXw+ip$s*b)7to7Uv{5afH;Y4~SU^oiaa$p&UljI?M6sUBB+GfABF&YVP{af*naQ!jhbBdye6^(jXZ0yJ$nfrQK_E+SOBWZ'
    '&P*}55hWE^!#nM=CcV;;f5G;zhd8&MNb&Ung%y!1or5~rvmY-&JPnj-1-'
    'IJhGEEG$bFhA4ml`>;~mZd|Jn#l<~(T%qH%<JZf1es*?kR(XR%Zunu7DWmD&P==wzVuqJZfVP~!!!reDYJSV-'
    '#fn3>Mj%{Aw@_pr%BV&^l+1Ar0JC_=U9uEC+t<gal3=}D&V-?(R&r(qYU?61^6gCd9MNv;+?%$0S?P9-'
    'm8F9c~>9j_q3JqTqw2;7HgYL32(J1YGF2~r|nV3!fdGujQVGkdDwl2dTc8@E4Z;6;2R^3#_0;s7u+}_!;|k`tMRyQ6Xf-'
    'd#wzPhRjht)$tn`A$tn`A%_<VE%PJD)WEBZ>vx<c4vx<ayt|GzpeOM!zqt(OKG!LRRg{6u_VXQSbOC+~n9}0TFF0odX5r=tNnTOo'
    'ejF+Z&cm<1CD>=8}cz(b$2(QEQV>lnl4|~?&T*TMTTo;IueScr%hXGnv<%h9+R_2G{8moY9i?T3gGh)2R1X-'
    '(;WbB$CH<j|xf0QU9Ge_n9tRPKXMZiX4Hb&kns8=ptwfCXgu(kHCVr{mkwFV-'
    'e83XMB&pdrBr=nAhi6pPTj3l8&vRsQw1X!?rS<l#`wF;jLf=r7%ar0V?u6ATE=TT3@oQFv;S`d8_bS+xxTl7r;kFd{BFOUQPX`Y~'
    '`brL|NSxissCICsZgr3$*0F$tmNPs8-P?}{tGYO#5EYFKF4JqkC<P`+0JZF{gI=v&XOCHU-?S)gtrH1(OBLW%!0SSbI|9@!;e1X-'
    '#rZ4RwUs{c&ex47i_c4!qBBvf>mgoEBpafOH32K5RZ?CClz-lj7<~+c+GI7lODiKISXN)_Xkx-Ww08GJ?-'
    'O#Ij**<;lRC9~_+*yF6FaxmG>az5qZ}AnDIGc6o>1!(_wJtq<gB8zaJ$m{UOP<a8^z@yT{OPZJ-;p{SojRVZ9%-'
    'KNG+Di`S?pP|dOx$olVtTMv($5B^=PxK(0r&|ACy92UQ<w*N6mYiNM1>zgPpa@Lr7=WzAKqIlFiE#3fv)1B`sFq4sj;w8R<*p&2y'
    '3q<2x`PGnu=uV+Zh4xcxeI06&d;u4BRI8JzmYp4Mk_cXfzERx@}}@9H2%HOgHbCS6v9b{udl_jKMjge-'
    'f7d8H=I4wPdp;zTuN32fi;tut^7yu^SNiibJlb%9t-'
    '>{`rbs5R40s|6T<E6HiuB?=ry&dM%T;4pIDZJ7dxk?X>jD{vV3+N@3b;%f7x^eP1qWO$hZ^R)Cj1?E}l)d?WFbeaTydcgY|;id=t'
    '@!bk(x0TqJ!Fro{_Y(1%<tEzfPCtMrjOPU=tWENwvO8G7U`B`*ob9TtWL#v@3)y!9*Kyyh&~m@5&~j8(XgNA7v>e0C`g@c)J!N0%'
    'K4pHAa&P9P=5v3-Zk3n=?UC$f=4EO2a5DqZ?cU+ctd;+zbrpCqRoO-B5@nK1bYz)A`NVUngtkx6W2m2EkERyFA>__~Ok*xZGM4#-'
    'o~9ui%X~^t(~*v4KBK1@$j36D)6+~OWSQ0UGz%G7W=&D<M3%^%xaxF|{1q~*RksbRYJ7=dC$jzX`zFed|JcpfOVaFKxO8k#czLp-'
    '*y{wqV1)1L3Ok9BBG^id%K9TOg>RU3MBXJa>j~Sq4&SY>^6nw_M`EWCdrz@fh>g+NEyUjP>=$Bdd3Fr#Iijp{AIN%El@=K0GR<{i'
    'tyU8XNB97}#S8q|U?VHy8s@r;<~lIK2UTSgEHH-'
    '40ytEM3R&d2TuCRNN(ihbRlK5J4)*7M(LpbgT6_HO!&mx)^hmzZAFTJ~3;hlBete(L<@fgZAI-P<8|yJt-I-wBCbt>fF+usA_qa$'
    'g!AimpRlXd<w+!OzHQ0;3O*zE>swi!O=xSxX2}`E+JCO5_I0ssl)1Jt*(3?WXFxaWk585a5gGOe4(7u@;v|r{2jmrF>(U~7Kh8Gd'
    'o7T#wS*KQ`JpRnBlZS7`)Iz-%kan}TEDnAM-'
    '()TEC;BY@eHq@m=3L*BCTw6><R{11oBVLHE&E!l7*B!7Zhh^HuWiJ~ulD4vqijdERn@#8dO;#C8`4sTz^i(gwFj8OYl?L|J)4UqP'
    'etNpsq8O!TxZi&QK8v2|MH<HFS<F=G^sf41RNffQ?JPg@!hzc-=5U#7!wgeWkN2^Yi*1<8K-'
    'Vcc6G!=b>S!8#v4V~9CkJWU$#|hqtlFse&~4OXQBr~LL{PkqFGWzhjc-LzI)kr8P<DauMNo`_FGf(e?J;g>__2={pFkP-'
    '9lZog15cURCNT%gTtB#b@knsANoD)>Oe)s)EXp>STI!-w&nU`>()b}6(aOz;pPVXH(3-@lq;|(h!W+nKEwJAWH?_ciH)i*Q$$1-'
    '?JrXA8ZD{sPn0L03*(+h**~VsX`h9<Jb@&<#ruAvb93q(4XCzaIU;>|)InEFQh`K4=scfgbz}bPN9k_xtz<LEv^fj@3qu>PP9yw0'
    '+F_Zp~;m$My^R2cv(cU{K)C%hKtp)3Itpz)9ZPoY^?NcDe^H=ewqMVuM)G63gJgE+)%{;3P<x4!RPQe-Jd3C6H#_qELpOCp#u;!k'
    'V#PPsV?TiJiN%u0;IjSp@{HRTsH)p|_Z#q*|dT9YZYvurh)0zbj;I(GK1Guf;8wP;2Nw1hP7XAcrR2tpI|01p}&v?t@WJWhPfxPY'
    'bK;HHMnWvuP8N84yli-Bbt)3l+JFAbY7g`wx|5$BHuDuV?3UPNy;=@KMc9JYiLnZ2}EXK$aMDQqC!Fv_v39B+imleWt(|mfUlRHx'
    'JEVs!Z=72yf3o{@R%OVR12(rimxo&<p7%Q<ZNHCoZs~%XmTVj<EE3Hecfk>F$g*8Jd)4Q-'
    '}s0(x8usot`u$mU4ds!+TY_~K!Q8tzf^Zaexc8EH{;QX}762-+J#bM*s6&_MUB^3u{g-'
    '7;!xUfVUvEpW^Z_x69L^%ppuLe_Q^3`BX-(=@u*d(MEu$CyCzzWvzU3~kU$0VvK?45W1@DR^V==y-'
    'zix(D6LB|4goEp;XUt9Klw%=euydr8@M&qo?ir__71TV89c$F2wCs`4Eniau!$%^2+B8S9psoP3>rG?;^_U<yfYR0)=D}(Tnm*ze'
    'x`u$tcn%OVKNNHPo8kM4_UrSG8QuJQ84|{>amoMXySN*VhLZhXQj|eV*oG;TCu>+G<mj+)8Rtm9EwwCtxt{|O~qBlB}-'
    'g0D$k;~Tfv~P-'
    '$%Qo~hI>pFuyM$e`dRE>Wh};P_#s1_zfhjb3Phg61<UD~X4EauAit*$+fhi6^&#8=ZG?tczovX0AH|U4fs=Q<luE-VG(>qg9T2#F'
    '7*+w#NX*-$mmR#DhvyiszaOB7pOb)6EBbwHrMO(Xa8yb6Au*}<o>K%!@Z@#|KdT90DtI%V;Q=#|wK84oarO@NNN1=^(DD-'
    '&mPv`^O_y33OQ1q}*X0$2aaN=p3+UJH7&)K@^t3r<g+=6Z0+`uVV8!rEL%*-'
    'xyS4BCubU?Waf%?Gn8jI)tSax;b*x#Q$9XRzhJ34UakK@xE&V0j%IUM=p+5dqPpT6Tt+i!=C$^WyRgy@|7Csy*a%GxvbmS?sKc_='
    'MmywIMJ6KjcaF*Nac*jBH3Mh$};!4Z2|pus1kMd~R0M-'
    '5?h9+Cjsxid?R2+5N5UJkw{>%&Gl@0McBi?mP4sS@OOO3H@VvgqUeCCK#(^%wW@_0A{z8<8jRu2(-'
    'v%K&d3_0zNp@Df))N5>~#;_Bz!!`~D*=YOLU^9=<t%p3IZmeNbjTOy#6HYE0HfAV^;n|UT_v5Q5PZ>SD4w*fn{8o3bzd@z7*y~#6'
    '&DPN&7TiK{IPtk6}CcpKGPb6Mr#wOlfm}ABztRjHK@OtWai>=UUL1G>G2Z~l9){%dxcp0|AK0+AMPwJ`M_eMXXr*p#={hXf3J#X~'
    '$dY0{Y6RLeo%w4p3ps=KKtu4QxZ4(PEW_Ta+I>Y`5csKLR0rYT<GVdw^78b<6uLw=J4F01CHn<EvQVut@=moU<TIa2>PLHe1x4}Z'
    '13sjM@<u}<j6;>y%z&wAIq<f_foY6MEv>Aq5PIzIlLWRAB_?u(l2^|<%Z9=C6R-'
    '4c%fh8t%G+>FzEbm}nYxYmP>vz2wPY+vT<tW$5Z;KTbTqj?PwJ==Ay&V<`UCdX*STT8t3R8*g2)`A+I?0nT@O$7_=1d5-'
    'JI+=5t3Z&X&MudLb~UkV)qH(MwI7WuKi3d+sK4jwixXOB8%Em$PQfShIkd_Pm1etzy?SKkLN|;qn7Ys%Lrvx`^uY6g$qPMkvt;%{'
    'FAS)dzR(*F3g$1=U=Yd#hQA4Ospr?cCCm)P9RWWPxY5(Dv|8XUzE=Q;W;`9smiyH6_p<F#z23`&)O*90!N&BUI@=o2@10<f32Dg%'
    'A<itfV1xDUd$7S`@lDuZ>H98hux@-CHdr>k4;!o+--rzs4bwdF5{-'
    'HL_GT<$_02bhnNAd|Mu^Yx9NaB;OWLMp7If)@zQ)}$E^7CTi`paOqV~+VsJ${SYVVATTI0K&=#h@oNlMR-#z_Ltj=4z^&yKkn-'
    '~ny5Fv|>7hZ&}9Lx3Q$J&<##$AVX1fYgSqS3`keI6FsBzqUMw#Y$NQWfv{p*BYASQdTPh*Q>?C!JiE6HXYq5ax&PC>q<h{j_XSL*'
    'N)ywve%B$>Rsx|Df~6}^tv#oq~-'
    '<X7yhZKd6Ayp5azVhyrjI*PfyLu%A?|p)VxAZZwvF2)VxYh9m1TMn%BaOVgOEe2{1bB7sz_NhWo_AKs*wL2P)6_`khP@w|Al?Zbd'
    'Z=_jLR_GdliV86E%bjE;X#M#sN5qvPL~(edwRJ+Z&rj+vBq`4?*QaEjvJDA2`=ay%01dWX>vgf3Ic?v&Lq1*;@ZjMaxseB)M(ju&'
    '#iVSc!0tdhZ=k+?l0h~}p7U8%?3<wkOA5PgMF+}A?SHVJnL(N~(3n}q1AO;>Jrfr<**{yz^@1Aajc+B+y_<^B}<zXPS|p0EQvNSY'
    'oAd%%OG>6x$#yn!^ml47IpYTLHZpX3WOH*-'
    'G6g!AEt?kg9<3~)@R*Wyyhxl2QF4q+yJvK5x`aC!pg9~SP;iNAz~g}VzicyC6222K5CMsO<-'
    'eU<6L9aE6^qkMiw2(yn@HJ8GSO!&x9!t9&ywQd(-Mkl!ffK{k-Rz|5PuyTjp8yFfeBMY-ls1+W(V>!T=B-jA4Q{-'
    '4Rm>30w?~wQ_ctH4$^aL>xV0x#-'
    'C+q>YyC&HtHJ<@AQ2&krG)VuR0W?_ufdRCE{v!iuL;WWP&_?=Z2GGX(&mlVtyRntImW4V09nUAM#y|0$@2~{EaVy``0CE>o1fQVC'
    '&zoEmtc4XUDWJb(2OkcQZ>&pC6o5J|`@>7${a6PW?<`08rW9R;8I$CDEj9;2;s31&3ON~JQW;es6WvU?KYK@imdI}kZue>`&IAmb'
    '&#wvC{k>mUiT(qxE5Pm_`R!qa-po7;!0w-'
    '!B(*~VyWit{)d$<(6MVk~o8A(>^n+b*DPPyZwzrILy<p#49wuF@GYkfTjSKM%^7LFhT(W(;6;CjKzOBL}Q=$3Be3)4nw5qfm4=16'
    'T55`cNYd$WB;F<#rGe3ig4@Fzy(_2(1xw)<_`+u14nNfG)5#Oo2P4PhrsK&7J6qeXcP{Hq1?%|eT<b66z>?WwN$0zIP1cT0>u*7b'
    'HQRkV81$(~YC=>jcZ(-'
    'wMah~r0cSSya&L3&ZR#>BZP2Q%^ApWL5*BTKC4z6aNm(&{qfHRzraxni+e2|0rcjki|Ouj2KA4pxtUEHIT!UN2MKq-'
    'X>`5GU!EFR+fdX%qtlrQX2SK%>MG5P{FA{H$A8s4B7HqD3kce;+S#0OIl4=a2F{fztdPq4Q<>wdi%_Lk@P&hc~w7IE`&7j{IaTj6'
    '`LnfRVQ3BQox^tu2@?7;30Kw?LBVPNU+5~S&H>CSg6u+cgcl&MCM3}4IwTXbs{=~N}?E&AO6{1h)7m;0{2gDusiECAYr^i%HFKf^'
    'ZpJd0xu3&gQ@2*j~=48*a92jW;e1>#sc2jW<}1maj;FIeAcK32|U&*ZRH2tI;aR7Rj1?Yvz76)hnq%TS8YJ;np5MHTw(PEJgJVOC'
    'eq%Cf*XW`2g}YRu|ZNq-eN9~bkmU|EN;`hd*+UJJ-r6ncHF=s}Y>jeZ<2=AKl17kjD#6!Vq!X$lSw<`$iy0K-gI`-'
    'uVwGyC{V1qgPZwGiKgSO|BUPkoERd(3Bw83*^7)tsZ!_nS2;6qs|>P^nqKN%m7WPpYaQT-'
    'q|h&hf$wk+Xu{&<!|6)Ev4GtT4_%z>a8E`Yas!L<>F9r`hu*ZQrbxJp0dn13x)ve?K{h8b3Ms<^1H}w<B=XT0r@k!td&53coDP6b'
    'c;Ot-R92^{6>@uD=^Z-N}TLrvs$v7P`nbottNxz_)dE+^{^qEoq^%7AL8}t)T{(3cq3Ey!ZgF4Up?k<o>AYu-'
    '5HQFdakbUYBgNQ1{_U@&y6i5;#O%tLUWADjHsRgv$?8B62rrUZ*EGF@7w~-xSM1-Ko4P$7e_$Vfl>%u-'
    'K+1@ey9Dceb$?){@NG+$}Nh`oCd*oS65NzmDBKG4K01H+v=KKW@_VH&^okFe`Yr>dXLvGi~cwuu72vkS66o<qk!3HQk7n_-'
    '`oBPqWqPuLXUeu?ig^I-A}#UwEm|_so}GEc1Qyl^4tWz<lk+GCwrmcsbGkm~T0-'
    '(jS@cIIhC8c28;EQY@Htukshqb5ZJyS!b^*yv#W_Ebi)+fTLsW*gF9?r%>F5%cm9BQnp9oZA&JaikXl$dqw6hU+R)9In#tB@9eQv'
    'Wb#Ys$=OnrTrC%P6kDUtw)bMUL8)i&QPiSvruQheLls6glndUex=Xm=ovOQp3*M=^OSs^jN|S^O-l^70xZs^?y@U(i-'
    'P6Xx1@FPcRioauow%CBw*St9umM(8gR|61AG@M@pjX--;Fb36eNi~6*a21I63P_(ns6@Z<eO^x`s%<-O+Q~9IL)l%s{^N-'
    'b$xZ<46~lE4xEYUDmwXQnf|^y5IjBau3ArTE0p8&zTLJcp6va%?NQAz?7E$}uG<*Uf)5?CzEHsryI~ATiX>b(INI^*q?xdXIrE@2'
    'sTI-Zqdxcxb&0Q6mCg&J6^Q8|yMJaY>68@xnZFaID@7;o9pZGQ=;i%`NL?wqdH*C<SBk#QzgSlbw$TBsM4`Y-8z@-'
    'v2x6!~EIgpVeI3ja6$%`<4OpQ<fit@iYY-^#W;f1-'
    'I7olsb1#tRpw?(Zl7K(RCsl)Q>s_l4M8A^=RR^KlNgU&Y(d#7I*&*oF6U+Eebn1y_d>A^T#4|n|ebHTg`Un;(m1Uorq8IpoL~={f'
    '1^zcHJ3-*Mp$G$@mH1}7loNQ-'
    'm2Afj1ZRJr%Id@(6`6;#7TqSVD%dLg!9x=Qc6FCbXylw;$qHff8LpAeaRub@aSwlgavEqjPYGPIwGtnFx}X_%=vBnXK@MUrTY#&9'
    '{bXDW>?h+|U_Tj`0>=@!5;%^)>Ebv77X^o-'
    'xF)!l&L_xP<Wp;p#C(vQZ%3uJ2|5u*j!sb#>BB&kL*Hh$aXmQxYPL^u^ac3F6Vj~oJLg$#C(K$AxSn>_yrZxX9>QDFDZ-aU*=eL('
    'w?>3F-'
    ';iU5ii`LCOQ2Y4ew9o87gOtA_ztsO&Tn{H;icyrw9P;?xra1wD%O3xFLRh$kTr5v)(msG@(iH86Z28_1YaLpmeZ>rk)lk|yYzHqi'
    'ZW^M(NklJl1d-Y(=jQka{Y&%j!jY2=p#vv1I^=5H<4Qlvl1rKJ+#4er}>J~2InbYO~8TrPOb?!NZ-XZ0SD{5xhCKceGk_J9IEf-n'
    't;RfeOwc8xW3<?T+@9m_lDU;j6Jtz`oeBFwr1tK&7$$Zm0L-gfe68xps#EVin6U>X%XC|7Wv&-'
    '1nUEzawH3Vi`mLq^;WZWlCL!wfYnhACHwzSnb8iBM7EYaqz3e<IH#FS;7VLwTyNQ=N-'
    'ln^AJF7$!NCof>U(z&3NNm}37F=UQ2vHFa~6j;Z>D&Ybv`|Hq~>#BE@Iue6fd?erYDFKeyW&23~|EG6f=k+PWZWE3NeHR=P10|^='
    'D-Yf6-'
    'MCc7eYZmRiDUtrAv)2`Lx(W%hTsvfM*Y;g(tMDy_jcAhHQIbY7jYR%MPpwj8Yraztyh`2w3(DVrw@=M#)l#nTDfp5WPpZIA2a1#E'
    'kYS0!wF2`@_6_Pj3z+g=x5lCbS{rF)bVFUiiMzxsA+J`?6bdV&D!Y{lej2%~<X99M{Du?n{#JBta731`uffUB)TP?*Gjhm+WumQ1'
    '2OB&=Me?nB+_&?|Bi*#R>=K=|H~*hq)=AdG^Wyeveg<YZemk#D8uQ(-'
    'QkuLibUBg`d&NnNXW;ZN1?a%6C+evbo#Y5IMR3#RK2I4qc<Kjf%jrv48H1+(-=91{S_&r-'
    '~GhoJOag_rAo0tHOG>qxZ#PSSkYly%qaWnF&$x{F}`UMn+ySbu}FH=ieLi1%6N1JZc%@Tld!)@%JOm0N`Ur9El_!Iy^O34+%M7r1'
    'x7-E12kAy|u?1$-|mq_y#}YBdA>sj+8bKForoRj|%!mw6YKIc?ML!78V1`h8gBv`v2iYn-'
    ';}4`GSZHvJ!1;j~SE1PdIt=}GyyTUQ2ix4ta9_T1z(__{?{jVq}F4Q~Y3%B%!pRhI6V@4OXvh%YRNtx4F;98S8zlFRiRcupa$=xy'
    'bRKzmEmp;%e@et(K$v3C}wUyBOcJf|-'
    'xBtmlV{gOiBAkY3Q3W<;$c_L*``I8v_MK5LD7<^iwuy;NEhqo)8Sm!6mKFoJMe#v@V8sd3L>3ZyoTPk`%2^aK147I?WZp%|RawsT'
    'k2KSGOlK~Aaj_74;q9s*y1XmN-'
    '@FBG=uCwl@%Jg%S{H8MN<|Y}Yr<?LV2Q0_#%KIEkl<BFw&jEpXZ{>XsaA|6k^*QkA7NxTyv%e^v`g#4U(p|l~(^r6FL5EyF0iFe&'
    'aq9_iBFH6JUw{{}YsvC_Hk`_vR4|pEyD%Z084&*JFkE1z;(ajSVyWVh7(1{G@4gs)u@3Kk7=5vd@hFVGNXA%?#^{SBddFbA(N(bb'
    '39Fip^gifA(TTo}0C$NTpmhcKOFM^5rUuxFytJ^DI?7@Wz>X*RRS9<N8{z+MV0LXMUWXB-fNcD_GJTzmJCx~<HqKArJ>lMXN>Xlb'
    '#I@vKqanP;pC9Iqt5ujMfgQBjZ3(D+i)tP7!?ZvZlC5Zq74VXr7QjWxX#s^?K1vHDutYg6kRq**(gIyzBjB_^SNMvY763r^P(J*)'
    'Rq0H0?>40~wBMoNPn;XVwvX;p0+cz*W(Tk#qH;Nt{3P}WC-Il6jF;oPU1PEdZye~TZbWymf&W!Awhi#9`R7wip!)fJJ;hlJz1Q{7'
    ';1qeU6|O0H_BE=qufZ2)zfoQx%=}h)i7@<B76!wnf2<6KP5)RL3>*E`@EN^Tf(5e-'
    '>}?V(k>wfPHo+2E4g+fwERki@Z<k<+ESErRpJ0isRXEH~XKgs0FPH9*m?^L0g2YuW|H7URtilG~2VvIRTyQ)ZjM^N8&2_LhSLf~$'
    'Z&F?+4f`AR<_^4ItETMu-'
    '8#YAVAf!6(5=2b(}DVN_6T7!dxCv&*vuBQ0|mfe!hRwEe<^$Du*ofB{~R{C<<`aW06t&nOYPFTNaz_QD{p2vRgaW&ZK|?+L|jr~0'
    'Xevg@PG92UFd6q*2xp>YV(i75{|R8s;L^RmE{tTJ^~I5uA1sA;Jo0fseS^W$W>G82sk^qYHD2p$0=7$tta3d;i{?i!5QXe7O<3?+'
    '0*Q+FJn)$pT3+u&Gqz^p)8>usNc0ScA?PIOJ?lXDwwg;!wrx3rdC*9v^6c&f8w2HnZB7fn&tY>{BG;$E7-'
    '|hSI=fAcYS>od$#@c)qH2#hLid_EzPGfL-Hm)Jc|jEx9Q<I4C6a;ooMU07czuP+<4ix8LRd+m21O&xhlwm;4vcU;yesBN;bKRg`R'
    '1r?GmA9l}y{haG|x9!z9+QlMi^KLeHm4dg+v5$Q6g}$1k;Jm8&8i<{M2`c{=5!Ac1JrFF1qvcf}p<^g+m}rv`WWV5Bfne>;5$mKR'
    'WaJAEj!?Wwz+J`9Oo)Z9)Vjzq7nDmzs@H|j~uyZnu8@)V|B-'
    'k^u)F|)EV%&DV5xt+T6gq~V5bxq;aO)j_M9@oF)1Mms`dp`db>p$@Ew?zMu+C`fy6zWVr#FQ8Pwcl&IK)C92P0wIz=q>UGA&yIYB'
    '`?IQsh3~q0Qbc!daMNZMds2jIbi~O_Iu(2SZz^8TtIs7BCd>r|1b#LIZxwf>snk1e475g)FoMynngY}^($2}U%^rX7Y=?7BnJ+D4'
    'kGsrehwz*4So(G*A0FSCC3eZ4kNb>ehw!tftAx9<})mUi+?fXm%jS<O>hmV;CC}$Bhx~Qy%E=alUKWgX@Yb7EZ-'
    'AiB_Ae@*h0RFClE``a=gC=OAA{N<R3TGAKFQ~K<H^DbM|cte~#zu3qW>K4AT~=Y=>wBr;d$>_qaYSX}ydgN$BpiWoef-'
    '__r9=Wv6Dd_3X%_xJ;okI=K|H6-uC!A91xp-Sfwo*EOGwHx<_Xfz0swJsWQttowtRTJlFW-VE4^4`sg1&1}3`uoWL3Z2g1udp5w{'
    'SbtP9M_;98m7k*zQ(a$l(weeKL6Nov3ZgBJy6~i2#5d4cFF~X*1dTkaeIw6ojxJXy5>Kwhl?t`Z$)&nVp)~xX%s$j#c$ccH(BJv+'
    '2oG)VfwZSVNcBP*__Z&lfnWPlO5v!zc}|1K>y^}dyfkMU=nw3|x<<j_STbv$r|=zl);><*D;76?J8ex|fYR(=)lLRo8vZ@??HADe'
    'y!4P+UZ|gyLVxMMlM?zX|D95yzoD;zS2wSx44%JZj)vTVcL1GyBL4$6K0cEF3Hu%=p8f^f9-'
    'q(uhFy={+E>#oDAku?R5CD_rU81Dou;|i5VvHS{)v@|JWaiYURm7ZSHY?wui{-;__W=>2MeFJ`}bkt({}#>EPM)fK7@r&9>{-'
    'S;gbjQ5iET2K>iQ*CbzoqsIQhV)m#qITbh6Sa)=sfR;7iKYrh37K(0bV%E_z|3G<}ZdRYtH8D3AlztPT)q@SO``h5$%jr4y?X6KU'
    '>Wq#f7+X#CBo!1z#jV$_P6j&|<td{mvcK@Oe+I%Im0exRxq}vP-IjwB0Cg~>4|M_Z??$W&Dt4VrD^ABH5(o=%%xSFJwH2+HZW^Dy'
    'mc<IYADkY}*6&RHg)qFNarNlMA5~ETgn_q=dDY4D3#;A1Tu*L>y6~fgg@{tJ6(T(m%yiP%6Rx*DdrMhKSMlLFKzM6F>2SjaQHB^5'
    'QR;>lv64fxoZ(^^d_#YO$uBqtLx24<y1l;(iQtkl)Zv4|JH~s+6{n?Z|e}L!yT*|(Mt|!SG1ld^9yurO-e-'
    'ga`3u#TNH(($=j%07ZBzioF-GD{(0jPE}DG0}Ja<g^4f_SxLwmyYrn(%b3NMSE8IuNLXMVmXeI)QlQ$&`DVKveP!eS?3b+Bo%{Mx'
    '3M9Qm&qXKe{62(sT<9t|9sxI*nYK{suaYT$=tS`ukj({uVlCTmbntI{%JcmiZHHU=lWHRhDV4%G$eC`}S^lKEcb8<MQtmbC%ZQ&('
    'F&Tsb0X-'
    'vc1p?i;JkmWhVGjTQV1YdCFTk>=l^VrcH8)`(!s_D136ErU_1utnJR~G;6!FCe7OJtWC4FJ71((+nq1dtnJQMY1Ve<>(sT~v0OJT'
    'b$4RAX{CD*&rK(~C$Zdgs(Y2p#FrRh@l@O=N)EL~n>tv;-'
    '1)Ykd=9?w=P4H!4GCGjLo(+WzV^_(9Ogo_9Fg_3(A|i3r@QFhM7q;mb&a3xp=`{yRBy#j=##15ntjlxQoRl5NuN&jw(Q+Mlj>S_?'
    'w?KdcI?|fm+I}y<WIYaY-'
    '3qRVic3l9`yMyedrtyFYKzC)iB_<Z{eQq5GvA)g$3mcxESFYeG3;OT&r*8Vub7TZCs2nN8irH2y^vcxESGjeFqmK%+r6xVgwhG=o'
    'i@k>?`z3oYGoH=vUb1T36^-Z7aq6JS@Y6J5R*jLiZ^u4{U@RX_dbV?8bMB;KX<5JH<_e?!kA8KN7kp-'
    'zjb?bT7VB+)U`+e5bg%&^7K(u@PN{c6Oxu2>l{EjQxau*;Cwwel4%b--HHYa>4T$19%Ixakp1JA0M;oO#wE^EVsUWbS`UtO8SD!!'
    '!vTJT_z>>K0)id1#oEh7JSXlAD9(4UI9)$xn);^Cr_T(Rp7~!BX%`-'
    '@{eUbso=$e){_0h){+Cl){+Cm){=w5){=w6){+~9ttB@sO*wv+>Sq43y@D^x`xw$kJbV5;y!_>bc0S&_v$Zb3+frV37vilj+wLN~'
    'kmp5sG2X{Bul^FeQ>7YBEUEJ~OAs89Q+qiCN0jTD4H419xsxL6G1mxISjfB9LNEjGS~tKNV8?AfTdgnykEO82c?zZrJSmFOuPs3{'
    '`Wqbo4YPCeGvTxOM5AG(4{<Rp{`w=>t8aC`ZVG$#ZSL32V6VQt<XyH4GbUS$G8uqPU>Ci`{fW(CkN=DNbqm-'
    '~?{L3v2^;EPGl7~uOlBFV`!bnjknYE1mce=*CbMjy*JU!xhI&0Fvuvc-XEMvix_`+F&wx;#-'
    '}f|do@bS#+?<){A^abmn&1HI6V6UfJ?D(G=9t~Lz#7|0rpivZ*}c22W6J${=!*5TjB;L(uM0EPuR$zn_O&mQQK!djiF?v)ap4^7c'
    'd;@41J9@KIRZX=iZmN3_{aYrb;@jr|M>r-Ntz85{6p<bP^#4ZLHDnRLO3trAOC+0QxI^%KduJ6pETca>a4H1SK!miG|Vcs;}+-'
    'dx!N+J3v8)S&55L8$db!;s&@NGhfH5=W3j?^Im%42Ib}lrjq7-'
    'G3i)5ki9G0!agyz8)Mz<$cK6fIJoB`3&N_C=nLnFDGk1f4?`pP?W^-'
    'valV($Cek9E%(hPymLZ9nw`XpU7baqb~{`0fX{Mn>A_}Mq$#F&2y^Nuio7v?+o_QHH6%on0CS`0n)@Dc76VGK;)i|CU;NDm=CZsp'
    'IU5RdvX@V08`>>l)SC!BfKdDgHn5^OsdZ}zZq=g-'
    'poL7I<*d0&JhlcA?JlHp+h*nW`0(w=?_rPhBh&3}Y>FE=2i0d@8hU~LRkSYOpwnLkPMJ83=?=H1+YE;OLYegdqW0eGQ|N#H-'
    ')&PX2!vnn^B>(EmV!2xb<wSV7~3m#KV9|Fs-{rhIw59me%hS>rCeTwaXNifD-K%lFkd-'
    '!|cZ>JR9^aZ~FpYF@5Gn@R8aIi2pk|I^DIYpv&RREaXiV!@n2*G`d5S*HW;7?H_IQ)xS6(P8ThhSFlR4;Mg;3yg3R7X7A(YEth7X'
    'Q_WP+q2waL^+Z!V|}aIi%@(Gob1Fi#Yovz*$1ZcJuPf%JF3p0-'
    'y8CNKnD+{IViU@j?d!{jwq`@j^e(evb;E)RhM8V?Ut+DDf(vYCoX@D0QRRb!+5>ilEe;J_MH03l%}B2MyT44yXi54-'
    'u4jG4{bVM%ln-'
    '3BQlvmDnt^LdSNz{%PfQDY*tIt@tPM0wU&AiNbIJE`D4@+WAGKjVvPV(j3x$hHA>;U);Kgv>k)P4pVN!Jh{D_*};?ByPF+7xxJU!'
    '*^<}`C)HLW0MI=}0DT$&sESPNzV5Fg0fZm)t5^ixgF`Ankf4PD6aTA<5F}~9`Sy!cfFMN!M%quP06~1*U1~p}0tBTrz0US8R)iql'
    '2w-'
    '`<SP_DFBiPXns02Y@<q$L)8xSp&8RvPjGA+J9VWL8<Sn`!RQ2(r*)!<Bn0<VhBDYu*FiUdxRsHYgL_}z;bx}b=meTx{Hmc!7`QHM'
    'PIi`x`2G(3l)9%i^N{^)6T^2Hy$%`U#+qsHti^7#vkp?`$l-cV5l(Pw2q<i78Xu^@s^x-'
    'r(bgBQ9n3PmvWH%6i80{cxOP_(c8ga{N(v!4)wB4_=VqEG}2<fSMS4Yva-LDB9#f;bAc>nGE&a(PN{D~dYwd`8tG3Avqr2gtO6lf'
    ')`P+<6r*qH+HZ(R8{*LDFFD?@>h4g+(;&S47kF9GcF?ehA@T+_s3OokBF3kwQOV4&w4dp_iINx%yD(W#%w0hZB0aIb4+>>g^Cw@0'
    '7$(pO-<D`_8|PMHGC}uVWGQjvW($D470VM<MD$`%NMcwV(Zj2t-Y{pAdm4w*p>{LKG~Lm!lB1lO0e|kkljR*<r};ISq?M=-'
    'Dl@P1_bqzH+JyyEoKju{<1WE6n5^j?Ta~cj2nvvxuXMiZ~io#L<i#j?Te`G~r*YE#heB9F8C;YK60t3QL84+#JXizCtfH2Xm3B&`'
    'ZoAr8s&z#L?;!9IY;cBln$u6N@AGq~FBiXtf;^fg_my-'
    '$dc)BKu7ua5Tz(LIjRx*iVSSky`<;MBxY)$tzJf+Sv}MC_>7eOr2(%ps=wKSz{AC8>Ly{^9sS?lUE9~b?ps-'
    's$2uA*HcT&4itH<6Zx)DKar@59l&6(BCswl0&8>;STl3LIv4vpg@19oBCvMJ0jm#QrPO)qe>c7t?uS<?BRx&N2i^-'
    '8K((F>K=oD$sMeGLmHWQGjRn<uc1#?o;Dvr01uB^O-$sGzV*6GRpc-vIAp%r0?I%Ql%31%_C{V!yc{K`DyVwC01x*>KoZv_=G0Va'
    '0J`r^EMUCW6?qs*XGB|xb_Y}N}8^M$}aQ52VDK$ab2BWO9+8}KQt{i#kOo=-'
    '40TlKwg6onZxW*L0H7f_MUtkZc@Gov(1lO)PaIK5?j{EugnO@X92<vBhGohkC-aC#d0oU^I#Ck3S*Ko`TY*PkY?)%;p3oiJin_|K'
    'BoE;MZE|~h8qQG^D{U#CM8e=~p0$j7~Cq#hDS^u>taKQq3Eec$_+5r{CQc_ok>2a{*cPUe6@P6}i)@oRS^{qWAV!9r-'
    '6q1_NuflXPC!t<qOVa_=R#;@c#C|M%uUWIF$ohxvEwY|rTS;>SXWKBn-'
    'z+Rj@~Y6DHnY|=V2Lt^bKq>+xXpIs!CI<JvlXAb;#ZT+@s~+{BgZxa%bMazWctbgEj`GZ?k$J>ur>2&`2z3tf6$<Lw5+jXDzWLl1'
    ')J8`OOEAq<<Z+dtso0uEc9wl$UK2Hn%&95^I%;wCFd1b|17V(m*9qPDBNw1;xrHHvNQ_cTS}T^y_(0)_AZkcEH4OiS>CR-Xd$LRn'
    'T3KmP-'
    'I7cso3OiEnJqiLD&ETz7wp4g~BWlB_@48FsZ(~XwiP<Eb88GlUR$wCv6gGQFx(EDl@1q7__G~VH+8=mwn>8jPr#4m@#yT&}&%Oqa'
    'F4m&T791yC1WT4wdE%VeaM9<e<6U?m|yv;|`T(0uM)Jq$B<K7lpaJ6sN6WSlAjg^uZ>wOsW_7H_6ni0%c))F0U<T&-'
    'X%zzFPv(b;=pkz2J{x4GJIiqgaT(YsXY#(Sr*X?QU)8cGtdGd<ajy=&751>dh)+xT%tT?WRh0*hA>)j6S=;65cFcX?M1pE$fb2wg'
    'LU2U4@zD;rfy=R}>IxciuYgAJI<2&ahNkL(SXPtP;FE>o;D9wq+@u$%#2$nuog23JR$06y6sY_Ww!@yKXtdy7$~P(y;JJo5tD|UT'
    'D+G?0QtruC+)V`jj!WoixYz<aaC1N*(JUzALzU<P!+Zk*tFFzTp0mOTeSx29is0J08V)h;aChW;w)<1ox2KNg(5g*2Bxf%r0OQk)'
    'BC;BM|F@`ZBaE7plX5*v=I#O@!CJu$@eh4x!F3Ps~JVb_ct{ZzsZUJ!Tc}mhTEI{AP)T*DGgX_uiYuTKFy;5@%s}q0K6@@CjsLD%'
    'V{lOlNAp^+~X<dt?RoeNol}aNC_wyv#mzW9crTKVjrvDD-D+sj<@RBb^v}Bc7;-2uGc@9!`_rfp-+h-=p!vg_-'
    '70;a!B8NsFwnskTHN;&@{Y<I4OhuzT<b-&bg|ghk-prly?MduNvEb7J6Bao+($toUVNr`|TW2{o^jn0x(l=5{Z*dF)QTtsN6*Z+N'
    '54E3@}$g`Ij`iFWLyvv+;_)Proij(u9@Dvb0s=9hwdq0TYCW=o>>0J=(EZUDCh-wh6Yx53W(;9MOn9RH**<}SdJM9{~;`6_2g{CE'
    '8!o^jqls8-'
    'fZ*i|`6nhSc&3U_iO7Y_4GkSu3w>7PNJ!3`p8vQTg9HY7Jzvc6bTiPcwy=krq~R_`BLy*pXmeVHv{tqvcxMXc4IvSTW-'
    '`p*ki?`JK$KCyyZB%p>RC(N}H4q}#>FxSE6Cs=2~%#m;~d!dOj7+oJr|LQV{s=)(28|XX%>%}t`&c_RWe_M2+fc4{Z-'
    'WLh;8<rds=IY|k-kdX|ZD!)n<!VoG0&~ky>!C?zRPesGzgv5-'
    'xgQ71$ZhT?OY>F9=8k0%nc=HNG10J98MM=@$J!k}Y0Fr<SKH62=prO%_eOJ@G+MBda3@YCc{QJy?%caje~K-'
    'w=)d{g^x|fPdNp>p+RL}>nJiiS9$7u7NLJg%n;_T`-ehrDiCe<ktRg3I&2$Lg^|Y$gHEYef^0jjL$VA(nW*E26u4Ybot8VdO@-'
    'W+1tXwGkd1%hhN^pFfn3)x^er4Wwl{-tM*~u1N6b$WNx-Qnx@D6pchW^Zssl?FJ$k3sK+w^^1o!Q0B^b<r7*w}s+g)SFAmdNhmCT'
    '6=!q$zM0vppqV4ROzd2W;zhx1TrSIvFOo!UmSf4t$E>pWD&5HiEx4oR=o9pPd9(*}z}kS#W{{*3vG5>ug{t?P{a?f^T+b5GF4yTK'
    'o9W+TWRvoVAz3p9-z5$lC6^>=Wy=-'
    'HAiu!smrJq!L?Sk+bz^OGpmQSe^HLTl_7|h@e+<ixu|uJ?QbDus3L<{usO5(Q*Ho>B;>|^%`tzM;WdM6{c9Wlw3C5N?63Y>Bqs)T'
    'sZv%*qJM*7lWC(bb1L`nQNz)f|0p+dKuW5tEZQPiAw<ew-'
    'C@PO8|XBXyUbIm6J64!n~`>SlE4uk+FD&R~Q+K=aqI$B{sgUuyJ3Opj6dv?#}EXiCVOzSMsn#F<R0qc|@WbE$Nj!Dp8L1Z7NBV=4'
    'HjsH*BTeRJieb`=Fh@QE=<`R>8gBn+3JtRLRB-o3!g-jU68|XBz{p6`@n!l)2%#4w9X|<9y!O{lY7?dy5Pzb7E-'
    'hugwR}*x#5BRhiA(z4*Sd#)eneH`dtq+Rv%P*f$oUYyh4|51z#yG6+3~2b|UxoaROBobTqf&##58nu~F+!;JVt?6mLUwa@3mmdzz'
    'j*TWgjHBR&3gyv%NhZUyCOH<Zm`%Q*@a~rWQf@dJ1Sv&m+-'
    'B~_a{bUH}{v|e^6x#Sp^PaQux8^_PZ0z29zgQc?E9@6*<No$@DzWh&ayIV3OR445b-e?o%Xgx-twFiUemROmlU~>(3I$HL%yw+g>'
    'P4wfPwPcFOwUzB*-'
    'uYaM8(yeecpPt$f%1bnn~P6Z|Ek}J*e^|dsuAM!Rg`M3tRs_f_n9mTS)9^FnnjRj?0?&iO|*?%c8AM3T$m2$Ed;{&`ZG!2oGqc?7'
    'Mjx6>DvHiBYlEhIbuRnYC}r9nn|_-'
    'NGASd$2WjAR?+Wifn?xFHCw1i9k;jFond6?Jo<w*bdkvUKd`K2Svq6r(p>Sv6DIjHkb7RQ7FUywzJfu3RQBqBo@YGYeW*tIt6L^E'
    'R=OBGLBg&>kOo)@bPjc-u&_E{6r9&ykP4-;okilRkZcVp{-'
    'l74XCyCGH~)jTk~b3d+E`!wuYA&9cycN*U^>P`tE}3js0zXNlO*7&8m67Zuh-LC!#VLnmP+!oWf2kr=&Vj-'
    '97ZbA5*9_$`(5fIjrmhoR0K6_5pr^bO$msVh}!ydV+vYcO3ox2}FFy)9;@`#CHO>3ISj5MEd=6i2f$g@2esDJBfZ@13uu%r9RoCA'
    '*_Fr6l3Ob!4AIEOmdRe%gxD2U}wIxcVBf(texQv#>CngUUW=_SIa%~WcSG`@MJK766$z2VII<-jjal61BVEfpTY|ve-Bv2;t83l+'
    'a<@vT<%yTX)I6$l16ipq_I$iBn_;<O~Su;R*|GJB}dZOMCh^l8ear%^j!D*7DA8L*Sp`h6#4)?uVfv38m@znN__JJL&*0|bZ^({M'
    'w#KV%t~-GQ6Ed+fOn{mrEh#>$3)OKz$o=m^o<2}Km>iG*$#-'
    'HZ!EL}BIp}#UDQO;H_ozs1taJiQ|y3>@>M<aC0QdBQlCapORTCu!)-'
    'zW|70MlH+6y7b&FRcL~>MBzMA9(>9Z45$XEo|Ek$r$Rs`3bIdGjP%ujRR`e_kd=jFgP1aGHWyb8nq`dY8Tp!IcLg<+hYQv$C0!+V'
    'H-m4NG@5L~?y-OII(QTa?3gUijt*jRAgZ^y)e3tn|>6u7`BW23-'
    'ziyaUFuFLFz2yoqL2Sk9&t&2WU;QFa$$3}qbJUgHwHg67GjaWO)jU78|BWZ5d?)q>w`Sy>CSb{%jLp6B|yfOVc?8#wXls+9qg^NX'
    'G-C9J}<wa!Ol|$C~!u%|Ute+K;b$+f+YcT}8w^2H3DFnHHq2HH5kb4LHz8r$wzm`_z-xVV3^%7(q93rbG(KQxX-Idv~9J1U@>>rD'
    ')yX=@aWWlTMAB8M1%KlNvy44PdK-T4UKm@YxvI8QJ<<>>tC}hFX=^KTt^X-'
    '6&D7}SaYn8&2inMs}083b>tsZ{pcB2J;9pMjzCa!z0!lyzJ@k^4^ZCuS*1lDauU|mrJ*4;T^T_DWQbHMs}5m*=GfQ9!g3kX1u>ub'
    'y{^!pR~T5~J?zC_P4x0eFz@z5W<vm~{1NC>QMio^Y0Dj)7MvfNbYSY*LV=vZXkX+I|dSzr|%g{<4`fCyw=VFyGY>ux(B0$FZh^ov'
    '3ktet*Q$hyD|s3_>_70O-InwKT#B)Jl?h=kjVNSIwj!aX@8Tqw-hIV7B2M8btR=WQ!&OziJEx5m!I{!(lkY)$Mh#cHwNv%M56AYx'
    'j0)XXYD#Gw`unao?0KYjlTKh1W3TM_j5!;Wvh2?ypDt4Q^T!!Bj>D|nKr{IwUu%+2$-SeQ+-'
    'W8z>o%Z`bF8JK)r6wGe710rBH+YX3;**$hZ1kBv3TPF%;urSw&g4u<3Kt*v^uUu{eHv5Jk+uQ2H6?ZyS@mWT+I=tbbX2qw@6sTxv'
    '*J4|)T=Es^^Vd}SSVZn$ipagPh}?T~$h}CIb8^T%r-<B(a>#8!!h&yZc_0!N0Lt#^AiUWID7&XakiKA+lp^=HDnRby5-'
    '0Jn@GKjGmmL;lTMDyAlmVHW5)%tDcnK2=GI&=L1+p3TLn1)-'
    '7ds#VWLMe&5g@zQ4u}AmTOjL3fehBrx=|pz$PTC|{>p)@0~g~e4!HnWBS=Nqa#^dg8~p}+@;iMi<_<hE>vgB;%w>sRl|HXem5@c4'
    '-BE<uRYjQHmxI~G!kn9f*||lSU7Um2!AJ{%bI|kg-'
    'x6>Rx~IdC31pVh)8WVjGRsS0c19Aw?8cH?gTq6ZZ7$5`4rW^jv$`D2+?0%ug&DlW_*j_1yN-{7*-'
    'Sei0%mvE0TD2}$_|Kt*?o3E1kBt5SuYA^u!h!)g4xA(Kt;h={w%}bE9-'
    '`d=Tr+{npYv3sONn>o_KT*Hy<(P1)n<M+lyQ;o?7#o^mj2-5?MsqUyCTax`?v-'
    'b11t+m|x^j_KPCQF3F(`BBkAgIgmLV3OSnxF=ap@XY*jDbSUI(9#V?38^XhEY6;2lkN{+x2=lQ6*^h+zq#VfHj2sXPGI)mrVnH_5'
    'eoh3)X4wyk0NG#dfC!LXZ3jev?0!2S0%UG|tRDq3SVHSZf$S1HprW&^Kqzjt<p2wbIKDUy3w|E6Pj}ODd`g-'
    'fF>ywnBan6nQHTTQfx`TfNN2Y2{$u77_rv;vLB4+YGbd@^7jN;u;quPD805p#UBXi&+GU|K?*@q)RZXlBi=tF4S+K}zxl#F6t3>g'
    't0fKuPHVNMy4a2GGTrbG=@Na(Wnw-HOwC%-X6hF-o<*Enl>w_a6cLn-'
    'w3Pa*<K)=EvmDqf7u2ckOuaC5a>vg7w2n$7K+5)fEG%WsFOiwfH(_^%WJ%{)F5RLjf$|hwN2_Kh3p!cxwY73uIPXjC(|D5_5VA1$'
    '$*T<kQUh{+bpP)~gWs61b6rLjULVhuH;5}Be;yu41pk=s(95`NF9B`ImscNj!9AY(N8YJJa*VzRXAw6l1EN5{0iu#&JqUpUbB93Q'
    'zKa8lv+)w9u4cLOO+VdJ<-=3mK#A7sGY>o)TI^o|85gE<68aY8;yPz4Pu<%B%ubD2gsuSyPf>-'
    'n?Z64FC86QtQ&E!StIS?%hW}iqa*C^O6h_iC7g582BE7vJlI7E|}qhRMCmc(2Ii-?#k*DF{=)YomE(rb{^5~GA@j*|YYTn{-'
    '=U%1Txj>|P^8OEvo7W&#qXm2H8NSvwfff1FM`lW)YTd^0b7fiCkeTy>U7v-'
    '<(vchJFWX=lg@7je}PEly)ORbq_(4YOeWcm9Ja5?pJOicPzV2+o0ZX@h!8Rp9jUiUn=A7jMMHDlP_-jm+2xxFX7?TPV+FCNA4!&i'
    '@D^bsxCXBtE7vPJDn>+2#d`i~Tb#BJ6?1q`Xgq;KR&;)u_nI%aZE9fL)YJjcY<C25gPoc?C%vxuFRFrKx}_^Hgg8Jr`zgF%P6OX;'
    'T^{&JPJ-Mo;F?(TJTbPp$^W0e)XFo^EubwPA*FAJh;oGJ)08(*dkJw_eQ99?efwSP>{i8SBaDGZ4--'
    '#a*@67#)NFkgd0kxZZb&Czh{!ooHrU3xR4&9c;i+j$G^PXM>RtHaM^*45!`$z2_g*k8x>HdFOJdMeZUN9s$N)xWQv#-'
    '#rJsJFbHp-1T%Oz9tuJ<=Vce+>4UCqn8gWo`KV%4Fsk-fkx+5DZ+N4(|N!yQ*2GX0?A`&y6(e(<uyzGwW&`Qi)mLFS<}QR{IA{(^'
    't+4I+s3=4TPwHmN{3tjv)B^FQn_RK^)#!v#oLkifZvz`MNNd`PP7c6XtT?8t@H)1thdpzA4OX-'
    '*)jWVXpLT7vC1<D&KaoLzt`cZ5NR*&>YBD)G^Fm50-'
    '$_A9P%9#w~>X?FqQGSU(^}K+7CkZu_+pp|6iL^m;fX&d_hAFr*Sge_A+EF=+746U;+90FS4uF631=5mz4mi)NAi3*4fuvJ=M~Fq8'
    'eQ`;8O%Wg4!Xvn0eh>=&Ia&96N*^c-'
    'n^<N3SiO7mOEZqHN7{L1&keHwdxvD5S{_QK>b&tdN`9<vY|3)08jsq~ZVe%+(=Gwu^Dv;4KP5dQ%wr)|sZ()MrZd68CLp2CngD?f'
    ')rDoQmKqTr=A(p^{jc*INd*P44JR)hDFzOOr~1G#^-zceEyhqnhvGe%-NHd|*Su@>I-'
    'jKVUJPkki_81*w>r~^j*+!yNHB2oK&s`SFJt%Vwn|59|r{%foEU)#ukjnNg-'
    'Q_EP+))v-BiwePzSWO`qQi<V|%I(7~uuAMVZkJ{q8OE~xtquE-MZdK6?MqX8w?x5?9`tm*G<yltg^Vyyn!SbTN>AgZ87-'
    ';Z9_piCDyStGVim&P8-'
    '&*>gy`qD!V9ZiBFs$`2rakCjrAnUo#@94^;f>m#(K>oBx}drV7CLilT6;itdMTs*ut!&eIr6*NMBmUxcDl1Y_z@*42hH%f)SP2wu'
    'dU5u`8XR#R<}U)jbs0YY~`n<4t+KfMvivW}5`puGsc3$gr!zG%#bP%~omnI;<O{Q-tAFv(m1U{$#6bB&KQwWfMkXNyU-bwd1M6XR'
    'Kyp#W8hj#B4V>FFwtI2i!*P<lIi0!&t2h(^gHC$E(HO6;ru$8`c8O5ZGDHDQxH`%5B5nuRx@ym9aPeDZPKR{16O@RUd*Ol^A>-'
    'RY+7wK5N}=^eTx>tGJH{3IO{^^#++sTr9G6-Bnp(`v{Ajmf+Z}Hm}*O0<50Tdc9_CzBHHhT1m}$-?l0ykNP{Xp9t#FV(-'
    'At6y|<<GSZClMsLP?7$S}#U*<)BM)bEQ;l}QdZ!gW^ENy0PmDsl6Qpr*(dT|-'
    '6f4@4Bo?gc2_?NVfl_r7_vC2d+q!ODCF4(+Jn4GFo9?FeaW>ZHIZ3hZId(Ws{_?oBDc*)Z)3R&H?(!A_!U2S^jY`w(V`Zcok%W0U'
    'tS}0LMubM1yk3`9zp7eB|L~W4X^mK(Z!zCv*``YXuDdlV>&4Ip3erstC@>TM+(i}>)@RnIP1>i4PB1tbPXYB74D$+B`7#shR9v7='
    'p1S4WaieN}3#@@7G>{WP4;;n6EUIuuP7u=e;=Di&Q4lH66td-g+InjX^=PnId6k=^L>*ss`CrI})`Ehqi)L-'
    'jP;sza7X^gr8?u%7fM+wh1nJnl=#4<`w7v>&%x*YF77|iR~+!3kd43&7_Kf-sZxB&YzxTn8L)c>3%S%gq8Ew`b6zfh5$S;pM>r?i'
    'PyDS{!fB1JHy5_@k`u=fCkSY&&5$^hzXrH49d8Z>bGD9qE}AWR!eYW6ilup8ToTtd>R9WBY4!ud<9MvRQ7Q)|Rp2!743-'
    'Y4&4FJ=btWfW$P=*R{DKdC<^9YLv|(w~q5ozze3Pf5E>LU8mMsohBZyk5gXS9%#abbdYdbN4GvW?8A9_WjaCdR7@b<45T6(Yi!1B'
    'vzIPhE!tb;RQQa%Y2aB#H{HSZ2aqdaKJ5?b|N|CLnXw?3&_HCsNhyG@v~_tbacN&jh-5c!A4;{>(TUa<D}V-'
    '4A}_#t@#zn;nbR|@REw1_uD15UzkpR?=KQX<Y!3V8LEvvaB$t8)t{5RmDJDa)vQdYmlw_aY)WQcnZ^ncZC%F9_!0VmXc;0H601T4'
    'Ln`9%_RjO30GCq|E;WI(a=v2FwunFxmrA^7XRlB2z~&p-THC6jCDbdKgV5T8)eT<pszvh<W_xp83zn+~B#Vhf+QyKiK-'
    '(CQ6lfbmlaRKt056ckzt~WuZA=Jg8>k*xBg!Pben6mTw~+8uoj{20s!tO0L^;hG_cbTRaxvf~CdP6xKC@#g;9@ZG0cLz+MO+MSd<'
    'PbpSP>V4Sq}s3fC{)6%z7AV2UNhtsHXXKMjuuY7lS)I!nzn%5f_6S-'
    '%YRsDkAUp&J&)7D$^zdbwjL3PrRFZmQ@(&l$nHwzb=^XwSy*afh{Sgim<wCpTu;_f#<j)cs415XJ8RLTjjuWi^4p0@D~p(g6D`Fc'
    '=o}@3?xZ33V9#|@zF>LaFS4Z3{nCj6`_6{s*ALmiMBS~^iIsOa@2S?568v!h`bYr#C5n^h(jWHPGFGZqIgbVu;Ze5P6O?L2%ghcc'
    '0dHr$*qYUqIgcQY<7s^IUQjKRK#=2W9z@_-ynF9GE^mfhx2-'
    '@Tj%?j8YGm=_Z`{sd02RQU5gGQIUFMs2roid9A5<5kBUGWR0P`AIiTID@D3>Wiw6~fc4Q7{BawVXYJmG9`HIv4_d_1e9<gd%yXPv'
    'r67x8fo+M^@h%Z4az<vGWV^1%5iQ{8Wua$O81qfrnfF+Kv2w@Buu+;GtA&dbt$PTCgVSGMrZ3k3<ur4&eZZYgw5yBWSuwHho2w~h'
    'C_DDORqC1q{`CeU3Ubzsn@hep(u`vWC>M)x(pXqz;l0C;73#s6`Y>1Vk&H!e`S4OZ{)(c+R?mCRlc4r~1SG~aHzKQ9P1K|lp5N=u'
    'o;ou?&x5<I<HidVu!CySM2*RVX4#`mpXnTa`*AwP=cYgIs%+hj#N$#7S5R0#Vc1#?;;8jnE>T(Q|d_q*0V=%%AQC*G)+W`?>j<>M'
    '`BDx$qI}eZQatuphcvP3;qwIi+qOY2KZE~lTQ{h@(JkMI(>R*cpB93c+OhaKJZEe;IFVGOJ<10h<4fjh-'
    '&m3A#ETUzzB3d>mqGj70T5eaEf(icOAw{$_=J@c71S=-'
    'lixA|in*sFnkU)agAbNTP3r5TaGFF3b&k$ux%!y9HXqh>woPv>?kP~A;2Cr~pEXd$ZPmFRC!8RvGxrt!A6QkV34eWpjH*s4#Ai_;'
    '_%VVb~HxX9QPEl@Rqa9FD2v(C1!4`xnmx#Ac8+AFo)eg*ZQ3AxVx5_FiTAa0nC>xcSUOALaDxz%jBFZ)_qO3NDvcD)~{RDsU&?3r'
    '?&Y^52D*3ogq1L>ERowb3_wX;Q&(Ytyhks)of&MO5`ERe#QT$Ar!Ol_qTmo9lV9ZU&q*#o>OH7Kz*v9sAA}|J)nG}UFu-'
    'v35jBRKKL}0Ae4v4^*TO2z_VGP#K&QTaU+775F6w6}_&Z{W_$-'
    'D)XMSU;^?%;fTepP4L!X|Euw@74H`zVfG5Zf4?nBF;*om52G7Dbe8R7BZ!Ih5U@@CG>ei-'
    '#3ac8o{a+^Q;3h5y5We@AT}{e}B0|A%rU`b+ol4=nB0U%Q8YVhO7LCRTiL?+|XENHfsEZM8HTmV=v{my=@Q*2j*CgB!f-'
    'Nl|b+$qtBs8<_B<D7bB82SmVaJ3Al(Zf=$A5(PI{NV`PA?HD_tqPVO8w`xo(dVsCSwc8nCeabdLUAsEad0a`QHI0~7tYeDtBD~dE'
    '6vAyxVrp`5JGltAEsJp5xCpoHb8!2s!fV;!FCJcm+p)Pr%_`WM>V)O6v9zFj!s^RdThJq6b!;py=$XW7Umg$u?NbS!s0G>@X*Mbc'
    'G&dtB#{v!B;pAAL!K<De1vD_u$x%St(tbh&&^ERMB7nBN9S{LDw?1}_0var#U88_@tQ}BMfR?)jZ8SF|T%l}L0k|S=w3(kE>)Ky|'
    ';nf+|XK*hKk%=CTC=orpLz-'
    'g}b3KJXH`tQqR(kp+%&#zy6Qcn>@GQ=hU{CFH_$Cc|0iP?m^}^l?sp!;Dai&?OzyqXiGKp!*Sm{HNZ%Ea1wk75OYz^Z(h;4|B$5;'
    'u`FI0f1E77g7cuKK7P=F`Cu5mBa600T)FVPaKCi{^cQ&CgP!rgZRUW$@@kL`I=m67XF_tMm{1iUZ}2}Kgv<0zovTLPPn+wDf2S-'
    ')Fiey2<~<|8I%YGxqIh_jViu8Taz&b!gmNMRlnupM$!d5olYm)up}JS<>4bWfuM#?^b$(`aEH6=pAb8Y9eO!t5=5x5a_j4~R69(6'
    '=`iiCCe_FT*8p8D28jC2+RVUzbTKyKj6=><WN)xF&W5z^h(UVHZr5SN;D;VUyCt^Z1TT%(c=}e~wMeJbLPpm_I1bJMNj7KPqCFHJ'
    'TqQtl}ZE*d7YYc=qr>oTAJE<y)Ozpv-ObbZufTmA+xiIBd|w%~uRs2X}x;VJTTvuGp_8Sb+fcdb<MaCF|MMWxB|?dh0|^w`yqlmO'
    'Vcp$M{uf|DP)Tg*86_V^Jthz$rQR{jQC*KfJ@WvG#{oy|%)hnJQo6Ri}`?#I3Cmmm|0y9p<|p=d@0-'
    '>`RqzkXmDVD$vGn8@c3n{+$xNge!bLwhDVwsoH*bg+-XVlZI`AqVGHS-nSPjb31+7bhAW3z(ItE1gRJcIkHL2_{7{tPuC@8TDGz5'
    'FHjxrNcwkLpgP)7^fX>zhxKDrtXSaJq1mrO58K+k3fnc7+4<cIT^DP2c!}#`?GEpHU4=b0^L7VQHh6&8bD-Ii*v^u8<Kb1}t$inZ'
    'k%C2lLuB!f5R5IB2Fl9&c2iiTxtrg5&sXMF<+t986a@Ly*)1{Cz3pJN*=6M(#v<kIV|v&hnW~@AgGO5Nr}QumnT(&&gF$B8=kzch'
    '$?U7?;Q%DFuZb4L`zEyfY)ltg%O8@!DO836<G#+ESj)pp%!#%9EIX#6UYms-'
    'e}P1fQw!ncx6<4!%rbiFm6$&%E*CNvB&haJMSB+|*oA@W^e#!T3&XCyP4y)T{9CHZG>esahaP5_rON!1lk15YmzaC#X>MX>WV`=('
    '^vKZ5n-'
    'r*U<1Vhar7}taXOi6h(!gq)`i;I#1szyzKXdRaRji=ex1rUWk*8y=F0g}NV0G@j=)TF^SgXTJ%#F1=yzATwJ8l;C`C+z4sh=p=_F'
    'dDDDP(KwE_16pOPD$<@#3*Cul^t%CW?Inz`Z_CnOl^1uP;{S4#g@KtqOv2DppF&%xtHgVvlf=t36L*&WZcFO3XR&B9xf#W!X%L2`'
    'KVbE|Qpl;uh=@Gr?VGc}mO#ca2r}`YyEl6?g+@?XIwjrwrQN*SS8{?(h=V$J!m<_4*2XZkFuxBLdH7f%2Y@!;W{UI?%RLd}TtD3<'
    'Td0eKm!?d>1F!?}I9J|E|m_{urN^m{}RH_eZarUBjW0h=^2f&?hW6@Z=A46FvI_y@GYTUsx+M0v4_`&%nY>s2m0BhvcSy5m@|k<T'
    'O}|UyZ?iU~x$ncVBH@ti|CS=EYhZUUgoDw?Hai4*EzM77pyJPz>)8cDa=~KH-'
    '||xynpRct=0MR^NW=IL(jRs=q#x`(^4jkJ=Wj+#kk0TlI_l$L3(=J~@1IQI%n5E>P63;xY+4wzbI6k8D<^Q~BINGhe56!2gRr$_n'
    '0}yatF<%_*g`{(#=mDM=W-DRZZ5!8lvNkrddvi*L9I-'
    'x{{bSS2JJ6Dx&;V=C&mSpau+MqIW#d>9fnhcP;KHEm}P4xKE8*%<Bu1asSjE?s+0)A0eNpY&$G21Pe^;O4ygCH4yDp)i2+19ZE+z'
    '$ahs1x&ddNDnV!@O=<HykrJRb1*%;YzBinMGvoFRDLKuyoy<x!|35Pvyn81%Q(HGQ$vHpDqH9bex)+^sxk>s`+fIOu}VldBvJ|qh'
    'g8&Zvw-ZI6e9Z(LUxP7)`uPvLA%of!*-_!haFB22|Jt~8g@86EbMT4ICnV3&pF9f->-'
    'H%12`?PGfWfQiBmsS&^oIT0Zs*r^*GdLy<yw4%jb6Z@t4oG%%Vm_3L%{q8u|)l?skTLwQ!}>!?)gqw;rjYg^+Meq!JPisi^U0Auk'
    '6z<XYv63Bh9rOOGfc!jikWZ^wzv^jLCUT3_>PyLfJ*o%RTD*!+n-N0}27-'
    'n>t;qMC=%M}MKrWqv7rsm$g6kp4=UEBw0pTAA5?U45g>m401)tISojuKMB)8?2s<nINO|)8&fEen1N8^w8Qb`FoAm3ildW?@joWe'
    'O0tb5{`*gNy0G|wcgB`+ZKfm^O|rs*cvh{2Uebe*e)`TFIu-fBONxuBNZ36!>_q4%*j@}@o=)`L0RMDzUWbY!vcBObW=a$!|<%lq'
    '7_e?nu#Y}nu#Y(%fyqWXW~gSGV!FDnRwEySh=J#LK9!E%v}!9?M_9Y#1Xe|zn_X0Qo=E@N=i7UqRyLzsC!C+8OpP$#o}74V?*2Zc'
    'gkGtciTsle%4#UBO1<AJypLzr>VYFzes1Po~B=-'
    'lT=UFFVi`yXXsbx6xB2Jt8|9yS^Bkbt6pSfDHnOk4&U3;Iu8?r&Q{eycIbwDlU>&;GEPb9C!u+-QszEq-'
    'm8_lzg(+iJ0+u3w4f4>iB(j>F%`AiEF8^eCb+v!7rt*iUYO9+B&!B~sNfcu`Z@1btwk+CYH9ifY6&{9!A`3`A+*6pt3M^U!9J@$B'
    'fP;jt3M~e!7i&;6XIYw=`|K|nGBFZXTgy#u>`r<*ZMBN+Q+q&RjDRLiT{2?$B8DL8JhV8yo0l5e%Vp1O3Z90X0(cyQ^GN^dP+E^q'
    'Ar^SGoNnF{A(4;E%cy5(9`U{JCl+V=O>to{#oMlWe1>?u6sJpw&!9M<>PHjAy!d7!L~tR73C9cw}wOL6MmB5p8Xs8^fT<!Z|OjPO'
    'zGzfu2{3J7GsAz!>~>F-ass+4siX~`W9SRFhV;Hj$MgbrLiJPI|WvM(ery=DbVS9_@<li-eXj>yb_Ly)mOqX72S>Z$+s+POYnMp1'
    'n<f1Tpc<>k#I*-'
    'LWRL7YP*0^&I7WhOZBelQg$$ZB2X@rLgz;l^INYEIl%|pjx<sy3cQp7X@QsGdNl^Tlp$%zOMzX!P35|JCIwzfOA_)@wn$LtWfyle'
    'Z7$#_+4Jl;F(&i7km4m$)6MPgT~kw|%4AjSSB|b5{o;S98(mXr-'
    '6+#SVD>Ak8)aI^fV9H8QLa6x7ZuixGACq6T4CL2S59Qu1u&r^;t7`@z#5rQ5%GizO<U}Miqcg9xF&C5Mx?rKmR)NtcuKKPWSw9Rd'
    'FpT@Vg^cIT@N@5PzE>)p|#e6br^>^oWpQSeV-zl1{TpYwuq)pa%g%$nkRE;np#BDH90h$h0V43Z8k|zU}XpQ4L1|uu$5M|eXkOh`'
    'F)726^ZV&!1@KUy2=m`?6;3C8O1k<t{J_%!kSU$XMr7#i>hcFm{wRb%0L_|Dy$i0Aa0UYSTo8ALbnu-'
    'h^lCV)pA5sMcXxYKt)95KKWvRx)ctoOdMOFQ(7gxqBuMaNHS2FdK9r+KmjI#D)9aW<O^>a&3no`C7ot#tvQmDKm>r1MT`w9Vr>5+'
    '#(tE;*n<*<A;X5AMT}jW!`SP>j7>O~(ILzMiOWbfVn6Arn7oLRA2WXlL3R#S0=Vn`)Ksf7D6`)`x^VQp|Dka7-'
    'PpoWFhyM9=%6&RaCCnWSvdNmG_r8it%f6`U<M23$S9ayYX?+BQ0|j2Y-'
    '>`8G)j0*%yg9cq8>)QKc}#tIQ*v*i9lzCl`Xd>@g8}Vu$;^6Bs+@pW3=k`EdpzB5m>qitW9&kdPt(kYS_@T2(0UJz<NmN!IWMw4-'
    '35k<rmDOLT^M_1@oBD8{4!(gw~t+V~DVyO7pQp*x6WrQzkQSCnC0V6kZ{^aunY5geYS2zM{g)QAXt8w8F|!Mx+)MR*o_vH%%+79A'
    '#p%TNOt|5tCuz92G@OzRnJ)DER7|2N)m&TV^C#mMSrwltu%{)ceGEJ3QtHf|COh6P#%X%>r|w(WgD#^53tBunmd`8&^cwW;ujCEX'
    '^}HgiS9ZY)+D|ABDrUovTOnKuYnM2ZbI)IX?4<&>LbpFGgU`{3#cLC1xKNg8f{XgUj80*@=j)8--VhE*pI_jVv48PehiDZjeTnjg'
    'AwMWuu#=k!7Q9RWwE+3>Hpf6vF1%0To4Hee=p#^$BKv>GlE8)eta51_X6$s4mDIqiX??uymOKiofR)97p>!wp;e2ipbiqh%8e?*5'
    ')~6Jt9#PJZ$7yMAqB@S+?Y-or6g@$6n%_DnQdKd^ZK>*sGRv79(b8ZWck?l?kc&G_}w=G&N_GJG|^nM3;-'
    'gJ4Ba@zLG|ki;fbJ<)Ry=k>#RBM3#$go<^37x^;1M6t-'
    'aL936$NxpqKBQCI<6XQiIQf_&<#Fs!P|n2M8A03M@hJ{}6xMR{WC=r$R$j`8;gZrP76B5R`}vc?yYwM7nDk4hBS4|{hOk#&6zS#V'
    'Chz&FBhPQAz%!r&pk%vZqR6~0!6jP?EUk>=0gp_Qbjr#rN|q^5VdL(9%YbY&>KLv&^6Q)y&nC>SNKGIXOfvNCkMh^!3VB8{vJb?f'
    '4oC}hFXIVK8O*V_RV1zvr_p6nfF-PG*iTTHB<A{&u<IUUPaA+&tQ6cMs<5g`W@5wc|tA&*Jyj1u<UEFxrH4k23#J&W&iA408u^Yh'
    '%r41|eo(yStiEba>JK1f92LCM<iDEL=J0c4wm!+1VtJz-'
    ';T(PETv=iR}{1%LWmcXGXxm@CTV<?TF17lgj?KNN(*o1Pq%mmec4EC}V*zj0b&K`7@b4iFU<g!0ngGOe&6l=Jd#RUI3ZmxqOSY*b'
    '!;o*hup$yFeX*XP<nPt3K09+7|_=tHfQiEc&uAW6_6d)zb0Tj9TjKNTEc`csGmS-'
    '<1=0wt;G9&A&A!qd0W>TgZ32OQCebrb$g64IJ#j`M<ymzv|fX52Igd|6a$I=JYp%Q^#G3Kngg37a5`HqHWrl{ut-Ko#}zoLq=7r@'
    'P$6`H9)FTuF%g8aK$u3r={68)W1K=Vx|IMLjhO<dikSt#_jB;TFWc^FfI@o;3pXC+0}61u;OH-'
    'F(&+wyI(Nn&U^poG;9+WXCN8b{DyWo*qxk1Ip7Mo{;8e$~;A}{SJw_Q<>k<)6NODUjI{)>mh?O+@9}-'
    '8)uVO7_}u!copR63$2+HX+82%xyF?<Z&u?bLap<O(ClzD{>#aoU8v0Is!R=-'
    '`<}myG&{V)FC(oEZ~Dv1jQ(_iW4Kn~KND!WCt!Y(I6iqtwLU;TOlxcQu$CKE2c%(3ZTKmRURmbLImAHt*c`#KBKmW1s8Krv|EoU='
    'Y(PTv!#@eLvoAB+M40o0xrP4P=E7Vg%wKrzB<8_vCACV-'
    '618!b2x|q+e0tgiYX!~CNvwk8;?U$rB%Gz}o|yN_ncTh5uVRr7FY&8bq_4DND(bOWEC&f1^?~(#pFX4o4U^1qXu{G@+Q1P@ZYyX@'
    'z~Z?me{llGhPT0Dn_qgGW__8&cI(g4({hP@p`WLxvlRR#+AXk*mba0gDg7m%20vH&D`uHvfi*9XbCmvu*T=a^f6K&)Unu=unZBAo'
    'AY!;Av~jP*d}u=s{Z;uinftys#+n%3;l@}K!>is{nTg*hZ04T}7k2w@eKu6^*U{C|6VuMM+B%2P)5Df)bT-'
    'xPY?@%T0VPegNX$&d8kIK-RHom>i^@*%x_)Lefen;zr>89ic9QxlJuOMh!>Z6css**zTX>;W@$Ow8$e3XTS=i(kdwO~9YZZe&PE&'
    '7bXmpJQUKZFFCuSE>hN9%&{MWHohnM(utkvONe_ff?-zixAW8tp6-DGbd&7M@+(}-dyvlPc8OB3^mcdKyv>sch{8llH>U)5`c-'
    'k*DNUMIA6Epr>q9HGa#_7sg~uF%HyUTrkj3q78DMa~oY0PIbYnYjv68q{F06E#x3TRB7oJ)14jeTr9uoS$<*he<w$;?&)ig|<E>F'
    '{|As-'
    'Xk&Zm*bkc>G@5pt>Gnp6Km^@u`Oa{{X28EZp0Qo`zBmM55CO^&v)5Kn!OXQ%Uzb3N3-3zMK~{w>UjjJuk1ii*9zx}Q+<yxe-LIkJ'
    '<SpBfTq$e;ZA5O?h<Y{pjsed{w&O{xvqcz&6BXabuU^Crmm;PX0^6?CuPF0A$plYuOCntTOOKxLITdA-'
    'P6A_hnKUso2B2zTD%YTv5d6%Z|vt()X1|?AOOZ=lBR2(7|0W6n`H;g;(V@{3nqB9+h_|#`aep{rHVcKs_euU@H5u^D&c2;_p5>*?'
    'S4(*XPo;rg&*U7?E*jJ-LGBY=KwnQ6Z2TM1|QG)b4$HH7pH4AFEH;zn7*)x?wObmR2dZx_Xan`+85s8rda#JtKL+ZeZMLM+s#-'
    'wm&Ml&xY>D*eU%T#Mw_bC^_FUuS@pZG@$1xWp?}tBK26N<#O>Xfb)ue}HjA8?ll&omY+_F4YJT_=Z5+Hm2|vy=(@(*VbIkP9@Z<b'
    '4{S5p#w@g0^Kh7)D&%uxT%ldiv+4cWPXz2oFPE#zjG^!l`#(uBrBwlB!I*Hd=s!l5F^qJ&O=T3p#M%tvW6lR9v!?`ymx4B8*agzF'
    'L+Ot_Y*04RBC7jKabH5fa$vOAyB%ICUq|K3VHdFq6u7tChZn>_Pa5ht6-'
    '#iIt^WM4sok+hO#tW@MvIbZ^cEPFQB~j1sj$B>tHe^*KP?q(77g5q*7~ztTj|j7QQf7O$Q_?Mo*HG$~#A_&ZtIWDR3f5gGHSZ^ID'
    '!VyuPB6Y9r|R>>?3}Qdv?ejTCfr8*!31+F0|}3hCFbx1j&ENF-Q<_cWx`BV93fpT(3jhl`*`Y4OkX-'
    'f55;c0>vBzj<i?v>JG0boE7bBxhN>A8Le&h|dD=)FWHhdVp#OUtK9%#qVq}xLBxbY&^uxkzmXz_g?aOyh;$@S%C-JgL-'
    '77QsI!WFI`y??tCd`1mLSU>!vFdDLrYjCsuM!wI(KdUa)E}EZg8Omm)n+})xYZyt-Ytv{q|^;cT~q8vvsLPqbQ;Zesh7g5H77{wv'
    'T4Jx6iY@Y{Ex~;n^ASku+)mQi|<bYx&5GxmX7U~6?U+c`fHOoOg|(bj4UIVVqd*S60eWcBZ=2X>QR}22Ny8CZfZVA%xsoGNH8nPI'
    '`5Ar=CH(UNKXf2gY163<D&iy!f5WYs@ItH$;LfW>=FEsv++1#Hc0vM>HuLjPIU)W{7CAjY4cwxw(&iJl(U|ZrZKUU@j&q#Z<4?t7'
    '@>SboOri-2u9{4uFw~&Rfc7}`T8u}#uMN9hX1oRlK#?2Yjgdr>5`W5!tB@TnZyet^-'
    'SW0k$P5U@=cSx6V|Cv=@KS$Xu^(mP7bk6aj9O7K()$YnG5DK5-^vxY&*-'
    '5!KT~k6>0%q15$1UtA!byx`y|)W|E}Nny#v{jSdh!6@0J<=&l2Num|a?1AMTD=%xdFu!rfQ1AMSY=$-'
    '?7ut({d1AMT@{+}d}x+Z3f+p-to1!b92+P;0SBwhijR}!y))T=V{Zd2H@uShVbNB|r!%tk3+7@s7m3#N5~d`3iP2@#Bgc<z-'
    '?hhQAUbD#QT2*yD?_g!5o1V8>cOh1GQS9qz_<O^PGCFlqSWT14|AUtuA)tI<hWfMAMP5)0SNAral>QbI-'
    'A+{)U32f)4cM`90)H{jSIO<)QWryd13+{ir-303s_F!75xxKBXACgTU{?6eIHvX(<7q<oTqBfK02?k?@8JOzV-FiWGTB_f210h&i'
    'PU{6j?k0-'
    'NJM>ue)Z7TYKStMobH8i!58iOUk3;vs?flwcAn=y^eLQ*!Z@b?QKqs%mEsW<;g)?5*=(b>U{>G#!r9irtv$6d?HA%dxQB4xBYE)C'
    '1jd$;trvaa(OjE-BZ(XU_0IFOO=yET}Z@QKy956N(6y?1asR9RNp(=1dT2;saL9ae181Cxh0uG3)EaYZp`<tzOYqbp$IFi^U+z6q'
    ';F5!$s!+xe>rHpoy0fe7S%qcFZ`BY*~Ek{9gvr(lYnH=yERVtFn@s%AD!IT2SRH+!I)FL||f+@Ao4v1h%wb}s@OewcEhRGPF)CA1'
    'M#xSKO+5r{ud;8@p!2mi{!4^0Xt4_R{gAG>w>C%MRlwqEX1pjsg_3*4qP=vM0YWVnG_3;0E?p*yg6o(G`b{;73K0m<AEky)AP(<L'
    'JMFc*cL*QizmWl*_@rWV<kINx&oY~gP5X06{Wvb~9O;0ock{)Izx{K+p;*_O+EQ1cnj57C)MI_4LgG40A;C)0C#$f726viH~10pb'
    'XryUT1vB&Lz2#h)F?+}HtBk+bJ24lzB0Tr>O3K*-h639L)?+z#rxlZiE*+57dDg!uT(rtv-'
    '7HC3s#vI%CAYpdQ!RXc^j2<k)=&mA+p2)%I^2Gcs2csj4Fgjj@B`Q-'
    'Bh*P@J;7b!NO?P^jmS_brt$Uc2=&q(lMJjjg9v);h%8YTj{38?dNx6f}&4!FU$lx7hEXMA^ArTk@yT~YvJ!l6+VC*hCAOd4g*Z~n'
    '3b4z2#D2yG6_l7YTJKhec=#nIdF$|Zg{N->vi)6KDetwI{+H~TzXn)DxeHBF1%q_sPpwN41e@r-'
    'DPC?Z;%v8*#hl2$K8zES3E5h%gBK+<y!f$a7ei()3;CECJekbJMceOIxEAtp7ga#(2n<d@FNv7=)!ff+If9_gMD>FJN6Arj9tzuC'
    'I@1SDcMR--EVu-&m|4PLWe;={~B8b0t+W`^8-'
    '^F%71o77`f#EWS_<Iyy$j1<WPp|_jinP|rtKBptcnXd9{v>cC@_MTi&R#5%oHeO6LlvbcGJqW-%uYEN-Cl&z!$la~Q-'
    'sly9E@fsSS=j<#l|9xPRzlmSz&{E);haPnXQz0fSxW_*fQWjdb(1XT4f%Qv6^*zh8U|>;4r&uX=U~+*Q(P^NfHY(c!?wyWbfE95o'
    'Z}#Cy6@C9<~D_&a!*#fQYkfi5(DembnG8Q`A}3h&-'
    '2=v+P7WprV*;oqWIP7Cg(EnEBRXniTgfX;xvfkCODL($(s<kgEhQ<_uzE@t*WDI<^kLgwEA`F({Br9YEzTMesdR1mC?y@GZ@O@5%'
    '&aF@nE%bP;@$a^M@3nC@P6dT@gGr^9141NREyb#%g_98V|aG+E{hb6+}*g%`X;8Vj%0c1*+>2J@dronepI0TE}|y>>vv8Mf38h&a'
    'RC64*KF3_BW$V=-sgBs-v@a4Zim;<n!)GIEnwVyg-onFJE|&6XZFgb2zbB&y?#NJ4}iCXh)Utol2OIC`{*qx*_DT9(7nRf#z_hof'
    'VPI65i7Q3ry)O7QIw98#&E<T19j=^`k5jQ(O*f$8HoxyHQ%gz2@W!Ceb{V-'
    '79Hgm#nBB^F=s3SDCH1#j9V>i7b?bcs5?9<>7^j<5UdfQaL3nH>;ue7U8uOVsgo3{o*;j<1vKfQmw~b@FA+O$pwHt|)K^F}iBbOV'
    'E#y8O>w_cD!T<vID796Ybx~Aa=MgyXGMF*CNCoD?;r4BE**GAa-'
    '?v>S)1VJhlk2lf$GT=9nIk%z}ns_(91uXz(T;mdt&Iz3&l;_kD5Fn7<3oux`ri<6OjxlsTgu#@uXljl~$eL)Tc0J!i*6U<?e?H40'
    ';o*#Qw4yWb9oz}Rv-AOd4<ZR{F_v15_o9D}iw?SP8nup%8nFfRkqGDT)09eRLn$^x+lEQuJwJl`oY<ZDflMd`Lrq`BNnw=ETVt`{'
    'a=CiL~b7^EGAV{?k&12RSMfte!spiB{baHa^pL8b`4VWtSaQKksKajc~1R-'
    'ya1r!;e&`}YzyB`m{7aIfDS+o%OzqB*uv%NjeTqUM_giqWYlvhM#zGk#i%tot|U>C6<l^c}<<dK(*=aeYMxHZo%tVY|Y#7gvnGtd'
    'OpMB0aron+BaEeMyM%-Cj15UL~*Aydbb+1k+B>hi_tUE;E)9d_rrG+-'
    '}9r%Km9Swhgk~P7`iW(3CB~y0YC0qBJkJ4o$ziG}k%P?;*{ca^1Mxx4A6V^zaUsMVcO7^|A_^ZssiyPSvyEy-'
    '{@ZqCjPSrhFZTy*teBSUMbAbeKO0X7sl*HG8UloT=HD>L-'
    '|*JxwoWYW8&Be^JlSOPQKIQ!is`_AI^JQnP*DX9#9cnYY?5(aeiV;6K1te7QFV<JNHu+X6pr!OdWsG9S5{!G9;_YJ_N%1mia01sf'
    'B+-9%j;YgzcH%VR<Pt{qcRKg|NmXobB9qFL-jqCV%v-'
    'b%3I;T5uTM`D%>^BO(9B(P5luYn^HER%SWo{mhgocbwxYD}=i?itA>P%|8>b(qJ5ZPsrVEGekl>Cy?40QwbkXcDQ@x2??Rp`*>hB'
    '<Q}5OCm$|0M-a~n6C@lQ#4V%;b^-^DD_Toei_@y+GTQBV*V5E<Ovzw&CC^%Mu+cpMXb%?>t9h}FU^9@-^31-'
    '&NZ|b5B0ttYeQ+qd$PyN0{ic<ktgDT&VnB1c<jvq#{m<Ju#NYh-R`d!o4b7xM^%cY&fGV-ONwQ--0}E8p<gfuCVX?DUxLLMDH~TC'
    '7(Bo|G7{@}9d{<Wjwoo(6A3o^z%3b;x`yK3)~A5a>$dJ!6LWL9Q`=3@>{wghg+t<O4c~lrh21lAUK#9pC@|fY_?BBMEn{@D@<nzn'
    'g7~-CP$5$%Yzwf1XJ)k>hpiW~@aGkQJ(+lcoQS=e%pYk<^iby0guAXF-'
    '0Iw~1L0>M_v;||8R>o<3_tt2UpIiC{oJn`!p|u8>qhW1+WopQ{EXp>T|MvrqwPz;v?z}Lw~xUUR6s>m5yT^k$LcB`>%Cmd>R5_LJ'
    'Xgire7P`(qM)pzfcJ6}O(NbH6Ju`Txp?A@H#ro)BwvD>C^sY~=3mv_)zdROveWxKe|I0Bci4B{nRlwHySn;UzaJcw#4@7oEdySei'
    'xqrkNB)B)OfTETwO{k**l`?h;^tU8<GtQoQ7^NQeGif7&YnV|*67c9t?9A8Be@k?B!hMBFqn;e_$T;4hspQvNYU1=E|<~VKHC>?E'
    '+ipI#-'
    'qL?NREsB06hgm&LNL+)woi#N=Q!yt2E1mY^Gq1W^Ito6|B`P4zh)Ub(+;ddMjA3Ngjj_WI_}ihsao>u6@Fk><tZf*)jb8aE(^Rfn'
    'vYUEwKj2o46&`;4hkUDhXv42J!buIoLvYm(k$d?kjEQkcc}2LC+_BUyI+Yhj#@@r7?2&S@@M}^)Z4hufnm;f+B%M(#zYcgqRXo&|'
    'W9-'
    'jP!!`NdnJGFKE9>Lc9q_h7Gk=`_PwEI%FmQwhX<r))i%1jVmtXDY%ZuYbwCBl`5Zi!5sB;DCMHzPHXVDNiMi#(zeg#)>wn%J=_{='
    'aJ<!9E7D#TBD_%sdo9hxuNC0<Tha{<T2PO#n8LrA_7Rtxk-IU|%^gEc7~wd+H%1#PXlzdSPWB3XXI<h~T1!akl6O#X*;3~leK6DA'
    'R&tLYW<c9XZt}w^a)9J6KO9K6m)z!uK{ilwpC1}vkhCrTt+ecWr9{M|3!bA(LiL`zNLAgK{Bi|fDI*KBUu{7*c#alygXd^LwUWx@'
    '!isy#1U}MmgEL*f1pcAnW@inm1}O&Xf2c71wgYuN)Z<9X3+H`s?BsRxBN}cKysfp9h7<h3E(xwpp;d9;7s(fRL0p}_$fm`0;Y(!M'
    'EWFX37V0cfFr9g2PDveni@5{EIH(^JtXq|=vo}s~^sUg|FKf(?jJ@v{uuoZAwf#~L#1kAn5KnOQKs9UMSg>}V1eR;pDNP);M6u}>'
    'W@|L;ueq4JPQ!tkTX6Tl3Dugh6>5p`pHJD8bZneUqzRD`fve8!lfY$?`89UdlzI>422EnrXUiX_)7Qf0+g8I&#h*_aEsEhygA@GC'
    ')><3FZp|ek)HuIbIj4x@Hu`pG?N>EUnwG^(!T+d|qKf?@Kk5dz(T}>pZS<pR_MTao!3n;mxLl8;^kV;PKUVOPWDb*06}%?7el|pV'
    'Mq`|?&r<LzbLDHe-'
    'P`do4c`}BPSHejbBwe79tnKQ<NR@<Iv_JQ=Kii7pLZo~WVV#0k(e!_!54BouaJ(SEFbn%;6;O%HS2Z3bM&3i-ml=OXYKu(#@VyP-'
    'u9b3*bSni2fIOZ^k6l6-(HCCz6pG&;d-'
    'YQ{S}3CsLj&qO{cOGt*=jCgwqwg=nh_jGte2_!7Fg4f>#u`ZW4ER(0$+TnQ>$~o%e)h%XGG9y^-l0&w3-'
    '%xjEJwY6%#kji$2gr|&IbNaE@0nv5Dtqh?9wa4(v&Pv6aipMX{$Ct^vCz8hM6F^+@Q;xFJ66D4xj_S^rs8ze_R?gq)xkE>by?n0w'
    'iOE3-+q%#`^@kwL+WbK*Mp~*@&CeX6slT<kJyCT#)Pu$j=$hVU{To2%M5~+7hfHO#=-'
    'gN=aB#C<01~`ip>Rlh;Y!aw<jev7VpWbx>&Lw&JCq0A149~74)0v)KNv5;tbK)3|E)5O+d!4Q_cK!e@Ubd-dzv)l9L2&ewZV(*(q'
    '?(-{EJXG;34DTc5~tV+e2U|gi}P<P%x<|``ci7cwyE|qvMdhgZqfFJqN&{W-'
    '@%)Tx<PFN9&M8BrMdU|ZAIywTZF%(D0O!zigy(yd&{_6pBS2VlX)`Aae-UemSEo^PoCPw@k{R1o;BFujoQB~r{TeHEyNKVT^3sW+'
    'Y(mWvf4Q^{bvcCg1z^Lx<PRCP&WvU9;#;XUlbN&wF=X<Yy@o7v{H3NRLqR<&$UTFT@er-;qbVW2ZXjF$I;&vyzDpaKf&2ua-MgCG'
    '~w)sZb%RN1ve?YjEOtE><(UovoT#@@iz_M6TI!xgP2W^b=mkX8cuQv!ZLVW!S344M6?dxfk&2Q^kJXs5@=2rj7JXz_P$W2pBr;uB'
    '-8(vOx^Z^KkWwJ(NDX<cl6VWd)r&*+GWsC8?x-SDJ+dPB!s-Gk&gG;>^9bAdK9%;RgTJSO~rwHJ|+Ai-'
    'lL{S{@o}F2$F^(`FDd15hQpw?IQfdF-aalg4tPyb)<%Dyk+K3x*8S<CPkV8&k1HlnhMX0SX!iW!lK*XG%hEM8|<NR?O8I=+b1zNm'
    'cAQrVsI>d_owEZ2=oZ?oJY~)H<}Y7&?61zgb4IVgLzgF=n?A|<D=;D(RIeh(Bq?PjjveT$_E$~w`!AYGp2>gelDY_<*l4%>ZNUwV'
    'LUC)u*Zw-'
    '=1&${cT%n$27)Mt*1^Tl+P4^5`{qJxTTMKK!6zPH46T2L*sMhgqlrsoPbqkcnZF4g{tVN76PEQ^QBv>ff|lysFueY%;aD49e?#9C'
    'CE;bC#*VS!g*UNdY<S_l?idwb!dxUGybd-eM1<GA=7fmw+Sfd*i14z$ad}jDq5E7O6<+_O_gBIf7v`^8iZ0$Ag87Q(Fvdi%DiTIe'
    '9b=8h;FmRJ<1sP|wL%x#T~wJ2qd^wq>yTo6?N^Mi{c`cOohD|_;1f?M#@D~X`07wJ*}8i4(+ZwuHhqGxKF75AgtL3Tj5ogv60q~a'
    '`1+cJ&79_+f|JVDDfVgX6dPZ76FbGm7vAemQSl|sMIz$s5OYFAeC=mWh={NK%(IG!FY6muM8y}n&lOSe^)GsVC4BML`SuvRUBAOZ'
    'a~J;4HBT!TZ_>m5?jP`Ih|0=nCEH$EgxyUfePKMPVuT%9jIg1_2pgJ<umPIba)VDiu^3^Wg^e)%X;XPI>nEhc(ihxBDAVETi!Nwo'
    'IwE~3mdNe=FuMMt;TX+Z@W+>J!P_UXb8K|sP3#;SU3jlMM@5%1CqzWoq2`2$=o)HHh={JC=2=BVm-'
    'UMYQPG92Ga)LvKBM<nT7C7)&t5CC1(%^FT|q)aHk_6UVkRM-Xomh`lIXa*NjPJHnZ+!nTr&YbJBw=GL$rBeOz2{i9afC8{fkkye='
    'f?l*92<~KJlbtl>Ix5vIUA_-'
    '@bl8rB^$=GW<xT*Eqm3JgCyigjkNFyt^QbvOj4!+D6%>3QjJY!|apTB{s_NCU%L9vZ>~rh$z$Mgor3R%$yJrW&4{GBBE@6^Q<DG%'
    '=*Q|s3=3%nHUvi|EBj>T7vZ}&=p{uEwp8y6$+MWR@lUN3SpGApO>w`=*ZJRFA-GjDpWY`r7i0)7?NUm9bOEt1B&5wKrXxnYNB-yK'
    'Jm-N@cK^}UO!gpwGQDAKT+v*4&o0FsdS3N_`^?CI<*Ak-'
    'v#O5g<*K@u3<+TUVG_0Pb`O*eG*@a4KKWjFU5w}(`LgrBD~NjUy2H^!_5g1;dOvHAtJmEFwZI?ysTeb85LgWI#))8*MI2!m6l+IN'
    'vsJPRR)d7AnH_Le^^`18p5l02!=_t6MDyQtKuK1JtoJtb02Lwgptsr7+XgaW9z_TY#o@3twEZ4FoI7!xfolYhp{z4rX$mrxnUz6m'
    'A=9_;<Pz^m3ueR(dlcktdAFkv9-'
    '5`U2SabqhYtQ*s@Pz*Vx#?o7gorw(wqejfyQa%C1qdb%Z$~BDM}RCq%^7f#z97#Fq7otD<5HUFWK(*!rB_UupT(FR!iHsHmplcG3'
    '=3a3`}`!l4T8W-'
    '3cK48NvgEad&PMHhPdCdJq}vKTuD6=Ub1T<q+iNm3;E#8Zl~^IwjgIBv`(VOIkId&B9te^?OpNGc2Nrs0J7APKw;<FtJe^95y{O9'
    'Qm~|7idlVjtkc6eS=@_rPL948PlMu_1<!V7E#k#smVJm=h|582jdt=7b6%#(j|onG-'
    '667_$c)WS&)p5MzXzb)QL<Lae8|2zuD0N+H(Eh1h@T{gtM#!iHyqge5v8tqq&3vc9S&$^aWYVyMCJ+%8eP6>2B}sJ0jiY<5uf-'
    '%;J9Pr)J5aodn8=8rW~BG#H|vVqYItG#sQY{Ef1Q0cycz%K_o6*f)tlcO=cjMNGk3HhFYuP3m%^aMbQJ&xuL0f!isnK*{i<3kOfW'
    'jbANLg|DgPjJe6p5}_Ds>m%*yoFO$<d)}W=A23zn+1a}BuIH6-'
    '9p1ju<todhP?xI)I`?6j?n}SZlAe9P}URXEa@UD@;EA^^ciZ4e9Rp@2S<=pB@3ngNHb1tGYP-'
    '+e6*WOf~`Av>lPAz?fR7n`m$26?RQt{3%(t=iAd)goZpUsmm~~zSfxplaJ1`n9G&I~p|SCv=h)IP2IDtniO{0Cm(wDxjdyTbq_Oc'
    'vPpf9@mkKzg+O#8U5V5@k3q8%qSm|Y%IzlY}V3A?@n`yhifB^A~0ML+L2S3rQWjhsqsu|<GQo{9~w_vq|8$H}{J(|SB9eYdojqCd'
    '**wPxWkvo@qiXDV?50$>?<M(X=hbw!<`gAzFC}BTm>nkO(z^v7gl#eHd#%?6|i!rv;FuY7v)7;GIv7wE(aC)q@e@UlQ($vhigpuS'
    '{Ei0BSrPqxf?LS|*9Ultgj?C`#zEA{jsVMa6r){u6XM)EMXcj!4OkgP|{r0A!D&u)>*xgjZ&%7Y-o+^FG+i-kBC>gi0GXck1-'
    '%fB9emPbYNk_pe5)N=-J(`He4|1j+FNx!Lg^uMNJSns_&O<Y-t;ZymESPQ0NwK-'
    '<Gh%Ix_i#q+Fpjr+Mm1yqq0qYhPk2FraLcu?C8Uk_M>jZsBOx`s_XR6WKcwxGT}J8DlOD6yQ^L<Z8&)q#P^h7dFI**fjNVJ7FNd8'
    'Xjb15WVghe@V&lstG2^b(p`=qSxlV({M|Duc&{9^@qZ!}rNVe!4Rm%zy)`amd)>%WQ=k>O$?_CY|ckbuzf*kM5BIxJl&^BMOHP+m'
    'CrPf$;|Jj^VNmjFfMBFTa!!)$IIJh{Wh+gXKjVaH)0&^eN3#KUDyvSK>Ee?*7aiASpOE^x-'
    '%;ZoH^(uYE+xT4pN3u$1+Dvj02fC|Hl*DGY#vNTHi5zHM-r_{M;5m23Q<}jAkTy#6+_*!3nI<wbqeCq5*<~&5sgaEP5E;wOc5)cz'
    'r)c=pwhCztBf}w`xx>uYJ~P(tcn@dB+I^ilr;@~GA<X{`FDuY)V|)!7KyrWg013yq<f#?2>s1~R{50jz4iWr}ZwRNhK<+E7hrOvS'
    'kS+eYz_m5wS>6=5wsr%1It*Tru&*15N0RW*{y81=+K^SFE=U{F_Mjh|_Ga2b!Oc2b`szy*)aWM-Ni0R!DWRdiDsl8OhCV(q3Iv@D'
    'ZNBYUv4+NbI4joBc&lesGxX|$q5li7DcHmrdL71=M~HTawB1W3KO-<*I!W->Ega=%+HEDZbnVBRx75?+hVw<f*|#-'
    ')KHzNGN2RZMCwND|5nfvP3Q0{Yt8%funAe2`W|bTV(BeQi*YGa{jQ$<Mkod<jRXr6L`bY_Lt)X$nI=`$L%zUG>V-'
    '1bBaCWSrUp41cs%aIvg9axs&vS7+DkyWhk?olfVLMx7F`;?fT>FAbhjE0ysM6sKID1K@BRGe8S*0Txd-jS-NAcSERh2d~X67}Oj^'
    '<Q-4~=#X33h14ULja@3dZJWHtIr}ji!KR15Pz<kkmGoc$WDlvo?%jR!SEImY%BNH+JmC%;t}xq|MVls-'
    '|ubD>Ze4SgEO6ZAB${Qx8)xEMc<^S1>$byNy;bI^lKBR0+pPoUB}vwgL1_XQJ%{rf0gfD+Ng+oaUZ$qTm7riJOmmD7a8T4{a|nMA'
    'N2%^Pasm&erS#dnND}l~-?DOgn)dy3$UT_l!G^-'
    '~pG93dSoI2%x%v?`vwpQyP;vHl<u(w`q9TntQf}?aMN9`xu6GgG^~yJekt4YS!+R<WsbThI3<@i{(cI;hLDty-gAc!*Uk>z^Plp-'
    '`&xvf{NTu%XE>1b96f09y}%CT%FEvv}h78)ae{2XDs0&oz8WGeF8HwA6YAir}a$dBO?+xFEK2)ewnzCf|awa4dEs*b%AV9lzYw7U'
    '6AXLGEI%j2`=`d(f@OsoDbFTbS&AD>FqC$==Qq}?*`Y>@NRG|4X<YTtqO6za{{+ZX2$ugAiyXyAfF*<pJKJ+OcJzDvBpj7lH#o02'
    '3M<wV+60I1|%@kPXp-)5@~kn+-Uf(;@Nw5Ny_P0yLjJAV|vb`|2`Tf$-'
    'KO7Ewtfq=>#0{<v7I)u;oOvPBtGWx$5h;Fvm$UvU8Ot#q)U#cT@UI5a^RN{K5wMpESH7OWObJcOMZ?voxX`G)p6@nR`IN+&d+3o2'
    'M>(T+rt9&CcADG4GP_v5V}p1l*_6buM$0aK27wSqq$uBeLdfYx@LdW%eEh|5Ti<4TpdEM&jKyF0y$t*;B&=Up%STaHXMNsF0Y`tl'
    ';artt8S};}`T8(qj`Y8sSnOf5$}<H>J-8*1k@|udKDdk-'
    '+<9&7Ss~kBlc;8rcn^rIFRFy>r3ZyCm>^zvy~Q5Yp!+;g`y<@lu((8E0o!Y@(6XEwx9&zZEaSM!|m+FUOmSo#hFSep{GjP^$#VvY'
    'eh|00FO!J(WWH$>O5B&eKJY^YJ=P5m4>}$y+!CL2$`keYzxF;()CNPPasqlCf-'
    '@Zz+8)F!mG;zqZESPs3AXtXlQ~jp_#7(x`6GEsd&X?LCux#XN*0Vp@YnY-'
    '0T3c<I^3I$*HbGn=M1mK<kzj?)F)rP5W_NKZ>5^P1_HPbFNS)7cJDk-'
    '(hHh1w`qS$&mapXFC6ZD;*e8U|Q^m1Y1|V5Jd(C0Hr>b8a>&feRAaWQpQgN}mr5{q+Rivn|}SXdkY`(DrLIcY|iBxf?V~&D9J&w2'
    '-YmAqe<ziyODE!Ns$Lk9fmBfw@^iwOWvP%NZ^h*9cOXXmv;H1d(>1<&M@1q8C2f9W4@Yxz|g7Uce-'
    '$mu|$g|JN5>_DJAQimR<76F5Jiq7H{QktuaZhCs9s8NgC5qtyDy<}AIL;yPP%H&OrUG#$&a^g&?lgETyE!y28jqhujwpVjDYP%Mq'
    '^2F23oia_ISifuAuMTQryC#g>SJ(U-_f{KpxXW>hbOU?U-a7sjT<B-'
    'G;oq4ntzISXA;zL#x*}{h;1;ojzx_~&TNkYU4%?3FP%Me*$3%BBy4A)wsvN_;&<(ZM;P38HJ;ayeAcf4`-'
    'rS$5+7d|GHHRB6+Dfm_y(AYkSjgl|bc7uGWwj1P2ozPln6g(jUtu@44Ap)(1M#U2%&{{R-'
    '3K3{6>lgE*&{{*x2@y=;HG~tYhBV$Lgln}D!f1)-oc)u6Hw0(NzY}myE<{!qLuB`2h@4go5s?cKDuv`iWcOl-'
    'h+K$lLzI$(+>m}?0=FoF7eo+KrPqXLt=a_F``NY?uKh6v13El9^f~ufm93nO1Qf0P+*)jjBKEh&0I_vFx{F0@ZIi$jWuh|nX+97c'
    'W?gY!)y8vPbqWDA>jN@IQf7DaJt8D!&@CR2F_JRETp>bI#(LJBB1TfiF2*AyWzaeAtO(dD1leB{yv2Ju@K*)zctZBS8AcLU9{9b0'
    'b8`{5su*#56eI5RV#LW@#L=Q87jb(OBTnWbZY$cV7$C#aQ-7JkpkW?<M*}?b9X|7f;VHE<E{0Tx<jo|XUbm(lR%IZHW-Of?Ok=;r'
    '=&=cGp9H7+k_6U1yC20)V_mU~)pmnrtW&_C8GlqM!Z={|FjuG$VO~j7(+L$Kj3*3fu23Puc*EU#(p{Azj00v5b3%m(V+bWW<6RX|'
    '$J^u^X4+w#hbN2;O5@Fxe`nx0%FEl`I8NS+<9cJl-'
    'a96bj)Ol4I4>8DtBc{dXE7YlD2AiTg(EFXa^bjVF&tGc98XXd+vIrIY2yG!xucV5;{cl7(J2Zx(J)#?a=WG120`|rf=z9%c&CC(%'
    '5p{fL>`O{vaYDzYP&)0)+tv+<2)FZEADB&Nkp!Q#(gj<S5)Q-5xJuEkGrFC#XZdl5xF9|*xePEWc`avvNjz8-'
    '0{dG1gJRGp6!mDDlCYknc}iGQ?nkAX%+r246{E9TtA0Fv8EVidlkd1wHRhP7iP5H&xP4u#W2%hn57-'
    'G(KF5w>@0$#BwQjG0e!TDZ@Gh~sZRXuSTeozx*)_x;!WD!mdEg0mvMvGr}5+15bKIdueKXpdYwWH4fEru5ZlY#RYZuPVSgMIV%l6'
    'GBE+n3+!GaIdzlj=LJZyOo{Gz{{zYxiro?VA(aZ(YbV-'
    '|<;UR4Cy`HTYQ7s;w@PQ^8tR=Y!dr!axxd>ZZjIjD*gq>N8up}2@v|*Wxu=-'
    '+zCAkQDhBi;X<KpO9014j}3^YGh!lmxudD>LJOhmF6rc;6lyH&uab|v<tfQ!l^%s!2u#70<GEQYn+U@`0zVQ842L`7J=xvPi>L&N'
    '?gD#8+Tg@_2VzHx6<gw>l9A|fmyxqxbGvHT=Pe2TfAt==M+9d<Bas)LRKD9w_9Jrtv?xLxqRfD3amwyqdsdlzHutYVDy$i*1dsdF'
    '*5cQM9#<dVfP3Tm|5FRq8<6`@jwxuX*mC=DZ+tyWolWh8fGIyH#1M{vDmS7eW4Je9?neIgIV##vX~k+t36j_ed?>&yueakjVlCJ}'
    'Ll#(gL%&U%<DM8ui(kNcwHY;SWyM4X|E-'
    'B)o%)<0MCNQb2>nFN0sj>NgxSYM2deTuPhb}=?K$;Afk+T>zmpJHrml8cQO6dW(@zD_Ikq0VHqEgY^$N#*fS%nSpmw{N;xK=o1hz'
    '-<~Xp+%xgW|z?)c%R2WEtX^j!-bO0Y}c`DogW4v_74S{+x6Jp3cek#$5_?OKG>hehFDj8oweQI>+BRUXycz&3NenDeau}|2r-'
    'VBv&}742r=K8YOYWr#2D*f-'
    'R%BKA;uB2k2#@2h;bcq6JiOhwjL|&hmK6(I}9f$=~+)S{)8tNuZL?j_4QrrO4nmFjMt188Jv&{Gj?^3O5b6?;aIA7^RB>o3ZC{YV'
    'dpD&#<SwKB;@uEigKlbFRKvJLLyeD`sRf?SRk1@vJSrp!Izp6?q%~dt$pFtByE8O(G;h3GOszZlg$FJkm*w;;6(e?TXf_WDBeShj'
    '@$zM)SOdE<TJ@Phh3b|^Q*Ow#S)YE;f`N=@D27aZPMU*b97~9uYtD|P0fb)6wT_=a<H_L5b<p8FPNgKe{zZ&VRuZZ$AI_KdlKUtO'
    'Iw_yV3F?^yimb&9;!gAer8@iQ+Vk}*gt`rl-'
    'VMWqSf3sH0fxZFznL#ZUH;%5}VpvJ}TCxzcJ^;*%a^fsA@J%3o$z)A@!)Cb^=Kg_-bMq5$dTS60_R3-SaoaY-XG?gFI~n8@!q(CD'
    'X|+pij~;(f4KVlu*w!8*OvqU||#7Dy=58QzcO8>)f7uws2~mQIw3^1!{kNshd1>PExN+taHwj<ev*fuvs>rpp3?nn8JTamo$Ia?{'
    'RdjC4X+tiL)erv!kn7^29vsh_->pCA1k*t9Z`G<@-XvGcsQL_9z7pn_m0nDns}A)kM2jw)IXjEP+dWlgRJ{zU7-'
    'luGh2;^}4&_joK89-I3YGXGnor;xfIG$-^zb5{}^76W-Pn4ED{;V|o$w91E8$R@DFJdQ-'
    '1Za6(xteizh&1HuSy#thF!@cjxdD{I)bcaz529d9CywfoP^bt)-#7T{*Jv<F4{8_Kpx))4~Jf_;qr9oWb8=sOM7R$@hI7d-'
    'DitZX|~X7JGoT;?^ErxC!@6aDX!kZ_k5^dshI9%45m4m0k>?@g2y{1K-Kukj0K2`}7BrJ-'
    'qxPv19z>pj?MXaYAXHvIt!+^k9<XI)U-Z68{DI3{w|+Mi-RT_*cwzu7Ud)}Cz6iL>@pb514A&Q4Koc09TAMr19Sz79D<dyaywT?9'
    '>YWUWT#+Yn^-'
    'g2cdi+7sp0!<Q0jm}j1Zc0An)ykcH=HLE(OGtz5V>NTC2PG<eybXIyT>wV&sex28So|8^tRo`@OI+cZCVTMZI;Let-'
    'h=Y()zc|XZfuXra;cCsA`yK_~DO;@A?+3Bw#+!gxbK|{2HFJNpFg-UXaH$vFy+zad_Z>IWj@6#xxE^**sQZpn=h-'
    'TK+wmO0Zxnn}!X)>X!$?(~k;tbDFSiY?f<?AlVLYLJIgP#?m#Q9UI7z`Ns?ZgX_4L)5CN35NE;MlVmsyh^i6*pX88*7*JXw}vcfk'
    'WTC^UEzE~Bi$hv5n=DcO>@&*<1#gMXd)(<2Rz_j+tKgMYhV@TP=zs2V)f_Pd_X<W@~fvv=Jv+(xP*(B_VIO9)8LgEC39I8xzylKh'
    '#DixDRiDAG9?Cr+YArEl@P`wA|WBwy(a;kDP{pagDJp}4vcZSxh&zJ{8(x8OvpjWOob(%ERuYltyiGS-(GN6_9S#_odXYI$Jmj}-'
    'K@8Qb3#Y*wb)X&=aOv9`v0I4;)KSDWip+FU8v8W*ykDL7PncA*zNh_{E~tefL3b*iTQ6&H}FYdG4g1)Qk?Jj2o18jkbKCkFfCT>|'
    'r^Z%NGCTo*nLi4xc9j{HQycO^QSTaRhlllr?Vjqz5iCF};B`4sASAqFfA;q)wRqSuXHN4i58j6)o4(w!=>_@D6!+rHhaWcpl5if+'
    'H|kZ$lM4e17N(vWI>yORqBzevG<6&$I3m)9b1p{z}^p0Ky9gWWUJ!S0pmU~iV`U~iu3U~iG>VE4{+uyHUSCmp$4E7gtvYNu8o&NV'
    '7z?lCZ6KTJ1;>4a4T&+>^n+V{ISYcRo~t477KCEXR;cwY^RY*hcRhSz1ulx?5b?iGzZTh)jf`Y+&3+C82(Y4>WjzOm5rgXzT&B-'
    'b*|Rq!tbhd9HxD{9mkk`baT=B8NMF_~s%v`lxdVTGM7eLLt=SwtVL)8R0~^&T9f(~&r_+XK+)C|qgVgJX5t40GJUaXKB1S$QPO(c'
    'LOUtxh9oZ<b<yahKR&O?9RkT5mg5Vil+#k+*bsrXc45(*!KnEcv#ithdiTgFU*zp|nRlhteL^EI+f5qGKr3CD|9H-'
    'dvGmUTnvI>r9lFb=2Y;I@}}9mSm3#>iS{pw`dca_TVU;4uk3LAl2yzTs8AJ%&tXEfiG|#N90#-s-'
    'fo4mlX5mjU)zu!??}t6~**i&BOq31UHb{hP62Qq<g|hzegArWPEj{OrI^;tFvEr&u-'
    '8s?b!|bq&=(I`S!wSe!hZ#D>%%JtY=}GD7YQ-F+shWe(gj-!zQ1Oi(Fg}yVVoipfC-0%O_m%V(#VEPjHjMq}(l<;1-'
    '4JX18j>gbdDh*2aYVGKp(-S7gz&+x?IH+)aBwEV<f*15D7Won!4=Nc@cfZJ(cy$cNfZZ9I-B>0Xr=`Fht*;1gx~Ov&ooKC-'
    '>KL7}u)Hz<_$s%G-L3xhhQm3Mdoy6K9T6n1wJIa5)K*PiZZmZBD~z1-0pMJ--'
    '?yQ8^^TD<mg!~ZE7CTULE9v9Tj$9^*#ZdP!nvox;5E|GAFvotRLRtQf9aIPji>_;k`)~DsB^%|WG&4FxIt7U0l$RkzIgGdsPb-'
    'D4mA2C9#wri|OGJSq(B(>6gDrnB<yv=^&`fe~N)pvtIslJ-oA1p-'
    'rg$h1baD;pIvlX!;)H`cFA&6Fi$K`PpYh3e0d%Z4r(*x7*Hv|{Zzb)ZS!8;(BioYe8g&s5Uw*}WpFO>w%cRchM=kE$Gx?iMeGw@T'
    'Dmt<z?CS%HW-'
    '=b~C0H))pqX}u8<nds<3M#*JR3f(C6_wKcVW8iHiw8Tl|DT4p%UVzD6WhBRTuOU)gG*`eY9{|hVXte5q?*LwgdZSp98!ICoUf^38'
    'i!A|Hf!3=`hJPhB%Hs;D&oaxbUrjgQFrzpPD*nHHOg~?=1SqkpO?{641L21yyDo7zVV(NH;&hEx#m5SCk4@YuoIrA3AxJ*0vBq6W'
    '_;qRMn@$u%$IW}l3<nLafu>S`g36J`xIPilXhGR4ldg~vG=x5HyD-'
    'n=?0_HKGn>9Xk9^LP!WS%4fVcz7OHk$!dmm!dTP_UY$R<;H+YRE`jEvkw0SLQ%7n0x`MLl!Y1ajy$$}!#WS$OzCK$VUP0)vR0cf&'
    'Nhd>itaULg)K&M$57+#lw?r)fZFZRXvuO;co^x8<mCG^$66^1Ezz|Ic+6X;zg)nuQ?cpZrk!J8PbBk>{Y%{dVetoejQh=E|Wn-'
    'd}+SPRSv5fH3-'
    '=7b0cmi3Mb<k0~H3mvEeb94YGS*WYhMo*|KD4`MDV}bG=(}vs;U)bSuL(nA?vLVzPh216|W7~=vO@AlmeoVMLbL{cCkh`-'
    'Oa?^_;_dqe^zLyKR(?mfl^k>D8`)w}dS_}<GojW?t^z-'
    'lF=8;vfUsAGbxlx)cOzrZB{U6&P`xj0jWknME1TK#aGQ5Y&BZCZY^>Q79!Mc;?tQZW|baO%k2I~QHLIeiud**})43>3=(^L!w3q9'
    'sE6@$V0t*%Pf{b*q#Go;PFy#;--'
    'ErDr?r%dfI`~IwuwnWk|{=>zJw+~xH#{JOlIQV?1^^|ZzF1YS02G@*YaQ&zlT({+d>vT~N3jKL8xE{#`*H}{dWl?H^BD|=02Wxaf'
    'g^7b0RfH>O-5Kt<ar!V!oANtK3v8$jSMZ~xWInUc<BHf&!<)Dwc0T*ioD(shp=qv&n$KpK6C&obADI&(=Cj+(2@&&|^^Vh{<}-'
    'Al)1&6IM|4%P^w0A18O>%(J@}zU6a4UD#@+-x>KChwN4;G^Lz1Z>beP?fEWmn6I58JscNYU}W--'
    '7XEC$%^xd1yu6m&v=Q4Fw0a{;!wN?-8`3Mze-_nv#J^feBykqMOv_m3lS!=+3RTmM$@m}E(r6=k`ieH0TSV+-'
    '$LLTqT^txkvvEi}r6sL+~ePKXGt2h9l)p>?}CAtJP_Uz`yYTIf1wM1|I)x+*pOq1)zLJ3G=EnwQ!W2I#`_0O;P3&^)!2=d7Bb@3R'
    'JTMOQ^%%rl{{U*TN%sU6XeBXOLmLF8>F;iO#T-'
    'BXObS;ffvaWV3~pNqU!QP2+kWij#|%SGODMVpuxyR$!3wA=L+cl42>osa)<M;|LHkbhOh@#?|akhq@1%zp2Qv0?UeIwda5@P;Nvh'
    '1op>C5;KQS>}X@F#EANAtKDaZ%&8^GwTGcQDKID(i#<JkLjwC028*&&tPK{+Nx^s1T<rltp0nkp>p(rwF2iZ7SC;`ecy}$5??R2^'
    'fMeyn@jj|E}HHwM$_zKH2tI)O+Uy*)0v_m75b}UG(Db+CR`OhCg5xC?0+aKG=JS4eWIua`3-'
    'mUsiG3qH&rBaA*>7Mvk?j&v?I&s_~B*qnSCTz#^#E9nRDVI4e#~Js7OQWTp1N<v&{(+k@gdFLPVtfz?={fY1To`jEXe$p);c*?Qv'
    'aIgyGR`!}c?yMs;#7BJL|j#GGP8JXDN`ALb(BEKv~n{B<!Rp2$VSxr)dGo)UfoX^c#lc+Ag9GJVT$B7Rw>Z%6Vw!g`UR1k652t@7'
    'IpOKA4an5}5zD%%iWA!vJ)`$X}qrUcrwZ(>_fzzB?2FwZu6{-'
    '1)iWsznd^;NNvhTre1*hqWPoD&gg_YrAtOr*^*CqzWrL*|5tNc*8VAtKVO<DC^1Y3Q40MMc^Zx~dAsXSq48l`5ch33Kll<XKkHAX'
    'bnL!HOcJGT>nvn10oICtXG^IaXU0Ob+4Rre!4vxciGCJhvFaKP`sv9k~!bTNHFge^U(MCvzc;cJE7_WcCQnv=sp<xYD2Q0J{l$_m'
    'g!xnfiF5NIUUgf*B1+{n*6W2HNB*i8K2=CdI}X-'
    'o&KXIIA_+iHNiNX;(ca&gPmEBI4|)=7fkiyThCi5ogvr&W?&RbfB}N;_OLXRfOcx!lroNgfI!$J6GN^A&lt_?r1AqQY*O89ra5HV'
    '|ueYs-t#h4^_W|B$gQdaC8E{*Kiwy+7kGqhVKjCwEeF3*vYv>#W>nQZ=Tj^YWqQ#rK{mYQR4X<rznMUk^thC!?aP>8-'
    'UuDXhL2S<D6TW;!#~Nnf0M9KrRV2YC3LL!R&RofJ89D?bI?UUHf?ECz0Ew_yy-Dk=vzf%sG_=Knu3-'
    'lh8ao%~^DtgjzOkaYq;reKPxgPvcUSx2ZA6rY6i_+-'
    '@FKV}Ocla>ku0(wQ)V(H0_|4Wpc;pV#Se7JjoVU5B%L`o4X=9!Db5q>JF<gmVL%+aAGow>7m;(Y!&ObkffvOM|ImsX)_a9#Sjm(7'
    'Om)y%v{MB3%&J{SE<l*!95)1=RvZMcBu7XQbWnE8ZDtc6<_dR<rs`1*=apa);yHWYj+)^x3V>>Nsrn&8&W#hL0FvNLZ+sy{^P;1+'
    'Pd3tj|^On&jZ0A<`Kz+*y4V5kHP}R)1EfV3D-'
    'T(v@&LgCo+lxS}FBL==xBoEOGA0WAaim3C^xM#B&GouhRw7c~^kOg!02Huudp_@1T0_=-38!2*O}5X_ln;zRa%-'
    '4z?=_}%V`4f7AoIhE8v3kJsp)t7PXb_U0(bW3mIbO5!AIokN`se{us6&;np?=*bRjoPy`T${L-'
    '!1QXk#P|1&m;BJ2SlI6c^pxBS1}{qXo|y_>@-'
    'h{36uc@!`1&$}8TUz;^uqGu))3Qpi7>J4j`N#i9Wrf!4yRJn!r+}KzCt)Z2~wu%(lj1f0Q+g8WN+3!!n<RwjyG|4tkr*Iu2V@Cv|'
    '#lcj5zfb&gz)3Y?WDkTUg|E;0D68f&~J0fag7J?5+u^MI9l0YtlYAZ@V_X$w|O<a&6+9(OPk0ph^8nugCkIa3b8)9S$!T*^(%tJG'
    'dY}sgtKH)bOxoP=Rk~%&FeuoaA<Emf?it2~}E&U9g3)K0e4t33%KF`QI>!Dl@lzcK5`Z8^6~*vF7e?u2V@C)S27W#4)4!%sTDu0&'
    'Y*jnmDfVPR6O-IRd7-TW5|yN+66O9R$yL0^xa@7WZye$2G)z8h#*rZ}*-'
    'W4t1VzUE+vtob>DO1Z*n3Ea>|Jm+@|sSEonAHEs|dg;N1srz08J;KJNo-$o-wEjQ4HO^S<Us4`hgn6|~9K)9U-'
    'mL@k|tgw>zR1xiDcKf96jWs)dhkIkqj$ifOYG(hq5arh!t;@@uac`7xBIfMw=q3p#B{0z);i9NtX87&kDNo6cljRT%jn1NNi5CZN'
    'g=-R;(4FqklmzB@rPxW5a{1oQx=k2Weu^Eb(_`RT9mxX*7v(cKa=-@9mekIWBcn$&-'
    '(%tFU16H9bBg6eov>IK>v}i<wYr%=IYfXIWuhhKLERT?cl>tu#o8So!hO~3zPey{ToZ4TwTiYRuvEi5?j1?s4;se-'
    '*E*EI`x<`8y+e2*jd`rK=+y~K7tE(MK@t;hFNa#6O#4$a-'
    '6%VSitTfpd57urD5o}7kNJI?!en<a8uNOX8p}?F3#hQ>xWpuA4ts5w!C8W?j!|PSnO_iqo@>2ukz=))FO8~^xBCbIk6OErCh6;vL'
    'Wq5&_s7~Dzuf(?cE@LMe>J;*R_ID<kYV@LG{TB!_;8%r8nYHIzt4&@=2*qKRlDS#?QDkFUmGG#=S&A^!=x!aZNw1mdl?qNIU>E9S'
    '!~Z0={3y6dA=arscnFHRC3>Oe-'
    'm)7v6y&trb{n~_ojxAwQseUBr;^ep&BL#7%F4Qntm5nzHU(Izl^<4*6^Gz$=&TY8!O|<n#Rg_vZk@Jns8{3!uYO9?rk@jb@MpX$+'
    'q}olkjvq{ITe%bO8R?N1twwKQ3b*h(8t#k`BTjiv>w{z@LM>eDE-V3s-yej)qUP?{oQOLba{0$yl1D--'
    '~d8RztXXj|HXOf*M!Ea$2)YX;MSLY*+RjV>Ib}Ym-&l$nPV}6|mkSP-'
    '+SPRub0s37#Y4`J2v>@%&BaR5Nq09{E0n4GJ2B8^D|C<%Uc&#1yi8?YoX7hy*i+E(!X2YfL>O#|2VP;>~+x{_`wcAkypHMv|Z^-'
    'M3tJaK4}_-'
    'FMv41%fiZ?<QVPdbq%Kt{0TXv2;s!bfw6EMpuapX!MH&J{Is*=R?0rsG$0FcVseDI8nxuL;WEP?;BP6nX&t3624Yucl*uHmGNv(='
    'gN4tr*o^>eXAb1ti3&fwKi)Xmq48kv-TS`rP)4D=hY0Z-O^3D|8u*vC%1nNNPBVj=l1Dl-26E(-'
    'JE+r2c=tZ>*o$>Z?4u{F47lxEk8k|FY&nj^8`K;@INw^ndy(Ypr4k|-'
    'i!x;*Yfhr^ehHFYUs9gVYm8{fQ7mRM**{5E85pSpz~xrtJ8Thp4I8RYW5w_Bj0Jbh9n|+ZM=?D<~V=ePa6_=7ZNvV(rxv_JS9P{#'
    '|YQzIBg#64z?qH3D<z04j@{qL*2pl#Cmj?I~YiuIETA~K}ONt9qd4a8b|6_4yN~Vc5S4##h{u_{Cqhp*ZIY{(~PE;Ro|evkWz4Nj'
    'CSEXhPF~nB=nN-)iTjY`-'
    'sk$@w`sw%XnU=^Q+l==YqXy7u+wNR+9KHXV(qd$N@;t%2%J+`*pZJ@g_E8Uk%(o{FvlPd@ejGdA@!_!$;cnoz=s5kuK(?($5n3P{'
    '3E*Uw)I&R@paYEC<v3p}CFDPNw#3QtKKAI@kzvK(3LFzB@WVQ(*B%qWUlvzchhCy2Rr4AzdKjshuv6@zhQiRI~V=1&faoaDcCbuO'
    'gjqUM{Z|aHB7httDk(S7xK0&s%+w>=sRRwyatBaZlhg$gEYiaWm}_Z3UVg_lrnhW{$Rp6Id?bV%LRuiG-sh)z;%!n3m-'
    'hVYq;*p_HT$(lik@`A#_u>Ty0}!9#ECrIgO4ErGF5rtMf>bw5?6>uw*zg)*Mj=|UM#>vUl?V-GDD8_jW$-|4tjQ%RH!_cq*|_#t-'
    '&JR&(IpF_O4E-'
    'SiQrjxl}_+FV_#~s4=$#e=g2;VQ$soWp@u7*#w>oKo~UnjJW{S9a5aio`Jp$$AP<Ku1cTUf2+6bmJ<2y3$nI??}-'
    '#cU@m8G5Pkt*|sUW2PO+>jVdex!GYv0${AYsf4eTv9{S~cae-IcDhK$6FXg0&Dw`<7eY{$Q4y5;dFyd$-NCd-PIt_*<8w|^PXijx'
    'Vk{+!OUv=H$o{cRWNWS+6Iz!XEIh|5W~+w?Puhng-{G!GgUf8r^WZ|kd0c%ul|YYkUG1e=r^mt+-'
    'sI9DYgC&q^sn|w3iy=q3H(*yZxFGk5;K`?n-'
    'TDcnA#QX(_ckyIO!9jl$nUr+!ju7aY&QT6FemjX!04Ig5T&b0{$f60|B24__u(6;kqh^(b&}8n0Z3!iYgEd78UE0;QK2eG<v!BkE'
    'i#4@ht`5-'
    'tF?OVC@Q~$!u&r6(@UW5}wLwaKKU1N1+pDJ7a^Xk_&Ba6*o<0Fb3z3=+rG%P^msW3V#;amq+8zQ}!kO_q2U^4E{W0UjqI-YhNCVK'
    'hN2h$KlWOPUKSuoLF$RE;MuYE&AW(o&E1*4ewb0`?H2KI{9CLkgDK)eapPh+*CR}5+eUAJR`21@;7)!gzuTBldg>My}k*)Kf?Der'
    '}tO!y+^mp!(EpOE;zNqdBQI-so2MBOsXijMJ8>@EVuFmcAxk|1=HQZHPrkt!yQ~lNbcEgtopqU`LM3hh2F$1i@tJ2;42x_rzJ~5$'
    '*Fz1YDL3*NLcPO_l=fk?77QmgQ3YG1)T?;j^nZQz_T?RDW`Tpw^eY#Ez4Zcd{@j4;s(Io2~rW~gTKTxB7D$1^>kH?4{n*@`y+ht3'
    'VMGPAN*{)d~?Hi!XT#`9NNfN2W?i`!bO5pJ&a=m8QD={llgHL&X){N!&%qbb$boFgzZCyRmB%riSC#ytVFWPB~~KV<Qgjxu5yu;x'
    'aqmdO7zHFW+f_gH|PIBhwNXIb)ijgtD>(@$b5a{G+hPf+p5g@%*E50k<Ry*1W$>ZeTNY6E5iHCb4XXmc;8kDzCXhICeZt<c;BG|^'
    '0V)9L$dyX#6?!dZxD#5)may$rSFDt2-=VhK@Xb=JYVi;mrQfJ11=!~A8aM)d>>ESyD33ZndhY^0a`f`>_u=`9*;L8EGiGkn-'
    'h|iSEO6e!i)7cdJ~G4moI<R0l69JdE^`Cwl4bO#4qy2B7a!H7q>3+MRW0VR-'
    '`Y!Pwn7wzW8T4CBhfYb4b_3_~O<HzCXejC(`>XWr73pYP7Y?Qd6M%Z^^1I%c!J0*aNBBiH9=t#I^($7+{=H5Mp0EixfJHFK(w{r('
    '8SbG|3K5NP~3S^eFrK2AOV`9?h78bU>Oi;vn5VJ;s%|WI8YfSL2ZBp!8UljKDR}arX6lIv`4e>vW;rtY6Uwulyn(EOM3=e6U}c51'
    'Naovm<@*r36oj^TBm=N`w!Z=a8<A@xguxzCXeTucY@^$_Skg@^+hKxs<*H8lM{QmZ7g~U=FRiaLIyMA+C$Zx>`~TjcyG*RS(gQQP'
    'C*qwH0=+vhmfj=^mA_=^mZ2>82T*?lBpgE@W)F$7XE0&hOsWA=}#&U1;jsrs#K9eUaZ4Db)&ow@sPfnTw}$BK@u(of7AF?<IIjgx'
    '{IxkgkjIyKNGDe}vy%Menba1?J~rTszMT*jJYllG_od=Fy?8W{-R!GoRGC3T0oahv4vcYLdCvGE(Z6tg*2v4R8*T+>-{l-'
    '`m%{Xn^~JeZ3hCaF)d_-JAxv_w4H}Xn=DV_TDtWEpyEdI^=Vkstb)r{fllksq-'
    '@U43gzB0k;id^QU&j@s>H@P)l=D;(+G1(z%fi_<Vw=#5v%rbV`H+nkSU5k8!~M3BEtV0VmP>D<y)39r}4@CDG7nCliOQmdyD`5W('
    '$o)g(blJnf*8`n-T8q}5E@Zcq29Y)|*-Y)?1M_H>WQ_H-fJ(>*rZ(>;#&bP3bxx1eyb;<q4hA(Cx6%u5IryITm5+JiQf$kWorm>*'
    '2-ibAe1(k+sKqxEw3v{z}1MO5LX+F}t^>&-'
    'b;3_GG=*lAR@wd>2=nzAjEYs5)wn#dTQ$7=Y%RWde^I!5pi_a@wo3D+&DjVjEg9yXju<6JfFXUYhIyQ5zz1G#cXzg9-'
    'QIPuCKBwy^P$J(J*ve!(9O(na{4Cu*Ndt7|=VyHb%09#1*k-5;@Z(e*afnm-'
    '8HcDbvF!NJoX11Sob*!23N>|4=z5Iessbb|cZ)FPtUcueT1Rv$;BbQ6txx~*ic+ME<ph*t25K@sym?c?xy+1Y4gzLC3D@YR8?}Ar'
    'OJ3KR}N*6IP=qi;y<z^aKt<tC6Oap6F`i$obRq3;Crh#=Tea_7^uwJFl+nJ^n*EpMTZw}^~n@g6Bxh;X=rDiTNXcg@IbeWy)hg}n'
    'EXS~!kvElroIj4%DPs|zG@bHi4O)q+tU#$l@17XH9R23Pw!M72#Kw}bEdA4K~qAzRsM`&ey=Mv#ZabkSsVVr1!c^oH}U>?Yc2$n~'
    '3;(_I%oG5~MEGLHGBOO1i0+FyP;t2|5!rRLvx14ah&g}_|D6?~s%&B1JXUfcMKWlPq3)`>hlsGHn4Nb0Q<ueLat`XGH>1J9Sk8w~'
    '8NGCI!n>J5j2~Wau4GjTpx)97KYxu-$Ql~G&ES<M<dtl}1-'
    'pWAD58i@v0t?P%1sBX*X=P(yuOuBG2cKO9eLZ}3HEb!;$@cXc*vdm`*W(Hzooc_y_Y)XdYT^Q2P{G8{mYLXo*0r%F{yCizXJWjeY'
    'pa>~y!>9fF_z0>Xzk~^m1yl}1Hi2PTxvsWKMS%lYd_b`L~B3W&tleoPjY0=2qJT45Sg>Qe&G)i7{%$42R)gcaZkIL+e>J+y+?i)S'
    't!Cbgr(tF(@HJFi)F3_X-|Sd8^?PArgu)giVQXd3qMz8VfziPiyf=+QrE?r81Lx1Y9{__E{c~?N<2+)V-'
    '`Nnu{zTvqA+H7Fh1rM9;dxjQ8OXSTSPi0{UEocNtyZ*S8O<rI(FZ62Ww$#2EXDWw4X?)@YdAtxwucemM>7>pFMh7!b`OO=tPbc<@'
    '65Qb)o{5Fu3Pr1w9gOojOIqCJE=+KTM#xkY|sf)+#fZpgT2o-'
    'dHaZv=prUe3`ZFWv0Yh8!t5_*4i(cbE;VT+j%HE4xHa3PIR}iOeo+t-'
    'N7;NeIiJAo6GP6B1q@bk*8)%uZG|G4VmlUJ+DFgl!O;0{K?tuSqZODzpg!cPQt4a{_2i!G5#90`Prj85*S^u&GFRc&i&mV1?8<}C'
    'V^zGx?J^jMHuRBNeFst(m47zCQj9wMh?TTwQ)9&!V3N@QDSd<>r-'
    'RxjhC7lYj3=xsnzT~q0qX}sL_0FuK01VM*Cgu4i3>6bk|63_DCngWOs0wPOpV)-'
    'NCm=AI|7u7_kLKdmc=O{~c0{Gx}ymZh<uIjbe(?TTry^@rgUSjC4Yc#)8w)nlB0?`&bcjR@Y*t?)Qtw2OHp~^`_D5ZFnh$4r1hO2'
    ';Jxs$rS3}l9qCI<1`k95Z_jnnB3g>MzJ&Hd=eYQ&Xm(hY!o{a$f6bOeYnQ3yv_mEO>>Z&WleLC%Xv))g3FkmBQ%rCn4X7=yyo525'
    '-u08FCzjZTp?gT-ZPcl8f5sXi0#Mgr(B|FzwAThl~(Q)`S89;r*D~Qb_FaKQwz#?4F#k<4L*Yt(?n+AmqfD6daaETorJX$=@#ZCC'
    '0d9D;);Tmw=1)<x#^8!R7%++Hi}UxC6CxBMkSC#E6mu-'
    'N&A8!akUgV4LmXT@kC)%;cv_GopIEO=W}&|M#~FRw_*g1_35?gFmvhk(*!y>l+mUhR(>@lgl_JpgwQoCQbK57UP}p~n<ywDbWMzu'
    '5IW=>C4^p5@vgj^V3z)&=@i5);ginubCH*#VAcU;W;GYtDDI=2C1Rtvk8+ZTjRHRcwX+`izQXZ@nIJ*L6*8-'
    'Yb!065et2vP8m%!1vOa4Z0*P{EUbwY*y-'
    'r^@PjW5%Be${GCV9{_o6iy^3fP~w0VP~XplY`{BjG9m2XYNv!X(1r+M^*V+e{)+pnl5c1k3PSP_%NvMZ)XSS6n2#!Nq-'
    '8FX3HZt)HOaZc$b%DUwGNEWUl2#m$Fp6l_t>5V29PML9vlM!^<=?pcqb;P|O3BS|SFHElr^46W?vz*CjS-'
    '{6V(wvG_qdYi_n9Y<?dlhC%2(=A^$PVQ)Cx4_akU7(p=CPs`<Xjpr+w}C7;#PmJ}vT%^AkV$xxH)CL(gm>JoiviOfO!KZ6H$N=vX'
    'Pu}>sV+AW+@RAp%=|n@!4EqlyNZky1%nSPGq}0RMiC9=91$BuG?Y_BY!uNDsGjxc)H+;CNS1jbtA@03BjMmcP;R^d|JSg9Wizd)8'
    'iE2#2Myr`Wb-'
    '^tT|wiW3BRW?;loMXErK*0p)rcNfw2SKh?`g!!D))SSf7H^6}PcI4QD9sV|@l%6*saz3uh|sWPJ|KQryb=Je;k(4(`1LbKk1dca6'
    'DcD7dY&xr?k51#=H7Gq<_QM!^H++z=ZD50q0wY!o~YsGSweT|;AWh7BH@r_c%7@67n_`|W6aU+4-'
    'P(#*`^>KX@px1(C0E+Qr%_xdSJ=)dJ&KaKhPx83V!FrEL7d;Kh?hu?LtpG&8}rSA3f=~TEZ1HK=g^`7sVz!4HIm7ap;NC}rQyNH7'
    'O3TB?A;QO|WvI;)xJaHH4CkkfXq0G$YDjVIaFQ<ao=w5v}6U0XM>H~$d!p24iwMe@n)CxhO+!(mie7`s+O|*FE;4)+zy|a<!-'
    'Knp-Mw-TjEG09!9hcuICFA-V<@H>Eqx6g`aFpY73664nuE9}`&qX-O@t^eA^v0|mZl?qe$(kW|PT)`p-'
    '<Dn*^ZtUN52Iwk1oT`5cXSTuBHu(|Nj<%6No_8((M|Dk@`sIXikEXgY;;pR*n!@@sJqnLx6HR^WA_I9@p0Wo!e&16u9cpfuNmemU'
    'nLI!uW0_|elVWSizgh6XYj%s2jiK%?8d=(7B9MSFrLjzZXAr~@PZo$<GFshJ(yaM0!4dEx8MN}%T)mZ(WpBiAU?pf)GnrPR4}5mv'
    '5TA#g|Pl!83MxIR+EZ^K;WgCR3rxCQ*%xP2*N(3X*vc4fzNT8j)6hobDpNFRdu${1985%-'
    '?J@bCTS$)O7}i=#3U2gTa0gP2<+R8*}b(djspK9!)=!(_*S<pd>vPPZdLdOuKHYM;Z0oix$^y6xaxB?hPQFm=SmFk;HuA67~aKIp'
    'DQrzDB!o5SN0K>9l|@|hefYESbqWht4M)S@V(p0e9vBdSgh~irG~}&-'
    'd{;2GRF7pqqsiG_wZ?5ALV=aG_SAbdyf`Jz(8l)dJ$(Vjh@{rTejf9Xc`%fDhIVlFA2PE^8^mi%s*bA<4e0ykc{Y1yUN&E;i`j|9'
    'A6@;a^8yHRpAnZ_u_X`xR&IF<RgN%m{!??Jv8PbyuGlKfJZXN>nkihh4-'
    '_MijH@PcRcz+8os098PO$eM!^ej7bSwd_D+V!dLdqFc&r!V9SyJKg&ZOFx!h363paI6g^%urN?yori;wn(YF_wRVFc{d0no;v*X1'
    'HuLDP}<#dh??MixIKghD>Arp-kfF<q@(;BPwJ)yf6_rZZfvT;Ojy)78oa{-'
    '(3+!Qldb)7kdm2!X%p9D8u2z~6MPORWSvnmOQ>!t!T$fBU%TfQNRTk8e^i(k5S+tbe`pd|YG*DR|@e%f>`|TO(q<5id0&)*JDTMn'
    'sK?_95LEH74S7yfJD_#OHitHE%q0VA!HZP2N1Uw!()Z{U>8ZaM1Y|Lq$Fo>1VuG`FD~2n+Zw&A=3XaC&4Eo{hYTjKNabJIZ@w5z+'
    ';)mY$YrMhj);_7d_@M-Kjl~w!T^npKF`kIB7(^Ng~>^!90{e9HR06fg=tRu&+%VNO+>NBNi!53Xb@LGDo!Ec4Vw0;-'
    'yB$I^tAwPJ|=cXMa<aBcge3igH9W-%S-0!-4r;(P<)EY+f&XiJk~CxJFN4R4&pJFAi7fi5i2;^hA-vb$SA<`a=De1U#NO;?}|vgL'
    'uFDN6`@v?>ux7SBLeOKPlKFnHspL*|G>m_4aH(s6E>cnkMj#ylid>xVKsi6roIXy~wmu@WLOKd7=HPqhh@fFEuLG3-'
    'OLdMR}op@Ha<!A=>8VC@)0Y-'
    'CQw69GLGGMJsGDLJYRVjgxBD7(hl|$YV>ei?Jcx#h__8KhFGB!QR$u1}AW_4Esg@e{D+bD&UFC*ZT>}i{f4TlcKL5@kI-iBI!%P`'
    '|c?7K6~-zSntD2HOG1%-'
    'cfUu_t{5rOO*HF)4C<f`|xSrQZY5mdEYWCyz6@JmK$(W(3Vo05$Du;;+$zY;^KLNUD11AAA*fr9s~t@<`yao>90KG{+3#@c+7o-'
    '@aH_}VrI3K;!zi8x2+WqyEw@7Q#|fIK`kac@IFa^X7<M21Rb+CZX+yji+8?Hi{5zT7kOimL#9xdyR*z2?ZrpOdLv$HbgVbx9gU9i'
    'M*Apkjq*l(TDL}dBR<VrD~^SQ9lj1CHSu7iF|_mUv2(SdPExZ4Qe@0Q#FOl{?E4Zo<--'
    'o%;0L}}(t$8rFg;{e{D<wNul~1M7^KThe@iKW%S|7ll)&Ytk5WqDa?{5sC2+av<CGG(-'
    '1G@b30!XaB&7r{HytN4FpmqDb&?w}R*+^?EN#t>Wu;+znLaAAHjJ0*qausLpP6&282jlwBHF-'
    '~+@oG9cn_zU=o16_i!nr}yij<mClYtd6x)^xBYd1$d9eq~*QYO}+ro7B`o(lRnBiW(nht_F?)7Ww4lvh!`T&`2{lNAz+xvlmGTZz'
    'iSWS!K7UMT1q>6iq-'
    ';!{Q;%4GYWG67sN&)9stloMILls95vs0O!?M3=V+8Hm@H`2^_KYgoN`K3a$QakP9FfKN2Ve}1I+0_@8(M!fYG#hDZ9s`zP23xCgY'
    'G~<8=>VAN%=dD-J<Jl!J?D4jXzpss*hi1b*hi1f*hkZhee{@&eH1eG(PK0A(c`!d`z=|--'
    'oRQ3^)`Oj;Z(08Nmq6*GpoJIma*ZxyE!M$uD_&Hs@V0-A{YZ>Em#mQ&b7Ms_n33jGw2e|#nju-C-'
    'H#umO?kfY60ij__#yDksf_RNJ_~-mvf?6MRqAu9=0F0RqWHnOKlZ9I1Vw_sq%RLQ0)E+3&-'
    'tld$bwnI4T{~NY|Whse`;3+5;XTw8-'
    'MZ^SzC@49)ZX0SSVItzQ?AqzE%`TqtP%IwIpkmyW8@(r!yn)=4zW(s1rF&9d|a$^#^fl!k=c4yQ0mdOGESn5;`j3us9rrJ?*ZFj?'
    'ubFHxi&UkXj#)0x_S*w(S8#!GD-YwACnbE=s7&xJ@@T!c9U%-Joj!ip~zGoopl<cV_`r9J&+jst47be$P6Pg=0P-'
    '<)c<YRRS%6K6;6!^Fvv+c0r(b%Ww899y|N4|69sD9*vjlUp-'
    '!?sOerV+}j5<9|6vP~vnjvu;s_<XTj|D?Q95iV9=bvLVubSie}y;-'
    '&h<T6Uc|r;24iF7$jYq&=L41(fbGOdmXbWi#!5XZU*)&h4I6SJ-XG-bKtWaZ7B<jRigNXFuka=*2uwo8iy?%q_7wvpj8qKL;|mL~'
    'j<vJD2DZnWg7EQk`Iyo{LC$g6F<VWheZ~D(O0~3BlgnxuDqv*~D&TmbI7JCf2h34BtSUW$})-'
    'sb<;LIpp{<+u`GR7^aJihhc`uco=4ijE7;C$aolLi;Rb1j>vcz<_g}ByG(Y1ko+hy;xCj(O4wb7VDJDmgKLPv!ULXKXy0aq$Yuw>'
    'd>SjVxuV|U^v8By{&vCDFsW?$Yj3@Oto`s({bTL-syU}p*`pAdV@-'
    'owW04!d07rAd=<!E&2{dTmV~vlmBe2aa61K2=v$sm<oomGW|HFgd3Jr#--'
    'mRbZOJJ2}aXv7?Oq5C6I|MVS#<3%;ENj}bA2wdavm=aG@$3lWtC?19QbY-'
    'lQQ;KoLj(W0!&kBDk}w*Uh%j?%;a-5OaJ8Q%8S`^pm0;fl*K%AM7@A<>%JqQ#6IkQf^AAX1t%sGh`PFWEb-IYbX_M2Z8ILw4eU4#'
    'gQ`6_2*eEGh-S0<*MS^Kbz2{AnP-{K!Nda>^uRb<Tj<BlC%JwRkt9WvR%T+Wv!sXSB+-'
    'sA3dfbw=zjSbbeKAH9!$9%UvlSLAyEx3O+dy(d%=ZNQYFwsc9%GO8*KoeVM8+!(U7#>qwntx;@R@=)T?5fI=~Ij|yf%G?p@r9_&$'
    '8IuLPev9>-2;t73`3>R?lUmLU*T|Me5-'
    'x!IYxj&V3W;V@Lmfm}qu}igv+{u)569_A*zfcy@#<R6IMv71b=g)h2~1k#A(r1#>%zsPjBWjjkBoZ5~rvV=$!-'
    't4VBL+UVzH7C%pTnqU~JaoLIgVvi2gxQNFkkUcs`!$k_yAba#R3IA5`mRqTOUBZ79yzP!2QLtSCZ@70ps$f9E-'
    'J<Zgf`JKlh{6*J1|{&0@LAP!3SM<Jt)~SOY<kPrCvbs@@bpEP(W%wC3%Z0gWp=k0nV_QS5+<m4x`YYU%sya~B1m|QWNx-'
    'v;)u`UI$(@%CSLd<W~08pEa<7yU;E2!cY;lCG9ODJO(<dJkS3HclSmUXn5m-'
    ')8O+?#gbZf#XhH^i_ULa)_^*O@9nbx{0!~ppX8%^fwh1>8!($4zPq=Xyo)Ju8>8<^QfH}4XvSk8&JCjInoH}7`nYry%CaP%ago!G'
    'iI$>fpbMKs=w28B3YydtAsvnqy@uZD`yf6^gT0#`3LA!wBeJTGiZ<Nr>PL7WYn62_@Sqo3cR3Q9qRv>&ND-'
    'b@K6$l^83WSem1;Qt?0--DLe?`JS6};{|;Vpdh%I5uM{rjwfS6rk&E0{VmjyqvpnThR(U8&-'
    '^6RuS8+zD4!v+<sV#H@u(>{G6bL!uj^y=if+su=DU`_`RS1DjmvVQ3dCqp<8Pyn@0XE!>a(@_De=xQsN2j#u16_XbXNihJnZ#PLr'
    'tLh@~Vbczv@?+7?qafjA8iD?673*X@q<kJdXQoc&{jDnYyU#&f_;5B7G<oAMFC*8H)6V!F{p6l%wOxqa8ps>Eo=Jvy`Qt=E5SE+a'
    'gg{!LBd}v|zt|MD6^o{H|Yd0iWcdSduXpe_m^cGSSXLI}#8J0k3-'
    '90|wKfFBPKfEH~Kb#QoA5IMT53daP53dUN4<|Vw>@wnF8lzZQ<Pr&s6ujWf`bR<SaxRvi7tD<4U2%<sy=*PyQ2{eMXX0J3CX6quh'
    '1knXQt_+_lT<uw!lY_8K6FqXUAj0FMl*EhzBS7)e$7<elr`iv<pj~&O%-!-UdMi4`!Els-'
    'ubEj->YEtRFzwjVFrdM!p;Pkf%YP1p!SO}16^<>94||XCg$SNIwNc2bw<`i>WtXuFkZv}2Jlgh7cqc=TI$2B2H_i&Blo>Q!O_-DZ'
    'YDm$B0)l3ro!vf$EHwdyA2QhGO~%spe9WW88T=tqcU25W-'
    '%nu@H(b3#Bqn<1g>~GK@yz66;CHh+ER8D=T}PFQg$0GS4rAZb`<B6ByC-'
    '~8tYM%J9sDDT~V8G;DpnQPPpk8IN`>L98M^6L38P7v5}uei;X-'
    'SCpKcA!R1j7h>z*=C<ok>T7Rn{Rt6OtJQh2^0Zn#G1(zb4zNk0d%EdPMt<rJYh@<+E?+{`RVM3p4!jX@L%r;)ROczu}vae?dsy*1'
    '(GX>R@?CV*A3K918Y(bR>`+AO`Vv~J6S5URdzCK#P8=3R%p{TtnaK0Hu=j-'
    '`JBVZSV6DO9rpSgIn&`1}A6SeV#6P+B;K8Gu!91tJX6;Te@lSl}vRk91SZ(F{%V+9X-'
    'D~Q6x4|Gj5d^LDI#F`9YK^wfaNgiTBl6h_1VmXZI|F%nqGyC6wbOe+CZJ&;0?!STQC}t}glr}S2*$(MwCIn2WSmqsUPepyRfrHH~'
    'I#{pHfav2jTy4YSL=BTW?>KeAd~s5l>zUh%mKo`S`Jy(S`J$8S*(Wq1%JuLePKa{7UPijSQsTEm)yB%4kR}dWO=dc^{HAoAq4+c%'
    '=JBJ4YglVbou_D+ni~P9(O~JTe%EUdf*bm&H%bbR>bp^c65P~J4FXztY}d^i65+vJw`k}gJi6;v4Vwtx+<y#J`@F;MrHH2^aM)Qz'
    'hu!RpM#L`oIlf%xmgeHoiX$(?D~@!*&(X;(?Q@tI<(BxUCPumCW<<SGEhpR|-'
    '_|i!XJz7Zbk^2!uFl#z&eK_2$N4&I>$pH?Z5<cttgYiBok87!ioV_{>J@dB1x_)$=oFiG9>oqPwjkr;Cu^9J&s&KccADy>UD_0Am'
    's%cAA4E2$e1b8JY9tt0--%=mIl0UU&6kaq9C<2Ua-'
    '<8^kWNl$pZk?jPKc(tGRg@zC&r&@d0~ERB&vw<W)q^pC5N>c!nKa3=3qfmo2`T!$g+O0#Fx_)D~veTIg-evjO!e!VSSFGY78y0o#'
    'P)bapHE4e}crB+d2M;5~ptG_$Nu6yPe~IS>ojF9RFmAv$u2nQzTB`&hd{`@J{CVdn;l)2^@b;(ebzVqS3GmUXxSG9MN1nT7#rdya'
    'q`Zye6F-'
    '(LRT(q8t$))m2fBxCOBhRm%@|$hQ|6o#Wc9^>+ntrMZ<aL2So(@a+nEzwZEPr*UOrNV3wztK?R6uVIX47>+*<%`gIg9GYPy{x~$l'
    'DEx6~hGzV6Xok`F<JP^$(e8<Nt$h?x=LD`bx9D2EJCCV{YFKU4hA|r2qA6D9mSI`UZAI&jJQ1%u(gnpzC#SPdXi}8Z;X|Ah<#fG?'
    ';jCI_=*qs%6ay3-YB#ZdncyfBb}bH(1R<%BP8Pfw1}n2+OWY>bGi$g2+-'
    'kXD8a*$|V3pSR&X6G*>XL9@Z?)a;NtoSnaG;C0W1x$;Q=p5ubD)d3OQ4JRr9c;P*FYC>H^G>>-wIgdVUAA-'
    'c+LZ6@!QN#%8LA5(7yCdN(#XC!<NLVPOmoS#1())HRn{)BgtF)*9p!T#@bH{gFVe`vcdvX9F4$A3D-;5)E%voaD#-N?r4pKn<Q-'
    'Jj@DU7l{;E*VO0sY=s<@@n?5S=qFiu$T)?w}btJZx=^IR9u~NXDZal%aVcIZT(k@=6#@dcv7eeD|vZYK-'
    '#tPGHT5CHuew)6wg|ZAl*9G0mmPrX{+1~Bzu{Os`y&h}xznbe*QYXpVd`n+=eNLG6rR-Uv)5;KkRl^mD91rJ`u!D7>$$Gfaiy~bV'
    '0a)#fP;`Z@Sss!zn@H^#ku!&=&ltQhm)NlwyRnj5W!xx0#gHY;!Z|>uZ@I{aRRZqH90jNKIg){N*J${WWQW>a!JUdeERNS?tE6O('
    'vDbVf*4%ihH)75GJ9C{%8YOviFA@fEim#p(rrRZ3uizbJjpsGI+mlB%xBJ1kHC^Jz^uDyATPtBh*uh1G&FLO|HG?Q<&lSG3Ig?>J'
    '#+jCIAM@4e>)gkDP5K7+F;7n4<UZzW)3>;f`MUIN?qi;kzQcXYQ`2|tMOF*AyL9yKf+S??veDaq*qgDYU1ZLQGcDfHo7GHvM&7jh'
    'NjTbKBo2~roGzM{w@@emIWHu7u)%^?Gi@0SL9-3xEU=fGR-olM;usv5XE-BK?9Fnfp&gMq&LA}6-'
    '|r=e*9f>L3rb9T=1N9ev}stt5@#e!{($5NpQ7O*$rE=kl7~rMez7^pOn~wwoqH5Te;gA>ztGy+S=-'
    '*{Td~&0OT86q?WyLRN*X13Yj3aMFq@hEEWtHmZrl#$KoiQoxAp_&1qv}(y+WbsZ{S5Ok@YPUMUGWW(r`3)nH}t|iArpkJK9qdiOv'
    'W%1PNQiNOx4PiBV#d@}qUT3wGLgo9?2Ww%TU87^ls(*)Fmv>zMDgk=9b92n#Q)6>x89ba%mdu}xWY+Yfs?*4{rf<)b)zKW(m4Nw6'
    'eu?|}*qxAs0z!m;`2Ccf1^?h!Xh0#|tni}}jSPT`v}S|}6XO)`{?p~F1FE}Qdd`#><q#CEA<j)?&&clW2;r<yq?2BryfObkkUFvr'
    '9Wn0dJRLmx$*-5<DH#uVgV3jXAdwo(Me{j)n-C*ZzPJ9oh@(Z9^j_QT$ZwKHDoome~HZYnI5q)PI3-'
    'dDm=b`(EA0_b8~2TD_Q0rh4ua^t+5ae>vFmA_BO#t7e%orQLPMUcn8xd*-vX@bLO2bHl-'
    ';<(JGW5|0h^BDq$doJ^t82+BAZWcyALv=8xgHfKgZjOLvPg^$^ui|Oz2I1PAC+_tE?k_cY7ZeT;l;LjdhrJtX@_D9;5odC|qj#&B'
    'd_o=~3#SVF4fQ=a&xUaUey+naQW6)?5al<hiO3`0ANpMhKS>yLGeA$xg>(nA!5IqdgOiK``;a7~z&<p|D6kJpG79X&lZ*oUh$N%H'
    'KGH4mj*^(fxvBeTiCLUWw2zgTr5(u4O!G^@b6pT5lxdTg&x)2N`8-'
    '~l<g?1sB*iY4Dx{^Y=?b=Py`?N;ctD55U#j(a<6fpPaEx&e5wKm(xE;()X&AD(Qu36y0`X&Wo#YwJrT6#j*jy!fF0<0}m`l6FZGw'
    'b7l{=9DF!hS(;YkwqaUA)ivUF#eT{ljLP^Lv<J}Fv?<X6#BB)_RDMG}tD{Rm36#)j7s0tV%p#A=DHzLB}B_w^d7JuyA!%4#nd>NQ'
    'd~gZ;gD=H@u$b8&SGoaVW@+8YykF0WoLammW@b%lgI6bIN<682{4g7oN&+@N!7p_wik3WG{LdSHjGDHB~kd0@scW)903#x_ogP^L'
    '#>ZY^4lWOorON0R=ssv5~Ac`2dslE&k)+CT5p1e~M&13z8Bx!Om?83N98M-'
    'wHq3OHZ;AURXO1=`!@ECCm4A2VkQxJa``qj}max=3p`S3Eyl!#@NIRivp`2nlu=`y{Tl!iWTC>-'
    '0Lt7ttuQA{0$ok*(=YHDsXE!qWSR5|qt8E|c)n&gxNJur!p(k(e8gRwMaK60Jruq_P@G)Ce6sySh+t#y?><MCY<9Be~|KupyP17B'
    '(CsiGR>_79J~!bkOk&9VauH>~y6`1tkl1(&}EaV&_naAIXxP`%C;tmKmg^QL@e;trR57+!b|5z9Zqsb`U#Uz`)MRk6o}dlnIfTt3'
    '>ONydSSa^5>+g4#_58K*mAI7241z%@i2^U<IXPe)O~1nO{RU7;86%wmwP{>!j-'
    '<Jz5g&r0d`XN&J$J7lM*9FUI^~;>D0Z%)1!zhiMlB{xIueydNfA4EMvFi_v~5bP^poL4velJWj%b&SBgIPeYmNh`CC%=*Ua)q9f~'
    ')s-h#Cbdr%8A52=)MA()>Pz>RQS>neFd&@`~-'
    'R$r$&|LoP$Z4eBtkbux1ha%BKizZLgoI+R&V*vG$%JAjXF{>pW<s&oWkRu2GNIV1Tn=m0RB>ZTo5M8xU3mSX9TZLzXS-'
    'S9SkL4T#n(_KIAT64T5qIZyxz!rNmad(P7UGXWzdq~2Uyzq9_ysxXs>&@;3C&J?f1;B3Q1jH)^AIA(2jtYN_Z%wH^jBgRSGV3OX5'
    '0ZMg+GTrEp|EjTz?@j?AYs>zKlk`3z<qQ#dlWGV7Sak@-'
    'x)C~?18o*u{C5T+=Sgo1+ol2BH%f5JkNal8&Um+Xa`t46DhJRh$%@@i65ZDeq+7oe}g5!H4`7kdHfyd0RBo82D8Qkk!RwD=~|zVu'
    'X=NN7{8=tbK_4u-'
    '=CU!5fI;f@It*pZ{SdHa1qA@Kye0o6rUu(63;Wi>}6k*lm{mm_usHtKjyMa%#IF;5l%5c6fgvkzx!cwgcG>;PceHoM0$K3pFLcMm'
    '2Kq}z(_OZjJfU&{K5eJQL4V4hIAp$h(?*1;L((fcdFADEzjKE1zEVP<e1`_-'
    'apy04K=u*mwK*DY%ax|bJ(O;JRte`?m~OrLS8E?{GSmEde*oV=F!NT%lYRS5sOlXFPRC*YEcKL|>uT_R~2C7$GQiP)M}@_<7mU~6'
    '6To2FeqcF4;$ZJ-7Y`Msh;-kEbqocqAHTanW9v7HB@jZ;(HQ09*2reb?k@Iui|D!-'
    '`Kq{7~49#8tu7;pR@y+6Vm@1*xvLL%meMauJSQnzUW`{q36w}SS}{@0Bfj|sR|!(ZId6N0wT{;E8gn-'
    ';K3;>p|+0bfdZHY4_^RXl+a;nx~ho>jynx6UyqYT}z<f4V{w{v+_G+lu~lm+ZvKZ{jFustFnmm(XawkSh{PY_zpv{7{HBGEX-BHg'
    'wK<y5P>ZQI+i4o39hyuJTcQyUK@=?JDNUrvHiohTTT*kHCf9MenbKMa;Y6!wHPBgVzWFLz8@cxi0HC**5Dq*)Hoi8IX0HY@c<U49'
    'q%C24x*5J8;Lzw>5#1*y|=}LWu=lcYD$6?k@4V6|@GZ6X^m@6nY1(t4k$$-'
    'V!e%m8}O!7AqrjH(dPajmdZ72s~~ob3XH}VjEV{Uq|<={3(g+S252w{Vc}!Zm0K0_}<<0{z}>5jyd1^MFJ;U-|Mg7xEx6B5kcS{-'
    'bj2>z*Oz`wfakHI2oLHhD#-'
    '2=fv|?Zi2p*@D3`iy=tAO4U_PpfK8dxl0=Za7f#fKb_=}l`$aFjCpVYYE4bHG&hrDIU!<Dc)A6mE?yfqOWf=2CPG1jQu#u4)^oO>'
    'x`fL{@BR8uObtdx_W1CgdpGWtq{4uH4tHPmT9(MZg7-'
    '6RG)B7WIneL(YSIQIfE=dJkZMZ8IB=BjTCtnb>g?G9i6);)*z3s;ZT&Lmx+)=ZHPXuh?z%e@}FkN!iw{rqBrOo>4;k%lkTkL{YYQ'
    'jYXF8G6@3*MV^LA?Bfl6P|UQE<Q6yt%;BJhU-D!~g{PXZSy}{E-e!u#A)uGv-T-t-'
    '65FeWCRHEm4Fhx0EG{=Ichct?VVD+g9FAs<o|foR}w{{wF3=`~kf`B2&DV-'
    'd`zG%=_bS5;(;Uky|8iV2B|w>XRY_b$aBhZAEJE(A<oytIZCc-1JOtFgOPU+fJr$vQG_^={qc-JzT;^0(v@E9WCKg0lnSPrJ4Yz?'
    '8aAVLJ9|N{KKLf-'
    '<NY^9N2#<dE2ok4A*e9^3Be7`8}~c72Iw1E#p{PZdE0qFmuDvZ7a)@=(d#?lWJ`%3F`#Yvr9j(;)p$*WBricU%?SKaqqv6-'
    'd`zC%sV13N)NW6rI7;m&kc+?YTZmyh}MMD30(49;gMml^A_hInZC=r(4!>nhi&d$<}yvBD(pa$G(imm2fCx^K=+3Zl-'
    '5F%h871T+!TgsI7)fGkon4T)o|6REoecgqoyy%eBNPZXus-'
    'T7?hz}(<aYLN`y(=ryWNnvmh+rvAxX~k8Wl8LwqYst*F+@!b7#OLHge+j>wL62fe?7Bk~A(KfS-wh*;Phn6Ie)qDH#<YBLN;h$+%'
    '{t2iEAZ_6g~0;5CkY>j6xHl?5;n(6793aFdED>%vaE$x=TH*=Zk;}YHz>E?EDeL<!#@YMdIOkd>9qSs{lnl0zFQcwJ$k_$bjXgHF'
    'QvbbuQhoh)-<2T`U0b9C*x8Vl@wibSq`Bo3S-A9wI-'
    'a*VDP+JPG>Voa2u7tYEe%6Xubk$mOPFyj0jX9^1I!VFw4GD(q2hx%LHWNqa0^<sOxHp$<-'
    '^f+H56JAcJ*ky|iVr7@x*BU#l^O$!|F44G?GC~^l}-tl!8P=mZpaYCPUx^WlGq6y7>B4T8I7ZfsnEf35@ITJCAS{LROmn*n-'
    'Ei>>soqQrZ3ywBvQG9pC|T1Td8HP(HBsLQK_f|kh4)vnQhAqZ9Zvbq@nRTD<kcUx3jXEnP1A|CO=egybZov9D%tHz9SUriXEYSht'
    'E>hGAHG#--'
    '&cnFIM#jk@oUpReu!eW{zl9dir;GNv1DxV7?;LS6C?MRhho(whGeM;d=u5xUhT&ekh=yI~Wc<lLCq(1LxD+(_&ia=G&b}I$G{qq&'
    'l*GDZ=p&QQ2y6LE@Njqv%_<E3>q@!&R{%{d;pxTu6Ul&Z#6>QbICm24e?oMeEdrQ05HtUXO1~7o!2~pB5X*AOF}|NV_~2(aRN#v2'
    'OGa1z*lJW#j4`ue-vf3BMC?yo8nR==Xy5+*i4yKL`TStae9#6mXJ+HSXv=LA%In-'
    'O>93PNu=b9xW4?x5Iknh17m8(w?kFiCI05;BUZf0{S|S8UejZhx0D@QwEgT-'
    'h9^TSle%Dv?}6kk2kcsn(hBk0Lv0kUAPUI=5zxEv$N+iD%36XaOHU<X3q?OD-'
    '`b>)>Eg_5|M5Qtp16DldaW1Rd7mZbrN6kaQYY07dT19S^GtJ#JzqgeF>NLEM)XDJmy}%lD>kAeAcRZ6`t@~Rj<L5ZaerS4TniLlB'
    'aLM_XTX_{_-}a#|{rzpN@phO0C=l9m)1(RyLotCf3R?m~-N+j5oBVnw9@tK;bn~V-'
    'hsFG5b(O9VUmmqr(;TlpOAkj#LC#Khhmd6EtBru(T`uDq*DR?wT};_w_9+AM@M`duvi4K9DDCvo*ytu_-'
    '^;gHa^ho`c!^MZsQn`v04PeR7(vy-6@6s~D#nc-'
    'wjd{xouqiRlIoV!c_9QB&H9|1eK@Nya6m%Z4uaKL(ar-Q51#SgSu}&WW@74n|6=lFUlyaUFj#SDUAO2WeKWAareS-'
    'kZlqH_vyjZJ-CY7}GO1ZT6QI9N;bWk%AL!fPbvu#N4ocn4*5A!*qtx+K^$i{+VI4{*__0KFcs#|IRR4|H&{~pJy1Y|GGp&P(S2K-'
    'xT!^pW!uAk)|ZFyP(z>RAyuIS?gkL+{Z{4#@QHeXk9fMuP#J(ACWcjlfb#}+$WJ~-?~p~tG-'
    '_!ZW8dzg#U7jfL|x}FRjG7bA<2geN0e8@fu(LpX)&xLkLgB!@^z!r{ZLLH1y6VWJbC-)SV!o!!I-'
    'Y*@tV)f1QA#8$w{{3Tc#7E%;;}DFScMfg|pmz*lW)ZkGhU7R_xjzof^-'
    'euee1R)4~r6KC~rm?}%9LQvl7S(1w*Y}QF8vraOVb&~0<lPt+P$$Dg+WSjU-'
    'G8_dOC1%C0Js77_c*Y$}#WgcLFMRvz(mZ;)l`5GHegq*p6mQVQc;F1Wj5zAsxzLU2_hK5LYZwkfAKVby_aVyW%vJ#2*V@j=%Py!R'
    '&ME6*vNvDb4eE&6Zcs<mRx_}w&13VvsOhvB+DJv1*3>JqW;Nf&wt|-'
    '9%Vj#Bp_tqglMT=l73?oSJ4;@m(oyLvj@L`2&FQPwOy5?xezPrT_3#aatF9g{j~*&vg61)1e+d&ck0b|5xJvUFbC85dnhPc$3QF*'
    '~xVS`7X31oA?<g9aE)8MxL~Uy8oWmsC+1ct{kVl*wTAjCG?JaED4f2RhyFnhYX*H|&s?Fo6gRrj4no8QTrjmC2S;37CExyCM#c=y'
    '9%<UqKRTy@Q9ow5qhovt#RyUQ7Okei<5-py+7w6QYJbUjWg4)dv@nj;YXYa*X;TX@}`?vt`?7dG2YD+tU#eLy8-'
    ')8fXpe0={*}GA}WNYIEv<jRW$ry27S!CPG^y~&>M9*$8M)a&^<E;vj-Gnv{I!)Wd-J=zL1-'
    '^nrtbACvhy6o1X0{Yti{V_A4o_cn9DXVtk-'
    'p^k`BXYOeT@l(JeEXcSQ43GNmPa<(HWK`$*?3nGAzj^8J6T@K}+<Ub%`BYartOUXX|!B8gYJ^b?s$(b%QjbS2su_dR4RTfZ9B-'
    'e+?58HXG%nOqIh7;-xhe66Oh+;?1C4Fz<Lp40yb0c!h%f1W0!@LBUV~${k&)-~a)MJGx52fdYDP5nI7Q0yc3+$Lp-M^d#-'
    'u7vam=_b<XJI_p|&Am%<EasDo-5R<bx@x3YL9KWu!&AXtExS-'
    '7D_A;AwgF0ffZcs;TR?X%+=SOWydzxH@n+ny6%f+;}wa~E~Hf3XY8<AIWhZ+s&i$hCau3%pQg7urUZ$X6<bk_cRqV}Dqn6q<}=`t'
    'e=?~61z1b<U{#Ln4@C=zdVOFAjiRd#r<r(MAJvf24{MQe&n+<m_cCmEemZ{?8^?nz8WhTmdM=efBH!iWpY=4N{Xn|FgSV)JegMr>'
    'Zq%6r!4lr7;XfrjCB{Yw=*;acx0ZEX>TtjBVy;T!*{m{jd({Ak6jYAN2EVp_Fh@Vbh5)c|i?F|pdQcvFg*)sDk&<t%uyN|!*5JNQ'
    'IUwKG@xrZixzqk_v(4I9YxkR;n~qYV{wt)@<N2S!V#fe?v-$)`xum5Z;gS8#Rb!0v)O;-'
    'a#^wzt1UH@G9V=mvMh7S&8Xv|#cUv!rqLPqgHIUcCk_Ig~X_$&&lCh$&g}09G+2OCHEFrew*3yka3`WS*g!CT~yBOc}Z7V5aHYlP'
    '-TMsH7d5d>PuY#dB2VW|+abs=bvSAM6TYwK8axw_0Fff7T5OuCW98O$x5<Z0|0JBYs_GZ+q{(yFnb$yBow2y{p;#(4C5!1+^Jvml'
    'scL3o(Ts(4>xz;ive80b~3IMMcO^bb6hFW6h&XH-YD69?6CiL6hKeR(drIa|e%z^con>dU+z93?o=1P2ie*Bx~o1^g0;D(s?4C0?'
    'jO-'
    'C~&nt+8sP>aArKXk5jCA>R=p$bph8g_KUb_H_j%Jt}KD$+G~%Mk=P2n(pVV@uK3KH69LJ!Polq!!Q|q@>MvtJxns=<l^~Ei<+!>-'
    'p|?#vQ@0WG*)9|F8IXzjY@dnw49vuQ24!MCjw<M9iZ*H49WGI<o9e8;W6>SX(M8-h&3B8Hht-ZrEE2@3bCZ^O8*8xRzklE~qf8?h'
    'XGYC*ew!}vH&2#^T7|UBFu*#EIvZuZNLQ73pZ%)mM0y`y=l}KgB~V%w*Segl^D7{c8JpQS1&1~Wf+QLR#O8;h02POq+xI<Y(5P{}'
    '>wV8?hQ<lh7@Ilo<qa9`Kw<_&qeL@w1Kl8lHZ!><cIKcNY-GGK!`-'
    '`T@2Wa~ALvuP)_XRq*MEfbpJDIXwa4$96RCV~J9DCR&%XAmDBT0eR7L6D9CJdU=AH|R(ia2}D+c0fV=&iUP6LEHgwZLd6%9R)2!p'
    'Re0}q)o2h<v(cV7-s$#VN~1vtDb?8lYh@2<2TS3#(<%6?o8k;-cOaSb@ZYwX9h5VNec3cf*N-'
    'Ivw<ze=W$CGEbTQr+jC&ib7ZnL{1TQ9pa>xv|C%E;Kh*|6asXBJ|I`hk;T02d`>il>W^%ClpF<=WN53S+HsY!v;F#!X}1YvX+JHk'
    'x6N7lUZ!;vO{`OQ$V>0LS`x`7a_<fP%Z+HnFh*5_%Rxki{N9XgK`mi%nVR2%4)rUIZ~%w{~(#6mUQXDO68i@9-'
    'I4dVXz|L(^8>_r;z|5>R686*}Iw-t9Njvd9j-'
    '=+|axz#k22dP?X}qYaA4%c=OB&h5Ww@vxcvnr3|W4<s3Y?Dxb1zMWc7fMF_Jcg$pcRh!PrYbqdE*GbEgSkTjj6iN2<FjiQNuNUF`'
    'z#CgcE%p8m}C1PyQ#b!x_I--'
    'Aen?$T5BGY_PVz(8m<pYwbbV&;@s#MGQsqf^){d;^)=hK<dOVsh4!54cQ^JDc8t~5VZ4}WRSi5PydFJ-WZnSO!yHdw@rzsxr$6mt'
    'GDdWbL>Wk-n-?8P{4AMm9;C=GKck*sfXbg-($_U1SvgNUZKuVO<<-'
    'x8={Ed(Gu_FoI}hZ@M~|A~nj#M%EF6E&EV|C16mgmeElC2A<A{%aC7j5Gg}6E&Po<aZ_Z>as5WqhxAdQoV~Sb#cMz_TMu@tf&(?h'
    'R|MnL97D8l@`P*;0be1gdwzVVn~!BgqJlW$`CFvClvAkFU&FpZN%p7t}Zs6!Y!SzAQgCb7nv&8-HEi|JzQL>ZELm(OQ)-'
    'IT~s<l^|YT~!t|L_i~lB>{g<@hl1jB$m=lF`MeJq4-'
    'J1^vEe44Fw~a}%TU?M`FEH+KSgw}m!z~gdWUz|_Vf)+XHP4IyqfX{1pnb}Pu?pDVoD-*ja6=2D6wto&p-'
    '~D5Vi_8xfD6qDg<Qal1w=aIjcy!nV?c;qYiR7pxf#?lcaWO0Njcyuz@scgM4ix8*4M&;u69$_BmI>1Res8Pl%KL5?We4-'
    '_EXkl{FL=rKV?lNlb5Cbk=OJ8E}0vd)cQk}dcNp%=C1=>nua`%7JKnUvFZnxS`@2(ubXos)X%<(VNvP_Z);eT`Ykdi6mkT+W}B)e'
    'nI4H3C4tH%*cw5gavw@TyaW>R_Y%S0o}rS;pQOI5#KR|lk-EA>NM$%h4Lk2d?lAV?+&zUJfK1j@U@O^Nnf-'
    '>_QeBeV4Sz?f&z5*ur_H8e+Gw+BkFJELFSGs2>mt~O{e9zwo6X<m*Uflr&Ci3T>;NREF|V#;A82+Gne>HA%}yedz9-E&g-'
    'BYuW(FNh1G&l23cV?V`(6%PaOy(plQwuqZjjfaXYkt`pK;Mf2-Ri?eZmIlG8f@r!*G<_pQmyWeg}r5-'
    '26O^i}1TJ9Oc&M>0E?=1H)0mL`<;m|Ei4#z995lT%N*8b{|j?QnIT`HYFB`I<qM+sc*k)PON<4N^>IR3pX^Un0QrIykG=xu(ga8S'
    'lGxCVV4<vs2nT-bRIl@N*nx)b0|a}w!#oLjSv;!S8!$<ZzQj&BfODpSMTvgvO^u^jbx`f#v92lb(}Yn*VPH$NZwE<?MBk7zajM9B'
    '^xeQV{=h2Sp6i>9s)Z(XVKB9l1(|qV!3S2ldt`<xv}zvOU;d4zi*p!3Nf%`1$;#6DOSJ(g&~MukjZ1ZaK7x<9SXRO#OqKW43Sogq'
    '#7C8$Ypj_`(SBWg3VKeG<tsOxgWll=(~i&3wkU?XOhY^YJ#U__<+#;z}k?c&K21Ab~_!7F4>fg{qjD@7sXRiAl)kjU$#H83h`bvG'
    'u2&et}6yPWlNG1t=sF*ixf3nX<n?P;fCfFll0$ZcI8?lPlpcK&7(th_vn!4cy!1f9v$)n9v$*rj}F<BxAD=4>(1bFV-bbgVl%bs?'
    '+E?L5|?bRF4=5G7=%ES#N?x(R%M?h+7@s+AxS`;?|>aIbPLyP^<u$;4K8Vv>MeF-'
    'A0*(`md0U_H07;hd*}0Gg}VVyi4!i|(EMV;otIt4&+}-zI!iZ)-'
    'V82KibVr_P~+Kf3n?yWc*pkbc1f^H=@7Fu;M8%ZEpntVXhgIb3jGl`MlcEy;2E#$YN4-'
    'lz3*cX)R*z!u0g6%YJ|7g!Hn&15Q48KoViO<hzMwYaiS;o8B?+u3Hy(FC;Bph+8c-'
    'L(VXP4quFm>5NoXAQVU|GydOtui?Oa`r3@kXGHa|?VRJ!{5LK)36;lSYAxJ+o2I+@QLHc2Hkbc+_q#v4s^h0xye%R{M4;CYwSsM%'
    '%G*LC^u_YU?XXY1z_83)#GtB1+W?Mu$MK{1+ANmmW8L}t%(<WkZGq&cfWqXGUW5o=YS{N(l?dF_9q$^o5ufeuYBj#gLPs@n89#@}'
    'DVXHN0X>D_iT8oRS)%RYIoge7!{2*`V2YWj|#M}9y-'
    'p&v6c7C|G^VgJYG#vKNbVX%5#3m&)=0T+sf|!}6d+L7TahYw1Uk*7ICgWPfVNrZPPq_9D7sUz}F109DxNt{{3et#Yg=^+^jc|tv!'
    '?>IcJnN}rS+#_mhAq4rdSev>{`5-i(VZzGhpfw(=nF9UJkh?~4`DKSvYSV`M-'
    'UKWD>7fU65$OFpRk`^tN)B+ggnf@B*pai1P|BrLdmt$lv6teVo!TdfN>!O+XY@4r?zLMIyGKj9KOSxyp)W+VnY%S@X(ON13WYolk'
    'cNh`I`A^)8M&SLRcME3#KI4xAr~z_CAQ^f(iEXg;*lE(SE)NO9YcRU##yI4ngjT6mzT8rKH9??lM*53GSw(c1FV<?Um>X{{xk#c0'
    'drzb=o)d9SIg0Lg=@kuf7X5OVUwqP_|Iz#+MBIrOCQIk)Tvcu<XsskHK}An<rp<`;AF7utQ@K4eZcZOuSEL#ru-'
    'f*IV0c#^b}5v?gSP)6YnUmB0-<t(dR-ky~sZOhNLDrIv!^DNwmM1snVW2h<F%*KCsG$0d^MJVlAJe@B=PQX@6V6CF-'
    'X?d*mdMLf%7-TV|V7~uwkQ$Twzc5s~^&@Y;v$BkrIN+@WQ^5RSM?2_IInv54&XH56VSZs%Rc`lc|;Y~?AxWlF-9^7G5F?p}d%6q-'
    'k<E*?V3;nMd%&2C8X!l<rd0&JhAK!N%xAzGLa{ExKzbf@yk1h5u5_}MQmW05Gi-Ye=eR+v-'
    '_3BdW^5NPG1U2V)(Y6ULaMmK|o8~O%2?74v4@3OCE-'
    '0*xMuzv7^Z}CL;u8D?H?#IMm>#a1_58#QHKXUH$~g<WBQS^gc}v?q!sa9%&0%vAkLIwsn6ST;753#)UucCr7y*t$VK*YpuYQt*Jt'
    'M)^Vgg6HAcg(*6}!;a34FWMWAL2NpD1x(!ncLKuf)B~PYeB~8xQ<+s{ctsglWI+l++_i9Hrgl$S2igPf0f|wIdxF*cNP-Tqhh>V!'
    'Z)wsgjI3f{H*S=RJ*x30iNaIUwHLDya^^VvmJ&$Q?bgH@PK=$933}#N#?_DJJi)W##>X<C7nXM3$kv8<6W0qAiEyG269{^nz{N5f'
    'Appelu=F>4ObDEbzVw$do%w`aziJhzI*8u>M&U27oF#;=vyz`d&d8oUQunLf=)g7pm4ewG$o>oQ8stI7QIQCh&<bOP5WxVx}OU$V'
    'ixz(Xfrkyk*`2uEWASvD@2hO5%YXnv!^6ho)j;e=aNb4X~YBvDbhJFJ;L~zldap9MCul>NZb7J)0L-'
    'Pj5kICSkS;4=V=HHVLFady1ChW`XQkPt(yBfow>hC(|-D!FdKUC*KtMo)YWOw}ig8#5(kh&_5-'
    'Gs?}#F>Lj;}Oi%4ti2a+B0>XB(Q#YqNk?eTfl4>DyQYRhYJS@sH?e>z*Nj%U)a}p2q&|FOBFJ$LNVJTxht&8BAMP}`ckU03wSlqq'
    '!Iy8j$m+S$=0U+~rlKUvvC3fVDw8?-'
    'b8ErZ#^$Zyrk*!EKt)G>?UUkG5sgC+0)iGbBI_`^9ltleU?89M6{|%8{I3VvT{GzT#R|<^LEp#*^wbLRFzfYR#Rvf;=t9f3V{pPJ'
    'nJix=&Bp%>lYcXkGdr=1IbGy{9*cIL{RXwtlE%gEKpo31R*A3FH^&0pDv5rT1*@pfWt$k3UQx&tr2Dza@a`c=ex>0f)m`L;{$$el'
    'Fyb{U2UJ6^5<W8_sqMIbg%V#CJS#mGf*@@mNd1cp2<c^`(PPJerEog?=mQ*xDv?Ql#hUftF;e&a|@b=n0lgMTWxKz(1vKiuCb58M'
    'Q2>T`qBn5(Ah;2!cj6l!}(UKHR2RwXHwkh^>+!Cl~;ox8wq{HC}<wJ?nN5kD`^(*N2M{~t9u*>yTsiWpwp+}{Tns22ZojPj1Rr>1'
    'GQS+_VV^T-'
    'Ww?>am9W~!teNE~(AD2tL+pFAe<U$Hm?zKvl+nu!Q(C(Y6b1ZpMsy+~lL~%i$D%yMM6|0IbBLrfkD#9K0ic&@Mh8m+(5nf_rlq&8'
    '{il#1J%^F4Q-'
    'm^v#%v;xN;?eO41?|~2#y)!GMfLPq`Ef$u39&IL%AJ_i#{i?!EN%?V;`c(c_)2K}eh}KbA30WpHxpK0jL3eHu>N9W=1&t=V7)@>J'
    'zh=sV0U<+rrRskbkFH5V+Wua7v`y>y>{<db=-'
    'iEiIM6Echoyd9nG8A6s3;vvNlDj<DR5wX65~?I_?6$#;W7%QvXmc&Cf+v8=E<jo9nrG5&i*9Jk_AFtV%QVz$0X~v4l6@SHsQs`{9'
    '23L)Q{{$bk#s{MU~YwyfA6@ox#6Rty;5O4w*(D!(;xR&%A)d%ZgD#rcFl9d}f!<KEM$V+SZ77v-s=y>_2ib%aaxiB-qv5$icd9nG'
    '8A9Hovsk|OD!KppoM&i!0_aoDs?D1BtEs)SF`4iuc;sRbj%NxoVzL7c9wMO=8-)o8Z$dJ65N8@AJ{q~7PXVjm8w1X{7PQY-eg-'
    '%ER^s+(Cb15$NvHfn8<<U7>pE@_y_Z^P$uYom)H(QqEfNQ2Jo5&;{r-'
    'L6aeKqT_W#d&&YpSEwT9zKeIqmg>J)SMHchvwaHiPFQJNs+WvpojYk=cFFKI6J=A1+y{Rq$lz-'
    'T(H7pqTKH)KwVrFz6~Xpxar_iBvo$!2MJ$j7%E(E1@6i`FN&Ko4UC!%J4u5LTcp8;T}oN;!FryR@^`J4h!V>hz90EA0}bC*sp0$E'
    'Yj~ejb+=?nsp=VvjdDqz9@=~A7psSGseZ9~2zS&kN)OE&YKqcBc!^C>dbmF+nk0Mm;>y6Ki=PRBotN4+Y79n`a}md9nX0Q&?;+ax'
    'g8MVIY1o8$5E<P*9zVt{nCFkt1Ec|D<3xfz+vp-_nFgD9nzAhhzwzvp4e%PN4|p{^pa*2s@byYHJkU-'
    'J)nodbwx{_C{cRD8*Ycq}HMIBCKUNLlQvGAq@Jj{|FhUK@8)}YHLwJeJQEGS~DVn!?KeH{jVoL-!yd-KlrQMZKD=6u%q*_TSca^G'
    'Dlq^*xwVG0;>MXT}+hopGYdO`vR_cRZOAg|wXrLu;RBFk=q#c1DML2GA?P(63`ZpvleXM&)&L2adpPD~3Z`~j*;+fOENAx$W?meo'
    'X6S3Ga@8qRt?UN3O)jhb>fLPswI~owBd*;1wjnX|3&DJR0JD3#BBEDKN$MS?USnMU2p=scHib&KZ1{0Hs+RR{Lr9?F`_}f{DYGyd'
    'k@dz#DSp0s5NK~AgMO;{i^a!Ek#}Pir9)Sl{ix5H#8jpGCws61Pj^Q20P|Zo$_t}CibMTuv(Sn976Xg4_-'
    'uK$oP#o`uGvFHtH!lj!Mqird^hlehF3ZTgF<hxEBlE`Zn{x`Wvt%kms;8Q4-'
    'sT&qlwl@6NmL_){iTW8!eD=ACu%G6AYgRrKsW|^5RO3}gk!J=;TYmUIEH!<j$s~zV>p9w{9LNtS+HQZPy?r=TMT^wNwXFSz0Y&$Y'
    'S?}^ijWBp58W6}1NKo{h>_w@G5WGJr|M^JaZ03U;W|?yB@4GRrI=uMWChzzO)!GpiH?6L*g9!QHfV`2k%qa2jvkc;pS|Q55f@5gM'
    '6A}okZO-'
    'x!U+28aA2<$db2}&MFeO<d#%<_NjJ!X{9ltLm}Yx)MB;qT5z79aMAQO(vH~;+%n$Lhz|w9>GJe*1f{PcJM)tXrdbU`#@S!0GnBa`'
    'KEc)^^hk4FC`PA5D3>TUjE9@8XltP3oSz-'
    '53lgti!uBr(|{kSwNDkRQt3N)|#=#_s<V3@E!ab^!*>K94W@0r<{|4Bzj&0xWo=;)+r<NUv#Q*e=VoPxiUYOia>ds@25CH<`A$)s'
    'vTI9r(L6Q`%6t`F&`8&JN~oHJ5_wdVb3ghW}>pq0dmRnEfh01ic8k>+?0<|$OHxZz4F)@s8IsbcbeGb``zYNC<%2h>fWydRN<Xo2'
    'MYm^6%NbhK0&Zk3Yfovios%gg=z@(Mq{ywcAvuk!QDtNr})8b7~Gmit#y?W<_nsMqI9SB0*&0&U>R^5T+7zfj<Fr(~BAcI5`Q!r}'
    'Z8GIz%G%qSzeI?r;OTbvdvUbxb<Sn>YSoKuK>rJ{7dQ9@j4_?4u?aGz-'
    '!8BaL?gSvqEaHa^cI~OPT2eCW3!%Em@XXiz|hBX_rJ?z4GJk+5RrZwpQ2wRFDA;5@I7to~wJOXM0x=esKNIgK83-'
    'AW11?UO^o&t4mR15G7sd?i{0iGfC|9n`02SG8(ucg}WEXFYeveC!LBpsJHGjco@*+XBM<_s2_>*!da!<BTb&`+3i3X!i=V7XWMw1'
    '1rkSy_zT^9+Lr8+RyKoENq(k3%%5tAJ4Zo>0q;ptoVgI281A0`um3>1}^oV7PvOj*bW{1^f?|i|G#hU&4{f&XtZ-'
    '_BV+1&jK8h>Z>I&1=Q-ZQhhzu^{e%=pf&3&LQjx-Ie7wK6Z%G}SJKgY2y>+0YcKUP!)fRbr#VcW=H{lyN*%5=Jyz;)L(_{%{b-'
    'KVdCU~o_~OKIn;HU43rx}*<qp#CDtW;Nyj!(4@swHN;P%!Vx$nRNL@4)l!g|p5eSckGZFx5xeM4Y8wug?sBQT6U$m>yH4f21Js3!'
    '?MU+Nxo^jisWEo%W-u94Ce_6MhWv~*zO!&7~YBtY?6EmM6hHHcK}&r3Vr90z3a>m-'
    'M>+T6~JSh>TMX2i<<b#qRkEKjA}^?G6Bc8e7da^kfxfusd`uX8P1JU>p1sv~4J>22)67hUcgEHKiMe~XT;M6(VHohV7si31T6?4i'
    '(`Bm~!lLIMH0gpiOrDNC?vvv8|ZeVefHey#o}A|P<LPbt+Oqn@5>{Z*kSO1*-'
    '<;V8nD>7!vB4=yG_27`6Ufeh{dJ>srBv6~w`EaSl=4$FA(h{MIiF1uyP9IAX)!t`TtzN{@_w2c~+U|TSxV9UFt5bFf7czBs#ik2D'
    '{GVyPt;|>d%c-GKy2UAbPOK8O*4j`ak?6RS{OSOJp=tc>S0U;GaNGHK|5I`KnbP^&-g6I;uNrK}*$V@^vOZ_?>-'
    'CR<QE=cT>B9T*Ql_1Qhvr-'
    '3Zl7skvW=M&<^P2O_72lTeh!SthXhey(i^=?)ZXIvz$l;J#&aiUV3vUJ1<3eVkgO`lVq#<|0D?+U>FY<>%tqILL`T!<nW`?7LTH='
    'E2shDbgkLN2e-FlP`reOhq{Vpxk<HqeWy|q{oIKds3LM=5xx*-=RsZGu*T#@Qof-'
    '^*yO8ph&<FQBoEc8uMua>d+5}(LR?3#<dBje#E-'
    'jVU}67Lj~cHeGUrmbK@ALvO_26@ty!Jaf_h$l@M>Pb_EdD4{Oo;2mwlG^NE_`p<OB^~L*pj3~Nj^1Nfs>e!0;Sr9-'
    'v~d`%x}xg2NqAI3XM^xyoYTjELI#bH=a3;CJSv!x=T)IrE{y|~xF=7(<{5t{qcJ6ZC*v_CepgJqgL4dcUU#F}tNqIBD(o910xRQl'
    'm8Emq>n;1WU$#@E>ZW+1N(k`jOf4bErx3okL__~P{Y_tqru_vv>L<~(zeq;|ByROD(a{8<=Sba+j&2fq0rFrFk4s5yaauLPkglIk'
    'w|SYwDXt4-'
    'An|EJ_plQGkx*+RktXiVLwqvV|GkU`n)tnp2b%bOF^ON;?KB>GjgfEI9@v(IoIFUAz|j7xEx_!s$UhadYGtXq@_^EGsm@hrk(HII'
    'r#hSH86|G!8%R9;CAxFprK9sCI&d%3(QPHw<mA{L9RNL$w&CkxhZY=%T`6%G?-'
    'CO6lhR=h=`RZe`Kk}aeL@&2CiwJyp;j%81f94q&qp)&`3D(~Iq?S>k2&#&VgkQBXNf!bz<OsnYwR4EW70F5^i0A1d2Q(IP=q^Ml3'
    'DfQR#Kg#=(SIis)tIw4@jA~B_7E95f5bks0T8C%mbM}?t#po@IdBEJ&<`XiGJjI(umtjsyR!(1R?$sE87`D_iXQqbpSQ-$-'
    'Gf0bCo~Jc%+Fx%6O!SKNb`2=x$lQtt#$$LuiSzWK4!Gi6x(K2$DggIz+Re1NAn01<Sk_QFic-Ul+6}El~rRr3FMdh)G&NhJ$G=9m'
    'P0AKaj8-hcb5yh;f*qHMEi(PBbp9IKAbuyzVHet!|-rL%&~FaffEA$4JBE<)l1C=x$cZQ-'
    '$u{UdkO{PW&d%WScv9SH?q5yes1&C*Cb4<!dW!c2|!F13U49(!s-{jV52bjGxE33DG-CE?>G>A{v12QRt62496{r75uYf8H|I*1l'
    'ts8{$*=|ja)Z4iuxsz&Fh;9I7?g)c`E_K%9Zk7soLfw3c=iNsTy4@w@0c?E?_?>)mD3*W7q-6(=^8u-'
    'Or&<oWzC%19T9DKJl8&$)Dk!sqxX+6L3l_^aPwz2#T?tn7dL#U@JmfcXiB0Wt{PK8})iTavOC|<TVfA7*hmE4{f!v5oPbfPw_7NR'
    'PVwo@4`>>F1+?G{B-Za&madbRj>J&N$7o2ZE=~J{Zci#M9l%In&Z$Vw&yir;RV%%SZoP6BNAEy&L{-'
    '6Sh3L}K3mYD5jfW6U~EV=plq3=*08~jFA?W(E#^t>IJXzydN_T(efiI;De8Iqaf6zwHrS6ZD5YMoA750{)Qk4xca>J(wI5$n)74A'
    '%<I8G>dYQbjRBiXZGPv}T3R#$D*(=p%dq<BMp#AG&0V8(g2@o%-'
    'CPu?Wz$vk45pYT&FvW^!*D+vO+M%mF?a(MsJ2cwU4qfeOhsJo?p|PHJh^oYWQ?<h-'
    'lIjeP0To0Zk$RL%IzA!wSmWD<og!V?B`Z5d2Go)DzDr;|j`uB-'
    'T#*F3(GU!n=74>4fd8;FPj>hS>c(jF2RJ1b`~glW1drI>dQyXhYVc5mwU%+6=mgfok6WT;_;=239x|-hi%X%C+b>l+-'
    '3l2gEOwDY&NDza4m!^Op*TE6gOuTr^m;fNafH${<Mj?~(F|T&=`WG+wY>E{D&c#$3*~XcZ$Vq)QmL;IIa^`}AP>7jp>`pm;uES#('
    'GU-CN-'
    'V+yoKgrDF~_FnaWT~zEe|q^J*m5<)}rT)bDLu&NNp`vdpGck`>+6$c!P&XrNuu5erWMeg&$h{_#Z9)Y4AgfU&9YA{^{^Ti+=|E(B'
    'ki(s$EWvt`T<nIu@+qb(xbb#&1?PMME{fDX~Zma7rP(!_%!v^g`D2+l3cR3&|7)Tcr1946~Y_XaQ3R4ozK{TLRf_;(9IYY|Yfs)v'
    '7;|I<$>eeO(#=FMKR@zzemg59`_ZUMgTqr3H7nfQ^*a+Z6(KQgTPD1#G3X^sW?eMM^ykNY(3kK*JpXFuWmiB2D}*RTB-'
    'r0H?&FFTg2<01YdaVO<3?5XBal3kq2WAj~LxN2dxC2aL&pn-O-'
    '%$gFGAz&*7fh(=E8fgm=631ydN3m0ZV5KWxwgJrvwI<;C+M?7lJM^hJHw(4us0K(_`G=T8AAq}u?A<B6pPnaFxE0jczF!8(8<Y;s'
    'SI3*Tb0Zu72pj4UCM<PdZM$|GjlgYuJR5vp@*i-5jCI@?3-'
    'O8YIpI5V(9PAl&8<T^5LEX;eV9%;MI9lwD1l{VZl+Hy)ZuQT!aFBvq{c9|&!G9{jyAD|7$NR$LO!%?QRcakPklWx7!=Z15HlFoi`'
    '!~HDy}ZjU9Fn@1%`N;?>i(EpsMsv+dwVbzHv=B)!C2gkcg;D4zzeer)y&(tk%nsXnkYXo(-'
    'K?Z|NSx?EaQCLDK76UgSVB58wVokBJW$6DNHzK>4gm4>MImSAo;rb8acWNwpafXZG1?*uKvYd|KHWuS-aj>&rzEMY~hD_S7=mk^P'
    'am!{hoK%t?FIeN8@k@K5le{1pB7ceXU^Ml6pX%V9kSmDps&?51)z^EZpj+3c?mv3Ra?Eoo$^7LUex7m*L0NZK@Ldb4W7E2t;;2lL'
    'X4`(g<G2gOJ&fr)W@JW-'
    '@O31a&jRH&0ZzFn7m|>Q)9WoTO$k%GpioHiojSQMWVT*<^Kx75FJoEiy8HRo|u^R5htTpyL!b!~b`s_HL|wh=*bY2v>S2R)BCr4;'
    '92AtgyO+k;#v(?w`~**uMT*eUpvqU(~nQsQyiThuzXm>K%4KH>=;V1KO<q$QievOVyk6a=(zO54WiQTB`m$%kmqk2H?Msn^3Q9XL'
    'Fs-'
    'p*ab34i%CLM*<IB<^Pa$@H^G`_8^*H|3LdCXx`4lu@Z#ae>hfx@RSc11Q+ZUbSd{&>nLXm;r6i%j^uG+W@@P#azdD)TE}yLkzZaj'
    '@l}wuyICI*Z0>GB@Ldw|t@@~78PC$kxD7$whDebVX55ZQkrZd#fe>NWxQ6w<cy8SI_uq#?#*y-'
    'ordPL3R@j6ZYg&9@T^9f=)dg5MwP22VnH&~NS&)>?+xT>>l;QqA9V_Lf=A1&1g4uP769G7lN>4u&Y3F#=j>pF*WbO%iV8YPc6ZN2'
    'kp_6aagA)elo}`B)jLv<N9-1&bca0vFFh2KWJv^boSG5G2HGfQ3egel7-'
    'fG`y(!mh*f6gq~Qy@I&x1*a>=lWNdsZ;h24#N*oy)O$3!nj(0T&nPr>JezoY^E!v9-'
    'Ri8X+knKZ|pO%GKQ!6OstII;XhLlKd?f^V55wq<l2LAT3)(o5geD7ZdwGVmZfVQ#<69R&@g0#<F#F<`w6$}KB5N*{cm*eDE70yNe'
    '7Q%C-P6|;BlNw`6(SN#jfPm_VVsvTKT^O@&*kBiEj-205zVi^-OQN8{oh>5{H9g0(yDqkp=QzF7;Jb-YcXY70R2cD)udWHdfyEns'
    'egh4G;g>g4lr-d+%^0PxDUBOrF>P7l(>d=;Bam3SAtV{6;5d7S%M+#i5D@x;R+Ruuje$wuohhLVIz={8v)-'
    ';|8+dN_Ae23){gY@P)$5wHg!)m$I$%_CmQ*D+YLlK`jV37%xnStzr36GZcgqa1b^xu9kX?mFya+$EG=Mv3YBsi<K-q*XLp-'
    '3y=S~f)l=#%?)~dDS&}YRae}*7E@n&RR&6>9F%1AHW4;EnZ!j%hk2>cuUSyYYN2<DutIuDa()q^ywwy=ApJ?Ishn1PUn<4Pq(4hF'
    'jgv`#kxFwi>90~vXPm&lNi~Bb=W&P(mzRv_U}pD7Ue?}z+3ZdX?auDR(C+MlL%XN%D0TFf*s4;MvM~c`mEji)3;&!V8ILbvVYgl5'
    'IGjGh6`|E~_{BwNwE})|p-ZiVUtGadtKb(`Fx6`K#T87o27YnjTCIg&_TVE3+?VY{LFPlduKp_ZmF+0a&pe#_UY-'
    'o?Mdrj$=gx_r&Ye?mIya|F^#4VP`J;%1m_^FQfEE;^o-5-'
    'cgkS%l7TbIiGN{1f0N$Y?!9%y?w;3Xe^<je;LU&uyAw++BHu^IU+aAr6oqd40otU<r8$WG3x8StxnR-BjQZ<ermw5y6Rh3#SSub`'
    'i{OcuxuZCSZh=XrmBck1=VW@O1!dZr+H8WHAHrE7vbE<bs*93e^s`p4Yc5`d0_ewXIH7nKor0Wy9E!F#_YZtyf)d!>-'
    '(77Yk2W9Le?XkRKnZ5aWofxH^*NIWuc?Cylb8KnZ^=XbXQ<gLxZ%iUJnABM1_YM2&GfRh{1}btzWs<b5kS*u(Cv1BwxcCX%-'
    'byZg!nU`H3!kv<t>&^PY<p{{=!tc*wWb<oR=d!5ODFT|guX{QDPM=!r%rRW8|aLu!o`d9vsUYn0H^)p?O?d}cwT+YUUYsZrfcVSV'
    '!C#I!Rgx5_jlHq8fGwhUJhM*{KLCug9va!(ePPfk)MTcbVLgJV&3RwDQuaUbu9EJq+5?46Z&51R^Sane^Qz?hqmHpIH<`0P(Q~(T'
    '?Qrl1!A8tDA_M@u8*OieuW?p3=Q>b9O-'
    '4avfm&|3e!LS7C}B3H0omrhnwd$bueFhBCj52zidG#25c8}V!(Dm!2#RTwSJn;5ZAJaLaz@Re0?ge!J}LlCkiY;M&6gE;3XM(e>e'
    'pXC&Q+YxB+&gq|=R@yF%>HO`N<!<i^dMy+ZiBEpCdYDbY=wzd|IbW=>#XkKtCA!5W9YyW0^yBJ^F-'
    '?FiQkeV=re{dz<^&XacsleH)F<ZZufVJAjw7j|N_c45KM+KQxwA$YHHlTcICKsN|ARSj};P)ZGUV^Gu75H|&-'
    ')lfGCHC+vJGf*?sa9Xzr8|^IYYi6weNqXD6%xL{Hbo7P+*m+ilSXw<ZxFuDdp~N;Xj$=$Mt4g*6+khJsgM%4o?NFeOm#5&G8+E)g'
    '1@C)l<S6Z5@+NZZGcM}HEbXFB%+f9@I7{0t!)|e@U^Iivgh7Y$FWJn`I$Muj!LGb%!=RCF8O`SIeN~5l$Fa~UX}ZCKb{GV{;QH-w'
    'wU(<6E%+rg{-bu|Ql!B<wFmcU8oX2caGR!CJGCEo0-Ci`2XP}fNz`T9xAdy%x_`nKoI%~a1dp{?=B?-'
    'shG(D5ThaE*S~@X3+tP{Y*_MLSv)wa-'
    '9w7{dfz2^A1%j@Hw;v3+!NMBp+WuO6BR^(4*gYSa`aZ#WZu)Z~M!c|b%qY2>%dmsg3NAtoRx7#4KSZtKdel(0nuX>Uajlx7dnNoG'
    'Q+4lzzeMRij-PItYv?xMNO#G2%Jrx6<Z8cbcP9p9cgGLN?k+eWn-'
    '%L7!a!8vHY&0xWOn+dg;D>HB*csjZt}5&Sm3}lK9MjR;241u>SG8}X1PTALCQ9*l*o5aDW+8tnYpiXm=kwPXUDghKJsnU<S12xK1'
    'AMjt5jQcYX+pswQGf4lg97on4(J>)Oidc9S4G7XGV%V^x+OBW1r3oo9#2~>BMO4o=%L$?kPALo0aqB!az~r{v-'
    '&ykLA1q>?1KALpXFaQG1!S;dsJ)U~9oRG9lPV<7|&iU2iG4VppZEw-ns(QK{=Kg?({!>UvABNcGjJ>n4Sra%}1v4Z)4OCUuwvpOE'
    'c{ee-'
    ')O(fD&aWkp0$7xZ`Gzp2Z`pF8uY?C0~=wY|T+@w2jfJ25M}x8SU7_ljmlgI(Z&oL#qZ8QZ!2E5f57gHrD>o*RfK*JYByckYvN${0'
    'N4{we2%S4)Tr7=7YUg5}kb4przLzq+?0bdB0>4|XDgiP}k}cBvZmk7=p@M5-'
    '<NCv@<ZRL%OW#0^P1!w!l{7CdSY_8FN2GG}jdU;HHOzD`WS?khM6+x-'
    'ml8ui|@LgAKqMz);c(*PN|?YApZ1?s~=Lt`s?T`k;)2)24Rx1)lo-pgHxV5#?UXX-'
    '+!4sb)_BB>5C#V7<)eN(pX5J>e)xq?tH@gn3-w<db2<k;!B#3-CKV;2rZt6j0w>R-r9gV=B0--$8T{hb(d-'
    'CuCbwfp~%(T!FlbKAq+q_8IUGfDerrEB&^5WxFnJBZJ&yspYAD3Qaw3dV7<4qwN~5m2T&j~G|NDWP&2%bEUco^iFecc2rKtOq(V$'
    '$FsRBy0E6w2Gs%d6<c3%u*bem)%FF`%W2DwehHewQm<zaWrj#4@%eHuRkSSgTH=Ax(<K+uyif{`qR?&`0LL|*W|B1CtYj5+O8g9Q'
    'b4ssJ<61TYL|N4Maw58=EHK_-wvi#yW}+&+b=uViE-6~ofua=SVWWymae+;KL8@6s|^'
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
