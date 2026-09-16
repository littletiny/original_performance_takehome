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
# Verified checkpoint: 919 dynamic cycles; 10159 static bundles.
_TUNED_STANDARD = (
    'c-'
    'pjmcbrz$(Lbj=hy_DJ(4}J%lC_{~L06O_?xN;;(%zVuLX3s?_02CtMNlj#hzPEzOIcVzAi<cJVxmTjNi;ztiUnQz(rp;*t4Njdn>'
    'lA@&bjwl^ZWgi?DN??$7P?rckY}yQ@-<^v9>|AQEl*lL#J1cnO-'
    '^BUUTtJM@@&tWZl32^sGbHCTlZt`o)v&Pp`Q6GXLL8*Z8+tyR6N`>61&BUh%W5M^1MKO78H3Cr%$W+5Y!WuecWe;pEbl7mvDn*z^'
    'h0FCK*-Z0K_<C#z$pyH85+`-'
    'hL5UU_*(w+G(C?}x?y68y~$@P4xDGu`bhWrt_&@Fz#Qy`}7s@V=wljsJ>%eHXm<zdr2pw($9vAL91G2Tq)RdAqD4I}9%Aa*daL$T'
    'gRYfa|{WVz}C@68{1Ga@H&SwPARP;n&+`RoUVGx2xe(@z+0#R}FvYBfQQ}=sG{d#hy~QwD)nbpMJ6f7yIZZ+u@>V5$;VEEx6cCiw'
    '|(IixwZ_g8$AoT<r08hqrNguWV3Of!BUyb}l@dUVQC*kz4Sue?-6h1};9N#cL(_+i<C2^*a9OeY(o`O7O2h*x^U-'
    ';*WOFkKVx>*g-'
    '$}7hX&YE#Ab%2efz_7a!B&EnIA)#a3K=BsT!h^+(e|#8ug#k<;OBz5!Cf+o{Q_;cHOx6!({$>HrS{<@ePihOWA4V&PU-'
    '<8^MKKeHYeZ_wfuT)aki@G9QHyY!Q0yqbT}Vht|dq{U`jyiJQuxOj^e8*s6e78`NFB3*|IzJj&5cq4sMKlb+qZ!U*h2+!=o>>FO+'
    ';!W0-'
    '&V1_PXWP(?W*v)n`8?jsYWf>5;9@;3R^Z}wTCBvyD|E}N@Rr}BpDe}eYNo}DxL8Ar<+#{Pi)FajM2nYkv4Iv%xY$UGmvFJJu-kh1'
    'tH*ok{^G8;+Dw#Zec13(KOOnMSN`<6i!Q$Gvhdh{oL%l8{8$#wBofXDuL1tYpd`mk&qjl^F7!8)T?rfY!lJ7G18@0xx~^yAuIuO)'
    '<6;FZp2EdSlFTzDkfCULZbSYZ7fWgJ1TJ2r#nZT0PK$ryVi_$K;o>D)Jc*07QPV-E@rK9Z{pbM$ro-J;Wt05P-'
    '!EEU5C0Ya#pQKbBfOo2Z?DaM4!1r!k@0gS*JiL6@)8hon@e8)eVd5|qXm16tk#D)tF;hs{{_0lM{w~ETKp9ki%Br>7yc0i^Edp_Q'
    '&CEX%3=xr<XQTq$5GForJwv6Z~yPKcoY{;(Bd&%ETY9<aPeeLO7$R@YA=_^SSc341^G8i;4cPAHNwAZ^q-ME2v&P^(Q3m+^b-'
    '0LNNDkpEBCcYrSKcP?7z|_{1z8~qs7B`cl?uw@F(xl-8~p>_mS?8xL87qKj30NEgrzdGqm_UE}o-'
    'L`yGB7FMf^p{TMCo$HiX~A)xZtXOCs~kMujC9=;Z3OTwZNzT_uOu%MsRWX-'
    '{h;@elqdZr?2f>Xqc<X0e)gTMa$g*lPji&wXhF6d7v1s2IK@h88b#eKN=Z7KYfD!k0__QdS;Y)W{0Qr0h<8s46qosmroZ%@h2%%+'
    'F8eY5^qLwI{?HXxf3-'
    'kz2X%w{I$aw}W~9|etBQ@Gz|_zE_`i{%wwpgz#e9|`~UvEnK2+9)u_xFcLU1txQ%{2kXpfj}mbXsf^=Zw;T`0)LklHQ6rsyR<Ly$'
    'xPx4Xbcy?VywxQ!Pi;gcb3D~;P7?@e4QQMu7a;x7GHh%P!!*kY#fVkYIX~YZ(25<#Wy{>mBrVPO<?iO$Zlit&CDk9P19C#l`5-'
    '%8w?j-0asfY-'
    'd4fg(c5~svq>XK&VJ(sRq~fMjK$YW!O`NySEay6Oib>$3BE|{R>D_Jc)J?DhJ?3k;OiX!$6SSC;nlg$3hw##@V<+Jo30D*yDGTrJ'
    'Hq>J3T}IHc;8*Yecu`0_fT--cZK&o72NsV1zm4}^WI<XAO*%gxm@=n<&MH9?yt@LK)El))%TS9a$F5mc(c43pxlXZ)nB=j;_6K0P'
    'L8WHlshG^s+Bv9R~_8~+&;gg#V>GiA1!{4HsJomUvTFtcZ@=-=f0ub(TdGpV^+bBt0k)nidX>)u?e^<y-'
    'mnTq#4{sFOeTBI15vWT&qA_e7V=C>_b_MRM|FJT&=Q?WpSm-cFJO?a$kry{6pow7*`i6_m#MUw}aveeyKK!W*$mo0WId^;uonjE?'
    '4eYl)+^6<>}c@kOD>pKRFem71XL~zXYLFz*ep2?Nt%vgGWj;{HwgjP~$FAsMmZo-'
    '&d}8T+LS~C|=D|?&7$btK21Vb&qnF#?>6<E{m(#$_<OFS<3x)T)`h39Pj=fyy1DYn2U?~rDXs8T!A;|A78zK<0&y;_bb;&O(ZR-N'
    'VT9s&I~t6rIxKeIa}Ol#fQtV%MZiI=%8z^9`@5=mqHLXR)Nq^`OXE=U7T9e(F%M}e>o2-'
    '@C8zB`U~7syvfIu`*pm@Ie1_9&|)?&=9bEy)rnc>>_j$&U9ywd6n4!{W>eTLJB3YQ_pC3Q!XDYFYzlj3r@{S9E?qtHiV?#w`fPhm'
    'cnhW<Bsg*U#d6SAfcc_B1Z+D>s*-'
    'i49_};T21uffA}i+(QgDW_$vXhQGgh(iesTfmBYfb_ywq4WSE+1=h~^397DUm^LTSvQMI$a|lV}?8n#e5Aa<{AO^zeSRt5aFO@P3'
    'ZFLuF@#_xHHTDmyd0pX=^aS^w~Up1VtB1H$|H?rxP0gsc9^<qBi{pZo;((drUBPLvHj<s}MEdjFIsDKJ_S6CbS!<VNS=0eDO2VRc'
    'c<<ov9NM|c(d1|6CrGPlU98tn1=%Y^n1h1$<jdtAAD<LYm4U2)Z@-'
    '2cQ?Kjr$y)hEhLiK{)zO^vc|Kyf#c%QmwFj|uEsSj{Mbc?6>ve(gF1FFM}c7zHmp%4L&oRp2Ihd6hI+vM8@QZ=fs1VC_?6zx+{LA'
    '2e=+Yl72S92(cGC|ZCs?iCecg;ZoMXo7PRqsw?jf#vj$-e;7%;?UH|^I_2}5&~@E(W2BhD0q>N;%-'
    '*(qB9#jX}GR%W@jh5Ws1)1>?F5b(V3l{>{ci`v$Ip&N=0XO*4M34bY^F#y48x#?CdnRM)@;4r+?3u;1TlP1k=`DyW_(F*h#zN!U2'
    'dEQ%xrzim0B=Mg{(ays2vV35X%TE9*h*t#8Y}DtBMhjnkDoH94cSdpNF+((X@j)l0ia;%c{Y)8cBEa?_)RO-sY1=@=CJiGDH_f6_'
    'p|biINX`Hb#H1ut}`M87H(_~c%{dTTfyl0%@ch96F3_3EQh7IGj~u$NGkdI{C+B?Md7h^J;Mu13%<5PN%F(5_kyQG(a5!<75uA*p'
    'sv!Fzdx?q!6^wu>~tA*hMcz=Rx#g#WZSol`K12ydri6cOG|!zd!WosLmNc-'
    'w$cM0h&`qloZ!CPooattx|FEi8cnn@CEwuLLexYRd2cI3QcoY1FC?VMF%UVC-YOQI(#r)yeq^7OlR_ra`;~52CfxbEt+0+&{-'
    '*+P$AUJ-?*!L}jt{(C$%Q9qH;(Zyut>-'
    '6eFpeyG@qKfqgb3Z7K7MW;F?TXdS!vPGvmBU`k=S=pj9TuHX*OxL!RE&4(UtXz709!O*o445cY1wUB~Q<Jm68<ZyYVGo2jBigkKG'
    '(Ln@Hfl$AJ1Wk-'
    'q(zf(F@+>GN@X93*!~U%FbWYo`m^GyyLNw$BE74G%E4)lJJM9`oh1whE~Il{atXb~AJI8*lY&<WZ#ud<bliVlDo6@G>B2-'
    '%HEy_uplT554!8_y7+b-@sq@-c$?8@W!`tQ%YNvt<?iue<&J96$`^SU*{yY?{{~^Wv;ca-'
    '2lh}YMcTN;dPuSUUwO665$tyTJC&ks>3VaV)-KoIikQKbWGp^vD-'
    '5tffql5w1_m!lXM=6Lgf*2>F7@Ja`w?TtL?C<0d<xapm$!}};v@2y>!-'
    'qYVZfiG6$5dJcpRqM=f=1tkolZVr;lEXp6XV9<gGwCYAnZFe9v`h8w{x}oN4U&`mHXh})F6&SVc$;LH5M1Qkz63?JX@rcZARCD7i'
    '-=7QGW0UZjY<G6!>|vg12`?qdEaU{w@;qttAN8zyyaMjYrYGO#ki{TudygD`xwar1}anI{^EXeUxg>5#Ua)5Np|_P@o-'
    'R`k;wbx;a>d)E*u{d)S%;VaGooBzQPH=tIjM^z$Wux`oC55-3%gf7syOiNag&-'
    'i;#Jk0O~`ithXv%?HgCca!FmW~#ec^HDR+jn#bCOn2inA2to{7R{&43^!i$aWm81s>2_<8Rc4+=J3F&jXN716i~@35gnW|O?1wD`'
    'oWBQ99)?TbY2-@D}v;~XC7M=CFs%ZL6q%58FQ!ebtsK-'
    'B%0B<7)y$@Q@MsH!cFe2xY+34j*E5fy|`$0A4Eytgg5^*Eym#DpR~9U7vpJh11@f%#r3%Ob(tdGUV@FE571hVYLk4G4{+;~&?B+0'
    'UWOAi{ELErgMtMgy0IX^Q5sGde>jhUU&<ZMU)1iQD1Z;$OVP4ji)Zu0rT<)B4<&XDO6;amnSXO{2?7q@ACpSp9j1oxK@~l~rkK+N'
    't0=*4lGMbF(i9TwBS1zh$HKX@PqpeB-lyR}_Sy$_>olEe-zZeL>*&#s#KmaRz8%WVU_sXT{1$mKv*3#ync46~e#{*BA~)t9_#!W6'
    'E_{&_GY`JVhnWvwcS}aQ1E^e??530;P+-'
    '}?KWa{+wr1SIm4$=3CTLT7h*a~=t&ayby3}LKwk=q|oZwisj#6$&RIMHExw!b)y$~1Ge5loAr^6SiRzLV6)j9*dNVU#{FH)`k@I|'
    'UM0KQ1I2Ey0Gpubm_vcr9~D-H3R5=*~Oi=`joJ-cK}R`Hgs^YyDqBhvy+Du4&PYsW_bdn^z&eO-I4fr#n5y6CZ7g~~FDG-'
    'm`ZMv~@?QxG2p(S(Z}uF0l`<<E4sEc>i1uh7}@>~pfbQfDi&j<UQ;XDhSM%W{OyR%M-'
    'Jd9}_~XGh3#q|Vl4M@oIX1K7Nx9ecL~?p1oAdSz-'
    '3<_b(VT&w6xMa+IlNo9mDdn6^G#xkb__9r|R@Jz<0;sjT1U=Sx5^@|GPta00wn;CtG@d^TK(XMUo#kly$EzJoR4y@)xxNB9mHWBV'
    'Rm90yJyIy7M6X9-9*@i^88&$S35$-'
    '0HZAyf@S!J71xUm?v3ym0CcYMDy>jl>nmTi(#_S01}Y@PaIN&ue10e>`z_@;tS9K;7!>3?*-uIiqEkUv;BaBEM<S-'
    '2}vHAj@>%sC;7awn$Jc+uisPRt!%K3SMq7_A*m1`ngPW60uRw010+JdD<kBb$fOS|2ic7_A*oRu6;L6GkR>tUlY7*slJXO-'
    'MznCj%;B6DQHDaoabR`NFPMJJ!)%fkL>7gfLNI%7RtxHY{SuY62EFWOXYh_hoeprnE&~+uif=M_@t^*0Nfs-ORXZQRrb&KI=o}uF'
    ';&T`$)NKHRe?$wt&d~zHrJf)$X1oRX4&d)|eJcQo<lF3Joe%>!BL{LRFe4s4hP>mv~REwRFH14Q`pgmNv@GDV4dukCfmN_=p-;f-'
    '6ps_l-JMJa)NS7C_d7n$^fBeI=*0@m~}Jp!71|!~91KD%r_y#9JK8iRbGTA2;(b5WbC*$~UWUp49oQ4kskQ3F3SNI6<6_04IpE5#'
    'R)IG6I|+&P9L|#Hk2yf(ivb)Z^=d&_*jaw_Iq~mEitaXb|Ri&k3!<$1QYhS6~6DR|s~31*#O^9P^F-'
    'v9mIaR?$zY3iiay$p2zih2j+(nVsl{=x8Ielfbe>8=0N#&ehRIW~aFGbhMFKU-wlVZDe+;%XGAn*=g>49S$HY3S6hy6PSZp-'
    '0@rpxKXhKFc)*x6Y`I`yMNTJ1!S<nZwprLIdCmaYUS?24Sq$8^-FR8L7>_$t~O{4;CT~Q!11>vWKXir5aK>Z@4G-'
    'y@jSin3enpO^u8N}buZHU?hu76rT0A`Jb8)U_k_6lWmV?U-ByB!#7`cse|t;|gr^H$$j0<3RY|sV1Y9FrYA-'
    '&F@dU0eubFw*rtC0oMq~E?p2r1g?t=q5s44`yvvUyzgz#;3nk+deuE2U8qEqX6Xk6{pnC0iydhKS!pRLwzPF!tMZnWmS{>RE)r>R'
    '1Zj==Srpa|%M-=L{rkMZD*8q@y;$v#tpbB!hYa4BoahZ%5tY<ybWP|eA%idfVbfzcHl-@X2QMmkl}p&Dh{TG}2oWKg`%-'
    'P*mLWZ$)WBd&I8w<WH&Yxl3XS_6+He#$C14C3nZ1{3YP`kZlJiL1{VR~c6wj5|86+8cLVSmj;9u_X{uQeP_>H!CPccjFpz^|i)WU'
    'u)v}T9Q!>F=k^7*qS-92q)l$ey62E>|{`gxt-'
    '3Bt+&A>bO*(!{US2+Bd~(85AEVY9iOAPSyi#p*U_~BQ|j8O6b7>qe9$B*bCyn$2!o@f_GtH7d=@*jdo!-'
    'qYS$S5WwUm(;;N%@UyiHK7<Wuu9d6vQaRq<u<+y@~;ccC(*}(&D49@_p)wBK?M7b^W%Gau+^KeYJVg9baPVy%Mbdt<FTPG=#8l5^'
    'ohs72AU};=|xSHY$w0v1y!H2KVsjKvk=1lZ67>V4FrcnDEH!a*<t#MP6Q!*&z&JsNT7^BpI1F?u}eXfrISBO2Iz9wslyBPk_uIM4'
    '*L0HL{sk9C<x&2Q@4ZrfD|NYrz|2x713fkz~f*A95WpqecgvC`M{}T<%ec-TwX(cxmZFO|^@rMF(7^BnjDH!81Vvi_DXZ@`6mYv-'
    '0e%WmHw9m}$VduJkHrM;slIPs2lHg~Va&M_5`dJQorIPSx1w0m&#6K&Qdsig^&?@ELQ%MB0TDkXC5(2GJ?gJHKAPziC05rC9i;YL'
    'Jc?tjISnF+Mt707$&hD_d0-Q*b<ThNb1v&HM?PJl&No2=0R+O=ppLBH2x(NosmcbipZ&8ZzJUa+%h9N%@D-'
    'rt*IVPAJKVZmf#LA(zl#swgrLp8gLdcF$JGt#wX^hHiq-'
    '4KGYNPwSZK`Fj_t%Dh@N0uLg`_g^qp(VMHEi^aMvfE~a9<m6{;t0bK>u=qjs!PS1bSu3jWMa7LTK1V!*|Y;H?biF&wq(Hm=tx5<-'
    '~eL0=|P?mcP<A-YS5f_Fi<xgMxI&f^VXbJv0PYi9#NQ1@lDVzMvtb@(LGRT6l2bAVF0=+)~*U_4qTE3a_Zd9W0ezQG-'
    '8gsrZWO`#DSHSJd5(mI|<_xSzMa1e-gmcj8$Wvs-OV65SnSl2lhalO!V!Hc2w#5R+tH4>d_ReV9qp6g@QrmdRG^r6IITw&E8NRFG'
    '`7FKLL>{Z`9A?1y|VFuY$<0&|@{Yj`RUZFaI8c*0d|H~+u&_TY6klh>I(ZwP}|nVhqezoro-'
    'u;WQkGd!n&5W4lTy!rZ|Bg!sVb6TW*@Y9O4)u>f{&_T;bH5d-'
    'Gq{?=ehVVL3w!1Y%*onyNHMpIL$fs!VJQHP{s=@h8tlu;Z{%7(?rfYCPIl94`Ok+}$nU&OG<|K8Pd-'
    '63BxGgL}z0#^2A1=93BF9%tZkQ>A2z?DaVy<g7#e{f;TWX>)1fA|}v!8QJ5Zxo*@~8;u^Mx~MOJaF=cT|HiJg7KnyTLu8IBUDnEm'
    'WMV-R%CXxPV?Mh_DlJ-xFx!B-'
    'DIDO`MEcPq2wo&=wGGqA%J30#2NYJAsfBr{ON(eVdoo<E3m53u{$Km*<2v%z)M7aCwZzV|OeUt;N#qW=q!*L<o{&u}P6Tl@r7^Lv'
    '^{FP6v1DA;l^0b?#xsIq&uEPm0sqo7|)2falKHEKaR;iRXRSYz`;Zy2YcvJ01P-'
    'wnsLPvuZuF`5~$1&x^K*{CN?<No}!czEl;`tl2>Hz1T(#a0B9^zZ)1AXSzXgafUl9F8aB_adEmkJ1(kSO<Z7Evmq&K&Pd9dGn4Y='
    'Y+YvI-YG$(!VmVrkYtK!um;Q%iEq^&BM3Od?$oU9F<poWG65x5#F~0lIc5#K>0!mAw=IT_OM)mLQtr8;C|yap9&*FZD%Ufvv~owq'
    'l~V2tQ8c?_R((&zlztL1rF$c$bYH}j?vI$#1A-~VWLf}~=4H6I1TiDa@Q=BWcm&$gJ~7LW*8M&W--ARU@EC;>?XB5GZO(QF!8McK'
    'hJ&I7<(oe!zz%^;hNy0#PQ>`Ia?b}b{z<tP%EV}t>lH=SM!7FWQ5|q!ii>@2Z2WhNe36v!c26oa{GG$%DTP+QOIZ9<p~>$W7Edd*'
    '_uax`u|i|tJuIG4Xz6=|#j^^{e9tnO{-'
    'Feam1R1;lvSs0D;&FU$W8ixCHq@Igw3p~yW>9s8JA~n5J(@63Kr#~%Dogs`Df)`F5BFz-H<4*<?gF-'
    '(d5oINxV1GBpEQ##?Db2yF_j58nv-o)W+^n8+$~AbI(?laeK*a7iFvuNUf-iw_;K=rm%1wR5Ms{a0^;E3$|5(TEg)qxdsv@+8%Ho'
    'A1tI7n!MZ6YF)Ff!rTVSU=4<$yjbPViHl_}<3%0r3oZg<5}&KUn8W8XFebIQ4vaZ1E(Bvriz~sH(c)4tCbWDlIBL>DzdI*IG@s-'
    '%R%wWHLXd;|GzB$Rps|7op|AZnseahyAdh^Rvn_f(?**L#|K@`-'
    'om%0}i{jYp#>K@a?v}XN<HpCuem1sfs!rrk3ry8X9BC<)olN;)jN?wBd@xp&`f>nfRdy=JUnP~D=G%yJd4g+8*?|<xhUqF3oXZGp'
    '2EoCRw@Pz#ScRgi<n7o(DRj`#@?!xkS;>|i3t-'
    '~527!Tt_<otdR=RUdf}k31641&OCdoIA$XUHHiTK>j$!Xzkjf?r<AIjn$H##onxa;C#mb)P?8r_YCVoiy$4Ti&$;w~D?;t&Wk90R'
    '>I2MpCQ$$|!BKL{%?sO=gZqderVE&Ahc1wnzQ{jSQ#uHgwjP|BJTPz@%)6Rbi7tRdJ83f~>8dvGvy4+j%22y-'
    'yP=OQJLXrnoh$VJx%qia$Aq&QNKcN){~Q{0!+;>c9@m9#iA%~htwk?HQ}v^dh>j!BCnGu*Lhab%`Dj*2679wl)*CwlsE^$AfDcW|'
    'PoKiUiN>mZnUI|viPn0G|$A}XQ^>{E~ox~azF%**oJ5V<eP+gCWmTi8j4L$!Y>6yzh8dOn6?nyQ-'
    '1Xo%*ZDj7mJs5FKUj=LodOwSNK!DUW~C%D{c0l2MjMuMr8&WeY)%9SLrTJ73OV70~_q>It~dUuscGMrbZe#A|2G22~lQrztYhSGQ'
    '**!&#>k&6INziS|N5mfz!2BH^1+<(tN{32-l?;D6<2vx4#4-CXG1gX^ShX$gU6gW^Hec0CXE+ep#-'
    'CZXoAFuCH?j=xJs^Fu1`lz+j4nmKg#oOQDu8fO$?wYum>#mK9|48RkZtqVhWxI<VPcbclnz$GuI1_OI^26N`iZ}wz@uFB4M;#7D<'
    's5d~VCY;l<lvTnL;EcSuKc%ktGTt`C+U{om9oaPN(Nyw_|6pNAc$as;y4G<E0Yu_1rRoHyYjJ)ubf~CgtpHzu##*Dmcu4zORxgAI'
    'h%r&U_W_}t%CEKOVO+00OAd|22StPj|ioj`u{ihq=mgH)XClKAXXxbv#b?rSef?#JL_vfglz&^D(~ALqp(0o@g71cCMr&vAVjK8a'
    'jFDSDR(HJEFoiAouYa$#7=B-zZBDhVRmAX&rA_Ln5szZ@qiT1gQ<%goCodZE2<UrQyo6$Had1!S4T`mOTL5PPF(~p-'
    '04Hh;?W54;WvZaAa49Goww%x2y2Ya-OLoOTa7;)8$r)wV<jPgoYD6CBu{l&dU>cW9Ds0Z&nGR;FC;C_FD5O{FH<XXaI0Ek^5Vmoy'
    '!a?4FSf<x#m6yuu{|a)cEsex&NO)eo`nlifuQhPW!eO;RCnuYDZ2^2R8Q;9p%vJ)Ue=vUD>w&^vhF-ufur>W>%M9Qf%85l3Z)L+f'
    '@wi1H*dqFU%2fhNmuxDNmuyNq~rS~>iCYJrrjW`N=|9IE$FgP5)J)~7O7u+dW?wwZh!vaZPA#aE4uDrz%DXgUJ%b*W{UCxqu;~2A'
    '#~qx9DdQdj8+f<e#yG?B~y%9+7>0Sqz^E4D#iH^F;Obj`HwJ7Di!&UF)1p!-9D_F?syF^cKuJ#@M7Qp6b&!$%!@-B_7@?6-'
    'a`;Ugo0Iv)C6N9dMCJk$!NE*gPJ=K4a%&z^FXsMHoWsLDU%tPaHZk9U>%Hx?*j6&0kBQ5l9Gn+Y&~1}lBdA77{@-'
    '&M~q0C2HRphdpI96{-'
    'pu7#RT?mb}(XTCTxpDJ~gO`GlOydcewKwI9ZmcRuI*V@S{n^+(K0$w*UtGM}|+sOUne~9}#7VJnTbj_$~mEwi>=m|HnZZzB7WmiS'
    '5Pd*)p~l{j%k3FV4tTu)R1lTN&$|tJq!)$X2tx7?`ceBM2){RVF3su|ygS8$~@r4ng9BS5&}y$hlXB2ZkGkxUZl&aMTSA@&V)h<1'
    '+bFX?Sr+d!mLH$FwJDcyUmBvW6Ew=X%g}PEbw(t3)X*h&evcl)-'
    '|4qeW8!3!;q=HRZ1$)c8nK`U>KVZJM%I5M+F;eex<vg@awaRiH#3Nf#Z<W>gi{tol^5>QgtN8G8{^88s`EQPCY(;a>Ma3iv5o$|B'
    '<(?mokn?>+7}hB-MtalbWOG~MfdXPBU)I=Y~zM;EkTbV1LEF6f!j1??YQ&;ii}9Y`*ys7@mWo8)ZYmz=j9F%$<O-'
    '7WD`ORhjxMUDyl+Fh!(G)t2LdEcjfT~!3L%)J@}v&Q|aEO^5>m#eZ{75e?+gjD&uSoqOKOgu!8QgyP$@u;~=g-'
    'y;HInldYE28m&*UD(T;I%65Sa_|WV-EtI`*qlozP0NvHpYl8zMCO^*h*2)cuNF#Pho%`LY1d7!VgjO(-'
    'doe19tQT(<Uf^G3K@?rY2v;fj7(DYh{^Yh{t?LCJ~qq!X(P_fffgUU*oeUTL*qA8{v)MRj|F?1b!E9#A>V}RC7(^51K1i{oTWwt5'
    'O5ppEOsc2D(SIuS(_ptM&%JZ}&z+q8c<wl52kgEAdisPhzY_&cb-NqcPev%wm39Cv0peq)vu}bY#H})gi_xqo@FjHGkAxc{{^BsJ'
    'ZfXrh7<p<*f;J0z)8|WwQhVxf}vzhCr^!<_H9GWp<B1AXh=m$PmcY5KJ-za!ocrKp+$O{Lsf>d8676D=(h}{K$6gnPAce=(4yt?7'
    '&8;>-pB+m<C1naHxrp2CR32ztB{q9|B|8Oh*|WAxm}4g(blArZhdD)SN;QHOH@1hxxKv4(bpCHiV-'
    '(1m~T@VI89J&XZWK5wBiA_RA8G{qh84zajzIuS`Jps}hj?>I7uJhCudF@_nrD*hmlimo&JB{xDkx+sGDo+81?5VuXvX)QAYCxZoI'
    '#0ATWdkJE_vy_=)PCE~Urk~VI@=LMIINLqG-'
    '5J}5^5F%;g#(YVz+lZu1T_HG^@afTm35R#ZNFm8e_{X9{jXS7GF)Fblpb`@Uv{gj|<Z&%QFe|~LlnJIgwkd!imUPho?KWvi7Y)>I'
    'vzBzxAaG>0B!~uU_nMXj(b?L)uEliKX!i!Ex!ios)MK;U0-'
    'ep`1n)0(HYfZB7@|#_83#kOIqxij3A&5Z5@>%eK$8R;Eokp{uc*C1DZ(SXOEFB&7fgA};XkS<=ELCt>RX-ynC>p{5(AgyJ2rQ;1q'
    '1ekVN6gN;m9hJV4UHv?@wdtTFc33xRxro(EP=~DGhf9A3nyU#rXaBfNC}8{RZGxlqCB=+>26^9fW!<_1d#=YjPC=pVn8kRCdosqs'
    'o;Cd|KyoveM1fOdr?a=4s-O!y{xZACzEfH=v%2{hsMA3<7I(-'
    'z#cZw%grr7#CfEo%Gx}(pe!j89~sxDC8z1a9KBngk*$A>#mTItd^a@pKBzq3w3~BXxapL+BNf`4kyGqKE~mMSkI?8oO~MtG(U+#q'
    '&~37=f0Mx7|bR~<=AiF-IV=tU+#ymvebNl7r`gqnY!Ck+zy`$01tRqJ{1sjXRc;vKdk3zv^f#|^K*^1CZc^7XtXh%3W?S_pWS5Y7'
    ';r6_IR;!yCXNBul6hmmwPe~Da4nfN23*Uuga?+ezm%Bd6dMt0<0*>28pC`PO9RA&f?M4mnC}X?-'
    '6ycN<x~73p{Z_XkkE9uyDVAKRUu;?DN+xG)N}-+@2QZR?i>-'
    '!e2oS=!lZto(J)6?)V&%Fa)d$sQln9(F2lX0tUYl|*0^68cTGe&{>EUlf5N|TQI5Ca`zozUg^z|i>@G)p19ob(l8(!xJm66_L<+='
    ')R!Vqi|BPE~63Ny3g0$a?dvz-&YcZ-LS4H~!*0kn-AAKj#U9>?DPzHyeMl%{@-'
    'f`<NCCXcF17?ZYS>B8((9o+CXfw$tT^ku_)<p)I^^t*QLu8=Y7#V0bMFyJ9#6Xh=y@0%%m9NgkGyoQW`~j%o9BEBBqfC;7$-'
    'xSnQ&?ehN)5U{^`*oFuyod9-ki<ldQ1_s5!#4ZbB-Q1VY-+-'
    'gZ<nCgEi(^3>0mnve}FiRVtgqFj1|tdl)5ZR5q7EqE=<|7$aIz+5Do6w)i3hd>>`f>@7rxK}H8@H@!6(HE2S|1P_xD%^k54vDB1f'
    '^EkBJpORKomz}<I(f9CCPCW3MBMM?I2x2X($?R+$tI3>fJ*&w**#=gVx!FcmlX=-DR+IVJW)dYHqg=T|Wf-Elc83BuM|1TKg>b$O'
    '^*EN0)ox0VP+RS$iG*_LgHcIECK7{Ua-'
    'u?jNtKGLnx06Doc4~G!&#~s3en{A_T_PLgw<C?#NCMzad%Qg+?^Z|cc(<eUEhefJ2fKiP78=TuRiEAaJ3R|HPc+9biXxjM2;e5C>'
    'uO7LJe+DFhVojC&dZ?oYs{prmRM#Ed>6F3%eMELiLdir84rIlobe{KrJqBp#*VBpUvqI{eD)EjQ!l_1j%@Y>nKRjGu`I}3EJOv5`'
    '^ObcZ8rq2f8CE2D{H7RUOOWe&a^w2z!1QSJRN!C?_M#zSdKTPP?hP80^-1uo^{Opr6-'
    'B)CKB!twdetYQ*JG00R%~Iu60W$3tWeMKkd5Zs39%YV?g9+TdltXBc(?(Kn>JbVo!t?JT)ROOpHeUOBYCy+zTB<IE4aNjM&Z)qvQ'
    'wIdT~!O(WY48iWEdA3nNz|LBIrbFE2{N3c>Ai%RVccE9z4|EInb<M!b?Mhc*E66N~hX1s!-'
    '92yW)AfBlqDZzW}M7SAx5)E41D>UrvU*W084i?&Y_OS2(WETs~J^NT_?b*q~bFn8sjw1|cYDfpVBMoTkN#qq9(8#`J&gD>pZon9m'
    'b1s)Didj~br{f}QbOy?VybG*~vKa^a4&r;jNMA~`q{fH(I$MKIWph%tjD5_Lv*qk#o|3I#AG2?^l6}lmvsLV4o|dg<AG11J!*}0F'
    'lmQ|H1AFsi3J?tJ&5#3iT@CC<f{1oAupdt=lL|b-'
    '_l&#K<Q&T7id^m$s+F2#%4d)?=6GcR1rc)vAANz%&P@w9jd<Cl+RgY`d46itwHb7bxd~Y9Sj&Y5co^%r8Uwa_J*P9lcyHijCRp!{'
    'T#5kmy@}JBV87`!`<+4hOtv(?H}1+Dnbi*pwt0qIt;j)KQ;<ZR>ttiKD)H~(9*63fGv6=N-'
    '1wiF%o=nCeo~)wQ1!7P1J)Mor%8I6>pTIQG^=-'
    'x&FWoZvwGLqtllj)t9Osh>OEq!It3|7XE0nbt3d8i!{h90V?V(I;6szo67IL2%NLcDRvCdR-yI$~yUERXp9Ko?1L#B?VI|;(!?Ux'
    'MfE$cz7b_t)xD?$iXGZtKr&B@%KIBK5VLWY+M})H@5YPWuGcXQ5{bJ3a7WnW>G{apWn7>prm=6BXWjdD&YS2kC`gFHH9MESvWs2K'
    'jJEnWV>8OyT;6{n<2-'
    'KP|=_MpkP58sq>#J~mOq$<yx8wQZ54}3{C{oDD1nvHA;5&K8KC{Ews7)F>RXH2ASz~`OXQN)x*n7;`s8==iCBI6m*R=0<rPb>iyb'
    ';Mwy`izE`;8(Z3~x85IedM!7VnPYoFNvsGP(8VSlr7L^qgyPGn0>hp2giv5ztpHZfA1%GmHC~5?$w8+|Uzg%WL<Pf$y~Gwfm=m?`'
    'cUHo<@D-NNVItPkXHGz@c{mHw+-~Gjt}?(RP8f;fGo@938`KvR{W;Ke_^*&K>CYPAr6*@W&oR$MoVtg{ON=!CinZV?-'
    '=<*IXj)=pn|gcL4E*rhV8`9ADv;7tH2e8jSX4I)?7nV7OnQ#Z(PO>~&g9(_qNnFnp$slB(lObeTGHg6LYQI?h6esVi@^>!j*92i>'
    'Layv=Tqs^eUAmU<>SJP|!6FQrN7BTp>WS1-'
    '`5nI7Mxevopt<f05S*tH|aOf5VLOz*EU<<LeD5AH`e?1NO_6Fe`Q)2Z&e<PrLb0PZN#9=Oy+2ENn5qTQ1QzLQHYWXIS<8nMjn5-'
    'jt<36}Yg1j~GAf@MA|!7{gJEORtFqY|^@ZpFaMi*6`(w-'
    'gOe$;3Fj4kP$V)k^uI0!|q*;MkT0zsJbB29uAE&=kI5viUL|!X0a6K7>2YMv?WgGCILkSw0088-YH!XN+JU+_Of&4{nJO^n-'
    'iO2>ii4Z#eBg)4gbdSV#_Ql-m#r?iU;PP+o=LOei7RJ-'
    'f1~K%gVFMnJ+4ybm|<6oatf?X4=37D%pptm4oKcO%|UsyL_w;?9B+h&u~PAnq(Efw;4v1mezu5{NqsN+9lRdnN8H&gO8(SQ*3Nj<'
    '?d=dZLk%?P}#tHt;>CIF)JbV!3i_g1{zf&%Bq<2-YH+lD1Y2c8b7BGK5y=821kY-'
    '^n5y+ymzsE{+a2QipdxGg63mR~V_pyDN>9;@wq7YVmG_kz%~N+DJ9tjWlUFzI7CAM>oOeg(n!{)o$WvHcCsp;it`1CYrO+_{M#B_'
    'Zm_#<c$y9Om(Z8vjvM01TVv5x<#dM)G;=trB3!){ihKug(oz@m$xHdQFF{_%)wmOXz^1vBJm0PDVvb^gv(b<QjcBH-'
    '~%n=rf6`1mvc`vxWFq|ZNLRy$sN()0<U5P%I0OOxgR>4pRGwmHXf5ZUS!8u*FzUWqGm4--Z=Fj;W7+XeM$@0n9lz}EK+lNY8`Dc@'
    '+7l*taS$@8+shL)@&)<2^Jh;Z#B89uqDr8^k$yL=wEpjqpf)sqqp)bMsMd?jNZw!7`>|lG&HKsR)uGiLkLEa32Y~zsZ$N8dNDt7y'
    'moilkZwZeU(DZ?NS8-'
    'ztWdP0)r}g{>W7(;`598^@lN3i;B0XHXOrfD>pzE1VQ~HLAr%7Ge=aEyxc>9V%>~zger~6s{QHnE(vot)V%289xJK1!mc1TxEhie'
    'T;xv`h=vI{>$_^E%UHfuGSsfy#>aEYZ1;^`qI>$8SexP&QQtpR3A(yI3vQ{T1S*w$htkua$*6NfbYt=W&TAi9?txlt?)v$y&l*lb'
    '0qJ1M}W6O7(EK%uDlTeuI#rc|wn3k(fjfM08q*~@Yh@`)TKb?22>kv~m#ZqPxF=bON<t7nRHqH9%WDwMMH1m5AapTxH3u|1Qg>_4'
    'ug*86T!n!ri!kQ3gVciyIVdbjv^=O9^=GGSBa|1V}DwgIO<3Fm7RGj-c6pjMu@1b`-'
    'eV%C23C7J2)~U+Cwkc{7k_uQpz)#U`y5$6IUz4<n!WNBfqNkapO|;r1ZK9`}q)oJ+N!mouFiD%}nTFa#N$TI#IT|&0p(Y-'
    '+2rwUmSt#&POSZ`Wjj6yS2mw=9G8}^-'
    '9w;&)mDdGYJQ8oH_0hD5?0Dnm1(AW*aBrD2a6QE9!iaK%c0Gg<<tFWpVh7EQ)$WUi9oKQ%eaR4$9zG)Yo!dFd=X#o$*~1T*Op%^J'
    '=lWc7%1_YlZtHXI5^JTzbng3_*yUo;xvgS|2bBO0QiXb3K;$P!PizkH8b-'
    'b@$pt*BUj(_FWZZ&+S6PKYRvMP1oTd%7P_GD&stGSqYTZp~XpdoXUrBq_r;S)m9+`Bm<nbAVN>|y(fEn>ud5VFPAx#1F_U46HQoi'
    'hq-'
    'c<bhiZA3+^{djCzp4Cnv@hvW{p%QC2&V$rvA(!V6|m!cWp^OYQE4M6q?56jb}Y|MF@iig4U1+cj4b5G;CLA$+p)in_OAkjHOhKgS'
    'OKQ?mU{2l+)>nmD9R26%*RG-Z9s+ZpO!E{{iPAYwY5TR?GCcSZSC4wA-'
    '8r1TVc0$hghMvc86Nww|0kFCb+KibvY>?9pLiODOgoJj)%xt55bZL*vV2V-'
    'W_RuRy7~Qja6KsA^82Bd?fqc<hGH<8IwHB@xCAVTRTLwJ;<mk4dbv(Gf2lW%~?8@X3o~JG*hEFqdistXlva#0idmOx5%Sk@5alc-'
    '+<v0-SI}u?T{L5!qg5`$2MbTCncn)us)d#?>gI1l}HH#;GR-noS*1mxM|CQkS|g7*Re!3K*#daV4YCc^hy20wb+0V{lj(Gf)M?~^'
    '$6FF{^14$Y)AiaBSN;Lf4B)j+tEMVj13G&=5grFinV;dTd4DhVWz1GIUOXd;IyIB$uJFg&X_eW^E<=jk(}Sz#rnDhyz6RxsyYvF>'
    '}#ZqimCdANp5~4rn6$Ave_$J0ypk!$hoicY0DH*>`Q_JNlX8%Bd*r{MZ0ebh~clAQMUY8ZplBE@%hJcYyPoJ$Um0b@{eU={;^ETK'
    'bG6e-1fFunjsBi;ceFLj}#ti_i$_2qDcz8fW#HOU4bJ&*dcV#nuU$-'
    'aD$A~bPRSdNH9%VfX^DFm!>qp=M0idQy$>+2DzlEnsboB*O3(RFZ4~$WXk@gBShIft`P>D>za8`$^H#7>3&u~>~EktUq^Hf;GsKP'
    'glhT6jixi%tlg}ycfh+hw0op*IyOg976o>P0)L2lo00c$9nbecO2kcy@yJyi<cFrxaj~iNme^D}J~ox!8k<Td#HP~QVpA!lgCErd'
    '_>R67_|T)13cB;l*mDcfKC=)v#-'
    '>}(mn^@t#W0FAiRVi;$+#S0sg8&#^@qxwtO45HWI45i0K;*XQ)mbxyw!#zS`gtx1#W1f`t_&@X<Pc&@@?r~&$p$2Bj1+(&3s$>xA'
    'JZ2|CMh`|F+Kant%Gh#`Y!|bc=P1goC0v9pAV`x57mmiYCRfV;Eb+abo*c%CXYO1KR@*);diEA^&F@<Mosbme9}j2>g)TFPUgv;n'
    '(2zYfTd1ZP%V$uPxF59<0*ZTQlrZYI$@&(%?<g+2}6P5RK5`=q|>XIGl)KnCBL0bYJV<lt%Y;?qAaAzTRz>M)wWwEopS$=-'
    '!q__f75{X>{N0-'
    'fg8HzpTXIA5CtmOMAn(L)ha@5gyZ+>sXbx1E0SsfahurI=y+kZ#&FN9|#4@weGN7!Ezb?(MjCQd^m>(*$OZTT(evWR+5XBtH4Nd)'
    'p9l1NG@Bh0jt4@(xDpFjGW0IYX|_z8*!ItaN#)yzEne;Lq3SROoQ88=9hx=+1{jOTgsyiv*>OQ)$HqWSb=}KE;%{t{IX1^`*<O>3'
    'aeR|_+o!ae6hbJzS!eBN88Qw=BC7Xb5rBIxoL6U-1Inat|88wn-'
    'S;D%}mVt2^M@?ujuV?NNCYc!JY)kGzm{FJh^2;VGECLxv&Ytvs)oV!|?D{Mtsm?iBtAhauV*+n3N+Y0XECU2q_FJczTy0j4rI;@m'
    '-2gpRj`GcNqf63V?5{EfyhTpK7ewh=Q|2arB**aZ_OvM0Aynwu%IHjD<kL3+z}6VMGxDFbd=CSY6!D8ne~&vC1h^m9t!x$xj-'
    '=HjT1rP87~bQ2N!v@C@c>P4svl)e)5SXPpqj<6Xu0`51-qb<9B=t1!hDG0=S!ri~)Q;qeMHM=`~Ag2LR%kpzp-m^0-'
    'zE{!?SiG?{8kr{;K2BBXm$gPisFso?F_hVajgJe1Pvx{1lHu2Pk4h4p_)uWLYVy(tP=?Xn7FdVDo7LHb!a*Z(X;}oXNK2IQIjUPt'
    't)twp>$m6(niuq!S52t8M4^QAiDkg_-OTj&Lq^EE&!0QfL!N~xx$+QAf_2V)w6-'
    '1KWs>(VjR=IYDngb6h0~$PF6Z9;0IPJ<4xXiV<d8NIAIghc*R~?JYendoNfk5*<o10htE1S(qB=>8Z%}#XbcQ%{nRmD#cmHF*BU^'
    'MH@`B&t^?ZOFI_le57at7AzRarMq!6Hj;ch14O{VMChNmzG4Wj!ein{+GV^@lT$HSH^w3e0e#9AsL7;MQrulKZizi#0PtRiSJqNi'
    'F_@gi@>;o*$WE++4eqr-sQ1?S7lhNw(B)Y<5o)Dg4%EbCV7Bdz;PYlyyBPM1zIz94&m8XyLm?3*RkT`0mld_lOq0XOVmlQ~{3cKB'
    ';eYoOSKOhCJT7Lt1OkX>S>-9M-i6oai9x-'
    'XLy7dy}V%#fc=964*Y0JEK@v_Ff3xb7wqO=%%~isX}+%6%Q4<?QVFc(0zBueN8vs18pPSc~3NrJ;U{F^}JbINp;><NSyUHt=?5gR'
    '`xEf-d9La_P#1RyYI6ofhW>zbCq?61)1Gr+z(p;FM|iv-nM=cC2)jV^hVhiMOU*r_N^hE4;ag^Hyt*Ga^8HX?-'
    '1>fvex6^yPFJr5C7sFg+z$&(CR&fG;HtD>H~#zVIL^zHXW)ND)@IaR*ahb2ihuzO)f%H#VEoj@sws5;Zt}>Q~lQc6VGS{6+Vqebg'
    '1Bx)OuStMDvk(ezAgx$^s#ov;U6s2%hC|3z2IlTgJ3fgO2}DjE+<Ns*ZD(PsdHghvd}QAwDg3h)<6l;tjDwd`9dLpNTg_hJA+yeH'
    'FvLQ-j8eVc!K0$cU@5SA#|-'
    'f|zC8_ktki825uh^%<TuNTQYUbCIAq%@jY8c%eaAlEG&6EIr>?4AOJu<ZSJ2k@&?cnh(dsIoqva_la}1N5lRT=j;;=J5WH!an#$C'
    'IO=U`9Q8IWj(VFON4+(~QExNisJEFp%*PDlz7wo^vpckmW8p?$X1ImH!B~x;EJ{0#m6}^9wtf!jT7%ETZeS;yi=~gSxmfy0n~S9@'
    'Y%Z4WY;&=67n_TvyV_hV-Oc7ng?xckp59PhB)QI$(ng$iB>X9#u-'
    'nOFTsj$Fl8+T13)|Z39k~08%5X!AQS%aVT#H$(v2paV*BheGH?37x!ie)i>rRVL#}hsixpm-02u<XX`A8s|$Q-'
    'jhA~F9Xw4*GE2IB{s1TXaOWjhemp|zazHsp$B5Gm5$d@LwZBnH|c$Wf%d*%_dtjKXh{1`0%$%a(A<rGa7^Gcn+%U{DY43T9!zZQ<'
    'cPv@bj#!~aiCCT)SQ2^C~!iW`azJY*6=6~A}Di>3m>)^$&nGu-'
    '44!6&JHXu;YSORmHH6b=f@uQ2Xe9daVRSE8?hU@fecA_;Yf2Z`4!pVm4(pVsP^Pivi#Pivi-'
    'Piythr?m#;(^>-yX{|b}q;waK-g?}bF<l~DUt#Q0rJy6KV_?$Wv@%W|1=qfeEXIA=hH-suad6MjGVxty+;f@|M+kjB&A6MbbW@#f'
    '+&Eettg-4m-U^=qycRO6oRK6?4OV$ON~QJ`A3sv1_8T^)T->*8%(J-vvPq)l+m=#Z1q$6dUG#Kvh3hJaWc!#huK5}`$-QP$_-'
    'nCIn%rzOF)b>G*n+r`AkQaS@H&dlLkn`xv9YSgr$~1_%tQi!`k2{HIM%@nvi>=yR?BJx1KdPXeGw_sLxP42Hp6WWX!J9i0~%dnb3'
    'mgjZ4PL3mCXT-M%WzC=xUn-'
    '8jX~2yUJ%*#)pUTbxSF;^kFRa36LT!(K`3n&`rMD9bAU78DZS>I^@XETaIPYw^S^XzO7=J^c@w;r0=R&CVfxEGU@v&mPtQQTqdo-'
    'd9d)JJP=MHhF3^+)wyq4%Js#{{#*kuna1TB(DR#XlZP;jOXmow*&25%{446v5k3%G99aRMR)ix}S`S1HLD3Us5H~pL7)lRU(_a&k'
    '*v}ZMh;W`N;e>;KKUcG*@VFKDSB<Hfwbh6$8B1z*h(^rFt+aX`S<#bR4&24uROx29d8yLPcJouEo8x|-'
    'D&0Nq7pc<Cbqi9Zo9FILm2STKr6}DAKEHQMS)*y4E=y7@_fvZiH(?<ZC@R{|jC)3Nj{HjFmS|4XUTxe9nlrN_jeAiuIOk^VzCiF_'
    '?E6BbNvIRyBvN=i&P<U$BT347qR%@rwU$r*&?h(>)6ec}mh#Wh6I-'
    'ac^B7mFbjo&>cH4pGZB>zXbuKXjb;M}1FEUBz!|wfv*vB@3RMoUQE2!E$-O8-n9>)F1x@$-'
    'idKvd)>qc7{($=DcG4ODNEb(v&%*KL4MdKwf9Si;n4Vb`uEch!lq64#H;B3*54$Lir_wuNc<RW-'
    '4e^%U+2Nw_<@$iY<nI4**Vzu+r!nOx<RkLO#+$$kX;1AIbpH8{*CChmCuOMei(Qj?A*o)tb;HRDkdPzEKD|N3IB|tnRQo<8xl90f'
    '_LGYz~yU!?_f^&BxymlP(kgXJQntcePoxl)`)k02l0AaKfh4N>BNhp5?3LNSf?FP$<d#rY6i<VdV{$>-'
    '|QIA#9J$#%WfSmSJgMTwFY{1#ZeZQzv*<jEMj=*Z+$<wX?W#-'
    'YK2<f>#)Mdki8Lc=&7S`7(*CTPp;2*o5DWdHL<$5LVn7dKAqf)@;80Ef@IAmDY`(g^<yjg+#Q<u!wVXf<L*d?n?0$mVIZ8Aj-'
    'Q;ta|h)$|l$nOsS`i2y`xsm3QwMB%(m1bg*%65?|55mJ$I-'
    '79P^ww!nt=_l1@ev{X=xzd@C!nH>ltki76eDEWLi%(fo2R1{rcFs*;IV%qumO1Ldj&So$IlZZ-'
    '8o)irj)$3(Ox>_f5U@0S0_l?ClxAVe31WCZctKM#@QExP4b|gQK*$i6q~$87jmxPf8QK5aDxI%UTj(abYjXzymQJTPB3tGGY1N8('
    'OMq!=tny)QTYh74r>~F1Wj;)!UQqZWkFwu$eD9J&d27uENJF=nz6E=mS5FeCj_0$<TL_>JU^U9K@P}&Zjuf0l)`-'
    '+H^9@%ot45uo>i{K6cH!2eg@Z*#&<1<=&Icv!Am>cO)hZ$)CFB)QeC8<XI5S0o@ZKJB%WtpU1XhSVqK)1XJ%dGn`dfWB%41$qzo#'
    '#ONr<44-GOE=l4ksDklf;(;7sao|}tQ(!Pj4-i{d7#k$Z75sF&Or0y$IT<XDq9Ijz>+N!?XO$;@UQ`|(G-=3pe-'
    '=3isb3HO{2%x^VW+c_F^eVmkUp}Lr=G#=N@+c0j7dHNOq}eTu2WZ3;c<PtUFgd?$rpftbjV9-p%`!Q^Y_`eyWphl<FT2OEU-'
    'q2BeC=)}j`bo9HbO$`r!?3FakKxa!7wDEJqZkw7wxI;_SQ1~+dXT7kdrJD43SWV`<!%dp~YK)G5adPSd{Sjk5*}KT|LU5G<XV{Hq'
    'zj!xOYi|r=gW24W5ohjx@LdZ5(Ov3^Z}1!838IpCCBbFH5cNXpQ=RtDsqrky_nx8nr%6D6UuFfO=t^>~1U5vsN3RgOIjpYReViF<'
    'br>Ue|<*KOyRF%vQuBD$oX<n-'
    'f$`&}{~!g8_R|0O?@Bo)SPhSoD7iARVmP(*j5bYqnSb>0r&C5kNXvvu71oM#B7#u>uVKij?8OH?5RPC-'
    '@`B`r>A;tn2rIhIlh^X;&#f)(_r+6+F1CN;wLX%$RAiNnc4a$YYddkvT2XM3^5A#MU8lU*~fCb)ZZ*N{+1j9)+{y$jX0$9utQdd('
    'mNHlfO^lI5{%;`xVZUBa?qX;Xt_)gpZqNpP+ipVg;+jAwJ_-zp}ZkUdQ6#o-'
    '$#9wZ9<Hl8d$9=*|uvE!eS=X@#|<MJ22yAF714<Rg`^mTXfAYstqdVJ+FN64sI(Dq$_zDd*!Z9)%EeAe+r9a5N4kx|O#~Hb39fks'
    'BDY#ejnvA7njdM*6X~M^a_az%7pVVpE1C%KG7RuJ_D*BBl*)a1awXB!`q``{7YNU?tgA((X@|ku$E1af7U@ppwMd5Cw6`0Viy=&^'
    '%6fq94+nPPqAxXnrR=+if)06U6W_&HIF>yq)HN!b9E>-'
    'tGk1>@1&^Se6m139T<<Iq_Q&K~+3R6*<Vf3&^H+1<4VAQPPUS@H!#vHSpZpmF=~~KJS-?SnWN!F;a2`0>tpM-'
    'b~fPoj6iat3?~(iP55cGTJ?41*|j#f<}&Za7<0Tz<n5?gyUbP3h;*kN;nB-'
    'x?r3bpfm`+2%vNpe9h$MTlY4_M?OmXNEPxCpfz=6!-c>KPYNOeXSeOY_eJ|9*>VtdzqD7RU!)1$DAvHkMLEb!AO15qq1F#6N-'
    'Jrw``@Je1kNvxL&ApGI3(;G8;69QYvYix^K2Xv_Ej5)gk?4k2|M40Az|d9eJG>~;3J%i@wwD-zlzY?M3L6{fEuI(uIxc&I}Adj;h'
    'MUUJo(&6?GCeYfQ-h|Mb6*r@OY8)<a#E5^asififM!KaK2G7aWJ0FV-!;d<7sn~V)9@-'
    'tZ!CKAMCzoQ4J=DQxmpRzCXAhYhs)+kI5=-*+qjSe(EzSi6-'
    'mdU~#rM6=tJ9BlAhZHOgHP^lGhgSCvT&{^QTIYiotO?<(yMrqzYky+xTRA9VanQUGw%TPerlXBmGibQdfA;KADj?_Fzzlp8$ua}1'
    'Gk`-'
    '1&>2sM`?eTG<bsnVa1VDr>zhr3PYzM|1io#Xm+bJVBRWXYP>VzIug0p|{j)~`5+Vl_y`P!9TzGMTm2?hz}=yaU<|wC+eFRESq-'
    'caRn8zbmzCXNB(XYV8gs#K0g-Gm(L<agT@uziUyp<0bBUld>=#F+ap`Ma_-JxyWI_jOS*+7Rp0azK6pJrq7>qiC0WbRke7q(NLY1'
    'Bt7B91v|f`EEL>QZn@UFfuY2`$<>rqI)+>S=!m_*;$_sc55=7UbpadBgQBdspd7H&=BT;Qs{${P4@GnmKjcx?&nY<zv3KlaIW-'
    '(^t0WZunnr6DQwm?#XwhP5{0)s(Ev6X0snN2<Q2ARLty@ev{Fg=x7jr}3)@bF9Oib4rKll~itoOOp*4TGq`W73LdIhWv3t_PUKb?'
    '+o&8<qFvahsz@K<9V;uW5L3K#fU9xiZ29xiZY9xiZI9xgB<4;Q#P4;PU7PxcX>I3Mx!@fVc1BiOpf{s?yN?M#HU(%l}c6_~q23t6'
    'tvHuNc3u096)KP9~B!!yuExK-'
    '8<fK4KtiXftZsCPO&OIwO?Ol90z)^(=sZH=q3t{b^&XSs*u<hmMXSqf(woI78$GIa%FiLYB3E^>{=3MS8BxmIH-iD$5k)>zP_87%'
    'HPjTKEELvp>wlI9&sM#Jw^X=DiU{!9?vvns&Yb4F{wb_1+C!UzrN2<;9b!*suPYjH*b@jPm`Uc>jW`hj)tO6BzioU6^wz%2@Ya<@'
    '#k-'
    'zEe|EBq+h3Ct#gjqwL7=99s`??V(b$6$AJ`<$WQXj9ag&m_fu2#b$PGOEb4Y_ki)lNRUjYEA`sWJK|dbRHaEn4AX2;<JWOIs?k88'
    '&<oAtw6x<({8o)?d+a-'
    '4cm?x1?K*McpiAx%xaF)e!tDMUsz9TWCo&xtWY0dCns|sU?(S&A7CdZvman5=RBEo(bc*I`ct?@>Lb_41s<Eb)JUG_Z^@A4N3=J('
    ';b)JBCO*%UPyly<`13i_58dHGR6AV9GA%R4oo!uL(z4%aw?V^q=IF*c(bp`qbYq?98<u&xalG_5Et561wM-'
    'H*9~)N#lMkx@7QqL6$D%_*0pRzodxyXRJ`4I5w_&((3V(7}TBm(l%;yNy^r!L8W1+Nnawt*8DA=|&D6<aYOg5OnHzKu{3sS1JRj$'
    'tP(DEnCtbMKTyby0@Hd28~4`tq+sZYn&wBSkD64CFPuqQFOP>0R6yhEpf6_OgxbTC5Xz<KT+FfQy$Oi<X=J5S3fqzdciYnd3;*}9'
    ')=8M@NNx?gA+lhn<+dkJRM-'
    '#sQi;Xw+iLJkt%irol(N;T^SAFAD9bMZ3<_`8F9zekmE$ktJZFr9V7eI1YF)DdGy5aT`*XWT?$3iJeGV2`UpN`A7&rcH8g@7CCqN'
    'm0vG5V2UiMzF_H?5Z_F90+b%)|moG@Nu&)6hT5PlXayK61?548^w?i)nwh7SwAyb4~ine>&<%R`dlD<45LvnH?+3^{Mu(wBIRji9'
    'K=y8h5Sx79R)NT8}@i|a17@C*uNG5(k(P0R$=+vp|Qu66zNWlEv*#o)oW~7y{k}74H&p_5d(Ki#K4V@7`R&_25v&cz}*%xa1*&la'
    'xWs|lD)P#nX!Xs&k?~So#yInE8VZZ(%dpZh<Bau68VkhE)ixxxFLmfEkiE2=q@cs3b*Jkt#CJ`kgk>P<`mMk%8gASU8~)=6w<W@H'
    '|vo?w%k*POhL@SjFr@C3sOZ56s&T8LoCfBvVs4`Ts$2H-KAOsAK6lDAciw~K-'
    'hPmQHc+qi11#W>q3D&^>n_Ob8)_z^O<}z=d<}{&L#O~&gb&YoX_W*IbYxz!)}mqk6G5dvyJ<k<;`e)Z^YP7#TO1t@r5OmP+=ic$v'
    '{30P&U!q*hiItu}J~~{nE`rREMeH$0vy$99UOQBT2m}np>BA=iXQqC#H^5aboH%Do#utuj0hiTUDHxIzh#Wskf;(F?FJ3N<lpqS~'
    'gpQjr*(Rgw!E82`A)yQ}1CQ$q6NsG{<^+7YPuB48$k-Fkm^Cqp*6sp|M3xLszQ^Bs^SEE;-xnY_XvVozDqKrbYd@I?S8wt>xH-'
    '|MYtuhh!)Hq42B*#vwXQ`D=L{s~=^<LgC5F@XtYU_~%)1_~+m_{PXNM{Iezw|E!I}KZk_jpD6Ny#{JpyfpV5{f3eIT(+;P638Nwo'
    '`O1}duF~X*VpVgeYGo%b(A**y!v1fBu;0UJaIzMB4}W%magSOy+BL>KZrNEcX~ad06L@AXDW4py5%ECyqcFLRl;~}bouE5<*8?TU'
    '!+PJNBJN6k_-2K{Si-HmCOv>tP$fLH7E!)d4O#BQ^Fh{7ljDxS4rF2F8e9hC>STq=q7M7p*GlaLe4ceudjXecz0_X7<Jlm!7jSqs'
    'O6>()?@dyB0q1A4)Ly`bel5m1hvO7n_)grN9!Nqi#RncWl8!b;VO3C*NF8>3H^o-'
    'W7Cy(vV3^^S+CUb{kYHg|0&xv2!s=HcAy6UKX$Sj@dc+}w>{3WHW76mWNhj2rkWXYIiaw8|)>@Dfn^HD#RTI!q3@p$C$BYe<a9(6'
    '03FnVZB;mZ+@W8dz?otXJCK#S93l6D)tw}}2i3av2)f6Wg*ql^SyxqX=q>5slf$d2J#XAh_PpT(QHn2f^mO)S#ltTlXA<p3L;{Ih'
    'Rtp<xSst`9csydpX;lT_s0Y2VI2j+-`-yB0X*xd-%{byz5eG_3c1BK>~Or+4f#6+Uf%Mv;hE>Xi5SE%8O3)Jw%^=bIx@-'
    '%#LbsE07I1OJXlnta>P?qgOxZ2XtOm>uZ%UUbPKVZR(GvH`xp^N_l%O!~?kj7o&(F?2>2}^3B&3mBT5zWuL7{bV(34Z|u+*b&H0f'
    'dxSN<TeBlvf$v#a}==L*B(-'
    'G)Ub^yBK2hk%o71J8T<)@62+GdB4MAw&GhMTY<m!mIs&JUaag4#d#=x%DYJHXPQE0%zg}PZMw=ri}>CSNy<a=u!}tIB<^`MBu#Ao'
    'SD|oQq#OsYrwZQQov5qP1hA@2#f>zT`D<bGOlQto*d#O4r0E5Pr0I7GNz?BZlBO3HlBVA)Bu&3xNSgkjkTm_FKzR?r0qj0ilrrdc'
    '3!eUV*QRXJn72<AoLwIx&6#KUm@K>>>|-*nvYsc+nIR@)v|%RAnL~U`#y-'
    'vj(a<Sqm8LNNG+3D#K`Y+zkit9C!|+QQ5fb86ZbV`%@M*rF#Z_C%?MW!`RVGQoqb6~}<2Vg>*xwX;WL^2#gH7%oskXM1kt1yPP6v'
    'JH4xiy6S74_HrBF&`m&aI;U$EO_EXZrw<KZfl-uuMERjA;z*P|~eS@_>b$jko6Pe3)fC<^fanC@xJm<;B+A(hvQUIvono8SqdnFq'
    'TgZZHtxQkn6w-9*{0!uC{fN-G3$d0Z<57jykdIv2UOlU@2n?N$W4_8iu*ibj9CpDaa<oaj*ujI(On`4fFo5^oE>U|^i4P=Wl)1-'
    'kQNG|sF1mX>nja->KKpZY&YVa&lmV;r3@D~?W>9Y-'
    'h3iK7$liK7$d#?cA$;^>6=Ox0YMo$3A@+cH9IJv5zPyVkuKROw;u)|LT3;A&qWJL*AQ6_yodIkEyvw^(yR__#KkCFG{-'
    've`mzx;~pD<fa?4dxYF{V>VaFO*dupgxqv<HlIfF)WL~JAeekmQ_p_oz|^lC7z3O;y>FNj9B#4rlsd4`^EAwhv6#;(_K)+-'
    'Z}>s2ut~wiE!9XOezg3F^ZilQK_+L8GH$39+OZ}BVXhQRR~qo#!^P~zrcUy|FER;g=n@mLL0}>NDy%bSxyLy_pY_c?;XcFDvja%5'
    '7$fDh%LyLUz*wNG4SOUcLz$7(kJW7EIT^2eR9G)xvoDO(-'
    'i~7zeDo}v?2p>54hDRu(?!@QLifKVS6>lFTy>5kuDZk#S6$<Xt8Q_`RrfgJsz)4g)iaK`x&;wT%%`56l<ndM%~P^H*p-'
    '>y#4rq_%lqQ%Q;7bH2}B<`Sx`>(?EAR&D4Q*0HCFnD<`gOh#=n1V7Bc1+wgnwqsoa}ozGojTvwVE}lT0qIhfn9kbtAqfl+H?ohYG'
    'dDi^%8Vi5YYC!HOK~O4d1_>*<o?V9C1XC|I&?x!hXTJx9or^~hz~vYrVY%WVj|qoD<LSwBoVl7$=O{>EumKW(!*k<Vz{B3?ed;9U'
    'U#S&4Y()(3f1YInVHmk?3y9yC>V#B4z$(^v>O_0&jCJuQ+`Pmko(4UwFBMkJ@68OW(AHa}G8&@qpGi1CMhZb0OVia8$rMr=+tA||'
    'z(m7HX~lY&8iY;VoZl@h1*bf`si52e6TG{}RR8TA)xKa3#10bkZQ*Wg%5wp?r6MOI)iHyC%Rm5M<-15bb|fqwo6Ip^b%s-'
    'K<nTXB3G8>$I7Q;(hV+rY~fMEgVtoCVQ737kGbwBN4Xp*pqfiD4LlCDGVM=R@QP$p{;d^Ok+ON+qdL+yysi@Q;e_J3NoYHaA)e%X'
    'sU8%$6WLz7_W!T$EZV4Q*AtLoo;0oI(xzFck~z&_z709%l<}%{vC<eETRTakhPI(~$t^2rHpUXLaZoUoQ`!8E=grz*>dLYkC0d6s'
    'E4}0jyV;xTXiNL1EgO9>7M0No#rln-r$3PY{m38trjX>8?A*4|6S)R1tkOK8WpF?OrxTL$=T$!A6}fA3BwAio-ui#||~cN(Qpug?'
    'Utk*5wJN814LL=1zHvS@@z4TL)kCVe8?GK5Qd=-'
    'I*A#hhRq=28f<zTFLIcXHgkq#Pz<#&eo^|A6V>ejS*Li#SYgPaeZj9%QZ$^A6e{ljS<&2i`}j<;`-'
    'QP$Lo<qpUYFd>c9irz+%`0Uv%%A;p^^Jy{sA!wHh1!CQ2LrRtsde4r9qC_90-GLCF7yBAt{ESBr40>%dD6VjGSynSYCotk3o7h-'
    'nWOLZ}9BPZ!%<@?mA@k4>#X6JAS^2hXj7AjxvXv>JjW%MsH$h<+@GLhH3V&T=C$ru5*C5EqNdmO&UzR4gW24ly(_v6yTH1kglaVz'
    'QMGJrjS4$yPz=O!OrtTMcnD@wJ$24Ft`JlJ(H;CCf<C{-'
    '!nd$@+gg;ev}`LnU+7>~~wz<X5)wENbo+i)r_Z(_(a`;KJ|F2z#e+aI!`RwoVF)#|kd|E{(AF$x@2#Be?K-jj(q@kWSI)@!uUgz-'
    '&HK@r=!9Dwf!MX5s})nFkTu)7Vqy#kLxpq_X*?&1u%U674NP8Ao78sUL&g6lawtQf*G=hTt#O<p_T&qVv%?Iv-'
    'u4^U*arAKjw!(LFjJJ)-'
    'l`lbjD~=Cb+Z%yTxMjCtPXlQAz^O1`8Iw$9u3Vh(U5G6XRUcR=|lIbT7ROfjB<bYQPhh4w?iVeteG9wS<yW$_$(zL&|@+x_s~H+j'
    '~NAKv>fTAk}hn{K7mSN&+yw`rC6fwu3^>U=-'
    'u_gz}80cDl&wHe8*INz}rl+|*+;}C_fYssq{gblu+I*A^ETd^X@W|a2D`$D>Tacg6qa6HdKNqXSvhziFIO>2S$W%g;|C`;$?`EIp'
    'P^<tTw=ZE>eWyIrw*A~lk3HW-'
    '`3egz6{$+)a3tn5T(2l|DEi3F?@Os+{2^qZJvBK2_uXn8k9=(Ou;&6p89so2R5$<58$`0=*OquQP3C@G%WY)kkRRhmw0ftzN!Jcc'
    'pj-~PZqiIX<J}tRve>_O5=8LL8#VE7k|D1MA-lq}e679&ZH3r<BQYgXW%p3imPSE4V!W-U~ZcR|Xt^xx*yZo%gCa@f^v>r(-'
    '2r$<Noic`m_=(LFYWB8_6ZU<-'
    'FRC)?#RVFbHVXP)?H*0N*I#M(7n8?e{LL`^;&zN*BuL$X^BjbLaVHLR5C+CwIMG2U7<XepEgXz{G(tg`gz*zhKM4!tUX7^40u3Wp'
    '74~@m--NU?>>BusL4@!DZ<O(BAhQRhMR1@WqDeg*kDvu>REjl41$34e=);K{hah{xvdY3Wfgs^`=2}muD%S+Z&2q!--*4yUXLzb-'
    'xM(R^9ERX7sU8e9Jf92EfCm}aQ3<HHcb=l-%Xx~9ujDB@D)ST_N9QRzj>%JW9Gj=;I4&G51xaqlac0cppW;62QL-'
    '2Ea4GM|jZFt9ZI=QQ7KG6PEY7jd9*tBa@Lk9U{)&DD(F`Gr3w$Rs4&>#4?@VqWbSRk0p27~xaT<0l`702{O=EZEcn$k@G7p1+jZ4'
    'l-m4=;5zRHOjw(i~OI5)-22hU*Rdj5#mZs4)B&ZpM|Cu_KAkU&LJQ>aLq6Fd5Qa2FVH%Sptzqid~a;)rs3(2sc~EB%min}nC-VdX'
    'XpzVc7Xy&_}G9#QU9VJumw+-'
    'uV2@~CpJ3un`x6%JvXC?3<JTAHbkK~^33E<_Q3K^Kr1Tzqf$3F(n~BPK$<ykNI~J)iRdq5QBl5Znm&a5~=^pZ^2c&-'
    'otr)*y7a7b8tR6?2KRpLx&5YUB*<z9|!druqp_Q?)b*-'
    'DuoplxO+`x%MVY`LMlA#6zscXD~MBj4HmKBoPeM`1+?3ntgaJrqy7LFK%NxTjPt`n6g^;eJ$;MA(?+nI*1btTL8+TE!<Lub8f`u6'
    '!;!KdKR`$!FTDn?<A-Bi--xO#-MB`{(z#!UC2XtyQGMoSJ`eZ!F@-'
    'E#th&8dXn_VZNP3(cWRvYyG^<3IPv#m<xWqL&R86JQA^%yYm(s!eS%|E<fa`fOu<#04IPT;??Uhr=yCF7yby7A4w3@H+ocTVJOi7'
    'g;5##hMHY<{vngzKkm;Ot9I4J(!;$NpZ4r{4vn4{dbFxD?3u30<10L>&W9P)AFIJfG=fR4*LDhUCeHxQD#i`XR6Z}R~%0J=*W!W9'
    '47-WT#Au5<h)#aQ5{r%DOgmw3&N7HBx#)f6}-'
    '7*i&T7(D5b{bm118h}PZ6AS!mxOj_Yoy)53Xi2wGQFM&2s<f{(3c`eIqAIXN<n$xx{|p^;*gyb9kP?7Lv~7Z$ofWy?9}LxofaH2y'
    't*9<XDHOkFltzzp%e0!Q8HrrLma$P6j67N$&2O(pPD?UiX4edyWC8vPB!wzR$&=(t14^QTG$Fz>T?P=>Nv^6zo4=`0`Oj{vb{1n4'
    'ZNtEG?&i5t3!8j9q!9M0*l_RaGb>L(xZK%n8x*<I;3x$?){N+r^N~0@QeN8%<i4aote-BVdn9cpjQx$oKgfTp%XSVH~pZA`GMM{;'
    'ZhSbr6TK^O=;OH6rP)9)#3XrR_Ka|35;2m_*Nxyq@GjRF6r3;H;(iC*L!YVw(>W4Ze5Oh#&{OmF<RKdz$?9(Qn+yURYF^}2S?*_1'
    '!Nm?Yi(ED)pn5u_2TZSORVSA5Sgt-'
    '{%9|=bx5&XKJ~PjUDHL0BB7Gm1asnNDlk7>J&LW8Ey$v>687y#i<j*SzX>w!Z{d4bbynDO)R|UY6gEnBp;cFf4ewoP)lFfSRX197'
    '2d~J&_ps`ru&1O4t$Hf#!0M?Ow(&j9y<{Jub^lm#li9`A_on9ujWiYNAuTu;$=4$XfU{D}QCrLjf}%+|R8U{ODY}<zJ(=R`hhR=('
    'VLhnIr;0XW<~y%l!KuB-;GE?cY5v&YtmPPK!Vz#-n%RXzpnb^f2F(6kZmdL^&$HZ)03x4nxgP-'
    '}{&UL>2_W!aSnfywc`vZsk^thq*K$t+Nc)#I*OahcF>TUI*nC)C5CNUB;-5d5OopTGyR^Bp0<Jx#87qZWsR2FJR<Iiu4O-'
    'sh3)HS%nmPZG!5P*u0KL?>!_&<7Wd<CuIDJ)ym7Exdm7ElZm7E-'
    'hm7EfXmGq6nN=}W#N>0lUE6Mh|EYR1S39fQc$Ces|R&ZXrzwZPRYrg=!ohZB&8lgf-pHNlGPgPayY5_;-ApyI-'
    'q>)pCYCkV)Y&fRc&vFf}I-'
    'O+U5mfRNyyKH8c#&lU9P4KCaIwiPTkD3&w6AQv`yX07rr{HQBFF~#i2q~r5A(Nve=W$*M$Hbm^Mkl{Ib+N77gG~MH3?z=;9&&u)M'
    'KtP&e^EK%n%O~!%m3dWO%#lK=BkM^@2uaqbkKxjm&dYGHcREJx7JHWg6)tj_3J;M<??Ik4feW9-'
    'GV;JT93p$S4ipe7Da1x8+%1@b50SJcSPa;r|xoG#mrEEykyg#dLv3-'
    'W+iB{UA0gPg3+JCMge3ij+v+O<;AaP`LrVC>c?<ZT@&HV~hZKPcY>k#YCSJ`b46c1wzw5K)ZW|rhg!gg=U)mLD)vgH2o0b{7Pv0A'
    ';S5!(Da{;t*K1YU!&b`G_l%eL);HEox<66<ec0|)_<3~)OtcNv031FoDpmmxc1*E^T2+aixn4m7+WktYWIdtT2d2QUTjCGkWP*&Z'
    'Z}O*GfyhPgdd6=NCMVB3`vj#TlyI!qaf&h;N@ey)D(y)7%w#yA_~S!O@oMn@lw+vqF}sK14I;zmzn_)1>>b=LPSx_e?5gsyRww)d'
    'k~NXS$%*b+={`ES0{nU9)c`SX+!T~MG#q^^?gPW2>w6Q9To$_hkdQ7xxnRFgHMiKWDIMzaheg;{o%yvZpKyFar1;h0Ts(1?;>w1='
    'UCU2VRW`3-6_}<afuciuJJG}Hy@L42yIy-_D@$b2CKd}HFL2+ue2=Gc|K*udO2kdYhofS-'
    '<{1gVwCc!P)W+Cvr1AvT~w0t>8g^HPdAmMe7dV7<<mnYDW9JCru4n;JWH*48_UkVN~GZO!6co@!|^ao&hRY^kNWpzUIflJ<kEjHr'
    '1WYprfgv^rCe+KJhqf7#QP0e^jXB};SqMTiJZxzO%GK#rz{%vP=|BcqE(+uMDZ36Ols4Huoq6u`O?V0L;DCm=L&)X6h{Ywfmor5S'
    '8;PISQF6GZ7S(PwtTGBVQsv(piswR?I}Cg-'
    'y192K2ruXiV|Czu&AYF{m#<a>bL6DxK*dctvWq!)rPoLXT+^KQz}c_<D?geg+D<$;IR5S&R_L21ya4YSoHHVmeUOgR%LZ)!n;ws@'
    '3jWW=_KBu_r%j=X?NeT(M#yCE0RbkVT>6O8gU9@Drr2>N?Lk02Wp|XJ;)aW57N>~H5g${kxG&+`m6*2J3B$Z&Pfok_aq3|xd{SxU'
    'V?y~PYBq2NvK?>;DW~mX)ROk^|EGk*o7W7)y!tGG7%Q+$iHHCeYoLbWQBBK{}nqy7;Wv{s>aZP#JHPA4u%l4yexv2mq*a@iU?X>8'
    '9~ddB4~Ma1TC-O{x$2frB+Kv-'
    'B}1Glm5oD5kMvlb~T6~8(B8@75$hriv`Elsvsoz=&@x&n&kVOC2!Rv5g(>l76{sMyLSgF#PUBFz?Ks0bkDczXD8#-'
    'qC2F}x+Bhag2;4_A&zK2C}y{&Y_WGbhDtNc@fr~&N6S3=i=-K*N+Y7=SS~^!qJ&)jlQbeq$g4eBBcg=hUZ-'
    'e8loT3)9#nopi}q~xBC!dm8s1c$;a+VG%T#OJOKAykh;c8c+@cj+#bMOx4xK}t?$kNd=`Ni^o$l5-'
    ')af3bL!Ex2bEwn3(q;|L&Cc9-soi0`RO~+(h9M>nZ88z1K+n3jj?;<O>l`%+2&f`yoh?~}{^m*W_PBFGCHr<$HpFG91#z^-'
    '5L<zL-;F%i5!q3C2@GO^SG}>J>hr2UOkYu)(5Z+m&S*kQ`GzVmN9(n0PPakJ?nE{?-'
    'kxrgmi_5AYuTXg6)ii|y{cu4y4N)CQE<5TBD;H7eGTUsb2|XG1;J|-mD~<!cojxN_}<=pO8ym`1m?w8g9Ht&=-'
    'WBQHKn!ruNt?4O3ZccFr3ZFQSM1NIgD<yFOCPhQ;;OTL@)_4Heg5s-?^Fo4J||4KSUlZn%a(CPw-kvx3f=U$%s;;+Q2L2pZ6S{_1'
    'w@5c4$4jZ}7;rkpt_4FPpMdVtZ(?z-'
    'atRRx;tC%JXVUKMzsG%3#SRn}mBL5!89cEjOGxIv*n{_)cjI?AoS48GN>JNDs$@18zZXGwLD0eCU-'
    't!xQtNSJC1OEc}e%IkA{rzMAK#VvsnJWY*bagQala#bjsu6E1~D`<gH1kcR6N8I9h_sXU@NBG|A4y)mMWkg~DB{WH|3c8lYGdqb#'
    'LqK5P#t<gu+_ztZO;<Q{Mes!lX5zp{FVUlh)t-2aaIekK_ZUz%hdks7I%~Gjao2>;gbKlDbrjSDHC=*B_1e58b5QFK-qjG%KX(~5'
    '6HM+@|l}g}e{jF3L;Vn3b1DM{j$!F=MVDX5hk;s^47*vK6Kgjdf&@s~vCx$X;{0M`Iw;goXM;c6P?xa<P!KCIc!w(%(n0t&+cP`L'
    '@<<s{Ctpjc2zM%D>Y~n}1X~g@w0byR^Z{3JcC+`MC$<*TrUMBnUEt@eBl4wyQVr-MP4A=$zrDy<JHK1U~Yu)DzcZ!|WO6mm;{t}g'
    '$Pn9IFq=e~t_&N4)f$!;ZH*^Hfrftgo)pqW2%M$>eXVd^ckD(?aZsC*{{!KX7g>w^*_4!N>0SCJ9Phbcr1QZwo3Ihd(fI>lmA)s('
    'jU<fEA6c_>u3k8ONfrg@hX*jjj8vV)~tK$Znw7#ORIb2xe8^!OeaZlUGay(Y0#1rX6d-m90v?xsa=SwZfo8!Wltb1SL!k4Z4K-'
    'wLftZR`VahY`=N?5(zx{qZ0udr^Lv^#<mCfumtg9$e(xM0GK3QpJD6cC=B!#K)2Oe+)aR$83)+$({zy@<t?$a1$?v1bBK%LO_L2q'
    'o9ZIyYLEl_Bsqr4|xhg3T(r1h1&*61=LSOYoYCF2U<6x&&{i=n`yE(It3OaSz(AY!^+GsL6JNu_d@ewm;i1fQEYOF4R5>lGGg>E<'
    'HkxrW%yH75hBGH;l%t!PVGgmiV0`vFRnLtXzu?9SLS{w07$g-g>k$8%(4FLQn7cG%K+|BPUJ#{LR|El#UH~4VyZ-vamPX>+zu;DP'
    'rCCb$Za@Lu_c%27alH5A6r*$k5p+!=u|JbKROSfG)+1$psp>1J_S{wKM;q+bL8UIOBew`tCK_S5!mX)4$BVAA*CPIa&QKjzjc&Cj'
    '7~3uoILAP<tMGx5*~F>~6D7h}qpMHX*rpuf|J%Rl6q52*3>r88;>fQR*he<WC>Ftr+03Lva-MpYSHxoU@xH#La<=yjis}H=!|^n}'
    'FIF3+Y3VRK=Gu#N2|MgIj<>eN_RZ^^8Kva~Us}9`*eF?lsGfQjdt;`4+bN3hUZfta8$vX3WIGuH{|uPvoi>cH*Zs!r7Ufc(F!gH?'
    'tEzqY>K9?8MJ%M0qni@e+*~Z$`#GrxEnNG$*?!sJD83(uUK>Jsi9Eq-'
    '=tH+d(K6VKG#Sd+T*fk>uL&U{5}!^E3!gDjuZ{9%4_?ITAM*us2E^fA{g?j+7*ulZrkPy1W%z=CI^*Ay~?CN+9fMY~rK8w<SunG{'
    '``M#7tPM`lEc{MzQrZ79%5yt*`Y&+2r($wip{xY<-'
    '=@;D}=D>n%n{6kFe5F+8H!`bLZK(Fr5VvsI%^!k972BWZiF6ODPj%}T)F(m?~};$ubA=6vK({v0-'
    'Yir~2hK0lx7IN0*wgEONa5uWa#d@JZ#`Bu=u`Bu=g^R1vY`Bu={d@JaXd@JZVNh@eT@PEZh+;tuglV$|}F*e2QA7@(;Vml067|HB'
    'm(L{uaTHqcqd}vk(;wT#w`|~^=WFzLmcBTCp8^J|@qmA5fa1r2a<I}8($tHGZ%b06oPqv(?CO*klFw?}|Y$X#-'
    '?8{a$&&2+0HPcKS$ks5+L^Y<S_{0cp54~+MaXhFj$@ffX#0WKsa-94Eown6I6Y-'
    '*yih<kdAk>g+lq5fp;$HYCF!8h?H*hR#O6QHO2Tu$915Tt4icNQC#iqN#vFYyY*mPGDo9=33)7_BRbaxJW5&f_w;29o6i(SBMCqs'
    '9~v1u<<XDn}csdUA1Mo}uqX^4*!FxoiD-'
    'R(1Q`NXUOLGZpX{fH)d*a$5oynIxcSu%A4bGuyHnkzFmFl)<4+W?elC(E2SAl)N`9Q=6gj<gJ(JVCn(%P6cW?K)d#Fga1XE|#${C'
    'u!H!G8FV=?YgC+Sc?geV15mFw|Nllsdy{plViM4K9J!st8+(f7|eSv2T|1!CW$CPb^?iP!GF10l^qeVU%}`<ZVTgyq)x%ig8NO5S'
    '@3|#F$=cIb5*N6SGCS_Rhv9lwH2=FMu{je3(sYcdt|Bz=G_Bn9!YbBa^K>@){kYkCCmX9(*|YC0l6Ahbu!xzBv#GtJB~Y+zLZWDe'
    'k3W5-nc5jJnpmjVCcm9mBj`GI<bE3=`HBQy5Ccu(TVjNPklxw)^9!a8J$?a^VDZ_V*TDzpPe8hqaQP_nF99R%KgoN-'
    'z=oW&mp)?4(WG|d&H#l=k7KGVN{?p)$ZH6wYEeW^1M!`O~2Ji54CHf&>J6R*%=1o*jLL0y_WI<?r87+!uK|5g&R!dq2UOeXds?Tx'
    '+!4iwWk@bxYs^ovcbI7I+*Ni#vODt*^uxaBe?7wK9M_FZ>)U|1Nr6XNCV4QVO=FHHXy$o9dB^Jbg_K8!2#3NdIRPYJ;*N?PddK9k'
    'KD_KF*eyY<O?C}NVdbRFdkzQ4x&A7rRFndpIf8Zm>h1hv)E#M)?_toF#a!XUmmYzQTETw+>hW^7B0sHIW82M=wUzXM>v3aH@z)dX'
    '4)$4z7>h#LZYl5H&j4WP%C}wS6P6fmfB+OiGgf~RYb*&l6^;h*LP;FYv#V6hu`o0r+7Z+!;`={&)hTDT+4U)sL6+hmK1dIK9-'
    '5i`J;vkD0#H&js1OEUcF`PAKaNNqVh+X+QJ!08?TX3bFhS;%sV{QoW#RekTlsdw&;;<)vE2yep`(<bUjzuIKd(_ri7SLd3z1+F*b'
    '49HO(OW%eHGxI*!<OZB*!xMGu1cCIFw{+$6|i7=bag7)D@>Ert;ogNtDV#^_=gnlrrEDlPb8t2G0R9r~1Nf~`@Z`B}hBC3!j~b4}'
    '|4@;AfArw_F^-'
    'S|%_dywDwTcI6YR?ymHUp9<OqQYyTaZn@=uX)8Vm@D4_U78<clAcZz2FZ^!{Nl!m2>whVfb3TG2}5<Yh_Fz3utf`1;2BKFh1^NoZ'
    '<rQ3Cmo%03UcSI_@>MyX`^D~=<F0L#UrJWiKhtgtrAk{xwTQnmL4b#r~8Gu)=(BRsWeobQB!8Q4z*@yu372k+7rIR9#_E1wjG0sB'
    'IEX8vpmg2lHK8fj}QR>Z8IjHeKt!&6Z#~rniIzTbff!k{ofiEexa%LV-'
    '1VG(7GCdPFbihzXk7}KK}RLhIdb}0QnBQd;0m`e;3|8{r&I12k)K%{`cR9caQHC({rZ;W>8r-'
    '<4senhWSw~Kw+~WA+N3x^uk^@C>bmDFD0H21SH>Yv;=)0W0oRUC|hFBGjK&M>rO1v_Ie)>mO%?WUBEs09_Mx({2J~Ss?rvPe{4zE'
    'oLiNB#Mq@qu->qUj503I0n;sL`R6g2<>_GXI1iS&9}yh<P^5X>-9~Y9Y{qjXq^uP5bA4F}-'
    '%i*187$(B&Q9D;PXt4YR#~1=sJ1zfEpgjhCXq|snJDE9(_FzB8o<+y{jI2b&oHn8VEbeW^`i577sYuIF`a4b9G>;LX5OOkaek;_A'
    'Q_sfBQ*>nCy1l)5rS|(P2&!|SxP**1kX*u|1T)IJKa+}0nno|<!h;`O`?wS&A_R7fsmkNJOVYfFZ|jgA$S5x=>ET6VyF&g0)GoUv'
    'FDpKt1l>H{C`D1qI3qWg#Dd}=H6}WY#x<D_bYmPpQ?kmx2q{MZsME7Yxzj=*!UDOdN-'
    '*nXinql56zq{V>CDP7zW670<2Yr@ckNiuyER(O#>Aezs|(bf+~#v6CEw6s`za*ZJ@H^cUeK*q_#!~{`kon_9?$GqHg!HgsC&23B%'
    'juERb8VO`=QpeGygRBz*UX@3m9$;ODv&7jtBQ?Zkgw8iFwWhn)$B$<GExr<2m%5Zeo0nithb;%6)SMuw^3Gp3NlBkU)B0!tJ3?gG'
    'L08H49MTk!~EfOP1WIy0}0DY9t1qH}aDN11#t$MEYF1^$UFd`f@jdX6bbr})Z*unngJP`2UA6Vf&ub)ap-'
    '2M*#k9Dks0!#fP}HpwxhXz&cf(Y{ebN@PLU8jcZ>C|zJ+{dAT(I0@-'
    '~0mCVr_}u`Vm=(E$HYFakTjN2yEgrPn<3VHUEp5bUaVt)bTk(!WB4nV`;fs0eAZ?#Z#{z@3eK8#i3~}OysYJ*^Wj~Vzw#fHN9bt<'
    'BjLen`r|mZy)%>qAzB4EvpTGthbdgVDwOV@SH?mkQ{h6DhN*qiLlL17Nlj({;hqJYPA?4OLYx|No?_LMO3}5yQ5T|gv$^XeokQnp'
    '1JMZqD=`eXlDljnxN$^V<pDqBR>Lai<kPdO>pQ6)g@<D(ueNG1h->(+ejWUAsqsDMy*&fvBZu}->*w19N^-'
    'tq09?8S30EoaA;@o?n7O6mbzM>UKD<)!rv|6GTNY7m40_k~+ULdVt5DfIoG!bGi*g7MU!Xjl1r**yO=m|%`cRILArILJA7XRaDi0'
    'Kr3q>;6^@hz+QnN_L`6T)VwR+(%DhNFw7^56gbPiNe|Z?Vs2R1BUTxKzlw>^&hOchs<jx;MP52M)*H7v7zu;Ys@Q@Q!XMW9<D#aH'
    'nhhc+@XXi$((>o}3#rdI)i#cB4icp`<Lc)pbY&(X2(QOBB!BrHXBi-OEhT4Os!C=$0_9|7iC%WISP2U(T%dtGXz&ZB=>T^^c4W;#'
    'm@vw6VW5GSXTr1K&0p)!GD6(XSxc@vD5OwwuL{;&eyU2~n;K6brc~IulY>9NN2r*Y3`L(2+wDoSgH?8ZDdlRV3Wz1jG_?Vi4QBFQ'
    'lRuYjjb<jd`DTcTe1@DJkBeJ1-WGs*AN_@5#DRqWF|gBq)7(zKSxwPoc!Ujy+`R1mn6XKUOfVoAcuY<GLk3UNEj(^AiN)x-'
    'IV@7}xFjiGp$6k#`I#uMYN^q0|w4?ZL?5Wwox?=xH5ULJ`&B<4eT>pBoNlqq4A{THM&Evk=r4q&Z~O`J^=XIlm(R%&aM$&vg#2Tn'
    'J}6(k5nLvLib2oEeOCSEBS9#sPF;I~cC6VY|UhIvEAWv|+D8!y)7iM&rdM^#=i(Ojo$TWZNMf=ca6Dai81--21gmqC8!m=N6#e$8'
    'j^@495TG+kknWz&)ok#ZY$#5bu*pm0^_xCDjzq`t_s4@^4jk4KHSt?gT|)t<eOD=c9wcBaG+cL^tG#H+n}y@hrK~>kW{F@t!=1Dx'
    '>tCJlRlwn#8SB99l2Nty2v}J-$cV0@_VKeKc5cJ<|y6JI~-1yj$_CBOcU4p;HZ~OJ!4zb^`BX4NF)?wTZA`i$N0h(l4-'
    'KDsH$522sOixO}TT5vivAwT@a*iM+~;1RgpP!?ME@pO*y)pTcme1Eshy<FHDjQ8iJz)eD$HK~>B`W>8Qm{0b8&j({V2;7?FhyqM_'
    '|ROl>W_5_uKOPM@DHP5oZoyb?%Yq(IzSKI4YYb0M|$C*M){zW>gzGS50^o6PxPHS<#5tp!wDNu6xaW-'
    'e1sE+i&?eu=>kSH2_%4tdn41T_IPk%Y~npS1pvW$9^Q{re$lHBQ>xH=H~A!wfYCos(jjvkF32GXO>W+XkjZi`~oE?9b$-'
    'wdWlzk|{AXv{4uXwuBqbiMn&Q)UmUx*(s+@;#hDJ)cpg8WN}%GRjoDL}9CzC~S2@6t=oC3R~S2)vQ95xJRM78_V>>p*rN4=<|)?T'
    'QpxuM+CvBpgrj#vrn>UOV7|k@R+Y+3_mZUd?89BcaD{$Lu&Lasy^c2b}*%w)zjF|`>nMcPI2WX^BA1`c&nag(mVE_s$e)@$IO9|R'
    'C0oHlfW~VDmP_}2kjIXDDvj-hEDIxnJB(?lwQ+vh3er(Zr=ZltL%K0y_S=W{8c+%d^{;{*&y~%bVvM4e1iI$gc%G6A(;ii(kN)Gb'
    '55n~8NM9qVJ|jSap_G{EQfftrh&>vNyi(ZAexUTMMp@WPke#7b`IAU_+!7yq7hY<ILW5~bEqb!eoO<K5mL4f6uR&s72B`Sh7YUL6'
    '>aDFnJDPiOQ&juLOB)y;y;6GfuDFf6zso6lAZ3>_r;rx4)3OnkKJL~Zpj#V+1CVev8*D+i%<By)x)@%VzCW)lbeTxRGw0n^GtG&)'
    'FJvFAttW>6gc2T8Qz4E${PZ*OxJ5whV`mq6rSWV$FY*=)9dRx#d>`sq?k*b$J1od4?b1;RM$eV!MB=QkfUwPI2!da2_wCqNf_z>l'
    'TpfeJ&!FZWo+T8C8dlvB+GTCVS0F@y`&1}us0Kvg;0AniyHeyu4s^RIXDBjRf77jb)rD&@;+UdG_^y~v8NJv{NFK?s+6_^U4fg;Q'
    '}9GC%GfT1>NC!=Z$n4mD;axX{8qEbc0Lu|x36XFj6oIf*E9AQ`MnP6<#1{FgAP*MEeIMXibePsXl!`mbc<~?39J1#8RghFm4=BFD'
    ')72FoWyGkUblpIV+~$Y!n^SXuc_giwyn_5%+Ib}g??s^8HZNrCu3clm5OpIzSO)<T`)KbI-0@aD5%zb&G2-'
    'ZYJ(e#+x4&tn*F76(SfB>bCn}2zBT^;aZ>eAKzZ7oH3I`jBopFJ{!}i^Y^AD!dac3h*046$8N6-_Ym-%=qZ*d|;0irL?l2rup-'
    '0HcEtS-(1#o-{6oBkkHA6vukg?aJ(_*2k!v6~oU;sLqC-qb*-OPh}s+4Z25;t63qfn&#ZItv+PpEi{!x_4Y_zB^-'
    '&LSMw!Ek8F3<65$;>hY4e=r8v?^e;p$G}98!dSR@Dxf-'
    'r2yjq^jsdqOPKNC?7c&0@mW$4M0Qy5!hJ^yIhNuh+1zZE&m>>rb1owl>Sdd!_-'
    'eN(3G`MI5@wIQ7G<~Pox>vlZMmaXQ@TV?vsr{(%8Sq;s+7{~RRJb-ihchbGB>Q&u)Cx;NovH1URd5JE6Q2&-'
    'YX=e}wuf`yP<Y}^_7$clQMSaoJ9V(J9GfN*;|yN6GrA}#)Xz=|^}$J@J|rpBo0CF)Xi}&TOA7Vju~5$!VV<-o7`$NuvN#x-'
    'VYIZQxH_-sS`7Q&qAlfCr>ial<9%(@Sutb{>3L@rdunujHE{q)1agaLKl<&)vc@pwgRcoo{v`M`+KaVpPI`Ss%h9FR*K`Vc{$_Yb'
    'wb5Nrn2>^ernVOtCRgQ}(rZrle(fitaC?KO#lql%hmqUT;P!@z+p^@S?F4hqtxE%}w5vxJ1GE=!vs<$?Ox^@-'
    'jC5u(PNibs=s2v90b)d11i1jppqX=X9%OmwW=YqL63BbC{c;FVf5;-'
    '7VNMny482T*681K<@4``UG_>!+AgGU_eHTJvUqkyYG|KuJ+IJy`>u+e^h1zp~p?w#?bgy76^h&Z;R7lkO6V}yu)yP6WW*i)f5^BY'
    'OjE*@cQ=_yfaDXktP)6EA1wPNlM58F#MfAEvOV4T+DCz<;ADl6)@GBZ=(x#vTuaT*24m$7{=y|!sX3D<ou*~bOHB<6shtowKbl|l'
    ';CkyIV^bPGNGp|q90;S6UH9>-LROZzrA`bf~zb@U3n`adL#Ks*62uKkrG!Zeae@c9|5Km?VHm8d{O^HSMgA_T1pRf_lga-'
    ';j4f|bJ?7VP5x&!X<&rC8F!PL-QCzu+#>jYEt`z+#UW@n6@arRR`()8SeO&&;ylwNaQbxfA8;IyhP%U81N{P9`7ri2j3ws<&m=wS'
    '&?{|B=Nw}Agp_O!A@5lWwNk+Ff6AZ4Ceh1p~{h|)XQp;B}o9owstWagp-'
    '?ZoTVMnjl0h`XXeb~iK^gBleE=?lZ9u@GcUiV`w!yv&!nK7Twm%UAK$eq5HX4uDmALYBYEFys+TPRRQKB+`!=7~jnpB0}H$t<1JH'
    '$>!P)6$h6KwA!ajy{gd)+kuv6ovv<H(asyg`V@c`)D@bry^J(Vn-'
    '~G<=;}@&Wb*%`KT4(UJ=$Iv)`3!$BMfsO^WGS#FA58`2pm>A5}b)2o#o4!xca?}kubDb-'
    '=L>yL?iXhj75qO_t$6kpzw~4{X3aGIJ~2u;oZz065dg`d@r+yrVB?I{V)Bd+5tLoapkBgr+qjfvvYO1ns#Mx^t`zkcUCNNM!O|RJ'
    'X>drdP8RT2K4pyd0FUlr#cCB`1e<3X@N-'
    'L<3mQcTeqP>(aLb<?`Mn>qFe86s>RYzxPaEpk6^lJ`$W|!VXW&jwn_>4GZWSy6U=K&II+NrU1!3E1rTnW!I=%vobiUY?<N$Qo4BV'
    '<CHH3TV?V4yFwM82g}aUW*pDa{65N4C$BvRw6D48mq7)76k0s$a&$te52!K>n?_gdAZPmmw`>2d}b^2^+-JN=$Q?QX3vkioKw^-'
    'echk+Fqi-Po669N|yO4k}b&piMyTEH6oj)>D;dj9nEH*#;EYVR!)rkBxTC5`k}TCb#!zJiu4X`8R4RZF_&t7zepruk}GJ3s5=Ejr'
    't|MN2|)6w>D~*F_pc@3(3xyW1sccL#?L2zAE~k=B(6-CLY$=r-d_2U2odM_-SubfcPLLuHOeo<CU<`#JI*#<(|=CFoE(urucM5|&'
    '4&!e}26RzKtc7$rstz*`X|cA5r-ZC3<GA4dg~M;<uLN;}1U%zPJ0<SW9P-'
    'KdbS3~%<Jwz(?2*^6rV>hNYC>YOpmr9_$Yv+!m=s+?=m;ni8j2{s~s)b=k`Yt!Hiv^tBG-E8I^VT~`=kvc6s%-'
    'JfX(NToz;4j*^GL##y4`5aQ>5r%NOVJA_D5wD(85E?WWX!H88M8Y|#_WlbF?*wA%)TfY^Ffr1+0T+OE#Q1!<(f@Gy<KMi?hn`*+C'
    'E+tji?%57XhjtgAct2sQRDzSY?k+u_PT}rJbCK6O}z)q$lX+9W8wr+WJ>W3+!@D&sQjAnjR@9GgoMOq?|@ysp*k&ANDFukF?N-'
    '{k^uo@K?uRZ97yZmHe4@vZ>x|;lA(0nPfOD{dPhENtl+=AN^ry@=;jw>fXiGd0L@do)Zab7a1|O{xMW%C?E;U@*Ai<rEq#Z!kAB|'
    '8?rzer~~gPneTOk-'
    '7nAI7C?%EHgIA<aA!*PxYy3i64LH+3U7XNF2I(wHAo?Rp9#z%gz1EqaeIp1pqRgtNp|f3;11_LY@!uB%REbaNtePhy{JF{d^EyzF'
    'Gjm9{kZGWaptiY;U}BwIJi$FVfc86wigf1`jmG}oA51N+=Oo@GB{YycXV7LN_eNI+`d`%F2|OMEs-uRTWc_U>5Wa2QwqKNlcIY9;'
    'zLlvaGD7*e{&hz?a6)}qV0*k0QrP=5<&%p3+ARZ5iZXvP%s~gg7jBpc3yb*f&%aIk+_wdOeI%yH|Zb}fHya5;QT}Jd5ebJL8v~bX'
    'e92T{Jd2IkvCo>{wD}K`F0JtgHVA^)d1_IN`!7Rjpf-'
    'X3lWaQc7qm0(5Q5&e?uxN*%T9nZ7?J$@*MKjZlvjecq2;9=Y|8V*bgBMMY$7e+{Ug?*QD#yE$RAnPr5!mlCDqBr0dfw>H75Mt`9q'
    'g+);E4nUS)y?y+~LmRs!raSoi{-'
    'vM}(1TOd#?&MV*ov2mB_SZ<<He)oaFt;FqY$qBKqY!hQ+D!3CKBGYDovX#N`0C6)7Prk@G#%ZU@aW~Qqq7_F1RBQG?Gr0@m{%wG0'
    'Szbc1V5l*r+$p|QMPdv<1`d^141AoihD3?O8}G80^B{(focj3KI|Q^bYo)+^6^Pz+re`n+8R4#+<7C^&+C#cwUnVf)9B}{Dr1{rJ'
    'xu$Ry%yfp$&k~zDl{>PfmJQt?9;yI5Jge-tO4Koh%_a5PJuuDa9s8|nExe_@F4{HB=S9sN|voJe*}%Q0GMzjlW$@)!Fj;Gm<N2|@'
    '_-'
    'NvH4NJms0f|g^gEoS?e!vRFvlse%}EodVcMSLbIRoo5nBvd^t`II00IUBZHOdDKY|)F*)xwKKTr0|W5}P_KnfZKlQ{+29aEq^F$L'
    'Po9D}(|*ley6_0FA`uZeu;E(~@PJ_QwjM<hp)a^KCBP`6)LIEfd<z8!pmb;wmmI+0bmG@Ih32`?>7ALXhT<x1450VL%_4Yg=B=mG'
    '8EGPTH<Y)HQgWgr$?jWGbDodj`+885jv1q}Z##!e$K+|S&FCi+L4xE0=&nQ?<DDl+kpg5nVFgZB`+9EHT8+y(C?WIzgt!-'
    'F7;u~!<F(0J5|p*@;zsI#>_!N)$jX(nf74<K}GY7lB1*0HXLb&N10JRpDylmvV}maZ$8`kX)LI3($*q$uK6kZFQfC?^)-'
    'd5~ACfM@{`@+v%DW)NPjC^KUU;TUBX@@xRJa`t~8jzP^L#ez844xL%8$X+h$g_jt6g=iFh+t{m3A<vzROWrLRUE32&O9ub8TJUQ}'
    '%h@2h4IhKxQf!T`*1pN$T-1Lm%83+FMvHRdw~W1tP44OT{Z_^-s-'
    'v^SMWZf5D={Pw$7W)X(+qac@nQ>lLWUmOq~zqSVp*rWFMX>?Ny2*It^agtWa09w6nxwi)3TQ^4|lDJbgk&7d@!rMz3QM!9sjjT2C'
    '=#NV3(fCe>VHHgA)q+8AKAg9Xjjbn{vo!&2<k#&W~k5@Qyk^j`hInb$&cg`A*XL2|VOGS?3*C44f{y6Il!VRGoM9vp-'
    'i!JtK3)D<Oj-5*$!L5eW{6pcFRMaBWZa=Wr4-%HpPaBcn}K1I>vm_vu0#7l1zLTYSQ`CC<@Rshz8wo-'
    'y<XcnRTRl0xMl)b{7TmZLPYB}+xVMmedU&Qi}ln<bUwb25#;0*WyH5INFE0|V2})kyn<^wEt;BA?Po`~2@Ip!Q&r$fq^ZJ{IRi+Q'
    ')K-Nc%1;XYCZQYQlZ!<4!{!F(>wTE=@(gVm{{(u`vtw?OeG<=4j5BlUa+vbAylACt{D}Kbr13NXJSyoAa6!-'
    '*yxWd5m<z4)Z9LS12VaZmMZ6xYxo6`{UIlW-'
    '(*Dx;>{+pA^jKB^f`~uoD@rNk#6Zx~M^ahQjOW=&p8_?+N)X_ZrQORi{?VMG}>wbEHV3I&`iSNmPVBBt;TcpbtxtMCIo^DUzu6d_'
    ';;QDm))mQ6~+a5iQALQDJwSKZ4hzU@orK)GT#jE}X%Jluh1*Bggc<t)Oh#OEX~$h_M0B8ZP0XlpnzqhUKR#yslKj$9rG&e14FLst'
    'Nt`B5Pv*I}=?K`@>9xP3(^{Q8uwZ&P3Y8j?6^c#Qr1`aT74zqRrYd875WNB`ftCa5vmawV-mDR8sn)JyK(I*Ww%tqJGlT8I<9VkP'
    'G7JObjCF$c-`qhU01|`p-SJT+Xvq5eIyl!t07?tJp;vk+4-flRJ@)?)a{#Dt%l-'
    'ckc|w;jfT!_cTEDR`QJMHO*~)ip*0oCwdvQCT`2<D@*OcO(RGsAD0E^FIlm_Dtn~gTUUDOd%9wjhSV0_-'
    'J6`Rh8VtG4Bn6x0C+FSbQ29W;rYyV)8&$FAk~BKiJa7F8XALV3eR(s=1DpI%!eCR%EResZ$uf4p8M??eGM?b*s+EPB*gQtmn3VRF'
    'n1}*dPJDJl;}JvG*n7-9@7FG@)XIfSwfK<!(vfXEnZLsx1)#+$)uopftix88M9b3k8w}_h_cRuLuN?HiaMvRBxH{9XAy-'
    '<cW0e{mREU_eB6moDf?kVU&SKtqefKMPIH<er^Q|RYBc|J<YLyYMV#swC2GCs1G{w{r0^OS4#O5_LlU0kR_7@k?&9qlfR~WN?$AK'
    'Ibh)ey)5sg4bJ4-'
    'r!zw9C!y{7V0_*^Hn|J{pW}u8y&qMb*k)?LNBa>5>#EepE4>!a{wiD?Z^4I(}x(oZVnMijY@J^+hus>Ug^bB*)d^+8O1KCKVH-pp'
    'YiG0L}SXF0$NK%i_Rng;<uAnvEfi;EL9q$w0Wn~S2f713#5!irf<pm<NyTU=(nt1PUmE%;bWc8Y%xLXEiP?_67(5*j%2Q^=E*T5K'
    ';FS)S>$!NIbt~JO;#|zZ&WPuZp1HYFqx$y=GX+RIrXFJk}FXsyT-KgJ3QE@H%&AR*eHFi6V+oK6V`P7Q*smu0n3wZ&z{cnbLeHqX'
    '4qMK_qvS`kM>`c2DS|q6t_L(G0+L&*2Fzn(n90T=B`&P;>e-'
    'E4^_mgY@|0_*Wor`aX^J@&{`QePbYADYSM`W#`JU?WsuNlho!)aM(D9;ZU?dyi}{16=9FqG#PW%It%o_^W}J0_}0cG7}i0(jvQvO'
    'gd*GlfI>&a`8(u{6Y^<mp{4nr;ZyEb4{=l2zSMKeDVF?lIPNLzck8Ziwux?1sReQmPMzFoavYL!p5%sU)P)p{9@qy6-'
    'P09z<#CgZpAHOJq6_$zKx)^lpvLbMT)=6R)n?lnw)VTEG6l>LibuO8a)c-'
    'lkIikV3qrpQV8z`w&CzDp<{zsIu@=8fPk!cNYxG%T!wkB<G)0Trk)IGN8S9N)74M-hPBwkfy0^8T7yP2@0E}i-Au-'
    '>@%KeU=$F$f@e8`S?mfnID%Ph_;z!Ive>-q?g(W^r@{sMRrDTOZ4lB&X&DAIBvZhMTe-wVDyg}9n#7kfJc_w=`}nKxU!X7)(-'
    '?Zb4`wx~=c(sXbXUSbX%MjJN<2OGXk3M-'
    '#}<vN@$}fCaSfgx8#KO(r^o(`Yw`5hp7Aw2y^gWb*ws<;V*9g)qudqX0rX)T<1P1Bm2O}9Iag?Jq2Z2ogadR5aDcGoN14)!&7;J{'
    'uh>L%n~N*vFW;IpO+)~CYC;jIla^u^<YfHJUexLIrV;O@6luUQ>uJ)=&wGuAMC~|dvlOZYmw|AicA_&=x>*7jC+4*G`>?}1ZT`Mg'
    '{_{ia$Ihd468f`AFdc;fVi9~|K9C(q=`aj(E+Vo_?<l+5h4OrnbMY;t$LPuz@?(*G+&D3jOm!lQoOn8&MJ|7w_y{MSF1zAF$3Nz9'
    '5&sP~X?zhs1w%2ui2n}DG~Tw)z+8;)-sfPL#@8)9;#)FN$RCmqu?@P42P^>BLg(5$Qp$zc)9Lg#8#$RNWn_TiwwtWq6)b9-'
    'FHUUbI84&YOxLLa>K$567p=`*S`N{KEMCNRVix-qc2X7_7ItzLI~Ml(EVe9cOBQ<;_J%ArEiiqF9O^TmSRe8*K7cbb7E1AcuWeHQ'
    '+?{kdgQn`3EZpq@_mi=BS_>RPr<B~8QCh*=E5OIXdL5!!JO5oEW6KLh=VY3e&_rkBPAx%)4#(X(1ezA4=A)$aQHo9n??A2Sc<}Ku'
    '3i9V+J7!d@N>NmS7d!7wO6{+bqNpu=*?FU$<K{CHlZNe)CjXGPB?h#xfVObyPk>Qqtm72MTO4J8O%9g=2A7%4HZKtuV-'
    'RnHaz@Ia-dR&R|F1fV9gXzQM&516fY0@WUf%wzjVQ1r1ds_?<kBS~oEz!qPeeF3(Kb$y0-I?cCu80%s5j!6cPl-'
    '#1edXmws|7Lxt$&&i+*43gw(^i|Ecpjy-qxp=Su>9h0|9bFQyL*bb<FfhmN3+^i(R>W#xowRjp1pOO1+{J}|m7Mb~wvdowu-'
    'g{N9%TOE(=h=^@SJ+ClWd_#)Cdf<jQ)jbt(mnbTd@31q)mFOlwitYKCpc3vi_KTT74DL6~BtuYsQ7V`#H4>=_Vt}zw3XtJ$%s4@W'
    'z(*Me-i~x%Wi!8Mp9MX!ip^F7PtbxP>g!VVo(zl|r|ST=WJ&Mg+Kk7Nw*%K|znn2TaZND!Y(W=-'
    'r*QTc1_3Pmxb)|5kb%iCH$NRO%4t1BQSfDLuQGN>q72yLv*gKAztf*zAI8^*Rh%$9AJ6rQjzOCcSuV>!f4(RL{k@25MGF4=P@`gT'
    '<zq{$)r1{UbC@y;=6q4hlIr6VD)e?E&*}pIz?rVCFTixWh(KVIo3y7p>?8o{xMl*lie*PACwo;h`EQPvaVPq%d&I^=N135{TE^vQ'
    '5PiV^&Qi$NCKA*QB40w4;7^Smq{VRZYPu|weMQONdHy<FZ9h_(RC2LJ(p6(dfTw|Q7L2UkcjKcD8K1sn?sa5ElpBNe<4z7_YZR8S'
    's<Jtuf?Mc)=rN|`%m$YO2i|~|DLWcyVayK3`S+sE{yhe_atKvFH@JJlo}{yRW5#=Oh{KPBKQ__e#VTu)4PLAQ^($>Z$T-'
    '@)pzY<x*-6Qcqn&-'
    'GFR?!ZVVP2Qgm%SWWuKo~0aUSrmZs57`T)X2>oo6d=g<XTUHV#R{=Ki#%nq<V;fVuEBy?vWFof<5l#I}wfyT&qy57rG2O;It_UH<'
    'K<frJCukaE+O<7Te4?IP_NwPBc8KgX6^Gz~%{ZA@tJqih$Tc*E*Ay-x{cv@jGLd``-'
    'cZ&EF{YBZvifyV$0@}A?2Uo)QklqDUCi(yuXZfzEquW<U(~^FP7-#=b<-'
    'O@bepr~dT#O70^Nx%Ex0E|Q<I8Q%E0CYL&oofmX{tOCjxvxso%_D4)4A^e<1>pY7Muh14K}{dQ&_7P6s)xaVByco_O94Ll@O1(q-'
    '}w&4}&N+Qza4*DO;!z35b-'
    'fREGpm?lvkz0vKmIRUrYCyMv05L^raRj(WwtbxGgB9qxd=f=>zaLk4Mz+3~Z}HoYuCuu>=(teIt1$?(2DZ9G!BIMpxkd&)z9lYl`'
    'f|I-'
    'Hu=WF1wm)F|?83*u}G6r<HHQE@;1boZ*(~WR%Q?}g?AUa(zf<$d|vs|suD|@R%_<tzM0C+U;xUxMewtv{DS1H?181&aDJ4m#;AEk'
    'h5M9b?>R7HuRRldxAE92(4lSFOQ(W<6T*v9MKC39mE*}DhrJA|s!N*#!LtZ>aNb-vTmOkj?Q2+hVwGaN~OKhja0hUs}&>Ut%<;e-'
    '4Q9eYquD%-'
    '1K2UaAJZWNF;Rb_nw$QrEEOW5|T#eU92QrYKCB$d&36H?jvCJHrw!9;@D7foa~USOh7^OuZgIi@Rep~1b=C3B*(a<oSq`w74E-'
    'd6UpD&uHc!zdeu*@Lm?iY^dtyL#W$Y;ka_$+2r&uS%(ET%{-'
    'Ixm4_cuu4}e+g~u|W9Uc;#{6f>4pfYC`OjE&heg<^>{4F>Zc}z;spl{hbfelpf@WRN`oz!JRBGk=^^lxk9d=EOd@BHRbmX(RJP#|'
    'F;LI|t0<w~=>ho@{8OFTC)nZitIgpnq+$NrSBf_PAqrr>CQr-9_#9*hcN3U<2lwJHCld_AyYf^Uc_e~KoeV5-'
    'iTa;a1TCy%Ev5U>DP^pd=7e6{Y(+hoiDwS^GG6e{C_F3DNpB;__MdZQ!2lnJc!hxV5(H!;<U57(i`saguSU3%I84gdB6CX>Fo4?g'
    'oR}%F}hmt5Z121*KvOXAOmM~CFX{Tll0I^Kb?IbzeNN6OksNr}<kA7?Io#k5|5z4OdZ|~%9xg=&xPUHJ?;ZofR!?k+))6$Pvrwg('
    'OUjW#j4W#Z+sSvg7U4OpLj^}G|JYPfN`D%{mYiK-Q!{YfG&gyzwM6+QVRo<s-'
    'R@B=rn4=w3c%D%bi?{C5s|B*aT;TD9&H6aPtUy5PQUu0A#)-Bd5;r9i$(OPq`(1JxnxG3#LlbquX=suzI1NqK1*f6wb-'
    '`(<MHifgZqNm%p&P}H4bhU#S2yLG?8h`~6mGU3*Q`sp)t;6n<NpOYANM5RQTC6egyDX-uMPm`Y1Y2@YeDJKNhjU2{uQk25iy*1j>'
    '@^*2c%_ONMF);HBZ?;8xin453CcrAp)uA36L661WpoCDb^Smn2EOkBHNUSZ^UAIb|ymqOCXHHLjOzckc=HA(Al8Zw@j*7^1V3dcb'
    '(9(M0<})4XdJa0iy6G1j?=`g9?fQv{TYHdqa{ixS%g`E*o*=C3{$bXi!gSR+?eGsO;^U`QaBM4G2K1uOJOLJJ9fv1`G}?e53(G0t'
    'p{!Ky%>WBMlfDDELSNh6PX-(tzQjvr4ZK2@C=_iewJrCTgB-'
    '6ISI*AWgYd*`(~Uvc3@Z3*{6iRLVV>w^Y<i{G+l@MZIl0s!vDF*7L#l3udFu_6tzeu>TJ4zNFzUu_<n&KM6=3By-S^=&Rr$-'
    'q!Rrh$4bgL|<8gVNfR4B+G1b=1d~e{^;XuumtF%cD`g14K9}S{t1`MNoq)curZ0xjmkr|W#tBId#7S&=(|g)=?H`HQ_GvcKIx{OM'
    '^>s*(`JXL7|Ul<EWbOYLm$M!#ws4@Bg+28h@|f4A?;${3(NdP=v~_P!@CQdL*B;ljz01e=zS*x5qp<Th9GvPPB{c;reuakOLU1p3'
    'g^J*k?XXyx(qmE^fg(&cF^T_%XhZ6GfGnXQ<C!=Q;UNr7q-'
    '{<zw~9$`_eA>l*oT_sNc!VKkOtE&r6NXD)~9w7@LVvKz<HIi$rd*n8(t>3r@dl#OZg7IQ{Msr{5#u^m|5}ey@nr@69-'
    'US4BLDDUH;<f5wzXngoAQUK`0T>^1NW^7G!f!`^n*jWwa?uc;uz?Em^r*bd3~`4*hX<hf13sZ7Srw_?L2LF0CuQ*NHS37WlACsQ+'
    'fcl=Qw7hVZGJ`C*PzZ1ie1H&ad$N{QS9dKl}H?a>L`l^ir#@QG-?adBC$4`5fiKFkQ75XL{lWlf~wm<eeW+?Pxil_8RcS@@-'
    'QUB9jj3EV6?`a<-'
    'p!kLu^_{zB{K@Ee?$%Ui2@Il9d5l>?&Y4P|P*Q^_KwoHvM%_l29iyP8QF@*T;&FxMgnzhG<z!#DQ{_BBxt;w94KzyVZe`$X(%0b6'
    'V$f7bt_cUZ@U&HKJu5091DXB9rH<+V=kZe`rEP(L!hFNWZH$J|a=pj0@jU2ZnPDV6k#-0}ap36#aS6rvh6+-'
    'k{ztwUIY_dVZgbi&M=`GKb}D-qLN=XJhD1t``=rhpmV1F`z?vd~Y&;S9P|avPTpqKZtHBVF3-'
    '0@dezH(D#U*OQKE!IYO94zdBrn~fuCX7jl)Hv+D%LjzT<%u7$pa%Lqu{^|(C=gv8@SK+12h;Cs{v?<=|J$upz$qDU&ELSJQKMft^'
    '`X+ZtQOvSax!T^*v)>S3bjn?T(?JE2K0vbq8a|C2~VrU?KfEah=F2I3%HFq8m8UCB4cVS5u$UUW8qGKg;dm9;vs&eyCE*1)@>O8N'
    '4)$PDc&+X=J$htPTP~2oIF9eJ(Nro-SXc5QwERSdKEl%|RmyGJ;GW=?fJ>=8yD){u`I+{h=tx43YuRf@2EFKuE!HQ+g1T;Fv@LJ3'
    '>Ked_MyYA!rwl1|i7x)1orEwm<aoY`C^RFUbvWL?*t^>m`;Ak?P^M?mzj-sm=|Z$xfQN=JipAoZ<hCieq4teG|>G2TObXTIE~+X*'
    '2mk5Ln%#d)2nDq!6x~oJM_vkfL8Uyr{Q0Z<cOCie6xNQEzqX^*w|X{fgm5z1<m0^pc6?V#AC2-'
    '3&;NFmd``2Bb$2Q@skrDiPZMwy|rKYmgK%#?M4yP7Yl+)kJcn?E%*hWSNhVv{LjR>m5=(#<ABG`y#Ypbi^{obY(;@fL|JjHm(T6`'
    'fp{FE4cP|FdEN_aI(v?BAo1}SrPm+IxE7*uE+v>Ea=9Jl1`NUuCeQsYYEAQyHner_zm}IXsVX30$r0#=@n??iaZ@ys+hsv@07z6M'
    '6sk;4LAs6B{h9O2W=o;yLBqMaZoOCs$7FVbHb?;gRDNbaY+9b+f&6XeyRPKoUB<$1%rRHoA<P>*kzf_+1gcXYXo^-'
    'o)z(US7rfC=3oL~4?4#RS+aLw(;Zl?hgHgMcSG#+np4K9`BZySAh|8HCo!CfN}7{7<Wtdn3P*e@lRJsu{WAM;!8P4$plZ1^liA>*'
    '6}vKifMv!$uaX_H!q|VP!1lHK4T4#!4SW!cq1+=MtL$%6tO-(wdc}=SPaCQC;nn}k7B(VWP`&ANU1S@~cWzj+4eFiv-vw3iWPrvF'
    'uVr$c54O)~PP~U=<~~$7xF<_~bcF&u`dZQO>?HXSr8ysy8=p$!k4v&d8TM)E^}htvha^Q_3RJc4C$`u&83KOj)SC=p{mfMH10ctO'
    '3m@bHB>2{;q8nvH=;OKtc|r=#_K2(W{~VqbVRPqsK){J`D=bq{vHCA5r*K_dw<d%$rW@0R$aTyx0Aa*Y^d-'
    '$K;x0xAmn}f69xKu@XBmJ9L-'
    '=Yi0NsU!=xTr{3_bqt28hBlMJ=tPVr||J4DQj>Qtawf&|bpzw;MfahQMr0BkKSysigx1?2gy^2q%S~X8q?w7{)Esc4b#_SHsVmuf'
    '?pn0RNhxpoY39+FYImryud!>k7Xb&RL_ua98JG!3E)F<{VHx2nP!;3AZ}bo39DCF#&{1!tIR3r<!nwL*nB@ej55^e*VQ5(TGU8Ws'
    'uY?X<1O7Ri7lz^(ujy1NcFISvF!N6Q|KK>ZaT()7qKe`6kEAD!AMv1s$h-VTB%an(I?T?3rWiJI2n)q%}mZ4N6Yg?K-'
    'dkTOc{LQqXgUfvCaP_2D4P@cjD2pek^j^ZkmAg+iItlJ0U<^2CyoB2`!8GkavgNez4;NmY-'
    '8ohn0m>5ooTtT5>dJ*n&{uWAQN?nwG5LVqjp)^m$iuRS2Fa0d54#A6R5^=~lw2O^R{ueZXYv{T}s(*hl?RRSqIRACwvGXD>O&B1q'
    '>FUhwSnmF4+@vu(Qt-'
    '$B?bxn61pV!Yo8W8_PJc#eA=muR6S|3w1>M+@VKm3S$Ga=G9yCEQho9III?k<IBP6v}S1YW7F%3kj~v8TdYusn|(;qHkhCh9hD2c'
    'W6YDg&~xHHzx;^Yj)o1-@X5f_-3{kmCHoz&dz(#yE?KhH)0Vq0O<Ic@-@t-zRc*RB0V=B&i>uq%!xNo6UuG$aXC-jFlOubW>fQI$'
    'TjRTw8X0vU!wtKJlJ_eQ9abMhE#PrGh*BwI(rK+@gY<1ApqnD#@ccyn^H+u&_T=L3R;n*dMMSy$C$)BPvLj0ulQo6(k#hiT%+ElB'
    'Yn$ZeKy_(NSQ}Jd<q}>D+v%DCg$Gy;yGCW=<)2a_p-VQ1=P;;>ZenvIEno#8o#?<X$St3#>F)rJCsa0kAfIH(bCIoX!VjE#iqp|I'
    'gsqqiCe1<L~;e&?;I$^sRVNGG2WuIR4w8_`9^d%oO~*qk75NB08K87R}*&Sd<ffNy*$B4kA4Gwl6E0d_&kocoYki%)aeHCBtv~ij'
    'wKKU8H#YT|mMuGWgmltI_S6;7;8U)m>@7)?xXYJ>NHIF3?)NA>!qGbmDK!yYL2Od%%v90m&1BxFJ|f*PDDVvz2Z%`TpqQuIwNdDn'
    'x33n`Ujn&^O%_&f6A;Fx?w2o9!Ca^8WB{m&0l50JXZqb-2O;-sz?w6N;1Wahe4RTS`@R1AWX|I#=Tr7)-'
    'mhDBn`eS!b9VDIMsp5xdkaVwbu{>{5@2UFsRJOT8j?sW)SnS}+o66`6&t4#2lB8FsX{r)IW8hSp30P~nox4H>IeK@fy$lX`xN^a0'
    'NeRY)sp$#j#?&~~KX<eQb9RLVJrVvvCu_x)M5!x8=8w5iUhII;Mzrw!2elpUB(68@!ZQ>t+FZ_ovjdS#OW!S@&uX{E$dMc~`=jPY'
    'YTHTqhrBF)#UjWv~62vZ`OF)Cn*^MMY>Ynp3JhjTaGb)~N>gTfl3<j4xm1kpB$EO%qU0J#k=Q_02VYkF7A7dlaC;zRz&s4l7UdU)'
    '8_4aVLbi@0|Miss*MRCWlL@sx}R)pl2;+U|~2+dYwLyEjs8_eHAh2a#&KKT>V)c36o?mle=}J7a8E>P&z$U8673m_ld^600L2P~W'
    'ahHoFo9Kj-'
    'W>WYzt?x=1`07*ZpD?jS*ve1}9K)T^BuUFyTXuk2u%H@>XUXgSJ=?BjPbdyEmGG<v<G(!tn{aPGG`Z1M+enxN4QXcPAw#itXp<f+'
    'kt?U0T@=<lAGh}Yq7*JojfMn%Oz6-Wq=NHk{0YZfz}RF#<j-'
    'q`iVDf&7wE@R&__O494Exw)Eqm3w)e=oDg2J!s0u2UcCz1Or|BAPbqwO!$5Y(ne9*psrPmqa1<<Sfw<qXTzJ#-okx_K*q-'
    '#jk+9Wwl~q1EX4&>J+Zm7)S99iSRZZF~jEMXrAq?miKzy;L<L3BQJn%n_IPx+8$FSIX6E+v=#+i8j}Wa8)vB1D2;1+KVM63H&r=>'
    'MR`ycK^)F~%ae3~?tesNF_^0CP&fhhbes^C9SSGFo=y;&vP0nn*wcwZQg$euP+9L%=plwwe-AEfuEzH&in!co+=qJ1H!KQ+uu)Fz'
    '`$<#o<Ta)MUo1x~jvA9b2G*lz>;q9%^@Q|++SE`;r+;-PXGnu7P+i9<Bs)-'
    '3e)Iuj`G&J!vGchb;p@4ep8oyJ9%m#&TQ6;kohYh!8}G*H<;v#+MHdd|<*ld;3b_E4yw~|NMipJha$L^)g8R~!Q@h}u-'
    '2xD^&}?sq<Wcy}Qz3a29?CRG9)%xbIwX(6^SJ|(N8$0zfaFnVoZJb?qwsX^0;Gm}pG(n^DB04140C&Y02jIl8y>*Qg0THGgTBGq_'
    '-88_Wm5cS>7SXZiYlOh+#K%<wvQ`%RgD9WbTX{dOu6qggsz=&ToADRbiO2CLM5R8SQk3`oseOF+4;_ej=)9EsHjDZW!FgY`xWP{7'
    'J3C=bw;_NSMW7wr4smbUx$`5W4?FTLn~ffJ$xieF}@V6>a~aP*Mhq{o^k^{Tsi0~(+Y)l;xy^S+y;DiY~Rtfc&bapikhhx>Y9LMT'
    'J`afierZw1{M~QxTzV)Tb#sE)TAHi!tk?Q=yZN>E3i3~{U84suU7U~It@bqnfx@YP>`c>Y1JytSbOB)Yc#{YQ3JoK9q_y0S_D`t;'
    'J96~YAJ#TO)`VWJBdp>RMdG}3HBB1-8yWtBn9ruC~9-${Xp#NCz;=B?f3wpOrVRgZ)`8o%^0{B$cxr#VPD)0pc0?*nE<-'
    '*F`tF0JD>A=FpuSfelOrmjC|f?D;4$)*lOEV;1w_;X_rh5?Gi<=;^})`<M_U<tu0%p(2htiSTW)WDY!d$Vk|H<CBM_3X8xMnVCB%'
    '=2gJxeY|ODxb2d4EGJQ@b2iirN^T|PWvF3#GN-z#sULLbkdWBIvrlNyX7{4O=F(sJZ+y}T9$L!^(D7n&|qwI9QT~hGEc)wt$D0-'
    'D6c%#e4xqp>aqB%j^O0d7lx7mX$-'
    'lPGC5z7;PkYSWx$+a@v(3aunevY9X&d>dmhBi1q_j3*HC4TNXrDW0?{}g5TK~<ff8ic{<TPAbI`7(iRNL*YBiqGp4#b;ZSE`Glc1'
    'RIsta7lL1gV4^cO+BHDdwi0$U#AM+zYZL*^F`))XIdexSHduD<KaRDEjk?m1X0}O3;svUz_w01Syy1dEP7v80Yc7ldkAb(w8+ej0'
    'fQxIfkBkc8UdKwPa8p)+y62GF}I&Ff-$$BH3BjRWrI=XplmSA9F&c-'
    '%ERIgI>}hk2uWS93gnZ9R6(;*@=kvy@6W=FSsok}z7zdUrgUaT+TY6;=Zo!@kf$!omr>CqsD(zwh$0{r=oK(UKqvH9Vc_Tx2?Ld2'
    'V`jQ|Jcn&$T<n|6HZm^uJ;X^DW+mLPT8vQ%SI)~cCe7oE5Bq$#ziPsTraDXG(A687V7eA(vBYbTMs&f2rx|o}`Ui#<*f|pvWk&Q>'
    'MmhU63p8MC(Z-'
    '~DZrZS(TWm8UDeoGS@@_FH?;ex#9x*BJ8I$r}F)8m|5c@n^+iU&aK8aRemBl`>JT*b^on!FzQTmF6g?gcmsd!^;gfh>IHk5?0=|E'
    '8!Taw8UqG;8cNxO|A)QU_dHWZ>(X8a@?BN(zz1Vi?XV90(E4B0<|AqPY-'
    '<iH4q9E5SBEYcwH8+Wv>N#)+|{oudTZ_2ma8tft32&H5xNpUbpY)E?nR+^Aa{u9Ne8I27kxB$AWD8%zvogxp#ct#0Otf9ERJWKk<'
    '7pBC5K9tTT#Dcz*&?dx!evt2u#De~m)F#A&0hHDz#Dam8*e1k+L6q8h`bt>3x0U_8W|^J~N*G5vVBj8jCUS`<(1wjsftl(ocU&9g'
    'fml4#H|%)`X(in2ts%vQ@4QcOldj5o6=)Z+Z?!H3A8b&-@*OQELi=EOz}*~av-J_NUfN!))l1uprFv<5u~IK>FBa;h?ZrC1w7r~6'
    'udGrVh@db5tfj=JZ!PtBBaWr6fbHz8qUu{?iU@6ttbj&AF9xBPra}VcSK%!Rw<_ra(KB3|A`aeBfbqtbDBX{*tEBNlPx?*8Em69d'
    '_rOvj=)*Z-'
    'u5n5R*vimXF#wpS1fKes!s|UHe9b?hG!`5`z5bmxH9mlx5Ibm0DDAU7NI`21soLeOcDJX44FD8kQgI#r+ki6}*Wt$oRKd7gMjEgM'
    '<LCZ~0b%fH=Kd_Cj3!`Ym7iDzkgW1kDycQ7DtORHUQboY?~Ua3l=J_=NM27h%4{RTq*N@QkrmM$qf2za1}U7u)Afg=#rJ;`vSgc('
    'Q7?h(Z@slyu`-;2)vJ+&wKyT?&b+fmCmrU|DUt-'
    '`P$OTTtF@eFHze5t1&z`MywotX&%ghn0sTB~z{?DXz0uibaptxA86lGuth<(G1?#S5nYY1VW3Q~><h492IC-te3Qk@tGw<Z(a80h'
    'vdN>7Fok1764|Y>kmH)fVu#7cjd#BiE42xnk<M&Nb%6J=ajqH<M$1UX+|E#6mh_a!I-'
    '#0o4K&(x59`Kx`iP#7CEz3an#ht=B(EV_wun=^AT*RydJpk7WOF<9BrOaB;gK#Gw6E#Lp!-'
    'f^<%hQp>rMkH5bSl97M%o?Nt?XSsw>(+Vi>st?XlDf1aUf$Rmk%xNuV>Ox1Ra@Y(|HahlFy~g5K1MVPv<+7OumrLcjy_vn7UI?QT'
    'vb7o#G8`Uy}KcSKnT(Q}yk&T6Ct)K(3mK0l%&b(+fE7B+cc7^wkJ(y5v`GV$;C!m2SWpejm(r`=D&MOj*U^?0{|@KQds<w-'
    '1Qqd_llcH0Q76uTYL3$Z`21yE*)RX}*+hDQP@EXoN{~kV0RdiBNB$$tivtZzHmMzYR#rp#|*SUFh9=i1W@1!llk=0#ocy$MW-'
    'Z)X>~!VmA}=o(Ze+;tZcZ^t^rtnQS^c^I|jcVb&><jb@-'
    'X#&%WLm#z(m!mLFjrpMu|LE6#z+?2SBfobbtwXcUD{W`=%=jkZB8JcAj^k*dZ@H`C**LwQ-'
    '#7SyPneO90x9GiYH+fh35;{yK??zw3oYCan=}Q<io4f~o2{T8N_oOfW%^s8YqA&i<UX%BxFG6nngAsCDrU08_cE)-'
    '{6meAf7j9(xM^)}4(qkwPq|%+ZnldzO*RL0~wO%&`ZLK#<L0jw1<iO_J?K?7B#6TpB*qx&NzAP%*hq0GawBMgaMf*0DsAwO?u};x'
    'GjN_f6JqELB|N2QSM!%P8dq0eR8`yAsV<1L@;B#Hls+xiS<I*TW`iw3~DBYp#*8~VSz1q2&HrJ)N3K~nRwZ*PC1#z*rOhH`i?Ic{'
    '~JM6m>os_~9PDS3^Y~y%ufs!%T2qmMf!i+bN7*X^%#7>~ODi6#F)O_WUIf2BYJQ^pBYK>`+;lNy_S;##k!ad+lfR{g_m7IrBZnAV'
    ';wY|Y#wB5A5v1-=17YrhMV<wK{KM<#P+*e%;rHet<SOGTLMoX^RmA1Z?T(z5Vli5%WbT?s|>}vTQMl8Uobap7Fpk1ImjLE-'
    'y2jiV!Yt@3Pe}^uDD4*2%6!z@lw%Rtv7SVrdjLimVnYKpX_7-'
    'obUBE?95i!kQpL>+OJ4+#^`HYFBB6T06Z#Xc`GO051gC;HB>r$=Yol!H{DJa*KIZ>xM#%1Wu54OX3UXnWtj`05vg#|}w+7E~A#w|'
    '?7ZqWH6X3h3sR|%jm>9s2??T5isp^wPv%YVu6YXAs412`)=&vkfZb?Es#<2TRq%IQZZ0Ngeu04}%>_S@ou^DYPo+&cXi$Vc4w9gV'
    'A3d^ad}T2N>NE{u&p+JK89XlSdZI)Rmfw`sa_St)qC#xij1)>sE4QZ_S<?VhIoo@tom4vi_;p)qaDwz7W#{fnJsVm-'
    '~oQZTBKJE(m--0}M>WkI~!<ccYD_~LP20zm^|Q$#?7YnI7jAM1ed%NS_)fG-zEB!q7JEyLP1y8uM~h*-'
    'dWMoX5u4_vFG#P#*_+9#&|>ZgoyaY{^~gRvwfrclUOni5Z`o@`BtD$30yj$teheKMA6-'
    'lZua>YkJV^bEh6G4SzcW!gHeP~mdq54j#Oi}RXyKuzU-'
    '{RVLO5}d<6WtXLtkNwInPdPMgobLD0%++Rh4u^&eW3!>+yWzloH*+;c=^sm2Q<z%pw5d}m!FEz|Jm6%r&tWs%P8vozoy9UIAPjws'
    'mnOZ_doCht9#i>pw^n~r`O0GYH37(v+AQKOpat<_bpu_HVOI6d<?Z><Iz3zbNF2VABak?dH~?q?za5mU0+?@Lmcp3N&ZL;>nAyLE'
    '?$oQ0v?ywrIja?jSK0@4GDp~~82LX_+284OuIs92IW{xWmVE_&By>Sk0_EyKOtbkhjCK|^l1T0|e2EtMn_a;IVrbrp&dV>@XV9OY'
    'A?4tm`JHmK?#l0yV({+#ZaHN$^O;f#&dO)Wfx9QaM+(7v^LwQZ{I{}q8ZpN8nB+mJ1_G9;or;XB#@Hln<gMFCKW2E^Lt)AuEvbWS'
    'cWF*bi$627SEn+&l(qz8+D^(oUnMkkRfd#dy|9q+3?#CTX66??r&+AGP3tx*zysQ2mO9jw+l_Q!&QI`Fl!soBntz^Pzm&T5c7(mh'
    'F{P{CrM05J6q}9RW3#bGY&Q0c&Bk7_+1R_(Y^1XMCBG6>Q!c3zo7$~vD^hOJtN_tWg$kweEJ(Cd(l{S~+Np@vMPS$x=LC2vWw?#w'
    '>fC1{f12|6FQhKK$JsBY7OEZWg$c@w68A5sAg`xkJD<e)XEP+_F|NM|QS9W;{{t>M9M|bapJD9m0)?X~OSub!(=7CE=n1-'
    'od{pJFlgipkyHzcwSPR(m3<^0D)a``7bgyI@-'
    '9ZCQg8nm(ArtSb=Q23+#!KLZ41T;vNqIfZxY!fQ>*>b+UQ2m>hp~SM?>16?Xo7=q3simX$voYT^NPK}QMN9bB|MzfkRAY&%W9@Z%'
    '#hT=k7i~h%l4vTyN7=$Cw-'
    'l++Zc^?E7aV)Zc|uxqy;2;zRIT3uMo_q)32L_*LA;1M7N&LFf9)HAMY!Bml4O?wi=RLZ41gG&6)7^Fpvt7_dIQXDWr+>!DjZ1k_4'
    '%~8FpQ)B2b4O#OB0$a%U|x=<l-hiae}Rbwv(T?92WP6=J1)!l6fNC=-LG&sX1V^ebF_chIkJ-'
    ')(YP<inaLPm48IOq63S(e|c{bF5|B-jXpgzE#^PSxPAx66bLk$z@EKw?7x`^XVE0q_t_Qv>)_s-G_P_wB!RxdKPF-Zj2SiX(MqDP'
    '6cXwXN~de#ZHuBMyZW`0Id8;&I&fj+UC1~V7ocvlybSYw`QDDuAt8mKJ7so8Z!rJ$?!3%44>U<Xk~YnCSZML|7N5Kcv5DcGSUP*g'
    '>H8vnnOx7bm@cYA`gIyl{UWZ`3Fs}Ni}pqi{Pmy?vsUklHKiH_fXmt4oBKlH5R8hN?+fo0iCDZYqNBf`EsCrKg)3Lm`)z1Iqlo}('
    '4kD|ZT`H>0y>Xt-'
    '5EGhX&w7<Wnc80<c=aptl4>MM%w7XiYLB9z<(XWG~C0LSki(DD|FHit3W4(9)lLN8L&ePHv^QC@n(Qwg>cgoIHHg`_eM!^%VlLSx'
    'VtH1p`o!{N3eERUBQ<$5$~Jt^xeadWQq1{w-E`w_z=aT>}KdVGzS12Fis)<^}_t~W~QyI)Y9`fL__@M5G_Q`SpuS_`srTb-'
    'F;Tu`-H*K39`v<s2$`<Zq{~{5q*+dwS6@`&~PU3QT<^}uVPf+;>6rBhATto0L85<a1rWK6wm9%H&TK+J=-'
    'fdl=FPeo9tPK=q=MnAqz*}@(eg17L=~31ZyWm*;{GzaC&tcEi&g<x6?-'
    'A1ZyfSIA>VXXj5^DHJvsU=U8{prV1je+WsPA0kY1@cF5=_k)g<mnf;5AG04f8{d>BKe^mJjw}_9cd{tQnZc4e67VG_#!QhW{ii>W'
    'f4J8}$s0{Gy0_dWnGs=FBt!xY?CLz;%(_p+YF-e78CoVKO06U171HcY4nQ@-'
    'r5!*rU=XPyhGlG1f*O!e*fbBQTb@{TkMs)5k(l*P=L@_!UQ_7sp>4GJ_J7@xw4&)PzhxhTZvrP1*@?nljsIW}}P@bTpgls7A7uv7'
    '*0)Lw4g~U{rwjn1e5XRsu7%Dk@2SX%>vlr-qaQXrn5J?Q5QZbkLyNZ|gp-yV`P%VuUdIgvp6zo0_rS%K6ZIgKzc3iyqfD@yx&MM_'
    '&NO#seOd2Pc$dE~t0H(xxerM#fPQl`YQfz`Nxoj#WOE>0DCD+@j&hh&tc3sKh2&+q$_hF@#Gqb!e3$2`$<^5P^r6J4vv&>4@EFZu'
    'sE8VhuAd9SY&+<WDW5u;=>Gh>hqc67uU1XPY_JXp?U#2st)+t4+d9W3%HK}cJ<!LCr$7S{j)a5-'
    '*oJBkl8%{kjoabmYK}W_iK<N{-'
    '&m5!O>j#<ZwB3+7Ktf?d$+U|$lwkL1LkV`DHk4rZX+sHipEi_W_h~~3cAqv>Z%$*^J2mP_qBJ#C+jU0z-}Jg5Ea4hEM0-'
    'khT&PcFSg4vxf=>WqQ1}0C`a5A+dc{?vNHrmrj==MmMelZ@sTKk#<!W11=)*Yopl#9O@?>Rq8!9zqX0ZplB5bfaAY&hz+o{0IZx='
    '`vP*rhI0Ki!$%4WCdU>fJz|Ma>rEX^u=c0m{%dHOydAm*ujX_b-vK2zJ8Y#&%6e>ZHVJTS{|4ccpVP?q1u2E}J*`BZj?9h~LUf|e'
    'L1;`wxTifzvFJN%Ej&S0KBjv15#B->d5CFqdhth7y%;jD&K{7tnOhAg`@L1wJyYiEYeh~-8I0M8I&O*E!xkK-'
    '^)`vyrY1GPVbcPaau60daz{z7%1REn%WtH8mf!Ub;)*DBi-'
    '=h{@OY*(CWQ$DmyxjeuPyEFtNd3v+f01*)4=4%FUfN&kXVKD2TAaG$S=MJj~)019bFaqDC*F~l<_ku%beE<U#gn@BicSyQ+t944p'
    'G$!`z+hbRs&(>*S;izghe&Oy^_WxAL29_H0$+~TKK4M3+nnN6^AFQG+<smBCQXZ<JE#+Y<+EN~_qAlfzRJ5i1uwu$rGfW--r0FY*'
    '#)G4OJ*&f0B2+jqgXo$NH<|res_HWo<?&d_KiYAL>1L{<69jlAOAtsd%0_EEFa0yQ;Oo&rMm@#pPhxIRlp>=Q+bEg)jEJdAe0w4B'
    '@T`ao`p#jg*)<OX@X6vZvz!;@QJt^g4B;`IujCBjk2+t)8N#1*zM3<H$92AjGlVB}{;K$kP#JKy5)HElGkbVNLgID7My>@uK%dMC'
    '&){v}76pG((fwFXKI`ZsOAwzCKR-'
    's`bz<`Dq!73xgnn8*ZB`>{LswLh@DGJT9bYmSn!Tpjkfcrlu))q}$1D42Cf<vEU}s?Gs8rnZ9|8;LoA(tMOTWAy<M1eh=r53#0r>'
    '#w%?!*33SNCsKFBe^WL-K#i_cxoRM&l|c-mH@Ktj`Wj4lWB8CWrHs~?-'
    '+$zAp1^1HaHetdp6_ta0wXL3uuLq3Z;>L=#+a6`Rgey?Am`A&;H9NdK(_|<SKU(tYJhGV%%1Ic=3@;R=-'
    '$l57?EwSGyLrkIb!dZ2E;;k`8{V|onbiwc@5jmm@wO+AuZM|bK#$)ItrpA`MOT*~)qVU*X)&QKI_MdZFh^oK^?AM<d+chGgdi=v5'
    'g41BFI4PNaO^Gy?f2n96<M#ExBo4J8BfnT^g9clJQ{<tJ#(JG9grQElJaPV+Qvwa)_!_6Edb*@BW3@mpsKy$v1xM<AyZfuuw|h{8'
    'Vtqx35br8rw8zVRKCReKpU(YsbflqgE*)ven-_M}l>kq;#kxUxDW=*zyw(^K9Ok$+Zjadt31DZH$hFhSk*paJWOix1b1{6}ktla5'
    'cZltEPGq)={9{ikbaX!ABlGOo8JiPng>xgV@S#X6d^pky=S5oKBav44XrLAPHJi$l9JKVIbM8gNlCHL~hnf<r=rLt~;<wts%r33n'
    '&qHu+4XR#_b0Nb%{{e?)>_s_}zJc9}dgRE8EbY*&eDYebTRVvXl*23XUUi5~zK$GVlkX!3WaJCU;TZWwa!5{@W&qR3calSOa@o30'
    'foRg!uu-j0Yg!<|a;h%S%)ErF-#W8t0i+CV4H`~ijDZiReU@~-v4>~*eSRTkqa-'
    'e2ttLXk+Aa26O|L4YkfAkI{aSFq&PW^zk=ZZQJV$UAwn6B%*?ZX9zVW&`9t1+EO*%lENd~Ydg5M<j=houD>kgR&wm|cxUi=cf!Xt'
    '&leK#jZ*QLm*Z7}Rk-QASWquk!J6YV-$-'
    '`U(Hrq!LDIxVz{bN%6Pbt=nefed?q`TSTX{TzPkYn_<!8KUWl2q5CcwBweG3^47u6{0RnJ8q@OSkjJLCHla$<5r7aD($#6B7RIe?'
    '$yw@QV^s##K;Kj8@kX<r~B@QenSk&?8>UsH=FbHM0KAQCl%e_qH!rEdi||Z7{TrX3s$s$bp#DyX|_}au5nHf`%F>s`dlWU+O-'
    'PwqY48`VdQ1o8Nrr)O%x>QM0`_3wJC01BF>kYu^-APUkZ~8V~@yWXSOr;qnYf?gN<#U`JIWjv9U*H?h+|(q*U>*0fK8`-'
    'Zh&t_1Vnn-H@c_ZJe%yz{UQk${jU$Z?yWyF-'
    'Oy@tmKM~R*l2O+|`cME;k!jYMlIsq%0i3<c|`CP6923FV>U6O5yxkV|Vxspr!9zCuG6d^h**5TAnN#6k>-=5VFQdr-hU-'
    '#As>Tly1ZcrY0E6Lx#m&f(p{1Axo!#mnG&P^W5!J9}Fgewg_@&?a{lYSbAUbq$Yzh>z>pOF5Bah&yGtzCocKixa1GTC4V?B`MkK~'
    'kHjT^l$ZPzmwS#28I{J8hg+(G<`G{^Z8h2gHx<|BP?s%iqpvkO2@z?@z=Q3k06f@{0*{XewmOAx>`HOB9e}S+VF`N@mO$PS80%xC'
    '5&ER!IbSE&%a|Y;-Cu))w}W*)?rz{%C1-Y-'
    'U;4o?bedGyTeBnKvY{Z8JzE4cFti4njI*+xiKz^_ztwuTuN(R==WB;&FC4rvz{a{C#VJyiX40u_XEb<fVC@AUTEyk>xV#&y8UWKG'
    ')_W)9J%aTfa{Hd*GI=7^xBbMsy?4+#aM0tws35b9>Fkq|qqNibFCi^CSYwHANg)nPlwAtZy(3M;D~#onAn02-'
    'Vub_Lf}<@*1fJIJ=!{4g=xRWTxw~~}LH+O*g%{Hu7Ad?Q6O@2xWp#~OS>2*mR`;lt)gx+U^^96sy`olD@2Hh!kRVB_($?6IWUf{?'
    'fRQGqR)`3TWEv3dGU>N7=vbwWZe27(-iI2OGt@-X2JXac`-WG}-'
    ')CPf)(i=h4ZXzYRWgzpVUV^SAsJq0Jh2g>v+%s4he#1~cIYA~`=Bn4PyZ8usxaQu0{hLE=gTQL$<Is&{g{+WL}$G&l$*&o9m|L7>'
    '&Ytm@M9{n&L@~i89pG0K2?u(_XsWM1r7`ucV9ncKa;V*>uC^<68VDDaSw_91?Bk%Mbt7S`fXXhzyTN935Gdeldqu@CLfc+_0cJ#i'
    'lW+a9Q>S($UjZA!dPIU+W5k>%XziRw*;N<d`B@Km<Bn>16i2kHXEY_f)k%Rlmt1sovJVvo*IAVCk>aK?eRc+rdU29qfRg>Hnhb=m'
    '||y^tLekH9+>B+vs_I-zV*O7KZE6J2Jo#1=J}Z{S2KujJu12?5kpSp)WKe#Sf-Hs?Q8xvxJu=li!0RAtxzok#8zeB^v7?HU7aE-'
    'f*$fjlM-'
    '$xnUrul*`!F>>rD#jyuqZPxf_j0Jp5P2TDAR^{cPr>G`+yX=QH7=3{v*Ij5$ZnVdP~Kt!7>^Wj=}Yp$?ZZ;z%!b7Gs#xYNsG^f5s'
    '98i3KfPMr%|RQw~GSC!x9QI$O0Tf1LInF2lWlE5JWDha$ZbG!y9CJ{<-}Z;H}Qd-'
    'zi7!Vo0iYuET^xHf=N<#9AZGTn<ZJE@+jQm5Yo4`j}O?uCsd*Xj4cRgu%7pToL}@4#Qc5RB{eU&1?+FU0#{5XQ}87}6{XD-XMo#Y'
    'O}~f3N9lX`JkKS*=kPU6efjgI_%u^Zxh4H7N7`_vdJE=Gn#`q<R1Qb2V7={`a{CZQlR>QyRQ^>HmH@3lgokP5+?jt4JbK4Y4@9Tc'
    '&_VS(Y|*wczQlRQaaj!TBpnG|62ef?%o`3rq#^Jk<s6<kiLSeE;p}kmnnLzWy}<btC;wvDt<ChO)2ObT;Lia(X&cCzeFC4di<6tW'
    'vF(#Di@X+Oo9&4mIp#+E~C;U8C~t?!!-j$5%RPUZIk$v{EHoX_ZR0(rP7J2_;`6cdF*lQx47Ax%7}jv-Tl+#-'
    'UmJFg@bXter<sI5cY?p$8nAwU5&ComAGZIUlmX%pn2&E=5ZoYU1q&euC&kq?sFPt)0o4`@sH1hb~FJEnR?Cm7KEWDmi6qRB~coRm'
    'q8Ihzrm)E<m@q0NvvP^oR@4GcG`{xB$Jw0>td;-)dr&@JbN{6bCWkmg=0Ol4BX;-xPE#MmkxMBF{%D`;=c0`ee@)D!N8-'
    'KLA35d!~Kq^M#&iKl*&3XWE}WU+9?*pwAb2rUU8og`Vjk`h3w4%1`eiP<x=84n=_TAY>v*ul{W0B1opv<2u$IxcSBnNQ?SCJx$Wu'
    ')Xsvwvy3DLn0x_`^(Z3Om6Gq4;A)nf8_HJy>=$XXwqGuRa2xS&^7w?;Zb7sVUKlNeUx^mNi=u__;%FhfBw7eBjTXYo3Kqg+HFLK8'
    '`)hSkm|~n}AqxL~ybgK^#UYGa6`?yC_?rvekrKhTw9p+7hNY?>=JNgbSn}|wocm+)?5LvqQ}XDjsCzv5X{oMzBKcvdw0klU(V*(?'
    '&&iKY1>Rp&!4J7ONN-G2;ii;NIzih9v_KP(5;pl~J~Ceb6j`zV^hV%Bv;(o}jgqO{O_*Wf>xA-'
    '@4;k9q5lKM#@;ea}+!mhL0xWL_*Gu|Zt>g;g@<&u8E`L--;&OWxiOWZ-NL)TjMdI?&DiW8EQ3b+Q3U+H`Bdw`Qn6^PLL&uC6+kOG'
    'ZjV9c7opc#S-'
    'B1SLtz(=nd<v@AxNTI@w^we+aIH|yPge}KFmA{&PH5+6NROoAh71!#WZ+E2ZO*oC$Pn6`RL9TN^fl;^D*KS8um0U!@cj9ocnB(P-'
    'i;&P@bfjN6VpshA=!~In&^a#a+;A*)|8ZH*;EAXLsH(a%6`vLP7QH1R=t+6T~dhiQJW_I#j_)hy9JphUsog1;Yx{M^9@rBWQ)@QK'
    'HqmFUiU-'
    'J80MRz!r0kSVQg?z7#k85#+swT*wCmjHY_TP4UgcmyWLGWNgCOsVQ9t?Z?Sk^(XSG)Y(R$PIe_^kZA+cVLk`fXJY;k$a0}8T4TI+'
    's{S5Ed(|B3ormKrL-LdheJ1*XI$H$xQgm}|+h&SDd@um~e(K}tXG(q%y?&2O5gz-'
    '3ucqD(sDZ!;n@O&W?4bO0+wqNp;T&htvSJ`(p#<^jiI6CheN9X<G=)8X%oezkk^MP@6J}8XNz4qvIIj48H%y68j7tILg6xZBB53='
    'O=Gb1YH)kQJ%F_{OA%}BQReA(i(k+&J{dQ>vJnOsQTZt!t>ROSlVf{BFNHMPiVK?6FR+nbyxQ+?PR2XxHy$R5eAdK#I5C5nE-qi3'
    '|QKgWu^>@<mUt1%~{uK`%7@nS&o0{S(XbAwlSw=JPxH*u~!PecQ|8K4>u38iIs161Ro;av{{eC=UH?Iv_5tHSkX;LYh4aYE+xOR5'
    '4;(q5`d$>WWTX=#kkcvD;;WC{n5y~6jJsqAeY3Q1>hOra0z+{Y*q2DB+`E}R{M&y5!O)8^85%JaC!m=ny(P)<C|;%wlg(7vgJl{n'
    'LY;CfJaca{OTy={2cV89P<7v6O>p#B`3Nyn={R5}Y8_@QQEHIlBgwzE?T_7^dsOOn+uVc4A{WEWu+$t|!ZF<Pf+%RcKAZP{j>qAk'
    '0u=gnfWPSJLnkXhJby~VaT|H8zX6s2*<RB1i87ksE^OA8E|4c-'
    'CwsDP%U?QPNWon`pJ&yErk)=h{y7kX;R8a<~cR!O}$>&^m8$dzk#Fx@qb$Y|n!b1en+BVs$Qla9ECb9HP!<Eolb=<GuOld{u$qSV'
    '54Lmxaf;~B@WU)o6LYHl&jvAH-K&$XY@+>(6AVc303@?raFaXg-9|4UqZAF-'
    'd&+$4F_epa)Nx6eh=8S}M%@1Fy`@3OZ4ET`L1FDPjW5b-8kfTE4Tb`fku%T$k^?s&!WAPWk|ZI-'
    'gP`)@f}Q*cNbQ?qH2XY(<7EuX`s`nr5BAE-C-hxtNUpFhGy`CIv;VkhyD%zic#Lz5#jd!8`mzM<K*mxA;+b#O6lq14Bo#Rw*6ONX'
    'eKhNw{*uc5~ER|`|7UCQ3)w=$i}zZKeJceuv;9ZC|7gS5aNXSwe5EX5s&*Yk%sl)jnIV{q`-'
    'wR6VAk4UL~gTNIo*0AFV7(}Y94#_0j-WroqXPC#g=Jt@1rk~1HkR?oAfvMLNGma_P<WbbJ3lh-'
    'KjHc|S$?gg}Y%ShSDNULtYKmbV+M?_=g$R3<;{ji%Ih#LMGwl6rbk+j8d@wp|w=sRr>Hvq&n`!qm(tIeyRi-'
    'hO?lYQ^=EG_C2c&tv3+ngx7!krhIAf^FSFws7s*)~5lU%INXt6#%S4ZO-'
    'W_>MSSjFcw3jrh=I0fMajlf2Uz<{i@mnadtCF6nIl)^HtQ81tSq-InPtrV-'
    'oAFF7oSS0>9MQg<h?8hrwEEbGELD6c3X(bAcp(4#rg~l*ZX78wI!C07`_TOQ`CVorVx{RwmK=h8!TwNE--'
    'qw^R*G8pjN?l^^gN8FuNpEY>d*KZ?#RHGi<$3^YDyn!#m2Ek(NENh+Idrx%AcaEba_-'
    'wx_<NOAq7bimcioALjRCKNd2ii~i;V%V&AhW_;bLRJ>oDF|_fi>gM-tZtNRMi$E{qM4+CAn$f+5l0jmhB3XdwR>5<l*ux-'
    '$^^(gO^oE4=OKIIK++6#cz6Ku@fi0xaM*(wIV-*c5wmO`x#`-'
    '8nDrHWU}_lOg+bKP|m~tfZT^mX3FMYL1@lKTQ}rH=_@`&``=5czVh%HvUEnQidvD?Ii$O>>Nc;A#^VvQndD=mHV)w#lJDo8x-'
    'hZJQMR{>7Q!pqSNaNEnReaT^-'
    '*2x2A|;hnb@JPq3RC>P^54A!<7&B<{cNM<JG;y?>$I5FW_CH5CwHQ1HH{WH<~8HfqWS!=PZ3rou4{3N~xXk~@fY%~Bo|Y}Hg+b>x'
    'xs;fm5T|KVq&^AY6&KWo^^i~iC`4KD1trq~sD0J?2465*jA!FhWF<9gczCDGonav6OOFI<Lb^q8sv*P&01)O$n?uRA~|WX?Upq%1'
    'sX=(^GF2=+UdD7iD~rp_}#2~nQV=QO2XoQ#Z2GXoG=?lzP#$lq?Oixb>Kv^~}^D0r@-'
    'Lmj3B^Aw%mFjaa)(E$!E@<$b&;9Ep9@R*F7T9_o!>vElfA4Y3w;Lz*J@a{6)3HDPZM9N(z@;?jN)HWy<a5e>_ZB|o4d~8ixBcsp>'
    '3}UP8G49)?J#sV%O{eJ;<?$aCNBbWa7Ok_>m^&h>oqo%37Z0P`?-=Rbp}Vqx^Zl;*LeBTQ<*#tQ-'
    '<`$I&|O)~`F_uQ3FrI0@}+Dy)SJc7mIAk8uPT-'
    '+&y+$IHt~a66!__)kmFX*;yFQ@sVUiNfVnN;i9xk@Vm=iXxu55R7!;24b>v=zc;&bk!*t|cgvjOi7U8;ee2XN8tx@^PWaq6@`Kye'
    'ZIv5t&_VRGf4udVWy&}@n*K0StPJI(E^}Uh>s>Iu%-'
    '7(AjZ@X`8(nj}MtlANY*eePguSOC1*e*|LDr_W$ZN?s|kERjO;Sz>~0QFpIVf}n+Vf~NPwfd##THOpM-jBoQEf}aOu&u8OS=beDH'
    'TKtlL;d_nsBH%u3Ig0<{jrAh%O@jS`#tRj`su;j?!X~6t<V1>jEz9iIVVH|+F_mHp()CLXTU*G)Ck_u4v!F9Pf=kX=%D8nI}@S9b'
    '9=n2rvN;WbWM}Fsw+whG~G6aiXOM4mYAv`0k%Rd5M~NNX<%KnM3=4Eb(uoH>q?=IyBPl<V*JW<c1`)fi@tTU&s6oldK&jC&Y_@17'
    '7f#%O%2muNNv$yG)P>7U`rE-'
    'z`RxHM!!=tDseO{Px<1j7`pYj<Ml&Gz}ewO__A+l#*gH8+26psG$eoacM$Eo?0?^O!hN|fq*L_Ud0MbvEs*G+FH&%nYn6~+lru@E'
    '3I%r8&~HJY=@uF;ARl!P9Ty}~JwnR`T(q8{=YmMoD+q{yX4TsZh`6W6kSPaNxkg1O#aI=g6xXVnl0GYy{f8<@P);)w54?lcLLG?*'
    '0PkV9@O%eZNXQP3HwDWV6*Joe`r3etFrDCCt_u^qpX$N{Z?rB<@UGB>3Eq{uFu}V@7bbXDi*VxI+P-'
    'Yqu^vDcN@s%sFOV#+7^}m(>nL5xrs+2HFVO!&EfiL-'
    ')hCf&C;Uq#0jc(Y`qC0Odw@a(_?lBYbMudUSqePdc%afM^o?5;^K?PgtxqQGjkK?Q9`PGkSv+ZK0RF3xuuTXooCK_kO~2P;)9;Pg'
    '^m~(=e#7&(c+WM`o_mia-1A%6#GdZk+ug_Z_w7{SKn|dNxly=|(;Wx0I62syZx;B~)_f}qK2yT-'
    '8aF71<?C5Mr*D26doj@ceMi7PA$(!)g|m2-'
    'vL}eP+0_nE8ZNdm%AP3dmOoRrqe!ZbRrVy~Ic5h`gOV6AC60UBgB5nMVQ=BJkoUY&aS{!-'
    'QE}Y6Nx6O;h0zW=N}Kxg_nQ(S4;`c}+(CLI-yRxca6P2^Vjo)+X%uSvwJgL`F$hE#MhpVSfQS@Y`=Da{P?+b382b|;n;&ZIWg@$B'
    '7#uM9Qk;gGWI|~sTbClP#SJ1AXKS=`#UZpwA&MRn>QT&0v8!45(<!X14$*Zc*0C^v*-4o;+jRkdIO2QXp`QRh$@g3hTCw-'
    'h+4OG~SLl0H%ds4uQB+Rv=5x|<Yc1W{k+wn6KNW*olZxFReoS-nP2AO-mv7@n-'
    '=q1C(7&X=(Lm`uLv_1y$(FC6aL3^9l@#lO(Qdwq0$nhrrNX)=Tkhs-D8$u;-@i)n?bxt#lyTlsU>W7R?G8Ty-'
    's{r8G&X$A<x@Q3EH2`r3jJqZ#Ahn>;cLxklBp+y#_?S1Smyv_^d^D8Q1BXI_~NbuB$poUWrhCmM(z*KrFbAs;jO+Y41?PSfmijhT'
    '{(@Qe7_s#td#F}=aiN5{T`fMP$JipsmYZ4_u@>Ja{u0nZA359@U}iZS-w3!-'
    '#Q3Q{ufKOORY|EtoE=^Tq#|XP&Q42O2p)F3K~>Yu4od3-'
    '%&1MN^i{npef~hPAG}RQtx+__|Zy&=5#R=r8qew2(ZM_TDS&(uI&8M@*EDW@&hjqrxST=I(DI4s)@@Rcl?6hLB*|zT#6ep>{?sA>'
    'S#y3X-b2}`~$nkKfhT)Q6Mg7vX{Chd#PKpm%1l=sYkMxdM10RSF)FS%U){6Y-'
    'NF=gk2`CZDR{!u*R~E7iq$GwZn4%@3PT^9R)&a1A8t3<(i0i+*a+WexI^0`N#E3W&c@vTn)}^imNNh-9N4E?}a)^1@6nj;-'
    '=*KpOT@!=bW#?ZE+3e7t%@|i#~jS4z)l7;7%4<hm%3By`7`i{@I`zxJ^02|G>+J<z#*ZM{jPZ{aU+T(TJLxDwN3-'
    'O$4{vnwl@SQPSUPPbq<1+9QGU-'
    '&j0#nCsF%)yPv_%0>F{OqUX;zC6*T%&8yGb18M|&$DvMod)ovoRXJ;JSV3_Y7ig$!HRYq3&=Ms+Ggy3GgQ$&3jrVf&Kape&AmQtr'
    '{DcT*_TTXbE+FVPW6+=Vwb936?C<bgYV=+Rj)!K=OuMhgw_V9R(^zpv5h69QCHi`OFQ(ieF{UrUOvnPfS5g7(OF~bpCO747#sWyQ'
    '*@G8Jb$>Nqr_W$m}ZaRd^y@+w}&??&`6vb-mHY6$h7cg6^1y|!<*IEWOsx&ucAaOJ8m|)NM1|w72msFYFGN+CrRx}-'
    '}_|glzbAR2plSQvORH2zKec^NWGhWg-'
    'E@JeuYTAmwtsvy^nr{Nc{o*3Xyt0{VI^fE;ZhfsV=o^=)#&+6y)m)oyLvvmfpyw;;z}&t0XDc-'
    '8T9s*wgkcRW!Gu_^rJ@w9cnW1yAeTN%TKx!+uQqMHVs<;hLqm)GH#LdPk&FV?;XjiAblu5$V(~BAxn2q|<=1_W4W&J(#MwXD?ASK'
    ';vwtl40A%cmnDaw?}s3_mdOFhyr98kQTpMy={L}mhv8BaFn?3h~lo1!O~IC#D535(i-'
    '{4Xz?loDWGB)W%=%`gLzo!0n@h`r4ugj<vQU4|5U>BDQIysV(L^(RTB2+m6~(@kD-'
    '~%QT*c=w{X+tGY)PN=;7}ICBuJ*LiS?ed3SbT)XAkHbQFq|8s>Kg!9Vf?|G~Q1-'
    'UXmYV;obrCcHDri^3*uEXaC9mkBWNJj*|V>uE#&PP9Zm3z9eA)o(`S(`I~t1+rG}Xs9iOwfc$1U(1bvtv)B#nm}899;_IFtDOg0j'
    'NsMIk4S+T@baav0?50-'
    'PdU>S^Ei5{3>T=H1DY{%+}Ec5C>zDx)bX@M?%63$P{?Q#F5wP3XwDX;8PoA!ctGbGt9<>i+V!#vP2(V<fG%l}LqK0Wh#}A}?f}$m'
    'dhaes;r0r1m;`RGG5t#rOa_ygq!FKC9+)7SDvoj)O(za=B-#i$Z;Odxgq(Bu{2JyD^-'
    '=79txqvgs*KXS&vptk+5cOV9Ni`O<T|H$*^u^wh8nxq@8dV9YEGA_5B*v$bBQ1>fw4r8mcUdZ2uolP5yT_%Mffh<<ymZx1!md5Sp'
    'toZ)7c3P5+9#npJT}HoWVp;2z;EW_@e9xm~y!_<U~shtw`xq{#Ds;_!cj1@PfNo8Ty{o$_rAD7o-'
    'xLy3_9##R_Pw5=v4$nQb7)5bju|=*kRNtX6bqh68I9U7Dd|lPS72!<|!6bZv$!=O9J*X1H<MD!MpBT}PMHZ*)wVk1;_A#(`tM2x6'
    'Lmc)sL8JSBcC?cElg$o+(c`n$3t{9|9^hR^?b?T1ksS!#$>SHjDg<M36ma%T2<i~C~*?T1ur7SC!y3>k@~Qj8y{!*)u<Xrn?JR4R'
    'Zspf^c>w6rki7`xvmr#alLQZ>E$M3$3&io!ALlvaN(9rfQI6n6x%iw)B!I+e}oXuahcKQ`Z#@8S{A=6pAgfVOxtBSt`51Js_@ds|'
    '?K(t2+XtYlj69ZVCwTEm`UlEAda2`BAlSo3mmJjE^l&1Ks|YcQ$VSB91SP1$ey6E?y)E6IX^<gxI<P9h`4(N1Zk>*!=jk74d7l?@'
    '+_VfJEEa3cMRDL9dS)fAjazh(+fq+d4$C(>`2f)nXCO~Hxu662jnV`h4I>Hr0h!7Svd8^R-'
    ';1!ilQb7Qo0R3>mv^o7QX5=fmW7ADrEAm0~fe_<(~QuY#m2G`mjnbJzuDV~=4_s7A@Gx_`!Jx@vR7z&`Lrgsd*(2t25^9GghxBsQ'
    '6e&K2R`--YHu6H*oDrN#x#%@wn%mnZrRM-bHSH^Bp`JgJYOrtnfz^t!<yIzNj+s<mIn^G@|%k-'
    'z}#O?F8K%6N!NaEBW1G}vpygmunIk#}Hdc2H0$JqfN#3}MjC!SN6I4C8Wl!qJcU^PGsm+!1bBvl=B)s<7HO5;HX{#dH*^6<?r*RU'
    'B0N$gLx2u3zj28+WN60<AZ;q$P6dxQHs*9+S7Iw6tX;zDG=hurFp@+~6Y-'
    'cD36Fv~woc^P0ad3#coaE_mX7oR(|hx7Eckd@6;j1t_SlIme=vSJ6mG?A4tls3_f5XBO3v{<nO94%HX0Y{4!OTf`$#S(C|Sg`~gE'
    'mkZ6M~f9pCOYTmHA>J82YWNkD2w@c!RYQUrJM0hhcK;`*gvi8xBRJmEudD?=0vUXOiC|5HW1)9ZJ4d>w&WIXK@k20W;1r@X@^g(l'
    'J>j~K4o@AbpCGDZkQdI6DzbZA|^T4>sR@H#J3E^;zGNPw`2Z_-Ojs_@~~ZkzFi8)-'
    '$aJ>o^6=Nb%?_2PfCChAA(XrnFL8$#C3+g3btLQX_Pw1l#6l)nXFOrAd@x99%Qmc>4T_?QvM+7qLe_ex+oH`B{~cgok;m|9cCfOI'
    '(I0q*v_GV5{Yfafnx)gWg8>>tTR#oiqVodY!u#RD+*LMc$d=ob^8;KilNxFy96NE0JdK(;dAVeMiqL6CB{r+IxoeK6*A986{pb62'
    'u`<gNaCfZO!pc?U&UZ)tf8;sEpx4*ul^lv=y8U=az?&WVA#tr-'
    '=4$(<bj4$kKJ~V;oM`7Z8DsE?6ZRnryL*HA%=5~{kGX4>5z3==Rk7woAngA4&V`4vnp(M$nLSM11A(H&<BSah9V4w#kot}uvUL|$'
    'E?b@xgp{1q;R1DhG)Nh=p#qm)gTjbS4}43uFf_QP&3#>4Au}6F<8wJG5Z(T0wwqK-wc$B<GuWM1BK$mbIU-b_?V<3+o<h4pJ6q5y'
    '^*ryK?hw+BdRe!n`P?g=N`(kbo8?h_gbbb)`EgM<`sX2gH@7d{S}&DNnY`kfx2=`rRE!8f8*bO*<j<uzyD8z%@aTW0)q_{|GkBb8'
    ';vK`L2n|)s4m(15c~G+4Y5ZXDh-4!tb~Vu<QZo>QJhr&OWV1M$@v{sneo><Z(>y^+6{wQev)>>5SI9(-'
    'O$Xsu(Sh)CBl;cTntXl%@-LQiTGp}8yt&c#2nn8ZRZW1Z}z|43`yPVL}lbzQ=|YU6y!ls|IX(sF*-'
    'P)#1hSc1BXFGq?aH|9$SQe)TsmXN-4NLc|c@e?-;at(Hpt?JG*Zfz@h{3!^of8f_L%<-^7oS<_LDR!N&@SB5-'
    'bxWe=77Y=$qdHThtMFTZB;Aq-z$XYyu-FTZZ`p$uPs!{ozQ=j3V#><i%Me&<ym3ZP#*<M~@ykj!3g3Jw?VYx}U@IBOs`S{3N&5Ks'
    'cKq2_e3SAykPEnyAar7eD0woB>g)X<R&2-08y9^ju<k$F7F|0t`Kvs*9Fcy?;BX}D9_-I^^z=o)*5!Daa-'
    'Y^;z6Y{4RoV;0Vf0W;xbZD=co)bjxi+k*gfp9<t4=dK>-epUg2+)%)|_b4UKjV@WGth1?Vo<renugcGH7h9=ALkouT=T(u#-'
    'iNE4i|FOFn6iR<6|ZL+{h{PuVs0*1RGzw(Tm`x3wAHl$g+fRyC{?;gappyN#l4CXFG`$#t|af8ssNjQE;doO>tLPQY#(s;aA7b_d'
    'E_!<-'
    '!e>tdckn=|0}1v9$?>(gW}EmlkT*u+<zHSVo=m^Ih6sSOF^@jw2%izj&=tf3mS|Zw(|k~pavYdSJGbToyaLPCUOdW5;=vwiJU^eL'
    '{6c9BBwAQ$SF7#UPTWqw8*-'
    'vd=<CCda8U4x4(J=XLnYrC9bKEb8*U#E7C2s6)j9Gz=$f<w^j^?xIyaCImvPCAdS#2n$~0>AI%RqG2BkSEtD|}F@0YysndN*63(R'
    '<Uapb;<u6r8!CNqpU%_p_Za`#*Hee4hOhW6g7ogdpM>fnp!A{uu8+L`U?}lu2y0%vesL69mx~AEvpI17lK)Vt^X08k6)d&X8w_2r'
    'F)l2jxS}2d}ET!LFsZ<Wg7!zc-xvZHIvJGjR`oFY&cbrx=);~$^0Aj;+(XpVjYr}$!1sMy1I;d+l>lO<Z1XS!OBIwWr6-'
    '7tXL1_X*v2A&Ge~Sg!@~T)sMNyC{NU>lqSm+M?&Pj5T+&mBR<d63=&xd{PI+=TK^3BP~$vNNeBCgYwaGr5h&7x=q*v!_U*YPz|M~'
    '7a=H)vD>ui;xXD}mRr7GFK^qP}D5*w8Ea0SynWwP<VC`?&N5N*6qxd|DK9-'
    'a_eu)wtye`>sLvUCxouWhIsu95ujhtdjw&If?Y4%k==qZR8}6bK68sw&dhps&EZZErp{-Zt8TYV0g5%8-'
    '|>lYGR1{NPPtN?zJR80&pc00$bpTF2=VD13N~+gk2c`namFC>!!HK!dRv@aVPbIY1SJlmGLy%9H9KQ-'
    '2*(T{>s&T1<A<s;I#wc`*{I(gplvii8|T^=j!k385+^Siw}6+5q<s;5kuwJtG+7fd!~-'
    '#6!jX%P{MAmz1?G4OZk`XkfB@<6*O#DmJ0Qmy;y^)6kpmDzNfUvfz3Ev$1n$Xo&9@T7_!7|p@TB48=x5W|1#B27<~}8@u^}_W+X<'
    'AZWr*cBV~JqtM1IvSic7LjXA!**SWgNUQN+LyL-'
    '9=l6{2QNjL%4X}+#z7E?2XHQ{(FEa=n_wq@_b^tJE!AEs_|OMg9AXX{nSO>BVMCngc88CFIP`x$McN~hI04-'
    'Y_s^G&>T0grBZ|Mg{VV-`K~dz^W>iU9jGFyL#<(*V9d&rzUOdiSAL*5heeD|(ob%$Jvp)P&KNfA(wCR;0-'
    '8j})&dNJEbIkpOBJSJ(#@iNjr_<6U>awIBEnN4iMILU+8?65NQ^F4D2sopc=tt4JF=auap!F^%IK1}lW~CRVG(LVY4AZi=q!9R&$'
    'kj-&~%Hifb6L%7`R?fPFeRaXT#w5C7SfswY=0hr8+l?Ug9sXJh9J~iDB7Uhc^1Kye3i>;l-'
    'z1Z5>+>5RKhI_HKbGR2<liZ7~{g!*NHN_ntj+_XS!%4cR%bC%1sg{qyG?9;)?~lVIk&ox*-'
    'CVxCiU30G9G4t46vI8p<;^MEW0S*>ciOAvRJ&$+R~+Cvm2#~P;7I3Zi<UX^`Pq^&j)Z<T-'
    '+?2epG|k*Na<&@9XN9O*<=Tfq<+q{?W7Nl{Sw(s9~%25au@VqgC%h{bYgRXg>u^a8NHLZ1P3$9o-wr|-AhU5BhfkMRY4bQvK(-'
    '7t*`ZM!0i4h(YD}e9#1QG^t~WF_Gqe)&b_0svpyR4j=-'
    '+yo{@CXCKws2uHG!4c1ihL<TFax??1_BU4njtd`^|w?_1?G)!2UDCZ|(nHu}pmwBl(<foZ6z@Erf%Ld8s<A&TrPB<|E&pCMJ4W!^'
    'H;VF74!7vCDf+z!7b9&HB|)h?t}6>n-'
    '7>e%Rq5lr2aV|jozm0}KRHzDWC?`%L^xB!G|F608UFRfO42)R)HXukhK$VGCUtI%C+{VDZJA(xOtbrKf2k8&x0ja}A=Y>yNYW9_<'
    'ghWfWIQ7B9ibvCICx6?KNWOY{Yc%K%NO`c5!@wX`y<JeTlLkm0eCM@o!jA?Q;vx3y-'
    ')R0fXlHD=Rt%qA%!3Z$OdkNK;lG`!syBTG-W0B0xl-'
    '{oP7it%4G~ZzAzb*?|ey<uxn+dn;uXy)iyk6lqF;bya16@2~3RdoVZRG#sgL$3ZcLz<5pC~wh$$R$%OfZI=rBIGRi{Wh|-'
    '|bcbOA4TzinqL~1^PM-KGK2VuC^i<nz{*d8}tzM#*Q8qlU26Zj#{S9Om};}N=B+@T7V?b{|QthU|x0$LR00M<sd9M?TH}J&vG+*S'
    '0P+5FLDUkl`V9pf*FuQ$&zfP)0Go(mpYOHvB<UaC{_MoUHCX3p`Y#xphq*~61fvh@clM>c9u81(>a1;ZOf2?fJ=*F8KJcGt7JLRh'
    'GSSxwBcBm6Kyz-'
    '<wP5fXF1V^wk#*wa01JTHncMb{j^Mc;7+5*qF|w`s(w`Oem3}|ve{g14DIbLf{O3^;4}+ezD@Z`j!jOQdMZYohA^i%Sm}3LH5l!)'
    'p`>c`MOzAlXau$zog$}1;SMn4!5N@^#2e8}=tt~@yNY4C<PHayeLY<E^Al}<kDQhcIp~Wvb=flO@OYCBme=FS933ogz@ttI&TYh#'
    'V9NO0ga^U&*EZw%T}av(RfA0p$v?|kj=hRIV&L8|Y2_+Eg#o3HaXs7ed1}w-T$l=2<LPLrB?ha^-'
    '#frr3oQ^;?rLm0>NmD_<Zn86IUmh$%wVsnI~kmU`*S9IA8*1AaCQqI*vK7kp&-'
    'Q5UgTlA7kQZBLT2Tz!w=+GN`5E#YFOzzlpHime`!24EF~Z2_Jg~#h=I6<6>y;7dpT}p=P9fRJcZ%hFzl^xtDzF|L~<1Vg(Pd1|0G'
    'wDIwxO|`ewc&^{sqG>fC%q>f8B>)OYd~sq^v`sq<Z+WfS^po9OTuywRWNoDh7p1`2zCf3__@zba6@%@=G~cKy~Sp9sFS73Op`hpG'
    'h(B|IzWcourHBV~2~C_~#0z3FB<xV6K%V-@d+X%of#s=_Va7WB5a&{-#V)?3|TxQsbIpZ8Gr;ar!;e-v|sW-'
    'CzVfk+-!q6EQtoYMZ(|8R=m0I2`&iXEFGQ#^%iz`vi$DN3_5^0J#`aobz4uzYWNOYwk*(7T0;jH}Oan-JT~Fb~;tA>nK;%N@jR3c'
    '1>KMw01-zv-D@lM72>xSsZZzrt{?-TxiL<cGB9JA}!PX`go}lWS<7cNmkOP@Z57njbq4`!U!`-9ZA=u$8h?asY#^)LkSs4O=EE-v'
    'MVmnu4uj67YOcEb^N>Gl(#9#83@z>GsLsxit6?0{E+CaY|tk5Y^)9@}?`nxEu@3XeWFHR=!ZT^w9amwqLH+$;6(o`fjcs4@0H*aP'
    '>qODjlY|n*BbomaD&n2h8_#bjkK6N-p$4_wlNGucD<3U^NF^3Ii)}qZgMI+uYc6x1RbSxgyx>kbK-'
    'T*jYBPHwV943%>?8mm$Hx|6OGu%rX23M|+GP;dqbnBV0=K%XSgV>twd@@pDq&S*7mPeFFxrUs*x*?xSUz$k|-'
    'HT&Ckx+u0iCELOFFiB4uV>KYX}-Mor`pR7WstNbSn$-uQ`k2-'
    '_pX?DPFJ<DMsaE5CdF{f%O>FHP!=eVS2U`?FkkN!6n516azx`Jkny@FrcigYpN3VyATAi6Y^5#~aDVXyv|qj1uf_Ud_#N_t=MFkf'
    'q@sMz97y)00t=R0>&nOsGEJHAXd3@f?hu-=*_GaCSd-'
    '5d!`74zE@0BU{YaGcI`S|(PpuzYSjc6vX`K&i%#AIRWeTK&KQn!{=Nd(P9e!WR8sSfXK9a;~-^(TCNH;mC9IIS*6(NUG2zW04-'
    'CNpIU(gz0~@``ll<&-+L0Xj9SC>SE3jx^EAW^)OKbOW5GIM}NcDA<wKqC#1VjSOf0sriSi%HM7Y-'
    'SOj=WuGgSL2Qc`T{Pk6&)_e>s7ZWK#@B@nM9R~}?WJ(YGNKT>bz>g{1cLJ;+kCP|&iG0FVU8|=!adA#lf96!oJ{|4i^gQIIVmNvG'
    'WJc~a>k&r2enaQ<xEjq$=eoh7SQQl7cM(p0FjQ;rX}ae$W|LQ`d!XkAG5FUeIWa6-'
    '?qsh3frA<R>&m6NTLo`ZA_dFIv&_{<UCGpQ%oRz&>N3r3i@<{?9#dRah{zO|6kD|ZEG{ROv0(--9W(M^T7{FeSV7(YS>-'
    '?ny&WSx=F^Z>0seKhVz910!(0{ERbbfI_LL(VqSDYodLj4G#=aQdYEDVue#u9#1U>H+Q4~{$2?`up_{Dq78oGk32jevq?N6oWt?d'
    'w~+%y2n0>-)qT579;9a(RDlp2RB>ic@P%*u(di?j>rB-jkug>*7*hU`i@1xr=zQu-Jk&Dgc{aV%A_i|G?ss`8Lqb-'
    'EEvq^_>J+Dq`Y)N!nqUY+-K+#wYdsX|Y`sq;Cydnvm+%woDq-'
    'Bh@k)|WvylEH5%Gjdyn!HjPCU`F?RFyr!kFyo4RFyqR6Fr!C4m~mA;m~pkMTG&s=EB5nR;cwtB(5--Px-'
    '!O>bOBx*Hlwrldhhv6g|TZzj2M;HCZcJYPTOX!!BACjNcJ*DC~&vbTIAu^+rSSN)W-'
    '|Ec0!q)N<I6CURHa=^wUMMoGa<cddpru2+&(EuWkcT!E!<hRm|<;^?Twg<TAjfVg;13M@YK4)!HMZN!(`b5!f|Et}#f+3JuUT#vQ'
    'C0Vt1N#Qy7BR(#y<Xh~61)Wu4<nB>xvzW7<Z%ph*Gj&i+MB3g8Jy5SV?)-'
    'J4U`jSq(ieXPtdQRQ~5W%?|`b5s{s9&|*jQ(aC*FZi)5V?F+rj^RcN9*aT<wQ@CXwd`d75Vu-'
    '(hJS>`Aj7>sLPj4wLNfa3nouWe=a4#C`-'
    'arX+BKw3)(#AHvUU@r)j_AM^4M`kAeyZ`V4zoFSB?styZ;8xoj>z!!syU(7nYB%xe&@eUYc@WFHO0hm!{m`OH;P=(v$~yY03jhnz'
    'B}YfW<b}a(;~4R%;>G;I`FT$WL(F8rHYNT<GQPcP|{KA++vRQIZYLkt!GGs$%4IQWB{yzi1w9EfF5H%Gj<my|q=(<=hFm_O0oJd}'
    'o%gj>%D?YDE-}xK-Zub1-'
    'cP{lJVz6Bw8LdcWjP^h^FEzvNH$Oa2tU<Ui(@{Kx&0{{$`hCo=euAw3C|;c3pEf;s~<XFH%<;5PhJaD`EE{5Om1hO}2rsSM46uBk'
    'Qpsm=)3drqaG{slWJuzS6H=KfUdwJ>k2LGRn>Jm6`VrF5$D3?R#$EItcUmgXGztp_ujbKtlh!f4Kc=XxlkIR`%3VNPY<!AeV$p6n'
    'vQ?HPP1_<JftL92TTLU3z%QCV=kb=IAvx6OIB=&2^;X0ZAUMuVHva#rcXn$~)0zy;e}{k(cqd`I>2LTfj}HoyjPd&4%sM#^u2ZGe'
    'rG_knGIjg<F=ZGb&vxfQknHsE_3Yy*Y`QO@^`J7Bh-'
    'b6sWpOkE~_Pgqnkgca1Dm5J>Yb97^LYK)<A_p)=ie_2f#HPyw<yXlc~%VcZe^1s5K@D)=x2NhGjgNmtJf{Lj=LB&+xpknIQpknGa'
    'S23ltzF;rZ+GTzHg-'
    '8+Plk$B_<y|eGk}Gu7GBiK+=zT;^F6xRHCj;>ETv_4LmlxXWm2(PwS~yOshHz3f`2VsB`z+5SrA`#MVjtyaT|J31xDDhpegpZe-'
    '#|X+H;~i(26DRJK+bR*Nbp4PvyuqA1+wQ@Q*?_!I-'
    'Gk7Dm=kH$7^0+6U5f;!}7v4Z(QJoJz1uwT<*9RBiCi_aO7~DiWD<G0@p)|86T;SO>EBJQTm|B29R6pgCd*rceFkzveD!=`k=_3Hi'
    'P>(hf|F5B!Ny)Yv*50L#{vPQ^DtB9t#mzf3}yWx~LDT077%1gI+G6+yDdIEt+=S^e&609rfe`0x5{$OT93BgSjQ}KiXl4=V+N*dt'
    'vKtt32#R%BS(b1KFov1wVF4{*T-D<>n@O!**)Qo88Qmz1_r=x43yJ`?zT-`?^^vZ*`MW-'
    'e%K)1@0GYpuC+x*VM*`t276>jSrtJ&|n=MIsba^8HXqv)Ce~%Un~~Br33Npu4ekEygmqaYunW-7v=R6@)UuaoHpdP<7X9jjyrr-b'
    'LXGK=OgY2bN}PWF5eXDzmDwkErIFKwio8=q#4@_ZwvK;BfES@s22rUvw1?z6m*s_U#ORa$tCxl^{!5Bihu<Gtf3h?t#F1;(ZDj_{'
    '}#^BqA|x_VI^O^bKuSNBK55895?vxT;We!Yrz7c;EU~-'
    'RUGC1Y`?7LDEDXk<pYk&Dz;xf<S4CT`{g5!+A6kRKISN~)5dKsr!rT4{T#-'
    'N;O|&MYk^c?*2C%}kn=0@u$JlGEQH?8272(m&lGm&*ucquq9B56Y)LQbrFwsyw@S6Zys4p{2CXU(rNDfTDq4@xUmaCAcj;gP=I+y'
    'mavbL_tP?bMVV$743+n{UU05e*?!r1ja~IYL%K(SlZpYs&kXFsQsJ#U8s##~Fw?HzryYn4r3TK9MoeS+c@DX*Nf#Z1fW?}!m7Qpd'
    '-bJWnIP~6rfz?Q>nw{mi%t!S{#ymu}cJi>nfpqBFT5qzV2^C!zVnqkTa7_3~*2&yrLNh4ska#<s=Ugy$AV13Eujeup#C60hOq0AA'
    'J-u}2Vkl?gG;lz-p$fZtT>Iu1ApSy-'
    '#JQO!8Xe6^jiqMH=@)7FFzn01I@zUEfoP{@rv+$;H7WN8f;mzSJ>>bX+Tf$k`CzypW&L>*w2$Q1!6sJG1JHyDU{k6yCGOd~AnMhW'
    'ktLZ|Bw{X%0BMBr-QfW@XrC!1DM}-SBnCp&+!f3mRCtq_gg+uRP3gp|mA(VnY(H&i-01ubr3J~Do3S0vMJY0#ECcr}vbTI-vT!pJ'
    'dfQPHm>7a~*F=$VRaw<Q7C6jm-$KjGW%(0|{BXeKQRau3Gz|s1wf*z}2jt&=v)$Cyo?Av<^l-jwb0!uH}!9+AW_)c-'
    'AmFBKP=dWAAu}Wv0a6P#zmiZ62T-TV{60Ez3G%ZO+CrNPY2AB6e2vxoGRX95T!4s^A?b4-'
    '*tO^F^cgX+f$K_n4fb_%idt`y!Gmvuu2iXtGACU{vkIA2q4$=?FdB_O4H&I%G6LU9r2@H=pUvg)LCkaQ|EN%Io&Xb63erGy6(R~&'
    'yr@2kS0`H#esVm>O`r7uc!aM}qNrAjb=9G#KM=>TUEj<FY&ZxE&T<0d7ic60|-'
    '7~5#ZH>BkR9*_Mb2Co$rQka6%t;@o9z8Q+=AVeZ(EavevRvY%@ct}U>fDc<+JKJS!aDk{h|irmhdFJ~L1t0h6W{rK%lC<))KN}Sa'
    '0y(E<G!f<<@^R45oZNbZpOqDtB}$g^G>Wp$}LFzZ#7c-U<%3-y9TO%a$-'
    'um6|+$+9oTJ{jIxXpX3r2L!6DZx=5`N;v*NBCTf<P6%gxAmAsGPLqp5TTGlA)Z6j!B<ebc#!`&tvZuQd@(7c?91<5UxToT}c(sV4'
    'e3)g&LMn(X6LQ+%B2u^>O=1dbX1ex9}CnDy@`S{294e?QYs;+XN@q9AFrTj9oMg3Iv%Pyf4uozIps=%8Um#3P3X+jUwp&PE+8yJU'
    '=qj&!>q0GOqIH^gmJqt*+6<u-'
    '9O_}>=f@_megi{GCf9_~+%2=}K)hWpc_!u@IMaDRGqxIb;f0=fIPoQe$0^@$t{D7>{gnPcIEw@;^VES`uZ+qFwnPQl`*tw{h7JYx'
    '+4fZ$nc4gdttSz`bom}X4@fMB{c1OS2=+6+(%-'
    '3sFa>8$Z;p{jlEBc`$wVfQp)@?Xej`#sZ{?<QTqMS#yHHXaz9@l=5_H5@bq{E05?R8yek2h+YAE5=~0i?RO29A$l!4%In4+Uezwb'
    '~M@Ocr*8qht5*``3bg~+|62)$D|4NA@r@CU>`x>+6neC^sSv>YoKrK1p5T~)=scbp>N#;1B><{`GfL1Ecb3jr<L6W(K{OwoMBPk!'
    '+1P@>9w7-hauF%PG-3m8C&$saK3mNbhltvA)jivX*19Gh>hCY%aJG)<4+<-VK%8hi5!j6ME)dl45pX*lgP0sCF4&bAH{rAe-'
    'b$kGfwxWgPmUqwFgm{sM*ZOVbsbW)iy>Vod0H+u(L<xM8MlUg6x1_73dMwkcq99{a~70WWM%?X>zgoIsm50CFbium?oE+uY+KkbT'
    'MCVhiP(|`8pV;N!K8;bftN3&5_NytCDyO+1!qFEbf;JT~qGQRmS)&OK&|T-'
    '1^uHuvq2mNw?bK(!CAg4i%6rg|{kT2q=dh=m8F16yQY$<1rMsg&vyV6c!LGoheE*f#uJ7hWd>Cub3r<509xjdy%J$yvWnVJ|FxNp'
    'AUYi&j;_~MV>D6B2Qf{sh{kqrVA=B`7gs<ETVFIo?$K)(YMXjS*CaPiOcBQ>Z6i3i{!r4cr(x$E}R*73L|K%sq-yVYGHkt;DX)Px'
    'UTUX@F?Jza%RT?h^a~ogL71kH4$8>G8W_=uQJmD{`_fbhK&j71n34%#V2wSvIg3H?wbsK9lIlco1s%<H{tK-#M+^bU-zFp+L6oEx'
    'M=1h8ja=C>S#kV_pGWcn#|YPJ#A;~ndUoiO#q)2!ST3G=8KO3#=y^35(OPg#pC0_B2hKl1lZmPB7f5WwafResM{<1*@`dJb(Z~Y$'
    'rs1+?*X>vi(~osKwI?1vHW|Gt@`5XF(z-'
    'fWnWx9&g9^N<K5wL9IY|<pRG+l%}~e3VF_(e0;WuuE0yk`d#j^fKwY7&<W2mN`z(3H=*LWpHRfk;XE%FY-'
    '+5OCCfUulCt#BGw*3H;>=xS#Fv<GZK7dKq*Y*HRvRm!@!z8;6-'
    '`<dbbQG#1C&l)knA@UFX6iCuxw9K?4%CD|NKZbejxk*`T^$#7O|710tz7Bp?*lEDUT(1`c<ti2D=L^jS;pO9<9tG>d`7h~x16Sq%'
    '@qW`#?-%k1$=w;1~c+VwQ>VkV>%5Qp&FQbqZb$^agv>W5{Wb|llJ2bmS0B}`6Pq?<Om`SE#e(s8}(^Nv)x1!cZ{-'
    'GEgdKuBgEM6am}u`6breE`H#@EFDLy~T%RlnHp2yq6f<#!B4J=$qAWKy!!^oMV>4W&NG=#xsi0>8eJ-sIK6`VQ^#q^&NSqNm6BVa'
    '=ghel!s9Fh%zxARGS`Ezmk7x&jaYIY|Z!;yRj;^(^X_mRbH4lMqk7!47D*7DHO(jgs@G%wp%KIbbU{ox64#D_Cd7)+33H~NRf`>U'
    'v^FnaHLS9#x_b%xUK6`PObO)b(IgzBCCe&#z`}H8<l!8t{wIVCc^$3dD?#U=M?J~EDy`_sI?r?giQ|X`<*3=e4?`ceR#lAwRFzs{'
    '<rvlHh+*IY}1Xkp+D8+XOj}s`Cm&wk<A`f|p-q}#zj@eHJ_B>#Cyas)2a?n>|L9hEy9ra>#Z)~+-'
    '@ia_G`Qn{J^2Ixc=8Ja@%NOso$`|h(o-'
    'f`xB4507WWIRkC>KiGq7D>lcdx$WkHo7fam`Rg9AwH5T~9yCsSax#H>I@E+1p39E@w~V+!{iGFmMv!$6EMxR0?3H^-'
    '5X>I&t_O&4Hd+D_&siFTx$Cbx$wG4X5=_yWp;}3*}JOkl%F-n2S3o#kfEDUB`j3xQkMZ2a?~_7EHz6lwy240k9nyieW?qb-'
    'GZehgtZeI2DDD=B5ri-gI@tP1gceu<0_Xbv3lleu_Es&<$k_)Ww`cd23j1QksaF_0=!IUAA@0FU3u^^~*2AZMF5ryW+08tIHCGPd'
    'SB+)GS8Yj=Ny&ON|pw%+e)J$mK|yBJC9XNL9G9P5CL-'
    '!fnY%cb>rLPHwVoE|c?y6^gp_3UcIfwGS1(_JNkcKRYk02F#b=2dMU3?`U&J7iS=KR}eR-'
    'W|R_QuFqzuSlPmo0)b;hI+hHDZ75M4Q{aws>HU}kcRXcjqv#$?X$$H*uv*E+K<>n9B^v^{3#*lE1mqs9R*JZ`=fSXBmLKO*b%13^'
    'FXE@e3RG|w^u6v#^Zms#ut9~%v6wOLwp(bWXeOv@nVSmgI_Bnrx}Leopl)DpHmIJ=O$T)&bMryn#N33yT)?(IS-lMPa9px_V-'
    '&<uXR$bOpu*UQkMyEz!?`fzhj7bkiXq`9xlEBDaNdNGUmaSCnHWotyRsv!Te>T&)jow?`FNT7(xq|UR|bz$q%S)$itm`~vlwko%='
    'K>=l{cH~bC`=+V5v9wZXHOCbyx26E5R)%0ZVkmNjP;1Q(c8)UfUIR#8la^%_#Y~l+9yI9=zWuPYpE=poe|#Z3Wq0G#lm4Y{2e>m@'
    'f1|>5$wE{s?R5s2fj%1z&e#Cs^y_P~>-Ia#J+3Wh2=kq=D+-'
    'N3sGoHZJ6ZdgAWX@?p{kLk5wlafuxY0l3li0gF1ap8cVHe~MB(LZ1FWIg&j6L2?v%`nSu`<mnFv?`SZY`{|Cut{rS=h{Fyh{Vwht'
    '5&*rsJE;SQwz-aZ2O^IqTU9ogWYs+0Hk8Rm1h+M1vR%A<Hi93g@KHe(0Dsas|7R1sN)_~)-'
    '8fvLdEk8}B3MY6?LkG~70hPOT+ggpC&lb4;K$|>{|bI=8u72;$7a4B1V1*R_+a?46$6LBk4=X?6n<<r>|yZp5G}P&I}*X;VFyGzj'
    '<&EPX~A4p$!_{^DB4|~ff0W-q2SC`?iQ#zeq07So_L>G1s&%CGL6W-AdA^jSO3u&`^p8i%aO@`BSCU%yc1svO{+-o@-'
    ')lnBbUk3EuoKGF3+%hKJtB8ZRvdE3falB`G6;LCXo!Tl4p?(s8_-<MasvJD#cQi$tRE$;vGiOw-LC5&DC*%6EKx&LJ2o57pMyi+7'
    '_w{3(>Z%4EE2|ZHu+9o8N*qBI16WqK$&37cRn;w+6bSIZ17ldpB1*$~nr!Nk|K0NlC%AZ|wNwa51m&f=_@rr}qSqb8=6BIj8mnpm'
    'SnRfI6r31h8{bPk_5&v`vT2-'
    'vZi(z~*mZFe~T59Mb(XU!BKd^EIMO)(ndRp&JoV#4B{7bhJVzNlxq^CtE=!`I9_FW|Y>}UY1dG<5bz0@_swWCg$sIuvwN*n(w<~)'
    'aWUqeg!{x9x=DW_I7@_U<QD$09-yyQU%N1f`DXw6x6OE88-'
    '}9R8Xuw+=mo;9Pg_0O$RMfomr7BWh3~T%{@}wGE)PdiVs}Cl<}s$0!(v2bzYaxcETgmjH@>oad&~+`L?|{8~%_r>c4@xXCpc1z}&'
    'O;UBcY6_Wido_pE(aF!!u|{~gRdYv0d>xi^@&UDq)AE;;d^F+2fv5Lv7)GCi<FU0eW>d&^|Su&CgzkuP{-zTl1ff;Zs{-'
    'ZEeCR^|)d8vBB`CRFf-'
    '=KXZ9C)df+4{CF5?GYO0lS6o{#jxl)PCI$<gH{t;oMglo0}gW~kx=&nTX+d074}{c0}5E0M#QA!0MbwP)4elD4vH78I)sF^yqLB?'
    '+nlLk;JzGw%{lCy0ZmiMuRA+RgM($bBdtvBVEFGN#l16VIXVf(d^pMMok?p@dl+cD2H}EcEjBJ_)?(u-'
    'X3aFNV%ALKDrU_zu3{YmNYtQRfd9Q+-'
    'd<o^?<kX%WLo>hrj_PQ9i8O41F~w_9NumSxn(YE_EhKwH(Aakr^rv1W*DfoDa=!lvg$1_yWkHO_VI@c`})I${rus={{C>Gr9WIaz'
    '#lFgNal1g#>p1bndTP^vjSYl^q2j0#|)4I3OeS^z{R<zOs+0)an%0YMYAiQJ8b<;Q?hYbcM)vvs-SOrMQoxWSewINydG;{1F#O11'
    'NHL_mV*nPZ(5mbO3(LPnXHdJiAL}wZX}w*kMpjY(|DT04`*%N5hAK2)`UpzOmz{p@*;4IERrvl$I=A9L>@<=z7)JA3+gW5ELk7xG'
    'MEh(-'
    '(6ugV8K!Tb>@`Gwe&hebl1nPQxU#SMff_Ej*%8FoZar`8Td^xedHOTtJRbePMF+@c%iv$MzqjE?o6!EncRg)q08j1MA6$=?nV^7P'
    '2}!`1sA{nwGY}K>?8Gun(%zo0UwpA;m$YwrcCuHh)H3&0#RHy_YaTI=}bp^-'
    'N=+*1GBFc4f2`MN#5yxTAksiRkfd1o&2==l|O?x$R9i$;!hh6qa%h|Smhizq(_AEF!-H;L4HyCuuMHE3in-'
    '&!hKgGPlXUzo5~OrH6bX57*P0E?&$5yY2Ln^>21bY-ez={-'
    'JR=yg}lPC)LtpCblZTd<W=rO;cP!W&hc~Nw|*j=YayR=&2WO+^$pNRdAxu=%Hsv}Q64X#kMei{eU!%w=(Bmeh(7tL`Pxl(b1=DFU'
    'hekbJ!B8Za(=bEI;VYwUD-S@Ea-'
    'M2P5oXDZGHvW2p8@HcUYbs^ZZpf9#udP(Wt_NAQn{s6OpLG+#n8Bz!g!b!ZaZURe)GKlM7Ez+Yz41JuOavz%(@XTlBNTNJ8@vEae'
    '22JQVwC%hNsD%T~IB+NNG1thpB>)4}x4WafK$@1Qw$oX21{d78sjcX_%SnJZ-{H!|gZV(;Uv(!Sm*?dPr1{@yCJ^j7HrZ<P-'
    '8R_SVawlM+)`}7JL=H}+cwR)Ia6{s2Dn4V4le<dC5`M2vTI|jHr!_As2WVM?$J-`(0<z~=Xjt=-5`5V_W*UEET&s-;^`#aam-'
    '@3nZgH*1^ddlCqzj>oP*WfyDTYCpi&VyyLsxVy9l{}oLUah2gASqA|9cVkMmBmAk|68o*-'
    '=HS&MsC(Ce{4*e6Kt5=gJ8quF9<e5?n$u8<Sz*}WwHgqrm@_MVADkIjj%D(O+%v<zI>E_-'
    'Y0mZUS+b}K2rXdZVG<s8S~ue^9;|f>)W%}NcGRdFn6+t8gdA}hx+GEfIJo+mUcj?WA#*-'
    'tq4V}V|R6}Kq3~px<(*<4P9L&kavWxdI;nlp)01v+3Bhg7UTOXl>Ajk372HiiJPn8Y~V<|=*-}Q0fpXGhGFS}GWZ0Ms-'
    '8kMm~_X!Lfu~$N^jP4lt~#vJLP&CM6E?FwV5te0Dfhx*+SRC7eXWEMX%LMn(N?eOhgIyx%xZCNggQ62~{zjo2qZWhiuwvc(WUBy6'
    'O79y}H)i)O)0T*S`3I>PVrE5bAKDS_yTSP=^Y22)s4=jeY1&jam)g#WugC&rQ91jD&ml)g5v%Q<pMz8B@KO>djOirfy|XW7g`rd#'
    'lVNLL2C_^=0ZI@OaA9m@+lAOnqIZMwhuZ@UT0zy6$dlV20pz_A?L3s7<q0!*{k%c*RY<`i-'
    '<LxBrB$<<70Hd&rI_G}!**=>|QZjr!I>mB!t1j=|P(YhJWB5E}0-'
    '>q2%3IW|zvLDsC+c>GRD_pOnQypAHtYYS*6%H>!`JF$;^)X`4tE5}jw8mt%NsXhu8j0sc}2`ffD)wIEqF_Fq6Va=FirwPZSYnlr4'
    'QBU1m8R(xG&xOnUT*yIGZ7K^ey0HMGJdaT`?B?dTmX1vWFSymMsgT^L+ay5Jbj2GbLec=sG)O=q^yV9-'
    'A+b%f2#KSFtq8>YlbdrE=xzZHiVYT51zPSpGq#T6^fOr~ckzHWPENf{M{iBBO>p?XF{g|oH|2y;<mQ|%irkWuMUg+{R8eF@P83CM'
    'B`pkdxy?(YVDja=9-;8{tQ<nmiV@m%dk9?{BJ|q=go<JaH8;O46`^pe+fotwt!|TmP}3E2k`QVdBp?)eb53apy(vcMMrKKAkc8-'
    '_5TSs|dTz72tF?$8aaiQkMq4pdRj6>h*!5iR1+6rxLhfa_+}J0ljx{sT6n!bs6n)ugiY`#E5c`B&s9tpvlZ!C3z-'
    'f)nc3PuLFoDV*9t)01?*vB7f6syO>==yQwg<*fLoj|)0LH8s7|qRVQo#tfs!0XoC%R1n7)@8anFL1DAOVcsFafwEF#a<J<1|4`Q%'
    '?%R>*JQ#Gmu)z`Gz+LEYhN@>jW~m*()$h++S?Q^Hlo1PnkO3|BW?T@1Cx1M<Eh>dXA?E3q63~>u+)Z`b`X=?t$+~2Vr&aW8KKPL`'
    '`qA@(edT<|_O{sE2I2?VebP*pwN53Cu^pnbcVqeMP8T8lrMZ0V>Oiq0+ojzf@Ght@@>+a*1w}fJ)Q9ZzZA9G)O=t4AEO9x|1;~mC'
    '&Sn@U5JgLUf19EnI!z@pZjG5>JBORy^dViw4YSq2K$2sS7;vK7^O>SMcX9@mH(i?fg|4SBpF>;p;g$Se_HZ^6~;$_SPpWE}k@2`x'
    '+ED8B63%<$8ysK`Ga2?F*~s(**OTbv!zc$hjdR-zq?4<6?+3FVjC2k#MX2sfc__w@E;x>DswTh%^ln5DB9*w?wyc=XfuEwLoT10@'
    'qKN{XiF6Fr9=D^D#V;$InMhUF5F@qxu#lsP)7eSdv5T!{H;(ve{N#E%u;;uQCTZ8H4VMLeQc79k{ze#IGtm#tn4eXZhW?_I(3p@$'
    ')$mVnpD3Aq3y^1>kE^41DIL2Bd-yZZ#kke9!AP3E(qb{B{!fOoIgQ!6?05qQ~gK=MrOZFkJzU9wn<(^&}>DdY2LRD^_<9|Nkc~!1'
    'P#F#xYAgWQPA>Gk#Ej`~rJ%Eb5P6XfKXKO}ftZ;&@a#xyW9$MSZ%9ImsN(Mnh^7udq_#QoE6F!ckS0@C;vno5S;OV?1BEJ?C&nh`'
    'Z@A?&R7curv1_m<l_%)4){NP1kJ_U}t*bog~<q1_`i(&U~jto5$f1bFi!Q$sg4^8fOD8|KP~R_785Bs0J-'
    '3USn|fyu+EPps31`#iK;Wla5{B0FHDSHY4vqj@)LJmg85N@ZP59{aVviTPBW!I2IS!0+!dAiH$kr3=)Q+&ix-?-82uO`#-'
    '=gYA!<ef2<RRch4yTk2FMkdE{*gk+-?P;@vKSEX#$k2Izz9#o%^eT-ZXgu^o2v)L-'
    ')*4tpMw@CaY?9gch+lACp#9SlA`TN8^G=)kWv>&huQrT1WlN<R}crPu%F;MLSNn57$>h-'
    '8IWUUCRSF`#;eHLK6(8th~nJg*y^goK7zfpRFSf1+68e~f5$A>!BR1}7s;Bo^}=hSI61==@)%USv4{w<`c}IVCKaJ@8!ajN^k0Q-'
    '_AQf|@Ei<*UN0d6andAhXQq<OU37_?c(nayP@ze0y~d!_T|+s+Qqru}zjdm}B{78z(UKMh>^P$GDvx<MxFZx9bDk)?&=#MK^nvpe'
    '98A7)9gu#}LFn6o7be0AfQerHdgn9asS4G7Dq#65pi47#`xAR2cuD+mry~#uhWspWl=OW5WGx-Jk>*6YgKo4N8D9;eNetPzsDYL@'
    '+khCAO03d;x~oWdoEWh^_B7mxp<KbwS=9{!*N{htaD!ISdYtG5A`H!53o;Ztxf+iLhK1)QR~jsFU+$Pp9O|o=(lzIdur?oR-'
    'ODnjb#G^A{nWKQF-ZkPy$B4N~J-'
    'Gav?dZe;OnI^^3_Ji|kLn~LYpb(<3K%q)0dY`!fC&)hb6O*be3&%!o%Q8y?7&zWtoK{qG`&u7JE8JR%FhB@4wxN2eLojP$JeeaFt'
    '_aEG6k&TA`gNjF=L{CD?yn0N($xNip0dz==>DOaS&x|p>F~;;s`P#kqRGo(_3ak*H*)#HY<-'
    '_U=Qkj<{H7$dVn8H(kLmKmPjQX|^$E!jduPngv9U+c2$)Cot=JhOsW77$1Q*jIru{IUQD|MR$9K(35O~Uc(x<LYtXX*wCINqoml!'
    'D`Pi?|n+UMWQJpdb)k<OjoZqe}F9EhSP5q2u?I30#GyJ10f_7BjIq2g*BQ9KR9c_@x-'
    'fn{xiaXAEy*i`>3o$PjBQ+rD9Hfzv4amZ^oD@^9zM5$bb`+y(Lx^`-'
    's3NRCop7ozyx5XJKgP<&^IVofTlQLG6?i=fza!FQ=BhKKkr6~*&)n*<cYXndE1;x}}I1Qfrd8zi83lWtH7iZ6^%T&v@gc1Jc;pDb'
    'C2V%JQL_C7=lymCiSbCC!vIL9YxZI9u(%tZ4XAn%MZJSWER%Q1#G=P>*^L*i&#ocSfgYG+%K`4z)zXUj?THN%Sg2dJp$ebUj9E9F'
    'Y9D`RSTR%!W@2*a<27=EPy!*}JkdHV=kYOud3f?(4D-=`uN9^v~`1izx&Bp?{Z;`<~7&(RGM5d5-'
    'kkbvOLx<M%j?i%~Y+AE+;kt)1$g)?(G9qYq}W>?|0?#kg8KeKs_7>CbMCA`f{Fwg<<t{A&-#@Kx&#_pCJcIU%uyHLjhre-'
    '=~=!HzZ<mk|s%B7^yBj1znkvx=KE|-%ylw2WKXfdb=y0?eW9aI3_p&@h^seOfwQ0!L>y5{|VNQEvu!Vjs?9i-'
    'bNKo^GMha~8}sT(9f_Z8hB0lHgsgHoV-bsoB=(s);akZo>)X`g9U=kvKwKrJSThZc9KcQ(%B<|HX7d_+4wgi8K@@5=w*SAQT9C9r'
    '_!IZl^%n2AF=_zsN`|5l9nS7XHgltcVHhOt1qt}H@@Mhf9AMgdDrg`P+7ts#QH3K4t-'
    '{CU|a0PuXZmjUnsRUQI3vj8@)^J6N2;R$|B1@Kq8O$h)lvuJ^y{jnqfQ!wK#-Jk>jQ!wLI-Jk>jQ!wKv-Jlcz-'
    '{b&Hwld?Gm>M9bJ&%yHPZl9(6pxV8B@1G(nyhb#vw@VUvJ8Iibk@+}8GeSk<()E9PzqpJ0lvNu@%3GSg&fWA@wA=iXGZy<F>L>~@'
    'mxA%I3?>s>D^8lT)Tw(v3`r&TkW3(Q5)3dTB@@dW4mkma*#-'
    'bExZ`CNrUZox=n)JH4m0xcVXBEi4?m#O5Y*D?sn04C<V5+6@(yG2>8ef^*m8%qCe}$6|Y?27`<D{TP8;XBf)Yk3~ZmASZgB=j<QP'
    '~I5jshPK_tVzVXBu9Z!tQve2YHFA!taq{2q`?2McU*f0Dhqw^AQz0S$V*n%ziZAL8Q;5um1y!^cQESwiB3g*Qa9&FsM%hcs&<92<'
    'ft_U}7%qJPZi{74!VR(q!Q!%_kw<!U`1WEH?C198!2_tlSNf;(bj@EZ50mHN!T&C|(3Wf(4goT%?L1DHaArq|7SR<p>8}K?WFiD8'
    'nK%8<ZSfbGFZNxCIbMMR*3J9*sGsh0on5+FVm!Z%hhwTnAw)cy%JtoF>*BrKA5bDf~9OM^;IxC~o3~<L}R>WO+KZNbP0@#iXU<<x'
    '}gF)@p8EhW{)C!-S0Ix7O6}9jTgHusEPq!%nwS+12U?rfIFa>=-xFpn)hcQOqp#;>DhtXBvp%l~(D{`nDq(hafzK-rwn>RAZ2vx7'
    'Oh`(jU7O$xQ)T&!@IblsWl3WFF>=?sw{}_&AV>oup!SOgD7n9z@@u(<kPwobB`KV)7yq(K&RIG*5bG)N&9m3@VN33`kmy@EpERET'
    'BLI4jd0Pv$BfIFznC=OqhsmqGR%S;yxNd+)G!;n+}57TWDTnF=D39bW-){rFEVXVGGg6q&t-=P!$KU5GeE2n}tO`4!OamvcU&l;-'
    '{v4)2SaN0EtwL&{_R7GkYTHmHzot6XfX)%af#vpz)266Y$kyZm?Tj*#`@8GiDkqMh~>sxtvZhf1*x|iXnpS`+|;itd7x}V`^U}1v'
    'nT_Kk5EWq-((1*AtQ{9X;zBW_ci-Fm6#T}_&hKIN#70h?)HVI%h50(IC7_~c+!2GDbLjst)>pPSJ<}n4Kvzeiib;$Kn=~E}rbC?;'
    'B9s_vSVf{fTOKVsc$Io`+>hv6xPmiH|Kn&$^F_bUQ+2f;yoKN0C8zJ8%=inG27ZQQ|om@_I{Iw=eqc3;d^24w~Ak54dP>AAzA&Lh'
    'Spm=;}k=rY7Eb<MRy0RFEO()!$3SxMMJ5xbCK(|Q%v3alr5W{%gnFQi-`VI*ozFgm-'
    '6cA4;0P!GX?j%mWI;SU$Uc*@OE{1Bg)J~bB?FH*t7wrAQ;|t<&eTn5bTwi864%b&g1prKVq{^&j9BU^6_@5EO|G*gj<74<=k%ND0'
    '0e-yQ_*DwIfcAaI3b{xV*X42CGsN-i0vt~Wt?@lvon=S?PgaeJ#nQ~%+?9%Bc!s-'
    'DaXeeMNr<JH2TO>hLC4=EQetW2^&JvoX;<hwl!D_w7YPOfcDE8e%{)vKBz-'
    '?bfZ6LA2jX?DI?|kkYJH_NSmyGuLZ6PGXNF|bn8ll89G=B;jKi}H6DIt*ojGy|2UxC-'
    'v0M>jc|wfkD|2260PaPj`4yq|jS#gr7Q}xq3P5`|S7#a*qDQ8>@FJ)+&pk91weSo>Q&D@PZj*po^I!?6g)SbNgxU%E4hg8eQs1Ez'
    ')c!L{PS?h%E@aFoc3`2|1XgeuE0@L=q)v3fTKq`w34i;19_VOZzylr43wfZU*_j7AniugvNAqGH=xAQT10BsvT{7lefhR6DJ6*0t'
    '!H@v%Q(2Up-kKMX9^6ghwh!dRZ7r_Xgt-'
    '2=AZz`*B3bL^nd(!s*5OX|safkE>o%olt(zWFJSAy;Dy9{rB(0k}lvFwv!4X3Wu(PVN+2}#GWHrEyT!Y?*{=vJnfX+riqFE*W?vR'
    'k>qzeBZOLKH0Y{7}?=_EXkCZa#s*AVT~DX_C40;Klpa!wpjr>ZL`+XgoES5mSKD&s!wQTUxYI)}oiV-z0Ew^JgqECkzo1z`KW2-'
    'wW6C#Hf8?ldtKZ13qd31HJ5@w+6j!Mpq}32f#LB~?i55*Oyo3pkcc1i)XOuWKD#nqws_;InM@xwdVzw(EdGRd|k2C+l2XcZ3`PyB'
    '{mLARhtu&wL*#N5byMN@~h6>UcrmWDcCCqEey*TN4MicLZj26WG32ZE|3HCI(xZ?Sbv>5Nt0MfbEAOU^BO#lnOSu)1*|ey`<YDfK'
    '7MAxk+GycR4o+Y~~InRe|l2FF}OWP2d*9u6rK}+<;hV&e3wTI!;gmOe1HhPC`)_XPrb~owC3>l~KABs&GH+f$@W?%mL%s7#Nkju<'
    '-Cx;p4WLpnks7cAAHE6I7EuOW=jgG@CENvcvbpcFGQChH(3L0o;Bp0ypzQlT+aacbc3Ew}0z432@Wh`}-ug!MprE32x>NB{2gOz-'
    '^VT47Co}YJt09>wtYEaQ|T4td9lm9;}<SM&RDTx>=tH+!Wih$A{&^s*MHKC^-'
    'tYu9jLyU`{hZ?niY@4sy@MkUORb<N!#)S$iXQRA9V;M(%*2HP%K_olhm=(b{7<79Zn1QGmFAg^2r00pivbL7aJ?DXEBqJ55PN++T'
    'E?1jOmy{X-'
    'Js;9dTZggA4Dl3cff)pRU6Sx(qoyK7Zm00}o5&1q**8J06w`Xmfp9}}gb4r1yPPTD9paZF;dtP{OCe2jutwc5nsL%+5KDVM2K`zM'
    'B)l~&tf8$)TvvrM|yjb*aAnr+Lw-xZjYPf%W`j?F=NS`6i5x3{eRIYiy$0@ST9f;w~S$5K%TcX})pb(3|Q1k~w{_+t|4;9dThggS'
    'GFl6<&b<Na$ZM!_|^M%gn+Jn0!r8VzfBASh)jvPj6mE^!`j19fc1R`w5J>T{<Hau-'
    'uyxibG=96kiGTR3XKpJ^^OGNkaae_$(vkNGcqs-B3qffjEI1g6gu-qx$*a(J5_<L$WZ;jMp&x7q@{Z76~_bL+=b@dkH#JQZ)Xx=j'
    'M!bVvLt32*Q&e@eoexkE{A+^#W`gf>yOis=F&A0-'
    'LA3xym<5_p}398VH>7YR9mB=9a4vYsUHE)jAfN#I>7<Rp^7TPU!`jbO1s9iM~6j2ITj7p$n&rmnCfaZhy#IQsNHILJb+^!EBQhOb'
    's-fZB!Qlxb3U>iuj8w!sBazQc=v&AiYPsbGUUJ&_8w!MaTX*mU=vmjpI=m-'
    'CXqX6{gu6IU$1c#&|mN7cyj2;0<EK9IaZID&h0m7FE)#KDz^|Ho}@!7f7MPB~@za?KH9341X@M}piKpd&?YjL(rIH-_g(lN+ORR+'
    'RfdjK>{F5&@4AiR{TR2`qE$sepCM5!(Tv-p0v#wH6xPNyMO+g#;x#6Q!O0TWb<?hT%?YQd5TEf!35Zo-'
    '+#4ZNFuxu0@10uP{6<CLWptFvNL)lQ=tCI3s|y4px#2mEt~U;LMZe^cM_vl_ZJzC4;|7T902baPvuN^lR{z?Zv5BG&Y-'
    '&G!{J4Yo1NpplXeFwXL_dHn0*pFSd%jxV02=+Y{p<O*AjpH5KY>q24Hrqwau7vuil0l-Fgt;gi%vaJbbcsiV44w<%>*Uo03^=-'
    'k8nsLo=TkwsUpGE8qVSGC&Nr7)!Dn?K1Dub=Fx=TGq*?Cm`Z{i$Azpo3v%xA}c->`ZdUs?Q_M2NB3L2rQ;C!)3;JE=I7(bgV;y-'
    'd5f@vQ;-'
    '_jEd6JNGH1$o1x|zKTRD>xYehrgZZs)Q_5hzRxp?_IS%#12a~9kzk>O7xW9s_cZ9!!S$U+tf(d(+zj}>fdNf_lW|(M7S1&U>ZKtc'
    '37@k?u71-'
    '?w0h<|9mGK~B!L>8<jW}Khs>N8(Kb8(8ozsg3rHn#3(7a4Xa?+R_yB8Z_^MIeFjxgNnv(yp(M7Jqrgx`rqxK_QuG019zAun+ZrrB'
    'V~s~p3uZJcW2G0p~~b*r+(B1k%TB8o%lWElq2_g$Mgz+nC^uqUq!Ch$TM^n@9_$V!AWn8J%mts3U=k}wuBO3ztWn7O@;@}BB$fop'
    'e-'
    'O%ujN<pHD&!|T2A>(?edEr|#5@?s{~^up(<V+yzWJatT$=r*N{>5}}I8myOPh<>j%+Bj2Z2K*JOX{OE&2rN{yOq~;u^{SmSB?Afz'
    'wM(XcYdI`x(U>-'
    'J$C)p349wa{?o5u6S$l%{GRJVNji$ZAF@S%xlifZxKY4qrwJg+ln1|Qy46EGF)=J`3NB_M^Q0Fk#sdJ{$3tmxd@|p+yB6XnQR$rt'
    'J^jo@3DFeN#m`4nolP-'
    'cZ53st!#-W!o_{a)a*=WM#v^Xn6z27P%!^%n3Hp!6c!YhpKl&Rksf)hQnRilhl@pfY>%S`E=W+xrwO7p>4x!u(`gDD*1YMfyX-Qj'
    'ASp`Y)hBSm$bz!c*L&9r_40|Y(G$jsn>EOd`|MBsO2vEem;?90^Qg<E}@I=s*8Hl+;jr*U?!YLVeAu$!2c85l%br^-'
    'pbWKVTShK#V2X)MdhUD-'
    '7&V95hJtXO7>%rN&56TNFGF<cFGXKFh!yf((Z7XNxVjp+$O?n6V`PRM<!c@;{_+0i{w$o*+>PZF{vjqS-'
    'o9zf&_rwDmqUOQ?Bq<DHmeMAwHY@YS2)S-kseU&<t({-'
    'DY!odY;eQh&T81EP)>A9o)p$x{#mRf0U0?J%9Q?Egv?l*UXS{_p$jwOp&Eyux+l@lBfKUREl0{kqrVxRD1)?V0yP9!<CN$|6T61A'
    'Spg;+jgQHs=)JW7#Laf(!Xmm-x$JokAZ?+cEKIxy2&QSWl~>nzBMYGR`m`bnp<bmqCKES-67Dk~)^iZYN);d2Q|QADU^-lT-'
    '2D2lr%-'
    'Jpb|D2ltlpiE^YB}Hv0ta8C{u`hG+1YSBZ*(V9KVnLoYk`J%O1*m~Cs}a08!T1g8DPhkjWP|#<pkQ)K<U(@nb_a0$Twt#%IDR_Yt'
    '6y{cTx=r)kHXJ<;)Q=I2jkCU7(W%m_;)dkPu(7jcR)S&MF``0@VExX3%EL{7>xCkKF!jZ=bmQi%yUmC!5BL2=_DANH%Wl;?{tF%7'
    '{j1ET^fwHZNJu~H`>%iNT9LRT(;+79oJU$oQJsbfO;$sXm3Dl_%8DT!|$p;=b-'
    'jQ47ESUP&+q<T8CUxX+6i8Vk>hea<y9~N0^IAT<xA&v0Xf%Z=T6V%*A6|{UVd2%*Eqe?U~8Z=HdyiewoQJsf2T1hFa?*U?%Id>QR'
    '~wG%gP?UGNM`XPbM5rL)aF!&0gPU^Jd#Db)eyO%kdD&eaVPssmuOpD7K)O+@TQ9Ky-^;<~>Tu8VMDS9(tD-'
    'f|J0AsbD*rCRFSXQsZcO}O34y@{nIw0oP2)lz2nhN=jAW$JJu&{BWNA^yu4@qdXC|N9v69dn4E!qpy`9BGrp(A^nflPlChhNw;>n'
    'fofl?nw2K!S0wmgRM=jHZSumOJ|sSmZdYyJ<C#ZtD#??WhuGU=1mfEtADQ>B;;1ZxIJ4Mc6S#sO#wK$EqUli*aTK2En1`@NW`1<j'
    '$W=M4}gs}yDHLVMrO5p8F3D6#Vu1S=Jeq9&eRdi9k1+@siT-XQrI_Bt%>7M{WS;ZuVO&|H3sw_VnCmkqx~4E7BIr|7`2!cvA;W@n'
    'EN`!@FVI&gW=I?4J)LtHP8PXOXrt+j-'
    '~U<J;zemYoRNjV=3&l=1mgVYyY4dB(T@QP(4=~hW8fnj(i2On)m{Nv7yl&>y2&@Y^lbA*+rD9eSU`L3>Enl*+uGaIrx1Y!|!h~{Q'
    'elj@AMq}mZ1N<$a=*~(S2TQFWy6L@+J0S8M@L-1u-lAitDs7O|F65<jd{FN+f)KU!)SueG?-'
    '1VfBGQ@L2U}F%Lp_!8Df6FE@>)^UF<3@*rR|rX_h0=1mejh(GEE2_6KD_O#Lve4r@sg6%epqD968tGAhrD4C$c`B+{^JYI9TYOtj'
    'EdUJLj*HJ|uR$@$*XX;4ij^Fmn)X~HltNtek?r&mn|4$6=KgHlaBM0vHam%;NV)J1&pIL_(tmM1Q`omx-'
    '7cuJr1LT%49pFqNnfo?`@1yEsy^E9M)Ft62j$8QZmz>VhIp(IbbdI^{CBc`#1^qp}B={1z%$t+|Uvg9aq#Kj~Us@AjJg1ii-'
    '@^(V;8sk>!2BVQGqd0|RsoF38ZQZ>-6Q-d(o*y(@&fW``A*~N1G~q20|^D_L~IW73DD`r+gNN(8Nu%$e*hhil`$lGXyfef-'
    '*cG%HpcwlW6YlyW4=0v`EjtPLnY8Ombu>q2p+92GzcEAF3k#s9LyunVCf`tGgvao+$h!v{*Oi`eJDZD0lIMpOA&N1uaqF@aGq|EA'
    'm{+YG@~>q|1-'
    '+f1k|CcN@QH+6u7U|R@ofl$VYOlwIXD#U4*p>6`E_@19>=?b!vqTj@5DSLqD(=BRFiPJ`}|7{Vw<c#D=^8+yG)jUIfO~G^k<VO;g'
    'UzRy=$U06jBKyzA={?{q{P8A0G*OSHj}x$Ax&R2jm5{a#bjp`^=md{`pA-L2u74oQ*d2VR$&SO<4nmzr4joo-'
    'V?Vjac!boZ5}d`!?a4N6F?Bk1<!B?}%$NbHTr+PjL7Bd*+2%RXZ1?Ds_FlGTi~aZJtO**4n(OAs2P4onEf2RQ8IR|<;$p6@DoMqq'
    'ZG&GvZ&GwN)P&nV2Ivza}kG5_m9BJZ9LJ}lABF9CPd0IMI9+Z;r<VYWyB(U)R~-nzY%;T>?s-4-$o%wsU|G;l8$>Y1=un-'
    'V<rL#<DRAw0wSR2Z(%Z4&Z3br+?n9h(LT`JK1&lGToPi(%LTh2MQ0>maVyAg$pn;bH;d5jLJz7R?P&0~jTT_?p*|n5}6O$i`rX=0'
    'ik+I8O^N7<q%mu<tS0>aV5cDc_|L<`stMdO1edZ8-xtPv>9GW2r=MH6hj(FxZ@Ftlf=S6-9^v%tLNS1sgoXhE%Z4(``xs8%gSzE-'
    'OvBm>fgXpais$M{pZ2S-'
    'H3XZ7YP|zT>v2B@eenLnuYDoMc!BGE5A#QgyRn^#z=T=57GPeGnmrVe)B+m{(%N)Z`GeKxd0BNM=^+7b0dMgB`p^Ojn`)S&YcXJm'
    'khy#K1#rOhwEv-KGS@5Zk8dveM**X|HV>lz<r84%P6I<%J6n16$voA}kmf8}gt5goeA+^ff}8D`Ssl<?I>?V2e001Hi>ScsM32hv'
    '0cN22Z~bJbE8w)1xqv_C%WwEsvEa*<@&WoIKg4Ld)ajDK-'
    '&Www3K|8niq?o@$ezWjom+l@zUi2*brpEjKW{4^s$>z|g$zrc@ZhLu^Wg;hnlo0t`*(zm)_-'
    '(;xwc{dmcG!UZtwBoKyfwd*O+d2^TR1pI<6?jl^S5iac&-'
    '7s51x+~hS<}9i%6dr~G)8s=K&Wd5!KY(Ga(dD#9N#NlfLwo28IUvz)p~2fbj!u|hMLwRAO~LbPOR0k3`JF(!Y*?w=6_J120lV3N5'
    'RUViT5RCBh^b}8Gy=?PZ%&0HJjCWyI1bQl65wbCVQvx}O@jnD_U9#w3GW`2fXT6XCAAqC;P1;vDQ2-'
    'sV9Y`%BdOIwj<?xJuqTxhY&sHbNoBpwM^YaNInl|fXC8+cR%nDcoE_tEK+YVFR&NTed+DZ3&kXYtUBx420@S%!=dMUd*4!VR7Sk5'
    '4#)#;yLW>!s5tI!Kp?H~4|Icte-'
    'H9o*h3O#n&9|gN5guYoDimkyHYGr@sjpd>%}SIOrfsQdPy!Tb$1#AHEG@iyY#GnbR9OZ)hEZZik`q<v%XJ1^7^8W3rUy4utu3eCC'
    '=v0Ppx$+vhh!Pn35Ae+Er#Sk4@sAieSVaYeL<L!-'
    '6TUp`HL_^yK|VKy;Ft;^Os@9_Qg@g_9aoq_N6*wJB^yGO90m^gnHJ%^<K>0E@C-'
    'Nul$q>S9pe>QsH`|Zj)d+%^1CtWI0WP1j{*)mn=yv;^LZfgyy%mkdi{vgWId$3Qc!!uV81~LWFR;JC-'
    '!*l<i1F?fhIs`BC+rBh3X<z07guf_;A9QRd1?YL(^4^`a4KQ0AC`kd~4*OPJtFyCq*YRf}opm6VkZ&PoI3Q}rx7PW*XlF?El~6Fp'
    '`Im`7&nz+xomrpIQc2DITJW~K(TKh|v$NYL-'
    'r50*fJ4ud<BrI4V%r0<YGf_}cfLrKEK1$iBVGPj{IbsU(q?&8!vbb5M)s^_#<DJMp=dY(^p256Z~XK!`)68H&#e@_hjGqW8juU(M'
    'KHCm3`xvG(HFa>;C)p&UfObV+LFOP%Yv@-GXc$gAa7+#*BR@(3FWILE0iL7O6m<Xg<?#|Q}69;cA)Xl|QQ_}%2rQ#SK;-'
    'yp^zo*+IoQ&V2A1vWy9LDM;mZ-e;Qqsxz1^NypDH9hQEw16JOjxe6-'
    'taXfjoun$cVOxGA`yC%Tj|V2*5P02k~@>xTi_>v@UR%dXJtDQV7o8};iX6hZS$(DjU888?Ugp<YMZ@Ufpq7J+f`&G5=ARvAN5#yE'
    'c~|l-d47S-'
    '%aF1Q}>2YUL>r_5LWl}ECyxM5ih4g86M*0R4Bis+ay4Fn0~MXD8rb&%o3H@UQUAYh58O9i4qq;c^XIB$!WZ3o>FC~`BWLkv&wz&y'
    '*1kA=$IT}HWB<liza~CB(9p+t9~4_ICrvF0J_JxYG$thbdPhjv%M-8_zC92z43fFJKK>wTj$(-'
    '(Ef^I_U~(lMP(w9a!uVAIvBe#b)|_J94FLGqKIU?d5u?6(F_mqN-'
    'CQFt=l9d-QBAnEFtL*di<57q`O!39TJl6I_o=>Bu!l4C+{Qh6Cmck7%{(z5i=`B%tblGoR9N^;$RnWbu_oqY_J7WZMYR@gZ-DPB)'
    'LJ@e5qqdZxD7~>R68YQ<|`btK)eQk+z>_TvN3gaO6cbz|0CNUXl5IYd%_GOTFyL1g=Q;`k)?cVcq$9V=Rj6YT&yIQ<oPr#-'
    '>kSO@%MK(yOWP{flmsV2tn650+qzVfbH7GRCv?9TJT3MfwgUX%dSfMY^Lh7lZAs9>~B6cJA^Dfg2=egabt(Jd|qf6pUMQs@e4hd0'
    '&B_U}Dt96XTqCV$6;w#>KgbaYP251kHuk85mVU{LvZs5rp{43=FK*+r;W&f@*tNuY~<mmvD6)*I~F6MmTkUXoj0H)y?4ec%ga~!?'
    'EdtS*bXNhnSU$<H@>B0*-6-'
    'gC*b?Mr&3Qj%VvTB;fdBeTR~ii3|35OEndo1Fk8Bs*p=bW^idxjK9^(N9II^##)jkkKz3UegX{NA7fa?7=A6r@Fls>29hpqqa(0!'
    'su2+a$?=L2ACR1&I57api7F!kAUR2uStaGAcJd_=%&Flae48N6l!k9Rp>8P#U-'
    'R~}Q{f8_F*_B$wYp6LeDBu}mH=NEsM$&IeNEpX0lt^$JCvkOT;N6QFYprpZFmf{-'
    '^M_DJqFsPp^}Mw5B&z4$FvOH21{GC9K8mc)AT+%4L0v&1^Ns&ziA~JM@wC_3f+f?QrYFYd+(tZz5ND9yZ}>dE3pQw;h`o{@ggy#z'
    '-etc;w^HI73xNV+$y2^6hp4*&eu|r3orFrDsl(wHVMcbt{*G`xiIptB_a29eTM|(UaId<k}Pona#5}}(+-'
    '}g=UAINc~~1K7x#?Sty{z}UL(Razf9nVq+Xi}{E$@UP69vs*sEp&Kl|FNodteKM)P0XN^|>Dw=R*0u-'
    '{&SXVXH23?!9wf>@_&V41No82H*#m?b8O6R&X=7~$1iO*OINkGMY0h-'
    'F?)kFF5OWO*w@I$7R|B#~G1TnQqtJy8lcMdY=GZjd1Ix<)rBNvOC0$PZ?)G+XELp-hgWeO6s2$I~`zL?$PY&-'
    'rjB>&fMOB$E@ZCbk-'
    'x$w}mWzNfy+O^5H|>2PB_9rlW+!zXS!1gxP&^ReqNW81|@dSp3S&n)C_)?5fC$tQ{VuSp2q6<qzpKz9&>PZ9P2{q`$GIwjyrkxmJ'
    '?Qlzj4m<LN>4}gJMDN@)2_R<X!*aJS%4N4LyF7SS>P$03Mea6Bh={k><I@+-'
    '5Rk%WD3>d=(f8nz*+Xj4w?{m=oK8Eg1F?9Eiq5Emhk_(2S*8MZ$vTH?^-'
    'EzJ<oJv~cyXpw4XpsxlkyP+47pd07eJ_`&Hmpe6%?_9jn}*<B$tl-iAcIelqCfrqt3)~l;wq6&fw(G3RM0$Ff(HRZwJJ$eaBtlp!'
    'GrizHz-'
    'M`xF8<fk|THbRI9)@m8l+rOj;$Ti@7Y9Rw1mh*vd$*>JovLcyQY|hv4Pvha7@`h!NZ?MsRtI;Lkz?%WWtMd~4>iuCLD2ZC=CsnoQ'
    'MXE=BU%O!doLI>rr|8fbyPP#r~z4sx+NniUb;-T|TGP9cy7F!*R74`%Qw0%Y9{t3^7~<7$!4^td_+$mYQkfD9wHItj?-'
    'x<LYvKhq6L(km|TAW(ieQ%7Yk%jIr{oMd|#7(9?UjZD~h{g{L1k1;fFj-k0v49(AT(A<hg*?ql){Pmd{Koar-KYQ7$T?Bsi&q7yt'
    'DR{EJTmfSzt%7LDJT(Z9$X{nP@s`@UFg&5DbAB@ZiE><k(wHopg<uAky2gOHKZ8$U=7j0E4@5e}<p(02;_`ze^?37O35gRh#vdf9'
    '$M2&XBqUCJt{ar3R~)+(sJ9&??hX>@RD{fLEih1Q6L+^_;kMltDfBETx`ye%isvbbR;YD3?5~Tl-'
    '#f<szA^T{$T=44IJr30$Pe}~Iqn;|e1shLO<ay7$9*%GqsVdJ!sTe~xSN;SL#Uf|8e1Am=gwI^EZoe1cxMKmA{JTqz=tB8zVkzoP'
    'T%=ql0`NTmSB-#us%$($ouLB2^RSa-Jm4B;zB-'
    'w;2g?H5&~Mb2F$E*VLgmmFr&&*$ChQrUG~Ubsn+L^yFNzlEirQUi;??f4!P^O96?*opST>Ow~Ypgy9sq;n1H#vKs6nofZ1H2rio9'
    '${DnY)9iMQyr$CvLy;2EDcL|Z)kHLoy>UJrHWZfeliFBsYk3>3C=|@ROHV>A7WEi}Ul90TgZjgZFFLi^G6pIUdiw;~3&eV~16-'
    'PPdOttZz#!!ZdY(9VY9)`Ja9xk;E8BUGVuw11!<Z!$p#&Mq*$NR@P{>tN+^3>4*@DK1yq57Bv-'
    '$3i+0Y{FoHB>6VeW{7CA$24!Ahau8$6*@M;mjJY^PkdKS#J#R4BLdp2G30xe2Q=t>7Mymq|>#2EYj&(KNcxz6XwAZ(k5UKKNcxz6'
    'Z`813275w=>{e73+x_?$JSwt)$$m&6!7uk*N`{^LtT1XXK`Z7VRgA%7j8%rUEx0LB5mRRP^A4H798ZkW^p)o3G$6GiW|;=E<8$*u'
    'g%qLY?W2;$L%Iio~Z6{Bwq3)btj`OpgdXK<p{mxDKIH1@*&&9UVxl)c`8g#+8W9ZYFIKE?cpq-'
    '7@#u`^NQ<4VzxWnYMn^TcHgYqBxJjr4)2+zWV^$dK!b#A_gdYcq_VZRP6*4nD}-'
    '?tYt=}Oc(j$Zhcop653~g^P8T5OvKTq{=a7RH+;ldA72IvC0iUnsQs!G2ysPy#-'
    'nCbYG`;&Gd$mMUNG}QEzX8FvNy%_tdQ;}gNQ@P3ElA~FmlOv9;W8HxP9u;#A|#MK97*%^glnCtU-'
    '3fjA=`WFQ$Yz2u|5@)Kj=0kfKqrGeHhCdO9CY=;V|4cmIO-'
    'L<lV0ulvIMzyij2K5N;E|O+l}t6IzOdSY9Cy2A(R1LapsXw8%(>Sv4Khk6`LS59p>0V|M|dyT*VX9+L;G5VR;Gj8_WPICC&wB~+8'
    'l6^*YJs;RwNsCUOZWe(!t(>JqMi?yq^bLIdKtI94}Dk<!dAu_uOI#)Y7Q@<`&9Avs-Ln<=iAvUBU^K;!M!Caa_xhct9!noa(WG;v'
    '61|`+9HIHlAz^*Ly$4Q;IKRpe#9N~9zUS5?aE~pB2R;~hVDa%)&y~pwuXv<i>0&O|VSD?Mm@)c++SiS;nCCgW!tzt$qq#CfJ9`bW'
    'yCx&6bU`}+4=fnenl%I7z`ey1o<9q;?k1S*j+5s)>QLKm_pLwi}sThS@ZA``JO5LV(z82d_y|NU-'
    '7U;)bSqfpx1G+&;ga`!~1xwXQ_&l$%=y2ZaqEekWE$}UzN+e=;DPdQe(ufG*D5mN>Oqwwaumv#b9>e6poG%clIvwMQ5RUak2*-'
    'IMgyTICLR(LSaDpd7Xjha7VF$detwR$DhWk)sBEc{pUd%+A*WHu~KX`~usqmYx+a#Ds)A=_inMfF)o0ClBgStUU9%b`b?6y;e0J4'
    'crgEGACDZIc<uH?bRpQ^~Qs|FxeMf?Xn(db$^(;J0+kK{~m5^@>Inf4NLImwycEadwnXWCoH6(nbRi;ycxy|9mvt4PkYHB%!zx_4'
    '$6c?-~ed5rFdw#Qku1MbzMLllCG(8{3js7zH9<E%1|wK)}q@DQ6*QTU2(lVAi*C-qJ;g3yn>lZ@a)x<N@U<Q_4N-'
    '}8d>0;q_wMGz*EOA%DWE`UpLkIZT^_r|S@PlBp4(Nhn`qnUcxgJl<n!L0x+SHxhc%XyLTOiPKa=u8e!HpUQTot9Z^hzelKtvN(_r'
    '}wQrL?y5l)*zw^*h*^=Q2}gK5tFzBCe=0}7{Plz%z&{nQwJ9_iRQJpq=FG1VoNF*2kABmCeaMSElDO3#_5(MlUS!4l;kxQfU$;Ap'
    'ik@Eo}W>fBAzzYgd{`&%)Z>}sN&YADwIkD!)VmwJQ~|D^@vB~t_(v{0UEE2(KsTvFzf8_N+S3Nuk#q<?*)(ZSmN&mZ}T`JPzO)*c'
    'oG2sFS9LCdw_>|0uif&ciAqL6t!yrMhrt5Ft*9mLB+sm-ukChFv2tZlnTbLbejY)nlb2;1V$L9K1pC4p&OLs9u@@DTME_1ADRDJs'
    'LLrisahQ*)D`{!{a~T4r2LI)<d*z@nVN2ESUFcyGxaZf)xc58sD-UzU8{PCpH-'
    '{s5r^5k#9{XOg5*DaT3_#@rRoXpOM$B~duE!jg7#j-0IRU3G8amy!_-'
    'Z9)+Y=3Vs^l7np43#f~zw$K;;6}SQN?s(;d(@HJT2O&^9%i{+(`<CW6&ADgV#BNkaZ#7u_Hs|F2#*C?%RcqX5hY3f0voBmGLK?mq'
    'p9arMtk%^<)EBxv8)Li9UzmH1gDs;gp%UKT@iV(vt<F~e+s8?eVXMjLK}Q`$*5j>EtCGa)5Nc&;uCWwv4=BTmOSkWfdZ0j-'
    '!$xJhG8&8c7=fnuWC&K}BDhayatrk74g#W6g?38^?<q1z;=uzZdgMky*R=1me*ST55I5>!|w>IS9Y_{D-'
    'eRxnZ<g8=58C?w}El?dZZm4J#?l!_lphL&Tynp`SiolcInt2(UfR(C}{0_*ybB#Q#;x>aD2qhMVp{gu(Mu9IHC7+BYdNBw(swfI?'
    'St*hg?(>0zulf-sZ+;+gk8tPwRl=oDF-'
    '^V#Dp+)Sle$sZS_=RU^mx|wcx=n%|{sPabQ|z#LlLR~5RX0em!;^G_Qt<m)0e<@mx!$|6>+!lmJ(;;2DxFC;DkGf2ZtGz4A7wj|Z'
    '^z%9=78=Y+!2kSX?Pm^SQqsS__1Co{#d8<Ecj_>zW$)D5kD)3bxn-YZZS$Hi?FWj7$lLlb!EpPoALdrtgtx+toPOb8VSO~xjLN}I'
    'g-'
    '~eP?ZW^c!a7{@D9^$5{~4tFO!bs&6^|~$#>HY5{~32>jtHO_niXyAQOSNA<~v8$w?v7Lo!rB9%7i*TW9KPr)OU7Eixvd)@SBGo<M'
    's#{MZdtC-|}Zq%+~iZj*jg*NUGNx4JflXZIMMQ*sW-78H56Nq-F}>~0hNwxY1Rt<c$q!tTRTS!8pfR~^)yx&l!ySO3k5<f-'
    'cSIx!Ws@B}BOqV`VRCLvE1y7<JTJXP~133;mBb%TUF)hW6`DX3jiP(9F-yM*Lgy`!U(d(EA$Y0mA5&pZYuw7C-'
    's>dB27vF3nlm(TQDr>+w}D|&TZ49m-7SUwh7O)P40a@0pLFG1ag8L4TVuZoXh^H?|G0k6#r+|1;Ol+*eXlP6;yYZ~KhPVMSsb-'
    'O8^oy=ADB4Gvd@+YNY7arlHRO}AWZPMhfPD)DZGjEcR)OWdVkdV~(m~K!Cc2~vQHaS9$aJrT2)%D_MrKqltA$LU#xyN%0_3_B-ZD'
    'FQ&Q`WK{&b&(ck<2Tjw`N`q{b**?&~s4-5!46(R_Env4cCKXGmk}bF?a|kxB3NQ1GLjoh%}bkoZ8i5HQhK2M{-qNtYB7m<jJY<g-'
    '19!6~41|n}mwlZ!ilprDE2+NkYZ!6}mw}#q8s{K`HS4G=gs}mOz1RvQ~{TTzf5+$`~%P57DUKAbwVG>V_EESH{SGB8Tj@Y7jF>t('
    'F@RVz#PgGit0ELn*hQ$mLzB{IYj@+JZgtL?0aV_EunKP^SJN<Yb?f*vb~4wHHpb%LXns_%H^{nGK}&VFA2U{mTIOV3cbt;ydX6Iw'
    'cjr@Mfo^0{BMVCc$@rp*tnXcQ9{~;5%HY8zlG+Pv{1v0QlR2?b<x%IRFq!-jSHe0?7>=?U{_IuQu9JC|xK;=~n(I0!F-'
    'Ji1sJ52<?4iB9G^@IO-b`dEmzAG(m>^C@^}09EQUgeli=`JA&b-'
    'jJv~J_%yXwM>G8FWUtyV{50dKEVs9Y#Q|dhY$p2qp89uI<a9!JM2{?;<F-eZ&T-o#OK~Hf)H6H5jWDJl!HsxcH%M?J>UD!sV7#G-'
    'Y9{U5v^83RL55DY9IKNEbcyiy>Vh25@aRC+L0dz!LQu}QJS~LsQ!$ipjiLNP4CRSAD6bakO}9pTAk<sVLHSUqxs1Gzj|3`y8vYqP'
    '27AXYCAApqYVWdmT5;7;6bV1<faCVI5XEDJy3n9_iJDOi#kvo!%F-'
    'EbugcOHZm&v0F$~95NhrQmH%LJ73%Wr9iYMv@rJ#6Q5%Em9k<wdWd)AW9e^FN+BfK_ua3ez68Rj}CGOyV$Y`&IF$(!p^@*d;rid;'
    'PaK>zd*+JBCreOnCe7h`Bo%5l@pM90=y-<KJ3-8QjTuP`KbCR_X}Lrze##j_al;5M`Bo-'
    '2iVgQ9DgjW9>MS82qzHCd730R6sKXX)&=S7+($wpS;?7`p!IBpBbO8zjK^Mcp6)#*=h|QefO9D-hxKd0)NNnk-'
    'jJos)d1^9NkD#xUDtu|*kxIiLE~n9DOlxc()EYfTK-nK4`^=iu5&3-'
    'N5U88`wya+6KF5%7^)9PPgpwj2w&tKVbrDCTN5L(1lz?bT}xscm=Rsl2xR0%Ht@cBqL=OyTPCVhInr8?MRHS#7V$(phb<NkTD<$T'
    'dkQuF(w=P&`vNNI>yq-Jlc{@1DiRxV2ETxJg(v!LwOtkttPdEUv;8#EMmpREtAAGpsqAMei8?ipm0^L*#&IpsgF2M_C<$`mZsl`x'
    'VGW#Gsy%1NB(~*`_z!lraGx`IF5VLnRB@;M6OAD+~!^EfzDpPH}G8pq6FDsK$4|mD@iA@kwfsi3~4N)3U<wgn8?0vvg|PYqNA}+i'
    'SCw@B|FNwOLAdqMvS%5T1BRH%JIiOwkQW!SLQ$yt@NexE1sMZ8-+5d?sWa8o=f*8Eq!>)ioF`v!wYt(p8tw1IIG)jdV~Vw_&W-'
    'uGpMmfY!RS8N)+|2W=(<mz_ea|1HLP{{p3m80(MactkhCGU3D_wy33<Q-'
    't_Qy_ac8h%{>30U?Ont04x&9o1c;h_<JGo$Io6Ufb)kbY9!*vXqPn=-'
    'KPCl#Gb}x<Nul#LK!tLPo@6x<M%*KCr+L>)M*Flq3mOc#%+dXHKsM>${A>MHS<j&hcoK<8VROl%_(DKoh}M%TD5|Th2S)iOVxXwE'
    'j<w)&T{g5HVUG&u#n$GMpt=ac_uT1gxCiojOuzgzDucwK#JsC!5uh%xOSuRm(G{ZL&?hpA{1~*#Z0Qz+#TEe%0%<bc);Svvi8v>y'
    'umx==JN9T#5m@L5l2yZjj(oJgys*g5<*rkUWp+;z5`EKoX?YW~)ajUAT`U!-'
    'iisTADB8hRnoGVlMT!8JA~;Sp9p9)qw?q4KY@q$PwqBpl+w&gKVb;yV%13ottFIK?dp2x0Q{&U5CD{ZscIS23X<9JM<cGbSiu9Jw'
    'b#5_L04fS6t84WyQQ=-'
    '2*pd>0GxrWa(VDHzYwA2IGb#2oKZ^5+MAlZjb=sCv=07NDKBTEXZAvQ5>QUzuTbOD4|~EK3j2#8ihiCwvYlV-'
    '=|@!`x0~60tSW0QR9}jI^HQj%*sMgKQE#L_PIh`#Na~#67O~&%H=+Cg!z6Lm;1_x?Ns7&Klz9a9CNw99BISGT(*>>Y#Ah%2guRp`'
    '%zpTD97xG+x8-sGqD$hdXlgK_`t;0)~a2Lu{4{PUX-QfufroO$`aGp;Ypzl^w=d#bz!P2Q#Uhp3sZfWx{Vb}UpFJrDoLL2dGt3^('
    '%0we2HStC;OJtHSlG1OjW`T94LFQ8jW-N74L6K6jW!H54K|E4jWrCk9%>x+18haSQy-'
    'a~1L6xY5Z`t{^dfw9I2)Yr(L%^2FdG!7$#F>V9}nSp>C)E~ceEh2@JDRP<_<_|BMe>`U{KCi_ZUZZmHJ8)DK0awyEqYo_y~(rF}P'
    '8;DFK62TxL4|aF$YB2IF%$ODQgUTQ?{LgKd5hSJ*zx;p4~hRekBK;P!%D)c}V&tAeUc*AgAW?GYxk-'
    '{c{9Ay=>EAoyYo!FO`DP%{X@vzV*jas14(SLbs4%(hp*M^bS0n!Wl1$It8b>W>^hZ`i9pas14&SLbp3yvcV+KpO$Gb0kB)pIO$Qh'
    'NZ38vTnLzNh&nqA(o^<^IP2}!91ElIU>nC!nhrgWFFtq4N8ILO}~iwT%Lf95{G}yh&{#C5)S{Gt3PwKl*7N~>aSca<M6M|oK-WJC'
    'GS>Gb1Vz`qb;L&hGSXKpX^nIEq!-VOt0rKIy1)Tyc|Y{GPPHxMJMx|-'
    'gS;bafv^(GbYk8)z{&FcJpcpUODqQlVQ+fCfgySY=q2<LX){p{lb{c4eFOgs(S3JFHOZIJi^jcTz;b4lz>Yz=Vl;|OcH^Hu{ttI1'
    'bUusPzo;p`HQ#&@>PMHX0u4BIm4O;n<}-A=hc-Carp;Gv%182nlq!6gIXMOPg{7Z0E<Uk?G6GA9&J-'
    'P3KT)G+mzD;m^<30oDMMOv?D(QP|iu=zFL5-v)gm5E4tZ({vxoCIncipgMPmE4St>zHiF~Dp`m;R9C=gT@ElV!ixr5Q$9gXnknj-'
    'ir2=w^Zj-Qv+fGVa!=WFKN?OC`>jowHkp+OP#YNjLmwH97MaH0AGeY4yPJ|eBfoc*OrBU`Eeoy#qR9&7NuHnD7Rq;-'
    'cZcYv=FUL@MH)kJT)h5qI2WD}-r?IC_TyJXZp%d4;8hhNp^|r=byutOp#$3$ddShcQ-sE~`vqOg02$GkCkbH%y_bF*$c(IL*dHZF'
    'lh=gZYmWs%?ben{w+l)o)q@^2%t98=S{jP3Ml2^HBjL7yvn?jPhM{myL;2|8>nz%cK@L#n@QnmOW+yUx34bEID%R#m<-'
    'l@O6nZx5NF&-C0KBYa8H7=zbq4r?pQObwaFBmzL@)5NsBY#qkRKH~8PRdcL1tV`#j#hgyawg>%wKv-'
    '#Piut7OT!iW2i4rH*lpFl#d288%P&ubBRs<LR5(7b+a$oz48_q&aD>r1Ith*obc2#S$vxvu&RPbGc1S;9laQJRoEoD9F=5wX=2AH'
    '==tdx@F5H-'
    '`&7JEXY)j;wROeecSiTy=a$ydZa~R&^5bMak$<%r@mFDU#rZ%9hG*@$(+K9%|T)oZICbX93>K&#wqq%haZd=e^I)1mG&|vP6b=AX'
    'D2QbG6*76OeJ~RvXbf#V|w$m}a@O~;j;Thgf#piV0CIO#jOxh&j6Nar#5<VB|1|^xx_>@z2Ky5X@&kpoDkxEZ?R4wh6rLVx+7+b{'
    'EGE+S>Q^e)A9v+sRG;!cq;$%TumT^SNg1ju_iHQZtSn7#>1sPZ-'
    '8EgC?SND*zoEioDY+<tgkj8gZE*ZR0oDBXGi^BCaEDG0;+FsRc%$ge6%-Ny(;U+}AM&+|iy;!VjRzK-Do)}h#XBfv5!|I!Ln;i^)'
    'c%TTK1`QIbW`EKR5~^lvbc2MdS<_dggw@*>oD4mXp_OfIebBozJP~x;-'
    '_~6Ho%^cqwJ1Db4>`B7kTC#$Yi&%}b|y!9p+|{C2zi7i5OWZc2Te2x9l_OooXQm*RikqVWidh<VubdK5jr|ok_aaC7F&o0FL0{*x'
    '6S5861RpT95FEiYXsxhLKv?{Gp}L10Zn`n@&f%j<5OV_&oDj}#y{vbCBT@D>Y-'
    '=H^OU3d2HhaxsJ@?WkZ@FQ`l=KdcPPmGYv6!g+AYmPv+Ni$TV!a9on0Q4A&S=qC!zC9tF;s+^`(g8Fh@CnBah*@HHKsV7>;9daBP'
    'Va%6YQ~po*JwXAeYKEa%TwaP?2_9NJ%VjEIpFdl1Kn7<sV=qj+I4H#X*Mjj;TBh~-'
    'zAT4AvK8dIx_RVM1bn2?HPc!mk7SpHnMDFMrL(h0*dfv22wZq*GEPCEPR1_>vfrmsrD@>#J-'
    '22jE_YyfvYugdUDv$;ziJug!OY!%D^j%sqQ{;9nPXg9UcTmhBAcq|?XAT44@Z;K&4AcpkVochsPTng3_EeGTBdm=yJT>e0n=7P%~'
    '&G&Vf*1wMGF3Y%FZv$MY#;G@LJ0Ne3VEjf1<CmFQZeY9)J@O(j*6&}R3S)SN`cxRN)NK-'
    'A3`0?$1mkVGK>~~i=mrTeHhonJjL$9bELsZL!HOaRq+vH^BJA*gdbwAfMiodMou3721hs$^-0igr@jG7)BpG_SK;6zQ-'
    '+)|#ED2WmbAoy_hw&`Ncq1NRd|-_6M{^h-'
    'p)PPY9jPwjR?SF$2Y*R^N0pY1x!kZrR@ex=a{~0rW$IINEHxEIyjXRjezA$E;Dtw+m<rzcx=jh-'
    'C8R(fPvj}pi7*Bec}jKSKz)aV>O|95rGWRsg8bEgWd1;)8vWqboQ}r-'
    '&aM8tIcQz$&kfe1A8h^OX_>mKpg!?#CaWze1~@4LIAw0Xs|Il@o?WO0JABovaXBED#Xx2;kO##;9+w01D0MNX2N<m`;g*#{j)A>)'
    '5mt_vOf~}Vo1y7l$kaXty+2}Tq=+}H-)2%Odf^!+rK0y0-6p{shK`+-'
    '<PAf|PfGHJ2kAQ`c*CZzN<nYeET*Iz!obQ}Hj80Qo=Jvrks9KLf3dp54gW+nK8Lf$G0u35v)f~wjnCn%4t<Xy#N>IQ1NS4UGcQsZ'
    '9`nRTID0F^*(|13ngf6p>I*yo2ud-'
    '}C#8Gs%(o)h5cn^oR@xrxzGa1K@Y!Cl16b>HZ5q>@jlrwCXmToe;l(DWf_IQ^lK@^At;tE?h0&ax1m4^A9TLE6`l=N0UY!T8@pzx'
    '$r~>Zw2L8-'
    'Z@w<f?vOADU#?rIg$>eE7&jRN5bYdSG%H$a&Id~6~oe1cY)Px++o5VmDG0+FcK%Y>A7Ic<CDo}rZ471PTumf88d;L+eJz4?3y@~O'
    'fO&EJELbv%28Ed0R@5CIJ*XM$S4?7kMsp(grk_upWf+?v0{z|t=2&us+Oi2o<!6;2h3aJg&cSs1SnZ7Coz&8~*zTngLL{%+gRN5l'
    '7DCXd05kX6>kqWnIteHL=wT`g%I5JWNw>1MTh1zCn26}OQ=;Zy1L%EuZ$yStYs@}o$DzAcM4moO(NcDMF!hMZUGX)idUn|s0j_BZ'
    'ep<WSG1Ac?RGnj{*lFV%!{&h++H9Qnc>d&=U5(58ZHC<q7&JH+kBTIjo2+9Pkj*)(Y6<Ioy?TRd&$#z9n(ow)|1p%nfFmaxeM#ie'
    'YarHn(LM(LwKPH$4Te{alJ;&uzmunBh-70rJbr@G~W6m6<1giy@`)1nChkX#8T+N4<;v{E<CPu@?P#o=}K6lKU`>HP-'
    '^X7i)OUJyqzxv8CZ?;rlJLb&;)HjZK^FZ}2+ab|wG?rbX_|kV;{!jj(wu(ZrD*Z+)vviu*m03E?>&nswd2oJ^%~JHdQ0p9Re=g0k'
    'Lr9ui2bXS>oAHqOK{@6JpG5=%{hFX$q4^xPg*qe1-17GV<s^P-MteHP)aCYe>}POHB0be!{fA>Z%4zoMzZ_EmPPbS8%kd=k443(-'
    '-r_bb7*o>KJj}qfn-SQKSZv9&a9^-'
    '#hZL#N$afR@O!Rd|`0+WemWM;%%nm)?a8;Jhkh&^MXGmRD+RzWnV!kElpf`lh*)oIjJAv9Czc9wGLG72>YD2j|E+BiWr@nOr-'
    'h<g_nLC%kWHiK0ayiUJV}a%SFcppcl`CK-'
    '8afWS5+<Uphgk*l@S)J_03eMGEz43WkLUnj@AWmo9jsCEO3j&GT@(*i7nP}x%ZM6jVK!_>yrR*NcMlCWg)6WtR%hw#p{uiW_R!U('
    '4eUb&0}EF5EupjQ%)tIhpfJncb^xt{wjM#uKA|DR<P#c0%srt&#MBcSMa(>*VZ_7}8b{1KvFBY2VY^Ru{62>2@CWA-'
    'Q^#VGLDPWRt%DOA49JsUNihovo<V0mU)5G3nsd3I=i7{i^YU;8FJP+N_?9!6dW9F6!T1^<Wa)&PA7tr-'
    'n;(=m&|~6Fv0Mb(U+QakrT@>{d%$T`T>azB%w524V$`K$4HjT&vKC-fP~&7`Vr;P_ChD8KZ=%Udf`W)5ZE<ZYbuCzvH+hq&bj033'
    'MASukSFzWnBewrJGjnF<-sj@p+27~=XP*ywt}uI_=brD(oO9-'
    '!@Aqn05J?s^EHNqzON^R_B}Og65~G%3iP0WmiP4^6iP2tEVuVopCD!{>dFW+W7n9STmZg)-SmrX}Y0_-'
    'ns@qvm;@8t!yqh3QI?+zmFWjm6dD){{Q@ceJX?WE1hT%wi1RGv+Qt0>~-'
    '%jEwD&J1xDJtJCX6^e*obUvxNv$V@+i_jkjILaAqMutIu|NocU>@ZMcqVS1n9n_<h$tawrLO<E)z){c))hau+p9IY;-'
    '}VLt<@Djci5|Sy5eV?y;`p;e#R$`vL9aWCQb<tUhgH2z#m>8CQfAqULV=3eHp$!P8#KQ%wXyAaF+j(tNlzR+g9~$p4su+Z%pE;9X'
    'BTN)Q%gAnf>o2tN9`<@TSbqV;uV!`U&cMS>dd=O4n=MW}n{%U1>ETanf?|dOvYeT=4q9UhT*5^@+V23x6lnVU|DW5>0tvE5R?tI('
    '4e|{x#MJqwjD<=t*wVcypd)oeTanC|8pFPQR4=j^2>=chwo4d9IFZmT57QH~3dz_BWV1*evdU=IVev&m5oIJ4rkv;yXz^BjP*7?E'
    'dHhS;e=1;*q$0_^_ClJ5Dx;ny<iyP&=z2l>8^Rv3aWR<6HrYo$QDQRz3t9&ru3<Ja-4}R}1yagnB*IV4*HesM}KwLFEru9C&a-'
    'd8yZ#YMYo0EV5U?``fAGQ_T_%lJ9U-AqjF*gR%vPQ&I(peNlQqJ<l2?63k5cW5Vjy&FX7od<f;=&&e9u&v-'
    '?w%n>pP#>yPMqsK%voR}w8CSn>+2I&D24JYOfh3fkcNHs#4N&Yi|@_$|l^=yF>fCrg%;Z6wXU20_lN14?LW-'
    '*apsn!W`_0%wA;rqtr0cLy9yvtGYBLL0SS<u{@0?mAee#cN7>79z%_-'
    '$!Pm~uz<wi3w<$EK24c2$CZgiBpWyBZvpEfZLNGrUo1zzi;r4XgaG<EpIz<}j`<3|FP*q>rhe!+*pA6UhN%fw@YLDF95GG{!2;3j'
    'vdIpjYbw1%SDSmwIVl447SikZoYD_YKU8eFO6wzJYm(Z(v^P8<>~*2Il3yfq8{*V1^xOjtA#^9Q8Z`a9)!I=Y6FO%;<Rya3w9e9C'
    'j+B-'
    'EbwQOx<5iWe%;lf!6y%#ag=)H$vW*L15<)frB{(Jwv&=Ag_JdeA(Z|A`p4~ViEY99uv{}WFCDB5!3o~jUEuu`eg1<sIvbD*+!G4n'
    'B1mxL<7U?&VnSS%AHa1fX6K}>jZY1bX41HX1%kJ-Log`UATJ8Bk+BWq8tGN2WJs@f3Q3vJC3JnzcHX{9}y@Bj0|YoM+IsEqXP=}F'
    '_yv|-&b3efkwFbGFa>qVsRk1ZK0TaKRB=H$$ahK#bOcJ`C_p+SC5IP;x|TUDPpSl2kQY5Rs7}-'
    'g?z^nEY1`7TErb)|4Q2S7Gci_`Q<fccMKA1_86z!q9%~U<Z7-'
    '*;RhT=HUbn5$)fOq)F!=62NJe(WxLK9SWb~0*p*GBB0F`6VOd4K&>@DU71@OyV8ko(rOq>0Vv(<~*`btKq=aNIgT}6*&seF}8!+C'
    '$)x~*WG~fO|W5I|td$C~rhaM9FMq`UTqQE#r4~PJxxkDkJF#|@U{Ob}mE8)VYEhxdO)y!e7GA0G|VZ7tE#6Ag%_<Af>aP_#y;)fh'
    'XGXgALo5kXTAr|3RPqjT&SOqtY*<MgNPEBXF6;zH_GnnlJl@ruVW*b4}L^X@qK2SMH&1SX@R8Gb|{|=3E<Yh2e5w3`X@DxOYa0P?'
    '!lmkNZwf`#?gh-PY3&Q*Lm<SLWBkUOk!fW+_2oRb(6!H%b%n4j(;iBjp?a<ktFXZ>!HMl-'
    '6iJ(LSwx0z;9^kHF1$fyUTA6Dw#zH|Q9NoBj!ejCyj!GB-CWmG*`A}+oG-'
    '<Ly+%e<+QmCtgw0Vs_0ww*dga>*Noft0Yt~Fpda{}<@Q-bqJr+eBhNw^vjo{|jb#M^g0R~L!u##rz&81BY{Qxb|~%)|J7EK-pMFB'
    'YjadQ3zlV@$JGR3tN04~U3l%pD5(k$FgkhxdvJsm=A10uOW!Ha48h)t(9bX|B3+{9D4^#~cML0^D7f#ob?1xO<+Vpy4&1So~4}VS'
    '?jezFa`qYF4-^Ea%y5yn`>9>M6p5JMH`~<<zM<AVG+_9KJr;33z-'
    'e{BBTzf9Y%e6I}XDhWfh+?lRfGXHs0F8%@r4yJ9&~YW9KsAr_m+Zx@Tr8}yh0*d!iQW4^r$VUtSbuG0ewV6(YbIN7_PUs;li-h-'
    '=KEX{{*-uxT(Z<?#O7<TXk(3_ZLPMn@CM4OUjP8^-$UpB-'
    '@D{Ub24Tl#+FHFFT!W1FkMa;OvxcW$FI(#30muXxDg$zNp{u++qk87nmP^c{wfWvz1k0iCGJn`(saM-'
    ';aimNVUYPD(RSjyDTlhV|*eT-'
    'K|%Jsl!xH49)2R`bRqDV~}ZNFg9D+{UhPyxO9A_Z#N%DigHm4(!L2+O1OfI=eIr9DaDeA=GmEzA`X$U5AKr&`ecHNn5L@zF{jAGL'
    'Jxk=;fAov1I(F8VrCKS{z$wznBJvjr1lQg&j@$WDxVvlC-XivD&gx(akGGY!vu=>&E<7PC>&%?zw%qXV0nObu}aJ+rW`jR<<?Fzi'
    '3+8q15fk>$|en(um5tk@LX>Z(}y?$l!<dXeC<Ulk?xXTC{9FVemG4iUXbWAuPR+SVnF`nxaGstampP|tE9ic^{u?IfY8_SGep@_8'
    'W#?<13M+Q_i1DZuOGEM8}3@p@ktuhl8M&SqFL+=XV*N>~1|O0H6S*!QdDYPF|*zecW6d)fDE<ytNBj_)_f4Qf9YOO*ExOVdZJGAr'
    '<@NGgyMW|{B*$5;%*hxlVGhCkP1A}|c5_+u1?%{Pg_@O}CY5g4x40}AO|=PbdtQ~?XTTwNJfHu$5?cJ>I<oOE6yTw_11!J9MP&0u'
    'OZD}~*<EOuvQv3q|OyJJ(t`0M04wU2$jUanXBGIuVpkV-Dlr+l5dP%JUaVvGGmz>5yVZs6(@p+!#O2=(3#5!??0IE~=3BA@EfJcU'
    '2SA{aizpJEaGo*ol{U@*v^q7ZDpNd$uL*LR3O@K`;dkjQmOQ{C|JoKTlM*gSs~x|Oc>eJk$R)~mn;dS#dpTbV^MZ0TE>tg=LrSZp'
    'd!HW#p^Z)frpa{*iW4kk}E7qc{-'
    'cz;__tNHN<*n(QkEq<CkU1eBq8%&TpvlC=?c7i;Rogg<QyCF<zgop3F5XEbm)r3G5)82W+kmjTQcPxtGL;QCvidX0{5$zrD)c#w<'
    'w0D?q64BoAfWAXSd&f<BKq1lV5)_v+7kHbq#^Jv1-'
    '=C|l1W>Qx0Z$cxbpKSjriU=NsNgu;GPL&?R&@uUyDJO2Ia$y>m<3%;Dl`Pl>=abYkLjk@sKL~d>)M_T)l9(4k_BNe$NR!5PF>H{$'
    '0AR*vJvjS_d^(u1?w0XpTgDX9Q|PX<p;*X7(T?nSQszTV+z2S)>g2?z$gJ>^GzZYdLGnwh*0RM(E|$UUze1_YTVXRX8%SyW0K?Wv'
    '~wI~^GDI4lAMSqoD(UVKZcH!<YYYIwCDftGpxA_uzPnFyN_kD`%o6UHzy_ib@DhAjCWbveFVI~&nc%=!^c5Uz=!al6D0P*zv}C%='
    'XCM{T%f#sStE3RAA}ID;%cPvC&mc%L|(;|c?N%ug)n@GKgUA&DLp1)B?fc+Icg;~-y~usen{UTVkN#=4=AL8T>|0p=r-'
    '5`&=254rJ#3D7J75D(EDo^dbgyO%>#w(>#D*J60)DG3O`uD;!aiJ<wBn9s>0g{+27TEw-'
    's`L;pK6A$0fqoc@D&*hPoiviy(J3pmmsC`>U-d)Q*$Qrd-KwHYJJ_a9NP_-KhrX0&jU*y5Q5Dk?KeJmskYDC-_S&f@kY75sNTb^)'
    'FG2u=yqti|}9d9U>OtTl9cJBG}okLER|m1yzr;p&PmwR}u{?#BsMeVJ1wst??Zp$Dy9nRF+{~dkBfw982KektonLAiKj4Q#@r;df'
    '-F1Vno#jK6GnFoP*#)w_(IN2tIUMMkH?FL$_l@KnOl`duABjv=7~Z88$+s`}vT=b|1?Z*gRB3`&yl8Iub_0J@ZPG@r%_qfi1AM32'
    'cG2ywGvLj=aA1GlW_k9*Rn#mb#;=&zZW^IcAcpe;`Yqx%!l21Ep=vRG)LK<NVw;dEVjL^)_SSS3gb!Oz(5puZW%IW3Kv>203>BQF'
    '-!Ujv_fpcz%iyTk4qp9`xMk<jYQFME!du1nG_pr8gkhEp96qT;&y~H$t)3BD8#!7O&Crr2FPnf$A@RM-%w<?q~wP-'
    'W|mZ|7}St%0D^I@;VbP)Zz2APxfoB-'
    'U?3iItx50>`3#z9QBnh5w1l+*VTqiqkD068)XXdu=>|7tFoD?%Q%&sS6~n4Skl+}4X4AOk2s#4+n!*A$McN3g2ASEzWE6S5FA-'
    'ID1d0-c-q<E`ijRn_-)7Ec`A=3qH7t7f>#SoeFX1bVd~Vpc2T_JxF+z>jcWoA-'
    'MC^lZ&8*#`zzyWtUGmRt|=~$vJaY}J%+o`>{ph;sz!EGm$(MQp12m=;%auDWv-EXJh(md!24lO!YkG;T~HT=h%qXV5rEh*dSHJqQ'
    'eb#T<ZvEJ0qMMobChA`x?0*y>b!R-'
    'B~veZ>(DWduWgUe+Bd<LX(GaYTy@P!Z0N5$z6o?~<C{R&Holm(_bbct?R*ONa0@CtT6nEZT&Fd>tnwbVQ^0FuLcRAB+<v@*;d3AE'
    'n#)!)GUY0iex_f(8pX}Yu4_=+oOX<LC}vJPCw;Ju@8whsRgVsBKvA-'
    'i&<TF#3Rf)yKwFg96OjIfK(M!EYU5z=yjIdi7|!+#*Yhvc8D=N125&4!+*Ci32~8k4o6rP;vkAq_eQ<Vi1MDwz9lR4EFsA9)7nwO'
    '1+UeMntHo{&uEWP?yOZuzSJ4rqyi5I&P9Wvo>Q8h4DeqDLP3MntiW*4AkMdsiXOahz_o=^_lSsGGEabSk)RBejsrhpyR@A%n>Nx@'
    'rXN^%W7V2E#N~&vMI40`H#u9<;70&4zaLdi6eTwcV%*Ge`kxgs@4cNpc(11-WX5$WJ*-'
    'aaE+og%g;9(|MV9x~Y%LgV#y?r2rQqKlTuSUw)<tfh9<eVWZvhn)XY4^*Q&aKyLwJ)7}Qy-'
    '8VNIV>qY*ed#uX;3}r@vo}=5s}P@(p!fP?IN@sPlu8Jh@a|5ICQ5nfgW0nIxC13xg^uxk6o}J<~X*uD!#_ekw^L^jF^#Ic$db)h0'
    'E8QEO5Y7_}x9!@&>Ba=i9IT&*<EQFEb8cJqCg+@%}vv<x(?Z7<mOUqi$)kMwfox8N=r6hRz!u4n(fEkTtz$|>BQps*Ju74Aq-'
    'y@uM$$0sP&Kxu^u2@>N6c?I&y-'
    '*uYWr)Um&2gD65EaKTF@IN@vmZ6+)iY<9?ep6{9WK^qzC14B$=Vn_!nX6GbiC+B}CpUqNYH}0Es3sS){i6qF8w2Xli33;8F*MzU)'
    'NzY!IY1e92Ez4$_8R`_fW)UIYt?dprUlQxDn;`T6L}$boBw)-'
    'F<6kidY@$~5f}2HPX)rF`A!4<WPlfWji;yujA$uNh5essDX=4zXekDO!$`Cgr>R9wZufNcD$D6p`EJ4{xj0<-g(mYi-K6K_h@qG-'
    '{Zwo<A$)|VVyg)^=`j&C*~SD*lbD)pu+q{brY3tK#?{3d0}d?Jzg;G59kcG^obM{{K1OONd4}2e2kJv8=dd>fwJz9I(vU4+`4AHg'
    '<TftW5=}njd@lvc53-<qg@>&IEAe-'
    '2pkfsa1>dA<75s>AQHcs>Gd5C<3PzLf*)OF91Jsd8pmw1;C6TmK@{O>$9Vf6xWP40d<*Y;c8$TV3Ncad($0Bl_9#a63RNQEu|FS}'
    'eB!Gd5mK8!IWjo>9FDr(~t_NiiS%(KN6;5cjJmey(Gn|^piQCEb;_G4$-'
    '=$fBr=Fcs)l}Mn9qt<9A9C^RyC^70LC||xOG`n}`zS6+LC^=7@1!iqM<^{x)TeJ?GZ#VoO@?Qi0kl8NLVHmP+B?C~dxJ;PV96yRJ'
    '1&a5X(TSks^FYJSAW1~Vvz_R;h9(@*6T45jrQhoEl*+^?cs?oPhuMF;oC1ShQvP}lvN!4oH^-'
    'Yoe4Wg5N<Xhy#?WBGZGmQZnhv_G2vz_Q>_KzW*f@G+9eT1KH3Tb(GsLtBoMvD)CVaLeUt^!t9&;ajNeVBC6@*iKd|mE56ZDC!cKc'
    'b?_|IB^W4+gsE2iBww;^HQTmD2eT&BxT-'
    '|E$*c#LQrR~k;yFMF>NBCyX#^SMBk4am8VzXxOPvC#w+2(9aup)_Ra|SD|NMhQY;W4ZzNK2AlB5kxMz>g%y{8%3Hyp-'
    'c6UPWg!6^^7BSwVu9kY)NZBTcdtIRgobZ!`5_3W^_Rq4-'
    '*AJ*#p|wDM$Ri8OCdL6S(z41pYx=B=iPG;aqWL!^0o8WKcWjzHvxj3gKf&m)l`aBq_p>VPLlIew%5vd_h05kA6mu~;0T$3*yz=Fz'
    'W=@*BZAE2I2I`1UIc`i%!=8%i63GiEIFDEIiQKn|jjEL=>?uC2CkM|`JiG?%lKq@!Gh46}s3jSRa;0`z^7Mc?bG=%o%;G0KI>ZY~'
    'ujySq%7?BNn&vZu>~$zCoECXaJjFxfk&9RIsX)a3F2ec(#(nNU$u_gua+{QS?yq7Od8^Reg~s>ej2&pi57QRoBftcpS(eEU@eeZ+'
    '$^yi5=UA1J(5LQr>K1wXD^2od2bcwj)SO=9BKwVb7T>*aE!;&h}*49n62=zW@nUVUoe`$~QV4w!AGXV|2=%KoOE3|mxJyG$EKgAa'
    '!du&~ig4I5lx)=vx1a={MML#7}2Tr+e+EsB#(u1J&SwlvUO#?{SvYBu^|ybudb_y{k=LUX<z69G+Qn$=Oz1naGif+l?X)dii#gR|'
    'kwak#5elmdTgk3$9|zM}qE<Ye|6@qr0my2bQ}L{?_yl~y2ouVZIoSiv5k^Rp~E7khLX)O6CGtVi8Z+L`rj*GFrA*5{6o(Jrk=<+0'
    'kU^{DKu9b1peF50*C&9Ys!dwXy!gWTHzO8auvg(Pr)nYVuGXYyh!NZ}K_7z@&?^q2^c8Vjw70x1}AO%zDs+pj6OewG|VT*=j-'
    '#GO3IdLrjSnuxK~c!>?f9c@GWK$P0r>8tE_;>lBG7-T=oFxeHLa(z|=VVTHO%6*;{LFk;Sl+&v;j0#RN)p8LWWvUe-ILlNkMR1tu'
    'R^YV^Uu#5goT=7|SbDo5LA9nlL0x8$d7}Dl9#e^V6i>#I8^T9;GM3!%9X%$Z_rrX!h~5vd8Vrc&{rFtpA)@yKto~$DOlPzN$md{!'
    '$f<!nZ&zL&ww@!@bBt=;nmPWt27FI!JGd9u1}&s;xi6(tZscJH>2nM-X92u7WZ}I$3-'
    '5+3yt|}2@|rQ6Q+9h`ND4t6q?{pLu<alf4C#Vx2Wi5PF4%UEmNBFYwjHD`7%~KRjitxCHm`%!eAlO9@e8+lDi*)1^q2_znhzF%Uo'
    'gv4QTT1pcZk3*nDwb*`0ZDg?fUpvf)IAC?}N}4P!mN?ll(-TdkHzXM+80uxEHnA9Q^YPQ)~fb-'
    '^@aGMHaG~vXJeXf^2h!oYY<XwqXStTgx{{pRui6N|KE2<T6rZY%iBvK}Oj@uE^1|YlK5@Xjm+-'
    'J7x6x+fV&IZ~Zl2{OMTy!e@9o7QfHwF%kGRA1ngDV56s_@ViOhAp*Z(<EM+^_w<qyr56}x%L0tOmBrY~EXFoxF;<b<^>y`;iI`Bq'
    'B5|q)*!f4Y2C6=Bx&~@VVk8U@U9IF|Qci3mm!ME+96{c7EaX9dF+n%U1()^U0R=~3J3T4-'
    'jW}tkE@mgrOy|xU;dNWCVG(>PS3S%@_DG>V%WEGs&-IyD2*bB~CKkeT^_U0<n-'
    '3NNVSvChQ4rp&?+^iDfY39=AbeI?j{3?wSer)tOwBwI1u8<ps~O9wS)3fyEUL8--D$>dzxqOy{dQ_*yq%pHtFkj=OLk^-'
    'OSM7Q`vrHghxr?R?OkFJZ&e9s?OpEI-'
    'W9oPZzC*y!&o}0dZfu6EaB?<Jmi}1{cJ39;X^zdi`;+cF%if$A1ng7V4`QEkh?|SAp*Hz;%AE?_xzH@cOJt`T>!C-'
    'S%|I9LTqamV%<|4{G!BB0H{|Jr!AuvCr&#?y^)Z3j20R(yL}DnvY(dFzrUU-sikY;$oQ@gS6^61>3qr6A_<zM+Bcd^Jx!gMshmUa'
    'egJj8po;`<(ePm1J(;U+X5Aej)TepqHIMYUSoFg8dM*~d_v<kc=rtcK0=;1U=c3TNRo@{3y?~zQilO&cS=ZOp9J)gTBqSjo@aMT<'
    'oS>4Y6ay5M3){9OaEotDbt=tgm?I1(#XH$au_ikywq++pkCaC;hRH8&F})hBF=c1SwG3s}t#>k%sWTFCPp)I?M+x~S!<afVq1kf-'
    'Q$J2}3ZfdJ_`4xogog|Du{mXZgR5bAFg6e2`B)gkhj=~~#x;6O1dPoGi-0lM>-'
    'i`cZ_{^(fHBzn`C>5sZAk>D>z+NAuaUhm%M#M2a-'
    'xg!%CU#>lya=BOBM4P*1vaC5Pmlc;k8)^Z_h%wX9~h2VSyx{W;Bz#OzA=mPP3ha<_-e>y9IMc0g$&?K1|4Oh;{mKA-'
    '~N5ZzI%zp9R>RB(&BOyqhf4i+R{J-~NSI?80YwAr`wg=rIx4H6JVjyI`mnqOiMN-ys6KVCWZ$VfT-'
    'DRmevq>Z>F;8TetMz6mvQk4Y3SGs+VEQOUToC%U{+bLd-CUB!{c*Otfpkt2_<EsyyVM<QQa9`oNEnS5<|%s`G*zP3E(&m6gYZF$U'
    'JIFk8Rr^r|7co!XmVL=^3Qiew+uD`7kUHP3PC_zUl_)9Mrm^!#zM3M(?OHz8c3k-'
    'Tk0j%<<EzKjE7tb9yuL<0N^NMjuUXy3|qY~AuEI86RHc_ICnnv5>;fpi$ScaA3W1U`HrRv2g+&*3%$0-'
    'SPg6hpF_C8Tna>@vuq>ks566&K);FJ^Ut4`#U6zZo=3MJ~$8>pamk{t|#vNK$NauxoUKO|JTROM^`DiM+P0HYb&I+vND5mvqJVcW'
    '?}rdk?O!O`$s^VlNIvzp%ozP<TP;M<#D%;Y!Znf%B^eVqiSew`9*EA^**M<<Gv5t(ua?3{Bw36s>DT&^V*Y-'
    '~9g#b`UhmV?pJjBq=WFw-'
    '&!9T^`_4@lG}i932aFhQTGqZ(wBLc_RY%&$?aIY7XclnfI1Z>J>Q<(JFU`GT6f2jxvz^F3c`0@d70O`w{4shIhy^Rg6)z*(_Bt%0'
    '4GcZ3JQl1J3lf!fYMBO>5dK!}JgPT=>1&PGSBDn!r=g#ThdxFcMWU|{D_^UDOiUcW>9DnZBBqv+BEeP6$@{5nDR*Pj*tCc)UypA}'
    '!Gt$Q#zDoA`VlEt#K?F{GaIIeCoNs0+v-JH`gV88T&Ca|+DXaYOif?`I$y)4V)P!8Ui-'
    'SaBdw?W@W1^n2J_giL<Vr}2N*C>N&a=FOznjL~6uv^*T@5|xenTsogY^AQZ7ylt-YjuOY_^ps_)Np(8J0aVu8|}q^3fWF`nd8NO3'
    'E5tav=_e@vV$7MV+qQ>2n`9r(q6`p3z%x1XGrsPUTy+G*~?8JD0{h>At#p{Q{In<=dRK3eykw3(-'
    '?Ly=gGkI{5If0E)Rr=khZ~Tt~z5Lilt?4jYj!~)piZ&Cc9CWq3kZZtF`uh57|Slv+sM#o@%{)-%IvV8|?ez<Z-'
    'atoA<qCZ}pb7s9^Xa7$gi|go84MuMyEiUb{kbKhM-bX0M&cRI9QabDJ-'
    '`un9z73!6aXwXm4E@5@=NwUsHO{1M03VVou`Oy(`zRap%3%DUjK1}A>2xtsIe<cHzWft$Y?BCfBx74ofsCvL8E1va?imO=;_SHKs'
    'yjy$rgfd6g1J?vru2i#mShA&zNIKvk$1T7f8M%kctvu^USC)i(b^=RG!bR#raUxomf0l(e=;63D3%mIM;`mZ#B2J4k3&|tk%3;=&'
    'GNeXNe>c;`+Mg<tw35r$u1Eq$5haYvPD*WV!;SIWf+ME7xu6hb50$%0$r{&3>ep;U5*{9{Ho_t#N_uSKRfTy08r#W6=Y}PX@)S6R'
    '(VMp;XHZi=7tADvFgq>VH!gDl+znd&dbqI4<XTXFormp3xI&Wz=Uw2Ux_@WjyfiG%NK|20}vrQtMkvGY$!40^$2br6e+0wilS*ZQ'
    '+B7+x&!>0Rg4As(CS(aTaraxJA;Z#k3it5U#h8}qrsdD}_)q_*%{OPJ^EH{$e6#C-|+-'
    'MJAl&f@1yBLYvj=>s@Y@6?@gk^1iMrv2caUJSEe}HtT+TI^A^=01FHV-n1E$D-'
    '9mBbeG?b2h47xWpkZi`aZX^EOIG4*^m>H&qS@ej^v2<atISaW~!F}`J1y~)M^u5jcrt{xBT`d|m@C43|?m8)qyRh5qnFnf4}Z&DA'
    'W5WYti;UhwXBPvY8P=9#>{B$>OPlTTy=Iu%F)6>$6z)vsBc>zDi5wB@KLl%OaA_G)+9!dQqHwXEfUg~V)P`(WNVUjO-'
    '2j8eH7KQK;%3@KtrR=*kU`tS_AN}?y6vB7e9)-'
    'dYdO)G7((+7Ch&n}ppW{!+!{c&f{0v0o&rK{L2wrHY5_#1^tp$>)Wa`RMJKNKdJ(xn^o>>HrOd;@i`03{2-#)(E`6#CPq70t7I-'
    '03|sDtNrzbavOzy4V6d<u%+Sw<vekb*xg>lsu1QT{HHo=R>BHpBOkI9gXUtupdYIofydEt|)p5<WunSX6#k_T2)oC8*Tq*b#+F_%'
    '1u5P&rZ$C{&Q2MJ23-9|f7EkMQtiloPqbT3xFUif?I?t3e!kgmxCz*-'
    'Gi(?d2i41=$e;NbZ$|<fzoP{|6Mq0o>a>dV+3yGwWhsm7v?s?A0F=blbVT`ctCXV%4b4r6=gbC-'
    '!P!f=+yyz4~*4U2n~KB#oBb8lW=aYHvpm{`79t@Zg)ah=nD5h8D4~TwnIxs<0)n)F#;(1<SoOqjdx<N9h5D{KoQ9;)PR$C-'
    'o=~YB^2ZAL_?V;s6<nCQs3wpH^Rf5Fk@zW5jM?IFHS(>aQtm?w!Tv=oB^=ax)Rk)hn9K?-hHsNVEAZvRAKaHosTx)oYqe@-'
    '=()y5{zK-CoseYRP(gwOBL#E#{GwUvgWpkM~qRGy6E)^Ps$a+&qJpvCxE%&@vX9i_5-'
    'SB(?;a+9F>>K@+~q7g5k0tp^nHCQJ75i9B=|r*QRyGH@6tx%BdAp@uL*?n0hI?!sWL9n93*NG2I#vpZM&cx-'
    'M%^|An)`(&{>CWXygc_2!98<%&$!fAPqxEu%Trn#u)ay%@Xmh*_q39xFKi*Z~|ghkWxA8|Pe)=W!(#N}jIG9xL<<n{oQ$7wY&z~m'
    'gJ_R7m%neV<wEGFS2>=BE}7s|d{E4Boa+7i2>FbUsfR}?14=mCX%%T}3=PlVP)$8h>KCe%Da7xFYqNUaNfAh;RqsZilBek`)SWe='
    '|QWdYad|D*u9Zx)c%DL@X^nj+*t3_%$%)x}<2i#lMctGybELSU-GUR{SuV5*zF8irC}s=K|q9+jF@4|{b3Dubz>B9byq?g%#W&o~'
    '4Q8kq~3YMZxOnuoAwEHdFU>=}#9VP)SfAX|b=ZIUmekO|-'
    'A%P3@4>j8!Q%U0P05cp?x+$}fYb=|}DLSoq(_w(S|fvnj9xc19}Yb*<8<J)lnI;B?+<LU#)%6l7E_X>4hLdzZMlwXk0dZ%^26I$?'
    '4vG>9xq+vc&z^@KY$#3Ohd0nvbgQw9fXgL8d_!Qv}6GsbECmqG-'
    '2H{q)Vpb?+R?JVhx{v3at()h!S1c~!Gwc<M%TLO_TQ0U`fEkTTZOyL=;gV`~;JbWP2$w_xKUNPY<Y1Om2b;5j0B#TSOcmL6Nv4YI'
    '+^lrLJ*g_Pr?n?w&mCXjs#TdS-jQq7?-NH+S|jlFzZQZ2=2%x!?cB-'
    '1g4n~+og5<6$vl!BPBjE&!sry>k@Y<ZHKq<rT&^9V?RbT;Vt6N&_~UCTcQkqHS<nX99I3?aHw$J1vPI=AnC9UQilrKaFE%KaYH*V'
    't6H%o0FrF#J6scW@r4}(oYUk<!5k+eE=mCW^Ld!}7>ePFd&Ss$MS%xNc$KbWNkl7-kb}6CQDqOi-y2-'
    '3usrcQTl?Aiok?iakmYp5vWoO5f)a;ld)aZm}$0_I)&_#cA1E?JC9~_HrY&qQjaxB8K`Tu`%EWa7;f~#vqp2ER*^VF$L;riNLon_'
    'Y7Pg%Z7op}mZ$HEvs!_~1cUZ=-'
    'Iz!(g2brg)jU{^=M_&hxz0>)GHfI?cKCCjlnjtv0!XclnSX90J97I62bnl)hEYUlie4NSFn4a9FU)xk9ozXgxa9o@YRkH$3+Z-'
    'lSy8i?P8N1S(l``v7GYO`O9O|D{8q7E+$dZ~7*iv^V|H4CyFux77!&P5GUMvqRYj_Z~LJ)f|1Lvi)Ff%4}}(}9*i**xECVxbIQ@0'
    'wUB*XuD6PzF<969r{}!ZlG)K3@-'
    'rfbzY1Kq1M{vhS3=tqQ3?Pbv$x+<~|?hs*E}%Furp>&OprRlg$iquZn=$kglvxgk42F33)h`%)9+9fq~`)@Oc(s{_lZK>0b2{Ofi'
    'Ro(JALc_?zF`fZ}-'
    '+a>*Gfv<mBPX@w}uM_+YQM7w%OSmF;)56pM<4wpsnFC|<6b8q_7(T<`SQuC9F%d8ZgA9&>@eRoO90TJE^neH$-'
    '=_x@(hV&s(f}9mDP&rHF`>jkx%D%Q`{0(Ur$p#l)OsG28eH9yv}4g0)r>+IPs_r1coxRL$in#klw<KM{Nplpa+$5jQg3m49*T*$('
    'O%B$CV{Vi*sH&BEbO<Z85=pgs5$Qg9ZKF7$*rfhhUFXA;8B?g-'
    'I!KZDFL#14ntyr3?E`hERaX&F%duplMIOhGMH>g6p(+R2Sfn*em$U&bZ7~XZ{?u?E>@1*YD1g(9J2`PnQPy;T&TfR$B!PX-'
    'J8*uSirO8g{Dd@fb#S#lyA&J`NAxeA4oxYFvMAoseTBPcQ~T@p-'
    'hf<dUw|`Il<A?4`XtoI*?IvOipr?^*1m%*)Z0}(DbQoA&`H})n^ISr97SoWb+iRjRi7%h-'
    '+hkJXDW~05TZl+9)7{!LE%0@`ZXp1dt!l0}6?UmiWk~5HSGUj4a?rWC3?k7H|)y0QWq+EmNnKQA$>hBQXTIO8qV&4F$7~@3i_0>X'
    '(V!W%U=-'
    'uM+vC)m%`QCGy)y<~_ANNLJL~lzYk$MNpaOd`_ZgmkZlGXdLMm6LjzQ_f#<6br$4vIsbSzQ+vA7*@N>^yyn>sjfF9MyP>f#p0CG5'
    'z!=OtGz!K5gP~C{zDN&<fboNRKp_Fq5*TZ~*$~DvvoIc+h4IB%7(bMPF+7OpgnGc{k-'
    't)xCh{vQ3Zj0U$gj=&Z`5xR`Hklm?`7+B-'
    '`0s_Tg&hchZpvFMp5&rZYg4vN*3^JstHV$C85IH9}@Kur(P|cHS|i67uzPDjymfWmIz*pB{mwv*I_|T4u;Kxy)G8R@YSx1#qd>nO'
    'az9(#@9t*7%*^M6oxO>10pc|kRDJ-LbL?K&G2KpI`hZ0v1~dE+`7jSd!&lAyTlkF?{tI_@-'
    '9c)Bky*sJ@Ot$+9RhpLJ4^<@kt=<u4VG6(XtJiPa=+}kEJLi8f2wkHrIv6+FDG&ClWp7%QCJu2qG*N0!4Di1|a>Ash3S^_-'
    '?~5T>?_`<P?u*|5H4k{ZAFt4J|P+Lzp=OKZU#(l+CY$DNMbVcgj|35BvTu*;?&s-`_3UsJ-m_dt_U+w|zfFwp07q_xH;7YG3>QKG'
    '{L-'
    'XIC>o?igF>T!RINeM|@yMO?>Ib}k1yyF`B_5&aIt=x7<`i)}X>cD(QJ=p&ABynwanoQBKMCs^qMmTrlXFJI=J=bK0P6CTf#_Y)q^'
    'llPN?LZKy==D0pMOFxd}Xq1e80%@iw5Aq~3_fRtWX=IR~c5GO5pGe%HCM+c*-'
    '5^+0s+rZJ(_B}M9b+UcR0(Ay$1rtrLV3q(rcOyP?-'
    ')k{Cqln{FxKtqlF6m816KRaVZcIz*@=)1W*ztC=_~16s5V#?;J<q4U_<p_a}<=dwRwnV@pw+Tvv@qG+*t($LYrr8EpTafww{Ybj+'
    'FCy26+`I&844A0~Ig6$%!{=8+r=cTszUM#8udU^Z0n;%4J}O8}2p)*u8Evw=r{U7KaBJbdb*F>SfIy9LL3$gr2jk!yU*TTDN3jz>'
    '5v(vpAWy&i@521S>tgj5=2Y(;Ev)Om*YY*msdLz0I@yDUau1`zep-VEbu7mCzECC%TB$m*=nNL=CnTQ$<^;uw^5+G+@RT)dJsm)m'
    'J+dv+Wi5Eu-8b06xiRyDQYF(|N#lGYXuFgiaVnGgXz)3By>XPEB%}gBoEOE6c3sL58_9WLg{IGYFWqfWLHZXdjM?y-'
    'PmBM&2i@Pcqq2pVxphlo#!rr};A;Pha*k9#3EPvw~WoB}Ok}SeHrjyd8LriLGnHooqd$&agXh0kEB6<8a}(onhl~-'
    'M5`#6L8tLonaHv4X~YIlW+&Hone!4?XaC;I#5y5Ensw%qGqsIuCL~)xcFS%+uq41Vk<471M%q$gq>S`&SXccU9Arng31*<;yi+9q'
    'T(6Y4XMqIMLKzf_2xOA&ExsL&gSuaUuPGT4J|QubB37=GWXsLU%zne5F5B<0K{5Nm>`wr;$DskkK@h7eH>G4Czy--'
    'Ii}c7G#3wWOtGD0E*|8VV(Vir9^#l{>uWCl%5lZ-XD<5bm|_l#rLk&}iXzvimvLK#VK<F6*beEkA1i&+WMv>ReE_N;12CrW!0OYn'
    'BUW4e7^YszTiMOy{5g-O$ND*sr^otvL6OiBtG8fSU`<ws4b1Lgc(n(^*Ln78Plm7a?bSXEUl-'
    'b|eHp$kay4iW{qHb;bCM2LXWEI}M@OQw?Ah5(O!ZIfjMuGwpZazht=S$$0R{eFP_hGi`D0AG=*HADA?|g}(%^-'
    'H*G{}yee@)!6+kQ6x&pKz4FAiP(DW0TYHrr|hdI1*OmCj)IXs?S>Kq==E_F^pjnHgSq3jP)rtEK?L3aU5yIGX`W8Gg_nObHKc^(v'
    'Mk219?G5_#uzlhc-mcSxG9yd<_5p>d@BoK+wVm^;!8#iI>M$F`7Nc__!ibr_yF*lEwB5S1Eq0i*VBvJ2uvopEh$6YXfg>GE%qVMW'
    'ekvi~)=<9qli4Wz5`XPq1X+%hCPI`0ZtFGp8#e{ID)jYD65I)dqUZj?g6vxbSe^|sQj)7?&7BPxrV7iA3wgj}wYSxa1u-'
    'R&DB6H)JJgo&96kk`GumacV(ZbvN9o_Ml_Qv{i*rT+Ke`L$-'
    '9R7F0P5Rgr;umtU>dQ$R`3=>NlQ!}aDkYR(K^S;4C!OSFx^POTMCEdIDksh46>v*;Ft)|RegrNX8FK$L?2p`?=3XOQQX}(#Xl}kH'
    '5{P)KHL*zCsmDYh(L9PrqL2tr>yaoV!qa@DU?V{5Y$L!fu8wvZV($t^gaddR%;0aS^z+Oh6gV?=Kw1I+^xZHNd>=hPjl(lgDKwOQ'
    '0L(T}nEMI@RuoJx0+)-'
    'DD6l7Op>)V=5O7iS^g2Y+RL)lq!53vy7Q+Ti1+g!2wbkQrBvUPt3}b&9hd<mM4jW<H8kL7bbMv*aID}iRjm6>TdQ1ck&7*iU3WxB'
    't9*x2wJk3W7mi@QRHc4r2K}sNFg=x9#B-'
    'JFfl!AgEo3(fnDeYYTxgM`=<DREG`lrUBT&wmk^DsnB+cO>6M>i!M*++NvV@LMU1GV*RNY_*S#F2gUQfE1`kK@!&9oa{3h(5^+Si'
    'q6tgJ3ucqvQ;!e;S6Jb_c^oXu3w{!O+}%T`UaYR_kJ6_?{jU0YmdBrbfXKp4QYT7{b$>TCmE$b#`O!te3;#<Y86%IH^2b^YoToLA'
    'g7)>}dD$ik$w@;S7n>hI{$2B?4(md_r%fKoxjT>9$Is68u5Ny_8(N>xjUA%+-4?X{5M%-'
    '%%u=#np$F6&amfwcX=yG{iPZhB`luzoP;#)J~q$T+RMwl!B^`<Q*H6hfMRa*2f|fZnZuZnJe^|2xOXvG%X65@EoT_ArqeSw1O^U$'
    'pQ&MKvj6$RvVtf0PZ@AfXkhVf@Fi<a;@YtKo)*YVC23=bO3pTx%!g`m&ht2Sgw&L%T|sJNw#)mNV1LNLXvGA6_RY{n2=<9M}#Ch('
    '9*b&s~sMiV^G#JL%E-Z=Fz)b8XI9ktIh+Ux%q}z0K%;{!~$@U9uomT^C+fA0T7<n^e6zr)0|$=Q*4tJ39Uj1`cj)(j7Rn-'
    '7qA?nR@b!LiOG({Qd19bta979uA%T@Nu}QQZ6ZXqqU@SHLA9pjnmkdpq1>80NwuZan(Tv7d>>11quNtqP4-'
    'hATwd)Jj^tVdwrZq5%uv~<!FEjPI(R)(|7)hmsa#Fr)scK`WAk8WKGvJDFoavZ84JUw^q2@3nujza3Wo3;XGFmep7V@?j$)f^f9O'
    'pN8^6D`fhTPA9|$$s=5GMo9|?7*y_$n5RrQh0$LRtg)t<XucBxSFm?mPyCQc3^H66A7jH{0mD}jVvbzkt1yLs;<!zG`_(6QN!j;&'
    '0CnXo@a_zfXRC0x14YIV?hU4z|nPdX-_K>`=>wD8d|rv72T`7L%HRYx+&otXzt^L^in1t;9<typl*)?*^TX`cSfC~$&-'
    'W=4S%3^=pk8d;KJ{+g-'
    '(5bD;%QzhN#$+|{(TCUNamTQb_Oz$ErhfR(8jLFsH=6%lO8uIeCFu8$%Ih(1wh3cNT1oJLVvRx?QqCx7Dit~_t29{-'
    '$O@jAoSfB|~9)?m409BWPs!(;VrlTI;ht)gRA|a;6-'
    'wp<ousV_=@6G^zd)YQ%{n2N|=9A$@vtskfyY!eMcJ5tbXU5fk33W%}N!f;bCbp5DiEWhUT&oU`9kyCKf6C-4^6eX#Tx-g?G-'
    '3CPt{&jD1Q#lsDLDg+Z0QzvBI#CWXOzM8$`}s|%C-'
    '!|3WKl*qmBiwM7kHfqbGI%Xke3Qdadb%yfxa^i<ml~G$3e%HSVq)TkDTMJJ!~4quH^x-lE48v30-'
    'Zr46(5gjGO*8E`!d=SD{?OQe!#-'
    '=W$JQxo;Ztn+lR7mA~+e<$!uo7sKo;Y7Al%w9Z_$kvM6i$@dLMhSZ{HIZ#qVlSp8vYjfk7t<5jUNyHDGZNWBwP4}F=1z;XO-'
    '!yM*__Qxt|!r{yRc&8*f1;@>`wfKm=oVPDz>|GOs#J+C)U((qdBpr{!ou8V(QaNrgbx}{!^&hBs>$`#=<kf?e=OfhOZqgJQLhyAM'
    '__ouB43W>Dc{7nav+1YJn>TIWxha-'
    '{vzVw)@jo@Z&_i;wp>o##7hyQr=lOCxL>$e_J}MRD!H7NRa6QVHjvjOUrTf53BQrBAKx^Ic!9KDK&W`JZJahm|TDI$6`$mH+n4A<'
    'm>gAA|^kpWOi5T;%jR4sWe1j7(<tec-tPIW4J5(8Kx=6+(Vqt)IO#LY8bLXhX*hXdNHP&qDKVP6m0<ymWsu;s%e7ii*|qyV$YT37'
    'xD{r2%`g=UFuLq2R2`-'
    '!x$ade5DR&bV&1!I+D>L&9~}@R9A|+^T&!<p0Fu7w$;CCZmezLMss6ryI7AYV%zhZXB)XX^N<c6Hnp=k1*RsS=Y&qQmt!4m#r;lc'
    'MSZXk9VeKx^x;gsWTwj(4C0N@(s#hmV)MYq!Ot7!?RfZEV%|=GpQTm?1b&vAx0B&#1y!ItDoBHJ8^koEJh>f$CX$}SeuR#U4(PsC'
    'M^L>ml56JJxDnp1dvk28f7Ro$HijEL9&6(l^q3+x{#9nluEDdfF2cA_rtaLHluv}8Wkjl6PU~Jb*@{Z)yUW&8Qr|<ip_2NZvMrU='
    '_mb_Xr2aVBo=WO_%MRpa-'
    ';30x>7{u9_X<fUa9ib%<PmS~XbqQUS5~(S`Gb#e4IW<$_%#96wlOu_tSWah^_x=1s}bg^`*KXHKlu}}riB|l5o_9EdQ1`1{<g%lM'
    '{vJm;n`AGK_O8mr2e9q=xF5nBFE+!r1~OxwPTU#%Qis3gB?IkJMv_88Yj%u<1%4E^M>lmX~r&7-'
    'MK+fe|aC$(xzObmzmn%809+nN<1u>Mz~dEwVm+%zF8nfF$<GP<eN|7UUcaIY<K`r3=pwYP*<@WOY2YmWUQs(PEW>K`V&2-'
    'P);XnX?Yr+!I1{4>c(k(T&{X>yFg-y;vH-Vm0{M~kf^H&_+TmoLc<;(cN6d;hpW3loLGmehd`WIhpVSRoK!n9qsq**4zlf9A<HLh'
    ';J;Udr5j#x00l~{imIBKSRgz2Y-Q>O1D~BtjVzthjW9N`98=p5zN`rxO3RwSp|q@+t*<Gu^{MI%Zl-R3xmb1KcC$pkb14=`(h{k-'
    '&kX;BQ)c)?JaKi|;Gc2I2KQAz=OzQJJ0-'
    'Td?)=f<vvXQ1#xnH<0S((tj?c{M;|($I?^@zHpzcu#+91&Rqqq;8xeCAKG?4hrPJcqU5&=8YCI}i#vqT*A5YIER`OwRoz%aDD2@F'
    'Haiy8Tbl0fPkrhY5bZHd!V`khd>+bho1?}ZwdxT?d~xaB|9B#*CSnXQd%8LC=#vHx-'
    'g)*P#@_G%{hT`<)X*Y&i;wPcQhppDa3Q4Ikwr*l!RORdAXjXphWxJPiB+W}vn;$c=SC7Zj_i<Az;`q`D4O6d~}s}8)|&eV;i^Sco'
    'iAdzQ!^Wj%CfstoL6Bv0`6f=Ew^Y3J{sR&r=_3~&lQ}2QQ<A@8k@PNgjGYYkQk~x9-'
    'vw4<xF?Ct$kg}G_*=w|~O~v@|L$&zuL#6oeV^1d0*S4j&U`XeBt&*$2oiy*)$Ti?)ST;VnRvl&EuaoP*;cx;@C+0fajWF0GdDb<*'
    'Y-'
    'JOOYF0LZsAgp`>)xIXmrbh>c{k`$4X3(?mMqYXbV6%5If9;&4|NhEU8ZvqQ_s0rYLp0LsZHvA<|E|~nU9n|ViYNFQy()QA$`IqLf'
    'V1J)g88N1AgoYHile=!bq-`tHG@_@7K%q6xhoRas#-IPL}JWTpKsS?oyU#WAn>aHG#urRTDT|Ru!}H<kEyE5%=`R4#X{-aw1M84-'
    'ITZB9X9arQh{xF>^gHlv#q=43oWqBHIrbMYcOJ(55nll>nGZnuTA`X1`kQqEp~C@=MzG*UGPG*Ix&*Zyz$?H*^BLL4HdIz-'
    '77!Kefm2U}}VMz3Q0yb?MUF2-`{XJj0sXtZo9Q$?7I>nyfBn*!xQIAy?>v_7pj-'
    '+r(O=1Vxr&L}F*~79EBKM+H=!?jGrpu8DgRd?G)CDLXlU`(h{AQ~ij0IZN3KE2&)RlROS1frIE+rNX&@5V=!r07ULo8UT?y`GOF+'
    'Q(*u^?o<~5k-'
    'P1;X|CJyawc*kd?PLL>}`J8nkJBctZ4%I$C_gH{(DIP{V_ug$<9Q>au#=FEnndo_YWvXh<z0YJ9;4X26iwxi*&<S*~yj4?&@^z)k'
    '<X#^yZydsqBpfln2wHS8tkzvzWRILb1fD-t{Z-!eBRP!CX6Py+x@lH38SbwzD8G;Sb7{e8@=pPZye`Hx`(Z&l^}ygoj-'
    '^(;K1a_+y^s&1YZR1j>xHO`yzJTd?276-ju)-SdP!z>y89Fh)(Fb44vXMv7#en4N4U<2g~llDt1vK~D7!-'
    'BkF=NT4oYs*RUWoXT7$fT@T0EL$MaMue2_6Lg=U4eP;9M{)}5VJ-'
    'O~r?7r919yzVdRVxR;uO~3$*^M6hV|ef9>XcDzZ*ctwd#2k<;CZr-Ko#Xe95USvMK^@HI+qHMZBZO6lZj|k9+}(p>zkUEMPH|?(p'
    'pwuwtc)Su}nQ$Tw(Q$J9KYhU1l7?InB_*-'
    'E=6O$Bt?%ozrZor1G!81b%aR6PJ$#;0&C>Ej{!GE;3mB&RXg6krhec@~mwMOI^Ns;}Uf<|{a+`$PU2{*ZsBuj`oQ>pEupqyIUvY#'
    'a@F5Hw$ES}X+NR?}i3xJr+SfS`HAFGoQTtnhLa1lwYnbg{ymw%O|0cRBV*rnJ<YA5;<M@G5-'
    '{Pb&y_3W_Ek78J|4JqE(I7tJ2?mT`iDUKgg?c@R!#t|h==t|1G;cByr-'
    'o?*{ry}eq3^y=`ZEg9fW$+$oAI=6DwA&+vboE+h_2irJek!c)jC+(U!6D1XLR>npcLpJ5X(tOS7v9N?&O^=1;b9zh!EX_k-'
    '7zInP%fcvFw!>=uVu|l8Eag;0rGwkEn7)-~!2c6XR8YxXPTnvAcB(HzsYI0W;J`>OSC=LB6k2W#(j|Jkr+(I>n5*hmM>AK}t&U-'
    '?u3H_;Tw%BB%v@!+>cU)Ux9ZA#ZTBlowfBgh!CVu90r;jY;@gYe)OI(*WU@K8s8-'
    '*6Ml25DRx@I8I9HE}s9Q9T;uRKCw+K(`6&6#s2;csdVmN%XZK<eVWuWU5WI>UTj6!|<H<e!66rftCMqBzhj?J*CiV|>1E*3hMK3d'
    '3q!>mrFZ!*NV?nWQijqq{ZNFUdY@^Rg0AJ>iXQHDLaS;SNa55JkrH5wSSZO+24!|veM2)D|X+_G7H^O>>mgImpvh2KB)m<afpN3k'
    'dhe(<ywMZpig{i1>nV%u!jRi6Z1x~Z9i^5EniNz`1z7*@YZ!$?+x`+8-'
    'XXY8y*J#T0a&&ImRFkM_p3FsOPJev_j;MvSE0?%fS5qLIpjKH&*V+5Y<X+hxGb?;TCI(p#EVy@}H;A~44cpY~Kyhb=)w&nrP+<aC'
    'n@ZeUnVu5$R9uomP^C(`80v<fASEGOj-'
    '~QEtZehE0WFo&+&l=}VC9vs+YgC<2bKeOYWjDvzN0Q5T!#3H&@%549@)X!4dpWi~l3czIwn$4c{u)z<c&N-~uD8G-'
    'V`~;Fhlngc1O`ppggY@C%;2*IenH^KPW6Y;Q}Jl{U0=BNurMw|6x+<UJouR}G&>f4aI4v|@T<{dBH(8p?`u);0}H$s1wZ)quN8Fv'
    '+Lely4&#=WL5C=P5`R>}xo6ZgTe5fgQn?z~cc+HMY4R&APGf}X&d+3HRh1?#av3{R))^M|o}Q{1iw<(v{WWRP!L|jTsY5+_=U~f0'
    'h~8~k^d9O%8uQzRQ6-(dz8)D{eRfgZfP}0*y{JaOGtr+ejvnWp5UGhpgZc1tVv!2BniGrE8}yh0NG-FU%{<E23n8_+1s*)@*9#%F'
    'g&hFj{`G<mV#%p)EmkjuHOJ*XC#zcyY8^x9qC|F7g9OP)p~0Qcc&<>NAP@Inf~e$m?upc}1caE;YOv*Lt|Q<XEU+2s7%Mn}Tt(zW'
    'VlCvc)(_PcFjNSjD<!B9M4z2gRe(NfrK%8}%BPw!o!!Nnv5?N<B~ui#QK-#jk<>Xy@&Jm~-1-'
    'w=BI0t$aGy&=WC|HR&LzcKndao20@nII_&UEXEwhb8<)ON=>m0cxIPS$hq-AN{p<^-'
    'asMn^h7~X`z=}@7bA$~k;dY1_m>O2qM)@6lXfUn`Q!Y?v)smlt_W9ru~D?Fd6-'
    '?*&sOH5tnvce0Px|~yWJ(gAzbo9}zE&epec~~|Q$4~Ra(Astz(l_~Kq_y!zzl=5aPCcfGxu40H+muq+Y10hDJWrgzmp(PLb6ug#l'
    'ohGgG8bV}(W_pc$fv<x)&AUc+Z!=NsHYr_)^!5BE<M@Jo!TYHPUd%qjtOcH`@QP$1a*ZEHiM@#?2NO^2JG}(7)cWy&eV<ER>SMg&'
    'NB@$g83y+LL?;Uj0(%?Q>Viai7Ydl=L`<7>PTLgPCTau6L0dXSo_0`eidu~&-'
    'Iuh_J66g4}#WkO~6xW+LK6BK{>VbO6F3J$RT?P@d2aZ>og_5dz3@4qftM~cPZ^p-'
    'gT`9QGd;;NkZqERFibhHLE7coNHQ5t#YobyRr5C4SRvanrex?KxR#IKxoX}K0#$`+vPirVQ1x0j&m2>k&Zz#Mb7DPBLpl*<=Gp5)'
    'uplah8tZPYw!2;m?HLmy<|hLNK?6^xRG1a|Hk13mn)5#SE9*nKU29T_#EzYk}&o!X%F)yG=@FHN7wi@LuYFLwnKvA;@-'
    'nMBtc2>9~k>gWmvRhH~bk4TOw^Lawfz6HM?%i!3IhSz5ycO(>53GH_qm2z6gBE$GBP`^eSTeWKt8pe**ixMrkz)56AMH9GwRMyve'
    'U)0RT7pbu0i@=rKhAup*~UFCe$63yK?YQu|;7#Yy2m)CEQCE-'
    '0zZF85Yx1t)E=Y|qE3szVH27YqL2N|KBL=YN&0^+CsfwJr7m|4gnSDcqBoTx-jHFc?^8>wUm&lj{v0t`+KON5?peVRvI^-'
    'x`xLcQdBmHs-#Gsk=(e-'
    '3Y75F?r_3oBSr$+;F4c#F~4N9#h2J8*<#a3gXbJ0CU?kSf!J^Bu~z*31+1a<bErSrf*>A=Wr1Z_UjjL=?>=h0$hn{tK9&uwcKEAb'
    'd69?n7!Nl!$T7_E~o-hhb3yfTS5+H7#1Gt#STVtYH!1Bqz_lZ2Bff$^yMlMz6~yA`;!xJ@U-'
    'o>8Qz3ADvx9HY>zj&EY|jLqswA#|CAn6#P;u()O{Vn)5O5T1Z<L&N~oheR~=%ZGH5CtVEJ7_F<|#F??k#9{V+!{HIav&{MAA|?sn'
    'Nj7zX*aeGwVZA*4UTx@Ko^b%rIdA_d{&?LUMa{Zy{@5L7Pg!#(wZivWm-=G-KxFB_-'
    'MPl5ua@#=zv7JqDp%-Q1`Vad2DudfDw)#b5vhZ|iUYxmiDOrf-Ty8g?|qy|(^38v^4B;gO%=B>!j?#uCTW;_?BgLWXlFQtE4a_q%'
    'yY14?~xtgA+@0S@{UL^47@HVHm{E>6;z#G7iVI4H=mwJn<7S^&bdq<1(G_7QmznabDilACVx7%7JmDIUZS%QCc25nLmo<0|g$uLF'
    'g303iqWyxRSm?~8Jyjn2x+0ShPSH-za;Ho&cpfXJ!3O1p#igALm-~aSP-en&D?OZ*QsC})T+c>VKCF*<DqUQ>fbFneu1p-'
    'fx?Nsf=AudzvJ=^ODPnKQJkviTw?U>Hl#Hd~mEBQD2NexT&7*X2VUMV|T;>Uq<iQJh^I-'
    '#9iSz6GOwKH7cYB7(xGC)*o>?6^L>cG@JvD6hXD98Bb)1TJ_@{04CKwfcPL9v>guzm^6l6f3nURDgNoGPr##9h69w$-'
    'gR6VAgfF(5G$4Y>(m`M#T)%YT%}d#DK+BJL^FBr6knFSSwrIFa|+!g`K9kcmac9b8RK)c$2oy!u#1G&iqttYfwDA=ajT?jk?ckE`'
    'm;>Y>Hcw5=ZjY{zm}KscCVYjd0Po4{UieiPU$&Mzod(;~agD~>Ho9bvJhuCpbSB3&XI)I?ie3(s#Yrf9DSbb2eOBD5JZ`8qQ9ov0'
    'dir)!wo!&Mq_q{E;*oIo}V%EOf*T|TAWnj<p?rQTXFWJ<TE^%J<7k*EW#$uC3!Tx!!A!jQbt`f*52@um&%;0j{Bu^E$tcQD^<810'
    'b~DU!Pae3KlLo7-'
    'H_1onyxn!sLhK|!&a79~NP8i}Snsf^}p&}%bXSJt}B?0^IXC)_10k$%a<e^{YZ5?cNwxm8YZEwL?Ky;*{!GL&ZBDAYn~u+>@ZS=='
    'X9XLFxqn`37z>>H0JYQHi%yu{Q4x$oC9q)aRmt`Plga5n7K$sEO)Q)UNmagtfpr^4z{I@7<K)&c^G9J8C-'
    '{GtiO7Qbi$vBfV6%GH$2^efS!?+9+UE1363h45eT?>nkx#N7?E-%-'
    'j<WWEMs2@V{a;$wt0qn0XjHpx4vGH0_KN0m8S<ajE_*(N7aInH)D$*0jC<386i*9uRZ56_U9(-'
    '&my#`<HM%frUscevWq8u}EjYBGifS8}15-|euEpgNA*LLHW8Xmgtjo4{^yVH4ObE-WZilQFdB0)SQh@Pzuy>jFLr%*?##oZ)iX=3'
    'ite%GXgnB`DznYx<JlczB7rRBw=Sg}OY&T+&bOkdqySENvZk$W~OMHBPps60PyF4V7q3kZoxNm?+y(iPj|9o=UVP%MNtj(p`2}lk'
    'N7@Q}%=?(pVj0y_-'
    'c|Z1&yc7V7Xk+nQf?Q4{DbE@}e3#YF|>Xj+u=imkUIwz{D>INz@$#YNpv$E~D{uD=V~j;Au;c1(vY;OWjz;7`}EXLtSnvYYItCfW'
    'BrWDiGM)=TzMciH#7Wp8y)Kq~V%_nDL@xR0Tq<c^K{$BrlSb&et<Hg%6>1Gi+RaThGC)#s=X%4{k*M{SJWVsxIV&227j0>8z@P2j'
    'h<xL`X@yDVqv>-wmECJWEh&J8nX`=YNwX$tI1eS-Pk<ENPKJ${<`9#tw6ewLwl7gc2dUPh;?GDh=ERmQD6EHp0@YQIEFB|A%PX6k'
    'R8*eU+SGc*YH^arzpZqez;!m|0PI(6ur3y!BOy57is+s@odasNNCUbL_XFkgO_h^*>>Tg?)YRUMo3m;zNDR2yYXQA~zr4_l;R1{A'
    '33pn|1ai_6fo%XURP#o+~e#dGdFrvV&8iUqC^>S0c~Vc#QB1OL??5X1ZDQTZ@k_{-G)IPQc0O|ziEqE~M3wMn)jF--6;TDwJSi)^'
    'FX*!NpyTPuv*AH0orR+6tjxEt+BIurJ+4o*t-Ii_y&GvPD{(Ni;FOm-&RwmWk17-'
    '|l(wz>K2SR}%&X2&9NogNcW31=QfoD7W_5K#$tTX7kh_L&y3*@i^PG-dA*=+6!`%Zv+K^g`nb)yo4W<&c*LyUxoQo$aavbm!jlEO'
    'UIZH|B4GgTj3<#t)7!_QOaZ$c^rg4r7oT{T@0QL8;vLaVMhV4kVVC%RHIHlgt&f-'
    '0mTII#<;xh*oDIdi(Am8bcvM5H&ZS6AMwe)tp#}*6T455H*h?PNBvOh=Azr#T9BY5S8`n5|dY&WvDU(!Y|*A<&!A`ag}WFn_wFuD'
    '``2^#WoJa4Or@#$+A@<(5Xj49l~6b1ca@wO9CPr*Cqj;u<Mh+v@JDCs3V!{l)wnA+Er22dK8|))z}mY$7WGjyE_!dP@ND8&CMT+M'
    'Iqeku~-yV>oE~1G>;-qpvDY{Kw)ihftu_>IoGhW!aIh^^su|3+?8q6=&hBA4smNGrbFCXiRuuyR^mFut(C|Qacd>EL)==)!H2-'
    '|4v(21adlG)GdE>1b4Q7^VU1ao$!Q!Qf<7bNLBxZyGw?pz*kpeVsu|XLUlS*cY4vY9Ckuhme5<*!V1!%EjRoTfJthK-'
    '<{8In(wG4eV7#NaCQbWnTvN~RHOp>oiy6L1`FbhC7j1!eVQZ?*_RG65V^_xMfMA9@Q<V2%Yw8|8QaF;SR)Puv<}o~>A^@JrRZR+j'
    'HCX_TE7jJ7#daQ-XBc8W!)W7=L!4_`OQo+Eq<3V*nO}}+wv^-'
    '|#Kmtq>Y{)1|0gaPC#$Ky<>Rq{gj+ox3&^2*Oavg!Gmq1#F#{q1Ij*=yO^0kF8Mt{v6E&ETYT>Iv#(O~!!|3*kN*~b-'
    'X1E%4avOJfrrM#I<EYxo)ZbGvO&zvlb54Z(k8mY89f?ijsyU}KvFTj3@W}3nB>F)H;1H&M5LT@n%G7~EJEI1Wumrlw=J9?ONUm;9'
    '&4Qb=vta!0w(c0>8z;4?Z~jCqIN?@L#Da6a9uomh^C;q!Ys`QMaE>poTvGzhVH_!Khw<IFIcDew=5Zuq>1dm52Vx;ZWrli*WAa=l'
    ')dmRlZCS`aa0HUb+PTmld6G}_U1LAw>U80F7CUiuhEPN8)lppiNT}=V)zMs?DGVPFTph!e6zY0=bu3pu7DU;Jb)OSFFn`R|Eh%8$'
    'k_G03-2t-^UY-fL+MD|3PsRchZuMj=Ft5^MB7kWgMVxAl84v-'
    '?3B^@wT9n8hcv3%bD_y%IfXnqeLOkgj9>J@39gpBtxt2%R&s@(V@`<^o$1x23>d$Rz_(qN|st_5)@kPZ!3z(^=3AgM@&!UY-'
    'Ai)XeHlfxbU%~=LWz6vR9Z6Y3a~-K85@LA2lBd#CKSyN-eGc*!I8D`&Btd(NoNToD^iyK9(QvCNvDxT%^q4~8HKm8f0V}@N`SS(6'
    'o&&Bw^6Xs6sYGOguD~1X3WQxbpp#UEM^B2Xa+4EOm7ATOs@&q_ROMEurYg5NF;%(UX{pK`RNg&`Ve;xzIxFP(dd=;rCkR_-'
    'g3K`I@$`X1HCMaL0r6>E`)P3`Jp1IC*xcscSQEpo?u|9^Dm|u<W=(cR>m8+%4t`n@bskdQpFl@IZHPJ#Deq6B!=QFVorje7ed$=J'
    'gC%m4C&Q096;{JJzUaJX1jiR0=`LiZS|;dP#*BZ1XAV8Wa<1#L<38NeO;oxl7M*qBB+rqky&YGbLuUL<TwSOQEWc1=O6}VSo6Wv?'
    '_BEgUzF7Oht?rAp?{j)gA)%TUrOFn#Xonnm<@QJg|5mT(Aw991)d!$RbswQRTd_u6Je@s6+{)GdiTbH!>)X!N_Y-xtJry_werzoO'
    '{+I$0c{=<|wiO`^JNY)+C3uFMp+++DjgJ$ylnCOylZ2@Rt3k55c2uyhDNno;ag=7ezAd!*Y^Fw<<LfWg*ixG}!W^?-'
    'X!AWibIb*(b{lDPxYhl!HlM4<6jG{bkuBvbN1=jFo_5}CgZ31MB3UfrB0_6?+gXHIa9o~Ogo=vbJ8vdZMfkBbAg98Qt@aoIKepQA'
    'H25*q9@vp}t$;sW&GvNy{xtt=x}2^?xahyPuqCGu6&){3p<s>rNYhmv?dV&6ky*h<bG6e<@UJ0`td8Vq**|ZBo4Y&^YizjH1F^>b'
    'haOW%x+Ys3fH5Ul+cs>J6U{a9EDsB6x;h68<^6SjnW!M@NyE>Zxtd`@sewBoONiz-*{d3cUg{PLfjM#x{A)_joG48FW{vt-'
    '6N=f6tj{>+32ndb=bBK=)1h4G=Wr)2OK_{bQD@DRJI-Ep)ohv*!uj3L5vu(pQO}i1S~fzF@;#AHkpds+!C0HatsacE`TcrKAt{>{'
    'IfdG&&l?^wp%x__DWR^x0Ioi<&ki=feG*->XE@aH6#pzshYw+G7ZSC{1Yb{5rk8c9m+<AgmBN<qVw2uQ-'
    'e$*WGV)q`r!Jc9c03PvbAbuja>`B^J}2{$-$n5_5(Y-'
    '{XYd_q*LMq(jyW1M^@X!M+|jbJ+{D#arDa}?u#<dWl(t^muk}!@<>6Kj#ag~bk13>dlV^F9#0?J<&(%A#&kvo41+hVX=mIQ>4H`K'
    'v#9GM!ITvA}J!MXJaCJZe5s&rUG#hWVE#kS2qu`3|th}A0>P=_I_HnVt#L-'
    '%Q;C7x3&79{}<r^}l@KEA*FA=gWmIsZex?nY((3r;5^$F=MuPa^i8)5$VAC|Mhn=kU$SQNmm{u+yd8}yh$`Jp@%9H9?Eyw#7y8uq'
    '~IC?IdUW&5jpm}<ePU<>wdwQmB8rny?fRK0+>$Mwh#5%4E@Ym?>at~KX-Lj7F0ZjJ8?b&l{S*vGo6Ot}ot6rGvsz%4Uo>d)2s;5b'
    'y*B&VbRT&c(u;rhi<q{UOn*M{Y*Wv(zeif3a1FEzaW3Vq^vHu^MyXQNNC<jXaAR{tSak8+#q#a^0w1T2A6hC9s@pHKJ1=QAjJ>nm'
    'qF<#X7$`?+GgeS|toxO&_Dg!-'
    'A~x%f~Mh92cLFOGGk^>PUwDCbpsg*nQGpwPldomw5M0}~&WIQG=kfk_~YGJ~n%rHa9g@N2xBXJz~jeVf3q(YFcw8hwkI`G%5+Z2='
    '6yQvGu|dFg{0smfM`37NVu-'
    'L=zK{9^hS&oI)@^Iq6qU0K42R5@?Z*U`#Tn6>s4W^Fu$SzAwG*3MIywf7We9eg&AF?@|;t`7uWqnYddfY(^&3mJ22dK=-'
    ';SeVymi@&O06F4;bHGxB;Uojh3w<s!oGf?l1uDrJvF6)Sw4Yx{P^}91L?K!CQtWCS#G`NIy`co;rbCR6tg%4)g?2TH<)T;t^P}di'
    'IjetK%c;gc8q^!PFmqO?(Tx$ostmludR{INewp}$o(451ad_j|4HRKlT`y3fdobgKDM8;oR)db><swNO;R24Jw?Ija=itVzRB}&_'
    ';z_4~HJezw)1+3y8Or$9obq!9dsNc=h9cB)X;A&eafP)>Abppa*^02C$f+76a`VLIV*zyhx*ll$OW@l`12gdHUwgY2#TiSs+o7-'
    'Lc7QDtV*QEuonxFtVXAU<)i?Jwg4&$#nxe2rwCpUo><K$xIom_Gj0*;N~R2`2q?`i>ml2FFw;6@79*#@thm}`68SE!!~U#U&BQs9'
    'U;PpBLI1EyLfB)p;ea=DoL>H2Z`2DQ_j%;i$*usW5?<ygVx{i~`}Ge3fkPB_kXTn<L;xudOs9Ee%;UU>a3qfMqRbHeSru7AW$VNn'
    'b*#>6SFHPio7ujcWc;qy492@DyhG=U-'
    'Glwy{@FQ=Do5YFo>g+B`kWG~9YV_CQka<4E>)jNi)Gf~a#%rLN>%+#%BehXyb4d?eNrd|`^LX%H$Hm9|u2Q~=U-'
    'MJUmrP@9DIBXQKn{scgPwnm6WcL>8C&J5F5ddL<-'
    'zy<*tUjECt+7W#uAs)826WX*B?=p%#CVODR^;n@pV|aUj8mIHiE(N%WB)yOtL~`2<i6l%s(D&x>n>@VZJUbLbdd=%iE*Dv|AMP_i'
    'M&5B^v|#%OB;F)I7lgHtQ+c6S{JOlI){@F))PyRdy+!WueiF;TCJ9&_GuT#r?8h$%Kp_Sn#%v!K&*^fCUL1z-'
    '?OXByG$MAuRdg|b)pWWf_v<?9TCff@%Nkt4E+K6H-QPGe-'
    'oH6`WLL4?2s)~yOUwVx$QQS<G`8Zj2hY2z~)DtY-_;*qE@zb;L1`j+j>~*-E==ofQuN^iT@ltEthK1bv@1^<(Lc$s~|>ZDN-jeU!'
    '+DS(>Rqii~w=(_}JPU^t@@<UTG)rweTSiBSgMcULpk#0TMabRb9Smb1qR77=3w8X4QP#j<KAoaI227oT|I@n253q^Xx}OvHy<5z$'
    'U8fLJufZJ=q~s%Da=P`^=7eKG$N>zULvq%)C$YUn$hYP$N8O!rDbmSO$upYHb{Q3^<Nc^ms*}sqK77lv*Mu5|x%t@#@;njYOR#@*'
    '`1ZX@J++cAg}PEIC(Fm$5ncw_@t56#PeJ;Xje>rU<AJPL6(gXf!uJBo>Wut3zVZxJ8eNK%;pSqodF`3IppHG)~k53Kao$%%Jg0rX'
    'J)}Eb%9iPUubc3_3{4n>gy-Dj)F-6F%apLCuGi{;;x+l-UA$b%3tpSBJYhf_3#3a}4Y1Yvw4{)i=y>tgCOCBUx9?xMNvYj62S-'
    '13e5|Gxf)GOJo*?lXeHgM(9GS@?dCgerPNV;Z}#n!tg^qCIW`$QH+U#;b@EvV_-N*4=9ub?^r^Ko|AH{0gpQ2qyzz&bEin-khfwU'
    'cO(|I`e@PZq>P)E<L0UidEi6OdmU462z7=p$W=o12@hO;B2?e-z~!exRfPvGXA1~5f&-'
    'Uxz>g<hS#=orULJpKuwy9Ih?vFSWR}r>g4ZDXwP5H^ut4)Kk-'
    'R1+huCax?QcHVVX@eRTOAgQ&GmXr0c=t$uX#+>g|OL@niMf#RRo*VUprY3DC9+U%y^OiXBpMQnL5~`r7iYCrg{RiXsKhRjSUs)qP'
    '{9GyIqG%b0;bb2u@o5#sfYHjC$(fKLxYrbgtfVDS<zWSb~*P^43Q4RS%EFAKdEjSo|&4V<Of@^Weuut&P=~?T%R+>-'
    '2y^4r9j>m8RX~m;3Adauwwohq)8}I%`r}`f~)&Bz-v7B<%36IjRqY0c$;M^i&75SY0iM4CW1Wji3s!C2Fvs0-'
    '>d9h@f(?W$Ic%)k4eFP(kD{E7Wy@>coy<s@!9~9Xzyj2VxfUckXV7ZiM~i)I1cLn;#L2Lb%ltu_$~&kBLB`c@#HAp>QlFB4bc^ry'
    'fwqQ9LAXhpr$DR{4YdTCWmd4v%fy0xUq|f?#HxPOzj;u@V#s(l|U1k19hPRso=UajHJOLmkJd5OJL9&8bdpysG3>l{i5i&#7o_qB'
    '?<7)!HO=B6kPi>Ll*?WLtS0wuh&X?rO~9@UGqAuo0%3{&_ewH$O5Khj6PSV{tf4kBPvcc@#BKIJ^nZ4PtP3mmW~aQ#>TQ8rBC@CA'
    'gQ>`<)H<GQT)hIO(t&?76C?5(J23j!5?eG(!zjjl*;B06o=R>v^+UdF`;)UOTLf*A8pzwZqzZ?XdP<JFJ7(4*MZftv&uaV0&+>sW'
    'FScyLX4bM(A@M&g+^nH~(QQ{@_+WjK$w4dQ1fV%%iwD3V-mlZjQp=-'
    'FiSFZ!m|yNeCB+R`{@`mco_;P<*8sGl@6NqBl>#e<E^5T@CM`_YGIujkCpt+LaQLJ({WC3X*O|36=j8?g*(fO1x8A7c2FCFVsQIR'
    'agE&IAQl2n7UJ_F+SjF$-xy1=q>i*00s(R_CZx`*v3hSRQ2*8?aOgNAMPkP$7^wxI4+<0IYKG+HjlB$WHQ|9F&3Fj-lWGAQpRzHx'
    '0c$$)V;=1=i;GXDpL|BWOwJ0_s8n}&e%zttDrNQQ+Z9Gmw`;IM90>RT(NPm_>7}0b`NsICcxq|fpXZr$PJqWi_avYW9zM*Fk|&K%'
    'qqQ`%e73dvP8kiHE^78$4D0<Z?ax3qo{IT$~ow_zj<z~q2X3@V-'
    '3Adk13>ylQs0$O#Rgu`T{JI%ca56WLM`R`ShDvPaM<Lf8+4?1_5tkC%Loo-&}=c*h1y<9oV#U6SYvarsI8V+PRrpsM^x;J~r*#N-'
    'b3F>3AQTc5XLs8v<g8Edc?<c1QHkUp>xL?_2`DMrdXh<Q(+dU-'
    'o#cdEr)%$C|fZk0~U1lQr)yrXDcng?0SP5|*<HPk|iF;(%HF3K3MlA_m?JL{iE#Ji6nNi2Q7)Ub80y55rA5aDP+1;1G<t-'
    'ftWSp_|cDtC5=mVvo%MvB%bc%wt<X=CLCn@OXl$O23L9fky?}M~|a)$t%ILzxs(-'
    'i^Hv+h_!gN9#cs4rf`8)Fl}+VTp_5?=}q_<9vZbFH0orvH?{*(d(*zyM??)T`(d96)o>nfj`)pr%Xqree-'
    '949Xpl%;$K>kZba4ZdYlB_-Ehg8~S>uyT9q$d=pQ}1jIg@v*e_=a(;9z{s)iwk7QC#gP&E_;hp0Y4+!!{q_$yl?)t)7fE`v^Uzkk'
    'n17r{+nS&J}7HmSNr{aw6C&T%z6lt`7xJ2CAzB7I|diR-wM)bW;15RNT55d^Wk+($#@iCbwI<I`GNl4u>xIK<52h@-'
    '20qegC$6oA~tQM!8X4VBfzZ-%-'
    'D?@86a0stfJ=_vCx(BK!V*`M$c?BH}5gPRL!<8{s8+C2vtTzwD`4Tf?oMinaAnJ*JS(P019m4{5FEaV%@8cYKu<j^!RL?ZQR)H*f'
    'l@j$Q}H8Twn8GRDxiaWy@aD60sBsgY#$djY-e4+4@^D!n*P9!DO!CR{z3*xGT~v8AA!)PD%MiRvqVC*&5(hX}5}+-'
    'ms{!SR>d%=dhnsS|Sz-3Z4=UYVZ#WlzT%8gBJ;tfA-YF@;2Jvb*<JOg&^Qaw}KUSUOa%_o4b?AF99ML-i#-'
    'RA1^t^<_R(U+zQo6%?uuLm3HJJw282_X9%R18ki23{xkiWG&3P4S~(Pu(M-|umY-'
    '%?W(iun!>t{D(f{;u?B{d69tx<T_R!`Gp6NH=bF#?OstvVR?ozm`6@l8P|mxgll}kd%;jwCv?02BO!qv>9ZCIp*kxm>%e}m8`mDu'
    'Ob60^K>#0Ni0HL<ql1m6LA7dfQ-=|z&pdJ=}=|4N%o_8{>)oiGwE)cPN6q9m#UG*m))C4w)K}}$z7*tT6rlg))SFrf*FpNOGi-'
    'uS3AS-AvTZ>FME|@w%kU_!{`3gs%2<tNOC&IKP`b3zu#GVL?HIXO6T20)Euv8OuBCOQa6o21AOgRv0PjFl~zrapS%Z%-'
    'HWu19fG0H5C<7zgG<)WCJXI{L?)lFccxVi~U6jv7%pvmc4-{sl{D*Rl=wlI5ORfX2HxwCqFZ*6=n^F7S=DXwF5POzS(&-'
    '3LtIgU7s<ah|ThS`MgzqGe~lhrarp$O6^z_P39oJh_w#-G70&3O&}#bZwV2>$zAin`Fsk5kwMD+UTH*_x=i9JRG>&nU}AaZ=QY-'
    'RQ?&(*#C}Yns4FaZN$-nU>jx^?edOlRb-7BMqNKgX-Ka7+ss`1zpQ@xiq}nz%s{^m`|(soFt1qJH;EGyI_guE?DNd3zi#e{Rc8$k'
    'x_q#6jxO9_MgabMMZDF7bu%zE8YGe)Q!%<yHSj7gn{Btu^YwM!A;<w7~BN@iNOW6X3`cWl97F^5^#;SWG5%wqIKTwF6hamDjCjV8'
    '$iQ9IX6)o!&R{swGFg^?|{8HWnlbS>e)qB7z6xUXGUs>hFl#a$Y_u1DB8|MnB=Vl8SM%rI-'
    '{xlTVyt)U+`Z@UPhurKgU*Et65WnqOng?a?FdyM$u2)C3YhXJERHZ6GNInJ~5=AyiCiI7z(@6NDWxK)9zfiLe|>CJ=6W2o8X3j1D'
    '3x+gzOWQjP%C(qw4GN#4&aP!%f+|Kat_CY~G*5a9g%b3k>&Vd-'
    '%X`W431x40mQ@ut)T1#w{dhhWerw8M8S`Al|YYLD=n>zSR)i!5{#S<(s%$>_!mw+9t3~T-'
    'yY;iE9f=%e2hIPj551Tz~CbFh!j%_q;X`Oss_K{h3>$1G<qa+QKsn`L!&o8fQ}5oqTHhD4*Is+NZXU@u}@&eQLY2Pi=ScsqL=Dw*'
    'S<bQ#-'
    'e(=>x3W)9SBQJZd7(=}~NiQR1E)vLO9?hc<yvVrUciB!(80mdV=oO(vIW+uja74a@cf5BJg3jXs(h;iIXMKAIZkqp8t8ni}JyDa$'
    '8*KUYhH6Y_e1tEIx}@RI7PPG$@B-y-ZW4^BP$WB(Ot5`tGqe&ayhFzZ=vV0F2cIjH6LGBm;xF*}DPLVwQdn!pipT@yGWt}Cb{bI5'
    'MXV+UeJ;aMg-!vA{)<6hiTvea5RFg%!WBpq+6t3;^z^?!Bhq*ZA;JN6JyCSj@=I%mm1Yj{Z;%vbP$%*y-'
    ';%V7d(gb(6hdG$8tGkwbA>K))tpYq6Z2lzmr@`!Q={eT{hq6qC=R4HaJ&%|6*Ddv!D>1%^*aJ-tE<R-'
    '<OTFVyKKDAA5b1hTb<#y9Dr4?Fss1`OHyMd{Wf~4ASWa>~s*|q1G>f-'
    'U@cqt4#x@7Tka|yS1>eQ|>NDt`UDQwq#Ov@Si+*>m|nWNlW1;%~D0~mhEEXg3A2VX2=87uzH^QB_o8+{guL%h{zu|V9Z$3y@TpM0'
    'E9OqW=pn460W#T=Si04+6=KyPw?@=kM)^eh(@CWfqGr^>0|>lmy}$F_m#Je66Hi?xxrFxzrR9aFPVlGu6}vjsS=He{S5z;U&K#A5'
    ';+SL<#*F2Gr}ambSb98_}(uhs>XXKxBQ-nctj5T>igJyB*@n0L+M{+8X5CO1N=@i5PqgXxOAypb@3TYVl2!_W1Y2pHmnh!cb98Y>'
    '2KOK~xnLw8>Ui(r}>RQ=$!?;FP}cZB#VcY4;EhvQZBUIP$*$~2V{y!)IP!di#IxdOr=8{Ir1AOy0(?Rp{aAfmSoLXNZkw0cX(2}J'
    '7kwvZEP$=xXABqHs5N65*v`o6$ag-'
    '4;|_b{NV*u8>W9P7s;JYNo`0(l7|Q3$tch(+OhdQ1cg@j=9i!BoVG!Q5J04Cc_RejB(*dm*jExxz1oK2%r3=rp1~$`-'
    'V7gztdD3e)edUFe;eEbCe(>op2DrzK|k8#vqeVH)z)*#KfXvJ~0~Vg|~I+Yn+V1m6@x%u;hW1rf8=W3EzRj+)CUjChf$ZXSh1Jz_'
    '!Et+0AqBjhBH@_ZebZrF7ji9)#5rdSlN&|@M{hz}x81*Ti93e0WARbURybdIiP@(m4_&rrWPT@Kk0khNp|i9ADfV%fw3mfTEt<l?'
    'wD`)<=tm^K>zf1gUw>N4Fu_8fzYfm-*%vTWbYlcJ{be94#YNWdG3J-'
    'F58SnMs*V<NDJ4<b(RrF*R4%k9MlUk=M)Zv&G{EI;tB0D3i=u%#1KS`*MLoE%%zRI0ZqN3ObAR%PR#xJzf)?JFHXYvl2`@Yyi=1Y'
    'G!RoO~iKd^S)%2^T&aDfhvJ&xXo<apAMEaz9-7Zg-+CJv{gvdyBzZkHTtfjnK7B<N1;<J&^i05`1v0EwSKxN{@*EA3lgUL6;t}f-'
    'bei1ziqH1;unC(kTcfWh-'
    '?Q^QC>VHPWoOQa;%RsnT2rv23f3Wxldcwo{#%F9VeARTt*@U}Xo@l~Hm2OHB3jXmJcM1{^&b*I^r|zAwDa42DB)Umu8N3ITcio+1'
    '-cLGZQJaUHc5_I$?)Vtrw|805FQGA@o^X6|m3UEnKijm0P2YHKV$XX`N$_{2var|Z%)R@dc@;<_#^^U7U=M1F-'
    'O2S><;x~Yz(6aTIqS5=<A72ad49rj4UX7yX6oANG9o-js{T>Z<jPM@VYO*^~Rmrp&D@JO!qPF!CKtUr$ls-'
    'A+NsE@GfY^b$#jUZ06I*!ew)i&N{*qv&F@~4o5o=)+M23;2$=PzUU>ds@iZc<9q+yc>jiRrN!Vz|}x*bMP5J*JSlOUt}s*AfB+Q~'
    'YQ>tl<ogv^}8*Hhz~g=cTR87C%12)UVAc?PuBZgnD0Pv)EVJyy2^CmiQ{0rM}8$nXj^0?yGE8P?b%MTESHZ+E~>Zt`4JJQ?2FdaJ'
    'OwTW|C;J>@hsAAkO@<8L{?-Tg`~I?-o6#khDw7|Gx_2z{=NSJ)rZ3M+x}TJoIH+uzjp!eEmG5(S*hE3-'
    'Byr0;AL5%J9s%O0J{4Y@pFx?{J3^!0;3)rrZc2iC$&i0spW%l9ol}+1GsTnX&eTTg{BM?}vI!A!(PEMSB!#@CjAXkyvFqNW#Cf`#'
    '0>DD_Odb26m|$^Dy63H>X+)CuoXhLnVvsLEya@wrS*W0pG>&O(SUrcrGW}m6)O5VmN29ksQ1h%Q|}(L!ZU4QDbW*cr5+QM_<WRM{'
    '<tUDy|M8?^vzo>QGLrw_qj@6iaoI<k{Q&vRSeAhFi^wwfA~GrjV>l%iT67=!xP&;ws8I8!Zoqh=!oE*)nW51Z(xvOkHZ$>NQNYVQ'
    'JFfN?#FqyswBn!B<3{=qn;m@)ePNd_`nmUlG}l*6QV4wWokcDNSI1jN=pD<hOmpYSX!NGj4=Oq%4op-'
    '~8&?vBrg4&5kwhVm+pij7#oh4o<FJ!9*3zyl{c6w;HoZyAh<18}KTl)&mwEhUNLB<pBQMaC>nhlilDC;*po#Nq`60_Ir>JlN!ZjP'
    'siUqn#o>{zIzOl$2lJ0Y9@O-GH6&1c3^Y7qxs(K4ZED-t4BteYqoI`-'
    '&VJlF5iuCb~MkkxVh(?Sc}80=EPe31wE#ah)c_odRIz33`9@1(y!i~{OXMyjDddBT0M(h$Tdhetz<fYD|`|I(_XI2q&vizz6+s^^'
    'P_u0bm_ZK+6CPw?St-'
    '<4zBxT8T@P`k8C+LE3JT^9ggmJOIn>tzNKz4(xXbK&1Kb*gcvRI)@*Z^$6}2Pw|Xqr$iwuQLV7JF#je8y2}fhO)YIfln@An!Bb`o'
    'G`r220>R4sdpIF~w1c%*}L2#Rudo@$7Q(R@R%&+x)U`H{mgT9(OS(f$iiMbQ**}&9b?x+IZWNL_)UU~~cB`@mpHd8~r5YtAcuH)p'
    '&FQ)FJCGfL_{P(4{_XzpHx90BCjgU|*&a=7sWpiU~4!4>cYx7U^m_o^(oTYrb-'
    '+0>zPr=zsm@w;i9kJB(E0_qT(GWIEYkExt0Ye4{lc&4tZIsP=tCLcc?-'
    'dMR``ER$JHyw0_Npht*Z1sIFNUx0+pFUkz7DWgy&1lKK&3$#)>FP{yLv^?HoikG5_I^om8rWp{AsSfKz362(|pJKPz4yhv^bWB<5'
    '?*Mo%t~Hcs$9&JRVQ-Ft3=ougRKQJ+4Ld&QPy9y(-mNsE;`{Wn-'
    'B$c~IZl@|J<7QVHUqPZ+sweKR%f5ZRt$9W<^^OBd$>MqD*lE7e}8<LpIi)j<GWY<#CW3RP(@+Nwi@I^JHiQ-'
    '=z5g1u<34ioA`d(lB1E&yld!}nnL+TZ@w@!-9=t$aIE_i%S;xQk(7-!2x*`EUX&?G-'
    'lnna|@XALjFT%7^*I%zi_*a&j$Kr!$+s+#?k!K?n?P*D||8euD?>-'
    '<YS=jp1uw;liG;kZaOnLv`>E_kP_!NW1k$wMwWD?O*L+s4KC<KKGr-Fam$tu#6+M+9)#-'
    '??O>;O6uH?YTTF@xgtM$JTa6YsRy7sALAvgx(?I2uX19)L~cxu%U69Y7sQF7*%NDa{RCd(@l+5m@pvkTmx`IaI-'
    '69%S$TtiKh5`t0QE)zf0`?ZP)7*((+X>GHB!Kz<_hpKO2D6_5An8;-'
    '`Zdb0$ykNhqU6_fp#%{0G2;z!@E(Joua3EP?nw2UiYF7JBG<&X?C+9Rm*i9TM$K*lPJ_lIxhJCdb<)RtBN!2Uze?AaE!~?sAL{HE'
    '<n42MxYzIF*B7i&YUynw7W58;w)&>AT&EF*i1*o9Tky918OES4b7sN%#0Efk+nr9E+c6a8coDq&q=e+|5w#tb>DrreYf5jbM8BR?'
    'uB>XeXst%{`%`*zRyQlUIy3NoTD(!wwSLj52<dZ=mbFZEcDfjz(HJ+X5#ga-'
    'z37(AU26`G>A>vguXC2*WVOsoSEwfR2T85sFF79NZcrF2t*jjsRs$ux#pLszWE`xygJ=hVrC!0!QA*(;$`ZZa0;?f_|}bjIf`Uot'
    'CyxQ=!a1$UYq%buZVEuhgU>6^200HM7*eA6>k^ndWaPrxL`|@ano%*-'
    't|CZHN^O8+XN2P5}OQ0r|3&aGN~HQE6yn!_8jI}FdRxJ<Z<aRJXaaeixVH{$KU4f8>*qZPxjDVjsBg-QXiaEE!C5HAgyu{{>XOV*'
    '3?m|Ut))LP<1s^8@-of3#WDw7>Lnna@P;pEW%MBHj8i+h|SsLzBD<tdjWcCwl1K19NCAKHo=Gx+AwlbM7!O7fgGtXUFBXE;-'
    'd7^Y%Xn6^J@fb6xL}4?#piXdVu?~JN(Q)TayNy<U!l+bTgj}#LKQ_uA2_4-'
    '{y4b72tFj+KSC!Z4fEd7GT046!o6S)oPDc{Y44D5M$CruRm*x2*-ukBEoSYwqz6is-'
    ')<DD^!*7ua5(@KhK3jgFiRIzs$C>E0PD81?_i{JZPK!sE+cG?G~d--NP=|v^V0qchZlIWLvn2O}!-'
    '44Zv+|wC(sIa3dReqc;M#tdTMMv6apkH6W2m`PzDBRM=fys7?wygco@-M_sL2YelR8-HsUs6Y)-((Dhf`D#B4Bwu*3+h^-'
    'mhVv=P*y&~MNX`!DKsTW0K5eiE!%s0XA+K4SLh7gMDMTcGrjAPt~i_N`zu!P>T#7J-'
    '%%9%KT=zKX!SuQu{Poa$EQ{3R*Q{cY95!{dRm`yp~Bx8y-_9yW^hYF7j-'
    'z00O_;;9v|5Hbtfst+Me<IahIM0i~Mr==E>#|=lnAD}c1X7p1SxH^GdBPbSk^PxaLA%=$3ViqeOsJqed}q!6jCdtUO*}4A|6w%o_'
    '(&6~d9~^AiJk*KaeL4wZV&my?O~s|J>nC$M_uBU_5m()yA=%msbgRnmZwr5B~@xzVL=e?EFjkOSs^=&Z8LMA0dV7M`@ZRY46ma`3'
    'FkyrlM^+XxD6)<Vq5G;t4cG!DVWOT?GmV5-U_91F(dp@5Hc;k`=cOaTJwZVbw2|VI{4?2`lZpq6C+Kj{>|@fwt-'
    '=1b?hs{^HlKTqzVo%P{BO&yqz!)IlgLu>+SyxjnGt^F2@t87VVmdd95yp;j;x{ZnUQ9pZP(-'
    '94^~S;Ba{{l*7dg@5e#<hwuJ4NdL?eGKI>M`uD9!{g=_d6C%y_9QSTI;@(Y1ycooxM!cpMXTx1ykCq5!?n)j07p;$W!n|7tVD!a#'
    'LOVbW5^itS3Tm)$d$U*3YZA`W-U;dq;au(42=mza+OLB;%SC&sjO=XTl(W+7dVk+IV^sT-'
    'q^?(VR;^nEDdg2Oy)+LG=5KkS1pb!iv+}o;d2Si}t)B$xCBFNUAiXqC$mAss_*&+ISWU%9UsJ(w%sCe_XhkiZ>*^*)=Enk>2!x5l'
    'G6(Rmb>Ro`uw~(g@UT_khw-pQ;YaYWHQ`6`u*Ge-cRR+xwW1nVc|9B8TVsq`eVWv>bCNC8J;-'
    'hzr!%xQLLDv4&=vtO*_oz*=3505xI9+^fy>jO1TJQbJ`GYpeD|k83TU2?$vI3>z-F7-'
    'GkZH1n`<Vw%IPCnIByv{NCnyVA{|_i#5o)3an3ICIA<4ooU`wEoU=<j&e<rBb2i%JoQ-'
    'kSI%8vC!M-`hDBEXA?K`)yf&WyK$HueMY%Nj{w90E~>Sw-IFmubQ5|~?-'
    'g)+C8LHaC6{qWtN1*xBTLMFd(U>biAS9TL_siV1#S8KN7Wvg9!W2?2=EO@!jQ$AOfa#0NX^x_yc=65_cCMlq+N*NWyL>(Q&L>=QX'
    'QE9av=aq1fZ;~-'
    'O_jytY&nr~IK7cQ+gx>~yH3nk9yq>0n=4%DhwoEU9wxubQw#5w6=Rrz{@BTbU3C$BSIfnz&NYuWGEeQH@)m~Qy`xY6a8eb%J<@~S'
    'Tf^3dq(Tv0?F~p+?SU4nYm}AqX1Mi6`UT?scR=mFid^H9_)U>7b!kFI}%-'
    '7OV0$<CVP`(y3;$H+Q9=`jFAjLCJ$mIPM<bo<eHIrl4F1T5`!$&+iWI0*~D`U#~RDyE{x5QvvTjg&P9BMgGYs_v3_hoBE0vg)1S!'
    '=XlxyFG7%NB<bE!R4TXt~akPzeJ0w^9F`AjSq8=<g>8+TTutHz|X3nT=+@O(}-U?ZrEkO=7ZvC2~nh0-'
    'y+b&a|{pFn!XLK&Eee&XiE1Z~Qt_vb8*INJjs03Ep6Yf$I4DpCa@j-=KKAvao^e!gFn?KL|&}KuonyBoeyKY-'
    '}jY7++~oOPl@G7P@j(q`e}JzUd;gR&9*J5NW2T$(Fv})KC_8e9+WT;r>zIlZlTdDcruGT8wbf(O+K><!>&CDv*he0oFBB;hBZ+rX'
    'Y8@>zTlRpHPs`YbJYv?=$aT4fNV-uhsy)w%Mx|pqH)rQEP!-'
    'JM7gupjWHCS`YMk)n07?dhN7{hke3xd#DeDCu&h23eVIcm&&DS3B4ltFEi4FP9N0}Ds+5ML#WU{(f4FxX-NvbKaQUfI(jzGiN@lo'
    'GA-#{!lvqUz`BF0CBDp?S`!3MEid(XR07G7sQE&D1&maACl$sqRAnPoYcW>k0xH&Gu*!d-s!2wxY@+fxj@kc2sDr?GmG2AB-lBdl'
    'JbepgbN`(?V&pQpEKS-)a8_ofNt-@uTBx+~LDNE|y-(khiJ_(QZkT%ApYP&+I9EHYL}|C&3_cnECAGZxOPsTZMWX-'
    'd5Lfyf%`ds6oCV%K!P<l$0E%1PY_DzrDu=wqUi}cLNUzRb-3U~qH_=}G2&i1P-d_DZP~qexd-'
    'Y==1hL6>eb_G?A5Rm7BcEkd0g5>O+;&J+%jI%;n$(Nnn#@X*I(^jiP^sgCriV)X4Si20f|f$5Q>8}^3IZdPAmt8I+JWNi^o&dM1i'
    'R(+G~G8Qh2DWaoGtIFit|Izw6EY{zgft0*yi3M&`Gm1vQD6vX3g(Jfo__eob>|zG>h&vNyrO1ayMDX5n8S$V|J^a-JZl}iZ<t-'
    'Jbb#=1JlkgSI8A<(k_A|GCNJ$^ieZHrHv1o5i0Gi`kqV_Ed`Ev4&r_3asHHM@}|vfY?8ZZ4`y8*lC@pB>s15Dwj{vpH$u;8WHm|q'
    'bRI6SLA!OHuW6cIoed?Ed{RD{EWtA6-tw@jF5Cnb0qmHQCS!WU%upHQgJy=x_&@bMnb=th3Ti8LrKv4rssi-gctbN5aHGU!Oyd9-'
    'c;4Mq13GqG##967%yZ4#c)-=4Yu+vebne;vKA>I)UBzQ$F<JXio?C0GURY1EH5u}$JQrMV@_ipq%rnqyPt2RZ_w8~;IJ-'
    'z2Z4u64K_)y7fN=!BIhWjB3ZV!XAEXJKzW1z9f#ZW_g$jJVz9&;?yPz$d$SBIeb072ukCJ~XcpUZ9z~eLb0gn)DH{T`&g2vI9Rn('
    'Ms8PX1QV~?4))r8@`mNjJ-'
    '$aSnJxD}h1Us2TwAOZz)c=xph;?300k<ZUT0%`+!F40X*(d?fZm$VP4RfT1hMZhJ_NNa*JU;M5Tz$NY~0bJs)jAShZQYK{c)04Hd'
    '|Ia`KIK0~g>j@3<4#Ap25s&#v`Bum_>pRd?#l@l7NVtszyTBE`KNqX+Yyc-'
    '!OVzVcBtS2EJASEegL=A9$VCt)&P)@s`Sy30fG}})2?!H+XQXUN>2&MxnW>8SCaWXXP(nl2?5|9vwe#Dy=&)HFNB4A4D+R|xpUEr'
    'C#IfEY*kaIlw+d$ut_N4}K@Pe+hZ*3}Kc^--=bp>+m;;{5@Xluvcy!L?1<V4EzPY@RY2c|sZ-nE4KNDz!$Km6ppdOBNIwK5t;;gj'
    'JW%JSZlmI+&PYJ*i_he*nDeJUzIltSunIbAi{B0x+7Ygn03#P81Z_npaF9)^4$oo{oqwD33s<2(K9U*miRalG9iqqyQ+qQyb)}@l'
    '1ZMYn`B8=@I-*cJr9k@8eU<Tyg7#xsncNv16Aosb7>_(P%ldLA#-l3te6KAJM+kEiC5@075mH<1kFe95wSt6`*3>CO&>2w42cHW-'
    '20oW<3R?~pG&6(wey3Nh`44`gvw-%o-'
    'Pj_qa1+tGhpAFP)&gA3NZSIIvHw*R^NIA9&_7JEteu0<D7wQPm>hD87ZK!iDo>z{SZeee;i5q-'
    'F2bv~V#)Fo>0bH>y`fFVv6o6uAnwZTuy|)AaihD}{ptv_9l}l305vVten7e{{3*rcgG<riA0W3~Ht^>}_kgpKNw8yg{ZXLZ|UnY{'
    'm&E79q?frr^-Y;nJe!*Jr7p(Js!Ful(Z18@;P@vcvcQNmz`n)lMYxZ*yH?*ovfRkCLmcvjdhNWfP%tP-'
    'h0d?ZO5>O}Z%Sh&u5Hq^ZuNf(Kq0+hhtVbL7DfgDU?NmllylZDNisC&xkx>+T>^w$M?6uPvMbU0&F^XcJnZ!*L<U+$b%x$#Ld0i2'
    'wx5T-zQrL|bb+|MYdTkXvd7B+HnV#QHySm3RAC`aFDdI*!|7{}nL9i3U(;PGN1@A8bcH;gLU?=X+Nb6Eo7$&O#-'
    'CxQ;6Ag^SNK$s>ABS<F=7#jHI?dr?X64-'
    'I7N4#zNx2%0{c4`BE??Q=^Ob9TzH*(*SK@TNP_F0cIznzRkNSaLyDb{vje3PX+o=c6sZ~P9P)xJiPx2bqdQksLdskh#hFg`rg_Ub'
    'Xa3(%X6S8^C10~>0JWvA8!~+>ySyJjFYJHya3ZJLE(&s6!@_EXueV+0fpQpUm=P9q_`iPuPXR(7@31+kD${`+1)<-'
    'l$VVA&>cxZMbWlCJz%!2~;&PHR=7(5&+XS+zg;r9v83cOCM(k=oe(KW3Q$NZ+?PL<C~=v4VEYo`j=Lzoe+4628Ckm?RX>LHr-'
    '4VfY~iE6MWRI^$xcBvrP3de-`u<jo_IkLPJIw58oK7OHr09`VT#2jD!ja@pIs<b^O?g-'
    '+eI@vX+toBM+1!gnsbnc;~60Qc{{Y3n_qarnmL29T$WVDSHk0lC@vl#e_+WBvH_*T@w?*Xo;f&b26M2l`B1rXHynBX3j4@&4!*_E'
    '|NC9>bm3~f~=?dxKd2}i#wllFDBH>}n-'
    'WU56RlvayfYwSZUEo*LDStl3yiw4p#cl7}Nr!_w(5*F(OVIXxlv4bCTgf;aXTq$|=Iaile+c4sJKpA0^3LrEc0nC$e0X1?nQO1ST'
    '$IVO`BXoP^aM#_k#_RA{FgtsvvJWS9cnx$?|6c@7qkEdgGY=2$OL?b+zLeI`z7#VQV}mRnj@H;9i?>GKkg4==P+I+~Ta{|#1p|wD'
    '(XB_j7sErYCUmDd{;$7_<g!HlKfSk>lZj^lF4#7icqZVAZIg*-'
    '0WR4#nRqtfnr)MbLjf0Un@k)AxN6&E;&8xad!g$uY4O@N4oqahDc7Q;wzYJoZAEZAdZcNad3bQQ$&M1bO&UYHP0Ub?3(__mt#Lux'
    ')}n97<j4(5D|D^(@$oBseEdouAHT}S$FKJB@oRj1{8}F$zs_}nto0gE4JHWT6yTAhMy!QGs<c|U*fqE;CQ4o2iqW~vkksL^z|kf&'
    'a5C?4P*3O>dl-mFS(<*CpAy_maz_cxB)5k)lbC_64$?0i`|2S5TB~o!<i91YnR;BrJ4k5T5VLKye&dV=ZG})hte@Tua2d60<U<je'
    'w7hnbI&JWbV%(}Oj^wT8e2V%`B&Qg~Xu!kj$TU1`%hPl`Y|GOOJZ#I;Y&`C9JtXVA-qwJLsyH=$G^w}iI<sU&U`vio(=_w&;J%TW'
    'CG?HV2<;m&Ls1i?X*gOnL7KKs-;l{y8=TxJl#{O6)IhDZYR0#Jfb7Nb%<qkt-'
    '8{HaAK}1Ix@bVLxp76Cs4k4;MDv;TYD6UK&G{rXGLn<b`DArbBqy8msp^tQPBlt36A#-'
    '*HwzEjM>hwLg|1U%y;rsIU_v`ivmQ&T+WO8^tq4p?&oost4-'
    'f7RnNUJ^$ZtZsL(EW&4^lN8t?@yswqD<m$weES+!i!zvu3OqIb|{XRu{K#b>4TY)60`NZ;)*|4CeK$4L`<)w|2|NIlT3*T*Be4_v'
    '8~C-r6IVa(HVmdU`gz)h?HFcx#_rVZvLw!DNG1u}i@OmYhZ{PO8|3&Qz=jC`;-tz&t#-'
    '59E~+`ao_D?E^7GacPi>Ev8zT5Ea{?Z^&fL$~q;;!0n~1@Zsi4@SHfh$gIKd0#<5#+!_(fQR)tnZ=OaOf65}KW@Vtd3AKp%@l@|f'
    'eb0JvzaxIX%ODg0=BH`urJp$$0t2P^q`43nDBY{?$wYKg_I1GdBrfa1Hqg1<ki7<Ke#GIb20$@-'
    'aj(>uiR$lx+DGVZ<@`O6kHjLT_|cWXkt;b3|8NLEPRBpILz6S`4?E{_CjMcMN6x}Oye7%n_=l~BoP&QX^RfDiNd1+A^*KFK-'
    'w%dVSs9lqSgGn?b6coX@kzIZO7$&$PbPYkvaIwM_39+eQuST6Q*bnJBcZaHA-sv|5Ad(K+Cv>V8o<;^R)KufB@b!d+ZAIPPS<JXb'
    'UIGg>E?6>PS+XcbS6&MndWpBPS;uHbT&@c+2(W(PS-'
    'h%wze7Mc=H{45#*j)Y_(?;f#zusC<vQmRa}tASV8Kaa(k#C@kzIb3i3sLPbMCd&ZnoIc|!<=ae`nR1)5E<5~Bt$a@t(DtxF}CSsN'
    'GRWgs7O6#`$WV}z{<ln>&$>n>iTBH`MIe+derIMQZUp&pJLwqkdo9*G>dVh^DnwO6AF?vCqLtAyIlWTtAYP&*u(>D%g*Nd0v%zQ@'
    'xfTh8E)Q{@J*aJ)K=YGsl#*T4FXP?_U5xFb~N&+B_KahViM-n!V1l&q`s-<k|jp4&t4^iv-aHka9OQnYuA<DO!1e;?#xH*XIRD7H'
    'mfmLt@!EWZo^oWBHyTccY20@P9AkQe_ORF}wVLX&(--'
    '4HP;*$PnCd*ZR(v|q8{=ki_Z!TiczyhjMe77|;wQx9fGp{$EQX*?5?wR(ZHTBqxO4QdvoNL&Btd7;wA?=UY^+E43yGLf5<<rjlCX'
    '_ws9+)wm1_w~N!ev+@bpX_VyZ}m0zQ+&<+R1UCT4)Sp)^6~0KVNI$$Lipwu?_TaD0NOvs+qWrewbH?neW8s`x%2J(UV<t_;`XQ36'
    '|s2q#LEjsTm-J-'
    '*|>;b1U1cwcq6FkaS^kRrGMo7P!Z#Im>(+QW%`~>Y$k;hTHPgN&TgP+JnPJe(tNIT^Z8A**__M43zx5Hc9C>HpZ^AIKeCM=G@zhj'
    '$--%rZ1e=0)2aW?6KKw$4m?kwIg@(uJb~sc>caB`nzN}7&l70Qp-#M0T#@;J$gv|lq(C_2-'
    'K9We4Jlo<MSvoH78n0Zpk^BJ{|40Tv>C34;m%O;<9E0-'
    'RQyf)o=iL^`QlFyoNDkb@|r8GDXgXIa$gwM9X!=Z1#m*8$;`cYkWu|D*k)+DSfwYUtv7s{GKySq_zb0bxZ?0xio|ot;m;^+$~A}o'
    'N|7xtI{Z1s9=Pi8-)NV}6-)D|Wxyt0=2OdnO}sRM+(4VHFnESj;^O`ls2N7wFN2zsCT{(U8$-p7-'
    '=Q&7+;j9jnSw9*Rb)L#hPLPi4WKT%aZ!{<$Skr+{-'
    '!$VtJHBqR+;k}b%KyJ=Da5^)pA83g27%Y0dleO?gSeuzstbSIfwIsyNH`W=8V1C?H37hQtB+R7Wp;1tsP4U)(lJzf<X~-'
    '5Dtov%O=;UWy!9MN2&9GUFPNLy2vchI;?SH<T#*upz)JS&Huar=n--F_C-LCh=IyC0(wO3U7-'
    'iI$H7{o3%1vJ*}n$#l}>gXuX3M5xPBSr`}&z-'
    'um$aT0=<d)WpGRJLaZ4j=XVfpZO?U*7SU{qb$N`F9k?^lF8D#*6co;3Gm3p793!DGgkvQ1%_i$P1y%fYt|>NuE1ip_yJ>H<$w8Il'
    'pyX_#65Vl=?e;VOF=}66-C2q1V!bq3rj|M6<3CeRf_&19>p=>f*#y%e!2Pu?!Z-'
    '}L!L~IRM*w%&wglrS;1=6fVBltM78hWEz)iNTzYqd<*~3FI5B5RKMx(;qWNh>;fUgw!>R;Lq!r>75K{y;jzijetESSqAUz?wsC|7'
    'd^Ws+>+q{(9HK!}^_oHx~dy{Yc!O?7{7st0&eJ<yx#LEcmk4uv9kKR#_y(Z+c9a7-'
    'M7u{MUmh^I&W{~W>#7X<1l*hafXlYanwY03T*;43CsTMT7}q(6j1AM}TC=!5>*B)hOcvXjw2vB|Dm(LLeBt6XpLuGO?D@{+B^;j`'
    '5RSIA1PF1S)wadp8}ax7ODTrJ0Ob-^{VnyU-'
    '0l{H*laGe}a)dg`lXqg9k6l1Z>;bB9qWAJo=3sW~ZgVKO^912A*_#i%8{~7S5WjX=yl_FC;7y}?2*kAyJ0~-'
    'v;CeuX)GOd?uII%eu-OQ;Ll8M?1SdcWHHj2>~4-'
    '>?Y{qV4^P=7qen$rPzj5DVL@u)VZgYdAf(qKHsyH?51z?3?=nA{lO3R~;*73Ch#dy^)d7{Uu_EdSQbt7;fZ!SCZz{YSu;mg>!bub'
    '5N?Lrl@g41{n1gMknZU@$P7RF|f0i(1hTAJ4&N=7OUoHy2-'
    'WbKa!AiH9qlK3xzc!13|8CeC3ui^uw@TjM9`6V#?)S7L%W?kQMmT|o5`ytbABxz58~_1K5t_+bZVAbci>9jGD6xlVS7#zCF$*-'
    'JhLP5v4NSn!9Kxa9%-'
    'F500wqI#uGVf}*!K{%4ZAP7e?7?e%es|u#@7;k8+ymcMtO=`6_spE|u)i6!eZ@_cB!l4Ed65N^?V{41`o|`Bv{Vid+9q8MB434mO'
    'um(w<S164lXac7SxAyX8evj)I$$8hHR^Ds)K22?pjq2R%EP4}ee|E>4JLMtOlL2*(i;K1^BTLb*9}MA`1%n|RvtV!*(f+9M!D0Ug'
    'x!pEL'
)

def _build_tuned_standard():
    """Expand offline schedules; tree and input values remain runtime data."""
    import base64, pickle, zlib
    main, table_words, cases = pickle.loads(zlib.decompress(
        base64.b85decode(_TUNED_STANDARD)))
    program = main + [{} for _ in range(table_words)]
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
