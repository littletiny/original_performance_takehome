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
# Verified checkpoint: 936 dynamic cycles; 9784 static bundles.
_TUNED_STANDARD = (
    'c-pkRcbr~TwLU((yfE~HI*>x_VL%=FPyz!|0%tai)QByVz^I^65##lW0fj_G4G<vqp^wz4Ad`fU(1Rj`KIs805gg!FhyGj7-'
    'fOM1&w0o1@2~sC&-cCuX68L-'
    'owe6`R(sY=e??1+X~F;d^xTQxnmh3lfA*J7J!39iT+%rGOJDrzxpTw2jiPnd++H8vX@re7nKgIXCH~B_zw~wbnM)eePyO0gr_H@+'
    '?x|m+-'
    '|e;ZYxHw(HHwYnFTCCOe&fLVPJOrqzvNCm`)l#<nqB9HHyiY4=?|Wp{pKZ&^Wt|GE%X=S@4wLqW)%Hy?^gK3qr>a*x89WBy2PKC{'
    '^li{o_E?;=vPlamF~9~RnYgHnr>#hqLqFt{;}c4(c6Fene($h_s{s}Ua#Cl{Eau|H;QfK$MN%T#Gn5s{X9LyXXX}dMQbsNE_&OG9'
    'g9th(dpw(#iqs9>EoDUvtpa{acr@9v2FS|uGpg3E`1zdY*}odKJHv>RqSvsecKq^#kgVueI4D!xMCvR2i-'
    '+3V7f8G0xl+G5A?tB2H(H~$)D+<pNW5_GyWNTQ!C&0C7Yk~wX@It-1M`*@!3<)I0KK<R!k@+p3C1QPyJsF`c3-'
    'Zx$!IhGk(SYHgKnWH~&L#dEBGJe{YGm^xuupa_B66j)}cp@$2HB#dqU7I`G~0po8WXU3Ar1|Gw86_{#s{?|U_V-'
    '+%Je|HONFo3H*OUcK8$fBJY5_MYkEi6r5@)5nuZtox>qQ|Q~e(#KEG3p*rz{0zN>BlKyuD<+dx^4tG@qvf>6Z)urTzLZ|NgDB0Z*'
    '-Mcd|DSl{@AA$4cf9de`Q~~^2p#F;$4OOoPalsXRoNqbJc0CMuk`UGQi^@j$5Tj~4ox3FN!s+W^zl=qO^2nApC)ZOJbnBusnC(<s'
    '^IaSC*wWS-A<O<o=&$FKg?il=#EO<8E=_h$JAmfi-HC4;1Sc`#%ZcN=vxiEr2i@30zK$|^VKU2lFgfZF@%eM^2I-'
    'J@kTm4AD#;x?xMexNxqd{Gu_;FMH`7N*PtHJAiD86^nLLO=BYkUp&w|&cXSmq>0)xxn=WS4#q?svxz)4tGCkEM$DVR$OZ=+-'
    'D4A*bJEVvC55DE7hw&!yJu^vGeB3V}{o&(&A-y6#?%$zT#K--'
    '+^osbnUqr8nkNd@>Ykb^i(c9V~ealM_%q#qrf5*i?8u@#iP0tr~r@QE*kDcl1TJuM3lSfUuHnW&5Dj$!!3V@z?COsg`<5v9F2S^4'
    'l^OtXkPybf=^zX`RhT49wxLS_a4~y$XSASI8kbeAcxbxTfcmIluA-?!CF8<CJf5FB7^2Lj|ki-'
    '1~E?#Y9HgGB3Lus@49piP+Ddwe%`E<#~1yY#VY{1&-'
    'zAxY>Y%dlTH|ggdl5C$G#P1i^h#C1oajl%mON#5njQqH`F@5Z#FXf|rb@3ZH%GVUXm7{!Z@jEd@*B5uf5KUv*x5no=RTX;~JrzGn'
    'cYKFS)2}UOD@p%$u-'
    'QaHBWChbbkkETOTX4lxBkcC{`B#L;?d%P^zp>vPsM}j<H^P2#lz_<t}E^kdvinad$Bh+7I%rYe7<<1Sdl*drFf=TlRo~nc(z!3Zp'
    'FIy<Z9Scs^JtE-1NOONo1Yr-_Fl9l3(-e;(PH(liJ4ryP+7Uy>&Q_zG7zj7QBG;npq&-'
    'q@0fw_onYUuJ}W7U;21b@mTRtv^O7|8#QGr{{0ZWv`-'
    'YLrjNfR6MI_v_&YMOr>Bp1kcmAbeY}%Q?6map_v9<iOdszeYkO9@u@{S{idD4-daYPEE8gnz@&&ikt6Wy}$l-'
    '0vrPd}=!wK(;?_1yXdio=S=?}b6JXx$vAOBoDU93(Y|5iL#tP_pyE4|FsrI)!zRP%UJ&OJrBuM-70k@R$LInOtUE}Tr-urKapV_>'
    't5&-'
    '?1Mv!{`K=pp7O19NKKZIiXw&?sO2GV!&aC>{}ac1rOFac7?_{wVJ3Q^lj=&OTlIN&NC>ipRvWeYSW!*@*S95iha9Sci+h#f$U9^Y'
    'ACHHS)vu&+M?_d@WB0Whk^5dKpvYIICq@M#9#&vY!6QOll8IMUutmw;;e=43ii(6t7jq&|WMkeys*(QE{`F*5Wxx=e3Gu{V)Epew'
    '_F8@z-'
    '?dc1#~{qL;l>`gk*`*qHS37JBbv)5lxsRE|p@Z=+K=K7G8Mv|;Be3Sf}h71cwvu?>$~T2y`RzVZx12vg*!l~2A5^2;7QebZ2wXy{'
    'R*RJ%f}DN$cn7-'
    'i_|vTt55)(t}L%nvfm30t#GNQe1MZ%R7MXL>WzVYYFblMb_u+k$kMZQPcm!w;p8TagYYa~F0rq=m_Sg`EtkVY2pNtRdw~Ha_fZNY'
    '8GpJZqR>@UWS>rxPAJ487TCtfmx;;{WX_79^w9hll4=xE|Og-ApT_urxc_q0P`QCkGV97&;Hh?SyfLPC@doVZ5PZa6^fm*GssuWA'
    'VI%8#@&*NVqYkcu~TQvBjSy+!$B<MZ%5o#a|`d*tz&y^}P<@4UTZKBfu-'
    '3V=slL>r9__44ar@zw~*hu&F8bPoKwx%}jAX`aCvlZi)lb=W$^RQ%p*q$A>LVaZviabJ)rh2d8uLEFI~0_{=;*pV#r{r|I*HRkRp'
    '^;C7+QM6Wtc^>DHwxu?^O_eWxHO={R?AoI)~57K{EeJw3=($@qXGhOJ6y3kWh$>POcUpx^uYApd(^>7c-'
    'pQ7(z#}h`I@E*giCv0uPdks6Eu#E}tGwgoCwkEvaumcL)neYL_E+}ko!UqjIp|FDqA2RHQ!pSBaX&6zGxtK}_EPZ~GKEF`GlW;L?'
    'Z`3&8#kLv2AA|{u0Ts+(8NLnDH9b;QZ~7te5%gwN)SD;LZ7eTwb$d9@ghLIZ=5T@uhZzRT;S&Zrl^j?2q=6zOe-%Duph?MNg-'
    ';u(Qu0~hvj+OJZ!)cG$tM)gq|a;MAO6BNcQyXQlcgNa57YT&!q5$nOvLJi-pqKlv(~hUuO2AzCMi(TrqboS5nWEHGgG7fjUb#AQ8'
    '>%fy{*jsLHMBw=kUV;546c~yP6I?pSEl0(DPxtmJU6iwd?56^HIB=4n3c=8|cvULA#L-'
    '{lV!=rWiI;Ih5;@BhX&vHip8SD1fPA(G;5Y)8i+F2{Uly`m4ufrhz@lj!W*521J5wd2V2RGKbw{ip0uv(ci|OSD_R7!<(CuThdbu'
    'tWUx?;WGxNWgqdagSfBcTUQjn7T>zExJi8Ls^Vtxt*eV$#J8>~ZWZ6Uwzy4v>$>9hq|SdMWqLM!{3|Kh+D7%+>6FZ=ymrx#4LSZN'
    '?I`hK!h!0B2TD*FU)B_pWHuD~)XlPP(zoTpc&`cnrow22`JwMgSDV28gj4xyQy8Fd8eeS&3lvW0tIc78!Wn$E1#D25##dXy2!%8G'
    'YAaZwaF&q>lBBVoP0wFQNPkTq|4jdSO-'
    ')Gi3=BsTkZ2G6Bo^|pVxro_7PUn&C{%8PUT`!Vov!N002H<oB8)Cx#1UJOf+jrd>4tFxp65)%m;sM-mJ!tOEkk;+w|bHp;@2U?Z>'
    'X6fv3*m`6iMyh)l8AlzNKc0WOlyjB}DdZQ^IS8lz6wpI0F@1QM>?C5gq&a#q;4N6K*!?W?S)U886{=o@NqEYNx+4AQ<Fz!lWDtyg'
    '~6VM*?pWI^{^<t>SKp1l}$dOC<14agQ9LcZ((IR>e~E=V47N99$Nqv2_?ofeh}9{`-'
    'K$e|_}J!&N<)AN3$J6!P5b@yb^6E87H$%YYCHIn5-**B(vefCS__77HXG-'
    '>FzA0r{9>k%YZti@PN@A6G1v2zY#Pk3_&b7fX^Z55i;ot$gnIApJ?)U3@0>ytc-'
    '4n?`i4W$Z9O@0DQ&UTzbf4g=9gG(UfMF8YWUQhf7lBiv@f?fm99$Kg!qw>>P!t&qx}TMwEyp?^4>*p&X^aAGt1hr@}@=^qX!wxEA'
    'FoY<27;c#Lr`e*0-)wh*SqAMZ^jTHvyAN?=TKQ&)Vq18z(!ZLjBtO{T-'
    'TFcB3h_!qzvzF(F*Z5~3{Rt*lL0Hbd+y<_v7sWzv56>Hl$=<^kFPP#cGJEObMMDwV`}pF|rnrTUZ@T!4DQ>0Xn=bxpireV;ri;Is'
    ';&wW|qFIZIH`$Sd`%U;G|NC3Q113Dm7q^9nOn9scn!<+o%+1~r!cyax4tTWhu>ACKJv-'
    'v=bnKRv&Z2Z9<aQye*YYevAJkCJ$b#Z^IU@^;H{@X4UA!d+V{!4e9E^L4cjRC!Dc)tLN;gR_lHF>!*Mvv-;-'
    '+w)34h>=Tf>7U{E07a4-'
    'cF0xTzbwQKmVFmR!Pf4Fwm2VtC1DuHy_<xWt6B)zp66ghLGHWWy5%KypVG;*$mdVJE(N$^ci6;j5<&faO@edd2`*j^nFm4Y0&`zI'
    'x68Xzpw(A;|wq!ur1}5y*rHu1X(0fkTk)=*H!(^_Agx7ojCGEgFC<!&N;1A-'
    'R*aZ=)Zr%Ikb&t#2{mR=(38nD7PlN6YC8>03oE#R8Iyh^1Iaf)S|{i%2dal;Um@i^!x{Oi~e%6!(x&L?Xo!l8KZ%$ir`7to;lu=T'
    'nM{>2YG3{nX~9L^`{pL<YEB7RgX(v5GsBJ6Wn|XlByC{6GyyF^_<>ACJnkiqk|-'
    'H=v^Tn{@S51FC#4Up?6%=n;O+PxM%{zA4}aqFO1KYPLayzzX?&6F#qK>v$7BX1D|pPBP&LlhXa8!xIqCAd6=?#PgSPN3M}k$Yo1Z'
    '+#TPy<gMmK0kK}qH#pS-f2J4-Me=~2L`c6g9||M&4>}5yZ7gunG3c?C%HnnqEZ1J?0Cm~CH1ek?$H&<nZ*9Y-'
    'mSd2xjSZXG6os^hADQq)#yLZw&Eh^e{N2%p-'
    ')Q)|lMTPs@OO+2ztiw{tPOW)_&d&qJ2m_rZ^Q33{N34xyJGmOku7;rDYJFMQnm`{7={cb<c-'
    'cjQto{zC6k}AdE}w!FI9`eP~nwHFk}U$R|J!yx5pZiT?z{gxfzK+`pVqi)n#t)nliU{ZJFD<uFUOSU*`61D06!^!WqZZCkd5~8OO'
    'Z1SY3(nTVX;tep?Qg%XzsCrc#Z?0b}SifS;>|aHLlB{>VFTF!e*V#+3QY^TTQb7x%DHgEs(AC7G?T$dIR%d|SBNkgHAaXq1Jrr<W'
    '8?G~sZ=m9j9=>H{De+@Lw9Md4*5nl>RSS6>psDhOen`B$fk-'
    'u3*VivV_#NoSd>q<`O59=<*!5q>r0Ny1r1zm**%GDDr|Ad#2owbg3y`C+9226&L4n~?un!T#Sf;d2~3bdx>)yXMfAgjYEa)*jQ#9'
    'fI1mwDgXaYQlTsVd9kLd*flcCP`uiB=H2lnzvCdkEgagCY{+a>C28uD}C#-'
    'B!&*ypsx79Nm2TY0Q+V5Ttjf37REPua*+@D)hS6n<Wr}reB>s2ER0P8IzThMMiikH-jnRPI!1kqqhaMcZ1iU(*xS*B1%^<bWKY9l'
    'L-<Ydr{Nw$04_c3b~bEbQ(mM!{Mdvq8Nm{V+c3$5>-laLhapZ>B2fBW87OJZ>|HIC;a%>?1rE4^olxP(WivU-'
    'W(0Wu<9LSFu)7&FWwe7_bXZ~tYjTb^Tqr(<!a1oqLS$`RWr4>Z!vz(fqOn|2K_+P&msDVu$8${ukG``MRel78Sz)U7W45GC>4lfE'
    '_4)WbCv^$XBVMw_=f8H&^v{0oi)Vc9D<}@p`?<K<Ow*Urv>r*#&dqYOxxK{$?_l#Tg1h*E7eKwgVbiVt$`<bu@JkmLTC%O-'
    '6#h@`me6KOBzA}2+i;#usjl|$vj~FN72j^dRW>C@+o3|ORw2<FhWwbqptlVvNJKkTD>71KhcLLUEZ+`l{<JGBwS-tpFo&xW%zgfJ'
    'M_SQU7gQ_h1B1!gW>)RA@;T{>I9@U13@hPevRufwscjz+Ts0a-'
    'yqBKXX6X=Cw;!_d%{C?AMvN9VSV=4V(S(<6N|m&SkJ)g53D@#*`i%`=w#j(4hg)ozYf~CGMk+lDROk#oWW!@tHTz*39=GXFeAq~J'
    '8s~x3@Xw8km}aP$<i<xwwRxUh-'
    '?s5;JIk3$dQc~5<1!mQ?*w36Zo?Ox0E{baIL9SSK`_{o4sbRw9PPq2eA2q{u;0`q(P0*Ood?plZf)sKrI@!3E>`5FiMCjfpbrDF1'
    'X;eyc`=jPDnMj1AXfVG{WgWgF)L_ssrl#0u%4WMj{Z7xPf18KZIdpIvavSO<|Aq;GZ^%@r?ST#3Zpy%im9<e-'
    'aa=>@UXkRR>H`suJoSEvdT%ki?$}X8Crc=KvLtp!J*3`J;~o)@kp^r`uvhB{!nb1KELdWKNg#%&;M}6qs8Xw^N=Iu*dl#?#TAbgT'
    'c*#iy5jL-tMqxQL-HtLm~fg4M;p%igwtIp4Cj5q87_1iWPU1Wcd{+kM>%zPQvUo+P4rn5PH$kTOW|`jePaKY0HkK;2g*K-'
    'VKxOd4@u&t|9`U0#dE3)M;T68g=sDX!#SyNrVGay&Ps)|TsYRGyj04Fc1VI}N6GDNN}gvAN$`vjm$f;0g1sccGfr}QTaq8xM-n_c'
    '$J}1!jQzl9#gDUa(vFu5#kPEw5A|W83op2Ib%YQ1xbRoLTH(UyoFoRqN=JuLG6P|iqthtKlCau^vmIwkhQfb3cu_vK|Lfqi81Vno'
    '!E2pW9<;FB(XGZ8IN7t+%%h)~PP9>NIsJ+#HGx16_2j0IWzhGjW$~j;aM&`@u2cvbuIbPeUo4Ne(BsJt<vKLg!_iE&Wm>M|+-'
    'l2RS3KFJr|7d~*1s!Eprb>sg#K-5xQ76khsDHgVhIj#JVnvi8zMFe#@-'
    'ZTx49+U!od`QpSLBJ+R_r5WXWzt9yS#chwz1M8nNALQ90*tc)+4u&VPz5`uO-}awT78A=3$tKGeb|lI@F8O6x6YPuAQQ1}zS-'
    'r1HXR4xVFFEtT1oTsAs9oEk~`D*9`Ktc!gXwqSzGvtDiZu|9qQJBCwjxY`P$I?aY__+m^r-'
    'G*!VVr)3WhU@rZT$pCV^?WftoN2=ie6e#l%Z3}-@7!n6RKUW&-'
    '=dn&(tpsRoX;XBzqd0haMoeq;<MQ>!%16n<E!U>=_}J>uk=RerO$}IUn9AFem_sYLlML%H>*4cJ%&9F9B(=Yl0#pfB{2^3;cgfH%'
    'm?%o6aL_W_33II9WECWbV+a<oNDflWVlIoBDj|v$xa0KvLo4v;QrxAb|SbTN3s*az2ZoABDhx_$u2`!ID{p{f;ME?^Hw}kr_D|at'
    'IB9-DoJi0D?}&WbaQf@)M9B>yr19K$7c5=%EueUd=cv<h+1=qdRMe|KS&@cRgh-'
    '0(dzPHz=icJ%e7>uT+Hz@e(JPR{CLEK{TL*5lK~9-Yu1F2)d8F}=_d0R4zygo4SzCW62~ds^jE_{3~RgT1%-'
    'nxV#FM(&$Ch?o8Zb|sXMBFo+4IAF(ck!2Uoitu7@q39abq4=Q~NI=GoW}pW%(A_TmvLHK~~09uDzgfeX(&wy`mgXLGWjC2@#3**{'
    '`;J~Rhd5{Dd`11*U|49!6V9n}RKOvq7N9!TBB8qqkY7y(VzGUylq&5D|lph!W=0yi-'
    '3FlR+<nC?sHJY1F>C&$`UW}tD|l$af$zbz`31ZZJwHt{4t3lp-'
    '1uZ6!bZ4|%#{}g@+Nz5=Y7sX#r#$s4s_6+287=>~F6uHjuk}BBQCj3Kn>We14qH6Rd6JE8_3NV0Y6f=oX`Y0oI5~-'
    '}CjTlNKatb4s5{aB{Bc>8TcQ9fr5p*A8#8@JfJ=R2PSqY2{On8S26LYzd#coY<Bcr@oO>THqmD*$FP0jEnM5XsRTwYDI^gr?*bA0'
    'ylJ95GG#+ZlC?B`7QyPnxE!~lqo={Y70$;l*yF+-}a6tR9os*}Y+q>Jy9e3ix!#oZc0AjR-'
    '4BQ0z9AU*Vu<bb{%%g@y^2~H7IFO$<U*t;mFKfsgKhY@0eE*XBZT*cJlX-'
    'V@fGg1j!fYZpOpoKV$Tnk!+v&dzwyKxk`FtnHqlT?Q8!KviZ&=MTVcLmoY1Ii8pPGFpgexM9N2FmC%>TuQqjgUxskmykG0YRbzKg'
    'hXmBxkd>8_42Oc<d;P$Kau>vS<&7`>@!Bzp$--'
    'mMoj$mX7co#g8U6b+m_HTI3LJ)3igl%ZA(e>c^f=3@Nf2b$#988$Nv5rwi0<DH`0Ge~FxEn-'
    'M<>FS?@<M+(n5#zg0tA;xH4TL0fY!b9{9dyOP?%YYEKG8raeX(pLR0C}Qh{s@Rum&}nqEOFs)ESV>4c-'
    '^EPneK3f58u*U*#a9@*zgm^3X5!5Wy8<-'
    'YOxJ#Z1@FV6)s$>#V2wd^AXId^OJKCQPSx)8T{_>Z$5m*r~iU<I!SPw3P3;rX7f|8@H}>UbYx){=MV=dUW@02`$jgn=A>>LDq}UP'
    '{U^Y^D6hu>@2b`(S5VlLta%5PnT*V^EpKe5k1#lyalK29B5L_lQj`g{l0V%Jzt0|cy<-'
    'ooxVEh3?pRiHcPgv7W6HYp*s`!Yt}N`1FAKXnOJR2iHf2EbC!{coG-X03>~1G6nS``B=1-'
    '<I<Eh`FNm2mS_BJf9<g!`}av;*$Gn{Grzw&`q4R4Pc9$Ou9@hvwHUrMo-'
    't;Bdk(n}n6vD!==+O2jIhh43<6NlZb_7jH>TWu%~AF<j|96oAObI=fu)&eJBhHUUcO*WAcy2lA*Aw62+vZ$4gkqPf{QOAC6!u#E<'
    'k;M3Wv<HY7-)@A?43`e!MmeU7U{kyhvZ!=+R%(|yI@65!sy5A9EI?4VpZKbsv~PD20+$2CSB=%a-'
    'NlGuCW)^auYJ2q5Y8N2tt`Z2QFG2*XzCadQq3_6sgPb?1~=VWd%tvMNHtqrn4O>1pPKOAB*$Nx@WGMg_*=plPUQGQ(#=YQKG7j&<'
    'lHKXrP?_G`4KWf_paxZrb8xnYMBPEgOIWy5nQ;Q+rEaNFz*Uto$UcN2aMwXGQaq^VaLxgI3f2``3~p%aGp;J2$xOZ>{26sn$XzgM'
    'jSPPvAIS(HDR$UjJRroVprBgIx6eV6oLNCg!d(Z{=$S0j4aR{Y?vbgEnWy+E{~UvbY(@OJsj)8ja*>q4j1`wy-'
    'P|ogu|$Li;rj?A_hT<kPrk3M2d-sQ?2~UgTT}&s2&}*BSEF+IAnuK<7KR32{7kXc~t|3bf%SQh*rE({2MQZncgGY_L(-'
    '1NPAtD^(TG8zvyZ*LUUZ0V`~y6V>cm5^j9W)$kl6L=zz3kn$aSm)xjxw3d=|VCn#X;l*BPk!%QT)(^V(B5$G@FIiiVKj;KA$3+<Z'
    'ag+8pgpj1eb{;^ok(%u;qw6t{wH74zxL6J!tXHaF*z8RF6v~31;ChnRkJ#Q-'
    '{m3uoIF0f(0B&qM(aG<Nh*(0K69qH1mn(M>Ab70dRj&b1z?W_ES4R_jbtJTn1kabMboMuH54_FEuvs~7hR_yh|mcqt#V7n=zKp-J'
    'q@i1L;jNoO8i!Q5ZR~u4STB&KJYgABu>EAT**k&>WH^)?R>-'
    '0W5%?NpnWLZheGNFCPh69q&F0x@#lX_8fHQ%Xv(Z?Pytu~`Pijw%h+$I#00Lx6-%-'
    'F?Q7oFKIW2G5J)82r=(v=Q$CvEAlY|@vGGcAqjIMULYj&~!i=`eoWo8FCNz~yQf542>}#3meM$&g8_?o}=v%>E}!LB@{gHVWu-LD'
    '6~XOoF=5hW(oaHR$la4$eT6AeL(?xHYyJbDDtc01_q3Om-'
    '+c%I?ReibyALE5O}ylOYpCb91z9cRx$^TtbQcE$I}4lQ7AW31<|7%*R#iAG%>`I>noZOI_Gc+<Canh5f~!hs#|!Kpc9Q>%xKJ(Zd'
    'xkOcIwKu5{s`=+hOUTx`QZNnEpRIJhRRp;))_;tKy|<0}&4b0l}yjwf!5+KaDjv8fHN)Tv8-_=bkum-+B-'
    'n#H)vhi_}H;<F|^?!v~bwLiDv4jXRa)BYVFu65EI@&iv%dsxJ}e~AzOAOy=-Kl0%ff@ti9zv;tQy=dFN`*5xoZTprFU-NhYwL;De'
    ';D|T|qY6fEWJ8h|$@w2O^F=Srqgcx?>`EWb*E;n8GT31rKhXDl_&dQ$zR&Ob@DkxlzWSjLLj=6}uD|WW*S++q|DF74xSD_NB{KNo'
    '8oqcHweuVK!o;l9e(YclCmoNK)6#+gi@C7W)xxYHdUHMc!Uymw)i?IUK309_lU?jOYie-'
    'T2_X>}DcvhcV04UMb@en48LRwT*27bmrS8`p{ja=S>geaQ?*D3Q@BrOoj?RY2@o}d|J5plrko!m%EiwDVJukt+U(jhvS1#tG4&W;'
    'f_d(|=T{U8kZxUZ^1aCqoDjAcHn{bpy3CA0XkHrC>U_!TsR3{sPOSRHKv^{muOQ-'
    '$sNp5u1Um6*s%}2+UfZpdDp`%i8*^s}nA!HS2dnwgVMIw~Ep4c~X3zB=Mk9~yAWtMJERst5uzD)3fn*-'
    'Xz#_*$Lpp&|GYTs)&tknWFc!q*jws&N!7OB-'
    'Y6t?m)Bulke%pe)6d&K^e@n0f{xd)mnkmp205In}G70fZy)i4Ly$Cr~V$sP`kEXzR?dNhC-ZHvBHmDD@R07lM;VD{KWR`!%pz}jL'
    '<ZS6$=3Axv)4B&cjWI8J<ip(^(s<k79`&PJ)!??cEa}-yXj-'
    '$Ay^c%&srS~kZE1hR?ed$h$8%jS~+^C*q2$Ok7YaDx$A#hcz@zK5Z0f!9dO1aR4CzBwTnDE?)f-'
    'IgZ21SfNNI8P?c|T#oBQ6*=f(va}Nzol&-EG5aij=*U`^hwHV9}iF!f)t27%iXQN$-'
    ')beqzy|@@p%Di+wtjyHa~R+MzcC2nw``e1Rh<stt*Y9U-'
    'eqsmkV^y2BJ9da&3C<l9j8Z4`Yn^bpOb01DXHIi@yDj=TunN&T!ZFyV<L*1Ju3dSv~)D|Mr$i&5yQo0WOh%YFE!Hs*fXgg?2^@H!'
    '^<m<t<uZI<{Qod%=52fru#%x8Cdua$%LVpW;mD^`<QN_W8;4SCy(wIv`g)@ckF$37Q2LSVVenVIz>&T&~UBCOIt`uDv?nq!8N_n9'
    'jBBZ_?HxO!HvttqRw*o0@2An!5Z*^%{gImJ*C*RGE((xrYrZNpoJX9LkT{YMv^*Rj+`UGRLh#D;Yg>+#hACVYnibatmJ$U{rB#sD'
    '4UUg>qCgZw!O;_1MiEipVD*mER=rvv*%qkZM~ID%-'
    'o6ldGe=On>0FNXGA0$WLTSpy|XeWYe=)F;E;Y1R_>ZMZp4+6tFmQ_TTj<S5ZaCOkE=YObOCP}6y&3tg2nKZU{#7h1eVu`><B!*=o'
    't;X;b8_-c{~7a20q>FOX8E;a-'
    'o($&Ex%%UjYNvVe7TdmYkd^<^MU&TA3zzfM%eqItNWGlbGpO=sw{8HZdhmdOm4E0hg)rmeGN}YgOoOr{M)+!&^ZO3GKPx24-'
    'jy4e^3Tjpi=dpG~F;eI|(Tf-la}-'
    'wQsX!B(i5XwG*!0Nwq7lc4rzHuHWAF0ffC&U1;>80K#yQlB3nmzIxEEhbDCP)BJdu%wX?%pm{;~;+932DpPzHSkkMh+kQ9mDZxhU'
    'ynPxV=3VB%Sq0Ad(TrTv0AdYrgaSj|2h_TF*O+eaH&%L4)KH&S+dqK{)ZRlu1cr)G(M4B?(gw2L5)&n4PY2IoV;#~C9AX&afW8);'
    '+1-AEDTIr1Iy7LIdd>BNVg;0Wl+A2`vGQ;}|vlN=po`D-'
    'UTf`F2BAZU4*m*Kt*(L*hBc?oSp#J_4O+q`soVgQ0OLzR?67vkFu$VUtq)lKq5U8ZSfo5Wu~>=c!(NLor!)hatenqj>fAG()pc79'
    'pxoa?AH%XQ(~yNqypA$>pG$X`K^)AmsU&5m%hREh{v9w}kZ|4AcU44Mq8VvoDZxgE&QEjfTy$^4P=U)>b$O|&o-'
    'eA;%gF#LSlcD3-I_e-RPdi|4*&5MlFgZgAeNqw@iq&`_yQlG3YsZZ9F)F*38>XUV;iYzB^W|X)FR0Hu+bDA@_KHkI4G}8ej#JqLm'
    'buG0renl^blh}F~=UF+76iw6)V|2**SyU<QW`Q3V<bT)#K(OWchy{XRnEz1=2*HsqeRD5dkHozw)juId$)Z;Olo%z8{)eaeiX!K8'
    '00JcHLp#K?e07mU#{P%wiDp?Ou7B)v_c|FJ++waEtX7~d?(TN@&x`@vk?*1Zb(tpMM-'
    'a5r@Hv*Ol#3%)$bPzTh^=c;y9H3aPXc)~Z9L}$1ueDGxVRwPE1B_PVYp9HoW-JWzZF8OyTb#L?JO3D2dz+A-'
    '4h;?#AmT2JZz;<ae<|%mCk*-'
    '`i@0e_Xib`^hSp*hPnfNt?7o?2z5BI?ipc58|F&XB+XQ6kz8E{M+kbC!vID>Jpmt7@1+?iTPp_@L866MoHD}CHYLu(-'
    'ICLo<J1<x&4XmC(wAp**dwTArbUKiHVfvvmO}V&313}ok&FJ37jm2nw8%Dwa9~M@^WBUP=c0@d=i-'
    'bIXO;{C5qu<&@*P&yD;$Bsi}>OyN04P!u24_9@P{O)r{JCIhI*<?X=I(Yoa)NPOZqRPJV#btc*tIp&-hQ4vR-'
    '=gDIr`ljvBrq*dG+!Kmd0I+dBmND+MJGs9md59PM5+Ef-'
    'p};eX#}<By%fF?1p4HrvCcj$rczEQ!k;fw~L%Vy+_){awDe(h+jLIG54WKq^s2lWl^e)v+XGEW>iEW^x!_v%|{xMVVTPN)>ZUacI'
    'dxV>&VrGuh@LJ;|?+eklF$QI>9g6TUdw(#>zm7loy_y%}G0TY8es`65_)^;_`8F_s={OTIYPk{)a&6EP5^<_vfB2tp(ax|<IN8qI'
    '=!!~;v;(^Q(7855lI^6Vx>F?(`x#)Q3GZDy39MI!D*WRWv>c1KE?rD6&vYkGuE-'
    '=HgVNi%UCOW(9SUaD^*3pxTXZ8v^wvJ)r!B$P%_>@zUv;!B=`J(t9JC-'
    '9BrVfQnH+a%>oLOae$%9&(#ypuqJM0SFc#5u(xCprl$NPH(bDPT|}a<a>-'
    ';a!NzW@m#O6ELn*K60m?uWqQz1Q~>c%8LeqO{~Z1VaxJ@^$?eU#jdGsj@;^QO>J`naUe%WtNWaGyCbJGQ~R+xVrVyP^CqM$m)ciJ'
    'RCS|c=qM-'
    'H<|()dN#H#V(<sjUSy)DiBu+5l5}UAC><E32%{oHg%kHZ^rO6nO#87%OOD`o|(JR^9(p%B^a0^SXMdQOQExniQ$P=9SNRka<iI3J'
    '>F+;zDxm(UvgjQsDmWngaiD=;1XO7Ua8HbCT>B1(vlHNwmz{w{3*e0AAdt)bv8%NRWed4cC`uTu3X%xLaXtNI352>48j>?u$$UTk'
    'Hsc_jngOZ?-$dRINVoBFExZc!~vTFdnl_ky1#6|JT9toCEyt22gSAXU^l9-'
    '%iaw0Rc)O#zmx~vA_s;rw=MvwcL$e<E^xRPK^CH8R>zGr3P@i-'
    'H{Z?kFpAJ~NNW6$vW8REo7whOGx#y*LnpYZ)Xg`%kt?LCW17qjcZPRz2*@e0Iqzg0U-'
    'j{PC4=9U2B!#3Jml^7l0jwCUP<MvgFRl3k8)P_;IlRAkpYK^vn2eank{A+<NwYJm0tPQ7R3=*ZEA^$a7{17sjKeWO>bdm`_vQj6X'
    'X;b^W!o7Q~IwW$y_gVEwgmE6Q>V*g^KWNnr&9+(rZKoH5Y(Br6AE5#wU-o0v0TP1{3yh-Ty^_Gtnd_ScHh@tMIXq>$eS#XMwn}qh'
    '7kw-()lf2NQ`P2oYF}VK+;E4<ZiPbk-'
    'P5f^7?P)`zK3F&sNN4y{gKATOHlj~>X9F!_QQN_$;h))BFlEQ7)LT}*NA_jbAGMZNmAPD#7xqWzh2xHo%kCh0-^(dV~pl1y_-FP!'
    '}ye3V6szkVa`eM9ldX|z2DXQJ`s7fOgk>t7j`nTY7z~<&m!HkE5GvhTcmq-'
    '<Esx^q<cQXS0Aznoj+R3@8U*VZCNZ<f7lweyiIW4e2p+WP0}n?vVIuK8Z`mE$2v@*NPCxL=P8=}@CaJfVqv|=Ft^xDuPXEK8_Icj'
    '3eKWhX$yH5!i2uewv|LkYUxz3j(1_tq*o`ouuqmw9zt$gqqj|9-'
    '8g#N6vmCCx6NSNIC|S04wIv|EpVbadfO5QnxnU^;8)I4UW5&T^CMBf-wnw|nF3xir0PsiX2XynX`Nf2PQXiBY88*o%u)wS9WK?Z_'
    'eTx~eW^XMrAhJ#6QYN!FtN46mXecxg<tQ(Fb<?Q$?_php)T<PLs7ASg9}YKRNHq5kbX>^C1KLTw3p{16Ao9;d9evcsOOwz!jUSX$'
    'u?0&F&f_y&Z08Q3eR8HulX=n@s<7#fUAbF{`H`EOazzobWp%9B<RKb_85&Fx*>rC1xtYu)*<Kq0DEk5*W{}n=BxW)b?KF+s|R40='
    'zylH2P3J06vQUPvEmL5I(8DDNHND4afuXij1>$)5yv=z5EO8X7f(s?#?CdKTo~Nu`qgi6B%{3U0`#giQ$tS%w3~XS553t4j`kklM'
    'B=_Vao9_dL**1sw#{ufJ<)m8a745=(%S$8M7ZESz9MKzz~FwTDV$EDHO7141hmF@e;k09I@x<rU=*wSL5vP03kpo#rD5w0-'
    '~kc_PO!8dtVlIl-OOmL0G^H#TMhi&Y{6#xl6+ag{nES>J>t|HgVB{H-'
    'c#O{HLXV*9%C7<js2qZPhSl}E#VslU#k#~g5yXSM?u{xl%t?;nWPpxx+le-'
    '(LX8vj1EeHee_TY?4yfPU>|*yBGu@ZGI3#cG>L01B8+Bn^~btc1$V)?^(PNd8}*Ivtgl4YhJ*Xq<dRu?{Fl<)ODJT3p~Tx;Kqztd'
    '4PYkm>l*<}N+?9AoS^ExtalW|5Nv;tuPBHieEN`6vwf2SU-HN|TP2)C1C$a@qW4i;`iC&RQh{B!x$wp0AXg$@Zk89lx3>DwG&1-'
    'Vb}AUhoVP6Fo@>q70ozJ;z-Z%GBTA9ASWPQ;qrv8W>BUEX%>&Ymk4}<@rR$zv(sK2kSCjO_cV0v1S$yZUbehF?UPq==eCPEfU-'
    '6wckaWd&-'
    'be<u(xLxbfWauIHAp4QYC>LRe(U2|e(Sg_zjZvCSyl<^D(+^se$;;!^D+gLe>4YnGMZV|<;ZD6Y#wZ}nu2fDafVjq1VgqUy89+F`'
    '#}x(tCNb&u<1_VrTDa~5W&}<_Pw#c<LDAGVA8L|f0*<u@n0-sYB$wP?OCSNLHKkN?W`hvx~Vo+5kB2a`>F_^Zmw-rgip87t}4Q(T'
    'WW6+;nS_EEG;8XOMzoJquC>gb|CXKAjR6jlJk0y4Pd1l)`@Y%Sf*Ba-'
    'R{Dd(kc2qatn2Ny^qVfW$8pSPKOve?2>Uh%#cf$Cgv88SHrUM=m}s^c@(T!QXU0s7L-S?0L#gvSAfOj(JR1G@+fMTLc-'
    'zMpsXzM9CM+f3nRq+MHk1ILq$J2mZsKSF-di>&X+a-2~G83N-'
    'm*0kcEm8+RM>Fu1jb_OfpR_@hN6b%lHCwr7XQ84FMhTm$3@drkHI7v#9~gtYkVhV3}3Srv@yunhDi_W!5mG8nDb-rc?u#Syv@E1@'
    'z6`XHV8EHxLiuRHY0@!gJ7xk0_n+5b08jSC%BzCeW%U`bLq|o{p?ul+?bCtXw_UGVC-'
    '%RNaQ6W!3h?jwXECv9Ra4t$PSqJ!M@>qkE_!*dvL~j~PNelIT3#kn@q;<`IT`kK{HtMEuHz`tR`O;^<z2GZ#nqGJLt1@PELSiwPf'
    'sCl?p^3UH6awy(mg*9KmZ93DhfIMPUnsP=Z`Mx&_qapXhmN=3oK)h6u6{*t1FYfRXmuP8#e&V&P{-'
    'Up(uFv*E1>6H!&)4?WImfq|HMvaxibFi6ZV~&LeR#vZ!1Ea<oAvxI6vN6X(1DhLjL;!i^0LV*@US1(<Pm)%1KA4QC#$6MA(G?Z0d'
    'IFa=m+U(nnYt*fJsd%iy0FNw^<grKvt6Kc3J35N1x+`Ya1dWn)O4c>2fG@HN#YjXz2p|&qvRIev*Z@utK=5myW|$$r{osSX{C{TF'
    '3(<0lajL+;B0Ns;R&?uTe0V{@3Oh1h;l<;z4vo>M{q3)=m0u`O)c)pc{Lq8q?HB)KhL)%t@S|JUkJ(0M6g)zOB*bc_JXkoGcKXnA'
    'I5k)wkR^*jvaT5w`0d0<L%gS$Hl|gamOJ>?6^BPLo<i_uqL78TU<EX*L9&4GcL-'
    '9_7fX+^b*qi)P|kBpw6G!Fvbh&{DlqUya3Hj<5<OSt-'
    '{!g04fy5UJOU6vFj{EgSA;Sk1AU=JBr>~<u`82BkO@YvYz6R2Q0^K2L+kM_N_i0q+wAU;=_t0iX(hj+f*Q(Z{tCgII1-'
    'Lsl#hPZlEN0i#s5Mkse1^xDm6kla+x1{{@JPHSGKjatTbiA4i|=uo|hA8g+;%dk0~_%l_4mnQYwpFjF>U!kTMGCaifmBXy+Vp3Lf'
    'e5lT%yOwRo{y{peAq;r!CpRZ|7cbrBlCu}cu<%I2JuAH#F+?5lyb6q)Mdxa|}Y_D|XgzZ&MnvqZSBr3`OoaRZxB>!`UCqYTdXPPH'
    '%OYPocl=!DZGR28bS0j`GT$gVTS6Osi;`$zwvNlyIbs&A*W~#Fvhxyn7)Qia?@~w5H8kIG>rI<S(P7rNPk$HBTISLt6RV;g?2y@N'
    'XOxfw4)JXDSXL>qslJz>v({ba*GJ0XH*+}#z*+}%J*+}$e*+}%}*+}#j*+}%3*+}$Ow(h^@Aik~*e02HPbk`d@WRGp0De~X~pQ;?'
    'aR!N~9s+DQuhvL?9qfbu3*v2MfT4rArw~*~rJVN$T5eV7Klcl{VPH_!NTdu@)guo@NBLFT@9pP^Y>Ii;IOh@QjLOKH99MP46Ss$_'
    'TY)?;>iRMTj)+Ny#M98LT7N)DRODl0iU_yr3eVt-byx?q!4}NsHEBDqk<;jo>_f^6^xFJP%<c1X7ksDHMN3KWET%7Xi5_0A6agU@'
    'C<MQJ?vPKNdkM~F$F)}~FBX7je{6vq$5o7a{Jd$M$&QJEpmhGH(Oj)w{GVu86x+^0wwaD_duo=d>l+t{2%ylWJ`4$-'
    'LVp8+&u)7V<YgD#}4KFA>-rI)1C?ej+hQBHz-'
    'q(h|X<+hm8^(IEyT7zyycbjZD;sw9Imm%&8c<F$d?W>BjcZYINMIZpxXK5U+v8wKQ<&P928V4N8Oeu3Kbd58FohOPJ&N?>>oqXl('
    '}ou{4%o|vKWku0-nGrk9K%4-'
    'qFrZ&(WbMa>^al%CORBmJI;!RHk=h3X}?*qan9b|6=|2UO;ZyMmw<+%%re50VI5BGN`plTpQF!jn#`VPz|LA52_--'
    'Li6own`Pj@^MMYa-zK+@L;~V~OCvW&^i!s5uMvKVj`#4^o?zB%DS1FpfT4l5od^QVd3|ux#X)HW8i)kDjHp^)|{51<|XSnMM&-'
    'wI7e%E2;paTLKC3BkUv?|&G7B?MG0rl;4n>7LV$DDXaHK4$6G!vr`z&j<0>rfw7Hi>H!aNy`_fla~ZqNKW8g1|tx1`xpf4<}fL4$'
    'J?#?6CaPN!5|=YlfF;Yvk(x&Bt-'
    ';I4<PE?mq2$=*IbaR}SLdF`4xjK3oS##WwGPEtiyl4nRh9B41Qx&zVjmjD|NOIZn}d5q)DmGfIToK9xj6NcS^-'
    'R*n4`V{|5CM3j4(V+eX@9zWlhGJqlwNAdBs4P@n`33%I>L!N4Djou~fB(VNpPH^E9792&F8f}62rWe`>WSqa{g(?DtrEmL4nh*zu'
    'T-e(SBl~Au*vIqsV$2Er2~1e&6;O=uhL}vUeKKkR6GMo4J-'
    '?29V!rIJw$eB_$7NmBotQn1f}BLLeoe=Qg7x7VMS#wSYZdW$AFfk4xRDPxF#Kr`qdgJ>^61-oqypsR-'
    'fF|!ijZ%QL{R&R?>FH$PJDYK+{yDLKL<5|<TQSP+f38J@=Du043z}1XcQ*3ViQZQOMCY5ItRpOu6KBF=`!layPww-JN7q4_&9MXf'
    'Q2_}Dn|C>PBeOn<kW-'
    '8hBT1XZ!v)E${DtnYz+y%x4ExrK1NV#)irnqMkIhs6bw4SOBgH@z(lCs2~653{N4#p+Bn?B5}pSwkWAB0zzHFlrk?^{OQz{(2-'
    '_vo^h@9-'
    '$tV2^?k?fcQmZDKC@7dETi2=<kAX(U(G4H2x5A2XOPu;*qrZy<$=B|J+gl{y+_4b6Q|v^3a3|qka<bK7EcwB)!oSpztq$YK4~`f9'
    'rH%6pWH+i>5?95<Y!+>{<lWQ*c#mq#1|yltys|TAcH%~@z)y!jyP61DWWu7PM~h8Z+>|NW!y~-'
    'knvcw$9zp9ie6_bnu)iH&Q4~K@`7y$FJ`Qae?eP7ICK7hF7`SKKw0kTQKW~-'
    'YX+VicK;^95BcO6t?h#NqNA8i*B~CFoUL9Jmn~@chTj(F1L1~dDtJjdb6;v_9kHl!*850v#N|%KuEVMF9cu##81|ZI-'
    'S*B<XNV{4cZ)3;F>romcy4x~y04hP{85xA<%WJX!HwqWi=q8duagN+XrYBC4n@IJ<S#lG(p0Y!=z=Q=lxVFU3s_jsD%~gHgRwcn*'
    'B$rQr_QMwZ3T%KsVnK7nP~f8$><0S^)hg0DLM~dY5W4I`eX&x=zCY%RRl@asm@iiIk~d7Vtr14;BYd$|2(^#&#X9h5m#xZ!=!n&H'
    'V@v6nTmEHbsXt>b?n;v%WJvp)ST0-YWH)5u2doVC-raQO_Vq}V?ZCp{+Q(r?zS`c$@kbfd1vzSXg$EQ{pkAC!j7fUVCR&7XoK1|w'
    'RGdxph=@6x*g#0f*~E<y9dkBuV?@U_juo;jEnbern7WE~6R7IfVIf(j!&&Dp8~UC0)|cJUam#2ImrPhnz>P^tT{dR3t5U)swNv`%'
    'OAV+IR`Atjm>vx)`RZ~E;f7UwHP?WnV>Mr0fyvUahOe%~WNBDi(l<`fsqt7yW@oTx>;QVPY~Pt@nV35E+ePq|$9%)+9-'
    'CpaFt4RKe}SbpX*r@6p0ZIHK-'
    'Z>OnSKbNUMD%kN@n{o+f;l5!%4}kaK0Bdh1fwDp}BSdMmR2<_a))ay{;tuxzCk^Kli(m@aF+n68=2sO2VIqTuJ!zuuFtL(`wx<zD'
    '9#M-KXh-'
    'm|bhn*aW+xUC&NtOKDE~H4iexvu#}U>~}I01Tt7WSURv3WFA9I?e^G!D@9WMqULdEZUZtg%QZ6SFUeDebx20^^*93z^Pf&TF=J4v'
    't(XKS|Ho-BW(<tV2wcWvV|tz`I#YUHS6BQj(vII;W+ZfA{TGpg{MIr(q3ox>iiG61SCy&-4^~H;2#1qW-'
    'R8oTO*Mq8FoPY|@qxJ4fzd%|6y|_9gnXgQ?S!jQVvxZdg7g$W$fy%x`Y^`J91nr|FxJcD6CtxO&dXR2VSI`;($yZZW8($>zI|i6+'
    '21%edD@YUg)ZiL_F7DQ)w-'
    'x=Rz@U94D&SSFa`B((v+4Fn*V!D=@L*A;+}4`Vh6eKkpw@Hk!rGHyL%J`MjPQ9D``@`;e{q8uCEwd_VqpFz}|2%Kkvg1tVpxiy2Q'
    'SgePlo^+=v8JW{RZ{L5BY=tu>H^e>WCo;uMD?NQi@6t}|S3OVR~^nyE6<8{<%&iSL8?(qwE))}zwTgHqL)rlm>C_H$tuBkgaWYSO'
    'a3Jc=aSvJ{rOxCxM-;XVgK`wRK%eg~R{@9@<F4m1tl<*NrBm@qC<%8%l8Ey$3Yd_(J4<R{<M;tV;*x3#84-'
    'tiqSX?2I^96_M!+|Nv-ZHe`%N0_EQZFHUxwWUQzZzkAkAl3v(r$jK31Jdp2rM{Z%<AE;hX6iEA$D;(dJ<IGCk4)ui<x*)Y`y5Y?e'
    '8evs?G0=ZMI;evy`LnceQhKljloEtls1cG)sR|0iE1MW%}C8Qt_~&7ix#+~0(A&{w9ea7_C7$KElSimGp`e)^*<|*WLZ(Hho}`H?'
    '(}h>C1n^s+$5}yxUpjt%iwphGwFWe0uvIVxMpUjvgh_FP%Mi{J{nxyVv-'
    '9`E{jP%2F%oAc#j3SaH5^=M(%b%=Rlrx6k&)@ea2Bx2IpCd4EN_t$S^a;id-ffWy*=y9#isAP-'
    'KRlY?F<b)vMDm`}GXz9bSes4sc=D`lKW&a=)Y<Nn7zE)e3HiC9h7+Ud@Ol+7?NXj`CSDx2;ZL|4m=gKz-W_#Zx5v+0o==KjX^Y<}'
    ')MtzqFK}70Lf!<>!rEg1allCaje72O(>8i!=x!^1n5G?Gugy9057cI*RlXJzl9?GmzJ+MTWoeGD)%`<(8}M&QqOM-'
    '=O*Vy4N`_AH{|1@hmOKc(R^?ftb3fa4{x_!zcBw*cbP(a<6|t(_8%>aVgg}Z}lkNa}(nI9+d*=XeQS^8li(4&TY}gqi$z=wC<?e*'
    '#W&fx_zZhiJhb0Hes(kV7$kNcZ^gcK5CjN0;DUkQWYGqbS9;^)sm$}MwvWBYm57{2r6p$KTwY3#{eXPv=WQm?vXcMC+y?GLDrsHK'
    '{@Vxz$3N$kJ8<YGCJ-{?q+MWKB>Fe2E9+}V75cQQ@R4aF<}qB0=_X}Prd?%F<~FRg0D7VU(-'
    'aMx4s@fuG6~w;Wjo!RV&ULVr$=2S4-lt%_kwY=*-zD?Xl?3PH<OR(_N`Jw}UC>4_GfD-O|f--Z?hh?kUPUS-UYzoUKToZJaqspIw'
    '|eNS}S2IY^%yapoX>-YCu-'
    'q|Y12nS=BMK!GOgZJL%W4%ChPA|KvNDt9HQR_cM!9kJJxeb$|^*YrUp>VJ>6TfFMRhjbR|CXZw{xA@-'
    'd(GbHuzPEX#zPa)n8;$wo-Rn_m60`M?N0C5hUp!hY*ic1WY{Kpp4x%}6`I{Pww7yN>=5nwklB3fGvCNys=tvg>B-'
    'Z)bPO|ASU0qeJB|f~Hg!Vfh2bika(q&+<mTc)GV5?LHbdXg_cK(lG-&F2+kZ&q?JXkDcKIgH>EG9(h*s!34I-'
    'cyk{OJ=s0brT!JRqJm8O?mqqY^w+*<BgUr>^g6DT((b_ZJ8By|%y2^T~WEisj(11||mT_I|*VZxag3lTC9baE|PLh?|RAO0T`t#e'
    '`qh%h_LUlvMmPfajQ+_k0}Z344f)kN1SxM8+q2@`%bT<jEl_vydl$C}U6iMe{C$p8H25CWD>_Mq|HuN!L28nb^6A7KH2hN}ylK3|'
    '@uMp+@hF^yM;NJB$l`c*6+S;)!NfbI9G2ji+N!M=-'
    ')NlPzsaRpQc9lEgZsaMEXaDYzJ{OM&eryLlO~y=0Rw2ey}N@)f}L<BTMT`z3|8WASUrpzT!LBnh-'
    'J#m!3PQQV?T9>uLn<WbzFJRZgEl}^07eRwMg;kQ1{SXW!~2dYj`4ApO9@BF>G<n&@AIh$X5H0!NM|7A3Gi^z{{jU6lk)VIN&5rI_'
    'OVr7z;z3s6vN&Z@A^l&niHYs{Bne;s<`hFRKJ~;aRttx9>TCL_>x})}pV`<5mT)_csM#`oXgCoT|Ab#;R3-'
    'x;+$0;#Y=CfgIk=7`qTd%tD1k-f%W(O>U5_GiG0|slOEjRy1p1xv5G{TW8vREmuyI57M631PvE>=f)6PXW-HPIbK=EGvGxbtFNv5'
    'uX&q8U{Bq{peHeDt=w=GI@j>sUeP_2sz2KvVxcX5IR*wIhyPs*F}@wf<5tlhxy#XQB_4cl$J%S0kHSdwjVE5#i<rD~1a3^-HM3D-'
    'Ghys3a>5;y=(9p=@)@G@^;MVi^qE7M4T`u-'
    'oC_Nr`BCn3XF{ZJWeY4aZ22mdhAm)SIk}u1yxY4`!p@y<9e^_DT<cl_7S!mh6x|yq!)Z{g<|Smm^Y37b*?YD=vIc8!4ZTFjp+*eI'
    '88`5*2<L<#CD{m1OS^8r7W@mzW)_G25^f%|^puNwIos7%WNqZv%rRHTxZ4bg!zY)&^5<rw9|kuL_1)FRIqZkT0qlL+D;potW=!8d'
    '#VU`3rkZtn8||E(5(?6xXqaP;_0lo{V@`oXWEi;cjV_c<>R8tFXgB4aNmv@)F~9A$ZZmczp+~UShny3sNsJUKfGWOAhzNp!AYV!z'
    '_?xNl)L6VFw=>BB$l-()}gF%)iVd$AmzK=Sh_zfXuQ-'
    '5$CimuZa2ZdJ@Zg&y34iON|F!ZtenUPnmsA%fVLm`?91v!Ia77nKlvjPqxZE@1!U;Ig+Bytndvjj{%t&^XFn{Asj)U#r8Lft(9%?'
    'Q7!|a!>hmS@cMPx;dN8l;dOJ_;dM*d;dN`-'
    ';dNWt;dOg`+a=bl88qZq8XyO?t?x*iO~ELgUfis~=42NaE>)uVk2);&K3Lmab}YfoQ!&yKoJhdu0&Q#m1lGHSZ<Iv1ub_u3{l(Mc'
    'BEkcd0rN2<#G%)trp+v*#SJLLU6~ho-'
    '~czIU3P5VxgjF17wpK&(ExlpAs=}wGRP<6I{EUO5!$gXM;$Lp>=;9VM_pnccHs(LAPK@`72jR1-27K8KvLyW-Dq)|6MIuELO$?v-'
    '6k{5W!q%VbW(pTmLS)173a%tLK<F-'
    '+|4MFi5zc7O1_#S?QvNEk0##<St)gb&FQmS;>aHNw8gbNGiB{3k3SCh8h4u+fwFdOYR*E#pN&IT)0*0`Y?;Dpi%d~VWniCfVpwB@'
    'SQ5^9#X7C@)?gd8W6N9SXb!O9JDMq-M3JVFQ-Qr&mHD?W+uaBzYh5(hma#((#28+e{+!|1onnVLw29yIuhCZ_fB9DQ#jy)6u0~VT'
    'rB!mvj=T+PO=_)8TMc@ezypQ7r<eU>9|Q-'
    'Mh65i0308~;(yfjvmwhb&K;2z+ZEMYmkgjc`DG_?2ZH+>!#Q<=7tKx01xe`*m9a5@<>$DWZO|D=EQkj9|^dh<wGn$4%vc%E+vR{i'
    '#w5CH6wM>oDgZ#*`sXy%~L%@4hErZ%O63u1Ixt~$Ci{9!nuOOi3Hji%oD~)ddxX=mGNagMMJ}+<o#^>ej-'
    '}=0~{X3tRx9{+ILHqY!qBVr7cX`D!zw~53@-'
    'B!LB`!sO(S&T!JC%C6M5b+#fk%nW285BJ4`OQypXb?(wt%I%t&a=`U@Rk0Sl6=CV)Hn63_U|aGpP}A2=huEO7*{q54&rgU{eaJG('
    '6tShdnhs-rR@1bl1WbKJ2Zr@|F}$Xso;y1r!=9-'
    '(ZT{$jaw!0(bh{ge^q}kH++K;$HX`nJXR;DGuf+D=TS4HnmzB9*(i4zLo2ECuuA?TJM_r-~<_}<@z^U(mJT5P7`f8-'
    'Q0!_gh5V54BEWwfL!t*h)8pbH7ZUmFV(>EUY7-'
    'y_h}p6wsC~{fh4@HSc>5HatI=i^j@$Ez8q*?>B@oTRjwRpUhB%i<(FLesSigR&13(<hhVA<Wt>d-'
    'C_&*_wVgXf%ZLrPSaAj0+#{6Zk${!^$YaZodYrzTIXSL%LdUt5I(kRg&(s?eY?WDxA!KK-'
    'P)$1$c_LA+L08UuuXg1u_ZnBua<6k~ew&N8gUaZ0f}!wpx3V(1CNlVNC3jsUVB!s<Kl7n5N>;cwhWa^c8RxG~a4|1Y?H~nR5_-'
    'jX&SbWSpPMS<31#bCi^cFxIiNG!mN~bXF6G?H6xN{#AmogG3_wQC=waBMBWLsoByFXM_v3~_@?FXr?Mme@I^IxFzH3>QU8`JF6T!'
    'pPAq|JcwbF0+VQ_5iEnQ~CjrBP=iVs`kgwE`|r-qPUNVe)EAI8_a-{aI_hhzrsE?Y*6s|9klgZ_)H*>0fx+Dp3LiW?-'
    'H{JRT3@#I=`yY<h>LhAk$oLXI=5>{}g2k!iPR)|oG>%dI*-'
    'Ab&}i;kDBc)sH>5#CN$!YiZk?s{d(+)(Ntn<{}~Uwug_MaLwU*MvQ*<WJ+cXLsamo7e(l+``Pj8}PgQb@3oMFe!K0(ne)$Tjk1D5'
    '%XFy5wcc-'
    'j>f(^#6Xa6fRnj5#fj&Ie08lUZVub<fxpfaw~*K2Vv8wmCx@=&pDQ(6HpvhI;!;3y4QTC{S+=+qBydc{OJU`YZIykOU=@@15F&b~'
    'TH=Zoy(&Ja@E)R1-o<5cWl!}AvYNX~9%`XXSg#_Rxu;~F7K<wXI!-'
    'h%aN60vw+oB;>Tgac9+}&8l`Yrogw>o3L`7$|jd59uc7AAuX_%s&A6aJ69m5OxNDbz>(94*79su$Nr;Bm3{+QV|UufQf|0=z>_?)'
    '-Oob1ESNr3wxgPqaSB$PTYqEy@gSf-;Y_U5XHI(Pd0t$m`mQ8>}*p;EIY@$J!({C&x0di0zAn6F-'
    'RIn(HTubsDqzO?n0u$T7U67tgKTS5rha!aU7du|DHNz-'
    'j)id|<*K0igQHjabK#AioV_My|?k(GVY3K=%Beer^GUL7tK|ME=0&5>ndk<+I3y<E7PuWpNE3imjjf!NoDC2a6hr-'
    '9%ssJG%Q2v~8Jr30oo3$RXc7VM_tEK68QaTc(2JvFc)(r(qGwl&BeCMV^?SXW=j+@ec<yP$*vm701-J^>MPgG%w*BHe)`%Y-'
    'INf!*!MgvxXxHUPQwMu$Ktahy7@>mwELG$p)3C9WjAy42`Gw*Aq)s~g!4z{*zLl64@)QFyykB)q!9=x(xu(C4em$qvRc)40lQT9)'
    'PgLz2y2sE}0+&J-6Z&zgZy^veCpk{-'
    'JwTXYp=RqaZE+ZD0pNZhWa;IK)_$!WnNB`3!PYk!=amqiZYFdrDKvT<@smN>Gg>6i>bv3nFxULCobS^ut$^vtYs*F}0}R=FD@BQt'
    'B+jggVLwo`3zq?XweeHfPv#~!G#)r+3}We4dHEbT5@^t>57fU=unn@FZ|zg8X|)G&YhNT%|TmLVQ?QiizNhW)(etgf};0IxZ#>uf'
    'mCYkK(x8xHbXfw|GbY;9$0^cmPm!^5RRSV=zM%8g_Q%(M$FCd!xArdi4!_Q85fXkYWmT!l<o6xyz)c(dMvFks5k95$oLbWjQ{IH?'
    '~qv3X}@OE#*HAB)q9Q(O)pZvr=$oWWZ_G$b$oR`6Jfy}1q4Ori<5gMLt;_a>3mb%jz_>~Ex?zgZj@T%~jj2O4RW>x%=Il1;tF!h='
    'YP`+6HD`B+S;wq`^8CU$*mFY}bN$h|kN#(TIa1=VTtz2`ARDA|({8R{%aH(F9l>W~(-'
    'XK*1+8N}%^mr;pLV!E<ZMvq~tCmHDo+A?zYu2sg4gH7s`NmE{$&ndEWPGQSzPT|UAPT|X3PSGe+ImJe0Ca2h#%Vexq-'
    '>~5~=`g;SGqdPc3>Pl5vISsToE>KjwA_cOCea48hbIV(nlvNY9-'
    'i~zEK}^j?0JpwGpujnJf#os(|qqOKD^(^<nwQQ%DV7DMK4R(MH-wyQ5ANRHXcx*g^%Qo8~M~oY<V-'
    '~8jwC+8VmmBEce^0jb5V~DSM?t`KArOOG5c~8}6u?`nE<!msIU8j*~jFCSk8QZz8J^_K6cr7qc98b>VW&*M7u>D|JKZnQ<h4pOPh'
    'QikG*(GMwV&Ev^fnWoUg3*y2LFbTxqyj^cFnaOe22x3BbTMAxFtiAHuNht5VK_hFCH=_RD=4ig(GK1ro364%sh9Zg3gE}`j-'
    '>rU^jm11V*x#4Q}XB^C70E{;BB_;y=*c}_`3pj>6EAkR>`0*@3RU>n;2as#i(YE(D!q2fCh_mB9yjQnZtcYfRmxfNQPsZ+fiCNG%'
    'x+`B@53;@R;Yw4`KqYrR+2+}3M6jtEIy%x;Hh3^4A$HG*J$AsDruo2$W(0b`ei!DXbnxviT-8+X+ar#|$?}b-d^p_{+sJ~-'
    '1<DC?ExCQ!dlVH!qT%^Ir8>o`tO%d*Vo6qoPkON=tHY<fSdum2(_Spe+VC0HyOq%_w>NU#8bsx&L%R6O9M9-'
    '9q}FSjS7oIfXqn(iu}4q7{^@)hexFX`X?6s5t-a$crL1Y0L<mx++zukR<Ao&lMglwn1THXf#MCMy7p5e+Y`T?4BP=h+KC(dvO-'
    'f#K&@-+Xt-0X9C!EZwU#txXCwT>vghJk_HA*SRLVIK`(7M7v2?_dQFRwCe#TCIt7qgPV-'
    '!xaOBHuz?jlN~Wok?b2FyYN6l9QPgot|F_At$<g>P|Opa{^zxp|N=gw>#)_Q2HbxI#X=Tn&-'
    'm)qIvZAqZ+z}(S*?rUBgK6O+%Liy0A?{w*>mIZ9_LSHo|rdUD3Ev*uEh<8p);Y(1_avW9}Hs90+4_4ju~}DAdi2`_ZVoMbQ)}y3M'
    'NXLmC{ZH#U~VGOSL<t0OC4Epz1{$x<u)Azi3uPsoP9(F<D5vUG<p_^euWwnr?)tLQ$|2)}Y+9qaW99|@lMYPFB7k0r$(hC8h?%3-'
    '=wV66)W>LlG@1XdH-5U!4hcejSdO<SArMj8JQa(>tR>!ob#l|JB1n-'
    'K(xhRv`UZ#j}(OUr#U*(@$^@0np`p_Fp3Os;bzBfC`$OJML@Tf8B-^mn#+Qv#IV+u|L`WcI|JAY8WpxQp|*oYr)4-'
    'kH;yF3vl1w%5gZXU_JzIHt=9PZ!5@IrZz}m~Nf!wWTT8!4|TlZBe>)d`0eA>yyO|6rV}lvWdq?5?2(DXpq?zgs?nP5)En<{wRglS'
    'j(gE8v9`sUSmIu!fPz-QTT~vF$zDiy~k9j6eBlBsrP-'
    '@+>(uEA(OVjhxNXRIIs%|%waWUG2Q8!IXaQXuNJCE4>`=HpziIadgyx_;ioRFbka^wUhA7|xHt7^E@z)VCk{7qN_v+@+*ZK9v}+^'
    'o@8PT68gYjYUwxzz_t)^%M;md64POm|z=YA!+pc(9GEb-YlrPLp{S+^Luoa}(NW5;~P4Hx|`vn+)oW2#b+PsOxadk<J6-'
    'BvRj(RV-@miL1(AA4Ca#B|1>uuKMOKg^USU4Hg^5p3MU2^!1n4aK=ZEwU49eg$2Mw7=z^;`>ZyBA-'
    '*>xy@!UHlVXv#10h%H;f~eM+8IVQ5z)Nu;X*A+&IIb1D|<3RZ}FeO!%Kki6?-NhsTzR7%NHzcE!hVkQPOlU|zI@-'
    '`&Yn6wXRH20RI#QA6pb4*S5Q@Xi(Jt3!WSm`n+SaiP3LVK_XGuryHMNF_e3+fF=CX2H&?>I73<C)^sVXYG%^_Lu0YjIn`%MLr9I8'
    '6A5!;U8om!^0vg1|JyOnqjlyBe9L9%ZER^Z|mmO*_~ox|mRwg7S2m^Wu#%Wn6_nD|Ebr6~ti6jZ7k;`#i-'
    'f|0Y9|o&uI{3lBjfE(~#6mt_l$wGeEdu`PkEGnOT=ZN{zyw#-<Sz;+p%64)wZQ37moJXk)l4b?UtwbkprkGk-'
    'W)Zh9(iUdqO8MPwcn6<jxdNhQV)?rPU-RJRk*&RLTVIDX$@{q^7K2wju7U4nZ3-'
    'iTR;bFF2?|0g27uDu_+CFuj4L?$*hTzHU?47qFV%}QQPT--<#?`=Dg#w$TE&2qPv_|?r#U-'
    'Q62y`4Hj*MRJ39zIg4<E_xtI}FiF4BrQflF=pICCRE>|rQ)<?aNh&4}%UyTcDOikpLYu~pkJiUs5dwWE9?nRo#!Sy}SHyg)lCFVG'
    '$&HOjo@qQfb~u+plTrXfu^#nhITb=NypVnU7EY-&cQzdqaXC-f9WSv}^$D^0T<Ys)=j@i@2g2ACnZs-vt_^DhSz%VyebQrsOj*Cv'
    'T#ao9pTPl|iOmfCnyETJ$)_a>IL?EUiE?g4o@_h6}IPs@^ukd+#SiH&lDUYyM;Eoob}b`SF5T05BJ4E>vWSMZ?2Fic;@eTv92f=P'
    'O<myNI~fz)_%qrNF&KG1Awg-EHfGDwH6=_{1~C8g8#_B(Yrg(u1@{)H6iigYvU!c}=)E9P!HN@AGZOJbNkN@AEjOJbP4N@AG3OJb'
    'OPN@5sZO?N2Vdse1@pRN@@97bDg$q^Rq&}Y^O6#6{mkmoMvm~1j!vX$}nb!afI)07g;9C@9UxUvD6S9SUkbkIee9_uNt$hSy$m8I'
    'v{kn$eg(0!Oip6?9p%lMEHFw#>}^h8RGkLOFZj?0&79iK1NIw4=Gbz;6$>!f_C*2#%>w@QiQbtOxj$hnYx^$;e<Ue}t({hoqmc?S'
    '>p@Rrs*F7g-'
    '>o~CeTlW;Ga@b|c^srr3R`>5N)QNDOx9*nf&O?i}KL8e#X<nFSHc9OB`^lGdSmbZ+s4vSZ=<@L&*lv_}RUKK7+3cWhatz|QNB1_D'
    'OAac-VBjHc^aJdb~dxgB4hWq%SebN`3h5N0dzRkk}R*}h;6dUqRHm7~&?csSarzDvS0gV@mR)=HCgZ1cj18Bsi@EDNY?Ne1!lu$e'
    'dSJvB{&bm!}U37w-'
    'Jl|W<dN7H^9=m++Og1~ib>C`F;!Zy7pY(raUb>RkM}`ffu)L+9J;g_TUR_(Ct`f}5k?il0v4ozuQ=s4FS%MF^toi;!PW$cS^1@E7'
    '8-uk?OT}Pq(^4^5+q5A3h5^RpAA8dTr4c1C4UEaE{<gtlSGGOx7%Ua#Kk%-'
    '>me*QExgTLX?q_0d*w=<DQ<mVvK0GsmWsiB$x@GV3SsNehn-XHgO?1pmPz+~M>`MJKI)d4WnW#!H6!`75s17<ot(Jqu&I@a^s!n$'
    'Vr1#25n(Pydfb@PDNt11Y5s*G8BWbcrFalCeIUWt##yv7!8t20#9$?Q~dE|C4_85d`fw#wbUVdW~pdO>eHjpH84*>3cxV-'
    'n*%X@#Ly!SWDdw;9E_qWS?f2X|ncXR%TcAU&!a_3G$ri}HiJ5u2Ja_Kj{vyO&#fJ|_r`gnoE`AW#`Fu%z4gFm|Pa&1g+I1(B(C*;'
    '6d-I?u`inAk5I%_(tWE}=@#HT7j<{Vv-)8W%Cl@W`v<N(;3$E`VUI5lzJk_%9?wvi=F&vq^LG)z!~b*UjgAWC+bA)_a6?Q%m-'
    'LC)D+Lx^9_*%gM)x9H-PCVWuO*;R%Pd;Ofvi-'
    ')W>qCl|j(sa(o`*2WGHV*hQ%U(7zQvXWQ2CI%smegdjG@(NvtcAzpN~uvyJ?t2|#`QvPI`st&OM#HaI~!Yy065^mRz{&3z*ffHeY'
    'ur!4_|I&+|!p^8Taz#R>r-'
    '3xs`DrUv6dGx8CPM$}mC)!bG@?D6L(5c%q43_p1%^X<^R>(~u&M!Sq_cM5uibuSoRZ4mnM7L)s+N4mG-zkxZnF=+%y)KZ7Prz{_7'
    'P_9gC<{@TQnd143&scv1byHO<c-9B#|54{c8NZC(SCY)Q4Mkx<vN8F;>$?5voxi`<3tC?&C_tHshgxAwaoa@u-'
    'ioDa;(K{Ql?77^dG9qP>Cw;ighU2`}g0D*X_qe(Q>#dl^f!-'
    '*}CyYRUiX0lJ?%u+Xg$QYVe1yb^jl4n7o`a8C@D!XS8=ZjnM;Y1C1jh00DkBQ>8dz*0y>Mj{*USb)#)Zz=@uE{#YV~U4*PI42*qz'
    'mUxXrRGJNmF+GUvPc@KlpdO>dxypo5d`10FuUm#-Fj&^&*huNHgk5IKjhmUs}>e~GV7A;j7U-)|^m@I{UA!-'
    'i(%E>T|euiNlrrF1;XV_6;p&~A@qZ+od-JA<&itM2y7NSk{xy~$$3Nt)+N!u!dBSz@lYH-'
    'n&4(8|<Nk=3%icF^v_lf0b(^YV4>mgyMvmzJ@~PQ8EZZ=I^EhUa;dqqOCIGNzZM{j4>&?L*-'
    '^9yD8Qe3$v`s6FO0XZpC$RH_$Kgp-Z-tnQ?Z_{J6sQob${-^Wg*^j}`|;$S0=o9{K=qJRcd=td(N5I*E?kTf_RzU^?h-'
    ')PLL?VukVd0KrL*bt@k^ca}zlKYN45Mo1Om+d(1k2=_dow*W<-HCGWy2JBcK|^@K3u=gw8Q6A=24Ht+qX{A~&^#H9x-'
    'cegniwo8lB_XUwj!xKo3p$VTq=9EHWi4WXcHW>N2CQX>c^&ADI_VV+RcZjlV+^}M?>8Y&i9dinG^1@BA5BaL09@k++eTC7FQ~fVX'
    't!F*Y?j3)+s9vZOCf(gB>~YS=p+L0i}2(RT73rdJ{tMSS(V0NAf;G?^rBWc1Q9<Lhe{BN!*UfW|5{$x3UJ1V8NHw85e7`Mz|4x`F'
    'XWP#3=U6_Br=MYb8To$_76+%vGQE=zj*}6!w~W)^&PZJ3j2gYx%O9u7)geC&2PoTak@AOcD-aYLO1&M-'
    'dOgYmpC9ry?N4Ng^T4D2s>?G94_?u2>nqqKJi}<F6{rqX_w2#Vf@+vZM73!q?Fe5cX~8_FB;Vb7%XXKrES#S)^z=I;Uun5Vt*i&K'
    'Fnf<iQtxaYGyn<LywfV^4|w;{cEn`^U~kW}^fHpCm|r35vpcXiVmDrblLSgU>?as3A&>ZX=zlFV}^Sv{AV|YU_Xo$t}7BnBV1rj?'
    'ip;mxJIlsJp(0ETYHrywcblr5W`<d3EUR;T&IFA19A+@g-'
    'l}7>+gCI6qdI;EzT#I9se^(df;ubsiWB1%mwyl!TIJ`8gN~#pV491djDN6mrJPT$~PfSp(rr7yg;>&=Cz@Dz4TmIk?JB+te(P?W!'
    '86?5gT9<@#S-#~`R8sB;_PQ)F3LwZ7nsYk8iiozPfu9gnBP4QXnaHX-zEtZs%Q<<`}3q}#e1j#OKh!;vY_?QmoXbUhqtJky?nQ51'
    'L~<J6_|-pgCVdoh_J*gabo64gX`A_mLtPHF1CzTK$@OVdLuA+APd7k2XD1<f%2xharYhRjIO#h(8ErL*+?N?|y`<g<eqiZL-uQ5^'
    '=%Xwfn|EsY&<MbW)trx!{@y-4Zv;)<dN#7-'
    '}ki25EO{q2q`ia2faYX_DYJ_WZpaAQedtphg}W8B*vDCW+p5}bZ8owYGOyx2sDuFLacVtcqcdy7z8FlRIaLUaNFEtQ-'
    '~i*5F96JqG!W1?%(cXfC}_#O9i6!zws@<IHHa=e81l-'
    ';W<QOQ=87gXJp<|Vwp97bM)Vx<zjVDMR)UNHB(FY98z)6IcHSetgCrqOS9;FDy?cbfx`&za}ex-'
    'T9F@3JIlKJSZbbPRF}pL)uL31+zT)~}b|`i;_Czgc?gw@PpQcImC(DZRCBQ27POQS*9Ua=QA`1_4-FjpEu?px-'
    't*c)Bze|5iBLkzYw3So$ekc(1Pc_&_cDOSrxNUy#S&Ymt5Zve(7;Z<z2EuY1|wG~ugW7vH~aDBwz#?om)JOC^)k96aIdvo`T*x>c'
    '7%#fet#eD<kTKmC!ymv!%eAs;4JI0i#L2j!9Q1XJd`TGhrzY}lHUK+N{B5S4O5Q_I$0lj_eZt|J?f<tS~km{1iLGDaf`3kt%29!-'
    '_6lfEmBVUMPq?gcrJY-PR=ye_hp`TocgznRiNo}gQq9|V1uY-N53#Co!o`C+i2VJkDo6JNDih2-'
    '2A;g$1aqfA`BSJv(|sWacv{|I^aN#<@;W$dIZ6OV@V*x<Mkf~9qU3-3$HMd9!Bwl1xK5m{J@W%SP|rk9AVA3URom$U)-'
    '^NC)z&xtu3ji_jbueM>}1d=t8p4GR=en`*i+hjhZXZ0Pj9@01cE*TFE&!!8dBBHaeXpby~G+(n>QRzIJ6_viOgq1I05i<YE5EdTq'
    'T?u$c<pJ+R3RN`?erHoyLaNrAEhp;B_qq?1d);IA@v;(=w>i4fJl-'
    '+^A7ygxFpn1WpL^NJfEM&$@)bpnn6u#H_lSwZF?{t06Q?ft>W?Ol%J3DPZ;Z<D6(L?sWbhRQpctItE3&Q_oZ%yV6z2K3$)(%V0k5'
    '%yZ-`tq^-SHaUOMr=*qqml4>8Abyn{OZOjj-#*S->{+n^ZKWf{-^hRAi-'
    '6+5%aT)}kgAQzYNWvu=YKm8RfwGsdFRZxtJe>vBZITH{5H7p4d4}KmtD@cCf>!5a*{K6p~(*i&7vR(s?-'
    '9PoRU;~?Ce&%JZ20EO7Q3EWg6QNF3^H~!%P8}Lw^5Nj7X6Xm8Whb1=fB0dnVG3XOviT!!L-Mk-'
    '10DIl^1|dF*FVVEuZs&GWXRFQ6%R5E>*9(B8F_SZ#e<AIx=38lSM<F5dkT`1PE7VPyk0(H+Mre~4i(E%nP19kGvgRsU%9cT^q%zI'
    '6^`}^U{Abm`5-K4zLUrf0d;77rh2;9-qj%_?s}B@YjSL+A9i5|KSPU-'
    'I<`kzf2)>kc`S}iuJlR|V_i5MVoe$!MP}4VuNbB}c5<2#?M`XB)O&~_g>g<g$uJHx-'
    'bpha=0A4ktBGDuLD<D>dsS%n+FuoR^V(GvKI&5gRc(nTJ!D%WjLKj-xx3j>WaR6<SLa?vbyLZFmdB-'
    'c!$K_A6&D`lTa}h)l;or0fmCbP0$SY>QY~2Y1y-WTE+4VlU?YhFPxsrxNL&O<Z6z=QQ>v552u!J7LL(UQhQvl-'
    'O5I3;BXBrvtkKb%;L{Cbq?y!fe;V0#?Me$D@u@9MYw&M|A0#FFj0qbxbxy5vAi`p6vf6<ojIGHU2SPu#ChHu??AU-'
    '1!o6HZtMIN70k_BsD31(*HFA>(h@Fu)j)>SBxoN=AhLPJwGGQYl`hakGW7E_aGJq~}CT-wzf{}D>@g9z=ue6TT7MH2(Yn~i0GnV9'
    '%!eWk6Sn%O|!i6Ui9{ZFDzDaI_4i=pc=L!eHJ2uuU9Y_G#Sg&=U7CTd0xFRvLumDC{E+;gAk(bK}4`3wba>Aw<`JbH7DMp?yCwz*'
    'LsLKguU}Wm55SC4jbd-JEvf)sm)@?kUNIVD^L?1+_N%l6!$;x)P`+<0j>QwS|AMj}t8cEmANj2tr(LHVluG6Q`9UrS5dSpiq){U#'
    ')gl_sDo3aL)NC}m5CeeTs?HDiQHN}dOyN2+QQr8r#C4aM>A*`gMy2aXZQny$qvDhviNeEM!@mq$Hm-_B`gng9eKJoz#kL(5dDu{M'
    'K5v^+2lP)}&H0%@;tZ!<M1$D8kWNlH>9k33mt?x8qXS*=v($=K<71I57I+qk^hND@84vnW2^8DXVeMT)R>s{^P-'
    'y0ZsSY1OF(;Lvpx>j8Y(;$p)FE=&jALEgse23%mM9n7qvR&3<LWid8DxotYT_xQx8lAc$mmJ{!PAY;e29fGymX|SL1UrhaHa1|!J'
    'l0ENqOdSwLcCao+|0B};L)M|A<J|D8FEUR9_n``zgDGf5X;D2oe_4j8Q}naUfYDu<~F&CsCIP4BVj*&ALCr{r!a{Vodg2*(^hcu^'
    'OKxVxq^aUi%E3MR55VQR5S3*sv0<TWub<Sohk>;ooWXT-o_Q~-'
    '0paSmdg2QG1<yrj1P>`X|#Sh4fr8F4)Z(&)mIf%O9V?(8lfjo_kGd?mrT{?QvI}Q-'
    'cNPf|1|`|T(&Kp;j~k!8!q_(%@+_@I?zeMp3u=jP9|ptFzm`x2i;+@4cj$@gnq{cswG?^9p6BKgk!#cYhYSvC2QNa8yNjr=PSgy8'
    'Tnm{4$a0Wu2V7&eZI^%AY(c#%MwPJ=(FoHX!SVgP;lL*R5S?fN1$Qu@siaa?~2F5LHu%ecE#i2V7{PuV?Q4b;S?YRg9rI=1dC&V4'
    'Wk=kyBFE8O+%*Ame{aELoD-'
    '*2F9ueSsdSJV6JKfUwyNIS*lf>=zjyeVt8F;dm~wemIeqLo$}M5i(U|Vi@3)LE^;fBQQIh!O>U<lu!?TQMV7UND!ri4l7)5~nnZ?'
    '$5;`q}NLpqyQEGTJ9LR?3qy`j-kpo3xlYBV5p%dBC_j@<Fx4m>WJ7oyE!#y@^-'
    'w?j<aln?6RXD$aak8~6sdqpP7_*ZjFI(%&hf^k^#KL37Sj?T}3b5s}FU!|hv}W|yrjF?a3^b|NDGf;ye!+(WMp&~M`(_!gdFWb4('
    '@UEOb}J-'
    'rc#dbLEk@}O&1`Jss}0e731F7u%p`>48Zh{Lx=(1}3^M{gxdAL9Q8TG<xj64o0~*KwEXSFpCwmc@{ylYQnMZ)bLX4V9ReNj$^}fB'
    'h(G+Z8Ff>^PXQIzCI8VFqbke3zG*Yg+f)ZLX8n#Gyqg)XK!WGenMB7wJN$6~S?6ImeDlSD(D8BQ4gc{n3j37|N@(`jv$t*mKppRu'
    '#ti{njhQpPX2EJk$yUqqWlOAE*(ZF=U9~t8{&~)=CgSrL=3LfK*Wrs-PlZ^~AU4^(Had<aKOx2unR|8Nify>H?`!X!NSeUQCMaru'
    'nH?<kohv@QOW0j@lj@UiESNg4W772TXr90A-'
    'D1Z0PhV0x`T1mJYlo8x`>5#8DrJ?P>16KLV_V>_<g%zN_OwD;0z*3tjv!eqTb?TBrSP>5s`@deJ3?b-1q+eX-'
    's@)yDH|)DUY+IAqdho#DN7mcA%&-'
    'v1%g|RZf}OYxJywws?Sw=4>KufL;c%}!LoJpQzC&0t_eUqZIfG~)6qI$K!wleovM`s7pmZURA<a<Q*`pjX`JZ4Bk_#bcvB2eN_@$'
    '9jxKkz^@%$6H_(o{5wSf%4^oCe{`sw9tz<PXcz*-'
    'w(z^aMl=DAt|5)+aN60F&^i6`NU2>8MgUV3j)OMinw)=C?Gs+=ZcN$Hg?cAOzBc7;~07J!9B3RR0h6v2(?4tcH5usA;9!qqyr@G*'
    '}~A)7Tcb9y>34$mq1aJP+QV5LXwY{Y38rgMzT>^k8gdm2q^W$-'
    'W2&uTfyhdRz^l+dgtR#|&})rRF+=r8(kUQ)t~ec0YNQ&3LMx<C4$>-'
    'mUZHZD2EW$QmtLRhLZRdFp&b)Pa6t1t2j@u|I<WR@;=GQ~CF6sIb8ZTO_q_i|nMl+*WeefTtAj5oy%;WJK;{f*(XJU)A>N4JQeMU'
    'Qq7_9v%%^oy_+J;S46<cu1QM~1#>-i^HBSCho9K$@g(_2*}qi>0nyc!Rc69%74f<cA-M>$Y_a-qsWLL)d=yn4*mXJehvodwVeG`g'
    'qZ1Y!q4Ql;vp?kK)pA%C?dFoa}^$VZC3s!lUT+fRnxOsB%8&!kuM<$wN-'
    '|#G@ehu;WefG3$Q@QZVP%S<eU!)P*(&nW#kE+@MH)R`UqXeBIoNmvM*VDquH?mG>zq>qHn4ZDO3|0~KiatJPY@NfdB4w0dzWIcB5'
    'Pi_;LyYfa)zB;C}TbW>EuUw#)_NC}7Vhzn9=Vb?}}J}26oD5DSfO0O`(=5!-'
    '1+=0{$_^nTukRCi4v#UY&6|RN;|CW|Wf6~l2e~u4dYX~{xF?F8Ln9ZIl#e3o88%^eWx~xNwkw~^JJ16U#?WHxPJfpuxOe)RqH@V^'
    'u+WUU9BTpbL-M1oIeUK-'
    '5Cb{sSmjLxZ5C(=nTdn<^g<=GgJ7J=c92PS;V%#Tu!D(ikESGRnJb|o_QuU1Y;LJXvQLhw@;170JI?yAr<Ew7l`tvMMDxGA!92V!'
    '5$-'
    'cV&Kib|qPOqxk8=qYc41Gc!2#_EmVQ7&$g1{&!(erGKSG@)h7^$%m6_r~JHL=HnMi3nO(BriYJt)Eu5`ul`L&q|Tdezu}>$~?_Yw'
    'vTO;q!a{xi5U~J4aHUXP>>-'
    'UgcXb>liJX(z7um&%bybMoulQR48M8%Q0LaDi(dm@hm%{CwH@{^WH(idYR%#UuVLVipo6Hfk@7k>LHNG3(xX84-'
    '___dlc56d$YwS&xqdU!zPtD&0~K5@y1<HkDuS@P<$}$uLG0Jdb2os+yY6;A7)z@o^)Cp(;0M98z1ZH>%xzO?B(hk!%wZa+XEKxX|'
    'E^pwN<#x>aBQq6)w*^<;W^r&0lpPhAEPrBsE)p_<E!izN!f-E`{`IPu-eI<h@MzQUk4pTXamkZ1PYDRKRtV!Ya;G>7wKKPg;Mi!&'
    '5phjj2BdR8k0;@Mk520xKbvA>_oR5`iC;l!;aNyAT)a^};_?@@xI@Pg8ONV{3Dz^f_rCFLv>cPdC1({&moY8Wl?G;@A||9lO$M3r'
    '92fFJcA|7wdK#-'
    'qu7LM+%;KDFc;gHJnXzv)l)pF|5mw@DZDV<n(8*RG57dK5P^{))Y$B&J;@3l__Ms*E3nEnieBA2GheesV|rwu1zi0^l)A3u%?IWQ'
    '#U(3+>n~t>EXuI%T5nBrM_aZVsmir<BA>Y=xT0{@R!@%bnZmQBxx=-tVd004>{t%<Pi-'
    '0IFw@A2|<r3WZsQYnfEDtjV$23zR7CmaktB^%q(S_S;{W6lznC?SDB@}US=t;pIFMe44YIDL_|OBpH#Y=m;2`o>XI4i0vFy5*E@A'
    'icn|CZRpbACE^;(@kpF^$H}8Y-'
    'tOOjGU@Fk{et0Ge&Sf<`x&_v`2KKblUU>lSv=XG&!ko5Avt8^G415JgoiVE@`s9)}D!uqIEr@zdvL$2_47{v}7Hukb(ywawp%y@C'
    'I!*o@Wijm)TV@dS`qcZ6k`p!E{*pl<v$eYap=Gj-'
    'k<AZohLg?zE+a`bZT@!~Nvdh{zsE>YO`HF{Mv`jU{O>c8RMX~vzmcSxHvj1+Ce=BDLM}0^^eSmHnmvYxQXDES*V0ohqU8&YW>+N2'
    '7xz9aBi%SxqqbnP4*}6UVoHK^){#in@nWxS6{Yx(7ztGQ#-^P_Q9(8%?H-B*Tji8-'
    'J)_G``)7TVmpvb@Yx3c`HXp9*^5MEZAFdnn;kq#&uA5qiE2;>`wKxGP;@yM0KDb8+CqH}X6iXbvsr2rOo{g$AH>zTootTr#)!Jjk'
    '>29-'
    '(tLZViDd{{?=$VnzYismS(;=itBpY@l@_>3&#FNS~qbh_{j#(95q;kxux+j%mUIiH6XX;9oKHqQZ^;FsNbjo*%)uO_C10LMDuwj>'
    'v)m&Rx6YgURhsFbJ;ll`rYQadrSzIs@Fcue#1boG1$PW{1!AroyS_b~G8y9sjxQ&Yt&1-'
    'CZU;XE%1j(qXXj4cTH`%cyzM<jlk)elbybvggt=19S?ZG~T?nzUyVWUa_dK*^}q~Bbl{cK@Rd7#aXjWP6%>lpH8+5^(`zJ+#xG+A'
    '$@b0i(`+h{My81i;HQPO?AgU*w*#_y!l#GPAPud4sTO%q$+U;nktc}Aj!ZjT)D5ijdCUS<^$!8KwFA)5-fNTJmUtx~>Gxe-'
    'V4gfac^aAD&XVsR8HMYmXK2P#h4?Q>sS7<}$;)$?)yIl`z@%+GEHU9;WzX?KG*>hAnRDe8?b>?O!6!1~fi2>j61_1`$DxAmI(?_B'
    'ZP`hof%jLow3+WMb`3=pNLBc*Toh?m$r9!<=L<VG8X5es+bB4SVhQe){3ZleoETAZfiT-YQXx(%l5gTDYi8sCyB_t995%zI^~Mr8'
    'E^P?iL@GB2i{b8OdqXB~ETPl@*#FzD+J@DlD}fJ57ZpY}Asq3y{}dmA9n_Ez7GF+q$_)_$`zjB2ZN-YLr3l`dEVLn+MTO2t2>C~H'
    '@KK4tB9HevdES&=B4n`{!A%M_iZDMB%pq0~xD!zi)X9L=Ue<H#4sxNSNXX9O)R$;Fvu(t*V3o3qDblv5Fp$0(;F9*<E@MLZs(oQi'
    'lnMmZJnc#Lu?_X@3;-XGr```OHn-rv&EKSY8F9rwe!u`2Cm(Vpr!&}ekz-'
    '!zUgun%p~^3I#%!=fttWA%s{8bWBl%e@gg|0_iadS14jjyT$89zx04rfFmE0gJ|hkVdCp_@1~(GGB6EOCj)Jxl@E0J^-'
    'D{2|jSJg?p&JhHWf>7`|4TAoU~lb<&ioAFZ!vOA8>0Z(vIcTR&dkD6OjciTb89x>?1utsu|C2k1mv#h!n#0^6G5(crZ23}GSW3Rf'
    'UsGh8}+xPoLV1E#}ADk#}9WEyn0T`hyAJNV;2$D1n3T~pTi6{~XB?*;UqH9H*p7Zr6Zpp#NNVy{lR+`OJ2y>FNKZ09)FO3QSGMmD'
    '-Fq19cQ#8q^z<6GM>jWGWo!g-'
    'uE{=yWq#$PGg?rjyyh*Ma^$6`Rn2Zh$;OKJsub#x(@zVoFHgDtMhlHe+H_jiEVttuew2z(6dMht{m@P1bTxYK;NvtinZ_QHtt`Vq'
    'MMeVXD;C5`Cm(gfPnMV7G8V`30PzpmG@@LNeN^<*f}HAbIy!RE({l}BB)S1vGe<?RFa*l?j$<^E+(Tm6?tLuBdoUlz@5rP+UZbk&'
    'q@{}phf(suuq(Y9Rr{a3-'
    '9iffZ8))Z#dg?^v<3gzUyVJ+6^Q1qnRS%>EoTC!pcGW)T#z)+}#sLl{0GkEFVYN93r42MjY_FekzS{SI6&~@2A=z5jIM4~9w>He_'
    'Q4HsL55q`-t{IU~n_LMqu+_}wDTEK(4?J&#I{_IRlPPIY1J(^5Wsie%N@|a>l9#a4=FfOd-l^)Z3-'
    'ALGbdF7V5Y!}X}u^Y^V6Z6A_-MWmZ9xRCZF**-wFKVGHiKY6%W)}q;m2yL-'
    '?MHQOc4fLjt(cH!3nAk#k+mQj?5~lvARFv&k+mQj>>rV*a6AtkhGLI}ug$|_>*;&ipuIpFv`5As3%6Uhh5vBumuH(is&LZ&c#DZD'
    'i7gBTN=Nk>PadoM0p>;$n}X5$eWUr@@AIf_6p-<<aTfQTQjcikN%+TdDL008p@e>HL*>cPIx0_w_N4M;Xi+ufza}=F?$Y-'
    '4=e2!(MC?Fu>hSv5m@5tM2GlR7(PA#UB@fy*2=(n1nl-0s&v3+QH2HI+s7N&VdnBGnjQD4ynw-GvNmtt!HK(E5>&xO!jnw@;CR}6'
    '{!h3;d=yNCB;;ACxoBeiAiT5Gwf8)Zzk-A;Tw3)Iti^of8IHRS#{8-'
    'uUTM#)E0;7qpxwFdvvsFbIs~8M*x^hPPy~9?mThMyWF7|jT4%Mj0AC2R@^Sp7`s_Snq^l5($8TPOl0RQp`927O1eLW5e%ePrbaDG'
    '>%lolFxB<+MTFIdV>xYbiF$2rj*o=TQO8ARr(WI0Ta=Jz~Dtf_FN3x_BNe8R>qTf^ur4H01+1Urj1MDWcOnuRP8_c9YFEc^v_F>%'
    'DgpK^BWB&}3Cj`u??Ls8$*?KK)lzUw)>PKB#oI7~G`ZjXF*?lL}R!<C+W-#QpR->eqECjwW3&D5MhjH;&1VlQ#;g)B-'
    'ukZcmhd{=qa!T+AR8CuCM_M+DFb`wr-'
    '`Qy@6HvLWhmI@uzlN8+cYlX${WN_9m<t5qKAQZ}wW><soceT}%c$e2<Arc&~ISx`9FDg&rDi;n_k<-'
    'gjH;3OlHRZU}tB2kbHeBta8CvU<<<?<&u152@0-'
    'tAG*(TAud`S8^<57>VrAHA7V{Lcbib}`0mUC>>Yi~~vXBRNwF?_O>dY@&<jCh<D>qw}0<8)XOi2RgBWwS^XL*zUjXwiH9n$1HKui'
    '{#`*S(66A(Ql)OfCkTc}&bLC~e>_*4c)F+AU&pK}VCjSmzo_CvFq_019@ui+KPYS?*#jF?3|_EKcHAY!u5{813!%?skWxm$Fg_$x'
    '#-'
    '~>d17iZTO8BGxfPp(8ywc%M25`bBtOVH9PsseKK0Z@VEGMG$i7W{+Z~UbB89P^ch~viRe2sXz;RVB=WnQdY){<|K`;56dV4KQ_oX'
    'v_@mRL|1=x^<kbJ<GSAR4UZ1bo@Uyf&+ncaQi)u{=c%%V~Q)6QwpoXOVGh`%B0AjGnp;Qv70|7m0Akx;H*A>)1?&&1U4){d0cIGY'
    'cDNg}j-uq{xvCf^U0bo_>0Gih#bwGh%wGJREFJcFf!WOjys6LC_0aT4e?*O{<_qI-7wC-zn0XDC~KstY0S7EPqjEC}g(Xx3>(Xx4'
    'M(Xu(bXxSW5v}|5iv}|5qv}}&lL}N>Xfv7N|Zi~uEJXd<$hO0b#Hg<p`x;6SX<<8rfieaEdvrfC;asSJ%O($TTPDXnHBV1<G`mxJ'
    'JjNUQTTolX3i&dHrcvoDdH=FRTt`X#r$6gY12Eb<)m;MVBAXL14r0!*7si`>Pk{n@DO}6uH`l3aB{5NGwea=T>4)deF;FVXljR7!'
    'peg0NELnA;gnq43fAT!)oKo3SpALo{6{kuJ-r+2F;TDC2E)P^g(_-NJ}2n1M9-'
    'rhE>O}XadaOr5CytzO^PIvO=M%5{0Va;>sDGjU7Efg%AB;`X<h(Wxvz;?hdS?o1UAF}O*TL##<oa@hlT8cS9oP>fvgN`>l2hk<`5'
    'A~~|{P&+q6@ZNeFX(KmT9#92#Qa&IZZC^~18!%$91LG+XY2}suW-6fp=8A(t8L5-'
    'Y*F*ea?Uhi_Y}zX^`UMB`lu!_mI(X^=2*8L9~Ll-qA#jZNP*pFN`NAg64TTiX}U$ypVs~?7ul$$2cakv_9{iR!dWF-'
    'W=UqOb#M+wrK0=3r#gpgxO+U+Irpdr-#jHk-'
    '5gbkc;tLX)P!Tq_%MdPlHHyR2&e4UQz<S8NT%MD5}kNcbTxYCI;pWKgJ0oQeJqZhS6UrX#*v5OCFWYG%?Dz0Kla1&-'
    'ugWFATmF>z@lmOcQFAkSXAd_3Q-5W`Gxvl?($SEa#i<TPt_t9b@zIz{_a)n!8fVg!6P<Y?#0-)-iT@<N#*LzC_|3-'
    ';!DXNJm#YB1oZyko6{bF(?#?=?RzVx?5QECDCe8u7R4GZ4y^IbD{1NWX|nYYNKxY73{I37ID-'
    '<kufW2$RQ%0jAVlpic=@drfAa+BPzN@pk9y?=E_PW*sKY3-'
    'LAi$)yWB8Y+#JHLU?1je>yj|b^pi7GGPM`v;}FvbKeu{=KSi(GY6A2$g%7J2`ZE-tt>#0|(q6Ed5Isk+g`WU3t~ezd0N*%vggN_m'
    'sn5%FX@f5JDwZd6AQNoO>wmg~D8M#V3H=^2h3vwF(!*lX`+`xUy(9u9)o-'
    '7PPU2qYW+*4ii2gelR;6SAM_^*xE3_0~d8(cGdT^<ut=NpYRu~Tpqc3QmXlh}wL#xlCS`IvcEh|OP-'
    ')^V~=jOIYczF$f+<kmF+flN?Jv*2-QCh*Jqz~siN-'
    'R>u{fc7&VErb)b>Y7BP5#$~)ow{?u2iZSM(UzA3+O}l(v?${h8>zS!y>nMoD&Jn3BcG)3HqW`7ylJeh3rMm9QzOrR4R5z@nbQB3~'
    '9)tU)0y=IANl-IQKF3g@GC6UOWYWgBfLC?CR(dmX9=Dd!v3R9$2wke^}rI^&|1X@^Os^R?OEQ6L?1bcs#J;O#eiLX6R~PY@^ILl*'
    'CLJ=o)x$>B4?$qVteTBNI;OT^gCBuf4s4S3Cd0-'
    'x$oZ)~f32vyNhJ`gL@|ZdA?9i(H}ce6cHJoiA}J>)fHE3{zchHT@>K<u)7twBiiY=u3;A*p{N0Jd^}hb%MG)B1fbf(_6JD^gV{|y'
    '0|39&_&noD=oJ|MoJ}XU;3NJ@t5fAY=KcRwFOafR6|`~h0T{34BDHZNb|eYoA8G6>CzjX60|sBVx0zq=^7J~erYgLV?xre3}%Z=N'
    'SZPjF0Nr|27_rEvy*lPG(#*v0nOmNeOEv*4%8XNwT|u%RyL(4g@RBhz0M|*aLz;eFpw^rClcyLjBq-'
    'U+f2A}&`?Za`pPLxg7potQ*Jb*jWJPQ4;6;vX~$M3+}>C#SUXA}+YQuuUZ??3>sJY`fDX<5G6bhXvsy;rbZFKH+JFwt12PP!L$g*'
    '0%?ng%^HOO3v9j4^(3U(%tVEBEN}Fx2$kM`cBm&BHM~p=*pIZNo;cE<wFtBPgg3f|Te5=dfCp&PUo>7aXaJ_kL*;KOLqISvgbfDg'
    '{_Vt$O$F1tB-YWgLb-f-N#Max?>n9)L4#*@b5_%<6*A5ZSz2j9R^lB(W9#*`X--9?6Urp5KXt(sEm_U#6v&Xa1a9_q%^x+$iC^l<'
    'h5w{n_ETPMI2^KkRKhLOUAsN(KQ*M&kR=vMbhu<LW$#=N$jHBvJJGP^ag1NRoZ*+9GtNHFtj%g4b+^eICo1!%28U?;kZgGRkn_lP'
    '&ar>`17OIb~WznYn;P$NgEX<0-rA9qhDYL(vpLPQ9inH&R0eHoA-zxyBIzZd!lW6bpUt%D?^gW$KU*nWtu&)Ku<~{s%C_TDf_2_C'
    '!o($8oI+Q5E!u*gjnVDC}0z{^P+@17F)jn%Mg<q#Gh@<r1iV19AOJ5>Nlt^v*Kv(j+I3i|7jP&bZ;&GXgW)N=hc}B^un=7grt@*7'
    'LuM8~Qla7i^!R5ov`wL<;=EC!iD$C+*e^+Pr*E?pzbSU2hQGv^;Uq!F?S2wL7XBXqro(QCoWCeSk_d9?-'
    '(hi||ZpdiYZ=wacSkF+YCFX|5RPB{H9}W6&75^Qt13pvbVULXdq@oWy(@}XMQ-'
    '@hDJi#1M3_vp2XLSMOd7uFAtYz>Hk7I9elnSOlimH;Ete{?g9X;+3Ej!c;UD=`^b)u}BrqD&}x-'
    'UQU6$)+<6V==;QTr#Z$tz4azG_ROuZZ(g?y}DJS=362#u1{b3L+DDA42cM@9r<P&g%OL{L`ue|8#$We_ErMr%sH{+$2PF<{shZN>'
    '7Q=nLC7ADp4gyXYLOoI&*t)o5WtxuRa^{5aLsKT-'
    '8q9wszqFm2!9u@TjVxd^nK2s+D^L@TiRO?}Y7Kcv#t<$GGqm9PNd%5}t$idT|-'
    '9EH1;l)RZ&(dwkWYaYr28%;9NlcEDb$uA}gt6@bz>qJ(UKiM$&|K?p3=u*77zc^`~-bw#2oR9vfFO*o-'
    'yw*vd2Z(G2l(2bO4_#`FtswDEFlglGws<NqZY+o&KJc^U)5t!uJ>ZHCeJDt?4ve8L>fA%@4S7)1(dQEmYsUOHDC-qvnIu@YV;H&3'
    'Yy60sOf1IN|#66tj9c`j=(7clgN2h)BaueRs;<EHb7uqysX{5avVLE02M6;)o@l()lI~WPguLLu(EMng$`L{tFty?aa)_0ldy27b'
    ';o@K;o^4O|uWF773IPoQWjHC0!+37nRog$uNpWx`g+$*R+snzJ(yhgINYv~_O+ODI2SdaF4`iJwL8|WWSdv2tEIP1BI{yFT7E~xo'
    'r^rCHzr_KXZC0uE8G#Jd$;8i&q9F(KM!8sbdI!A*;ax^$JM}xzPl#|~)l>r+o;f*r?O?`Uz<ydvD4*wwM$QKdv>Pf{S#mTDDI#4k'
    'SXl?;iwdRxt;qP`-J0|%o7j`h+DEt-Bw37*<yu)j4_=K0DZd(@~;;VED=W3odZg=xAZ02J#;9=Oz=Vo^g!)87>dw3W&^U2xM!?2l'
    ';&R!md&3tzD_AqSb!$Vc-W?p)CZ}Q=JdafFxBFbZHWZEx{S%@>(vE0F(6K5K8n`e(lg0!i-T~E#hA~0N$Z3c&wbnv-'
    '2$l+|`^O)Pq;(Y<5XSvP42uOl-'
    'Tra>tU)q^p0z5(B1z*O1TJ}%O>F*I|P4$n7Isi|Zuwm6i*)21Y$@P_)+^A%74NOH$%EUr&D$I^Un$oeE?k;XXP-'
    'OmDYX~?TDZ@oHl7(7=p>MdF7wV?y-jY%n{CzL#{X_TOZ<PW@3E}TF1*iPz4^EJaR90``C8GSoK(_R|N-%mxhg-Cs&nldOmhgG)_6'
    '8rGHIYtwN=b3+De4-@EQTWVt-'
    'z5Fi7yNaW(twg;ZCA}L)_;paX@rJ+63s?Ml6~rsQ<W&7S!(}_)jTFgQih)(Qy}(xY1IRe=KYA$L1KlK5m;S<KKmuYPkJFS2|o$mh'
    'pS-?og8Pze>NOT>OtJ#X_m+pQIMbGw}|cZE3jIR{QM~Ag_~zNrzt@3Xn3gpa6NJBuwk4)_k|y^<d)aG^blbra+a`ol#KSR^ae!-'
    'KsmAcJd1CsRE8`S#EGMWv&criez9WE8ho(yz+(MD=Xg!-n{aaV9hJv3C_IorC>dKuWDG%^{g_><v1tbQwEy4-&Z2x(^2hO-'
    '`e|K*ip0QpSEDYeOQWV-E;aJrZXInX1MT5ohEJW!ai!ly@d<=s=U}%F6^%(m2F%&P-'
    'PM!7_X;^!rPNNP@?eml+H3KZat^!V!OcmEnMsrarFupE^$rS!(x(vk>#L-D*${Kdb#j0;awpU-oBz;JP^S^4Wt=Gn^~#wdan!Hx5'
    'lNknMQ<T*2tLp070VvD>RC^DrYpN86{w5WiMz>g$p)!JR&})AwAFu2UhiF61w{||D%**MlLWm&jrR7xxm;m7Z_XR0%Pl3U~D5hYg'
    'r6Ll9>B2X}Gr@-Lamea9QdbMIq6onO`NySwpP{8n)9}681#ixSpjkDT$-'
    't{9L^~mKiShlgDU5XBx&xqs{8LCc1(1zTClu$E;*e<M?wq0lN<bCC<QBf$PR8*!|!taSpZ`WG_y_)_}Ief6fD-'
    '!*Lq679231qF!#$UOiahVLQ3-'
    '71wT+c2hdmH{nj!^mAcb(`Dh|O85_D(Bqo44aJfQcLb|`(4Q}$>;${g>ST`NMa`|lwLs`>qb#3gHbX`}XES8v^EN|9zF;$C<cl^#'
    'MlP@!GV&#xAtPT_B;Y%I;pBI`S0}$XX6^L_QqLl5-'
    'XSf>LO9TO$z}E0?gAxp+xlEV<V0U}q0_W1XDVoM<Wwb&jeKNsWhix17oIlAWj1WiOxe-'
    'w+}Q~?Mz?uRs`{g+)u3t!@9SA@xD!5EslRb2e5L|V^PZx)`hD?7R+rxnsGYmWmIIq}R~7Qw6*iaQYvuUwb5*nxpr4jkv?-'
    'Q!ZMtdH36S9Sxkm0<h$l?~qL8bDn2T})D~e*x5vCB<d0C=X=O!z>DI_@p6N};?b5-sGx@-'
    'nNtOlHHOJ3L=U3gqYG$^6p&#OrKF&+d+z7YCcY$P$j27OGyZf%aKg<!X~LSIV2ZZCtiXt>aI>G8~?s>j1KL675CD)H>TxQdSSZEe'
    '37w257$wvB9kD6L86srISVP3(q_bwSYjvfRS}#h}Y=*n*en0if!(<|R6ysyCGtqFUxauj25>deB07uycY3<&jg<EzsK)1mKovUka'
    'dYYqYcl0eF-T&ly$-iizlK+m@Gq0anxDMb8ddyk2LUuw6Q>9T%SILRn1|S#Ufe<2wBA0l8B2-'
    '09|{e<!$IJqX1z=^taHsuMF=F*toGiM&Xn2kM(62H!|`90&-Jdwqq)bp8NS%s9ByeVhk-l#`Kd(A{UGA86S-'
    'X`A;lgMK^bIae5r(74vR%AosxXmKVlsHj?q3lm&;wgouur&HMtv4UlJj#EPgiMf+_m@9<<i$ZO&(_68x5af-'
    'c@P=1!eM@8_2n{)sHiU;9X$e9^jtmZAB1evbP>~~@L!kH;f5%*%MF4Rk>bg5aRYnzcUv7aI5J}x#ED!^tsk^HMHsZiC&w6zwl%cO'
    'y;a9fRnF86r%(LXer`PbPPAtv^W<aI?VVEPg4qf3_+OQR$r0Z<>q?a7^`e+6EYW>V#wL&i(@W3*NbHkn<SO)QG*vkXU@VHYe)P+h'
    'W{Zs{x{@uk_vnJe<zS<jHc+xLvXqS$a+NEO!1Hwn}uk3LdjCG>8if)Dl7?8#ksa&p+Xa+FNa?wV8T-'
    'p3P0i~SymYw8t%op4KRlSKwRqf^h?|6v6=*KI_6mYHliAqiCJK@t+;ky3WN?q5ZiRv>ZY+Nbod$x;ZEt4o-'
    '?!s4Hlay9iQH<C9k|xIX)rc56El`o*xE=zXiAFjNO2k2xP>tT`!c%SiO_a6M^%&qDsD;PuD0wofLFGa#%+#HMxf5#YE<jWyK58fI'
    'Th$w-Ef$6M=V(*wVTY0QiE3D|XK7flw>GGjJ55+4!C+?>E^;p+JKbLnP)5+r)-'
    'Keth!$E}iB@pwQ9T(4%*~O?7piEK28k#ao^Guc?eYHUrH&pfbU+uH`~vYY`%9Y{p1_PNHUMWQ8p|wf$RNHVifY}c(c)_sgkE+R)r'
    'rlv^|q?{zFobYYQE2`XR7A=_VxCv`MyKFgKEC-SnsHs@3ZPz>XP=j0|bC+7+<kKD7~^+pmR;wK5eH<s_^SpnCh*tou)Z=bagzeI0'
    '{Pc9#a_vn0#sN3wK<;$fqDG_%rQ|tjb4?QUpyWDT#Wob`93#6G)ju&NA2PVe9}~b82HSH-'
    '!)~TUo;tS^7(&__%DUp^UeTme9hkPob&Qjj)(k&rEA{uccVfCK~rmw?cH&KBU8QDyAM$)rcFQjh4$e4YZU^=2-'
    'N~s@_2E>HVsD;}p=NSIF+ASI8bl=*G(}jsxd!Uvar4#DM-'
    'Z+v4L>3X6q$A?3QIkw*rEkC~U!{cOAR(xOcowecEj%lv$)iY~7J^j6`QU1BYipG_#}fQo}BtT-iZ0EhNey>-'
    'GqRlOn4^5O_rX~9&n@1`uUi!5MRV*Y6{vExGRVhc<bXGND-'
    'V6wPsyU^vL8PKg$8o+>r^$_)^Xbr3YLnGF}0Q8Ej?@$6s=a9iQRx@Sl=^7G!xeCvxWqOkf&vzxCFPMC{PDQ^<eou3K5-'
    'RT3QQnZvw9$TCP(>;~&$o5PHud4Bs;h>fxeiV+B35ZB%!pN53PWO*mcnpdrKK=3R%t1W*Oi(%gqEVb=hs^RZ~5k?`&8`9I~cQ`%;'
    '*DNm<y}$i?lFDx$s;oh%a#2&!iE<=lRGe<b63|i(~{gP*nTMn0y_3vqSA~UIA^ITPf-'
    'xNAj}(=~S<#B`&OW?71`rz11ISmw02?QOoJ1n(Z%!S%y<obD-qp5Oqpqf#1%p*9$G6HC!Ej%>r7(it<NVz>_)dUF5X4Gl22aKzcR'
    '0WE64sO(E+Zs_I*F(b;jyy4ntL+&xcgyE(2jgU2?K!gnZ|X_OGe<sz>UlZ!cy`~0$I=KxEkwn^u?u*Ru})z%)Py~mj}xFOu(C7am'
    '9ho2~l^5;J6R>}0^7d{MB^2>hZ!yc8i<A3eLUX@&TCr4e1+qe#UjRj7e1MJ}z;PCIYPi24CFoL4f+_aJunhx|!rW=+n$1s_GD95D'
    '`Vsu&&o()K|dU{dPxX!5V%_|HDee3bw_@xiKSF(queAu&+75I%0dsotfIyI75@8GZYS_{~_aI-'
    'D2?mRYqodwvPFK4O^_DCl+Pj*(=axU^B9B(jN>6aHhkzr8%Lbgn8W$F430)N`NGO3Q)FRdD+K)c~Rz;&x}+2$T|s)vP{?a}dnj5k'
    'EDM>;jEdE;M=a*nn&@K0C$=v(WVsu+EHy@U8Ugtu0515mX-'
    'q6$s#j0z?<i?*$WZqRYAX~#N|FT9BYtP#u7F{Ss6RJb387#2WLZnkX<nkZd<(I7+6)uH6zm1%XpYpLvLp3c#V;zM^pLy;gxQOHtg'
    'VJ5sf_C<sh-$cv3V3DTqOGYZ?e&Lt3Y$9iP3=AL2tm(fSH3aL&jnl7c-'
    'M7@+sXq5@_4caBeMh~c3Uc3B&r;#oH&>De?%=}PL<0{z;P;qtt0)XxsAGv#!;MkE_4qQ9QL4F5TC3RJYOd9axK>^BiWtbPT!{nG4'
    'RD?0dJj6G#g&TO>x_1?Qk^yt-)LT_Ej)0Ck67h{zTJnL_+#H@Vx{f1TmEOGJUJ=}|7Mh5xT^t6dVQl**@c=}fM9yM33u|-'
    'lLoDSUcl!JTC-ffJ#R1}=8o7k7FuIRmn*hq75+D^*kM(h91fra_JmEAevLzUxrKiHb(#PAuRh%9bi<=+^y(#pV%w{iBZ^3Vy4A!g'
    '(yMC<%1E!SGbn`k=}r?XAuq~TK-%!6<iw~fa|<uUvj!RS{-'
    'vqyLa2g``s_WfII8gisTg%%mKVuT47}2cn{DM5)1^Dhgl*H(QBkvhSDR~Wx>4dl*wO>4@~BlB@!Nd3K~(Nt_?JrV^O-'
    'KZrIO@yM;HFJl6N}Gg|}DoPLDM(*Of3gX$oA~QwB&A{^OoD==<!eq{TU!xdBD6cT2|;6tp=R^%NB;Y5<W%!?)x&#mj$CDJEkDp=k'
    'oM;dxEQ*TRzVwRshOmlp9V6AtLI8lY_2=YtkC(0il{vQ8Bciwt85dEBu23(i4vq#edaicocFaW7m}gssbqJK~BWgk4$O7grVG?CK'
    '(%&9BhF5;p#uD~vZp1mP_eI@Th>^VSOUp}R|U7`ogB*ektC+}(2I>JY-FnE@^L<*clvpG~Dz)2Keb%7C>6Mq8^uTcDV2wC6h@n7u'
    '(oT|NyVUukoF1}bcllh+Wy3cK@u>OfW{?7>f?V5YJaO5FY(6V5eyeFqcH6J9VtN?{3)TA(d2S<3aUUM-EzHLkugD)#A;(Rtb<Eg9'
    'tkd)g&kX<DOg!^{(^eEtY9;8tON(#OdABS5z*?8%@1DCl{Gz4+-'
    'm(3t5?;wuOmOq5%3I|Gm{3Jd8oLnvQX4$}Bq(z|+%EaHu>en8f6m8&mPmBO{IzM6#yW0a(8Z|A1*nsVtPKVs773#@Rkosi3Y0Ano'
    '_YewJv^0q}?tTU(SaFv*0cG2yT-85JK-YJxplj?^IFsv=u9C3G*JL^-'
    'Bu90_o1Jre>vXL&=NFoNK$#rS{ko34Ns~?sY*X8vi(&4(IepDJ<SJsb7f9tCHacOT|T|dESDOazR#w^`RbQSJon))%nO`+&eL(#V'
    '7QI5DWXu_~>myUBvS2utO2dmunC;&zMrPeXhORb6McubloqMG;M8z7(;cH>=l4XE^#f^rE52^zZdI8TQl0hKNs%R#KC)IgKvH=1y'
    '^u<!vK4Cfeu+=g=%p*-p|UB@;lV2^YuQ?k1Y|EARrqob3HRr`$zd#A(qfT<sEAs|o=b!Vqhq-MEDwZcxumR9ofaz<Q=v_)Qczqn`'
    '4i(2)~3%ReNk|cf<DiZgyB_xa1@+_A@$W&rw6f#wuQbN(OKTnIuFH<a1JFy`XRRF5=pT=XX<gHE>{?Urmq{8hO?Ql~ad#pC#sEj>'
    'M+uc;q9B-17Tcd+Y$AtdxkZG&&yt<-'
    'zD*b^z;6zy<G1kOKnQ*?)<yCPLYP<sUxJ9)np~WjOkK0s{6H2_&WxPZ5DLc^N6>#BN2awfiZA5LqoY=AKh|rT$<4BWH`LcKoNKa?'
    '6!HX`@4_jB^kLmUO98MUmLdRh%b@fiCV2m_JQI7{zVL;#PAr-'
    '}NS#Zj3_g3Q3Cm<j$cgrWCTp+>uvrr(Dj^eWHwhJUW6{6BBID#rx7rw}2op-_hiQ|6ER4JR<wX~#-'
    '^sjt9LowTz1R37dFHtlP#Kvz^V{qKm6LF~yHk49yb!^661TA*S6EB6vteo@LP+{r9=lDyERXHiTxynh=Emcm6Zmn`sbX%2^qT8#S'
    '6x~tfr0C8nCq>7^P;^VSbRRHLb}mz@hP}4@khF|uCro4zvcFzR#xl0=$D8A?t*Y>cw4=Z;Zzl$&zD#3F9b8c=Bggj8itc`S?-'
    'VR9@~TXAxC#5Z@P$femh+4n#vL7Vftxc(fg&$!B4FfBS<mfGF+p^G?{JD>pd9GVTr2KjRDP8zA}V-'
    'aVOnm{^zTwHVi`S(#6WYs$O?3zfz)|S%KYdXVY-P4w6I<N1glN`Xk5%Srhcp&9RjeNlfPh|wqa;Z`i-'
    'KD#u^7<(v@{Igf>R1=#rEJHzmQlIqBDbuZm9FUM+xTctODY(9vZl2$&x^y6Xh_@nc6xryxIGN0pij7gS1(JHUl6RZ0~*(1kBoirb'
    '<@^)?P&Z^V@=UH^b=C3XF?y6m6QM!TY-'
    '%w|d3hG^Q!VeHbE8l;Gpc}|j{21D3eOX2qRLiS8q;By{77BzBLwud%lgK{$+R^mS}hUh1bGH^i({nXJ_BA}R`Im*UG@~K~|gigIJ'
    'h+vGi+zv!Ap3BVy5sc?@+k-'
    '#HbGaQr3gfxlj^NVqTy7S)bS0_lK2tvwS8%PVpXhd+d(f3B!h<fC>GMn$DlMB(^iAg}MyV$<F<Y>nr4??Ec@_QFVu`ZWX!O+<t&m'
    '%?bHlCKx#70#+;Dq#Znz^mH{6+=8=|HJi~%|Q8bCvqDvR^^>2w2wF)4jINa-3=jd>_3)3~Ri7}3_0T~m0PP*hv)mBe*d6^-'
    'lp$b7$wcJ%yoPZjOxy|PC4apCioYI4l8l4HOK*2Psym>N24mhn<Yp#l<Q7IMWZ+)(9<unIRdr~y5ZE4EHlH(EIaGgu*L)~{6Xxh}'
    ';C4JTCHw=AQFEnX!3*=LrP(W^}Ta4Mt?72)l!@^>pBsZQiC|85iBXf;uOFHAwTNPV9{MR=ZqtYAG>38DyIaFu|F=sZ=GfQaZkt;d'
    'Tr&ud)n07h(-&r+kD6g1A8kh_-'
    '(4@1#LsdPmdDoB8h`eG#47cp7wQRc}4#{g{IO!RH4l(HXY!rQAZZRe3@w31DBf40=AAIQErF@G5m8zCy##72k;Hn9<+f=z6Ms9+N'
    'tAu8C!Mu-YFu@RzzO%PfXx_u=$EDA~MmwGa2aVr!kx{A+LArch`o^%oUuG&~V5v;`e)U>DPRt*J=t1HR?<I3D`>PK1t5Rb9;XoS`'
    '_7v{d~LR7EHCPejG_7jSBm$C0Z%-ikm3J9%}`RU#Yu+h`_X<3Eu#X;=z6_gn{{rX}BIY!R8zEnYjkrS;iR}f+3OzV_XOgC&-'
    '=K<)eeD!iqjg^LfaTZ-'
    '}qan<I(|<d&LO#(P*3&8RaVOQ6h9_GTCc4dgs&0E<WV7nvRIe_anLbxRk&@HSFI14B<XrQD3fhx9sI7F5QGtSYrplO~PE3_CKb@3'
    '%HQ7KCbuvGlnmRYxh!0h>EA{$%=$(nEV!wLUH%QZCaoVqm)KJqoaHx$hy0i)(PrGZ-g!xq$8mp(8j!gl@PJRgya-kENep-rB+IQI'
    'JSN8(gI{k`5Zg&@`9R7+|oy#wp@KvwMmQ({@>{Z#4%AQNSDqB)X{57x2mS_}IVYJE}l`33Q<pjS9*H*cMRfX%SoYhw0`s^;hR~2r'
    'o)Ys>~s?@&EYXei8u30O^Yn!6j8!M_V+H${ihwi8H`c&~>=vm1V&JQmr*Z&$71)i#}&0_!cby@wkzCP>z*Egu_|5SaWisDYyHx+W'
    '!v<r`^!a-'
    'K|)(KTO%oeIQbG&%VbkG1UFp;T$lwcWhE%vG^)jTvnM&;a&=gGy8Z8$qQ%_Q47{WNMdXdV~Tp(xa|xNrrGT^BWH$^C^i$JD-'
    '{8bRgUeO}~=4`<c<ZrS@nt9^O}{6fJ!i{Y1I?iB5gtHL2xt)%`<Cy7^?@ZWK0q9>5bi>2uV9%9;VhFIEZJH?|__(ytMuQuVW)sjY'
    'x5Cl{wH9iJaxV;~>*SPkyN)@iC2B3x6XyVmn%<`CC-pDpg&OlLuND7LyL2^(ei;{$*GTC3Q%=Pv9nP(EBMqNK6zJc5ICZPb-kTb#'
    'nm>p~ql%#nc=OAUC202Pu&w82`{NG{~2hdH3NCuujn1s^cp1nz!q2|Y4?9n!->!U>Rujj*#X-'
    'f=Q?h#0W(JS!HCh<Ah*p);3#l_oA(|#?@7K@f*6Q7mr8g;PME5$h`csTOhfXak$Y>P?H$T`IHxmA~5wTli|(Hup;p7AoIJgN%Em!'
    'eT|zxU`1x<-c3^n$JxiF0~E*U4;=UeNV2nx+?YgG~PE1>MMi6uh9DcwpV3KT#1h^g9)0Lce-zEdy_2m{PNx=|xE5pjU5ztBQ3U>k'
    'SddMb>d6G-'
    'Ft!uihAU7)$iko8Ss#g}!=I++ZwBSZ{_4Y@14`u8J|E@2sF0`p#QuF?$0;VooWj3cYzROnQ0}2`hlWCKV;nSJ5xk=0+Nfs)q%;*J'
    '@;X1HCHdOjO~xRtI4@Ug~thwDj?N3)g8M7Qt?jkypr&v#rr6;<`{5qLHM4x)(*Mm<B_Rc0PSGUVl2fPx8+j(%F5Af8L1B?$iA9#&'
    'mX{;h#65v->RnyeXaC=lJK%=<Gi4HETP%3a_%7nZ2V5ueO?*9bbh*vjgCNR1FF*Rn6<Am3qD9E`I~5x@?*OX$P06s?_zD-'
    'Y%ng1zF{U$5~_1=ww*D$0GQ`m)9E=a<5c3pEr7FVV<Zd>Vk@XkEWr=R^ec)U4ReeE&x@UXXHxr{U&^*(G&Qn^aP?~;+YD0xXgrqS'
    'D?TZCj6(umaaD8V^zZ|(t81I9}c~(=6cOR2{bK9725*at7t5))FPos2zrIMJ{1-'
    'I6cM;s1tGGEP=6LEpiQOv1hlDCo`5!$a_4DNDRrJUm8cd>(WX)sJ#DH#>5ztA(IRbHpe_G_+?J;b|AV>0r<(r5sixO;I@0nWtG+x'
    'u-Ku3KV=uKE1WDU!kxn_6SJ+J2ksH%&AgQ@!8<6;@sXDFyh+*2bY^~rSW5}b6qda5`Sv25H9txAE$rZbT{#J?LUk_iYB!d4>xUiB'
    '4{*7=^B@z6a;j5KI@Nb2SD~aIW4wqCC!M_u}#`z}gq_tFcH?Nu5Wjj-eCMNt#%8}jxxI-I`JQ#&ghWK8i-'
    '@iQL%nCdX)@7m|;@Kl+NrN_bL;<E|wvVh+?I8qtl0vDBs@d-h=%2KJcL7XI`kb!-'
    'd{X+Hbfm^(#~BZlIo=n3zz=l3R)rt(1BIab!;koZ4%_PRV}77wWli`AKTx*$K=^4nfxwuODJ1}YG<VeqB1-'
    'p6D?&bd)X3mvfC5Wr`{h7~Ni%y_fWyQ-'
    'w9ln0M%kge!Lv~5p#z3WGUY|?Zm1+vV&oo%N;2h+?rFm9MtPEZ87k0}9Jsd$cb0=KdMEQBCLb16+EP)6qNFFvx*1V-'
    '$CkGhecHKI$xX&%vtT~L3b5lCPY1ZUdMlA4;A;<6?Nb?TIO6`A0dNHEzG1G?hpLh*%`TG5nHo22@DW!kp8L$I&4=Eu5R9H#2u5#T'
    '2uAOqg3%qyT(U9Og@tX<MTPmzC2W3Eez!%GI&{(H=~lcd1@d=zx;WZrj23VOFj~EQ`S+^mbVL|XCDyAN!Nd546NRX<Q-'
    'E=wk$qG^w&qvo7Y*|MLQZ?zLQZ?fLLPe-%VV>8;B6I^x+*DoM@8AFN=n{Yg*R1-'
    'Mpc8Li`81GLpfVE1zO><bxQajhyK_;rEy&qE|05JUlpF~(i!@_55I@+R{Gt4^x=<iX-iLkqN*GoIq3<SgK7=_0l1fOXA@WSj)wb{'
    'SWGI}7kJI9W;@kkd^eOYRfq9C(EL^Hx%WfoS5+BLhb$)B$QP~O-UfBQH^U-'
    '&%Nw{K?|o4HD#x;n^IVHbNjYP`vI?JOk8w}w#**?S4+59h=%LV3Yo{c9-Q`vZ);${|e!~@3LEm(ROX9a&ZkL2_yWB4c-'
    '*MV7nF0d#or=!*iwP&{4F7K?oUH0O|1{w=S7H!`;;t@NB9lWOj4AH8TrwH1m<siky4xv4uV{%?^iFg7uJ_@aX<v<m$!?P}j9pel-'
    'IJva|IuMw0CMxC(Z9>YoL<GNqsLe{xb_&e7&I#3t#9!LAHJ17%a!3-S3=i%qIVTHk4-'
    'e;<c6w^wp`pvcG0ksyQ@CBpmAHdik^+!YF#f{S#png{pgQ2E4_+!P!gA&@mCX0(s}vcO*lnS7ymHf)R?I?PvP%;_#J%k(jWW>AN~'
    'N(F7_XVs|1e=!0Gy}qkJMgZG_P~-J<Df6ErEQR7y{qqDe`cIz4TMCS^Fmt2Mw*PBkFk3uQ)DeT-'
    'u6yEs(~onbJCyty2(zX(SK@r>=o%N$mf@;tg4SHs<PrlG*z$68kF%2abFFz%9Y9B_t9T5Y=C>E<tyow2_aYBJp>4=b92u9HU?UC@'
    'CM&0b2D+QMLxajWVMt<f}SYXiX69qdU~!6k&jiYHYtWMM$e$rW5em>+UV1$P$4E}mAwGUVP85!4G24gp&CgN(14-'
    'KZ}}faq^dG{h0-de%9>l?^*?I!aaZP8_SS{H;@Yc@;j%iYatXx>VfX;wcf7uR__xeyY??H}gXZ&!Fq&3C3JR6~|GPIT^6wS{3Wt-'
    'G=LwI=`0<H>v>2MAX^OF=bq{x6KXakhiGu&03S3vLrq%@MW`=B2RB#vw2l&6pm{{EmMe<K1dnOGT-Km98-nM(z3i91i40-'
    'pJ&hSh9MMmw_X5B7E~FwsCQ80$7<h;e$_~oTOSJyWM&(FT+4leLd%LLD70+Z1cheqrB6_3CSQ8k>&nUgnupmVpXE3_62S|AZ0^en'
    '499xf>)cM93RhK>=?iXWPmj$nRF%CvcK>>&jehOkUbj^Yd3i7UbTv>3<H%QZkXF9B2;|cDNO|RojKoRl6?68d@Z+#+tXmu_sisGL'
    'vq*XX#h0}#>RYpvMSWXVvZ!y*LKgKMS;wNjQ)MiU^ArcTzdD7U_-'
    'q}v4|wcYF*w}AV@&DG(6tK~uW?D<le~YLo?TRsuAqe5wVMgxi3th_Jp+Zg^^0jrgykzi7MK~815|=+7ML4T!3J8P4KmDe!whrWD8'
    'n2#&M?PKGR$$)40GH}lFJi3oltEte-'
    's9GX&gO;>D^As9G(NLS`(62)80!Z@P$5{*eX_5L{Dp|9CqP{*FiF%SNgyV1w@iPk{4`DnJCFSOdEMaphGjkLIe?!g(O``aJkFufP'
    '&3<{=Jj|J|LPf189zHXLw?d-pCFnP3Nb=Wfj$+l8{|dQ4J~y8D*2ZXnJ>bMFpvJVy-'
    'l}EV#dbLCmR@bQmBIwM8c8?d=g3*Ed~?J7aPBJV%SWqG<(pk~|O8TC~WQ<GyN+u*xBKe}OOt##8w!<MDJh_^1;$J0?D4vt#1JHaj'
    'LhVzXo7qc%GxK4!CH;^Q_uCO%=4V<H31R%fZ0GvEoBAQDr!-F7z8-q_6}Ju&5$qX0Xc<mV`4p3-'
    'GIfCcH7j?08{Vv=}M0RR{;2~l$jQ{kiz-K@zeaG#X^&YLSBw(>$zjlL=?4lM%9S8yQ^Ch23E_oWokX7JCKQAFFEf4-'
    'a|+7|rt6%^67<e#skh_)5~d=*8st@-'
    'DxDWYxTv^hFlNh=K9(G#oK0Ba7BaYvy6;M;BV!Dxk5tVq!kUX1Eak{Y<;jAY`|Jf{wyohV>4ST?6-'
    'xWamEb5~feZRrZ@wQZbQuN6RQH1u>zx?P@<&dgKN?emm$hdd?SF;7Wnr73B<vS~{R+)^=4>W(S`_!~<z>cXOEMdmXS(U+aP#ah6O'
    'sAo`=uHko&T`=BIlaKxohdA+v*xF^6zc>iv_1u8IKTnS~r0>t8nT_cC^JHdY`u;pp+JwG8gA6yN@6W@f&FK5ztM}S#O?aczm~*5F'
    'Zz`iEWgsV~ZGDLkr?#T^ya-ICrI@sB7W^X0y%ebYBC79Ri6k#tw)3E^F<vr3#&fdO52j>qRW1%?!-'
    'Gy+>aOte+rkx2iCZ~!N<_}a!}l~WGe$j=HUuC>Op`VO9!6Y~HU=C<Y?C$tWJr9IHbwJ8jFUD4GN$32L<!kxX{q{5L?yQ1WBt>YFH'
    'c&S@ZF`}ItRJJTjyX!;6G#G^LV%VDjcS8{7?9-9sfz6wc|hSMK3-'
    'kdapAfWTC3pn{bqq2>{}aOj;02&S>N2X}Me13TnO=HFORZ^ESM^!345r&4WpyxuNVU+uD=BK(}i6TPpy}l3H5m(&b<2=wCW^gOzk'
    'PL{K+D{q&FSHwD+c1F~J82-QZRmDK%_ckrAIhdME#TI#b#|HpmS=>L?@TKu2!S%3etUi9~)n8}D-'
    'itjRr=`!6y9GBS^;<(IP5XWV%h1lJWOWn!$n<x=W2^EzN?`+?0DQIgAIYLKN<5Gmh4@w_|{_qiX+-'
    'dmgOrl%Ul2MK9GhRTj&p5RN`;1ddu+KQP1pACrOR&#4wFLW&Q%kVV4%3toXga62`D;vggVU7ga1&~$Y4;H(bjt8_X9ZAnlE*o!YA'
    'aLWRZcS<ULIrCI;;9%nh=2Sg?*DryNhi0Hyw=EzP6=C$~`oN_UBPbh=>AKOp6+T`rD5)E#@hzB~+c{IjXswlF}*XbJ88Y)S&&qqb'
    'tmNPpV{Kh0*58m5i(~+dQq(;nX*(@U*rzzEg$gO~ZqFcj*HIN?2E2T2)J@kph@P>tRrJ6J9Ezq<kifI3>6*hc9!#%GONUW*YEe$R'
    'J958j8ntt`cyzmF79Fr|o36fEc!e0=ns{nx(z;rzv?>$)XhzZotRzf2#15b{f7}g=ch5^sOp9t9_quSK&F6LLaZxUOHExs`&W!j4'
    'B?aE)?e!`gFB6{XPmABFSM!B40_8z59f=NC#+@kQV6#-7h3sIzp?3zDQ?ijnY7`ib7tJ?5$N&<1<v1s!-'
    'F{tMDXu>;#ia$#e=ERLLIG`Z{wGtlfuYMT{>F#!%ev1qox0(Vh!_W~Gct42|U0Unr8%O#tE|t1Or)3=rO+9qtc8j!$l~Gaw%-'
    'IK>Hv4(?c`{GypMVnRgNUZJvdtL>=6i5ZSErTdiS+vj9|J?Zh^cz-'
    '?XDP>5&N?xels&J`~4CuuPkC?ksleWC)dUNBcKbPg>Q5XpSzg$ZrIfTGNsA^eNLsMX~v6uLR>`{d)(gwV}ip-'
    ')?812bY80~2lM!PaP5`14JNS_mB$NVk<UD<%!-;_db$jxs`AvfaIH>HppbK{#*$W6HIO)2E2-'
    '1Mdtax=8N^U4fCWmo~_iK{$U3V@Mf@;PD}b<{|KsC)rJ%Bt&Hu=^|g$}_7tO>Ktq2W`p+Ryaxc&@?qUG!;aE9H;Qd^%z!Lir3wPU'
    '8>>9d10lhygsE;eOE*Wh`Uv4?RhKNbgZX2y*C;2k2$q~{xN+Q)IUbn0{h1xUU2`I!wc{qlkAB6&(^ml4$K$go71wl`YACib!)lc>'
    'rzTS?iH3y3*RfQX>liy6w0}yPI}cng{jX7p1A5Q`tq@)ZE@n}RFrnaGerNfsUq;O!_4KDN_})4l0NPQILub=sAP1X;eT{v-'
    'Xe~T6zFdi$3}{Hw~1pTo#WfZv5_v-JH)Y(?$kTQeDrmo(}&kty+Te0lCwRC8AtV^qu={{_-'
    '*!VU*$2UkyhaS9<x)&{G3ano5A3n%b=UV;GN5%o5A3ntD%W;n)>c~bW`=-'
    'dlwL>8s**t@TNw#_XE6HY=kqA0mO~4rD>w2<5|zyoG-'
    '<xQYM@Wt+F6Rn3Re>B|>d)MSy;3gLOe%K}{M_@c0#w24Q^Cm5>Hue9~2r201n3w=)cOg8~M-'
    'VF82PsDQz4T)<#ADPXXh7BJY&SU)&JReKAinu5_W#}^>PXkbKFx)M72vu`18`qAdFcq<c;h)HemO|26+$0C?d172C{F-'
    'aHIq9+^%hj;Ru|9A!V)aP{HWA#UzLRPME;ZmoNm1|wN%qe8$Iu|Z?3R$_{g)5vwR&H?NN~e&O8(p}{DP-'
    'j)7p``id@VGr)WlzLKnEq;n8d3qzT#ZWdbL}Um2m{18|7(SmOSglYOl<4#yalmkCeP+8f^L{yfhYWjmIcdh7=EYz!Ml(3)xD8jvL'
    '!|SD>0*z@UjaUMm(VE3{+@172$uDk-#5p{iKwlQ^Q8Efk+Pl-ZfekW2`CH!#-'
    'C3E#>}@C_(Roz5}lL`3tjvYqHt^(x5b%t}y|gDA2aY_u6B0>&Vi>cWw&U3O;7aHkQx>A_v5L_#pZi|NZqDLAS(bgXkqApX24!!Z='
    'c75Z^Lgz6)HoV#5(qD2wt#UG(NLSEcs4rBA%SsMaZbtQS<4{W$gg$RFW!*^AP@JBX$Po?{QOn;lI68^-'
    '7@2igAPi?rTk_!B1v__SP80_w~4h?&Htv<uvKGmGfN$9~YBaVJy^Tzm<m5m{?jg0%H(geC)jQoX3_RFkbe<>}6ZWp6};qmYz;lX#'
    '17tT?YhX-6SjK0tV%=SjF3GI8qJSf(F?<!oKw!)r1QWSIEHHucZ;wb1vWK-ot9Ub+;$?9j>z9_Gkn<F!jUDGRHSaqpxJcu4#G&E|'
    '4veiEPuV>7A2Q5nlB>mjxbM|YS&(ro%+>N2oRKr2gjKR>9u|v>|;n48H_-W>z-&v_~K?kxSyX(|?O;pL|M0age$-'
    'cLNPw?55yH|#3otP62%J?x~0dW>mD0SxAMJX~Al+!!7j6$$)jE*PV`7I}#UL0ty08&#(?;0G!ana5bMXsr+p|9!#gHa`@m66t_`1'
    '--<*N5~I90m5|6{Nj&6m2iBmkA~2*&Ku3(9I%DYm6e;SD_#Jl)7+uqc7`X51eSoQ|DbxcoRQSXnr3Gj$)nm@JxNAD=S>nrT*4C9j'
    'Ll7TIeKoV7|5IL5FcDk8Aq03o()xHL9skG;QQQijGM990NWQ;_^bi0UAeAJ-rc%8<N_0perG%{ZU5GXa`NVy+yq~;*gXFa6g*I0{'
    'CBpe(|&F)UlpXnEzVzi=WRk|D8;DgO`#3VQAlGT0H@B`81+aNmhNOULkYT-#E6`<6w|w%Vt+>78fN-T^56jW-dcro+B~;tT!lbxq'
    '3qp{Y2+*j*4(WTzPLshkg@RDC<=nNhwjzD?U+y5iiVnTCzbc!D;8cN=tCsc?W4jjCS6uwTnUPcc^PH)vll%CmqAjR}?b4R6l7WPz'
    '^NNrOOy|9JzU8YF%$s#M^q~qCBWKDRS<5)1p+UH%s67>qsYf1N{%u3En^7Mhd|@<~z7=?v>&WQF4E&=+L}Bu;kSCDxawVk%*F{BU'
    ';anYgabXxg)N1r3&UR$CW+Xw(aev0i}P>g~OA?+w3a*wbf(ChZ$1x-iY)y2BMgtFv-'
    '>Adw|M5Ecimc3;J<SwiP$%WA^3>G1IL2^GI*oJkr}PkMw5dk>2)sq_;yJ>Ft<Ddb8w~SmDKJyc52GT!16yw~z~Pz<j+8UrO1qZmc'
    'g>p~GU&F1)^Kchyln2q3;Fm+=-Ta<A|r{5A<Mt_d0Z4f4!kMtpbSOy4}?W7Bl^ck@Dc!T_Wsrs4jp=?dgi_&WL(n%_sL-'
    '9jy=a&vLJqPaM;XfAGFG#7U$nu|LY&Ba;TT%4j~7(Sxi?FS(uOK-'
    'H@+B?&$xZPCvcMoSA&g%Zd!wHSEy8rSpvf@PUzddYnIFb99hpQE%U%%sV;`rep7e45ONAyFOc7*M$3fVm4!}eC?TAuS^N2_`*&-'
    '*aTp3y3fr^Q3W-'
    'CnX83+aeS7Zo5<!Zc~aeA@H2=?KV5H+xko_ZG8*di?w@qy`1kGktrcHV}p#{*{X*PbH8jD2#5{;9u==LFFquEwfx*#!QXi?^-'
    'Mn19;7xB^U#EpPVfq1MIAG1Z4o@rgH^mfYjnV!5JX4IA4GUxCNG|-asC9o#^vG@nkQ<uBq_R=t6Z`qt(jdHhV-'
    's*(qiB^CLjU#Q#WF!uE}%cd;$gh1`2Ba{{P+mO=m^pA8qf^i-'
    '!WwrySdr+`qBkNz|e9WqFH7C;i26+DkgFXL2l7dTP^C5~;F<Jd+yj%}Rd*d{rSZJOiQW+{#}O1-EiK8HnI$`x7!Xh&S3-'
    'K(U^ky(9<ZD#e2s3>0Sg7%9B=13-'
    '~>n+n^8g@$(Xl@0lmR=;XUe5p>Amfzh01!w>o;d#>(opc>=%savVp8U)o8zTWw39sGGDy5i9&kBjyXs$d1(ZxBDYz11rgE8H1w}*'
    'Y_+Jgp#khVYPn}N7@6s|+qOo1Ekp_4Z^mdU4(zv7O9gh^Y2Wex8j&J-'
    'r<sO=h71@}M8s4egtN0#R!4u#7EW;X}_~vIDR`J9)KgY0+C%*Z)hLt?=&CfHe<%w^8zF{>_eDftH>3L3t4@K)Y4z_1RpT(0~jPEE'
    'sx){9J)}HzRMvif@J`9{%sTKJ_i`T;e{X^hP%HQ!35KbkAe-wN{nR$)n#`4Ws1Lu~kfpcrtz_~4J;M|@yaPG(&ICn}Xyc|!p>pFB'
    'ICo|i!6o81(bd3jdtw_^1RtC5p<VACBd$PAE9*jJ=+D%EWXMm3-'
    'ZHf<DY|m5vFNZDybAvTP0s_EHql}RB9TotdG*>lO$8xbW*J`g87sRj>4ITWL0|)#)=Gt~u(h@-`potbMr0LB6-'
    '))<6%ayq~A)0QBZ0Sd}*k_O{e{Cw4X4;s(o|zOV^4Ytk3TNAJq7^B|aIOs}TM=Un=h<+I)yBdS8&0#WSV2qkm$d!gNoT<?Y2@g`>'
    'et(^b5WFBqh4~pfwbWS-'
    'G}coS8nh{T|sq`jZCC=P#vr2PoY;Knv>xyd>T=j4CmNzk`+zKaJ~(vS`O$~ECz`^6wVU+^7`R$wilS#BjFq`II%~=xn6)`kA?HRA'
    'jKXJ=X-'
    '&QJrS1ptV58!e3pAbj`Zdx$`tO{+{9Q3fwD|OiV(^{HIbml7J+usmFcj|6)vT(FC&#J=zZq7!g=a3do{hwj1&|APK2r4oJEpv$s)'
    '<OW|8FEvPkmnStR+6ERtM;Cr$trE^fU!uB&n?rwx(5+Z;A%D!DW^fmwx&V(_@OhbipPE1Wo^Ax80kHzrBJa<SU{UPgx}Ppr;SYu~'
    'GAmnBVJUIxrgtuQYCw?&;_7TNC8w5I2qFj2VBE%q`mz_*-`?gITJGFZ`uX}+lu1?1Dtu1ZcoCrV-nED4Gjf=hxRh5(bGhat!$$YB'
    'UH32GREO@bJPfRmtw(YE>mn5C<bKFQKu8`t4zAAZ~A3_B*MNB97R{tA$zc}^Hpi+z^q&wa?KUZvb%2v(u!A1@Z)n6PbH{K|wK%El'
    'IbLJ>3T(7E}!sOiy*{e_5h(u@722<y>#`jwHzxDt#+UW}_iNIX<b<bPw@i!m>Hz_=nKl}|8%_cbUn=*{{G9tm&JzxA>3HdA_o4bj'
    'pRy;t-KlsH`~|BfKGFjr&(Z9%Ta#-'
    'c9N;?jS7ZWT#S^wYjo7aL<v)n=0AuI+r)(;Y8{rZL{@mDFb<q>fcxhoXg{{>QY*n+EL^f&WF<|IzSPUGvAozv?<a5#DYX5*Kl?{R'
    '$M4tl_eHm1GT<*Y``-a7DdZvW6?`HIg-4RX-qE!`1cLB5T+RmF#cQ6?Xt6d>f7IvFiNR3+>fr3Z*ZXX5jH<8F+ko1|DCLfyY;7;P'
    'F)%c&r_<H=FPQ7ygZV8E-'
    'M+LoWOWKfTq2GhFyjetNqJA9dklzT7awSY><@;jd4^hM3;ToJzV<^fG(d{^(4vvYAt8lW2C>Yb7o>o7Cf7U71T@v=kR@fu}3KebQ'
    'lF*{kMH%d4zn)r$tas4UK@@WHae@Py4;SR>ebO%=hf!nZd5rBB%Ke9336w}VdSh9X}9-'
    'NcPWzXD>Wn}}!yG!!?>W91<$y#Nf7Uu&iQ#~}G|D-'
    '8e)i(hYJonM|C+S+GEYxHGpVS#v`OK#MQi(lT$u*oL}dBa@CZv_MbdQKyf8T3D5Jzi)KOy~pci~}rX?8+)4!V|mxt}1LGN*>TZ-b'
    'f@npx!E`)hnI&JaoeMT>Xd)`tEl1qcYr^;h_0-'
    'Rvw`rVWszh5&G+_G(RvjKhnmAM}~aEwl2P*SGs7D(CS9dsiyN)>P0`j$H(NlE4r+#O36Azyu5xR8B{fvX4`qSqGcB6`vb@;hZRq`'
    'uSFwcUlkoVS@CiQSd{DssN}$DjHjHU#Vw2hHmcMMpcB69>W5`Y`h8bFCZoT5T>ZH4KJRt)6Y0|6g*wnld+bpgo@oZ1-'
    'WKt)RMi=Pyy$Gx#RJ1#mmL{rM?Zz3Fd}}!xSxhiQ(|X%A+ht7LSpB_LSpBlLSpBug~ZOqg~ZM!g~ZO+RAT3IhD~8vRx$eze#xjyW'
    ')zCt)iq>@sN`CdR^}1grLQv#qSrK9p|D@hnlAhJLJ4JGUkJP$;0sNc?Wr8{>aaA63LsB`MFkpq^N9Q1E^L^G+V62;qde4pzYCk@Y'
    '3=DQY-V|^HHDjaHiH;))s(~S0$)<9`b#L~2sn}>GREo`c5L(-'
    'NYe;iFgp1^=N#O}Vy&@YZhD%<0GQNSJ7zy$$i?jM3l*FLy=uo4LU*0m+SAK#yL|c0%$MKx`SRN#Uw%90%WqbiP#3^PPwUg}0v7^-'
    'DYqz8{YQ#kHKYcISlA|ZtH&Fv7~?H|+9@`CI3e<Ss_5|39{S8x;O+{|%p<pXq#k10?NBr`Mnawf@eC+SxG>vuA_fc<x5Z`IDVo0A'
    'hW$7ozYNG@2_8EG$S5Is7htL+l<W%3k?;d{q$1a|w`F9#3B)g%zuergdl9{!ce${EfHbC1L+_-'
    ')?6wMiwxw0eeonmJ4sg+GqXG1eHlsy3Q`GOSe0?%q-'
    'U+xJ!9ctm$UOEPreC~5e1ND{`MQ@Fv>|4Q9D_E(%!*^s#+a>g4B7<qD~>^%V$R4hXfw<iIR^Eke59m8d4K>t>bKn2h2OJZVzPqGe'
    'Y6Qo@rHTn7N`qi%(LhK_8NKL^i@I4;`hDMM=`iA^d#M<JJF5E!I-'
    'j)m)WonKW$6*iWjTIhz1*1`Xht2QJ*us6ZJWxJ5iJipfg!#jWx`s?#MD08w=883Jsx`#rU2}167`y>!K}AxeN=uLPjvz8`cYYBG;'
    '|+qjAH0lh&Z$L>vjXHGE`wS8WycHtInZiqeT)>2Bnu*{%v_dv)-'
    'fNq4SStHd3uaGqyB&;OgzsV>kfRdJE>2G^LoP=YapptHz!QTsd4c92;yxYzT2)K=z>UZE1&7m;pRg*PPu{idzHK>1;BsUpx{wh{Z'
    'Yz`}}3=hZ{e+TJz9vlhBmH3mNJ*^2l$&ZDKyFrwo<TIxK%Kf$A=ey?oX`XS*G%&aJrykG8A$|P6IjY>(++Vr8~q(OS+v0*8_@;J<'
    'rUh(=S%xX@=?nwsn^b=)ZKX39aqX5FRMr6I&^$d7y^drY~1qKnYC!D0<%Zy$cgHe%Vh=~<(^+sK*2suDyB{#8Q7uys{)#moLxw*Y'
    'vZf?)a&F$@Tb9;x}+}<%aw`Zm1c8X&#8<j&r+2$%a1e9&Akz0YX%?Db|Y3zkjAkW*aCkoO-Sd^Y(w5vXV{rC%q1tg4au9BR=jRwv'
    '>o%K<G>%)V-yx&$v-#5mU+=c0?0BF98>8b!|zK7|mAWrVabX5>1-^X-'
    'S5GVIwx+;j1dof)V#7WXaVOYb_9<w9HpB(EkE@F_wJ3PikcQ*qN*fXZ9c|{+4Jib@@VE4Dvfqv9ZoDMvQznT-WGB=MRzOz*rZh<c'
    'gmqk#aYi%W}Qh$}U(q>ioO^ZuB<lC8BS$USiQ~5JKs42^2nca;?xywm0HpLz<ruul)DKP!IuCkk6;eGK_(o0&M_xp_c$?!LdHE}c'
    'icUT(l#igcw`BB32H$zpHApF$&ufCb`5xMejq3l|&{97rvmMi}@y1C@azdbywSN<K8am$r|C*|AH?Y<6rRUGMWgea8=fkTt3HAr4'
    ')rjhU3ka7VN)4~zSrd33@s{7z=%KAT((OW0`N{lH#Vl0A@h3$i%6=8`Xa9Z8QdF-'
    '|II#t?>LhJKTzbzN6Dn_NfR{fg5tPc<JRrp{mbQc>vusvNsHmZf81;W39(vd$Ky4D!`WueQpuT>Du-'
    '%Z$`pE@1~=owM`_P8CM<+1o#CVW8Cx^qnU5T~wC5MHLTf7e1xj4$EqAs;4k?NASEsrQe>HEg{P3K^^7=A)y(wo7!t1K}~m$p_XCq'
    '8&q$ryxiGe?(_V_|+M0A0^jbKoDHP!9Gd?4>jy3W487e6ZYZG*={cUK&PAoF8olXeCPT2w~27NM@)KB6v#d9*r!B++#^IiEehlwx'
    '9oEwqwaCdKF>=$<atOIz2$xQkd^9SG1yBDU1*A#YvQ(jA^DXSsl=#jrxH<vF6R>CrR=~&Av*x`!U;9=T^z^tn%_V_#xy>IQH1__r'
    'lBF;!7kgEKm9=1%~HY}j^<4^U@6B7$MDncvF<;XpZ2tr<Avk+X)g=Wvg0}Fy9T0l++Y}X=nOFg?L-'
    'Hu3Z|Ex<iMb3e%YxG+7(x&&;LrB1c8c`T4%bTNeM2}inSQtWd*?nk*+wdG_o;Xu_$yryOKU&A03PZPKo}*u2qnByi{FBd9J!-'
    '(oQ=ot2@wE$_gQLJhMUwolcc{z^fZcTYCVulb6<CXMa7Y8~l%WLQ`$=4VISy3;&xe`~(*Mw^;oL3JBk3)gv(SztfKJ4@NCP{{BCR'
    'T7sMz{xE6@a<=lLs3pjG;*X=2ASa4HiCTi3$ow>F336fcv#2GwR<NJgf<t169*wnM!%r`RWOH~eKkW?RY&tPz<7FJKJkoeEYGKSm'
    '%OfS*-_W~K<U6i<h371T@@qWhnRVvFqp6QToAA%A_R9kF$Ga)__%sh%!Ia{JNd_?-'
    'KMH3ML$Knlht$?%5T|i4>w(Q~BD@#Dz$s;>RhVj}6MYac?qZ~jB3t9|IRHY``Hp2yJCG?Cl`NcHk;#I#pon8ZZ+cUqG`!CN5w^c}'
    '#d0I{bf=BfPIOol<%wuw?c^_kDYJ`%urA5WX(kXKbYY+LZEGJ3`HW)fwJ~rp{n(`Ivd`(eA1q$>;+X$vM|nJMF%{>)U){P(zwV+E'
    'fl*yqs0i`Xw_TL5;-'
    '~Mps6=37j7!xp_xkIHz@glLA07sUazlQ21PscJ_~B77OE%_*$G|SwgdZLU?`u<jcmiCy&6>~+3%uz;DV}_Y>gI-'
    'd41JP*+>4?@GABX-haw)AxxNc@7mndLXoA@91SN#aq7Td4n+|0TS4;6lvtkH+QMocdfiIf1S?G&O8TcuDQ3+(9#upVp>}T*rv$hC'
    '*QR(lja8b~Sm-`zoibC+yH(eBk;HPi7s5I6R4;zkF*F9AnwKq^-'
    'ANAphbf|l)2IYOAUYERg)xCO!=o$f1FPhdfa5eVDMp#U;D@yOOK6yfEy_7VE?zBOZ1&rwro74_ZJ%dsUY4p`wQi7&^zH3orsbcDN'
    'sIpkaxRW$+`QX7h*GcyFODOco@rFsd6AB>wk(~@N;Na{sV@%GW4-coMT3JP0VF@f=m$NNO7h-'
    'xP{EKBVVG=P<m=WbHgjN|1SibgYk)vMn8BJi*F@}zwqJWMy^!63lbey60t(c(W4IRt$bdX^kXgs_+VhMTlb7+JQa>P0;!UvO-'
    'M!h+uZX$|aZ$X)xXrk9!Q8p|h==Ii=#l@~$kx&e4LLnQ=g#%O6ItO=Dxmsh*ENr68&&G1vH9y`*oO<-'
    'e+<(|BqW1;U?C!)XYkoc?pEtBQG+td`!rRiZ;K9tl=IHfhgGuvgMlHO0tpMs_+O?roWAR%J<{@Hrf15$S@x9p<d=o2RdRTaEY6VU'
    'Y-'
    '^a~t1k@KRHg4L~u`!Ii&h&ERv*sl>#<((QXp2y?yiQnB7;kBjyD&a>EsPL224c#Ra>)*0ov+n)*oSR+oe>4HUmJ}?1Hn}2SYN}8o'
    'eC}MYni@Op=W&^%M^Rei)L18_b{#ok2gmVVLpR*M#<N5MEDm^`Apm@fAisp7IADHNNn8-'
    '8IJ?~7JttXCcNA=x7S#C&CN5Bev*LCLSCUic1WP~Q;EqU4TUoSFj-fwA2K^RNl5Hupo+Xs4i(ZmNePhE<*{GARuY0mHjCo{M*14<'
    '<b246!?lxh2ELO}(LQ3s>$NfYQ5%jl?NQQX1bi05W?rEQE2@`I*rw{`|C#d`M6@Xob2Rf6rWn_Ad&)q5hl4GhoYm~+s)Ju~Q5&L&'
    'I9|TLWeS{L<mk0Y2YLc8N?Y@M_I(%3l3s1MgITP@PE3pQPalqK%mUbbtSeBB=W=l>&P`;)4EsDbSrIKB8;R5ua=)Al{Kn|L-'
    'bs6n56Qpz`XM<Z|L*go{Xcw$FZ`z$_(CWAm(LUU|MqzT|1qB?@UfF^AOgh?cSL>bmajrj)P3s4>^1zoXdy;341NOPR}!d|4x_D=3'
    'V~Yb3bp3klFx^UKK9?`pwP$uhis$qJP^7s4}?~!_6OudWj3@r&xY3MR1jLCTunhrG}TfaT#3&K)Cxg*P1C1(ou`y8Ug~`UDK66{`'
    'm)nHzv?Ez3A#qGrq|*O$qBknu%_4J49N+)L9nJb;ta_Nx=Ex@>Bav=LJht6ze=c~m-'
    '<f$BS#wuJiY3S){UAkQIz8Oe#Mg}wUed<!mlh++&Dy}?Ypu>xf}`5LQ>7Hy5OGlizzfu1NWSS8=R7KMwPT0rx-Hd?gd(INa++fT@'
    '$juCJlcK*)()c*2xV<=j6}Q4WaY!H;KTI^QWB9DjvCCH3#15F~;G{{3MS#4kzQMc?@q7)2v>h9ODkI`uo*}_+g)6eBp7=*gajPmr'
    '*(xaS4>W{<XD*R=R0*)u3NEsY0ov^z8<7Q>u_%oyb6maa^c#mBia!uPsB*Z&a`Oo8`);>+2S|vFZA{RW5A0zHUq5IbL74*Ei_(bw'
    '_=pUSD_CH>I^Y(PJjVVeDj&2^d!~r+7?gxQaQ|V?<MehUs(n#v?|VaI0^C>phCkEbCH%?uor+@e{xn7!|WpwU`8ic|xMk!YItD_3'
    'ld7H33Wp5H6f$v$cZ@=h^J-;KKPfn>)C$#7YZXWmu!SnyPjh6=IFrd#c%KREjk!&B>e*LUVGPL!Pf;_`Y)$exEL-'
    'U8?YhHn<wpBYdfzsx83Nk2^;ZPNvETCtGENldCeq$yXWSR8<+_v|g1FPU~0d-'
    'fGYpzDz_PD26(@c&L_mK+C2_JY#*!z*K<Y7c^w1N8ucy7dXASF>NYG*--'
    '8+M!NGt*Jc*e?;Y*Z?R_oKS8)MfXTmP7jn(W26i7f4h~VlSFe);3eX-p<tP|81EE@@k?sfJ7Qj*>@vlR**8^zB-'
    'x6LLcO=T`_l*PQoUJ1tq*^B$YzT#{vP%iK_tU$Ry%&>w|0yBem<*O*K*4mb8ywy$sC*Ohe6(<?n8Vt60!)|Xd+~S=%%V5~hB6*}C'
    'yCz_x+U1S5ZIdQ94A(`Qq=m{+r}ZhW9eQdWzL-Uf3nlCtx#B4{$rVqbNv?Q`Omf9jV3I4I;*wnP6qcl2x}OK5#R{eilooLsHn7OK'
    'ni&t@fPUCX&DXa8D#6hyDtm-'
    '(g=Wj|5>KA~%WtZ#jU*_HNHZIK3G495!bD<<k&3*tCZbeS3oAY<D2fg(&bES|zqZ~^0OZ%zGX+V1eZ9Rv$#1B45cK|y^^O86zp0*'
    ';^1iA1TaZWN;7qlfW}=jw^Te?O`I>X0;IIaxce!w1D#~`M62l>vu4lGzn^W;<sV!?k>59g%x*~&xO>C5~7}Uw_AdFqvJ|MrH7Xgu'
    'MGlSAft;{QRS6->_=9T(hUa7nDN_{`C)IE8n?p2pcsu#wMLR?whW~%-'
    '=bR#)|_@Zc)>sswvC2|miTq4Hkl=f(4x+(Uk_70gmHYzhv2xepx^?9aTK|E?Yx7=y<uOH?`@QPSFk|l%Ge%b(d$Vt%~On8-'
    '7vrZEZlCd=8W)7CQG=x!JErV%j&m7{_h0%mVy}B@(aF`d${<nNx$#mx5EycA9xvBA=mQLzdxpbKDOc`m8Iu`?dB{2FpyIhRXz>8L'
    'jiVQXwh<?4gR)gFC7gdxotj~hhmcowgir2@Ouw7cSRi@r>NdrBcSyI~;gtYAn#?;J$F|~a`RNJ8-'
    's_n>3s&uz5(F*V$m%Q#`Ib<o&U&5>NgjZTf2OzL#BTKK*ki3|3mn1v>?>?5ZT{Hi#G03Afyu*bfc!%HR#EfqWg?pA#|8k+gAO6kK'
    'HT4yKga2XaX1a(!<D))PTEbghp`f#a!JL*gH$Lq#QGXx*rkNE`51c3MSOMN}Ptgg$<v-7qF4G0Fq<r=LxQ-^Qaxo`$)-'
    'i+LJne<~uC-oYb>ZB!UVrjoZNnuD(^eN=Z`7amLyj(SHRC<Qh1Es@?rOYuMAX(5)WGX4j>L2P>!@J3Rx(3cr0Zp61`*OEm3&LqwH'
    'kmh)^RrM$((~-Z8*V(y<JIunPW40sT0n0g}p$(5}0NH=y+zv=FYQ#Yc8}2ys``;sWChkIu&E>u{rcby^c-'
    'Ng*VDcoGS>$@Rd{q)q==8(2s{MzHLm+&w(o*8)ZbKVP1bs!d*Drh>Qe=UaRyN_sJQD<@@Fos>JdKlBfrZpAQ5J4>OAHpjh>+I0aj'
    '5kTZE@Z*bvuRT22O2ZZ5WB0=Ucl74?4wH{-`fNQjEm)dsQHl~%!oh-;4wGLXY-pPp(IJc-Jxao6d1hEV@Fh~*nM?d|SkH$g`E+V#'
    'hH*a`uc4Qj>iu08!tgw&8baNkNgY0VyHpqS|tgwGHQrlmtAqUuk8gihO{@#GYg4`ki1{aRC;Z^}KIPBir`RTJB2#?c!JGC6osb$H'
    '`NNMJzzVNxCVi?Dp`OHsXwMt*^@N<j)^wIpHe`(Q^K9--REEPB5IDY!IrSc{mpKYco5DF#YeI^{|1Y`?g*b^KRBs$?BOn%Zy9sn%'
    'nXxhH~3-@s09X8x%Rm|_*R<&v9{89)%&@_-'
    'Maac`1{f7@H*(Kfg(zG@aN84<fdW_AMsmEKnIyw|8a#VV|L!<a(ULCY4vOUgUhQ8%-'
    'Djr7}`ULGjHSc+tE^CDxJmi^i+(Bb#QbX3P3Lvz58}{$$11Y=Xr5cBv5SJ<pMS4ty)@V}&bBT*cK@ero4r;CzMV9{QnVkJP>xgNh'
    'lBN`)JQ-~%=4MRUe8UJcd?sWNO=m3(q11x)Ser#7-vL`AzWk1}k}D3xa&k;Mnc5p%m3xEt=icDz+#6hzdxH<;-'
    'r(BQ8`M>KFWp2bEkDIjhPq_>(DV{E5|Rf}&a_m=*2aX7>)6_w@QD(={K*o%{3#t<S0;Q~WnG>$;Rh5#8D@NQc#W0P?pCOga6xxFg'
    'kYqMqq{QIaUGnr6I!N;#ceYN_bwF|A3^|(0;S8j%{SZSBs#{BKBIu<Wo?6zlT7%Y6T;QZF1z4~lxd16%xH-_A1`S9QDaAVjuf4W3'
    '}x5|U!bUZfrK2F+v~Mfirw3w&aztWt|Kj1voR`amjZ+}<xGIFCM<zSSrd{#<g7_Bi=;Ir41l$!6dKUhVqDWDIUC_TjnODSYK0_#e'
    '5VmKo&xCYGYV8#--'
    'OR9P~j;Pen?@EZ`!9#_z|Xp>{4LDk3DT*mQ^z0CzKTO)AJ_$jP47kA`#LlKu<$D0q8qPD*$aL=><RqJ24x}iTqf}8YmduV<qQue&'
    'I^f0Wf+(;@on_z&Bkh%4r`<Yc*tP12)%+sxI27#hkJPr}vol8H{1*K}yT`8{g-'
    '&Are+O?TCc?owh_MW6dp(K2xwfX8@%mH~!`rddrQ!g(>)-'
    'TbTm3w6!T<OWPQREuGTs|4=w^rC$D^T+;JhHiDS%vavQgT;&kq93%;G-Hx>3Yqsqf*^7HYU&vs?5q-'
    'WZ{O)5)Ijj!eE6@YyysLvo)L_HfX8>Dbd+Be-e)xSw-'
    '+Oh@>wZ8XE<K7pq+@5)s*H}ERUjfAJEw&b9Xqe6L^^hrn*Y(UTTi{6>DaAr8f=f(+Hhf7l*4VfsGSR7s0K#YeBUKhFpcmLUcjK$k'
    '_pGB&^PZzOF)H)(~HWB6s2%^Xz)%L7hEGf-'
    '+DV48k_{<0v;Ni0<!`h6r2jPf*YmNR?kX<0v<$+7sH2D#qi<&V&Jf*7&4TEwR4eFr9LiIxvj<Nh5V76v<rHA=zpXCR|fJI6X=U^a'
    'hO+{((4e9nv+DHTWRQu$2<~)Wj3QAWyR>adzDp-RSIzjThU+vF6L0H29gxi4zsMc(m{Q5Y-VdXz)-'
    'm)r2Oo7m4sf7S}iIp;^`8qMyzTYevLw;jhcpBFFxj3=4D2B*o1#np|{L8Ut_~p(lSzMdP!>;AB+L3OV0oQj@v_rchIWUCS4SiSc^'
    '3xT@A0c>i9@k!yz_)vD_8tL+em=5NVT(G|ni!-'
    '##rIhK&+dmzfd=GEw7L6^pa2p?fth&W?sI$mZe<#yvL8RJ^}T3P#JMphuz6DeEny$Hq!B+eC|VGRx4796!f=VI>wjjgm74PhyR{5'
    'sW0(s4@bQ#7<k^L||l^6iR4@)bpE#oL;XA|03M<`u^}1F-@vhhqsFEZoMYFP2_6p2g1J!eZ5{wWks7-(i6LnU9-'
    '(1fkUcU_XzxVG+P;ttF?Plgc8AQN;&%54bANhr8aSe&O?E@q~!}dx9|miW#J3FPvHx^Z{Z8PU*QY9f8h&!K;a8~U>-'
    '#E7}c$K3O>MsU&YCHMgXih`Rkd8Q*FhnMGjwT!gb>-'
    'vI_;I+qL~W0i@}WRVB8xsp93gO!zglVw0vVVcAz<`}c`iiOGZ%*r_Tb1$Mq)R*>@14w(771bR7WBqgb5V7w$*{pP?Ea#~k!jj`YR'
    'ola&QH)oMxe$0hGS;eM5;^WfmtK7~|{u*E9`8M3qI&_p(z2N9G^i^%)>*3<?FUyA?-'
    'IfWL#|~p`Ip0t$?qkK`yiPMJy6MCYGHhO9zR7-'
    '&VT%e?0vkexEh|bQ#1S%VRiWl$OUSTwMfab0LWXTB%$l|_N<&j(5d63^edB(UX<M+IzCaVQ2Mm+Abg}x;-'
    'i(Vy<;GY$HwR91I<t|b$XgW!`Edt*3c+xH($U|pz~N6h+7t>L#yaWdnPDh!=n5H%9J)Y;LWi!8q1d6zV<>p&>KKY1x;Tcy$2O+qD'
    '2xK8uOe6zxqC2Cx`d1aM96I+0J0G)it!pkMUz=F(@;s2NK`i5Y1<sy(dh*uKH-2C6{zT^9nhiz75$6@T2!E-'
    'pLIZso+cjD3f`jN^H8?jemGlhKawrCAI+BAk7dj4$Ft@36K%1rxYgS;?tgLNcc}qz78+3{*qtphWu(?;@+lb3y4|=jRq6`AsF>OI'
    'HEdY&%UYUGS+X&1bvmM<S4p;9T#_vZBC2Jg%Y3-'
    'UbbY(Gt6I#YG=1bP^#CNU5CH=bRpp&ofSr|h<{+c0yfY8%vo`4~2&rmgcs)R&m6NtUu+Yp&1OEA7BeiS2CC2Mqy4G6(1HqMRy$vA'
    'Lr@L-0$3j=Wy&U$(3UBF`prb`c?%eiFfMlgbE-HzSyl!t!B1H5PT~gm|2u6*ilvlblGc+8eUiyh;Fa0FBCsRo+otv%nJiu*Qk3(v'
    '?3vfExs)^ii)T+ZP(pASqD6cb+2W<jvhGZmA-'
    '77PwZl4UQ+c$&i_RFBU{WGZUfDEcTP<O|6*dV$)FzwE}1CWTkA9lo+Xd+Wbh1J^MXkacb=KBSJt&`3XUspq*l{X!hD6P9IwV^0!x'
    'W{$DeCXd7{4`Yd5d+Zs5YML@7(}=-GOQ%+s+@i5USYeR@pjC7TDxM;N=LCca(#-'
    'vjTXuxTO2KvMYcFvD2r@yv``k=;%K2Pvc=ItS!9c&g|f&NNBD(Q-'
    'nW=6iUOptDxm;5o+M8ibgxL%A2ZR>?*9$K$}ckOQ5HLQ3V5_0lXc?IUu<UVl!)J*7L3yFdtFKX95*6b*a71GK0)2A0)m`}f%gMJ&'
    'Sl_gAjr82Tmu9-!^9o{f}CMuYk?qV!{?h#7LNf$Q59o=Y*m>UAbpcgOpBox2|!_G)y~&-'
    'Cv~DRxor2r*bvRW?>1r(c_v_Tt1Pna8?CapsN&4Ox(34&TUG5|DH^HUD+MDpd!<;UE}s;N)Z~*Qk$S*VAX2|iibL5Bw#P|_gm$T;'
    's+Ch<=csJu#A`iA73u+9!q_jwHr=3|ZXe}_iu5+7qQ1T&IEFAx;foR5sn-fbXxwEkye!b$T=+v;AZx>Ard*(jqD-zIDtZm|!$rwl'
    'KT`A^>PL(6xqeKl=R}{&4T<`R+>bz&IN@~Ub|3uI_%@|%?_npGDW!3byLdx3>f`9j3QK9%)&U)rRJ{{}xMFjz&DfmlGB)S>jLo?r'
    'V{>lI*qochdFOr@M)I9bS?8lpu5`*eA9He}Q=<8Vi&yw5g&6Co;*rK15qyy?{SgzGW{9?dhJ~Jdp-'
    'q5w6Su#<#J8$?)Ot2tW!m#Rh^2NBVh6y)=UT7Pw@W%+Y8RUX6o&PaEVeoW>9xvE&$_%2_D<5UHe6ZmCu|@pK|a7m*}(uOh7a-'
    '6AwVdG5Azf46skf#Vg*`@J<M}2$~8BzGK4(ag3rKH$a5^XjXV-'
    'N*Mi%~GqLj^93wzcs`oavGOeWKd^6k5c86WP0wly?*T%PwW6c<Yb6@zmN@=VLzfdWS`@=6)N@I2Sl}c%>2~#Si@j&>sN@=VOzftw'
    '9L7PGt6XyT|3LoSrI#*OYpTSRbe5iW<XnC7E8!k^P?`^okl<eF^R@H2Da*eOwV|eJ#n98g*5MQ-'
    'Kju=>byR^f5ql@x%vShgVyWAP+Z(eF67A6|)q6wE(yXQwL*>KUh@5vEtp_Z_VV==n8{MyEbOVih0--fFj8}1-'
    'mBdp1us@J3Ak&(Pg3VT=doZ>C@?&vwiTk1X0q=~oGd!tDcZ>jf1lP2C$?~f);yroW$CQXNFYvvrI_?>5B0kGOk12ijTSbK6Ait7!'
    'z*2*G>Ds^Ofiqd90EmhfIpcMu{IsNUgvbCYrlG9hi^;Wi44aYP&i8auAvd!xCX<fAk6{{&7ww?txopz+C?pc-lrssj+q8+L8!T8Y'
    '*;Svz<y4biBXcoIDZKUYe@q~GWV4zK$;6}F4L;uXx8a&Tu;_#-BB3x?AiY7r`^*_#c(;#5S=}+c@`o3Bfq*v7vY$$OO_Acd3s^-'
    '0>uB4!Tpk7Z3>b3RyvLB4AH;@jDclE~739DSaiF$Lndb5Ta+7BSLB>FgQW)QRl-1GnhEYG;?jjWC{q3Sp>;Jje86;3Nd|HD##-'
    '7iGu-'
    'u$vv{UaX<wDi+YAyZbKwWYQ=vmXagPt1xx0X&Cz6@L<_Jtuu1fH1vuJ1pf3;;{M|01(As^|O}p1u?B1#YlynO$WjaKI{M!Mpfb45'
    'vG>u=_Vg$!IQD2ARN62*M)WMswOU1F=p@SY8vq7Sq_i!#ZJ2oOKmP|mSW9X6>-'
    '$Aii`FWA7PSt(H=ySr@7h>9InW1jT%fHggzxJc{p5{GmY9!y&COk*79(;K6fx`K6NNY32Hud7+U*0OSF_-'
    'N!EUmr9?)u_KPj$E;=!^lxnrJ_WzHz_kh!)y86fW-kC>Nuq@W42_{B2Ys1P~Svzs|Dt6Quu@JH1FEM$suMMzcEJ4%O25ZN@AkDst'
    '0x_1gP*gxHYl$ff3jEJG_nvd_%<~v#_<a8F^2y7{%(L^{bHC@F`aQ7GXz1?T=tuXNji99(s-'
    '>(37HY2aG>s!R?3vsETg!ETp47%=L}D9p_kPWQ2-n-0`#VOs)0}6sQ-nLsxi&jTxYL|(vrB|K%{e!_M!3_Qce7iBJI%Q_10&pN&c'
    '7M2C7|1B?$lt1vLpCs4JIEuf`8Ls<z*c~Eb76z6J@PLv_G(<m{UWvbmG<^&puK@%FbFE?5uHU#rYNx09%)$vBlnA4?5Rgg`fAh+~'
    '+mt9)r@F<(y+sTCY3T7?jo<&M^k1HQTwxptR;#r??*9xQo6ErDk=q%-'
    'rlQPSMG7^9Ofvx=vP^KV}+VyU3%>^1ek99bMZdozgQba8+iOb)ivd)>3hZN70-'
    'ZndH(XI|^%Z%8OqB(BjL0N*7lKbh>ylpwz{Y0j(~645)Q+V?eKq7Xyl25;Ujk<a=|ATZ+?k@`Jh6T@2UB3Uiyg__a<}n%m=Gvunv'
    'jtW}==gM9=$aZ*IZacS!vdw?vNU=vvkGdcRxAxVi2neEyL>(i`#!wl@EnurYCaG*wq%ygKSs);45%l7%#qX7=S@-'
    'lmI1!|~dxxKg&6;$%Qy|@bXQ}TnoxEj?{vcg`BM(vcWv=?JgIVG#?#X1_x8adkEU1MP<NBGy(Smekd{td7o*+ugGY7u0JEmmjAf2'
    '!PUR!elzCB=p`blg=JqkS)KG|d2Us+p|Sab;<)Q6d2<O^012jRvr*q)HujmGm6E&I7zJ!7{y)RNKR@a_zNPKDo98Tk9gpgml9KV9'
    '9>0hxOT#<XB&u?IX%^h&bOx>QLKG9F_HgVki5L5gm?Kuvi(}9oKJ^4nk4ro`yQ#k5k7D%SPEKB{3_zqODQdo_G^!P-)M-'
    '6;nmLU)=j+(=#$ZXO5NF^omS9bF9RsH`c2Q>vA8g__tSeI17!U0aPW6v)#<@?y4S#yO}-'
    'Q)wVd$&Ftl_cEGu9W{@l5ft7wuNRnWoeV-w0yEy?POR%OfEK*j{4Z-'
    '8e8nDWBoXV+b!{^L2&=s~#j|e#5&Xb}~8v2M7=~@`rMrDeSM+5{LX9&SYJ-ReP9C21a9C3C)9C1!S9C2<y95GUeBkEzzku7ddw`S'
    'Yp;5)NV#DgC(r0aYt*p9XxGu8KSBp%DO91T&LxCT*+DWlZMD07=UU554CoQq6E7UMNLu$9%m6yd%q%Ry)2PX0{X*`JBK_%m@=e<t'
    'qc&qTKn+u;B}v$y-'
    '|ptXGaBPgv|LhS=O%l1MB)v*i93isPFWj@5M>Urq2v3=R11HO7Y0Ngh=0N7;M4n{4iQ}z2}uxXgUb)q&uQ(^lRY)emWcSIDB+Z`P'
    'Re0DG66mS0W6VbE1%%O~Ck+=X&y&XY8!3fJUIXp5iMNA0#>j*wan3C?+E}qwf`u!imPq;h=D+^^3tW%vDohkS1K+aD0+V6oO!lvx'
    '|V2AKG_y7)KYt@oP)#1mDe1>MFjM?}c&s`Zl@&#HXGJK>HPe2*E@fDuSZscea@><>Iv_ZZ34jCfNJvudR0Cu6%l&xg$iE?bym?*B'
    '60k$oeUy+n@$ez<21?)Lap0{9f!%pSJWJXz`9_WH{cAC>PS83yspV|Z;f|0^@>^!hh*!rIjCJL7eE&!X3gHvsq{<uY%tm%)tnxg5'
    'CyLwjBA9poX(;s&=P17HDHC@vmcQr%PpUljcrExGTDJa%l8U(9A(5(`zf>m$T+(!*ca;^}h425Ng_;eZ(@hpRsjyhGFS$4&4$23v'
    'dGI!|g%Y?O%czjl~LTbeoONoEHqS`NaQr5~?T>MnatK@b0Qf0rEY+m?tzn0Ale;&}XdEw84?&?d0e;(4ZmEq6BTDCI$dBk0Pt?<v'
    'IPM1CwnVX#wt2h6Qk(B9fcjLce<N~_a-'
    '3ZpkJGxA_pRBn}_jGOMq`B`R+d1hryOo{n_3=t5w5<9IMA_y=NuX#e8<z`@R+Ft_7YPTjDPAL-z{dSJ3P-'
    'RhULl;p#=SWThp=&Lj>0KyikAn+u(^wGV<bwt%Pq*iW2C9N&t3f|Ms}nLZh62qeb>j7GxL|j9n;P`29-'
    'pEP&IamA2SStF=3wy4ww>maV%}8>dbV^cKnF4zAP}Frf5k#gR-'
    '8XDeY~o?&Ll5cg=QXGS9rNg{y41_FPivyzFLC)f$UE*omgv1=6<xmfD5Vw*jWv#nP(rNQ{=QQ!J0hXzB)a$T&=da+%3}rWBQxW``'
    '3z+k<)82<c{touZ97PRVxY!o~Di6pc4`q(+0(;+B%k3K*5oYDvh0Q3-'
    'mB6Y_nrS5FEDsxjKdY|z~>!1bnPuO;~a&NX&gk`K*WnvL{iJ^+Uum}CKL6T28-'
    '&t4?$Ef2@&91IUu<}Rb`%1ktCXOi?wv2dkQ?X2uCJM{HcKlJrAKlF8$ANu;bANu-'
    '+ANo4m4}Epz{xc(t`413!`9oyJ&`XgqT(F6}@rOlW9X}3>sST>Z+u<N@Vb8QWh}`1J2E!9{7(*J&W{dBHz+OLFI~$o_Wy-'
    '8N&{!(Vdtu}t9NjI`<!H8*umZIQ++V1@x&l60ScJOJje%V>)V@Ve?8+OH=H{=)Q$W^eJ>Q!6GL12su4TzK!1IsyF}!65r(Ut~nKQ'
    '50#?FaX?1ARID|WPV+7&j?JkGjeFE=M$0dfn@&u*Z>SinapN!L1pav3G8hf5WYm&3iq)m-'
    'GaG?cr6HafLbg(3A8>0P6Nf;<*dx(I2YAP<F<el{8?$Ri=8%a8^N4w8Wf>-a{<=V@#dH-'
    '^4IW23k+^hFxWGBr#d^})VLk{sy7lQO{<_zJBR7?aTP$3FvtdFCzbxy=mrc!#fgYnQ4wx5HDXSf@P^Wh~EU%8%vwT=}s)UnoD8=S'
    '$_s@^mUcmgg(w$MSrw0^6-wA~<vsaL|Sn9tUl}78^~W8DA%hjkdHvy|Kx5Og8zxkTRCp<U83jqqHbuFuO0%axjD0eW8|<8BFU-wV'
    'b)6aXD;63YWu1yW=s%>83qU<#M)ZZ#=U&*)#|h>l`IS``x9__&~}6rwQ<>D=}+ZqSMuq;wB45ueT(vfOjq<=fR-'
    '34p=JXdpWHQlRm||h$>%Tgx%mXBR^Lws5ZrdM$Xn$ivd*QTSUBunT({|9FL=8ox^Djno77O;rD?gBKN{`>2)QE(fN{{Yt)kUrKtt'
    '$NCWqNlxyEf_y*RMp6!Duqwb;w2UA9UU|FWPmAU<d*+x??3rTFlImSUB{zY0WwhuMvH!<13H@ifO&9*>;USm7WD<?eo#NqGe9scg'
    '#;a}G~{Ofs#e|_)pZ(tq%VCZK!5D3$8Z9xdST`*3iGbszVL3;~#l%^V;u+0dzye$?6;iV6;7W80hqOvMH)EONT-EGAe5hmYufB1L'
    '6tBm&yE~C<KHA-g(6SWab{!F2sJ2m1g%5<>*BF>e}Q0DZAb0wpQSj`!O7hsVG8-'
    'o`Tjfpb`FCsQ7XXRXs)eLMIUPAO#&i%X;Yd_dbybQ}6QtfJ}xh$D4XswKMM%8y!HmX*s{WxP56CLcyLP#}5nNuQ8AWc=~G-'
    '0Qj26nJ;h>WLV9M)`iP-hHlwmYdWgEiZqs4s)J@Smw`hPUuvsB4C|@L#ECgLUBFsE6insnVfV+y(=DwKMGpPLKo@0b!!MRr}K2z?'
    'JeW0H9lK3k2x=Y!7mK41Yvf;k4map4k-'
    'Eb)tf5)t<K;{a*=Wwn_`z@W3#YG%)~Fvc!N;NfHA@B}WVxl@u|6R5HYXQb`a4OeH^@^fVVl=3D7gzA!TXlm_HWBl90^d09TfHnmi'
    'b8lI1~njN%hcA_I9r98NDhPb?E+b~NwbFxKdYqtjGrseqZsZZ=C6LgMEqqD!x>iS+fc}-'
    'f?`sn0!=~U~hlQ%d=g%l&LY(1JgVtmh@?qvE?Y|etS++F=S#y9UHjocSS<~wQRzBn@fl1A=JBJ*!woHc<<)q9w+3f8h(W%~ti!2-'
    '9$E!a5#bh}_9IYtne>*YsraKxq?@7jXKJ9LxVq}>Q?uwlN+2=r`KHeyhV)ua4uPo2CfEtI`=GD|urH`mE*X`t+<lR1*=`&OI#*&v'
    '7AWgcLQ92U<!$R;@~o_UCEa#%d`FdOBt3C$yHm7D)+^Qbk;!<9M{G^bsuvur%d7juEV1z*8+wT(qW8Qy4B9+zZ;%w^AmLQ!LN&}<'
    'sv?#fV79v-1W<n-GS5o$!vHys(_X|NF&eB(<Di+Cu^tEfhK_|R*pMtL&SEIdJY6w&LbQ#nTb2I^Ff5zof6g{MZ%!Lx<y>Q2#ia$-'
    'ohfqSc*Wg6){?TVcXCrB#()P+$uZo@@v@^R&|d_2ghjlAhb@Q5VQRU34KS4GG|kl9T`qG0o3A?8%yr*(9MM+YbAj*0LL<yNF)BRo'
    'Vmlf>gN?F$P}7ILpym{*o!!qc=bCLFGPG2ySZFD5))XUh?6SwUBMR<xc>2uH|-'
    '%{4=!AL)c@^^4G<X+R2hRl|4xMmtY0{FNb=nJ@&Wk}>#KrILN%g4X$t+(RNv)9oatK92kNSS(kZF|kN2f1EL~SWJH${_&|t>3fmc'
    'F%p=*kJC#8ryoRS=SZ+QFEYDE0?zr7*)0-uF2MN#=T`_w1<aIUb9Qh`6Pi-omDPyUS8cMBdf0+L_>UO^&yaX0C`D`crD(7EQnc56'
    'DcUSwiuSrMMSH`SqRsZDXyP~Tj&+mLx3nh~k4mS~H(34RDp5$0eHfWt!iWd*Ra;q}iNm_Gg?ZJ!@_n%OT$@;%&?d842Ky+l!ixqW'
    'l&~I1$VXZ2`6$=%e3achA7yvXN4c)&qg>DPQLgX#C^uj-yWSd$@m;3wCrE|BDc`M;B7#@yun1GbDFUUlaDp2vo0Td>e$!k&(TWVS'
    'a?w`%S|;%NrIZQ0eyL;vuU`t8!0VSfCh+>Dj0wDcsbT`JpNp8#?M~if_YGN<*n9f`&M4~xb(&AK3oH0OqjtY<=c<sS1gUP30>>Ew'
    '?Nl_<DiN!>5sw3FrBi7@5X1c5Ttw@YtTLCCZx>vLYH0>=BsDz{BVf%>(yV^Ny4Y{%S2Qpc9Qv23LUgS0p!x^!pzJgmRyNmQlNvkB'
    '*J2+d$3m{d3^m6>Zp4-mE>pP)8v?Cg-'
    '9!_qB<*^$GJlui;oFpXTdIQpsLZ><o7hWZ56B*_A6X#>1832<CX%l~QFf5%Ldv&|Q)S^xMV7hL4pfJ|d#%AJIofhwF;%<V9ZD5PV'
    'WoQ8lON2*RzNGvB`U1A`-3uXN<s83%KSqLqHk5^Eh&h;U72?j^G8-uO2@ZreHONBeIB-'
    'JeG#^6eHpfEb%yO)Uxn>jUn_}XTEune(V6h~gTB)3Az$hCDA%+#0cOos?sYqHMaOvumb#?bfu-Jcc3`PxnjKhbWM>DK`ls1}rT%G'
    'lV5!CZ^blUvK~vtRWO2yW@6WvIHvZe_N<GLD2~%bZ!)7qAW%?_#Kvjr%!xm``w@7QaMQY^tgDVMIU9@luk}kPHLU@$n&I^Z0w9rT'
    'gw`zBh#NDdhO%ivjb`MG1t=hdLakpysk;L7qO(2Q8Rhvi>cf_P`X|So-'
    '6^D&+U(Ds3urVgYj;;hW%%7~WnLbi+MJf<O;0e{r>(LSBmO4^(zk-<>>Ohne9O=Nolgm^)PQvDw>`bZUV7*A-'
    'Y!xh;X3B#%HJI2;=w~9VDA`p>XGV)mIt7jOcP0GXpr!tygp(UI)4R&?aMvfxF$BTp_X><Mu#vqA!wRQIQa}}<Taj%<=pST8>S@=s'
    '`!xzCq0|Wfe+|~ir6%{e!KnW9rY2N(e7}dCo9_I69~(E_0saB@ZMqZuLu}i0NBBJK+H`05d~Diuhxh{Q*>uPAGO%JKO8EmAhb}PL'
    'gyRpN6;e~s=kF+DFAp?FTc#ApCKttr>W?S3i^QT0H@C#C>(%&ggB?2CgPau}gv}~5Yz*wWW{pU14>ty=%ta+L)TNDUHCMehp}V=Z'
    'J;a-EQsXosO1uRr@5RrD<7Q4Iv_7Cf?~N@^?09-'
    '~h@?!ugswODfXZ2f&7RnN;EpVCQGOLvZHdNWC%h9+_WnXE3h06r8Q`NFqM$xiK1U0#XbJyAYv?w?;kdXY^GJzc0(Opli~r)5W*Y`'
    'OPPDW{VULNHmMCnd(b5uy9Vc2^$gtx?ONG)Okdod7OR(7$n=jnK1zzuQQo99i?g?ZOri0PQ0?LuF9kJ4L9VRseGrSvfn!=gWoheP'
    '>&{>xmO~Ddhj|ok|6knhDOyM-zK(cqrG|A9GI8KZV9fafLP#HQ1$H`$j&5qXUJRS_U%`F8e6-'
    'vz}JTC_5;LI7Ir4Mlw))TTV;5TeDMv%&fXd(y(XzwM4+5$I5Gbcu*Q@y_m1C*mAKpD1~j$$ZLu2Y#`$dpDne-Du<jd1?DvRZgEY0'
    '$siH|SsC8}zUA4f<F42K}pjgZ^mWpg+bo=x52`;Hy?gxB_K|M9|pKh-'
    ')6Wg3b79_O7wg8RCs8>O^uNzefAE>}$1e%f3$gw(RS*Z_B<x`?l;GwQtM5Npo9vy`{AQSH}JeIJR$#>cFNe%|&|Vakxvt@}6I2iV'
    'qM|92dER^M|1IZDSDjin1QWrD#qOsWxY&+bSvoBSn8N37LX<vm|T^<{u@YQ!sCp{KV?FOG2n%-'
    'YNOX;_sHY5c3W=P)lxjaTB_#Zz<M+PiVBXFH+p??wq~wz#gmda4NG8QvSXDlG21wMMYi1{1Uk`9{8GL8DfC1q3LQaVh0<*W0YZ=y'
    'IJ6{W8Zh9uF-'
    '1!fTx1Pp8e2Ga7SK=c|0E+cI|vnBW%wu05xKN<zw1xp#VIN>L)fQN9GJIeNk&+OpIIlvF&`HY}rtaDyJDsvHZ3B0XJiHAa>gG0*8'
    'iR<WfCOnH%eRd;b4+=D)$ZJjT|4@UUkhyGdlLjp7lp8a=Md(h)0o%PNYHw*L=9aV*-'
    'BjXX33zBDspVfVDe^gD2Sj1|&Y@JrJ{d?c~aR#74%Knq;XhyX2c1tS8qz?F;$&;nO6B0vjV&4>UkFq#noT40PMp_Z1+FSVqdJ}>b'
    'AZ_6BNe&(*8*5Cxe0h7kW0-A?hfpRWYgw+}rVZoPd4e1$hIoiFh^lh0`QIXORopv+v<XTt_8C(miA#-'
    'bCHDqiptcFajh1HOuwXhm8vldo^{fEtUvKH&gZ_$!=BxB}DZMIZ!@;XBrN7zs*AdQQb|Ls4(hQ{wSv#f!HMn|)mi~)*pfRC-EZBS'
    'XCA9o-'
    'KAk0xNP<}GsLggp(eWd(kzK@lk%=d}%llc}YKbdc_@{{?ND4xt$Z%OUUmQI28(TUiQuf;gm{Y>{SM@QzwU{tg|@zxP>@(=T$>M=o'
    '8Zq|BkjhRfY6@|aR&&LRBzD5y2f55HYr5wLOn7?gLCb{llMiMHX4z>a==Mt$a;OBj|0<PpRz*X?`AzJ}gb9vNg_?gdEz!(k=^wmj'
    ';PulO$+hQLAd%3l67CePz^<b>IB{_3oTvY{*#qRL`U}1*nF+m<nOla~J#+ok8?9M~rOH`yR$1Fv_6jLpwa2`76d275A%Tu=y7yLJ'
    'PzB77$iw7aY=X~7tjGy1(S6~4B9@POO=>Jena`_NZRGKZ^_r|twEj$i4!<tCj9M<n#j8*4lDy3CdC{wgeY}FF26vO)cz_9)xFswf'
    'e4C{G;VLd-EtQUC08bbn~mCT9mRf%8IY^B8s-c$Jrs%CaX={{evh#_j3mW+@e;@O9Xup`(`$uL)5PL0?+AEv)}g_)%|53K`-au-h'
    'qz4WRrvx5CLE7K6!nGE%%9_U>9EFZi_m7{3Iyw*@o;NY3p9Ax4#G{LrL=#>uUOj^(|Q_YYCX*d#I6o%vAMOiosUKE9+HAYD|GF}v'
    'f<Kjg*FcojDw&5vfr;_>kk>p*`Q2SU?*$AseVYnR(**EO+z`;Jd!dRo$WT-'
    'V(Lse9n(ySAFn_3y$2jZ9{{lwMcg4bghqrZ05)vSv96GMi(c?;w}VEg?iG~?PC@IOV;2e#n1q=DDYg#Q`xBCuxJil$$?gK2A;e*F'
    'q3rSa1jN}eA-'
    '<P%9t<AaMOFO5$w3D*kjpmw`f;78BVWt<7Pm7IXb`i9|l%dyUmWNoT#4>l7Eu0M7MR;OyH$HNPh%<<mV+(+oVAC1i*;X|4gn^Plm'
    'mb>qb7~6!-aaT7Y9hf8luRza45`b5uj+K}PrGL#ZEnO%VVC#X@GG2rY98%19Db8jN*5;quyk^%19EI1*2*Ahso-'
    ';%cT#MDe08uwgiobbcA7$b>b48`-1pnb&<kq2?z~US2s2h0M{-VqmC3BLNcuj+_oZKIq-G!+0f!ORJ^r;WUW-'
    'p;neK<CI3v1_`*qk1jkvv8u!g5Dmv&+#jkSd2O(I*mxJqB%IDQ%cWm36~2>bn`kf-S_ynzjhXQbD66gy3Q<tH>(B6)jA)_G${0me'
    'W!c{z_-'
    'm$>13q7(CYm2G6yD!E;?;@LV4lJU0Xe&y9h>a}!UYYYie;SE2nUMKq()7?zrkDO9F<N@gu;5es68STV!exFLP*Dkx=S$l)r9=`$&'
    '&)8DN|qh*crXZ~Eel*riv3=sz8H$x*we#5&6RuI<NH6a5GAJ^P@3|9Zsn%jzDr#+)>Uoq}k{7|>TplcSFc!@zRFDdaFgI4~u#ET3'
    '{XlaR888q}~Omu<@>GKk=Gw7o)O2J6=O|T(*nOz%r_y1=4kYQuMzqiG9=3$u;^>@>%LhEuY`~>vra1}1p0ws$IpV;3Ul(?cbF3xI'
    'C#<DJ=#U*f{xqti9l6gXN(>AzLPipSk23z7O&28IrX<*}AChx$?8TemE-hq`f@V}0{11o3Xe;s)T7Kp(AI`R%I5P|=7<Q-'
    'Tb0{`pCJFq|m{x{HI_h|Ekt#X;D%`@?sGMr*HZHrE?W*sf3`JiNG!g|_^>y;U&!?Ge=Pglp%iCxr-'
    '!f{%21M0@Plv5Eo(8~Lv7Guj0jkF`+w(tZ4%tH&62DmCbvOt5YBK#xZs_?)94X%nCi~NAMX~9DllNLN=v1q|V7K0W%WU*(#L*6=F'
    'NP3?(PuZ26pv}{nS~T!1ka(f4Ecr@#v82((!WvxwNviuSd}s2Rfr-Fj!n8A2v_P^*$ge{AVeh)icsa<^0?wWLruU19-'
    ';{Kk$uU<9d}XG{WT>7Rbmoq1XMxV#NebF-<b(Mxl^iK4q<d9zjEr@?PbJ67Sl9lTeP>OvrA8#|6)fRh+B{~L@NR7$$0bZ}xH%=-'
    'WA@?=doeu4(bp1GK-?8nK-?WvK-?2lK-?QtK-?EpKuicKASR}0T(}vx9tRd3;HmxZR>@IrRBIDXnR-m82AQMqK-OJasD?<i4(?5M'
    '5S-'
    '9chFoSuLD0seN;qaOS(KurY=K3!t9CD}Xa>S<q{qNSYk@70Ud?M%u$nihU^Ru%k3v{2@)lrJ)HsIdAwEKdinxv^O`!<tWOg5x6{o'
    '^=omR!Mi%A-ekxVL$y-'
    'Lz(ZiZ{}t(g{c*txfclX_>VSiPQ&mwp;f<bhoN4DZLYi2tUN!zI@VSUW;y5#OVdqiLY7Eai5{fl5ocJ(9*|4;T18Ly_*OY$+cPmv'
    'U2Gwv=|fDYg|36<*k-'
    ';(?K3@Hx+KFeM`0`|SvNu9U`q_^ZmQLpm8)2a@t<#YeiIH6N*d_P~eGRGA_6;_wLV!^5Rc6YRqyq)Zch+9RbH5sby7q!tm3#iOMZ'
    '5sdU>q!JN~^kai%twS4!mUUZX<xMZ^U`?~Hte=}+*W}}1Syz<GmlbqdE0#5!=Lw`6P@A?X8G0=>h`8qMDn-FQKq|)UoWX_)G3=CO'
    'rjh+)q5Ue`AoD<IfMbkOI(BwPo_X%>-2-'
    '{%x%YWb<e9&Rnevcl{$6IvL!No=0p16B<|i<19`ejjbO8ItOpRG*eP?FG?B&&Jv6gW{!4t%VkkOz|gd26U3L6z{=}8z&a?#dFxHH'
    '{Z+yj}%&cm6TE=-'
    'T#SeCPa>d*&x>Oc%hr;6U`WNR}W;d0g^X*4b84oDguvc>)i`^=YU$=Z$*6_|W&o{c$#HiNo#PIvb`7i018j#^6a9+wiltu)%N`-'
    'n2@ubc@-'
    'tS;ambMkl_MFmw_SY5zd=IHnKSmr~~@65tJ9WLztQLwO58!$H>og%LUQ@c(}QDZZUZYch}K}tT^b%Et>6Pmp`O~Ir9Vqj(;LF))s'
    '^-<xjd_6X&Mdobxi)P2jpK-'
    '3cIxE7Cww>LoehFqSpPm0S)431^qn9_ZSNKwl3I1K(`u<F#`MS51I6fmKj`a#d80^Dur!Z2%QL`Nsm8g}k;dS8}*60ey`&AgQb6b'
    'B52Ikz>e=9b3O0Cx;)UVFI|0Gn7;`={_XN>r&PoS<8ul7k)mZ7&+D^)hI$@<Ey&^9^<IbtwfVTj__Iy~#~pCdmFbS#=m!?=nbHHu'
    'rm4Ux=`PtlJ{h%f0uKz2O%Z)uG*3ARH9L3bnT5J~<%WE?89nI;8`n!H<KO|Qg&r~}ifk$EuY(-'
    'pSOLopw(!z1%>%)8+9$UGVgg^xA|!aA}O6Qhmlvb=A{=1)>}_fBm7ELC^!#^x`I`QRQ!D?q{t$KW0&&ue%<+v)t_!8ZPp%CrS*BJ'
    '(1Mb{*6g`^dQN662f%`e}kYP$y7-'
    'ns=8qUW)V4ZoVryvpT)1&xETww}R=kJrOs$@`Xv5w`h>xfLTAz=)p~syrnl^;S0^>gdhf|TH@4DGb&7V{`<+1xj$yh{FKN%5c7To'
    'j6V`v+q?-sagq!<ok%?EF3Q+Sg!C@Tct4S@yC~xW#7^&`j1Lk6yNfbDL=5aM%J?wx*1r;zeO_yxw`HF<n3q)9W;{3w5Pncpwr-uU'
    'RND(ytZXLn6e^J71a26EmGVx~eC7LQKQFmv9Dk6UfYc_RFh3homJ7<xjj-92AwM$0Zc`3j{XND?OIK$4IGUqk>pz8-l-'
    'zhY*fb@xk^2Qlp)Lj+(Ah#5*^p+b;W)ig^v&rwu`*o-'
    '{$0uZoxq%2Z(b~4zK=rxr!lH%4^0SM>2~a_@^lAa3RQ8bO*J0g63(vi8Tb0+Ei?}Qnp(o+kt^n7_buQcZNx$?cXn#CKPXjq1?Twz'
    'uvG5q%m}-DS@*sfV>6#C)%{0o?vPU5VZ?&NLOBITfyz^-;rLIfwEClr9zc)Fd{v?-e`sdRjA0RG#;I13<VN$d-'
    'IX_)SIU>QBRqHJsWKgjiNQ(qfi^Svn0ZxmYVdLMn&#Z#6K0m?<lvL$b<Nqqr_39g(}Pc&*_!i%&zL#x?<XR|&SJN|XGGX2%_EBMl'
    'tBtTK+%j<Wxa7U557zTP9T3$!ddO)(KyQu?+>bMV6obQP18Y_k|umwC2wUa;m1<g2mW%6d9G|VlZC-'
    'z9#CPA;+AX<2VsKA*(g)cac45hhhS>cS3=>fHDYYSTX%%2F&T=5XpYJ*IS#v&K5`c)AxF`tetPx3aNf8bIt0&HIEO`{zUK!Zo{Cd'
    'w**e0lD-~6Ll7SW&-'
    '!R{nT>hj9^O(=Lg47<#n;@`MOXz<~=4}h_>&y$~EB8@S%L~_|+%#`WXz84u>xh#@Qgb&G&pK(KZRTWRBkIy%sT*idq$<cTSYy5^$'
    ';5#noRh3i4u>CCTXh8dxYDX4;m1{09R)wGu<B^|adlP4z>h1dIu?FhWisr)rxmjf&M75t(!$`y5@lv*raQvveS{rD@{$5?vJ9^)O'
    'J;n`EA>8Xg|gGP0ngE-(QPRV-wp`Apgon*WcO7zlkge7sTp53Rc-'
    'Z4B*a?e%(lPC+!8uZpOf<x&c>JdY)qzNHK#j4U!!&{=ZMVGX0r4=yiO(CE)M<%)fYQ^dN!4*yEyn9D%u@nm+TdVk+9T~{i|f&v{v'
    'A<l6kk>3Ix4zUNlB(3Cho1DzqBLYt)#0X}`oUsex*weI;6AD8kmOml)QpJ(b)1vkZE{=%DE=Np|)gBw=^?M>0EJ2+fWcL$l*0Han'
    '8ZVB2yq?^&#weSrhxpn}^ZE)F_)vV_G!2~Sb1IhSPGv;R{vZ`l=_q0HfBn+%>9Dhu`|b%Lh1*ia%aYLQsNBa#9hOs1b(v|+e5a?Z'
    'Bkm8{w$6x<yfqVY(~o|sLXJ~|pRCg7$^8~+z2PMjv_e-CxF`1>Z8o8#Qy_e-'
    'uY$GgAp561lo?(YYH^Ylyi_XCrw&57>s2PLD;uiW1c2B+$z$ZoQim1r+-Yon$fY-fdt1M*MsTkJx<pv=+bXq{b3hA>!{;ArTKTEk'
    '^%k65pLSVGOtq)Clluhqli`G^&6Rwl`5#<kDnU+T2k4;(+I<+s!3U@+O-'
    ')e)GkG8=0#mFKew*n~F8EZ$lz%6wUJS#z?YXNnaKhV%1mI49GvRQn~)u)wK4O$kn4QKIMuYFN|s9{hL7{KGC?SINAiE5*APilTc('
    'B}22MCfjOr5O}R6Zkh6S@L#D+Z@v;c{S_A{YD%VHsLQ?kbSy1@nFG?G1)r8Q5|}N|D((}#oa-uQft|A_V=DM|@YUc8R;M^7s4bie'
    'vByf2Jh2xw%E((JX=Qo0c-'
    '9(9<~C%f!A(bFhgR%4Wsa<b+IGoE!7B|ZD_0=<ron{^zbtv0Tztc^u!O?g#qpTik*TVI>+LZ;`Oi|g)j)J$qcZu;@)GW-'
    '*@~Mm7nfLTt&hIR%Ut@(As#R)uHZt|*THU-ZriJUx9w=(Z9B%F0S=ycF{pfNJ+I7BcA;KW=9qF+g6KrQVe3F+d0oeisMI*fk+cfS'
    'Mh(J(jdfTNfGk^VW3eMaLV_Qb;8Uij28jF%r{G9^6WBvwWX%$d)*|b_Z*mvd`mE9JW_lFRX!kHV3TU)@nHmK&+I>un0vc@s)1rVz'
    'o9IYU%J=Iaw_h_m2$zC4P%^fRN**a7W_s-;2O@mgMo--'
    '{mg9D~m71lc1@FpeD$oI7b}|nYoJ!#SyvBA22XxWvxW;!ST<g0MuJc_9*ZZ!78+=#7jlL_v<wr7`XH`iyPj-tWRlF2~Fz=8(0;l6'
    'hU8an0Ml?BNsw%<t5bmjF8|<JB(IO?zw$8;b1Q;&hA2aRN_a*d?nRg35n)@@rl8^=s@c*Pk18fRu(Ey89dNjZ~mnIEBu5@W&WXH`'
    '>EZ-WdN)@*TeB(g5>{jO4z-v}{a~p6G4i0@K92|-'
    'jbg&v?7>vcMpotPbsXZ)mZ$new#b#$<3(zZEB>>*&Gm`8F_j0OJJ=K_SpUM5-'
    'OG>`~`_q!||6W@1{okK)jcT$17G1pJjP;y*#cU;BAKdk?3bS=;*zg7=FiH&X7IaMg$nz?0keL*=`P7uR`Cxn*+<y)e{FIDJ4(x78'
    'P7Q}^enu`Dhh4T-7-'
    'kvhr$K?7UQyESGQT8X?5)g#lCZ+{RTzI^J>O4Z&}X(=7w|~t_zkq8wE7K{{JlJAn9!`gf0d;8sf7pLQu&0qd#^}KgkO!G%FHW?V('
    'kN-'
    'a7ozNHdpB6GP3?vf@7ZfMgCoa{nJB~57^Qhif#s5nujHHmB5x}7LsKpFby`5^A7ghrUF{9el}CQad06j?3FozUiqLrsn>+xR&)vO'
    'tUA|G%-x}$uN)05*2Z)3n`Ks|zY5Q)EyxXF_kJr3D$D4NcKk&$&jSs58R&r-'
    'C7I}f&5knC1ABmErU!Nb%1{r?lE_pK?2sAg5j3YFf>xINs=!S(W!koeQdV^`r84OH?oYJ^17u<W<=<Rl6X46YH6WCIY!u$+B~f*m'
    'gf>WSuIRcs%0|snAu;L>=W)rDKXZeI5-P=YbB2-'
    '#k?F<^B{2fiO&Ll`h)XwQC|L`|;bsgaYa>Scd1$?)0>hGe%ABvBdFhueBe}hJ3D;^&s*=8rSJ$sJta`OeCtM$&`MYz@yVW#og)O|'
    'pP1Tn8O-)0$q%jp*h%oxeYuy-OhnwP@dQvMnN}+*mOlp10N)PlKTP#NoZ_^%_qX98=xQ`heVVS}JX?W@~$D+M@y}I~ToQbd*{(<W'
    '=4{N|-)7oqr!5{e#10u5t{*nLCKQbHRA1Om#sO<;Ze}<PKKSln-Ot9!cDZ2gh^I+C%y$gYnkvU=$Eab&<I$7O;E;xgm#a_TE{$ZN'
    'T<N_iyq3^vxqU)~jcJ|1TXZ`NHlSYmF&8_y!)<zgLKSrbXdz40x7#z04HQ4g!-->NA=eysORMlO-'
    'u6z2o=a2l&@LTbB!_)9}K=|N{f{XB|Lf_{>W$ss*r&ZmVr#0ADglFIzzmIWi^AB*uN~jh;BJa^X0`D5tUEkgPwo^x*cL9l_mZY%}'
    'j^)J3eE-'
    'F2`{bGAiSW6&?#wfsplVT2W<wl0iu3LMgZ0D7)5+uE?Im~n0Qz*0WDlGGio@~#jaB{0Q^{lD?Q6N)`_t_k6SLzYEcM2M<ZpEHB&J'
    'XN?Q7Sac{1HjyP!?|kCj&ZOtt$JE%d;TFRnbWj(cDxJ@BI|OZQ`)iW%h~=89jdlexsRQVZv;&t#*K1|S*hK(en&mWA8+=UHI9ngz'
    'y=Szv6D0>&2Fe3SylU0Gl}mjXszFc5j0VDRy_E?(~oA!#l^wPZTZV>(p$ueI!C!hfvg@aU*08Rkk8dYJR@qBfN)r|tVrfU>X3P<C'
    'O6GUbF$OV%?#x(FZei+UD?6ueK*Jdo0Eqk!ys=0H;9Zhw_%u~&;E2HuYJ>8nK&18)nueYHqh%Pp$iHj{H9X>Io_?xF{BAc>*(96g'
    'W&No^iU2qZ9q#{?xOEi%f>()aMWg`1COe9@#3X-'
    'BRF^ZY!w6UX9Ctct?Ivp;GxGUAG|2er9HCmT6(UzIEm!Bn4x=WAJb?v#b+Pg3ywi8ddn;CXizp3kS?xqUEad<Pe=&p_zY079RaLF'
    'l6NL1m>)U#nM92z`o=6oU}FPp^EZlHDP&zP$<@Mz0YOckaVzC;If<htW^y_S}b&744Klhtb{iK!L;Pd3qoRgt{{J-'
    'Z*dJ5p7akkdmy$|1iu26YEfgi^n<)ucUBpe1vvi+UWa;yu?Wkb~kHthMS~GgesrQ=t*!ZzSP`8yT3}l4^gm97FV;fxY{|3tDmND^'
    ';3<EZNVqrlf~5wf~(=8SH3Hi*?k?M-M6p|P8Wx8ik;-'
    '?+w?97C;Uyli{fb^J+pc|@s5Xe?VSTpyyIsPo&9R@#0S96^y#a`6L0pP((S9o(>iWdtw^8Eg{N-'
    'rSKLDn<iJyR_rMGEKn^@LbWh`{&W{1d`D00&j~BNh8F~$q(0MSyUu%Ql7<jjJsi5)!D)}MY!rNv+@p={%yJSJJWeOBqYO^Q>ihHx'
    '5criF^>XSCiH@SkqWHOe-xDLW(3Rc9N;x3-WTDeopY&rAy5IRfCpmRwGohm0}`p$idLI-'
    '|RpQ6xNO3$nwI?C+~Sp7aZpu@)B>ohcDwb0>%ZWsFW)k234x-'
    'IGU)k3G_R@DlAYA)z7_V1+!azKaC{~|q*$1h4jr_oNT_wnVyIBL|>gdu}OEgwbjkC=XQX_?v}YSsz=)wZ5`03E2372$T@E(@SHvH'
    ';pO3!tB+0Q56$7N-'
    'DoUlu?wr2sS)GnuD3m`uZb?QnN79mB)Fb{8|SO8oS40KKOI0R5{BK$lhm5Pj#qMF9lAsc%sL!3*^*0w7rZz6AjE22Cx>4WM1=({l'
    'spXLNgR09nCLD+HkX=z#(NdWjy$;|_ICxkJe(W26nPNAd}pU)MSF>sD>fc0;(SllmN$Y~qHIsAOfhlef=eV|ErByJfMlRSFwhX|p'
    '7SjR{$7yc}So2j;SdnVUi)39Mjk49M4!@za-owZUOI)6qgx>kYG!gGt)vwZ{krihza>mgDKh5Kq^tES@e4@f119(>iTl6i@J*HZO'
    '{&Yw4NQ<B7NM_B2~`wRmFFZ#K>TS}mS<_wGiYzFIu--rb6BUoD=h-KyHnGCdcb*wvXp59GiT8-Fj;19^O-'
    'j4ju!#5cl(qYO+>y0BbQ?79&Owq%>Vs}{GROZ~OcWa7{#`VM`f<P*DlH_>=8*8=o*ZO)BsCtJN~im=Dj<#!KO$*K@eJ7fVmCkxPl'
    'S%7Yx0_fJ-'
    'e3}C2#4JEx@h(vsIFE(Ec`5`>BL)Jt8Ewm^KpC}D7kimUL|>*~Q53<i?N<~<Pth~^J)#|GCU0(!Xbw$4%k2>jq)*T75p7Mk=k|!K'
    'L}wK8h$hkl1w5ix=z%;Qkxr_8Cani{&cjLcM#=8_8|dmL7slF@Mq#<8wH;`9(Ps8vPSrxwk24yWhCk_zTrsXDq#ts?3=~KI36<O<'
    '4$hM*xmOOPr&Tf`7=&_cs+(z{3Yl1k>VT*XKtCcVaiCe+?jO&bf=9&&tg*?YI|&@iLZ9cqWhUFQ@QCT|#At`K=+#Ow4G#EC1N3SI'
    'nFhyy>6sj|Y;>|@3uED!fRFbA<>~S-'
    'QcRIwZ!X8CM`<_Qp5{|+Y9)D^mWc~Ek;9E#z(`ZGG*|XT&YxX*2GM9H$4OQIT6Pr`_KZp<I!+sv+@BIDr5l8jzcyrC2KmxuI9qI~'
    'BqK*C61a;8YrJ<vx%KdADB8|4(JrqfT6)JFi;5P0(~d<&+ey#l5bfX$Os^qeUhlsB`xL#q!IhRR^!eK+tleKn^W5XBu%lesM~5iP'
    'an^9{<yTB_pY%lTryY4x(pBzo>jAb77=K(5GDRg1*mDW~<vL}4u&?97a)`e7Q}KR0kwHe1NUrAJJPCJG0~Zn`lVt1x;n$gnG*x7*'
    't|Z|Fp@f%|N%;Fv!rh#N>E(ASDq;9dI~A4i5_%?wgok7#{4`1U8W}`|SMNKu{0v?t3+-'
    'B7jFgCu6MQq8PdG@Lp<Y!<4{3~?r8psShcailf58ge&xNUcDstBEC2sRmD!I>*PpRa=v?flvLVn~9A4j8to}^K#Br$k4iF1ryTnh'
    'q9WGGo^nLNnbUYN;0va<Z45?blib}lGO_&qxp6eYaQ&bcHxG$Y9;NOoMyycAJjV<R;8{5X~Ll(FZ>tE87qJwHJuy=Cb6FICb<W}c'
    's@lD;zX{8uX3Tqd5Mq>_Gik35B7z1D4?C-LF~W-(OBLn&DzI7ca$TWdlOu8Zj$&%^dQ&)`@a9=?q#HL6<4z<EobiF^-j-'
    'OEDr^Fz44T?W?`l|)Q$yh~BI!f)E8C|uvBXL1O6SVqW?laQYXRWMfv+@X_Yj=Q+YY)rF79mK|Hv|djVo_GL7qm=uB@wC5G(19L4K'
    'uZ7M&^=J9>EP5oNXh`g$U2zG&Q$WSld@}E9;eYF0Ksesg6S5H5TnTmp2j>ds%UFy>n{;Yr;Lv6Oj=~DZ)4JipJ9uM(|BGINP9t<^'
    'S-i@wCT-vEehx*_((BfFF^8uJi;E~4J)h(nUuUs>$r(Bqq>JhA@6#%oJ>OndWjS5cssGjkrvx|G<H8cj)ea_I_D7$t|ePvr)Zvng'
    '$0#xDDNwEp-'
    'C^ZpOTRf>uS&r7Uz#4&TsT}@{eS6c8cV<HiT~*wthFd79y<m<YEv&c^emlv|#+E!Wo<_4y61+nUq&mk}|#BZbhZM7#}Gn<qs6~1*'
    '~4m=Vc(h8M^!cZ2R4*EG~JM1HoG%|DvN|wF97BxX#WPa+G^WBO>(FjmQUMT9iy2-'
    'b0zYmCTd^c)Ett`?a}Q^TEAJ;pFs_+<h}t@@Q6f;|vnbB@z;LuMMkHvBaXq>6RW?GKjtWUN^+ZeKe{YZR*XG>FPcvYh=w0wf2)`v'
    '4o#j5HP*!z@kbUe$l|95`L1}ee&cmG7>%@LJ1#a2VR@ARq_M7J0n%HitVVGm`-!1Zaqab9ZDuAJ(Ig!YAxxN-'
    '0c!;NpI)bCR39>&aX|TC4HSOl}t}IcMfhcBk6~ZkuT4x!n1b(8%`Zeqg)HW*7xnfh2XXWeA;kcR)?I|83`^ACHSAprI6gOv!60Qv'
    '8`bHEAZ8^a5WQPwQJH96_hT#L|0T$y8oePa!9v1CEX_U9=h1E0CRU4^FW@dlI5&epTq2^$hM0)&FkTA_Lkvho-LPrFMcy@l@GbaW'
    'koWNUEr0;eD;7>B@0sHwSQFk+S)tUKiC#Zy2DGl3EL9MP0q;m%TTV1D;M9fTn|uYODord6}aUU<Z9PrSy8#dODroY*TwWq4!Mrap'
    'mu*k?E#TYg6(o+hKa4=@dylQyJDkbGa=%p<(e2t&fIp`IL0!Ty_g*fIb?5+m;?)XQ*U)w&tt+|+}!=J>g+)GPe&;;HF6|-'
    '_2yV*W<*3S7*cEx4K#2Yj=dV8%JQRRILlp#^bYSWwMS(F|11>vyvn6wEO44VNdljYgJ%MPb4=fU!R18-'
    '4ll91sKDpZGdTo4E+g<p68QdZ-T#5pDIH0-'
    'xf4mX%mdDIxhgjI%IKppv6&e0{EQy4`Ao9{*eEt%Xbx9y7MridWqtumB1GJ)Fxj=AYeC)@WzT)OM=JAdL==-'
    '=+ARUcMI%M0d~Zmza^5cf)3^wvEac_Y-lRsYz&=!3=8FQECyOeUGqK30;;bbS`7|7~R6*qSOMYKe<nRvP7Zf?X>i4-'
    'MKHf>(4;1K-euWe_OsF359|HZ@7~{&i?vnP%0?d~coF18HB5;u%msx}oauw5)BRn5mU%`|J4+vN820}hd$j$-'
    'nXA9Z+v<TC(j)rJ@WX=_`b5H{#h3x#O7;JM;D2`1k25;Src2_TAm4P+p($c;uOx(OzxdeyBJq2e-'
    'k+`ShD6K%;Y=_xj^FvW_!%O^76yfl$Kje~kOG@4p%kAL7Ox51N0%a&%EFBn9bn^<~KZAe;LM>)U`Q+*t1BlJciWHmUP|LfX``8?h'
    'l=cqGu{i-'
    'b?H$WOY)(WTd#BZ7u)NI}cl8ohg0QiG*83kKvtP_aisqKc>>tY^a%*G`h?zan+#Z>OVx~_te~iq*ahce^2^ITBx?<r7i+x{Zwy<J'
    '9P=TWvh@ES}?aQtxDt35@6-'
    'C9q5&fPVVsA}}oit&!F9K+cy<Y4j72sxu*Mc;ehN~^|&G7A!ZHD(!e;LbjL_*+gjf>!8<(R)YGW*6%Z)t9e%z?3^i3LT%M7?0vGf'
    'yv=^-R+XW<9g?f?3Zby<pZeM=zN5OwkKweOZAw((8HYzF(?X#=z2T#6goJ-Ggi^>+o#Bko}@7i%NGxdZw6k;aykelJ2C`v0aa|9a'
    'ktjH=`3r<DThgVTnfT$;f=^ba8WxrI*e!yF4~`$-u%ZVsp35a@;V+I%IdTNo+cW>=910{e^<_6)eh<E%3QQh5lM3p;OS?Z$#m--G'
    '-H(Ja{>&mNUfF4K%U~)=Jda)@bXrr_wMVENLl|_q!pQzb$1w<|GjJ4Y8SL#eGw37KGyFoU#2<tBQ&n-'
    'eFZyaeqtC<jEYbn=S}M4sfXe#FcE8me3KDyJ?PjXd-'
    '5vQ!!;JVsD}Pj1bbz5hkjF0*JwJrlg`qWlAbQ_oquLYE@WLF_s241fK}5LAs;@ObK5Ph%<ekWpcQ$6@YsI^PZZzn9nENH!bk8W&7'
    'Zt;Xe4f4EK{m+`ki<*_J0GLD#zi?(J%f(gkZW;3Y=sf;Aam(=+)i1wk%wf9^`b8u}W!D+Pzpr{}H|tff!ST`6d{#Cf_%rC<&54;Q'
    'EuY@`SBgsIjot5R>~u5*mK2TdPs(zOT2Wi8LG{1Kl*Fm-uk4o_j~7g<d8$YSd7ET(#<Fm(;4nZr-)k;T-$DNKE-'
    ';1vCkwK5!|^CV~abDR$&)d<_;#GNuweG-'
    'D|<1$c92|@KXYJ37!6GlKPfXXh%1w}yxFL6OpP<>3#<OdZ9=7IvC>Oo&4H>eJ$PtOgip7iOtL1p*E|0@J4*f#&K5UBR02lABPra;'
    'w+P0Bp!v=iHv>o}K%R9h{<bg|l?s;CL45X_8ZPIlJGydpA3q=0ou7FZi*fptU{SiMrfx>lQRDPV1w1=fBkU|ks;R#ykW8m+)UozG'
    'K|>>-v(;=C2a7~af6>e&!d??z^hh14M!EUo}4yAT%^gcQ8Og#{4>Z+c-'
    '7K!HFmEC8qt=}Y7W)DiUQxdGLSK0P;}>|XdyAwa=)`As1}?MDyfslQFzf42HgqA_(54x(z{Q>w+EQY~EI16Cp6j>m>#UZXlonQ~='
    '>p|)W6Cs{mgl*QALSv>Vl;psYUx~K58Q5H}8r|{IQBp{1bQIna2`3eQj;rl#i%N}MKJbe<JQ6H9_QBy-'
    'e{Q^CD0+d0Iz5<}^N?cSFP@mu<#f~U=)r*QC3Pf^I0Yq&?-y%1nj-'
    '*e|ji}!A>A4YQcf)TBAqqCkZwn!6e|jK~N3~wsqnb+|l|ZV2s90wDC>bXQag$0=w3&=Wi9-'
    '*F;JE}#JVIPu6=C`!KtWR$R~u(>byOBteNwo(9_w1dPuw_*s{>NF>aW3)6)q&5+$B^<KheqEo=Hn5_YlQV#sJ(TjHb*lt2mezg6m'
    'KWqY$_b!^BqwaM|U!xTpcR35|m-hOYVad-'
    '9_T1aol#bZtx@pBr6A(WmD|S0DQH+~~4<;`~DBf^BntA#@!;59INxGUy6}kM(9m41a6^26f1fw;l2hYA%r6M9$PqcErX-'
    '*{Ji?kvTerE0e|5CRtn^oyAq(6s~U2X1x@yHp$}Zz!a{w)XBY`YfdNk3Dw+II+-BT1-'
    '917M9cP{#?^sQfUEDy4z1}Su0YcauvXw0Wd=qSaAlX{lA^eRm$;-'
    'TuHaoSDS|5y%q0bIwF!NV+_*ZLK0P<C`qHQ8#+BU@zbk|**fzf_gsTJTfjo|t!xdv-bYzZ6Vc^g#1~$!N;Fv51Hcw&TM(jHYKXKD'
    '61`bMLU>j|=!SUB_W|B7B;$T@}U)~NU%Ssc=4micP3<vv#IM}!>7c)bR4kmY2V<DuEx8RlZq_l)eW+}$?x%Rg~`AkNe;!K9a*44`'
    'FYOyt1ncXU3%P#Y!MX?26?9!sx+L&ey=f@TZ{L%v0+LXRVZfqSxpPn0Ao71P~#+Kb{zb}L>*p9z1gsp?<fjmA|23unyEHnv_a99>'
    'wn`N<eY!+MnQrNl)I|jl}+$@W&gHza|h_`UuQ>;!%?g?0qaJ=3E3y@NVtyN+0Zr>=I896c>Oy=uk09SRvDB&RrFU2;d%H)|mnqFq'
    '$5{Hy_b;<&+%az&Dg6j%pcB%v}yUv#t1s8m?%Zh>v7T~fX-~x$XRsdX^(bvcgu4C!bbAziNeR^(i+0FKUg}?>-'
    '@&5{e>tK2y&%u=`GpeC_-|rlM{ejwCA_X1?fd*7eO+U<Mq+okZijdyI#StHi%r9hw=;M(&L?&TA5t&5hi9Q*bCZ6eQo{Ef-'
    '8JSN<=1`fD`AlREb7OsjA*f5WW7ciNWO&ePUyUem-0+idbQt_E&veIr1<smNVQ@owj{8bG(QKi%6TS<vXkM->bl}@Xsw-OlyAsC`'
    '7R*?FPtW8K_{>bzNuvs==IZ<qR3c&)o?lqkR11uBI@sA?laMv2JtU&ripHWk8E&m{=IiNp=?~WCGARu@*(YJBI<QuHDko#T|BlSr'
    '(@Syn>9g1svNMyE^x>xSWE%E_?8=qtn@bPo4D1LQSdgvpib~S7@A~ti)%x$?BgKRXFZA<V!fa2Ww-'
    'brm?i95BRO47Ahqj+-9FjAfnci!8*ukqwPaY)kTGESK;ASPgU1?zQdeX;_K<Mj7AZ+eOAoSzP!BZsi@7ro-A_CK#I0M*?toE_dTJ'
    'C5aj1x(jH5*_G*X;#~0<Y8+D)H^R)fbiCr}#)Q>A?%t=aSysjP&5l9xqyS3yq^soEF_u<Mf`>GG(*%$vW_38|RX4@MGKNlJ4+ho9'
    '2>r;m5YjCF{YDZJ0~ehacN6muvt({bah<ky+`rN2ylib{xfKsv|c(fC_cnSqTgdZX8U469=G8=KKyX#8G!uB}v-'
    '%+@`2jg<rKzQBf|WXL5+LBU9A1G(u_eLj8}PBij0Y*k$B-'
    ')tb!0$aWoht@;gBY`PutEmdr~9rAA~#T+iBMMsHNH60N*GS!>+qGU2Rz`q|QQ+QOv2T}4YH^YAzB~zKoV_uX@;}-'
    'V$Q8JxJH7tmd8Rc4ab-Za;S5m9?Ww$LVPWV;Z78U1GdM1ZB7iO#?VnJ~YMorkmG)n4b_tnTM@Ek9fImb`T_(uc^*nAOL34KRpv`!'
    'uWi^^zElOdExdvVn&!jkm7y$aTpG(ElnW|H(d&H>9P<PcaGp)uqffT}oP4QRyfk><#pr|h`%loG)L)436!z#S2MPEkIu(VBTwV)J'
    '_G+EVXtR;{WE$W)kE)PB?Lipu&udZw7H;U%}rCF`Z>zQ$_mYup4qrf1Zcu+<ySF=1hVzjtx3cY9dCIaq~-'
    '?WvxnHm=n6Guigj=0aC!jX=d=sC@R#jo>r@MdHSaMA)$YnY(%`g7d(9>8{?6kRYqmUA=>&Ld{o^4AVNs@2y@q_WTgvT{{cxap|so'
    '9qe&wsGSWacu^LrF(JHJ<n{%&FDmjI>6v08zm}fKA@XlBHt+x)a4#denGhTaxke$WA5Rcqf^h)KVCiKTMRV&dxnH^u*7R@-'
    '|7HY+nfcra<=qHJ$9)~i)Y)Ub4P492O!Z)y^-Lz)zxINkDUMDCex7x_6!0_EQINpTG{-'
    ';!KhquE4E)RpoE+Ft^Av6GAvl3$OhLA!Z7Zm6`}R8&mG5KpOfmUBMbG4k7^RDF!G8e5_(Cng&o>;G9NM6*EYY6H{ao|hCwYjgoBJ'
    'jYb9M9P$s=6d+%I{QJ(y(*3w(~_c`FKYsV9i`#$GZWP1}K?&urT&Yhm;{YiUfk2y3n;Mk?!@{<mCMt0C;!w?f0QeTTL+Ana*d1Hz'
    'uRT*9h~(qR0?Cq)}cIs4Etxc2Rc%n^M-'
    '#BLUWTdbkV%_yj*9qzMHa6a#@%w0|*;Cv2JW}=djeCIOJ4*=`|E_CmmJZLMaJCnQ2N7~s~$vx&1?GEtdUb9HM;6-'
    'wuS*+RgTB$H}`YS)nXr`7q#qS~2#j%CydG@O$NcybFYe3I4c@5}!Cg&1lbrc??7)yAsWJQvEJSGU=%vAw@cbAgB6=FoBn1k@7V(('
    'rRUe8q$@$B+tY1mBImg#!wk$jAb*N*#kBPu4jBKz5JF3NC4IN%p5H`Rd>B(JMvV|(8dSPsFxaVKkVl6PmehlN}&mrOLDYG&H+rLc'
    '$4&7H;!@rzofO}~%z_s15d?AgDP)akRPtN~@ulr^C2nUYKDH5vF;dFft?TY^W;fw|ObbsYxjhDZ%ZEk0o}T@{%X;d~LC{LOrW*yW'
    'fk2V==0S0^jYr7j<hgAh5MalcWD0~!w)Eu=mVnusU-CJ&jI2mB@vn-'
    'b6WO&&38@p#|lQM0yU0=Ey4h$h7%GD;EQI#|0k$~1HvQ;5LlfJ&03-|=j50-'
    'tA#6ZkxvORlwPxh9j6Nn#?^V5O+gOtvB<8(d2Wh<TVWYYkC6Wu*?S;?|H@l18#6AKO|3%e+6&Q8QR?>)R5$B>mAMbCsPMgZ(GCt2'
    '?QmW<z`NfHr^7eDSb0w`q6Cn_uf>wv}sVa<}<dy8u}-!7R~^z%uy|`GrCm!zO*-'
    'ig+W?Ib|YV9fi+<m4r;cWa=7F_)J{`3ZJRD#N0p1ct4-GNk6koq0X+-'
    'fzGzU9nijR&O&X34Aza@%iTswxsjXsTTu8C;iw}8pYnnn@$W~akjL%25{G+#s?9~3<GovAwUHI^0UA3D?{!=9AdMY{6P%)d5-'
    'V=F2l^Szws43N?Cz0{v9Z&fqRbJ}kJVdYQ}8VJtL9;!RJqn(4VBM9m84D|HLW<6&$Kn5@|l)P=ye&Ljpv}Bpf*kg-Vgz|M#$*Kq1'
    '<b=keEkECfVjk^tw3!PE9uAi03iF!{`ORFd&tEx8zyfFkHX2HkY|T%Nq(C+}t4Nt2HL$PGmA6BF3w+ay(g?!zB`Ssxn7PoMpH&M@'
    'v|&ufm4VH(X=;d!=MpvdI3{`N$!5Y*9*|gDXj!K5BY#N}uU#K<P6*m$V0DAZ~6@4&xmvJZ&w{>57uD88VMk6V6-'
    'wjEBAB@WHR$xYH)gv7IN35>H`4jw8s#0L9s2Q4K@O-JTKVG-ZyG#NV%#IY#nCr-Kk&o~S-qfK+ZNcL`J2erI-'
    '(DaeDnFe2v)*G6GRt448b#_>vkeKAzSw<^l&Q(r;G^lCHKfYoQl8nF7z$X7PIe!8&lyx0trG57UmRP4#D#tE5KJ@zXObrpYsH+m_'
    ';Ym`cjB)|&rutcINTXZor&J<l#XNxX|#bMFK0FC`@!6&w6Nu4L9CK4bm9E%%yz&JUGG+>+@M7B5#V;7_SNwpugN2bHpMZ6Z7_bb#'
    'z;6l&@$Rb4%%S-'
    'S`v6_hg;3N4<BC2AzJ$FUKP`o>LK}0p)ox2`l7~Y+`9D=q1F&yWvhG^Are}Q6%NqB#r;@A|1koB)on{l=~9@R~4%v~Gc)X($@a7z'
    'LyCQeI02}#L<MwM#N$Nq_M-'
    '>WQ=j>{rxoh*`0P9bSWjZJL9C*Ge$(qkzkA&nxZ&5=dXQx_j3X>;UItdQVY4bf7n=WuLnI&4Sx?8wZogd|u9ni*LHNne)Wkzz<%j'
    'F03;l8WK>+(<eO@6L^+b@1-oNIDts&W$A621G2K8%g(TxW53B9>e?dR28kCX^Z+X)-'
    'Qy|;Ft}}Dx8wzlT>IwHhQwa7NQ2#Vfy5d*gP+E2qp&ej_D8%jm^uNX|E28%`5ByV8>F36`e)j@mchB%cAd;6#913*zp#8;saUqJ^'
    'o|E30A{mm1;M9H8Stnb7@Xw7KCVWlM3*Q(TK+)X!@)Kj}$}GJbWbolmZ>Y?YV~(9FKSBo>9;Z@6J7<;1s+&_k;oxE3td$9#HUrhW'
    'iW5CwLt1&y&u`o=d5@QsXf7RL9vPJeLN@XvD@KR!T?;Pbs|2q$3B%_`I{`e-WGKgf1+J!8_)VXj5!nWac>Rv<-'
    'KEl*Q8tSv+;m;_1{Bo_5yQ?-qRGgIPR1@nd-U-wa`?X2WY^^S(Wt-iXY+N^pYjMUx%N#ebWwQdRhWKaQ(KC3v=25APy)B0sF67;e'
    'uEs}u0<+_377cjt!Hsd#s8SdnyzMm;yI9@KDu0a!hO_vdk$Hb}cnEh^08kE4lBV&Y*(A+`{<1pQEh9hVE7VRP}{ngdUTPAzZpO-'
    '()ww{)BZ$uF}YSvL!k(^4SWMPu7q@QDv)LGom3_--T3c;u25Oj)uwzn?PQWU}f0n7+i+S1~fSRHxzfvH8#<>&Dp3ufW@ci$UWdiy'
    'Txp;dI4f2i1EecqG5s7sqgWZnN)~cz15IZ(Y1Qx7l|Z-'
    'ksa*qpd(>_PNczhcw(@!0dYx@6Y2vZIE@KmdZec6l?y{6l;Eq4ts~txeA(x@FQw7E%C_c5N`fb7EUK-'
    ';j~^BPQz1h+Ert>TJVVvXW{fz3Qh+g;W)D(9;nQET8iW`|H~|h2P18D1sb%~kldvj46lmJ0TxbIN9MpvaDv64VUb1P^i2sKDF&yF'
    'kkmUroJuj=o*Pal;@!F7v>x7_8&1RV?%Z&qtw2))a>MCi4fhv-'
    '(^GhV9xp0$KrL0s5$athjE2_4(%sM)%f#DiYQS?POxjw4d^WXU>>>oy^kjax_1DUR=~r1Wt)B(cuTx;!O=Ejn@QIIP!Sr+rOl{P3'
    'z%-;`q<Km>b^BR`gtade<rJI|g6WFL>}SC=Ix+`U0uw9-'
    'O_wYJrVTL}T?|a`m*A27U|K7N+jE2IS9o`BFs+Yw=LXZS@$TGUqOCv!7;=N@5e@ej0MpZWe;y|)1E!i7!4y<pV=)OAY<Jo|qG8NH'
    'UhOG4X<XtkKB>x&rWwhCaPP03Mbk-HG;NSY)9EQR;qZVInjX!f>6sLoruyOZO~`)SDM!<EU;WaAgYmwubTqAovGLFVOjkx`e+#BD'
    'kvX^$m|!hv5@ZoDy<37uih=1{d?Y`Z){f!!++aEh@6HXT4e;*VU^*S|&J8Bo2{id4H<%vPaDM?XJ%jh>ah;-'
    'cyF(2QPVy$(uqfbcX*Ay;X}`=Kg~JGLpX9c83!P3aLGkN=kV|5`#-|4e&-p(i?pXMXHvf*eHDOe2iyAW|fKADJzP;&v-'
    '`@0rZ*Th0w>Qo6?M?H2d(#5j-c*dx^5!VO$x{)L6T{ygj?6i6g@FV1^#>GXHiefMP?Xv9YkDTn9L2PVLGPUxv~hKyWVkkcuqK8m-'
    '^$e?bF=C&I5k*{58bM>Y#~D<Gk`OY=9iJ#CAK-^dL-<A9hI4@lGZlyY}HrK*pIU46Z6Otq9Z*o=KTVm(UUkn-'
    'd<f0+j>b!d^X4w#XWEg4R5Tyu&xBtBgN<z{}77&v51+e_lnGmQ3d{a`!+ik6*;`bjzvZOF+G!KzG9Rnps&SL@k}Ybm83e#er|YnB'
    '8GiCXo{Y`MrE5G>dgs}**WGhd?!U_U~CI=>doho`H2YP%gAggGlfsqwg|yqJxJpZc6Rd;V?W}gcbt50p<Qn-'
    'jBTl;W4^{ZU~YZdTbsQ(?PUgGO>l)4(f>&WcuOem-6Qjyjm=jhy=et;+jrTipt#``b}Bk;;7xoa&#1*r4G<{eaf-vr%}7NrG8~6j'
    'qRl~hw*=&?*gaIExKz*1MD?ORJ~BJSL51FOk?uvYEq8S6@L0ZObC00E{V>erJtUlC4UU9!tiqAxk##td&#@9m!bjHPNchNV90?x>'
    '6{M%TH5BsG5wVcN-&!LxB2?qr3E6(7or?<j6MUo?q~TR}&L!ocnS;8TpejMmrN|KMOZ7Ui1)<T+ks%k-'
    'G`m`_F5lHuPYCY_#k^UJ^hbYWuIR}8lzX(1mwXm_ZZhIaigqwfy-XkpPU1@50j4jw>|95*tv2V!oUXpxys3EyT^!q5N-'
    '4VDShU6F%4|v>R*W(2wou+rMm#5IU1Zm)KuB(1V3(pgd=q-'
    '6n84v(cgZF2VNtrLSjFZpZeT!XWTMopUlQA*M5$ok&|$NQs(7U<W*K+I^h%cd28r)^5<nj~e5F@rh3}PF>3d~Xk)I)&22PLh$jl('
    'NVq4Gv(10DVa+I`IY@f4Jv9SWFAu3AVb$ck{NfD12xIZ#yRD!f!q+N@O7+zx6q9TTO-'
    '8GkpN2HHwRLBzB$1!+{t5oVhZ)2lOo#G1oF5mOn>gHJ{tz|6~^+@k{r82iTJYO2yf=20SsSybGRHlbOxVJJJ3GVwUvxywB{gl~^='
    'bqy@lgP6?>>hC;*Hkc;Je$MbY}_X^%%G@3DavZtss0!W`iY1pxWV5ZjLcb~pp_G}{f@g86*RoWZbb#XG5wxAvlyf7q!0{QNH-'
    'L?3=1Q$et48{L&R11tz3og@FXLc;PKY<u#n4QTgb>n8+$~aXk*Wai8iL2eIjoC#+mOdejF0c(#LVqteSBgF>eH%Tg!SGr<(Gfz}c'
    '7lnK&k~f{0f`lJxsP!jDHht?!vAJ=KlrjO;fZSX9FB4g-ry7+!T?E(wo{DzWc8<N6?M`^zVGiT00i5Ys1-'
    'IJ`?E=@JKL;<HHF#K8>tB9g{lIHbOeq_Y?FS!X0Y<gn?#im1`oZH2ERYAMI7vtsj8mOoaLj!@Y#sz@y08@F^Y!2)?AhUuWcoGCxr'
    'eC6pV>a1-qPvWnJXy-'
    'W{Nc}xf+t%>@FfwOH6$U@qH?Lj;3ZCjUpx~*_Rjb>aQQ+gTP|kDO{un_@n{*qG3aM^myv)>bToYs4nj3h3c5Jp}#cD>zX6wjm;!Z'
    'OzMqUm_CvkI(lhU1~@Q2vECPXr~#5gwHRn*@aW22m_-o7oyURp=nbbD-OODXpsV{q5m)cAj~Iiq5I{r_g+c|H_1HOG^viDaRIsO@'
    'iAr#L^)I%~ksvraBakIghug{%w6VO1ShaG6J`(9mf>vI!X*Gad_4@qZgqb#P-b^}{~!wK0xOc13a5#bAMR-'
    'Olx~IZfJPZivlrDVMu3#xALSga)?11l`Vv%@%w_n=@nc6N$i`6Puq&1n%6}Y$Zp;Z(?(1MS1^k-kujidA}c-Gp)QofFrw-yzOu4R'
    '-DnN+Zr(Xbjv00ahVglAvIHcyv_`p4lLa^yl!-'
    '1Q6k!D(7`}V{14+(BEKm%r%PAUZ)0<orx2)yPjhp&lXbwvx<D~S&=hW<OuEDQ?#4AI>w@WZv0|2>X*{_(Ss%{*OBM43%`lhoaifx'
    'DpnBc;f`u6$Mn#!@UJPYDKQiZ9Sucpp$f$yOX1`$fHQ@Z|z6P8>-'
    'E#?he7fHmPQ!)Ntv@NkR`j6BUwG$;x~@%&en*%&TfKi|o95cNhc2i^BIi3}p`Ra{vsvOz*2(4(jTATK4A5|5J9vp3AyGrS3=joWn'
    'Z*EfP8WOw7dTVNa=@RnCRFl0K+u^FDp>*g*I5rLSqY+Z#)C>$f%J+p0=*PchOnh(zuUTNzzDSN8ZZK_n@iX&=_aan+b4;ta-IqsC'
    '2^=DsAI8lM#R49xy?5{3waFoR<xm$*1~}nlto6^zl+T|Uf9Es`;hbH6c^=u`6(Jt?dRFn;;3jcS%T%s6dwEBlswC!(xzmp4E;`~a'
    'pa{bnGR083!lxflfa9y0=-'
    ';Gxc0lOw+0kI>#YF=(0aLq+Zv^Z_C2pLC2*&fdANIO^CvCyFZb5wFIooe4btXsF7ecb;C5@i%uMFuSwPbiK9B)S&+>T;igPL-'
    '$AGJ8d>R9iW{7J4eQeInva+=)vH6^}-'
    '6oyZ&9Sw<{e*<J%$}32QH<T^l~7}!hEXxn*q~F^k1L$c_Py6%1KywY*MRqD{akXMly<i)IYf;UCAk0dwfU=q?k^&9mjpKcq_O7sL'
    '3iU!w7%#xcEi(m*X9l_<81fR=FjfxWsQODzq;MA5`G1nSQo~a>tH6P|BKB?Po}!Jc1E=q_y~%*i|LSZ0kBq~Ip$i0u*?qR9CTU`F'
    ';(071u1`a4ulTsrJEHxoE;m4Lht5;ZokL|Yd{9H!5WYOZIG`svL-!9Za{?HkS_iN@8HIfH=3m}hNpe^2`u_xF@uLues-Aa9UfLPt'
    'uf6;@Jf0V*3PFUDwZ;(A45u+(vR9wru1WIDJ=aM9AU^O_{2$BTIz*ySo(nx0hu9+Zhpk(J$m^v1RkCIsDd_M9iLDs*a}-'
    'H^9wspWNT#(t29o;E=NmLu>1pFq9rO=|M6dXCVv4)4c(u+0%Qo?p1TC3mTu2o12U9u&s_v!_XH71<*ovOZL?GrDFazZ59Eno)ufB'
    '6hEX?Hqcece!mWz^q$@z1cr*w*S3rO^&oId-'
    '_P}in_)3H80_;UejaJNp>+meNdS=0OTnb#E;BQWW>#;1jK1y-E|5jt~!p`n^SQddv)zfi)MPNu-'
    'zE$%>1eR2v!VFKPbOZ8I8_67`z(TJ8ExQ<Z6@?bO#9c+9)k)9fhgJ`|KR2`vr`vNwt0&!_8(PQF?YW_4cf@Cfpaq-'
    'evqI4Nh#tsO{F{MRRgBzI=qOq^ErgvC%V~v>GlNS$o&aJ`6=76)Lx6~RC};1Z30p<Tg!Zx+C(ozHEoQ-XL>6qlvS2$t1-'
    '3&X91|aW;^SGceVhW@|7omt8|c7xLhQ+Do3Pktm%?OPA;9)lU|$iy`XiA|g=m9afV+zV3|`{yq5xY$&#WF`GK-P!UoF6779-'
    'uhT7bzcM!J2q0OKaK<LUO*0&Fd}tacZCo(o`%7uYDD=K>hFqkT*d<S7Ua05+m8jS(2b#1wGRBGoSaOa9nuNvfm@YDk8<F2dw{BSl'
    'gC!Szy{%rlp0$;`Zvp1YU@){$9Y_09t8gcPun$lQ_w))QG^eUbv!Z!|Wx?&N^=JB{rQySl5B;!2#(tKl*Wfwj9buUcU3rOfM<fMp'
    'lpo`S%FSGcDjuHa4YDFQ1H#61OIbtK)M8&<vP_S~>Kfo{(YE4v52C<H6mCSMeS)hF~o9(Sup#@&jMczQ%^ecO%*sd9%|>#zk6CdZ'
    '*WJ~AD#FHCBVn74;a#~A78@TL%27h@N4YT9BJO-'
    'E(X)F+FkU#8I16q#F7XnHb>rbQ_<)#7{)DTToC5F9b+h0_K&QPLX;>nqUGtd2r1#MVbrV)#L9POUVq$1cacMGq}_iF=D4T5r=c`K'
    '`i@>HgeS;ZbyZZmX~l-JaVj{3YF<+bXns;>$u-'
    'A#9s33t5GW=z%;AR|Z{O5n`x?3v0&3eo$1GWap_ws5#=0@etvELI@5?lha^@z)R^_i&<11okdmOEUHdSp~^(&wiK$K%A#s<3RQh@'
    'AOPp9`yyjJCmA<KE^$sW_Cw<73g^{o82UV@Hi47f{If)$x-Cvu=2!XpFIbKHiXK=?@R4Gef>*t-'
    '2&O<R_Z7g@(R6!mO!cMPb7Sg6x;;0h?2hOxgellGorN&9m>$UEX{9jL32qd2B)BuG3e$Wud5A}w?|`V~ku?A!?l++mp_Xh%52#C#'
    'i6%9;F$<_;vVht=3#ebE0Ci|&ZchQ~=`5g@qyV*V$y^SSbXWV8unlrU=FKc&E99n3k6prEuFX_YBE;~d+ESS&YXxql%%PRy7j{J^'
    '6b02{e54qt;8iCS0Tqa5LIF@6L$~J!)#h}2ZczP-ZqE%WyDPpb1S;4#Uljt?5_%wychw^uzL=?fQ46_F4Y4m5tHIPn3zs?!jp5&!'
    'I~LI7(;0Y`=_R=gshv`z8?)#-'
    'HjA!)S#+J0Lf2uD`C|%Q&t%c{X$oD9B{Xnd3xwIfgnf}4xvK+8*uTHAyE?FhJ^Y)Lm%Ocpg7Wd;$N~kk(jHm8W3%Uv)H%Y6Oe|^#'
    '!aGbX3NLuoiACVG8U2pj@H&=m&ke7BbbD@iokX|ihL_zIUl)QG?3}L)!Ru3cAkUeVDKr`uV-'
    '2_j{HnE?83&9S5nf6FusX)^eU1&Z1!IHm^jaDrEff}|`N-'
    'hz8V(OGFc^frzK;4e>4S5&_mX<l8ef2f!AF;1Q;86D>V#W2k|>eV;y97!juQO1n(wvX$1xGE4L<{mQhM$fR)61$1|?x;KW(1Zl^Z'
    'b8L@t`QgqLV8nz#H9J(FipV@kxG*q+|3IL(U%gQ7k`W|MT7(QHwiSD|4F{awInT*O^nUVq*nBbnsQE`!*h13zIeBp>14j-F-'
    '<DQD<qej?=zeR00JD_iJ?6V(SvNpcUrAzHBAeq6*oJR8R55ANa~>@~Q>UEGV4?{AHYvQF+43VYKSPC^p)L*TiEHvQT@eFCp`c~N1'
    '*OI%)5*o*0zJcArF6)0OMb9iKWx`X4V${ZP)-aOr2nWG}phsWD1b97|-Isxv3%obeX^<;!p0IuBg=?J_3T{+L@W-'
    'B2_d<eT_`L0Ja5-Rv2g$@1esIMdQw>UL%tR58$s+y=g=YWyYK;ux$&m<{gg@2W}=_G6<t~eZUHS~x(hhqP^N|uq-'
    '4^gI9rP2iZWv(bHb$Ev>ib{PRJ(FhuWJc<b#z_9aZFK#W8G_>%UCWjEspihsi)bi;OihKAcE;0W`0%@!=29njGecjL+~ay|K-'
    '=EyT5Leu-sjpoK-'
    '*4m%{HKICx#=6s(kf+3&#z(k5DuaPOg*_h5EbrNU4jRZYo68w@WBmqfB30cCvFO^cDHAGOWjyMP&=`aAnc)8H@0dJOd#!qJ1Dn+R'
    '#5Z(f&l4BO=qwf%b_AnQiuzevOTTX<&9~ne}=LsWc)b%q42HIk-ZN&Sfq^BK^$OY6N>%j#N+Dnkv^koN;7j%mwOIep|KR5!0&u(I'
    'Y)bE!2B9faj09;mgI?<8}>&{ed<oSYaQe!Aq<lY`Z{L6&3bP^h_}z!@FLUZ(d|Z+O5dGZpQhhn@zkPeN3CXuuZ{UJp-mXRx;TYY}'
    'JSsSHW|?M!sD~a`1pgx=>ft{~*?*aZ!Ite?P2|U25+#W!{9eG0)KH?5oNEr9qMyBrh7fiD2(S^ONGB{hQ(*qU1YxTxl_av)w|8Z?'
    'BS-B=AImPg5ajWS8&iq5_9^xVoso;Z?8BH#9QCj?sqZn$&A|pnWZi1TYH8k6XfH5tTP*7HiWz)h$@@?>a>4am(DH%{uPt9)&fD-Q'
    'Cr_3gB+mbyu*R{)Xjw(&DwL-+OS@5@PMu<J#Pf4HWk335_4y-'
    'CaGY@nhF@S5Ikk9~R)+tEaV@fF+kjskjD)Vh78#i^M(<4!uyRmwc26-h6aXvBNuzE-H3-)zSF|M`jo@JWA|HbdhX-gjTKe>`c-'
    'je}YeKQA2!POuE%e9)xlymRj07VB!Cw&3f(%7XGhTaVsr*n@Qm)sOf5%PIebke4flRkGjdsqkE9xQ)bw$8_cj<&&{yIxk&4~xkxz'
    'WYy;{TFGdq}dmw1A?-'
    'Of~H!0Jn68(++7GsJ^8s1?{QAvMG&*X_;Wi)z2ghjS4)wfq<u5w;0sN5KAa3(8wDmJ$%b6ez?AxlVX=>BPNDR7bToZB-'
    't+mv{SWYUvUhd-'
    '#K4`)ACsH86^KUS(_bIyINQb|8@iba>t)%eP2O>=!6{Ff0*VBcSH!$DVs6by~r6^=lx;uCy?3kDQpJ32h1*n2QCx7Y)Gab#Yq1Z='
    'wy4W;62M-8RoY)1{HTtg`1bUbAg9)@Dx4#sXQ*KONKpbzwU5Z1(AJqc?zQPXBBxAu!t;1r`2Wa|BLmGt6t@%Jj}&E-'
    'xsE=%P2bA^A-a98yT|D5Trwo&-!EO)i7!arxbtL+s2ImcaXukg>g?rH~xe@3dJd`CxwLO(;9WftSZwRx#hRMLL+-'
    'Al#!j&?5<=R4Xxm(Z)@G{$$<NHTe)1MjwxxmDs@C&zx`44%cm(0l<3`B2Tcd>l=Dj%{EX_*O;n46^jlvon9>invnfTQAUNP;7Q^+'
    'Ta$A=KPUP$M2)CxpKES?UcFGrOC{Uu~V=?rfJTM18yGNV{IHf|FJP_1z3-'
    'c428ZVGA~=9pQg==l~|ALm)xUNnDuCnQeoDkJ#vY>CWG^(3Ol<R9a$>QMlyqSz@Jg&jk>DP@U3OT*U{m)xo3gxHduQ8z^VdY+>up'
    '*XV71Xg&&(OnB54S|E+oCB-ZELqBt&wX`0)Wxt(jiN`d!}JuCafW_!1F8nENc3~*a#4}~4OT<tYTVfQx{RTd&aIx3X;*~<K2Wqzt'
    'QFX#%<HT!mZmI{*~?O7^Lg0yEYq1R?`J}5F{V%r1Xgu(DpF-'
    'I#VMrM$9JSC>76nf%q3^^NR>FJ{^y?m6Vw~w;)@llq(KFYGWkFxX&kMD629zl)V5RTQpgO#t}=pwkz<fEM_l$tEU8)Wjl^URB}9v'
    'vMD_)KM%TLGV|%*sl^7P~Bam5Q?-?NutydbC$A3HOiFefBu$x-'
    'Q}{nm5IlVtGBnUHj9U$x(ul?#ORA*bYRLL<m^O(;{x9T|Wl*ggxs=VzZUvnCW0Tu)Eyh`U^U+m$|D$=)hj?t_&T@E8NwgbSSUne='
    'CI%gWVN&;qT~Ryr)8L4sJx)OCi6*E^$#Fq+>#%e+u_np$|iH=n9(LzR=#K;yg%umx}Ws?VU^Lbs1ew>RL{?hOp*6p<!~Z+?4`9Gl'
    'f@!$C`r78_GIw6SAtlcVcNGtLg`Lbr@Mrt5|@g(9s@(lzF!C5~~8=OD5aDAEA=>lPUJ^N2=t5<XQXoqg3)?GS&Y5XqC)MrrEzAqm'
    'uc_bo=*XRk9$NQIrSi*ig`;kn@TJJpwjqC3M@bu}`Tu57IuR;yg(E<P!9NjQ+k?nI_FH61c}kvyU`UnZrbXA6D|tF_M&`upKNl=y'
    '2D)bBG2d-pie=8nn3U<bmHX)hTiO1=E}sM`LfgQ{(u%W~B6ZIt5becVMU0F3vgc!2K~yY9Bh-'
    '{p!NMHx#8odLR(+IiOLjfJegCtR!H&DuYVJX^;k$iqjws$|c~24Q;Uq3wUOnX?*-9%c5PCE$ZmXvS?R_MIA{CRvrf*xG-DOu{2fF'
    'fv&kW*SW2;SBxZt*E3ZmSX<iM$Z@jRp7q5@jP9UFSZ~Mr)DfFD+fDl8;lkl8;lpFo(@OY=*z~GY2y9nxM9~5ec;yjA3qZc7XYv<-'
    'fE-5@C;+LVkI!8I(nX)1y8xt`K0R*%h}}Ysx=8&8?6O8(r2b<mJ&>mkc*FFhl`5o!7-0J+kxojs6OsW8z`n8ZG1o1UH6$~d$7-'
    '|O78DJ^D*G|P;rCpeX_5Rk3qzHK;j%0Y*Ga+fE@bQqKJcR~3_nZ3aQE2!T$^hg)b@_ecG}$Fu0D>CljjI$<SmMj$K$A|Aj8z1Q3Z'
    'wy`?b#~TAuh0K2i)m@Iq%4fe)<z83o{@=;L$4XBmBZZuqQ2pPm~&B>H^|!3TE7zJ=iP89k7vd@2neG<*G|qE<6h`4|e;(+tz$0Q7'
    '*BsYA2%NaV?xSrF#=IMep{&n$p+7C_6h0P2<k(A~(36@1{wSpa>W0#J*BZ&M>Fq~w0n!(H61k_XI&l43|6Hk-'
    'r+>AL<DrgIi(>Q#on)vIKAg;U9{#hFD>^(j753{~)^XBI&f2<6NIsM7TDxly&8K0P<8y3wcSMiuRc{R*K9w#<HoQ1v-'
    'IkSAxdVS4K8j~T;MbjYP7U>LWA{J#qG%`&&VCPre@RBY{qIMed@T^3YP7F6G7LDfA4s(X-'
    'PF8IJtvY`4R1*(Nq^fbhsJRe2K1$cNQ{9fafjHvI_m<`%h6(vXebBL^?BeTLHYd+F<Re+XVk+X^#eDD%y6~)#<dL}=%KsIL;z*a;'
    'bpBr1>)2HXgR(JaJ+}NURv40_K!Oq#g5VpRc2lBW`8)XbW2}fcjJ(p-9cBDqDPUAlX;jN*{C#xsL*-'
    'H8s<4p78zp}WBv$*;pi>q~0xVjfvS%VK;l*QGTDO@dykjCj4hosLnrc!odN_ma(@SWY&*BVnP165JBxxa+C>OdMb%N?uJ-'
    'O7wByCP>7#TC57*+p@+l%C0tE0E3E1#lJ9$LGe?5A^A|akVaedTw0Nwm6^=u3+aJPzYCF(gS%MsWM!(divc~hEj7Ba@VwCX=bxa6'
    'T*R2$K<{E-y4*5uRH-jru*^VSy+{_uv(FY)p{vd-G>ag!3QqR!m2X`s{yD6T<vYW`AnOemD$n#+m{+Ml)JdAuQaBac2h-p{{9-'
    'mYCDyzAXFV0o8Fbe6Lvw)DT*q1hjWUe3SRY`BB%n<oKpZ*CHnZ>s9Hgvo*Px`(WmD|743=x3!w@&&Vhwc)kzQJ@vk;YTY{}Ph(uf'
    '}Nu^=_iu(<)O#uKWH5cG(4VC0|75-z!xq3OybV2?n3#_%Wz*?CF*7_-6O+X^r-'
    '~*Rrf%R1iSUYMAP(0><wUah`#OCo>hM4aen<w1W;s}W-'
    'kB*8`F8(b9)_~Xyvw^=|V>7%Gvh0eSTNGLF66Y30*0uCZeq@1c&Mkl}kleWikhPLNJvXw}r%%s~EZP<a6+#y5oP!D>>nnO7kB^l`'
    '78-'
    'f2u`eea5|P=bEZar{VQ3f!D6mXf@CBw}m2*_6a@&sj^6+{k&a^@PHw&+|v+!D#h1Uitcuhoh=imcB&BE*J6ufrP7<hj&mI=*!$L6'
    'V6M$GOLo2O$LkvS+f&%{N!EnC7#0-'
    'CvN9*)dym8byhf{ZMRE_jELMbY&XJ(C|@AexZ{&;^1USpZ$D=+kqfYXkc9+~}fRad07Y!Nxhb5W2pm2l9AXnM~v1)GD^S+mT06=f'
    'Pg?>S!uU-'
    '`gSR7ntVfHpxRU%Wrv@X~1D~$&7UcI>$xk@{+lx#0f{UjSSgn(q>zk(=awR<4cmftfI_(t#ge^RA$#V3M;}oc(OpJ8Hd?jZWOHkk'
    'zz$GIB#(!jaaEzDet%%LeN*k%<>xmPJ24-X=m+}P0O<1sIh33@3K;!97-7>XudQ`V@BS;FX6Q7-'
    'JEo$f)fFrP>GYkW;+>oa7}FPDEZnTF;#G)(Q(RU`;#f=c;&MF$+PAJ<+A<BRP#&avi-'
    '?4bE0zD{$#rOm2%nsWQI9OQMNy9*XZ))Xw_&^dCB6H^k}r;9#|+wKRi~=6);1k`n(uL<!hm^XMt&Bh5b5c=t@~;yFU9C751m4JQ4'
    'pgipqRJ|G^nyA4_HDgB*YdW7Q6~z9v}nVfpmyah}GV9*Vz4j%e_cWGd<3xMQygi>Hof{&h^npXiwP>y6UL)ZWtpIHK~o-'
    'tKBGBGKO5U9C;*%KcK{uJ-'
    'b)#U9=u12IC_e>&g`nsFuWzN|DjO_F|~lfJe|zkab9glH{9r#MRmX)_kNU8entihMySPdK`aG%;U9dPqj(L#WTY!3pmOY6;!LT^&'
    'hHgnK&EX-temsZO_|Y{H@+o~W>o0$DLt=5?|8Q%RWlM<|K0HDOKkadIM@tq}6=6V6qLbJwIhQX$Y^35!Tt!r2dP!IThNEFyC4^7;'
    'VxNDF&QwQyck!ubED@_Ri*_+B#naQoQ26C#{h=j=D%zv%RpxAO!@vs8ZhLZm}8!v0Ok93PvVocf*|n|)(*M#-'
    '+a!OA<X<~t0lxOfuAq41Sukj-J*>=1E1(e<(Uv)2QsXmh7;=sZoEKV!AG{lG=B3`GPdTc#r72$K^?t%OswhQvWX9bQR0!OX}v-'
    'Z_U@(v6d1ze$=<-c-'
    '_7j9~7KP~!WRz@a6H?_&p@mPy=x{{xBwe10iU?5>PpE?@X=SVrc*!;TxXtCRV3@T$NybTU5_>nj<!gSDv>a`!{Dsh6=vH^k;IUd*'
    '7i?_hoe8fN;Z*v&<3GBK8jxyH-2uq@0Kgf5I_ekRze+3}Y{akNW^n+XD-oylzT9L1+uyhEky1K<qTweBeIAaoZbJ)5lp2AR6Zd~b'
    'FCtFd{kQiG0Nfdh*|dr6*XQW@!4zJSt*l$@Jz7W5sOmFsKRFUQ9&_`J13VX?b{Gna4MQ^AqThx(1N`Ky;QnA3m3&MiR`6Xb!dolN'
    '6?+O@IVZ;<`_l9(r)Vv#0R=<gOVBXlB*cU#OgeuI(lpxWta+<KPZ2nzvQw4dBi>qBHM@K0?~K^m?(A)fb;frx!$vshOUvt5FNii&'
    'x0p3p`a4OhPC#(9~N-Yz;&CAu*VZ8Yh_Y6)j+4OV{Eg{|;y2n^m-'
    'Rc{jxB>t1;op4iZ{^oZAxLM=5bGzyPqwPDu`zVgLwdHJq4VW%>(}ifJM#k8PZa57k1xzS0-'
    '608>4hcz2H6i&Yrjsr=tRxg0Fv#6NAaZvKLGBI&aEb$@n*Pn~?(FTI&ghos`653=XW#wqem66_v$OBKFD7c~Y}c4?q*Mm8s1|qYr'
    'Ca;Ht+a=4+e&--7QS>F-@%t|>l^se?Ogv}6c5z%dC767Fh5MeUb3WZ+J3K-Vu$M$NtMNQCmyJ3>+2gk;q+@@Fk|syML-EQUg2LXa'
    'JZYnrhn0c0tN~CXSd@yNYX)Lh}545vlOqs2PKAD)+N{D1RNyc2A8h&(C`h-'
    'ybI4uT&^v0=S@;DRPwrfttmYnz#TT!nc>~#9!Z=GXShPty^=^7KAS=FcgoP>!$=R;!2RQxjAn*G{dE#LWx?Wm%E)c(<F$x*CZiS+'
    '&t%k6iL7dDA*u(3N)WhI^-PMC{rJEgL|e2Wf*;L3$otB8Sq$ncwdba4Ad=jtsJ7xPsHubf0C#k|Fq-'
    '1d3AWMjP0b^GQ$ks!8~Tq+h<V<*#}oy_Jk0zN1;c&M-w;j3A9jdFV%GQlA45nU+4O(JF&fPZjZO@-'
    'wlw)|QnxKBt=LDHAmSN~CWv@OqY0IaewQw^%9GhW{GdY?)Cv62=A2eRW8fSeGT%?zOkFX|P1XP};{=Z1-'
    'Sv1(hB=PAssZ*+;AhG!)xCgYIxJSK(3H-DW$xes4T-ikZl-+rY|Rp?EsCDRWoHu<4Ds!sM{26}vGs?N0=8)viK8-'
    '_9on0S9Id^7so>r+E+c!>i6WlMXrhScGMZS)-'
    'sAH_JB2UBxm`{7G9&gVMpV{uAd|H>`(AX@KwU>|aCM0*T+!cj<a94bYrkbk;#=uVSm?}mj|QRHE<-'
    'gHaWWd6EGX079FvwtX9wSsmG))pSsI;e^iKQp;B9GicJZxR=|CzXeJmmgj^>00Cpuhfa3UELCFL~xZ6=9$f}=?yn&4<sC3`=d%g8'
    'suDw_K&%LLI%w|N_N5CfF&#v4m;2^izz<yHX=60UYbaZe35Ywoxv!cfejSynBb14|Su@TGHMse@k}ON)Hw#W_BKbNvS#rKxU{7tU'
    '+!)Y0oX=Ig1t>PMk5bd+)4(1lsqg^tQtZ@S<)nj6}jbeycsNp?n*2<X}8pDf~ejwXwEo}<Z?jNVdMZm-uwV04gTF~oE}EO!U@X;7'
    'L|sQWc&rg<ct=x&;4&xbU0*IZ%2Gz+r9bjz;N0JjNh(7DDP-65zm;5v77r=SL%>vLk|HN>6hi5ZMN7x?~&b`oCg;GK{f-'
    'vjD5NE?g->*iRRqj{mh7b*CwP1x(GW51-'
    'nV!!<q5zlipMa1(QO{rw?*3Jog29K)YRBfZcKWP(_;L@^|7)SjWE|`_RjVqSfKJ#!*g&DSaThKQ-Gw7R~74%Kc4*DkN1bvfpgTBd'
    'mLEq$j*EdPO-y*1^<SN-Y6ATvzakNJBLo;Iru-'
    'L}6nPf`#k?j|LM8xwNJtE?HjUK6F=4plU%H<kvRa{ZILVHfh2G}iuA1W{YyjKE0RbCKyl_pB526wbh6H(NG?&x|+oAyUI8Ib8ReV'
    '3)?pO)zdtS3K1rXRAL{IfFsm<8mICgRE5SWd%3*4K@LcSoj&Swn8bRTVlf!@x3Mlaa%0vmM@1%tnueRHKvy7+aeZpkxfU&plPdGa'
    'F46@yte3s|mF1lt+-@nycN5@1?z9Z~}vzEPXo>O1YP}y=kv2=<<<<Fzk`>RgM<{zysfX9x$>|7XTw0ihz-'
    '~b;!hBE2&l%AVx+O5F=@A2pIXh#IfKohB9LOAHis0jQ`KXk{WFvM0NTx{J?S=HE6Q{?^JBD-'
    '+h{n1c%@=SWV9eaL8Ncn#$5f2J2m2*=Qr}8dp~~GMmiRm5tsur&m@svJTL$W0Z~1N7{9SvJv`FyRKBbD#T4q0yQ-0-RPDwV-'
    '2MIZIQn1_!Nog8UMY-jd?8u&36Ui8zs-9+Y>lsb5TonH$f=fAe4F+Lg_EXQ2J9bl-'
    '|jO(pH#f=0fT0Vkm7DD#_v(Yv&re7U>e+%t}PMlsB`wiF6sw3URbiJA`5N5qw|sF7%satWowoOpgsKe1_?<VfCuHrZVOVEizqKhg'
    '|)IfTGnQSAR06SB6}@V@|J3WpABfp^m{^p}#ED5tu9Vr-hZmsx2Q@xO7|*Y>2kgk>3y<Br%FeZJ{Qk$r*$(<{%JM7-'
    ')AX`}TMtg;qYUb{EuC62w)XLR|f|7+0?r<LaNexY}CM=3x+5Uo6Je)?r);kuK!zC@IoKoMI`FF1Gs-'
    '&Cp$>%gv5q1TWW)VG4??;KS@nV~vK3%Yef6H9Q#^R`>`{Mn)Ar>60-'
    '*)rU^698`ZLymj@Udexj>IjH_=POltP))N**1r<8WqNt$Sn$E9+63zuxGc*bO<4CRu_?Nc3_z^HBP`4x^vq92RdQ+e|fM7fAqvf~'
    '^_Yl;P6@*gXLMZ*M7)q}dL+M|+Q0gHG4jF{fImJ-w5oUX52wcW-wQr+j5*6xj!DZvY+%|?w#X}<r&vpu@s1M+Kc8WS%fGC3n+xPH'
    'PY%IOzlsY43D16eVVuGnJonASZ{zmNz)r09Zb9&`q`j<Jqaxht6SR55h=q`(+f~g0cUuD6xb3XAKFEavPDs|TQTG9$>>mcK0iAu*'
    'PMgKnDRYlrKSFAk@s67SsVg&)!uMkjwF9y_~ivjiTTtM}d1St*z>fB;L^~?fl6BTg=F?}#;RPV<8!6c)$7n1{%KFvO5jKE!x%<UW'
    'o)q8Na4XSk-E-jm&?7Mh6Ht)k{cse$!)|vZMkE(ujdgZA4J9RKrkE%bL(<?{Szs>2DqsltOlBlRcpIH(WRXyqaD$A*z^HbCqNrAP'
    '%D-'
    '_iTmSDzds8$D9H$l~T)~~d{b*mtYYnFn#&0c~U@PnA@Ux=wq#h7}%7*qer#nd*EaNt2qomY&hZGwD{>X$5<vnrj%nmMZ(RMJdEN<'
    'xjrarf;KW_$Zu5EuToe*ufl!DL^;jM!kphnNu?O!%xbVuGnZonASZHc<mm^<aA4oL)JY{$ozB98A^|mPQ2=I?K|iVA_Vxud;~RIn'
    'QU)D2P$4NqbJpI^~<fdI<uSj^bR9L5G<7VHVZm_hB2Ub}=Hyys@{S{@@^*1{9*{AH`^TqZm#9%|+9;lHj&MG@W0JrftJ$>d16!*L'
    'dV5)%;n2YQ4qh#StX#8b;HB8lJJybdb(tEjXI&OL!(Wn(!f>iH#<F)@Nd(X#kyGIhy`K4RqC`=?!yw<!Jh^IlXc;Sx;CN6;0?Y%c'
    '7!bTROkW>ZuS-ryC`Xb{9WH5iPR#X|X$pl+odNHOwU%(!!f+T7hk2A3+orK`ad{#L`>ESbDP<OaIHo(sq*2>Om}hsTfP!h1uR;h2'
    'O5<M9S`auqp%zQSZsJ5G3}wx6y)#qoiCOBzqsi|7qU$+)!5Jv#;XW*oeYMcs4eo@JXMIiKu~edgX|Ei<(fYN7S3<^vV(SKXZEJh_'
    'e2$JSw8lWtK-p)OK`!mHDZF8@O1*@htJ3)>5tEbXOsJSZCPlpKHcjZ-'
    'U!3?e$*l)DjfjD;TeTzJfaw_`U=AKPBNSf<b$)OE|%U_N0cZwbwmfpb1LM+s>&bSlFQ<RHRiwJ3*1`Huk1;Gd!Gd`JOmG6rRB?S%'
    'gy<sa$_%6HHV1Cv76YfYESJ)S1;NVo5@0%3xwzuHY?O)Vo*1<z=8S`;7*x$h}5<h`}myukn3zO%)BxyxlRM*$bb{4MAhI-'
    '>2OKw`kh$zrZPAD7a5BWB!vwO3nzM8ANf*Xw7Gcq@0PjUnO-'
    '<_KDXd9Ov=7Z%R1HlL9}d;oZ#cr)a|c9ujUVH~Kz|CTb!bCE1WoWXQ&V8E`XgPO;5LnSK(FA0nSnrTPk=Oh`E^wD~fhj69^_$}*e'
    'VZ*y>@&G8uyjx{+x>cN#v-d-^IRt6yLfFV7Mu!TF=melCZbO+m!8r@lL5csk7D<t@-'
    '=KQV!J|v(V12mr|H39~E;p*`&FVAzG#3_fvUTL^SdqL*sHN2Mx5;W7aU9Ws`6z3=hN#CBZLFfW)I%~J{op^&unhQ3wIG@>@FgT)?'
    'NEXG}p~c%Z^K-wd;CWSoD6n7PkXVbqNteV~93S<NN)}&H7~KD$;eLE9xA^#@h6nLcEi4y)sCh-'
    ')03Q*~)hiki@{xn|OA5v%@PmZex?x|%6<w|2TJ06JyKA^X8@+nR4ETOP_^k;;=&-'
    '{$UA#wQNYHqPKWt5t#s%%NfkfK~0;mpVSW{DCv%Mzrk`YwuYgDo1ese;DFVYO?AFpAwF2Q!&FLh|_u#OLLXspFIm}{ztTNbdsy{P'
    '$SF!x{K?>?AtrexybS`w(L<Bf)_Srv2T0_pxa3T{i_yUzUQD!4s?@42I^CETrfm=_wZB8CPB64mfQX8uz(fxC}jSr@dT%PQ^{=?8'
    'AuNLNx>-'
    '{FVTRq#c@9U$pyI7cu_JY56l3Ijj&4b+<z{LePg9j$QW8W%}KH&+G?)H<V^!{<LN*3x*Z!(uJH(OgqS*s{>Nf$3RGhKsEYI<U}?w'
    'b_zqv>-i`P=LH95Fj^g9)m+v_jJ;bVkMksiM^hca9-'
    '5#q3|8nV|;^1Gnzi_&6ixQ(>|OdT%*&zJbPTL(|+8dd7V!CbLw!tP6u$i<_$U>=<b%TgY$*y2Z&=pn-'
    '|9PK^k7MG5u!+zYYg+nNgtOcRD=Q(ksn1ahAqsJ-m{oHx@?mtEm404t8O8rV(Qt?jmirhT}AR#r@k$8a~S64m1I@-'
    'Q;_kCQz_b+QF3WeYOjZ^&j&)@1SN^hxAU@!g<1hyBpH=@Fn2^#Zpqwbsv=<#QnJ4U>uOg7h+7A=2@Eq#abMbMhmA{O-;8%2{1;-'
    '5_8QD?R^{$n>IDSN5hxP1P<*RIwIEI-#6F9+52vDO%*B20<w2A4$u#4-'
    '<6Re<MR@CW&rU2L@zT`gDni@&BRi|gb0`kdOn17>x?J^lfrWx*G6;NGowks#RBtZG-'
    'Iyl851y7{F3k1|BZ%^v(e!mk-p0#0w#(da<pTb@li>cy$gD%=fdTetp?(6jS5q}5``T5B%g`3@pN-doQ-FgYpN8o@}u{b1=@`w3D'
    'jHi5E3WJY)qxEL4g$2Al4}HBgJ^r&A{=0ynv%5{M!ZCiGn2Q|En0SO<fJgC!7TEo;_yxYdNiNlh7;SeV{%GhCR>r`{EKrLvNQ?He'
    'h<Hp^rP-FClU8z7A@K>qlHdTog-aw66{<eog{=Sc`utfqlv<!S;KPQt^aFqf|Vh(WpumuSq)N^<7QB#q?7)nbFR5VRnQK=WX~mJb'
    'L!hutC{1(CLaKB1Y$Wuyz*k1+Qz01O7JJ>sa+w>0(Es(v0cV89w6IlCIH<);?z>FhtqKelsbd{@AOZ-'
    '(&h=*q4qIFjT_7ocDY)AujB-&c>L5J*>HU@PUG#XsQmyks39G27g7tD|Q$^H-'
    'SA>NpG!vxC>M~snG>0p48}qN(Qe@@(Eg_%yl>sw|bMG2{))f9&*d8o+iuBUdYO4NYc4I-'
    '0fSEGSy4=<)5D5pvac^TmnNCmy*Abz%XTj1Ex9pl+3hEMpE@{XWA1443qF5cXX0~qb2-Lak~e#za8i5dvhf0lyGBb--J5!JK=k}q'
    'A%JnH2VsIWf-'
    '%cAYqdz;YqL$exZsdHo8#76B}Jv$?UxfW^Yl@O2IwW1$V2ox|ZWvvzG8*en=lL$Y>*RCUYZxFCv57%o4W^b`Zw@hW78H(nX#u={^'
    'amH)h(-%ejVhVLE}?&labXnDA_AI)ypTmZgs{#o6+7D);5!l5djWg5AJ`s5V?JV1HbyXz1sn{zF9tp-'
    'Y05*xL#|vMaIsH2heNi{d=mUkB?j#`4?mc9Du^Ho8bfGaFr0$=Y>EzCCA*(R_9I?Ck}dp&4bcnpEW+vBmwSb)0Q9lPDvY$7a5u)|'
    '8XnM7$`S$SieB(#hOTbwENyv;Ex!dtd@bXc*v*4oXP2eW3QewzDP74l{G86Zi)mkUKa#H1XHDy6OT=jU_&%Kv&zKGDJrWf3tQ)R4'
    'k3Lv_=Pn;r)9BzqjFitA-uQl5_i-'
    'E>`ihMi;AiTBD0A8GKN|;5!QVoTnoAiXsA&<J`;ts)8i~j(0~VYuHM==H_OrGv~7JRir)WNHHETJ;&@BbBRLa0l^f6!xgjJrSqw}'
    '@sRM%nR6w~RW=YmR<u2HS!nV$4I`4Y$=6G3h=?akH|fY%0P2)aOwxAR2JnT|HrgyLBQN`8ilYNVqra@+Woz{F6WF_qz0AIWOH?$)'
    '(IqOL;^>k}Hm@((oU{wIqZ>-'
    'skV2$e7p*5H$lo|lBLy#b$9T|i2UxRig6C<m%sad<5?`pBi7~%xRBmP<j>;ny)0m~R=>zW*%u;!ja+GGK3OCv8*aafguz>n=IDf2'
    'R6uz;bs|Jpc_<JK`h}&3Sm1dweQPO_b1;f!np_Tur;E#Sw>_?%MxuRxY$fYWp;pkEo&v0~UB`XguSa}x#XKLQVx?B;L)v+#?pOUb'
    'LpA#Q4<N$2j=ZH(v?UoT!l<;;~Ph<OOF3FM!6}}fa*4TvB#7kY_JfRZIa#y8M@Pxu;jxWAXHI?|hs-'
    '@2w+p@)B18sGZ)T!s%S)f@(t0lv}HbK2MJEFKeiRC<67a07!1opB^vg0NEy^JTpzNO1lJm=A6DxUM`vPu?j+$Aq=`Kp8+{O0vtni'
    'k#xYNIA8;6QFHvTTn?3OJA&8R9@N$8X8eY(HG2+3v{EY^V5@C4ATU+(JRbClj3i;5eEH!+aD*Jzn3zY&VobnmrJQbEzrNx_HGW94'
    'eZ8%7U3dYYAOp7feVCLpJ>V1m<bR7iOV(laiK9`)U`&@^#}wEQsan{;#>Fa!V!|d4a0llX;^#y>d_H2y=Sno=kL~1*&>aruDCBDy'
    'Ap%VG93<o=n3TQYpA@+$9gyY9ovhE{d7pBk;$Xz#mNt+8E>`wX~Cui70FG63Vz&4*d8XO)V!uT!~_&-'
    'c*d#lZ%lma*_HcMN~q;C;qS)sVj4liaArO+io^>mS4e--9bN<J_<i^2Lo051pHjaQcf)jqwC8WuCj}&Z3_uhO-AsHuj9Gc=)#A1E'
    ';hPeH`i2W!x!fK%BZcI%;}X;TPK^-E2Fm1<j+-S!*@P0UB#fben<%L7&`ow=7K7)tz9~aj(LP*Ysf!N;Svy#Qvmf6xf1e-'
    ';BbV16{-k`yj$Y}C5#4HjIXa3<Li`Se92sVy{cehF1~(LjIUL>_`;0p7w|(DMVLDO5`N?kFm-'
    '+me(DYesPu98nT)0QS{%mL*$ReOP-JZagUaH|zK-'
    'W(;|m|+`PlecYp$stU(%dkIljJbPOltar<l_##}}IX`H1+kKJjE!eEo<p8Zq&;%3M%oe`c3_XW3{&8ta2<P_CBSN;Q*=4iS%Cb0#'
    'gO9`>y>y%IVet-'
    '{#)u7=O$VoMcc>l?+`N{g|ja<TQAB96@96MtNct<^eDC_n?k0`7u_?M2oGP3f@jgN7X~L3Rw8+LAD|_7!lCO$=wEO_P$)vTx&s*w'
    'Dgfcp);h@KIlgi7aJqQ8}`{VNS0cS*ba_a%7>wUx<h-'
    '>k&^yMb?i=M<OP&R+|f|EV_0rNYPU8WN;K25|En7%4@dP(n7^E&o<EpLB>2=M<8EI%<O~fdm28U3ocy@uA7U&)m#iNoeQo%E0~-'
    'MuAdZxYfUb=G!e~a6dka$$T&J+7q2wXUBIrQGqXTfv~^3v=xUZQ$5I}htKg`zerWqTejOWKKQPzCO<ee_zm5qnZO*S8UN@W5D~DI'
    'JIlXdtp~-(85nk3Oo{kEypHM?!On9v^7gSkx<-'
    '?2Or!C{@AjNc@E&UB|FO%xE(#+)r4SI3MOasjj7<&288mI{X^pNtz{9nigT2c(OZx#a$ih-'
    '8o0_}B0yuQIF{<IiqYjYXmQUNC<FhzP``tBZ>{;Y(3L`m~m7ZicZ!r;16!wq)T^(6&I>yny=eFZPZ1{XfWi?P9l&-'
    '!9aO(QY)s9e+dra8TGO#{s7m1`P_dHBjT4eJLpqG}pHCBdthn#NjlL6uckF1R)kdup9#6~9_SAZ3`hHri+#;U=p_1%m?$^*V!I=P'
    'lh9;!~ZL+tK~LCh3B}{ks(d>RZKtI<**3-Esl-'
    'h9Yk4;1hpV45)QF>_Qun2^4N33hT24a}Jzhq?Fw5MQU_YE{Y{)TpkA37Zn_7mt5y57*;lO+1K$>Y;fU2yc8Q;_^dC*%v{~f`ITp`'
    'Z<*68&s?XP(<{$h-'
    'OLxRJabu}cqVG*`Wbb;#LQgl%mr1JT)XB=8U#YC@vyhnjMsy9x0tWIHQdX<u<Dg$b6p4Ra~C*Gf1u&4TwrxC2G+NWfpuCju)600>'
    'rDkybAk2qVqmS$1y-}sn{rK@V3f!lf5)jteCkqraGDXHx-6Cib7M~L86L3ft=VYSvh|jI8^4K-'
    'D|&|5xO&H2Q$4P_oAWEj)wj**mE-'
    'C(b9&{tLW}<<BCf1UJR22PKPMP#OkAxu7gSkr6@Z7WX=_FYj6pJ9`o&hOa!A8wqIljcyx@Z<00GQDOK8c};V(6j{*sG}&k`#%69b'
    '<`1iltg)C|W6nBb*<juW^zWrDSn1im;15VsKY+@sQlXy~orPcFf$7oH3nj@2X=x3$|KI7yRw;CAk)zou;9gd~z;G+h{k(IFa6v3r'
    '}RB+#HsAPe?M$LPqtO?-'
    'wiI&yFGEpts3{mcRkZIsN0P^VVs?Kv`66Gr!F%%F&brCnhNCs^hR#w!D6%MAZHqWosw=g$*!Gb5G0K=j`hUuLMzT0Hms1nTbb@bG'
    ';_>dQ|1iPSs90U}L!#eAShyD`@>ru&y*y5JJ1Ck?^d<<9Y*F@yzIs7U_HbWvdQ(<IEdi|Io(q-'
    '6jG`}|kM+8m$Zs#u%9YObjwqnYFzMP8FsvvV+oKT9e_9Om1vcNW|Va)H@&c+S1Pi%cQs1xMn4yuigR7cfo~m@lvi2I9Ev1^6aXds'
    'f!FD+uCpw`(l?vVxlu?ks#oQL*EoB<%7;e;VoGCuRs@j7q(xgGOGTw({SvAZKpma%+tdsfZwDOcw{HZbNG&EZz>Pq<_GExvL{hjZ'
    'biOq^0peudZb1(+h^)ASr-dk8K}K-E#Lja}LndH}#-9qE-UQEbeC#8$VOA_fhbw7uxHOv!WN&okZPs-nQq_hRe&60V(Md@CxIGHF'
    'fONx{>#fib`Auh5!ZHoyI>+tSF*kFbymtl|!#km#!an+669&vegt>xy^wYa2%;~15^xqV7eq!a*;642KamlXC)<&a{INfiM9G`bV'
    '*!%<C9)f$?PKwX8#jW*>n4Cou+Yoyfbt^O^uQFE8dcZ*Swf@Kg?mgnDu1pp7h2YN&DC<+`$)VpY3;&L1yWr@CSE*x#@2tCXY@{SH'
    'U+E?&rBe;S$F*^!V_uO~^SLnE?KDzp5OoX+Cho4LD!<=6<6><rm8jYeGm5j-'
    '<6qmj>3x*<qHo_CW$3DpTCE@8R0m;T)gg+E{brqh4Fd+@lKSeuH@JxhuP`rZ(97oPjSi`&!StL3;>}=3XFa3N`I{3!hCpslSz;eQ'
    'E)9=R1*e0l${;J$HnQk(VScyghfv9)ddrpO&zv;MPr?ZTA+T!A(nP3vn~lc+QPB?E&xvQ>sZ|u>0T^FX-'
    '50iXd5S(i5XbF|eh}*z6C;8c{?48*S*o=34vDLBl2`)46?1*TvfZ1anQC{qdQvt7QMt1^d5AY6d*x)+s_}yCMG(>WuX^CT@fT-'
    '>f|%2yz0G-'
    'O(}uCnoTSOS>*t@b!f2q+e0+jf8FdRRuREY~w2xuCKW3_jigm1{>Tk{|7~T9Q!9BCb5HR%pP4_L;g^Uqq$;5k}&lggws924>t;iF'
    'lLdTw$qHz7ReeGG8_>NNSo90kuXz~H2m6EcYUnkk2TlC86F??`bvf$Q!spQP0jH4D8HsT4_Ef?Xezbgdcb&1Mj9v5nXH6~%P>~LR'
    'OxJ1!qn*;=J`p|xvYfQEuF_onBCL)+#qZ_jD}@a-'
    'X2#xB*No*e{d+@C>TxL=0lpQoJ~60&l^mn;P{um(a4PLu@qKs1_o}xOvlzZ_Q&Z?mKfN+pc`ThjL&dGtbva)_o=c;Q&^nU5G)|`d'
    'oSf<^l95T>%~+2i@niE9QZJ56x4NR6%Mj>*>nlJ`TU?wSZH7-'
    'rsT{uGv)SvbQ8%;n6uGU924f8R5PRB+>~c}bdzqbO&{H)J9mt)!ig5ATkaU2D{$<MWmnoar2V^1!;ZE%d};y*mhHUQFIJbtvn$ml'
    '@$5=<NhMv)0+t*H{9%$8vgjj6ORjP>YZ6!cz!e8qD2gkWAK^&5$h8TO<dVgnxolmei@d`50Ff@^#t`az#U<28H#|2hcvQi;?g$h-'
    'rr^B9EB9dTJ70wvdM|-vx33PhlaCGdZEL2DZGp^uw2jB?j%*)(yddu$97(R!5SsiR4L`I6ms>P!uS>Eu`!aeb(d0_KlX!Ba-jytV'
    'SHa>q(hrxsR&J1lhT7U0??6d~`fc6OL6XXg+qt7@GFbkumGD!^nRTP-'
    '5N%}9EsK=mH5ukFyj(k9;he@<;xQV|(6E)Wcs=D^Mz1`}&M^V3;X2<IyU;CxfX5J`kKLtN+rf-'
    'pBUI^n?34MG_6v<ZTH~~5jE-|juQL5y`<?qF(fmq%l6ZclK9!6<zF>6B>yMJ$0^3TG7cN{#5>H9!A$`GWci(7W_ov-NyGFuKBvWV'
    '278bQ*kA`cQhpSfQmwPmMCVtKTBMrwUer@n$?#ne>^*^^Ag>h_1`y2BFQ&gSEt;bzck!dIC@Z=usyqE=gzoHsTeZpPK&fte>9G#5'
    '8A%O$R%BuG3_f6vYmHH;}{7QW*S^VKbYWAo~m*BF=zUe1ax*S)=o?vA=ZRAHy%ci8A?`>R@&~FtiRJ@=1vcjc@-2-Z-'
    'uD;)BPPOL<3oG$@wp66?OLb}mAIwX2nCSiKcnzP&Nu{&ByI?*#Ahb8N2#PHH+d{ZwN6=;zSf9$^ZN8DTPWURaH~BaXyV|+DIf4Dl'
    'V%&b?en~vhQokggXsKT%leZLt{2EOP<gP%yQo;`<FSRk-nUSSfEv#tQOpVt}wlEQsEXS@g36m^Gt}+>uEZ0Lg1(PgCt?~#aS&ms{'
    'DkfRK@ui)gSgH=+X+1K5f7-$Gv;+>)<03hd4h$@B*5^(g*r-i)u!KUJ3`hxRiPw@X*6-5v&yM{Gr<pptaxdoM_67A%;@Oq@C-'
    'Lk`{VSQbwXi+bLF@;PiNnl+e)ZBwxp)iH6PW>yhdqqO(HUB*PJ|R%4e+?a1nngkxcf^2nDvx?vtn;QTh9}AGoI@tIBbq`MH4}o*$'
    'do(>8!fY9Vpt}y~rJCDsWuv4if4LzQi4Lqsik^chH?OzssVzl@1Cg?5}FL*81EX8g?#g-?m?8KoZZbG$4uRRvJ*r+S3Z-'
    '_XussB3=6Z2L($M@6ErWaN)=@m-'
    'B_)&$a)AIr3;<#=Jxm@u}Oxzf=?PsoTT9OcPt$WxgsiO2SCt^Slw#V8I=M2T1suJHqAu4>cSU)>P;r>qCq=eZT@&?i;DlPZ!u*PB'
    'U`oqnO(*7`BO^t*ulXxeKzTR|BJ?51na^j)~+5Rbq7eq6Q}MWJ?2+c(SE|m5kn4R}?ZEFN3wSr4#0CwnL`9G6ID0lFDojY|@MV(b'
    'uI1nVqV(Fwh54%`(J-'
    'jMoKpNllSEyrsw;J~reI$D!pe9e&`nBDkbIhc~`e!I6p!dbcSed~pT6I~5F9%&Yf;PJdN`<?n)oDb(GBl?qO_m9_1*K*D5SMpG<#'
    'I6g#EEP43<%r(`)CF4nfp*pyv#++UmT+(7ruM933Yfi5UF0mfb7)2gF&0G+HEonCwR4HWTBa6ngF;pcuJz+uidXbG<t&}3xOHc5J'
    'f5s}#9r0fm=s@tjz;U#jAlyh03HKC3OB6$ELNT;%$%WQT>XZyVaC$Mc7Un|huL3Ujq4hTbU-6;!mVhf2=j^jJq-DfMUGQCn;+wEW'
    'L9>mmA4&L~Dp%jcw{c8tY~e#36B}Eb%r(_x3+-'
    '}DL~IFjdga)fU{0?bTeq0gE60{`i6K$3HQihg5nBt*1y#Vth1eQ{*`VTWHOfY5Hgb#YcINV+8cr+EQJaXGuEi^2v=T~blrE@l9-'
    'E7*dy7#ei%~VP7*)6CqG}fPM+P7GWHG81g}D8+HsBgdw}vCp_U@o(KvtM;!z5&JbYeO2%?<@4Y*ejMaEd4w-'
    '#nH`J|d$EAL7{9sCvU(Q$4ECF2_bhl{BYUj;e{~^vY3nt2w=LR2i2zGAgQ`G#5lf)gp626_jyZp7Uc5K@h~iVeczOfGS47q+$fzm'
    'WzPd)W#Tm;8Vp2Se%<zP8WVFhBz5L)$(VB-;5#VFf-'
    'f45UA~xE8k({KTP|qy!STNw9B3D`3q=s$rY0fufCJeFjdT6T+_i%XyxW`7gRryxM;E(GQ`ziuEKd7abiWPaK22J#I<z(o-'
    'U~#Qui6E<jNtX%;}XwYLYp<a!B1~POltN#>a+6h165#f{2h>Y%Zv>yy{(OIn%UN6ttYxXH3N8TD@qs)xiMd8uqSwN(i$}hE_vM<5'
    'UA(ooj)Hx9sxsf3!Fsm)Gb#J{QOL7voqL<9Kp0j&IM!@f>QH3_kGbVjM5Y#qsBb-zFtKRgZ`|Q}~TiVh8J9(uvjuYn~D4EYaaSp2'
    '+1Q=PJC_@v(E&%jTNufrW4W_=v#L=Jd*eHQAhAIk0Xwr&kUvWAvk<0_$mWK}29JF&9)>UG>gW=!~X%&SppOAMF2I96t}{O;lxa0I'
    'Mdd(=>&_$l4k?CC(Fak@G+?a*|@?Oesdr9l6MvOWlyc2hJ!)&eB}u{6)Z*6;FGA7w}cZakNRm*AxfRKLlK%I&+10!9JK%<Xo*FwJ'
    'G4wh@`GukrQ9W39+-)8+1urRIN9+sUB5ml@lVODlw;5j;bl<^vY3nhdI4+R2hdD78O-9%mooqwbWctWnI-fp8@VI2-'
    'qID>w`uA?^g8xM~eP`XU_lUQNv>JfzK5Ef0_6HD2~OP9%iou>g{G)e*p)WxzGdfGC+oF_3&(3qfP5hhsWXtDgh1jFPI5BD4xM?xp'
    'HTj)F!^O6Jx{Vb-E-jOjeuQR1Xt0-iZ-'
    'm(#@P+IZPfgr&kV>JI(2p!^HT_@Tf3(##|5)Cd<qPRTfHxtnO{vuE_R%X~yX#Jg3y?0F2f|J{Ahj;Hts<*al%UzMS6XNxArYs2G3'
    'Ui}5$L7=Pn(@i*TzHfA6AY%%_pZz2A=puB|cVF3Pz_U%moK1@_3<?@;M%}$EV(eNQoiVc&e&3&qe3BKc#BEqD*IlXe2Of{!h4wG@'
    '_^vYpk%zJcHm^^DPhzOJA=7K7-Nnt~=LsOxSF&l?Sn3cdO4kh~&oi1T)&nN*aB>va6U<vrG#Q!=pzkuIK{I6@s67YM8|FuUK<MT-'
    'TuRXd%z{?W<YmY7!@JEUNwMUl;ctzrWT^oQR)bL&^mfG`-B)|p`S6rQaRZW&5BqN_~x3u<-e-'
    'y3O&I`L5Bg^Ygtn<x=zKU>ygaa&!cO>nsmV}aht&K8rTM@rpqm0~E{I9vDiiBpt^eZ)7m6?99MBjiB4umyb!7Uz&`;3A+-'
    'KIZDXV)=IWM$f$VIr&2J`59Co%Usz$eOet!$j7m{TU{*E*-!yk@e}o5*nc{_$5D^gutI#suOFN=K4UUY1V>irft#=hIBoQ>-'
    'dbNGo9u&iZ)Rck1|E)OM-UF?$*8&6S%vK;laMUA+h$wZ#E>>zOS2Ws_11F?7K>n#+`@biJ%+ja0zo00B7u{6x`<fE}vF#yYIW4q2'
    'LaW)O=RKIN#4PQBl|4?yl-KNl|yt9`0zeqGql=-O(e8I{Nlz{J$b1o_$m-'
    '>EY+XxF(@rHw4m_^Jd{B=}k)LF~B(E*Jo=^D@E18F@0lFs0<z>VP9+W^9UeOVsraGkBqfBez7BCZNAoAQ$;beVDr@)uE}iPD1tuA'
    'qeRe9d$b68b1(p=X^a+ogmd6@4GY~7PKvm+S?Z1smM}AclbzojDq(g4X~KFPE4hJV5sn9kunfc+!}T7+!uR<wz+rGgkonCln{Igh'
    'YXd@X1siA!w_b50CXVDd+FNx2wDe}bp;Swp>2-'
    'n4&rV>1U6Y)x;mc(<x9@Xktj+QJ4UM!pzJ{TdOx{s2`5Fz^W+p#U1f1Z*M9?;ksnC-'
    'e7CB2|_VTobCGH3_CCtp1X%=Dea$1XVI6st?SJraV%`zOrk6`7M_3TCy6t(^ArebN~z7UwZndYHh9*8?l1$-'
    '%FzYNcUao?1%B&g$sUg3OTg+dR>PuU3xkXS}6LOGR~+P=r5Vom)6qf!<Z-uMoVs$}a8&epD;jPu<4ww|04K0B-@zpZJ-'
    '!!@;cG^Vetl??yZl%Rgz9j(<QYs~&KR0KWuQot<lJW9Z9|1aGI%=hC3=HO3hSnOUsCgsnP+Pgj4TEjnaYH&v_)G7~yR4kd<nYj@F'
    '2dB5GBXdl`xf*N48l^@i{ZN@WW3z>JH+sH^TB_mKUE3M|W#7iISgYgn4~w<>j^;j91T+f~zD~pSnZ<^Ppo>}v5<7PXiGaBr?l}2P'
    '^IK#+(E)Hyc19VcXua#2>Mt;f-b|Oen?%47V_a+!0Y{8+iAe<Jy1d=wgY#lZw9d*IwU^1V-'
    '8#G3QMR~t5n7~J0Z!Ux!DNkryZG0f(c#G6E~rs+qzJ;ihWbTJgd8iOL6(HMeVM~!4Zef9CeGma3WirQ_}c}8uh(!xX7C#GYMd#xf'
    'H^#irz<f}=Q(#3Jjr|r>1s^rc-UTp86BU$785%5k9C;V^B}$+b2|?13l-'
    'cdm<Q)Flk&}vWOahGa?t#CGPKXUYbh%k>{eZM(5fBjTr;(I4`RL*4CQv4;B8Mj&~-'
    '3Y23A%i&}gjOowTD$tZd)l(Xm#>?{##nmESSfRH;xEtX!vPTbtq29n^}}BG|C^2$&$)viAy@DA=_33790<w)YE|EZDdY2$&++x(^'
    'C?M6h`u5-'
    '^qJcLa&?Xm^Bb>|rF$;rKBGHFR_rk@Q2DI)(PGBX{*s`xfD`K2YzJ?dw=PF3^pnBoE(6z4Uw<qhujP|9Jv$*%X~<0Llo$%qvYdM-'
    'v<2C9%9lcu5r%&2D*{Pmt8$_6tT%<KP?gxY7&ZCRb9R<t|r}-'
    'r+V^(mv)sR}vuSMpshg;Z9dlspVEz66ZPEv!pgE&ysq9gm*Dp6Opt-'
    'aCz8#K~LP4iQ3~>UB~-&fg)%fo#u)w8;o6jd9_}c3Tt5G9TWJMjcsCgD3fC`Z!LW{n#l++i6t_^OR6YmcFSdHO%mR-Sz2QPqji|2'
    '?WzfVxn44nCNa=3a;1+VI5ZFXXC`pI_DvXPC6u;utJV^gPKN#6Y>un&BZ=a`byEsbUF+tKMoC&558^T%J{K<8-'
    '7aRU_FTHn@JkK5ODf<*uoA&n@@+U=KqL9=8qUvr-'
    '!`o(>RbmdW$SGL#^Npmz{v01a0`xkE9rZ&Tt;|BB$W|fQAI7YTfP&1g<~6gSY^CFEuI^qJ?l;-'
    '#A6_enj0c<<hx7fm5?V&B8mHD#)orTKvyMcLTFI@vg=qevp4=Y#mqkV<AfOd;*S&0?1w)#p-'
    'cPYkJHf{fIm)0b0GdW9nEDbeFP5Fahyf))o{mw_UCLUlP*+QD4Q>qz8}k3gjd8;7U3mT<S`4wbuZ$H@ETQ<C3Px3syr{z?U_+VNS'
    'eRvnM^RjB@jIC{_P71)s$XzM+;Rt0rqv#yI7@@U_W<ov4jr<?8&uq2_FjBi>u?3M8WrQ!TT!>d&rWGh)?Dig0JN;b{lgdTOBg&pR'
    'Pf#^1e$h=JPjV%V_6*`@6K=8*>I9pTGt?K@U&h3SGi(VP0DLK{Ra<UJ}b%gqKv&$1GU9KQ44KtDYt)bvj6Ky7hN_rJgKelYq5MQz'
    'VGQNbnAbg!cvP!Mhw1J`%9Ev)nH<>?WBy@KFtWN@lBjOv7HDbK!9fdrPL<e?r4PvZPM_$<#t{MWG=Yr)0k!O8wAvY~8l(@7rl?b^'
    '{o#$-'
    ';JsHlw>3nEOrv18jWXo4~7Oi!Jju(+{Iritv(Hk|MmMiZo_n{Kk>EMF!+QSp!<Lar*_HL3CswJ2aBA9v?ZQg>O{zZZtqq2NUP^tE'
    'd}emE;WmZvs|J&eQ)PV4dU_T*u-O>CCh@9L&O+X&*R*bv4tza45@arv2bB!MgYT;c!-'
    '7O$WdcEWDZygwKdbP9nH69LDcY;LkRypOf&Nvb5WL+w`MoP9nS{mXZiBsiKXUH#1Gj0ae#%vq`Xktz46>IcvOl$7V8Wo$H*F^rtn'
    '~i7*gX#^K^En<jAa_7?#wy^3a#@Qoxl@}0M56H7U+oqQgLe=c`N=MYahW>|jBhJU`At=aI;*RqTc|6Gx+;PB6tu^dI{E`h=SM*OJ'
    '8;6oGmO4;(-'
    'eEam{XpSPhB$lEGFR6m*?UN_t_9a4R8^)0vHm^Dc45iZo)p{KHvJA}>o^vrB)Eb*H&)SMtXMNJGd2!Y^?ZIoaerZo$n)Oe&;g#8d'
    'bX#7S4NSM=b(z^$_pUUJQuY80GA1d2L2gXoC4ql}fw-'
    '0SKaIhphCZG9>5BwLFXf|sNC?<v7E$|D?IIEkg12fHkzkPb%{BO?el6g60Sg2?F5poCzZB3aLPW(@4pCv=(MVARQL(iHNZ@N6DXJ'
    'hUdSp_WBR}b(0@!PxJV4J%*0sCcG*hiz=(b?Y=8RL(-O*Iu7rX5kvpKiK?YWt48>g?ki`lnb=4$O`+s=vY8U%*-'
    'o9VQi5B%n;qTk#U_)Qolb$*eaK#k_rO2B!C!{x?cwYJe52CVgM#jVsw68OG#$8$7}VdJ{sRw{Vok!9X!zw5$CZ^Sz-jCDqQpoLM+'
    'XkPr;QO=0AIXlW3Z!#BDamKbj`QCA|aFEE?W`?8nxG%od=nVhTmXJw<g#q>Pg^dl8>IF|)3uA8jY*6B8CXON2>4tlfbRh$!?@iK0'
    '+#hsbk}l?M)cX?#88yQLNxGEb(+?);GVVioC`p%lK1V_td9QzxN*aOJUtRS2uV-HWiCj+wCmdSlg!a~pVx90!x+KmC@qrdaIiYzE'
    'UyO1>d{tkJa>B2h3#vHb=KO3-'
    'I~ZeySvz=)23MtWLy%W{&l*H&{;tvJRe|^68fv=nzH~DDIoDM~yGRN1`|KuQW|FS72fGWHm87fe!5#u;C+TW?u&01INxH@!>?L4s'
    'lCHG}dkdJCr0eX#J_6<^>3Vx$0B+t1pQ7bn;Dpx{o$wo5aKcWMRs}CSs>}=Ry%)z$%y_HCv2M7*Tod7j=1rUv<%amO&WUovZ<q_J'
    'xM5@8PTFFjZa@cv1l$TQ!-vsG*%8+60z`Vf(L$;Z)31>Uv{-'
    '+f3U7w~NUqJrlC3alT<H3SF=@0(049umT?&8+V?UPwV8YnnP5+oM4seq{CX54l>Nk)|?|)BI*(mV8Ym5GO^OpRtNU>G$zhPzmXYa'
    'iv*8lKUOJe<Rqq!!+|IC{>H_HF;Wt|)4e>a;8DkX*m|HB1G4QT|NuCwVlaI)IW$p%YyoF*d69~J0}k0O|h*+9)!Cbi6Z)r@Yok};'
    'WEr>hu~*&|)en9QE(8pdR9ldfe<=C<iN#$;}nu4n2lgVptJHiPPsfty`dbhB@EUYR_u;Tanm6X8|WCDB*G|Av?OpM9vMvHrKxTod'
    'Pi_(V&i{Lj3i^P>C@-{N^u{`XCDL8V+U=YQxXUOCs)_G(1H6-'
    'Mjg+H6Omok7y*iGGYUvvvVj^`!)++D!1v0`|=fejRq@z(o=K_rLfZ?aKxH)$eG3MZn+uj`mjt{N3+pe@(z9zoUJHfPeTM?JEVm<?'
    ';!GA@^ST3{_hLFTK9#rQh1pU|3`mD>&oPWzJ~ty)4!lzi+OIbH=;PH4)Be-'
    'o*J)&WJDT{3vJqmbsu(PS`h}6K*t<aV>*F#&eB^Ae)VXDZ$1QXgAQ;C65#yA@Hy+tV@2S;VJ8l=qv{Y-'
    'e@yQH%2bRA2&uW!XGzAF2)}>MlQi0H%2bSA2&uW!yh+BF2^4?Mw*@l?|skGE=}NlHx#|^+gtL!BCFX;`2yX@GVinZULNax)6F$;-'
    'Z#Ts6XAX4O?)ZJ`|xFbDa!l4Z7!&k3Kr`ArzeDgWAqud12FciMp(d=L#$~0lgkoYR8|=oOW3r0dvaKL+zNjV=9F)1250rapF=t2>'
    '&eipZSd!CPWiTFVAgi{^BIQN`&5xc$6PAdOnl5wcm~$H63P$G3b9hVE4a`WbH`JA$GEOY)C!~C$ujV}9?tLVZF;H5`Y+z9mx?R^|'
    'Ib`g#l%hB^2C^Ah-'
    'rSP2qrt6m|C^9(WZ17bqS8Lukzu<V-TC|n$>Q~yEnPFf=lh2yqvb*!*wS*1t<D2QTg<q2B;3F1Jpq6RBK?4JNT(4x_Ni-'
    'GmX>RJV%eXo>X6WblHhAZM$dR45Jd-mT-`b3lm)Ov)cWo36Ay5-'
    'u<0~P@QMVE;+plraq<2)b_)+k2N*kYWrAIZ!*_ZG4+UUd5YW5YT~DUo?+;IrA1m`yr_aH%CD+O7R)WHnqglmC~%c(e+iFzT8slEJ'
    'f^uGwTi?W_<gUIB>VP)r#(Nw>7+Nn-'
    '3cKma}juM0+a2ebS3R2hn>$`uFje5h@e@mSCFh3twZz%uY)+YOmC>jHC3>7T4rtg^gF~_8*jBkthL`T*Hp3g=>=<_p-'
    'CpfZJ{3{;b2c_{3C6V7o93gG!pmDM&dr%NZdCYiTh<EasO;29*~X11MOYDn-EYsXX5h`m|{%~0uE2c1%&mjnlO(G80o{37|o{I<^'
    'CG2iL;EOL_OWu)5%PYN*wn06RDI6rfn`Wt$pepBTb7p+A-'
    '3y_&hsSGwjHMVNW;mdG4KjUXyB*!$y2Tlb(yiMx3dMgV<ps&eFt2{~M>lBx$|k2JmAg9O4<Nj+1bx<dXA=5{{5uTRus`XQZ7!4L$'
    'b0&b1ogdkJCSb9O#ez!4VOcYgxSft{NOcUqfGF|`EkZAmzmTT~Y29dzWE^DW8)D2chd!vJ8r;M*yf8_LXWAAYCUNXA?36dTDen`^'
    '2Vd{lmlCQ$%fhuU&Tvw3GsUE4vDd9)Rg4UJjjPe$QBh`_~lVfwX@Hs9}Y%fv;5yKu52>0;W~zRw*jAt=xN?qC^#Fq|Msx|}xb9}-'
    '^PV1K~^38zyo=(v{~XcXF4!@KV2a~k&3@V+}j-`$@@QX1g@5^A5w`OzK<F0!e_J=9J&t_zNsg27KMGq}CW&av_R23-'
    '<oZ+xJgtJ!;W&fc4-'
    'Ly7h0sekM3Ok1hN0Jm#1;7z0N^=k_zJ|oE3mQ(TlnYLJ2*Q7?MtOxwoLS2@M_s&xBK3OW>H%rC)WvO`oEEOM+rQ!oQ6~}a=L1Xr1'
    'kG`K!Lrl)RBLob#!vyU{jO&6mrC{CD%B*W|vP-OWU#Cmrtcwq{OEv3`DXghB2dJn96S8%zeoS_Bg{NM|(jI8hA!*q2cyL@+)LSHt'
    'f0-u>IM}|-Qv@86WNG&%95GmsaZ|b#C4N?-BHfyj@C)tlJ!s{0k^Q|VC4`nPJ>7;9Ld$}lZc7QFB|%TOqt%t;K*wq401Y2HL*u-'
    '2pvFAgL9#U*l<8&sAfaBFlAOE9&{0^{pC?L;Z9i<+*s*(sk%o`6Ha^j=)vSGEr{z^U$7>76EA=pp@@;pcOjhH!m~rfEOE@0eNp2z'
    '>x9hT4?VZJHpDb4UX0h5Yi`D*FtPaRxbzl~&2NQ6B8{`h54P5STJe=yt+#K7Bkox%`-'
    'i?qO#_x6~*v7c7h$sqSyj58k+uPKLcv^@W5l;(IQ>`{Kwg8r^4JTW&3k&B^a@A%-'
    '+OS1<A<R;sd#0R|!2Xtari0k&!*w%eVtZpU#;QO`V4l91uln}Wffy04{d5xvw2t+nV_PTGA7Eo0dM1P}c6tcgB!oS6ItSY(giv%k'
    '2iqltUc1bGljl^rlGhL0C-7xm7R5zIih`}TF0-}$Fd^b8B7}&ih!E9meODm^Go6-'
    '>E~7>JW?D0Kmf^oiVx%Pi&9SJ8H#TO_f3n72C*h5pvFj2V99*v3I|<^lf0BhFx{n8#FHC3hp6Q}=7VnuZPG|F;>5_B~@0l)5=klK'
    'EvUDErnJ!P~^PcGr30z)k+#>Tt!MHujjB7tkig+#vDdM>xWHsZCF9tG68a1V@taKNGy+$dIz-'
    'Y5$v4ww+R(K*?x}7(B0@l(wRL!uTc!GqzBrbPdKLoBgJio*M%vYE)hPe3`B)qO0qT#uOSMNt^cs}8x%1{k2BwSQEO2e-'
    'cUj7Z!@M7XBEa@*O?>^Y&jXM%RH9u}2oS1zm{B5m-*E{by7Pu!0*6vwmZF?6b;#nY+h-'
    'QIM)r|dcA^WP4Xz$I^G}{}Po}|B`|IZ@Y*OS^p%zqth&@fVG!*_vj(|~GlbO=y2FhK{f8knd9Tn$Xp0hR`)=m1OukLUnR15=%<@^'
    'B3=CA^|1;;Y2fKhsu98@OBe3Cg%W<1D{ZLXti?%O8@MvMl~~DM{Ck>xze>VES#!OmA<aMLa8n7V)eQx|;1<ItBV55e(OK<iYNvM2'
    ';e%F4Kw`seQ86j4ogiSN>&wCNZ^2{Ov*sSL-'
    '~=%I|SdC5|^as0Pcs93;QZ+Z<HK;e8G&<M2iYRdINygNiu3)sg;^mRE;3qyLH))`z--'
    '$0#2^+<Cyx34E<|KrfO@6zsfhnVs#2B_f^&A`$UC5J@#Vx0Y;a)cO7U@gj_2{EJb-&!E1Kp;mfO9a-'
    'B5FJ+seZJFlV%uMrbR;KwjJJWodlWD%q%{1TUWtwmEGtIXLh`Nm@^hc=}a+tH=<Ftl4!dY+^>JrPx@WrH1Z&eRu8DgZ!F;TGXc4f'
    'A;x9KM0DIdCtc*=)v)oeShIDU(;e%Nl*rd5`I1u0k7j7JRj?abtX7c59{uFV01vVtlukbTu07s$L{(u@n_J%k4bWi?k^Ap4RlE|7'
    'i26&J|+`0c6jhO)s81@{pH7{~nmL;%J?ziR?l7VLZ&$;uf!Unb#(&SQ0vVWO~Z9$&U@wzugn;z=L6i+Iw9?iF{Q`{hA6Eflv4I8f'
    'UiBN^@KjLdCOj`gy-BY7v%w4M=nXI`f=t$~Ma$n;3yZz3G2>&=>Lq$AikMY=9;Ish588y-V+cDxE2Lp{yD1ro0xkz*%qR0T-'
    'G2*r>FM;kLINhf>$koieE)!yVV>6>Dx<i1@AvwV?fq7dEhhE~_k>h^JaC6OQqyj8Cx5(M#<xuyaLf`y;$%d4<a9&mp)sw%tW0r%$'
    '!RdtK}#(sGW&Ui}pv*rB)#T(`$1?T>b56lYzV#JTpu58<6#3u_2!0QzpVe<xDwR|yb4M?YvAG^_LI?)cf(dbEHoN=SkQ^Y{zMx&>'
    'RfXIzTGl+l4jYiK9{g4}to+W}BHyT|fsZWF*7gwSu=NxxP(Q!{!TfpKJDKH9d_+A+n$KHJVSU1F5Z6E7~ubOMBV{zQ8*j~keafYZ'
    'WJNN+@=VVoN`@XGT*f6z;nsCSag3d$%_xgg)WC8d4g3eR{5BbvUpG?n<>yRaa-'
    '!BU8ktGV>^8|Op694b{g8N~K!SQ#Qk6?~*O3p`)Ec(bPs@OHvWVqI7cdA8XB@tgRGfqA&TTEs9ei;LV*}~r-'
    'XzcfKbDf5@7Ac#+Pdm+r1tyMyL%v_;koKGI5b2P3qa7ms5uay=D0j4PeTOJ_JW^GG7zFNkimHk)*xWA<({9w5HPceMQL|=r3-'
    '_f=!(X{C1x?+i{Omo}&)&^+dArW7rg|pucUw%x)c?G2HI*d+4F;IFlIvZIT(^n=Ca&gs*AmyMVt|Qjx!$$Rb*UI&qU%w)MpCN?d-'
    '`e#Y0lG!7Ck-'
    'P!f04z{3!U|2W7rz@4aKJ@4ZHs#EpgcKs!b`pM4QKMmgV5RRskRIA5x&Vjvp(=bA^2uT8VIoPa;m?XMH@=SlnPB>Z{G{yG_dp0>Y'
    'E!JirS*GKT@8T;#0{CU>e|5{0n8|)2hBsAx|;i#fFG^<Wn{;L!mY-'
    'je`1U8Dif<t@OSJNJ{dJRgbcYcKAPI{v4bXA`w?el>KqPA?;B`BsMqe;R0J}mP-`(1a6or-rh*TlIWKG9B5?q}cpPEqc6l&S*C2;'
    '8q(RYhdn*uNk(99H++1IhHXSJq=PEmLnaoMpaMv{8_S^ZDR)3Jy13lS=qeZs4oS7^8aQz~{I87G+z0i?c1iCE1qW(rn9bS+?c3oV'
    'Wb0lhh-'
    '^uD(_R<XnAN(bZu~X<(6GrQmxXmHD2%`OdMvhqu}}*7xv<c8>Br`zm&h^1WfI3P2?AJy2EAA#MHhY2X9__wW`zOcHP(@7u#80v>d'
    '`ccjQ+^rBoo_nPT9bc`q@4e=#m_)bi%QR1QjPT&HxA*?yk9bGS}cY~c|orF_!PBOgcB&Vv*-'
    'SZSI?XF*ymP`vgbx23ri+SejI2r7q*zkO(P53Qjm?EUz;AeIfUxWYDxgb(xY$-'
    'V7$7RlFzvM2l&WN|#CDs{tGWUsaM*GTliE_r_stTSZaK=+rRWQruf-_DL@POMrZXl75bA_YY*+g97?+W(Qa|rdtetJH^LwJ|@ON0'
    '*L<;tEC*RpUt?|w*JwZcj6=mtrR9PEngC7hOX#iNU^cv{JlgqHEy4BQeDn@r32a}5>pc)vMdqz~#P_u8x|Y4jgKOC2+MEBJ9|r!2'
    'C{6smMBWlm|o?yj*;iMQG{)+zCcc8zjM`|@{<a>}Ds71&MSl&7hxn47j7Kz0+{WXQvu=S?H&5ylB0BXR7639g^uIEjNNOmauZOI$'
    '?5WJ`!o<)txxv5kxTtNf6QS?)D{$i<xFTDS6UfZr<klZM{TLw~Q}H4S}Pnbqiq*xGIxtw`G9C`ZvPVMB(4rD1M^2ZC)+wJ7|kLE0'
    'X*5UW3omU*_Z|7zdJ5=(NEE-Wb++h@Nki7Xg@WUh%T7{70>sbcg61*6|=Na8ML>Q74U_IJu9Piy$9;PS`}lC5C8%QG6T@FNj==nc'
    'Mba*RwTc&55zsV|qgLyspu1ttbQL8enYFYwC>Uf0myS^O3w0ufrgi3X0WnLe%cN^HD+;USzG1&q?LzzuyH6af)f!L80LoAc=-'
    ';bX1UA4)Jb<G5tfMI}bJkN=%WqvMUf6KnH-nA=pbd3#|qs4l@j#CWfZjQ192?t0Z07zXJDu5oCT&Is$qnB|lk%NbvfF?@ky=*6oV'
    'E>yf3fbo2hV(7)6HC(J1dVzMoL~(P=B1L7D2Iqa#Ni%~756ly<^Yi6#GMxm<InRkQeZ(`;o<tl8-'
    'QB^hMr|s$Erj{nf2_G5OfWwf7sc9=E-o>*y~*9N=6;hdi8J?K&26ffdqshCa-pK?>Avp0K1HNSb{sbNJ2b2ZM#2T!BO~EL?V*uyk'
    '@na~xLA8|BwV6BIub6`9v%soX;xqL19Su2W&~aGk&QuFuL6eKT7%Q~L-'
    '>NA9rAPmZB!RMU4%?gzB1ZEURf7f`@RHHYwd%Qbp5zEc9e8UiM8!F_-^co-eInZGxr8_O%-'
    '!@6#87=&~T~ZZoOtAW9N<1Q;EBom$YPh*E?uTKT~C<pQSR>&sLe~U3;N1z3VPCrk|%W)6aK#!R?a3Tsdo_ZQC=0Vjh{br;!(A{b>'
    'Zt2^!RBRPy=fCiE5Bqzo0c8j^ICG4ugRx^`R~4N1DR#L)KPzZYxh-<oUU4874@Q^n953d8ur=<g~EfHK{ZX4`Pg+q-'
    'o+gbvP>wk5{D$J+$6qu1Nc0zZa~(HLONKYX>udEGHY!!?>XhTXySnrNNf!3`QmW5;ew+lVzDo_R8{*)p!@6yg_WU=J<?l*i=)X$L'
    'Tz=QNTI;3CW&Mq#W3ob$<)AL+6ZTieh2zgS!2o&GP@*8ev5sbcGm#VK2nx>OS*DeLU}<dLsVc}8^ly^-'
    '5Gn`D+6HA$MyN9kfW`Tvg|#;Y`uP&><CtBFtE9bBg|@jJjB+$jmPmz$w)74T2%f!zcQkmKT5G}7fI_O*BUeyn}*PT!BU@7?B_D)x'
    'OlB=wiXmi8mtg>s94cXEt(8#JZA4&M5vhRc-QklsXkTbB#Fm)St+8RKkmwoWIp&dw$H!o9K%MyKOq#~C`EYRT9)p}8n;cd6FZn(}'
    'tHLwdcY1k07i#z{g|7Dn_fwHQ-'
    '>#u%NsUp>hn8lpA&#}O(39nmP{P)BjB72hg@>DZ_IL9Ds)RzHX}_Y8AQrTR|+QrSlNb8h4g=(7Db?jS>QKh14u;EHvF<dGczzDX1'
    '_@F6NAL_AVugoqm@?<+RLTbLp<pTj^wwW2pQ%OcTu;YMg&!=OvJ5gIdGbO|>?V*-'
    'Q;xErBylEihLOD}K~#dVz<u3H_0LlD~sYUpM`SqEvOfEmXXu~L;-+umk`i06tJA>z3rMpOf3)pQT7-'
    '3)cUFZ+`M1_$W60HZ~DGq<?a5PeN3;m{06sf^fixXQ>Zk5)`(*$jWfg%WoqZ^9L~B^w|WQ;J2dapTJVk_^v)LnTJMgVy7UtidxME'
    'TZ8CUUkd#<)m{5Z@QyjI1-1z+L*%CSZhyJ@I>c=N*8nytIDixZ*#JUr;9jQ#M4EbT+Q0G-'
    'SaTowoJ38Hc%}gT*pS{iI%3BTx1(zew?f^aQGS#+PYC?e3dQPlszGvvL|L!_M~jeo}5kDQ?e=hk!;GInoZdYX}Jx5au2wS3Ow+JJ'
    'Nl&~Z3wLUoPuNQg!Q6=6FT$0bU_cXy3D%vHm8VqdWcg*JUzrI)vVjAdqLJ%rZb_p;EnEd4)kN?Lz&Kn{(|f5>3keto!qe|dSONr;'
    'R5`kY)|QFO?-'
    '!rXP=}FFXJ7iW?H!K>!~P~sGwQAovO1q9oVB^Ip&3+NJ>6ai~p(8Zxs+=5u}iH>^D#=VWD1jfM=q(mxR2>MG;4=DKol#_Eba@N2D'
    'T}I3lfP^16b_I|x=#Cm7R}&VoLIVK3=y=qq@aFP#Si+yUlh7iN&LXEgD+9pTJ+q4cz1aH;gPUvQbsc)xx|EEI4m>$p(O23~XKe$+'
    '89gn3vU8Qg1Wv}8zr9Cb!fqt0NK0ka@|M+4d72O-1WmjnKiwDRND*v~6CwsVYkK^(ER%-r_bn?*cvM6-'
    'w|j%coC@Il@4i>*dQHuvdiyE*hBLtzz4jAh*_IB?esJl*%6Hx=V&KW&6n@wX`o9?RwVO(I*gA0xa1030j41OOZ-yaoUqFEYII3Bs'
    '!Yz=^`k0KiEi(*eLt^%VsVC2s96Gk%y$B(u;pz+;Z&;o3OPh;?PQwx0zenll1KJZA({Gj)AovTn&qfwqJSuT7GH^tFbTSGd{=HM%'
    'VDO%Y?6Pk=`Y%B-Z<KNZ9u<n&KzK#;6`NmGFIVyJ@P>nw(f$(Zy@HbG=Eyk--GY@ctfc7gOfgK&}b?1FHyWO~5{c-'
    ')aX1RjXbYqnf>rY7r;l1TQ5^<{RpcR5u=vqzjN;@Klkt!C%J-Sb7!5k_oif!C`3t2YGBK7aFuz_ID?-'
    'Vo@#o4g_L8~(!^Vpm1$sQJFIGM<=2c+U&&RWQ+$vQblOvm3^j(1sfP*$v}QI1+|z7$?e!QA|$x3H-'
    'gIB&P5zAA>_vJN}I?nb(NICj0FI{%wcxNeX_|IT7!IL*j<AMYetR(?mRn#Aza)L*lfGy)Xj`;sn7yp=Asi><Rleh_uP2M+BEPa=%'
    '_evOe2ZAnUFmMdPp$RB?^!aAo$BvO&B)7{vDt2Jwbq5Z^Bt#P<&d@dJWE{J>xkKPVW)@ll#Q9VpFHEb(fw8>j06m}9Ko0{E;8dWq'
    'hpnl0w$<74?i@mAww`9S|`u8DwF*jG`T#6T<XZPg|*(2B9<f=a;00b39Z3TpFKk*l5;fwU#-H$3V^Kp@Hh@O&z9hy-'
    'a<iLI&eD)hoZ1o2`w#5D4JZGDN>iOh3AA{$1K-7O)sRb;nEUsJG+gxB4U;p~Ju?;LVtUIKf2u-'
    'tEP6_~l>kW6pZdfypEcf4gw?$`wbMxRpA7IX7^V%-sMbx*82zHY9Ga7X(pwu*8`d|O*Zx#KP7f=WoA0eKCfu^CwdHA-'
    '<Aff}W_jzEo4Tu7isDXt_?qZF4Cs8Nb*3DhX%iwVEO^+M(sM`qHq)(t*e^ov_{C(Ic>U|YCmZ3|a7?U?|q#|r1FDD}dIP!KO$N!Z'
    'QCOAiTqbgn*j!PC*VRI<f<&wFEC5N~yFtP8F+*F?CWed}9CxgZ*5>nInz)m%`i0^Vsv?C=yJ8JSncqfH$!85{RVO;f8UO5UK;*1~'
    'Z3VuHz<G4FN~_K_hHJYXQ1nNDC0+gUUivg#pkm!xyj$*f^J*KeGp^ZdR^I-'
    'j>q3{vg5%*%&nD!SIGKUehf+qN|NeKKcAzfzeNbMO0NJrD16U!>>Z1Kk(pc=ko~h;lr9S3RN}?>2KmrL1tEbv)-'
    'Mhf7GTpKMF)P*v!1iA-'
    '=KuIiZjBp{7%!oLm0I&ls!3j4vdgl1#6)}Efg*_xMaPZFCfSAw2O;2h11xTh01S2IXz1_}IfFZDCTM#vSTXGyr5OGbZCaD3*DM`i'
    'lK*84tRbjRDb<c?hsne;D}Z80~$Kh__AV6KT93Gs>Uk8()+DtbmaB)+YlQ4V>#xu8;Rn0Ls}NV6A)zx^h`Rpq!cpsSq#eRhb>SVu'
    'LzQHpC78-'
    '3G%2YMvBp>+a0tAlz7f_6aT$!^>=Y^Dc?Y^H~VY^H~XY^JOPYbY=e52;O$2&qj!lVLhf$eeLlCVOms?+Zm|ykkqw*ah3lfKm+?bM'
    'pscoe^*KK&&(36Fm^+jP_M*6XlHfwzi3K#yiXfl`_MD#WnNT9M{j-'
    'OpDQGjlMwE&a{jTGzL1J5bD+IDA0NtPLaW0cbGe|W@Y4yvorF=IT`ul+>Cs2UPit+KO<lKqk<DN9~+)&B3qX_v*=@Y>P}6EFC_3c'
    '8+^5-!W+v6Gq6;&#XQ!7vEGNbdN9`e@QEIb@;>{LwvF;We2?2kdEcGpf=Wr@z|PBLkL-'
    '6meVd4Gh(#1k_ovSfJNzCt^*d>tX#>+PNMNHiZBGe%1_|LNs*=@7Zc-2edVBCoBA~xyl)FJ7@ly9OL*F#l-'
    'R}uQ1T){Y?95GwkH}p$KY@Kb9R3vrCuJ^rbf)TTJ@Txgi;mmUFxUlQ%|oU7E#~GA#X2J1>Y-Rie8*f9;fVHCY!~H-__nr-'
    'a>Q}wf=WqZ_Z%E+wPXVAn>8Gz!#TMP-EpKSIwQ3`d4<zfooid&tb?YgCTdx36L}Im!`$y>O=pS>#WPD}D4y9OlQo?qGW5q>k;$6Q'
    '6B!a@zQ~XuM+#V^><1Z$!88^ZZ%#h|ov@hn$Iv~A=gnCdIv*R@-egcL%P`(*P%O*vTjrWdA})C|e@nw~TgZK1!*O{t6G+DPM-'
    '*tnjn&&Z*&3e-kK!1Fmf!5qTEtzIgpeJ=&=UV8=VsSyxI)2zgt4-'
    '4rGkO(2*;%9UjCgvwU^`0F@?Gvm21(bVQ5!nd+?jn%}|ptQSy5r6V?rXA5_A^;U7Yhuw3{@&@CZfiDRc(6dGGOW83H5E!Nn0tKDL'
    'a{i?a9l8j5<*r#Z&+JB_sL>(^t+JfZTm^WZm3A8zzNFR(jIWUZy0?`!omO+MoI=2zHSvVFKv&Hv;fXTw|@qH*@itw9!9|?HG!2`|'
    '{@Va2A)>Rt5te|(o=vgYw_IF28z>}Gm{DLTW9HRpjcXv>Z(}AY0ot5KnAj!mC6fro&Fu*Jhjo#fE-QIrpSkm}E&?Rxmjo;BFm9$*'
    ')CVyAMG1lZCYB(Vm-|d+$2AP$4(ncFtrANruNV2LQ;>pV6oy~eFB0y~o>F_!-'
    '^_7*F>DnPYU4@yhi^SEK>AFB%gPE=ybk}00>q2rJX1aFtPS<0md$iv~epkQ*;Wv)o7cj{kDFIJqKJX}UKM-'
    'k06vxVv(7IbY>)H?ABi6dF(Is&ay@s|IswukU&3h}s#f^E#QO9%Mylhe&qr*7GL3*^Oz8h^09HMqwZq`m){iGd;k7r5)%~XS!;}v'
    '8u3!mpjV;-W+kVRt}2@t|l`$)6#=nlRp;8nqs`#Az$6V^F4rB9O;FXmngxLiS>gt7GSRR#UrkrwcDsVqPjv?fc-'
    'wsh@>?HN0U<E{3LwfH(>Z>%QilDGI8V(>H;ze&T89BXWiv^~B~8eUKu<;raL(l!^D+3urVE-'
    'te@Kp8zlU(AS(4&m5l6DsXr?vD0OXiay8J4zCmq0`6gLH7in)#(%NUj)p^Qmc1CBs2_!Gff$ny-_C;Su#m$(FDJdO)_eTm64m?M$'
    '$^vWsgyGBFoBx+&=DJu{J-'
    '!ToY&Wf6*GLnygFS=4&NfVr_mWb!q35v<(_Y=mN33+260v8l&i)jg%bo^~y10xR%0=1r)tqSDwzgvgt}VgJosYRq$C>l}%T}=U7x'
    'WT?3zIP1$rUe1Rop({*qrE6S$p;VkDpyCt-5`DJ&sPeO~dE8R-'
    '+?*dn=9QwHae5N$KyCCaWUKZZ=!}g9f^|9ufI8*Or#0)EGyyQ)N3$?!(Q{O?ox?y;0!!mHYo9FgOX!qkQ?iKEtQ0@7vOv{o`@$YN'
    'yg>DoSFK(&-7Xe%OrT*Up^pL!(y-'
    'C0}lKYEm5`xz_Ui)qdJfqXcU91au)`vF^Dwqw^0G_q8;yPp_Rc3o79O6OQwkbz@*+z1ogZVM3Q0vGRMw>~ZGgnm?Y#6O&0d607pI'
    'EDZOf`&1s~=<TQ>g&ny)*B0M?vew?ArPM)DF(pnxRAcUJ9rabXW-A|6fus2HBa%((NT+XX~%$X!w_m<cxrsNlDSh-lks1lSb6*c+'
    '!Y^T}{KKd!c=4OVBh@R!ewwT+8Avh3$(DO`~pp`xYH_pdIDEi0ic6_H7G(Xou#Ohn28WGnQo)tkT@{uo_lt?s`}QYc#h#tcA6j`y'
    'ST8I?as_>tVfHsrOCb1;vGxehK_q+048F_7w!dF9E}TA}s{WN=hQpe%8J^nipbU9nTA~Z#A>t6%M~ULyd<2+MbT{@wbXwqb(e6G@'
    'AzWZt%9s_}v;{v!ZA{J=fM!C%m!~^cEPauISZAV6eKpxUaxyW!nf0S6B4vFECzR(QAOffOSQ$fdV7e6}{>b;ty~)=##+nia&KA2B'
    'zZWRRh(=%B=i}Y!EOzDVb&MXEo?(I*0}xPY2OZ&C25o!)=F8lt}3?Ao-'
    '_8V&77XsnOdG9$9C27zQ&=t#iio|EG<(nN)n9Q_LIwSpioo__~AAo+*e5`djX3KLLviWAl#{^tE)7`v}-'
    'oM6y4?oHFa$+w7;~=^ysf@$?V-'
    'RWt9yd8D(!O))2bCc3b7Cc4lg6J6+;i7srDi7srLi7sr%q6^KWqV%HTvJ`s97zH=F*Z&1UJdEFTNBaxvRm#(RY!o4ZPfl^5C1^*o'
    'gNDae;GnTtWrwwzC=1>e>Dz&w_Y|;`orBNQ@E;jT{s42!>}<cm{yLuhVSgRZ{;+>FJGbP+m_o9RCGu(m0DzZNGaV?kSvE>Ef^RE6'
    't&RS*HQX@!k%B%pVg?E5B|<=H4+-yk0-tRpe5e?XG63JGVsvQ>-'
    'XWW*7X~x+qF|<89L&^9f|+`0FjFrJX6of`rfz@(1T0nYtPe1+%*gh`4$$$m4+rRY+J^(G8M(EPXte|bv?IK!&06{beHI*}>yk`0&'
    '4*eCCtlvnhvUOoGyhaU|Gb16jrlAz-'
    '&?>tij~g$2>7RB<ZxdB|58pP4JTswkFwHehM9V^=PKAr!n>XiYHJDac|NF~5<c)0P}@rQNO9?GI|&~v1~eTgU|FfTKbZsq=9ih<e'
    '%OIJo(1AS9nS)BU^R13>z-Gz#msntmmD=SEw2eR4TiPSYh4aj_;((GR+lk<wktdSXJWttcC-'
    'WJ-U4>Xg*6TiwVAbWy70!r`NG)>*T_9M@_UlR#;<u4z$xQJHX+U#-hw!3I0N<*@NeZ5^e{_<|0*Y(N6Z8NQ!Jf#kbvc-cJ6}y;d^'
    'Cu4*OvT>3I5wgLFLo!$H;T+&Cx)`8rt}j*roXF!w*Y2>BYEEv1;;kk*7_GSWM<$jWBT8j<Z;8G)606>dc}n<bjtnONwXQqo1fCna'
    '6%TT;>`z9S`F>KjthWxgLJU7oe0)Dyv{-+fr1sB9U)8&(v!MKP@`M>llA{BS`Ty20MNJr>=7w`z|?H~i0B6M=59Z=y-'
    'Zpc|;WMn|9<@MSht^S&Dg<rN6V3ZvbZsWpz3X$==qYq^LzQe;HyG>>e~Qnt<blV84nQ^PRp(PK1@V&SCHme95bZ-dTENLYjSvgRb'
    'je9N1n^An<1p6ZT<2x?7m3?xGZ^&>dakzs<G5L~flS7K$J@2=SwSJ;_9KBuVG8u;UHi~cxa3;x&zH^qfz{%G&LFxDUORtsbOag(_'
    'w!XNFMI3~&;sl6u3A1BcHl`x0-'
    '(eQR^Q8d1DmBy5DTquPi@ed}j9f}>VF5IXump?ILmA$gGiIh;%&Ed{`i&4XJ+rJY?>V^GmA}PHv)6FCz|7NF~OeD?BbTgHRz_~Cu'
    'OTc@0nRB|?W_$J=chsPuA#=#*6>Td84*9#HLr&a+Lv}%@aZ#B=+IugGbx6F`qF9G~!(0>LkoHX+8|9GHY!l^>6Y2a)FvPq=-'
    'l(C`4rbRVI8=oh(@>GV&*iA2Mfx#Mgi}Zph^dbrAxR)so@_HHkde*<<`@*nx$bDLL4lm-j^-'
    'H@$ocMQKLz_`p7VmDJ&(Y1eqZ#QNu3Gg{U&&`QP{E>z$O}v+Y(MJn*{Apb8mYqg$~Ow5a?*ZwNVTo7nk{={U(be{Sa@oIJQ&e_jE'
    '~w7uvUaT$C45<4u$oPNMTGC5gM`yzm<uhFUK?O~e0kNg_?guIAcO*i&i|3}xAzF@#y*KcQZgUrAz?xG_&MB=A#Rv2=(?-'
    '}OXdB7KkRrbmhN1Fpah6X}OsfgLW=k6aM|W*W50+0Jq23-}lZOLw%tg8eh6{<Wg5yTGacP;~0aTNn<zU>>=o%nj|mm&6W-'
    'c&jC`!{N*3ng~a<Z{qkUN2HFMC`X)3=U2)X^Nu){NFxkm1g<;ZN<zs!<8Q3+iZiK~c*Tj-YrNt-'
    '>P2308uco#IB0sAR~$3F&MVHYynsAF!2y}Syr^h%H}IF2i~cgDvvk2o5%y;pB&TO`Y4=E6N3mH?{K6wW)FKEiikF(C^R8?cyeOBJ'
    'd7}MROJfJbH|UZ$N5ltO8s&)g6`v60h}3Np<%m=0{7Q*p-'
    'Vwj8;b=P|Uaw(Y0nXQf16VK!wsD<0r%%ky!ju6viA){fMoHGcG30ScR}nyK7UMk#pf#I89t6>x%g7E|v+(-'
    'kKm`Y8{`!(4+)&`Je=PdzBcZhEFHFz4Lz@s$((Q_IB}Aojhhk(2v2RcxL1!(sPbP1<tW4k3KJv0yXMCM5iE~DLpk*RP-qgOo6LpN'
    'dDK+2d2zgU9!HE@<#Jn@ElyHIFu)bEoVJcj;;Ih1x29oh2kUPzKNeTI(Jl9Bvuj*`1MTSAjA;^gpm6k4rUaAh0zA-'
    'w9*3vT>hD1y0IgC9bQObGFUG5R+Rd9juI?A6DxMJ`^%AXgQxB0nyE9s$02Prrx^XA_u!YT#c{7TWAr*2`k?t)-'
    'uYS|cR@4Y;Bw#Hj6kM&D@qUBM8q<s@7MGcbFixV|SPNnlJ4Uz@6x)UWWBUl6dBuV+#XB=W_Lx?~9lfXZm<-'
    'Hn$QD4izsLiBK^k>0I-|HDRy_uv>-w?d<{-'
    '(e`b~&_(QvTmd))?4qyFdahs;A*mJC1k9)G!>!9W&2h`ze^gj?gS0G*||qVz##0MVR=G@d&@u1+82fV<~TbQ9^BHpS>}b<_Pc97)'
    'x{XUvo_*1(yO_UAxixYH})wj`lkPKMMO&9RS(9#RXpKY>f6Y;6|)DR16dFLu;iwRk|!}pTqp+We1jPHaw=1UFu}w*x+3RoaC>Q>?'
    '?N&qW*RVcM2kkck6`<6wFN039c_f!Q3RBVvYK)?I>|OAjBWdHJi&7T<FU*Ur{)kQJE%=Q|6bd^RQ!YIV9HJc&8z;_I}-'
    '5Q%Sw0fZu4(Qks!Ir7>dc_0;v~_6s!qqtAO6P!TqyFl72^Z|&*u3v2Cd6ZnS+&5aAEKS-'
    '7V!g2W(X}cSniR<hT$%Dr6k~r*`kdTNLJR_kb4m+kKq>gt+=$uLv47_u6tcI(zPt$O%|Fm&}v;=IaC4JAfoLn2p-}0-'
    'n@HU@yWURUIPDjR?d#$;ql6p%ayay`^;LQ@IDX!7CXKNHe3TvL@tb^>nAn~Q0!ODrLn31)~N<u&I&sjC*e{A|`n64NRwi!7}%1yM'
    'zvciLBF;V=Z>xOM6L9l}*BXjmQz<e%h;k3P8x)pyj4oaj;==K;f)4D!$1PwQMN4Q;(On`;mrSIFe6Z>&rR+zbPdG@OcQxz)0MiHn'
    'ysyg>4+M5iGH9g*FXe{sSpXriHLM{c%8@zDVMjB+1L7TlB@q=}2FJ&2DH@h@$EK#J}{Xl2%XB@9<`m819_BEhU($@)zV3G>sLzuI'
    'c3gbsI-'
    '6R#phcbI8l|F|toh_9<hclloRmq)&QzBjBjpSzP$mX5o(}=H!QAW24QWvp}s_6%|9VM2a<gv0)H=lJ>tiAD0N5zIZKG0Fs48Ea|x'
    ';<&=q~2qmXhZSyzF=BU+J88T_gy`)IP9yux-'
    'B#)FK>%EnAf+(A>0ex;_U4eZZW34#4V2OUgI|X(DsA`E~DWm)@8cG&x4FD=z$Jt$_a9Wf6^u|88b#xnlRW;PK`l-'
    '#t5@pIzq*Ae>`4hXY=U8V(pA~8WwA3e4t^~?7Xpn>S>2Y3ea{&w+FjXW2tq;6H_)yO47z=%#`?AYbm{D%_I`i$hDF;P(#lNw6EZp'
    'rwtj!M@%nn7y&*OoErL0W{UKc4EDi5U&UA-jPunD^+Eqz!)Oi+`L!1H(?Aiwj)5H*@ar9}BmKy>n*?6;B>d8P(Nkqaq+O70JW*zD'
    'bC2P%=Ege>k2N<w(C}*Je!DP^6W20L;A8v*-ob>e_*=;HI8x*YSJ2>%o`-'
    'sSS`Xphj8~zxrx{L89_ptXO2l`WTHbIneIUGNQRzT2>z}C|DCYe$#V*Cnf2P{SI$lPd)+v8c9#6$D7Vr_K&Y+k9(A2`*xBS?)odh'
    '5Og`aD4OpTw}+5TN{T>QKQ?qlzAbgcdHPDjVu{~dEprAkpDVJAeU?IucFsb|gCDOX4?_82les8#7*A&;*gAvd@d<!q)tP{5rob8a'
    'x)p~f#PsZ;L+?omwY)F**^6_YykP2fJoq)z=3xL+};Q~v}WP)zDHAb|%JlR6Dd;3363p0Hf;8vg-'
    '>gABJ!tq5dVowfH)Ob@~6orwVBz0U0MU64xLBul_u_BM5&f>fgJQ;<s3RcmLesm+h>IdpStD7Mg!haSz`uHbE|64po$S_oJOs8O`'
    'Q5kiEPK9$~Az%72Iy`O;Fd~S7+fN?&Z8cK+DZYzG0I02ln(>=<oOgF<y4QKdFZ>@$ec>00gdNEepu;B#bY?lVyY7m{?;yCo|W|5@'
    '}MY`3vNMebv%Mun5d;i{_f>@&Wry!Q-UCrRNwOfeqX0&fRYoy+#@eh%-'
    'X%6?n+V0Kf`q>7%ezwfc4ftXu!!P$2aJ$cxN0Fosr{@=Hkbb#*5$OtWMt-'
    'pd?bptiXh_^jd=o}>pddb7r(R$6DV;io-v;=d*G+YOGq38Lng6BHrz}o9fwwy$?mwAH;v2H0H_G0zPdt-EpLiyTKGkg8tG3wjRu}'
    'FCQYbTeuEyuJHIh^-'
    'oABQ|b$?QpXY~&=LtmglNDjM8NgT_soJb#w3m&b+bLE`q7wJkk*!}*?bQK)pe*aav8V+^8Kbo$A!`$zWrEB4E_xt1NIyl1p{zSSS'
    'KI2gJzYk<tor#-hBf^;YzZ#aQadCVSH<u0F_AY(n`6T*&3O<Rx)r?$Mo7d?YrMY(|r;JN^(-'
    '`e*F0~R(;#0Q4o{E+};bHC6HZv<9Am9#ZH@c0U-t|l=mbTJIZFYtw0QS4iumr$<&l#2g*zY^T5&-'
    ')HXIKJYf9MQL0PK&PVF`f!vAx;73fGdX5BP^bj@4Ni7Y2J<EB{Bs(oR<Hibdj^WoEXw>GvsEB>H^{7Kwh<>^!J8zs^2Xr?Xko7{e'
    'f)@AQf{k(tN%iq};dt_o!lulBNt*J!xL_l{p{N|mlYdYvhHJ1V#vOo{6<JNrIEygP2_9j~aI;~-'
    'csXIiFR=MV(bY0YXqwWeSb7W?sU++aaExeqI|E$&UeM5H|_CrdY*R?j#JiEot|+&+E(PeCEk|5H#%^si>{`hvk}lJ(Jiuuf;P&@~'
    'RA=`2>d#!)k!!}8WRY^HNr-'
    'x|lwbRG*_Q{8?muZ?)5plEe9m!k}Q!(kq#yp5Mb#HHi5l1nXr?5ulwg@R>vZS;<Yg(}20gv$pqzgK!~^xQQg44~Dgmgoi#XKGNim'
    '>409j(Z%B#J9`rZJ&L>r{Iwo@F{pC22`{6;Mz`Vxs5atbAw{7K{2t^LR~hO(`&4ifmZSe!TDRgS%SIkcW0%4YFK1P%ztZG6827EP'
    '_{D3_%HBhCeP8oWRme;;m>TIp&w<E@yGCIF3-'
    '=8Gs*Z9_~S^%8x4WQU=M@u;k7q$HHKro10lbXX|qPLM$9Pdp|Kw}@Kdlx4Ez+V5d*8)xN-'
    'M9S13);4py7Je7kT83ga6B?x(wm(GMD>gTb0MnNeXe<qZLUBvwAZBRbMc1~p3G+hTILCK*B@Zj=Qc&uJ<|_N<hcw%sJXX54B_gS~'
    '*@8qAlxfZ$r3RlS<rIvjDmn&5g2$G{W6rQuQwKe-HVtYWDrz7vMFWDOOY(d$Ue#Yg6vILyWS=9&ta3&${M-'
    '}rb@1#+Ps<l;v8$+|Nlz#Dzp8Et&E3eSzZ7g&?GsIS-'
    '>=8A1H%N2>e1h1pNbc0V)Xx#gXg2i@9xREsc!k(Hl6lNxHlOwR`B{D6y?M0>vwxh^Y!FCd?3f2re3)XvUhA%2A_d12wA<{R@7<AD'
    'wIVZoV=;YILPQIsvHQLaGN0DXREla2c?Ss4%>4|u!cVZp!ALcd{9FY+M=B?aQ$q~15esz=lWW||IU?F_5_f&Jlw%tF8KpB0?rL~H'
    'P?p#`1lTcw|6w}-'
    'VN{6N?JWqPpV4T)DX#>oZxKf$v4rWPQs?6fP7l~_?+1%zJaj`OoyBs90R_1b(gT&>^JO+nLT(8V$l^E%pV+{K8*qrlyz36;Ts!n9'
    'ymn&Fg1Ezzdl*h&Kd3;xvz|ibN{WI1P-=s_89PzK_Hq}vR_6^-DVnAqLm!GUV(}_SCU-'
    'G>bYu<$su`T0!*hJVZ#t{&w{M!Ly!Oz^y)a*6bW0P)Fbk*R{ykAqW)SCB71<S)8Lvni06}45`oA)({fP{B7h=GLnG>C$PcQlBDg!'
    'eOugoJl9h=qjrGGM6m?KcJtd|b{qzfttfr_>gZkX?|De6P#}?Y;jMJ1cjXYvO!xgSjTc2hE$fFUkkMAwOAxrjrli%e=3m{@)&<%2'
    'hLsTug;R<BlUWX@JiuT)o3L+}hyt3YYJ2GgFx;T)#UilV;dKWJ<leh)k(>SCJ|89wmJji$UQYpYx5Ii@x!+>fEC+R?)M1Fk`J*uc'
    '(Y{DTtOjt{dZ`ppKV+rY4YIsa$}s)~;06O6V_QX;J>K%oFW5{dcS<{?=R*=ZPE5H4&a@-'
    'uwMgo_MqTWbK(wo`|NqzhZt^@I;(8nTdyHO{U_ZNt3yFXwGCZ9-1<ljfW=5`Rr?_6AOS})t;%~dg(ckhDqNYV|o-'
    '$$ob(ni+(txlOGa|3~%o=QBs*X9(h~469x|$_L_}DHuA1^t0AbkBtXdjzviwyzN+e4XYYLi0jFRcf>jJNL`7O_#lF_k_O-'
    '>H>=tWPoTH+oGo4-m2jU2|4s=o+s!>}T6eP|;5qNFiYnahGgc%hSZBu!DuQd+nTWjyV*52plkeq%0=+7Vg+FZ%G=ial{-g^z-'
    '`o4j7u6Kjfhje$rUPxPR$a6vaCI1oYf^e&U#Jb@BnfpYzpn2rKigLkAWzUK;-CPi?_N#)C;m}NEh-J+-'
    '4&!Fy{zBwIXjl@_s{$AEO~{sT!n%#%XBP%+yoXu?Z;^UXNY@$GtdN9po+n|v&yz6T?@1WvdlJS4o`mrMPr~T<bx)8yQe&nZzL0V9'
    '%SxU6u{|Bcy)ek!nCE-;-v5mCJ-F3BV}0*hb4`TrnI~~ml<!^Ei{4B(--'
    'CyFQ^8;`<9n!TOht90r&~>EC$GtiJ4qEeVOx+*<1g2g_+6vY?KTkHD)r$Rn%GZE9#S#0SzpZf$IMdycs!HA&JkS0gZzt&cn|_}6%'
    'RrYF5|&rz;!%Wm0ZY!r^J;!eY3Eoyf|i?rD0XBf+SPScY8C||KL_{#`@o*=9&WjC!GJ8Cw+4v|C7$0X3CyrW<*c|`nogL%>@I&st'
    'm30=|T^*vCx^qkOBsMmdsE|tqdx|m_$xuv_qTWkmj=yf0bRd-MlPGn=eX`pm2?IttpBAF(fVS09~X92z|VJu~;8~AXBG#r4JPPQ|'
    '`qQh&lBM4*z*P(qq^dqE7!#>NlmH=?az4NknY?7f%r`;^xFch2euM5kr1eA@WMZy|C~6B(Du%-'
    ')x$UED^)4Y^^k+MEs_?rjWc#*6wEu!*2#Zoex)Ln9lwxb%NY8+_m)}^;;62VZj{)yl!bF6Ejaz;1&*;F7)XNf*c2pk~pI9CmsVmN'
    '4hFUJ$AlNX2$tv;X%0(=sNvB5|vX3>3&_J>l6L0<cB4_pVCmv_+LZf-FBBzE*NhQ+)k`vj_2*R9jW|`ws{=I&9!+r4f}bg#~K}OH'
    '9gkoo#vWCGA~)9->Hlg1j<7w{rbNkD5=-'
    'CCVB`1c7C1c;gVwaI}?Vs*Xh4YluQZxyx7pI4jOts4r}*<L*uQs)Oc1Nv4cL2XHhSc__rpSGUlgwtJ;3V8L<Y2Tg`|y@GIt;LZU8'
    'N15ZzA@b-'
    'yHqk!If*BrrDwbGHT*W+4u_3}=fkZGpU6w>dtmrUH&>)VoGaN+hO7+km`;prt@Xg+vY9t007g5Y6g5In33f``>X@USKb9@Yxq2mD'
    '`ypUq(@?vaq?x;!h}x4Jme%5bBLBaIB7=i*{Eo>bDX--&Ds4slbbzag+svBB+=-'
    'xT^xr8m0W^^+1OhyKVD%jZg0QLaa{=tP!ld0HaI(dFQfIgb%!x&JeGWZH#c{eD~O!`!P{TxhH^I07=A#OR&ob?4Cq+Tbjq!g{+13'
    '+wH|dm~P8)w(uh1+He>hI9@-'
    '8J?Ftww|oisym}?FXSh=r5pPlFNs~)aH~sV7xr`JnnL0(*@b<JBK3!hAk51g*KbIaSvl94;j2o2Upgk}PK3m_b8q8Cfs4;Uc3v|2'
    '3=XohUEq>)D4kc3ihy8Po3OC6xRO#zNOP7{GQHs>$w~7+B<iJ*pA3<DLt^xDv-'
    '=NM@d7|V8|#ME=@x~^f{8Q=fUma+MjXIVUM0*<A4M~AL*8j^Kkucn%ls8rD~d4rR&$#|+Adj>Hz~c-'
    'n*1uI|0=?{B7Bz1C?5=9(_>{;tSy+38x>5*4Zs=)JkYTjP7gQ}Sp0ZiI*_%G?@JG1>Erv;53};|{Pa*3K3<R>#=6H3q=Q)Ycwu@t'
    'sUD~6v30kol3G{~DLA93XYjCs^=y6MpAzL*a`y6``8#gR+sy5IT^4KQ9j;gsVdahHHicwevR1xEnTcYnra3uOxJHw;*oI}j(TG^)'
    'dFe@n4b}(^m?LnyonSnef6SFxkbEan7r-'
    'g&ka;_h{)WJKuOiKZ(`7Yt7~T;U1&;7U;0R9!j<7gzgr@>WSQ0qGQkK7d({LN)<XXLOVLX`!$FXlVGj{WaTg{BM@gwFwg-YC68&5'
    ')VF=OMUl}69p2T;qD*Eap5BHdC`37SC61a;U|KM@Hj?C+h1k<Xt<@dZyvcZ|cHTjyf!PE_fsphV{`iRsC7x69ruabaBMUd@$yT*5'
    'o|PZOOc`kP7Efq~E65FmwG6jlZjyOFeDUsBh)SFI9fnj0lGGvQZx18NZX(iQ?@xF^y@Ior4W&>t}FiKeX8&q4^Uneo}CAWsjE?DG'
    '=+Jq3rcHQOH)Y1h_a+xkn$Hp04REo9tNG4^#WwvCBC!iC*+;FFMv-'
    '{~@*yCvo+Gn{RIBQdzS#J##j>f^nouT4<(^BV~RRL+w6NEJkH*p?<T1+Zl!S9m%U&SOm#=_Eq#;IDE!1bn3i{)aO#ILF3{NCJxVi'
    'JW;C`}sd$1QfHgM!!QD8Ma28?$(EAp{lwDF=VW#2M@Hy30x`UL~01v2`?MCrbT&NRInDkJAF~=uOP#Wb&iuUk@^D<FOB9&sI7@Un'
    'iKBdN{oRnbyYHm8tYgE@D!zw;*#0@3LA09Fz?o4hVb4h^(VYL)k+H*WTl|(Ch>^kp-'
    '7+1vAVs@2aJbePS)ykm1O}$Ql@mjaHX3%siP1@tRt4iK0`XU2`YoEc4qs%@@m#6VUFyqb_(Xdu4DoU))-Ezr^&R#WmNUp{2ax2%-'
    'f{?H*cB2a8s#AE0&FU(UeL<$DD(~b<k4a>KXjQD4bMptz_r45^jP*^ahlRcN1{ZZ4xmA6zSp|o7*q&!6KlTTLPori3}n*q8I?Ded'
    'w(xA;fy8Adoqvyrfb&F=Qph1>IPJ<@ub@Lv6UQxKbw>xv!R>KQs^Vj31$?nEi?JC!LRY#*fj&%fUp&oNmSj=~Gti1ks2`N~#Q8ir'
    'Cs;C!B+|F{b)Hg)Mmp;J;n!PkOh4eE%G&$EZjGiu9?R<J`XU2aJHCIlIPhQ~FA~#+Ov;K5|xWAsUPERH!tAzwmdNA{j&U17hq#M#'
    'X@F$h<63)U6IAzy<^oVEYFWU>^!3zzzr`zzz%~zzzx|zz${!FdU!y5HWbC3-'
    'F3#X=LgkW@!Gk)SpW7V)dTsC(<Q3p}Kvq4;cMKOV-M_Dw0iJBhr1toqV&@XCr%rn;n+=2ML|Bn-hH$Pq96gq<x49=o$EOe5^h%F@'
    '|#y+zS%Jco)Q<r1WSmzn!M^(aiDsfI@JVv&9`!pWxTuY*cqOIwu)N<gK{{Hj8PJDs1ay*fqwN98yd(Aq8w!lHxb?&YX#uo+&8Or8'
    'x`Re&i1r1x0Jt<mU)I$ZoeY6FnsZC4XINS*g!(yZ8o)mDbM+8avSEB4Y>*J+#8U>4Lk@Nu44~pyPs;$nNG&@Dq@q)jB~ZN+b7mNw'
    '>a7V(Ix}7pmVW^`{daGae+6X3ib^L!~}b>LXOKiMVPJV(bkTE<?6yy~PxSzz=43U+2B>PCOxUj&J+vKVZZY<A!A@O3#;shkjPNX('
    'h&q+vy07va0l4LBxu|^V2&M4DX1<XRcXQ?iiS3$}i)>Aic@Zic`H5shlL^!><$wx+H)&G5|`(fsaC(GCGfsfI~=8G=3EF-'
    'qH-+(MWqs-Rv6p_CaWFSoQ;`$0$((^_VO700ruqhaz2`Q%$$`u2qpp47gLRibP_(VXlcFC$&#vqKbh6nSyi-'
    '6bK&X#DZGC!%8M)SU%Foo3RkA8;_~-ae0%87Z{DP({N9?2KZD-'
    'N~0W6{#g<$!E@3MGqYP>E^u0grWTNDPV#U9@={8kPC%kT$rA~qk&xL_Ya!jmN14a|Dn&`+7=>Hc$5}OPUXYg<F8shPN0++gEqijy'
    'p4l?e6*+Ec?>#uyE#XdsW8Ly4b4`R>+9z>Rlv|EA7eu%vJj|1dxn)~+r)-'
    'dfD1%_OU6l%QH9K)?hQ}HOg;B~SKlP4?%Q+|PnL87IAT>*TS%MClCW8!Qx(pyYGh~pCTr7ji#3eGQOk9diOB>^5G6<n&$^i3o2z+'
    '_(jzf%6P2i4orS5p^9^A1<Qjc_Hjx*Xj4~cX}xX+MCUxbe{B+3=-'
    'Q}}9>E7q9{B3uz3<X4NiVprA`FO)Rpa*;G6;l4<1_D3FEuEN*kaw0<`)*>@BVy4UH|Fua3^4l!3jlSLNq&sojyUR&;;;whMlkUWW'
    '`5q_Ti5uW=oOCCi(T=nCUfc`sb<&-'
    '9T>p+WBk%W)IMj$F1&(+`sUzODCr9j&dL&(y<B9gxLt{PhWxOPASA@?qG|Cn2b2vH56^}3%M7Sb6%99I@SP#z%#Z@Q>m$Ck%XfOp'
    'sMbdh0=AO0@CIzo!aQP%+t2MsZ(v73gbk!MFaXRjQ)PmDz;08!dIDIzmbJT9Ls5)x6S*QoM++Y{I91#S(SzZN)v|u61`QH1684;$'
    'y`;IL2zT0KD;lih-p=E%dAjw@5OO}$Z&T&lpsfNX#*Kns{v7QMZXjqhI+Gq5&D9=37ToB=z@R+|=FpL~tGUuviE61UT=W-'
    'i2Zp<DTY;|6$-taIs`}oqT>6Ww^&cp+ky1v*hS}dvWi)VI|a%7z5oW24JDSS|0iN!NMoUew&Meyl-'
    '4Yo}9c%Fq2MOM#U;2m?g(E|${^QcnC{B?K7d`1!{#S2oeRgt7F>6#qJw4Z8ttYgBRhQ~T4e4yb`j%lCKDN&Ajl(`_nG2t<vQZSGl'
    'UdjXzj_V<2yqQo8J3X}s%FP6HEdimF=`wXC76Qa>IS@hJytfU)iWxf;<lzk*O8!jf^Ej0JxzOixDESMaFW^w}mqJhGQ1WJ>r*ObH'
    '2)=4i05_6!fkTcdb;vuqIV6MziSb8>1^RS%$DTQ7(zQA6Xg|}4Sa*awjfize_&_6~+|fRvQ={B*jJY7f9pNFLS}>R#9u6L6<b?ad'
    'pO4U3bbsok>;-'
    '?0p`qx#si(3J{5h6JqW5)~MY<pSIga(47kW1tX++)vH#xf0P41N4rwv(@xD%H~3jsN%p)ehZ&~}NS2rwQ?A=8B3CDL84&tw9W%yb'
    'Y$tkSyQJz8~lzh3BYUdz*>v0rv%tpCBSM#lOde4>$2{^uTjl>Z%VE{O0yu-<6}x5E-'
    '@On4eBXH9C92I$xrBqWXIr7zYC?DG=*%k+)1lzQ++6<*F-csXn7<*b#LvrfF6b)}cH?&IaG`;v<19ECJd<kevIH(2ELR)x%w-'
    '<LdB^E+wK2z(6hFJQkl$M8Wy-y~uQd0xyz?b^>%6$|`=TUEsZzuqy|6q29GwiduKZnSXxg9RKVVd(^o-'
    '|poKLY!tt0EV{iz#}%(1UDK*spa*9Dt(KFpne9?zPDD53FpRg+2eKAhk<COleJmrDkKM+>MpuP;nDg%f_@uLx*tg%q`FHwK$^jb+'
    '#_ATqfWn@K<rBA=lcmP5cHEM-kX;~gS#<0_m%p^f;peZOXBPeAE;k3d$(upeUj2YvGzVdKpYW{U!c9Yw2N!h6PYK32WUJjZB1XCh'
    '5z5CXT#+NJ5UL~;Q#_;=~k6?5iaaumA*qzu)QkNjgSM|7kgKxn;`A5FZQcUx5D1$Vjqaj**PE*`Hs++2`<sT#)=}hD<qNpp`>>41'
    'xo)xvF6@yrQYGLf|zWs%+Y`BnN;TG(8{E1XFps2SS!DZm&92aK2ZN+R$iC2GNfhKSt~=}aYF_PP=N)`bqP%}&=SZ0T8lka%W^*ee'
    'kE;4TkX#s__L#VnX2&RvM{{uSDCKYqiH%vpu0HB4;Bu}gD-{%M@5V;h6?>Mcc&p+-'
    'JZ}`4rB;BxETrY*((Wkq~Wl9m520x0r{HWK5kP;8v6rDSd;UW{=JIjZ`qzlVQufTSL`MZx7sV#+}q4Gg`{S(=KiYE*IIK!+H-'
    'worV1_0n4gMJY`6m#Z7pd@AXy&#uOmoSn8rB01N=2u;Wd?^$*U^i`cdxU{*~zlUFTlxQ<-jtjLp8-'
    'w=&%V@v_TW6l%rOn*Oc+w$PUfE-'
    '(ItrP>b_`mZe24#~sc2<nk8)KehvQ4Bn|SL#>XJG_(V+hmT4ST95=J7kWgrv0FM$C?~&wRfz^cbaPoiOpn2tm?7GyPV)QR|LSkD;'
    'cU=k2T&^1h;uarSB4&>Q_o%?JaQ^&caS7G(8Hs{ih<F)?}ss=qF#ACF8x{O7tz9sqRi7<;El07rS8_X!9yY^~jf7;fp-jZXr0wco'
    'dKUe`30+brRwWmrd+6EuAJ6L}~*Bq{+tF4;1=W(Yz$D<ZbEpf$S4&@%2a%8)<R)MEex8_|6iGUyaH?me?Q9YElh>D_AoGLTE_>L='
    'ZzOCqGmNL3Bb&A`nGalD-Ut(S1l=2IA<xtSxhs(${zoxf@6LW_V@W-$=bZ44zMyu+tlpRO9qhFb9Oe;RF>J-'
    'Z;fNH$1n^jOaHI_$;j^vqRoEJ8|@&`0MzOwgw%=1M+I#hHk&wzOlBx0U3ZJZGEk|rjX`L33L<k2BR)9SCG9)Fba2cuH2)<rgHo60'
    'DRW)gqszwsxDC~LeUfqbD%CHf$nCZdX$s;dPt>j(=q+1N_W#){FqALqoeq7mA+3W@y$xl^1jk+lK^TneM{&|2=1jDl!F@9;~IF*='
    'o+vSO_H!_7$^FW(7))uu6yC!*e-'
    'I2DD6kvFV@x%A~kQMtsgbl6e{wRXcM%PR){TfBRYa}t*PcE<9V7*ue;jqsVxZ(KRYDnt^{+ansBGY0=J~`ysI*>u^KZs?b5A*m@3'
    'g;4r%h;Ty^{~rcHb<<HDzN`7n&GDJvU<M=>?9569%%=G~pxxq64^xlZ-+73k3&j2`QU{TV#W;+QQ)i|%+n^S*6m@LIH$!E4c0%-'
    '|LMGSks%m0!&LSo|&R&)RS4UM&5V?#;??={_v{mhQ{CZ|Qy{`__rM{QEp2Cm<%L==cvxr-aE$@4*S|i%Uv*EJNBTtP652r#YOPF*'
    '@rl3H#=*EKp)sQbP5E4HxTy*ih*{3*8HgMO_Yb&wQ~(WpG$5DucsfQ85Eo^~;vcux#tnsv1Mx%0YKK*GJ65Z3vcn1FmhKbI(!mH!'
    'Jdrz11Ohi#$g{Y7|cCHBzw!Uzm7~f<tq3Q*>xdPvN22n<+jt(Kz9;O}b5I!*_+A$<oj#dEuc>Of%ngL)jR5Y+#3bFVT0H4Iz%y;)'
    'uKjjqm$J8I%@JltF3nL@|3;_sgEv5cM=;?mWkl2UG=3Oj9T4YFQU}K&Dvcx}bG9Ue#r~?tYWL#3*D?|MF6#!{Jssc*{CP1nQ{u`L'
    'u#C%o?oKxXw5_&GpX}DRQcGBirbYO4l<8ZWjt?z!G-juPA&0yEsu*Vo2`ZZf``RqR+5XV&l}V@iQf)^|PV~E{+T>-'
    'p5%SKjoA0^cGLX(_1`Q%-'
    '%J7T;E{bE^q2C_!FxCH3p&%HK@shdQ!r!om!fx3s7Zvelx6MhQPC!fD{)a^%ZxvFGD~$11bJ0kjw8X7ZbiA5coYyQSD9tf{N*e0a'
    'MX}M?N^^RJ<WS*u;+UpGu$TaFg{|e|pB$<=Ni96#4)+KPdG`5zB9JRNkJBw_jWayT#%%*ew<pvw3aF8gEZfbbGvRRDO;7Z5{o9dt'
    '5v5_i7dQxN2~3aFh`U%7MQPnT0+b8O|6%b0LJrN}uh}5nZT(V9ROgTlz&B!bsoNFVWma`i_2?7^>2DArf|_v~;)L;ie?()A#f%G}'
    '@8AuV3ZK4y0vMG;f2on~bgrA|LGHK1=Fh-'
    '4}PyycT2fBp>k}PsOuZJXHp(#Z$#hJt`BQ)f>~gCyrfL_rS61>Yg`tUESlxuB&_6*mZRe8@sOVS!37LB}y}K-'
    'j)GR=flcUaHQK;7fJm*(vDGDzNLg|3QiFsKmwPj;gq^Yg`;|NaUkD%Vm(%LoB8*);r4^%Boo670TEZ$yf+JdaCh_eOlWa*o_X=3F'
    'NvqKSW*U^#gby?9b2+{cj3dN8t63lxH@t7o$H3*1{mcLhlqp7R=|?=f!a3nG8^drOy~i2Q~bHm`)4>UFOd4r3eHArp~-'
    'N5Di&LxBJ~T3#MZ&WWLI!J*b&(C;2>bvgJXbw55fa>J_ruj``~b3_XC%6{+ISmx0b<au_MpQ_*P5HV6<3T2BXE&VrCwfF*82RT85'
    'Pwtqz9XqK@adZe8sTgSzFV)MT`{I!gFwT6nY-{(QuX1bx(t1Rdi=f<ER&f{yhfK_B-bLC3Ma(Hu`r^q$bKP&N%){w+eE=MM9=Lia'
    'U$we!0Jb*t58g`e7zJV~&gi7Oz#$!S~LPg7e4b46_#%oVl8W~0WJ=Fn(ew~G5A<6}RUMI)DAf0srh^V?o7k4ECRy<H-'
    'W6o31;Od2`h_I0T=vi<F6jnL1jgt)LBch&}Xf-gjMN4+-'
    'TVaRixO~~_F(BGCZt#)n`y*g~7<GWWAt5m&VEz6ry@0pMSl9wD~+V3&A3`UB<WiV0<E@sRL-'
    'Q#Ij(#%pqndQm36AEcoyD_(QBC}W<6tcRE6(!SbV1^df%N{5kWh73v9U>fQBo4NHSUB2994<OkIO0g0Ejmm%>PVb;8zdZgB+eNfE'
    '{M~*zmxHBeJG({b``UFBxAfgA$6YXA*@3!$w<Zt$We04YHvEE49bZiWl&BGDQ4D**^oH|Q+A?%H}PzFiwrrzwpZfG%iD8{-'
    'o{nUZ4wdayTV;q-;P>Hn;hl?wU9P*6LBMCpdS7MY0yZQ!~PwDE!Iie%ZXla4SgGzLU#+j!<9Abk!A6ogn$e4Q0-MOcdNeL!~o-'
    '%*<zA;5yL70c}Pwp%6_z=Wzb3tji;3uTFk7Ia<bpCbdoabH?B=qK(EJY{`N|O0W>1%5}m~j5}$X;K-'
    'FA92XP%7c<J(Vy~Ayxg2x8e<3~NLjc(8!)jYPiNn}*>*lO=I0sVz;d!p3ym|7;4`hMmC{j$^#Na|&LMe2o2I<vQvp@`&+fK(%gPQ'
    'bp|urf#^hLu4YF|3$58@g4aT`|LAOqY9%=?afAUFk8Vt31YZwa1vQ@fg#!_OX6R>id{hw_fV`OuVY4ULc9RXCEiJkfqgyt)hc?O1'
    '63?Cydy;+@@|Op~eUF-{Ylj_cAQFt}9e`#7{4@5Rg>l7}9>o;brhb3@?KhVt6q_Hs)-'
    '%!H9h@IngeI4Ym%c^jdI5x;~){6=|nRic7-9iI=nz=*~<FJt%2`>eqyRh+sGf+|v)cSGOaoLq9_BJcM=VM<p#_-'
    '6`~Al9sUko6wKDSNl3G1V`|KLg{)%!Zc84D5_?E4KgWH&x)dffP^CFM71{^Q3emhh%$H}Mievg)SRuCr0UVx9Sd?3f?4ToU1v^H?'
    '%6<mJwXQCFHI$o%n4?19>+W8i7aUj3;0W{P!HQ@$_n+cn(JAi9#*rK)Hr~R^~)@S4jb!NBz2WPEK@~i>3&XZfF~eU_(JFWL?=?ON'
    '@8gqR9>&&-e_bQG!G-'
    'opm`WsP+h3RsIw87*lL$;t7py}uiFG_t<hu&?1VKE{u|Apgk9r<3=46|lpX}dJ7u7GIzh7F>PEEZnXEMgCOn%pS-vc}47C+ac&;;'
    '3dxG!S=w5w8>Q9LD3j5;siC!V}N2%MG=#@erLv4mcuM+xW?$w_XOk$6dku(na=>)7x#g-'
    'kF=}D&6@|Vi=H2Aerr49SVr^c2i;3G_pEl<2*t|>&lSW-'
    '`h3~#oNd0XKRamS<4dF!4eLyBC(cgX)XK{C=lAZ}FpHE;FV@V>5M@sspHs#si`K13CZpQaB}#p1g35vo{RpFT<ziyP9%s1|W!`Zz'
    'Ilt;en6bQXL3lB7arx(RoOvt%S8f_^46Fhq_Mje%dQOi#%%u>FqTi#0HOgzv=~_$6~qA)3W(T!j$S*F8WL5{60A0iTd19y_?tOD0'
    '$wC(<VS;#*RGQlwYe7vDybsPt<4;yXw=lwM<Bd{^pEi*%NgH<w7d=kOQHkOkimC*!OCPVi<QXJJo$h=FiK&;#f{8`JP^g+ApstDw'
    '!<;bn6>LP0#-$J*RTP`z$WV7IbIkhAuymFdaG+OJopr{-AOKFIIK)|8*cOX5<ZXYi6j<ccL7gC<w7G4)=h|KjbwTHp-ybCMJRuxa'
    ')JNsvf=yhty1wD?m0n@F#;FUFev+~uSnk;stg;5i?aNTBQBImbvOS#|K7V<l32IYiEJ5{WdAiyOb!&jyx8^Ue;<JKLA5z*}2g&VZ'
    'E|_`BVa_Q7%)#k2-'
    '32{@zUc_|&0uutg+u||CvFNs^h@HKx>%&K$y1*qx+F=}?T)n%@>%4>VFLJ691lWq5#>~g=!uJD`eO25gj@|)~xzsaugo9tTBp+IE'
    'i)6QEyCXo-(_3PUtuA=VAP0wTO`nk}+ILF$KaLfit+*@c_R$p%g!-'
    'jm3%M$Ma>o6ansIq}V$Zkv5KBphXZqx^wYvRnj#qa_ZVofaBs4rBy(;KtZpO_E&6Z0W|Vm|Co%t!o*`KUiJAM+>X<9uRbNOUIB9F'
    'cqoHdLQwiDe9rkNsiX?6-'
    'bCW6cSffMPb9av4==TB)9P6Xey5Fo1V#NoXCzVzqe!@5_*)jk3+by9DerIXkm`azBbS@8^w3Yn*xEv;L@<d0R^M<}36DNtM5gl>V'
    'FFn<trbcQ&(Fz-IgtCb59%_FDVmOtUlplsN_9c)pI=E#P>*p2;oXc)o$TE#P>*ku+rR4WA*^Y%`8JrF(?b^JukxA+R>umnv%3W$o'
    '!|dJN0BMBWyx=5}Ph<tg=XX4kF_Tm6uOkGAG+7I4PQ>b_^&r}rPR=KieFIgK;-'
    '0p>o1s1i%eeYsvwHNW&~y_vL}(rfe<<|Uq@P?F>{7Dd+w^ZELu)C2iQT_W|N99Jze^jznWc)H<7I-'
    'k_j4M)ud6lED=kjb>T4I9IS-'
    'R<4WEOV_4!Y)0??6JmIJHhyqSdSZxALa+GK0jFJ^^ox8y#iez+qb?)K+s@}{u==?WMFjVjBcORk7JDvU+%}TMu%tc<6=f{+f!Y{@'
    'Pf1}CN?%n^aP(zWi?UTeKv&?Dkzg12Br|7y=9yF5Qp#1Xpwbbh)b#=2uO85Z$Pl3`*|b6*xb*X5a8r~-'
    'i!bk_wyD+OSqr6Vo%*cyAPAPQc+O_Y~M#wQDvCa|1L>`p~~RFbE?(7j4NNBr&D3ytfP!^<&HAOl{*TKD;F(t!Ui`XD3)|!qTys7('
    'CvfPE4W8T$ujtDqC2gn?@f%@i4XNmdDNYeC(3l53-7QrcAtyyurhYP3-'
    'GWoHs3{fSQlI1LOd*sJ>X(Itcoq9Any>wi@2GcdOcj~&vP+tgw$Vf#Wd59J}=MIz04_Jk!NZ9UhCuMl-HLrr@X%4obsNQG&X+Qsq'
    'SUcOpZyr&^p0DV<WQv(XvmwN;K8MA)H##U5S1zW8}jG4wLP6Mi?lyP*epyRO*B=leH2%4z5@_SnAJkv2>`^pOYk!(ch4_2IQublp'
    'd#;Q}T4Bk5^3TcZSlRP|R#~meQY8%-(X2(kH0MndK{UAhqrjZHS*)-'
    'cZKO@`i#l%Q;J%535GyJH)|&0!BDp6ycbK%vFX7rsUVCvVfg-'
    '57=4m0Xr)^U}vQV?5y&Doz)(&v&I8<+@yuk5%BHb!=*#zxZd&wGGv50Tj@`Qkh^;l{TJ(nzf1HR-'
    'Qlyn%qU-#1Ce((-B`wq^2Rb|ls6WfQT~^pD(i9CGiq+L{x+lFw(9SypsPDsmNpp~Q5m3gkPHtTtb>4-?aUZD-YosGTzRFPhYfpoV'
    ')RNeluE1^TGS-'
    '+)gr%vJlP1fM^7SeDq|9PQ^85(>`rZ^Gpl%}WfGPYrU^nxw};1dd&nEoE?`S`2+IU90a@ivh^oHM5XcSs8w`QmsK3b&$W8iN41wI'
    'DzoUq^e4Ak*a+9SmNu(2TLV~YKq=lKn86a4=i}SpdDt(8uxiu<%k9*V?=7w3l%o<-'
    'K@_WhfW}C~HHQrpttnuc8v&Nal+G(ug9$&iJtFWbOSa`8hzbKK2!;J!MmpHUf`iv5X_AlbZdbvv9;>3D|O5f&OdX-Aw<y?BTN_TV'
    'myjG>}b36E=(#2bai@t-x0}B2PUW_}RO#}e2XN7JGn5dd03k)#x)O-3eVak1tO25?|O4-Y-'
    '@vJ;s+mF7bj9KF?Wy~6HDL89fV(aBzb2DAxB{$QRUWzYW<<&RSwO)u1%ibMSr`{wX&2jZf$h$XFEgJIfEtEuWm5}DRam5`HiG42Q'
    'mhhCUk;fx9Sc?zuLXcn!7Out{F#spR$_X(_o9_x{7F!Yo$yNlFj`S@(-E7q9+j>UyobkJPcDA2>YZ-IKTg#X;-'
    'db?ZxL<}OWUz2w8-'
    'vannQS(Qwee`4=OIA%c?i(`9s)GqLx2`|2+#u_0<@5+WGC|WdL7cz&}zRPsY+<Ik2k~LrKfYd6F9KI*`N+xAu?;ciR7IOkuAJetz'
    '>1WT9Hc|(92x$`aB4b{g4%9Ocqy^F<D$uaI(06#>BNaQD{s&9H%)$^64%SM%S}ZNW(wVsz8CmfZRi=W2*933mb&{tmzq<ZlG4wpJ'
    'lp{%4W|78^LqIM(}*F5!4vObC>6I8o(qL2;}^bAnFRFgE>GDc?I&p93+Uc0zsY!3ns1jjA5e4Ii`CVGv1JAZ~JDVj6q{j#-'
    'On%IB1-'
    ';_b?pbHTE8fGrFPvZWUVGg|6_RXHz%PpO*C|mmU{Rsc|i=9T`ls#$Fff^iRulJrzt};2qt{yH;~mY&qDJs$whj{#+GX3D%{m*eb9'
    'sRmE0A4o+3EHDG0`ime43v-'
    'IQ{xd!fKu6Sdff$f{gGA4^<8I#4b;AC;uz(a9d+ZcF+FwsdE0d+Z|N5q27=B6vcBg1)fgpB0OA50V7aekjOBfLb)@VwMZSY$m_`U'
    'p>~T(9(zo>=+IN{=B{?|LnSzKbcp`U=Q$DG7s32Vwgp?um$OeTC%1;aQ=tay__r5jr^8kr2Iyji`&dS3!E5E8Y}ZoZ40P`>FWJVi'
    'iAGtO`yR_b;h>VOCVjgeld8b=~t=1pZDH6T;LAGt(9V{H6)s3hHr;_#GM*#fkFWG#`p1#qY89*G$}0X+m_MOj-Kt5Sel_?cdE%D7'
    'niz#HP`0o|bwMU-N?0OZm-'
    '~3s+l<n;PCMbJ$(%2T0188&1lY8%_$&4fo&U5QIHjgYH9%@72`edFTOs)h2fF#JQ3MBD`Uz<B3TRc?Q;y32l7|o|lg7%a6F3J=p>'
    '4aZ7f%dfbs6tR6RHH(7!Eu^X(w?bu=MaW}S5`#Sx31P$gDq<fwT-jcT$+xM!Bp9-'
    '!lV=B0^;8bwQ(rou*4;?i2P=~$JF5)|KyJ|4VQq7#<9WqC?ql(K;7<GH-{e2#Kf4_&`&-'
    'c*#1s;0;fQQ~MB<TG+I6^|j=r0(y5UNO?sPt$r)-cXAd$);5CHt*;SX+CaKJlZ!eaaa9?Ne~{w|{olp~bI-'
    'd44V2=hwml@AVIm*PGU0Et9Lfh2wNo_J13Pum~glF3xRGt6(?I3s5KLeFPy>iTI0#IU$?sRtMpECpu}fE5BeQH;p5?Il><aHuv%8'
    'Kn0F%=0Jja8Si~RFQ2s^v2PgzzJ1FW@a<c~m>16a<(U5ml%xrZ'
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
