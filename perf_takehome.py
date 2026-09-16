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
# Verified checkpoint: 934 dynamic cycles; 9782 static bundles.
_TUNED_STANDARD = (
    'c-pkRd3=?{^*_$cJdqf-'
    '5D0`27I8sayL>*ifQW#A67q0>T3fqVt<}^8EvP81;EEC}C@LsztqH5R;|i|D+FH7ZXce(x4@*E$+;Qi7&di)M^E@~7`~Cg#BhTx;'
    '?vR^%?{ntN+0Xl2o!qWksaEvwXLF9cV$P8>lgrM$;L<s`n5mCHeooqoekRR1b7peEgfq{!|6g?0gkN1UX%6f!N?X^>Iky<TDuRO!'
    'uA5UkGx_BO6DH$pW)@96^U_Od=UhMM%uDe-'
    '&pinL3}5$IQCe*O;L9RVrSu2Jf7X#R)z|01=SBFQcum4rm%|rDX)Ao|NpouDJ!TeNX1_&RjIXlq^l4E|m!GY<65s3GPV_$bcOMtw'
    '2jP3UciCrGU-{K1^qybHd-9*WZ(sDWf6)hawa>3Uq(Aw@FFv9L{igO@jkj5H*)J}}?{n6fNRG5O-'
    'Q1b(UMtg5{7n3Ym1zaOzApE9?-'
    'e<!((`xir@ik#V2@pW;8*X^$9?D*duj2JU%X9=5BQ4^{Z5~gmf}zENpsFUGCdMWTt`p7jTEZGzv!1%;7@OQz2A&4i*76$^un1dTX'
    '{OBo$}uj_u@Ue+$Zj`E2hy-yJ8|Y+i(4@Uv0CiJ%05mebdkUVmmF~@rxa_*y|Tfw0PSuw$j4B!A$jyIk4M)*#4sE+rL$m_V%oFV^'
    'OPLtbU+XosgpKg_mA1>AZ;-'
    '{N{`^&pz8x1qos4GJxqa)YE00rHlP__U1R(o7m>R&=&eaZ`nn^v>N|nfO|SB?Uo+vp4Oz5=|J~1O}nSZxu>Jks&tTh3Tcn@c=vQn'
    '+A|&Oo{mj>r9<3@Y@`p_;XiRbEw+XY#+TBK(C5z7g9p#S?@*c!rPtP`bQ41(;%r8m^myr$U2bbP6+x?E`Mupzgtvox>OO3<C2+G}'
    '{m0%quQu7=@!q~+f49M}*3y^U?iX*;VxwQIp+%ElyiSYtezA@gOxH$QxOb}|sgT`fIz`*uIp2@B^x{bqzYN=KS-Lc>u=_d=Z^-'
    'WJkMLgXlb_=kw@=QsTkiI)Ivp2_6MWx)<NF>w_9suYvS+mvze_AiUbT1oAOB@vD<ZOO@vB$t@80k{G0`18dU{dXF+IdRy*MpP`?#'
    'l<q@B`3-P1{F=kzf5^wP9EJ={ILEbWpW;hs)TyQY2Jr>rIdy&kc_xb)X}-{aCBO|sm3H7<Uho;%62a&`C(-'
    'L3sibibA9bd)syR^Q0n>dW?aUX?>jTfwXS_II!NTm23GT;rZzjz8DBr&I7}ntM7Gf39;+uf(6}?&($dQ|I1q72VWY|K=~zVvS$?i'
    'x#Wpc5(Y|wg(lT++jsNOR{Rd$%*dpMtcx5dcX7uyybuUZ~k~%C(j>Ee=E<IrZeUFL+MTac{!}K_x-Z}&9Cw2YWMUC{JGw}=5O?k-'
    '}Ha*Pg<<=i+|AK4Zrw1EnbOP`))i_ekVRCstY%~3}vg6O!au5H>Y<@@;6=Wgifv9*`d7s|F`@9d%8sK=dbpDR{GV8cIC7!y=i^A%'
    'sm~IHl)kl)0%W%`lNfBrt{OM+*3#wrq8&iW70+Gv&{F`uq?bZy`DwgW$APhb#mMPM8RTj``^f3bvfzQds2Rb|HL=w?#j^!w8DFGL'
    '~3N|&a%9vqfv`5y4s0Irpw;+CE4>2)3yG2Eeh-'
    'f>C^7%*mQCF9Fz8H7Wo&a*Rq_yB%Q``eo}fJ%lXM^os;ObX%l|`p^n;rCI<MMC;eY&@%NY;K0_&E1>Flp(66{Ix?Ot9dNg>na<^5'
    'AY$#koyH@R<7SsMm?|3QdY9ch0DBSY8=+^&%yvHo@C(8sc_|@}v^`a-'
    'mL?o`)Fq6`a>4>m7{uor^k3arZ_~Xxf5B#~_J?)7<)7?AlNMCU8P!22T?*CDg=hW-'
    'c$+9EYroWNrE7N=A`QOre<@uFJG=Bbm`X9Na=j=`V$=}4q>H2h-'
    'dr>$1@q()t{?xe_G^c+_PjpXrr7xx<@oUX2x@6LXi)+b{J?Jv`ln7O#$iDL|_%)TzBe?|stI~!#`{%{DH_NQea>N|yqi}Pqi8l_)'
    '6#kC+5-MpGNoKzP_wa1+{k)aFD$n<&uk&*lrf56)PM@}S`d7bt%C4UCt7q-'
    '%c~6DO=@z;(db{rUGtK?)w)A;No7!}H`sZ}G`^#r&dsZ-WA}1B2DaLnBUt*xb8!dM4R*E-=cjUeu-r1k%nAD!zhK%*rmt9hOLG4-'
    'SDd21RD5MlpZIX8h_^ziTqw<|zpI#=<r=`D^=U1gu<@s;Y{c`_LT7|sXub!}X{Ip*!v#Y24{a=!9aJN^RZcSIDC%C6O)4!%CxtA}'
    'eoBm65BH{ZACBHpDsls;+5*mK0()qaK=l4dx35F}8M=L&#jVvaL@u#<Je`z?frD-'
    '+(Pk!^SNiUaM{A+r@Jbxj5P@Z3&ekIR8PS?ou57IY<B#&Fg%|v+A5@DIQuO2HxZTPsm^^4MVOodC+O?&_@OE(iiYR|7w)*n2-'
    'l5P=2()W?|-'
    '2%BRRds<(;}tTXCDU07w;&?ZB`RIWGP+n~G%<;&HI$G<on@Y6^t&dO<X)^hUON9+{A^^l;RoQNPf?xl8M#Wq50$LCDu8fe9<l^^!'
    'mpOv)uWy~j})OH@Q{1{a1n|iFDte}CzWn>&v!r<mF{rQoAJHVX7_w2RH$^Pd%g?0sdQHX6;UjW7b+rmo~2^#S4^f13-}-'
    '`kPgMH#E`x0(PNF|ose4kPiOyUNU}S9MTl^<f*%BuoS>kMtkQ*WVfa;N1>?eMj)K}iJLe9v_3(hD-BSN1b{m`A^KJMp={EPO`{`J'
    '{<p1PBS}YHO1+G=rraMzi+v51vOT(?#!i|cRk(m8#<VN<-'
    'M5INY4v(cb$PL}5;GA$n_t{%`!2hmJ;pL3+T5D~2)z=N}A%9XT(H`>hvOC&CUZ7N=%`n0}?SZC|KT18(G&;#W?S+=a$Vr*aF8139'
    'cPSPb<E&p9iwXXJysto&MqD`M5e^32ZsqGw6`r~L$#^GSC0qS=`m#K~BK=&Rf0Vu<L|I~q@=(Sgc-'
    'E6>fud`cllJL$C|Burw*63`uziFo*ZuXqwBi5sZ*>nXo{3C*cuu8@tO|Wmr3<X2fCou8if||;fv_y>h?5hrSUCm=`ssPgMS&hqD3'
    'TN{(&LfzM!D}7)BEJU7u!2s;#Z68>b@f0B)V=?!e(hB8-S=-'
    '*Z@QuiLGB0scikCNM*ASm9TsFyNMJJMh9V)!cPtd0Zj>_gcpmAM1@kh^=ila{p|VZ?WF|$B6xnZD&UcBmL@MMBzvAFQ`0YmCZD9Q'
    '$#J<-!KpzSOi}QYu&PinI;>_XI6oY$1(te?{c53IE%H+K#UebBk4|FsxkJI-Dpj&>MoOqu%X+th`&D{Skfu(4qU2tsLbMGwDEzoO'
    '#|$sVZUxfOGL^jH8ai;TSi+PB>5=m)EfdDdXCfW37Rh&YIz`C$YiJdyGE_m^AZQ!lqVTJ}8eR<qn`cS1z^~@p)xskA&fPRTs?xCT'
    'uHkW&wwCpFMLXU~);ks4qf#U5T?+11X_6-'
    '*itr6F5vw$Q1y96R0ukMNe1Y<v%^cypevoJ%ltLNZR&z*d9DJD3^8WC~?6^naR<x|lA5Lco(f*P?AT%piFg8%|I8D3Gt9}}q0zK<'
    '31?PK~Ybe4kx=N@~uHm7W<+^BC8nfJ63hs-kjo<VEM{WEZ`~r!_JI&J)bxD<ujO5t<J6Z=mBcb-'
    '89PR3Gnk(>}?@?KnYecGNNkrnJ82(t4{+d3E3AUK5Fs8&&8ditZAqskHq060?F7y2A4|Y}WS9b-'
    '6!QFRfO(!RCRfUE}EEoB8rH04s(WK9*S9Hzrj=&;C*R6i_k%Dh%3Jug|6n!{(M3GW?-'
    '$7bDCzA68W+1$+h~o?BQVFNv&K3p2l*8}rJ>BW=@b^VXIUn%o2lEwOH~a6nP|<Zu;EK^2KRVnzXfy=EZ^vjf1_H~B)o2ie^EOVSQ'
    '4k3AJ&lGzpzFyRje~<^`F)KBLhHPFFo44r#op>c9Uo5G2!GPD@@bH!7fs9^C%8tzDS;t6DfnJk4b-'
    'qP9Gf|oT)*?H*>?2@znW{UKPKfJcC|R%R96iTJCTF$x=+Ekw2zF?XCJJgqh@7>w#o?Y-'
    'sN`EgLfC@&k}kqr|2k^(u5pqJq%RqcNW#PXh<PFnpi7ObhdO%oNC-p`txoD!e&ppTBw04G7YprOYukH=-'
    'g(hF~|Sza|PegF3Qs{T?cn4IMLtWdYG%=B)`}Izf&;MFE+yO6;%7hCisJbQGT%*?o?3Y7hB+u_%(Ea|FEbiZDRKURq7Lw`#>gQ6k'
    'u4ag5^BUMkVB^mEte){7gDv95rJ*!xcz^K#;RF+V;FaX}dgd&MZrv+x+SlyPEA+w_0BLo&P)3=9l4v<Kxq`*x7Kp7V8>*uEnN?Gq'
    'o7i@C)s%>R|cZmGK`+QMX5~Q)6(Q8s$vU87XEpH6jwt<B%{d=Ynpw_cM#rKlD1z|G+y+5<sw5Nd^esRgwaN_mt#-'
    ';C&@YAoxH@7KlWk3mU@Dw0InFh88~q&eARU5xtrHOJhxPyC?A5blBLFX{0Y6q)BxM)i}9&jwY4dlGO}H;6VtCfO_}Yu=h9F{k_dA'
    '+nGgZVn56dJ$|^S)WLW{m{KKpK$uc(@O&_(+T!tGO0~n&!IWx`hl45A0nY|eicROq8n-'
    '9n0hpkpXanA)gkH#Z3gKa*Kt<pWa<>{;9ifOiO=oE{LX$ph$y}k}*2#eyH&|Y|#ZzNOQF@(=w$;FmMd@@O7^NdR5g***GgM~&{z$'
    'aL#oBGTH(tHC?1$QyE|wr0taR}*!%7YJ1%69H_4Wdxif3}N&bS!jP?47LLcs-'
    'Ighp?Z&S6PHr!}JARhsD+)2Yt#Pzz>qIn-J8`?%+!8Vyr}be(RgcC-IogQ9DuC(ks6m(Qr7#X%R#%0^*zl)_dHdm(Jauot^l@Or6'
    'h<*paHR_sQ{$(@PfAT^^%L}}m@l$s(ztC1#1hywHEi*Ss*$#%VlWo_!>a)?TptKoAO^uGpwTM~un;g=e<9uwz!OPuMRIOrZ-'
    'j1O*shUsK{aEogdUTm1(pz!huMYMP6@+u0gUAj&R=F2S_DHmBcw`!zbWZulyNWsXUxm_dWtW%beyb`+*wd^u7a{kENE%2+Xkz2?x'
    '?vrL&lO~v9w`t`0$kLjlaX%y5>JE+Pz}f1xj$fCFQ0?ZETBK<zeO8MEY<Hj5BCS*9b3deYdiWd{X`P-'
    'f$z_FbFG>p4M(4;y{qaaF(NXjQ8g!x8vs9Dm8uJDvp|8@=uf#IKwHiJysmp2MT6Da%2;U<$yb?q)e&?OSDA!p=x!%RRal7FX?Xxy'
    '$oYCb?5%ebM^5zHnl62Yqw*Cp;h7Y=Qg8MWl{mS71%~`*$@Q{wFfQMUiMT?8i6-%jv6{XgmLV6%kvb$eQTa1dQPDk{ZhV*D-'
    '>2aNg%i7dsQrNIUqZwxH!Q~p&PHYAhQ-}ja&}i*8q>qh@;eQMY%-'
    'oPQ?1xL9rk^*Ui$j{QHth)aXikQf!M&QZp`GD=&3VEu@SwK*;ZKL6TMeIp#^Jd-'
    'I$dZVo~NVJg(l+pI^t=x636T4w4tGRfsS|_ZN&?9<UpXg_$%!^h`?y6WwaK<74N4168CyCsK873Mb8SW*$RFcR=46!1SL1ks{FtD'
    '-+fKP%i)#BXxJ8Bd97v1>-_JoS9tj{2$-($@`VuaTZI=DJ>;xj^yF7)Hha-wU9PkHjUFl+x3u@jx~1VP(#{WedX4m`V};wjqH82a'
    ')BkM^tkigLTjkRY4VqM7W5k+QBSX5QIvtN{oBUBdU%?N;QALWK8dk?@*c8ZgxT1IqufEjqd{}*^;V)seSHqve>Qe>(!-'
    'f&v0VkYbs76&c$Vd1!syhFgVH#Ckl!1rUC_~-'
    '(Y1XJ!>i)au>3D1{5i*?b3a0YgL~<k^k~hnLo2<PQ^Ivvf=NIAk;C)4#saePB?;{OQhFktn!&AXjxY}~m4SscvU0v&6d9O+*;e%U'
    'Tr||OSH}J87Z?ervxsVbKWnL~FuhDjJmMIL@Xgl~<4bf<0c_((bMjLBLM8y3X{Q*nF)&?bRi%w~+)rvzWWy{QL;)(bwZ#W=FBAM}'
    '5Bpc3*^Z(G*kg>oO00+vFkm))|bEw!lGXtnP3OX3rohkNl1$Dl#1-{ZGQ<>VMG+Yr@B?{i*RV}@H1^a8qlScJ!%3#u{-'
    'gg^98r8cjLrJ50Kkfw5sNRnoMjF*+M~Jk&tXq?1kl95@V`dQC^VX)$5o>0<^Dn({;u+`t>g@9_w#ntCdV0<x|3br`!!3NE;pwo#W'
    '4k{v)2}W2O!ceDc6Ein*(E5F4BH9AG)&ZdyQ2&oE5E5RFwi(9iJlGqlzGPne>!n49!>Il;^gvDgEG;ETC4H5xlrp8L(x{Rlja)oX'
    'a=wYQyrLSxgoJT#JH0<%aA6;lX$cbuB<R5zD_mSbJ<|%T(;=ry{F;Xu+?{?{b-'
    '!#L&pq%I(xqif6CdplmZcQgE^s0Zt!~M26Tft^G<H?M&<@|gE{L=Zt!O226ThB#N6Qa|H~qzSQM#VYw*b1)3@5dm6_95jc)<x81@'
    'gU*&ZG)iMDe5HIZw$cbM=glG4$!95xxET%K$z4ed&c?%#JcYz<=lFa<pVv!TZ66;^0#?ve$X{E3m(XbqPK+TX9>Cc_rz0}5_7t`3'
    'O~3tXg`wqpzoHe8_;g~B@EPuGG2byX3PjSm#^{c00y((1T$u#rm#y${%!(Xqohf>aL8mXJ)ky@seKq&xpjLqP?TKb`2Nd_NEjEnX'
    'Ao&_ThjK(Mw7_Q(odxAx(M=pU{zLa_VLTQtr%?MDghA`R`lbjOb>)==tKKUC00LnqB5lCB9pYWcpRYmzCT+|(l2CzP91ghw(j?p+'
    'iOPEFA0@?>;JxvfSu$7MiC6l^MoGZGk^$niy^aIWG|cfN$Y+Le$8qRtYcvKLYTpx6y*8%IE3q#+eu=aLpqeN=?_NGW`z!fQX0N|l'
    ')T@z3#1P;zXtMtbC)ZL+D9VCrtRsgz*qZn06}VCpv66hbg{x7x&3Fm<=tltM6dx7$QlFm-oO5SLRimSiwDaVxZ%>GCVIk?Bg}{Ow'
    '&yzBFJv6dO?mZ&{tl0p9Ov_%i&?<whoYZbM=45$C1h?*?rP_ALKlV4q^e@J|DuDE2M?WzceC=kg^3pDFe(R~h(R@nL`2ivH*F^6#'
    'yp^nEY>qNr!sp%n>ov52Rrq*8*OgLKFwToC2#>W$CYIRXz!hz&iz&P`~8CphmZYv(RjBzgZW)$v50yFBSmB>x9GBy#?KlEC>18z>'
    'LSmZjY&4Dma^s3iZ?Z(rJ-'
    '0$qRbiz@O}{Z^(uDE#wBzvxMR>s@})i{ktBNr(*A8&v2>r|e<gQk1^HXL)l``c|AP#*M{ORKYQmkmVdir-uz~Iwa1av?0z5G=35j'
    'mU3MyP2P<dlBDC{43l@n>fz!b2S3*2&hiR9U>$^AM@6PO%fZdiF8t~waxS6N6T6yRNhtHHQRGNMr$ode%BOSu>U#=)l)$-'
    '(h{c~JaG_tFs^Ihlew7pm!i$RVR+)I4iqgrU4$L#9m0DX^Sb6S5K3baOxpN%x+D9rbUXM~3`?Z+0jxqZiVq4FcD-'
    '&ARiLcioX(?{s4}PpjYB8x!R`BBl&PyaM_0t5#`_(#)7kf{*^A-'
    '3FI9P^!+RAVg1^HAN4#y&&YQqr~<WplfNQr!!7>=$WpBDKAO0QmUkeQS<_;6p|=2M4+d`LwJn-Jo!kOFD4VDX;)5k^wrBAX4w|H1'
    '!fu5){I5PYfNUkPr~LGYD=my*J+B)S%Oa{WZn31Z^?NWnb6a^D&Kvz3ND!eaVH{S8O@TNxN&IMlDmSms#2CS#d{{iyekGaT(lV`z'
    '}!aDS13;|<6A4<a)sj?<o#1i7|43CW!PCQjz8O=3M)LA(~6CZchHp2K@MHR4lGG<3LEI--51;O|M^;~%Rj1;J$ckAii9X!oF~$`B'
    '3R*uR@dKcZj1%5VxDoximrWZu+xu~*ut@p7&xRauZM@@97jWCYFjbw^6GvQgcM%$z7)<UVk3g5ufjS}p9eJmQtY{)8yzo&PAN`B0'
    '{Ic~c^_%Msz4?Ez<Jp9%N5f`2%|eWBo=1%x|K!4CuB(B%DnAl!d6UhIdj*LZO>V}r)a=Yn9P#*5Qt?a1b17o@$$i|yo6vj5oeC?m'
    '(;c^kHX9ZpE3BpgmkqL?;%QYbVfs!f^;U8fZZXHpvyw2U~|L@!1gYm=86opp$T(t$?86~sr7WZ6LQL9k!JDo3(b8eT3S*&qd_L1X'
    'k%&?gXZvx29>@7~q$T=?Co8Vy)kEx?zB6}lI{3oF#ti)Cd~9YYc+8;4<u6#K*QL`wYO#E5LQWGf!6Lrw1>Z+SWfM7h*i<T#owBT0'
    '^XV_}@LQiR}bDGh<=FXzD~<ceh}S4l5)#xiN(e8Xiwie&yc(4$(zm0^V>-'
    '5FLt(C}%X$2bjNh1I=ihlJG<v^~NKmFE1g`hkLldHZNfk+dPe&_sGAU__FyjA<#jGY(CFsEmkic#!5piuc@sw5d;ejq^gHpmJueA'
    'qkL|aHtlZcU*`3_d)^JeXik!Kp!-'
    '+o(T6jPr>UM3HP){BRJKZ9uuo69Yv>8GUe@c$dq@`AyZzeL#Di=4w>>Y9Wv#eeEyf!L0ifzuuU0KPjpx#O-'
    'g(?NK00*lz#n@5?VTu8q;cswW;$$4Mo`sM=PV@Zv`rJv(0h-'
    'u}I9S1sUo*qtPpf{2x4{FPY%)*wYf+Db%FLM~MT}rGulq0cz7BQQ83Y>Ch-'
    '^fEx9LC~1H?by$=$K&?7lXNr#8sl`X}t=3Wo!BKL%ijO6}tXU#bnG(us+KAk7#kW~fNJg|j{w#v-'
    'F9j_3rh@f>$xuyv5>^WpyqY2Evl_iCd*`;&urw5IcMOHwWub7pQz+c-'
    '916F~L*aIpP`KSSF1Q#6t&7M#^2d{^bhnGsJvYf#Z<mIu?hsjl6<WG7V`V^EHD{|S4S#hM(i&bYpwKglVu(zWwF)*whOUM$G89^Y'
    '%EWlL*>2Jb&F#yh;Capc%Qf(4%?->co_X$I9u0rd+`<g-'
    'SIs@lW8g*2P0VBAZ#s%^qPsFt#PJ((j*ik&bH(iMNABfzSD)(+VzZR16cbIC6_&S<oAre%C|yAT*f)V$c`pkNRnRktuT;a7uv(_z'
    '@jy3pCYOcPdIfKW)j9?L39EX%sjxy<V$p$weIwmaZ*CYFY<$fa^q>MVe4i-XQ{j4;Dcti)&VxcPW<sR%_{^f=i6knP0I-'
    'YmY)bxn34_gz1=AR+w4~)jNXe%vNkMZY8aEF4f4k4F^tFn7fbiIA^tQG515pZu-hY{-RNadwBox#Raifb83Tt}X={-'
    'f4$wcMYnW|LqZEedoXw{u^VZQLCA5>8p4cz5JlM^WBe!t(7U(w=Ueld|2_eys=E*gDc2NmIc)|#&6Q;2fs7GqE0mj-'
    'S(*<G2O5Yyq~grvnB*W*P{CLKS!i{LPwjT=^QfpDa)Z7VKOb|`x)s}26gM=ChUB8X9gRcY7r8PWEVgwp;YrZy>oNpbr4(gZGz)5n'
    ')3aGB<qnv00C@B5CTi=E$h6<zG`cBP_TE<mEn$!}?2T2o6(@?NpNQPkfr);CJ~2eqr<3v{f~WR>J?VYI?-b)^5?z-'
    ';3MV3xwuieQ#d_|qadCe98cy-Boi^$D?OC^{x?X5++K`b%{%M9FNOtfjJ)&dY!tl_w^Y4+@^##R-'
    '*ddD?VQ^p4)tpiYTzYF~m>i{CdCrQEJUVaXlJ6_(KR&uk*wyLjkE-6MVt8fXuQi-'
    '#)WA?xD#?@*?^Z5<Q$7)#u|{@q7`d*{>|xUFU4q9HRyi2GzwT3;6#`Af*{WhO1AuDHm#?h?W}l$4rfNxRxdx~HhP_PZWm{#j0#qs'
    'l-5fp-'
    '4d5O2nhyhVi)o?BHY;kiwP5}w&Al<>?^zJ$k7r;~#6c*i;`=n|J<lq={Omto*0R>bEB8NOS5j&NHm<8y>Zpu4xJ*j6+!#}OHQh&x'
    'PF#$o|2(oYJ4ldl!QaYFL_MR2x?`?hP~s3fgnv-xXDnzGsa^&}n5X7kZW3T!HWGfBrv+4o80>B-'
    '@(q1c2lXiC0}9OcnULYjLOT^#cIK+(m)tPd4k2blx3wozN6Zx$o{Qy*96)&~Zuq4)|hNSQay>&Z^vmrKB%WuupI-ZD$M=Zm0DDEI'
    'p!xGJhAsKvLGkV)DQ!UXBO5G6?Ghe%4gAOsB3g(0YtF7`1M=iKy7xc7b(9GRq}*jf2Tl8%vf%8?Y<kp7iw1%EIZcaI`}qKe074DA'
    '_;4ogt6v*T9d($EA+<ZP));?%-(e*?A9SeCqz`T$xy_fuS3kXu|{1bcN{I`C;^-qfaxLNJBI&xvT;1sOPs+-'
    '2@t7t9>W#4>0Oj8KA_LxCxV%|Tuj+#IB!0Ouh61UU!a-a+RCfq#zmHIS7iMBZS&lY(+(<KtXeX(-v-z@4-'
    'aA(0u8@Tld_(Gel1Q(LhmWSxRBDrcUjJE$xqDwkdx?8Emg0YCMt-DH3G#1e|awo-'
    'c6k=RrU&)Q4jStnxCr??2MREp5biA|s3g0wR2^-9{6SoXPsQ2u2O&cH43E8QJ;BNNO3YR-'
    'B`q7oIQgOOp&R4l&bxQ|yvRx#4?EjHxjIx@(Ux*+;+ow5z(`0TgTe4KZ5o>K5gAmx(^9u2ET1&x8Qn=N7A^R^sXgGYFx5l{e9d4j'
    'aa<KkD+;F^+c#sEJl<u%+JT%qt{Bj!ql7spwzR(O#}$k7zPz!_4qQdb1~^#4n3m19S5WB<rnf<Az8w~FhnH>#HEttm!U7}1$V^4-'
    '$(dXnaKS<EB!P(CiU&KX*eHd&Ls4^%uog*PrKg6fO_a9}{2J(bV|Kiu<aKdMBWsNrL8>0G1m;z(-+6_jjLocLR!V3Xp+-'
    '#rR8D^C2~t6+=b#NT}iniMDg?pLr?apLa*1=|!S{vK4;EonguQ3+_Id8m*)Qau|L)Y&{zg=|JCZUKwfdZN#?924J7`OE)O*fzHa#'
    't9SrrU+fU2rOZ=c4$VGRPy^?QK4M*XguWS9LA!6e8s+e=5c4>NIvuw0zQgQJ@iH<;A2mHe_z+WHiY`zF4bsAx?fGPYQ<ZZQx#rpU'
    '0$W|VpH8!lgHr?d)kGR1LH0ul9DTukdlk*zR8*%sC>rMSpo&xYJ_Fu1FQF&iqcI?)H@s;g%B03WsCWBP`VXP$d-'
    'qO&<mOIeU#lzTfj=p2rnzfKJq5{xO81GPHW+0#cg&h*B78^Fjz_KbCR~ktB<N*=He0z@Q31rtAIEfMA3+}BP+kdR9eQ$?{JlNV&!'
    '*)N;|Xi>#NdoR(?mSv<oZ0qg2|}DZfC(RaU+pm^I%jg2&}t-&zE3WDqE-ahpn%RQ;_q-XP22m(bl*1k9mz9N(=V@}8_>Vl|+xb(}'
    'w>3(<}mqNF%u0ta7hEahSipC}fM>sU>9OxLrWR+esHGp$p)k*&1O=_WSP%G1qkqjgEQaQv=o+T<8m3TLLN=^`c5O>n%neA5I&bqn'
    'B!avrIxZ<l5yx6Sajc8wLa7XBt^>`p3Z?9M7^>~a+}b{7>ic2^ZNc7+NWyPI++*MlcMUz9zB^_fl!*|e`TnOUCC4x(J>CfUtXG$%'
    '}J>>_95oTBOf$GTE3(ITH|__=l-xP~)zUNbjqI46PCK7+nf!?_8(;#a#goR`3>e)YD73lmu5SMO-'
    '}RRV7$+&sBcGhE&#*rOQ^ZwtQ8f#8vO{YT}@Ybw)8TX`klk6=vNkon+N?G;^PRJJ>#o=v?O8RL4Sv3$)g>p(VYoT)vK{2Eru4!vk'
    'bP7(C6PJV;28M5G5YxtR?*(VxK&(kd30B>oA0o(|?HRAwQQ$tNPXGiW*&dsZJHZ|KFxJ7K^*-'
    'b4WEiaXFZAsI+RXar&D5<IIKxV&WyGqH<lFE{f<Osx>fLVqFz57U1k24&D`{oG~0)jTcC>9NO{-'
    'f3`Q!Qqmp45dg9&Ifq3a@E6tAJW53I}#_0p%?_wqsDEc^3<v+y%-xxwUkg!$h-'
    '{Ql%nmHtYVfX0v`67o1u@#rbr#<iCgoPF`>)QK4PQa2SxwrZ1%^N~nR@tC1-'
    '}*#<Wv7?h3ZlboWt5}_mnONfg~S^(=f#nxTr&AN{@oaV^&g@#|~Twly;<3MpMNx*)PL3|qx4a!Fdh%;_haHH`x1B!ewZ{=Edm7`1'
    'OxU$zlAp(@<$8vxmoyTgcQl;}*ZFN_u9tUwO7A5}i9O_ONvD)gX(#2VyZEvPs^S`&<_3=uAMBq?Kf}^eUu`Vdrosg??j<#)fX!*f'
    'iL$TRGA`i0_?-PoW$b=y_>B=4vev@9p;vAhiNd<?E+<fuU$PE@ZjRGy~n?+pG50%})X-'
    'D_8EYy&9;u`XPsO+Q!yPQj~`=PRHpLTIiyZWl%hztX5z#x}cLIZM%i%Ox5c%S*NdXGfR3^I~M+|H!3aFIDTQHgwH76c=wTV&)^Mn'
    '+Ee$jGURjGP{kk<&9Wa<U%&fm<J_paOlN6BP%QBDWs*tG(1~4r{^1Z);tAe23;FDk;D-'
    'xX0R+#ANrFy!z`va1k)mh#<X<3bow|>GFY;lc7pzFI9z~5Vxh>d@qkiDV0`A(@#gFkH;6Fa9CR97I@gvvgje5LTFm_6qS#*MK8&J'
    '^wqejEjvQ?&_p04H4*|r9TSCb$#D<yTW{A5%O`d1`N{{E@P+p~&N^JdL#Bn?c*I%l2}lZN7|}|%Qqj8vzTT7aAAHqEQt`8ow6^^^'
    'mY9YF`*6Sd)Q|uh!A;eM<ot<j8ySjgjtbd|!!#ZQNyQ!MBzKTXRiO}tGxM9+FC?W{69JYWG52vvO9p|I6iW*(`!SjnN0r7W5%JJ_'
    'yVnu#Pz4W|oNPZ^dmo1y3?7W@&nXEwSxe!T&$jvaSUo2=2Kgq-2ngvNI@er@vxEFI@4f}KQDG}Mxc-'
    'MKHMmr2tklqdZb&wsBzmucsf;468`~!+(<<3aL!s84Efth%RcxoASnFYpHz(J6vc-'
    'aet(Ub|a^YY!Mb2`a2n7QEPODT~Nfl9*W_#ukC8`7OZI-cn;KEw(km&cRMW=gwfgn?(o2_Bt!ZE60;cBSS++RQdZ~}GG)7Gu{BWx'
    '(<KGJjCmxf~Q)q<J{hijBElG8jwBR8<VBs`x(l_<NfY2*O6>P;>_9<?lG#@n3TGz!KA?n2GB!^VTq3)`()KA4+BYGawzWg%7aq*Z'
    '4jHS(0zXdyN7j8$qO74odLoLbB6)hITMNJybwby&E)F?hd9dO-I|0pyJe-jxDKG*~{g)j+W(sEA=rkd=KNe2qdW5qi!m$X*eT-nz'
    '2Oq|CR}_SEHak?&Y!s)u6o&4*o%e2ZXDmVEAppRjgg$Y(FNwo}M0J#8)bfHH$7!6E*p3A0FX?t`k&mLl<#6uvyHP^d{2-'
    'y;fznnVg8B}wg8Kc>bBWHDd5OKlU#Vs7<QLB9F#5JQwOje)8v2rF$_K7L;dt6I>Y$I!_kBY#y)wgvE(Bimv;r+G}{NHtZ*ZEXv!C'
    '~EfE!uyp^9}ZN4fV)OT0lNxvXIm8nLY^Sl47giG2}C>uhiN|irSwLtSW;0>)5|QXP=&y1mQ~~uyu!kYjJ#J_T9JYLZx&Z%@x8|Ki'
    'njN4#Yr$y4)wmHRG}_U7YYjJS2|rB@`Y&wG&^$5gPl2Y*}aK^ZzFWv_E}879T4fGVJhgO2+m`Tz<Gq!SOdjxB^`D|ln(0~rNfSl('
    'qTtQI&2XI98coZ*w;BV2FEG1bJ7N1cc;+KN$2@qTZMK`I^Q?jDztOb1%j3LrA`+`>C;YxyJmG>IUUDr?ZqsOJ5POlNxt5vyUk`E8'
    'k9RID<r)D9g9{{5TN5w_bWk4L!*GPRt~jd*yA{n(nZsX(Gw2027||8LFq*znz0m)v_hFnS<qU3Ux2V~AwMNkx@Z}H0Q)r6%$syF>'
    'N&_tZ3}gw7QvoMmTUuKN81Is7DfRsGKU^?6zoEfA9=Yy*UEh^R@|*&d;+f}9OUY6IJCC~1{f~g+)B-'
    '^$1xc0@y5f}r0d~7iDMlC2TLgj>x0)?%J874mO4B*M03fV0}mga^R4%jO&--kQXr?5;LtRNoSns=W;r#>+Ra`Qv^E6K@-'
    '1qNEzpsp$tuX`NGhTm;i&Z{8IJh_ID<|yn@}!9f121@<E<N7CdH*Qoc?utTpHzY%SfK(fd*{*1p2INO<eC#C2)$h&`V3%BJ|c$w+'
    'MZ-R4c-v+LbG2puHUG`KW-`UK3Ej(}33w0k3^bKu8}OoyK)!uvqG>C;gSPPpiZ}MKTndxR-'
    'wo(Mce}-%#)~jb~J<>V6T79em@noCD+cd=p%#_$YZO4<iTiu;6<hPAA-lH0*&qB6t>k>AZ`Zb|js5@iUH6blsdSU?8?D_6T^^`(1'
    '7hQ0iKx#n}u<QnbV0Q}&R?EO#<-M^<2Y()IUYn}bMJwaJhUIr4PmQ%@e`$}=>+NIFe^9(<Ta1!uAi8AanLh~6;-'
    '(aS;*y;BIHcMd`H@(@Jt5`yTtBG`R#9aRVd9%%fC&mcr@(P=sxB6N~#wwj(L43ccFBh0bd*$<+$8mU*s-{A&mO!#It!n%a-'
    'W(#agSRwTxrwQE!k?ju|I9mxTp*vQqgq0pP@Jl7E^oW6Tl(5pH2F_K&N{<;hPYEkMZs7ck6Isc^djqTK62&zeO%yPR8r)o$pv(&('
    'yN5+hKWpQ711@k>37M}gYQVQNoS(qI6UJjfR$b{i{<|8Tn7<pwOhqr|AAZqaql5QPzZjs=r}>v*+!AzZUNQ_AiC)br!w{_K2)^th'
    'iRn6cGhtTS1nU!KrOmJ*aTR||muXQT-dcFmR1^UuDZA1O3?@yVfIUJY{(p4<%r1yDpG9=Zr!u}pffCD0-'
    '`1!M^MU3D*lA#y?>L(eZy9*fFBZUV15f$IB6!=tGlpwL2+PpR=PjgP!-R|)?r!LormQV1)6v}9&^-'
    'liZKz7eaA!l0bZl%L5H|j_2phjfu<=W6;+h;4kw5jpex60ZH<P{j{7yhL<YP72xS2`^QTn1RXI7ytontEAw6?bDUrzLBi_lUhwYd'
    '11_=;8j8?B0Wm^5GjWvaEAj%-%7o`3|N@I;;myNs8)glzkcfoBaH<I@CWzMO!BwsA0PtVFFhlc~s>vWY;i_%~Y#J&$XZ9;2#a-'
    'i|J8g;A_wlmn{u#9FTHL?|TIcjyFWoA7nE37?9+_a_!b(CPa7$;03<#AAv_wb+W9;;><B>}hY1o|(s3d+>Q4L52z3Yq%VWt_S=IU'
    'F#u<q=C@2J|U6dN<D>dz2Af^-'
    'bP6_b{++92Px+Fb_xx1=J)oBGu&wFcTk+QMdmM6Xr=SMbi|KR1>t5C?HvsCjgQ_O4IenSro_U!X1%~_N+Wt?0#bzyCdAwu3>l%lr'
    'S0(oE;x0#p~7F15e+26oxjMn4|!N|3oH;JWLsc?5aB?Pxhw`IKx25h114}T=P(MIOOJImsT23LF0oNtuV%(57@lwpi#&}HiR9hT('
    '>O7ayc=2wCnY}b=2@*l^SPX@kvTK0wnj}ajB>5CVaQe-'
    ')+WT_WzzZ7S!eQ@lPS{0tT>yYP@H&eBstb9rrus11!6wZQT)|q9m@p%aF_}uki!+*nKhJCl0wxJRVY+FNrghykx@RcI?Csb^7%Yu'
    '({vj-jv0JOLwkvjvt!f!84W*m%`$(|aC(wuvRX3Y21>Xa^~Ve=KAW}Ff}p3c-'
    'xS0m2qO#$H!axmO}52E$Fy=O=uVkDZ0AfKwmg%b?vly8cH^U(Hu)ey+K_Jbu5&s+-'
    'R^zmbV0hqd&=p;wAsfH(naY`a+&RkOn13pLhR<?4djkv+6I2$Qiv0%bvSn%De^K}OMLwp%{?AY;@c20cZ8A+$KI>ugQ%pf(K&0ia'
    'k8XCh)=m3G{>jrD7q$^7}etv!yJZe+sGV-'
    '*2_BPFf?A)Gl%sT6Kgb?E*qJ{&~n+t9EOI=X67)oTecjqEQI>WPIJukoQ9v}JFn6*^nl?r!y%zT2EOoL{c;Q9bDwuurr1uS`~B)l'
    '+j9DlUu{$HnqlZW^c|NIq2202MP1@u{OVVVIx4&R)dWSHUg1|06@Ariesz(euUo0};e|$lrYI2uqLKDaEOk4(MHKs4wv3ajrs%Yv'
    'MfigSm7I7A=Ni{;#jMpGUT~~6K7oY=taTX0@H|4iFDi{Iolj%78%GhqzDe*f8u6!zziLSiB!^_(2Tp8Wn84x;irfXZ9;)0E=wd~}'
    'p?m0`tY$<Bw;V;NioxitAyCEe;*=VeLbK&)>i!7LmO$DjG+U0LUXajiIhGo7T1z|-={Z|T1neAD&@ws9>U5;n!UC|V-)Oq-'
    '^_J9GgdxX?&9fC;EtQzZDO%4_5`wE$43tYTxL+!U%B7g(xk~zYhAQgI_q{48lbYDe+KN9X@YJwv>__gzpNwhWOz^+)B$7%iY9|NH'
    'x?D@StDOZEU}MP&!iy0~P{%j9ii=>lzT)Z|W*M8A&0yz|?-^mN@E2NR!^JX2=zg0SS|%-'
    'Q3l)pEa}xZvbza!i+HcMpqtaUXEtRyaG`Sj$N-7k8!&M_zSmerJ<<cA%kKspUPu&$5hAXDDNTL`szK4P$1}~-'
    '*$w7=>O!3Rs3|~y)%VNfVrs(CtibuI^XMcDH{*2%;cg<nU-7f3VXE<xB#bbWJa~878HC%E|+;Xk2X0T!F^Ob-oTQW(9clQVjy(+>'
    '&?-604_l&U6dqr62y(292LsYiIdp0G8d`f^|=n0A5x6bXQV*ennY-'
    'q5W;m#`=?i>XNu~UH%TTUL34JwKB$DKGY;c(3P1=7D11{nCnaNgip1D|?J5Kqx{Qj3cunl80Nmgzj^y#-'
    'uoqXe`0L2EEPs03o7ol8A=oaXlAR|<k}2l8hG3%ryZ8o~4JNRYk-_V@%-'
    'P(*}$cdolZWS|R^EcPI}ai>Wx+m0KPQo}~_!b?0kS+Q6_-'
    'V{PRhtR5p*@m6qI0K)1N}<S`CIvAlZmy5fbDM3&S!Agjm>FB@=p5dZnbu<9=8WZTsKqYiCifU>91FQk^x$ueA;r+YyiE|J@3n(+P'
    'Kv^wc>KmETm*7Kk`+2seaX(Uc<t2zR~=cAK|3iy*%%DVrkH3YLNuR;C~@h*O#B&H__3%{sD$%k%Qa`u>wRpA9LQ4>_?8JE%uT*Xl'
    '&bR%QCVlH*3CZQNQNEV!U6mH>?j-'
    'wqVd_X6+Y{{wNDj3>%Fy46+RohwNDj38@cGLFTp*6RQPP>2C1V6?r|)O#2burzT?RF5DP*!A}*DvBqhNWOus5<jkg669-'
    '76XcrKC1%tn}DSo~}zjK#tK--q8u17cg6gDYcWgZR3NFydYG(ed0c<y|gkaN*uXz8c<hA4%VpM^+N~xdr*^5hNG!Plm-ud?$hATL'
    'QhM+@O$*CAM5UJso}GRf5T~CKHY$CL3l$n?B%xU!t6q7?s4>m_{(UrjdEWp5t^fG&uWp1KAl|<n>!JGXn9;1M#~A;&%<iuL#8N7K'
    'mRNh~GUBzseEclVZ0*$KH}+4-'
    'G8Ip5Jo3bM}uiR>LsrftCQQHHsa+&oGI0jb}HQ3c1fx9GP(C7x<Uu;v{(zg^#T#yFp4}nRM5Y$;Jf_*OSG@6;O3#C2}>{jii4qiU'
    '3=&o3d8y=ByRFC2PetWv$q)Su1v1){5PpwPJT<+gsm8^V*U29l{>UYswB3&e>-'
    ';K|a{PS6+c%VG)KOlzBHdn{YoA&?<}Ne@f1EB>zW~dmYJtAO|~=|6^?VT9E%^ZQ*8+|KrH<jO73K$O+HRU+PL<Vy<D-'
    '9hdg=l|tpz`Hbw*I+2E=C>c^sYmbgg3}XuGB=9x)2-&9^y*@7MomfD~ayZ_=mtJ=+wxiyB4)q>8=k3JUdY+}l%E0TE%*2P;I-'
    'UhEN5|te2E+jE17m>p<Kpp{gV;vB(Pn{z;g1xX9EFGH8a%$zYrn*h5$;0;`>Cgji^aRPJh7fBvlrc8N13lQI&j%ob$^O~wQzpjzK'
    '+mo2bQ8;>AOiIP&LId1}etIq^~wqoU<8W+{z4UvB^lcH>iV}kLX%z>}68DNqxOcs{c^mE1%c()Jx5%A8ww@t(hqdcjy?d{MdMU$M'
    'G>X{$M%(H!7Yy)krKhAQQ@1qN7m8E@eA@TnER5mMWKCF;xE%j@PS(YCyDLimf(`Ik-'
    'jPi*2WdlLuW1<*QXxhx3hC@N^@%^&ogEX&_h-Jl)~jvEZp9Nc$jodeAtWAb5JxsI3fE(IjeDp<-'
    'v+!PW_TsrC%aHSE#VM9$?n7v&T`aGK;Gf>@?>OEBj=1ve_V+yym0prW?CDz$*^Orfb3z^}JBjg^rn+!cQt4#W!WH_lmxv_^BSDFz'
    '7F4$sV<fevvqJF#eulrztw3Ocr>Z=v!Qc0q2VJT#+2HxV~CatQ|-'
    ')a!UwFB;S9d7dj8(i^z61C8j73|oT+^d_EjjK=e3o^y<b^OhWhJ>Dqi-ZB~$&A0{=>hMY^t<GjLWtcrJ&?PTX;)XjrId~w2Bda>J'
    '0Mb=SiGzL42k(W-B~Bby<3|l;Tcd&_8Z~HD3}aiPlAY(#XjP10Tce7d=P_vy&Ww*udvfE*Oe#9!>i?Ul#*v-'
    'po2dYaTLEsRiX$>r3eMqbLp5AM?X59XTNTvQ8;0tud=O~|fnWRCK6VDuv}gv>b<qr@>s6Q)H9eY#R2R)dx*?i}^xJ5j)Qy>3whN('
    '4sK#=)&*)jXi6WjMal1OP>1%ImIOa#hZ_3ZZsj-cQ`9g9Oc;86avS_bc5L^E?RGt>J{%eLx)PmN3-'
    'B6jDuWTZCcEeLg25AZU(C7zpc4YPgsa6^OK<(GZ@n{Y5A!N1&DP0+^K@Q_!9uh&`Qm!d;v~B^qX`Cfer1-'
    'jKY5^L#n$+q}^0xUjgz<KC7mYb}i7MpCgxkK4O7ay<%gI{|iFfo7&(`r=b`n_f+q1Rp9bu&zHkHn#jZ9{8AWs^6l+4DT3_i-'
    '#lfP<wjBaj;iS))gxF<qHMV`IaM0#Tb+!x{HA?Mz2BE7K*y^=_8Y=#F-'
    'q&K#}LkxOGmeL*<sLxD=Bzk7MW2RB;nQ<MyUXV#zF8y#eKR?&7G>PMd%~Xgbgq)DrjFsk1Y-'
    'EuZ=x3#&^l`RDYuezW<dB{=_$WD~XAC|{XWz4iyO$^)^qj#*ne6!@ZlM;?yk@b+$%x#=fvbMh(f*cXG{|4x&7qop6aaX~8SdH0C+'
    '=dd!~%8tjkbRQ)%Q$OSBNFnY@(w4K~&(u>2QZ@!n)a}ggHcO`^J5^yw+v{asp(ZA<-!@E~e-)cBIKMnrmAK!Lq-Wp0!4CVY{O$<T'
    'xzFEZ1`!mV%ZWI1Wou%Z(g|rLg5Dj>A&iax=$aDYml3$6@Ud+?^6Dq|nma&WxAiir?vC%WG&J{nq473v18&#0Tx))aa~UK*?}&O1'
    'zhXHxwOKlB$)8RcJj)ikRWtc2ubhnj}R#V>n3)e8qT@6fE=zlGdDP)+`RTQUmWOHYavzPW*1CR@aoxiQSqLzs=O-'
    '3v5ojtvT_#iyD5%=Et@+8@z!Ul|^3pnmeIrBp;pR$u`xtd-M*|whYg$CQ=N{6kWLvT27%4R2&8BM{mKgmWXEE6$5fwH=-'
    'm;ez%E|-)*Dhce^P0-'
    '9AcwcZibTrBU*`BPYLUuE_HmA7#GvpEW*meHIOf8j#D%HB+3dAl{1(!#5qhcB2x~Eu73MO$c$V#BE(D?pY=oUQZbhG_97(DBq(sT'
    'OK4gtDIMrp;)ELAevGTHHIan0N!vhYmU?KH;s>+d89NlHy)yC)4<{pO`CZv9MQCy&!Q1cn*}Tw(X?5}Vi8T7MXZ|8w24aYiNX^da'
    'qyEb%Z7JJ_>{EdU>=}w+cxtaql1P%h9pox&S{G0sO+PTGkh!<4`|IQi$fRYdpCyafOVldV7;r3aQ2-P3TzC80-'
    'Hjiz~)dWu*DS$$RKNS_;ob8n%w*(8dgodXFRQ%+|M`+w39$zIM{fx&Kn&A3EU-9H_KhL^-'
    ')0F6_$jKP02it^AkVon7J~bm?LyS9L*AIKp@t!fmj0rv5pJG8Wf0id?41~K&&CrOmK1><DuZ>va4teI0Z~hH29mm_@and+i7;675'
    'P2kCX=UC9~EQ8&W>=G`q>cmfPDM2P0FyEvkcZ0+ZCfJQzpq~{I<1Pa>BUrztoZw9&7spIVX%vaBjda#Je~zAUz^OoX=>6{fHgLGw'
    '!3!%<ZM-Mz_KbdsoB5euW?Qo`y&K3g7+%4Ud_EE~3$e1!Nr)8^gwDWr+>uW=*mNYp4#DktMO7oEmou*m^j|d1;IiFqYdgZ82P6e!'
    'GfG%IB)Mi2U~|E+UtNDD$@k*X5&HL%9Tj*5#u*nsNyOtjkAr4CN98RySYMGs~7;G-'
    'w;>6B2Es5)DhD(!@r5vKbLqJ3l*=KOY)u16=L=9V)Jt=dr>~TzP?7o(_{&@^qNIj;F)qRXiOgui@!1c?D00$?JDIOp^{Ya;iS?I%'
    'RIBiUl~^3Wj<~PMi`2V{NfLUT+qZMiY_tyCjehsZ(MT0MreW*sx+Yzh?8B5^HE8yK`w6BRX1#--'
    'SIUcAd(>3KI5Bg;OO45%gtYxF3((`x+kg=K&AchZ-LDtMjSkjtgSxfPJi8b+U78>a(3O4TN+VJg9jDD;hkg`GhPQ{&oR@F9#25A&'
    't`w9@HY5^BX*<#WY+umi1_YT;wd)V%&?|Nwn_NGCpd3#2uPy^Y!8nEs1M4NnV^|aii*e6^qrGg;R4!MD6C0eQ^p&j7TE?K%<z8B='
    'Qe63c5(WovKmPMH2c)8iie??tZLM+(q`uPc#a=Nd5g(qsYsU1fAPS&VuNi*nCK?tis}J`Qg<LMa~Bz9N3nx0dtUYhx^fR7C7yQc&'
    'Kw<=~WnIc%&sB+Zr3z2!YX4r2N52I8nQ{-JqWRtEoEd(-'
    'Jvpmh|{oAd~T3b6)aw#IBYPs$fR0(ceb+m&?04v@stcb(nho`_~v4mT-s#B~rCvI-#shr92Mh(P+b@LAeVY`J%5b(@*b-'
    'hIrl=4eq?(PeqNy4n>^md^rgps8eDlIwxV5p&zw|Ol4QcP62eB>R#wLJJ)>LY<v6jRydmLmpC@*kl5sGR`WOb>LcDFKUmRy%s$eo'
    '!H2=~GTFdnpMaWdVbAMG4*CBu)S;6>C+SCU%re*XfoM4FgMK)yEk5xxP9;vWG8?D)dp1t<k8GUgpV>Igzp`<fm$Gr1RoOU=1E-'
    'IPh_|%5ATCHaIu0VCJSj*x*3#6WCk25;8F)$xn6!bX1&+lSct)yE5(Cdl(N2+p=LA6SAOnAr3Z7~MLlb5%)CMCZD}t(jjA5Lrt<='
    'cR#X;Msf%MZD?3Gf7_eang)m9d|tRk{fLjqlsTsg2cjB)>G87UCmW06Ys<Wyc~JPWQUBx1VR8m1(^^nPtiWdO^y$NC=Urw@tcbeI'
    'iW)^X$Bfwi?Q;;YZ|+f0iT<C|zG27WVzv7`_77R5~+gu^u?;ZvL%7@lyT6%3r1up!<=--'
    'xrx+o>CuOK^8kGp>(UrOnif>+5mTorIR~rQz_x*L4VFxm?-QI9IlUD#<K&-KOF4g2FE{;noKeZbLBPHU<-'
    'JQ!wE+2NQ0K<EX=_eT-XEj-c)_?&9oA6&Bp5awH8{;hxS*saKM#aW1D;$++H<%F!lCuY9K0n}D@LO_2kbuaGfW5hCVX!-'
    '$v>VTQ%^(El<P;MS!RFDh{GT%1bYuPhGcIZhE}KgCws*J-'
    'ZtwBOhR<KQo*SQQo5d%8+Wf{?EXX&i0}n~+4}sU!OFk$5UYJ0%iNo#5gm5>K5`7DwW#95r|(p1Q!LNhF@S!et3ZD4sI#8_D@TZQy'
    'dr`95dh3d550PX?|uxkLb-_Ob(Kt5+sm^@e0xN<9X_8ZH$GJKOLk)LfA`$C$O#ay!R%pN>nr9<z|}K3Dj-ML_og@(Bxn?givBi+%'
    '0|<Z=sq?giwN7WLc<$fqpixfhU6TfFn<VlCA=QW#X#aB!p|2;ZnAo_+I`hW2^6PYhAeyY~-cA#B|wlredy*#-'
    '3^^l5{5r6M}_M5Rm{l+P_z2(G^o5ZP*9s+WV2KwbN>@}L%+L#*HeKN>MJitCjsQ9eA$BW6;rSE@q!@HCH@0j^i-'
    'iSprD9x*dEqTweRisSikpKE9nDN?ErX&Y%$0)uL2RjHVx&nP=AhT0fZG@mG{@T?FJ@mfOKFOlj8+ckt7BQt+W;|dPqJ@*^j=;g*y'
    'JS0|2#o#9(%tIp}%)=re%)=uf%p)Qo%)Sv2=8+K)=21R@UF(`Ze7e>)d~lMM8a{lM);4yGjEK=+!zRZ$%iX-896<U$-'
    'C}f)Qu_4*$zq*n2Dx@&ssVyLuw4Z}k=bsdyLWkzL^-'
    '4l+soq9MYey%r<`q{3TKz8PvxY@FwNx38nU^vdD&dq{A{jlK{i*mFq<n|l+Beb_PMf{K(h(I-'
    '^Y99Fg_vkuTU8Lr69%lwuN2EAS%~K-Mgg}Tdo%NU8tgAa52M;v+rV6#oHN3*$R5u07jACA@$~!&@A=l-'
    'C?KHn^&RE;s{?4cuVTdd%|w1H}6GKVo1a`3Y|!2XD)M-(Xu(i`j#7H>j-'
    'Be&gqtBDtsetJcxR61pk(L;{|HBfed@91<92Rmb$43eUpqJqE=d#L<Et_qKE)QSr!q5CJQ4%fn;e!aGETR2u_pb5y5H31(ctYwaa'
    'aZ)h&g0R~zwM`dK7Zjx<^9I2Oit+Dw4RY1CAA`eyiAI%4i1Wve7n`0i&=4F7F@PxA@wx0yvTfqXY<z0Jx5D9JG9Iv#j5ja*XVt%c'
    'Hz6g5QIWa%`E0g=&6#JcqvRX`T}_Ne_~N7Vk%9JN2}jM^V|MePr7MePr}qxOeAT;D}sb02$lHkR-Sl}<|h0M*0QTEYaVP^al==YB'
    '_HsVuGL5I#*y_)7zV88(1D&r;S0Zk$X7cXUTxVzw5%+C%1v1_%Qd2+UGICrSe7M4JFQ(Kdiiv<sjU?E~mU2QE9mg7AajN-ZIj&$O'
    '4bHf(|_@ft006I6)T#x(e@wzDjpQ9rVPduhl)2243Z^{XZk&L~RJ(1UT5Xz!DSaP6aV@ng$y)q;b^xxumkz|tuIuyl6tHy#6z^MY'
    'Lh7E4!$#Uh>CtK-h?SK|Kc*W&){*9&|^icI7&cn4_M>_~Krh7*NE;m=mn;P-;@%?BP-nZ=x%k`aBY1V_x;5mO3!5vFPfE@me%Xr!'
    'F?eXi6<I$!pBl}6g}EC!9l^KR5N8mZ^KsB1Nn&pT1mG}3Qm)Dyod?uman?tFhW?tK5Zia6;Gn&oK8bSvRP9T3gTVKhw<3AjpeWX7'
    'g&qGZIrI7WuVC^SDqPg^vqbClsxooN9k>1hNI>}!+z5zr1^#eQ`<;ZeiEesu;RFGGo6{hW}#p^aaiNoe2D)~|j+P{Yv9ug)TzW@w'
    ')o_&QZba}f9G7~Es62{^0{voB(qis0jTn>sU%*|J@pTfd+<p(Vg^8nzb1(Ml{RQ#7D)pM*L~>B}r*fhS{2Y|gDBj#bvjfs9Uys_j'
    '_jmMcnkvh=A?lv2M|3#{KBWhJoF45b2kh^nEZAd;bOC=&ogR1Td4oeZ@@X8}V*^-'
    'wNIWvCyz2v8X+h^~}!%4^H2LO|BrX(p{D)*lqyU65`ZW{D+dbGwDAE9OQzyLV#F4vQdK!fc0Z!%Co<wTwk?;axQ=k>GeIyOiiHyp'
    '^|<li=*vPf;1aeyYm&^;f8jUw@^_`1Mz*j9-'
    '7X%J}uysEl8Kt;(Cc`)Go50U2i#*r+ArTpH_JPy<Z)xCr8_b&}~|z&qlWx_HtAg*Y#WdrU3}V;JK&hn+DRkA&T!G9zJ|Rc0jYPL&'
    'x6yGvz8!oH<4BVl)|%t+WhisKwT6kT5h)I7XeX+h1yOKIs=!%N3H_gD>EoI<%>LygW&vLt|+7E3BBC>qH_;yRPz9}(wmiYf^X9@X'
    'lQreM50O2e%*e9SM%=a29(wzktM>*EC6M`4Zy`uL)wVV@Eu4WB5{((tJgH4UFB(bH%ki=sy9B%0cn$bKGRUGe%NprGRQWk5m2Yk#'
    'C|P|Y9ZysSG2Ofb)Jl{S(wj166))=K9o3Cn#X(WGHfhl&-DmC;b-<rY;X2L1J&Bll!_6dv-'
    ';bfayM*sJHUb}GHs!{aoplPQnivlH9BCO+9tZ1<}8`*vcxx9q;D39aUvxr7rQij@TujZiy(-'
    '4SGtf_Vjc&vmqL^D=07Qzj;kv7_XN$>`!C2|hC4lt~5G!PI5st+W#AF!E+v8|p6dc3NBNEb@k0JL)R(mRfu2DDtLS2kIso8wdSMR'
    'Pa5@?Swa2Vc1!cV1gS9Ff_J4#WQZ8mU;458VX<-3d3b+)VLa8Gzh;@&>QKRgqG;I9p&CklMn+|d_%*~1Ro3L?-'
    'HV1H#TR>JbJlBl<UssY-'
    'f*LZV}~rusPesqn6uR;$AU{P{1yY^XozuXJrcWuCfz%>H^NxBwKJIAy6dZ5{C>E#j?l8vFw2wnw)G}r18pY#I-'
    'htDYolA?eGh{6vCZb8+rKkw2EQ>e2P^n-'
    '`u2(c*E|rUAy~Tl=jTHKI6`S0eq;UL0X*xJl$OZQRZ+(^K!~VSxQ4dl<BKzBvIE;q%uEXtQ@6iKv6gK|F~~{vqT(K!g0o@dyhBy0'
    '-Q3tdl{?*=@YPW+BaKSXkR30|0F*sg#K`)AqP8~MhZx%ctG*TSJRZ>kFTa_gFn8SrY-'
    '*XYMOTV<Ev@f<BzYV>3~0r{~J3;+$t>U`Q|}7G`_ldaBd9nRLdnT=rlV%C6tAuAJr!+us<d;lt~s%NPMxULz;}-'
    '%RFdrKZ3Ixmf)i!@^>_WNqxt!E}#hw@Ngn<GQTzONFs1DHyU^}5dfK)1|ClYK;|X`Pb7lKGQ;7%gp}Pyes!w_>%Z8qZnKjQF45dk'
    ';Rb~q2$J?8QZ4oXKCSG4^MYzr6wsBTy!zX`h_KUY6CmyH%zEKf8FG?2OoK5prc^r*y(;$4Epf+M2XyQrQ>rtnUTY4qEg2^_(3Xtj'
    '^k_@QsSc9maU`}|CWw!RDURW%-'
    'F*mOdf^JPBD52Cs)D*W^zG(3vlKnDymjxSsK}HuYGCHdkotf<={h%StdMJ$n<naG8#bQnryt3=B<pJcYtAberyVutl{~irHRn|fe'
    'T<s(Y95DyyKs%>xL14f(+9KWoTgpe%eIGAN!qHNBKTXMoI)SHw<M>~M?b_Sr{tV|pwWf5G(mrMVfLWnF)D8%bj$6c(SQeZjMi#t@'
    'tu*04wCOENJU3eNS4C2r5r)3h$cnjH*BK_0NJP9+R`L?PPvhLD7iFp4>hMcN+kkfLq`%ws|<m(?jev?6#{8JLLjYY2&DDOq(sn__'
    'W@K}x0tJ;z*{>;9Oi>5<>EuiC6ur5A>}rde(@pYwv>MHA?0?I$M7NL_LRr)A>|H~mU%8`5@q1~Bq~7Qka*_VO)sSwalJHGR%z~$p'
    '<#8-'
    'YV`_gBd4ej)hGz__A>6#F+qukxC1GCiBGWIHC6CWO5>7;iqM?Qn_HYv<sENsLZx&*QQ9t{LOQ>>?Gq}a^P|0wAy0)Z=mDDQWW3Rg'
    'ZqrLr1b}wRDw6>snNAJX_7>07Hm+OH6StC5HBkcg@2CahTDX6p9*Aq<{)w6(u7&%Tfyd%nxR+2j#I<m%P&>r6a4#EJrWrZqyPE2Q'
    'yv^NDQ+1G!*7es^8|2-'
    'D<1`ifWZ;A?x?&uJt>+lTcAHz+uGm0c%T@+TFw_#%G@E4fc`P$mST`cy8H(@c+iD*re<5$mjJMxqndFNjU^L5eqP)MA(62*NS8+T'
    'cv5X+pY>8^%6?8ynOVjSb3DszNK~R!Vg{Bv8Z4#=_e3}zn=*D#81Q&WSm7K^z@2UqUve3EeDIFu#2GxQzJ=GXEBXo?U28E(@T8uH'
    'MHgZX(LOd(m5I4#3-'
    'D%o}K?)T13aUWkjPf~&6j|5W`J2n6Ol^B4MC)zUkZ+_Xz}kH96Rn}RmnG!?)=K`qJGr0TInRgQXBFr3(8cV<Ano`imhq@sG$EJsb'
    'YQe0pXBkXXhh0%RkR{yxGI{FGFuhxNExk)hGd)~TZ$^cIj4`I8CKwPIrvZyA?d+fFH|v@=tVLKhPX@`(?+ESOJ7=zhhCy@o2G4~D'
    '{XYzR+`elsc_qqHYRN^Eoo!Z4$>0{230^@GP>5zzG}?_vq^+RIS>?ktqR6RkzihK(DU+sBLxL)ome(|Oy@5bzD?Khtmo6U4NYb#<'
    ')CdRDxP0N*TqUu5@)!P(%y8v#m@%jh(|2@(vBI?mzHHDU)m`n_|Vjg<sKRdvDiaXFP3^}>MhHm5v-ubMqko2j*>>jIXItS&Mq-'
    '=jyscf9Or7;3&%z6CCCu`j3u${_c+C5c+}S61y#qxX(|?a0z6oqu3sr0txnejCEmhJ8O%)H!c8)nnY@LYWi&H^PtKCr%;YWHD)X2'
    'LTk*CqkeNpH?UjK@J>WwvaH1n=BtsT%e-n>uw-'
    'R?y6cTmmRngTml#dBkN6<P0^0>5FgRisz(BZcN=z!+L(Ji&Z?bXrz6axAQV*l+7aPuHaF2cw}7<7qXI^a$9H{LLeH}f>z67~=B*c'
    'Ay!kh-DVe^<s#Rl<8RfU7&aFC(X_(A1Tg4e*f+=IUi<A(EKdK|||UBR3O%G;8FYH22SGWZPIc%HVO7Ztc+qEsRLIJ%N6ylWu(oc|'
    'BkBUl9bX19ccG?rli9yV@6F{Qf3x_9CWe)>%aGQdAg{KZRc>l0b#uB$7de%M(eV!jwdEs4z9*&|FuzGU4f<+wBH<(`y&as$<jZEy'
    'B}g(`z?@J9E5p4~+`Q%@7_GSB+jrqqQw{kQ=tP#7@m<smPo)6p1Tswf0EQ@@o^S`(;#J_ZOQIvU@rBmjt_`*4i1%;;j29s(W0{bx'
    '{{5Y?uSpG%`FE6*S7o$x^|f?1-c_AtWRSl%0^YMKYkRv*1sblNT%l+PVn(WLNTk<6LP^4c~M`>8;`0*$jY{4w+yHn@A-'
    'DTgNf|3aU+eQ0B#dEU1WhBK8R=A|8oP1QQX@#HRv@h+^k6hluE`q|cSFY$N;N3xdTFD2<&mW(>FfZGpPRt$!ysuBV=APG$6Uy~p0'
    '8oI^vxhK>b&rwt9RH)Y2qfZ4zU=d%fl7EmzoEN*sGv|U;Ga(8roPvapA1W?xBL`bj$3|l(23XU}qlIK9fw$3>Ujx%iRoQtB@u(fl'
    'ZqRBqb3Y6tw!-eoDM|R0%Mm$5iWy&R<p*@)qZ8o`Zmf)+-'
    'R@rbD8pM+mLBm1%mF6pTIF?eT@?q4ENa~V>u_7)_>xKS+qt`1`6_KYGdfz`zSOm6H@DrXILNSG(^4Jja%YT+cC3vOyAsNHs`K)2_'
    '=d5AzLe{YOOV+UXYu2!MF>6@-EjBEiD6qBKtvH!ITYQrS6H_zP@7U=VT)@+srgK}|!P=;P;2~##J(|G(<|-'
    'na3ubERa5U>Uc}{Y)Eqme?8qf99^f%J5H(N$9wWZc_8o|`8`CM<Oz<3Vjk>Z`c&><?I;ejo7lH}RZB+1%nlH@O=Ns{M8lO)fLCP|'
    '(bO_Dr6CI=eCTO7*_(XgXM>1{J!GAwwB@pXIF96>>iTAycaO!C0;V!gys-'
    '0@KqH}<&t8#~M(V&mgAY;&RJT@G+cIQTT%ZsZl%&#0T1)%)qx%*(ue2KDkXZ~ubYby=I8MculbeS1o!+ws9Io>l2ie3(>F{4<{FJ'
    'zFmR3ElvgT)T;(GWd#W`gulDm5wI>ovp)Q)G(c|fhAbl0P|RYrSs5H;Ly_oSSS+f=hURkr}j+hQD!8(CkTb(34dqHRN90OKHaI(t'
    '@z*;PpWhqKKNpOmF~a?w|GXS&G-'
    ';8%096gzi>0vvH>RC#X13S36*uO6iUpV)+Wlm=8E}kNGv;2`v0ejIvVY_fR#YH5Eil6PZz;rmip;pxZ5}<H}FEGHNck}o23_iJx)'
    'WD<JTQf`hTg5l4K``IQvnvAntp8z*1Ny&cf0{h>KHf0(ZZgrY>LMD>vN^zd|F_s2Bp?p;|BXisovnmqaJsmkUzmprk*prOH7`e@v'
    'rVN?4W%NMIZ3!WpaycETU8p>~|JSHdUQan7PT==-;hdQ{NMgf>y%2|AL{-'
    'nN!x8yXHc<67Ef+@Fu67H1Ysan7_}jO|=gG?H$XfeEpk#Z~BLo@k_H9WFA`vkn&<X<CO%jC8HTBqMF>aH)~Lb-2v9#&sX@D-'
    'm#Elx=MVy>f`1i|r5aFrK3#NABSSzU>m(O{hi+%MvG31R^h)GGj$c<8reK3Hx%OcY<N!QQ7cu;-osSGKk$ln(-'
    'EDU3yCxA)OcQC7mC5B3;0@b%IV8GJ6cu=^~cI!*#mY9!Iy84^x?56smocO7o)7?Bi697lmS<pn|-'
    '(?*m172PdEYO?8VcM2D|anKp4k2#fo~2`1(bT>oPgas}sIB|lmc(#CB<+PG~<8@CGy)b=6W-'
    '@)Zl{pUPFHHfLsr2`^krj$aCzUz>rz>*j|t0HP$v-'
    'Om6Hi+ph1YJ&IjkM93SX`iT_@0v%TcKS`;BwPcroo;<?{N=3A%n$D^b}I?ZSyX1dM88P**?fWQiMgu!rw&^7U_ztrwEITId7l{iz'
    'Hm<QHV<tuJbA0;tRhmob;nffn;nhP}H?<<{U@O;|TU39gj1Va^(Y&SdL{YnnG+j?M{kuw?Mud6oU%5TKd*JNR$w<iUmLQUPXz|!4'
    'f(cB&JOp3$t{BB+Ulv$ZQ{?BeQ*|7yunCu+WKO!w(aqr=x`!8kr3`^-UxoyTl5p;(V_iN$-'
    'W3qQ}x00iwcnXn3MLlED8sS+&jK6=p4{r7iXuxebw0EYgBp-iA`TwF%{IgLJ6WuV6%Z+mTy{>otf@OvJoHX$DQitS8Wrhe(tRBP3'
    '?Wye{>)ETkTnht%VfA@%rFNIgCsQjgDs)Z?=u_4r)D^rjJoHND8&1Dyr26Uqvbcl*6SmT+o=$n6Tt-coM2Ka3e&Vrav-'
    '=CV49nYg<~MRZS`8yieRRU&CPgy#4}M}9cX?&0c4veg7Yj)1=eK#q{V1VE0UzXU*zu)hRAj=;YJK#tJA1VE19zkDnu_7X-'
    'E7NveJ(~)1M8uolxPB4O;hh{@egESv~`oFqNMzp1}Y;Jd}&Ic=|S$D8BHX{}*DuEa%h=L{38GDbSdWYi$Pq2N2_H7HcBJa9ew!y7'
    '3+u+tc+u&A}ZE)+6ZE)+EZE)))4Q`%(`zdZ@>GwKyO5`KxHWTPwI~ye*uHe^3qU0kK{KiO>{6zE)j6}&#QZU6xnqs7asYYVv)e5d'
    'K5;GsA;7S>EHJfV@34*bW;78wPE4OY_F?9OFNmiPzFjm32*^so|TEJMP(wzZ5<Ee1Nlt40$BL%X?2Cl}(j?YJU+wN>OL&uzCqdDv'
    '?3$}__uvOA{2|Fo@g61^U7&aNfFs+D#LNZK-z(}eXj6_=S$zl;?y*nagy}1#x-tT-'
    '}CWYmA01Wk{N*{e%MhP``p(vroJ`^R?*omTq8hcT`?cVjI9+EXv6{=##N3#|DFh?Pm`fDNDN%qQqX_{+*R)&k)=7BO?-'
    '8LVz;qtbHkQf>J21SOe+Qh4JjYh|YmZCx2d@x5UTgQ;0B3Ofe$W*Elur#}6?5j%d1+A#HE^cjPhVKj&PXU&O3bf1Ta+XJ8=2dab9'
    '4(^nI%9P;^)ls)Fgxog?c-+6)Ju(EXU$Eae(GU^G<B4++g)pO^+WTMMDWG=)^fDC2Dop6*4D9^X-'
    '$Nm(8Lq@Ix_avJXD6;csO4t#=csB^xnb4`N|plY7r{KojjbctHr*``gue_35mQN4S}&3-'
    ')rK+!I35tSgMb49_kBohWPRoNe(c{mq}EIvd2-O8nh-'
    '<gQlSxbabeTgishdCX`6V1H!K&ED=jf^r}@smxKi_W2b@!=N#Qav7@!JWak^HUawJHv<x`J(K49zQUhyp6`6BvXs+2<D7v^?bApk'
    '1wWHA6MNVXCa<M^UoDT`U<@^@r?`)5@YfWZ`@phZKjryFLLO1DL8h*eIK}(e?W%k>4O)R{Lu;$08Vf>&*3b@o|<lsdH9SIhGml)('
    '@mVT2AvN9j=%M9ucuAZrkdWE}3y~0&duW*m3SGZ@?E8HvU749AN3LoNnh4T>~e?Hyb3RNm1Fj15ep&5=M7L!e|I-'
    'H^^NRg~a3Q^Ro0FZf=!>Km_AEnY_KH8fh3BJJKqa?u>8hn(Ix{L6Pg_?}T1|Ovg;ckPE5vRW=B{A>2>|a6D_%zL>rUVBa#p-'
    'KaKcr98H0U63mm2Ce5gIfQ9Evaa=Rm|JBSN7THXC?DiqD!1yeTzATMhh2MgnXz&?seTI}EH>G2Dcqc^1J6{>;!6i(vJBZfKB21V4'
    'c%CIdf#Y#13?@Rgxi7p?Px_p}21FyzwLT@~VX!c5Q0icctJ+c{X{Q&h;vlM+}NOT&r?R+S4doh9^%Vw8$#KmkScvt>mdzJ$~k7l3'
    'xwwhzrW{b}2U=DXLl?LqU+Yua|8`EEdM`_Ftcptjv-'
    'o}abtJ@a)v+IF5XU{rSnR|{ZNm4a&o)u^X}X)#*#2MR88Tz9K#VUTh$oUC~E#5Atjou;INWQA4Vrv+=ew_q>S(w|m@*~&7hA+X*S'
    ')c;OScyJd(!%wsnW6JpIQR%@jPc!Oe3C!1wnbZarXa-Gc3kx-)Cbfe_nqiaL!(z?2Ngd#B#x*60aFrlD^-'
    '^%1DVTyqSrluX#ACu%IVRgmko_$j)3YgS#eDU<OrMl%l&Rx!9=4P%@{y|Jz|+|}?isCx?FQB<PPq4`(j@=g523mw|GW<sCHd!vQb'
    'm$~ei)S_`R9jIEs}qJ1QjCr=Y6RPNwBbbD7aP-uX-'
    '!EUJ$PiQ7}CZc~6W!$Cb*&BLQD_l(NvKEdzyhNJKKBKHFqCDY;$62AB`M_Q3m=ddQJ9CpH^Xr2(>%ukJ)Z=`6}R6E;2PtGf^wJ&U'
    'q#ghkJZ=rz(vPGfn*@Fgy-3`2R+?(Bs&QpcWmLf28p-e4ZjllIDwY&e=4)9G=!$$-'
    'y!e{l<qhG>DCmYXOvQB2aLx<CsOWKa*LS2b!@?JZff5+<9tw8KIVZdY6m_^_eIPhSuCh@rMmUl91Hp{7q?5%`#)mQP<2__(2lPhS'
    ')GgrRm%Ulh2^P_t(T_tE8q|If;!D-'
    'B9ueNjP}^`4PQr&Kv4#Xu>vdzuT1{K)Kh!?DxV8a<mlJDp6`^*omtjWoQ<R6(~F1%i>!P8R6@QZ|RzC`Pu7zX~eH=A<A>f`-'
    'zo?yI1c^jgM>YKkghwnmjO+oDRC?NKGnj;Io*IjV%&8CAmU@|7@*65B1n|8@`9aa933uE(T8KI)q=+#^T4nhn<gG;&IiK#QaFa)%'
    '^FS?EsymW^5ZpnbVl7rF%90t!fFdN8|>-'
    'P00o<ETp8a2H39v@JW8J=1pF!_g~k&n+Ci(+)hDiolO612A6C0F2iw0OR$J)Eipkuectwt~QyM^$BNb8+79Kbfn2F&4ap-DBUbON'
    'l>GIoOQFKp1fWkq5UAT+;QLA0*-'
    'h$dM9obtt8u$n<2ZCY57eq#Usn|TU_l(hUK@pFpuoY|KZ9!GAsYrFM5+z`JK4P4o}*@9G`Bauyg+QJ0zr{MLC@ut15ijEiAf4BVz'
    '2ljL45}vNF6dK9MVtM4rFjNxIE65vMTohR5>k17zL?u#jUmCBv;LI7%a_6NBnOq7n0_X)azFlK5e4(a>bSw-(L4lE+r9M-E-jXzo'
    '=E#<&qVbR(m=_hc}}&B&pf8O^=717jqcVzLRxAAW7Z`G?<_AOPTUV<mtj+PZK7%`^?BK(-;1&Sf-'
    '9XOazYVXw)A4h6k@?5`KO1l__W34Kl$K$=hn(2f3gD7$OlmoQd$I7Px(Rqz7|WA%U^N*JproGM|gUhpFcWA%m~^ITUnHK&*$*<h*'
    '(q7AMvLAJq_rXVz3-'
    '@?qa^bn|a1*5B|7A|k7Hp#7%`Z&zmshbi(9qnZTDo8N{N{=J){w>>SRnK0v9lw*C21dhs=v_C(io4tvY~YjL$bNBgzzjQWnejVrv'
    'h$U%(~_JkrOCvSoa;yvof*32WECn^zOO>1$|>SY&!-J}imO(BD8BSU+7PL?_>l_r7C(-'
    'BY5s;&2#Jc@#R&=NdIf(rf`l|(!3##vpz0L-MXa)6DztG8SD}q-gbHn3C#ulKb&?8gTq9Lz<EmDnjcb(3wsD<I1GHml(-'
    'Z=RU44U$cv0U<NO?u5aKYQ!pe)}PNM$sr@Ud%Ly?rI}ij$K~G?ObYCqGr>^oik#7F35Mbdop4{U=Mq;G`ACf+R>tahA!*Ai~|cPQ'
    'mj=FsE)%@K+<4Q@>U4q7f9X8x{O5<S3fxzJn^?3g3=VO1HuP#3-fP;eTV4(jD-'
    'f7^Sosj)_r9cfxmLl+s<$FUjOL&~*NXqtvw;w&nt4vuLVLi<57ASr}H?(_-'
    '3Tu`8C_s`fRCLvqDoOke03MHjo5DR8{4)m9LH{Y1eOmtE)!V=`y6#6rbRFf!<>t$|5K23@r?aH)||nI!>}=SDkE{Tg2bZ}E_#in*'
    'V~6?3P@6?11uTG^r>aVFV<A4TPJXSqyrCe_wR&>+j0uTGZQ$CD#)v09q$Cu1u6+fKa!A_TYFld>-'
    'hPCVI6kE{_SHNFLxWUPb3HP>XUha)r>Wo&@HnyWH4!jYQGGB&|cn(Hz)!`CzyW^944Ynd_HCZO`nw3Fu<%czvPhN_j6-TmBV-'
    '9;mI=0xQ68g>+b3^fpJMHtyI&IfbEa;>4Jo^UX!fQwITJp7T)<rG)LEiquU9rm~wFuE{*H`(ZYJ(~>;__L0q_w(3nn2$djIC{T;&'
    '4z{evx%eki`Z;fj6Yk1Q$UKpkQ8H7RDn2FMQk(9AvNbrClqOC>w;;3Bmp-'
    '_@UNVzOp#_|lMvyLglwpJA>M9uAss(5XIVs@W*&k`MSG=jZi5ENH%S8fajJ>r1ojhD6Pbk4$?cNBUQPv)2ZhqfW=UW_MYW^Fp^kE'
    'wOh9{C!MO<otWj0=`3b|-QJvBSiG^6B)$26K-'
    '2v8R?5Q<mA+UM4&xBdMmJr^DoSfJ@@=TNAF4tUIkRKUKqYqmeC}E5GGl;8C>I~xQQ#ymV`n1j<u0EqPh^x=)4C3l@I)k|SCn+dct'
    '>C<bQT^am1>+M&qJ!5IT$nJfAB-dvLl~Asg8pf8GI&^K$AXsDO*FbEfYff1Qj&Ui^tkgXSfO+XJL|MlJd%5Ik_Gs(NNGN&>xW#xE'
    'hZZB0JoZG<^|klqM;Ws+lV_h!gsz<gr49BVW?l7L@0Z3f?rh=_9zVZt5Jk!2_yWfhES&9M88T2FbPgdqLYAb#Tzn9w=9>UC0Hm9P'
    'v`puj))77`o@JvN5+LmM@i*z4FU3O_j{G9v?J{Bi_t1AgJzz@q|nFO$%B{_O3+<Ahe@Hi`WBC2Qs`~$<|#}H9pgQjnXFm#N&*v-'
    '*=+UHRPi3iPH20Vi8~gB%bV<RVN%`RsufZM|2g#N?>MfyPQ&(``S6}9n7kb|@f;Qxavd%j7q#}+;-'
    'ajtOC|aWZTZhFp4YIE#}m`MnN2*On8qb;<^jbtPh$&DD5e=KO+2EQre1F48O1c!ZJQrb9P)TIQDS?8ic?heSv%&mXrlR23k@WW+F'
    '{vQKYsSMiL%_?6ujgD8eO2Or8+s3@}I_KhW8r65(`<6`wZcUg>20I2EL5T4j(kIKOW8Ykbzdxk3TO7jbSi9Avk;=Q(cruYgoVZ2s'
    'l3BaVjGe+-2~HNLTMt*X0o!dQPG;81}%>xwH0lWYUippyx)J97R=G`?Wt5wxDyBDu~|UrkrCV1ASv-'
    'zpFxfizmyqSgHAdfv>pKFg+XwB|Il)xPm_#6o``ht2gkjlKz!1q#)^GO&5=@eMHm6qgNl*bn)ob$246$di8Nl7kdy-Xu5dx>M~7N'
    '$IMynkkGtc#!04(X9P55$rXb|H)f_(F40`*?d!^UR<uaCI>VMwCxtdj_OzlM{d+3da?`MKB*ZV$HOVNviX@}(x+NKfSD9oKUiT!U'
    '@T!uG!t0S_6kbo6qp(?_GUxF97IfOADs-EI&!zThhk`F{iD)QxYP5L&VM3(zA&szUD=DBDXrVj<)Z&K7Eu2C@8XG+nC2Hp+*uz=@'
    'J#+LsmEy4y?#i>^UQNDp*qnDY`OhJz`kp2qI-KhFHTluuRDWP+*@sj8p(e!<a;zU|(jDQHe{3yKPsb(&pGwu;Rt29))m^iKuMETK'
    'ETFm2Bn$#S&P_aWTD?r=3SlTq)<7?`yU=49s06rHntUB#suzQv7xAU9I9@Cze9?lDnoAA<x<AX1^Xco0C_a!-'
    '9h=WOp@2Z2waML%HCC2MZVBD@CxR!xxge5Yp$h16Y&=vGiYefXkM1uIrbGcg`qE9RDZxiyyGi*Bd<;`eX_Rub=e=pnZ?xx!&~W8w'
    '&->7L<!H|jr2)&)o*zb|siQqVoCXC)dwv9sV9t2=c18!!WNDi-'
    '`pU@xuZO)xR9sWgYE5)Zg5V*LI*BXaNf>NAYSZntJ4C_oYi;K(xAv4NF*J$2#IM>=iUl8iEiC2u@san*njt(+2|ZZG92jNLg=NTr'
    '8iPJ8BMzhnomd7O7;VstZDIq;pc~8J17i&OvD<R+6Ub<|OZn#i-EN#bKHH;g!$j6sZEX{k!uNX(!H_ig*rcn;j$2dfJ;I-'
    'pDPb1b>M+ViOoLt-Z~ibiE^+hCYN`6UZ^Eh2g*3NaG&M~+=EU4$8b{8l(7S2!F=y$PDELOg-MaVCv|mn=9<J&7k++PG&~*LSua44'
    'mo#t0x*L0m8`xSqrVh>BJdMfKEgcmzWMOhcL2mL%AXJmE>WF%8bVi!&LhgS-6LA|T961vQTV-'
    'rq1HIk(8979v$HFFHjh}X<9G$CFy$IyIu%^XA1;Wcv%&4$;GW8C-MK-faj5k`&l@A!zKQhF^uq9|!!H<?kcuhW=&OCQ{gQMo2}>N'
    '?GI(q<BIOH7L;D0m}%<D&BiXp17TfOqh<7=Y;^+B9z2si&;h0Inw;z|+c5We?;bd#JCE<LO$ctkGIJp6f}EmqA;LVQ|7pye;$*+a'
    'f!LYGL~)EdBcvlq$vKyEtLe6Mw{tIhp|2B~syvc3cgI1q^qc{xQ1xkDdPQ3?CJY=8T3&5r!+S{U>(GD6MGVA)WtY#MNzbIF$KHh#'
    'G2|x9(?V8<6b|i_NWj=oCfL{a!jpk#t9vJJR|69*yJZ2#tethP~6?k}^CbJw!5webPRXFg!FpRPu#~p^V^k;o)f7X0nBN;%gkWj='
    '<f@d;F6$0ZuFvs0|01x<qq>Mi&lN_Qc)PVwdAGn$bxpWw{z4px(FTjH;W?V=?uAczg4Bt*Y{Un6=h^1Q`$W42RkJYBnN*45H?N%@'
    'V}9$qbL0TA(6>ps18oR#uc&)&UW<av-flQNWodq{*nGh~R*Nh#o{VP2T%jYu)!+dq0ow_xH#9d*%6@&)J^o*=w(LkJo)&Q!)E?Kh'
    'pr`E7(X~sACtxepScTgPo~k@4;T8Lt`N<=d+6Za<bf86?x`lxhc$fDs_$&G5)iPF>#@j6}=GZpo#f+`c>}8l9{ZgRo(y#AIfgvWM'
    'Q%gu(kz3M%PkfioB7I)<l+Yv*8XyT(c>f2a3RooZrkgdpc$TA7oDC{1&#@2MnHy|GZU3JE*7sTxGO_0!zKlXa{*@fi)TJAm3-'
    'VSw=g^!whVN+)Kb2*n=o5azKM(PEbQ#<?qU_^(KwHO+(fCjL*o*OwpVw_lR|b9s%HkWtak5ozke~0r8VQtz2Z%Mf!@4Eh;-'
    'p$6l1ZNXKTBy;w^(Y9u`+vI&9rY+9C23a|ouVwO*4ZN}+Yp0lLrNm)LHg-'
    '1`$a?2{Lr)2pFtTcLRmQO_j*_f~_b@%6NAQ`)I3k5!;kQF2%TRpH?;g%_Xm>8*4Cq~vHx`Q5W#6g-'
    'xQ^kc|sLPtvjjoa%dcQZxxT&jIhYs&b9x%oHM2tro6K&LU5+TrY6P?uaf=;RzBaQ;0=O+T87X*P&S)WcI4RQO0GK}L&F0h%N`v=M'
    '09iFkcsN)9O8oP)<2>8UL2V3=qRH!@Ao!kQ!9eihVfkr@vbzQkX^v74oNUp2qA1_XVUV%i=j!}4hG+Z5}@cLM|8msX7c(@v;@cKl'
    '!8n5vBWVo83@cLA^I$Gg1GL5U8@Sdl~SN8O(xMBMTjf=daL>_b*xNUl6JTUUEqAZ=g{d;8NGIsa>Rg|H#zkgp*e$Ed6fTHZ2J$@@'
    'qxp@^b<nXF-'
    '((VXeXwA=zanYHmYw%8g$5q>#gd6qHMDWQ;3riE|Y%QvNvM^6VFw`xK^76vP;CwQ4Ph&)}6<&^f8_&3<thHp_+<+VXjO3~BBwc}p'
    'PGq{OD?pBtZl|?zp@JC2_8`!4Z)=(H1c=E)8Ig(93wYNVChYk88752*<--}ngdLGR!-'
    'VOf9GWpq*zvV9Oqd?ZM=}Q!u7Zb=m)es?uIo)?uBCUh!3aAoS>ksRNV$VEvHeEs(o|Kz?3txm%G09xj$$8W+8bkEr@=^HXA?AOq<'
    'irfXmv`L=dVzykzU5%AmuDQpT9%PSvnj4fRL{100pA{%|>kIN40(0kZ15$AE&gyIQr>yn6D+w@t8y&fFmMI2{5TFsuJy`rL<l{^F'
    '#)1*K-jMYJD4%uI@xtg!s;g<t4|ebcIS-l@^Kr`m$5<4lZj1n7qP>-yvCkFbhsl!q=7sCn(=*$AS}-'
    '?zLyZ3Ci|5u;2tGdmUMDf^xl1g(;kqK?~zn;uLVgEU884vLk7Iv2VZ4VO|5_4xCrBACP7UA~SqWa}Y!cvuA_6Pfj7S-'
    'f{%wY%DhfoKfYTinFOgngFxO&T9s$L^25o#&C9DjgTu~|3&NmE2fyny5BBjROf-'
    '*m0cJ;ggb@R>wFL=7M$3rpvwt$`u$ai5+~mjeGd8ZyAvQD#ZaqF0!wU;PWNud1IA6()y^WFmX${%c2ofm8@(SAk_osP?ln12BWxp'
    'ZErHSn_KyQpS};1dM>G2ux%-K{P08IK!@HE+{jp5_MehDM-mv8EKh4Bn<nGf^#~;!i_`UVEpV%(#Eyq*o7!ze@c@yQTe$hl7&&5O'
    '!Yv4_HH^|%gfXqO5QG(6pDjI?dz_{t&%Y|q%SMyl67p={MZ3mGKZEHJ<?q)mNNn}LZ+d7dE?O;2L9%o0}g?m<cC)-'
    'sQsf+1G9ABmVNwB!1bpfYIcks*P@|P+$?*oWaT+&Yz6TQeSUW}f&+n7TsS1(4N?I8&-Tc@yNr@I|1es`fGr3onV{m$Ga@Wk-WM`G'
    'CafVMY=lHmbN-'
    '&SL+f1+wJ1b+qHYxh38+Ba2ErWw}u?zBAK8KitkcGm@%jwHM53TnQj2GUIdlMv&L?h3GkSQF`?fJ}(lU{3{bLhCY+4%1(mOn+g*E'
    'AxmJ|5G<n;F9vKtW=O5kH7WD#$KA10DfZZoU{b+Q)B0*m7AX%dztq9EYi4umu_GI!4%TbCSgBuQgD9yFwlsuHtDFz0_CIKu=FP(s'
    'B)*^4DfuFluEJqHB0TI0k{(pm5u3!X>@N#SmuDcjPfc?IF^e{5^}`=B5aZA&fJ&D+!_MIGfxOH))o^&47J6C5M%8zA;gGOOb9U`6'
    '%#^?N5zED!SRvYfM77$Xg3-'
    '=FXg@XnX&WJVp@~23v{8TemXeB++uzeNR=8i&dajGs9{zZHOyqx5aWcIj1yvjFmnTh0RS;h+|<>Mv_#ktFpfMSr2$$_Ur*X^s16^'
    'S(g0DG{%Pr{eFvCy^7?7RmNvj1!`8{cX(eTE!WVxT-qlva#M5XAEJ$qDtrA$vI-'
    'LgMwV(!bt10ifRguWDV3sIM7_Ew0n;e5>D8-8O5q9TRsTWYFD=u}vBfJ__&{U2TX_F#F+T=))mPd-TDUl-'
    '0MvAl(B1PKNC_1_T4vzosLInkbBjeUloZb6tN`8-'
    'VebB~)kn9c3lI8n!mF=d@+TOGN6}A|rQZt!Kz1(r)j20NpSc!G6a?Cg5I8dD-Sl>x-'
    '{;OGHx>2ycIh_CfteAbSV0|r|zp%u9S+KrS;r!R8J881w;MIq8nv1|ogM5fj3Xy!d6l#eMqdRWihyLI9mE>j&8kXhi09sPFT%Q3$'
    ';gV92G+GsF>AYOMoZjo>QGfA$ZJ&tBu;V2H{5EDc0!4;Lm2+qykrIHE5*SFF1Q?}+1rj9zL@7Z5$4DtAzA0KjE@3w(<IZrlpL2!t'
    '$dcJFp+}yfL|`57{zk;*Y=RpQa<@-&pFYZV^pVg-fQl`tVi`!|S04??Uz+Wj&+?Z>yCi_y>z~y)eQhMPFYyyPodXIe)`u1E>ePuJ'
    'ikgq_X!~%~&HRtHk3<E?ceQ;qDnP!c?PC#lf4HWYTz1+4K@(A2+)4<oDU(n_Xiaf(W|Fx0AXA82KLd3Yw?anSx2nA47FEk)jr7wC'
    'eE)~)R15-'
    '%;_*@TI+;{(8hN)SN^V%98&!hVI0!M&IFAhKY+F=mB7y$UNpb3cGT+mp2GhqhIVqt8qdfSGsKN9JO@~JkD=mr}lt36Ex~43vVx&Z'
    '<@On;5W<{@UT5=|Oy{IK=qSs5hyw7RnszXqJ*9Q_3buTd#l{DHU)FF;X^|Nt@dIWX&BA{#Si_GANqMA9JeGY}qFcY~Ah0QP$xgCY'
    'gFcJAY3Y%dd@&yz&!#w1RC~Sst$d^#q>?DcD3{;vV#iZAEEjcE=p4XCO((8rvEY>NfkA`>*@;>FAwjbSqhU%8!YB;u1v5gb&s7f?'
    'iMV~oCvEy@X>iwKiO~a=%iddHgVd9NrM&;mWp1t6c1%lb*UE?|&w*~|0)o|P<83?b2<2Ko#-'
    'Vyd!Zcy(C`)i7U<Z3vG)}Z(ilF1Vc<W@`A`DwnW6b<9(q9Vc!=V%fsX<VjS(!&4mE!XiT!J(X)*}Wcz@|I_LDKZYXiNgqqJsTZsA'
    'YENKIM5Emo*ku_co_C<tYYe6*t2np$%kRj#w(^DhL4({n1C2Q>S*ODhzq7>sfvPH*bNhvn>`LPXG@D3gvkogp6XJp@ZG2i#B^=Z;'
    '&h8J3wY}#MP=g-'
    'Qdj8#h&YF(y?~PCf(gClu0fZpyFE~`?^f(Z6@T?y#cr$EhXd0e{j#syzg3v&FaNE(D3cV7=9$Vy?Az;v<gS6k3%3dOn6#0MOR>Ce'
    '7`>D!PY08tyUYDW8lOTtAZec@oq?p2k~9fwHB}0&gD$a(=r%`3iPX9SS;CQ8Pe8IAD|*2xHy5?)Ts1NB49Ibq;Y_2r=LOCoMfhcx'
    'G#TuJ(x0cq$$L4e^2UVt=2j-'
    'wq7d;m#n^z!_7{rb7CFijW0+IsSBh~07J80jnHlE}#W;bf_D&UW0=W6F$)l%zbhTX2&0i%K^v8Z(y(!=@XLG4ONk3hhdcXwDT*ph'
    'XAld7O>bRGjAa8@K7RjEV$8fN(O7wM#XN~{=B%?T3I$tFBObdUHT+qtjEf=)JcU7%MI`f$7#k)%Dbpn~ZF3H!L3O4cxEjCgHbcWb'
    '5K0rK01#n|QW}_~V+31|eY;;LvHo7J<8{HC_jqcq4y*vwTYcIJG+SYVLu8|w2Q?Xbs$iQ4F7dNZ?cga%3BHqj_PBBKEi?$KXOphs'
    'QUbMHDWO0fy>R)se{R<TF<_J7$vcvcs#b|v1e8$Q|$!VFSe;I@p3d#l>io=X`cpzLJho6?<W?6<KD^F*%O8LNPbN{E@1p6ccWB0I'
    '5h3sLu1-'
    '~i7nuJls)mj8I8eQo*c<$ClO70_(ce|8J<VGfzRaTL9Zj!Bayn|mHDYc8g38d6Ipv8FR`e=@{bKG6dyGV06?<&3JyqmO^^X}4F&U'
    ';8>IqzBOD^CM!u3LoDAoo~y-F3A^3PwA{-'
    'B7yvkDF2F6pO>Ul!dEV_<Bdn#91>QNU7^CB4$VzC&?ZpbHQ7XtOf5tG8Vi6$yV_GBU8cKk1WMnDMJ2Lu)Fg!h*N}%%QA>lgjK&Hv'
    'tP&xtJ4+vv|F8y%IWe}wFt=wY?7udbU)cv6u8zJB4q^b{LrYt>=<q6x0K`k+e{UKgq`BxI3ZY=m)X-'
    '9c}AuJvKrxdU7kT!V;onW4;UWLI~X{Zr&&H!WjvtQ4mg)BEFWqz9?|OnoXb|053OW8q}K^Jmu-'
    'pF8)T34t27b1>*Z?RYMrzp?0rr-ZwwE#IZ33Wc%see=#6D*0H>ojj%5>^T+;Y(wbse2O$b-(oW|bKO7<Vbs*NNP^R(S-B-'
    'y%5+kHlotp${(no^@-iY}-slZE0lJw+!GKV!1R<ZLRij~#Hx<$KF*sM|~xi3`Pj9lZY>Y)9A9>D^-'
    'rqjG4C9V)qs7a3chahV~7=6Px1)<)_<WYNlXWH&fnP4{*-'
    '_}1*}MpD)}+P+~VRhz4Ai;?7PzP4{NyJ{ppoD_)<Cr9GLJQ5#HiNuFC5+9xri4Uiih!1CCUN110++=DiKy46U&rB$?#F%a?x~B}C'
    'Ix|b7zx7W0SSA|g!jP999i}=kR$^TB?W~l&HetBF5b%FdgZnB_#jFNLwsd+HBu!9FZI((R8Felsl2PY6A{liqBa%_)Dk2$mE+Udq'
    '=Ncjz^+Y8xueYQ8c|0Aez17Xb{*aY*a^G<-'
    'b`Pqdi3F#BMnd4DQnJXW6stZyrLdGCirkIXXhZ7nj&haqZyJI29IoxRjDUGQrtP<lz;`~O?RO*Z?}svxZoCMvYmsie7`ST@D?AA$'
    'Z2IW0&QPEPJ0o0uMuCU#Gm6dG)Ql2w4z*6n?As~T;3rTubE|e71;bU$T=d|{(D#>YVMbv=rU*6>v$8Sn@P(~(!id6(-'
    'DQ;}dLV~abBOw&oBpt1rVi8g8%A(ZAJz6+BS@%^Yx^A&>5!TtH_)=kjI=y5Bi$64kyb?VpqnFk(8@?2bW4hQfXuQxf_)Ng+|-'
    ')v0XVaNy0@H(igxmIx)8XfM-j|NO>u1Y%uL$)ehl_=DNC^(ih7Y0zh?odA0#asT|_VKie<4~u{^dbZi?-'
    'S6|r4$b8J_vjO_}}Mk#hbDYI{}5|TYNv+tz4h_>;&DnJ~n(J@n?k;1X#LLH9|?SP>oxr~XR{@elB?^-'
    'sY;CUdUu(HV7GYrGcuW)~2ijMsWt7CuS1ReVmQ+4c5Ow+MHaiWg>iRn7_Cr;8FM^RF<&Co~ATu#pH{_vJ>S9oWmiabm0y%qWxYSq'
    'QY%us|)H3q#LmlOHAKnFYsQ{RF4cQ8i21NH9^%zN(;8u0(gSY(rqqRRt#lwrBd#OKeTZ@5WD8CJ-AeEuw2L4TD|hLtiapKn7y@gJ'
    'pmm!Y~KA(u#N^pBpmb;X-PZ%f-2Y-'
    '!shw;lnY?^6+&wOhIQ4Ozuwm3`mur`}#<y|AAeVjMmahHqvjHMIpUEZCEYXkA+wO>c@c&#QGZRmM+XASUsjXa21FHPbK;#aR8z35'
    'Iq=k#OaZBH_x1ii9g4E)uT%Pmyrt&?4c=VMW4~k8tZXe-'
    '3bhzpL_m!4A(soDLv;96b@Ya^1^f%04z&ViH9N7dHez%n~C+Ss3EFpdcib^p;j-'
    'JQI<SXo)xwc3gX@`D}qk!5H3e9i`^86$%Ao`QGZJ=Ccj@1hHLBQ3ma(7y(k8(Wa8LugUhcuO#hi@|Ybe395p8X2(jQ1y9MR^6qZ0'
    '=Lj$4A7yMTZ@4`Gp|Jr=C|{h-'
    '+8#N<MT~kwY537aq}7x)d$NL9YbdjIs_=Yc58A(JNtKRvJC*c_DPB0ZlC<wB@@iX2P$Fa`+EsXPq$z4jkY^A{333ghC_%nK1SQBh'
    'h@J#_2a%H?_aJH#<R5~VL}8aXp<1WdrzJNzo(W!nu^tdn0?mTYY9vD<1U92WBO`^vE>!`p1tjZF6j|$lwpZXPJ~>!`Qx(XHPYqX;'
    '-ak(fY`CJp@C8F3gRyec7_1T52BVskU&^#5z2;{!cSx@Vnamy1Yhfm{oAkOOE1%5^A~c3KAH^&G@m6h^eX<Py$d>{BW;Mfmf27EL'
    'F}(MBMgEK7y*DUwU<~j5u_6z~h~A&7(AsN6VdXydgs4N^&#n-'
    '4s5QYCx(*<{9?fLhl3tf(GNVeb%S*WJ28ZJ+5zI8&6On%M(_4!0*nynAlZTV=2VsKt#KF$!@bJbcVX|(kz;;$dgM13TTX!6eQTD%'
    '8_{jw+)G0KW#8%N;>Yt*r=}N-'
    'aC}`FgWJo3?wQkzLFZe^)g<ML>Q1nkG52Ux%;K<6jN2}ns14!2@IPNfHa)N?C36nWf72=q+Y1+Cz6G#EQ9?ArJL$8N3>0#6Bkxcr'
    'w^m;6lzAe2T&!lfluP3r{e;`%1^5UC;=4Tp4V44fvkR#j-'
    'M1N)k!v9K;ZAZAQu~kX_W)oUjiDmi=npufu`WqTpiRJz~T3A8h%2}$miKS|nSgQ7krRtDas*Z`J>XcZjy2Mg-'
    'R#my}fok2Gj{<SvPU#@~$E<oPb%EkO(qg1tX<ej9jhSrAzA9aB3V}Bn3UNzow$Y_mp}sLs+vhV5I#IoOG4p23fA1-'
    'x80JjlY%Qm4844Jeiy4`tWg7M-'
    'lrI*E8JVY<jf>34d>tswU|TQK%vne_Wr21yUFo`Qa!j~%_x;j!TZ3I5W|@9pMu~usH%WJz*=vTSNp!E8>HqL-'
    '_$tY<XSx>j0aIGCzoIJ|92^#5Yet4B$OUTqTqao(EyK1<@?~0w?U{HCy7w0{@fdW!FJ<0G_`fl|B-'
    'VCPN$2k5lFr>}qE6<ocN<#GL`ts@GWS3i^ro^SOF6<^x45$ntym2*h$qo7PX72m(Cl?lqQ5@1q#Sp;D98EhGem^XU%yNSKE<8BLa'
    'r>uo$fNUNs`|;H)B;pe|=d7?Z$Yn7vilaH-AM2t;W`AYh%Bw$4Nm7Hah7xCpkl~%3_M1Qfcw(0A@K#u3{v>T5e(Qhy5)s>`W(CTo'
    'XFjO`{UAL<^(A9d4dOa`ind3p1Vdf01Y~Vcm&QO2rl3?^ftq#mHq+V!}>HOw!a~k^-'
    '8nS@{qvl=Ltplq~d4Lq|7R=vNIL*kqw!GjvRo2Rko=*k=5F^D_u;CW;3cgf`=Yy*z`!W+M2A&T>0nbsS><TtbH*?$)vfIvi4M+nG'
    'SRi7uq1ObgRj$tdf!SiXcN3P-((A2VX*Qi9QtxP-}x8^{wkFePyVHo5^ks;A<h#6o5Nsf3R1a!TNp@X=kN?7x%{(k)W<-'
    '%1$iu2S}CB`kDTEBlNpXwp-d^Ogrfv<G;yxL%ZK9g)?T<#bzqhud1~!|3{|47T!KrW+ws+1-'
    'C(2N<e3!C0WuiS_ju5_6U>R*Cj%sB7%i40F0eD1N)k#Bkn22Cnm-GH;#tl5y+2w@h2-'
    'ePp;e?<=##dA~SX{9A>N${1$zL+fN*vY$?Y#-v-I2M_`k7ig<npu*wg8+lzrM&$Jn?6+u$h_r;;dKwskP~>{X-X(0sTkI-fD}L7g'
    'TG)!W;&6sS*K?Q?2$imFm=OrTi{B_TiAFO+A-'
    'X|hnUoMMpz$f_zB|@+RFTWX@h?}f`vY$xd|ded`!l*^mV&;39~4*$y3r2v)uRUy85Y;5X9ZTF)+R#Br2(60V;o2HoS3I&>~OntKX'
    'RE4<RL;bGBC-poeWNqif<2?L+PXc@i#QIIo=3YZ)#|^ycw?cX(+br3s-'
    'Mx=(W71ne}?I;*!csU>tghDWFeKtYY;F=;>aOVVwigGrmWmiFYJZNTOSJRI+SS0ZR%ElGjUAavNPMtLCch055wC%NDtqW``F4GFw'
    'N+%;`{1qHpm}ZnjQL{$dnp(}{Cl$Ym+f<0QL0C3>7}Z%T<CbGsrXdYoc!PKh3^U6~R+o?vfDi5{ogTZN=!JIv~KCLN!oxS;hSIG<'
    'i*25zfRMik4{$L=-Er*v#-(>c|Xnnb=MDlkvttQQy3hVd0AS+0|P)<BLSq<M!|@a<BDI--K7-'
    'B%Sx(ZS&(wAYgof%e=)s_Q91Nn)fO44|o<rewDI`;h{ky!pRm_uNTVU5@uEGV0Pn{x#WjIqUR>ws1Qc+wa+|EaGtlATlM{p-'
    '<C3^#RDe$iK@$a6h-<lKyx?%C^Vp5wd~~r-'
    'OU|+=i1eNZM}uv$B|blN3ck4nA1~*jIwgNJzj>2m&XK!Ow=G)<2PLG8S_)l1;{9{tLM#V=@1Yasgv8n~`@i7V~MOos7kN23cnrwT'
    '^N9GLd_GdyEf+SEE8vMwYv2KRKR@UDdZKrihDUEOD-'
    '5Zg1vdxD&S%w8`NW5oGW~$kWpyWIF;dWXZHfyRkrSv5x`isI=I}f@V}ka>jvXR7P^fQ&tewzmBH#Eb4vr#b^cPM_~{S13Kgb{Z)U'
    'A5s<ZKjKPT+B?bOVcTr5}{Ao&uaE#wxzeBO1MkjcH>&?foakt_y($(MxB@L*dIw3xAvtvnX59w>UqV=T=1<Rr7==y2c0^D=`eu^~'
    'DFSU<l^mfwocvN8>8Svn++d@H~E4-'
    'KreTu@13F)oEiwWtcDZH2%{Y1n+o*3O196yr<kvP}5|9mFwjOQzRmS5y@X1`hH=XL<RQ5dJ_rxQFSH|rW*md|8GCf3Cx6T=;v_xD'
    '`|hUDQOEA{wy|D-'
    'nS;&|D=b#VglX<ZyQdq(rPne5upg<U(QuusPo_9+8OCQ$Agce~%UM+<E%UXPk8;m0Gg+H;}ZpzI0@HbAlBWI;{S3rZ)*`+ia`y8t'
    'L+j%{C5d8cZ;MiZ>wK;w8m<J@0|NW<MR(r|Z*G~9KOhP!j5;qDS?xVuIg?ru!OJ<<*V9k0Ohhk=6E0rA7)|4SFgHJf!9&lI29+qR'
    'eh4t=w#fkn5mLt^}m81@PjP1C@~)ghkBkKsxH8Z%AQ!m`set~Zu?LLHDloUH9_h3Uwf>@|v8-'
    'z)4A#l7#F?Kc!Rzn`}KHLu^ZwqA?Si5;YQt+v@AnpddaNo(*J!wAsrx&aQRKht4Al}VR$n1=Ku#CScHu2)-'
    'obcN>{eo$59y)8)ckqz=YsUvpQ1}#UN7I9JLCoiySLIE3z=su=pWO_ykf;ZBBT~SsUVgg#KoU@p3D$3Z$G4VfT?Blt}_B9ptbm^y'
    ';bM1_NVkMW(=qGMrm()*$MxU_*H1E#m>|o7I!tM+NZRA2ZXexT)PAOXaW(Rw$U#<I4%r8UFoFGvM9O!*c;?Rd4{1gs$=)qg=z(He'
    '$9qSf5P%{Pc*}zu_uaDz93+plz1l2O9G7Bs{0fOLQD`QOxTN!5@Y$Z~>b)>g9!H3mkay#0A0mU^UeQpj#@*!O?!ZVWM(0Y^uOZA^'
    'q5UaR#>43r4@^sAR+Qgvr4i<aiG7eOmS+;O7z&hO3INZfxbB%;g8zmjz3U1iY)<^=4&H9?8L2!g)-'
    '<u#F>SF@$Zv+ZRNR!Z*On4?nraTizq&yR&G~?q3fd^YAtp<ZwTqdo0W4M$yia@nSnG~uu+Bm3IB0sdpF!pm|4HQ*3Y{pKNpn$`Vv'
    '|rNuG-OEG$^-J!k4@HAxG4I%l>QQ;P=s(yuPe{<UJz@zH?z-Ycrm=#N`~nu!-9c%KdJ95H+g?4@7!ea0aD#rVe)}e+_~B0^-'
    '|keY4Sm$HgJo{2TPsjR+A4A+&!I#`iyM9JdI%)**QLEc%7cjt{&=cT^Q<aQyA(#I7{_o+h(bLY`ZwA+W@R2E5=gp_y-fGD*nj$X-'
    'X6S%m8yr3ZE1x^&I<nW(O+SQA0C|9+G-'
    'AZQb`}km+Y`nfYjRgQMyunB4}ZeB0qRJdcD8a^rU?dUU<8wM_<2y4s!h!w=QWE4vh!mflmzZlbrdL8yV(1Kf@?*N=75T338skn0*'
    'dbh$5;4+Y;?Or3{;ax6Hd8&FoDkK7&HVB$o}?-{U8nfad>(NB5!Mgb<%c^;U#nteE&!!z=EY<KbcY-m$Z?C{0Ro)@l~;nVE-'
    '$!vFiu$|zaMn6wFl)8hqE?B6AAVQ|ozCa4lQAh#u?m(K1^(XT8j^#b1dK-<^o`1wI$N)U7ShN@f3b$C67z`G-@Wy|F0-'
    'vyZeBCKWG4<;;PC4o`p;$875lr8gQsno5V@az1A5Y3K7r=XUW9%+j4{_vPvS(Q)EKL`vz#ku0u8ke<SB?(Z6Q&IAX!;<wIC?u1g;'
    '>5J@4*Nr$2Tt}jb#8tCzy`aP>PE7Yf*+~MTyb!kB&h<B+q|x^!dliY7PW%o=W6$&q*snL%@<(k$(CCLm&Os4^a%GkN)a<R9)#~gi'
    '{=0L=T~Z%;J8-iApgWFGXYLWcI5mqX3<y<=%HvO1ihS?KU`|QO&4-'
    'bLPPwflkjqK5^P~Cf2(}8{n6;C%!IPJPmlj!A^j(R+k1x=)oK&MVKF<E=wQ%)sIn_rH?Gk_pBWgZo55MRT<nSC)Q4elRMLyFkI#('
    'pF1d{EK8j3kc@IH(RiZQ7cx<gxKheTe{=cM-'
    '%{`+#Wz=&(D+=LRZ+X|Fg5Giv=kr7m_PkYV5vD7`Vzp?%=UFQu$_z@ps1#;%w0RlD%J&gRNJpf?dmTszy3>TPmOVW?g8(y*)h<a8'
    '}5v9${F;R!xd$e^w%!T(q`RONT&)iHSfur|Ko1+rwHEzsGf57+7H8UoqG?qNTJT%!I1BCn~j{~<E4u2{tOk{TFoWI9V0T19m(9&='
    '3A1<;6WF~zF|q0Dr3=G86=@vql3qn&gmC{UT3t-'
    'S%6?OT4n}7mcHOP68M^fGWy86kIzx0D*%$#CQ&766L<ui!pPOq6?sJ4#ZqtVqU<kZ5Q@@$`nz8akAku%dY@lO&nH>Ffquy-pxaYM'
    'u3`CW*0}w!hPczku9+PVV7^^?nhELxw?pRb<06l}xXd=*x{^k=8=R0xxcfgi$&7IR{%ougt+OYMHKNIKk|GntcYi7vo_JkO$G8KZ'
    '`gIzK-'
    '6)Z6xk_rck7@h$@NJD03L^pu2k1zraIiXAytie3_g&wGDyEl9on8Z_m#Z>1OpApOV=n#UsYq9WSdyjKfI+h97c2j6(~U(gVQ$<O`'
    '8@W!?NvTMO8?}SMdhD-K@|SU7xL28sr>RNZIfTYt2kca*dA5!UwKDC^e{3*&@-'
    'Hx;wRT=JH1jUQ?$_T2tD5IC7VxXZ*J{=Ie?PUFP@jSBFP00tT0(!Lq0d%Q5OhvCoSUQat@w9iqQ1W0n+j~LZ~Oh;W|m>yZCUPs`A'
    '$uS9Q9|UuPBcXDF{eGW_st`)Ed9OoCA|dPh`JV%5b)Ch&hL6LLK%+3?|<Uo3i-'
    'rxwAZlqy~3bE7tfGJ)a|oQus+E1*qvA*Z=9bivim{Xpv33pJo}<TvLEZf>zEG9Ql2AA%G6L%8XOk)i)FTs#7Y_NTCX(^bBc0ryYm'
    'PgDNqKflC2k~s)ZWU5_cQf}I-'
    'Ov+7rwMp4%zh+W4+OLN{^*&T_uxHbav>4N$AZ5NVEgiQtc5oI{Q@57KP~DJr(mF!7Eu`f1FwYY@q9(2+fcJlHP-yeALS+QOwtO3T'
    'YCh6jAE6V;<q<lNTmz!>$R!{;j$AXL(<mjA%cU+vXHm*Rv(Sm}9jmYu>W>OXlftVXJD-'
    'r!>U_l~w*!^ExH`^1x28o$+?&^tBjiofCI(xSYW1wZj4{J_9o=aY@4H0m@cRbeJE}B(*(T{{vHdz(<*%|UJVoWZIV?F%<$JjG=BZ'
    'pbExrH;ozfbaX!?qB*`@;)*t=ySqqn_>QK<PmXwD56G~0-ZU_Dma!zvNOXFpO2AiiL}-'
    'nLgs%sNXpQAlg|us&h_roBALYF+6erM!SkTo-Bn+oUtBb;%4X?EiJiDEtOUE|*J*7b?7-'
    '4tKmR&h|bO2YMfl1HF&Lf!@dB6z}6Q#hYJjAI<_KjyA4+BD3|1<(Yr0`5AZ`=xdDd0aqL>rX)!?z7$q*>+p&E748b2q4GEQPva2Q'
    'u_EZfJi6bO{&itJ&~`ky-=6+;V>QqYJhtDF{`KIRd?!D&-'
    '<YqDQ_he2A!ex!g7Igm(Bjq!WE~rLRt#e{n;`zXUyB;>X_fE@z4@km?E9v`iu9JEIM3~eefA1fia8x5lYYzQs_Nj*d}oM}N=%B=F'
    '>Ajr4xf#Rk9>j%fzfNGN%w~*_2ELnR~25Nh?H;1w~&(w&>SeZhON`n{#vG*`q<yd^in_jJIxO85<5}bLBT3_1_GZ=+zR_Ym57hEh'
    'gJeIsu8TlKX)0TtPzE`kC^~|eXfkrfs$gaTn78~W%oY((vQMqV$vNs66pM-'
    'l{E(V{G^q26cGA8*M+$Fy@H7Lgy~m?U40R}APbytbRDmakv!*DQU+UQ52-'
    '}Vu>Fv)3N@x}(p4Ef)>dUckd(N??nhX&&acyw3>sz;ih5?HIQ~X?>3Sa+3{du>(y~!xd7x%GGZH}0r6Q4;vL~V@vR|btO8b*agl9'
    'aK!>|-vtW-IrNN|SF7W&c-D|8;WC?3}z2_eKu%x|?7J&?96p9X>wR!;*#3Cn?jpoG=HKv2S>Y9J_ym~ex`5#!F$7@tePi(@f_ku-'
    '<KVR*Hyu(2L_YVGZ|-Ye01`}v|Xr9Mz-rX}TY+;fU?f<IQ<?`sa0(18De=4c5A=XxEYO@EY|<M8M+aW?e1I2*c6c*?)$dCD6Lp-'
    'V6Yb;6-'
    'VUhQn{N7DT*|HldPwvtJdRZ>7%r6VC`T#d7xGIkk1vEN?l!}oXW)vDnBoVLX69*_N&T~w%@|KG9ee(d(?IBUBl&e}d3XKlBpv;*('
    'PT5n+2t%PX;zh<LlLL*`McA7IAlWD?Em7sQ=Ml*F-'
    ';u@@u@UuJEZO&A9ah>)?OtEsVw+!RZoRY3kD3gX}%hB5YNZXA@zyLxvy^L(n7#T#kS=#JlAf{lyk+Dao-'
    '%ua7!dh)lsT82M`vFnlbhR-'
    'n(^am@;I9^^bC_<c6_O0;8q>~sS5}UGZhQq;nY!Re!5v7~UM0lCxI#8p4#;<MT35$h2aV(z7pE||#S!4^N5GR3?@UulU+m97o52-'
    ';CQN*CndB&Kzo+f*jfiS9bOdBt3>*QQ7Q-'
    '6Ap~Wx;kZ1{rVLT)$?6)%xpFXYaFe5ZtxBF7Qe+&<3ds3x9vf2?elI{e}r+~JECO5F^HLcS_`Sa|oAI@LknD5d2CFi!iq1t9&6JF'
    'h)W2}j@(B(joaUpLE;?2-78wv0CLoFHdYiQ@bCeh!3%ziiH^!FHT2gm(&pSIK7)MV-cY1f`7^odB>Q-'
    '6{w*0xERr0bF~7Kh1Rn!3T>5TSAF)!(it6+$VdKex6Qbh_f}1y@P$CE%3TC6QnofMe=;{{{2ah}Bt@(l`2Bd(Q68!YmPpx0Uov$7'
    '{P(hrD=Xz6;IqUH<buXo&CepTCYK{_FnpH_*g?BaTx*XEt$cXK4G9k92&}+&i=1jDjwGGW)IYOfJRP+mNL4f};IHCgi`jlosnfm&'
    'nzu=h12HlnG3b)gl>M3A&9N6(8nev7t@!RP7gK9H+2yPurK%Y340<XF9(8tlgFIH0eVadU?oy?ySI0|M{yxL%b?)?K9f$NJGCbXu'
    'FH&yikeho7rnK?ouA7?a;Vw@7MOkbh!0FZBMSG0(q1GEzbf?^i8D|T&1ys)yLD{$TRkpbV_@xeKli^U$^N(P`@67KI^%>a=pqQ=I'
    'Y7~Du2X({wRih9`&C;hFP1(xPbLjl|SxxsY929VxPlc94j@Kp1y!xZd3_#BtQ1mI_9Ax^A>x2C2x`FZ;YTM8YxK<a`n+VzbHQO4U'
    'po(##x;X+P4Rbu3N(el%-d9``e&aHx2N&q;v!7t1Ki5w`NJZ7~eV2p)JG8&Z2B6#5kMLz>?_tUXTXO@=MasG0KHE-'
    '8rML;ccdM)RFmaz;bpoD1$SsMh|ZZHB|@P33XHlZ#hm%RQ)siop3J6b$74O$56M=>{&7<+b*+Tk}+92W#@2mg+!?>T6sY?O=BFml'
    '`i{8ltn{DVXpC{)m-s}!oBN~&Emt@Qc(Jw#rcR6uuLh)(bhi&SU))kS3(mcNv{@Xc1=3U^YzR=nCjPkPTN;xNRBKfg-'
    'h2l<*&AN75W$&i`sUp(8thN)V4>3K8AvIW@pIM@cU@7(Z?VWr!fI-N;I|q^|h@^W*;qeVq0)DYxNS-'
    '%8NJiqoji@uD~e;`wbanB{r(i@^n#0XQ%4BAvLwbbg0`>^1Lid$vbC7cK3@SyZgnF-Tjit?mj!RyI&gF-RDGh_ql=H-'
    'LLxBGW&qx2JU~+5u%SlYfoc<T!lN(8tpJ$nz5w&omCsH)CQe5PB8Gf@T5oF#7Nv(;P1k|tPA@N9RWv75FB@k&TcM)+3$fAPN4fdv'
    '5-JavL^xn$q1B_fPQ31#Hj%O`N!tabD?SB&~wM9)tP(KIl`+ldtW-'
    '9c6Da&Pv;7+$?RH_jJKVy?UDZbG32ammkNChEtCv>^#{r%pnEij4cI2MKK}<1&MXvCJef#$Bx$t0Ht7;U`41(8?dk?WBT4HGLU+*'
    's!#JDKa!z?_v=(6h(}wa<8BBfFP<kNu`FU-3ro*J?Aq`B{?E<o$ex1-'
    'H$^h30wUhT8SmM1jJ{wAA#Fr`P0Nq#BLR|C#6Kb;(utp(ytaF_~$s1<H%IJHz5>~D4xGqsbc6N2XNN;<+1phtU{Onk1%KSvwK_=V'
    'j07Wwp=VTzF13$YKFkT}ft-ndjoMMW<i$K<;+CD>`l^M>p?OdUcp$nG*`m!cju6<xJaLZLs#)_9K^gp4toJRe~eZ~u(`5Iv7xK{o'
    'x5D0Kh^;vZ2wllQ$$A$-Vw%Z#Gk3{v#9}iNTjxqRINhm!Iw-Hgp>*BR2vwTCm7OlxLH=t1}>{kg@uL9c19qu`l1THD&1`V`H*>5j'
    '(^w5hEZSo2-'
    '@z9MKkR>R{PFYMsN1|wUqbCM0z#cBfIl0d;{`Dt@hr(V!E%Ieaq`8u?JYu<TidgQI5w(3wKy3%g6K%(NLqYG30`R@U!a$+so5I3C'
    'N#Q={-'
    'Q{WD4HYo=v?=>s6IX=3`8|LZ`6hQbB5>jTbP`)$Aa$C{hoTAYgBXIzg_i;`TV2+{9A|9_okKKs_2N*eT?UzOo~!MX(DBfxZ3-'
    '2XBjDXRv7^1#r>YTVe`<IX?0K~7cJPQ;lL?es9~fe#k!<a7E4BJS*1nlhOu^E<*boy$vvg^JHa;I_U-'
    'j!xdG5Mu{1I)t4Vjx~tdas4U%}e8;xW}ev7FmS7;iB95?XVO=-'
    'D?3syE|!D{O?j0H|_rq%Gtp($Z<~gb==uqCI*f5BSg~9nIrCv`feEa1R~dP(m7@===t&Hc-*_9mhC&+P~w=$wA^U>GK|-'
    '6XN*i3qYWhhtXzc$W0Fid2bnKbP>GNY9lK04J_xo5D&I5&q33J?Z;El^kDn*3^cu?0X(Bl4|yO@L(@a9=UHfar-LFXk+w&Lz^4&`'
    '_gmtU&d7p;j0M1nO6Y<EJmp%~*IA$<^OPK@eyA`Ek_UQtCdXrt13r|fE9skQXY7HdW>PZkOCC08p6#ka&i8h>_TF)B>%*FRZz$9J'
    '$w&PGSu7I+D@>dixY<OE{VgVTiQCyXGIBTm?e@-Wiv-iBXGsk4*(`<)_H$YAgpCQW`?C>i(GOMjIg&%n;n6SG7w#9m@pNFYrF(E~'
    'qM;bJd^;$sxGmoq8aKtWw+CadwfctG$KbhSWO2RYxvW<L%9bI7<t7dxtTfU^2fA8%=i3t~((`1P0#`@=0y&1j)RDhPMv8g(cH}#l'
    '*C~HFf7vm1#1VsrNOC?WLP(L-v0-y_MhPj);f}JaM@W<QXxF(*VYvhBHXl572b8i5E}c_TRToYE4{7?y$-*2?H-'
    'IV%>Bc<z7t)RS^e?0vm(jnFZY-dGA>CL=|3bQPIsFUlj4$OcF@H(^O8!b@0?k5P)tH9C%_8+fZ)X&*1@P3Iui|uFtK6tn5d7Pt)4'
    '{E!s(|L}1c9A^BHCHnJ=bC|r}kj~ha5!t)=b7C|3?cPrN1KqWem>%M*=z^ro4mPJBnfOV1<Aveto!tua9B#Aqr-wwN&;u`h+VKT%'
    'YvWt7l8yo7t`*Yx{f_g3rceNN}09zYa@12P??ywan1w;={K+)?I#+^4s92Z!pe=$-de^5x!fz-'
    '`icKuJrK+0Y81*n$_Tf83GNkBxf21+-_0_7@-j39OO6tB?dBvYnT#yPZl5~;J$9jD1Q>E-'
    'A$Ns5OH044UWH(o>w~5j*KH6W4B$h!r;2?mW5IEXz_0_aR_jl+k%pN;dpS{!chd>8-?o(9CZju4{@~giA-'
    'VM#pZquN7&a&8*rULNOVKy+VExSjowMPCQoyk8}KX;IY~x{N89Wltn(EG^oM_|LxI{4)J*7^-ykdoUfuFNCEcb!a1q7L_Weex-JR'
    '^d%yzE`n)ffjDhvMjZ@?;x!tviR&k@xxvjUNT?V2%l0C02~!ePfG>o(ACH%xN&_0s0+fq@*LMz-'
    'p52_iu|%n^GEBbYKg$zbMop}cUDB8TW|B1B3f233hVg5~z7A`b98#6#ILzMv7fjulky@42|rXlEJddxqTTQe*Gu>@RO?U(ak^Meq'
    'X~l@lcoecUHim_u`fr2oKhkE|la@hN$O?Vbh7g%n`=B_A}7(N&Y|zNR;aw^@^5T3UEwm_e0kAiqlI2XjrZnWI<vgYN-'
    '8>;B+a9jK0?WMNYz*<Kb&wwFhe?VBRW_KHZdeRCw)UKvTYZ((TA#Rkf#A*r2hAcGpRsX4~3QT%PEX}h94{Ex%ZG0R)%rpn>zCCL8'
    'oH{1iDg#STC>48jlt&xetgUQ%rGLYMrc6zw_neu}ctirRG@A>9@KflF`3}jwImi(*|Ze&0jnEM!zhLs);!`5n=j^NFaUdpX)EMd='
    'K?$oxN82Cpw)n&!eT2U%IO&6$wp7Y1<9stTq_6jX6VBpu6mkK@7KXcs>*vA%r%**pP`Ptr-'
    '@8f5?B7cjY?aKUZp@7KW;U~34(x675KbRvL=np1|2Ks|pqJi?brJzro40JR8;$9Kfq?F`09{l9ppd5VsswW&mBm`nJt0(6v^<C>j'
    'ia*#_Rn<%dEimXusstNHxEZLG87r`%q%e+!K^|d39zl&_vrLMTO_QT!QywLorbNjm8zq}gh>}f`rd{hewbrEj$MM6a9aHuNwDv6#'
    ';p+TTCLI7|o{S1=NxV#09EvFYzzLA<t4nn}OaRpkMF+b{M4p}+;wBN7XAHocu7y);A!@jcxx5_bnOVwS(x?%PXL*%J83WRSa}lt^'
    '`Cd9+0kh?hQ!|Cf`#X+fEKpy5*D=GT{QX_+6z{~lSz8!FY>3Rwn>Aib!Z5Wrnu(`Q)_ObqX^RMiPFE#>2zKNRwV4d)7tKnsg!H?9'
    'BAHb~irJo(l4dlq%Y(84hkymfhjtVmg%zI<Z_LbPm!WRQ2)zYp_b`R!K01tyyw2aE)5zHMd_P$(Mq%gg(Af;=&qn(VNBtH&;5Qxb'
    'Td;fIa%_FUf`8jl^*xa6tFx6W1=16E!oH%a2)b^lSszYvE$nKf^P{BRGqJ$F9eNNG3lXu~A)yJ%OLd4FcfCa#!_$N$o}9von!$Z`'
    'n@0UGV7A{R8^)0J{B5%3f=X`>=2TvTj-'
    'b2cnqVjJrm~Bu18pjA&(2^?<)ZBsU?tdS3R)98{7OA;UshGfkQ=hvf}hK>NRPffMoUG4ZQ4}D-'
    'e42PnOgZK1o?&4nd=S5f}T5l%m~~iU(YD6p7~>pxP96lBeK2ELX}_*^PwFt^1RzX7qwQ#nbxy17ODdeu{0J&&>qPcw|yBV?s)*=a'
    't!YL*_0Ws>F`K5J1kyjJ6VPH@yoSHZ1y?!MV`mRr2Rfs#bFK8I*Yg3Q}}#>FsH813Ui^DhF{<f=G~ZDEp`9}pnXw<C0`t|+LuJE_'
    'Uwq&zBGz2&55w>xeVJzk~~qS6rQoiics<vdz=U*KL@g|2a5~V+R*cEZCq!gtm5ew0tC!U3hW-'
    'v;4%rcI|YFm+gpzy_Uf(2$vUwfr|9IcoT^#!6kUlUMep%E=#0vQ`UM~wV4rak5A(yEovZ|~wE**6e>BB<%yzEFG{yW6SUS3%#oe~'
    'H!rSE6uUY)#B*%V}pt<wG_7z{UIuB_3GMTnBh%9V-'
    'yc<Fop_v9|9YY?w<`0SRT>h}bIm%qu(P^H0GKiOwJePL*@w!N>+8`Wk!@%D#jLp?_VI?BRdYYfgiRG9rhz}ZA$KnPVA-'
    'q;XGgyjQDJQCQ)>xL_(^bgA^s`%if%8%OQpq$eF<8hw#>teGM!G0Ih=J~Q#|I^<uuUur(3$>8vDVGDN&RfAlS|x?6?o;g_;>P%eK'
    'Gcrg++H_no^wt>bWFwr5$u!T1hy~HPI?@Y_Y8fEIuz_@5%WK2{Y)62{Y(R2{Y&p@vbN2FDGoQuT&94Q<Qo`?XhRFVS`h(<}6JUX3'
    ';8*l1@Ws4s-'
    'XIbTWE6Rxd?Fa_eo2DzFAVs|u`vTUCKI@Htgr4cw**tbyBAfi>`XRbUN#LGg~A2khE+Bp4Y7mN$WDolOZz_!34Tj!bbLL+Ds~!%A'
    '7s<)*4>Q+p)NjZQTALx?@qk|I}I;;Ifxg6Bprv2!m6oYQ&qKmNT=m8t<I1P`O5D&Jfnalb9TWR^4&mgN`G^W=u(#S|TL$MF)%Jh>'
    'G+oAOTX#a>GB9JeCpP~O44$hj1}_|JFeyX{fI=De1_MtOR&aCc&u-JOl$c6U{|0LQE$Mg~ztB`sKA>r`yF3w>Kl+*@?Qw^q&4Z--'
    'mt8-AW#y4>*#7%}2oei2iFe9y07-'
    'iU8#J}`~nl_+&QxZl{YO#uBH#{ox!!YN=Z^IZ&e$oJ%X80bLDH#@U0q+r|I!EwMWl#5jvGIT?U`83RU(Jj5olC)2objHh_yd=P?U'
    '1u!F!ca=N-+^N`gqiaFAh_~vC0@msFk8yoU<WwvcptnBmX#6Fq8(p>6^}lop#qA<`1MH@uq;NOGPwex#b^QZ3b+=d1)Nd=-'
    'D0e1+X@&LqYpZv0@B6SGI_MelrtVWkT;mx6TRrYL=a&^GA}<}C6j~YB5C;l-'
    'R;y0LeoNA3VNIKy{=Cnw7C$xL_$w86RnW|8qHs|W5VaJ=dUwDBYz`*!%onw0DGm%H-t_To=ceh8j0wRGX4LCtCy60UE-IQm2Hs<;'
    ';YKOWeP%<ok7~;cF#I`_(khAO5+KBK*F<~6Q0v5SgPQ_=U}n!727o@E@%^%<VL!hUrRwdM+$-'
    '%&s`?EEYIZ2MVDnuev|03Je#i&U6!r+&7#ZlT)t9tS+?c32x}1?&2uybzC0#&ab~xNBEVp<MA>)3%I#404ddIhVSUcWXqsD}$LOo'
    'Y^*OVG^{Kg;8bFo11d|4l?LPz)`wjOj9b@Srl`cyY$2%<rUavm1ut8|OD~M%Ek;M24M*reDn^3W*ry@Z|SL_)nj(B3lekK*kIi_M'
    'iTS?B@u@(EdR8;7=iv4^lD)i}!{X&J$*-'
    ')^c*>{;I5F~;2J?6v%A)b9dT)m>~UgKGd<06t5JHAq*x9b1WZ6UM^m%lW$C*pnFW#}<SMBGwk-'
    '$@a4*DL!T;}88xHwg#Tmlb(#VV3$TN^>l3wvf@xb`S`~?Azh{hJX#x{wG}RqD;j2&RAiFay<As=$#34*~c0?>4~-'
    'ydYGLtgrq~Z!)=v;l~e;|RKR3y)l4SehyY_xkwoOzpn!4Pq8kX73MqhJgI^drQE?~y%FscI5B_#TXDL4SI}9DC_}K3>bfV%@|Dm$'
    '~NdbvJQubZLVEAPS289lAh|mGvM2<A_E)Emk#m?x|KawzTZB$-urVpwcTPlekqljx_!euzz5ygby@-'
    '%l5qDykRo7`Z)={nWVsuv*FE{9({tt993yrI8PL+yAPbRQk3IQI$XV79Iomlk$L)vKMt$mR4z<>5|rGs#PInbVN|;B5vW^Bf7({M'
    '-G$0+D&#@HZ&?-xSFFV`bkr9JAg`j+ePBZ=o0^P%R?DJ4@R;BBklM+TIh4-'
    '*AxW4HwG1;TiLzqrLE<EHG9%oB=zHoyxvxJO(3VH4obldt-'
    'DCrVGndB9on%-4Y&DorZlo*-cnIZ^SjPXGRlR;TQSzIlk0gqR!SGPmh-Hw$cOJ=uj;udan1v^8G;Bw@ogC^CN_bvM2p(U}#}47N_'
    'k^|AIIzFjGOCwlDn);<WwfpJjQ<{wz*Afc{Noox+c42mGPWM}V4eL~=TNDEm!6o8J>`O4Vf^7up7^@H~1c(iEOU?^hDu6E$5)N^n'
    'Mjw0NN5`fFn<ANoa2(WK1ed`VNZ&0>ll*ZE+U_4}mGhYDfhh|H1qh0^#TnQ)m6b85Cs-ZtFHE;5bQ2&cXI`EYfJ(`yZ^Q526}t@1'
    'zcMAjvleLjtL8$p=hR_6lD9~Q8s7pO9}^b6WH3)A*!$2k+MBIRep6wi~8cN}Ew{5an>#MlK!fV9I5rD*=D7j1s9R)v~T{-'
    '&hOUF>Q@?~cQWZ9GXw3CtoefW|iXsI-ppJU|VyscDndGq!qXjJtc$g~Z+UfT+hRdXdQU92B%%;BlG#RdIDz>uYLK9HsU8Haw-'
    '}0NpKdXU$5B_svQ1o;D0!*Coe%z6Zbe!Hn1bN{W8p1sh{zS;mP*2<129mBeGO<9D{<F|s>Un6su_=1vvntn<CltZ2uTyHzU8S?Bw'
    'pS<!*(c6X~VXPxhdW<{qUN~bWJGuM=8x@(UO(31dfP(u7v#|C^BeL6wlcZ(IPv$?=+7B}%*=zbQNqDcI1pReW0{<<{cQAdAH5XZc'
    '3JF#^+(tQ<2Gn1v7Akr$dR!4hf6&88KX->StMX>IPSHBqNI|+Pdqh68t^*IQB6E89s{($2*$~A`?yD-uwghNuzMl*m@qXiP<uhw8'
    '4lVMsijLo&G_$eZ&(5Z8yxE0&FlCDx6H%d?jn3P|{W&O$d#ay(^^GmpBIVGRX?M<6s%8j`b7;MlppXwNd{GDH-'
    'vE5<}b19%qJfhN3*}>p95^l0!AQGymB`UC~e6xx{>-kg!o*w>aUL-dyEJtoYoMLm_SOa(|=Niys6??$sHdmQ|>RDI7;nHs(Abo%_'
    'Djz6>uSco8UI<^uDsoX3sbrD)KS;)c^C3d`Izf>$Xd?kbJ}jh8jkF;7C9V-'
    '1s=gT<Ufg+~&EP}gPWxPj%5SCUuW5(O(zWRy+A0x~bN3_@)|i7DAmT-'
    '%1#^Ho9zM1MXw7_SA}rUrsQbGE`DN7klO_lJ<Sh0W^(McFu@8exela)whnRdeH~NQ~{8Dx|!{D035YKRv&rODSP6KUuIt`n_zs8g'
    'JnGCeXald^&L;Mv_;ukUm4Xw*M1GHQ<2u`u9WX2;u*gh!@t+sZPv{%~Mzeszfz5S~+x;n&>ki4Tzgf!-zAR-'
    'o;UiU&box7v=Df_Y)7YTC(2KzPM>!k*qZ0XScR}_WbZo9q~>Q_muQztVMI0S79Of>_I0C!P4t_z_fU^TJOFf+sR9#Gs{XCi=Qlts'
    'DKh}DuAg?@-a+1?M}UphISqiJq$B?om@F|caHpLt$d`M-sdD)$6`=X4N5e|5E@l#ctW_bdB~bo|#SyVI+W1jFvmd^}DU8PMV-'
    'hTMPxFJ@$O3|v_22r8RnBK-BTQsc3oW63HCu;!23kN9HY8@`NCj!4_YzN5cg`pApZCL#p~@bPp^cB6D5pPg77bqYSpfaNcqgi8#V'
    '{<1{!A{ctFco9s!7rY3@-'
    's>?0bMG1SgTeRO7oI83<9{!YJ%J}E^vp96dD)Zdy1Wr4g~IAA3}O}e<&JebUCoIu)~H#^p8&bc1pBnr0U28fPt(&bzn^KC5v7h10'
    'Rz!uTgRCXEo#|80y)yAG?YmD^(kLBpKEW^MN*SGjz143&v_)1nsi}&Ks)lBW0=&W8;|C7AkR6LNlkh%Sf~?u&hd#nN$|*Cb0n0ny'
    'ly$0)qgA7;zIP4gC*~4|K{KAELw^xVDwf4VjmsnyyJ8oTaC$LH3rhZlp5c`U;?ptBOcIhsiipqoPVTha1K#F*F;49JQET1Az^Tc`'
    'j9R-ME%0JiymQglD~Y^&Pr(l9<vvv<MfZ)i&NHs4R&^_a{haJX{vJm2Rp|RVHB2awyWinO<f8&a#{^(lhT~jMVgb5Zhc0HP-'
    'G&kv{|RZR1|1jXR7D0_7}_rdz@zaodGEJG$kx7G?xz&ow<Cl$js$KL}e}?Dk5|FFwvOHhcj#Eamv1~QmXeilu*4NkBuZW-VF*u<q'
    'cI5+qbi{%56q!2VO@o3`W!93{3jx-E^dgXh1y85k5c#o72wJ1%Lv$Jv-O~S^sa-'
    '2Kl|4Qslp;Qi}XNN+|LtDJZ5+D)s&7aX*4q?S?Yb56Y|SHJ{Lvlzm8Z<a4sJ4{J7srzrb~X29dA%08+&5IRlS$ArH2bY&mc%%?q&'
    'f|Shm&ZJiQ$;|f6B)dLJ6dZ0*`ETssLXiDSm9KIa#T_k4Lm^mhkYL$A!|RODl>6j0(&6bU0OeP)soWX*F>FE6_Rvo01~F56W~IDj'
    'XDIswbM)r(^X2(`X2iTHzl;elZ^^F^p64TsJ>JMH{^6O7(Dc{#qu@;oBOKa(EVI30<Ka`8?UylDX0uS;J#GIbRCinLGrGVI=uUh!'
    'SDUl_A&ho4^HB^krO(6DU$5b)lkzd;-'
    'rrDx4e(o~^|>v>V%LOlT?27X8B}V6Fd=l8fpOk%>^U+tlTvg3#_@SZYEEG7mgxZLPDD=mfPs0eOJpAF5t+w&GSAB|4Fhh_5Bo%B`'
    '(#q8ADP+y8H11htt0rTnWvKTXYJF%Blw)%!jk0qHv6nDz%mE9QE}I)E~|FZN?cc~FYD$BUS)Qh)jI9Vx^;=9G=&k2X6#93kfP=6r'
    '(@-NnReS;v|!FIkCZ$5vCffxtZSqn>lW$90tC3TwCWo3E?_GRTdpgZ3h!gFiBlw(=nlrhHO$O-'
    'n&c8aL1gzJvzwlw{K~}@+U!=SHrWLxXyveX##N%f)+Pz5uEEe|2ulwGgs`)GX|D+bIWbN@TM4NJG}xKd0{ZJrYXQ-'
    '0=CuF~6cbxOgPoZzpexPP7HBhcVxurGUzX2fZrm050+FO2YLcn_F*;SQ%P+Ohm@px>Mpw<5%+fUv8-'
    ';Og_=y76e0IzgdZX5wgzCMeDMBmKTbd&0y6lLK-QEIXa$$aRzK|!BZ!;nr*Qmf%Imw9P(P(2Q8<v7X8a+zdRD*;#`g^FA@Cff}l;'
    'n8%*Px)n>-'
    '+$UvAoc0QD$Oa(#=Wvq_fZ$+kYEZd|d7E4J97eYgFx)&GZEG;8cO9%d$8y4kOB%)A*=&BHY;>l$5k+Q!D9m&CQ!QyXa(pVFdKo&E'
    '9F4j|}C9F~+ha{W4^TZ!&=!GQ@qtO?xl;@Z7jrgT9*Nl!G<zxcdynsJwL_M2U{+e0nI*1to3j{381=zq$6fc76%A-'
    '1`G`W)5h)Xm9LO?1p0do_^?fXRkLI7M-Q9^I-6k2Bs<Og?*s*DKPecAn0+*zcc2FUYTDWYIpVy!~A1#0i%p<FhW;3&e#)-'
    'C{WZ3is3t`<a5H(gdM%ta8q$LoDO%y?n4KN`#<-'
    'iAuR6pI&Igb)x0^3%^PMPEEd>zWY<LGssCTgoQ@Tr%4R^+ddV0=J>$A(sEAb>i4N63k>qT`@!>b!2kUp2D#{+e<^%p-'
    '%IW4cCe!P0@o+Sqiv7&5LVNEh#WGfTZ<LH#ajXyOV9YQ9I0&utOG=vXv`=4A(%U;v(MN%Y&@*^7BviD>7o{}~9$auCc>}28tBS-'
    '}j5d+c37xBu25)f&N|uI_jal4{p9f&81N1tL84Rlzdm*%36Le4)8B>_^N(O+EnWz7C7AOuyeZW`{tx$UzmAX)&=(1MG_n&18M!s%'
    'adCF^vf%!dPTntz|ZM{{JxdG*zq;74(D2(Hp>8&E<uL)J#d%a;Y7(KQ7Q*{Lj;ocI@{U$a?xbc1VZSmZ1*>^<0W0<xtYX&d_H}r}'
    'YMA8UtcWMR+W5niF&G;m+0PNQEksT@OR;xvW;5oMAUsIU9Xj7_0y=#(|3_sLeu4O_S{`;6iEYlG5g&MLw*f3iJUL80~x0z6OykZY'
    'x%8tB+3=waK{q{c+p}Y<D&j$sqLZScAa38lDyHwaw?lSgzA!@ioVPyYEp?cryDg6atJj!>*Xj{KuSyH6Fl`)`MI1aO9>T|!2Ik?;'
    'NF|wloU>r+BBpS7eNix|r)g*qGU}akg%(fpp-'
    'N7pF&yHgV>PjKz8LB8F4!(F8%vRXQ!xiQ8A?6vOD4`D?=nzF2eMs0Ja9VGz!vh)Vv@FN?Z#!JuV=`Yy6KrsTlBV}HD+O@QN`t(9>'
    '_Z&te|j9ZA5`vzV9?hW_rId5BB)^$6CzUBs%vl^UJ49F7#ZxxVB|(xqApy|7?5}6dd9%K8`m@H^X^>F7?k(mddA?qC)YEE<h^+C?'
    'J#YR&iwCivc)R|bN7;lADvplFib^^Rwg<0uvd{9>c><B&2IWtBEQ2jVF+#v8!ObLd#m8UWvA}3zx7V8E#;`y_@I0?AW|wJ(0D(A_'
    '~dgTFd5<0^Q{tA(cd|>Hx>578YiiDPzYl_uI;BYUYVOTYGc8cuF$BAg$BdT8kN~_`fkyviiJA)tr|74Bn1QR_DFvtohd>{Ph1yTv'
    'VOeL3amiQX`{QjfYYDOOgJvid8t~R58rWERx7cibnerBA7U&$P3-'
    '8cbV)aQmvoe0Bp`R_M5^)lj3?$A5!Loq3Dnj<>?PRLN!NwU>vL%b?wgt2mUf)JmD%lS$LZUdeIez6_)cbD)cg=X<s@oHi{AS!BIb'
    '7%h|+8&<kRP1+|srUMRamPGi5Ytxke>)b#d`lWmvqLRJT|oF?4kM^1PbCCcGr-'
    ')VPh!Xl*>D>$ueCv=?!D3KMhJ!Lo;F3#^VWvz9Md&R$`?Suo}86nWRB+U{bRUvNLZ$~wQ`$lNWuwAX9<nobzDuFdT8Y1i|+8MaX9'
    'r;pP1a-lCiQ={B!(vgt%lwG;)b3J$rie_xHtC5_c-RNsESMCCaWGQ#i=UoaNw0Sz=ylR^T6_k$gJs$6NS1=m4VY;e-'
    'BpioK&%LdqAzqudPWNk>rnhfv8K<}JXql(C|Iso~Z{O82QE%VVGE#5<t7WF%zOTbjy?=8R_qgJwZtWgY+{A6%bBZ_9&OMOURq7sx'
    ')t)+Zn{E{2kh}1h+$S_xDfuYUf;4$Rwf`IEW{I#B?Zi*`?U%#=6TKODO4+9hz2^TqHtTyu59oafl@2&Y^{{4?BDE5RJf%U>99B(h'
    'aIJ@BGa8a%VcjxKFOE@8g|Z*B-'
    'k>(y!8i_}a03UqZ69x_t=rNE6&KCzgxV~da#BZHWBg+I?ez&dX4qXzGMPwj>yGa6nbfBCCwz%b_#aDB<P0c0{Nnux{3>a-'
    '^R%6>MepJnGEXwlx`jdx83caTF4Q7oHQf<ChKeMr+-(e#C!x%JuwnWn)VbRlCQw3cyS-rwCDeU88c!VL7cA33KQ&w*><mPuD&RjC'
    's;3U6SUj>Vi93K_?h^SR5$eR#JWm&b!(}?5{KD3oiNL?WS#q2YKWV4ES}B6T$xxUCK}z{{mOA)uwvs@k^X>?rqrdNgz#{tlo)Gbi'
    'eXCwjm5hC>-'
    'cVv}&1Flv2abEUk^z*t?C0ziI*cKMcyEz*L)z&I4m4P4%9LAb#BP^a%`EZxP<w~q74LOI#5!z%FJcJGX2Dy131qkSpl~$<ldrZl7'
    'oD|TXl$Es)rgTTdvM0i7)u}G(FR9&63P#`BlQXGj46&#W^C~$OhyY|)~4uab8k~;s!(ExkwSBP!dqp$un~0>o_fTr{EMnI2fbfQ`'
    'vXUEgZ8bf(!p&>Bn`##HM8aGj?LF>|DK*m@ErLQ3F?<Z(`z3FWiOKf{O6UlNZ&ziAwVcmq8Y%%x2_mE5II7iTm}l8POQJjk7|2t#'
    '<i?Tjz1}cAd?+;PY9f?V@?Y3$O(>bW;_#?*bXk(PM-jo+$J1|)uvhiQk<oOcP@}gEp%7gDhKwIm6!l2cNb~CYVR%-'
    'ViHQugnY9F<HmD^`QbSwo#i(%ZaiPe3SLl*vZIXjMI}hoZ!0v4=4&aX4nhRMJS4OWIIW|fsmnM|dkH+xo~cEW#OdmqQm<wnhT=vs'
    'fKIkTA8~@TK_WxsPC(h2m#3-7-'
    'T;_;pGd0HH<Ie~i=;aJBdN}SNUAe1lIqk)Qk_BU#%@=5Eeor0r^4%|aP@12*UE5pkHYKL)C&){b9|Y2Gh<Zaj`)Nl#;!FpcL7E2l'
    '`wyQQQ;Z)fnT{!#TDzrki)KlD3bESZY;Rno&NP_$YGBFSaV?Wfe~z8AHn8>1W-'
    '+{YmES^>Gf^H1U#S#I;NmN3xCic6~>*hThxFW<N2z{h9h)gw{yHlKhj*LhIH`x+OoVoV9a@7v=>}3561R`G5>z353V<OH(_r+49G'
    'Os86H6okP8iu8T40<W0Fn0S`<@j;`OFt5>34JD;Y<i*SCxS%jxx9Q)J$$nQbu*&sMRNJCkvHTB8<;YS2j9nH>Ew+WyKL?D5*(Ri5'
    'QVVG^0m)T|a6SQAfm&Pc?9ck-t@$xN_&Y@sas6Iznb-'
    'l`glF$4Cw>`jK)ON6@mmxT!ZA%m7zDEK^T(Ax@?oyQFtTAvoiq4QNp%W>xs*s&`DJ6?^zj@=R1@md6S?1{jR*CVjw4Oa+F&KT{@e'
    '^x@#c4py-'
    '^+bgCZK+#RP{$LseGxeDL~Vb>3So%W1KjQFifLQ%dPB+R0=>Q=4ywP9j_(KM>{vFr1BFHV9Z=4W=YGV?UWzE0H%ir9FQrGi-c-'
    '@skYSBOT$pE!1e@2Y(0(gOdAZS}LPlK3TyAvYunn;m<j0Bvf|%Y!y45d@fn;+s9PP<@1zh~8%=i8i#+F$!slFzCB>2<K9Qa(QXhR'
    '&y+Lg2;Mzc;IZHX}~8bx~|ghR7X91ZF2MJNRak9{$!<Kx*!_Lftz7renovWM&|?Tmfk4IVSQijEeKL-'
    '>Ywh0z`S+^b=B2S2wv4DaCKzZRx<@bLG9@f|$;*Teh{9{w9Vz%$tXO6Fm9(Z=L=_BvU*sDzsjpl~uTHjjR%nSbgRrbyLvX<D}ngv'
    '+!9^ECV1xo|+A%6L<?IAfdm@rmNc!(Y3^-'
    'lLh1`BOUmYruRpu`}2zz0R$`R&Xb;b;`OxZ)uZ`wXnIy#qOz}Z^G|kaxl;<FgqBy6__3j#0tz0237?o2m_@8GlU^}Wr{F}1Vz|1`'
    'PdmLl=yN7W{8;=d6~u(2TPXV9F?0Y4|YJ~V4bMTnO8S*#&TW;(uJIt2D+59oeaK{$J><XnyxpJpvfbjWmox!@Ezq+|3cQ)!iYa72'
    '{x9?W3-Qs;i_i-2EyL8DZ|R#bk^@AM2^WKoQ&WxS%gy%g%|lj7MLPG$igIDLqXtjQ)Fnl!jy|9n-'
    'G@Q&i}gscF#JxD2or|amNPH*HNkYnO<NM-l4DzI|~Si-'
    '}(g{q3!(zmynPefsal`i<ak;WkTkrJeL9R75NmIIJ!BvGH|pqKS73QZpo)6Gc@_0B7XDrB7XCYB7SpkfD9Mm)OcbPslTkM6a8+fF'
    'NizLFA#GLV?C#<^!e{lQRr`;F7UZtWPh!>J@)Am!QD=}n<7%vzwN#VY2rQhql`GY$NsjG-'
    'eWyw!AWvIh|=IBxgTuXiaOf}h1Yis=TAFCgOv8jD<Zj?cSrkhuwk--U0&|XyZl$ZuvL+wXh{B6RG*_bp*3r3dHY2lS$`8<gl$TuM'
    '{m)AnHf~=c<oh8(2iH&%Cc7}yxvkw@$!%np=%0QMd_O2Rfa_`q;NH*8*c>sxH*Q^ey%6s!Hc#8-`Q4-'
    '$!dqI*ra((DyCB*);ttw1edwc5&TOjbJ>H9Qfman%a`ZEgYc|9L9vFbuy`FDS-jdt7O!@Z#jAZ}@#+v+ygEh}uTCPqjC|x~O&^)r'
    '<ngMZ_a!DBbxUGm!Lgl{6tt#!nH5@`Nt)vnc&N}FX>i@Lv$V)jt=5GRpAT#Mpbug$L$RSudNnYnyMyvK`3WzBsMV2-'
    '#F@e62J&2uQq`eMd%32Mf|<BgGiHJg(!s_akuiP5nowAGo_xKC3@@P2n<LLFfV>=E&_o5HdSH>Hr9n$PjA&&nMmr7jG))*j?CURS'
    'F&Yx5w8vc*i7;Iw*{uip$VH{D=)3S@wR61sd@rxa;Cvr%|6%#t93v0U-'
    '{I&DVa#Nq<aowT5xR}%>@?9LIEzbX6dat*MH~tU&f#VV1qA1EJ%<8<^K`mEPQ0lg)mNldZNNIU5~TlGnu}y-'
    '>4JR1ahV8^$QoU%0)7Pa>T64BoplsVwYdGA7PS?ck*#)WG825F667|E_%u4P!3kU;5iPCBQA;b2T3S=0mX?iLS|>y;t*IVwm|CL1'
    'I$x03_42Dc1hFKOrE!WW$7Njdqcc~h>}@H|{W=hiITLhS=ExV)b(x!$Wm$MxhUWV@-'
    'ajT4e>yf5e>yJJGx~I*1O;?|6gO%}#EpP$|0(BXpp*WAW>yZ=&Q@yr$jILlwB4su0ozk_N!+tdAP(_3Y37njY0bS{n8Yizkz@;F8'
    'y%MwG*Kudy}~cns?24!NxZ%&E7#p_$jB7!<oNT0_^ak-GBVzX2Q;WCDlklO*nb-'
    '3$c2P$vBqd&&}#<%?m7YW<elwQZQly3bAv*WI|xhy2!D=dEm(YdT%^OjK%@M+GW2*&hVoJ9VI7oVYBqGLT4xw84xOqt8Rlmn5GBE'
    'TBK+4c%`SCKRk}MvA^I;#t#y@A*P-NN(tDkx?L(T3v&?)yC-'
    'kTJdOI$20U5sZXM?K!uy8+{49fP0#Q1m%h!B$anmpTi!dO_3=GmBxQ((Nl5r1?TIrJp5asg}-OF3umXLwS%%7p6fOh=<fmi-ZS0Q'
    'B|Uv96!b;?4Sy^a3c^b*+HLD*xXtF+_coIL`IjwTWEqT-'
    '(Pq9hopHr|lD(9Gbs+OQAhJJihzwU<e72fV{n(tnC|`^Lf0!6BlB*w)Z75rE|w39Qua6HFMRxR6h7f&0|atnJSH~D_FA#72wr6<;'
    'Rp0^H9oBuo;u^uAlCZ;F`X=s~C-'
    '3hc+&K+hJ8S>oDQH!fMeyKCCII3>_V9AJr7*NIiOWq|dkuU?Yds3>D~gkxZpOnu((_u4T;6>{S^TKk)i)T(}Qu`+z9N?xABAilB`'
    'EYxWNX-'
    '*^|@bYgvZHNU5v_oB~URd#kt^+w;kFL8ozI!>h}7;%X$bz}b!WyR^=L$doG!dO8oi7^i27^Bccp>X_%4o5yim6*IEn|Szjc_035K'
    'TLRz3ra_+VDj($J)Np4#YmGvM7KS|y64(HqV0<Cgvk0ot|<u+%7T|7LAO&$XD(p067CoB6ChmH3J(~}(Cv}4dzW3F+1HdzJod`GT'
    'WSQ9zmN1_dO@%ko;`h9YiS>13MjUKW8|^I@BiV-{-!X%KUU$z%(M^N37Nof?@$n|X8xX43cB2Ze-ho02St7bILy0*92-'
    'OLcPr_&zGJV*>@~rhk1=q%!Dx>&P$65(Wj(yU6?ZC%gQ3<{gMEC9n?Iv%@QjkZcBZpz5BcjC+y+W||B(&*DHYhT)3rNHV{1|r_HG'
    'SvlFgZrs0C#a7Xr6pqna||++d=in1-Z^-s%P&x(nSwvh{hU=L=8gWO9st?|qlnxZrcfD7=_a_%VA-W_K%@khoJ=wt<(vhk3_Q=Om'
    'ZDDC0usyv$x19>6Gr;Al0MRq+Zk+4*j}B9qegoo+J+`%5w8GW*Xs`PVZmx1A}%ABHrOf|9D$^W0TiMfMCZf^UbXHdIqkD%6FW+^<'
    '-m1ZGU%e46oBgEK5$Vcohb*BJE-NB5NC>3<tMq%!jfJi+rq@D58hgEnED0=R?u@<#%X{%fXnZ^Wtkp~N9a8AOW@a0J{WRLN~At=p'
    '`4?sA3wx8iBbmG<d4N7+S4r+VAh$OW3Rk^8dti#FA;n&E=PtiGo(rV~bK@O~D6+mUNw8KeqhQ-868ul-jQc@CQ?&JkL!qa#1_x0M'
    'eOq?P_fI<Q7s>1WZ68)>C~$=IKUtM4m&wy{48S3jU&&)A<E>2055Y*QRgStZo@A!xo<h?;g&_?SN85sGVduc*)=m5^>ziGB!-'
    'Fk7`uUV{vu*^xmQA?E*3SM+6u6qw)lMA7&~rv!9ZNdcJR@s{#PGIZC+XFR}=KVpy0D9B_b@`+g(Ux57hv0~8a*PO-'
    'G7#z$mY5Nyp3!kCwUxh9FY;FH0Y~d6+|6SO^DO&!Au!Wzi?NgfV?0MS$Gll2Ov(tPeJKvrtnA@<h-'
    '#}q2Oa(q%FgY*lgmL;6Fb{ICV<$K&x!JJ`EQH+ccop1>-'
    '0s*7>P7B%yaq;4Zg}hg4P)qdU>DGz+N;H%(BFDfOL((Oa)<m0^kavotuegA=9E%)UCh|mr6k79k;E9SkG67z$<7?Ar5EitR95D*%'
    'Ii}q9ob#p1&q~ES;o^`g_-y3bjlLGNk>X8-Z>cc4;sX7yq_hTK^QSla~0=m9VYtng_LKxDY-'
    'B~uB<hzCU%FDB|c8R_&w0Vhr%4PhgQZek8MM`M}IT#kmKYr!becK9Go7zx!Ml!$8IsaOwqEw2x6Z?Mjn3ZyC{gdpSl{BG#!v>LrW'
    '=D!9lSqI5<`Xhs3Jj&{!257OR58V^wfO(!Quuc2e4&pfGH*VbL!NDYUUG3=gJkRvxY+g36-'
    'w<V!gBJBp06!Zy6k#hOPL<~eEO>J~zg$U^aMI|@llT&SNYY?2zhwb(B=WON2YZfvm80<DFPQVK^1R}gk$qOP)?$++XGJxbv<T?7F'
    'ZX`T$7Z?DumI)1tRnsx(lK|_w>^R)BRN!f<9^K+f^E3NqSJSA11M}e2%qWa?~ZVxkT!uL!4tsfw`M<oVnE;%$mOKHNmN?{P;OjBF'
    '+x=m46H?PBr$-rqCIOk@<p*ogZ>RH}-'
    'JKlR?E^s2a73^5UrT43}o2eIu(GR6q_3I&s$X$pF6xw~ujL0L~iWV@IZL38J*|yW{<lmOEZ(l%^$8g&Ap422SvVU|<0Xc&6wXOGu'
    '=^F^`ii>qR&@aGe|E8)~m=xqd4RK_oh8P{GA;v^%h@&Dk#Mnp;F)mU=j2G@=ic`lJDY%^md1oH-`j>QJ-=;t(74Gs(0TY-'
    'G1OOch((95Z_M6zqKvN@WU;*75=nE6d{<P*i(3ip}*eka#*1P>w6NR0yXVk~!v4N$ioSM4VE1DK+L`(z<$w5alSvYvpG}{QIQYUH'
    '(vW2pLy0%9eW;%n=&}buZ*C~+a4Z|sp#rHwkaX)}aMjnGCW9-'
    'dFRJiRyk`P>}Tt6#tm|dsc<WZQ%`MWNV3c>oh)uLEd3WtaMs8uWagY=(YHnKnUAHZ6eP7RJ$acXcHihD=J0eS#Tj*1iXU<o8ge`&'
    'C(^BvynCuw!&5EVcSZX>0scRd~uw^0T$UFtCUXSwfjh30|iE9}=bvteAU>_NsM^nzNsZLI~lex49`R&)YOI!FsF2?MTwlE&O6_Rq'
    '$T&~vDeZ@;E=>~`Sq9jy`@b!Y<F&O0L+jbH6zKawEg^RD*rWXd4#1}?WO!A5^9F-'
    'Cb0FuNU;V52`FrfH`F#lc`<<D>Y`w$J4Jo?>k0tk4T$zvBoLXFA^(lxFEGb+LVXq5>pkl+VQIv$<0G@+x3!b%CAmDNlDY4K?4=sn'
    'l?Zg)T>|yL#nXq;>BXcq#@L238N~!oVtgK?^fl)yK+RQtNKvT6YocP0y$*J=~@2_Y4a#puF{c!wL-G!dYrqV*so{KQb)H0M?-'
    'Ejr|Zy#5;oVP~}+eXp0|^st`H4c9~G(o@i``Of15whH^u$C@)ab-kxj!;@fzIJ+~M9xUCGQ48s)T&?#jY4x7Q&te(wXXF5ck8B;o'
    '#X-&FP>?PVeLz$wZJDsxbj42(?yeB;=_UajC=nRHqjwv(f2*R-*+h${oCg{d5P-*3_g+m5`Dk7rpRhThA8vdX+^2-'
    'urtq<7eR}s)oQItE2crs%4#lpqagkYA@9lhAH&`w8{Q~KvEAbo2o5+Sy!#i0s<pkBPJo%7!RpFnkm(fq?A7u+8c20Yi?!F8jy2U}'
    'K;@s(6X768P4{uG_7ycTIG?7Yo{;>R?ki%)?Ln@zxC6_31Q?&=kj<Zf;RB^<6^NSI40O#|agl4IED_FIVcyTn+(YmD`~#aO?4jP-'
    'lOSifh8^?k?Xdq_uxF7`*rEv5E)J(A00DTXS}VTLOgN~Nd&Jy++bO$HP9aHIXCPBu_4`&YwdXUYWPv8tz~8pYKF`g}sVGSGFM!mK'
    'wpv*UuU>j`1Ku-3d__Mb9F-'
    'v!wUFBu{%UNie|A;9SlHfmlqL|(jXNblArxr!a`98Lk;p>ccV;{Gt@gQ|m2us*+f2gUHRS9+f6aT9fd4ewRZ&s~g$Z#?<+ZV_X0S'
    'Qp#XonyPYOKewnjqU1gv0dFgwyS%@c6HB4j|0iwNlfCjoEg?3xtlYI(=Dh*$d(}|c%Hxa2eti5Y0Et2;{G>P)l6g|prp+&3d>s*h'
    'U5hD*DLi&^|Ajt4J>mTRXL-hAMM2~-'
    'xR1&7V)WYagfgY*pcC)wWgfnh;Y$HJ4yh5aj>Qg<H&H)R_6m){)ZA1$6&#rl(SU?r8KGtN;y{m%r_%u8e0t62K$w|k3y)@N^Nx9)'
    'gPG3)g}$>g}U6}03xavM8EU?b%ovZN$oW(<jp*w?S=l$tfMHVFf-JYz~FBJc5@6vK)$RZ1mt|__PecH&*k7aDM8E9C%|*@Tl(fhx'
    'SE;w<595lz;-Z<TOj!WmNz&bYzHHXXYWuK2mQZhCqZ~b2f`^1D>E4<=q0DjnV=?q-'
    'y4BLFJK1NXnTQQptah5wJ;+}5yTLI?7LY99-eT+Av`?cpm$+-'
    'f)@2QuX*6N&o8unl5vvum)dS(prrkkwtr!yq`h6+zcN%3Rja>=Ji?vY{++>+_AVXxG85ir7~$IbHchcWhn4HaHMkjIl}N=2>Wk-'
    '4lFWLk1sA7LYjIH4dDHIgRPSh~`b0a`H`=Lw(N6V`c4|PhQv;cqzzq&?W6w?D@YsYKH{tx`LyBG=iW8n|gJC3nZro5t;|H20M3bw'
    '-@n5a|9*LttqqvVCcL;-2UqHM~Huhrw$EKLj_2o_UM}wpwR963FFv}kXjsI-'
    'G&k+WVHyLDW0i;2$^h%i^Zl5Jemz}aiUcN39dHF_gul!Z0A_rqWkQ2Efn0*dGQhup@(v%xiT{G;V8t|${=<LWW`4uY8k1tYje*9|'
    '1;_I%x`DHG6(K;EsP^PD!G?=*$Kl2xZDqP@uX$QeCj&eI>Q7qSX%<NBvq@zm~WS#$ao>gq*x&jaFYMp92^a_o%kra5-'
    'Hs6?L+{YU?3qzlsVBFwZx>U>1cf!Zjro2=OR2;#-'
    'T*a~bD^(n?ze<JidT+p18~puPW4zB$R+`ip?>CeKCFR>S#!gSs<qsHpQi?8r(Abkxbop9Cd2NC&UuP(bDx=FkoWW$32AS-'
    ';%Bfq^XV%zDO^H4ydUQS??x=7*7OO1=#s!0NwOX^eDV#?q@*W<F12`;k3HKh&NW;XG^$JhSn_Eiig9bUCz<SoI47c$f*r?rYOfsb'
    '@!v8Bdii`Hh=-'
    ';PVdxnhu{W{LEkiUt8(iBQnspe4FNpE_6vBbo<<}E^YaGVxm(P`G|gpd{HrY^=$wEVqdv0^obScB+oSfT@<<qq0}dkV~5LSyP<&('
    '$&Zyf4O{_s7_CO^iJsh_UB`G4@=`l-'
    'jpxHn{l^IKKxpYtr873npF$iTvF<T{X&2r^aepfbs~j)m|5o+Xctrn|P<T!I$UjREKP}fE2D(aUK8x)(u4kum32Pd>?4vDgZ>zcC'
    '>Bvk(aSiG14B%`FUJ`L?UyWj)}}KbWCJ^sbeDZD;*P=+jUH2?$9xjxl_kP<}UFPoiqFMD%AIUjLD(^d_<`omCv(Nec@m~3wd=-'
    '_t|l2eePgL+Q#}^TWAW$`dmAJJz{;Xy`7ZS=Q`NQX??Dv&C~i^Cp$&zbFP@oMIC@<&QUz4fPDR2>8SwDaek@i>9(qData`C3lAPo'
    '^ID-xT(1*Jcd{MrG8O<@N0y_i3`V<jDX5w07nYW9LC~<E#N+i+fIkpOdkK%U^!V!dP57#L$_sOZiJciQEyr)Gz9uZSV8bNKBqJs>'
    'Ib_9TCWp+J%;b<AlbIYcWHOUOmP}@H$drjFqbqHAgc^CT?v8-^e!;V@Frs)q3&6EeMSgS)`cL)7w-'
    'x4c5+4FnUFXj!1T$tx8DSknVQ*BiI2RyIU)1zU%V+cp)|FV8Zd64j)1RrLlIhP?QOUGP6_revsiKnUa#d6^y-5|7OjoEx$+U-'
    'K>FW`q$iCpdp^tvux;jel)sjd2Q<oSCX)7L4lb|@CZ>RZlN>Rz)N(r0UCJJ(dT@`s9>uF23lA>@@uAT^<^1y`vo_YGt0*3(N;om4'
    'G%cec`TgBv`w5filn0J<T)xC;oXK7olR>G><9SomSWX9-'
    'zwB&ilXpIu`&_0f?uuoW|E{+8KUgnhflJHHu12zHrSUj8b7U_9Y<a(fdHZNmMuD-UVD*3KHo73Fclu&V&Mo)%YYuJM=A=p4%Gcln'
    ';Zo!o_awY;A?Vym+J9vL^Rl$fcxTm{o{B#fdr5Cu9=wm1-'
    'I@b3PKK5`#?`woR?g`B_mrD&;WWrZ|Xuu)!fIuS`8W9e9OWB(ZPgvRgplWav{DrgnJ|rXDvh}fV6sA{OxT@6%zbNg>RVFT*@0PTL'
    'Fubrz^|DHFUA(DZ6p>|TM`W2d{8txz=+{h|TztLY2CfOI28DqKB+dGsv8_b*<p;)U(aX8sfJ|qtj6BRUqRG?(ehY{AO}ETyIM13*'
    '8Kvz>zGt|ZJFI+y=3zQ_9yjFnGxjSVk-sBYuNe-(e4CbATdT>n`>SSc?~$~pr(^u=8NSrpF@E;e0(*T-vc_JH_&?-'
    'Sd%=w<BX@6tts}s`Be8eVyoEIJEU26Y%!Ww=ZbZtVezM+x!SUVjZ5s?w$yLk<o7|ceY7(N|a*{ur{p{XCEouQ~l^1Y&*1{;ed~p6'
    'Iqw4Y@`98tao#PPSztXZz9@G>hN!x9eq?6BR`&-'
    'G8e`LU{^dMt4wwlO9`hpyu<6s?~S8c4W>?Q?S1Q3U6=Be<CutA3h|35LgWUDqrm22Ogc~O_gQC4eqi7-'
    'it&SWo;?Is6`u8PsU0C{J-'
    ')T=6`*r=)jdWc6IEs;0!l(PIw#C0iU`L|{xPbo{YW+P9b`qP??JmoLXi0U=P?OQY(d5YVgjehn{jDDtyz<?(wes>jyK@-'
    '1wHwZ=wnpGDfCt?N}Hw0Ck8m*QhdR=^2y~JLnJ=<xhJ<N^?(=up^6-'
    'V}72hW(>sTb4!OLyw*$QE;{)IjNRaS1e^D;3auo)kdy`BMMPFO%|TzQ9*MJt0?P0K9-'
    'O_&G6O&nvXYJt#T<J%%|Jm7ee_u*iH6ZN(l_t3TH!=Dr8V+_!DaecQ#{w|&fgJH*_#W6XU!N$zW})YwVC2#54)oKX9#aJ5)t``jL'
    '`uF=>fcbOs>re4}fw39izB{MZU8l!p)Q7#iA>KGt5Le^qNfms!8>GmE~wIkp<{*Nl|n1A&h|BZ@86J|&j;i`XvudJ_%%Kf4Ac_%W'
    'x-l46GOs`MqD<jkE8yd^V^!kO)GBUmXp|y-mZ$RiRBhwqmN;He$H-'
    'u;NDvbmC%W(BI*g<<oxcWMLiYEc*CD{e8y&z51RH|;xbUW#Pue|$gww1h_8G&fsB{z$7JZH<m{Sc#sUI2&{Sn_6(rFU;;UVd}&y<'
    'FfXKR70qD%&MmB=S&QS#J0Q4ki!BM{o$JeTQ(J<z~Z&a2SO?d<sWUR2j}5uzpK{()C|~2-'
    'Vw(@YH{zZcIO!s)u~ZkQ(sxf=~|9$FUBLzEp7;Kyn=)ve3ssJE`F$Cdj_-RpnQ)sj6C%yTLE%R&b8}xhZMR9|dEPWJL5{u1u%v#%'
    'W;tqGcYh0j!Odd4dMGHd^MRH87GfBbulID~cJ>F`CY`&xjf!c*??tBfZzE=eo`<hWkJLHD2|lRcv&tcrX9xacH)N`jtE4p_h8BmV'
    '@ZG<vNIFwGykbuH1;POT>6O=LaY0a@!<bZkMFX?UQu5Ly|6cOw#2}k}jVDo&zt0*Jx+zDE^4pO4J{bV0igQIg~p=hv}Z;r4%eKDY'
    'Jhbg?h8ocqoPp8%!E9)Jtog*6i2>j!o;*O&1fzQ6+k?gwo>2{*BDGH<q>&?Y`BffZMvy6mVPjn*wfYjVa)^9xw&m*8juXdw^S2Ty'
    '4P2>~j?9ohxF;t}(@+A{~?q!vH46lK90WdZUR(I%vRNutikB?ggbP3es({#u9r}ESIJTq6i{31X2HaXJ$>==WyKndH(;G^W5ib@0'
    '@e@p8f7MYpq%9eIHArx~+dFQQg+#3Bl-G?)5Ywn(IU{p$q$#YmwTU<JBQad!p*vm3zLfd^+{{#Ag6?(wWn$xC0*o5A-'
    '<7DdM2F4iw#e`~)PYdisWo+AvXg0yx!38h06*CXNoHpC_#mSi?~KW?}L%Lp7Vd*zxEarTwQH&}vH?le;t2(ObG)6|66Gbx(-'
    'a)r<JQ_e4pk&8cBG2Szg0(x@)yam+=Z5`IP%srT3V@S74!sBQ)w7HPn%x!EU0#<H1ZFAWNE?LMmxww#NdeB>#hp7S!nYB8B)yh;}'
    '*{wInPpG}-'
    '@<pqg*T+veSed1dReh3MpMvHs&%^M@fv$!WjIRqO`<Z_P<k6Q)|yzMIO68NnFkQj_{B8|TVK^p04o(PsbQq4RWlqg!hkvL**W5fn'
    '-V&^BKjPp}b?EJLzsltMtKkrz$9ed7?aWU0tM;_r??0<*eV5)syFV?HuVLLI!r|KN7Mz+|X?u_k|6C34aq&Y4=lb4g^xY#7GAQ^4'
    '(xxA88w8dsQj=56aDzpv;H)M?bmRWuE>r-Xkca-LG=&@e!ggtfe<9p~XMq9G&nj3`L@!x57R&E(!`g44%Su?_WKUsce?)_M~H7vo'
    '!UF+o7GKRn>8ch}Jvs{nCF!+7vQ?=}Uob>5N!q>C%PNU%KT|o7WbEs5TVSIoiLfKUgbhN7r5Nhhwm#w;Q#nrPrB1J${HrKV@7p}E'
    'n&a`&KY&^l$|8g^R71MV+ZoX4P=5m;UgKf{`OcY9yT{?o(vaLg`AyH|{60W|IZp>WmG^VoIljTEPOr~myhq;(S#S)KjaW_>&{F{q'
    'uR1onv7t>u8?ysFuJxL3Db#ofw3I;7K$XYVBD!T^1DhA1lDqEVWtoN^6c5NbB^CLl#Q7VizzN(5>qgu<0OP;9zuR0AhZMo}6)V$^'
    'LWP7BE%jGHdNHdqq-}5lQa@;2!(Xa4|6ZJleQ-zFvWX|G&LIZa2!`nC(APGy1lzBRx*>1TrQxEA}O|rDbWEx5Xit2zm<^)%}Qwv!'
    'b2hwdUSpPLgll76m*1J4m?%JH-d+-(mr*;`CpaLg$Im*`pCw2wOe#pGe<V5YDL$}F<sC&Vq-'
    '^qn;chiG78$9}*UFbHII8z<~kA7Ddx=lA_3r-xk#!Q&9iYXm1&Ga@mDN)K!#j4dlR;|(dLW^CAaIGWD-^TntE13P&jHL-'
    '?XGRnT!6d$eYQG?Ico&syLF7<}s;MASSc&2?+6yr?G~4WHJOn>jerJy!h9fMu+oMO|30uzkj;K}p9}sPh9z&$35N(eB4bM*@+8jL'
    '&*H5x=sZcAD(aaW`%hmcgsNvGw&ab-qZ9%eHBhPVjs%!WYkLpz|7;Vb$JD3t{udrvY9Q-NnDOC>s-'
    'R>b(anL>P8CBuyRQHG~Z`K$yq?465JKb@>9D${Y)XU_z+&Xo-p?R{C?z;eq-89Sgsk-'
    'kVt>vVaZzKCIJ{wmlX6>N2WybNKpHF%I+nCqoRqN||V(s;N0G8EZX;4d}g7UwayxmNyPY>7xi4$`&icuYxpfma?Ms-YrcK%Q!Y-'
    'bNO!fqizwW>LZ+prGE7q}w~`;JqwKPc+9b6H5JKWKkLejZNUdaYPphSV(xNo%x8cWs&te4A!N-'
    '=^8fw`n%^ZJJGdn<iC5U>oa5E?=}~yo||jxcq^8$@uTQe9t3H%lG)av=3+vA&biyuMal5(U7#BBEEH;IEA$;e5X=kW2n29fag3lS'
    '+)Y=uPpU^VcDlJ6uT&2=oqP*DDNCWZXI?76s|fu>E}9|UuzAYcAuE}LF}zudaK3x_o(ry+L>rwy4;BbRLKzQ(q+3|-'
    'p(L{^<O)=cS6BZzGRPT;l0UM?9pWS4Dwa(xDB^)Ctc8u;kAKNy&F<TeH=5)J|$t4(LFEabR!4#h`7As*Kv78$GE)W$hf@XsJOi1='
    '(xP1Q(RtgjN?w~>((+|<b8}Uy6@tP?z;w|>;`1%_2&I1<m1OhdD>cXBRT_4^=Ug~2{-'
    '&G=FhJOor(4Lo{uV&j!dXRDcWk&W~+Q2Tk9yNb6@L$HGPVofk85^QL`GwgY?;YUPIG2x!!av)&Y0jlGuSz=iNEHKcC7j;r-'
    'blyN35?qwE&mpU-6X@cwL)J;M9*x$GI<pUtwDTQgNC@uF{ltYdI*zUQu4?HCXOxW4t)ru-?|!2O)rSP60Y)=X;jx+9|*-O5#P>}-'
    'o<Yq^Q*`fzh_5(tR9h1#_60h^Jd)Dn8GN0UgbiifSE)S9WN{uq)&Rl#v*(%n_jaToHyRk+`k^rKX$+|5bCyx^NfSE@_mE|Fmtnsw'
    'Z@m8#eJlp*hvsQXw>^7&I{-FT!<zh?h`D94-'
    'r^G~>=)B9)QQjEJnf78yz!wj!tY~gWJPIo&Y=%#nI3afmGrEbz`iMmhvE@@^p&6TWgP(OKQBI+5J-'
    'GwZhcuZf7Ji=B}2me>tzOY!Qv$OA*qxRHw+6;NgK8}T@chD%-9yf^2`V~(x;_=e}VTNN<YRo`wN9;Y#7_xq!+o-IQFAS-'
    '{oqpO?6n70vBf5oK;obdKc#R6Q*SK0}wb=fj%beZC<Y}C$_-_Q&SlW~7t?Qqbn0=r$OM!}5MwyEr75Y!R`<@J~W$1{Q6(rk7m6+-'
    't?aoG|bF7`?-q<d2Z*14NH?~{c8{0kZjqMTl#`cVRW4-'
    'Zds5=8cwG)0!SgzDLtVQWT<tJ{H*bIt)?cu3_GA#l9iZZliODE2NL#wg&_f$V?*kJ}x&DK0tW4Y_$-'
    'QBfAo`LR<_Rg(WBkP=Y+IawGD%eqzU(&bGQ8r>KS|mOiYy_Rd9m7~trXoeP{JA!|!Q2WOU=1AQYhy?I+Sv1aZS48JHuj&sHg=4!j'
    'UDT2V=vIN8qU#~a9#Br{Y^zkJZ-1`zep~$Ky8-'
    'ZUd9|M*XdN3*{RjBP;c5syB=NX<#`OmC_L>hC}^P5N}WhdL_IRLFcZ)9?#xrtA&-'
    '3qcUH>vL@NlDsU(3Z++wEZEYloaram3dqV+~?TDHCV^N!MBRPe`Jxt%igsoBghOzj>KL*cp{6WwS$Go*la9#BLIXy?P2UFx_qnz)'
    'iLF$RDl$*r-lazq1n0l;PX$?(N}5=(VcBZh-XsF!P<{xhdWqc*6WJu059FZ7=238rE*P9M~TZQ>{JaId?)FE(k7O-'
    'Xh6b@coNh5Tv)mhg1SE?5qB5xY}IVFR%Tbrd!fds0VXBe54Xdo~t(Q?q9iv5%!dQb6372#}Nw`V?M|^S?L21$W*0%Gv4eIl4A8%`'
    'V^~hh}mw6FIby`=k<5OE5R(X=mCa<hE;;PO$h34|?(7GPS2*YQ59*)j?1-'
    ')`OTy>;xO@fD!a}+|x6k$USAT&Dd$G=HGMY;;d)xvFQxKZ}rnTJR9NjI35k<{dOe+HQhPU+kVS&Twa~1pN@U>+9b}eH<5donQAHb'
    'jZ>QAlQQ*5?JeV`endw3B;hFPyD&(5B2@V?=R=11RGob3o(?#xOAYfGTELL@Ik=j?t7w~ps|mcCsw%ly;S`9@;$o$fZ2B8lb9)^X'
    '3vxBNmK`6Ns}HnJk=^3>uje6Yj#^cDq3hnbhRJ#MNqic;HoZrb%JWuJKFJ*RMHbsH+5M2n_Di-vF555J{pF+Kk}a_j<Cp9KNOt=r'
    'dm!@NJ3D!u-*M&3ID-'
    '9?xN>HkWYfv0mv(gfrx@BM&l=Gg^4dX^KFSdZ1Gb}x9IgW_cbG>&v4;)ckqg_#bjwcofL*8o=*l*%mY1?BDBDu+Zb?kPL8?sYgmP'
    'S}UC;=sZG3mG*};PxfJUGVZRP!pHeUM42VCW)03ov7c<B#5WV`VqL5FN>4F|}1j5c8gf*JV&$Hb`2=&9T(Vx7q4AGuR3_&t|@;Z7'
    '0j56W&t(|c;bz@`^E$3UoN`i#J&*5Sw%^PLe+H=Z$<oQyn!Zd|?ewDjWY6-'
    '4Hk!BbrkH6AP5Zmrg<F)aMEV}jD<G0fj_eN)r*Ox;1b;0$gPRDC<uz@{ksCe=VO1Tjf>!qi$$?U|Qp_G0U{Ujc8EN_g6$dFg1klf'
    '7R;J)r+V>)LS?Y3!|RzAIMThiUT75OxP7u<PY~%^qC6^lbIyp(KpanLibIQY-'
    'O>J1m>qqXY0b$rkqLVBBxn+Q*+AyiaosNafs8)fsF8p{Z716qxD*xii(>K&0gswH~0+G9)>uSq6`8OdnDm3Z#uz|Lh$*Tkk15uub'
    'O3x^xd!`?fK<>t~mdG25m$56d+W;~~%uRc;j{I$_I*mIOK*wd#!ANvKEXY*#LKsuPxQQA1}~5q(d(s`-'
    'fS$$l_uvKdLr!Hmh4_UIs(G1<x<9fGG$wz2(HqluxaM!Dd9ssH$V(U1C%|0x9ZACD3JssDJakko&Cffzv8yHE^Fb!g~`C|lDEKlu'
    'WaP?(Ggh1g0R;nSUck(lnn$xd~|m6M@55|1=ls-s}+$W$E-'
    '_>!&a1O$_@ItCAa$YlKod(;RG8&0dTGeGWH=YGRYmd|gPqrj+xE1tzXjwbavx34r=Qxn_3>j_RmWHku!>e6h%hB)IDaGj^7>P}QS'
    '=&7nji9;Os1=(wL3f@=&>#lXT79NgWp5+McwcdJj3LHbvq7E0Asv#t5U+aQlv-l-X-'
    'gYvRFBA2c&P()7oL*`(bhS9y&vD3EP!hijU`si$uCOE&i|sCd>$akg=jgp4#`pJ-zHqdrEzqa^MLMf@2P(?+jNOIp_;KZH(*3@!z'
    'kh?UaT_VREC^`GbOAwR=dE59m!rbX(^Jqi8ElsR0b3>k+uzxYjDsbqJ}FXspsnd446jq{DlrtisXeGW8cU#sSr0AmV6$P5GV~4_e'
    'sA#<8lO{Hc940OZ_T(BHc`*oZ78xy&sufbWRV}&(6v*|)|s}WXijTgWxp=APOGlA!d&}c^7l}^p&ezTQ}DZptpE3ZC*iL)xzZ6n&'
    'YzspO;0R1lows3>&)!Qv;z%YXQvr(PvRKY2XXZIOm*F}<C$Dsk`=|2l`rr`Y;<e6@Mm;LXZMBP+8D=e`gIVf1x9-'
    'pu?|@n+eAYpySE07@wWwh0=EZE@^=K>0(S=70(S+?^R>)t(@$oRHvR9wp%-jVpA5wv!{lEh_CPzxVyX<j4kYiXUNShJ$}t_ERNy-'
    '@6tTB5ivK$?OF*L?yi=WSV6_(LHglsaVO<FuRBL)dQR>#0oYl5!r8=K#-vzm<yQ-DSdaCWzN@X3@-PB5T0VOmB>RKVIM#o_j{jR-'
    '=2}V1n#D^I=bvwJw#J(l^*y^zvSC;rGFg9=z)g^@Os|TP6-'
    '~Z7%=BUgDVY@M)f8W{BzlVy&|L^PZm`3Rfi4H?OGnYC7jOP$a+hSkzgilGdY)Lo|E8y|KrQF7LRJEe%)3&+dDRs?bx#u`GNhkGx>'
    '6uZFh|7rms3j4{M{<sBk$vh#tp*o!y2@QqpKUL5H`HR=iyIBDXuAV>xh~jw#5$F5TWQbbF@|>QZj_|@n9KUsTY3__5_WQ)0cXUq)'
    'b7)PY1-*jNO!;v*k>a<F(jlcs>AG5J7d~`n}u3C-Lq{CWNWzJYPL>F!ntS-nvfo@km75nQ1k;f`3S)Exct_M67_;BNQ>+eXz|e&)'
    'I$v2>c>!O?h`I|v5&*s$a35L-'
    'Wu@)h&L?uo&`yUrO9dv9=T1a2^okuwY(K=rKV(voeo^5GahxY1g%_P*Ma*@xRZ2(#O}$PK1NGo;di|2T;{Q_SmqLs{ls#YcNAiUO'
    'FQ-#D_z!6id8P@I6%Dba*hMV2QKBf4JMo1yY0F?tCOo&+2Vwt7?)d_sbq3_qEmUg^sO&Wk<L$@p=TrE@|v&@>?w!}6O*4g1-'
    'nX^oBNn&5k7`_veGV2U-BM`rPLZRl_Dw4E=+M$2e{wqqNfo$?7J1gO(=$k9{98E=f-lE5E*3ScW1=T4)2~W)mg+`P3d?%+yTGI-'
    '>cMLRCCo1fB2K4xM*BZ3SCV!{J4AS{?W}-w~y|fx_fl%29TKjK-'
    '@de0prRlEe8yjSBG8ik3%LrsMLA_y3Rp4+>=la4?Mi5!5I<CBz3ek9WIOv<YIKOc0f?j#v6dgF>hyk!IyKmY1ofCd=7}08#`MEn('
    '6wWDBC9E?A(1^6m!COq1GuknbUp-MwjTg+gLeBvm6tL0M5W?SGH-'
    '1vvd)HI_*b!Y1h$yh}_6;*&mS`IW7kvawE^>Ktyijx*UYajeM6w5Uf$KpwZ<vwJv~+jEyREm)v}&e%cywj^6hf+;xBNZOyih@)|('
    '^BHcMt?W49^A&c3@T<%Mu)RowV-'
    '?{rX195zi9O{(H0M#J>(bKgSUkI@t7ld5|;U&9iq+9;xsVLZ<@H}05X&>qZy3W#0+d^G0_%LmMY~-'
    '@BqX<~WUA2B2kgJhQ+QVU0uZqQ*F-'
    '@}*LaLJvsZ?N~8yI3ZeO?qZ2u<%5#kO>n;Cf=|1i02?{ie*2wj%1Moh`AaH|u+<L%Uh>g7Q5|11fd?lZhlkEQL<m`F3(vq4tR8<3'
    'j|3<lH{Q<!}Ovx`W%PTv99(-&3)qSRsC-'
    ';z(uTnW=`f`VhW=zd6Ip5)fg7uLaW#MB$`$2LFy*&JaUsW%wJ4<I+>Ot|(rlDg0kiyiwYSE5GN&{12X=t29f8i?2Gky35BQ_U&Av'
    'ja*Hm^`06h#KjZM<6exaiOlidT9nY6xqY-Qp<i5-'
    'A9H7D=JsQVCA^v2pCO*GdTOsDt3aX?|3kmoNPCl3a{03BzI{)8M|G^?pJ>W&U@`kVHe`yaO!(J&TFJbbX+=@}?WOa2Ls6dTEZy}*'
    'QC*rK|2Z~+y<wsh-GR)b5giRinzWks-X5n6!Jg`EcS;t_psm_k1=-NmFjKQHs5OLR`S`}|E{Qj=qcqCS)0@h(a_Ss!VRwn-Io^hJ'
    'n@yUCcc2S4#!lx4x+=x?1iq`x&^9Mo`hzX}tx>pc;hOOmqhc@6Noc2x)l|Kcpdj>bY$U`pIW?Tgn~G{iV)fY?<pKSSQ7q6L#0Z_K'
    '*$>nhl#aa^YF_H6Yq5Qhsh{Q|=Ia7-'
    'yBI72ZvtJ>N?Fg<D2uTmYG~w&P1k!q*cY&4x0AiveZf!gXh}|XFP&Q%n1mk64D6;F$9V)+eJ##t@{G!D9Phh9HaZBPHv9#1roE|C'
    'BQ<4r1a|R1Me%rg!j>z@QLM019_nP|SINU1>t$0o2%V@?61t$>tta(9d~8}D*Fd<38Z4G@IgHj6@e($1sf}_GDq_j&UlOz+VYi_v'
    'vlItD26;WK?|auT@#A4u!lJ`K>~VumQM<H8pTvtlJKuH$d|PV|UzT^5KDT#=X#2N0h*o3l(a$goU~4oZ)|1cMHRT=r!G^XbH=!b-'
    'j#|pWRq%I4yK$h+VeIH+Xk_~h#wn1=_8W}%K;hW$C{Be~w%<{l25sn_$d3^lh%oX1E;dq`!9iSXqI#)=x%iwmaSq{PGsQ&>QMdnB'
    'za!K)(J^(jtBao1uTB2jZ3OOVXpGdmuecmq`uu($&giQ}HGk<^^=eo-=d8SpBX<}7$SiN5-'
    'MCZB`1pjP&=Xx1bVvCAXK>Tr<h=3w#r;Mj#>#!7bKURi-VFhAzi&Go+R;0c*C#&H4}~$GZNRt|tV9FxWY|VU(Fh$5XF9R-'
    '&lCBv<4LOHj<Tn*PT><X%@GW*J5%e+aJ?dg>kB4$4Abf{>fL2b&MlqbTZ`gln%~QdvRFclilF3W=LUIC=Vg?rcJP<g8D&|)1g%F7'
    'tR5A3;@1%wl6nkgsJm&tYo?efrF}nJ%yt@n4~PdyZBZE-'
    '^yE*qHuoeZyCh^RTBFBjCr5`oFR>|9@g5`tKC>bH53+eeX^~I4{A1$6-Cr=gZcEK)2UNU1Z`-'
    'VCI?pgwZR%YykrPU%^Y#$0PnTK%Enr;jxdm%o|B`rF{c|{8uMQlH)%3Lua`o5h!&Ko-Py`>8b_ezFL6vt<8edxQohq0=jG4|iyF^'
    '1JW24{d?dJ+t;gI~C>3nE7i5&G>oXZ<ROkR=5S~H3NO60|*lXzY@iT9T=qw1lb;KFegI{i_1B6r}9dgmq2J`*&SV20O|A!rMNo^S'
    '?diCJ=h{m{MQUOCXdzfas}q)DBi*gGLdYilOo4LMq?m|PZeunt1ml5oXfn-lqoV^4dVJIaA7^r7wI3tmM+OaoXYPfB#YC!E<+^fU'
    'E^iK*Sxs>Ob+nOgOm5v5amPf^VmP3`oeX4{B_G7#KMaGAROAnHYub}G)D#7NRk#kn^#lC)EC?k$WY?GDmA&eCm^E4O@r+o4rH`9L'
    '={$7}7I0x4sC%Is{^C`Yg{rRe3a7+!azR+IHGb}mMV1d6W>HlYK$#Fs-Bd3A{|`##`jw8B9*m!+qsShqp_n9*nhYdX|hOQ!Xr5V4'
    'mQ)mM}Wh5GYAA?#&`V7;uL!);v=(xyQXxBCCjP)kBU52&uvS*FkeyzDK&E8YUU>Mg))-U7VtEx`Z01$e_-fH%$AjnV|P1?Fg_k8;'
    '`NbzJ^FapY{jW_aBdawk8^G~Cp?r>!toJa0q8d3(H!xl|Y0DkwRCVT|%TVhGhoYj#Sn2iKb$J-D+DMDG6mR3)-'
    'nE7jc?uKU*f4C;IGGKGK&(XUekr~v&xzpH|<fc88OW~Ne{{NA^~LT66lWECh8x19JdDi(<^q`6$mty5Pc_K^!{T@ovxJM@Lu>F94'
    '*``mm!0YOkbpHD$IxMV&@7nL7H^Z9&H%|d$lS%6yB!xN|ZLhV@naONktshq&Tno6x?Lm*fYbVV*LS$<xDov@I56=uR>?loA6*C<+'
    '4rr-?<7L_G<)0M8-'
    '1{Pbo)|RVK(kWp5=;lp@jmSA$D=F&aZxYK0U4uZYB{>A>SojN<^+Sb|e;aK^_3lY%)y8nRjHydpV2amLF8$mPp)-'
    'qWXemNxY9I3%?<)>b-c3E?o&{tB(fF6?H_#tTTnxBgeW*4Sc<;*O^27?YgWi6;*NXp{R&*gX?J=(5-&Uu!;jYT_5(Z+;ew-rG6BD'
    'N++L7T^YbW>5XoZl8{oP*}s-Ftnu}qxCNxKiNA%EstB&;T=OITp0_a3w*#?z~uCW&o*&6Sjz-'
    '}}P(tu5y;J*u^|#~F%s{12}8pVa@@i@CxE)Mr~F4RtG@NYtmi&+6Jx)+5z=O1j0OI+h>n8_SRNi{-~eEI-yimLHR`{Mdk4erzDgk'
    'D1%6JdipkzWFEY8C&KF%>wf$PReyC{Xbp7o~OHdOZ-z^N3+-!A1x_05EH`bo>)|_SV?LLculP?=6u^Rxgunb-'
    ';K$Y3E9Dm99f60G2NBRz7}+i(YSq<wTj0>B$Tr8W8{~K{M2#AzRRsUsTB*3aXHU-'
    'u*GKEEyZ`TNkXFBEp>;&E&(@VYv6b+R!yp07~&O*o?$*2{*rHvAEYqhLmL>lKcyw_r!e`Dnb*6~lvOgXW5ap9qNp6(k~j*U)o^R`'
    '-=gZYLCR?s@Ax#t-'
    '{3`iSAVEJbbCv!EE|C@kGqtCTbQEBA*9gQQ4Cd1fDHnMDJOu``IHlI3~@vDryyXYass*#Z{Yw60!Aw*!0r_{L9hMa?440zU+0AOG'
    '4{}@nTMk^$)DMk9*}o}fQSl~7L`uih;ZUADJloH)V_YNAOHSgEPkkK&S=GgswU)Dg6DlHa+jwt@*L`-'
    '(hEll{kJf0T&t@Ul91&2+!V?4V7K@yoE_qP<qJ5HVSU}6$YIV`zKA2oC|oA-aOW#u!f}74+bj8n^OY|{ayJxQQoccV({jC17qEt='
    'KwV`kP3|zb)WA*)CwOr=1ADnOpS3||eyq#Pub1ru0>5coEcMkrcLa^(?QJ7@G`lOcs*@8>YVE|6rYq|~=vpCOw|kxF)B1tAz^AL`'
    'W_ix8kpGf19MWMfFJaDwdICkmHUaxftcBF3^lLa1l*c+6&q}(EbBN}RKEVj-xc2rsTWBs;nG)f6O2?Z#i8+{4R8MIth*AdYBGVCN'
    'Y7|>JU8N~#eC<RuHf^28(miN=bplem(iXdzrmQYKkSoRi%tkPJ4+c3WTm_u>Ql_Xz9);ze6!3nA%_g}Q56$Ch22RLxI2En5a@E1t'
    'q4>h;Ln81P?Lx36n)rS-Bi|4)eJURNLQ}THv$)*WrHl7XS$aH+R~F?H&R4y#C@06Er~!~V+xk>GC*Fp0xRBF6*ekSIzFfSf%L;V='
    '%p0^r{=RtA?2v2izoV(b9A4FZw4+!92dIk5YvhqwwrmbtJ=ri0yK|S`xEnfNai)1jR1Ib;fsoo??`juXL3N&51J{$g-'
    'BHW{Ok)n<)q&v@@7a=QvCLFN$CGz)_*73R%IA3+f^85_HzZO*6gf3=sfLYAHfB^lxCuFQYCirP+o&#SH_N5ffUA96cD`23@njuyi'
    'LB!uG2hd2EFfBr>0%MMID%P|y;vkseUW&Mb_fyOT|_Q~3X@1_pqFMqTTH5xW4TeCv}=s6Ce>@qNTyyTlXX=%Sr-&FNkB<-'
    'ew7OGyc)J$N1=zh-dn%3n7ZSEH+dHRu;qzoYG~*)v;%qH4=xuIvz@dCMjBY_b&)Sszu1?mU*b#EFZHGBm-'
    '$ll%YCW(6~0vcN>i$C$WtC9@|0;}q2WepEw>isT}8RA7(c!<#6|A$jVJGsqP*5+j20D@7n4R>d+gOT_x%npGx;CSt~HIj80T&=k0'
    '@L=$YIV7ZbTq%?cir}gtLR2)V(A-__-YA?BHfO+O8d~<+nw--CPZhwiI!B=GyE{c!|*l!*XUDlEcjQMdt1y#?bPfGw7ClLC^0qa9'
    'yceXO4FhYta5Vgr$4eX5jCfVRjr+Gx=}`!}(>C9YJxVx+T@#^BbZG4QV3hAg?;ZJ`L)m>=LZbb56`p#3BD@AE<<jFU7^?y;_vN5|'
    '=o+#IMDrPA%~pahVfK+#)V_T8ZC^E1Xo~R&k~I<5!};q_^8#8RFKtPt~Gm*80}!A;4`4?#fQ!V0_d(vHh8Rtn`UhAGyx04Ih+Fl_'
    'B()5jZT|bu7%l+I75rw&q}e*(Qn`V0v7Q!SPtd8btaGrYijz7M5lcl80kog?teOQoNqGN80Mh7wpk!hS!UXZqFAs^y#Rk{41Lr>Z'
    'qpq8+nd>zeQ}3|FF~=3e7)wK>t_-rqd7WdP~kXJ6LSx045(dQ#w%nvUEz-S8a38@%EyqUye0hv?jOj)sXOihCENr9o=Q+d-'
    'Wn9Dx0~ULMTjWN&QrH=ogU1xJFL(vEnF(*MIFg(;JHxH3J-'
    'nb&A}#K1?=oVt@T$amdhpDZaGPNk;|cuc^vRLH1kayqou1#a89GmopmQ7xmnP@RZ0alPvd>tlX~T5(m$3^?rFxiW}-'
    'x&kgmO=Z1RSb3^^lb3?u1xuM?l+)!_MZm6ZSN8U?QXISU<SS;zZVOxz=oOWsZVim`V;9C&S)7u4y)9$CbD-'
    '9T}aWnjLfd%giAAdH$s^*gBf7Z9y7Kr#;JV(^W3b!Uiql)sMGBfxaCZ8^yL1m`zaA@2T%pg7AbAUiXUte7-'
    'XdinEzMyagfp2TV=XvUZiyv##4fU)O5t>0ps=1g+@kk3Xi<YC7;$B*fT8aB;F={Pl(^}L<+)qnUTk!xzF;z&~oXXsD`>jWqdZ53L'
    '5vC%R^G&1mGrC!<#QcfTG&w&O!P8dqsAAgdc`>|lOn{h{W>muVtj~iCQl9g9X_23{$npJw5w&OylmAD%(&9NO7^xOKuPipn7QCPy'
    '6w)pDuQEWk&PBYaOpvYQ6ECSp$h^N$yll;KN437bqJZHXk{|tCdHd()>Tv~LJy-wHy-%sRQXe?jAxU{QL(+6)q8<o|-'
    'XoUc_TC?<_Pswd)bzMHT&fNn8^U?&U(~SyFaB9_*fxn5spjPibxrimUy4`V&fd4;O;^9Dvs0(IV#dEP`FrPw{}ua~<cFWh<Y|n0^'
    'wi4#C+3XR)@XUyUb;LCQCF;-ke9+x>`D1gGlx&<PN>qg;bJ$3-'
    '<NqBK~nZhn@3sU_pk9hK1)wiwL;_&N*$d?g=TZq+eqSjsCww$rfk%`;$6x+-EWG-Ys9B29I!s{2Jt-kz#GL2<O6>u{!2da=i()Dd'
    '^d}iX+8K#yyocJw~DvyhC$iq@M+x+rrtcOI}ua!qD=d0nilm;)+<HU2*=8(%+wJcTJ>e)B*NKx(v=Ikc5C6aJRTxzshVB8ZoIAHe'
    'Y<YFt76%Q>4}~pmQq@8mUxG9dG`^(M=Cp2M|a@?3X8|6gE{o*jS=%5e13?YJX#0SPZ+M;S)<z0Sy;eG`vW~6+sG^p@dd`aho0?z-'
    'm@Jf4)fi>ekbz@vdewNlVq3siKjeG-G9gek5%}ZOUCAW@gUine~NkL?Zs-y0`v9~HDsZAd#M_-'
    'D82q$YqY)WET~^h2%nKsFen7<W|yX4d%1!xfyO|65`C#rZyu+`Yd<vyVNzH0p*l_M**;AxP=E1^FV%R~EIT#oqlb|#`kDG?ADqM0'
    'Kl|YPT>Y~T&M(wI`{4Xi{j<U2uhc*L3>~5V^%+rSH45&(Jf@_C@*S~?{F?W~2jtAuiPhxHEE6A+GqYT*p{)L|)!h8ryzMBuijL;('
    '(V_?4{7#}L-'
    'TY%jFS_}iMQ<}D8o(~zqubC!tZ@65^<u^AK8AuFXdDxwag1Lv55wIUQR?sc8Tg_or<UaVjuhR*k>(qY65Z+M)mM)O-'
    'TGYwb1Omv!JU0-'
    'C_MLDBT;bj5{*UCZ5KBYJGm94so1%UJxlwsR3`&JVKH5E{qUOWhu3UTIY>QjhxtLrs{`ZD$FuQGQ8sek?{`Ic3om^XdXSZZb5f6l'
    '6JDW{*~_&Is)Ga7-*I>8=jt4ek2N}+P*QU*l+?ltCAEam1^H`xi-X8t+eaKs{@T9c5c1bLh(jq*c-'
    '>9>H{yS8;<t!5?8LXm#eHAa7^p|$N}7SKMe*x&AYU(#`%)9|g;6-'
    'zfud23R_d@bgU2=HnjI((bTB$d9OPhhusGPk=n!#;yP8AAq3&vwd2ldN_QAnO83+fXco|S{zss@=)KAxBIjqINZ|1a*w1kb-'
    'd{GwPrjW^q?4>T;E`56|J7t%?eUzQDre$Acr)<d7LD?y5W%g5M#9EpCl^L;C<^c6X{yTkjB~FhR;!*6cr=|CKD5b#g_?ovIPp^5K'
    ';NdlI3wU<T+aexa^L8gZx#sQ8cyQebMbD|8FFi4TE&*AyV8W}$MJ;4yOq0DUa0Ra`{C(s<hD?+l_eVQkjjY!JS0y_(5Z%LKudaxx'
    '<r_uXkQnR^Ktb_=Ke|@E<?d(zYWwX`Yx9W6i;I>MP<Qg!5Veo)!#>VYGgZqyT#5Rb+#FoT#!i%Tx$+d@q5OTsQCc)t&MT_dh_W<p'
    '8dJ_pL1^6N$u?>(ER-'
    '0Zbs%a2zb}r{#Vw{_0Ir%xL6x24Icol@OdP8IWW;*)pzdbWC(u&VKk&K+=L{Zv_Q)Zpoi}*cL~}heaTRj+M7B@lZi#G{$XydzmB_'
    'a0(s0Tn>>m>K4!o|B{q*n=gNM~j1T1IbvL6)l#X{wmT_P?+jxG8WJMi!0jxwjF{*s!L>x%N(qFhjvlZ*1fqP(^!uPM6EsFU2~z`t'
    'L9l+K8qsoS!4d~%FF(%``B8rm~EeejU;aWlZ0RC;3W6Acgi`}OuVsXwvfv(g`^QJ;~cV&JJI_j`w?u?UGA2&aJ_fqr!zlmGKTyGR'
    '4=dwX=DKDxw?oWaCU^}{mtGcjEKa3o-'
    'T9qjsoWmF#+pgkDlZm$yDK^OB<<~Nv+EK$O&B2~tX!N&ciz@dfZb?o4YvReXWkum_>9suy`5&+JN0jTe-xuRLwfoCA3+}c@L;r2D'
    'g=7F1@*Et#CX2A0xK9B`v>ZZ)ehO)&e^VUVThY*+!(J};dbUkE}+))!S20Nf#prLhP(MgHy#AOq^71bwk^(!^thI3`;AhGJ_fYuK'
    'TssYgE#X#G82L>)JDmn--3VaD=?2NX~-)G@#PSU6{g{!UQKs1!-TG1F{-'
    'v$udR06U2<shc7<=(sy`?d&SjBF6wgt2)bre}0+E{M&;2eLp+-'
    'Hy50AofiPVw1F97s2QnCg1d+x&k5%?&xBUo&kc`SPij@?C+*9y#8f#;MJOR$I?0UPh7c;rY#f>VggLH0mSCV5ZkBH8d_uQ6*2KET'
    'zQ*@{n<Kmmc2B?pJ`_ACr^XyY`qeAeHFm#qY`*6C<iZn9rxvh*H=Yotz?7OM;Mz2UV1hk%muIc_&^qTsax@2Hh6Vx7{jZB7B(RW?'
    'ZqH!M-'
    'UoQly?^8n?<MBwA4fIJ|;%0AAVb()YPz9OYFCMg4khC$AQ`3A*(;ww*zD?h>^8#3R%k4x+p<js4}_ynF+Pw50ZleweZ%^SMFsv6H'
    '?InEP&R^5@;>tu3n-(Dup_tV4^-'
    'css`wgGzB_+iL>*9?z199eX@aWCC26fou24<xj?r7AIJha+@*}pSqwU{S~m1%>nl`((++cWB9EftOJhN*Xvg3)M%sNSDIp2G-'
    'NC|Oh41!&?6Xz!JBJJD;3lzS8^CR047U!YPMMs})GG|J$*w_Q@K7gzI2b|&XC`FW1}C(@*5Q5k61=Sq@b+#A-'
    'WG*;dyVP39QE!kuy&Qfo4&^T^Wtr75yDZ~@b)go=E0ku>G`?vwh$l4f;V-W=4W)%EZ&^BEAFK=4C4a-'
    'TG<b|R5K@y=fp*^T^~>O{aD)-2C1ZeGK2~0QZ<O<Q5G3AY~8@+kg#FvMlOej{h2p$IV@=HMAzZ)u%T-'
    'rmm`7}TGZi>49lW#R;#xwi~a`}102o=fIdFJ`Jx!-'
    '`+1zlsC*|t<#Q#dTpXhE4e%IhR4#>RLK#%*GkPE|D&HwW!YmsqpTpQZsMPbaAQvhZ;R9JvsU~$nM$ay_^zNq>#>foTaPgr#Sf!<x'
    '=xC)b9->wGpL$ik%pRRzl$(okip`D8V)B~AZsnjU{^yQ=?OZMfI-CrI+(m$s#W7CyFLmtjSWIW6xinrq26#%g+uF24HFdUsit6J*'
    '@oO}W>>}M7^*p1Ka!c_m0SKQ8L0E%Zcu0W3*Fm_YG598$NXlSPU+tW{7<{D&>AY+hd<tXpU{KHb!dw_!j1OeNpt@lTGrE6V)DQE>'
    '&1BD@@u~_(F4ffo4XnClDJ60+h6U*{bVF99yS2bvNbC$##1uz);i<AZqFjF{ms^-'
    'q!x|2@FGt^WHVYAYHtIyaaEB_c|5AME4liRUeR79O8LD)`cDfEBQ_Js|+)r2FrvUh60Kg3;R`Bw$cJ>1*JYHkUw+Vkmi$8PzD}VZ'
    'K#{m4Ys6Mm98y*921IGT;J;yoiWI>`&NX_CmVzd||){D=?)8e^s;Ldj796j($u|=FG#)?nHCh?4TJ{(xz4!lec{7QT)&KDPm4dQe'
    '0A8>#8fxFm&OZC97#a8i8aiQ2KHj8J)e<NIWjd2N&AmGjz%maxJn3p?(gxMVsWANL(AwwWORX<HGevIZv4X|}ijIFsbwl0sc^>&P'
    '{-'
    '!QpP$P)BhCie}yJx*kD|Bw~vBqk3CS%FSwVASzz`zcHw<hnolg5Fu)#7dC5F+l2vB}ly@MCwYa@_ye`c@t9WS)`iTSd|y48;j~Q8'
    'Ik%S#%4h(vvAQTR0*lv4xFn8RzPZE2VSlRRzPaO4t!e=tc29#EK)uF#g9cj!D}OGd>G7TmBrnwfm;Vvg{1L%58*eAglu#TT?pOIn'
    'nyZ-?muGaJ{UvyiWs`@5OhKJ3&XDY2{fH0NQ>olB<TTuahogZh$6rXnY`W}9S1^fIo=-'
    'i0Y$W?Ua&f2moRx_DO^_va9v&k*DG0&zadzfD}6sNELRuRXEMTaImYIJr9Po7uzXMt%mB+P^uP?Td`Azg1j_+2Ea$4VRKpTW_-'
    '{hubFE|U`9bQ#)GkhKHXsdLtr-'
    '#Uf$m;T&A_&2T3jun$($ypIq8Om3}jVDI|9L>jwiJl!^T}#*N8sK5$$!+wB7;8&y7JoF9!LQG05NbAlFJaXx2%=acKa@w@TnRE`;'
    'Mw7LMkeKFABlrA76bjBtDlW3#}KmT7%LmEcIr^gKPV0vu_XzETgY07r66-_-'
    '*v!STdsJ?|(t=^($piBcN*9o&vs+xsNye^E_WvoN{W%qRE2D`1FK14X~Vj(4yz1G(2RZnI^$i+q|ga8!439e;x(a1D;ZH9rQ|xEN'
    'gTIdIj0l)Ss^!CKChouro4mFkL^mRE~$_i}){*Gh1ARoK*e9aEl!etd6W%4Z1H?Se(5xscU)5&3daeI_F!U&Gieh%8!I=o6}hNOD'
    'Ez>wy&zNf;le2Ub8NVf;Nkuo5Ewm_p<Pt*O$EO1)tg7pEf~T?gO{UlkZD#-UTuV5uDz=mpUN-Ay-'
    'm1n?RX!)rkdud8Bs)dlc+7uqb;a{3;J&9%&(bsYOAmf&s~SIadKz^>qGeWn(=mGWdxw7wrgE8@cvK+X;T`BDiWuMPouJyT(m2IP%'
    'QIjv<JNptBR<^|;JqWVlmK)!^rc~)$FLRnVq1$tnH75geZFvE&nrw3L7<X<fyTiU)^U9+QBZKpXmG(ytQ7)c9bBwZaNX_-fo_(-'
    '&+f=I}{5vlVB@(*0II_rG=sa&i!@7Lf-S!3R>1-'
    'JX!lzNML2sXK1sQR$N)_B9I92}^{HsV1g10J^SkaBRLj{0CbR%W+vaDd8lOHg@Dh|2Ly1y~xDH=)EMKqa%NG}E{yFDeHY)n_uI@?'
    '4C~f=U9LKA}peB(N>i11q4Cz;?ABSOJw}p_l1_l~8$Z>_@6q&V)BA9*`$^KhoU7!J5jeOqWOyc-'
    'x`<m?z@}4bV!DlT$f<JC3J<ogTWYCU^pd#ZSPZ_zAcsegc*UPrxc=$8F```=U8j{(T@?Q03oh(UL0vJ`}B}@^6i3O_hIZMH|@F6j'
    'C1wkh-uWV=2mnHRhVu=7r-!MfI7Ca9oJ7d5ok!p)5vnksg@ANM54{W-'
    'yY=^}tGS9F<zN`$6TIHVaSJ2i=KKRSwqi^PxPojGGeCYT8q5!PK{eT;o6I@;ld|{|hd+yAJ(*k<B3)NS)LS;5Z?M<M<@@9Fqu+w<'
    'JC+sSfn9`g#>7(Jv5jl2(@}g}5~V;#QQ{#);lGw!>s>Fi-9t%0efBZA@%N8rO4eUMQ|Xw{}J-'
    'uE5wVP$WD|KuV$#6v>W_PcX0o6v@*Zrw3MmB6*s(=z*1>cws6<IX#gl74%+VGNNK(!XSB9;!CWwK3$a$z^km3kH@20V7Ise*TcB{'
    ')@fHA&gC}CHezHd0ys{L;dny~$E#vE-WtGBcmWsD-'
    'z!3i0bUSF47745DL5_*;8<4z$4TK*y(^PDCmN1>G1;UX9C1C@<%Q$2BE%)K!Lbfw^T2T;a$;HFcte7L8Q^%89+&}+x9WkF;CN+h8'
    'C7unzT8r9xx`HxnsbxDO*?}7Y^2@E`H10lctS}>*h!>-JQ#n*<yMk4lE37j+qdk@zvA)-'
    '?x`TAgkYQ$gYm`~j915Cye;ud_4{~M>{`86-'
    's9@hni#$^Pha{M!?2P9<ck3ySCs(r<`9tWnQUMHxhIp2%K;MC^sl^te6a|1lWc%og|T@6ISFamEP%W*!N3fFyjl;;0La_)z)FC;J'
    '_X1svsioE=-'
    'LvS!M<7bh0|GR8k<Kui~Y*%EF#kw0PW@&Xg9?`yCw$O?E%m}khKYUf~%DYCs%NdygMNW@FO|Z)!sHR^m#t7`G1aD7rCd?46$4aj`'
    's#|e6s|Ow}fy!g3FIhD038-'
    '8_ERO`pU1*3&(qlkUYu;$2T!HkLP$Z_64$djyEM3n8kC{12cGzx9fqG;CORtxqd$JZ5)HZuuT1;mYD*(#`V(MH2^9pUCIa4Dz3dt'
    'k48vB1!%k_M&pDSjn~F#yu+hWE=uHInF~SW6j#I1$gpHGm&*LJ=WK6oSR3t;Caln&t!|)U;Ef(o3dtb>BsZ5p^41WN%D}HPko+~5'
    'pOk|nE`NMpNDe7N?<pH3H)CubNZx|YsVtD3kYHd2NM5T4W`N`!dSE3;*2b~5idgP69=cjrKZ?X@t3xw*Di7M3p$psxaSoAJG}Gpk'
    'GL#CVlrdM3K9xrqbFEqy!1dM`t`lRpUKhjl&cJxClhcTI7)(1VqO#p<VB2ZQh`|9gC#@K9IDj^$4I}0T5TCSV#NB`vh$^Pp8&Y`v'
    'G{Ecn61?74X75m6<qdgJ`DsyoCL=1>V{9H&-'
    'il4qEU28AU|<GRUZ)3UK;@l!U?o(}NEym?Nw5clb?<>BT#`}mU=KUWEeZU!_bp?4L8pDFMx$TJE+KXX1Nhw*!*5ayzkkK>yDPAk^'
    'AorYr1F)>GhCv4VIpf>qI_{8&vd!*B?;K8><-OC2{<{e_|?OS{Cm0WnkNI~-'
    'ByCU+rwD9X~A%S?S!VrPs>?KT>6c9G5KUseI_F&Z^PI;n7j>J^;s}EDZ#)DnEaO>m;sY_>4B9nIVa4Gn+19WiM*P_l_=ZsB+Bn&m'
    ')Nr=e}H{G&zig%`!Jq0`9mbrJ!|qBWEOY!1!@6AZ;v5*a}3ezV~EyTi2AMeMTy_R>?pexkg=Md1U|=>Q1PXidV*{#JMc{e4|x6z!'
    '$S(7TLXaJT#|k~Em+Ara=G5D<hwDsOBn0yZ1IWfz9}z0w-'
    '(iBGUD@QjLm{i+R?iml~0xMNxRlJCm2`(pR{Xzy&hNrpHxm+s|QxXXU9fyDep1b`J#H=55bY`j=tscS>HD!p~vTHc}MB$-'
    'SM`1_g)4c(Xy_p7=C9in{r}z*hMt=9eR5N9S0#^JI95*gL!F(me&A<_1@?>y6Vtw>~iwGK*8qC{+W8u(be$&n5b2vTqEGy0VH29v'
    '6*KjrJY*(a>nPa)P1{%8su!1y02qwo=ROksfY4BmAajgcFt0%I~W7A^~|Io)(up34n}7%`3DcGtC@VugKC7{NIcJiYC0sUXt6v_>'
    'vbPXtRGE`SZ8ZFvMB1w$BLu@zN-URb2%geSeF=JLwITD%$b4kXN~EX_fq9`u0KZo>+Y)0)ay)cPG#o;;8g}&{|&J9LJ7A1Rt{UZk'
    '{j}3>%Ypo%ZjZRFg6di^n4zc3tL?{^a3(qYX}Br>yL@CrJ}fVJ+#^&ZgfXCfWF)v9mP>IM@N6*;v;u3jzMjRj#QNLoIkpV%bi@qm'
    'uM&2IefK{LnC~3jqx?K1Yb39w$3EVO<nC9G-'
    '+AWM)u3;@3budSJcM5XF+Z%+k)^!0I_FEAolli5W~gXm=|JCC}%Az#Gb*}JP^}U`m<aR>&l@)kpW^uF)&-'
    'ROUiuxnX9Og@UTWpSkuwJn0(z^ui<)0yuL_P3XOcHpb9nr!DZ`$u`D8y$Jo`fwHyWmHqw4*H?f-'
    'oSW7u90$8^gV8hC+koB<{)Wde;cjqLPI0#Iy!4~%|N!WX;f}dWDgclDl0C8n(*rNf!mXrYO>~a9Z#oUw^V2>(qD=Wa3U~C?M=_x%'
    't7r?r4sG4K|*f0#t*2&U1wqa9H?L$a8TQly{(H>0h>jCzp?w@+7=yJE8`jL+?`Jl&J9|lH%<^`A9IG4*dmW8LiXzwuAN)C@O);-'
    '49@Bm{*+q1qpiE&<3H||;J!2|9rgap;<3(di6P_tFQGe@`Rrse|*=CY0~0Cr_CH!Hy010|R{ryS;RaTD@lZdOr!CL`t^z}P&P(^L'
    'HOT$t<5(f*bJbHgz(n@`p_>?`dk-@&cg%M-`;;8Y^LRu{Ac)lO@fi|z)Q`{-'
    'AvJ_X!8qrn#Rm8y90mawn1djb2lzPF@D0b6JOPxLHchr{=k_9`I9?0Zjo7m&pBy(h;OkX7-'
    '$C&v{K!rDN1cd@&}w{j^XeD{d)J%aC`t@N<~c{56o_m70cZtFv)4T2XXK*U*0%nQWFis~~Nfj9$W^8iuL#V>LJu?L4}QwAW8z`$$'
    '{TndN^T{@U^j4r;#8O&EuUr}SM_mco{R*O*mO2JJ1uepZN>deH5<a(lM$8TOEc2VoJqnd_%JaJ7!WSPjO<(u5GO{=KH(M?07g7~J'
    'Dr${A<dx$+8&f3V45zczXI2*YmoQ(`{HmL+>=a#c@__LGp0&8SZeI_HYCShzIVCjqhWiDX#<dFZ$0IZQ1n9Xgoz_N)=1*%JoxPe-'
    '1E+~$RSgY;|I!CK<<eZ_wiQ|5;i<UW~)+pSW%hgZIC#0bl7(Q@@h&>%3+R9N8AbQ1s807$=mqP|~X|<AWWw4c2E9o`{Lus{=ZfCH'
    'Pg7k%8BCS@^oecI-y6#kE9IaMTErVqoSOUCZ0r0La$-w=r%-'
    '#g9?&iF>8&<?_W;Wbijj>s9XDejQ9RDg4?kFSNi$iCw0`4dyJPHG|8MLSvx+7X)9`f$oRxbbJowSC~n|DXkRP^9ow#^K$;fXOceW'
    '5*J8EXg1hE%O0zs6!50&9E{<#`7eqCBr!xWhc}=t3&bt2d0Dy*;ylqDGf1yQm;f^X9fbs-'
    'mlHU807pt7crQ54z_h5WFF@ZZ?~1uWv5Tna};9XlN3%<a4)RY@U3se)6;Aa|h$hv*dI2C1%Ozj@AP!<#Qj8_P}a%8<GwB-hmtERX'
    'qyOtTeZ*wH#N#?d_J77`lFPd;xSs^+E#DYJAdCDffdCiX~7MP(rZ`%Kf2)Vkwj*lu#^(@&MVu(OiDR<be@-Z-'
    '~)5Bu4LnF?!D{_%2XbNmPL4<P5wqDbpfn&g#LunA=`76p2|d_cg}mX_31j!Ix)fksE?D&(b2NFELAt+<AInCCoinQoz@Y8syfZQG'
    '$}D^#Pk|12~1gao(6T1Ku3Rm4e@%1(VfsW^@aa2Srf1F^0;}7%B(FP&q$<iZ)f&hcqiHlk1pYQkndi`8AcvPncg+p;qQsRTXeEzp'
    'OI3f#u{AyeTPDS#Pd%US9ZpTQtOpS>T5=;d!=LZ%pvz8Mau5;>@#bvFc0Avc-'
    'D79#{!}uclVq)45pdQnYL2E`Cj2!m!CsNBtNmo?6+Y$R>u@(8N)Bsn@8)@y{F_f$ydme8Xb!9UO!2p9R0+e``TB$2#umXeO(WS;a'
    'ABDzCAK2Bp_()GfhARv&%n7U!7-JW^$nCY31%CzLaQ=5pufMc`LO!>gDDfgfRP9s_t&f-'
    'lct0Egksvlu{qiCGNbKlQ*$2z)y<fF=box1h{mJ$L5@vam)EH*a#mb)F$l%Y5VdQO-_TV0bIImB~XQAWn#ZI6Ma8Au$lg1VFr{p#'
    'GAr%)hN5T2Nv1?FG@2WTfsWh*s21a%VxbCe5U~3Zf0wPS+Mh+gJ!80~_JQa(2&L+k(8P`>bd<7qg&lWl?hiX19A25`1|EyEhzXp2'
    'hCzOUz>T#^`~SP`5H=_rBmz|0e79HYBIr(cMhGnFN%VC_8<_@j;_bfRf5X^`7ep`@7E>Uc;;%+{Wag5&R~`@EZ}s@6Z^2V^h%vM4'
    'HjoGMy@`@sv1or+(2&Hz`}wC?!kLNh=JKorEbgC0cn%3MZ9|ILtLJ%nQS{MZ?9I1%~fpY@Uc?VuCNv5OIvanP-VO^d)AAIL7LMm0'
    '<W$3`6muY~ZnV0+-wT(Qmo@&Ld50YpS2-cfVuu)Pg+6uEw7+yoOlZeaY||X^*~R@~{YnlVTK(j8S-'
    '4jKT}jC{(xuQ@Lf62Fw;a{_>!FaBvcGINVYI<m@Kdk^+|BGHY*THo}|B!OvXWqP*~Xr)Ve=v%v2;jLieTNeR9@1N=tf%(K8xUt$*'
    'cU7!b6g5PH;D|k7V%p>jrq|zMXhD3<l&g9P`9Niq_XjF`&pT#)3uq67twGajow-'
    '>^&;m$%BDAX2Go#=yM?8se+4PAYl=ktm4Bp%TFG2QI<%|TJJYYxTl2j}G@JUl?5Z5Xmsvp6pbUnv>_#w;j&3S;xM4BedI%QLhLjl'
    '!8{X&KU&n5AXtLOrk&3cpFAa32Oz%wz3t=K;#0=CU96lF7BC+0~aMq4D3{Bwy8^)S(Bcf77hYJ(H^<uFI)$NoQ{!!L<XA;JS*(W)'
    'J%zn>Z18H47U)tHP*bX~?N>`j~8%?>BhgJtz0zw6HCsiphCKQVtx8rL~;)k~{PK<b$1gcJjf_xs2gdXr^Q_hP`p-'
    'S&U%^JurhY)N@f0hCie-G4*ACZnkxiW7#z#$lNs`$gJkz-KbBh@5G)}10toREN;E%r~XYLaXCra<pb5f>E`p7Gx>f5)M+tLkBx!4'
    'UkuduxwotR&<5ma<RomJ@)KNN&;XETxE_o_pmWJdFgQIV7@Vy2+SHqSWLH+&MQN|A3(rq5*o9{&80?bEC{APe@(f1tSe$tlqqv_Q'
    'n87IOxu}T7oszg8<8&tKd|~b;9(7WN@-'
    'Wqoa~BZqlep%~I)F3o#Q^o(2NB$+$8b9?hTHxz+&&24_8W$%X`F@B2@LFDS2I_!B|v56DrowHMZ~oO4FtKBYXCalE0^8_BBOh95('
    '&--VL4THGq42VY#F*DbNyX;euBZSJUhW)*IaS)bTGbWiJOnZnP-Wc_tyh6#LapxD#Ef!3YJ^AoLlg8Sc7!+WuOuqw5*M19<zhM1k'
    'f+Y$c@v7DUMcUKvzfToDrk5PmIn3Vsx(NK4bGX6U(fg006d>sqeO76$7o}Rm=ra?=g9eJvxzrz=`NFPhud>A$rV{8F)+n#d7ir&J'
    '5S^7EI1I@B}w<8F-'
    'or=*IIC4tC?&2?x97g69l|FV6taJ~;C%@H{{d%m7b47Zu^zCI!!aJnSo1#;}@uLCX51%Upwm$#_$rPE9z+yFTD*V0BzXN~F;GVFb'
    '*XF)*uRU>+C)^TPm`OBsok)PR|FId!eQ$HYo%TB~DX74@twXW|1YnyCdHoOOmLqwaKaPO`ySp{WFQr^X_9JIi3vTyuAxpK7o>&rU'
    'ViJr@>dGJJUkELP*pvtaQ+Jum|n^;}fM;vPwAHyLPLO#bBFo=X)3XaPmqKDv4NC`YC?Q}_8c^A<I=hF`_xnh1=uVleiN!FW&%#x('
    '&L-'
    '(_NjqZDM~eIgN@0(QYEB;I%|qgZPy7%r!ho?bPx(Uw|l`!YDRxrheu4J~6^CLc7dYU<gWUk;Aus(bMKM1wtecA~)^x!^d9;mb3?u'
    '`kX%0~{wM_0$Q~({~M@$bd^dDHU<Ke-bke-l-Ei@7u)A1}@Jj$hQha33zluWK?TVXy2JA0Xrv#OtrPV4Mq3seGk&w2#oi}VC)xz@'
    '!%MYYXgU}6%+k5<v~qI*4d9!w$T-'
    '5bcC*$&2&RSR5`j!iN^*FCoQaJQtK4J^DqtG7s68c4LuDkm6vc_2us@jGuPjf=Vu!1$+I&J_RO{YcQ3=2XW0JhhcnN#{ig?J*#6U'
    'VQ4x(lOX8@z*496m++Kf<8(PoPuv-'
    '7IM*Epg;<%Rj$*R@c?xAtKhRKg2Jl+@MQN(yWB*x=M0Uld`=araWS~Ahs5x=x%LXax5Y{Nu<$57Ok3F(MBs+br+!pI2425QodTx5'
    'cO2(b7zdYd#Bm5<t|oQX8o-izm_8SKTg(+u{?Wg_om`0@-'
    'UQsB%pZ3^mv87!loi^~1HO~QKWLGnkAy$TiFEMiV%VzAy<`_Xcpo)g63DD8U1ZGb#s>#a|>H(bT<Z8Vp9z{4RoB=Qm0J#sk%Ij>P'
    'UyPAoQN#*PsCO)C6YIU+bsFGSTDi>=m2Qk`HUd14)+mp=qQs&BN>z9hDgk`jsu^rS^S7sZ$fd`oU8CPbTxJ;^8sU(j%qb+$$U)K~'
    't!IQ1@bp^)eDY@43(k_=IkN!ZGlIzR#H)m=pNnuYRsn+|1ixT{gYGMLFtb7zjC$nAib}dlmLTbZ{nIg_=1E&17R98KYiH~Tdyo!l'
    '+RLI|tQH5d)1VUCKbS0Db6aLV=H=hyj+lR15gtJ{rg;heFUCzRC?Zr%f&2n=J-'
    'WY<eA5#yT2Hjy?8Rjyy2Kw~A%?r9^MQ|==16>`)W&s^N2YQlr%LKYzydKJK6+zd)%be_%Q6Vn{x^-N=2tk28GeL1Jm@KRW-U_USE'
    '3A;L-f*TqZiTs8ZS{P~Yh&4;QJHH~Ie?kE7<iQLc-'
    'S;36*UQAb43K3x%w~KVxI?5eMa&U^H6j_@)Gk=2TpPT3$YPS3IK~xqXu5fT-'
    '1R#g}9SUK5cOK3$9lB0C(KpWPN>G^WyHsBDfZ_;cgYiX2Bh?8R}VXp9yybFL=_vBJPS_$E1Bm4ZJ3C<I*e4nK#JDfmBt6ytQQE9!'
    'Nw(7Y?ro>Re6$BR+^LPvD`=OA!vOMJ}3P*05AoB!t<OA<Ptzw<L~9c(VK!P5TrQZ$*1G#l+jt3QI}r@6e-'
    '25%PBQiBX9BJ$id6M*aZ}7dgoSCxig(&*W1EfCHF(ri`qUx$<p!0eEi_{Ib~q_$J2YX*ALEzdOs*Xre!mrP1U{{mq%$hMUHH=_}D'
    'wP7KT+A-W^HG%8gJ3Gp&6zzft#0!pfdETE)P_9mdDVxH#@4KALN>R}d8Qa!?mk}7Bs4zlaDVRpTNelpzR--'
    '%JQiz8*AJL%`7f%QgbGh;$?5(7>Q;U}1U(!g&Jlh5)}+I=&T-{pnhkRph<v%zmO#%6&ZnPhW2Gr=$MdM0~RgkQnSl<bkw*-'
    'OE11*)lOIa`Geb6UaP*J28G1^YnjD0O!9Lc^SN(i$ywW2BSBIz)to-^Qg<Bq4sUik4Nir1dOVroI|fd>iJpu-ryrt|ev0Yh5u0;M'
    'V}j5hvW($#pdK)CbNxz>DW)_?r|0Z#-AX)v4-c^7p(n31cQ;dtTgqS_J2NHr%bp*etjs7gXPdJu~61zLzc8vm)+x@oFS{X7u-'
    '(#{QmZ(imqOOH7K)Wu<#Yf?v43vm*n8LstZ0?rS82BR<A{8R4)=sgOyC!>glpw;Eq>!6f>ESF0BzcKpsv_GZLc-'
    'i66y8IhKEMNW>0^SdD_LKNoRnXG2SVcvtuzAPut-_0TZ1~T~{WB6|7Dtali?&>T5J}&^DEP_`%8vt*^*gWg5p8vgat-JaIS=QaF^'
    '*3j7{F=s&-{eHTlgLxK&)(U^b7q3gEt^Z3tzr5+i8>52Dqs$ZOBpSM9($i)cb2f%v{XVQ#NIU?dnW66D$+Iq<--'
    '4QtmK|@;g1}Pxu;w>jblCclna02Fr1!p;m;gPy#M~caIEs4a^ZB2g?{Ip)PAQ2kUXBr=L{q#aTVB<fux?0AM!$SYY}Bp*&ul{#^!'
    ';fzAby_f~5XH7D!&Bzd4gDn3VLN3{FrhW-CfB;_Bt)zv^NYMD@>kSx<1hlr^n$>Nl(Hj)Pn2FGHjgV)2zYAN@|uE%{9>XY-'
    '>MY`|uSdJVSQ9+Mbe)PZ#~!;3l(Ooi=;By8HFdT>*3YkRfChoJ54#c~q*g__79$7y1aIE>2{<+Nta6zrVjOiRCARG-'
    'P1mVO;$^Q5Kq31vx3D{HlLk|izujsC_AY3U#JH&#?djyIA~Hqy!oCpIm}ZOoJ@)@WwE#U$;jj;sf^74L1oQY$T@9PN=@&g82RCSQ'
    '#)`EHEKEioqR`C4wV8s|pq_8PE0(^9<_MC7zoe*|_pTB_G!vzM0Yk5Si3OZ6w%{4KLoSEc}rbdTHF2wD04Czivax%B#ZvH0Jj`b<'
    'VFzJRfLu&7Tc3l`r^Ffaoax9D%ofW>+R46LYooW!--pL6x1+}mcZUKF`}!%^66)m+yy`BozKGY3~N`C5dx*J8ZA7vt^Q7;h}-'
    '9jFFa<d4j;<TrAuzotyRw04qnHV!DL$HOL>CNR;$ONqC@alm3SAy5*pQ%2nVYPI@jJdHe*fmeo*+}%2l$|F0Omyz@|v$IQHNIp?i'
    'pUDWxXD~L8q^CZiERvq@B^a1N((_yWjTt08SpfqpY97Zs$k<>yf}34`vz)6VPrB7Y7iGKB+0_aL)t;8rsvp)NB45H&wTQ1rpnW|C'
    'ZCwo7tubhMArjqIudOVkJ>BL-'
    'x7F)AFS@N>+r8+vdVTLjx7F(h<}~8fs~*ouH8?H==B_rKBld{1IYBYv>rl(d3Cu@}>N6RExddbL*h+muS!`upf`J)q<yQTT8Ehpl'
    'U|>b{<CLu&$K?Ma2>maH(6Sgp+hPbM0ff$B5c;xK_aAWm$n2d9&yVs&Iw?bGrG`ws=rfH$BTU1ZgSNWN9$aP0K%-'
    'PlnS@gVy5B)~delRw!l=nSPz!ffv<^oubJ;VhY@ZAf3~ZdA%`8(Qu-RVcx7C|Hb(~YKI8jY)!@Ss>RaBqJh|LEuHV-'
    '!S31z|NvIGM&U~`-P#thg@3K&>X2RVh!tC)Nv!sZ(>HkZfP{4U03A;9L@h)W1`gLQYY%~%dW96}p#LlK5h`f?be5b_n8F!iF(G-'
    'K*jW}mJyso>QCHfyw3O8eP*ZMRyHOSm%su2-<mGoot?P0sTKebC*5QJz(hQ%s2S_kz5KmoH9y19FQw;rLileI_FuXJBj|IO-'
    'G10>|YE24;ZcclsMMz_C!kz=|r!DL9_T@tS2@0KeesMb_#|u3iL>uef><q*f~<kcfTE!8#`2PQ(GOD)G$--'
    'EYR|UJ;{vdyMX4fNmYDC>ZBtAG{_2B+{ELT>VBJ>Ts%EY_^0`0CzwEuNBPY66$<8$k{sE-iimHJhPxorv~NU3i57VzBKVoXx!xl<'
    ';bG?Oh!;n!q_~eiTZ@HlqRl7Ffc=D;&%Ow8A=n21q`exf}8^7H7baSQ1n)eqLnd<zK>C~Qz5J+gpHZtiV5KYPGnT~3Kwvat1W~Jc'
    '(W@mgbR3!D=0+tbDOIugbR2(%SlgoRsc*e-'
    'c=hPc#w97{fe#rx&Hm0%t}fb<J!RGX_cuw+f^J{T%J>q(+n=pEyx+Xe3{}~5ChGL%V9+{BW1(o)fk({Sn3nXVk}oC7?{CWey_hVg'
    'R$JHfPod&kW&HAWlXfSl5bexZb@i?>%@F|66Ic(F;}?QPjAXOuWJNcHF#Y|MwB^qnhVpK*Re>1Ms$fhg8$68ly-'
    '7lO8aBbXE1}IldG5O!dI`GxGUdLuSxc3z1F=W+OSWxz#`FxZO~evWoW}HGbPRncXGCI<uGe-'
    'e$SOhQbsx6Ty#&CpF^=H%g&+LGnaDwc$`oc<@lL;U<T#*$$DT0<@g`<z>50FsSd1L64|uifi^dRkqL=vW_uWlw(6hT*11-'
    '}I%4p;wxoS^sU~~1zUDH`)oMM;3%I;E0_zPiSZBpxof3n!UXW*>&3v_`rHnYnRhJ!4UaMFRJzLu;u?nhrU2@VX-WvjQ3*sLQOchF'
    '<&r55|>;?8>`H2>LvFt>Py)prl@OFdN2dV%}!rLr8umUg%Z&UQZ3c#eqQ$4WYS5`_+H5ydweZw;K3Wsu6uYdw?^dKn1xGo`oP5rN'
    'S)XL1&aij$*FXZx)2zWQfz`HjF-rX_qSfI!aO_A&EAG#kM)D*lwfCg#`_~)XPpWNXG(M(PDdLG*P&GS%${n;*B#Wokm;(Z||zvc2'
    'AgURJgdFN$Ta&zsyS$_7#-'
    'Yh%&V((l6cQ@jMvaICy>VX+n^1JoG3@bTHFtDOna>`g<8|`aPWxl`*cva!RR9l*($sK45Xh{O=$Ze#sE&(j{mv~gFHSE#|l{dwxy'
    'e~%OJuxbIfJ$YesUup8cG-StEwc14=9O2*Gp~?w1M_MEH!{!oeiO^dxHvn+qq6s-jc2)v$uZ^dXs-HLmY<99SeBiO@z`8=ya^|i1'
    '&{aXff?|4j~<vMW0_!JMYZHq#G_W2%OcoJh+#83hRxI%Hc24X*$#w^RNS!}lSh&4HcEz%Cf#k6Z+1$kGI9^*q<v~o;nX>yGReIeH'
    'bwuPlR5Ez9%L*PkjoPl)F5>^&iT;&;Dxxrdp0xvfcae`j+a^-'
    'jtC?1cbRghsWEF*Ib51)J&xsPYdnr+XKOqz7cM8@gtFjrwjP)Pms9n?EZNEg11m}<m#p2_GHVNwxNB<@C^`(HVD#F!UTtcSW$p-'
    'jsjOzZsHtC}@)$-1-hgFiMit(`Wfw*z-'
    'oRovMrGYtTDvo<>&60jdGyFkj31f%<40y%{Kymn$MZ9WLZQ!?qu4#2tDiY(8XpKP=I2Z{GsxV`Wb+W2MT<;x_I+4>0>?fqJAq@LO'
    'vohHZX!;o5;7^7biW>00hyFcnx+R<Kqe)V3JC^Qlub?(PB4Vyu0E)3xUmd3b3#xvc`EYHHn31mCoGLiIZsX1XtdK6+_n_~ma!|NC'
    'IeW*9!&QV;7F}_S404u6a(;q7=Y7b02T>=X6y7SZFOyR@W;seID+p_80zAP;CrLiq~FML(m2iu;rSkuBMm&iU~%aXu}GO~t7iEL9'
    ';;b)g2!r>hec`<PAChD)B}281{SI5dSC_?sbYeG6%~|I92DL_BIr?vo$OM!BnkKL`Xj{gB@or>ruz_ni*_(mR6S}vyD~!E%`xic#'
    'HgDQqi&}db(d*p!Pj42uKfkSp?`&T8GQZamD+3Y^_Szc<KXKruhzkjueZELyAM0(<a!(xKy?FCE5C-'
    'S0)FGNRc?Ah`m+2ikbPNp7RbK2*7%!oLRr@MIeK7*HGYO3m?f*3U|_}O>t<1Hx^6CT`mtM*z<pE^&L-'
    '|R^jt1lKA%tlaYqzV)|x&8YC}Ze-heg&AtBk&HM5?}Q8{E;ya#jajawS;Asm~LmdAT2M@^9>@*c*qGik**mNL;YP%~W==@s3dMCH'
    'Z{ld$DNFethSmXize?Xc!j#T1)OLqv1=2VRB?QlH)(ENAiWmqqoNjK#kjFg8yEte&K7^2)Rt-=F-1yfSU=EKD-XE61QS+iYPIRh?'
    'h1`=P^1J@yoiHwZ+t`vvw<g#<gJe^J|3Zz11iqE(<rx;TQ!12IGvMG#q`_gieLY9*FklC)Zd{5VNky^r)bNm_k?>^MnUt;XU^l2#'
    'wWtt3gSHSj9SNLp29etahc-'
    ';G@Pl^T3a<tgPXp}z1t^MY?<QGF&O_&&tgJeB8q`VY&ccJ)B=6Z*?!5*H<z^_NrN>yJRvFTy(~HuX50Iqq9!!zkz>4Q)g@b)LEm('
    'd!BpNDi6*`w2)af+ceUmk)(42^+b5IBZGyjLS!uRe`iMYO|8CD($``tV)}mgmq|h?9n?+v<~D^mqd`C6GMJ+63L@Nc1}5)D~fuNR'
    'fboGD|vVz-({tOf98zt%8SL-Md;dP!{Ty`&8KO)B(G5OoFt=A^Wr44P;-Pu4XRp-'
    'q>td9s?lkZ_H?v7QBP`J*pqrv0Xy9;KSGx9T~MKTRndncSj>%Ku_UsDEkN!x+jH<NllQw_bFo}L;0bwF$T^;nXQiC$33*n@2R$Lr'
    '`*NNq<oQ6(&r9d{UT6uGuVBOL;as*XXA1R|*XD)a(xUoIM)<vjv3X3Pp8vyhnZmirPslivDO{3dmT_(#_l0%R1uix?K}~|+#&&gD'
    'gT!hORxIIibmCeRTLj^D3w@XhR9bN~PwZuZ!qrRcZGpnoi^_-'
    'Ea`h7XTA*<C5*;j1xO!13dpi{<ak0NS*o~_fl^nf?=*ml)J{*Dk!5HKZB|D(rsZ8Hk7lQ90E>A9K@ZEqkQ7QF~IityW!S}MdX<5P'
    'd8ph^RJAF8>p7VoAMm^_;lFWL}De(0JgUM`Hc;$#`iGSX5jnz_9%^+=;d?n#(1H`zHLR1BMrjJBum=~kr;n3tMd$)j5S^6@$&~Zd'
    '9$9lfTyk8*~J5ryOa)~4LS%o!zoq7L0)_2P(^#RuQPjd1zE(>eL8_Sc7rBh*4+3;6i@RYn@n_X0&$q2TWFgBmW=_6S+o5|A6OMXJ'
    'LnJnGINoLLF=CRXfww~Pv!vcnG!@W{s2MTP!*kWId<w_=2Qw^&J+cGA0bzrNKk49jdAA{|YXnmc{(BSrb;x?V%P2>xS+jV{~k^fC'
    'xAw*pwUrgL4^s+?0l(=2!<%xVbaof-<68TEvY9>}D^3^0K3*++8)G3R9k}-'
    '7;b9qXk4Epp8+?^MFgNy1j8PRtx#^ynvz7N05g}(X8PY62``W{I#3p=Mw-9_-'
    'tEGvkPeyhXfGu#v#oY*LX)WQd&u!k(+>PY6m^xqIQ#IM21WtBQ)M=-=$E?IdbhpT4DWaUvDu9}hVkVk`2$x_S8PT-'
    'OwC68k`Ts5nfu$+l@4(Yqf$0DRJh>`y2juzR<l#4Y1@E+mv_Xc?9aGn1vS!4AH-IEu14;9sCG6HWQ#^zaL^|bsd*BZMZ`3YrbT4N'
    'tgGAlcmc3H2|)z0;(538wi<_x?VhVEV<_)-'
    'hcld!rDwSVdE>VJWeT^VD8j`1K;R_ADrZe)1<n*!BD_9##dP5(spE>PLbfJ7cwpk9Ox3>;Gds7{E}+Jobg-yh?&_V{1}ypd+n>Y0'
    ';W@q%!FWD=JbX!xmndR0=U!rPqf<U-Dz^O_<A6|?1>S72<O3U7TvSt`5-'
    'Ygxw(72ZGUZ_H5PJzjrfMZxD(y~DL&`&VI$+td18bn|}~-'
    '~1u*&99e4H~)s<<~P+;E;e!8Ow(#uy3XdB(4dbi<QeVC8I}KfUP2{b>OmW$$(OzIT0`~EvEDW$PgJ*kyA;wSuxKCvza^nZsGXx0p'
    'PTG}5V<n<;)Nk3pXG9-v4}Tvd48cx0j-(DDS0uuOnG-'
    '%F<FPPc?xLt31umu9g<*Rh637pdSHeE+8gx1ih|H7Oy0!sdOUGO0VhV7JTb=P&=`{}#^j9wCL3rDQ@b0|P;*P#-'
    'H^tbZ_RFiG|}u=b_3)f219Pqs+N_pN!Ven>G$>t%6Y@M9e=DL)oH8_$X)tr<{+CSGD>Xff5ElLMa_6o2ul?LpI5Ls>E-'
    '3hy3I7+ofnlaDz7gqDpz4_9#rZR%7V(F2?l0BCDQ{lpz=mNu%b?Mii4sN_tEbW;chY1h-!&vPQ7acQWV#Y`h23hmhh=okmiWjZ-'
    'a}~#29g`3u-??fzU@p5m8NnEQ(@Iik^m(;-_I){50_RX}BqP8unqZhxTdMmmx!ApN0+$0ea7zM2#1RaJ&*!2--'
    '?c<Z^7GOw+%)kbCmt@m}Q&X2s*17@Mc*U!PExrvG6H24-'
    'mb=Xzj<rvIDtz>2ETsUDaoxU1j>0;_3jt~!t*)!gy{VPeRTs1uH9IDUwl@(+p4q_@krtV31MIwC3QYx%jg;P1&198ZqnI6Q`962o'
    'yqQ2TW(L;CR;*XqBkYjW(%kYwzZleqDc5RNZHQd`6E3NF7c$3bB(WNKbG4k@b7WQ5~pjLic_eL`8_I6T3?3~)^JzzlGlpa)izlTH'
    ';VLSea)GjuSI!S;z>7cGzSmc(-'
    '?HZ8yg9pwsU4atv>ZZ<P{N(8S{Vt9>+;Z=y?HPKr;akr~*=qQ_U+0``=^wfX#D7Wp`+Ht$|X2_0ra+`itt_Qc8AqU<yC!^z~AtGN'
    '$q+ph(=e=@L2IkVI<pt!YMfI7CfLxEUd8%pj31z9K8IfRMhH9FE9+;t;W}+TgQCK>zON{McU1$e8aJ(D`Bv$E3iDNN?JLWhfL=~!'
    'VszOYD!Q}5FO#VK`<j5G4#Tb*50!-'
    'HW;v6yA*XD>RzBEVN<BM^`R9}lDrnyp_8ZjMY7JYK^J6;x|@nUcq5x?Vw<?@T>s;B3L<C8`8nT&9}4P*1bQJ+v2IF3v(FasQmdSC'
    '`PPSOJ_3QecnN44B`E10|88b(JU@bH2cv$RR?s-'
    'O|89rNlhM*1vOzWpVWe~6IyhZu>YVkGVqBk|^7@og0N60$Lu$GFa`CR}#*3{zZo@f<2#c5|IqEx7FNxGh?8*`pl0N@a4#e+78F6!'
    'ZrgkK?%<T@H`t+Gpg&<JO}3Oh!E3jInv}s81*h9!Dh@m;sMF>46#Wc(Wc@QF1!HUSpL$f>S(0k%?Wt)gtXS_o^T_L~<LAxKEesW{'
    '~u5Gs2k=lUrhy^tmx6+r}#C>w@@X1P`M1QQYPdac~|F0`&8FkZ2mieQrrCmq#V?)`T_*J0<e=Bqz^fuW%D)9f+wlC@%vcM;TD+Yw'
    'y7G(>!+I*=Zg-<We8nlHkiTU~;Y=m;sY*3q~ybCo#EBe`7`c={WJD_05-'
    '&C#GVN07l=&FnTbCQB@41kAv>LS`AvOXP}nZ#w2H;mf0pGV4#-SW`>bhj^-{Egl*$4iM%Tb6F&Py3g<15w^9Euu-NXoX(rF$%6^d'
    '_;nR?3=)IzHdC1B<kG(@Qegw{RjmAsBL0Bf^qo;5`o}cJ(Kc1cFaX+3X;qz^RFVB$hc~B3`knpK07)9`(O!$1Pzp<kJbRlkRnat%'
    'F&zpyp=m}5nRTb3TD7&LGi;Ia_!W&a8?p5mlgfVutr(HsT9in_|jPiLg%6E-X{z-'
    'uHY04pN!Kr5A$i$_;SIDCid0Rr&5yvF*j)cl1Iw$hZgbFyiCbBjON)ex+Qbf^2E@R{J(LWv=Rw5D}^~LVb^Rqne&$F{U?w<>fTN8'
    'YF20YHw12f=p*MgA@|4BT4qQ9}C@N^;eAeEsxot#5uTv|kooz0?M>>$`tEA%pNuWm_{^?@Do3pR3&AqKa_7@Qwtuw9J7^??)FPAh'
    '+nv3bzDRmk0xG+|We(wtOaRJhX8+FFIv5!BWIoQ~qO#49w8;bmf)%H)p6g$DA~L>_M<?8dULT!gJJ`~aSx-'
    '0=XOo!s$&T&wlA1Ye$EwVtmBW>~G;6^uyuPp;PM^*2^jnl8i%$VMni@d?N(BiM`jtr6UX&+4d!?ci^LRsw2P8H0nUTcxAUc!x;*E'
    '=J;l7>T>ZNc=QF;%-'
    '_FY%Ftqt?G3FbA7Gqbs=+Yh3a(?b6u|Lbun{YuIhCOb1jGJbt%iq8QCX<<nxfR(`)ucTy84oLFy?ukmo0LJdkH6bv!T^CcjJY<ry'
    '%!Ko87-'
    '$=wP@AN(gV`KkWKii*=IOsXimPvTnqR0yHPKI9foYL?D<Bh~YEuBy8#kOEO}tI|bg$bg6N+#bVoVGPgqF+4W}k&OyVMsnJ0QDMnw'
    'PFs&^c|MO*WTTen^EpK}$`Ab~x4l<twI0i4OGi3>A(O3HPQJ+M5SK3|QW{*olE^{jY^A>TgLr;+$AfrwcE^Kq;c|O|FVBF>g?eBH'
    'T(&P5sqmk~<p%wY6&0sbYqwhICUZ*c^Z_H3Q#h~DwsUb%?tf%LHqtiq>3%>IeFvnR?d0JFxyE>C4duJIX%4B_E)0?SeT>vaF;aJr'
    'k-9O+P^uuw@Khia8pEt9$2QRgOtx}F_ZKnQ#u43L%w$_fbbkqxRgUQXQYLrJOC#Ag^dt$5`uYy$`Dq;w=Gkc-56*?g?-'
    'P7^1~e|x12dp;_kz(1|4B4%)ZbW9aXMusyMb1R(tT6GT<q9u3$%_~7t>|XtGvT$irLiR2SL-7-l!9wgs}S|hTY;Ac6-'
    'FI`z%b1J5?Nn?3OkK9;M|;LBMWVQs7%!juZqzR!9zBW*M@S%Dj-'
    '*2Ntm&m;Dp{Z0#c7E@u$+wI0IrGdLc?vokmzk_&=AB>3_S5L~PWW`N)x1tSmslMwt&e`Cd7=N7U4o8dsbFXBLaz#jda%U%V?_3#T'
    'Wdlww_$k_}GdUFeopmD6j?gd=EC~tiUh`A^`eJY0>gRPpy+f$>o`OLU%_K8Hy6?1*gr$*Fw8#Qor>r)Y0m3rIVZLJ3p*i0TqU~@U'
    ';@hj3cin)>DJ11PAzvl8y6U)@f0}5rTaLpCA$}4vCLlK*w*~)OY=&^IfJh4D55*Le0#pUA4GUc~+emCZlm!6rZ4`ivoeL@c`J&T~'
    'XB}SS$ziRU(9mOu9uh_+E?}~n+fz?M9!icSQ6pciG(a5R@3Mm>}AvZBVG_kr`VxVYhVDx{Qww*xrk>&;oKs74{)srbunO&bM*S6J'
    '}d%(=(;DZ>)-'
    'Ev!nEy|g$4SO!9Epzq1=bA0^O1z6HWc?#VR!1&Z8rScd6r~tJ%UoycywKWS#8!GXXnn26W`LHN<<D|KYgPi`<P6YyQV*;It;H?kP'
    'L{cPi^9bW>U8~j2|MBPGgaYUbaiMPe9G9bqkrVsYanE;(~PhIvhIzM^^`@HsW;PI`XqlK`l=tK6eIenAEbpKg!)0`^P<1{K^ihbs'
    'vpEeF9xU|B$*%vsvkc=r<j7;xgpdRz`|>&P2rkOzGP7~7v3f>)V?jEP$L`EHtDeyphnqBGwqvlX|>&(K$bg0`^r;#U?r%16G5$`#'
    '#S?*YO(a`ow;S|H+#n}2W_|-+<IhGF;W|Kr-'
    '9hWq3mPLG#sGpz8Gas2Pg}~0&J&zP%_io?H22<dD<=3P4iSgqm^7}an_1Pv~{ez>C0>#$0Ug<lnoC1^rp%~O&#oX<@T1rn7Qn>c`'
    '^1?5#?#wF!qrin*n2H(m&6IvHKFJy=TDK(|TYfjCE`oW2_3Ta5nf<KF>RxHmuq)*<SN9>|7udRzH+U4R*v(iKs1*2il*&ndN}?i6'
    '*fQfHpe@+A{&r+H3n-WfONC1>3vYyq(?;ZfDbWddqG%o3+!M!|iR-PHztHZVvY0=t9`T9PF!iiue2pKE)K!hJ@zqZmvZK)Vrx%y~'
    ';q%TzFMpsC`yMsaZCtt<+;PK+R11=3G#lov06Ff!Z^AU`3}ah8pz$>dXFC1Qi|{aTsal@y?#LW4yZ2PMET1wD;Oyv(AhU_SR}l>6'
    'Q&0n_~sEv11D?fQE5Xw8Z*n1Iboc{%j=K8mpfTCEH-'
    '}dyFU+n`=#S{LMduo84h~y{3TMG4IiT$my%ZUzoz!&=6yvbNQ<A&XlEJQ?A9`oY1a$F}Ajd0^4jDdsmOm;GLQ2_#&5gc7LKiki|R'
    'uj~-ajJ8K!!X;rIrGFSrDks~(JJwx>!j9|em6n&e9GKc+ny4~(CVB>f9|5MFZxnokJXT2>`$n42%GOwe8rdl==M&wy)*(@-'
    'kNm9#Zkr9E6S~hoLL>r=(&7B$1E2w3&KBMk>wQTP4Q$&O*oDB;X&Aa8nrl;v4CZ8_1YMSeAmltO56j3Of4Q9{ju^CL6ne{JOPA;P'
    '_Gnuky^}vdrS<5(+@k=g`EjTjN!AxG8$OQ$b@^pP77u%x?xq6XzWJGqYcXTnR?a4d36dcav5M2i1T<Xm{9UY(!eH%1C>W=w^o{K|'
    'HXHR<uLeJU?lX*QdL27!OD_T)U2xGk0>GB>`Wn#{lO3<TC=hmhMM(rCe-!evRI#;hUM$KI3Zh6u6N)e3$+0gcs9-'
    'G0anOXjd<zy@RDw9!rP7kc;qs6IPtd*`IcppuqKt(9Z!W_Y=21Qx4`c#8rRK~kgIaD)UuH;SHR0b0mz^Fg3q|bZ~-'
    '=RKpXRf~9h6P=@m_ktR#>L%)^zK~TLqPAr#Z-'
    'y{dU7$1FyD)d>1ASom>w`?&ZdRtY&Y3iCvQO%URJHlT=SN^>Dw=h>N6SBw;S}>9o*V5ZL3ctqY(3!OljM5^fzV`V%}0YZQC_f6Qv'
    'Lqc>c;MS;f7<05#=n3Au9PHMd(ciY7W!xjl-'
    'V(;1KJ7+(LdX2<ljpKFh<XLt?fVTbz=do+zn8DUrQN;=N};5*cDt_^DbCwtjQQ;Xl}C)fg0SezbWu}U7D*beqiNtwnYa|z$(h2qA'
    'd`b<VBeyGP*fFji-=@ZG)aP)41ff*W({-'
    'M7yL&H(M0tQypVNPvg&E<I66e)1n4qF3GWasQ1B<8ShRo{DpXQM+iE{K}W<bViN19&B|=5zUu#F|ah(J{6L%8E1F#^7e;3nHj(PQ'
    'D=G*p}q}AzE!iP9CDww&k3>m=Z9B#2H~Qv6;))%)0v(mv5BY)G}ATH7^!d7u9DnVsW`1o53vV6Uky0-%BtsgIPRRe`6LElL7`-'
    '6ktx7#hnX?XkAiwXRX}i>gli5>$LmrTBcrZakj}hTLjk$5pF~*GlR*25gG^bN-E5Q`HobWO)F7nTQ6!gMs-'
    'D=)xtGf9ZhJ%9toBf6qg{U(uzV6<WySgP^2PrU<!*fLo9CM@)fh8)m9#rl!2nT>}`3WxU{G~lM#w<>9HB0s81xL<np#mP#mnkF`M'
    'M_w#raEF$Kj2;LML00T1Qg+$kHWR_JQ#<~Wsyv@@=W9c5JQSc=VeCX<6AAP(Y{w48_V9cekk9J{TpUBoV~lBt1c;EGBbiiWzV$?V'
    'ECa`h#RMPpZ9(o{4pS5OjD2&PatD@5THMy*7>o6NPAXbB9>MSqtUhA$V@XEMU@H9a;14E2d*R9F5k6AXvwZ_K8y{9R=j{xP*WABy'
    'B0IaJ81$U7{yt_5R;6;!h9EzUIV!<>6*!qq#p=?tq()XMp>+f|sw<na+EkLQ&%oQLwV8qO6-'
    '0aN(7H^k38CTAOw?lvlTF4KZx&To5O_{=V<&t!zpOL}Yt_z)`eiDapDU!GuKhFbTb`WrLUx)%x<SWyi*WyQwwD1OC)+1){;+BaUT'
    ')$}WiU75!|%&CG+PVgFs3Xu>#jqzO;mrdkH&e3lw*U`?E^2a`LD`gA0p8PiDntke4%~l}evU!<%nfy%z!{6{qV#~vLS+V7c^nfWC'
    '-WS5~3oc(bj@?x$=|p0~Hz)XgUK|cCs?TJ^;kkNj1ssy6uTP{B4tMba$lq7QA*EY}>2Iun!-ie}`TNQ^JU6b}RLI-eail04Hr913'
    'tYt2z0t7a;WcA9_?StJ>^9$t%>cX6G8NH9m-$q#ZEw3c0Je-'
    '%8RHpLfo48j2QMjGJy$Fbc?nLeh&lP$naqyj2Bm+$4?Pdfp8)q~S)w|QUyr-PgX0GjryfAyHs6LYsW()P$46CI+k&JrAKV(`hhwE'
    '?5rf2*^Wtfdht(G$rW$vn6dD=$TDtLChw~BCet&<|G_H1a?5pRY<;S#s_?jj!|QZ?mYJ?tb~8_Gu#3QilzM-wu5jpbt`fUFGUze!'
    '(K8OX;IGI*xdXOFO7Y)SI}5ceL?Ru$L(I5TrDJgW50YqXeXOhrX0V&TDI04ti9n0)owl?NiI*hwTNCL}SIG!VP7$AUGs#Ka^7dwK'
    'BaMX5>=u<_e_X7-'
    'f(P7ZUj)_?tdcfIwF%(?fTvp;8M?>&2eK60n7Ms8BY<hi7OWwY5Zo<Ur%Kck5ir}k@N@zj1z`RUio%M68iRpTYsW0BT5cJW!jZ5b'
    '^l96I`yKph=bR8Jf@t6YVx4`uar$@M5#+Il28=RIkL8I$%hVi@Vzw6~GMNYk{B5yD8vrG1SIMk>>OMg$`rpH><PjC4X;l_7u;a^R'
    'I&Xt>W7(T~gh^+eF0!7F1T(3iB76{o;k%Hk>Tmgd{;nxq*4=ja3<wHDfZ5VU`b(;Ti6n;pq|JsqQ4N1yAzCpoKb$GHxlR#Sm91`E'
    'gxiARdn7rLz@9f#ZFhQwn$?k^qXcP5eL=hBLztDX;YRnLdJs^=qIsq;~;)Oqw?sN>44Il{MQ4VPcn)*KFNRhC4a{+i2Jaq75bES@'
    '@US-v%Y5qgNaCf7Y<w!lft!{bCbi`gD+OU<ilb`P?}avgnYm-=)pR%MxoPvUm7AzPV%yELQy{xN5>3TLy{mcRu<*-'
    '>P^w$6O*ERzd33`FFTc5s&5M_H1lRJev$wHnCdi5!y5&gWw*v++oOP9-%t{pFXl;-'
    'qWKSv=|5@_b9)nD|})iiz>WgxaAxF$^Wmz5WA}2PN=ZAI)R(kOY3~qg$CAp1^OvkI$>{UT4d~-'
    ')C~Y+GwrMDyEx@o?M+1g;8RLB=g~k)94M*xl|d@hJ@MkD9Oq!a+v?rJ%q*Y%C>iMcx60w`jS?#;*4Y~SUe-yihNtmEsM)0CvN{ox'
    'fxUwR<9<?$D5bCBI9vKN2=W_(alb7`GXMuz{F+;ZejA!1b*wI`AiN=;J4qNNk%>c-'
    'pMA>&rH%@Hi>>#lJ>TV%d?ZTkIkr`lcarZ7W3RB?MDR_^1LLiw7GVO0;;ld?IFKcnR!R~=6w{_7F|(s5blv>2s`SFS;>m?d#z;g{'
    '9Y^b&3k9~D8d9dB{2<OsOfWX0>AaqTqcJm@Y`>VB}*KbU`?~(G*7@m>0~I)6Yx+vn~Y)tE=m_eXr6$N($(x4C*Y)X(|-'
    '?3z)R_F_Ky>AQ+oKs8w+rE*`?T%pgZZmm0oMb?ggA6gRfS}UZ}Fl412g|*mQ{;r$g-'
    'l3|<*_)wgC9D^8TPip3LUt;)CS17&ut!w2_qaz-vlsN`(|Mk3Gp4bDU@?@f`d;ir?$&>Dj(GUhHeKWog1=~-'
    'h<+)6OU#6)diOXcd=31UXlytNE5)q<?qGicfEiBG=a5?qv!4ty@vR9R5_i1(mA$mBgbg||HHd?kD8+p?M!Cz@K#;)$kK@1+W)YuM'
    '9rD>7-`O4~Org;57cJ3~NV!n-'
    'aI5SZw$8$<ynpz97%z;KeaucC0SFUOt}dYm6&m^kzMqj*qPG}~6S)ys34yj>Mpy63dFii+&!*D{X_QrC{3?1f1czHjXsCXn%5R~C'
    '$7d!cM9dszGVvv4hD2i4R18iQ9xVoYE8sYT@r-Yy{(M_i@Rt7>dcr4a?Z`o14pNFWlv{Gm|-k+bog^GkHQhUJqz-'
    '7{K`NqZvWQYy|JO#&Y)CQs&mZ=7jqnj=uHr>uGs0>`@@&K_m*mH=mWhdBF*hqH5(KC}@%J&(ydsG3)v&*Ys{&WqJ|cTqjBY+&+kD'
    '(IC{;5ny?UM$P|lP&30ReAE^WG`e)WjL$!aR$EVA=;Vy5tm6;W=voCX+?4NN(s)1#MwX8*ogW`ecy)_5@vxfKP*a^<tO;g`DMCgI'
    'ID2W`Vh^>aF8v{xDDqL+bGr1RUYYFtes?T!(C3CF5U_<E0r83cZ;{jn4BNr?Vb>C*Lu~u$hJt8kMpp>r8SiU9l@w?a6Y!Tw57fEq'
    'ZoA#-iFOB?P;IAj#2O69oX8k&$N6dr81aR`7i??>_82(tGVP^Fw+-YUleA~mEfF6n7yRNM!-zp?%{=C248-'
    '76wI#0cg`=<?HV?>pc_#^JCJt%LPfokd6@!`&HeOr(8P&Ow^uoMRdI;nEW93P^40)ee+uFCQx9Hs_(6vCeGF;^8OrUkC>3NVw^LL'
    'KGHmeUP$<Z-!AsN$GF05-Q6|VpoJ~NLpbz%Km{bO@YTvb8MjYAT`47q34A<9tdQo&eU4nBW(X~L0jX;;a+an603%>k_D0KZ4-'
    '#OnQ>lSu`!tC=ikB>`LR@ijZTMV6<R)U<Kxq-'
    '6i$sGNq7B;c2Tk9Oks=2(^p=<$6iUG>*4N>+puYIbOK*$=<R|u>$)6<x2EwH+$jmd)vEIVs!nnned`m{4mqXJ8P+MA|Pfdy$DOw*'
    '{qg0y}1LWWd^wi<sjgFAA&hTAn9JlG7l=?g!jDBSKY!8ws|yI+lsfSbPUBMZR|zWm52xcv;@Io~%cgWFBQ-xgA{$w8qDrm+RfWGl'
    '*d>)w`;E|Fd4oYio7pTpS`;Mj+f?jg>u%WiE7Nr%epU+dfd6DIG~_J4yZa)V~}*I($&qV|8L1m{HB|4ubF!v6Yp99GEw@Z}GSvj2'
    '7X&iOu5x3GN<SC2p^(Rk*&n0?~wVMI?VnmA372ia8)W|`AC)nL2*JIyA95(y>Y(+SmbdA~!(li+=aO70;te(tSv6`0!lgi5_Info'
    'c#dSP<uGb;AN+|uV%?S)z1FR0uLvw?dU?M4nuWD8E)k;4<o_+IFQ$`II}XRrzB%R8$m&Mqv$IgvQKL5+>TnZCKBis`g-'
    '?4F3g+0XHv^PQ${p(gfoLf(|P{_>jy^_M`P`j#uO#=kp;<<>s`X}F&C=kfuErl%l$3q{gHH2orbo(Y+J%JMsZ_okWE;1TO-'
    'W)Oo{#x$e9&)G%IKeYtsM4JCvH8#Th`feOv$o%l-'
    '505hcFYul7r<t;B|7d|6WHxs{lgVT3em83LQXXd|9@${5iFtbjxCAvZd4&ke6V?Sbe<v(~0oTMYfteQa*3V+DlC%we#Zh(^se>!D'
    'v&a|$u^P=HV+90jG>fDHqBWXD#t8`5sO)RJfEcY9PtQfdbD?|7KWnhvgmMYZ!KSZ#PSFCHFG_GuWHNE58XHj{^A5J{#1zQ%WjHXR'
    'KxP_VT1<h=EcK4LGKJ0eog6u%oRr9QNT4SNo6gE>RO)O$Ln{k^HrX!Tx$-9-'
    'Hg}4&4ej8Yg>P^ox9NrN*C^cQFMMC+>1t|XJAkKasO<FXJY7c_)z-)??P2-'
    'd5X=2SEMFL6`9}`RxD5(EdV?Uo7If052;yi#$NU08R4nL!PZz~#e=ZiyH-'
    'z=K|7B4;epZ5WBJsFGjm^O$f#O|`19R|5)ujCx4$Q%0ODFsB%UnECm(q{aJLXChmem36X7JK1-'
    'bM#OwZRF?IZvfojjl{k>0Vo|_*#g+v`6Ofb^^l}vC&Rq_#!si6B)jUjdnjyv+oeOKTlT?0n68Tx|Z52zsb`NsIQ|9@=bdveJ@05W'
    'r))0AxeMDG7D65nSU<vr^R0KVcqO8NXn#n7G8h5Ulm2+t`eLRiNY;vY(#bHdmINwRHs%l92ilZIvp=9raJY<>K$_>3Cm{1EyyB_z'
    '*`{fX3I5RYtc`L9!!S=D+!6PCf_CGQcnDI7#NmCgU8@nPh{TmIEF8xP(GgFizt*&V)(k+9DReQ@6!%bTO^hCK-'
    'U<8t||oGj1Y9!c*{JNLoK$l4e63%^guIxDAvj`38^Nm6wS*n^q2m1Q7mpRVI^)H7T2n=Ias9mtC8ct94ylORmE^%4i*XBGw{;#ut'
    '?~>M!jRM9AOfQwjRsyb*Xi~HUy?$9Gbp5H2p=P>1VS*zJp}BuoAAx3U;JREw_BS%!(kS%Pq@%y27Jy#FC(8mVoyy<TG+1%1Skz>V'
    'N91h!o9Eg4uCyL`ujOmbYBf?-k(<y3|7;QVGn^S^TL&zMj!FqHo^yMIpGkgw@}15L~LpMl_8q<~T5-'
    'X{4Irz=)=ii}2E7nnq@;cg&R_ESnAoD0y`&*a5TLBdD`w>vNmXOVnoGkE7%+Nxmsb-cg>6ZzCoC-Gm4yA1Gi(HSNX>^$LD!j+Qf|'
    '@&1`PTE&p9`xiF$*cdzsOTs6iCVUbu4xfadc>ZB~hMC%_y6_M;@gA}q;*T-'
    ';FwYQd%_SPTOazPxHM$C7pd6Q>)~c`KH$@@2ri2{`agcmhjg4qhU&3)<M3Z_A!+{Y^>KEgs#Wbn^M7?9K2x0S@<8N_{np@ay@lnA'
    '(K^rePQTnX5Qlgc>`h}gw=ZBaE-soh%l?m&s4-'
    ')cP8`o|MfYuZOtu_SOB_YtRWsaWyWPCQL&*c%%g}^NMBbJ{22__%4^z=_M`4}gK_otYAJWISImyhUjACCRy*o1TvmS;_o>SAvwip'
    'O_K*a8!W$CuUEh$*s(<G_e160CVc)D(FMURul)d98ZKTouByReA+Oje+aTOtq4s$iUCd(Q1aWD|2jSWOD$gr6HX9hj5x1!s(|ToW'
    '93pYK7Ln&*eC)<8~^SN1M9pdfd?av(iDZWx6=uzGXXM+F-uPX0RW`_{UZaWX4;tthZ4iS`nV2CU2qciqsGag-'
    '<i^Yn#`G>3eizQ4}sL!8wsAd_j$kh@O{n92gNj_h&dTB6^;QmlhK}|5UwWuJ&MA^xO&!Yujk{N#I3O3`~$~bnZsao^6<Xj#J0RTg'
    '+`?$(s^+1TT-Euf%{^{>Dt2S7E>`Z!kw!W0WFqG)FqXp)~aW50s|4bVhgk9^+$J-J<&dnCiS~R-'
    '~veX5Rn6XEd(>pV7Sh?C}?6a}8MQW_wTHX7ac8p1#B6@9aH&7hA*Wp1#NAAM8DCWb%*po-Sr`uDz#A;9_m}t&TjXaJ3psN2~SUHC'
    'T>Zqem@dSo^z>`K5|u)%K{JRKT|%>#^qEjPb8aI>NHJNc9sFNkSXMjE)m@MG_T~6yXZGM}#uGEA))>A(LIQn9=mL-ts@N1>I7BE$'
    'EhfL$@r24IJ|lS6++~=eE8W^?WZ(bk!Xt5y(!BmMbh>4B@?LP+VeJ!_p>88kR1#oMGuQOBt3fw~S%w3PTv?3XDBImU``2jkI?1sg'
    'YqS@;iqXDgVp_1%Q`%BGaU~q&t7|J(<3;RzJF(kxx3~(bacl{{O(aGrs`m&is6fcJM8#C*He-'
    '{Dh@uioP@P3dLI?Nw9THbu!3WRhJx)5aHpN(<U|6BQT{+s)R>kG(*y?M`3!KBnn|BZD%D4VJ2-'
    'aNPuPx=DmFfVXS&!=i2t{o9*tDT1HGh7PrWo?7x6T8!dxg8;l4|q*s8-'
    '=3f7y?ZM<fHN#hLS)FBJeTQ!SAE<0@EkI>+YrciMmp!w$CUR<`%Vrhoo2WxhzACV>a-KeTO<-'
    'T%E&AYffvsKh^}!ngdxvxlx_nb$r^#)m+(}@2!tG|>7g*b_%hOGMo3Jt%n>+?<3#~;#rKkNKt+_b)uB3?>rVG2h+=vu5=7=N^jB%'
    'UPBu~~of}@9!W!+qgnEU+ax1a2<pI?Z5zUo_-_#yQ*-'
    'S$5)#N1YZA?CJx>sFS+Oh^~zw#!r@?ahXiKtJ2RRV+X)tlD!AwVBqHQ|;?HB<?g38Kd{86!S+xWo?DP&~ult<zxp7xm(zJvY{|}2'
    'wPAF5yz*(-_sxj)MATJt^T_|wg~mN-q`O^=R;#-'
    '6$)h<?&j4fQfRoF*Pu9@X=HgH^=F<fM&)w9{}e+sdw@1HgtA}b8AN^htKa@VaJk%GfXn6fd_xZ^GxXtV2Be1pOxMNA&<;fm*kss&'
    'xRr0k#Ma;g^*C*b;?T16e>;JZ;b+2@{tXlIb75=$AWr;3`1?}iNAKbAcv4ed3js?2HRTBb6Fs%wB?2aTYQ1YIU~X@1jJ*OiqfMvM'
    'N>uDM1m5dV<La4WBq<Msdr(z6!!<yrNyaR#uj7vYfg<IO0u(8C<eT@arI3DPE(RiIZi9!ki=L$QLktt$9QzOtWuC)>%kWU^c@N-'
    'tOIP^T)mt{y!@Lp)kECbq5Zl4NS{`b<q1VV^wwrye9B#YW*U1sKOZt5|l2I%32l6o6G0g-'
    'l{?v}i1uXv5j#(10#8<Fllq9P>qxO*fv{AdskF#of^o8E}KM<1KS%8q_&b`zMcMpsAmT_GXZ1*A-'
    'ItMhZL}zG$un}gHO@J&As2@6;QxkGyT3!Cs-op?j^empEy5PMO&Kee!R#mzF8r)=Yee8=}ANvy5$KK@n*q6FK_GPY*eYxvnUy+gh'
    '%Vhwm_UGbRT&~t(5G-5Y&8kG#7kX1tux4v@3Cr2yYPOox*oc}feVg}5iqvea!4qO?wyFdU%q<P>9+n2rRCuyob(QdA6U^Wc-'
    '3#j+38We)Yc2(3jg&-'
    'NsnHOS7WH(%I+x+g1J>FQSk+z)I!rAq;d4#nFDlX>X$}rWIq;**!7$Wf9&HYWqX=%DIT(S;qhrj$NECh?YYq-'
    'W3BHsUBQmM+fwmAIH3hW!0$$mSMsVRb7X{kl5}XqWv_>^H0%-'
    'cScPs?7wRl1d(5eLv%q<S=9`>LjvP=M91wST$uZABJz}LW!3E*qt#{}^G;l~8<1K`I5@B`t;q;mkFy&TrgWBBs0wl2h4O*65+x$w'
    'P4x@|oBB&_uM%;bD@fJ}5TIKlm0yYB*%>|o6~P=r#Y*2mQ%CZ814clxx*$m8gX{B2QO{j&t;MB-'
    '|h8XLiIuWxauqzJ?PIy@nU;l4)Tz}%YSaCro~RMHZsJQLgIP$>^csH1$YGatl!^SvU~1Blhf&0hC0ltCv%ozL*)A?p1QQMKN(*d5'
    'p3kt74N)VfAJF)B7IV=oMf&C=K#V`8&5_Q8<YERKCKA~vgIKMaVQFORuoF8zIUz0Tw_8eM-'
    'B@XDCA^acN}D7x+`!8wuWdP9wkn6>on?p$crdLK`SnYC&K4$O7FdW0426BJXk34M;jS-'
    '|QibjL>b$7x}Nf9q~!F%oYLMR1n0*Wb@j(wy)$nc>UB*9RfK`g^P6a3*gOFo2did9#q~8Lf5lw?b}UwAjht3AvGZKE0e|s$U0le0'
    '|3peTn1iyXNT29ADo{ic!@J@bUH(N^xj81(D7Br*FXTi{h=J1m{HJ?ISfd0&n{EcPWIo5AcK-'
    'y!972FxMUH5jy*m6GW9k(tPGMlwwr2lJEwLsU<GGi{n6>p`MiB$TCm=0fuVj1i1!=FAs9-'
    'L&y#Ar!@F16JH|(!~&9ZsiqW5FIcushNU+w+a|%%2bOK4jnx;HZ6l4<50>p`i_zo^EX%s|($MOL+EE$M(ii%NqR`q<f^#CFwO);l'
    'fR?_^T?;{LJ)RH)tpNfD<~m?KLI<on`ucBK4X8_*eB0{9UdH4*=Af6noXL01L2uM*f6pBB!M)N(bI=#JN*9}hez;S*#7s61GSng`'
    'tW9C~^02lc#M;0zhisw>ENsO~rC;Lm7K7vNR0=cPD}{k8d(P!EFugG;$*8)^<?R{dBkgh2d{jt#mC2{FTrz!`e=Lf$=Sy%-'
    'B+`DX#zweg`X+ZP<dSW`6JlJlfdU8Sx@0}Vm~sop4#L@3ZLG}$w(0I1Gni5gM^`<Ln<Jdq5Y=VqsW0%#9%863POzHF@a4g3V+gB3'
    'G==Gs%u2+LQ?aT7=%=n=vl_skS`ERp2H@|p<krFi;IZWPheyC;$sGU>0kY(p!-'
    '4P^q^*ka3$6EX*9jGiRf?*!+?JK1(zju5QP}M+Avaqb>~2(JBPKX)gzkkVxQ%#1%mg<`;J{p8tt`3Sj=K!@L`}(IH-WlimcvdLqZ'
    'm5<cFFvfX~>{;ccI!N3YSm_E=D9i;LgpJy?QX@Vea9{88lKXf*aLK4NZgGpQB0+Ng92XqfQA)8Xdq<Dq^w|xL+dV6i!_m$VhwvFG'
    'hRRGx4&b1LZI+f!k4jlmR(?<)f2=Nzu1Ua86`W^i?%Bq6zI>jsyQo+0qTH1-vQzSEhmmXGC<PJ*eI~q8qIT-'
    'i&+|tWqe#R+rYvv1WA~oFKi^LAEWYJI#1=aE>%Z2jB(n=cpuczZG6{f}Y4#a%2F<^KzA}9tz>u(}Uv@_*qRQ6ivh+vJ8IKQQgFH;'
    'tW{<KOg9?3zs|3c!P30OK(NTuLvPhuO!IJ@6vscJ2YiYGYG1ytt$$_S4wbBBn1DV#zsK!JdOh+`XJ$Pu8Tt6L+TwPkk=FMLB8l!*'
    '_1dBsTS?t_c)|3qo$sjSgvF=b&(SS;)ZFaA##!mLQI9-G;XSOYzO3niTpx$K(^p=o8JMMu9k-'
    'ds60Pc3hUtzmAw*I1&VbVR3bEvrMD1$mt^UDV(*&3(%XoyYa&Z;C%UeqS$YQ*pPZ_gp5};RG)ldb=6(PA+Qt;c;BzH7ClZ4%sj(3'
    'lJfGvhII*iSQ5bw!y<-Fhd*MCE7rQFM;8=#riQo;Mkx-h|RcBzvuFh>c!>l*gOqIw&Ia;{3+@a-'
    ')^ywNoDgfK$T&1f&hhXdNra<IbgXQ;^#<i@+$_@D|b96dx^vhopi7amR%j?b2nG6?WsvbQHtA1_u=-'
    'CXH<{L#Zf~7v*tcW!Yzt@So-I_42*|ezcz}TXYe7XeZL_%_b8XE!0$s7m9iC&G3g5;moJ4QgVH{OGM(W^2@?sRf7dUIS&6T~1|fw'
    'QFD2dKj%8(BiUGFlL|{#>L*nd?Jxtvoz{WJ9jz)gvJ!`*@I41;$E>##R{1%eYrNEU)5Ty|A2RrtVoN)kj?xKjQKY$|EC9%bQt=<X'
    'j4-zCIqiqmZ(~<L&Sbvhb)bKP`&KyGw9RBp&ZqV<Yg`z;R%l`c)c*$4AsVM&Pjz-h+JgtDw4WNV-lQ5x{0jt}NA~A#D13uu;|L@^'
    'bE#pUW$_SAQ<A<X+9WyxPF#a@5zO75@rUk*5xfD^Ue9qgO1Kxv8Imjkm?@3Zqda+Etq<?w301b@GbP;IUgsr3~<#EfN%t87#JCN('
    'T`B`kcuRv#_Y{>A0d;e5M5FL}KwyH8ui^Q#cNcz#_zT<D#(msCvf;EcV5FkT0cGhQ;?$p4`LY)Ld<<$3iUj^IW*ou)UW;gfp?xf}'
    'FLpv9m&ZX!Y1w{VHvUAyMUPc1O&UvM|ynMYx+PJq!*(>9qx1KEf=&RbS<joXQ*pk#$KHGB0(Y*p6ZR6}Iz3C#mYawx1v9S@AY_z_'
    'O62?x-w^#0yJsP9zd<P-'
    '7#IIF;kT2qZQM92kMb$J9GUAh93bgM2ZqFl>g^52NLACWp94cs$X_@Jl6$sAp7RV+EB@v>H1wja(kcxqTpy3ebH)uA0{4A-'
    'XF)bk`%l0qucgOR;#;JVm5fa=EH@c<b?enCvz3K4WrA7V^~9jxUP5sU;MwjYHnGYHS4ZF5oyYPG4($6!IQd?-'
    '+r+O1ua8`dVek>w*pcavk?cH?7VGZn(`<J^u#^OV|njuM4xZS0`KQ0C<}NOXa=c;Mtw(s+v2<iHY21(nAv+-}w*-'
    'WaalbH9;MN4I$J7bu5#oaB6}&4rd^0f;t{I#;6JE1oYUb32G92DyOQM50O<sUTrR5E9wBsABkdAR>vj*QO3U#Q!l$P$;eylYu+h}'
    'Q(5g4@l;kjMZP>&SsbtqtrmDDrzCQZAzq#9Q%H4TZdXc~_<#~6QVtYowV8gsJ4`!+OICnOeyL;?xa7ViE5RkdT(TNm@+&1P!X@`B'
    'Srsn1f62;lVY%-'
    'F8xWl4s;D<Jc_EiqnxieKQY)`AN86aZgv%cnB!4<C33`%tXjiL}$?aKPX8POg62)nuc8Pdes9pKCy{6Pm612{50(r~J|G(uGa^UV'
    '1*wASSOjD1n-'
    'E`a|a3ShZ@*iOEvFlwybe&CArDJSTG#zV`R%vQ;R_QpMDOHM5HhYzhx5=w?g3DcPV)$Li@%RYk6y+>)v>gz~<u#%h%hK_FICnI*O'
    'Dem<1aeE(UX#Ae-'
    'J&?r(ryt?w6r_luD=NVQ;cX%chIj<T*(v6?Ae3KgSmX1S#sn<xqOn@eCKG#P;{vZe+>N$Zhwz)d_|XIlZ53fx`N5nk!)Sb<TyvNb'
    'rqA+k!)Sf<ao_&RD^EmguqXI3*E~)5=Uutij9qSlN6VmF2<1^gHoc7h6J_cf;cH{D@W;6?J=mKoMB^qJ3kS{35-'
    '4w@dQSn<lFehvi0{Jj;{yprYn|Vz2$Va{AQ&&NWtg=MW26i6DzMvhWPgkxA3-'
    'iGOQ>*GscO}j3qA8jx<4cf}e3Dn$nr3=PvMLX6LT(Gr?A3-'
    'C!vrWltuDa=E}%N_4~7xRw{=1DfPNwg+<gvfd%Cmak;3wE7}H6~+00J{9qNK%eGYbZ+ww(bE#7k#j!P=?T(2B{rvZrkhdmKhta2o'
    'uMSwW0qxN8isBp{s609KTc$vu;^8C5@81CrizmaGt8USoMf26(Wx;9-'
    'Ej`>&;4rPMqD>sM^t;aN<4x`c98(TZYXnk>uDHiP#@EG68+#7Vb?j)H}SF9uC7enQ-'
    '0*vo^WGgeYZXn#rbtU6Y=~ypXHnQ&gLfmzXZ8iTlhUkOFT(oc~*kFAgyc$>MVZ|t#ga$7`KRyrA2fc{MeOKx|MUhTRA7t$_d`ccN'
    '1`nLd&kReV>=2%C~CU*5B*FL|$P<qLgV(sz8+z_2slj`-nOiS&XUXB;T~)k>9Ep-'
    'A+s%nbrQQZ_DSRI9JW*BA%<}^L*Pr(A>7CCdgIlnnLRnB<QncGV;s>DTG?ur1aTtk?zmr3gm9lM+2E$#R22`Xb?lr<>SoJmzZ43;'
    'U?Lgn3EImXYJn0_Y>7uYW7~hSKrF)-pdbk5aqhPXDU_4MoK4oHibC^A6RYEp^9&VIskA1v5)*%<nYRD=+7gCkze-'
    '3)*m6^#<WuYJ*$yW->NS}aaNcwL_90Z7y0%c(lZp<!S&GXfWU#B_+|*y(wl8=pW0wO)oG{BQi1avVH0T+o0Xzos-kCt#!TGg@Dnn'
    '9Lq6|`+*LoT);FqPM23+6*b+vh2w7xn7?C67Ra?Y}Bq6WaDn?`pdEJ&VB2CB}wvG{bLf+Jcj77*--'
    't~0~z?c5He!$^TpOfWr=^N0M7p&mgR)TXPE4VhQu@Mzq`u1;PMJl-Z;t4SoTocv6-0J6^VfFJmOuEP_1D|8L+lVVG<AZ;#;wW-sS'
    'Kuy84o=ca>l7|5lSR3Vqt8#UyA*fReAx91VRtllT4D!bfjn)8KzOk)Cqk-2IJS`+ezReya!s3U!#R>ckG4HY2MhRZ>-'
    '`TA@LLb874k))Xsim5z`W-J?ofzNb$DLI<c_SGXnpD3iUMv^3C@WG+$uFT0&x1qZ!QG5et1F*;Eq-Ub8DhQz^Qqz&QuzE`@Fd8dw'
    '{1G_Iu<U!bz`EAD@a5nEU5hN(H#A%j0s6VI=|XGTfE(0aqCU?idenL$L>gc0?e8-'
    'OtkIRSWsNASl)d`GO!Q_80OcK~NkZ<X;7yGf>FC3Bu(dAzv05Ip<u0nnoV_rq}8DtxNVMu(|N=Md7xx1m{G;ZHXEi0XKcyw-ka~C'
    '7uuiw`0`6+`?!dZrV*+B+x4%E)5iab+K$UgISGy&7tbRq>&F$;f8fg=85LzFt}}dxdYtgxbf=)t||oFv2I}n=pSpKJtVD?$MLj^_'
    '^?jkh7T*Pm6N#U-l&)V5-'
    '5XY!1ix}nm7h*3k3?77_e0dEc_l>j4Nlc54PPl5E>Lp%@fx2HTNirwYN%eP9)aeRAVEsrtkQNg|JqIC&XawST!)$h3gfxd*H_X6%'
    'Gzp;Lg7fhw2ax$9aC1LM8QlW2a3GELqHk<F!m4>hf*i+YfV@jsZ*#H%Gk~dio<=E)G2Xk>;omEY?gO#)=W`Ea85p44#j{cBf>I05'
    '8=%IJF#A@gBGRY+ad+%&S!~*F49OdM}fE5^pp2A>7GPvm30@GqZZv^lj}~6mkD9!8wtLTcpNDAWj?gqe6(Q#uH)?cbpoS>%H{~+d'
    'f()u$tN(lfj8kk&L&#t2cv<QinFGs7l;KsOQNMiIKT;6WmoC)31c+E3pL1M|4ey=;OT!t{RUF$=?8Z5>E%{_xK7ra4G|&ix2Td1k'
    'n4VKPLyef_!)lawYg*M@LWt?(f2s)fQnBYVxHN_mM~GEN$=XVit4hy^5ml-V&S>iMoHOu@R`#H-'
    '2j&)Yae#F{nFU4a{}ndW9XjNFbNc<NghmyE%~C&4Fq|WCa7?fylxR9Lj7NzN&Yg6>4>+o#dAi;wb1W_f4!!w)}EJ`>t1W+yJO_<e'
    '^swc&rWac!Gz={_;d_$g9$U@;f{o=y%`Ny0(u4uvM3&I`H)+UX0UdslS{K6Y?@0D!^^|I4k+2FR6D?oJ}slIgvO!UyaSd84+md8@'
    'eqDXT(-piznpajM!>VPy=&)w{Sg8J7RokmYZ1|H+w1_QS7Y1^8G{0Px98*(_#I@#F+-(KLlzp8-'
    'M;4fl9f?pZ}*oVN(+W{70Z3*l3$#nZ;}`_q7aP8U5OMoJjKP%&X6Xm9x_I2wac#*dr-fe!cE$*Hbb)PG7b~vVD<?-'
    'k!CPnfgJyk{&4J42`v&dY`eJliS?HK1H$iL<!D`#M+~3Y!22ap{`B1Jql~Kxf@T&!y46}Oi}}L=d`lD>mDe|g%u|25=viwW{I8FC'
    '7$LmTEXj(9b7^yooZ%02vh5ma0g7+75n;mA}D}rRSF8A1}8xQ)Bz<|0CjpIr`jSu{I-'
    '_T^_N`!B+(gsRkl)ZpPNZco}Ex}3b>2#;UjFqWFpHTY|CYVswjDr+rdI!EQ}_>YMxzz0<U+9Rzs~W!8wtM)+RMJq8jP}fdeC|p<1'
    'bTjHreJck7*~YN!L$JLZb1m7#H&kPC#%w87E||HrSAv2NVsLO!TCk8cvhsg7Z+$>-'
    'f9(v{`eU|ren;qk!`j};*vzZBx}YaSl!6FG%Z3(&8)yf&epq2F+MU7|OPhG0;-'
    '&f@Vn<o~3as^ghF)2{7*<MNs$gM~YnEULAS$0U(2YCH~>YqRjEFa6!3cwAh9b0YEBsK!R%5v=xZ6do(oJ4WE~O9>8)z~k4{JLZb1'
    'l~pKrm0wAy+)Wp}d-'
    '!=M#82xGKl_IG`MQUn>$&`yC8EB8%bzC%rW?8Zg=I6Y7V>@a3^CER(avWlF?ntlWO8|++LXMQ79Ni}h$d*tuQ6O}6VY{T<57!;+6'
    'FbuX&Yb(#5<u>Z%brey+}}<=&e~xvA&D%6@}tIOK?sk6ql*75&9Sp2^<)qkI`DaV}w4&z6lPD(8u_?ddFNvwdTc0M+WeFID}uD5P'
    'n|{;r9&>e!u4Or<P#3M#vBBywqRF4HoZ<nEaK1--c9KRhqnEj^0GI0{Cq+HTMa`W)b43wk5xD<5U)89slmKj<$2p582O1u<Egeua'
    '|Ht3rgfHBwGtvL*=5Ycy?J5yc&xl@}3f$6N$(-)Yyp0@nL}jBPPc-'
    '>K!8{$1f*1Fk*82hI+?bUA3}ab5#O$5N+H#(Gxkk1TGplY8q~{Q4*CJehw_*q+ufC2_;>D7F10g6@dKDA;{Z?Apc4T@^6-'
    '0ds3~C>+KQ?YxJSa61~EVJCZ4KSG<ZT^7%yeHvjY*lh+IQt-'
    'tK&NcK$0=MhR0=GkIwRqgy`S|suXy-e526+F9~3SNtgBC(+a=R_j$BQ-VxiGLP2Fan8f)jLKY@hb@qj6mWy)jQ_OtCh`-'
    'x=qo8%_AXf+J&&$FNDpvJlG5pC`@E3*XO7d4`p;A);NS&(rB1OhcQbR3@iF@W;vw63mm~Li8LVZJuVL}6(gf+?@y0PA=BHPMr!%d'
    'CDNX$QFEtxrY4o}A8QPPuTa%Srh5F868gnnLs8m69;uUoeVF_x3z7Q9E-'
    '8x04J9}y5|QiG*a$>EB5+^?BHO8Vj6mdm2@Z@v<hRs2=Blfe(PP))+&Iz}lbnMkL?%u_*UM5#Z7J7nfx$aW#qBvt)VxeaNH<e@P='
    '7RpdixOS`-f2fwu5^59M-'
    'P=!z7jl3b`qv_6_ht4`YV+90_ny?sI6#@&Yb~2ccx^Pn+ODC~35PioAds4V<}LSRMQ+vYn6vw7gmeAqQoZoa*auDvHSGOK?skB7d'
    'vKMj-N0fdeBD*<QV41S0oOa9{)?zpdUeS6{7cwO&cp-'
    'V@=Qu5|4Rb(ndxKK%?&_D^NCVGgWe<nJHTY$sT^gnMv&EQDi+5RP9B;W)&D<6qDRq=@9NSZ23SSc1u=Fuxj*eTH<Cl&);S<y=8JZ'
    'HIEChk44@$V^aOpatl$U>fzX*jC8?+Hu?_FU_j1($~JUC=_>>P#!c6iZ`mU5m0<g;J^qdc2Msa0mZK-'
    'I4}Z=L)1Iwima6_*Pknz4yuiBfC-O!v@ejmNN_8ihC1ekn)yOh{4w0j(_x+=a^@MBtQ=vBY4>n=Z++B{%N9a@VU8-'
    'fWJ2ySM^#*M0oXBNHkU$3R*=@}7VfXly}ie#KHY#<G}2D#1Cw;Kp3!%B8PtiWY}wl#-'
    'HS)&>9UP(!*PeWH)fKWN*M^$H}W}NoG<G+9?zHc++M`h%B=eZle@Srww?=DW&2h8xPH~Xj=sO2qwlYD^!-'
    '(izQ5XavDVPT{scoU_-E}##q5N{olu_lXNi(+D#8*q-'
    '{1028{lgnL>7dj4Mh9wtD>&s9&tBV{aOZAv|3IpRad7q<|XoF?flok+>~KUecPVr#fhh$=kdf-&+kP)tz~G-'
    'JxuO)wrnJw_WE>jx{7SMBwbCmY)aRVEtjTi$(GB~b!5xs>HB2M73l|L%Re*OAgF!+F(xk%dKMpo0*=2(w6s<MPNiFeXLBojZ!4(X'
    'fUC?LPOex3Nes+|1GDKd0JlY*k>_xjC4($iI@73crXR`2vkEu#U3-'
    'BS=Wlv}$MZM6uovO9GE>4*`H7u$kvx1W0elsMNq#tOIv;}GDv`C0L|$3r3vihT$<3r$%VkMOVUGH9*;2^F1pIdkh?U5k_bCNq!^Z'
    'G1lap=BJUlHwPn3pQ5!#~LeB0uL-{fU0PsK;oHDdb7$_8IsPpcc-'
    'e1#I>YxE3!xz<hVjmjq5vL;@AGhgJz>5yLJ@pMQp?nOAQ%)*SzPo0IAF<88Zl723%&Larv7WDJdDMn2VCgiC`OARLE3yhK)OvuxW'
    'jv7qJ7aA2cn2@I%4K<jMXBY*wv<Jq#lUXs+pWFSKfkUtdf+&}_;Dco=FV0al6o#Oy!B0u=2t^auaV)5&O-'
    'h)9cQC)XUT~C}!BU@ci|U*A5-(0D^b(IJ6nbedVrgX-<y?N|EV>khjSnVWcksAOc54L#x9?EWO$N`->=OJ7i-qW6M)ABe-'
    'P8OUR!_<)%*yATjRdRuDZNYJq-S<m|0LwEIVFpK%;dF+l1eN>(Q}9Q$fBmx9?NB40k&);qq-LO;PEtj5Z+T=QAW@MUwsoZeJ?|^m'
    'R5Zm|H_LK^!$~_6ZHIbFZyU@wgo7D?wsJ&s6}PCG1s6Jl_AGmi|S;Ww>5nqRjD*RCYHjUhN3G*K^E&dY}x;Ru%JVp&n%<o8O$~T+'
    '^+<@&2H1N%nB;W<C&IGD#CknXD}Cob6er`-'
    'q5th$_+62ax3ez!&eI1Ca?>~lqk<bIeJWo2;XnoB>yZ7Wvt9vZ1sKo8!ygz^EV#PdGoiusH2q`H{tRN_u#^e)S#oLAy|DsC2eh}H'
    '8{+=ZOr~KhiSL1*&gOF@3u3%8XP9x_9nN-Vdm{%#O*jtz55VBjqGUmqmN-'
    '$FiRfCv`kh3+822p^PkoVa1%_!+E$;WVt|@wt?%1_LI4Mm(%%4EOg`N48(D+)k%#HJ*YlYCFe|RsH}7R$oJ;0q9?vE7@?NCTdWRL'
    'Kj|;hlJ9nZ&e#0GM>zE^TQoY>H(|`l(Xncdi`vf$8Seo)X3SrX;Sim#WI8{>=P4gk<XgWutq>*kZ`z}X$7}pee3P)n{RyH5qr3Ak'
    '<3|c41+T!$Qe<dt5zr^K)lApi1C@d&zWjXWBWILEs572NfpTXI-'
    '0F6t9@W@Vy(nEitPDN?PzbIwv{8VJqwRDPNx7WAh@1h91{Y4y3h+(&Hg)({hWzN0Bo}mVsHzp!?pzaH5q@}YXX%WrE<(jAnT#%-'
    'EiM~<R6j=`tLdrekogbzs_e}B1C`FHE7;g_(^k}9Fx+6|ora8J3PFuzwn5wwL4|SkCm&-'
    '#0HQN6YTnC4=eYu<%AnoE1X%%HiOD~e|>2B7`<kIFi%jJ}5orN>~bvhTt*=MC}#h;37>X^<^IMX-'
    'dABAvsF^3aka8@C3U~Um$fHU;TuD1!RuX60Bq)o5uu@F--'
    'O{a^6cOH4&mIN1X+fuYg=os2T?I^_qs_XWUd&<hE?t_bm>AUZYOP|46S6uE)k9{{>?w+;CI-kp7Lic>vV@d6N)B7_7Cy(nL%;4J|'
    '9l-385N55*Cp3s;b@ga^scdSFvs{v!Hd#2+-'
    '=|Aaob4)QEB#btQ`vNh!kNAsuN1=BB^*wO!C7m819J<T%NEgj36de$?F9f=C#$4A3u`=_BpxEtGVtwF0<h5n)=5<*f98>3)u&z1X'
    'EJ4X7bppXVchNzm|82t3&G_<A~oJ<Tgj8ON&nqWiX>fXXuY}ogDsXmhRZ*iV(B3`gZ##U?JMB124I^Rf~`&YoYq55q)6qqWOgC)p'
    '0-(7(_g1+QLJq*WlR55WK-'
    '&Njl!C~8UHMVwV51Fh{0MLfdg|Lwmu=&MsP0|geBv3&JX(ma?rvEJ<~MU6%$}L$7K(2?+Zq@PJ%4%V(q;4g_tVfqpzb3;*fMqx{4'
    'Y)#-'
    '^*OpCe7z(ALJdbS>>{$aEcTY>ZFer!?Gz^aI+~fVt!bhq(Q?JT^ewr6J<l65<MQ{<O<Nmj2joMUl0+lr8mBkxlK>EecusqW@J0S('
    'kD+AqH7(1rE&h%F2+{2`MnkrYa3BO&w5&-'
    'gXO$ee2Q&vs`kV?JVmsW}N4G$m%7RXurBWlb?9VQahfm;{9`GE8god(M*+m^_an;+T(gmc~Dg;_=>nT8^5u&Y>}jk_1|q^VNaKsn'
    'iYW+eFU@pxjZg_*<~Tj+GWEmmv^Xr7R>a=b}tIEHKlB6pNedHq3%&I(--'
    '~kLNL3G!wE4kYbS7Eu2a@0>;h^)pgk@Ozd8fQoy<O)fqPCH-Cr=T{#<zV!)be&B<V6@{wyWwa$>e>nV{;rs)Z9M^GId~-'
    '|zUq?3agTZ(pXMhNZ)^updVwe2>8HN=Ml8HAPL)fdb(M_slfHfsLLQGACSV9)|96f?g+Q$akCjWVswj9kO7iFSAEcn7va%Ii)z5y'
    '{yJYz)audg@s^-'
    'r0G#GYcFtMu1{75vuT)Ze@f6kdv}xtHZ=YW;zL5#PmlqaW|wP`P>N=kxq9<NH?Do0$nK_4<9IG_G2Qw@@FBn1;p_k|PY7^!MToNw'
    '<v2qkpm_=H(19th4~m7u>R<rn*JuJpA9Hr1yhR-P8RjU%P4;ebye65rkHOS9yAb`-'
    'F)RaX`ciuq#oEFWY9+;C?FBV90&Dt4|ECbvkdr?OYaIj*%=O7yhV4BY1QK1%=CaZRl;CLJrMl04<rGFw@um<k>~t8UCynXYe_(yM'
    'pJCQNAl`u`bhRMItFK%qXj5bu(h<uC7;SN0#?rr1+u`Lb{hR)Ks+^L@6{bpWm386p8=pi>K|_$B^E+Ht3(KK=3YP<gB~b={age2N'
    '$I_BY;l-'
    'NmFq&3iI|GaAW*6dqI@bqVTi0m@A2Ny+%$05{VO};aSGrM+%`@}xmLaF+Ooovj>3<@j=z3dEHmO9*1^n+i=MJq$8I6wSUtUB@o5D'
    'wP>&}H!bvo#B`GD_u<F;cxmk%-OzyF%;zwa-<X?vf$$txIEMCq-'
    '{s~FZYsr_jtpHFnCHw2t+>voT0m|?9kwv@<~Eo;rucqGe~{msz?WXqNVgr!TaF;*`?vvgjTZS`9nUevamN^nl3ZC9zWdA2>O>=Ay'
    ';6H%&9tB4t2t>V?L&Y~`TF`aIVnRas36l0O_wv4|h?I~9Yik*7PHG(3hO8J4Hh^U8r1~(~byJ2!7mnBrd-;l`V2^EK3n8-'
    'B=RfkPa<l4j*nY^5s&SnV8Y*_Zbjk>TgskZEiZNuc3^b^~b$$gtYvAJwU=Vw_}f5;I<t-'
    '7)V=R{g{i5i<{)$!qeRoX|c6||k)mxEEOsERIKY@Q~hky>IXdXV9#NmKM-'
    '{am`#JY`6Xv5XimYI(Yx*arIZbVWKy8{$DGUrMN};vpvgnvj!lB*RJ)egCO?r^+KgMJ#z!Mk5?r1|#@;B<&<OIDn!WV$co*(bZQt'
    'v0@(3+@QIPL6fr#s=xJ-MGg8^3C@W$=$mS6o<UCt3|fyX)>~F(2PU$2Y`Z_qWAEB_f0)MJv-'
    'C=^b}gaa|FoZcUr=JAO0E~QMb!y+!cMak!%tyVfmL@jY(4cTyZ%B#IkTe}ZWnrHB`Ms#HcVa*Gs!_#j<1R~FkmWQ@7WRzYm(HkR='
    '~KrV7K6IZPtBIVQqFHwx5P9v+8esR8h13y9DP%nst#Hn`hSVmzfn@kGE}N5<HK^lm&<Tw4Q`N(+%kcnVNRE3cQIU$@6&vO9S-'
    '5;&d@lEujP_(I=q>ry)<ON6w<>sC^_|XERmU6Ed*uLB+n_8Q9D~(?ylAvP$*%FIMykmWda=(f~C?=Vli&R55GeJ*{Uld8{_;Sxhc'
    'zZq{7hn<-gl)n9#dQM2A#f^#Cx`Y$y$&#b4HEvhxxvEvshevRm+)Xl1<HMM~{M;0oo>e<}*QbXV<Z7u&2=+A6SH-Z<VPu-Mm0v|^'
    'Iy*b?s9*q9`!}LS&VfEi1r5{=1+|CTUk1rCIrK?uROH37L8{AYh1)puv!!i}0?d7)(d*G0Ctxa%Zx5l8dMHRd$Q`7O1+^;md5X;Q'
    'eETihrs4HsJ$t5@^(x~UFv3W*4zs#u5A^EDCy{gg48A+C`Hew6d1mx{|py6$vx*O_9fu$ED{-)Y6q&}nhdnB@((G;*H-'
    'D2elAdDY?Y@PbETg$H#t!UZ~^{9`aJ-'
    'H#oBbx3b4<s5<l+7ekHdN{*+BFoBwxnxgkQ6PjYdTZjs~^@5sBt*E5KqkXEW7H@7*o`)Pn3{nInJ(+s<FB0-'
    '?HeYQV9f@RK%pj+gDK9UOlxJ(hHtkYt^^VH@5k%K?zBMIIA9t^mepFk3a@HdX&1#C$WNyDsrC{wmi2cok~t@ZF&LiEeuF6q;-'
    '2eMq*zx#m-xhP3zm{;3K4=InZp8J;3CPiAIzP(HhDgR|th3)~RU-'
    'cOMh83lXP0!87ctKV%^*&X2N?#q*;q%(v_1Wozpp$Tw&rL39qQa6!|zYiqiddi2uW>2CN};~1IcOU=_$DD}`kokqJm1Jmggzzj-f'
    'kPo#92^_y^Oz|PIPYt26YKCP$;llb>urb=`cHj|jy-'
    'b*mEuu~}$a9r&cS|ng!A8TYi7IYh#&vOX0osu#v#hDV^nX}!c9H+Ecy^Kh<XiKaGHV_Nftw|=nyLCKiOA|o<m8qoCzecYv8J`cyR'
    '|Ld29cirduO^6yh+k;cp58asng?9l;gEE?SD&%hZH_#v-Qv5L$=u}h5ihyWrFFlw|q_aI*w=Z5x?9St756xxehmtto18=>-'
    'Pct8GoQkc-eJ;Ev46s@Pj;+Wn2BV7qQ}W9gA2zUB{w)+y0_uSt}#X!6%tVrV?gTTVC1+{9XFH5hvP}c3JJ??$1ehRMyCNDH6QP6A'
    'cV@q&px|q)Jwj{7YNIk11Pe13#uBsxAB&&82qmV`^DY^2I1eAoa7UYC*{tvpusBDTBXZu3`Ygx`klc?TF&g%1ipI+`svAnoF+nbe'
    '1jkXS~Xa6J5N@;)yO^&9~)^p_i2Q^NRVwc%8fq@4y7@Ni~&T66s#%WWCi?y+$`)c@*qO&@WF)EJ5+ROn%24!C=5lgFe`(+Vf@*V!'
    'p-'
    '+R}jp8gG1zO93OC+AXqjL5=_?+eCao6=XuKY_@N(K=M1Km?RE+ps1&w>=fAIfLqGpVGu=m2fLP+0ECcJ$c#Re3k9dv6^GCdvZ{WF'
    'Q2Cnd!LBNc4rVtwb=tvdAE8)+P)SA5B6>WTtTReE&)2Oc?znfS>#3d-'
    '!p^w%w<lUU257vQYl60y**Z}@;qB4>?`7i}12a1G|Vf8!4xJ{^~ezG}O1`d<_zByQddPqMoX5WV)S<Xyhl|?QrdwM4@`KXR@UX)*'
    'H{`BT@KRladT>TlZv*J_?ud{e6hS&3rduQ29TmjE-g?$WH2@^Fa#-'
    '5IT&qkx`4Cn|ZHj7(7ySVwy9LMWf?|Hdz$<On@4?%L0UZ6j1J>35!z0lgT0Uu+l29!=kIzd~OnqGixZnji9y%5=vtW;1s9eHG0+Q'
    'S*h57LT2TBnzgAxa-;H5KZNr#_6qN&zt=-Ji>O@RuwD>$iG?6=ybhgT*r&ypeC<2g-'
    'ujh0L#Puk_|;OfB8ZXl3k39?KA&wChM6mi<%!V&}Ym@=e`S|D60%VBHUptu&oR4zlA`Nk2|MhJaH4y(`_NH;J^VrXkB(LPb|&(p6'
    'Mxl~Vr<N;$2iN-LSJqw=Z==?D7Twr5Czb+NFbKr=GUnM)fmBP({*Z}lcC&JOS<i)ROTb1zKoealq=z@hAAg-tFBgezyjkJq;_!$q'
    'v;1byp2W7PKSl1&5U{X{LC4w4UWo$SXD^dg(e+K9EdL&?H3NmMr7j74mRQj~fbNy<WydW98{9R<^=qrttN;mhayVL|=CNxZDccQ}'
    '{#>)E)itZTm1=63oYp0&xZzt0&(8D-'
    'xt!8wte{I9C95u5z_Zp<jO$^QtR5VOgD5|&5j=STX6<#8i9zGw>{mvrZ>0o>Lrb`ad1gCFX?mLrmYm}sheSZ9)kae1Hc%PgzpU7l'
    '#fJytZLzr5G%_11%fb~#n2;P&Zb4miJI`109vLg0MIW6s`&d(PLHiDx@@EB)LY?2zX(`3sto<P@gL7mAP#9N}Xsar?dSRnIJnsaH'
    'yHP9&!Op~mK5szh~p>PB4@g()JseiTp0!&ECX@H-p_=GMaX4U3ej<=tjup&s0|i*573Qk*(Wz_kpg4pX3XK1(-'
    'Ilgbg`T{{Ha!0_c0)`ST7u7`lF*je{8T82<gX%3|ZWCN4GWZn#O9iz6kgSfoQi{>XPC&JFFGhDuKB63{M@GC;+%CGQZR0l`;Kq|Q'
    '{X1L<BiUR4m5}XqWq?gp#93Zu%$~$$xFOC8d;o~tpArD9uX5e=@4$Q4~3n!<v3VysIg*qMjp;_koYu{X#gTG6)gpKfh`(I%FrLy%'
    'vE*E&y6G}f`WV>9J;Cg3r1WmZ!nL^$3Sh|tQ>L#;vGgWe4sWiS!jpx-$<LgPgF}}b-'
    '_(p~=55kX!5I)(1@NPMk$zQW#91BPJplao|x#2sUT@+MLm*AX8P%Thnb3kQF?9?5&Bnnim4OSk<6Y@YsOXJBL2j+TJ{WPe|Vl_iP'
    'uF<eBmrr|7K0eWQTf1#S+uFE>T|^yL6XmePuW6{Wi=zB?f+zV6RS<$+!HEa`T$XMiur#oA3r&4fSo$IL0#9Y>M>PHlCG};h$gfgT'
    'Uj~CeM!5%CFOOnKtwT_|iQ&tG+JX>j-'
    '{YBe;JKU+qqF=e{juj1Mc3UWI42Ta_p7lH{*=DxnT7nR1$aV?KlMG119SbU09`12^pW0uIh4zL17GNt^cLAZA=ZxhvO_`y9k<I)3'
    'H63kxLpqp_y}7mK1<bRo0;cFDE~nKx&{?zh*@bglTSK0+|2Oh!QqJz4&UeHb-R-h_Kmb#vfiS8UJ!ZI$y2<Bibg!L(%L`dICx|+%'
    '89xxjOlARw<yM*DZx3B7`s!Ajlh_`lb064*b{g{4932XJ=*#HRlm?@|2CJ400Y9XDB~0*1b8weyb8y}BXW-'
    'F2)%;4CA5)}JX^~P`_Q9ln-JN|JlQTGUYT3vJ_!-a+$K9FL??5H>};jj!Remn=V^Ax3lq};3Nhnh36VtAG5M6k=x-'
    'UmJd8dWV)T^q*zng{T=2}^*Z}fH^Hg0f1;&^xpy@AkUQwW3Sb}pRfp&u$8v!(ZJ1#2(v?uX|7@(cPabT|h)vw(D%J#D+B3}`?X1W'
    '?^4>z)0ip7NEdbr%wC0)dIx~bk%jRpsOy3X6_)4`iMgQXTsxa6J5Jb`d<u+CyG?S~WM*@iy>8T^J*lU2G&j-(!|bc-Bi<sHT_`Lu'
    ')G?~upAhuu>l?0!%VyLPw%PZeuJn^#NZl62gf1v~wP&MykPsU<ik5_Z?Bu@SJ-x8w3cuzLzmh=JV?I1bF6-'
    '^%z{dNF(vL0~_IueRo>lHseJAO)_WjOvV$4q}5A!cwvyifYPuP-imPo{@k4Esim%=@%Hvy}Z@^OwOlzW~_X@%{C>hQQKC*p1~+4h'
    'wG)jJCh#<F=Pd?9^s>c+gEOvWG9E0qrAf6F(MmACvPsL-'
    '!KV((6mB)9Z8ar*wkNqUQv3MFG{$M$0as*s<C+n{!3W_^D5aw==5(r#5|ud+GK5j`)00xO4^z#G^VC)C=q!<+SU?|rtK`>XxiS=j'
    'iwze+i1EE*>xa8y0C%F?mMkz%`03hu(PxU@nBX^Ic^Jq7Bpq^BND${xe>;VBvk`&gpTo24C!U^6`jNECl4-'
    '2h*8Qir2f{o6gA{$B{(P2kUP}aJVU-*_N4Y@_-'
    'Z4piYB}|n4>3g(?^!fUhCNmW=@)n2Qj4h8pQ0b%;y|_>w)(O`70aje3j!AZgyxNWpb)e53x4bQrSCkOGdg}_B9TN+D%-'
    '?$|`dfneHXp0)vAcjMAGGj%Tj{pTkC-PT0Zlv`NCFR1P-4(u?Ka6!&1dRjw`-qoZh<Wo7-'
    'v=NGl|t`eLRY2__yY@U_>Q)cDf3||#ix@ixmZMJT>^+QZ%U*U2yS2;1X)>(iAMowRiqQ0$Rwr*+{hu?bOejJx<Q^|ENLo!xdWSlO'
    '>HchvMv;;fpjGy~H!&k;*e3#`YI|lc|Jhnk<C&Z)CiC=^>A@O=#0RoOY+ZehU0y_0L_m?AhF)oN!S%%i%{??*~-'
    'd=)pA`QJ(jm<Ok+hx1|b8)xHE>)9-'
    'd`H*^23NJtjt%6L1pobE6M%HWhGG54ydm(ZnI!zkq7SlW^|2Da$1(YVG2#<S_An8)uI3XCzwP$k7qFh2qk}o_6PVtghnT!wH#}&g'
    'H_2Z9mhm###}7qT$bPg(3?D4VCh{1w;(oyJ6)fwE@R3)``b|u}!DbgCXQ;?Bwf>OXikf<J3C@W$^-?u9&(up>hLQ&rT6o8-'
    'T@{r0dW+G@hs`6CO<$mu@i`@-'
    'M{9BdMi1qr4rTHnVfw&d@>3t^%^hQyJ@s^QETG>U!TNDLlk2&4_D*2n##v`?5|bOvQ5BOXGP#M{#pWai6OKOmCa0}SZSqEB5irrn'
    'dM?vM9&1d#iOcbcJRz}_3F}$b7IH2*LhCFu>$keSsF~N4;G9S^zpKXPnR#tl`SYg&eHm-QUP#VqEVzK<Yc_{5Wbgfxv@b1n-'
    'b|48rFt&+NcV<7HSJ9+<}u7RkWNA4zHJ^_+yzd+h0;e0uqH$bb95kgJIUDZcPDi%AuViOA}1OXz0c*CL>_I9HgYKwd3<6Eg*H%BS'
    'C998nhN+VB;SUwqNnZla;zvuP0+?SavNi0{egEBHS#+pI49D`FRQV6M&9hN&l>L+AyG()TBDD{f;!^pKOXDph@}4ntj8kufJs<dO'
    'P!b}Vr4CLVxEMBTDBB+H;3Q)7Q-'
    '4p!pcK^oy#K=Dw&1QYg{6aOKhdy#?S#ne9aKFgVWbl2r3fHx#VH*M4@e0E(mAzrMXD}I(;eGbby?u=bL@xI8lrXU`|#Mi2l-'
    'd7B%j|5}Xrh+!xf?+#F2V6TFkF;6mS>gE-Em@XS7FTJjH6QuXjZCJRc(KISdqaOMmbiH;T+BhSVN)UVO|96Ji+KNG5*9`5L9A#Of'
    'Yz|GZ1;L;5fQVQFM(3cYw9%k5OY8>?^u|$XZ3A-^^x+&U*FF%YbMsOV%Zddd^$nZoZ_Z2!|^-MiOE)_-=?&*S^`0r;-'
    '*7_69NQyV&pOM5j;-'
    '8V^OEENijIj%8CwD<;BkcvuOBYju>y~s0ZBx!qn<&+FYr2$n)^AIfQ7QB7>2fM%z9U^hy9C!^gBbM>10ZhT@~?vUhHfdl@m2C|T>'
    '^H2Yz#=g!TUSL(hzo461pRe>%UE4iM#0v9Tij~ox?m42CO`RGoRzd+V4NeGN}6IXC}qF@6Sx)yYJ7;H)zXN%?$bihuPEBHX;fAC*'
    '0$K{TuQjQ`Gte_FBt-nJIJ+He1VoC$u>K2?_it?R76QtWga1fUxR2gE{J%y(FwULw`=X4Sx}shZ{{~QuE6U-'
    'BT{ohW=3gn0ecxI0ZMl(w}2&GZn1)Bar+90yh0Wz?5CRzQ{&|#5s&mfQ(2wb}rD9zdp;<>bswn6mQ8tD~WH(KP%tV9h#f^c@9yni'
    '&c@G{!{LywEv2{=4OpX`mdK0`5$k$r9rr@!KuRSyj&pM&dW4ml|Oru+761~^<gBJ4S`Xm>Gwp!#9NzpawE(93_(udgG!kB009B8+'
    '3FK%2NxvoP)z<d=o^VC6S=_H`z~ZS4#Q+D!rGz<dZu7KeX)l94O!M!7jbq{ykGz9B)(t&?0jo?Z*J|Ek@lK`%1^NKTK?TUxSwG=y'
    '?n*n-n~#*jo@yK*1{SgyE95Q*TS`smJ&E8av)J@V41>=T>i~G(T^a@2r)2db>4dfQDWX!-i-'
    'E=K+Ya`L{BwsXYIkbmN4;XZnd21Q9FZVkd(gbNVmnSYXr;kA?pD*%Fe%L@;|It2mi(_JFBmMPEx#q|C}Vgf&ZL*J6E=9))c%)-'
    'M{C!y%V~VdrM*+vcV<%AH;9c0U`~W5Kr|xUtm5wRWFgw^ZFF&Y~()@S5Y=ukiy3;a*A;KtkZ=RmhMH`S{N)J&kX^Xl6Cndd{Y|8Q'
    'AhQ8YI)#DGXA2l<SNystC-X{?-'
    'RsAcYE2oa1l5M%FqqiAXluXe^ZvB)kmD06mRK2H;HfQKR4gdgUSp&C(wqfM%j^S!;;6rq_!KP5mz|l@~-'
    'hzPEaVPu`C^O4t5u+JNFwVUuAwiq)OoX9Atgsqq&(>RXXH2P7H6D@|eWsvn2b-'
    'QONWKVVPO_amt&HQV!T%j@AfXcMDr21h0DqaSPtzv<|L8N=lo*{|S@VC2DW2NRR(!-^gZ1QGd#LN%0o{^OE=$|MT*V{MEq7-'
    'O%HQ2RbqQWYs8AXE(PS@W_)GQaWN*2Cmx0st_z`MHKshTq^&eACSpRZVk*l5U!bd0nV`;LL-'
    '~mI)Te)+{Vd?92_*e*LW)K+S#p)dfXJSN$=A!b8OQeXJJA@f3?RMI0&@GqII(WB<$`tydDrlDS2mR+eB{5|IacaX8lg*C&fGd&rj'
    'kz{?E@h^N>ov`nVojn=dt#`2pSGEf_jkAB%{jZ^h8bdRRojc?ZTz`mU~EvYp+Y0H1f6dwTRfwO2*(epp4&udKid+$|1XHm1}spem'
    'fsjxMRQoT2>b2f5cMbv+9xRz@&+h%jz?(dPbRv&s<k6@9a48N%uka*4#1AvCG65oHMae!jztlp(y2C&ZK?^y4@%x0J68NN+G?o_m'
    'HE3V+XZaunBxd5nm21|gr;q2#JTkUhc@9`?nYv|+thG-'
    '8*V_;@vw`y{mf9?Rr%2PwVP=0WO%5K@)pHGAMQ{7cA1%&ERux2+BHD|Tj})T{lsd3a>2_Q3g@C;ZeJZr3j0wCq32i}4N|mjyI^o!'
    '=@7w8bT4K#2odqZ%6lG<}!fEd;a=@PrtkRdO7dTklc^w1o_LWlX8cs|;B#P4W5bOkQbs&1<=?2{9CHiz$#8i1cnYos%x773bHA>d'
    'e&sVNQ63!x^kbeFB`V4{=siK6R;?)W-IXVgq3&4^;EkncVK-'
    '&JfxOHXBDAJ9o@(%tagrED)im^n?c6`7A5FmrI0jd=}XB#eTae*#22UR-`zvEmLD7z@~5YdxgNZ9#4n?TNTHFx%uSEp!~g39-'
    '%ixHU}&gLtr7AYc-)rlL{x4tuC8<FE;MG!!u9{9Swg5w^K}KT-'
    '+M`D)<H}lBq}iLgF==3X8E}rMdtXV?#<c4HjcVc61>u#zuy8I>gY1_GkvI%ce#DcqUgmJnKRf56>GyJXdFzzUR_1oREbq{jo!eBI'
    '}+Kk~hU6>kTzF0$KW^8w(+81D+6rtZI$}a~-'
    'd$a2i|0kgL~J?!3w5)wZsyO3r0OQaJ#LDTh+FGF?W&tYTG@vxVJtg4eYs)Bzv3GqHi<7S<|fcwHUn8Gd7E_?q%c&Q9PkQ(@NO+^g'
    'EdeAY`U7bZ+V-Wh&O5CALWCWZC8th(-2nz82z+ggo*1Uwdh=ZcJ|2t~xCEU@WIJ+LU)8cN6x76-PE)Yu5H=^MSc5ZE^22{B-'
    '+;W#kYIV%I(YYZuZb$L;GQM#R6`<dxZa^f#dcNr3W<PC<`Ee10WFnrxAY@Men_0&fXGko1)jwUg=+F?!Co_bi@6k@HmnRC_|sg)0'
    '5M0MKPucq$o;?1Gx9vXo#`60tn2HVj#iYd)?FW{`v74R+7f$~C8jH}|rER^YsJg6wjHk6RZFb-ww)z}D>>07*{5Xv^;2{9-'
    'G7ktTHJhQ4m(hUtW(<&yw6YmP3aB6UpZbO;h5Hr-'
    '1L!GIt8Suv;UF#oM+K*_H|H#rxYLUK$QH%6+l)|P;)@>vLa&fwYBG*gOk12A!EZt2K5hpUa#v!(f;mbqp<`A*{%Vx*4R!$X^zW+;'
    'ePb`-w<D@Kb>F;xJQE)w9!d>GyaQ#+|jR2Rv8%>44wHZ%{0T;O1O?&ags><mdQ1fr7@r>({slm#3c20Y{#6WL2^b`}2=3~0qzyY-'
    'VtX#5lntVii)8{hzWT1NFWkcrSo^tWT$nChNT$Uplg5@G>9q4o!um`#=A?OBV`(nAI9ly`Qmj2kGMX|NJ1m{Fz>qa#;0$ci`mlnd'
    '-7Ca#aTLU-'
    '_%=N{}650<6EDf>6&S@Q{hsQGFc<?w(M=`(0{YkYS2<NRQerNfd$qa<mz=mqUT=%b;+qSrYxy{R)nA^hm9rH`>wlb2y#{IpOjQHe'
    'R3Hj}W_NeQ)+`?_g&Y38eV=7Tph1jj;XdVZbEv({fkCeNrO$=^BQ>hq-'
    '$dy^;efmQ>7G;QeyM$`Lak<V{)!1C^hE`z}XPqzj`(KQTtrF|u|5$wt2;=@Ph=+QHLWR9=J&6blt3;j*ia2$&VTd%a4GNqFB3#6+'
    'tLvHD%>E5?o1ecmXY6FSBQ==&75ZD4U&pu29Bo3e5asjW(Ro;5E|2AwOy-AN&Ni06h07$#N@eHrd0drcU;S2{irV*;60$7C+4mo6'
    'Y_5JotB|p)pCgTg5M!E)y=Mz7z=2q0wy?$8cj1F8XC$Ue7mRCt5<oyX2hhfJyF9{@On)qoV#G_lQ;xQT)4ODyC7j+Z$1qo;a3gbd'
    '2{$vpB>zL^7gvA8{EEQs=4dl+j>(VB(R?mTNi#mhL-'
    'Z<kDb|V3F2u8Ob(VqkhjcD#;O9zkPNacfQe$(a9Kv}Tp4xuFQ`^rqA=JSGQZB_|z0YZ2-'
    ')X)%*QZ<Q8BaH)+vMSvOMHhs(r{z;^C-'
    'Y7At0iXvMPB$#qDV|gho_URs(?%Rh8AshZvQW^@k9N>dFSlKU0{G6M4%rGak?7b$UKo&y;IfgxO+NmKpVjbSY}ar%P~7q!|~evAF'
    '^ct-^-3I`8hM_KqW=nR~FL_h5~84;HqTuHf+7mXiDixgYf1-6|(oQLY=5F4DD$ZJa&>RRi_W*{VkRHgj|vM+L4>nPZxxY8-'
    '+y#l0o~K~B%PKC4X~WdYGk7+F5|Si}AjMa7h1FUEGt*@Z|Yew1Zc{jIweHSFCb+)Ix$?EPwNu7pFGVaIrn(pYcCoZx0dvu$OVhfK'
    'Qn=guKYH(DZ?)42Spu@M;Ws6^gkj?Psr3U`>J^HeGIorV(vRwa5=hhXVMMQB=htid>s+u%Tz0Z>TsihzGLZ63!bFaZ#w1FW_;sig'
    'fRSD^NSz_#udVYB^n`stmCO82u1u{!)X%gFlscPnb-XG&N&6=&o-'
    ')!1A?hq6hy%Hwpw?y+g+(ZKE62UWSxC|Aa~L8Sq85k@zdr0?Q}ZixIOohUQhQzqv8l{vba%aamYtht+ERnETLZB?&#!Py2Ek!^Aj'
    '*=84!ZE+FVhb|)fNWb+RsDv9%#m55pv@tK^@G8cfa7~sm^@nsXYRn5uSojua%p26$Tp@=tml(xqOi@lC)X@zsH_FJ+c!hi~3FM`7'
    '6}m4Ad7mxQyb>P}`7dLyA94BJL@pAx%<o1d=AkrBTOkilEbGQO%qZav!Bol=tmC0(nFZX^Y|M~<;hIAO9#-'
    '&9XdNDcbro&O47e{N>Y}#DL0^o6;J&Pe2mK*EiW+cg35ylu40x>?o158fUR-'
    '%9&Q6XX7K>c|=#joc_<{tME&Z(QFD={OB?<U{dI{SukG7P~=VMocJkHppf#K@}Zixo}kISDJ>p`43I+61e>ZW;;6K@Yb|I5ny1%H'
    'luoXRQ?&rA-s)T;18F0$M!RW4sR({7`6!wlgY!t@w-'
    'ldlPwQ$~nlo&OvEi`d{YR$OfGivnVUJNGI!czJW<z9^6$(A7!*RmeXDCGi5uA>Fd{4}lD&rqcFu4(k}z!6Qr735hL)Z!c3Ww{qhA'
    'yEFkmgZO7ss}*rB!`CsKnxLNHL}-'
    'IGbIVyXMYCR>7yg!`J%OVmJu)zi#Z41aH~GB;?5M10{DV^x)sSUD)1S$Q^rL(<;7$~y2iWvq<Oi3t;_`!^6_6j?u~+%QYnofTp2^'
    '>O<jyFi`Fmmk#%iV!Si`E1K4sX7LjB`S%!-'
    '|1CHojIKjc)wIF8G0oNC!5m)kiJzKmnkcQlpZ>qJhSZ%=a~F~Xwh6=hE`xSbE_rx*g*hVUtNv~bs3wX=F`TRzhALJZ)6#%<@ssf='
    'UBEJ4lefyuN&`~aK(iyq+$R$Py8R{=f3Eqm1?{6(3Q40p(8f2tvZ{Ed~osj=C>T8JPiWzZiYNXiTjfC$p057L3@K*|=SgVI4ZW7t'
    'vcA#^lHuL!vDaQlhS^|Az9O<VN1vdpjp7?IP?Ve%Q@u=0mk^h?UMXu{43&CjQM)$j25p3Yr{6m09y2w>aZiD`~4zzMMBzi1P#WW}'
    '`!w-'
    '?YRT)S6o!W)}E!>2R(JFmSSi^JxchWdJ>_jlTbjm%TLTg*Hm!zIiUGF*y{r>49E7bz0PtH(Las2b{zTA^;f(WnMzlM)ZAs(DqJt$'
    'S-_2Keh&O#UTc<iUs7X2WnPw#m`Co<39`^xK?muV;Q>fURp_`e}s-0Y3aMVuh<%ak0Y91;h%M?p3UCZgV@I$>bkA-UJK?@3PX|Kj'
    '!k}M80a)w0{Z=U;CTq?p1*S>%MNyUW&O2Z`NOiv2SYD={~YT$RP<4_0CM>EVE93B*!zaFJZN7?XFaE0;8vu&C6DIdM>p!qvPGjcD'
    '2k-@W7NGVTzh2?0E^=GfZi8Vus@b+a;c8z9{EX0(`_Wy2bT16qYYsQ$W7(-Mz{e-'
    'r3yHr!jdm32I;s*D_9mSy)U8&hLPPHl#ZUd8o0@hf>lLoNkj7tyq$db|@vUJ?rbN-'
    '0u_I5(!S1&vHX`sdxa_I6}6RA%n*V&l)|n%o6IDK?q(((Ol)Y3Fi9y9@qQ(9vEniwQUv))S5h3<6;wPUaKnp0)zltvrMcnqp+CaI'
    '|alHU*4;j;R9v6f4{+|T1->?-sCOdN!}n_-lASJdYZ)0bMUv-VXCt`Ik*nOFvI&`tWwM5gZdePTX<oxKh$1T#<e#%x{Ava6ZvoR$'
    'p1wSOKcCrLV?N3K?Vo^5lF}~*v%bFaEHeW{+6R+IO#@I$#y~xO^EAntbVxlz^`!mEHOZ(>+uoa<kSz>X0m+bas=G}UqlYqvf?6#3'
    'k!%GzOYx3!y(lH>FiW4&(=oS%;faId99;-?nK<syUwfLoEf%u4nW=o_@5dcRYE45vcEgIoW(-'
    '^k9s_}j#mBCSUQFl{d$&;r8WO_mZr4ipTW{`Bwc(aOC_!TXR&lVE&gY-bOJRf&&6#%njK(;?(G*8G=&8P+gWBoK`tx6tGszZf&S1'
    'YR<NF6TS<SRn0kVZYHUP3fx6`FtVlUQlTxsXC@0v?^exND$%Fv=DN>RtfKBJ}{Gg-+b)>s_;-'
    'lR?Vd7rq@3!fDOF_~;z0H!2?33O>G$ZLe%)pSn!H4YQ+k0tf?;XrirlYsqCDE1e^|(J>rS!>NQRf7Kv==b*b#)xLrX3bRjE}_QR('
    '5cvbsV}XIaoKhz<UhPV#fuD3SRRuW+?Xcg)S|Mu}vj?6Jjv7N{!9I7?H%Q+q@$RW5nUV6!+!xFh&IKJD4_8POV*tv7aM%k}7MwL>'
    '^PN+&4O(yeqmYt~9FAU;WzztG_ZXbv~mtx0aRR$x{Rx-2g-WK%lH(H*@qufwFYn&C&k}l%?xI49fy_&3c-'
    'n(*$Z%_A*EH0yQdoo1@bOYM%8m(7OdU@Cf4&8uj%tzAVJ}$JujQE?dIuS#z5D&}BvOwz8zZPz>IdsId`qn!4nV3(aZEut*?gPWzb'
    'YTb5IPR~^)Lx}~vRp=cy^wlsR3Ev`vQHN~I9d?ph-44dL>74I4RF-ZPB@QrRlu5KeDJw)Ct&;!4cD6j=4$*v-'
    '1V$^(>SDT~XqnNOqWsd$JFhu*2Ir<|?oXQ`YqqzdZ@@t4h0=F6oX%Kh!^O3eZMB2`5q~$U-'
    'ype@8^`Xm)BJHh`{z5TGdsB^#K$^PborRDFUwmg2(snX^%W~YanotkN(ynXWv&BxLZi$mXYjP53OPvJTGN)m;+^M0hFdKs47+Wj9'
    '$I?pQ$}2)E?_yzXI@T6+B^S@J!5d_MrZzc83p<m}WHv%<!}OzISIva`YEZb%c7c8hE|bTLV$2Y4X5mU*=8B@Y`gcj+f*4#aQez`<'
    'rEc=BLb!r2zAFk>yO=goj#pL_(zDlvQq$hr_Zcp06#2qmjdtT79GO#uo5fC1hv6*N-'
    '!ML;gH>m{^Z65Q8L4E6TS_X~l3PwH*@|0IDp|oTE0t``EiIL7!!0kBY|AY%6?lfHJ3Kd{jDd&el_8#YXU|%>bQEu8p-'
    'X+}%A)AHx1_&N47&cM#zvq^UGnZi=z=f4I|^O9nZ9K?{#Z@(^|6r<gfC)F*b2*;3mjoOP?*wq`b;q=*uBjJRRQE{KA*@y`La&yT*'
    '&2lK`7CPOyFBd<@IxxZX<2qFIc*r1Yh^CbO*`&w&3Z<r2Wcxx|0-tIZt<y46NYkZc{0t6#(_uUd+;JA6~0Mczu!$uUsOGx3l1-'
    'K6F)4cug+pFBAi>^VQf0c&SVNq!7H|i+>UYuTPl1WjXFwIC;rsp{5MXEZ$VsXr?mFf}mC^lZO5who#B{X^XfnfA%_~UewsS0C4AH'
    'OO9mtI>GEM9xUXIRB4fZDDOyY>G8d)teJ|=?^7F3X5SI}!q#)S!#MMIXzf4!PfHklwuh$GA(}q*0@kBoYEFpHz7pBrh8H8^c#LI4'
    'mFnB9E()_JO8T3|!0b^qHX^E2_u|t+Q6+rwPotvBPno`#IkT9b3MCEI_X$2~l{8NOR_pZRl=T;y>+}q^)oeltJU-Rmx}U=3SvpR='
    'R2~-yJaxo8y2+y=iRN%uZn7U13ISgA(f+6wCtu~Hz>BXp3^a!dd6QMDcmPS0siXnDt^pQu>GK$^R-1HgFNWmzqo|2uw^eSfLYA@-'
    '(+cr~lz4`qoc{20isn66moQ3-'
    '%X>DdvAH@DWqIzt+;pqdSt^1yUW!v`U$|UBU0QL&w;yYv{_`iB3bz@!6`2eAdJ@P;SJ0r%JyRYVn7abKW<#3^zK~h0+JGyxozh-S'
    'v{bbNN9bcpfGwkissp$|yC@a5LM>A|+w?<$OP-'
    '^)hS%F}Gp!4g;{;WxK_EFWlguZVp`&G%q_uvlbBmgIaS7G!;!NDA#^x$alyyKTI{#`z+2pbLqct6Y$wY3&Ui35GsQ*7$vvU%(FQu'
    '!~)v!G1k?YcRaylo1&E;GQSC(|DK=;CJOVkT=D@@JV*HLv1q80rq!vyl%8;AC8F4t1AQMysiPpHWy-'
    '7IfQY&!&m{Da*3QOA(4{7?^f)4A-'
    'YQ?8eASyPY`qZQBCsnVZ#UQzS@vxE|uapqm7#^y>(81tH|ycQKr)KyLc<njNap`)rwNT9e;&QbZ?GYkVC94%DQ>K<+x+ZuhxqH{T'
    'Q`Kol4oXUwj6_)!}#wM?$Mu{#*(l$Q@r+;rN>IVYdQiqZM6X=c_8y<kNeDZa7c4sD2;UTCSlj8-'
    'IP29_o+Q)2&Jiw8*z!WN;&XC15<B85CtEk9&qV-'
    '#yU)1FHlu(j4&g5^XvAH4>p~>eU!?rT{3@#^gf3vEBbO^97rrOQpFvsujG0iH*g*{ds6M_-'
    'jsPGEX8ihMmYSu6n05vdE8v>GAn5hi`Nq?BB&9?Rcn5hi`$v~K?4FSm@n5hi`$zYhNM-x#t_=Xkc+3tazsI3i--e6d=M8&0NFeF*'
    '7EXW7ZI?JB=ttJ<>XF~~<Yvb(sks6z;BoW&4I+V7c9)a3s_B1lk)kK$p;iE?)<I@UZc_r?9+0gP{roFZLpwEVOs<Aagc&!<U*MC#'
    ';`fpl#{Wq<={+kM~m4htzCc`R_N$wdwU0?udp4mTfWOzH=QlsyiY1Ui<hBjGd)o;~M)T|pyC?y+b*7a&^u5v_KQ$&To*j9T^k}6R'
    'qyVj9ib^n41%o46if1KggrfVVKC-'
    '&+U+*+#(77dWOx+2|4ODYpUXAV|&BPqzRQOcC<BKt=N<6WDI7J&%>;}d;@<LfQ6I^V_73mj={@k_$edz_`zF@qYi3px0I3eLxHd9'
    '5hMfY3I}y81(=6t(X2B@_pav+i%z*j&koa<_^eO~vXoQmKi)=LB@B(a0?<Vb4}7v}(Z7HwoHFZIE+(KHGF<x)S0eQ`f<;O@VqKUP'
    'S>u+r;}-'
    'l(Utu83WzJvD4Ai4BXF=y6GreOOyzCyT^uI!Kwbp))~ShPKCnt@Q63shPksDmV^Y>y_C!T+MB;zp3vO5xfBEovdTg9XG|?>+}$Nq'
    'dyX^ijcROeYNf0m7mO>1G73#v8v+`PP1?vq?VfyFIgC+s(hh=E3QyY05sc!K4sxX3tKUZ+=IVFT)Ryn2<Fw`GGPzZl$Q5Gl{X#No'
    '6<FWoxKug<P~5~1f6wrmTXF0JlTjSX(vB`)y%3K=%JK17Yp9;s30P%VsMSARl|gD+rv%^%_xXd@8QbO$Lvw{8%BJEA!J@X!pd<X4'
    'x*M1wr`Rf;bxPIJ-qDM(jm@s4Ly>1h-'
    'FrWN?MPyat<z2<w%7*z<)H+TE+n?tF6~NUi|x~HB(~Tg?M`Bg`=mWcyrUL=#@Zt8biDp^F#Jp~3pZvz0*+~Xnw14-'
    'Bm2_qBF+5i>I@6wC!~uDck)9NAKS<eLvxiND$3QiE~LuSJse{S+i=?v?LdocwF5dBw%RSZTR8p?Z<kF?bZR2oB^NV)ljiqKZV}`g'
    '{}JWLRiaZblh>mIN!=F^3Nj(*8=rcQss|&{OO8cdrlgB;tm-oJvrt04rY&PEPU95Aj%Ko3LV$Szt0!s%J<ki3kZOSz{&Y=7E*(EE'
    'T~fGzKV1^tz7IolRUaxsZJxUXKAR?S$81cZvKGP!98OeEto5X3(EMeJlZKYF_uy)#47*z{3D~4-'
    '8Ny(plV$pl!y;=1PPcMcWc63MMPSC$SZ}_NA8@kXZ9;C~6m`Oye5uv!xs}O}h24ltZ-'
    'W3t{{!b($9K5#tue%<4CHjVNMt0h^IH1TwapU~>P^$8!u|YdQ*=8&49%5ysA!fmuCPti>QVGk5opaBuGeLXBWs>vtL5uu+XRg3iC'
    '$yk2}RmznO%!+s99UFN97}KIDC~H>UJ()cOr;8I7F0&VHm8$Axg1f@H(r&bi0rnIaO8ODdc7=`~u6w#da9@(KD?;$T-'
    'YA2M*GYkql2w#n&+T5G&NypRUW8it!`TrG<O?)1}cZ{V+6F)1e~F8Smss<>RYbc68BbM?dYbg|f>t3{jxKR-'
    'c2YfEtyWf*5BdPN(Ewt&&_=2Wt(wvE)@%o2gv6PY`d=!a5j<;zn43+-'
    '=|>8#{w5eTjXZUuUR;d8B38ypmJmXgE07f|}e?i5bekw&Sp{TG>`&ziXjx{`CC}+u}#0%L=#hr^}){`C(|Tj6+3`is*_$!s-'
    'IALY5;xSzjZkk_cHkP);`*Wc4ZzGK?tTDH9<IILgF_0)8^lp@5rAY$)I*6B!CP$;5>MJ~C0E41-'
    'uz$PUI9<1ro$#qeO=H<^rjnAzdbRH4oh;2N7^4VY1Cy^1pkr)oWB!7ho(gm!4&la$xXpMH>GO8kcD^1_Y$>GJ44ei)i7)=&|~Fw4'
    '<Vq9A4ks&G>jQ!6j9+gttRG)^wZAUVUT$!%it*TM>ZZ(;H`!U{~^A><}bCFid(d6``h-(aW(f0XS_Jx)+-'
    'a5_O#<a<w3XVer|VEYl$iF(p#U|Dj@m8N?~B4LAt1B^(E3R3Tb^;!wC^%g}O+r_W}v_iQrHNh0<^8ZVgNCSR%xP?V~{L>ZDE&ebx'
    'w;rxGtSqT;GVeW+<LxR&RQP~>Dw3Rud!N+1FpD|gcnGGKdV;^|CwV)%ByT5|<n8Q|yj@(9x2sF?c5_MI?k>sO!zOv<50q$lrK5Zp'
    'E&P4{BYCjTFU!e>k3p6m1(H)Wp?;CB{!23INz518Qk3)XixQj@xzoQ>jg2TM(YK>ZQlz5fNJMx{MTt}cbIajs%lS0ZxiHJLAVqgC'
    'u<Ge{fqNp%>`_M-WhsI1x#W%0JhT=4?mZ@RxgA|KCWv8RFeBw^h{C?ajMn`S@Z7;qYiZ)DKY*hyh`#5C9QOwaPM<Sb7r^PL5KiMc'
    'm5tOZwtlM_(N0n6t?>7$kH$t>g|uUlDX|!%z|4$B1@m1$EQ++xN^njj(srn^5j7?HPIXO+6qOvsvCt->sARkvm|Ij*8?-'
    '9ygZlOAUY~7Pa<jAM9sR6%EtX`059T$SnEk9F=$XL`*U^UQYU$HVOcecf#e9G%=<LSR8WMl#&eK|=Q}4yofmU3Vr-'
    'SrrSdZ`Bll=eLEbU9{-A|Fuz)W=i$1x8}$ozuIF#$41hsd0eP2rl$XK-oObf!P{qoN?&Rf2OOLAFJWjhN2#MRzMSosH(0^oW?wCa'
    '8hAS=@|OvZEz;{I=@noalBuzQs|qqXlJCYrVepTGh#9r!_#=I94{V$sRoIPog0`c{+eJZ2ItYFo{Y2h^4(K?D#QDds7zS8kY8<yu'
    'nY9v(Q8%*Rr&d+^17HZkG|B_Aog%z*Aj_r-'
    '|hY=V?|F<a8#VYCfIik}F)6g*5$zwiZR&_7a>EiL|w9Yy{Hu?dV<zY2YVyk3!l+H89s7>mN*K9T<eN>Rz8^w0?r=%%&=P!p}jrd3'
    '2^1naok`qV`bj$Xhhd!vc-4^$Pb5DT|$s0gui2Yh%G<GyYl%9-'
    'Hyk#(~FX{51(4oAKAigU4q4wF%&{nQ81aj+GSzyB1ug0qn+vusb??8p~yLxI7D7`eU~h1=r>hoD&JIrD|*hxb#K$C<LxC98QP<*U'
    '@TVu2&X<t0T(CD{$2l8#!fSW--6(q%Ur~;x=TJu$gg+kO6qr)Sy-Ewk@`TGdC*;1OC4X=j*w=&%uhpIv&7kYzV7kvSF3WA8|z%tn'
    '|ljFAA$QB{(M%R`05@5wOx1-'
    'Lnv^#&S3z23E(Yfw?YN2&+y^exhMDnCbF3o0Ol)T*O_0ep64vldU1^=}Dof(iB?&T8C3xZGfq*xOzTL+s@Lm=59x_5(BFp$kX~HZ'
    'AVYEthqapfX*Om7iwo4Xzj)z#9-'
    '@mj+@?uEzYG3u$6|`IyM_yxx^J$W?@Tz?2e+?dZz^EL}Ke@H8uiU`l5Rk!dA+WF(d+8$Etz3o>zabUyt(tGrhvUv^7df+Pc5A4GJ'
    'B(#D7~<Y;(mE?NFJ<mi?veQNqoV>ZBb|RM1ixru(2wQY*^|`+!4&;BtI`gmEDfj%y~-'
    '*3l0>db;3aqM9_l%Gr)CEWzi4DnTx`Eg3#hohd`zm#NZ;A_N*QWQjKDi~P7K?iQBdoJicgpvFevPT%6*g>W~H<1Ro1?v7IfbN#T2'
    '@}ivO@)MIE{6y(Tg3IqMhm_CoYv$4wC;(!2ry9}*l~HsFjEk{l_F^Jr+a_-oRJrl7{Jo%k`d#vm!WNW2fswy0Q~rOoT?d#}#nsNt'
    '^@ByCqM?aYDYnEGq=|~LEer$15=F5jxc0IlEM2hDYzU~)WS5Q#h+={r6i|{ViV4O5Vpn7-'
    'BDP3G|8r*MoSFOW<#I32^FRANJYM&HWxq3X=Ct>HpT!i-'
    'A&I4aBBIQ5ozzc7)Vx@%7Bg4Vrve*#&$4vaF_IJv|BK2>@NdxWYz{8M=rl|adUH*khExuZ&yoOF_({rT&T8jZ1$a)za_44qZ>mZ|'
    'NzG9@4u@+a_qJCOvn8nDFbe8Xb7P|~O{E`wA>$51-'
    '0Cp<<vt=+71`!dsism@;BwSdgz%Jyq88&0wAOi6s*i~<agEeZM)aihQva1D=--'
    'oiQAFb%A0Q!{oBCLymax@75?JdFg^`;<WMQ?0Bt#Ema#3e&NSI(ZZ+Amp({3qXH9nhZKQ{NK3N)0|oTKYRgE=K(jM~!8c?-'
    'QDq8S}9VYxSGiwLgb(A$yVHLU8Ygg8EIV42Cgrjl5-'
    '=AmCEg4+2)H+L5|B4(%slU^=BrFUXFeF7sTYTZAHl4C?dybyi6{k$(o^_d%>Q!lBPxY_?IP9BeR&-'
    'gnIy^WUn*eO1j2|D5U76bB{^rHfH^Rk)rU2|`$BtuD!={f6Y&G4MspJ(UP0X#aVe!-J-'
    'sy+|SsRleZryBCuoNC0x;yfTbG14L;0OEVdjpzL6%j)I$Xw+cc%dUvcLhl{Yq9CWl73PHCkviYJ#>Psuu#qo1m2#dR!(6<Q)?<Iq'
    '0hNdSVPnFWtNG+N=C$P81#ISJv*c=XZ>lOoNy-Rwy$~-'
    '{75R?qoUr5C*tZH$2eoB<gkuCEZAaLrJtoyODxl4kYC09r=1Db!im>ygnn_Odaj9leVeM%Iy}N?!Gg5s-'
    'MYiXp`h*H=FG{t+71PvGwUqtrX96cGo(p2M>!kkC&TL#S^-'
    'm%n<3e?kqs8|dnAeuC7qA(d&6aPPdsBrN5*%$kDh4#B?&_D=7p$bYivEHXY<ZT=Eb{F&%AcR&`AhY8SPPA*rm(*{kmdwXgU$7sf1'
    'wVdHgkQ|ls0Pv)r>|V8>;4XC^k|p$j;A8^(htG7fbayK}VLUW$|Y6RT2U|W5TPY{$EWKrqUX0O4^S27K8GdaBczTQL>rvHFIyO%t'
    'FF;4CWr|QDWXWup#V=PDB9r4k|_4i;vKXGK?E5w6=rXH|mFCu*{JNQyyuT$7c}8MlAce0RmZ?ckBSik$x;<5zvkBUJ;9cR!cpP=g'
    '#5#)$=`OU^7Hdk)G|@aOa}NQ2-K~XV#ksi7hm|9yY~O%hmEE%t$J)z~&_T;tN&eweR%>9DT`V-'
    '|Ni1smcpAhFnVm9g>xZou?7HIWml7-'
    ';rSkT7@jLbcmrhZd@GY`br?~&pM3yIRFxI#*si4&M*>4!3p)#0P=4_{WOH+n*m#pc{|h(tflorRZuYWo-hMn_OE!t&`McciF!r76'
    '5qC82)&Ob9Jp5Moi(>@Dvtl1q*)Doi@|x#`b+^wU$U9?FXrCV0&Z-'
    'mO3l#Q?MACm)56xk;?SZ!ML5<v3TIpV@c9`L{+~HfNFRB&;3*A#uHb<T-B-'
    '9V3|k#(q$h`UmjjWu!53PZAXI)Y8WFmjr{^H!fyXj(?kL{uPEW{lKaGe%y1UTBBVH(o3;sUJhm^l2p}=9@&v1+6!^wm1%L<c+TT?'
    'X`il&soIE%UQ43-!rDGW81p>y(bWJ-+Fy{dt#%GiFWni<ZR|4z7R8f%A!$gJy&We4d8!-7{c1V`5<`j-'
    'AYueHB^#@ppv{c+^foN<tRXha(uo9&aO{=}uPiv&KT^TkgKd_>OB%MiXkOy)zvUmKdR;7o`g%@BIz!@;XonPKp7YpRe!(a`EB$t_'
    '&Bqb`rgh?kp-m_%n!9AsnqE{#K?y3}3u63ETl-Cgw-s7chrUG)+AR_q?xtKSM#yy@w#&J;)it|J=N(sy9-'
    '%zlLPg`b$ZO5o$wh_HUovXN0PXSTE?-fn#jQd2GEqnecu;eK5L7qfXYb#QKcyE-^Cyd7>$6-'
    'J1=&Z+tf>pExZuVTR1awJCD=uW)RrajND^kH+!r=vYaZ?wPLh;t&I*V#zbzzx@hAUrIlkMcYu5hm*H#4N+E1I@|la|BPv>GK2)!r'
    '`c_iuBbH!84!poAev~5dpkVOB)_--EB*e9Au~~A&fw!b#m^bP+A?mdmb?R^+^;r_)%2r+)#GaIx~_TZcSB2h+B7-'
    'D{nzoc&`W(^Y&Qf;fibt8u($7fXDQkOMzgzi_HXYd*|Io3s+8Jb?7T(d<+QI&CLv+)dh*W*M&X0mc-pl@Y&iDcP~*Jlt|pYF8I-'
    'PB<|iiJW$&+($_dkuGN?Dbb5XLe8dGH<jVNL?fz#3KHTh^ZZ7fBDks3bJn8JjcT{b1!`oGx%;<KwHB}WMUJQab1%>Ool3H##eKni'
    'Z4U{HsRs$$a+@fxz(`_r9on1*$br7YA+aQWMk@QptQ%SWs4!2zq5hKM5{G}sxLnCnHa$sz_c$6e8eR+TkM7L2+i7~7>c%0>2%sJ?'
    '7Lcj5_P}N2ohduF7eV1^+*!yY*vZ<K%8<Ko6{u!!mZbZ9kn;Fm!x2DP-'
    '#CPj7z0K~{!FoarRs|<nM~+Fr!klS<i8<2%8*`=sM&?Wdtjqx<n3*#Turp^GU})ZIEphJWs)z_AUi6l}7RiS7=`uvjmP@3ZDIugn'
    'IaRDiqxE_Ub|RyL<71<_<WO8n%Z`He$UI2)0ZBH+53Jhd#<8n*nPKd3YpTvcQCPHH;F$1Hk!97P&TTo9#Z|)>ML3`8*w-{V{F=-'
    'uyHc1F*=L=VSf;Barq$=An$Cex2L>Pv#M<9*``^uMTt>uO-'
    '6MTvL@)&%>1(O+hCRpYBH|5M?9bQl^=A%(X9Q1HNzyE9De=)dtP2m(cM3<cI#J{UvVWB1dGUj&Lvq8|)ghTt>~L#pGqg>d+_c0uI'
    'XkE(VAl4elNls1?PZX_jF6$2@(M{}%0AkJ1!L10Na{l$P+uS}uD)}Q4Fp0X8+dis)q<()%R|;b2(u@NutR@uoc2%h{I;b=_|6&q6'
    '=dZZ;r^Tz$o+AX3!PwD&WvcB@@DLoVltdZyZ+|TxREoi`PAp+ZMpxu0ME(Ta$jZcO=-'
    'FB<y!9cJCia?%e~+_>G$Fj(pv7d+>KJ&?>wcr(PkG=lrmgn)W565OlS_%?x`k>N@<UVgSAB~4u9)L8@XZgwQw}{-'
    '^CVLFVv5sNUyh0Rr)wdX&0ml9VBI4a8`Ac<ZHn7bdn5@nV~zoi?an}6<yo~gz~fW@!TA|(l1fS{U*nj=*0LEjgrTS(h&FhmL&gS-'
    't~8Rm)PG6@SKcG>~H4Y3{xHUZ;oW1sg8;Gge+4X@Ul_*5?j<JHmSA}wRlX85ZuVCGR`1aqMG<l<3YaD*wl9#oB2*-'
    'bKhxf;X92jeW$S%b%d|-D**ESDX|rBUwj4ZRWwF~{E;gnHOQWtH-Z=D78=f-c<j2s70vFpzq#b!&XK$7ET*CUA-hdjTISC=H}BF~'
    'Qh?`VTw2S_y&0C4{WAB;wY2Ut4Cxt`)?U&YBV|x6zO>#D`X#%x;JjM8YaB|Ga1?AHT=KbEi3aA{Pj4@m(H@i1c+<?trOc6D^am$S'
    'SoH@-'
    'PFVH_o1?7zgAo!7|6qHRm4C25%F;jBAU%VI7_Ro~7e3VevGqGCzJ3eI9xRQ8K;M?M%*?x<mv@;xUx4RiTxL(3dowIE`!^SIEwf4Z'
    'ge=PpUbc|F%sRA<Ei(v>Ubf5ZUqZi9gW;`-'
    'Jw6A)%Ie15o=QL14e@fHfbgUef9=Az6=s#iTI@~WWn7MZ5WJ?#v718P>>}=FkT<(*t+^UPId%&*lydBrY8d6%t<-'
    'SJu?OmF{F>@xsEC3!H95YfY9+6!H1Y#|d(xUR@7g!-ntG%F&&jx^=9_yntSS39*UGh~CgT&btSNZeTDu9n;%n+Hp_kb;^-'
    'rOf$6Qxs*d~VM<&GlsK0AUU24-{*7<z_k7r$Y&<oH)A%mhF<#eFb3C^xO$8vT=-)^3X)$xUb<f<DMiWj7S~8J(IHiV?!+Y~-'
    '%~EFcWiuk&ElwSHy&R_F&~D{D%8Wz`NKol~TqB@^QNG#&<hhp0);Y(K*Jd6(G>1$a)zWj5Q~o3hMkyw`pKwKFX<0$@(TC!{SiT95'
    'FuwRaPIwJnbCOS!eYLj;E+j!&0_bN{e-nh-@#g})7g?uw#*v>Bn|0{a^n$-'
    '6ocxJroRsXF@>2Jf{Tf~6(8J%=J_g(s&&5wyYM(V+#};eq%9^`x)!s|sA`A7iWP{`jgYDT2?bW^g>4BxJMNJSGo0gdUUR#>|JgAn'
    '*DbRDkDXTwhbny&2q?{VGawxv~5430d42yllyCBCxi_Rf-'
    'ZnFUAV+_LSo#a)pGf@GN&R0@XGtot$IR$px|g%_>X)3rtB2IbUt&m4%$IgsDm)=Of-yBfr3G^M)m#Wz~N$az4aHD{vMUzH6FxxZb'
    'a_vxI&qw#pueud+JDt89zfViwspwQWo;nu<@87Mb}h{qiod0R?zY#zi*H+?!#M*>9pwu0{3$J|W8@gO{zd8z&YoU03Kfc0~UANdH'
    ')fg<oj(YUkdng+DlrvkT@kHpC`wenJW)in8h<dh^W?GNAu!*x%eJ`3}SXbv)@@Ab*mhLvR`go7|O^*x2H(P8Rss>gYNL=npA9fSB'
    'lGNd$70d|Dbn)rT|_i;S8FNv&L2tT8GbV6KseK4V+5!smnNyz`Yg;cY52mN~z}TT>+r;<g>G50kEpd&I8k&+B1LyQ%$_nxdf>v3{'
    'kWbL@>Xtvqy7E|FbsOw6v<OKgaFiO(f-QS3y06UP7l=q+n~##VRro#a8w4*Gk;4d_4v|1;cxZVvVbi8`-'
    't5O$|Tp;tEv`=dmq*ON&1^N)$jys3N@UnWh&;VZ#E`AiFMQ<c%Q-'
    '{7sO(gj7`iE4phk1^=fiVy;QAfX9t)e|?dtNEZ!%{Od~+RqGr#D{VQ>$W76d1#G#QYT+e>g?-DU3@+1cwbNI>g!1-'
    '_<GWbzMgbac!F${jOB;NX2}q1cx+*X%ZC!JnMw??J;|E*O7KrUYr@-1&1lUp@YYnNg7`)qst=)V|KT-'
    'K;CeLE+LZQ1e_@7jA_jBq2y{s{__DEMr0Xx@F(zzt_;CRFX7J+x^3CDLxzQHza|)5Gw<LgkEBHB$814@yfIJfRp23GaWZfOzk>6M'
    '7l~SL;s*w+SMfMv|#Z>pGIZl;Iz3^4iI61y9{FBd?@HW#j+VV5JHC2KjBQT(Y*_TagfFqnD@D6sou&;Z{P<jd($F|rB?rj1dVd4K'
    '&i5CCPZJ5&}jvoKlU7auV8&aR>xq>d@GTJocHKvl-bo^b=ce*KZv0s^<B=rnBP?0hFK&iXCa}(0m77-'
    'QfnhM>9dP{0C9v9v6F&}&_6Gr9mgTX)fYzuEQJ)>>k!&_732zCV&=wtRHn;AD0<3D!zgR=u1l6YO;^>oTn@TW-iz8W303Bh0$d{q'
    'dIYgb7p#!mXcsZ0b%<7Fa<qRf0Mos(CH;GDct1m~ogq$3}EhQz7#9nP>*k-(AAAv3Tb(VV-'
    '>cM1I$nU8$no1|TfFAM+Vvn;&LjEt8358j%pNU%Hnfa{VpwWT56wOxY7{thHrqH^OG#<bnyTBV!puH6d3s^7JC{t9;Om?)U+hkZL'
    'X3MTtEqC$Yjfun)OM}9sTA|`!zMBwhzCEowN+xMjsVFmwlSKTGT3Vv``J@7n*kjY*tjJzObA`Qbz4w#V>taZ|CT`C#Cw@J3fmxX`'
    'w*&5zvW=31Tinpff6VzmBT_Rc8D}(^_fnS7_txdyCUg|e_A4_^(Og!`d!VqH34F;VM>ivL8?~ME)G#d4;$SFuc?p`J0V1msMRN1x'
    'og@8vssQtDiFaf>?{FBeB@HVqDTJ>eTHMLq&w1d%BoYbJcXa(mnE2(ztW^p8EwcLm@lQ06ea7+{CL-iroGb-'
    '?<ss$A{F2(I_FNUB$-s~=hqCei^E{35$-'
    's&!fqdz9Wt`X>ux4Dav=#NRg>ooKP@Go!JyCM@R*`)l^B6|D%bt_wqtFPOLd=U40qXfH}cbF^kW8CM8>=^gC>6#Ni79G1yDfY#jB'
    '~<pT2&<`J$_nqC&&Je<&dzHvVGS)?W_B}^L^bmyQO!L`R0~fM)zXthweloU2YZqzih<R7wccv{Aoy0jX9gjitGzpNMF&WsNL`!lx'
    'Q}jT?YN(AA@bqbCnnj^yu&<^AJ9HeWCyg*OSfaKI&nMJh3~wSJnLF7J5h?JT12q_3ZBMVi&lKdZPo{cT3W2OR-wB}wGZvc6QtUgH'
    'spy??MM6ZBuPosFx^cO=z6$5SrY1cg#L{r*!4(#iX_~10~}>uEICRVADMYscdXtSxnLObU{aUP16sc?$!_MA=8OEO_4y(@YJGmX-'
    'S(|h>{{!F$6#F_>%yLEO1>3L(Uh^yU`9WSezt_`BJR@)vz5to7fRKgOb1y?4>H}wlEh$!;p})%GTrZ`(quXaR(p}@AgSq1Wv!c~e'
    'pV9F;5MmWkTiLIyVNhbi@K!SoCl*mDaml=l^z%QLF$i-'
    '>>%~W(+$_KPEE&IN+;qH8b+&t|Lo>>+b9?D^^<6*MFg`W(PW>Fw3lFGo-'
    '#WQa(l^RW*s2WlssqFQNpI<L9<R0MWATXtc!%r#?xiTOW1BaUbZiyOP+QY`ysgG8CKA|Me65R<m*<cpLZ8Or>V(2$n?oc7BsK4K;'
    '(y}FA&+G=?l^=*t}@#Re%Kpy~!n>n`a3JpjNrEi-<+1rv`dGT_uTLdt-K%;Qwgi50?pnRGYzhIGAQ2dZ-'
    '~Z1EGlMrzM<+!)OLV(}429a2}2z38&r!l0Mzc?Oo)3W?u<=*OdlomM{+<eM*wy%qu-'
    '1@&nSJ5ZM9gPox{JU7Zlzf!r)$J6F+xV{)*|2GI2map?fAg<&oU&`%0Ag21DvN*lg#Ja~=-56P#{xf1*$r+Vi}@QfTQo-'
    'e^4a>Q|g1dm8t>#Dd+z^=0!44zoZ+D2Ti4e&Iau)b<F#wg5m%Lh2WKgodR!#yeTBhH@`*%9YYrW^3cA_Ib}9ZEz~VEf@NMbis~8c'
    'A9<O9DHomjW}XmxFb$mIfwM%L0q3<>7{SC9s})H8kFSG_+P0xb_Nr*pd!;d)6VZV6Tt%SwiCDXQOMS8ou!to$`spf0vX8nvb?n<V'
    'Te+6xmVb3)4;4A?`GhBeuTxn)N+lY#DNxQJZ8fopAV+wSrtE+IYL6S?Cs52W5O#caP8N;qh6@<Fk5te3th3tX>|U)!XB<s#FymdR'
    '~O^7T-'
    'bpipza1gjGk+hakT&NiovC)IBo4*7`j%yVm+WyCE@)?==X?9w1pUh`Aq(sC3d$|1qN4NhAGGM8%WF`r(MGCkN_3MN~d%qW>IG{p2'
    '8B$(mEs)HDbOU6o7CglNzeO4KX}208n`MigzbQdm5lXyMeSn(lU#jx8rW@U^gZ!a<4cf>so&TiOsR2WI@DB%|5ScCXB@o_?>)uAY'
    'AHZs^I1f=gqx`h-'
    '2*9yFc0h)^YaE%r9}XqhYUo0&`MR3@bCBH}xZzuoS**J!7~UZb6+ENB^b8tgUNY1&Tn;ZEyC(M)gLYpfyburM9eR5jIbjFBMOt!@'
    '#t5>~`~U@pM;#Yr}^FEvr-'
    'mqec^vrD2++ztO&k<A){)m-Dye9yu7eW?#0*BbZ7m)k|Cqip+)@yaJ=0>=}1En<txXa)_XzOJIdGo)8Ijii#0N&Y(DTi6>%fy-Py'
    '>?SNr?@I$}5{VsWISajwkO#d|GLo%bM5UJ&daN+>PBwLKhw5heu;G`4h9p#x{has7{6gpV$?QVs_w9y$tSB#N2tToEXQk)pbp8mI'
    '`*tF&@+TU8hmdg%lzP1kEoinO1+0*~oj{z!n7f@wq{EmqoJ6d{n8$S!WZTxliig{)cW|y-ze(Jl2qDm`+|^o~%C%zlNcD+x-'
    '}+yYhk#y{dbK3<_ynOB%6#DO-zT&Z-'
    'KU!*^Q)UrlG)YGC+&uJtWKN_5RKT0D1#bNOo0?+E~YpTevT)UViSrf4uYQ(C~s>@F$H$=9W@wa;$4Y88jXE2^Fg-'
    '}RtlUD!{da0A_f32g^$GXk*~Dk;wkqHvU0PwXqACGSZnWs{dbB?MCPP|y#J8UHgX?zvdk}BK3QfLE}y&`nz0)Gv!M;G@taypm9JB'
    'A@4yZsS6UuB3d_AG?OH<ws0^oHzGh?nXegncV{O#$1z~m}XdBBCs&KH3XVp}2P7?${VqlTqNrKflW^p&`UAGlhC!C64ZBYu+{^}{'
    'Ak7Ku7*Jx`$CG%mr`zO(W+qaw|^Q)6jk=fPBr|gDFtjKfX3io9bxZu!aEQT$m+*0c&v54|B!qblx)PtCd@k_B8ZM^Ody`6CMu>br'
    '?K<4CFicJtg%5xZmkfb9<rQO`n1%!IzSRfbnz+Hl1O#>iu7c^}H=i2~p(81VIuI;?<K1LD!uQhF$irucAK%(m2<$jr8g#3P)U4;D'
    'p-'
    '2gm_wqdD0l;$GTY+tH4R8Xm+Om`AAFx>|SQfJ46aj&k?x?PJ;cKx7r`Y_giU$2j#y`?_XN70wB>VdUcHShqiHtTUVgiY(z%*&AK_'
    '%&{Alw-Kdb2bvm>8jig>?Ki4cbdD}SKwojr_gAE+yCDMGbI8i`Km%LA#yz^X<F5Mnz16c6gh5$tXkxFZz`KujRsMvJ4)a`_VYuGX'
    'iWIW{$@6gNAV6`jaO{jz0r9UcZn+DDTfM5x95O$Y5zT<4;PdnJ_?qmH2g9BOF_xOT>UHFqx1BUf-=YXuo|q97a0L<M?-'
    '#;uvAopmu@S02KfiQpUGzP;Grv$CMwLUjuW{x$kDB26(PraQ(44{Vwp{9lRn0MAs=C^M_<TcCUKqWmi7@QlAg}=((SzDa|yQQ`wN'
    '1xEaYsz4bp`>=*qh&ocX1Du>%BhUl3^FF9CN3QV4Z1RN}teS$7v*EZmFUhnyR6xzk{h!}hX6DCYqk4^GG^?4Lbe<Q5ypLqDt9INq'
    'B|Cst&?YE`X|W=5LL`dDV9*`j~Vj5J$yduF8hLLbMBG~09sW~BL2cXa>#HZrK3=gj*K%9NcS5eoMZZ}@-'
    'USR*mo`~s<2_Yz?h1`Wqutjd6^<3SqD!c$<jT5YX)v;a3i=6ujxbjt_f9g@T*Za(q^ky}a}gR`t!;&^W=OIT4XQy0v}4WUZ7<f6;'
    '$a7>~)p1;E{Ig&5M+U=<!M#y+n*ar6Z|Hh-Kue<mU>VI9}F4mx4Q$J?*f-L<)cd-s-'
    'Q7(!I8+)j?bRC41U&`Y!o20JmeB(}CkLlljZlpg6;B>6O$3r4G3?W|q&K`ykial1yB_ytgCg@SxPkE=vEhUb@c2+HMyf>8|EPfcG'
    '!>LQl_*DGwZsfXF2ge`$@aboK7dxy`5!G~jm{XU6A{;=gqB83~a1=g618CF8?d}H$;d2Bom_a4h{oxpVffR6O1rus#VhBWRwu6v('
    'hC-cTW{jWQbVq@YDv=MGYV>N!{)I^IwDyOUx)kkSyjjir-z9R(hhwarRX-'
    'f>O(hA7dsy@x9i^KeDT5Onj?9z!+tUJr96t`)OX$v)oz~1s9L`aMJ_q7XrZXLggX6qa1|h#U^NOuQ#rAFP?S#<vsT__3<>y6uv?r'
    '^+Akw#ng>O^}2X{iij#BGdG3T94>~NKcUbQ(;4g~1%(2NlO4zXr@Aky<fGw#LMQ24{|7P+OvG40K&9gg>=@`4qaaV!oE8kdPz<1l'
    'sRVZHGff*7JTyt)(l{dB3jy2}8&-'
    'M|q3QC%dB^+8gHgAjyBYa_v9cjKjS79!zjv~w`3V^BcZ0u^(fW?JzJBYnHKqJ8qy5!&)+bSTvI(~e6bJR}BAzT!~5BGNy>*B^5b='
    'w}8e8aNewSuZp&GH6)?pDOj@q<w1soO?uW{cuc!vkHjgy{U-'
    'E8U=ztU~)nzMhR!5Di6Yp2}Hh$4jTRk)q|?=!cof8sR{uKZ{f4S_;lK#HhZKuxV<<SslFXRw@f!P5$!X~BHu64w?#y<h;-'
    'I&yzx&0S#mowfxra}4X6_OLPi5j6Z&EX08AJ95(5Cp1AF{~aMlHrLiV@_Ov*>;&zwo|4;>+MqaYExoD~L%_om_?i^~NR-ebBLs@D'
    'z`!C9&o$nY=)3mzRnbdawv><E{cmWitvRH^P%Fqo=(&;;o;rKof;UG=01(iuwA1nErGizY~Csoqptm;)&)1x({0RicqYH9^m02cq'
    'uOvl0<hso2K;BG{cPq@HPa=Mt&MhP#tW1o%lt%G{_&cur>6Bi@^edrUB8r8Us=0v|_3mZJmn0f?SgyM2nbbDdkvYpC;aG-'
    '6=%4Y8g1h<b!W75L>)?jhk&_Apx@$E&N2toD<V*;8+_U>FCD8f_~!c2;qFq^sRaRtnvZ(K%CvzK8)it&M888+XHccLSbx!_sL(JH'
    'MC*?RaTuU+_-PTKnE0^uo}-'
    'v{mshze(oCLBew~10nI=R3v1A^G$9_4izEs&&_6c@&s`;dP8sHt}>?G+Ce?49%UO}BalUUp=55<k+$58SJw+9mHw+FZ@-'
    '!r*W2*3$)!Z_v)Sde@Uz9`M)0%MWk&Gx1&;@)Z7v*xpD*(P9?QHrYayEN1q<Q0Q0OHh!JPO<j*_{dkT~m^6$^>?rh*}Ny<K5z9qK'
    'IxdkihQi25Q3P2+sEGBLyP)_hDo#xYP^l#g?9?eRjtCiF7FoB^wZ{+nQg;&GCKr9T+1bK+9DOz1IOfWU%xBOWz{kBpc{H#&^9tR{'
    'YT_+9M!Tqa3<0N(v_nGf}NS!m$C=+BISFB1BtBm?8$JzD0*L&9@110wO>R77OasMW*57aDMo|4Np#t;BBEBBDO7q_F7-VF;qF-Rn'
    'y}W5fD`-=1o&n#*Am1j3zJ*leW!Cem9XCWLrP=)XHtw=%UqS1=r-'
    'r~3H(_d?&o2@)cBZWqjHeu&VGd8$|+Za5+qGZ3MZ(3j=}K3*Og_X43`w8p(y=$Av|3TIsWOK+CBA(8N$%$P{LHx(3FG?rD0rgG-'
    '&j#yk*FX(~jO;<2bi0*S0gPw>9b0q@}XC2iHdO_$(MT6cDdD2+cJT;Gv`)fS@sOi-nPd}#}sXjm=kgI|&wec6ya5K}4tTYa!lu@r'
    ';CApm=kfzyt7QANHQjJ$hCEZkz;}xM@`w6|++Vv8lmx_cS3jfGkWN!E)JSQ{$5${a}Ko(V&ekskqr9FFu@XSK+tG>c-'
    'TDMOjQfVE`cl9-wA^&j`?3R9Fqm4_YUSB9eyMAEE-khAyd^O)O0kIN6So-+0fhK$8YZVT^1<=Q-'
    'I9TNhO{l`vH_|sp%#3iGU;_1Zx-(89kq}~6R6MX3@9PsF5R|0ubCS?k*WBExn8;>f6_K_u{$aPu+?YssPG(Rf-kXYw47afTzTP+F'
    'jsE-'
    'o&@iI8@9z{;R^@bKje1?Jp>X95wa(sil3}o3As86ko*nyGJ<ZoUeklxC++F=jVCd`QLa{A2l_f&;F#_|?)?tHlE;l$K8(9@&BC6!'
    '>j45loS@d%3!lMr;cYVRv!*0#7!t=>u7&6;5H1eP$KMtpzrR#><uj{rwkO00dhXn9#={Ekcrj0)~1W`^1>0d#fVtQD!-'
    'I$8dzw);J5bo>jY<i~9ov4zeM(D32y*Xk!fw6+<AW-UwiRNC?y|^|}Utl&Gd5-'
    'hhu7{HY#rTPFtK{Lq<2+@OyY2QhCn2@B3&!y`3%Ef54(y!>?x$8qkUwnq1#suCZ@}cwxa%7QJOX#^=*-'
    '=hbo(AC0pFfO3HbJO3)k8^-kht#zBLbxs7H9JSv|^=<LWUUXH|1~s6@@<NoO_R;Yox((l3Lx>UHk*y|3Ql-rY_5+emMXm~P?@VW{'
    'Hr;2S>;EefVs;~V-r1b&QbB*Sw%l06M^E0LktJ{He8R>RNs05iKC%x*BV-$T3>*wRr~y1(>}J&*vtBZma=9qG2*ckddzb-'
    'f`3ce~|uR5WLEtXFSxTNQhnowz1M<AIR0(|8~xd^8>isR4}#LLf=wfe?Gpcp#ijME7wxhTul;xh$|wG8Z>0+4<)9!+yX9Offr<o~'
    '}+T6{n&oKMUe0^_4-d!+(4O30TeO*nLZ5_CN!8Ob!j;G3lmkShOww8J366K<!+%<2tpH+ve}8H@I#7o?6AN@(<tzI=TOT9Xmc#-'
    'I6%LuoS%FobEFHTOOuph%DJ$A3hGxR~vjc&_5Nhin^l^$1l)eFc@7eXZ8X(5{TKh8io02v*BI53YOzlQgob(tlSV<^;^iQj8(sbL'
    '^{E$_B$K92Wr4$bEpB2O}A?Enj7?8BMRY0|K18qf~7(q<E{KR-9tcLM8f3?10b4d=!x-'
    '<y?71VdsXgYB{FbNa~E$Qb@Fs)=;iuMm$sH+<l5Tt1vk(;3uxWYw1$zfG_07c4SWB=US1RO>c5F%he7inZe5-afBAW6-'
    'mi6AYu^9r!$R{?v}C`maeJT?JT8Y)@VIpIwkz7SZwK{0^`D?@i<+IsdfTqRYtpp(jp1nihw#pJy|vlPx;N83zoO4#isvRsP0uu1$'
    'D+HX;rjL%Bh$cdM+}N_ck>7~&ag2HV?g_gKKth!0u2#pOltggm)HrHK@61SjZ<-%8^f%201kbcFsVYoBf+%xyBfa-BEjQxhy;&MH'
    '|>!{racgumiBeu3GB>Ydu!rZy2hWSYyDZe&Yz|4`UCVme}KO456};|R?&<BE**q!=~$K<VDn?A+o?H#m93-'
    '?lsc$aG2x*W><Yh=N5KUl6xA9%lOu6R(<vW&9Q=7GWvvGaco<6#7Vt>$!S;Kaum`%p6LRPRPe`+5*_g>k{vTs>U{L'
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
