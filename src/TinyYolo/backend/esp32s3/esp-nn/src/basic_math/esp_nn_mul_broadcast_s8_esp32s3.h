// Generated from esp_nn_mul_broadcast_s8_esp32s3.S (preprocessed assembly, header-only build).
#pragma once
__asm__(".pushsection .text\n"
R"ASM(
       
.macro MUL_S8_CONST_PTR r
    addi \r, a1, 63
    srli \r, \r, 4
    slli \r, \r, 4
.endm
.macro MUL_S8_SETUP rs, t0, t1
    movi.n \t0, 1
    ssl \rs
    sll \t0, \t0
    srli \t0, \t0, 1
    slli \t1, \t0, 14
    addmi \t1, \t1, 8192
    s32i \t1, a1, 44
    ee.zero.q q0
    addi \t0, a1, 24
    ee.vldbc.16 q1, \t0
    addi \t0, a1, 28
    ee.vldbc.16 q7, \t0
    MUL_S8_CONST_PTR \t1
    addi \t0, a1, 40
    ee.vldbc.16 q2, \t0
    ee.vst.128.ip q2, \t1, 16
    movi \t0, MUL_S8_FRAME__esp_nn_mul_broadcast_s8_esp32s3 + 12
    add \t0, \t0, a1
    ee.vldbc.16 q2, \t0
    ee.vst.128.ip q2, \t1, 16
    addi \t0, \t0, -4
    ee.vldbc.16 q2, \t0
    ee.vst.128.ip q2, \t1, 0
    addi \t0, a1, 44
    ee.vldbc.32 q6, \t0
.endm
.macro MUL_S8_STEP8 p1, p2, po, rs14, mult, t0, t1, t2, t3, t4, t5, t6, t7, rs0=0, cp
    ee.vld.l.64.ip q2, \p1, 8
    ee.vld.h.64.ip q2, \p2, 8
    ssai 16
    ee.vcmp.lt.s8 q5, q2, q0
    ee.vzip.8 q2, q5
    ee.vadds.s16 q4, q2, q1
    ee.vadds.s16 q5, q5, q7
    ee.vmul.s16 q3, q4, q5
    ssai 0
    ee.vmul.s16 q2, q4, q5
    ssai 15
    ee.vzip.16 q2, q3
    ee.vsl.32 q2, q2
    ee.vsl.32 q3, q3
    ee.movi.32.a q2, \t0, 0
    ee.movi.32.a q2, \t1, 1
    ee.movi.32.a q2, \t2, 2
    ee.movi.32.a q2, \t3, 3
    ee.movi.32.a q3, \t4, 0
    ee.movi.32.a q3, \t5, 1
    ee.movi.32.a q3, \t6, 2
    ee.movi.32.a q3, \t7, 3
    mulsh \t0, \mult, \t0
    mulsh \t1, \mult, \t1
    mulsh \t2, \mult, \t2
    mulsh \t3, \mult, \t3
    mulsh \t4, \mult, \t4
    mulsh \t5, \mult, \t5
    mulsh \t6, \mult, \t6
    mulsh \t7, \mult, \t7
.if \rs0 == 0
    ee.vcmp.lt.s32 q4, q2, q0
    ee.vcmp.lt.s32 q5, q3, q0
    ssai 14
    ee.vsl.32 q4, q4
    ee.vsl.32 q5, q5
    ee.vadds.s32 q4, q4, q6
    ee.vadds.s32 q5, q5, q6
.endif
    ee.movi.32.q q2, \t0, 0
    ee.movi.32.q q2, \t1, 1
    ee.movi.32.q q2, \t2, 2
    ee.movi.32.q q2, \t3, 3
    ee.movi.32.q q3, \t4, 0
    ee.movi.32.q q3, \t5, 1
    ee.movi.32.q q3, \t6, 2
    ee.movi.32.q q3, \t7, 3
.if \rs0 == 0
    ee.vadds.s32 q2, q2, q4
    ee.vadds.s32 q3, q3, q5
.else
    ee.vadds.s32 q2, q2, q6
    ee.vadds.s32 q3, q3, q6
    ee.vadds.s32 q2, q2, q2
    ee.vadds.s32 q3, q3, q3
    ee.vadds.s32 q2, q2, q2
    ee.vadds.s32 q3, q3, q3
.endif
    ee.vld.128.ip q5, \cp, 16
    wsr.sar \rs14
    ee.vsr.32 q2, q2
    ee.vld.128.ip q4, \cp, 16
    ee.vsr.32 q3, q3
    ee.vunzip.16 q2, q3
    ee.vadds.s16 q2, q2, q5
    ee.vld.128.ip q5, \cp, -32
    ee.vmin.s16 q2, q2, q4
    ee.vmax.s16 q2, q2, q5
    ee.vunzip.8 q2, q3
    ee.vst.l.64.ip q2, \po, 8
.endm
.macro MUL_S8_PUT16 v, cp, a
    s16i \v, a1, 44
    ee.vldbc.16 q2, \a
    ee.vst.128.ip q2, \cp, 16
.endm
.macro MUL_S8_SETUP_Q rs, mult, cp, t0, t1, ls=
    MUL_S8_CONST_PTR \cp
    ee.vld.128.ip q3, \cp, 16
    ee.vld.128.ip q4, \cp, 16
    ee.vld.128.ip q5, \cp, -32
    ee.vst.128.ip q5, \cp, 16
    ee.vst.128.ip q1, \cp, 16
    addi \t0, a1, 44
    movi \t1, -32768
    MUL_S8_PUT16 \t1, \cp, \t0
.ifb \ls
    extui \t1, \mult, 0, 14
.else
    ssl \ls
    sll \t1, \mult
    extui \t1, \t1, 0, 14
.endif
    slli \t1, \t1, 1
    MUL_S8_PUT16 \t1, \cp, \t0
    srli \t1, \t1, 1
    MUL_S8_PUT16 \t1, \cp, \t0
.ifb \ls
    extui \t1, \mult, 14, 14
.else
    sll \t1, \mult
    extui \t1, \t1, 14, 14
.endif
    slli \t1, \t1, 1
    MUL_S8_PUT16 \t1, \cp, \t0
    srli \t1, \t1, 1
    MUL_S8_PUT16 \t1, \cp, \t0
.ifb \ls
    extui \t1, \mult, 28, 3
.else
    movi.n \t1, 28
    sub \t1, \t1, \ls
    ssr \t1
    srl \t1, \mult
.endif
    slli \t1, \t1, 1
    MUL_S8_PUT16 \t1, \cp, \t0
    srli \t1, \t1, 1
    MUL_S8_PUT16 \t1, \cp, \t0
.ifb \ls
    addi \t1, \rs, -1
    ssl \t1
    movi.n \t1, 1
    sll \t1, \t1
    MUL_S8_PUT16 \t1, \cp, \t0
    movi.n \t1, 8
    MUL_S8_PUT16 \t1, \cp, \t0
.else
    movi.n \t1, 0
    MUL_S8_PUT16 \t1, \cp, \t0
    MUL_S8_PUT16 \t1, \cp, \t0
.endif
    ee.vst.128.ip q3, \cp, 16
    ee.vst.128.ip q4, \cp, -160
    movi.n \t1, 1
    s16i \t1, a1, 44
    ee.vldbc.16 q6, \t0
    addi \t0, a1, 24
    ee.vldbc.16 q4, \t0
    ssai 1
.endm
.macro MUL_S8_STEP8_Q p1, p2, po, s14, srs, cp, step=, t=
.ifb \step
    ee.vld.l.64.ip q2, \p1, 8
    ee.vld.h.64.ip q2, \p2, 8
    ee.zero.qacc
    ee.vcmp.lt.s8 q3, q2, q0
    ee.vzip.8 q2, q3
.else
    ee.ld.128.usar.xp q2, \p1, \step
    addi \t, \p1, -1
    ee.vld.128.ip q3, \t, 0
    ee.zero.qacc
    ee.src.q q2, q2, q3
    ee.ld.128.usar.xp q3, \p2, \step
    addi \t, \p2, -1
    ee.vld.128.ip q5, \t, 0
    ee.vcmp.lt.s8 q1, q2, q0
    ee.src.q q3, q3, q5
    ee.vzip.8 q2, q1
    ee.vcmp.lt.s8 q5, q3, q0
    ee.vzip.8 q3, q5
.endif
    ee.vadds.s16.ld.incp q1, \cp, q4, q2, q4
    ee.vadds.s16.ld.incp q5, \cp, q3, q3, q7
    ee.vmul.s16 q2, q4, q3
    ee.andq q4, q4, q3
    ee.vmulas.s16.qacc.ld.ip q3, \cp, 16, q1, q1
    ee.andq q4, q4, q6
    ee.vmulas.s16.qacc.ld.ip q1, \cp, 16, q2, q5
    ee.vmulas.s16.qacc.ld.ip q5, \cp, 16, q4, q3
    ee.srcmb.s16.qacc q3, \s14, 0
    ee.vmulas.s16.qacc.ld.ip q3, \cp, 16, q2, q1
    ee.vmulas.s16.qacc.ld.ip q1, \cp, 16, q4, q5
    ee.srcmb.s16.qacc q5, \s14, 0
    ee.vmulas.s16.qacc.ld.ip q5, \cp, 16, q2, q3
    ee.vcmp.lt.s16 q2, q2, q0
    ee.vadds.s16.ld.incp q3, \cp, q2, q2, q5
    ee.vmulas.s16.qacc.ld.ip q5, \cp, 16, q4, q1
    ee.vmulas.s16.qacc.ld.ip q1, \cp, -192, q2, q3
    ee.srcmb.s16.qacc q2, \srs, 0
    ee.vadds.s16.ld.incp q3, \cp, q2, q2, q5
    ee.vmin.s16.ld.incp q4, \cp, q2, q2, q1
    ee.vmax.s16 q2, q2, q3
    ee.vunzip.8 q2, q5
    ee.vst.l.64.ip q2, \po, 8
.endm
    .set MUL_S8_FRAME__esp_nn_mul_broadcast_s8_esp32s3, 336
    .set ARG_OUT_MULT__esp_nn_mul_broadcast_s8_esp32s3, MUL_S8_FRAME__esp_nn_mul_broadcast_s8_esp32s3 + 0
    .set ARG_OUT_SHIFT__esp_nn_mul_broadcast_s8_esp32s3, MUL_S8_FRAME__esp_nn_mul_broadcast_s8_esp32s3 + 4
    .set ARG_ACT_MIN__esp_nn_mul_broadcast_s8_esp32s3, MUL_S8_FRAME__esp_nn_mul_broadcast_s8_esp32s3 + 8
    .set ARG_ACT_MAX__esp_nn_mul_broadcast_s8_esp32s3, MUL_S8_FRAME__esp_nn_mul_broadcast_s8_esp32s3 + 12
    .set ARG_SPATIAL__esp_nn_mul_broadcast_s8_esp32s3, MUL_S8_FRAME__esp_nn_mul_broadcast_s8_esp32s3 + 16
    .set ARG_CHANNELS__esp_nn_mul_broadcast_s8_esp32s3, MUL_S8_FRAME__esp_nn_mul_broadcast_s8_esp32s3 + 20
    .text
    .align 4
    .literal_position
    .literal .LC_nudge__esp_nn_mul_broadcast_s8_esp32s3, 1073741824
    .type esp_nn_mul_broadcast_channel_s8_esp32s3, @function
    .align 4
    .weak esp_nn_mul_broadcast_channel_s8_esp32s3
esp_nn_mul_broadcast_channel_s8_esp32s3:
    entry a1, MUL_S8_FRAME__esp_nn_mul_broadcast_s8_esp32s3
    s32i.n a3, a1, 4
    s32i.n a6, a1, 8
    s32i.n a2, a1, 16
    s32i.n a4, a1, 24
    s32i.n a5, a1, 28
    s32i a7, a1, 40
    movi.n a5, 0
    l32i a8, a1, ARG_SPATIAL__esp_nn_mul_broadcast_s8_esp32s3
    l32i a9, a1, ARG_CHANNELS__esp_nn_mul_broadcast_s8_esp32s3
    s32i.n a9, a1, 12
    blti a8, 1, .Lexit__esp_nn_mul_broadcast_s8_esp32s3
    blti a9, 1, .Lexit__esp_nn_mul_broadcast_s8_esp32s3
    l32i a15, a1, ARG_OUT_SHIFT__esp_nn_mul_broadcast_s8_esp32s3
    movi.n a11, 0
    max a14, a15, a11
    sub a12, a14, a15
    s32i.n a12, a1, 20
    l32i a13, a1, ARG_OUT_MULT__esp_nn_mul_broadcast_s8_esp32s3
    s32i a11, a1, 36
    blti a9, 8, .Lsetup_done__esp_nn_mul_broadcast_s8_esp32s3
    bgei a15, 1, .Llshift__esp_nn_mul_broadcast_s8_esp32s3
    movi.n a8, 16
    bge a8, a12, 1f
    movi.n a12, 15
    movi.n a13, 0
1:
    MUL_S8_SETUP a12, a8, a10
    addi a8, a12, 14
    movi.n a10, -1
    moveqz a8, a10, a12
    s32i a8, a1, 36
.Lsetup_done__esp_nn_mul_broadcast_s8_esp32s3:
    l32r a4, .LC_nudge__esp_nn_mul_broadcast_s8_esp32s3
    movi.n a10, 0
    s32i a10, a1, 32
    l32i a12, a1, 36
    blti a12, 1, .Lspatial_loop__esp_nn_mul_broadcast_s8_esp32s3
    or a10, a2, a3
    or a10, a10, a6
    or a10, a10, a9
    extui a10, a10, 0, 3
    addi a11, a12, -15
    movi.n a8, 15
    bgeu a11, a8, .Lspatial_loop__esp_nn_mul_broadcast_s8_esp32s3
    addi a12, a12, -14
    MUL_S8_SETUP_Q a12, a13, a14, a8, a11
    addi a7, a12, 3
    bnez a10, .Lrows_q__esp_nn_mul_broadcast_s8_esp32s3
.Lrows_back_to_back__esp_nn_mul_broadcast_s8_esp32s3:
    l32i a10, a1, ARG_SPATIAL__esp_nn_mul_broadcast_s8_esp32s3
    srli a12, a9, 3
    mov.n a11, a3
    movi.n a5, 14
.Lchannel_q__esp_nn_mul_broadcast_s8_esp32s3:
    mov.n a3, a11
    loopgtz a12, .Lchannel_q_end__esp_nn_mul_broadcast_s8_esp32s3
    MUL_S8_STEP8_Q a2, a3, a6, a5, a7, a14
.Lchannel_q_end__esp_nn_mul_broadcast_s8_esp32s3:
    addi.n a10, a10, -1
    bnez a10, .Lchannel_q__esp_nn_mul_broadcast_s8_esp32s3
    j .Lexit__esp_nn_mul_broadcast_s8_esp32s3
.Llshift__esp_nn_mul_broadcast_s8_esp32s3:
    bgei a15, 12, .Lsetup_done__esp_nn_mul_broadcast_s8_esp32s3
    or a10, a2, a3
    or a10, a10, a6
    or a10, a10, a9
    extui a10, a10, 0, 3
    bnez a10, .Lsetup_done__esp_nn_mul_broadcast_s8_esp32s3
    movi.n a12, 0
    MUL_S8_SETUP a12, a8, a10
    MUL_S8_SETUP_Q a12, a13, a14, a8, a11, a15
    movi.n a7, 3
    j .Lrows_back_to_back__esp_nn_mul_broadcast_s8_esp32s3
.Lrows_q__esp_nn_mul_broadcast_s8_esp32s3:
    s32i.n a7, a1, 0
    movi.n a12, 1
    s32i a12, a1, 36
    movi.n a14, 0
.Lspatial_loop__esp_nn_mul_broadcast_s8_esp32s3:
    l32i a8, a1, ARG_SPATIAL__esp_nn_mul_broadcast_s8_esp32s3
    l32i a10, a1, 32
    bge a10, a8, .Lexit__esp_nn_mul_broadcast_s8_esp32s3
    l32i.n a3, a1, 4
    l32i.n a9, a1, 12
    movi.n a11, 0
    mov.n a15, a9
    l32i a12, a1, 36
    beqz a12, .Lchannel_leftover__esp_nn_mul_broadcast_s8_esp32s3
    srli a8, a9, 3
    or a10, a2, a3
    or a10, a10, a6
    extui a10, a10, 0, 3
    bltz a12, .Lchannel_rs0__esp_nn_mul_broadcast_s8_esp32s3
    beqi a12, 1, .Lrow_q__esp_nn_mul_broadcast_s8_esp32s3
    bnez a10, .Lchannel_leftover__esp_nn_mul_broadcast_s8_esp32s3
.Lchannel_simd__esp_nn_mul_broadcast_s8_esp32s3:
    MUL_S8_CONST_PTR a14
    loopgtz a8, .Lchannel_simd_end__esp_nn_mul_broadcast_s8_esp32s3
    MUL_S8_STEP8 a2, a3, a6, a12, a13, a4, a5, a7, a8, a9, a10, a11, a15, cp=a14
.Lchannel_simd_end__esp_nn_mul_broadcast_s8_esp32s3:
    j .Lchannel_vec_done__esp_nn_mul_broadcast_s8_esp32s3
.Lchannel_rs0__esp_nn_mul_broadcast_s8_esp32s3:
    bnez a10, .Lchannel_leftover__esp_nn_mul_broadcast_s8_esp32s3
    movi.n a12, 16
    MUL_S8_CONST_PTR a14
    loopgtz a8, .Lchannel_rs0_end__esp_nn_mul_broadcast_s8_esp32s3
    MUL_S8_STEP8 a2, a3, a6, a12, a13, a4, a5, a7, a8, a9, a10, a11, a15, 1, a14
.Lchannel_rs0_end__esp_nn_mul_broadcast_s8_esp32s3:
    j .Lchannel_vec_done__esp_nn_mul_broadcast_s8_esp32s3
.Lrow_q__esp_nn_mul_broadcast_s8_esp32s3:
    neg a15, a6
    extui a15, a15, 0, 3
    movi.n a5, 1
    j .Lchannel_leftover__esp_nn_mul_broadcast_s8_esp32s3
.Lrow_q_body__esp_nn_mul_broadcast_s8_esp32s3:
    l32i.n a15, a1, 12
    sub a8, a15, a11
    srli a8, a8, 3
    MUL_S8_CONST_PTR a14
    addi a14, a14, 32
    movi.n a5, 14
    l32i.n a7, a1, 0
    movi.n a12, 8
    ssai 1
    loopgtz a8, .Lrow_q_end__esp_nn_mul_broadcast_s8_esp32s3
    MUL_S8_STEP8_Q a2, a3, a6, a5, a7, a14, a12, a10
.Lrow_q_end__esp_nn_mul_broadcast_s8_esp32s3:
    slli a8, a8, 3
    add.n a11, a11, a8
    movi.n a5, 0
    movi.n a14, 0
    j .Lchannel_leftover__esp_nn_mul_broadcast_s8_esp32s3
.Lchannel_vec_done__esp_nn_mul_broadcast_s8_esp32s3:
    movi.n a5, 0
    movi.n a14, 0
    l32r a4, .LC_nudge__esp_nn_mul_broadcast_s8_esp32s3
    l32i.n a15, a1, 12
    movi.n a11, -8
    and a11, a15, a11
.Lchannel_leftover__esp_nn_mul_broadcast_s8_esp32s3:
    bge a11, a15, .Lleftover_done__esp_nn_mul_broadcast_s8_esp32s3
    ssl a14
    l32i.n a8, a1, 24
    l8ui a10, a2, 0
    sext a10, a10, 7
    add.n a10, a10, a8
    l32i.n a8, a1, 28
    l8ui a12, a3, 0
    sext a12, a12, 7
    add.n a12, a12, a8
    mull a10, a10, a12
    sll a10, a10
    l32i.n a9, a1, 20
    mulsh a8, a10, a13
    mull a12, a10, a13
    ssai 31
    add.n a12, a4, a12
    saltu a10, a12, a4
    add.n a10, a10, a8
    src a10, a10, a12
    blti a9, 1, .Lskip_div_scalar__esp_nn_mul_broadcast_s8_esp32s3
    addi a8, a9, -1
    ssl a8
    movi a7, 1
    sll a7, a7
    extui a8, a10, 31, 1
    sub a10, a10, a8
    add a10, a10, a7
    ssr a9
    sra a10, a10
.Lskip_div_scalar__esp_nn_mul_broadcast_s8_esp32s3:
    l32i a8, a1, 40
    l32i a7, a1, ARG_ACT_MIN__esp_nn_mul_broadcast_s8_esp32s3
    l32i a12, a1, ARG_ACT_MAX__esp_nn_mul_broadcast_s8_esp32s3
    add.n a10, a10, a8
    min a10, a10, a12
    max a10, a10, a7
    s8i a10, a6, 0
    addi a2, a2, 1
    addi a3, a3, 1
    addi a6, a6, 1
    addi a11, a11, 1
    j .Lchannel_leftover__esp_nn_mul_broadcast_s8_esp32s3
.Lleftover_done__esp_nn_mul_broadcast_s8_esp32s3:
    bnez a5, .Lrow_q_body__esp_nn_mul_broadcast_s8_esp32s3
.Lspatial_next__esp_nn_mul_broadcast_s8_esp32s3:
    l32i a10, a1, 32
    addi a10, a10, 1
    s32i a10, a1, 32
    j .Lspatial_loop__esp_nn_mul_broadcast_s8_esp32s3
.Lexit__esp_nn_mul_broadcast_s8_esp32s3:
    retw.n
    .size esp_nn_mul_broadcast_channel_s8_esp32s3, . - esp_nn_mul_broadcast_channel_s8_esp32s3
.purgem MUL_S8_CONST_PTR
.purgem MUL_S8_SETUP
.purgem MUL_S8_STEP8
.purgem MUL_S8_PUT16
.purgem MUL_S8_SETUP_Q
.purgem MUL_S8_STEP8_Q
)ASM"
".popsection\n");
