// Generated from esp_nn_depthwise_conv_s8_kx1_esp32s3.S (preprocessed assembly, header-only build).
#pragma once
__asm__(".pushsection .text\n"
R"ASM(
       
    .text
    .literal_position
    .type esp_nn_dw_s8_kx1_esp32s3, @function
    .align 4
    .weak esp_nn_dw_s8_kx1_esp32s3
esp_nn_dw_s8_kx1_esp32s3:
    entry a1, 160
    ee.zero.qacc
    mov.n a8, a2
    mov.n a9, a5
    loopgtz a4, .Lcl_mac_done__esp_nn_depthwise_conv_s8_kx1_esp32s3
    ee.vld.128.xp q0, a8, a3
    ee.vld.128.xp q1, a9, a3
    ee.vmulas.s8.qacc q0, q1
.Lcl_mac_done__esp_nn_depthwise_conv_s8_kx1_esp32s3:
    mov a10, a1
    ee.st.qacc_l.l.128.ip a10, 16
    ee.st.qacc_l.h.32.ip a10, -16
    addi a10, a1, 32
    ee.st.qacc_h.l.128.ip a10, 16
    ee.st.qacc_h.h.32.ip a10, -16
    l32i.n a12, a1, 0
    l32i.n a13, a1, 4
    sext a8, a12, 19
    extui a12, a12, 20, 12
    slli a14, a13, 24
    srai a14, a14, 12
    or a14, a14, a12
    slli a15, a13, 4
    srai a15, a15, 12
    ee.movi.32.q q0, a8, 0
    ee.movi.32.q q0, a14, 1
    ee.movi.32.q q0, a15, 2
    l32i.n a12, a1, 8
    extui a13, a13, 28, 4
    slli a8, a12, 16
    srai a8, a8, 12
    or a8, a8, a13
    ee.movi.32.q q0, a8, 3
    l32i.n a13, a1, 12
    extui a14, a12, 16, 16
    slli a8, a13, 28
    srai a8, a8, 12
    or a8, a8, a14
    slli a15, a13, 8
    srai a15, a15, 12
    l32i.n a12, a1, 16
    ee.movi.32.q q3, a8, 0
    ee.movi.32.q q3, a15, 1
    extui a13, a13, 24, 8
    slli a8, a12, 20
    srai a8, a8, 12
    or a8, a8, a13
    srai a15, a12, 12
    ee.movi.32.q q3, a8, 2
    ee.movi.32.q q3, a15, 3
    l32i.n a12, a1, 32
    l32i.n a13, a1, 36
    sext a8, a12, 19
    extui a12, a12, 20, 12
    slli a14, a13, 24
    srai a14, a14, 12
    or a14, a14, a12
    slli a15, a13, 4
    srai a15, a15, 12
    ee.movi.32.q q1, a8, 0
    ee.movi.32.q q1, a14, 1
    ee.movi.32.q q1, a15, 2
    l32i.n a12, a1, 40
    extui a13, a13, 28, 4
    slli a8, a12, 16
    srai a8, a8, 12
    or a8, a8, a13
    ee.movi.32.q q1, a8, 3
    l32i.n a13, a1, 44
    extui a14, a12, 16, 16
    slli a8, a13, 28
    srai a8, a8, 12
    or a8, a8, a14
    slli a15, a13, 8
    srai a15, a15, 12
    l32i.n a12, a1, 48
    ee.movi.32.q q2, a8, 0
    ee.movi.32.q q2, a15, 1
    extui a13, a13, 24, 8
    slli a8, a12, 20
    srai a8, a8, 12
    or a8, a8, a13
    srai a15, a12, 12
    ee.movi.32.q q2, a8, 2
    ee.movi.32.q q2, a15, 3
    ld.qr q4, a6, 0
    ee.vadds.s32 q0, q0, q4
    ld.qr q4, a6, 16
    ee.vadds.s32 q3, q3, q4
    ld.qr q4, a6, 32
    ee.vadds.s32 q1, q1, q4
    ld.qr q4, a6, 48
    ee.vadds.s32 q2, q2, q4
    st.qr q0, a1, 64
    st.qr q3, a1, 80
    st.qr q1, a1, 96
    st.qr q2, a1, 112
    s32i a7, a1, 56
    l32i a6, a7, 4
    l32i a7, a7, 8
    mov.n a10, a6
    mov.n a11, a7
    call8 esp_nn_multiply_by_quantized_mult_ver1_esp32s3
    st.qr q0, a1, 64
    ld.qr q0, a1, 80
    addi a10, a6, 16
    addi a11, a7, 16
    call8 esp_nn_multiply_by_quantized_mult_ver1_esp32s3
    st.qr q0, a1, 80
    ld.qr q0, a1, 96
    addi a10, a6, 32
    addi a11, a7, 32
    call8 esp_nn_multiply_by_quantized_mult_ver1_esp32s3
    st.qr q0, a1, 96
    ld.qr q0, a1, 112
    addi a10, a6, 48
    addi a11, a7, 48
    call8 esp_nn_multiply_by_quantized_mult_ver1_esp32s3
    st.qr q0, a1, 112
    l32i a7, a1, 56
    addi a8, a7, 12
    ee.vldbc.32 q5, a8
    addi a8, a7, 16
    ee.vldbc.32 q6, a8
    addi a8, a7, 20
    ee.vldbc.32 q7, a8
    l32i a8, a7, 0
    ld.qr q0, a1, 64
    ld.qr q1, a1, 80
    ld.qr q2, a1, 96
    ld.qr q3, a1, 112
    ee.vadds.s32 q0, q0, q5
    ee.vadds.s32 q1, q1, q5
    ee.vadds.s32 q2, q2, q5
    ee.vadds.s32 q3, q3, q5
    ee.vmin.s32 q0, q0, q7
    ee.vmin.s32 q1, q1, q7
    ee.vmin.s32 q2, q2, q7
    ee.vmin.s32 q3, q3, q7
    ee.vmax.s32 q0, q0, q6
    ee.vmax.s32 q1, q1, q6
    ee.vmax.s32 q2, q2, q6
    ee.vmax.s32 q3, q3, q6
    st.qr q0, a8, 0
    st.qr q1, a8, 16
    st.qr q2, a8, 32
    st.qr q3, a8, 48
    retw.n
    .size esp_nn_dw_s8_kx1_esp32s3, . - esp_nn_dw_s8_kx1_esp32s3
)ASM"
".popsection\n");
