// Generated from esp_nn_dot_s8_esp32s3.S (preprocessed assembly, header-only build).
#pragma once
__asm__(".pushsection .text\n"
R"ASM(
       
    .text
    .align 4
    .type esp_nn_dot_s8_aligned_esp32s3, @function
    .align 4
    .weak esp_nn_dot_s8_aligned_esp32s3
esp_nn_dot_s8_aligned_esp32s3:
    entry a1, 32
    ee.zero.accx
    beqz a4, .Lalign_done__esp_nn_dot_s8_esp32s3
    srli a5, a4, 4
    beqz a5, .Lalign_done__esp_nn_dot_s8_esp32s3
    ee.vld.128.ip q0, a2, 16
    ee.vld.128.ip q1, a3, 16
    addi a5, a5, -1
    beqz a5, .Lalign_last__esp_nn_dot_s8_esp32s3
    loopgtz a5, .Lalign_loop_end__esp_nn_dot_s8_esp32s3
    ee.vmulas.s8.accx.ld.ip q0, a2, 16, q0, q1
    ee.vld.128.ip q1, a3, 16
.Lalign_loop_end__esp_nn_dot_s8_esp32s3:
.Lalign_last__esp_nn_dot_s8_esp32s3:
    ee.vmulas.s8.accx q0, q1
.Lalign_done__esp_nn_dot_s8_esp32s3:
    nop
    nop
    rur.accx_0 a2
    retw.n
    .size esp_nn_dot_s8_aligned_esp32s3, . - esp_nn_dot_s8_aligned_esp32s3
    .type esp_nn_dot_s8_unaligned_esp32s3, @function
    .align 4
    .weak esp_nn_dot_s8_unaligned_esp32s3
esp_nn_dot_s8_unaligned_esp32s3:
    entry a1, 32
    ee.zero.accx
    extui a7, a2, 0, 4
    extui a6, a3, 0, 4
    bnez a7, .Lunalign_scalar__esp_nn_dot_s8_esp32s3
    beqz a4, .Lunalign_done__esp_nn_dot_s8_esp32s3
    ee.ld.128.usar.ip q0, a3, 16
    addi a5, a4, -1
    beqz a5, .Lunalign_last__esp_nn_dot_s8_esp32s3
    loopgtz a5, .Lunalign_loop_end__esp_nn_dot_s8_esp32s3
    ee.vld.128.ip q1, a2, 16
    ee.ld.128.usar.ip q2, a3, 16
    ee.src.q.qup q4, q0, q2
    ee.vmulas.s8.accx q4, q1
.Lunalign_loop_end__esp_nn_dot_s8_esp32s3:
.Lunalign_last__esp_nn_dot_s8_esp32s3:
    ee.vld.128.ip q1, a2, 16
    beqz a6, .Lunalign_pass__esp_nn_dot_s8_esp32s3
    ee.ld.128.usar.ip q2, a3, 16
.Lunalign_pass__esp_nn_dot_s8_esp32s3:
    ee.src.q.qup q4, q0, q2
    ee.vmulas.s8.accx q4, q1
.Lunalign_done__esp_nn_dot_s8_esp32s3:
    movi.n a3, 0
    nop
    ee.srs.accx a2, a3, 0
    retw.n
.Lunalign_scalar__esp_nn_dot_s8_esp32s3:
    slli a4, a4, 4
    movi.n a7, 0
    beqz a4, .Lunalign_scalar_done__esp_nn_dot_s8_esp32s3
    loopgtz a4, .Lunalign_scalar_loop_end__esp_nn_dot_s8_esp32s3
        l8ui a5, a2, 0
        l8ui a6, a3, 0
        sext a5, a5, 7
        sext a6, a6, 7
        mull a5, a5, a6
        add a7, a7, a5
        addi.n a2, a2, 1
        addi.n a3, a3, 1
.Lunalign_scalar_loop_end__esp_nn_dot_s8_esp32s3:
.Lunalign_scalar_done__esp_nn_dot_s8_esp32s3:
    mov.n a2, a7
    retw.n
    .size esp_nn_dot_s8_unaligned_esp32s3, . - esp_nn_dot_s8_unaligned_esp32s3
)ASM"
".popsection\n");
