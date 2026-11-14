// Generated from esp_nn_fc_s8_mac16_esp32s3.S (preprocessed assembly, header-only build).
#pragma once
__asm__(".pushsection .text\n"
R"ASM(
       
    .text
    .align 4
    .type esp_nn_fc_s8_mac16_esp32s3, @function
    .align 4
    .weak esp_nn_fc_s8_mac16_esp32s3
esp_nn_fc_s8_mac16_esp32s3:
    entry a1, 32
    ee.zero.accx
    beqz a4, .Ldone__esp_nn_fc_s8_mac16_esp32s3
    ee.ld.128.usar.ip q0, a3, 16
    srai a5, a4, 1
    beqz a5, .Lsingle__esp_nn_fc_s8_mac16_esp32s3
    ee.vld.128.ip q1, a2, 16
    ee.ld.128.usar.ip q2, a3, 16
    loopgtz a5, .Lloop2_end__esp_nn_fc_s8_mac16_esp32s3
    ee.src.q.qup q4, q0, q2
    ee.vld.128.ip q3, a2, 16
    ee.vmulas.s8.accx q4, q1
    ee.ld.128.usar.ip q0, a3, 16
    ee.src.q.qup q5, q2, q0
    ee.vld.128.ip q1, a2, 16
    ee.vmulas.s8.accx q5, q3
    ee.ld.128.usar.ip q2, a3, 16
.Lloop2_end__esp_nn_fc_s8_mac16_esp32s3:
    bbci a4, 0, .Llast_qup__esp_nn_fc_s8_mac16_esp32s3
.Lsingle__esp_nn_fc_s8_mac16_esp32s3:
    ee.vld.128.ip q1, a2, 16
    ee.ld.128.usar.ip q2, a3, 16
    ee.src.q.qup q4, q0, q2
    ee.vmulas.s8.accx q4, q1
    j .Ldone_mac__esp_nn_fc_s8_mac16_esp32s3
.Llast_qup__esp_nn_fc_s8_mac16_esp32s3:
    addi a2, a2, -16
    addi a3, a3, -16
.Ldone_mac__esp_nn_fc_s8_mac16_esp32s3:
.Ldone__esp_nn_fc_s8_mac16_esp32s3:
    movi.n a3, 0
    nop
    ee.srs.accx a2, a3, 0
    retw.n
    .size esp_nn_fc_s8_mac16_esp32s3, . - esp_nn_fc_s8_mac16_esp32s3
)ASM"
".popsection\n");
