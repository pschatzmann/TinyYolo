// Generated from esp_nn_relu_s8_esp32s3.S (preprocessed assembly, header-only build).
#pragma once
__asm__(".pushsection .text\n"
R"ASM(
       
    .text
    .align 4
    .literal_position
# in place relu6 function. a2: data, a3: size
    # Program Unit: esp_nn_relu6_s8_esp32s3
    .type esp_nn_relu6_s8_esp32s3, @function
    .align 4
    .weak esp_nn_relu6_s8_esp32s3
esp_nn_relu6_s8_esp32s3:
    entry a1,48 #
    extui a5,a2,0,4 # a5 = data & 15
    beqz.n a5,.Lt_head_aligned__esp_nn_relu_s8_esp32s3
    movi.n a6,16
    sub a5,a6,a5 # head count = 16 - (data & 15)
    min a5,a5,a3 # clamp to size
    sub a3,a3,a5 # size -= head count
    movi.n a6,0 # zero (relu lower bound)
    movi.n a4,6 # six (relu6 upper bound)
    loopgtz a5,.Lt_head_done__esp_nn_relu_s8_esp32s3
    l8ui a8,a2,0
    sext a8,a8,7
    max a8,a6,a8
    min a8,a4,a8
    s8i a8,a2,0
    addi.n a2,a2,1
.Lt_head_done__esp_nn_relu_s8_esp32s3:
.Lt_head_aligned__esp_nn_relu_s8_esp32s3:
    mov.n a9,a2 # [0], data
    mov.n a7,a3 # [1], size
    movi.n a4,6 # [4]
    s8i a4,a1,0 # [5] six
    addi a10,a3,-7 # [2]
    ee.vldbc.8 q1,a1 # [6] id:72 six+0x0
    blti a3,16,.Lt_0_5634__esp_nn_relu_s8_esp32s3 # [7]
    srai a8,a3,4 # [0]
    ee.zero.q q2 # [1]
    loopgtz a8,.LBB37_esp_nn_relu6_s8_esp32s3__esp_nn_relu_s8_esp32s3 # [3]
    ee.vld.128.ip q0,a2,0 # [0*II+0] id:73
    ee.vmax.s8 q0,q0,q2 # [0*II+2]
    ee.vmin.s8 q0,q0,q1 # [0*II+3]
    ee.vst.128.ip q0,a2,16 # [0*II+4] id:74
.LBB37_esp_nn_relu6_s8_esp32s3__esp_nn_relu_s8_esp32s3: # 0x34
    slli a8,a8,4 # [0]
    bge a8,a10,.Lt_0_3586__esp_nn_relu_s8_esp32s3 # [1]
.Lt_0_3842__esp_nn_relu_s8_esp32s3: # 0x3a
    sub a6,a7,a8 # [0]
    srai a6,a6,3 # [1]
    loopgtz a6,.LBB52_esp_nn_relu6_s8_esp32s3__esp_nn_relu_s8_esp32s3 # [2]
    ee.vld.l.64.ip q0,a2,0 # [0*II+0] id:75
    ee.vmax.s8 q0,q0,q2 # [0*II+2]
    ee.vmin.s8 q0,q0,q1 # [0*II+3]
    ee.vst.l.64.ip q0,a2,8 # [0*II+4] id:76
.LBB52_esp_nn_relu6_s8_esp32s3__esp_nn_relu_s8_esp32s3: # 0x4f
    addx8 a8,a6,a8 # [0]
.Lt_0_3586__esp_nn_relu_s8_esp32s3: # 0x52
    bge a8,a7,.Lt_0_6402__esp_nn_relu_s8_esp32s3 # [0]
.Lt_0_4866__esp_nn_relu_s8_esp32s3: # 0x55
    movi.n a5,0 # [0]
    sub a3,a7,a8 # [1]
    add.n a2,a8,a9 # [2]
    l8ui a6,a2,0 # [3] id:78
    addi.n a3,a3,-1 # [4]
    sext a6,a6,7
    max a6,a5,a6 # [6]
    min a6,a4,a6 # [7]
    s8i a6,a2,0 # [8] id:79
    loopgtz a3,.LBB67_esp_nn_relu6_s8_esp32s3__esp_nn_relu_s8_esp32s3 # [9]
    l8ui a3,a2,1 # [0*II+0] id:78
    addi.n a2,a2,1 # [1*II+1]
    sext a3,a3,7
    max a3,a5,a3 # [0*II+3]
    min a3,a4,a3 # [0*II+4]
    s8i a3,a2,0 # [0*II+5] id:79
.LBB67_esp_nn_relu6_s8_esp32s3__esp_nn_relu_s8_esp32s3: # 0x81
.Lt_0_6402__esp_nn_relu_s8_esp32s3: # 0x83
    retw.n # [0]
.Lt_0_5634__esp_nn_relu_s8_esp32s3: # 0x85
    blti a10,1,.Lt_0_5890__esp_nn_relu_s8_esp32s3 # [0]
    movi.n a8,0 # [0]
    ee.zero.q q2 # [1]
    j .Lt_0_3842__esp_nn_relu_s8_esp32s3 # [2]
.Lt_0_5890__esp_nn_relu_s8_esp32s3: # 0x90
    beqz.n a3,.Lt_0_6402__esp_nn_relu_s8_esp32s3 # [0]
    movi.n a8,0 # [0]
    j .Lt_0_4866__esp_nn_relu_s8_esp32s3 # [1]
    .size esp_nn_relu6_s8_esp32s3, . - esp_nn_relu6_s8_esp32s3
)ASM"
".popsection\n");
