// Generated from esp_nn_depthwise_conv_s16_mult8_esp32s3.S (preprocessed assembly, header-only build).
#pragma once
__asm__(".pushsection .text\n"
R"ASM(
       
    .text
    .literal_position
    # Program Unit: esp_nn_depthwise_conv_s16_mult8_esp32s3
    .type esp_nn_depthwise_conv_s16_mult8_esp32s3, @function
    .align 4
    .weak esp_nn_depthwise_conv_s16_mult8_esp32s3
esp_nn_depthwise_conv_s16_mult8_esp32s3: # 0x14d7
    # qacc_scratch = 0
    # gra_spill_temp_183 = 48
    # gra_spill_temp_184 = 52
    # gra_spill_temp_185 = 56
    # gra_spill_temp_186 = 60
    # gra_spill_temp_187 = 64
    # gra_spill_temp_188 = 68
    # gra_spill_temp_189 = 72
    # gra_spill_temp_190 = 76
    # gra_spill_temp_191 = 80
    # gra_spill_temp_192 = 84
    # gra_spill_temp_193 = 88
    # gra_spill_temp_194 = 92
    # gra_spill_temp_195 = 96
    # gra_spill_temp_196 = 100
    # gra_spill_temp_197 = 104
    # gra_spill_temp_198 = 108
    # gra_spill_temp_199 = 112
    # gra_spill_temp_200 = 116
    # gra_spill_temp_201 = 120
    # gra_spill_temp_202 = 124
    # gra_spill_temp_203 = 128
    # gra_spill_temp_204 = 132
    # gra_spill_temp_205 = 136
    # gra_spill_temp_206 = 140
    # gra_spill_temp_207 = 144
    # gra_spill_temp_208 = 148
    # gra_spill_temp_209 = 152
    # gra_spill_temp_210 = 156
    # gra_spill_temp_211 = 160
    # gra_spill_temp_212 = 164
    # gra_spill_temp_213 = 168
    # gra_spill_temp_214 = 172
    # gra_spill_temp_215 = 176
    # gra_spill_temp_216 = 180
    # gra_spill_temp_217 = 184
    # gra_spill_temp_218 = 192
    # gra_spill_temp_219 = 208
    entry a1,256 #
    s32i a2,a1,144 # [0] gra_spill_temp_207
    s32i.n a4,a1,56 # [1] gra_spill_temp_185
    s32i a5,a1,172 # [2] gra_spill_temp_214
    l32i a9,a1,284 # [3] id:241 out_data+0x0
    l16ui a8,a1,292 # [4] id:242 out_ht+0x0
    s32i a8,a1,64 # [5] gra_spill_temp_187
    s32i a9,a1,124 # [6] gra_spill_temp_202
    beqz.n a8,.Lt_9_8450__esp_nn_depthwise_conv_s16_mult8_esp32s3 # [7]
    s32i a1,a1,128 # [0] gra_spill_temp_203
    neg a13,a7 # [1]
    movi.n a4,0 # [2]
    neg a12,a6 # [3]
    l32i a9,a1,280 # [4] id:243 bias+0x0
    slli a11,a5,1 # [5]
    l16ui a10,a1,264 # [6] id:244 ch_mult+0x0
    l32i a14,a1,268 # [7] id:245 filter_data+0x0
    s32i a14,a1,160 # [8] gra_spill_temp_211
    s32i a10,a1,92 # [9] gra_spill_temp_194
    s32i a11,a1,156 # [10] gra_spill_temp_210
    s32i a9,a1,112 # [11] gra_spill_temp_199
    sext a12,a12,15 # [12]
    s32i a4,a1,68 # [13] gra_spill_temp_188
    sext a13,a13,15 # [14]
    l16ui a4,a1,272 # [15] id:246 filter_wd+0x0
    s32i a13,a1,100 # [16] gra_spill_temp_196
    s32i.n a12,a1,48 # [17] gra_spill_temp_183
    mul16u a8,a5,a10 # [18]
    extui a9,a9,0,4 # [19]
    l32i a11,a1,304 # [20] id:249 out_mult+0x0
    s32i a11,a1,80 # [21] gra_spill_temp_191
    s32i a9,a1,104 # [22] gra_spill_temp_197
    s32i a8,a1,148 # [23] gra_spill_temp_208
    addi a10,a10,-7 # [24]
    l32i a12,a1,300 # [25] id:248 out_shift+0x0
    l16ui a13,a1,256 # [26] id:247 stride_wd+0x0
    s32i a13,a1,72 # [27] gra_spill_temp_189
    s32i a12,a1,76 # [28] gra_spill_temp_190
    s32i a10,a1,116 # [29] gra_spill_temp_200
    slli a8,a8,1 # [30]
    l16ui a9,a1,260 # [31] id:251 stride_ht+0x0
    s32i.n a9,a1,60 # [32] gra_spill_temp_186
    s32i a8,a1,152 # [33] gra_spill_temp_209
    l16ui a10,a1,276 # [34] id:250 filter_ht+0x0
    s32i.n a10,a1,52 # [35] gra_spill_temp_184
    l16ui a8,a1,288 # [36] id:252 out_wd+0x0
    s32i a8,a1,84 # [37] gra_spill_temp_192
    j .Lt_9_8962__esp_nn_depthwise_conv_s16_mult8_esp32s3 # [38]
.Lt_9_9218__esp_nn_depthwise_conv_s16_mult8_esp32s3: # 0x1561
#<loop> Part of loop body line 1083, head labeled .Lt_9_8962__esp_nn_depthwise_conv_s16_mult8_esp32s3
    l32i a15,a1,64 # [0] gra_spill_temp_187
    l32i.n a9,a1,60 # [1] gra_spill_temp_186
    l32i a14,a1,68 # [2] gra_spill_temp_188
    l32i a8,a1,100 # [3] gra_spill_temp_196
    addi.n a14,a14,1 # [4]
    s32i a14,a1,68 # [5] gra_spill_temp_188
    add.n a9,a8,a9 # [6]
    sub a14,a14,a15 # [7]
    sext a8,a9,15 # [8]
    s32i a8,a1,100 # [9] gra_spill_temp_196
    beqz a14,.Lt_9_8450__esp_nn_depthwise_conv_s16_mult8_esp32s3 # [10]
.Lt_9_8962__esp_nn_depthwise_conv_s16_mult8_esp32s3: # 0x157f
    l32i a10,a1,84 # [0] gra_spill_temp_192
    beqz.n a10,.Lt_9_9218__esp_nn_depthwise_conv_s16_mult8_esp32s3 # [2]
    l32i.n a7,a1,52 # [0] gra_spill_temp_184
    movi.n a11,0 # [1]
    l32i.n a8,a1,56 # [2] gra_spill_temp_185
    l32i a9,a1,100 # [3] gra_spill_temp_196
    l32i.n a12,a1,48 # [4] gra_spill_temp_183
    s32i a12,a1,168 # [5] gra_spill_temp_213
    neg a10,a9 # [6]
    sub a8,a8,a9 # [7]
    max a10,a10,a11 # [8]
    s32i a10,a1,108 # [9] gra_spill_temp_198
    min a7,a7,a8 # [10]
    movi.n a11,0 # [11]
    s32i a11,a1,88 # [12] gra_spill_temp_193
    j .Lt_9_9730__esp_nn_depthwise_conv_s16_mult8_esp32s3 # [13]
.Lt_9_9986__esp_nn_depthwise_conv_s16_mult8_esp32s3: # 0x15a9
#<loop> Part of loop body line 1085, head labeled .Lt_9_9730__esp_nn_depthwise_conv_s16_mult8_esp32s3
    l32i a13,a1,84 # [0] gra_spill_temp_192
    l32i a15,a1,72 # [1] gra_spill_temp_189
    l32i a12,a1,88 # [2] gra_spill_temp_193
    l32i a14,a1,168 # [3] gra_spill_temp_213
    addi.n a12,a12,1 # [4]
    s32i a12,a1,88 # [5] gra_spill_temp_193
    add.n a15,a14,a15 # [6]
    sext a14,a15,15 # [7]
    s32i a14,a1,168 # [8] gra_spill_temp_213
    beq a12,a13,.Lt_9_9218__esp_nn_depthwise_conv_s16_mult8_esp32s3 # [9]
.Lt_9_9730__esp_nn_depthwise_conv_s16_mult8_esp32s3: # 0x15c5
#<loop> Loop body line 1085, nesting depth: 2, estimated iterations: 100
 #1086 const int16_t base_x = (out_x * stride_wd) - pad_wd;
 #1087 const int32_t *out_mult_ptr = out_mult;
 #1088 const int32_t *out_shift_ptr = out_shift;
 #1089 uint32_t bias_ptr = (uint32_t) (bias);
 #1090 for (int ch_idx = 0; ch_idx < channels; ch_idx++) {
    l32i a8,a1,172 # [0] gra_spill_temp_214
    l32i a9,a1,80 # [1] gra_spill_temp_191
    l32i a10,a1,76 # [2] gra_spill_temp_190
    l32i a11,a1,112 # [3] gra_spill_temp_199
    s32i a11,a1,120 # [4] gra_spill_temp_201
    s32i a10,a1,140 # [5] gra_spill_temp_206
    s32i a9,a1,136 # [6] gra_spill_temp_205
    beqz.n a8,.Lt_9_9986__esp_nn_depthwise_conv_s16_mult8_esp32s3 # [7]
.LBB9_esp_nn_depthwise_conv_s16_mult8__esp_nn_depthwise_conv_s16_mult8_esp32s3: # 0x15dc
#<loop> Part of loop body line 1085, head labeled .Lt_9_9730__esp_nn_depthwise_conv_s16_mult8_esp32s3
    movi.n a8,0 # [0]
    l32i a5,a1,168 # [1] gra_spill_temp_213
    movi.n a13,0 # [2]
    movi.n a14,0 # [3]
    s32i a14,a1,96 # [4] gra_spill_temp_195
    s32i a13,a1,184 # [5] gra_spill_temp_217
    neg a6,a5 # [6]
    max a6,a6,a8 # [7]
    sub a5,a3,a5 # [8]
    min a5,a4,a5 # [9]
    sub a12,a5,a6 # [10]
    s32i a12,a1,164 # [11] gra_spill_temp_212
    j .Lt_9_10498__esp_nn_depthwise_conv_s16_mult8_esp32s3 # [12]
.Lt_9_10754__esp_nn_depthwise_conv_s16_mult8_esp32s3: # 0x1600
#<loop> Part of loop body line 1090, head labeled .Lt_9_10498__esp_nn_depthwise_conv_s16_mult8_esp32s3
    l32i a10,a1,172 # [0] gra_spill_temp_214
    l32i a12,a1,92 # [1] gra_spill_temp_194
    l32i a9,a1,184 # [2] gra_spill_temp_217
    l32i a11,a1,96 # [3] gra_spill_temp_195
    addi.n a9,a9,1 # [4]
    s32i a9,a1,184 # [5] gra_spill_temp_217
    add.n a11,a11,a12 # [6]
    s32i a11,a1,96 # [7] gra_spill_temp_195
    beq a9,a10,.Lt_9_9986__esp_nn_depthwise_conv_s16_mult8_esp32s3 # [8]
.Lt_9_10498__esp_nn_depthwise_conv_s16_mult8_esp32s3: # 0x1619
#<loop> Loop body line 1090, nesting depth: 3, estimated iterations: 100
 #1091 for (int ch_mult_idx = 0; ch_mult_idx < ch_mult - 7; ch_mult_idx += 8) {
    l32i a13,a1,116 # [0] gra_spill_temp_200
    blti a13,1,.Lt_9_10754__esp_nn_depthwise_conv_s16_mult8_esp32s3 # [2]
.LBB12_esp_nn_depthwise_conv_s16_mult8__esp_nn_depthwise_conv_s16_mult8_esp32s3: # 0x161f
#<loop> Part of loop body line 1090, head labeled .Lt_9_10498__esp_nn_depthwise_conv_s16_mult8_esp32s3
    l32i a2,a1,96 # [0] gra_spill_temp_195
    movi.n a14,0 # [1]
    s32i a14,a1,132 # [2] gra_spill_temp_204
    j .Lt_9_11266__esp_nn_depthwise_conv_s16_mult8_esp32s3 # [3]
.Lt_9_11522__esp_nn_depthwise_conv_s16_mult8_esp32s3: # 0x162a
    l32i a9,a1,128 # [0] gra_spill_temp_203
    ee.st.qacc_l.l.128.ip a9,16 # [2] id:257
    ee.st.qacc_l.h.32.ip a9,0 # [3] id:258
    l8ui a10,a1,15 # [4] qacc_scratch+15
    l16ui a8,a1,10 # [5] qacc_scratch+10
    l8ui a13,a1,16 # [6] qacc_scratch+16
    l8ui a12,a1,6 # [7] qacc_scratch+6
    l8ui a11,a1,5 # [8] qacc_scratch+5
    s8i a11,a1,2 # [9] qacc_scratch+2
    s8i a12,a1,3 # [10] qacc_scratch+3
    s8i a13,a1,7 # [11] qacc_scratch+7
    s16i a8,a1,4 # [12] qacc_scratch+4
    s8i a10,a1,6 # [13] qacc_scratch+6
    movi.n a8,16 # [14]
    ee.st.qacc_h.l.128.ip a9,16 # [15] id:268
    ee.st.qacc_h.h.32.ip a9,-32 # [16] id:269
    ee.srcmb.s16.qacc q1,a8,0 # [17]
    l16ui a13,a1,26 # [18] qacc_scratch+26
    l8ui a15,a1,32 # [19] qacc_scratch+32
    l8ui a12,a1,22 # [20] qacc_scratch+22
    l8ui a11,a1,21 # [21] qacc_scratch+21
    l16ui a10,a1,16 # [22] qacc_scratch+16
    l8ui a14,a1,31 # [23] qacc_scratch+31
    s8i a14,a1,14 # [24] qacc_scratch+14
    s16i a10,a1,8 # [25] qacc_scratch+8
    s8i a11,a1,10 # [26] qacc_scratch+10
    s8i a12,a1,11 # [27] qacc_scratch+11
    s8i a15,a1,15 # [28] qacc_scratch+15
    s16i a13,a1,12 # [29] qacc_scratch+12
 #1138 EE_VZIP_16(q0, q1);
 #1139
 #1140 if (bias) {
    l32i a15,a1,112 # [30] gra_spill_temp_199
    ee.vld.128.ip q0,a9,0 # [31] id:281
    s32i a9,a1,128 # [32] gra_spill_temp_203
    ee.vzip.16 q0,q1 # [33]
    beqz.n a15,.Lt_9_13570__esp_nn_depthwise_conv_s16_mult8_esp32s3 # [34]
.LBB23_esp_nn_depthwise_conv_s16_mult8__esp_nn_depthwise_conv_s16_mult8_esp32s3: # 0x168e
#<loop> Part of loop body line 1091, head labeled .Lt_9_11266__esp_nn_depthwise_conv_s16_mult8_esp32s3
    addi a14,a1,112 # [0]
    l32i a8,a1,104 # [1] gra_spill_temp_197
    l32i a15,a1,120 # [2] gra_spill_temp_201
    wur.sar_byte a8 # [3]
    ee.vld.128.ip q3,a15,16 # [4] id:284
    ee.vld.128.ip q6,a15,16 # [5] id:285
    ee.vld.128.ip q4,a15,0 # [6] id:286
    s32i a15,a1,120 # [7] gra_spill_temp_201
    ee.src.q.qup q5,q3,q6 # [8]
    ee.vadds.s32 q0,q0,q5 # [9]
    ee.src.q.qup q2,q3,q4 # [10]
    ee.vadds.s32 q1,q1,q2 # [11]
    st.qr q1,a14,96 # [12] gra_spill_temp_219-112
.Lt_9_13570__esp_nn_depthwise_conv_s16_mult8_esp32s3: # 0x16b5
 #1158 q0 = esp_nn_multiply_by_quantized_mult_ver1_esp32s3(q0, out_mult_ptr, out_shift_ptr);
    l32i a10,a1,136 # [0] gra_spill_temp_205
    l32i a11,a1,140 # [1] gra_spill_temp_206
    addi a9,a1,112 # [2]
    st.qr q1,a9,96 # [3] gra_spill_temp_219-112
    call8 esp_nn_multiply_by_quantized_mult_ver1_esp32s3
 #1159 out_mult_ptr += 4;
 #1160 out_shift_ptr += 4;
 #1161
 #1162 q1 = esp_nn_multiply_by_quantized_mult_ver1_esp32s3(q1, out_mult_ptr, out_shift_ptr);
    l32i a11,a1,140 # [0] gra_spill_temp_206
    addi a12,a1,112 # [1]
    l32i a10,a1,136 # [2] gra_spill_temp_205
    st.qr q0,a12,80 # [3] gra_spill_temp_218-112
    ld.qr q0,a12,96 # [4] gra_spill_temp_219-112
    addi a10,a10,16 # [5]
    addi a11,a11,16 # [6]
    call8 esp_nn_multiply_by_quantized_mult_ver1_esp32s3
    addi.n a2,a2,8 # [0]
    l32i a14,a1,116 # [1] gra_spill_temp_200
    l32i a15,a1,124 # [2] gra_spill_temp_202
    l32i a13,a1,132 # [3] gra_spill_temp_204
    l32i a10,a1,140 # [4] gra_spill_temp_206
    l32i a11,a1,136 # [5] gra_spill_temp_205
    addmi a9,a1,256 # [6]
    addi a8,a1,112 # [7]
    ld.qr q7,a8,80 # [8] gra_spill_temp_218-112
    addi a9,a9,56 # [9]
    ee.vldbc.32 q2,a9 # [10] id:290 activation_max
    addi a11,a11,32 # [11]
    addi a10,a10,32 # [12]
    addi.n a13,a13,8 # [13]
    s32i a13,a1,132 # [14] gra_spill_temp_204
    s32i a10,a1,140 # [15] gra_spill_temp_206
    s32i a11,a1,136 # [16] gra_spill_temp_205
    addmi a10,a1,256 # [17]
    addmi a11,a1,256 # [18]
    addi a11,a11,52 # [19]
    addi a10,a10,40 # [20]
    ee.vldbc.32 q3,a10 # [21] id:289 out_offset
    ee.vldbc.32 q1,a11 # [22] id:291 activation_min
    ee.vadds.s32 q0,q0,q3 # [23]
    ee.vadds.s32 q7,q7,q3 # [24]
    ee.vmin.s32 q7,q7,q2 # [25]
    ee.vmin.s32 q0,q0,q2 # [26]
    ee.vmax.s32 q0,q0,q1 # [27]
    ee.vmax.s32 q7,q7,q1 # [28]
    ee.vunzip.16 q7,q0 # [29]
    ee.vunzip.8 q7,q0 # [30]
    ee.vst.l.64.ip q7,a15,8 # [31] id:292
    s32i a15,a1,124 # [32] gra_spill_temp_202
    bge a13,a14,.Lt_9_10754__esp_nn_depthwise_conv_s16_mult8_esp32s3 # [33]
.Lt_9_11266__esp_nn_depthwise_conv_s16_mult8_esp32s3: # 0x1740
    ee.zero.qacc # [0]
    l32i a12,a1,108 # [1] gra_spill_temp_198
    s32i a12,a1,180 # [2] gra_spill_temp_216
    bge a12,a7,.Lt_9_11522__esp_nn_depthwise_conv_s16_mult8_esp32s3 # [3]
    mull a15,a12,a4 # [0]
    l32i a14,a1,100 # [1] gra_spill_temp_196
    add.n a8,a15,a5 # [2]
    add.n a14,a14,a12 # [3]
    mull a14,a3,a14 # [4]
    s32i a8,a1,176 # [5] gra_spill_temp_215
    bge a6,a5,.Lt_9_12290__esp_nn_depthwise_conv_s16_mult8_esp32s3 # [6]
.LBB18_esp_nn_depthwise_conv_s16_mult8__esp_nn_depthwise_conv_s16_mult8_esp32s3: # 0x175f
#<loop> Part of loop body line 1091, head labeled .Lt_9_11266__esp_nn_depthwise_conv_s16_mult8_esp32s3
    l32i a10,a1,184 # [0] gra_spill_temp_217
    l32i a11,a1,172 # [1] gra_spill_temp_214
    l32i a12,a1,168 # [2] gra_spill_temp_213
    l32i a8,a1,148 # [3] gra_spill_temp_208
    add.n a9,a15,a6 # [4]
    mull a8,a8,a9 # [5]
    add.n a12,a12,a6 # [6]
    l32i a9,a1,160 # [7] gra_spill_temp_211
    add.n a12,a14,a12 # [8]
    mull a11,a11,a12 # [9]
    add.n a8,a2,a8 # [10]
    l32i a12,a1,156 # [11] gra_spill_temp_210
    addx2 a8,a8,a9 # [12]
    add.n a10,a10,a11 # [13]
    l32i a11,a1,144 # [14] gra_spill_temp_207
    l32i a9,a1,164 # [15] gra_spill_temp_212
    addx2 a10,a10,a11 # [16]
    l32i a11,a1,152 # [17] gra_spill_temp_209
    loopgtz a9,.LBB45_esp_nn_depthwise_conv_s16_mult8__esp_nn_depthwise_conv_s16_mult8_esp32s3 # [18]
    mov.n a9,a8 # [0*II+0]
    ee.vldbc.16 q0,a10 # [0*II+1] id:255
    ee.vld.128.ip q1,a9,0 # [0*II+2] id:254
    add.n a10,a10,a12 # [0*II+3]
    add.n a8,a8,a11 # [0*II+4]
    ee.vmulas.s16.qacc q0,q1 # [0*II+5]
.LBB45_esp_nn_depthwise_conv_s16_mult8__esp_nn_depthwise_conv_s16_mult8_esp32s3: # 0x17a2
.Lt_9_12290__esp_nn_depthwise_conv_s16_mult8_esp32s3: # 0x17a2
    add.n a14,a14,a3 # [0]
    add.n a15,a15,a4 # [1]
    l32i a10,a1,180 # [2] gra_spill_temp_216
    l32i a11,a1,176 # [3] gra_spill_temp_215
    addi.n a10,a10,1 # [4]
    add.n a11,a11,a4 # [5]
    s32i a11,a1,176 # [6] gra_spill_temp_215
    s32i a10,a1,180 # [7] gra_spill_temp_216
    sub a10,a7,a10 # [8]
    beqz a10,.Lt_9_11522__esp_nn_depthwise_conv_s16_mult8_esp32s3 # [9]
.Lt_9_12034__esp_nn_depthwise_conv_s16_mult8_esp32s3: # 0x17bc
    blt a6,a5,.LBB18_esp_nn_depthwise_conv_s16_mult8__esp_nn_depthwise_conv_s16_mult8_esp32s3 # [0]
    j .Lt_9_12290__esp_nn_depthwise_conv_s16_mult8_esp32s3 # [0]
.Lt_9_8450__esp_nn_depthwise_conv_s16_mult8_esp32s3: # 0x17c2
    retw.n # [0]
    .size esp_nn_depthwise_conv_s16_mult8_esp32s3, . - esp_nn_depthwise_conv_s16_mult8_esp32s3
)ASM"
".popsection\n");
