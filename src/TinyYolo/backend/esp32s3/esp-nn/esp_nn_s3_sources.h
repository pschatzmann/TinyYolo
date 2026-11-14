// Header-only S3 ESP-NN: every source of the vendored ESP-NN (v1.5.0, Apache-2.0) is a header.
// The order puts the common helpers first. Included by Esp32NnBackend.h for ESP32-S3 only.
#pragma once

#include "src/common/common_functions.h"
#include "src/common/esp_nn_common_functions_esp32s3.h"
#include "src/common/esp_nn_dot_s8_esp32s3.h"
#include "src/common/esp_nn_filter_sum_esp32s3.h"
#include "src/common/esp_nn_mean_ansi.h"
#include "src/common/esp_nn_mean_s8_esp32s3.h"
#include "src/common/esp_nn_multicore.h"
#include "src/common/esp_nn_multiply_by_quantized_mult_esp32s3.h"
#include "src/common/esp_nn_multiply_by_quantized_mult_ver1_esp32s3.h"
#include "src/activation_functions/esp_nn_hard_swish_ansi.h"
#include "src/activation_functions/esp_nn_hard_swish_s8_esp32s3.h"
#include "src/activation_functions/esp_nn_relu_ansi.h"
#include "src/activation_functions/esp_nn_relu_s8_esp32s3.h"
#include "src/basic_math/esp_nn_add_ansi.h"
#include "src/basic_math/esp_nn_add_mt_esp32s3.h"
#include "src/basic_math/esp_nn_add_s8_esp32s3.h"
#include "src/basic_math/esp_nn_mul_ansi.h"
#include "src/basic_math/esp_nn_mul_broadcast_s8_esp32s3.h"
#include "src/basic_math/esp_nn_mul_mt.h"
#include "src/basic_math/esp_nn_mul_s8_esp32s3.h"
#include "src/convolution/esp_nn_conv_1x1_panel_esp32s3.h"
#include "src/convolution/esp_nn_conv_ansi.h"
#include "src/convolution/esp_nn_conv_dispatch_esp32s3.h"
#include "src/convolution/esp_nn_conv_esp32s3.h"
#include "src/convolution/esp_nn_conv_mt_split.h"
#include "src/convolution/esp_nn_conv_opt.h"
#include "src/convolution/esp_nn_conv_s16_mult4_1x1_esp32s3.h"
#include "src/convolution/esp_nn_conv_s16_mult8_esp32s3.h"
#include "src/convolution/esp_nn_conv_s8_1x1_esp32s3.h"
#include "src/convolution/esp_nn_conv_s8_3x3_opt_esp32s3.h"
#include "src/convolution/esp_nn_conv_s8_filter_aligned_input_padded_esp32s3.h"
#include "src/convolution/esp_nn_conv_s8_mult8_1x1_esp32s3.h"
#include "src/convolution/esp_nn_depthwise_conv_ansi.h"
#include "src/convolution/esp_nn_depthwise_conv_opt.h"
#include "src/convolution/esp_nn_depthwise_conv_s16_mult1_3x3_esp32s3.h"
#include "src/convolution/esp_nn_depthwise_conv_s16_mult1_3x3_no_pad_esp32s3.h"
#include "src/convolution/esp_nn_depthwise_conv_s16_mult1_esp32s3.h"
#include "src/convolution/esp_nn_depthwise_conv_s16_mult4_esp32s3.h"
#include "src/convolution/esp_nn_depthwise_conv_s16_mult8_3x3_esp32s3.h"
#include "src/convolution/esp_nn_depthwise_conv_s16_mult8_esp32s3.h"
#include "src/convolution/esp_nn_depthwise_conv_s8_esp32s3.h"
#include "src/convolution/esp_nn_depthwise_conv_s8_kx1_esp32s3.h"
#include "src/convolution/esp_nn_depthwise_conv_s8_mult1_3x3_padded_esp32s3.h"
#include "src/convolution/esp_nn_scratch_sizes_esp32s3.h"
#include "src/fully_connected/esp_nn_fc_batch_conv.h"
#include "src/fully_connected/esp_nn_fc_internal.h"
#include "src/fully_connected/esp_nn_fc_s8_mac16_esp32s3.h"
#include "src/fully_connected/esp_nn_fully_connected_ansi.h"
#include "src/fully_connected/esp_nn_fully_connected_esp32s3.h"
#include "src/fully_connected/esp_nn_fully_connected_per_ch_s8_esp32s3.h"
#include "src/fully_connected/esp_nn_fully_connected_s8_esp32s3.h"
#include "src/logistic/esp_nn_logistic_ansi.h"
#include "src/pooling/esp_nn_avg_pool_ansi.h"
#include "src/pooling/esp_nn_avg_pool_s8_esp32s3.h"
#include "src/pooling/esp_nn_max_pool_ansi.h"
#include "src/pooling/esp_nn_max_pool_s8_esp32s3.h"
#include "src/softmax/esp_nn_softmax_ansi.h"
#include "src/softmax/esp_nn_softmax_opt.h"
#include "src/softmax/esp_nn_softmax_s8_esp32s3.h"
#include "src/softmax/softmax_common.h"

// The sources include the ANSI and generic headers, whose public names (for example
// esp_nn_fully_connected_per_ch_s8) point at other kernels. In one translation unit the last
// definition wins, so the ESP32-S3 names are defined again here, as in esp_nn_esp32s3.h.
#undef esp_nn_add_elementwise_s8
#define esp_nn_add_elementwise_s8 esp_nn_add_elementwise_s8_mt_esp32s3
#undef esp_nn_mul_broadcast_channel_s8
#define esp_nn_mul_broadcast_channel_s8 esp_nn_mul_broadcast_channel_s8_esp32s3
#undef esp_nn_depthwise_conv_s8
#define esp_nn_depthwise_conv_s8 esp_nn_depthwise_conv_s8_esp32s3
#undef esp_nn_get_conv_scratch_size
#define esp_nn_get_conv_scratch_size esp_nn_get_conv_scratch_size_esp32s3
#undef esp_nn_set_conv_scratch_buf
#define esp_nn_set_conv_scratch_buf esp_nn_set_conv_scratch_buf_esp32s3
#undef esp_nn_get_depthwise_conv_scratch_size
#define esp_nn_get_depthwise_conv_scratch_size esp_nn_get_depthwise_conv_scratch_size_esp32s3
#undef esp_nn_set_depthwise_conv_scratch_buf
#define esp_nn_set_depthwise_conv_scratch_buf esp_nn_set_depthwise_conv_scratch_buf_esp32s3
#undef esp_nn_conv_s8
#define esp_nn_conv_s8 esp_nn_conv_s8_esp32s3
#undef esp_nn_relu6_s8
#define esp_nn_relu6_s8 esp_nn_relu6_s8_esp32s3
#undef esp_nn_get_hard_swish_scratch_size
#define esp_nn_get_hard_swish_scratch_size esp_nn_get_hard_swish_scratch_size_esp32s3
#undef esp_nn_set_hard_swish_scratch_buf
#define esp_nn_set_hard_swish_scratch_buf esp_nn_set_hard_swish_scratch_buf_esp32s3
#undef esp_nn_hard_swish_s8
#define esp_nn_hard_swish_s8 esp_nn_hard_swish_s8_esp32s3
#undef esp_nn_mean_nhwc_s8
#define esp_nn_mean_nhwc_s8 esp_nn_mean_nhwc_s8_esp32s3
#undef esp_nn_get_mean_scratch_size
#define esp_nn_get_mean_scratch_size esp_nn_get_mean_scratch_size_esp32s3
#undef esp_nn_set_mean_scratch_buf
#define esp_nn_set_mean_scratch_buf esp_nn_set_mean_scratch_buf_esp32s3
#undef esp_nn_avg_pool_s8
#define esp_nn_avg_pool_s8 esp_nn_avg_pool_s8_esp32s3
#undef esp_nn_max_pool_s8
#define esp_nn_max_pool_s8 esp_nn_max_pool_s8_esp32s3
#undef esp_nn_fully_connected_s8
#define esp_nn_fully_connected_s8 esp_nn_fully_connected_s8_esp32s3
#undef esp_nn_fully_connected_per_ch_s8
#define esp_nn_fully_connected_per_ch_s8 esp_nn_fully_connected_per_ch_s8_esp32s3
#undef esp_nn_fully_connected_per_ch_s8_batch
#define esp_nn_fully_connected_per_ch_s8_batch esp_nn_fully_connected_per_ch_s8_batch_esp32s3
#undef esp_nn_get_softmax_scratch_size
#define esp_nn_get_softmax_scratch_size esp_nn_get_softmax_scratch_size_esp32s3
#undef esp_nn_set_softmax_scratch_buf
#define esp_nn_set_softmax_scratch_buf esp_nn_set_softmax_scratch_buf_esp32s3
#undef esp_nn_softmax_s8
#define esp_nn_softmax_s8 esp_nn_softmax_s8_esp32s3
