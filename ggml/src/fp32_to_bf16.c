#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>

// 位掩码定义
#define FP32_SIGN_MASK  0x80000000U
#define FP32_EXP_MASK   0x7F800000U
#define FP32_MANT_MASK  0x007FFFFFU
#define FP32_FULL_MASK  0xFFFFFFFFU

#define BF16_SIGN_MASK  0x8000U
#define BF16_EXP_MASK   0x7F80U
#define BF16_MANT_MASK  0x007FU
#define BF16_EXP_BIAS   127

// 联合体用于类型转换
typedef union {
    float f;
    uint32_t u;
} float_uint32_t;

typedef union {
    uint16_t u;
    struct {
        uint16_t mantissa : 7;
        uint16_t exponent : 8;
        uint16_t sign : 1;
    } fields;
} bfloat16_t;

/**
 * @brief 将 FP32 转换为 BF16（带舍入）
 * 
 * @param fp32_val FP32 浮点数
 * @return uint16_t BF16 格式的16位整数
 * 
 * 算法步骤：
 * 1. 提取 FP32 的符号、指数、尾数
 * 2. 处理特殊值（NaN、Inf、零）
 * 3. 对尾数进行最近偶数舍入
 * 4. 处理舍入进位导致的溢出
 * 5. 组合为 BF16 格式
 */
uint16_t fp32_to_bf16_round(float fp32_val) {
    float_uint32_t converter;
    converter.f = fp32_val;
    uint32_t fp32_bits = converter.u;
    
    // 1. 提取 FP32 的组成部分
    uint32_t sign = (fp32_bits >> 31) & 0x1;
    uint32_t exp = (fp32_bits >> 23) & 0xFF;
    uint32_t mant = fp32_bits & 0x007FFFFF;
    
    // 2. 处理特殊值
    if (exp == 0xFF) {  // NaN 或 Inf
        if (mant != 0) {  // NaN
            // 保留部分 NaN 信息
            uint16_t bf16_bits = (sign << 15) | 0x7F80U | ((mant >> 16) & 0x3FU);
            // 确保尾数不为0（NaN的标志）
            if ((bf16_bits & 0x7F) == 0) {
                bf16_bits |= 0x0040U;
            }
            return (uint16_t)bf16_bits;
        } else {  // Inf
            return (uint16_t)((sign << 15) | 0x7F80U);
        }
    }
    
    // 3. 处理零
    if (exp == 0 && mant == 0) {
        return (uint16_t)(sign << 15);
    }
    
    // 4. 处理反规格化数（可能上溢为规格化数）
    if (exp == 0) {
        // 反规格化数：隐含位为0，需要规格化
        // 找到第一个非零位
        uint32_t shift = 0;
        while ((mant & (1U << 22)) == 0 && shift < 22) {
            mant <<= 1;
            shift++;
        }
        exp = 1 - shift;  // 调整指数
        // 移除隐含位（对于反规格化数，隐含位为0）
        mant &= 0x007FFFFFU;
    } else {
        // 规格化数：添加隐含位
        mant |= 0x00800000U;  // 添加隐含的1（第23位）
    }
    
    // 5. 尾数舍入（最近偶数舍入）
    // 保护位(G)、舍入位(R)、粘滞位(S)
    // mant位：23-0（23是隐含位，22-0是尾数）
    // 保留位：22-16（7位）
    // G: 位15, R: 位14, S: 位13-0的或
    
    // 提取舍入位信息
    uint32_t guard_bit = (mant >> 15) & 0x1;      // 位15
    uint32_t round_bit = (mant >> 14) & 0x1;      // 位14
    uint32_t sticky_bits = mant & 0x3FFFU;        // 位13-0
    
    // 计算粘滞位（如果低位有任何1，则粘滞位为1）
    uint32_t sticky = (sticky_bits != 0) ? 1 : 0;
    
    // 获取要保留的尾数部分（不包括隐含位）
    uint32_t bf16_mant = (mant >> 16) & 0x7F;     // 位22-16
    
    // 最近偶数舍入
    if (guard_bit == 1) {
        if (round_bit == 1 || sticky == 1) {
            // 情况1: G=1且(R=1或S=1)，进一
            bf16_mant += 1;
        } else {
            // 情况2: G=1, R=0, S=0，向偶数舍入
            if ((bf16_mant & 0x1) == 1) {  // 尾数最低位为1
                bf16_mant += 1;
            }
        }
    }
    
    // 6. 处理尾数溢出（进位导致指数增加）
    if (bf16_mant > 0x7F) {
        bf16_mant = 0;
        exp += 1;
        
        // 指数溢出检查
        if (exp > 0xFE) {  // 最大指数为254（0xFE）
            // 变为无穷大
            return (uint16_t)((sign << 15) | 0x7F80U);
        }
    }
    
    // 7. 处理指数下溢
    if (exp < 1) {
        // BF16不支持反规格化数，直接变为0
        return (uint16_t)(sign << 15);
    }
    
    // 8. 组合 BF16
    uint16_t bf16_bits = (sign << 15) | ((exp & 0xFF) << 7) | (bf16_mant & 0x7F);
    return bf16_bits;
}

/**
 * @brief 简化版本：直接截断转换（无舍入）
 * 
 * @param fp32_val FP32 浮点数
 * @return uint16_t BF16 格式的16位整数
 */
uint16_t fp32_to_bf16_trunc(float fp32_val) {
    float_uint32_t converter;
    converter.f = fp32_val;
    
    // 直接右移16位（截断）
    uint32_t fp32_bits = converter.u;
    uint16_t bf16_bits = (uint16_t)(fp32_bits >> 16);
    
    // 处理NaN：确保尾数不为0
    uint16_t exp = (bf16_bits >> 7) & 0xFF;
    uint16_t mant = bf16_bits & 0x7F;
    
    if (exp == 0xFF && mant == 0) {
        // 如果尾数为0但指数全1，可能是Inf转NaN，需要修正
        bf16_bits |= 0x0040U;  // 设置尾数非0
    }
    
    return bf16_bits;
}

/**
 * @brief 将 BF16 转换回 FP32（用于验证）
 * 
 * @param bf16_val BF16 格式的16位整数
 * @return float FP32 浮点数
 */
float bf16_to_fp32(uint16_t bf16_val) {
    float_uint32_t converter;
    
    // 提取BF16的组成部分
    uint32_t sign = (bf16_val >> 15) & 0x1;
    uint32_t exp = (bf16_val >> 7) & 0xFF;
    uint32_t mant = bf16_val & 0x7F;
    
    // 处理特殊值
    if (exp == 0xFF) {  // NaN 或 Inf
        if (mant != 0) {  // NaN
            // 将BF16 NaN转换为FP32 NaN
            converter.u = (sign << 31) | 0x7F800000U | (mant << 16);
        } else {  // Inf
            converter.u = (sign << 31) | 0x7F800000U;
        }
    } else if (exp == 0) {  // 零或反规格化数
        if (mant == 0) {
            converter.u = sign << 31;  // 零
        } else {
            // BF16不支持反规格化数，这里简单处理为0
            converter.u = sign << 31;
        }
    } else {
        // 正常数：扩展尾数
        converter.u = (sign << 31) | (exp << 23) | (mant << 16);
    }
    
    return converter.f;
}

/**
 * @brief 打印浮点数的二进制表示
 * 
 * @param value 浮点数
 * @param name 名称
 */
void print_float_bits(float value, const char* name) {
    float_uint32_t converter;
    converter.f = value;
    
    printf("%s (float): %.8f\n", name, value);
    printf("Binary: ");
    for (int i = 31; i >= 0; i--) {
        printf("%d", (converter.u >> i) & 1);
        if (i == 31 || i == 23) printf(" ");
    }
    printf("\n");
    printf("Hex: 0x%08X\n\n", converter.u);
}

/**
 * @brief 打印 BF16 的二进制表示
 * 
 * @param value BF16 值
 * @param name 名称
 */
void print_bf16_bits(uint16_t value, const char* name) {
    printf("%s (bf16): 0x%04X\n", name, value);
    printf("Binary: ");
    for (int i = 15; i >= 0; i--) {
        printf("%d", (value >> i) & 1);
        if (i == 15 || i == 7) printf(" ");
    }
    printf("\n");
    
    // 解析字段
    uint16_t sign = (value >> 15) & 0x1;
    uint16_t exp = (value >> 7) & 0xFF;
    uint16_t mant = value & 0x7F;
    
    printf("Sign: %u, Exp: %u (0x%02X), Mant: %u (0x%02X)\n", 
           sign, exp, exp, mant, mant);
    
    if (exp == 0xFF) {
        if (mant == 0) {
            printf("Type: %sInfinity\n", sign ? "-" : "+");
        } else {
            printf("Type: NaN\n");
        }
    } else if (exp == 0) {
        if (mant == 0) {
            printf("Type: Zero\n");
        } else {
            printf("Type: Denormal (unsupported in BF16)\n");
        }
    } else {
        printf("Type: Normalized\n");
        // 计算实际值
        float f_val = bf16_to_fp32(value);
        printf("Approx float value: %.8f\n", f_val);
    }
    printf("\n");
}

/**
 * @brief 批量转换测试
 */
void test_batch_conversion() {
    const float test_values[] = {
        0.0f,
        -0.0f,
        1.0f,
        -1.0f,
        3.1415926f,   // π
        2.7182818f,   // e
        1.5f,
        1.25f,
        1.125f,
        1.0625f,
        1.03125f,
        1.015625f,
        1.0078125f,
        1.00390625f,
        INFINITY,
        -INFINITY,
        NAN,
        0.1f,
        0.2f,
        0.3f,
        1e-10f,
        1e10f,
    };
    
    const char* test_names[] = {
        "0.0",
        "-0.0",
        "1.0",
        "-1.0",
        "PI",
        "E",
        "1.5",
        "1.25",
        "1.125",
        "1.0625",
        "1.03125",
        "1.015625",
        "1.0078125",
        "1.00390625",
        "Inf",
        "-Inf",
        "NaN",
        "0.1",
        "0.2",
        "0.3",
        "1e-10",
        "1e10",
    };
    
    int num_tests = sizeof(test_values) / sizeof(test_values[0]);
    
    printf("FP32 to BF16 Conversion Test\n");
    printf("=============================\n\n");
    
    for (int i = 0; i < num_tests; i++) {
        printf("Test %d: %s\n", i + 1, test_names[i]);
        printf("------------------------");
        
        print_float_bits(test_values[i], "Original");
        
        // 使用舍入转换
        uint16_t bf16_round = fp32_to_bf16_round(test_values[i]);
        print_bf16_bits(bf16_round, "BF16 (rounded)");
        
        // 使用截断转换
        uint16_t bf16_trunc = fp32_to_bf16_trunc(test_values[i]);
        print_bf16_bits(bf16_trunc, "BF16 (truncated)");
        
        // 转换回 FP32 并计算误差
        float fp32_back = bf16_to_fp32(bf16_round);
        
        if (!isnan(test_values[i]) && !isinf(test_values[i]) && test_values[i] != 0.0f) {
            float error = fabsf(fp32_back - test_values[i]) / fabsf(test_values[i]);
            printf("Relative error: %.2e\n", error);
        }
        
        printf("------------------------");
        printf("\n");
    }
}

/**
 * @brief 性能测试
 */
void performance_test() {
    const int NUM_SAMPLES = 1000000;
    float* fp32_array = (float*)malloc(NUM_SAMPLES * sizeof(float));
    uint16_t* bf16_array = (uint16_t*)malloc(NUM_SAMPLES * sizeof(uint16_t));
    
    if (!fp32_array || !bf16_array) {
        printf("Memory allocation failed\n");
        return;
    }
    
    // 生成随机数据
    srand(42);
    for (int i = 0; i < NUM_SAMPLES; i++) {
        fp32_array[i] = (float)rand() / RAND_MAX * 100.0f - 50.0f;
    }
    
    printf("Performance Test: %d samples\n", NUM_SAMPLES);
    printf("----------------------------\n");
    
    // 测试舍入转换性能
    clock_t start = clock();
    for (int i = 0; i < NUM_SAMPLES; i++) {
        bf16_array[i] = fp32_to_bf16_round(fp32_array[i]);
    }
    clock_t end = clock();
    double time_round = (double)(end - start) / CLOCKS_PER_SEC;
    
    // 测试截断转换性能
    start = clock();
    for (int i = 0; i < NUM_SAMPLES; i++) {
        bf16_array[i] = fp32_to_bf16_trunc(fp32_array[i]);
    }
    end = clock();
    double time_trunc = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Round conversion time: %.4f seconds\n", time_round);
    printf("Trunc conversion time: %.4f seconds\n", time_trunc);
    printf("Speed ratio: %.2f\n", time_round / time_trunc);
    
    // 统计舍入差异
    int diff_count = 0;
    for (int i = 0; i < NUM_SAMPLES; i++) {
        uint16_t round_val = fp32_to_bf16_round(fp32_array[i]);
        uint16_t trunc_val = fp32_to_bf16_trunc(fp32_array[i]);
        if (round_val != trunc_val) {
            diff_count++;
        }
    }
    
    printf("Different results due to rounding: %d (%.2f%%)\n", 
           diff_count, (float)diff_count / NUM_SAMPLES * 100.0f);
    
    free(fp32_array);
    free(bf16_array);
}

/**
 * @brief 示例：单个数值转换
 */
void single_value_example() {
    printf("Single Value Example\n");
    printf("====================\n\n");
    
    float fp32_num = 3.1415926f;
    
    print_float_bits(fp32_num, "Original FP32");
    
    uint16_t bf16_num = fp32_to_bf16_round(fp32_num);
    print_bf16_bits(bf16_num, "Converted BF16");
    
    float fp32_back = bf16_to_fp32(bf16_num);
    print_float_bits(fp32_back, "BF16 back to FP32");
    
    float absolute_error = fabsf(fp32_num - fp32_back);
    float relative_error = absolute_error / fabsf(fp32_num);
    
    printf("Conversion Results:\n");
    printf("  Original FP32:    %.10f\n", fp32_num);
    printf("  Converted BF16:   0x%04X\n", bf16_num);
    printf("  Reconstructed:    %.10f\n", fp32_back);
    printf("  Absolute error:   %.2e\n", absolute_error);
    printf("  Relative error:   %.2e\n", relative_error);
    printf("\n");
}

/**
 * @brief 验证舍入正确性
 */
void verify_rounding() {
    printf("Rounding Verification\n");
    printf("=====================\n\n");
    
    // 测试边界情况
    struct {
        float input;
        const char* description;
    } test_cases[] = {
        {1.0078125f, "Exact representation (no rounding needed)"},
        {1.0078745f, "Just below midpoint"},
        {1.0078435f, "Just above midpoint"},
        {1.00784375f, "At midpoint, even mantissa"},
        {1.007859375f, "At midpoint, odd mantissa"},
    };
    
    for (size_t i = 0; i < sizeof(test_cases) / sizeof(test_cases[0]); i++) {
        float input = test_cases[i].input;
        uint16_t bf16 = fp32_to_bf16_round(input);
        float output = bf16_to_fp32(bf16);
        
        printf("Test: %s\n", test_cases[i].description);
        printf("  Input:  %.10f\n", input);
        printf("  BF16:   0x%04X\n", bf16);
        printf("  Output: %.10f\n", output);
        printf("  Error:  %.2e\n\n", fabsf(output - input) / fabsf(input));
    }
}

int main() {
    printf("FP32 to BF16 Converter in C\n");
    printf("============================\n\n");
    
    // 运行示例
    single_value_example();
    
    // 验证舍入
    verify_rounding();
    
    // 运行批量测试
    test_batch_conversion();
    
    // 运行性能测试
    performance_test();
    
    return 0;
}
