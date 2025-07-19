#include <iostream>
#include <vector>
#include <cstring>

#define ROTL32(x, n) (((x) << (n)) | ((x) >> (32 - (n))))

// 常量初始化
static const uint32_t IV[8] = {
    0x7380166f, 0x4914b2b9, 0x172442d7, 0xda8a0600,
    0x5a63e28c, 0x2f5f1b22, 0x3b101e6d, 0x9b4e430d
};

// P0 和 P1 变换
uint32_t P0(uint32_t x) {
    return x ^ ROTL32(x, 9) ^ ROTL32(x, 17);
}

uint32_t P1(uint32_t x) {
    return x ^ ROTL32(x, 15) ^ ROTL32(x, 23);
}

// 消息扩展
void message_schedule(const uint8_t* message, uint32_t* W) {
    for (int i = 0; i < 16; i++) {
        W[i] = (message[i * 4] << 24) | (message[i * 4 + 1] << 16) | (message[i * 4 + 2] << 8) | message[i * 4 + 3];
    }
    for (int i = 16; i < 68; i++) {
        W[i] = P1(W[i - 16] ^ W[i - 9] ^ ROTL32(W[i - 3], 15)) ^ ROTL32(W[i - 13], 7) ^ W[i - 6];
    }
}

// 主算法
void sm3(const uint8_t* input, size_t len, uint8_t* output) {
    uint32_t H[8];
    memcpy(H, IV, sizeof(IV));

    // 填充数据并分块处理
    size_t block_count = (len + 8 + 63) / 64; // 每个块 512 位，最后一个块包含数据长度
    std::vector<uint8_t> padded_input(block_count * 64, 0);
    memcpy(padded_input.data(), input, len);

    padded_input[len] = 0x80; // 添加 '1' 位
    uint64_t bit_len = len * 8;
    padded_input[block_count * 64 - 8] = (bit_len >> 56) & 0xff;
    padded_input[block_count * 64 - 7] = (bit_len >> 48) & 0xff;
    padded_input[block_count * 64 - 6] = (bit_len >> 40) & 0xff;
    padded_input[block_count * 64 - 5] = (bit_len >> 32) & 0xff;
    padded_input[block_count * 64 - 4] = (bit_len >> 24) & 0xff;
    padded_input[block_count * 64 - 3] = (bit_len >> 16) & 0xff;
    padded_input[block_count * 64 - 2] = (bit_len >> 8) & 0xff;
    padded_input[block_count * 64 - 1] = bit_len & 0xff;

    // 处理每个块
    for (size_t i = 0; i < block_count; i++) {
        uint32_t W[68];
        message_schedule(padded_input.data() + i * 64, W);

        uint32_t W1[64];
        for (int j = 0; j < 64; j++) {
            W1[j] = P0(W[j] ^ W[j + 4]);
        }

        uint32_t A = H[0], B = H[1], C = H[2], D = H[3];
        uint32_t E = H[4], F = H[5], G = H[6], H0 = H[7];

        // 进行 64 次迭代
        for (int j = 0; j < 64; j++) {
            uint32_t SS1 = ROTL32(ROTL32(A, 12) + E + ROTL32(0x79cc4519, j % 32), 7);
            uint32_t SS2 = SS1 ^ ROTL32(A, 12);
            uint32_t T = P1(A ^ B ^ C) + D + SS2 + W1[j];

            D = C;
            C = ROTL32(B, 9);
            B = A;
            A = T;
            E = F;
            F = G;
            G = H0;
            H0 = P1(E ^ F ^ G) + H[4] + SS1 + W[j];
        }

        // 更新 H
        H[0] ^= A;
        H[1] ^= B;
        H[2] ^= C;
        H[3] ^= D;
        H[4] ^= E;
        H[5] ^= F;
        H[6] ^= G;
        H[7] ^= H0;
    }

    // 输出结果
    for (int i = 0; i < 8; i++) {
        output[i * 4] = (H[i] >> 24) & 0xff;
        output[i * 4 + 1] = (H[i] >> 16) & 0xff;
        output[i * 4 + 2] = (H[i] >> 8) & 0xff;
        output[i * 4 + 3] = H[i] & 0xff;
    }
}

int main() {
    // 测试输入
    const char* input = "abc";
    uint8_t output[32];

    sm3(reinterpret_cast<const uint8_t*>(input), strlen(input), output);

    // 打印输出
    for (int i = 0; i < 32; i++) {
        printf("%02x", output[i]);
    }
    printf("\n");

    return 0;
}
