#include <iostream>
#include <vector>
#include <cstring>
#include <cstdint>

// 32位循环左移
#define ROTL32(x, n) (((x) << (n)) | ((x) >> (32 - (n))))

// 初始哈希值IV
static const uint32_t IV[8] = {
    0x7380166f,0x4914b2b9,0x172442d7,0xda8a0600,
    0x5a63e28c,0x2f5f1b22,0x3b101e6d,0x9b4e430d
};

// 内联P0变换
inline uint32_t P0(uint32_t x) {
    return x ^ ROTL32(x, 9) ^ ROTL32(x, 17);
}

// 内联P1变换
inline uint32_t P1(uint32_t x) {
    return x ^ ROTL32(x, 15) ^ ROTL32(x, 23);
}

// 预先计算的Tj数组，全局静态，避免循环内重复计算
static uint32_t T[64];

void init_T() {
    for (int j = 0; j < 16; j++)
        T[j] = ROTL32(0x79cc4519, j);
    for (int j = 16; j < 64; j++)
        T[j] = ROTL32(0x7a879d8a, j % 32);
}

// 消息扩展函数，避免多余拷贝，逻辑紧凑
void message_schedule(const uint8_t* message, uint32_t* W) {
    for (int i = 0; i < 16; i++) {
        W[i] = (uint32_t(message[4 * i]) << 24) |
            (uint32_t(message[4 * i + 1]) << 16) |
            (uint32_t(message[4 * i + 2]) << 8) |
            (uint32_t(message[4 * i + 3]));
    }
    for (int i = 16; i < 68; i++) {
        uint32_t tmp = W[i - 16] ^ W[i - 9] ^ ROTL32(W[i - 3], 15);
        W[i] = P1(tmp) ^ ROTL32(W[i - 13], 7) ^ W[i - 6];
    }
}

void sm3(const uint8_t* input, size_t len, uint8_t* output) {
    // 初始化T数组，避免每次调用重复计算
    static bool initialized = false;
    if (!initialized) {
        init_T();
        initialized = true;
    }

    uint32_t H[8];
    memcpy(H, IV, sizeof(IV));

    size_t block_count = (len + 8 + 63) / 64;
    std::vector<uint8_t> padded(block_count * 64, 0);
    memcpy(padded.data(), input, len);

    padded[len] = 0x80;
    uint64_t bit_len = len * 8;
    for (int i = 0; i < 8; i++) {
        padded[padded.size() - 1 - i] = (bit_len >> (8 * i)) & 0xFF;
    }

    for (size_t i = 0; i < block_count; i++) {
        uint32_t W[68];
        message_schedule(padded.data() + i * 64, W);

        uint32_t W1[64];
        for (int j = 0; j < 64; j++) {
            W1[j] = P0(W[j] ^ W[j + 4]);
        }

        // 明确命名状态变量，避免混淆
        uint32_t A = H[0], B = H[1], C = H[2], D = H[3];
        uint32_t E = H[4], F = H[5], G = H[6], H_ = H[7];

        for (int j = 0; j < 64; j++) {
            uint32_t SS1 = ROTL32((ROTL32(A, 12) + E + T[j]) & 0xFFFFFFFF, 7);
            uint32_t SS2 = SS1 ^ ROTL32(A, 12);

            // j区分逻辑，保证32位操作
            uint32_t FF = (j <= 15) ? (A ^ B ^ C) : ((A & B) | (A & C) | (B & C));
            uint32_t GG = (j <= 15) ? (E ^ F ^ G) : ((E & F) | ((~E) & G));

            uint32_t TT1 = (FF + D + SS2 + W1[j]) & 0xFFFFFFFF;
            uint32_t TT2 = (GG + H_ + SS1 + W[j]) & 0xFFFFFFFF;

            D = C;
            C = ROTL32(B, 9);
            B = A;
            A = TT1;

            H_ = G;
            G = ROTL32(F, 19);
            F = E;
            E = P0(TT2);
        }

        for (int k = 0; k < 8; k++) {
            if (k == 0) H[k] ^= A;
            else if (k == 1) H[k] ^= B;
            else if (k == 2) H[k] ^= C;
            else if (k == 3) H[k] ^= D;
            else if (k == 4) H[k] ^= E;
            else if (k == 5) H[k] ^= F;
            else if (k == 6) H[k] ^= G;
            else if (k == 7) H[k] ^= H_;
        }
    }

    for (int i = 0; i < 8; i++) {
        output[i * 4] = (H[i] >> 24) & 0xff;
        output[i * 4 + 1] = (H[i] >> 16) & 0xff;
        output[i * 4 + 2] = (H[i] >> 8) & 0xff;
        output[i * 4 + 3] = H[i] & 0xff;
    }
}

int main() {
    const char* input = "abc";
    uint8_t output[32];
    sm3(reinterpret_cast<const uint8_t*>(input), strlen(input), output);

    for (int i = 0; i < 32; i++) {
        printf("%02x", output[i]);
    }
    printf("\n");
    return 0;
}
