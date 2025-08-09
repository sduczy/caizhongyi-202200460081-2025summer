#include <iostream>
#include <vector>
#include <cstring>
#include <chrono>
#include <cstdint>
#include <cstdio>

#define ROTL32(x,n) (((x) << (n)) | ((x) >> (32 - (n))))

// 初始向量
static const uint32_t IV[8] = {
    0x7380166f,0x4914b2b9,0x172442d7,0xda8a0600,
    0x5a63e28c,0x2f5f1b22,0x3b101e6d,0x9b4e430d
};

// 基本变换（inline / 宏）
inline uint32_t P0_u(uint32_t x){ return x ^ ROTL32(x,9) ^ ROTL32(x,17); }
inline uint32_t P1_u(uint32_t x){ return x ^ ROTL32(x,15) ^ ROTL32(x,23); }
#define P0(x) P0_u(x)
#define P1(x) P1_u(x)

// FF, GG 按 j 分段
inline uint32_t FF(uint32_t x,uint32_t y,uint32_t z,int j){
    return (j<=15) ? (x ^ y ^ z) : ((x & y) | (x & z) | (y & z));
}
inline uint32_t GG(uint32_t x,uint32_t y,uint32_t z,int j){
    return (j<=15) ? (x ^ y ^ z) : ((x & y) | ((~x) & z));
}

// T 常量基值
static const uint32_t T_BASE[64] = {
    // 0..15 : 0x79cc4519
    0x79cc4519,0x79cc4519,0x79cc4519,0x79cc4519,0x79cc4519,0x79cc4519,0x79cc4519,0x79cc4519,
    0x79cc4519,0x79cc4519,0x79cc4519,0x79cc4519,0x79cc4519,0x79cc4519,0x79cc4519,0x79cc4519,
    //16..63 : 0x7a879d8a
    0x7a879d8a,0x7a879d8a,0x7a879d8a,0x7a879d8a,0x7a879d8a,0x7a879d8a,0x7a879d8a,0x7a879d8a,
    0x7a879d8a,0x7a879d8a,0x7a879d8a,0x7a879d8a,0x7a879d8a,0x7a879d8a,0x7a879d8a,0x7a879d8a,
    0x7a879d8a,0x7a879d8a,0x7a879d8a,0x7a879d8a,0x7a879d8a,0x7a879d8a,0x7a879d8a,0x7a879d8a,
    0x7a879d8a,0x7a879d8a,0x7a879d8a,0x7a879d8a,0x7a879d8a,0x7a879d8a,0x7a879d8a,0x7a879d8a,
    0x7a879d8a,0x7a879d8a,0x7a879d8a,0x7a879d8a,0x7a879d8a,0x7a879d8a,0x7a879d8a,0x7a879d8a,
    0x7a879d8a,0x7a879d8a,0x7a879d8a,0x7a879d8a,0x7a879d8a,0x7a879d8a,0x7a879d8a,0x7a879d8a
};

// ---------------- 原方案（直接实现，突出清晰） ----------------
void expand_orig(const uint8_t* block, uint32_t W[68]){
    for(int i=0;i<16;i++){
        W[i] = (uint32_t(block[4*i]) << 24) | (uint32_t(block[4*i+1]) << 16) |
               (uint32_t(block[4*i+2]) << 8) | uint32_t(block[4*i+3]);
    }
    for(int i=16;i<68;i++){
        W[i] = P1_u(W[i-16] ^ W[i-9] ^ ROTL32(W[i-3],15)) ^ ROTL32(W[i-13],7) ^ W[i-6];
    }
}

void compress_orig(uint32_t V[8], const uint8_t block[64]){
    uint32_t W[68];
    expand_orig(block, W);
    uint32_t W1[64];
    for(int j=0;j<64;j++) W1[j] = W[j] ^ W[j+4];

    uint32_t A=V[0], B=V[1], C=V[2], D=V[3];
    uint32_t E=V[4], F=V[5], G=V[6], H=V[7];

    for(int j=0;j<64;j++){
        uint32_t tmpT = ROTL32(T_BASE[j], j);
        uint32_t SS1 = ROTL32((ROTL32(A,12) + E + tmpT) & 0xFFFFFFFF, 7);
        uint32_t SS2 = SS1 ^ ROTL32(A,12);
        uint32_t TT1 = (FF(A,B,C,j) + D + SS2 + W1[j]) & 0xFFFFFFFF;
        uint32_t TT2 = (GG(E,F,G,j) + H + SS1 + W[j]) & 0xFFFFFFFF;

        D=C; C=ROTL32(B,9); B=A; A=TT1;
        H=G; G=ROTL32(F,19); F=E; E=P0(TT2);
    }

    for(int i=0;i<8;i++) V[i] ^= (i==0?A:i==1?B:i==2?C:i==3?D:i==4?E:i==5?F:i==6?G:H);
}

void sm3_original(const uint8_t* input, size_t len, uint8_t out[32]){
    // 填充（标准大端 length）
    uint64_t l = (uint64_t)len * 8;
    size_t k = (448 - (l + 1)) % 512;
    if ((l + 1) % 512 > 448) k += 512;
    size_t padded_len = (l + 1 + k + 64) / 8;

    std::vector<uint8_t> buf(padded_len, 0);
    if(len) memcpy(buf.data(), input, len);
    buf[len] = 0x80;
    for(int i=0;i<8;i++) buf[padded_len - 1 - i] = (l >> (8*i)) & 0xFF;

    uint32_t V[8];
    memcpy(V, IV, sizeof(IV));
    size_t blocks = padded_len / 64;
    for(size_t i=0;i<blocks;i++){
        compress_orig(V, buf.data() + i*64);
    }
    for(int i=0;i<8;i++){
        out[4*i] = (V[i] >> 24) & 0xFF;
        out[4*i+1] = (V[i] >> 16) & 0xFF;
        out[4*i+2] = (V[i] >> 8) & 0xFF;
        out[4*i+3] = V[i] & 0xFF;
    }
}

// ---------------- 优化方案（在保证等价的前提下加速） ----------------

// 优化思路：
// 1) 预先计算 rotT[j] = ROTL32(T_BASE[j], j) 并缓存；
// 2) 将消息扩展和 W1 计算合并为一个函数输出 W 和 W1；
// 3) inline / 少量临时变量，减少循环内重复工作。

void expand_opt(const uint8_t* block, uint32_t W[68], uint32_t W1[64]){
    for(int i=0;i<16;i++){
        W[i] = (uint32_t(block[4*i]) << 24) | (uint32_t(block[4*i+1]) << 16) |
               (uint32_t(block[4*i+2]) << 8) | uint32_t(block[4*i+3]);
    }
    for(int i=16;i<68;i++){
        uint32_t tmp = W[i-16] ^ W[i-9] ^ ROTL32(W[i-3],15);
        W[i] = P1_u(tmp) ^ ROTL32(W[i-13],7) ^ W[i-6];
    }
    for(int i=0;i<64;i++) W1[i] = W[i] ^ W[i+4];
}

void compress_opt(uint32_t V[8], const uint8_t block[64], const uint32_t rotT[64]){
    uint32_t W[68], W1[64];
    expand_opt(block, W, W1);

    uint32_t A=V[0], B=V[1], C=V[2], D=V[3];
    uint32_t E=V[4], F=V[5], G=V[6], H=V[7];

    for(int j=0;j<64;j++){
        // 使用预计算 rotT[j]（避免循环内每次 ROTL(T_BASE[j], j)）
        uint32_t SS1 = ROTL32((ROTL32(A,12) + E + rotT[j]) & 0xFFFFFFFF, 7);
        uint32_t SS2 = SS1 ^ ROTL32(A,12);
        uint32_t TT1 = (FF(A,B,C,j) + D + SS2 + W1[j]) & 0xFFFFFFFF;
        uint32_t TT2 = (GG(E,F,G,j) + H + SS1 + W[j]) & 0xFFFFFFFF;

        D=C; C=ROTL32(B,9); B=A; A=TT1;
        H=G; G=ROTL32(F,19); F=E; E=P0(TT2);
    }

    V[0]^=A; V[1]^=B; V[2]^=C; V[3]^=D; V[4]^=E; V[5]^=F; V[6]^=G; V[7]^=H;
}

void sm3_optimized(const uint8_t* input, size_t len, uint8_t out[32]){
    // 先准备 rotT 表
    uint32_t rotT[64];
    for(int j=0;j<64;j++) rotT[j] = ROTL32(T_BASE[j], j);

    // 填充
    uint64_t l = (uint64_t)len * 8;
    size_t k = (448 - (l + 1)) % 512;
    if ((l + 1) % 512 > 448) k += 512;
    size_t padded_len = (l + 1 + k + 64) / 8;

    std::vector<uint8_t> buf(padded_len, 0);
    if(len) memcpy(buf.data(), input, len);
    buf[len] = 0x80;
    for(int i=0;i<8;i++) buf[padded_len - 1 - i] = (l >> (8*i)) & 0xFF;

    uint32_t V[8];
    memcpy(V, IV, sizeof(IV));
    size_t blocks = padded_len / 64;
    for(size_t i=0;i<blocks;i++){
        compress_opt(V, buf.data() + i*64, rotT);
    }
    for(int i=0;i<8;i++){
        out[4*i] = (V[i] >> 24) & 0xFF;
        out[4*i+1] = (V[i] >> 16) & 0xFF;
        out[4*i+2] = (V[i] >> 8) & 0xFF;
        out[4*i+3] = V[i] & 0xFF;
    }
}

// ---------------- 辅助：打印与测试 ----------------
void print_hex(const uint8_t* d){
    for(int i=0;i<32;i++) printf("%02x", d[i]);
    printf("\n");
}

int main(){
    const char* msg = "abc";
    uint8_t out1[32], out2[32];

    // 运行原方案
    auto t1 = std::chrono::high_resolution_clock::now();
    sm3_original(reinterpret_cast<const uint8_t*>(msg), strlen(msg), out1);
    auto t2 = std::chrono::high_resolution_clock::now();
    double us_orig = std::chrono::duration<double, std::micro>(t2 - t1).count();

    // 运行优化方案
    auto t3 = std::chrono::high_resolution_clock::now();
    sm3_optimized(reinterpret_cast<const uint8_t*>(msg), strlen(msg), out2);
    auto t4 = std::chrono::high_resolution_clock::now();
    double us_opt = std::chrono::duration<double, std::micro>(t4 - t3).count();

    std::cout << "原方案输出: "; print_hex(out1);
    std::cout << "原方案耗时(us): " << us_orig << std::endl;
    std::cout << "优化方案输出: "; print_hex(out2);
    std::cout << "优化方案耗时(us): " << us_opt << std::endl;

    if (memcmp(out1,out2,32)==0) std::cout << "两个方案输出一致 (正确性通过)" << std::endl;
    else std::cout << "两个方案输出不一致！" << std::endl;

    // 打印标准向量以便比对（SM3( "abc" )标准输出）
    std::cout << "标准向量: 66c7f0f462eeedd9d1f2d46bdc10e4e24167c4875cf2f7a2297da02b8f4ba8e0" << std::endl;

    return 0;
}
