# sm4_fast.py
import struct

# 固定S盒
SBOX = [
    0xd6, 0x90, 0xe9, 0xfe, 0xcc, 0xe1, 0x3d, 0xb7,
    0x16, 0xb6, 0x14, 0xc2, 0x28, 0xfb, 0x2c, 0x05,
    0x2b, 0x67, 0x9a, 0x76, 0x2a, 0xbe, 0x04, 0xc3,
    0xaa, 0x44, 0x13, 0x26, 0x49, 0x86, 0x06, 0x99,
    0x9c, 0x42, 0x50, 0xf4, 0x91, 0xef, 0x98, 0x7a,
    0x33, 0x54, 0x0b, 0x43, 0xed, 0xcf, 0xac, 0x62,
    0xe4, 0xb3, 0x1c, 0xa9, 0xc9, 0x08, 0xe8, 0x95,
    0x80, 0xdf, 0x94, 0xfa, 0x75, 0x8f, 0x3f, 0xa6,
    0x47, 0x07, 0xa7, 0xfc, 0xf3, 0x73, 0x17, 0xba,
    0x83, 0x59, 0x3c, 0x19, 0xe6, 0x85, 0x4f, 0xa8,
    0x68, 0x6b, 0x81, 0xb2, 0x71, 0x64, 0xda, 0x8b,
    0xf8, 0xeb, 0x0f, 0x4b, 0x70, 0x56, 0x9d, 0x35,
    0x1e, 0x24, 0x0e, 0x5e, 0x63, 0x58, 0xd1, 0xa2,
    0x25, 0x22, 0x7c, 0x3b, 0x01, 0x21, 0x78, 0x87,
    0xd4, 0x00, 0x46, 0x57, 0x9f, 0xd3, 0x27, 0x52,
    0x4c, 0x36, 0x02, 0xe7, 0xa0, 0xc4, 0xc8, 0x9e,
    0xea, 0xbf, 0x8a, 0xd2, 0x40, 0xc7, 0x38, 0xb5,
    0xa3, 0xf7, 0xf2, 0xce, 0xf9, 0x61, 0x15, 0xa1,
    0xe0, 0xae, 0x5d, 0xa4, 0x9b, 0x34, 0x1a, 0x55,
    0xad, 0x93, 0x32, 0x30, 0xf5, 0x8c, 0xb1, 0xe3,
    0x1d, 0xf6, 0xe2, 0x2e, 0x82, 0x66, 0xca, 0x60,
    0xc0, 0x29, 0x23, 0xab, 0x0d, 0x53, 0x4e, 0x6f,
    0xd5, 0xdb, 0x37, 0x45, 0xde, 0xfd, 0x8e, 0x2f,
    0x03, 0xff, 0x6a, 0x72, 0x6d, 0x6c, 0x5b, 0x51,
    0x8d, 0x1b, 0xaf, 0x92, 0xbb, 0xdd, 0xbc, 0x7f,
    0x11, 0xd9, 0x5c, 0x41, 0x1f, 0x10, 0x5a, 0xd8,
    0x0a, 0xc1, 0x31, 0x88, 0xa5, 0xcd, 0x7b, 0xbd,
    0x2d, 0x74, 0xd0, 0x12, 0xb8, 0xe5, 0xb4, 0xb0,
    0x89, 0x69, 0x97, 0x4a, 0x0c, 0x96, 0x77, 0x7e,
    0x65, 0xb9, 0xf1, 0x09, 0xc5, 0x6e, 0xc6, 0x84,
    0x18, 0xf0, 0x7d, 0xec, 0x3a, 0xdc, 0x4d, 0x20,
    0x79, 0xee, 0x5f, 0x3e, 0xd7, 0xcb, 0x39, 0x48
]

# 固定参数
FK = [0xa3b1bac6, 0x56aa3350, 0x677d9197, 0xb27022dc]
CK = [0x00070e15 + 0x070e151c * i for i in range(32)]  # 简化写法

def rotl(x, n):
    return ((x << n) | (x >> (32 - n))) & 0xFFFFFFFF

def T(x):
    # inline tau and L
    a = (
        SBOX[(x >> 24) & 0xFF] << 24 |
        SBOX[(x >> 16) & 0xFF] << 16 |
        SBOX[(x >> 8) & 0xFF] << 8 |
        SBOX[x & 0xFF]
    )
    return a ^ rotl(a, 2) ^ rotl(a, 10) ^ rotl(a, 18) ^ rotl(a, 24)

# 优化：T_prime函数手动展开，提高速度
def T_prime(x):
    a = (
        SBOX[(x >> 24) & 0xFF] << 24 |
        SBOX[(x >> 16) & 0xFF] << 16 |
        SBOX[(x >> 8) & 0xFF] << 8 |
        SBOX[x & 0xFF]
    )
    return a ^ rotl(a, 13) ^ rotl(a, 23)

# 优化：使用局部变量代替列表操作，减少内存复制
def encrypt_block(block, rk):
    X0, X1, X2, X3 = struct.unpack(">4I", block)
    for i in range(32):
        tmp = X0 ^ T(X1 ^ X2 ^ X3 ^ rk[i])
        X0, X1, X2, X3 = X1, X2, X3, tmp
    return struct.pack(">4I", X3, X2, X1, X0)

def decrypt_block(block, rk):
    return encrypt_block(block, rk[::-1])

def key_expansion(key):
    # 优化：使用局部变量展开，避免不必要的数组操作
    MK = struct.unpack(">4I", key)
    K0, K1, K2, K3 = [MK[i] ^ FK[i] for i in range(4)]
    rk = []
    for i in range(32):
        temp = K1 ^ K2 ^ K3 ^ CK[i]
        a = (
            SBOX[(temp >> 24) & 0xFF] << 24 |
            SBOX[(temp >> 16) & 0xFF] << 16 |
            SBOX[(temp >> 8) & 0xFF] << 8 |
            SBOX[temp & 0xFF]
        )
        rk_i = K0 ^ a ^ rotl(a, 13) ^ rotl(a, 23)
        rk.append(rk_i)
        K0, K1, K2, K3 = K1, K2, K3, rk_i
    return rk

def pad(data):
    pad_len = 16 - (len(data) % 16)
    return data + bytes([pad_len] * pad_len)

def unpad(data):
    pad_len = data[-1]
    return data[:-pad_len]

# 优化：使用bytearray避免字符串拼接开销
def sm4_encrypt(data, key):
    rk = key_expansion(key)
    data = pad(data)
    ciphertext = bytearray()
    for i in range(0, len(data), 16):
        ciphertext.extend(encrypt_block(data[i:i+16], rk))
    return bytes(ciphertext)

def sm4_decrypt(data, key):
    rk = key_expansion(key)
    plaintext = bytearray()
    for i in range(0, len(data), 16):
        plaintext.extend(decrypt_block(data[i:i+16], rk))
    return unpad(bytes(plaintext))

# ==== 测试 ====
if __name__ == "__main__":
    key = b"0123456789abcdef"
    data = b"Hello SM4  2025summer"
    print("原文:", data)

    enc = sm4_encrypt(data, key)
    print("密文:", enc.hex())

    dec = sm4_decrypt(enc, key)
    print("解密:", dec)
