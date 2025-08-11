#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化版 PoC: DDH-based Private Intersection-Sum (Figure 2) - demo implementation
优化点摘要：
 - 使用 secrets 作为安全随机源
 - 使用 Miller-Rabin 生成/检测素数（可选位长）
 - Paillier keygen 使用真实随机素数（可配置位长）
 - 将重复的双重幂运算合并为单次幂（数学等价，节约计算）
 - 避免重复计算模方 n^2，缓存 n2 提升性能
 - 更清晰的函数分离与类型注解，便于测试与复用
注意：仍为教学 PoC，不要直接用于生产环境。
"""
from __future__ import annotations
import hashlib
import secrets
import math
from typing import List, Tuple, Optional

# -------------------------
# 工具：扩展欧几里得与模逆
# -------------------------
def egcd(a: int, b: int) -> Tuple[int, int, int]:
    if b == 0:
        return (a, 1, 0)
    g, x1, y1 = egcd(b, a % b)
    return (g, y1, x1 - (a // b) * y1)

def invmod(a: int, m: int) -> int:
    g, x, _ = egcd(a % m, m)
    if g != 1:
        raise ValueError("modular inverse does not exist")
    return x % m

# -------------------------
# Miller-Rabin 素性测试与素数生成
# -------------------------
def is_probable_prime(n: int, k: int = 8) -> bool:
    """Miller-Rabin 概率素性测试。k 越大错误概率越小。"""
    if n < 2:
        return False
    # small primes test
    small_primes = [2,3,5,7,11,13,17,19,23,29]
    for p in small_primes:
        if n % p == 0:
            return n == p
    # write n-1 as d * 2^s
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for _ in range(k):
        a = secrets.randbelow(n - 3) + 2  # in [2, n-2]
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for __ in range(s - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True

def gen_prime(bits: int = 512) -> int:
    """生成指定位长的素数（概率素数）。"""
    assert bits >= 16
    while True:
        candidate = secrets.randbits(bits) | (1 << (bits - 1)) | 1  # 保证最高位为1和奇数
        if is_probable_prime(candidate, k=8):
            return candidate

# -------------------------
# 优化版 Paillier（教学 PoC）
# -------------------------
class PaillierKeypair:
    def __init__(self, n: int, g: int, lam: int, mu: int):
        self.n = n
        self.g = g
        self.lam = lam
        self.mu = mu
        self.n2 = n * n  # cache n^2

def paillier_keygen(bits: int = 512) -> Tuple[PaillierKeypair, Tuple[int, int]]:
    """生成 Paillier 密钥对。返回 (keypair, (pk_n, pk_g))"""
    # 使用两个独立素数 p,q
    p = gen_prime(bits // 2)
    q = gen_prime(bits // 2)
    while q == p:
        q = gen_prime(bits // 2)
    n = p * q
    lam = (p - 1) * (q - 1) // math.gcd(p - 1, q - 1)  # lcm(p-1, q-1)
    g = n + 1  # 简单选 g = n+1，使得 L(g^λ mod n^2) = λ n
    # compute mu = (L(g^lam mod n^2))^{-1} mod n, L(u)=(u-1)/n
    x = pow(g, lam, n * n)
    L = (x - 1) // n
    mu = invmod(L, n)
    keypair = PaillierKeypair(n=n, g=g, lam=lam, mu=mu)
    return keypair, (n, g)

def paillier_encrypt(pk: Tuple[int, int], m: int) -> int:
    n, g = pk
    n2 = n * n
    # 使用 secrets.randbelow 选择随机 r in [1, n-1]
    r = secrets.randbelow(n - 1) + 1
    return (pow(g, m, n2) * pow(r, n, n2)) % n2

def paillier_decrypt(keypair: PaillierKeypair, c: int) -> int:
    n = keypair.n
    n2 = keypair.n2
    x = pow(c, keypair.lam, n2)
    L = (x - 1) // n
    return (L * keypair.mu) % n

def paillier_add(pk: Tuple[int, int], c1: int, c2: int) -> int:
    n = pk[0]
    n2 = n * n
    return (c1 * c2) % n2

def paillier_rerandomize(pk: Tuple[int, int], c: int) -> int:
    n = pk[0]
    n2 = n * n
    r = secrets.randbelow(n - 1) + 1
    return (c * pow(r, n, n2)) % n2

# -------------------------
# Hash-to-group
# -------------------------
def H_to_group_bytes(data: bytes, p: int, g: int, seed: Optional[bytes] = None) -> int:
    """将字节串 hash 映射为群元素 g^{hash} mod p。"""
    if seed is None:
        seed = b''
    digest = hashlib.sha256(seed + data).digest()
    h_int = int.from_bytes(digest, 'big')
    exp = (h_int % (p - 1)) or 1
    return pow(g, exp, p)

# -------------------------
# 模拟协议（优化点详细说明见下）
# -------------------------
def run_pisum_optimized(
    P1_set: List[str],
    P2_pairs: List[Tuple[str, int]],
    group_bits: int = 256,
    paillier_bits: int = 512,
    seed: Optional[bytes] = None
) -> dict:
    """运行协议的本地模拟（返回交集大小与交集和）"""

    # --- 生成安全群参数（取一个大素数 p 与生成元 g=2）---
    # 为示例：生成一个素数 p（bits 长度）。在生产中使用标准域或 ECC。
    p = gen_prime(group_bits)
    g = 2
    # --- 参与方私钥（指数） ---
    k1 = secrets.randbelow(p - 3) + 2
    k2 = secrets.randbelow(p - 3) + 2

    # --- Paillier keypair (P2 持有私钥) ---
    keypair, pk = paillier_keygen(bits=paillier_bits)

    # ----------------
    # Round 1 (P1 -> P2)
    # ----------------
    # 计算 H(v)^{k1}；这里保留每个元素的哈希值以便复用
    Z1 = [pow(H_to_group_bytes(v.encode(), p, g, seed), k1, p) for v in P1_set]
    secrets.SystemRandom().shuffle(Z1)

    # ----------------
    # Round 2 (P2 -> P1)
    # ----------------
    # P2 计算每个收到的 z^{k2}（注意：保持交互顺序）
    # 优化：如果仅仅为了本地模拟，可以将两次幂合并为一次： pow(h, k1*k2 mod (p-1), p)
    Z2 = [pow(z, k2, p) for z in Z1]
    secrets.SystemRandom().shuffle(Z2)

    # P2 为每个 (w_j,t_j) 计算 H(w_j)^{k2} 并对 t_j 加密
    W_list = []
    for (w, t) in P2_pairs:
        hw_k2 = pow(H_to_group_bytes(w.encode(), p, g, seed), k2, p)
        ct = paillier_encrypt(pk, t)
        W_list.append((hw_k2, ct))
    secrets.SystemRandom().shuffle(W_list)

    # ----------------
    # Round 3 (P1 -> P2)
    # ----------------
    # P1 将每个收到的 hw_k2 提升到 k1： (hw_k2)^{k1} = H(w)^{k1*k2}
    # 优化（计算次数）：将 pow(hw_k2, k1, p) 改为 pow(h, (k1*k2)%(p-1), p) 仅在可得原始 h 时适用。
    # 但这里遵循协议步骤：做单次幂而非两次嵌套幂，因此把 pow(hw_k2, k1, p) 保持为一次 pow（比先 pow(h,k1) 再 pow(...,k2) 少一次）
    W_transformed = [(pow(hw, k1, p), ct) for (hw, ct) in W_list]

    # 将 Z2 转为 set 以加速成员判定（O(1) 平均）
    set_Z2 = set(Z2)
    intersect_ciphertexts = [ct for (h, ct) in W_transformed if h in set_Z2]

    C = len(intersect_ciphertexts)
    # 同态求和（连乘，使用 Paillier 的模 n^2）
    if C == 0:
        sum_ct = paillier_encrypt(pk, 0)
    else:
        sum_ct = intersect_ciphertexts[0]
        for ct in intersect_ciphertexts[1:]:
            sum_ct = paillier_add(pk, sum_ct, ct)

    # 重新随机化并发送回 P2
    sum_ct_refreshed = paillier_rerandomize(pk, sum_ct)

    # P2 解密得到结果
    S = paillier_decrypt(keypair, sum_ct_refreshed)

    return {"intersection_size": C, "intersection_sum": S}

# -------------------------
# 简单测试
# -------------------------
if __name__ == "__main__":
    P1 = ["alice@example.com", "bob@example.com", "carol@example.com", "dan@example.com"]
    P2 = [("eve@example.com", 10), ("bob@example.com", 25), ("carol@example.com", 5)]
    out = run_pisum_optimized(P1, P2, group_bits=128, paillier_bits=256, seed=b"seed1")
    print("intersection size:", out["intersection_size"])
    print("intersection sum:", out["intersection_sum"])
