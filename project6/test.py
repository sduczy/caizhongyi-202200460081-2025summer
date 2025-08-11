#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Two implementations + comparison harness for DDH-based Private Intersection-Sum PoC.

- ToyPaillier: 修正后的“原版”教学实现（使用简单 Miller-Rabin 生成素数以保证解密正确）
- OptimizedPaillier: 更完善的实现（使用 secrets、缓存 n^2 等优化）
- compare: 在相同输入上运行多次，比较 correctness/time/size
注意：仍为教学 PoC，切勿用于生产环境。
"""

from __future__ import annotations
import hashlib
import random
import secrets
import time
import math
from typing import List, Tuple, Optional

# ----------------------------
# Utility: Miller-Rabin (shared)
# ----------------------------
def is_probable_prime(n: int, k: int = 8) -> bool:
    if n < 2:
        return False
    # small primes quick check
    small_primes = [2,3,5,7,11,13,17,19,23,29]
    for p in small_primes:
        if n % p == 0:
            return n == p
    # write n-1 = d * 2^s
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for _ in range(k):
        a = secrets.randbelow(n - 3) + 2
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

def gen_prime_bits(bits: int = 128) -> int:
    assert bits >= 16
    while True:
        cand = secrets.randbits(bits) | (1 << (bits - 1)) | 1
        if is_probable_prime(cand, k=8):
            return cand

# ----------------------------
# ToyPaillier (修正后的“原版”)
# ----------------------------
class ToyPaillier:
    """Toy Paillier for demo but corrected so keys are valid."""
    @staticmethod
    def keygen(bits: int = 128) -> Tuple[Tuple[int,int], Tuple[int,int]]:
        # use small-ish primes but actually prime (Miller-Rabin)
        p = gen_prime_bits(bits // 2)
        q = gen_prime_bits(bits // 2)
        while q == p:
            q = gen_prime_bits(bits // 2)
        n = p * q
        lam = (p - 1) * (q - 1) // math.gcd(p - 1, q - 1)
        g = n + 1
        n2 = n * n
        L = (pow(g, lam, n2) - 1) // n
        mu = pow(L, -1, n)  # modular inverse exists if p,q primes
        return (n, g), (lam, mu)

    @staticmethod
    def enc(pk: Tuple[int,int], m: int) -> int:
        n, g = pk
        n2 = n * n
        # toy: use random from python.random (matches "original" flavor)
        r = random.randrange(1, n)
        return (pow(g, m, n2) * pow(r, n, n2)) % n2

    @staticmethod
    def dec(sk: Tuple[int,int], pk: Tuple[int,int], c: int) -> int:
        n, g = pk
        lam, mu = sk
        n2 = n * n
        x = pow(c, lam, n2)
        L = (x - 1) // n
        return (L * mu) % n

    @staticmethod
    def add(pk: Tuple[int,int], c1: int, c2: int) -> int:
        n = pk[0]
        n2 = n * n
        return (c1 * c2) % n2

    @staticmethod
    def refresh(pk: Tuple[int,int], c: int) -> int:
        n = pk[0]
        n2 = n * n
        r = random.randrange(1, n)
        return (c * pow(r, n, n2)) % n2

# ----------------------------
# OptimizedPaillier (更完善版)
# ----------------------------
class OptimizedPaillierKey:
    def __init__(self, n:int, g:int, lam:int, mu:int):
        self.n = n
        self.g = g
        self.lam = lam
        self.mu = mu
        self.n2 = n * n

class OptimizedPaillier:
    @staticmethod
    def keygen(bits: int = 512) -> Tuple[OptimizedPaillierKey, Tuple[int,int]]:
        p = gen_prime_bits(bits // 2)
        q = gen_prime_bits(bits // 2)
        while q == p:
            q = gen_prime_bits(bits // 2)
        n = p * q
        lam = (p - 1) * (q - 1) // math.gcd(p - 1, q - 1)
        g = n + 1
        n2 = n * n
        L = (pow(g, lam, n2) - 1) // n
        mu = pow(L, -1, n)
        key = OptimizedPaillierKey(n=n, g=g, lam=lam, mu=mu)
        return key, (n, g)

    @staticmethod
    def enc(pk: Tuple[int,int], m: int) -> int:
        n, g = pk
        n2 = n * n
        r = secrets.randbelow(n - 1) + 1
        return (pow(g, m, n2) * pow(r, n, n2)) % n2

    @staticmethod
    def dec(key: OptimizedPaillierKey, c: int) -> int:
        n = key.n
        n2 = key.n2
        x = pow(c, key.lam, n2)
        L = (x - 1) // n
        return (L * key.mu) % n

    @staticmethod
    def add(pk: Tuple[int,int], c1: int, c2: int) -> int:
        n = pk[0]
        n2 = n * n
        return (c1 * c2) % n2

    @staticmethod
    def rerandomize(pk: Tuple[int,int], c: int) -> int:
        n = pk[0]
        n2 = n * n
        r = secrets.randbelow(n - 1) + 1
        return (c * pow(r, n, n2)) % n2

# ----------------------------
# Hash to group (shared)
# ----------------------------
def H_to_group(u: bytes, p: int, g: int, seed: Optional[bytes] = None) -> int:
    if seed is None:
        seed = b''
    digest = hashlib.sha256(seed + u).digest()
    h_int = int.from_bytes(digest, 'big')
    exp = (h_int % (p - 1)) or 1
    return pow(g, exp, p)

# ----------------------------
# Demo group (small, for PoC)
# ----------------------------
def find_demo_prime(bits: int = 64) -> int:
    # NOT cryptographically secure; just to have a prime-ish p for the multiplicative group
    return gen_prime_bits(bits)

# ----------------------------
# Protocol implementations
# ----------------------------
def run_pisum_toy(P1_set: List[str], P2_pairs: List[Tuple[str,int]], seed: bytes = b'runseed'):
    # Use toy Paillier
    P = find_demo_prime(64)
    G = 2
    k1 = random.randrange(2, P-1)
    k2 = random.randrange(2, P-1)
    pk, sk = ToyPaillier.keygen(bits=128)  # pk=(n,g), sk=(lam,mu)

    # R1: P1 -> P2
    Z1 = [pow(H_to_group(v.encode(), P, G, seed), k1, P) for v in P1_set]
    random.shuffle(Z1)

    # R2: P2 -> P1
    Z2 = [pow(z, k2, P) for z in Z1]
    random.shuffle(Z2)

    W_list = []
    for (w, t) in P2_pairs:
        hw_k2 = pow(H_to_group(w.encode(), P, G, seed), k2, P)
        ct = ToyPaillier.enc(pk, t)
        W_list.append((hw_k2, ct))
    random.shuffle(W_list)

    # R3: P1 -> P2
    W_transformed = [(pow(hw, k1, P), ct) for (hw, ct) in W_list]
    set_Z2 = set(Z2)
    intersect_ciphertexts = [ct for (h, ct) in W_transformed if h in set_Z2]

    C = len(intersect_ciphertexts)
    if C == 0:
        sum_ct = ToyPaillier.enc(pk, 0)
    else:
        sum_ct = intersect_ciphertexts[0]
        for ct in intersect_ciphertexts[1:]:
            sum_ct = ToyPaillier.add(pk, sum_ct, ct)
    sum_ct_refreshed = ToyPaillier.refresh(pk, sum_ct)
    S = ToyPaillier.dec(sk, pk, sum_ct_refreshed)
    return {"intersection_size": C, "intersection_sum": S, "paillier_n_bits": pk[0].bit_length(), "ciphertext_bits": sum_ct_refreshed.bit_length()}

def run_pisum_opt(P1_set: List[str], P2_pairs: List[Tuple[str,int]], group_bits: int = 128, paillier_bits: int = 256, seed: bytes = b'runseed'):
    p = gen_prime_bits(group_bits)
    g = 2
    k1 = secrets.randbelow(p - 3) + 2
    k2 = secrets.randbelow(p - 3) + 2
    key_obj, pk = OptimizedPaillier.keygen(bits=paillier_bits)  # key_obj holds lam,mu etc.

    Z1 = [pow(H_to_group(v.encode(), p, g, seed), k1, p) for v in P1_set]
    random.shuffle(Z1)
    Z2 = [pow(z, k2, p) for z in Z1]
    random.shuffle(Z2)

    W_list = []
    for (w, t) in P2_pairs:
        hw_k2 = pow(H_to_group(w.encode(), p, g, seed), k2, p)
        ct = OptimizedPaillier.enc(pk, t)
        W_list.append((hw_k2, ct))
    random.shuffle(W_list)

    W_transformed = [(pow(hw, k1, p), ct) for (hw, ct) in W_list]
    set_Z2 = set(Z2)
    intersect_ciphertexts = [ct for (h, ct) in W_transformed if h in set_Z2]

    C = len(intersect_ciphertexts)
    if C == 0:
        sum_ct = OptimizedPaillier.enc(pk, 0)
    else:
        sum_ct = intersect_ciphertexts[0]
        for ct in intersect_ciphertexts[1:]:
            sum_ct = OptimizedPaillier.add(pk, sum_ct, ct)

    sum_ct_refreshed = OptimizedPaillier.rerandomize(pk, sum_ct)
    S = OptimizedPaillier.dec(key_obj, sum_ct_refreshed)
    return {"intersection_size": C, "intersection_sum": S, "paillier_n_bits": key_obj.n.bit_length(), "ciphertext_bits": sum_ct_refreshed.bit_length()}

# ----------------------------
# Comparison harness
# ----------------------------
def compare(P1, P2, trials=4):
    results = []
    for i in range(trials):
        # toy
        t0 = time.perf_counter()
        out_toy = run_pisum_toy(P1, P2, seed=b"seed_example")
        t1 = time.perf_counter()

        # optimized
        t2 = time.perf_counter()
        out_opt = run_pisum_opt(P1, P2, group_bits=128, paillier_bits=256, seed=b"seed_example")
        t3 = time.perf_counter()

        expected_sum = sum(v for (k,v) in P2 if k in set(P1))

        row = {
            "trial": i+1,
            "toy_time_s": t1 - t0,
            "opt_time_s": t3 - t2,
            "toy_size": out_toy["intersection_size"],
            "opt_size": out_opt["intersection_size"],
            "toy_sum": out_toy["intersection_sum"],
            "opt_sum": out_opt["intersection_sum"],
            "expected_sum": expected_sum,
            "toy_paillier_n_bits": out_toy["paillier_n_bits"],
            "opt_paillier_n_bits": out_opt["paillier_n_bits"],
            "toy_ciphertext_bits": out_toy["ciphertext_bits"],
            "opt_ciphertext_bits": out_opt["ciphertext_bits"],
        }
        results.append(row)

    # summary
    toy_times = [r["toy_time_s"] for r in results]
    opt_times = [r["opt_time_s"] for r in results]
    summary = {
        "trials": trials,
        "toy_mean_s": sum(toy_times)/len(toy_times),
        "opt_mean_s": sum(opt_times)/len(opt_times),
        "match_size_every_run": all(r["toy_size"] == r["opt_size"] for r in results),
        "match_sum_every_run": all(r["toy_sum"] == r["opt_sum"] == r["expected_sum"] for r in results),
    }
    return results, summary

# ----------------------------
# Demo run
# ----------------------------
if __name__ == "__main__":
    P1 = ["alice@example.com", "bob@example.com", "carol@example.com", "dan@example.com"]
    P2 = [("eve@example.com", 10), ("bob@example.com", 25), ("carol@example.com", 5)]
    results, summary = compare(P1, P2, trials=4)

    print("=== Summary ===")
    for k,v in summary.items():
        print(f"{k}: {v}")
    print("\n=== Per-trial results ===")
    for r in results:
        print(r)
