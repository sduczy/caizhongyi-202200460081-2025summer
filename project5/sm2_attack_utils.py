#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SM2 攻击辅助工具：统一复用，减少重复代码
"""

from __future__ import annotations

import hashlib
from typing import Tuple


# SM2 椭圆曲线阶
SM2_N = 0xFFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFF7203DF6B21C6052B53BBF40939D54123


def parse_signature_hex(signature: str) -> Tuple[int, int]:
    """将签名解析为 (r, s) 整数对。

    支持两种格式：
    - 逗号分隔: "{r_hex},{s_hex}"
    - 纯拼接: 128位十六进制，前64为r，后64为s
    """
    sig = signature.strip()
    if "," in sig:
        r_hex, s_hex = sig.split(",", 1)
        return int(r_hex, 16), int(s_hex, 16)
    # 默认按拼接解析
    if len(sig) < 128:
        raise ValueError("签名长度不足，无法按拼接格式解析")
    return int(sig[:64], 16), int(sig[64:128], 16)


def mod_inverse(a: int, m: int) -> int:
    """计算 a 在模 m 下的乘法逆元（扩展欧几里得）。"""
    def extended_gcd(x: int, y: int):
        if x == 0:
            return y, 0, 1
        g, x1, y1 = extended_gcd(y % x, x)
        return g, y1 - (y // x) * x1, x1

    g, x, _ = extended_gcd(a % m, m)
    if g != 1:
        raise ValueError("模逆不存在")
    return x % m


def recover_private_key_from_nonce_reuse(
    msg1: bytes, msg2: bytes, signature1: str, signature2: str, n: int = SM2_N
) -> str:
    """SM2随机数重用下恢复私钥（无需 r 相等）。

    SM2签名：
      r = (e + x1) mod n,
      s = (1 + d)^-1 * (k - r d) mod n
    同一k签两个消息得到 (r1,s1), (r2,s2)：
      d = (s2 - s1) / (r1 - r2 + s1 - s2) mod n
    该公式不依赖哈希值。
    """
    r1, s1 = parse_signature_hex(signature1)
    r2, s2 = parse_signature_hex(signature2)
    numerator = (s2 - s1) % n
    denominator = (r1 - r2 + s1 - s2) % n
    inv = mod_inverse(denominator, n)
    d = (numerator * inv) % n
    return hex(d)[2:].zfill(64)


def recover_private_key_from_predictable_nonce(
    msg: bytes, signature: str, k_hex: str, n: int = SM2_N
) -> str:
    """SM2可预测随机数下恢复私钥。

    由 s = (1 + d)^-1 * (k - r d) 推得：
      d = (k - s) / (s + r) mod n
    与消息哈希无关。
    """
    r, s = parse_signature_hex(signature)
    k_int = int(k_hex, 16)
    numerator = (k_int - s) % n
    inv = mod_inverse((s + r) % n, n)
    d = (numerator * inv) % n
    return hex(d)[2:].zfill(64)


