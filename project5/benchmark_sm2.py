#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time

# 将工作目录切换到 project5/hggm，使得 hggm/ 路径与库中相对路径一致
CUR_DIR = os.path.dirname(__file__)
HGGMDIR = os.path.join(CUR_DIR, 'hggm')
os.chdir(HGGMDIR)
sys.path.insert(0, os.getcwd())

from hggm.SM2 import SM2  # noqa: E402


def ensure_kG_precomputed():
    # 如果预计算文件不存在，创建一个最小的占位来触发普通路径；真实预计算需要调用pre_kG()，但那会耗时较长
    # 这里我们通过存在与否对比开关来体现差异
    return os.path.exists('hggm/SM2_kG.bin')


def bench(label, rounds=200):
    sm2 = SM2(genkeypair=True)
    d, P = sm2.sk, sm2.pk
    msg = b"benchmark message for sm2"

    # 准备签名输入，避免测到随机IO
    sigs = []

    t0 = time.perf_counter()
    for _ in range(rounds):
        sigs.append(sm2.sign(msg))
    t1 = time.perf_counter()

    verify_ok = 0
    for sig in sigs:
        if sm2.verify(msg, sig, sm2.ID, P):
            verify_ok += 1
    t2 = time.perf_counter()

    enc = []
    for _ in range(rounds):
        ok, c = sm2.encrypt(msg, P)
        if ok:
            enc.append(c)
    t3 = time.perf_counter()

    dec_ok = 0
    for c in enc:
        ok, m = sm2.decrypt(c)
        if ok and m == msg:
            dec_ok += 1
    t4 = time.perf_counter()

    print(f"{label}")
    print(f"  sign  : {(t1 - t0):.4f}s  ({rounds} ops, {(rounds/(t1-t0)):.1f}/s)")
    print(f"  verify: {(t2 - t1):.4f}s  ({rounds} ops, {(rounds/(t2-t1)):.1f}/s, ok={verify_ok})")
    print(f"  enc   : {(t3 - t2):.4f}s  ({rounds} ops, {(rounds/(t3-t2)):.1f}/s)")
    print(f"  dec   : {(t4 - t3):.4f}s  ({len(enc)} ops, {(len(enc)/(t4-t3)):.1f}/s, ok={dec_ok})")


def main():
    # 基准1：无预计算（删除文件）
    existed = ensure_kG_precomputed()
    if existed:
        os.rename('hggm/SM2_kG.bin', 'hggm/SM2_kG.bin.bak')
    try:
        bench("无预计算 kG (普通标量乘)")
    finally:
        if existed:
            os.rename('hggm/SM2_kG.bin.bak', 'hggm/SM2_kG.bin')

    # 基准2：有预计算（存在文件）
    if not ensure_kG_precomputed():
        # 若确实没有文件，给出提示，不中断流程
        print("提示: 未发现 hggm/SM2_kG.bin，无法体现预计算加速对比。")
        return
    bench("启用预计算 kG (查表加速)")


if __name__ == '__main__':
    main()


