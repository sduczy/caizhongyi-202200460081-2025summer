import hashlib
import random
from math import gcd

# ---------------------------
# 简单 Paillier (玩具) 实现
# ---------------------------
def lcm(a,b): return a//gcd(a,b)*b

def invmod(a, m):
    # 模逆
    return pow(a, -1, m)

class Paillier:
    @staticmethod
    def keygen(bits=256):
        # 注意：演示用的小素数；实际使用中请使用安全素数
        # 生成 p,q 素数（简单方法）
        def gen_prime(nbits):
            while True:
                r = random.getrandbits(nbits) | 1
                if pow(2, r-1, r) == 1 and r % 2 == 1:
                    # 非常弱的素性检查；仅适用于演示
                    return r
        # 演示使用小素数（不安全）
        p = 2**((bits//2)-1) + 1
        q = 2**((bits//2)-1) + 3
        # 回退到小素数以避免演示中缓慢的素性检查
        # 计算 n
        n = p * q
        lam = lcm(p-1, q-1)
        g = n + 1
        mu = invmod((pow(g, lam, n*n) - 1)//n, n)
        return ( (n, g), (lam, mu) )

    @staticmethod
    def enc(pk, m):
        n, g = pk
        n2 = n*n
        r = random.randrange(1, n)
        c = (pow(g, m, n2) * pow(r, n, n2)) % n2
        return c

    @staticmethod
    def dec(sk, pk, c):
        n, g = pk
        lam, mu = sk
        n2 = n*n
        x = (pow(c, lam, n2) - 1) // n
        return (x * mu) % n

    @staticmethod
    def add(pk, c1, c2):
        n2 = pk[0]*pk[0]
        return (c1 * c2) % n2

    @staticmethod
    def refresh(pk, c):
        # 重新随机化：乘以 r^n
        n = pk[0]
        n2 = n*n
        r = random.randrange(1, n)
        return (c * pow(r, n, n2)) % n2

# ------------------------------
# 工具：哈希到群元素
# ------------------------------
def H_to_group(u, p, g, seed=b''):
    # 将字节 u（或字符串）哈希为整数并转换为群元素 g^{hash}
    if isinstance(u, str): u = u.encode()
    h = hashlib.sha256(seed + u).digest()
    h_int = int.from_bytes(h, 'big')
    # 映射到 [1, p-1) 中的指数
    exp = (h_int % (p-1)) or 1
    return pow(g, exp, p)  # 模 p 乘法群中的元素

# ------------------------------
# 玩具乘法群参数
# ------------------------------
# 小演示素数（不安全）。实际使用：使用安全的 2048 位素数或 ECC。
P = 0xFFFFFFFBFFFFFFFDCFFFFFFFFFFFFFFF  # **占位符小素数** (演示；不安全)
# 我们将改为通过简单方法为演示选择一个随机安全素数：
def find_demo_prime(bits=64):
    # 找到一个随机素数（非密码学安全）
    while True:
        candidate = random.getrandbits(bits) | 1
        if candidate % 2 == 0: candidate += 1
        # 微小素性测试
        if pow(2, candidate-1, candidate) == 1:
            return candidate

P = find_demo_prime(64)
# 选择生成元 g
G = 2
# ------------------------------
# 协议（两方的本地模拟）
# ------------------------------
def run_pisum_poc(P1_set, P2_pairs, seed=b'runseed'):
    # P1_set: 标识符列表（字符串）
    # P2_pairs: (标识符, 整数值) 列表
    # 设置：
    k1 = random.randrange(2, P-1)
    k2 = random.randrange(2, P-1)
    # P2（输出方）生成 HE 密钥对
    pk, sk = Paillier.keygen(bits=128)  # 玩具小密钥
    # --- 第1轮 (P1) ---
    Z1 = []
    for v in P1_set:
        h = H_to_group(v, P, G, seed)
        Z1.append(pow(h, k1, P))
    random.shuffle(Z1)
    # 发送 Z1 -> P2
    # --- 第2轮 (P2) ---
    # 1) 对每个收到的 H(vi)^{k1}，计算 ^k2 -> H(vi)^{k1k2}
    Z2 = [pow(elem, k2, P) for elem in Z1]
    random.shuffle(Z2)  # P2 发送打乱的 Z 回 P1
    # 3) 对每个 (wj, tj) 计算 H(wj)^{k2} 和 AEnc(tj)
    W_list = []
    for (w, t) in P2_pairs:
        hw_k2 = pow(H_to_group(w, P, G, seed), k2, P)
        ct = Paillier.enc(pk, t)
        W_list.append( (hw_k2, ct) )
    random.shuffle(W_list)
    # 发送 (W_list) -> P1 连同 Z2
    # --- 第3轮 (P1) ---
    # 对每个 (H(wj)^{k2}, ct)，计算 ^k1 -> H(wj)^{k1k2}
    W_transformed = [ (pow(hw, k1, P), ct) for (hw, ct) in W_list ]
    # 计算交集：变换后的 H(wj) 在 Z2 中的索引
    set_Z2 = set(Z2)
    intersect_ciphertexts = [ct for (h, ct) in W_transformed if h in set_Z2]
    C = len(intersect_ciphertexts)
    # 同态求和密文
    if len(intersect_ciphertexts) == 0:
        sum_ct = Paillier.enc(pk, 0)
    else:
        sum_ct = intersect_ciphertexts[0]
        for ct in intersect_ciphertexts[1:]:
            sum_ct = Paillier.add(pk, sum_ct, ct)
    # 重新随机化并发送给 P2
    sum_ct_refreshed = Paillier.refresh(pk, sum_ct)
    # --- 输出 (P2) ---
    S = Paillier.dec(sk, pk, sum_ct_refreshed)
    return {'intersection_size': C, 'intersection_sum': S}

# ------------------------------
# 演示运行
# ------------------------------
if __name__ == "__main__":
    # 示例数据
    P1 = ["alice@example.com", "bob@example.com", "carol@example.com", "dan@example.com"]
    P2 = [("eve@example.com", 10), ("bob@example.com", 25), ("carol@example.com", 5)]
    out = run_pisum_poc(P1, P2, seed=b'myseed123')
    print("交集大小:", out['intersection_size'])
    print("交集和:", out['intersection_sum'])  # 期望 25 + 5 = 30
