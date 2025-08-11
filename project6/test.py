import time
import secrets
import string

# 生成测试数据
def random_email():
    name = ''.join(secrets.choice(string.ascii_lowercase) for _ in range(8))
    return f"{name}@example.com"

def gen_test_data(n1=1000, n2=1000, overlap=200):
    # 生成 P1
    P1 = [random_email() for _ in range(n1 - overlap)]
    overlap_emails = [random_email() for _ in range(overlap)]
    P1.extend(overlap_emails)
    # 生成 P2
    P2 = [(random_email(), secrets.randbelow(100)) for _ in range(n2 - overlap)]
    P2.extend([(em, secrets.randbelow(100)) for em in overlap_emails])
    return P1, P2

# 假设原始和优化版本函数如下：
from one import run_pisum_original
from enhance import run_pisum_optimized

def benchmark(n1=1000, n2=1000, overlap=200, group_bits=128, paillier_bits=256):
    P1, P2 = gen_test_data(n1, n2, overlap)

    # 原始版本
    start = time.perf_counter()
    res_orig = run_pisum_original(P1, P2, seed=b"seed1")
    t_orig = time.perf_counter() - start

    # 优化版本
    start = time.perf_counter()
    res_opt = run_pisum_optimized(P1, P2, group_bits=group_bits, paillier_bits=paillier_bits, seed=b"seed1")
    t_opt = time.perf_counter() - start

    # 输出对比
    print(f"=== 运行对比（数据规模：P1={n1}, P2={n2}, overlap={overlap}） ===")
    print(f"原始版本耗时: {t_orig:.6f} s")
    print(f"优化版本耗时: {t_opt:.6f} s")
    print(f"加速比: {t_orig / t_opt:.2f}x")
    print(f"交集大小一致: {res_orig['intersection_size'] == res_opt['intersection_size']}")
    print(f"交集求和一致: {res_orig['intersection_sum'] == res_opt['intersection_sum']}")

if __name__ == "__main__":
    benchmark()
