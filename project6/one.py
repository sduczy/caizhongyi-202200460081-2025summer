#!/usr/bin/env python3
# PoC: DDH-based Private Intersection-Sum (Figure 2) - toy/demo implementation
# Not production-secure. Uses multiplicative group mod p and a toy Paillier.

import hashlib
import random
from math import gcd

# ---------------------------
# Simple Paillier (toy) impl
# ---------------------------
def lcm(a,b): return a//gcd(a,b)*b

def invmod(a, m):
    # modular inverse
    return pow(a, -1, m)

class Paillier:
    @staticmethod
    def keygen(bits=256):
        # NOTE: tiny primes for demo; use secure primes in real use
        # generate p,q primes (simple method)
        def gen_prime(nbits):
            while True:
                r = random.getrandbits(nbits) | 1
                if pow(2, r-1, r) == 1 and r % 2 == 1:
                    # VERY weak primality check; ok for demo only
                    return r
        # For demo use small primes (unsafe)
        p = 2**((bits//2)-1) + 1
        q = 2**((bits//2)-1) + 3
        # fallback to small primes to avoid slow primality in demo
        # compute n
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
        # rerandomize: multiply by r^n
        n = pk[0]
        n2 = n*n
        r = random.randrange(1, n)
        return (c * pow(r, n, n2)) % n2

# ------------------------------
# Utility: hash-to-group element
# ------------------------------
def H_to_group(u, p, g, seed=b''):
    # Hash bytes u (or string) to integer and convert to group element as g^{hash}
    if isinstance(u, str): u = u.encode()
    h = hashlib.sha256(seed + u).digest()
    h_int = int.from_bytes(h, 'big')
    # Map to exponent in [1, p-1)
    exp = (h_int % (p-1)) or 1
    return pow(g, exp, p)  # element in multiplicative group mod p

# ------------------------------
# Toy multiplicative group params
# ------------------------------
# small demo primes (unsafe). For real: use safe 2048-bit prime or ECC.
P = 0xFFFFFFFBFFFFFFFDCFFFFFFFFFFFFFFF  # **placeholder small-ish prime** (demo; not secure)
# We'll instead pick a random safe-ish prime via simple method for demo:
def find_demo_prime(bits=64):
    # find a random prime-ish number (NOT cryptographically secure)
    while True:
        candidate = random.getrandbits(bits) | 1
        if candidate % 2 == 0: candidate += 1
        # tiny primality test
        if pow(2, candidate-1, candidate) == 1:
            return candidate

P = find_demo_prime(64)
# choose generator g
G = 2
# ------------------------------
# Protocol (local simulation of both parties)
# ------------------------------
def run_pisum_poc(P1_set, P2_pairs, seed=b'runseed'):
    # P1_set: list of identifiers (strings)
    # P2_pairs: list of (identifier, int_value)
    # Setup:
    k1 = random.randrange(2, P-1)
    k2 = random.randrange(2, P-1)
    # P2 (output party) generates HE keypair
    pk, sk = Paillier.keygen(bits=128)  # toy small keys
    # --- Round 1 (P1) ---
    Z1 = []
    for v in P1_set:
        h = H_to_group(v, P, G, seed)
        Z1.append(pow(h, k1, P))
    random.shuffle(Z1)
    # send Z1 -> P2
    # --- Round 2 (P2) ---
    # 1) for each received H(vi)^{k1}, compute ^k2 -> H(vi)^{k1k2}
    Z2 = [pow(elem, k2, P) for elem in Z1]
    random.shuffle(Z2)  # P2 sends shuffled Z back to P1
    # 3) for each (wj, tj) compute H(wj)^{k2} and AEnc(tj)
    W_list = []
    for (w, t) in P2_pairs:
        hw_k2 = pow(H_to_group(w, P, G, seed), k2, P)
        ct = Paillier.enc(pk, t)
        W_list.append( (hw_k2, ct) )
    random.shuffle(W_list)
    # send (W_list) -> P1 along with Z2
    # --- Round 3 (P1) ---
    # For each (H(wj)^{k2}, ct), compute ^k1 -> H(wj)^{k1k2}
    W_transformed = [ (pow(hw, k1, P), ct) for (hw, ct) in W_list ]
    # compute intersection: indices where transformed H(wj) in Z2
    set_Z2 = set(Z2)
    intersect_ciphertexts = [ct for (h, ct) in W_transformed if h in set_Z2]
    C = len(intersect_ciphertexts)
    # homomorphically sum ciphertexts
    if len(intersect_ciphertexts) == 0:
        sum_ct = Paillier.enc(pk, 0)
    else:
        sum_ct = intersect_ciphertexts[0]
        for ct in intersect_ciphertexts[1:]:
            sum_ct = Paillier.add(pk, sum_ct, ct)
    # rerandomize and send to P2
    sum_ct_refreshed = Paillier.refresh(pk, sum_ct)
    # --- Output (P2) ---
    S = Paillier.dec(sk, pk, sum_ct_refreshed)
    return {'intersection_size': C, 'intersection_sum': S}

# ------------------------------
# Demo run
# ------------------------------
if __name__ == "__main__":
    # example data
    P1 = ["alice@example.com", "bob@example.com", "carol@example.com", "dan@example.com"]
    P2 = [("eve@example.com", 10), ("bob@example.com", 25), ("carol@example.com", 5)]
    out = run_pisum_poc(P1, P2, seed=b'myseed123')
    print("intersection size:", out['intersection_size'])
    print("intersection sum:", out['intersection_sum'])  # expect 25 + 5 = 30
