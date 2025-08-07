# 实现 Google Password Checkup 协议的简化模拟版本（仅用于学习用途）
# ⚠️ 实际商用应使用更强密码哈希、盲化安全曲线映射等技术

import hashlib
import secrets
from tinyec import registry
from tinyec.ec import Point

# 选择椭圆曲线（secp256r1）
curve = registry.get_curve('secp256r1')

# 模拟密码哈希到椭圆曲线点（使用简化方式）
def hash_to_point(pwd: str) -> Point:
    pwd_hash = hashlib.sha256(pwd.encode()).digest()
    int_val = int.from_bytes(pwd_hash, 'big') % curve.field.n
    return int_val * curve.g  # scalar * generator = Point

# 盲化一个点
def blind_point(P: Point, r: int) -> Point:
    return r * P

# 模拟客户端
class Client:
    def __init__(self, password):
        self.password = password
        self.r = secrets.randbelow(curve.field.n)

    def generate_blinded_pwd(self):
        self.H_pwd = hash_to_point(self.password)
        self.blinded = blind_point(self.H_pwd, self.r)
        return self.blinded

    def deblind_and_check(self, blinded_db):
        # 再次盲化数据库集合的每个点: r * server_point
        reblinded_set = [blind_point(P, self.r) for P in blinded_db]
        return self.blinded in reblinded_set

# 模拟服务器端
class Server:
    def __init__(self, leaked_passwords):
        self.leaked_passwords = leaked_passwords
        self.db = [hash_to_point(pwd) for pwd in leaked_passwords]

    def send_blinded_db(self):
        # 可分片或全部发送，这里全发
        return self.db

# 示例用法
if __name__ == "__main__":
    leaked_passwords = ["123456", "password", "letmein", "abc123"]
    server = Server(leaked_passwords)

    # 模拟用户
    user_pwd = "letmein"
    client = Client(user_pwd)

    # 1. 客户端生成盲化密码并发送
    B = client.generate_blinded_pwd()

    # 2. 服务器返回盲化数据库（未加盲）
    db = server.send_blinded_db()

    # 3. 客户端使用同样因子对服务器数据库再盲化进行匹配
    compromised = client.deblind_and_check(db)

    print("⚠️ 密码已泄露" if compromised else "✅ 密码未泄露")
