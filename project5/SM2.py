from gmssl import sm2, func
import base64

class SM2Utils:
    def __init__(self, private_key=None, public_key=None):
        self.private_key = private_key
        self.public_key = public_key
        self.sm2_crypt = sm2.CryptSM2(public_key=public_key or '', private_key=private_key or '')

    @staticmethod
    def generate_keypair():
        """生成 SM2 密钥对（私钥和公钥，十六进制字符串）"""
        private_key = func.random_hex(64)
        sm2_tmp = sm2.CryptSM2(private_key=private_key, public_key='')
        public_key = sm2_tmp._kg(int(private_key, 16), sm2_tmp.ecc_table['g'])
        return private_key, public_key

    def sign(self, data: bytes) -> str:
        """签名数据，返回签名（hex）"""
        rand_k = func.random_hex(self.sm2_crypt.para_len)
        return self.sm2_crypt.sign(data, rand_k)

    def verify(self, data: bytes, signature: str) -> bool:
        """验签"""
        return self.sm2_crypt.verify(signature, data)

    def encrypt(self, data: bytes) -> str:
        """SM2 加密，返回 base64 编码"""
        encrypted_bytes = self.sm2_crypt.encrypt(data)
        return base64.b64encode(encrypted_bytes).decode()

    def decrypt(self, cipher_base64: str) -> bytes:
        """SM2 解密，输入 base64 编码密文"""
        cipher_bytes = base64.b64decode(cipher_base64)
        return self.sm2_crypt.decrypt(cipher_bytes)



# === 第一步：生成密钥对 ===
private_key, public_key = SM2Utils.generate_keypair()
print("私钥:", private_key)
print("公钥:", public_key)

# === 第二步：初始化工具类 ===
sm2_tool = SM2Utils(private_key=private_key, public_key=public_key)

# === 第三步：签名 ===
msg = "你好，SM2".encode('utf-8') 
signature = sm2_tool.sign(msg)
print("签名:", signature)

# === 第四步：验签 ===
is_valid = sm2_tool.verify(msg, signature)
print("验签结果:", is_valid)

# === 第五步：加密解密 ===
encrypted = sm2_tool.encrypt(msg)
print("加密:", encrypted)

decrypted = sm2_tool.decrypt(encrypted)
print("解密:", decrypted)















