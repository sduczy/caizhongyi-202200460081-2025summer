from gmssl import sm2, func

# 生成随机私钥
private_key = func.random_hex(64)

# 根据私钥计算对应公钥
sm2_crypt = sm2.CryptSM2(public_key='', private_key=private_key)
public_key = sm2_crypt._kg(int(private_key, 16), sm2_crypt.ecc_table['g'])

# 创建新的 SM2 实例（含公钥）
sm2_crypt = sm2.CryptSM2(public_key=public_key, private_key=private_key)

# 要签名的消息
msg = b"Hello, SM2!"

# 签名
sign = sm2_crypt.sign(msg, func.random_hex(sm2_crypt.para_len))
print("签名：", sign)

# 验签
verify = sm2_crypt.verify(sign, msg)
print("验签结果：", verify)
