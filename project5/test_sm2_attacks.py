#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SM2签名算法误用测试脚本
简化版本，用于快速验证各种攻击方法
"""

from gmssl import sm2, func
import hashlib
from sm2_attack_utils import (
    recover_private_key_from_nonce_reuse as util_recover_from_reuse,
    recover_private_key_from_predictable_nonce as util_recover_from_pred,
)

def test_nonce_reuse_attack():
    """测试随机数重用攻击"""
    print("=" * 50)
    print("测试1: 随机数重用攻击")
    print("=" * 50)
    
    # 生成密钥对
    private_key = func.random_hex(64)
    sm2_tmp = sm2.CryptSM2(private_key=private_key, public_key='')
    public_key = sm2_tmp._kg(int(private_key, 16), sm2_tmp.ecc_table['g'])
    
    print(f"真实私钥: {private_key}")
    print(f"公钥: {public_key}")
    
    # 创建SM2实例
    sm2_crypt = sm2.CryptSM2(public_key=public_key, private_key=private_key)
    
    # 使用相同的随机数k签名两个不同消息
    k = func.random_hex(64)
    print(f"\n使用固定随机数k: {k}")
    
    msg1 = "Hello, SM2!".encode('utf-8')
    msg2 = "Hello, World!".encode('utf-8')
    
    try:
        sig1 = sm2_crypt.sign(msg1, k)
        sig2 = sm2_crypt.sign(msg2, k)
        
        print(f"消息1: {msg1.decode('utf-8')}")
        print(f"签名1: {sig1}")
        print(f"消息2: {msg2.decode('utf-8')}")
        print(f"签名2: {sig2}")
        
        # 验证签名
        print(f"\n签名1验证: {sm2_crypt.verify(sig1, msg1)}")
        print(f"签名2验证: {sm2_crypt.verify(sig2, msg2)}")
        
        # 尝试恢复私钥
        print("\n尝试恢复私钥...")
        recovered_key = recover_private_key_from_nonce_reuse(msg1, msg2, sig1, sig2, k)
        
        if recovered_key:
            print(f"恢复的私钥: {recovered_key}")
            print(f"私钥匹配: {recovered_key == private_key}")
            if recovered_key == private_key:
                print("✅ 攻击成功！私钥被完全恢复")
            else:
                print("❌ 攻击失败，私钥不匹配")
        else:
            print("❌ 攻击失败，无法恢复私钥")
            
    except Exception as e:
        print(f"签名过程中出错: {e}")

def recover_private_key_from_nonce_reuse(msg1, msg2, sig1, sig2, k):
    try:
        return util_recover_from_reuse(msg1, msg2, sig1, sig2)
    except Exception as e:
        print(f"私钥恢复过程中出错: {e}")
        return None

def test_predictable_nonce_attack():
    """测试可预测随机数攻击"""
    print("\n" + "=" * 50)
    print("测试2: 可预测随机数攻击")
    print("=" * 50)
    
    # 生成密钥对
    private_key = func.random_hex(64)
    sm2_tmp = sm2.CryptSM2(private_key=private_key, public_key='')
    public_key = sm2_tmp._kg(int(private_key, 16), sm2_tmp.ecc_table['g'])
    
    print(f"真实私钥: {private_key}")
    
    # 创建SM2实例
    sm2_crypt = sm2.CryptSM2(public_key=public_key, private_key=private_key)
    
    # 使用可预测的随机数
    msg = "Predictable nonce attack".encode('utf-8')
    predictable_k = hashlib.sha256(msg + b"predictable").hexdigest()
    
    print(f"\n使用可预测随机数k: {predictable_k}")
    print(f"消息: {msg.decode('utf-8')}")
    
    try:
        signature = sm2_crypt.sign(msg, predictable_k)
        print(f"签名: {signature}")
        
        # 验证签名
        is_valid = sm2_crypt.verify(signature, msg)
        print(f"签名验证: {is_valid}")
        
        # 尝试恢复私钥
        print("\n尝试恢复私钥...")
        recovered_key = recover_private_key_from_predictable_nonce(msg, signature, predictable_k)
        
        if recovered_key:
            print(f"恢复的私钥: {recovered_key}")
            print(f"私钥匹配: {recovered_key == private_key}")
            if recovered_key == private_key:
                print("✅ 攻击成功！私钥被完全恢复")
            else:
                print("❌ 攻击失败，私钥不匹配")
        else:
            print("❌ 攻击失败，无法恢复私钥")
            
    except Exception as e:
        print(f"签名过程中出错: {e}")

def recover_private_key_from_predictable_nonce(msg, signature, k):
    try:
        return util_recover_from_pred(msg, signature, k)
    except Exception as e:
        print(f"私钥恢复过程中出错: {e}")
        return None

def test_signature_verification_bypass():
    """测试签名验证绕过攻击"""
    print("\n" + "=" * 50)
    print("测试3: 签名验证绕过攻击")
    print("=" * 50)
    
    # 生成密钥对
    private_key = func.random_hex(64)
    sm2_tmp = sm2.CryptSM2(private_key=private_key, public_key='')
    public_key = sm2_tmp._kg(int(private_key, 16), sm2_tmp.ecc_table['g'])
    
    print(f"私钥: {private_key}")
    print(f"公钥: {public_key}")
    
    # 创建SM2实例
    sm2_crypt = sm2.CryptSM2(public_key=public_key, private_key=private_key)
    
    # 构造伪造的签名
    msg = "Forged signature".encode('utf-8')
    print(f"目标消息: {msg.decode('utf-8')}")
    
    # 方法1: 构造无效的r值
    fake_r = "0" * 64
    fake_s = func.random_hex(64)
    fake_signature1 = f"{fake_r},{fake_s}"
    
    print(f"\n伪造签名1 (r=0): {fake_signature1}")
    try:
        result1 = sm2_crypt.verify(fake_signature1, msg)
        print(f"验证结果: {result1}")
        if result1:
            print("⚠️  警告：无效签名被接受！")
        else:
            print("✅ 正确：无效签名被拒绝")
    except Exception as e:
        print(f"验证异常: {e}")
    
    # 方法2: 构造无效的s值
    real_r = func.random_hex(64)
    fake_s2 = "0" * 64
    fake_signature2 = f"{real_r},{fake_s2}"
    
    print(f"\n伪造签名2 (s=0): {fake_signature2}")
    try:
        result2 = sm2_crypt.verify(fake_signature2, msg)
        print(f"验证结果: {result2}")
        if result2:
            print("⚠️  警告：无效签名被接受！")
        else:
            print("✅ 正确：无效签名被拒绝")
    except Exception as e:
        print(f"验证异常: {e}")
    
    # 方法3: 构造格式错误的签名
    malformed_signature = "invalid_signature_format"
    print(f"\n格式错误签名: {malformed_signature}")
    try:
        result3 = sm2_crypt.verify(malformed_signature, msg)
        print(f"验证结果: {result3}")
        if result3:
            print("⚠️  警告：格式错误签名被接受！")
        else:
            print("✅ 正确：格式错误签名被拒绝")
    except Exception as e:
        print(f"验证异常: {e}")

def main():
    """主函数"""
    print("SM2签名算法误用攻击测试")
    print("=" * 50)
    
    try:
        # 测试1: 随机数重用攻击
        test_nonce_reuse_attack()
        
        # 测试2: 可预测随机数攻击
        test_predictable_nonce_attack()
        
        # 测试3: 签名验证绕过攻击
        test_signature_verification_bypass()
        
        print("\n" + "=" * 50)
        print("所有测试完成")
        print("=" * 50)
        
        print("\n安全建议:")
        print("1. 每次签名必须使用唯一的随机数k")
        print("2. 随机数k必须具有足够的熵值，不可预测")
        print("3. 实现严格的签名验证，拒绝无效签名")
        print("4. 定期更换密钥对")
        print("5. 使用安全的随机数生成器")
        
    except Exception as e:
        print(f"程序执行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 