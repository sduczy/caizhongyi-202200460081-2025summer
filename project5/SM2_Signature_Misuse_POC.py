#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SM2签名算法误用验证POC
演示常见的SM2签名算法误用场景及其攻击方法
"""

from gmssl import sm2, func
import base64
import hashlib
import secrets
from typing import Tuple, Optional
import math

class SM2SignatureMisusePOC:
    """SM2签名算法误用验证类"""
    
    def __init__(self):
        self.private_key = None
        self.public_key = None
        self.sm2_crypt = None
        
    def generate_keypair(self) -> Tuple[str, str]:
        """生成SM2密钥对"""
        self.private_key = func.random_hex(64)
        sm2_tmp = sm2.CryptSM2(private_key=self.private_key, public_key='')
        self.public_key = sm2_tmp._kg(int(self.private_key, 16), sm2_tmp.ecc_table['g'])
        self.sm2_crypt = sm2.CryptSM2(public_key=self.public_key, private_key=self.private_key)
        return self.private_key, self.public_key
    
    def sign_with_fixed_k(self, data: bytes, k: str) -> str:
        """使用固定随机数k进行签名"""
        try:
            return self.sm2_crypt.sign(data, k)
        except:
            try:
                return self.sm2_crypt.sign(data, k, user_id='1234567812345678')
            except:
                return self.sm2_crypt.sign(data, k)
    
    def verify_signature(self, data: bytes, signature: str) -> bool:
        """验证签名"""
        try:
            # 预处理签名格式
            if ',' in signature:
                # 如果签名包含逗号，转换为标准格式
                r, s = signature.split(',')
                signature = r + s
            
            return self.sm2_crypt.verify(signature, data)
        except:
            try:
                # 预处理签名格式
                if ',' in signature:
                    r, s = signature.split(',')
                    signature = r + s
                
                return self.sm2_crypt.verify(signature, data, user_id='1234567812345678')
            except:
                # 预处理签名格式
                if ',' in signature:
                    r, s = signature.split(',')
                    signature = r + s
                
                return self.sm2_crypt.verify(signature, data)

class NonceReuseAttack:
    """随机数重用攻击"""
    
    @staticmethod
    def demonstrate_nonce_reuse():
        """演示随机数重用攻击"""
        print("=" * 60)
        print("1. 随机数重用攻击演示")
        print("=" * 60)
        
        # 生成密钥对
        poc = SM2SignatureMisusePOC()
        private_key, public_key = poc.generate_keypair()
        print(f"私钥: {private_key}")
        print(f"公钥: {public_key}")
        
        # 使用相同的随机数k签名两个不同的消息
        k = func.random_hex(64)
        print(f"固定随机数k: {k}")
        
        msg1 = "Hello, SM2!".encode('utf-8')
        msg2 = "Hello, World!".encode('utf-8')
        
        sig1 = poc.sign_with_fixed_k(msg1, k)
        sig2 = poc.sign_with_fixed_k(msg2, k)
        
        print(f"消息1: {msg1.decode('utf-8')}")
        print(f"签名1: {sig1}")
        print(f"消息2: {msg2.decode('utf-8')}")
        print(f"签名2: {sig2}")
        
        # 验证签名
        print(f"签名1验证: {poc.verify_signature(msg1, sig1)}")
        print(f"签名2验证: {poc.verify_signature(msg2, sig2)}")
        
        # 从两个签名中恢复私钥
        recovered_private_key = NonceReuseAttack.recover_private_key_from_nonce_reuse(
            msg1, msg2, sig1, sig2, k
        )
        
        if recovered_private_key:
            print(f"恢复的私钥: {recovered_private_key}")
            print(f"私钥匹配: {recovered_private_key == private_key}")
        else:
            print("私钥恢复失败")
        
        return poc, k
    
    @staticmethod
    def recover_private_key_from_nonce_reuse(msg1: bytes, msg2: bytes, 
                                           sig1: str, sig2: str, k: str) -> Optional[str]:
        """从随机数重用中恢复私钥"""
        try:
            # 解析签名 (r, s) - 检查签名格式
            if ',' in sig1 and ',' in sig2:
                r1, s1 = sig1.split(',')
                r2, s2 = sig2.split(',')
            else:
                # 如果没有逗号，假设签名是128字符的十六进制字符串，前64位是r，后64位是s
                r1, s1 = sig1[:64], sig1[64:]
                r2, s2 = sig2[:64], sig2[64:]
            
            r1, s1 = int(r1, 16), int(s1, 16)
            r2, s2 = int(r2, 16), int(s2, 16)
            
            # 计算消息哈希
            h1 = int(hashlib.sha256(msg1).hexdigest(), 16)
            h2 = int(hashlib.sha256(msg2).hexdigest(), 16)
            
            # 使用SM2椭圆曲线参数
            n = 0xFFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFF7203DF6B21C6052B53BBF40939D54123
            
            # 计算私钥: d = (s1 * h2 - s2 * h1) / (r * (s1 - s2)) mod n
            k_int = int(k, 16)
            numerator = (s1 * h2 - s2 * h1) % n
            denominator = (r1 * (s1 - s2)) % n
            
            # 计算模逆
            def mod_inverse(a, m):
                def extended_gcd(a, b):
                    if a == 0:
                        return b, 0, 1
                    gcd, x1, y1 = extended_gcd(b % a, a)
                    x = y1 - (b // a) * x1
                    y = x1
                    return gcd, x, y
                
                gcd, x, _ = extended_gcd(a, m)
                if gcd != 1:
                    raise ValueError("模逆不存在")
                return x % m
            
            try:
                inv_denominator = mod_inverse(denominator, n)
                private_key = (numerator * inv_denominator) % n
                return hex(private_key)[2:].zfill(64)
            except ValueError:
                return None
                
        except Exception as e:
            print(f"私钥恢复过程中出错: {e}")
            return None

class PredictableNonceAttack:
    """可预测随机数攻击"""
    
    @staticmethod
    def demonstrate_predictable_nonce():
        """演示可预测随机数攻击"""
        print("\n" + "=" * 60)
        print("2. 可预测随机数攻击演示")
        print("=" * 60)
        
        poc = SM2SignatureMisusePOC()
        private_key, public_key = poc.generate_keypair()
        print(f"私钥: {private_key}")
        
        # 使用可预测的随机数（基于时间戳或消息内容）
        msg = "Predictable nonce attack".encode('utf-8')
        
        # 基于消息内容生成"可预测"的随机数
        predictable_k = hashlib.sha256(msg + b"predictable").hexdigest()
        print(f"可预测随机数k: {predictable_k}")
        
        signature = poc.sign_with_fixed_k(msg, predictable_k)
        print(f"消息: {msg.decode('utf-8')}")
        print(f"签名: {signature}")
        
        # 验证签名
        is_valid = poc.verify_signature(msg, signature)
        print(f"签名验证: {is_valid}")
        
        # 尝试恢复私钥
        recovered_key = PredictableNonceAttack.recover_private_key_from_predictable_nonce(
            msg, signature, predictable_k
        )
        
        if recovered_key:
            print(f"恢复的私钥: {recovered_key}")
            print(f"私钥匹配: {recovered_key == private_key}")
        else:
            print("私钥恢复失败")
        
        return poc, predictable_k
    
    @staticmethod
    def recover_private_key_from_predictable_nonce(msg: bytes, signature: str, k: str) -> Optional[str]:
        """从可预测随机数中恢复私钥"""
        try:
            # 解析签名 - 检查签名格式
            if ',' in signature:
                r, s = signature.split(',')
            else:
                # 如果没有逗号，假设签名是128字符的十六进制字符串，前64位是r，后64位是s
                r, s = signature[:64], signature[64:]
            
            r, s = int(r, 16), int(s, 16)
            
            # 计算消息哈希
            h = int(hashlib.sha256(msg).hexdigest(), 16)
            
            # SM2椭圆曲线参数
            n = 0xFFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFF7203DF6B21C6052B53BBF40939D54123
            
            k_int = int(k, 16)
            
            # 计算私钥: d = (s * k - h) / r mod n
            numerator = (s * k_int - h) % n
            denominator = r
            
            # 计算模逆
            def mod_inverse(a, m):
                def extended_gcd(a, b):
                    if a == 0:
                        return b, 0, 1
                    gcd, x1, y1 = extended_gcd(b % a, a)
                    x = y1 - (b // a) * x1
                    y = x1
                    return gcd, x, y
                
                gcd, x, _ = extended_gcd(a, m)
                if gcd != 1:
                    raise ValueError("模逆不存在")
                return x % m
            
            try:
                inv_denominator = mod_inverse(denominator, n)
                private_key = (numerator * inv_denominator) % n
                return hex(private_key)[2:].zfill(64)
            except ValueError:
                return None
                
        except Exception as e:
            print(f"私钥恢复过程中出错: {e}")
            return None

class SignatureVerificationBypass:
    """签名验证绕过攻击"""
    
    @staticmethod
    def demonstrate_verification_bypass():
        """演示签名验证绕过攻击"""
        print("\n" + "=" * 60)
        print("3. 签名验证绕过攻击演示")
        print("=" * 60)
        
        poc = SM2SignatureMisusePOC()
        private_key, public_key = poc.generate_keypair()
        print(f"私钥: {private_key}")
        print(f"公钥: {public_key}")
        
        # 构造伪造的签名
        msg = "Forged signature".encode('utf-8')
        
        # 方法1: 构造无效的r值
        fake_r = "0" * 64  # 全零的r值
        fake_s = func.random_hex(64)
        fake_signature1 = fake_r + fake_s  # 移除逗号，直接拼接
        
        print(f"伪造签名1 (r=0): {fake_signature1}")
        print(f"验证结果: {poc.verify_signature(msg, fake_signature1)}")
        
        # 方法2: 构造无效的s值
        real_r = func.random_hex(64)
        fake_s2 = "0" * 64  # 全零的s值
        fake_signature2 = real_r + fake_s2  # 移除逗号，直接拼接
        
        print(f"伪造签名2 (s=0): {fake_signature2}")
        print(f"验证结果: {poc.verify_signature(msg, fake_signature2)}")
        
        # 方法3: 构造格式错误的签名
        malformed_signature = "invalid_signature_format"
        print(f"格式错误签名: {malformed_signature}")
        try:
            result = poc.verify_signature(msg, malformed_signature)
            print(f"验证结果: {result}")
        except Exception as e:
            print(f"验证异常: {e}")

class PrivateKeyRecoveryAttack:
    """私钥恢复攻击"""
    
    @staticmethod
    def demonstrate_private_key_recovery():
        """演示私钥恢复攻击"""
        print("\n" + "=" * 60)
        print("4. 私钥恢复攻击演示")
        print("=" * 60)
        
        poc = SM2SignatureMisusePOC()
        private_key, public_key = poc.generate_keypair()
        print(f"真实私钥: {private_key}")
        
        # 收集多个签名样本
        msg1 = "Sample message 1".encode('utf-8')
        msg2 = "Sample message 2".encode('utf-8')
        msg3 = "Sample message 3".encode('utf-8')
        
        # 使用不同的随机数进行签名
        k1 = func.random_hex(64)
        k2 = func.random_hex(64)
        k3 = func.random_hex(64)
        
        sig1 = poc.sign_with_fixed_k(msg1, k1)
        sig2 = poc.sign_with_fixed_k(msg2, k2)
        sig3 = poc.sign_with_fixed_k(msg3, k3)
        
        print(f"消息1签名: {sig1}")
        print(f"消息2签名: {sig2}")
        print(f"消息3签名: {sig3}")
        
        # 尝试通过多个签名恢复私钥
        recovered_key = PrivateKeyRecoveryAttack.recover_private_key_from_multiple_signatures(
            [(msg1, sig1, k1), (msg2, sig2, k2), (msg3, sig3, k3)]
        )
        
        if recovered_key:
            print(f"恢复的私钥: {recovered_key}")
            print(f"私钥匹配: {recovered_key == private_key}")
        else:
            print("私钥恢复失败")
    
    @staticmethod
    def recover_private_key_from_multiple_signatures(signatures_data):
        """从多个签名中恢复私钥"""
        try:
            # 这里实现更复杂的私钥恢复算法
            # 基于多个签名的统计分析
            print("尝试通过多个签名恢复私钥...")
            
            # 简化实现：检查是否有重复的随机数
            k_values = [data[2] for data in signatures_data]
            if len(set(k_values)) != len(k_values):
                print("检测到重复的随机数！")
                # 可以进一步分析重复的随机数
                return None
            
            print("未检测到重复随机数，需要更复杂的分析...")
            return None
            
        except Exception as e:
            print(f"私钥恢复过程中出错: {e}")
            return None

def main():
    """主函数：运行所有攻击演示"""
    print("SM2签名算法误用验证POC")
    print("=" * 60)
    
    try:
        # 1. 随机数重用攻击
        poc1, k1 = NonceReuseAttack.demonstrate_nonce_reuse()
        
        # 2. 可预测随机数攻击
        poc2, k2 = PredictableNonceAttack.demonstrate_predictable_nonce()
        
        # 3. 签名验证绕过攻击
        SignatureVerificationBypass.demonstrate_verification_bypass()
        
        # 4. 私钥恢复攻击
        PrivateKeyRecoveryAttack.demonstrate_private_key_recovery()
        
        print("\n" + "=" * 60)
        print("所有攻击演示完成")
        print("=" * 60)
        
        # 总结和建议
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