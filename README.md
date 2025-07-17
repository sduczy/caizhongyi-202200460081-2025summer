# 蔡中仪-202200460081-2025网安创新创业实践
（ps：具体commit看master分支）
## project1:
### 题目：
做SM4的软件实现和优化
### 我的优化：

1、减少函数嵌套 

2、减少内存使用

3、函数手动展开，提高速度

4、使用局部变量代替列表，减少内存复制

5、避免数组

6、使用bytearray避免字符串拼接开销

## project2: 
### 题目：
编程实现图片水印嵌入和提取（可依托开源项目二次开发），并进行鲁棒性测试，包括不限于翻转、平移、截取、调对比度等
### 我的优化：
1、
2、
3、
4、
5、
6、

## project3: 
### 题目：
 用circom实现poseidon2哈希算法的电路
 
要求： 

 （1）poseidon2哈希算法参数参考参考文档1的Table1，用(n,t,d)=(256,3,5)或(256,2,5)
 
 （2）电路的公开输入用poseidon2哈希值，隐私输入为哈希原象，哈希算法的输入只考虑一个block即可。
 
 （3）用Groth16算法生成证明

参考文档：

1. poseidon2哈希算法https://eprint.iacr.org/2023/323.pdf
   
2. circom说明文档https://docs.circom.io/

3. circom电路样例 https://github.com/iden3/circomlib

### 我的优化：
1、
2、
3、
4、
5、
6、

## project4: 
### 题目：
sm3 的软件实现与优化 跟 SM4一样 用 C 语言来做 然后不断改进效率
### 我的优化：
1、
2、
3、
4、
5、
6、

## project5: 
### 题目：
sm2 的软件实现优化 

a). 考虑到SM2用C 语言来做比较复杂，大家看可以考虑用python来做 sm2的 基础实现以及各种算法的改进尝试

b). 20250713-wen-sm2-public.pdf 中提到的关于签名算法的误用 分别基于做poc验证，给出推导文档以及验证代码

c). 伪造中本聪的数字签名

### 我的优化：
1、
2、
3、
4、
5、
6、

## project6: 
### 题目：

Google Password Checkup验证

来自刘巍然老师的报告  google password checkup，参考论文 https://eprint.iacr.org/2019/723.pdf 的 section 3.1 ，也即 Figure 2 中展示的协议，尝试实现该协议，（编程语言不限）。

### 我的优化：
1、
2、
3、
4、
5、
6、


