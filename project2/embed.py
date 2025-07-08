import cv2
import numpy as np
from scipy.fftpack import dct, idct


#水印嵌入函数
def embed_watermark(host_path, watermark_path, output_path, alpha=10):
    host = cv2.imread(host_path, cv2.IMREAD_GRAYSCALE)
    watermark = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)
    watermark = cv2.resize(watermark, (64, 64))
    wm_bin = (watermark > 128).astype(np.uint8)

    h_dct = dct(dct(host.astype(np.float32), axis=0), axis=1)

    for i in range(64):
        for j in range(64):
            h_dct[i + 1, j + 1] += alpha * wm_bin[i, j]

    watermarked = idct(idct(h_dct, axis=1), axis=0)
    watermarked = np.clip(watermarked, 0, 255).astype(np.uint8)
    cv2.imwrite(output_path, watermarked)
    print(f"Watermark embedded and saved to {output_path}")

# 原始图像（灰度渐变）
host = np.tile(np.arange(256, dtype=np.uint8), (256, 1))
cv2.imwrite("host_image.jpg", host)

# 水印图像（64x64的“W”字）
watermark = np.zeros((64, 64), dtype=np.uint8)
cv2.putText(watermark, 'W', (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 3)
cv2.imwrite("watermark.png", watermark)

    
# 设置路径
host_img_path = "host_image.jpg"         # 原始图像路径
watermark_img_path = "watermark.png"     # 水印图像路径
output_img_path = "watermarked_image.jpg" # 嵌入水印后的输出路径

# 调用嵌入函数
embed_watermark(host_img_path, watermark_img_path, output_img_path)
