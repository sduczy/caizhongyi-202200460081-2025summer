import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
 
class DCT_Embed(object):
    def __init__(self, background, watermark, block_size=8, alpha=30):
        b_h, b_w = background.shape[:2]
        w_h, w_w = watermark.shape[:2]  # Adjust to handle 2D watermark
        assert w_h <= b_h / block_size and w_w <= b_w / block_size, \
            "\r\n请确保您的的水印图像尺寸 不大于 背景图像尺寸的1/{:}\r\nbackground尺寸{:}\r\nwatermark尺寸{:}".format(
                block_size, background.shape, watermark.shape
            )
 
        self.block_size = block_size
        self.alpha = alpha
        self.k1 = np.random.randn(block_size)
        self.k2 = np.random.randn(block_size)
 
    def dct_blkproc(self, background):
        background_dct_blocks_h = background.shape[0] // self.block_size
        background_dct_blocks_w = background.shape[1] // self.block_size
        background_dct_blocks = np.zeros(shape=(
            (background_dct_blocks_h, background_dct_blocks_w, self.block_size, self.block_size)
        ))
 
        h_data = np.vsplit(background, background_dct_blocks_h)
        for h in range(background_dct_blocks_h):
            block_data = np.hsplit(h_data[h], background_dct_blocks_w)
            for w in range(background_dct_blocks_w):
                a_block = block_data[w]
                background_dct_blocks[h, w, ...] = cv2.dct(a_block.astype(np.float64))
        return background_dct_blocks
 
    def dct_embed(self, dct_data, watermark):
        temp = watermark.flatten()
        assert temp.max() == 1 and temp.min() == 0, "为方便处理，请保证输入的watermark是被二值归一化的"
 
        result = dct_data.copy()
        for h in range(watermark.shape[0]):
            for w in range(watermark.shape[1]):
                k = self.k1 if watermark[h, w] == 1 else self.k2
                for i in range(self.block_size):
                    result[h, w, i, self.block_size - 1] = dct_data[h, w, i, self.block_size - 1] + self.alpha * k[i]
        return result
 
    def idct_embed(self, dct_data):
        row = None
        result = None
        h, w = dct_data.shape[0], dct_data.shape[1]
        for i in range(h):
            for j in range(w):
                block = cv2.idct(dct_data[i, j, ...])
                row = block if j == 0 else np.hstack((row, block))
            result = row if i == 0 else np.vstack((result, row))
        return result.astype(np.uint8)
 
    def dct_extract(self, synthesis, watermark_size):
        w_h, w_w = watermark_size
        recover_watermark = np.zeros(shape=watermark_size)
        synthesis_dct_blocks = self.dct_blkproc(background=synthesis)
        p = np.zeros(8)
        for h in range(w_h):
            for w in range(w_w):
                for k in range(self.block_size):
                    p[k] = synthesis_dct_blocks[h, w, k, self.block_size - 1]
                if corr2(p, self.k1) > corr2(p, self.k2):
                    recover_watermark[h, w] = 1
                else:
                    recover_watermark[h, w] = 0
        return recover_watermark
 
def mean2(x):
    y = np.sum(x) / np.size(x)
    return y
 
def corr2(a, b):
    a = a - mean2(a)
    b = b - mean2(b)
    r = (a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum())
    return r
 
if __name__ == '__main__':
 
    alpha = 10
    blocksize = 8

    # 设置图片路径
    watermark_path = r"watermark.png"
    host_image_path = r"host_image.png"

    # 读取水印图像
    if not os.path.exists(watermark_path):
        raise FileNotFoundError(f"水印图像不存在：{watermark_path}")

    watermark = cv2.imread(watermark_path)
    if watermark is None:
        raise ValueError(f"无法读取水印图像，请确认格式正确：{watermark_path}")
    watermark = cv2.cvtColor(watermark, cv2.COLOR_BGR2RGB)

    # 二值化水印（用于嵌入）
    watermark_bin = np.where(watermark < np.mean(watermark, axis=(0, 1)), 0, 1)

    # 读取宿主图像（host image）
    if not os.path.exists(host_image_path):
        raise FileNotFoundError(f"宿主图像不存在：{host_image_path}")

    background = cv2.imread(host_image_path)
    if background is None:
        raise ValueError(f"无法读取宿主图像，请确认格式正确：{host_image_path}")
    background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)

    # 备份原图像
    background_backup = background.copy()

    # 转换颜色空间为 YUV
    yuv_background = cv2.cvtColor(background, cv2.COLOR_RGB2YUV)
    Y, U, V = yuv_background[..., 0], yuv_background[..., 1], yuv_background[..., 2]
     
    channels = cv2.split(background)
    embed_synthesis = []
    extract_watermarks = []
    for i in range(3):
        dct_emb = DCT_Embed(background=channels[i], watermark=watermark_bin[..., i], block_size=blocksize, alpha=alpha)
        background_dct_blocks = dct_emb.dct_blkproc(background=channels[i])
        embed_watermark_blocks = dct_emb.dct_embed(dct_data=background_dct_blocks, watermark=watermark_bin[..., i])
        synthesis = dct_emb.idct_embed(dct_data=embed_watermark_blocks)
        embed_synthesis.append(synthesis)
        extract_watermarks.append(dct_emb.dct_extract(synthesis=synthesis, watermark_size=watermark_bin[..., i].shape) * 255)
 
    rbg_synthesis = cv2.merge(embed_synthesis)
    extract_watermark = cv2.merge([ew.astype(np.uint8) for ew in extract_watermarks])
 
    images = [background_backup, watermark, rbg_synthesis, extract_watermark]
    titles = ["image", "watermark", "systhesis", "extract"]
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        if i == 1 or i == 3:
            plt.imshow(images[i])
        else:
            plt.imshow(images[i])
        plt.title(titles[i])
        plt.axis("off")
    plt.show()
