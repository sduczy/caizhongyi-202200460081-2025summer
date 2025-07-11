import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

class WatermarkProcessor:
    def __init__(self, host_channel, wm_channel, blk_size=8, strength=30):
        h_h, h_w = host_channel.shape
        wm_h, wm_w = wm_channel.shape
        assert wm_h <= h_h // blk_size and wm_w <= h_w // blk_size, \
            f"水印尺寸需≤宿主的1/{blk_size}, 宿主:{host_channel.shape}, 水印:{wm_channel.shape}"
        self.blk_size = blk_size
        self.strength = strength
        self.key1 = np.random.randn(blk_size)
        self.key2 = np.random.randn(blk_size)

    def _block_dct(self, img):
        h_blocks = img.shape[0] // self.blk_size
        w_blocks = img.shape[1] // self.blk_size
        dct_blocks = np.zeros((h_blocks, w_blocks, self.blk_size, self.blk_size))
        for i in range(h_blocks):
            for j in range(w_blocks):
                block = img[i*self.blk_size:(i+1)*self.blk_size,
                            j*self.blk_size:(j+1)*self.blk_size]
                dct_blocks[i, j] = cv2.dct(block.astype(np.float64))
        return dct_blocks

    def _inverse_dct(self, dct_blocks):
        h, w = dct_blocks.shape[:2]
        recon_img = []
        for i in range(h):
            row = []
            for j in range(w):
                row.append(cv2.idct(dct_blocks[i, j]))
            recon_img.append(np.hstack(row))
        return np.vstack(recon_img).astype(np.uint8)

    def embed(self, dct_blocks, wm_bin):
        assert set(np.unique(wm_bin)) <= {0, 1}, "水印必须是二值图"
        out = dct_blocks.copy()
        for i in range(wm_bin.shape[0]):
            for j in range(wm_bin.shape[1]):
                key = self.key1 if wm_bin[i, j] == 1 else self.key2
                out[i, j, :, -1] += self.strength * key
        return out

    def extract(self, mixed_img, wm_shape):
        dct_blocks = self._block_dct(mixed_img)
        wm_extracted = np.zeros(wm_shape)
        for i in range(wm_shape[0]):
            for j in range(wm_shape[1]):
                coeffs = dct_blocks[i, j, :, -1]
                c1 = self._correlate(coeffs, self.key1)
                c2 = self._correlate(coeffs, self.key2)
                wm_extracted[i, j] = 1 if c1 > c2 else 0
        return wm_extracted

    def _correlate(self, vec1, vec2):
        v1 = vec1 - np.mean(vec1)
        v2 = vec2 - np.mean(vec2)
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# 工具函数
def load_image(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"图像未找到: {path}")
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"图像加载失败: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def apply_attack(img, method):
    if method == "gaussian_noise":
        noise = np.random.normal(0, 10, img.shape).astype(np.int16)
        return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    elif method == "rotate":
        M = cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), 10, 1)
        return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    elif method == "crop":
        h, w = img.shape[:2]
        return cv2.resize(img[20:h-20, 20:w-20], (w, h))
    elif method == "contrast":
        return cv2.convertScaleAbs(img, alpha=1.5, beta=0)
    elif method == "jpeg":
        _, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        return cv2.imdecode(buf, 1)
    elif method == "blur":
        return cv2.GaussianBlur(img, (5, 5), 1)
    return img

def test_robustness(attacked_img, processors, wm_shapes):
    yuv = cv2.cvtColor(attacked_img, cv2.COLOR_RGB2YUV)
    recovered = []
    for ch in range(3):
        wm = processors[ch].extract(yuv[..., ch], wm_shapes[ch]) * 255
        recovered.append(wm.astype(np.uint8))
    return cv2.merge(recovered)

# 主流程
if __name__ == "__main__":
    strength = 10
    blk_size = 8
    host_path = "host_image.png"
    wm_path = "watermark.png"

    host = load_image(host_path)
    wm = load_image(wm_path)
    wm_bin = np.where(wm < np.mean(wm, axis=(0,1)), 0, 1)

    yuv = cv2.cvtColor(host, cv2.COLOR_RGB2YUV)
    watermarked_channels = []
    recovered_channels = []
    processors = []

    for ch in range(3):
        proc = WatermarkProcessor(yuv[..., ch], wm_bin[..., ch], blk_size, strength)
        dct = proc._block_dct(yuv[..., ch])
        encoded = proc.embed(dct, wm_bin[..., ch])
        img = proc._inverse_dct(encoded)
        rec = proc.extract(img, wm_bin[..., ch].shape) * 255

        processors.append(proc)
        watermarked_channels.append(img)
        recovered_channels.append(rec.astype(np.uint8))

    final_img = cv2.merge(watermarked_channels)
    final_wm = cv2.merge(recovered_channels)

    # 展示原始水印与水印图像
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    titles = ["原图", "水印", "嵌入图", "提取水印"]
    images = [host, wm, final_img, final_wm]
    for ax, img, title in zip(axs.flat, images, titles):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.show()

    # 鲁棒性测试
    attacks = ["gaussian_noise", "rotate", "crop", "contrast", "jpeg", "blur"]
    wm_shapes = [wm_bin[..., i].shape for i in range(3)]

    for attack in attacks:
        attacked = apply_attack(final_img, attack)
        recovered = test_robustness(attacked, processors, wm_shapes)

        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        axs[0].imshow(attacked)
        axs[0].set_title(f"攻击后图像 ({attack})")
        axs[0].axis("off")

        axs[1].imshow(recovered)
        axs[1].set_title("提取水印")
        axs[1].axis("off")

        plt.tight_layout()
        plt.show()
