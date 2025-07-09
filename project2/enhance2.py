import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 重命名类名和函数名，调整参数顺序
class WM_Processor(object):
    def __init__(self, host_img, wm_img, blk_size=8, strength=30):
        h_h, h_w = host_img.shape[:2]
        wm_h, wm_w = wm_img.shape[:2]  
        assert wm_h <= h_h // blk_size and wm_w <= h_w // blk_size, \
            f"水印尺寸需≤宿主的1/{blk_size}, 宿主:{host_img.shape}, 水印:{wm_img.shape}"
        self.blk_size = blk_size
        self.strength = strength
        self.key1 = np.random.randn(blk_size)
        self.key2 = np.random.randn(blk_size)

    # 重命名函数，调整处理顺序
    def block_process(self, img_data):
        h_blocks = img_data.shape[0] // self.blk_size
        w_blocks = img_data.shape[1] // self.blk_size
        dct_blocks = np.zeros((h_blocks, w_blocks, self.blk_size, self.blk_size))
        
        split_h = np.vsplit(img_data, h_blocks)
        for i in range(h_blocks):
            split_w = np.hsplit(split_h[i], w_blocks)
            for j in range(w_blocks):
                block = split_w[j]
                dct_blocks[i, j] = cv2.dct(block.astype(np.float64))
        return dct_blocks

    # 调整参数顺序，简化循环
    def encode_wm(self, dct_blocks, wm_data):
        flat_wm = wm_data.flatten()
        assert flat_wm.max() == 1 and flat_wm.min() == 0, "水印需二值化"
        modified_blocks = dct_blocks.copy()
        
        for row in range(wm_data.shape[0]):
            for col in range(wm_data.shape[1]):
                curr_key = self.key1 if wm_data[row, col] == 1 else self.key2
                for k in range(self.blk_size):
                    modified_blocks[row, col, k, -1] += self.strength * curr_key[k]
        return modified_blocks

    # 合并冗余代码
    def inverse_dct(self, modified_blocks):
        reconstructed = []
        h, w = modified_blocks.shape[:2]
        
        for i in range(h):
            row_blocks = []
            for j in range(w):
                idct_block = cv2.idct(modified_blocks[i, j])
                row_blocks.append(idct_block)
            reconstructed.append(np.hstack(row_blocks))
        return np.vstack(reconstructed).astype(np.uint8)

    # 调整检测逻辑顺序
    def decode_wm(self, mixed_img, wm_shape):
        wm_h, wm_w = wm_shape
        extracted = np.zeros(wm_shape)
        dct_blocks = self.block_process(mixed_img)
        
        for r in range(wm_h):
            for c in range(wm_w):
                coeffs = [dct_blocks[r, c, k, -1] for k in range(self.blk_size)]
                corr1 = self._calc_corr(coeffs, self.key1)
                corr2 = self._calc_corr(coeffs, self.key2)
                extracted[r, c] = 1 if corr1 > corr2 else 0
        return extracted

    # 内部函数重命名
    def _calc_corr(self, vec1, vec2):
        vec1 = vec1 - np.mean(vec1)
        vec2 = vec2 - np.mean(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# 工具函数合并重命名
def prepare_image(img_path):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"文件未找到: {img_path}")
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def apply_attack(image, attack_type):
    if attack_type == "gaussian_noise":
        noise = np.random.normal(0, 10, image.shape).astype(np.int16)
        return np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    elif attack_type == "rotate":
        center = (image.shape[1]//2, image.shape[0]//2)
        M = cv2.getRotationMatrix2D(center, 10, 1.0)
        return cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    elif attack_type == "crop":
        h, w = image.shape[:2]
        margin = 20
        cropped = image[margin:h-margin, margin:w-margin]
        return cv2.resize(cropped, (w, h))
    elif attack_type == "contrast":
        return cv2.convertScaleAbs(image, alpha=1.5, beta=0)
    elif attack_type == "jpeg":
        _, enc_img = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        return cv2.imdecode(enc_img, 1)
    elif attack_type == "blur":
        return cv2.GaussianBlur(image, (5, 5), 1)
    else:
        return image

def test_robustness(attacked_img, processor_list, wm_shape):
    recovered_channels = []
    yuv_attacked = cv2.cvtColor(attacked_img, cv2.COLOR_RGB2YUV)
    for ch in range(3):
        rec_wm = processor_list[ch].decode_wm(yuv_attacked[..., ch], wm_shape[ch]) * 255
        recovered_channels.append(rec_wm.astype(np.uint8))
    return cv2.merge(recovered_channels)

# 主流程重构
if __name__ == '__main__':
    # 参数设置
    strength = 10
    block_size = 8
    wm_path = "watermark.png"
    host_path = "host_image.png"

    # 数据准备
    wm_img = prepare_image(wm_path)
    host_img = prepare_image(host_path)
    host_backup = host_img.copy()
    wm_bin = np.where(wm_img < np.mean(wm_img, axis=(0,1)), 0, 1)

    # 分通道处理
    processed = []
    recovered_wm = []
    yuv_host = cv2.cvtColor(host_img, cv2.COLOR_RGB2YUV)
    
    for ch in range(3):
        processor = WM_Processor(yuv_host[..., ch], wm_bin[..., ch], 
                                blk_size=block_size, strength=strength)
        dct_blocks = processor.block_process(yuv_host[..., ch])
        encoded = processor.encode_wm(dct_blocks, wm_bin[..., ch])
        synth = processor.inverse_dct(encoded)
        processed.append(synth)
        rec_wm = processor.decode_wm(synth, wm_bin[..., ch].shape) * 255
        recovered_wm.append(rec_wm.astype(np.uint8))

    # 结果合并展示
    final_img = cv2.merge(processed)
    final_wm = cv2.merge(recovered_wm)
    
    fig, axs = plt.subplots(2, 2, figsize=(10,8))
    display_data = [host_backup, wm_img, final_img, final_wm]
    titles = ["原图", "水印", "合成图", "提取水印"]
    
    for i, ax in enumerate(axs.flat):
        ax.imshow(display_data[i])
        ax.set_title(titles[i])
        ax.axis('off')
    plt.tight_layout()
    plt.show()

    


    # === 添加鲁棒性攻击测试 ===
    attack_list = ["gaussian_noise", "rotate", "crop", "contrast", "jpeg", "blur"]
    wm_shape_channels = [wm_bin[..., ch].shape for ch in range(3)]

    for attack in attack_list:
        attacked_img = apply_attack(final_img, attack)
        attacked_wm = test_robustness(attacked_img, [processor]*3, wm_shape_channels)
        
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        axs[0].imshow(attacked_img)
        axs[0].set_title(f"攻击后图像 ({attack})")
        axs[0].axis('off')
        axs[1].imshow(attacked_wm)
        axs[1].set_title("提取水印")
        axs[1].axis('off')
        plt.tight_layout()
        plt.show()

