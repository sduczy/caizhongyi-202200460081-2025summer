from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import os
 
class WatermarkText():
    def __init__(self):
        super(WatermarkText, self).__init__()
        self.image_path = input('图片路径：')
        self.watermark_text = input('水印文字：')
        self.position_flag = int(input('水印位置（1：左上角，2：左下角，3：右上角，4：右下角，5：居中）：'))
        self.opacity = float(input('水印透明度（0—1之间的1位小数）：'))

        # 设置字体（注意你的字体文件路径要正确）
        self.font = ImageFont.truetype("cambriab.ttf", size=35)

    # 文字水印
    def add_text_watermark(self, img):
        global location
        image = Image.open(img).convert('RGBA') 
        new_img = Image.new('RGBA', image.size, (255, 255, 255, 0)) 
        image_draw = ImageDraw.Draw(new_img) 
        w, h = image.size  # 图片宽度和高度

        # 使用 getbbox 替代 getsize 获取文字宽高
        bbox = self.font.getbbox(self.watermark_text)
        w1 = bbox[2] - bbox[0]  # 文字宽度
        h1 = bbox[3] - bbox[1]  # 文字高度

        # 设置水印文字位置
        if self.position_flag == 1:  # 左上角
            location = (0, 0)
        elif self.position_flag == 2:  # 左下角
            location = (0, h - h1)
        elif self.position_flag == 3:  # 右上角
            location = (w - w1, 0)
        elif self.position_flag == 4:  # 右下角
            location = (w - w1, h - h1)
        elif self.position_flag == 5:  # 居中
            location = ((w - w1)//2, (h - h1)//2)  # 注意这里要用宽w，高h，不是h/2,h/2

        # 绘制文字，fill 可以改成你想要的颜色及透明度，比如带alpha的RGBA元组
        image_draw.text(location, self.watermark_text, font=self.font, fill="blue")

        # 设置透明度
        transparent = new_img.split()[3]
        transparent = ImageEnhance.Brightness(transparent).enhance(self.opacity)
        new_img.putalpha(transparent)

        # 合成图像并保存
        result = Image.alpha_composite(image, new_img)
        result.save(img)
        print(f'水印添加成功，保存文件：{img}')

if __name__ == "__main__":
    watermark_text = WatermarkText()
    image_path = input("图片路径：").strip()

    print(f"你输入的路径是：{image_path}")
    print(f"当前工作目录是：{os.getcwd()}")
    print(f"文件是否存在：{os.path.exists(image_path)}")

    if not os.path.exists(image_path):
        print('输入的文件路径有误，请检查')
    else:
        if os.path.isfile(image_path):
            # 单张图片，直接处理
            ext = os.path.splitext(image_path)[1].lower()
            if ext == '.png':
                watermark_text.add_text_watermark(image_path)
                print('图片处理完成')
            else:
                print('图片格式有误，请使用png格式图片')
        elif os.path.isdir(image_path):
            # 目录，批量处理
            file_list = os.listdir(image_path)
            for filename in file_list:
                filepath = os.path.join(image_path, filename)
                if os.path.isfile(filepath) and filepath.lower().endswith('.png'):
                    watermark_text.add_text_watermark(filepath)
                else:
                    print(f"跳过非png文件：{filename}")
            print('批量添加水印完成')
        else:
            print('路径不是文件也不是目录，请检查输入')
