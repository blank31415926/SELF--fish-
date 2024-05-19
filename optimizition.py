"""
Author:  Blank
Date: 2024/5/19
Description: 
"""
from pathlib import Path
import numpy as np


from PIL import Image, ImageSequence
path = list(Path("./").glob("SElf *.png"))  # 根据实际情况修改文件名或扩展名

# 读取图像序列
images = [Image.open(img_path) for img_path in path]

# 将图像序列保存为 GIF
images[0].save("output.gif", save_all=True, append_images=images[1:], duration=500, loop=0)

