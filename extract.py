import os
from shutil import copyfile

# 这个脚本是将没有label的图像从数据集剔除的脚本
images = "videos/images"
labels = "data/labels"

output_images = "output/images"
output_labels = "output/labels"

os.makedirs(output_images)
os.makedirs(output_labels)

for img_name in os.listdir(images):
    img_name = img_name[:-4] # 移除.jpg
    label = os.path.join(labels, img_name + ".txt")
    if os.path.exists(label):
        input_image = os.path.join(images, img_name + ".jpg")
        input_label = os.path.join(labels, img_name + ".txt")
        output_image = os.path.join(output_images, img_name + ".jpg")
        output_label = os.path.join(output_labels, img_name + ".txt")
        copyfile(input_image, output_image)
        copyfile(input_label, output_label)