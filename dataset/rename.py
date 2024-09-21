import os
import glob
import shutil
import cv2

def read_all_files(folder_path):
    # 获取文件夹下的所有文件路径（不递归）
    file_paths = glob.glob(os.path.join(folder_path, '*'))
    return file_paths

def put_zero(s):
    # 将索引号补足3位数的字符串格式
    return s.zfill(3)

def process_image(image_path):
    """
    处理图像，将其转为黑底白字的二值化图像，并调整分辨率为 96x96
    """
    # 读取图像
    image = cv2.imread(image_path)

    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 使用阈值分割生成二值化图像
    _, mask = cv2.threshold(gray, 165, 255, cv2.THRESH_BINARY)

    # 取反，生成黑底白字的图像
    mask_inv = cv2.bitwise_not(mask)

    # 调整分辨率为 96x96
    resized_image = cv2.resize(mask_inv, (96, 96), interpolation=cv2.INTER_LINEAR)

    return resized_image

def copy_and_rename_files(letter, file_paths, target_folder):
    # 确保目标文件夹存在
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    for index, file_path in enumerate(file_paths):
        index += 1  # 从1开始计数
        # 提取文件名和扩展名
        file_name, file_ext = os.path.splitext(os.path.basename(file_path))

        # 补足三位数索引
        idx = put_zero(str(index))

        # 创建新的文件名，按照 letter + 固定编号 + 三位数索引
        new_file_name = f"{letter}2227405008{idx}.png"  # 保存为png格式

        # 生成目标文件路径
        new_file_path = os.path.join(target_folder, new_file_name)

        # 处理图像并保存结果
        processed_image = process_image(file_path)
        cv2.imwrite(new_file_path, processed_image)

        print(f"Copied, processed, and renamed: {file_path} -> {new_file_path}")

if __name__ == "__main__":
    c='A';lst=[]
    for i in range(26):
        tmp=chr(ord(c)+i)
        lst.append(tmp)
    # 循环处理 6 个文件夹
    for i in range(0,26):
        # 每个源文件夹路径
        idx=i+11
        source_path = f"E:\\program\\VAE\\dataset\\raw\\Sample0{idx}"
        # 获取对应的字母前缀
        letter = lst[i]

        # 目标文件夹路径
        target_path = "E:\program\VAE\dataset\processed"

        # 读取指定文件夹的所有文件
        files = read_all_files(source_path)

        # 选取第 300 到 360 个文件
        selected_files = files

        # 复制、处理并重命名文件
        copy_and_rename_files(letter, selected_files, target_path)
