import os
import shutil
import yaml
from pathlib import Path
import re

def load_yaml(file_path):
    """加载 YAML 配置文件"""
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)


def create_class_dirs(base_path, classes):
    """根据标签类别创建新的目录结构"""
    for label in classes:
        class_name = str(label)  # Convert the label to a string
        class_dir = base_path / class_name  # Now combine the path and string
        if not class_dir.exists():
            class_dir.mkdir(parents=True, exist_ok=True)


def get_labels_from_txt(txt_file):
    """从标签文件中提取类别标签"""
    labels = []
    with open(txt_file, 'r') as f:
        for line in f:
            label = int(line.split()[0])  # 获取标签（第一列数字）
            labels.append(label)
    return labels


def clean_filename(filename):
    """清理文件名中的非法字符"""
    return re.sub(r'[<>:"/\\|?*]', '_', filename)


def move_images_by_label(images, labels, dest_base_dir, class_names):
    """根据标签将图片移动到对应的类别目录中"""
    for img_path, label in zip(images, labels):
        if label >= len(class_names):
            print(f"Warning: Label {label} is out of range. Skipping {img_path}")
            continue

        class_name = class_names[label]  # 获取标签对应的类别名称
        dest_dir = dest_base_dir / class_name
        img_name = clean_filename(img_path.name)  # 清理文件名中的非法字符
        dest_img_path = dest_dir / img_name

        # 检查源文件是否存在
        if not img_path.exists():
            print(f"Warning: Source image {img_path} does not exist. Skipping.")
            continue

        # 确保目标文件夹存在，包括所有父文件夹
        if not dest_img_path.parent.exists():
            print(f"Creating directory: {dest_img_path.parent}")
            os.makedirs(dest_img_path.parent, exist_ok=True)  # 创建目录

        # 使用 try-except 捕获 FileNotFoundError 并给出警告
        try:
            print(f"Moving {img_path} to {dest_img_path}")
            shutil.copy(img_path, dest_img_path)  # 将图片复制到目标目录
        except FileNotFoundError as e:
            print(f"Warning: Failed to move {img_path} to {dest_img_path}. Error: {e}. Skipping.")


def process_data_yaml(yaml_file, src_base_dir, dest_base_dir):
    """根据 data.yaml 文件转换数据集"""
    data = load_yaml(yaml_file)

    # 获取训练、验证和测试数据路径
    train_images_dir = Path(data['train'])
    val_images_dir = Path(data['val'])
    test_images_dir = Path(data['test'])

    # 获取标签类别
    class_names = data['names']

    # 创建类别目录结构
    create_class_dirs(dest_base_dir, class_names)

    # 处理训练集图像
    print("Processing training images...")
    train_images = list(train_images_dir.glob('**/*.jpg')) + list(
        train_images_dir.glob('**/*.png'))  # 假设图像为 .jpg 或 .png 格式
    train_labels = []

    for img in train_images:
        label_txt_file = (train_images_dir.parent / 'labels' / (img.stem + '.txt'))  # 标签文件路径与图像文件名一致，扩展名为 .txt

        # 检查标签文件是否存在
        if label_txt_file.exists():
            labels = get_labels_from_txt(label_txt_file)  # 获取标签
            train_labels.extend(labels)
            print(f"Labels for {img}: {labels}")  # 打印标签信息
        else:
            print(f"Warning: No label file for {img}. Skipping.")  # 没有标签文件的警告

    # 确保每个图像和标签都有对应的存在，跳过没有标签的图像
    if len(train_images) != len(train_labels):
        print("Warning: Mismatch between images and labels. Some images might be skipped.")

    move_images_by_label(train_images, train_labels, dest_base_dir, class_names)

    # 处理验证集图像
    print("Processing validation images...")
    val_images = list(val_images_dir.glob('**/*.jpg')) + list(val_images_dir.glob('**/*.png'))
    print(f"Total {len(val_images)} validation images found.")
    val_labels = []

    for img in val_images:
        label_txt_file = (val_images_dir.parent / 'labels' / (img.stem + '.txt'))  # 标签文件路径
        if label_txt_file.exists():
            labels = get_labels_from_txt(label_txt_file)  # 获取标签
            val_labels.extend(labels)
        else:
            print(f"Warning: No label file for {img}. Skipping.")

    move_images_by_label(val_images, val_labels, dest_base_dir, class_names)

    # 处理测试集图像
    print("Processing test images...")
    test_images = list(test_images_dir.glob('**/*.jpg')) + list(test_images_dir.glob('**/*.png'))
    print(f"Total {len(test_images)} test images found.")
    test_labels = []

    for img in test_images:
        label_txt_file = (test_images_dir.parent / 'labels' / (img.stem + '.txt'))  # 标签文件路径
        if label_txt_file.exists():
            labels = get_labels_from_txt(label_txt_file)  # 获取标签
            test_labels.extend(labels)
        else:
            print(f"Warning: No label file for {img}. Skipping.")

    move_images_by_label(test_images, test_labels, dest_base_dir, class_names)


if __name__ == "__main__":
    # 配置路径
    src_base_dir = Path(r'dataset/AI challenge 2018')  # 原始数据集目录（包含 train、test、data.yaml）
    dest_base_dir = Path('dataset/AI')  # 新的数据集目录
    yaml_file = r'dataset/AI challenge 2018/data.yaml'  # data.yaml 文件路径

    # 执行数据转换
    process_data_yaml(yaml_file, src_base_dir, dest_base_dir)
    print("Dataset transformation completed!")


