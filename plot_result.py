import matplotlib.pyplot as plt
import pandas as pd

# 绘制PR
def plot_PR():
    pr_csv_dict = {
        'RT-DETR-r18': r'E:\yolov9-main\runs\val\RTDTER-plantdoc\PR_curve.csv',
        'MoblileNetV4-Conv-S': r'E:\yolov9-main\runs\val\MobilenetV4_S_AI\PR_curve.csv',
        'MoblileNetV4-Conv-M': r'E:\yolov9-main\runs\val\MNV4_AI\PR_curve.csv',
        'YOLOv9-T': r'E:\yolov9-main\runs\val\exp6\PR_curve.csv',
        'YOLOv10n': r'E:\yolov9-main\runs\val\v10-AIcha\PR_curve.csv',
        'YOLOv11': r'E:\yolov9-main\runs\val\v11-AIcha\PR_curve.csv',
        'MFVNet': r'E:\yolov9-main\runs\val\MFVNet-AIcha\PR_curve.csv',
    }

    plt.rcParams['font.family'] = 'Times New Roman'  # 设置字体
    plt.rcParams['font.size'] = 12  # 设置全局字体大小
    # 绘制pr
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=150, tight_layout=True)


    for modelname in pr_csv_dict:
        res_path = pr_csv_dict[modelname]
        x = pd.read_csv(res_path, usecols=[1]).values.ravel()
        data = pd.read_csv(res_path, usecols=[2]).values.ravel()
        ax.plot(x, data, label=modelname, linewidth='2')

    # 添加x轴和y轴标签
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(loc='lower left', bbox_to_anchor=(0.02, 0.02))
    # plt.grid()  # 显示网格线
    # 显示图像
    fig.savefig("pr_AI.png", dpi=500)
    plt.show()

# 绘制F1
def plot_F1():
    f1_csv_dict = {
        'RT-DETR-r18': r'E:\yolov9-main\runs\val\RTDETR-AI\F1_curve.csv',
        'YOLOv9-s': r'E:\yolov9-main\runs\val\v9s-AI\F1_curve.csv',
        'YOLOv10n': r'E:\yolov9-main\runs\val\v10-AIcha\F1_curve.csv',
        'YOLOv11': r'E:\yolov9-main\runs\val\v11-AIcha\F1_curve.csv',
        'MFVNet': r'E:\yolov9-main\runs\val\MFVNet-AIcha\F1_curve.csv',
    }

    plt.rcParams['font.family'] = 'Times New Roman'  # 设置字体
    plt.rcParams['font.size'] = 12  # 设置全局字体大小
    fig, ax = plt.subplots(1, 1, figsize=(7, 8), dpi=150, tight_layout=True)


    for modelname in f1_csv_dict:
        res_path = f1_csv_dict[modelname]
        x = pd.read_csv(res_path, usecols=[1]).values.ravel()
        data = pd.read_csv(res_path, usecols=[2]).values.ravel()
        ax.plot(x, data, label=modelname, linewidth='3')

    # 添加x轴和y轴标签
    ax.set_xlabel('Confidence')
    ax.set_ylabel('F1')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(loc='lower left', bbox_to_anchor=(0.02, 0.02))
    # plt.grid()  # 显示网格线
    # 显示图像
    fig.savefig("F1.png", dpi=500)
    plt.show()

if __name__ == '__main__':
    plot_PR()   # 绘制PR
    plot_F1()   # 绘制F1
