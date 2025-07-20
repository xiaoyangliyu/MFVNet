import json

# 加载JSON文件
with open('dataset/AI challenge 2018/train/AgriculturalDisease_train_annotations.json', 'r') as f:
    annotations = json.load(f)

# 打印每个标注的键名，以检查结构
for annotation in annotations:
    print(annotation.keys())  # 打印出每个标注记录的键名
    # 你可以只打印一次，或者设定条件限制打印
    break
