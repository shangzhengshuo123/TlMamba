import re
import logging
import os
import json
import torch
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch.autograd import Variable
from tqdm import tqdm
from models.TlMamba import VSSM as tlmamba  # 导入模型
from collections import Counter, defaultdict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE


# 定义清理字符串的函数
def clean_string(s):
    return re.sub(r'[\ud800-\udfff]', '', s)

# 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using {} device.".format(device))

# 定义类别字典（根据训练时生成的JSON文件加载）
json_path = ""
assert os.path.exists(json_path), "class_indices.json file does not exist."
with open(json_path, 'r') as json_file:
    class_indict = json.load(json_file)

# 定义类别列表（确保顺序与训练时一致）
classes = list(class_indict.keys())

# 定义数据预处理
transform_test = transforms.Compose([
    # transforms.Resize((64, 64)),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),

    transforms.Normalize(mean=[0.90062904, 0.90062904, 0.90062904], std=[0.26650605, 0.26650605, 0.26650605])
])

# 加载模型
num_classes = len(class_indict)
model = tlmamba(num_classes=num_classes).to(device)
model_weight_path = ""
assert os.path.exists(model_weight_path), "Model weight file does not exist."
model.load_state_dict(torch.load(model_weight_path, map_location=device))
model.eval()

all_preds = []
all_targets = []
# 测试函数
def open_images_in_folders(folder):


    # 检查数据集路径和结构
    print(f"Test dataset folder: {folder}")
    sub_folders = [f for f in os.listdir(folder) if os.path.isdir(os.path.join(folder, f))]
    print(f"Number of sub-folders: {len(sub_folders)}")
    for sub_folder in sub_folders:
        sub_folder_path = os.path.join(folder, sub_folder)
        files = os.listdir(sub_folder_path)
        print(f"Sub-folder: {sub_folder}, Number of images: {len(files)}")

    for sub_folder in tqdm(sub_folders, desc="Processing folders"):
        sub_folder_path = os.path.join(folder, sub_folder)
        files = os.listdir(sub_folder_path)

        for file in files:
            file_path = os.path.join(sub_folder_path, file).replace("\\", '/')
            if file.endswith('.png'):
                try:
                    # 打开图像并预处理
                    image = Image.open(file_path)
                    # print(f"Image size: {image.size}")
                    image = image.convert('RGB')
                    img = transform_test(image).unsqueeze(0).to(device)
                    # print(f"Processed image shape: {img.shape}")

                    # 模型推理
                    with torch.no_grad():
                        output = model(img)
                        # print(f"Model output shape: {output.shape}")
                        # print(f"Model output: {output}")
                        _, pred = torch.max(output, 1)
                        # print(f"Predicted class index: {pred.item()}")
                        pred_label = classes[pred.item()]
                        true_label = sub_folder

                    # 记录预测结果和真实标签
                    all_preds.append(pred.item())
                    all_targets.append(int(class_indict[true_label]))

                #     print(f"Image Name: {file}, Predict: {pred_label}, Target: {true_label}")
                # except Exception as e:
                #     print(f"Error opening image at {file_path}: {e}")
                    # 清理字符串后打印
                    print(
                        f"Image Name: {clean_string(file)}, Predict: {clean_string(pred_label)}, Target: {clean_string(true_label)}")

                    # === 分类统计，用于后续tail类分析 ===
                    class_counts = Counter(all_targets)
                    tail_threshold = np.percentile(list(class_counts.values()), 25)  # 定义后25%样本数量为尾部类

                    tail_class_indices = [cls for cls, count in class_counts.items() if count <= tail_threshold]
                    tail_correct = 0
                    tail_total = 0

                    for pred, target in zip(all_preds, all_targets):
                        if target in tail_class_indices:
                            tail_total += 1
                            if pred == target:
                                tail_correct += 1

                    tail_accuracy = tail_correct / tail_total if tail_total > 0 else 0.0

                except Exception as e:
                    # 清理字符串后打印错误信息
                    print(f"Error opening image at {clean_string(file_path)}: {clean_string(str(e))}")


    # 检查预测结果和标签
    if len(all_preds) == 0 or len(all_targets) == 0:
        raise ValueError("No predictions or targets found. Check data loading and model output.")

    # 计算性能指标
    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    confusion_mat = confusion_matrix(all_targets, all_preds)

    return accuracy, precision, recall, f1, confusion_mat, tail_accuracy



# 主函数
def main():
    folder = ''  # 测试数据集路径
    # accuracy, precision, recall, f1, confusion_mat = open_images_in_folders(folder)
    #
    # print(f"Test accuracy: {accuracy:.4f}")
    # print(f"Test precision: {precision:.4f}")
    # print(f"Test recall: {recall:.4f}")
    # print(f"Test f1: {f1:.4f}")

    accuracy, precision, recall, f1, confusion_mat, tail_accuracy = open_images_in_folders(folder)

    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Test precision: {precision:.4f}")
    print(f"Test recall: {recall:.4f}")
    print(f"Test f1: {f1:.4f}")
    print(f"Tail class accuracy (last 25% classes): {tail_accuracy:.4f}")


    # 记录日志
    # logging.basicConfig(filename='xidaitest.log', level=logging.DEBUG, format='%(asctime)s:%(levelname)s:%(message)s')
    # logging.info(f"\n"
    #              f"Test accuracy: {accuracy:.4f}\n"
    #              f"Test precision: {precision:.4f}\n"
    #              f"Test recall: {recall:.4f}\n"
    #              f"Test f1: {f1:.4f}\n")

    # plt.figure(figsize=(12, 10))
    # sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues',
    #             xticklabels=classes, yticklabels=classes)
    # plt.title('Confusion Matrix')
    # plt.xlabel('Predicted Label')
    # plt.ylabel('True Label')
    # plt.xticks(rotation=45, ha='right')  # 旋转横轴标签
    # plt.yticks(rotation=0)
    # plt.tight_layout()
    # plt.savefig('confusion_matrix.png')
    # plt.close()


if __name__ == "__main__":
    main()

