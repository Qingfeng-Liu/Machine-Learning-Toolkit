import os
import glob
from datetime import datetime


def rename_ipynb_from_pivot(folder_path, pivot_file, start_num=21):
    # 获取所有.ipynb文件
    files = [f for f in glob.glob(os.path.join(folder_path, '*.ipynb'))
             if os.path.isfile(f)]

    # 按创建时间排序（旧→新）
    files.sort(key=lambda x: os.path.getctime(x))

    try:
        # 找到基准文件的位置
        pivot_path = next(f for f in files if os.path.basename(f) == pivot_file)
        pivot_index = files.index(pivot_path)

        # 从基准文件开始编号（包括该文件）
        for i, file_path in enumerate(files[pivot_index:], start=start_num):
            dir_name = os.path.dirname(file_path)
            original_name = os.path.basename(file_path)

            # 跳过已编号的文件
            if original_name.split('_')[0].isdigit():
                continue

            new_name = f"{i}_{original_name}"
            new_path = os.path.join(dir_name, new_name)

            os.rename(file_path, new_path)
            print(f"已处理: {original_name} → {new_name}")

    except StopIteration:
        print(f"错误：文件夹中未找到 {pivot_file}")


# 使用示例
rename_ipynb_from_pivot(
    folder_path="./",
    pivot_file="housing_linear_regression_example.ipynb",
    start_num=1
)