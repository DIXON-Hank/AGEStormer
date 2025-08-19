import os

def rename_files(folder_path):
    # 检查文件夹路径是否存在
    if not os.path.exists(folder_path):
        print(f"文件夹 {folder_path} 不存在！")
        return
    
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 检查文件是否符合 "_clear.png" 的模式
        if filename.endswith("_rain.png"):
            # 获取数字部分（去掉 "_clear"）
            new_name = filename.replace("_rain", "")
            old_file = os.path.join(folder_path, filename)
            new_file = os.path.join(folder_path, new_name)
            
            # 重命名文件
            os.rename(old_file, new_file)
            print(f"重命名: {filename} -> {new_name}")
    
    print("重命名完成！")

# 使用示例
folder_path ="/home/gagagk16/Rain/Derain/Dataset/RainGAN/train/raindrop/val/"
rename_files(folder_path)
