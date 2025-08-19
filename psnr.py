import os
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from PIL import Image

def calculate_metrics(image1, image2):
    """
    计算 PSNR 和 SSIM。

    Args:
        input_image (numpy.ndarray): 输入图像。
        output_image (numpy.ndarray): 输出图像。

    Returns:
        tuple: PSNR 和 SSIM。
    """
    # 确保 input_image 和 output_image 尺寸一致
    if image1.shape[:2] != image2.shape[:2]:
        input_pil = Image.fromarray(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
        output_height, output_width = image2.shape[:2]
        print(output_height, output_width)
        input_pil = input_pil.resize((max(image1.shape[1], output_width), max(image1.shape[0], output_height)), Image.Resampling.LANCZOS)
        image1 = cv2.cvtColor(np.array(input_pil), cv2.COLOR_RGB2BGR)

    psnr_value = psnr(image1, image2, data_range=255)
    ssim_value = ssim(image1, image2, multichannel=True, data_range=255)
    return psnr_value, ssim_value

def process_folder(folder_path):
    """
    遍历文件夹，计算所有数字前缀的 _input.png 和 _output.png 的 PSNR 和 SSIM。

    Args:
        folder_path (str): 文件夹路径。
    """
    psnr_list = []
    ssim_list = []

    for root, _, files in os.walk(folder_path):
        # 找到所有 _input.png 和 _output.png 文件
        input_files = [f for f in files if f.endswith("_gt.png")]
        output_files = [f for f in files if f.endswith("_output.png")]

        # 匹配数字前缀的文件对
        for input_file in input_files:
            prefix = input_file.split("_gt.png")[0]  # 提取数字前缀
            output_file = f"{prefix}_output.png"

            if output_file in output_files:
                input_path = os.path.join(root, input_file)
                output_path = os.path.join(root, output_file)

                # 读取图像
                input_image = cv2.imread(input_path)
                output_image = cv2.imread(output_path)

                # 确保图像有效
                if input_image is None or output_image is None:
                    print(f"Failed to read images: {input_path} or {output_path}")
                    continue

                # 计算 PSNR 和 SSIM
                psnr_value, ssim_value = calculate_metrics(input_image, output_image)

                psnr_list.append(psnr_value)
                ssim_list.append(ssim_value)

                print(f"Pair: {input_file} and {output_file}")
                print(f"PSNR: {psnr_value:.2f}, SSIM: {ssim_value:.4f}")

    # 计算平均值
    if psnr_list and ssim_list:
        avg_psnr = np.mean(psnr_list)
        avg_ssim = np.mean(ssim_list)
        print("\n--- Overall Metrics ---")
        print(f"Average PSNR: {avg_psnr:.2f}")
        print(f"Average SSIM: {avg_ssim:.4f}")
    else:
        print("No valid image pairs found.")

if __name__ == "__main__":
    folder_path = "/home/gagagk16/Rain/Derain/AGEStormer/data/testDS"
    process_folder(folder_path)
