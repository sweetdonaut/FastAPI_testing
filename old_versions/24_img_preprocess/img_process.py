from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def add_sem_noise(image_path, output_path, noise_level=0.2):
    # 讀取圖片並轉為灰階
    img = Image.open(image_path).convert('L')
    img_np = np.array(img)
    
    # 產生高斯雜訊
    noise = np.random.normal(0, noise_level * 255, img_np.shape)
    
    # 產生橫向條紋
    stripes = (np.sin(np.linspace(0, 20 * np.pi, img_np.shape[0])) * 20).reshape(-1, 1)
    stripes = np.repeat(stripes, img_np.shape[1], axis=1)
    
    # 合併雜訊
    noisy_img = img_np + noise #+ stripes
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    
    # 儲存與顯示
    noisy_pil = Image.fromarray(noisy_img)
    noisy_pil.save(output_path)
    plt.imshow(noisy_img, cmap='gray')
    plt.axis('off')
    plt.show()

# 範例用法
add_sem_noise('/home/yclai/vscode_project/FastAPI_testing/sem_img.png', './sem_noisy_output_raw.jpg')
