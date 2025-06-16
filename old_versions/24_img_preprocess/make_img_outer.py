from PIL import Image, ImageDraw
import numpy as np

# 載入圖片
img = Image.open('/home/yclai/vscode_project/FastAPI_testing/sem_img.png').convert('L')
img_np = np.array(img)

# 在中間加上白色突起的函式
def add_white_bump(image_np, center_x, center_y, radius=30, intensity=150):
    img_copy = image_np.copy()
    for y in range(center_y - radius, center_y + radius):
        for x in range(center_x - radius, center_x + radius):
            if 0 <= x < img_copy.shape[1] and 0 <= y < img_copy.shape[0]:
                # 計算點到中心的距離
                dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                if dist <= radius:
                    # 距離越近中心，亮度越高，形成突起效果
                    bump_intensity = intensity * (1 - dist / radius)
                    img_copy[y, x] = min(255, img_copy[y, x] + bump_intensity)
    return img_copy

# 計算圖片中心
center_x = img_np.shape[1] // 2
center_y = img_np.shape[0] // 2

# 加上白色突起
img_with_bump = add_white_bump(img_np, center_x+35, center_y, radius=20, intensity=110)

# 儲存結果
result_img = Image.fromarray(img_with_bump)
result_img.save('image_with_bump.jpg')
result_img.show()

