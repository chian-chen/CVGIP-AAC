import numpy as np
import os
from PIL import Image
import cv2


from utils import dc 


# # 開啟檔案
# exr = OpenEXR.InputFile('./HDR_datas/507.exr')

# # 取得 header dict
# hdr = exr.header()
# for key, value in hdr.items():
#     print(f"{key}: {value}")


base_dir = './HDR_raw/HDR_png'
file_list = [
    os.path.join(base_dir, fn) for fn in os.listdir(base_dir)
]
print(f"Total files found: {len(file_list)}")
file_list.sort()
count = 0
for file in file_list:
    img16 = cv2.imread(file, cv2.IMREAD_UNCHANGED)
    im = np.array(img16).astype(np.int16)  # convert to int16 for further processing
    if im.shape == (2848, 4288):
        count += 1
        print(file)
        dc_array = dc(im)
        dc_vector = dc_array.flatten()
        pred_value = np.diff(dc_vector, n=1)
        print(max(pred_value), min(pred_value))
        break
        # print(dc_array.shape)
        # print(np.max(dc_array), np.min(dc_array))

print(f"Total images with shape (4288, 2848): {count}")