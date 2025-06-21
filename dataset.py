import os
import numpy as np
import requests
import cv2
from bs4 import BeautifulSoup
import OpenEXR
import Imath
import cv2

from utils import dc 

def exr_to_12bit_gray(exr_path):
    exr = OpenEXR.InputFile(exr_path)
    dw  = exr.header()['dataWindow']
    W   = dw.max.x - dw.min.x + 1
    H   = dw.max.y - dw.min.y + 1
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)

    # 讀三通道
    r = np.frombuffer(exr.channel('R', FLOAT), dtype=np.float32).reshape(H, W)
    g = np.frombuffer(exr.channel('G', FLOAT), dtype=np.float32).reshape(H, W)
    b = np.frombuffer(exr.channel('B', FLOAT), dtype=np.float32).reshape(H, W)
    Y = 0.2126*r + 0.7152*g + 0.0722*b

    # 2. 选一个合理的峰值（这里用 99% 百分位）
    peak = np.percentile(Y, 95)
    peak = max(peak, 1e-6)      # 避免除零

    # 3. Reinhard 映射
    Y_tm = Y / (1 + Y/peak)

    # 4. 再做最大值归一化
    Y_norm = np.clip(Y_tm / Y_tm.max(), 0, 1)

    # 5. 可选伽玛校正
    Y_gamma = Y_norm ** (1/2.2)

    # 6. 量化到 12-bit
    Yq = np.rint(Y_gamma * (2**15 - 1)).astype(np.uint16)

    # 線性灰階
    # Y = 0.2126*r + 0.7152*g + 0.0722*b

    # max_val = Y.max()
    # if max_val <= 0:
    #     max_val = 1e-6  # 避免除零
    
    # Y_norm = Y / max_val 
    # Y_norm = np.clip(Y_norm, 0.0, 1.0)
    # # Clamp & Normalize
    # # Y = np.clip(Y, 0.0, 1.0)
    # # Y = Y / 1.0

    # # Quantize to 12-bit
    # Yq = np.rint(Y_norm * (2**16 - 1)).astype(np.uint16)
    return Yq

def read_exr_as_gray16(path_exr):
    exr = OpenEXR.InputFile(path_exr)
    header = exr.header()
    dw = header['dataWindow']
    width  = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)

    r_str = exr.channel('R', FLOAT)
    g_str = exr.channel('G', FLOAT)
    b_str = exr.channel('B', FLOAT)

    r = np.frombuffer(r_str, dtype=np.float32).reshape((height, width))
    g = np.frombuffer(g_str, dtype=np.float32).reshape((height, width))
    b = np.frombuffer(b_str, dtype=np.float32).reshape((height, width))

    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    y = np.clip(y, 0.0, 1.0)
    y16 = (y * 65535.0).astype(np.uint16)

    return y16

def read_exr_as_floatY(path_exr):
    exr = OpenEXR.InputFile(path_exr)
    header = exr.header()
    dw = header["dataWindow"]
    width  = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    # 把原本的 half 或 float channel 一律讀成 32-bit float
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    r_str = exr.channel("R", FLOAT)
    g_str = exr.channel("G", FLOAT)
    b_str = exr.channel("B", FLOAT)

    r = np.frombuffer(r_str, dtype=np.float32).reshape((height, width))
    g = np.frombuffer(g_str, dtype=np.float32).reshape((height, width))
    b = np.frombuffer(b_str, dtype=np.float32).reshape((height, width))

    y_float = 0.2126 * r + 0.7152 * g + 0.0722 * b

    return y_float

def fetch_all_exr_links():
    BASE = "https://markfairchild.org/HDRPS/"
    THUMBS_URL = BASE + "HDRthumbs.html"
    resp = requests.get(THUMBS_URL, verify=False)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    exr_links = []

    for a in soup.find_all("a"):
        href = a.get("href")
        if not href:
            continue
        lower_href = href.lower()
        if lower_href.startswith("scenes/") and lower_href.endswith(".html"):
            scene_name = os.path.basename(href).rsplit(".", 1)[0]
            exr_url = BASE + "EXRs/" + scene_name + ".exr"
            exr_links.append(exr_url)

    exr_links = sorted(set(exr_links))
    return exr_links

def download_exr(output_dir='./HDR_datas'):
    all_links = fetch_all_exr_links()
    print(f"Finding {len(all_links)} .exr link:")
    for u in all_links:
        print(u)

    os.makedirs(output_dir, exist_ok=True)
    for url in all_links:
        fname = os.path.basename(url)
        local_path = os.path.join(output_dir, fname)
        if os.path.exists(local_path):
            continue
        print(f"Downloading {fname} ...")
        r = requests.get(url, stream=True, verify=False)
        if r.status_code == 200:
            with open(local_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        else:
            print(f"Failed: HTTP {r.status_code}")
    print("Done!")

def process_exr_folder(
    input_dir: str = './HDR_datas', output_dir: str = './HDR_datas/HDR_png', stats_csv_path: str = './HDR_datas/1nfo.csv'
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    exr_list = [
        fn for fn in os.listdir(input_dir)
        if fn.lower().endswith(".exr")
    ]
    exr_list.sort()

    if len(exr_list) == 0:
        print(f"no .exr in: {input_dir}, download it first!")
        return

    stats_list = []
    for filename in exr_list:
        exr_path = os.path.join(input_dir, filename)
        print(f"Handling：{filename}")

        img = exr_to_12bit_gray(exr_path)
        
        h, w = img.shape[:2]

        if img.ndim == 2:
            channels = 1
        else:
            channels = img.shape[2]

        stats = {
            "filename": filename,
            "width": w,
            "height": h,
            "channels": channels
        }
        base = os.path.splitext(filename)[0]
        out_path = os.path.join(output_dir, base + ".png")
        cv2.imwrite(out_path, img)

    #     if channels >= 3:
    #         B = img[:, :, 0].astype(np.float32)
    #         G = img[:, :, 1].astype(np.float32)
    #         R = img[:, :, 2].astype(np.float32)

    #         stats["min_R"]  = float(np.min(R))
    #         stats["max_R"]  = float(np.max(R))
    #         stats["mean_R"] = float(np.mean(R))
    #         stats["std_R"]  = float(np.std(R))

    #         stats["min_G"]  = float(np.min(G))
    #         stats["max_G"]  = float(np.max(G))
    #         stats["mean_G"] = float(np.mean(G))
    #         stats["std_G"]  = float(np.std(G))

    #         stats["min_B"]  = float(np.min(B))
    #         stats["max_B"]  = float(np.max(B))
    #         stats["mean_B"] = float(np.mean(B))
    #         stats["std_B"]  = float(np.std(B))

    #         Y = 0.2126 * R + 0.7152 * G + 0.0722 * B

    #         stats["min_Y"]  = float(np.min(Y))
    #         stats["max_Y"]  = float(np.max(Y))
    #         stats["mean_Y"] = float(np.mean(Y))
    #         stats["std_Y"]  = float(np.std(Y))

  
    #     else:
    #         stats["min_Y"]  = float(np.min(Y))
    #         stats["max_Y"]  = float(np.max(Y))
    #         stats["mean_Y"] = float(np.mean(Y))
    #         stats["std_Y"]  = float(np.std(Y))

    #         base = os.path.splitext(filename)[0]
    #         out_path = os.path.join(output_dir, base + ".png")
    #         cv2.imwrite(out_path, Y)

    #     stats_list.append(stats)

    # if len(stats_list) > 0:
    #     keys = list(stats_list[0].keys())
    #     with open(stats_csv_path, "w", newline="") as csvfile:
    #         writer = csv.DictWriter(csvfile, fieldnames=keys)
    #         writer.writeheader()
    #         for row in stats_list:
    #             writer.writerow(row)
    #     print(f"Done! statistic result is stored in : {stats_csv_path}")

def filter_dataset():
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
            # print(dc_array.shape)
            # print(np.max(dc_array), np.min(dc_array))

    print(f"Total images with shape (4288, 2848): {count}")


if __name__ == "__main__":
    # download_exr(output_dir='./HDR_datas')
    process_exr_folder(
        input_dir='./HDR_raw',
        output_dir='./HDR_raw/HDR_png',
        stats_csv_path='./HDR_datas/1nfo.csv'
    )