import os
import numpy as np
import csv
import requests
from bs4 import BeautifulSoup
import OpenEXR
import Imath


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
    input_dir: str = './HDR_datas', output_dir: str = './HDR_datas/HDR_npy', stats_csv_path: str = './HDR_datas/1nfo.csv'
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

        img = read_exr_as_floatY(exr_path)
        if img is None:
            print(f"The img: {filename}, load failed, skip it.")
            continue

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

        if channels >= 3:
            B = img[:, :, 0].astype(np.float32)
            G = img[:, :, 1].astype(np.float32)
            R = img[:, :, 2].astype(np.float32)

            stats["min_R"]  = float(np.min(R))
            stats["max_R"]  = float(np.max(R))
            stats["mean_R"] = float(np.mean(R))
            stats["std_R"]  = float(np.std(R))

            stats["min_G"]  = float(np.min(G))
            stats["max_G"]  = float(np.max(G))
            stats["mean_G"] = float(np.mean(G))
            stats["std_G"]  = float(np.std(G))

            stats["min_B"]  = float(np.min(B))
            stats["max_B"]  = float(np.max(B))
            stats["mean_B"] = float(np.mean(B))
            stats["std_B"]  = float(np.std(B))

            Y = 0.2126 * R + 0.7152 * G + 0.0722 * B

            stats["min_Y"]  = float(np.min(Y))
            stats["max_Y"]  = float(np.max(Y))
            stats["mean_Y"] = float(np.mean(Y))
            stats["std_Y"]  = float(np.std(Y))

            base = os.path.splitext(filename)[0]
            out_path = os.path.join(output_dir, base + ".npy")
            np.save(out_path, Y)
        else:
            Y = img.astype(np.float32)

            stats["min_Y"]  = float(np.min(Y))
            stats["max_Y"]  = float(np.max(Y))
            stats["mean_Y"] = float(np.mean(Y))
            stats["std_Y"]  = float(np.std(Y))

            base = os.path.splitext(filename)[0]
            out_path = os.path.join(output_dir, base + ".npy")
            np.save(out_path, Y)

        stats_list.append(stats)

    if len(stats_list) > 0:
        keys = list(stats_list[0].keys())
        with open(stats_csv_path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=keys)
            writer.writeheader()
            for row in stats_list:
                writer.writerow(row)
        print(f"Done! statistic result is stored in : {stats_csv_path}")


if __name__ == "__main__":
    # download_exr(output_dir='./HDR_datas')
    process_exr_folder(
        input_dir='./HDR_datas',
        output_dir='./HDR_datas/HDR_npy',
        stats_csv_path='./HDR_datas/1nfo.csv'
    )