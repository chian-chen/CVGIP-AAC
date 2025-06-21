import numpy as np
import os
from PIL import Image
import argparse
from scipy.stats import norm

from prediction_methods import edp_predict, gap_predict, med_predict
from utils import dc, enc, calculate_entropy, find_files

DATASET_DIRS = {
    'datas': './datas',
    'kodak': './Kodak',
    'hdr': './HDR_datas/HDR_npy',
}

def parse_args():
    parser = argparse.ArgumentParser(description="CAAC Settings")
    parser.add_argument('--datasets', nargs='+', default=['datas'],
                        help='Dataset names or paths to process')
    parser.add_argument('--output', type=str, default='./output',
                        help='Output folder for processed images')
    parser.add_argument('--prediction_methods', nargs='+', help='Prediction method to use')
    parser.add_argument('--context_settings', nargs='+', help='Context settings for the prediction methods')
    parser.add_argument('--context_type_features', nargs='+', help='Context features for the prediction methods')
    parser.add_argument('--visualization', action='store_true', help='Enable visualization of the prediction process')
    parser.add_argument('--context_features_num', type=int, default=4, help='Number of context features to use')
    return parser.parse_args()

def load_image(path: str) -> np.ndarray:
    """Load image file (bmp/png/jpg/npy) as int16 numpy array."""
    ext = os.path.splitext(path)[1].lower()
    if ext == '.npy':
        img = np.load(path)
    else:
        img = Image.open(path).convert('L')
        img = np.array(img)
    return img.astype(np.int16)

def get_prediction_image(image, prediction_method):
    if prediction_method == 'EDP':
        return edp_predict(image)
    elif prediction_method == 'GAP':
        return gap_predict(image)
    elif prediction_method == 'MED':
        return med_predict(image)
    else:
        flat = image.flatten()
        pred_flat = np.zeros_like(flat)
        pred_flat[1:] = flat[:-1]
        return pred_flat.reshape(image.shape)

def get_context_element(image, prediction_image, i, j):
    h, w = image.shape
    if i < 0 or i >= h or j < 0 or j >= w:
        return 0
    return np.abs(image[i, j] - prediction_image[i, j])

def get_context_features(image, prediction_image, context_type_features):
    h, w = image.shape
    features = np.zeros((h, w, 4), dtype=np.float32)
    for i in range(h):
        for j in range(w):
            features[i, j, 0] = get_context_element(image, prediction_image, i - 1, j)
            features[i, j, 1] = get_context_element(image, prediction_image, i - 1, j - 1)
            features[i, j, 2] = get_context_element(image, prediction_image, i, j - 1)
            features[i, j, 3] = get_context_element(image, prediction_image, i - 1, j + 1)
    return features

def context_map(context_features):
    value = np.sum(context_features) * 2
    if value <= 8:
        return 0
    elif value <= 20:
        return 1
    elif value <= 40:
        return 2
    else:
        return 3

def apply_CAAC_method(image, prediction_image, context_features):
    h, w = image.shape
    symbols = 256
    encoded_image = image - prediction_image
    Lw, Up = 0.0, 1.0
    code_str = ""

    base_tb = np.exp(-0.15 * np.abs(np.arange(symbols)))
    base_tb = base_tb / base_tb.sum() * 100
    base_tb = np.maximum(base_tb, 1.0)
    p = np.tile(base_tb, (4, 1))

    p_sign = np.ones((4, 2), dtype=np.float64)
    
    alpha = 0.1

    context = 0
    Lw, Up, cnew = enc(encoded_image[0][0], Lw, Up, p[context])
    code_str += cnew

    for i in range(h):
        for j in range(w):
            if i == 0 and j == 0:
                continue
            context = context_map(context_features[i, j])
            d = abs(encoded_image[i][j])
            if d > symbols - 1:
                d = symbols - 1
            Lw, Up, cnew = enc(d, Lw, Up, p[context])
            code_str += cnew

            if d != 0:
                sign_val = 1 if (encoded_image[i][j]) > 0 else 0
                Lw, Up, cnew = enc(sign_val, Lw, Up, p_sign[context])
                code_str += cnew
                p_sign[context][sign_val] += 1 - weight
                p_sign[context + 1][sign_val] += weight
                if p_sign[context].sum() > 1e6:
                    p_sign[context] = np.maximum(p_sign[context] / 2, 0.01)
                if p_sign[context + 1].sum() > 1e6:
                    p_sign[context + 1] = np.maximum(p_sign[context + 1] / 2, 0.01)

            new_p = norm.pdf(np.arange(symbols), loc=d, scale=2)
            new_p = new_p / new_p.sum() * 100
            p[context] = (1 - alpha) * p[context] + alpha * new_p
            if p[context].sum() > 1e6:
                p[context] = np.maximum(p[context] / 2, 0.01)

    bn = 0
    fn = (Up < 1.0) or (Lw > 0.0)
    while fn:
        bn += 1
        Lw *= 2
        Up *= 2
        fn = Up < (np.ceil(Lw) + 1)

    if bn > 0:
        code_str += format(int(np.ceil(Lw)), f'0{bn}b')

    return code_str

def mixed_context_map(context_features):
    value = np.sum(context_features) * 2
    center1, center2, center3, center4 = 4, 14, 30, 60
    c1, w = 0, 0
    if value <= center1:
        c1 = 0
        w = 0
    elif value <= center2:
        c1 = 0
        w = (value - center1) / (center2 - center1)
    elif value <= center3:
        c1 = 1
        w = (value - center2) / (center3 - center2)
    elif value <= center4:
        c1 = 2
        w = (value - center3) / (center4 - center3)
    else:
        c1 = 2
        w = 1.0
    return c1, w

def apply_MIXED_CAAC_method(image, prediction_image, context_features):
    h, w = image.shape
    symbols = 256
    encoded_image = image - prediction_image
    Lw, Up = 0.0, 1.0
    code_str = ""

    base_tb = np.exp(-0.1 * np.abs(np.arange(symbols)))
    base_tb = base_tb / base_tb.sum() * 100
    base_tb = np.maximum(base_tb, 1.0)
    p = np.tile(base_tb, (4, 1))

    p_sign = np.ones((4, 2), dtype=np.float64)

    alpha = 0.1

    context = 0
    Lw, Up, cnew = enc(encoded_image[0][0], Lw, Up, p[context])
    code_str += cnew

    for i in range(h):
        for j in range(w):
            if i == 0 and j == 0:
                continue
            context, weight = mixed_context_map(context_features[i, j])
            if context >= 3:
                context = 2
                weight = 1.0
            current_p = p[context] * (1 - weight) + p[context + 1] * weight
            d = abs(encoded_image[i][j])
            if d > symbols - 1:
                d = symbols - 1
            Lw, Up, cnew = enc(d, Lw, Up, current_p)
            code_str += cnew

            if d != 0:
                sign_val = 1 if (encoded_image[i][j]) > 0 else 0
                Lw, Up, cnew = enc(sign_val, Lw, Up, p_sign[context])
                code_str += cnew
                p_sign[context][sign_val] += 1 - weight
                p_sign[context + 1][sign_val] += weight
                if p_sign[context].sum() > 1e6:
                    p_sign[context] = np.maximum(p_sign[context] / 2, 0.01)
                if p_sign[context + 1].sum() > 1e6:
                    p_sign[context + 1] = np.maximum(p_sign[context + 1] / 2, 0.01)

            new_p = norm.pdf(np.arange(symbols), loc=d, scale=2)
            new_p = new_p / new_p.sum() * 100
            p[context] = (1 - alpha) * p[context] + alpha * new_p * (1 - weight)
            p[context + 1] = (1 - alpha) * p[context + 1] + alpha * new_p * weight
            if p[context].sum() > 1e6:
                p[context] = np.maximum(p[context] / 2, 0.01)
            if p[context + 1].sum() > 1e6:
                p[context + 1] = np.maximum(p[context + 1] / 2, 0.01)

    bn = 0
    fn = (Up < 1.0) or (Lw > 0.0)
    while fn:
        bn += 1
        Lw *= 2
        Up *= 2
        fn = Up < (np.ceil(Lw) + 1)

    if bn > 0:
        code_str += format(int(np.ceil(Lw)), f'0{bn}b')

    return code_str

def run_one_setting(image, file_path, prediction_method, context_setting, context_type_feature, args):
    prediction_image = get_prediction_image(image, prediction_method)
    residual = image - prediction_image
    entropy = calculate_entropy(residual)
    output_path = os.path.join(
        args.output,
        os.path.splitext(os.path.basename(file_path))[0],
        f'{prediction_method}_{context_setting}_{context_type_feature}'
    )
    print(f'Setting: {output_path}, entropy = {entropy}')
    context_features = get_context_features(image, prediction_image, context_type_feature)
    if context_setting == 'MIXED':
        result = apply_MIXED_CAAC_method(image, prediction_image, context_features)
    else:
        result = apply_CAAC_method(image, prediction_image, context_features)

    print(f"Setting: {output_path},  Code length = {len(result)}, corresponding bpp: {len(result) / (image.shape[0] * image.shape[1])}")
    print("====================================================================")

    return {
        'setting': output_path,
        'entropy': entropy,
        'bits': result,
        'bpp': len(result) / (image.shape[0] * image.shape[1]),
    }

def write_log(info):
    head, tail = os.path.split(info['setting'])
    os.makedirs(head, exist_ok=True)
    with open(f'{head}/improve_2.log', 'a', encoding='utf-8') as f:
        f.write('==============================================================\n')
        f.write(f'method: {tail}\n')
        for name, value in info.items():
            if name == 'bits':
                continue
            f.write(f'{name} : {value}\n')

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    for dataset in args.datasets:
        dpath = DATASET_DIRS.get(dataset.lower(), dataset)
        files = find_files(dpath, (".bmp", ".png", ".jpg", ".npy"))
        if not files:
            print(f"No files found in {dpath}")
            continue

        print(f"=== Dataset: {dataset} ===")

        record = {
            "EDP": {
                    "BASE": 0,
                    "MIXED": 0,
                },
            "GAP": {
                    "BASE": 0,
                    "MIXED": 0,
                },
            "MED": {
                    "BASE": 0,
                    "MIXED": 0,
                },
            "DIFF": {
                    "BASE": 0,
                    "MIXED": 0,
                }
            }

        for file in files:
            image = load_image(file)
            dc_image = dc(image)
            print(f"Processing file: {file}")
            print("============================================================")
            for method in args.prediction_methods:
                for context_setting in args.context_settings:
                    for context_type_feature in args.context_type_features:
                        info = run_one_setting(
                            dc_image,
                            file,
                            method,
                            context_setting,
                            context_type_feature,
                            args,
                        )
                        record[method][context_setting] += info["bpp"]

        print("Final Results for", dataset)
        for method, bpp in record.items():
            for context_setting, value in bpp.items():
                print(f"Prediction method: {method}, Context: {context_setting} average bpp: {value / len(files)}")

        with open(f'global2.log', 'a', encoding='utf-8') as f:
            f.write(f"Final Results for {dataset}\n")
            for method, bpp in record.items():
                for context_setting, value in bpp.items():
                    f.write(f"Prediction method: {method}, Context: {context_setting} average bpp: {value / len(files):.4f}\n")


