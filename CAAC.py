import numpy as np
import os
from PIL import Image
import argparse
import cv2

from prediction_methods import edp_predict, gap_predict, med_predict
from utils import dc, enc, dcd, calculate_entropy, find_files, visualize_prob_tables

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

def apply_CAAC_method(image, prediction_image, context_settings, context_features):
    h, w = image.shape
    symbols = 256
    encoded_image = image - prediction_image
    positive_count, negative_count = 0, 0
    value_str, sign_str = 0, 0
    
    Lw, Up = 0.0, 1.0
    code_str = ""
    p = np.ones((4, symbols), dtype=float)
    # p_sign = np.ones((4, 2), dtype=float)
    # p_sign = np.array([1, 1])
    context = 0 # initial context = 0
    Lw, Up, cnew = enc(encoded_image[0][0], Lw, Up, p[context])
    code_str += cnew


    for i in range(0, h):
        for j in range(0, w):
            if i == 0 and j == 0:
                continue
            context = context_map(context_features[i, j])
            if encoded_image[i][j] > 0:
                positive_count += 1
            else:
                negative_count += 1

            d = abs(encoded_image[i][j])
            if d > symbols - 1:
                print(f"Warning: value {d} at ({i}, {j}) exceeds {symbols - 1}, clipping to {symbols}")
                d = symbols - 1
            Lw, Up, cnew = enc(d, Lw, Up, p[context])
            code_str += cnew
            value_str += len(cnew)

            if d != 0:
                sign_val = 1 if (encoded_image[i][j]) > 0 else 0
                Lw, Up, cnew = enc(sign_val, Lw, Up, np.array([0.5, 0.5]))
                # Lw, Up, cnew = enc(sign_val, Lw, Up, p_sign)
                # p_sign[sign_val] += 1
                # Lw, Up, cnew = enc(sign_val, Lw, Up, p_sign[context])
                # p_sign[context][sign_val] += 1 
                code_str += cnew
                sign_str += len(cnew)
            p[context][d] += 5

    fn = (Up < 1.0) or (Lw > 0.0)
    bn = 0
    while fn:
        bn += 1
        Lw *= 2
        Up *= 2
        fn = Up < (np.ceil(Lw) + 1)
    
    if bn > 0:
        code_str += format(int(np.ceil(Lw)), '0{}b'.format(bn))

    return code_str, positive_count, negative_count, value_str, sign_str

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

def apply_MIXED_CAAC_method(image, prediction_image, context_settings, context_features):
    h, w = image.shape
    symbols = 256
    encoded_image = image - prediction_image
    Lw, Up = 0.0, 1.0
    code_str = ""
    p = np.ones((4, symbols), dtype=float)
    context = 0
    Lw, Up, cnew = enc(encoded_image[0][0], Lw, Up, p[context])
    code_str += cnew

    for i in range(0, h):
        for j in range(0, w):
            if i == 0 and j == 0:
                continue
            context, weight = mixed_context_map(context_features[i, j])
            current_p = p[context] * (1 - weight) + p[context + 1] * weight
            d = abs(encoded_image[i][j])
            if d > symbols - 1:
                print(f"Warning: value {d} at ({i}, {j}) exceeds {symbols - 1}, clipping to {symbols}")
                d = symbols - 1
            Lw, Up, cnew = enc(d, Lw, Up, current_p)
            code_str += cnew

            if d != 0:
                sign_val = 1 if (encoded_image[i][j]) > 0 else 0
                Lw, Up, cnew = enc(sign_val, Lw, Up, np.array([0.5, 0.5]))
                code_str += cnew
            p[context][d] += 5 * (1 - weight)
            p[context + 1][d] += 5 * weight

    fn = (Up < 1.0) or (Lw > 0.0)
    bn = 0
    while fn:
        bn += 1
        Lw *= 2
        Up *= 2
        fn = Up < (np.ceil(Lw) + 1)
    
    if bn > 0:
        code_str += format(int(np.ceil(Lw)), '0{}b'.format(bn))

    return code_str

def get_true_prob_table(encoded_image, context_features):
    h, w = encoded_image.shape
    p_gt = np.zeros((4, 129), dtype=float)

    for i in range(h):
        for j in range(w):
            context = context_map(context_features[i, j])
            d = abs(encoded_image[i][j])
            if d > 128:
                print(f"Warning: value {d} at ({i}, {j}) exceeds 128, clipping to 128, original value: {d}")
                d = 128
            p_gt[context][d] += 1

    for context in range(4):
        total = np.sum(p_gt[context])
        if total > 0:
            p_gt[context] /= total

    return p_gt

def apply_CAAC_method_with_visualization(image, prediction_image, context_settings, context_features, output_path):
    h, w = image.shape
    encoded_image = image - prediction_image
    p_gt = get_true_prob_table(encoded_image, context_features)

    Lw, Up = 0.0, 1.0
    code_str = ""
    p = np.ones((4, 129), dtype=float)
    context = 0
    Lw, Up, cnew = enc(encoded_image[0][0], Lw, Up, p[context])
    code_str += cnew

    for i in range(0, h):
        for j in range(0, w):
            if i == 0 and j == 0:
                continue
            context = context_map(context_features[i, j])
            d = abs(encoded_image[i][j])
            if d > 128:
                print(f"Warning: value {d} at ({i}, {j}) exceeds 128, clipping to 128, original value: {d}")
                d = 128
            Lw, Up, cnew = enc(d, Lw, Up, p[context])
            code_str += cnew
            visualize_prob_tables(p[context], p_gt[context], os.path.join(output_path, f'c_{context}_i_{i}_j_{j}_d_{d}.png'))

            if d != 0:
                sign_val = 1 if (encoded_image[i][j]) > 0 else 0
                Lw, Up, cnew = enc(sign_val, Lw, Up, np.array([0.5, 0.5]))
                code_str += cnew
            p[context][d] += 1

    fn = (Up < 1.0) or (Lw > 0.0)
    bn = 0
    while fn:
        bn += 1
        Lw *= 2
        Up *= 2
        fn = Up < (np.ceil(Lw) + 1)
    
    if bn > 0:
        code_str += format(int(np.ceil(Lw)), '0{}b'.format(bn))

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
        result = apply_MIXED_CAAC_method(image, prediction_image, context_setting, context_features)
    else:
        if args.visualization:    
            result = apply_CAAC_method_with_visualization(image, prediction_image, context_setting, context_features, output_path)
        else:
            result, p_count, n_count, value_str, sign_str = apply_CAAC_method(image, prediction_image, context_setting, context_features)

    print(f"Setting: {output_path},  Code length = {len(result)}, corresponding bpp: {len(result) / (image.shape[0] * image.shape[1])}")
    # print(f"In this setting, we found the positive and negative count is : {p_count} and {n_count}, the ratio is: {p_count / (p_count + n_count) * 100}% and {n_count / (p_count + n_count) * 100}%")
    # print(f"In this setting, we found the value / sign coding length is : {value_str} and {sign_str}, the ratio is: {value_str / (value_str + sign_str) * 100}% and {sign_str / (value_str + sign_str) * 100}%")
    print("=====================================================================================")

    # sign_image = np.sign(residual)
    # sign_mapped = np.zeros_like(sign_image, dtype=np.uint8)
    # sign_mapped[ sign_image >  0] =   0   # black
    # sign_mapped[ sign_image == 0] = 127   # gray
    # sign_mapped[ sign_image <  0] = 255   # white
    # head, tail = os.path.split(output_path)
    # sign_img_pil = Image.fromarray(sign_mapped)
    # sign_img_pil.save(f'{head}/{prediction_method}_sign_image.png')


    return {
        'setting': output_path,
        'entropy': entropy,
        'bits': result,
        'bpp': len(result) / (image.shape[0] * image.shape[1]),
        # 'positive_ratio': p_count / (p_count + n_count) * 100,
        # 'sign_bits': sign_str
    }

def write_log(info):
    head, tail = os.path.split(info['setting'])
    os.makedirs(head, exist_ok=True)
    with open(f'{head}/improve_2.log', 'a', encoding='utf-8') as f:
        f.write('=====================================================================================\n')
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
        # record_sign_ratio = {"EDP": 0, "GAP": 0, "MED": 0, "DIFF": 0}
        # record_sign_bits = {"EDP": 0, "GAP": 0, "MED": 0, "DIFF": 0}
        # record_bitplane = {"EDP": 0, "GAP": 0, "MED": 0, "DIFF": 0}

        for file in files:
            image = load_image(file)
            # img16 = cv2.imread(file, cv2.IMREAD_UNCHANGED)
            # image = np.array(img16).astype(np.int16)  # convert to int16 for further processing
            dc_image = dc(image)
            # dc_image = image
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
                        # record_sign_ratio[method] += info["positive_ratio"]
                        # record_sign_bits[method] += info["sign_bits"]
                        # write_log(info)


        print("Final Results for", dataset)
        for method, bpp in record.items():
            for context_setting, value in bpp.items():
                print(f"{context_setting} average bpp: {value / len(files)}")
        
        with open(f'global2.log', 'a', encoding='utf-8') as f:
            f.write(f"Final Results for {dataset}\n")
            for method, bpp in record.items():
                for context_setting, value in bpp.items():
                    f.write(f"Prediction method: {method}, Context: {context_setting} average bpp: {value / len(files):.4f}\n")
        # for method, positive_ratio in record_sign_ratio.items():
        #     print(f"{method} average positive_ratio: {positive_ratio / len(files)}")
        # for method, sign_bits in record_sign_bits.items():
        #     print(f"{method} average sign_bits: {sign_bits / len(files)}")
    
