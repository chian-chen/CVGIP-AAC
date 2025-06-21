import numpy as np
from scipy.stats import norm
from typing import Iterable, Tuple

from CAAC2 import load_image, get_prediction_image, get_context_features, context_map
from utils import dc, enc
from CAAC import apply_CAAC_method



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Search CAAC frequency table parameters")
    parser.add_argument('--file', help='Path to the image file')
    parser.add_argument('--method', default='EDP', help='Prediction method')
    args = parser.parse_args()

    image = load_image(args.file)
    # dc_image = dc(image)
    dc_image = image
    prediction_image = get_prediction_image(dc_image, args.method)
    context_features = get_context_features(dc_image, prediction_image, None)
    code_str, positive_count, negative_count, value_str, sign_str = apply_CAAC_method(dc_image, prediction_image, args.method, context_features)
    print(f"Code string: {len(code_str)}")
    print(f"bpp: {len(code_str) / (dc_image.shape[0] * dc_image.shape[1]):.6f}")