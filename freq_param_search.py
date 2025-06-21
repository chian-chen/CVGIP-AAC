import numpy as np
from scipy.stats import norm
from typing import Iterable, Tuple

from CAAC2 import load_image, get_prediction_image, get_context_features, context_map
from utils import dc, enc


def apply_caac_with_params(image: np.ndarray,
                           prediction_image: np.ndarray,
                           context_features: np.ndarray,
                           coef: float,
                           add_init: float,
                           alpha: float) -> int:
    """Apply CAAC2 style coding with configurable frequency table parameters.

    Returns the length of the generated code string in bits."""
    h, w = image.shape
    symbols = 256
    encoded_image = image - prediction_image

    base_tb = np.exp(-abs(coef * (np.arange(symbols) - 1)))
    base_tb = base_tb / base_tb.sum() * 100.0
    base_tb = np.maximum(base_tb, 1.0)
    p = np.tile(base_tb, (4, 1))

    add = np.ones((4, 1), dtype=np.float64) * add_init

    Lw, Up = 0.0, 1.0
    code_str = ""
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
                sign_val = 1 if encoded_image[i, j] > 0 else 0
                Lw, Up, cnew = enc(sign_val, Lw, Up, np.array([0.5, 0.5]))
                code_str += cnew
            p[context] += norm.pdf(np.arange(0, symbols), loc=d, scale=1) * add[context]
            add[context] *= alpha
            if p[context].sum() > 1e5:
                p[context] = np.maximum(p[context] / 2, 0.01)
                add[context] /= 2

    fn = (Up < 1.0) or (Lw > 0.0)
    bn = 0
    while fn:
        bn += 1
        Lw *= 2
        Up *= 2
        fn = Up < (np.ceil(Lw) + 1)
    if bn > 0:
        code_str += format(int(np.ceil(Lw)), f'0{bn}b')

    return len(code_str)


def search_frequency_parameters(file_path: str,
                                method: str = "EDP",
                                coef_list: Iterable[float] = (0.1, 0.2, 0.3),
                                add_list: Iterable[float] = (40, 80, 120),
                                alpha_list: Iterable[float] = (0.8, 1.0, 1.2)
                                ) -> Tuple[Tuple[float, float, float], float]:
    """Brute-force search for best frequency table parameters."""
    image = load_image(file_path)
    dc_image = dc(image)
    prediction_image = get_prediction_image(dc_image, method)
    context_features = get_context_features(dc_image, prediction_image, None)

    h, w = dc_image.shape
    best_params = None
    best_bpp = float('inf')

    for coef in coef_list:
        for add_init in add_list:
            for alpha in alpha_list:
                length = apply_caac_with_params(dc_image,
                                                prediction_image,
                                                context_features,
                                                coef,
                                                add_init,
                                                alpha)
                bpp = length / (h * w)
                print(f'coef={coef}, add={add_init}, alpha={alpha}, bpp={bpp:.6f}')
                if bpp < best_bpp:
                    best_bpp = bpp
                    best_params = (coef, add_init, alpha)

    print(f'Best params: {best_params}, bpp={best_bpp:.6f}')
    return best_params, best_bpp


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Search CAAC frequency table parameters")
    parser.add_argument('file', help='Path to the image file')
    parser.add_argument('--method', default='EDP', help='Prediction method')
    args = parser.parse_args()

    search_frequency_parameters(args.file, method=args.method)
