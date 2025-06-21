import numpy as np
from PIL import Image
from scipy.stats import entropy



def calculate_entropy(arr: np.ndarray) -> float:
    arr = arr.astype(np.uint16)
    hist = np.bincount(arr.ravel(), minlength=256)

    probs = hist / np.sum(hist)
    probs = probs[probs > 0]

    return entropy(probs, base=2)

def loco_predict(a: int, b: int, c: int) -> int:
    if c >= max(a, b):
        return min(a, b)
    elif c <= min(a, b):
        return max(a, b)
    else:
        return a + b - c
    
def get_value(image, i, j):
    """Get pixel value with boundary checks."""
    if i < 0 or i >= image.shape[0] or j < 0 or j >= image.shape[1]:
        return 0
    return image[i, j]

def MED_main(image):
    image = np.array(image, dtype=np.int16)
    h, w = image.shape

    # Create feature matrices
    predicted_image = np.zeros(shape=image.shape).astype(np.int16)
    for i in range(h):
        for j in range(w):
            a = get_value(image, i - 1, j)
            b = get_value(image, i, j - 1)
            c = get_value(image, i - 1, j - 1)
            predicted_image[i, j] = loco_predict(a, b, c)
            

    residual = image - predicted_image
    entropy = calculate_entropy(residual)
    print(f'entropy using MED: {entropy}')

    diff =np.diff(image.flatten())
    print(f'entropy using difference: {calculate_entropy(diff)}')

    return Image.fromarray(predicted_image.astype(np.uint8))

def med_predict(image: np.ndarray) -> np.ndarray:
    img = image.astype(np.int16)
    h, w = img.shape

    def get_value(i, j):
        if i<0 or i>=h or j<0 or j>=w:
            return 0
        return img[i, j]

    def loco(a, b, c):
        if   c >= max(a, b): return min(a, b)
        elif c <= min(a, b): return max(a, b)
        else:                return a + b - c

    out = np.zeros_like(img, dtype=np.float32)
    for i in range(h):
        for j in range(w):
            a = get_value(i-1, j)
            b = get_value(i, j-1)
            c = get_value(i-1, j-1)
            out[i, j] = loco(a, b, c)

    return out.astype(np.int16)



# 範例使用
if __name__ == "__main__":
    image = Image.open('../datas/Lena.bmp').convert('L')
    MED_main(image)
    
