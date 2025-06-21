import numpy as np
from PIL import Image
from scipy.stats import entropy



def calculate_entropy(arr: np.ndarray) -> float:
    arr = arr.astype(np.uint16)
    hist = np.bincount(arr.ravel(), minlength=256)

    probs = hist / np.sum(hist)
    probs = probs[probs > 0]

    return entropy(probs, base=2)


        
    
def get_value(image, i, j):
    """Get pixel value with boundary checks."""
    if i < 0 or i >= image.shape[0] or j < 0 or j >= image.shape[1]:
        return 0
    return image[i, j]

def GAP_predict(image, i, j):
    # predicting context
    In = get_value(image, i, j-1)
    Iw = get_value(image, i-1, j)
    Ine = get_value(image, i+1, j-1)
    Inw = get_value(image, i-1, j-1)
    Inn = get_value(image, i, j-2)
    Iww = get_value(image, i-2, j)
    Inne = get_value(image, i+1, j-2)
    # input to GAP
    dh = abs(Iw - Iww) + abs(In - Inw) + abs(In - Ine)
    dv = abs(Iw - Inw) + abs(In - Inn) + abs(Ine - Inne)
    # GAP
    if dv - dh > 80:
        predict = Iw
    elif dv - dh < -80:
        predict = In
    else:
        predict = (Iw + In) / 2 + (Ine - Inw) / 4
        if dv - dh > 32:
            predict = (predict + Iw) / 2
        elif dv - dh > 8:
            predict = (3 * predict + Iw) / 4
        elif dv - dh < -32:
            predict = (predict + In) / 2
        elif dv - dh < -8:
            predict = (3*predict + In) / 4
    return predict


def MED_main(image):
    image = np.array(image, dtype=np.int16)
    h, w = image.shape

    # Create feature matrices
    predicted_image = np.zeros(shape=image.shape).astype(np.int16)
    for i in range(h):
        for j in range(w):
            predicted_image[i, j] = GAP_predict(image, i, j)
            

    residual = image - predicted_image
    entropy = calculate_entropy(residual)
    print(f'entropy using MED: {entropy}')

    diff =np.diff(image.flatten())
    print(f'entropy using difference: {calculate_entropy(diff)}')

    return Image.fromarray(predicted_image.astype(np.uint8))


def gap_predict(image: np.ndarray) -> np.ndarray:
    img = image.astype(np.int16)
    h, w = img.shape

    def get_value(i, j):
        if i<0 or i>=h or j<0 or j>=w:
            return 0
        return img[i, j]

    def GAP_context(i, j):
        In  = get_value(i,   j-1)
        Iw  = get_value(i-1, j)
        Ine = get_value(i+1, j-1)
        Inw = get_value(i-1, j-1)
        Iww = get_value(i-2, j)
        Inn = get_value(i,   j-2)
        Inne= get_value(i+1, j-2)
        dh = abs(Iw-Iww) + abs(In-Inw) + abs(In-Ine)
        dv = abs(Iw-Inw) + abs(In-Inn) + abs(Ine-Inne)
        if   dv-dh >  80: return Iw
        elif dv-dh < -80: return In
        else:
            pred = (Iw+In)/2 + (Ine-Inw)/4
            if   dv-dh >  32: pred = (pred + Iw)/2
            elif dv-dh >   8: pred = (3*pred + Iw)/4
            elif dv-dh <  -32: pred = (pred + In)/2
            elif dv-dh <   -8: pred = (3*pred + In)/4
            return pred

    out = np.zeros_like(img, dtype=np.float32)
    for i in range(h):
        for j in range(w):
            out[i, j] = GAP_context(i, j)

    return out.astype(np.int16)


# 範例使用
if __name__ == "__main__":
    image = Image.open('../datas/Lena.bmp').convert('L')
    MED_main(image)
    
