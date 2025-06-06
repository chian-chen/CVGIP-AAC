import numpy as np
from PIL import Image
from scipy.stats import entropy

def calculate_entropy(arr: np.ndarray) -> float:
    arr = arr.astype(np.uint8)
    hist = np.bincount(arr.ravel(), minlength=256)

    probs = hist / np.sum(hist)
    probs = probs[probs > 0]

    return entropy(probs, base=2)

def pad_image(image, padding):
    return np.pad(image, pad_width=padding, mode='constant', constant_values=0)


def create_N_features(image, points):
    """Extract features from the image using the given number of causal neighbors."""
    h, w = image.shape
    padded_image = pad_image(image, 2)  # Padding to handle borders
    features = np.zeros((h, w, points), dtype=np.float32)
    
    # Neighbor indices relative to the center pixel (X(n))
    neighbor_offsets = {
        4: [(-1, 0), (0, -1), (-1, -1), (-1, 1)],
        6: [(-1, 0), (0, -1), (-1, -1), (-1, 1), (-2, 0), (0, -2)],
        8: [(-1, 0), (0, -1), (-1, -1), (-1, 1), (-2, 0), (0, -2), (-2, -1), (-1, -2)],
        10: [(-1, 0), (0, -1), (-1, -1), (-1, 1), (-2, 0), (0, -2), (-2, -1), (-1, -2), (-2, 1), (-1, 2)],
        12: [(-1, 0), (0, -1), (-1, -1), (-1, 1), (-2, 0), (0, -2), (-2, -1), (-1, -2), (-2, 1), (-1, 2), (-2, -2), (-2, 2)]
    }
    
    offsets = neighbor_offsets[points]
    for i in range(h):
        for j in range(w):
            x, y = i + 2, j + 2  # Adjusted for padding
            features[i, j] = [padded_image[x + dx, y + dy] for dx, dy in offsets]
    
    return features

def create_feature_matrix(features, T, N):
    """Convert extracted features into an M x N matrix for each pixel."""
    h, w, _ = features.shape
    feature_matrices = np.zeros((h, w, 2 * T * (T + 1), N), dtype=np.float32)

    padded_features = np.pad(features, pad_width=((T, T), (T, T), (0, 0)), mode='edge')

    # First Part:
    for i in range(h):
        for j in range(w):
            index = 0
            for ii in range(i + T - T, i + T + 1, 1):
                for jj in range(j + T - T, j + T, 1):
                    feature_matrices[i, j, index] = padded_features[ii, jj]
                    index += 1
    # Second Part:
    for i in range(h):
        for j in range(w):
            index = T * (T + 1)
            for ii in range(i + T - T, i + T, 1):
                for jj in range(j + T, j + T + T + 1, 1):
                    feature_matrices[i, j, index] = padded_features[ii, jj]
                    index += 1

    return feature_matrices

def create_y_features(image, T):
    h, w = image.shape
    padded_image = pad_image(image, T)  # Padding to handle borders
    y_features = np.zeros((h, w, 2 * T * (T + 1)))
    # First Part:
    for i in range(h):
        for j in range(w):
            index = 0
            for ii in range(i + T - T, i + T + 1, 1):
                for jj in range(j + T - T, j + T, 1):
                    y_features[i, j, index] = padded_image[ii, jj]
                    index += 1
     # Second Part:
    for i in range(h):
        for j in range(w):
            index = T * (T + 1)
            for ii in range(i + T - T, i + T, 1):
                for jj in range(j + T, j + T + T + 1, 1):
                    y_features[i, j, index] = padded_image[ii, jj]
                    index += 1
    return y_features

    


# Example usage with a sample grayscale image
def EDP_main(image):
    image = np.array(image, dtype=np.int16)
    h, w = image.shape

    features = create_N_features(image, 4)
    y_features = create_y_features(image, 7)
    Cmatrixes = create_feature_matrix(features, 7, 4)
    
    predicted_image = np.zeros(shape=image.shape)

    for i in range(h):
        for j in range(w):
            C = Cmatrixes[i][j]
            CT = C.T
            y = y_features[i][j]
            a = np.linalg.pinv(CT @ C) @ (CT @ y)
            predicted_image[i][j] = a @ features[i][j]
    
    residual = image - predicted_image
    entropy = calculate_entropy(residual)
    print(f'entropy using EDP: {entropy}')

    diff =np.diff(image.flatten())
    print(f'entropy using difference: {calculate_entropy(diff)}')


    return predicted_image

def edp_predict(image: np.ndarray) -> np.ndarray:
    img = image.astype(np.int16)
    h, w = img.shape
    features    = create_N_features(img, 4)
    y_features  = create_y_features(img, 7)
    Cmatrices   = create_feature_matrix(features, 7, 4)

    predicted = np.zeros_like(img, dtype=np.float32)
    for i in range(h):
        for j in range(w):
            C  = Cmatrices[i, j]
            CT = C.T
            yv = y_features[i, j]
            # a  = np.linalg.pinv(CT @ C) @ (CT @ yv)
            λ = 1e-2
            A = CT @ C + λ * np.eye(CT.shape[0])
            a = np.linalg.inv(A) @ (CT @ yv)

            predicted[i, j] = a @ features[i, j]

    return predicted.astype(np.int16)



if __name__ == "__main__":
    image = Image.open('../datas/Lena.bmp').convert('L')
    EDP_main(image)
