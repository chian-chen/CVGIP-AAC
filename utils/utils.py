import numpy as np
from decimal import Decimal, getcontext
from scipy.stats import entropy
import os
from math import sqrt
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

getcontext().prec = 300


def dc(image):
    A = np.array(image, dtype=float)
    (M, N) = A.shape
    M1 = M // 8
    N1 = N // 8

    C = np.zeros((8,8), dtype=float)
    for p in range(8):
        for q in range(8):
            alpha = sqrt(1.0/8.0) if p == 0 else sqrt(2.0/8.0)
            C[p,q] = alpha * np.cos((2*q+1)*p*np.pi/16.0)

    dc = []
    for a1 in range(M1):
        x1 = a1*8
        for a2 in range(N1):
            y1 = a2*8
            A_block = A[x1:x1+8, y1:y1+8]
            cf = C @ A_block @ C.T
            dc_val = round(cf[0,0]/16.0)
            dc.append(dc_val)
    dc = np.array(dc, dtype=int).reshape(M1, N1)
    return dc

def enc(value, Lw1, Up1, pb):

    pcum = [Decimal('0')]
    s = sum(pb)  # float
    s_dec = Decimal(str(s))
    run_sum = Decimal('0')
    for x in pb:
        run_sum += Decimal(str(x))
        pcum.append(run_sum)

    total = pcum[-1]
    for i in range(len(pcum)):
        pcum[i] = pcum[i] / total
    
    p1 = pcum[value]
    p2 = pcum[value + 1]

    Lw1 = Decimal(str(Lw1))
    Up1 = Decimal(str(Up1))
    lu = Up1 - Lw1
    Lw = Lw1 + lu * p1
    Up = Lw1 + lu * p2

    cnew_bits = []

    half = Decimal('0.5')
    one  = Decimal('1')
    two  = Decimal('2')

    while (Lw >= half) or (Up <= half):
        if Lw >= half:
            Lw = (Lw * two) - one
            Up = (Up * two) - one
            cnew_bits.append('1')
        else:
            Lw = Lw * two
            Up = Up * two
            cnew_bits.append('0')

    return Lw, Up, ''.join(cnew_bits)

def dcd(Code, Lw1, Up1, Lw2, Up2, pb):

    pcum = [Decimal('0')]
    run_sum = Decimal('0')
    for x in pb:
        run_sum += Decimal(str(x))
        pcum.append(run_sum)

    total = pcum[-1]
    for i in range(len(pcum)):
        pcum[i] = pcum[i] / total
        
    Lw1 = Decimal(str(Lw1))
    Up1 = Decimal(str(Up1))
    Lw2 = Decimal(str(Lw2))
    Up2 = Decimal(str(Up2))

    lu = Up1 - Lw1
    p1 = [Lw1 + lu * pcum[i] for i in range(len(pcum))]

    Ln = len(pb)
    bt = 0
    fd = False
    fst = 0

    vn = None
    half = Decimal('0.5')
    one  = Decimal('1')
    two  = Decimal('2')

    while not fd:
        search_array = p1[fst+1 : Ln+1]
        idxs = [idx for idx, val in enumerate(search_array) if val > Lw2]
        if len(idxs) > 0:
            vn = fst + idxs[0]
        else:
            vn = Ln - 1

        fst = vn - 1

        if (vn < Ln) and (p1[vn+1] >= Up2):
            fd = True
        else:
            if Code[bt] == '1':
                Lw2 = Lw2 + (Up2 - Lw2)/two
            else:
                Up2 = Lw2 + (Up2 - Lw2)/two
            bt += 1

    Lw = p1[vn]
    Up = p1[vn+1]
    Code_rem = Code[bt:]

    while (Lw >= half) or (Up <= half):
        if Lw >= half:
            Lw  = Lw*two - one
            Up  = Up*two - one
            Lw2 = Lw2*two - one
            Up2 = Up2*two - one
        else:
            Lw  = Lw*two
            Up  = Up*two
            Lw2 = Lw2*two
            Up2 = Up2*two

    return vn, Code_rem, Lw, Up, Lw2, Up2

def calculate_entropy(arr: np.ndarray) -> float:
    arr = arr.astype(np.uint8)
    hist = np.bincount(arr.ravel(), minlength=256)

    probs = hist / np.sum(hist)
    probs = probs[probs > 0]

    return entropy(probs, base=2)

def find_files(folder_path: str, extensions=(".bmp",)):
    """Recursively collect files with given extensions."""
    file_list = []
    exts = tuple(e.lower() for e in extensions)
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(exts):
                file_list.append(os.path.join(root, file))
    return sorted(file_list)

# Backward compatibility
def find_bmp_files(folder_path: str):
    return find_files(folder_path, (".bmp",))

def visualize_prob_tables(p: np.ndarray,
                          p_gt: np.ndarray,
                          path: str) -> None:
    p    = p.astype(float)
    p_gt = p_gt.astype(float)
    if p.sum()>0:
        p    /= p.sum()
    if p_gt.sum()>0:
        p_gt /= p_gt.sum()

    D = len(p)
    x_full = np.arange(D)
    x20    = np.arange(20)

    # 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    fig.suptitle(path, fontsize=12)

    # Top-left: Estimated full-range CDF
    axes[0,0].plot(x_full, p, marker='o', markersize=4, linestyle='-')
    axes[0,0].set_title("Estimated CDF (full)")
    axes[0,0].set_xlim(0, D-1)
    axes[0,0].set_ylim(0, 1)
    axes[0,0].set_ylabel("CDF")
    axes[0,0].grid(True)

    # Top-right: True full-range CDF
    axes[0,1].plot(x_full, p_gt, marker='o', markersize=4, linestyle='-')
    axes[0,1].set_title("True CDF (full)")
    axes[0,1].set_xlim(0, D-1)
    axes[0,1].set_ylim(0, 1)
    axes[0,1].set_ylabel("CDF")
    axes[0,1].grid(True)

    # Bottom-left: Estimated CDF, first 20 points
    # axes[1,0].plot(x20, p[:len(x20)], marker='o', markersize=4, linestyle='-')
    axes[1,0].bar(x20, p[:len(x20)], width=0.75, align='center')
    axes[1,0].set_title("Estimated CDF (0–19)")
    axes[1,0].set_xlim(0, len(x20)-1)
    axes[1,0].set_ylim(0, max(p[:len(x20)])*1.05)
    axes[1,0].set_ylabel("CDF")
    axes[1,0].grid(True)
    axes[1,0].xaxis.set_major_locator(MultipleLocator(1))
    axes[1,0].yaxis.set_major_locator(MultipleLocator(0.2))

    # Bottom-right: True CDF, first 20 points
    axes[1,1].bar(x20, p_gt[:len(x20)], width=0.75, align='center')
    # axes[1,1].plot(x20, p_gt[:len(x20)], marker='o', markersize=4, linestyle='-')
    axes[1,1].set_title("True CDF (0–19)")
    axes[1,1].set_xlim(0, len(x20)-1)
    axes[1,1].set_ylim(0, max(p_gt[:len(x20)])*1.05)
    axes[1,1].set_ylabel("CDF")
    axes[1,1].grid(True)
    axes[1,1].xaxis.set_major_locator(MultipleLocator(1))
    axes[1,1].yaxis.set_major_locator(MultipleLocator(0.2))

    fig.savefig(path, dpi=300)
    plt.close(fig)

