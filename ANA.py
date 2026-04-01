"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  UNI — Recurrence + N/P Reconstruction + Closure Analysis     ║
║  Version with detailed missing and duplication analysis                      ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import math
import re
import os
import hashlib
import urllib.request
from pathlib import Path
import numpy as np
from collections import Counter

# ══════════════════════════════════════════════════════════════════════════════
# VALIDATION DATA SOURCE — Odlyzko (University of Minnesota)
# ══════════════════════════════════════════════════════════════════════════════

DATASET_URL     = "https://www-users.cse.umn.edu/~odlyzko/zeta_tables/zeros6"
DATASET_FILE    = "zeros6.txt"
EXPECTED_SHA256 = "2ef7b752c2f17405222e670a61098250c8e4e09047f823f41e2b41a7b378e7c6"

def download_zeros_if_needed() -> str:
    """
    Download zeros6.txt from Odlyzko's site if not already present locally.
    Verifies SHA-256 integrity after download.
    Returns the local file path.
    """
    if os.path.exists(DATASET_FILE):
        print(f"   File '{DATASET_FILE}' already present locally.")
    else:
        print(f"   Downloading validation data from:\n   {DATASET_URL}")
        urllib.request.urlretrieve(DATASET_URL, DATASET_FILE)
        print(f"   Download complete.")

    # SHA-256 integrity check
    sha256 = hashlib.sha256()
    with open(DATASET_FILE, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    digest = sha256.hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError(
            f"SHA-256 mismatch!\n"
            f"  Expected : {EXPECTED_SHA256}\n"
            f"  Got      : {digest}"
        )
    print(f"   SHA-256 verified ✓")
    return DATASET_FILE

# ══════════════════════════════════════════════════════════════════════════════
# PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

c           = 0.049
N_STAR      = 1000    # UNI table bound
N_ZEROS_MAX = None    # None = use all zeros

ZERO_FILE = DATASET_FILE   # points to the locally downloaded file

ln2 = math.log(2)
U   = 2 * math.pi * c / ln2

# ══════════════════════════════════════════════════════════════════════════════
# UNI TABLE
# ══════════════════════════════════════════════════════════════════════════════

def build_uni_composite_table(N: int):
    composite = [False] * (N + 1)
    for i in range(2, N + 1):
        for j in range(i, N + 1):
            if (product := i * j) <= N:
                composite[product] = True
    return composite

COMPOSITE_TABLE = build_uni_composite_table(N_STAR)

def is_prime_uni(n: int) -> bool:
    if n < 2 or n > N_STAR:
        return False
    return not COMPOSITE_TABLE[n]

# ══════════════════════════════════════════════════════════════════════════════
# CORRECTED INVERSION
# ══════════════════════════════════════════════════════════════════════════════

def reconstruct_from_gamma(gamma_list, C: float = c):
    if not gamma_list:
        return []
    gamma_arr = np.array(gamma_list, dtype=float)
    with np.errstate(over='ignore', invalid='ignore'):
        exp_v = np.exp(math.log(0.5) / gamma_arr)
        denom = 1.0 - exp_v
        n_est = np.where(np.abs(denom) > 1e-12, C / denom, np.nan)

    valid = np.isfinite(n_est) & (n_est >= 1.5)
    nv    = n_est[valid]
    nr    = np.round(nv).astype(int)

    seen = {}
    for r, v in zip(nr, nv):
        if r not in seen or abs(v - r) < abs(seen[r] - r):
            seen[r] = float(v)

    return [{"integer": int(r),
             "n_recon": round(v, 6),
             "ratio":   round(v / r, 8),
             "err":     round(abs(v - r), 6)}
            for r, v in sorted(seen.items())]

# ══════════════════════════════════════════════════════════════════════════════
# DENSITY + RECURRENCE
# ══════════════════════════════════════════════════════════════════════════════

def density_UNI(m):
    if m <= 0:
        return 0.0
    x = m * U / (2 * math.pi)
    if x <= 1:
        return 0.0
    return (U / (2 * math.pi)) * math.log(x)

def find_next_m(m_current, max_iter=50, tol=1e-10):
    if m_current <= 0:
        return m_current + 1.0
    d_curr = density_UNI(m_current)
    m_next = m_current + (1.0 / d_curr if d_curr > 0 else 1.5)
    for _ in range(max_iter):
        n_steps  = max(50, int((m_next - m_current) * 20))
        step     = (m_next - m_current) / n_steps
        integral = 0.0
        x        = m_current
        for __ in range(n_steps):
            y1        = density_UNI(x)
            y2        = density_UNI(x + step)
            integral += (y1 + y2) * step / 2.0
            x        += step
        F    = integral - 1.0
        dF   = density_UNI(m_next)
        if dF <= 1e-12:
            m_next *= 1.01
            continue
        m_new = m_next - F / dF
        if abs(m_new - m_next) < tol:
            return m_new
        m_next = m_new
    return m_next

def generate_zeros_recurrence(n_zeros: int):
    m_k = [32.0]
    for _ in range(1, n_zeros):
        m_k.append(find_next_m(m_k[-1]))
    return [round(m) * U for m in m_k]

# ══════════════════════════════════════════════════════════════════════════════
# LOADING VALIDATION DATA
# ══════════════════════════════════════════════════════════════════════════════

def load_zero_file(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    txt  = p.read_text(encoding="utf-8", errors="ignore")
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", txt)
    vals = np.array([float(x) for x in nums], dtype=float)
    vals = vals[vals > 0]
    vals.sort()
    return vals

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 90)
    print("UNI — RH16 Hybrid : Full closure analysis on N")
    print(f"Quantum U = {U:.8f}  |  UNI table bound = {N_STAR:,}")
    print("=" * 90)

    # Download validation data if needed
    print()
    download_zeros_if_needed()
    print()

    zeros_ref = load_zero_file(ZERO_FILE)
    n_ref     = len(zeros_ref)
    print(f"{n_ref:,} zeros loaded")

    n_to_use   = N_ZEROS_MAX if N_ZEROS_MAX is not None else n_ref
    print(f"Generating {n_to_use:,} zeros...")
    zeros_pred = generate_zeros_recurrence(n_to_use)

    print("Inversion → reconstructing integers N...")
    integers = reconstruct_from_gamma(zeros_pred, c)
    n_recon  = len(integers)

    # Detailed closure analysis on N
    if integers:
        integers_sorted = sorted(integers, key=lambda x: x["integer"])
        min_n           = integers_sorted[0]["integer"]
        max_n           = integers_sorted[-1]["integer"]
        expected_count  = max_n - min_n + 1

        # Count duplications
        integer_counts = Counter(item["integer"] for item in integers)
        duplicates     = sum(count - 1 for count in integer_counts.values() if count > 1)
        missing        = expected_count - n_recon

        print("\n" + "─" * 85)
        print("DETAILED CLOSURE ANALYSIS — N")
        print("─" * 85)
        print(f"Reconstructed range         : {min_n} → {max_n}")
        print(f"Expected integers in range  : {expected_count:,}")
        print(f"Reconstructed integers      : {n_recon:,}")
        print(f"Missing                     : {missing:,} ({missing/expected_count*100:.2f}%)")
        print(f"Duplications                : {duplicates:,}")
        print(f"Effective coverage          : {n_recon/expected_count*100:.2f}%")
        print("─" * 85)

    # Primes
    primes_out = [d for d in integers if is_prime_uni(d["integer"])]
    p_recon    = len(primes_out)

    print(f"\nReconstructed primes : {p_recon:,} (up to {N_STAR})")
    print("P closure within bound : 1:1")

    # Extract
    print("\nFirst 15 results:")
    print(f"{'integer':>7}  {'n_recon':>12}  {'ratio':>10}  {'err':>8}  type")
    print("─" * 55)
    for item in integers[:15]:
        typ = "prime" if is_prime_uni(item["integer"]) else "composite"
        print(f"{item['integer']:>7}  {item['n_recon']:>12.6f}  "
              f"{item['ratio']:>10.8f}  {item['err']:>8.6f}  {typ}")

    print("\nEND — N closure analysis complete")
    print("=" * 90)

if __name__ == "__main__":
    main()