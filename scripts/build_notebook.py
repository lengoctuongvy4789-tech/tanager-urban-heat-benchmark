#!/usr/bin/env python3
"""Build the reproducible four-method HSI/MSI benchmark notebook."""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebook" / "01_tanager_hsi_vs_landsat_uhi.ipynb"
nb = nbf.v4.new_notebook()
cells = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text.strip()))


md(r"""
# Ba phương pháp HSI–MSI cho vật liệu đô thị và LST

Notebook có hai thí nghiệm nối tiếp nhau và dùng cùng ba cách biểu diễn phổ:

| Mã | Cấu hình | Ý nghĩa |
|---|---|---|
| **M1** | Multispectral Baseline | band rộng và chỉ số phổ |
| **M2** | Full-Spectrum Hyperspectral | toàn bộ band HSI hợp lệ |
| **M3** | Hyperspectral Unmixing Features | MSI + MNF components + FCLS abundance fractions |

**Phần A — Pavia University:** dùng nhãn vật liệu/lớp phủ để kiểm tra ba cấu hình bằng bài toán phân loại.

**Phần B — Tanager + Landsat:** dùng cùng ba cấu hình để dự đoán Landsat LST và phát hiện hotspot. LST là target chung cho cả M1–M3 trong phần này.

Phần nhiệt sử dụng một module đánh giá độc lập: tính LST anomaly so với bề mặt thực vật tham chiếu và xác định hotspot theo phân vị nhiệt độ.
""")

md(r"""
## 0. Cấu hình

`QUICK_RUN=True` mặc định dùng ba spatial folds và mẫu nhỏ hơn để kiểm tra notebook trên máy cá nhân. Đổi thành `False` trước lần chạy lấy số liệu cuối; notebook sẽ dùng năm folds, 120.000 pixel tối đa và 10.000 PPI projections cho Tanager.
""")

code(r"""
from pathlib import Path
import json, warnings, time

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio
from rasterio.warp import reproject, Resampling
from scipy.io import loadmat
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)
np.random.seed(42)

ROOT = Path.cwd()
if not (ROOT / "data").exists():
    ROOT = ROOT.parent

TANAGER_DIR = ROOT / "data" / "tanager"
LANDSAT_DIR = ROOT / "data" / "landsat"
RSR_FILE = ROOT / "data" / "rsr" / "L8_OLI_RSR.xlsx"
PAVIA_DIR = ROOT / "data" / "benchmark" / "pavia_university"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

H5_FILE = TANAGER_DIR / "20250407_035451_00_4001_ortho_sr_hdf5.h5"
RGB_FILE = TANAGER_DIR / "20250407_035451_00_4001_ortho_visual.tif"
UDM_FILE = TANAGER_DIR / "20250407_035451_00_4001_ortho_beta_udm.tif"
STAC_FILE = TANAGER_DIR / "20250407_035451_00_4001.json"
PAVIA_FILE = PAVIA_DIR / "PaviaU.mat"
PAVIA_GT_FILE = PAVIA_DIR / "PaviaU_gt.mat"

QUICK_RUN = True              # False cho kết quả cuối của report
N_MNF = 16
N_ENDMEMBERS = 7
CV_FOLDS = 3 if QUICK_RUN else 5
MODEL_ITERATIONS = 140 if QUICK_RUN else 250
TRAIN_SAMPLES = 30_000 if QUICK_RUN else 60_000
NOISE_SAMPLES = 20_000 if QUICK_RUN else 40_000
PPI_SAMPLES = 10_000 if QUICK_RUN else 20_000
PPI_PROJECTIONS = 500 if QUICK_RUN else 10_000
PAVIA_FIT_SAMPLES = 15_000 if QUICK_RUN else 30_000
PAVIA_PPI_SAMPLES = 10_000 if QUICK_RUN else 20_000
PAVIA_PPI_PROJECTIONS = 500 if QUICK_RUN else 2_000
PURE_FRACTION = 0.005         # 0,5% pixel có PPI cao nhất
FCLS_ITERATIONS = 40 if QUICK_RUN else 80
SPATIAL_BLOCK_M = 1_000
MAX_BENCHMARK_PIXELS = 30_000 if QUICK_RUN else 120_000

for p in [PAVIA_FILE, PAVIA_GT_FILE, H5_FILE, RGB_FILE, UDM_FILE, STAC_FILE, RSR_FILE]:
    assert p.exists(), f"Thiếu file: {p}"
print("Project root:", ROOT)
print("Mode:", "QUICK_RUN" if QUICK_RUN else "FINAL")
""")

md(r"""
## Phần A — Benchmark có nhãn: Pavia University

Pavia University có 103 band VNIR và chín lớp đô thị. Benchmark này trả lời câu hỏi **HSI có phân biệt vật liệu/lớp phủ tốt hơn MSI không**. Nó không có thermal/LST, vì vậy LST chỉ xuất hiện ở Phần B.

M1 được tạo trực tiếp từ Pavia HSI thành bốn band rộng blue–green–red–NIR cộng NDVI. M2 dùng 103 band. M3 dùng MNF/PPI/FCLS không giám sát rồi đưa các fractions cùng MNF và MSI vào cùng classifier.
""")

code(r"""
PAVIA_LABELS = {
    1: "Asphalt", 2: "Meadows", 3: "Gravel", 4: "Trees",
    5: "Painted metal sheets", 6: "Bare soil", 7: "Bitumen",
    8: "Self-blocking bricks", 9: "Shadows",
}

pavia_cube = loadmat(PAVIA_FILE)["paviaU"].astype(np.float32)
pavia_gt = loadmat(PAVIA_GT_FILE)["paviaU_gt"].astype(np.uint8)
pavia_wl = np.linspace(430.0, 860.0, pavia_cube.shape[2], dtype=np.float32)

# Pavia thường được phân phối dưới dạng DN. Một scale chung giữ nguyên hình dạng phổ
# và giúp bài toán unmixing ổn định số học; tree models không phụ thuộc scale này.
pavia_scale = np.percentile(pavia_cube[pavia_gt > 0], 99.9)
pavia_cube /= max(float(pavia_scale), 1.0)
pr_all, pc_all = np.where(pavia_gt > 0)
pavia_y = pavia_gt[pr_all, pc_all]
pavia_X_full = pavia_cube[pr_all, pc_all, :]

def mean_window(cube, wavelengths_nm, low, high):
    idx = (wavelengths_nm >= low) & (wavelengths_nm <= high)
    if not idx.any():
        raise ValueError(f"Không có band trong cửa sổ {low}-{high} nm")
    return cube[..., idx].mean(axis=-1)

p_blue = mean_window(pavia_cube, pavia_wl, 450, 520)
p_green = mean_window(pavia_cube, pavia_wl, 520, 600)
p_red = mean_window(pavia_cube, pavia_wl, 630, 690)
p_nir = mean_window(pavia_cube, pavia_wl, 760, 860)
p_ndvi = (p_nir - p_red) / (p_nir + p_red + 1e-9)
pavia_X_msi = np.column_stack([
    p_blue[pr_all, pc_all], p_green[pr_all, pc_all],
    p_red[pr_all, pc_all], p_nir[pr_all, pc_all], p_ndvi[pr_all, pc_all],
]).astype(np.float32)

print("Pavia cube:", pavia_cube.shape)
print("Labeled pixels:", len(pavia_y))
display(pd.Series({PAVIA_LABELS[k]: int((pavia_y == k).sum()) for k in PAVIA_LABELS}, name="pixels"))

rgb_pavia = np.stack([p_red, p_green, p_blue], axis=-1)
lo, hi = np.percentile(rgb_pavia, [2, 98], axis=(0, 1))
rgb_pavia = np.clip((rgb_pavia - lo) / (hi - lo + 1e-9), 0, 1)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(rgb_pavia); axes[0].set_title("Pavia University — RGB giả lập"); axes[0].axis("off")
axes[1].imshow(np.ma.masked_where(pavia_gt == 0, pavia_gt), cmap="tab10", vmin=0, vmax=9)
axes[1].set_title("Ground truth — 9 lớp"); axes[1].axis("off")
plt.tight_layout();
""")

md("### A1. Tạo MNF, PPI và FCLS features cho M3")

code(r"""
def fit_mnf_generic(X, differences, n_components):
    X = np.asarray(X, np.float64)
    differences = np.asarray(differences, np.float64)
    mean = X.mean(axis=0)
    noise_cov = np.cov(differences, rowvar=False) / 2
    reg = max(np.trace(noise_cov) / noise_cov.shape[0] * 1e-5, 1e-10)
    nv, ne = np.linalg.eigh(noise_cov + reg * np.eye(noise_cov.shape[0]))
    whitener = (ne * (1 / np.sqrt(np.maximum(nv, reg)))) @ ne.T
    Z = (X - mean) @ whitener
    sv, se = np.linalg.eigh(np.cov(Z, rowvar=False))
    order = np.argsort(sv)[::-1]
    components = whitener @ se[:, order[:n_components]]
    return mean, components, sv[order]

def ppi_counts_generic(scores, n_projections=1000, batch=128, seed=42):
    gen = np.random.default_rng(seed)
    counts = np.zeros(len(scores), np.int32)
    for start in range(0, n_projections, batch):
        n = min(batch, n_projections - start)
        directions = gen.normal(size=(scores.shape[1], n))
        directions /= np.linalg.norm(directions, axis=0, keepdims=True)
        projection = scores @ directions
        np.add.at(counts, np.argmax(projection, axis=0), 1)
        np.add.at(counts, np.argmin(projection, axis=0), 1)
    return counts

def project_simplex_generic(V):
    U = np.sort(V, axis=1)[:, ::-1]
    cssv = np.cumsum(U, axis=1) - 1
    j = np.arange(1, V.shape[1] + 1)
    rho = (U - cssv / j > 0).sum(axis=1) - 1
    theta = cssv[np.arange(len(V)), rho] / (rho + 1)
    return np.maximum(V - theta[:, None], 0)

def fcls_batch(X, endmembers, iterations=60):
    G = endmembers @ endmembers.T
    C = X @ endmembers.T
    step = 0.95 / np.linalg.norm(G, 2)
    F = np.full((len(X), len(endmembers)), 1 / len(endmembers), np.float32)
    for _ in range(iterations):
        F = project_simplex_generic(F - step * (F @ G - C)).astype(np.float32)
    return F

pavia_rng = np.random.default_rng(42)
fit_idx = pavia_rng.choice(len(pavia_X_full), min(PAVIA_FIT_SAMPLES, len(pavia_X_full)), replace=False)
left = pavia_cube[:, :-1, :].reshape(-1, pavia_cube.shape[2])
right = pavia_cube[:, 1:, :].reshape(-1, pavia_cube.shape[2])
noise_idx = pavia_rng.choice(len(left), min(PAVIA_FIT_SAMPLES, len(left)), replace=False)
pavia_noise = left[noise_idx] - right[noise_idx]

pavia_mnf_mean, pavia_mnf_components, _ = fit_mnf_generic(
    pavia_X_full[fit_idx], pavia_noise, min(N_MNF, pavia_X_full.shape[1])
)
pavia_X_mnf = ((pavia_X_full - pavia_mnf_mean) @ pavia_mnf_components).astype(np.float32)

ppi_idx = pavia_rng.choice(len(pavia_X_full), min(PAVIA_PPI_SAMPLES, len(pavia_X_full)), replace=False)
ppi_z = pavia_X_mnf[ppi_idx]
ppi_z = (ppi_z - ppi_z.mean(0)) / (ppi_z.std(0) + 1e-9)
ppi_count = ppi_counts_generic(ppi_z, n_projections=PAVIA_PPI_PROJECTIONS)
n_pure = max(9 * 10, int(np.ceil(0.01 * len(ppi_idx))))
pure_idx = ppi_idx[np.argsort(ppi_count)[-n_pure:]]
em_model = KMeans(n_clusters=9, n_init=30, random_state=42).fit(pavia_X_mnf[pure_idx, :10])
pavia_endmembers = np.vstack([
    pavia_X_full[pure_idx][em_model.labels_ == k].mean(axis=0) for k in range(9)
]).astype(np.float32)
pavia_fractions = fcls_batch(pavia_X_full, pavia_endmembers, FCLS_ITERATIONS)
pavia_X_unmix = np.column_stack([pavia_X_msi, pavia_X_mnf, pavia_fractions]).astype(np.float32)

fig, ax = plt.subplots(figsize=(12, 5))
for k, spectrum in enumerate(pavia_endmembers):
    ax.plot(pavia_wl, spectrum, lw=1.2, label=f"E{k}")
ax.set(xlabel="Wavelength (nm)", ylabel="Scaled signal", title="Pavia endmembers — PPI + K-means")
ax.legend(ncol=3);
""")

md(r"""
### A2. Spatial cross-validation cho ba cấu hình

Các block 32×32 pixel giữ các pixel lân cận trong cùng fold. Ba cấu hình dùng cùng classifier và cùng spatial folds. `QUICK_RUN` dùng ba folds, chế độ final dùng năm folds.
""")

code(r"""
PAVIA_BLOCK_PX = 32
pavia_groups = (pr_all // PAVIA_BLOCK_PX) * (pavia_cube.shape[1] // PAVIA_BLOCK_PX + 1) + (pc_all // PAVIA_BLOCK_PX)

def run_pavia_cv(X, name):
    cv = GroupKFold(n_splits=CV_FOLDS)
    records = []
    oof = np.zeros_like(pavia_y)
    for fold, (tr, te) in enumerate(cv.split(X, pavia_y, pavia_groups), 1):
        Xtr, Xte = X[tr], X[te]
        model = HistGradientBoostingClassifier(
            max_iter=MODEL_ITERATIONS, learning_rate=.07, max_leaf_nodes=31,
            l2_regularization=1, random_state=42,
        )
        model.fit(Xtr, pavia_y[tr])
        pred = model.predict(Xte)
        oof[te] = pred
        records.append({
            "method": name, "fold": fold,
            "accuracy": accuracy_score(pavia_y[te], pred),
            "macro_F1": f1_score(pavia_y[te], pred, labels=list(PAVIA_LABELS), average="macro", zero_division=0),
            "kappa": cohen_kappa_score(pavia_y[te], pred),
        })
    return pd.DataFrame(records), oof

pavia_methods = {
    "M1_MSI": pavia_X_msi,
    "M2_Full_HSI": pavia_X_full,
    "M3_HSI_Unmixing": pavia_X_unmix,
}
pavia_metric_parts, pavia_oof = [], {}
for name, X in pavia_methods.items():
    result, pred = run_pavia_cv(X, name)
    pavia_metric_parts.append(result); pavia_oof[name] = pred

pavia_metrics = pd.concat(pavia_metric_parts, ignore_index=True)
pavia_summary = pavia_metrics.groupby("method")[["accuracy", "macro_F1", "kappa"]].agg(["mean", "std"])
display(pavia_summary.round(3))

pavia_metrics.to_csv(OUT_DIR / "pavia_three_method_metrics.csv", index=False)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, (name, pred) in zip(axes.ravel(), pavia_oof.items()):
    cm = confusion_matrix(pavia_y, pred, labels=list(PAVIA_LABELS), normalize="true")
    im = ax.imshow(cm, vmin=0, vmax=1, cmap="Blues")
    ax.set_title(name); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_xticks(range(9)); ax.set_yticks(range(9));
    ax.set_xticklabels(range(1, 10)); ax.set_yticklabels(range(1, 10))
fig.colorbar(im, ax=axes.ravel().tolist(), shrink=.65, label="Row-normalized rate")
plt.show();
""")

md("## Phần B — Tanager HSI và Landsat LST")

md("### B1. Kiểm kê dữ liệu và lưới không gian")

code(r"""
inventory = []
for p in sorted((ROOT / "data").rglob("*")):
    if p.is_file():
        inventory.append({"file": str(p.relative_to(ROOT)), "size_MB": p.stat().st_size / 2**20})
display(pd.DataFrame(inventory).round(2))

with rasterio.open(RGB_FILE) as src:
    target_profile = src.profile.copy()
    target_transform = src.transform
    target_crs = src.crs
    height, width = src.height, src.width
    rgb = src.read([1, 2, 3]).transpose(1, 2, 0)

with open(STAC_FILE) as f:
    tanager_stac = json.load(f)

print("Tanager bbox:", tanager_stac["bbox"])
print("Grid:", width, "x", height, "|", target_crs, "|", target_transform)
plt.figure(figsize=(10, 8))
plt.imshow(rgb)
plt.title("Tanager ortho visual — 2025-04-07")
plt.axis("off");
""")

md(r"""
### B2. Đọc hyperspectral và tạo quality mask

Giữ các band được nhà sản xuất đánh dấu tốt trong ba cửa sổ: `400–1340`, `1490–1770`, `2050–2450 nm`. Các vùng hấp thụ khí quyển quanh 1.4 và 1.9 µm bị loại.
""")

code(r"""
FIELDS = "HDFEOS/GRIDS/HYP/Data Fields"
with h5py.File(H5_FILE, "r") as h5:
    sr_ds = h5[f"{FIELDS}/surface_reflectance"]
    sr_shape = sr_ds.shape
    wavelengths = sr_ds.attrs["wavelengths"].astype(float)
    fwhm = sr_ds.attrs["fwhm"].astype(float)
    good_native = sr_ds.attrs["good_wavelengths"].astype(bool)
    nodata = h5[f"{FIELDS}/nodata_pixels"][:]
    cloud = h5[f"{FIELDS}/beta_cloud_mask"][:]
    cirrus = h5[f"{FIELDS}/beta_cirrus_mask"][:]
    aod = h5[f"{FIELDS}/aerosol_optical_depth"][:]

spectral_windows = (
    ((wavelengths >= 400) & (wavelengths <= 1340)) |
    ((wavelengths >= 1490) & (wavelengths <= 1770)) |
    ((wavelengths >= 2050) & (wavelengths <= 2450))
)
usable = good_native & spectral_windows
usable_idx = np.flatnonzero(usable)
wl = wavelengths[usable]

valid = (nodata == 0) & (cloud == 0) & (cirrus == 0) & np.isfinite(aod) & (aod != -9999)
print("HDF5 shape:", sr_shape)
print("Band giữ lại:", usable.sum(), "/", len(wavelengths))
print("Pixel valid ban đầu:", f"{valid.mean():.1%}")
print("Mask values — nodata/cloud/cirrus:", np.unique(nodata), np.unique(cloud), np.unique(cirrus))

# ~700 MB; đủ cho toàn bộ pipeline nhưng vẫn nhỏ hơn việc nhân bản cube 426-band.
with h5py.File(H5_FILE, "r") as h5:
    cube = h5[f"{FIELDS}/surface_reflectance"][usable_idx, :, :].astype(np.float32)

valid &= np.isfinite(cube).all(axis=0) & (cube > -1).all(axis=0)
cube[:, ~valid] = np.nan
print("Pixel valid sau kiểm tra reflectance:", f"{valid.mean():.1%}")
""")

code(r"""
def nearest_band(target_nm):
    return int(np.argmin(np.abs(wl - target_nm)))

fig, ax = plt.subplots(figsize=(12, 5))
rr, cc = np.where(valid)
take = np.random.default_rng(42).choice(len(rr), size=min(1500, len(rr)), replace=False)
spectra_preview = cube[:, rr[take], cc[take]].T
q10, q50, q90 = np.nanpercentile(spectra_preview, [10, 50, 90], axis=0)
ax.fill_between(wl, q10, q90, alpha=.25, label="P10–P90")
ax.plot(wl, q50, lw=1.5, label="median")
for x, label in [(720,"red edge"), (960,"~960"), (1240,"water"), (1660,"materials"), (2200,"minerals")]:
    ax.axvline(x, ls="--", lw=.7, color="grey"); ax.text(x, ax.get_ylim()[1]*.9, label, rotation=90)
ax.set(xlabel="Wavelength (nm)", ylabel="Surface reflectance", title="Phân bố phổ Tanager sau quality mask")
ax.legend();
""")

md(r"""
### B3. Tạo M1 — Landsat OLI giả lập từ Tanager

RSR lấy từ USGS. Mỗi band được tính bằng tích phân có trọng số trên đúng các bước sóng Tanager. Ta dùng B2–B7 vì chúng nằm trong miền phổ của Tanager và phục vụ NDVI/NDBI/MNDWI.
""")

code(r"""
sheet_map = {"B2":"Blue", "B3":"Green", "B4":"Red", "B5":"NIR", "B6":"SWIR1", "B7":"SWIR2"}
rsr_weights = []
rsr_rows = []
for band, sheet in sheet_map.items():
    frame = pd.read_excel(RSR_FILE, sheet_name=sheet)
    x = pd.to_numeric(frame.iloc[:, 0], errors="coerce").to_numpy()
    y = pd.to_numeric(frame.iloc[:, 1], errors="coerce").to_numpy()
    ok = np.isfinite(x) & np.isfinite(y)
    response = np.interp(wl, x[ok], y[ok], left=0, right=0)
    response[response < 0] = 0
    response /= response.sum()
    rsr_weights.append(response)
    rsr_rows.append({"band":band, "sheet":sheet, "effective_nm":float((response*wl).sum())})
rsr_weights = np.asarray(rsr_weights, dtype=np.float32)
band_names = list(sheet_map)
display(pd.DataFrame(rsr_rows).round(2))

# Matrix multiplication theo từng chunk hàng để không tạo thêm bản sao cube lớn.
msi = np.full((len(band_names), height, width), np.nan, np.float32)
for r0 in range(0, height, 64):
    r1 = min(r0 + 64, height)
    block = cube[:, r0:r1, :]
    msi[:, r0:r1, :] = np.tensordot(rsr_weights, block, axes=(1, 0))

B2, B3, B4, B5, B6, B7 = msi
safe_ratio = lambda a, b: np.divide(a-b, a+b, out=np.full_like(a, np.nan), where=np.abs(a+b)>1e-6)
ndvi = safe_ratio(B5, B4)
ndbi = safe_ratio(B6, B5)
mndwi = safe_ratio(B3, B6)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, img, name, cmap in zip(axes, [ndvi, ndbi, mndwi], ["NDVI", "NDBI", "MNDWI"], ["RdYlGn", "RdBu_r", "BrBG"]):
    im=ax.imshow(img, cmap=cmap, vmin=-.5, vmax=.7); ax.set_title(name); ax.axis("off"); plt.colorbar(im, ax=ax, shrink=.7)
plt.tight_layout();
""")

md(r"""
### B4. MNF: noise whitening + PCA

Noise được ước lượng từ sai phân hai pixel kề nhau. Sau whitening theo covariance của noise, PCA được chạy trên mẫu 60.000 pixel. Giữ 16 thành phần đầu.
""")

code(r"""
rng = np.random.default_rng(42)
rr, cc = np.where(valid)
n_train = min(TRAIN_SAMPLES, len(rr))
sel = rng.choice(len(rr), n_train, replace=False)
X_train = cube[:, rr[sel], cc[sel]].T.astype(np.float64)

pair_mask = valid[:, :-1] & valid[:, 1:]
pr, pc = np.where(pair_mask)
n_noise = min(NOISE_SAMPLES, len(pr))
ps = rng.choice(len(pr), n_noise, replace=False)
noise_diff = (cube[:, pr[ps], pc[ps]] - cube[:, pr[ps], pc[ps]+1]).T.astype(np.float64)

def fit_mnf(X, differences, n_components=16):
    mean = X.mean(axis=0)
    noise_cov = np.cov(differences, rowvar=False) / 2
    reg = max(np.trace(noise_cov) / noise_cov.shape[0] * 1e-5, 1e-10)
    nv, ne = np.linalg.eigh(noise_cov + reg*np.eye(noise_cov.shape[0]))
    whitener = (ne * (1/np.sqrt(np.maximum(nv, reg)))) @ ne.T
    Z = (X - mean) @ whitener
    sv, se = np.linalg.eigh(np.cov(Z, rowvar=False))
    order = np.argsort(sv)[::-1]
    components = whitener @ se[:, order[:n_components]]
    return mean, components, sv[order]

t0=time.time()
mnf_mean, mnf_components, mnf_eigenvalues = fit_mnf(X_train, noise_diff, N_MNF)
mnf_map = np.full((N_MNF, height, width), np.nan, np.float32)
for r0 in range(0, height, 64):
    r1=min(r0+64, height)
    X=cube[:,r0:r1,:].reshape(len(wl),-1).T
    ok=np.isfinite(X).all(axis=1)
    scores=np.full((X.shape[0],N_MNF),np.nan,np.float32)
    scores[ok]=((X[ok]-mnf_mean)@mnf_components).astype(np.float32)
    mnf_map[:,r0:r1,:]=scores.T.reshape(N_MNF,r1-r0,width)
print(f"MNF hoàn tất sau {time.time()-t0:.1f}s")

fig, axes=plt.subplots(2,4,figsize=(16,8))
for i,ax in enumerate(axes.ravel()):
    lo,hi=np.nanpercentile(mnf_map[i],[2,98]); ax.imshow(mnf_map[i],cmap="gray",vmin=lo,vmax=hi); ax.set_title(f"MNF {i+1}"); ax.axis("off")
plt.tight_layout();
""")

md(r"""
### B5. PPI, endmember và kiểm tra phổ

PPI chiếu mẫu MNF theo các hướng ngẫu nhiên, đếm số lần mỗi pixel nằm ở cực đại/cực tiểu. Top 0,5% được gom thành 7 endmember bằng K-means. Vì chưa có ground truth vật liệu, notebook gọi chúng là `E0…E6`; bảng chỉ số hỗ trợ đặt tên sau khi xem ảnh và phổ.
""")

code(r"""
sample_n=min(PPI_SAMPLES,len(rr))
ppi_pick=rng.choice(len(rr),sample_n,replace=False)
ppi_rc=(rr[ppi_pick],cc[ppi_pick])
ppi_X=cube[:,ppi_rc[0],ppi_rc[1]].T
ppi_scores=(ppi_X-mnf_mean)@mnf_components
ppi_scores=(ppi_scores-ppi_scores.mean(0))/(ppi_scores.std(0)+1e-9)

def ppi_counts(scores,n_projections=2000,batch=128,seed=42):
    gen=np.random.default_rng(seed); counts=np.zeros(len(scores),np.int32)
    done=0
    while done<n_projections:
        n=min(batch,n_projections-done)
        directions=gen.normal(size=(scores.shape[1],n))
        directions/=np.linalg.norm(directions,axis=0,keepdims=True)
        projection=scores@directions
        np.add.at(counts,np.argmax(projection,axis=0),1)
        np.add.at(counts,np.argmin(projection,axis=0),1)
        done+=n
    return counts

counts=ppi_counts(ppi_scores,PPI_PROJECTIONS)
n_pure=max(N_ENDMEMBERS*5,int(np.ceil(PURE_FRACTION*sample_n)))
pure_local=np.argsort(counts)[-n_pure:]
kmeans=KMeans(n_clusters=N_ENDMEMBERS,n_init=30,random_state=42).fit(ppi_scores[pure_local,:10])
endmembers=np.vstack([ppi_X[pure_local][kmeans.labels_==k].mean(axis=0) for k in range(N_ENDMEMBERS)]).astype(np.float32)

fig,ax=plt.subplots(figsize=(13,6))
for k,e in enumerate(endmembers): ax.plot(wl,e,lw=1.4,label=f"E{k}")
ax.set(xlabel="Wavelength (nm)",ylabel="Reflectance",title="Endmember phổ lấy từ ảnh"); ax.legend(ncol=4);
""")

code(r"""
# Tích phân các endmember sang OLI để diễn giải nhóm phổ.
em_msi=endmembers@rsr_weights.T
em_B2,em_B3,em_B4,em_B5,em_B6,em_B7=em_msi.T
em_table=pd.DataFrame({
    "endmember":[f"E{k}" for k in range(N_ENDMEMBERS)],
    "NDVI":(em_B5-em_B4)/(em_B5+em_B4+1e-9),
    "NDBI":(em_B6-em_B5)/(em_B6+em_B5+1e-9),
    "MNDWI":(em_B3-em_B6)/(em_B3+em_B6+1e-9),
    "R_960":endmembers[:,nearest_band(960)],
    "R_1660":endmembers[:,nearest_band(1660)],
    "R_2200":endmembers[:,nearest_band(2200)],
})
display(em_table.round(3).sort_values("NDVI"))
""")

md(r"""
### B6. FCLS và SAM

FCLS dùng projected gradient descent trên simplex: mọi fraction không âm và tổng bằng 1. SAM tạo lớp cứng bổ trợ. Kết quả gồm fraction, reconstruction RMSE, nhãn SAM và spectral angle nhỏ nhất.
""")

code(r"""
def project_simplex(V):
    U=np.sort(V,axis=1)[:,::-1]
    cssv=np.cumsum(U,axis=1)-1
    j=np.arange(1,V.shape[1]+1)
    rho=(U-cssv/j>0).sum(axis=1)-1
    theta=cssv[np.arange(len(V)),rho]/(rho+1)
    return np.maximum(V-theta[:,None],0)

G=endmembers@endmembers.T
step=0.95/np.linalg.norm(G,2)
fractions=np.full((N_ENDMEMBERS,height,width),np.nan,np.float32)
recon_rmse=np.full((height,width),np.nan,np.float32)
sam_class=np.full((height,width),255,np.uint8)
sam_angle=np.full((height,width),np.nan,np.float32)
En=endmembers/(np.linalg.norm(endmembers,axis=1,keepdims=True)+1e-9)

for r0 in range(0,height,48):
    r1=min(r0+48,height)
    X=cube[:,r0:r1,:].reshape(len(wl),-1).T
    ok=np.isfinite(X).all(axis=1)
    Xv=X[ok]
    C=Xv@endmembers.T
    F=np.full((len(Xv),N_ENDMEMBERS),1/N_ENDMEMBERS,np.float32)
    for _ in range(FCLS_ITERATIONS):
        F=project_simplex(F-step*(F@G-C)).astype(np.float32)
    pred=F@endmembers
    rmse=np.sqrt(np.mean((pred-Xv)**2,axis=1))
    cosine=(Xv@En.T)/(np.linalg.norm(Xv,axis=1,keepdims=True)+1e-9)
    angles=np.arccos(np.clip(cosine,-1,1))
    flat_n=(r1-r0)*width
    fb=np.full((flat_n,N_ENDMEMBERS),np.nan,np.float32); fb[ok]=F
    fractions[:,r0:r1,:]=fb.T.reshape(N_ENDMEMBERS,r1-r0,width)
    rb=np.full(flat_n,np.nan,np.float32); rb[ok]=rmse; recon_rmse[r0:r1]=rb.reshape(r1-r0,width)
    cb=np.full(flat_n,255,np.uint8); cb[ok]=np.argmin(angles,axis=1); sam_class[r0:r1]=cb.reshape(r1-r0,width)
    ab=np.full(flat_n,np.nan,np.float32); ab[ok]=np.min(angles,axis=1); sam_angle[r0:r1]=ab.reshape(r1-r0,width)

print("Median reconstruction RMSE:",float(np.nanmedian(recon_rmse)))
fig,axes=plt.subplots(2,4,figsize=(16,8))
for k,ax in enumerate(axes.ravel()[:N_ENDMEMBERS]):
    im=ax.imshow(fractions[k],vmin=0,vmax=1,cmap="magma"); ax.set_title(f"Fraction E{k}"); ax.axis("off")
axes.ravel()[-1].imshow(sam_class,cmap="tab10"); axes.ravel()[-1].set_title("SAM class"); axes.ravel()[-1].axis("off")
plt.tight_layout();
""")

md(r"""
### B7. Landsat LST: QA, scale và đồng đăng ký

`QA_PIXEL` loại fill, dilated cloud, cirrus, cloud, cloud shadow và snow. `QA_RADSAT` loại pixel bão hòa. Công thức Collection 2 Level-2: `K = DN × 0.00341802 + 149`; sau đó đổi sang °C. Hai rows được warp trực tiếp về lưới Tanager và lấy trung bình nơi chồng lấn.
""")

code(r"""
def find_one(folder, suffix):
    hits=list(folder.glob(f"*{suffix}"))
    if len(hits)!=1: raise FileNotFoundError(f"Cần đúng 1 file *{suffix} trong {folder}; tìm thấy {len(hits)}")
    return hits[0]

def warp_landsat_lst(folder):
    st_path=find_one(folder,"_ST_B10.TIF")
    qa_path=find_one(folder,"_QA_PIXEL.TIF")
    sat_path=find_one(folder,"_QA_RADSAT.TIF")
    with rasterio.open(st_path) as st_src, rasterio.open(qa_path) as qa_src, rasterio.open(sat_path) as sat_src:
        dn=st_src.read(1); qa=qa_src.read(1); sat=sat_src.read(1)
        bad=np.zeros(qa.shape,bool)
        for bit in [0,1,2,3,4,5]: bad |= ((qa>>bit)&1).astype(bool)
        src_valid=(dn>0)&(~bad)&(sat==0)
        lst=np.where(src_valid,dn.astype(np.float32)*0.00341802+149-273.15,np.nan)
        dst=np.full((height,width),np.nan,np.float32)
        reproject(lst,dst,src_transform=st_src.transform,src_crs=st_src.crs,
                  dst_transform=target_transform,dst_crs=target_crs,
                  src_nodata=np.nan,dst_nodata=np.nan,resampling=Resampling.bilinear)
    return dst,src_valid.mean()

scene_dirs=sorted([p for p in LANDSAT_DIR.iterdir() if p.is_dir() and p.name.startswith("LC08")])
assert len(scene_dirs)==2, f"Cần hai thư mục Landsat, hiện có {len(scene_dirs)}"
warped=[]
for folder in scene_dirs:
    arr,rate=warp_landsat_lst(folder); warped.append(arr); print(folder.name,"native valid",f"{rate:.1%}")
lst_c=np.nanmean(np.stack(warped),axis=0)
common=valid&np.isfinite(lst_c)
print("Common valid Tanager + Landsat LST:",f"{common.mean():.1%}","| pixels:",common.sum())

fig,ax=plt.subplots(figsize=(10,8)); lo,hi=np.nanpercentile(lst_c,[2,98]); im=ax.imshow(lst_c,cmap="inferno",vmin=lo,vmax=hi)
ax.set_title("Landsat 8 LST (°C) — 2025-04-09"); ax.axis("off"); plt.colorbar(im,ax=ax,shrink=.75,label="°C");
""")

md(r"""
### B8. Reference-based LST anomaly and hotspot assessment

Dùng pixel thực vật có `NDVI ≥ 0.35` và `MNDWI < 0.1` làm tập tham chiếu. Chỉ số nhiệt tại mỗi pixel là `ΔLST = LST − mean(LST vegetation reference)`; hotspot quan sát được định nghĩa bằng phân vị P90 của LST hợp lệ.
""")

code(r"""
VEGETATION_NDVI_THRESHOLD=0.35
vegetation_reference=common&(ndvi>=VEGETATION_NDVI_THRESHOLD)&(mndwi<0.1)
assert vegetation_reference.sum()>100, "Quá ít pixel vegetation reference; kiểm tra NDVI threshold"
veg_mean=float(np.nanmean(lst_c[vegetation_reference]))
delta_lst=lst_c-veg_mean
hotspot_threshold=float(np.nanquantile(lst_c[common],0.90))
observed_hotspot=common&(lst_c>=hotspot_threshold)

uhi_summary=pd.Series({
    "valid_pixels":int(common.sum()),
    "vegetation_reference_pixels":int(vegetation_reference.sum()),
    "vegetation_NDVI_threshold":VEGETATION_NDVI_THRESHOLD,
    "vegetation_reference_mean_LST_C":veg_mean,
    "scene_mean_LST_C":float(np.nanmean(lst_c[common])),
    "scene_mean_minus_vegetation_C":float(np.nanmean(delta_lst[common])),
    "hotspot_threshold_P90_C":hotspot_threshold,
})
display(uhi_summary.round(3))
""")

md(r"""
### B9. Ba phương pháp dự báo LST với spatial cross-validation

Tất cả phương pháp dùng **cùng Landsat LST target**, cùng tập pixel, block không gian 1 km và cùng `HistGradientBoostingRegressor`. `QUICK_RUN` dùng tối đa 30.000 pixel và ba folds; chế độ final dùng tối đa 120.000 pixel và năm folds.

- **M1 — Multispectral Baseline:** sáu band OLI giả lập + NDVI/NDBI/MNDWI.
- **M2 — Full-Spectrum Hyperspectral:** toàn bộ band Tanager hợp lệ.
- **M3 — Hyperspectral Unmixing Features:** M1 + 16 MNF components + 7 FCLS fractions.

Metrics gồm R², RMSE, MAE và F1 phát hiện 10% pixel nóng nhất.
""")

code(r"""
rows,cols=np.where(common)
if len(rows)>MAX_BENCHMARK_PIXELS:
    keep=rng.choice(len(rows),MAX_BENCHMARK_PIXELS,replace=False); rows,cols=rows[keep],cols[keep]
y=lst_c[rows,cols]
X_msi=np.column_stack([msi[:,rows,cols].T,ndvi[rows,cols],ndbi[rows,cols],mndwi[rows,cols]])
X_full_hsi=cube[:,rows,cols].T
X_unmix=np.column_stack([X_msi,mnf_map[:,rows,cols].T,fractions[:,rows,cols].T])
ok=(np.isfinite(X_msi).all(axis=1) & np.isfinite(X_full_hsi).all(axis=1) &
    np.isfinite(X_unmix).all(axis=1) & np.isfinite(y))
rows,cols,y=rows[ok],cols[ok],y[ok]
X_msi,X_full_hsi,X_unmix=X_msi[ok],X_full_hsi[ok],X_unmix[ok]
block_px=max(1,round(SPATIAL_BLOCK_M/abs(target_transform.a)))
groups=(rows//block_px)*(width//block_px+1)+(cols//block_px)
print("Samples:",len(y),"| spatial groups:",len(np.unique(groups)),"| block pixels:",block_px)

def run_cv(X,name):
    records=[]; cv=GroupKFold(n_splits=CV_FOLDS)
    for fold,(tr,te) in enumerate(cv.split(X,y,groups),1):
        Xtr,Xte=X[tr],X[te]
        model=HistGradientBoostingRegressor(max_iter=MODEL_ITERATIONS,learning_rate=.06,max_leaf_nodes=31,l2_regularization=1,random_state=42)
        model.fit(Xtr,y[tr]); pred=model.predict(Xte)
        threshold=np.quantile(y[tr],.90)
        records.append({"features":name,"fold":fold,"R2":r2_score(y[te],pred),
                        "RMSE_C":mean_squared_error(y[te],pred)**.5,
                        "MAE_C":mean_absolute_error(y[te],pred),
                        "Hotspot_F1":f1_score(y[te]>=threshold,pred>=threshold)})
    return pd.DataFrame(records)

metric_parts=[]
for X,name in [
    (X_msi,"M1_MSI"),
    (X_full_hsi,"M2_Full_HSI"),
    (X_unmix,"M3_HSI_Unmixing"),
]:
    metric_parts.append(run_cv(X,name))
metrics=pd.concat(metric_parts,ignore_index=True)
display(metrics.groupby("features").agg(["mean","std"]).round(3))
metrics.to_csv(OUT_DIR/"spatial_cv_metrics.csv",index=False)
""")

code(r"""
summary_metrics=metrics.groupby("features")[["R2","RMSE_C","MAE_C","Hotspot_F1"]].mean()
comparison=summary_metrics.copy()
for metric in ["R2","Hotspot_F1"]:
    comparison[f"Delta_{metric}_vs_M1"]=comparison[metric]-comparison.loc["M1_MSI",metric]
comparison["RMSE_reduction_vs_M1_C"]=summary_metrics.loc["M1_MSI","RMSE_C"]-comparison["RMSE_C"]
display(comparison.round(3))

ax=summary_metrics[["RMSE_C","MAE_C"]].plot.bar(figsize=(10,4),rot=15,title="Three-method spatial CV error")
ax.set_ylabel("°C"); plt.tight_layout();
""")

md(r"""
### B10. Xuất raster và bảng kết quả

Các raster giữ đúng `EPSG:32648`, transform và kích thước của Tanager. Tệp float dùng `NaN` làm nodata; lớp dùng 255.
""")

code(r"""
def write_float_tif(path,array,descriptions=None):
    data=array[None] if array.ndim==2 else array
    profile=target_profile.copy(); profile.update(driver="GTiff",count=data.shape[0],dtype="float32",nodata=np.nan,compress="deflate",predictor=3)
    with rasterio.open(path,"w",**profile) as dst:
        dst.write(data.astype(np.float32))
        if descriptions: dst.descriptions=tuple(descriptions)

def write_byte_tif(path,array,description):
    profile=target_profile.copy(); profile.update(driver="GTiff",count=1,dtype="uint8",nodata=255,compress="deflate",predictor=2)
    with rasterio.open(path,"w",**profile) as dst:
        dst.write(array.astype(np.uint8),1); dst.set_band_description(1,description)

write_float_tif(OUT_DIR/"tanager_msi_sim.tif",msi,band_names)
write_float_tif(OUT_DIR/"spectral_indices.tif",np.stack([ndvi,ndbi,mndwi]),["NDVI","NDBI","MNDWI"])
write_float_tif(OUT_DIR/"mnf_16.tif",mnf_map,[f"MNF_{i+1}" for i in range(N_MNF)])
write_float_tif(OUT_DIR/"endmember_fractions.tif",fractions,[f"E{i}_fraction" for i in range(N_ENDMEMBERS)])
write_float_tif(OUT_DIR/"landsat_lst_c.tif",lst_c,["LST_C"])
write_float_tif(OUT_DIR/"delta_lst_from_vegetation.tif",delta_lst,["Delta_LST_C"])
write_byte_tif(OUT_DIR/"sam_class.tif",sam_class,"SAM_class")
em_table.to_csv(OUT_DIR/"endmember_summary.csv",index=False)
uhi_summary.rename("value").to_csv(OUT_DIR/"uhi_index_summary.csv",header=True)
print("Đã ghi:")
for p in sorted(OUT_DIR.iterdir()): print(" -",p.name)
""")

md(r"""
## Cách diễn giải kết quả chung

1. Dùng Pavia để kết luận phương pháp nào phân biệt vật liệu/lớp phủ tốt hơn qua accuracy, macro-F1 và Kappa.
2. Dùng Tanager–Landsat để kết luận phương pháp nào dự đoán LST và hotspot tốt hơn qua R², RMSE, MAE và hotspot F1.
3. M3 cho biết MNF và abundance fractions có cung cấp tín hiệu dễ giải thích hơn phổ thô hay không.

Không chuyển classifier Pavia trực tiếp sang Tanager. Ta chuyển **experiment protocol** và ba cách tạo feature; model được fit lại cho target tương ứng. Pavia không có LST, còn Tanager không có nhãn vật liệu. LST Landsat lệch Tanager hai ngày, nên phần B là proof-of-concept và chưa loại bỏ hoàn toàn ảnh hưởng thời tiết/thời gian.
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python (.venv HSI Lab)", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"},
}
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print(OUT)
