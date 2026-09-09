# HSI Lab — benchmark hyperspectral và multispectral cho UHI

Mở `notebook/01_tanager_hsi_vs_landsat_uhi.ipynb` trong VS Code và chọn kernel
`Python (.venv HSI Lab)`. Notebook bao gồm kiểm tra dữ liệu, mask chất lượng,
giả lập Landsat OLI bằng RSR chính thức, MNF, PPI, endmember, FCLS, SAM,
pipeline bốn lớp kiểu Brazil, Landsat LST và spatial cross-validation.

Môi trường Python nằm trong `.venv`. Nếu cần tạo lại:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Dữ liệu Landsat có thể tải lại hoặc tiếp tục bằng:

```bash
.venv/bin/python scripts/download_landsat.py
```

