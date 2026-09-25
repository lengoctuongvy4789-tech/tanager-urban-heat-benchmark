# HSI Lab — bốn phương pháp hyperspectral/multispectral cho UHI

Mở `notebook/01_tanager_hsi_vs_landsat_uhi.ipynb` trong VS Code và chọn kernel
`Python (.venv HSI Lab)`, sau đó chọn **Run All**. Notebook chạy hai phần:

1. Pavia University có nhãn: so sánh M1 MSI, M2 Full HSI, M3 Selected HSI và
   M4 HSI unmixing cho phân loại vật liệu/lớp phủ.
2. Tanager + Landsat: chạy lại bốn cấu hình với Landsat LST làm target chung,
   rồi so sánh R², RMSE, MAE và hotspot F1 bằng spatial cross-validation.

Pipeline Brazil được giữ để tính và diễn giải LST/UHI theo lớp bề mặt.

Môi trường Python nằm trong `.venv`. Nếu cần tạo lại:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Dữ liệu Landsat có thể tải lại hoặc tiếp tục bằng:

```bash
.venv/bin/python scripts/download_landsat.py
```

Pavia University có thể tải lại bằng:

```bash
.venv/bin/python scripts/download_pavia.py
```

Tạo lại notebook sau khi sửa builder:

```bash
.venv/bin/python scripts/build_notebook.py
```
