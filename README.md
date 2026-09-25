# HSI Lab — ba phương pháp hyperspectral/multispectral cho UHI

Mở `notebook/01_tanager_hsi_vs_landsat_uhi.ipynb` trong VS Code và chọn kernel
`Python (.venv HSI Lab)`, sau đó chọn **Run All**. Notebook chạy hai phần:

1. Pavia University có nhãn: so sánh M1 MSI, M2 Full HSI và M3 HSI unmixing
   cho phân loại vật liệu/lớp phủ.
2. Tanager + Landsat: chạy lại ba cấu hình với Landsat LST làm target chung,
   rồi so sánh R², RMSE, MAE và hotspot F1 bằng spatial cross-validation.

Paper Brazil chỉ được dùng để tham khảo cách tính và diễn giải LST/UHI. Notebook
không chạy SLIC hoặc quy trình phân lớp bốn bề mặt của paper đó.

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
