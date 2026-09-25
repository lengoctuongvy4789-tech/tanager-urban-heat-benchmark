# Dữ liệu thử nghiệm HSI–UHI

- `tanager/`: Tanager surface reflectance, mask, RGB và STAC metadata ngày 2025-04-07.
- `landsat/`: Landsat 8 Collection 2 Level-2 ngày 2025-04-09, path 125 rows 052–053.
- `rsr/`: Relative Spectral Response của Landsat 8 OLI từ USGS.
- `benchmark/pavia_university/`: cube `PaviaU.mat` và ground truth `PaviaU_gt.mat`.

Notebook dùng Tanager để tạo cả HSI-full và MSI-sim trong cùng điều kiện quan sát.
Landsat `ST_B10` là biến nhiệt độ bề mặt dùng chung. Chênh lệch thời gian là hai ngày.

Nguồn Pavia:

- Trang dữ liệu gốc: https://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes
- Cube: https://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat
- Ground truth: https://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat
- Mirror công khai dùng khi máy chủ gốc trả 403: https://github.com/gokriznastic/HybridSN/tree/master/data

Chạy `../.venv/bin/python ../scripts/download_pavia.py` từ thư mục này, hoặc
`.venv/bin/python scripts/download_pavia.py` từ project root.
