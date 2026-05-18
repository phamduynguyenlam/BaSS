import time
import torch
import numpy as np
from tabpfn import TabPFNClassifier
from sklearn.base import clone

# ==========================================
# 1. CẤU HÌNH THÔNG SỐ BÀI TOÁN
# ==========================================
NUM_ENVS = 24
N_SAMPLES = 80
N_FEATURES = 30
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
ENSEMBLE_CONFIG = {'n_estimators': 1} 

print("1. Đang tạo dữ liệu giả lập cho 24 environments...")
tasks = []
for _ in range(NUM_ENVS):
    # Dữ liệu dạng float32 để khớp với tensor của PyTorch
    X_train = np.random.randn(N_SAMPLES, N_FEATURES).astype(np.float32)
    y_train = np.random.randint(0, 4, size=N_SAMPLES).astype(np.float32) # TabPFN core nhận label dạng float
    X_test = np.random.randn(N_SAMPLES, N_FEATURES).astype(np.float32)
    
    tasks.append((X_train, y_train, X_test))

# ==========================================
# 2. KHỞI TẠO VÀ WARM-UP GPU
# ==========================================
print(f"\n2. Đang nạp mô hình TabPFN lên {DEVICE}...")
base_clf = TabPFNClassifier(device=DEVICE, **ENSEMBLE_CONFIG)

print("   -> Đang chạy Warm-up (khởi động CUDA & khởi tạo models_)...")
# Phải gọi fit() 1 lần để API bọc ngoài (wrapper) khởi tạo core model PyTorch bên trong
base_clf.fit(tasks[0][0], tasks[0][1])
_ = base_clf.predict(tasks[0][2])

# Trích xuất Core PyTorch Model từ wrapper
# Trích xuất Core PyTorch Model từ wrapper và ÉP lên GPU
pytorch_model = base_clf.model_.to(DEVICE)
pytorch_model.eval()

# ==========================================
# 3. CHẠY SONG SONG TUYỆT ĐỐI (TENSOR BATCHING)
# ==========================================
print("\n3. Bắt đầu chạy SONG SONG TUYỆT ĐỐI (Gộp 24 task vào 1 Tensor 3D)...")
start_par = time.perf_counter()

# Kích thước theo chuẩn Transformer của TabPFN: [seq_len, batch_size, features]
# seq_len = train_samples + test_samples
seq_len = N_SAMPLES + N_SAMPLES 

# Cấp phát sẵn VRAM cho ma trận khổng lồ
x_full = torch.zeros((seq_len, NUM_ENVS, N_FEATURES), dtype=torch.float32, device=DEVICE)
y_full = torch.zeros((N_SAMPLES, NUM_ENVS), dtype=torch.float32, device=DEVICE)

# Nhồi 24 bộ dữ liệu vào khối Tensor
for i, (X_tr, y_tr, X_te) in enumerate(tasks):
    x_full[:N_SAMPLES, i, :] = torch.from_numpy(X_tr).to(DEVICE)
    x_full[N_SAMPLES:, i, :] = torch.from_numpy(X_te).to(DEVICE)
    y_full[:, i] = torch.from_numpy(y_tr).to(DEVICE)

# Suy luận toàn bộ 24 task trong CHỈ 1 LƯỢT GỌI (True Parallelism)
with torch.inference_mode():
    # Tự động tối ưu dtype bằng autocast nếu GPU hỗ trợ
    with torch.autocast(device_type='cuda', enabled=(DEVICE=='cuda')): 
        logits = pytorch_model(
            x_full,
            y_full,
            only_return_standard_out=True
        )

# Xử lý output: Logits có shape [test_samples, batch_size, num_classes] (ví dụ: [80, 24, 4])
# Chuyển về nhãn phân loại (argmax) và đảo chiều (permute) thành [24, 80] để khớp với danh sách task ban đầu
preds_par = logits.argmax(dim=-1).permute(1, 0).cpu().numpy()

end_par = time.perf_counter()
time_par = end_par - start_par
print(f"   -> Hoàn thành trong: {time_par:.4f} giây")

# ==========================================
# 4. CHẠY TUẦN TỰ (SEQUENTIAL) ĐỂ SO SÁNH
# ==========================================
print("\n4. Bắt đầu chạy TUẦN TỰ (1 task / lượt)...")
start_seq = time.perf_counter()
results_seq = []

for X_train, y_train, X_test in tasks:
    clf = clone(base_clf)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    results_seq.append(preds)

end_seq = time.perf_counter()
time_seq = end_seq - start_seq
print(f"   -> Hoàn thành trong: {time_seq:.4f} giây")

# ==========================================
# 5. BÁO CÁO KẾT QUẢ
# ==========================================
print("\n" + "="*60)
print("BÁO CÁO HIỆU NĂNG: TRUE BATCHING vs SEQUENTIAL".center(60))
print("="*60)
print(f"Số lượng Environments : {NUM_ENVS}")
print(f"Kích thước mỗi Task   : Train {N_SAMPLES}x{N_FEATURES} | Test {N_SAMPLES}x{N_FEATURES}")
print(f"Thiết bị (Device)     : {DEVICE}")
print("-" * 60)
print(f"Thời gian TENSOR BATCH: {time_par:.4f} giây")
print(f"Thời gian TUẦN TỰ     : {time_seq:.4f} giây")
print(f"Tốc độ cải thiện      : Nhanh hơn {time_seq / time_par:.2f} lần")
print("="*60)