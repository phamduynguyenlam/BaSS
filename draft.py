import matplotlib.pyplot as plt
import re

# Dữ liệu log của bạn
log_data = """
[DISC] iter 0 | front = 14 | HV = 2.604352
[DISC] iter 1 | front = 3 | HV = 3.564087 | reward = 4.462951 | disc_forward_sec = 0.053
[DISC] iter 2 | front = 4 | HV = 3.825826 | reward = 5.978568 | disc_forward_sec = 0.009
[DISC] iter 3 | front = 5 | HV = 3.971056 | reward = 3.228021 | disc_forward_sec = 0.009
[DISC] iter 4 | front = 5 | HV = 4.221045 | reward = 3.167552 | disc_forward_sec = 0.009
[DISC] iter 5 | front = 4 | HV = 4.416767 | reward = 2.341870 | disc_forward_sec = 0.009
[DISC] iter 6 | front = 5 | HV = 4.426374 | reward = 1.285129 | disc_forward_sec = 0.009
[DISC] iter 7 | front = 6 | HV = 4.468890 | reward = 1.809340 | disc_forward_sec = 0.011
[DISC] iter 8 | front = 6 | HV = 4.468890 | reward = -1.000000 | disc_forward_sec = 0.009
[DISC] iter 9 | front = 5 | HV = 4.527147 | reward = 2.163219 | disc_forward_sec = 0.009
[DISC] iter 10 | front = 6 | HV = 4.527266 | reward = 1.723850 | disc_forward_sec = 0.010
[DISC] iter 11 | front = 7 | HV = 4.534056 | reward = 1.832261 | disc_forward_sec = 0.009
[DISC] iter 12 | front = 7 | HV = 4.538252 | reward = 1.452967 | disc_forward_sec = 0.012
[DISC] iter 13 | front = 7 | HV = 4.567177 | reward = 1.634554 | disc_forward_sec = 0.010
[DISC] iter 14 | front = 8 | HV = 4.593055 | reward = 2.088149 | disc_forward_sec = 0.010
[DISC] iter 15 | front = 9 | HV = 4.593656 | reward = 1.298118 | disc_forward_sec = 0.010
[DISC] iter 16 | front = 10 | HV = 4.595061 | reward = 1.279868 | disc_forward_sec = 0.010
[DISC] iter 17 | front = 10 | HV = 4.595061 | reward = -1.000000 | disc_forward_sec = 0.012
[DISC] iter 18 | front = 11 | HV = 4.595544 | reward = 1.254534 | disc_forward_sec = 0.011
[DISC] iter 19 | front = 11 | HV = 4.596318 | reward = 1.111852 | disc_forward_sec = 0.013
[DISC] iter 20 | front = 12 | HV = 4.596959 | reward = 1.123319 | disc_forward_sec = 0.009
[DISC] iter 21 | front = 11 | HV = 4.617803 | reward = 1.964363 | disc_forward_sec = 0.009
[DISC] iter 22 | front = 11 | HV = 4.617803 | reward = -1.000000 | disc_forward_sec = 0.017
[DISC] iter 23 | front = 11 | HV = 4.618185 | reward = 1.078608 | disc_forward_sec = 0.012
[DISC] iter 24 | front = 12 | HV = 4.619295 | reward = 1.354105 | disc_forward_sec = 0.010
[DISC] iter 25 | front = 12 | HV = 4.619295 | reward = -1.000000 | disc_forward_sec = 0.014
[DISC] iter 26 | front = 13 | HV = 4.646457 | reward = 1.873718 | disc_forward_sec = 0.011
[DISC] iter 27 | front = 13 | HV = 4.646457 | reward = -1.000000 | disc_forward_sec = 0.010
[DISC] iter 28 | front = 12 | HV = 4.647664 | reward = 1.181410 | disc_forward_sec = 0.018
[DISC] iter 29 | front = 12 | HV = 4.659708 | reward = 1.597458 | disc_forward_sec = 0.024
[DISC] iter 30 | front = 13 | HV = 4.660601 | reward = 1.403681 | disc_forward_sec = 0.021
[DISC] iter 31 | front = 14 | HV = 4.660787 | reward = 1.101909 | disc_forward_sec = 0.020
[DISC] iter 32 | front = 14 | HV = 4.662176 | reward = 1.238665 | disc_forward_sec = 0.011
[DISC] iter 33 | front = 14 | HV = 4.665944 | reward = 1.451621 | disc_forward_sec = 0.012
[DISC] iter 34 | front = 14 | HV = 4.665944 | reward = -1.000000 | disc_forward_sec = 0.010
[DISC] iter 35 | front = 14 | HV = 4.665944 | reward = -1.000000 | disc_forward_sec = 0.011
[DISC] iter 36 | front = 14 | HV = 4.665944 | reward = -1.000000 | disc_forward_sec = 0.014
[DISC] iter 37 | front = 14 | HV = 4.676725 | reward = 1.404411 | disc_forward_sec = 0.012
[DISC] iter 38 | front = 14 | HV = 4.676725 | reward = -1.000000 | disc_forward_sec = 0.019
[DISC] iter 39 | front = 15 | HV = 4.676727 | reward = 1.165293 | disc_forward_sec = 0.017
[DISC] iter 40 | front = 15 | HV = 4.676727 | reward = -1.000000 | disc_forward_sec = 0.019
[EHVI] iter 0 | front = 14 | HV = 2.604352
[EHVI] iter 1 | front = 3 | HV = 3.361868 | reward = 3.898838 | ehvi_select_sec = 4.499
[EHVI] iter 2 | front = 3 | HV = 3.472412 | reward = 1.825503 | ehvi_select_sec = 0.427
[EHVI] iter 3 | front = 4 | HV = 3.546592 | reward = 1.556512 | ehvi_select_sec = 0.357
[EHVI] iter 4 | front = 5 | HV = 3.549556 | reward = 1.259391 | ehvi_select_sec = 0.366
[EHVI] iter 5 | front = 5 | HV = 3.549556 | reward = -1.000000 | ehvi_select_sec = 0.127
[EHVI] iter 6 | front = 5 | HV = 3.549556 | reward = -1.000000 | ehvi_select_sec = 0.160
[EHVI] iter 7 | front = 5 | HV = 3.549556 | reward = -1.000000 | ehvi_select_sec = 0.312
[EHVI] iter 8 | front = 5 | HV = 3.549556 | reward = -1.000000 | ehvi_select_sec = 0.202
[EHVI] iter 9 | front = 5 | HV = 3.565285 | reward = 1.463470 | ehvi_select_sec = 0.178
[EHVI] iter 10 | front = 6 | HV = 3.565308 | reward = 1.292708 | ehvi_select_sec = 0.402
[EHVI] iter 11 | front = 5 | HV = 3.565408 | reward = 1.125496 | ehvi_select_sec = 0.835
[EHVI] iter 12 | front = 6 | HV = 3.565472 | reward = 1.025114 | ehvi_select_sec = 0.495
[EHVI] iter 13 | front = 7 | HV = 3.565505 | reward = 1.149215 | ehvi_select_sec = 0.872
[EHVI] iter 14 | front = 7 | HV = 3.565505 | reward = -1.000000 | ehvi_select_sec = 0.749
[EHVI] iter 15 | front = 7 | HV = 3.565505 | reward = -1.000000 | ehvi_select_sec = 0.743
[EHVI] iter 16 | front = 5 | HV = 3.597297 | reward = 1.225099 | ehvi_select_sec = 0.886
[EHVI] iter 17 | front = 4 | HV = 3.597665 | reward = 1.233068 | ehvi_select_sec = 0.346
[EHVI] iter 18 | front = 4 | HV = 3.597665 | reward = -1.000000 | ehvi_select_sec = 0.192
[EHVI] iter 19 | front = 4 | HV = 3.597665 | reward = -1.000000 | ehvi_select_sec = 0.295
[EHVI] iter 20 | front = 4 | HV = 3.597665 | reward = -1.000000 | ehvi_select_sec = 0.064
[EHVI] iter 21 | front = 4 | HV = 3.597665 | reward = -1.000000 | ehvi_select_sec = 0.064
[EHVI] iter 22 | front = 4 | HV = 3.597665 | reward = -1.000000 | ehvi_select_sec = 0.110
[EHVI] iter 23 | front = 4 | HV = 3.699579 | reward = 1.440203 | ehvi_select_sec = 0.041
[EHVI] iter 24 | front = 4 | HV = 3.699579 | reward = -1.000000 | ehvi_select_sec = 0.084
[EHVI] iter 25 | front = 4 | HV = 3.699579 | reward = -1.000000 | ehvi_select_sec = 0.057
[EHVI] iter 26 | front = 4 | HV = 3.699579 | reward = -1.000000 | ehvi_select_sec = 0.086
[EHVI] iter 27 | front = 4 | HV = 3.699579 | reward = -1.000000 | ehvi_select_sec = 0.052
[EHVI] iter 28 | front = 4 | HV = 3.699579 | reward = -1.000000 | ehvi_select_sec = 0.035
[EHVI] iter 29 | front = 4 | HV = 3.699579 | reward = -1.000000 | ehvi_select_sec = 0.124
[EHVI] iter 30 | front = 4 | HV = 3.699579 | reward = -1.000000 | ehvi_select_sec = 0.121
[EHVI] iter 31 | front = 4 | HV = 3.699579 | reward = -1.000000 | ehvi_select_sec = 0.034
[EHVI] iter 32 | front = 4 | HV = 3.699579 | reward = -1.000000 | ehvi_select_sec = 0.069
[EHVI] iter 33 | front = 4 | HV = 3.699579 | reward = -1.000000 | ehvi_select_sec = 0.077
[EHVI] iter 34 | front = 4 | HV = 3.699579 | reward = -1.000000 | ehvi_select_sec = 0.266
[EHVI] iter 35 | front = 4 | HV = 3.699579 | reward = -1.000000 | ehvi_select_sec = 0.147
[EHVI] iter 36 | front = 4 | HV = 3.699579 | reward = -1.000000 | ehvi_select_sec = 0.072
[EHVI] iter 37 | front = 4 | HV = 3.699579 | reward = -1.000000 | ehvi_select_sec = 0.059
[EHVI] iter 38 | front = 4 | HV = 3.699579 | reward = -1.000000 | ehvi_select_sec = 0.100
[EHVI] iter 39 | front = 4 | HV = 3.699579 | reward = -1.000000 | ehvi_select_sec = 0.066
[EHVI] iter 40 | front = 4 | HV = 3.699579 | reward = -1.000000 | ehvi_select_sec = 0.079
"""

# Khởi tạo các mảng lưu trữ
disc_iters, disc_hvs = [], []
ehvi_iters, ehvi_hvs = [], []

# Regex pattern để tìm tên thuật toán, số iter và giá trị HV
pattern = re.compile(r'\[(DISC|EHVI)\] iter (\d+).*?HV = ([\d\.]+)')

# Xử lý log
for line in log_data.strip().split('\n'):
    match = pattern.search(line)
    if match:
        method = match.group(1)
        iteration = int(match.group(2))
        hv = float(match.group(3))
        
        if method == 'DISC':
            disc_iters.append(iteration)
            disc_hvs.append(hv)
        elif method == 'EHVI':
            ehvi_iters.append(iteration)
            ehvi_hvs.append(hv)

# Vẽ đồ thị
plt.figure(figsize=(10, 6))

# Cấu hình đường cho DISC
plt.plot(disc_iters, disc_hvs, label='DISC', color='blue', linewidth=2, marker='o', markersize=4)

# Cấu hình đường cho EHVI
plt.plot(ehvi_iters, ehvi_hvs, label='EHVI', color='red', linewidth=2, marker='s', markersize=4)

# Thêm tiêu đề và nhãn
plt.title('Hypervolume (HV) progression over Iterations', fontsize=14)
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Hypervolume (HV)', fontsize=12)

# Bật lưới và chú thích
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)

# Hiển thị đồ thị
plt.tight_layout()
plt.show()