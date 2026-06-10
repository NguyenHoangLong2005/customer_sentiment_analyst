import ctypes
import time
import sys
from datetime import datetime

# Hằng số của Windows API để giữ màn hình và hệ thống thức
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002

def prevent_sleep():
    # Gọi API Windows để ngăn máy tính và màn hình tự động tắt/ngủ
    ctypes.windll.kernel32.SetThreadExecutionState(
        ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
    )
    print("🟢 [ON] Đã bật chế độ chống Sleep. Màn hình và máy tính sẽ luôn bật.")
    print("Nhấn Ctrl+C để tắt và cho phép máy tính ngủ trở lại...")

def allow_sleep():
    # Phục hồi trạng thái bình thường
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
    print("\n🔴 [OFF] Đã cho phép máy tính Sleep trở lại bình thường.")

if __name__ == "__main__":
    try:
        prevent_sleep()
        # Chạy vòng lặp vô hạn, mỗi 10 phút in ra một thông báo để biết script vẫn đang chạy
        while True:
            time.sleep(600)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Script vẫn đang chạy...")
    except KeyboardInterrupt:
        allow_sleep()
        sys.exit(0)
