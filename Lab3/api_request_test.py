import requests
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

API_URL = "http://192.168.28.94:8000/predict/" 
IMAGE_PATH = "Lab3/images.jpeg"
NUM_REQUESTS = 100 
MAX_WORKERS = 10   

def send_request():
    try:
        # Kiểm tra file tồn tại
        if not os.path.exists(IMAGE_PATH):
            return f"File not found: {IMAGE_PATH}", 0
        
        with open(IMAGE_PATH, "rb") as f:
            files = {'file': (os.path.basename(IMAGE_PATH), f, 'image/jpeg')}
            response = requests.post(API_URL, files=files, timeout=30)
            
            if response.status_code != 200:
                return f"HTTP {response.status_code}: {response.text[:100]}", response.elapsed.total_seconds()
            
            return response.status_code, response.elapsed.total_seconds()
            
    except requests.exceptions.ConnectionError as e:
        return f"Connection error: {str(e)[:100]}", 0
    except requests.exceptions.Timeout as e:
        return f"Timeout error: {str(e)[:100]}", 0
    except requests.exceptions.RequestException as e:
        return f"Request error: {str(e)[:100]}", 0
    except FileNotFoundError as e:
        return f"File error: {str(e)[:100]}", 0
    except PermissionError as e:
        return f"Permission error: {str(e)[:100]}", 0
    except Exception as e:
        return f"Unknown error: {str(e)[:100]}", 0

def main():
    start = time.time()
    
    # Kiểm tra trước khi chạy
    print("🔍 Kiểm tra trước khi chạy test:")
    print(f"📁 File path: {IMAGE_PATH}")
    print(f"📁 File exists: {os.path.exists(IMAGE_PATH)}")
    print(f"🌐 API URL: {API_URL}")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(send_request) for _ in range(NUM_REQUESTS)]
        results = []
        for future in as_completed(futures):
            results.append(future.result())

    # Phân tích chi tiết kết quả
    success = sum(1 for r in results if isinstance(r[0], int) and r[0] == 200)
    failed = NUM_REQUESTS - success
    
    # Thống kê các loại lỗi
    error_types = {}
    for result in results:
        if not (isinstance(result[0], int) and result[0] == 200):
            error_msg = str(result[0])
            error_type = error_msg.split(':')[0] if ':' in error_msg else error_msg
            error_types[error_type] = error_types.get(error_type, 0) + 1
    
    avg_time = sum(r[1] for r in results if isinstance(r[1], float) and r[1] > 0) / max(success, 1)

    print(f"\n✅ Tổng số request: {NUM_REQUESTS}")
    print(f"🟢 Thành công: {success}")
    print(f"🔴 Thất bại: {failed}")
    print(f"⏱️ Thời gian xử lý trung bình: {avg_time:.3f}s")
    
    if error_types:
        print(f"\n📊 Chi tiết các loại lỗi:")
        for error_type, count in error_types.items():
            print(f"   {error_type}: {count} lần")
    
    # In một vài ví dụ lỗi chi tiết
    if failed > 0:
        print(f"\n🔍 Ví dụ lỗi chi tiết:")
        error_examples = [r for r in results if not (isinstance(r[0], int) and r[0] == 200)][:5]
        for i, error in enumerate(error_examples, 1):
            print(f"   {i}. {error[0]}")

if __name__ == "__main__":
    main()
