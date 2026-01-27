import urllib.request
import urllib.error
import json
import uuid
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- 設定測試參數 ---
API_URL = "https://cji102-model-test-133954051088.asia-east1.run.app/predict/cfp"

# 請在這裡填入你要測試的所有 GCS 圖片路徑
TEST_IMAGE_LIST = [
    "gs://fundus-ai-project/test/32_left.jpg",
    "gs://fundus-ai-project/test/62_right.jpg",
    "gs://fundus-ai-project/test/68_left.jpg",
    "gs://fundus-ai-project/test/82_left.jpg",
    "gs://fundus-ai-project/test/1251_right.jpg",
    "gs://fundus-ai-project/test/IDRiD_001.jpg",
    "gs://fundus-ai-project/test/IDRiD_077.jpg",
    "gs://fundus-ai-project/test/js2973836-fig-0001c-m.jpg",
    "gs://fundus-ai-project/test/nogood.jpg",
]

# 設定同時發送的請求數量 (建議 3-5，避免對 Server 造成過大壓力)
MAX_WORKERS = 3

def send_single_request(image_path):
    """
    發送單一圖片請求，並回傳完整的資料結構以便存檔
    """
    request_id = f"batch_{str(uuid.uuid4())[:8]}"
    
    # 1. 建構 Payload
    payload = {
        "image_gcs_path": image_path,
        "request_id": request_id
    }
    
    json_data = json.dumps(payload).encode('utf-8')
    headers = {
        'Content-Type': 'application/json; charset=utf-8',
        'Content-Length': str(len(json_data))
    }

    req = urllib.request.Request(API_URL, data=json_data, headers=headers, method='POST')

    start_time = time.time()
    
    # 初始化結果物件
    result_data = {
        "image_path": image_path,
        "request_id": request_id,
        "success": False,
        "status_code": None,
        "response": None,   # 這裡將會儲存 API 回傳的完整內容
        "error_message": None,
        "elapsed_time": 0
    }

    try:
        # 設定 Timeout 為 300 秒 (5分鐘)
        with urllib.request.urlopen(req, timeout=300) as response:
            elapsed = time.time() - start_time
            response_body = response.read().decode('utf-8')
            
            # 解析 API 回傳的 JSON
            response_json = json.loads(response_body)
            
            # 填寫成功資訊
            result_data["success"] = True
            result_data["status_code"] = response.getcode()
            result_data["response"] = response_json
            result_data["elapsed_time"] = round(elapsed, 2)
            
            return result_data

    except urllib.error.HTTPError as e:
        result_data["status_code"] = e.code
        error_content = e.read().decode('utf-8')
        # 嘗試解析錯誤訊息中的 detail
        try:
            err_json = json.loads(error_content)
            result_data["error_message"] = err_json.get("detail", error_content)
        except:
            result_data["error_message"] = error_content
        return result_data
        
    except Exception as e:
        result_data["error_message"] = str(e)
        return result_data

def run_batch_test():
    print(f"[INFO] 開始批量測試，共有 {len(TEST_IMAGE_LIST)} 張圖片...")
    print(f"[INFO] Target API: {API_URL}")
    print(f"[INFO] Parallel Workers: {MAX_WORKERS}")
    print("-" * 60)
    
    full_report = []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_url = {executor.submit(send_single_request, url): url for url in TEST_IMAGE_LIST}
        
        for future in as_completed(future_to_url):
            try:
                data = future.result()
                full_report.append(data)
                
                # 在 Console 顯示簡短狀態，詳細內容存檔
                status_mark = "[OK]" if data["success"] else "[FAIL]"
                file_name = data['image_path'].split('/')[-1]
                print(f"{status_mark} {file_name} ({data.get('elapsed_time', 0)}s)")
                
                if not data["success"]:
                    print(f"       Reason: {data.get('error_message')}")

            except Exception as exc:
                print(f"[ERROR] System Exception: {exc}")

    # --- 統計與存檔 ---
    print("-" * 60)
    success_count = sum(1 for item in full_report if item["success"])
    fail_count = len(full_report) - success_count
    
    print(f"測試完成。成功: {success_count}, 失敗: {fail_count}")

    # 將結果寫入 JSON 檔案
    output_filename = "api_test_results.json"
    try:
        with open(output_filename, "w", encoding="utf-8") as f:
            # ensure_ascii=False 確保中文字能正常顯示，indent=2 讓格式易讀
            json.dump(full_report, f, indent=2, ensure_ascii=False)
        print(f"[FILE] 詳細結果已儲存至: {output_filename}")
        print(f"       請打開此檔案查看完整的 API 回傳內容 (Analysis, Image URLs 等)。")
    except Exception as e:
        print(f"[ERROR] 存檔失敗: {e}")

if __name__ == "__main__":
    if not TEST_IMAGE_LIST:
        print("[WARN] 請先在 TEST_IMAGE_LIST 變數中加入圖片路徑")
    else:
        run_batch_test()


