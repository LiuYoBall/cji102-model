import urllib.request
import json
import uuid

# --- 設定測試參數 ---
# 請確保這張圖片真的存在於您的 GCS Bucket 中，且 API 有權限讀取
# 範例格式: "gs://bucket-name/folder/image.jpg"
TEST_GCS_PATH = "gs://fundus-ai-project/test/1251_right.jpg" 
API_URL = "http://localhost:8080/predict/cfp"

def test_json_api():
    # 1. 建構符合 Pydantic 定義的 JSON Payload
    payload = {
        "image_gcs_path": TEST_GCS_PATH,
        # request_id 是選填的，但加上去方便追蹤 Log
        "request_id": f"test_{str(uuid.uuid4())[:8]}"
    }

    print(f"🚀 Sending POST request to {API_URL}")
    print(f"📦 Payload: {json.dumps(payload, indent=2)}")

    # 2. 轉換為 JSON Bytes
    json_data = json.dumps(payload).encode('utf-8')

    # 3. 設定 Headers (關鍵：告訴 API 這是 JSON)
    headers = {
        'Content-Type': 'application/json; charset=utf-8',
        'Content-Length': str(len(json_data))
    }

    # 4. 發送請求
    req = urllib.request.Request(API_URL, data=json_data, headers=headers, method='POST')

    try:
        with urllib.request.urlopen(req) as response:
            print(f"✅ Status Code: {response.getcode()}")
            
            response_body = response.read().decode('utf-8')
            response_json = json.loads(response_body)
            
            print("📄 Response JSON:")
            print(json.dumps(response_json, indent=2, ensure_ascii=False))

            # 簡單驗證
            if "images" in response_json and "analysis" in response_json:
                print("\n🎉 Test Passed: Received analysis and image URLs.")
            else:
                print("\n⚠️ Test Warning: Missing expected keys.")

    except urllib.error.HTTPError as e:
        print(f"❌ HTTP Error: {e.code} {e.reason}")
        error_content = e.read().decode('utf-8')
        print(f"Details: {error_content}")
        
    except Exception as e:
        print(f"❌ Request Failed: {e}")

if __name__ == "__main__":
    test_json_api()