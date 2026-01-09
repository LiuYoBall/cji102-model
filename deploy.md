# 部署流程與 CI/CD 設定

## 1. 服務架構
- CI/CD 平台: GitHub Actions
- Image Registry: Docker Hub (Public)
- 部署環境: Google Cloud Run

## 2. GitHub Secrets 設定
請在 Repo Settings > Secrets and variables > Actions 新增：
- `DOCKER_USERNAME`: (您的帳號)
- `DOCKER_PASSWORD`: (您的密碼)
- `GCP_SA_KEY`: (從 GCP 下載的 JSON Key)

## 3. 自動化流程
- Push 到 `main` 分支 -> 自動 Build Docker Image -> Push 到 Docker Hub -> 自動更新 Cloud Run。