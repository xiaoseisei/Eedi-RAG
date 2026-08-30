import os
import urllib.request
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# 优先使用镜像源加速下载，若失败可自动回退
HOSTS = [
    "https://hf-mirror.com",
    "https://huggingface.co"
]

REPO_ID = "Eedi/Question-Anchored-Tutoring-Dialogues-2k"

FILES_TO_DOWNLOAD = [
    "README.md",
    "dataset_config.yaml",
    "dialogue-subjects.csv",
    "dq-question-metadata.csv",
    "anchored-dialogues/train.csv",
    "anchored-dialogues/test.csv",
    "anchored-dialogues/train-00000-of-00001.parquet",
    "anchored-dialogues/test-00000-of-00001.parquet",
    "dq-question-metadata/train-00000-of-00001.parquet"
]

def download_file(file_rel_path):
    target_path = os.path.join(DATA_DIR, file_rel_path.replace("/", os.sep))
    os.makedirs(os.path.dirname(target_path), exist_ok=True)

    if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
        print(f"[已存在跳过] {file_rel_path} ({os.path.getsize(target_path)} bytes)")
        return True

    for host in HOSTS:
        download_url = f"{host}/datasets/{REPO_ID}/resolve/main/{file_rel_path}"
        print(f"正在下载: {file_rel_path} 从 {host} ...")
        try:
            req = urllib.request.Request(download_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as response, open(target_path, "wb") as out_file:
                total_size = response.info().get("Content-Length")
                downloaded = 0
                chunk_size = 1024 * 64
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    out_file.write(chunk)
                    downloaded += len(chunk)
            print(f"-> 成功: {file_rel_path} ({downloaded} bytes)")
            return True
        except Exception as e:
            print(f"-> 下载失败 ({host}): {e}")
            if os.path.exists(target_path):
                try:
                    os.remove(target_path)
                except OSError:
                    pass
            time.sleep(1)

    return False

def main():
    print(f"=== 开始下载 Eedi 数据集到: {DATA_DIR} ===")
    success_count = 0
    for f in FILES_TO_DOWNLOAD:
        if download_file(f):
            success_count += 1
    
    print(f"\n=== 下载完成: {success_count}/{len(FILES_TO_DOWNLOAD)} 个文件 ===")

if __name__ == "__main__":
    main()
