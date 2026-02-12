import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
HF_MIRROR = "https://hf-mirror.com"   # 设置为 None 则不走镜像
if HF_MIRROR:
    os.environ['HF_ENDPOINT'] = HF_MIRROR
    print(f"已设置 HuggingFace 镜像: {HF_MIRROR}")

# 嵌入模型配置
EMBED_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
# 本地模型路径——设置为空字符串，transformers 会自动从 HF 缓存加载（若已下载）
LOCAL_EMBED_MODEL_PATH = ""   # 或者填入具体的 snapshot 路径，如 "/home/.../snapshots/xxx"

VECTOR_STORE_DIR = BASE_DIR / "data" / "vector_store"
DOCUMENTS_DIR = BASE_DIR / "data" / "documents"
RECORDINGS_DIR = BASE_DIR / "data" / "recordings"

for d in [VECTOR_STORE_DIR, DOCUMENTS_DIR, RECORDINGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# DeepSeek API
DEEPSEEK_API_KEY = "sk-76709f31313e484c9280da66d697fb4b"   # 请填入你的真实 Key
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"

MEMORY_TOP_K = 3
BEHAVIOR_SIMILARITY_THRESHOLD = 0.75
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50