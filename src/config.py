# src/config.py
import os
from pathlib import Path
from .config_manager import get_api_key   # 新增导入

BASE_DIR = Path(__file__).parent.parent
HF_MIRROR = "https://hf-mirror.com"
if HF_MIRROR:
    os.environ['HF_ENDPOINT'] = HF_MIRROR
    print(f"已设置 HuggingFace 镜像: {HF_MIRROR}")

EMBED_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
LOCAL_EMBED_MODEL_PATH = ""

VECTOR_STORE_DIR = BASE_DIR / "data" / "vector_store"
DOCUMENTS_DIR = BASE_DIR / "data" / "documents"
RECORDINGS_DIR = BASE_DIR / "data" / "recordings"

for d in [VECTOR_STORE_DIR, DOCUMENTS_DIR, RECORDINGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# DeepSeek API - 从配置文件读取，不再硬编码
DEEPSEEK_API_KEY = get_api_key()          # 如果未设置，返回空字符串
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"

MEMORY_TOP_K = 3
BEHAVIOR_SIMILARITY_THRESHOLD = 0.75
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50