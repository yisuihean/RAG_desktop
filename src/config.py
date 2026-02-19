# src/config.py
import os
from pathlib import Path
from .config_manager import get_api_key

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

DEEPSEEK_API_KEY = get_api_key()
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"

MEMORY_TOP_K = 3
BEHAVIOR_SIMILARITY_THRESHOLD = 0.6
BEHAVIOR_SEARCH_TOP_K = 5
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# 鼠标录制采样间隔（秒）
MOVE_SAMPLE_RATE = 0.03

# 回放点击延迟（秒）——按下后释放的间隔，设为0则使用 pyautogui.click 的默认行为
CLICK_DELAY = 0.05

# 是否在回放点击时使用 pyautogui.click 而不是分离的按下/释放
# True 使用 click，False 使用 mouseDown+mouseUp
USE_CLICK_API = True

# 移动轨迹简化选项
SIMPLIFY_MOVES = True               # 是否简化移动轨迹
MOVE_SIMPLIFY_TOLERANCE = 5.0       # 像素容忍度，大于此值的移动点才保留
MIN_MOVE_DISTANCE = 3.0              # 最小移动距离，小于此值的相邻移动点忽略

# 调试模式
DEBUG = True