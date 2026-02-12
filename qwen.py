# qwen_mirror.py
import os

print("=== 使用 huggingface 镜像站 ===")

# 1. 清除所有代理设置（避免干扰）
for key in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY']:
    if key in os.environ:
        del os.environ[key]

# 2. 设置 huggingface 镜像站（国内可用）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

print(f"使用镜像站: {os.environ['HF_ENDPOINT']}")

# 3. 测试连接
import requests
try:
    response = requests.get('https://hf-mirror.com', timeout=10)
    print(f"✅ 镜像站连接成功: HTTP {response.status_code}")
except Exception as e:
    print(f"❌ 镜像站连接失败: {e}")
    print("尝试其他镜像站...")

# 4. 如果 hf-mirror.com 不行，尝试其他镜像
mirrors = [
    'https://hf-mirror.com',
    'https://huggingface.co',
    'https://huggingface.co.uk',
]

for mirror in mirrors:
    print(f"\n尝试镜像: {mirror}")
    os.environ['HF_ENDPOINT'] = mirror
    try:
        from transformers import pipeline
        print("开始加载模型...")
        pipe = pipeline("feature-extraction", model="Qwen/Qwen3-Embedding-0.6B")
        print(f"✅ 使用 {mirror} 加载成功！")
        break
    except Exception as e:
        print(f"❌ {mirror} 失败: {type(e).__name__}")