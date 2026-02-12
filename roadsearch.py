# find_model_location.py
import os
from pathlib import Path
from transformers import pipeline
import huggingface_hub

print("=== 查找模型位置 ===\n")

# 方法1: 查看 huggingface 缓存目录
hf_cache = os.path.expanduser("~/.cache/huggingface")
print(f"1. HuggingFace 缓存目录: {hf_cache}")
print(f"   是否存在: {'✅' if os.path.exists(hf_cache) else '❌'}")

# 方法2: 查看 hub 目录
hub_dir = os.path.join(hf_cache, "hub")
if os.path.exists(hub_dir):
    print(f"\n2. Hub 目录: {hub_dir}")
    print("   模型列表:")
    for item in os.listdir(hub_dir):
        if "models--" in item and "Qwen" in item:
            model_path = os.path.join(hub_dir, item)
            print(f"   📁 {item}")
            # 显示模型文件
            if os.path.exists(model_path):
                files = os.listdir(model_path)
                print(f"      包含 {len(files)} 个文件/目录")
                for f in files[:5]:  # 只显示前5个
                    print(f"        - {f}")

# 方法3: 通过环境变量
print(f"\n3. 环境变量:")
print(f"   HF_HOME: {os.environ.get('HF_HOME', '未设置')}")
print(f"   TRANSFORMERS_CACHE: {os.environ.get('TRANSFORMERS_CACHE', '未设置')}")
print(f"   HF_ENDPOINT: {os.environ.get('HF_ENDPOINT', '未设置')}")

# 方法4: 使用 huggingface_hub 获取缓存路径
try:
    cache_info = huggingface_hub.scan_cache_dir()
    print(f"\n4. HuggingFace Hub 缓存信息:")
    print(f"   总大小: {cache_info.size_on_disk / 1024 / 1024:.2f} MB")
    print(f"   模型数量: {len(cache_info.repos)}")
    
    for repo in cache_info.repos:
        if "Qwen" in repo.repo_id:
            print(f"\n   📦 {repo.repo_id}")
            print(f"     路径: {repo.repo_path}")
            print(f"     大小: {repo.size_on_disk / 1024 / 1024:.2f} MB")
            print(f"     最后访问: {repo.last_accessed}")
except Exception as e:
    print(f"   无法扫描缓存: {e}")

# 方法5: 直接通过已加载的模型获取路径
print("\n5. 通过已加载的模型获取路径:")
try:
    print("   加载模型...")
    pipe = pipeline("feature-extraction", model="Qwen/Qwen3-Embedding-0.6B")
    
    # 获取模型对象
    model = pipe.model
    if hasattr(model, 'name_or_path'):
        print(f"   模型名称/路径: {model.name_or_path}")
    
    # 获取配置文件路径
    if hasattr(model, 'config'):
        if hasattr(model.config, '_name_or_path'):
            print(f"   配置文件路径: {model.config._name_or_path}")
    
    print("   ✅ 模型已加载")
except Exception as e:
    print(f"   ❌ 无法加载模型: {e}")