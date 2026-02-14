# src/config_manager.py
import json
from pathlib import Path

CONFIG_DIR = Path.home() / ".rag_assistant"
CONFIG_FILE = CONFIG_DIR / "config.json"

def load_config():
    """加载配置文件，返回配置字典"""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_config(config):
    """保存配置到文件"""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)

def get_api_key():
    """获取保存的 API key"""
    return load_config().get("deepseek_api_key", "")

def set_api_key(key):
    """设置 API key 并保存"""
    config = load_config()
    config["deepseek_api_key"] = key
    save_config(config)