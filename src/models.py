import os
import torch
from pathlib import Path
from transformers import AutoModel, AutoTokenizer
import openai
import numpy as np
from . import config

# 确保镜像设置在导入模型前已生效
if hasattr(config, 'HF_MIRROR') and config.HF_MIRROR:
    os.environ['HF_ENDPOINT'] = config.HF_MIRROR

class EmbeddingModel:
    """单例模式 - 使用 transformers 直接加载 Qwen 嵌入模型"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_model()
        return cls._instance
    
    def _load_model(self):
        """加载模型：优先本地路径（若配置），否则从 HuggingFace 镜像下载"""
        model_source = config.LOCAL_EMBED_MODEL_PATH
        if model_source and Path(model_source).exists():
            print(f"从本地路径加载模型: {model_source}")
        else:
            model_source = config.EMBED_MODEL_NAME
            print(f"从 HuggingFace 加载模型: {model_source} (镜像: {getattr(config, 'HF_MIRROR', '官方')})")
        
        # 加载 tokenizer 和 model（Qwen 需要 trust_remote_code=True）
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_source, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            model_source, trust_remote_code=True
        )
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()
        
        # 获取维度
        with torch.no_grad():
            dummy = self.tokenizer("test", return_tensors="pt").to(self.device)
            output = self.model(**dummy)
            self.dimension = output.last_hidden_state.shape[-1]
        print(f"嵌入模型加载成功，维度: {self.dimension}")
    
    def encode(self, texts):
        """将文本列表转为归一化的向量，返回 numpy 数组 (n, dim)"""
        if isinstance(texts, str):
            texts = [texts]
        
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            token_embeddings = outputs.last_hidden_state
            attention_mask = inputs['attention_mask']
            mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sentence_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1) / \
                                  torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        
        return sentence_embeddings.cpu().numpy()

class DeepSeekAPI:
    """DeepSeek V3 API 调用"""
    def __init__(self, api_key=None):
        self.api_key = api_key or config.DEEPSEEK_API_KEY
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=config.DEEPSEEK_BASE_URL
        )
    
    def chat(self, messages, temperature=0.7, max_tokens=2000):
        try:
            response = self.client.chat.completions.create(
                model=config.DEEPSEEK_MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"API 调用失败: {e}")
            return f"错误: {e}"