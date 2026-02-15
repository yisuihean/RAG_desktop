import os
import pickle
from pathlib import Path
import faiss
import numpy as np
from typing import List, Dict
from . import config
from .models import EmbeddingModel

class DocumentVectorStore:
    def __init__(self, store_name="documents"):
        self.store_name = store_name
        self.index_path = config.VECTOR_STORE_DIR / f"{store_name}.faiss"
        self.meta_path = config.VECTOR_STORE_DIR / f"{store_name}.pkl"
        self.emb_model = EmbeddingModel()
        
        self.index = None
        self.metadata = []
        self._load_or_create()
    
    def _load_or_create(self):
        if self.index_path.exists() and self.meta_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            with open(self.meta_path, "rb") as f:
                self.metadata = pickle.load(f)
        else:
            self.index = faiss.IndexFlatIP(self.emb_model.dimension)
            self.metadata = []
    
    def save(self):
        faiss.write_index(self.index, str(self.index_path))
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.metadata, f)
    
    def add_documents(self, texts: List[str], sources: List[str], batch_size=100):
        """分批添加文档到向量库"""
        if not texts:
            return
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_sources = sources[i:i+batch_size]
            
            vectors = self.emb_model.encode(batch_texts)
            self.index.add(vectors)
            
            for text, src in zip(batch_texts, batch_sources):
                self.metadata.append({"text": text, "source": src})
            
            print(f"已添加 {i+len(batch_texts)}/{len(texts)} 个块")
            if hasattr(self.emb_model, 'device') and self.emb_model.device == 'cuda':
                import torch
                torch.cuda.empty_cache()
        
        self.save()
    
    def search(self, query: str, top_k=5) -> List[Dict]:
        if len(self.metadata) == 0:
            return []
        
        query_vec = self.emb_model.encode(query).reshape(1, -1)
        scores, indices = self.index.search(query_vec, min(top_k, len(self.metadata)))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and idx < len(self.metadata):
                results.append({
                    "text": self.metadata[idx]["text"],
                    "source": self.metadata[idx]["source"],
                    "score": float(score)
                })
        return results


def parse_document(file_path: str) -> str:
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        try:
            import pypdf
        except ImportError:
            raise ImportError("请安装 pypdf: pip install pypdf")
        reader = pypdf.PdfReader(file_path)
        text = "\n".join([page.extract_text() or "" for page in reader.pages])
    elif ext == ".md":
        try:
            import markdown
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError("请安装 markdown beautifulsoup4: pip install markdown beautifulsoup4")
        html = markdown.markdown(open(file_path, 'r', encoding='utf-8').read())
        soup = BeautifulSoup(html, 'html.parser')
        text = soup.get_text()
    elif ext == ".txt":
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        raise ValueError(f"不支持的文件格式: {ext}")
    return text


def split_text(text: str, chunk_size=None, overlap=None) -> List[str]:
    """
    将文本分割成小块，防止内存溢出和无限循环
    """
    if chunk_size is None:
        chunk_size = config.CHUNK_SIZE
    if overlap is None:
        overlap = config.CHUNK_OVERLAP

    # 安全性检查
    if overlap >= chunk_size:
        print(f"警告: overlap ({overlap}) >= chunk_size ({chunk_size})，已自动调整 overlap 为 chunk_size//2")
        overlap = chunk_size // 2

    # 超大文件自动调小 chunk_size
    if len(text) > 10 * 1024 * 1024:  # >10MB
        print("警告：文件较大，正在优化处理...")
        chunk_size = 200
        overlap = 20

    chunks = []
    start = 0
    total_len = len(text)
    # 防止无限循环的最大块数（安全保护）
    max_chunks = total_len // (chunk_size - overlap) + 1000
    loop_count = 0

    while start < total_len and loop_count < max_chunks:
        loop_count += 1
        end = min(start + chunk_size, total_len)

        # 可选：在句子边界处切割（此处注释掉以提高性能）
        # 如果你需要语义完整性，可以取消下面的注释
        # if end < total_len:
        #     for i in range(end - 1, start, -1):
        #         if text[i] in '.。!！?？\n':
        #             end = i + 1
        #             break

        chunks.append(text[start:end])

        # 计算下一个起始位置
        next_start = end - overlap
        if next_start <= start:          # 防止停滞（例如 overlap 为0）
            next_start = end
        start = next_start

        if len(chunks) % 100 == 0:
            print(f"已处理 {len(chunks)} 个文本块...")

    if loop_count >= max_chunks:
        print(f"警告: 分块处理可能进入无限循环，已强制停止，生成了 {len(chunks)} 个块")

    print(f"文件分割完成，共 {len(chunks)} 个块")
    return chunks