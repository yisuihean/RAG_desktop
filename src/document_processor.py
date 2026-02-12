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
            # 使用模型的实际维度
            self.index = faiss.IndexFlatIP(self.emb_model.dimension)
            self.metadata = []
    
    def save(self):
        faiss.write_index(self.index, str(self.index_path))
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.metadata, f)
    
    def add_documents(self, texts: List[str], sources: List[str]):
        if not texts:
            return
        vectors = self.emb_model.encode(texts)
        self.index.add(vectors)
        for text, src in zip(texts, sources):
            self.metadata.append({"text": text, "source": src})
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

def split_text(text: str, chunk_size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks