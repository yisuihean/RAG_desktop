import faiss
from . import config
from .document_processor import DocumentVectorStore

class ConversationMemory:
    """对话记忆：支持增删改查（CRUD）"""
    def __init__(self):
        self.vector_store = DocumentVectorStore("conversations")
    
    def add_dialogue(self, question: str, answer: str):
        """自动存储一问一答"""
        text = f"用户: {question}\n助手: {answer}"
        self.vector_store.add_documents([text], ["conversation"])
    
    def add_memory(self, content: str, source="manual"):
        """手动添加一条记忆（纯文本）"""
        self.vector_store.add_documents([content], [source])
    
    def retrieve_relevant(self, query: str, top_k=3):
        """检索相似记忆"""
        return self.vector_store.search(query, top_k=top_k)
    
    def get_all_memories(self, limit=100):
        """返回所有存储的记忆（最新在前）"""
        if not hasattr(self.vector_store, 'metadata'):
            return []
        # 倒序，最新的显示在最上面
        return list(reversed(self.vector_store.metadata[-limit:]))
    
    def delete_memory(self, index):
        """删除指定位置的记忆（基于当前列表顺序）"""
        if index < 0 or index >= len(self.vector_store.metadata):
            return False
        
        # 移除 metadata 中对应项
        del self.vector_store.metadata[index]
        
        # 重建 FAISS 索引
        if self.vector_store.metadata:
            texts = [item['text'] for item in self.vector_store.metadata]
            sources = [item['source'] for item in self.vector_store.metadata]
            vectors = self.vector_store.emb_model.encode(texts)
            self.vector_store.index = faiss.IndexFlatIP(self.vector_store.emb_model.dimension)
            self.vector_store.index.add(vectors)
        else:
            self.vector_store.index = faiss.IndexFlatIP(self.vector_store.emb_model.dimension)
        
        self.vector_store.save()
        return True
    
    def clear_all(self):
        """清空所有记忆（慎用）"""
        self.vector_store.index = faiss.IndexFlatIP(self.vector_store.emb_model.dimension)
        self.vector_store.metadata = []
        self.vector_store.save()