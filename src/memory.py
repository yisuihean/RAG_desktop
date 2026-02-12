from .document_processor import DocumentVectorStore

class ConversationMemory:
    """存储历史对话向量，支持检索"""
    def __init__(self):
        self.vector_store = DocumentVectorStore("conversations")
    
    def add_dialogue(self, question: str, answer: str):
        """将一问一答合并为一个文本块存储"""
        text = f"用户: {question}\n助手: {answer}"
        self.vector_store.add_documents([text], ["conversation"])
    
    def retrieve_relevant(self, query: str, top_k=3):
        """检索相似的历史对话"""
        return self.vector_store.search(query, top_k=top_k)