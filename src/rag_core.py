from . import config
from .models import DeepSeekAPI
from .document_processor import DocumentVectorStore

class RAGEngine:
    def __init__(self):
        self.doc_store = DocumentVectorStore("documents")
        self.llm = DeepSeekAPI()
    
    def answer_question(self, question: str, top_k=3) -> str:
        """检索文档并生成回答"""
        # 1. 检索相关文档块
        results = self.doc_store.search(question, top_k=top_k)
        if not results:
            context = "没有找到相关文档。"
        else:
            # 拼接上下文
            context = "\n\n".join([f"[来源:{r['source']}]\n{r['text']}" for r in results])
        
        # 2. 构造prompt
        system_prompt = "你是一个基于用户提供的文档进行问答的助手。请根据以下文档内容回答问题，如果文档中没有相关信息，请说明未找到。"
        user_prompt = f"文档内容：\n{context}\n\n问题：{question}\n\n请回答："
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # 3. 调用LLM
        answer = self.llm.chat(messages)
        return answer