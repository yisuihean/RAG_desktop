from . import config
from .models import DeepSeekAPI
from .document_processor import DocumentVectorStore

class RAGEngine:
    def __init__(self):
        self.doc_store = DocumentVectorStore("documents")
        self.llm = DeepSeekAPI()
    
    def answer_question(self, question: str, top_k=3, memory_top_k=3) -> str:
    # 1. 检索相关文档
        results = self.doc_store.search(question, top_k=top_k)
        context = "\n\n".join([f"[来源:{r['source']}]\n{r['text']}" for r in results]) if results else "没有找到相关文档。"

    # 2. 检索相关历史对话记忆（新增）
        from .memory import ConversationMemory  # 避免循环导入
        memory = ConversationMemory()
        memories = memory.retrieve_relevant(question, top_k=memory_top_k)
        memory_context = ""
        if memories:
            memory_context = "\n\n## 相关历史对话：\n" + "\n".join([f"用户: {m['text'].split('助手:')[0].replace('用户:', '').strip()}\n助手: {m['text'].split('助手:')[1].strip()}" for m in memories])

    # 3. 构造 prompt
        system_prompt = "你是一个基于用户文档和历史对话进行问答的助手。请根据以下文档内容回答，并参考相关历史对话保持上下文连贯。"
        user_prompt = f"文档内容：\n{context}\n\n{memory_context}\n\n当前问题：{question}\n\n请回答："
    
        messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
        answer = self.llm.chat(messages)
        return answer