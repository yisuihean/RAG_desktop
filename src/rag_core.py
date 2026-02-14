import json
from . import config
from .models import DeepSeekAPI
from .document_processor import DocumentVectorStore

class RAGEngine:
    def __init__(self):
        self.doc_store = DocumentVectorStore("documents")
        self.llm = DeepSeekAPI()
        self._behavior_lib = None
        self._memory = None

    @property
    def behavior_lib(self):
        if self._behavior_lib is None:
            from .automation import BehaviorLibrary
            self._behavior_lib = BehaviorLibrary()
        return self._behavior_lib

    @property
    def memory(self):
        if self._memory is None:
            from .memory import ConversationMemory
            self._memory = ConversationMemory()
        return self._memory

    def execute_behavior(self, query: str):
        """执行匹配的行为，返回 (success, message, behavior_name?)"""
        results = self.behavior_lib.search_behavior(query, top_k=1)
        if not results:
            return False, f"没有找到与“{query}”相关的自动化脚本"

        best = results[0]
        score = best["score"]
        if score < config.BEHAVIOR_SIMILARITY_THRESHOLD:
            return False, f"找到脚本“{best['name']}”，但相似度 {score:.2f} 低于阈值 {config.BEHAVIOR_SIMILARITY_THRESHOLD}，无法执行"

        from .automation import replay_mouse
        try:
            replay_mouse(best["file_path"])
            return True, f"已执行脚本“{best['name']}”", best['name']
        except Exception as e:
            return False, f"执行脚本失败: {e}"

    def answer_question(self, question: str, top_k=3, memory_top_k=3) -> str:
        # 检索文档
        doc_results = self.doc_store.search(question, top_k=top_k)
        context = "\n\n".join([f"[来源:{r['source']}]\n{r['text']}" for r in doc_results]) if doc_results else "没有找到相关文档。"

        # 检索记忆
        memories = self.memory.retrieve_relevant(question, top_k=memory_top_k)
        memory_context = ""
        if memories:
            memory_context = "\n\n## 相关历史对话：\n" + "\n".join([
                f"用户: {m['text'].split('助手:')[0].replace('用户:', '').strip()}\n助手: {m['text'].split('助手:')[1].strip()}"
                for m in memories
            ])

        # 系统提示词，明确告诉模型何时调用工具
        system_prompt = """你是一个智能助手，可以基于文档回答用户问题，也可以执行用户保存的自动化脚本（鼠标轨迹回放）。

当用户明确要求执行某个操作时（例如“帮我打开计算器”、“运行那个登录脚本”、“执行刚才录制的打开浏览器的操作”），你必须调用 `execute_behavior` 工具，并传入**准确的查询词**（尽量使用用户原话中的关键词，比如“打开计算器”）。

如果用户没有明确要求执行脚本，或者问题与文档相关，则不要调用工具，直接回答用户问题。

调用工具后，工具会返回执行结果，你需要根据结果向用户做出最终回复。"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"文档内容：\n{context}\n\n{memory_context}\n\n当前问题：{question}"}
        ]

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "execute_behavior",
                    "description": "执行一个已保存的自动化脚本（鼠标轨迹）",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "行为名称或关键词，例如'打开计算器'、'登录流程'"
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        ]

        try:
            response = self.llm.client.chat.completions.create(
                model=config.DEEPSEEK_MODEL,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.7,
                max_tokens=2000
            )
        except Exception as e:
            print(f"API 调用失败: {e}")
            return f"错误: {e}"

        message = response.choices[0].message

        if message.tool_calls:
            # 处理工具调用
            tool_call = message.tool_calls[0]
            if tool_call.function.name == "execute_behavior":
                try:
                    args = json.loads(tool_call.function.arguments)
                    query = args.get("query", "")
                except:
                    query = ""

                success, result_msg, *extra = self.execute_behavior(query)

                # 将工具结果追加到对话中
                messages.append(message)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_msg
                })

                try:
                    second_response = self.llm.client.chat.completions.create(
                        model=config.DEEPSEEK_MODEL,
                        messages=messages,
                        temperature=0.7,
                        max_tokens=2000
                    )
                    return second_response.choices[0].message.content
                except Exception as e:
                    return f"执行脚本后生成回复失败: {e}"
            else:
                return message.content or "模型试图调用未知工具。"
        else:
            return message.content or "（无回答）"