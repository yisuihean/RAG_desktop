import json
import re
from . import config
from .models import DeepSeekAPI
from .document_processor import DocumentVectorStore

class RAGEngine:
    def __init__(self):
        self.doc_store = DocumentVectorStore("documents")
        self.llm = DeepSeekAPI()
        # 延迟导入避免循环依赖
        self._behavior_lib = None
        self._memory = None
    
    @property
    def behavior_lib(self):
        """延迟初始化行为库"""
        if self._behavior_lib is None:
            from .automation import BehaviorLibrary
            self._behavior_lib = BehaviorLibrary()
        return self._behavior_lib
    
    @property
    def memory(self):
        """延迟初始化记忆"""
        if self._memory is None:
            from .memory import ConversationMemory
            self._memory = ConversationMemory()
        return self._memory
    
    def detect_behavior_intent(self, question: str):
        """
        检测用户是否有执行自动化行为的意图（备选方案，当前主流程已由模型直接调用替代）
        返回: (is_behavior_request, behavior_query, confidence)
        """
        # 关键词匹配（快速判断）
        behavior_keywords = [
            "执行", "运行", "播放", "回放", "开始", "启动",
            "帮我", "请帮我", "给我", "帮我做",
            "点击", "打开", "关闭", "输入", "填写"
        ]
        
        # 检查是否包含行为关键词
        has_keyword = any(kw in question for kw in behavior_keywords)
        
        # 检查是否是询问类问题（排除）
        question_patterns = ["什么是", "为什么", "怎么", "如何", "吗？", "？", "?"]
        is_question = any(p in question for p in question_patterns)
        
        # 简单启发式：有关键词且不是纯问题，可能是行为请求
        if has_keyword and not is_question:
            # 提取可能的查询词（去掉常见动词）
            query = question
            for kw in ["帮我", "请帮我", "给我", "执行", "运行", "播放", "开始", "启动"]:
                query = query.replace(kw, "")
            query = query.strip()
            return True, query, 0.7
        
        return False, None, 0.0
    
    def execute_behavior(self, query: str):
        """
        执行匹配的行为
        返回: (success, message)
        """
        results = self.behavior_lib.search_behavior(query, top_k=1)
        
        if not results:
            return False, f"未找到与 '{query}' 匹配的行为"
        
        best = results[0]
        score = best["score"]
        
        if score < config.BEHAVIOR_SIMILARITY_THRESHOLD:
            return False, f"找到行为 '{best['name']}'，但相似度 {score:.2f} 低于阈值 {config.BEHAVIOR_SIMILARITY_THRESHOLD}"
        
        # 执行行为
        from .automation import replay_mouse
        try:
            replay_mouse(best["file_path"])
            return True, f"已执行行为 '{best['name']}'（相似度: {score:.2f}）"
        except Exception as e:
            return False, f"执行行为失败: {e}"
    
    def answer_question(self, question: str, top_k=3, memory_top_k=3) -> str:
        """
        回答问题，支持模型直接调用自动化脚本（通过 function calling）
        """
        # 1. 检索相关文档
        results = self.doc_store.search(question, top_k=top_k)
        context = "\n\n".join([f"[来源:{r['source']}]\n{r['text']}" for r in results]) if results else "没有找到相关文档。"
        
        # 2. 检索相关历史对话记忆
        memories = self.memory.retrieve_relevant(question, top_k=memory_top_k)
        memory_context = ""
        if memories:
            memory_context = "\n\n## 相关历史对话：\n" + "\n".join([
                f"用户: {m['text'].split('助手:')[0].replace('用户:', '').strip()}\n助手: {m['text'].split('助手:')[1].strip()}"
                for m in memories
            ])
        
        # 3. 构造系统提示和用户消息
        system_prompt = """你是一个智能助手，可以基于文档回答用户问题，也可以执行用户保存的自动化行为（鼠标轨迹回放）。

如果用户要求执行某个操作（例如“帮我打开计算器”、“运行那个登录脚本”），请调用 `execute_behavior` 工具，并传入合适的查询词（如“打开计算器”）。

如果你认为需要执行行为，请使用工具调用；否则直接回答用户问题。"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"文档内容：\n{context}\n\n{memory_context}\n\n当前问题：{question}"}
        ]
        
        # 4. 定义可用工具
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "execute_behavior",
                    "description": "执行一个已保存的鼠标自动化行为",
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
        
        # 5. 调用 DeepSeek API（支持工具调用）
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
        
        # 6. 处理工具调用
        if message.tool_calls:
            # 假设只有一个工具调用
            tool_call = message.tool_calls[0]
            if tool_call.function.name == "execute_behavior":
                # 解析参数
                try:
                    args = json.loads(tool_call.function.arguments)
                    query = args.get("query", "")
                except:
                    query = ""
                
                # 执行行为
                success, result_msg = self.execute_behavior(query)
                
                # 将工具执行结果作为新消息追加
                messages.append(message)  # 助手的工具调用请求
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_msg
                })
                
                # 再次调用模型生成最终回复
                try:
                    second_response = self.llm.client.chat.completions.create(
                        model=config.DEEPSEEK_MODEL,
                        messages=messages,
                        temperature=0.7,
                        max_tokens=2000
                    )
                    final_answer = second_response.choices[0].message.content
                except Exception as e:
                    final_answer = f"执行行为后生成回复失败: {e}"
                
                return final_answer
            else:
                # 未知工具，回退到普通回答
                return message.content or "模型试图调用未知工具。"
        else:
            # 无工具调用，直接返回回答
            return message.content or "（无回答）"
    
    def answer_with_behavior_support(self, question: str) -> dict:
        """
        增强版回答，返回结构化结果（当前已由 answer_question 实现，此方法可保留或弃用）
        返回: {
            "type": "answer" | "behavior_executed" | "behavior_not_found",
            "content": str,
            "behavior_name": str (optional),
            "similarity": float (optional)
        }
        """
        # 检测意图（使用规则，但主流程已用 function calling，此方法可保留供其他用途）
        is_behavior, behavior_query, confidence = self.detect_behavior_intent(question)
        
        if is_behavior:
            results = self.behavior_lib.search_behavior(behavior_query, top_k=1)
            
            if results and results[0]["score"] >= config.BEHAVIOR_SIMILARITY_THRESHOLD:
                best = results[0]
                from .automation import replay_mouse
                try:
                    replay_mouse(best["file_path"])
                    return {
                        "type": "behavior_executed",
                        "content": f"已执行行为 '{best['name']}'",
                        "behavior_name": best["name"],
                        "similarity": best["score"]
                    }
                except Exception as e:
                    return {
                        "type": "behavior_not_found",
                        "content": f"执行失败: {e}"
                    }
            else:
                return {
                    "type": "behavior_not_found",
                    "content": f"未找到匹配的行为（查询: {behavior_query}）"
                }
        
        # 普通问答
        answer = self.answer_question(question)
        return {
            "type": "answer",
            "content": answer
        }