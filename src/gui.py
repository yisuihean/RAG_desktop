import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
from pathlib import Path
from . import config
from .rag_core import RAGEngine
from .document_processor import parse_document, split_text
from .automation import MouseRecorder, BehaviorLibrary, replay_mouse
from .memory import ConversationMemory
from .models import DeepSeekAPI, EmbeddingModel
import tkinter.font as tkfont

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("RAG桌面助手 - 红岩网校考核")
        self.root.geometry("900x700")
        
        # 初始化核心组件
        self.rag = RAGEngine()
        self.recorder = MouseRecorder()
        self.behavior_lib = BehaviorLibrary()
        self.memory = ConversationMemory()
        self.embed_model = EmbeddingModel()
        
        # 创建标签页
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True)
        
        self.create_rag_tab()
        self.create_automation_tab()
        self.create_memory_tab()
        
        # 菜单栏（设置API Key）
        menubar = tk.Menu(root)
        root.config(menu=menubar)
        settings_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="设置", menu=settings_menu)
        settings_menu.add_command(label="配置API密钥", command=self.set_api_key)
    
    # ---------- 文档问答标签页 ----------
    def create_rag_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="文档问答")
        
        # 上传文件区域
        frame_upload = ttk.LabelFrame(tab, text="上传文档")
        frame_upload.pack(fill='x', padx=10, pady=5)
        btn_upload = ttk.Button(frame_upload, text="选择文件", command=self.upload_document)
        btn_upload.pack(side='left', padx=5, pady=5)
        self.lbl_upload_status = ttk.Label(frame_upload, text="未上传文件")
        self.lbl_upload_status.pack(side='left', padx=5)
        
        # 问答区域
        frame_qa = ttk.LabelFrame(tab, text="问答")
        frame_qa.pack(fill='both', expand=True, padx=10, pady=5)
        
        # 问题输入
        ttk.Label(frame_qa, text="问题:").grid(row=0, column=0, sticky='w', padx=5, pady=5)
        self.question_entry = ttk.Entry(frame_qa, width=60)
        self.question_entry.grid(row=0, column=1, padx=5, pady=5, sticky='ew')
        btn_ask = ttk.Button(frame_qa, text="提问", command=self.ask_question)
        btn_ask.grid(row=0, column=2, padx=5, pady=5)
        
        # 答案显示
        ttk.Label(frame_qa, text="回答:").grid(row=1, column=0, sticky='nw', padx=5, pady=5)
        self.answer_text = scrolledtext.ScrolledText(frame_qa, width=70, height=20, wrap=tk.WORD)
        self.answer_text.grid(row=1, column=1, columnspan=2, padx=5, pady=5, sticky='nsew')
        frame_qa.grid_columnconfigure(1, weight=1)
        frame_qa.grid_rowconfigure(1, weight=1)
    
    def upload_document(self):
        file_path = filedialog.askopenfilename(
            title="选择文档",
            filetypes=[("文档", "*.pdf *.md *.txt"), ("PDF", "*.pdf"), ("Markdown", "*.md"), ("文本", "*.txt")]
        )
        if not file_path:
            return
        try:
            # 解析文档
            text = parse_document(file_path)
            chunks = split_text(text)
            sources = [Path(file_path).name] * len(chunks)
            # 添加到向量库
            self.rag.doc_store.add_documents(chunks, sources)
            self.lbl_upload_status.config(text=f"已上传并索引: {Path(file_path).name} ({len(chunks)}块)")
            messagebox.showinfo("成功", "文档已处理并加入知识库")
        except Exception as e:
            messagebox.showerror("错误", f"处理失败: {e}")
    
    def ask_question(self):
        question = self.question_entry.get().strip()
        if not question:
            messagebox.showwarning("提示", "请输入问题")
            return
        # 启用加载提示（此处可加）
        self.answer_text.delete(1.0, tk.END)
        self.answer_text.insert(tk.END, "正在生成答案...")
        
        def task():
            # 检索记忆（可选增强）
            memories = self.memory.retrieve_relevant(question)
            # 目前RAGEngine未集成记忆，可自行扩展，简单起见先只做文档检索
            answer = self.rag.answer_question(question)
            # 存储对话记忆
            self.memory.add_dialogue(question, answer)
            # 更新界面
            self.root.after(0, lambda: self.answer_text.delete(1.0, tk.END))
            self.root.after(0, lambda: self.answer_text.insert(tk.END, answer))
        
        threading.Thread(target=task, daemon=True).start()
    
    # ---------- 行为自动化标签页 ----------
    def create_automation_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="行为自动化")
        
        # 录制控制
        frame_record = ttk.LabelFrame(tab, text="鼠标轨迹录制")
        frame_record.pack(fill='x', padx=10, pady=5)
        self.btn_record = ttk.Button(frame_record, text="开始录制", command=self.toggle_record)
        self.btn_record.pack(side='left', padx=5, pady=5)
        self.record_status = ttk.Label(frame_record, text="未录制")
        self.record_status.pack(side='left', padx=5)
        
        # 保存命名
        frame_save = ttk.Frame(frame_record)
        frame_save.pack(side='left', padx=20)
        ttk.Label(frame_save, text="名称:").pack(side='left')
        self.behavior_name = ttk.Entry(frame_save, width=20)
        self.behavior_name.pack(side='left', padx=5)
        ttk.Label(frame_save, text="描述:").pack(side='left')
        self.behavior_desc = ttk.Entry(frame_save, width=30)
        self.behavior_desc.pack(side='left', padx=5)
        btn_save = ttk.Button(frame_save, text="保存轨迹", command=self.save_behavior)
        btn_save.pack(side='left', padx=5)
        
        # 行为检索与回放
        frame_query = ttk.LabelFrame(tab, text="检索行为并执行")
        frame_query.pack(fill='x', padx=10, pady=5)
        ttk.Label(frame_query, text="关键词:").pack(side='left', padx=5)
        self.behavior_query = ttk.Entry(frame_query, width=40)
        self.behavior_query.pack(side='left', padx=5)
        btn_search = ttk.Button(frame_query, text="检索并回放", command=self.search_and_replay)
        btn_search.pack(side='left', padx=5)
        self.search_result = ttk.Label(frame_query, text="")
        self.search_result.pack(side='left', padx=10)
    
    def toggle_record(self):
        if not self.recorder.recording:
            # 开始录制
            self.recorder.start()
            self.btn_record.config(text="停止录制")
            self.record_status.config(text="正在录制...")
        else:
            # 停止录制
            events = self.recorder.stop()
            self.btn_record.config(text="开始录制")
            self.record_status.config(text=f"录制结束，共{len(events)}个事件")
    
    def save_behavior(self):
        if not hasattr(self.recorder, 'events') or not self.recorder.events:
            messagebox.showwarning("提示", "请先录制轨迹")
            return
        name = self.behavior_name.get().strip()
        desc = self.behavior_desc.get().strip()
        if not name:
            messagebox.showwarning("提示", "请输入行为名称")
            return
        # 保存JSON
        file_path = self.recorder.save(name)
        # 存入行为库
        self.behavior_lib.add_behavior(name, desc, file_path)
        messagebox.showinfo("成功", f"行为 '{name}' 已保存")
        # 清空输入
        self.behavior_name.delete(0, tk.END)
        self.behavior_desc.delete(0, tk.END)
    
    def search_and_replay(self):
        query = self.behavior_query.get().strip()
        if not query:
            messagebox.showwarning("提示", "请输入关键词")
            return
        file_path, score = self.behavior_lib.search_behavior(query)
        if file_path:
            self.search_result.config(text=f"匹配行为，相似度{score:.2f}，开始回放")
            # 在新线程中回放，避免阻塞GUI
            threading.Thread(target=replay_mouse, args=(file_path,), daemon=True).start()
        else:
            self.search_result.config(text="未找到匹配行为")
    
    # ---------- 记忆管理标签页 ----------
    def create_memory_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="记忆管理")
        
        # 显示历史对话
        ttk.Label(tab, text="最近对话记忆（可检索）").pack(pady=5)
        self.memory_text = scrolledtext.ScrolledText(tab, width=80, height=25)
        self.memory_text.pack(fill='both', expand=True, padx=10, pady=5)
        
        # 刷新按钮
        btn_refresh = ttk.Button(tab, text="刷新记忆", command=self.refresh_memory)
        btn_refresh.pack(pady=5)
        
        # 简单检索测试
        frame_search = ttk.Frame(tab)
        frame_search.pack(fill='x', padx=10, pady=5)
        ttk.Label(frame_search, text="检索记忆:").pack(side='left')
        self.memory_query = ttk.Entry(frame_search, width=40)
        self.memory_query.pack(side='left', padx=5)
        btn_search_mem = ttk.Button(frame_search, text="检索", command=self.search_memory)
        btn_search_mem.pack(side='left')
        
        self.refresh_memory()
    
    def refresh_memory(self):
        """从向量库加载最近记忆并显示（简单展示元数据）"""
        # 由于没有直接获取全部，这里展示metadata的前50条
        self.memory_text.delete(1.0, tk.END)
        if self.memory.vector_store.metadata:
            for i, item in enumerate(self.memory.vector_store.metadata[-20:]):
                self.memory_text.insert(tk.END, f"{i+1}. {item['text'][:200]}...\n\n")
        else:
            self.memory_text.insert(tk.END, "暂无记忆")
    
    def search_memory(self):
        query = self.memory_query.get().strip()
        if not query:
            return
        results = self.memory.retrieve_relevant(query)
        self.memory_text.delete(1.0, tk.END)
        if results:
            for r in results:
                self.memory_text.insert(tk.END, f"相似度: {r['score']:.3f}\n{r['text']}\n\n")
        else:
            self.memory_text.insert(tk.END, "未找到相关记忆")
    
    # ---------- 设置 ----------
    def set_api_key(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("设置API密钥")
        dialog.geometry("400x150")
        ttk.Label(dialog, text="DeepSeek API Key:").pack(pady=10)
        key_entry = ttk.Entry(dialog, width=50)
        key_entry.pack(pady=5)
        key_entry.insert(0, config.DEEPSEEK_API_KEY)  # 显示当前密钥
        
        def save_key():
            new_key = key_entry.get().strip()
            if new_key:
                config.DEEPSEEK_API_KEY = new_key
                # 更新RAG中的llm实例（重新创建）
                self.rag.llm = DeepSeekAPI(new_key)
                messagebox.showinfo("成功", "API密钥已更新")
                dialog.destroy()
        
        ttk.Button(dialog, text="保存", command=save_key).pack(pady=10)

# 启动主程序
def run_gui():
    root = tk.Tk()
    app = App(root)
    root.mainloop()