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
        # 检查是否有录制事件
        if not hasattr(self.recorder, 'events') or not self.recorder.events:
            messagebox.showwarning("提示", "请先录制轨迹（点击开始录制，移动鼠标并点击，然后停止）")
            return
        name = self.behavior_name.get().strip()
        desc = self.behavior_desc.get().strip()
        if not name:
            messagebox.showwarning("提示", "请输入行为名称")
            return
        try:
            file_path = self.recorder.save(name)
            # 存入行为库
            self.behavior_lib.add_behavior(name, desc, file_path)
            messagebox.showinfo("成功", f"行为 '{name}' 已保存\n路径: {file_path}")
            # 清空输入
            self.behavior_name.delete(0, tk.END)
            self.behavior_desc.delete(0, tk.END)
            # 清空录制事件，避免下次保存重复
            self.recorder.events = []
            self.record_status.config(text="已保存，可重新录制")
        except Exception as e:
            messagebox.showerror("保存失败", str(e))
    
    def search_and_replay(self):
        query = self.behavior_query.get().strip()
        if not query:
            messagebox.showwarning("提示", "请输入关键词")
            return
        # 检索多个结果（可调阈值）
        results = self.behavior_lib.vector_store.search(query, top_k=3)
        if not results:
            self.search_result.config(text="未找到匹配行为")
            return
        
        # 显示检索结果供用户选择（简单起见，取第一个）
        best = results[0]
        file_path = best['source']
        score = best['score']
        
        if score < config.BEHAVIOR_SIMILARITY_THRESHOLD:
            self.search_result.config(text=f"相似度{score:.2f}低于阈值，不执行")
            return
        
        self.search_result.config(text=f"匹配行为，相似度{score:.2f}，开始回放...")
        
        def replay_task():
            try:
                replay_mouse(file_path)
                self.root.after(0, lambda: self.search_result.config(text="回放完成"))
            except Exception as e:
                self.root.after(0, lambda: self.search_result.config(text=f"回放失败: {e}"))
        
        threading.Thread(target=replay_task, daemon=True).start()
    
    # ---------- 记忆管理标签页 ----------
    def create_memory_tab(self):
        """记忆管理标签页（完整 CRUD）"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="记忆管理")

        # ========== 顶部：手动添加记忆 ==========
        frame_add = ttk.LabelFrame(tab, text="添加新记忆")
        frame_add.pack(fill='x', padx=10, pady=5)

        ttk.Label(frame_add, text="内容:").grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.new_memory_entry = ttk.Entry(frame_add, width=70)
        self.new_memory_entry.grid(row=0, column=1, padx=5, pady=5, sticky='ew')
        btn_add = ttk.Button(frame_add, text="添加", command=self.add_memory)
        btn_add.grid(row=0, column=2, padx=5, pady=5)
        frame_add.grid_columnconfigure(1, weight=1)

        # ========== 中间：记忆列表（可多选删除） ==========
        frame_list = ttk.LabelFrame(tab, text="已存储的记忆")
        frame_list.pack(fill='both', expand=True, padx=10, pady=5)

        # 创建 Treeview，添加复选框列
        columns = ("选中", "序号", "内容", "来源", "操作")
        self.memory_tree = ttk.Treeview(frame_list, columns=columns, show="headings", height=15)
        
        # 设置列标题
        self.memory_tree.heading("选中", text="✅")
        self.memory_tree.heading("序号", text="#")
        self.memory_tree.heading("内容", text="记忆内容")
        self.memory_tree.heading("来源", text="来源")
        self.memory_tree.heading("操作", text="删除")
        
        # 设置列宽
        self.memory_tree.column("选中", width=40, anchor='center')
        self.memory_tree.column("序号", width=50, anchor='center')
        self.memory_tree.column("内容", width=450)
        self.memory_tree.column("来源", width=100)
        self.memory_tree.column("操作", width=80, anchor='center')

        # 滚动条
        scrollbar = ttk.Scrollbar(frame_list, orient="vertical", command=self.memory_tree.yview)
        self.memory_tree.configure(yscrollcommand=scrollbar.set)
        self.memory_tree.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        # 绑定点击事件（用于复选框切换和删除按钮）
        self.memory_tree.bind('<ButtonRelease-1>', self.on_memory_tree_click)

        # ========== 底部：操作按钮 + 检索 ==========
        frame_bottom = ttk.Frame(tab)
        frame_bottom.pack(fill='x', padx=10, pady=5)

        # 左侧按钮组
        btn_frame = ttk.Frame(frame_bottom)
        btn_frame.pack(side='left', fill='x', expand=True)
        
        btn_delete_selected = ttk.Button(btn_frame, text="删除选中", command=self.delete_selected_memories)
        btn_delete_selected.pack(side='left', padx=5)
        
        btn_clear_all = ttk.Button(btn_frame, text="清空全部", command=self.clear_all_memories)
        btn_clear_all.pack(side='left', padx=5)
        
        btn_refresh = ttk.Button(btn_frame, text="刷新列表", command=self.refresh_memory_list)
        btn_refresh.pack(side='left', padx=5)

        # 右侧检索框
        search_frame = ttk.Frame(frame_bottom)
        search_frame.pack(side='right')
        ttk.Label(search_frame, text="检索:").pack(side='left')
        self.memory_search_entry = ttk.Entry(search_frame, width=25)
        self.memory_search_entry.pack(side='left', padx=5)
        btn_search = ttk.Button(search_frame, text="搜索", command=self.search_memory)
        btn_search.pack(side='left')
        
        # 初始化显示列表
        self.refresh_memory_list()
    
    def refresh_memory_list(self):
        """刷新记忆列表（从 memory 对象加载最新数据）"""
        # 清空现有行
        for row in self.memory_tree.get_children():
            self.memory_tree.delete(row)
        
        # 获取所有记忆
        memories = self.memory.get_all_memories(limit=100)
        
        # 插入数据
        for idx, mem in enumerate(memories):
            # 内容截断显示
            content = mem['text']
            if len(content) > 60:
                content = content[:60] + "..."
            
            # 插入行，第0列留空（复选框用变量存储状态）
            item_id = self.memory_tree.insert('', 'end', values=(
                '□',          # 复选框（未选中）
                idx + 1,      # 序号
                content,
                mem['source'],
                '🗑️ 删除'     # 删除按钮文本
            ))
            # 存储完整内容以备查看
            self.memory_tree.set(item_id, column='内容', value=content)
            # 可以将完整内容作为隐藏数据存储（可选）
            self.memory_tree.item(item_id, tags=(str(idx),))  # 用 tags 存储原始索引

    def add_memory(self):
        """手动添加记忆"""
        content = self.new_memory_entry.get().strip()
        if not content:
            messagebox.showwarning("提示", "请输入记忆内容")
            return
        try:
            self.memory.add_memory(content, source="manual")
            messagebox.showinfo("成功", "记忆已添加")
            self.new_memory_entry.delete(0, tk.END)
            self.refresh_memory_list()
        except Exception as e:
            messagebox.showerror("错误", f"添加失败: {e}")

    def on_memory_tree_click(self, event):
        """处理 Treeview 点击事件（复选框切换、删除按钮）"""
        region = self.memory_tree.identify_region(event.x, event.y)
        if region != "cell":
            return
        
        column = self.memory_tree.identify_column(event.x)
        row_id = self.memory_tree.identify_row(event.y)
        if not row_id:
            return
        
        # 点击“操作”列（删除按钮）
        if column == '#5':  # 第5列是“操作”
            # 获取该行的原始索引（从 tags 或序号）
            item_values = self.memory_tree.item(row_id, 'values')
            idx = int(item_values[1]) - 1  # 序号从1开始，转0基
            if messagebox.askyesno("确认删除", f"确定要删除这条记忆吗？"):
                if self.memory.delete_memory(idx):
                    messagebox.showinfo("成功", "记忆已删除")
                    self.refresh_memory_list()
                else:
                    messagebox.showerror("错误", "删除失败")
            return
        
        # 点击“选中”列（复选框）
        if column == '#1':
            current_val = self.memory_tree.item(row_id, 'values')[0]
            new_val = '☑' if current_val == '□' else '□'
            self.memory_tree.set(row_id, column='选中', value=new_val)

    def delete_selected_memories(self):
        """删除所有被选中的记忆"""
        selected_indices = []
        for row_id in self.memory_tree.get_children():
            if self.memory_tree.item(row_id, 'values')[0] == '☑':
                idx = int(self.memory_tree.item(row_id, 'values')[1]) - 1
                selected_indices.append(idx)
        
        if not selected_indices:
            messagebox.showwarning("提示", "请先勾选要删除的记忆")
            return
        
        # 从后往前删除，避免索引变化
        if messagebox.askyesno("确认删除", f"确定要删除选中的 {len(selected_indices)} 条记忆吗？"):
            for idx in sorted(selected_indices, reverse=True):
                self.memory.delete_memory(idx)
            self.refresh_memory_list()
            messagebox.showinfo("成功", f"已删除 {len(selected_indices)} 条记忆")

    def clear_all_memories(self):
        """清空所有记忆"""
        if messagebox.askyesno("警告", "确定要清空所有记忆吗？此操作不可恢复！"):
            self.memory.clear_all()
            self.refresh_memory_list()
            messagebox.showinfo("成功", "所有记忆已清空")

    def search_memory(self):
        """检索记忆并显示结果"""
        query = self.memory_search_entry.get().strip()
        if not query:
            messagebox.showwarning("提示", "请输入检索关键词")
            return
        
        results = self.memory.retrieve_relevant(query, top_k=10)
        
        # 清空现有列表，显示检索结果
        for row in self.memory_tree.get_children():
            self.memory_tree.delete(row)
        
        if not results:
            self.memory_tree.insert('', 'end', values=('', '', '未找到相关记忆', '', ''))
            return
        
        for idx, res in enumerate(results):
            content = res['text']
            if len(content) > 60:
                content = content[:60] + "..."
            self.memory_tree.insert('', 'end', values=(
                '□',
                idx + 1,
                content,
                f"{res['source']} (相似度{res['score']:.2f})",
                '🗑️ 删除'
            ))
    
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