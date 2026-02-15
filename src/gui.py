import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext, simpledialog
import threading
from pathlib import Path
from . import config
from .rag_core import RAGEngine
from .document_processor import parse_document, split_text
from .automation import MouseRecorder, BehaviorLibrary, replay_mouse, ScriptBuilder, HAS_CV2
from .memory import ConversationMemory
from .models import DeepSeekAPI, EmbeddingModel
from .config_manager import set_api_key as save_api_key, get_api_key
import faiss

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("RAG桌面助手 - 红岩网校考核")
        self.root.geometry("1000x750")
        
        # 初始化核心组件
        self.rag = RAGEngine()
        self.recorder = MouseRecorder(on_stop_callback=self.on_recording_stopped)
        self.behavior_lib = BehaviorLibrary()
        self.memory = ConversationMemory()
        self.embed_model = EmbeddingModel()
        
        # 回放速度因子
        self.replay_speed = tk.DoubleVar(value=1.0)
        
        # 创建标签页
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True)
        
        self.create_rag_tab()
        self.create_automation_tab()
        self.create_behavior_repo_tab()
        self.create_memory_tab()
        self.create_documents_tab()          # 新增：文档管理标签页
        
        # 菜单栏（设置API Key）
        menubar = tk.Menu(root)
        root.config(menu=menubar)
        settings_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="设置", menu=settings_menu)
        settings_menu.add_command(label="配置API密钥", command=self.set_api_key)
        
        # 启动时检查 API key 是否已设置
        if not config.DEEPSEEK_API_KEY:
            self.root.after(100, lambda: messagebox.showwarning("提示", "请先设置 DeepSeek API 密钥"))
            self.root.after(200, self.set_api_key)
    
    def on_recording_stopped(self, event_count):
        """当录制停止时的回调函数（F2 触发或按钮触发）"""
        self.root.after(0, self._update_ui_after_stop, event_count)
    
    def _update_ui_after_stop(self, event_count):
        """更新 GUI 状态"""
        self.btn_record.config(text="开始录制")
        self.record_status.config(text=f"录制结束，共{event_count}个事件，可保存")
        print(f"GUI 状态已更新: 录制结束，{event_count} 个事件")
    
    # ---------- 文档问答标签页 ----------
    def create_rag_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="文档问答")
        
        # 上传文件区域
        frame_upload = ttk.LabelFrame(tab, text="上传文档")
        frame_upload.pack(fill='x', padx=10, pady=5)
        self.btn_upload = ttk.Button(frame_upload, text="选择文件", command=self.upload_document)
        self.btn_upload.pack(side='left', padx=5, pady=5)
        self.lbl_upload_status = ttk.Label(frame_upload, text="未上传文件")
        self.lbl_upload_status.pack(side='left', padx=5)
        
        # 问答区域
        frame_qa = ttk.LabelFrame(tab, text="问答")
        frame_qa.pack(fill='both', expand=True, padx=10, pady=5)
        
        ttk.Label(frame_qa, text="问题:").grid(row=0, column=0, sticky='w', padx=5, pady=5)
        self.question_entry = ttk.Entry(frame_qa, width=60)
        self.question_entry.grid(row=0, column=1, padx=5, pady=5, sticky='ew')
        btn_ask = ttk.Button(frame_qa, text="提问", command=self.ask_question)
        btn_ask.grid(row=0, column=2, padx=5, pady=5)
        
        ttk.Label(frame_qa, text="回答:").grid(row=1, column=0, sticky='nw', padx=5, pady=5)
        self.answer_text = scrolledtext.ScrolledText(frame_qa, width=70, height=20, wrap=tk.WORD)
        self.answer_text.grid(row=1, column=1, columnspan=2, padx=5, pady=5, sticky='nsew')
        frame_qa.grid_columnconfigure(1, weight=1)
        frame_qa.grid_rowconfigure(1, weight=1)
    
    def upload_document(self):
        """异步上传文档，避免界面卡死"""
        file_path = filedialog.askopenfilename(
            title="选择文档",
            filetypes=[("文档", "*.pdf *.md *.txt"), ("PDF", "*.pdf"), ("Markdown", "*.md"), ("文本", "*.txt")]
        )
        if not file_path:
            return
        
        # 禁用按钮，显示处理中状态
        self.btn_upload.config(state='disabled', text="处理中...")
        self.lbl_upload_status.config(text="正在解析文档，请稍候...")
        
        def task():
            try:
                text = parse_document(file_path)
                chunks = split_text(text)
                sources = [Path(file_path).name] * len(chunks)
                self.rag.doc_store.add_documents(chunks, sources)
                # 处理完成后更新UI
                self.root.after(0, self._upload_success, file_path, len(chunks))
            except Exception as e:
                self.root.after(0, self._upload_error, str(e))
        
        threading.Thread(target=task, daemon=True).start()
    
    def _upload_success(self, file_path, chunk_count):
        self.btn_upload.config(state='normal', text="选择文件")
        self.lbl_upload_status.config(text=f"已上传并索引: {Path(file_path).name} ({chunk_count}块)")
        messagebox.showinfo("成功", "文档已处理并加入知识库")
        self.refresh_document_list()   # 刷新文档管理列表
    
    def _upload_error(self, error_msg):
        self.btn_upload.config(state='normal', text="选择文件")
        self.lbl_upload_status.config(text="上传失败")
        messagebox.showerror("错误", f"处理失败: {error_msg}")
    
    def ask_question(self):
        question = self.question_entry.get().strip()
        if not question:
            messagebox.showwarning("提示", "请输入问题")
            return
        self.answer_text.delete(1.0, tk.END)
        self.answer_text.insert(tk.END, "正在思考...")
        
        def task():
            answer = self.rag.answer_question(question)
            self.memory.add_dialogue(question, answer)
            self.root.after(0, lambda: self.answer_text.delete(1.0, tk.END))
            self.root.after(0, lambda: self.answer_text.insert(tk.END, answer))
        
        threading.Thread(target=task, daemon=True).start()
    
    # ---------- 行为自动化标签页 ----------
    def create_automation_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="行为录制")
        
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
        
        btn_new_script = ttk.Button(frame_record, text="新建脚本", command=self.open_script_builder)
        btn_new_script.pack(side='left', padx=5)
        
        # 回放速度调节
        frame_speed = ttk.LabelFrame(tab, text="回放速度")
        frame_speed.pack(fill='x', padx=10, pady=5)
        ttk.Label(frame_speed, text="速度因子:").pack(side='left', padx=5)
        speed_scale = ttk.Scale(frame_speed, from_=0.2, to=3.0, orient='horizontal',
                                 variable=self.replay_speed, length=200)
        speed_scale.pack(side='left', padx=5)
        self.speed_label = ttk.Label(frame_speed, text="1.0x")
        self.speed_label.pack(side='left', padx=5)
        def update_speed_label(*args):
            self.speed_label.config(text=f"{self.replay_speed.get():.1f}x")
        self.replay_speed.trace_add('write', update_speed_label)
        
        # 快速检索与回放
        frame_query = ttk.LabelFrame(tab, text="快速检索并执行")
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
            self.recorder.start()
            self.btn_record.config(text="停止录制")
            self.record_status.config(text="正在录制... (按 F2 停止)")
            print("开始录制...")
        else:
            events = self.recorder.stop()
            self.btn_record.config(text="开始录制")
            self.record_status.config(text=f"录制结束，共{len(events)}个事件，可保存")
            print(f"停止录制，{len(events)} 个事件")
    
    def save_behavior(self):
        if not self.recorder.has_recorded_data():
            messagebox.showwarning("提示", "请先录制轨迹！\n\n步骤：\n1. 点击'开始录制'\n2. 移动鼠标并点击\n3. 按 F2 或点击'停止录制'\n4. 输入名称和描述\n5. 点击'保存轨迹'")
            return
        
        name = self.behavior_name.get().strip()
        desc = self.behavior_desc.get().strip()
        
        if not name:
            messagebox.showwarning("提示", "请输入行为名称")
            return
        
        try:
            file_path = self.recorder.save(name)
            print(f"轨迹文件已保存: {file_path}")
            self.behavior_lib.add_behavior(name, desc, file_path)
            messagebox.showinfo("成功", f"行为 '{name}' 已保存并建立索引！\n路径: {file_path}")
            
            self.behavior_name.delete(0, tk.END)
            self.behavior_desc.delete(0, tk.END)
            self.recorder.events = []
            self.record_status.config(text="已保存，可重新录制")
            self.refresh_behavior_list()
        except Exception as e:
            print(f"保存失败: {e}")
            messagebox.showerror("保存失败", f"保存行为时出错:\n{str(e)}")
    
    def search_and_replay(self):
        query = self.behavior_query.get().strip()
        if not query:
            messagebox.showwarning("提示", "请输入关键词")
            return
        results = self.behavior_lib.search_behavior(query, top_k=3)
        if not results:
            self.search_result.config(text="未找到匹配行为")
            return
        
        best = results[0]
        file_path = best['file_path']
        score = best['score']
        name = best['name']
        speed = self.replay_speed.get()
        
        self.search_result.config(text=f"匹配 '{name}'，相似度{score:.2f}，速度{speed:.1f}x，开始回放...")
        
        def replay_task():
            try:
                replay_mouse(file_path, speed_factor=speed)
                self.root.after(0, lambda: self.search_result.config(text="回放完成"))
            except Exception as e:
                self.root.after(0, lambda: self.search_result.config(text=f"回放失败: {e}"))
        
        threading.Thread(target=replay_task, daemon=True).start()
    
    def open_script_builder(self):
        """打开脚本构建器窗口"""
        builder_win = tk.Toplevel(self.root)
        builder_win.title("新建自动化脚本")
        builder_win.geometry("700x550")
        
        # 步骤列表
        frame_list = ttk.LabelFrame(builder_win, text="步骤列表")
        frame_list.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.script_listbox = tk.Listbox(frame_list, height=8)
        self.script_listbox.pack(side='left', fill='both', expand=True, padx=5, pady=5)
        scrollbar = ttk.Scrollbar(frame_list, orient='vertical', command=self.script_listbox.yview)
        scrollbar.pack(side='right', fill='y')
        self.script_listbox.config(yscrollcommand=scrollbar.set)
        
        btn_delete_step = ttk.Button(frame_list, text="删除选中步骤", command=self.delete_selected_step)
        btn_delete_step.pack(pady=5)
        
        # 步骤编辑区域
        frame_edit = ttk.LabelFrame(builder_win, text="添加步骤")
        frame_edit.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(frame_edit, text="类型:").grid(row=0, column=0, padx=5, pady=5)
        self.step_type = ttk.Combobox(frame_edit, values=["移动", "点击", "图像点击", "等待", "键盘输入"], state="readonly")
        self.step_type.grid(row=0, column=1, padx=5, pady=5)
        self.step_type.current(0)
        self.step_type.bind("<<ComboboxSelected>>", self.on_step_type_change)
        
        self.param_frame = ttk.Frame(frame_edit)
        self.param_frame.grid(row=1, column=0, columnspan=3, padx=5, pady=5, sticky='ew')
        
        self.create_move_params()
        
        btn_add = ttk.Button(frame_edit, text="添加步骤", command=self.add_script_step)
        btn_add.grid(row=2, column=0, columnspan=2, pady=10)
        
        btn_save = ttk.Button(builder_win, text="保存脚本", command=self.save_script)
        btn_save.pack(pady=5)
        
        self.current_steps = []
    
    def delete_selected_step(self):
        selection = self.script_listbox.curselection()
        if not selection:
            messagebox.showwarning("提示", "请先选中要删除的步骤")
            return
        index = selection[0]
        del self.current_steps[index]
        self.script_listbox.delete(index)
    
    def on_step_type_change(self, event=None):
        for widget in self.param_frame.winfo_children():
            widget.destroy()
        step_type = self.step_type.get()
        if step_type == "移动":
            self.create_move_params()
        elif step_type == "点击":
            self.create_click_params()
        elif step_type == "图像点击":
            self.create_image_click_params()
        elif step_type == "等待":
            self.create_wait_params()
        elif step_type == "键盘输入":
            self.create_typewrite_params()
    
    def create_move_params(self):
        ttk.Label(self.param_frame, text="X:").grid(row=0, column=0, padx=5)
        self.move_x = ttk.Entry(self.param_frame, width=10)
        self.move_x.grid(row=0, column=1, padx=5)
        ttk.Label(self.param_frame, text="Y:").grid(row=0, column=2, padx=5)
        self.move_y = ttk.Entry(self.param_frame, width=10)
        self.move_y.grid(row=0, column=3, padx=5)
        ttk.Label(self.param_frame, text="持续时间(秒):").grid(row=1, column=0, padx=5)
        self.move_duration = ttk.Entry(self.param_frame, width=10)
        self.move_duration.insert(0, "0.2")
        self.move_duration.grid(row=1, column=1, padx=5)
    
    def create_click_params(self):
        ttk.Label(self.param_frame, text="X(可选):").grid(row=0, column=0, padx=5)
        self.click_x = ttk.Entry(self.param_frame, width=10)
        self.click_x.grid(row=0, column=1, padx=5)
        ttk.Label(self.param_frame, text="Y(可选):").grid(row=0, column=2, padx=5)
        self.click_y = ttk.Entry(self.param_frame, width=10)
        self.click_y.grid(row=0, column=3, padx=5)
        ttk.Label(self.param_frame, text="按键:").grid(row=1, column=0, padx=5)
        self.click_button = ttk.Combobox(self.param_frame, values=["left", "right", "middle"], width=8)
        self.click_button.grid(row=1, column=1, padx=5)
        self.click_button.current(0)
        ttk.Label(self.param_frame, text="点击次数:").grid(row=1, column=2, padx=5)
        self.click_clicks = ttk.Entry(self.param_frame, width=5)
        self.click_clicks.insert(0, "1")
        self.click_clicks.grid(row=1, column=3, padx=5)
    
    def create_image_click_params(self):
        ttk.Label(self.param_frame, text="图片路径:").grid(row=0, column=0, padx=5)
        self.image_path = ttk.Entry(self.param_frame, width=30)
        self.image_path.grid(row=0, column=1, padx=5)
        btn_browse = ttk.Button(self.param_frame, text="浏览", command=self.browse_image)
        btn_browse.grid(row=0, column=2, padx=5)
        ttk.Label(self.param_frame, text="置信度:").grid(row=1, column=0, padx=5)
        self.image_confidence = ttk.Entry(self.param_frame, width=10)
        self.image_confidence.insert(0, "0.8")
        self.image_confidence.grid(row=1, column=1, padx=5)
        ttk.Label(self.param_frame, text="按键:").grid(row=1, column=2, padx=5)
        self.image_button = ttk.Combobox(self.param_frame, values=["left", "right", "middle"], width=8)
        self.image_button.grid(row=1, column=3, padx=5)
        self.image_button.current(0)
        if not HAS_CV2:
            ttk.Label(self.param_frame, text="⚠️ 未安装 opencv-python", foreground="red").grid(row=2, column=0, columnspan=4, pady=5)
    
    def create_wait_params(self):
        ttk.Label(self.param_frame, text="等待秒数:").grid(row=0, column=0, padx=5)
        self.wait_seconds = ttk.Entry(self.param_frame, width=10)
        self.wait_seconds.insert(0, "1.0")
        self.wait_seconds.grid(row=0, column=1, padx=5)
    
    def create_typewrite_params(self):
        ttk.Label(self.param_frame, text="文本:").grid(row=0, column=0, padx=5)
        self.type_text = ttk.Entry(self.param_frame, width=30)
        self.type_text.grid(row=0, column=1, padx=5)
        ttk.Label(self.param_frame, text="间隔(秒):").grid(row=1, column=0, padx=5)
        self.type_interval = ttk.Entry(self.param_frame, width=10)
        self.type_interval.insert(0, "0.1")
        self.type_interval.grid(row=1, column=1, padx=5)
    
    def browse_image(self):
        filename = filedialog.askopenfilename(filetypes=[("图片", "*.png *.jpg *.jpeg *.bmp")])
        if filename:
            self.image_path.delete(0, tk.END)
            self.image_path.insert(0, filename)
    
    def add_script_step(self):
        step_type = self.step_type.get()
        step = {}
        try:
            if step_type == "移动":
                x = int(self.move_x.get())
                y = int(self.move_y.get())
                duration = float(self.move_duration.get())
                step = {"type": "move", "x": x, "y": y, "duration": duration}
            elif step_type == "点击":
                x = self.click_x.get().strip()
                y = self.click_y.get().strip()
                button = self.click_button.get()
                clicks = int(self.click_clicks.get())
                step = {"type": "click", "button": button, "clicks": clicks}
                if x:
                    step["x"] = int(x)
                if y:
                    step["y"] = int(y)
            elif step_type == "图像点击":
                img_path = self.image_path.get().strip()
                if not img_path:
                    messagebox.showwarning("提示", "请选择图片")
                    return
                confidence = float(self.image_confidence.get())
                button = self.image_button.get()
                step = {"type": "image_click", "image_path": img_path, "confidence": confidence, "button": button}
            elif step_type == "等待":
                seconds = float(self.wait_seconds.get())
                step = {"type": "wait", "seconds": seconds}
            elif step_type == "键盘输入":
                text = self.type_text.get()
                interval = float(self.type_interval.get())
                step = {"type": "typewrite", "text": text, "interval": interval}
            else:
                return
        except ValueError as e:
            messagebox.showerror("错误", f"参数格式错误: {e}")
            return
        
        self.current_steps.append(step)
        summary = f"{len(self.current_steps)}. {step_type}: {step}"
        self.script_listbox.insert(tk.END, summary)
    
    def save_script(self):
        if not self.current_steps:
            messagebox.showwarning("提示", "请先添加步骤")
            return
        
        name = simpledialog.askstring("保存脚本", "请输入脚本名称:", parent=self.root)
        if not name:
            return
        desc = simpledialog.askstring("保存脚本", "请输入脚本描述:", parent=self.root) or ""
        
        file_path = config.RECORDINGS_DIR / f"{name}.json"
        builder = ScriptBuilder()
        builder.steps = self.current_steps
        builder.save(file_path)
        
        self.behavior_lib.add_behavior(name, desc, str(file_path))
        messagebox.showinfo("成功", f"脚本 '{name}' 已保存并加入行为库")
        self.refresh_behavior_list()
        self.script_listbox.master.destroy()
    
    # ---------- 行为仓库标签页 ----------
    def create_behavior_repo_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="行为仓库")
        
        frame_search = ttk.LabelFrame(tab, text="搜索行为")
        frame_search.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(frame_search, text="关键词:").pack(side='left', padx=5)
        self.behavior_search_entry = ttk.Entry(frame_search, width=40)
        self.behavior_search_entry.pack(side='left', padx=5)
        ttk.Button(frame_search, text="搜索", command=self.search_behavior_repo).pack(side='left', padx=5)
        ttk.Button(frame_search, text="显示全部", command=self.refresh_behavior_list).pack(side='left', padx=5)
        
        frame_list = ttk.LabelFrame(tab, text="已存储的行为")
        frame_list.pack(fill='both', expand=True, padx=10, pady=5)
        
        columns = ("选中", "序号", "名称", "描述", "创建时间", "操作")
        self.behavior_tree = ttk.Treeview(frame_list, columns=columns, show="headings", height=15)
        
        self.behavior_tree.heading("选中", text="✓")
        self.behavior_tree.heading("序号", text="#")
        self.behavior_tree.heading("名称", text="行为名称")
        self.behavior_tree.heading("描述", text="描述")
        self.behavior_tree.heading("创建时间", text="创建时间")
        self.behavior_tree.heading("操作", text="操作")
        
        self.behavior_tree.column("选中", width=30, anchor='center')
        self.behavior_tree.column("序号", width=40, anchor='center')
        self.behavior_tree.column("名称", width=150)
        self.behavior_tree.column("描述", width=350)
        self.behavior_tree.column("创建时间", width=120, anchor='center')
        self.behavior_tree.column("操作", width=150, anchor='center')
        
        scrollbar = ttk.Scrollbar(frame_list, orient="vertical", command=self.behavior_tree.yview)
        self.behavior_tree.configure(yscrollcommand=scrollbar.set)
        self.behavior_tree.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        self.behavior_tree.bind('<ButtonRelease-1>', self.on_behavior_tree_click)
        
        frame_bottom = ttk.Frame(tab)
        frame_bottom.pack(fill='x', padx=10, pady=5)
        
        btn_delete_selected = ttk.Button(frame_bottom, text="删除选中", command=self.delete_selected_behaviors)
        btn_delete_selected.pack(side='left', padx=5)
        
        btn_clear_all = ttk.Button(frame_bottom, text="清空全部", command=self.clear_all_behaviors)
        btn_clear_all.pack(side='left', padx=5)
        
        btn_refresh = ttk.Button(frame_bottom, text="刷新列表", command=self.refresh_behavior_list)
        btn_refresh.pack(side='left', padx=5)
        
        self.refresh_behavior_list()
    
    def refresh_behavior_list(self):
        print("刷新行为列表...")
        for row in self.behavior_tree.get_children():
            self.behavior_tree.delete(row)
        
        behaviors = self.behavior_lib.get_all()
        print(f"获取到 {len(behaviors)} 个行为")
        
        if not behaviors:
            self.behavior_tree.insert('', 'end', values=('', '', '暂无行为', '请先录制并保存行为', '', ''))
            return
        
        for idx, behavior in enumerate(behaviors):
            desc = behavior.get('description', '')
            if len(desc) > 50:
                desc = desc[:50] + "..."
            
            item_id = self.behavior_tree.insert('', 'end', values=(
                '□',
                idx + 1,
                behavior['name'],
                desc,
                behavior.get('created_at', '未知'),
                '[执行]  [删除]'
            ))
            self.behavior_tree.item(item_id, tags=(str(idx),))
    
    def search_behavior_repo(self):
        query = self.behavior_search_entry.get().strip()
        if not query:
            self.refresh_behavior_list()
            return
        
        results = self.behavior_lib.search_behavior(query, top_k=10)
        
        for row in self.behavior_tree.get_children():
            self.behavior_tree.delete(row)
        
        if not results:
            self.behavior_tree.insert('', 'end', values=('', '', '未找到', f'没有匹配 "{query}" 的行为', '', ''))
            return
        
        for idx, res in enumerate(results):
            desc = res['description']
            if len(desc) > 50:
                desc = desc[:50] + "..."
            
            self.behavior_tree.insert('', 'end', values=(
                '□',
                idx + 1,
                f"{res['name']} (相似度: {res['score']:.2f})",
                desc,
                res.get('created_at', '未知'),
                '[执行]  [删除]'
            ), tags=(str(res['index']),))
    
    def on_behavior_tree_click(self, event):
        region = self.behavior_tree.identify_region(event.x, event.y)
        if region != "cell":
            return
        
        column = self.behavior_tree.identify_column(event.x)
        row_id = self.behavior_tree.identify_row(event.y)
        if not row_id:
            return
        
        tags = self.behavior_tree.item(row_id, 'tags')
        if not tags:
            return
        
        try:
            original_idx = int(tags[0])
        except ValueError:
            return
        
        if column == '#6':  # 操作列
            bbox = self.behavior_tree.bbox(row_id, column='#6')
            if not bbox:
                return
            col_x, col_y, col_width, col_height = bbox
            relative_x = event.x - col_x
            if relative_x < 70:  # 执行
                self.execute_behavior_by_index(original_idx)
            else:  # 删除
                values = self.behavior_tree.item(row_id, 'values')
                name = values[2] if len(values) > 2 else "未知"
                if messagebox.askyesno("确认删除", f"确定要删除行为 '{name}' 吗？"):
                    if self.behavior_lib.delete(original_idx):
                        messagebox.showinfo("成功", "行为已删除")
                        self.refresh_behavior_list()
                    else:
                        messagebox.showerror("错误", "删除失败")
        
        elif column == '#1':  # 选中列
            values = self.behavior_tree.item(row_id, 'values')
            current_val = values[0] if values else '□'
            new_val = '☑' if current_val == '□' else '□'
            self.behavior_tree.set(row_id, column='选中', value=new_val)
    
    def execute_behavior_by_index(self, index: int):
        behaviors = self.behavior_lib.get_all()
        if index < 0 or index >= len(behaviors):
            messagebox.showerror("错误", "行为索引无效")
            return
        
        behavior = behaviors[index]
        file_path = behavior['file_path']
        name = behavior['name']
        speed = self.replay_speed.get()
        
        print(f"执行行为: {name}, 文件: {file_path}, 速度: {speed}x")
        
        def replay_task():
            try:
                replay_mouse(file_path, speed_factor=speed)
                self.root.after(0, lambda: messagebox.showinfo("完成", f"行为 '{name}' 执行完成"))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("错误", f"执行失败: {e}"))
        
        threading.Thread(target=replay_task, daemon=True).start()
    
    def delete_selected_behaviors(self):
        selected_indices = []
        for row_id in self.behavior_tree.get_children():
            values = self.behavior_tree.item(row_id, 'values')
            if values and values[0] == '☑':
                tags = self.behavior_tree.item(row_id, 'tags')
                if tags:
                    try:
                        selected_indices.append(int(tags[0]))
                    except ValueError:
                        pass
        
        if not selected_indices:
            messagebox.showwarning("提示", "请先勾选要删除的行为")
            return
        
        if messagebox.askyesno("确认删除", f"确定要删除选中的 {len(selected_indices)} 条行为吗？"):
            for idx in sorted(selected_indices, reverse=True):
                self.behavior_lib.delete(idx)
            self.refresh_behavior_list()
            messagebox.showinfo("成功", f"已删除 {len(selected_indices)} 条行为")
    
    def clear_all_behaviors(self):
        if messagebox.askyesno("警告", "确定要清空所有行为吗？此操作不可恢复！"):
            self.behavior_lib.clear_all()
            self.refresh_behavior_list()
            messagebox.showinfo("成功", "所有行为已清空")
    
    # ---------- 记忆管理标签页 ----------
    def create_memory_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="记忆管理")

        frame_add = ttk.LabelFrame(tab, text="添加新记忆")
        frame_add.pack(fill='x', padx=10, pady=5)

        ttk.Label(frame_add, text="内容:").grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.new_memory_entry = ttk.Entry(frame_add, width=70)
        self.new_memory_entry.grid(row=0, column=1, padx=5, pady=5, sticky='ew')
        btn_add = ttk.Button(frame_add, text="添加", command=self.add_memory)
        btn_add.grid(row=0, column=2, padx=5, pady=5)
        frame_add.grid_columnconfigure(1, weight=1)

        frame_list = ttk.LabelFrame(tab, text="已存储的记忆")
        frame_list.pack(fill='both', expand=True, padx=10, pady=5)

        columns = ("选中", "序号", "内容", "来源", "操作")
        self.memory_tree = ttk.Treeview(frame_list, columns=columns, show="headings", height=15)
        
        self.memory_tree.heading("选中", text="✓")
        self.memory_tree.heading("序号", text="#")
        self.memory_tree.heading("内容", text="记忆内容")
        self.memory_tree.heading("来源", text="来源")
        self.memory_tree.heading("操作", text="删除")
        
        self.memory_tree.column("选中", width=30, anchor='center')
        self.memory_tree.column("序号", width=40, anchor='center')
        self.memory_tree.column("内容", width=500)
        self.memory_tree.column("来源", width=100)
        self.memory_tree.column("操作", width=80, anchor='center')

        scrollbar = ttk.Scrollbar(frame_list, orient="vertical", command=self.memory_tree.yview)
        self.memory_tree.configure(yscrollcommand=scrollbar.set)
        self.memory_tree.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        self.memory_tree.bind('<ButtonRelease-1>', self.on_memory_tree_click)

        frame_bottom = ttk.Frame(tab)
        frame_bottom.pack(fill='x', padx=10, pady=5)

        btn_frame = ttk.Frame(frame_bottom)
        btn_frame.pack(side='left', fill='x', expand=True)
        
        btn_delete_selected = ttk.Button(btn_frame, text="删除选中", command=self.delete_selected_memories)
        btn_delete_selected.pack(side='left', padx=5)
        
        btn_clear_all = ttk.Button(btn_frame, text="清空全部", command=self.clear_all_memories)
        btn_clear_all.pack(side='left', padx=5)
        
        btn_refresh = ttk.Button(btn_frame, text="刷新列表", command=self.refresh_memory_list)
        btn_refresh.pack(side='left', padx=5)

        search_frame = ttk.Frame(frame_bottom)
        search_frame.pack(side='right')
        ttk.Label(search_frame, text="检索:").pack(side='left')
        self.memory_search_entry = ttk.Entry(search_frame, width=25)
        self.memory_search_entry.pack(side='left', padx=5)
        btn_search = ttk.Button(search_frame, text="搜索", command=self.search_memory)
        btn_search.pack(side='left')
        
        self.refresh_memory_list()
    
    def refresh_memory_list(self):
        for row in self.memory_tree.get_children():
            self.memory_tree.delete(row)
        
        memories = self.memory.get_all_memories(limit=100)
        
        for idx, mem in enumerate(memories):
            content = mem['text']
            if len(content) > 70:
                content = content[:70] + "..."
            
            item_id = self.memory_tree.insert('', 'end', values=(
                '□',
                idx + 1,
                content,
                mem['source'],
                '[删除]'
            ))
            self.memory_tree.item(item_id, tags=(str(idx),))

    def add_memory(self):
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
        region = self.memory_tree.identify_region(event.x, event.y)
        if region != "cell":
            return
        
        column = self.memory_tree.identify_column(event.x)
        row_id = self.memory_tree.identify_row(event.y)
        if not row_id:
            return
        
        if column == '#5':  # 操作列（删除）
            item_values = self.memory_tree.item(row_id, 'values')
            idx = int(item_values[1]) - 1
            if messagebox.askyesno("确认删除", f"确定要删除这条记忆吗？"):
                if self.memory.delete_memory(idx):
                    messagebox.showinfo("成功", "记忆已删除")
                    self.refresh_memory_list()
                else:
                    messagebox.showerror("错误", "删除失败")
            return
        
        if column == '#1':  # 选中列
            current_val = self.memory_tree.item(row_id, 'values')[0]
            new_val = '☑' if current_val == '□' else '□'
            self.memory_tree.set(row_id, column='选中', value=new_val)

    def delete_selected_memories(self):
        selected_indices = []
        for row_id in self.memory_tree.get_children():
            if self.memory_tree.item(row_id, 'values')[0] == '☑':
                idx = int(self.memory_tree.item(row_id, 'values')[1]) - 1
                selected_indices.append(idx)
        
        if not selected_indices:
            messagebox.showwarning("提示", "请先勾选要删除的记忆")
            return
        
        if messagebox.askyesno("确认删除", f"确定要删除选中的 {len(selected_indices)} 条记忆吗？"):
            for idx in sorted(selected_indices, reverse=True):
                self.memory.delete_memory(idx)
            self.refresh_memory_list()
            messagebox.showinfo("成功", f"已删除 {len(selected_indices)} 条记忆")

    def clear_all_memories(self):
        if messagebox.askyesno("警告", "确定要清空所有记忆吗？此操作不可恢复！"):
            self.memory.clear_all()
            self.refresh_memory_list()
            messagebox.showinfo("成功", "所有记忆已清空")

    def search_memory(self):
        query = self.memory_search_entry.get().strip()
        if not query:
            messagebox.showwarning("提示", "请输入检索关键词")
            return
        
        results = self.memory.retrieve_relevant(query, top_k=10)
        
        for row in self.memory_tree.get_children():
            self.memory_tree.delete(row)
        
        if not results:
            self.memory_tree.insert('', 'end', values=('', '', '未找到相关记忆', '', ''))
            return
        
        for idx, res in enumerate(results):
            content = res['text']
            if len(content) > 70:
                content = content[:70] + "..."
            self.memory_tree.insert('', 'end', values=(
                '□',
                idx + 1,
                content,
                f"{res['source']} (相似度{res['score']:.2f})",
                '[删除]'
            ))
    
    # ---------- 新增：文档管理标签页 ----------
    def create_documents_tab(self):
        """文档管理标签页：查看和管理已上传的文档"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="文档管理")
        
        # 文档列表
        frame_list = ttk.LabelFrame(tab, text="已上传文档")
        frame_list.pack(fill='both', expand=True, padx=10, pady=5)
        
        columns = ("文件名", "大小", "状态")
        self.doc_tree = ttk.Treeview(frame_list, columns=columns, show="headings", height=15)
        
        self.doc_tree.heading("文件名", text="文件名")
        self.doc_tree.heading("大小", text="大小")
        self.doc_tree.heading("状态", text="状态")
        
        self.doc_tree.column("文件名", width=300)
        self.doc_tree.column("大小", width=100, anchor='center')
        self.doc_tree.column("状态", width=100, anchor='center')
        
        scrollbar = ttk.Scrollbar(frame_list, orient="vertical", command=self.doc_tree.yview)
        self.doc_tree.configure(yscrollcommand=scrollbar.set)
        self.doc_tree.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        frame_bottom = ttk.Frame(tab)
        frame_bottom.pack(fill='x', padx=10, pady=5)
        
        btn_refresh = ttk.Button(frame_bottom, text="刷新列表", command=self.refresh_document_list)
        btn_refresh.pack(side='left', padx=5)
        
        btn_delete = ttk.Button(frame_bottom, text="删除选中", command=self.delete_selected_document)
        btn_delete.pack(side='left', padx=5)
        
        btn_clear = ttk.Button(frame_bottom, text="清空所有", command=self.clear_all_documents)
        btn_clear.pack(side='left', padx=5)
        
        self.refresh_document_list()

    def refresh_document_list(self):
        """刷新文档列表"""
        for row in self.doc_tree.get_children():
            self.doc_tree.delete(row)
        
        if hasattr(self.rag.doc_store, 'metadata') and self.rag.doc_store.metadata:
            file_stats = {}
            for item in self.rag.doc_store.metadata:
                source = item.get('source', '未知')
                if source not in file_stats:
                    file_stats[source] = {
                        'count': 1,
                        'size': len(item.get('text', ''))
                    }
                else:
                    file_stats[source]['count'] += 1
                    file_stats[source]['size'] += len(item.get('text', ''))
            
            for filename, stats in file_stats.items():
                size_kb = stats['size'] / 1024
                if size_kb < 1024:
                    size_str = f"{size_kb:.1f} KB"
                else:
                    size_str = f"{size_kb/1024:.1f} MB"
                
                self.doc_tree.insert('', 'end', values=(
                    filename,
                    size_str,
                    f"{stats['count']} 个块"
                ))
        else:
            self.doc_tree.insert('', 'end', values=("暂无文档", "", ""))

    def delete_selected_document(self):
        """删除选中的文档"""
        selection = self.doc_tree.selection()
        if not selection:
            messagebox.showwarning("提示", "请先选中要删除的文档")
            return
        
        item = self.doc_tree.item(selection[0])
        filename = item['values'][0]
        
        if filename == "暂无文档":
            return
        
        if messagebox.askyesno("确认删除", f"确定要删除文档 '{filename}' 吗？\n这将从知识库中移除所有相关片段。"):
            # 从 metadata 中删除该文件的所有片段
            new_metadata = [m for m in self.rag.doc_store.metadata if m.get('source') != filename]
            
            if len(new_metadata) < len(self.rag.doc_store.metadata):
                self.rag.doc_store.metadata = new_metadata
                
                # 重建索引
                if new_metadata:
                    texts = [item['text'] for item in new_metadata]
                    vectors = self.rag.doc_store.emb_model.encode(texts)
                    self.rag.doc_store.index = faiss.IndexFlatIP(self.rag.doc_store.emb_model.dimension)
                    self.rag.doc_store.index.add(vectors)
                else:
                    self.rag.doc_store.index = faiss.IndexFlatIP(self.rag.doc_store.emb_model.dimension)
                
                self.rag.doc_store.save()
                messagebox.showinfo("成功", f"文档 '{filename}' 已删除")
                self.refresh_document_list()
            else:
                messagebox.showinfo("提示", "未找到该文档")

    def clear_all_documents(self):
        """清空所有文档"""
        if messagebox.askyesno("警告", "确定要清空所有文档吗？此操作不可恢复！"):
            self.rag.doc_store.index = faiss.IndexFlatIP(self.rag.doc_store.emb_model.dimension)
            self.rag.doc_store.metadata = []
            self.rag.doc_store.save()
            messagebox.showinfo("成功", "所有文档已清空")
            self.refresh_document_list()
    
    # ---------- 设置 API Key ----------
    def set_api_key(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("设置API密钥")
        dialog.geometry("400x150")
        ttk.Label(dialog, text="DeepSeek API Key:").pack(pady=10)
        key_entry = ttk.Entry(dialog, width=50)
        key_entry.pack(pady=5)
        current_key = get_api_key()
        key_entry.insert(0, current_key)
        
        def save_key():
            new_key = key_entry.get().strip()
            if new_key:
                save_api_key(new_key)
                import src.config as config
                config.DEEPSEEK_API_KEY = new_key
                from .models import DeepSeekAPI
                self.rag.llm = DeepSeekAPI(new_key)
                messagebox.showinfo("成功", "API密钥已保存")
                dialog.destroy()
            else:
                messagebox.showwarning("提示", "请输入API密钥")
        
        ttk.Button(dialog, text="保存", command=save_key).pack(pady=10)


def run_gui():
    root = tk.Tk()
    app = App(root)
    root.mainloop()