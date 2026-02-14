import json
import time
import threading
from pathlib import Path
import pyautogui
import faiss
import numpy as np
from . import config
from .models import EmbeddingModel

# 尝试导入键盘监听库用于中断（改用 pynput 全局热键）
from pynput import keyboard as pynput_keyboard

# 尝试导入图像识别所需库
try:
    import cv2
    import pyautogui as pag
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("警告: 未安装 opencv-python，图像识别功能不可用。请执行: pip install opencv-python")

# 全局中断标志（用于 pynput 热键）
_interrupt_flag = False

def on_activate_interrupt():
    """热键回调函数"""
    global _interrupt_flag
    print("中断热键 Ctrl+F3 被按下")
    _interrupt_flag = True

def start_hotkey_listener():
    """启动全局热键监听（在单独线程中运行）"""
    with pynput_keyboard.GlobalHotKeys({
            '<ctrl>+<f3>': on_activate_interrupt}) as listener:
        listener.join()

def check_interrupt():
    """检查是否收到中断信号，并重置标志"""
    global _interrupt_flag
    if _interrupt_flag:
        _interrupt_flag = False
        return True
    return False

# 启动热键监听线程（在模块导入时自动启动）
hotkey_thread = threading.Thread(target=start_hotkey_listener, daemon=True)
hotkey_thread.start()


class MouseRecorder:
    """鼠标轨迹录制器（支持快捷键）"""
    def __init__(self, on_stop_callback=None):
        self.recording = False
        self.events = []   # 存储(时间戳, x, y, event_type)
        self.listener = None
        self.hotkey_listener = None
        self.start_time = 0
        self.on_stop_callback = on_stop_callback  # 停止时的回调函数
        
    def _on_move(self, x, y):
        """鼠标移动事件处理"""
        if self.recording:
            self.events.append((time.time() - self.start_time, x, y, "move"))
    
    def _on_click(self, x, y, button, pressed):
        """鼠标点击事件处理"""
        if self.recording:
            btn = str(button).split('.')[-1]  # left, right, middle
            event = f"click_{btn}_{pressed}"
            self.events.append((time.time() - self.start_time, x, y, event))
            print(f"✅ 点击事件已记录: {event} at ({x},{y})")  # 确保用户看到
    
    def _on_press(self, key):
        """键盘快捷键处理（仅用于 F2 停止录制）"""
        try:
            if key == pynput_keyboard.Key.f2 and self.recording:
                print("快捷键 F2 触发停止录制")
                self.stop()
                return False  # 停止监听器
        except AttributeError:
            pass
    
    def start(self):
        """开始录制"""
        self.events = []
        self.recording = True
        self.start_time = time.time()
        
        # 启动鼠标监听
        from pynput import mouse
        self.listener = mouse.Listener(
            on_move=self._on_move, 
            on_click=self._on_click
        )
        self.listener.start()
        
        # 启动键盘监听（仅用于 F2）
        from pynput import keyboard
        self.hotkey_listener = keyboard.Listener(on_press=self._on_press)
        self.hotkey_listener.start()
        print("🎥 开始录制鼠标轨迹... (按 F2 停止)")
    
    def stop(self):
        """停止录制"""
        self.recording = False
        if self.listener:
            self.listener.stop()
        if self.hotkey_listener:
            self.hotkey_listener.stop()
        print(f"⏹️ 录制结束，共 {len(self.events)} 个事件")
        
        # 调用回调函数通知 GUI
        if self.on_stop_callback:
            self.on_stop_callback(len(self.events))
        
        return self.events
    
    def save(self, name: str) -> str:
        """保存轨迹到JSON文件，返回文件路径"""
        if not self.events:
            raise ValueError("没有录制数据可保存")
        
        file_path = config.RECORDINGS_DIR / f"{name}.json"
        with open(file_path, "w", encoding='utf-8') as f:
            json.dump(self.events, f, indent=2, ensure_ascii=False)
        print(f"💾 轨迹已保存到: {file_path}")
        return str(file_path)
    
    def get_events_count(self) -> int:
        """获取当前录制的事件数量"""
        return len(self.events)
    
    def has_recorded_data(self) -> bool:
        """检查是否有录制数据"""
        return len(self.events) > 0


class BehaviorVectorStore:
    """行为向量存储 - 专门用于存储自动化流程的元数据"""
    def __init__(self):
        self.index_path = config.VECTOR_STORE_DIR / "behaviors.faiss"
        self.meta_path = config.VECTOR_STORE_DIR / "behaviors.json"
        self.emb_model = EmbeddingModel()
        
        self.index = None
        self.metadata = []  # [{name, description, file_path, created_at}]
        self._load_or_create()
    
    def _load_or_create(self):
        """加载或创建新的索引"""
        if self.index_path.exists() and self.meta_path.exists():
            try:
                self.index = faiss.read_index(str(self.index_path))
                with open(self.meta_path, "r", encoding='utf-8') as f:
                    self.metadata = json.load(f)
                print(f"📚 已加载行为库: {len(self.metadata)} 个行为")
            except Exception as e:
                print(f"加载行为库失败: {e}，创建新的")
                self.index = faiss.IndexFlatIP(self.emb_model.dimension)
                self.metadata = []
        else:
            print("✨ 创建新的行为库")
            self.index = faiss.IndexFlatIP(self.emb_model.dimension)
            self.metadata = []
    
    def save(self):
        """保存索引和元数据"""
        try:
            faiss.write_index(self.index, str(self.index_path))
            with open(self.meta_path, "w", encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)
            print(f"💾 行为库已保存: {len(self.metadata)} 个行为")
        except Exception as e:
            print(f"保存行为库失败: {e}")
            raise
    
    def add_behavior(self, name: str, description: str, file_path: str):
        """添加行为到向量库"""
        # 构造搜索文本：名称+描述
        search_text = f"{name}: {description}"
        
        # 生成向量
        vector = self.emb_model.encode([search_text])
        
        # 添加到索引
        self.index.add(vector)
        
        # 添加元数据
        self.metadata.append({
            "name": name,
            "description": description,
            "file_path": file_path,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        })
        
        self.save()
        print(f"✅ 行为 '{name}' 已存入向量库")
    
    def search(self, query: str, top_k=5):
        """搜索相似行为"""
        if len(self.metadata) == 0:
            return []
        
        query_vec = self.emb_model.encode([query])
        scores, indices = self.index.search(query_vec, min(top_k, len(self.metadata)))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and idx < len(self.metadata):
                results.append({
                    "index": int(idx),
                    "name": self.metadata[idx]["name"],
                    "description": self.metadata[idx]["description"],
                    "file_path": self.metadata[idx]["file_path"],
                    "created_at": self.metadata[idx]["created_at"],
                    "score": float(score)
                })
        return results
    
    def get_all_behaviors(self):
        """获取所有行为（最新在前）"""
        return list(reversed(self.metadata))
    
    def delete_behavior(self, index: int):
        """删除指定索引的行为"""
        if index < 0 or index >= len(self.metadata):
            return False
        
        # 删除元数据
        deleted_name = self.metadata[index]["name"]
        del self.metadata[index]
        
        # 重建索引
        if self.metadata:
            texts = [f"{m['name']}: {m['description']}" for m in self.metadata]
            vectors = self.emb_model.encode(texts)
            self.index = faiss.IndexFlatIP(self.emb_model.dimension)
            self.index.add(vectors)
        else:
            self.index = faiss.IndexFlatIP(self.emb_model.dimension)
        
        self.save()
        print(f"🗑️ 行为 '{deleted_name}' 已删除")
        return True
    
    def clear_all(self):
        """清空所有行为"""
        self.index = faiss.IndexFlatIP(self.emb_model.dimension)
        self.metadata = []
        self.save()
        print("🧹 所有行为已清空")


class BehaviorLibrary:
    """行为知识库：统一的对外接口"""
    def __init__(self):
        self.store = BehaviorVectorStore()
    
    def add_behavior(self, name: str, description: str, file_path: str):
        """添加行为"""
        self.store.add_behavior(name, description, file_path)
    
    def search_behavior(self, query: str, threshold=config.BEHAVIOR_SIMILARITY_THRESHOLD, top_k=5):
        """检索相似行为"""
        results = self.store.search(query, top_k=top_k)
        # 过滤低于阈值的
        filtered = [r for r in results if r["score"] >= threshold]
        return filtered
    
    def get_all(self):
        """获取所有行为"""
        return self.store.get_all_behaviors()
    
    def delete(self, index: int):
        """删除行为"""
        return self.store.delete_behavior(index)
    
    def clear_all(self):
        """清空所有行为"""
        self.store.clear_all()


# ------------------------------------------------------------
# 脚本步骤构建与执行
# ------------------------------------------------------------

class ScriptBuilder:
    """脚本构建器，用于手动创建自动化步骤"""
    def __init__(self):
        self.steps = []  # 每个步骤为字典，包含 type 和参数
    
    def add_move(self, x, y, duration=0.2):
        """添加移动步骤"""
        self.steps.append({
            "type": "move",
            "x": x,
            "y": y,
            "duration": duration
        })
    
    def add_click(self, x=None, y=None, button='left', clicks=1):
        """添加点击步骤，如果未提供坐标则使用当前位置"""
        self.steps.append({
            "type": "click",
            "x": x,
            "y": y,
            "button": button,
            "clicks": clicks
        })
    
    def add_image_click(self, image_path, confidence=0.8, button='left'):
        """添加图像识别点击步骤"""
        if not HAS_CV2:
            raise RuntimeError("图像识别功能不可用，请安装 opencv-python")
        self.steps.append({
            "type": "image_click",
            "image_path": image_path,
            "confidence": confidence,
            "button": button
        })
    
    def add_wait(self, seconds):
        """添加等待步骤"""
        self.steps.append({
            "type": "wait",
            "seconds": seconds
        })
    
    def add_typewrite(self, text, interval=0.1):
        """添加键盘输入步骤"""
        self.steps.append({
            "type": "typewrite",
            "text": text,
            "interval": interval
        })
    
    def save(self, file_path):
        """保存脚本到 JSON 文件"""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.steps, f, indent=2, ensure_ascii=False)
        print(f"💾 脚本已保存到: {file_path}")
    
    def load(self, file_path):
        """从 JSON 文件加载脚本"""
        with open(file_path, 'r', encoding='utf-8') as f:
            self.steps = json.load(f)
        return self.steps


def replay_mouse(file_path: str, speed_factor: float = 1.0):
    """
    回放鼠标轨迹或脚本，支持速度因子和中断 (Ctrl+F3)
    
    :param file_path: 轨迹文件路径（JSON 格式）
    :param speed_factor: 速度因子，>1 加速，<1 减速，=1 原速
    """
    try:
        with open(file_path, "r", encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        return

    # 自动判断文件类型：如果 data 是列表且第一个元素是字典且有 'type' 键，则视为脚本
    if isinstance(data, list) and data and isinstance(data[0], dict) and 'type' in data[0]:
        # 新格式：脚本
        execute_script(data, speed_factor)
        return

    # 否则视为传统轨迹事件列表
    events = data
    if not events:
        print("轨迹文件为空")
        return

    print(f"▶️ 开始回放轨迹，速度因子: {speed_factor:.1f}x，共 {len(events)} 个事件 (按 Ctrl+F3 中断)")
    # 以第一个事件时间为基准
    base_time = events[0][0]
    for i, ev in enumerate(events):
        if check_interrupt():
            print("⏸️ 回放已中断")
            break
        ts, x, y, ev_type = ev
        # 等待至正确时间点（考虑速度因子）
        delay = (ts - base_time) / speed_factor
        if delay > 0:
            time.sleep(delay)
        
        if ev_type == "move":
            pyautogui.moveTo(x, y)
        elif ev_type.startswith("click"):
            parts = ev_type.split('_')
            button = parts[1]
            pressed = parts[2] == "True"
            btn_map = {"left": "left", "right": "right", "middle": "middle"}
            if pressed:
                pyautogui.mouseDown(button=btn_map.get(button, "left"))
            else:
                pyautogui.mouseUp(button=btn_map.get(button, "left"))
        # 忽略其他类型
    print("✅ 回放完成" if i == len(events)-1 else "⏸️ 回放中断")


def execute_script(steps, speed_factor=1.0):
    """
    执行脚本步骤列表
    """
    print(f"▶️ 开始执行脚本，速度因子: {speed_factor:.1f}x，共 {len(steps)} 步 (按 Ctrl+F3 中断)")
    for i, step in enumerate(steps):
        if check_interrupt():
            print("⏸️ 脚本执行中断")
            break
        step_type = step.get("type")
        try:
            if step_type == "move":
                x = step["x"]
                y = step["y"]
                duration = step.get("duration", 0.2) / speed_factor
                pyautogui.moveTo(x, y, duration=duration)
            elif step_type == "click":
                x = step.get("x")
                y = step.get("y")
                button = step.get("button", "left")
                clicks = step.get("clicks", 1)
                if x is not None and y is not None:
                    pyautogui.click(x, y, button=button, clicks=clicks)
                else:
                    pyautogui.click(button=button, clicks=clicks)
            elif step_type == "image_click":
                if not HAS_CV2:
                    print("图像识别不可用，跳过该步骤")
                    continue
                image_path = step["image_path"]
                confidence = step.get("confidence", 0.8)
                button = step.get("button", "left")
                try:
                    location = pyautogui.locateOnScreen(image_path, confidence=confidence)
                    if location:
                        center = pyautogui.center(location)
                        pyautogui.click(center, button=button)
                    else:
                        print(f"❌ 未在屏幕上找到图像: {image_path}")
                except Exception as e:
                    print(f"图像识别失败: {e}")
            elif step_type == "wait":
                seconds = step["seconds"] / speed_factor
                time.sleep(seconds)
            elif step_type == "typewrite":
                text = step["text"]
                interval = step.get("interval", 0.1) / speed_factor
                pyautogui.typewrite(text, interval=interval)
            else:
                print(f"未知步骤类型: {step_type}")
        except Exception as e:
            print(f"执行步骤 {i+1} 时出错: {e}")
    print("✅ 脚本执行完成" if i == len(steps)-1 else "⏸️ 脚本执行中断")