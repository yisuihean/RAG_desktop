import json
import time
from pathlib import Path
import threading
from pynput import mouse
import pyautogui
import faiss
import numpy as np
from . import config
from .models import EmbeddingModel
from .document_processor import DocumentVectorStore

class MouseRecorder:
    """鼠标录制器（后台线程）"""
    def __init__(self):
        self.recording = False
        self.events = []   # 存储(时间戳, x, y, event_type)
        self.listener = None
    
    def _on_move(self, x, y):
        if self.recording:
            self.events.append((time.time(), x, y, "move"))
    
    def _on_click(self, x, y, button, pressed):
        if self.recording:
            btn = str(button).split('.')[-1]  # left, right, middle
            self.events.append((time.time(), x, y, f"click_{btn}_{pressed}"))
    
    def start(self):
        self.events = []
        self.recording = True
        self.listener = mouse.Listener(on_move=self._on_move, on_click=self._on_click)
        self.listener.start()
    
    def stop(self):
        self.recording = False
        if self.listener:
            self.listener.stop()
        return self.events
    
    def save(self, name: str) -> str:
        """保存轨迹到JSON文件，返回文件路径"""
        file_path = config.RECORDINGS_DIR / f"{name}.json"
        with open(file_path, "w") as f:
            json.dump(self.events, f, indent=2)
        return str(file_path)

class BehaviorLibrary:
    """行为知识库：存储轨迹描述向量与文件路径映射"""
    def __init__(self):
        self.vector_store = DocumentVectorStore("behaviors")
        self.emb_model = EmbeddingModel()
    
    def add_behavior(self, name: str, description: str, file_path: str):
        """添加行为：描述文本向量化，元数据保存路径和名称"""
        text = f"{name}: {description}"  # 索引文本
        self.vector_store.add_documents([text], [str(file_path)])  # 使用source存储文件路径
    
    def search_behavior(self, query: str, threshold=config.BEHAVIOR_SIMILARITY_THRESHOLD):
        """检索相似行为，返回(文件路径, 相似度)"""
        results = self.vector_store.search(query, top_k=1)
        if results and results[0]["score"] >= threshold:
            return results[0]["source"], results[0]["score"]
        return None, 0.0

def replay_mouse(file_path: str):
    """回放鼠标轨迹"""
    with open(file_path, "r") as f:
        events = json.load(f)
    if not events:
        return
    # 以第一个事件时间为基准
    base_time = events[0][0]
    for ev in events:
        ts, x, y, ev_type = ev
        # 等待至正确时间点（相对延迟）
        delay = ts - base_time
        time.sleep(delay)
        if ev_type == "move":
            pyautogui.moveTo(x, y)
        elif ev_type.startswith("click"):
            # 解析click_left_True 等
            parts = ev_type.split('_')
            button = parts[1]  # left/right/middle
            pressed = parts[2] == "True"
            btn_map = {"left": pyautogui.PRIMARY, "right": pyautogui.SECONDARY, "middle": pyautogui.MIDDLE}
            if pressed:
                pyautogui.mouseDown(button=btn_map.get(button, pyautogui.PRIMARY))
            else:
                pyautogui.mouseUp(button=btn_map.get(button, pyautogui.PRIMARY))