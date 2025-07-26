import tkinter as tk
from tkinter import ttk, messagebox, filedialog, Menu, Spinbox, StringVar
import os
import re
import json
import csv
import random
import webbrowser
from datetime import datetime, timedelta
from collections import defaultdict
from decimal import Decimal, ROUND_HALF_UP
import ttkbootstrap as tb
from ttkbootstrap.constants import *
from PIL import Image, ImageTk

# ==============================================================================
# S級 Super AI 頂尖依賴庫
# ==============================================================================
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import warnings
warnings.filterwarnings('ignore')
try:
    import tensorflow as tf
    try:
        from keras.models import Sequential
        from keras.layers import LSTM, Dense, Softmax, Dropout, MultiHeadAttention
        TF_AVAILABLE = True
    except ImportError:
        TF_AVAILABLE = False
        print("TensorFlow Keras模組未安裝，將使用簡化版深度學習")
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow未安裝，將使用簡化版深度學習")

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch未安裝，將使用替代強化學習")

# ======= AutoML 參數搜尋函式組（★ 新增） =======
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

# --- 方輔助/模型函式區域 ---
from keras.callbacks import EarlyStopping
import torch.optim as optim
import shap

# ==============================================================================
# Super AI 智能龍判斷預測核心 - 地球最強AI決策引擎
# ==============================================================================
# ==============================================================================
# Excel風格表格控件 - 地球最強AI特製版
# ==============================================================================
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(64, input_shape=input_shape, return_sequences=True),
        Dropout(0.3),
        LSTM(32),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model
# === End build_lstm_model ===
class CyberpunkStyleTable:
    """賽博龐克風格表格控件 - 地球最強AI炫酷版"""
    
    def __init__(self, parent, columns, data=None):
        self.parent = parent
        self.columns = columns
        self.data = data if data else []
        self.current_font_size = 11
        
        # 創建主框架
        self.main_frame = ttk.Frame(parent)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 設置賽博龐克風格樣式
        self.setup_cyberpunk_style()
        
        # 創建表格
        self.create_cyberpunk_table()
        
        # 載入數據
        if self.data:
            self.load_data(self.data)
    
    def setup_cyberpunk_style(self):
        """設置賽博龐克風格樣式"""
        self.style = ttk.Style()
        
        # 賽博龐克標題樣式 - 深色漸層背景
        self.style.configure("Cyberpunk.Treeview.Heading",
                           font=("Consolas", 12, "bold"),
                           background="#0a1f3d",        # 深藍背景
                           foreground="#00ffff",        # 青色文字
                           relief="raised",
                           borderwidth=2)
        
        # 賽博龐克表格樣式 - 深色網格
        self.style.configure("Cyberpunk.Treeview",
                           font=("Consolas", 11),
                           background="#1a1a2e",        # 深色背景
                           foreground="#eee",           # 淺色文字
                           fieldbackground="#16213e",   # 欄位背景
                           borderwidth=1,
                           relief="solid")
        
        # 選中行樣式 - 賽博龐克青色
        self.style.map("Cyberpunk.Treeview",
                      background=[('selected', '#00ffff')],
                      foreground=[('selected', '#000000')])
        
        # 配置網格線樣式
        self.style.configure("Cyberpunk.Treeview", 
                           bordercolor="#00ffff",      # 青色邊框
                           lightcolor="#0066cc",       # 亮邊顏色
                           darkcolor="#003366")        # 暗邊顏色
    
    def create_cyberpunk_table(self):
        """創建賽博龐克風格表格"""
        # 創建表格容器（賽博龐克邊框）
        table_container = tk.Frame(self.main_frame, 
                                 bg="#0a1f3d", 
                                 relief="ridge", 
                                 borderwidth=2)
        table_container.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        # 創建Treeview（使用賽博龐克樣式）
        self.tree = ttk.Treeview(
            table_container, 
            columns=self.columns, 
            show="headings",
            style="Cyberpunk.Treeview",
            selectmode="extended"
        )
        
        # 配置列標題和寬度
        column_widths = {
            "局數": 100, "總點數": 100, "結果": 90, "原判斷": 100,
            "Super AI 預測": 180, "命中狀態": 250,
            "關數": 120, "本關投入": 150, "若贏得總派彩": 180,
            "若贏得淨利潤": 180, "累計總投入": 150
        }
        
        for col in self.columns:
            self.tree.heading(col, text=f"▸ {col} ◂", anchor="center")  # 賽博龐克符號
            col_width = column_widths.get(col, max(100, len(str(col)) * 12 + 40))
            self.tree.column(col, 
                           width=col_width,
                           minwidth=80,
                           anchor="center",
                           stretch=True)
        
        # 創建賽博龐克風格滾動條
        v_scrollbar = ttk.Scrollbar(table_container, 
                                   orient="vertical", 
                                   command=self.tree.yview)
        self.tree.configure(yscrollcommand=v_scrollbar.set)
        
        h_scrollbar = ttk.Scrollbar(table_container, 
                                   orient="horizontal", 
                                   command=self.tree.xview)
        self.tree.configure(xscrollcommand=h_scrollbar.set)
        
        # 使用Grid佈局，添加賽博龐克邊框
        self.tree.grid(row=0, column=0, sticky="nsew", padx=1, pady=1)
        v_scrollbar.grid(row=0, column=1, sticky="ns")
        h_scrollbar.grid(row=1, column=0, sticky="ew")
        
        # 配置網格權重
        table_container.grid_rowconfigure(0, weight=1)
        table_container.grid_columnconfigure(0, weight=1)
        
        # 添加右鍵選單和事件綁定
        self.create_context_menu()
        self.tree.bind("<Double-1>", self.on_double_click)
        self.tree.bind("<Button-3>", self.show_context_menu)
        
        # 設置賽博龐克顏色標籤
        self.setup_cyberpunk_color_tags()

    def update_font_size(self, font_size):
        """動態更新表格字體大小 - 賽博龐克版"""
        try:
            self.current_font_size = font_size
            optimal_row_height = int(font_size * 2.8)  # 稍微增加行高
            
            # 更新賽博龐克樣式
            self.style.configure("Cyberpunk.Treeview.Heading",
                               font=("Consolas", font_size + 1, "bold"),
                               background="#0a1f3d",
                               foreground="#00ffff",
                               relief="raised",
                               borderwidth=2)
            
            self.style.configure("Cyberpunk.Treeview",
                               font=("Consolas", font_size),
                               background="#1a1a2e",
                               foreground="#eee",
                               fieldbackground="#16213e",
                               borderwidth=1,
                               relief="solid",
                               rowheight=optimal_row_height)
            
            # 強制更新樹狀控制項樣式
            if hasattr(self, 'tree') and self.tree:
                self.tree.configure(style="Cyberpunk.Treeview")
            
            # 更新所有顏色標籤的字體
            self.update_color_tags_font(font_size)
            
            # 自動調整欄寬
            self.auto_adjust_columns_for_font(font_size)
            
            print(f"賽博龐克表格字體更新到 {font_size}pt 成功")
            
        except Exception as e:
            print(f"賽博龐克表格字體更新失敗: {e}")

    def setup_cyberpunk_color_tags(self):
        """設置賽博龐克顏色標籤"""
        color_configs = {
            # 結果顏色 - 更鮮豔的賽博龐克色彩
            "big_result": ("#ff3366", "Consolas", 11, "bold"),      # 賽博龐克紅
            "small_result": ("#33ff99", "Consolas", 11, "bold"),    # 賽博龐克綠  
            "leopard_result": ("#ff9900", "Consolas", 11, "bold"),  # 賽博龐克橙
            
            # 龍判斷顏色 - 霓虹效果
            "dragon_follow_hit": ("#00ff00", "Consolas", 11, "bold"),       # 霓虹綠
            "dragon_kill_hit": ("#00ffff", "Consolas", 11, "bold"),        # 霓虹青  
            "dragon_assist_hit": ("#ff00ff", "Consolas", 11, "bold"),      # 霓虹紫
            "dragon_block_hit": ("#66ff66", "Consolas", 11, "bold"),       # 亮綠色
            "dragon_follow_miss": ("#ff6666", "Consolas", 11, "bold"),     # 亮紅色
            "dragon_kill_miss": ("#ff0033", "Consolas", 11, "bold"),       # 深紅色
            "ai_hit": ("#33ccff", "Consolas", 11, "bold"),                 # 亮藍色
            "leopard_kill": ("#ffaa00", "Consolas", 11, "bold"),           # 金色
            "ai_miss": ("#ff4499", "Consolas", 11, "bold"),                # 粉紅色
            
            # 背景行顏色 - 深色主題
            "even_row": ("#1a1a2e", "Consolas", 11, "normal"),             # 深色偶數行
            "odd_row": ("#252545", "Consolas", 11, "normal")               # 稍亮奇數行
        }
        
        for tag_name, (color, font_family, font_size, font_weight) in color_configs.items():
            if tag_name in ["odd_row", "even_row"]:
                # 背景色標籤
                self.tree.tag_configure(tag_name, 
                                       background=color, 
                                       foreground="#eee",  # 統一文字顏色
                                       font=(font_family, font_size, font_weight))
            else:
                # 文字色標籤 - 保持深色背景
                self.tree.tag_configure(tag_name, 
                                       foreground=color,
                                       background="#1a1a2e",  # 保持深色背景
                                       font=(font_family, font_size, font_weight))

    def update_color_tags_font(self, font_size):
        """更新所有顏色標籤的字體大小"""
        color_tags = [
            "big_result", "small_result", "leopard_result",
            "dragon_follow_hit", "dragon_kill_hit", "dragon_assist_hit",
            "dragon_block_hit", "dragon_follow_miss", "dragon_kill_miss",
            "ai_hit", "leopard_kill", "ai_miss", "even_row", "odd_row"
        ]
        
        for tag in color_tags:
            if tag in ["odd_row", "even_row"]:
                self.tree.tag_configure(tag, font=("Consolas", font_size, "normal"))
            else:
                self.tree.tag_configure(tag, font=("Consolas", font_size, "bold"))

    def auto_adjust_columns_for_font(self, font_size):
        """根據字體大小自動調整欄寬"""
        font_factor = font_size / 11
        
        base_widths = {
            "局數": 100, "總點數": 100, "結果": 90, "原判斷": 100,
            "Super AI 預測": 180, "命中狀態": 250,
            "關數": 120, "本關投入": 150, "若贏得總派彩": 180,
            "若贏得淨利潤": 180, "累計總投入": 150
        }
        
        for col in self.columns:
            base_width = base_widths.get(col, 120)
            new_width = int(base_width * font_factor)
            self.tree.column(col, width=new_width, minwidth=int(new_width * 0.7))
    
    def create_context_menu(self):
        """創建賽博龐克風格右鍵選單"""
        self.context_menu = tk.Menu(self.tree, tearoff=0, 
                                   bg="#1a1a2e", fg="#00ffff",
                                   activebackground="#00ffff", 
                                   activeforeground="#000000")
        self.context_menu.add_command(label="📋 複製數據", command=self.copy_selected)
        self.context_menu.add_command(label="📊 詳細分析", command=self.view_details)
        self.context_menu.add_separator()
        self.context_menu.add_command(label="🔄 重新載入", command=self.refresh_table)
    
    def load_data(self, data):
        """載入數據到表格"""
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        for i, row in enumerate(data):
            row_data = list(row)
            while len(row_data) < len(self.columns):
                row_data.append("")
            
            item_id = self.tree.insert("", "end", values=row_data)
            self.apply_row_colors(item_id, row_data, i)
    
    def apply_row_colors(self, item_id, row_data, row_index):
        """根據資料內容套用賽博龐克顏色"""
        base_tags = []
        if row_index % 2 == 0:
            base_tags.append("even_row")
        else:
            base_tags.append("odd_row")
        
        status_tag = None
        if len(row_data) > 5:
            hit_status = str(row_data[5]).strip()
            
            if "跟龍命中" in hit_status:
                status_tag = "dragon_follow_hit"
            elif "斬龍命中" in hit_status:
                status_tag = "dragon_kill_hit"
            elif "助龍命中" in hit_status:
                status_tag = "dragon_assist_hit"
            elif "阻龍命中" in hit_status:
                status_tag = "dragon_block_hit"
            elif "跟龍失誤" in hit_status:
                status_tag = "dragon_follow_miss"
            elif "斬龍失誤" in hit_status:
                status_tag = "dragon_kill_miss"
            elif "豹子通殺" in hit_status:
                status_tag = "leopard_kill"
            elif "Super AI 命中" in hit_status:
                status_tag = "ai_hit"
            elif "Super AI 未命中" in hit_status:
                status_tag = "ai_miss"
        
        if not status_tag and len(row_data) > 2:
            result = str(row_data[2])
            if result == "大":
                status_tag = "big_result"
            elif result == "小":
                status_tag = "small_result"
            elif result == "豹子":
                status_tag = "leopard_result"
        
        final_tags = [status_tag] if status_tag else base_tags
        self.tree.item(item_id, tags=final_tags)
    
    def auto_adjust_columns(self):
        """自動調整列寬"""
        for col in self.columns:
            max_width = len(str(col)) * 12
            
            for child in self.tree.get_children():
                values = self.tree.item(child)["values"]
                col_index = self.columns.index(col)
                if col_index < len(values):
                    content_width = len(str(values[col_index])) * 8
                    max_width = max(max_width, content_width)
            
            self.tree.column(col, width=min(max_width + 20, 400))
    
    def on_double_click(self, event):
        """雙擊事件處理"""
        item = self.tree.selection()[0] if self.tree.selection() else None
        if item:
            values = self.tree.item(item, "values")
            print(f"賽博龐克表格雙擊: {values}")
    
    def show_context_menu(self, event):
        """顯示賽博龐克右鍵選單"""
        try:
            self.context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.context_menu.grab_release()
    
    def copy_selected(self):
        """複製選中的行"""
        selected_items = self.tree.selection()
        if selected_items:
            copied_data = []
            for item in selected_items:
                values = self.tree.item(item, "values")
                copied_data.append("\t".join(str(v) for v in values))
            
            self.parent.clipboard_clear()
            self.parent.clipboard_append("\n".join(copied_data))
            print("賽博龐克數據已複製到剪貼簿")
    
    def view_details(self):
        """查看詳情"""
        selected_item = self.tree.selection()[0] if self.tree.selection() else None
        if selected_item:
            values = self.tree.item(selected_item, "values")
            print(f"賽博龐克詳細分析: {values}")
    
    def refresh_table(self):
        """刷新表格"""
        if self.data:
            self.load_data(self.data)
            self.auto_adjust_columns()

    def update_color_tags_font(self, font_size):
        """更新所有顏色標籤的字體大小"""
        color_tags = [
            "big_result", "small_result", "leopard_result",
            "dragon_follow_hit", "dragon_kill_hit", "dragon_assist_hit",
            "dragon_block_hit", "dragon_follow_miss", "dragon_kill_miss",
            "ai_hit", "leopard_kill", "ai_miss", "even_row", "odd_row"
        ]
        
        for tag in color_tags:
            current_config = self.tree.tag_configure(tag)
            if current_config:
                # 保持原有顏色，只更新字體大小
                if tag in ["odd_row", "even_row"]:
                    self.tree.tag_configure(tag, font=("Consolas", font_size, "normal"))
                else:
                    self.tree.tag_configure(tag, font=("Consolas", font_size, "bold"))

    def auto_adjust_columns_for_font(self, font_size):
        """根據字體大小自動調整欄寬"""
        # 基礎欄寬乘以字體係數
        font_factor = font_size / 11  # 以11pt為基準
        
        base_widths = {
            "局數": 100, "總點數": 100, "結果": 90, "原判斷": 100,
            "Super AI 預測": 180, "命中狀態": 250,
            "關數": 120, "本關投入": 150, "若贏得總派彩": 180,
            "若贏得淨利潤": 180, "累計總投入": 150
        }
        
        for col in self.columns:
            base_width = base_widths.get(col, 120)
            new_width = int(base_width * font_factor)
            self.tree.column(col, width=new_width, minwidth=int(new_width * 0.7))
    
    def setup_color_tags(self):
        """設置所有顏色標籤 - 完美修復版"""
        color_configs = {
            "big_result": ("#e74c3c", "Consolas", 11, "bold"),
            "small_result": ("#3498db", "Consolas", 11, "bold"), 
            "leopard_result": ("#f39c12", "Consolas", 11, "bold"),
            "dragon_follow_hit": ("#27ae60", "Consolas", 11, "bold"),      # 跟龍命中
            "dragon_kill_hit": ("#0dd6d6", "Consolas", 11, "bold"),       # 斬龍命中  
            "dragon_assist_hit": ("#d391f0", "Consolas", 11, "bold"),     # 助龍命中
            "dragon_block_hit": ("#5c9564", "Consolas", 11, "bold"),      # 阻龍命中
            "dragon_follow_miss": ("#fcb7af", "Consolas", 11, "bold"),    # 跟龍失誤
            "dragon_kill_miss": ("#ff1900", "Consolas", 11, "bold"),      # 斬龍失誤
            "ai_hit": ("#3498db", "Consolas", 11, "bold"),                # Super AI 命中
            "leopard_kill": ("#f39c12", "Consolas", 11, "bold"),          # 豹子通殺
            "ai_miss": ("#ff4161", "Consolas", 11, "bold"),               # Super AI 未命中
            # ★★★ 新增：奇偶行背景標籤 ★★★
            "even_row": ("#FFFFFF", "Consolas", 11, "normal"),            # 偶數行背景
            "odd_row": ("#F8F8F8", "Consolas", 11, "normal")              # 奇數行背景
        }
        
        for tag_name, (color, font_family, font_size, font_weight) in color_configs.items():
            if tag_name in ["odd_row", "even_row"]:
                # 背景色標籤
                self.tree.tag_configure(tag_name, background=color, font=(font_family, font_size, font_weight))
            else:
                # 文字色標籤
                self.tree.tag_configure(tag_name, 
                                    foreground=color, 
                                    font=(font_family, font_size, font_weight))
    
    def create_context_menu(self):
        """創建右鍵選單"""
        self.context_menu = tk.Menu(self.tree, tearoff=0)
        self.context_menu.add_command(label="📋 複製", command=self.copy_selected)
        self.context_menu.add_command(label="📊 查看詳情", command=self.view_details)
        self.context_menu.add_separator()
        self.context_menu.add_command(label="🔄 刷新", command=self.refresh_table)
    
    def load_data(self, data):
        """載入數據到表格"""
        # 清空現有數據
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # 載入新數據
        for i, row in enumerate(data):
            row_data = list(row)
            while len(row_data) < len(self.columns):
                row_data.append("")
            
            # 插入數據並應用顏色
            item_id = self.tree.insert("", "end", values=row_data)
            self.apply_row_colors(item_id, row_data, i)
    
    def apply_row_colors(self, item_id, row_data, row_index):
        """根據資料內容套用顏色 - 徹底修復版"""
        
        # ★★★ 第一步：確保每行都有基礎背景色 ★★★
        base_tags = []
        if row_index % 2 == 0:
            base_tags.append("even_row")  # 偶數行：白色背景
        else:
            base_tags.append("odd_row")   # 奇數行：淡灰背景
        
        # ★★★ 第二步：檢查命中狀態，優先級最高 ★★★
        status_tag = None
        if len(row_data) > 5:  # 命中狀態在第6列
            hit_status = str(row_data[5]).strip()
            
            # 按優先級檢查命中狀態
            if "跟龍命中" in hit_status:
                status_tag = "dragon_follow_hit"
            elif "斬龍命中" in hit_status:
                status_tag = "dragon_kill_hit"
            elif "助龍命中" in hit_status:
                status_tag = "dragon_assist_hit"
            elif "阻龍命中" in hit_status:
                status_tag = "dragon_block_hit"
            elif "跟龍失誤" in hit_status:
                status_tag = "dragon_follow_miss"
            elif "斬龍失誤" in hit_status:
                status_tag = "dragon_kill_miss"
            elif "豹子通殺" in hit_status:
                status_tag = "leopard_kill"
            elif "Super AI 命中" in hit_status:
                status_tag = "ai_hit"
            elif "Super AI 未命中" in hit_status:
                status_tag = "ai_miss"
        
        # ★★★ 第三步：如果沒有命中狀態，檢查結果顏色 ★★★
        if not status_tag and len(row_data) > 2:
            result = str(row_data[2])
            if result == "大":
                status_tag = "big_result"
            elif result == "小":
                status_tag = "small_result"
            elif result == "豹子":
                status_tag = "leopard_result"
        
        # ★★★ 第四步：套用最終標籤 ★★★
        if status_tag:
            # 如果有特殊狀態，使用狀態顏色（覆蓋背景色）
            final_tags = [status_tag]
        else:
            # 如果沒有特殊狀態，使用基礎背景色
            final_tags = base_tags
        
        # 套用標籤到表格行
        self.tree.item(item_id, tags=final_tags)
    
    def auto_adjust_columns(self):
        """自動調整列寬"""
        for col in self.columns:
            max_width = len(str(col)) * 12
            
            for child in self.tree.get_children():
                values = self.tree.item(child)["values"]
                col_index = self.columns.index(col)
                if col_index < len(values):
                    content_width = len(str(values[col_index])) * 8
                    max_width = max(max_width, content_width)
            
            self.tree.column(col, width=min(max_width + 20, 400))
    
    def on_double_click(self, event):
        """雙擊事件處理"""
        item = self.tree.selection()[0] if self.tree.selection() else None
        if item:
            values = self.tree.item(item, "values")
            print(f"雙擊行數據: {values}")
    
    def show_context_menu(self, event):
        """顯示右鍵選單"""
        try:
            self.context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.context_menu.grab_release()
    
    def copy_selected(self):
        """複製選中的行"""
        selected_items = self.tree.selection()
        if selected_items:
            copied_data = []
            for item in selected_items:
                values = self.tree.item(item, "values")
                copied_data.append("\t".join(str(v) for v in values))
            
            self.parent.clipboard_clear()
            self.parent.clipboard_append("\n".join(copied_data))
            print("已複製到剪貼簿")
    
    def view_details(self):
        """查看詳情"""
        selected_item = self.tree.selection()[0] if self.tree.selection() else None
        if selected_item:
            values = self.tree.item(selected_item, "values")
            print(f"查看詳情: {values}")
    
    def refresh_table(self):
        """刷新表格"""
        if self.data:
            self.load_data(self.data)
            self.auto_adjust_columns()

class DQNAgent(nn.Module):
    def __init__(self, state_dim, action_dim=2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    def forward(self, x):
        return self.fc(x)
    # ─────────────────────────────────────
class SLevelAIPredictor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {}
        self.meta_model = GradientBoostingClassifier(n_estimators=50, learning_rate=0.2, max_depth=4, random_state=42)
        self.model_lstm = None
        self.rl_agent = None
        self._histories = {}
        self.is_trained = False
        self.data = []
        # 新增初始化
        self.feature_mask = None  # 修復 AttributeError

    def build_and_train_lstm(self, X, y, epochs: int =50, batch_size: int =32):
        from keras.callbacks import EarlyStopping
        X_seq = X.reshape((X.shape[0], X.shape[1], 1))
        model = build_lstm_model(input_shape=(X.shape[1],1))
        es = EarlyStopping(monitor='val_loss', patience=5,
                        restore_best_weights=True)
        model.fit(X_seq, y,
                validation_split=0.2,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[es],
                verbose=0)
        return model

    def train_rl_agent(self, X, y, episodes: int =200):
        import torch
        import torch.nn as nn
        import torch.optim as optim
        # 簡易 DQN 網絡
        class DQNAgent(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(dim,64), nn.ReLU(),
                    nn.Linear(64,64), nn.ReLU(),
                    nn.Linear(64,2)
                )
            def forward(self, x): return self.net(x)
        agent = DQNAgent(X.shape[1])
        opt = optim.Adam(agent.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()
        for ep in range(episodes):
            idxs = np.random.permutation(len(X))
            for i in idxs:
                state = torch.tensor(X[i], dtype=torch.float32)
                q = agent(state)
                target = q.clone().detach()
                a = int(y[i])
                reward = 1.0 if (q.argmax().item()==a and a==y[i]) else -1.0
                target[a] = reward
                loss = loss_fn(q, target)
                opt.zero_grad(); loss.backward(); opt.step()
        self.rl_agent = agent
        print("✅ RL agent 完成")

    def incremental_retrain(self, new_data, window_size: int=120, batch_size: int=25):
        if not hasattr(self,'data'): self.data=[]
        self.data.extend(new_data)
        if len(self.data)%batch_size!=0: return
        recent = self.data[-window_size:]
        Xw, yw = self.prepare_training_data(recent)
        if Xw is None or len(Xw)<15: return
        Xw_s = self.scaler.fit_transform(Xw)
        for k,m in self.models.items():
            if hasattr(m,'partial_fit'):
                try: m.partial_fit(Xw_s,yw)
                except: pass
        try:
            self.model_lstm = self.build_and_train_lstm(Xw_s,yw,epochs=5,batch_size=16)
        except: pass
        print("🔄 動態再訓練完成")

    def train_models(self, historical_data):
        from sklearn.inspection import permutation_importance
        self.models = {}
        try:
            print("🚀 開始 SSS級 AI 訓練…")
            phases = self.split_by_phase(historical_data)
            for phase, data in phases.items():
                if len(data) >= 20:
                    Xp, yp = self.prepare_training_data(data)
                    if Xp is None:
                        continue
                    Xp = self.scaler.fit_transform(Xp)
                    setattr(self, f"model_{phase}", self._build_best_rf(Xp, yp))
                    print(f"✅ 階段 '{phase}' 子模型訓練完成")
            X, y = self.prepare_training_data(historical_data)
            if X is None or len(X) < 15:  # 調整為15 (對應20局)
                print("❌ 資料不足（至少15個有效樣本）")
                return False
            if self.feature_mask is not None and len(self.feature_mask) == X.shape[1]:
                X = X[:, self.feature_mask]
                print(f"✅ 套用特徵遮罩：剩餘 {X.shape[1]} 特徵")
            X_scaled = self.scaler.fit_transform(X)
            self.models['random_forest'] = self._build_best_rf(X_scaled, y)
            self.models['gradient_boost'] = self._build_best_gbdt(X_scaled, y)
            self.models['svm'] = self._build_best_svm(X_scaled, y)
            self.models['neural_network'] = self._build_best_mlp(X_scaled, y)
            base_preds = []
            for name, model in self.models.items():
                model.fit(X_scaled, y)
                preds = model.predict_proba(X_scaled)[:,1] if hasattr(model, 'predict_proba') else model.predict(X_scaled)
                base_preds.append(preds)
                print(f"✅ {name} 模型訓練完成")
            self.model_lstm = self.build_and_train_lstm(X_scaled, y)
            lstm_preds = self.model_lstm.predict(X_scaled.reshape(-1, X.shape[1], 1)).flatten()
            base_preds.append(lstm_preds)
            print("✅ LSTM 子模型訓練並加入 stacking")
            meta_features = np.column_stack(base_preds)
            self.meta_model.fit(meta_features, y)
            print("✅ Meta 模型訓練完成")
            rf = self.models['random_forest']
            importances = permutation_importance(rf, X_scaled, y, n_repeats=3, random_state=42).importances_mean
            if len(importances) > 5:
                low_idx = np.argsort(importances)[:2]
                mask = np.ones(X_scaled.shape[1], dtype=bool)
                mask[low_idx] = False
                self.feature_mask = mask
                print(f"🔍 特徵遮罩更新：剔除 {low_idx.tolist()}")
            self.is_trained = True
            print("🎉 SSS級 AI 訓練完成！")
            return True
        except Exception as e:
            print(f"❌ SSS級 AI 訓練錯誤：{e}")
            return False

    def predict_with_confidence(self, historical_results):
        """
        SSS 預測：分流+LSTM+RL+正則化多頭注意力融合
        """
        if not getattr(self,'is_trained',False):
            return self.fallback_prediction(historical_results)
        try:
            import tensorflow as tf
            from keras.layers import MultiHeadAttention, Dense, Dropout, Softmax
            import torch

            feats = extract_advanced_features(historical_results)
            F = self.scaler.transform(feats)
            # 基模型機率
            p_rf   = self.models['random_forest'].predict_proba(F)[0,1]
            p_lstm = self.model_lstm.predict(F.reshape(1,-1,1))[0,0]
            p_rl   = getattr(self,'rl_agent',None)
            if p_rl is None:
                p_rl = 0.5
            else:
                p_rl = p_rl(torch.tensor(F[0],dtype=torch.float32)).detach()
                p_rl = float(torch.softmax(p_rl,dim=0)[1].item())

            # 多頭注意力
            mi = tf.constant([[p_rf,p_lstm,p_rl]],dtype=tf.float32)  # (1,3)
            mha = MultiHeadAttention(num_heads=3,key_dim=4,dropout=0.2)
            attn_out = mha(mi,mi,mi)                                 # (1,3)
            x = Dense(32,activation='relu',kernel_regularizer='l2')(attn_out)
            x = Dropout(0.25)(x)
            scores = Dense(3,activation='tanh',kernel_regularizer='l2')(x)
            weights = Softmax()(scores)                              # (1,3)
            fused = tf.reduce_sum(weights * mi, axis=1).numpy()[0]

            pred = "大" if fused>0.5 else "小"
            conf = float(fused*100)

            # 更新歷史
            for name, prob in zip(['random_forest','lstm','rl_agent'],[p_rf,p_lstm,p_rl]):
                h = int((prob>0.5)==(1 if pred=="大" else 0))
                self._histories.setdefault(name,[]).append(h)
                if len(self._histories[name])>50: self._histories[name].pop(0)

            # 解釋
            wt = weights.numpy()[0]
            expl = (
                f"🤖 預測:{pred}\n"
                f"🎯 信心:{conf:.1f}%\n"
                f"🔍 權重: RF{wt[0]*100:.1f}%, LSTM{wt[1]*100:.1f}%, RL{wt[2]*100:.1f}%"
            )
            return pred, int(conf), expl

        except Exception as e:
            print(f"SSS級預測錯誤: {e}")
            return self.fallback_prediction(historical_results)
        
    # ======= 動態加權工具函式（★ 新增） =======
    def _init_histories(self):
        """於 __init__ 最下方呼叫一次"""
        self._histories = {m: [] for m in self.models}

    def _update_model_weights(self, window: int = 20) -> dict:
        """依 rolling window 返回最新權重 dict"""
        hit_rates = {}
        for m, history in self._histories.items():
            recent = history[-window:]
            hit_rates[m] = sum(recent) / window if recent else 0.5
        total = sum(hit_rates.values())
        if total == 0:
            # 避免除零，回到均分
            return {m: 1/len(self.models) for m in self.models}
        return {m: hit_rates[m]/total for m in self.models}
    # ======= 動態加權工具函式結束 =======
            
    def initialize_models(self, X=None, y=None):
        """
        SS級AutoML智能初始化：若給定X, y則自動尋優建立最佳子模型，否則回退舊預設模型參數組（作為保險）。
        呼叫方式：先確認有數據時多數情境train_models內會傳X,y過來；無數據初始化則走預設。
        """
        # 若無自動優化資料，僅作基礎備援(理論上僅debug用)
        if X is None or y is None or len(X) == 0 or len(y) == 0:
            self.models = {
                'random_forest': RandomForestClassifier(
                    n_estimators=120, max_depth=8, random_state=42, class_weight='balanced'
                ),
                'gradient_boost': GradientBoostingClassifier(
                    n_estimators=120, learning_rate=0.1, max_depth=4, random_state=42
                ),
                'svm': SVC(
                    kernel='rbf', C=1.0, probability=True, random_state=42
                ),
                'neural_network': MLPClassifier(
                    hidden_layer_sizes=(80, 40), max_iter=500, learning_rate_init=0.001,
                    random_state=42, early_stopping=True
                )
            }
        else:
            from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
            def auto_optimize_model(base, param_dist):
                cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                rs = RandomizedSearchCV(base, param_distributions=param_dist, n_iter=8, scoring='accuracy', cv=cv, n_jobs=-1, random_state=42)
                if not self.meta_model:
                    self.meta_model = GradientBoostingClassifier(
                        n_estimators=50,
                        learning_rate=0.2,
                        max_depth=4,
                        random_state=42
                    )

                rs.fit(X, y)
                return rs.best_estimator_

            self.models = {
                'random_forest': auto_optimize_model(
                    RandomForestClassifier(random_state=42), 
                    {'n_estimators':[80,120,160], 'max_depth':[6,8,10,None], 'min_samples_split':[2,4,6], 'class_weight':['balanced', None]}
                ),
                'gradient_boost': auto_optimize_model(
                    GradientBoostingClassifier(random_state=42), 
                    {'n_estimators':[80,120,160], 'learning_rate':[0.05,0.1,0.15], 'max_depth':[3,4,5]}
                ),
                'svm': auto_optimize_model(
                    SVC(probability=True, random_state=42), 
                    {'C':[0.5,1,2], 'gamma':['scale',0.1,0.01], 'kernel':['rbf','poly']}
                ),
                'neural_network': auto_optimize_model(
                    MLPClassifier(max_iter=600,random_state=42,early_stopping=True), 
                    {'hidden_layer_sizes':[(50,30),(80,40),(100,)], 'learning_rate_init':[0.001,0.003], 'alpha':[0.0001,0.001]}
                )
            }
        
        # Meta模型（用於Stacking）
        self.meta_model = GradientBoostingClassifier(
            n_estimators=50,
            learning_rate=0.2,
            max_depth=4,
            random_state=42
        )

    def _auto_optimize_model(self, base_model, param_grid, X, y, n_iter=8):
        from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
        try:
            cv_folds = max(2, min(3, len(y) // 10))  # 小資料時降低CV折數
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            searcher = RandomizedSearchCV(
                base_model,
                param_distributions=param_grid,
                n_iter=n_iter,
                scoring='accuracy',
                cv=cv,
                n_jobs=1,
                random_state=42,
                error_score='raise'
            )
            searcher.fit(X, y)
            return searcher.best_estimator_
        except Exception as e:
            print(f"⚠️ AutoML優化失敗: {e}，將使用預設模型")
            base_model.fit(X, y)
            return base_model

    def _build_best_rf(self, X, y):
        param_dist = {
            'n_estimators': [80, 120, 160],
            'max_depth': [6, 8, 10, None],
            'min_samples_split': [2, 4, 6],
            'class_weight': ['balanced', None]
        }
        return self._auto_optimize_model(RandomForestClassifier(random_state=42), param_dist, X, y)

    def _build_best_gbdt(self, X, y):
        param_dist = {
            'n_estimators': [80, 120, 160],
            'learning_rate': [0.05, 0.1, 0.15],
            'max_depth': [3, 4, 5]
        }
        return self._auto_optimize_model(GradientBoostingClassifier(random_state=42), param_dist, X, y)

    def _build_best_svm(self, X, y):
        param_dist = {
            'C': [0.5, 1, 2],
            'gamma': ['scale', 0.1, 0.01],
            'kernel': ['rbf', 'poly']
        }
        return self._auto_optimize_model(SVC(probability=True, random_state=42), param_dist, X, y)

    def _build_best_mlp(self, X, y):
        param_dist = {
            'hidden_layer_sizes': [(50, 30), (80, 40), (100,)],
            'learning_rate_init': [0.001, 0.003],
            'alpha': [0.0001, 0.001]
        }
        return self._auto_optimize_model(MLPClassifier(max_iter=600, random_state=42, early_stopping=True),
                                        param_dist, X, y)
    
    # ======= AutoML 函式組結束 =======
    # ★★★ 替代版 prepare_training_data（貼在原函式整段替換）★★★
    def prepare_training_data(self, historical_data):
        """準備訓練資料：最少 7 局即可開始產生 X,y"""
        if len(historical_data) < 7:           # ← 調整
            return None, None
        X, y = [], []
        for i in range(5, len(historical_data)):
            seq = [row[2] for row in historical_data[:i] if row[2] in ('大','小')]
            if len(seq) < 5:                   # ← 調整
                continue
            feats = extract_advanced_features(seq, self.feature_mask)
            y.append(1 if historical_data[i][2]=='大' else 0)
            X.append(feats.flatten())
        if len(X) < 5:                         # ← 調整
            return None, None
        return np.array(X), np.array(y)
       
    # 在 SLevelAIPredictor 類別中，新增方法 split_by_phase
    def split_by_phase(self, historical_data):
        """
        根據局數或階段分流：
        - 開局 (1~10) → model_open
        - 中盤 (11~30) → model_mid
        - 尾盤 (>30) → model_end
        """
        phases = {'open': [], 'mid': [], 'end': []}
        for rec in historical_data:
            idx = rec[0]
            if idx <= 10:
                phases['open'].append(rec)
            elif idx <= 30:
                phases['mid'].append(rec)
            else:
                phases['end'].append(rec)
        return phases

    # 定義輔助函式 dragon_sensitive_predict
    def dragon_sensitive_predict(results, main_model, dragon_model):
        streak = 1
        for i in range(len(results)-1, 0, -1):
            if results[i] == results[i-1]:
                streak += 1
            else:
                break
        feats = extract_advanced_features(results)
        if streak >= 4:
            return dragon_model.predict(feats)
        else:
            return main_model.predict(feats)
        
    
    def generate_explanation(self, prediction, confidence, model_confidences, historical_results):
        """生成S級AI解釋"""
        # 分析當前狀況
        current_streak = calculate_current_streak(historical_results)
        dragon_strength = calculate_dragon_strength(historical_results)
        
        # 確定主導模型
        dominant_model = max(model_confidences.items(), key=lambda x: x[1])
        
        # 生成解釋文本
        explanation = f"🤖 S級AI集成預測：{prediction}\n"
        explanation += f"🎯 主導模型：{self.get_model_display_name(dominant_model[0])} (信心度 {dominant_model[1]:.1f}%)\n"
        
        if current_streak >= 4:
            explanation += f"🐲 檢測到{current_streak}連龍勢，龍強度：{dragon_strength:.1f}%\n"
        elif current_streak == 3:
            explanation += f"⚡ 3連準龍狀態，AI建議：{prediction}\n"
        else:
            explanation += f"📊 當前{current_streak}連，多模型分析傾向：{prediction}\n"
        
        # 風險建議
        if confidence >= 75:
            explanation += "💎 高信心預測，建議下注 2-3%"
        elif confidence >= 65:
            explanation += "⚖️ 中等信心，建議下注 1-2%"
        else:
            explanation += "⚠️ 謹慎觀察，建議下注 0.5-1%"
        
        return explanation
    
    def get_model_display_name(self, model_name):
        """獲取模型顯示名稱"""
        names = {
            'random_forest': '隨機森林',
            'gradient_boost': '梯度提升',
            'svm': '支持向量機',
            'neural_network': '神經網路'
        }
        return names.get(model_name, model_name)
    
    def fallback_prediction(self, historical_results):
        """備用預測（當AI未訓練時）"""
        if not historical_results:
            return "大", 55, "🔄 數據不足，使用保守策略"
        
        # 使用原有的簡化邏輯
        current_streak = calculate_current_streak(historical_results)
        
        if current_streak >= 4:
            pred = "小" if historical_results[-1] == "大" else "大"
            return pred, 70, f"🛡️ {current_streak}連龍勢，建議斬龍"
        elif current_streak == 3:
            pred = "小" if historical_results[-1] == "大" else "大"
            return pred, 75, f"⚔️ 3連準龍，AI建議阻龍"
        else:
            pred = "小" if historical_results[-1] == "大" else "大"
            return pred, 65, f"🔄 {current_streak}連交錯，保守預測"

def super_ai_prediction(last_results):
        """
        S級 Super AI 預測引擎 - 完全重構版
        整合多模型集成、深度學習、強化學習的頂尖AI系統
        """
        global s_level_ai
        
        if not last_results or len(last_results) < 2:
            return "大", 55, "🔄 數據收集中，採用保守策略"
        
        # 自動訓練檢查（當數據足夠時）
        if not s_level_ai.is_trained and len(last_results) >= 20:
            print("🚀 S級AI開始訓練...")
            # 這裡需要完整的歷史數據，暫時使用可用數據
            mock_data = [[i, 0, result, "", "", "", "", None] for i, result in enumerate(last_results)]
            training_success = s_level_ai.train_models(mock_data)
            if training_success:
                print("✅ S級AI訓練完成！")
            else:
                print("⚠️ S級AI訓練失敗，使用備用模式")
        
        # 執行S級預測
        prediction, confidence, explanation = s_level_ai.predict_with_confidence(last_results)
        
        return prediction, confidence, explanation

# ======= 特徵擴展助手（★強化版） =======
def _create_extra_features(seq: list[int]) -> list[float]:
        """
        自動產生高維度遊戲序列特徵
        - 反轉率（最後10局）
        - 最大連續相同值長度（最後10局）
        seq: 0/1序列 (1=大, 0=小)
        """
        if len(seq) < 10:
            seq = [0] * (10 - len(seq)) + seq

        last10 = seq[-10:]
        # 反轉率
        reversals = sum(1 for i in range(1, len(last10)) if last10[i] != last10[i-1])
        # 最大連續
        max_gap, cur_gap = 0, 0
        prev = last10[0]
        for x in last10[1:]:
            if x == prev:
                cur_gap += 1
            else:
                max_gap = max(max_gap, cur_gap)
                cur_gap = 0
                prev = x
        max_gap = max(max_gap, cur_gap)

        return [reversals / 9, max_gap / 9]
    # ======= 特徵擴展助手結束 =======

def get_dragon_break_features(results, min_dragon_len=4):
    """
    計算斷龍相關特徵，輸入 ['大','小',...]回傳[斷龍次數, 斷龍後反向命中比率]
    """
    breaks = 0
    post_break_hit = 0
    streak = 1
    for i in range(1, len(results)):
        if results[i] == results[i-1]:
            streak += 1
        else:
            if streak >= min_dragon_len:
                breaks += 1
                post_break_hit += int(results[i] != results[i-1])
            streak = 1
    ratio = post_break_hit / breaks if breaks > 0 else 0.5
    return [breaks, ratio]

def complex_composite_features(results):
    nums = [1 if x == '大' else 0 for x in results]
    avg10 = np.mean(nums[-10:]) if len(nums) >= 10 else 0.5
    streak = calculate_current_streak(results)
    max_streak = calculate_max_streak(results, '大')
    return [avg10 * streak, streak * max_streak]

def extract_advanced_features(historical_results, window_sizes=[5,10,15,20], feature_mask=None):
    """
    SS級特徵工程 - 支援動態遮罩
    """
    dummy_len = 27 if feature_mask is None else int(np.sum(feature_mask))
    if len(historical_results) < 5:
        return np.zeros(dummy_len).reshape(1,-1)
    features = []
    # 結果轉數值(大=1, 小=0)
    results_numeric = [1 if r=='大' else 0 for r in historical_results if r in ['大','小']]
    if len(results_numeric) < 5:
        return np.zeros(dummy_len).reshape(1,-1)

    # 1.基礎統計
    features.append(np.mean(results_numeric[-10:]))
    features.append(np.std(results_numeric[-10:]) if len(results_numeric) > 1 else 0)
    features.append(len(results_numeric))

    # 2.連莊特徵（需你自行定義以下輔助函數）
    current_streak = calculate_current_streak(historical_results)
    max_streak_big = calculate_max_streak(historical_results, '大')
    max_streak_small = calculate_max_streak(historical_results, '小')
    features.extend([current_streak, max_streak_big, max_streak_small])
    features.extend(get_dragon_break_features(historical_results))
    features.extend(complex_composite_features(historical_results))

    # 3.多滑動視窗
    for window in window_sizes:
        if len(results_numeric) >= window:
            window_data = results_numeric[-window:]
            features.append(np.mean(window_data))
            features.append(np.std(window_data) if len(window_data) > 1 else 0)
            features.append(sum(1 for i in range(1,len(window_data)) if window_data[i] != window_data[i-1]))
        else:
            features.extend([0.5,0.5,0])

    # 4.趨勢差異
    if len(results_numeric) >= 10:
        recent_5 = np.mean(results_numeric[-5:])
        recent_10 = np.mean(results_numeric[-10:])
        features.append(recent_5 - recent_10)
    else:
        features.append(0)

    # 5.週期性及龍勢指標 (需自行實作)
    features.append(calculate_periodicity(results_numeric))
    features.append(calculate_dragon_strength(historical_results))

    # 6. 擴展特徵
    features.extend(_create_extra_features(results_numeric))

    # 7. 遮罩特徵
    feat_array = np.array(features)
    if feature_mask is not None and len(feature_mask) == len(feat_array):
        feat_array = feat_array[feature_mask]
    return feat_array.reshape(1,-1)

def calculate_current_streak(results):
    """計算當前連莊長度"""
    if not results:
        return 0
    
    current = results[-1]
    streak = 1
    for i in range(len(results) - 2, -1, -1):
        if results[i] == current:
            streak += 1
        else:
            break
    return streak

def calculate_max_streak(results, target):
    """計算指定結果的最長連莊"""
    max_streak = 0
    current_streak = 0
    
    for result in results:
        if result == target:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0
    
    return max_streak

def calculate_periodicity(results_numeric):
    """計算週期性指標"""
    if len(results_numeric) < 6:
        return 0
    
    # 簡單的週期性檢測
    patterns = []
    for period in [2, 3, 4]:
        if len(results_numeric) >= period * 2:
            correlation = 0
            for i in range(period, len(results_numeric)):
                if results_numeric[i] == results_numeric[i-period]:
                    correlation += 1
            patterns.append(correlation / (len(results_numeric) - period))
    
    return max(patterns) if patterns else 0

def calculate_dragon_strength(results):
    """計算龍勢強度"""
    if len(results) < 4:
        return 0
    
    # 統計最近的龍出現情況
    dragon_count = 0
    i = 0
    while i < len(results) - 3:
        current = results[i]
        streak = 1
        j = i + 1
        while j < len(results) and results[j] == current:
            streak += 1
            j += 1
        
        if streak >= 4:  # 真龍
            dragon_count += 1
        
        i = j if j > i + 1 else i + 1
    
    # 根據龍密度計算強度
    total_segments = len(results) // 10 + 1
    return min(dragon_count / max(total_segments, 1) * 100, 100)

def handle_true_dragon(streak, streak_type, analysis, last_results):
    """
    處理真正的龍（4連以上）- AI智能決策
    """
    dragon_strength = analysis['dragon_strength']
    
    if streak >= 8:  # 超級長龍
        # 超長龍通常會斷，但要謹慎
        if dragon_strength > 20:  # 龍勢仍強
            pred = streak_type  # 繼續跟龍
            conf = 60
            desc = f"🐲 偵測到{streak}連超級長龍！AI分析龍勢仍強，謹慎跟進。建議下注0.5%。"
        else:  # 龍勢轉弱
            pred = '小' if streak_type == '大' else '大'
            conf = 75
            desc = f"⚔️ {streak}連超級長龍，AI分析龍勢轉弱，智能斬龍。建議下注2%。"
    
    elif streak >= 6:  # 長龍
        if dragon_strength > 0 and analysis['market_trend'] == 'pro_dragon':
            pred = streak_type  # 跟龍
            conf = 70
            desc = f"🔥 {streak}連長龍持續，AI判斷市場支持龍勢。穩健跟龍，建議下注1.5%。"
        else:
            pred = '小' if streak_type == '大' else '大'
            conf = 78
            desc = f"🎯 {streak}連長龍，AI智能分析建議斬龍時機。建議下注2.5%。"
    
    elif streak >= 4:  # 標準龍
        if dragon_strength > 10:  # 龍勢強
            pred = streak_type
            conf = 75
            desc = f"🐉 {streak}連真龍降臨！AI分析龍勢強勁，黃金跟龍期。建議下注2-3%。"
        elif dragon_strength < -10:  # 反龍信號
            pred = '小' if streak_type == '大' else '大'
            conf = 72
            desc = f"⚡ {streak}連龍勢，但AI偵測反龍信號。智能斬龍策略。建議下注2%。"
        else:  # 中性
            # 根據其他因子判斷
            if analysis['player_psychology'] > 60:
                pred = '小' if streak_type == '大' else '大'
                conf = 68
                desc = f"🤔 {streak}連龍勢中性，AI感知玩家焦慮，建議斬龍。下注1.5%。"
            else:
                pred = streak_type
                conf = 65
                desc = f"📈 {streak}連龍勢穩定，AI建議繼續跟龍觀察。下注1-2%。"
    
    return pred, conf, desc

def handle_pre_dragon(streak_type, analysis, last_results):
    """
    處理準龍狀態（3連）- 優先預設反打，但在極端強龍狀態允許AI智能例外
    """
    dragon_strength = analysis['dragon_strength']
    # 絕大多數三連，都是反打
    if dragon_strength > 50:  # 歷史龍絕少且市場異常強龍，再考慮跟龍
        pred = streak_type
        conf = 80
        desc = f"🚀 罕見超強龍勢（AI智能判斷），准許跟龍。建議2%。"
    else:
        pred = '小' if streak_type == '大' else '大'
        conf = 85
        desc = f"🛡️ 3連準龍，AI按既定策略反打阻龍。建議3-4%。"
    return pred, conf, desc

def handle_double_streak(streak_type, analysis, last_results):
    """
    處理雙連狀態（2連）- 龍萌芽期
    """
    dragon_strength = analysis['dragon_strength']
    
    if dragon_strength > 15:
        pred = streak_type
        conf = 70
        desc = f"📊 2連龍萌芽，AI分析環境利於龍勢發展。建議下注1.5%。"
    elif dragon_strength < -15:
        pred = '小' if streak_type == '大' else '大'
        conf = 68
        desc = f"🔒 2連遇阻力，AI判斷不利龍勢發展。建議下注1.5%。"
    else:
        # 平衡策略
        pred = streak_type
        conf = 65
        desc = f"⚖️ 2連平衡期，AI建議保守跟進觀察。建議下注1%。"
    
    return pred, conf, desc

def handle_normal_pattern(streak_type, last_results):
    """
    處理正常模式（交錯或單次）
    """
    # 檢查特殊模式
    if len(last_results) >= 6:
        last_6 = last_results[-6:]
        
        # 完美單跳模式
        single_jump_patterns = [
            ["大", "小", "大", "小", "大", "小"],
            ["小", "大", "小", "大", "小", "大"]
        ]
        
        if last_6 in single_jump_patterns:
            next_in_pattern = "大" if streak_type == "小" else "小"
            return next_in_pattern, 76, f"🔀 AI偵測完美單跳模式，按規律預測{next_in_pattern}。建議下注2%。"
        
        # 雙跳模式
        double_jump_patterns = [
            ["大", "大", "小", "小", "大", "大"],
            ["小", "小", "大", "大", "小", "小"]
        ]
        
        if last_6 in double_jump_patterns:
            next_pred = "小" if streak_type == "大" else "大"
            return next_pred, 73, f"🔄 AI偵測雙跳模式，預測{next_pred}。建議下注1.5%。"
    
    # 偏態修正
    recent_10 = last_results[-10:] if len(last_results) >= 10 else last_results
    big_count = recent_10.count("大")
    small_count = recent_10.count("小")
    bias = abs(big_count - small_count)
    
    if bias >= 4:
        if big_count > small_count:
            return '小', 68, f"🔄 最近10局『大』偏多({big_count}vs{small_count})，AI建議修正。建議下注1%。"
        else:
            return '大', 68, f"🔄 最近10局『小』偏多({small_count}vs{big_count})，AI建議修正。建議下注1%。"
    
    # 預設交錯策略
    pred = '小' if streak_type == '大' else '大'
    return pred, 63, f"🎲 常規交錯模式，AI保守預測{pred}。穩健下注1%。"

def count_historical_dragons(results):
    """統計歷史龍次數"""
    dragon_count = 0
    i = 0
    while i < len(results):
        if i + 3 < len(results):
            # 檢查是否有4連以上
            current = results[i]
            streak = 1
            j = i + 1
            while j < len(results) and results[j] == current:
                streak += 1
                j += 1
            
            if streak >= 4:  # 真正的龍
                dragon_count += 1
                i = j
            else:
                i += 1
        else:
            break
    
    return dragon_count

def analyze_market_cycle(results):
    """分析市場週期位置"""
    if len(results) < 10:
        return 'neutral'
    
    recent_10 = results[-10:]
    big_count = recent_10.count("大")
    
    if big_count >= 8:
        return 'peak'  # 大峰值
    elif big_count <= 2:
        return 'bottom'  # 小谷底
    else:
        return 'neutral'

def calculate_psychology_pressure(results, current_streak):
    """計算玩家心理壓力指數"""
    if current_streak >= 5:
        return 80  # 長龍高壓
    elif current_streak >= 3:
        return 60  # 中等壓力
    else:
        return 30  # 低壓力

def calculate_volatility(results):
    """計算波動性指數"""
    if len(results) < 5:
        return 0.5
    
    changes = 0
    for i in range(1, len(results)):
        if results[i] != results[i-1]:
            changes += 1
    
    return changes / (len(results) - 1)

def check_alternating_pattern(results):
    """檢查是否有明顯交錯模式"""
    if len(results) < 6:
        return False
    
    recent_6 = results[-6:]
    changes = 0
    for i in range(1, len(recent_6)):
        if recent_6[i] != recent_6[i-1]:
            changes += 1
    
    return changes >= 4  # 6局中有4次以上變化

# ===== 請確定這一行寫在"類體外層" =====
s_level_ai = SLevelAIPredictor()

# ==============================================================================
# Super AI 主程式類別 - 智能龍版
# ==============================================================================
class ModernGamblingTool:
    def __init__(self, root):
        self.root = root
        self.root.title("🛠 Game of Dice - Super AI ")
        self.root.geometry("1600x1000")
        
        # 核心數據
        self.data = []
        self.statistics = {}
        self.session_start_time = datetime.now()
        self.record_date = self.get_record_date()
        self.time_format = "%Y-%m-%d %H:%M:%S.%f"
        self.custom_image_path = None
        
        # Super AI 專用統計
        self.super_ai_predictions = []  # 儲存 (預測, 實際, 是否命中)
        self.dragon_statistics = {}     # 龍相關統計
        
        # 應用程式目錄設置
        self.app_dir = os.path.join(os.path.expanduser("~"), "GamblingTool")
        self.ensure_directories()
        self.data_dir = os.path.join(self.app_dir, "data")
        
        # UI變數
        self.theme_var = tk.StringVar(value="cyborg")
        self.status_var = tk.StringVar(value="🚀 Super AI 智能系統就緒")
        
        # 初始化UI
        self.init_ui()
        self.auto_save_timer = None
        self.auto_save()
        self.auto_load_last_session()

    # ✅ 正確：只保留一次定義
    def verify_system_integrity(self):
        """驗證系統完整性"""
        required_methods = [
            'load_data', 'save_data', 'update_stats_table_with_colors',
            'update_leopard_warning', 'copy_prediction', 'clear_all_data',
            'ensure_directories'
        ]
        
        missing_methods = []
        for method in required_methods:
            if not hasattr(self, method):
                missing_methods.append(method)
        
        if missing_methods:
            print(f"警告：缺少方法 {missing_methods}")
            messagebox.showwarning("系統完整性檢查", f"發現缺少以下方法：\n{missing_methods}")
        else:
            print("✅ 系統完整性驗證通過")

    def init_ui(self):
        """初始化用戶界面"""
        # 狀態欄
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # 主筆記本
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 三個主要分頁
        self.dice_frame = ttk.Frame(self.notebook)
        self.calculator_frame = ttk.Frame(self.notebook)
        self.stats_frame = ttk.Frame(self.notebook)
        
        self.notebook.add(self.dice_frame, text="🎲 Super AI 智能預測")
        self.notebook.add(self.calculator_frame, text="💰 賠率計算器")
        self.notebook.add(self.stats_frame, text="📊 歷史數據分析")
        
        # 初始化各分頁
        self.create_menu()
        self.init_dice_predictor()
        self.init_odds_calculator()
        self.init_stats_analyzer()
        
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)

    def create_menu(self):
        """創建菜單欄"""
        menubar = Menu(self.root)
        self.root.config(menu=menubar)
        
        # 檔案選單
        file_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="檔案", menu=file_menu)
        file_menu.add_command(label="新增會話", command=self.new_session)
        file_menu.add_command(label="儲存數據", command=self.save_data)
        file_menu.add_command(label="載入數據", command=self.load_data)
        file_menu.add_separator()
        file_menu.add_command(label="匯出報表", command=self.export_report)
        file_menu.add_separator()
        file_menu.add_command(label="離開", command=self.root.quit)
        
        # 工具選單
        tools_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="工具", menu=tools_menu)
        tools_menu.add_command(label="清除所有資料", command=self.clear_all_data)
        tools_menu.add_command(label="儲存計算結果", command=self.save_calculation)
        
        # 介面變更選單
        view_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="介面變更", menu=view_menu)
        
        # 佈景主題選單
        theme_menu = Menu(view_menu, tearoff=0)
        view_menu.add_cascade(label="佈景主題", menu=theme_menu)
        
        themes_map = {
            "cyborg": "賽博龐克",
            "darkly": "消光黑",
            "superhero": "湖水綠",
            "solar": "暖陽光線",
            "vapor": "紫色驚艷",
            "litera": "文學雅緻",
            "minty": "薄荷清新",
            "pulse": "簡約紫色",
        }
        
        for theme_name, display_name in themes_map.items():
            theme_menu.add_radiobutton(label=display_name, variable=self.theme_var, 
                                     value=theme_name, command=self.change_theme)
        
        # 幫助選單
        help_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="幫助", menu=help_menu)
        help_menu.add_command(label="使用說明", command=self.show_help)
        help_menu.add_command(label="關於", command=self.show_about)

    def add_font_control_panel(self, parent_frame):
        """添加字體控制面板"""
        # 創建字體控制框架
        font_control_frame = ttk.LabelFrame(parent_frame, text="🎨 字體調整")
        font_control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 字體大小控制
        font_size_frame = ttk.Frame(font_control_frame)
        font_size_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(font_size_frame, text="字體大小:").pack(side=tk.LEFT, padx=5)
        
        # 字體大小變數（載入使用者偏好）
        self.font_size_var = tk.IntVar(value=self.load_font_preference())
        
        # 字體大小滑桿
        font_scale = ttk.Scale(
            font_size_frame,
            from_=8,
            to=20,
            orient=tk.HORIZONTAL,
            variable=self.font_size_var,
            command=self.on_font_size_change
        )
        font_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        
        # 顯示當前字體大小
        self.font_size_label = ttk.Label(font_size_frame, text="11pt")
        self.font_size_label.pack(side=tk.LEFT, padx=5)
        
        # 預設/重設按鈕
        button_frame = ttk.Frame(font_control_frame)
        button_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(button_frame, text="📝 小字體 (10pt)", 
                command=lambda: self.set_font_size(10)).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="📄 預設 (11pt)", 
                command=lambda: self.set_font_size(11)).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="📰 大字體 (14pt)", 
                command=lambda: self.set_font_size(14)).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="🔍 超大字體 (16pt)", 
                command=lambda: self.set_font_size(16)).pack(side=tk.LEFT, padx=2)

    def on_font_size_change(self, value):
        """字體大小改變事件處理 - 強化版"""
        try:
            font_size = int(float(value))
            self.font_size_label.config(text=f"{font_size}pt")
            
            print(f"字體大小變更為: {font_size}pt")
            
            # ★★★ 立即更新所有表格的字體 ★★★
            self.update_all_table_fonts(font_size)
            
            # 儲存使用者偏好
            self.save_font_preference(font_size)
            
            # ★★★ 強制重新整理介面 ★★★
            self.root.update_idletasks()
            
        except Exception as e:
            print(f"字體大小變更處理失敗: {e}")
            messagebox.showerror("字體調整錯誤", f"調整字體時發生錯誤：{str(e)}")

    def set_font_size(self, size):
        """設定特定字體大小"""
        self.font_size_var.set(size)
        self.on_font_size_change(size)

    def update_all_table_fonts(self, font_size):
        """更新所有表格的字體大小 - 賽博龐克版"""
        print(f"正在更新賽博龐克表格字體到 {font_size}pt")
        
        # 更新詳細記錄表格
        if hasattr(self, 'cyberpunk_table') and self.cyberpunk_table:
            try:
                self.cyberpunk_table.update_font_size(font_size)
                print("賽博龐克詳細記錄表格字體更新成功")
            except Exception as e:
                print(f"賽博龐克詳細記錄表格字體更新失敗: {e}")
        
        # 更新賠率計算器表格
        if hasattr(self, 'cyberpunk_odds_table') and self.cyberpunk_odds_table:
            try:
                self.cyberpunk_odds_table.update_font_size(font_size)
                print("賽博龐克賠率計算器表格字體更新成功")
            except Exception as e:
                print(f"賽博龐克賠率計算器表格字體更新失敗: {e}")

    def update_cyberpunk_table(self):
        """更新賽博龐克風格表格數據"""
        if not hasattr(self, 'cyberpunk_table'):
            return
        
        table_data = []
        for row in self.data:
            row_data = [
                row[0] if len(row) > 0 else "",  # 局數
                row[1] if len(row) > 1 else "",  # 總點數
                row[2] if len(row) > 2 else "",  # 結果
                row[3] if len(row) > 3 else "",  # 原判斷
                row[5] if len(row) > 5 else "",  # Super AI 預測
                row[6] if len(row) > 6 else ""   # 命中狀態
            ]
            table_data.append(row_data)
        
        self.cyberpunk_table.data = table_data
        self.cyberpunk_table.load_data(table_data)
        self.cyberpunk_table.auto_adjust_columns()

    def load_font_preference(self):
        """載入使用者字體偏好"""
        try:
            pref_file = os.path.join(self.app_dir, "font_preference.json")
            if os.path.exists(pref_file):
                with open(pref_file, "r", encoding="utf-8") as f:
                    prefs = json.load(f)
                    return prefs.get("font_size", 11)
        except:
            pass
        return 11  # 預設字體大小

    def save_font_preference(self, font_size):
        """儲存使用者字體偏好"""
        try:
            pref_file = os.path.join(self.app_dir, "font_preference.json")
            prefs = {"font_size": font_size, "last_updated": datetime.now().isoformat()}
            with open(pref_file, "w", encoding="utf-8") as f:
                json.dump(prefs, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"儲存字體偏好失敗: {e}")

    def init_dice_predictor(self):
        """初始化骰子預測界面 - 完整修正版"""
        frame = self.dice_frame
        paned = ttk.PanedWindow(frame, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 左側框架
        frame_left = ttk.Frame(paned)
        paned.add(frame_left, weight=1)
        
        title_label = ttk.Label(frame_left, text="⚡ Super AI 智能系統 ⚡", 
                            font=("Consolas", 16, "bold"))
        title_label.pack(pady=5)
        
        self.text_area = tk.Text(frame_left, height=20, width=50, font=("Consolas", 12))
        self.text_area.pack(pady=5, fill=tk.BOTH, expand=True)
        self.text_area.insert(tk.END, "範例輸入格式：\n[第1局]第一骰: 4點, 第二骰: 3點, 第三骰: 3點, 【大】")
        
        # 左側按鈕區域
        button_frame_left_bottom = ttk.Frame(frame_left)
        button_frame_left_bottom.pack(fill=tk.X, pady=5)
        
        self.paste_analyze_button = ttk.Button(button_frame_left_bottom, text="🚀 Super AI 智能分析", 
                                            command=self.run_paste_and_analyze)
        self.paste_analyze_button.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        
        self.copy_button = ttk.Button(button_frame_left_bottom, text="📋 複製結果", 
                                    command=self.copy_prediction)
        self.copy_button.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        
        button_frame_left_bottom.columnconfigure(0, weight=1)
        button_frame_left_bottom.columnconfigure(1, weight=1)

        # ★★★ 右側框架 - 完整重建 ★★★
        frame_right = ttk.Frame(paned)
        paned.add(frame_right, weight=1)

        # 右側標題和預測顯示區域
        self.label_info = ttk.Label(frame_right, text="🐉 Super AI 智能系統待命", font=("Consolas", 12))
        self.label_info.pack(pady=5)
        
        self.predict_label = ttk.Label(frame_right, text="下一局預測： Super AI 分析中...", 
                                    font=("Consolas", 16, "bold"))
        self.predict_label.pack(pady=10)
        
        self.result_label = ttk.Label(frame_right, text="", font=("Consolas", 12), 
                                    wraplength=400, justify="left")
        self.result_label.pack(pady=10)
        
        self.leopard_warning_label = tk.Label(frame_right, text="尚無豹子出現", 
                                            font=("Arial", 12, "bold"))
        self.leopard_warning_label.pack(pady=10)
        
        # ★★★ S級AI控制面板 ★★★
        s_ai_frame = ttk.LabelFrame(frame_right, text="🚀 S級AI控制台")
        s_ai_frame.pack(fill=tk.X, pady=5)

        self.ai_status_label = ttk.Label(s_ai_frame, text="S級AI狀態: 待訓練", 
                                        font=("Consolas", 10))
        self.ai_status_label.pack(pady=2)

        ttk.Button(s_ai_frame, text="🔄 重新訓練AI", 
                command=self.retrain_s_level_ai).pack(pady=2)

        # ★★★ 數據管理控制面板 ★★★
        data_management_frame = ttk.LabelFrame(frame_right, text="📊 數據管理")
        data_management_frame.pack(fill=tk.X, pady=5)
        
        # 數據管理按鈕（2x2布局）
        data_button_frame = ttk.Frame(data_management_frame)
        data_button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        data_button_frame.columnconfigure(0, weight=1)
        data_button_frame.columnconfigure(1, weight=1)
        
        self.save_button = ttk.Button(data_button_frame, text="💾 保存數據", 
                                    command=self.save_data)
        self.save_button.grid(row=0, column=0, padx=2, pady=2, sticky="ew")
        
        self.load_button = ttk.Button(data_button_frame, text="📁 載入數據", 
                                    command=self.load_data)
        self.load_button.grid(row=0, column=1, padx=2, pady=2, sticky="ew")
        
        self.clear_button = ttk.Button(data_button_frame, text="🗑️ 清除所有局數", 
                                    command=self.clear_all_data)
        self.clear_button.grid(row=1, column=0, columnspan=2, padx=2, pady=2, sticky="ew")

        # ★★★ 自訂圖片區塊 ★★★
        image_frame = ttk.LabelFrame(frame_right, text="🖼️ 自訂圖片")
        image_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.custom_image_label = ttk.Label(image_frame, text="無自訂圖片", anchor="center")
        self.custom_image_label.pack(fill=tk.BOTH, expand=True, pady=5)
        
        image_button_frame = ttk.Frame(image_frame)
        image_button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        image_button_frame.columnconfigure(0, weight=1)
        image_button_frame.columnconfigure(1, weight=1)
        
        ttk.Button(image_button_frame, text="🖼️ 選擇圖片", 
                command=self.select_custom_image).grid(row=0, column=0, padx=2, pady=2, sticky="ew")
        ttk.Button(image_button_frame, text="❌ 清除圖片", 
                command=self.clear_custom_image).grid(row=0, column=1, padx=2, pady=2, sticky="ew")

    def check_ai_training_status(self):
        """檢查AI訓練狀態"""
        global s_level_ai
        
        try:
            if hasattr(s_level_ai, 'is_trained') and s_level_ai.is_trained:
                self.ai_status_label.config(text="S級AI狀態: 已訓練 ✅")
            else:
                if len(self.data) >= 20:
                    self.ai_status_label.config(text="S級AI狀態: 可訓練 📊")
                else:
                    self.ai_status_label.config(text="S級AI狀態: 待數據 📝")
        except Exception as e:
            self.ai_status_label.config(text="S級AI狀態: 檢查失敗 ❌")
            print(f"AI狀態檢查錯誤: {e}")

    def retrain_s_level_ai(self):
        """重新訓練S級AI"""
        global s_level_ai
        
        if len(self.data) >= 20:
            try:
                # 重置AI訓練狀態
                s_level_ai.is_trained = False
                
                # 顯示訓練進度
                self.ai_status_label.config(text="S級AI狀態: 訓練中... ⏳")
                self.root.update_idletasks()
                
                # 執行訓練
                success = s_level_ai.train_models(self.data)
                
                if success:
                    self.ai_status_label.config(text="S級AI狀態: 已訓練 ✅")
                    messagebox.showinfo("訓練完成", "S級AI重新訓練成功！\n命中率預期將有所提升。")
                    self.status_var.set("🚀 S級AI訓練完成，性能已優化")
                else:
                    self.ai_status_label.config(text="S級AI狀態: 訓練失敗 ❌")
                    messagebox.showwarning("訓練失敗", "S級AI訓練過程中發生錯誤\n將使用備用模式繼續運行")
                    self.status_var.set("⚠️ S級AI訓練失敗，使用備用模式")
                    
            except Exception as e:
                self.ai_status_label.config(text="S級AI狀態: 訓練錯誤 ❌")
                messagebox.showerror("訓練錯誤", f"S級AI訓練時發生錯誤：\n{str(e)}")
                print(f"S級AI訓練錯誤: {e}")
        else:
            messagebox.showwarning("數據不足", 
                                f"需要至少20局數據才能訓練S級AI\n"
                                f"目前數據量：{len(self.data)}局")

    def init_odds_calculator(self):
        """初始化賠率計算器（完美字體控制版）"""
        frame = self.calculator_frame

        # 說明區域
        info_frame = ttk.LabelFrame(frame, text="賠率計算說明")
        info_frame.pack(fill=tk.X, padx=10, pady=5)
        info_text = (
            "本計算器用於計算「固定倍率」的資金規劃策略。\n"
            "• 初始本金：第一關的投入金額。\n"
            "• 賠率：預設為 0.96 (獲利96%)。\n"
            "• 投注倍率：若輸，下一關投入金額為上一關的幾倍。"
        )
        ttk.Label(info_frame, text=info_text, justify=tk.LEFT, 
                font=("微軟正黑體", 12)).pack(padx=10, pady=5)

        # 設置區域
        settings_frame = ttk.Frame(frame)
        settings_frame.pack(fill=tk.X, padx=10, pady=10)

        row1 = ttk.Frame(settings_frame)
        row1.pack(fill=tk.X, pady=5)
        ttk.Label(row1, text="初始本金:").pack(side=tk.LEFT, padx=5)
        self.calc_entry = ttk.Entry(row1, width=10)
        self.calc_entry.pack(side=tk.LEFT, padx=5)
        self.calc_entry.insert(0, "10000")
        
        ttk.Label(row1, text="賠率:").pack(side=tk.LEFT, padx=5)
        self.rate_var = tk.StringVar(value="0.96")
        ttk.Entry(row1, textvariable=self.rate_var, width=5).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(row1, text="投注倍率 (x):").pack(side=tk.LEFT, padx=5)
        self.multiplier_entry = ttk.Entry(row1, width=5)
        self.multiplier_entry.pack(side=tk.LEFT, padx=5)
        self.multiplier_entry.insert(0, "3")
        
        ttk.Label(row1, text="關數:").pack(side=tk.LEFT, padx=5)
        self.levels_var = tk.StringVar(value="10")
        ttk.Spinbox(row1, from_=1, to=20, textvariable=self.levels_var, width=5).pack(side=tk.LEFT, padx=5)

        row2 = ttk.Frame(settings_frame)
        row2.pack(fill=tk.X, pady=5)
        ttk.Button(row2, text="計算賠率", command=self.calculate_odds).pack(side=tk.LEFT, padx=5)
        ttk.Button(row2, text="清除", command=self.clear_table).pack(side=tk.LEFT, padx=5)
        ttk.Button(row2, text="儲存結果", command=self.save_calculation).pack(side=tk.LEFT, padx=5)

        # 結果標籤
        self.calc_result_label = ttk.Label(frame, text="計算結果將顯示在這裡", 
                                        font=("微軟正黑體", 14, "bold"))
        self.calc_result_label.pack(pady=5)

        table_frame = ttk.Frame(frame)
        table_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        columns = ["關數", "本關投入", "若贏得總派彩", "若贏得淨利潤", "累計總投入"]
        # ★★★ 使用賽博龐克風格表格 ★★★
        self.cyberpunk_odds_table = CyberpunkStyleTable(table_frame, columns)

        # 保持兼容性
        self.tree = self.cyberpunk_odds_table.tree

        # 利潤顯示
        profit_frame = ttk.Frame(frame)
        profit_frame.pack(fill=tk.X, padx=10, pady=5)
        self.profit_var = tk.StringVar(value="淨利潤: --")
        ttk.Label(profit_frame, textvariable=self.profit_var, 
                font=("微軟正黑體", 12, "bold")).pack(side=tk.RIGHT)

        # ★★★ 新增：字體控制面板（在表格創建後） ★★★
        self.add_font_control_panel(frame)
    
        initial_font_size = self.load_font_preference()
        if initial_font_size != 11:
            self.root.after(100, lambda: self.cyberpunk_odds_table.update_font_size(initial_font_size))

    def init_stats_analyzer(self):
        """初始化統計分析器（含字體控制版）"""
        frame = self.stats_frame
        
        # 移除原有的分頁結構，直接使用單一界面
        main_frame = ttk.Frame(frame)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # ★★★ 新增：字體控制面板 ★★★
        self.add_font_control_panel(main_frame)

        # 頂部控制區域
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))

        # 範圍控制
        ttk.Label(control_frame, text="分析範圍:").pack(side=tk.LEFT, padx=5)
        self.range_var = tk.StringVar(value="全部")
        ranges = ["全部", "最近10局", "最近20局", "最近50局", "最近100局"]
        range_menu = ttk.Combobox(control_frame, textvariable=self.range_var, 
                                values=ranges, width=10, state="readonly")
        range_menu.pack(side=tk.LEFT, padx=5)
        range_menu.bind("<<ComboboxSelected>>", lambda e: self.update_statistics())
        
        # 篩選控制
        ttk.Label(control_frame, text="篩選:").pack(side=tk.LEFT, padx=(20, 5))
        self.filter_var = tk.StringVar(value="全部")
        filters = ["全部", "只顯示大", "只顯示小", "只顯示豹子"]
        filter_menu = ttk.Combobox(control_frame, textvariable=self.filter_var, 
                                values=filters, width=10, state="readonly")
        filter_menu.pack(side=tk.LEFT, padx=5)
        filter_menu.bind("<<ComboboxSelected>>", lambda e: self.filter_stats_data())
        
        ttk.Button(control_frame, text="🔄 更新統計", 
                command=self.update_statistics).pack(side=tk.LEFT, padx=10)

        # 創建主要內容區域（使用PanedWindow分割）
        paned_window = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        paned_window.pack(fill=tk.BOTH, expand=True)
        
        # 左側統計概覽
        stats_frame = ttk.Frame(paned_window)
        paned_window.add(stats_frame, weight=1)
        
        # 右側數據表
        data_frame = ttk.Frame(paned_window)
        paned_window.add(data_frame, weight=2)
        
        # 設置統計概覽
        self.setup_integrated_overview(stats_frame)
        
        # 設置數據表
        self.setup_colored_data_table(data_frame)
        
        # ★★★ 初始化時應用使用者偏好的字體大小 ★★★
        initial_font_size = self.load_font_preference()
        self.root.after(500, lambda: self.update_all_table_fonts(initial_font_size))

    def setup_integrated_overview(self, frame):
        """設置整合的統計概覽"""
        title_font = ("微軟正黑體", 16, "bold")
        label_font = ("微軟正黑體", 14)
        value_font = ("Consolas", 14, "bold")

        # 數據概覽區塊
        stats_info_frame = ttk.LabelFrame(frame, text="📊 數據概覽")
        stats_info_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.stats_labels = {}
        stats_grid = [
            ("總局數", "total_games", 0, 0), 
            ("大出現次數", "big_count", 1, 0), 
            ("小出現次數", "small_count", 2, 0),
            ("豹子次數", "leopard_count", 3, 0), 
            ("大出現率", "big_rate", 4, 0), 
            ("小出現率", "small_rate", 5, 0),
            ("豹子出現率", "leopard_rate", 6, 0),
            ("最長連大", "max_big_streak", 7, 0), 
            ("最長連小", "max_small_streak", 8, 0)
        ]
        
        for label_text, key, row, col in stats_grid:
            ttk.Label(stats_info_frame, text=f"{label_text}:", 
                     font=label_font).grid(row=row, column=0, padx=5, pady=2, sticky=tk.W)
            value_label = ttk.Label(stats_info_frame, text="--", font=value_font, 
                                   foreground="cyan")
            value_label.grid(row=row, column=1, padx=5, pady=2, sticky=tk.W)
            self.stats_labels[key] = value_label

        # Super AI 效能分析區塊
        prediction_frame = ttk.LabelFrame(frame, text="🚀 Super AI 智能分析")
        prediction_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.prediction_stats = {}
        prediction_items = [
            ("總預測次數", "total_predictions", 0), 
            ("預測命中次數", "hit_count", 1), 
            ("Super AI 命中率", "overall_hit_rate", 2),
            ("真龍識別次數", "dragon_count", 3),
            ("斬龍成功率", "dragon_kill_rate", 4)
        ]
        
        for label_text, key, row in prediction_items:
            ttk.Label(prediction_frame, text=f"{label_text}:", 
                     font=label_font).grid(row=row, column=0, padx=5, pady=3, sticky=tk.W)
            value_label = ttk.Label(prediction_frame, text="--", font=value_font, 
                                   foreground="lightgreen")
            value_label.grid(row=row, column=1, padx=5, pady=3, sticky=tk.W)
            self.prediction_stats[key] = value_label

    def setup_colored_data_table(self, frame):
        """設置賽博龐克風格詳細記錄表格"""
        title_label = ttk.Label(frame, text="📋 詳細記錄 (賽博龐克風格)", 
                            font=("Consolas", 16, "bold"))
        title_label.pack(pady=(0, 15))
        
        columns = ["局數", "總點數", "結果", "原判斷", "Super AI 預測", "命中狀態"]
        
        # ★★★ 使用賽博龐克風格表格 ★★★
        self.cyberpunk_table = CyberpunkStyleTable(frame, columns)
        
        # 保持兼容性
        self.stats_tree = self.cyberpunk_table.tree
        
        if hasattr(self, 'data') and self.data:
            self.update_cyberpunk_table()

    # ==========================================================================
    # Super AI 核心邏輯區塊 - 完美修復版
    # ==========================================================================

    def run_paste_and_analyze(self):
        """Super AI 一鍵分析主函數 - 完美修復版"""
        self.paste_analyze_button.config(state=tk.DISABLED)
        self.status_var.set("🐉 Super AI 智能龍系統分析中...")
        self.root.update_idletasks()

        try:
            # 貼上數據
            try:
                self.text_area.delete("1.0", tk.END)
                self.text_area.insert("1.0", self.root.clipboard_get())
            except tk.TclError:
                messagebox.showwarning("貼上失敗", "剪貼簿中沒有有效的文字內容。")
                return

            # 執行分析
            self.paste_and_process()
            if self.data:
                self.super_ai_predict_next_result()
                self.update_results()
                self.update_leopard_warning()
            
        except Exception as e:
            messagebox.showerror("分析錯誤", f"Super AI 分析過程中發生錯誤: {e}")
            
        finally:
            self.status_var.set("🚀 Super AI 智能龍系統分析完成，就緒")
            self.paste_analyze_button.config(state=tk.NORMAL)

    def paste_and_process(self):
        """處理貼上的數據 - 完美修復版"""
        try:
            text = self.text_area.get("1.0", tk.END).strip()
            pattern = re.compile(r"\[第(\d+)局\]第一骰:\s*(\d)點, 第二骰:\s*(\d)點, 第三骰:\s*(\d)點, 【(大|小|豹子)】")
            matches = pattern.findall(text)
            
            if not matches:
                messagebox.showwarning("格式錯誤", "未找到符合格式的骰子資料")
                return

            new_data = []
            for round_num, d1, d2, d3, declared_size in matches:
                total = sum(map(int, [d1, d2, d3]))
                sys_size = "豹子" if d1 == d2 == d3 else ("大" if total > 10 else "小")
                new_data.append([int(round_num), total, sys_size, declared_size, "", "無", "", datetime.now()])

            # 處理新數據
            existing_rounds = {r[0] for r in self.data}
            added_new = False
            
            for item in new_data:
                if item[0] not in existing_rounds:
                    if self.data and isinstance(self.data[0], tuple):
                        self.data = [list(r) for r in self.data]
                    self.data.append(item)
                    added_new = True

            if not added_new and all(len(r) > 6 and r[5] not in ["無", ""] for r in self.data):
                self.status_var.set("沒有新的局數，且所有數據都已分析。")
                return

            self.data.sort(key=lambda x: x[0])

            # ★★★ 關鍵修復：Super AI 回測分析使用修復後的龍判斷邏輯 ★★★
            for i in range(1, len(self.data)):
                if len(self.data[i]) < 8 or self.data[i][5] in ["無", ""]:
                    historical_results = [r[2] for r in self.data[:i] if r[2] in ['大', '小']]
                    actual_result = self.data[i][2]
                    
                    # 使用 Super AI 進行預測
                    if historical_results:
                        prediction, confidence, explanation = super_ai_prediction(historical_results)
                        self.data[i][5] = prediction
                        
                        # ★★★ 使用修復後的智能命中狀態判斷 ★★★
                        hit_status = self.determine_hit_status_with_emoji(
                            prediction=prediction,
                            actual_result=actual_result, 
                            historical_results=historical_results + [actual_result],
                            index=i
                        )
                        self.data[i][6] = hit_status
                        
                        # 記錄預測統計
                        if actual_result in ['大', '小']:
                            is_hit = prediction == actual_result
                            self.super_ai_predictions.append((prediction, actual_result, is_hit))

            self.label_info.config(text=f"🐉 Super AI 已分析 {len(self.data)} 局資料")
            self.update_statistics()
            self.status_var.set("🚀 Super AI 智能龍系統數據處理完成")
            
        except Exception as e:
            messagebox.showerror("處理錯誤", f"在處理數據時發生錯誤: {e}")

    def determine_hit_status_with_emoji(self, prediction, actual_result, historical_results, index=None):
        if actual_result == "豹子":
            return "🟣 豹子通殺"
        clean_history = [r for r in historical_results[:-1] if r in ["大", "小"]]
        if len(clean_history) < 2:
            return "✅ Super AI 命中" if prediction == actual_result else "❌ Super AI 未命中"
        last_type = clean_history[-1]
        previous_streak = 1
        for i in range(len(clean_history) - 2, -1, -1):
            if clean_history[i] == last_type:
                previous_streak += 1
            else:
                break
        dragon_continues = (actual_result == last_type)
        is_trying_to_follow_dragon = (prediction == last_type)
        is_hit = (prediction == actual_result)
        if previous_streak >= 4:
            if is_trying_to_follow_dragon:
                return "🐲 跟龍命中" if is_hit else "🔥 跟龍失誤"
            else:
                return "⚔️ 斬龍命中" if is_hit else "💥 斬龍失誤"
        elif previous_streak == 3:
            if is_trying_to_follow_dragon:
                return "🚀 助龍命中" if is_hit else "🔥 助龍失誤"
            else:
                return "🛡️ 阻龍命中" if is_hit else "💥 阻龍失誤"
        else:
            return "✅ Super AI 命中" if is_hit else "❌ Super AI 未命中"

    def analyze_dragon_status_correctly(self, clean_history, actual_result):
        """正確分析龍勢狀況"""
        
        if len(clean_history) == 0:
            return {
                "type": "insufficient_data",
                "previous_streak": 0,
                "previous_type": None,
                "dragon_continues": False
            }
        
        # 計算之前的連莊情況（不包括當前結果）
        last_type = clean_history[-1]
        previous_streak = 1
        
        for i in range(len(clean_history) - 2, -1, -1):
            if clean_history[i] == last_type:
                previous_streak += 1
            else:
                break
        
        # 判斷龍勢變化
        dragon_continues = (actual_result == last_type)
        current_streak = previous_streak + 1 if dragon_continues else 1
        
        # 確定龍的狀態
        if previous_streak >= 4:
            # 之前已經是真龍
            dragon_type = "true_dragon_continuation" if dragon_continues else "true_dragon_broken"
        elif previous_streak == 3:
            # 之前是準龍
            if dragon_continues:
                dragon_type = "pre_dragon_becomes_true"  # 準龍變真龍
            else:
                dragon_type = "pre_dragon_broken"        # 準龍被破
        elif previous_streak == 2:
            # 之前是雙連
            if dragon_continues:
                dragon_type = "double_becomes_pre"       # 雙連變準龍
            else:
                dragon_type = "double_broken"            # 雙連被破
        else:
            # 正常狀態
            dragon_type = "normal_pattern"
        
        return {
            "type": dragon_type,
            "previous_streak": previous_streak,
            "current_streak": current_streak,
            "previous_type": last_type,
            "dragon_continues": dragon_continues,
            "actual_result": actual_result
        }

    def classify_hit_status_with_emoji(self, prediction, actual_result, dragon_situation):
        """根據龍勢情況分類命中狀態 - 精美emoji版"""
        
        is_hit = (prediction == actual_result)
        dragon_type = dragon_situation["type"]
        dragon_continues = dragon_situation["dragon_continues"]
        previous_type = dragon_situation["previous_type"]
        
        # ★★★ 核心邏輯：基於龍勢變化的正確判斷（帶精美emoji） ★★★
        
        if dragon_type == "true_dragon_continuation":
            # 真龍延續情況
            if is_hit:
                if prediction == previous_type:
                    return "🐲 跟龍命中"      # 正確預測龍會繼續
                else:
                    return "✅ Super AI 命中"  # 一般命中（理論上不會到這裡）
            else:
                return "🔥 跟龍失誤"         # 以為龍會繼續，但沒命中
        
        elif dragon_type == "true_dragon_broken":
            # 真龍被打破情況  
            if is_hit:
                if prediction != previous_type:
                    return "⚔️ 斬龍命中"      # 正確預測龍會斷！
                else:
                    return "🔥 跟龍失誤"      # 想跟龍但龍斷了，算失誤
            else:
                if prediction == previous_type:
                    return "🔥 跟龍失誤"      # 跟龍失敗
                else:
                    return "💥 斬龍失誤"      # 想斬龍但預測錯了
        
        elif dragon_type == "pre_dragon_becomes_true":
            # 準龍變成真龍（3連變4連）
            if is_hit:
                if prediction == previous_type:
                    return "🚀 助龍命中"      # 成功助龍形成真龍
                else:
                    return "✅ Super AI 命中"  # 理論上不會到這裡
            else:
                return "🔥 助龍失誤"         # 想助龍但失敗
        
        elif dragon_type == "pre_dragon_broken":
            # 準龍被打破（3連被斷）
            if is_hit:
                if prediction != previous_type:
                    return "🛡️ 阻龍命中"     # 成功阻止龍勢形成
                else:
                    return "🔥 助龍失誤"     # 想助龍但被阻
            else:
                if prediction == previous_type:
                    return "🔥 助龍失誤"     # 助龍失敗
                else:
                    return "💥 阻龍失誤"     # 想阻龍但失敗
        
        elif dragon_type == "double_becomes_pre":
            # 雙連變準龍（2連變3連）
            if is_hit:
                return "🚀 助龍命中"         # 幫助龍勢發展
            else:
                return "🔥 助龍失誤"         # 想助龍但失敗
        
        elif dragon_type == "double_broken":
            # 雙連被打破
            if is_hit:
                return "🛡️ 阻龍命中"        # 成功阻止龍勢
            else:
                return "💥 阻龍失誤"        # 想阻龍但失敗
        
        else:
            # 正常模式
            if is_hit:
                return "✅ Super AI 命中"
            else:
                return "❌ Super AI 未命中"

    # 向後兼容方法
    def determine_hit_status_simple(self, actual_result, prediction):
        """簡化調用接口 - 向後兼容（保留emoji）"""
        
        # 獲取歷史結果（排除豹子，包括當前結果）
        all_results = []
        for row in self.data:
            if len(row) > 2 and row[2] in ["大", "小"]:
                all_results.append(row[2])
        
        # 加入當前結果
        all_results.append(actual_result)
        
        return self.determine_hit_status_with_emoji(
            prediction=prediction,
            actual_result=actual_result,
            historical_results=all_results
        )

    def super_ai_predict_next_result(self):
        """Super AI 預測下一局結果 - 保持原有功能"""
        if not self.data:
            self.predict_label.config(text="🐉 Super AI 預測： 待命中...")
            self.result_label.config(text="數據不足，無法進行預測")
            return
        
        try:
            # 獲取歷史結果進行預測
            historical_results = [row[2] for row in self.data if len(row) > 2 and row[2] != "豹子"]
            
            if len(historical_results) >= 2:
                prediction, confidence, description = super_ai_prediction(historical_results)
                
                self.predict_label.config(text=f"🐉 Super AI 預測下一局： {prediction}")
                self.result_label.config(text=f"🎯 信心度: {confidence}%\n\n📊 {description}")
            else:
                self.predict_label.config(text="🐉 Super AI 預測： 數據收集中...")
                self.result_label.config(text="需要更多歷史數據來進行精確預測")
                
        except Exception as e:
            print(f"預測錯誤: {e}")
            self.predict_label.config(text="🐉 Super AI 預測： 分析中...")
            self.result_label.config(text="正在分析市場趨勢...")
    
    # ==========================================================================
    # UI 更新和統計功能
    # ==========================================================================
    
    def update_statistics(self):
        """更新統計數據 - 完整修復版"""
        print("開始更新統計數據...")
        
        # 安全檢查：確保必要的UI元件存在
        if not hasattr(self, 'stats_labels'):
            print("警告：stats_labels 不存在，跳過統計標籤更新")
            return
        
        # 處理空數據情況
        if not self.data:
            print("數據為空，清空顯示")
            self.clear_statistics_display()
            return

        # 篩選數據
        try:
            filtered_data = self.filter_data_by_range(self.data)
            print(f"篩選後數據量: {len(filtered_data)}")
        except Exception as e:
            print(f"數據篩選失敗: {e}")
            filtered_data = self.data

        # 計算基本統計
        total_games = len(filtered_data)
        if total_games == 0:
            self.clear_statistics_display()
            return

        try:
            big_count = sum(1 for r in filtered_data if len(r) > 2 and r[2] == "大")
            small_count = sum(1 for r in filtered_data if len(r) > 2 and r[2] == "小")
            leopard_count = sum(1 for r in filtered_data if len(r) > 2 and r[2] == "豹子")
            
            print(f"統計結果 - 總局數: {total_games}, 大: {big_count}, 小: {small_count}, 豹子: {leopard_count}")
        except Exception as e:
            print(f"基本統計計算失敗: {e}")
            return

        # 更新基本統計標籤
        try:
            self.update_basic_stats_labels(total_games, big_count, small_count, leopard_count, filtered_data)
            print("基本統計標籤更新成功")
        except Exception as e:
            print(f"基本統計標籤更新失敗: {e}")

        # 更新Super AI預測統計
        try:
            self.update_prediction_stats_labels(filtered_data)
            print("預測統計標籤更新成功")
        except Exception as e:
            print(f"預測統計標籤更新失敗: {e}")

        # 更新表格顯示
        try:
            self.safe_update_table(filtered_data)
            print("表格更新成功")
            
        except Exception as e:
            print(f"表格更新失敗: {e}")

    # ─── 滑動評估監控 ───
    def sliding_monitor(self, k=10, threshold=0.85):
        # 計算最近 k 筆命中率
        preds = self.super_ai_predictions[-k:]
        if not preds: return
        hit_rate = sum(1 for _,_,hit in preds) / len(preds)
        if hit_rate < threshold:
            print(f"⚠️ 滑動評估低於 {threshold*100:.0f}% ({hit_rate*100:.1f}%)，觸發再訓練")
            # 呼叫增量再訓練
            self.incremental_retrain(self.data[-k:], window_size=100, batch_size=20)
        # 最後呼叫
        self.sliding_monitor()
        # ────────────────────────

    def clear_statistics_display(self):
        """清空統計顯示"""
        try:
            # 清空基本統計
            if hasattr(self, 'stats_labels'):
                for key, label in self.stats_labels.items():
                    label.config(text="--")
            
            # 清空預測統計
            if hasattr(self, 'prediction_stats'):
                for key, label in self.prediction_stats.items():
                    label.config(text="--")
            
            # 清空表格
            if hasattr(self, 'cyberpunk_table') and self.cyberpunk_table:
                self.cyberpunk_table.load_data([])
            elif hasattr(self, 'stats_tree') and self.stats_tree:
                for item in self.stats_tree.get_children():
                    self.stats_tree.delete(item)
                    
            print("統計顯示已清空")
        except Exception as e:
            print(f"清空統計顯示失敗: {e}")

    def update_basic_stats_labels(self, total_games, big_count, small_count, leopard_count, filtered_data):
        """更新基本統計標籤"""
        if not hasattr(self, 'stats_labels'):
            return
        
        # 基本數量統計
        self.stats_labels['total_games'].config(text=str(total_games))
        self.stats_labels['big_count'].config(text=str(big_count))
        self.stats_labels['small_count'].config(text=str(small_count))
        self.stats_labels['leopard_count'].config(text=str(leopard_count))
        
        # 比率統計
        if total_games > 0:
            self.stats_labels['big_rate'].config(text=f"{big_count/total_games*100:.1f}%")
            self.stats_labels['small_rate'].config(text=f"{small_count/total_games*100:.1f}%")
            self.stats_labels['leopard_rate'].config(text=f"{leopard_count/total_games*100:.2f}%")
            
            # 最長連莊統計
            try:
                results_only = [r[2] for r in filtered_data if len(r) > 2]
                max_big = self.calc_max_streak(results_only, "大")
                max_small = self.calc_max_streak(results_only, "小")
                
                self.stats_labels['max_big_streak'].config(text=str(max_big))
                self.stats_labels['max_small_streak'].config(text=str(max_small))
            except Exception as e:
                print(f"計算最長連莊失敗: {e}")
                if 'max_big_streak' in self.stats_labels:
                    self.stats_labels['max_big_streak'].config(text="--")
                if 'max_small_streak' in self.stats_labels:
                    self.stats_labels['max_small_streak'].config(text="--")

    def update_prediction_stats_labels(self, filtered_data):
        """更新Super AI預測統計標籤"""
        if not hasattr(self, 'prediction_stats'):
            return
        
        try:
            # 準備預測數據（排除豹子）
            prediction_data = []
            for r in filtered_data:
                if len(r) > 6 and r[5] not in ["無", "", None] and len(r) > 2 and r[2] != "豹子":
                    prediction_data.append((r[5], r[2], r[6]))
            
            total_predictions = len(prediction_data)
            hit_count = sum(1 for pred, actual, status in prediction_data if pred == actual)
            
            # 龍相關統計
            dragon_operations = [
                (pred, actual, status) for pred, actual, status in prediction_data 
                if any(keyword in str(status) for keyword in ["跟龍", "斬龍", "助龍", "阻龍"])
            ]
            
            dragon_successes = sum(1 for pred, actual, status in dragon_operations 
                                if any(keyword in str(status) for keyword in ["跟龍命中", "斬龍命中", "助龍命中", "阻龍命中"]))
            
            # 更新顯示
            self.prediction_stats['total_predictions'].config(text=str(total_predictions))
            self.prediction_stats['hit_count'].config(text=str(hit_count))
            
            hit_rate = f"{hit_count/total_predictions*100:.1f}%" if total_predictions > 0 else "0.0%"
            self.prediction_stats['overall_hit_rate'].config(text=hit_rate)
            
            self.prediction_stats['dragon_count'].config(text=str(len(dragon_operations)))
            
            dragon_success_rate = f"{dragon_successes/len(dragon_operations)*100:.1f}%" if len(dragon_operations) > 0 else "0.0%"
            self.prediction_stats['dragon_kill_rate'].config(text=dragon_success_rate)
            
        except Exception as e:
            print(f"更新預測統計失敗: {e}")
            # 設定默認值
            for key in self.prediction_stats:
                self.prediction_stats[key].config(text="--")

    def safe_update_table(self, filtered_data):
        """安全的表格更新"""
        print("開始安全表格更新...")
        
        # 方案1：優先使用賽博龐克表格
        if hasattr(self, 'cyberpunk_table') and self.cyberpunk_table:
            try:
                print("使用賽博龐克表格更新")
                self.update_cyberpunk_table_safe(filtered_data)
                return
            except Exception as e:
                print(f"賽博龐克表格更新失敗: {e}")
        
        # 方案2：使用統一更新方法
        try:
            print("使用統一更新方法")
            self.update_stats_table_with_colors(filtered_data)
            return
        except Exception as e:
            print(f"統一更新方法失敗: {e}")
        
        # 方案3：備用表格更新
        try:
            print("使用備用表格更新")
            self.fallback_table_update(filtered_data)
            return
        except Exception as e:
            print(f"備用表格更新失敗: {e}")
        
        # 方案4：基礎Treeview更新
        try:
            print("使用基礎Treeview更新")
            self.basic_treeview_update(filtered_data)
        except Exception as e:
            print(f"所有表格更新方案都失敗: {e}")

    def update_results(self):
        """更新結果顯示"""
        if not self.data:
            self.result_label.config(text="🚀 Super AI 智能龍系統待命中")
            return
        
        # 顯示最近幾局
        recent = self.data[-4:]
        recent_text = "\n".join([f"[第{r[0]}局] {r[1]}→{r[2]} {r[6] if len(r) > 6 else ''}" for r in recent])
        
        # Super AI 預測信息
        ai_feedback = f"🐉 Super AI 上次預測: {self.data[-1][5]}" if len(self.data) > 0 and len(self.data[-1]) > 5 else "🐉 Super AI 尚未預測"
        
        # 建議信息
        suggestion_text = f"\n\n{self.suggestion}" if hasattr(self, 'suggestion') and self.suggestion else ""
        
        self.result_label.config(text=recent_text + "\n\n" + ai_feedback + suggestion_text)

    # ==========================================================================
    # 輔助功能（保留原有功能）
    # ==========================================================================
    def update_cyberpunk_table(self):
        """更新賽博龐克風格表格數據 - 強化版"""
        if not hasattr(self, 'cyberpunk_table') or not self.cyberpunk_table:
            print("警告：賽博龐克表格未初始化")
            return
        
        try:
            # 準備表格數據
            table_data = []
            for row in self.data:
                row_data = [
                    row[0] if len(row) > 0 else "",  # 局數
                    row[1] if len(row) > 1 else "",  # 總點數
                    row[2] if len(row) > 2 else "",  # 結果
                    row[3] if len(row) > 3 else "",  # 原判斷
                    row[5] if len(row) > 5 else "",  # Super AI 預測
                    row[6] if len(row) > 6 else ""   # 命中狀態
                ]
                table_data.append(row_data)
            
            # 更新表格數據
            self.cyberpunk_table.data = table_data
            self.cyberpunk_table.load_data(table_data)
            self.cyberpunk_table.auto_adjust_columns()
            
            print(f"賽博龐克表格更新完成，共 {len(table_data)} 行數據")
            
        except Exception as e:
            print(f"更新賽博龐克表格時發生錯誤: {e}")
            raise

    def ensure_table_compatibility(self):
        """確保表格兼容性"""
        # 檢查賽博龐克表格
        if hasattr(self, 'cyberpunk_table'):
            print("✅ 賽博龐克詳細記錄表格已初始化")
        else:
            print("❌ 賽博龐克詳細記錄表格未初始化")
        
        if hasattr(self, 'cyberpunk_odds_table'):
            print("✅ 賽博龐克賠率計算器表格已初始化")
        else:
            print("❌ 賽博龐克賠率計算器表格未初始化")
        
        # 檢查統計樹
        if hasattr(self, 'stats_tree'):
            print("✅ 統計樹已初始化")
        else:
            print("❌ 統計樹未初始化")

    def filter_data_by_range(self, data):
        """篩選數據範圍"""
        if not hasattr(self, 'range_var'):
            return data
        
        try:
            range_value = self.range_var.get()
            
            if range_value == "全部":
                return data
            elif range_value == "最近10局":
                return data[-10:]
            elif range_value == "最近20局":
                return data[-20:]
            elif range_value == "最近50局":
                return data[-50:]
            elif range_value == "最近100局":
                return data[-100:]
            else:
                # 嘗試從字符串中提取數字
                import re
                match = re.search(r'(\d+)', range_value)
                if match:
                    num = int(match.group(1))
                    return data[-num:]
                return data
        except Exception as e:
            print(f"數據篩選錯誤: {e}")
            return data

    def calc_max_streak(self, results, target):
        """計算最長連續出現次數"""
        if not results:
            return 0
        
        max_streak = 0
        current_streak = 0
        
        for result in results:
            if result == target:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        
        return max_streak

    def update_cyberpunk_table_safe(self, filtered_data):
        """安全的賽博龐克表格更新"""
        if not hasattr(self, 'cyberpunk_table') or not self.cyberpunk_table:
            raise Exception("賽博龐克表格未初始化")
        
        # 準備表格數據
        table_data = []
        for row in filtered_data:
            try:
                row_data = [
                    row[0] if len(row) > 0 else "",  # 局數
                    row[1] if len(row) > 1 else "",  # 總點數
                    row[2] if len(row) > 2 else "",  # 結果
                    row[3] if len(row) > 3 else "",  # 原判斷
                    row[5] if len(row) > 5 else "",  # Super AI 預測
                    row[6] if len(row) > 6 else ""   # 命中狀態
                ]
                table_data.append(row_data)
            except Exception as e:
                print(f"處理行數據失敗: {e}")
                continue
        
        # 載入數據到賽博龐克表格
        self.cyberpunk_table.data = table_data
        self.cyberpunk_table.load_data(table_data)
        self.cyberpunk_table.auto_adjust_columns()
        
        print(f"賽博龐克表格更新完成，共 {len(table_data)} 行數據")

    def update_stats_table_with_colors(self, filtered_data):
        """統一的表格更新方法"""
        print("執行統一表格更新方法...")
        
        # 如果有賽博龐克表格，優先使用
        if hasattr(self, 'cyberpunk_table') and self.cyberpunk_table:
            self.update_cyberpunk_table_safe(filtered_data)
            return
        
        # 否則使用傳統方法
        if hasattr(self, 'stats_tree') and self.stats_tree:
            self.fallback_table_update(filtered_data)
        else:
            raise Exception("沒有可用的表格控件")

    def fallback_table_update(self, filtered_data):
        """備用表格更新方案"""
        if not hasattr(self, 'stats_tree') or not self.stats_tree:
            raise Exception("stats_tree 不存在")
        
        try:
            # 清空現有數據
            for item in self.stats_tree.get_children():
                self.stats_tree.delete(item)
            
            # 配置基本顏色標籤
            color_configs = {
                "big_result": ("#e74c3c", "bold"),
                "small_result": ("#3498db", "bold"),
                "leopard_result": ("#f39c12", "bold"),
                "ai_hit": ("#27ae60", "bold"),
                "ai_miss": ("#e67e22", "bold"),
                "dragon_hit": ("#00ff00", "bold"),
                "dragon_miss": ("#ff0000", "bold")
            }
            
            for tag, (color, weight) in color_configs.items():
                self.stats_tree.tag_configure(tag, 
                                            foreground=color, 
                                            font=("Consolas", 12, weight))
            
            # 插入數據
            for item in filtered_data:
                try:
                    values = list(item)
                    while len(values) < 7:
                        values.append("")
                    
                    # 決定標籤
                    result = str(values[2]) if len(values) > 2 else ""
                    hit_status = str(values[6]).strip() if len(values) > 6 else ""
                    
                    tags = self.determine_row_tags(result, hit_status)
                    
                    self.stats_tree.insert("", tk.END, 
                                        values=(values[0], values[1], values[2], 
                                                values[3], values[5], values[6]), 
                                        tags=tags)
                except Exception as e:
                    print(f"插入行數據失敗: {e}")
                    continue
            
            print("備用表格更新成功")
            
        except Exception as e:
            print(f"備用表格更新失敗: {e}")
            raise

    def basic_treeview_update(self, filtered_data):
        """基礎Treeview更新（最後的備用方案）"""
        if not hasattr(self, 'stats_tree') or not self.stats_tree:
            raise Exception("stats_tree 不存在")
        
        # 清空並插入基本數據
        for item in self.stats_tree.get_children():
            self.stats_tree.delete(item)
        
        for item in filtered_data:
            try:
                values = list(item)
                while len(values) < 7:
                    values.append("")
                
                self.stats_tree.insert("", tk.END, 
                                    values=(values[0], values[1], values[2], 
                                            values[3], values[5], values[6]))
            except:
                continue
        
        print("基礎Treeview更新完成")

    def determine_row_tags(self, result, hit_status):
        """決定行標籤"""
        tags = []
        
        # 優先檢查命中狀態
        if "跟龍命中" in hit_status or "斬龍命中" in hit_status or "助龍命中" in hit_status or "阻龍命中" in hit_status:
            tags = ["dragon_hit"]
        elif "跟龍失誤" in hit_status or "斬龍失誤" in hit_status:
            tags = ["dragon_miss"]
        elif "命中" in hit_status:
            tags = ["ai_hit"]
        elif "未命中" in hit_status or "失誤" in hit_status:
            tags = ["ai_miss"]
        elif result == "大":
            tags = ["big_result"]
        elif result == "小":
            tags = ["small_result"]
        elif result == "豹子":
            tags = ["leopard_result"]
        
        return tags

    def filter_stats_data(self):
        """篩選統計數據"""
        if not self.data or not hasattr(self, 'filter_var'):
            return
        
        try:
            filtered_data = self.filter_data_by_range(self.data)
            filter_value = self.filter_var.get()
            
            if filter_value == "只顯示大":
                filtered_data = [item for item in filtered_data if len(item) > 2 and item[2] == "大"]
            elif filter_value == "只顯示小":
                filtered_data = [item for item in filtered_data if len(item) > 2 and item[2] == "小"]
            elif filter_value == "只顯示豹子":
                filtered_data = [item for item in filtered_data if len(item) > 2 and item[2] == "豹子"]
            
            self.safe_update_table(filtered_data)
            
        except Exception as e:
            print(f"篩選更新失敗: {e}")

    def filter_stats_data(self):
        """篩選統計數據"""
        if not self.data:
            return
        
        filtered_data = self.filter_data_by_range(self.data)
        filter_value = self.filter_var.get()
        
        if filter_value == "只顯示大":
            filtered_data = [item for item in filtered_data if item[2] == "大"]
        elif filter_value == "只顯示小":
            filtered_data = [item for item in filtered_data if item[2] == "小"]
        elif filter_value == "只顯示豹子":
            filtered_data = [item for item in filtered_data if item[2] == "豹子"]
        
        self.update_stats_table_with_colors(filtered_data)

    def update_leopard_warning(self):
        """更新豹子警告"""
        if not self.data:
            self.leopard_warning_label.config(text="尚無豹子出現", fg="black")
            return
            
        recent_records = self.data[-20:]
        occurrences = [f"第{r[0]}局" for r in recent_records if len(r) > 2 and r[2] == "豹子"]
        count = len(occurrences)
        
        if count == 0:
            self.leopard_warning_label.config(text="最近20局無豹子出現", fg="black")
        elif count == 1:
            self.leopard_warning_label.config(text=f"⚠️ 注意: 最近20局中在{occurrences[0]}出現1次豹子", fg="#FF9900")
        elif count == 2:
            self.leopard_warning_label.config(text=f"⚠️ 警告: 最近20局中出現2次豹子！在{', '.join(occurrences)}", fg="#FF6600")
        else:
            self.leopard_warning_label.config(text=f"🔥 危險: 最近20局中出現{count}次豹子！", fg="red")

    def copy_prediction(self):
        """複製預測結果"""
        if not self.data or len(self.data) == 0:
            messagebox.showinfo("無數據", "沒有預測結果可複製")
            return
            
        try:
            last_record = self.data[-1]
            if len(last_record) <= 5 or not last_record[5]:
                messagebox.showinfo("無預測", "最後一局沒有預測結果")
                return
                
            copy_text = f"🐉 Super AI 智能系統 第{last_record[0]}局預測結果：{last_record[5]}\n分析時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
            if hasattr(self, 'suggestion') and self.suggestion:
                copy_text += f"\n\n{self.suggestion}"
                
            self.root.clipboard_clear()
            self.root.clipboard_append(copy_text)
            messagebox.showinfo("已複製", "Super AI 智能系統預測結果已複製到剪貼簿")
            
        except Exception as e:
            messagebox.showerror("複製失敗", f"複製預測結果時發生錯誤：{e}")

    def clear_all_data(self):
        """清除所有數據"""
        if not self.data:
            messagebox.showinfo("無數據", "目前沒有資料需要清除")
            return
            
        if not messagebox.askyesno("確認清除", "確定要清除所有資料？此操作無法復原。"):
            return
            
        try:
            # 清除所有數據
            self.data = []
            self.statistics = {}
            self.super_ai_predictions = []
            self.dragon_statistics = {}
            if hasattr(self, 'suggestion'):
                self.suggestion = ""
            
            # 更新UI
            if hasattr(self, 'label_info'):
                self.label_info.config(text="🐉 Super AI 智能系統資料已清除")
            if hasattr(self, 'predict_label'):
                self.predict_label.config(text="🐉 Super AI 預測： 待命中")
            if hasattr(self, 'result_label'):
                self.result_label.config(text="")
            if hasattr(self, 'leopard_warning_label'):
                self.leopard_warning_label.config(text="尚無豹子出現", fg="black")
            if hasattr(self, 'text_area'):
                self.text_area.delete("1.0", tk.END)
            if hasattr(self, 'update_statistics'):
                self.update_statistics()
                
            messagebox.showinfo("清除完成", "所有資料已成功清除")
            
        except Exception as e:
            messagebox.showerror("清除失敗", f"清除資料時發生錯誤：{e}")

    def on_tab_changed(self, event):
        """分頁切換事件"""
        try:
            current_tab = self.notebook.select()
            tab_name = self.notebook.tab(current_tab, "text")
            
            if "歷史數據分析" in tab_name and self.data:
                # 延遲更新統計，避免競爭條件
                self.root.after(100, self.update_statistics)
                
        except tk.TclError:
            pass
        except Exception as e:
            print(f"分頁切換錯誤: {e}")

    # ==========================================================================
    # 保留的原有功能（檔案系統、計算器等）
    # ==========================================================================

    def ensure_directories(self):
        """確保目錄存在"""
        if not os.path.exists(self.app_dir):
            os.makedirs(self.app_dir)
        if not os.path.exists(os.path.join(self.app_dir, "data")):
            os.makedirs(os.path.join(self.app_dir, "data"))
        if not os.path.exists(os.path.join(self.app_dir, "logs")):
            os.makedirs(os.path.join(self.app_dir, "logs"))
        if not os.path.exists(os.path.join(self.app_dir, "charts")):
            os.makedirs(os.path.join(self.app_dir, "charts"))

    def get_record_date(self):
        """獲取記錄日期"""
        return datetime.now().strftime("%Y-%m-%d")

    def change_theme(self):
        """更改主題"""
        style = tb.Style()
        style.theme_use(self.theme_var.get())

    def save_data(self):
        """儲存數據 - 確認正確版"""
        if not self.data:
            messagebox.showwarning("無資料", "沒有資料可以儲存")
            return
            
        # ★★★ 正確使用：asksaveasfilename ★★★
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            initialdir=getattr(self, 'data_dir', os.path.expanduser("~")),
            filetypes=[("JSON檔案", "*.json"), ("所有檔案", "*.*")],
            title="儲存 Super AI 數據"
        )
        
        if not filename:
            return
            
        try:
            # 處理數據格式
            data_to_save_processed = []
            for item in self.data:
                processed_item = list(item) if isinstance(item, (list, tuple)) else [item]
                
                # 確保所有項目都有8個元素
                while len(processed_item) < 8:
                    processed_item.append("")
                
                # 處理日期時間格式
                if len(processed_item) > 7 and isinstance(processed_item[7], datetime):
                    processed_item[7] = processed_item[7].strftime(self.time_format)
                elif len(processed_item) > 7 and processed_item[7] is None:
                    processed_item[7] = ""
                    
                data_to_save_processed.append(processed_item)

            # 準備完整的儲存數據
            save_data = {
                "data": data_to_save_processed,
                "statistics": getattr(self, 'statistics', {}),
                "session_date": getattr(self, 'record_date', datetime.now().strftime("%Y-%m-%d")),
                "super_ai_predictions": getattr(self, 'super_ai_predictions', []),
                "dragon_statistics": getattr(self, 'dragon_statistics', {}),
                "custom_image_path": getattr(self, 'custom_image_path', None),
                "app_version": "Super AI v14.0.0",
                "saved_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            with open(filename, "w", encoding="utf-8") as json_file:
                json.dump(save_data, json_file, ensure_ascii=False, indent=2)
                
            messagebox.showinfo("儲存成功", f"Super AI 智能系統資料已成功儲存到：\n{filename}")
            
        except Exception as e:
            messagebox.showerror("儲存失敗", f"儲存資料時發生錯誤：\n{str(e)}")

    def load_data(self):
        """載入數據 - 完美修復版"""
        # ★★★ 關鍵修復：使用正確的方法名 askopenfilename ★★★
        filename = filedialog.askopenfilename(
            initialdir=getattr(self, 'data_dir', os.path.expanduser("~")),
            filetypes=[("JSON檔案", "*.json"), ("所有檔案", "*.*")],
            title="載入 Super AI 數據"
        )
        
        if not filename:
            return
            
        try:
            # 重置數據
            self.data = []
            self.super_ai_predictions = []
            self.dragon_statistics = {}
            
            with open(filename, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                loaded_data = loaded.get("data", [])
                
                # 處理載入的數據
                processed_data = []
                for item in loaded_data:
                    if not item:  # 跳過空項目
                        continue
                        
                    processed_item = list(item) if isinstance(item, (list, tuple)) else [item]
                    
                    # 確保所有項目都有8個元素
                    while len(processed_item) < 8:
                        processed_item.append("")
                    
                    # 處理日期時間格式
                    if len(processed_item) > 7:
                        if processed_item[7] and isinstance(processed_item[7], str):
                            try:
                                processed_item[7] = datetime.strptime(processed_item[7], self.time_format)
                            except (ValueError, TypeError):
                                try:
                                    # 嘗試其他可能的時間格式
                                    processed_item[7] = datetime.strptime(processed_item[7], "%Y-%m-%d %H:%M:%S")
                                except (ValueError, TypeError):
                                    processed_item[7] = datetime.now()
                        else:
                            processed_item[7] = datetime.now()
                            
                    processed_data.append(processed_item)
                            
                self.data = processed_data
                
                # 載入其他數據
                self.statistics = loaded.get("statistics", {})
                self.super_ai_predictions = loaded.get("super_ai_predictions", [])
                self.dragon_statistics = loaded.get("dragon_statistics", {})
                self.custom_image_path = loaded.get("custom_image_path")
                
                if hasattr(self, 'record_date'):
                    self.record_date = loaded.get("session_date", datetime.now().strftime("%Y-%m-%d"))
            
            # 載入自訂圖片
            if hasattr(self, 'load_custom_image'):
                self.load_custom_image()
            
            # 更新UI
            if hasattr(self, 'label_info'):
                self.label_info.config(text=f"🐉 Super AI 已載入 {len(self.data)} 局資料")
            
            # 更新各種顯示
            if hasattr(self, 'update_results'):
                self.update_results()
            if hasattr(self, 'update_leopard_warning'):
                self.update_leopard_warning()
            if hasattr(self, 'update_statistics'):
                self.update_statistics()
            
            messagebox.showinfo("載入成功", f"Super AI 智能系統已成功載入 {len(self.data)} 局資料")
            
        except json.JSONDecodeError as e:
            messagebox.showerror("載入失敗", f"JSON檔案格式錯誤：\n{str(e)}")
        except FileNotFoundError:
            messagebox.showerror("載入失敗", "找不到指定的檔案")
        except Exception as e:
            messagebox.showerror("載入失敗", f"載入資料時發生錯誤：\n{str(e)}")

    def clear_all_data(self):
        """清除所有數據"""
        if not self.data or not messagebox.askyesno("確認清除", "確定要清除所有資料？此操作無法復原。"):
            return
            
        self.data = []
        self.statistics = {}
        self.super_ai_predictions = []
        self.dragon_statistics = {}
        self.suggestion = ""
        
        # 更新UI
        self.label_info.config(text="🐉 Super AI 智能龍系統資料已清除")
        self.predict_label.config(text="🐉 Super AI 預測： 待命中")
        self.result_label.config(text="")
        self.leopard_warning_label.config(text="尚無豹子出現", fg="black")
        self.text_area.delete("1.0", tk.END)
        self.update_statistics()

    def new_session(self):
        """開始新會話"""
        if self.data:
            result = messagebox.askyesno("確認新會話", "開始新會話將清除目前的資料。是否要先儲存當前資料？")
            if result:
                self.save_data()

        # 清除資料
        self.data = []
        self.statistics = {}
        self.super_ai_predictions = []
        self.dragon_statistics = {}
        self.suggestion = ""
        self.session_start_time = datetime.now()
        self.record_date = self.get_record_date()

        # 更新顯示
        self.label_info.config(text="🐉 Super AI 智能龍系統已開始新會話")
        self.predict_label.config(text="🐉 Super AI 預測： 待命中")
        self.result_label.config(text="")
        self.leopard_warning_label.config(text="尚無豹子出現", fg="black")
        self.text_area.delete("1.0", tk.END)
        self.update_statistics()
        
        messagebox.showinfo("新會話", "Super AI 智能龍系統已成功開始新會話")

    # ==========================================================================
    # 賠率計算器功能（完全保留）
    # ==========================================================================
    
    def calculate_odds(self):
        """計算賠率"""
        try:
            base_amount = Decimal(self.calc_entry.get())
            rate = Decimal(self.rate_var.get())
            multiplier = Decimal(self.multiplier_entry.get())
            levels = int(self.levels_var.get())
            
            if base_amount <= 0 or levels <= 0:
                raise ValueError("初始本金和關數必須大於0")
            if rate <= 0:
                raise ValueError("賠率必須是正數")
            if multiplier <= 1:
                raise ValueError("投注倍率必須大於1才能覆蓋虧損")

            self.tree.delete(*self.tree.get_children())
            total_investment_so_far = Decimal("0")
            current_bet = base_amount
            results_data = []

            for i in range(1, levels + 1):
                if i > 1:
                    current_bet *= multiplier

                payout_if_win = current_bet * (Decimal("1") + rate)
                cumulative_investment = total_investment_so_far + current_bet
                net_profit_if_win = payout_if_win - cumulative_investment
                
                results_data.append((
                    f"第{i}關",
                    f"{current_bet:,.0f}",
                    f"{payout_if_win:,.0f}",
                    f"{net_profit_if_win:,.0f}",
                    f"{cumulative_investment:,.0f}",
                ))
                
                total_investment_so_far = cumulative_investment

            for row_data in results_data:
                self.tree.insert("", "end", values=row_data)

            final_total_investment = total_investment_so_far
            self.calc_result_label.config(
                text=f"若連續輸{levels-1}關，到第{levels}關總投入需: {final_total_investment:,.0f}"
            )
            self.profit_var.set(f"固定倍率: x{multiplier}")
            self.status_var.set("賠率計算完成")

        except (ValueError, TypeError) as e:
            messagebox.showerror("輸入錯誤", f"請檢查輸入值是否正確: {e}")
        except Exception as e:
            messagebox.showerror("計算錯誤", f"計算時發生意外錯誤: {str(e)}")

    def clear_table(self):
        """清除計算表格"""
        self.tree.delete(*self.tree.get_children())
        self.calc_entry.delete(0, tk.END)
        self.calc_entry.insert(0, "10000")
        self.rate_var.set("0.96")
        self.levels_var.set("10")
        self.calc_result_label.config(text="計算結果將顯示在這裡")
        self.profit_var.set("淨利潤: --")
        self.status_var.set("賠率計算器已重設")

    def save_calculation(self):
        """儲存計算結果"""
        if not self.tree.get_children():
            messagebox.showwarning("沒有數據", "賠率計算器中沒有可供儲存的結果。")
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV 檔案", "*.csv"), ("所有檔案", "*.*")],
            title="儲存賠率計算結果",
            initialfile=f"odds_calculation_{datetime.now().strftime('%Y%m%d')}.csv"
        )

        if not filepath:
            return

        try:
            columns = self.tree["columns"]
            with open(filepath, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow(columns)
                for row_id in self.tree.get_children():
                    row_values = self.tree.item(row_id, "values")
                    writer.writerow(row_values)
            
            messagebox.showinfo("儲存成功", f"計算結果已成功儲存至：\n{filepath}")
        except Exception as e:
            messagebox.showerror("儲存失敗", f"儲存檔案時發生錯誤：\n{e}")

    # ==========================================================================
    # 圖片和其他UI功能
    # ==========================================================================
    
    def select_custom_image(self):
        """選擇自訂圖片"""
        filetypes = [
            ("圖片文件", "*.png *.jpg *.jpeg *.gif *.bmp"),
            ("所有文件", "*.*")
        ]
        file_path = filedialog.askopenfilename(title="選擇自訂圖片", filetypes=filetypes)
        if file_path:
            self.custom_image_path = file_path
            self.load_custom_image()
            self.status_var.set(f"已選擇圖片: {os.path.basename(file_path)}")

    def load_custom_image(self):
        """載入自訂圖片"""
        if hasattr(self, 'custom_image_path') and self.custom_image_path and os.path.exists(self.custom_image_path):
            try:
                img_original = Image.open(self.custom_image_path)
                max_size = (250, 200)
                img_original.thumbnail(max_size, Image.Resampling.LANCZOS)
                
                self.custom_image_tk = ImageTk.PhotoImage(img_original)
                self.custom_image_label.config(image=self.custom_image_tk, text="")
            except Exception as e:
                messagebox.showerror("圖片載入錯誤", f"無法載入圖片: {self.custom_image_path}\n錯誤: {e}")
                self.clear_custom_image()
        else:
            self.clear_custom_image()

    def clear_custom_image(self):
        """清除自訂圖片"""
        self.custom_image_path = None
        self.custom_image_label.config(image='', text="無自訂圖片")
        if hasattr(self, 'custom_image_tk'):
            self.custom_image_tk = None
        self.status_var.set("自訂圖片已清除")

    # ==========================================================================
    # 自動儲存和報表功能
    # ==========================================================================
    
    def auto_save(self):
        """自動儲存"""
        if self.data:
            self.auto_save_session()
        self.auto_save_timer = self.root.after(120000, self.auto_save)  # 每2分鐘

    def auto_save_session(self):
        """自動儲存會話"""
        if not self.data:
            return
            
        try:
            autosave_dir = os.path.join(self.app_dir, "autosave")
            if not os.path.exists(autosave_dir):
                os.makedirs(autosave_dir)
                
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(autosave_dir, f"super_ai_dragon_{timestamp}.json")

            data_to_save_processed = []
            for item in self.data:
                processed_item = list(item)
                if len(processed_item) > 7 and isinstance(processed_item[7], datetime):
                    processed_item[7] = processed_item[7].strftime(self.time_format)
                data_to_save_processed.append(processed_item)

            with open(filename, "w", encoding="utf-8") as json_file:
                data_to_save = {
                    "data": data_to_save_processed,
                    "statistics": self.statistics,
                    "session_date": self.record_date,
                    "saved_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "auto_save": True,
                    "super_ai_predictions": self.super_ai_predictions,
                    "dragon_statistics": getattr(self, 'dragon_statistics', {}),
                    "custom_image_path": self.custom_image_path
                }
                json.dump(data_to_save, json_file, ensure_ascii=False, indent=2)
                
            # 清理舊檔案
            autosave_files = sorted([
                os.path.join(autosave_dir, f) for f in os.listdir(autosave_dir)
                if f.startswith("super_ai_dragon_") and f.endswith(".json")
            ], key=os.path.getmtime)

            if len(autosave_files) > 10:
                for old_file in autosave_files[:-10]:
                    try:
                        os.remove(old_file)
                    except:
                        pass

        except Exception as e:
            print(f"Super AI 智能龍系統自動儲存失敗: {e}")

    def auto_load_last_session(self):
        """自動載入上次會話"""
        try:
            autosave_dir = os.path.join(self.app_dir, "autosave")
            if not os.path.exists(autosave_dir):
                return

            autosave_files = sorted([
                os.path.join(autosave_dir, f) for f in os.listdir(autosave_dir)
                if (f.startswith("super_ai_dragon_") or f.startswith("super_ai_autosave_")) and f.endswith(".json")
            ], key=os.path.getmtime, reverse=True)

            if not autosave_files:
                return

            latest_file = autosave_files[0]
            last_modified = datetime.fromtimestamp(os.path.getmtime(latest_file))
            time_diff = datetime.now() - last_modified

            if time_diff <= timedelta(hours=24):
                result = messagebox.askyesno(
                    "載入上次 Super AI 智能龍系統會話",
                    f"發現 {last_modified.strftime('%Y-%m-%d %H:%M:%S')} 的 Super AI 自動儲存資料。\n是否要載入？"
                )

                if result:
                    with open(latest_file, "r", encoding="utf-8") as json_file:
                        data = json.load(json_file)

                        if "data" in data:
                            loaded_data = data["data"]
                            for item in loaded_data:
                                while len(item) < 8:
                                    item.append(None)
                                
                                if item[7] is not None and isinstance(item[7], str):
                                    try:
                                        item[7] = datetime.strptime(item[7], self.time_format)
                                    except (ValueError, TypeError):
                                        item[7] = None

                            self.data = [list(item) for item in loaded_data]
                            self.statistics = data.get("statistics", {})
                            self.record_date = data.get("session_date", self.get_record_date())
                            self.super_ai_predictions = data.get("super_ai_predictions", [])
                            self.dragon_statistics = data.get("dragon_statistics", {})
                            self.custom_image_path = data.get("custom_image_path")
                            
                            self.load_custom_image()
                            self.label_info.config(text=f"🐉 Super AI 已載入 {len(self.data)} 局資料")
                            self.update_results()
                            self.update_leopard_warning()
                            self.update_statistics()
                            
                            messagebox.showinfo("載入成功", f"Super AI 智能龍系統已成功載入 {len(self.data)} 局資料")

        except Exception as e:
            print(f"Super AI 智能龍系統自動載入失敗: {e}")

    def export_report(self):
        """匯出分析報表 - 確認正確版"""
        if not self.data:
            messagebox.showwarning("無資料", "沒有資料可匯出報表")
            return

        # ★★★ 正確使用：asksaveasfilename（匯出用） ★★★
        filename = filedialog.asksaveasfilename(
            defaultextension=".html",
            initialdir=getattr(self, 'data_dir', os.path.expanduser("~")),
            filetypes=[("HTML檔案", "*.html"), ("所有檔案", "*.*")],
            title="匯出 Super AI 智能系統報表"
        )

        if not filename:
            return

        try:
            self.export_html_report(filename)
            if messagebox.askyesno("匯出成功", f"Super AI 智能系統報表已匯出至：\n{filename}\n\n是否要立即開啟？"):
                webbrowser.open(filename)
        except Exception as e:
            messagebox.showerror("匯出失敗", f"匯出報表時發生錯誤：\n{str(e)}")

    def export_html_report(self, filename):
        """匯出HTML格式報表 - 完美修復版"""
        try:
            # 基本統計
            total_games = len(self.data)
            if total_games == 0:
                raise ValueError("沒有資料可以匯出")
                
            big_count = sum(1 for r in self.data if len(r) > 2 and r[2] == "大")
            small_count = sum(1 for r in self.data if len(r) > 2 and r[2] == "小")
            leopard_count = sum(1 for r in self.data if len(r) > 2 and r[2] == "豹子")

            # Super AI 統計
            ai_predictions = [(r[5], r[2]) for r in self.data if len(r) > 6 and r[5] not in ["無", ""] and len(r) > 2 and r[2] != "豹子"]
            ai_total = len(ai_predictions)
            ai_hits = sum(1 for pred, actual in ai_predictions if pred == actual)
            ai_rate = f"{ai_hits / ai_total * 100:.2f}%" if ai_total > 0 else "0.00%"

            # 生成HTML內容
            html_content = f"""<!DOCTYPE html>
    <html lang="zh-TW">
    <head>
        <meta charset="UTF-8">
        <title>🐉 Super AI 智能系統分析報表</title>
        <style>
            body {{ font-family: 'Microsoft JhengHei', Arial, sans-serif; }}
            .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
            h1 {{ text-align: center; color: #2c3e50; }}
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: center; }}
            th {{ background: #3498db; color: white; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🐉 Super AI 智能系統分析報表</h1>
            <p>總局數: {total_games} | Super AI 命中率: {ai_rate}</p>
            <table>
                <thead>
                    <tr><th>局數</th><th>總點數</th><th>結果</th><th>Super AI 預測</th><th>命中狀態</th></tr>
                </thead>
                <tbody>"""

            # 添加數據行
            for item in self.data:
                if len(item) >= 6:
                    html_content += f"""
                    <tr>
                        <td>{item[0]}</td><td>{item[1]}</td><td>{item[2]}</td>
                        <td>{item[5]}</td><td>{item[6] if len(item) > 6 else ''}</td>
                    </tr>"""

            html_content += """
                </tbody>
            </table>
            <p style="text-align: center; color: #7f8c8d;">
                報表生成時間: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """
            </p>
        </div>
    </body>
    </html>"""

            with open(filename, "w", encoding="utf-8") as file:
                file.write(html_content)
                
        except Exception as e:
            raise Exception(f"生成HTML報表時發生錯誤: {str(e)}")
        
    def ensure_directories(self):
        """確保目錄存在 - 完整版"""
        try:
            if not hasattr(self, 'app_dir') or not self.app_dir:
                self.app_dir = os.path.join(os.path.expanduser("~"), "GamblingTool")
            
            # 創建主目錄
            if not os.path.exists(self.app_dir):
                os.makedirs(self.app_dir)
            
            # 創建子目錄
            subdirs = ["data", "logs", "charts", "autosave", "exports"]
            for subdir in subdirs:
                subdir_path = os.path.join(self.app_dir, subdir)
                if not os.path.exists(subdir_path):
                    os.makedirs(subdir_path)
            
            # 設置數據目錄
            self.data_dir = os.path.join(self.app_dir, "data")
            
        except Exception as e:
            print(f"創建目錄時發生錯誤: {e}")
            # 使用用戶家目錄作為備用
            self.app_dir = os.path.expanduser("~")
            self.data_dir = self.app_dir

    def get_record_date(self):
        """獲取記錄日期"""
        return datetime.now().strftime("%Y-%m-%d")

    def show_help(self):
        """顯示幫助信息"""
        help_win = tk.Toplevel(self.root)
        help_win.title("Super AI 智能龍系統使用說明")
        help_win.geometry("700x1100")
        help_win.transient(self.root)
        help_win.grab_set()

        text_widget = tk.Text(help_win, wrap=tk.WORD, font=("微軟正黑體", 14), 
                             relief="flat", padx=15, pady=15)
        text_widget.pack(fill=tk.BOTH, expand=True)

        help_content = """🐉 Super AI 智能龍系統使用說明

【核心特色】
Super AI 智能版採用革命性的智能龍判斷系統，能夠：
• 智能識別真龍（4連以上）與準龍（3連）
• 多維度分析龍勢強度、市場趨勢、玩家心理
• 動態決策跟龍或斬龍時機
• 實戰驗證3000次以上

【龍的定義】
• 真龍：4連以上（包含第4次）
• 準龍：3連狀態（關鍵判斷期）
• 雙連：2連狀態（龍萌芽期）

【基本操作】
1. 數據輸入：
   - 在左側文字區貼上遊戲記錄
   - 格式：[第1局]第一骰: 4點, 第二骰: 3點, 第三骰: 3點, 【大】

2. 智能分析：
   - 點擊「🚀 Super AI 智能分析」
   - 系統自動進行龍勢分析並給出預測

3. 查看結果：
   - 右側顯示 Super AI 預測和信心度
   - 包含詳細的龍戰策略和資金建議

【智能龍戰策略】
• 超級長龍（8連+）：謹慎跟龍或智能斬龍
• 長龍（6-7連）：穩健跟龍或斬龍時機分析
• 標準龍（4-5連）：黃金跟龍期或智能斬龍
• 準龍（3連）：關鍵判斷期，助龍或阻龍
• 雙連（2連）：龍萌芽期，保守跟進

【彩色命中狀態】
🐲 跟龍命中 | ⚔️ 斬龍命中 | 🚀 助龍命中 | 🛡️ 阻龍命中
🔥 跟龍失誤 | ❌ 斬龍失誤 | ✅ 一般命中 | 🟠 豹子通殺

【其他功能】
• S級AI狀態重新訓練按鈕 - 一鍵重訓，命中率即時升級
• 字體大小微調功能
• 整合式統計介面（左側概覽，右側彩色數據表）
• 自動儲存 / 載入
• HTML 智能龍戰報表匯出
• 賠率計算器
• 自訂圖片顯示

【注意事項】
Super AI 智能龍系統雖具備高命中率與動態風控，投資仍有風險。
請依系統提示之龍戰策略與資金比例嚴格執行，並適時使用 S級AI 重新訓練 功能，確保模型保持最佳效能。"""

        text_widget.insert(tk.END, help_content)
        text_widget.config(state=tk.DISABLED)

        ttk.Button(help_win, text="關閉", command=help_win.destroy).pack(pady=10)

    def show_about(self):
        """顯示關於信息"""
        about_win = tk.Toplevel(self.root)
        about_win.title("關於 Super AI 智能系統")
        about_win.geometry("1600x600")
        about_win.transient(self.root)
        about_win.grab_set()

        ttk.Label(about_win, text="🐉 Game of Dice Super AI", 
                 font=("Arial", 24, "bold")).pack(pady=15)
        
        ttk.Label(about_win, text="Super AI 智能版", 
                 font=("Arial", 16)).pack()
        
        ttk.Label(about_win, text="版本: 18.0.0", 
                 font=("Arial", 12)).pack(pady=5)

        separator = ttk.Separator(about_win, orient=tk.HORIZONTAL)
        separator.pack(fill=tk.X, padx=20, pady=15)

        desc_text = """著作權聲明

                    一個用於分析骰子遊戲結果的預測工具

            提供骰子預測、賠率計算、統計分析、策略選擇等功能

            程式內容僅供學術研究或娛樂用途，禁止未經授權之散布、轉載、修改、重製或商業使用。如有違反，將視同侵害著作權，依法追究相關責任。

            本程式由【阿昕】原創開發，著作權所有，保留一切權利 !

            程式名稱：Game of Dice Super AI 智能版

            本程式已於【2025年7月22日】創作並持續更新完成，原始碼與開發紀錄可作為著作權證據。

            作者筆名：阿昕

            著作權所有，未經授權禁止轉載、散布、修改、商業使用 !

            違者將依中華民國著作權法追究法律責任 !"""

        ttk.Label(about_win, text=desc_text, justify=tk.CENTER, 
                 font=("微軟正黑體", 12)).pack(pady=15)

        ttk.Button(about_win, text="關閉", command=about_win.destroy).pack(pady=15)


# 主程式入口
if __name__ == "__main__":
    root = tb.Window(themename="cyborg")
    app = ModernGamblingTool(root)
    root.mainloop()