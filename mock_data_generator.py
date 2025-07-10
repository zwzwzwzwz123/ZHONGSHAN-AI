#!/usr/bin/env python3
"""
模拟数据生成器
用于在无法连接数据库时生成测试数据
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import yaml

class MockDataGenerator:
    """模拟数据生成器"""
    
    def __init__(self, config_path="./config/configs.yaml"):
        """初始化生成器"""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                self.configs = yaml.load(f, Loader=yaml.FullLoader)
        except:
            # 如果配置文件读取失败，使用默认配置
            self.configs = self._get_default_config()
        
        # 基础参数
        self.room_temp_range = (18, 26)  # 机房温度范围
        self.ac_temp_range = (17, 30)   # 空调温度范围
        self.ac_onoff_values = [0, 1]   # 开关状态
        
        # 时间模拟
        self.current_time = datetime.now()
        self.data_change_probability = 0.1  # 数据变化概率
        
        # 存储上一次生成的数据，模拟数据连续性
        self.last_data = {}
        
    def _get_default_config(self):
        """获取默认配置（当配置文件无法读取时使用）"""
        return {
            "uid": {
                "input_optimization_uid": {
                    "room_temperature": [f"mock_room_temp_{i}" for i in range(10)],
                    "ac_temperature": [f"mock_ac_temp_{i}" for i in range(6)],
                    "ac_onoff_setting": [f"mock_ac_onoff_{i}" for i in range(8)],
                    "ac_temperature_settings": [f"mock_ac_temp_set_{i}" for i in range(6)]
                }
            }
        }
    
    def generate_room_temperature(self, uid):
        """生成机房温度数据 (18-26°C)"""
        if uid in self.last_data:
            # 基于上次数据生成，模拟温度缓慢变化
            last_temp = self.last_data[uid]
            change = np.random.normal(0, 0.5)  # 小幅度变化
            new_temp = np.clip(last_temp + change, *self.room_temp_range)
        else:
            # 首次生成
            new_temp = np.random.uniform(*self.room_temp_range)
        
        self.last_data[uid] = new_temp
        return round(new_temp, 2)
    
    def generate_ac_temperature(self, uid):
        """生成空调温度数据 (17-30°C)"""
        if uid in self.last_data:
            last_temp = self.last_data[uid]
            change = np.random.normal(0, 0.3)
            new_temp = np.clip(last_temp + change, *self.ac_temp_range)
        else:
            new_temp = np.random.uniform(*self.ac_temp_range)
        
        self.last_data[uid] = new_temp
        return round(new_temp, 2)
    
    def generate_ac_onoff_status(self, uid):
        """生成空调开关状态 (0或1)"""
        if uid in self.last_data and random.random() > self.data_change_probability:
            # 大部分时间保持状态不变
            return self.last_data[uid]
        else:
            # 偶尔变化状态
            new_status = random.choice(self.ac_onoff_values)
            self.last_data[uid] = new_status
            return new_status
    
    def generate_ac_temperature_setting(self, uid):
        """生成空调温度设定值 (20-28°C)"""
        setting_range = (20, 28)
        if uid in self.last_data and random.random() > self.data_change_probability:
            return self.last_data[uid]
        else:
            new_setting = random.uniform(*setting_range)
            self.last_data[uid] = round(new_setting, 1)
            return self.last_data[uid]
    
    def generate_single_point_data(self, uid, data_type):
        """为单个测点生成数据"""
        if data_type == "room_temperature":
            return self.generate_room_temperature(uid)
        elif data_type == "ac_temperature":
            return self.generate_ac_temperature(uid)
        elif data_type == "ac_onoff_setting":
            return self.generate_ac_onoff_status(uid)
        elif data_type == "ac_temperature_settings":
            return self.generate_ac_temperature_setting(uid)
        else:
            # 默认生成随机数值
            return round(random.uniform(10, 50), 2)
    
    def generate_batch_data(self, uid_list):
        """批量生成数据，模拟Get_last_len_data的返回格式"""
        data = {}
        
        # 根据配置文件确定各个uid的数据类型
        input_optimization_uid = self.configs.get("uid", {}).get("input_optimization_uid", {})
        
        for uid in uid_list:
            # 确定数据类型
            data_type = "unknown"
            for key, uids in input_optimization_uid.items():
                if uid in uids:
                    data_type = key
                    break
            
            # 生成数据
            value = self.generate_single_point_data(uid, data_type)
            data[uid] = [value]  # 包装成列表，模拟pandas Series
        
        # 转换为DataFrame格式
        df = pd.DataFrame(data)
        return df
    
    def simulate_data_change(self):
        """模拟数据变化，增加真实感"""
        self.current_time += timedelta(seconds=60)
        
        # 随机修改一些历史数据，模拟传感器数据变化
        for uid in list(self.last_data.keys()):
            if random.random() < 0.05:  # 5%概率发生数据变化
                if uid in self.last_data:
                    self.last_data[uid] += random.uniform(-0.5, 0.5)

# 全局模拟数据生成器实例
_mock_generator = None

def get_mock_generator():
    """获取全局模拟数据生成器"""
    global _mock_generator
    if _mock_generator is None:
        _mock_generator = MockDataGenerator()
    return _mock_generator

def generate_mock_data(uid_list):
    """生成模拟数据的便捷函数"""
    generator = get_mock_generator()
    return generator.generate_batch_data(uid_list)

def simulate_time_progress():
    """模拟时间推进"""
    generator = get_mock_generator()
    generator.simulate_data_change()

if __name__ == "__main__":
    # 测试代码
    print("🧪 测试模拟数据生成器...")
    
    test_uids = [
        "mock_room_temp_1", "mock_room_temp_2",
        "mock_ac_temp_1", "mock_ac_temp_2", 
        "mock_ac_onoff_1", "mock_ac_onoff_2",
        "mock_ac_temp_set_1", "mock_ac_temp_set_2"
    ]
    
    generator = MockDataGenerator()
    
    print("生成第一批数据:")
    data1 = generator.generate_batch_data(test_uids)
    print(data1)
    
    print("\n生成第二批数据 (模拟时间变化):")
    generator.simulate_data_change()
    data2 = generator.generate_batch_data(test_uids)
    print(data2)
    
    print("\n✅ 模拟数据生成器测试完成！") 