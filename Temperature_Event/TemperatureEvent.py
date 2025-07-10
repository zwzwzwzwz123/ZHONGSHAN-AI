import threading
import time
import random

# 假设这是一个用来存储和订阅事件的类
class EventManager:
    def __init__(self):
        # 用于存储事件类型和对应的处理回调函数
        self.subscribers = {"temperature_rise": [], "temperature_fall": [], "temperature_stable": []}

    def subscribe(self, event_type, callback):
        # 订阅事件，添加回调函数到对应的事件类型列表
        if event_type in self.subscribers:
            self.subscribers[event_type].append(callback)

    def publish(self, event_type, data):
        # 发布事件，调用所有订阅该事件类型的回调函数
        if event_type in self.subscribers:
            for callback in self.subscribers[event_type]:
                callback(data)

# 模拟的温度传感器类
class TemperatureSensor:
    def __init__(self):
        # 初始温度随机设置在18到30度之间
        self.current_temperature = random.randint(18, 30)

    def read_temperature(self):
        # 模拟温度读取，每次读取温度可能会随机上下浮动1度
        change = random.choice([-1, 0, 1])
        self.current_temperature += change
        return self.current_temperature

def temperature_query(sensor, event_manager):
    # 初始化上一次的温度
    last_temperature = sensor.read_temperature()
    while True:
        time.sleep(1)  # 模拟每秒钟读取一次温度
        current_temperature = sensor.read_temperature()
        print(f"当前温度: {current_temperature}°C")

        # 判断温度变化并发布相应的事件
        if current_temperature > last_temperature:
            event_manager.publish("temperature_rise", current_temperature)
        elif current_temperature < last_temperature:
            event_manager.publish("temperature_fall", current_temperature)
        else:
            event_manager.publish("temperature_stable", current_temperature)

        # 更新上一次的温度记录
        last_temperature = current_temperature

def handle_temperature_rise(data):
    # 处理温度升高事件
    print(f"处理事件 - 温度升高到: {data}°C。需要检查冷却系统。")

def handle_temperature_fall(data):
    # 处理温度降低事件
    print(f"处理事件 - 温度降低到: {data}°C。可以降低冷却系统负荷。")

def handle_temperature_stable(data):
    # 处理温度稳定事件
    print(f"处理事件 - 温度保持不变: {data}°C。保持监控。")

def event_subscription(event_manager):
    # 订阅事件并绑定处理函数
    event_manager.subscribe("temperature_rise", handle_temperature_rise)
    event_manager.subscribe("temperature_fall", handle_temperature_fall)
    event_manager.subscribe("temperature_stable", handle_temperature_stable)

    # 线程保持运行，模拟持续订阅和处理
    while True:
        time.sleep(1)

if __name__ == "__main__":
    # 创建温度传感器和事件管理器实例
    sensor = TemperatureSensor()
    event_manager = EventManager()

    # 创建并启动温度查询线程
    query_thread = threading.Thread(target=temperature_query, args=(sensor, event_manager))
    query_thread.start()

    # 创建并启动事件订阅线程
    subscription_thread = threading.Thread(target=event_subscription, args=(event_manager,))
    subscription_thread.start()

    # 保持主线程运行
    query_thread.join()
    subscription_thread.join()
