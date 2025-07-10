from collections import OrderedDict
import os
import threading
import time
import yaml
import torch
from data_process import data_toMLP
from device import device_init
from get_data import get_predict_data, get_uid, new_uid
from output_data import updown
from model import train
import paho.mqtt.client as mqtt
import json
import pickle
import logging
from logging.handlers import TimedRotatingFileHandler


##模型更新
def thread_update(event):
    global update_flag
    try:
        while True:
            event.wait()
            logger.info("模型更新中")
            all_data = data_toMLP(
                devices,
                RP_u,
                init=0,
                host=host,
                port=port,
                database=database,
            )  # 获取总功率测点数据
            logger.info("获取最新数据中")
            save_model, _, _ = train(all_data, model_path, output_size=output_size)
            torch.save(save_model, model_path)
            logger.info("模型更新完成")
            lock.acquire()
            try:
                update_flag = 1
            finally:
                lock.release()
            del save_model
            del all_data
            event.clear()
    except Exception as e:
        logger.error("An error occurred: %s", str(e))


##1分钟内未传输完成，传上一时刻的值
def thread_monitor(event):
    global json_data
    try:
        while True:
            event.wait()
            start_time = time.time()
            while predicted == 0:
                current_time = time.time()
                elapsed_time = current_time - start_time
                if elapsed_time > 60:
                    client.publish(topic, json_data)
                    logger.info("传输上一时刻结果成功！")
                    start_time = time.time()
            event.clear()
    except Exception as e:
        logger.error("An error occurred: %s", str(e))


logger = logging.getLogger("my_logger")
logger.setLevel(logging.INFO)
handler = TimedRotatingFileHandler(
    "run.log", when="midnight", interval=1, backupCount=7
)
handler.setLevel(logging.INFO)
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)
try:
    # 载入参数
    with open("configs.yaml", "r") as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
    # 额定功率

    logger.info("机组额定功率为：%s", configs["RP_u"])
    # 检查设备uid、数目等是否匹配
    if len(configs["input_uid"]) != 1 + sum(configs["num_list"]):
        raise ValueError("输入设备列表与uid个数不匹配")

    # 初始化设备列表
    devices = device_init(configs["num_list"])

    # 配置设备参数
    RP_u = float(configs["RP_u"])
    input_uid = list(OrderedDict(configs["input_uid"]).items())
    devices = get_uid(devices, input_uid)
    for dev in devices:
        print(dev.node)
    output_size = len(devices)
    host = configs["influxdb"]["host"]
    port = configs["influxdb"]["port"]
    database = configs["influxdb"]["database"]

    # AI模型初始化
    if not os.path.exists("model"):
        os.mkdir("model")
    model_path = "model//model.pth"
    if not os.path.exists(model_path):
        logger.info("准备训练模型中...")
        logger.info("读取训练数据中...")
        all_data = data_toMLP(
            devices=devices,
            RP_u=RP_u,
            init=1,
            host=host,
            port=port,
            database=database,
        )
        logger.info("开始训练...")
        model, _, _ = train(all_data, model_path, init=True, output_size=output_size)
        logger.info("训练完成！")
        torch.save(model, model_path)
        del all_data
    else:
        logger.info("检测到当前路径已有模型，直接载入")
        model = torch.load(model_path)

    times = 1
    output_uid = list(OrderedDict(configs["output_uid"]).items())
    net_uid, format_uid = new_uid(output_uid)
    
    
    # mqtt连接
    # MQTT 代理的连接信息
    broker_address = configs["mqtt"]["broker_address"]
    mqtt_port = configs["mqtt"]["port"]
    topic = configs["mqtt"]["node"] + "/data/raw/point"
    client_id = configs["mqtt"]["client_id"]
    client = mqtt.Client(
        client_id=client_id,
        transport=configs["mqtt"]["transport"],
        clean_session=False,
    )
    client.connect(broker_address, mqtt_port)
    logger.info("连接成功！")

    update_flag = 0
    updating = 0
    old_form_out = 0
    old_net_out = 0
    lock = threading.Lock()
    event1 = threading.Event()
    event2 = threading.Event()

    thread1 = threading.Thread(target=thread_update, kwargs={"event": event1})
    thread1.start()
    thread2 = threading.Thread(target=thread_monitor, kwargs={"event": event2})
    thread2.start()

    old_data = get_predict_data(devices, host, port, database)
    while True:
        start_time = time.time()
        predicted = 0
        event2.set()
        new_data = get_predict_data(devices, host, port, database)
        if not new_data.equals(old_data):
            old_data = new_data
            times = (times + 1) % 1000
        lock.acquire()
        try:
            if update_flag == 1:
                updating = 1
                model = torch.load(model_path)
                updating = 0
                update_flag = 0
        finally:
            lock.release()
        # if updating == 1:
        #     model = torch.load(model_path)
        #     updating = 0
        with torch.no_grad():
            device = next(model.parameters()).device
            input_data = new_data.values.astype(float)
            new_data = new_data.astype(float)
            data = torch.Tensor(input_data).to(device)
            output = model(data)
            scaler = pickle.load(open("model//scaler.pkl", "rb"))
            net_out = scaler.inverse_transform(output.cpu().detach().numpy())

        end_time = time.time()
        predicted = 1
        sleep_time = 60 - (end_time - start_time) % 60
        time.sleep(sleep_time)

        ###输出接口
        json_data = updown(net_out, net_uid)
        client.publish(topic, json_data)
        logger.info("传输成功！")
        if times == 0:
            event1.set()

except Exception as e:
    logger.error("An error occurred: %s", str(e))
