from collections import OrderedDict
import os
import threading
import time
import yaml
import torch
import paho.mqtt.client as mqtt
import json
import pickle
import logging
from logging.handlers import TimedRotatingFileHandler
from typing import List
import pandas as pd
from influxdb import InfluxDBClient
import numpy as np
import random
from get_data import new_uid, Get_last_len_data,Get_all_training_data
# from output_data import updown, oa_desc
from utils.output_data import Make_point_Energy_Saving_Suggestions, Make_point_Predict_Result, Make_point_Algorithm_Accuracy, Make_point_Algorithm_Log, Make_control_list, Make_together_desc, Output_Alldata_to_Db
from model import train
from sklearn.metrics import mean_squared_error
from Model_Manager.model_manager import Model_Manager
from OptimizationAlgorithm_Manager.oa_manager import OptimizationAlgorithm_Manager
from utils.interfaces import SafetyBoundary,OptimizationInput,OptimizationOutput,ObservingBoundary
from Optimization_Algorithm.custom_oa import Custom_Optimization
import traceback

def thread_update(event, model_args, database, input_uid_list, output_uid_list, all_uid_list,model_path):
    """
    模型更新
    """
    init = 0
    try:
        while True:
            event.wait()
            logger.info("模型更新中")
            #用uid_list根据不同房间获取数据
            all_data = Get_all_training_data(all_uid_list, average=False, init=init, database=database)
            logger.info("获取最新数据中")
            try:
                save_model = train(df=all_data,
                                   manager=model_manager,
                                   model_name=model_name,
                                   model_path=model_path,
                                   input_columns=input_uid_list,
                                   output_columns=output_uid_list,
                                   args=model_args,
                                   init=init)
                logger.info("模型更新完成")
            except Exception as e:
                logger.error("An error occurred when updating model: %s", str(e))

            lock.acquire()
            try:
                model_update_flag = 1
            finally:
                lock.release()

            del save_model
            del all_data
            event.clear()
    except Exception as e:
        logger.error("An error occurred: %s", str(e))


def thread_monitor(event):
    """
    1分钟内未传输完成, 传上一时刻的值
    """
    try:
        while True:
            event.wait()
            start_time = time.time()
            while predicted == 0:
                current_time = time.time()
                elapsed_time = current_time - start_time
                if elapsed_time > 60:
                    # client.publish(topic, predict_data)
                    logger.info("传输上一时刻结果成功！")
                    start_time = time.time()
            event.clear()
    except Exception as e:
        logger.error("An error occurred: %s", str(e))


def Init_predict_model(model_args, database, input_uid_list, output_uid_list, all_uid_list, model_path,device):
    '''
    初始化预测模型
    '''
    init = 1
    if not os.path.exists(model_path):
        logger.info("准备训练模型中...")
        logger.info("读取训练数据中...")
        all_data = Get_all_training_data(all_uid_list, average=False, init=init, database=database)
        logger.info("开始训练...")
        model = train(df=all_data,
                    manager=model_manager,
                    model_name=model_name,
                    model_path=model_path,
                    input_columns=input_uid_list,
                    output_columns=output_uid_list,
                    args=model_args,
                    init=init,
                    device=device)
        logger.info("模型已保存，训练完成！")
        del all_data
    else:
        logger.info("检测到当前路径已有模型，直接载入")
        model = torch.load(model_path)
    return model

def Init_logger():
    '''
    初始化日志
    '''
    logger = logging.getLogger("my_logger")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler("run.log",encoding="utf-8", mode="a")
    handler = TimedRotatingFileHandler(
        "run.log", when="midnight", interval=1, backupCount=7
    )
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    return logger


def Init_MQTT_client(configs):
    '''
    初始化MQTT客户端
    '''
    
    broker_address = configs["mqtt"]["broker_address"]
    mqtt_port = configs["mqtt"]["port"]
    topic = configs["mqtt"]["node"] + "/data/raw/point"
    client_id = configs["mqtt"]["client_id"]
    client = mqtt.Client(
        client_id=client_id,
        transport=configs["mqtt"]["transport"],
        clean_session=False,
    )
    try:
        client.connect(broker_address, mqtt_port)
        logger.info("连接MQTT成功!")
    except Exception as e:
        logger.error("An error occurred when connecting MQTT: %s", str(e))
        raise e
    return client, topic

def Init_input_database(configs):
    '''
    初始化 输入数据库
    '''
    host = configs["influxdb"]["influxdbin"]["host"]
    port = configs["influxdb"]["influxdbin"]["port"]
    username = configs["influxdb"]["influxdbin"]["username"]
    password= configs["influxdb"]["influxdbin"]["password"]
    database_name = configs["influxdb"]["influxdbin"]["database_name"]
    timeout= 10
    database = InfluxDBClient(host=host, port=port, timeout=timeout, username=username,password=password)
    database.switch_database(database_name)
    return database

def Init_output_database(configs):
    '''
    初始化 输出数据库
    '''
    host = configs["influxdb"]["influxdbout"]["host"]
    port = configs["influxdb"]["influxdbout"]["port"]
    username = configs["influxdb"]["influxdbout"]["username"]
    password = configs["influxdb"]["influxdbout"]["password"]
    database_name = configs["influxdb"]["influxdbout"]["database_name"]
    timeout= 10
    database = InfluxDBClient(host=host, port=port, timeout=timeout, username=username, password=password)
    database.switch_database(database_name)
    return database

def Init_custom_optimization(oa_configs:dict,last_optimization_input_data:OptimizationInput)->Custom_Optimization:
    '''
    初始化 定制化优化算法
    '''
    custom_optimization = Custom_Optimization(
        last_optimization_input=last_optimization_input_data,**oa_configs
    )
    return custom_optimization

def Get_predict_result(input_data:pd.DataFrame, model):
    '''
    获取预测结果
    输入{
        model 预测模型
        input_data 预测模型输入数据 （归一化的）
        }
    返回 predict_result 预测结果
    '''
    with torch.no_grad():
        input_data = input_data.values
        predict_result = model.pred(input_data)
    return predict_result

def Count_accuracy(real_data, predict_result):
    '''
    获取准确率
    输入{
       测试数据
        }
    返回 mse 准确率的值
    '''
    if predict_result == 0:
        return 0
    mse = mean_squared_error(real_data, predict_result)
    return mse

def Get_optimization_result(input_data:OptimizationInput,custom_optimization:Custom_Optimization)->list:
    '''
    获得优化结果
    '''
    optimization_output = custom_optimization.handle_optimization_process(input_data)
    pass


def Renew_output_data(predict_result, optimize_result):
    '''
    更新输出数据
    '''
    output_data = {}
    output_data["predict_result"] = predict_result
    output_data["optimize_result"] = optimize_result
    return output_data


def Make_PredictResult_to_Dict(output_name:list,predict_result:list)->dict:
    '''
    将预测模型输出结果和对应output_uid_list合并为后续需要的dict字典形式
    '''
    result_dict = dict(zip(output_name, predict_result))
    return result_dict


def Filter_Current_temperature(new_data:pd.DataFrame,current_temp_uid_list:list)->pd.DataFrame:
    '''
    从新读取的数据中筛选获得用于作为优化决策的房间当前温度
    '''
    return new_data[current_temp_uid_list]

def Filter_PredictModel_input_data(new_data:pd.DataFrame,in_predict_uid_list:list)->pd.DataFrame:
    '''
    从新读取的数据中筛选获得用于作为预测输入的参数
    !Normalized
    '''
    new_data=new_data[in_predict_uid_list]
    [mean, std] = np.load('model/x_scale.npy')
    new_data = (new_data-mean)/std
    return new_data

def Filter_Optimization_input_data(new_data:pd.DataFrame,input_optimization_uid_list:dict[list])->OptimizationInput:
    '''
    从新读取的数据中筛选获得用于作为优化输入的参数
    '''
    room_temperature = new_data[input_optimization_uid_list['room_temperature']]
    ac_onoff_setting = new_data[input_optimization_uid_list['ac_onoff_setting']]
    ac_temperature = new_data[input_optimization_uid_list['ac_temperature']]
    ac_temperatue_settings = new_data[input_optimization_uid_list['ac_temperatue_settings']]
    optimizationinput_key = ['room_temperature', 'ac_onoff_setting', 'ac_temperature', 'ac_temperatue_settings']
    input = [room_temperature, ac_onoff_setting, ac_temperature, ac_temperatue_settings]
    optimization_input = dict(zip(optimizationinput_key, input))
    return optimization_input

def Filter_PredictModel_output_data(new_data:pd.DataFrame,out_predict_uid_list:list)->pd.DataFrame:
    '''
    从新读取的数据中筛选获得用于作为预测输出(标签)的参数
    '''
    return new_data[out_predict_uid_list]

def Inverse_Normalization(predict_data:torch.Tensor)->torch.Tensor:
    '''
    从新读取的数据中筛选获得用于作为预测输入的参数
    !Normalized
    '''
    [mean, std] = np.load('model/y_scale.npy')
    if type(predict_data) is torch.Tensor:
        predict_data = predict_data.cpu().detach().numpy()*std+mean
    else:
        predict_data=predict_data*std+mean
    return predict_data

def main():
    # 创建日志
    global logger
    global lock
    global model_update_flag
    global predicted

    global model_name
    global model_manager
    # global predict_model
    # global model_path

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    try:
        # 载入参数
        with open("./config/configs.yaml", "r",encoding="utf-8") as f:
            configs = yaml.load(f, Loader=yaml.FullLoader)
        with open("./config/model_args.yaml", "r",encoding="utf-8") as f:
            model_args = yaml.load(f, Loader=yaml.FullLoader)
        with open("./config/oa_args.yaml", "r",encoding='utf-8') as f:
            oa_args = yaml.load(f, Loader=yaml.FullLoader)
            
        # 数据库初始化
        input_database = Init_input_database(configs)
        output_database = Init_output_database(configs)
        # client, topic = Init_MQTT_client(configs)
        logger = Init_logger()

        # uid列表初始化
        all_uid_list = configs["uid"]["all_uid"]
        # current_temperature_uid_list = configs["uid"]["server_room_temperature_uid"]
        input_predict_uid_list = configs["uid"]["input_predict_uid"]
        output_predict_uid_list = configs["uid"]["output_predict_uid"]
        # input_optimization_uid_list = configs["uid"]["input_optimization_uid"]
        # output_optimization_uid_list = configs["uid"]["output_optimization_uid"]

        # 模型初始化
        model_name = configs["model_name"]
        model_manager = Model_Manager()
        if not os.path.exists("model"):
            os.mkdir("model")
        model_path = rf"model//{model_name}.pth"
        predict_model = Init_predict_model(model_args[model_name],database=input_database, input_uid_list=input_predict_uid_list, output_uid_list=output_predict_uid_list, all_uid_list=all_uid_list, model_path=model_path,device=device)
        time_rest = configs["time_rest"]

        # 获取优化算法边界
        boundary = configs['boundary']

        # 初始化 数据更新计数器和模型更新标志
        new_data_counter = 1
        model_update_flag = 0

        print("初始化完毕")

        # 设置线程
        lock = threading.Lock()
        event1 = threading.Event()
        event2 = threading.Event()
        thread1 = threading.Thread(target=thread_update, kwargs={"event": event1, "model_args": model_args[model_name], "database": input_database, "input_uid_list":input_predict_uid_list, "output_uid_list":output_predict_uid_list, "all_uid_list":all_uid_list, "model_path":model_path})
        thread1.start()
        # thread2 = threading.Thread(target=thread_monitor, kwargs={"event": event2,"output_database": output_database})
        # thread2.start()
        thread2 = threading.Thread(target=thread_monitor, kwargs={"event": event2})
        thread2.start()
        print("线程设置完毕")


        # 初始化最后一条数据
        old_data = Get_last_len_data(all_uid_list, input_database)

        # 初始化优化边界、优化所需上一条数据
        # optimization_boundary = configs["optimization"]["optimization_boundary"]
        # last_optimization_result = configs["optimization"]["default_optimization_result"]
        # last_input_data = Filter_Optimization_input_data(old_data, input_optimization_uid_list)
        # last_optimization_input_data = Filter_Optimization_input_data(old_data, input_optimization_uid_list)


        # 初始化自定义优化算法
        # custom_optimization = Init_custom_optimization(oa_configs=oa_args,last_optimization_input_data=last_optimization_input_data)
        # 初始化默认预测结果
        last_pred_result = 0
        while True:
            start_time = time.time()
            predicted = 0
            event2.set()

            # 获取最新一条数据
            new_data = Get_last_len_data(all_uid_list, input_database)
            if not new_data.equals(old_data): # 如果数据不相等，即发生数据更新
                old_data = new_data # 更新数据
                new_data_counter = (new_data_counter + 1) % 1000 # 计数器+1
                if new_data_counter == 0: # 每1000次数据更新，开启模型更新线程
                    event1.set()
            print("读取了最新的数据，最新的数据为", new_data)

            # 紧接着，获取最新sequence_length条数据用于预测
            # NOTE: 确认有新数据更新，才更新？
            new_predict_data = Get_last_len_data(all_uid_list, input_database, model_args[model_name]["sequence_length"])
        
            # 检查模型是否更新
            lock.acquire() # 加锁
            try:
                if model_update_flag == 1:
                    predict_model_status = 0
                    predict_model = torch.load(model_path) # 重新载入模型
                    model_update_flag = 0
                else:
                    predict_model_status = 1
            finally:
                lock.release()

            # 获取预测结果
            predict_input_data = Filter_PredictModel_input_data(new_predict_data, input_predict_uid_list)
            predict_result = Get_predict_result(predict_input_data, predict_model)
            predict_result= Inverse_Normalization(predict_result)
            print(f"predict_input_data is {predict_input_data}")
            print(f"predict_result is {predict_result}")

            predicted = 1

            # 获取优化结果
            # optimization_input_data:OptimizationInput = Filter_Optimization_input_data(new_data, input_optimization_uid_list)
            # print(f"optimization_input_data is {optimization_input_data}")
            # optimization_result:OptimizationOutput = Get_optimization_result(optimization_input_data,custom_optimization)
            # last_optimization_result = optimization_result # 记录上一次优化结果
            # last_input_data = optimization_input_data #  记录上一次优化输入
            # print(f"optimization_result is {optimization_result}")

            # 输出接口
            # count accurancy and renew pred result
            # real_data = Filter_PredictModel_output_data(new_data,output_predict_uid_list)
            # predict_accuracy_value = Count_accuracy(real_data, predict_result=last_pred_result)
            # print(f"predict_accuracy_value is {predict_accuracy_value}")
            # last_pred_result = predict_result
            
            # Get together and send at once
            # points_list = Output_Alldata_to_Db(
            #     predict_accuracy_value=predict_accuracy_value,
            #     predict_result=predict_result,
            #     predict_model_status=predict_model_status,
            #     time_rest=time_rest,
            #     model_name=model_name,
            #     optimization_result=optimization_result,
            #     configs=configs
            #     )
            # print(f"points list is {points_list}")
            # output_database.write_points(points_list)
            # print("output finished.")

            end_time = int(time.time())
            sleep_time = 60 - (end_time - start_time) % 60
            time.sleep(sleep_time)


    except Exception as e:
        logger.error("An error occurred: %s", str(e))
        logger.error("Traceback: %s", traceback.format_exc())

main()