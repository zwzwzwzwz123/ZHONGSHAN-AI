import time
import yaml
import logging
from logging.handlers import TimedRotatingFileHandler
from typing import List
import pandas as pd
from influxdb import InfluxDBClient
from get_data import Get_last_len_data
from utils.output_data import (
    Make_point_Energy_Saving_Suggestions,
    Make_point_Predict_Result,
    Make_point_Algorithm_Accuracy,
    Make_point_Algorithm_Log,
    Make_control_list,
    Make_together_desc,
    Output_Alldata_to_Db,
)
from utils.interfaces import OptimizationInput, OptimizationOutput
from Optimization_Algorithm.custom_oa_V2 import (
    Custom_Optimization_V2 as Custom_Optimization,
)
from utils.help_functions import find_different_indices
import copy


def Init_logger():
    """
    初始化日志
    """
    logger = logging.getLogger("my_logger")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler("run.log", encoding="utf-8", mode="a")
    handler = TimedRotatingFileHandler(
        "run.log", when="midnight", interval=1, backupCount=1
    )
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    return logger


def Init_input_database(configs):
    """
    初始化 输入数据库
    """
    host = configs["influxdb"]["influxdbin"]["host"]
    port = configs["influxdb"]["influxdbin"]["port"]
    username = configs["influxdb"]["influxdbin"]["username"]
    password = configs["influxdb"]["influxdbin"]["password"]
    database_name = configs["influxdb"]["influxdbin"]["database_name"]
    timeout = 10
    database = InfluxDBClient(
        host=host, port=port, timeout=timeout, username=username, password=password
    )
    database.switch_database(database_name)
    return database


def Init_output_database(configs):
    """
    初始化 输出数据库
    """
    host = configs["influxdb"]["influxdbout"]["host"]
    port = configs["influxdb"]["influxdbout"]["port"]
    username = configs["influxdb"]["influxdbout"]["username"]
    password = configs["influxdb"]["influxdbout"]["password"]
    database_name = configs["influxdb"]["influxdbout"]["database_name"]
    timeout = 10
    database = InfluxDBClient(host=host, port=port, timeout=timeout, username=username, password=password)
    #此处存在数据库切换问题
    #database.switch_database("test_822")
    #database.switch_database(database_name)
    #database.switch_database("database")
    database.switch_database(database_name)
    return database


def Init_custom_optimization(
        oa_configs: dict, last_optimization_input_data: OptimizationInput
) -> Custom_Optimization:
    """
    初始化 定制化优化算法
    """
    custom_optimization = Custom_Optimization(
        last_optimization_input=last_optimization_input_data, **oa_configs
    )
    return custom_optimization


def Filter_Optimization_input_data(
        new_data: pd.DataFrame, input_optimization_uid_list: dict[list]
) -> OptimizationInput:
    """
    从新读取的数据中筛选获得用于作为优化输入的参数
    """
    room_temperature = new_data[
        input_optimization_uid_list["room_temperature"]
    ].values.tolist()[0]
    ac_onoff_setting = new_data[
        #此处原始代码存在问题，原始代码已经注释
        #input_optimization_uid_list["ac_onoff_status_setting"]
        input_optimization_uid_list["ac_onoff_setting"]
    ].values.tolist()[0]
    ac_temperature = new_data[
        input_optimization_uid_list["ac_temperature"]
    ].values.tolist()[0]
    ac_temperature_setting = new_data[
        #此处原始代码存在问题，原始代码已经注释
        #input_optimization_uid_list["ac_temperature_setting"]
        input_optimization_uid_list["ac_temperature_settings"]
    ].values.tolist()[0]
    
    """
    注释掉压缩机部分
    converter_freq_setting = new_data[
        input_optimization_uid_list["converter_freq_setting"]
    ].values.tolist()[0]
    
    """
    
    optimizationinput_key = [
        "room_temperature",
        "ac_onoff_status_setting",
        "ac_temperature",
        "ac_temperature_setting",
        #注释掉压缩机部分
        #"converter_freq_setting", 
       
    ]
    input = [
        room_temperature,
        ac_onoff_setting,
        ac_temperature,
        ac_temperature_setting,
        #注释掉压缩机部分
        #converter_freq_setting,
    ]
    optimization_input = dict(zip(optimizationinput_key, input))
    return optimization_input


def Get_optimization_result(
        input_data: OptimizationInput, custom_optimization: Custom_Optimization
) -> OptimizationOutput:
    """
    获得优化结果
    """
    optimization_output = custom_optimization.handle_optimization_process(input_data)
    return optimization_output


def Prepare_optimization_output(
        optimization_input_data: OptimizationInput,
        optimization_result: OptimizationOutput,
        configs,
):
    # device_uid_list
    ac_device_uid_list = configs["uid"]["device_uid"]["ac"]
    #注释掉压缩机部分
    #converter_device_uid_list = configs["uid"]["device_uid"]["converter"]

    # 获取映射关系
    ac_onoff_to_device_mapping = configs["mapping"]["ac_onoff_to_device"]
    ac_temp_to_device_mapping = configs["mapping"]["ac_temp_to_device"]

    # point_uid_list
    ac_onoff_point_uid_list = configs["uid"]["input_optimization_uid"][
        #此处原始代码存在问题，原始代码已经注释
        #"ac_onoff_status_setting"
        "ac_onoff_setting"
    ]
    ac_temp_point_uid_list = configs["uid"]["input_optimization_uid"][
        #此处原始代码存在问题，原始代码已经注释
        #"ac_temperature_setting"
        "ac_temperature_settings"
    ]
    """
    注释掉压缩机部分
    converter_freq_point_uid_list = configs["uid"]["input_optimization_uid"][
        "converter_freq_setting"
    ]
    """
    # name列表初始化
    ac_onoff_status_setting_name_list = configs["name"]["optimization"][
        "ac_onoff_status_setting"
    ]
    ac_temp_setting_name_list = configs["name"]["optimization"][
        "ac_temperature_setting"
    ]
    """
    注释掉压缩机部分
    converter_freq_setting_name_list = configs["name"]["optimization"][
        "converter_freq_setting"
    ]
    """
    # 获得本次优化的结果indices
    ac_onoff_indices = find_different_indices(
        optimization_input_data["ac_onoff_status_setting"],
        optimization_result["ac_onoff_status_setting"],
    )
    ac_temp_setting_indices = find_different_indices(
        optimization_input_data["ac_temperature_setting"],
        optimization_result["ac_temp_setting"],
    )
    """
    注释掉压缩机部分
    converter_freq_setting_indices = find_different_indices(
        optimization_input_data["converter_freq_setting"],
        optimization_result["converter_freq_setting"],
    )
    """
    
    # 根据indices得到改变的names名单
    ac_onoff_names = [ac_onoff_status_setting_name_list[i] for i in ac_onoff_indices]
    ac_temp_names = [ac_temp_setting_name_list[i] for i in ac_temp_setting_indices]
    """
    注释掉压缩机部分
    converter_freq_names = [
        converter_freq_setting_name_list[i] for i in converter_freq_setting_indices
    ]
    """
    
    #删去压缩机部分
    #all_point_name_list = ac_onoff_names + ac_temp_names + converter_freq_names
    all_point_name_list = ac_onoff_names + ac_temp_names
    # 根据indices得到改变的values数值
    ac_onoff_values = [
        optimization_result["ac_onoff_status_setting"][i] for i in ac_onoff_indices
    ]
    ac_temp_values = [
        optimization_result["ac_temp_setting"][i] for i in ac_temp_setting_indices
    ]
    """
    注释掉压缩机部分
    converter_freq_values = [
        optimization_result["converter_freq_setting"][i]
        for i in converter_freq_setting_indices
    ]
    """
    
    #删去压缩机部分
    #all_change_value_list = ac_onoff_values + ac_temp_values + converter_freq_values
    all_change_value_list = ac_onoff_values + ac_temp_values
    
    # 根据indices得到改变的point_uid名单
    ac_onoff_point_uid = [ac_onoff_point_uid_list[i] for i in ac_onoff_indices]
    ac_temp_point_uid = [ac_temp_point_uid_list[i] for i in ac_temp_setting_indices]
    """
    注释掉压缩机部分
    converter_freq_point_uid = [
        converter_freq_point_uid_list[i] for i in converter_freq_setting_indices
    ]
    """
    all_point_uid_list = (
            #删去压缩机部分
            #ac_onoff_point_uid + ac_temp_point_uid + converter_freq_point_uid
            ac_onoff_point_uid + ac_temp_point_uid
    )

    # 根据indices得到改变的device_uid名单 (使用映射关系)
    ac_onoff_device_uid = [ac_device_uid_list[ac_onoff_to_device_mapping[i]] for i in ac_onoff_indices]
    ac_temp_device_uid = [ac_device_uid_list[ac_temp_to_device_mapping[i]] for i in ac_temp_setting_indices]
    """
    注释掉压缩机部分
    converter_freq_device_uid = [
        converter_device_uid_list[i] for i in converter_freq_setting_indices
    ]
    """
    all_device_uid_list = (
            #删去压缩机部分
            #ac_onoff_device_uid + ac_temp_device_uid + converter_freq_device_uid
            ac_onoff_device_uid + ac_temp_device_uid
    )

    return (
        all_change_value_list,
        all_point_uid_list,
        all_device_uid_list,
        all_point_name_list,
    )


# 载入参数
with open("./config/configs.yaml", "r", encoding="utf-8") as f:
    configs = yaml.load(f, Loader=yaml.FullLoader)
with open("./config/model_args.yaml", "r", encoding="utf-8") as f:
    model_args = yaml.load(f, Loader=yaml.FullLoader)
with open("./config/oa_args.yaml", "r", encoding="utf-8") as f:
    oa_args = yaml.load(f, Loader=yaml.FullLoader)

# 数据库初始化
input_database = Init_input_database(configs)
output_database = Init_output_database(configs)
logger = Init_logger()

# uid列表初始化
all_uid_list = []
optimizationinput_key = [
    "ac_temperature",
    "room_temperature",
    "ac_onoff_setting",
    "ac_temperature_settings",
]    # 删除了 "converter_freq_setting",
for input_key in optimizationinput_key:
    all_uid_list += configs["uid"]["input_optimization_uid"][input_key]
input_optimization_uid_list = configs["uid"]["input_optimization_uid"]
output_optimization_uid_list = configs["uid"]["output_optimization_uid"]

print("初始化完毕")

# 初始化最后一条数据
old_data = Get_last_len_data(all_uid_list, input_database)
print("获取最后一条数据")
last_optimization_input_data = Filter_Optimization_input_data(
    old_data, input_optimization_uid_list
)
print("获取最后一条优化输入数据")

# 初始化自定义优化算法
custom_optimization = Init_custom_optimization(
    oa_configs=oa_args["Custom_Optimization"],
    last_optimization_input_data=last_optimization_input_data,
)

while True:
    start_time = time.time()

    # 获取最新一条数据
    new_data = Get_last_len_data(all_uid_list, input_database)
    if not new_data.equals(old_data):  # 如果数据不相等，即发生数据更新
        old_data = new_data  # 更新数据

    # 得到优化输入，获取优化结果
    optimization_input_data: OptimizationInput = Filter_Optimization_input_data(
        new_data, input_optimization_uid_list
    )
    print(f"\n optimization input is \n {optimization_input_data}")

    optimization_result: OptimizationOutput = Get_optimization_result(
        optimization_input_data, custom_optimization
    )
    print(
        f"last optimization input is \n {custom_optimization.last_optimization_input}"
    )
    print(f"optimization result is \n {optimization_result} \n")

    # 根据优化结果，得到 最终输出数据的应有格式
    (
        all_change_value_list,
        all_point_uid_list,
        all_device_uid_list,
        all_point_name_list,
    ) = Prepare_optimization_output(
        optimization_input_data, optimization_result, configs
    )

    # 得到 输出用的control_list 和 desc
    control_list, control_list_str = Make_control_list(
        optimization_result=all_change_value_list,
        point_uid_list=all_point_uid_list,
        device_uid_list=all_device_uid_list,
        device_dems_point_name_list=all_point_name_list,
    )

    together_desc = Make_together_desc(
        room_name=configs["room_name"], control_list=control_list
    )

    # 配置信息，得到最终输出的建议
    space_id = configs["space_id"]
    generate_time = int(time.time())
    is_auto_execute = False
    Energy_Saving_Suggestions = Make_point_Energy_Saving_Suggestions(
        desc=together_desc,
        space_id=space_id,
        generate_time=generate_time,
        is_auto_execute=is_auto_execute,
        control_list_str=control_list_str,
    )

    # 输出最终输出建议
    output_database.write_points([Energy_Saving_Suggestions])

    end_time = int(time.time())
    # sleep_time = 60 - (end_time - start_time) % 60
    sleep_time = 60
    time.sleep(sleep_time)
