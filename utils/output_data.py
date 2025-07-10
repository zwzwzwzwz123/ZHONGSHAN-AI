import time
from datetime import datetime, timedelta
import json

def Make_point_Energy_Saving_Suggestions(desc:str,space_id:str,generate_time:str,is_auto_execute:bool,control_list_str:str):
    '''
    得到Energy_Saving_Suggestions 节能建议 的point表
    '''
    table_name="ai_energy_saving_suggestions"
    Energy_Saving_Suggestions_data_dict = {
        "desc": desc,
        "space_id": space_id,
        "generate_time": generate_time,
        "is_auto_execute": is_auto_execute,
        "control_list": control_list_str
    }
    Energy_Saving_Suggestions_point = Make_point(table_name,Energy_Saving_Suggestions_data_dict)
    return Energy_Saving_Suggestions_point

def Make_point_Predict_Result(predict_result_dict:dict,time_rest:int = 300)->list[dict]:
    '''
    得到Predict_Result 预测数据对接 的point表
    '''
    time_stamp = int(time.time())
    table_name_list=["ai_predict_it_load","ai_predict_indoor_temp","ai_predict_air_inlet_temp","ai_predict_air_return_temp"]
    Predict_Result = []
    for table_name in table_name_list:
        temp_data_dict = {
            "create_time": time_stamp,
            "predict_value": predict_result_dict[table_name],
            "predict_interval": time_rest,
            "predict_time": time_stamp + time_rest
        }
        temp_point = Make_point(table_name,temp_data_dict)
        Predict_Result.append(temp_point)
    return Predict_Result


def Make_point_Algorithm_Log(start_time:int,end_time:int,status:int,type:str,content:str,remark:str):
    '''
    得到Algorithm_Log 算法执行日志 的point表
    '''
    table_name = "ai_algorithm_log"
    Algorithm_log_data_dict = {
        "start_time": start_time,
        "end_time": end_time,
        "status": status,
        "type": type,
        "content": content,
        "remark": remark
    }
    Algorithm_log_point = Make_point(table_name,Algorithm_log_data_dict)
    return Algorithm_log_point

def Make_point_Algorithm_Accuracy(value:int):
    '''
    得到Algorithm_Accuracy 算法准确率变化 的point表
    '''
    time_stamp = int(time.time()) * 1000000000
    table_name = "ai_algorithm_accuracy"
    Algorithm_Accuracy_data_dict = {
        "create_time": time_stamp,
        "value": value
    }
    Algorithm_Accuracy_point = Make_point(table_name,Algorithm_Accuracy_data_dict)
    return Algorithm_Accuracy_point

def Make_point(table_name:str,data_dict:dict):
    time_stamp = int(time.time()) * 1000000000
    point = {
        "measurement": table_name,
        "time": time_stamp,
        "fields": data_dict
    }
    return point

def Write_points(point_list, output_database):
    '''
    将points_list变为json格式后写入数据库
    '''
    # json_point_list = json.dumps(point_list)
    output_database.write_points(point_list)
    
def Make_control_list(optimization_result:list,
                      point_uid_list:list,
                      device_uid_list:list,
                      device_dems_point_name_list:list):
    '''
    得到制冷建议需要的control_list
    返回control_list
    '''
    control_list = []
    for set_value,point_uid,device_uid,device_dems_point_name in zip(optimization_result,point_uid_list,device_uid_list,device_dems_point_name_list):
        desc = f'{device_dems_point_name}设定为{set_value}'
        tmp_control_dict = {
            "desc": desc,
            "device_uid": device_uid,
            "point_uid": point_uid,
            "set_value": str(set_value)
        }
        control_list.append(tmp_control_dict)
    control_list_str = json.dumps(control_list,ensure_ascii=False)
    return control_list,control_list_str

def Make_together_desc(room_name:str,control_list:list)->str:
    '''
    得到总的制冷建议
    '''
    together_desc = f'建议把{room_name}'
    for control_dict in control_list:
        together_desc += control_dict["desc"]
        together_desc += ","
    together_desc = together_desc[:-1]
    return together_desc

def Check_output_bounds(variables:list, bounds_dict:dict)->list:
    """
    检测变量是否超过指定范围，并进行限制处理, 保证安全下发

    参数:
    variables (list): 需要检测的变量数组。
    bounds_dict (dict): 每个变量对应的边界值字典，格式为 {'变量名': (最小值, 最大值)}。

    返回:
    list: 处理后的变量数组，超出范围的变量将被修正为对应的最小值或最大值。

    示例:
    >>> variables = [10, 25, 5]
    >>> bounds_dict = {'var1': (0, 20), 'var2': (10, 30), 'var3': (0, 10)}
    >>> check_range(variables, bounds_dict)
    [10, 25, 5]
    """
    result = []
    for i, var in enumerate(variables):
        var_name = f'var{i+1}'  # 构造默认的变量名 var1, var2, var3, ...
        if var_name in bounds_dict:
            min_val, max_val = bounds_dict[var_name]
            if var < min_val:
                result.append(min_val)
            elif var > max_val:
                result.append(max_val)
            else:
                result.append(var)
        else:
            # 如果未提供边界值，默认不做处理，直接加入结果
            result.append(var)

    return result


def Make_PredictResult_to_Dict(output_name:list,predict_result:list)->dict:
    '''
    将预测模型输出结果和对应output_uid_list合并为后续需要的dict字典形式
    '''
    result_dict = dict(zip(output_name, predict_result))
    return result_dict

def Output_Alldata_to_Db(predict_accuracy_value, predict_result, predict_model_status, predict_result_dict_name, time_rest, model_name, optimization_result, configs):
    predict_result_name_list = configs['predict_result_name_list']
    predict_result_dict_name =  Make_PredictResult_to_Dict(predict_result_name_list, predict_result)
    Algorithm_Accuracy = Make_point_Algorithm_Accuracy(value=predict_accuracy_value)
    
    # Make Predict_Result 
    Predict_Result = Make_point_Predict_Result(predict_result_dict=predict_result_dict_name)
    
    # Make Algorithm_Log for prediction
    end_time = int(time.time())
    start_time = end_time-time_rest
    model_type = model_name
    content_predict = "Prediction"
    remark = "None"
    Algorithm_Log = Make_point_Algorithm_Log(start_time=start_time,end_time=end_time,status=predict_model_status,type=model_type,content=content_predict,remark=remark)

    # Make optimization desc
    device_dems_point_name_list = configs['device_dems_point_name_list']
    point_uid_list =configs["uid"]["output_optimization_uid"]
    device_uid_list = configs['uid']['device_uid_list']
    room_name = configs['room_name']
    control_list,control_list_str = Make_control_list(optimization_result=optimization_result,
                                    point_uid_list=point_uid_list,
                                    device_uid_list=device_uid_list, 
                                    device_dems_point_name_list=device_dems_point_name_list)

    # Make energy saving suggestions
    together_desc = Make_together_desc(room_name=room_name,control_list=control_list)
    space_id = configs['space_id']
    generate_time = int(time.time())
    is_auto_execute = False
    Energy_Saving_Suggestions = Make_point_Energy_Saving_Suggestions(desc=together_desc,space_id=space_id,generate_time=generate_time,is_auto_execute=is_auto_execute,control_list_str = control_list_str)
    
    # Get together and send at once
    points_list = Predict_Result+[Algorithm_Accuracy,Algorithm_Log,Energy_Saving_Suggestions]
    return points_list