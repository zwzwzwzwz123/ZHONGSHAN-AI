from typing import TypedDict,Literal,Tuple

class SafetyBoundary(TypedDict):
    '''
    安全边界 Dict接口
    '''
    room_temp_lower_boundary: list[float]
    room_temp_higher_boundary: list[float]
    ac_temp_lower_boundary: list[float]
    ac_temp_higher_boundary: list[float]

class ObservingBoundary(TypedDict):
    '''
    观察边界 Dict接口
    '''
    room_temp_boundary: list[float]
    ac_temp_boundary: list[float]

class OptimizationInput(TypedDict):
    '''
    优化算法输入 Dict 接口
    '''
    room_temperature: list[float] # 房间温度
    ac_onoff_status_setting: list[int] # 空调开启与否
    ac_temperature: list[float]  # 空调回风温度
    ac_temperature_setting: list[float] # 空调设定回风温度
    #注释掉变频器部分
    #converter_freq_setting: list[int] # 变频器设定频率

class OptimizationOutput(TypedDict):
    '''
    优化算法输出 Dict 接口
    '''
    ac_onoff_status_setting: list[int] # 空调开启与否状态设定 on为1,off为0
    ac_temp_setting: list[float] # 空调设定温度设定
    #注释掉变频器部分
    #converter_freq_setting: list[float] # 变频器设定频率