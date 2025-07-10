from datetime import datetime
import pandas as pd
from influxdb import InfluxDBClient
import warnings
import logging

# 导入模拟数据生成器
try:
    from mock_data_generator import generate_mock_data, simulate_time_progress
    MOCK_AVAILABLE = True
except ImportError:
    MOCK_AVAILABLE = False
    warnings.warn("模拟数据生成器不可用，数据库连接失败时程序将报错")

# 全局变量：标记是否使用模拟数据模式
_use_mock_data = False
_mock_mode_announced = False

def enable_mock_mode():
    """启用模拟数据模式"""
    global _use_mock_data, _mock_mode_announced
    _use_mock_data = True
    if not _mock_mode_announced:
        print("🔄 切换到模拟数据模式 - 数据库连接失败")
        print("📊 使用模拟数据继续运行程序...")
        _mock_mode_announced = True

def is_mock_mode():
    """检查是否在模拟数据模式"""
    return _use_mock_data

def test_database_connection(database):
    """测试数据库连接"""
    try:
        database.ping()
        return True
    except:
        return False

def min_average(df):
    """
    函数功能:同一分钟的数据取平均（防止数据过大）
    输入{
        df:数据集
    }
    """
    new_data = pd.DataFrame(columns=df.columns)
    df["time"] = df["time"].apply(lambda x: ":".join(x.split(":")[:-1]))
    common_time = df.at[0, "time"]
    common_data = []
    for index, row in df.iterrows():
        if row["time"] != common_time:
            new_row = df.iloc[common_data[0], 1:]
            if len(common_data) != 1:
                for i in common_data[1:]:
                    new_row = new_row + df.iloc[i, 1:]
            new_row = new_row / len(common_data)
            new_row["time"] = common_time
            new_data.loc[len(new_data)] = new_row

            common_time = row["time"]
            common_data.clear()
            common_data.append(index)
        else:
            common_data.append(index)
    return new_data

def Read_training_data(uid, node_name, init, database):
    """
    函数功能:从数据库读取训练数据

    init为1代表初次训练模型，为0代表更新模型
    如果init为1，即初次训练模型，从数据库中选取最近的5000条记录。
    如果init为0，即更新模型，从数据库中选取最近的1000条记录。
    """
    if init == 1:
        selected = database.query(
            f'SELECT * FROM "{uid}" ORDER BY time DESC LIMIT 100'
        )
    if init == 0:
        selected = database.query(
            f'SELECT * FROM "{uid}" ORDER BY time DESC LIMIT 50'
        )
    df = pd.DataFrame(columns=["time", "abs_value", "origin_value", "value"])
    for select in selected:
        df = pd.DataFrame.from_records(select)
    df = df.drop(columns=["abs_value", "origin_value"])
    df = df.rename(columns={"value": node_name})
    return df

def Get_all_training_data(uid_list, average, init, database):
    """
    函数功能:获得所有测点的训练数据
    """
    df = pd.DataFrame()
    # 获取各设备数据
    for uid in uid_list:
        node_name = uid
        temp = Read_training_data(
            uid=uid,
            node_name=node_name,
            init=init,
            database=database,
        )
        assert not temp.empty
        if average == True:
            temp = min_average(temp)
        df = pd.concat([df, temp[node_name]], axis=1)
    return df


def Read_last_len_data(
    uid,
    database,
    sequence_length=1,
):
    """
    函数功能:从数据库读取最新 sequence_length 条数据
    支持数据库连接失败时自动切换到模拟数据
    """
    global _use_mock_data
    
    # 如果已经在模拟模式，直接使用模拟数据
    if _use_mock_data:
        if MOCK_AVAILABLE:
            mock_data = generate_mock_data([uid])
            return mock_data[uid]
        else:
            raise RuntimeError("模拟数据生成器不可用，且数据库连接失败")
    
    # 尝试从数据库读取数据
    try:
        selected = database.query(f'SELECT * FROM "{uid}" ORDER BY time DESC LIMIT {sequence_length}')
        df = pd.DataFrame()
        for select in selected:
            df = pd.DataFrame.from_records(select)
        
        if df.empty:
            # 数据库查询成功但无数据，尝试模拟数据
            if MOCK_AVAILABLE:
                print(f"⚠️  测点 {uid} 无数据，使用模拟数据")
                mock_data = generate_mock_data([uid])
                return mock_data[uid]
            else:
                raise ValueError(f"测点 {uid} 无数据且模拟数据不可用")
        
        value = df["value"]
        return value
        
    except Exception as e:
        # 数据库连接或查询失败，切换到模拟模式
        if MOCK_AVAILABLE:
            enable_mock_mode()
            mock_data = generate_mock_data([uid])
            return mock_data[uid]
        else:
            print(f"❌ 数据库查询失败: {e}")
            print("❌ 模拟数据生成器不可用")
            raise e

def Get_last_len_data(uid_list, database, sequence_length=1):
    """
    函数功能:获取所有最新 sequence_length 数据
    支持数据库连接失败时自动切换到模拟数据模式
    """
    global _use_mock_data
    
    # 如果已经在模拟模式，直接使用模拟数据
    if _use_mock_data:
        if MOCK_AVAILABLE:
            simulate_time_progress()  # 模拟时间推进，增加数据真实感
            return generate_mock_data(uid_list)
        else:
            raise RuntimeError("模拟数据生成器不可用，且数据库连接失败")
    
    # 尝试从数据库获取数据
    try:
        df = pd.DataFrame()
        for uid in uid_list:
            node_name = uid
            temp = Read_last_len_data(uid=uid, database=database, sequence_length=sequence_length)
            new_data = pd.DataFrame({node_name: temp})
            df = pd.concat([df, new_data], axis=1)
        return df
        
    except Exception as e:
        # 如果在循环中出现错误且已切换到模拟模式，直接返回模拟数据
        if _use_mock_data and MOCK_AVAILABLE:
            simulate_time_progress()
            return generate_mock_data(uid_list)
        else:
            print(f"❌ 批量数据获取失败: {e}")
            if not MOCK_AVAILABLE:
                print("❌ 模拟数据生成器不可用")
            raise e

def get_time():
    """
    函数功能:获取当前时间
    """
    current_time = datetime.now()
    current_time = current_time.strftime("%Y-%m-%dT%H:%M")
    current_time = current_time + ":00Z"
    return current_time

# output
def new_uid(output_uid):
    net_uid = []
    for uid in output_uid:
        net_uid.append(uid[1])
    return net_uid


