from datetime import datetime
import pandas as pd
from influxdb import InfluxDBClient


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
    """
    selected = database.query(f'SELECT * FROM "{uid}" ORDER BY time DESC LIMIT {sequence_length}')
    df = pd.DataFrame()
    for select in selected:
        df = pd.DataFrame.from_records(select)
    value = df["value"]
    return value

def Get_last_len_data(uid_list, database, sequence_length=1):
    """
    函数功能:获取所有最新 sequence_length 数据
    """
    df = pd.DataFrame()
    for uid in uid_list:
        node_name = uid
        temp = Read_last_len_data(uid=uid, database=database, sequence_length=sequence_length)
        new_data = pd.DataFrame({node_name: temp})
        df = pd.concat([df, new_data], axis=1)
    return df

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


