def remove_list1_elements_exist_in_list2(list1:list[str], list2:list[str])->list[str]:
    '''
    在list1中删掉所有list2中出现的元素
    '''
    # 创建一个映射，将 list2 的元素及其索引存储在字典中
    removal_indices = set(list2)
    
    # 创建新列表来存储结果
    new_list1 = []
    
    # 遍历 list1 和 list2，同时过滤掉 list3 中的元素
    for i in range(len(list1)):
        if list1[i] not in removal_indices:
            new_list1.append(list1[i])
    
    return new_list1

def sort_list1_by_list2(list1:list[str], list2:list[float])->list[str]:
    '''
    将list1(str)根据list2(float)升序排序
    '''
    # 1. 创建一个包含(list1, list2)元素的元组列表
    combined = list(zip(list1, list2))
    
    # 2. 根据list2中的值进行排序 
    sorted_combined = sorted(combined, key=lambda x: x[1])
    
    # 3. 解开排序后的元组列表，提取排序后的list1和list2元素
    sorted_list1 = [item[0] for item in sorted_combined]
    return sorted_list1

def find_indices_of_list1_in_list2(list1:list, list2:list)->list:
    """
    返回 list1 中每个元素在 list2 中的下标。
    假设 list1 是 list2 的子集。

    :param list1: 子集列表
    :param list2: 超集列表
    :return: 包含下标的列表
    :raises ValueError: 如果 list1 中的元素不在 list2 中
    """
    indices = []
    
    for item in list1:
        try:
            index = list2.index(item)
            indices.append(index)
        except ValueError:
            raise ValueError(f"元素 {item} 不在 list2 中")
    
    return indices

def find_different_indices(list1:list,list2:list)->list:
    '''
    返回两个list中元素不同的indices
    用于获得本次改变的优化变量
    '''
    different_indices = [index for index in range(len(list1)) if list1[index]!=list2[index]]
    return different_indices