class Optimization_Algorithm_Basic:
    def __init__(self):
        pass
    
    def pretrained_and_save(self):
        pass
    
    def run_and_save(self):
        raise NotImplementedError
    
    def check(self, variables, bounds_dict):
        '''
        check OA
        '''
        result = []
        for var in variables:
            if var in bounds_dict:
                min_val, max_val = bounds_dict[var]
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