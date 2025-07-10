from utils.interfaces import SafetyBoundary,OptimizationInput,OptimizationOutput,ObservingBoundary
from typing import Literal
from utils.help_functions import remove_list1_elements_exist_in_list2,sort_list1_by_list2,find_indices_of_list1_in_list2
class Custom_Optimization_V2():
    optimization_process_status_type = ['TurningOff_AC',
                                        'Observing_Temperature_TurningOff_AC',
                                        'Re_TurningOn_AC',
                                        'Adjusting_AC_Temperature',
                                        'Observing_Temperature_Adjusting_AC',
                                        'Re_Adjusting_AC_Temperature',
                                        'Exceeding_SafetyBoundary']
    def __init__(self,temp_safety_boundary:SafetyBoundary,
                 ac_uid:list[str],
                 #注释掉压缩机部分
                 #converter_uid:list[str],
                 default_ac_onoff_status_setting:list[int],
                 default_ac_temp_setting:list[float],
                 #注释掉压缩机部分
                 #default_converter_freq_setting:list[int],
                 temp_raising_sharply_threshold:float,
                 temp_cross_peak_threshold: float,
                 observing_times_threshold_turningoff_ac: int,
                 observing_times_threshold_control_temp_ac: int,
                 #注释掉压缩机部分
                 #observing_times_threshold_control_freq: int,
                 temp_observing_boundary:ObservingBoundary,
                 temp_control_highest_boundary:list[float],
                 freq_control_lowest_boundary:list[int],
                 last_optimization_input:OptimizationInput,
                 turningoff_ac_priority:list[str],
                 temp_control_ac_priority:list[str],
                 min_temp_control_step:float = 1.0, 
                 #注释掉压缩机部分
                 #min_freq_control_step:int = 1,
                 max_num_ac_can_turnoff:int = 2,
                 init_type:str = 'last') -> None:
        
        # 初始化设置超参数（全程不变）
        self.init_type = init_type # 初始化类型
        self.temp_safety_boundary = temp_safety_boundary # 温度安全边界
        self.max_num_ac_can_turnoff  = max_num_ac_can_turnoff # 最大可关闭空调数量
        self.temp_observing_boundary = temp_observing_boundary # 观察温度边界
        self.temp_control_highest_boundary = temp_control_highest_boundary # 调整过程温度设置边界
        # self.freq_control_lowest_boundary = freq_control_lowest_boundary # 调整过程频率设置边界
        self.temp_raising_sharply_threshold = temp_raising_sharply_threshold # 判断 急剧增长的阈值
        self.temp_cross_peak_threshold = temp_cross_peak_threshold # 观察达峰阈值
        self.observing_times_threshold_turningoff_ac = observing_times_threshold_turningoff_ac # 观察时间窗口长度阈值（关闭空调）
        self.observing_times_threshold_control_temp_ac = observing_times_threshold_control_temp_ac # 观察时间窗口长度阈值（调整空调温度）
        # self.observing_times_threshold_control_freq = observing_times_threshold_control_freq # 观察时间窗口长度阈值（调整变频器）
        self.default_ac_onoff_status_setting = default_ac_onoff_status_setting # 默认空调开启设定
        self.default_ac_temp_setting = default_ac_temp_setting # 默认空调温度设定
        # self.default_converter_freq_setting = default_converter_freq_setting
        self.min_temp_control_step = min_temp_control_step # 温度控制最小调整步长
        # self.min_freq_control_step = min_freq_control_step # 频率控制最小调整步长
        self.ac_uid = ac_uid # (全部)空调uid
        # self.converter_uid = converter_uid # 变频器uid


        # 初始化设置其他参数（过程中更新改变）
        self.init_start_flag = True # 初始开启flag 默认初始状态为True
        
        self.turningoff_ac_priority = turningoff_ac_priority # 关闭空调优先级
        self.temp_control_ac_priority = temp_control_ac_priority # 控制空调温度优先级
        #注释掉压缩机部分
        #self.freq_control_converter_priority = converter_uid # 控制变频器优先级

        self.last_control_ac_uid:str = self.temp_control_ac_priority[0] # 初始化设置上一次控制的ac uid
        #注释掉压缩机部分
        #self.last_control_converter_uid:str = self.freq_control_converter_priority[0] # 初始化设置上一次控制的变频器 uid

        self.last_optimization_input = last_optimization_input # 初始化设置 last_optimization_input

        self.observing_turningoff_ac_times:int = 0 # 初始化 尝试空调关闭的观察时间
        self.observing_temp_control_ac_times:int = 0 # 初始化 尝试温度调整的观察时间
        #注释掉压缩机部分
        #self.observing_freq_control_converter_times:int = 0 # 初始化 尝试变频器调整的观察时间

        self.optimization_output = OptimizationOutput()
        # 根据不同的初始类型设定初始值
        if self.init_type == 'default':
            self.optimization_output['ac_onoff_status_setting'] = default_ac_onoff_status_setting
            self.optimization_output['ac_temp_setting'] = default_ac_temp_setting
            #注释掉压缩机部分
            #self.optimization_output['converter_freq_setting'] = default_converter_freq_setting
            self.optimization_process_status = 'TurningOff_AC' # 优化过程状态
        elif self.init_type == 'last':
            self.optimization_output['ac_onoff_status_setting'] = self.last_optimization_input['ac_onoff_status_setting']
            self.optimization_output['ac_temp_setting'] = self.last_optimization_input['ac_temperature_setting']
            #注释掉压缩机部分
            #self.optimization_output['converter_freq_setting'] = self.last_optimization_input['converter_freq_setting']
            self.optimization_process_status = 'Adjusting_AC_Temperature' # 优化过程状态
        
        print(f'optimization_output after init:{self.optimization_output}')

        # 初始化设置已经关闭的空调uid
        self.closed_ac_uid = [] # 已经关闭的空调uid

        # 根据设定的默认状态，更新调整已经关闭的空调uid
        closed_ac_uid_index_list = [i for i,u in enumerate(self.optimization_output['ac_onoff_status_setting']) if u == 0]
        for uid_index in closed_ac_uid_index_list:
            self.closed_ac_uid.append(self.ac_uid[uid_index])

        self.temp_control_finished_ac_uid = [] # 已经完成了温度调整的空调uid
        #注释掉压缩机部分
        #self.freq_control_finished_converter_uid = [] # 已经完成了频率调整的变频器uid

        
        # 若初始优化过程状态的关闭空调数量就已经超过了最大可关闭空调数量
        if len(self.closed_ac_uid)>=self.max_num_ac_can_turnoff:
            # 则直接进入调整温度环节
            self.optimization_process_status = 'Adjusting_AC_Temperature'
    
    def _set_optimization_process_status(self,optimization_process_status:Literal['TurningOff_AC',
                                                                                  'Observing_Temperature_TurningOff_AC',
                                                                                  'Adjusting_AC_Temperature',
                                                                                  'Observing_Temperature_Adjusting_AC',
                                                                                  #注释掉压缩机部分
                                                                                  #'Adjusting_Freq',
                                                                                  #'Observing_Temperature_Adjusting_Freq',
                                                                                  'Sleeping']):
        '''
        设置优化过程状态
        '''
        self.optimization_process_status = optimization_process_status
    
    def _get_optimization_process_status(self):
        '''
        获取优化过程状态
        '''
        return self.optimization_process_status

    def isintemp_safety_boundary(self,optimization_input:OptimizationInput)->OptimizationOutput:
        '''
        判断当前房间状态是否在安全边界内,若在安全边界内,返回True;反之,返回False
        '''
        # if len(optimization_input) != len(self.temp_safety_boundary['lower_boundary']):
        #     raise ValueError("optimization_input length is not same with temp_safety_boundary")
        
        room_temp_safe = all(lower_boundary < optim_input < higher_boundary for lower_boundary, optim_input, higher_boundary in 
                   zip(self.temp_safety_boundary['room_temp_lower_boundary'],optimization_input['room_temperature'],self.temp_safety_boundary['room_temp_higher_boundary']))
        # ac_temp_safe = all(lower_boundary < optim_input < higher_boundary for lower_boundary, optim_input, higher_boundary in 
        #            zip(self.temp_safety_boundary['ac_temp_lower_boundary'],optimization_input['ac_temperature'],self.temp_safety_boundary['ac_temp_higher_boundary']))

        opening_ac_uid = remove_list1_elements_exist_in_list2(self.ac_uid,self.closed_ac_uid)
        opening_ac_index = find_indices_of_list1_in_list2(opening_ac_uid,self.ac_uid)

        ac_temp_safe = all(self.temp_safety_boundary['ac_temp_lower_boundary'][index]
                           < optimization_input['ac_temperature'][index]
                           < self.temp_safety_boundary['ac_temp_higher_boundary'][index]
                           for index in opening_ac_index)
        
        safe_result = room_temp_safe and ac_temp_safe
        return safe_result
    
    def get_ac_control_priority(self,optimization_input:OptimizationInput)->tuple[list[str],list[str]]:
        '''
        根据输入 返回 ac关闭优先级和ac温度控制优先级的uid
        返回ac控制优先级的列表list,index从0开始递增,关闭优先级递增
        '''

        list1 = self.ac_uid # 全部空调的uid
        list2 = [real_temp - setting_temp for real_temp, setting_temp  # optimization_input的回风温度和回风温度设定值的差值
                 in zip(optimization_input['ac_temperature'],optimization_input['ac_temperature_setting'])]
        
        sorted_ac_uid = sort_list1_by_list2(list1,list2) # 根据差值升序后的全部空调的uid

        opening_sorted_ac_uid = remove_list1_elements_exist_in_list2(sorted_ac_uid,self.closed_ac_uid) # 仍打开的空调的uid
        temp_controllable_opening_sorted_ac_uid = remove_list1_elements_exist_in_list2(opening_sorted_ac_uid,self.temp_control_finished_ac_uid) # 仍打开且仍能调整温度的uid
        return opening_sorted_ac_uid,temp_controllable_opening_sorted_ac_uid
    
    def check_last_optimization_output_is_adopted(self,optimization_input:OptimizationInput):
        '''
        根据算法输出是否被采纳
        判断算法状态与真实状态是否发生变化，若不同，则更新算法状态
        '''
        # 若 实际（本次优化输入中显示的优化参数） 与 预期（上一轮的优化输出） 相同
        if ((optimization_input['ac_onoff_status_setting'] == self.optimization_output['ac_onoff_status_setting']) and
        (optimization_input['ac_temperature_setting'] == self.optimization_output['ac_temp_setting'])):

            # 若上一次是睡眠状态
            # 现在重新唤醒
            if self._get_optimization_process_status() == "sleeping":
                self._set_optimization_process_status(self.process_status_before_sleeping)

            # 若上一次就是正常运行
            # 则不做任何其他操作
            else:
                pass

        # 若 实际（本次优化输入中显示的优化参数） 与 预期（上一轮的优化输出） 不同
        else:
            # 若 本次优化输入 与 上次优化输入 相同 （可能未更新）
            if ((optimization_input['ac_onoff_status_setting'] == self.last_optimization_input['ac_onoff_status_setting']) and
                (optimization_input['ac_temperature_setting'] == self.last_optimization_input['ac_temperature_setting'])):

                # 备份sleep前状态
                self.process_status_before_sleeping = self._get_optimization_process_status()
                # 程序暂停运行，进行休眠
                self._set_optimization_process_status('Sleeping')


            # 若 本次优化输入 与 上次优化输入 不同
            else:
                # 说明外部产生了手动操作
                # 则根据外部输入重置整个优化过程
                # 按照输入条件更新优化输出
                self.optimization_output['ac_onoff_status_setting'] = optimization_input['ac_onoff_status_setting']
                self.optimization_output['ac_temp_setting'] = optimization_input['ac_temperature_setting']

                # reset过后，进入温度调整状态
                self._set_optimization_process_status('Adjusting_AC_Temperature')

                # 获取关闭空调的uid， 更新关闭空调的uid list
                self.closed_ac_uid = []
                closed_ac_uid_index_list = [i for i,u in enumerate(optimization_input['ac_onoff_status_setting']) if u == 0]
                for uid_index in closed_ac_uid_index_list:
                    self.closed_ac_uid.append(self.ac_uid[uid_index])
                
                # 更新（重置）完成了温度调整的空调uid
                self.temp_control_finished_ac_uid = []

                # 重置时间调整观察时间计数器
                self.observing_temp_control_ac_times = 0

    def handle_optimization_process(self,optimization_input:OptimizationInput)->OptimizationOutput:
        '''
        主要逻辑函数
        根据不同的优化过程状态optimization_process_status,调用不同的对应的优化函数。
        首先判断输入是否超过安全边界，若超过安全边界，则直接设置状态为超过安全边界
        '''
        # 判断当前房间状态是否在安全边界内
        if not self.isintemp_safety_boundary(optimization_input):
            print('当前房间不在安全边界中')
            # 若在超出安全边界，直接处理超出安全边界的情况
            self._Handling_Exceed_SafetyBoundary()
        
        # 检查上一次的优化输出是否被采用
        # 若是第一次进行，则不检查
        if self.init_start_flag:
            self.init_start_flag = False
        # 反之，后续进行优化输出检查
        else:
            self.check_last_optimization_output_is_adopted(optimization_input=optimization_input)
        

        # 更新优先级
        self.turningoff_ac_priority, self.temp_control_ac_priority = self.get_ac_control_priority(optimization_input)

        print(f'close_uid:{self.closed_ac_uid}')
        if self.optimization_process_status == 'TurningOff_AC':
            print('TurningOff_AC')
            self._TurningOff_AC()

        elif self.optimization_process_status == 'Observing_Temperature_TurningOff_AC':
            print('Observing_Temperature_TurningOff_AC')
            self._Observing_Temperature_TurningOff_AC(optimization_input)

        elif self.optimization_process_status == 'Adjusting_AC_Temperature':
            print('Adjusting_AC_Temperature')
            self._Adjusting_AC_Temperature()

        elif self.optimization_process_status == 'Observing_Temperature_Adjusting_AC':
            print('Observing_Temperature_Adjusting_AC')
            self._Observing_Temperature_Adjusting_AC(optimization_input)
        
        # 注释掉压缩机部分
        # elif self.optimization_process_status == 'Adjusting_Freq':
        #     print('Adjusting_Freq')
        #     self._Adjusting_Freq()
        # 
        # elif self.optimization_process_status == 'Observing_Temperature_Adjusting_Freq':
        #     print('Observing_Temperature_Adjusting_Freq')
        #     self._Observing_Temperature_Adjusting_Freq(optimization_input)
        
        elif self.optimization_process_status == 'Sleeping':
            print('Sleeping')
            self._Sleeping()
        
        #2025.7.7 更新下一行
        self.last_optimization_input = optimization_input  # 更新last_optimization_input
        return self.optimization_output
    
    def _TurningOff_AC(self):
        '''
        尝试关闭空调
        运行逻辑:
        1. 获取优先级，按照优先级,依次关闭空调。
        2. 每关闭一次空调，进入温度观察状态。
        3. 温度观察状态通过后，返回尝试关闭空调状态，继续关闭空调。
        4. 若关闭空调次数达到可关闭空调数量上限，进入温度调整状态。
        '''
        # 1. 获取优先级，按照优先级, 依次关闭空调。
        # 获取优先级
        turningoff_ac_priority:list[str] = self.turningoff_ac_priority

        # 将优先级最高的空调加入关闭list
        attempting_closed_ac_uid = turningoff_ac_priority[0]
        self.closed_ac_uid.append(attempting_closed_ac_uid)

        # 修改优化输出
        selected_index = self.ac_uid.index(attempting_closed_ac_uid) # 根据uid选中需要关闭空调的index
        #ac_onoff_status_setting需要改为mode！！！
        self.optimization_output['ac_onoff_status_setting'][selected_index] = 0 # ac_onoff_status_setting设置为0，关闭空调

        # 2. 每关闭一次空调，进入温度观察状态。
        self._set_optimization_process_status('Observing_Temperature_TurningOff_AC')

    def _Observing_Temperature_TurningOff_AC(self,optimization_input:OptimizationInput):
        '''
        观察关闭空调后温度变化
        1. 若温度急剧上升，则观察失败，进入重新打开空调模式
        2. 若温度上升超过温度观察上限， 则观察失败， 重新进入打开空调模式
        3. 若温度实现过峰，则观察成功，进入下一次尝试关闭空调模式
        4. 若观察次数达到阈值，则观察成功， 进入下一次尝试关闭空调模式
        5. 其他情况，则继续进入观察模式
        '''

        def observing_times_is_arrives_threshold()->bool:
            '''
            观察次数是否达到观察上限
            '''
            return self.observing_turningoff_ac_times >= self.observing_times_threshold_turningoff_ac
        
        def observing_turningoff_success()->bool:
            '''
            观察关闭空调尝试成功
            条件：温度达峰 或 观察时间超过观察时间阈值
            '''
            return (self.temp_is_cross_peak(optimization_input) or (observing_times_is_arrives_threshold()))
        
        def observing_turningoff_fail()->bool:
            '''
            观察关闭空调尝试失败
            条件：温度急剧上升 或 温度上升超过温度观察上限
            '''
            return ((not self.temp_is_in_observing_boundary(optimization_input)) or self.temp_is_raising_sharply(optimization_input))
        
        def reach_turningoff_ac_max_num()->bool:
            '''
            达到可关闭空调数量上限
            '''
            return (len(self.closed_ac_uid) >= self.max_num_ac_can_turnoff)
        
        self.observing_turningoff_ac_times += 1 # 记录空调关闭观察时间 +1

        # （1）若观察到关闭失败
        if observing_turningoff_fail():

            # 1. 重新调整回之前的开关状态
            self._Re_TurningOn_AC()

            # 2. 进入 尝试调整温度模式 
            self._set_optimization_process_status('Adjusting_AC_Temperature')
            
            # 3. 重置观察时间
            self.observing_turningoff_ac_times = 0
            return

        # （2）若观察到关闭成功
        elif observing_turningoff_success():

            # 1. 若未达到可关闭空调数量上限，则继续进入 尝试关闭空调模式
            if not reach_turningoff_ac_max_num():
                self._set_optimization_process_status('TurningOff_AC')
                self.observing_turningoff_ac_times = 0 # 重置空调关闭观察时间计时器
                return

            # 2. 若达到可关闭空调数量上限，则进入 尝试调整温度模式
            elif reach_turningoff_ac_max_num():
                self._set_optimization_process_status('Adjusting_AC_Temperature')
                self.observing_turningoff_ac_times = 0 # 重置空调关闭观察时间计时器
                return

        # （3）若未观察到成功或失败事件， 则继续观察模式
        else:
            self._set_optimization_process_status('Observing_Temperature_TurningOff_AC')
            return

    def _Re_TurningOn_AC(self):
        '''
        重新打开上一次关闭的一个ac
        '''
        reopened_ac_uid = self.closed_ac_uid.pop() # 获取最后一次关闭的ac_uid，并将其从关闭列表中删除
        selected_index = self.ac_uid.index(reopened_ac_uid) # 根据uid选中需要重新打开空调的index
        #ac_onoff_status_setting需要改为mode！！！
        self.optimization_output['ac_onoff_status_setting'][selected_index] = 1 # ac_onoff_status_setting设置为1，打开空调


    def _Adjusting_AC_Temperature(self):
        '''
        调整空调设定温度
        1. 从开着的空调中，获得空调温度调整的优先级（与关闭顺序相同倒序）
        2. 按优先级顺序轮询，取优先级最高的空调，将其设定温度升高 最小步长 度。
        3. 每调整一次空调温度，进入温度调整观察模式。
        '''
        # 1. 获得温度控制优先级
        temp_control_ac_priority:list[str] = self.temp_control_ac_priority

        # 2. 根据优先级，获得优先级最高的空调，将其设定温度升高1度
        #以下为2025.7.7新添加
        if not temp_control_ac_priority:
            self._set_optimization_process_status('Sleeping')
            return
        #以上为2025.7.7新添加
        temp_control_ac_uid = temp_control_ac_priority[0]
        selected_index = self.ac_uid.index(temp_control_ac_uid)
        self.last_control_ac_uid = temp_control_ac_uid # !!!特别注意!!! 在修改温度前，需要记录上一次控制的ac uid
        self.optimization_output['ac_temp_setting'][selected_index] += self.min_temp_control_step # ac_temp_setting设置为增加min_temp_control_step

        # 3. 每调整一次空调温度，进入温度调整观察模式。
        self._set_optimization_process_status('Observing_Temperature_Adjusting_AC')

    
    def _Observing_Temperature_Adjusting_AC(self,optimization_input:OptimizationInput):
        '''
        观察空调温度调整后温度变化
        1. 若温度急剧上升 或 温度超过温度观察边界，则观察失败
        2. 若温度过峰 或 温度观察次数达到阈值，则观察成功
        3. 若观察失败 且 仍存在可调空调，则调整上一步空调回温，并重新进入 尝试调整温度 （调整剩余空调）
        4. 若观察失败 且 不存在可调， 则则调整上一步空调回温
        '''
        def observing_times_is_arrives_threshold()->bool:
            '''
            观察次数是否达到观察上限
            '''
            return self.observing_temp_control_ac_times >= self.observing_times_threshold_control_temp_ac
        
        def observing_temp_adjusting_success()->bool:
            '''
            观察关闭空调尝试成功
            条件：温度达峰 或 观察时间超过观察时间阈值
            '''
            return (self.temp_is_cross_peak(optimization_input) or (observing_times_is_arrives_threshold()))
        
        def observing_temp_adjusting_fail()->bool:
            '''
            观察关闭空调尝试失败
            条件：温度急剧上升 或 温度上升超过温度观察上限
            '''
            return ((not self.temp_is_in_observing_boundary(optimization_input)) or self.temp_is_raising_sharply(optimization_input))
        
        def exist_temp_controllable_ac()->bool:
            '''
            存在可以继续控制温度的空调
            判断条件：已经完成温度控制的空调uid数量 + 关闭的uid数量 < 总的uid数量
            '''
            return (len(self.temp_control_finished_ac_uid) + len(self.closed_ac_uid) < len(self.ac_uid)) 
        
        def reach_control_boundary()->bool:
            '''
            （上一次）调整的空调是否达到了设定的温度调整边界
            判断条件：输出设定温度 是否小于等于 设定的温度调整边界
            '''
            selected_index = self.ac_uid.index(self.last_control_ac_uid)
            return self.optimization_output['ac_temp_setting'][selected_index] >= self.temp_control_highest_boundary[selected_index]
        
        self.observing_temp_control_ac_times += 1 # 记录空调温度调整观察时间+1
        # （1） 若观察 调整失败
        if observing_temp_adjusting_fail():

            # 1. 更新可以控制空调的状况
            self.temp_control_finished_ac_uid.append(self.last_control_ac_uid) # 将调整失败的空调uid 记录为已完成调整的空调uid

            # 2. 重新调整空调设定温度
            self._Re_Adjusting_AC_Temperature()

            # 3.1 若仍存在可调整空调 设定下次重新进入 温度调整模式
            if exist_temp_controllable_ac():
                self._set_optimization_process_status('Adjusting_AC_Temperature')
                self.observing_temp_control_ac_times = 0 # 重置空调温度调整观察时间计时器
                return 
            
            # 3.2 若不存在可继续调整空调 完成优化过程，进入休眠模式
            else:
                self._set_optimization_process_status('Sleeping')
                return

        # （2） 若观察 调整成功
        elif observing_temp_adjusting_success():

            # 1. 更新可以控制空调的状况
            # 若调整后温度达到了调整温度边界，则完成此空调完成调整
            if reach_control_boundary():
                self.temp_control_finished_ac_uid.append(self.last_control_ac_uid)
            
            # 2.1 若仍存在可调整空调 设定下次重新进入 温度调整模式
            if exist_temp_controllable_ac():
                self._set_optimization_process_status('Adjusting_AC_Temperature')
                self.observing_temp_control_ac_times = 0 # 重置空调温度调整观察时间计时器
                return
            
            """
            注释掉压缩机部分
            # 2,2 若不存在可调整空调 完成优化过程，进入变频器调整模式
            else:
                self._set_optimization_process_status('Adjusting_Freq')
                self.observing_temp_control_ac_times = 0 # 重置空调温度调整观察时间计时器
                return
            """
        # 3 若观察没有 调整成功或调整失败事件，则继续观察
        else:
            self._set_optimization_process_status('Observing_Temperature_Adjusting_AC')
            return
    
    """
    注释掉压缩机部分
    def _Adjusting_Freq(self):
        '''
        调整变频器频率
        '''
        # 1. 获得频率控制优先级
        freq_control_converter_priority:list[str] = self.freq_control_converter_priority

        # 2. 根据优先级，获得优先级最高的变频器，将其设定频率降低1度
        freq_control_converter_uid = freq_control_converter_priority[0]
        selected_index = self.converter_uid.index(freq_control_converter_uid)
        self.last_control_converter_uid = freq_control_converter_uid # !!!特别注意!!! 在修改频率前，需要记录上一次控制的converter uid
        self.optimization_output['converter_freq_setting'][selected_index] -= self.min_freq_control_step # converter_freq_setting设置为减少min_freq_control_step

        # 3. 每调整一次变频器频率，进入频率调整观察模式。
        self._set_optimization_process_status('Observing_Temperature_Adjusting_Freq')
        
    def _Observing_Temperature_Adjusting_Freq(self,optimization_input:OptimizationInput):
        '''
        观察 变频器 调整后温度变化
        1. 若温度急剧上升 或 温度超过温度观察边界，则观察失败
        2. 若温度过峰 或 温度观察次数达到阈值，则观察成功
        3. 若观察失败 且 仍存在可调空调，则调整上一步变频器，并重新进入 尝试调整变频器 （调整剩余变频器）
        4. 若观察失败 且 不存在可调， 则调整上一步变频器
        '''
        
        def observing_times_is_arrives_threshold()->bool:
            '''
            观察次数是否达到观察上限
            '''
            return self.observing_freq_control_converter_times >= self.observing_times_threshold_control_freq
        
        def observing_freq_adjusting_success()->bool:
            '''
            观察频率调整尝试成功
            条件：温度达峰 或 观察时间超过观察时间阈值
            '''
            return (self.temp_is_cross_peak(optimization_input) or (observing_times_is_arrives_threshold()))
        
        def observing_freq_adjusting_fail()->bool:
            '''
            观察频率调整尝试失败
            条件：温度急剧上升 或 温度上升超过温度观察上限
            '''
            return ((not self.temp_is_in_observing_boundary(optimization_input)) or self.temp_is_raising_sharply(optimization_input))
        
        def exist_freq_controllable_converter()->bool:
            '''
            存在可以继续控制频率的变频器
            判断条件：已经完成频率控制的变频器数量 < 总的变频器数量
            '''
            return (len(self.freq_control_finished_converter_uid) < len(self.converter_uid )) 
        
        def reach_control_boundary()->bool:
            '''
            （上一次）调整的变频器的频率是否达到了设定的频率调整边界
            判断条件：输出设定温度 是否小于等于 设定的频率调整边界
            '''
            selected_index = self.converter_uid.index(self.last_control_converter_uid)
            return self.optimization_output['converter_freq_setting'][selected_index] <= self.freq_control_lowest_boundary[selected_index]
        
        self.observing_freq_control_converter_times += 1 # 记录变频器调整观察时间+1
        # （1） 若观察 调整失败
        if observing_freq_adjusting_fail():

            # 1. 更新可以控制变频器的状况
            self.freq_control_finished_converter_uid.append(self.last_control_converter_uid) # 将调整失败的变频器uid 记录为已完成调整的变频器uid

            # 2. 重新调整变频器设定频率
            self._Re_Adjusting_Freq()

            # 3.1 若仍存在可调整变频器 设定下次重新进入 频率调整模式
            if exist_freq_controllable_converter():
                self._set_optimization_process_status('Adjusting_Freq')
                self.observing_freq_control_converter_times = 0 # 重置变频器频率调整观察时间计时器
                return 
            
            # 3.2 若不存在可继续调整变频器 完成优化过程，进入休眠模式
            else:
                self._set_optimization_process_status('Sleeping')
                self.observing_freq_control_converter_times = 0 # 重置变频器频率调整观察时间计时器
                return

        # （2） 若观察 调整成功
        elif observing_freq_adjusting_success():

            # 1. 更新可以控制空调的状况
            # 若调整后频率达到了调整频率边界，则完成此空调完成调整
            if reach_control_boundary():
                self.freq_control_finished_converter_uid.append(self.last_control_converter_uid)
            
            # 2.1 若仍存在可调整变频器 设定下次重新进入 频率调整模式
            if exist_freq_controllable_converter():
                self._set_optimization_process_status('Adjusting_Freq')
                self.observing_freq_control_converter_times = 0 # 重置变频器频率调整观察时间计时器
                return
            
            # 2,2 若不存在可调整变频器 完成优化过程，进入休眠模式
            else:
                self._set_optimization_process_status('Sleeping')
                self.observing_freq_control_converter_times = 0 # 重置变频器频率调整观察时间计时器
                return

        # 3 若观察没有 调整成功或调整失败事件，则继续观察
        else:
            self._set_optimization_process_status('Observing_Temperature_Adjusting_Freq')
            return
    """
    
    
    def _Re_Adjusting_AC_Temperature(self):
        '''
        重新调整空调设定温度
        根据上次温度调整的空调uid，调回空调设定温度，空调设定温度降低 最小步长 度
        '''
        selected_index = self.ac_uid.index(self.last_control_ac_uid) # 根据uid选中需要重新调整空调设定温度的index
        self.optimization_output['ac_temp_setting'][selected_index] -= self.min_temp_control_step # ac_temp_setting设置为 减少 min_temp_control_step

    """
    注释掉压缩机部分
    def _Re_Adjusting_Freq(self):
        '''
        重新调整变频器设定频率
        根据上次温度调整的空调uid，调回空调设定温度，空调设定温度降低 最小步长 度
        '''
        selected_index = self.converter_uid.index(self.last_control_converter_uid) # 根据uid选中需要重新调整变频器设定频率的index
        self.optimization_output['converter_freq_setting'][selected_index] += self.min_freq_control_step # converter_freq_setting设置为 增加 min_freq_control_step
    """
    
    def _Sleeping(self):
        '''
        程序执行完毕，优化进入sleeping模型
        '''
        return

    def _Handling_Exceed_SafetyBoundary(self):
        '''
        处理超出安全边界的情况
        '''
        self.optimization_output['ac_onoff_status_setting'] = [1 for _ in self.optimization_output['ac_onoff_status_setting']] # 打开所有空调
        self.optimization_output['ac_temp_setting'] = self.default_ac_temp_setting # 设定空调温度为默认空调设置温度
        #注释掉压缩机部分
        #self.optimization_output['converter_freq_setting'] = self.default_converter_freq_setting # 设定变频器频率为默认变频器设置频率
        return
    
    def temp_is_raising_sharply(self,optimization_input)->bool:
        '''
        判断温度是否急剧变化
        '''
        last_optimization_input:OptimizationInput = self.last_optimization_input
        raising_sharply_threshold:float = self.temp_raising_sharply_threshold

        room_max_diff_temp = max(abs(last_temp-new_temp) for last_temp,new_temp in 
                            zip(last_optimization_input['room_temperature'],optimization_input['room_temperature']))
        ac_max_diff_temp = max(abs(last_temp-new_temp) for last_temp,new_temp in 
                            zip(last_optimization_input['ac_temperature'],optimization_input['ac_temperature']))
        return (room_max_diff_temp > raising_sharply_threshold) or (ac_max_diff_temp > raising_sharply_threshold)
        
    def temp_is_in_observing_boundary(self,optimization_input)->bool:
        '''
        观察温度是否在温度观察边界内
        '''
        observing_boundary:ObservingBoundary = self.temp_observing_boundary
        room_temp_safe = all(optim_input < higher_boundary for optim_input, higher_boundary in 
                zip(optimization_input['room_temperature'],observing_boundary['room_temp_boundary']))
        ac_temp_safe = all(optim_input < higher_boundary for optim_input, higher_boundary in 
                zip(optimization_input['ac_temperature'],observing_boundary['ac_temp_boundary']))
        
        return room_temp_safe and ac_temp_safe
    
    def temp_is_cross_peak(self,optimization_input)->bool:
        '''
        观察温度是否过峰
        '''
        last_optimization_input:OptimizationInput = self.last_optimization_input
        cross_peak_threshold:float = self.temp_cross_peak_threshold
        room_temp_cross_peak = all(new_temp + cross_peak_threshold < last_temp for last_temp, new_temp in 
                                    zip(last_optimization_input['room_temperature'],optimization_input['room_temperature']))
        ac_temp_cross_peak = all(new_temp + cross_peak_threshold < last_temp for last_temp, new_temp in 
                                    zip(last_optimization_input['ac_temperature'],optimization_input['ac_temperature']))
        return room_temp_cross_peak and ac_temp_cross_peak
    