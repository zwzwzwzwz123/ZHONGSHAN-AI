from Optimization_Algorithm import simulated_annealing_algorithm, genetic_algorithm, gradient_descent_algorithm, bayesian_optimization, DQN_Optimization, BDQ_Optimization, DDPG_Optimization, PPO_Optimization, SAC_Optimization

class OptimizationAlgorithm_Manager:
    '''
    The Manager of OA (OptimizationAlgorithm)
    '''
    def __init__(self):
        self.optimization_dict = {'SA':simulated_annealing_algorithm,
                           'GA':genetic_algorithm,
                           'BO':bayesian_optimization,
                           'GDA':gradient_descent_algorithm,
                           'DQN':DQN_Optimization, 
                           'BDQ':BDQ_Optimization, 
                           'DDPG':DDPG_Optimization, 
                           'PPO':PPO_Optimization, 
                           'SAC':SAC_Optimization
                           }
        self.exist_optimization_repositoy = {}
        
        
    ### 获得模型    
    def _get_optimization(self, oa_name):
        return self.exist_optimization_repositoy[oa_name]
    ### 存入模型
    def _set_optimization(self, oa, oa_name):
        self.exist_optimization_repositoy[oa_name] = oa
        
    ### 创建初始模型
    def create_init_optimization(self, oa_name, args, obj_func):
        oa = self.optimization_dict[oa_name](args, obj_func)
        self._set_optimization(oa,oa_name)
        return oa
    
    ### 运行优化得到结果
    def run_optimization(self, oa_name, obj_func):
        result = self.exist_optimization_repositoy[oa_name].run(obj_func)
        return result
    
    ### 检查边界条件
    def check_bounds(self, oa_name, variables, bounds_dict):
        oa = self._get_optimization(oa_name)
        result = oa.check(variables, bounds_dict)
        return result