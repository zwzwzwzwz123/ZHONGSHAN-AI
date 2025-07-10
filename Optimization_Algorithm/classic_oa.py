from scipy.optimize import minimize
from scipy.optimize import differential_evolution
from scipy.optimize import basinhopping
from bayes_opt import BayesianOptimization, UtilityFunction
from .OA_basic import Optimization_Algorithm_Basic


'''
result.x:(result['x'])得到最优解的参数对应值
result.fun:(result['fun'])得到的最优值
result.nfev:(result['nfev'])迭代次数
'''

class simulated_annealing_algorithm(Optimization_Algorithm_Basic):
    '''
    The SA OA for optimization
    '''
    def __init__(self, args, objective_func):
        self.args = args
        self.initial_guess = args['initial_guess']
        self.objective_func = objective_func

    def run(self, obj_func):
        initial_guess = self.initial_guess
        result = basinhopping(obj_func, initial_guess)
        # 示例：params_result= [14.  17.1 23.  29.   6.  29. ]
        return result

class genetic_algorithm(Optimization_Algorithm_Basic):
    '''
    The GA OA for optimization
    '''
    def __init__(self, args, objective_func):
        self.args = args
        self.bounds = args['bounds']
        self.objective_func = objective_func

    def run(self, obj_func):
        bounds = self.bounds
        result = differential_evolution(obj_func, bounds)
        # 示例：params_result= [14.18696019 17.16368795 27.38474262 30.07998673  5.78528344 28.53857264]
        # result.nfev:(result['nfev'])迭代次数5137
        return result

class gradient_descent_algorithm(Optimization_Algorithm_Basic):
    def __init__(self, args, objective_func):
        self.args = args
        self.initial_guess = args['initial_guess']
        self.objective_func = objective_func

    def run(self, obj_func):
        initial_guess = self.initial_guess
        result = minimize(obj_func, initial_guess)
        # 示例：params_result= [14.  17.1 23.  29.   6.  29. ]
        return result
    
class bayesian_optimization(Optimization_Algorithm_Basic):
    def __init__(self, args, objective_func):
        self.args = args
        self.bounds = args['bounds']
        self.objective_func = objective_func
        
    def run(self, obj_func):
        bounds = self.bounds
        optimizer= BayesianOptimization(
        f=obj_func,          # 目标函数
        pbounds=bounds,       # 取值空间
        verbose=0,            # verbose = 2 时打印全部，verbose = 1 时打印运行中发现的最大值，verbose = 0 将什么都不打印
        )

        optimizer.maximize(init_points=self.args['init_points'], n_iter=self.args['iter_times'],acq='ei')
        result = {}
        result['x'] = optimizer.max['params']
        result['fun'] = -optimizer.max["target"]
        return result