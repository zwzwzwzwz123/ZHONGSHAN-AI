#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
佛山AI项目依赖验证脚本
用于验证项目所需的所有依赖是否已正确安装
"""

import sys
import importlib
import pkg_resources
from typing import Dict, List, Tuple, Optional

# 定义颜色代码
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"

def print_color(text: str, color: str) -> None:
    """打印彩色文本"""
    print(f"{color}{text}{RESET}")

def print_header(text: str) -> None:
    """打印标题"""
    print("\n" + "=" * 80)
    print_color(f"{BOLD}{text}{RESET}", GREEN)
    print("=" * 80)

def check_python_version() -> bool:
    """检查Python版本是否满足要求"""
    required_version = (3, 12)
    current_version = sys.version_info
    
    if current_version >= required_version:
        print_color(f"✓ Python版本: {sys.version.split()[0]} (满足要求 >= {required_version[0]}.{required_version[1]})", GREEN)
        return True
    else:
        print_color(f"✗ Python版本: {sys.version.split()[0]} (不满足要求 >= {required_version[0]}.{required_version[1]})", RED)
        return False

def check_package(package_name: str) -> Tuple[bool, Optional[str]]:
    """检查包是否已安装及其版本"""
    # 包名映射，处理导入名和包名不一致的情况
    package_mapping = {
        "PyYAML": "yaml",
        "scikit-learn": "sklearn",
        "stable-baselines3": "stable_baselines3",
        "paho-mqtt": "paho.mqtt",
        "bayesian-optimization": "bayes_opt",
    }

    clean_name = package_name.split(">=")[0].split("<")[0].strip()
    import_name = package_mapping.get(clean_name, clean_name)

    try:
        # 尝试导入包
        module = importlib.import_module(import_name)

        # 获取包版本
        try:
            version = pkg_resources.get_distribution(clean_name).version
            return True, version
        except pkg_resources.DistributionNotFound:
            # 某些包可能无法通过pkg_resources获取版本
            try:
                version = getattr(module, "__version__", "未知")
                return True, version
            except AttributeError:
                return True, "未知"
    except ImportError:
        return False, None

def check_dependencies(dependencies: Dict[str, List[str]]) -> Dict[str, List[Tuple[str, bool, Optional[str]]]]:
    """检查所有依赖"""
    results = {}
    
    for category, packages in dependencies.items():
        results[category] = []
        for package in packages:
            # 提取包名（去除版本信息）
            package_name = package.split("#")[0].strip()
            if not package_name or package_name.startswith("# "):
                continue
                
            package_name = package_name.replace('"', '').replace(',', '')
            if ">=" in package_name or "<" in package_name:
                clean_name = package_name.split(">=")[0].split("<")[0].strip()
            else:
                clean_name = package_name
                
            installed, version = check_package(clean_name)
            results[category].append((package_name, installed, version))
    
    return results

def main():
    """主函数"""
    print_header("佛山AI项目依赖验证")
    
    # 检查Python版本
    python_ok = check_python_version()
    
    # 定义依赖分类
    dependencies = {
        "核心数据处理库": [
            "numpy>=1.24.0,<2.4.0",
            "pandas>=2.0.0,<2.4.0",
        ],
        "数据库连接": [
            "influxdb>=5.3.2,<6.0.0",
        ],
        "配置文件处理": [
            "PyYAML>=6.0,<7.0",
        ],
        "机器学习框架": [
            "torch>=2.0.0,<2.5.0",
            "torchvision>=0.15.0,<0.20.0",
            "scikit-learn>=1.3.0,<1.6.0",
            "joblib>=1.3.0,<1.5.0",
        ],
        "深度强化学习": [
            "stable-baselines3>=2.0.0,<3.0.0",
            "gymnasium>=0.28.0,<1.0.0",
        ],
        "优化算法": [
            "scipy>=1.10.0,<1.15.0",
            "bayesian-optimization>=1.4.0,<2.0.0",
        ],
        "通信协议": [
            "paho-mqtt>=1.6.0,<2.0.0",
            "requests>=2.28.0,<3.0.0",
        ],
        "进度显示": [
            "tqdm>=4.64.0,<5.0.0",
        ],
    }
    
    # 检查依赖
    results = check_dependencies(dependencies)
    
    # 打印结果
    all_ok = True
    for category, packages in results.items():
        print_header(category)
        for package_name, installed, version in packages:
            if installed:
                print_color(f"✓ {package_name} (已安装，版本: {version})", GREEN)
            else:
                print_color(f"✗ {package_name} (未安装)", RED)
                all_ok = False
    
    # 总结
    print_header("验证结果")
    if all_ok and python_ok:
        print_color("✓ 所有依赖已正确安装，项目可以运行", GREEN)
    else:
        print_color("✗ 部分依赖未安装或版本不符，请安装缺失的依赖", RED)
        print("\n安装命令:")
        print_color("pip install -e .", YELLOW)
        print("或")
        print_color("pip install -r requirements.txt", YELLOW)

if __name__ == "__main__":
    main()
