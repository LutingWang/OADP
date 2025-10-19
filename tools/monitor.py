import requests
import logging
import time
from enum import Enum
import subprocess
import logging
import re
import os
import glob
import argparse

class State(Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"

class BaseMonitor:
    def __init__(self, url, checkers=None, check_interval=10):
        self.check_interval = check_interval
        self.url = url
        self.state = State.ACTIVE
        
        # 使用指定的检查器，如果没有提供则收集所有以check开头的方法
        if checkers:
            self.check_funcs = []
            for checker in checkers:
                checker_name = f"check_{checker}" if not checker.startswith("check_") else checker
                if hasattr(self, checker_name) and callable(getattr(self, checker_name)):
                    self.check_funcs.append(getattr(self, checker_name))
                else:
                    logging.warning(f"指定的检查器 '{checker}' 不存在或不可调用")
        else:
            # 保持原有的自动收集行为作为后备选项
            self.check_funcs = [getattr(self, name) for name in dir(self)
                        if callable(getattr(self, name)) and name.startswith("check")]

    def send_text(self, text, info: bool = True):
        data = {
            "msg_type": "post",
            "content": {
                "post": {
                    "zh_cn": {
                        "title": "项目更新通知",
                        "content": [
                            [{
                                "tag": "text",
                                "text": text
                            }, {
                                "tag": "at",
                                "user_id": "7459274597810208772"
                            }]
                        ]
                    }
                }
            }
        }
        try:
            requests.post(self.url, json=data)
            logging.info("已通过WatchDog发送消息。")
        except Exception as e:
            logging.error(f"WatchDog消息发送失败: {e}")

    
    def run(self):
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        logging.info("启动监控程序...")
        
        if not self.check_funcs:
            logging.error("没有可用的检查器，程序退出")
            return

        logging.info(f"已启用的检查器: {[func.__name__ for func in self.check_funcs]}")

        while True:
            for check_func in self.check_funcs:
                success, msg = check_func()
                if not success and self.state == State.ACTIVE:
                    self.state = State.INACTIVE
                    self.send_text(msg)
                    break

            time.sleep(self.check_interval)


class SlurmMonitor(BaseMonitor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def check_nvidia_smi(self):
        """
        检查GPU显存使用率，若所有GPU显存占用率都过低，则认为可能没有任务在运行。
        使用nvidia-smi命令获取GPU信息，解析显存使用情况。
        若任一GPU显存占用超过阈值，则认为有任务在运行中。
        """
        try:
            output = subprocess.check_output("nvidia-smi", shell=True, stderr=subprocess.STDOUT)
            output_str = output.decode("utf-8").strip()
            
            # 使用正则表达式提取显存使用信息，格式如: "11136MiB / 24564MiB"
            memory_usage = re.findall(r"(\d+)MiB\s+/\s+(\d+)MiB", output_str)
            
            if not memory_usage:
                return False, "无法获取GPU显存信息，可能GPU出现问题"
            
            # 设置显存使用率阈值，低于此值认为GPU空闲
            threshold = 10  # 10%
            
            # 计算每个GPU的显存使用率
            usage_percentages = []
            for used, total in memory_usage:
                used = int(used)
                total = int(total)
                if total > 0:
                    usage_percentages.append((used / total) * 100)
            
            # 检查是否至少有一个GPU的显存使用率超过阈值
            if not usage_percentages:
                return False, "无法计算GPU显存使用率"
            
            has_active_gpu = any(percentage >= threshold for percentage in usage_percentages)
            
            if not has_active_gpu:
                return False, "所有GPU显存占用率均低于{}%，可能没有任务在运行".format(threshold)
            
            return True, "GPU正常运行中"
        
        except subprocess.CalledProcessError as error:
            logging.error(f"nvidia-smi命令执行失败: {error.output.decode('utf-8') if hasattr(error, 'output') else str(error)}")
            return False, "执行nvidia-smi命令失败，无法检查GPU状态"
        except Exception as e:
            logging.error(f"检查GPU显存时发生错误: {str(e)}")
            return False, "检查GPU显存时发生错误"
    
    def check_parajobs(self):
        """
        执行parajobs命令并提取所有的jobs id。
        返回一个列表，列表为空表示没有运行的任务或命令执行失败。
        
        解析逻辑：
        - 使用 subprocess.check_output 执行 "parajobs -s" 命令
        - 通过正则表达式提取每一行以数字开头并紧跟 "|" 的job id, 忽略了表头和其它无关信息
        """
        try:
            output = subprocess.check_output("parajobs -s", shell=True, stderr=subprocess.STDOUT)
            output_str = output.decode("utf-8").strip()
            # 提取每一行以数字开头的内容，忽略表头中的 "JOBID"
            job_ids = re.findall(r"^\s*(\d+)\|", output_str, re.M)
            return len(job_ids) > 0, "没东西在上面跑了"
        except subprocess.CalledProcessError as error:
            logging.error(f"parajobs命令执行失败: {error.output.decode('utf-8')}")
            return False, "没东西在上面跑了"
    
    def check_squeue(self):
        """
        执行squeue命令并检查是否有正在运行的任务。
        
        解析逻辑：
        - 使用 subprocess.check_output 执行 "squeue" 命令
        - 解析输出以确定是否有任务正在运行（通常会有JOBID行以外的行）
        - 返回结果和相应的消息
        """
        try:
            output = subprocess.check_output("squeue", shell=True, stderr=subprocess.STDOUT)
            output_str = output.decode("utf-8").strip()
            
            # 将输出分割成行
            lines = output_str.split('\n')
            
            # 移除空行
            lines = [line for line in lines if line.strip()]
            
            # 如果只有标题行（即只有一行包含"JOBID PARTITION NAME..."）
            # 或者没有行，则表示没有任务在运行
            if len(lines) <= 1:
                return False, "Slurm队列中没有正在运行的任务"
            
            # 提取正在运行的作业数量（排除标题行）
            job_count = len(lines) - 1
            
            # 可选：进一步解析以仅计算状态为'R'（运行中）的作业
            running_jobs = []
            for line in lines[1:]:  # 跳过标题行
                # 尝试提取作业状态（通常在第5列，即ST列）
                parts = line.split()
                if len(parts) >= 5:
                    job_id = parts[0]
                    status = parts[4]  # ST列通常是第5列
                    if status == 'R':
                        running_jobs.append(job_id)
            
            if not running_jobs:
                return False, "Slurm队列中没有正在运行的任务（所有任务均处于非运行状态）"
            
            return True, f"Slurm队列中有{len(running_jobs)}个任务正在运行"
            
        except subprocess.CalledProcessError as error:
            logging.error(f"squeue命令执行失败: {error.output.decode('utf-8') if hasattr(error, 'output') else str(error)}")
            return False, "执行squeue命令失败，无法检查Slurm队列状态"
        except Exception as e:
            logging.error(f"检查Slurm队列时发生错误: {str(e)}")
            return False, "检查Slurm队列时发生错误"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='监控Slurm集群上的任务运行状态')
    parser.add_argument('--url', default="https://open.feishu.cn/open-apis/bot/v2/hook/2c5c687f-1501-42f6-a79b-44a4b2b15ec1",
                        help='飞书机器人webhook URL')
    parser.add_argument('--checkers', nargs='+', default=['nvidia_smi', 'parajobs', 'squeue'],
                        help='要启用的检查器列表，例如 "nvidia_smi parajobs squeue"')
    parser.add_argument('--interval', type=int, default=10,
                        help='检查间隔时间（秒）')
    
    args = parser.parse_args()
    
    monitor = SlurmMonitor(args.url, checkers=args.checkers, check_interval=args.interval)
    monitor.run()