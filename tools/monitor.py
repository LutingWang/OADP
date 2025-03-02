import requests
import logging
import time
from enum import Enum
import subprocess
import logging
import re
import os
import glob

class State(Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"

class BaseMonitor:
    def __init__(self, url, check_interval=10):
        self.check_interval = check_interval
        self.url = url
        self.state = State.ACTIVE

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
        
    # def check_exp_process(self):
    #     try:
    #         files = glob.glob("logs/*.out")
    #         if not files:
    #             return False, "未在 logs 文件夹中找到 .out 文件。"
    #         latest_file = max(files, key=os.path.getmtime)
    #         # 使用 tail 命令获取最新文件的最后一行
    #         output = subprocess.check_output(["tail", "-n", "1", latest_file], stderr=subprocess.STDOUT)
    #         last_line = output.decode("utf-8").strip()
    #         self.send_text(last_line)
    #         return True, last_line
    #     except subprocess.CalledProcessError as error:
    #         return False, f"tail命令执行失败: {error.output.decode('utf-8')}"
    #     except Exception as e:
    #         return False, f"处理最新的 .out 文件时出错: {e}"

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

if __name__ == "__main__":
    url = "https://open.feishu.cn/open-apis/bot/v2/hook/2c5c687f-1501-42f6-a79b-44a4b2b15ec1"
    monitor = SlurmMonitor(url)
    monitor.run()