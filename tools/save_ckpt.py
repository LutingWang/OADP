import os
import shutil
import time
import logging

# 确保 cache 目录存在
if not os.path.exists('cache'):
    os.makedirs('cache')

# 设置日志配置，将日志保存在 cache 目录中
log_file_path = os.path.join('cache', 'copy_checkpoint.log')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename=log_file_path, filemode='a')

def copy_last_checkpoint_to_cache():
    try:
        # 读取 last_checkpoint 文件中的路径
        with open('work_dirs/last_checkpoint', 'r') as f:
            checkpoint_path = f.read().strip()
        
        # 检查路径是否存在
        if os.path.exists(checkpoint_path):
            # 复制文件到 cache 目录
            shutil.copy(checkpoint_path, 'cache')
            logging.info(f"Copied {checkpoint_path} to cache.")
        else:
            logging.warning(f"Checkpoint path {checkpoint_path} does not exist.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")

def main():
    while True:
        logging.info("Starting the copy process.")
        copy_last_checkpoint_to_cache()
        # 每隔六个小时执行一次
        logging.info("Sleeping for six hours.")
        time.sleep(6 * 60 * 60)

if __name__ == "__main__":
    main()
