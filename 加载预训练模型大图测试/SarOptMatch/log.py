import sys
import re
import os
from datetime import datetime
import sys
import re
import os
from datetime import datetime

class StreamLogger:
    """
    日志分流器：只记录epoch结束的日志内容，batch进度条不写日志
    """
    def __init__(self, file_path):
        self.original_stdout = sys.stdout  # 保存原始stdout
        self.original_stderr = sys.stderr
        self.log = open(file_path, "a", encoding="utf-8")
        self.pattern = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")

        self.buffer = ""

        # 接管输出
        sys.stdout = self
        sys.stderr = self

    def write(self, message):
        # 原样输出到控制台
        self.original_stdout.write(message)
        self.original_stdout.flush()

        # 累积到缓冲区
        self.buffer += message
        if '\n' in message:
            lines = self.buffer.split('\n')
            for line in lines[:-1]:  # 完整行
                cleaned = self.pattern.sub('', line)
                if not self.is_progress_bar_line(cleaned):
                    self.log.write(cleaned + '\n')
                    self.log.flush()
            self.buffer = lines[-1]  # 缓存最后的半行

    def flush(self):
        pass

    def close(self):
        # 关闭日志文件并恢复stdout
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        self.log.close()

    def is_progress_bar_line(self, line):
        return bool(re.match(r"^\s*\d+/\d+\s+\[", line))

def setup_logger(dataset_name: str, logs_dir="logs"):
    """
    设置日志系统，每个数据集独立一份
    """
    os.makedirs(logs_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{dataset_name}_{timestamp}.log"
    log_path = os.path.join(logs_dir, log_filename)

    logger = StreamLogger(log_path)

    print("=" * 80)
    print(f"🆕 日志已启动！数据集: {dataset_name}，保存路径: {log_path}")
    print("=" * 80)

    return logger

