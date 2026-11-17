# 将标准输出和标准错误重定向到日志记录器，同时保留原始的输出到控制台

class LoggerWriter:
    def __init__(self, level):
        self.level = level
        self.buffer = ''

    def write(self, message):
        self.buffer += message
        while '\n' in self.buffer:
            line, self.buffer = self.buffer.split('\n', 1)
            self.level(line + '\n')

    def flush(self):
            if self.buffer != '':
                self.level(self.buffer.strip())
                self.buffer = ''
