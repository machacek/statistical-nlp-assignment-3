import sys

class LabeledReader(object):
    def __init__(self, filename):
        if filename == '-':
            self.file = sys.stdin
        else:
            self.file = open(filename, mode='r', encoding='utf-8')

    def __iter__(self):
        for line in self.file:
            yield tuple(line.strip().split('/',2))

    def __del__(self):
        if self.file is not sys.stdin:
            self.file.close()

class UnlabeledReader(object):
    def __init__(self, filename):
        if filename == '-':
            self.file = sys.stdin
        else:
            self.file = open(filename, mode='r', encoding='utf-8')

    def __iter__(self):
        for line in self.file:
            yield line.strip()

    def __del__(self):
        if self.file is not sys.stdin:
            self.file.close()
