import sys

class SPLReader(object):
    def __init__(self, filename):
        if filename == '-':
            self.file = sys.stdin
        else:
            self.file = open(filename, mode='r', encoding='utf-8')

    def __iter__(self):
        for line in self.file:
            yield [item.split('/',2) for item in line.split()]

    def __del__(self):
        if self.file is not sys.stdin:
            self.file.close()

class SentenceReader(object):
    def __init__(self, filename):
        if filename == '-':
            self.file = sys.stdin
        else:
            self.file = open(filename, mode='r', encoding='utf-8')

    def __iter__(self):
        for line in self.file:
            yield line.split()

    def __del__(self):
        if self.file is not sys.stdin:
            self.file.close()
