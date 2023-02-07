import logging

# formatter 생성
formatter = logging.Formatter('[%(asctime)s][%(levelname)s|%(filename)s:%(lineno)s] >> %(message)s')

# handler 생성 (stream, file)
streamHandler = logging.StreamHandler()
fileHandler = logging.FileHandler('E:\\99.lmk810501\\13.AI\\deepface-lmk810501\\deepface.log')

# logger instance에 fomatter 설정
streamHandler.setFormatter(formatter)
fileHandler.setFormatter(formatter)


# logger instance 생성
rootLogger = logging.getLogger()
# logger instance에 handler 설정
rootLogger.setLevel(level=logging.DEBUG)
rootLogger.addHandler(streamHandler)

myLogger = logging.getLogger('MY LOG')
# logger instance에 handler 설정
myLogger.setLevel(level=logging.DEBUG)
myLogger.addHandler(streamHandler)
myLogger.addHandler(fileHandler)
myLogger.propagate = False
