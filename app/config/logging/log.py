import logging

# formatter 생성
formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s|%(filename)s:%(lineno)s] >> %(message)s')

# handler 생성 (stream, file)
streamHandler = logging.StreamHandler()
fileHandler = logging.FileHandler('E:\\99.lmk810501\\13.AI\\deepface-lmk810501\\deepface.log')

# handler 생성 (stream, file) 디텍팅 성공
streamHandler_detected_success = logging.StreamHandler()
fileHandler_detected_success = logging.FileHandler('E:\\99.lmk810501\\13.AI\\deepface-lmk810501\\detected_success.log')

# handler 생성 (stream, file) 디텍팅 실패
streamHandler_detected_fail = logging.StreamHandler()
fileHandler_detected_fail = logging.FileHandler('E:\\99.lmk810501\\13.AI\\deepface-lmk810501\\detected_fail.log')

# logger instance에 fomatter 설정
streamHandler.setFormatter(formatter)
fileHandler.setFormatter(formatter)

streamHandler_detected_success.setFormatter(formatter)
fileHandler_detected_success.setFormatter(formatter)

streamHandler_detected_fail.setFormatter(formatter)
fileHandler_detected_fail.setFormatter(formatter)

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


detectedSuccessLogger = logging.getLogger('DETECTED SUCCESS LOG')
# logger instance에 handler 설정
detectedSuccessLogger.setLevel(level=logging.DEBUG)
detectedSuccessLogger.addHandler(streamHandler_detected_success)
detectedSuccessLogger.addHandler(fileHandler_detected_success)
detectedSuccessLogger.propagate = False


detectedFailLogger = logging.getLogger('DETECTED FAIL LOG')
# logger instance에 handler 설정
detectedFailLogger.setLevel(level=logging.DEBUG)
detectedFailLogger.addHandler(streamHandler_detected_fail)
detectedFailLogger.addHandler(fileHandler_detected_fail)
detectedFailLogger.propagate = False
