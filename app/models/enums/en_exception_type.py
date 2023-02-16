from enum import Enum


class EnExceptionType(Enum):
    S0001 = ('S0001', '성공')
    E0000 = ('E0000', '알 수 없는 오류 입니다')
    E0001 = ('E0001', '이미지 파일이 아닙니다')
    E0002 = ('E0002', '얼굴을 찾을 수 없습니다')

    def __init__(self, code, desc):
        self.code = code
        self.desc = desc

    def __str__(self):
        return self.code
