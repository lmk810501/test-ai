import json
from dataclasses import dataclass, asdict
from os import path, environ

BASE_DIR = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
CONFIG_JSON_FILE_PATH = path.join(BASE_DIR, 'config', 'config.json')
with open(CONFIG_JSON_FILE_PATH, 'r') as f:
    config_json = json.load(f)


@dataclass
class Config:

    DB_HOST: str = config_json['DB']['DB_HOST'],
    DB_PORT: str = config_json['DB']['DB_PORT'],
    DB_USER: str = config_json['DB']['DB_USER'],
    DB_PASSWORD: str = config_json['DB']['DB_PASSWORD'],
    DB_DATABASE: str = config_json['DB']['DB_DATABASE']
    DB_POOL_RECYCLE: int = config_json['DB']['DB_POOL_RECYCLE']

@dataclass
class LocalConfig(Config):

    DB_HOST: str = config_json['DB']['DB_HOST'],
    DB_PORT: str = config_json['DB']['DB_PORT'],
    DB_USER: str = config_json['DB']['DB_USER'],
    DB_PASSWORD: str = config_json['DB']['DB_PASSWORD'],
    DB_DATABASE: str = config_json['DB']['DB_DATABASE']
    DB_POOL_RECYCLE: int = config_json['DB']['DB_POOL_RECYCLE']


@dataclass
class ProdConfig(Config):

    DB_HOST: str = config_json['DB']['DB_HOST'],
    DB_PORT: str = config_json['DB']['DB_PORT'],
    DB_USER: str = config_json['DB']['DB_USER'],
    DB_PASSWORD: str = config_json['DB']['DB_PASSWORD'],
    DB_DATABASE: str = config_json['DB']['DB_DATABASE']
    DB_POOL_RECYCLE: int = config_json['DB']['DB_POOL_RECYCLE']


def conf():
    """
    환경 불러오기
    :return:
    """
    config = dict(prod=ProdConfig(), local=LocalConfig())
    return config.get(environ.get("API_ENV", "local"))
