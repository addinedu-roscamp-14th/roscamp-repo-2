import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'core-super-secret-key-2026'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///' + os.path.join(BASE_DIR, 'core_auto.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    WTF_CSRF_ENABLED = True
    TIMEZONE = 'Asia/Seoul'
    SYSTEM_MAINTENANCE_START = '03:00'
    SYSTEM_MAINTENANCE_END = '05:00'
    DEFAULT_RESERVATION_INTERVAL = 10