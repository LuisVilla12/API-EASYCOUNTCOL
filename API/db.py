import os
import mysql.connector

DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'database'),
    'user': os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASSWORD', 'lkqaz923'),
    'database': os.getenv('DB_NAME', 'easycountcol'),
    'port': int(os.getenv('DB_PORT', '3306')),
    'charset': 'utf8mb4',
    'collation': 'utf8mb4_general_ci'
}
def get_db():
    return mysql.connector.connect(**DB_CONFIG)
