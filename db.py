import os
import mysql.connector

DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'user': os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASSWORD', 'lkqaz923'),
    'database': os.getenv('DB_NAME', 'easycountcol'),
    'port': int(os.getenv('DB_PORT', '3307')),
    'charset': 'utf8mb4',
    'collation': 'utf8mb4_general_ci'
}
def get_db():
    db=mysql.connector.connect(**DB_CONFIG)
    print("Conexión a la base de datos exitosa")
    return db
