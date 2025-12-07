from pydantic import BaseModel, EmailStr
from API.db import get_db
from fastapi import HTTPException
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

class usuario(BaseModel):
    name: str
    lastname: str
    username: str
    email: EmailStr
    password: str

def registrar_usuario(data: usuario):
    try:
        conn = get_db()
        cursor = conn.cursor()

        # Hashear la contraseña antes de guardar
        hashed_password = hash_password(data.password)

        sql = """
            INSERT INTO users (name, lastname, username, email, password)
            VALUES (%s, %s, %s, %s, %s)
        """
        cursor.execute(sql, (data.name, data.lastname, data.username, data.email, hashed_password))
        conn.commit()

        cursor.close()
        conn.close()

        return {"success": True, "message": "Usuario registrado correctamente."}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al registrar: {e}")
