from pydantic import BaseModel, EmailStr
from fastapi import HTTPException
from API.db import get_db
from passlib.context import CryptContext

class LoginUsuario(BaseModel):
    email: EmailStr
    password: str
    
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verificar_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)



def login_usuario(data: LoginUsuario):
    conn = get_db()
    cursor = conn.cursor()

    # Buscar usuario por email
    cursor.execute("SELECT id,name,lastname,username,email,password  FROM users WHERE email = %s", (data.email,))
    user = cursor.fetchone()

    if not user:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")

    id, name,lastname,username, email, hashed_password = user

    # Verificar contraseña
    if not verificar_password(data.password, hashed_password):
        raise HTTPException(status_code=401, detail="Contraseña incorrecta")

    cursor.close()
    conn.close()

    return {
        "success": True,
        "message": f"Bienvenido {name}",
        "name": name,
        "lastname": lastname, 
        "username": username,         
        "email": email, 
        "idUser": id
    }
