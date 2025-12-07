#Importa la clase FastAPI para crear la aplicación.
from fastapi import FastAPI, HTTPException,Form,File,UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
# Importa los modelos a utilizar
from models.User import usuario, registrar_usuario
from models.Login import LoginUsuario, login_usuario 
from models.Sample import RegistarMuestra 
from fastapi.responses import FileResponse
import shutil
import os
from db import get_db

#Crea una instancia de la aplicación FastAPI.
app = FastAPI()
# ✅ Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

#Define una ruta raíz
@app.get("/")
def home():
    print("➡ Working directory (os.getcwd()):", os.getcwd())

    return {"mensaje": "Bienvenido a la API"}

#Ruta para el registro de un usuario
@app.post("/registrar-usuario")
def registrarUsuario(data: usuario):
    try:
        registrar_usuario(data)
        return {"mensaje": "Usuario registrado con éxito"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

#Ruta para inicio de sesión
@app.post("/login")
def login(usuario: LoginUsuario):
    return login_usuario(usuario)

#Ruta para registro de una muestra
@app.post("/registrar-muestra-file")
async def registrar_muestra_file(
    sampleName: str = Form(...),
    idUser: int = Form(...),
    typeSample: str = Form(...),
    volumenSample: str = Form(...),
    factorSample: str = Form(...),
    medioSample: str = Form(...),
    sample_file: UploadFile = File(...)):
    return RegistarMuestra.save_with_file(
        sampleName=sampleName,
        idUser=idUser,
        typeSample=typeSample,
        volumenSample=volumenSample,
        factorSample=factorSample,
        medioSample=medioSample,
        sample_file=sample_file
    )

#Ruta para mostar la imagen procesada
@app.get("/imagen-procesada/{id_muestra}")
def get_processed_image(id_muestra: int):
    try:
        conn = get_db()
        cursor = conn.cursor()

        sql = "SELECT sampleRoute FROM samples WHERE id = %s"
        cursor.execute(sql, (id_muestra,))
        result = cursor.fetchone()
        cursor.close()
        conn.close()

        if not result:
            raise HTTPException(status_code=404, detail="Muestra no encontrada")

        filename = result[0]
        processed_path = f"ia/resultados/clustering_img/{filename}"

        if not os.path.exists(processed_path):
            raise HTTPException(status_code=404, detail="Imagen procesada no encontrada")

        return FileResponse(processed_path, media_type="image/png")

    except Exception as e:
        print(f"ERROR AL CARGAR IMAGEN procesada: {e}") 
        raise HTTPException(status_code=400, detail=f"Error al cargar imagen procesada: {e}")

#Ruta para mostar la imagen original
@app.get("/imagen-original/{id_muestra}")
def get_original_image(id_muestra: int):
    try:
        conn = get_db()
        cursor = conn.cursor()

        sql = "SELECT sampleRoute FROM samples WHERE id = %s"
        cursor.execute(sql, (id_muestra,))
        result = cursor.fetchone()

        cursor.close()
        conn.close()

        if not result:
            raise HTTPException(status_code=404, detail="Muestra no encontrada")

        filename = result[0]
        original_path = f"ia/resultados/img/{filename}"  # Asegúrate de que el archivo está en esta ruta

        if not os.path.exists(original_path):
            raise HTTPException(status_code=404, detail="Imagen original no encontrada")

        return FileResponse(original_path, media_type="image/png")

    except Exception as e:
        print(f"ERROR AL CARGAR IMAGEN ORIGINAL: {e}")  # <-- log
        raise HTTPException(status_code=400, detail=f"Error al cargar imagen original: {e}")

#Ruta para mostar la informacion de la muestra
@app.get("/muestra-info/{id_muestra}")
def getSample(id_muestra: int):
    conn = get_db()
    cursor = conn.cursor()
    sql = "SELECT * FROM samples WHERE id = %s"
    cursor.execute(sql, (id_muestra,))
    result = cursor.fetchall()
    cursor.close()
    conn.close()

    if not result:
        raise HTTPException(status_code=404, detail="Muestra no encontrada")
    
    return {"sample": result[0]}

#Ruta para mostar todas las muestras
@app.get("/samples")
def getSamples():
    conn = get_db()
    cursor = conn.cursor()
    sql = "SELECT * FROM samples WHERE state = 1"  # Solo muestras activas
    cursor.execute(sql)
    result = cursor.fetchall()
    cursor.close()
    conn.close()

    if not result:
        raise HTTPException(status_code=404, detail="Muestras no encontradas")
    
    return {"samples": result}

#Ruta para mostar las muestras de un usuario
@app.get("/samples/{idUser}")
def getSamplesUser(idUser: int):
    try:
        conn = get_db()
        cursor = conn.cursor()

        sql = "SELECT * FROM samples WHERE idUser = %s"
        cursor.execute(sql, (idUser,))
        result = cursor.fetchall()
        cursor.close()
        conn.close()
        # if not result:
        #     raise HTTPException(status_code=404, detail="No cuenta con muestras")
        
        return {"samples": result}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Eror: {e}") 

# Ruta para cambiar el estado de una muestra
@app.put("/sample/{sample_id}")
def update_state_sample(sample_id: int):
    conn = get_db()
    cursor = conn.cursor()
    try:
        sql = "UPDATE samples SET state = 0 WHERE id = %s"
        cursor.execute(sql, (sample_id,))
        conn.commit()
        
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Muestra no encontrada")

        return {"message": "Estado de la muestra actualizado a 0"}
    finally:
        cursor.close()
        conn.close()


# Ruta para editar una muestra
@app.put("/sample/state/{sample_id}")
def update_sample(sample_id: int):
    conn = get_db()
    cursor = conn.cursor()
    try:
        sql = "UPDATE samples SET state = 0 WHERE id = %s"
        cursor.execute(sql, (sample_id,))
        conn.commit()
        
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Muestra no encontrada")

        return {"message": "Estado de la muestra actualizado a 0"}
    finally:
        cursor.close()
        conn.close()
        
#Ruta para editar  una muestra
@app.put("/sample/update/{sample_id}")
async def update_sample(
    sampleName: str = Form(...),
    idUser: int = Form(...),
    typeSample: str = Form(...),
    volumenSample: str = Form(...),
    factorSample: str = Form(...),
    medioSample: str = Form(...),
    timeProcessingSample: str = Form(...),
    countSample:str=Form(...),
    dateSample: str = Form(...),
    timeSample: str = Form(...),

):
    return RegistarMuestra.update_sample(
        sampleName=sampleName,
        idUser=idUser,
        typeSample=typeSample,
        volumenSample=volumenSample,
        factorSample=factorSample,
        medioSample=medioSample,
    )



@app.get("/ping")
async def ping():
    return JSONResponse(content={"status": "ok"})