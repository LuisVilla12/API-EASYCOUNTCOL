from pydantic import BaseModel
from datetime import date
from datetime import datetime
from passlib.context import CryptContext
from db import get_db
from fastapi import HTTPException, UploadFile
from PIL import Image  # <-- ¡IMPORTANTE! necesitas importar PILLOW
import shutil
import os
import uuid
import time
import cv2
# from ia.algoritmo_water import tratamiento_imagen
from ia.infer import tratamiento_imagen


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class RegistarMuestra(BaseModel):
    sampleName: str
    idUser: int
    typeSample: str
    volumenSample: str
    factorSample: str
    sampleRoute: str
    processingTime: float
    count: int
    creationDate: date
    creationTime: str
    medioSample: str

    @classmethod
    # METODO PARA GUARDAR MUESTRA
    def save_with_file(cls, sampleName: str, idUser: int, typeSample: str,volumenSample: str,factorSample: str,sample_file: UploadFile, medioSample: str = "N/A"):
        try:
            #Verificar existencia de las carpetas donde esta almacenada las imagenes
            os.makedirs("ia/resultados/img", exist_ok=True)
            os.makedirs("ia/resultados/clustering_img", exist_ok=True)
            
            # Asignar un nombre unico
            filename = f"{uuid.uuid4().hex}_{sample_file.filename}"
            file_location = f"ia/resultados/img/{filename}"
            # Reinicia el puntero del archivo al primer
            sample_file.file.seek(0) 
        
            # Reiniciar lectura del archivo
            with open(file_location, "wb") as buffer:
                # Guarda el archivo subidoa la carpeta uploads
                shutil.copyfileobj(sample_file.file, buffer)
                
            start_time = time.time()  
            # Extreaer los resultados del procesamiento de la imagen
            resultado = tratamiento_imagen(filename)
            end_time = time.time()   
            processing_time = end_time - start_time 
            
            # Asignar variables del resultado
            image_resultado = resultado["image_resultado"]
            labels = resultado["labels"]
            
            # Determinar la hora
            ahoraActual = datetime.now()
            creation_time = ahoraActual.strftime("%H:%M:%S")
            
            # Crear la instancia de RegistarMuestra
            muestra = cls(
                sampleName=sampleName,
                idUser=idUser,
                typeSample=typeSample,
                volumenSample=volumenSample,
                factorSample=factorSample,
                sampleRoute=filename,
                count=labels,
                processingTime=processing_time,
                creationDate=date.today(),
                creationTime=creation_time,
                medioSample=medioSample,
            )

            # 4. Guardar en la base de datos
            conn = get_db()
            cursor = conn.cursor()

            sql = """
                INSERT INTO samples (
                    sampleName, idUser, typeSample, volumenSample,
                    factorSample, sampleRoute, creationDate,processingTime,count,creationTime, medioSample,state
                ) 
                VALUES (%s, %s, %s, %s, %s, %s, %s,%s,%s,%s,%s,1)
            """
            cursor.execute(sql, (
                muestra.sampleName,
                muestra.idUser,
                muestra.typeSample,
                muestra.volumenSample,
                muestra.factorSample,
                muestra.sampleRoute,
                muestra.creationDate,
                muestra.processingTime,
                muestra.count,
                muestra.creationTime,
                muestra.medioSample
            ))
            sample_id = cursor.lastrowid

            conn.commit()
            cursor.close()
            conn.close()

            # Regresar el id de la muestra y mensaje
            return {
                "success": True,
                "idSample": sample_id,
                "message": "Muestra registrada correctamente."
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error al registrar muestra: {e}")
    @classmethod
    # METODO PARA ACTUALIZAR MUESTRA
    def update_sample(cls, sampleID: int, sampleName: str, typeSample: str,
                      volumenSample: str, factorSample: str, medioSample: str):
        try:
            # muestra = cls(
            #     sampleName=sampleName,
            #     typeSample=typeSample,
            #     volumenSample=volumenSample,
            #     factorSample=factorSample,
            #     medioSample=medioSample,
            # )

            conn = get_db()
            cursor = conn.cursor()

            sql = """
                UPDATE samples
                SET 
                    sampleName = %s,
                    typeSample = %s,
                    volumenSample = %s,
                    factorSample = %s,
                    medioSample = %s
                WHERE id = %s
            """
            cursor.execute(sql, (
                sampleName,
                typeSample,
                volumenSample,
                factorSample,
                medioSample,
                sampleID))

            conn.commit()
            cursor.close()
            conn.close()

            return {
                "success": True,
                "idSample": sampleID,
                "message": "Muestra actualizada correctamente."
            }

        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error al actualizar muestra: {e}")