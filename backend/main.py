from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sdk.prompt_validator import PromptValidator
from dotenv import load_dotenv
import os

# Cargar variables de entorno
load_dotenv()

# Configura las credenciales de Azure
content_safety_key = os.getenv("CONTENT_SAFETY_KEY")
content_safety_endpoint = os.getenv("CONTENT_SAFETY_ENDPOINT")
text_analytics_key = os.getenv("TEXT_ANALYTICS_KEY")
text_analytics_endpoint = os.getenv("TEXT_ANALYTICS_ENDPOINT")
openai_key = os.getenv("OPENAI_KEY")
openai_endpoint = os.getenv("OPENAI_ENDPOINT")

# Crea una instancia del SDK
validator = PromptValidator(
    content_safety_key=content_safety_key,
    content_safety_endpoint=content_safety_endpoint,
    text_analytics_key=text_analytics_key,
    text_analytics_endpoint=text_analytics_endpoint,
    openai_key=openai_key,
    openai_endpoint=openai_endpoint
)

# Crea la aplicación FastAPI
app = FastAPI()

# Configurar CORS manualmente
@app.middleware("http")
async def add_cors_header(request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response

# Endpoint para la ruta raíz
@app.get("/")
def read_root():
    return {"message": "Bienvenido al backend de la hackathon de IA con Azure."}

# Define el modelo de datos para el prompt
class PromptRequest(BaseModel):
    text: str

# Endpoint para validar y corregir el prompt
@app.post("/validate-prompt")
def validate_prompt(request: PromptRequest):
    try:
        text = request.text

        # Detección de lenguaje dañino
        harmful_categories = validator.detect_harmful_language(text)
        if harmful_categories:
            return {
                "status": "error",
                "message": "El prompt contiene lenguaje dañino o sensible.",
                "categories": harmful_categories
            }

        # Análisis del texto
        text_analysis = validator.analyze_text(text)

        # Sugerencia de prompt mejorado
        improved_prompt = validator.suggest_improved_prompt(text)

        return {
            "status": "success",
            "message": "El prompt es válido y ha sido optimizado.",
            "improved_prompt": improved_prompt,
            "sentiment": text_analysis["sentiment"],
            "key_phrases": text_analysis["key_phrases"],
            "sensitive_data": text_analysis["sensitive_data"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))