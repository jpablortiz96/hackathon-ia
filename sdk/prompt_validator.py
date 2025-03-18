from azure.ai.contentsafety import ContentSafetyClient
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI

class PromptValidator:
    def __init__(self, content_safety_key, content_safety_endpoint, text_analytics_key, text_analytics_endpoint, openai_key, openai_endpoint):
        """
        Inicializa el SDK con las credenciales de Azure Content Safety, Text Analytics y OpenAI.
        """
        # Configura el cliente de Content Safety
        self.content_safety_client = ContentSafetyClient(
            endpoint=content_safety_endpoint,
            credential=AzureKeyCredential(content_safety_key)
        )

        # Configura el cliente de Text Analytics
        self.text_analytics_client = TextAnalyticsClient(
            endpoint=text_analytics_endpoint,
            credential=AzureKeyCredential(text_analytics_key)
        )

        # Configura el cliente de OpenAI
        self.openai_client = AzureOpenAI(
            api_key=openai_key,
            api_version="2023-05-15",
            azure_endpoint=openai_endpoint
        )

    def detect_harmful_language(self, text):
        """
        Detecta lenguaje dañino o sensible en el texto usando Azure Content Safety.
        """
        try:
            # Analiza el texto con Content Safety
            request = {"text": text}
            response = self.content_safety_client.analyze_text(request)

            # Devuelve los resultados del análisis
            harmful_categories = []
            for category in response.categories_analysis:
                if category.severity > 0:  # Severity > 0 indica contenido dañino
                    harmful_categories.append(category.category)
            return harmful_categories
        except Exception as e:
            print(f"Error al detectar lenguaje dañino: {e}")
            return []

    def analyze_text(self, text):
        """
        Analiza el texto usando Azure Text Analytics.
        """
        try:
            # Análisis de sentimientos
            sentiment_analysis = self.text_analytics_client.analyze_sentiment([text])
            sentiment = sentiment_analysis[0].sentiment

            # Extracción de frases clave
            key_phrases_analysis = self.text_analytics_client.extract_key_phrases([text])
            key_phrases = key_phrases_analysis[0].key_phrases

            # Detección de información sensible
            pii_analysis = self.text_analytics_client.recognize_pii_entities([text])
            sensitive_data = [entity.text for entity in pii_analysis[0].entities]

            return {
                "sentiment": sentiment,
                "key_phrases": key_phrases,
                "sensitive_data": sensitive_data
            }
        except Exception as e:
            print(f"Error al analizar el texto: {e}")
            return {}

    def suggest_improved_prompt(self, text):
        """
        Sugiere un prompt mejorado usando Azure OpenAI.
        """
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-35-turbo",  # Nombre del despliegue en Azure OpenAI Studio
                messages=[
                    {"role": "system", "content": "Reescribe el siguiente texto para que sea más claro y preciso:"},
                    {"role": "user", "content": text}
                ],
                max_tokens=100,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error al generar sugerencia con OpenAI: {e}")
            return text  # Devuelve el texto original si hay un error