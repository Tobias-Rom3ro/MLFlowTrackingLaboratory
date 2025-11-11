from google import genai

from ..config import get_genai_api_key, GENAI_MODEL


def generate_insights(prompt_text: str) -> str:
    api_key = get_genai_api_key()

    if not api_key:
        return "API key de Gemini no disponible. Configure GOOGLE_API_KEY para obtener insights."

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=GENAI_MODEL,
            contents=prompt_text
        )
        return getattr(response, "text", str(response))
    except Exception as e:
        return f"Error al conectar con Gemini: {str(e)}"