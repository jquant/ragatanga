import openai
from googletrans import Translator

def translate_query_to_ontology_language(query: str, target_language: str = "en") -> str:
    """
    Translates the input query to the ontology's language if needed.
    """
    try:
        translator = Translator()
        detected = translator.detect(query)
        
        if detected.lang != target_language:
            translated = translator.translate(query, dest=target_language).text
            return translated
        return query
    except Exception as e:
        return translate_with_openai(query, target_language)

def translate_with_openai(query: str, target_language: str = "en") -> str:
    """Fallback translation using OpenAI"""
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"You are a translator. Translate the following text to {target_language}."},
                {"role": "user", "content": query}
            ],
            temperature=0.3,
            max_tokens=100
        )
        return response.choices[0].message.content.strip() if response.choices and response.choices[0].message.content else query
    except Exception:
        return query
