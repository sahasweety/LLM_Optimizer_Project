import os
import time
import logging
from groq import Groq
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Lazy Google genai client
_google_client = None

def _get_google_client():
    global _google_client
    if _google_client is None:
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set in .env")
        from google import genai
        _google_client = genai.Client(api_key=api_key)
    return _google_client


class LLMClient:
    def __init__(self):
        self.groq = Groq(api_key=os.getenv('GROQ_API_KEY'))
        self.openrouter = OpenAI(
            api_key=os.getenv('OPENROUTER_API_KEY'),
            base_url="https://openrouter.ai/api/v1"
        )

    def call(self, system: str, prompt: str, model_info: dict) -> dict:
        start    = time.time()
        provider = model_info.get('provider', 'groq')
        model    = model_info.get('name', 'openai/gpt-oss-20b')

        try:
            if provider == 'groq':
                res = self.groq.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user",   "content": prompt}
                    ],
                    max_tokens=512,
                    timeout=20.0
                )
                text   = res.choices[0].message.content
                tokens = res.usage.total_tokens

            elif provider == 'google':
                from google import genai as gai
                from google.genai import types as gtypes
                client = _get_google_client()
                full_prompt = f"{system}\n\n{prompt}"
                res = client.models.generate_content(
                    model=model,
                    contents=full_prompt,
                    config=gtypes.GenerateContentConfig(
                        max_output_tokens=512,
                        temperature=0.7,
                    )
                )
                text   = res.text
                tokens = len(full_prompt.split()) + len(text.split())

            elif provider == 'openrouter':
                res = self.openrouter.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user",   "content": prompt}
                    ],
                    max_tokens=512,
                    timeout=20.0
                )
                text   = res.choices[0].message.content
                tokens = res.usage.total_tokens if res.usage else 100

            else:
                raise ValueError(f"Unknown provider: {provider}")

        except Exception as e:
            logger.warning(f"Provider '{provider}' / model '{model}' failed: {e}. Falling back to Groq.")
            res = self.groq.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": prompt}
                ],
                max_tokens=512,
                timeout=20.0
            )
            text   = res.choices[0].message.content
            tokens = res.usage.total_tokens

        latency = (time.time() - start) * 1000
        cost    = (tokens / 1000) * model_info.get('cost_per_1k', 0.0002)

        return {
            'response':   text,
            'tokens':     tokens,
            'latency_ms': latency,
            'cost_usd':   cost
        }