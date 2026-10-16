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

        actual_provider = provider
        actual_model    = model
        fallback_occurred = False
        fallback_reason = None

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
            fallback_reason = f"Provider '{provider}' ({model}) failed: {str(e)}"
            logger.warning(f"{fallback_reason}. Explicitly falling back to OpenRouter meta-llama/llama-3.1-8b-instruct.")
            fallback_occurred = True
            try:
                # Try fallback via OpenRouter Llama first
                res = self.openrouter.chat.completions.create(
                    model="meta-llama/llama-3.1-8b-instruct",
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user",   "content": prompt}
                    ],
                    max_tokens=512,
                    timeout=20.0
                )
                text = res.choices[0].message.content
                tokens = res.usage.total_tokens if res.usage else 100
                actual_provider = 'openrouter'
                actual_model = 'meta-llama/llama-3.1-8b-instruct'
            except Exception as e2:
                # Fallback to Groq if OpenRouter also failed
                logger.warning(f"Secondary fallback failed: {e2}. Falling back to Groq openai/gpt-oss-20b.")
                res = self.groq.chat.completions.create(
                    model="openai/gpt-oss-20b",
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user",   "content": prompt}
                    ],
                    max_tokens=512,
                    timeout=20.0
                )
                text = res.choices[0].message.content
                tokens = res.usage.total_tokens
                actual_provider = 'groq'
                actual_model = 'openai/gpt-oss-20b'

        latency = (time.time() - start) * 1000
        cost    = (tokens / 1000) * model_info.get('cost_per_1k', 0.0002)

        return {
            'response':          text,
            'tokens':            tokens,
            'latency_ms':        latency,
            'cost_usd':          cost,
            'intended_model':    model,
            'intended_provider': provider,
            'actual_model':      actual_model,
            'actual_provider':   actual_provider,
            'fallback_occurred': fallback_occurred,
            'fallback_reason':   fallback_reason
        }