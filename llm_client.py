import os
import time
from groq import Groq
from openai import OpenAI
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

class LLMClient:
    def __init__(self):
        self.groq = Groq(api_key=os.getenv('GROQ_API_KEY'))
        self.openrouter = OpenAI(
            api_key=os.getenv('OPENROUTER_API_KEY'),
            base_url="https://openrouter.ai/api/v1"
        )

    def call(self, system: str, prompt: str, model_info: dict) -> dict:
        start = time.time()
        provider = model_info.get('provider', 'groq')
        model = model_info.get('name', 'llama-3.1-8b-instant')

        try:
            if provider == 'groq':
                res = self.groq.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=512
                )
                text = res.choices[0].message.content
                tokens = res.usage.total_tokens

            elif provider == 'google':
                gemini = genai.GenerativeModel(model)
                full_prompt = f"{system}\n\n{prompt}"
                res = gemini.generate_content(full_prompt)
                text = res.text
                tokens = len(full_prompt.split()) + len(text.split())

            elif provider == 'openrouter':
                res = self.openrouter.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=512
                )
                text = res.choices[0].message.content
                tokens = res.usage.total_tokens if res.usage else 100

            else:
                raise ValueError(f"Unknown provider: {provider}")

        except Exception as e:
            print(f"Provider {provider} failed: {e}, falling back to Groq")
            res = self.groq.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=512
            )
            text = res.choices[0].message.content
            tokens = res.usage.total_tokens

        latency = (time.time() - start) * 1000
        cost = (tokens / 1000) * model_info.get('cost_per_1k', 0.0002)

        return {
            'response': text,
            'tokens': tokens,
            'latency_ms': latency,
            'cost_usd': cost
        }