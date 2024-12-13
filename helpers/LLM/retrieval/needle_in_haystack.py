import json
import logging
import os
from openai import OpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI API key
openai_api_key = os.getenv("OPEN_AI_API_KEY")

with open(f"text.txt", "r", encoding="ANSI") as f:
    document_content = f.read()


system_prompt = f"""
Retrieve the Pizza recipe

Document Context:
{document_content}
Answer:"""
