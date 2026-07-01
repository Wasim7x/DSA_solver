import os
from dotenv import load_dotenv
from autogen_ext.models.openai import OpenAIChatCompletionClient
from config.constants import MODEL, GEMINI_API_KEY
# Load environment variables
load_dotenv()

api_key = os.getenv(GEMINI_API_KEY)


GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


def get_model_client():
    if not api_key:
        raise ValueError(f"Please set the {GEMINI_API_KEY} environment variable.")

    # Initialize the Gemini model client via OpenAI-compatible API
    model_client = OpenAIChatCompletionClient(
        model=MODEL,          # e.g. "gemini-2.0-flash" or "gemini-1.5-pro"
        api_key=api_key,
        base_url=GEMINI_BASE_URL,
        model_info={
            "vision": True,
            "function_calling": True,
            "json_output": True,
            "family": "unknown",  # required by autogen when using a non-OpenAI model
        },
    )
    return model_client