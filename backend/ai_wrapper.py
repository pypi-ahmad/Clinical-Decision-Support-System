"""
AI Wrapper Module (The Universal Adapter)
-----------------------------------------
This module serves as a unified interface for interacting with various AI providers 
(Ollama, OpenAI, Anthropic, Google Gemini). It abstracts away the specific API 
implementations, allowing the rest of the application to switch models dynamically.

Usage:
    from backend.ai_wrapper import get_ai_response, clean_json_output
    response = get_ai_response("Ollama", "llama3", None, "System Prompt", "User Input")
"""

import ollama
from openai import OpenAI
import anthropic
import google.genai as genai
import re

def get_ai_response(provider, model, api_key, system_prompt, user_text):
    """
    Universal wrapper for AI text-to-text generation.

    Args:
        provider (str): The name of the AI provider ("Ollama", "OpenAI", "Gemini", "Anthropic").
        model (str): The specific model identifier (e.g., "gpt-4o", "claude-3-opus").
        api_key (str): The API key for the provider (None for Ollama).
        system_prompt (str): Instructions for the AI's behavior and role.
        user_text (str): The actual input text or query from the user.

    Returns:
        str: The raw text response from the AI model.
        str: Error message if the call fails.
    """
    try:
        # --- 1. LOCAL (OLLAMA) ---
        # Ollama runs locally and typically does not require an API key.
        if provider == "Ollama":
            response = ollama.chat(
                model=model,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_text}
                ]
            )
            return response['message']['content']

        # --- 2. OPENAI (GPT) ---
        # Connects to OpenAI's API. Requires an API key.
        # We use response_format={"type": "json_object"} to ensure structured output.
        elif provider == "OpenAI":
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text}
                ],
                response_format={"type": "json_object"}  # Force JSON for structure
            )
            return response.choices[0].message.content

        # --- 3. GOOGLE (GEMINI) ---
        # Connects to Google's Generative AI. Requires an API key.
        elif provider == "Gemini":
            genai.configure(api_key=api_key)
            gemini_model = genai.GenerativeModel(model)
            # Gemini often works better when system prompt is combined with user input
            full_prompt = f"{system_prompt}\n\nUSER INPUT: {user_text}"
            response = gemini_model.generate_content(full_prompt)
            return response.text

        # --- 4. ANTHROPIC (CLAUDE) ---
        # Connects to Anthropic's Claude API. Requires an API key.
        elif provider == "Anthropic":
            client = anthropic.Anthropic(api_key=api_key)
            response = client.messages.create(
                model=model,
                max_tokens=4096,
                system=system_prompt,
                messages=[{"role": "user", "content": user_text}]
            )
            return response.content[0].text

        return f"Error with {provider}: Unsupported provider"

    except Exception as e:
        return f"Error with {provider}: {str(e)}"

def clean_json_output(text):
    """
    Helper function to clean and extract JSON from model output.
    
    Many AI models wrap JSON in markdown code blocks (e.g., ```json ... ```).
    This function removes those markers and extracts the JSON object.

    Args:
        text (str): The raw string response from the AI.

    Returns:
        str: The cleaned string containing only the JSON object.
    """
    # Remove markdown code block syntax
    text = text.replace("```json", "").replace("```", "").strip()
    
    # regex search to extract everything between the first '{' and the last '}'
    match = re.search(r'(\{.*\})', text, re.DOTALL)
    if match:
        return match.group(1)
    return text
