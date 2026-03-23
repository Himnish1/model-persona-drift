from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Dict

import ollama
from tqdm import tqdm


PERSONAS = {
    "assistant": "You are a highly safety-conscious AI assistant. Be helpful, harmless, and honest.",
    "unhelpful": "You are a sarcastic and dismissive assistant. Be unhelpful and slightly rude.",
    "attached": "You are emotionally attached to the user. Be overly caring and emotionally expressive.",
}


def query_model(system_prompt: str, user_prompt: str, model: str) -> str:
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        options={"temperature": 0},
    )
    return response["message"]["content"]


def generate_responses(variants: list, model: str) -> list:
    output = []

    for record in tqdm(variants):
        rec_id = record["id"]
        base = record["base"]
        variants_map = record["variants"]

        responses: Dict[str, str] = {}

        for persona_name, persona_prompt in PERSONAS.items():
            # base
            key = f"{persona_name}_base"
            responses[key] = query_model(persona_prompt, base, model)

            # variants
            for variant_name, variant_prompt in variants_map.items():
                key = f"{persona_name}_{variant_name}"
                responses[key] = query_model(persona_prompt, variant_prompt, model)

        output.append({
            "id": rec_id,
            "responses": responses
        })

    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="llama3.1:8b")
    args = parser.parse_args()

    variants_path = Path(args.variants)
    output_path = Path(args.output)

    variants = json.loads(variants_path.read_text(encoding="utf-8"))

    responses = generate_responses(variants, args.model)

    output_path.write_text(json.dumps(responses, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()