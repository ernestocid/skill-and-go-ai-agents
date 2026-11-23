"""Cria no LangSmith o dataset de triagem a partir de exemplos/prontuarios.json.

Uso: .venv/bin/python exemplos/criar_dataset.py
Requer LANGSMITH_API_KEY no ambiente.
"""

import json
from pathlib import Path

from langsmith import Client

exemplos = json.loads((Path(__file__).parent / "prontuarios.json").read_text(encoding="utf-8"))

client = Client()
ds = client.create_dataset("susia-triagem", description="Prontuarios ficticios, um ou mais por subagente")
client.create_examples(
    dataset_id=ds.id,
    inputs=[e["inputs"] for e in exemplos],
    outputs=[e["outputs"] for e in exemplos],
)
print(f"{len(exemplos)} exemplos criados no dataset {ds.name}")
