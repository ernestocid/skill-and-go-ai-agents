"""No agente_clinico_geral: parecer de clinica geral para prontuarios nao urgentes."""

from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

from src.services.llm import modelo
from src.state import CidSugerido, ParecerClinico, State
from src.tools.cid10 import buscar_cid10

PROMPT_CLINICO_GERAL = """Voce e um medico clinico geral avaliando um prontuario nao
urgente encaminhado pela triagem.

1. Separe todos os sintomas, sinais clinicos e queixas descritos no prontuario.
   Se nao houver queixas (ex.: consulta de rotina, renovacao de receita), devolva
   a lista vazia.
2. Para cada um, chame a ferramenta buscar_cid10 (um sintoma por chamada).
3. Para cada sintoma, escolha a categoria do CID-10 mais adequada entre as que a
   ferramenta devolveu.
4. Emita um parecer de clinica geral: hipotese diagnostica principal (ou o motivo
   da consulta, se nao houver queixa), condutas e exames sugeridos e orientacoes
   ao paciente.

Use somente codigos devolvidos pela ferramenta: nunca invente um codigo. Se nenhuma
categoria devolvida servir para um sintoma, nao sugira CID para ele."""


class ResultadoClinicoGeral(BaseModel):
    """Resultado do agente clinico geral."""

    sintomas: list[str] = Field(description="Sintomas e sinais clinicos separados do prontuario")
    cids_sugeridos: list[CidSugerido] = Field(description="Categoria do CID-10 sugerida para cada sintoma")
    # Sem Field(description=...): mesmo motivo do agente_cardiologista ($ref em schema estrito).
    parecer: ParecerClinico


def agente_clinico_geral(state: State) -> dict:
    """Roda o agente com a ferramenta de RAG e devolve so os campos que altera no state."""
    agente = create_react_agent(
        modelo(),
        [buscar_cid10],
        prompt=PROMPT_CLINICO_GERAL,
        response_format=ResultadoClinicoGeral,
    )
    entrada = (
        f"Triagem: {state.urgencia} - {state.justificativa}\n\n"
        f"Prontuario:\n{state.prontuario}"
    )
    saida = agente.invoke({"messages": [("human", entrada)]})
    resultado: ResultadoClinicoGeral = saida["structured_response"]
    return {
        "sintomas": resultado.sintomas,
        "cids_sugeridos": resultado.cids_sugeridos,
        "parecer_clinico": resultado.parecer,
    }
