"""Arestas condicionais: decidem o proximo passo a partir do state, sem chamar o LLM."""

from typing import Literal

from src.state import State


def rotear_triagem(state: State) -> Literal["cardiologico", "neurologico", "urgente", "nao_urgente"]:
    """Escolhe o ramo a partir da triagem.

    Casos nao urgentes vao para o clinico geral, qualquer que seja a especialidade.
    Casos urgentes vao para o especialista indicado ou, se gerais, para a urgencia.
    Sem classificacao, trata como urgente.
    """
    if state.urgencia == "nao_urgente":
        return "nao_urgente"
    if state.especialidade == "cardiologia":
        return "cardiologico"
    if state.especialidade == "neurologia":
        return "neurologico"
    return "urgente"
