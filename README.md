
# Árbitro Inteligente — Sistema Fuzzy

Protótipo de sistema especialista baseado em lógica fuzzy para auxílio à tomada de decisão de árbitros de futebol na análise de infrações.

Desenvolvido como trabalho prático da disciplina **Lógica Nebulosa** — UFF.

## Como executar localmente

bash
pip install -r requirements.txt
python -m streamlit run app.py

## Variáveis de entrada

| Variável |          | Descrição |

| Intensidade |       | Agressividade física da infração (0–10) |

| Intenção |          | Grau de malícia ou imprudência (0–10) |

| Região |            | Periculosidade da região do corpo atingida (0–10) |

| Reincidência |      | Histórico disciplinar do jogador na partida (0–10) |

| Contexto |          | Gravidade da situação tática da jogada (0–10) |

## Decisão de saída

| Score |       | Decisão |

| 0 – 3.4 |     | Sem cartão |

| 3.5 – 6.4 |   | Cartão amarelo |

| 6.5 – 10 |    | Cartão vermelho |

## Tecnologias

- Python
- Scikit-Fuzzy
- Streamlit
- Matplotlib
