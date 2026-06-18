import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from fuzzy_engine import SistemaArbitroFuzzy

st.title("Sistema Fuzzy para Arbitragem de Futebol")
st.write("Protótipo de sistema especialista para auxílio à decisão de árbitros.")

def carregar_sistema():
    return SistemaArbitroFuzzy()

sistema = carregar_sistema()

# ── Entradas ──────────────────────────────────────────────────────────────────
st.header("Parâmetros da Falta")

intensidade = st.slider("Intensidade da falta", 0.0, 10.0, 5.0, 0.1,
    help="0 = contato mínimo | 10 = falta muito violenta")

intencao = st.slider("Intenção / agressividade", 0.0, 10.0, 4.0, 0.1,
    help="0 = acidental | 10 = claramente agressiva")

regiao = st.slider("Periculosidade da região atingida", 0.0, 10.0, 4.0, 0.1,
    help="0 = tronco / pé | 10 = cabeça / joelho / tornozelo")

reincidencia = st.slider("Reincidência no jogo", 0.0, 10.0, 2.0, 0.1,
    help="0 = primeira infração | 10 = histórico grave na partida")

contexto = st.slider("Gravidade do contexto da jogada", 0.0, 10.0, 3.0, 0.1,
    help="0 = disputa normal | 10 = interrupção de contra-ataque")

# ── Inferência ────────────────────────────────────────────────────────────────
st.header("Resultado")

score = sistema.inferir(intensidade, intencao, regiao, reincidencia, contexto)

if score < 3.5:
    decisao = "Sem Cartão"
elif score < 6.5:
    decisao = "Cartão Amarelo"
else:
    decisao = "Cartão Vermelho"

col1, col2 = st.columns(2)
col1.metric("Score de punição", f"{score:.2f} / 10")
col2.metric("Decisão recomendada", decisao)

# ── Gráfico da saída fuzzy ────────────────────────────────────────────────────
st.subheader("Funções de pertinência — Saída (Cartão)")

u = sistema.get_universo()
mfs = sistema.get_mfs_saida()

fig, ax = plt.subplots(figsize=(7, 3))
ax.plot(u, mfs['sem_cartao'], label='Sem Cartão',  color='green')
ax.plot(u, mfs['amarelo'],    label='Amarelo',      color='goldenrod')
ax.plot(u, mfs['vermelho'],   label='Vermelho',     color='red')
ax.axvline(x=score, color='black', linestyle='--', linewidth=1.5, label=f'Score = {score:.2f}')
ax.set_xlabel("Valor de saída")
ax.set_ylabel("Pertinência")
ax.set_ylim(0, 1.1)
ax.legend()
ax.grid(True, alpha=0.3)
st.pyplot(fig)
plt.close()

# ── Gráficos das entradas ─────────────────────────────────────────────────────
st.subheader("Funções de pertinência — Entradas")

variaveis = {
    "Intensidade":  ("intensidade",  intensidade,  ['leve', 'moderada', 'grave']),
    "Intenção":     ("intencao",     intencao,     ['acidental', 'imprudente', 'agressiva']),
    "Região":       ("regiao",       regiao,       ['baixo_risco', 'medio_risco', 'alto_risco']),
    "Reincidência": ("reincidencia", reincidencia, ['baixa', 'media', 'alta']),
    "Contexto":     ("contexto",     contexto,     ['normal', 'relevante', 'critico']),
}

cores = ['green', 'goldenrod', 'red']

col_a, col_b = st.columns(2)
colunas = [col_a, col_b]

for i, (titulo, (var_nome, valor_atual, termos)) in enumerate(variaveis.items()):
    mfs_entrada = sistema.get_mfs_entrada(var_nome)

    fig, ax = plt.subplots(figsize=(4, 2.5))
    for j, termo in enumerate(termos):
        ax.plot(u, mfs_entrada[termo], label=termo, color=cores[j])
    ax.axvline(x=valor_atual, color='black', linestyle='--', linewidth=1, label=f'= {valor_atual:.1f}')
    ax.set_title(titulo)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    colunas[i % 2].pyplot(fig)
    plt.close()

# ── Justificativa simples ─────────────────────────────────────────────────────
st.subheader("Justificativa")

fatores = []
if intensidade >= 7:
    fatores.append("alta intensidade da infração")
elif intensidade >= 4:
    fatores.append("intensidade moderada")

if intencao >= 7:
    fatores.append("intenção claramente agressiva")
elif intencao >= 4:
    fatores.append("conduta imprudente")

if regiao >= 7:
    fatores.append("região de alto risco atingida")

if reincidencia >= 7:
    fatores.append("alto histórico de infrações na partida")
elif reincidencia >= 4:
    fatores.append("reincidência relevante")

if contexto >= 7:
    fatores.append("contexto crítico da jogada")

if fatores:
    st.info(f"Decisão '{decisao}' influenciada por: {', '.join(fatores)}.")
else:
    st.info("Fatores sem agravantes significativos. Jogada de baixo risco.")
