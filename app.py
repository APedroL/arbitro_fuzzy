import streamlit as st
import matplotlib.pyplot as plt

from fuzzy_engine import SistemaArbitroFuzzy

st.set_page_config(
    page_title="Árbitro Fuzzy",
    page_icon="⚽",
    layout="wide"
)


@st.cache_resource
def carregar_sistema():
    return SistemaArbitroFuzzy()


sistema = carregar_sistema()

st.title("Sistema Fuzzy para Arbitragem de Futebol")
st.write(
    "Protótipo de sistema especialista para auxílio à decisão de árbitros."
)

# ── Entradas ────────────────────────────────────────────────────────────────

st.header("Parâmetros da Falta")

intensidade = st.slider(
    "Intensidade da falta",
    0.0,
    10.0,
    5.0,
    0.1,
    help="0 = contato mínimo | 10 = falta muito violenta"
)

intencao = st.slider(
    "Intenção / agressividade",
    0.0,
    10.0,
    4.0,
    0.1,
    help="0 = acidental | 10 = claramente agressiva"
)

regiao = st.slider(
    "Periculosidade da região atingida",
    0.0,
    10.0,
    4.0,
    0.1,
    help="0 = tronco / pé | 10 = cabeça / joelho / tornozelo"
)

reincidencia = st.slider(
    "Reincidência no jogo",
    0.0,
    10.0,
    2.0,
    0.1,
    help="0 = primeira infração | 10 = histórico grave na partida"
)

contexto = st.slider(
    "Gravidade do contexto da jogada",
    0.0,
    10.0,
    3.0,
    0.1,
    help="0 = disputa normal | 10 = interrupção de contra-ataque"
)

# ── Inferência ──────────────────────────────────────────────────────────────

resultado = sistema.inferir(
    intensidade,
    intencao,
    regiao,
    reincidencia,
    contexto
)

score = resultado["score"]
decisao = resultado["categoria"]

# ── Resultado principal ─────────────────────────────────────────────────────

st.header("Resultado")

col1, col2 = st.columns(2)

col1.metric(
    "Score de punição",
    f"{score:.2f} / 10"
)

col2.metric(
    "Decisão recomendada",
    decisao
)

# ── Gráfico da saída fuzzy ──────────────────────────────────────────────────

st.subheader("Funções de pertinência — Saída (Cartão)")

u = sistema.get_universo()
mfs = sistema.get_mfs_saida()

fig, ax = plt.subplots(figsize=(7, 3))

ax.plot(
    u,
    mfs['sem_cartao'],
    label='Sem Cartão',
    color='green'
)

ax.plot(
    u,
    mfs['amarelo'],
    label='Amarelo',
    color='goldenrod'
)

ax.plot(
    u,
    mfs['vermelho'],
    label='Vermelho',
    color='red'
)

ax.axvline(
    x=score,
    color='black',
    linestyle='--',
    linewidth=1.5,
    label=f'Score = {score:.2f}'
)

ax.set_xlabel("Valor de saída")
ax.set_ylabel("Pertinência")
ax.set_ylim(0, 1.1)
ax.legend()
ax.grid(True, alpha=0.3)

st.pyplot(fig)

plt.close()

# ── Gráficos das entradas ───────────────────────────────────────────────────

st.subheader("Funções de pertinência — Entradas")

variaveis = {
    "Intensidade": (
        "intensidade",
        intensidade,
        ['leve', 'moderada', 'grave']
    ),
    "Intenção": (
        "intencao",
        intencao,
        ['acidental', 'imprudente', 'agressiva']
    ),
    "Região": (
        "regiao",
        regiao,
        ['baixo_risco', 'medio_risco', 'alto_risco']
    ),
    "Reincidência": (
        "reincidencia",
        reincidencia,
        ['baixa', 'media', 'alta']
    ),
    "Contexto": (
        "contexto",
        contexto,
        ['normal', 'relevante', 'critico']
    ),
}

cores = ['green', 'goldenrod', 'red']

col_a, col_b = st.columns(2)
colunas = [col_a, col_b]

for i, (titulo, (var_nome, valor_atual, termos)) in enumerate(variaveis.items()):

    mfs_entrada = sistema.get_mfs_entrada(var_nome)

    fig, ax = plt.subplots(figsize=(4, 2.5))

    for j, termo in enumerate(termos):
        ax.plot(
            u,
            mfs_entrada[termo],
            label=termo,
            color=cores[j]
        )

    ax.axvline(
        x=valor_atual,
        color='black',
        linestyle='--',
        linewidth=1,
        label=f'= {valor_atual:.1f}'
    )

    ax.set_title(titulo)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    colunas[i % 2].pyplot(fig)

    plt.close()

# ── Explicação da decisão ───────────────────────────────────────────────────

st.header("Explicação da decisão")

with st.expander("Resultado da inferência", expanded=True):

    st.write(f"**Score final:** {score:.2f}")
    st.write(f"**Categoria recomendada:** {decisao}")

with st.expander("Regras ativadas"):

    regras = resultado["regras_ativadas"]

    if regras:

        for regra in regras:

            st.markdown(
                f"**{regra['id']}** — {regra['descricao']}"
            )

            st.write(
                f"Grau de ativação: **{regra['grau']:.2f}**"
            )

            st.divider()

    else:
        st.info("Nenhuma regra foi ativada.")

with st.expander("Graus de pertinência"):

    nomes_variaveis = {
        "intensidade": "Intensidade",
        "intencao": "Intenção",
        "regiao": "Região",
        "reincidencia": "Reincidência",
        "contexto": "Contexto"
    }

    for variavel, termos in resultado["pertinencias"].items():

        st.markdown(f"### {nomes_variaveis[variavel]}")

        for termo, grau in termos.items():

            termo_formatado = termo.replace("_", " ").capitalize()

            st.write(
                f"• {termo_formatado}: {grau:.2f}"
            )
