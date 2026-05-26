import streamlit as st
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from fuzzy_engine import SistemaArbitroFuzzy

# ─── Configuração da Página ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Árbitro IA — Sistema Fuzzy",
    page_icon="🟨",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Oswald:wght@400;600;700&family=Inter:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0a0f1e;
    color: #e8eaf0;
}

.main { background-color: #0a0f1e; }

h1, h2, h3 { font-family: 'Oswald', sans-serif; letter-spacing: 0.04em; }

.hero-title {
    font-family: 'Oswald', sans-serif;
    font-size: 3.2rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    background: linear-gradient(135deg, #f5c518 0%, #ff6b35 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
    margin-bottom: 0.3rem;
}

.hero-sub {
    font-size: 0.95rem;
    color: #8891aa;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    font-weight: 300;
}

.card {
    background: #131929;
    border: 1px solid #1e2d4a;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

.card-title {
    font-family: 'Oswald', sans-serif;
    font-size: 0.8rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #f5c518;
    margin-bottom: 1rem;
    border-bottom: 1px solid #1e2d4a;
    padding-bottom: 0.5rem;
}

.resultado-box {
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin: 1rem 0;
}

.resultado-nada {
    background: linear-gradient(135deg, #0d2b1e, #0a1a14);
    border: 2px solid #1db954;
}

.resultado-amarelo {
    background: linear-gradient(135deg, #2b2100, #1a1500);
    border: 2px solid #f5c518;
}

.resultado-vermelho {
    background: linear-gradient(135deg, #2b0d0d, #1a0808);
    border: 2px solid #e53935;
}

.resultado-label {
    font-family: 'Oswald', sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

.resultado-score {
    font-size: 4rem;
    font-weight: 700;
    font-family: 'Oswald', sans-serif;
    line-height: 1;
}

.badge {
    display: inline-block;
    padding: 0.2rem 0.7rem;
    border-radius: 20px;
    font-size: 0.72rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    font-weight: 500;
}

.badge-verde { background: #0d2b1e; color: #1db954; border: 1px solid #1db954; }
.badge-amarelo { background: #2b2100; color: #f5c518; border: 1px solid #f5c518; }
.badge-vermelho { background: #2b0d0d; color: #e53935; border: 1px solid #e53935; }

.regra-ativada {
    background: #0e1a2e;
    border-left: 3px solid #f5c518;
    padding: 0.5rem 0.8rem;
    border-radius: 0 8px 8px 0;
    font-size: 0.85rem;
    margin: 0.3rem 0;
    color: #c8d0e0;
}

.metric-box {
    background: #0e1a2e;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}

.metric-value {
    font-family: 'Oswald', sans-serif;
    font-size: 2rem;
    font-weight: 600;
    color: #f5c518;
}

.metric-label {
    font-size: 0.72rem;
    color: #8891aa;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

div[data-testid="stSlider"] > label { color: #c8d0e0 !important; font-size: 0.9rem; }

.stSlider [data-baseweb="slider"] { color: #f5c518; }
</style>
""", unsafe_allow_html=True)

# ─── Header ───────────────────────────────────────────────────────────────────
col_logo, col_title = st.columns([1, 5])
with col_logo:
    st.markdown("<div style='font-size:4rem;text-align:center;margin-top:0.5rem'>⚽</div>", unsafe_allow_html=True)
with col_title:
    st.markdown('<div class="hero-title">Árbitro Inteligente</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Sistema Especialista com Lógica Fuzzy — Análise de Infrações</div>', unsafe_allow_html=True)

st.markdown("---")

# ─── Inicializa engine ────────────────────────────────────────────────────────
@st.cache_resource
def carregar_sistema():
    return SistemaArbitroFuzzy()

sistema = carregar_sistema()

# ─── Layout principal ─────────────────────────────────────────────────────────
col_entrada, col_resultado = st.columns([1, 1], gap="large")

with col_entrada:
    st.markdown('<div class="card-title">📋 Parâmetros da Falta</div>', unsafe_allow_html=True)

    intensidade = st.slider(
        "⚡ Intensidade da Falta",
        0.0, 10.0, 5.0, 0.1,
        help="0 = contato mínimo, 10 = falta extremamente violenta"
    )

    intencao = st.slider(
        "🎯 Intenção / Agressividade",
        0.0, 10.0, 4.0, 0.1,
        help="0 = totalmente acidental, 10 = falta claramente agressiva"
    )

    regiao = st.slider(
        "🦵 Periculosidade da Região Atingida",
        0.0, 10.0, 4.0, 0.1,
        help="0 = contato no tronco/pé, 10 = cabeça/joelho/tornozelo em risco"
    )

    reincidencia = st.slider(
        "🔁 Reincidência (histórico no jogo)",
        0.0, 10.0, 2.0, 0.1,
        help="0 = primeira infração, 10 = histórico grave no jogo"
    )

    contexto = st.slider(
        "🏟️ Gravidade do Contexto da Jogada",
        0.0, 10.0, 3.0, 0.1,
        help="0 = disputa normal, 10 = interrupção de contra-ataque / fora da jogada"
    )

    st.markdown("---")

    analisar = st.button("🔍 Analisar Falta", use_container_width=True, type="primary")

# ─── Análise ──────────────────────────────────────────────────────────────────
if analisar or True:  # Mostra resultado em tempo real
    resultado = sistema.inferir(intensidade, intencao, regiao, reincidencia, contexto)

    with col_resultado:
        # Decisão principal
        score = resultado["score"]
        decisao = resultado["decisao"]
        cor_classe = {"Sem Cartão": "nada", "Cartão Amarelo": "amarelo", "Cartão Vermelho": "vermelho"}[decisao]
        emoji_mapa = {"Sem Cartão": "✅", "Cartão Amarelo": "🟨", "Cartão Vermelho": "🟥"}

        st.markdown(f"""
        <div class="resultado-box resultado-{cor_classe}">
            <div style="font-size:4rem;margin-bottom:0.5rem">{emoji_mapa[decisao]}</div>
            <div class="resultado-label">{decisao}</div>
            <div class="resultado-score" style="margin-top:0.5rem">{score:.1f}<span style="font-size:1.5rem;color:#8891aa">/10</span></div>
        </div>
        """, unsafe_allow_html=True)

        # Métricas de pertinência
        st.markdown('<div class="card-title" style="margin-top:1rem">📊 Pertinências das Entradas</div>', unsafe_allow_html=True)

        pert = resultado["pertinencias"]
        variaveis_label = {
            "intensidade": ("⚡", "Intensidade"),
            "intencao": ("🎯", "Intenção"),
            "regiao": ("🦵", "Região"),
            "reincidencia": ("🔁", "Reincidência"),
            "contexto": ("🏟️", "Contexto"),
        }

        for var, (emoji, label) in variaveis_label.items():
            p = pert.get(var, {})
            leve_val = p.get("leve", 0)
            mod_val = p.get("moderada", 0)
            grave_val = p.get("grave", 0)

            dominante = max(p, key=p.get) if p else "—"
            badge_map = {"leve": "verde", "moderada": "amarelo", "grave": "vermelho"}
            badge_cor = badge_map.get(dominante, "amarelo")

            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:0.7rem;margin:0.4rem 0;background:#0e1a2e;padding:0.6rem 0.8rem;border-radius:8px">
                <span style="font-size:1.1rem">{emoji}</span>
                <span style="flex:1;font-size:0.85rem;color:#c8d0e0">{label}</span>
                <span style="font-size:0.75rem;color:#8891aa">L:{leve_val:.2f} M:{mod_val:.2f} G:{grave_val:.2f}</span>
                <span class="badge badge-{badge_cor}">{dominante}</span>
            </div>
            """, unsafe_allow_html=True)

# ─── Seção inferior — Regras e Gráficos ───────────────────────────────────────
st.markdown("---")
tab_regras, tab_graficos, tab_sobre = st.tabs(["📜 Regras Ativadas & Justificativa", "📈 Funções de Pertinência", "ℹ️ Sobre o Sistema"])

with tab_regras:
    resultado = sistema.inferir(intensidade, intencao, regiao, reincidencia, contexto)

    col_j, col_r = st.columns([1, 1])
    with col_j:
        st.markdown("#### 💬 Justificativa da Decisão")
        st.info(resultado["justificativa"])

        st.markdown("#### 🏷️ Regras Principais Ativadas")
        for regra in resultado["regras_ativadas"]:
            st.markdown(f'<div class="regra-ativada">📌 {regra}</div>', unsafe_allow_html=True)

    with col_r:
        st.markdown("#### 📊 Contribuição por Fator")
        fatores = {
            "Intensidade": intensidade,
            "Intenção": intencao,
            "Região": regiao,
            "Reincidência": reincidencia,
            "Contexto": contexto,
        }

        fig, ax = plt.subplots(figsize=(5, 3))
        fig.patch.set_facecolor('#131929')
        ax.set_facecolor('#0e1a2e')

        nomes = list(fatores.keys())
        valores = list(fatores.values())
        cores = ['#1db954' if v < 4 else '#f5c518' if v < 7 else '#e53935' for v in valores]

        bars = ax.barh(nomes, valores, color=cores, edgecolor='none', height=0.5)
        ax.set_xlim(0, 10)
        ax.set_xlabel("Valor", color='#8891aa', fontsize=8)
        ax.tick_params(colors='#c8d0e0', labelsize=8)
        for spine in ax.spines.values():
            spine.set_color('#1e2d4a')
        ax.axvline(x=resultado["score"], color='#f5c518', linestyle='--', alpha=0.5, linewidth=1.5)
        ax.text(resultado["score"] + 0.2, -0.7, f'Score: {resultado["score"]:.1f}', color='#f5c518', fontsize=7)

        for bar, val in zip(bars, valores):
            ax.text(val + 0.1, bar.get_y() + bar.get_height()/2, f'{val:.1f}', va='center', color='#e8eaf0', fontsize=8)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

with tab_graficos:
    st.markdown("#### Funções de Pertinência — Visualização das Variáveis Fuzzy")

    figs = sistema.plotar_funcoes_pertinencia(intensidade, intencao, regiao, reincidencia, contexto)

    cols = st.columns(2)
    for i, (nome, fig) in enumerate(figs.items()):
        with cols[i % 2]:
            st.markdown(f"**{nome}**")
            st.pyplot(fig)
            plt.close(fig)

with tab_sobre:
    st.markdown("""
    ### 🧠 Como funciona este sistema?

    Este protótipo implementa um **Sistema Especialista baseado em Lógica Fuzzy** para auxiliar árbitros de futebol na análise de infrações.

    #### Variáveis de Entrada
    | Variável | Conjuntos Fuzzy | Descrição |
    |---|---|---|
    | Intensidade | Leve / Moderada / Grave | Agressividade física da infração |
    | Intenção | Acidental / Imprudente / Agressiva | Grau de malícia percebida |
    | Região | Baixo / Médio / Alto risco | Periculosidade da região atingida |
    | Reincidência | Baixa / Média / Alta | Histórico disciplinar no jogo |
    | Contexto | Normal / Relevante / Crítico | Situação tática da jogada |

    #### Variável de Saída
    | Faixa | Decisão |
    |---|---|
    | 0 – 3.5 | ✅ Sem Cartão |
    | 3.5 – 6.5 | 🟨 Cartão Amarelo |
    | 6.5 – 10 | 🟥 Cartão Vermelho |

    #### Processo de Inferência
    1. **Fuzzificação** — valores numéricos convertidos em graus de pertinência
    2. **Inferência** — regras SE-ENTÃO aplicadas com operador mínimo (Mamdani)
    3. **Defuzzificação** — centroide do conjunto resultante como score final

    #### Tecnologias
    - **Python 3** + **Scikit-Fuzzy** (motor fuzzy)
    - **Streamlit** (interface interativa)
    - **Matplotlib** (visualizações)
    """)
