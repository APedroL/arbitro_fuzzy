"""
fuzzy_engine.py
Motor de inferência fuzzy para o sistema de arbitragem de futebol.
Baseado em Scikit-Fuzzy (Mamdani), conforme Szwarcfiter & Markezon.
"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class SistemaArbitroFuzzy:
    """
    Sistema de inferência fuzzy para recomendação de punição em faltas de futebol.

    Variáveis de entrada (universo 0–10):
        - intensidade   : agressividade física da infração
        - intencao      : grau de imprudência/malícia percebida
        - regiao        : periculosidade da região do corpo atingida
        - reincidencia  : histórico disciplinar do jogador no jogo
        - contexto      : gravidade do contexto tático da jogada

    Variável de saída (universo 0–10):
        - cartao        : 0–3.5 = sem cartão, 3.5–6.5 = amarelo, 6.5–10 = vermelho
    """

    def __init__(self):
        self._construir_variaveis()
        self._construir_regras()
        self._construir_sistema()

    # ─── Construção das variáveis e funções de pertinência ────────────────────

    def _construir_variaveis(self):
        u = np.arange(0, 10.01, 0.1)   # universo de discurso

        # ── Entradas ──────────────────────────────────────────────────────────

        self.intensidade = ctrl.Antecedent(u, 'intensidade')
        self.intensidade['leve']     = fuzz.trapmf(u, [0, 0, 2.5, 4.5])
        self.intensidade['moderada'] = fuzz.trimf(u,  [3, 5, 7])
        self.intensidade['grave']    = fuzz.trapmf(u, [5.5, 7.5, 10, 10])

        self.intencao = ctrl.Antecedent(u, 'intencao')
        self.intencao['acidental']   = fuzz.trapmf(u, [0, 0, 2, 4])
        self.intencao['imprudente']  = fuzz.trimf(u,  [2.5, 5, 7.5])
        self.intencao['agressiva']   = fuzz.trapmf(u, [5.5, 7.5, 10, 10])

        self.regiao = ctrl.Antecedent(u, 'regiao')
        self.regiao['baixo_risco']   = fuzz.trapmf(u, [0, 0, 2.5, 4.5])
        self.regiao['medio_risco']   = fuzz.trimf(u,  [3, 5, 7])
        self.regiao['alto_risco']    = fuzz.trapmf(u, [5.5, 7.5, 10, 10])

        self.reincidencia = ctrl.Antecedent(u, 'reincidencia')
        self.reincidencia['baixa']   = fuzz.trapmf(u, [0, 0, 2, 4])
        self.reincidencia['media']   = fuzz.trimf(u,  [2.5, 5, 7.5])
        self.reincidencia['alta']    = fuzz.trapmf(u, [5.5, 7.5, 10, 10])

        self.contexto = ctrl.Antecedent(u, 'contexto')
        self.contexto['normal']      = fuzz.trapmf(u, [0, 0, 2.5, 4.5])
        self.contexto['relevante']   = fuzz.trimf(u,  [3, 5, 7])
        self.contexto['critico']     = fuzz.trapmf(u, [5.5, 7.5, 10, 10])

        # ── Saída ─────────────────────────────────────────────────────────────

        self.cartao = ctrl.Consequent(u, 'cartao', defuzzify_method='centroid')
        self.cartao['sem_cartao']    = fuzz.trapmf(u, [0, 0, 2.5, 4])
        self.cartao['amarelo']       = fuzz.trimf(u,  [3, 5, 7])
        self.cartao['vermelho']      = fuzz.trapmf(u, [6, 7.5, 10, 10])

    # ─── Base de regras (13 regras) ───────────────────────────────────────────

    def _construir_regras(self):
        I  = self.intensidade
        N  = self.intencao
        R  = self.regiao
        Re = self.reincidencia
        C  = self.contexto
        K  = self.cartao

        self.regras = [
            # Regras para Cartão Vermelho (expulsão)
            ctrl.Rule(I['grave']    & N['agressiva'],                         K['vermelho']),
            ctrl.Rule(I['grave']    & N['agressiva']   & R['alto_risco'],     K['vermelho']),
            ctrl.Rule(I['grave']    & Re['alta']       & C['critico'],        K['vermelho']),
            ctrl.Rule(N['agressiva'] & Re['alta'],                            K['vermelho']),
            ctrl.Rule(I['grave']    & R['alto_risco']  & C['critico'],        K['vermelho']),

            # Regras para Cartão Amarelo (advertência)
            ctrl.Rule(I['moderada'] & N['imprudente'],                        K['amarelo']),
            ctrl.Rule(I['grave']    & N['acidental'],                         K['amarelo']),
            ctrl.Rule(I['moderada'] & Re['alta'],                             K['amarelo']),
            ctrl.Rule(N['imprudente'] & R['alto_risco'],                      K['amarelo']),
            ctrl.Rule(I['moderada'] & C['critico'],                           K['amarelo']),
            ctrl.Rule(Re['media']   & N['imprudente']  & C['relevante'],      K['amarelo']),

            # Regras para Sem Cartão
            ctrl.Rule(I['leve']     & N['acidental'],                         K['sem_cartao']),
            ctrl.Rule(I['leve']     & N['acidental']   & R['baixo_risco'],    K['sem_cartao']),
        ]

    # ─── Sistema de controle ──────────────────────────────────────────────────

    def _construir_sistema(self):
        self.sistema_ctrl = ctrl.ControlSystem(self.regras)
        self.simulacao    = ctrl.ControlSystemSimulation(self.sistema_ctrl)

    # ─── Inferência ───────────────────────────────────────────────────────────

    def inferir(self, intensidade: float, intencao: float,
                regiao: float, reincidencia: float, contexto: float) -> dict:
        """
        Realiza a inferência fuzzy e retorna score, decisão, pertinências e justificativa.
        """
        self.simulacao.input['intensidade']  = intensidade
        self.simulacao.input['intencao']     = intencao
        self.simulacao.input['regiao']       = regiao
        self.simulacao.input['reincidencia'] = reincidencia
        self.simulacao.input['contexto']     = contexto

        self.simulacao.compute()
        score = float(self.simulacao.output['cartao'])

        # Decisão categórica
        if score < 3.5:
            decisao = "Sem Cartão"
        elif score < 6.5:
            decisao = "Cartão Amarelo"
        else:
            decisao = "Cartão Vermelho"

        # Pertinências de cada variável
        u = np.arange(0, 10.01, 0.1)
        pertinencias = {
            "intensidade": {
                "leve":     round(float(fuzz.interp_membership(u, self.intensidade['leve'].mf,     intensidade)), 3),
                "moderada": round(float(fuzz.interp_membership(u, self.intensidade['moderada'].mf, intensidade)), 3),
                "grave":    round(float(fuzz.interp_membership(u, self.intensidade['grave'].mf,    intensidade)), 3),
            },
            "intencao": {
                "leve":     round(float(fuzz.interp_membership(u, self.intencao['acidental'].mf,   intencao)), 3),
                "moderada": round(float(fuzz.interp_membership(u, self.intencao['imprudente'].mf,  intencao)), 3),
                "grave":    round(float(fuzz.interp_membership(u, self.intencao['agressiva'].mf,   intencao)), 3),
            },
            "regiao": {
                "leve":     round(float(fuzz.interp_membership(u, self.regiao['baixo_risco'].mf,   regiao)), 3),
                "moderada": round(float(fuzz.interp_membership(u, self.regiao['medio_risco'].mf,   regiao)), 3),
                "grave":    round(float(fuzz.interp_membership(u, self.regiao['alto_risco'].mf,    regiao)), 3),
            },
            "reincidencia": {
                "leve":     round(float(fuzz.interp_membership(u, self.reincidencia['baixa'].mf,   reincidencia)), 3),
                "moderada": round(float(fuzz.interp_membership(u, self.reincidencia['media'].mf,   reincidencia)), 3),
                "grave":    round(float(fuzz.interp_membership(u, self.reincidencia['alta'].mf,    reincidencia)), 3),
            },
            "contexto": {
                "leve":     round(float(fuzz.interp_membership(u, self.contexto['normal'].mf,      contexto)), 3),
                "moderada": round(float(fuzz.interp_membership(u, self.contexto['relevante'].mf,   contexto)), 3),
                "grave":    round(float(fuzz.interp_membership(u, self.contexto['critico'].mf,     contexto)), 3),
            },
        }

        regras_ativadas = self._identificar_regras_ativadas(
            intensidade, intencao, regiao, reincidencia, contexto)
        justificativa = self._gerar_justificativa(
            score, decisao, intensidade, intencao, regiao, reincidencia, contexto, pertinencias)

        return {
            "score": score,
            "decisao": decisao,
            "pertinencias": pertinencias,
            "regras_ativadas": regras_ativadas,
            "justificativa": justificativa,
        }

    # ─── Regras ativadas (texto explicável) ───────────────────────────────────

    def _identificar_regras_ativadas(self, intensidade, intencao,
                                      regiao, reincidencia, contexto) -> list:
        u = np.arange(0, 10.01, 0.1)

        def p(var_mf, val):
            return float(fuzz.interp_membership(u, var_mf, val))

        ativadas = []

        # Verifica cada regra textualmente com threshold > 0.05
        checks = [
            (min(p(self.intensidade['grave'].mf, intensidade),
                 p(self.intencao['agressiva'].mf, intencao)),
             "SE intensidade é GRAVE E intenção é AGRESSIVA → Cartão Vermelho"),

            (min(p(self.intensidade['grave'].mf, intensidade),
                 p(self.reincidencia['alta'].mf, reincidencia),
                 p(self.contexto['critico'].mf, contexto)),
             "SE intensidade é GRAVE E reincidência é ALTA E contexto CRÍTICO → Cartão Vermelho"),

            (min(p(self.intencao['agressiva'].mf, intencao),
                 p(self.reincidencia['alta'].mf, reincidencia)),
             "SE intenção é AGRESSIVA E reincidência é ALTA → Cartão Vermelho"),

            (min(p(self.intensidade['moderada'].mf, intensidade),
                 p(self.intencao['imprudente'].mf, intencao)),
             "SE intensidade é MODERADA E intenção é IMPRUDENTE → Cartão Amarelo"),

            (min(p(self.intensidade['grave'].mf, intensidade),
                 p(self.intencao['acidental'].mf, intencao)),
             "SE intensidade é GRAVE E intenção é ACIDENTAL → Cartão Amarelo"),

            (min(p(self.intensidade['moderada'].mf, intensidade),
                 p(self.reincidencia['alta'].mf, reincidencia)),
             "SE intensidade é MODERADA E reincidência é ALTA → Cartão Amarelo"),

            (min(p(self.intensidade['moderada'].mf, intensidade),
                 p(self.contexto['critico'].mf, contexto)),
             "SE intensidade é MODERADA E contexto é CRÍTICO → Cartão Amarelo"),

            (min(p(self.intensidade['leve'].mf, intensidade),
                 p(self.intencao['acidental'].mf, intencao)),
             "SE intensidade é LEVE E intenção é ACIDENTAL → Sem Cartão"),

            (min(p(self.intencao['imprudente'].mf, intencao),
                 p(self.regiao['alto_risco'].mf, regiao)),
             "SE intenção é IMPRUDENTE E região é ALTO RISCO → Cartão Amarelo"),
        ]

        for ativacao, texto in checks:
            if ativacao > 0.05:
                ativadas.append(f"[ativação: {ativacao:.2f}] {texto}")

        return sorted(ativadas, reverse=True)[:5] if ativadas else ["Nenhuma regra com ativação significativa."]

    # ─── Justificativa explicável ──────────────────────────────────────────────

    def _gerar_justificativa(self, score, decisao, intensidade, intencao,
                              regiao, reincidencia, contexto, pert) -> str:
        fatores = []

        if intensidade >= 7:
            fatores.append("alta intensidade da infração")
        elif intensidade >= 4:
            fatores.append("intensidade moderada da infração")

        if intencao >= 7:
            fatores.append("clara intenção agressiva do jogador")
        elif intencao >= 4:
            fatores.append("conduta imprudente do jogador")

        if regiao >= 7:
            fatores.append("região de alto risco atingida (joelho/tornozelo/cabeça)")
        elif regiao >= 4:
            fatores.append("região sensível atingida")

        if reincidencia >= 7:
            fatores.append("alto histórico de infrações no jogo")
        elif reincidencia >= 4:
            fatores.append("reincidência relevante no jogo")

        if contexto >= 7:
            fatores.append("contexto crítico (ex: interrupção de contra-ataque)")
        elif contexto >= 4:
            fatores.append("contexto de jogada relevante")

        if not fatores:
            return (f"A análise resultou em '{decisao}' (score {score:.1f}/10). "
                    "Todos os fatores avaliados apresentaram baixa relevância, "
                    "indicando uma jogada sem elementos agravantes.")

        fatores_str = ", ".join(fatores[:-1])
        if len(fatores) > 1:
            fatores_str += f" e {fatores[-1]}"
        else:
            fatores_str = fatores[0]

        return (f"A decisão '{decisao}' (score {score:.1f}/10) foi influenciada principalmente "
                f"por: {fatores_str}. O sistema ponderou esses fatores simultaneamente "
                f"por meio de inferência nebulosa com {len(self.regras)} regras.")

    # ─── Visualização das funções de pertinência ──────────────────────────────

    def plotar_funcoes_pertinencia(self, intensidade, intencao,
                                    regiao, reincidencia, contexto) -> dict:
        u = np.arange(0, 10.01, 0.1)

        STYLE = {
            "facecolor": "#131929",
            "edgecolor": "#1e2d4a",
        }

        variaveis = {
            "⚡ Intensidade": (
                self.intensidade,
                ["leve", "moderada", "grave"],
                ["#1db954", "#f5c518", "#e53935"],
                ["Leve", "Moderada", "Grave"],
                intensidade,
            ),
            "🎯 Intenção": (
                self.intencao,
                ["acidental", "imprudente", "agressiva"],
                ["#1db954", "#f5c518", "#e53935"],
                ["Acidental", "Imprudente", "Agressiva"],
                intencao,
            ),
            "🦵 Região": (
                self.regiao,
                ["baixo_risco", "medio_risco", "alto_risco"],
                ["#1db954", "#f5c518", "#e53935"],
                ["Baixo Risco", "Médio Risco", "Alto Risco"],
                regiao,
            ),
            "🔁 Reincidência": (
                self.reincidencia,
                ["baixa", "media", "alta"],
                ["#1db954", "#f5c518", "#e53935"],
                ["Baixa", "Média", "Alta"],
                reincidencia,
            ),
            "🏟️ Contexto": (
                self.contexto,
                ["normal", "relevante", "critico"],
                ["#1db954", "#f5c518", "#e53935"],
                ["Normal", "Relevante", "Crítico"],
                contexto,
            ),
            "🃏 Saída — Cartão": (
                self.cartao,
                ["sem_cartao", "amarelo", "vermelho"],
                ["#1db954", "#f5c518", "#e53935"],
                ["Sem Cartão", "Amarelo", "Vermelho"],
                None,
            ),
        }

        figs = {}
        for titulo, (variavel, conjuntos, cores, labels, valor_atual) in variaveis.items():
            fig, ax = plt.subplots(figsize=(5, 2.5))
            fig.patch.set_facecolor('#131929')
            ax.set_facecolor('#0e1a2e')

            for conj, cor, label in zip(conjuntos, cores, labels):
                mf = variavel[conj].mf
                ax.plot(u, mf, color=cor, linewidth=2, label=label)
                ax.fill_between(u, mf, alpha=0.08, color=cor)

            if valor_atual is not None:
                ax.axvline(x=valor_atual, color='#ffffff', linestyle='--',
                           linewidth=1.2, alpha=0.6)
                ax.text(valor_atual + 0.15, 0.95, f'{valor_atual:.1f}',
                        color='#ffffff', fontsize=8, va='top')

            ax.set_xlim(0, 10)
            ax.set_ylim(0, 1.05)
            ax.set_xlabel("Valor", color='#8891aa', fontsize=8)
            ax.set_ylabel("Pertinência", color='#8891aa', fontsize=8)
            ax.tick_params(colors='#c8d0e0', labelsize=7)
            for spine in ax.spines.values():
                spine.set_color('#1e2d4a')

            leg = ax.legend(fontsize=7, loc='upper right',
                            facecolor='#131929', edgecolor='#1e2d4a',
                            labelcolor='#c8d0e0')

            plt.tight_layout()
            figs[titulo] = fig

        return figs
