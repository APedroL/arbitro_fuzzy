"""
fuzzy_engine.py — versão 3.0

Motor de inferência fuzzy para arbitragem de futebol.

Características:
- Inferência Mamdani
- Defuzzificação por centroide
- Explicabilidade baseada em regras ativadas
- Cálculo dos graus de pertinência
- Nova simulação a cada inferência
- Cobertura validável do espaço linguístico
"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from skfuzzy import interp_membership
from itertools import product


class SistemaArbitroFuzzy:

    def __init__(self):
        self._construir_variaveis()
        self._construir_regras()
        self.sistema_ctrl = ctrl.ControlSystem(self.regras)

    def _construir_variaveis(self):
        self.u = np.arange(0, 10.01, 0.1)
        u = self.u

        self.intensidade = ctrl.Antecedent(u, 'intensidade')
        self.intensidade['leve'] = fuzz.trapmf(u, [0, 0, 2.5, 4.5])
        self.intensidade['moderada'] = fuzz.trimf(u, [3, 5, 7])
        self.intensidade['grave'] = fuzz.trapmf(u, [5.5, 7.5, 10, 10])

        self.intencao = ctrl.Antecedent(u, 'intencao')
        self.intencao['acidental'] = fuzz.trapmf(u, [0, 0, 2, 4])
        self.intencao['imprudente'] = fuzz.trimf(u, [2.5, 5, 7.5])
        self.intencao['agressiva'] = fuzz.trapmf(u, [5.5, 7.5, 10, 10])

        self.regiao = ctrl.Antecedent(u, 'regiao')
        self.regiao['baixo_risco'] = fuzz.trapmf(u, [0, 0, 2.5, 4.5])
        self.regiao['medio_risco'] = fuzz.trimf(u, [3, 5, 7])
        self.regiao['alto_risco'] = fuzz.trapmf(u, [5.5, 7.5, 10, 10])

        self.reincidencia = ctrl.Antecedent(u, 'reincidencia')
        self.reincidencia['baixa'] = fuzz.trapmf(u, [0, 0, 2, 4])
        self.reincidencia['media'] = fuzz.trimf(u, [2.5, 5, 7.5])
        self.reincidencia['alta'] = fuzz.trapmf(u, [5.5, 7.5, 10, 10])

        self.contexto = ctrl.Antecedent(u, 'contexto')
        self.contexto['normal'] = fuzz.trapmf(u, [0, 0, 2.5, 4.5])
        self.contexto['relevante'] = fuzz.trimf(u, [3, 5, 7])
        self.contexto['critico'] = fuzz.trapmf(u, [5.5, 7.5, 10, 10])

        self.cartao = ctrl.Consequent(
            u,
            'cartao',
            defuzzify_method='centroid'
        )

        self.cartao['sem_cartao'] = fuzz.trapmf(u, [0, 0, 2.5, 4])
        self.cartao['amarelo'] = fuzz.trimf(u, [3, 5, 7])
        self.cartao['vermelho'] = fuzz.trapmf(u, [6, 7.5, 10, 10])

    def _adicionar_regra(
        self,
        regra,
        regra_id,
        descricao,
        saida,
        antecedentes
    ):
        self.regras.append(regra)

        self.regras_metadata.append({
            "id": regra_id,
            "descricao": descricao,
            "saida": saida,
            "antecedentes": antecedentes
        })

    def _construir_regras(self):

        I = self.intensidade
        N = self.intencao
        R = self.regiao
        Re = self.reincidencia
        C = self.contexto
        K = self.cartao

        self.regras = []
        self.regras_metadata = []

        # R01
        self._adicionar_regra(
            ctrl.Rule(I['grave'] & N['agressiva'], K['vermelho']),
            "R01",
            "SE intensidade é grave E intenção é agressiva ENTÃO cartão vermelho",
            "vermelho",
            [
                ("intensidade", "grave"),
                ("intencao", "agressiva")
            ]
        )

        # R02
        self._adicionar_regra(
            ctrl.Rule(I['grave'] & Re['alta'] & C['critico'], K['vermelho']),
            "R02",
            "SE intensidade é grave E reincidência é alta E contexto é crítico ENTÃO cartão vermelho",
            "vermelho",
            [
                ("intensidade", "grave"),
                ("reincidencia", "alta"),
                ("contexto", "critico")
            ]
        )

        # R03
        self._adicionar_regra(
            ctrl.Rule(N['agressiva'] & Re['alta'], K['vermelho']),
            "R03",
            "SE intenção é agressiva E reincidência é alta ENTÃO cartão vermelho",
            "vermelho",
            [
                ("intencao", "agressiva"),
                ("reincidencia", "alta")
            ]
        )

        # R04
        self._adicionar_regra(
            ctrl.Rule(I['grave'] & R['alto_risco'] & C['critico'], K['vermelho']),
            "R04",
            "SE intensidade é grave E região é de alto risco E contexto é crítico ENTÃO cartão vermelho",
            "vermelho",
            [
                ("intensidade", "grave"),
                ("regiao", "alto_risco"),
                ("contexto", "critico")
            ]
        )

        # R05
        self._adicionar_regra(
            ctrl.Rule(I['moderada'] & N['agressiva'], K['vermelho']),
            "R05",
            "SE intensidade é moderada E intenção é agressiva ENTÃO cartão vermelho",
            "vermelho",
            [
                ("intensidade", "moderada"),
                ("intencao", "agressiva")
            ]
        )

        # R06
        self._adicionar_regra(
            ctrl.Rule(I['moderada'] & N['imprudente'], K['amarelo']),
            "R06",
            "SE intensidade é moderada E intenção é imprudente ENTÃO cartão amarelo",
            "amarelo",
            [
                ("intensidade", "moderada"),
                ("intencao", "imprudente")
            ]
        )

        # R07
        self._adicionar_regra(
            ctrl.Rule(I['grave'] & N['acidental'], K['amarelo']),
            "R07",
            "SE intensidade é grave E intenção é acidental ENTÃO cartão amarelo",
            "amarelo",
            [
                ("intensidade", "grave"),
                ("intencao", "acidental")
            ]
        )

        # R08
        self._adicionar_regra(
            ctrl.Rule(I['moderada'] & Re['alta'], K['amarelo']),
            "R08",
            "SE intensidade é moderada E reincidência é alta ENTÃO cartão amarelo",
            "amarelo",
            [
                ("intensidade", "moderada"),
                ("reincidencia", "alta")
            ]
        )

        # R09
        self._adicionar_regra(
            ctrl.Rule(N['imprudente'] & R['alto_risco'], K['amarelo']),
            "R09",
            "SE intenção é imprudente E região é de alto risco ENTÃO cartão amarelo",
            "amarelo",
            [
                ("intencao", "imprudente"),
                ("regiao", "alto_risco")
            ]
        )

        # R10
        self._adicionar_regra(
            ctrl.Rule(I['moderada'] & C['critico'], K['amarelo']),
            "R10",
            "SE intensidade é moderada E contexto é crítico ENTÃO cartão amarelo",
            "amarelo",
            [
                ("intensidade", "moderada"),
                ("contexto", "critico")
            ]
        )

        # R11
        self._adicionar_regra(
            ctrl.Rule(Re['media'] & N['imprudente'] & C['relevante'], K['amarelo']),
            "R11",
            "SE reincidência é média E intenção é imprudente E contexto é relevante ENTÃO cartão amarelo",
            "amarelo",
            [
                ("reincidencia", "media"),
                ("intencao", "imprudente"),
                ("contexto", "relevante")
            ]
        )

        # R12
        self._adicionar_regra(
            ctrl.Rule(I['leve'] & N['agressiva'], K['amarelo']),
            "R12",
            "SE intensidade é leve E intenção é agressiva ENTÃO cartão amarelo",
            "amarelo",
            [
                ("intensidade", "leve"),
                ("intencao", "agressiva")
            ]
        )

        # R13
        self._adicionar_regra(
            ctrl.Rule(I['grave'] & N['imprudente'], K['amarelo']),
            "R13",
            "SE intensidade é grave E intenção é imprudente ENTÃO cartão amarelo",
            "amarelo",
            [
                ("intensidade", "grave"),
                ("intencao", "imprudente")
            ]
        )

        # R14
        self._adicionar_regra(
            ctrl.Rule(I['leve'] & N['acidental'], K['sem_cartao']),
            "R14",
            "SE intensidade é leve E intenção é acidental ENTÃO sem cartão",
            "sem_cartao",
            [
                ("intensidade", "leve"),
                ("intencao", "acidental")
            ]
        )

        # R15
        self._adicionar_regra(
            ctrl.Rule(I['leve'] & N['acidental'] & R['baixo_risco'], K['sem_cartao']),
            "R15",
            "SE intensidade é leve E intenção é acidental E região é de baixo risco ENTÃO sem cartão",
            "sem_cartao",
            [
                ("intensidade", "leve"),
                ("intencao", "acidental"),
                ("regiao", "baixo_risco")
            ]
        )

        # R16
        self._adicionar_regra(
            ctrl.Rule(I['leve'] & N['imprudente'], K['sem_cartao']),
            "R16",
            "SE intensidade é leve E intenção é imprudente ENTÃO sem cartão",
            "sem_cartao",
            [
                ("intensidade", "leve"),
                ("intencao", "imprudente")
            ]
        )

        # R17
        self._adicionar_regra(
            ctrl.Rule(I['moderada'] & N['acidental'], K['sem_cartao']),
            "R17",
            "SE intensidade é moderada E intenção é acidental ENTÃO sem cartão",
            "sem_cartao",
            [
                ("intensidade", "moderada"),
                ("intencao", "acidental")
            ]
        )

    def calcular_pertinencias(
        self,
        intensidade,
        intencao,
        regiao,
        reincidencia,
        contexto
    ):

        entradas = {
            "intensidade": intensidade,
            "intencao": intencao,
            "regiao": regiao,
            "reincidencia": reincidencia,
            "contexto": contexto
        }

        resultado = {}

        for nome, valor in entradas.items():

            variavel = getattr(self, nome)
            resultado[nome] = {}

            for termo in variavel.terms:
                resultado[nome][termo] = float(
                    interp_membership(
                        self.u,
                        variavel[termo].mf,
                        valor
                    )
                )

        return resultado

    def obter_regras_ativadas(self, pertinencias):

        regras_ativadas = []

        for regra in self.regras_metadata:

            graus = [
                pertinencias[variavel][termo]
                for variavel, termo in regra["antecedentes"]
            ]

            grau = min(graus)

            if grau > 0:
                regras_ativadas.append({
                    "id": regra["id"],
                    "descricao": regra["descricao"],
                    "saida": regra["saida"],
                    "grau": float(grau)
                })

        regras_ativadas.sort(
            key=lambda r: r["grau"],
            reverse=True
        )

        return regras_ativadas

    def inferir(
        self,
        intensidade,
        intencao,
        regiao,
        reincidencia,
        contexto
    ):

        simulacao = ctrl.ControlSystemSimulation(
            self.sistema_ctrl
        )

        simulacao.input['intensidade'] = intensidade
        simulacao.input['intencao'] = intencao
        simulacao.input['regiao'] = regiao
        simulacao.input['reincidencia'] = reincidencia
        simulacao.input['contexto'] = contexto

        simulacao.compute()

        score = float(simulacao.output['cartao'])

        pertinencias = self.calcular_pertinencias(
            intensidade,
            intencao,
            regiao,
            reincidencia,
            contexto
        )

        regras_ativadas = self.obter_regras_ativadas(
            pertinencias
        )

        if score < 3.5:
            categoria = "Sem Cartão"
        elif score < 6.5:
            categoria = "Cartão Amarelo"
        else:
            categoria = "Cartão Vermelho"

        return {
            "score": score,
            "categoria": categoria,
            "pertinencias": pertinencias,
            "regras_ativadas": regras_ativadas
        }

    def get_universo(self):
        return self.u

    def get_mfs_saida(self):
        return {
            'sem_cartao': self.cartao['sem_cartao'].mf,
            'amarelo': self.cartao['amarelo'].mf,
            'vermelho': self.cartao['vermelho'].mf,
        }

    def get_mfs_entrada(self, variavel):
        var = getattr(self, variavel)
        return {
            termo: var[termo].mf
            for termo in var.terms
        }

    @staticmethod
    def validar_cobertura():

        sistema = SistemaArbitroFuzzy()

        centros = {
            'intensidade': {'leve': 1.0, 'moderada': 5.0, 'grave': 9.0},
            'intencao': {'acidental': 1.0, 'imprudente': 5.0, 'agressiva': 9.0},
            'regiao': {'baixo_risco': 1.0, 'medio_risco': 5.0, 'alto_risco': 9.0},
            'reincidencia': {'baixa': 1.0, 'media': 5.0, 'alta': 9.0},
            'contexto': {'normal': 1.0, 'relevante': 5.0, 'critico': 9.0},
        }

        ok = 0
        falhas = []

        for (il, iv), (nl, nv), (rl, rv), (rel, rev), (cl, cv) in product(
            centros['intensidade'].items(),
            centros['intencao'].items(),
            centros['regiao'].items(),
            centros['reincidencia'].items(),
            centros['contexto'].items(),
        ):

            simulacao = ctrl.ControlSystemSimulation(
                sistema.sistema_ctrl
            )

            simulacao.input['intensidade'] = iv
            simulacao.input['intencao'] = nv
            simulacao.input['regiao'] = rv
            simulacao.input['reincidencia'] = rev
            simulacao.input['contexto'] = cv

            try:
                simulacao.compute()
                _ = simulacao.output['cartao']
                ok += 1

            except Exception:
                falhas.append((il, nl, rl, rel, cl))

        return ok, falhas
