"""
fuzzy_engine.py
Motor de inferência fuzzy para arbitragem de futebol.
"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


class SistemaArbitroFuzzy:

    def __init__(self):
        self._construir_variaveis()
        self._construir_regras()
        self.sistema_ctrl = ctrl.ControlSystem(self.regras)
        self.simulacao = ctrl.ControlSystemSimulation(self.sistema_ctrl)

    def _construir_variaveis(self):
        u = np.arange(0, 10.01, 0.1)

        # Entradas
        self.intensidade = ctrl.Antecedent(u, 'intensidade')
        self.intensidade['leve']     = fuzz.trapmf(u, [0, 0, 2.5, 4.5])
        self.intensidade['moderada'] = fuzz.trimf(u,  [3, 5, 7])
        self.intensidade['grave']    = fuzz.trapmf(u, [5.5, 7.5, 10, 10])

        self.intencao = ctrl.Antecedent(u, 'intencao')
        self.intencao['acidental']  = fuzz.trapmf(u, [0, 0, 2, 4])
        self.intencao['imprudente'] = fuzz.trimf(u,  [2.5, 5, 7.5])
        self.intencao['agressiva']  = fuzz.trapmf(u, [5.5, 7.5, 10, 10])

        self.regiao = ctrl.Antecedent(u, 'regiao')
        self.regiao['baixo_risco'] = fuzz.trapmf(u, [0, 0, 2.5, 4.5])
        self.regiao['medio_risco'] = fuzz.trimf(u,  [3, 5, 7])
        self.regiao['alto_risco']  = fuzz.trapmf(u, [5.5, 7.5, 10, 10])

        self.reincidencia = ctrl.Antecedent(u, 'reincidencia')
        self.reincidencia['baixa'] = fuzz.trapmf(u, [0, 0, 2, 4])
        self.reincidencia['media'] = fuzz.trimf(u,  [2.5, 5, 7.5])
        self.reincidencia['alta']  = fuzz.trapmf(u, [5.5, 7.5, 10, 10])

        self.contexto = ctrl.Antecedent(u, 'contexto')
        self.contexto['normal']    = fuzz.trapmf(u, [0, 0, 2.5, 4.5])
        self.contexto['relevante'] = fuzz.trimf(u,  [3, 5, 7])
        self.contexto['critico']   = fuzz.trapmf(u, [5.5, 7.5, 10, 10])

        # Saída
        self.cartao = ctrl.Consequent(u, 'cartao', defuzzify_method='centroid')
        self.cartao['sem_cartao'] = fuzz.trapmf(u, [0, 0, 2.5, 4])
        self.cartao['amarelo']    = fuzz.trimf(u,  [3, 5, 7])
        self.cartao['vermelho']   = fuzz.trapmf(u, [6, 7.5, 10, 10])

    def _construir_regras(self):
        I  = self.intensidade
        N  = self.intencao
        R  = self.regiao
        Re = self.reincidencia
        C  = self.contexto
        K  = self.cartao

        self.regras = [
            # Cartão Vermelho
            ctrl.Rule(I['grave']    & N['agressiva'],                     K['vermelho']),
            ctrl.Rule(I['grave']    & N['agressiva'] & R['alto_risco'],   K['vermelho']),
            ctrl.Rule(I['grave']    & Re['alta']     & C['critico'],      K['vermelho']),
            ctrl.Rule(N['agressiva'] & Re['alta'],                        K['vermelho']),
            ctrl.Rule(I['grave']    & R['alto_risco'] & C['critico'],     K['vermelho']),
            # Cartão Amarelo
            ctrl.Rule(I['moderada'] & N['imprudente'],                    K['amarelo']),
            ctrl.Rule(I['grave']    & N['acidental'],                     K['amarelo']),
            ctrl.Rule(I['moderada'] & Re['alta'],                         K['amarelo']),
            ctrl.Rule(N['imprudente'] & R['alto_risco'],                  K['amarelo']),
            ctrl.Rule(I['moderada'] & C['critico'],                       K['amarelo']),
            ctrl.Rule(Re['media']   & N['imprudente'] & C['relevante'],   K['amarelo']),
            # Sem Cartão
            ctrl.Rule(I['leve']     & N['acidental'],                     K['sem_cartao']),
            ctrl.Rule(I['leve']     & N['acidental'] & R['baixo_risco'],  K['sem_cartao']),
        ]

    def inferir(self, intensidade, intencao, regiao, reincidencia, contexto):
        self.simulacao.input['intensidade']  = intensidade
        self.simulacao.input['intencao']     = intencao
        self.simulacao.input['regiao']       = regiao
        self.simulacao.input['reincidencia'] = reincidencia
        self.simulacao.input['contexto']     = contexto
        self.simulacao.compute()
        return float(self.simulacao.output['cartao'])

    def get_universo(self):
        return np.arange(0, 10.01, 0.1)

    def get_mfs_saida(self):
        u = self.get_universo()
        return {
            'sem_cartao': self.cartao['sem_cartao'].mf,
            'amarelo':    self.cartao['amarelo'].mf,
            'vermelho':   self.cartao['vermelho'].mf,
        }

    def get_mfs_entrada(self, variavel):
        u = self.get_universo()
        var = getattr(self, variavel)
        return {term: var[term].mf for term in var.terms}
