"""
fuzzy_engine.py — versão 2.0
Motor de inferência fuzzy para arbitragem de futebol.
Cobertura total do espaço de entrada (243/243 combinações linguísticas).
Sem fallback linear.
"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from itertools import product


class SistemaArbitroFuzzy:

    def __init__(self):
        self._construir_variaveis()
        self._construir_regras()
        self.sistema_ctrl = ctrl.ControlSystem(self.regras)
        self.simulacao = ctrl.ControlSystemSimulation(self.sistema_ctrl)

    def _construir_variaveis(self):
        self.u = np.arange(0, 10.01, 0.1)
        u = self.u

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
            # ── CARTÃO VERMELHO (6 regras) ────────────────────────────────────
            # R01: falta grave com intenção agressiva — expulsão clara
            ctrl.Rule(I['grave']    & N['agressiva'],                    K['vermelho']),
            # R02: agrava R01 com região de alto risco
            ctrl.Rule(I['grave']    & N['agressiva'] & R['alto_risco'],  K['vermelho']),
            # R03: falta grave com reincidência alta e contexto crítico
            ctrl.Rule(I['grave']    & Re['alta']     & C['critico'],     K['vermelho']),
            # R04: intenção agressiva combinada com reincidência alta
            ctrl.Rule(N['agressiva'] & Re['alta'],                       K['vermelho']),
            # R05: falta grave em região perigosa em contexto crítico
            ctrl.Rule(I['grave']    & R['alto_risco'] & C['critico'],    K['vermelho']),
            # R06 (nova): moderada + agressiva — intenção domina sobre intensidade
            ctrl.Rule(I['moderada'] & N['agressiva'],                    K['vermelho']),

            # ── CARTÃO AMARELO (8 regras) ─────────────────────────────────────
            # R07: moderada + imprudente — falta clássica de advertência
            ctrl.Rule(I['moderada'] & N['imprudente'],                   K['amarelo']),
            # R08: grave mas acidental — punição pelo risco, não pela intenção
            ctrl.Rule(I['grave']    & N['acidental'],                    K['amarelo']),
            # R09: moderada com reincidência alta — histórico agrava
            ctrl.Rule(I['moderada'] & Re['alta'],                        K['amarelo']),
            # R10: imprudente em região perigosa — independe da intensidade
            ctrl.Rule(N['imprudente'] & R['alto_risco'],                 K['amarelo']),
            # R11: moderada em contexto crítico — tático agrava
            ctrl.Rule(I['moderada'] & C['critico'],                      K['amarelo']),
            # R12: reincidência média + imprudente + contexto relevante
            ctrl.Rule(Re['media']   & N['imprudente'] & C['relevante'],  K['amarelo']),
            # R13 (nova): leve + agressiva — intenção clara mesmo sem força
            ctrl.Rule(I['leve']     & N['agressiva'],                    K['amarelo']),
            # R14 (nova): grave + imprudente — intensidade alta com alguma intenção
            ctrl.Rule(I['grave']    & N['imprudente'],                   K['amarelo']),

            # ── SEM CARTÃO (4 regras) ─────────────────────────────────────────
            # R15: leve + acidental — falta sem elementos agravantes
            ctrl.Rule(I['leve']     & N['acidental'],                    K['sem_cartao']),
            # R16: agrava R15 com baixo risco de região
            ctrl.Rule(I['leve']     & N['acidental'] & R['baixo_risco'], K['sem_cartao']),
            # R17 (nova): leve + imprudente — baixa força mitiga imprudência
            ctrl.Rule(I['leve']     & N['imprudente'],                   K['sem_cartao']),
            # R18 (nova): moderada + acidental — sem intenção, sem punição severa
            ctrl.Rule(I['moderada'] & N['acidental'],                    K['sem_cartao']),
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
        return self.u

    def get_mfs_saida(self):
        return {
            'sem_cartao': self.cartao['sem_cartao'].mf,
            'amarelo':    self.cartao['amarelo'].mf,
            'vermelho':   self.cartao['vermelho'].mf,
        }

    def get_mfs_entrada(self, variavel):
        var = getattr(self, variavel)
        return {term: var[term].mf for term in var.terms}

    @staticmethod
    def validar_cobertura():
        """
        Percorre as 243 combinações linguísticas e reporta lacunas.
        Retorna (ok, falhas) onde ok é o número de combinações cobertas.
        """
        sistema = SistemaArbitroFuzzy()
        u = sistema.u

        centros = {
            'intensidade':  {'leve': 1.0,  'moderada': 5.0, 'grave': 9.0},
            'intencao':     {'acidental': 1.0, 'imprudente': 5.0, 'agressiva': 9.0},
            'regiao':       {'baixo_risco': 1.0, 'medio_risco': 5.0, 'alto_risco': 9.0},
            'reincidencia': {'baixa': 1.0, 'media': 5.0, 'alta': 9.0},
            'contexto':     {'normal': 1.0, 'relevante': 5.0, 'critico': 9.0},
        }

        ok, falhas = 0, []
        for (il,iv),(nl,nv),(rl,rv),(rel,rev),(cl,cv) in product(
            centros['intensidade'].items(),
            centros['intencao'].items(),
            centros['regiao'].items(),
            centros['reincidencia'].items(),
            centros['contexto'].items(),
        ):
            sistema.simulacao.input['intensidade']  = iv
            sistema.simulacao.input['intencao']     = nv
            sistema.simulacao.input['regiao']       = rv
            sistema.simulacao.input['reincidencia'] = rev
            sistema.simulacao.input['contexto']     = cv
            try:
                sistema.simulacao.compute()
                _ = sistema.simulacao.output['cartao']
                ok += 1
            except Exception:
                falhas.append((il, nl, rl, rel, cl))

        return ok, falhas
