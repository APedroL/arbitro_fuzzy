# Árbitro Fuzzy: Sistema Especialista para Apoio à Arbitragem no Futebol

A arbitragem no futebol é uma atividade complexa que exige decisões rápidas em cenários frequentemente marcados por subjetividade e incerteza. A interpretação de uma infração depende de diversos fatores, como a intensidade do contato, a intenção do jogador, a região do corpo atingida, o contexto da jogada e o histórico disciplinar do atleta. Embora existam regras estabelecidas pelas entidades responsáveis pelo esporte, a aplicação prática dessas regras envolve julgamento humano e análise contextual.

Nesse cenário, a lógica fuzzy apresenta-se como uma alternativa adequada para modelar problemas que não podem ser representados de forma precisa por sistemas binários tradicionais. Diferentemente da lógica clássica, em que uma afirmação é considerada totalmente verdadeira ou totalmente falsa, a lógica fuzzy permite trabalhar com diferentes graus de pertinência, aproximando o comportamento computacional do raciocínio humano.

Este projeto propõe o desenvolvimento de um sistema especialista baseado em lógica fuzzy para auxiliar a tomada de decisão em lances de futebol. O objetivo é estimar a severidade de uma infração e sugerir uma recomendação disciplinar entre três possíveis resultados: ausência de cartão, cartão amarelo ou cartão vermelho.

É importante destacar que o sistema possui caráter educacional e demonstrativo, não tendo a finalidade de substituir árbitros profissionais, assistentes de vídeo ou protocolos oficiais de arbitragem.

## Objetivos

O principal objetivo deste projeto é demonstrar a aplicação prática da lógica fuzzy na resolução de um problema real caracterizado por incerteza e subjetividade.

Entre os objetivos específicos, destacam-se a modelagem de critérios utilizados na arbitragem esportiva, a construção de um sistema especialista baseado em regras linguísticas, o desenvolvimento de uma interface interativa para análise de lances e a investigação da capacidade da lógica fuzzy de representar decisões humanas em cenários complexos.

## Modelagem do problema

Para representar o processo de tomada de decisão do árbitro, foram selecionadas cinco variáveis de entrada consideradas relevantes na avaliação de uma infração.

A primeira variável corresponde à _intensidade da infração_, representando o nível de força empregado pelo jogador durante o lance. A segunda variável é a _intenção do jogador_, que busca estimar o grau de intencionalidade da ação. A terceira variável considera a _região atingida_, levando em conta o potencial de risco associado à parte do corpo afetada.

Além disso, o sistema incorpora a _reincidência do jogador_, representando a frequência de faltas cometidas ao longo da partida, e o _contexto da jogada_, que avalia o impacto esportivo e tático da infração, como a interrupção de um ataque promissor ou de uma oportunidade clara de gol.

Todas as variáveis são representadas em uma escala contínua de 0 a 10, permitindo uma avaliação gradual e mais próxima da interpretação humana.

## Sistema de inferência fuzzy

O sistema foi implementado utilizando o modelo de inferência de Mamdani, amplamente empregado em sistemas especialistas devido à sua interpretabilidade e facilidade de representação por meio de regras linguísticas.

As variáveis de entrada são modeladas por funções de pertinência triangulares e trapezoidais, permitindo a representação de termos linguísticos como baixo, médio e alto, ou equivalentes específicos de cada variável.

O processo de inferência utiliza o operador mínimo para representar a conjunção lógica AND e o operador máximo para agregação dos resultados das regras ativadas. A etapa de defuzzificação é realizada por meio do método do centroide, responsável por converter o conjunto fuzzy de saída em um valor numérico representativo.

A saída do sistema corresponde a um índice contínuo de severidade da infração, posteriormente interpretado em três categorias disciplinares: sem cartão, cartão amarelo e cartão vermelho.

## Base de regras

O comportamento do sistema é definido por um conjunto de regras linguísticas construídas a partir de princípios gerais utilizados na arbitragem esportiva.

Essas regras estabelecem relações entre as variáveis de entrada e a recomendação disciplinar resultante. Como exemplo, infrações caracterizadas por alta intensidade e elevada intencionalidade tendem a resultar em cartão vermelho, enquanto faltas moderadas e imprudentes geralmente conduzem à aplicação de cartão amarelo.

A utilização de regras linguísticas permite representar o conhecimento humano de forma interpretável e transparente, possibilitando futuras expansões e refinamentos do modelo.

## Tecnologias utilizadas

O projeto foi desenvolvido na linguagem Python e utiliza as bibliotecas Streamlit, NumPy, Matplotlib e Scikit-Fuzzy.

A interface gráfica foi implementada com Streamlit, permitindo a interação do usuário por meio de controles deslizantes e a visualização dos resultados em tempo real. O Scikit-Fuzzy foi empregado na construção do sistema de inferência, enquanto NumPy e Matplotlib foram utilizados no processamento numérico e na geração de visualizações das funções de pertinência.

## Estrutura do projeto

O código-fonte está organizado de forma modular, separando a interface gráfica da lógica de inferência.

O arquivo _app.py_ é responsável pela interface com o usuário e pela apresentação dos resultados. O arquivo _fuzzy_engine.py_ implementa o sistema especialista, definindo as variáveis fuzzy, as funções de pertinência e as regras de inferência. O arquivo _requirements.txt_ contém as dependências necessárias para a execução do projeto.

## Execução

Para executar a aplicação localmente, é necessário clonar o repositório, instalar as dependências e iniciar o servidor Streamlit utilizando os seguintes comandos:

bash

git clone https://github.com/APedroL/arbitro_fuzzy.git
cd arbitro_fuzzy
pip install -r requirements.txt
streamlit run app.py

Após a inicialização, a aplicação estará disponível no endereço `http://localhost:8501`.

## Limitações e trabalhos futuros

Embora o sistema apresente resultados coerentes em diversos cenários, sua base de regras foi construída de forma empírica e ainda não passou por um processo formal de validação com especialistas em arbitragem.

Além disso, a cobertura de regras pode ser expandida para contemplar um maior número de situações específicas encontradas em partidas reais. Trabalhos futuros incluem a validação do modelo por meio da análise de lances oficiais, a comparação das recomendações geradas com decisões tomadas por árbitros e pelo árbitro assistente de vídeo (VAR), bem como o refinamento das funções de pertinência e da base de regras.

O projeto também pode ser aprimorado com a incorporação de dados históricos, métricas quantitativas de desempenho e mecanismos de explicabilidade que permitam identificar quais regras foram ativadas durante o processo de inferência.
