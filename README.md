# Viabilidade_Projeto

'''
Este programa implementa um modelo preditivo de viabilidade de projetos usando Regressão Logística com a biblioteca scikit-learn.
Ele foi projetado para ser reutilizável: treina o modelo apenas uma vez, salva em disco, e nas execuções seguintes apenas carrega
o modelo e faz previsões sobre novos projetos.

✅ Objetivo:

Prever se um projeto é viável ou não com base em três variáveis:

investment (investimento necessário)

expected_return (retorno esperado)

impact_score (nota de impacto do projeto)

🔁 Etapas do Programa:

1. Importação de bibliotecas
Importa bibliotecas necessárias para:

manipulação de dados (pandas, numpy)

modelagem (regressão logística, normalização)

avaliação do modelo (classification_report)

persistência de arquivos (joblib, os)

2. Definição da função train_or_predict(new_projects)
Essa é a função central do programa. Ela pode:

Treinar um novo modelo, se ainda não existir um modelo salvo.

Usar um modelo já treinado, se já tiver sido salvo anteriormente.

Fazer previsões sobre novos projetos, se fornecidos.

Internamente, essa função realiza:

Carregamento dos dados históricos de projetos de projects_data.csv.

Separação das variáveis independentes (X) e alvo (y).

Normalização dos dados (com StandardScaler).

Divisão entre treino e teste.

Treinamento de um modelo de regressão logística.

Avaliação do modelo com relatório de métricas.

Salvamento do modelo, scaler e métricas em arquivos .joblib.

Previsão da viabilidade de novos projetos passados como argumento.

3. Teste do modelo com novos dados
Na parte final do código, é feito um teste simples:

new_projects = [
    {"investment": 40000, "expected_return": 60000, "impact_score": 6}
]

Esse dicionário representa um novo projeto hipotético. A função train_or_predict() é chamada com esse projeto, e então:

A função retorna o projeto com a previsão de viabilidade (0 ou 1) e a probabilidade de viabilidade (valores entre 0 e 1).

Também imprime as métricas do modelo previamente treinado.

📦 O que o programa gera:

Arquivos .joblib:

logistic_model.joblib: o modelo de regressão treinado.

logistic_model_scaler.joblib: objeto de normalização.

logistic_model_metrics.joblib: dicionário com métricas (precisão, recall, F1, etc.).

📌 Aplicações:
Esse tipo de programa é comum em ambientes de data science aplicada a negócios ou gestão pública, onde se deseja automatizar a avaliação da viabilidade de projetos usando dados históricos e aprendizado de máquina.

'''
