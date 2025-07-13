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


#!/usr/bin/env python
# coding: utf-8

# Título do script
# 2.1.Viabilidade de Projetos

# --- Bibliotecas necessárias ---
import pandas as pd  # Manipulação de dados com DataFrames
from sklearn.model_selection import train_test_split  # Separar dados em treino e teste
from sklearn.linear_model import LogisticRegression  # Modelo de Regressão Logística
from sklearn.preprocessing import StandardScaler  # Normalização de dados
from sklearn.metrics import classification_report  # Relatório de desempenho do modelo
import numpy as np  # Operações numéricas (não utilizado diretamente neste script)
import joblib  # Salvamento e carregamento de modelos treinados
import os  # Operações com o sistema de arquivos

# --- Função principal: treina o modelo ou faz previsões ---
def train_or_predict(new_projects):
    # Caminho onde o modelo será salvo ou carregado
    model_path = 'logistic_model.joblib'

    # Verifica se o modelo já foi salvo anteriormente
    if os.path.exists(model_path):
        # Carrega o modelo e o scaler do disco
        model = joblib.load(model_path)
        scaler = joblib.load(model_path.replace(".joblib", "_scaler.joblib"))
    else:
        # Carrega os dados do arquivo CSV
        df_projects = pd.read_csv("projects_data.csv")

        # Seleciona as variáveis preditoras (X)
        X = df_projects[["investment", "expected_return", "impact_score"]]
        # Seleciona a variável alvo (y)
        y = df_projects["viability"]

        # Normaliza os dados
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Divide os dados em conjunto de treino e teste
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.3, random_state=42
        )

        # Inicializa o modelo de Regressão Logística
        model = LogisticRegression()
        # Treina o modelo com os dados de treino
        model.fit(X_train, y_train)

        # Faz previsões com os dados de teste
        y_pred = model.predict(X_test)
        # Gera relatório de classificação (métricas)
        report = classification_report(y_test, y_pred, output_dict=True)

        # Salva o modelo treinado, o scaler e as métricas no disco
        joblib.dump(model, model_path)
        joblib.dump(scaler, model_path.replace(".joblib", "_scaler.joblib"))
        joblib.dump(report, model_path.replace(".joblib", "_metrics.joblib"))

    # Se houver novos projetos para prever a viabilidade
    if new_projects:
        # Cria um DataFrame com os novos projetos
        df_new_projects = pd.DataFrame(new_projects)

        # Aplica o mesmo processo de normalização dos dados
        X_new_scaled = scaler.transform(df_new_projects)

        # Faz a previsão de viabilidade (0 ou 1)
        predictions = model.predict(X_new_scaled)
        # Obtém a probabilidade da classe 1 (projeto viável)
        probabilities = model.predict_proba(X_new_scaled)[:, 1]

        # Adiciona colunas com a probabilidade e a previsão ao DataFrame
        df_new_projects["probability"] = probabilities
        df_new_projects["viability"] = predictions

        # Retorna o DataFrame com previsões e as métricas do modelo
        return df_new_projects, joblib.load(model_path.replace(".joblib", "_metrics.joblib"))

    # Caso não haja novos projetos, retorna apenas as métricas
    return None, joblib.load(model_path.replace(".joblib", "_metrics.joblib"))

# --- Teste do modelo com novos dados ---

# Lista de dicionários representando novos projetos
new_projects = [
    # {"investment": 13000, "expected_return": 69000, "impact_score": 7}  # Exemplo comentado
    {"investment": 40000, "expected_return": 60000, "impact_score": 6},  # Projeto a ser avaliado
    {"investment": 10000, "expected_return": 90000, "impact_score": 6},
    {"investment": 10000, "expected_return": 90000, "impact_score": 6}
    
]

# Executa a função de treino/previsão com os novos projetos
predictions, metrics = train_or_predict(new_projects)

# Exibe os resultados das previsões, se houver
if predictions is not None:
    print("\nNovos Projetos e Viabilidade:")
    print(predictions)

# Exibe as métricas do modelo
print("\nMétricas do Modelo:")
print(metrics)





