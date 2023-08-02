import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('diabetes_dataset.csv')
#df.head()

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
#print(df)

features = ['Pregnancies',	'Glucose',	'BloodPressure',	'SkinThickness',	'Insulin',	'BMI',	'DiabetesPedigreeFunction',	'Age',	'Outcome']
df = df[features]

#separa as bases em dois grupos
outcome_0 = df[df['Outcome'] == 0]
outcome_1 = df[df['Outcome'] == 1]

#calcula medianas
median = outcome_0.median()
median1 = outcome_1.median()

#prenche os campos vazios com as medianas
outcome_0 = outcome_0.fillna(median)
#print(outcome_0)
outcome_1 = outcome_1.fillna(median1)
#print(outcome_1)

#juntando as duas tabelas novamente (saudáveis e doentes)
merge_tables= pd.concat([outcome_0, outcome_1])

#aqui tratamos os outliers
# Função para tratamento de outliers usando o IQR
# Função para tratamento de outliers usando a regra do desvio padrão
def treat_outliers(column):
    std_dev = column.std()
    mean_val = column.mean()
    lower_bound = mean_val - 2 * std_dev
    upper_bound = mean_val + 2 * std_dev
    return (column >= lower_bound) & (column <= upper_bound)

# Verifica e remove as linhas com outliers em cada coluna, exceto 'Age', 'Pregnancies' e 'Outcome'
#columns_to_exclude = ['Outcome']
#for column in merge_tables.columns:
    if column not in columns_to_exclude:
        merge_tables = merge_tables[treat_outliers(merge_tables[column])]
#merge_tables.info()
#print(merge_tables)

merge_tables.info()

# Separar as colunas "Age", "Pregnancies" e "Outcome"
#age_column = merge_tables['Age']
#pregnancies_column = merge_tables['Pregnancies']
outcome_column = merge_tables['Outcome']

# Excluir as colunas "Age", "Pregnancies" e "Outcome" antes da normalização
#merge_tables_normalized = merge_tables.drop(columns=['Age', 'Pregnancies', 'Outcome'])
merge_tables_normalized = merge_tables.drop(columns=['Outcome'])


# Normalizar todas as colunas, exceto as colunas "Age", "Pregnancies" e "Outcome"
merge_tables_normalized = (merge_tables_normalized - merge_tables_normalized.min()) / (merge_tables_normalized.max() - merge_tables_normalized.min())

# Adicionar as colunas "Age", "Pregnancies" e "Outcome" novamente ao DataFrame normalizado
#merge_tables_normalized.insert(len(merge_tables_normalized.columns), 'Age', age_column)
#merge_tables_normalized.insert(0, 'Pregnancies', pregnancies_column)
merge_tables_normalized.insert(len(merge_tables_normalized.columns), 'Outcome', outcome_column)

#print(merge_tables_normalized)
merge_tables_normalized.info()


merge_tables_normalized.to_csv("diabetes_dataset_normalized.csv", index=False)

#mude a key para true se quiser gerar graficos
key = False

columns_to_visualize = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
if(key):

    # Plotando os gráficos para cada coluna
    for column in columns_to_visualize:
        # Criando o gráfico
        plt.figure()
        plt.hist(merge_tables[column], bins=20)  # Você pode alterar o número de bins conforme desejado

        # Definindo título e rótulos dos eixos
        plt.xlabel(column)
        plt.ylabel('Contagem')
        plt.title(f'Histograma: {column}')

        # Salvando o gráfico em um arquivo (por exemplo, formato PNG)
        plt.savefig(f'histograma_{column}.png')

        # Fechando o gráfico para liberar memória
        plt.close()
    
