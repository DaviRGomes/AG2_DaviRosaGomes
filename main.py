import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE  # NOVO: para balanceamento

# 1. Conexão com o banco de dados e carregamento dos dados
host = 'localhost:3306' 
user = 'root'
password = '1212'
database = 'ag2_germancard'

connection_string = f"mysql+pymysql://{user}:{password}@{host}/{database}"
engine = create_engine(connection_string)
query = "SELECT * FROM southgermancredit"

df = pd.read_sql(query, engine)

# 2. Mapeamento dos nomes das colunas para inglês
column_mapping = {
    'laufkont': 'status',
    'laufzeit': 'duration',
    'moral': 'credit_history',
    'verw': 'purpose',
    'hoehe': 'amount',
    'sparkont': 'savings',
    'beszeit': 'employment_duration',
    'rate': 'installment_rate',
    'famges': 'personal_status_sex',
    'buerge': 'other_debtors',
    'wohnzeit': 'present_residence',
    'verm': 'property',
    'alter': 'age',
    'weitkred': 'other_installment_plans',
    'wohn': 'housing',
    'bishkred': 'number_credits',
    'beruf': 'job',
    'pers': 'people_liable',
    'telef': 'telephone',
    'gastarb': 'foreign_worker',
    'kredit': 'credit_risk'
}
df = df.rename(columns=column_mapping)

# 3. Pré-processamento dos dados
df['credit_risk'] = df['credit_risk'].replace({1: 1, 2: 0})

# Definir tipos de variáveis conforme documentação
numeric_features = ['duration', 'amount', 'age']
ordinal_features = [
    'status', 'credit_history', 'savings', 'employment_duration',
    'installment_rate', 'personal_status_sex', 'other_debtors',
    'present_residence', 'property', 'other_installment_plans',
    'housing', 'number_credits', 'job', 'people_liable',
    'telephone', 'foreign_worker'
]
nominal_features = ['purpose']  # 'purpose' é nominal (sem ordem)

# Pipeline de pré-processamento melhorado
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('ord', OrdinalEncoder(), ordinal_features),
        ('nom', OneHotEncoder(handle_unknown='ignore'), nominal_features)
    ]
)

# 4. Divisão dos dados
X = df.drop('credit_risk', axis=1)
y = df['credit_risk']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# 5. Balanceamento com SMOTE no treino
X_train_transf = preprocessor.fit_transform(X_train)
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_transf, y_train)

# 6. Treinar o modelo Perceptron com dados balanceados
perceptron = Perceptron(
    random_state=42,
    max_iter=2000,
    eta0=0.01
)
perceptron.fit(X_train_bal, y_train_bal)

# 7. Avaliação (com limiar ajustado para favorecer "BOM")
X_test_transf = preprocessor.transform(X_test)
y_scores = perceptron.decision_function(X_test_transf)

# Ajuste do limiar: favorece "BOM" (good, classe 1)
limiar = -0.3  # Experimente valores entre -1 e 0.5 conforme o recall desejado
y_pred = (y_scores > limiar).astype(int)

print("\nMétricas de Avaliação (com limiar ajustado):")
print(f"Acurácia: {accuracy_score(y_test, y_pred):.2f}")
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))
print("\nMatriz de Confusão:")
print(confusion_matrix(y_test, y_pred))

# 8. Função para classificação de novos dados
def get_valid_input(prompt, valid_values=None, value_type=int):
    while True:
        try:
            value = value_type(input(prompt))
            if valid_values is None or value in valid_values:
                return value
            else:
                print(f"Valor inválido! Deve ser um desses: {valid_values}")
        except ValueError:
            print("Por favor, insira um valor válido!")

def classificar_novo_dado():
    print("\nInsira os dados para classificação:")

    # Dados numéricos
    duration = get_valid_input("Duração do crédito (meses) [duration]: ")
    amount = get_valid_input("Valor do crédito [amount]: ")
    age = get_valid_input("Idade [age]: ")

    # Dados ordinais conforme documentação
    status = get_valid_input("Status da conta [status] (1-4): ", [1,2,3,4])
    credit_history = get_valid_input("Histórico de crédito [credit_history] (0-4): ", [0,1,2,3,4])
    savings = get_valid_input("Poupança [savings] (1-5): ", [1,2,3,4,5])
    employment_duration = get_valid_input("Duração do emprego [employment_duration] (1-5): ", [1,2,3,4,5])
    installment_rate = get_valid_input("Taxa de prestação [installment_rate] (1-4): ", [1,2,3,4])
    personal_status_sex = get_valid_input("Estado pessoal/sexo [personal_status_sex] (1-4): ", [1,2,3,4])
    other_debtors = get_valid_input("Outros devedores [other_debtors] (1-3): ", [1,2,3])
    present_residence = get_valid_input("Tempo na residência atual [present_residence] (1-4): ", [1,2,3,4])
    property = get_valid_input("Propriedade [property] (1-4): ", [1,2,3,4])
    other_installment_plans = get_valid_input("Outros planos de pagamento [other_installment_plans] (1-3): ", [1,2,3])
    housing = get_valid_input("Habitação [housing] (1-3): ", [1,2,3])
    number_credits = get_valid_input("Número de créditos [number_credits] (1-4): ", [1,2,3,4])
    job = get_valid_input("Emprego [job] (1-4): ", [1,2,3,4])
    people_liable = get_valid_input("Pessoas dependentes [people_liable] (1-2): ", [1,2])
    telephone = get_valid_input("Telefone [telephone] (1-2): ", [1,2])
    foreign_worker = get_valid_input("Trabalhador estrangeiro [foreign_worker] (1-2): ", [1,2])

    # Dados nominais
    purpose = get_valid_input("Propósito [purpose] (0-10): ", list(range(11)))

    # Criar DataFrame com o novo dado
    new_data = pd.DataFrame({
        'duration': [duration],
        'amount': [amount],
        'age': [age],
        'status': [status],
        'credit_history': [credit_history],
        'savings': [savings],
        'employment_duration': [employment_duration],
        'installment_rate': [installment_rate],
        'personal_status_sex': [personal_status_sex],
        'other_debtors': [other_debtors],
        'present_residence': [present_residence],
        'property': [property],
        'other_installment_plans': [other_installment_plans],
        'housing': [housing],
        'number_credits': [number_credits],
        'job': [job],
        'people_liable': [people_liable],
        'telephone': [telephone],
        'foreign_worker': [foreign_worker],
        'purpose': [purpose]
    })

    # Pré-processa e faz a previsão
    new_data_transf = preprocessor.transform(new_data)
    score = perceptron.decision_function(new_data_transf)
    prediction = (score > limiar).astype(int)

    print("\nResultado da Classificação:")
    print(f"Risco de Crédito: {'Bom' if prediction[0] == 1 else 'Ruim'}")
    print(f"Score (distância da fronteira de decisão): {score[0]:.2f}")

# 9. Menu interativo
while True:
    print("\nMenu:")
    print("1. Classificar novo dado")
    print("2. Sair")

    option = input("Escolha uma opção: ")

    if option == '1':
        classificar_novo_dado()
    elif option == '2':
        print("Saindo do programa...")
        break
    else:
        print("Opção inválida! Tente novamente.")
