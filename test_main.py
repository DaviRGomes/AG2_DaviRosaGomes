import pytest
import joblib
import pandas as pd

@pytest.fixture(scope="module")
def modelo_e_preprocessador():
    modelo_data = joblib.load("modelo_perceptron.pkl")
    preprocessor = joblib.load("preprocessador.pkl")
    return modelo_data, preprocessor

def test_classificar_bom(modelo_e_preprocessador):
    modelo_data, preprocessor = modelo_e_preprocessador
    # Exemplo de dados típicos de bom pagador
    new_data = pd.DataFrame({
        'duration': [12], 'amount': [1000], 'age': [35],
        'status': [1], 'credit_history': [4], 'savings': [5],
        'employment_duration': [5], 'installment_rate': [1],
        'personal_status_sex': [1], 'other_debtors': [1],
        'present_residence': [4], 'property': [2],
        'other_installment_plans': [1], 'housing': [1],
        'number_credits': [1], 'job': [2], 'people_liable': [1],
        'telephone': [1], 'foreign_worker': [1], 'purpose': [0]
    })
    X = preprocessor.transform(new_data)
    score = modelo_data['modelo'].decision_function(X)
    prediction = (score > modelo_data['limiar']).astype(int)
    assert prediction[0] == 1  # Espera-se "bom"

def test_classificar_ruim(modelo_e_preprocessador):
    modelo_data, preprocessor = modelo_e_preprocessador
    # Exemplo de dados típicos de mau pagador
    new_data = pd.DataFrame({
        'duration': [48], 'amount': [10000], 'age': [20],
        'status': [4], 'credit_history': [0], 'savings': [1],
        'employment_duration': [1], 'installment_rate': [4],
        'personal_status_sex': [4], 'other_debtors': [3],
        'present_residence': [1], 'property': [4],
        'other_installment_plans': [3], 'housing': [3],
        'number_credits': [4], 'job': [4], 'people_liable': [2],
        'telephone': [2], 'foreign_worker': [2], 'purpose': [10]
    })
    X = preprocessor.transform(new_data)
    score = modelo_data['modelo'].decision_function(X)
    prediction = (score > modelo_data['limiar']).astype(int)
    assert prediction[0] == 0  # Espera-se "ruim"
