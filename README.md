# Classificador de Risco de Crédito — README

Este projeto implementa um classificador de risco de crédito usando Perceptron, com pipeline de pré-processamento, balanceamento de classes e ajuste de limiar, baseado no South German Credit Data.

---

## Funcionamento do Código

- **Carregamento dos Dados**
  - Os dados são lidos de uma tabela MySQL chamada `southgermancredit`.
  - As colunas são renomeadas para nomes descritivos e a variável alvo (`credit_risk`) é binarizada: 1 para "bom", 0 para "ruim".

- **Pré-processamento**
  - **Numéricas:** `duration`, `amount`, `age` são padronizadas.
  - **Ordinais:** Variáveis categóricas com ordem lógica são codificadas com `OrdinalEncoder`.
  - **Nominais:** `purpose` recebe `OneHotEncoder`.
  - Tudo é integrado via `ColumnTransformer`.

- **Balanceamento**
  - Usa-se SMOTE para gerar exemplos sintéticos da classe minoritária, equilibrando o dataset.

- **Treinamento**
  - O Perceptron é treinado nos dados balanceados.
  - Busca-se o melhor limiar de decisão (threshold) para maximizar o F1-score da classe minoritária.

- **Salvamento**
  - O modelo treinado, o pipeline de pré-processamento e o limiar ótimo são salvos em arquivos `.pkl`.

---

## Teoria

- **Perceptron:** Algoritmo de classificação linear que aprende um hiperplano separador entre duas classes.
- **SMOTE:** Técnica de oversampling para balancear classes desbalanceadas, criando exemplos sintéticos da classe minoritária.
- **Ajuste de Limiar:** O limiar padrão (0) pode não ser o ideal. O código busca o limiar que maximiza o F1-score para melhor sensibilidade à classe minoritária.

---

## Dicionário de Variáveis

| Coluna                | Descrição                                              | Códigos/Valores                                                                 |
|-----------------------|--------------------------------------------------------|---------------------------------------------------------------------------------|
| status                | Status da conta corrente                               | 1: sem conta, 2: <0 DM, 3: 0-200 DM, 4: >=200 DM/salário 1 ano                 |
| duration              | Duração do crédito (meses)                             | Numérico                                                                       |
| credit_history        | Histórico de crédito                                   | 0: atraso, 1: crítico, 2: sem crédito, 3: pagos até agora, 4: todos pagos      |
| purpose               | Propósito                                              | 0: outros, 1: carro novo, ..., 10: negócio                                     |
| amount                | Valor do crédito                                       | Numérico                                                                       |
| savings               | Poupança                                               | 1: desconhecido, 2: <100 DM, ..., 5: >=1000 DM                                 |
| employment_duration   | Duração do emprego                                     | 1: desempregado, 2: <1 ano, ..., 5: >=7 anos                                   |
| installment_rate      | Taxa de prestação                                      | 1: >=35, 2: 25-34, 3: 20-24, 4: <20                                            |
| personal_status_sex   | Estado civil/sexo                                      | 1: masc. divorciado, 2: fem. não solteira ou masc. solteiro, ...               |
| other_debtors         | Outros devedores                                       | 1: nenhum, 2: co-proponente, 3: fiador                                         |
| present_residence     | Tempo na residência atual                              | 1: <1 ano, 2: 1-3 anos, 3: 4-6 anos, 4: >=7 anos                               |
| property              | Propriedade                                            | 1: desconhecido, 2: carro, 3: seguro, 4: imóvel                                |
| age                   | Idade                                                  | Numérico                                                                       |
| other_installment_plans| Outros planos de pagamento                            | 1: banco, 2: lojas, 3: nenhum                                                  |
| housing               | Habitação                                              | 1: gratuito, 2: aluguel, 3: próprio                                            |
| number_credits        | Número de créditos                                     | 1: 1, 2: 2-3, 3: 4-5, 4: >=6                                                   |
| job                   | Profissão                                              | 1: desempregado, 2: não qualificado, 3: empregado qualificado, 4: gerente      |
| people_liable         | Pessoas dependentes                                    | 1: 3 ou mais, 2: 0 a 2                                                         |
| telephone             | Telefone                                               | 1: não, 2: sim                                                                 |
| foreign_worker        | Trabalhador estrangeiro                                | 1: sim, 2: não                                                                 |
| credit_risk           | Risco de crédito (alvo)                                | 0: ruim, 1: bom                                                                |

Veja o [codetable.txt](codetable.txt) para todos os detalhes dos códigos[2][3].

---

## Como usar

1. **Pré-requisitos:**  
   - MySQL com a tabela `southgermancredit` populada.
   - Bibliotecas Python: `pandas`, `sqlalchemy`, `scikit-learn`, `imblearn`, `joblib`, `numpy`, `pymysql`.

2. **Execução:**  
   - Execute o script Python.
   - Se não houver arquivos `.pkl`, o modelo será treinado automaticamente.
   - Para forçar novo treinamento, apague os arquivos `modelo_perceptron.pkl` e `preprocessador.pkl`.

---

## Observações

- O pipeline trata automaticamente tipos de variáveis e valores ausentes.
- O limiar de decisão é ajustado para maximizar o desempenho na detecção de maus pagadores.
- Para detalhes dos códigos de cada variável, consulte o arquivo `codetable.txt`[2][3].

---

## Referências

- South German Credit Data — [codetable.txt](codetable.txt)
- Documentação do Perceptron e SMOTE em scikit-learn e imblearn
