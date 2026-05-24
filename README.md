# WOE Policy Tree Classifier

Uma árvore de decisão explicável para problemas de crédito, risco e políticas de decisão, combinando **Weight of Evidence (WOE)**, restrições de negócio, monotonicidade e calibração isotônica nas folhas.

A proposta deste projeto não é criar mais uma árvore genérica de machine learning.  
A ideia é construir uma árvore que ajude a transformar modelos de risco em **regras claras, auditáveis e utilizáveis em políticas de crédito**.

---

## Por que este projeto existe?

Em crédito, nem sempre o melhor modelo é o mais complexo.

Modelos como XGBoost, LightGBM e redes neurais podem ter ótima performance preditiva, mas muitas vezes são difíceis de transformar em política de decisão.

Na prática, áreas de crédito, cobrança, validação, auditoria e governança precisam responder perguntas como:

- Por que este cliente foi aprovado?
- Por que este perfil foi negado?
- Este corte faz sentido do ponto de vista de negócio?
- A regra é estável?
- A política pode ser explicada para um comitê?
- A decisão pode ser implantada em SQL ou em um motor de decisão?

A `WOEPolicyTreeClassifier` foi criada para esse espaço:  
um modelo mais simples que um ensemble, mas muito mais alinhado com explicabilidade, governança e decisão.

---

## O que é a WOE Policy Tree?

A `WOEPolicyTreeClassifier` é uma árvore de decisão binária customizada para classificação, especialmente útil para risco de crédito.

Ela cria segmentos de risco usando regras do tipo:

```text
REGRA: atraso_medio <= 10 AND score > 650 | PD=4.30% | N=1500 | Bads=65
```

Cada folha da árvore representa um segmento com:

- regra de decisão;
- probabilidade estimada de inadimplência;
- quantidade de registros;
- quantidade de maus pagadores;
- interpretação direta para negócio.

---

## Principais recursos

- Critério de split baseado em WOE;
- Critério alternativo por diferença de bad rate;
- Critério híbrido combinando bad rate e WOE;
- Controle de volume mínimo por folha;
- Controle mínimo de bons e maus por folha;
- Diferença mínima de risco entre os lados do split;
- Restrições monotônicas por variável;
- Calibração isotônica das probabilidades finais das folhas;
- Impressão das regras finais da árvore;
- Interface parecida com estimadores do `scikit-learn`.

---

## Instalação

Clone o repositório:

```bash
git clone https://github.com/seu-usuario/woe-policy-tree.git
cd woe-policy-tree
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

Ou instale manualmente:

```bash
pip install numpy pandas scikit-learn
```

---

## Dependências

O projeto utiliza:

```text
numpy
pandas
scikit-learn
```

Exemplo de `requirements.txt`:

```text
numpy>=1.23.0
pandas>=1.5.0
scikit-learn>=1.2.0
```

---

## Exemplo rápido de uso

```python
from woe_policy_tree import WOEPolicyTreeClassifier

model = WOEPolicyTreeClassifier(
    max_depth=3,
    min_samples_leaf=500,
    min_bad_leaf=30,
    min_good_leaf=30,
    max_bins=10,
    min_bad_rate_diff=0.02,
    criterion="woe_gain",
    monotonic=True,
    monotonic_constraints={
        "atraso_medio": 1,
        "score_credito": -1,
        "comprometimento_renda": 1
    },
    leaf_monotonic=True,
    leaf_monotonic_direction="increasing",
    leaf_sort_method="empirical_bad_rate"
)

model.fit(X_train, y_train)

model.print_rules()
```

---

## Predição

Para obter probabilidades:

```python
probas = model.predict_proba(X_test)

pd_inadimplencia = probas[:, 1]
```

Para obter a classe final:

```python
pred = model.predict(X_test, threshold=0.10)
```

---

## Exemplo de saída

```text
REGRA: atraso_medio <= 10.0000 AND score_credito > 650.0000 | PD=4.30% | N=1500 | Bads=65
REGRA: atraso_medio <= 10.0000 AND score_credito <= 650.0000 | PD=9.10% | N=800 | Bads=73
REGRA: atraso_medio > 10.0000 AND comprometimento_renda <= 0.3500 | PD=14.20% | N=600 | Bads=85
REGRA: atraso_medio > 10.0000 AND comprometimento_renda > 0.3500 | PD=28.70% | N=350 | Bads=100
```

Essas regras podem ser usadas para:

- política de crédito;
- classificação de risco;
- estratégia de cobrança;
- política de limite;
- segmentação IFRS 9 / CPC 48;
- documentação de modelo;
- implantação em motor de decisão.

---

## Critérios de split disponíveis

A árvore suporta três critérios principais.

### 1. `woe_gain`

Usa a diferença de Weight of Evidence entre os dois lados do corte.

```python
criterion="woe_gain"
```

Esse é o critério mais alinhado com práticas tradicionais de crédito e scorecard.

---

### 2. `bad_rate_gain`

Usa a diferença absoluta de taxa de inadimplência entre os dois lados do corte.

```python
criterion="bad_rate_gain"
```

É simples, direto e fácil de interpretar.

---

### 3. `hybrid`

Combina diferença de bad rate com diferença de WOE.

```python
criterion="hybrid"
```

Pode ser útil quando se deseja equilibrar separação estatística e interpretação de risco.

---

## Monotonicidade por variável

Em crédito, muitas variáveis têm uma direção esperada de risco.

Exemplos:

| Variável | Direção esperada |
|---|---|
| Atraso médio | Quanto maior, maior o risco |
| Comprometimento de renda | Quanto maior, maior o risco |
| Score de crédito | Quanto maior, menor o risco |
| Tempo de relacionamento | Quanto maior, menor o risco |

A árvore permite informar essas restrições:

```python
monotonic=True,
monotonic_constraints={
    "atraso_medio": 1,
    "score_credito": -1,
    "comprometimento_renda": 1
}
```

A interpretação é:

| Valor | Significado |
|---:|---|
| `1` | relação crescente com risco |
| `-1` | relação decrescente com risco |
| `0` | sem restrição |

Exemplo:

```python
monotonic_constraints={
    "atraso_medio": 1
}
```

Nesse caso, a árvore só aceita cortes em que o lado com maior atraso tenha bad rate maior ou igual ao lado com menor atraso.

---

## Monotonicidade nas folhas

Além da monotonicidade nos splits, também é possível calibrar as probabilidades finais das folhas usando regressão isotônica.

```python
leaf_monotonic=True,
leaf_monotonic_direction="increasing"
```

Isso ajusta as PDs finais para respeitarem uma ordem crescente ou decrescente.

Métodos disponíveis para ordenar as folhas:

```python
leaf_sort_method="tree_order"
leaf_sort_method="proba_bad"
leaf_sort_method="empirical_bad_rate"
```

### `tree_order`

Mantém a ordem natural da árvore, da esquerda para a direita.

### `proba_bad`

Ordena as folhas pela probabilidade estimada de inadimplência.

### `empirical_bad_rate`

Ordena as folhas pela taxa empírica de inadimplência observada.

---

## Parâmetros principais

| Parâmetro | Descrição | Default |
|---|---|---:|
| `max_depth` | Profundidade máxima da árvore | `3` |
| `min_samples_leaf` | Mínimo de registros por folha | `500` |
| `min_bad_leaf` | Mínimo de maus por folha | `30` |
| `min_good_leaf` | Mínimo de bons por folha | `30` |
| `max_bins` | Número máximo de bins para candidatos de corte | `10` |
| `min_bad_rate_diff` | Diferença mínima de bad rate entre os lados do split | `0.02` |
| `criterion` | Critério de escolha do split | `"woe_gain"` |
| `monotonic` | Ativa restrições monotônicas nos splits | `False` |
| `monotonic_constraints` | Dicionário com direção esperada por variável | `None` |
| `leaf_monotonic` | Ativa calibração isotônica nas folhas | `False` |
| `leaf_monotonic_direction` | Direção da monotonicidade final | `"increasing"` |
| `leaf_sort_method` | Método de ordenação das folhas | `"empirical_bad_rate"` |
| `random_state` | Semente de aleatoriedade | `42` |

---

## Exemplo com dados sintéticos

```python
import numpy as np
import pandas as pd

from woe_policy_tree import WOEPolicyTreeClassifier

np.random.seed(42)

n = 10000

X = pd.DataFrame({
    "score_credito": np.random.normal(650, 80, n),
    "atraso_medio": np.random.exponential(5, n),
    "comprometimento_renda": np.random.beta(2, 5, n)
})

logit = (
    -3
    - 0.004 * (X["score_credito"] - 650)
    + 0.08 * X["atraso_medio"]
    + 2.5 * X["comprometimento_renda"]
)

prob = 1 / (1 + np.exp(-logit))

y = np.random.binomial(1, prob)

model = WOEPolicyTreeClassifier(
    max_depth=3,
    min_samples_leaf=300,
    min_bad_leaf=20,
    min_good_leaf=20,
    criterion="woe_gain",
    monotonic=True,
    monotonic_constraints={
        "score_credito": -1,
        "atraso_medio": 1,
        "comprometimento_renda": 1
    },
    leaf_monotonic=True,
    leaf_monotonic_direction="increasing"
)

model.fit(X, y)

model.print_rules()
```

---

## Quando usar

Este modelo pode ser útil quando você precisa de:

- regras claras de decisão;
- segmentação de risco;
- explicabilidade para negócio;
- documentação para validação;
- política de crédito auditável;
- modelo simples de implantar;
- alternativa interpretável a modelos caixa-preta;
- árvore mais alinhada com práticas de scorecard.

---

## Possíveis aplicações

### Crédito

- concessão;
- renovação;
- aumento ou redução de limite;
- pré-aprovação;
- segmentação de risco.

### Cobrança

- priorização de acionamento;
- definição de estratégia por perfil;
- régua de cobrança;
- segmentação de atraso.

### IFRS 9 / CPC 48

- agrupamento de contratos;
- apoio à classificação de stage;
- segmentação para PD;
- explicabilidade de provisão;
- análise de estabilidade por safra.

### LGD

A lógica pode ser adaptada para modelagem de perda dado default, criando segmentos de severidade de perda.

---

## O que este modelo não tenta ser

Este projeto não tem como objetivo substituir modelos mais complexos em todos os cenários.

Em geral, modelos como XGBoost e LightGBM podem ter maior performance preditiva em termos de AUC, KS ou log loss.

A proposta aqui é diferente.

A `WOEPolicyTreeClassifier` foi pensada para contextos em que explicabilidade, governança e implantação são tão importantes quanto performance.

---

## Limitações atuais

A versão atual ainda possui alguns pontos de evolução:

- não possui tratamento nativo específico para valores nulos;
- ainda não exporta automaticamente regras para SQL;
- não calcula métricas automáticas por folha;
- não possui pruning pós-treinamento;
- não possui validação temporal por safra;
- não possui relatório automático de estabilidade.

---

## Roadmap

Ideias para próximas versões:

- [ ] Tratamento explícito de missing values;
- [ ] Exportação automática para SQL `CASE WHEN`;
- [ ] Exportação para JSON;
- [ ] Relatório de folhas com bad rate treino/teste;
- [ ] PSI por folha;
- [ ] Validação temporal por safra;
- [ ] Pruning por ganho mínimo;
- [ ] Suporte para regressão;
- [ ] Adaptação para LGD;
- [ ] Visualização gráfica da árvore;
- [ ] Compatibilidade maior com API do `scikit-learn`.

---

## Exemplo de exportação desejada para SQL

Uma evolução futura seria gerar automaticamente regras como:

```sql
CASE
    WHEN atraso_medio <= 10 AND score_credito > 650 THEN 0.043
    WHEN atraso_medio <= 10 AND score_credito <= 650 THEN 0.091
    WHEN atraso_medio > 10 AND comprometimento_renda <= 0.35 THEN 0.142
    ELSE 0.287
END AS pd_segmentada
```

Isso permitiria implantar a política diretamente em bancos de dados, motores de decisão ou pipelines produtivos.

---

## Estrutura sugerida do repositório

```text
woe-policy-tree/
│
├── README.md
├── requirements.txt
├── LICENSE
│
├── woe_policy_tree/
│   ├── __init__.py
│   └── classifier.py
│
├── examples/
│   ├── example_synthetic_data.py
│   └── example_credit_policy.ipynb
│
└── tests/
    └── test_classifier.py
```

---

## Como importar

Caso use a estrutura acima, o arquivo `woe_policy_tree/__init__.py` pode conter:

```python
from .classifier import WOEPolicyTreeClassifier
```

Assim, o uso fica simples:

```python
from woe_policy_tree import WOEPolicyTreeClassifier
```

---

## Licença

Sugestão: MIT License.

```text
MIT License

Copyright (c) 2026

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files, to deal in the Software
without restriction, including without limitation the rights to use, copy,
modify, merge, publish, distribute, sublicense, and/or sell copies of the Software.
```

---

## Contribuições

Contribuições são bem-vindas.

Algumas ideias úteis:

- melhorar tratamento de missing;
- adicionar exportação para SQL;
- criar visualização da árvore;
- adicionar métricas por folha;
- adaptar para regressão e LGD;
- criar exemplos com bases públicas de crédito.

---

## Referências conceituais

Este projeto se inspira em práticas comuns de:

- scorecard de crédito;
- Weight of Evidence;
- árvores de decisão interpretáveis;
- monotonicidade em modelos de risco;
- calibração isotônica;
- governança e validação de modelos financeiros.

---

## Motivação final

Em crédito, prever bem é importante.  
Mas transformar previsão em decisão é outra coisa.

A `WOEPolicyTreeClassifier` foi criada para ajudar nesse segundo problema: criar regras de risco que sejam simples de explicar, fáceis de documentar e possíveis de implantar.

Nem sempre o melhor modelo é o mais sofisticado.

Às vezes, é o que as pessoas conseguem usar.
