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
