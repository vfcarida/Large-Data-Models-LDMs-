# 4. Fine-Tuning e Redes DCNv2

Após o pré-treinamento, temos um Transformer `MMTTTransformerEncoder` que entende perfeitamente os históricos de transação. Mas a vida real das instituições financeiras não é feita apenas de logs transacionais.

Temos dados tabulares clássicos (Features Estáticas):
*   Score de Bureau (Ex: Serasa)
*   Idade, Renda declarada
*   Risco do CEP
*   Features criadas manualmente por engenheiros de dados

Como combinar o supercérebro do Transformer com as features clássicas?

## A Armadilha da Concatenação Simples

A abordagem ingênua é pegar o Embedding gerado pelo Transformer, concatenar (colar lado a lado) com as features clássicas, e jogar em uma rede neural profunda (MLP - Multi-Layer Perceptron).

O problema: um MLP aprende interações **implícitas** e precisa de muitas camadas (e muitos dados rotulados) para descobrir que "Score baixo" + "Compra rápida de eletrônicos" = Alto risco.

## A Solução: Deep & Cross Network v2 (DCNv2)

O DCNv2, introduzido pelo Google, é o estado da arte para Recomendação e CTR (Click-Through Rate), e se encaixa perfeitamente aqui.

Em vez de apenas usar camadas ocultas tradicionais, a `EndToEndFusionLayer` usa uma arquitetura paralela:

### 1. Cross Network (Interações Explícitas)
O Cross Network força as features originais (a concatenação do embedding com os dados tabulares) a se multiplicarem de forma polinomial.

Matematicamente: Cada camada cruza a saída atual com a **entrada original**.
*   Camada 1 aprende interações de grau 2 (Feature A * Feature B).
*   Camada 2 aprende interações de grau 3 (A * B * C).

Isso garante que interações cruciais entre os dados do Transformer e os dados do Bureau de crédito sejam aprendidas explicitamente e rapidamente.

### 2. Deep Network (Interações Implícitas)
Em paralelo, usamos um MLP padrão (com camadas densas, GELU, e Dropout) para aprender padrões não-lineares complexos que as multiplicações de features podem deixar escapar.

### A Fusão Final
As saídas da Cross Network e da Deep Network são concatenadas e projetadas para um único número: o **Logit** (a pontuação crua de probabilidade de fraude).

## Loss Ponderada para Classes Desbalanceadas

No arquivo `lightning_module.py`, observe que usamos `BCEWithLogitsLoss(pos_weight=X)`.

Em problemas como Fraude ou Inadimplência, temos 99% da classe 0 (Normal) e 1% da classe 1 (Fraude).
Se não usarmos o `pos_weight`, o modelo vai rapidamente aprender a dizer "Tudo é normal", alcançando 99% de acurácia, mas tendo zero utilidade prática.

O `pos_weight` multiplica o erro quando o modelo erra uma fraude. É como dizer ao modelo: "Errar uma transação normal custa 1 real. Mas errar uma fraude custa 10 reais". Isso força o modelo a caçar as anomalias ativamente.

---
**Próximo Passo**: Leia [05_avaliacao_robustez.md](05_avaliacao_robustez.md) para entender por que Acurácia é uma métrica perigosa e como medimos a robustez.
