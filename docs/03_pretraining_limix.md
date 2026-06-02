# 3. Pré-treinamento com Masked Joint-Distribution (LimiX)

No repositório, o arquivo `src/training/objectives.py` implementa a classe `ContextConditionalMaskedLoss`. Esta é a nossa função de perda para o pré-treinamento.

## O Desafio do Aprendizado Autossupervisionado

Como ensinamos um modelo sem dizer a ele as respostas corretas (sem rótulos de fraude)?
Nós escondemos partes da pergunta e pedimos para ele adivinhar.

Existem duas abordagens principais:
1.  **Autoregressiva (Next Token Prediction)**: Usada no GPT. Dado T1, T2, T3... adivinhe o T4.
2.  **Bidirecional (Masked Modeling)**: Usada no BERT. Esconda tokens aleatórios e adivinhe usando o que vem antes E o que vem depois.

Nós usamos a abordagem **Bidirecional**, inspirada no modelo LimiX.

## Por que Bidirecional para Dados Financeiros?

Ao contrário da fala ou do texto, que são estritamente sequenciais (uma palavra após a outra), o comportamento financeiro é uma **distribuição conjunta** (Joint-Distribution).

Isso significa que a correlação não é apenas no tempo. Se eu sei que o usuário tem uma renda alta (dado estático) e gasta muito em passagens aéreas (categoria), eu posso inferir o valor gasto no restaurante de hoje — mesmo sem olhar a transação de ontem.

O objetivo do pré-treinamento é reconstruir a distribuição de dados mascarada.

## O Cronograma Heterogêneo de Máscaras (Heterogeneous Mask Schedule)

O LimiX introduziu um conceito poderoso: se você mascarar as coisas de forma puramente aleatória (como o BERT faz com texto), o modelo "trapaceia". Se eu esconder a transação 5, o modelo simplesmente copia o valor da transação 4 ou 6, sem aprender padrões profundos.

Para forçar o aprendizado real, usamos um cronograma heterogêneo (implementado no `objectives.py`):

1.  **Máscara de Célula (Cell Masking)**: Esconde features individuais aleatoriamente (~60%).
2.  **Máscara de Bloco (Block Masking)**: Esconde transações *consecutivas* (~30%). O modelo não pode mais copiar do vizinho; ele precisa entender a tendência de longo prazo.
3.  **Expansão Aleatória**: Adiciona ruído na máscara para garantir que os padrões não fiquem previsíveis (~10%).

## A Perda (Loss) de Reconstrução

Durante o treino, o modelo recebe as transações com ruído (zeros nas posições mascaradas) e tenta reconstruir a sequência original.

Nós calculamos o **MSE (Mean Squared Error)** apenas nas posições que foram mascaradas. O modelo é forçado a usar sua representação interna para preencher as lacunas.

Quando a `pretrain/val_loss` cai, significa que o modelo aprendeu a lógica por trás de "como e por que as pessoas gastam dinheiro".

---
**Próximo Passo**: Leia [04_finetuning_dcnv2.md](04_finetuning_dcnv2.md) para ver como usamos esse conhecimento para caçar fraudes.
