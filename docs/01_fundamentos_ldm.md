# 1. Fundamentos: O que são Large Data Models (LDMs)?

Bem-vindo ao LDM Research Lab! Este documento introduz os conceitos básicos por trás dos Foundation Models para dados estruturados.

## Por que não usar apenas LLMs (Large Language Models)?

LLMs como GPT-4, Llama e Claude são fantásticos para texto, mas falham miseravelmente em dados estruturados transacionais e financeiros. O motivo?

1. **Falta de Precisão Numérica**: A tokenização de texto (BPE, WordPiece) divide números de forma imprevisível (ex: 1542.50 pode virar tokens [15], [42], [.], [50]). Isso destrói a relação matemática entre os números.
2. **Contexto Desnecessário**: Transformar uma tabela em texto (ex: "O usuário X gastou 50 no mercadinho Y no dia Z") adiciona muito ruído (palavras de ligação) e aumenta o tamanho da sequência, tornando a atenção do Transformer muito custosa.
3. **Distribuições Diferentes**: Dados financeiros possuem distribuições muito específicas (ex: quantias seguem leis log-normais, frequências de categorias seguem a Lei de Zipf) que não existem na linguagem natural.

## O que é um LDM?

Um **Large Data Model (LDM)** é uma arquitetura projetada *nativamente* para dados estruturados, séries temporais irregulares e logs transacionais.

Assim como um LLM é treinado em terabytes de texto da internet, um LDM é pré-treinado em bilhões de transações financeiras. O modelo aprende "como o dinheiro se move", capturando:

*   **Padrões de Gastos**: Como os gastos em diferentes categorias se relacionam.
*   **Sazonalidade**: Comportamentos de final de mês, feriados, horários comerciais.
*   **Estruturas de Correlação**: Qual a relação entre a idade, a renda e a propensão a pedir um empréstimo.

## O Paradigma "Pré-treinar e Fazer Fine-tuning"

A maior revolução dos Foundation Models é o uso de **Self-Supervised Learning (SSL)**, ou aprendizado autossupervisionado.

### Fase 1: Pré-treinamento (Sem Rótulos)
No ambiente financeiro, é difícil conseguir bons rótulos (labels). Fraude, por exemplo, afeta menos de 1% das transações e requer investigação manual para ser confirmada. No entanto, temos *bilhões* de transações não rotuladas.

Durante o pré-treinamento, o LDM usa técnicas como **Masked Joint-Distribution Modeling** (inspirado no LimiX) ou **Next Token Prediction** (inspirado no TransactionGPT). O modelo esconde partes dos próprios dados e tenta adivinhar o que está escondido.

*O que o modelo ganha com isso?* Uma representação interna (embeddings) profunda sobre o comportamento do usuário.

### Fase 2: Fine-Tuning (Com Rótulos)
Com o modelo pré-treinado entendendo profundamente as transações, nós adicionamos uma pequena "cabeça" (head) de classificação (no nosso caso, a rede **DCNv2**) e treinamos com os poucos dados rotulados que temos (ex: histórico de fraudes confirmadas).

Como o modelo já entende o que é um comportamento normal, ele precisa de muito menos exemplos para aprender o que é uma fraude.

---
**Próximo Passo**: Leia [02_arquitetura_mmtt.md](02_arquitetura_mmtt.md) para entender como convertemos transações brutas em tensores que o Transformer consegue processar.
