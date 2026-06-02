# 2. Arquitetura MMTT (Multi-Modal-Temporal-Tabular)

Para alimentar um Transformer com transações financeiras, precisamos resolver um problema crítico: como representar dados heterogêneos?

Uma transação contém:
*   **Modo Categórico**: Código do lojista (MCC), país, tipo de cartão.
*   **Modo Contínuo**: Valor da compra, saldo da conta.
*   **Modo Temporal**: Data, hora, segundos desde a última compra.

A abordagem **MMTT** (inspirada no *TransactionGPT* e no tokenizador *KVT* do PRAGMA) resolve isso de forma elegante.

## Tokenização KVT (Key-Value-Time)

Em vez de transformar tudo em texto, tratamos cada componente separadamente:

### 1. Key (Chave / Categoria)
*   **O que é**: IDs de lojas, MCCs (Merchant Category Codes).
*   **Como é tratado**: Mapeado para um dicionário (vocabulário) e passado por uma camada `nn.Embedding`. Semelhante ao que é feito com palavras em NLP.

### 2. Value (Valor Contínuo)
*   **O que é**: O valor financeiro da transação.
*   **Como é tratado**: É normalizado (Z-score) e projetado para a dimensão oculta do modelo através de uma camada linear `nn.Linear(1, hidden_size)`.

### 3. Time (Tempo)
*   **O que é**: O tempo exato em que a transação ocorreu.
*   **Como é tratado**: Em vez de usar timestamps absolutos (que o modelo não entende bem), usamos o **delta de tempo** (segundos desde a última transação). Esse valor é quantizado (agrupado em "caixas" ou *bins*) e passado por um `nn.Embedding`. Isso permite que o modelo entenda ritmos de compra (ex: várias compras em minutos = possível fraude ou viagem).

## Fusão Aditiva

Como juntamos Key, Value e Time?
A forma mais eficiente é a **Fusão Aditiva**. Em vez de concatenar os três vetores (o que triplicaria o tamanho da entrada e deixaria a atenção muito lenta), nós simplesmente somamos os três:

`Contexto = Embedding(Key) + Projeção(Value) + Embedding(Time)`

## Virtual Tokens (Prompts Contínuos)

Além do histórico transacional, um usuário tem **dados estáticos** (renda, idade, score de crédito, embeddings de grafo de relacionamento).

Colocar esses dados em *todas* as transações é redundante. A solução do TransactionGPT é usar uma **Virtual Token Layer**.

Nós criamos um pequeno número (ex: 2) de "tokens artificiais" e os colocamos no *início* da sequência do usuário. Durante o treinamento, a atenção do Transformer usa esses tokens virtuais como um "resumo global" do usuário. É como se estivéssemos dando um *prompt* inicial ao modelo antes de mostrar as transações.

---
**Próximo Passo**: Leia [03_pretraining_limix.md](03_pretraining_limix.md) para entender como ensinamos o modelo a aprender sem rótulos.
