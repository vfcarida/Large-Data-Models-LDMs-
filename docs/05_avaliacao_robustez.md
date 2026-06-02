# 5. Avaliação e Robustez Estratégica

Treinar um modelo é fácil; garantir que ele gere valor real sem quebrar em produção é a parte difícil. Este módulo explica as ferramentas encontradas em `src/evaluation/`.

## Métricas para Classes Extremamente Desbalanceadas

No arquivo `metrics.py`, banimos o uso de "Acurácia Simples". Utilizamos:

### AUC (Area Under the ROC Curve)
Mede a capacidade geral do modelo de separar fraudadores de clientes bons, independente de qual limiar (corte) escolhamos.
*   0.5: O modelo está adivinhando aleatoriamente.
*   0.8+: Bom modelo.
*   1.0: Perfeito (suspeite de vazamento de dados).

### F1-Score com Limiar Flexível (Flexible Threshold)
O F1-Score é a média harmônica entre **Precision** e **Recall**.
*   **Precision (Precisão)**: "Dos alertas que o modelo gerou, quantos eram realmente fraude?" (Evita irritar clientes com falsos positivos).
*   **Recall (Revocação)**: "De todas as fraudes reais, quantas nós pegamos?" (Evita perder dinheiro).

Fornecemos um `threshold` flexível. Um limiar de 0.3 prioriza pegar mais fraudes (alto recall), enquanto 0.8 prioriza não bloquear clientes bons (alta precisão).

## Robustez e "Strategic Shift" (Manipulação Estratégica)

Diferente de identificar fotos de cães e gatos, no mundo financeiro, os fraudadores mudam ativamente seu comportamento para escapar dos modelos. Isso é testado em `strategic_shift_tester.py`.

Nós rodamos três simulações:

### 1. Desempenho Limpo (Baseline)
Como o modelo atua com os dados originais.

### 2. Random Shift (Ruído Natural)
Adicionamos ruído Gaussiano aos valores das transações. Isso simula o envelhecimento natural do modelo devido a inflação ou novas tendências de mercado. O modelo não deve quebrar com pequenas flutuações.

### 3. FGSM (Ataque Adversarial de Caixa Branca)
Usamos o *Fast Gradient Sign Method*. Nós computamos o gradiente do modelo em relação aos DADOS de entrada (e não aos pesos), e empurramos os valores na exata direção que maximiza o erro do modelo.
Isso simula o fraudador "perfeito", que sabe exatamente como contornar os filtros.

Se a Degradação de AUC for menor que 5% sob o FGSM, nosso modelo aprendeu a estrutura profunda do usuário (comportamento base), em vez de apenas decorar valores anômalos rasos.

## LoRA: Adaptação de Emergência

Quando uma nova onda de fraude ataca e os padrões mudam bruscamente, nós não temos tempo (e às vezes recurso computacional) para re-treinar todo o LDM do zero.

É aqui que entra o método `inject_lora_adaptation()`.
O LoRA (Low-Rank Adaptation) congela as matrizes gigantescas do Transformer original, e insere matrizes bem menores ao lado. O modelo treina apenas esses pequenos "remendos", que representam ~1% dos parâmetros originais.
Isso permite fine-tuning imediato (em horas ou minutos) em reposta a ataques, mantendo a inteligência pré-treinada intacta.
