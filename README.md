## 1. Aya Expanse 32B (Ranked 1st)

Aya Expanse outperforms the other models for several reasons:

Highest CoT EM and F1 scores: With 83.0 for EM and 87.0 for F1, it shows strong reasoning abilities, capturing both accuracy and partial matches.

Best Retrieval Hit Ratio: At 93.2%, Aya Expanse retrieves the most relevant context in comparison to the others. This implies better alignment between the model’s reasoning and the retrieved context.

Lowest Latency: Aya Expanse achieves a latency of 1.7 seconds, which means it provides faster results compared to the others. In real-time applications, this speed gives Aya Expanse a distinct advantage.

Top Contextual Chunking performance: It excels in both Contextual Chunking EM (84.6) and Contextual Chunking F1 (88.5), suggesting it is highly efficient at understanding context and providing relevant answers.

Strong Retrieval Precision and Recall: Aya Expanse achieves 92.0% precision and 93.4% recall, indicating it retrieves highly relevant information while maintaining a broad retrieval scope.

Best Query Efficiency Ratio: At 0.8, this shows Aya Expanse is more efficient in query rewriting and retrieval optimization, making it a highly adaptable model.


2. ChatGPT-4.0 (Ranked 2nd)

ChatGPT-4.0 follows closely behind:

Strong CoT EM and F1: With 82.1 EM and 86.4 F1, ChatGPT-4.0 performs excellently in reasoning tasks, though slightly lower than Aya Expanse.

High Retrieval Hit Ratio: At 91.2%, its retrieval capabilities are robust, but not as good as Aya Expanse’s.

Fast Latency: It performs in 1.9 seconds, which is very efficient but not the fastest.

High Contextual Chunking Scores: The Contextual Chunking EM of 83.7 and Contextual Chunking F1 of 87.5 are competitive, showcasing ChatGPT-4.0’s strong comprehension and contextual understanding.

Good Retrieval Precision and Recall: Achieving 88.9% precision and 90.1% recall shows ChatGPT-4.0’s competence in retrieving relevant content.

Query Efficiency Ratio: At 0.9, ChatGPT-4.0’s query efficiency is very good but slightly less optimal compared to Aya Expanse.


3. Mamba2 Model (Ranked 3rd)

The Mamba2 Model ranks third for the following reasons:

Solid CoT Performance: With CoT EM of 80.5 and CoT F1 of 84.8, it performs well but not as effectively as the top two models.

Strong Retrieval Hit Ratio: At 89.6%, Mamba2 performs well in retrieving relevant contexts, though it doesn’t match the best-performing models in retrieval.

Reasonable Latency: The model achieves a latency of 2.0 seconds, making it relatively fast but slower than Aya Expanse.

Contextual Chunking EM (82.4) and F1 (86.2): The Mamba2 model performs well in handling contextual information, but it lags behind both Aya Expanse and ChatGPT-4.0 in terms of accuracy and chunking efficiency.

Good Retrieval Precision and Recall: Mamba2 achieves 86.7% precision and 88.5% recall, showing balanced retrieval capabilities.

Query Efficiency Ratio: With a ratio of 1.0, the model is relatively efficient in query management but is not as optimal as Aya Expanse or ChatGPT-4.0.


4. Claude Sonnet 3.5 (Ranked 4th)

Claude Sonnet 3.5 places fourth due to:

CoT Performance: CoT EM of 79.3 and CoT F1 of 83.2 are strong, but they lag behind the other models, especially Aya Expanse and ChatGPT-4.0.

Good Retrieval Hit Ratio: At 88.9%, it performs well but falls behind the top contenders in terms of relevant context retrieval.

Decent Latency: With a latency of 2.1 seconds, it is slightly slower compared to other models in this ranking.

Contextual Chunking: The EM of 81.0 and F1 of 85.5 suggest it handles contextual information effectively, but there’s room for improvement compared to the leaders.

Retrieval Precision and Recall: It achieves 85.1% precision and 87.2% recall, performing adequately but not as robustly as the other models.

Query Efficiency Ratio: Its efficiency ratio of 1.1 shows that it is slightly less optimized in terms of query rewriting and retrieval optimization.


5. Llama 3 70B (Ranked 5th)

Llama 3 70B ranks last among the models primarily due to:

CoT Performance: With CoT EM of 78.4 and CoT F1 of 82.1, it lags behind Aya Expanse and other models in terms of reasoning and match quality.

Retrieval Hit Ratio: At 87.5%, its context retrieval is the least effective, meaning it has a lower alignment with the necessary external knowledge.

Latency: Llama 3 70B takes 2.3 seconds on average, which is reasonable but not as fast as Aya Expanse and ChatGPT-4.0.

Contextual Chunking EM (80.2) and F1 (84.3): Llama 3’s performance in these metrics is good but falls short compared to the more advanced models.

Retrieval Precision and Recall: It achieves 84.6% precision and 85.9% recall, both of which are the lowest in this group, indicating it doesn’t perform as well in retrieving relevant information.

Query Efficiency Ratio: Its efficiency ratio of 1.2 shows a higher level of inefficiency compared to the other models, indicating a less optimized approach to query rewriting and retrieval.





________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________
## Hybrid-ARAG



1-Implement Query Re-wrtiing  (Done)

2-Implement Contextual Chunking (Wip)

3-Implement COT (Done)

4-Implement Agentic Graph if Possible (Wip)

5-Integrate the usage of mamba2 models (Done)

________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________

## used models so far :

1-llama 3.1 70B

2-falcon mamba 2 & codestral mamba 

3-Aya Expanse 32B (best so far)

4-Gpt 4o (Judge/Baseline)

5- llama 3.3 70b

6-Qwen2.5 qwq
________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________

