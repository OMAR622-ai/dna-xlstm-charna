Masked-base prediction on 2,048 valid sequences of 1,024 bases (312,461 hidden bases, 15% masking, seed 0).

| Model | Step | Loss (nats) | Perplexity | Accuracy | Acc. A | Acc. C | Acc. G | Acc. T |
|---|---|---|---|---|---|---|---|---|
| GBST-BERT | 10000 | 0.9689 | 2.635 | 56.80% | 62.4% | 47.3% | 47.4% | 64.0% |
| Transformer++ | 10000 | 0.9728 | 2.645 | 56.65% | 62.1% | 47.6% | 47.2% | 63.7% |
| Random guess | – | 1.3863 | 4.000 | 25.00% | 25% | 25% | 25% | 25% |
| Always "T" (most common hidden base) | – | – | – | 29.97% | 0% | 0% | 0% | 100% |

Accuracy = the model's best guess among A, C, G, T is the true base. Base composition of the hidden positions: A 29.5%, C 20.3%, G 20.2%, T 30.0%.
