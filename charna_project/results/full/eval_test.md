Masked-base prediction on 2,048 test sequences of 1,024 bases (311,767 hidden bases, 15% masking, seed 0).

| Model | Step | Loss (nats) | Perplexity | Accuracy | Acc. A | Acc. C | Acc. G | Acc. T |
|---|---|---|---|---|---|---|---|---|
| GBST-BERT | 10000 | 0.9753 | 2.652 | 56.44% | 61.8% | 48.5% | 48.8% | 62.1% |
| Transformer++ | 10000 | 0.9774 | 2.657 | 56.33% | 61.3% | 48.9% | 48.9% | 61.9% |
| Random guess | – | 1.3863 | 4.000 | 25.00% | 25% | 25% | 25% | 25% |
| Always "A" (most common hidden base) | – | – | – | 29.42% | 100% | 0% | 0% | 0% |

Accuracy = the model's best guess among A, C, G, T is the true base. Base composition of the hidden positions: A 29.4%, C 20.6%, G 20.8%, T 29.2%.
