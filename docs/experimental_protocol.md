# Experimental protocol

## Training schedule

- Memory: eight feed-forward blocks with RMSNorm and residual connections, a final RMSNorm and a vocabulary projection; hidden size 4096, intermediate size 14336. It reads the input of the final decoder block's feed-forward sublayer of the frozen `mistralai/Mistral-7B-v0.3`.
- Teacher: the frozen base reading the top-3 BM25 passages retrieved for each 64-token Wikipedia span; top-64 probabilities are cached and renormalized.
- Loss: 0.5 KL(teacher ‖ softmax(z_B + s)) + 0.5 CE over all valid positions; only the memory is trained.
- Optimizer: AdamW, learning rate 5e-4, 2,000 warmup steps, linear decay, global batch 1,024 positions. The knowledge-QA checkpoint is taken after one epoch over the December 2021 English Wikipedia.

## Checkpoint sources

- MLP Memory: the public checkpoint `Rubin-Wei/MLPMemory-Mistral-wikipedia`.
- Memory Decoder: the official pipeline, one epoch on full English Wikipedia.
- LoRA: rank 8, alpha 16 on all attention and feed-forward projections; strength 0.25.

## Scoring configuration

- Test sets: NQ, TriviaQA and WebQuestions (DPR test), HotpotQA (distractor validation), PopQA (all rows); 38,627 questions.
- Validation sets for coefficient selection, disjoint from every test question: DPR dev for NQ, TriviaQA and WebQuestions; HotpotQA train; EntityQuestions dev for PopQA.
- For each memory and dataset, the coefficient with the highest validation EM is selected (ties: F1, then the smaller coefficient) and used unchanged for EM, F1 and PPL on the test set.
- Generation: concise-answer prompt, 384 input tokens, at most 12 new tokens, greedy, BF16 base, batch 16. BM25-RAG: five passages, 4,096 input tokens, up to 64 new tokens.
- EM and F1 use normalized answer aliases. PPL covers every reference-answer token, excluding the prompt and EOS.
