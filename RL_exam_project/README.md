# Reinforcement Learning for Robust Text-to-Cypher Schema Selection on Hetionet

This repository contains an exam project for Reinforcement Learning. The project studies a robust Text-to-Cypher pipeline for the Hetionet biomedical graph database.

The agent controls intermediate decisions instead of generating the full Cypher query token by token:

1. query rewriting strategy selection;
2. hop-count / query-complexity selection;
3. schema-path selection over Hetionet relations.

## Main Files

- `final_hetionet_rl_project.ipynb`: clean notebook for running and explaining the final project.
- `hetionet_rl_llm_pipeline.py`: helper code, Gymnasium environments, rewriting utilities, training and evaluation functions.
- `HETIONET_dataset.json`: dataset used in the experiments.
- `final_results.json`: final metrics.
- `requirements.txt`: Python dependencies.
- `llm_cache/safe_qwen15_fewshot_original_train_rewrite_cache.json`: cached Qwen 1.5B rewrites for training examples.
- `llm_cache/safe_qwen15_fewshot_80_rewrite_cache.json`: cached Qwen 1.5B rewrites for the test subset.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then open:

```bash
jupyter notebook final_hetionet_rl_project.ipynb
```

## Final Result

The final model uses:

- Qwen 1.5B few-shot rewriting;
- hierarchical Gymnasium environment;
- schema-path actions;
- MiniLM embeddings plus schema cue features;
- lexical action masking.

Final evaluation on the paraphrased Qwen subset:

```text
Overall exact match: 0.45
Overall F1:          0.625

1-hop exact match:   0.70
1-hop F1:            0.767

2-hop exact match:   0.367
2-hop F1:            0.578
```

## Notes

The Qwen rewrite outputs are cached, so the notebook can be executed without regenerating LLM rewrites. MiniLM embeddings are regenerated at runtime to keep the repository lightweight.
