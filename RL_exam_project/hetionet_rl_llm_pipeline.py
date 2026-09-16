import json
import pickle
import random
import re
from collections import Counter
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces


FULL_SCHEMA_PATTERNS = [
    ("Anatomy", "DOWNREGULATES_AdG", "Gene"),
    ("Anatomy", "EXPRESSES_AeG", "Gene"),
    ("Anatomy", "UPREGULATES_AuG", "Gene"),
    ("Compound", "BINDS_CbG", "Gene"),
    ("Compound", "CAUSES_CcSE", "SideEffect"),
    ("Compound", "DOWNREGULATES_CdG", "Gene"),
    ("Compound", "PALLIATES_CpD", "Disease"),
    ("Compound", "RESEMBLES_CrC", "Compound"),
    ("Compound", "TREATS_CtD", "Disease"),
    ("Compound", "UPREGULATES_CuG", "Gene"),
    ("Disease", "ASSOCIATES_DaG", "Gene"),
    ("Disease", "DOWNREGULATES_DdG", "Gene"),
    ("Disease", "LOCALIZES_DlA", "Anatomy"),
    ("Disease", "PRESENTS_DpS", "Symptom"),
    ("Disease", "RESEMBLES_DrD", "Disease"),
    ("Disease", "UPREGULATES_DuG", "Gene"),
    ("Gene", "COVARIES_GcG", "Gene"),
    ("Gene", "INTERACTS_GiG", "Gene"),
    ("Gene", "PARTICIPATES_GpBP", "BiologicalProcess"),
    ("Gene", "PARTICIPATES_GpCC", "CellularComponent"),
    ("Gene", "PARTICIPATES_GpMF", "MolecularFunction"),
    ("Gene", "PARTICIPATES_GpPW", "Pathway"),
    ("Gene", "REGULATES_GrG", "Gene"),
    ("PharmacologicClass", "INCLUDES_PCiC", "Compound"),
]

FULL_PATTERN_TO_ID = {pattern: i for i, pattern in enumerate(FULL_SCHEMA_PATTERNS)}
STOP_ACTION = len(FULL_SCHEMA_PATTERNS)

REWRITE_STRATEGIES = [
    "keep_original",
    "entity_normalization",
    "relation_normalization",
    "full_normalization",
    "aggressive_rewrite",
]

REWRITE_COSTS = np.array([0.00, 0.03, 0.03, 0.06, 0.15], dtype=np.float32)

TOKEN_RE_V3 = re.compile(
    r"\(([A-Za-z0-9_]*)\s*:?\s*([A-Za-z][A-Za-z0-9_]*)?[^)]*\)"
    r"|"
    r"(<-|-)\s*\[[A-Za-z0-9_]*:([A-Za-z][A-Za-z0-9_]*)[^\]]*\]\s*(->|-)"
)


def load_hetionet_dataset(path="HETIONET_dataset.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def tokenize_cypher_pattern_v3(cypher):
    tokens = []
    variable_labels = {}

    for match in TOKEN_RE_V3.finditer(cypher):
        if match.group(3) is None:
            var = match.group(1)
            label = match.group(2)

            if var and label:
                variable_labels[var] = label

            if label is None and var in variable_labels:
                label = variable_labels[var]

            if label is not None:
                tokens.append(("node", label))
            else:
                tokens.append(("node_unknown", var))
        else:
            left_dir = match.group(3)
            rel = match.group(4)
            right_dir = match.group(5)
            tokens.append(("rel", left_dir, rel, right_dir))

    return tokens


def extract_schema_sequence_v3(cypher):
    tokens = tokenize_cypher_pattern_v3(cypher)
    sequence = []

    for i in range(len(tokens) - 2):
        left = tokens[i]
        rel = tokens[i + 1]
        right = tokens[i + 2]

        if left[0] == "node" and rel[0] == "rel" and right[0] == "node":
            left_label = left[1]
            relation = rel[2]
            right_label = right[1]
            left_dir = rel[1]
            right_dir = rel[3]

            if left_dir == "-" and right_dir == "->":
                pattern = (left_label, relation, right_label)
            elif left_dir == "<-" and right_dir == "-":
                pattern = (right_label, relation, left_label)
            else:
                continue

            if pattern not in FULL_PATTERN_TO_ID:
                return None

            sequence.append(FULL_PATTERN_TO_ID[pattern])

    if not sequence:
        return None

    return sequence


ENTITY_SHIFT_RULES = {
    "compound": "drug",
    "compounds": "drugs",
    "disease": "illness",
    "diseases": "illnesses",
    "gene": "genetic factor",
    "genes": "genetic factors",
    "pathway": "biological route",
    "pathways": "biological routes",
    "side effect": "adverse effect",
    "side effects": "adverse effects",
    "symptom": "clinical symptom",
    "symptoms": "clinical symptoms",
}

RELATION_SHIFT_RULES = {
    "treats": "is used for",
    "treat": "is used for",
    "binds": "attaches to",
    "bind": "attach to",
    "participates in": "takes part in",
    "participate in": "take part in",
    "causes": "leads to",
    "cause": "lead to",
    "presents": "shows",
    "present": "show",
}

HARD_SHIFT_RULES = {
    "compound": "medication",
    "compounds": "medications",
    "disease": "medical condition",
    "diseases": "medical conditions",
    "gene": "protein",
    "genes": "proteins",
    "pathway": "signaling cascade",
    "pathways": "signaling cascades",
    "biological process": "biological mechanism",
    "biological processes": "biological mechanisms",
    "side effect": "unwanted reaction",
    "side effects": "unwanted reactions",
    "treats": "is prescribed for",
    "treat": "is prescribed for",
    "binds": "physically interacts with",
    "bind": "physically interact with",
    "participates in": "is involved in",
    "participate in": "are involved in",
    "causes": "is responsible for",
    "cause": "is responsible for",
}


def clean_paraphrase_text(text):
    replacements = [
        (r"\bare used to is used for\b", "are used for"),
        (r"\bis used to is used for\b", "is used for"),
        (r"\bused to is used for\b", "used for"),
        (r"\bto is used for\b", "to treat"),
        (r"\btake part in in\b", "take part in"),
        (r"\btakes part in in\b", "takes part in"),
        (r"\bare involved in in\b", "are involved in"),
        (r"\bis involved in in\b", "is involved in"),
        (r"\bparticipate in in\b", "participate in"),
        (r"\bparticipates in in\b", "participates in"),
    ]

    cleaned = text
    for pattern, replacement in replacements:
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)

    return re.sub(r"\s+", " ", cleaned).strip()


def apply_text_rules(text, rules):
    new_text = text
    for old, new in sorted(rules.items(), key=lambda x: -len(x[0])):
        new_text = re.sub(rf"\b{re.escape(old)}\b", new, new_text, flags=re.IGNORECASE)
    return clean_paraphrase_text(new_text)


def make_paraphrased_variants(example):
    q = example["question"]
    q_entity_relation = apply_text_rules(apply_text_rules(q, ENTITY_SHIFT_RULES), RELATION_SHIFT_RULES)

    return [
        {**example, "question": q, "variant": "original"},
        {**example, "question": apply_text_rules(q, ENTITY_SHIFT_RULES), "variant": "entity_shift"},
        {**example, "question": apply_text_rules(q, RELATION_SHIFT_RULES), "variant": "relation_shift"},
        {**example, "question": q_entity_relation, "variant": "entity_relation_shift"},
        {**example, "question": apply_text_rules(q, HARD_SHIFT_RULES), "variant": "hard_shift"},
    ]


def build_schema_sequence_examples(data, max_hops=3):
    examples = []
    for ex in data:
        seq = extract_schema_sequence_v3(ex["cypher"])
        if seq is not None and len(seq) <= max_hops:
            examples.append({**ex, "gold_schema_sequence": seq, "hop_count": len(seq)})
    return examples


def build_augmented_examples(schema_sequence_examples):
    augmented = []
    for base_id, ex in enumerate(schema_sequence_examples):
        variants = make_paraphrased_variants({**ex, "base_id": base_id})
        for v in variants:
            v["gold_schema_sequence"] = ex["gold_schema_sequence"]
            v["hop_count"] = ex["hop_count"]
            v["question"] = clean_paraphrase_text(v["question"])
        augmented.extend(variants)
    return augmented


def split_by_base_id(schema_sequence_examples, augmented_examples, seed=42):
    base_ids = np.arange(len(schema_sequence_examples))
    rng = np.random.default_rng(seed)
    rng.shuffle(base_ids)

    n = len(base_ids)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)

    train_ids = set(base_ids[:n_train])
    val_ids = set(base_ids[n_train : n_train + n_val])
    test_ids = set(base_ids[n_train + n_val :])

    train = [ex for ex in augmented_examples if ex["base_id"] in train_ids]
    val = [ex for ex in augmented_examples if ex["base_id"] in val_ids]
    test = [ex for ex in augmented_examples if ex["base_id"] in test_ids]
    return train, val, test


def example_key(ex):
    return f'{ex["base_id"]}::{ex["variant"]}::{ex["question"]}'


def print_dataset_summary(schema_examples, train, val, test):
    print("Base examples:", len(schema_examples))
    print("Base hop distribution:", Counter(ex["hop_count"] for ex in schema_examples))
    print("Train/Val/Test:", len(train), len(val), len(test))
    print("Train hops:", Counter(ex["hop_count"] for ex in train))
    print("Val hops:", Counter(ex["hop_count"] for ex in val))
    print("Test hops:", Counter(ex["hop_count"] for ex in test))


def save_pickle(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def build_embedding_cache(examples, model_name="sentence-transformers/all-MiniLM-L6-v2", batch_size=64):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    questions = [ex["question"] for ex in examples]
    embeddings = model.encode(
        questions,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return {example_key(ex): emb.astype(np.float32) for ex, emb in zip(examples, embeddings)}


LLM_REWRITE_PROMPTS = {
    0: None,
    1: """Rewrite the biomedical question by normalizing biomedical entity names toward Hetionet terminology.
Use words such as compound, disease, gene, pathway, biological process, side effect, symptom when appropriate.
Preserve all named entities, filters, numbers, and the original meaning.
Question: {question}
Rewritten question:""",
    2: """Rewrite the biomedical question by normalizing relationship expressions toward Hetionet relation terminology.
Use words such as treats, binds, participates in, causes, presents, associates, regulates, expresses, localizes when appropriate.
Preserve all named entities, filters, numbers, and the original meaning.
Question: {question}
Rewritten question:""",
    3: """Rewrite the biomedical question to make both biomedical entities and relationships explicit using Hetionet-style terminology.
Preserve all named entities, biomedical terms, numbers, filters, and the original meaning.
Question: {question}
Rewritten question:""",
    4: """Rewrite the biomedical question in a more explicit and disambiguated form for generating a Cypher query over Hetionet.
Preserve all named entities, biomedical terms, numbers, filters, and the original meaning. Do not answer the question.
Question: {question}
Rewritten question:""",
}


def load_llm_rewriter(model_name="google/flan-t5-base"):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    device = "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    return tokenizer, model, device


def llm_rewrite_question(question, strategy_id, tokenizer, model, device, max_new_tokens=96):
    import torch

    if strategy_id == 0:
        return question

    prompt = LLM_REWRITE_PROMPTS[strategy_id].format(question=question)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=384).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=4,
        )
    rewritten = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    return clean_paraphrase_text(rewritten if rewritten else question)


def build_llm_rewrite_cache(examples, tokenizer, model, device, limit=None):
    selected = examples if limit is None else examples[:limit]
    cache = {}
    for i, ex in enumerate(selected, start=1):
        key = example_key(ex)
        cache[key] = {}
        for strategy_id in range(len(REWRITE_STRATEGIES)):
            cache[key][strategy_id] = llm_rewrite_question(
                ex["question"], strategy_id, tokenizer, model, device
            )
        if i % 10 == 0 or i == len(selected):
            print(f"LLM rewrite cache: {i}/{len(selected)}")
    return cache


QWEN_SCHEMA_SYSTEM_PROMPT = """You are a conservative biomedical query rewriter for Hetionet Text-to-Cypher.
Rewrite the user question using Hetionet-compatible terminology while preserving the exact meaning.
Keep quoted names, identifiers, numbers, filters, ordering constraints, and limits unchanged.
Do not answer the question. Return only the rewritten question."""

QWEN_SCHEMA_RULES = {
    0: "",
    1: """Normalize entity words only when appropriate:
- drug, medication -> compound
- medical condition, illness -> disease
- genetic factor, protein -> gene
- adverse effect -> side effect
Do not change relation meanings.""",
    2: """Normalize relation words only when appropriate:
- prescribed for, used for -> treats
- show symptom, commonly show -> present symptom
- lead to side effects -> cause side effects
- involved in -> participate in
Do not replace a specific relation with vague words such as associated with.""",
    3: """Normalize both entity and relation words when appropriate:
- drug/medication -> compound
- medical condition/illness -> disease
- genetic factor/protein -> gene
- prescribed for/used for -> treats
- show symptom -> present symptom
- involved in -> participate in
Do not replace a specific relation with vague words such as associated with.""",
    4: """Make the question explicit for Hetionet schema selection.
Use canonical words such as compound, disease, gene, pathway, biological process, molecular function,
cellular component, side effect, symptom, treats, binds, causes, presents, participates in, regulates.
Preserve the exact biomedical meaning and all constraints.""",
}

QWEN_FEW_SHOT_EXAMPLES = [
    (
        "Find drugs that are used for 'hypertension'.",
        "Find compounds that treat 'hypertension'.",
    ),
    (
        "Which medical conditions commonly show the symptom 'Apnea'?",
        "Which diseases present the symptom 'Apnea'?",
    ),
    (
        "What proteins are involved in the biological process 'cell cycle'?",
        "What genes participate in the biological process 'cell cycle'?",
    ),
    (
        "Which drugs lead to side effects named 'Nausea'?",
        "Which compounds cause side effects named 'Nausea'?",
    ),
    (
        "Find anatomical locations associated with genetic factors expressed in 'lung'.",
        "Find anatomy locations that express genes in 'lung'.",
    ),
]


def _format_few_shot_examples():
    blocks = []
    for source, target in QWEN_FEW_SHOT_EXAMPLES:
        blocks.append(f"Input: {source}\nOutput: {target}")
    return "\n\n".join(blocks)


def load_qwen_rewriter(model_name="Qwen/Qwen2.5-0.5B-Instruct"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    device = "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    return tokenizer, model, device


def qwen_rewrite_question(question, strategy_id, tokenizer, model, device, max_new_tokens=96):
    import torch

    if strategy_id == 0:
        return question

    rules = QWEN_SCHEMA_RULES[strategy_id]
    messages = [
        {"role": "system", "content": QWEN_SCHEMA_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"{rules}\n\n"
                "Question:\n"
                f"{question}\n\n"
                "Rewritten question:"
            ),
        },
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=768).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    rewritten = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    rewritten = rewritten.split("\n")[0].strip()
    if rewritten.lower().startswith("rewritten question:"):
        rewritten = rewritten.split(":", 1)[1].strip()
    return clean_paraphrase_text(rewritten if rewritten else question)


def qwen_few_shot_rewrite_question(question, strategy_id, tokenizer, model, device, max_new_tokens=96):
    import torch

    if strategy_id == 0:
        return question

    rules = QWEN_SCHEMA_RULES[strategy_id]
    examples = _format_few_shot_examples()
    messages = [
        {"role": "system", "content": QWEN_SCHEMA_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Use the following examples as the target rewriting style.\n\n"
                f"{examples}\n\n"
                f"Rules for this rewrite:\n{rules}\n\n"
                "Now rewrite this question. Preserve quoted names, identifiers, numbers, filters, ordering, and limits.\n\n"
                f"Input: {question}\n"
                "Output:"
            ),
        },
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    rewritten = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    rewritten = rewritten.split("\n")[0].strip()
    if rewritten.lower().startswith("output:"):
        rewritten = rewritten.split(":", 1)[1].strip()
    if rewritten.lower().startswith("rewritten question:"):
        rewritten = rewritten.split(":", 1)[1].strip()
    return clean_paraphrase_text(rewritten if rewritten else question)


RELATION_CUE_GROUPS = {
    "downregulates": [r"downregulat"],
    "upregulates": [r"upregulat"],
    "binds": [r"\bbind", r"\bbinds", r"\bbound"],
    "treats": [r"\btreat", r"used for", r"used to treat", r"prescribed for"],
    "palliates": [r"\bpalliat"],
    "causes_side_effect": [r"\bcause", r"lead to", r"side effect", r"adverse effect"],
    "presents_symptom": [r"\bpresent", r"\bshow", r"symptom"],
    "participates": [r"participat", r"involved in", r"take part"],
    "localizes": [r"localiz"],
    "expresses": [r"express"],
    "associates": [r"associat"],
    "interacts": [r"interact"],
    "regulates": [r"\bregulat"],
    "resembles": [r"resembl"],
}


def _has_any_cue(text, cues):
    text_l = text.lower()
    return any(re.search(cue, text_l) for cue in cues)


def _relation_profile(text):
    return {
        group
        for group, cues in RELATION_CUE_GROUPS.items()
        if _has_any_cue(text, cues)
    }


def is_safe_rewrite(original, rewritten):
    if not rewritten or len(rewritten.split()) < 4:
        return False

    original_l = original.lower()
    rewritten_l = rewritten.lower()

    for quoted in re.findall(r"'([^']+)'|\"([^\"]+)\"", original):
        value = quoted[0] or quoted[1]
        if value and value.lower() not in rewritten_l:
            return False

    original_profile = _relation_profile(original_l)
    rewritten_profile = _relation_profile(rewritten_l)

    for required_group in original_profile:
        if required_group not in rewritten_profile:
            return False

    risky_added_groups = {"binds", "downregulates", "upregulates", "regulates", "palliates"}
    if rewritten_profile.intersection(risky_added_groups) - original_profile:
        return False

    return True


def safe_qwen_rewrite_question(question, strategy_id, tokenizer, model, device, max_new_tokens=96):
    rewritten = qwen_rewrite_question(
        question,
        strategy_id,
        tokenizer,
        model,
        device,
        max_new_tokens=max_new_tokens,
    )
    if strategy_id == 0 or is_safe_rewrite(question, rewritten):
        return rewritten
    return question


def safe_qwen_few_shot_rewrite_question(question, strategy_id, tokenizer, model, device, max_new_tokens=96):
    rewritten = qwen_few_shot_rewrite_question(
        question,
        strategy_id,
        tokenizer,
        model,
        device,
        max_new_tokens=max_new_tokens,
    )
    if strategy_id == 0 or is_safe_rewrite(question, rewritten):
        return rewritten
    return question


def build_qwen_rewrite_cache(examples, tokenizer, model, device, limit=None, cache_path=None):
    selected = examples if limit is None else examples[:limit]
    cache = {}
    if cache_path is not None and Path(cache_path).exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        cache = {
            tuple(k.split("|||")): {int(a): text for a, text in rewrites.items()}
            for k, rewrites in raw.items()
        }

    for i, ex in enumerate(selected, start=1):
        key = example_key(ex)
        if key not in cache:
            cache[key] = {}
        for strategy_id in range(len(REWRITE_STRATEGIES)):
            if strategy_id not in cache[key]:
                cache[key][strategy_id] = qwen_rewrite_question(
                    ex["question"], strategy_id, tokenizer, model, device
                )
        if cache_path is not None and (i % 5 == 0 or i == len(selected)):
            serializable = {
                "|||".join(k): {str(a): text for a, text in rewrites.items()}
                for k, rewrites in cache.items()
            }
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(serializable, f, indent=2, ensure_ascii=False)
        if i % 5 == 0 or i == len(selected):
            print(f"Qwen rewrite cache: {i}/{len(selected)}")
    return cache


def build_safe_qwen_rewrite_cache(examples, tokenizer, model, device, limit=None, cache_path=None):
    selected = examples if limit is None else examples[:limit]
    cache = {}
    if cache_path is not None and Path(cache_path).exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        cache = {
            tuple(k.split("|||")): {int(a): text for a, text in rewrites.items()}
            for k, rewrites in raw.items()
        }

    for i, ex in enumerate(selected, start=1):
        key = example_key(ex)
        if key not in cache:
            cache[key] = {}
        for strategy_id in range(len(REWRITE_STRATEGIES)):
            if strategy_id not in cache[key]:
                cache[key][strategy_id] = safe_qwen_rewrite_question(
                    ex["question"], strategy_id, tokenizer, model, device
                )
        if cache_path is not None and (i % 5 == 0 or i == len(selected)):
            serializable = {
                "|||".join(k): {str(a): text for a, text in rewrites.items()}
                for k, rewrites in cache.items()
            }
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(serializable, f, indent=2, ensure_ascii=False)
        if i % 5 == 0 or i == len(selected):
            print(f"Safe Qwen rewrite cache: {i}/{len(selected)}")
    return cache


def build_safe_qwen_few_shot_rewrite_cache(examples, tokenizer, model, device, limit=None, cache_path=None):
    selected = examples if limit is None else examples[:limit]
    cache = {}
    if cache_path is not None and Path(cache_path).exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        cache = {
            tuple(k.split("|||")): {int(a): text for a, text in rewrites.items()}
            for k, rewrites in raw.items()
        }

    for i, ex in enumerate(selected, start=1):
        key = example_key(ex)
        if key not in cache:
            cache[key] = {}
        for strategy_id in range(len(REWRITE_STRATEGIES)):
            if strategy_id not in cache[key]:
                cache[key][strategy_id] = safe_qwen_few_shot_rewrite_question(
                    ex["question"], strategy_id, tokenizer, model, device
                )
        if cache_path is not None and (i % 5 == 0 or i == len(selected)):
            serializable = {
                "|||".join(k): {str(a): text for a, text in rewrites.items()}
                for k, rewrites in cache.items()
            }
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(serializable, f, indent=2, ensure_ascii=False)
        if i % 5 == 0 or i == len(selected):
            print(f"Safe few-shot Qwen rewrite cache: {i}/{len(selected)}")
    return cache


def build_rewrite_embedding_cache(rewrite_cache, model_name="sentence-transformers/all-MiniLM-L6-v2", batch_size=64):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    pairs = []
    texts = []
    for key, rewrites in rewrite_cache.items():
        for strategy_id, text in rewrites.items():
            pairs.append((key, strategy_id))
            texts.append(text)

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return {pair: emb.astype(np.float32) for pair, emb in zip(pairs, embeddings)}


SCHEMA_CUE_PATTERNS = [
    ("compound", [r"\bcompound", r"\bdrug", r"\bmedication"]),
    ("disease", [r"\bdisease", r"medical condition", r"\billness"]),
    ("gene", [r"\bgene", r"genetic factor", r"\bprotein"]),
    ("anatomy", [r"\banatom", r"\btissue", r"\borgan"]),
    ("symptom", [r"\bsymptom", r"\bclinical symptom"]),
    ("side_effect", [r"side effect", r"adverse effect"]),
    ("pathway", [r"\bpathway"]),
    ("biological_process", [r"biological process"]),
    ("molecular_function", [r"molecular function"]),
    ("cellular_component", [r"cellular component"]),
    ("treats", [r"\btreat", r"used for", r"used to treat", r"prescribed for"]),
    ("palliates", [r"\bpalliat"]),
    ("binds", [r"\bbind", r"\bbinds", r"\bbinding"]),
    ("causes", [r"\bcause", r"lead to", r"leads to"]),
    ("downregulates", [r"downregulat"]),
    ("upregulates", [r"upregulat"]),
    ("associates", [r"associat"]),
    ("presents", [r"\bpresent", r"\bshow", r"commonly show"]),
    ("localizes", [r"localiz"]),
    ("expresses", [r"express"]),
    ("participates", [r"participat", r"involved in", r"take part"]),
    ("regulates", [r"\bregulat"]),
    ("interacts", [r"interact"]),
    ("covaries", [r"covar"]),
    ("resembles", [r"resembl"]),
    ("count_or_aggregate", [r"\bcount", r"\baverage", r"\bavg", r"\bnumber", r"\btop"]),
    ("filter_numeric", [r">", r"<", r"greater than", r"less than", r"at least", r"highest", r"fewest"]),
]


def schema_cue_features(text):
    text_l = text.lower()
    features = []
    for _, patterns in SCHEMA_CUE_PATTERNS:
        features.append(float(any(re.search(pattern, text_l) for pattern in patterns)))
    return np.array(features, dtype=np.float32)


def augment_embedding_with_schema_cues(embedding, text):
    return np.concatenate([embedding.astype(np.float32), schema_cue_features(text)])


def build_schema_cue_embedding_cache(examples, base_embedding_cache):
    cache = {}
    for ex in examples:
        key = example_key(ex)
        cache[key] = augment_embedding_with_schema_cues(base_embedding_cache[key], ex["question"])
    return cache


def build_schema_cue_rewrite_embedding_cache(rewrite_cache, rewrite_embedding_cache):
    cache = {}
    for key, rewrites in rewrite_cache.items():
        for strategy_id, text in rewrites.items():
            cache[(key, strategy_id)] = augment_embedding_with_schema_cues(
                rewrite_embedding_cache[(key, strategy_id)],
                text,
            )
    return cache


def pattern_partial_score(pred_id, gold_id):
    pred = FULL_SCHEMA_PATTERNS[pred_id]
    gold = FULL_SCHEMA_PATTERNS[gold_id]
    score = 0.0
    if pred[0] == gold[0]:
        score += 0.2
    if pred[1] == gold[1]:
        score += 0.6
    if pred[2] == gold[2]:
        score += 0.2
    return score


class LLMRewriteSchemaEnv(gym.Env):
    def __init__(self, examples, original_embedding_cache, rewrite_cache, rewrite_embedding_cache, max_schema_steps=3, shaped=True):
        super().__init__()
        self.examples = [ex for ex in examples if example_key(ex) in rewrite_cache]
        if not self.examples:
            raise ValueError("No examples are covered by the rewrite cache.")
        self.original_embedding_cache = original_embedding_cache
        self.rewrite_cache = rewrite_cache
        self.rewrite_embedding_cache = rewrite_embedding_cache
        self.max_schema_steps = max_schema_steps
        self.shaped = shaped
        self.n_rewrite_actions = len(REWRITE_STRATEGIES)
        self.n_schema_patterns = len(FULL_SCHEMA_PATTERNS)
        self.stop_action = STOP_ACTION
        self.n_actions = max(self.n_rewrite_actions, self.n_schema_patterns + 1)

        first_emb = self.original_embedding_cache[example_key(self.examples[0])]
        self.obs_dim = len(first_emb) + self.n_rewrite_actions + self.max_schema_steps + 1
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.n_actions)
        self.current_example = None
        self.chosen_rewrite_action = None
        self.selected_schema_sequence = None
        self.step_id = 0

    def _question_embedding(self):
        key = example_key(self.current_example)
        if self.chosen_rewrite_action is None:
            return self.original_embedding_cache[key]
        return self.rewrite_embedding_cache[(key, self.chosen_rewrite_action)]

    def _make_obs(self):
        q_features = self._question_embedding()
        rewrite_one_hot = np.zeros(self.n_rewrite_actions, dtype=np.float32)
        if self.chosen_rewrite_action is not None:
            rewrite_one_hot[self.chosen_rewrite_action] = 1.0

        selected_sequence_features = np.zeros(self.max_schema_steps, dtype=np.float32)
        if self.selected_schema_sequence is not None:
            for i, pattern_id in enumerate(self.selected_schema_sequence[: self.max_schema_steps]):
                selected_sequence_features[i] = (pattern_id + 1) / self.n_schema_patterns

        step_feature = np.array([self.step_id / self.max_schema_steps], dtype=np.float32)
        return np.concatenate([q_features.astype(np.float32), rewrite_one_hot, selected_sequence_features, step_feature])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        idx = self.np_random.integers(0, len(self.examples))
        self.current_example = self.examples[idx]
        self.chosen_rewrite_action = None
        self.selected_schema_sequence = []
        self.step_id = 0
        return self._make_obs(), {
            "question": self.current_example["question"],
            "variant": self.current_example["variant"],
            "gold_schema_sequence": self.current_example["gold_schema_sequence"],
            "hop_count": self.current_example["hop_count"],
        }

    def _final_reward(self):
        gold = self.current_example["gold_schema_sequence"]
        pred = self.selected_schema_sequence
        correct_positions = sum(1 for p, g in zip(pred, gold) if p == g)
        precision = correct_positions / max(len(pred), 1)
        recall = correct_positions / len(gold)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        exact_match = pred == gold
        reward = -1.0 + 2.0 * f1
        if exact_match:
            reward += 0.5
        if len(pred) < len(gold):
            reward -= 0.2
        if len(pred) > len(gold):
            reward -= 0.2
        reward -= float(REWRITE_COSTS[self.chosen_rewrite_action])
        return reward, {"precision": precision, "recall": recall, "f1": f1, "exact_match": exact_match}

    def step(self, action):
        if self.step_id == 0:
            if action >= self.n_rewrite_actions:
                return self._make_obs(), -1.0, True, False, {"error": "invalid rewrite action"}
            self.chosen_rewrite_action = int(action)
            self.step_id = 1
            key = example_key(self.current_example)
            return self._make_obs(), 0.0, False, False, {
                "phase": "rewrite",
                "rewrite_action": self.chosen_rewrite_action,
                "rewrite_strategy": REWRITE_STRATEGIES[self.chosen_rewrite_action],
                "rewritten_question": self.rewrite_cache[key][self.chosen_rewrite_action],
            }

        if action == self.stop_action:
            reward, reward_info = self._final_reward()
            return self._make_obs(), reward, True, False, {
                "phase": "schema_sequence",
                "rewrite_action": self.chosen_rewrite_action,
                "rewrite_strategy": REWRITE_STRATEGIES[self.chosen_rewrite_action],
                "predicted_sequence": self.selected_schema_sequence,
                "gold_sequence": self.current_example["gold_schema_sequence"],
                **reward_info,
            }

        if action >= self.n_schema_patterns:
            return self._make_obs(), -1.0, True, False, {"error": "invalid schema action"}

        position = len(self.selected_schema_sequence)
        gold = self.current_example["gold_schema_sequence"]
        self.selected_schema_sequence.append(int(action))
        self.step_id += 1

        step_reward = 0.0
        if self.shaped:
            if position < len(gold):
                step_reward = pattern_partial_score(int(action), gold[position]) - 0.3
            else:
                step_reward = -0.5

        if len(self.selected_schema_sequence) >= self.max_schema_steps:
            final_reward, reward_info = self._final_reward()
            return self._make_obs(), step_reward + final_reward, True, False, {
                "phase": "schema_sequence",
                "rewrite_action": self.chosen_rewrite_action,
                "rewrite_strategy": REWRITE_STRATEGIES[self.chosen_rewrite_action],
                "predicted_sequence": self.selected_schema_sequence,
                "gold_sequence": gold,
                **reward_info,
            }

        return self._make_obs(), step_reward, False, False, {
            "phase": "schema_selection",
            "selected_sequence": self.selected_schema_sequence,
        }


def patterns_connected(pattern_a, pattern_b):
    labels_a = {pattern_a[0], pattern_a[2]}
    labels_b = {pattern_b[0], pattern_b[2]}
    return bool(labels_a.intersection(labels_b))


class HierarchicalHopSchemaEnv(gym.Env):
    def __init__(
        self,
        examples,
        original_embedding_cache,
        rewrite_cache,
        rewrite_embedding_cache,
        max_hops=2,
        shaped=True,
        connectivity_mask=True,
    ):
        super().__init__()
        self.examples = [ex for ex in examples if example_key(ex) in rewrite_cache and ex["hop_count"] <= max_hops]
        if not self.examples:
            raise ValueError("No examples are covered by the rewrite cache.")
        self.original_embedding_cache = original_embedding_cache
        self.rewrite_cache = rewrite_cache
        self.rewrite_embedding_cache = rewrite_embedding_cache
        self.max_hops = max_hops
        self.shaped = shaped
        self.connectivity_mask = connectivity_mask
        self.n_rewrite_actions = len(REWRITE_STRATEGIES)
        self.n_schema_patterns = len(FULL_SCHEMA_PATTERNS)
        self.n_actions = max(self.n_rewrite_actions, self.n_schema_patterns, self.max_hops)

        first_emb = self.original_embedding_cache[example_key(self.examples[0])]
        self.obs_dim = len(first_emb) + self.n_rewrite_actions + self.max_hops + self.max_hops + 1
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.n_actions)
        self.current_example = None
        self.chosen_rewrite_action = None
        self.chosen_hop_count = None
        self.selected_schema_sequence = None
        self.phase = 0

    def _question_embedding(self):
        key = example_key(self.current_example)
        if self.chosen_rewrite_action is None:
            return self.original_embedding_cache[key]
        return self.rewrite_embedding_cache[(key, self.chosen_rewrite_action)]

    def _make_obs(self):
        q_features = self._question_embedding()

        rewrite_one_hot = np.zeros(self.n_rewrite_actions, dtype=np.float32)
        if self.chosen_rewrite_action is not None:
            rewrite_one_hot[self.chosen_rewrite_action] = 1.0

        hop_one_hot = np.zeros(self.max_hops, dtype=np.float32)
        if self.chosen_hop_count is not None:
            hop_one_hot[self.chosen_hop_count - 1] = 1.0

        selected_sequence_features = np.zeros(self.max_hops, dtype=np.float32)
        if self.selected_schema_sequence is not None:
            for i, pattern_id in enumerate(self.selected_schema_sequence[: self.max_hops]):
                selected_sequence_features[i] = (pattern_id + 1) / self.n_schema_patterns

        phase_feature = np.array([self.phase / (self.max_hops + 2)], dtype=np.float32)
        return np.concatenate(
            [
                q_features.astype(np.float32),
                rewrite_one_hot,
                hop_one_hot,
                selected_sequence_features,
                phase_feature,
            ]
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        idx = self.np_random.integers(0, len(self.examples))
        self.current_example = self.examples[idx]
        self.chosen_rewrite_action = None
        self.chosen_hop_count = None
        self.selected_schema_sequence = []
        self.phase = 0
        return self._make_obs(), {
            "question": self.current_example["question"],
            "variant": self.current_example["variant"],
            "gold_schema_sequence": self.current_example["gold_schema_sequence"],
            "hop_count": self.current_example["hop_count"],
        }

    def get_valid_actions(self):
        if self.phase == 0:
            return list(range(self.n_rewrite_actions))
        if self.phase == 1:
            return list(range(self.max_hops))

        if not self.selected_schema_sequence or not self.connectivity_mask:
            return list(range(self.n_schema_patterns))

        selected_patterns = [FULL_SCHEMA_PATTERNS[i] for i in self.selected_schema_sequence]
        valid = []
        for pattern_id, pattern in enumerate(FULL_SCHEMA_PATTERNS):
            if any(patterns_connected(pattern, selected) for selected in selected_patterns):
                valid.append(pattern_id)
        return valid if valid else list(range(self.n_schema_patterns))

    def _final_reward(self):
        gold = self.current_example["gold_schema_sequence"]
        pred = self.selected_schema_sequence
        correct_positions = sum(1 for p, g in zip(pred, gold) if p == g)
        precision = correct_positions / max(len(pred), 1)
        recall = correct_positions / len(gold)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        exact_match = pred == gold

        reward = -1.0 + 2.0 * f1
        if exact_match:
            reward += 0.5
        if self.chosen_hop_count == len(gold):
            reward += 0.2
        else:
            reward -= 0.3 * abs(self.chosen_hop_count - len(gold))
        reward -= float(REWRITE_COSTS[self.chosen_rewrite_action])
        return reward, {"precision": precision, "recall": recall, "f1": f1, "exact_match": exact_match}

    def step(self, action):
        if self.phase == 0:
            if action >= self.n_rewrite_actions:
                return self._make_obs(), -1.0, True, False, {"error": "invalid rewrite action"}
            self.chosen_rewrite_action = int(action)
            self.phase = 1
            key = example_key(self.current_example)
            return self._make_obs(), 0.0, False, False, {
                "phase": "rewrite",
                "rewrite_action": self.chosen_rewrite_action,
                "rewrite_strategy": REWRITE_STRATEGIES[self.chosen_rewrite_action],
                "rewritten_question": self.rewrite_cache[key][self.chosen_rewrite_action],
            }

        if self.phase == 1:
            if action >= self.max_hops:
                return self._make_obs(), -1.0, True, False, {"error": "invalid hop action"}
            self.chosen_hop_count = int(action) + 1
            self.phase = 2
            gold_hops = len(self.current_example["gold_schema_sequence"])
            hop_reward = 0.2 if self.chosen_hop_count == gold_hops else -0.2
            return self._make_obs(), hop_reward if self.shaped else 0.0, False, False, {
                "phase": "hop_count",
                "chosen_hop_count": self.chosen_hop_count,
                "gold_hop_count": gold_hops,
            }

        if action >= self.n_schema_patterns:
            return self._make_obs(), -1.0, True, False, {"error": "invalid schema action"}

        valid_actions = self.get_valid_actions()
        if action not in valid_actions:
            return self._make_obs(), -1.0, True, False, {"error": "masked schema action"}

        position = len(self.selected_schema_sequence)
        gold = self.current_example["gold_schema_sequence"]
        self.selected_schema_sequence.append(int(action))

        step_reward = 0.0
        if self.shaped:
            if position < len(gold):
                step_reward = pattern_partial_score(int(action), gold[position]) - 0.3
            else:
                step_reward = -0.5

        if len(self.selected_schema_sequence) >= self.chosen_hop_count:
            final_reward, reward_info = self._final_reward()
            return self._make_obs(), step_reward + final_reward, True, False, {
                "phase": "schema_sequence",
                "rewrite_action": self.chosen_rewrite_action,
                "rewrite_strategy": REWRITE_STRATEGIES[self.chosen_rewrite_action],
                "chosen_hop_count": self.chosen_hop_count,
                "predicted_sequence": self.selected_schema_sequence,
                "gold_sequence": gold,
                **reward_info,
            }

        return self._make_obs(), step_reward, False, False, {
            "phase": "schema_selection",
            "chosen_hop_count": self.chosen_hop_count,
            "selected_sequence": self.selected_schema_sequence,
        }


def build_schema_path_actions(examples, max_hops=2):
    paths_by_hop = {hop: set() for hop in range(1, max_hops + 1)}
    for ex in examples:
        seq = tuple(ex["gold_schema_sequence"])
        if 1 <= len(seq) <= max_hops:
            paths_by_hop[len(seq)].add(seq)
    return {hop: sorted(paths) for hop, paths in paths_by_hop.items()}


PATTERN_CUE_MAP = {
    0: ["downregulates"],
    1: ["expresses"],
    2: ["upregulates"],
    3: ["binds"],
    4: ["causes", "side_effect"],
    5: ["downregulates"],
    6: ["palliates"],
    7: ["resembles"],
    8: ["treats"],
    9: ["upregulates"],
    10: ["associates"],
    11: ["downregulates"],
    12: ["localizes", "anatomy"],
    13: ["presents", "symptom"],
    14: ["resembles"],
    15: ["upregulates"],
    16: ["covaries"],
    17: ["interacts"],
    18: ["participates", "biological_process"],
    19: ["participates", "cellular_component"],
    20: ["participates", "molecular_function"],
    21: ["participates", "pathway"],
    22: ["regulates"],
    23: ["compound"],
}


def active_schema_cue_names(text):
    feats = schema_cue_features(text)
    return {name for (name, _), value in zip(SCHEMA_CUE_PATTERNS, feats) if value > 0}


def path_cue_score(path, active_cues):
    score = 0
    for pattern_id in path:
        if active_cues.intersection(PATTERN_CUE_MAP.get(pattern_id, [])):
            score += 1
    return score


class HierarchicalSchemaPathEnv(gym.Env):
    def __init__(
        self,
        examples,
        original_embedding_cache,
        rewrite_cache,
        rewrite_embedding_cache,
        schema_paths_by_hop,
        max_hops=2,
        shaped=True,
        lexical_path_mask=False,
    ):
        super().__init__()
        self.examples = [ex for ex in examples if example_key(ex) in rewrite_cache and ex["hop_count"] <= max_hops]
        if not self.examples:
            raise ValueError("No examples are covered by the rewrite cache.")
        self.original_embedding_cache = original_embedding_cache
        self.rewrite_cache = rewrite_cache
        self.rewrite_embedding_cache = rewrite_embedding_cache
        self.schema_paths_by_hop = schema_paths_by_hop
        self.max_hops = max_hops
        self.shaped = shaped
        self.lexical_path_mask = lexical_path_mask
        self.n_rewrite_actions = len(REWRITE_STRATEGIES)
        self.max_path_actions = max(len(v) for v in schema_paths_by_hop.values())
        self.n_actions = max(self.n_rewrite_actions, self.max_hops, self.max_path_actions)

        first_emb = self.original_embedding_cache[example_key(self.examples[0])]
        self.obs_dim = len(first_emb) + self.n_rewrite_actions + self.max_hops + 1
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.n_actions)
        self.current_example = None
        self.chosen_rewrite_action = None
        self.chosen_hop_count = None
        self.phase = 0

    def _question_embedding(self):
        key = example_key(self.current_example)
        if self.chosen_rewrite_action is None:
            return self.original_embedding_cache[key]
        return self.rewrite_embedding_cache[(key, self.chosen_rewrite_action)]

    def _make_obs(self):
        q_features = self._question_embedding()
        rewrite_one_hot = np.zeros(self.n_rewrite_actions, dtype=np.float32)
        if self.chosen_rewrite_action is not None:
            rewrite_one_hot[self.chosen_rewrite_action] = 1.0
        hop_one_hot = np.zeros(self.max_hops, dtype=np.float32)
        if self.chosen_hop_count is not None:
            hop_one_hot[self.chosen_hop_count - 1] = 1.0
        phase_feature = np.array([self.phase / 3], dtype=np.float32)
        return np.concatenate([q_features.astype(np.float32), rewrite_one_hot, hop_one_hot, phase_feature])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        idx = self.np_random.integers(0, len(self.examples))
        self.current_example = self.examples[idx]
        self.chosen_rewrite_action = None
        self.chosen_hop_count = None
        self.phase = 0
        return self._make_obs(), {
            "question": self.current_example["question"],
            "variant": self.current_example["variant"],
            "gold_schema_sequence": self.current_example["gold_schema_sequence"],
            "hop_count": self.current_example["hop_count"],
        }

    def get_valid_actions(self):
        if self.phase == 0:
            return list(range(self.n_rewrite_actions))
        if self.phase == 1:
            return list(range(self.max_hops))
        paths = self.schema_paths_by_hop[self.chosen_hop_count]
        if not self.lexical_path_mask:
            return list(range(len(paths)))

        key = example_key(self.current_example)
        text = self.rewrite_cache[key][self.chosen_rewrite_action]
        active_cues = active_schema_cue_names(text)
        scored = [(i, path_cue_score(path, active_cues)) for i, path in enumerate(paths)]
        max_score = max(score for _, score in scored)
        if max_score <= 0:
            return list(range(len(paths)))

        min_score = 1 if self.chosen_hop_count == 1 else min(2, max_score)
        valid = [i for i, score in scored if score >= min_score]
        min_candidates = 3 if self.chosen_hop_count == 1 else 5
        if len(valid) < min_candidates:
            valid = [i for i, score in scored if score >= 1]
        return valid if valid else list(range(len(paths)))

    def _score_path(self, pred):
        gold = tuple(self.current_example["gold_schema_sequence"])
        correct_positions = sum(1 for p, g in zip(pred, gold) if p == g)
        precision = correct_positions / max(len(pred), 1)
        recall = correct_positions / len(gold)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        exact_match = tuple(pred) == gold
        reward = -1.0 + 2.0 * f1
        if exact_match:
            reward += 0.7
        if self.chosen_hop_count == len(gold):
            reward += 0.2
        else:
            reward -= 0.3 * abs(self.chosen_hop_count - len(gold))
        reward -= float(REWRITE_COSTS[self.chosen_rewrite_action])
        return reward, {"precision": precision, "recall": recall, "f1": f1, "exact_match": exact_match}

    def step(self, action):
        if self.phase == 0:
            if action >= self.n_rewrite_actions:
                return self._make_obs(), -1.0, True, False, {"error": "invalid rewrite action"}
            self.chosen_rewrite_action = int(action)
            self.phase = 1
            key = example_key(self.current_example)
            return self._make_obs(), 0.0, False, False, {
                "phase": "rewrite",
                "rewrite_action": self.chosen_rewrite_action,
                "rewrite_strategy": REWRITE_STRATEGIES[self.chosen_rewrite_action],
                "rewritten_question": self.rewrite_cache[key][self.chosen_rewrite_action],
            }

        if self.phase == 1:
            if action >= self.max_hops:
                return self._make_obs(), -1.0, True, False, {"error": "invalid hop action"}
            self.chosen_hop_count = int(action) + 1
            self.phase = 2
            gold_hops = len(self.current_example["gold_schema_sequence"])
            hop_reward = 0.2 if self.chosen_hop_count == gold_hops else -0.2
            return self._make_obs(), hop_reward if self.shaped else 0.0, False, False, {
                "phase": "hop_count",
                "chosen_hop_count": self.chosen_hop_count,
                "gold_hop_count": gold_hops,
            }

        valid_actions = self.get_valid_actions()
        if action not in valid_actions:
            return self._make_obs(), -1.0, True, False, {"error": "invalid path action"}
        pred = self.schema_paths_by_hop[self.chosen_hop_count][int(action)]
        reward, reward_info = self._score_path(pred)
        return self._make_obs(), reward, True, False, {
            "phase": "schema_path",
            "rewrite_action": self.chosen_rewrite_action,
            "rewrite_strategy": REWRITE_STRATEGIES[self.chosen_rewrite_action],
            "chosen_hop_count": self.chosen_hop_count,
            "predicted_sequence": list(pred),
            "gold_sequence": self.current_example["gold_schema_sequence"],
            **reward_info,
        }


def make_supervised_transitions_schema_path(examples, env_factory):
    transitions = []
    for ex in examples:
        for rewrite_action in range(len(REWRITE_STRATEGIES)):
            env = env_factory([ex])
            obs, _ = env.reset(seed=0)
            transitions.append((obs, rewrite_action, get_valid_actions_multihop(env)))

            obs, _, _, _, _ = env.step(rewrite_action)
            hop_action = len(ex["gold_schema_sequence"]) - 1
            transitions.append((obs, hop_action, get_valid_actions_multihop(env)))

            obs, _, _, _, _ = env.step(hop_action)
            gold_path = tuple(ex["gold_schema_sequence"])
            path_action = env.schema_paths_by_hop[len(gold_path)].index(gold_path)
            transitions.append((obs, path_action, get_valid_actions_multihop(env)))
    return transitions


def get_valid_actions_multihop(env):
    if hasattr(env, "get_valid_actions"):
        return env.get_valid_actions()
    if env.step_id == 0:
        return list(range(env.n_rewrite_actions))
    if len(env.selected_schema_sequence) == 0:
        return list(range(env.n_schema_patterns))
    return list(range(env.n_schema_patterns)) + [env.stop_action]


def masked_policy_probs(theta, obs, valid_actions):
    logits = obs @ theta
    masked_logits = np.full_like(logits, -1e9)
    masked_logits[valid_actions] = logits[valid_actions]
    masked_logits = masked_logits - np.max(masked_logits)
    exp_logits = np.exp(masked_logits)
    return exp_logits / np.sum(exp_logits)


def supervised_pretrain_policy(transitions, obs_dim, n_actions, n_epochs=20, alpha=0.03, seed=0):
    rng = np.random.default_rng(seed)
    theta = rng.normal(0, 0.01, size=(obs_dim, n_actions))
    losses = []
    for _ in range(n_epochs):
        rng.shuffle(transitions)
        total_loss = 0.0
        for obs, target_action, valid_actions in transitions:
            probs = masked_policy_probs(theta, obs, valid_actions)
            total_loss += -np.log(probs[target_action] + 1e-8)
            grad_log = -np.outer(obs, probs)
            grad_log[:, target_action] += obs
            theta += alpha * grad_log
        losses.append(total_loss / len(transitions))
    return theta, np.array(losses)


def make_supervised_transitions_all_rewrites(examples, env_factory):
    transitions = []
    for ex in examples:
        for rewrite_action in range(len(REWRITE_STRATEGIES)):
            env = env_factory([ex])
            obs, _ = env.reset(seed=0)
            transitions.append((obs, rewrite_action, get_valid_actions_multihop(env)))
            obs, _, terminated, truncated, _ = env.step(rewrite_action)
            for pattern_id in ex["gold_schema_sequence"]:
                transitions.append((obs, pattern_id, get_valid_actions_multihop(env)))
                obs, _, terminated, truncated, _ = env.step(pattern_id)
                if terminated or truncated:
                    break
            if not (terminated or truncated):
                transitions.append((obs, env.stop_action, get_valid_actions_multihop(env)))
    return transitions


def make_supervised_transitions_hierarchical(examples, env_factory):
    transitions = []
    for ex in examples:
        for rewrite_action in range(len(REWRITE_STRATEGIES)):
            env = env_factory([ex])
            obs, _ = env.reset(seed=0)
            transitions.append((obs, rewrite_action, get_valid_actions_multihop(env)))

            obs, _, terminated, truncated, _ = env.step(rewrite_action)
            hop_action = len(ex["gold_schema_sequence"]) - 1
            transitions.append((obs, hop_action, get_valid_actions_multihop(env)))

            obs, _, terminated, truncated, _ = env.step(hop_action)
            for pattern_id in ex["gold_schema_sequence"]:
                transitions.append((obs, pattern_id, get_valid_actions_multihop(env)))
                obs, _, terminated, truncated, _ = env.step(pattern_id)
                if terminated or truncated:
                    break
    return transitions


def train_reinforce(env, n_episodes=10000, alpha=0.001, baseline_alpha=0.01, gamma=0.99, seed=0, initial_theta=None):
    rng = np.random.default_rng(seed)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    theta = rng.normal(0, 0.01, size=(obs_dim, n_actions)) if initial_theta is None else initial_theta.copy()
    baseline = 0.0
    history = {"rewards": [], "f1_scores": [], "exact_matches": [], "sequence_lengths": [], "rewrite_actions": []}

    for _ in range(n_episodes):
        obs, _ = env.reset(seed=int(rng.integers(1_000_000)))
        trajectory = []
        episode_reward = 0.0
        terminated = False
        truncated = False
        final_info = {}

        while not (terminated or truncated):
            valid_actions = get_valid_actions_multihop(env)
            probs = masked_policy_probs(theta, obs, valid_actions)
            action = rng.choice(n_actions, p=probs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            trajectory.append((obs, action, probs, reward))
            episode_reward += reward
            obs = next_obs
            final_info = info

        returns = []
        g = 0.0
        for _, _, _, r in reversed(trajectory):
            g = r + gamma * g
            returns.append(g)
        returns = list(reversed(returns))

        baseline += baseline_alpha * (returns[0] - baseline)
        for (obs_t, action_t, probs_t, _), g_t in zip(trajectory, returns):
            advantage = g_t - baseline
            grad_log = -np.outer(obs_t, probs_t)
            grad_log[:, action_t] += obs_t
            theta += alpha * advantage * grad_log

        history["rewards"].append(episode_reward)
        history["f1_scores"].append(final_info.get("f1", 0.0))
        history["exact_matches"].append(float(final_info.get("exact_match", False)))
        history["sequence_lengths"].append(len(final_info.get("predicted_sequence", [])))
        history["rewrite_actions"].append(final_info.get("rewrite_action"))

    return theta, {k: np.array(v) if k != "rewrite_actions" else v for k, v in history.items()}


def train_actor_critic(
    env,
    n_episodes=5000,
    actor_alpha=0.0005,
    critic_alpha=0.01,
    gamma=0.99,
    seed=0,
    initial_theta=None,
):
    rng = np.random.default_rng(seed)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    theta = rng.normal(0, 0.01, size=(obs_dim, n_actions)) if initial_theta is None else initial_theta.copy()
    value_w = np.zeros(obs_dim, dtype=np.float32)
    history = {"rewards": [], "f1_scores": [], "exact_matches": [], "sequence_lengths": [], "rewrite_actions": []}

    for _ in range(n_episodes):
        obs, _ = env.reset(seed=int(rng.integers(1_000_000)))
        episode_reward = 0.0
        terminated = False
        truncated = False
        final_info = {}

        while not (terminated or truncated):
            valid_actions = get_valid_actions_multihop(env)
            probs = masked_policy_probs(theta, obs, valid_actions)
            action = int(rng.choice(n_actions, p=probs))
            next_obs, reward, terminated, truncated, info = env.step(action)

            value = float(obs @ value_w)
            next_value = 0.0 if (terminated or truncated) else float(next_obs @ value_w)
            td_error = reward + gamma * next_value - value

            value_w += critic_alpha * td_error * obs

            grad_log = -np.outer(obs, probs)
            grad_log[:, action] += obs
            theta += actor_alpha * td_error * grad_log

            episode_reward += reward
            obs = next_obs
            final_info = info

        history["rewards"].append(episode_reward)
        history["f1_scores"].append(final_info.get("f1", 0.0))
        history["exact_matches"].append(float(final_info.get("exact_match", False)))
        history["sequence_lengths"].append(len(final_info.get("predicted_sequence", [])))
        history["rewrite_actions"].append(final_info.get("rewrite_action"))

    return theta, value_w, {k: np.array(v) if k != "rewrite_actions" else v for k, v in history.items()}


def evaluate_policy(env, theta, n_eval=1000, seed=0):
    rng = np.random.default_rng(seed)
    rewards = []
    f1_scores = []
    exact_matches = []
    sequence_lengths = []
    rewrite_actions = []
    predicted_sequences = []
    gold_sequences = []

    for _ in range(n_eval):
        obs, _ = env.reset(seed=int(rng.integers(1_000_000)))
        terminated = False
        truncated = False
        final_info = {}
        while not (terminated or truncated):
            valid_actions = get_valid_actions_multihop(env)
            probs = masked_policy_probs(theta, obs, valid_actions)
            action = int(np.argmax(probs))
            obs, reward, terminated, truncated, info = env.step(action)
            final_info = info

        rewards.append(reward)
        f1_scores.append(final_info.get("f1", 0.0))
        exact_matches.append(float(final_info.get("exact_match", False)))
        sequence_lengths.append(len(final_info.get("predicted_sequence", [])))
        rewrite_actions.append(final_info.get("rewrite_action"))
        predicted_sequences.append(tuple(final_info.get("predicted_sequence", [])))
        gold_sequences.append(tuple(final_info.get("gold_sequence", [])))

    return {
        "mean_reward": float(np.mean(rewards)),
        "mean_f1": float(np.mean(f1_scores)),
        "exact_match": float(np.mean(exact_matches)),
        "avg_sequence_length": float(np.mean(sequence_lengths)),
        "rewrite_actions": Counter(rewrite_actions),
        "predicted_sequences": Counter(predicted_sequences),
        "gold_sequences": Counter(gold_sequences),
    }


def evaluate_fixed_rewrite_policy(env_factory, examples, theta, rewrite_action, n_eval=1000, seed=0):
    rng = np.random.default_rng(seed)
    rewards = []
    f1_scores = []
    exact_matches = []
    sequence_lengths = []

    for _ in range(n_eval):
        env = env_factory(examples)
        obs, _ = env.reset(seed=int(rng.integers(1_000_000)))
        obs, _, terminated, truncated, _ = env.step(rewrite_action)
        final_info = {}
        while not (terminated or truncated):
            valid_actions = get_valid_actions_multihop(env)
            probs = masked_policy_probs(theta, obs, valid_actions)
            action = int(np.argmax(probs))
            obs, reward, terminated, truncated, info = env.step(action)
            final_info = info
        rewards.append(reward)
        f1_scores.append(final_info.get("f1", 0.0))
        exact_matches.append(float(final_info.get("exact_match", False)))
        sequence_lengths.append(len(final_info.get("predicted_sequence", [])))

    return {
        "rewrite_strategy": REWRITE_STRATEGIES[rewrite_action],
        "mean_reward": float(np.mean(rewards)),
        "mean_f1": float(np.mean(f1_scores)),
        "exact_match": float(np.mean(exact_matches)),
        "avg_sequence_length": float(np.mean(sequence_lengths)),
    }


def select_subset(examples, n, seed=0, hop_values=(1, 2)):
    candidates = [ex for ex in examples if ex["hop_count"] in hop_values]
    rng = random.Random(seed)
    rng.shuffle(candidates)
    return candidates[:n]


def prepare_v3_data(dataset_path="HETIONET_dataset.json"):
    data = load_hetionet_dataset(dataset_path)
    schema_examples = build_schema_sequence_examples(data)
    augmented = build_augmented_examples(schema_examples)
    train, val, test = split_by_base_id(schema_examples, augmented)
    return data, schema_examples, augmented, train, val, test
