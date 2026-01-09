import argparse
from datetime import datetime
import json
import os
import time

import torch
from pymilvus import MilvusClient, DataType, CollectionSchema, FieldSchema
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

import config


def ensure_dir(path):
    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)


def make_run_dir(output_dir, prefix):
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"{prefix}_{run_id}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def load_processed_data(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Processed data must be a list of dicts.")
    return data


def load_questions(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Questions file must be a list.")
    return data


def normalize_evidence(evidence):
    if isinstance(evidence, list):
        return [e.strip() for e in evidence if isinstance(e, str) and e.strip()]
    if isinstance(evidence, str):
        parts = [p.strip() for p in evidence.split(";") if p.strip()]
        return parts if parts else [evidence.strip()]
    return []


def load_embedding_model(model_name):
    return SentenceTransformer(model_name)


def load_generation_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def get_milvus_client(db_path):
    ensure_dir(db_path)
    return MilvusClient(uri=db_path)


def ensure_collection(client, collection_name, dim, reset=False):
    collections = client.list_collections()
    if collection_name in collections:
        if reset:
            client.drop_collection(collection_name)
        else:
            return
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="content_preview", dtype=DataType.VARCHAR, max_length=500),
    ]
    schema = CollectionSchema(fields, f"GraphBench (dim={dim})")
    client.create_collection(collection_name=collection_name, schema=schema)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        index_type=config.INDEX_TYPE,
        metric_type=config.INDEX_METRIC_TYPE,
        params=config.INDEX_PARAMS,
    )
    client.create_index(collection_name, index_params)


def get_entity_count(client, collection_name):
    try:
        if hasattr(client, "num_entities"):
            return client.num_entities(collection_name)
        stats = client.get_collection_stats(collection_name)
        return int(stats.get("row_count", stats.get("rowCount", 0)))
    except Exception:
        return 0


def build_docs(data):
    docs = []
    for doc in data:
        title = doc.get("title", "") or ""
        abstract = doc.get("abstract", "") or ""
        content = f"Title: {title}\nAbstract: {abstract}".strip()
        if not content:
            continue
        docs.append({
            "title": title,
            "abstract": abstract,
            "content": content,
        })
    return docs


def index_documents(client, collection_name, docs, embedding_model, batch_size=128):
    current_count = get_entity_count(client, collection_name)
    if current_count >= len(docs):
        return

    for start in range(0, len(docs), batch_size):
        batch = docs[start:start + batch_size]
        texts = [d["content"] for d in batch]
        embeddings = embedding_model.encode(texts, show_progress_bar=False)
        data_to_insert = []
        for i, emb in enumerate(embeddings):
            doc_id = start + i
            data_to_insert.append({
                "id": doc_id,
                "embedding": emb,
                "content_preview": texts[i][:500],
            })
        client.insert(collection_name=collection_name, data=data_to_insert)


def extract_hit_value(hit, key, fallback=None):
    if isinstance(hit, dict):
        return hit.get(key, fallback)
    if hasattr(hit, key):
        return getattr(hit, key)
    return fallback


def search_similar_documents(client, collection_name, query, embedding_model, top_k):
    query_embedding = embedding_model.encode([query])[0]
    search_params = {
        "collection_name": collection_name,
        "data": [query_embedding],
        "anns_field": "embedding",
        "limit": top_k,
        "output_fields": ["id"],
    }
    try:
        res = client.search(**search_params)
    except Exception:
        try:
            res = client.search(**search_params, **config.SEARCH_PARAMS)
        except Exception:
            final_params = dict(search_params)
            if "nprobe" in config.SEARCH_PARAMS:
                final_params["nprobe"] = config.SEARCH_PARAMS["nprobe"]
            res = client.search(**final_params)

    if not res or not res[0]:
        return [], []

    hit_ids = []
    distances = []
    for hit in res[0]:
        hit_id = extract_hit_value(hit, "id", extract_hit_value(hit, "pk"))
        dist = extract_hit_value(hit, "distance")
        if hit_id is not None:
            hit_ids.append(hit_id)
            distances.append(dist)
    return hit_ids, distances


def generate_answer(query, context_docs, gen_model, tokenizer):
    if not context_docs:
        return "I couldn't find relevant documents to answer your question."

    context = "\n\n---\n\n".join([doc["content"] for doc in context_docs])
    prompt = (
        "Based ONLY on the following context documents, answer the user's question.\n"
        "If the answer is not found in the context, state that clearly. Do not make up information.\n\n"
        f"Context Documents:\n{context}\n\n"
        f"User Question: {query}\n\n"
        "Answer:\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(gen_model.device)
    with torch.no_grad():
        outputs = gen_model.generate(
            **inputs,
            max_new_tokens=config.MAX_NEW_TOKENS_GEN,
            temperature=config.TEMPERATURE,
            top_p=config.TOP_P,
            repetition_penalty=config.REPETITION_PENALTY,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    return response.strip()


def main():
    parser = argparse.ArgumentParser(
        description="Run Easy-RAG on GraphRAG-Bench questions and export results."
    )
    parser.add_argument(
        "--processed",
        default="data/graphbench_medical_processed.json",
        help="Processed corpus JSON path.",
    )
    parser.add_argument(
        "--questions",
        default="GraphRAG-Benchmark-main/Datasets/Questions/medical_questions.json",
        help="Questions JSON path.",
    )
    parser.add_argument(
        "--output",
        default="GraphRAG-Benchmark-main/results/easy_rag_medical.json",
        help="Output results JSON path.",
    )
    parser.add_argument(
        "--collection",
        default="graphbench_medical",
        help="Milvus collection name.",
    )
    parser.add_argument(
        "--db_path",
        default="milvus_lite_graphbench.db",
        help="Milvus Lite DB file path.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=config.TOP_K,
        help="Top-k retrieval.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Embedding batch size for indexing.",
    )
    parser.add_argument(
        "--max_questions",
        type=int,
        default=None,
        help="Limit number of questions for a quick run.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop collection before indexing.",
    )
    parser.add_argument(
        "--output_dir",
        default="output",
        help="Directory to save run outputs and logs.",
    )
    args = parser.parse_args()

    run_dir = make_run_dir(args.output_dir, "batch_infer")
    log_lines = []

    def log(message):
        print(message)
        log_lines.append(message)

    processed = load_processed_data(args.processed)
    docs = build_docs(processed)
    if not docs:
        raise ValueError("No valid documents found in processed data.")

    embedding_model = load_embedding_model(config.EMBEDDING_MODEL_NAME)
    embed_dim = embedding_model.get_sentence_embedding_dimension()

    client = get_milvus_client(args.db_path)
    ensure_collection(client, args.collection, embed_dim, reset=args.reset)
    index_documents(
        client,
        args.collection,
        docs,
        embedding_model,
        batch_size=args.batch_size,
    )

    gen_model, tokenizer = load_generation_model(config.GENERATION_MODEL_NAME)

    questions = load_questions(args.questions)
    if args.max_questions:
        questions = questions[:args.max_questions]

    results = []
    start_time = time.time()
    for idx, q in enumerate(questions, 1):
        query = q.get("question", "")
        hit_ids, _ = search_similar_documents(
            client,
            args.collection,
            query,
            embedding_model,
            args.top_k,
        )
        context_docs = [docs[i] for i in hit_ids if 0 <= i < len(docs)]
        answer = generate_answer(query, context_docs, gen_model, tokenizer)
        evidence_list = normalize_evidence(q.get("evidence", ""))
        results.append({
            "id": q.get("id", f"q{idx}"),
            "question": query,
            "source": q.get("source", "Medical"),
            "context": [d["content"] for d in context_docs],
            "evidence": evidence_list,
            "question_type": q.get("question_type", "Fact Retrieval"),
            "generated_answer": answer,
            "ground_truth": q.get("answer", ""),
        })
        if idx % 10 == 0:
            log(f"Processed {idx}/{len(questions)}")

    ensure_dir(args.output)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - start_time
    log(f"Done. Wrote {len(results)} results to {args.output} in {elapsed:.1f}s")

    run_results_path = os.path.join(run_dir, "results.json")
    with open(run_results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    meta = {
        "processed": args.processed,
        "questions": args.questions,
        "output": args.output,
        "output_dir": args.output_dir,
        "collection": args.collection,
        "db_path": args.db_path,
        "top_k": args.top_k,
        "batch_size": args.batch_size,
        "max_questions": args.max_questions,
        "embedding_model": config.EMBEDDING_MODEL_NAME,
        "generation_model": config.GENERATION_MODEL_NAME,
        "docs": len(docs),
        "questions_count": len(questions),
        "elapsed_seconds": round(elapsed, 2),
    }
    with open(os.path.join(run_dir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    log_path = os.path.join(run_dir, "run.log")
    with open(log_path, "w", encoding="utf-8") as f:
        for line in log_lines:
            f.write(line + "\n")

    log(f"Run data saved to {run_dir}")


if __name__ == "__main__":
    main()
