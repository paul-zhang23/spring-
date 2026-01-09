import argparse
from datetime import datetime
import json
import os


def chunk_text(text, chunk_size, overlap):
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")
    step = chunk_size - overlap
    chunks = []
    for start in range(0, len(text), step):
        chunk = text[start:start + chunk_size]
        if chunk:
            chunks.append(chunk.strip())
    return [c for c in chunks if c]


def load_corpus(corpus_path):
    with open(corpus_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or "context" not in data:
        raise ValueError("Corpus JSON must be a dict with a 'context' field")
    return data


def build_processed_data(context, title, chunk_size, overlap):
    chunks = chunk_text(context, chunk_size, overlap)
    processed = []
    for chunk in chunks:
        processed.append({
            "title": title,
            "abstract": chunk,
        })
    return processed


def main():
    parser = argparse.ArgumentParser(
        description="Prepare GraphRAG-Bench corpus into processed JSON."
    )
    parser.add_argument(
        "--corpus",
        default="GraphRAG-Benchmark-main/Datasets/Corpus/medical.json",
        help="Path to GraphRAG-Bench corpus JSON.",
    )
    parser.add_argument(
        "--output",
        default="data/graphbench_medical_processed.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=512,
        help="Chunk size in characters.",
    )
    parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=50,
        help="Chunk overlap in characters.",
    )
    parser.add_argument(
        "--output_dir",
        default="output",
        help="Directory to save run outputs and logs.",
    )
    args = parser.parse_args()

    corpus = load_corpus(args.corpus)
    title = corpus.get("corpus_name", "GraphRAG-Bench")
    context = corpus["context"]

    processed = build_processed_data(context, title, args.chunk_size, args.chunk_overlap)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(processed, f, ensure_ascii=False, indent=2)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"prepare_corpus_{run_id}")
    os.makedirs(run_dir, exist_ok=True)
    run_output = os.path.join(run_dir, "processed_data.json")
    with open(run_output, "w", encoding="utf-8") as f:
        json.dump(processed, f, ensure_ascii=False, indent=2)

    log_path = os.path.join(run_dir, "run.log")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"corpus={args.corpus}\n")
        f.write(f"output={args.output}\n")
        f.write(f"output_dir={args.output_dir}\n")
        f.write(f"chunk_size={args.chunk_size}\n")
        f.write(f"chunk_overlap={args.chunk_overlap}\n")
        f.write(f"chunks={len(processed)}\n")

    print(f"Wrote {len(processed)} chunks to {args.output}")
    print(f"Run data saved to {run_dir}")


if __name__ == "__main__":
    main()
