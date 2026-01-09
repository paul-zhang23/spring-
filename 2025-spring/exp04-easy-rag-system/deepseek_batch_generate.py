import argparse
import json
import os
import time
from typing import List, Dict

from openai import OpenAI


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_prompt(question, context_list, context_limit):
    context_text = "\n\n---\n\n".join(context_list or [])
    if context_limit and len(context_text) > context_limit:
        context_text = context_text[:context_limit]
    return (
        "Based ONLY on the following context documents, answer the user's question.\n"
        "If the answer is not found in the context, state that clearly. Do not make up information.\n\n"
        f"Context Documents:\n{context_text}\n\n"
        f"User Question: {question}\n\n"
        "Answer:\n"
    )


def generate_answers(
    input_items,
    client,
    model,
    context_limit,
    sleep_seconds,
    resume_ids,
    output_path,
    max_questions,
    start_index,
):
    results = []
    if resume_ids:
        results.extend(resume_ids.values())

    total = len(input_items)
    processed = 0
    for idx, item in enumerate(input_items):
        if max_questions is not None and processed >= max_questions:
            break
        if idx < start_index:
            continue

        item_id = item.get("id")
        if item_id in resume_ids:
            continue

        question = item.get("question", "")
        context_list = item.get("context", [])
        prompt = build_prompt(question, context_list, context_limit)

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )
            answer = response.choices[0].message.content.strip()
        except Exception as exc:
            answer = f"ERROR: {exc}"

        new_item = dict(item)
        new_item["generated_answer"] = answer
        results.append(new_item)
        processed += 1

        save_json(output_path, results)
        print(f"Processed {processed}/{total}: {item_id}")

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Generate DeepSeek answers based on existing RAG context."
    )
    parser.add_argument(
        "--input",
        default="GraphRAG-Benchmark-main/results/easy_rag_medical_50.json",
        help="Input JSON with context and questions.",
    )
    parser.add_argument(
        "--output",
        default="GraphRAG-Benchmark-main/results/deepseek_medical_50.json",
        help="Output JSON with DeepSeek answers.",
    )
    parser.add_argument(
        "--model",
        default="deepseek-chat",
        help="DeepSeek model name.",
    )
    parser.add_argument(
        "--base_url",
        default="https://api.deepseek.com",
        help="DeepSeek API base URL.",
    )
    parser.add_argument(
        "--api_key_env",
        default="LLM_API_KEY",
        help="Environment variable name for API key.",
    )
    parser.add_argument(
        "--context_limit",
        type=int,
        default=8000,
        help="Max characters of context to send.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Sleep seconds between requests.",
    )
    parser.add_argument(
        "--max_questions",
        type=int,
        default=None,
        help="Limit number of questions to generate.",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Start index in input list.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume if output exists (skip existing IDs).",
    )
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise ValueError(f"Missing API key in env: {args.api_key_env}")

    input_items = load_json(args.input)
    resume_ids = {}
    if args.resume and os.path.exists(args.output):
        existing = load_json(args.output)
        resume_ids = {item.get("id"): item for item in existing if "id" in item}

    client = OpenAI(base_url=args.base_url, api_key=api_key)

    generate_answers(
        input_items=input_items,
        client=client,
        model=args.model,
        context_limit=args.context_limit,
        sleep_seconds=args.sleep,
        resume_ids=resume_ids,
        output_path=args.output,
        max_questions=args.max_questions,
        start_index=args.start_index,
    )


if __name__ == "__main__":
    main()
