import argparse
import json
import os
import xml.etree.ElementTree as ET


def normalize_space(text):
    return " ".join(text.split())


def get_first_text(elem, path):
    node = elem.find(path)
    if node is None:
        return ""
    return normalize_space("".join(node.itertext()))


def extract_title(article_elem):
    return get_first_text(article_elem, ".//ArticleTitle")


def extract_abstract(article_elem):
    abstract_nodes = article_elem.findall(".//Abstract/AbstractText")
    if not abstract_nodes:
        return ""
    parts = []
    for node in abstract_nodes:
        text = normalize_space("".join(node.itertext()))
        if not text:
            continue
        label = node.get("Label")
        if label:
            parts.append(f"{label}: {text}")
        else:
            parts.append(text)
    return "\n".join(parts)


def extract_pmid(article_elem):
    pmid = get_first_text(article_elem, ".//MedlineCitation/PMID")
    if not pmid:
        pmid = get_first_text(article_elem, ".//PMID")
    return pmid


def iter_pubmed_articles(xml_path):
    context = ET.iterparse(xml_path, events=("start", "end"))
    _, root = next(context)
    for event, elem in context:
        if event == "end" and elem.tag == "PubmedArticle":
            yield elem
            root.clear()


def convert(xml_path, output_path, limit=None, min_abstract_len=0):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    entries = []
    count = 0
    kept = 0

    for article in iter_pubmed_articles(xml_path):
        title = extract_title(article)
        abstract = extract_abstract(article)
        pmid = extract_pmid(article)

        if not title and not abstract:
            continue
        if min_abstract_len and len(abstract) < min_abstract_len:
            continue

        entry = {
            "title": title,
            "abstract": abstract,
            "pmid": pmid,
        }
        entries.append(entry)
        kept += 1

        count += 1
        if count % 1000 == 0:
            print(f"Processed {count} articles, kept {kept}")
        if limit and kept >= limit:
            break

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)

    print(f"Done. Kept {kept} articles -> {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert PubMed XML to JSON for the RAG demo."
    )
    parser.add_argument(
        "xml_path",
        nargs="?",
        default="data/pubmed25n0003.xml",
        help="Path to the PubMed XML file.",
    )
    parser.add_argument(
        "--output",
        default="data/processed_data.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of articles to keep.",
    )
    parser.add_argument(
        "--min-abstract-len",
        type=int,
        default=0,
        help="Drop records with shorter abstracts (0 disables).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    convert(
        xml_path=args.xml_path,
        output_path=args.output,
        limit=args.limit,
        min_abstract_len=args.min_abstract_len,
    )


if __name__ == "__main__":
    main()
