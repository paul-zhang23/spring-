"""
简单实验脚本：对比有图谱 vs 无图谱的RAG效果
运行方式: python run_experiment.py
"""

import json
import time
from pathlib import Path
from collections import Counter

# 导入系统模块
from config import (
    DATA_FILE, EMBEDDING_MODEL_NAME, GENERATION_MODEL_NAME, TOP_K,
    MILVUS_LITE_DATA_PATH, COLLECTION_NAME, id_to_doc_map
)
from data_utils import load_data
from models import load_embedding_model, load_generation_model
from milvus_utils import get_milvus_client, setup_milvus_collection, index_data_if_needed, search_similar_documents
from rag_core import generate_answer

# 加载知识图谱构建函数
import networkx as nx
import re

# ============== 简化版知识图谱构建（从app.py复制） ==============
STOPWORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
    'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that',
    'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
}

MEDICAL_WHITELIST = {
    'bcc', 'uv', 'ct', 'mri', 'pet', 'dna', 'rna', 'hiv', 'aids', 'copd',
    'skin', 'face', 'head', 'neck', 'eye', 'eyes', 'age', 'body', 'cell',
    'basal cell carcinoma', 'squamous cell carcinoma', 'melanoma',
    'carcinoma', 'cancer', 'lymphoma', 'tumor', 'disease', 'syndrome',
    'surgery', 'radiation therapy', 'chemotherapy', 'systemic therapy',
    'treatment', 'biopsy', 'uv radiation', 'sun exposure', 'fair skin',
    'immune suppression', 'tanning beds', 'lymph nodes', 'brain',
    'basal cells', 'epidermis'
}

def is_valid_entity(entity, seen_entities):
    entity_lower = entity.lower().strip()
    if entity_lower in seen_entities:
        return False
    if entity_lower in MEDICAL_WHITELIST:
        return True
    if entity_lower in STOPWORDS:
        return False
    if len(entity_lower) < 3:
        return False
    if not any(c.isalpha() for c in entity_lower):
        return False
    return True

def build_knowledge_graph(corpus_path="GraphRAG-Benchmark-main/Datasets/Corpus/medical.json", max_sentences=80):
    """构建知识图谱"""
    if not Path(corpus_path).exists():
        print(f"⚠️  语料库文件不存在: {corpus_path}")
        return None

    with open(corpus_path, 'r', encoding='utf-8') as f:
        corpus_data = json.load(f)

    context = corpus_data.get('context', '')
    graph = nx.DiGraph()
    seen_entities = set()
    entity_types = {}

    # 疾病实体识别
    disease_pattern = r'\b(?:[A-Z][a-z]+\s+)?(?:basal\s+cell\s+|squamous\s+cell\s+)?(?:carcinoma|cancer|lymphoma|tumor|disease|syndrome)\b'
    for match in re.finditer(disease_pattern, context, re.IGNORECASE):
        disease = match.group(0).strip()
        if is_valid_entity(disease, seen_entities):
            graph.add_node(disease, type='Disease', color='#e74c3c')
            entity_types[disease] = 'Disease'
            seen_entities.add(disease.lower())

    # 解剖位置识别
    anatomy_keywords = ['skin', 'face', 'head', 'neck', 'lymph nodes', 'brain',
                       'eyes', 'basal cells', 'epidermis', 'body']
    for anatomy in anatomy_keywords:
        if anatomy.lower() in context.lower() and is_valid_entity(anatomy, seen_entities):
            graph.add_node(anatomy.title(), type='Anatomy', color='#3498db')
            entity_types[anatomy.title()] = 'Anatomy'
            seen_entities.add(anatomy.lower())

    # 治疗方法识别
    treatment_keywords = ['surgery', 'radiation therapy', 'chemotherapy',
                         'systemic therapy', 'treatment', 'biopsy']
    for treatment in treatment_keywords:
        if treatment.lower() in context.lower() and is_valid_entity(treatment, seen_entities):
            graph.add_node(treatment.title(), type='Treatment', color='#2ecc71')
            entity_types[treatment.title()] = 'Treatment'
            seen_entities.add(treatment.lower())

    # 风险因素识别
    risk_keywords = ['UV radiation', 'sun exposure', 'fair skin', 'age',
                    'immune suppression', 'tanning beds']
    for risk in risk_keywords:
        if risk.lower() in context.lower() and is_valid_entity(risk, seen_entities):
            graph.add_node(risk.title(), type='RiskFactor', color='#e67e22')
            entity_types[risk.title()] = 'RiskFactor'
            seen_entities.add(risk.lower())

    # 基于共现添加边
    sentences = context.split('.')[:max_sentences]
    nodes_list = list(graph.nodes())

    for sentence in sentences:
        entities_in_sentence = []
        for node in nodes_list:
            if node.lower() in sentence.lower():
                entities_in_sentence.append(node)

        if len(entities_in_sentence) >= 2:
            for i in range(len(entities_in_sentence) - 1):
                for j in range(i + 1, len(entities_in_sentence)):
                    source = entities_in_sentence[i]
                    target = entities_in_sentence[j]

                    source_type = entity_types.get(source, 'Other')
                    target_type = entity_types.get(target, 'Other')

                    if source_type == 'RiskFactor' and target_type == 'Disease':
                        relation = 'risk_factor_for'
                    elif source_type == 'Treatment' and target_type == 'Disease':
                        relation = 'treats'
                    elif source_type == 'Disease' and target_type == 'Anatomy':
                        relation = 'affects'
                    else:
                        relation = 'related_to'

                    graph.add_edge(source, target, relation=relation)

    return graph

def extract_graph_context(query, graph):
    """从知识图谱提取相关上下文"""
    if not graph:
        return ""

    # 识别查询中的实体
    entities_found = []
    for node in graph.nodes():
        if node.lower() in query.lower():
            entities_found.append(node)

    if not entities_found:
        return ""

    # 获取实体邻居
    graph_info = []
    for entity in entities_found[:3]:  # 最多3个实体
        neighbors = list(graph.successors(entity)) + list(graph.predecessors(entity))
        if neighbors:
            graph_info.append(f"Entity '{entity}' is related to: {', '.join(neighbors[:5])}")

    return "\n".join(graph_info) if graph_info else ""

# ============== 简单评估指标 ==============
def calculate_word_overlap(text1, text2):
    """计算词重叠率"""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    if not words1 or not words2:
        return 0.0
    overlap = len(words1 & words2)
    return overlap / max(len(words1), len(words2))

def simple_rouge_1(generated, reference):
    """简化的ROUGE-1分数"""
    gen_words = generated.lower().split()
    ref_words = reference.lower().split()

    if not gen_words or not ref_words:
        return 0.0

    # 计算召回率
    matches = sum(1 for word in ref_words if word in gen_words)
    recall = matches / len(ref_words)

    # 计算精确率
    precision = matches / len(gen_words) if len(gen_words) > 0 else 0

    # F1分数
    if precision + recall == 0:
        return 0.0
    f1 = 2 * (precision * recall) / (precision + recall)

    return f1

# ============== 主实验函数 ==============
def run_experiment(num_questions=10):
    """
    运行对比实验

    Args:
        num_questions: 测试问题数量（默认10题）
    """
    print("=" * 60)
    print("🧪 GraphRAG 对比实验")
    print("=" * 60)

    # 1. 加载测试问题
    questions_file = "GraphRAG-Benchmark-main/Datasets/Questions/medical_questions.json"
    if not Path(questions_file).exists():
        print(f"❌ 测试问题文件不存在: {questions_file}")
        return

    with open(questions_file, 'r', encoding='utf-8') as f:
        questions_data = json.load(f)

    print(f"📋 加载 {len(questions_data)} 个测试问题，将测试前 {num_questions} 题")

    # 2. 初始化模型和数据库
    print("\n🔧 初始化系统组件...")

    print("  - 加载嵌入模型...")
    embedding_model = load_embedding_model()

    print("  - 加载生成模型...")
    generation_model, tokenizer = load_generation_model()

    print("  - 连接向量数据库...")
    milvus_client = get_milvus_client()
    if not milvus_client:
        print("❌ Milvus客户端初始化失败")
        return

    print("  - 设置数据集合...")
    setup_milvus_collection(milvus_client)

    print("  - 加载文档数据...")
    data = load_data()

    print("  - 索引文档（如需要）...")
    index_data_if_needed(milvus_client, data, embedding_model)

    # 3. 构建知识图谱
    print("\n🌐 构建知识图谱...")
    knowledge_graph = build_knowledge_graph()

    if knowledge_graph:
        num_nodes = knowledge_graph.number_of_nodes()
        num_edges = knowledge_graph.number_of_edges()
        print(f"  ✅ 图谱构建完成: {num_nodes} 个节点, {num_edges} 条边")
    else:
        print("  ⚠️  图谱构建失败，将仅测试传统RAG")

    # 4. 运行实验
    print("\n" + "=" * 60)
    print("🚀 开始实验...")
    print("=" * 60)

    results = {
        'traditional_rag': [],  # 无图谱
        'graph_rag': []         # 有图谱
    }

    for i, qa in enumerate(questions_data[:num_questions]):
        print(f"\n📝 问题 {i+1}/{num_questions}: {qa['question'][:60]}...")

        # 检索文档
        start_time = time.time()
        retrieved_ids, distances = search_similar_documents(
            milvus_client, qa['question'], embedding_model
        )
        retrieval_time = time.time() - start_time

        if not retrieved_ids:
            print("  ⚠️  未检索到相关文档，跳过")
            continue

        retrieved_docs = [id_to_doc_map[doc_id] for doc_id in retrieved_ids if doc_id in id_to_doc_map]

        if not retrieved_docs:
            print("  ⚠️  文档映射失败，跳过")
            continue

        # 方案1: 传统RAG（无图谱）
        print("  🔹 传统RAG生成中...")
        start_gen = time.time()
        traditional_answer = generate_answer(
            qa['question'], retrieved_docs, generation_model, tokenizer
        )
        trad_gen_time = time.time() - start_gen

        # 方案2: GraphRAG（有图谱）
        graph_answer = None
        graph_gen_time = 0

        if knowledge_graph:
            print("  🔹 GraphRAG生成中...")

            # 提取图谱上下文
            graph_context = extract_graph_context(qa['question'], knowledge_graph)

            # 增强文档（添加图谱信息）
            enhanced_docs = retrieved_docs.copy()
            if graph_context:
                enhanced_docs.append({
                    'title': 'Knowledge Graph Context',
                    'content': f"Graph Information:\n{graph_context}"
                })

            start_gen = time.time()
            graph_answer = generate_answer(
                qa['question'], enhanced_docs, generation_model, tokenizer
            )
            graph_gen_time = time.time() - start_gen

        # 获取标准答案
        ground_truth = qa.get('evidence', [''])[0] if isinstance(qa.get('evidence'), list) else qa.get('evidence', '')

        # 计算评估指标
        trad_overlap = calculate_word_overlap(traditional_answer, ground_truth)
        trad_rouge = simple_rouge_1(traditional_answer, ground_truth)

        results['traditional_rag'].append({
            'question': qa['question'],
            'answer': traditional_answer,
            'ground_truth': ground_truth,
            'word_overlap': trad_overlap,
            'rouge_1': trad_rouge,
            'retrieval_time': retrieval_time,
            'generation_time': trad_gen_time
        })

        print(f"    传统RAG - 词重叠: {trad_overlap:.2%}, ROUGE-1: {trad_rouge:.3f}")

        if graph_answer:
            graph_overlap = calculate_word_overlap(graph_answer, ground_truth)
            graph_rouge = simple_rouge_1(graph_answer, ground_truth)

            results['graph_rag'].append({
                'question': qa['question'],
                'answer': graph_answer,
                'ground_truth': ground_truth,
                'word_overlap': graph_overlap,
                'rouge_1': graph_rouge,
                'retrieval_time': retrieval_time,
                'generation_time': graph_gen_time
            })

            print(f"    GraphRAG  - 词重叠: {graph_overlap:.2%}, ROUGE-1: {graph_rouge:.3f}")

            # 对比
            if graph_rouge > trad_rouge:
                print(f"    ✅ GraphRAG 更优 (+{(graph_rouge - trad_rouge):.3f})")
            elif graph_rouge < trad_rouge:
                print(f"    ⚠️  传统RAG 更优 (+{(trad_rouge - graph_rouge):.3f})")
            else:
                print(f"    ➖ 两者相同")

    # 5. 汇总结果
    print("\n" + "=" * 60)
    print("📊 实验结果汇总")
    print("=" * 60)

    if results['traditional_rag']:
        trad_avg_overlap = sum(r['word_overlap'] for r in results['traditional_rag']) / len(results['traditional_rag'])
        trad_avg_rouge = sum(r['rouge_1'] for r in results['traditional_rag']) / len(results['traditional_rag'])
        trad_avg_time = sum(r['generation_time'] for r in results['traditional_rag']) / len(results['traditional_rag'])

        print(f"\n🔹 传统RAG (无图谱):")
        print(f"   平均词重叠率: {trad_avg_overlap:.2%}")
        print(f"   平均ROUGE-1:  {trad_avg_rouge:.3f}")
        print(f"   平均生成时间: {trad_avg_time:.2f}s")

    if results['graph_rag']:
        graph_avg_overlap = sum(r['word_overlap'] for r in results['graph_rag']) / len(results['graph_rag'])
        graph_avg_rouge = sum(r['rouge_1'] for r in results['graph_rag']) / len(results['graph_rag'])
        graph_avg_time = sum(r['generation_time'] for r in results['graph_rag']) / len(results['graph_rag'])

        print(f"\n🔹 GraphRAG (有图谱):")
        print(f"   平均词重叠率: {graph_avg_overlap:.2%}")
        print(f"   平均ROUGE-1:  {graph_avg_rouge:.3f}")
        print(f"   平均生成时间: {graph_avg_time:.2f}s")

        # 对比提升
        overlap_improvement = ((graph_avg_overlap - trad_avg_overlap) / trad_avg_overlap * 100) if trad_avg_overlap > 0 else 0
        rouge_improvement = ((graph_avg_rouge - trad_avg_rouge) / trad_avg_rouge * 100) if trad_avg_rouge > 0 else 0

        print(f"\n📈 GraphRAG 相对提升:")
        print(f"   词重叠率: {overlap_improvement:+.1f}%")
        print(f"   ROUGE-1:  {rouge_improvement:+.1f}%")
        print(f"   生成时间: {((graph_avg_time - trad_avg_time) / trad_avg_time * 100):+.1f}%")

    # 6. 保存结果
    output_file = "experiment_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'traditional_rag': results['traditional_rag'],
            'graph_rag': results['graph_rag'],
            'summary': {
                'traditional_rag': {
                    'avg_word_overlap': trad_avg_overlap if results['traditional_rag'] else 0,
                    'avg_rouge_1': trad_avg_rouge if results['traditional_rag'] else 0,
                    'avg_gen_time': trad_avg_time if results['traditional_rag'] else 0
                },
                'graph_rag': {
                    'avg_word_overlap': graph_avg_overlap if results['graph_rag'] else 0,
                    'avg_rouge_1': graph_avg_rouge if results['graph_rag'] else 0,
                    'avg_gen_time': graph_avg_time if results['graph_rag'] else 0
                } if results['graph_rag'] else None
            }
        }, f, indent=2, ensure_ascii=False)

    print(f"\n💾 实验结果已保存至: {output_file}")
    print("\n✅ 实验完成！")

if __name__ == "__main__":
    # 运行实验，测试前10题
    run_experiment(num_questions=10)
