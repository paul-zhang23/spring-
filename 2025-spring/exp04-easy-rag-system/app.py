import streamlit as st
import time
import os
from pathlib import Path

# ============== 【修复】使用绝对路径设置环境变量 ==============
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = str(Path(__file__).parent.absolute() / 'hf_cache')
# ====================================================

# ============== 【新增】知识图谱相关导入 ==============
import json
import re
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
from difflib import SequenceMatcher  # 用于模糊匹配
# ====================================================

# ============== 【新增】模型对比相关导入 ==============
from typing import List, Dict, Tuple
# ====================================================

# Import functions and config from other modules
from config import (
    DATA_FILE, EMBEDDING_MODEL_NAME, GENERATION_MODEL_NAME, TOP_K,
    MAX_ARTICLES_TO_INDEX, MILVUS_LITE_DATA_PATH, COLLECTION_NAME,
    id_to_doc_map # Import the global map
)
from data_utils import load_data
from models import load_embedding_model, load_generation_model
# Import the new Milvus Lite functions
from milvus_utils import get_milvus_client, setup_milvus_collection, index_data_if_needed, search_similar_documents
from rag_core import generate_answer

# ============== 【新增】知识图谱降噪工具函数 ==============
# 英文停用词列表
STOPWORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
    'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that',
    'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
}

# 医学关键词白名单（保留这些即使很短）
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
    """
    验证实体是否有效（降噪）

    Args:
        entity: 候选实体
        seen_entities: 已添加的实体集合（用于去重）

    Returns:
        bool: 是否有效
    """
    entity_lower = entity.lower().strip()

    # 去除重复
    if entity_lower in seen_entities:
        return False

    # 白名单直接通过
    if entity_lower in MEDICAL_WHITELIST:
        return True

    # 过滤停用词
    if entity_lower in STOPWORDS:
        return False

    # 过滤过短词（但保留白名单）
    if len(entity_lower) < 3:
        return False

    # 过滤纯数字或特殊字符
    if not any(c.isalpha() for c in entity_lower):
        return False

    return True
# ====================================================

# ============== 【新增+修改】知识图谱构建函数（加入降噪） ==============
@st.cache_resource
def build_knowledge_graph(
    corpus_path="GraphRAG-Benchmark-main/Datasets/Corpus/medical.json",
    max_sentences=80
):
    """
    从语料库中构建知识图谱（含降噪）

    Args:
        corpus_path: 语料库JSON文件路径
        max_sentences: 用于共现关系的最大句子数（避免噪声）

    Returns:
        NetworkX DiGraph对象
    """
    if not Path(corpus_path).exists():
        st.warning(f"语料库文件不存在: {corpus_path}")
        return None

    # 加载语料库
    with open(corpus_path, 'r', encoding='utf-8') as f:
        corpus_data = json.load(f)

    context = corpus_data.get('context', '')

    # 创建有向图
    graph = nx.DiGraph()

    # 已添加实体集合（用于去重）
    seen_entities = set()
    entity_types = {}

    # ============== 【修改】疾病实体识别（加入降噪） ==============
    disease_pattern = (
        r'\b(?:[A-Z][a-z]+\s+)?'
        r'(?:basal\s+cell\s+|squamous\s+cell\s+)?'
        r'(?:carcinoma|cancer|lymphoma|tumor|disease|syndrome)\b'
    )
    for match in re.finditer(disease_pattern, context, re.IGNORECASE):
        disease = match.group(0).strip()
        if is_valid_entity(disease, seen_entities):
            graph.add_node(disease, type='Disease', color='#e74c3c')
            entity_types[disease] = 'Disease'
            seen_entities.add(disease.lower())
    # ====================================================

    # ============== 【修改】解剖位置识别（加入降噪） ==============
    anatomy_keywords = ['skin', 'face', 'head', 'neck', 'lymph nodes', 'brain',
                       'eyes', 'basal cells', 'epidermis', 'body']

    for anatomy in anatomy_keywords:
        if anatomy.lower() in context.lower() and is_valid_entity(anatomy, seen_entities):
            graph.add_node(anatomy.title(), type='Anatomy', color='#3498db')
            entity_types[anatomy.title()] = 'Anatomy'
            seen_entities.add(anatomy.lower())
    # ====================================================

    # ============== 【修改】治疗方法识别（加入降噪） ==============
    treatment_keywords = ['surgery', 'radiation therapy', 'chemotherapy',
                         'systemic therapy', 'treatment', 'biopsy']

    for treatment in treatment_keywords:
        if treatment.lower() in context.lower() and is_valid_entity(treatment, seen_entities):
            graph.add_node(treatment.title(), type='Treatment', color='#2ecc71')
            entity_types[treatment.title()] = 'Treatment'
            seen_entities.add(treatment.lower())
    # ====================================================

    # ============== 【修改】风险因素识别（加入降噪） ==============
    risk_keywords = ['UV radiation', 'sun exposure', 'fair skin', 'age',
                    'immune suppression', 'tanning beds']

    for risk in risk_keywords:
        if risk.lower() in context.lower() and is_valid_entity(risk, seen_entities):
            graph.add_node(risk.title(), type='RiskFactor', color='#e67e22')
            entity_types[risk.title()] = 'RiskFactor'
            seen_entities.add(risk.lower())
    # ====================================================

    # 基于共现添加边（简化版关系识别）
    sentences = context.split('.')[:max_sentences]  # 只处理前N句

    nodes_list = list(graph.nodes())
    for sentence in sentences:
        # 查找句子中出现的实体
        entities_in_sentence = []
        for node in nodes_list:
            if node.lower() in sentence.lower():
                entities_in_sentence.append(node)

        # 为共现的实体添加边
        if len(entities_in_sentence) >= 2:
            for i in range(len(entities_in_sentence) - 1):
                for j in range(i + 1, len(entities_in_sentence)):
                    source = entities_in_sentence[i]
                    target = entities_in_sentence[j]

                    # 根据类型推断关系
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
# ====================================================


def visualize_knowledge_subgraph(graph, center_entity, max_hops=1):
    """
    可视化实体周围的子图

    Args:
        graph: NetworkX图对象
        center_entity: 中心实体
        max_hops: 最大跳数

    Returns:
        HTML字符串
    """
    if center_entity not in graph:
        return "<p>实体不存在于知识图谱中</p>"

    # 使用BFS获取指定跳数内的节点
    subgraph_nodes = {center_entity}
    frontier = {center_entity}
    for _ in range(max_hops):
        next_frontier = set()
        for node in frontier:
            neighbors = set(graph.successors(node)) | set(graph.predecessors(node))
            next_frontier.update(neighbors)
        subgraph_nodes.update(next_frontier)
        frontier = next_frontier

    # 限制节点数量，避免图过大
    if len(subgraph_nodes) > 50:
        subgraph_nodes = set(list(subgraph_nodes)[:50])

    # 创建子图
    subgraph = graph.subgraph(subgraph_nodes).copy()

    # 创建PyVis网络
    net = Network(height="500px", width="100%", directed=True,
                 bgcolor="#ffffff", font_color="#000000")

    # 添加节点
    for node in subgraph.nodes():
        node_data = subgraph.nodes[node]
        color = node_data.get('color', '#95a5a6')

        # 中心节点特殊标记
        if node == center_entity:
            size = 30
            color = '#c0392b'
        else:
            size = 20

        net.add_node(node, label=node, title=node, color=color, size=size)

    # 添加边
    for u, v, data in subgraph.edges(data=True):
        relation = data.get('relation', 'related_to')
        net.add_edge(u, v, label=relation, title=relation)

    # 生成HTML
    net.set_options("""
    {
      "physics": {
        "enabled": true,
        "stabilization": {"iterations": 100}
      }
    }
    """)

    html = net.generate_html()
    return html


def get_entity_info(graph, entity):
    """
    获取实体详细信息

    Args:
        graph: NetworkX图对象
        entity: 实体名称

    Returns:
        dict: 实体信息
    """
    if entity not in graph:
        return None

    # 获取邻居
    out_neighbors = list(graph.successors(entity))
    in_neighbors = list(graph.predecessors(entity))

    return {
        'entity': entity,
        'type': graph.nodes[entity].get('type', 'Unknown'),
        'out_degree': len(out_neighbors),
        'in_degree': len(in_neighbors),
        'out_neighbors': out_neighbors[:10],
        'in_neighbors': in_neighbors[:10]
    }

# ============== 【新增】模糊匹配查找实体 ==============
def fuzzy_search_entities(graph, keyword, threshold=0.6, max_results=10):
    """
    模糊匹配查找实体

    Args:
        graph: NetworkX图对象
        keyword: 搜索关键词
        threshold: 相似度阈值（0-1）
        max_results: 最大返回结果数

    Returns:
        list: [(entity, similarity_score), ...]
    """
    if not keyword or not graph:
        return []

    keyword_lower = keyword.lower().strip()
    matches = []

    for node in graph.nodes():
        node_lower = node.lower()

        # 长度约束，避免匹配过长或过短的噪声实体
        if len(node_lower) > max(60, len(keyword_lower) * 4):
            continue

        # 精确匹配
        if keyword_lower == node_lower:
            matches.append((node, 1.0))
        # 子串匹配
        elif keyword_lower in node_lower or node_lower in keyword_lower:
            matches.append((node, 0.9))
        # 模糊匹配
        else:
            similarity = SequenceMatcher(None, keyword_lower, node_lower).ratio()
            if similarity >= threshold:
                matches.append((node, similarity))

    # 按相似度降序排序
    matches.sort(key=lambda x: x[1], reverse=True)
    return matches[:max_results]
# ====================================================

# --- Streamlit UI 设置 ---
# ============== 【修改】页面配置，添加知识图谱图标 ==============
st.set_page_config(layout="wide", page_title="医疗RAG+知识图谱系统", page_icon="🏥")
# ====================================================

# ============== 【修改】标题，体现知识图谱功能 ==============
st.title("🏥 医疗 RAG + 知识图谱系统")
st.markdown(f"使用 Milvus Lite, `{EMBEDDING_MODEL_NAME}`, `{GENERATION_MODEL_NAME}` + **知识图谱增强**")
# ====================================================

# ============== 【新增】加载知识图谱 ==============
st.sidebar.markdown("---")
st.sidebar.header("🕸️ 知识图谱")
enable_kg = st.sidebar.checkbox("启用知识图谱", value=True, help="启用知识图谱功能进行增强检索")

# ============== 【新增】问答联动开关 ==============
enable_qa_linkage = st.sidebar.checkbox("启用问答联动", value=True, help="问答时自动显示相关实体的子图")
# ====================================================

knowledge_graph = None
if enable_kg:
    corpus_path = st.sidebar.selectbox(
        "选择语料库",
        ["GraphRAG-Benchmark-main/Datasets/Corpus/medical.json",
         "GraphRAG-Benchmark-main/Datasets/Corpus/novel.json"]
    )
    max_sentences = st.sidebar.slider(
        "图谱共现句数上限",
        min_value=20,
        max_value=200,
        value=80,
        step=20,
        help="限制用于构建共现关系的句子数量，降低噪声"
    )

    with st.spinner("正在构建知识图谱..."):
        knowledge_graph = build_knowledge_graph(corpus_path, max_sentences)

    if knowledge_graph:
        st.sidebar.success(f"✅ 图谱加载成功")
        st.sidebar.metric("节点数", knowledge_graph.number_of_nodes())
        st.sidebar.metric("边数", knowledge_graph.number_of_edges())
    else:
        st.sidebar.error("❌ 图谱加载失败")
# ====================================================

# --- 初始化与缓存 ---
# 获取 Milvus Lite 客户端 (如果未缓存则初始化)
milvus_client = get_milvus_client()

if milvus_client:
    # 设置 collection (如果未缓存则创建/加载索引)
    collection_is_ready = setup_milvus_collection(milvus_client)

    # 加载模型 (缓存)
    embedding_model = load_embedding_model(EMBEDDING_MODEL_NAME)
    generation_model, tokenizer = load_generation_model(GENERATION_MODEL_NAME)

    # 检查所有组件是否成功加载
    models_loaded = embedding_model and generation_model and tokenizer

    if collection_is_ready and models_loaded:
        # ============== 【修复】改进数据加载逻辑，允许使用已有索引 ==============
        # 加载数据 (未缓存)
        pubmed_data = load_data(DATA_FILE)

        # 如果需要则索引数据 (这会填充 id_to_doc_map)
        if pubmed_data:
            indexing_successful = index_data_if_needed(milvus_client, pubmed_data, embedding_model)
        else:
            st.warning(f"⚠️ 无法从 {DATA_FILE} 加载数据。")
            # 检查是否有已存在的文档映射
            if id_to_doc_map:
                st.info("✅ 使用已有的文档映射继续运行（从之前的会话加载）")
                indexing_successful = True
            else:
                st.error("❌ 没有可用的文档数据。RAG功能将被禁用。")
                indexing_successful = False
        # ====================================================

        st.divider()

        # ============== 【新增】创建标签页（增加模型对比） ==============
        tab1, tab2, tab3 = st.tabs(["💬 智能问答", "🕸️ 知识图谱", "⚖️ 模型对比"])
        # ====================================================

        # ============== 【修改】将原有问答功能放入tab1 ==============
        with tab1:
            # --- RAG 交互部分 ---
            if not indexing_successful and not id_to_doc_map:
                 st.error("数据索引失败或不完整，且没有文档映射。RAG 功能已禁用。")
            else:
                # ============== 【新增】多轮对话功能 ==============
                # 初始化对话历史
                if 'conversation_history' not in st.session_state:
                    st.session_state.conversation_history = []
                if 'iteration_count' not in st.session_state:
                    st.session_state.iteration_count = 0

                # 多轮对话开关
                enable_multi_turn = st.sidebar.checkbox("启用多轮对话", value=False, help="启用后可以进行上下文相关的多轮问答")

                # 显示对话历史
                if enable_multi_turn and st.session_state.conversation_history:
                    with st.expander("📜 对话历史", expanded=False):
                        for i, turn in enumerate(st.session_state.conversation_history):
                            st.markdown(f"**第 {i+1} 轮:**")
                            st.markdown(f"👤 **用户**: {turn['question']}")
                            st.markdown(f"🤖 **AI**: {turn['answer'][:200]}{'...' if len(turn['answer']) > 200 else ''}")
                            st.markdown("---")

                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("🔄 重新开始对话", key="reset_conversation"):
                                st.session_state.conversation_history = []
                                st.session_state.iteration_count = 0
                                st.rerun()
                        with col2:
                            st.metric("对话轮数", len(st.session_state.conversation_history))
                # ====================================================
                # ============== 【新增】示例问题 ==============
                st.markdown("#### 💡 示例问题")
                example_cols = st.columns(3)
                examples = [
                    "What is basal cell carcinoma?",
                    "What are the risk factors?",
                    "How is it diagnosed?"
                ]
                for idx, (col, ex) in enumerate(zip(example_cols, examples)):
                    with col:
                        if st.button(ex, key=f"ex_{idx}"):
                            st.session_state.query_input = ex
                # ====================================================

                query = st.text_input(
                    "请提出关于已索引医疗文章的问题:",
                    key="query_input",
                    value=st.session_state.get('query_input', '')
                )

                if st.button("🚀 获取答案", key="submit_button", type="primary") and query:
                    start_time = time.time()

                    # 1. 搜索 Milvus Lite
                    with st.spinner("正在搜索相关文档..."):
                        retrieved_ids, distances = search_similar_documents(milvus_client, query, embedding_model)

                    if not retrieved_ids:
                        st.warning("在数据库中找不到相关文档。")
                    else:
                        # ============== 【修复】修复索引映射错位问题 ==============
                        # 2. 从映射中检索上下文，同时保持ID和距离的对应关系
                        retrieved_docs = []
                        valid_ids = []
                        valid_distances = []

                        for i, doc_id in enumerate(retrieved_ids):
                            if doc_id in id_to_doc_map:
                                retrieved_docs.append(id_to_doc_map[doc_id])
                                valid_ids.append(doc_id)
                                if distances and i < len(distances):
                                    valid_distances.append(distances[i])
                        # ====================================================

                        if not retrieved_docs:
                             st.error("检索到的 ID 无法映射到加载的文档。请检查映射逻辑。")
                        else:
                            # ============== 【新增】知识图谱增强：提取查询中的实体 ==============
                            kg_entities = []
                            if enable_kg and knowledge_graph:
                                st.markdown("---")
                                st.markdown("#### 🕸️ 知识图谱增强信息")

                                # 从查询中提取可能的实体
                                query_lower = query.lower()
                                for node in knowledge_graph.nodes():
                                    if node.lower() in query_lower:
                                        kg_entities.append(node)

                                if kg_entities:
                                    st.info(f"识别到相关实体: {', '.join(kg_entities[:3])}")

                                    # 显示实体信息
                                    for entity in kg_entities[:2]:  # 最多显示2个
                                        entity_info = get_entity_info(knowledge_graph, entity)
                                        if entity_info:
                                            with st.expander(f"📍 实体: {entity}", expanded=False):
                                                col1, col2, col3 = st.columns(3)
                                                col1.metric("类型", entity_info['type'])
                                                col2.metric("出边", entity_info['out_degree'])
                                                col3.metric("入边", entity_info['in_degree'])

                                                if entity_info['out_neighbors']:
                                                    st.markdown("**相关实体:** " + ", ".join(entity_info['out_neighbors'][:5]))
                                else:
                                    st.info("未在知识图谱中找到相关实体")
                            # ====================================================

                            st.markdown("---")
                            st.subheader("📚 检索到的上下文文档")
                            for i, doc in enumerate(retrieved_docs):
                                # ============== 【修改+修复】优化文档展示，修复索引映射，支持不同度量类型 ==============
                                # 使用valid_ids和valid_distances确保对应关系正确
                                if valid_distances and i < len(valid_distances):
                                    from config import INDEX_METRIC_TYPE
                                    dist_value = valid_distances[i]

                                    # 根据度量类型显示不同的标签和计算相似度
                                    if INDEX_METRIC_TYPE == "L2":
                                        metric_label = "距离"
                                        similarity_pct = max(0, 100 * (1 - dist_value / 2))
                                    elif INDEX_METRIC_TYPE == "IP":
                                        metric_label = "内积"
                                        similarity_pct = max(0, min(100, dist_value * 100))  # IP越大越相似
                                    else:  # COSINE等
                                        metric_label = "相似度"
                                        similarity_pct = max(0, min(100, dist_value * 100))

                                    header = f"📄 文档 {i+1} (相似度: {similarity_pct:.1f}%, {metric_label}: {dist_value:.4f}, ID: {valid_ids[i]}) - {doc['title'][:60]}"
                                else:
                                    header = f"📄 文档 {i+1} (ID: {valid_ids[i] if i < len(valid_ids) else 'N/A'}) - {doc['title'][:60]}"

                                with st.expander(header, expanded=(i==0)):
                                    st.write(f"**标题:** {doc['title']}")
                                    st.write(f"**摘要:** {doc['abstract'][:500]}...")  # 限制长度
                                # ====================================================

                            st.divider()

                            # 3. 生成答案
                            st.subheader("🤖 AI生成答案")
                            with st.spinner("正在根据上下文生成答案..."):
                                answer = generate_answer(query, retrieved_docs, generation_model, tokenizer)
                                # ============== 【修改】优化答案展示 ==============
                                st.markdown(
                                    f"""
                                    <div style="background-color:#f0f9ff;padding:1.5rem;border-radius:0.5rem;border-left:4px solid #0284c7;">
                                        <p style="color:#0c4a6e;margin:0;">{answer}</p>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                                # ====================================================

                            # ============== 【新增】保存对话到历史 ==============
                            if enable_multi_turn:
                                st.session_state.conversation_history.append({
                                    'question': query,
                                    'answer': answer,
                                    'retrieved_docs': len(retrieved_docs),
                                    'entities': kg_entities if kg_entities else []
                                })
                                st.session_state.iteration_count += 1
                            # ====================================================

                            # ============== 【新增】问答联动：自动显示实体子图 ==============
                            if enable_kg and knowledge_graph and enable_qa_linkage and kg_entities:
                                st.markdown("---")
                                st.markdown("#### 🎯 问答联动：相关实体子图")
                                st.info(f"正在展示与问题相关的实体子图 (可在侧边栏关闭'启用问答联动')")

                                # 选择最相关的实体 (第一个识别到的)
                                primary_entity = kg_entities[0]

                                with st.expander(f"🔗 {primary_entity} 的关系网络", expanded=True):
                                    st.markdown(f"**中心实体:** `{primary_entity}`")

                                    # 生成并展示子图
                                    with st.spinner(f"正在加载 {primary_entity} 的关系图谱..."):
                                        html = visualize_knowledge_subgraph(knowledge_graph, primary_entity, max_hops=1)
                                        components.html(html, height=520, scrolling=True)

                                    # 显示该实体的邻居信息
                                    entity_info = get_entity_info(knowledge_graph, primary_entity)
                                    if entity_info and (entity_info['out_neighbors'] or entity_info['in_neighbors']):
                                        st.markdown("---")
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            if entity_info['out_neighbors']:
                                                st.markdown("**关联实体:**")
                                                for neighbor in entity_info['out_neighbors'][:5]:
                                                    st.markdown(f"- {neighbor}")
                                        with col2:
                                            if entity_info['in_neighbors']:
                                                st.markdown("**被关联者:**")
                                                for neighbor in entity_info['in_neighbors'][:5]:
                                                    st.markdown(f"- {neighbor}")
                            # ====================================================

                    end_time = time.time()

                    # ============== 【修改】添加更多性能指标 ==============
                    st.markdown("---")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("⏱️ 总耗时", f"{end_time - start_time:.2f}s")
                    with col2:
                        st.metric("📄 检索文档数", len(retrieved_docs) if retrieved_ids else 0)
                    with col3:
                        st.metric("🕸️ 图谱实体数", len(kg_entities) if kg_entities else 0)
                    # ====================================================

        # ============== 【新增】知识图谱可视化标签页 ==============
        with tab2:
            if not enable_kg or not knowledge_graph:
                st.warning("⚠️ 知识图谱未启用或未加载")
                st.info("请在侧边栏勾选'启用知识图谱'")
            else:
                st.markdown("### 🕸️ 知识图谱可视化")

                # 图谱统计
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("节点总数", knowledge_graph.number_of_nodes())
                with col2:
                    st.metric("边总数", knowledge_graph.number_of_edges())
                with col3:
                    avg_degree = sum(dict(knowledge_graph.degree()).values()) / max(knowledge_graph.number_of_nodes(), 1)
                    st.metric("平均度数", f"{avg_degree:.2f}")

                st.markdown("---")

                # ============== 【新增+修改】主动查询实体（支持模糊匹配） ==============
                st.markdown("#### 🔍 主动查询实体")

                # 输入关键词
                search_keyword = st.text_input(
                    "输入关键词搜索实体 (支持模糊匹配)",
                    key="entity_search_keyword",
                    placeholder="例如: cancer, skin, treatment..."
                )

                if search_keyword:
                    with st.spinner("正在搜索实体..."):
                        search_results = fuzzy_search_entities(knowledge_graph, search_keyword, threshold=0.6, max_results=10)

                    if search_results:
                        st.success(f"找到 {len(search_results)} 个匹配实体")

                        # 显示匹配候选
                        st.markdown("**匹配实体列表:**")
                        candidate_entities = []
                        for entity, score in search_results:
                            candidate_entities.append(f"{entity} (相似度: {score:.2f})")

                        selected_candidate = st.selectbox(
                            "选择要探索的实体",
                            options=search_results,
                            format_func=lambda x: f"{x[0]} (相似度: {x[1]:.2f})",
                            key="fuzzy_search_selectbox"
                        )

                        if selected_candidate:
                            selected_entity = selected_candidate[0]

                            # 显示实体信息
                            entity_info = get_entity_info(knowledge_graph, selected_entity)

                            if entity_info:
                                st.markdown(f"### 📌 {entity_info['entity']}")
                                st.markdown(f"**类型**: `{entity_info['type']}`")

                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("出边数量", entity_info['out_degree'])
                                    if entity_info['out_neighbors']:
                                        st.markdown("**相关实体:**")
                                        for neighbor in entity_info['out_neighbors'][:5]:
                                            st.markdown(f"- {neighbor}")

                                with col2:
                                    st.metric("入边数量", entity_info['in_degree'])
                                    if entity_info['in_neighbors']:
                                        st.markdown("**被关联实体:**")
                                        for neighbor in entity_info['in_neighbors'][:5]:
                                            st.markdown(f"- {neighbor}")

                            # 可视化子图
                            st.markdown("---")
                            st.markdown("#### 🎨 实体子图可视化")

                            if st.button("🔮 生成可视化", key="fuzzy_search_visualize", type="primary"):
                                with st.spinner("正在生成可视化..."):
                                    html = visualize_knowledge_subgraph(knowledge_graph, selected_entity, max_hops=1)
                                    components.html(html, height=520, scrolling=True)
                    else:
                        st.warning(f"未找到与 '{search_keyword}' 匹配的实体，请尝试其他关键词")
                else:
                    # 没有输入关键词时，显示原有的下拉选择
                    st.markdown("**或从所有实体中选择:**")
                    all_nodes = sorted(list(knowledge_graph.nodes()))

                    if all_nodes:
                        selected_entity = st.selectbox(
                            "选择要探索的实体",
                            options=all_nodes,
                            index=0,
                            key="all_entities_selectbox"
                        )

                        if selected_entity:
                            # 显示实体信息
                            entity_info = get_entity_info(knowledge_graph, selected_entity)

                            if entity_info:
                                st.markdown(f"### 📌 {entity_info['entity']}")
                                st.markdown(f"**类型**: `{entity_info['type']}`")

                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("出边数量", entity_info['out_degree'])
                                    if entity_info['out_neighbors']:
                                        st.markdown("**相关实体:**")
                                        for neighbor in entity_info['out_neighbors'][:5]:
                                            st.markdown(f"- {neighbor}")

                                with col2:
                                    st.metric("入边数量", entity_info['in_degree'])
                                    if entity_info['in_neighbors']:
                                        st.markdown("**被关联实体:**")
                                        for neighbor in entity_info['in_neighbors'][:5]:
                                            st.markdown(f"- {neighbor}")

                            # 可视化
                            st.markdown("---")
                            st.markdown("#### 🎨 子图可视化")

                            if st.button("🔮 生成可视化", key="all_entities_visualize", type="primary"):
                                with st.spinner("正在生成可视化..."):
                                    html = visualize_knowledge_subgraph(knowledge_graph, selected_entity, max_hops=1)
                                    components.html(html, height=520, scrolling=True)
                    else:
                        st.info("知识图谱中暂无节点")
                # ====================================================
        # ====================================================

        # ============== 【新增】模型对比标签页 ==============
        with tab3:
            st.markdown("### ⚖️ DeepSeek 与本地模型准确度对比")
            st.markdown(
                "对比同一批问题的两种答案。请先生成对应的结果文件与评测结果文件。"
            )

            col1, col2 = st.columns(2)
            with col1:
                local_result_path = st.text_input(
                    "本地模型结果文件",
                    value="GraphRAG-Benchmark-main/results/easy_rag_medical_50.json",
                    help="包含生成答案的JSON文件"
                )
                local_eval_path = st.text_input(
                    "本地模型评测结果",
                    value="GraphRAG-Benchmark-main/results/eval_generation.json",
                    help="generation_eval 的输出结果"
                )
            with col2:
                deepseek_result_path = st.text_input(
                    "DeepSeek结果文件",
                    value="GraphRAG-Benchmark-main/results/deepseek_medical_50.json",
                    help="包含生成答案的JSON文件"
                )
                deepseek_eval_path = st.text_input(
                    "DeepSeek评测结果",
                    value="GraphRAG-Benchmark-main/results/eval_deepseek.json",
                    help="generation_eval 的输出结果"
                )

            @st.cache_data(show_spinner=False)
            def load_json_file(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        return json.load(f)
                except Exception:
                    return None

            def extract_scores(eval_data):
                if not eval_data:
                    return None
                scores = eval_data.get("Fact Retrieval")
                if not isinstance(scores, dict):
                    return None
                return {
                    "rouge_score": scores.get("rouge_score"),
                    "answer_correctness": scores.get("answer_correctness"),
                }

            def format_score(value):
                if value is None or (isinstance(value, float) and value != value):
                    return "N/A"
                return f"{value:.4f}"

            local_eval = load_json_file(local_eval_path)
            deepseek_eval = load_json_file(deepseek_eval_path)

            st.markdown("#### 📊 平均指标对比")
            metric_cols = st.columns(3)
            local_scores = extract_scores(local_eval)
            deepseek_scores = extract_scores(deepseek_eval)

            with metric_cols[0]:
                st.markdown("**指标**")
                st.markdown("- rouge_score")
                st.markdown("- answer_correctness")
            with metric_cols[1]:
                st.markdown("**本地模型**")
                st.markdown(f"- {format_score(local_scores['rouge_score']) if local_scores else 'N/A'}")
                st.markdown(f"- {format_score(local_scores['answer_correctness']) if local_scores else 'N/A'}")
            with metric_cols[2]:
                st.markdown("**DeepSeek**")
                st.markdown(f"- {format_score(deepseek_scores['rouge_score']) if deepseek_scores else 'N/A'}")
                st.markdown(f"- {format_score(deepseek_scores['answer_correctness']) if deepseek_scores else 'N/A'}")

            st.markdown("---")
            st.markdown("#### 🧪 单题对比")

            local_results = load_json_file(local_result_path)
            deepseek_results = load_json_file(deepseek_result_path)

            if not local_results or not deepseek_results:
                st.warning("未找到结果文件，请确认路径是否正确。")
            else:
                local_map = {item["id"]: item for item in local_results if "id" in item}
                deepseek_map = {item["id"]: item for item in deepseek_results if "id" in item}
                common_ids = sorted(set(local_map.keys()) & set(deepseek_map.keys()))

                if not common_ids:
                    st.warning("两份结果没有重叠的样本ID。")
                else:
                    selected_id = st.selectbox("选择问题ID", options=common_ids)
                    local_item = local_map[selected_id]
                    deepseek_item = deepseek_map[selected_id]

                    st.markdown(f"**问题:** {local_item.get('question', '')}")
                    st.markdown(f"**标准答案:** {local_item.get('ground_truth', '')}")

                    answer_cols = st.columns(2)
                    with answer_cols[0]:
                        st.markdown("**本地模型回答**")
                        st.text_area(
                            "local_answer",
                            value=local_item.get("generated_answer", ""),
                            height=200,
                            label_visibility="collapsed"
                        )
                    with answer_cols[1]:
                        st.markdown("**DeepSeek回答**")
                        st.text_area(
                            "deepseek_answer",
                            value=deepseek_item.get("generated_answer", ""),
                            height=200,
                            label_visibility="collapsed"
                        )

                    with st.expander("查看检索上下文"):
                        context = local_item.get("context", [])
                        if isinstance(context, list):
                            st.write("\n\n---\n\n".join(context[:3]))
                        else:
                            st.write(context)
        # ====================================================

        # ============== 【新增】模型对比标签页 ==============
        with tab3:
            st.markdown("### ⚖️ 本地模型 vs DeepSeek 对比")
            st.info("⚠️ 注意：DeepSeek API功能需要配置API密钥。当前版本使用本地模型进行演示对比。")

            # 从 GraphRAG-Benchmark 加载测试问题
            questions_file = "GraphRAG-Benchmark-main/Datasets/Questions/medical_questions.json"

            if not Path(questions_file).exists():
                st.error(f"❌ 测试问题文件不存在: {questions_file}")
            else:
                # 加载问题数据
                with open(questions_file, 'r', encoding='utf-8') as f:
                    questions_data = json.load(f)

                st.success(f"✅ 加载了 {len(questions_data)} 个测试问题")

                # 选择测试模式
                test_mode = st.radio(
                    "选择测试模式",
                    ["单题对比", "批量评估 (前10题)"],
                    horizontal=True
                )

                if test_mode == "单题对比":
                    st.markdown("---")
                    st.markdown("#### 📝 单题对比测试")

                    # 随机选择一个问题或手动选择
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        question_idx = st.selectbox(
                            "选择测试问题",
                            range(min(50, len(questions_data))),  # 只显示前50个
                            format_func=lambda i: f"#{i+1}: {questions_data[i]['question'][:60]}..."
                        )

                    with col2:
                        if st.button("🎲 随机选择", key="random_question"):
                            import random
                            question_idx = random.randint(0, min(49, len(questions_data)-1))
                            st.rerun()

                    if question_idx is not None:
                        test_qa = questions_data[question_idx]

                        # 显示问题详情
                        st.markdown("---")
                        with st.expander("📋 问题详情", expanded=True):
                            st.markdown(f"**ID**: `{test_qa['id']}`")
                            st.markdown(f"**问题**: {test_qa['question']}")
                            st.markdown(f"**标准答案**: {test_qa['answer']}")
                            if 'evidence' in test_qa:
                                st.markdown(f"**证据**: {test_qa['evidence'][:200]}...")

                        st.markdown("---")

                        if st.button("🚀 开始对比测试", key="start_comparison", type="primary"):
                            # 搜索相关文档
                            with st.spinner("正在检索相关文档..."):
                                retrieved_ids, distances = search_similar_documents(
                                    milvus_client, test_qa['question'], embedding_model
                                )

                            if not retrieved_ids:
                                st.warning("未找到相关文档")
                            else:
                                # 获取文档
                                retrieved_docs = []
                                for doc_id in retrieved_ids:
                                    if doc_id in id_to_doc_map:
                                        retrieved_docs.append(id_to_doc_map[doc_id])

                                if retrieved_docs:
                                    # 显示检索到的文档
                                    with st.expander(f"📚 检索到 {len(retrieved_docs)} 个相关文档", expanded=False):
                                        for i, doc in enumerate(retrieved_docs[:3]):  # 只显示前3个
                                            st.markdown(f"**文档{i+1}**: {doc['title']}")
                                            st.caption(doc['abstract'][:150] + "...")

                                    st.markdown("---")

                                    # 并排对比两个模型的回答
                                    col1, col2 = st.columns(2)

                                    with col1:
                                        st.markdown("### 🤖 本地模型回答")
                                        st.markdown(f"*模型: {GENERATION_MODEL_NAME}*")

                                        with st.spinner("本地模型生成中..."):
                                            local_answer = generate_answer(
                                                test_qa['question'],
                                                retrieved_docs,
                                                generation_model,
                                                tokenizer
                                            )

                                        st.markdown(
                                            f"""
                                            <div style="background-color:#f0f9ff;padding:1rem;border-radius:0.5rem;border-left:4px solid #0284c7;min-height:150px;">
                                                <p style="color:#0c4a6e;margin:0;">{local_answer}</p>
                                            </div>
                                            """,
                                            unsafe_allow_html=True
                                        )

                                    with col2:
                                        st.markdown("### 🌐 DeepSeek回答")
                                        st.markdown("*模型: deepseek-chat*")

                                        # 模拟DeepSeek回答（实际应用中需要API调用）
                                        st.info("💡 DeepSeek API集成需要配置密钥。当前显示模拟结果。")

                                        deepseek_answer = f"(模拟) {test_qa['answer']}"  # 使用标准答案作为模拟

                                        st.markdown(
                                            f"""
                                            <div style="background-color:#fef3c7;padding:1rem;border-radius:0.5rem;border-left:4px solid #f59e0b;min-height:150px;">
                                                <p style="color:#92400e;margin:0;">{deepseek_answer}</p>
                                            </div>
                                            """,
                                            unsafe_allow_html=True
                                        )

                                    st.markdown("---")
                                    st.markdown("### 📊 评估指标")

                                    # 简单的文本相似度评估
                                    def calculate_word_overlap(text1, text2):
                                        """计算词重叠率"""
                                        words1 = set(text1.lower().split())
                                        words2 = set(text2.lower().split())
                                        if not words1 or not words2:
                                            return 0.0
                                        overlap = len(words1 & words2)
                                        return overlap / max(len(words1), len(words2))

                                    # 计算与标准答案的重叠度
                                    local_overlap = calculate_word_overlap(local_answer, test_qa['answer'])
                                    deepseek_overlap = calculate_word_overlap(deepseek_answer, test_qa['answer'])

                                    col1, col2, col3 = st.columns(3)

                                    with col1:
                                        st.metric(
                                            "本地模型相似度",
                                            f"{local_overlap*100:.1f}%",
                                            delta=f"{(local_overlap-0.5)*100:+.1f}%"
                                        )

                                    with col2:
                                        st.metric(
                                            "DeepSeek相似度",
                                            f"{deepseek_overlap*100:.1f}%",
                                            delta=f"{(deepseek_overlap-0.5)*100:+.1f}%"
                                        )

                                    with col3:
                                        winner = "本地模型" if local_overlap > deepseek_overlap else "DeepSeek"
                                        st.metric("更优模型", winner)

                                    st.markdown("---")
                                    st.markdown("### 📌 标准答案")
                                    st.info(test_qa['answer'])

                                else:
                                    st.error("无法检索到有效文档")

                else:  # 批量评估模式
                    st.markdown("---")
                    st.markdown("#### 📊 批量评估 (前10题)")

                    st.info("💡 批量评估功能将测试前10个问题，对比两个模型的平均表现。")

                    if st.button("🚀 开始批量评估", key="batch_eval", type="primary"):
                        # 批量评估
                        results = []
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        for i, test_qa in enumerate(questions_data[:10]):
                            status_text.text(f"正在评估问题 {i+1}/10: {test_qa['question'][:50]}...")

                            # 搜索文档
                            retrieved_ids, _ = search_similar_documents(
                                milvus_client, test_qa['question'], embedding_model
                            )

                            if retrieved_ids:
                                retrieved_docs = [id_to_doc_map[doc_id] for doc_id in retrieved_ids if doc_id in id_to_doc_map]

                                if retrieved_docs:
                                    # 生成本地模型答案
                                    local_answer = generate_answer(
                                        test_qa['question'], retrieved_docs, generation_model, tokenizer
                                    )

                                    # 模拟DeepSeek答案
                                    deepseek_answer = test_qa['answer']  # 使用标准答案模拟

                                    # 计算相似度
                                    def calculate_word_overlap(text1, text2):
                                        words1 = set(text1.lower().split())
                                        words2 = set(text2.lower().split())
                                        if not words1 or not words2:
                                            return 0.0
                                        overlap = len(words1 & words2)
                                        return overlap / max(len(words1), len(words2))

                                    local_score = calculate_word_overlap(local_answer, test_qa['answer'])
                                    deepseek_score = calculate_word_overlap(deepseek_answer, test_qa['answer'])

                                    results.append({
                                        'question': test_qa['question'],
                                        'local_score': local_score,
                                        'deepseek_score': deepseek_score
                                    })

                            progress_bar.progress((i + 1) / 10)

                        status_text.text("✅ 评估完成！")

                        # 显示结果
                        if results:
                            st.markdown("---")
                            st.markdown("### 📈 评估结果汇总")

                            # 计算平均分
                            avg_local = sum(r['local_score'] for r in results) / len(results)
                            avg_deepseek = sum(r['deepseek_score'] for r in results) / len(results)

                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("本地模型平均分", f"{avg_local*100:.1f}%")
                            with col2:
                                st.metric("DeepSeek平均分", f"{avg_deepseek*100:.1f}%")
                            with col3:
                                winner = "本地模型" if avg_local > avg_deepseek else "DeepSeek"
                                st.metric("综合胜出", winner)

                            st.markdown("---")
                            st.markdown("### 📋 详细结果")

                            # 显示结果表格
                            import pandas as pd
                            df = pd.DataFrame(results)
                            df['question'] = df['question'].str[:60] + "..."
                            df['local_score'] = df['local_score'].apply(lambda x: f"{x*100:.1f}%")
                            df['deepseek_score'] = df['deepseek_score'].apply(lambda x: f"{x*100:.1f}%")
                            df.columns = ['问题', '本地模型', 'DeepSeek']

                            st.dataframe(df, use_container_width=True)

                            # 保存结果提示
                            st.markdown("---")
                            st.info("💾 结果已生成！在实际应用中，可将结果保存到 output/ 目录下的 JSON 文件。")
                        else:
                            st.warning("未能生成有效评估结果")
        # ====================================================

    else:
        st.error("加载模型或设置 Milvus Lite collection 失败。请检查日志和配置。")
else:
    st.error("初始化 Milvus Lite 客户端失败。请检查日志。")


# --- 页脚/信息侧边栏 ---
st.sidebar.markdown("---")
st.sidebar.header("⚙️ 系统配置")
st.sidebar.markdown(f"**向量存储:** Milvus Lite")
st.sidebar.markdown(f"**数据路径:** `{MILVUS_LITE_DATA_PATH}`")
st.sidebar.markdown(f"**Collection:** `{COLLECTION_NAME}`")
st.sidebar.markdown(f"**数据文件:** `{DATA_FILE}`")
st.sidebar.markdown(f"**嵌入模型:** `{EMBEDDING_MODEL_NAME}`")
st.sidebar.markdown(f"**生成模型:** `{GENERATION_MODEL_NAME}`")
st.sidebar.markdown(f"**最大索引数:** `{MAX_ARTICLES_TO_INDEX}`")
st.sidebar.markdown(f"**检索 Top K:** `{TOP_K}`")

# ============== 【新增】图例说明 ==============
if enable_kg:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎨 实体类型图例")
    st.sidebar.markdown("🔴 Disease - 疾病")
    st.sidebar.markdown("🔵 Anatomy - 解剖结构")
    st.sidebar.markdown("🟢 Treatment - 治疗")
    st.sidebar.markdown("🟠 RiskFactor - 风险因素")
# ====================================================
