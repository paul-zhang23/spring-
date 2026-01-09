# app.py 修改说明文档

## 📋 修改概述

本次修改在原有 `app.py` 的基础上嵌入了知识图谱功能，**保留了原有架构**，所有修改部分均已使用注释标注。

---

## 🔧 详细修改内容

### 1. 新增导入（第7-15行）

```python
# ============== 【新增】知识图谱相关导入 ==============
import json
import re
from pathlib import Path
import networkx as nx  # 图数据结构
from pyvis.network import Network  # 交互式图可视化
import streamlit.components.v1 as components  # 嵌入HTML
from collections import defaultdict
# ====================================================
```

**说明**: 添加知识图谱构建和可视化所需的库

---

### 2. 新增知识图谱构建函数（第29-132行）

```python
# ============== 【新增】知识图谱构建函数 ==============
@st.cache_resource
def build_knowledge_graph(corpus_path):
    """从 medical.json 或 novel.json 构建知识图谱"""
    # 1. 加载语料库 context 字段
    # 2. 使用正则表达式识别医学实体（疾病、解剖、治疗、风险）
    # 3. 基于共现关系添加边
    # 4. 返回 NetworkX DiGraph 对象
```

**核心逻辑**:
- 从 `GraphRAG-Benchmark-main/Datasets/Corpus/medical.json` 读取 `context` 字段
- 识别4类实体: Disease, Anatomy, Treatment, RiskFactor
- 共现分析建立实体关系

---

### 3. 新增可视化函数（第135-192行）

```python
def visualize_knowledge_subgraph(graph, center_entity, max_hops=1):
    """使用 PyVis 生成交互式子图可视化"""
```

**功能**: 生成以某实体为中心的邻居子图，返回可嵌入的HTML

---

### 4. 新增实体信息查询函数（第195-220行）

```python
def get_entity_info(graph, entity):
    """获取实体的入边/出边邻居信息"""
```

---

### 5. 修改页面配置（第224-231行）

```python
# ============== 【修改】页面配置，添加知识图谱图标 ==============
st.set_page_config(layout="wide", page_title="医疗RAG+知识图谱系统", page_icon="🏥")

# ============== 【修改】标题，体现知识图谱功能 ==============
st.title("🏥 医疗 RAG + 知识图谱系统")
st.markdown(f"使用 Milvus Lite, `{EMBEDDING_MODEL_NAME}`, `{GENERATION_MODEL_NAME}` + **知识图谱增强**")
```

**说明**: 更新标题以体现知识图谱功能

---

### 6. 新增侧边栏知识图谱控制（第233-255行）

```python
# ============== 【新增】加载知识图谱 ==============
st.sidebar.header("🕸️ 知识图谱")
enable_kg = st.sidebar.checkbox("启用知识图谱", value=True)

if enable_kg:
    corpus_path = st.sidebar.selectbox(
        "选择语料库",
        ["GraphRAG-Benchmark-main/Datasets/Corpus/medical.json",
         "GraphRAG-Benchmark-main/Datasets/Corpus/novel.json"]
    )

    knowledge_graph = build_knowledge_graph(corpus_path)

    if knowledge_graph:
        st.sidebar.success(f"✅ 图谱加载成功")
        st.sidebar.metric("节点数", knowledge_graph.number_of_nodes())
        st.sidebar.metric("边数", knowledge_graph.number_of_edges())
```

**功能**:
- 可开关知识图谱功能
- 支持切换 medical 或 novel 语料库
- 实时显示图谱统计

---

### 7. 新增标签页布局（第285-287行）

```python
# ============== 【新增】创建标签页 ==============
tab1, tab2 = st.tabs(["💬 智能问答", "🕸️ 知识图谱"])
```

**说明**: 将界面分为两个标签页，原有问答功能在 tab1

---

### 8. 修改问答界面（第289-406行）

#### 8.1 新增示例问题（第295-307行）

```python
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
```

**功能**: 点击按钮快速填充示例问题

---

#### 8.2 新增知识图谱增强信息（第331-360行）

```python
# ============== 【新增】知识图谱增强：提取查询中的实体 ==============
kg_entities = []
if enable_kg and knowledge_graph:
    st.markdown("#### 🕸️ 知识图谱增强信息")

    # 从查询中识别实体
    query_lower = query.lower()
    for node in knowledge_graph.nodes():
        if node.lower() in query_lower:
            kg_entities.append(node)

    if kg_entities:
        st.info(f"识别到相关实体: {', '.join(kg_entities[:3])}")

        # 显示实体详细信息
        for entity in kg_entities[:2]:
            entity_info = get_entity_info(knowledge_graph, entity)
            with st.expander(f"📍 实体: {entity}"):
                col1, col2, col3 = st.columns(3)
                col1.metric("类型", entity_info['type'])
                col2.metric("出边", entity_info['out_degree'])
                col3.metric("入边", entity_info['in_degree'])
                st.markdown("**相关实体:** " + ", ".join(entity_info['out_neighbors'][:5]))
```

**功能**:
- 自动识别查询中的实体
- 显示实体类型和邻居信息
- 提供额外的上下文知识

---

#### 8.3 修改文档展示（第365-376行）

```python
# ============== 【修改】优化文档展示，添加相似度百分比 ==============
if distances and i < len(distances):
    similarity_pct = max(0, 100 * (1 - distances[i] / 2))
    header = f"📄 文档 {i+1} (相似度: {similarity_pct:.1f}%) - {doc['title'][:60]}"
else:
    header = f"📄 文档 {i+1} - {doc['title'][:60]}"

with st.expander(header, expanded=(i==0)):
    st.write(f"**标题:** {doc['title']}")
    st.write(f"**摘要:** {doc['abstract'][:500]}...")  # 限制长度
```

**改进**:
- 显示相似度百分比（更直观）
- 限制摘要长度避免过长

---

#### 8.4 修改答案展示（第384-393行）

```python
# ============== 【修改】优化答案展示 ==============
st.markdown(
    f"""
    <div style="background-color:#f0f9ff;padding:1.5rem;border-radius:0.5rem;border-left:4px solid #0284c7;">
        <p style="color:#0c4a6e;margin:0;">{answer}</p>
    </div>
    """,
    unsafe_allow_html=True
)
```

**改进**: 使用卡片样式美化答案显示

---

#### 8.5 修改性能指标（第397-406行）

```python
# ============== 【修改】添加更多性能指标 ==============
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("⏱️ 总耗时", f"{end_time - start_time:.2f}s")
with col2:
    st.metric("📄 检索文档数", len(retrieved_docs))
with col3:
    st.metric("🕸️ 图谱实体数", len(kg_entities))
```

**新增**: 显示识别到的图谱实体数量

---

### 9. 新增知识图谱可视化标签页（第408-472行）

```python
# ============== 【新增】知识图谱可视化标签页 ==============
with tab2:
    if not enable_kg or not knowledge_graph:
        st.warning("⚠️ 知识图谱未启用或未加载")
    else:
        st.markdown("### 🕸️ 知识图谱可视化")

        # 显示图谱统计
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("节点总数", knowledge_graph.number_of_nodes())
        with col2:
            st.metric("边总数", knowledge_graph.number_of_edges())
        with col3:
            avg_degree = sum(dict(knowledge_graph.degree()).values()) / max(knowledge_graph.number_of_nodes(), 1)
            st.metric("平均度数", f"{avg_degree:.2f}")

        # 实体搜索
        all_nodes = sorted(list(knowledge_graph.nodes()))
        selected_entity = st.selectbox("选择要探索的实体", options=all_nodes)

        if selected_entity:
            # 显示实体详细信息
            entity_info = get_entity_info(knowledge_graph, selected_entity)
            st.markdown(f"### 📌 {entity_info['entity']}")
            st.markdown(f"**类型**: `{entity_info['type']}`")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("出边数量", entity_info['out_degree'])
                # 显示出边邻居
            with col2:
                st.metric("入边数量", entity_info['in_degree'])
                # 显示入边邻居

            # 生成可视化
            if st.button("🔮 生成可视化", type="primary"):
                html = visualize_knowledge_subgraph(knowledge_graph, selected_entity)
                components.html(html, height=520, scrolling=True)
```

**功能**:
- 图谱统计面板
- 实体搜索与信息展示
- 交互式子图可视化

---

### 10. 新增图例说明（第492-500行）

```python
# ============== 【新增】图例说明 ==============
if enable_kg:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎨 实体类型图例")
    st.sidebar.markdown("🔴 Disease - 疾病")
    st.sidebar.markdown("🔵 Anatomy - 解剖结构")
    st.sidebar.markdown("🟢 Treatment - 治疗")
    st.sidebar.markdown("🟠 RiskFactor - 风险因素")
```

---

## 📊 修改统计

| 类型 | 数量 | 行号范围 |
|------|-----|---------|
| **新增代码** | ~220行 | 7-15, 29-220, 233-255, 285-472, 492-500 |
| **修改代码** | ~30行 | 224-231, 309-313, 365-376, 384-393, 397-406 |
| **保留原有代码** | ~80行 | 其余部分完全保留 |
| **总行数** | 500行 | - |

---

## 🎯 主要功能增强

### 1. 知识图谱构建
- ✅ 从 `medical.json` 或 `novel.json` 的 `context` 字段构建
- ✅ 自动识别4类实体（疾病、解剖、治疗、风险）
- ✅ 基于共现关系建立边
- ✅ 使用 `@st.cache_resource` 缓存图谱

### 2. 知识图谱增强检索
- ✅ 自动识别查询中的实体
- ✅ 显示实体类型和邻居信息
- ✅ 提供额外的上下文知识

### 3. 知识图谱可视化
- ✅ 交互式子图展示（基于PyVis）
- ✅ 实体搜索与详情查看
- ✅ 图谱统计面板

### 4. UI优化
- ✅ 双标签页布局
- ✅ 示例问题快速测试
- ✅ 相似度百分比显示
- ✅ 卡片式答案展示
- ✅ 更多性能指标

---

## 🚀 使用方法

### 1. 确保数据集存在

```bash
# 检查数据集
ls GraphRAG-Benchmark-main/Datasets/Corpus/medical.json
ls GraphRAG-Benchmark-main/Datasets/Corpus/novel.json
```

### 2. 安装依赖（已更新 requirements.txt）

```bash
pip install networkx pyvis
```

### 3. 启动应用

```bash
streamlit run app.py
```

### 4. 使用流程

1. **侧边栏**: 勾选"启用知识图谱"，选择语料库（medical 或 novel）
2. **智能问答标签页**:
   - 输入问题或点击示例问题
   - 查看向量检索结果
   - 查看知识图谱识别的实体
   - 查看AI生成答案
3. **知识图谱标签页**:
   - 查看图谱统计
   - 搜索实体
   - 生成可视化

---

## 🔍 关键技术点

### 1. 知识图谱构建策略

```python
# 实体识别：正则表达式 + 关键词匹配
disease_pattern = r'\b([A-Z][a-z]+\s)?(?:cell\s)?(?:carcinoma|cancer|lymphoma|tumor|disease|syndrome)\b'

# 关系识别：基于共现 + 类型推断
if source_type == 'RiskFactor' and target_type == 'Disease':
    relation = 'risk_factor_for'
elif source_type == 'Treatment' and target_type == 'Disease':
    relation = 'treats'
```

### 2. 缓存优化

```python
@st.cache_resource  # 知识图谱构建结果会被缓存
def build_knowledge_graph(corpus_path):
    ...
```

### 3. 数据源适配

```python
# 直接读取 GraphRAG-Benchmark 的原始数据格式
with open(corpus_path, 'r', encoding='utf-8') as f:
    corpus_data = json.load(f)
context = corpus_data.get('context', '')  # 获取完整语料文本
```

---

## 📝 论文写作建议

### 第3章：系统设计与实现

**3.1 知识图谱构建**
- 介绍实体识别策略（正则+关键词）
- 说明关系识别方法（共现+类型推断）
- 展示代码片段（第29-132行）

**3.2 知识图谱可视化**
- PyVis交互式可视化技术
- 子图生成算法（第135-192行）
- 前端集成方案

**3.3 知识图谱增强RAG**
- 查询实体识别（第337-341行）
- 实体信息展示（第343-357行）
- 与向量检索结合

### 第5章：实验对比

**对比实验：有无知识图谱增强**

| 指标 | 无KG | 有KG | 提升 |
|------|-----|------|------|
| 用户体验 | 基础 | 增强 | 实体识别+可视化 |
| 可解释性 | 低 | 高 | 显示实体关系 |
| 功能完整性 | 单一检索 | 检索+图谱 | 多维信息 |

---

## ✅ 修改验证

所有修改已标注为：
- `# ============== 【新增】... ==============`
- `# ============== 【修改】... ==============`

可通过搜索关键词 `【新增】` 和 `【修改】` 快速定位所有改动。

---

**修改完成日期**: 2025-01-07
**修改版本**: v2.0 (Knowledge Graph Enhanced)