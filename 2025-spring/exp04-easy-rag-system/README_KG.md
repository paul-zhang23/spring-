# 医疗知识图谱RAG系统 🏥

> 集成知识图谱的医疗检索增强生成系统 | 基于GraphRAG-Benchmark医疗数据集

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## ✨ 核心特性

### 🔍 智能检索
- **向量语义检索**: 基于Milvus Lite的高效相似度搜索
- **知识图谱增强**: 自动提取医疗实体和关系
- **混合检索**: 结合向量和图结构的多维检索

### 🕸️ 知识图谱
- **自动构建**: 从2062个医疗问答中提取实体-关系
- **交互可视化**: 基于PyVis的动态图谱展示
- **路径推理**: 多跳实体关系查找
- **统计分析**: 实体类型分布、关系统计

### 🤖 智能问答
- **上下文感知**: 基于检索文档生成准确答案
- **可解释性**: 显示来源文档和相似度分数
- **医疗专业**: 使用Qwen2.5中文友好模型

---

## 🚀 快速开始

### 1. 克隆项目
```bash
cd exp04-easy-rag-system
```

### 2. 安装依赖
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. 启动系统
```bash
# 方式1: 使用启动脚本（推荐）
bash run_kg.sh

# 方式2: 直接运行
streamlit run app_with_kg.py
```

### 4. 访问应用
打开浏览器访问: **http://localhost:8501**

---

## 📊 系统功能

### Tab 1: 💬 智能问答
输入医疗问题，获得基于向量检索和知识图谱增强的答案。

**示例问题**:
- What is the most common type of skin cancer?
- What are the risk factors for basal cell carcinoma?
- How is BCC diagnosed?

### Tab 2: 🕸️ 知识图谱可视化
交互式探索医疗知识图谱，查看实体间的关系。

**功能**:
- 实体搜索
- 邻居可视化（1跳/2跳）
- 实体信息卡片

### Tab 3: 📊 图谱统计
查看知识图谱的统计信息和分析。

**指标**:
- 节点/边数量
- 实体类型分布
- 关系类型Top 10
- 热门实体排行

### Tab 4: 🛤️ 路径查找
查找两个医疗实体之间的知识关联路径。

**应用**:
- 因果推理
- 诊断辅助
- 治疗建议

---

## 📁 项目结构

```
exp04-easy-rag-system/
├── 🌟 app_with_kg.py              # 主应用（带知识图谱）
├── 🌟 kg_builder.py               # 知识图谱构建
├── 🌟 kg_visualizer.py            # 图谱可视化
├── 🌟 README_IMPLEMENTATION.md    # 完整实现文档
├── app.py                         # 原始应用
├── config.py                      # 配置
├── models.py                      # 模型加载
├── milvus_utils.py                # 向量数据库
├── rag_core.py                    # RAG核心
├── requirements.txt               # 依赖
├── run_kg.sh                      # 启动脚本
└── GraphRAG-Benchmark-main/       # 数据集
    └── Datasets/
        ├── Questions/medical_questions.json  # 2062问答
        └── Corpus/medical.json               # 医疗语料
```

---

## 🛠️ 技术栈

| 组件 | 技术 | 版本 |
|------|------|------|
| **Web框架** | Streamlit | 1.32+ |
| **向量数据库** | Milvus Lite | 2.x |
| **图数据库** | NetworkX | 3.x |
| **可视化** | PyVis | 0.3+ |
| **嵌入模型** | all-MiniLM-L6-v2 | 384维 |
| **生成模型** | Qwen2.5-0.5B | 0.5B |
| **深度学习** | PyTorch + Transformers | Latest |

---

## 📊 知识图谱统计

基于GraphRAG-Benchmark医疗数据集构建：

- **实体总数**: ~400-500个
- **关系总数**: ~2000-3000条
- **实体类型**: 7类（疾病、解剖、症状、治疗、诊断、风险、其他）
- **关系类型**: 15+（is_subtype_of, arises_from, risk_factor_for等）
- **数据来源**: 2062个医疗问答对

### 实体类型颜色映射
```
🔴 Disease      疾病
🔵 Anatomy      解剖结构
🟠 Symptom      症状
🟢 Treatment    治疗
🟣 Diagnostic   诊断
🟤 RiskFactor   风险因素
⚪ Other        其他
```

---

## 📖 详细文档

查看 **[README_IMPLEMENTATION.md](README_IMPLEMENTATION.md)** 获取：

- 🏗️ 完整系统架构
- 🔧 技术实现细节
- 💡 使用场景与案例
- 🔬 实验与评估
- 🛠️ 扩展与优化
- 🐛 故障排查
- 🎓 论文写作建议

---

## 🎯 实验指导

### 论文章节建议

#### 第3章: 系统设计与实现
- 3.1 知识图谱构建（实体抽取、关系识别）
- 3.2 知识图谱可视化（PyVis技术）
- 3.3 RAG与知识图谱融合（混合检索）

#### 第5章: 实验结果
- 5.1 有无知识图谱对比实验
- 5.2 不同图谱构建策略对比
- 5.3 案例分析（5-10个代表性问题）

### 创新点
1. 医疗领域特化的知识图谱自动构建
2. 交互式知识图谱可视化系统
3. 混合检索与多跳推理机制

---

## 🐛 常见问题

### Q1: 首次运行很慢？
**A**: 首次需要下载模型（~1GB）和构建知识图谱（1-2分钟），后续会直接加载缓存。

### Q2: 知识图谱未加载？
**A**: 检查`GraphRAG-Benchmark-main/Datasets/Questions/medical_questions.json`是否存在。

### Q3: 内存不足？
**A**: 减小`config.py`中的`MAX_ARTICLES_TO_INDEX`（500→100）或使用CPU模式。

### Q4: PyVis可视化空白？
**A**: 升级PyVis：`pip install --upgrade pyvis`

详细故障排查请查看[实现文档](README_IMPLEMENTATION.md#-故障排查)。

---

## 📚 参考资源

### 论文
- **RAG**: Lewis et al. "Retrieval-Augmented Generation" (NeurIPS 2020)
- **GraphRAG**: "From Local to Global: A Graph RAG Approach" (2024)
- **BioBERT**: Lee et al. "BioBERT" (Bioinformatics 2020)

### 数据集
- **GraphRAG-Benchmark**: https://github.com/HKUDS/GraphRAG-Benchmark

### 工具
- **Streamlit**: https://docs.streamlit.io/
- **Milvus**: https://milvus.io/docs
- **NetworkX**: https://networkx.org/
- **PyVis**: https://pyvis.readthedocs.io/

---

## 🎓 适用课程

- 数据挖掘与知识处理
- 自然语言处理
- 知识图谱技术
- 人工智能应用

---

## 📄 许可证

MIT License

---

## 🤝 贡献

欢迎提交Issue和Pull Request！

---

**最后更新**: 2025-01-07
**版本**: v2.0
**作者**: 数据挖掘与知识处理实验团队

---

## 🌟 截图预览

### 智能问答界面
- 输入医疗问题
- 显示检索文档（带相似度分数）
- AI生成答案
- 性能指标统计

### 知识图谱可视化
- 实体搜索与信息卡片
- 交互式图谱（可拖拽、缩放）
- 实体类型颜色编码
- 关系标签显示

### 图谱统计分析
- 基础统计指标
- 实体类型分布柱状图
- 关系类型Top 10
- 热门实体排行表

### 路径查找
- 起始/目标实体选择
- 路径列表展示
- 路径可视化
- 关系链条显示

---

**立即开始体验医疗知识图谱RAG系统！** 🚀