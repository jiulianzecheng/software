# 使用指南

## 目录

1. [安装](#安装)
2. [快速开始](#快速开始)
3. [核心概念](#核心概念)
4. [详细教程](#详细教程)
5. [高级用法](#高级用法)
6. [性能优化](#性能优化)
7. [常见问题](#常见问题)

---

## 安装

### 依赖要求

- Python 3.7+
- NumPy >= 1.21.0
- scikit-learn >= 1.0.0
- scipy >= 1.7.0

### 安装步骤

```bash
cd llm_retrieval_reranking
pip install -r requirements.txt
```

---

## 快速开始

### 最简示例

```python
from llm_retrieval_reranking.core import (
    StreamDataHandler, Retriever, Reranker
)

# 1. 准备数据
data = [
    {'id': 'doc1', 'content': '深度学习使用神经网络'},
    {'id': 'doc2', 'content': '机器学习是AI的分支'},
]

# 2. 流式处理
handler = StreamDataHandler()
docs = handler.process_stream(iter(data))

# 3. 检索
retriever = Retriever()
retriever.index_documents(docs)
results = retriever.retrieve("深度学习", top_k=5)

# 4. 重排序
reranker = Reranker()
final_results = reranker.rerank("深度学习", results, top_k=3)

# 5. 查看结果
for r in final_results:
    print(f"文档: {r.document.content}")
    print(f"分数: {r.score:.3f}")
```

---

## 核心概念

### 1. 流数据处理

流数据处理器负责实时接收和处理数据流，将其转换为可索引的文档格式。

**特点：**
- 支持无限数据流
- 自动缓冲管理
- 批处理优化

**使用场景：**
- 实时日志分析
- 动态知识库更新
- 持续数据摄入

### 2. 检索（Retrieval）

检索阶段快速从大量文档中筛选出可能相关的候选文档。

**算法：**
- TF-IDF向量化
- 余弦相似度计算
- 高效索引结构

**优势：**
- 速度快
- 可扩展
- 召回率高

### 3. 重排序（Reranking）

重排序阶段对检索结果进行精细排序，提升准确性。

**评分维度：**
- **语义相关性**：基于向量相似度
- **关键词匹配**：查询词在文档中的覆盖率
- **文档质量**：长度、完整性、信息密度

**权重配置：**
```python
reranker = Reranker(
    semantic_weight=0.5,   # 语义权重
    keyword_weight=0.3,    # 关键词权重
    quality_weight=0.2     # 质量权重
)
```

### 4. LLM集成

LLM接口模块提供与大语言模型的无缝集成。

**功能：**
- 上下文格式化
- 提示词构建
- RAG支持

---

## 详细教程

### 教程1: 构建知识库问答系统

```python
from llm_retrieval_reranking.core import *

# 步骤1: 准备知识库
knowledge_base = [
    {'id': 'k1', 'content': 'Python是一种高级编程语言'},
    {'id': 'k2', 'content': 'TensorFlow是深度学习框架'},
    {'id': 'k3', 'content': 'PyTorch也是深度学习框架'},
]

# 步骤2: 建立索引
handler = StreamDataHandler()
docs = handler.process_stream(iter(knowledge_base))

retriever = Retriever(embedding_method='tfidf')
retriever.index_documents(docs)

# 步骤3: 用户查询
query = "什么深度学习框架比较流行"

# 步骤4: 检索相关文档
candidates = retriever.retrieve(query, top_k=10)

# 步骤5: 重排序
reranker = Reranker()
top_docs = reranker.rerank(query, candidates, top_k=3)

# 步骤6: 生成LLM提示
llm = LLMInterface()
prompt = llm.build_prompt(query, top_docs)

print("生成的提示:")
print(prompt)
```

### 教程2: 流式数据更新

```python
import time

# 持续监听数据流
def data_generator():
    while True:
        yield {
            'id': f'doc_{time.time()}',
            'content': f'新数据 {time.time()}',
        }
        time.sleep(1)

# 实时索引更新
handler = StreamDataHandler(buffer_size=100)
retriever = Retriever()

batch = []
for data in data_generator():
    doc = handler._parse_item(data)
    batch.append(doc)
    
    # 每10个文档更新一次索引
    if len(batch) >= 10:
        retriever.index_documents(batch)
        batch = []
```

### 教程3: 自定义评分权重

```python
# 针对不同场景调整权重

# 场景1: 学术论文检索（重视质量）
academic_reranker = Reranker(
    semantic_weight=0.3,
    keyword_weight=0.2,
    quality_weight=0.5
)

# 场景2: 新闻搜索（重视关键词）
news_reranker = Reranker(
    semantic_weight=0.2,
    keyword_weight=0.6,
    quality_weight=0.2
)

# 场景3: 语义搜索（重视语义）
semantic_reranker = Reranker(
    semantic_weight=0.7,
    keyword_weight=0.2,
    quality_weight=0.1
)
```

---

## 高级用法

### RAG（检索增强生成）应用

```python
from llm_retrieval_reranking.core import *

# 完整的RAG流程
class RAGSystem:
    def __init__(self):
        self.retriever = Retriever()
        self.reranker = Reranker()
        self.llm = LLMInterface()
    
    def add_documents(self, documents):
        """添加文档到知识库"""
        self.retriever.index_documents(documents)
    
    def query(self, question, top_k=5):
        """查询并生成回答"""
        # 检索
        candidates = self.retriever.retrieve(question, top_k=20)
        
        # 重排序
        top_docs = self.reranker.rerank(question, candidates, top_k=top_k)
        
        # 创建上下文
        context = self.llm.create_rag_context(question, top_docs)
        
        # 构建提示（这里返回提示，实际应用中会调用LLM）
        prompt = self.llm.build_prompt(question, top_docs)
        
        return {
            'prompt': prompt,
            'context': context,
            'top_documents': top_docs
        }

# 使用示例
rag = RAGSystem()
rag.add_documents(docs)
result = rag.query("深度学习是什么？")
```

### 批量处理优化

```python
# 批量检索多个查询
queries = ["查询1", "查询2", "查询3"]
all_results = []

for query in queries:
    results = retriever.retrieve(query, top_k=10)
    reranked = reranker.rerank(query, results, top_k=5)
    all_results.append(reranked)
```

### 配置管理

```python
from llm_retrieval_reranking.utils import Config

# 创建自定义配置
config = Config({
    'retrieval': {
        'embedding_method': 'tfidf',
        'top_k': 30,
    },
    'rerank': {
        'semantic_weight': 0.6,
        'keyword_weight': 0.3,
        'quality_weight': 0.1,
        'top_k': 10,
    }
})

# 使用配置
retrieval_config = config.get_section('retrieval')
retriever = Retriever(embedding_method=retrieval_config['embedding_method'])
```

---

## 性能优化

### 1. 索引优化

```python
# 预先建立索引，避免重复计算
retriever = Retriever()
retriever.index_documents(all_documents)

# 后续查询直接使用索引
results = retriever.retrieve(query)
```

### 2. 批处理

```python
# 批量处理文档
handler = StreamDataHandler(batch_size=100)

# 分批索引
for batch in document_batches:
    retriever.index_documents(batch)
```

### 3. 缓存策略

```python
# 使用缓冲区管理内存
handler = StreamDataHandler(buffer_size=5000)

# 定期清理旧文档
if handler.processed_count > 10000:
    handler.clear_buffer()
```

---

## 常见问题

### Q1: 如何处理中文文本？

A: 系统自动支持中文。对于更好的中文分词，可以考虑集成jieba等分词工具。

### Q2: 如何提高检索准确率？

A: 
1. 调整重排序权重
2. 增加top_k值获取更多候选
3. 使用更高质量的文档

### Q3: 系统支持多大规模的文档库？

A: 当前实现适合中小规模（<100万文档）。对于更大规模，建议：
1. 使用专业的向量数据库（如Faiss, Milvus）
2. 实现分布式索引
3. 添加缓存层

### Q4: 如何集成实际的LLM？

A: LLMInterface提供了标准接口。实际使用时：

```python
class CustomLLM(LLMInterface):
    def generate(self, prompt):
        # 调用实际的LLM API
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

llm = CustomLLM()
prompt = llm.build_prompt(query, top_docs)
answer = llm.generate(prompt)
```

### Q5: 如何评估系统性能？

A: 建议从以下维度评估：
1. **检索召回率**：相关文档是否被检索到
2. **重排序精度**：Top结果的相关性
3. **响应时间**：端到端延迟
4. **系统吞吐**：每秒处理查询数

---

## 更多资源

- [API文档](API.md)
- [示例代码](examples/)
- [测试用例](tests/)

---

## 贡献

欢迎提交Issue和Pull Request！
