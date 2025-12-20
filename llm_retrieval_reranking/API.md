# API 文档

## 核心模块

### StreamDataHandler - 流数据处理器

用于处理实时流入的数据，将其转换为可检索的文档格式。

#### 初始化

```python
from llm_retrieval_reranking.core import StreamDataHandler

handler = StreamDataHandler(
    buffer_size=1000,  # 缓冲区大小
    batch_size=100     # 批处理大小
)
```

#### 主要方法

**add_document(doc: Document) -> None**

添加单个文档到缓冲区。

**process_stream(stream: Iterator[Dict]) -> List[Document]**

处理数据流，返回文档列表。

```python
documents = handler.process_stream(data_stream)
```

**get_recent_documents(n: int = None) -> List[Document]**

获取最近的n个文档。

**get_statistics() -> Dict**

获取处理统计信息。

---

### Retriever - 检索器

实现基于向量相似度的文档检索功能。

#### 初始化

```python
from llm_retrieval_reranking.core import Retriever

retriever = Retriever(
    embedding_method='tfidf'  # 'tfidf' 或 'simple'
)
```

#### 主要方法

**index_documents(documents: List[Document]) -> None**

为文档建立索引。

```python
retriever.index_documents(documents)
```

**retrieve(query: str, documents: List[Document] = None, top_k: int = 10) -> List[RetrievalResult]**

检索相关文档。

```python
results = retriever.retrieve(
    query="深度学习和神经网络",
    top_k=10
)
```

**clear_index() -> None**

清空索引。

**get_index_size() -> int**

获取索引中的文档数量。

---

### Reranker - 重排序器

实现基于多维度特征的文档重排序算法。

#### 初始化

```python
from llm_retrieval_reranking.core import Reranker

reranker = Reranker(
    semantic_weight=0.5,   # 语义相关性权重
    keyword_weight=0.3,    # 关键词匹配权重
    quality_weight=0.2     # 文档质量权重
)
```

#### 主要方法

**rerank(query: str, retrieval_results: List[RetrievalResult], top_k: int = None) -> List[RerankResult]**

对检索结果进行重排序。

```python
reranked = reranker.rerank(
    query="深度学习",
    retrieval_results=results,
    top_k=5
)
```

**set_weights(semantic: float = None, keyword: float = None, quality: float = None) -> None**

更新权重配置。

```python
reranker.set_weights(semantic=0.6, keyword=0.3, quality=0.1)
```

**get_weights() -> dict**

获取当前权重配置。

---

### LLMInterface - LLM接口

提供与大语言模型的集成接口。

#### 初始化

```python
from llm_retrieval_reranking.core import LLMInterface

llm = LLMInterface(
    model_name="gpt-4",
    max_context_length=4096
)
```

#### 主要方法

**build_prompt(query: str, rerank_results: List[RerankResult], instruction: str = None) -> str**

构建完整的LLM提示。

```python
prompt = llm.build_prompt(
    query="什么是深度学习",
    rerank_results=reranked,
    instruction="基于检索到的文档回答问题"
)
```

**create_rag_context(query: str, rerank_results: List[RerankResult], max_docs: int = 5) -> Dict**

创建RAG上下文。

```python
context = llm.create_rag_context(
    query="深度学习",
    rerank_results=reranked,
    max_docs=5
)
```

**extract_relevant_snippets(query: str, documents: List[Document], snippet_length: int = 200) -> List[Dict]**

从文档中提取相关片段。

---

## 工具模块

### TextEmbedder - 文本嵌入器

提供文本向量化功能。

```python
from llm_retrieval_reranking.utils import TextEmbedder

embedder = TextEmbedder(method='tfidf')
embedder.fit(texts)
vector = embedder.embed(text)
```

### RelevanceScorer - 相关性评分器

提供多维度相关性评分。

```python
from llm_retrieval_reranking.utils import RelevanceScorer

scorer = RelevanceScorer()
keyword_score = scorer.keyword_match_score(query, document)
quality_score = scorer.document_quality_score(document)
```

### Config - 配置管理

管理系统配置参数。

```python
from llm_retrieval_reranking.utils import Config

config = Config()
buffer_size = config.get('stream.buffer_size')
config.set('retrieval.top_k', 20)
```

---

## 数据结构

### Document

```python
class Document:
    def __init__(self, doc_id: str, content: str, metadata: Dict = None):
        self.doc_id = doc_id
        self.content = content
        self.metadata = metadata or {}
        self.timestamp = time.time()
```

### RetrievalResult

```python
class RetrievalResult:
    def __init__(self, document: Document, score: float, rank: int):
        self.document = document
        self.score = score
        self.rank = rank
```

### RerankResult

```python
class RerankResult:
    def __init__(self, document: Document, score: float, rank: int, 
                 original_rank: int, score_details: dict = None):
        self.document = document
        self.score = score
        self.rank = rank
        self.original_rank = original_rank
        self.score_details = score_details or {}
```

---

## 完整示例

```python
from llm_retrieval_reranking.core import (
    StreamDataHandler, Retriever, Reranker, LLMInterface
)

# 1. 处理流数据
stream_handler = StreamDataHandler()
documents = stream_handler.process_stream(data_stream)

# 2. 检索
retriever = Retriever(embedding_method='tfidf')
retriever.index_documents(documents)
retrieval_results = retriever.retrieve(query, top_k=20)

# 3. 重排序
reranker = Reranker(
    semantic_weight=0.5,
    keyword_weight=0.3,
    quality_weight=0.2
)
reranked = reranker.rerank(query, retrieval_results, top_k=5)

# 4. 生成LLM提示
llm = LLMInterface()
prompt = llm.build_prompt(query, reranked)

# 5. 创建RAG上下文
rag_context = llm.create_rag_context(query, reranked)
```
