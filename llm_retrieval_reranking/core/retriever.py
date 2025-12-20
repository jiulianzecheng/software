"""
检索器模块

实现基于向量相似度的文档检索功能。
支持多种相似度计算方法和高效的文档索引。
"""

from typing import List, Tuple, Optional
import numpy as np
from ..utils.embeddings import TextEmbedder
from .stream_handler import Document


class RetrievalResult:
    """检索结果"""
    
    def __init__(self, document: Document, score: float, rank: int):
        """
        初始化检索结果
        
        Args:
            document: 文档对象
            score: 相似度分数
            rank: 排名位置
        """
        self.document = document
        self.score = score
        self.rank = rank
    
    def __repr__(self):
        return f"RetrievalResult(doc_id={self.document.doc_id}, score={self.score:.4f}, rank={self.rank})"


class Retriever:
    """文档检索器"""
    
    def __init__(self, embedding_method: str = 'tfidf'):
        """
        初始化检索器
        
        Args:
            embedding_method: 向量化方法 ('tfidf' 或 'simple')
        """
        self.embedder = TextEmbedder(method=embedding_method)
        self.document_embeddings = {}
        self.documents = {}
    
    def index_documents(self, documents: List[Document]) -> None:
        """
        为文档建立索引
        
        Args:
            documents: 文档列表
        """
        contents = [doc.content for doc in documents]
        embeddings = self.embedder.embed_batch(contents)
        
        for doc, emb in zip(documents, embeddings):
            self.documents[doc.doc_id] = doc
            self.document_embeddings[doc.doc_id] = emb
    
    def retrieve(self, query: str, documents: Optional[List[Document]] = None, 
                 top_k: int = 10) -> List[RetrievalResult]:
        """
        检索相关文档
        
        Args:
            query: 查询文本
            documents: 待检索文档列表（如果为None，使用已索引的文档）
            top_k: 返回top K个结果
        
        Returns:
            检索结果列表
        """
        # 如果提供了新文档，先建立索引
        if documents is not None:
            self.index_documents(documents)
        
        if not self.documents:
            return []
        
        # 计算查询向量
        query_embedding = self.embedder.embed(query)
        
        # 计算相似度
        similarities = []
        for doc_id, doc_emb in self.document_embeddings.items():
            similarity = self._cosine_similarity(query_embedding, doc_emb)
            similarities.append((doc_id, similarity))
        
        # 排序并返回top K
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_results = similarities[:top_k]
        
        results = []
        for rank, (doc_id, score) in enumerate(top_results, 1):
            doc = self.documents[doc_id]
            results.append(RetrievalResult(doc, score, rank))
        
        return results
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        计算余弦相似度
        
        Args:
            vec1: 向量1
            vec2: 向量2
        
        Returns:
            余弦相似度值
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def clear_index(self) -> None:
        """清空索引"""
        self.documents.clear()
        self.document_embeddings.clear()
    
    def get_index_size(self) -> int:
        """获取索引中的文档数量"""
        return len(self.documents)
