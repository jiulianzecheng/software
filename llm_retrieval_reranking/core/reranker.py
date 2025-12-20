"""
重排序器模块

实现基于多维度特征的文档重排序算法。
综合考虑语义相关性、关键词匹配、文档质量等因素。
"""

from typing import List, Optional
import numpy as np
from .retriever import RetrievalResult
from .stream_handler import Document
from ..utils.scoring import RelevanceScorer


class RerankResult:
    """重排序结果"""
    
    def __init__(self, document: Document, score: float, rank: int, 
                 original_rank: int, score_details: Optional[dict] = None):
        """
        初始化重排序结果
        
        Args:
            document: 文档对象
            score: 重排序后的分数
            rank: 重排序后的排名
            original_rank: 原始排名
            score_details: 评分细节
        """
        self.document = document
        self.score = score
        self.rank = rank
        self.original_rank = original_rank
        self.score_details = score_details or {}
    
    def __repr__(self):
        return (f"RerankResult(doc_id={self.document.doc_id}, "
                f"score={self.score:.4f}, rank={self.rank}, "
                f"original_rank={self.original_rank})")


class Reranker:
    """文档重排序器"""
    
    def __init__(self, 
                 semantic_weight: float = 0.5,
                 keyword_weight: float = 0.3,
                 quality_weight: float = 0.2):
        """
        初始化重排序器
        
        Args:
            semantic_weight: 语义相关性权重
            keyword_weight: 关键词匹配权重
            quality_weight: 文档质量权重
        """
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
        self.quality_weight = quality_weight
        self.scorer = RelevanceScorer()
        
        # 权重归一化
        total = semantic_weight + keyword_weight + quality_weight
        self.semantic_weight /= total
        self.keyword_weight /= total
        self.quality_weight /= total
    
    def rerank(self, query: str, retrieval_results: List[RetrievalResult], 
               top_k: Optional[int] = None) -> List[RerankResult]:
        """
        对检索结果进行重排序
        
        Args:
            query: 查询文本
            retrieval_results: 初始检索结果列表
            top_k: 返回top K个结果（默认返回全部）
        
        Returns:
            重排序后的结果列表
        """
        if not retrieval_results:
            return []
        
        rerank_scores = []
        
        for result in retrieval_results:
            # 计算各维度分数
            semantic_score = result.score  # 使用检索阶段的相似度分数
            keyword_score = self.scorer.keyword_match_score(query, result.document.content)
            quality_score = self.scorer.document_quality_score(result.document.content)
            
            # 综合评分
            final_score = (
                self.semantic_weight * semantic_score +
                self.keyword_weight * keyword_score +
                self.quality_weight * quality_score
            )
            
            score_details = {
                'semantic_score': semantic_score,
                'keyword_score': keyword_score,
                'quality_score': quality_score,
                'weights': {
                    'semantic': self.semantic_weight,
                    'keyword': self.keyword_weight,
                    'quality': self.quality_weight
                }
            }
            
            rerank_scores.append((result, final_score, score_details))
        
        # 按综合分数排序
        rerank_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 生成重排序结果
        results = []
        limit = top_k if top_k is not None else len(rerank_scores)
        
        for new_rank, (original_result, score, details) in enumerate(rerank_scores[:limit], 1):
            rerank_result = RerankResult(
                document=original_result.document,
                score=score,
                rank=new_rank,
                original_rank=original_result.rank,
                score_details=details
            )
            results.append(rerank_result)
        
        return results
    
    def set_weights(self, semantic: float = None, keyword: float = None, 
                    quality: float = None) -> None:
        """
        更新权重配置
        
        Args:
            semantic: 语义相关性权重
            keyword: 关键词匹配权重
            quality: 文档质量权重
        """
        if semantic is not None:
            self.semantic_weight = semantic
        if keyword is not None:
            self.keyword_weight = keyword
        if quality is not None:
            self.quality_weight = quality
        
        # 重新归一化
        total = self.semantic_weight + self.keyword_weight + self.quality_weight
        if total > 0:
            self.semantic_weight /= total
            self.keyword_weight /= total
            self.quality_weight /= total
    
    def get_weights(self) -> dict:
        """获取当前权重配置"""
        return {
            'semantic': self.semantic_weight,
            'keyword': self.keyword_weight,
            'quality': self.quality_weight
        }
