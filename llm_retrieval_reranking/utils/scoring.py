"""
评分工具模块

提供多维度相关性评分功能。
"""

import re
from typing import Set


class RelevanceScorer:
    """相关性评分器"""
    
    def __init__(self):
        """初始化评分器"""
        pass
    
    def keyword_match_score(self, query: str, document: str) -> float:
        """
        计算关键词匹配分数
        
        Args:
            query: 查询文本
            document: 文档文本
        
        Returns:
            匹配分数 (0-1)
        """
        query_keywords = self._extract_keywords(query)
        doc_keywords = self._extract_keywords(document)
        
        if not query_keywords:
            return 0.0
        
        # 计算交集和并集
        intersection = query_keywords & doc_keywords
        
        # Jaccard相似度
        match_ratio = len(intersection) / len(query_keywords)
        
        # 考虑关键词出现频率
        frequency_bonus = 0.0
        doc_lower = document.lower()
        for keyword in intersection:
            count = doc_lower.count(keyword)
            frequency_bonus += min(count / 10, 0.1)  # 最多增加0.1分
        
        score = min(match_ratio + frequency_bonus, 1.0)
        return score
    
    def document_quality_score(self, document: str) -> float:
        """
        计算文档质量分数
        
        Args:
            document: 文档文本
        
        Returns:
            质量分数 (0-1)
        """
        if not document:
            return 0.0
        
        # 文档长度评分（适中长度得分高）
        length = len(document)
        if length < 50:
            length_score = length / 50 * 0.5
        elif length < 500:
            length_score = 0.5 + (length - 50) / 450 * 0.5
        else:
            length_score = 1.0 - min((length - 500) / 1500, 0.3)  # 太长的文档略微降分
        
        # 句子完整性评分
        sentences = re.split(r'[。！？.!?]', document)
        complete_sentences = [s for s in sentences if len(s.strip()) > 10]
        sentence_score = min(len(complete_sentences) / 5, 1.0)
        
        # 信息密度评分（词汇多样性）
        words = re.findall(r'\w+', document.lower())
        if words:
            unique_ratio = len(set(words)) / len(words)
            diversity_score = min(unique_ratio * 2, 1.0)
        else:
            diversity_score = 0.0
        
        # 综合评分
        quality_score = (
            0.4 * length_score +
            0.3 * sentence_score +
            0.3 * diversity_score
        )
        
        return quality_score
    
    def semantic_coherence_score(self, query: str, document: str) -> float:
        """
        计算语义连贯性分数（简化版）
        
        Args:
            query: 查询文本
            document: 文档文本
        
        Returns:
            连贯性分数 (0-1)
        """
        query_words = set(self._extract_keywords(query))
        
        # 检查关键词在文档中的分布
        doc_lower = document.lower()
        positions = []
        
        for keyword in query_words:
            pos = doc_lower.find(keyword)
            if pos != -1:
                positions.append(pos / len(doc_lower))
        
        if not positions:
            return 0.0
        
        # 如果关键词分布比较集中，说明相关性高
        if len(positions) == 1:
            coherence = 0.5
        else:
            # 计算分布的标准差，分布越集中得分越高
            import numpy as np
            std = np.std(positions)
            coherence = max(0.0, 1.0 - std * 2)
        
        return coherence
    
    def _extract_keywords(self, text: str) -> Set[str]:
        """
        提取关键词
        
        Args:
            text: 输入文本
        
        Returns:
            关键词集合
        """
        # 简单的关键词提取：分词并过滤停用词
        words = re.findall(r'\w+', text.lower())
        
        # 简单的中英文停用词列表
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were',
            'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from',
            '的', '了', '和', '是', '在', '有', '我', '你', '他', '她', '它',
            '这', '那', '个', '们', '也', '就', '都', '吗', '呢', '吧'
        }
        
        keywords = {word for word in words if word not in stopwords and len(word) > 1}
        return keywords
    
    def combined_score(self, query: str, document: str, 
                      keyword_weight: float = 0.5,
                      quality_weight: float = 0.3,
                      coherence_weight: float = 0.2) -> float:
        """
        计算综合评分
        
        Args:
            query: 查询文本
            document: 文档文本
            keyword_weight: 关键词匹配权重
            quality_weight: 文档质量权重
            coherence_weight: 语义连贯性权重
        
        Returns:
            综合分数 (0-1)
        """
        keyword_score = self.keyword_match_score(query, document)
        quality_score = self.document_quality_score(document)
        coherence_score = self.semantic_coherence_score(query, document)
        
        total = (
            keyword_weight * keyword_score +
            quality_weight * quality_score +
            coherence_weight * coherence_score
        )
        
        return total
