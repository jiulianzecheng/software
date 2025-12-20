"""
文本嵌入工具模块

提供多种文本向量化方法，包括TF-IDF和简单词频向量。
"""

from typing import List
import numpy as np
from collections import Counter
import re


class TextEmbedder:
    """文本嵌入器"""
    
    def __init__(self, method: str = 'tfidf'):
        """
        初始化嵌入器
        
        Args:
            method: 向量化方法 ('tfidf' 或 'simple')
        """
        self.method = method
        self.vocabulary = {}
        self.idf_scores = {}
        self.is_fitted = False
    
    def _tokenize(self, text: str) -> List[str]:
        """
        文本分词
        
        Args:
            text: 输入文本
        
        Returns:
            词语列表
        """
        # 简单的分词：按空格和标点分割，转小写
        text = text.lower()
        tokens = re.findall(r'\w+', text)
        return tokens
    
    def _build_vocabulary(self, texts: List[str]) -> None:
        """
        构建词汇表
        
        Args:
            texts: 文本列表
        """
        all_tokens = set()
        for text in texts:
            tokens = self._tokenize(text)
            all_tokens.update(tokens)
        
        self.vocabulary = {token: idx for idx, token in enumerate(sorted(all_tokens))}
    
    def _calculate_idf(self, texts: List[str]) -> None:
        """
        计算IDF分数
        
        Args:
            texts: 文本列表
        """
        n_docs = len(texts)
        doc_freq = Counter()
        
        for text in texts:
            tokens = set(self._tokenize(text))
            doc_freq.update(tokens)
        
        self.idf_scores = {}
        for token, freq in doc_freq.items():
            self.idf_scores[token] = np.log((n_docs + 1) / (freq + 1)) + 1
    
    def _embed_tfidf(self, text: str) -> np.ndarray:
        """
        使用TF-IDF方法嵌入文本
        
        Args:
            text: 输入文本
        
        Returns:
            TF-IDF向量
        """
        tokens = self._tokenize(text)
        token_counts = Counter(tokens)
        
        vector = np.zeros(len(self.vocabulary))
        
        for token, count in token_counts.items():
            if token in self.vocabulary:
                idx = self.vocabulary[token]
                tf = count / len(tokens) if len(tokens) > 0 else 0
                idf = self.idf_scores.get(token, 1.0)
                vector[idx] = tf * idf
        
        return vector
    
    def _embed_simple(self, text: str) -> np.ndarray:
        """
        使用简单词频方法嵌入文本
        
        Args:
            text: 输入文本
        
        Returns:
            词频向量
        """
        tokens = self._tokenize(text)
        token_counts = Counter(tokens)
        
        vector = np.zeros(len(self.vocabulary))
        
        for token, count in token_counts.items():
            if token in self.vocabulary:
                idx = self.vocabulary[token]
                vector[idx] = count
        
        # 归一化
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector
    
    def fit(self, texts: List[str]) -> None:
        """
        拟合嵌入器
        
        Args:
            texts: 训练文本列表
        """
        self._build_vocabulary(texts)
        if self.method == 'tfidf':
            self._calculate_idf(texts)
        self.is_fitted = True
    
    def embed(self, text: str) -> np.ndarray:
        """
        嵌入单个文本
        
        Args:
            text: 输入文本
        
        Returns:
            文本向量
        """
        if not self.is_fitted:
            # 如果未拟合，使用简单方法
            self.fit([text])
        
        if self.method == 'tfidf':
            return self._embed_tfidf(text)
        else:
            return self._embed_simple(text)
    
    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        批量嵌入文本
        
        Args:
            texts: 文本列表
        
        Returns:
            文本向量列表
        """
        # 先拟合所有文本
        self.fit(texts)
        
        # 批量嵌入
        embeddings = [self.embed(text) for text in texts]
        return embeddings
    
    def get_vocabulary_size(self) -> int:
        """获取词汇表大小"""
        return len(self.vocabulary)
