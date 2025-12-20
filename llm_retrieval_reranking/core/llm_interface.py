"""
LLM接口模块

提供与大语言模型的集成接口，用于增强检索和重排序能力。
"""

from typing import List, Dict, Any, Optional
from .stream_handler import Document
from .reranker import RerankResult


class LLMInterface:
    """大语言模型接口"""
    
    def __init__(self, model_name: str = "default", max_context_length: int = 4096):
        """
        初始化LLM接口
        
        Args:
            model_name: 模型名称
            max_context_length: 最大上下文长度
        """
        self.model_name = model_name
        self.max_context_length = max_context_length
    
    def format_context(self, query: str, rerank_results: List[RerankResult]) -> str:
        """
        格式化检索结果为LLM上下文
        
        Args:
            query: 查询文本
            rerank_results: 重排序后的结果列表
        
        Returns:
            格式化的上下文文本
        """
        context_parts = [f"查询: {query}\n", "\n相关文档:\n"]
        
        for result in rerank_results:
            doc_text = self._truncate_text(result.document.content, 500)
            context_parts.append(
                f"\n[文档 {result.rank}] (相关度: {result.score:.2f})\n{doc_text}\n"
            )
        
        full_context = "".join(context_parts)
        return self._truncate_text(full_context, self.max_context_length)
    
    def build_prompt(self, query: str, rerank_results: List[RerankResult], 
                     instruction: Optional[str] = None) -> str:
        """
        构建完整的LLM提示
        
        Args:
            query: 查询文本
            rerank_results: 重排序后的结果列表
            instruction: 自定义指令（可选）
        
        Returns:
            完整的提示文本
        """
        if instruction is None:
            instruction = "基于以下检索到的文档，回答用户的查询。"
        
        context = self.format_context(query, rerank_results)
        
        prompt = f"""{instruction}

{context}

请基于上述文档信息回答以下问题:
{query}

回答:"""
        
        return prompt
    
    def extract_relevant_snippets(self, query: str, documents: List[Document], 
                                   snippet_length: int = 200) -> List[Dict[str, Any]]:
        """
        从文档中提取相关片段
        
        Args:
            query: 查询文本
            documents: 文档列表
            snippet_length: 片段长度
        
        Returns:
            包含文档ID和相关片段的字典列表
        """
        snippets = []
        query_keywords = set(query.lower().split())
        
        for doc in documents:
            content = doc.content
            words = content.split()
            
            # 找到包含最多查询关键词的窗口
            best_score = 0
            best_start = 0
            window_size = min(snippet_length // 5, len(words))  # 大约200字符对应的词数
            
            for i in range(len(words) - window_size + 1):
                window_words = set(word.lower() for word in words[i:i+window_size])
                score = len(query_keywords & window_words)
                
                if score > best_score:
                    best_score = score
                    best_start = i
            
            # 提取最佳片段
            snippet_words = words[best_start:best_start+window_size]
            snippet_text = " ".join(snippet_words)
            
            snippets.append({
                'doc_id': doc.doc_id,
                'snippet': snippet_text,
                'relevance_score': best_score
            })
        
        return sorted(snippets, key=lambda x: x['relevance_score'], reverse=True)
    
    def _truncate_text(self, text: str, max_length: int) -> str:
        """
        截断文本到指定长度
        
        Args:
            text: 原始文本
            max_length: 最大长度
        
        Returns:
            截断后的文本
        """
        if len(text) <= max_length:
            return text
        return text[:max_length-3] + "..."
    
    def create_rag_context(self, query: str, rerank_results: List[RerankResult],
                          max_docs: int = 5) -> Dict[str, Any]:
        """
        创建RAG (Retrieval-Augmented Generation) 上下文
        
        Args:
            query: 查询文本
            rerank_results: 重排序结果
            max_docs: 最大文档数量
        
        Returns:
            RAG上下文字典
        """
        top_results = rerank_results[:max_docs]
        
        context = {
            'query': query,
            'documents': [],
            'metadata': {
                'total_retrieved': len(rerank_results),
                'used_in_context': len(top_results),
                'model': self.model_name
            }
        }
        
        for result in top_results:
            context['documents'].append({
                'id': result.document.doc_id,
                'content': result.document.content,
                'score': result.score,
                'rank': result.rank,
                'metadata': result.document.metadata
            })
        
        return context
