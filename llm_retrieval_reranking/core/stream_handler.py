"""
流数据处理器模块

用于处理实时流入的数据，将其转换为可检索的文档格式。
支持批处理和增量更新。
"""

import time
from typing import List, Dict, Any, Iterator, Optional
from collections import deque


class Document:
    """文档数据结构"""
    
    def __init__(self, doc_id: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """
        初始化文档
        
        Args:
            doc_id: 文档唯一标识符
            content: 文档内容
            metadata: 文档元数据（可选）
        """
        self.doc_id = doc_id
        self.content = content
        self.metadata = metadata or {}
        self.timestamp = time.time()
    
    def __repr__(self):
        return f"Document(id={self.doc_id}, content_len={len(self.content)})"


class StreamDataHandler:
    """流数据处理器"""
    
    def __init__(self, buffer_size: int = 1000, batch_size: int = 100):
        """
        初始化流数据处理器
        
        Args:
            buffer_size: 缓冲区大小
            batch_size: 批处理大小
        """
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.buffer = deque(maxlen=buffer_size)
        self.processed_count = 0
    
    def add_document(self, doc: Document) -> None:
        """
        添加单个文档到缓冲区
        
        Args:
            doc: 文档对象
        """
        self.buffer.append(doc)
        self.processed_count += 1
    
    def process_stream(self, stream: Iterator[Dict[str, Any]]) -> List[Document]:
        """
        处理数据流
        
        Args:
            stream: 数据流迭代器，每个元素为字典格式
        
        Returns:
            处理后的文档列表
        """
        documents = []
        
        for item in stream:
            doc = self._parse_item(item)
            if doc:
                self.add_document(doc)
                documents.append(doc)
        
        return documents
    
    def _parse_item(self, item: Dict[str, Any]) -> Optional[Document]:
        """
        解析单个数据项
        
        Args:
            item: 数据项字典
        
        Returns:
            Document对象或None
        """
        try:
            doc_id = item.get('id', f"doc_{self.processed_count}")
            content = item.get('content', item.get('text', ''))
            
            if not content:
                return None
            
            metadata = {k: v for k, v in item.items() if k not in ['id', 'content', 'text']}
            
            return Document(doc_id, content, metadata)
        except Exception as e:
            print(f"Error parsing item: {e}")
            return None
    
    def get_recent_documents(self, n: int = None) -> List[Document]:
        """
        获取最近的n个文档
        
        Args:
            n: 返回文档数量，默认返回全部
        
        Returns:
            文档列表
        """
        if n is None:
            return list(self.buffer)
        return list(self.buffer)[-n:]
    
    def clear_buffer(self) -> None:
        """清空缓冲区"""
        self.buffer.clear()
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取处理统计信息
        
        Returns:
            统计信息字典
        """
        return {
            'total_processed': self.processed_count,
            'buffer_size': len(self.buffer),
            'buffer_capacity': self.buffer_size
        }
