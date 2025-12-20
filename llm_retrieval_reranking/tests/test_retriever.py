"""
测试检索器
"""

import sys
sys.path.insert(0, '/home/runner/work/software/software')

from llm_retrieval_reranking.core.stream_handler import Document
from llm_retrieval_reranking.core.retriever import Retriever


def test_retriever_initialization():
    """测试检索器初始化"""
    retriever = Retriever(embedding_method='tfidf')
    
    assert retriever.embedder is not None
    assert len(retriever.documents) == 0
    print("✓ 检索器初始化测试通过")


def test_index_documents():
    """测试文档索引"""
    retriever = Retriever()
    
    documents = [
        Document("doc1", "机器学习是人工智能的一个分支"),
        Document("doc2", "深度学习使用神经网络"),
        Document("doc3", "自然语言处理处理文本数据"),
    ]
    
    retriever.index_documents(documents)
    
    assert retriever.get_index_size() == 3
    assert "doc1" in retriever.documents
    assert "doc1" in retriever.document_embeddings
    print("✓ 文档索引测试通过")


def test_retrieve():
    """测试检索功能"""
    retriever = Retriever(embedding_method='tfidf')
    
    documents = [
        Document("doc1", "机器学习是人工智能的重要分支，它使计算机能够从数据中学习"),
        Document("doc2", "深度学习是机器学习的子领域，使用多层神经网络"),
        Document("doc3", "自然语言处理让计算机理解人类语言"),
        Document("doc4", "计算机视觉处理图像和视频数据"),
    ]
    
    retriever.index_documents(documents)
    
    query = "深度学习和神经网络"
    results = retriever.retrieve(query, top_k=3)
    
    assert len(results) <= 3
    assert results[0].score >= results[1].score  # 分数递减
    
    # 验证doc2应该在结果中（最相关的文档之一）
    doc_ids = [r.document.doc_id for r in results]
    assert "doc2" in doc_ids  # doc2应该在top3中
    
    print("✓ 检索功能测试通过")


def test_cosine_similarity():
    """测试余弦相似度计算"""
    import numpy as np
    retriever = Retriever()
    
    vec1 = np.array([1, 0, 0])
    vec2 = np.array([1, 0, 0])
    vec3 = np.array([0, 1, 0])
    
    # 相同向量的相似度应该为1
    sim1 = retriever._cosine_similarity(vec1, vec2)
    assert abs(sim1 - 1.0) < 0.001
    
    # 正交向量的相似度应该为0
    sim2 = retriever._cosine_similarity(vec1, vec3)
    assert abs(sim2) < 0.001
    
    print("✓ 余弦相似度测试通过")


def test_empty_query():
    """测试空查询"""
    retriever = Retriever()
    
    documents = [
        Document("doc1", "测试内容1"),
        Document("doc2", "测试内容2"),
    ]
    
    retriever.index_documents(documents)
    
    results = retriever.retrieve("", top_k=5)
    
    # 空查询应该返回结果
    assert len(results) > 0
    print("✓ 空查询测试通过")


def test_clear_index():
    """测试清空索引"""
    retriever = Retriever()
    
    documents = [
        Document("doc1", "内容1"),
        Document("doc2", "内容2"),
    ]
    
    retriever.index_documents(documents)
    assert retriever.get_index_size() == 2
    
    retriever.clear_index()
    assert retriever.get_index_size() == 0
    
    print("✓ 清空索引测试通过")


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("运行检索器测试")
    print("=" * 60)
    
    test_retriever_initialization()
    test_index_documents()
    test_retrieve()
    test_cosine_similarity()
    test_empty_query()
    test_clear_index()
    
    print("\n" + "=" * 60)
    print("所有测试通过! ✓")
    print("=" * 60)


if __name__ == '__main__':
    run_all_tests()
