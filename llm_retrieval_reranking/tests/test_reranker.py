"""
测试重排序器
"""

import sys
sys.path.insert(0, '/home/runner/work/software/software')

from llm_retrieval_reranking.core.stream_handler import Document
from llm_retrieval_reranking.core.retriever import RetrievalResult
from llm_retrieval_reranking.core.reranker import Reranker


def test_reranker_initialization():
    """测试重排序器初始化"""
    reranker = Reranker(
        semantic_weight=0.5,
        keyword_weight=0.3,
        quality_weight=0.2
    )
    
    # 权重应该被归一化
    total = reranker.semantic_weight + reranker.keyword_weight + reranker.quality_weight
    assert abs(total - 1.0) < 0.001
    
    print("✓ 重排序器初始化测试通过")


def test_rerank():
    """测试重排序功能"""
    reranker = Reranker()
    
    # 创建模拟的检索结果
    doc1 = Document("doc1", "机器学习是AI的分支")
    doc2 = Document("doc2", "深度学习使用神经网络进行复杂的模式识别，它是机器学习的重要子领域，"
                           "在图像识别、自然语言处理等任务中表现出色")
    doc3 = Document("doc3", "数据科学")
    
    retrieval_results = [
        RetrievalResult(doc1, 0.8, 1),
        RetrievalResult(doc2, 0.75, 2),
        RetrievalResult(doc3, 0.7, 3),
    ]
    
    query = "深度学习和神经网络"
    reranked = reranker.rerank(query, retrieval_results, top_k=3)
    
    assert len(reranked) == 3
    # 由于doc2质量更高且关键词匹配更好，应该排在前面
    assert reranked[0].document.doc_id == "doc2"
    
    print("✓ 重排序功能测试通过")


def test_score_details():
    """测试评分详情"""
    reranker = Reranker()
    
    doc = Document("doc1", "深度学习是机器学习的一个重要分支，使用多层神经网络结构")
    retrieval_results = [RetrievalResult(doc, 0.9, 1)]
    
    query = "深度学习神经网络"
    reranked = reranker.rerank(query, retrieval_results)
    
    assert len(reranked) == 1
    result = reranked[0]
    
    # 检查评分详情
    assert 'semantic_score' in result.score_details
    assert 'keyword_score' in result.score_details
    assert 'quality_score' in result.score_details
    assert 'weights' in result.score_details
    
    print("✓ 评分详情测试通过")


def test_set_weights():
    """测试设置权重"""
    reranker = Reranker()
    
    reranker.set_weights(semantic=0.6, keyword=0.3, quality=0.1)
    
    weights = reranker.get_weights()
    total = weights['semantic'] + weights['keyword'] + weights['quality']
    
    assert abs(total - 1.0) < 0.001
    print("✓ 设置权重测试通过")


def test_empty_results():
    """测试空结果"""
    reranker = Reranker()
    
    reranked = reranker.rerank("测试查询", [])
    
    assert len(reranked) == 0
    print("✓ 空结果测试通过")


def test_top_k_limit():
    """测试top_k限制"""
    reranker = Reranker()
    
    documents = [
        Document(f"doc{i}", f"内容{i}") for i in range(10)
    ]
    retrieval_results = [
        RetrievalResult(doc, 1.0 - i*0.1, i+1) 
        for i, doc in enumerate(documents)
    ]
    
    query = "测试"
    reranked = reranker.rerank(query, retrieval_results, top_k=5)
    
    assert len(reranked) == 5
    print("✓ top_k限制测试通过")


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("运行重排序器测试")
    print("=" * 60)
    
    test_reranker_initialization()
    test_rerank()
    test_score_details()
    test_set_weights()
    test_empty_results()
    test_top_k_limit()
    
    print("\n" + "=" * 60)
    print("所有测试通过! ✓")
    print("=" * 60)


if __name__ == '__main__':
    run_all_tests()
