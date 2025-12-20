"""
集成测试

测试整个检索重排流程。
"""

import sys
sys.path.insert(0, '/home/runner/work/software/software')

from llm_retrieval_reranking.core import (
    StreamDataHandler, Retriever, Reranker, LLMInterface
)


def test_full_pipeline():
    """测试完整的检索重排流程"""
    # 准备数据
    data_stream = [
        {'id': 'doc1', 'content': '机器学习是人工智能的核心技术'},
        {'id': 'doc2', 'content': '深度学习使用多层神经网络结构进行特征学习'},
        {'id': 'doc3', 'content': '自然语言处理是AI的重要应用领域'},
        {'id': 'doc4', 'content': '计算机视觉技术在图像识别中发挥重要作用'},
        {'id': 'doc5', 'content': '强化学习通过与环境交互来学习最优策略'},
    ]
    
    # 1. 流数据处理
    stream_handler = StreamDataHandler()
    documents = stream_handler.process_stream(iter(data_stream))
    assert len(documents) == 5
    
    # 2. 检索
    retriever = Retriever(embedding_method='tfidf')
    retriever.index_documents(documents)
    
    query = "深度学习和神经网络"
    retrieval_results = retriever.retrieve(query, top_k=3)
    assert len(retrieval_results) > 0
    
    # 3. 重排序
    reranker = Reranker()
    rerank_results = reranker.rerank(query, retrieval_results, top_k=2)
    assert len(rerank_results) <= 2
    
    # 4. LLM接口
    llm = LLMInterface()
    prompt = llm.build_prompt(query, rerank_results)
    assert len(prompt) > 0
    assert query in prompt
    
    print("✓ 完整流程测试通过")


def test_streaming_update():
    """测试流式更新场景"""
    stream_handler = StreamDataHandler(buffer_size=10)
    retriever = Retriever()
    
    # 第一批数据
    batch1 = [
        {'id': f'doc{i}', 'content': f'初始内容{i}'}
        for i in range(3)
    ]
    docs1 = stream_handler.process_stream(iter(batch1))
    retriever.index_documents(docs1)
    
    assert retriever.get_index_size() == 3
    
    # 第二批数据（增量）
    batch2 = [
        {'id': f'doc{i}', 'content': f'新增内容{i}'}
        for i in range(3, 6)
    ]
    docs2 = stream_handler.process_stream(iter(batch2))
    retriever.index_documents(docs2)
    
    assert retriever.get_index_size() == 6
    
    print("✓ 流式更新测试通过")


def test_multilingual_content():
    """测试多语言内容"""
    documents_data = [
        {'id': 'doc1', 'content': 'Machine learning is a subset of artificial intelligence'},
        {'id': 'doc2', 'content': '机器学习是人工智能的一个子集'},
        {'id': 'doc3', 'content': 'Deep learning uses neural networks 深度学习使用神经网络'},
    ]
    
    stream_handler = StreamDataHandler()
    documents = stream_handler.process_stream(iter(documents_data))
    
    retriever = Retriever()
    retriever.index_documents(documents)
    
    # 中文查询
    results_cn = retriever.retrieve("机器学习", top_k=2)
    assert len(results_cn) > 0
    
    # 英文查询
    results_en = retriever.retrieve("machine learning", top_k=2)
    assert len(results_en) > 0
    
    print("✓ 多语言内容测试通过")


def test_rag_workflow():
    """测试RAG工作流"""
    # 准备知识库
    knowledge_base = [
        {'id': 'kb1', 'content': 'Python是一种高级编程语言，广泛用于数据科学和机器学习'},
        {'id': 'kb2', 'content': 'TensorFlow是Google开发的深度学习框架'},
        {'id': 'kb3', 'content': 'PyTorch是Facebook开发的深度学习框架，具有动态计算图特性'},
    ]
    
    # 初始化组件
    stream_handler = StreamDataHandler()
    docs = stream_handler.process_stream(iter(knowledge_base))
    
    retriever = Retriever()
    retriever.index_documents(docs)
    
    reranker = Reranker()
    llm = LLMInterface(max_context_length=1024)
    
    # 用户查询
    query = "什么深度学习框架比较流行"
    
    # 检索相关文档
    retrieval_results = retriever.retrieve(query, top_k=3)
    
    # 重排序
    reranked = reranker.rerank(query, retrieval_results, top_k=2)
    
    # 创建RAG上下文
    rag_context = llm.create_rag_context(query, reranked, max_docs=2)
    
    assert rag_context['query'] == query
    assert len(rag_context['documents']) == 2
    assert rag_context['metadata']['total_retrieved'] == 2  # 传入的是reranked结果
    assert rag_context['metadata']['used_in_context'] == 2
    
    print("✓ RAG工作流测试通过")


def test_performance_with_large_dataset():
    """测试大数据集性能"""
    import time
    
    # 生成较大的数据集
    large_dataset = [
        {'id': f'doc{i}', 'content': f'这是第{i}个文档，包含一些关于主题{i%10}的内容描述'}
        for i in range(100)
    ]
    
    start_time = time.time()
    
    # 流处理
    stream_handler = StreamDataHandler(buffer_size=200)
    documents = stream_handler.process_stream(iter(large_dataset))
    
    # 建立索引
    retriever = Retriever(embedding_method='tfidf')
    retriever.index_documents(documents)
    
    # 执行检索
    query = "主题5相关内容"
    results = retriever.retrieve(query, top_k=10)
    
    # 重排序
    reranker = Reranker()
    reranked = reranker.rerank(query, results, top_k=5)
    
    elapsed_time = time.time() - start_time
    
    assert len(documents) == 100
    assert len(results) == 10
    assert len(reranked) == 5
    
    print(f"✓ 大数据集性能测试通过 (处理100个文档用时: {elapsed_time:.2f}秒)")


def run_all_tests():
    """运行所有集成测试"""
    print("=" * 60)
    print("运行集成测试")
    print("=" * 60)
    
    test_full_pipeline()
    test_streaming_update()
    test_multilingual_content()
    test_rag_workflow()
    test_performance_with_large_dataset()
    
    print("\n" + "=" * 60)
    print("所有集成测试通过! ✓")
    print("=" * 60)


if __name__ == '__main__':
    run_all_tests()
