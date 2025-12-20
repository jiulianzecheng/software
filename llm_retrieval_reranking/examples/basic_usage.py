"""
基本使用示例

演示如何使用LLM检索重排系统的基本功能。
"""

import sys
sys.path.insert(0, '/home/runner/work/software/software')

from llm_retrieval_reranking.core import (
    StreamDataHandler, Document, Retriever, Reranker, LLMInterface
)


def main():
    print("=" * 80)
    print("LLM检索重排系统 - 基本使用示例")
    print("=" * 80)
    
    # 1. 准备示例数据
    print("\n[1] 准备示例数据...")
    sample_documents = [
        {
            'id': 'doc1',
            'content': '机器学习是人工智能的一个重要分支，它使计算机能够从数据中学习并改进性能。',
            'category': 'AI'
        },
        {
            'id': 'doc2',
            'content': '深度学习是机器学习的子领域，使用神经网络来处理复杂的模式识别任务。',
            'category': 'AI'
        },
        {
            'id': 'doc3',
            'content': '自然语言处理(NLP)是让计算机理解和生成人类语言的技术。',
            'category': 'NLP'
        },
        {
            'id': 'doc4',
            'content': '大语言模型如GPT和BERT在各种NLP任务中展现了强大的能力。',
            'category': 'NLP'
        },
        {
            'id': 'doc5',
            'content': '计算机视觉技术使机器能够理解和分析图像和视频内容。',
            'category': 'CV'
        },
    ]
    
    # 2. 初始化流数据处理器
    print("\n[2] 初始化流数据处理器...")
    stream_handler = StreamDataHandler(buffer_size=100)
    
    # 模拟流式处理
    documents = stream_handler.process_stream(iter(sample_documents))
    print(f"   处理了 {len(documents)} 个文档")
    print(f"   统计信息: {stream_handler.get_statistics()}")
    
    # 3. 初始化检索器并建立索引
    print("\n[3] 初始化检索器...")
    retriever = Retriever(embedding_method='tfidf')
    retriever.index_documents(documents)
    print(f"   索引了 {retriever.get_index_size()} 个文档")
    
    # 4. 执行检索
    query = "什么是深度学习和神经网络"
    print(f"\n[4] 执行检索: '{query}'")
    retrieval_results = retriever.retrieve(query, top_k=5)
    
    print(f"   检索结果 (共 {len(retrieval_results)} 个):")
    for result in retrieval_results:
        print(f"   - Rank {result.rank}: Doc {result.document.doc_id} "
              f"(Score: {result.score:.4f})")
        print(f"     内容: {result.document.content[:50]}...")
    
    # 5. 初始化重排序器
    print("\n[5] 执行重排序...")
    reranker = Reranker(
        semantic_weight=0.5,
        keyword_weight=0.3,
        quality_weight=0.2
    )
    
    rerank_results = reranker.rerank(query, retrieval_results, top_k=3)
    
    print(f"   重排序结果 (共 {len(rerank_results)} 个):")
    for result in rerank_results:
        print(f"   - Rank {result.rank}: Doc {result.document.doc_id} "
              f"(Score: {result.score:.4f}, 原排名: {result.original_rank})")
        print(f"     内容: {result.document.content}")
        print(f"     评分详情: 语义={result.score_details['semantic_score']:.3f}, "
              f"关键词={result.score_details['keyword_score']:.3f}, "
              f"质量={result.score_details['quality_score']:.3f}")
    
    # 6. 创建LLM上下文
    print("\n[6] 生成LLM提示...")
    llm_interface = LLMInterface(model_name="example_model", max_context_length=2048)
    
    prompt = llm_interface.build_prompt(query, rerank_results)
    print(f"   生成的提示 (前500字符):\n")
    print(prompt[:500] + "...")
    
    # 7. 创建RAG上下文
    print("\n[7] 创建RAG上下文...")
    rag_context = llm_interface.create_rag_context(query, rerank_results, max_docs=3)
    print(f"   RAG上下文包含:")
    print(f"   - 查询: {rag_context['query']}")
    print(f"   - 文档数: {len(rag_context['documents'])}")
    print(f"   - 元数据: {rag_context['metadata']}")
    
    print("\n" + "=" * 80)
    print("示例运行完成!")
    print("=" * 80)


if __name__ == '__main__':
    main()
