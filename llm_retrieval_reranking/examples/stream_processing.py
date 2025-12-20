"""
流式数据处理示例

演示如何处理持续流入的数据。
"""

import sys
sys.path.insert(0, '/home/runner/work/software/software')

from llm_retrieval_reranking.core import StreamDataHandler, Retriever, Reranker
import time


def data_stream_generator():
    """模拟一个数据流生成器"""
    
    documents_data = [
        {'id': f'stream_doc_{i}', 
         'content': f'这是第{i}条流式数据，包含关于主题{i%3}的内容。' + 
                   f'内容描述了一些重要信息和数据分析结果。' * (i % 3 + 1),
         'timestamp': time.time() + i}
        for i in range(20)
    ]
    
    for doc in documents_data:
        yield doc
        # 模拟实时数据流的延迟
        time.sleep(0.1)


def main():
    print("=" * 80)
    print("流式数据处理示例")
    print("=" * 80)
    
    # 初始化流处理器
    stream_handler = StreamDataHandler(buffer_size=50, batch_size=10)
    retriever = Retriever()
    reranker = Reranker()
    
    # 处理流数据
    print("\n[1] 开始处理流数据...")
    documents = []
    
    for i, doc_dict in enumerate(data_stream_generator()):
        doc = stream_handler._parse_item(doc_dict)
        if doc:
            stream_handler.add_document(doc)
            documents.append(doc)
            
            if (i + 1) % 5 == 0:
                print(f"   已处理 {i + 1} 个文档")
    
    print(f"\n   总共处理了 {len(documents)} 个文档")
    stats = stream_handler.get_statistics()
    print(f"   统计信息: {stats}")
    
    # 在流数据上执行检索
    print("\n[2] 在流数据上执行检索...")
    retriever.index_documents(documents)
    
    query = "数据分析"
    results = retriever.retrieve(query, top_k=5)
    
    print(f"   查询: '{query}'")
    print(f"   检索到 {len(results)} 个结果:")
    for r in results[:3]:
        print(f"   - {r.document.doc_id}: {r.document.content[:50]}... (Score: {r.score:.3f})")
    
    # 重排序
    print("\n[3] 对结果进行重排序...")
    reranked = reranker.rerank(query, results, top_k=3)
    
    print(f"   重排序后的Top 3:")
    for r in reranked:
        print(f"   - {r.document.doc_id}: Score={r.score:.3f}, "
              f"原排名={r.original_rank} -> 新排名={r.rank}")
    
    print("\n" + "=" * 80)
    print("流式处理示例完成!")
    print("=" * 80)


if __name__ == '__main__':
    main()
