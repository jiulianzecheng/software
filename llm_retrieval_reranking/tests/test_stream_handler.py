"""
测试流数据处理器
"""

import sys
sys.path.insert(0, '/home/runner/work/software/software')

from llm_retrieval_reranking.core.stream_handler import StreamDataHandler, Document


def test_document_creation():
    """测试文档创建"""
    doc = Document("doc1", "测试内容", {"key": "value"})
    
    assert doc.doc_id == "doc1"
    assert doc.content == "测试内容"
    assert doc.metadata["key"] == "value"
    assert doc.timestamp > 0
    print("✓ 文档创建测试通过")


def test_stream_handler_initialization():
    """测试流处理器初始化"""
    handler = StreamDataHandler(buffer_size=100, batch_size=10)
    
    assert handler.buffer_size == 100
    assert handler.batch_size == 10
    assert handler.processed_count == 0
    print("✓ 流处理器初始化测试通过")


def test_add_document():
    """测试添加文档"""
    handler = StreamDataHandler()
    doc = Document("doc1", "内容1")
    
    handler.add_document(doc)
    
    assert len(handler.buffer) == 1
    assert handler.processed_count == 1
    print("✓ 添加文档测试通过")


def test_process_stream():
    """测试处理流数据"""
    handler = StreamDataHandler()
    
    stream_data = [
        {'id': 'doc1', 'content': '内容1'},
        {'id': 'doc2', 'content': '内容2'},
        {'id': 'doc3', 'content': '内容3'},
    ]
    
    documents = handler.process_stream(iter(stream_data))
    
    assert len(documents) == 3
    assert documents[0].doc_id == 'doc1'
    assert documents[1].content == '内容2'
    print("✓ 处理流数据测试通过")


def test_get_recent_documents():
    """测试获取最近文档"""
    handler = StreamDataHandler()
    
    for i in range(5):
        doc = Document(f"doc{i}", f"内容{i}")
        handler.add_document(doc)
    
    recent = handler.get_recent_documents(3)
    
    assert len(recent) == 3
    assert recent[-1].doc_id == "doc4"
    print("✓ 获取最近文档测试通过")


def test_buffer_overflow():
    """测试缓冲区溢出"""
    handler = StreamDataHandler(buffer_size=3)
    
    for i in range(5):
        doc = Document(f"doc{i}", f"内容{i}")
        handler.add_document(doc)
    
    # 缓冲区大小限制为3
    assert len(handler.buffer) == 3
    # 但总处理数为5
    assert handler.processed_count == 5
    print("✓ 缓冲区溢出测试通过")


def test_statistics():
    """测试统计信息"""
    handler = StreamDataHandler(buffer_size=10)
    
    for i in range(5):
        doc = Document(f"doc{i}", f"内容{i}")
        handler.add_document(doc)
    
    stats = handler.get_statistics()
    
    assert stats['total_processed'] == 5
    assert stats['buffer_size'] == 5
    assert stats['buffer_capacity'] == 10
    print("✓ 统计信息测试通过")


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("运行流数据处理器测试")
    print("=" * 60)
    
    test_document_creation()
    test_stream_handler_initialization()
    test_add_document()
    test_process_stream()
    test_get_recent_documents()
    test_buffer_overflow()
    test_statistics()
    
    print("\n" + "=" * 60)
    print("所有测试通过! ✓")
    print("=" * 60)


if __name__ == '__main__':
    run_all_tests()
