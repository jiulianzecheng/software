"""
运行所有测试
"""

import sys
sys.path.insert(0, '/home/runner/work/software/software')

# 导入所有测试模块
from test_stream_handler import run_all_tests as test_stream_handler
from test_retriever import run_all_tests as test_retriever
from test_reranker import run_all_tests as test_reranker
from test_integration import run_all_tests as test_integration


def main():
    """运行所有测试套件"""
    print("\n" + "=" * 80)
    print(" " * 25 + "LLM检索重排系统 - 完整测试套件")
    print("=" * 80 + "\n")
    
    test_modules = [
        ("流数据处理器", test_stream_handler),
        ("检索器", test_retriever),
        ("重排序器", test_reranker),
        ("集成测试", test_integration),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in test_modules:
        print(f"\n{'='*80}")
        print(f"正在运行: {name}测试")
        print(f"{'='*80}\n")
        
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n✗ {name}测试失败: {e}")
            failed += 1
    
    print("\n" + "=" * 80)
    print(" " * 30 + "测试总结")
    print("=" * 80)
    print(f"通过: {passed}/{len(test_modules)}")
    print(f"失败: {failed}/{len(test_modules)}")
    
    if failed == 0:
        print("\n✓ 所有测试通过!")
    else:
        print(f"\n✗ {failed} 个测试失败")
    
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
