"""
配置管理模块

管理系统配置参数。
"""

from typing import Dict, Any


class Config:
    """配置管理类"""
    
    # 默认配置
    DEFAULT_CONFIG = {
        # 流数据处理配置
        'stream': {
            'buffer_size': 1000,
            'batch_size': 100,
        },
        
        # 检索配置
        'retrieval': {
            'embedding_method': 'tfidf',  # 'tfidf' 或 'simple'
            'top_k': 20,
        },
        
        # 重排序配置
        'rerank': {
            'semantic_weight': 0.5,
            'keyword_weight': 0.3,
            'quality_weight': 0.2,
            'top_k': 5,
        },
        
        # LLM接口配置
        'llm': {
            'model_name': 'default',
            'max_context_length': 4096,
            'max_docs': 5,
        }
    }
    
    def __init__(self, custom_config: Dict[str, Any] = None):
        """
        初始化配置
        
        Args:
            custom_config: 自定义配置字典
        """
        self.config = self._deep_merge(self.DEFAULT_CONFIG.copy(), custom_config or {})
    
    def _deep_merge(self, base: dict, override: dict) -> dict:
        """
        深度合并字典
        
        Args:
            base: 基础字典
            override: 覆盖字典
        
        Returns:
            合并后的字典
        """
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def get(self, path: str, default: Any = None) -> Any:
        """
        获取配置值
        
        Args:
            path: 配置路径，用点分隔 (例如: 'stream.buffer_size')
            default: 默认值
        
        Returns:
            配置值
        """
        keys = path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, path: str, value: Any) -> None:
        """
        设置配置值
        
        Args:
            path: 配置路径，用点分隔
            value: 要设置的值
        """
        keys = path.split('.')
        config = self.config
        
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        config[keys[-1]] = value
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        获取整个配置节
        
        Args:
            section: 配置节名称
        
        Returns:
            配置节字典
        """
        return self.config.get(section, {})
    
    def to_dict(self) -> Dict[str, Any]:
        """
        导出为字典
        
        Returns:
            配置字典
        """
        return self.config.copy()
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """
        从字典创建配置对象
        
        Args:
            config_dict: 配置字典
        
        Returns:
            Config对象
        """
        return cls(config_dict)
