#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Configuration Management

.env 파일에서 환경 변수를 안전하게 로드하고 관리하는 모듈입니다.
"""

import os
from pathlib import Path
from typing import Any, Optional, Dict
import warnings


class Config:
    """
    환경 변수 기반 설정 관리 클래스

    .env 파일에서 설정을 로드하거나 환경 변수에서 읽어옵니다.
    """

    _instance = None
    _loaded = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._loaded:
            self._load_env()
            self._loaded = True

    def _load_env(self):
        """
        .env 파일에서 환경 변수 로드

        python-dotenv 패키지가 설치되어 있으면 사용하고,
        없으면 수동으로 .env 파일을 파싱합니다.
        """
        # 프로젝트 루트 디렉토리 찾기
        current_file = Path(__file__)
        project_root = current_file.parent.parent.parent
        env_file = project_root / '.env'

        # .env 파일 존재 확인
        if not env_file.exists():
            warnings.warn(
                f".env file not found at {env_file}. "
                f"Copy .env.example to .env and configure your API keys."
            )
            return

        # python-dotenv 사용 시도
        try:
            from dotenv import load_dotenv
            load_dotenv(env_file)
            print(f"✓ Loaded configuration from {env_file}")
            return
        except ImportError:
            pass

        # 수동으로 .env 파싱
        try:
            with open(env_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()

                    # 빈 줄이나 주석 무시
                    if not line or line.startswith('#'):
                        continue

                    # KEY=VALUE 형식 파싱
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()

                        # 따옴표 제거
                        if value.startswith('"') and value.endswith('"'):
                            value = value[1:-1]
                        elif value.startswith("'") and value.endswith("'"):
                            value = value[1:-1]

                        # 환경 변수에 설정 (기존 값 우선)
                        if key not in os.environ:
                            os.environ[key] = value

            print(f"✓ Loaded configuration from {env_file}")

        except Exception as e:
            warnings.warn(f"Failed to load .env file: {e}")

    @staticmethod
    def get(key: str, default: Any = None, required: bool = False) -> Any:
        """
        환경 변수 가져오기

        Args:
            key: 환경 변수 이름
            default: 기본값 (변수가 없을 때)
            required: 필수 여부 (True이면 없을 때 오류 발생)

        Returns:
            환경 변수 값 또는 기본값

        Raises:
            ValueError: required=True인데 변수가 없을 때
        """
        value = os.environ.get(key, default)

        if required and value is None:
            raise ValueError(
                f"Required environment variable '{key}' is not set. "
                f"Please set it in .env file or environment."
            )

        return value

    @staticmethod
    def get_int(key: str, default: int = 0) -> int:
        """정수형 환경 변수 가져오기"""
        value = Config.get(key, default)
        try:
            return int(value)
        except (ValueError, TypeError):
            return default

    @staticmethod
    def get_float(key: str, default: float = 0.0) -> float:
        """실수형 환경 변수 가져오기"""
        value = Config.get(key, default)
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    @staticmethod
    def get_bool(key: str, default: bool = False) -> bool:
        """불리언 환경 변수 가져오기"""
        value = Config.get(key, str(default)).lower()
        return value in ('true', 'yes', '1', 'on')

    @staticmethod
    def get_list(key: str, default: Optional[list] = None, separator: str = ',') -> list:
        """리스트 환경 변수 가져오기 (쉼표 구분)"""
        value = Config.get(key)
        if value is None:
            return default or []
        return [item.strip() for item in value.split(separator)]

    @classmethod
    def get_all(cls) -> Dict[str, str]:
        """모든 환경 변수 반환 (민감 정보 마스킹)"""
        sensitive_keys = ['API_KEY', 'SECRET', 'PASSWORD', 'TOKEN']

        result = {}
        for key, value in os.environ.items():
            # 민감한 키는 마스킹
            if any(sensitive in key.upper() for sensitive in sensitive_keys):
                if value:
                    result[key] = value[:8] + '...' + value[-4:] if len(value) > 12 else '***'
                else:
                    result[key] = '(not set)'
            else:
                result[key] = value

        return result


# 설정 인스턴스 (싱글톤)
config = Config()


# 편의 함수들
def get_anthropic_api_key() -> str:
    """Anthropic API 키 가져오기"""
    return config.get('ANTHROPIC_API_KEY', required=True)


def get_claude_model() -> str:
    """Claude 모델 이름 가져오기"""
    return config.get('CLAUDE_MODEL', 'claude-3-7-sonnet-20250219')


def get_trading_config() -> Dict[str, Any]:
    """트레이딩 설정 가져오기"""
    return {
        'initial_capital': config.get_float('INITIAL_CAPITAL', 100000.0),
        'trading_cost': config.get_float('TRADING_COST', 0.0005),
        'slippage': config.get_float('SLIPPAGE', 0.0001),
        'risk_free_rate': config.get_float('RISK_FREE_RATE', 0.02),
    }


def get_claude_config() -> Dict[str, Any]:
    """Claude 설정 가져오기"""
    return {
        'model': get_claude_model(),
        'max_tokens': config.get_int('CLAUDE_MAX_TOKENS', 2048),
        'temperature': config.get_float('CLAUDE_TEMPERATURE', 0.7),
        'consultation_frequency': config.get_int('CLAUDE_CONSULTATION_FREQUENCY', 20),
    }


def get_device() -> str:
    """계산 디바이스 가져오기 (cuda 또는 cpu)"""
    device = config.get('DEVICE', 'cuda')

    # PyTorch로 실제 사용 가능한지 확인
    try:
        import torch
        if device == 'cuda' and not torch.cuda.is_available():
            print("⚠ CUDA not available. Falling back to CPU.")
            return 'cpu'
    except ImportError:
        pass

    return device


def validate_config() -> bool:
    """
    설정 유효성 검증

    Returns:
        bool: 모든 필수 설정이 올바르면 True
    """
    try:
        # 필수 API 키 확인
        api_key = config.get('ANTHROPIC_API_KEY')
        if not api_key or api_key == 'your-api-key-here':
            print("❌ ANTHROPIC_API_KEY is not set or using default value")
            print("   Please set your API key in .env file")
            return False

        # API 키 형식 확인
        if not api_key.startswith('sk-ant-'):
            print("❌ ANTHROPIC_API_KEY format looks incorrect")
            print("   It should start with 'sk-ant-'")
            return False

        print("✓ Configuration validated successfully")
        print(f"  API Key: {api_key[:10]}...{api_key[-4:]}")
        print(f"  Claude Model: {get_claude_model()}")
        print(f"  Device: {get_device()}")

        return True

    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return False


def print_config_summary():
    """설정 요약 출력 (디버깅용)"""
    print("\n" + "="*60)
    print("Configuration Summary")
    print("="*60)

    # API 설정
    print("\n[API Configuration]")
    api_key = config.get('ANTHROPIC_API_KEY', '(not set)')
    if api_key and api_key != '(not set)':
        print(f"  Anthropic API Key: {api_key[:10]}...{api_key[-4:]}")
    else:
        print(f"  Anthropic API Key: (not set)")
    print(f"  Claude Model: {get_claude_model()}")

    # 트레이딩 설정
    print("\n[Trading Configuration]")
    trading_config = get_trading_config()
    print(f"  Initial Capital: ${trading_config['initial_capital']:,.2f}")
    print(f"  Trading Cost: {trading_config['trading_cost']*100:.3f}%")
    print(f"  Slippage: {trading_config['slippage']*100:.3f}%")
    print(f"  Risk-Free Rate: {trading_config['risk_free_rate']*100:.1f}%")

    # Claude 설정
    print("\n[Claude Configuration]")
    claude_config = get_claude_config()
    print(f"  Max Tokens: {claude_config['max_tokens']}")
    print(f"  Temperature: {claude_config['temperature']}")
    print(f"  Consultation Frequency: {claude_config['consultation_frequency']} steps")

    # 시스템 설정
    print("\n[System Configuration]")
    print(f"  Device: {get_device()}")
    print(f"  Log Level: {config.get('LOG_LEVEL', 'INFO')}")
    print(f"  Default Symbol: {config.get('DEFAULT_SYMBOL', 'SPY')}")

    print("\n" + "="*60 + "\n")


if __name__ == '__main__':
    # 테스트
    print("Testing configuration...")
    print_config_summary()
    validate_config()
