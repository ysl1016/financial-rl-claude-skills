#!/usr/bin/env python3
"""
Colab 노트북 생성 스크립트

이 스크립트는 Google Colab에서 사용할 수 있는 학습 노트북 파일들을 생성합니다.
.ipynb 파일이 gitignore에 포함되어 있어 직접 생성할 수 없으므로,
이 스크립트를 실행하여 노트북을 생성하세요.

사용법:
    python scripts/create_colab_notebooks.py
"""

import json
import os
from pathlib import Path


def create_training_notebook():
    """메인 학습 노트북 생성"""
    
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# DeepSeek GRPO Trading Model - Training Notebook\n\n"
                    "이 노트북은 Yahoo Finance 데이터를 사용하여 DeepSeek-R1 기반 GRPO 강화학습 모델을 학습합니다.\n\n"
                    "## 📋 목차\n"
                    "1. 환경 설정 및 GPU 확인\n"
                    "2. Yahoo Finance 데이터 수집\n"
                    "3. 데이터 전처리 및 분할\n"
                    "4. 트레이딩 환경 생성\n"
                    "5. GRPO 에이전트 학습\n"
                    "6. 학습 결과 시각화\n"
                    "7. 간단한 백테스트"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## 1. 환경 설정 및 GPU 확인"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Google Drive 마운트\n",
                    "from google.colab import drive\n",
                    "drive.mount('/content/drive')\n\n",
                    "# 저장 디렉토리 생성\n",
                    "import os\n",
                    "os.makedirs('/content/drive/MyDrive/financial-rl-trading/models/checkpoints', exist_ok=True)\n",
                    "os.makedirs('/content/drive/MyDrive/financial-rl-trading/results', exist_ok=True)\n",
                    "os.makedirs('/content/drive/MyDrive/financial-rl-trading/data/cache', exist_ok=True)\n",
                    "print('✓ Google Drive 마운트 및 디렉토리 생성 완료')"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 프로젝트 클론 (GitHub에 업로드한 경우)\n",
                    "# !git clone https://github.com/[your-username]/financial-rl-claude-skills.git\n",
                    "# %cd financial-rl-claude-skills\n\n",
                    "# 또는 로컬 파일을 압축하여 업로드한 경우:\n",
                    "# !unzip /content/drive/MyDrive/financial-rl-claude-skills.zip\n",
                    "# %cd financial-rl-claude-skills"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 필수 패키지 설치\n",
                    "!pip install -q yfinance>=0.1.70 pandas numpy matplotlib seaborn\n",
                    "!pip install -q torch>=2.0.0 gym>=0.21.0\n",
                    "!pip install -q scikit-learn scipy tqdm\n",
                    "print('✓ 패키지 설치 완료')"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# GPU 확인\n",
                    "import torch\n",
                    "import sys\n\n",
                    "print('='*60)\n",
                    "print('환경 정보')\n",
                    "print('='*60)\n",
                    "print(f'Python: {sys.version}')\n",
                    "print(f'PyTorch: {torch.__version__}')\n",
                    "print(f'CUDA 사용 가능: {torch.cuda.is_available()}')\n\n",
                    "if torch.cuda.is_available():\n",
                    "    print(f'GPU: {torch.cuda.get_device_name(0)}')\n",
                    "    print(f'GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')\n",
                    "    device = 'cuda'\n",
                    "else:\n",
                    "    print('⚠️  GPU 없음 - CPU로 학습 (느림)')\n",
                    "    device = 'cpu'\n\n",
                    "print(f'\\n사용 디바이스: {device}')\n",
                    "print('='*60)"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## 2. Yahoo Finance 데이터 수집"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import yfinance as yf\n",
                    "import pandas as pd\n",
                    "import numpy as np\n",
                    "from datetime import datetime\n\n",
                    "# 설정\n",
                    "SYMBOLS = ['SPY']  # S&P 500 ETF\n",
                    "START_DATE = '2020-01-01'\n",
                    "END_DATE = datetime.now().strftime('%Y-%m-%d')\n\n",
                    "print(f'Yahoo Finance 데이터 다운로드')\n",
                    "print(f'종목: {SYMBOLS}')\n",
                    "print(f'기간: {START_DATE} ~ {END_DATE}')"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "def download_and_process_data(symbol, start_date, end_date):\n",
                    "    '''Yahoo Finance 데이터 다운로드 및 기술적 지표 계산'''\n",
                    "    ticker = yf.Ticker(symbol)\n",
                    "    data = ticker.history(start=start_date, end=end_date)\n",
                    "    \n",
                    "    if len(data) == 0:\n",
                    "        raise ValueError(f'데이터 없음: {symbol}')\n",
                    "    \n",
                    "    # 이동평균\n",
                    "    data['SMA_20'] = data['Close'].rolling(20).mean()\n",
                    "    data['SMA_50'] = data['Close'].rolling(50).mean()\n",
                    "    \n",
                    "    # RSI\n",
                    "    delta = data['Close'].diff()\n",
                    "    gain = delta.where(delta > 0, 0).rolling(14).mean()\n",
                    "    loss = -delta.where(delta < 0, 0).rolling(14).mean()\n",
                    "    data['RSI'] = 100 - (100 / (1 + gain / loss))\n",
                    "    \n",
                    "    # MACD\n",
                    "    ema12 = data['Close'].ewm(span=12).mean()\n",
                    "    ema26 = data['Close'].ewm(span=26).mean()\n",
                    "    data['MACD'] = ema12 - ema26\n",
                    "    data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()\n",
                    "    \n",
                    "    # Bollinger Bands\n",
                    "    sma20 = data['Close'].rolling(20).mean()\n",
                    "    std20 = data['Close'].rolling(20).std()\n",
                    "    data['BB_Upper'] = sma20 + (std20 * 2)\n",
                    "    data['BB_Lower'] = sma20 - (std20 * 2)\n",
                    "    data['BB_Width'] = (data['BB_Upper'] - data['BB_Lower']) / sma20\n",
                    "    \n",
                    "    # ATR\n",
                    "    hl = data['High'] - data['Low']\n",
                    "    hc = (data['High'] - data['Close'].shift()).abs()\n",
                    "    lc = (data['Low'] - data['Close'].shift()).abs()\n",
                    "    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)\n",
                    "    data['ATR'] = tr.rolling(14).mean()\n",
                    "    \n",
                    "    # 수익률 및 변동성\n",
                    "    data['Returns'] = data['Close'].pct_change()\n",
                    "    data['Volatility'] = data['Returns'].rolling(20).std()\n",
                    "    \n",
                    "    return data.dropna()\n\n",
                    "# 다운로드\n",
                    "datasets = {}\n",
                    "for symbol in SYMBOLS:\n",
                    "    print(f'\\n{symbol} 다운로드 중...')\n",
                    "    datasets[symbol] = download_and_process_data(symbol, START_DATE, END_DATE)\n",
                    "    print(f'  ✓ {len(datasets[symbol])} 거래일')\n\n",
                    "print('\\n✓ 데이터 다운로드 완료!')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. 데이터 전처리 및 분할\n\n",
                    "시계열 데이터는 순차적으로 분할해야 합니다 (랜덤 분할 X)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "def split_data(data, train_ratio=0.7, val_ratio=0.15):\n",
                    "    n = len(data)\n",
                    "    train_end = int(n * train_ratio)\n",
                    "    val_end = int(n * (train_ratio + val_ratio))\n",
                    "    return data.iloc[:train_end], data.iloc[train_end:val_end], data.iloc[val_end:]\n\n",
                    "data = datasets[SYMBOLS[0]]\n",
                    "train_data, val_data, test_data = split_data(data)\n\n",
                    "print(f'데이터 분할:')\n",
                    "print(f'  학습: {len(train_data)} 일 ({train_data.index[0].date()} ~ {train_data.index[-1].date()})')\n",
                    "print(f'  검증: {len(val_data)} 일 ({val_data.index[0].date()} ~ {val_data.index[-1].date()})')\n",
                    "print(f'  테스트: {len(test_data)} 일 ({test_data.index[0].date()} ~ {test_data.index[-1].date()})')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 4-7. 모델 학습 및 평가\n\n",
                    "**⚠️ 중요**: 전체 학습 코드는 프로젝트의 실제 모델을 사용해야 합니다.\n\n",
                    "다음 중 하나를 선택하세요:\n\n",
                    "### 옵션 A: 프로젝트 전체 사용 (권장)\n",
                    "```python\n",
                    "from src.models.trading_env import TradingEnv\n",
                    "from src.models.deepseek_grpo_agent import DeepSeekGRPOAgent\n",
                    "from src.utils.evaluation import calculate_performance_summary\n",
                    "```\n\n",
                    "### 옵션 B: 기존 학습 스크립트 실행\n",
                    "```bash\n",
                    "!python examples/train_grpo.py --symbol SPY --episodes 100 --device cuda\n",
                    "```\n\n",
                    "### 옵션 C: 백테스트 스크립트 사용\n",
                    "```bash\n",
                    "!python examples/backtest_deepseek_grpo.py \\\n",
                    "    --model_path /content/drive/MyDrive/models/best_model.pt \\\n",
                    "    --symbols SPY \\\n",
                    "    --plot \\\n",
                    "    --mc_simulations 100\n",
                    "```\n\n",
                    "자세한 내용은 implementation_plan.md를 참조하세요."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "print('✓ 노트북 준비 완료!')\n",
                    "print('\\n다음 단계:')\n",
                    "print('1. 프로젝트를 GitHub에 업로드하거나 zip으로 압축')\n",
                    "print('2. Colab에서 프로젝트 클론 또는 압축 해제')\n",
                    "print('3. GPU 런타임 선택 (런타임 > 런타임 유형 변경 > T4 GPU)')\n",
                    "print('4. 위의 옵션 A, B, C 중 하나로 학습 실행')\n",
                    "print('\\n예상 시간 (100 episodes):')\n",
                    "print('  GPU (T4): 20-30분')\n",
                    "print('  CPU: 2-3시간')"
                ]
            }
        ],
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "gpuType": "T4",
                "provenance": []
            },
            "kernelspec": {
                "display_name": "Python 3",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }
    
    return notebook


def save_notebook(notebook, filename):
    """노트북을 JSON 파일로 저장"""
    output_dir = Path("colab_notebooks")
    output_dir.mkdir(exist_ok=True)
    
    filepath = output_dir / filename
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
    
    return filepath


def main():
    """메인 함수"""
    print("=" * 60)
    print("Colab 노트북 생성 스크립트")
    print("=" * 60)
    
    # 학습 노트북 생성
    print("\n1. 학습 노트북 생성 중...")
    training_nb = create_training_notebook()
    training_path = save_notebook(training_nb, "colab_training.ipynb")
    print(f"   ✓ 생성 완료: {training_path}")
    
    print("\n" + "=" * 60)
    print("노트북 생성 완료!")
    print("=" * 60)
    print("\n사용 방법:")
    print("1. colab_notebooks/ 폴더의 .ipynb 파일을 Google Colab에 업로드")
    print("2. 런타임 > 런타임 유형 변경 > T4 GPU 선택")
    print("3. 모든 셀 실행")
    print("\n또는:")
    print("- Google Drive에서 직접 열기")
    print("- GitHub에 업로드 후 Colab에서 열기")
    print("\n자세한 내용은 implementation_plan.md를 참조하세요.")


if __name__ == "__main__":
    main()
