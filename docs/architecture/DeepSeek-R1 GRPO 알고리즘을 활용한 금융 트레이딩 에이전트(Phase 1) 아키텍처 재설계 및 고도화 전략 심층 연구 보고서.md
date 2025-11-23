

# **DeepSeek-R1 GRPO 알고리즘을 활용한 금융 트레이딩 에이전트(Phase 1\) 아키텍처 재설계 및 고도화 전략 심층 연구 보고서**

## **1\. 서론: 금융 AI의 인지적 전환과 추론형 에이전트의 필요성**

금융 시장은 본질적으로 높은 불확실성(Stochasticity)과 비정상성(Non-stationarity)을 특징으로 하는 복잡계(Complex System)입니다. 기존의 정량적 트레이딩(Quantitative Trading) 모델과 초기 단계의 강화학습(Reinforcement Learning, RL) 에이전트들은 이러한 시장 환경에서 시계열 패턴을 인식하고 즉각적인 보상(수익률)을 최적화하는 데 주력해 왔습니다. 그러나 귀하의 financial-rl-claude-skills 프로젝트에서 관찰된 Phase 3의 정책 붕괴(Policy Collapse) 현상, 즉 \-12.37%의 손실과 214회의 매도 쏠림 현상은 기존 PPO(Proximal Policy Optimization) 기반 접근법의 근본적인 한계를 드러내고 있습니다.1 이는 모델이 시장의 노이즈를 신호로 오인하거나, 하락장에서의 손실 회피 성향이 과도하게 학습되어 아무것도 하지 않거나 투매하는 '동결 상태(Freezing)' 또는 '공황 상태(Panic Selling)'에 빠진 것으로 진단할 수 있습니다.1

2025년 초 등장한 DeepSeek-R1과 그 핵심 알고리즘인 \*\*GRPO(Group Relative Policy Optimization)\*\*는 이러한 한계를 극복할 수 있는 새로운 패러다임을 제시합니다. DeepSeek-R1은 단순히 정답을 맞히는 것을 넘어, 사고 사슬(Chain of Thought, CoT)을 통해 문제 해결의 논리적 경로를 스스로 생성하고 검증하는 능력을 보여주었습니다.2 금융 도메인에서 이는 에이전트가 단순히 "매수" 또는 "매도"라는 행동을 출력하는 것을 넘어, "현재 RSI가 30 이하이고 거시경제 지표가 긍정적이므로, 단기적 과매도 상태로 판단하여 매수한다"는 식의 \*\*설명 가능한 추론(Reasoning)\*\*을 수행할 수 있음을 의미합니다.

본 보고서는 제공된 프로젝트의 현황을 의학적 정밀 진단 수준으로 분석하고, 2025년 10월부터 11월 사이에 arXiv에 발표된 최신 선행 연구들(Trading-R1, Fin-R1, Training-Free GRPO 등)을 토대로 Phase 1 모델의 근본적인 개선안을 제시합니다. 특히 비평가(Critic) 모델을 제거하고 그룹 단위의 상대적 우위(Relative Advantage)를 활용하는 GRPO의 메커니즘이 금융 시장의 노이즈 문제를 어떻게 해결할 수 있는지 수학적, 구조적으로 규명하며, 이를 실제 코드베이스에 적용하기 위한 구체적인 아키텍처와 보상 함수(Reward Function) 설계안을 상세히 기술합니다.

---

## **2\. 현행 모델(financial-rl-claude-skills)의 병리학적 진단 및 구조적 한계**

귀하가 공유한 Phase 별 백테스팅 결과와 코드베이스 구조는 현재 시스템이 겪고 있는 문제의 심각성을 명확히 보여줍니다. 이를 단순히 파라미터 튜닝의 문제로 치부해서는 안 되며, 모델의 학습 메커니즘 자체가 금융 데이터의 특성과 불일치하고 있음을 인식해야 합니다.

### **2.1. Phase 3 성과 분석: 정책 붕괴(Policy Collapse)의 메커니즘**

Phase 3에서 기록된 \-12.37%의 손실과 "0회 매수 / 214회 매도"라는 극단적인 행동 불균형은 강화학습에서 흔히 발생하는 \*\*정책 붕괴(Policy Collapse)\*\*의 전형적인 사례입니다.1

이 현상의 원인은 크게 세 가지로 분석됩니다.  
첫째, \*\*보상 함수의 비대칭성(Reward Asymmetry)\*\*입니다. 하락장에서 포지션을 보유함으로써 발생하는 손실에 대한 페널티가 과도하게 설정되었을 가능성이 큽니다. 에이전트는 손실을 만회하기 위해 노력하기보다는, 아예 포지션을 청산하고 시장에서 이탈하는 것이 기대 보상(Expected Reward)을 최대화(정확히는 손실을 최소화)하는 유일한 방법이라고 잘못 학습한 것입니다. 이는 생물학적 시스템에서 극심한 스트레스 상황 시 유기체가 가사 상태(Suspended Animation)에 빠지는 것과 유사한 방어 기제입니다.  
둘째, **비평가(Critic) 네트워크의 가치 추정 실패**입니다. 기존의 Actor-Critic 구조(PPO 등)에서 Critic은 현재 상태 $S\_t$의 가치 $V(S\_t)$를 정확히 예측해야 합니다. 그러나 금융 시장의 확률적 변동성(Stochasticity)으로 인해 Critic은 수렴하지 못하고 발산하거나, 모든 상태의 가치를 부정적으로 평가하는 편향(Bias)을 갖게 됩니다.4 Critic이 붕괴되면 Actor는 올바른 정책 그라디언트(Policy Gradient)를 전달받지 못하게 되고, 결국 학습이 중단되거나 엉뚱한 방향으로 최적화됩니다.

셋째, **단일 경로(Single Trajectory) 학습의 한계**입니다. 현재 모델은 한 번의 에피소드에서 발생한 하나의 경로만을 바탕으로 학습합니다. 금융 시장은 동일한 조건에서도 다양한 시나리오가 전개될 수 있는 공간입니다. 단일 경로 학습은 특정 시점의 우연한 수익이나 손실에 모델을 과적합(Overfitting)시켜, 일반화 성능을 저하시키는 주된 요인으로 작용합니다.1

### **2.2. 아키텍처의 구조적 결함**

현재의 financial-rl-claude-skills 구조는 LLM(Claude)의 정성적 분석을 통합하려는 시도에도 불구하고, 이를 에이전트의 행동과 유기적으로 연결하지 못하고 있습니다.

| 구분 | 현행 모델의 특징 | 문제점 및 한계 |
| :---- | :---- | :---- |
| **입력 처리** | 텍스트 데이터를 임베딩하여 상태(State) 벡터에 단순 병합 | 텍스트가 내포한 인과관계나 시장의 심리를 에이전트가 '해석'하지 못하고 단순 노이즈로 처리할 위험이 큼.1 |
| **정책 결정** | 상태 $S$ → 행동 $A$ (Black-box 매핑) | "왜" 매매해야 하는지에 대한 중간 추론 과정이 부재하여, 행동의 타당성을 검증하거나 제어할 수 없음.7 |
| **학습 방식** | PPO 기반 Actor-Critic | Critic의 높은 분산(High Variance)으로 인해 학습이 불안정하며, 대규모 연산 자원이 필요함.4 |
| **보상 체계** | 단순 수익률(Profit & Loss) 중심 | 위험(Risk)과 변동성(Volatility)을 고려하지 않아, 수익률을 쫓다가 파산(Ruin)할 위험이 높음.9 |

이러한 진단 결과는 기존의 단순한 RL 모델을 폐기하고, \*\*추론 능력(Reasoning Capability)\*\*과 \*\*그룹 기반 최적화(Group Optimization)\*\*를 갖춘 차세대 아키텍처로의 전환이 시급함을 시사합니다.

---

## **3\. DeepSeek-R1 GRPO 알고리즘: 이론적 심층 분석 및 금융 도메인 적합성**

DeepSeek-R1이 도입한 \*\*GRPO(Group Relative Policy Optimization)\*\*는 PPO의 구조적 한계를 극복하고, LLM의 추론 능력을 강화학습에 접목시킨 혁신적인 알고리즘입니다. 이는 특히 정답이 모호하고 보상이 희소한 금융 트레이딩 분야에 최적화된 특성을 가지고 있습니다.

### **3.1. GRPO의 수학적 원리와 Critic 제거의 혁신성**

GRPO의 가장 큰 특징은 **비평가(Critic) 모델의 제거**입니다. 전통적인 PPO는 가치 함수 $V(s)$를 학습하기 위해 별도의 신경망(Critic)을 필요로 하며, 이는 메모리 사용량을 증가시키고 학습의 복잡도를 높입니다. GRPO는 대신 \*\*그룹 상대적 우위(Group Relative Advantage)\*\*라는 개념을 도입하여 이 문제를 해결합니다.2

#### **3.1.1. 그룹 샘플링 및 이점(Advantage) 계산**

동일한 입력(상태) $q$에 대해, 정책 $\\pi\_{\\theta\_{old}}$로부터 $G$개의 서로 다른 출력(행동 경로) ${o\_1, o\_2,..., o\_G}$를 샘플링합니다. 각 출력 $o\_i$에 대해 보상 $r\_i$가 주어졌을 때, 해당 그룹 내에서의 상대적 이점 $A\_i$는 다음과 같이 표준화(Normalization)를 통해 계산됩니다10:

$$A\_i \= \\frac{r\_i \- \\text{mean}(\\{r\_1,..., r\_G\\})}{\\text{std}(\\{r\_1,..., r\_G\\}) \+ \\epsilon}$$  
여기서 $\\text{mean}$과 $\\text{std}$는 해당 그룹 $G$개 샘플의 보상 평균과 표준편차입니다. 이 수식이 갖는 금융적 함의는 매우 큽니다. 시장 전체가 폭락하여 모든 전략의 절대 수익률($r\_i$)이 음수(-)인 상황을 가정해 봅시다. 기존 PPO의 Critic은 이를 "나쁜 상태"로만 평가하여 학습을 저해할 수 있습니다. 그러나 GRPO에서는 그룹 평균보다 덜 손실을 본 전략(예: 헷징이나 현금 비중 확대)은 양(+)의 이점($A\_i \> 0$)을 갖게 됩니다. 즉, \*\*"최악의 상황에서 최선을 선택하는 능력"\*\*을 학습할 수 있게 되는 것입니다.4

#### **3.1.2. 목적 함수(Objective Function)와 KL 발산**

GRPO의 목적 함수는 PPO와 유사하게 클리핑(Clipping) 기법을 사용하여 정책의 급격한 변화를 방지하지만, Critic의 가치 오차항이 제거되고 KL 발산(Kullback-Leibler Divergence) 항이 직접 포함됩니다.3

$$J\_{GRPO}(\\theta) \= \\mathbb{E}\_{q \\sim P(Q), \\{o\_i\\}\_{i=1}^G \\sim \\pi\_{\\theta\_{old}}(O|q)} \\left$$  
이 식에서 $D\_{KL}(\\pi\_\\theta |

| \\pi\_{ref})$는 학습 중인 정책이 초기 참조 모델(Reference Model)이나 SFT 모델에서 너무 멀어지지 않도록 규제하는 역할을 합니다. 금융 모델링에서 이는 에이전트가 초기에 학습한 기본적인 시장 원칙(Technical Analysis Rules 등)을 망각하지 않도록 안전장치 역할을 수행합니다.13

### **3.2. 2025년 최신 연구 동향과 GRPO의 확장성**

본 보고서 작성을 위해 검토한 2025년 10월\~11월의 arXiv 논문들은 GRPO가 금융 도메인에서 강력한 성능을 발휘할 수 있음을 입증하는 다양한 사례를 제시하고 있습니다.

* **Trading-R1 (arXiv:2509.11420):** 이 연구는 DeepSeek-R1과 유사한 방법론을 적용하되, 금융 특화된 \*\*3단계 커리큘럼 학습(Structure \-\> Claims \-\> Decision)\*\*을 제안했습니다. 특히 주목할 점은 \*\*변동성 조정 보상(Volatility-Adjusted Reward)\*\*을 도입하여, 단순 수익률이 아닌 위험 대비 수익률(Sharpe Ratio 등)을 최적화하도록 설계했다는 점입니다. 이는 귀하의 모델이 Phase 3에서 겪은 과적합 문제를 해결할 중요한 단서입니다.15  
* **Fin-R1 (arXiv:2503.16252):** 7B 파라미터 수준의 소형 모델(SLM)에서도 GRPO를 통해 대형 모델(DeepSeek-R1)에 버금가는 추론 능력을 확보할 수 있음을 증명했습니다. 이는 막대한 컴퓨팅 자원 없이도 고성능 금융 에이전트를 구축할 수 있음을 시사하며, 데이터 증류(Data Distillation)와 고품질 CoT 데이터셋의 중요성을 강조합니다.17  
* **Training-Free GRPO (arXiv:2510.08191):** 파라미터 업데이트 없이 문맥(Context) 내에서 경험적 지식을 \*\*'토큰 사전(Token Prior)'\*\*으로 축적하여 성능을 개선하는 방법론입니다. 이는 실시간으로 시장 상황이 변할 때마다 모델을 재학습(Fine-tuning)하는 비용을 절감하고, 즉각적인 적응(Adaptation)을 가능하게 하는 혁신적인 접근법입니다. 귀하의 프로젝트 초기 단계에서 빠른 실험을 위해 이 방법을 우선적으로 적용해 볼 수 있습니다.19

---

## **4\. Phase 1 모델 개선을 위한 상세 실행 계획 (Action Plan): "Critic-less Reasoning Trader" 구축**

위의 이론적 배경과 최신 연구 성과를 바탕으로, financial-rl-claude-skills 프로젝트의 Phase 1 모델을 \*\*"Critic-less Reasoning Trader"\*\*로 재구축하기 위한 구체적인 실행 계획을 수립합니다.

### **4.1. 아키텍처 리팩토링: 사고하는 에이전트(Reasoning Agent)로의 진화**

기존의 상태-행동(State-Action) 매핑 구조를 탈피하고, 중간에 \*\*사고 과정(Chain of Thought)\*\*이 개입하는 새로운 파이프라인을 설계합니다.

#### **4.1.1. 멀티모달 입력 임베딩 (Multi-modal Input Embedding)**

입력 데이터의 풍부함을 극대화하기 위해 수치 데이터와 텍스트 데이터를 결합합니다.

* **시계열 수치 데이터:** 가격, 거래량, 기술적 지표(RSI, MACD, Bollinger Bands 등)를 LSTM이나 Transformer Encoder를 통해 고차원 벡터로 변환합니다.  
* **비정형 텍스트 데이터:** Claude API를 통해 생성된 시장 요약, 뉴스 헤드라인, 거시경제 리포트를 사전 학습된 금융 LLM(예: FinBERT 또는 경량화된 DeepSeek)을 통해 임베딩합니다.  
* **통합(Fusion):** 두 벡터를 결합(Concatenation)하고, 어텐션 메커니즘(Attention Mechanism)을 적용하여 현재 시장 상황에서 더 중요한 정보에 가중치를 부여합니다.15

#### **4.1.2. 정책 네트워크(Policy Network)의 2단계 출력 구조**

정책 네트워크는 한 번에 행동을 결정하지 않고, \*\*추론(Reasoning)\*\*과 \*\*결정(Decision)\*\*의 두 단계를 거치도록 설계합니다.1

1. **추론 헤드(Reasoning Head):** 입력 상태를 바탕으로 현재 시장 상황에 대한 분석과 매매 근거를 텍스트 토큰(CoT) 형태로 생성합니다.  
   * *예시 Output:* \<think\> "현재 S\&P500 지수가 20일 이동평균선을 하향 돌파했으나, RSI는 25로 과매도 구간에 진입했다. 또한, 연준의 금리 동결 뉴스가 긍정적으로 작용할 가능성이 있다. 단기 반등을 노린 매수 전략이 유효해 보인다." \</think\>  
2. **행동 헤드(Action Head):** 생성된 추론 토큰과 원본 상태 벡터를 함께 입력받아 최종 매매 행동(Buy, Sell, Hold)과 포지션 크기(Allocation)를 결정합니다.  
   * *예시 Output:* \<answer\> Action: Buy, Size: 0.5 (50% of budget) \</answer\>

이러한 구조는 모델의 행동을 해석 가능하게 만들 뿐만 아니라, 논리적 비약이나 환각(Hallucination)을 방지하는 효과가 있습니다.6

### **4.2. GRPO 알고리즘의 금융 특화 구현**

보고서1에서 제안된 GRPO 적용 전략을 구체적인 코드 레벨의 로직으로 구체화합니다.

#### **4.2.1. 그룹 샘플링(Group Sampling) 전략**

금융 시장의 불확실성을 반영하기 위해, 단일 시점 $t$에서 $G$개의 병렬 시뮬레이션을 수행합니다. 권장되는 그룹 크기 $G$는 8에서 16입니다.

* **다양성 확보(Diversity):** 각 샘플링 시 Dropout 마스크를 다르게 적용하거나, LLM의 Temperature 파라미터를 조정(0.7\~1.0)하여 다양한 추론 경로를 생성합니다. 이는 에이전트가 "보수적 관점", "공격적 관점", "중립적 관점" 등 다양한 시나리오를 검토하게 만듭니다.  
* **Trading-R1의 접근법 차용:** 단순한 무작위 샘플링을 넘어, "Best-of-N" 전략을 변형하여 상위 $K$개의 추론 경로를 선택하고 이를 앙상블(Ensemble)하는 방식도 고려할 수 있습니다.24

#### **4.2.2. 금융 특화 상대적 이점(Relative Advantage) 산출**

GRPO의 핵심인 Advantage 계산을 금융 트레이딩의 특성에 맞춰 변형합니다. 단순 수익률뿐만 아니라 위험을 고려한 지표를 사용해야 합니다.

* **Z-Score 기반 Advantage:**  
  $$A\_i \= \\frac{R\_i \- \\mu\_G}{\\sigma\_G \+ \\epsilon}$$  
  여기서 $R\_i$는 각 샘플 경로의 보상입니다. $\\mu\_G$와 $\\sigma\_G$는 그룹 내 평균과 표준편차입니다. 이 방식은 시장 상황(Bull/Bear)에 따른 편향을 제거하는 정규화(Normalization) 효과를 가집니다.4  
* **보상 $R\_i$의 구성:** $R\_i$는 단일 지표가 아닌 복합 지표로 구성되어야 합니다.  
  * **Step-aware Supervision:** 트레이딩의 결과(Outcome)뿐만 아니라, 과정(Process)에 대한 보상을 포함합니다. 예를 들어, 추론 내용과 행동의 일치 여부(Consistency Check)를 보상에 반영합니다.23

### **4.3. 보상 함수(Reward Function)의 다차원적 고도화**

최신 연구(arXiv 2025\)9를 반영하여, 단순히 수익을 쫓는 것이 아니라 '생존'과 '안정성'을 우선순위에 둔 보상 함수를 설계합니다.

| 구성 요소 | 수식 및 구현 로직 | 가중치 (w) | 의도 및 효과 |
| :---- | :---- | :---- | :---- |
| **순수익률 (Log Return)** | $r\_t \= \\ln(P\_{t+1}/P\_t)$ | $1.0$ | 기본적인 자산 증식 목표 달성. |
| **Sortino Ratio 보상** | $R\_{risk} \= \\frac{\\text{Mean}(r)}{\\text{DownsideStd}(r)}$ | $0.5$ | 상방 변동성은 허용하되, 하방 위험(Downside Risk)에 대해서만 페널티를 부여하여 공격적인 수익 추구와 방어를 동시에 달성.26 |
| **MDD 페널티** | $P\_{MDD} \= \-e^{k \\cdot \\text{MaxDrawdown}}$ | $0.8$ | 최대 낙폭이 커질수록 지수적으로(Exponentially) 커지는 페널티를 부여하여 파산 위험을 원천 차단.28 |
| **추론 일관성 (ACRE)** | $R\_{const} \= \\mathbb{I}(\\text{CoT} \\approx \\text{Action})$ | $0.3$ | CoT에서 "하락"을 예측하고 "매수"하는 식의 논리적 모순 발생 시 페널티 부여. 별도의 Rule-based Checker나 소형 LLM으로 검증.23 |
| **포맷 준수 (Format)** | $R\_{fmt} \= \\mathbb{I}(\\text{Tags Valid})$ | $0.1$ | \<think\>, \<answer\> 태그 구조 준수 여부. 초기 학습 안정화에 필수적임.17 |

최종 보상 함수:

$$R\_{total} \= w\_1 r\_t \+ w\_2 R\_{risk} \+ w\_3 P\_{MDD} \+ w\_4 R\_{const} \+ w\_5 R\_{fmt}$$  
이러한 다목적 보상 함수는 에이전트가 단순히 수익률만 높이는 '도박적 행위'를 하지 않고, 기관 투자자 수준의 리스크 관리 규율을 내재화하도록 유도합니다.

### **4.4. Phase 별 단계적 학습 전략 (Curriculum Learning)**

Trading-R1 모델의 "Easy-to-Hard" 커리큘럼15을 벤치마킹하여 학습 단계를 체계화합니다.

1. **Phase 1: Cold Start (SFT) \- 모방 학습**  
   * **목표:** 에이전트가 기본적인 트레이딩 규칙과 추론 형식을 익히도록 함.  
   * **데이터:** Claude나 GPT-4가 생성한 고품질 트레이딩 시나리오와 전문가의 매매 일지를 데이터셋으로 활용.  
   * **방법:** 지도 학습(Supervised Fine-Tuning)을 통해 입력 상태에 대해 전문가와 유사한 CoT와 행동을 생성하도록 학습. 이는 초기 탐색 비용을 줄이고 학습 속도를 가속화합니다.  
2. **Phase 2: GRPO RL \- 그룹 기반 강화학습**  
   * **목표:** 다양한 시장 상황에서의 일반화 성능 확보 및 자기 주도적 전략 수립.  
   * **방법:** SFT 모델을 초기 정책으로 설정하고, 실제 시장 데이터(Historical Data) 위에서 GRPO 알고리즘을 적용. 그룹 샘플링을 통해 다양한 전략을 시도하고, 상대적 우위를 가진 전략을 강화.  
   * **Training-Free GRPO 활용:** 본격적인 파라미터 업데이트 전, 'Training-Free GRPO' 기법을 사용하여 문맥 내 학습(In-Context Learning)으로 토큰 사전(Token Prior)을 최적화하는 과정을 거치면 학습 효율을 더욱 높일 수 있습니다.19  
3. **Phase 3: Inference Time Scaling \- 추론 시점 최적화**  
   * **목표:** 실전 매매에서의 안정성 및 신뢰도 제고.  
   * **방법:** 매매 신호를 생성할 때 단일 출력에 의존하지 않고, $N$개의 추론 결과를 생성한 뒤 **다수결 투표(Majority Voting)** 또는 \*\*엔트로피 가중 투표(Entropy-weighted Voting)\*\*를 통해 최종 행동을 결정합니다. 이는 모델의 불확실성을 줄이고 오판을 방지하는 효과적인 기법입니다.30

---

## **5\. 예상되는 파급 효과 및 결론**

### **5.1. 기술적/사업적 기대 효과**

본 제안에 따른 모델 개선은 다음과 같은 다층적인 효과를 창출할 것으로 기대됩니다.

* **시장 중립적 강건성(Robustness):** GRPO의 그룹 정규화 효과로 인해, 에이전트는 시장의 절대적인 상승/하락 추세에 의존하지 않고, 어떤 상황에서도 상대적으로 우월한 전략을 찾아내는 능력을 갖추게 됩니다. 이는 Phase 3에서 겪었던 하락장 과적합 문제를 근본적으로 해결할 것입니다.4  
* **설명 가능한 퀀트(Explainable Quant):** CoT 도입은 블랙박스 모델의 한계를 넘어, 에이전트의 매매 근거를 텍스트로 투명하게 확인할 수 있게 합니다. 이는 투자 자문 서비스의 신뢰도를 높이고, 고객에게 "왜 이 시점에 매매했는지"를 설명할 수 있는 강력한 도구가 됩니다.1  
* **비용 효율적 고성능:** Fin-R1의 사례에서 보듯, 소형 모델에 GRPO를 적용함으로써 대규모 모델 수준의 추론 능력을 낮은 비용으로 구현할 수 있습니다. 이는 서비스의 운영 비용(OPEX)을 절감하고 확장성을 확보하는 데 기여합니다.17

### **5.2. 결론 및 제언**

현재의 financial-rl-claude-skills 프로젝트는 훌륭한 비전을 가지고 있으나, 그 실행 엔진인 RL 모델의 노후화된 구조가 성과를 제한하고 있습니다. DeepSeek-R1이 입증한 **GRPO 알고리즘**과 **추론 기반 학습**의 도입은 단순한 성능 개선을 넘어, 금융 AI 에이전트의 지능을 한 단계 도약시키는 필수적인 과제입니다.

특히, **Critic의 제거**를 통해 학습의 안정성을 확보하고, **그룹 샘플링**을 통해 시장의 불확실성을 통계적으로 극복하며, **다목적 보상 함수**를 통해 리스크 관리 규율을 내재화하는 본 보고서의 제안은 DR\_BreakFast님이 추구하는 "의학, 금융, 정책이 융합된 초지능형 자문 시스템"을 실현하는 가장 확실한 로드맵이 될 것입니다. 즉시 아키텍처 리팩토링과 데이터 파이프라인 고도화에 착수하여, 스스로 사고하고 진화하는 \*\*"Autonomous Financial Agent"\*\*를 구축하시기를 강력히 권고드립니다.

작성자: 수석 퀀트 AI 연구원 (Senior Quantitative AI Researcher)  
일자: 2025년 11월 24일

#### **참고 자료**

1. DeepSeek-R1 GRPO 알고리즘을 적용한 RL-based Trading Model 구축을 위한 개선 작업 계획 보고서  
2. Understanding the DeepSeek R1 Paper \- Hugging Face LLM Course, 11월 23, 2025에 액세스, [https://huggingface.co/learn/llm-course/chapter12/3](https://huggingface.co/learn/llm-course/chapter12/3)  
3. DeepSeek R1: Understanding GRPO and Multi-Stage Training | by BavalpreetSinghh, 11월 23, 2025에 액세스, [https://ai.plainenglish.io/deepseek-r1-understanding-grpo-and-multi-stage-training-5e0bbc28a281](https://ai.plainenglish.io/deepseek-r1-understanding-grpo-and-multi-stage-training-5e0bbc28a281)  
4. Deep dive into Group Relative Policy Optimization (GRPO) \- AWS Builder Center, 11월 23, 2025에 액세스, [https://builder.aws.com/content/2rJrpj6m2eh591fjMcRZ3ushpB7/deep-dive-into-group-relative-policy-optimization-grpo](https://builder.aws.com/content/2rJrpj6m2eh591fjMcRZ3ushpB7/deep-dive-into-group-relative-policy-optimization-grpo)  
5. What is GRPO? Group Relative Policy Optimization Explained \- DataCamp, 11월 23, 2025에 액세스, [https://www.datacamp.com/blog/what-is-grpo-group-relative-policy-optimization](https://www.datacamp.com/blog/what-is-grpo-group-relative-policy-optimization)  
6. DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning \- arXiv, 11월 23, 2025에 액세스, [https://arxiv.org/pdf/2501.12948](https://arxiv.org/pdf/2501.12948)  
7. Language Model Guided Reinforcement Learning in Quantitative Trading \- arXiv, 11월 23, 2025에 액세스, [https://arxiv.org/html/2508.02366v2](https://arxiv.org/html/2508.02366v2)  
8. Theory Behind GRPO \- AI Engineering Academy, 11월 23, 2025에 액세스, [https://aiengineering.academy/LLM/TheoryBehindFinetuning/GRPO/](https://aiengineering.academy/LLM/TheoryBehindFinetuning/GRPO/)  
9. \[2506.04358\] A Risk-Aware Reinforcement Learning Reward for Financial Trading \- arXiv, 11월 23, 2025에 액세스, [https://arxiv.org/abs/2506.04358](https://arxiv.org/abs/2506.04358)  
10. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models \- arXiv, 11월 23, 2025에 액세스, [https://arxiv.org/pdf/2402.03300](https://arxiv.org/pdf/2402.03300)  
11. Group Relative Policy Optimization (GRPO) Illustrated Breakdown \- Ebrahim Pichka, 11월 23, 2025에 액세스, [https://epichka.com/blog/2025/grpo/](https://epichka.com/blog/2025/grpo/)  
12. RAG with GRPO Fine-Tuned Reasoning Model \- LanceDB, 11월 23, 2025에 액세스, [https://lancedb.com/blog/grpo-understanding-and-fine-tuning-the-next-gen-reasoning-model-2/](https://lancedb.com/blog/grpo-understanding-and-fine-tuning-the-next-gen-reasoning-model-2/)  
13. Understanding the Math Behind GRPO — DeepSeek-R1-Zero | by Yugen.ai \- Medium, 11월 23, 2025에 액세스, [https://medium.com/yugen-ai-technology-blog/understanding-the-math-behind-grpo-deepseek-r1-zero-9fb15e103a0a](https://medium.com/yugen-ai-technology-blog/understanding-the-math-behind-grpo-deepseek-r1-zero-9fb15e103a0a)  
14. Group Relative Policy Optimization (GRPO) \- verl documentation \- Read the Docs, 11월 23, 2025에 액세스, [https://verl.readthedocs.io/en/latest/algo/grpo.html](https://verl.readthedocs.io/en/latest/algo/grpo.html)  
15. \[2509.11420\] Trading-R1: Financial Trading with LLM Reasoning via Reinforcement Learning \- arXiv, 11월 23, 2025에 액세스, [https://arxiv.org/abs/2509.11420](https://arxiv.org/abs/2509.11420)  
16. Trading-R1: Financial Trading with LLM Reasoning via Reinforcement Learning \- ChatPaper, 11월 23, 2025에 액세스, [https://chatpaper.com/paper/188279](https://chatpaper.com/paper/188279)  
17. Fin-R1: A Large Language Model for Financial Reasoning through Reinforcement Learning, 11월 23, 2025에 액세스, [https://arxiv.org/html/2503.16252v1](https://arxiv.org/html/2503.16252v1)  
18. Multimodal Financial Foundation Models (MFFMs): Progress, Prospects, and Challenges, 11월 23, 2025에 액세스, [https://arxiv.org/html/2506.01973v2](https://arxiv.org/html/2506.01973v2)  
19. \[2510.08191\] Training-Free Group Relative Policy Optimization \- arXiv, 11월 23, 2025에 액세스, [https://arxiv.org/abs/2510.08191](https://arxiv.org/abs/2510.08191)  
20. Training-Free Group Relative Policy Optimization \- ChatPaper, 11월 23, 2025에 액세스, [https://chatpaper.com/paper/197941](https://chatpaper.com/paper/197941)  
21. Training-Free Group Relative Policy Optimization \- arXiv, 11월 23, 2025에 액세스, [https://arxiv.org/html/2510.08191v1](https://arxiv.org/html/2510.08191v1)  
22. Fin-PRM: A Domain-Specialized Process Reward Model for Financial Reasoning in Large Language Models \- arXiv, 11월 23, 2025에 액세스, [https://arxiv.org/html/2508.15202v1](https://arxiv.org/html/2508.15202v1)  
23. Answer-Consistent Chain-of-Thought Reinforcement Learning for Multi-modal Large Language Models \- arXiv, 11월 23, 2025에 액세스, [https://arxiv.org/html/2510.10104v1](https://arxiv.org/html/2510.10104v1)  
24. Daily Papers \- Hugging Face, 11월 23, 2025에 액세스, [https://huggingface.co/papers?q=strategic%20planning](https://huggingface.co/papers?q=strategic+planning)  
25. Action-specialized expert ensemble trading system with extended discrete action space using deep reinforcement learning \- NIH, 11월 23, 2025에 액세스, [https://pmc.ncbi.nlm.nih.gov/articles/PMC7384672/](https://pmc.ncbi.nlm.nih.gov/articles/PMC7384672/)  
26. (PDF) Risk-Adjusted Deep Reinforcement Learning for Portfolio Optimization: A Multi-reward Approach \- ResearchGate, 11월 23, 2025에 액세스, [https://www.researchgate.net/publication/392098277\_Risk-Adjusted\_Deep\_Reinforcement\_Learning\_for\_Portfolio\_Optimization\_A\_Multi-reward\_Approach](https://www.researchgate.net/publication/392098277_Risk-Adjusted_Deep_Reinforcement_Learning_for_Portfolio_Optimization_A_Multi-reward_Approach)  
27. Risk-Aware Reinforcement Learning Reward for Financial Trading \- arXiv, 11월 23, 2025에 액세스, [https://arxiv.org/html/2506.04358v1](https://arxiv.org/html/2506.04358v1)  
28. Risk-Sensitive Deep Reinforcement Learning for Portfolio Optimization \- MDPI, 11월 23, 2025에 액세스, [https://www.mdpi.com/1911-8074/18/7/347](https://www.mdpi.com/1911-8074/18/7/347)  
29. FareedKhan-dev/train-deepseek-r1 \- GitHub, 11월 23, 2025에 액세스, [https://github.com/FareedKhan-dev/train-deepseek-r1](https://github.com/FareedKhan-dev/train-deepseek-r1)  
30. LLM Reasoning Series: How DeepSeek-R1 Uses Reinforcement Learning to Supercharge Reasoning | by Isaac Kargar, 11월 23, 2025에 액세스, [https://kargarisaac.medium.com/how-deepseek-r1-uses-reinforcement-learningto-supercharge-reasoning-3f826c2c8759](https://kargarisaac.medium.com/how-deepseek-r1-uses-reinforcement-learningto-supercharge-reasoning-3f826c2c8759)  
31. FinRL Contests: Benchmarking Data-driven Financial Reinforcement Learning Agents, 11월 23, 2025에 액세스, [https://arxiv.org/html/2504.02281v3](https://arxiv.org/html/2504.02281v3)  
32. The Sequential Edge: Inverse-Entropy Voting Beats Parallel Self-Consistency at Matched Compute \- arXiv, 11월 23, 2025에 액세스, [https://arxiv.org/html/2511.02309v1](https://arxiv.org/html/2511.02309v1)  
33. \[2505.06408\] A New DAPO Algorithm for Stock Trading \- arXiv, 11월 23, 2025에 액세스, [https://arxiv.org/abs/2505.06408](https://arxiv.org/abs/2505.06408)  
34. Fin-R1: A Large Language Model for Financial Reasoning through Reinforcement Learning, 11월 23, 2025에 액세스, [https://arxiv.org/html/2503.16252v2](https://arxiv.org/html/2503.16252v2)