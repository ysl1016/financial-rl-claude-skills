

# **DeepSeek-R1 GRPO 알고리즘 기반 금융 트레이딩 에이전트 고도화: Phase 2 보상 함수(Reward Function) 재설계 및 실행 전략 보고서**

## **1\. 서론: 금융 AI의 인지적 전환과 정책 붕괴(Policy Collapse)의 극복**

금융 시장은 본질적으로 높은 불확실성(Stochasticity)과 비정상성(Non-stationarity), 그리고 적대적(Adversarial) 특성을 지닌 복잡계(Complex System)입니다. 이러한 환경에서 강화학습(Reinforcement Learning, RL) 에이전트를 훈련시키는 것은 단순한 최적화 문제를 넘어선 난제를 제시합니다. 귀하의 financial-rl-claude-skills 프로젝트에서 관찰된 Phase 3의 백테스팅 결과—**\-12.37%의 수익률과 0회 매수/214회 매도라는 극단적 행동 비대칭**—은 전형적인 **정책 붕괴(Policy Collapse)** 현상을 시사합니다.1 이는 에이전트가 시장의 노이즈를 학습하거나, 손실 회피(Loss Aversion) 성향이 과도하게 최적화되어 시장 진입 자체를 '위험'으로 간주하고 포지션을 청산하여 '동결(Freezing)' 상태에 빠진 병리학적 결과로 해석됩니다.

본 보고서는 2025년 10월과 11월에 발표된 최신 arXiv 논문들, 특히 **DeepSeek-R1**의 **GRPO(Group Relative Policy Optimization)** 알고리즘과 **Trading-R1**, **Fin-PRM** 등의 연구 성과를 바탕으로 Phase 2 모델의 근본적인 개선안을 제시합니다. 단순한 파라미터 튜닝이 아닌, 에이전트의 '동기 부여 체계'인 보상 함수(Reward Function)를 다차원적으로 재설계하여, 수익성(Profitability), 안정성(Stability), 그리고 논리적 정합성(Reasoning Consistency)을 동시에 달성하는 **Tri-Axial Reward Architecture**를 제안합니다.

우리는 기존의 PPO(Proximal Policy Optimization) 기반 접근법이 가지는 **비평가(Critic) 모델의 한계**를 극복하고, DeepSeek-R1이 입증한 **추론 기반 강화학습(Reasoning-based RL)** 패러다임을 금융 도메인에 이식함으로써, 단순한 매매 기계를 넘어 '시장 상황을 해석하고 논리적으로 대응하는 자율 금융 에이전트'를 구축하는 것을 목표로 합니다.

## **2\. 현행 모델(Phase 1)의 구조적 결함 진단 및 병리학적 분석**

성공적인 개선을 위해서는 현재 시스템이 왜 실패했는지에 대한 냉철한 분석이 선행되어야 합니다. Phase 1 모델의 실패는 단순한 데이터 부족이나 학습 시간의 문제가 아닌, 알고리즘과 금융 데이터 특성 간의 불일치(Mismatch)에서 기인합니다.

### **2.1 비평가(Critic) 네트워크의 가치 추정 실패와 분산 문제**

전통적인 Actor-Critic 구조(PPO 등)에서 비평가(Critic)는 상태 가치 함수 $V(s)$를 학습하여 행위자(Actor)에게 이점(Advantage) 신호를 제공합니다. 그러나 금융 시계열 데이터는 신호 대 잡음비(SNR)가 극도로 낮습니다. 2025년 11월 발표된 연구에 따르면, 금융 데이터의 높은 변동성은 Critic의 가치 추정에 막대한 분산(Variance)을 유발합니다.3  
Critic이 미래 보상을 정확히 예측하지 못하고 잦은 오류를 범할 때, Actor는 잘못된 가이드라인(Noisy Gradient)에 따라 학습하게 됩니다. 특히 하락장이나 변동성 구간에서 Critic이 과도하게 비관적인 가치를 예측하면, Actor는 "아무것도 하지 않거나(Hold)", "모든 것을 팔아치우는(Panic Sell)" 것이 기대 보상(정확히는 손실 최소화)을 극대화하는 유일한 전략이라고 오판하게 됩니다. Phase 3에서 관찰된 214회의 매도 쏠림 현상은 Critic이 "시장 참여 \= 고통"이라는 잘못된 인과관계를 학습시켰기 때문일 가능성이 매우 높습니다.2

### **2.2 보상 함수의 비대칭성과 희소성(Sparsity)**

현재의 단순 수익률 기반 보상 체계는 금융 시장의 복잡성을 반영하지 못합니다.

* **결과 편향(Outcome Bias):** 에이전트가 논리적으로 타당한 진입을 했음에도 시장의 무작위적 등락으로 손실을 볼 경우 페널티를 받고, 반대로 뇌동매매로 운 좋게 수익을 낼 경우 보상을 받습니다. 이는 \*\*강화학습의 오학습(Misalignment)\*\*을 초래합니다.4  
* **위험 인식의 부재:** 단순 P\&L(Profit & Loss) 보상은 위험 조정 수익률을 고려하지 않습니다. 50%의 파산 위험을 감수하고 10%의 수익을 낸 행동이 긍정적으로 강화된다면, 장기적으로 에이전트는 필연적으로 파산(Ruin)하게 됩니다.5  
* **희소한 피드백:** 매매 결정 시점과 수익 실현 시점 간의 시차(Time Lag)로 인해, 에이전트는 자신의 어떤 행동이 결과에 기여했는지 파악하기 어려운 '신용 할당 문제(Credit Assignment Problem)'에 직면합니다.6

### **2.3 블랙박스 의사결정의 한계**

Phase 1 모델은 입력 상태(State)에서 행동(Action)으로 직결되는 구조를 가집니다. 이는 에이전트가 "왜" 매매했는지에 대한 중간 사고 과정(Chain of Thought, CoT)을 검증할 수 없게 만듭니다. DeepSeek-R1의 연구 결과는 중간 추론 과정에 대한 보상이 최종 성능 향상에 결정적임을 보여줍니다.7 추론 과정이 없는 에이전트는 시장의 미묘한 뉘앙스를 해석하지 못하고 단순한 패턴 매칭에 의존하게 되어, 시장 국면(Regime)이 바뀔 때 급격히 성능이 저하됩니다.

## **3\. DeepSeek-R1 GRPO 및 NGRPO 알고리즘의 도입 전략**

Phase 2의 핵심은 Critic 모델을 제거하고 그룹 단위의 상대적 최적화를 수행하는 \*\*GRPO(Group Relative Policy Optimization)\*\*의 도입입니다. 여기에 더해, 최근 제안된 \*\*NGRPO(Negative-enhanced GRPO)\*\*를 통해 학습 안정성을 극대화합니다.

### **3.1 Critic-less 학습의 이점과 GRPO 메커니즘**

GRPO는 별도의 가치 함수(Critic)를 학습시키지 않습니다. 대신, 동일한 상태(질문 또는 시장 상황) $q$에 대해 $G$개의 서로 다른 행동 경로 ${o\_1, o\_2,..., o\_G}$를 샘플링하고, 이들 간의 상대적 우위를 평가합니다.8

$$J\_{GRPO}(\\theta) \= \\mathbb{E}\_{q \\sim P(Q), \\{o\_i\\}\_{i=1}^G \\sim \\pi\_{\\theta\_{old}}(O|q)} \\left$$  
여기서 이점(Advantage) $A\_i$는 그룹 내 보상의 정규화된 값으로 계산됩니다:

$$A\_i \= \\frac{r\_i \- \\text{mean}(\\{r\_1,..., r\_G\\})}{\\text{std}(\\{r\_1,..., r\_G\\}) \+ \\epsilon}$$  
이 방식은 금융 트레이딩에 있어 혁명적입니다. 예를 들어, 시장이 폭락하여 모든 전략이 손실을 기록하는 상황을 가정해 봅시다.

* 전략 A: \-5% 수익률 (적극 매수 후 물림)  
* 전략 B: \-1% 수익률 (적극 헷징 또는 현금 비중 확대)  
  기존 RL은 두 전략 모두에게 음(-)의 보상을 주어 "매매 행위 자체"를 처벌할 위험이 있습니다. 반면 GRPO는 전략 B가 전략 A보다 상대적으로 우월함을 인식하고($A\_B \> 0$), 하락장에서도 최선의 방어 전략을 학습할 수 있게 합니다.9 이는 Phase 1의 정책 붕괴를 막는 핵심 기제입니다.

### **3.2 균질한 실패 극복을 위한 NGRPO (Negative-enhanced GRPO)**

그러나 모든 샘플이 유사하게 나쁜 결과를 낼 경우, GRPO의 이점이 0에 수렴하여 학습이 정체될 수 있습니다. 이를 방지하기 위해 2025년 9월 발표된 NGRPO 알고리즘을 도입합니다.11  
NGRPO는 가상의 '최적 샘플(Virtual Best Sample)'을 그룹에 포함시켜, 실제 수행한 모든 행동이 최적해보다 못했다면 전체 그룹의 이점을 낮추는 방식으로 작동합니다. 이는 에이전트가 "모두가 망한 상황"에 안주하지 않고, 더 나은 글로벌 최적해를 탐색하도록 강제합니다.

### **3.3 KL 발산(Divergence)을 통한 정책 안정화**

GRPO 손실 함수에 포함된 $D\_{KL}(\\pi\_\\theta |

| \\pi\_{ref})$ 항은 학습된 정책이 초기 참조 모델(Reference Model)이나 SFT(Supervised Fine-Tuning) 모델에서 너무 멀어지지 않도록 규제합니다.8 금융에서는 이것이 "기본적인 시장 원칙"이나 "기술적 분석의 공리"를 망각하지 않도록 하는 안전장치 역할을 합니다. 에이전트가 과도한 최적화(Overfitting)로 기이한 매매 패턴을 보이는 것을 방지합니다.

## **4\. Phase 2 핵심: 3축 보상 아키텍처 (Tri-Axial Reward Architecture)**

보고서 1에서 제안된 바와 같이, 단순 수익률 보상을 넘어선 입체적인 보상 체계가 필요합니다. 우리는 최신 연구 10를 종합하여 **(1) 재무적 성과(Financial)**, **(2) 위험 관리(Risk)**, \*\*(3) 과정 및 추론(Process/Reasoning)\*\*의 3축으로 구성된 복합 보상 함수 CompositeReward를 설계합니다.

| 보상 축 (Axis) | 구성 요소 | 목표 및 기능 | 가중치 (wi​) |
| :---- | :---- | :---- | :---- |
| **1\. Financial** | 차분 로그 수익률 (Log Returns) | 대칭적 수익 구조 확보 및 자산 증식 | 1.0 |
|  | 벤치마크 초과 수익 (Alpha) | 단순 시장 상승 편승 방지, 시장 대비 우위 확보 | 0.5 |
|  | 시장 충격 페널티 (Liquidity Cost) | 현실적인 거래 비용 반영 및 유동성 위험 회피 | 0.3 |
| **2\. Risk** | 차분 소르티노 비율 (Sortino Ratio) | 하방 변동성(Downside Risk)만 선별적 제어 | 0.8 |
|  | 기하급수적 MDD 페널티 | 파산(Ruin) 방지 및 자산 보존 최우선화 | 2.0 (동적) |
| **3\. Process** | 형식 준수 보상 (Formatting) | CoT 태그(\<think\>) 준수 및 구조적 사고 유도 | 0.2 |
|  | 논리적 일관성 보상 (ACRE) | 추론 내용과 행동 간의 인과관계 검증 (Hallucination 방지) | 0.5 |

### **4.1 Axis 1: 재무적 성과 \- 마찰 비용과 상대적 우위**

단순 수익률 대신 로그 수익률을 사용합니다. $r\_t \= \\ln(P\_t / P\_{t-1})$은 덧셈이 가능하고 대칭적이어서, 50% 손실 후 100% 수익이 필요한 비대칭성을 수학적으로 완화합니다.13  
또한, 시장 충격(Market Impact) 페널티를 도입합니다. Almgren-Chriss 모델에 기반하여, 거래량의 1.5승에 비례하는 페널티를 부과함으로써 에이전트가 유동성이 낮은 종목에서 과도한 물량을 거래하는 것을 억제합니다.14

$$R\_{\\text{fin}} \= (r\_t \- r\_{\\text{benchmark}}) \- \\eta \\cdot (\\text{Volume})^{1.5}$$

### **4.2 Axis 2: 위험 관리 \- 생존을 위한 비대칭 처벌**

Phase 3의 실패를 막기 위해 위험에 대한 페널티를 강화합니다.

* **Sortino Ratio 기반 보상:** Sharpe Ratio는 상방 변동성(급등)도 리스크로 간주하는 단점이 있습니다. Sortino Ratio는 하방 변동성(Downside Deviation)만을 분모로 사용하므로, 에이전트가 "안전한 수익"을 추구하도록 유도합니다.16  
* **기하급수적 MDD(Maximum Drawdown) 페널티:** 최대 낙폭이 커질수록 페널티가 선형적이 아닌 기하급수적으로 증가하도록 설계합니다.17  
  $$P\_{\\text{MDD}} \= \- \\beta \\cdot (e^{k \\cdot \\text{CurrentDrawdown}} \- 1)$$

  낙폭이 작을 때는 페널티가 미미하지만, 임계치(예: \-10%)를 넘어서면 페널티가 급증하여 에이전트가 즉각적인 리스크 관리에 돌입하게 만듭니다. 이는 기관 투자자의 "손절매(Stop-loss)" 규율을 내재화하는 효과가 있습니다.

### **4.3 Axis 3: 과정 및 추론 \- Fin-PRM과 설명 가능성**

DeepSeek-R1과 Fin-PRM 연구10에 따르면, 결과뿐만 아니라 과정(Process)을 보상해야 일반화 성능이 높아집니다.

* **논리적 정합성 검증 (Consistency Check):** Claude API나 경량화된 검증 모델을 사용하여, 에이전트가 생성한 생각(CoT)과 행동이 일치하는지 평가합니다. 예를 들어, \<think\> 태그 내에서 "RSI가 80 이상이라 과매수 상태"라고 분석해놓고 Buy 주문을 낸다면, 수익 여부와 관계없이 강력한 페널티를 부여합니다.18 이는 에이전트가 '운 좋게 맞춘' 경우를 학습 배제하고, 논리에 기반한 승리만을 강화하게 합니다.  
* **환각(Hallucination) 페널티:** 에이전트가 관측된 상태(Observation)에 없는 뉴스나 수치를 근거로 댈 경우 페널티를 부여하여 사실(Fact)에 입각한 추론을 강제합니다.20

## **5\. 단계별 구현 및 학습 계획 (Curriculum Learning)**

Trading-R121과 "Train Long, Think Short"22 연구에서 제안된 커리큘럼 학습 전략을 차용하여, 에이전트를 단계적으로 성장시킵니다.

### **Phase 2.1: 구조적 기초 확립 (1주차) \- "말하는 법 배우기"**

* **목표:** 에이전트가 지정된 XML 형식(\<think\>, \<answer\>)을 준수하고, 기본적인 시장 데이터를 인지하도록 훈련.  
* **보상 구성:** 형식 준수 보상(80%) \+ 단순 시장 방향성 예측(20%).  
* **방법:** 소량의 전문가 데이터(SFT)를 이용해 GRPO의 'Cold Start'를 수행합니다. 이 단계에서는 복잡한 리스크 관리를 배제하고 올바른 형식으로 사고를 전개하는 데 집중합니다.

### **Phase 2.2: 논리적 정렬 (2주차) \- "생각과 행동의 일치"**

* **목표:** 기술적 지표와 매크로 변수에 대한 올바른 해석 능력 배양.  
* **보상 구성:** 논리적 정합성 보상(50%) \+ 재무적 성과(50%).  
* **방법:** Fin-PRM 스타일의 과정 보상을 활성화합니다. 에이전트가 "골든크로스 발생 \-\> 매수 유리"와 같은 인과관계를 학습하게 합니다. 이때부터 NGRPO를 적용하여 논리적으로 틀린 그룹을 적극적으로 교정합니다.

### **Phase 2.3: 위험 인식 실전 배치 (3주차) \- "생존 기술 습득"**

* **목표:** 변동성 장세에서의 자산 방어 및 MDD 최소화.  
* **보상 구성:** 위험 조정 보상(Sortino/MDD, 50%) \+ 재무적 성과(30%) \+ 논리 보상(20%).  
* **방법:** 기하급수적 MDD 페널티를 활성화하고, 다양한 시장 시나리오(폭락장, 횡보장)를 시뮬레이션하여 에이전트의 위기 대응 능력을 극한으로 시험합니다.

## **6\. 기술적 구현 로드맵 및 파일 구조**

본 계획은 기존 코드베이스의 리팩토링을 포함합니다.

### **6.1 src/utils/reward\_functions.py 리팩토링**

단일 함수 대신 CompositeReward 클래스를 구현하여 유연성을 확보합니다.

Python

class CompositeReward:  
    def calculate(self, trade\_log, reasoning\_trace, market\_data):  
        \# 1\. Financial Component  
        fin\_reward \= self.\_log\_returns(trade\_log) \- self.\_impact\_penalty(trade\_log)  
          
        \# 2\. Risk Component (Asymmetric)  
        risk\_penalty \= self.\_sortino\_penalty(trade\_log) \+ self.\_geometric\_mdd(trade\_log)  
          
        \# 3\. Process Component (Fin-PRM)  
        process\_score \= self.\_verify\_consistency(reasoning\_trace, market\_data)  
          
        return w1 \* fin\_reward \+ w2 \* risk\_penalty \+ w3 \* process\_score

여기서 \_verify\_consistency 메서드는 정규식(Regex) 또는 소형 LLM을 통해 CoT의 논리적 모순을 감지합니다.

### **6.2 src/models/grpo\_agent.py 신규 구현**

기존 DeepSeekTradingModel을 수정하여 PPO의 Value Head를 제거하고, 그룹 샘플링 기능을 추가합니다.

* **그룹 샘플링:** 단일 입력에 대해 $G=16$개의 출력을 병렬 생성하는 sample\_group 메서드 구현.  
* **Advantage 계산:** 그룹 내 보상 정규화 및 KL Divergence 페널티($\\beta$) 적용 로직 구현.

### **6.3 적응형 보상 클리핑 (Adaptive Reward Clipping)**

학습 초기 보상 값의 스파이크로 인한 불안정을 막기 위해, 보상 값의 범위를 동적으로 제한하는 적응형 클리핑을 적용합니다.23 이는 ±3% 또는 표준편차의 3배수로 제한하여 그래디언트 폭주를 방지합니다.

## **7\. 결론 및 기대 효과**

Phase 3의 실패는 역설적으로 우리에게 올바른 방향을 제시해주었습니다. 단순한 이익 추구는 시장의 불확실성 앞에서 에이전트를 무력화시킵니다. 본 보고서에서 제안하는 **GRPO 기반의 Tri-Axial 보상 체계**는 에이전트에게 다음과 같은 능력을 부여할 것입니다.

1. **강건성(Robustness):** GRPO의 상대적 평가를 통해 하락장에서도 '덜 나쁜' 전략을 찾아내며 동결 상태를 방지합니다.  
2. **생존력(Survivability):** 기하급수적 MDD 페널티와 Sortino 비율을 통해, 수익보다 생존을 우선시하는 기관 투자자의 규율을 체득합니다.  
3. **설명 가능성(Explainability):** 과정 보상(Process Reward)을 통해, 에이전트의 모든 매매 행위는 논리적인 텍스트 근거(CoT)를 동반하게 되며, 이는 사용자의 신뢰를 확보하는 핵심 자산이 될 것입니다.

이제 우리는 단순한 'Trading Bot'을 넘어, 스스로 사고하고 위험을 관리하는 진정한 의미의 \*\*'AI Financial Analyst'\*\*를 구축하는 단계로 진입합니다. 제시된 로드맵에 따라 즉각적인 코드 리팩토링과 커리큘럼 학습 착수를 권고합니다.

#### **참고 자료**

1. DeepSeek-R1 GRPO 알고리즘을 적용한 RL-based Trading Model 구축을 위한 개선 작업 계획 보고서  
2. DeepSeek-R1 GRPO 알고리즘을 활용한 금융 트레이딩 에이전트(Phase 1\) 아키텍처 재설계 및 고도화 전략 심층 연구 보고서  
3. (PDF) Learning Without Critics? Revisiting GRPO in Classical Reinforcement Learning Environments \- ResearchGate, 11월 23, 2025에 액세스, [https://www.researchgate.net/publication/397321775\_Learning\_Without\_Critics\_Revisiting\_GRPO\_in\_Classical\_Reinforcement\_Learning\_Environments](https://www.researchgate.net/publication/397321775_Learning_Without_Critics_Revisiting_GRPO_in_Classical_Reinforcement_Learning_Environments)  
4. Fin-PRM: A Domain-Specialized Process Reward Model for Financial Reasoning in Large Language Models \- ChatPaper, 11월 23, 2025에 액세스, [https://chatpaper.com/paper/182447](https://chatpaper.com/paper/182447)  
5. A Risk-Aware Reinforcement Learning Reward for Financial Trading \- arXiv, 11월 23, 2025에 액세스, [https://arxiv.org/pdf/2506.04358](https://arxiv.org/pdf/2506.04358)  
6. \[2412.10917\] Adaptive Reward Design for Reinforcement Learning \- arXiv, 11월 23, 2025에 액세스, [https://arxiv.org/abs/2412.10917](https://arxiv.org/abs/2412.10917)  
7. Understanding DeepSeek R1—A Reinforcement Learning-Driven Reasoning Model, 11월 23, 2025에 액세스, [https://kili-technology.com/blog/understanding-deepseek-r1](https://kili-technology.com/blog/understanding-deepseek-r1)  
8. Deep dive into Group Relative Policy Optimization (GRPO) \- AWS Builder Center, 11월 23, 2025에 액세스, [https://builder.aws.com/content/2rJrpj6m2eh591fjMcRZ3ushpB7/deep-dive-into-group-relative-policy-optimization-grpo](https://builder.aws.com/content/2rJrpj6m2eh591fjMcRZ3ushpB7/deep-dive-into-group-relative-policy-optimization-grpo)  
9. Group Relative Policy Optimization (GRPO) Illustrated Breakdown \- Ebrahim Pichka, 11월 23, 2025에 액세스, [https://epichka.com/blog/2025/grpo/](https://epichka.com/blog/2025/grpo/)  
10. Fin-PRM: A Domain-Specialized Process Reward Model for Financial Reasoning in Large Language Models \- arXiv, 11월 23, 2025에 액세스, [https://arxiv.org/html/2508.15202v1](https://arxiv.org/html/2508.15202v1)  
11. NGRPO: Negative-enhanced Group Relative Policy Optimization \- arXiv, 11월 23, 2025에 액세스, [https://arxiv.org/html/2509.18851v1](https://arxiv.org/html/2509.18851v1)  
12. Risk-Aware Deep Reinforcement Learning for Dynamic Portfolio Optimization \- arXiv, 11월 23, 2025에 액세스, [https://arxiv.org/html/2511.11481v1](https://arxiv.org/html/2511.11481v1)  
13. A Systematic Approach to Portfolio Optimization: A Comparative Study of Reinforcement Learning Agents, Market Signals, and Investment Horizons \- MDPI, 11월 23, 2025에 액세스, [https://www.mdpi.com/1999-4893/17/12/570](https://www.mdpi.com/1999-4893/17/12/570)  
14. (PDF) Reinforcement Learning for Trade Execution with Market Impact \- ResearchGate, 11월 23, 2025에 액세스, [https://www.researchgate.net/publication/393538861\_Reinforcement\_Learning\_for\_Trade\_Execution\_with\_Market\_Impact](https://www.researchgate.net/publication/393538861_Reinforcement_Learning_for_Trade_Execution_with_Market_Impact)  
15. Right Place, Right Time: Market Simulation-based RL for Execution Optimisation \- arXiv, 11월 23, 2025에 액세스, [https://www.arxiv.org/pdf/2510.22206](https://www.arxiv.org/pdf/2510.22206)  
16. Reinforcement Learning-Based Market Making as a Stochastic Control on Non-Stationary Limit Order Book Dynamics \- arXiv, 11월 23, 2025에 액세스, [https://arxiv.org/html/2509.12456v1](https://arxiv.org/html/2509.12456v1)  
17. Application of Deep Reinforcement Learning to At-the-Money S\&P 500 Options Hedging \- arXiv, 11월 23, 2025에 액세스, [https://arxiv.org/html/2510.09247](https://arxiv.org/html/2510.09247)  
18. Improving LLM Reasoning with Multi-Agent Tree-of-Thought Validator Agent \- arXiv, 11월 23, 2025에 액세스, [https://arxiv.org/html/2409.11527v2](https://arxiv.org/html/2409.11527v2)  
19. A Risk-Aware Reinforcement Learning Reward for Financial Trading \- Semantic Scholar, 11월 23, 2025에 액세스, [https://www.semanticscholar.org/paper/A-Risk-Aware-Reinforcement-Learning-Reward-for-Srivastava-Aryan/95ddbe69d5ee240af5a532fd26be8ff73ace8b75](https://www.semanticscholar.org/paper/A-Risk-Aware-Reinforcement-Learning-Reward-for-Srivastava-Aryan/95ddbe69d5ee240af5a532fd26be8ff73ace8b75)  
20. FG-PRM: Fine-grained Hallucination Detection and Mitigation in Language Model Mathematical Reasoning \- ACL Anthology, 11월 23, 2025에 액세스, [https://aclanthology.org/2025.findings-emnlp.228.pdf](https://aclanthology.org/2025.findings-emnlp.228.pdf)  
21. Trading-R1: LLM Reasoning in Financial Trading \- Emergent Mind, 11월 23, 2025에 액세스, [https://www.emergentmind.com/papers/2509.11420](https://www.emergentmind.com/papers/2509.11420)  
22. Train Long, Think Short: Curriculum Learning for Efficient Reasoning \- arXiv, 11월 23, 2025에 액세스, [https://arxiv.org/html/2508.08940v1](https://arxiv.org/html/2508.08940v1)  
23. Adaptive and Regime-Aware RL for Portfolio Optimization \- arXiv, 11월 23, 2025에 액세스, [https://arxiv.org/html/2509.14385v1](https://arxiv.org/html/2509.14385v1)