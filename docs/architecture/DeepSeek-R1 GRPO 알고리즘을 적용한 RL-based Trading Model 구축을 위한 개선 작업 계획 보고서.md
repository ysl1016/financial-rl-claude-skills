# **DeepSeek-R1 GRPO 알고리즘을 적용한 RL-based Trading Model 구축을 위한 개선 작업 계획 보고서**

**수신:** DR\_BreakFast (ETIV AI Institute 설립자 / 가정의학과 전문의 / 전문 투자자)

**날짜:** 2025년 11월 23일

**주제:** DeepSeek-R1의 GRPO 알고리즘 도입을 통한 금융 트레이딩 에이전트 고도화 및 수익률 안정화 전략

## **1\. 개요 (Executive Summary)**

본 보고서는 현재 구축 중인 financial-rl-claude-skills 프로젝트의 코드베이스와 financial-rl-claude-skills.ipynb의 백테스팅 결과를 심층 분석하고, 이를 바탕으로 최신 강화학습 기법인 \*\*GRPO(Group Relative Policy Optimization)\*\*를 도입하여 모델의 성능을 비약적으로 향상시키기 위한 구체적인 로드맵을 제시한다.

현재 모델은 하락장 및 변동성 장세에서 과도한 매도 포지션 진입 또는 소극적 대응으로 인해 Phase 3 기준 \-12.37%의 손실을 기록하고 있다. 이는 '단일 경로(Single Trajectory)'에 의존하는 기존 RL 방식의 한계와 보상 함수(Reward Function)의 최적화 부족에 기인한다.

이에 대한 솔루션으로 DeepSeek-R1에서 검증된 **Reasoning(추론) 기반의 GRPO**를 금융 도메인에 적용, 에이전트가 단순한 매매 행위뿐만 아니라 \*\*"왜 지금 매매해야 하는가?"에 대한 논리적 사고 과정(Chain of Thought)\*\*을 내재화하도록 하여, 노이즈가 심한 금융 시장에서의 일반화 성능을 확보하는 것을 목표로 한다.

## **2\. 현황 분석 (Status Diagnosis)**

### **2.1 코드베이스 (financial-rl-claude-skills) 구조 분석**

의학적 관점에서 시스템의 '해부학적 구조'를 분석한 결과, 상당히 체계적인 모듈 구성을 갖추고 있으나 핵심 엔진(RL Core) 부분의 개선이 필요하다.

* **강점:**  
  * **하이브리드 아키텍처:** src/claude\_integration을 통해 LLM(Claude)의 정성적 분석을 RL의 정량적 데이터와 결합하려는 시도가 우수함.  
  * **환경 설계:** src/models/enhanced\_trading\_env.py에서 기술적 지표뿐만 아니라 거시경제(Macro) 심리를 포함하려는 시도는 펀더멘털 분석을 반영하는 좋은 접근임.  
  * **모니터링:** Grafana/Prometheus 연동은 실거래(Live Trading)를 염두에 둔 전문적인 설계임.  
* **약점:**  
  * **RL 모델의 단순성:** DeepSeekTradingModel 등의 구현체가 존재하나, GRPO의 핵심인 'Group Sampling'과 'Relative Advantage' 계산 로직이 금융 시계열에 맞게 최적화되지 않음.  
  * **피드백 루프의 부재:** Claude가 분석한 텍스트 데이터가 RL 에이전트의 State로만 주입될 뿐, 에이전트의 행동에 대한 직접적인 피드백(Reward Shaping)으로 연결되는 고리가 약함.

### **2.2 성능 분석 (ipynb 실행 결과)**

노트북 실행 로그의 Phase별 결과는 환자의 '활력 징후(Vital Sign)'와 같다.

* **Phase 1 (-1.12%, 1회 거래):** 탐색 부족. 에이전트가 시장 진입을 두려워하여(Risk Averse) 거의 활동하지 않음.  
* **Phase 2 (-2.98%, 17회 거래):** 잦은 손절매. 방향성을 잡지 못하고 노이즈에 반응하여 휩소(Whipsaw)에 당하는 패턴.  
* **Phase 3 (-12.37%, 0회 매수 / 214회 매도?):** 심각한 과적합 또는 로직 오류.  
  * *진단:* 로그상의 "Sell 횟수 214"가 비정상적임. 이는 에이전트가 공매도(Short) 포지션을 잡으려 했거나, 보유 물량이 없는데 매도 신호를 지속적으로 보낸 것으로 추정됨. 또는 청산(Liquidation) 로직이 반복 호출되었을 가능성이 큼. 이는 **정책 붕괴(Policy Collapse)** 현상으로, 보상 함수가 '손실 회피'에만 너무 큰 가중치를 두어 아예 포지션을 청산하고 아무것도 안 하는 상태로 수렴했을 가능성이 높음.

## **3\. DeepSeek-R1 GRPO 알고리즘 적용 전략**

DeepSeek-R1의 핵심은 \*\*GRPO(Group Relative Policy Optimization)\*\*이다. 이를 금융 트레이딩에 적용하기 위한 이론적 배경과 적용법은 다음과 같다.

### **3.1 왜 PPO가 아닌 GRPO인가?**

기존 PPO는 가치 함수(Critic)를 학습시키기 위해 별도의 메모리와 연산이 필요하며, Critic 자체가 금융 데이터의 노이즈로 인해 부정확할 경우 에이전트 전체 성능이 저하된다.

* **GRPO의 원리:** Critic 모델을 제거한다. 대신, 동일한 상태(State) $S$에서 $G$개의 서로 다른 행동(Action) 또는 추론(Reasoning) 경로를 샘플링한다.  
* **금융 적용:** 특정 시점(State)에서 8\~16개의 병렬 시뮬레이션을 수행(Group Sampling)하고, 그중 수익률이나 샤프 지수가 평균보다 높은 행동의 확률을 높이고, 낮은 행동의 확률을 낮춘다.  
* **이점:** Critic 학습의 불안정성을 제거하고, \*\*상대적 우위(Relative Advantage)\*\*를 통해 시장의 절대적 등락과 무관하게 '현재 상황에서 최선'인 행동을 학습할 수 있다.

### **3.2 금융 특화 GRPO 아키텍처 제안**

DR\_BreakFast님의 목표인 '퀀트 자문사' 수준의 모델을 위해 다음과 같은 아키텍처를 제안한다.

1. **State 입력:** 기술적 지표 \+ 매크로 센티먼트 \+ **LLM의 시장 요약(Embedding)**  
2. **Policy Network (DeepSeek-style):**  
   * 단순히 \[Buy, Sell, Hold\] 확률만 출력하는 것이 아니라, **CoT(Chain of Thought) 토큰**을 먼저 생성하도록 유도.  
   * *예:* "RSI가 30 이하이고 매크로가 긍정적이므로 과매도 구간이다 \-\> 따라서 매수" 라는 내부 잠재 벡터를 형성한 후 Action을 출력.  
3. **Group Sampling:**  
   * 하나의 State에 대해 Dropout 등을 적용하여 $G$개의 서로 다른 예측을 생성.  
4. **Relative Reward Calculation:**  
   * $G$개의 예측 결과에 대해 1시간/4시간 뒤의 수익률을 측정.  
   * 해당 그룹 내에서의 표준화된 점수(Z-score)를 Advantage로 사용.

## **4\. 구체적 실행 계획 (Action Plan)**

### **Phase 1: 모델 코어 리팩토링 (GRPO 구현)**

**파일:** src/models/deepseek\_grpo\_agent.py 및 src/models/grpo\_agent.py

1. **Critic Network 제거:** Actor-Critic 구조에서 Actor Only 구조로 변경.  
2. **Group Sampling 로직 구현:**  
   def get\_action\_group(self, state, group\_size=8):  
       \# 동일 상태에서 노이즈를 주입하여 여러 개의 Action/Thought 샘플링  
       actions, log\_probs \= \[\], \[\]  
       for \_ in range(group\_size):  
           action, log\_prob \= self.policy\_net.sample(state)  
           actions.append(action)  
           log\_probs.append(log\_prob)  
       return actions, log\_probs

3. **GRPO Loss Function 구현:**  
   * 기존 PPO Loss에서 Advantage 항을 (Reward \- Group\_Mean) / (Group\_Std \+ epsilon)으로 대체.  
   * KL Divergence 페널티를 추가하여 구(Old) 정책에서 너무 벗어나지 않도록 제어.

### **Phase 2: 보상 함수(Reward Function) 고도화**

**파일:** src/utils/reward\_functions.py

현재의 단순 수익률 기반 보상은 위험 관리에 취약하다. 의학에서 환자의 예후를 다각도로 보듯 보상 체계를 입체화해야 한다.

1. **기본 보상:** 로그 수익률 (Log Returns).  
2. **위험 조정 보상:**  
   * Sortino Ratio 기반 (하방 변동성에만 페널티).  
   * **MDD 페널티:** 포지션 진입 후 최대 낙폭이 커질수록 기하급수적 페널티 부여.  
3. **Reasoning 보상 (DeepSeek-R1 Style):**  
   * Claude API를 활용하여, 에이전트의 매매 시점이 기술적 분석 원칙(예: 골든크로스, 지지선)에 부합했는지 판단하고 추가 점수 부여 (Rule-based Reward).

### **Phase 3: Claude Skills와 DeepSeek의 결합 (Hybrid Reasoning)**

**파일:** src/claude\_integration/hybrid\_agent.py

* **Teacher-Student Learning:**  
  * 초기 학습 단계에서 Claude(Teacher)가 현재 차트를 보고 "매수 추천"을 하면, RL 모델(Student)이 이를 모방하도록 **BC(Behavior Cloning)** 손실 함수 추가.  
  * 어느 정도 학습된 후에는 RL 모델이 독자적으로 판단하되, Claude는 '리스크 관리자'로서 GRPO의 그룹 샘플 중 너무 위험한 행동을 필터링(Reject Sampling)하는 역할 수행.

## **5\. 예상 결과 및 BM 연결 (Business Model Implication)**

### **5.1 기술적 기대 효과**

* **안정성 확보:** GRPO의 그룹 정규화 효과로 인해 Phase 3에서 보였던 급격한 성능 하락(Policy Collapse) 방지.  
* **설명 가능성(XAI):** CoT 방식을 도입함으로써, 모델이 왜 매매했는지에 대한 '근거'를 어느 정도 역추적 가능 (투자 자문 서비스 시 고객 브리핑에 활용 가능).  
* **일반화:** 특정 종목에 과적합되지 않고, 다양한 자산군(Multi-Asset)에 적용 가능한 범용 에이전트 확보.

### **5.2 퀀트 자문사 비전과의 연계**

DR\_BreakFast님의 비전인 "의학, 금융, 정책 도메인의 융합"을 실현하기 위해 본 모델은 다음과 같이 활용될 수 있습니다.

1. **AI 리터러시 교육용 데모:** ETIV AI Institute에서 'AI가 스스로 생각(Reasoning)하며 투자하는 과정'을 시각화하여 보여줌으로써 AI 교육 자료로 활용.  
2. **개인화된 포트폴리오 관리:** 노화 방지 및 스포츠 의학이 개인 맞춤형 처방이듯, GRPO 모델의 보상 함수 파라미터(리스크 성향)를 조절하여 고객별 맞춤형 퀀트 상품 구성 가능.  
3. **정책 분석 봇:** 향후 정치/정책 텍스트 데이터를 입력으로 받아, 정책 변화가 특정 섹터에 미칠 영향을 DeepSeek-R1의 추론 능력으로 분석하고 선제적 포지셔닝 제안.

## **6\. 결론**

현재의 코드는 훌륭한 골격을 갖추고 있으나, 근육(알고리즘)과 신경계(보상 체계)의 미세 조정이 필요합니다. DeepSeek-R1의 GRPO 알고리즘은 \*\*"비평가 없는 효율적 학습"\*\*과 \*\*"집단 지성을 통한 최적화"\*\*라는 특성을 통해, DR\_BreakFast님의 시스템을 단순한 자동매매 프로그램에서 \*\*'스스로 사고하는 금융 AI 에이전트'\*\*로 진화시킬 것입니다.

우선적으로 \*\*GRPO Agent 구현(Phase 1)\*\*과 \*\*보상 함수 재설계(Phase 2)\*\*에 착수하실 것을 권해드립니다.

**작성자:** Gemini (DR\_BreakFast님의 AI 파트너)