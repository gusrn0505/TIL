### 1주차 - Probability Theory



##### 서론

- 통계와 확률 이론은 **불확실성을 다루기 위한 학문**이다. 
  
  - 확률 이론은 Uncertain 하고 noise가 있을 data로부터 통계적인 추론의 근간이다. 
  
  - 따라서 불확실성을 다루는 모든 분야에 대해서 적용가능하다. 



- 통계적 추론은 Descriptive / Inferential 과정을 모두 포함한다. 
  
  - Descriptive Statistics : Data의 특성을 알기 위한 것 
  
  - Inferential Statistics : 문제의 Sample로 부터 결정을 내리거나 추론하는 것



##### 용어 정리

> Experiment : 1개 이상의 결과가 나올 수 있는 어떠한 과정 또는 과정들 
> 
> Sample Space : 일어날 수 있는 모든 경우의 수로 구성된 집합 
> 
> 확률(Probabilities) : 다음 조건을 만족하는 Sample space 속 경우의 수의 가능성
> 
> - 0 <= $p_i$ <= 1 & $\sum^n_{i=1} p_i = 1 $ for i in [1,2, ..., n] 
> 
> Equally likely : 모든 경우가 n이라 할 때, 모든 확률이 $\frac{1}{n}$ 이 되는 것 
> 
> Event : Sample space의 부분집합. 일어날 수 있는 일부 경우들의 집합
> 
> Contidional Probability : $P(A|B) = \frac{P(A\cap B)}{P(B)}$ 



- **Independence**
  
  - Mutual : $P(A_1 \cap ... \cap A_n) = P(A_1) P(A_2) ... P(A_n)$
  
  - Pairwise : $P(A_i \cap A_j) = P(A_i) P(A_j) for \forall i,j \in [1,2,...,n] $ with $i\neq j$ 
    
    - Pairwise 은 Mutual 에 비해 약한 Indep 조건이다. 
  
  - Conditional : 관측 유무에 따라 Indep 결정됨. 



- Law of Total Probability 
  
  > $P(B) = \sum^n_{i=1} P(A_i)P(B|A_i)$ 



- <mark>Bayes' Theorem(사후 확률 계산)</mark>
  
  > $P(A_i|B) = \frac{P(A_i)P(B|A_i)}{\sum^n_{j=1} P(A_j)P(B|A_j)}$
  > 
  > - 계산 또는 관측 가능한 $P(A_i), P(B|A_i)$ 를 기반으로 <mark>직접 구할 수 없는 $P(A_i|B)$ 를 구해낸다. </mark>
  > 
  > > ex)- $A_i$ : parameter / Disease type 
  > > 
  > > ex)- B : observable random variable / Symptom 




