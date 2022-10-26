### 7주차 - Node lovel Network Embedding

##### Outline

- Overview

- Homogenous Network Embedding

- Multi-aspect Network Embedding

- Attributed Network Embedding

- Heterogeneous Network Embedding

- Task-guided Network Embedding



-----

##### Overview

![](./picture/7-1.png)

- Representation Learning은 기계가 자동으로 특징을 추출한다! 

<br>

##### Graph representation learning

- goal : Original network와 유사하게 Embedding Space로 node 정보들을 Encode 하자!
  
  ![](./picture/7-2.png)
  
  > Embedding 한 후에도 기존 공간에서의 정보가 유지되도록 한다.
  
  - Embedding은 각 Task와 독립적으로 진행된다. 
  
  - Embedding 된 Vector들은 각각의 Task에서 활용된다. 



<br>

##### Node embedding

- 고려해야할 2가지 
  
  - 1). 어떻게 node들을 encode 할 것인가? -> Encoder
  
  - 2). Similarity(유사성)을 Embedding 공간에서 어떻게 정의할 것인가? -> Decoder





###### Encoder

- 목적 : 각 노드들을 저차원의 Vector로 Mapping한다. 
  
  > $ENC(v) = z_v$
  > 
  > > $v$ : Node in the input graph
  > > 
  > > $z_v$ : d-dimensional embedding vector 

- 가장 단순한 Encoding 접근은 One-hot vector을 통한 Embedding-look up 이다. 
  
  > ![](./picture/7-3.png)

<br>

###### Decoder(similarity fuction)

- 어떻게 original graph의 관계를 Embedding space의 관계로 Mapping할지 구체화한다.
  
  > ![](./picture/7-4.png)

<br>

###### Encoder + Decoder Framework (Simplest version)

- 목적 : Maximize $z_v^Tz_u$ for node pairs (u,v) that are similar 
  
  - decoder를 $z_v^Tz_u$로 설정했기 때문 

- Node similarity를 정의하는데 있어 여러 방안이 있다. 
  
  - Node 간 연결되었는가? 
  
  - 동일한 이웃을 공유하는가?
  
  - 유사한 구조적 역할을 맡고 있는가? 
    
    +ETC 



---------------

##### Homogeneous Network Embedding



###### Random walk

- 방법 : Starting Node에서 시작하여, 갈 수 있는 neiborhood node들을 전략에 따라 확률적으로 이동하여 이동한다.

- 의도 :  **Random walk는 Local, High-order neighborhood 정보 모두 고려할 수 있다.**
  
  > ![](./picture/7-6.png)
  > 
  > Strategy R이 무엇이냐에 따라서 Random walk의 방식이 달라짐 
  > 
  > 노드 간 거리가 가깝고, 연결 Egde가 많을수록 확률값이 높다. 
  > 
  > 노드 간 거리가 멀고, 연결 edge가 적을수록 확률값이 낮다. 



##### Random Walk-based node embedding

- Input : G = (V,E)

- Goal : Mapping 함수 $f : u -> R^d $ for $u \in V$  을 학습하자! 
  
  > $f(u) = z_u \in R^d$

- 과정 
  
  - 1). 일부 Random walk Strategy R 기반으로 고정된 길이의 Random walk을 각 node 별로 진행한다. 
  
  - 2). 각 노드 $u$에 대해서 Random walk를 통해 방문했던 Node들의 set $N_R(u)$ 을 만든다. 
    
    > ![](./picture/7-7.png)
  
  - 3). $max_f \sum_{u\in V} logP(N_R(u)|z_u)$ 을 만족시키도록 Embedding을 최적화시킨다.
    
    > ![](./picture/7-8.png)
    
    > $P(v|z_u)$ 은 Softmax를 통해서 Parameterize 한다.
    > 
    > > $P(v|z_u) = \frac{exp(z_u^Tz_v)}{\sum_{j \in V} exp(z_u^Tz_j) }$
    > 
    > 단, 분모의 $\sum_{j \in V} exp(z_u^Tz_j)$은 $O(|V|^2)$ 의 계산 복잡도를 가지고 있어 다른 방식이 필요하다. 
    
    
    
    > By Noise Constractive Estimation(NCE), 
    > 
    > $P(v|z_u) = \frac{exp(z_u^Tz_v)}{\sum_{j \in V} exp(z_u^Tz_j) } \sim log(\sigma(z_u^Tz_v)) - \sum_{j=1}^k log(\sigma(z_u^Tz_j))$
    > 
    > > $\sigma(x) = \frac{1}{1+e^{-x}}$
    > > 
    > > $j \sim P_V$. s.t. $P_v$ : Random distribution over nodes 
    > 
    > -> 전체 노드에 대해서 정규화를 해주는 것이 아니라, 중요한 k개의 샘플(Negative samples)에 대해서만 정규화해준다. k개의 샘플링은 각 노드의 degree를 고려한다.
    
    > 최적화 간 모든 node에 대해서 gradient descent를 하는 것이 아닌, 각 단일 노드에 대해서만 진행한다. 
    > 
    > - 전체에 대해서 update 하는 것은 계산양이 많아 비현실적이다.
    > 
    > ![](./picture/7-9.png)
    
    
    
    > f에 대한 어떠한 예시도 없어서 어떻게 업데이트 되는지 감이 안오네 
    
    - 주어진 Node u에 대해서 각 Neighboring nodes의 확률을 최대화한다.
    
    - Node u의 Embedding이 각 Neighboring node을 예측할 수 있게 만든다. 

<br>

###### Random walk의 전략 R에 따라서 $N_R(u)$를 만드는 방식이 달라진다.

- 1). Deep walk : Simple random walk. (나중에 다룰 예정)

- 2). node2vec 방식 : Biased random walk



---- 

##### Node2Vec : Biased Walks

- Idea : Local 과 Global 시각을 trade off 할 수 있도록 flexible하고 biased 한 Random walk를 활용한다. 
  
  > Random walk는 크게 2가지 방향성이 있다. 
  > 
  > ![](./picture/7-10.png)
  > 
  > - Deepwalk's simple walk는 Global view(BFS)에 집중한다.

- 방법 : BFS와 DFS를 조율하는 2개의 parameter을 사용한다.
  
  > ![](./picture/7-11.png)
  > 
  > u에서 시작해서 random walk가 $s_1$을 거쳐 w까지 간 상태이다. 
  > 
  > $s_1$으로 가면 return 하는 것이며, $s_2$ 는 $s_1$와 연결되어 있기에 '$s_1$ -> w' 와 '$s_1 $-> $s_2$' 거리가 같게 이동하는 것이다. 마지막으로 $s_3, s_4$ 는 기존 노드에서부터 더 멀어진다. 
  > 
  > > Return parameter p : 이전 노드로 돌아온다. $\frac{1}{p}$는 이전 노드로 돌아갈 확률을 의미한다. 
  > > 
  > > Walk away parameter q  : 더 나아간다. $\frac{1}{q}$ 는 더 나아갈 확률을 의미한다.
  
  - p의 값이 낮으면 return 할 확률이 높아져 BFS-like walk가 된다. 
  
  - q의 값이 낮으면 더 나아갈 확률이 높아져 DFS-like walk가 된다. 
    
    > 이것만 보면 p와 q는 독립적으로 보이는데? 아마 확률 값이 1이다를 통해서 trade off 하겠지? 



----

#### LINE : Large-scale Information Network Embedding

- Idea : First-order and second-order proximity를 보존하자 [BFS 방식]
  
  > ![](./picture/7-13.png)
  
  
  
  > First-order Proximity의 경우 많은 link들이 관측되지 않는다는 점, 그리고 전체 네트워크 구조를 담기엔 충분하지 않다는 단점이 있다. 
  > 
  > ![](./picture/7-14.png)
  
  > Second-order proximity는 노드들의 이웃 구조 사이의 근접성을 의미한다.
  > 
  > ![](./picture/7-15.png)
  
  > First-order와 Second-order의 Proximity 는 $\sum$ 에 들어가는 항목만 다르다. 



----

#### SDNE - Structural Deep Network Embedding

- Idea : 다수의 Layer을 가진 Auto encoder를 기반으로 Node embedding을 적용한다.
  
  - 이를 통해서 Shallow 모델이 포착할 수 없는 highly non-linear network structure을 embedding 한다. (Deepwalk, node2vec, Line 모두 Shallow-단층 모델이다)
  
  > ![](./picture/7-16.png)
  
  - second order proximity를 측정하기 위해서 각 노드별로 원본과 복원한 값의 차이를 구한다. 
    
    > ![](./picture/7-17.png)
    > 
    > Q. 왜 복원한 값과의 차이를 구하는게 second order의 Loss가 되는 거지? 
  
  - first order proximity를 측정하기 위해서 서로 다른 노드 간 연결된 경우 때로 한정하여 측정한다.
    
    > ![](./picture/7-18.png)
  
  - 최종 Loss 함수는 first order의 loss와 second order loss의 합으로 한다.
    
    > $L_{mix} = L_{2nd} + \alpha L_{1st}$






















