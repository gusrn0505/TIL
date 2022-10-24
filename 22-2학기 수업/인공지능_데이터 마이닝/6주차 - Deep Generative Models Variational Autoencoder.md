### 6주차 - Deep Generative Models Variational Autoencoder

##### Taxonomy of deep Generative model

![](./picture/6-1.png)

> 수년 전에 작성된 분류 모델. 이 중에서 살아남지 못한 것들이 꽤 있음 
> 
> - Ex)- Tractable Density, Markov Cahin Sampling(GSN), Markov Chain도 에너지 모델의 형태로 있으나 RBM은 잘 사용이 안됨 
> 
> 교수님의 생각으론 크게 2가지 기준 1) Explicit / Implicit, 2) Likelihood / free-Likelihood 이렇게 나눠 볼 수 있을 듯 

----

##### Autoencoders

- Hidden 층의 차원을 정하는 것 : 정보의 Loss 또는 Hidden layer의 크기 중 무엇을 중요시 할 것인가? **[Trade off 관계]**
  
  ![](./picture/6-2.png)
  
  - Hidden layer의 크기를 키우면 계산양이 많아지는 대신 정보의 loss가 줄어든다. 
  
  - 반대로 줄이면 계산양이 적어지는 대신 정보의 loss가 늘어난다. 

----

##### Amortized Analysis and Inference

- 기존의 variational inference에서는 데이터셋이 주어지지 않았을 때 만들어진 수식들이있다. 
  
  - 데이터 셋이 없어도 모델의 구조, genetic process만 명시되면 구할 수 있다.
    
    > 구체적으로 무엇이 구조고 Genetic process는 뭘 의미하지? 
    > 
    > 구조 : Analytically soved 된 것. 음.. E-M Algotirm과 Loss 함수를 의미하나? 
    > 
    > - $P(H|E)$ 이건 데이터 기반이라고 해야하지 않을까? 아닌가? 
    > 
    > Genetic(유전) Process : 세대를 거쳐가며 점차 발전하는 모델? 
  
  - Big O 등을 계산할 수 있다. 하지만 데이터셋을 활용하지 않은 것으로 의미가 크게 없다.
  
  -> <mark>즉, 데이터에 기반하지 않은 inference다.</mark>

- **CS 분야에서 계산양(big O)에 대해 유의미하려면 Amortized Analysis를 말해야 한다.**
  
  > Amortized Analysis : 데이터 셋이 존재했을 때의 Complexity를 분석하는 방법
  > 
  > ex)- 데이터 셋이 다음과 같은 분포일 때Binary search tree 를 만들면 Complexity가 어떻게 되는가? 해쉬 구조를 만들면 어떻게 되는가? 

<br>

- **Amortized inference는 데이터셋이 존재했을 때의 inference를 의미한다.** 
  
  - Amortized inference는 경험적으로 모델을 학습하는 방식의 추론을 의미한다.
    
    - 새로운 데이터가 들어왔을 때 파라미터를 전부 새롭게 바꿔야 하는지, 또는 새로운 경우만 고려하면 되는지로 나눠 볼 수 있다.  
    
    - 데이터 셋 없이 Analytically parameter inference를 한 VI는 Amortized inference가 아니다.
    
    - 반면 Black box inference는 경험적으로 데이터셋이 있는 상황에서만 가능하니 Amortized inference에 속한다. 
      
      - Black box는 Amortized inference로 계산 방법 중에서 MCMC로 국한하여 계산하려고 했던 것 
  
  - <u>MCMC보다 좋은 Inference 방법이 개발되었다. </u>
    
    - Variational inference의 결과물로써 $\phi_i$ 에 대한 update 수식을 얻을 수 있다. 
      
      > P(H|E, $\lambda$) 에 대한 variational distribution $q_E(H; \phi)$ 에 대해, 
      > 
      > Mean-field assumption을 통해서 $q_E^{MF}(H; \phi) = \prod_iq(H_i; \phi_i)$ 로 표현할 수 있다. 
      > 
      > 그리고 ELBO, KL-term의 최적화하는 과정에서 $\phi_i$에 대한 업데이트 수식을 계산했다. 
    
    - 이후 데이터 셋이 들어왔을 때, $\phi_i$ 업데이트 수식을 통해서 $\phi_i$ 가 생성된다.
    
    - <mark>$\phi_i$ 가 만들어지는 과정을 Neural network로 표현하자!</mark> Update 수식도 함수니까 Neural network로 표현할 수 있다. 

<br>

- **해결해야 하는 문제 : 어떻게 inference를 Neural Network로 훈련시킬 수 있는 구조를 만들 수 있을까?**
  
  > **어떻게 하면 Stochastic 변수를 Neural Network에서 학습시킬까?** 
  > 
  > ↔ 기존 NN 연구들은 Deterministic 한 환경에서 작동해왔음. 
  
  - Kingma "Auto-encoding variational Bayes" 논문에서 구조를 제시함. 
  
  ![](./picture/6-3.png)
  
  > ELBO 구조에서 P와 Q 분포의 대칭성을 찾아낸 것 
  > 
  > $P_\theta(E|H)$ : Hidden 층을 input으로 받아 Evidence를 output으로 생성 
  > 
  > $Q_\phi(H|E)$ : Evidence를 input으로 받아 Hidden 층을 output으로 생성 
  > 
  > - Variational Inference이기도 하다. Variatoinal inference는 Latent space에 대한 Distribution이니까. 
  
  > $p_\theta(H)$ : Prior. 사실상 학습에 어떠한 영향을 미치지 않음. 
  
  <br>
  
  - **핵심은 "reparametrization Trick(또는 Stochastic Gradient Variational Bayes)"이다.**     
    
    - $q_\phi(H|E)$ 를 미분가능하고 결정적인 변수로 reparameterize 하는 것이다. 
    
    > ELBO에서 미분을 한다는 것은 아래 식으로 풀어볼 수 있다. 
    > 
    > ![](./picture/6-4.png)
    > 
    > $E[X] = \lim_{n -> \infin} \frac{\sum_{i=1}^n X}{n}$
    > 
    > $f'(x) = \lim_{h-> 0} \frac{f(x+h)-f(x)}{h}$ 
    > 
    >  이 두개의 lim은 특정 조건을 만족하지 않는 이상 서로 위치를 바꿀 수 없다.   바꿀 수 없다면 우리는 ELBO 식에 Differentiation이 불가능해진다.  
    > 
    > - Hidden Variable의 개수가 Finite이다? 
    > 
    > - Gaussian distribution은 조건을 충족하여 ELBO를 계산할 수 있음
    
    > Neural Network의 Chain rule을 적용할 때, Stochastic한 h의 값을 1)update 해야하는 deterministic value와 2)update를 하지 않아도 되는 Stocastic Value(e)로 구분하자. 
    > 
    > ![](./picture/6-6.png)
    > 
    > ![](./picture/6-5.png)
    > 
    > - $\epsilon$ 은 고정되어 있음. Chain rule에서 포함되어 있지 않음. 
    > 
    > - 학습해야하는 deterministic value $\phi, e$ 에는 Chain rule을 통해서 학습시킴. 
    > 
    > - Neural Network가 학습하기 위해선 E(dataset) 이 들어와야 하니 Amortized inference 임.  

    > H의 값을 Stochasitic 이라하면 H 자체에도 Variance가 생김. 그 결과 전체 모델에 어마어마한 Variance를 가져오게됨 
    > 
    > ![](./picture/6-7.png)
    > 
    > MC를 통해서 계산가능. Black box의 잔재. 

<br>

- 앞으로의 질문 - Gaussian distribution 이 아닐 때는 어떻게 다룰 것인가? 
  
  > ex)- gamma, beta
  > 
  > 문일철 교수님이 Drichlet 상황에서는 해놓음. 
  > 
  > Continuous 상황에서는 DNSP Assmuption을 적용하여 또다른 Reparametrization trick을 적용하여 진행할 수 있음. 

<br>

##### Derivation of Evedence Lower Bound for Gaussian VAE

![](./picture/6-9.png)

>  계산을 줄이기 위해서 analytic 하게 풀이함.
> 
> - 자세한 풀이 과정은 교수님의 유튜브 참고하기 
> 
> E는 Explicit 모델로 계산해주자! 
> 
> ![](./picture/6-10.png)

- Loss 수식 구했고, 미분 가능하고, reparameterization trick으로 모델도 안정화되었다. 

    →  계산하자!

![](./picture/6-11.png)

- VAE를 통해 Stochastic한 Value 들도 Deterministic value들과 동일하게 NN으로 학습할 수 있다.

---- 

#### Variants of Variational Autoencoder with conditional probability

- 기존의 Topic modeling 방식과 다르게 VAE는 Variation이 매우 낮아서 Flexible하고 안정적이였다. 즉, 기존의 방법들을 다 압도할 방법이 나온 것이다. 
  
  - 이제 더 이상 Parameter inference에 대해서 더 고민을 안해도 된다.

- 이후 지금까지 구상했던 구조들을 모두 VAE를 통해서 다시 표현하고자 했다.  
  
  > VAE는 Bayesian network로 표현하면 H -> E 인 구조다. 

<br>

###### Conditional Variational Autoencoder

- VAE에 Condition을 추가하자!(Hierarchical model을 만들자!)
  
  ![](./picture/6-13.png)
  
  - "e|y" 로 Condition을 표현하자!
  
  > y는 random variable로 표현하면 가장 일반화한 것. 지금은 Observed data로 대입함. 아직 연구가 더 필요한 부분  
  
  ![](./picture/6-14.png)
  
  - 정리하고 보니 기존의 VAE와 크게 다르지 않은 모델이 나옴. 
  
  - VAE 에서의 구조와 Variational Inference 구조는 동일함
    
    - Y가 관측되지 않는다고 하면 y가 왼쪽으로 넘어갈 뿐. (VI)
    
    - 둘다 Mean field assumption을 적용함
    
    - VAE와 VI를 구분하는 것은 하나, Dataset이 주어져서 Amortized variational inference가 일어난다는 것 뿐이다. 
    
    - 즉, Neural Network를 Parameter Inference 수식을 유도하기 쓴다는 점 외에는 모두 동일하다.

<br>

- **Conditional VAE가 계속 연구가 계속되는 이유** 
  
  - Mode Collapse를 해결하자! 
  
  - Mode Collapse는 Generative model, MCMC Sampling 간 Random walk 가 좋지 않을때도 일어남 
    
    - -> unobserved 데이터 분포를 inference를 하는 모든 상황에서 일어날 수 있음 

<br>

##### Variational Auto(?) Deep Embedding(VADE)

![](./picture/6-16.png)

> e ↔ x 
> 
> h ↔ z

- q distribution을 구했으니, 이를 ELBO 형태로 표현하자. 

##### Key Idea on Probabilistic Modeling

![](./picture/6-15.png)

> by mean field assumption, $q_\phi(h,c|x) = q_\phi(h|e) q_\phi(c|e)$
> 
> c는 Discrete 한 값으로 따로 고려해줘야 함. 

----

#### Variants of Variational Autoencoder with Elaborated losses

- Support Vector Machine(SVM) 
  
  ![](./picture/6-17.png)
  
  ![](./picture/6-18.png)
  
  - Kernel trick 까지 적용될 때 많은 의미를 가짐 
    
    > Kernel trick : Kernel trick은 Vector의 Basis를 확장하는 것을 모델링 하지 않고 Inner product를 계산하는 것. 원 정보를 담고 있는 Vector의 Basis를 Scalability 하게 등 다양하게 확장시킬 수 있는데, 이때 Kernel trick을 사용하면 계산양을 늘리지 않고도 원하는 목표를 이룰 수 있다는 것
  
  - SVM의 Loss는 Margin-based Loss로 Likelihood loss로 다르다. 
    
    > SVM의 Loss 함수를 "Hinge Loss" 라고 부름 
    
    ![](./picture/6-19.png)
    
    > 빨강색 : Hinge loss. 차이가 나는 지점부터 Linearly loss 값이 커짐.  
    > 
    > 파란색 : log logg. 항상 penalty를 부여함. 
    > 
    > - NN에서 마지막 층에서 Logistic function을 적용하기 때문에 다 log loss가 된다. 
    > 
    > 검은색 : Z 로몬 loss? 차이가 커져도 항상 1의 Penalty를 부여함. 
    > 
    > 값이 1에서 차이가 나는 것은 SVM 모델에서 (we +b)y = 1 로 둬서 이지 않을까? 

- 다음 시간 질문 : SVM Margin-based 사람들에게 ELBO를 적용할 때에는 어떻게 사용할까? 
