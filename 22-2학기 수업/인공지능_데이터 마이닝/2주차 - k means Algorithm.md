#### 2주차 - k means Algorithm

출처 : GSDS 문일철 교수님 - "Application of AI and DM Technology"

##### K-means Algorithm

- k 개의 중심점을 설정한 다음, 거리를 기반으로 가장 가까운 점들을 Clustering 하는 것
  
  > Minimize J : <mark>$\sum^N_{n=1} \sum^K_{k=1} r_{nk} ||x_n-\mu_k||^2$</mark>
  > 
  > > $r_{nk} :$ n개의 점들이 k개의 Cluster에 속했는지 여부(0,1의 BInary 값)
  > > 
  > > $\mu_k$ : k번째 중심점의 위치 (평균이 아님!)
  
  - $r_{nk}, \mu_k$ 을 최적화함으로써 J를 최소화할 것임.
    
    - $r_{nk}, \mu_k$ <u>두 값은 상호작용하기 때문에 교대하며 최적화할 것이다</u>. (ex- EM 방식)
    
    - 이 방식은 <mark> Decision Variable이 많은 경우 항상 적용하는 방법</mark>이다. 

<br>

- **<mark>E-M Algorithm </mark>**
  
  - E(xpectation) : 주어진 파라미터 기반으로 기댓값을 갱신(계산)
    
    - 거리 계산을 통해서 $r_{nk}$  를 가장 가까운 Centroid(k)로 반영
    
    > $r_{nk} =$ {0,1}
  
  - M(aximization) : 주어진 값(ex-$r_{nk}$)을 기반으로 가장 가능성 높은 파라미터를 갱신
    
    - $\mu_k$ 를 업데이트 
    
    > $\frac{dJ}{d\mu_k} = \frac {d}{d\mu_k} \sum^N_{n=1} \sum^K_{l=1} r_{nl}||x_n-\mu_l||^2 = \frac{d}{d\mu_k} \sum^N_{n=1} r_{nk}||x_n - \mu_k ||^2$
    > 
    >         $= \sum^N_{n=1} -2r_{nk}(x_n-\mu_k) $
    > 
    >         $= -2(-\sum^N_{n=1} r_{nk}\mu_k + \sum^N_{n=1} r_{nk} ||x_n - \mu_k||^2$
    > 
    > <mark>$\mu_k = \frac{\sum^N_{n=1} r_{nk} x_n}{\sum^N_{n=1} r_{nk}}$</mark>

- 아직까진 E-M Algorithm을 사용하는 데 있어 어떠한 분포도 가정하지 않았다.  
  
  - 하지만 Loss 함수 J가 Mean-sqaured error로 정의된 것으로 보아, Normal distribution을 이미 가정하고 있다. 
    
    > Normal distribution : $\frac{1}{\sigma \sqrt{2\pi}} exp(-\frac{(x-\mu)^2}{2\sigma^2})$
    > 
    > MSE는 Normal 분포에 log를 취한 것에서부터 유래한다. <mark>Loss 함수를 MSE로 정의하는 것은 Normal 분포를 전제하는 것과 동치이다</mark> 
  
  - 이미 Normal을 가정하고 있기 때문에 언제든 Probability를 계산할 수 있다.

<br>

- **Property of K-means algorithm** 
  
  - 클러스터의 개수가 정해져 불확실하다.
  
  - 중심점의 최초 위치가 어디냐에 따라 결과가 달라질 수 있다. 
  
  - 거리 측정 방식에 따라 한계가 있다. 
  
  - Hard 클러스트링 했을 때, 기존 특성이 덮여버릴 수 있다. 
    
    > ex)- 빨간 30%, 파랑 70% 였는데, Clustering 시 파랑 100%로 단순화된다.

----

#### Gaussian Mixture model

**Multinomial Distribution**

- k 개의 Binary Variable($\mu_k$) 중 1개를 선택하는 것. [Binomial distrubution 일반화]

> $\sum_k x_k = 1, P(X|\mu) = \prod^K_{k=1} \mu_k^{x_k} s.t. \mu_k >=0, \sum_k \mu_k =1$
> 
> > $x_k : ${0,1}. 해당될 때에만 1의 값을 가진다. 
> > 
> > $\sum_k x_k =1 : $ k 번째에 해당될 경우에만 1의 값을 가지며, 나머지 경우에는 0의 값을 가진다. 

- N개의 Selection을 할 경우,
  
  >  $P(X|\mu) = \prod^N_{n=1} \prod^K_{k=1} \mu_k^{x_k} = \prod^K_{k=1} \mu_k ^{\sum^N_{n=1} x_{nk}} $=<mark> $\prod^K_{k=1} \mu_k^{m_k} $</mark>
  > 
  > > $m_k = \sum^N_{n=1} x_{nk}$. N개의 선택 중 k 번째 옵션을 선택한 횟수를 의미 
  > 
  > => <u>Distribution 형태로 표현할 수 있다. </u>

<br>

- **Distribution을 알 때 할 수 있는 것**
  
  - 샘플을 생성할 수 있게 해준다.
    
    > *바로 이 말을 할 수 있나? CDF 랑 Inverse를 구할 수 있어야 할 텐데?* 
    > 
    > -> 구할 수 있는 분포로 가정하겠지 ㅡㅡ
  
  - 특정 event에 대한 밀도 측정(Density estimation)이 가능하게 해준다.
  
  - 단, <u>정확한 distribution 함수를 만들기 위해서 파라미터 $\mu$을 최적화하는 것이 선행되어야 한다.</u>  
    
    - 파라미터 $\mu$ 를 최적화하는 방법은 많다. 그 중에 하나가 Maximum likelihood 이다. 

<br>

- **Lagrange Method - 파라미터 $\mu$ 최적화 방법의 일환**
  
  - 파라미터 $\mu, m, \lambda$ 에 대해서 Lagrange Function으로 정의 
    
    > $L(\mu, m, \lambda) = \sum^K_{k=1} m_k ln\mu_k + \lambda(\sum^K_{k=1} \mu_k -1) $
    > 
    > > $\mu_k$는 $\prod$ 연산을 거치기 때문에 반응 정도를 낮추기 위해 log를 사용 
    > > 
    > > $\sum_{i=1}^k \mu_k =1$ 로 agrange function을 적용할 때 위처럼 수식을 세움 
  
  - Lagrange function을 편미분하여 optimal value를 찾는다.
    
    > $\frac{d}{d\mu_k} L(\mu,m, \lambda) = \frac{m_k}{\mu_k} + \lambda = 0  →$ <mark>$\mu_k = -\frac{m_k}{\lambda}$</mark>
  
  - 제약조건을 사용하여 제약 조건 내 최적값을 찾는다. 
    
    > 제약조건 : $\mu_k >= 0 , \sum_k \mu_k =1$, when $m_k = \sum^N_{n=1} x_{nk}$
    
    > $\sum_k \mu_k =1$ 
    > 
    > $→ \sum_k -\frac{m_k}{\lambda} =1 $
    > 
    > $→ \sum_k m_k = -\lambda $
    > 
    > $→ \sum_k \sum^N_{n=1} x_{nk} = -\lambda$
    > 
    > $→ N = -\lambda$
    > 
    > <mark>$\mu_k = \frac{m_k}{N} = \frac{발생 수}{ 총 시행 수}$ : MLE of multinomial distribution</mark>
    > 
    > - 추후에는 Gaussian Distribution 외에 다른 분포에서도 MLE를 잘 도출해낼 수 있어야 한다.

-----

##### **Multivariate Gaussian Distribution**

- ![](picture/2-1.png)
  
  > $\sum : \sigma$
  > 
  > Tr(A) : $\sum_{i=1}^n A_{ii}$. 행렬 A의 대각행렬의 값을 다 더하는 것 
  
  > 파라미터 $\sigma, \mu$ 에 대해서 미분하여 최적 값 찾기. 

- 나중에는 Gaussian Distribution 외에 다른 분포에도 MLE를 도출해낼 수 있어야 한다.

----

#### **(Gaussian) Mixture model**

- 데이터가 하나의 분포가 아닌, 여러 분포에서 나왔다고 가정한 모델
  
  ![](picture/2-2.png)
  
  > $P(x) = \sum^K_{k=1} \pi_k N(x|\mu_k, \sigma_k)$
  > 
  >             $= \sum^K_{k=1} P(z_k) P(x|z)$
  > 
  > > $\pi_k$ : Mixing coefficients. s.t. $\sum^k_{k=1} \pi_k = 1, 0 <= \pi_k <= 1$
  > > 
  > > - 각 정규분포 $N(x|\mu_i, \sigma_i)$ 별 가중치이자, x가 k번째 분포에 속할 확률
  > > 
  > > <mark>$z_k$</mark> : New variable s.t. $z_k \in$ {0,1}. x가 k번째 분포에서 유래했는가 여부
  > > 
  > > -  $\sum_k z_k = 1, P(z_k =1 ) = \pi_k$. P(z) = $\prod^K_{k=1} \pi_k^{z_k}$  
  
  > $P(X|z_k=1) = N(x|\mu_k, \sum_k)$       ($\sum_k$ 는 k번째 분포의 공분산을 의미)
  > 
  > → $P(X|Z) = \prod^K_{k=1} N(x|\mu_k, \sum_k)^{z_k}$

<br>

- **Mixture 모델의 Graphical 형태**
  
  ![](picture/2-3.png)
  
  > $\pi, \mu, \sum$ 은 파라미터로 Decision Variable, X는 Observed Variable. 
  > 
  > 즉, Latent variable인 z값만 Inference 대상이 됨
  > 
  > 만약, 파라미터가 Hard Value가 아닌 Stochastic value 형태가 된다면, 이는 Bayesian 방식이 된다. 
  > 
  > - Parameter의 Randomize 화! 확률 분포를 지닌 random variable이다.   



- [E- step]**각 $x_n$이 특정 Cluster에 속할 확률 계산하기 [$p(z_k=1|x_n)$ 구하기]**
  
  > $\gamma(z_{nk})$을 $p(z_k =1 |x_n)$ 이라 하자 
  > 
  > $\gamma(z_{nk}) = \frac{P(z_k=1) P(x_n|z_k=1)}{\sum^K_{j=1} P(z_j=1)p(x_n|z_j=1)}$     [by Law of total probability]
  > 
  >               $= \frac{\pi_k N(x|\mu_k, \sum_k)}{ \sum^K_{j=1} \pi_i N(x|\mu_j, \sum_j)}$
  > 
  > - 원하는 $\gamma(z_{nk})$ 를 계산할 수 있도록 맞춰서 함수의 형태를 구성했다. 
  > 
  > $\theta = [\mu, \sum, \pi]$ : 파라미터 set. 이 값을 알아야 z의 값을 구할 수 있다. 
  > 
  > <mark>-> 가장 $\gamma(z_{nk})$ 의 값이 큰 것으로 $r_{nk}$을 갱신한다</mark> 

<br>

- [M-step]**각 $x_n$ 특정 Cluster에 속할 확률을 최대화하는 파라미터들로 갱신**
  
  ![](picture/2-4.png)
  
  > 여러 값들을 $\gamma(z_{nk})$ 로 변경
  > 
  > $\mu_k$, $\sum_k, \pi_k$ 값 갱신하기 

<br>

- **단, Gaussian Mixture Model은 Cluster 모델이 아니다!**
  
  - Mixture 모델 설정 상 Cluster가 나온거지, Cluster 자체가 목적이 아니다. 
  
  - <u>Gaussian Mixture Model의 목적은 Density estimation 이다! </u>
    
    - 다른 Distribution을 적용할 때에도 동일하게 Density estimation을 추구한다.

<br>

- **k-means 와 GMM의 관계**
  
  - GMM의 Covariance($\sum$)를 $\epsilon I$ 라고 가정해야 K-mean과 유사해진다. 
    
    <mark>[k-mean은 GMM의 매우 특수한 경우이다.]</mark>
    
    ![](picture/2-5.png)
    
    - 이때, $\sum_k = \epsilon I$ 라고 해보자 
      
      ![](picture/2-6.png)
    
    > $\epsilon → 0 $ 일때, $||x - \mu_k||$는 0에 가장 늦게 수렴한다. 
    > 
    > 다른 항목들이 0이 될 때, 가장 큰 값인 $exp(-min(||x-\mu_k||^2))$ 만이 유의미한 값으로 수렴한다. 
    > 
    > <mark>즉, min($||x-\mu_k||^2)$ 외에는 고려하지 않는 Hard assignment가 된다. </mark>
  
  - GMM은 k-means을 Soft assignment + Covariance matrix learning 한 일반화이다. 

----

##### EM Algorithm

- Supervised / Unsupervised learning 이 섞여 있는 경우가 너무 많다. 

- <mark>즉, SL/USL 두가지를 언제 나눠서 사용하는지, 상황을 이해하는게 더 중요하다.</mark>
  
  - Latent Variable 유무는 주어지는 X,y의 형태, 조건에 따라 변한다. 
    
    > Latent Variable : 숨겨져 있는 변수들 
    
    - Training 모델에서는 X(input), Y(label) 이 모두 주어짐(SL 버전)
      
      - Latent variable이 없음 
    
    - Tasking 모델에서는 X만이 주어짐. 이때는 Latent Variable이 있음 
      
      > Tasking : Query를 날려서 그에 대한 그럴듯한 답안을 얻는 것. 
      > 
      > '그럴 듯한' 의 기준을 세우기 위해 확률 모델을 가정해야 하며, 이때 확률 모델의 parameter(latent variable) 을 구해야 한다.  
  
  - **즉, Bayesian Network을 설정하고 나서, 사용 형태와 시점, 용도에 따라서 Latent Variable이 생겼다 없어졌다 한다.**

<br>

- 앞에서  우리가 원하는 값인 Observed data $x_n$ 에 대해 k번째 class로 분류될 확률 인 $P(z_k|x_n, \theta)$을 구할 수 있었다.   
  
  - 해당 값은 $\frac{P(z_k=1|\theta) P(x_n|z_k=1, \theta)}{\sum^K_{j=1} P(z_j=1|\theta)p(x_n|z_j=1, \theta)} = \frac{P(z_j=1, x_n|\theta)}{\sum_{j=1}^KP(z_j=1, x_n|\theta)}$ 을 통해서 구할 수 있었다. 
  
  - 하지만 $P(z,x|\theta)$ 값을 아예 구해낼 수 없는 경우에는 어떻게 해야할까? 
    
    > ex)- $P(x|\theta)$ 가 우리가 익히 아는 정규 분포, 이산 분포 등을 따르지 않는다



- Evidence Lower bound를 통해 계산을 해준다. 자세한 것은 아래에 다시 설명. 
  
  > $P(X|\theta) = \sum_ZP(X,Z|\theta)$
  > 
  > -> $lnP(X|\theta) = ln\{ \sum_ZP(X,Z|\theta)\} = ln\{ \sum_Z \frac{P(X,Z|\theta)}{q(Z)} q(Z) \}$
  > 
  > > $q(Z) >= 0 $ for $\forall Z$. 임의의 함수임.  
  
  > 이때 log를 $\sum$ 안에 넣어줄 수 있다면 미분해주기가 매우 용이해짐 
  > 
  > By Jensen's inequality,
  > 
  > $ln\{ \sum_Z \frac{P(X,Z|\theta)}{q(Z)} q(Z) \} >= \sum_Z q(Z) ln \frac{P(X,Z|\theta)}{q(Z)}$
  > 
  >                                                 $= \sum_Zq(Z) lnP(X,Z|\theta) - q(Z) lnq(Z)$



- 또한 $P(X|\theta)$ 를 최적화 하는 것은 Latent variable Z가 있을 때,  $\sum_Z P(X,Z|\theta)$ 를 최적화하는 것과 동일하다.  이는 **편미분을 통해 최적값으로 업데이트 하는 것이 아닌, EM 또는 다른 번거로운 방법을 써야만 하는 이유**이다.
  
  > Latent variable이 존재해야한 하는 상황은 언제인가? 조건은 무엇인가? 
  
  > *뭔가 하나 부족한 느낌*
  > 
  > Latent Varible은 파라미터에 따라서 변화하며, 파라미터는 E에 따라 계속 변하기 때문?
  
  - 더 나아가, Latent Variable이 존재함에도 불구하고 EM, Evidence Low bound를 사용해야 하는데 사용 안했다면 잘못 접근한 것이다. 



- 또한 likehood를 알기 위해 Log를 사용하면 미분이 잘 적용되지 않는다. 
  
  > $P(X|\theta) = \sum_Z P(X, Z|\theta)$
  > 
  > $→ ln P(X|\theta) = ln(\sum_Z P(X,Z|\theta))$
  > 
  > - ln를 미분하면 식이 분모에 그대로 간다. 
  > 
  > - 즉, n번 미분하는데 계산이 많이 필요하다.

-----

**Probability Decomposition** - ELBO

- **'Log를 씌었을 때 미분이 잘 안된다'를 해소하기 위해** q(z)와 Jensen's equality를 적용 
  
  > Likelihood($\theta$) = $ln P(X|\theta) = ln(\sum_Z P(X,Z|\theta) $= <mark>$ln(\sum_Z q(z) \frac{P(X,Z|\theta)}{q(z)})$</mark>
  > 
  > > [전제] q(z)는 0 이상이여야 한다. 0에 가까워지면 GAN에서 오류가 발생
  > 
  > ![](picture/2-7.png) [Jensen's Equality. log를 식 안으로 넣기]
  > 
  > $         = ln(\sum_Z q(z) \frac{P(X,Z|\theta)}{q(z)}) >= \sum_Z q(Z) ln \frac {P(X,Z|\theta)}{q(Z)} $
  > 
  > $= \sum_Z q(Z) ln P(X,Z|\theta) - q(Z) lnq(Z)$
  > 
  > $= E_{q(Z)} ln P(X, Z|\theta) + H(q)$ 
  > 
  > > H(q) = $-\sum_Z P(Z) ln q(Z)$. Cross Entropy와 동일한 형태를 띈다. 
  > 
  > - $q(Z)$ 는 우리가 아는 분포이기 때문에 Sampling을 통한 근사값 계산이 가능하다. 
  
  
  
  > $Q(\theta, q) = E_{q(Z)} ln P(X,Z|\theta) + H(q)$
  > 
  > - $Q(\theta, q)$ 는 모든 q 분포에 성립하며, 이러한 분포를 우린 Posteriority(사후 확률)이라 한다. 
  > 
  > - 사후 확률은 Basian Network를 통해서 사전 확률로 바꿀 수 있다.

<br>

- 다른 방식으로도 표현할 수 있다. 
  
  > $P(X,Z|\theta) = P(Z|X, \theta)P(X|\theta)$
  
  > ![](picture/2-8.png)
  > 
  > $L(\theta, q) = lnP(X|\theta) - \sum_Z(q(Z) ln\frac{q(Z)}{p(Z|X, \theta) } )$
  > 
  > > 이때 두번째 항은 KL Divergence와 동일하다. 
  > > 
  > > $KL(q(Z)||P(Z|X,\theta)) = \sum_Z(q(Z) ln \frac{q(Z)}{P(Z|X,\theta)} )$

------

##### **KL Divergence**

- KL Divergence는 두 분포간의 차이를 측정한다. [Aymmetric]
  
  > $KL(P||Q) >= 0 $
  > 
  > P와 Q가 완전히 동일할 때만 KL(P||Q) =0 이 된다. 

- KL Divergence는 크게 2가지 행동 패턴이 있다. 
  
  1. Covering Behavior : 모두를 포괄하는 일반화된 1개의 분포 찾기
     
     ![](picture/2-9.png)
  
  2. Capturing Behavior : 각각 가장 적합한 분포로 매칭 
     
     ![](picture/2-10.png)
  
  > 둘다 상황에 따라 필요로 하는게 다르다. 

<br>

**Maximizing the Lower Bound** 

- $lnp(X|\theta) = L(\theta, q) + KL-term$ 으로, Lower bound $L(\theta, q)$ 를 최대화(<-> KL term을 최소화)함으로써 $lnp(X|\theta)$ 의 추론 정확도를 올릴 수 있다.  

![](./picture/2-12.png)

- Lower bound에 제약을 추가함으로써 Q의 값을 최적화한다.  
  
  > $L(\theta, q) = lnP(X|\theta) - \sum_Z(q(Z) ln\frac{q(Z)}{p(Z|X, \theta) } )$
  > 
  > 이 때 Loss 함수의 첫번째 항은 t 시간일 때 $\theta$가 고정됨에 따라 Fix 된다. 
  > 
  > 두번째 항(KL-Divegence)은 q(z)를 설정함에 따라 변동 가능하다. 
  > 
  > - "$KL(q(Z)| P(Z|X,\theta)) = 0 $" 을 만족할 때 Loss는 $lnP(X|\theta)$ 와 같다. 
  > 
  > - 따라서 <mark>$q^t(Z) = P(Z|X, \theta^t)$ 로 정의</mark>한다. 
  >   
  >   - 이때 $q^t(Z)$ 은 X와 $\theta^t$가 주어졌을 때의 Z가 나올 확률임. 
  >   - 즉,  해당 Z는 E-Step에서 계산할 값으로, <u>Z 값을 Sudo-Known Variable로 여길 있게 됨 .</u> 
  
  > 시간이 t 일때, Q($\theta^t, q^t$) = $E_{q^t(Z)} lnP(X,Z| \theta^t) + H(q^t)$ 이다. 
  > 
  > 앞서 $q^t(Z)$ 의 정의와 E-Step을 통해 Z 값을 구했다. 
  > 
  > 따라서 파라미터 $\theta$를 최적화 하는 것은 아래와 같다. 
  > 
  > - <mark>$\theta^{t+1} = argmax_\theta Q(\theta, q^T) = argmax_\theta E_{q^t(Z)} lnP(X,Z|\theta^t)$</mark>
  >   
  >   > 위의 식에서 X, Z를 모두 알고 있으니 $\theta$ 를 갱신할 준비가 되었다.
  >   > 
  >   > 주어진 X,Z 를 통해 $\theta$ 를 구하는 것은 Supervised learning 과 동일하다.
  
   

--------------

##### EM 과정의 시각화

![](picture/2-11.png)

- E- Step 
  
  - Z를 $P(Z|X,\theta)$ 로 정의하여 계산할 수 있도록 한다. 이는 Z를 semi-known variable로 만들어 추후 $\theta$ 를 계산할 여지를 준다. 
  - p의 근사 분포인 q를 최적화함으로써 Lower bound를 최대화 시킨다. 

- M-Step 
  
  - q가 변화했기 때문에, 이에 맞춰서 걸맞는 Parameter $\theta$를 할당해준다. 
  
  - 이때 우리의 목표는 $lnP(X|\theta^{t+1})$ 을 키워주는 것으로, 변화한 q에 맞춰 더욱 크게 설정하게 된다. 
  
  - 이때 생겨나는 $lnP(X|\theta)$의 차이를 Variation Gap이라고 부른다. 
