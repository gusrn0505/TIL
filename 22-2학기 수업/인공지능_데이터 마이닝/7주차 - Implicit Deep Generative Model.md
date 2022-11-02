### 7주차 - Implicit Deep Generative Model

##### Implicit Density Modeling

![](./picture/7-1.png)

- 앞서서 Deep Generative Model 중 Explicit Approximate Density에 해당하는 Variational Autoencoder에 대해서 배웠다. 

- 이젠 Implisit Density Modeling 중 Direct Sampling에 해당하는 GAN에 대해서 배울 것이다. 
  
  - 앞서 Explicit Model 에서는 PGM(probability graph model)이 가능했으나, Implicit Model에서는 불가능하다. 

<br>

- 기존 Variational Inference 에선 Conjugacy 관계와 tractable Likelihood를 필요로 한다. 
  
  > Conjugate 관계 예시 ) Drichlet ↔ Multinomial distribution 
  > 
  > Tractable Likelihood는 Explicit density를 의미하나? 
  
  - VAE에서 Variational distribution에 대한 Inference Network를 형성하여 Conjugate 관계를 더 이상 필요로 하지 않는다. 
  - 하지만 여전히 Explicit likelihood function을 필요로 한다.

- Implicit Likelihood Model을 고려할 수 있으면 Explicit model에 비해 1) 표현력이 좋아질 수 있으며, 2) P 분포가 implicit 한 경우에도 적용할 수 있다. 
  
  <mark>-> Learning in Implicit Model 을 연구하자! </mark>

-----

#### Generative Adversarial Network(GAN)

![](./picture/7-4.png)

- Hidden Variable Z에 Generator을 통해서 이미지를 재 생산하자! 그리고 진짜 이미지와 생성한 이미지를 구별하기 위해 Discriminator을 적용한다! 

- 서로 Adversarial한 학습 관계를 가지고 있는 Generator과 Discriminator을 연결하여 서로 성능이 향상되도록 만든다! 

- 이때 구별해야 할 것이 있다. 
  
  - 생성한 이미지들은 Generator을 통해서 Sampling 되는 것처럼 보이지만, 정확히는 Z가 Random Variable이 이기 때문에 가능한 것이다. 
  
  - Generator은 Neural Network로 Z를 이미지로 만드는 것 밖에 하지 못한다. 
    
    - 단, Neural Network는 매우 Complex 하면서도 Flexible 한 response curve를 가지게 되었다. [활용도가 넓으면서 정확도가 높다로 이해]
  
  - 단, Z가 Random Variable이기 때문에 확률 분포를 지니며, 이로 인해 Sampling 되는 것이다. 

<br>

- **Notion & Modeling**
  
  > $p_z(z)$ : Noise variable의 Prior distribution
  > 
  > - Uniform, Normal distribution 등 다 가능함
  
  > $p_{data}(x)$ : x에 대한 data distribution. 데이터셋이 주어졌을 때 바로 정해짐
  
  > $p_g(x)$ : $z \sim p_z$ 일 때 G(z) 의 샘플링 분포 
  > 
  > - 구체화할 수 없다! 하지만 Sampling이 가능하다! (How?)
  > 
  > - Gan 모델에서 복원한 값(ex- Image)들을 얻을 수 있지만, 이 이미지들이 어디로부터 나왔는지(x)는 알 수가 없다. 즉, 접근 권한이 Sampling 결과에만 있는 것! 
  > 
  > - pdf를 알고 있을 때와 반대의 상황. sampling은 pdf을 알고 있는 것의 종속되는 게 아니다..! 
  
  > $G(z; \theta_g)$ : Generator 
  > 
  > - 파라미터 $\theta_g$ 를 가지며 미분 가능한 함수로 표현된  Neural Network이다. 
  > 
  > - Noise variable을 Data space X로 매핑한다.
  
  > $D(x; \theta_d)$ : Discriminator 
  > 
  > - x가 Generator로 부터 왔을 확률 값을 계산

###### Formalization of GAN - Parameter inference

- Training 예시에 대해 옳은 Label(진짜 이미지인지, 생성한 이미지인지)을 부여하는 확률 값을 Maximize 하는 것 
  
  - 먼저 데이터셋만 활용하여 Discriminator을 학습시키자! 
  
  - 학습할 때 데이터 셋에서 온 x에 대해 $D(x) = 1$ 의 값을 부여하여 학습시킨다. 
  
  > Maximize $E_{x \sim p_{data}(x)} [logD(x)]$ w.r.t. D 

- **Objective Function** 
  
  - 생성한 것인가 아닌가하는 Binary case 이므로 Bernoulli trial로 여겨진다. 또한 Cross entropy 로 목적 함수를 정한다.
    
    > ![](./picture/7-5.png)
    > 
    > Noise variable z에 대해서 고려할 때
    > 
    > - 이 때 $p_{data}$ 는 이미 정해진 분포라 따로 고려할 점이 없다.
    > 
    > - 두번째 항 $E_{z\sim p(z)} [log(1-D_\phi(G_\theta(z))]$ 을 어떻게 할 것인가에 대해 고려해주면 된다.
  
  - 목적 함수를 두 단계로 고려한다.
    
    - 먼저 Discriminator 관점에서 데이터 셋에서 온 x($x \sim p_{data}$)에 대해 높은 확률 값($D_\phi(x)$)을 부여하고, 생성한 $G_\theta(z)$ ( $z \sim p(z)$)에 대해선 낮은 확률 값(1- $D_\phi(G_\theta(z))$)을 부여해라. 이를 극대화하자 
    
    - 반대로 Generator 관점에선 잘 속여야 한다. $E_{z \sim p(z)}$을 최대한 떨어트리자! 
  
  - 이때, Generator와 Discriminator은 각각 학습 하는 방향이 다르다. 따라서 제대로 학습이 안될 때가 많다. 
    
    > ![](./picture/7-6.png)
    > 
    > 매우 이상적인 상황에서만 이렇게 된다. 

<br>

###### Objective function of GAN의 이상적인 해답 찾기

![](./picture/7-7.png)

> $z \sim p_z(z) -> x= G(z) -> x \sim P_g(x)$ 

> $D(x) = \frac{P_{data}(x)}{P_{data}(x) + P_g(x)}$ 
> 
> - 이 식이 어떻게 나왔지? 

- 두 개의 Expectation을 Jensen-Shannon-Divergence로 표현한다.
  
  >  ![](./picture/7-8.png)

- JS을 통해서 Explicit 하게 구할 수 없었던 $P_g(x)$ 의 Bounded와 최적 값을 구할 수 있게 되었다. 

-----

##### Jensen Shanon Divergence & Training 방식

- 특징 
  
  - KL-divergence는 Lower bound( - 0 이상) 만 있는데 반해, JS-divergence는 양쪽으로 bound 되어 있다. 
    
    **=> 더욱 안정적이다.**
    
    > 0 <= JS(P||Q) <= ln2
    > 
    > - V(D,G)가 가질 수 있는 값의 범위를 생각해보면 얼추 맞음.
  
  - JS(P||Q)는 Symmetric 하다.
  
  - JS(P||Q) =0 이라면 P=Q와 동치이다.

- **정보 이론과 밀접한 관계를 맺고 있다.**
  
  - 상보 정보양(Mutual Information) I(X;Z)을 고려해보자. 
    
    - <mark>I(X;Z)는 X와 Z를 같이 고려했을 때 알게되는 정보를 의미한다.</mark> 
    
    > ![](./picture/7-10.png)
    > 
    > $I(X;Z) = H(X) + H(Z) - H(X,Z)$ 
    > 
    > $H(X) = \sum P logP $. H(Z), H(X,Z)도 동일하게 구할 수 있음. 
    > 
    > > X : Abstract function on the events. GAN에선 Data set + 생성된 Data
    > > 
    > > M : Mixture distribution. $X \sim M = \frac {P+Q}{2}$ 라고 가정. 
    > > 
    > > Z : mode selection between P and Q. 
    > > 
    > > - Gan에선 P와 Q는 각각 데이터셋에서 유래할 확률, 생성한 이미지에서 유래할 확률을 의미한다. Z는 Discriminator 느낌 Z=0 
    > > 
    > > > Z=0 라면, X가 P로 부터 왔음을 의미 
    > > > 
    > > > Z=1 라면, X가 Q로 부터 왔음을 의미 
    
    > ![](./picture/7-11.png)

- 즉, Jensus's Divergence는 X와 Z 사이의 Mutual information과 동일하다. 
  
  - Jensus's Divergence의 값이 0 이라면, X와 Z 사이에 상호 정보는 없는 것과 동일한 의미를 가진다. [예측함에 있어서 Z는 어떠한 영향도 미칠 수 없다]
  
  - Jensus's Divergence 값에 따라 Z가 X의 값을 유추함에 있어서 얼마나 많은 정보를 제공하는지 계산할 수 있다. 

<br>

- 다시 Gan으로 돌아와서 우리가 고려하는 분포 $P_{data}(x), P_g(x)$로 생각해보자. 
  
  - 우린 Objective function V(D,G)를 최적화시킬 것이며 이 값이 JS 와 같음을 보였다
  
  - 또한 JS은 Mutual Information으로서 한 Variable이 다른 Variable에 대해 제공하는 정보를 의미한다.
    
    - $P_{data}(x)$ 는 데이터셋이 주어질 때 고정된 값이다.
    
    - 그럼 우리가 조정할 수 있는 것은 $P_g(x)$ 뿐이다.
  
  - <mark>-> $JS(P_{data}(x); P_g(x))$ 를 최적화 시킬 때 $P_g(x)$ 에만 변화를 주게 된다.</mark>
    
    - <mark>한편으론 $P_g(x)$ 를 통해서 $P_{data}(x)$을 유추할 정보를 얻도록 한다는 것이다.</mark> 
    
    - 앞서 Implicit Likelihood인 <mark>$P_g(x)$ 가 $P_{data}(x)$을 최대한 유추할 정보를 주도록 최적화한다.</mark>

-> 구체화할 수 없었던 $P_g(x)$을 최적화할 수 있게 되었다.

---

###### Theoretical Results of GAN

- 연구 차원에서 증명해냈을 때 가치가 있는 것은 2가지가 있다.
  
  - 1). 이상적인 상황은 존재하는가? 그리고 어떤 상태인가?
  
  - 2). 어떻게 이상적인 상황으로 다가갈 수 있는가?

- 1). 이상적인 상황 고려하기 [증명 완료 by Jensen ]
  
  - 앞서서 V(D,G) = $JS(P_{data}(x); P_g(x)) - ln4$  임을 보였다. 
  
  - 즉, $min_G max_D V(D,G) = min_Gmax_D [JS(P_{data}(x) ; P_g(x) - ln 4]$ 로, 만약 D가 이미 Optimal로 고정되어 있다면 Generator는 V(D,G) = -$ln 4$ 일때 최적이다. 
    
    - 즉, $p_{data} = p_g$ 로 $JS(P_{data}(x); P_g(x)) =0$ 일 때를 의미한다. 
    
    - 한편으로 의미 측면에서 Generator가 데이터셋의 분포($p_{data}$) 동일한 분포($p_g$)를 만들 때 최적임을 알 수 있다. 

- 2). 어떻게 최적으로 다가갈 것인가? [Convergence Path 보장 못함]
  
  - (가정) Global optimum으로 다가가기 까지 공간이 있다면, D와 G가 순차적으로 최적값으로 수렴할 것이다. 
  
  - 한편으론 Nash 균형으로 인해 둘다 최적으로 갈 수 있다~ 라고 하는 데 뻥임. 
    
    <u>-> D와 G는 서로 다른 학습 방향을 가지고 있기 때문에 순차적으로 학습할 수 밖에 없으며, 최적으로 수렴을 보장을 해주지 못한다.</u>

----

###### GAN의 단점 - Mode Collapse

- Generator 관점에서, 최적값을 가지는 특정 값(ex- Local optima)에 빠져 그것만 계속 생산할 수 있다. 
  
  > $G(z) = x^* $ s.t. $x^* = argmax_xD(x)$
  > 
  > 이때 $x^*$ 은 $z \sim p_z(z)$의 샘플링과 관계없이 고정된 결과가 될 수 있다. 
  > 
  > ![](./picture/7-12.png)최적의 솔루션이 좌측 그림처럼 8개 점에 대해서 있다. 
  > 
  > ![](./picture/7-13.png)
  > 
  > 하지만 학습해보면 최적의 솔루션 중 일부만 계속해서 생산해내는 Mode Collapse가 발생한다. 

- 따라서, 추가 조치를 하여 새로운 결과값을 찾도록 자극을 줘야 한다. 
  
  - 이때 G와 D 중에서 조치를 취해야하는 것은 D이다. 왜냐하면 G는 따로 참고할 것이 없으나, D는 정보(생성된 데이터인지, 데이터셋에서 왔는지 여부)를 더 알고 있으니까  
  
  - 그럼 어떻게 D를 설계할 것인가. 

<br>

- 이상적인 경우를 고려하자. 
  
  > ![](./picture/7-14.png)
  
  - $\theta_D^*$ 의 값을 Gradient signal 값을 하나 하나 추가함에 따라 최종적으로 수렴한다고 가정한다. 
    
    > ![](./picture/7-15.png)
  
  - 하지만 실제론 $\theta_D^*$ 의 값을 구하기 위해 K을 무한대로 늘릴 순 없다. 따라서 적절한 K로 적용한다.(Surrogate 한다)
    
    > ![](./picture/7-16.png)

<br> 

![](./picture/7-17.png)

- 즉, 요점은 D에 대해서 바로 직전의 상황만 고려하는 것이 아니라, k 단계 이후의 경우의 수를 고려하는 것 [강화학습의 방식과 유사]
  
  - 이전 방식보다 좀 더 넓은 영역을 탐색할 수 있다. 

- Parameter을 update할 때, Generator에는 여러 Unrolled 상황을 고려하며 Discriminator에는 통상적으로 학습한다.
  
  > Q. Why? 왜 Generator에만 여러 경우를 고려하나? 
  
  > Generator을 편미분 했을 때, 
  > 
  > ![](./picture/7-18.png)[좌 : 정미분, 우 : 편미분]
  > 
  > k가 무한대에 가까워짐에 따라, $\theta_D^k = \theta_D^*$ 에 근사한다. 즉, 더 이상 변화가 없기 때문에 미분 값이 0에 수렴한다.
  > 
  > 반대로 K=0 이라면 일반적인 Gan과 동일해진다. 
  > 
  > k가 많을 수록 좋지만, Gradient sigmal은 Discounting 되며 점차 약해진다. 

- 이 방법은 D을 몇차례에 걸쳐 반복 학습하는 것과 다르다. D를 몇 차례 진행시켜 여러 버전을 만든 다음에 학습하는 것이다.

- 정리하면, Surrogate Function : $f_K(\theta_G, \theta_D) = f(\theta_G, \theta_D^K(\theta_G, \theta_D))$ 을 도입하여 기존 방식에 대한 이득을 discounting 시켜 탐색을 촉진시키며, 이를 통해 mode collapse를 해소한다.
  
  > ![](./picture/7-19.png)
  
  - 단, 이 방식은 Z가 알아서 다양한 값을 가지리라 기대하는 것으로, 강력히 다른 값을 가지도록 조치할 필요가 있다. 지금은 기존의 이득을 discounting 하는 것이 끝이다.
    
    - 예시로, 다수의 Discriminator을 적용하거나 Augmentation을 부여가 있겠다.

-----------

#### Variants of Generative Adversarial Network - GAN Model의 변주

###### Conditional GAN

- 크게 1) Condition 조건을 아는 Supervised setting, 2) 조건을 모르는 Unsupervised learning 두가지 경우가 있다. 
  
  > ![](./picture/7-20.png)
  > 
  > 이 경우는 y값이 주어진 Supervised learning setting! 
  > 
  > D와 G 모두 주어진 Condition y를 고려하자! 

- Supervised setting 에서는 G와 D에 Concatenate를 통해 추가해주면 된다. 끗. 
  
  > ![](./picture/7-21.png)
  > 
  > $NN_G(z,c;w_G) =x $
  > 
  > $NN_D(x,c;w_D) = p$ 

##### Info GAN

- Unsupervised setting 에서는 Latent variable을 추가하는 것을 고려한다.
  
  - 앞서 Mutual Information I(X;Z)을 통해서 Variable을 추가했을 때 얼마나 더 정보를 얻을 수 있는지 확인했다. 이를 활용한다. 
  
  > $min_G[max_DV(D,G) - \lambda I(c; G(z,c))]$ 
  > 
  > Latent variable을 C를 추가했을 때의 Mutual Information을 같이 고려하여 Generator을 최적화한다. 

- 이때 c는 직접 구할 수 없어 근사를 필요로 하는 Latent Variable이다. 
  
  - Mutual Infomation 식을 통해서 Lower bound를 찾아 Tight 하게 만들어주자 
  
  > ![](./picture/7-22.png)
  > 
  > 3번째에서 4번째로 내려갈 때 KL-term(>=0 ) 을 생략한 것. 
  > 
  > 우리의 관심은 c와 z의 관계이므로, H(C)는 고려안해도 된다. 
  > 
  > $P(c'|x) logP(c'|X)
  > $ 의 값은 우리가 계산할 수 없다. 따라서 임의의 Q 분포를 도입하여 계산가능한 형태로 바꿔준다.  
  > 
  > 부등식이 Tight 해지는 시점은 KL-term이 0이 되는 순간, 즉, $P(c'|x) = Q(c'|x)$ 일 때이다. 
  
  > 이로써 I(c; G(z,c)의 Lower bound을 찾았다. 이제 Objective function에 적용하자. 
  > 
  > ![](./picture/7-24.png)
  
  > 이때 Q는 주어진 x를 기반으로 c 값을 만들어 내는 임의의 분포를 의미한다.

<br>

- Info GAN은 Conditional GAN과 상당히 유사하다.
  
  > ![](./picture/7-25.png)
  > 
  > 이때 c가 x에 잘 반영되어 있다는 가정하에 적용한다
  
  - 단, Info GAN에서는 C가 Latent variable 이므로, Conditional GAN과 다르게 Concatenate 형태로 넣어줄 수가 없다. 
  
  - 결국, Q는 Latent variable c가 있다 치고, c를 맞춰야 한다. [= 시뮬레이션 과정]
    
    - C를 맞출 수 있도록 Q 또는 G을 학습시킨다. 

------------

### Modifying the Loss Characteristics - 과연 이 Loss가 옳을까?

- 지금까지의 모델들은 기능을 넓히자! 에 초점에 맞춰져 있었다. 

- 이제는 기존의 기능을 더 좋은 방법으로 하자!에 초점을 맞춘다.

- 앞서 GAN의 Objective function을 Jensen-Shannon divergence을 통해 표현했다. 
  
  > ![](./picture/7-26.png)
  
  - 하지만 Divergence는 Distance 조건을 충족시키지 못한다.
    
    > ![](./picture/7-27.png)
  
  - Distance는 일종의 함수이며, 함수는 벡터의 일종이다.
    
    > $R^\infin$ : Countably many. Sequential Array 
    > 
    > $R^R$ : Uncountably many. Dense. <u>실수 공간의 함수</u> 
    
    - 즉, 함수도 벡터의 일종이기에 Vector의 특성을 활용할 수 있다. 
    
    - 또한 함수가 아닌 Divergence는 Vecotr의 특성을 활용할 수 없다. 

<br>

- **벡터의 특성을 활용하기 위해 함수인 PDF을 통해 distance 형태로 Objective function을 표현해보자**
  
  - 앞서 Variational Inference 에서 Convex Duality를 다뤘던 것을 활용하자. 
  
  > ![](./picture/7-28.png)
  > 
  > $f^*(\lambda)$를 Dual function, 또는 Conjugate function이라 한다.

- Fenchel conjugate라고 알려진, Convex conjugate function을 활용한다.
  
  > ![](./picture/7-29.png)
  
  - Fenchel conjugate의 특성 
    
    - Fenchel's inequality : For all $a, x \in X,  <a,x>$  <= $f^*(a) +f(x)$
    
    - 순서 변경 : If f(x) <= g(x) for all x $\in X $, $g^*(a) <= f^*(a) $ for all $a \in X$
    
    - $f^*$ 또한 항상 convex하며, Lower semi-continuous 하다.
      
      > ![](./picture/7-30.png)
    
    - f(x)의 미분 값 a에 대해 아래 수식이 성립한다.
      
      > ![](./picture/7-31.png)
      > 
      > 첫번째 줄 : f(y) - f(x) >= <a,y-x> [Convexity of f]
      > 
      > 두번째 줄 : <a,y-x> = <a,y> - <a,x> 
      > 
      > 세번째 줄 : Fenchel conjugate 정의 $f^*(a) = sup[<a,x> - f(x)] $ 
      > 
      > 네번째 줄 : Fenchel's inequality 와 위의 식 함께 고려. 부등호 식이 반대임.  




