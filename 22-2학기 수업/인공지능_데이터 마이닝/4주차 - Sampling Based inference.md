#### 4주차 - Sampling Based inference

###### 서론

- 모델 추론에 있어서 대부분의 방법은 계산량 때문에 Infeasible 하다 
  
  > Why? 계산량이 얼마나 많길래? 

- 그나마 ELBO method가 Feasible 하나 단점이 있다. 
  
  1. 직접적으로 Optimizing을 못한다.
  
  2. Jensen's equaility & Variatinal Error(?) 로 Optimal Solution과 차이가 생긴다
  
  3. <mark>Local Optima에서 못 빠져나온다.(- 모든 Gradient based method에게 해당)</mark>

**→ Gradient-based 외의 방법이 필요하다.**

- Sampling-method은 이 문제를 해결할 수 있다. 



--------

##### Sampling-based 방법의 종류

- Forward sampling : 모든 경우가 나올 때까지 경우의 수를 확인한다. 느려서 안쓴다. 

- Rejection Sampling : 조건을 충족하지 않는다 싶으면 계산 중간에 그만둔다.
  
  > Rejection Sampling 은 요즘에 잘 사용되고 있지 않음. 역사적인 관점에서 다룸 
  
  ![](picture/4-1.png)
  
  > p(x) : Rejection Sampling 을 적용했을 때의 분포 
  > 
  > q(x) : 임의의 확률분포. 충분히 큰 값 M을 곱하여 p(x)를 전부 포괄하도록 한다. 
  > 
  > - M이 작다면 중간에 p(x)와 Mq(x)가 겹치는 부분 이후는 고려하지 못한다. 
  > 
  > - 왜냐하면, q를 nomal distribution으로 고려하며, 조건 식을 $u < \frac{p(x_{(i)})}{Mq(x_{(i)})}$ 로 두었기에 때문에, M이 충분히 크지 않다면 고려해야 하는 상황인데도 잘라버릴 수 있다. 
  >   
  >   ![](picture/4-2.png)[좌 : p(x), 우 : p(x)를 덮지 못한 m(qx)] 
  
  - 단점 : P(x)의 값을 모르기 때문에 일단 M을 크게 잡아야 한다. 하지만 그러면 쓰지 못하고 버려지는 Sampling이 많이 발생하게 되어 효율이 떨어진다. 



------------

#### Importance Sampling

- VAE 등과 같이 현재에도 쓰이는 방법 

- <u>Rejection 방법은 많은 양의 데이터를 버려 비효율성이 커진다는 문제가 있다.</u> 
  
  - 즉, 데이터를 버리지 않으면서 각각의 중요도를 고려할 수 있는 방법이 필요 
  
  - Importance Sampling은 인스턴스를 버리지 않기 떄문에 최대한의 (자료의) 효율성을 모델링하고자 한다. 



- 과거에는 <u>확률의 분포(PDF)을 알아내어 데이터의 분포를 확인하는 것</u>을 목표로 했다. 
  
  - 그러나 최종적으로 우리의 목적은 Query에 대해서 Most Probable Answer을 하는 것을 목표로 하며, 따라서 중간 단계로 **Expectation을 구하는 것을 목표로 한다.** 
  
  - 즉, PDF를 굳이 만들기 위해 샘플링을 많이 하지 말고, **"샘플링 한 것을 버리지 않고서 Expectation을 구해볼 수는 없을까"** 가 해결하고자 하는 문제이다. 
  
  -> Expectation을 확장하여 고민해보자! 
  
  > $E_p(f(z)) = \int f(z)p(z)dz $
  > 
  >                   $= \int f(z) \frac{p(z)}{q(z)} q(z) dz$   (= $E_q (f(z)\frac{p(z)}{q(z)})$)<mark>[like ratio trick]</mark>
  > 
  >                   $\sim \frac{1}{L} \sum_{l=1}^L \frac{p(z^l)}{q(z^l)}f(z^l)$ 
  
  > > L : num of samples 
  > > 
  > > $z^l$ : sample of Z 
  
  - 이때 <u>q-distribution은 P의 Long Tail에서도 절대 0이 되면 안된다는 강력한 전제를 필요로 한다.</u> 
  
  - 그렇기 때문에 <u>Like ratio trick은 0이 될 가능성을 가지고 있기 때문에 위험성을 내포하게 된다. </u>
  
  > let $r^l = \frac{p(z^l)}{q(z^l)}$ 
  > 
  > - 이때 $r^l$ 은 계산 가능하다! $q$는 우리가 임의로 정해주는 Distribution이니 구할 수 있다. 
  > 
  > - P(z)의 값 또한 "like ratio trick" 을 통해 q-sampling으로 변환시 계산할 수 있다. 
  
  > $P(z>1) = \int^\infin_1 1_{z>1} p(z) dz$
  > 
  >                    $= \int^\infin_1 1_{z>1} \frac{p(z)}{q(z)} q(z) dz$
  > 
  >                    $\sim \frac{1}{L} \sum^L_{l=1} \frac{p(z^l)}{q(z^l)} 1_{z^l>1}$
  > 
  > - P는 모르니 Sampling 할 수 없다. 그러니 알고 있는 Q에 대해서 Sampling 하여 $r^l$ 을 계산하겠다. 
  > 
  > -> <mark>$ r^l = \frac{p(z^l)}{q(z^l)}$ 을 구할 수 있다.</mark>
  > 
  > -> <mark>즉, $E_p(f(z))$ 은 p(z)을 모름에도 구할 수 있게 된다. </mark>



- 이를 통해서 **q-분포를 통해서 나온 모든 값을 버리지 않고 반영한다.** 
  
  - 단, 매우 작은 weight를 부여! 
  
  - => IID 조건에서는 가장 효율적으로 정보를 사용하는 것일 것. 



- 값이 이산인 경우에는 Likely load trick으로도 부르기도 한다. 
  
  - 각 Likelyhood를 가중치로 부여하여 계산한다. 



----- 

##### Markov Chain

- Q-sampling : i.i.d. 조건에서 Efficiency를 최대로 올린 것 

- 더 정보를 효율적으로 쓸 수 없을까? [문제]
  
  - => 과거와 현재가 연관성이 없다고 가정하는 것은 현실과 다르다. (=IID 조건은 현실과 맞지 않다)

- 과거와 현재의 연관성을 받아들인 방법을 채택하자! - Markov Chain




