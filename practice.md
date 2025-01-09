## 8-6. 정규화(Regularization)와 MAP(Maximum A Posteriori)  
overfitting 방지..
> ### Regularization  
: loss에 weight의 크기를 더해서 같이 고려하려 함  

$$
loss = L + λ\lVert w \rVert^{p}_{p}
$$

<img src="https://github.com/user-attachments/assets/0082c827-eb79-40e4-a36d-cb6d31a05d2a" alt="description" style="width:40%; height:auto;">

> #### L2
> : 작은 애들은 살살 때리고, 큰 애들은 팍팍 때림
> -> **`중요한 commection들만 살리기 위해!`**  
> &nbsp;&nbsp;수학적으로 **MAP(Maximum A Posteriori)** 로 해석

  
Q. **weight를 줄이려고 하는 이유?**  
1. **`모델 경량화`**  
   loss를 줄이는 데 영향 없는 weight는 0으로
2. 어느정도 수렴하고 나서도 계속 학습시키면 weight가 자꾸 커짐  
   **L이 어느정도 줄었을 때 L이 도로 커지지 않는 선에서 도입**

<img src="https://github.com/user-attachments/assets/3a9127ff-9bb3-474d-a762-e5c78d16730b" alt="description" style="width:70%; height:auto;">  
loss = L + λ\lVert w \rVert^{p}_{p}
 **λ = 1 / σ²**
 즉, λ를 설정하여
 
 


Q&A  
Q. 엔트로피?  
A. 불확실성. 엔트로피가 높다는 건 정보가 많고 확률이 낮다는 것  
