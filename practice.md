## 5-1. Linear activation  
결국 MLP는..  
**행렬** 곱하고 **벡터** 더하고 **activation**의 반복  
그리고 인공 신경망은 **함수**  
  
예시: 

$$
\begin{bmatrix} x_{1} \\ x_{2} \end{bmatrix}
\begin{bmatrix} 
w_{1} & w_{3} & w_{5} \\ 
w_{2} & w_{4} & w_{6} 
\end{bmatrix}=
\begin{bmatrix} b_{1} \\ b_{2} \\ b_{3} \end{bmatrix}
$$

각각을 x, W₁, b₁이라고 하면  
**f₁( xW₁ + b₁ )**  
두 번째 layer을 거친다면  
**f₂( f₁( xW₁ + b₁ )W₂ + b₂ )**  

그런데 만약 activaion들이 모두 linear activation이라면 깊어져도 복잡한 함수를 구현할 수 없다  
왜?  
위의 식에서 두 함수가 모두 linear activation이라면 들어온 대로 나가므로  
**( xW₁ + b₁ )W₂ + b₂**


  
Q&A  
Q. 1주년 예시에서 environment는 그럼 여자친구인가?  
Q. Q-fuction에서 지금까지 이동하며 얻은 보상은 포함되는가?  
A. 아니다.
Q. 1주년 강화학습에서 그럼 인공지능은 이전 데이터들을 참고하여 이동할 방향을 찾는 것인가? 아니면 그냥 랜덤인가?
