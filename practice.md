## 6-2. 로지스틱 회귀  
### sigmoid를 이용한 이진 분류  
- 총 parameter 개수 : gardient의 길이  
- **`Loss`** : 인공신경망의 출력의 의미를 정함
  #### →출력이 잘 나오도록 `Loss`를 잘 정하는 것이 중요하다!!

> ### BCE (Binary Cross-Entropy)
: 이진 분류에 적합한 Loss 함수 구하는 법  
  
예를 들어,  강아지 사진일 경우 **`q`** 를 1에 가깝게 **최대화**,  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;고양이 사진일 경우 **`1-q`** 를 **최대화**  
  
만약 batch-size가 2 이상일 경우  

$$
P(y_1 ∩ y_2) = P(y_1)P(y_2)
$$

그러나 곱하면 계속 작아지므로 

$$
BCE에서 Loss 값 = -\frac{1}{N} \sum_{n=1}^{N} \log ( {q}_n^{y_n}{1-q}_n^{1-y_n} )
$$

즉 **분류 ⊂ 회귀**
> ### Logistic Regression
: 입력과 출력 사이의 관계를 확률 함수로 ㅍ현하고 이 함수를 은

Q&A  
Q. 
