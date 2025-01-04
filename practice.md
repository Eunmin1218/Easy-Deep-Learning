## 1-4. 강화  학습  
보상을 함으로써 어떤 행동을 하게끔 강화  
> 행동 → 보상  

```yaml
Agent: 행동을 취하는 주체 (예: 강아지, 흑돌)
Action: Agent가 취할 수 있는  모든 행동 (예: 손, 뒤집기, 수)
Reward: Agent가 Action에 따라 받게 되는 보상 (예: 간식, 승점)
Envirionment: Reward를 언제, 얼마만큼 줄지 설계된 환경 (예: 주인, 심판, 백돌)
State: 현재 상태 (예: 위치)
Q-function: 특정 State에서 특정 Action을 했을 때 얻을 수 있는 Reward 합의 기댓값(Q(state, action))
Episode: 각각의 시행
Q-learning: 
exploration:
discount factor: 
```

  
Q&A  
Q. 1주년 예시에서 environment는 그럼 여자친구인가?  
Q. Q-fuction에서 지금까지 이동하며 얻은 보상은 포함되는가?  
A. 아니다.
Q. 1주년 강화학습에서 그럼 인공지능은 이전 데이터들을 참고하여 이동할 방향을 찾는 것인가? 아니면 그냥 랜덤인가?
