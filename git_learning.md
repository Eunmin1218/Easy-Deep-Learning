Q. 2일 전 코드로 되돌아가려면?   
1. 파일로 매번 저장
2. 버전관리 소프트웨어 사용
  
git을 사용하면 작업한 코드들을 기록하고 보관 가능  
-> 과거 작업내용을 열람하거나 되돌아갈 수 있음  

> `git config --global user.email "홍길동@naver.com"`  
> `git config --global user.name "홍길동"`  
> : 현재 깃을 쓰고 있는 사람 명시

---
## Thm2. git add, commit  
  
> `git init`  
> : git이 코드 짜는 것, 파일 만드는 것 감시  

> `git add 파일명`: 기록할 파일 고르기  
> `git commit -m '메모'`: 고른파일 기록명령  
> : 파일의 현재 상태 기록  
  
`작업폴더` -git add- `staging area` - git commit - `repository(저장소)`  

> `git add app.txt app2.txt`: 파일 같이 스테이징    
> `git add .`: 모든 파일 스테이징  

> `git status`   
> : 어떤 파일들이 스테이징 됐는지, 어떤 파일이 수정 됐는지 등 상태를 알려줌  

> `git log --all --oneline`  
> : 커밋한 내역을 보여줌
> 노란색 글자는 **commit id**

---
## Thm3. git add, commit, diff  
vs코드에서 git 기능 제공  
파일 수정 후 저장 시, 수정됐다는 내역이 뜸  
+버튼: git add 버튼  
-버튼: git add 취소  
체크 표시: git commit  

> `git diff`  
> `git difftool` : :qa 입력하면 종료  
> : 최근 commit과 현재 파일의 차이점을 보여줌
응용: `git difftool 커밋id_1 커밋id_2`  
그러나..  
**Git graph** : 위의 것들을 더 편리하게 할 수 있음
