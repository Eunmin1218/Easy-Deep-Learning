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

---
## Thm4. git branch  
코드를 짜다가 기능을 추가하고 싶을 때   
원본에 수정해도 되지만  
프로그램이 망자길 수 있다는 우려  
따라서 파일 복사본을 만들어서 먼저 코드를 짜봄  
**git의 branch 기능을 이용하면 복사본 만들기 쉬움**  
**branch**: commit의 복사본  

> `git branch 브랜치명`  
> : 브랜치명으로 브랜치를 생성해줌-> 사본이 만들어짐  

> `git switch 브랜치명`  
> : 브랜치로 이동  
  
만약 그 브랜치 코드를 main브랜치에 합치고 싶다?  
우선 main 브랜치로 이동  
> `git merge 합칠branch명` 입력  

이 때, 같은 파일의 같은 줄을 수정했을 경우 **충돌(conflict)** 이 일어날 수 있음  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1st. 원하는 코드만 남기고  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2nd. git add  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3rd. git commit  

---
## Thm5. 다양한 git merge 방법(3-way, fast-forward, squash, rebase)  
### TYPE1) 3-way-merge  
브랜치에 각각 신규 commit이 1회 이상 있는 경우  
두 브랜치의 코드를 합침  
- merge 사용
  
### TYPE2) fast-forward merge  
새로운 브랜치에만 commit이 있는 경우  
신규 브랜치를 main 브랜치로 지정  
- merge 사용 시 자동으로 발동  
  
### TYPE3) rebase and merge  
**rebase**: 브랜치의 시작점을 다른 commit로 옮겨주는 행위  
왜 사용하는가?  
commit 내역을 한 줄로 이어서 남길 수 있기 때문  
- 새로운 브랜치로 이동하여 `git rebase main` 사용

> 1st. `git switch 새로운브랜치`  
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`git rebase main`  
> 2nd. `git switch main`  
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`git merge 새로운브랜치`  

> `git branch -d 브랜치명`  
> : merge 완료된 브랜치 삭제  
> `git branch -D 브랜치명`  
> : merge 안한 브랜치 삭제  

---
## Thm6. 다양한 git merge 방법(3-way, fast-forward, squash, rebase)  
