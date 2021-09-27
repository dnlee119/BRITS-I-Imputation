# BRITS-I을 이용한 시계열 데이터 Imputation

[원본코드](https://github.com/Doheon/TimeSeriesImputation-BRITS) <br/>
[논문](https://doheon.github.io/%EB%85%BC%EB%AC%B8%EB%B2%88%EC%97%AD/time-series/pt-brits-post/)

원본의 내용을 토대로 작성된 커스텀된 형태의 코드입니다.
<br/>불필요한 부분을 제거하였고 최신버전에 맞게 최대한 작성했습니다.
<br/>아직 많이 부족한 부분이 있지만 노력해보려합니다!
<br/>최대한 많은 데이터에 대하여 공통적으로 잘 작동하도록 해보려 합니다.


### <사용방법> [현재 에디터에서만 시험해봄]
! 전제조건 : 데이터는 단일 독립변수여야한다, 시계열 데이터야한다.
1. 'put_csv_in_here' 폴더에 사용할 csv 파일을 넣음
<br/>(1개의 파일만 가능)
2. main을 실행
   <br/>(데이터의 형태에 따라 학습이 불가능 할 수 있음)
3. 필요에 따라 main파일의 epoch와 learning_rate를 수정하여 사용할 수 있음