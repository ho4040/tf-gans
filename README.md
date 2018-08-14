# 개요

![ss](https://i.imgur.com/IQZXAnj.png)

모두연풀립스쿨에서 실습한 코드입니다.

기본 MNIST 와 Tensorflow 를 기준으로 작성되어 있습니다. 


# 구성

* original-gan.py : ffnn으로 구성된 기본 GAN

* deep-conv-gan.py : convolutio, transpose convolution 레이어를 활용하여 성능을 개선한 GAN

* conditional-gan.py : z vector 에 컨디션을 추가한 GAN

* ali.py : z vector를 inference 하는 네트워크가 추가된 GAN


# 기타

각 파일들은 따로 의존성이 없습니다.

소스코드 상단에는 결과이미지를 링크해뒀습니다.
