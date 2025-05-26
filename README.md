## 📑 목차
- [🧾 프로젝트 개요](#-프로젝트-개요)
- [📆 개발 인원 및 기간](#-개발-인원-및-기간)
- [🛠️ 주요 기능](#️-주요-기능)
- [💻 사용 기술](#-사용-기술)
- [📷 결과 영상](#-결과-영상)
- [🏆 프로젝트 성과](#-프로젝트-성과)
- [💡 프로젝트 회고](#-프로젝트-회고)
- [🔖 관련 블로그 글](#-관련-블로그-글)
- [📚 참고 자료](#-참고-자료)

<br>

## 🧾 프로젝트 개요
- 이 프로젝트는 **Fluid-Implicit Particle(FLIP) 기법**을 활용한 실시간 **액체(Liquid) 시뮬레이션**입니다.  
- OpenGL을 사용하여 직접 시뮬레이션을 구현했으며, **파동 난류 표현**, **GPU 병렬 처리**, **GLSL 쉐이더 적용** 등을 통해 액체의 세밀한 시각적 특징을 표현하고, 성능상의 제약을 극복하고자 했습니다.  
- 본 프로젝트는 **게임 개발 및 실시간 그래픽스 분야**에서 활용되는 물리 기반 시뮬레이션 기술을 직접 구현하며 **이론부터 최적화까지 심층적으로 학습**하기 위해 진행되었습니다.

<br>

## 📆 개발 인원 및 기간
- 1인 개발
- 2024년 1월 ~ 2024년 12월 (약 12개월)

<br>

## 🛠️ 주요 기능
- FLIP 기법을 활용한 액체 시뮬레이션
- 파동 난류 표현
- CUDA를 활용한 GPU 병렬화
- GLSL을 활용한 Screen Space Fluid Rendering

<br>

## 💻 사용 기술
### 개발 언어

<img src="https://img.shields.io/badge/c-A8B9CC?style=for-the-badge&logo=c&logoColor=white"> <img src="https://img.shields.io/badge/c++-00599C?style=for-the-badge&logo=cplusplus&logoColor=white">

### 라이브러리
<img src="https://img.shields.io/badge/opengl-5586A4?style=for-the-badge&logo=opengl&logoColor=white"> <img src="https://img.shields.io/badge/GLSL-5586A4?style=for-the-badge&logo=opengl&logoColor=white"> <img src="https://img.shields.io/badge/nvidia%20cuda-76B900?style=for-the-badge&logo=nvidia&logoColor=white">

### 시뮬레이션 기법
- Fluid-Implicit Particle (FLIP)
- Surface Turbulence
- Screen-Space Fluid Rendering
- Narrow-Range Filter for Screen-Space Fluid Rendering

<br>

## 📷 결과 영상
<p align="center">
  <img src="https://github.com/user-attachments/assets/2f08841d-e189-416a-8f85-b1072ee5a048" width="45%">
  <img src="https://github.com/user-attachments/assets/0b2e2519-bb7e-4b6d-ab63-8fb9e0dbb24a" width="45%"><br>
  <img src="https://github.com/user-attachments/assets/18ef9050-6f62-4fe2-a54c-091e91f8391c" width="45%">
  <img src="https://github.com/user-attachments/assets/17057ed5-7f94-42b6-9cf6-49385a7a1aa6" width="45%">
</p>

[🔗 결과 영상 링크 (Youtube)](https://www.youtube.com/playlist?list=PLL7N-Nw3U-P3vHmnxfkImf6tOA4b5e9gY)

<br>

## 🏆 프로젝트 성과
- 본 프로젝트를 바탕으로 **2024 한국컴퓨터그래픽스학회 학술대회 KCGS 2024**에 **제1저자 및 발표자**로 참가하였습니다.
- **KCGS 2024에서 학부 우수논문상**을 수상하였습니다.
- 논문 제목: 유체-옷감의 상호작용에 의한 파동 난류, 확산, 주름을 표현하는 GPU 프레임워크
- [🔗 논문 링크(DBpia)](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE11866013)
- [🔗 수상 링크(학회)](http://cg-korea.org/index.php?hCode=AWARD_02_03)

<br>

- 본 프로젝트를 바탕으로 **한국컴퓨터그래픽스학회 논문지 제31권 제1호**에 제1저자 논문을 작성하였습니다.
- 논문 제목: 액체-옷감 상호작용에서 파동 난류, 확산 및 주름을 표현하기 위한 통합 GPU 프레임워크
- [🔗 논문 링크(DBpia)](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE11866013)
- [🔗 논문 링크(학회)](http://journal.cg-korea.org/archive/view_article?pid=jkcgs-31-1-25)

<br>

## 💡 프로젝트 회고
- 이 프로젝트를 통해 게임 개발에 종종 활용되는 **유체 시뮬레이션 기술의 개념과 구현 방식**을 직접 체험할 수 있었습니다.  
- 액체 시뮬레이션은 단순한 그래픽 효과를 넘어, **게임 내 물리 상호작용, 환경 반응, 연출 표현 등에서 몰입도를 높이는 핵심 요소**로 작용합니다.  
  특히 **Fluid-Implicit Particle(FLIP) 기법**을 적용하여 입자 기반 시뮬레이션의 물리적 정확성을 확보하고,  
  **파동 표현, 난류 효과, 충돌 처리** 등의 시각적 요소를 직접 구현하며 현실감 있는 액체 움직임을 표현하는 데 집중했습니다.  
- 추가된 시각적 표현으로 인해 발생한 **성능 저하 문제를 해결**하기 위해, **GPU 병렬화 및 GLSL 쉐이더 최적화**를 적용하여 효율성을 개선했습니다.  
- 이번 프로젝트를 통해 게임 클라이언트 개발자로서 **시뮬레이션부터 렌더링, 쉐이더 구현까지 그래픽스 파이프라인의 전체 흐름을 직접 설계 및 구현**해볼 수 있었고,  
  **CUDA 기반의 병렬 연산 처리 경험**을 통해 **게임 엔진 구조와 실시간 시각 효과 기술 간의 연계 방식**에 대한 실질적인 이해도를 높일 수 있었습니다.

<br>

## 🔖 관련 블로그 글
- [🔗 자세한 구현 및 학습 과정 정리 (Tistory)](https://coding-l7.tistory.com/category/%EB%AC%BC%EB%A6%AC%20%EA%B8%B0%EB%B0%98%20%EC%8B%9C%EB%AE%AC%EB%A0%88%EC%9D%B4%EC%85%98/Fluid%20Simulation)

<br>
  
## 📚 참고 자료

### FLIP
- [Animating sand as a fluid - Yongning Zhu, Robert Bridson](https://www.cs.ubc.ca/~rbridson/docs/zhu-siggraph05-sandfluid.pdf)
- https://matthias-research.github.io/pages/tenMinutePhysics/index.html
  
### Surface Turbulence
- [Surface turbulence for particle-based liquid simulations - Olivier Mercier, Cynthia Beauchemin, Nils Thuerey, Theodore Kim, Derek Nowrouzezahrai ](https://cim.mcgill.ca/~derek/files/surfaceWaves.pdf)

### Screen Space Fluid Rendering 
- [Screen Space Fluid Rendering for Games - Simon Green, NVIDIA](https://developer.download.nvidia.com/presentations/2010/gdc/Direct3D_Effects.pdf)
- [A Narrow-Range Filter for Screen-Space Fluid Rendering - Nghia Truong, Cem Yuksel](https://ttnghia.github.io/pdf/NarrowRangeFilter.pdf)


