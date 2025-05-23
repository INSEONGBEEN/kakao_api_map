
# 📍 카카오맵 API 기반 POI 데이터를 활용한 공간 기반 다목적 접근성 최소화 위치 추천 시스템

@injjang  
2025.05 ~ 2025.05

---

## 🔗 Live Demo  
(https://inseongbeen.github.io/kakao_api_map/recommended_locations_map.html)

## 📂 GitHub Repository  
https://github.com/INSEONGBEEN/seoul-residence-score

## 📘 Dev Log  
https://lnjjang.tistory.com/

---

> 📌 **카카오맵 API 기반으로 서울 관악구의 POI 데이터를 수집하고**, 도보 네트워크 기반 접근성 계산을 통해 **다목적 접근성 최소화 위치를 추천**하는 시스템을 구현했습니다.  
> 도보 그래프 기반 최단거리 계산과 공간 분산 알고리즘을 적용한 **위치 추천 알고리즘과 시각화가 핵심**입니다.

---

## 🛠️ 주요 기능

- **카카오맵 API 기반 POI 수집**
    - 관악구 전체에 대해 카테고리 기반 슬라이딩 방식으로 데이터 수집
    - 8가지 카테고리: 편의점, 약국, 병원, 코인빨래방, 헬스장, 스터디카페, PC방, 프린트카페
- **도보 네트워크 분석 (osmnx)**
    - OSM을 기반으로 관악구 도보 네트워크 그래프 구성
    - POI 및 후보지 간 거리 계산 시 도보 네트워크 경로 기반 최단거리 사용
- **후보지 접근성 평가 및 추천**
    - 각 후보지에서 모든 카테고리 POI까지의 최단거리를 합산한 점수 기반 순위화
    - 가까운 POI에만 의존하지 않고 **전체 편의시설에의 접근성 고려**
- **공간 분산 로직**
    - 단일 위치 집중을 방지하기 위해 후보지 간 최소 거리(min_separation) 조건 적용
- **Folium 기반 시각화**
    - 추천 위치에 번호 마킹 및 카테고리 아이콘 표시
    - 경로 대신 추천 위치 → POI 간 **직선 연결 시각화**

---

## 🧱 기술 스택

| Category | Tools |
|----------|-------|
| 언어 | Python |
| API 연동 | Kakao Map REST API |
| 네트워크 분석 | osmnx, networkx |
| 시각화 | folium |
| 개발 환경 | Jupyter Notebook |
| 배포 | HTML export (`.save()` 방식) |

---

## 🗂️ 디렉토리 구조

```
📁 seoul-residence-score
├── data/
│   └── 관악구_POI_슬라이딩최대수집.csv
├── src/
│   ├── api.py
│   ├── network.py
│   ├── analysis.py
│   └── visualization.py
├── main.py
├── requirements.txt
└── README.md
```

---

## 🚀 실행 예시

- 사용자 입력: `서울 관악구 관악로 1`
- 반경: 1km
- 추천 Top-3 위치와 주변 POI까지의 거리 및 직선 시각화 결과 포함 (HTML)

---

## 🔧 보완할 점 & 향후 아이디어

| 한계점 | 보완 아이디어 |
|--------|----------------|
| folium 시각화의 중첩 경로 가시성 한계 | Kepler.gl, ipyleaflet 등 대안 고려 |
| 추천 기준 단일화 | 사용자 선호 가중치 기반 거리 점수 커스터마이징 |
| 실시간 데이터 반영 어려움 | 크롤링 주기 자동화 및 최신화 기능 추가 가능 |

---

## ✍️ 느낀 점

- **API 활용 능력 향상**: 실제 API 호출, 파싱, 캐시 저장 등의 반복작업을 체계화
- **지도 시각화의 직관성**: 단순 표보다 위치 기반 시각화가 이해도와 전달력에서 매우 유리함
- **모듈화 중요성 체감**: 이후 확장(예: 사용자 선호 반영 추천) 고려해 코드 구성 방식 개선
