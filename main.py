# main.py

import pandas as pd
from src.network_analyzer import load_walk_graph, get_candidate_nodes, haversine_distance
from src.recommender import geocode_address_kakao, map_poi_nodes, find_top_n_nodes, apply_spatial_diversity
from src.visualizer import visualize_top_n_straight_lines

# 📍 사용자 설정
address = "서울 관악구 관악로 1"
kakao_api_key = "YOUR_KAKAO_API_KEY"  # 🔑 꼭 개인 발급 키로 교체!
radius = 1000
top_n = 3
min_separation = 200

# 📂 1. POI 데이터 로딩
df_poi = pd.read_csv("data/관악구_POI_슬라이딩최대수집.csv")

# 🧭 2. 도보 그래프 불러오기
G = load_walk_graph()

# 🗺️ 3. 주소 → 위경도 변환
lat, lon = geocode_address_kakao(address, kakao_api_key)
if lat is None:
    raise ValueError("입력한 주소의 위경도 정보를 찾을 수 없습니다.")

# 🛣️ 4. 후보 노드 탐색
candidate_nodes = get_candidate_nodes(G, lat, lon, radius_m=radius)

# 🧭 5. POI → 그래프 노드 매핑
df_poi = map_poi_nodes(df_poi, G)

# 🧠 6. 최적 위치 추천
categories = df_poi["카테고리"].unique().tolist()
raw_top_nodes = find_top_n_nodes(G, candidate_nodes, df_poi, categories, top_n=50)
top_nodes = apply_spatial_diversity(raw_top_nodes, G, min_separation=min_separation, top_n=top_n)

# 🎨 7. 시각화 및 저장
map_result = visualize_top_n_straight_lines(top_nodes, df_poi, G, categories)
map_result.save("outputs/recommended_locations_map.html")

print("✅ 지도 시각화 완료: outputs/recommended_locations_map.html")
