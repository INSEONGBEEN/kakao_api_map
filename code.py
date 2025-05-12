# kakao_api_map

# poi data 수집
import requests
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import numpy as np
from tqdm import tqdm

# 🔑 Kakao API 설정
KAKAO_API_KEY = "api_key"
headers = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"}
radius = 4000
max_page = 45

# ✅ 카테고리 코드
category_codes = {
    "편의점": "CS2",
    "약국": "PM9",
    "병원": "HP8"
}

# ✅ 키워드 기반 검색어
keyword_queries = [
    "코인빨래방",
    "헬스장",
    "스터디카페",
    "PC방",
    "프린트카페"
]

# ✅ POI 수집 함수
def fetch_poi(x, y, mode, value, label):
    results = []
    for page in range(1, max_page + 1):
        if mode == "category":
            url = "https://dapi.kakao.com/v2/local/search/category.json"
            params = {"category_group_code": value, "x": x, "y": y, "radius": radius, "page": page}
        else:
            url = "https://dapi.kakao.com/v2/local/search/keyword.json"
            params = {"query": value, "x": x, "y": y, "radius": radius, "page": page}
        response = requests.get(url, headers=headers, params=params)
        if response.status_code != 200:
            print(f"[ERROR] {label}, page {page}")
            break
        documents = response.json().get("documents", [])
        if not documents:
            break
        for doc in documents:
            results.append({
                "이름": doc.get("place_name"),
                "주소": doc.get("address_name"),
                "카테고리": label,
                "위도": float(doc["y"]),
                "경도": float(doc["x"]),
                "source": mode
            })
    return results

# ✅ gdf_gwanak을 기반으로 내부 격자 포인트 생성 (약 500m 간격)
unified_poly = gdf_gwanak.unary_union
interval = 0.0045  # 위경도 기준 약 500m
minx, miny, maxx, maxy = unified_poly.bounds

grid_points = []
for x in np.arange(minx, maxx, interval):
    for y in np.arange(miny, maxy, interval):
        pt = Point(x, y)
        if unified_poly.contains(pt):
            grid_points.append(pt)

# ✅ 좌표별 슬라이딩 수집
poi_data = []
for pt in tqdm(grid_points, desc="관악구 슬라이딩 좌표 수집"):
    x = pt.x
    y = pt.y
    for label, code in category_codes.items():
        poi_data.extend(fetch_poi(x, y, mode="category", value=code, label=label))
    for keyword in keyword_queries:
        poi_data.extend(fetch_poi(x, y, mode="keyword", value=keyword, label=keyword))

# ✅ 결과 정리
df_poi = pd.DataFrame(poi_data)
df_poi.drop_duplicates(subset=["이름", "주소"], inplace=True)
df_poi.to_csv("관악구_POI_슬라이딩최대수집.csv", index=False)

print("✅ POI 수집 완료! 총 수:", len(df_poi))

# 진행과정
import osmnx as ox
import networkx as nx
import pandas as pd
import requests
import time
import os
from shapely.geometry import Point
from math import radians, cos, sin, asin, sqrt

# 1. 위경도 거리 계산 (Haversine)
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = radians(lat1), radians(lat2)
    d_phi = radians(lat2 - lat1)
    d_lambda = radians(lon2 - lon1)
    a = sin(d_phi/2)**2 + cos(phi1)*cos(phi2)*sin(d_lambda/2)**2
    return R * 2 * asin(sqrt(a))

# 2. 주소 → 위경도 변환
def geocode_address_kakao(address, api_key):
    url = "https://dapi.kakao.com/v2/local/search/address.json"
    headers = {"Authorization": f"KakaoAK {api_key}"}
    params = {"query": address}
    res = requests.get(url, headers=headers, params=params)
    if res.status_code == 200 and res.json()["documents"]:
        x = float(res.json()["documents"][0]["x"])
        y = float(res.json()["documents"][0]["y"])
        return y, x
    return None, None

# 3. 도보 그래프 로딩
def load_walk_graph(filepath="walk_graph.graphml"):
    if os.path.exists(filepath):
        print("📁 저장된 도보 그래프 불러오는 중...")
        return ox.load_graphml(filepath)
    else:
        print("📡 도보 그래프 새로 생성 중...")
        G = ox.graph_from_place("Gwanak-gu, Seoul, South Korea", network_type='walk')
        ox.save_graphml(G, filepath)
        return G

# 4. 후보 노드 추출
def get_candidate_nodes(G, lat, lon, radius_m=1000):
    center_node = ox.distance.nearest_nodes(G, lon, lat)
    lengths = nx.single_source_dijkstra_path_length(G, center_node, cutoff=radius_m)
    candidates = list(lengths.keys())
    print(f"✅ 전체 후보 노드 수 (풀스캔): {len(candidates)}")
    return candidates

# 5. POI 노드 매핑
def map_poi_nodes(df_poi, G, cache_path="poi_nodes.pkl"):
    if os.path.exists(cache_path):
        print("📁 노드 매핑된 POI 데이터 불러오는 중...")
        return pd.read_pickle(cache_path)
    else:
        print("🧭 POI 노드 매핑 수행 중...")
        df_poi = df_poi.copy()
        df_poi["node"] = df_poi.apply(lambda row: ox.distance.nearest_nodes(G, row["경도"], row["위도"]), axis=1)
        df_poi.to_pickle(cache_path)
        return df_poi

# 6. 추천 후보 계산
def find_top_n_nodes(G, candidate_nodes, poi_df, categories, top_n=50):  # default: 넉넉히 뽑음
    result_list = []
    for i, node in enumerate(candidate_nodes):
        if i % 100 == 0:
            print(f"🔍 후보 {i+1}/{len(candidate_nodes)} 처리 중...")
        total, detail = 0, {}
        dists_from_node = nx.single_source_dijkstra_path_length(G, node, weight='length')
        for cat in categories:
            cat_pois = poi_df[poi_df["카테고리"] == cat].copy()
            cat_pois["도로거리"] = cat_pois["node"].apply(lambda n: dists_from_node.get(n, float("inf")))
            idx = cat_pois["도로거리"].idxmin()
            nearest_row = cat_pois.loc[idx].copy()
            if pd.isna(nearest_row["도로거리"]) or nearest_row["도로거리"] == float("inf"):
                total = float("inf")
                break
            geo_dist = haversine_distance(G.nodes[node]['y'], G.nodes[node]['x'], nearest_row["위도"], nearest_row["경도"])
            nearest_row["거리"] = geo_dist
            path = nx.shortest_path(G, source=node, target=nearest_row["node"], weight='length')
            detail[cat] = {
                "이름": nearest_row["이름"],
                "거리": nearest_row["거리"],
                "위도": nearest_row["위도"],
                "경도": nearest_row["경도"],
                "경로": path
            }
            total += nearest_row["거리"]
        if total < float("inf"):
            result_list.append((node, total, detail))
    result_list.sort(key=lambda x: x[1])
    return result_list[:top_n]  # 상위 후보만 반환

# 7. 공간 분산 필터링
def apply_spatial_diversity(top_nodes, G, min_separation=200, top_n=3):
    selected, selected_points = [], []
    for node, total, detail in top_nodes:
        x, y = G.nodes[node]['x'], G.nodes[node]['y']
        pt = Point(x, y)
        is_far = all(haversine_distance(y, x, p.y, p.x) > min_separation for p in selected_points)
        if is_far:
            selected.append((node, total, detail))
            selected_points.append(pt)
        if len(selected) >= top_n:
            break
    return selected

# 8. 실행 함수
def run_recommendation_top_n(address, kakao_api_key, df_poi, radius=1000, top_n=3, min_separation=200):
    print(f"\n📌 사용자 입력 주소: {address}\n")
    lat, lon = geocode_address_kakao(address, kakao_api_key)
    if lat is None:
        raise ValueError("주소를 찾을 수 없습니다.")
    t0 = time.time()
    print("📡 도보 네트워크 로딩 중...")
    G = load_walk_graph()
    print(f"⏱️ 도보 그래프 로딩 시간: {round(time.time() - t0, 2)}초\n")
    t1 = time.time()
    print("🧭 후보 노드 탐색 중...")
    candidates = get_candidate_nodes(G, lat, lon, radius_m=radius)
    print(f"⏱️ 후보 노드 탐색 시간: {round(time.time() - t1, 2)}초\n")
    categories = df_poi["카테고리"].unique().tolist()
    print(f"📂 카테고리 수: {len(categories)}\n")
    t2 = time.time()
    print("🔁 최적 위치 계산 중...")
    poi_df = map_poi_nodes(df_poi, G)
    print(f"⏱️ POI 노드 매핑 완료: {round(time.time() - t2, 2)}초")
    t3 = time.time()
    raw_top_nodes = find_top_n_nodes(G, candidates, poi_df, categories, top_n=100)  # 충분히 많이 추출
    top_nodes = apply_spatial_diversity(raw_top_nodes, G, min_separation=min_separation, top_n=top_n)
    print(f"⏱️ 최적 위치 계산 시간: {round(time.time() - t3, 2)}초\n")
    for idx, (node, total, detail) in enumerate(top_nodes, 1):
        point = Point((G.nodes[node]['x'], G.nodes[node]['y']))
        print(f"🏡 추천 위치 {idx}: {point}")
        print(f"🧭 총 거리 점수: {round(total)} m")
        for cat, info in detail.items():
            print(f"  - {cat}: {info['이름']} ({round(info['거리'])} m) at ({info['위도']:.5f}, {info['경도']:.5f})")
        print()
    print(f"✅ 전체 실행 시간: {round(time.time() - t0, 2)}초")
    return top_nodes, G

# 📌 실행 예시
address = "서울 관악구 관악로 1"
kakao_api_key = "api_key"
top_results, G = run_recommendation_top_n(address, kakao_api_key, df_poi, radius=1000, top_n=3, min_separation=200)

import folium
from folium.plugins import BeautifyIcon

# 지도 시각화
# 카테고리별 아이콘
category_icons = {
    "편의점": "shopping-cart",
    "약국": "medkit",
    "병원": "plus-square",
    "코인빨래방": "tint",
    "헬스장": "dumbbell",
    "스터디카페": "book",
    "PC방": "desktop",
    "프린트카페": "print"
}

# 추천 위치별 고유 색상 (최대 10개까지)
rank_colors = [
    "red", "green", "purple", "orange", "darkblue",
    "darkgreen", "cadetblue", "pink", "gray", "black"
]

def visualize_top_n_straight_lines(top_nodes, df_poi, G, categories, zoom_start=16):
    first_node = top_nodes[0][0]
    m = folium.Map(location=[G.nodes[first_node]['y'], G.nodes[first_node]['x']], zoom_start=zoom_start)

    for rank, (node, total_dist, detail) in enumerate(top_nodes, 1):
        lat, lon = G.nodes[node]['y'], G.nodes[node]['x']
        color = rank_colors[(rank - 1) % len(rank_colors)]

        # 추천 위치 마커
        folium.Marker(
            location=[lat, lon],
            popup=f"🏡 추천 위치 {rank} (총 거리 {round(total_dist)}m)",
            icon=BeautifyIcon(
                icon_shape='marker',
                number=rank,
                border_color=color,
                text_color=color,
                background_color='white'
            )
        ).add_to(m)

        for cat, info in detail.items():
            icon = category_icons.get(cat, "info-sign")

            # POI 마커
            folium.Marker(
                location=[info["위도"], info["경도"]],
                popup=f"{cat}: {info['이름']} ({round(info['거리'])}m)",
                icon=folium.Icon(color="blue", icon=icon, prefix='fa')
            ).add_to(m)

            # 직선 라인 (추천 위치 → POI)
            folium.PolyLine(
                locations=[(lat, lon), (info["위도"], info["경도"])],
                color=color,
                weight=2.5,
                opacity=0.8,
                tooltip=f"{cat}까지 직선 연결"
            ).add_to(m)

    return m

categories = df_poi["카테고리"].unique().tolist()
map_result = visualize_top_n_straight_lines(top_nodes=top_results, df_poi=df_poi, G=G, categories=categories)

# HTML로 저장
map_result.save("recommended_locations_map.html")
map_result
