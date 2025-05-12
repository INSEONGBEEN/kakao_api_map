# 📁 src/poi_collector.py
import requests
import pandas as pd

def fetch_poi_data(x, y, category_codes, keyword_queries, headers, radius=4000, max_page=45):
    results = []
    def fetch(mode, value, label):
        for page in range(1, max_page + 1):
            if mode == "category":
                url = "https://dapi.kakao.com/v2/local/search/category.json"
                params = {"category_group_code": value, "x": x, "y": y, "radius": radius, "page": page}
            else:
                url = "https://dapi.kakao.com/v2/local/search/keyword.json"
                params = {"query": value, "x": x, "y": y, "radius": radius, "page": page}
            response = requests.get(url, headers=headers, params=params)
            if response.status_code != 200 or not response.json().get("documents"):
                break
            for doc in response.json()["documents"]:
                results.append({
                    "이름": doc.get("place_name"),
                    "주소": doc.get("address_name"),
                    "카테고리": label,
                    "위도": float(doc["y"]),
                    "경도": float(doc["x"]),
                    "source": mode
                })

    for label, code in category_codes.items():
        fetch("category", code, label)
    for keyword in keyword_queries:
        fetch("keyword", keyword, keyword)
    return results
