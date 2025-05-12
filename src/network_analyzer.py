import osmnx as ox
import networkx as nx
import pandas as pd
from shapely.geometry import Point
from math import radians, cos, sin, asin, sqrt

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = radians(lat1), radians(lat2)
    d_phi = radians(lat2 - lat1)
    d_lambda = radians(lon2 - lon1)
    a = sin(d_phi/2)**2 + cos(phi1)*cos(phi2)*sin(d_lambda/2)**2
    return R * 2 * asin(sqrt(a))

def load_walk_graph(filepath="walk_graph.graphml"):
    if os.path.exists(filepath):
        return ox.load_graphml(filepath)
    else:
        G = ox.graph_from_place("Gwanak-gu, Seoul, South Korea", network_type='walk')
        ox.save_graphml(G, filepath)
        return G

def get_candidate_nodes(G, lat, lon, radius_m=1000):
    center_node = ox.distance.nearest_nodes(G, lon, lat)
    lengths = nx.single_source_dijkstra_path_length(G, center_node, cutoff=radius_m)
    return list(lengths.keys())

def map_poi_nodes(df_poi, G):
    df_poi = df_poi.copy()
    df_poi["node"] = df_poi.apply(lambda row: ox.distance.nearest_nodes(G, row["경도"], row["위도"]), axis=1)
    return df_poi

def find_top_n_nodes(G, candidate_nodes, poi_df, categories, top_n=50):
    result_list = []
    for node in candidate_nodes:
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
            detail[cat] = {
                "이름": nearest_row["이름"],
                "거리": geo_dist,
                "위도": nearest_row["위도"],
                "경도": nearest_row["경도"]
            }
            total += geo_dist
        if total < float("inf"):
            result_list.append((node, total, detail))
    result_list.sort(key=lambda x: x[1])
    return result_list[:top_n]

def apply_spatial_diversity(top_nodes, G, min_separation=200, top_n=3):
    selected, selected_points = [], []
    for node, total, detail in top_nodes:
        x, y = G.nodes[node]['x'], G.nodes[node]['y']
        pt = Point(x, y)
        if all(haversine_distance(y, x, p.y, p.x) > min_separation for p in selected_points):
            selected.append((node, total, detail))
            selected_points.append(pt)
        if len(selected) >= top_n:
            break
    return selected
