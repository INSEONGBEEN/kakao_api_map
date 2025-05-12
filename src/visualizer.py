import folium
from folium.plugins import BeautifyIcon

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
        folium.Marker(
            location=[lat, lon],
            popup=f"🏡 추천 위치 {rank} (총 거리 {round(total_dist)}m)",
            icon=BeautifyIcon(
                icon_shape='marker', number=rank,
                border_color=color, text_color=color, background_color='white')
        ).add_to(m)

        for cat, info in detail.items():
            icon = category_icons.get(cat, "info-sign")
            folium.Marker(
                location=[info["위도"], info["경도"]],
                popup=f"{cat}: {info['이름']} ({round(info['거리'])}m)",
                icon=folium.Icon(color="blue", icon=icon, prefix='fa')
            ).add_to(m)
            folium.PolyLine(
                locations=[(lat, lon), (info["위도"], info["경도"])],
                color=color, weight=2.5, opacity=0.8,
                tooltip=f"{cat}까지 직선 연결"
            ).add_to(m)
    return m
