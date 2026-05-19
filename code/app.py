from shiny import App, ui, render, reactive
from shinywidgets import output_widget, render_widget
import requests
import folium
import pandas as pd
import time
from datetime import datetime

DATA_URL = "https://raw.githubusercontent.com/YOUR_USER/YOUR_REPO/main/predictions/latest.json"

app_ui = ui.page_fluid(
    ui.h2("YouBike 缺車熱點即時預測與調度地圖"),
    ui.output_ui("status_banner"),
    ui.card(
        output_widget("map_view", height="600px")
    )
)

def server(input, output, session):
    
    @reactive.poll(
        poll_func=lambda: int(time.time() / 60), 
        value_func=lambda: requests.get(DATA_URL).json()
    )
    def fetch_latest_data():
        """Send GET request to fetch the latest inference results"""
        pass  # Implementation is defined in the arguments of reactive.poll

    @render.ui
    def status_banner():
        data = fetch_latest_data()
        status = data.get("health_status", "ok")
        update_time = data.get("update_time", "未知時間")
        
        if status == "ok":
            return ui.HTML(f"<div style='color: green; padding: 10px;'>🟢 系統正常 | 模型版本: {data.get('model_version')} | 更新時間: {update_time}</div>")
        elif status == "stale":
            return ui.HTML(f"<div style='color: orange; padding: 10px; font-weight: bold;'>🟡 資料延遲中 | 最後成功時間: {update_time}</div>")
        else:
            return ui.HTML(f"<div style='color: red; padding: 10px; font-weight: bold;'>🔴 系統異常 | 正在顯示最後一次可用資料 ({update_time})</div>")

    @render_widget
    def map_view():
        data = fetch_latest_data()
        df = pd.DataFrame(data["predictions"])
        
        m = folium.Map(location=[25.0330, 121.5654], zoom_start=13, tiles="CartoDB positron")
        
        color_map = {
            "HH": "red",
            "LL": "blue",
            "HL": "orange",
            "LH": "lightblue",
            "NS": "gray"
        }

        for _, row in df.iterrows():
            is_significant = row['moran_p_value'] < 0.05
            node_color = color_map.get(row['moran_type'], "gray") if is_significant else "gray"
            node_radius = 8 if row['pred_prob'] > 0.7 else 4

            popup_html = f"""
            <b>站點: {row['sno']}</b><br>
            當下缺車率: {row['shortage_rate']:.0%}<br>
            可借車輛: {row['available_bikes']} / {row['total_capacity']}<br>
            空間群聚 (LISA): {row['moran_type'] if is_significant else 'NS'}<br>
            <b>60分鐘內缺車機率: <span style='color:red'>{row['pred_prob']:.0%}</span></b>
            """

            folium.CircleMarker(
                location=[row['lat'], row['lng']],
                radius=node_radius,
                color=node_color,
                fill=True,
                fill_color=node_color,
                fill_opacity=0.7 if is_significant else 0.3,
                popup=folium.Popup(popup_html, max_width=300)
            ).add_to(m)

        return m

app = App(app_ui, server)