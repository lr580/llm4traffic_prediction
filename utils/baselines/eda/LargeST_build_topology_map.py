import json
from pathlib import Path
from string import Template
import textwrap

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0088

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
META_PATH = PROJECT_ROOT / 'data/LargeST/ca_meta.csv'
ADJ_PATH = PROJECT_ROOT / 'data/LargeST/ca_rn_adj.npy'
OUTPUT = SCRIPT_DIR / 'LargeST_topology.html'
NEIGHBOR_DISPLAY_LIMIT = -1  # -1 表示展示全部邻边
MAP_CLICK_RANGE_KM = -1      # -1 表示始终选取全局最近探测器

DIRECTION_COLORS = {
    'N': '#d7191c',
    'S': '#2c7bb6',
    'E': '#fdae61',
    'W': '#1a9850'
}

DIRECTION_LABELS = {
    'N': '北向 North',
    'S': '南向 South',
    'E': '东向 East',
    'W': '西向 West'
}

def haversine_by_index(lat_rad: np.ndarray, lng_rad: np.ndarray, src_idx: np.ndarray, dst_idx: np.ndarray) -> np.ndarray:
    dlat = lat_rad[dst_idx] - lat_rad[src_idx]
    dlon = lng_rad[dst_idx] - lng_rad[src_idx]
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat_rad[src_idx]) * np.cos(lat_rad[dst_idx]) * np.sin(dlon / 2.0) ** 2
    return EARTH_RADIUS_KM * 2.0 * np.arcsin(np.minimum(1.0, np.sqrt(a)))

def build_visualization() -> None:
    df = pd.read_csv(META_PATH)
    lat = df['Lat'].to_numpy(dtype=np.float64)
    lng = df['Lng'].to_numpy(dtype=np.float64)
    lat_rad = np.radians(lat)
    lng_rad = np.radians(lng)
    sensor_ids = df['ID'].astype(np.int64).to_numpy()

    adj_matrix = np.load(ADJ_PATH, mmap_mode='r')
    if adj_matrix.shape[0] != len(df):
        raise ValueError('Adjacency matrix shape mismatch with ca_meta.csv')

    row_idx, col_idx = np.nonzero(adj_matrix)
    mask = row_idx != col_idx
    row_idx = row_idx[mask]
    col_idx = col_idx[mask]
    weights = np.asarray(adj_matrix[row_idx, col_idx], dtype=np.float64)
    directed_distances = haversine_by_index(lat_rad, lng_rad, row_idx, col_idx)

    adjacency = {str(int(sensor_ids[i])): [] for i in range(len(sensor_ids))}
    min_weight = float('inf')
    max_weight = float('-inf')
    for src_row, dst_row, dist_val, weight_val in zip(row_idx, col_idx, directed_distances, weights):
        src_id = int(sensor_ids[src_row])
        dst_id = int(sensor_ids[dst_row])
        weight_float = float(weight_val)
        adjacency[str(src_id)].append({
            'target': dst_id,
            'distance': round(float(dist_val), 3),
            'weight': round(weight_float, 6)
        })
        min_weight = min(min_weight, weight_float)
        max_weight = max(max_weight, weight_float)

    for neighbor_list in adjacency.values():
        neighbor_list.sort(key=lambda item: item['weight'], reverse=True)

    edges = []
    for src_row, dst_row, dist_val, weight_val in zip(row_idx, col_idx, directed_distances, weights):
        edges.append({
            'source': int(sensor_ids[src_row]),
            'target': int(sensor_ids[dst_row]),
            'distance': round(float(dist_val), 3),
            'weight': round(float(weight_val), 6)
        })

    nodes = []
    for row in df.itertuples():
        nodes.append({
            'id': int(row.ID),
            'lat': round(float(row.Lat), 6),
            'lng': round(float(row.Lng), 6),
            'district': int(row.District),
            'county': row.County,
            'fwy': row.Fwy,
            'lanes': int(row.Lanes),
            'type': row.Type,
            'direction': row.Direction
        })

    stats = {
        'sensors': len(nodes),
        'counties': int(df['County'].nunique()),
        'districts': int(df['District'].nunique()),
        'highways': int(df['Fwy'].nunique()),
        'edges': len(edges),
        'weight_min': round(min_weight if edges else 0.0, 6),
        'weight_max': round(max_weight if edges else 0.0, 6)
    }

    bounds = {
        'min_lat': float(lat.min()),
        'max_lat': float(lat.max()),
        'min_lng': float(lng.min()),
        'max_lng': float(lng.max())
    }

    legend_items = [
        {'direction': key, 'label': DIRECTION_LABELS[key], 'color': DIRECTION_COLORS[key]}
        for key in ['N', 'S', 'E', 'W']
    ]

    neighbor_note = (
        "邻接表将展示全部来自 ca_rn_adj.npy 的邻边"
        if NEIGHBOR_DISPLAY_LIMIT <= 0
        else f"邻接表默认展示最多 {NEIGHBOR_DISPLAY_LIMIT} 条权重最高的边"
    )
    map_click_note = (
        "点击地图自动定位全局最近探测器"
        if MAP_CLICK_RANGE_KM <= 0
        else f"点击地图（半径约 {MAP_CLICK_RANGE_KM:.1f} 公里）以定位邻域"
    )

    template = Template(textwrap.dedent('''
    <!DOCTYPE html>
    <html lang="zh-Hans">
    <head>
    <meta charset="utf-8"/>
    <title>LargeST / PeMS 探测器拓扑</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
    <link rel="stylesheet" href="https://unpkg.com/leaflet.markercluster@1.5.3/dist/MarkerCluster.css"/>
    <link rel="stylesheet" href="https://unpkg.com/leaflet.markercluster@1.5.3/dist/MarkerCluster.Default.css"/>
    <style>
    html, body { height: 100%; margin: 0; font-family: 'HanHei SC', 'Microsoft YaHei', 'Segoe UI', sans-serif; background: #f8f8f8; }
    #map { height: 100%; width: 100%; }
    .info-panel { position: absolute; top: 16px; left: 16px; width: 380px; max-width: calc(100% - 32px); background: rgba(255,255,255,0.96); border-radius: 12px; padding: 18px; box-shadow: 0 12px 24px rgba(0,0,0,0.18); z-index: 1000; font-size: 13px; line-height: 1.5; }
    .info-panel h2 { margin: 0 0 10px 0; font-size: 20px; }
    .info-panel ul { margin: 8px 0 12px 18px; padding: 0; }
    .info-panel li { margin-bottom: 4px; }
    .search-box label { font-weight: 600; }
    .search-box input { width: 100%; padding: 6px 8px; margin-top: 4px; border: 1px solid #c6c6c6; border-radius: 6px; font-size: 13px; }
    .search-actions { display: flex; gap: 6px; margin-top: 6px; }
    .search-actions button { flex: 1; border: none; border-radius: 6px; padding: 7px; cursor: pointer; font-weight: 600; }
    .search-actions button#search-btn { background: #0052cc; color: #fff; }
    .search-actions button.secondary { background: #e6e6e6; color: #333; }
    #selection-info { margin-top: 12px; background: #f4f7fb; border-radius: 10px; padding: 10px; }
    .selection-summary { margin: 0; font-size: 12.5px; color: #1d1d1f; }
    .neighbor-overflow { margin-top: 6px; font-size: 12px; color: #8a6d3b; }
    .neighbor-table-wrapper { margin-top: 8px; max-height: 220px; overflow-y: auto; border: 1px solid #dce3f0; border-radius: 8px; background: #fff; }
    .neighbor-table { width: 100%; border-collapse: collapse; font-size: 12px; }
    .neighbor-table th, .neighbor-table td { padding: 6px 8px; border-bottom: 1px solid #eef3ff; text-align: left; white-space: nowrap; }
    .neighbor-table th { position: sticky; top: 0; background: #f8faff; font-weight: 600; }
    .neighbor-table tr:last-child td { border-bottom: none; }
    .legend-panel { position: absolute; bottom: 20px; left: 16px; width: 280px; max-width: calc(100% - 32px); background: rgba(255,255,255,0.94); padding: 14px; border-radius: 10px; box-shadow: 0 8px 18px rgba(0,0,0,0.18); font-size: 12px; z-index: 1000; }
    .legend-panel h4 { margin: 0 0 6px 0; font-size: 13px; }
    .legend-row { display: flex; align-items: center; margin-bottom: 4px; }
    .legend-color { width: 18px; height: 10px; border-radius: 4px; margin-right: 6px; border: 1px solid rgba(0,0,0,0.2); }
    .edge-gradient { height: 12px; border-radius: 6px; background: linear-gradient(90deg, #4575b4, #fee08b, #d73027); border: 1px solid rgba(0,0,0,0.2); margin: 6px 0 2px; }
    .edge-scale { display: flex; justify-content: space-between; font-size: 11px; color: #4a4a4a; }
    .custom-marker .marker-dot { width: 18px; height: 18px; border-radius: 50%; border: 2px solid #fff; display: inline-block; box-shadow: 0 0 4px rgba(0,0,0,0.3); }
    .hidden { display: none; }
    @media (max-width: 768px) {
      .info-panel { width: calc(100% - 32px); }
      .legend-panel { width: calc(100% - 32px); left: 16px; right: 16px; }
    }
    </style>
    </head>
    <body>
    <div id="map"></div>
    <div class="info-panel">
      <h2>LargeST / PeMS 拓扑</h2>
      <p>基于 $sensor_count 个探测器与 ca_rn_adj.npy 中 $edge_count 条有向连边，覆盖 $county_count 个县、$district_count 个 District，涉及 $highway_count 条高速/干线。</p>
      <ul>
        <li>$map_click_note</li>
        <li>节点颜色表示“北南东西”方向，边的颜色与粗细表示 ca_rn_adj 权重，悬停线段可查看权重与测地距离</li>
        <li>$neighbor_note，可结合图例查看 ID、距离与边权</li>
      </ul>
      <div class="search-box">
        <label for="sensor-input">输入探测器 ID：</label>
        <input id="sensor-input" type="text" placeholder="例如 317802"/>
        <div class="search-actions">
          <button id="search-btn">定位邻域</button>
          <button id="reset-btn" class="secondary">重置视图</button>
        </div>
      </div>
      <div id="selection-info">
        <p id="selection-summary" class="selection-summary">提示：点击任何探测器或地图背景以锁定邻域，表格会同步显示 ca_rn_adj 权重与测地距离；再次点击“重置视图”可清空选中状态。</p>
        <div id="neighbor-overflow" class="neighbor-overflow hidden"></div>
        <div id="neighbor-table-wrapper" class="neighbor-table-wrapper hidden">
          <table class="neighbor-table">
            <thead>
              <tr><th>#</th><th>邻居 ID</th><th>测地距离</th><th>ca_rn_adj 权重</th></tr>
            </thead>
            <tbody id="neighbor-table-body"></tbody>
          </table>
        </div>
      </div>
    </div>
    <div class="legend-panel" id="legend-panel"></div>
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script src="https://unpkg.com/leaflet.markercluster@1.5.3/dist/leaflet.markercluster.js"></script>
    <script>
    const nodes = $nodes_json;
    const edges = $edges_json;
    const adjacencyRaw = $adjacency_json;
    const directionColors = $direction_colors;
    const legendItems = $legend_items;
    const bounds = [[${min_lat}, ${min_lng}], [${max_lat}, ${max_lng}]];
    const neighborDisplayLimit = $neighbor_limit;
    const mapClickRangeKm = $map_click_range;
    const weightMin = $weight_min;
    const weightMax = $weight_max;

    const adjacency = new Map();
    Object.keys(adjacencyRaw).forEach((key) => {
      adjacency.set(Number(key), adjacencyRaw[key]);
    });
    const idToNode = new Map();
    nodes.forEach((node) => {
      idToNode.set(Number(node.id), node);
    });

    const map = L.map('map', { preferCanvas: true, minZoom: 5 });
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 19,
      attribution: '&copy; OpenStreetMap 贡献者'
    }).addTo(map);
    map.fitBounds(bounds, { padding: [20, 20] });
    L.control.scale({ imperial: false }).addTo(map);

    map.createPane('edges').style.zIndex = 390;
    map.createPane('highlight').style.zIndex = 650;
    const edgeLayer = L.layerGroup([], { pane: 'edges' }).addTo(map);
    const highlightLayer = L.layerGroup([], { pane: 'highlight' }).addTo(map);

    function normalizeWeight(weight) {
      if (weightMax === weightMin) {
        return 0.5;
      }
      return Math.min(1, Math.max(0, (weight - weightMin) / (weightMax - weightMin)));
    }

    function weightToColor(weight) {
      const t = normalizeWeight(weight);
      const hue = 210 - 170 * t;
      const lightness = 70 - 25 * t;
      return `hsl(${hue}, 70%, ${lightness}%)`;
    }

    function weightToStroke(weight) {
      const t = normalizeWeight(weight);
      return 0.6 + 1.5 * t;
    }

    function formatDistance(km) {
      if (km >= 1) {
        return `${km.toFixed(2)} km`;
      }
      return `${(km * 1000).toFixed(0)} m`;
    }

    function formatWeight(weight) {
      return weight.toFixed(6);
    }

    edges.forEach((edge) => {
      const src = idToNode.get(Number(edge.source));
      const dst = idToNode.get(Number(edge.target));
      if (!src || !dst) return;
      const color = weightToColor(edge.weight);
      const line = L.polyline([[src.lat, src.lng], [dst.lat, dst.lng]], {
        color,
        weight: weightToStroke(edge.weight),
        opacity: 0.8,
        pane: 'edges'
      });
      line.bindTooltip(`探测器 ${edge.source} ⇄ ${edge.target}<br/>测地距离: ${formatDistance(edge.distance)}<br/>ca_rn_adj 权重: ${formatWeight(edge.weight)}`);
      edgeLayer.addLayer(line);
    });

    function createMarkerIcon(color) {
      return L.divIcon({
        className: 'custom-marker',
        html: `<span class="marker-dot" style="background:${color}"></span>`,
        iconSize: [22, 22],
        iconAnchor: [11, 11],
        popupAnchor: [0, -12]
      });
    }

    function createPopup(node) {
      return `<div><strong>探测器 ${node.id}</strong><br/>道路: ${node.fwy}<br/>方向: ${node.direction}<br/>车道数: ${node.lanes}<br/>District ${node.district} · ${node.county}</div>`;
    }

    const clusterGroup = L.markerClusterGroup({
      disableClusteringAtZoom: 12,
      spiderfyOnMaxZoom: true,
      showCoverageOnHover: false,
      maxClusterRadius: 60,
      chunkedLoading: true
    });
    const markerMap = new Map();
    nodes.forEach((node) => {
      const marker = L.marker([node.lat, node.lng], { icon: createMarkerIcon(directionColors[node.direction] || '#888') });
      marker.bindPopup(createPopup(node));
      marker.bindTooltip(`ID: ${node.id}`, { direction: 'top', offset: [0, -12] });
      marker.on('click', () => focusSensor(node.id));
      clusterGroup.addLayer(marker);
      markerMap.set(node.id, marker);
    });
    map.addLayer(clusterGroup);

    const selectionSummary = document.getElementById('selection-summary');
    const neighborTableWrapper = document.getElementById('neighbor-table-wrapper');
    const neighborTableBody = document.getElementById('neighbor-table-body');
    const neighborOverflow = document.getElementById('neighbor-overflow');

    function clearSelectionSummary() {
      selectionSummary.textContent = '提示：点击任何探测器或地图背景以锁定邻域，表格会同步显示 ca_rn_adj 权重与测地距离；再次点击“重置视图”可清空选中状态。';
      neighborTableWrapper.classList.add('hidden');
      neighborOverflow.classList.add('hidden');
      neighborTableBody.innerHTML = '';
    }

    function focusSensor(sensorId) {
      const numericId = Number(sensorId);
      if (!Number.isFinite(numericId)) {
        selectionSummary.textContent = '请输入合法的数值 ID。';
        return;
      }
      const node = idToNode.get(numericId);
      if (!node) {
        selectionSummary.textContent = `未找到 ID 为 ${sensorId} 的探测器。`;
        return;
      }
      map.flyTo([node.lat, node.lng], Math.max(map.getZoom(), 12), { duration: 0.7 });
      highlightLayer.clearLayers();
      L.circleMarker([node.lat, node.lng], {
        radius: 10,
        color: '#000',
        weight: 2,
        fillColor: directionColors[node.direction] || '#fff',
        fillOpacity: 1,
        pane: 'highlight'
      }).bindTooltip(`中心探测器 ${node.id}`, { permanent: false }).addTo(highlightLayer);

      const neighbors = adjacency.get(numericId) || [];
      if (!neighbors.length) {
        selectionSummary.textContent = `探测器 ${node.id} · ${node.fwy} 暂无来自 ca_rn_adj.npy 的邻接边。`;
        neighborTableWrapper.classList.add('hidden');
        neighborOverflow.classList.add('hidden');
        neighborTableBody.innerHTML = '';
        return;
      }

      const limitedNeighbors = neighborDisplayLimit > 0 ? neighbors.slice(0, neighborDisplayLimit) : neighbors;
      neighborTableBody.innerHTML = '';
      limitedNeighbors.forEach((info, idx) => {
        const neighborNode = idToNode.get(Number(info.target));
        if (!neighborNode) return;
        L.circleMarker([neighborNode.lat, neighborNode.lng], {
          radius: 7,
          color: '#111',
          weight: 1,
          fillColor: '#fff',
          fillOpacity: 0.9,
          pane: 'highlight'
        }).bindTooltip(`邻居 ${neighborNode.id}`, { permanent: false }).addTo(highlightLayer);
        L.polyline([[node.lat, node.lng], [neighborNode.lat, neighborNode.lng]], {
          color: '#111',
          weight: 3,
          opacity: 0.9,
          pane: 'highlight'
        }).bindTooltip(`中心 ${node.id} → 邻居 ${neighborNode.id}<br/>测地距离: ${formatDistance(info.distance)}<br/>ca_rn_adj 权重: ${formatWeight(info.weight)}`).addTo(highlightLayer);

        const row = document.createElement('tr');
        row.innerHTML = `<td>${idx + 1}</td><td>${neighborNode.id}</td><td>${formatDistance(info.distance)}</td><td>${formatWeight(info.weight)}</td>`;
        neighborTableBody.appendChild(row);
      });

      neighborTableWrapper.classList.remove('hidden');
      if (neighborDisplayLimit > 0 && neighbors.length > neighborDisplayLimit) {
        neighborOverflow.textContent = `共 ${neighbors.length} 条邻接边，仅展示权重最高的前 ${neighborDisplayLimit} 条。`;
        neighborOverflow.classList.remove('hidden');
      } else {
        neighborOverflow.classList.add('hidden');
      }

      const shownCount = neighborDisplayLimit > 0 ? Math.min(neighborDisplayLimit, neighbors.length) : neighbors.length;
      selectionSummary.textContent = `中心探测器 ${node.id} · ${node.fwy}，显示 ${shownCount} / ${neighbors.length} 条邻接边。`;
      const marker = markerMap.get(node.id);
      if (marker) {
        marker.openPopup();
      }
    }

    function findNearestSensor(latlng, maxDistanceKm) {
      let bestNode = null;
      let bestDistance = Infinity;
      nodes.forEach((node) => {
        const dist = haversineDistanceKm(latlng.lat, latlng.lng, node.lat, node.lng);
        if (dist < bestDistance) {
          bestDistance = dist;
          bestNode = node;
        }
      });
      if (bestNode && (maxDistanceKm <= 0 || bestDistance <= maxDistanceKm)) {
        return bestNode;
      }
      return null;
    }

    function haversineDistanceKm(lat1, lng1, lat2, lng2) {
      const toRad = Math.PI / 180;
      const phi1 = lat1 * toRad;
      const phi2 = lat2 * toRad;
      const dphi = (lat2 - lat1) * toRad;
      const dlambda = (lng2 - lng1) * toRad;
      const a = Math.sin(dphi / 2) ** 2 + Math.cos(phi1) * Math.cos(phi2) * Math.sin(dlambda / 2) ** 2;
      return 6371.0088 * 2 * Math.asin(Math.min(1, Math.sqrt(a)));
    }

    document.getElementById('search-btn').addEventListener('click', () => {
      const value = document.getElementById('sensor-input').value.trim();
      if (value) focusSensor(value);
    });
    document.getElementById('sensor-input').addEventListener('keyup', (event) => {
      if (event.key === 'Enter') {
        const value = event.target.value.trim();
        if (value) focusSensor(value);
      }
    });
    document.getElementById('reset-btn').addEventListener('click', () => {
      document.getElementById('sensor-input').value = '';
      highlightLayer.clearLayers();
      map.fitBounds(bounds, { padding: [20, 20] });
      clearSelectionSummary();
    });

    map.on('click', (event) => {
      const nearest = findNearestSensor(event.latlng, mapClickRangeKm);
      if (nearest) {
        focusSensor(nearest.id);
      } else if (mapClickRangeKm > 0) {
        selectionSummary.textContent = `附近 ${mapClickRangeKm} 公里内没有探测器，请进一步放大后重试。`;
      }
    });

    function renderLegend() {
      const panel = document.getElementById('legend-panel');
      const nodeLegend = legendItems.map((item) => `<div class="legend-row"><span class="legend-color" style="background:${item.color}"></span>${item.label}</div>`).join('');
      panel.innerHTML = `<h4>节点方向</h4>${nodeLegend}<h4>边权重 (颜色/粗细)</h4><div class="edge-gradient"></div><div class="edge-scale"><span>${weightMin}</span><span>${weightMax}</span></div><p>线条越暖色/越粗表示 ca_rn_adj 权重越高，悬停可查看测地距离与精确权重。</p>`;
    }

    renderLegend();
    clearSelectionSummary();
    </script>
    </body>
    </html>
    '''))

    html = template.safe_substitute(
        nodes_json=json.dumps(nodes, ensure_ascii=False, separators=(',', ':')),
        edges_json=json.dumps(edges, ensure_ascii=False, separators=(',', ':')),
        adjacency_json=json.dumps(adjacency, ensure_ascii=False, separators=(',', ':')),
        direction_colors=json.dumps(DIRECTION_COLORS, ensure_ascii=False),
        legend_items=json.dumps(legend_items, ensure_ascii=False),
        sensor_count=stats['sensors'],
        highway_count=stats['highways'],
        county_count=stats['counties'],
        district_count=stats['districts'],
        edge_count=stats['edges'],
        weight_min=stats['weight_min'],
        weight_max=stats['weight_max'],
        neighbor_limit=NEIGHBOR_DISPLAY_LIMIT,
        neighbor_note=neighbor_note,
        map_click_note=map_click_note,
        map_click_range=MAP_CLICK_RANGE_KM,
        min_lat=bounds['min_lat'],
        max_lat=bounds['max_lat'],
        min_lng=bounds['min_lng'],
        max_lng=bounds['max_lng']
    )

    OUTPUT.write_text(html, encoding='utf-8')
    total_directed = len(edges)
    print(f'构建完成：有向边 {total_directed} 条。')
    print(f'HTML saved to {OUTPUT.as_posix()}')

def main() -> None:
    build_visualization()

if __name__ == '__main__':
    main()
