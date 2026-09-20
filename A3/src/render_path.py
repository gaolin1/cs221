"""Render path.json to a PNG, framed on the path instead of the whole map."""
import sys, json, argparse
sys.path.insert(0, ".")
import plotly.graph_objects as go
from mapUtil import readMap, addLandmarks
import visualization as viz

p = argparse.ArgumentParser()
p.add_argument("--path-file", default="path.json")
p.add_argument("--out", required=True)
p.add_argument("--title", default="Stanford")
a = p.parse_args()

m = readMap("data/stanford.pbf")
addLandmarks(m, "data/stanford-landmarks.json")
data = json.load(open(a.path_file))
path, wtags = data["path"], data["waypointTags"]

captured = {}
go.Figure.show = lambda self, *args, **kw: captured.__setitem__("fig", self)
viz.plotMap(m, path, wtags, a.title)
fig = captured["fig"]

# Frame on the path. Calibrated empirically: projection_scale 20000 shows roughly
# 0.018 degrees of longitude across the plot, so scale = 360 / span.
lats = [m.geoLocations[x].latitude for x in path]
lons = [m.geoLocations[x].longitude for x in path]
cLat, cLon = (min(lats) + max(lats)) / 2, (min(lons) + max(lons)) / 2
span = max(max(lons) - min(lons), (max(lats) - min(lats)) * 2.2) * 1.25
fig.update_layout(
    geo=dict(projection_scale=360 / max(span, 1e-5),
             center=dict(lat=cLat, lon=cLon), showland=True,
             landcolor="#f7f7f5", bgcolor="white"),
    width=1500, height=820, showlegend=True,
    legend=dict(font=dict(size=11), itemsizing="constant"),
)
# Collapse the dozens of identical parking_entrance / food legend rows to one each.
seen = set()
for tr in fig.data:
    if tr.name in ("solution",) or tr.name is None:
        continue
    if tr.name in seen:
        tr.showlegend = False
    seen.add(tr.name)
fig.write_image(a.out, scale=2)
print("wrote", a.out, "| path nodes:", len(path))
