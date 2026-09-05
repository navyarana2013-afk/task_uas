import os
import json
import heapq
import argparse
import itertools
from dataclasses import dataclass
import numpy as np
import cv2

# constants
 
AGE_SCORE = {"circle": 3, "star": 1, "square": 2}       # children/adults/seniors
SEVERITY_SCORE = {"red": 3, "yellow": 2, "white": 1}    # critical/moderate/safe
TERRAIN_SPEED = {0: 20.0, 1: 15.0, 2: 10.0}             # px/s by level (0=lightest)
#i took exact BGR values from paint colour picker because all the 5 input images had the same solid colour
 
# Order : based on colour and speed, index 0 = fastest and lightest terrain 
TERRAIN_COLORS = [
    (0, 215, 84),   # lightest green
    (55, 194, 97),   # medium green
    (0, 137, 76),   # darkest green
]
 
MARKER_COLORS = {
    "orange": (31, 116, 255),
    "purple": (230, 107, 203),
    "red":    (24, 0, 255),
    "yellow": (89, 222, 255),
    "white":  (255, 255, 251),
}
COLOR_MATCH_TOLERANCE = 4000   # ~36.5 per-channel average difference
# looser default , colours are solid , but due to jpg input need higher tolerance
MIN_MARKER_AREA = 40

# colour matching helpers

def color_distance_mask(img_bgr, ref_color, tolerance=COLOR_MATCH_TOLERANCE):
    b, g, r = cv2.split(img_bgr.astype(np.int32))
    rb, rg, rr = ref_color
    #ref_blue, ref_green, ref_red
    dist = (b - rb) ** 2 + (g - rg) ** 2 + (r - rr) ** 2
    return dist < tolerance
 
def classify_shape(cnt):
    #corner counting , circularity 
    peri = cv2.arcLength(cnt, True)
    if peri == 0:
        return "unknown"
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    area = cv2.contourArea(cnt)
    circularity = 4 * np.pi * area / (peri * peri + 1e-6)
    v = len(approx)
 
    if v == 3:
        return "triangle"
    if v == 4:
        x, y, w, h = cv2.boundingRect(approx)
        ar = w / float(h + 1e-6)
        if 0.75 <= ar <= 1.33:
            return "square"
    if circularity < 0.75 and v >= 7:
        return "star"
    if circularity >= 0.75:
        return "circle"
    return "star" if v >= 7 else "square"

# terrain segmentation


def build_masks(img_bgr):
    """traversable: bool - True where rover CAN move
      level_map: terrain level (0/1/2...), -1 elsewhere"""
    level_map = np.full(img_bgr.shape[:2], -1, dtype=np.int8) #-1 where no terrain level
    terrain_mask = np.zeros(img_bgr.shape[:2], dtype=bool) 
 
    for level, color in enumerate(TERRAIN_COLORS):
        m = color_distance_mask(img_bgr, color)
        level_map[m] = level 
        terrain_mask |= m 
 
    marker_mask = np.zeros(img_bgr.shape[:2], dtype=bool)
    for color in MARKER_COLORS.values():
        marker_mask |= color_distance_mask(img_bgr, color)
 
    traversable = terrain_mask | marker_mask
    trav_u8 = (traversable.astype(np.uint8)) * 255
    trav_u8 = cv2.morphologyEx(trav_u8, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    traversable = trav_u8 > 0
    obstacle_vis = np.where(traversable, 255, 0).astype(np.uint8)
    return traversable, level_map, obstacle_vis
 
 # marker detection

@dataclass
class Marker:
    kind: str
    shape: str
    color: str
    x: int
    y: int
    level: int = 0
    age_score: int = 0
    severity_score: int = 0
    priority: int = 0
 
def sample_level_near(level_map, cx, cy, cnt):
    x, y, w, h = cv2.boundingRect(cnt)
    rad_px = max(w, h) // 2 + 6 #picking a radius half the width/height +6 pixels
    h_img, w_img = level_map.shape
    candidates = []
    for ang in range(0, 360, 20):
        rad = np.radians(ang)
        px = int(cx + rad_px * np.cos(rad))
        py = int(cy + rad_px * np.sin(rad)) #computes coordinates at that radius from centroid
        if 0 <= px < w_img and 0 <= py < h_img and level_map[py, px] >= 0:
            candidates.append(level_map[py, px]) #check if sample is inside image
    if not candidates:
        return 0
    #if not, default level 0
    vals, counts = np.unique(candidates, return_counts=True)
    return int(vals[np.argmax(counts)])
 
def detect_markers(img_bgr, level_map): 
    markers = []
    for color_name, ref_color in MARKER_COLORS.items():
        mask = color_distance_mask(img_bgr, ref_color)
        mask_u8 = (mask.astype(np.uint8)) * 255 
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN,
                                    np.ones((3, 3), np.uint8)) 
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE) 
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_MARKER_AREA:
                continue
            M = cv2.moments(cnt)
            if M["m00"] == 0: # if 0 , no area, skips invalid shapes
                continue
            cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            shape = classify_shape(cnt) 
 
            if color_name == "orange" and shape == "triangle":
                markers.append(Marker("start", "triangle", "orange", cx, cy))
            elif color_name == "purple" and shape == "triangle": 
                markers.append(Marker("end", "triangle", "purple", cx, cy))
            elif color_name in ("red", "yellow", "white") and shape in AGE_SCORE:
                lvl = sample_level_near(level_map, cx, cy, cnt)
                age = AGE_SCORE[shape]
                sev = SEVERITY_SCORE[color_name]
                markers.append(Marker("casualty", shape, color_name, cx, cy,
                                       level=lvl, age_score=age,
                                       severity_score=sev,
                                       priority=age * sev))
    return markers
 
# a* pathfinding

NEIGHBORS = [(-1, -1, np.sqrt(2)), (-1, 0, 1), (-1, 1, np.sqrt(2)),
             (0, -1, 1),                        (0, 1, 1),
             (1, -1, np.sqrt(2)),  (1, 0, 1),  (1, 1, np.sqrt(2))]
def astar(traversable, start, goal): #runs a* algorithm on grid 
    h_img, w_img = traversable.shape
    sx, sy = start
    gx, gy = goal
 
    def in_bounds(x, y):
        return 0 <= x < w_img and 0 <= y < h_img
 
    def h(x, y):
        return np.hypot(x - gx, y - gy) 
    open_set = [(h(sx, sy), 0.0, (sx, sy))]
    came_from = {}
    gscore = {(sx, sy): 0.0}
    visited = set()
 
    while open_set:
        _, g, cur = heapq.heappop(open_set)
        if cur in visited:
            continue
        visited.add(cur)
        if cur == (gx, gy):
            path = [cur]
            while cur in came_from:
                cur = came_from[cur]
                path.append(cur)
            path.reverse()
            return path, g
 
        cx, cy = cur
        for dx, dy, cost in NEIGHBORS: 
            nx, ny = cx + dx, cy + dy
            if not in_bounds(nx, ny) or not traversable[ny, nx]: 
                continue
            ng = g + cost 
            if ng < gscore.get((nx, ny), 1e18):
                gscore[(nx, ny)] = ng
                came_from[(nx, ny)] = cur
                heapq.heappush(open_set, (ng + h(nx, ny), ng, (nx, ny)))
    return None, float("inf")
 
 
def pairwise_paths(traversable, points):
    names = list(points.keys())
    dist = {a: {} for a in names}
    path = {a: {} for a in names}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            p, d = astar(traversable, points[a], points[b])
            dist[a][b] = dist[b][a] = d
            path[a][b] = p
            path[b][a] = list(reversed(p)) if p else None
    return dist, path
 
 
# order optimization
 
def path_score_for_order(order, start_name, end_name, dist, casualties_by_name, start_xy):
    cum_dist = 0.0 
    prev = start_name
    details = [] 
    total = 0.0 
    for name in order:
        d = dist[prev][name]
        cum_dist += d
        c = casualties_by_name[name]
        displacement = np.hypot(c.x - start_xy[0], c.y - start_xy[1])
        score = (displacement / cum_dist) * c.priority if cum_dist > 0 else 0.0
        details.append({
            "name": name, "coords": (c.x, c.y), "shape": c.shape,
            "age_score": c.age_score, "severity": c.color,
            "severity_score": c.severity_score, "priority": c.priority,
            "displacement": displacement, "distance_travelled": cum_dist,
            "casualty_score": score,
        })
        total += score 
        prev = name 
    cum_dist += dist[prev][end_name]
    return total, details, cum_dist
 
 
def optimize_order(casualty_names, start_name, end_name, dist, casualties_by_name, start_xy):
    n = len(casualty_names)
    if n == 0:
        return [], 0.0, [], dist[start_name][end_name] 
 #brute force optimization step
    if n <= 8:
        best_order, best_score, best_details, best_dist = None, -1, None, None
        for perm in itertools.permutations(casualty_names):
            score, details, total_d = path_score_for_order(
                list(perm), start_name, end_name, dist, casualties_by_name, start_xy)
            if score > best_score:
                best_score, best_order, best_details, best_dist = score, perm, details, total_d
        return list(best_order), best_score, best_details, best_dist
 
    order = list(casualty_names) 
    order.sort(key=lambda nm: dist[start_name][nm] / max(casualties_by_name[nm].priority, 1)) 
    score, details, total_d = path_score_for_order(
        order, start_name, end_name, dist, casualties_by_name, start_xy) 
#refinement heuristic
    improved = True
    while improved:
        improved = False
        for i in range(len(order)):
            for j in range(i + 1, len(order)):
                cand = order[:i] + order[i:j + 1][::-1] + order[j + 1:]
                cscore, cdetails, cdist = path_score_for_order(
                    cand, start_name, end_name, dist, casualties_by_name, start_xy)
                if cscore > score: 
                    order, score, details, total_d = cand, cscore, cdetails, cdist
                    improved = True
    return order, score, details, total_d

# time calc
 
def compute_time(full_path, level_map): 
    total_time = 0.0 
    for (x1, y1), (x2, y2) in zip(full_path[:-1], full_path[1:]):
        seg = np.hypot(x2 - x1, y2 - y1) 
        lvl = level_map[y2, x2] if level_map[y2, x2] >= 0 else level_map[y1, x1]
        lvl = lvl if lvl >= 0 else 0
        #looks up at the destination cell, if <0 (invalid) , falls back to original cell
        #if still invalid , defaults to 0, base terrain
        speed = TERRAIN_SPEED.get(int(lvl), TERRAIN_SPEED[0])
        total_time += seg / speed 
    return total_time
 
 
# visualization

 
def draw_path(img_bgr, full_path, order_markers, start_m, end_m, out_path):
    vis = img_bgr.copy()
    for (x1, y1), (x2, y2) in zip(full_path[:-1], full_path[1:]):
        cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 0), 2) 
    cv2.circle(vis, (start_m.x, start_m.y), 6, (0, 0, 0), 2)
    cv2.circle(vis, (end_m.x, end_m.y), 6, (0, 0, 0), 2)
    for m in order_markers:
        cv2.circle(vis, (m.x, m.y), 8, (0, 0, 0), 2)
    cv2.imwrite(out_path, vis) 

 
def process_image(img_path, outdir):
    name = os.path.splitext(os.path.basename(img_path))[0]
 
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {img_path}")
    print(f"[DEBUG] Image loaded: {img.shape}")
 
    traversable, level_map, obstacle_vis = build_masks(img) 
    mask_path = os.path.join(outdir, f"{name}_mask.png")
    cv2.imwrite(mask_path, obstacle_vis)
    os.startfile(mask_path) 
    print("[DEBUG] Traversable mask built")
 
    markers = detect_markers(img, level_map)
    print(f"[DEBUG] Markers detected: {len(markers)}")  
    for m in markers:
        print(f"   -> {m.kind} {m.shape} {m.color} at ({m.x},{m.y})")
 
    start_m = next((m for m in markers if m.kind == "start"), None) 
    end_m = next((m for m in markers if m.kind == "end"), None) 
    casualties = [m for m in markers if m.kind == "casualty"] 
 
    if start_m is None or end_m is None:
        raise ValueError(f"{img_path}: could not find start/end triangle") 
 
    points = {"START": (start_m.x, start_m.y), "END": (end_m.x, end_m.y)}
    casualties_by_name = {}
    for i, c in enumerate(casualties):
        cname = f"C{i}"
        points[cname] = (c.x, c.y)
        casualties_by_name[cname] = c
        #assigns each casualty a name, adds their coordinates
 
    dist, subpaths = pairwise_paths(traversable, points) #runs a* pathfinding between every pair of points
    print("[DEBUG] Pairwise paths computed")
 
    order, total_score, details, total_dist = optimize_order(
        list(casualties_by_name.keys()), "START", "END", dist,
        casualties_by_name, points["START"])
    print(f"[DEBUG] Optimized order: {order}")
    #prints optimized casualty order 
 
    full_path = [points["START"]] 
    prev = "START" 
    for nm in order:
        seg = subpaths[prev][nm] 
        if seg is None:
            raise ValueError(f"Path segment from {prev} to {nm} is None")
        #if no path exists between two points, error raised
        full_path.extend(seg[1:])
        prev = nm
    seg = subpaths[prev]["END"]
    if seg is None:
        raise ValueError(f"Path segment from {prev} to END is None")
    full_path.extend(seg[1:])
 
    total_time = compute_time(full_path, level_map) 
    print(f"[DEBUG] Total travel time: {total_time:.2f} sec")
    print("\n2. Casualty Information")
    print(f"Number of casualties = {len(casualties)}")
    print(f"Casualty coordinates = {[(c.x, c.y) for c in casualties]}")
    print("\n3. Rover Path")
    print(full_path)
    print("\n4. Path Score")
    for i, d in enumerate(details, start=1):
        print(f"Casualty {i}: Priority={d['priority']}, Score={d['casualty_score']:.3f}")
    print(f"Total Path Score = {total_score:.3f}")
    print("\n5. Total Travel Time")
    print(f"Total Time = {total_time:.2f} seconds")
    path_out = os.path.join(outdir, f"{name}_path.png")
    draw_path(img, full_path, [casualties_by_name[n] for n in order],
              start_m, end_m, path_out)
    os.startfile(path_out) 
    #draw path visualization
 
    result = {
        "image": name,
        "num_casualties": len(casualties),
        "casualties": [
            {"coords": (c.x, c.y), "shape": c.shape, "age_score": c.age_score,
             "severity": c.color, "severity_score": c.severity_score,
             "priority": c.priority, "level": c.level}
            for c in casualties
        ],
        "visit_order": order,
        "path": full_path,
        "casualty_scores": details,
        "total_path_score": total_score,
        "total_travel_time_sec": total_time,
    }
    #build result dictionary
    with open(os.path.join(outdir, f"{name}_result.json"), "w") as f:
        json.dump(result, f, indent=2, default=str)
    return result
 

# ranking
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="image file or folder of images")
    parser.add_argument("--outdir", default="output")
    args = parser.parse_args()
 
    os.makedirs(args.outdir, exist_ok=True)
    if os.path.isdir(args.input):
        files = sorted([os.path.join(args.input, f) for f in os.listdir(args.input)
                         if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    else:
        files = [args.input]
 
    results = []
    failures = []
    for f in files:
        print(f"Processing {f} ...")
        try:
            r = process_image(f, args.outdir)
            results.append(r)
            print(f"  -> score={r['total_path_score']:.3f}  time={r['total_travel_time_sec']:.2f}s")
        except Exception as e:
            print(f"  !! failed: {e}")
            failures.append({"image": os.path.basename(f), "error": str(e)})

 
    if len(results) > 1:
        by_score = sorted(results, key=lambda r: -r["total_path_score"])
        by_time = sorted(results, key=lambda r: r["total_travel_time_sec"])
        ranking = {
            "path_score_ranking": [r["image"] for r in by_score],
            "time_ranking": [r["image"] for r in by_time],
            "failed_images": failures, 
        }
        with open(os.path.join(args.outdir, "ranking.json"), "w") as f:
            json.dump(ranking, f, indent=2)
        print("\n6. Ranking")
        print(f"Path Score Ranking = {ranking['path_score_ranking']}")
        print(f"Time Ranking = {ranking['time_ranking']}")
        if failures:                                                          
            print(f"Failed ({len(failures)}):", [x["image"] for x in failures])
        '''if more than one image processed, sort results by path score (descending) ,
        by travel time (ascending) , save rankings to ranking.json , print rankings to console
 '''
 
if __name__ == "__main__":
    main()

 
