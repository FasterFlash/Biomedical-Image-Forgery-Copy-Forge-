"""where_left.py -- draw WHERE the surviving (post-mask) matches land on authentic images."""
import sys, os, cv2, numpy as np
import config, utils
from pipeline import detect
folder, outp = sys.argv[1], sys.argv[2]; ids = sys.argv[3:]
config.EXCLUDE_ENABLED = True
tiles = []; T = 420
for i in ids:
    path = os.path.join(folder, i + ".png")
    g = utils.load_image(path); r = detect(g, utils.load_color(path)); kp = r["keypoints"]
    c = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    ex = r["exclude"]
    c[ex > 0] = (0.65 * c[ex > 0] + 0.35 * np.array([255, 0, 0])).astype(np.uint8)   # blue tint = excluded
    lw = max(1, int(max(g.shape) / 700))
    for a, b in r["inliers"]:
        p, q = tuple(map(int, kp[a].pt)), tuple(map(int, kp[b].pt))
        cv2.line(c, p, q, (0, 255, 0), lw); cv2.circle(c, p, 3 * lw, (0, 0, 255), -1); cv2.circle(c, q, 3 * lw, (0, 255, 255), -1)
    h, w = g.shape; s = T / max(h, w)
    t = cv2.resize(c, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    tile = np.full((T + 18, T, 3), 255, np.uint8); tile[18:18 + t.shape[0], :t.shape[1]] = t
    cv2.putText(tile, f"{i} {len(r['inliers'])} inliers", (2, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1)
    tiles.append(tile)
while len(tiles) % 4: tiles.append(np.full_like(tiles[0], 255))
rows = [np.hstack(tiles[k:k + 4]) for k in range(0, len(tiles), 4)]
cv2.imwrite(outp, np.vstack(rows))
