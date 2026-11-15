"""viz_mask.py -- render exclusion overlays on a set of images (contact sheets)."""
import sys, os, cv2, numpy as np
import config, utils
from exclude import build_exclusion_mask

COL = {"ocr": (0, 0, 255), "text": (0, 128, 255), "corner": (255, 0, 255), "scalebar": (0, 165, 255),
       "ruler": (255, 255, 0), "card": (0, 255, 0)}

def overlay(path, T=330):
    g = utils.load_image(path)
    m, parts = build_exclusion_mask(g, config.EXCLUDE_RULES, utils.load_color(path))
    col = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_GRAYSCALE), cv2.COLOR_GRAY2BGR)
    out = col.copy()
    for name, pm in parts.items():
        out[pm > 0] = (0.45 * out[pm > 0] + 0.55 * np.array(COL[name])).astype(np.uint8)
    h, w = g.shape
    s = T / max(h, w)
    t = cv2.resize(out, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    c = np.full((T + 18, T, 3), 255, np.uint8); c[18:18 + t.shape[0], :t.shape[1]] = t
    frac = 100.0 * (m > 0).mean()
    cv2.putText(c, f"{os.path.basename(path)[:-4]} excl={frac:.1f}%", (2, 13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 200), 1)
    return c

if __name__ == "__main__":
    folder, outprefix = sys.argv[1], sys.argv[2]
    ids = sys.argv[3:]
    tiles = [overlay(os.path.join(folder, i + ".png")) for i in ids]
    while len(tiles) % 5: tiles.append(np.full_like(tiles[0], 255))
    per = 15
    for k in range(0, len(tiles), per):
        chunk = tiles[k:k + per]
        while len(chunk) % 5: chunk.append(np.full_like(tiles[0], 255))
        rows = [np.hstack(chunk[r:r + 5]) for r in range(0, len(chunk), 5)]
        cv2.imwrite(f"{outprefix}_{k // per}.png", np.vstack(rows))
