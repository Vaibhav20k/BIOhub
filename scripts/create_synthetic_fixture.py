"""Generate a synthetic 3D+t microscopy clip with division for testing."""

from pathlib import Path
import numpy as np
import polars as pl
import tracksdata as td
import zarr

DEFAULT_SCALE = (1.625, 0.40625, 0.40625)


def make_synthetic_clip(out_dir: Path, name: str = "synthetic_clip"):
    out_dir.mkdir(parents=True, exist_ok=True)
    zarr_path = out_dir / f"{name}.zarr"
    geff_path = out_dir / f"{name}.geff"

    T, Z, Y, X = 5, 16, 64, 64
    vol = np.zeros((T, Z, Y, X), dtype=np.float32)

    # Initialize graph
    g = td.graph.InMemoryGraph()
    for k in ("z", "y", "x"):
        g.add_node_attr_key(k, pl.Float64, -999999.0)

    # Track 1 (migrating cell):
    t1_nodes = []
    for t in range(T):
        coord = {"t": t, "z": 8.0, "y": 20.0 + t * 2.0, "x": 20.0 + t * 2.0}
        nid = g.add_node(coord)
        t1_nodes.append((nid, coord))

    for i in range(len(t1_nodes) - 1):
        g.add_edge(t1_nodes[i][0], t1_nodes[i + 1][0], {})

    # Track 2 (dividing cell):
    p_nodes = []
    for t in range(3):  # t=0, 1, 2
        coord = {"t": t, "z": 8.0, "y": 42.0, "x": 42.0}
        nid = g.add_node(coord)
        p_nodes.append((nid, coord))
    for i in range(len(p_nodes) - 1):
        g.add_edge(p_nodes[i][0], p_nodes[i + 1][0], {})

    # Daughters at t=3, 4
    da_nodes = []
    db_nodes = []
    for step, t in enumerate([3, 4]):
        ca = {"t": t, "z": 8.0, "y": 36.0 - step * 2.0, "x": 42.0}
        cb = {"t": t, "z": 8.0, "y": 48.0 + step * 2.0, "x": 42.0}
        n_a = g.add_node(ca)
        n_b = g.add_node(cb)
        da_nodes.append((n_a, ca))
        db_nodes.append((n_b, cb))

    # Link parent at t=2 to both daughters at t=3 (Division)
    g.add_edge(p_nodes[-1][0], da_nodes[0][0], {})
    g.add_edge(p_nodes[-1][0], db_nodes[0][0], {})

    # Link daughters from t=3 to t=4
    g.add_edge(da_nodes[0][0], da_nodes[1][0], {})
    g.add_edge(db_nodes[0][0], db_nodes[1][0], {})

    all_nodes = (
        [c for _, c in t1_nodes]
        + [c for _, c in p_nodes]
        + [c for _, c in da_nodes]
        + [c for _, c in db_nodes]
    )

    zz, yy, xx = np.meshgrid(np.arange(Z), np.arange(Y), np.arange(X), indexing="ij")
    sigma_z, sigma_y, sigma_x = 1.5, 3.0, 3.0

    for n in all_nodes:
        t = n["t"]
        cz, cy, cx = n["z"], n["y"], n["x"]
        blob = np.exp(
            -(
                ((zz - cz) ** 2) / (2 * sigma_z**2)
                + ((yy - cy) ** 2) / (2 * sigma_y**2)
                + ((xx - cx) ** 2) / (2 * sigma_x**2)
            )
        )
        vol[t] += blob.astype(np.float32)

    np.random.seed(42)
    noise = np.random.gamma(shape=2.0, scale=0.05, size=vol.shape).astype(np.float32)
    vol = np.clip(vol + noise, 0.0, 10.0)

    z_grp = zarr.open_group(str(zarr_path), mode="w")
    z_grp.create_array("0", data=vol, chunks=(1, Z, Y, X))

    flat = vol.ravel()[::10].astype(float)
    quantiles = {
        str(q): float(np.quantile(flat, float(q)))
        for q in ["0.0", "0.001", "0.01", "0.1", "0.9", "0.99", "0.999", "1.0"]
    }
    z_grp.attrs["multiscales"] = [
        {
            "datasets": [
                {
                    "coordinateTransformations": [
                        {"scale": list(DEFAULT_SCALE), "type": "scale"}
                    ],
                    "path": "0",
                }
            ],
            "version": "0.4",
        }
    ]
    z_grp.attrs["image_statistics"] = {"quantiles": quantiles}

    g.to_geff(str(geff_path), overwrite=True)
    print(f"Created synthetic fixture at {zarr_path} and {geff_path}")
    print(f"Nodes: {g.num_nodes()}, Edges: {g.num_edges()}")


if __name__ == "__main__":
    make_synthetic_clip(Path("data/fixtures"), "synthetic_clip")
