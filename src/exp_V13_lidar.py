#!/usr/bin/env python3
"""
Exp 13: ley rho en LiDAR voxelizado (simulacion controlada).

Analogo estructural:
  - puntos 3D = codevectors (su posicion = informacion)
  - voxel = dimension del observador (resolucion)
  - rho = puntos_por_voxel
  - "decoder gram": agrupamiento por voxel puro (todo lo que cae junto se
    fusiona -> si hay >1 punto por voxel, informacion perdida)
  - "decoder pure" analogo: clustering con sub-voxel (resolucion real)

Medimos: accuracy de recuperacion de centros conocidos cuando la densidad
de puntos cruza rho=1.
"""
import json
import numpy as np
from pathlib import Path


def make_scene(rng, n_clusters, spread, n_points_per_cluster, vol=10.0):
    """Escena: n_clusters de centros con n_points_per_cluster puntos alrededor."""
    centers = rng.uniform(0, vol, size=(n_clusters, 3))
    pts = []
    for c in centers:
        p = c + rng.normal(0, spread, size=(n_points_per_cluster, 3))
        pts.append(p)
    return centers, np.vstack(pts)


def voxel_decoder(points, voxel_size, vol):
    """Decoder tipo Gram analogo: agrupar por voxel + centroide.
    Si 2+ clusters caen en el mismo voxel -> se confunden."""
    vox = np.floor(points / voxel_size).astype(int)
    grid = {}
    for i, v in enumerate(vox):
        key = tuple(v)
        grid.setdefault(key, []).append(points[i])
    return np.array([np.mean(v, axis=0) for v in grid.values()])


def subvoxel_decoder(points, voxel_size, sub=4):
    """Decoder 'puro': sub-resolucion (observador con mas resolucion
    que el voxel base)."""
    fine_size = voxel_size / sub
    return voxel_decoder(points, fine_size, vol=10.0)


def n_recovered(centers, recovered, tol):
    """Cuantos centros verdaderos tienen un estimado cerca."""
    n = 0
    for c in centers:
        d = np.linalg.norm(recovered - c, axis=1).min()
        if d < tol:
            n += 1
    return n


def main():
    rng = np.random.RandomState(42)
    vol = 10.0
    n_clusters = 32
    results = []

    print("=" * 70)
    print("EXP 13: ley rho en LiDAR voxelizado (analogo estructural)")
    print("=" * 70)

    # Clusters DENSOS: centros escalados para que haya solapamiento
    # real cuando el voxel es grueso. pitch centers en [0, 4] en vez de [0,10]
    # -> cuando el voxel es 2.0, ya hay 2x2x2=8 voxels para 32 centros -> colapso.
    for vol_eff in (10.0, 6.0, 4.0, 2.5):
        for voxel_size in (1.0, 2.0):
            n_voxels = max(1, int((vol_eff / voxel_size) ** 3))
            rho = n_clusters / max(n_voxels, 1)  # clusters por voxel (la ley rho)

            accs = []
            for trial in range(20):
                centers, points = make_scene(rng, n_clusters, spread=0.35,
                                             n_points_per_cluster=25, vol=vol_eff)
                rec_gram = voxel_decoder(points, voxel_size, vol_eff)
                rec_fine = subvoxel_decoder(points, voxel_size)
                # Tolerancia estricta: el estimado debe estar cerca del centro
                # real, no solo dentro del voxel contenedor
                ok_gram = n_recovered(centers, rec_gram, tol=0.5)
                ok_fine = n_recovered(centers, rec_fine, tol=0.4)
                accs.append({
                    "gram": ok_gram / n_clusters,
                    "pure": ok_fine / n_clusters,
                })

            g = np.mean([a["gram"] for a in accs])
            p = np.mean([a["pure"] for a in accs])
            results.append({
                "vol_eff": vol_eff,
                "voxel_size": voxel_size,
                "rho_clusters_per_voxel": round(rho, 3),
                "acc_gram": round(g, 3),
                "acc_pure": round(p, 3),
            })
            print(f"vol={vol_eff:4.1f} voxel={voxel_size:3.1f} rho={rho:5.2f}  "
                  f"gram={g:.3f}  pure={p:.3f}")

    out = Path(__file__).parent.parent / "data" / "exp13_lidar_rho.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nGuardado: {out}")


if __name__ == "__main__":
    main()
