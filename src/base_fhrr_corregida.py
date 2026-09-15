#!/usr/bin/env python3
"""
Implementación corregida tras la auditoría (Luciano, 2026).

NOTA (auditoría 2026-09): el base_fhrr.py incluido en este repo YA tiene VM con
16 símbolos únicos y TERR usando VL_TERR (distinto de VL/LOC) — esos dos bugs
fueron corregidos aguas arriba en algún punto anterior a este export y ya NO
están presentes en el código que ves. VM_CORREGIDO más abajo es funcionalmente
equivalente a VM (mismo tamaño, mismo grado de unicidad; solo difiere en una
palabra por razones cosméticas). Se conserva por compatibilidad, pero no
corrige nada real a esta altura.

El único fix funcional de este módulo es:
  - El decode multi-rol NO aplica mat_inv/pinv sobre el Gram matrix: usa el
    resonator puro ("pure"), que es estable para todo rho (ver paper, Sec. 5).
  - mat_inv se conserva en base_fhrr.py solo con fines de diagnóstico/comparación.
"""
from base_fhrr import (BlockBundle, rnd_phase, bind, unbind, vsub, bundle,
                       sim, mat_inv, mat_vec, proj_fhr,
                       VS, VR, VO, VL, VM, VL_TERR,
                       ROLE_CONFIGS, make_fact_flat, make_fact)

# Codebooks corregidos (16 símbolos únicos cada uno, 6 codebooks distintos)
VM_CORREGIDO = ["mañana", "tarde", "noche", "hoy", "ayer", "siempre",
                "nunca", "ahora", "antes", "despues", "pronto",
                "anochecer", "temprano", "mediodia", "frecuente", "raro"]

ROLE_CONFIGS_CORREGIDOS = {
    n: dict(cfg) for n, cfg in ROLE_CONFIGS.items()
}
ROLE_CONFIGS_CORREGIDOS[6] = {
    "names": ["SUJ", "ROL", "OBJ", "LOC", "TIME", "TERR"],
    "codebooks": {"SUJ": VS, "ROL": VR, "OBJ": VO, "LOC": VL,
                  "TIME": VM_CORREGIDO, "TERR": VL_TERR},
}


class BlockBundleCorregido(BlockBundle):
    """BlockBundle con decode puro (sin corrección por Gram matrix).

    El colapso para rho < 1 y la degeneración para rho = 1 son artefactos
    del dual del frame (ver informe, Sec. 4); el resonator puro los evita.
    """

    def decode_block_multi(self, seg, b, roles, mem):
        T = self.T
        est = [rnd_phase(self.rng, len(seg)) for _ in roles]
        for _ in range(T):
            new = [None] * len(roles)
            for idx, (j, rname) in enumerate(roles):
                others = vsub(seg, bundle([bind(self.roles[b][o], est[o])
                                           for o in range(len(roles)) if o != idx]))
                fj = unbind(others, self.roles[b][idx])
                # FIX 3: NO se aplica self.Minv[b] (mat_inv sobre M singular
                # devuelve basura numérica; pinv no salva el caso cuadrado).
                cw, _ = self.cleanup_sym(fj, rname)
                new[idx] = self.sym[cw]
            est = new
        out = {}
        for idx, (j, rname) in enumerate(roles):
            sym_name, _ = self.cleanup_sym(est[idx], rname)
            out[rname] = ("SYM", sym_name)
        return out


if __name__ == "__main__":
    import random
    rng = random.Random(42)
    for n, K, N in [(2, 1, 128), (3, 1, 128), (4, 1, 128), (6, 1, 128)]:
        cfg = ROLE_CONFIGS_CORREGIDOS[n]
        t = BlockBundleCorregido(7, K, N, n, cfg)
        accs = []
        for _ in range(10):
            f = make_fact_flat(rng, n, cfg)
            cf, mem = t.encode_fact(f)
            dec = t.decode_fact(cf, mem)
            ok, tot = t.fact_accuracy(dec, f)
            accs.append(ok / tot)
        print(f"n_roles={n}, K={K}, N={N}: acc = {sum(accs)/len(accs):.3f}")
    print("Esperado: 1.000 en todos los casos (decode puro, sin Gram).")
