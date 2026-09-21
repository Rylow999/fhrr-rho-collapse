# Roadmap — camino a "Frame-Dual Stability Principle" (quitar el provisional)

Doc de planificación. Estado: 2026-09-21. Viven en `docs/`, repo fhrr-rho-collapse.

## Estado actual

La Frame-Dual Stability Principle (Prop. 1 del paper) está en estado **provisional**.
Para quitar el "provisional" se requiere, en orden de criticidad:

1. **Réplica independiente de Exp 18** (no compartir harness BundleV3).
2. **Generalización del ensemble** (no solo Gaussian unit-norm).
3. **Barrido de ρ continuo** (no solo puntos del grid).
4. **Control de ruido** (la medida de estabilidad real).
5. **Abstracción fuera de VSA** (la estructura es Cᵀ A C en general).
6. **Conexión con teoría de frames clásica** (posicionar el resultado).
7. **Operadores alternativos** (ridge, SVD truncada, iterativos).
8. **Análisis en iteraciones** (dinámica del error, a>1 vs a<=1).

### Bloqueante actual (audit 2026-09-21)
- [ ] **FHRR-causal**: Exp 18 es HRR-real + BSC. La Frame-Dual Stability Principle necesita
  Exp 18 en FHRR (fase compleja) para que el título del paper pueda volver
  a mencionar FHRR, o confirmar que el título actual (sin FHRR) es el correcto.

## Capa 0 — Sanidad (bloqueante para todo lo demás)

- [ ] **0.1 Réplica independiente de Exp 18.** Reimplementar desde cero, sin
  importar nada de exp_observer_taxonomy_v3. Idealmente 2 implementaciones:
  NumPy puro Y otra (Rust/PyTorch u Octave). Acepta si: gram ~0.16,
  dual_same ~0.98, idéntico M⁻¹, estados iniciales independientes por seed.
  **Si falla → la ley es artefacto del harness, paper queda con Exp 18 como
  evidencia dentro de un harness particular y la Prop 1 se debilita.**
- [ ] **0.2 Control de inicialización**: N_init ∈ {5, 10} por codebook,
  pareado entre observers. Descartar dependencia del estado inicial.
  (Actualmente Exp 18 usa 1 init por seed, pareado.")

## Capa 1 — Generalización del ensemble

- [ ] **1.1 Familias de C**: gaussiana (hecho), Rademacher, uniforme en
  esfera, correlacionadas (Toeplitz), casi-duplicadas, tight frames,
  frames patológicos, matrices construidas (Hadamard, DFT).
- [ ] **1.2 Barrido continuo de ρ**: {0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0,
  1.01, 1.05, 1.25, 1.5, 2, 4}. Medir en cada punto: λ_min, κ, ‖M⁻¹‖,
  ‖CᵀM⁺C − I‖_F, accuracy ambient vs dual. Identificar el predictor.
  **Hipótesis a falsar**: ρ solo es proxy; la variable real es λ_min.

## Capa 2 — Estabilidad directa (la medición real)

- [ ] **2.1 Amplificación de ruido**: f + ε, ε ~ N(0, σ²I) con σ en
  {1e-6, 1e-4, 1e-2, 1e-1}. Medir ‖Δy‖/‖ε‖ para ambos wirings. Reportar
  mediana y percentil 99.
  Predicción FDL: ambient ~ 1/λ_min, dual ~ 1.
- [ ] **2.2 Dinámica del error**: T ∈ {1,2,5,10,20,50,100}, medir e_t
  = ‖f_t − f*‖. Fit e_{t+1} ≈ a·e_t por observador. Si gram tiene a>1 y dual
  a≤1, conecta con estabilidad de sistemas dinámicos.

## Capa 3 — Abstracción fuera de VSA

- [ ] **3.1 Experimento puramente lineal**: y = Cx; ŷ_ambient = M⁻¹(Cx+ε),
  ŷ_dual = CᵀM⁻¹C(Cx+ε). Sin resonator, sin decode simbólico. Si sigue,
  HRR era el vehículo, no la causa. Paper se reescribe alrededor de eso.
- [ ] **3.2 Comparación con operadores alternativos**: M⁻¹, M⁺,
  ridge (M+λI)⁻¹ con λ ∈ {1e-6,1e-3,1e-1}, SVD truncada (k = n-1, n-4),
  LSQR iterativo. En ambos wirings. Ma peor pregunta: ¿cuándo la geometría
  análisis-síntesis neutraliza un mal operador interno?

## Capa 4 — Posicionamiento en la literatura

- [ ] **4.1 Frame theory (duales canónicos)**: Duffin–Schaeffer (1952),
  Christensen (2003). El frame dual canónico S⁻¹f con S = CCᵀ — CᵀM⁻¹C es
  exactamente el *canonical dual frame operator* aplicado en estado. Dar cita.
- [ ] **4.2 Proyectores oblicuos**: Eldar (Sampling in distinct subspaces,
  Springer, 2018) — reconstrucción con proyectores oblicuos en frames
  redundantes es la aplicación de ingeniería más cercana.
- [ ] **4.3 Compressed sensing**: Oblique Pursuits (arXiv 1207.2681) —
  diccionarios bi-ortogonales cuando el sistema real no cumple RIP.
- [ ] **4.4 Sigma-Delta / oversampling**: la misma proyección CᵀM⁻¹C es el
  mecanismo de noise shaping en ADC sobremuestreados.
- [ ] **4.5 Reescribir Related Work del paper** con estas 4 referencias.

## Capa 5 — Software

- [ ] **5.1 Crate fhrr-resilient v0.2**: exponer `OperatorPlacement` como
  enum (Ambient | Dual), detectar closed-loop por tipo, ruteo principal
  por placement+geometry, κ solo como fallback. Tests nuevos:
  invariancia CᵀM⁻¹C≈I en ρ=1, fallo esperado en ambient.
- [ ] **5.2 Verify script público** (`verify.py` en la raíz): corre todos los
  checks numéricos del README contra data/. Garantía para el revisor externo.

## Lo que NO se afirma hasta tener Capas 0–3

- Generalidad más allá de codebooks iid.
- Universalidad cross-algebra completa (falta FHRR causal + cómputo en
  álgebra no-VSA).
- Frame-Dual Stability Principle sin etiqueta "provisional".

## Notas de estilo

- Exp 19+ siguen la convención exp_VNN_descriptor.py.
- Cada experimento nuevo persiste JSON en data/ ANTES de que su número
  aparezca en paper/README.
- The crate tracks the paper, no al revés.
