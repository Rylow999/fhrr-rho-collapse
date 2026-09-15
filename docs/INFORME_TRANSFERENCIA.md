# Informe de Transferencia: Colapso Binario en FHRR y HRR Real

**Fecha:** 2026-09-15
**Autor:** Luciano Benjamín Nieto
**Estado:** Datos verificados, código reproducible, listo para investigación paralela

---

## 1. Resumen Ejecutivo

### Lo que encontramos

Identificamos una **ley de ρ** que gobierna la decodificación composicional en Vector Symbolic Architectures (VSA) continuas, pero con **topologías diferentes** según el álgebra:

**FHRR (fases complejas):**
- ρ < 1: matriz singular → `mat_inv` devuelve basura → colapso
- ρ = 1: dual degenerado → colapso parcial
- ρ > 1: frame estable → funciona

**HRR real (Gaussianos + convolución circular):**
- ρ < 1: funciona perfectamente (1.000)
- ρ = 1: **anti-resonancia** → colapso catastrófico (0.167) ← descubrimiento nuevo
- ρ > 1: funciona (0.95-1.00)

**Universal:**
- El decoder `pure` (resonator sin Gram) funciona en todo el grid en ambas álgebras
- El decoder `gradient` colapsa en todos los casos multi-rol (problema general de optimización continua vs selección discreta)

### Lo que queda abierto

1. **Teoría analítica:** ¿Por qué ρ=1 es singular en HRR real? (interferencia destructiva en convolución circular)
2. **VSA binarias:** ¿La ley de ρ aplica a BSC/MAP? (tests preliminares ambiguos)
3. **Teoría unificada:** Predicción de topología desde propiedades algebraicas del bind/unbind
4. **Decodificadores alternativos:** Beam search, learned decoders, belief propagation

---

## 2. Estado del Proyecto

### Líneas cerradas (publicables)

| Línea | Estado | Datos | Código |
|-------|--------|-------|--------|
| **FHRR: ley de ρ** | ✅ Cerrada | F2, H2, diag_A4, diag_H2_cond | `exp_F2.py`, `exp_H2.py` |
| **HRR real: anti-resonancia** | ✅ Cerrada | V3 | `exp_V3_hrr_real.py` |
| **Gradient: fracaso universal** | ✅ Documentado | diag_gradient, G2 | `diag_gradient.py`, `exp_G2_gradient_variants.py` |
| **Parametrización geométrica** | ✅ Descartada | I | `exp_I_geometrica.py` |

### Líneas abiertas (trabajo futuro)

| Línea | Prioridad | Estimación | Dependencias |
|-------|-----------|------------|--------------|
| **VSA binarias (BSC/MAP)** | Alta | 1 semana | Ninguna |
| **Teoría analítica de anti-resonancia** | Media | 2-3 semanas | Background en análisis de Fourier |
| **Beam search / learned decoders** | Media | 2 semanas | Ninguna |
| **Teoría unificada** | Baja | 1-2 meses | Todas las anteriores |

---

## 3. Datos Empíricos Completos

### 3.1 FHRR: Resultados de F2 (variación de T)

**Configuración:** K=1, N=128, n_roles ∈ {2,3,4,6}, T ∈ {25,50,100,200,500}

```
n | vec_dist |  BLK |   rho |  rank |      cond
2 |       32 |  128 |  0.25 |    32 |  9.45e+17
3 |       48 |  128 |  0.38 |    48 |  1.06e+18
4 |       64 |  128 |  0.50 |    64 |  3.98e+17
6 |       78 |  128 |  0.61 |    78 |  2.86e+17

--- n_roles=2, K=1, N=128 (BLK=128) ---
T |  original |      pure |      pinv
25 |     0.000 |     1.000 |     1.000
50 |     0.050 |     1.000 |     1.000
100 |     0.050 |     1.000 |     1.000
200 |     0.100 |     1.000 |     1.000
500 |     0.150 |     1.000 |     1.000

--- n_roles=3, K=1, N=128 (BLK=128) ---
T |  original |      pure |      pinv
25 |     0.067 |     1.000 |     1.000
50 |     0.100 |     1.000 |     1.000
100 |     0.000 |     1.000 |     1.000
200 |     0.067 |     1.000 |     1.000
500 |     0.000 |     1.000 |     1.000

--- n_roles=4, K=1, N=128 (BLK=128) ---
T |  original |      pure |      pinv
25 |     0.025 |     1.000 |     1.000
50 |     0.050 |     1.000 |     1.000
100 |     0.050 |     1.000 |     1.000
200 |     0.125 |     1.000 |     1.000
500 |     0.025 |     1.000 |     1.000

--- n_roles=6, K=1, N=128 (BLK=128) ---
T |  original |      pure |      pinv
25 |     0.050 |     1.000 |     1.000
50 |     0.083 |     1.000 |     1.000
100 |     0.117 |     1.000 |     1.000
200 |     0.050 |     1.000 |     1.000
500 |     0.083 |     1.000 |     1.000
```

**Interpretación:** `original` (mat_inv) colapsa en todos los casos. `pure` y `pinv` funcionan perfectamente. El colapso NO es convergencia lenta ni techo de capacidad: es el Minv basura.

---

### 3.2 FHRR: Resultados de H2 (grid de ρ con 4 decoders)

**Configuración:** n=6, CFG6 (6 codebooks disjuntos de 16 símbolos), K ∈ {1,2,3}, N ∈ {64,72,96,120,128}

```
--- n=4 K=1 N=128 | BLK=128 | b0:64/128=0.50 ---
decoder |    T=25 |   T=100
     gram |   0.075 |   0.075
 gradient |   0.350 |   0.450
     pure |   1.000 |   1.000
     pinv |   1.000 |   1.000

--- n=6 K=1 N=128 | BLK=128 | b0:96/128=0.75 ---
decoder |    T=25 |   T=100
     gram |   0.033 |   0.050
 gradient |   0.417 |   0.517
     pure |   1.000 |   1.000
     pinv |   1.000 |   1.000

--- n=6 K=2 N=128 | BLK=64 | b0:48/64=0.75  b1:48/64=0.75 ---
decoder |    T=25 |   T=100
     gram |   0.150 |   0.183
 gradient |   0.367 |   0.183
     pure |   1.000 |   1.000
     pinv |   1.000 |   1.000

--- n=6 K=3 N=120 | BLK=40 | b0:32/40=0.80  b1:32/40=0.80  b2:32/40=0.80 ---
decoder |    T=25 |   T=100
     gram |   0.567 |   0.583
 gradient |   0.300 |   0.367
     pure |   1.000 |   1.000
     pinv |   1.000 |   1.000

--- n=6 K=3 N=96 (CUADRADO, replica Prueba B) | BLK=32 | b0:32/32=1.00  b1:32/32=1.00  b2:32/32=1.00 ---
decoder |    T=25 |   T=100
     gram |   0.783 |   0.800
 gradient |   0.167 |   0.133
     pure |   1.000 |   1.000
     pinv |   0.683 |   0.767

--- n=6 K=3 N=72 | BLK=24 | b0:32/24=1.33  b1:32/24=1.33  b2:32/24=1.33 ---
decoder |    T=25 |   T=100
     gram |   0.933 |   0.967
 gradient |   0.150 |   0.117
     pure |   1.000 |   1.000
     pinv |   0.967 |   0.933

--- n=6 K=2 N=64 | BLK=32 | b0:48/32=1.50  b1:48/32=1.50 ---
decoder |    T=25 |   T=100
     gram |   0.983 |   0.950
 gradient |   0.183 |   0.283
     pure |   1.000 |   1.000
     pinv |   0.967 |   1.000

--- n=3 K=3 N=126 (control) | BLK=42 | b0:1rol  b1:1rol  b2:1rol ---
decoder |    T=25 |   T=100
     gram |   1.000 |   1.000
 gradient |   1.000 |   1.000
     pure |   1.000 |   1.000
     pinv |   1.000 |   1.000
```

**Interpretación:**
- ρ < 1: `gram` colapsa, `pinv` arregla, `pure` funciona
- ρ = 1.00: `gram` da 0.80, `pinv` da 0.77, `pure` da **1.000** ← doble disociación
- ρ > 1: todos funcionan
- `gradient` colapsa en todos los casos multi-rol

---

### 3.3 FHRR: Condicionamiento de M por zona ρ

```
caso                                   |  BLK |  rank |         cond |   lambda_min | rcond_pinv=1e-10 trunca?
n=6 K=1 N=128 (rho=0.75)               |  128 |    96 |    3.148e+17 |   -1.326e-13 | SI (elimina modos)
n=6 K=3 N=120 (rho=0.80)               |   40 |    32 |    1.252e+17 |   -3.542e-14 | SI (elimina modos)
n=6 K=3 N=96  (rho=1.00, CUADRADO)     |   32 |    32 |    1.253e+03 |    9.043e-02 | NO (usa el modo chico -> amplifica ruido)
n=6 K=3 N=72  (rho=1.33)               |   24 |    24 |    6.631e+01 |    1.357e+00 | NO (usa el modo chico -> amplifica ruido)
n=6 K=2 N=64  (rho=1.50)               |   32 |    32 |    4.927e+01 |    2.889e+00 | NO (usa el modo chico -> amplifica ruido)
```

**Interpretación:** La doble disociación en ρ=1 se explica por el comportamiento de `pinv`:
- ρ < 1: M tiene autovalores negativos/cercanos a cero → `pinv` con rcond=1e-10 **trunca** esos modos → funciona
- ρ = 1: M tiene autovalores positivos pero chicos (λ_min=0.09) → `pinv` **NO trunca** (λ_min/λ_max > 1e-10) → amplifica ruido → colapso parcial
- ρ > 1: M bien condicionada → `pinv` funciona

---

### 3.4 HRR Real: Resultados de V3 (grid de ρ con 4 decoders)

**Configuración:** n=6, CFG6 (6 codebooks disjuntos de 16 símbolos), K ∈ {1,2,3}, N ∈ {64,72,96,120,128}

```
--- n=6 K=1 N=128 (rho=0.75) | BLK=128 | b0:96/128=0.75 ---
  decoder |    T=25 |   T=100
     gram |   1.000 |   1.000
 gradient |   0.333 |   0.267
     pure |   1.000 |   1.000
     pinv |   1.000 |   1.000

--- n=6 K=3 N=120 (rho=0.80) | BLK=40 | b0:32/40=0.80  b1:32/40=0.80  b2:32/40=0.80 ---
  decoder |    T=25 |   T=100
     gram |   1.000 |   1.000
 gradient |   0.233 |   0.283
     pure |   1.000 |   1.000
     pinv |   1.000 |   1.000

--- n=6 K=3 N=96 (rho=1.00, CUADRADO) | BLK=32 | b0:32/32=1.00  b1:32/32=1.00  b2:32/32=1.00 ---
  decoder |    T=25 |   T=100
     gram |   0.167 |   0.167
 gradient |   0.133 |   0.200
     pure |   1.000 |   1.000
     pinv |   0.167 |   0.167

--- n=6 K=3 N=72 (rho=1.33) | BLK=24 | b0:32/24=1.33  b1:32/24=1.33  b2:32/24=1.33 ---
  decoder |    T=25 |   T=100
     gram |   0.983 |   1.000
 gradient |   0.117 |   0.083
     pure |   0.967 |   0.983
     pinv |   1.000 |   0.983

--- n=6 K=2 N=64 (rho=1.50) | BLK=32 | b0:48/32=1.50  b1:48/32=1.50 ---
  decoder |    T=25 |   T=100
     gram |   0.950 |   0.950
 gradient |   0.150 |   0.167
     pure |   0.967 |   0.950
     pinv |   0.950 |   0.917

--- n=3 K=3 N=126 (control) | BLK=42 | b0:1rol  b1:1rol  b2:1rol ---
  decoder |    T=25 |   T=100
     gram |   1.000 |   1.000
 gradient |   1.000 |   1.000
     pure |   1.000 |   1.000
     pinv |   1.000 |   1.000
```

**Interpretación:**
- ρ < 1: `gram` funciona perfectamente (1.000) ← **lo opuesto a FHRR**
- ρ = 1.00: `gram` colapsa catastróficamente (0.167) ← **anti-resonancia**
- ρ > 1: `gram` funciona (0.95-1.00)
- `pure` funciona en todo el grid (0.95-1.00)
- `gradient` colapsa en todos los casos multi-rol

---

## 4. Código Fuente Completo

### 4.1 Estructura del Repositorio

```
fhrr-rho-collapse/
├── README.md
├── LICENSE (MIT)
├── requirements.txt (numpy>=1.20.0)
├── src/
│   ├── base_fhrr.py              # Implementación FHRR base
│   ├── base_fhrr_corregida.py    # Con fixes de auditoría
│   ├── exp_F2.py                 # FHRR: variar T
│   ├── exp_H2.py                 # FHRR: grid de ρ
│   ├── exp_V3_hrr_real.py        # HRR real: grid de ρ
│   ├── diag_A4.py                # Diagnóstico run_decode_0059h.py
│   ├── diag_H2_cond.py           # Condicionamiento de M
│   ├── diag_gradient.py          # Diagnóstico de gradient
│   ├── exp_G2_gradient_variants.py  # Variantes de gradient
│   └── exp_I_geometrica.py       # Parametrización geométrica
├── data/
│   ├── resultados_reales_completos.txt
│   └── out_V3.txt
└── paper/
    ├── main.tex
    └── references.bib
```

### 4.2 Archivos de Código

Los siguientes archivos están disponibles en el contexto de esta conversación. El swarm debe extraerlos y guardarlos en `src/`:

1. **base_fhrr.py** - Implementación FHRR base (ver sección 3.1 del informe anterior)
2. **base_fhrr_corregida.py** - Con fixes de auditoría (ver sección 3.2 del informe anterior)
3. **exp_F2.py** - Ver archivo adjunto en este mensaje
4. **exp_H2.py** - Ver archivo adjunto en este mensaje
5. **exp_V3_hrr_real.py** - Ver archivo adjunto en este mensaje
6. **diag_A4.py** - Ver archivo adjunto en este mensaje
7. **diag_H2_cond.py** - Ver archivo adjunto en este mensaje
8. **diag_gradient.py** - Ver sección "Script 1" del informe de gradient
9. **exp_G2_gradient_variants.py** - Ver sección "Script 2" del informe de gradient
10. **exp_I_geometrica.py** - Ver sección "Script 3" del informe de parametrización geométrica

### 4.3 Instrucciones de Ejecución

```bash
# Clonar repo
git clone https://github.com/Rylow999/fhrr-rho-collapse.git
cd fhrr-rho-collapse

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Instalar dependencias
pip install numpy>=1.20.0

# Ejecutar experimentos (orden recomendado)
cd src
python exp_F2.py > ../data/out_F2.txt
python exp_H2.py > ../data/out_H2.txt
python exp_V3_hrr_real.py > ../data/out_V3.txt

# Diagnósticos opcionales
python diag_A4.py > ../data/out_diag_A4.txt
python diag_H2_cond.py > ../data/out_diag_H2_cond.txt
python diag_gradient.py > ../data/out_diag_gradient.txt
python exp_G2_gradient_variants.py > ../data/out_G2.txt
python exp_I_geometrica.py > ../data/out_I.txt
```

---

## 5. Líneas de Investigación Abiertas

### 5.1 Línea A: VSA Binarias (Prioridad Alta)

**Objetivo:** Testear si la ley de ρ aplica a BSC (Binary Spatter Codes) y MAP (Multiply-Add-Permute).

**Contexto:** Tests preliminares (V2) dieron resultados ambiguos. BSC funcionó mal en todos los casos, pero el test era simplificado (sin resonator, sin Gram).

**Tareas concretas:**
1. Implementar `BSCBundle` con resonator completo (análogo a `HRRRealBundle`)
2. Implementar `MAPBundle` con resonator completo
3. Correr el grid de ρ completo (mismo que H2) con 4 decoders (gram/pure/pinv/gradient)
4. Comparar con FHRR y HRR real

**Criterio de éxito:**
- Si BSC/MAP muestran transición en ρ=1 → ley universal para todas las VSA
- Si BSC/MAP no muestran transición → ley específica de VSA continuas
- Si muestran comportamiento diferente → mapear el espacio completo de topologías

**Estimación:** 1 semana de trabajo

---

### 5.2 Línea B: Teoría Analítica de Anti-Resonancia (Prioridad Media)

**Objetivo:** Explicar analíticamente por qué ρ=1.00 es singular en HRR real (anti-resonancia).

**Contexto:** En FHRR, ρ<1 colapsa (matriz singular). En HRR real, ρ=1.00 colapsa (interferencia destructiva en convolución circular). Necesitamos una explicación matemática.

**Hipótesis:**
La convolución circular en HRR real puede causar cancelación destructiva cuando el número de codevectors es exactamente igual a la dimensionalidad del espacio (ρ=1.00). Esto no ocurre en FHRR porque el bind es multiplicación de fases (preserva norma).

**Tareas concretas:**
1. Derivar expresión analítica para el residuo en función de ρ para HRR real
2. Identificar el término que causa cancelación en ρ=1.00
3. Verificar con experimentos controlados (variando solo el número de codevectors, manteniendo BLK fijo)
4. Comparar con FHRR (mostrar que el término problemático no existe)

**Background necesario:** Análisis de Fourier, teoría de frames, análisis espectral

**Estimación:** 2-3 semanas de trabajo

---

### 5.3 Línea C: Decodificadores Alternativos (Prioridad Media)

**Objetivo:** Encontrar un decoder que funcione en ρ<1 sin Gram inverse.

**Contexto:** El gradient colapsa porque optimiza en espacio continuo pero el problema es discreto. El pure funciona pero es iterativo y lento.

**Enfoques a probar:**

**C1. Beam Search:**
- En cada iteración, mantener las top-K hipótesis por rol
- Podar las hipótesis con baja similitud al residuo
- Propagación hacia adelante (similar a decodificación de códigos)

**C2. Learned Decoder (Autoencoder):**
- Entrenar una red neuronal que mapee `seg → (sym_1, ..., sym_r)` directamente
- Arquitectura: MLP con capas fully-connected
- Loss: cross-entropy sobre símbolos correctos

**C3. Belief Propagation:**
- Tratar el decode como inferencia en un factor graph
- Nodos: roles y símbolos
- Mensajes: probabilidades marginales
- Iterar hasta convergencia

**Tareas concretas:**
1. Implementar cada enfoque (empezar con beam search, el más simple)
2. Correr en el grid de ρ completo
3. Comparar accuracy y tiempo de cómputo con pure/gram/pinv
4. Analizar en qué regímenes de ρ cada decoder es óptimo

**Estimación:** 2 semanas de trabajo

---

### 5.4 Línea D: Teoría Unificada (Prioridad Baja)

**Objetivo:** Desarrollar teoría que prediga la topología de la transición desde propiedades algebraicas del bind/unbind.

**Contexto:** FHRR y HRR real muestran topologías diferentes (monótona vs no monótona). ¿Qué propiedades algebraicas determinan cuál topología aparece?

**Propiedades a analizar:**
- Conmutatividad del bind
- Preservación de norma
- Distribución espectral del bind
- Estructura del espacio vectorial (complejo vs real)

**Tareas concretas:**
1. Catalogar propiedades algebraicas de FHRR, HRR real, BSC, MAP
2. Correlacionar con topología observada
3. Proponer teorema: "Si el bind tiene propiedad X, la transición es de tipo Y"
4. Verificar con nuevas VSA (MBAT, etc.)

**Estimación:** 1-2 meses de trabajo (depende de líneas A, B, C)

---

## 6. Checklist de Tareas para el Swarm

### Tareas inmediatas (esta semana)

- [ ] **Configurar repositorio:** Crear estructura de carpetas, subir código base
- [ ] **Replicar experimentos:** Correr F2, H2, V3 y verificar que los outputs matchean los datos de este informe
- [ ] **Documentar bugs conocidos:** Agregar sección "Known Issues" al README (VM duplicado, ROLE_CONFIGS sin clave 2, TERR compartiendo VL)
- [ ] **Generar figuras:** Crear scripts para generar Fig 1-4 del paper

### Tareas de investigación (próximas 2-4 semanas)

- [ ] **Línea A:** Implementar `BSCBundle` y `MAPBundle` con resonator completo
- [ ] **Línea A:** Correr grid de ρ para BSC/MAP
- [ ] **Línea B:** Derivar expresión analítica para residuo en HRR real
- [ ] **Línea C:** Implementar beam search decoder
- [ ] **Línea C:** Implementar learned decoder (autoencoder)

### Tareas de paper (paralelas)

- [ ] **Escribir sección 1-4:** Introduction, Background, Experimental Setup, Phase Diagram
- [ ] **Escribir sección 5:** Empirical Validation (con datos de F2, H2, V3)
- [ ] **Escribir sección 6:** Diagnosis of Numerical Artifacts (A2, A3, A4, A5)
- [ ] **Escribir sección 7:** Failed Alternatives (gradient, parametrización geométrica)
- [ ] **Escribir sección 8:** Discussion (universalidad parcial, anti-resonancia, trabajo futuro)
- [ ] **Generar abstract y conclusiones**
- [ ] **Subir a Zenodo + arXiv**

---

## 7. Contacto y Créditos

**Autor principal:** Luciano Benjamín Nieto
**Afiliación:** Investigador independiente, General Alvear, Mendoza, Argentina
**GitHub:** [Rylow999](https://github.com/Rylow999)

**Colaboradores:**
- Lautaro Emanuel Luconi (algoritmo FATE para DDSD)
- Nexus (asistencia en auditoría de código y diseño experimental)

**Licencia:** MIT

**Citación recomendada:**
```bibtex
@article{nieto2026fhrr,
  title={Phase Diagrams in Vector Symbolic Architectures: Universal Transitions vs Algebra-Specific Singularities},
  author={Nieto, Luciano Benjamín},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```

---

## 8. Notas Finales

### Honestidad metodológica

Todos los resultados fueron obtenidos con semillas fijas (7 para BlockBundle, 42 para make_fact) y son completamente reproducibles. Los bugs identificados en el código base están documentados y corregidos en `base_fhrr_corregida.py`.

### Limitaciones conocidas

1. El código usa FFT de numpy, que puede tener diferencias numéricas menores entre plataformas
2. Los experimentos de gradient usaron solo lr=0.1; no se exploró grid search de hiperparámetros
3. La parametrización geométrica fue manual; no se intentó entrenar un autoencoder

### Trabajo futuro inmediato

El siguiente paso lógico es la Línea A (VSA binarias) porque:
- Es independiente de las otras líneas
- Puede hacerse en paralelo
- Responde una pregunta clara (¿la ley es universal o específica de VSA continuas?)
- Tiene impacto directo en aplicaciones prácticas (BSC es muy usado en hardware neuromórfico)

---

**Fin del informe.** Todo lo necesario para replicar y extender este trabajo está contenido en este documento. El swarm puede trabajar en paralelo en las líneas A, B, C sin bloqueos mutuos.

*Per Aspera, Ad Astra.* 🚀