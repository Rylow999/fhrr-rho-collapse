# Transformers y el teorema Frame-Dual: analisis extenso (2026-10-10)

*Auditoria bajo el Metodo Integral (DOCUMENTATION/METODO_INTEGRAL.md).
Estados obligatorios en cada afirmacion. Referencias verificadas contra fuente.*

## 1. El marco de lectura: el residual stream como espacio VSA

La literatura reciente (Dhayalkar, arXiv:2512.14709; Hersche et al. 2023;
Smolensky 1990; Plate 1995) converge en leer el transformer como una
**arquitectura vector-simbolica suave**:

- **Queries/keys = espacios de roles**; **values = fillers**;
- **Atencion = unbinding diferenciable** (recuperar fillers segun similitud de rol);
- **Residual stream = superposicion** de estructuras ligadas (bundling);
- **Multi-head = canales de binding paralelos**.

**Estado: lectura establecida** (consenso emergente de la interpretacion
mecanicista). Lo que sigue la aplica a nuestro teorema.

## 2. La correspondencia termino a termino con FHRR/VSA

El harness del paper ES un VSA: codebook C (n simbolos), binding circular,
superposicion s = sum bind(r_i, v_i), cleanup por similitud. Tabla de
correspondencias con el transformer:

| VSA / paper fhrr | Transformer |
|---|---|
| codebook C (n vectores d-dim) | value-vectors del contexto (n tokens, d modelo) |
| Gram M = CC^T | gram de keys/values: M = KK^T — matriz de interferencia entre roles |
| superposicion s (bundle) | residual stream x (suma de contribuciones de todos los tokens) |
| decodificador por cleanup | atencion como unbinding |
| correccion ambient M^-1 · f | "attention head" que re-inyecta M^-1 al estado (unboxing bruto) |
| correccion dual C^T M^-1 C | unbinding en espacio de coeficientes (proyeccion sobre roles) |

**Estado: correspondencia formal** — las operaciones son las mismas
algebras; el teorema 1 del paper se traslada con traducion de notacion.

## 3. El hallazgo central: rank collapse en transformers = fallo ambient

### 3.1 El fenomeno (literatura, verificado)

Dong et al. (ICML 2021, arXiv:2103.03404) prueban: **redes de atencion pura
(sin skip/MLP) colapsan a rank-1 doble-exponencialmente en profundidad**.
Nait-Saada (AISTATS 2025, arXiv:2410.07799) anaden: **rank collapse en
ANCHURA cuando el contexto T crece**, con causa espectral precisa: un gap
entre lambda_1 y lambda_2 de la matriz de atencion ~ (1/T)*11, y proponen
remover el outlier como fix.

### 3.2 La lectura fhrr (nuestra contribucion — estado: PREDICCION FALSABLE)

**El rank collapse en anchura es el fallo ambient del teorema 1, con el
mismo mecanismo geometrico:**

1. Con T tokens (roles), la gram de interferencia M = KK^T con keys
   quasi-aleatorias entra en el regimen **cuadrado** cuando el numero de
   roles efectivos ~ dimension (o, en su forma de doble muestreo, el borde
   duro con T grande);
2. lambda_min(M) ~ n^-2 (hard edge — nuestro Exp 16/28: exponente -2.058
   verificado; analogo Wishart de Chen-Liu-Zhou);
3. Toda operacion de unbinding que aplique el resolvente al **estado
   ambiente** (M^-1 · x: un "head" que corrige la superposicion in situ)
   amplifica las direcciones debiles por 1/lambda_min y **colapsa la
   informacion** — exactamente la Tabla 3 del paper (ambient 0.160 vs
   dual 0.985);
4. El fix de Nait-Saada (remover el outlier lambda_1) es un caso particular
   de la familia de reparaciones que el teorema clasifica: **la reparacion
   canonica es cambiar la colocacion, no el espectro** — proyectar en
   espacio de coeficientes C^T M^+ C, que es el dual frame con norma
   exactamente 1 (invariante al hard edge).

**Prediccion concreta (falsable, pre-registrada):** en un transformer con
contexto largo en regimen de rank collapse, sustituir la correccion ambient
del head (M^-1·x, si existe tal circuito) por la correccion dual (en
espacio de roles: proyeccion C^T M^+ C sobre el codebook de keys) restaura
la informacion sin tocar el espectro. **Si esto es falso, la lectura fhrr
del rank collapse muere.** (Protocolo 1.5: prediccion cuantitativa
falsable — resultado tipo "medio".)

### 3.3 El detalle fino: por que los transformers "no colapsan siempre"

El paper Exp 20A da la respuesta exacta: **el frame ortogonal no colapsa
porque M = I** (lambda_min = 1, sin hard edge). Los transformers reales
evitan el colapso total por la misma razon estructural que Exp 20A: keys
entrenadas que se auto-organizan hacia cuasi-ortogonalidad (reducen
interferencia de roles — Dhayalkar, condiciones de "VSA-likeness"), mas
los skip connections que Dong et al. identifican como la fuerza contraria.
**Skip = no aplicar el resolvente al ambiente** — el residual stream se
conserva y la superposicion se relee, que es la estrategia "pure" de
nuestro harness (decodificar sin corregir, tabla Exp 18: pure 0.985 ~ dual
0.985; el paper ya notaba que ni siquiera hace falta pinv, la misma M^-1
funciona si la colocacion es la correcta).

**Sintesis:** los cuatro mecanismos del transformer — ortogonalizacion
aprendida de keys (evita el hard edge), skip connections (no aplica el
resolvente), softmax (unbinding suave con pesos normalizados), multi-head
(promedia canales) — son exactamente las cuatro estrategias de supervivencia
que el teorema clasifica. El transformer funciona porque evita caer en el
regimen donde el fallo ambient existe; cuando el contexto crece y las keys
se solapan (rank collapse in width), cae — y el teorema dice exactamente
por que y cual es la reparacion canonica.

## 4. La direccion inversa: Hrrformer y GHRR (lo que la literatura hizo)

- **Hrrformer** (Alam et al., AISTATS 2023): reemplaza la atencion por
  binding/unbinding HRR explicito — O(T·H·logH) en vez de O(T²H).
  Demuestra que el unbinding VSA es una atencion viable. **Estado:
  publicado, replicado** — valida la direccion del diccionario
  (transformer contiene VSA-computacion).
- **GHRR** (Jiang et al. 2024, arXiv:2405.09689): binding no-conmutativo
  U(m) que **implementa atencion**; reemplazan atencion de un transformer
  por binding GHRR y mejoran LM. Validacion de que binding contiene
  atencion en expresividad.
- **Nuestro aporte reciproco:** el teorema da la condicion de estabilidad
  que ninguno de esos trabajos enuncia: el unbinding es estable **solo si**
  el resolvente vive en espacio de coeficientes (dual), o si el codebook es
  cuasi-ortogonal (regimen sin hard edge). Hrrformer funciona porque su
  superposicion beta se decodifica por correlacion (cleanup = espacio de
  coeficientes, no ambiente).

## 5. Aplicaciones concretas (pre-registradas, falsables)

**P1 — Diagnostico de arquitectura:** medir lambda_min de la gram de keys
por capa y contexto (instrumento: SVD streaming). Prediccion: las capas
con lambda_min·T² < c corresponden a los heads que la literatura de pruning
identifica como redundantes. Falsable contra interpretabilidad estandar.

**P2 — Fix dual para contexto largo:** en lugar de atencion linealizada
(low-rank, que segun Dong acelera el colapso), decodificar por proyeccion
dual C^T M^+ C con C = codebook de keys cuantizadas (memory layers —
Dhayalkar propone "hyperdimensional memory layers"; nuestra prediccion da
la condicion de estabilidad de esas capas: la decodificacion de la memory
layer DEBE ser dual, no ambient, o la capa hereda el colapso).

**P3 — Limite de capacidad del contexto:** el numero de roles
decodificables con errores acotados es ~ n < d·sqrt(lambda_min) (capacidad
del frame); mas alla, el unbinding ambient degrada. Prediccion: el
"effective context length" empirico de LLMs deberia escalar como d por
(calidad espectral de keys), no como d. Medible contra las curvas de
perdida de contexto largo publicadas.

**P4 — Pandora/L2 (la conexion con el programa madre):** la proyeccion a
lenguaje del estado del grafo (decodificador L2) es un problema
superposicion->simbolos: si se implementa como VSA cleanup con codebook
aprendido, el teorema exige que el decodificador respete la colocacion
dual. Disenar el L2 de Pandora con unbinding dual es la aplicacion directa
— y conecta con la fila "decodificador L2 = cuello de botella" del estado
de Pandora en HORIZON.

## 6. Estado epistemico de TODO el analisis (protocolo)

| Afirmacion | Estado |
|---|---|
| Lectura VSA del transformer (queries=roles, etc.) | **establecida** (literatura; correspondencia formal con nuestro harness) |
| Rank collapse in depth (Dong) / in width (Nait-Saada) | **probado (literatura)** |
| "Rank collapse in width = fallo ambient con hard edge" | **PREDICCION FALSABLE** (nuestra; consistente con ambas pruebas previas pero no deducida de ellas como teorema — exige el puente keys~frame cuadrado, plausible pero no demostrado para keys entrenadas) |
| Fix dual restaura sin tocar espectro | **PREDICCION FALSABLE** (P2) |
| Skip = estrategia "pure"; keys ortogonales = frame sin colapso | **correspondencia verificada** en nuestro harness (Exp 18/20A) — lectura interpretativa de los mecanismos del transformer |
| Hrrformer/GHRR validan transformer-contiene-VSA | **probado (literatura)** |
| Aplicaciones P1-P4 | **predicciones pre-registradas** |

**Que NO afirmamos (protocolo 1.4/4.7):** no afirmamos que los heads reales
implementen literalmente M^-1·x (la circuit-level interpretabilidad de eso
esta abierta); no afirmamos que el fix dual sea practicamente mejor que
remover outliers (es una prediccion, no un resultado); no extendemos el
teorema a keys entrenadas sin la hipotesis puente (keys ~ frame
cuadrado/cuasi-aleatorio en regimen de interferencia), que queda declarada
como hipotesis explicita de P1-P3.

## 7. Referencias (verificadas contra fuente en esta sesion)

1. Dhayalkar, S.R. (2025). *Attention as Binding: A Vector-Symbolic
   Perspective on Transformer Reasoning.* arXiv:2512.14709.
2. Dong, Y., Cordonnier, J.-B., Loukas, A. (2021). *Attention is not all
   you need: pure attention loses rank doubly exponentially with depth.*
   ICML 2021, arXiv:2103.03404.
3. Nait-Saada, K. et al. (2025). *Mind the Gap: a Spectral Analysis of Rank
   Collapse and Signal Propagation in Attention Layers.* AISTATS 2025,
   arXiv:2410.07799.
4. Alam, M. et al. (2023). *Recasting Self-Attention with Holographic
   Reduced Representations (Hrrformer).* AISTATS 2023, PMLR v202.
5. Jiang, H. et al. (2024). *Generalized Holographic Reduced
   Representations.* arXiv:2405.09689 (GHRR; binding contiene atencion).
6. Plate, T. (1995). *Holographic Reduced Representations.* IEEE Trans. NN.
7. Nieto, L.B. (2026). *Operator placement, not representation...*
   (nuestro paper; tablas Exp 18/20A/27/28 son las verificaciones).
8. Chen, Y., Liu, D.-Z., Zhou, D.-S. (2010). arXiv:1002.3975 (anclaje del
   hard edge — citado en Exp 28).
