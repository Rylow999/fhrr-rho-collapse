# Umbral critico κ* — derivacion (empirica + teorica)

## Mediciones directas (Exp 14, BundleV3 HRR real)

| BLK | rho | lambda_min | lambda_max | kappa | gram acc |
|-----|------|-----------|-----------|-------|----------|
| 40 | 0.800 | 2.13e-2 | 3.20 | 1.5e2 | 0.991 |
| 36 | 0.889 | 9.08e-3 | 3.46 | 3.8e2 | 0.986 |
| 33 | 0.970 | 1.76e-3 | 3.39 | 1.9e3 | 0.991 |
| **32** | **1.000** | **4.87e-4** | **3.52** | **7.2e3** | **0.146** |
| 31 | 1.032 | (singular) | — | ~1e16 | 0.995 |
| 28 | 1.143 | (singular) | — | ~1e16 | 0.975 |

## Modelo del mecanismo

El decoder gram aplica M^-1 sobre el candidato del unbind. El ruido de
crosstalk entre roles ~ sigma_R² ~ (r-1)/BLK (r = roles por bloque).

Tras la inversa: ruido amplificado = sigma_R² / lambda_min.

El cleanup argmax necesita que el margen del simbolo correcto supere el
ruido amplificado:

    gap_tipico ~ 1/sqrt(BLK)    (separacion de codebook aleatorio gaussiano)

    fallo si:   sigma_R / lambda_min  >  gap_tipico

    -> lambda_min_crit  ~  sigma_R / gap
                      ~  sqrt((r-1)/BLK) * sqrt(BLK)
                      ~  sqrt(r-1)   (independiente de BLK!)

  Para r = 2 (caso tipico): lambda_min* ~ 1

Pero el punto de fallo medido tiene lambda_min = 4.87e-4 (lam_max ~ 3.5),
que es MUCHO menor que 1. La iteracion del resonator acumula el error
(no es un solo shot), lo que baja el umbral real:

    lambda_min* ~ sigma_R / (gap * sqrt(T))

Con T=50 iteraciones: lambda_min* ~ sqrt((r-1)/BLK) * sqrt(BLK)/sqrt(T)
                                 = sqrt(r-1)/sqrt(T) = 1/7.07 = 0.14

Tampoco llega a 4.87e-4. El modelo simple subestima el colapso, lo que
sugiere que la **iteracion resonator amplifica el error mas alla del ruido
a primer orden** (feedback positivo dentro del resonator cuando la Gram
esta cerca de singular).

## Ley empirica del umbral

    Una decodificacion Gram basada en resonator falla cuando

        kappa(M)  ∈  [1e3, 1e5]

    con singularidad completa (colapso de ~85% de accuracy) en el punto
    donde el espectro cruza la singularidad sin disparar el truncamiento
    de la pseudo-inversa.

## Implicacion para diseno

    En VSA con decodificacion basada en Gram inversa, elegir codebooks y
    dimensiones de bloque tales que la Gram sea o bien:
      a) bien condicionada:      kappa < 1e2
      b) claramente singular:    kappa > 1e15  (y usar pinv con rcond)

    La banda intermedia [1e3, 1e5] es un pozo de fallo evitable con
    resonator puro (sin correccion) o decoder aprendido.

