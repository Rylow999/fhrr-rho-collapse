# Roadmap — estado post-falsacion (2026-09-21)

Doc historico del estado de la agenda de falsacion. Verificacion cruzada en
`verify.py` (31/31 checks). Para el estado actual del proyecto ver README.md.

## Estado: COMPLETADO

Toda la agenda de falsacion fue ejecutada. La Frame-Dual Stability Principle
(Prop. 1 del paper) sobrevivio a todas las capas planeadas:

- [x] **Capa 0 (sanidad)**: Exp 18 replica from-scratch en Exp 19.
  \- [x] **Capa 0 extra**: Exp 18b BSC, Exp 27 FHRR (cierre del circulo en el
  algebra originaria). Bloqueante FHRR-causal resuelto.
- [x] **Capa 1 (ensembles)**: 6 familias de C (gauss/radem/sphere/toeplitz/
  orthogonal/near-dup). near-dup = control negativo correctamente falla.
- [x] **Capa 1b (barrido rho)**: dims enteras distintas (era el bug: 0.99 y
  1.00 compartian d=32).
- [x] **Capa 2 (ruido/amplificacion)**: Exp 21 lineal puro + Exp 22 dinamica.
- [x] **Capa 3 (operadores)**: zoo completo (inv/pinv/Tikhonov/truncSVD).
- [x] **Capa 4 (frames clasicos)**: identificacion como canonical dual frame
  (Duffin-Schaeffer). Referencias en la bibliografia del paper.

## Lecciones aprendidas (historico)

- La banda kappa [1e3, 1e4] fue refutada por Exp 17 (muchos seeds con
  kappa bajo el umbral colapsan igual). Esa teoria vivia en la doc vieja.
- El shape guard de BundleV3 hace que gram ≡ pure fuera de rho=1: toda
  la estructura fuera del punto cuadrado era trivial.
- La normalizacion de MP en la primera tabla estaba mal (dividida por d de
  mas). Corregida.
- r=-0.38 escrito de memoria cuando el dato era -0.259. Nunca mas: todo
  numero del paper pasa por verify.py.
- La réplica from-scratch requiere que C tenga las filas RIGHT (no tiles).

## Que quedaria para llevar el principio a teorema

El proximo escalon formal (fuera del scope de este repo): probar que la
estabilidad C' T C se extiende a operadores T no-espectrales del Gram.
Eso es trabajo matematico distinto, no experimental.
