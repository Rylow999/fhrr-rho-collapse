# Tests

Correr:

```bash
python -m pytest tests/ -v
```

Hay 5 tests en `tests/test_pipeline.py`:

1. **test_encode_decode_roundtrip** — un BundleV3 con rho<1 debe recuperar el fact correctamente.
2. **test_pure_survives_square_point** — el decoder `pure` en rho=1 mantiene accuracy alta.
3. **test_gram_collapses_at_square** — el decoder `gram` colapsa en rho=1 (sanitiza la firma del fenomeno).
4. **test_rho_grid_coverage** — un barrido pequeno corre sin errores y reporta accuracies consistentes.
5. **test_kappa_computable** — el numero de condicion de la Gram se calcula y es positivo/finito en rho<1.

Los tests usan BundleV3 (el harness HRR del paper). No tapan la causalidad de
Exp 18; para eso esta el verify.py en la raiz.
