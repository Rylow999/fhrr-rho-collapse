# Compilar el paper

Dependencias: `tectonic` (recomendado) o `pdflatex` + `bibtex`.

## Con tectonic (una pasada autocontenida)

```bash
cd paper
tectonic main.tex
```

Genera `main.pdf`. Tectonic se encarga de los references sin necesidad de
correr bibtex por separado.

## Con pdflatex

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Las figuras usadas están en `paper/figures/` (se generan desde
`src/fig_exp*.py`). Si regenerás un experimento, volvé a correr el
generador de la figura correspondiente y recompilá.
