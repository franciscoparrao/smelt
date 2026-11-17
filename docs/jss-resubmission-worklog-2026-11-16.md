# Bitácora: respuesta al prescreening de JSS (desde 2026-11-16)

Comentarios de los editores: `docs/jss-prescreening-comments-2026-11-16.txt`. Decisión de
venue pendiente (ranking en curso con `/paper-match`). Lo de abajo es independiente del venue.

## Hecho el 2026-11-16 (paquete de replicación)

- `paper/replication/requirements.txt`: nuevo. Pins exactos de las librerías de comparación con
  las versiones con que se produjeron las cifras del paper (scikit-learn 1.8.0, xgboost 3.1.1,
  lightgbm 4.6.0, catboost 1.2.8, geoxgboost 1.1.1, smelt-ml 0.8.1), pisos para numpy, pandas,
  scipy, matplotlib. Nota: el venv actual de `smelt-py/.venv` ya derivó (sklearn 1.9.1, xgboost
  3.0.0, catboost 1.2.10); los números de la Tabla 13 de abajo se recalcularon con ese venv.
- `paper/replication/replicate.sh`: autocontenido. Si no encuentra `Cargo.toml` de smelt-ml dos
  niveles arriba (el caso del editor, que extrajo solo la carpeta), clona el tag `SMELT_REF`
  (defecto `paper-jss-submission`) en `./smelt-src`; si el tag no está en el remoto, clona la
  rama por defecto y lo avisa. Intérprete configurable con `PYTHON=`. Comprueba `cargo`, `git` y
  las dependencias Python antes de empezar. Probado fuera del repo: detecta la ausencia del
  crate y clona (el tag aún no está publicado, ver pendientes).
- `paper/replication/README.md`: "Quick start" con un solo punto de entrada y tabla "Map from
  manuscript to scripts" (cada tabla y figura → paso de `replicate.sh` → archivo de salida).
- `examples/case_study_walkthrough.rs`: nuevo. Listado completo y ejecutable del caso de
  estudio con la API actual (`with_coords`, `GeoXGBoost::coords_from_task()`,
  `resample_regress` con `SpatialBlockCV` y `MoranResiduals`, `SplitConformal`). Es el código
  que irá tal cual en la sección del caso de estudio, con su salida
  (`paper/replication/case_study_walkthrough_output.txt`, paso D1). Reemplaza al pseudocódigo
  de la sección de inicio rápido, que tenía variables sin definir (objeción justificada).
- `examples/accuracy_validation.rs` y `paper/replication/accuracy_sklearn.py`: fila de
  regresión logística en Digits (10 clases, one-vs-rest) y configuración emparejada del lado
  scikit-learn (estandarizada dentro del fold, sin penalización, OvR) junto a la de defecto.

## Cifras nuevas

### Tabla 2 (caso de estudio, `case_study_walkthrough`, coordenadas UTM 10N en metros)

| | XGBoost | Geo-XGBoost (k = 50, alpha 0,5) |
|---|---|---|
| Hold-out 80/20, RMSE log precio | 0,330 | 0,260 (parte global sola: 0,330) |
| CV aleatoria 5 folds, RMSE / Moran's I residual | 0,323 / +0,371 | 0,249 / +0,212 |
| CV bloques espaciales 5 folds, RMSE / Moran's I residual | 0,357 / +0,194 | 0,298 / +0,235 |
| Conformal 90 % sobre Geo-XGBoost | | cobertura 88,0 % (176/200), ancho 0,77 log($) |

Diferencias con la Tabla 2 publicada: ahora Geo-XGBoost aparece en la comparación de CV (antes
solo XGBoost), las coordenadas son proyectadas (antes grados, lo que distorsiona los bloques y el
kernel), 5 bloques en vez de 4, y el conformal envuelve a Geo-XGBoost (antes a XGBoost). La
cobertura 88 % con 200 puntos de calibración está dentro de la holgura de muestra finita del
split conformal; decirlo en el texto.

### Tabla 13 (validación de precisión, 5-fold CV, semilla 42)

| Dataset | Learner | smelt-ml | sklearn defecto | sklearn emparejado (std, sin penalización, OvR) |
|---|---|---|---|---|
| Wine | Logistic | 0,989 | 0,955 | 0,972 |
| Breast Cancer | Logistic | 0,981 | 0,949 | 0,953 |
| Digits (10 clases) | Logistic | 0,952 | 0,967 (multinomial, L2) | 0,929 (OvR) |

Respuesta al editor sobre la estandarización: tiene razón en que la estandarización por sí sola
no cambia el óptimo de un modelo convexo. Lo que cambia el resultado es (a) la penalización L2 de
defecto de scikit-learn, que sí depende de la escala, y (b) que con `max_iter=1000` sobre
características sin escalar lbfgs no converge (ConvergenceWarning en Wine y Breast Cancer, la
fila "defecto" publicada era una solución no convergida). smelt-ml ajusta descenso de gradiente
sin penalización sobre características estandarizadas, OvR en multiclase; en Wine y Breast
Cancer (casi separables) el MLE sin penalización no es finito y ambas implementaciones devuelven
una solución limitada por iteraciones, de ahí las diferencias residuales de 0,02 a 0,03 incluso
emparejando. La frase del paper "due to automatic feature standardization" se reemplaza por esta
explicación, y la tabla incluye la fila de Digits con el esquema de ajuste declarado.

## Pendiente

- Decidir venue (ranking en curso). Las ediciones del manuscrito (sección de caso de estudio
  con listado y salida, Tabla 2, Tabla 13, modelo estadístico, revisión de software con mboost,
  Tabla 6 con geographical random forest, no mezclar boosting y RF) dependen del formato.
- Publicar el tag `paper-jss-submission` en GitHub (solo está local; `replicate.sh` cae a la
  rama por defecto mientras tanto). Para la resubmisión habrá un tag nuevo.
- Volver a correr la Tabla 13 con el venv de versiones fijadas (`requirements.txt`) antes de
  publicar las cifras.
- Repetir Tablas 9 a 12 solo si el venue las quiere (JSS dijo que no le interesan).
