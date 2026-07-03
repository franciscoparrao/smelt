# Causal meta-learners (T/S/X/R/DR-learner) — 2026-07-03

## Por qué

El usuario pidió "implementar los algoritmos SOTA" sin especificar dominio
(vía `/plan`). Se investigaron 4 alternativas con datos reales de
esfuerzo/factibilidad (exploración vía subagentes Explore/Plan):

| Alternativa | Esfuerzo | Veredicto |
|---|---|---|
| **Causal: meta-learners (X/R/DR-learner)** | pequeño-mediano | **elegido** |
| Espacial: MGWR-like sobre GeoXGBoost | pequeño-mediano | descartado — el usuario indicó que GeoXGBoost debe conversarse primero con George Grekousis (colaborador/revisor externo del paper asociado, ver `paper/reply_grekousis_*.txt`); no se debe diseñar/implementar unilateralmente. Ver memoria `project_geoxgboost_review_george` |
| Cerrar gaps del audit (DART, EFB, ordered boosting) | mediano | técnicas de 2017-18, no "SOTA" estricto |
| Deep learning tabular / espacial neuronal (TabNet, NODE, GNNWR) | bloqueo fundacional | requiere candle/burn desde cero (confirmado: cero infraestructura de autodiff en el crate) — proyecto aparte |

Se eligió causal meta-learners porque: (a) es genuinamente SOTA en el
sentido pedido (Künzel et al. 2019, Nie & Wager 2021 — literatura activa
de ML causal, a diferencia de los gaps del audit que son de 2017-18), (b)
no involucra ninguna colaboración externa sensible, (c) esfuerzo
pequeño-mediano, reutilizando patrones ya probados en el motor.

Varias preguntas de aclaración durante la planificación quedaron sin
respuesta del usuario (4 rondas de `AskUserQuestion` sin respuesta). Las
decisiones de diseño abiertas se resolvieron con la opción recomendada por
el agente de diseño en cada caso — documentadas abajo para que se puedan
redirigir fácilmente.

## Decisiones de diseño

1. **API standalone (`estimate()`), no `Learner`.** Cada meta-learner
   necesita 3 inputs alineados (`features`, `treatment`, `outcome`), no el
   par `(X, y)` de `Learner::train_regress(&RegressionTask)`. Forzar
   `treatment` como columna de feature rompe `predict()` para CATE (que
   necesita evaluar ambos brazos, no "predecir target desde features +
   treatment-que-resulta-estar-ahí"). `CausalForest` ya estableció este
   mismo patrón (`estimate(features, treatment, outcome, feature_names)`).
   El argumento "`Learner` da `get_params`/`set_params` gratis" no aplica:
   `declare_params!` (la macro de smelt-py) no depende de `Learner`.
2. **Arrays crudos, sin un tipo `CausalTask` nuevo.** Mismo criterio que
   `CausalForest::estimate` — un `CausalTask` no compra reutilización real
   hoy (ningún `Resample`/`Measure`/tuning sabe consumir un concepto de
   tratamiento). Internamente cada meta-learner arma `RegressionTask`/
   `ClassificationTask` ordinarios por subconjunto de filas.
3. **Incluir DR-learner** además de T/S/X/R-learner: una vez existe el
   scaffold de cross-fitting para R-learner, DR-learner es barato y de
   hecho más fiel a su propia literatura (regresión final sin ponderar).
4. **R-learner: regresión no ponderada, documentada.** El motor no tiene
   soporte genérico de pesos por muestra (`with_sample_weights` solo existe
   en `XGBoost`/`GeoXGBoost`, fuera del trait `Learner`). Se filtran filas
   con residuo de tratamiento cercano a cero (`residual_clip`, default
   1e-3) y se hace regresión no ponderada sobre el pseudo-target — pierde
   algo de eficiencia estadística vs. el R-loss ponderado del paper de Nie
   & Wager, pero mantiene la propiedad quasi-oracle de cross-fitting.
   Documentado en un comentario de doc largo en `r_learner.rs`.

## Qué se construyó

```
src/causal/
  mod.rs                    (existente, +`pub mod meta_learners;`)
  meta_learners/
    mod.rs                  LearnerFactory, MetaLearnerResult, validate_causal_inputs,
                             fixtures de test sintéticos (synthetic_linear_cate,
                             synthetic_confounded_nonlinear_cate)
    cross_fit.rs             oof_regression, oof_propensity, oof_regression_by_arm
                             (generaliza el loop OOF que Stacking::train_regress ya tenía)
    t_learner.rs              TLearner: dos modelos por brazo, sin cross-fitting
    s_learner.rs              SLearner: un modelo con T como feature aumentada
    x_learner.rs              XLearner: T-learner + imputación + propensity blend
    r_learner.rs              RLearner: residuo-sobre-residuo vía cross-fitting
    dr_learner.rs             DrLearner: pseudo-outcome doblemente robusto
```

Además: `Prediction::CausalEffect { estimated, true_effect }` (nueva
variante, `src/prediction/mod.rs`), `Pehe`/`AteBias` (nuevas medidas,
`src/measure/mod.rs`), wiring en el prelude de `src/lib.rs` (incluyendo
`CausalForest`, que faltaba del prelude desde antes — gap preexistente
corregido de paso).

**Fix incidental en smelt-py**: `common.rs::predict_values` tenía un match
exhaustivo sobre `Prediction` que no compilaba tras agregar la variante
`CausalEffect` — se agregó el brazo correspondiente (los meta-learners no
están expuestos a Python, pero el match debía seguir siendo exhaustivo).

## Validación

- `cargo build --workspace`: limpio.
- `cargo test -p smelt-ml --lib`: 95 tests verdes (74 preexistentes + 21
  nuevos de `meta_learners`), 0 regresiones.
- `cargo test -p smelt-ml --doc`: 66 doctests verdes (61 preexistentes + 5
  nuevos, uno por meta-learner).
- `cargo test --test integration`: 272 tests verdes, sin cambios.
- `cargo check -p smelt-py`: limpio tras el fix del match exhaustivo.
- Cada meta-learner tiene tests con dos fixtures sintéticas compartidas
  (`synthetic_linear_cate`: efecto lineal heterogéneo, propensity RCT;
  `synthetic_confounded_nonlinear_cate`: efecto no-lineal, propensity
  confundida) evaluadas con PEHE/AteBias contra el CATE verdadero conocido,
  más tests de rechazo de inputs inválidos (tratamiento no-binario, un solo
  brazo, dimensiones desalineadas).
- Un test inicial de S-learner falló (`LinearRegression` como base sobre el
  fixture con efecto heterogéneo `tau=2*x0`): un modelo lineal aditivo
  `[X, T]` no puede representar ninguna interacción `X*T`, es un techo de
  especificación del modelo, no solo el sesgo "regularized away" de Künzel
  et al. — corregido usando `RandomForest` como base para ese test (igual
  que T/X/R/DR-learner) y agregando un test separado con `LinearRegression`
  sobre un efecto constante (sin interacción que perder), documentado en el
  comentario del módulo.

## Fuera de alcance (seguimiento futuro)

- Soporte genérico de pesos por muestra en `Learner`/`RegressionTask`
  (permitiría R-learner fielmente ponderado).
- Estimación de error estándar/CI por bootstrap (`CausalForest` tiene su
  propio jackknife infinitesimal; los meta-learners no tienen ese
  mecanismo).

## Actualización 2026-07-03 (misma sesión): bindings Python

Se agregaron los bindings de `smelt-py` que el punto anterior dejaba como
seguimiento — `smelt-py/src/causal.rs` (nuevo módulo), con `TLearner`,
`SLearner`, `XLearner`, `RLearner`, `DrLearner` como clases PyO3.

- **Mismo patrón que `Bagging`/`Stacking`/`DynamicEnsemble`** (no
  `define_learner!`, que asume la forma `(X, y)`; no `declare_params!`
  genérico tampoco): los learners base se seleccionan por **id string**
  (`smelt.registered_learner_ids()`), validados eagerly en el constructor Y
  revalidados en `set_params` — se movió `validate_learner_id` de privada a
  `pub(crate)` en `learners/ensemble.rs` para reutilizarla, en vez de
  duplicarla.
- `estimate(x, treatment, y)` devuelve un dict `{"cate": ndarray, "ate":
  float}` vía un helper compartido `meta_learner_result_to_dict`.
- `XLearner`/`RLearner`/`DrLearner` exponen sus hiperparámetros propios
  (`propensity_clip`, `cv_folds`, `cv_seed`, `residual_clip`) como kwargs
  con default, iguales a la API de Rust.
- Compiló limpio al primer intento (`cargo build -p smelt-py`).
- **Gap de re-export descubierto en el smoke test**: las clases se
  registraban en el módulo nativo `_smelt` pero `smelt/__init__.py` (el
  wrapper puro-Python) no las re-exportaba — mismo patrón que cada learner
  existente requiere (import explícito + entrada en `__all__`). Corregido.
- Validación: `maturin develop --release` + smoke test en Python cubriendo
  las 5 clases (`estimate`/`get_params`/`set_params` round-trip, PEHE
  calculado contra el CATE verdadero de un fixture sintético) y los paths
  de error (id de constructor inválido, id de `set_params` inválido, clave
  de `set_params` desconocida, tratamiento no-binario) — todos fallan
  limpio, no panic. `cargo test -p smelt-ml --lib`/`--test integration`
  siguen en 95/272 verdes, sin regresiones.
