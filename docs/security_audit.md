# Auditoría de Seguridad — smelt-ml v1.2.0

**Fecha**: 2026-04-01
**Auditor**: Claude Security Engineer
**Scope**: Librería Rust completa (13,377 líneas, 40 archivos)

## Resumen Ejecutivo

- Critical: **0** | High: **0** | Medium: **2** | Low: **3**
- Acción recomendada: **Cuando sea posible** — no hay vulnerabilidades de seguridad activas.

Esta es una **librería de ML pura** (no un servidor web), por lo que la superficie de ataque es limitada a: procesamiento de datos de entrada (CSV), deserialización (JSON), y operaciones matemáticas.

## Puntos Fuertes de Seguridad

- **0 unsafe**: 13,377 líneas sin un solo bloque `unsafe`
- **0 secretos hardcodeados**: escaneo limpio
- **0 dependencias con CVEs**: `cargo audit` limpio
- **0 SQL/command injection**: no aplica (librería pura)
- **0 archivos sensibles en git**: .env, .pem, .key — ninguno
- **Error handling consistente**: `Result<T>` con `SmeltError` en toda la API
- **No hay estado global mutable**: sin `static mut`, sin `lazy_static!`

---

## Hallazgos

### [MEDIUM] S1: Deserialización JSON sin límite de tamaño

- **Ubicación**: `src/serialize.rs:93`
- **OWASP**: A08 Data Integrity Failures
- **Descripción**: `load_json()` lee un archivo completo a string y lo deserializa sin límite de tamaño. Un archivo JSON maliciosamente grande podría causar OOM.
- **Impacto**: Denial of Service local si un usuario carga un archivo JSON de varios GB como "modelo serializado".
- **Riesgo real**: Bajo — el usuario controla qué archivos carga. No hay endpoint de red expuesto.
- **Fix propuesto**:
```rust
pub fn load_json(path: impl AsRef<Path>) -> Result<SerializableModel> {
    let metadata = fs::metadata(&path)?;
    if metadata.len() > 100_000_000 { // 100MB limit
        return Err(SmeltError::Other("Model file too large (>100MB)".into()));
    }
    let json = fs::read_to_string(path)?;
    SerializableModel::from_json(&json)
}
```

### [MEDIUM] S2: CSV parsing sin límite de filas/columnas

- **Ubicación**: `src/data/mod.rs:49-70`
- **Descripción**: `CsvLoader` lee todo el CSV en memoria sin límite. Un CSV de 10GB causaría OOM.
- **Impacto**: Denial of Service local.
- **Riesgo real**: Bajo — el usuario controla qué archivos carga.
- **Fix propuesto**: Agregar `.with_max_rows(n)` opcional al builder.

### [LOW] S3: Division por cero potencial en SHAP y Imputer

- **Ubicación**: `src/importance/shap.rs:93`, `src/preprocess/imputer.rs:76`
- **Descripción**: `bg_vals.len() as f64` y `non_nan.len() as f64` podrían ser 0 si no hay background samples o si todos son NaN, produciendo `NaN` o `Inf`.
- **Impacto**: Resultado numérico incorrecto (NaN propagado), no crash.
- **Fix**: Guard check `if len == 0 { return default; }` antes de la división.

### [LOW] S4: `as usize` truncación de float negativo

- **Ubicación**: `src/conformal/mod.rs:91`, `src/resample/spatial.rs:62-63`
- **Descripción**: `(value as f64).ceil() as usize` con un valor negativo produce `usize::MAX` (wrapping). Esto podría ocurrir con alpha > 1.0 o coordenadas negativas.
- **Impacto**: Index out of bounds panic.
- **Fix**: Validar inputs en los constructores o usar `.max(0.0)` antes del cast.

### [LOW] S5: 564 accesos a arrays por índice sin bounds check explícito

- **Descripción**: Rust paniquea en bounds check por defecto (no es un security issue — es un crash, no un buffer overflow). Pero en una librería, panics son indeseables.
- **Impacto**: Los dimension checks en `predict()` cubren la mayoría de los casos. Los panics restantes son internos (e.g., `features[[i, j]]` donde i/j son derivados del tamaño del array).
- **Riesgo real**: Muy bajo — Rust garantiza memory safety.

---

## Lo que NO es un problema

| Pattern | Resultado | Nota |
|---------|:---:|------|
| `unsafe` | 0 | Memory safe garantizado |
| Secretos hardcodeados | 0 | No aplica |
| SQL injection | N/A | No hay SQL |
| Command injection | N/A | No ejecuta comandos |
| XSS | N/A | No hay frontend |
| CORS | N/A | No hay HTTP server |
| JWT/Auth | N/A | No hay autenticación |
| Dependencias con CVEs | 0 | `cargo audit` limpio |

---

## Recomendaciones de Hardening

1. **Agregar límites de tamaño** a `load_json()` y `CsvLoader` — previene OOM con inputs maliciosos
2. **Guard checks de división por cero** en SHAP y Imputer — previene NaN propagado
3. **Validar rangos de parámetros** en constructores (alpha ∈ [0,1], k > 0, n_estimators > 0) — algunos ya se validan, completar los que faltan
4. **Agregar `#[must_use]` a Result** en funciones públicas que retornan Result — previene errores ignorados por el usuario

## Herramientas utilizadas

- `cargo audit` — 0 CVEs
- Análisis manual con grep patterns (secretos, SQL injection, command injection, deserialización, path traversal)
- Revisión de código fuente completa
