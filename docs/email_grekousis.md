# Email Draft: George Grekousis — GeoXGBoost in Rust

**To:** george.grekousis@gmail.com (o buscar email institucional actualizado)
**Subject:** Rust implementation of Geographical-XGBoost — potential collaboration

---

Dear Professor Grekousis,

I am Francisco Parra, a researcher at the Department of Geographic Engineering, Universidad de Santiago de Chile. I am writing because I have implemented your Geographical-XGBoost algorithm (Grekousis, 2025, Journal of Geographical Systems) as part of a Rust-based machine learning framework called smelt-ml.

The implementation includes the core components of your method: bi-square spatial kernel weights, adaptive bandwidth via k-nearest neighbors, per-location local XGBoost models, and the global–local ensemble with adaptive α weighting. It is, to my knowledge, the first implementation of Geographical-XGBoost outside your original Python package.

smelt-ml integrates GeoXGBoost with spatial cross-validation (SpatialBlockCV) and conformal prediction in a single composable pipeline — enabling spatially-aware prediction with distribution-free uncertainty quantification without Python or R dependencies. The framework is published on crates.io (https://crates.io/crates/smelt-ml) and the source code is available at https://github.com/franciscoparrao/smelt.

I am currently preparing a manuscript for the Journal of Statistical Software describing the framework. A case study on the King County housing dataset demonstrates that spatial structure accounts for a 32% improvement in prediction accuracy, and that standard cross-validation underestimates error by 31% compared to spatial block CV — consistent with Roberts et al. (2017).

I am reaching out for two reasons:

1. **Validation**: I would appreciate any feedback on whether the implementation captures the essential aspects of your method correctly. I am happy to share the source code for your review.

2. **Collaboration**: If you find the work interesting, I would welcome the possibility of a co-authorship on the JSS manuscript. Your expertise would strengthen the spatial modeling sections considerably.

I understand you may be busy, and I appreciate any level of engagement — even a brief confirmation that the implementation approach is sound would be valuable.

Thank you for your pioneering work on Geographical-XGBoost. It fills an important gap in spatial machine learning.

Best regards,

Francisco Parra
Departamento de Ingeniería Geográfica
Universidad de Santiago de Chile
francisco.parra.o@usach.cl
https://github.com/franciscoparrao/smelt

---

## Notas internas (no enviar)

- Grekousis está en University of the Aegean, Grecia (verificar afiliación actual)
- Su paper tiene DOI: 10.1007/s10109-024-00449-w
- El paquete geoxgboost está en PyPI: https://pypi.org/project/geoxgboost/
- Si responde positivamente: compartir acceso al repo, enviar draft del paper
- Si no responde en 2 semanas: enviar follow-up breve
- Si declina: agradecer y continuar como single-author, mantener la citación
