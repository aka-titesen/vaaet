---
name: vaaet-eda-data-storytelling
description: Explorá y auditá datos VAAET de forma reproducible, sin fuga ni deriva de contratos. Usá esta skill para EDA de telemetría, perfiles de calidad, nulos, outliers, correlaciones, muestreo, gráficos narrativos o decisiones de ingeniería de features basadas en evidencia.
---

# EDA VAAET y narrativa de datos

## Propósito y alcance

Usá el análisis exploratorio de datos (EDA) para formular y comprobar preguntas sobre la telemetría de VAAET, detectar problemas de calidad y producir evidencia legible para decisiones humanas. Mantené los notebooks como orquestadores finos: la lógica reutilizable y testeable pertenece a `src/vaaet/`.

Esta skill guía análisis futuros; no afirma que VAAET ya tenga un notebook, un pipeline o utilidades genéricas de EDA. Las funciones `plot_training_history()` y `plot_training_evaluation()` de `vaaet.evaluation.reporting` sirven para evaluar entrenamiento: no las presentes como un subsistema EDA completo.

Antes de modificar una exploración, leé `AGENTS.md` y las ADR aplicables, en especial ADR-0013 a ADR-0019. Los contratos, snapshots inmutables, input locks y holdouts humanos congelados prevalecen sobre una hipótesis exploratoria.

## Preservá datos, cohortes y contratos

- Partí de un snapshot, catálogo y `pipeline_run` identificables. Registrá procedencia, versión, filtros, intervalo temporal, clips incluidos y semilla de muestreo sin exponer rutas privadas ni secretos.
- Definí por escrito la pregunta, la cohorte de análisis y la decisión que podría informar antes de graficar. Separá por clip y tiempo cuando la observación pueda repetirse dentro de un video.
- No inspecciones validation ni test congelados para elegir features, umbrales, transformaciones o arquitectura. Usá únicamente la cohorte de desarrollo autorizada; el holdout humano sólo admite evaluación final compatible.
- Conservá exactamente las 19 `FEATURE_COLS`, su orden y los contratos de `vaaet.data.datasets`, `vaaet.features`, `vaaet.training` y los bundles. Solicitá autorización y un ADR antes de proponer cambios a features, etiquetas, umbrales, datos gobernados o MLP.
- Reconocé que `Normal`, `Reduced` y `Congested` son las salidas aprendidas. `Accident` requiere confirmación humana: no lo infieras, etiquetes ni lo cuantifiques como ground truth automático.
- En `SEED_BOOTSTRAP`, tratá observaciones proxy y sintéticas como evidencia limitada. En `HITL_RETRAINING`, usá etiquetas humanas efectivas y conservá la separación entre datos semilla, HITL y holdouts.

## Ejecutá un EDA reproducible y acotado

1. Verificá esquema, tipos, rango temporal, clips, filas duplicadas, volumen y procedencia contra los contratos antes de calcular métricas.
2. Perfilá nulos, cardinalidad categórica, distribuciones, cuantiles, asimetría y valores no finitos. Clasificá cada anomalía como dato faltante esperado, error de adquisición, valor fuera de dominio o caso pendiente de revisión; no imputés, descartés ni normalicés en silencio.
3. Explorá una variable por vez con estadísticas robustas y visualizaciones legibles. Usá IQR, z-score u otro criterio sólo para señalar candidatos a investigar; un outlier no autoriza su eliminación, relabeling ni cambio de umbral.
4. Explorá relaciones bivariadas o multivariadas con una pregunta concreta. Indicá tamaño muestral, cohortes y limitaciones; asociación no demuestra causalidad.
5. Cerrá con hallazgos accionables, incertidumbres y próximos controles. Separá observación, interpretación e hipótesis.

Muestreá de forma determinista cuando el volumen impida explorar la cohorte completa. Fijá y registrá semilla, método y tamaño; no presentes muestras sesgadas como si representaran toda la distribución. No ejecutes perfiles pesados o gráficos masivos en Colab sin estimar RAM y tiempo.

## Contá la historia con gráficos útiles

- Mostrá tendencias de flujo, densidad, horarios, condiciones de captura o calidad sólo si esas columnas y semántica existen en la cohorte. No inventes variables climáticas, targets ni significados de columnas.
- Incluí título técnico, ejes con unidades, leyenda, escala y paleta accesible. Preferí alternativas que no dependan sólo del color y evitá escalas que exageren diferencias.
- Acompañá cada visual relevante con hipótesis, hallazgo, limitación y decisión posible. Eliminá gráficos que no respondan una pregunta ni cambien una decisión.
- Controlá cardinalidad y sobreimpresión: agregá, estratificá o muestreá de forma documentada antes de renderizar nubes de puntos ilegibles.
- No uses celdas monolíticas ni repitas bloques extensos. Si una rutina se reutiliza, proponé extraerla a `src/vaaet/evaluation/` con pruebas y autorización; no crees un paquete paralelo de visualización.

## Gestioná memoria y salidas con prudencia

- Trabajá sobre copias efímeras de análisis. Permití downcasting sólo después de verificar rangos, precisión y ausencia de pérdida; nunca alteres el dtype contractual, snapshot fuente ni input de entrenamiento para ahorrar RAM.
- Usá Pandas, Matplotlib, Seaborn e ipywidgets ya declarados cuando sean suficientes. Proponé Polars, PyArrow, ydata-profiling, formatos, lockfiles o dependencias sólo con autorización y evaluación para Colab Free.
- No exportes Parquet por defecto ni asumas que forma parte del flujo vigente. Toda salida persistente futura debe respetar APIs, catálogos, snapshots, locks y almacenamiento inmutable existentes; no sobrescribas datasets, punteros, bundles ni catálogos.
- Guardá sólo diagnósticos autorizados y trazables bajo el directorio de una ejecución. No persistás muestras crudas, PII, credenciales, DSNs, rutas privadas ni excepciones completas.

## Entregá un handoff verificable

Presentá cohorte, procedencia, filtros, versión y semilla; calidad observada y tratamiento; y hasta tres hallazgos priorizados, cada uno con evidencia, limitación y siguiente paso humano. Indicá compatibilidad con snapshot, input lock y holdout cuando la exploración se use cerca de entrenamiento o evaluación.

Las metas de velocidad, cobertura de nulos o cantidad de insights son benchmarks que se deben medir en el entorno real; no garantías ni sustitutos de revisión humana.

## Rechazá estos antipatrones

- Analizar un holdout congelado para orientar entrenamiento, o mezclar clips entre cohortes.
- Mutar datos crudos, snapshots, datasets versionados, locks o catálogos durante la exploración.
- Tratar nulos y outliers como basura que se elimina automáticamente.
- Confundir correlación con causalidad, evidencia proxy con evidencia operacional o autopredicciones con etiquetas humanas.
- Instalar dependencias, usar `pip freeze` como fuente declarativa o ejecutar instalaciones dentro de notebooks.
- Crear dashboards pesados, reportes automáticos opacos o gráficos sin propósito que comprometan la RAM de Colab.
- Exponer secretos, PII, rutas privadas o datos sin redactar en artefactos exploratorios.

## Validá cambios relacionados

Cuando una tarea autorizada modifique código, notebooks o documentación vinculados al EDA, ejecutá los gates de VAAET en proporción al cambio:

1. `ruff check src tests scripts`
2. `pytest tests/ -v --tb=short`
3. `python -m compileall -q src tests scripts`
4. Parseo con `ast.parse()` de las celdas de los tres notebooks.
5. Comprobación de enlaces Markdown.
6. `git diff --check`

No agregues notebooks, utilidades, dependencias, datasets, artefactos, DVC, ADRs ni CI sólo por aplicar esta skill. Pedí autorización cuando el hallazgo requiera alterar un contrato, un dato gobernado o una decisión arquitectónica.
