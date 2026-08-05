<!-- context: VAAET/docs/product/feasibility.md — Estudio de factibilidad.
Complementa PRD.md y RISK_MATRIX.md. -->

# Estudio de Factibilidad de Software — VAAET

## Identificación del Proyecto

| Campo | Detalles |
|---|---|
| **Nombre del Proyecto** | VAAET — Video Advanced Analysis of Traffic |
| **Versión** | 4.1.0 |
| **Estado** | Aprobado |
| **Responsable Técnico** | Facundo Nicolás González |
| **Última Revisión** | 2026-07-23 |

---

## 1. Factibilidad Técnica

### 1.1 Viabilidad del Stack Tecnológico

| Tecnología | Madurez | Disponibilidad | Riesgo |
|---|---|---|---|
| YOLO 11 (Ultralytics) | Alta | Open source, modelos pre-entrenados | Bajo |
| TensorFlow/Keras | Alta | Instalación estándar en Colab | Bajo |
| OpenCV | Alta | Librería establecida de visión artificial | Bajo |
| Google Colab | Alta | Gratuito con GPU | Medio (sesiones efímeras) |
| PostgreSQL (AWS RDS) | Alta | Servicio gestionado | Bajo (opcional) |
| SORT Tracking | Alta | Algoritmo simple y probado | Bajo |

### 1.2 Competencias Técnicas Requeridas

| Competencia | Disponible | Brecha |
|---|---|---|
| Python, pandas, numpy | ✅ | — |
| Computer Vision (YOLO, OpenCV) | ✅ | — |
| Machine Learning (TF/Keras) | ✅ | — |
| SQL / PostgreSQL | ✅ | — |
| DevOps / CI/CD | ⚠️ Parcial | GitHub Actions recién implementado |
| MLOps (Model Registry, Drift) | ❌ | Implementación pendiente |

### 1.3 Conclusión Técnica

**FACTIBLE.** El stack tecnológico es maduro, las dependencias son open source, y el entorno de ejecución (Google Colab) es gratuito. Las brechas en MLOps son abordables incrementalmente.

---

## 2. Factibilidad Operativa

### 2.1 Contexto Operativo

| Factor | Estado |
|---|---|
| **Acceso a datos** | Videos SISE disponibles bajo autorización |
| **Infraestructura** | Colab (gratuito) + AWS RDS (costo recurrente bajo) |
| **Equipo** | 1 desarrollador (todos los roles) |
| **Mantenimiento** | Actualización de modelos YOLO, re-entrenamiento del clasificador |

### 2.2 Restricciones Operativas

- Google Colab Free tiene sesiones máximas de ~12 horas y GPU no garantizada
- Los videos de vigilancia SISE son datos restringidos (no se distribuyen con el proyecto)
- El re-entrenamiento requiere acumulación de datos validados por operadores SISE

### 2.3 Conclusión Operativa

**FACTIBLE con restricciones.** El sistema funciona dentro de las limitaciones de Colab Free. La evolución a un backend web eliminará las restricciones de sesión efímera.

---

## 3. Factibilidad Económica

### 3.1 Costos del Estado Actual

| Componente | Costo | Frecuencia |
|---|---|---|
| Google Colab Free | $0 | — |
| GitHub (público) | $0 | — |
| AWS RDS db.t3.micro | ~$15 USD/mes | Mensual (opcional) |
| Dominio (futuro) | ~$12 USD/año | Anual |
| **Total mensual** | **~$15 USD** | — |

### 3.2 Costos de Evolución a Web App

| Componente | Costo Estimado | Frecuencia |
|---|---|---|
| Backend (Railway / Render) | $0-$25 USD/mes | Mensual |
| Frontend (Vercel / Netlify) | $0 | Gratuito |
| AWS RDS (escalado) | $15-50 USD/mes | Mensual |
| GPU para inferencia (futuro) | $50-200 USD/mes | Según demanda |
| **Total estimado** | **$65-275 USD/mes** | — |

### 3.3 Conclusión Económica

**FACTIBLE.** El costo actual es mínimo ($0-15/mes). La evolución a producción web tiene un costo escalable y predecible.

---

## 4. Factibilidad Legal y de Privacidad

### 4.1 Marco Legal

| Aspecto | Estado |
|---|---|
| Licencia del software | MIT (permisiva) |
| Datos de video | Propiedad de SISE/Vialidad Nacional — no distribuidos |
| Privacidad | No se procesan datos personales (patentes, identidades) |
| Dependencias | Todas open source con licencias compatibles |

### 4.2 Conclusión Legal

**FACTIBLE.** El software es MIT, los datos no contienen información personal identificable, y las dependencias tienen licencias compatibles.

---

## 5. Recomendación Final

| Dimensión | Veredicto |
|---|---|
| Técnica | ✅ Factible |
| Operativa | ✅ Factible (con restricciones menores) |
| Económica | ✅ Factible (costo mínimo) |
| Legal | ✅ Factible |
| **General** | **✅ PROYECTO FACTIBLE** |

---

Responsable del documento: Facundo Nicolás González
Fecha de revisión: 2026-07-23
