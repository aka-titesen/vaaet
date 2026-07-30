<!-- context: VAAET/docs/DVC_GUIDE.md — Guía de uso de DVC para versionado de modelos.
Complementa PROJECT_PLAN.md, DEPLOYMENT.md y ADR-011. -->

# Guía de DVC (Data Version Control) — VAAET

## Identificación del Proyecto

| Campo | Detalles |
|---|---|
| **Nombre del Proyecto** | VAAET — Video Advanced Analysis of Traffic |
| **Versión** | 3.1.0 |
| **Estado** | Aprobado |
| **Responsable Técnico** | Facundo Nicolás González |
| **Última Revisión** | 2026-07-30 |

---

## 1. ¿Qué es DVC?

**DVC (Data Version Control)** es una herramienta open source que funciona como "Git para datos y modelos". Mientras Git versiona tu código fuente (archivos livianos), DVC versiona los archivos pesados que Git no puede manejar eficientemente:

| Git versiona | DVC versiona |
|---|---|
| Código Python (`.py`) | Modelos entrenados (`.keras`) |
| Notebooks (`.ipynb`) | Scalers y mappings (`.joblib`) |
| Documentación (`.md`) | Datasets grandes (`.csv`) |
| Configuración (`.toml`) | Videos de entrenamiento (`.mp4`) |

### ¿Por qué DVC y no solo Google Drive?

| Aspecto | Solo Google Drive | DVC + Google Drive |
|---|---|---|
| Historial de versiones | ❌ Solo la última versión | ✅ Cada commit de Git tiene su modelo |
| Reproducibilidad | ❌ Manual | ✅ `dvc checkout` + `git checkout` |
| Cambiar de storage | ❌ Re-subir todo | ✅ `dvc remote default s3` (una línea) |
| Integración con CI/CD | ❌ Imposible | ✅ `dvc pull` en GitHub Actions |
| Tracking de integridad | ❌ Ninguno | ✅ Hash MD5 verificado |

---

## 2. Arquitectura de Storage

```
┌──────────────────── Git (GitHub) ────────────────────┐
│  models/intelligence/                                 │
│  ├── traffic_classifier.keras.dvc  ← metadata (~200B)│
│  ├── feature_scaler.joblib.dvc     ← metadata (~200B)│
│  └── label_mapping.joblib.dvc      ← metadata (~200B)│
└──────────────────────┬────────────────────────────────┘
                       │ dvc push / dvc pull
                       ▼
┌──────────────────── DVC Remote ──────────────────────┐
│  [DEFAULT] Google Drive (15 GB gratis)                │
│     └── Carpeta: VAAET-DVC-Storage                    │
│                                                       │
│  [ALT] AWS S3 (Free Tier)                             │
│     └── Bucket: vaaet-model-registry                  │
│                                                       │
│  [ALT] Local (sin internet)                           │
│     └── /tmp/vaaet-dvc-local                          │
└───────────────────────────────────────────────────────┘
```

---

## 3. Comandos Esenciales

### 3.1 Registrar un artefacto nuevo

Después de entrenar un modelo (Módulo 1), registrá los artefactos:

```bash
# Registrar cada artefacto
dvc add models/intelligence/traffic_classifier.keras
dvc add models/intelligence/feature_scaler.joblib
dvc add models/intelligence/label_mapping.joblib

# Commitear la metadata en Git
git add models/intelligence/*.dvc models/intelligence/.gitignore
git commit -m "feat(models): registrar artefactos MLP v1.1 en DVC"

# Subir los artefactos al remote
dvc push
```

### 3.2 Descargar artefactos

En un nuevo entorno (o en Google Colab):

```bash
# Descargar todos los artefactos
dvc pull

# O solo los modelos
dvc pull models/intelligence/
```

### 3.3 Ver estado

```bash
# ¿Hay cambios sin registrar?
dvc status

# ¿Qué remotes tengo configurados?
dvc remote list

# Diagnóstico general
dvc doctor
```

### 3.4 Cambiar de storage

```bash
# Cambiar a S3
dvc remote default s3
dvc push  # Re-sube todo a S3

# Volver a Google Drive
dvc remote default gdrive
dvc push
```

### 3.5 Viajar en el tiempo

```bash
# Ver qué modelo corresponde a un commit anterior
git log --oneline

# Ir a esa versión
git checkout abc1234
dvc checkout  # Descarga el modelo de esa época

# Volver al presente
git checkout main
dvc checkout
```

---

## 4. Uso en Google Colab

### 4.1 Celda de Setup (agregar al inicio del notebook)

```python
# === DVC Setup (ejecutar una sola vez por sesión) ===
!pip install -q dvc[gdrive]

# Descargar artefactos del modelo
!dvc pull models/intelligence/

# Verificar que los artefactos existan
import pathlib
for f in pathlib.Path("models/intelligence").glob("*"):
    if f.suffix in (".keras", ".joblib"):
        print(f"✅ {f.name} ({f.stat().st_size / 1024:.1f} KB)")
```

### 4.2 Después de re-entrenar (Módulo 1)

```python
# Registrar los nuevos artefactos
!dvc add models/intelligence/traffic_classifier.keras
!dvc add models/intelligence/feature_scaler.joblib
!dvc add models/intelligence/label_mapping.joblib

# Subir al remote (pedirá autenticación OAuth la primera vez)
!dvc push
```

### 4.3 Autenticación con Google Drive

La primera vez que ejecutes `dvc push` o `dvc pull` con el remote de Google Drive, DVC abre un navegador para autenticación OAuth. En Colab:

1. Aparece un enlace en la celda
2. Hacé click, autenticá con tu cuenta de Google
3. Copiá el código de autorización
4. Pegalo en la celda

Esta autorización se mantiene durante la sesión de Colab.

---

## 5. Estructura de Archivos DVC

| Archivo | Commiteado en Git | Propósito |
|---|---|---|
| `.dvc/config` | ✅ Sí | Configuración de remotes (compartida) |
| `.dvc/config.local` | ❌ No (en .gitignore) | Credenciales locales |
| `.dvc/.gitignore` | ✅ Sí | Exclusiones internas de DVC |
| `.dvcignore` | ✅ Sí | Archivos que DVC no debe procesar |
| `*.dvc` | ✅ Sí | Metadata de artefactos (hash MD5, tamaño) |
| `.dvc/cache/` | ❌ No (en .gitignore) | Cache local de artefactos |
| `.dvc/tmp/` | ❌ No (en .gitignore) | Temporales de DVC |

---

## 6. Remotes Configurados

| Remote | URL | Uso | Cómo activar |
|---|---|---|---|
| `gdrive` **(default)** | `gdrive://VAAET-DVC-Storage` | Storage principal, gratuito 15 GB | `dvc remote default gdrive` |
| `s3` | `s3://vaaet-model-registry` | Alternativo para producción | `dvc remote default s3` |
| `local` | `/tmp/vaaet-dvc-local` | Fallback sin internet | `dvc remote default local` |

### Agregar un nuevo remote

```bash
# Ejemplo: Azure Blob Storage
dvc remote add azure azure://vaaet-container/models
dvc remote default azure
```

---

## 7. Troubleshooting

| Problema | Solución |
|---|---|
| `ERROR: failed to push` | Verificar autenticación: `dvc remote modify gdrive --local gdrive_acknowledge_abuse true` |
| `ERROR: No remote configured` | `dvc remote default gdrive` |
| Cache ocupa mucho espacio | `dvc gc --workspace` (limpia versiones viejas del cache) |
| Conflicto de `.dvc` en merge | Resolver como cualquier conflicto de Git, luego `dvc checkout` |
| `dvc pull` falla en Colab | Verificar que `dvc[gdrive]` esté instalado |

---

Responsable del documento: Facundo Nicolás González
Fecha de revisión: 2026-07-30
Documentos de referencia: [ADR-011](adr/ADR-011-dvc-model-registry.md), [DEPLOYMENT.md](DEPLOYMENT.md), [MODEL_CARD.md](MODEL_CARD.md)
