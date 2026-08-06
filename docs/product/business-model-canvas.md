<!-- context: VAAET/docs/product/business-model-canvas.md — Business Model Canvas.
Proyección comercial futura del proyecto. -->

# Business Model Canvas — VAAET

## Identificación del Proyecto

| Campo | Detalles |
|---|---|
| **Nombre del Proyecto** | VAAET — Video Advanced Analysis of Traffic |
| **Versión** | 4.2.0 |
| **Estado** | Borrador (visión futura) |
| **Responsable Técnico** | Facundo Nicolás González |
| **Última Revisión** | 2026-07-23 |

---

## Canvas

### 1. Segmentos de Clientes

| Segmento | Descripción | Necesidad |
|---|---|---|
| **Entes viales provinciales/nacionales** | Vialidad Nacional, DPV Corrientes | Datos objetivos de tráfico para planificación |
| **Municipalidades** | Ciudades con puentes o accesos congestionados | Alertas tempranas, reportes automáticos |
| **Fuerzas de seguridad** | Operadores SISE, policía caminera | Detección de accidentes en tiempo real |
| **Consultoras de ingeniería** | Empresas que realizan estudios de tráfico | Automatización de conteos vehiculares |
| **Universidades** | Grupos de investigación en transporte | Datasets y herramientas de análisis |

### 2. Propuesta de Valor

> **Automatizar el análisis vehicular mediante visión artificial, reemplazando el conteo manual y proporcionando clasificación del estado del tráfico en tiempo real, con un costo de infraestructura cercano a cero.**

- **Para entes viales**: Datos de tráfico por minuto sin intervención humana
- **Para seguridad**: Detección temprana de condiciones anómalas (congestión, accidente)
- **Para investigadores**: Dataset de alta calidad con 19 features de telemetría
- **Para consultoras**: Reducción de costos de conteo manual en un 95%

### 3. Canales

| Canal | Etapa | Descripción |
|---|---|---|
| GitHub (open source) | Conciencia/Evaluación | Repositorio público como demostración técnica |
| Web App (futuro) | Entrega | Dashboard de tráfico en tiempo real |
| API REST (futuro) | Entrega | Integración programática para sistemas existentes |
| Presentaciones académicas | Conciencia | Publicaciones y conferencias de transporte |
| Demo de portfolio | Conciencia | Demostración visual del pipeline |

### 4. Relación con Clientes

| Tipo | Descripción |
|---|---|
| **Autoservicio** | Documentación completa, guías de usuario, notebooks ejecutables |
| **Asistencia personalizada** | Consultoría para calibración y despliegue en contextos específicos |
| **Co-creación** | Colaboración con investigadores para expandir el dataset |

### 5. Fuentes de Ingresos (Proyección)

| Modelo | Descripción | Estimación |
|---|---|---|
| **SaaS (futuro)** | Suscripción mensual a Web App con procesamiento de video | $50-200 USD/usuario/mes |
| **Consultoría** | Calibración y despliegue para puentes/accesos específicos | $500-2000 USD/proyecto |
| **Licencias institucionales** | Acuerdos con entes viales para uso exclusivo | $5.000-20.000 USD/año |
| **Open source + soporte** | Software gratuito, soporte premium | Variable |

### 6. Recursos Clave

| Recurso | Tipo | Descripción |
|---|---|---|
| Pipeline YOLO 11 + MLP | Tecnológico | Motor de visión artificial |
| Código fuente modular | Tecnológico | `src/` reutilizable y testeable |
| Dataset del Puente Belgrano | Datos | ~2.000 registros reales de telemetría |
| Conocimiento de dominio | Humano | Calibración específica del puente |
| Google Colab | Infraestructura | GPU gratuita para procesamiento |

### 7. Actividades Clave

1. Mantenimiento y evolución del pipeline de visión artificial
2. Re-entrenamiento del clasificador con datos nuevos
3. Desarrollo de la Web App para acceso en tiempo real
4. Calibración para nuevas ubicaciones (otros puentes, accesos)
5. Publicaciones académicas para validación técnica

### 8. Socios Clave

| Socio | Contribución |
|---|---|
| SISE / Vialidad Nacional | Acceso a cámaras y datos de video |
| Google (Colab) | Infraestructura de cómputo gratuita |
| Ultralytics | Modelos YOLO pre-entrenados |
| AWS | Infraestructura de base de datos |
| Universidad (futuro) | Validación académica, datasets |

### 9. Estructura de Costos

| Costo | Tipo | Estimación Actual | Estimación Futuro (Web App) |
|---|---|---|---|
| Infraestructura (Colab) | Variable | $0 | $0-$50/mes (Colab Pro) |
| Base de datos (AWS RDS) | Variable | $15/mes | $50-200/mes |
| Hosting Web App | Variable | $0 | $25-100/mes |
| Desarrollo (tiempo) | Fijo | 100% del esfuerzo | — |
| **Total** | — | **$15/mes** | **$75-350/mes** |

---

## Roadmap Comercial

```mermaid
gantt
    title Roadmap VAAET — Visión Comercial
    dateFormat  YYYY-Q
    section Académico
    Pipeline de percepción           :done, 2025-Q1, 2025-Q2
    Clasificador MLP + documentación :done, 2025-Q2, 2025-Q3
    Publicación académica            :active, 2026-Q3, 2026-Q4
    section Producto
    Web App MVP (dashboard)          : 2026-Q4, 2027-Q1
    API REST para integración        : 2027-Q1, 2027-Q2
    Piloto con ente vial             : 2027-Q2, 2027-Q3
    section Comercial
    Modelo SaaS                      : 2027-Q3, 2027-Q4
    Expansión a otros puentes        : 2027-Q4, 2028-Q2
```

---

Responsable del documento: Facundo Nicolás González
Fecha de revisión: 2026-07-23
