#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# VAAET — Script de configuración inicial de DVC
# Uso: bash scripts/dvc_setup.sh
#
# Este script prepara DVC para el primer uso. Solo necesitás
# ejecutarlo una vez por máquina/entorno.
# ─────────────────────────────────────────────────────────────
set -euo pipefail

echo "╔══════════════════════════════════════════════════╗"
echo "║     VAAET — Configuración de DVC                ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# 1. Verificar que estamos en la raíz del repo
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: Ejecutá este script desde la raíz del repositorio VAAET"
    exit 1
fi

# 2. Verificar/instalar DVC
if ! command -v dvc &> /dev/null; then
    echo "📦 DVC no encontrado. Instalando..."
    pip install -q "dvc[gdrive]>=3.50.0"
else
    echo "✅ DVC encontrado: $(dvc version | head -1)"
fi

# 3. Verificar que DVC está inicializado
if [ ! -d ".dvc" ]; then
    echo "🔧 Inicializando DVC..."
    dvc init
else
    echo "✅ DVC ya inicializado"
fi

# 4. Verificar remotes
REMOTES=$(dvc remote list 2>/dev/null || true)
if echo "$REMOTES" | grep -q "gdrive"; then
    echo "✅ Remote 'gdrive' configurado"
else
    echo "🔧 Configurando remote 'gdrive'..."
    dvc remote add -d gdrive gdrive://VAAET-DVC-Storage
fi

if echo "$REMOTES" | grep -q "s3"; then
    echo "✅ Remote 's3' configurado"
else
    echo "🔧 Configurando remote 's3' (alternativo)..."
    dvc remote add s3 s3://vaaet-model-registry
fi

if echo "$REMOTES" | grep -q "local"; then
    echo "✅ Remote 'local' configurado"
else
    echo "🔧 Configurando remote 'local' (fallback)..."
    dvc remote add local /tmp/vaaet-dvc-local
fi

# 5. Mostrar estado
echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║     ✅ DVC configurado correctamente             ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""
echo "Remotes configurados:"
dvc remote list
echo ""
echo "Próximos pasos:"
echo "  1. Entrenar el modelo (Módulo 1 en Colab)"
echo "  2. dvc add models/intelligence/traffic_classifier.keras"
echo "  3. dvc add models/intelligence/feature_scaler.joblib"
echo "  4. dvc add models/intelligence/label_mapping.joblib"
echo "  5. git add models/intelligence/*.dvc && git commit -m 'feat(models): registrar artefactos'"
echo "  6. dvc push"
echo ""
echo "Para más información: docs/DVC_GUIDE.md"
