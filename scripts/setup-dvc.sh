#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# VAAET — Script de configuración inicial de DVC
# Uso: bash scripts/setup-dvc.sh
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
echo "  1. Entrenar el modelo (training workflow en Colab)"
echo "  2. Verificar los cuatro archivos, incluido model-manifest.json"
echo "  3. Eliminar artifacts/traffic-state/.gitkeep"
echo "  4. dvc add artifacts/traffic-state"
echo "  5. git add artifacts/traffic-state.dvc .gitignore && git commit -m 'feat(models): registrar bundle'"
echo "  6. dvc push"
echo ""
echo "Para más información: docs/ml/dvc-guide.md"
