# SPDX-FileCopyrightText: 2026 VAAET Contributors
# SPDX-License-Identifier: AGPL-3.0-only
"""Componentes de visión artificial de VAAET.

Los submódulos no se importan acá deliberadamente: importar ``vaaet.vision``
no debe cargar OpenCV, Ultralytics ni un modelo en memoria. Importá la API
necesaria desde su submódulo explícito.
"""
