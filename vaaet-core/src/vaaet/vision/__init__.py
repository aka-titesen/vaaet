# SPDX-FileCopyrightText: 2026 VAAET Contributors
# SPDX-License-Identifier: AGPL-3.0-only
"""Computer-vision components for VAAET.

Submodules are intentionally not imported here: importing ``vaaet.vision``
must not load OpenCV, Ultralytics, or a model into memory. Import the required
API from its explicit submodule instead.
"""
