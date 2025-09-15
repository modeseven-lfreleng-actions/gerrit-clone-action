# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Matthew Watkins <mwatkins@linuxfoundation.org>

"""Gerrit Clone - A multi-threaded CLI tool for bulk cloning Gerrit repositories."""

__version__ = "0.1.0"
__author__ = "Matthew Watkins"
__email__ = "mwatkins@linuxfoundation.org"

from gerrit_clone.models import CloneResult, Config, Project, RetryPolicy

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "CloneResult",
    "Config",
    "Project",
    "RetryPolicy",
]
