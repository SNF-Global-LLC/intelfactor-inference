"""
Compatibility wrapper for the station API.

The maintained Flask surface is api_v2. Keep this module as a small import
shim so old entrypoints do not expose the retired unauthenticated API routes.
"""

from __future__ import annotations

import logging
from typing import Any

from packages.inference.api_v2 import create_app

logger = logging.getLogger(__name__)


def run_api(runtime: Any = None, host: str = "0.0.0.0", port: int = 8080) -> None:
    """Run the maintained station API server."""
    app = create_app(runtime)
    logger.info("Station API starting on %s:%d", host, port)
    app.run(host=host, port=port, debug=False, threaded=True)
