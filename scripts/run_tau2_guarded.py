from __future__ import annotations

from tau2.cli import main

from guarded_agent.adapters import tau2_agent  # noqa: F401  (registers our agent on import)

if __name__ == "__main__":
    main()
