from __future__ import annotations

import numpy as np

from src.models.integration import run_all


def main() -> None:
    cashflows = np.array([-1000, 300, 400, 500, 600], dtype=float)
    result = run_all(population=800000, penetration=0.08, cashflows=cashflows, discount_rate=0.12)
    print(result)


if __name__ == '__main__':
    main()
