from __future__ import annotations

from pathlib import Path

import duckdb


def main() -> None:
    db_path = Path('data/interim/analysis.duckdb')
    db_path.parent.mkdir(parents=True, exist_ok=True)
    duckdb.connect(str(db_path)).close()
    print(f'created {db_path}')


if __name__ == '__main__':
    main()
