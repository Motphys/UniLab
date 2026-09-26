from __future__ import annotations

import numpy as np
import pytest
from scripts.motion.csv_to_npz import load_csv_motion


def _write_csv(path, rows: list[list[float]], *, header: bool) -> None:
    lines = ["root_x,root_y,root_z,qx,qy,qz,qw,joint"] if header else []
    lines.extend(",".join(str(value) for value in row) for row in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_load_csv_motion_preserves_headerless_first_frame(tmp_path):
    path = tmp_path / "headerless.csv"
    rows = [
        [0.0, 0.1, 0.2, 0.0, 0.0, 0.0, 1.0, 0.3],
        [1.0, 1.1, 1.2, 0.0, 0.0, 0.0, 1.0, 0.4],
    ]
    _write_csv(path, rows, header=False)

    base_poss, base_rots, dof_poss = load_csv_motion(str(path))

    np.testing.assert_allclose(base_poss, np.asarray(rows, dtype=np.float32)[:, :3])
    np.testing.assert_allclose(base_rots, [[1.0, 0.0, 0.0, 0.0]] * 2)
    np.testing.assert_allclose(dof_poss, [[0.3], [0.4]])


def test_load_csv_motion_line_range_is_frame_based_with_or_without_header(tmp_path):
    rows = [
        [0.0, 0.1, 0.2, 0.0, 0.0, 0.0, 1.0, 0.0],
        [1.0, 1.1, 1.2, 0.0, 0.0, 0.0, 1.0, 1.0],
        [2.0, 2.1, 2.2, 0.0, 0.0, 0.0, 1.0, 2.0],
        [3.0, 3.1, 3.2, 0.0, 0.0, 0.0, 1.0, 3.0],
    ]
    for header in (False, True):
        path = tmp_path / f"motion-{int(header)}.csv"
        _write_csv(path, rows, header=header)

        base_poss, _, dof_poss = load_csv_motion(str(path), (2, 3))

        np.testing.assert_allclose(base_poss, np.asarray(rows, dtype=np.float32)[1:3, :3])
        np.testing.assert_allclose(dof_poss, [[1.0], [2.0]])


def test_load_csv_motion_rejects_non_numeric_data(tmp_path):
    path = tmp_path / "invalid.csv"
    path.write_text("root_x,root_y\nnot,a-number\n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_csv_motion(str(path))


def test_load_csv_motion_rejects_invalid_line_range(tmp_path):
    path = tmp_path / "headerless.csv"
    _write_csv(path, [[0.0, 0.1, 0.2, 0.0, 0.0, 0.0, 1.0, 0.3]], header=False)

    with pytest.raises(ValueError, match="line_range"):
        load_csv_motion(str(path), (2, 1))


def test_load_csv_motion_rejects_range_beyond_available_frames(tmp_path):
    path = tmp_path / "headerless.csv"
    _write_csv(path, [[0.0, 0.1, 0.2, 0.0, 0.0, 0.0, 1.0, 0.3]], header=False)

    with pytest.raises(ValueError, match="available motion frames"):
        load_csv_motion(str(path), (2, 2))
