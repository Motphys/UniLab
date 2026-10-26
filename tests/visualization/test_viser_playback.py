from types import SimpleNamespace

import torch

from unilab.visualization.interactive_playback import PlaybackControls
from unilab.visualization.viser_playback import _advance_session


class _Session:
    def __init__(self, terminated: bool, truncated: bool) -> None:
        self.env = SimpleNamespace(
            state=SimpleNamespace(
                terminated=torch.tensor([False, terminated]),
                truncated=torch.tensor([False, truncated]),
            )
        )
        self.advances = 0

    def advance(self, controls: PlaybackControls) -> bool:
        del controls
        self.advances += 1
        return True


def test_pause_on_done_latches_any_completed_environment() -> None:
    session = _Session(terminated=False, truncated=True)
    controls = PlaybackControls()
    completed = _advance_session(session, controls, pause_on_done=True, completed=False)
    assert completed
    assert session.advances == 1
    assert _advance_session(session, controls, pause_on_done=True, completed=completed)
    assert session.advances == 1


def test_pause_on_done_disabled_preserves_playback() -> None:
    session = _Session(terminated=True, truncated=False)
    controls = PlaybackControls()
    assert not _advance_session(session, controls, pause_on_done=False, completed=False)
    assert not _advance_session(session, controls, pause_on_done=False, completed=False)
    assert session.advances == 2
