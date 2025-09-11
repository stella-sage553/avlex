import re

import avlex


def test_version_is_exposed():
    assert isinstance(avlex.__version__, str)
    assert avlex.__version__


def test_version_is_pep440_ish():
    assert re.match(r"^\d+\.\d+\.\d+", avlex.__version__)
