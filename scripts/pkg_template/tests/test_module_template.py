import sys
sys.path.append("../../")

from pkg_template import module_template

def test_add():
    assert module_template.add(5, 2) == 7

