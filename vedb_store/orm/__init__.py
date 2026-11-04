"""Deprecated alias for `vedb_store.session`.

The `orm` subpackage held database-mapped classes that have since been
removed; the remaining classes now live in `vedb_store.session`. This
alias keeps old imports (e.g. `from vedb_store.orm.session import Session`)
working and will be removed in a future release.
"""
import sys
import warnings

from .. import session

warnings.warn("vedb_store.orm is deprecated; import from vedb_store or "
              "vedb_store.session instead", DeprecationWarning, stacklevel=2)

sys.modules[__name__ + '.session'] = session
