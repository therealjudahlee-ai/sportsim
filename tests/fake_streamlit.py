"""A recording stand-in for streamlit so pages can be executed headlessly.
Widgets return their defaults; buttons return True (to exercise every path)."""
from __future__ import annotations

import types
from unittest.mock import MagicMock

LOG: list[tuple] = []


class StopExecution(Exception):
    pass


class _CM:
    def __init__(self, name="cm"):
        self.name = name

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __getattr__(self, item):
        return getattr(_st, item)


class _State(dict):
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e

    def __setattr__(self, k, v):
        self[k] = v


class _Secrets:
    def get(self, k, d=None):
        return d


def _cache(*dargs, **dkw):
    def wrap(fn):
        store = {}

        def inner(*a, **k):
            key = repr((a, sorted(k.items())))[:2000]
            if key not in store:
                store[key] = fn(*a, **k)
            return store[key]
        inner.clear = store.clear
        return inner
    if dargs and callable(dargs[0]):
        return wrap(dargs[0])
    return wrap


class _ST(types.ModuleType):
    BUTTONS = True

    def __init__(self):
        super().__init__("streamlit")
        self.session_state = _State()
        self.secrets = _Secrets()
        self.sidebar = _CM("sidebar")
        self.column_config = MagicMock()
        self.cache_data = _cache
        self.cache_resource = _cache

    # ---- layout
    def columns(self, spec, **k):
        n = spec if isinstance(spec, int) else len(spec)
        return [_CM(f"col{i}") for i in range(n)]

    def tabs(self, names):
        LOG.append(("tabs", names))
        return [_CM(n) for n in names]

    def container(self, **k):
        return _CM("container")

    def expander(self, label, **k):
        LOG.append(("expander", label))
        return _CM(label)

    def spinner(self, *a, **k):
        return _CM("spinner")

    # ---- widgets
    def selectbox(self, label, options, index=0, format_func=str, **k):
        options = list(options)
        LOG.append(("selectbox", label, len(options)))
        if options:
            format_func(options[index])
        return options[index] if options else None

    def multiselect(self, label, options, default=None, format_func=str, **k):
        LOG.append(("multiselect", label))
        for o in options:
            format_func(o)
        return list(default or [])

    def radio(self, label, options, format_func=str, **k):
        options = list(options)
        for o in options:
            format_func(o)
        return options[0]

    def slider(self, label, min_value=None, max_value=None, value=None, step=None, **k):
        LOG.append(("slider", label))
        return value if value is not None else min_value

    def number_input(self, label, min_value=None, max_value=None, value=None, step=None, **k):
        return value if value is not None else (min_value or 0)

    def toggle(self, label, value=False, **k):
        return value

    def checkbox(self, label, value=False, **k):
        return value

    def date_input(self, label, value=None, **k):
        return value

    def file_uploader(self, *a, **k):
        return None

    def button(self, label, **k):
        LOG.append(("button", label))
        if label in ("Reload data", "Retry", "Try live DraftKings salaries"):
            return False
        return self.BUTTONS

    def download_button(self, label, data, *a, **k):
        assert isinstance(data, (bytes, str)), label
        LOG.append(("download", label, len(data)))
        return False

    # ---- output
    def dataframe(self, df, **k):
        LOG.append(("dataframe", getattr(df, "shape", None)))

    def metric(self, label, value, delta=None, **k):
        LOG.append(("metric", label, value, delta))

    def altair_chart(self, chart, **k):
        LOG.append(("chart",))

    def _msg(kind):
        def f(self, body="", *a, **k):
            LOG.append((kind, str(body)[:300]))
        return f

    markdown = _msg("markdown")
    caption = _msg("caption")
    write = _msg("write")
    info = _msg("info")
    warning = _msg("warning")
    error = _msg("error")
    success = _msg("success")
    title = _msg("title")

    def divider(self):
        pass

    def page_link(self, *a, **k):
        pass

    def set_page_config(self, **k):
        pass

    def stop(self):
        raise StopExecution()

    def rerun(self):
        raise StopExecution()


_st = _ST()


def install():
    import sys
    sys.modules["streamlit"] = _st
    sys.modules["altair"] = MagicMock()
    return _st
