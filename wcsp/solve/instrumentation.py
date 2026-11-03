"""Optional streaming solver events and aggregates for experiment artifacts."""
from contextlib import contextmanager
from contextvars import ContextVar

_sink = ContextVar('solver_event_sink', default=None)
_scope = ContextVar('solver_event_scope', default={})


@contextmanager
def capture_events(sink):
    token = _sink.set(sink)
    try:
        yield
    finally:
        _sink.reset(token)


@contextmanager
def solver_scope(**fields):
    token = _scope.set({**_scope.get(), **fields})
    try:
        yield
    finally:
        _scope.reset(token)


def emit(event, **fields):
    sink = _sink.get()
    if sink is not None:
        sink({'event': event, **_scope.get(), **fields})


def iteration_totals(stats):
    totals = {'icm_calls': 0, 'icm_sweeps': 0, 'icm_pixel_updates': 0,
              'icm_sweep_limit_hits': 0, 'toulbar_calls': 0,
              'toulbar_search_nodes': 0, 'toulbar_backtracks': 0}
    def visit(node):
        kind = node.get('solver')
        if kind == 'icm':
            totals['icm_calls'] += 1
            totals['icm_sweeps'] += node['sweeps']
            totals['icm_pixel_updates'] += sum(node['pixel_updates_per_sweep'])
            totals['icm_sweep_limit_hits'] += int(node['stop_reason'] == 'sweep_limit')
        elif kind == 'toulbar2':
            totals['toulbar_calls'] += 1
            for source, dest in [('search_nodes', 'toulbar_search_nodes'), ('backtracks', 'toulbar_backtracks')]:
                value = node.get(source)
                totals[dest] = None if value is None or totals[dest] is None else totals[dest] + value
        for container in ('blocks', 'stages'):
            for child in node.get(container, []):
                visit(child['stats'])
    visit(stats)
    return totals
