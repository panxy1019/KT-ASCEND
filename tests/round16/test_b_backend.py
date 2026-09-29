from __future__ import annotations

import importlib
import unittest


def _modules():
    try:
        return (
            importlib.import_module("round16.b.routing_bridge"),
            importlib.import_module("round16.b.kt_expert_backend"),
        )
    except ModuleNotFoundError as error:
        raise AssertionError("Round16-B routing and backend modules must exist") from error


def _row(expert: int) -> list[int]:
    return [expert, -1, -1, -1, -1, -1, -1, -1, -1, -1]


class BackendTests(unittest.TestCase):
    def test_chunks_same_expert_and_executes_tail_without_losing_token_identity(self) -> None:
        routing, backend_module = _modules()
        plan = routing.plan_routes(
            [_row(40) for _ in range(5)],
            [[1.0] + [0.0] * 9 for _ in range(5)],
            resident_logical_ids=(),
        )
        seen = []

        def worker(task):
            seen.append(task)
            return tuple((float(row[0] * 10),) for row in task.hidden_rows)

        backend = backend_module.KTExpertBackend(worker, chunk_size=2)
        output = backend.execute(
            plan,
            ((0,), (1,), (2,), (3,), (4,)),
            identity_key=("request-1", 3),
        )

        self.assertEqual([len(task.route_keys) for task in seen], [2, 2, 1])
        self.assertEqual([output[route.key] for route in plan.cpu_routes], [(0.0,), (10.0,), (20.0,), (30.0,), (40.0,)])
        self.assertEqual(len({task.identity_key for task in seen}), 3)
        self.assertTrue(all(task.identity_key[:2] == ("request-1", 3) for task in seen))

    def test_empty_cpu_branch_does_not_submit_work(self) -> None:
        routing, backend_module = _modules()
        plan = routing.plan_routes(
            [_row(4)],
            [[1.0] + [0.0] * 9],
            resident_logical_ids=tuple(range(32)),
        )
        calls = []
        backend = backend_module.KTExpertBackend(lambda task: calls.append(task), chunk_size=2)

        output = backend.execute(plan, ((7,),), identity_key=("request-2", 1))

        self.assertEqual(output, {})
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
