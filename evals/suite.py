from __future__ import annotations

from datetime import UTC, datetime

from evals.contracts import L1ComponentReport, L1ComponentSuiteSpec
from evals.runners.assembler import AssemblerRunner
from evals.runners.chunking import ChunkingRunner
from evals.runners.grounding import GroundingRunner
from evals.runners.retrieval import RetrievalRunner
from evals.runners.rewrite import RewriteRunner
from evals.runners.storage import StorageRunner


def run_component_suite(spec: L1ComponentSuiteSpec) -> L1ComponentReport:
    started_at = datetime.now(UTC)
    observations = []
    executed_modules: list[str] = []
    if spec.storage is not None:
        observations.extend(StorageRunner().run(spec.context, spec.storage))
        executed_modules.append("storage")
    if spec.grounding_cases:
        observations.extend(GroundingRunner().run(spec.context, spec.grounding_cases))
        executed_modules.append("grounding")
    if spec.chunking_cases:
        observations.extend(ChunkingRunner().run(spec.context, spec.chunking_cases))
        executed_modules.append("chunking")
    if spec.rewrite_cases:
        observations.extend(RewriteRunner().run(spec.context, spec.rewrite_cases))
        executed_modules.append("rewrite")
    if spec.retrieval_cases:
        observations.extend(RetrievalRunner().run(spec.context, spec.retrieval_cases))
        executed_modules.append("retrieval")
    if spec.assembler_cases:
        observations.extend(AssemblerRunner().run(spec.context, spec.assembler_cases))
        executed_modules.append("assembler")
    return L1ComponentReport.from_observations(
        context=spec.context,
        observations=observations,
        executed_modules=executed_modules,
        started_at=started_at,
    )
