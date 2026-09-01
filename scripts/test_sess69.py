import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.storage_manager import DualEngineStorageManager
from src.retriever import DualMetricRetriever

storage = DualEngineStorageManager(db_path='data/db/tutoring_knowledge.duckdb', chroma_dir='data/chroma', embedding_backend='deterministic')
retriever = DualMetricRetriever(storage_manager=storage, query_rewrite_mode='deterministic')

res = storage.chroma_client.get_collection('student_misconceptions').get(where={'session_id': 69})
print("Sess 69 Misconception in Chroma:")
print(res)

print("\nRunning RRF retrieval for Case 8:")
rrf_res = retriever.retrieve_multi_perspective_rrf("不等式 -2m>12 中，学生为什么会得到 m>-6 而非 m<-6？该错误的核心原因是什么？", top_k_each=10)
for idx, m in enumerate(rrf_res['misconceptions'], 1):
    print(f"Rank {idx}: sess={m['metadata'].get('session_id')}, score={m.get('rrf_score')}, name={m['metadata'].get('misconception_name')}")
