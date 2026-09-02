from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import duckdb
import pytest

from evals.storage_migration import plan_metadata_migration


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    db = tmp_path / "db.duckdb"
    con = duckdb.connect(str(db))
    con.execute("CREATE TABLE misconception_chunks (chunk_id VARCHAR, session_id BIGINT, question_id BIGINT, subject_path VARCHAR, misconception_name VARCHAR, error_choice VARCHAR, deep_mechanism VARCHAR, confusion_triggers VARCHAR[], verbatim_student_quotes VARCHAR[], source_turn_ids INTEGER[])")
    con.execute("CREATE TABLE tutor_strategy_chunks (chunk_id VARCHAR, session_id BIGINT, question_id BIGINT, subject_path VARCHAR, pedagogical_goal VARCHAR, strategy_category VARCHAR, key_aha_question VARCHAR, scaffolding_steps VARCHAR[], analogy_or_metaphor VARCHAR, talk_moves VARCHAR[], resolution_outcome VARCHAR, source_turn_ids INTEGER[])")
    con.execute("CREATE TABLE sliding_window_chunks (chunk_id VARCHAR, session_id BIGINT, question_id BIGINT, subject_path VARCHAR, window_start_turn INTEGER, window_end_turn INTEGER, content VARCHAR, source_turn_ids INTEGER[])")
    con.execute("INSERT INTO misconception_chunks VALUES ('m1',1,2,'Math','name','', 'mechanism', ['trigger'], ['quote'], [3])")
    con.execute("INSERT INTO tutor_strategy_chunks VALUES ('s1',1,2,'Math','goal','cat','aha',['step'],'', ['move'],'outcome',[4])")
    con.execute("INSERT INTO sliding_window_chunks VALUES ('w1',1,2,'Math',1,2,'doc',[1,2])")
    con.close()

    chroma = tmp_path / "chroma"
    chroma.mkdir()
    dbs = sqlite3.connect(chroma / "chroma.sqlite3")
    dbs.executescript("""
      CREATE TABLE collections (id TEXT PRIMARY KEY, name TEXT);
      CREATE TABLE segments (id TEXT PRIMARY KEY, collection TEXT, type TEXT, scope TEXT);
      CREATE TABLE embeddings (id INTEGER PRIMARY KEY, segment_id TEXT, embedding_id TEXT);
      CREATE TABLE embedding_metadata (id INTEGER, key TEXT, string_value TEXT, int_value INTEGER, float_value REAL, bool_value INTEGER);
    """)
    ids = [('c1','student_misconceptions'),('c2','tutor_strategies'),('c3','fallback_windows')]
    dbs.executemany('INSERT INTO collections VALUES (?,?)', ids)
    dbs.executemany('INSERT INTO segments VALUES (?,?,?,?)', [(f's{i}',f'c{i}','x','METADATA') for i in range(1,4)])
    dbs.executemany('INSERT INTO embeddings VALUES (?,?,?)', [(i,f's{i}',rid) for i,(_,rid) in enumerate([('','m1'),('','s1'),('','w1')],1)])
    dbs.executemany('INSERT INTO embedding_metadata VALUES (?,?,?,?,?,?)', [
        (1,'session_id',None,1,None,None), (1,'question_id',None,2,None,None),
        (2,'session_id',None,1,None,None), (2,'question_id',None,2,None,None),
        (3,'session_id',None,1,None,None), (3,'question_id',None,2,None,None),
    ])
    dbs.commit(); dbs.close()
    return db, chroma


def test_plan_reports_missing_and_changed_metadata(tmp_path: Path) -> None:
    db, chroma = _fixture(tmp_path)
    diff = plan_metadata_migration(db, chroma)
    misc = diff['collections']['student_misconceptions']
    assert misc['expected_count'] == 1
    assert misc['change_count'] == 1
    fields = misc['changed'][0]['fields']
    assert json.loads(fields['source_turn_ids']['after']) == [3]
    assert 'deep_mechanism' in fields


def test_plan_rejects_missing_chroma(tmp_path: Path) -> None:
    db, _ = _fixture(tmp_path)
    with pytest.raises(FileNotFoundError):
        plan_metadata_migration(db, tmp_path / 'missing')
