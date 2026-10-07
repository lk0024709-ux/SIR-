"""Teachers, critics, verification and consensus.

The behaviours under test are the ones that keep SIR from becoming "an API wrapper that believes
whatever it was told": no drafts means no records, one teacher is not consensus, disagreement is
flagged rather than resolved by preference, and checkable arithmetic is checked by execution.
"""

from __future__ import annotations

import pytest

from curriculum.graph import load_default_graph
from teachers.consensus import ConsensusStatus, reach_consensus
from teachers.critics import critique_draft, summarise as summarise_critiques, worst_verdict
from teachers.pipeline import GenerationConfig, TeacherPipeline, build_request
from teachers.providers import (
    GenerationRequest,
    ProviderRegistry,
    RecordedProvider,
    StaticProvider,
    TeacherDraft,
    TeacherSpec,
    TemplateProvider,
)
from teachers.verify import (
    CheckStatus,
    VerificationMode,
    check_arithmetic,
    check_exact,
    check_numeric,
    check_set,
    extract_arithmetic_claims,
    verify_answer,
)

GOOD = {"task": "What is 17% of 240?", "response": "17% of 240 = 40.8, so the answer is 40.8.", "answer": "40.8", "rationale": "Multiply by 0.17."}


def _spec(model: str, permitted: bool = True) -> TeacherSpec:
    return TeacherSpec(model_id=model, terms_note="fixture", output_reuse_permitted=permitted)


def _draft(model: str, payload: dict, request_id: str = "req.1") -> TeacherDraft:
    return TeacherDraft(request_id=request_id, teacher=_spec(model), payload=payload)


# --------------------------------------------------------------------------------------
# deterministic verification
# --------------------------------------------------------------------------------------
def test_arithmetic_claims_are_verified_exactly():
    assert check_arithmetic("17% of 240 = 40.8").status == CheckStatus.PASS.value
    assert check_arithmetic("3/4 + 1/2 = 5/4").status == CheckStatus.PASS.value
    assert check_arithmetic("2 + 2 = 5").status == CheckStatus.FAIL.value
    assert check_arithmetic("1/3 + 1/6 = 1/2").status == CheckStatus.PASS.value


def test_arithmetic_claim_extraction_finds_both_forms():
    claims = extract_arithmetic_claims("Check: 12 * 7 = 84 and 15% of 200 = 30.")
    kinds = {c.kind for c in claims}
    assert kinds == {"binary", "percent_of"}


def test_uncheckable_text_is_unverifiable_not_a_pass():
    result = check_arithmetic("Photosynthesis converts light into chemical energy.")
    assert result.status == CheckStatus.UNVERIFIABLE.value
    assert "NO_VERIFIABLE_CLAIM" in result.error_codes


def test_numeric_and_exact_and_set_modes():
    assert check_numeric("40.80", "40.8").status == CheckStatus.PASS.value
    assert check_numeric("41", "40.8").status == CheckStatus.FAIL.value
    assert check_numeric("41", "40.8", tolerance=0.5).status == CheckStatus.PASS.value
    assert check_exact("Jaipur.", "jaipur").status == CheckStatus.PASS.value
    assert check_exact("", "jaipur").status == CheckStatus.FAIL.value
    assert check_set("किताब, पुस्तक", "किताब, पुस्तक").status == CheckStatus.PASS.value
    assert check_set("किताब", "किताब, पुस्तक").status == CheckStatus.FAIL.value


def test_manual_mode_never_claims_verification():
    result = verify_answer("some free-form answer", None, VerificationMode.MANUAL)
    assert result.status == CheckStatus.UNVERIFIABLE.value
    assert "REQUIRES_HUMAN_REVIEW" in result.error_codes


def test_division_by_zero_is_not_a_crash():
    result = check_arithmetic("5 / 0 = 0")
    assert result.status in (CheckStatus.UNVERIFIABLE.value, CheckStatus.FAIL.value)


# --------------------------------------------------------------------------------------
# critics
# --------------------------------------------------------------------------------------
def test_critic_rejects_placeholders_and_empty_payloads():
    critiques = critique_draft(_draft("t1", {"task": "TODO", "response": "placeholder", "answer": ""}), language="en")
    assert worst_verdict(critiques) == "reject"
    codes = {c for cr in critiques for c in cr.codes}
    assert {"MISSING_FIELD", "PLACEHOLDER_TEXT", "TOO_SHORT"} & codes


def test_critic_flags_language_script_mismatch():
    payload = {"task": "Hindi lesson", "response": "This lesson is written entirely in Latin script.", "answer": "yes"}
    critiques = critique_draft(_draft("t1", payload), language="hi")
    assert any(c.critic == "language_script" for c in critiques)
    assert worst_verdict(critiques) != "reject", "a script mismatch is a warning to review, not an automatic delete"


def test_critic_rejects_false_arithmetic():
    payload = {"task": "Compute", "response": "The result is 17% of 240 = 39.8 exactly.", "answer": "39.8"}
    critiques = critique_draft(_draft("t1", payload), language="en")
    assert any(c.critic == "arithmetic" and c.verdict == "reject" for c in critiques)


def test_critic_flags_degenerate_repetition():
    sentence = "This is a repeated sentence that is long enough to be counted."
    payload = {"task": "Lesson", "response": " ".join([sentence] * 3), "answer": "ok"}
    critiques = critique_draft(_draft("t1", payload), language="en")
    assert any(c.critic == "degeneracy" for c in critiques)


def test_critique_summary_counts_codes():
    critiques = critique_draft(_draft("t1", {"task": "", "response": "", "answer": ""}), language="en")
    summary = summarise_critiques(critiques)
    assert summary["findings"] >= 1
    assert "MISSING_FIELD" in summary["codes"]


# --------------------------------------------------------------------------------------
# consensus
# --------------------------------------------------------------------------------------
def test_two_agreeing_teachers_produce_agreement():
    outcome = reach_consensus([_draft("a", GOOD), _draft("b", GOOD)])
    assert outcome.status is ConsensusStatus.AGREED
    assert outcome.canonical_answer == "40.8"
    assert outcome.teachers == ["a", "b"]
    assert outcome.verification_status_for_record() == "multi_teacher_agreement"


def test_single_teacher_is_insufficient_by_default():
    outcome = reach_consensus([_draft("a", GOOD)], min_teachers=2)
    assert outcome.status is ConsensusStatus.INSUFFICIENT_EVIDENCE
    assert not outcome.usable
    assert any("single-teacher" in n for n in outcome.notes)


def test_disagreement_is_resolved_only_by_deterministic_verification():
    good = {"task": "Compute", "response": "17% of 240 = 40.8", "answer": "40.8"}
    bad = {"task": "Compute", "response": "17% of 240 = 39.8", "answer": "39.8"}
    outcome = reach_consensus(
        [_draft("a", good), _draft("b", bad)],
        mode=VerificationMode.ARITHMETIC_CLAIMS,
    )
    assert outcome.status is ConsensusStatus.RESOLVED_BY_VERIFICATION
    assert outcome.canonical_answer == "40.8"
    assert outcome.verification_status_for_record() == "deterministically_verified"


def test_unresolvable_disagreement_is_flagged_not_chosen():
    a = {"task": "Name a colour", "response": "Red", "answer": "red"}
    b = {"task": "Name a colour", "response": "Blue", "answer": "blue"}
    outcome = reach_consensus([_draft("a", a), _draft("b", b)], mode=VerificationMode.EXACT, expected="green")
    assert outcome.status is ConsensusStatus.REJECTED
    assert outcome.canonical_answer is None
    assert not outcome.usable


def test_disagreement_without_resolution_mode_stays_disagreement():
    a = {"task": "Explain", "response": "Because A.", "answer": "A"}
    b = {"task": "Explain", "response": "Because B.", "answer": "B"}
    outcome = reach_consensus([_draft("a", a), _draft("b", b)], mode=VerificationMode.MANUAL)
    assert outcome.status is ConsensusStatus.DISAGREEMENT
    assert "not silently chosen" in outcome.notes[-1]


def test_no_drafts_means_no_output():
    outcome = reach_consensus([])
    assert outcome.status is ConsensusStatus.INSUFFICIENT_EVIDENCE
    assert outcome.canonical_answer is None
    assert "no fabrication fallback" in outcome.notes[0]


def test_two_answers_that_both_verify_are_escalated():
    a = {"task": "Name the gas", "response": "oxygen", "answer": "oxygen"}
    b = {"task": "Name the gas", "response": "O2", "answer": "O2"}
    outcome = reach_consensus([_draft("a", a), _draft("b", b)], mode=VerificationMode.INTEGRITY_ONLY)
    assert outcome.status is ConsensusStatus.DISAGREEMENT
    assert "verification cannot choose" in outcome.notes[-1]


# --------------------------------------------------------------------------------------
# provider + pipeline behaviour
# --------------------------------------------------------------------------------------
def test_teacher_spec_requires_terms_note_and_flags_reuse():
    assert any("terms_note" in e for e in TeacherSpec(model_id="x").validate())
    assert TeacherSpec(model_id="x", terms_note="permitted").output_reuse_permitted is False


def test_request_plan_is_deterministic_and_bounded():
    graph = load_default_graph()
    cfg = GenerationConfig(
        kinds=("learn", "practice"),
        languages=("hi", "en"),
        difficulties=("beginner",),
        max_records_per_node=3,
        seed=5,
    )
    pipeline = TeacherPipeline(graph, ProviderRegistry(), cfg)
    first = [r.request_id for r in pipeline.plan_requests(["math.g05"])]
    second = [r.request_id for r in pipeline.plan_requests(["math.g05"])]
    assert first == second
    assert len(first) == 3, "max_records_per_node caps the plan"
    assert all(rid.startswith("math.g05.") for rid in first)


def test_request_prompt_forbids_hidden_chain_of_thought_and_requires_checkable_arithmetic():
    graph = load_default_graph()
    request = build_request(graph, "math.g05", "practice", "en", "intermediate")
    assert "Do not dump private step-by-step" in request.prompt
    assert "no placeholder text" in request.prompt.lower()
    assert "write it as an explicit statement" in request.prompt
    assert request.required_fields == ("task", "response", "answer", "rationale")


def test_pipeline_accepts_two_teacher_agreement_and_records_provenance():
    graph = load_default_graph()
    request = build_request(graph, "math.g05", "learn", "en", "beginner")
    payload = {"task": "Teach percentages", "response": "17% of 240 = 40.8, so multiply by the rate.", "answer": "40.8", "rationale": "Multiply the base by the rate."}
    a, b = StaticProvider("a", _spec("teacher_a")), StaticProvider("b", _spec("teacher_b"))
    a.add(request.request_id, dict(payload))
    b.add(request.request_id, dict(payload))
    cfg = GenerationConfig(kinds=("learn",), languages=("en",), difficulties=("beginner",), max_records_per_node=1)
    result = TeacherPipeline(graph, ProviderRegistry([a, b]), cfg).run(["math.g05"])
    assert len(result.accepted) == 1
    record = result.accepted[0]
    assert record.provenance == "synthetic"
    assert record.verification_status == "multi_teacher_agreement"
    assert record.teacher_models == ["teacher_a", "teacher_b"]
    assert record.validate() == []
    assert result.outcomes[0].status is ConsensusStatus.AGREED


def test_pipeline_quarantines_when_teacher_terms_are_unstated():
    graph = load_default_graph()
    request = build_request(graph, "math.g05", "learn", "en", "beginner")
    payload = {"task": "Teach percentages", "response": "Multiply the base by the rate to get 40.8.", "answer": "40.8", "rationale": "Rate times base."}
    a, b = StaticProvider("a", _spec("teacher_a", permitted=False)), StaticProvider("b", _spec("teacher_b", permitted=False))
    a.add(request.request_id, dict(payload))
    b.add(request.request_id, dict(payload))
    cfg = GenerationConfig(kinds=("learn",), languages=("en",), difficulties=("beginner",), max_records_per_node=1)
    result = TeacherPipeline(graph, ProviderRegistry([a, b]), cfg).run(["math.g05"])
    assert result.accepted == []
    assert len(result.quarantined) == 1
    assert "reuse terms unstated" in result.quarantined[0][1]
    assert result.quarantined[0][0].license_status == "unknown_quarantine"


def test_pipeline_requires_verification_for_transfer_items():
    graph = load_default_graph()
    request = build_request(graph, "math.g05", "transfer", "en", "beginner")
    payload = {"task": "A shop gives 25% off 800 rupees; what is the price?", "response": "800 - 200 = 600 rupees.", "answer": "600", "rationale": "Take 25% then subtract."}
    a, b = StaticProvider("a", _spec("teacher_a")), StaticProvider("b", _spec("teacher_b"))
    a.add(request.request_id, dict(payload))
    b.add(request.request_id, dict(payload))
    cfg = GenerationConfig(kinds=("transfer",), languages=("en",), difficulties=("beginner",), max_records_per_node=1)
    result = TeacherPipeline(graph, ProviderRegistry([a, b]), cfg).run(["math.g05"])
    assert result.accepted == []
    assert any("requires verification" in reason for _r, reason in result.quarantined)


def test_pipeline_reports_shortfall_when_no_provider_produces_anything():
    graph = load_default_graph()
    cfg = GenerationConfig(kinds=("learn",), languages=("en",), difficulties=("beginner",), max_records_per_node=1)
    result = TeacherPipeline(graph, ProviderRegistry(), cfg).run(["math.g05"])
    assert result.accepted == []
    assert result.draft_count == 0
    assert any("Nothing was fabricated" in n for n in result.notes)


def test_template_provider_is_marked_synthetic_and_unverified():
    graph = load_default_graph()
    cfg = GenerationConfig(kinds=("learn",), languages=("en",), difficulties=("beginner",), max_records_per_node=1)
    result = TeacherPipeline(graph, ProviderRegistry([TemplateProvider()]), cfg).run(["math.g05"])
    assert result.accepted == [], "a single template draft is not consensus and must not be accepted"
    assert result.draft_count == 1


def test_pipeline_result_writes_qc_report(tmp_path):
    graph = load_default_graph()
    request = build_request(graph, "math.g05", "learn", "en", "beginner")
    payload = {"task": "Teach percentages", "response": "Multiply by the rate: 40.8 for 17% of 240.", "answer": "40.8", "rationale": "Rate times base."}
    a, b = StaticProvider("a", _spec("teacher_a")), StaticProvider("b", _spec("teacher_b"))
    a.add(request.request_id, dict(payload))
    b.add(request.request_id, dict(payload))
    cfg = GenerationConfig(kinds=("learn",), languages=("en",), difficulties=("beginner",), max_records_per_node=1)
    result = TeacherPipeline(graph, ProviderRegistry([a, b]), cfg).run(["math.g05"])
    paths = result.write(tmp_path / "gen")
    assert paths["records"].exists() and paths["qc_report"].exists()
    import json

    qc = json.loads(paths["qc_report"].read_text(encoding="utf-8"))
    assert qc["accepted"] == 1
    assert qc["providers"][0]["provider"] == "a"
    assert qc["config_fingerprint"]


def test_recorded_provider_loads_a_transcript(tmp_path):
    import json

    draft = _draft("recorded_teacher", GOOD, request_id="math.g05.learn.en.beginner.00")
    path = tmp_path / "transcript.jsonl"
    path.write_text(json.dumps(draft.to_dict()) + "\n", encoding="utf-8")
    provider = RecordedProvider.from_jsonl(path)
    request = GenerationRequest(
        request_id="math.g05.learn.en.beginner.00",
        node_id="math.g05",
        kind="learn",
        language="en",
        difficulty="beginner",
        prompt="p",
    )
    assert len(provider.generate(request)) == 1
    assert provider.spec.model_id == "recorded_teacher"
    other = GenerationRequest(request_id="other", node_id="x", kind="learn", language="en", difficulty="beginner", prompt="p")
    assert provider.generate(other) == []


def test_generation_config_rejects_unknown_kinds():
    with pytest.raises(ValueError):
        GenerationConfig(kinds=("learn", "telepathy")).check()
    with pytest.raises(ValueError):
        GenerationConfig(require_verification_for=("nonsense",)).check()
