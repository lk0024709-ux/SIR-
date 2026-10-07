from pathlib import Path
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sir_human_development.yaml"


def test_generation_order_is_stable_and_complete():
    data = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    ids = [g["id"] for g in data["generations"]]
    names = [g["name"] for g in data["generations"]]
    assert ids == ["nano", "lite", "flash_lite", "flash", "pro", "pro_plus", "pro_max", "ultra", "expert"]
    assert names == [
        "SIR-Nano", "SIR-Lite", "SIR-Flash Lite", "SIR-Flash",
        "SIR-Pro", "SIR-Pro+", "SIR-Pro Max", "SIR-Ultra", "SIR-Expert",
    ]
    assert len(ids) == len(set(ids))


def test_development_axes_include_brainstorming():
    data = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    required = {
        "knowledge", "understanding", "reasoning", "application",
        "verification", "experience", "transfer", "continual_learning", "brainstorming",
    }
    assert required.issubset(set(data["development_axes"]))


def test_foundation_curriculum_is_broad():
    data = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    subjects = set(data["curriculum"]["foundation"]["class_1_to_10"]["subjects"])
    required = {
        "mathematics", "science", "social_science", "hindi", "english",
        "computer_science", "logical_reasoning", "indian_languages",
        "scientific_thinking", "indian_knowledge", "world_knowledge",
    }
    assert required.issubset(subjects)


def test_advanced_curriculum_exists():
    data = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    advanced = set(data["curriculum"]["advanced"])
    assert {"class_11_12", "undergraduate", "advanced_mathematics", "ai_ml", "research_methodology"}.issubset(advanced)


def test_honesty_gates_are_enabled():
    data = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    gates = data["gates"]
    assert gates["no_fake_metrics"] is True
    assert gates["no_unverified_capability_claims"] is True
    assert gates["require_evaluation_before_generation_promotion"] is True
