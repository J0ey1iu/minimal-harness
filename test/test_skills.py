"""RFC #57 SKILL.md frontmatter parser tests."""

from __future__ import annotations

from minimal_harness.skills import (
    FrontmatterError,
    SkillFrontmatter,
    install_from_folder,
    parse_skill_md,
    render_skill_md,
)

SAMPLE = """\
---
name: my-skill
description: When to use this skill
version: 1.2.0
license: MIT
vendor: acme
---

# Body heading

Some markdown body.
"""


def test_parse_basic_frontmatter():
    fm, body = parse_skill_md(SAMPLE)
    assert fm.name == "my-skill"
    assert fm.description == "When to use this skill"
    assert fm.version == "1.2.0"
    assert fm.license == "MIT"
    assert fm.extra == {"vendor": "acme"}
    assert body.startswith("# Body heading")


def test_parse_validates_name():
    assert not SkillFrontmatter(name="ok-skill", description="d").validate()
    assert SkillFrontmatter(name="Bad_Name", description="d").validate()
    assert SkillFrontmatter(name="", description="d").validate()
    assert SkillFrontmatter(name="a" * 65, description="d").validate()
    assert SkillFrontmatter(name="ok", description="").validate()
    assert SkillFrontmatter(name="ok", description="x" * 1025).validate()


def test_no_frontmatter_returns_empty_and_full_body():
    fm, body = parse_skill_md("# just markdown\nno yaml here")
    assert fm.name == ""
    assert fm.description == ""
    assert body == "# just markdown\nno yaml here"


def test_non_mapping_frontmatter_raises():
    with pytest_raises(FrontmatterError):
        parse_skill_md("---\n- a\n- b\n---\nbody")


def pytest_raises(exc):  # tiny alias kept local
    import pytest

    return pytest.raises(exc)


def test_render_roundtrip_roundtrip():
    fm, body = parse_skill_md(SAMPLE)
    rendered = render_skill_md(fm, body)
    fm2, body2 = parse_skill_md(rendered)
    assert fm2.name == fm.name
    assert fm2.description == fm.description
    assert fm2.version == fm.version
    assert fm2.license == fm.license
    assert fm2.extra == fm.extra
    assert body2 == body


def test_render_preserves_unknown_keys():
    fm = SkillFrontmatter(
        name="s", description="d", extra={"x": 1, "nested": {"k": "v"}}
    )
    rendered = render_skill_md(fm, "body text")
    fm2, _ = parse_skill_md(rendered)
    assert fm2.extra == {"x": 1, "nested": {"k": "v"}}


def test_install_from_folder(tmp_path):
    skill_dir = tmp_path / "my-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(SAMPLE, encoding="utf-8")
    meta = install_from_folder(skill_dir)
    assert meta.frontmatter.name == "my-skill"
    assert meta.body.startswith("# Body heading")


def test_install_from_folder_missing_skill_md(tmp_path):
    with pytest_raises(FrontmatterError):
        install_from_folder(tmp_path)
