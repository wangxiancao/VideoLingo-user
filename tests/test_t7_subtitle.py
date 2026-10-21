import json
import re
import shutil
from pathlib import Path

from pipeline.runner import run

REPO_ROOT = Path(__file__).resolve().parents[1]
TRANSLATION = REPO_ROOT / "tests/fixtures/translation/sample_6s.json"
SEGMENTS = REPO_ROOT / "tests/fixtures/segments/sample_6s.json"
SRT_TIME = re.compile(r"^\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}$")


def _config(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("{}\n", encoding="utf-8")
    return path


def _workdir(tmp_path):
    workdir = tmp_path / "work"
    workdir.mkdir()
    shutil.copyfile(TRANSLATION, workdir / "translation.json")
    shutil.copyfile(SEGMENTS, workdir / "segments.json")
    return workdir


def _blocks(path):
    text = Path(path).read_text(encoding="utf-8").replace("\r\n", "\n")
    return [block.strip() for block in text.split("\n\n") if block.strip()]


def test_subtitle_writes_three_consistent_srt_files(tmp_path):
    workdir = _workdir(tmp_path)

    assert run("subtitle", workdir, _config(tmp_path)) == 0

    source = _blocks(workdir / "subs/source.srt")
    target = _blocks(workdir / "subs/target.srt")
    bilingual = _blocks(workdir / "subs/bilingual.srt")
    assert len(source) == len(target) == len(bilingual) > 2

    source_times = []
    target_times = []
    for index, (source_block, target_block, bilingual_block) in enumerate(
        zip(source, target, bilingual), start=1
    ):
        source_lines = source_block.split("\n")
        target_lines = target_block.split("\n")
        bilingual_lines = bilingual_block.split("\n")
        assert source_lines[0] == target_lines[0] == bilingual_lines[0] == str(index)
        assert SRT_TIME.match(source_lines[1])
        assert source_lines[1] == target_lines[1] == bilingual_lines[1]
        source_times.append(source_lines[1])
        target_times.append(target_lines[1])

        source_text = "\n".join(source_lines[2:])
        target_text = "\n".join(target_lines[2:])
        bilingual_text = bilingual_lines[2:]
        assert len(bilingual_text) == 2
        assert bilingual_text == source_text.split("\n") + target_text.split("\n")
        assert len(source_text) <= 75
        assert len(target_text) <= 90
    assert source_times == target_times


def test_subtitle_validate_only_accepts_generated_set(tmp_path):
    workdir = _workdir(tmp_path)
    assert run("subtitle", workdir, _config(tmp_path)) == 0
    assert run("subtitle", workdir, _config(tmp_path), validate_only=True) == 0


def test_bilingual_srt_must_have_two_lines(tmp_path):
    workdir = _workdir(tmp_path)
    config = _config(tmp_path)
    assert run("subtitle", workdir, config) == 0

    path = workdir / "subs/bilingual.srt"
    blocks = _blocks(path)
    lines = blocks[0].split("\n")
    lines.insert(2, "extra line")
    blocks[0] = "\n".join(lines)
    path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")

    assert run("subtitle", workdir, config, validate_only=True) == 1