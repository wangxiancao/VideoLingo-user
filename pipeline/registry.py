from pipeline.contracts import Artifact, StageSpec
from pipeline.stages.burn import run as run_burn
from pipeline.stages.ingest import run as run_ingest
from pipeline.stages.segment import run as run_segment
from pipeline.stages.subtitle import run as run_subtitle
from pipeline.stages.transcribe import run as run_transcribe
from pipeline.stages.translate import run as run_translate


ARTIFACTS = {
    "source_video": Artifact("source_video", "media/source.mp4", "media"),
    "audio": Artifact("audio", "media/audio.wav", "media"),
    "transcript": Artifact("transcript", "transcript.json", "json"),
    "segments": Artifact("segments", "segments.json", "json"),
    "translation": Artifact("translation", "translation.json", "json"),
    "srt_source": Artifact("srt_source", "subs/source.srt", "srt"),
    "srt_target": Artifact("srt_target", "subs/target.srt", "srt"),
    "srt_bilingual": Artifact("srt_bilingual", "subs/bilingual.srt", "srt"),
    "output_video": Artifact("output_video", "output.mp4", "media"),
}


STAGES = (
    StageSpec(
        "ingest",
        "Ingest video and extract audio",
        (),
        ("source_video", "audio"),
        ("_1_ytdlp",),
        run_ingest,
        external_input="source",
    ),
    StageSpec(
        "transcribe",
        "Transcribe audio with word timestamps",
        ("audio",),
        ("transcript",),
        ("_2_asr",),
        run_transcribe,
    ),
    StageSpec(
        "segment",
        "Split transcript into natural sentences",
        ("transcript",),
        ("segments",),
        ("_3_1_split_nlp", "_3_2_split_meaning"),
        run_segment,
    ),
    StageSpec(
        "translate",
        "Translate segments in two steps",
        ("segments",),
        ("translation",),
        ("_4_1_summarize", "_4_2_translate"),
        run_translate,
    ),
    StageSpec(
        "subtitle",
        "Generate source, target and bilingual SRT files",
        ("translation",),
        ("srt_source", "srt_target", "srt_bilingual"),
        ("_5_split_sub", "_6_gen_sub"),
        run_subtitle,
    ),
    StageSpec(
        "burn",
        "Burn subtitles into the source video",
        ("source_video", "srt_source", "srt_target"),
        ("output_video",),
        ("_7_sub_into_vid",),
        run_burn,
    ),
)


def validate_registry(stages: tuple[StageSpec, ...] = STAGES) -> None:
    ids = [stage.id for stage in stages]
    if len(set(ids)) != len(ids):
        raise ValueError("stage ids must be unique")

    for stage in stages:
        if stage.external_input not in (None, "source"):
            raise ValueError(
                f"stage '{stage.id}' has unknown external input: {stage.external_input}"
            )
        for key in stage.requires + stage.produces:
            if key not in ARTIFACTS:
                raise ValueError(f"stage '{stage.id}' references unknown artifact '{key}'")

    for index, stage in enumerate(stages):
        for key in stage.requires:
            producer_index = next(
                (
                    producer_index
                    for producer_index, candidate in enumerate(stages)
                    if key in candidate.produces
                ),
                None,
            )
            if producer_index is not None and producer_index >= index:
                raise ValueError(
                    f"stage '{stage.id}' requires '{key}' before it is produced"
                )

    for key in ARTIFACTS:
        producers = [stage.id for stage in stages if key in stage.produces]
        if len(producers) > 1:
            raise ValueError(f"artifact '{key}' has multiple producers: {producers}")


def producer_of(key: str, stages: tuple[StageSpec, ...] = STAGES) -> str | None:
    for stage in stages:
        if key in stage.produces:
            return stage.id
    return None