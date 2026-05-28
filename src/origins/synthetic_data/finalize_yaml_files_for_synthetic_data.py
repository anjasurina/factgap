import os
from pathlib import Path
import yaml
import random
import re

import inflect
import fire

# Initialize the engine
inflect_engine = inflect.engine()


def flawless_a_an_fix(text):
    def replacer(match):
        original_article = match.group(1)
        following_word = match.group(2)

        # p.a() automatically returns "a [word]" or "an [word]" based on phonetics
        corrected_phrase = inflect_engine.a(following_word)

        # Preserve original capitalization (e.g., if it started a sentence)
        if original_article.istitle():
            corrected_phrase = corrected_phrase.capitalize()

        return corrected_phrase

    # Find any standalone "a" or "an" followed by a word
    return re.sub(r'\b([Aa]n?)\s+([A-Za-z0-9]+)', replacer, text)


class SkipFile(Exception):
    """Raised when a YAML file should be skipped during conversion."""


def load_yaml(path: Path):
    """Load YAML from *path* and return the parsed object."""
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def dump_yaml(data: dict, path: Path):
    """Write *data* as YAML to *path*. Creates parent directories if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False, allow_unicode=True)


def slugify(value: str) -> str:
    """Return a lowercase ASCII slug suitable for filenames/identifiers."""
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()


def build_entities(raw: dict, real_map: dict[str, bool]) -> tuple[dict, list[str], list[str]]:
    """Create entities mapping and return unchosen relationship_head values.

    New instantiations format:
      instantiations:
        real_pairs: [{head: "...", tail: "..."}, ...]
        imaginary_pairs: [{head: "...", tail: "..."}, ...]

    Mapping:
      head -> relationship_head
      tail -> topic
    """
    entities: dict = {}
    unchosen_heads: list[str] = []
    leftover_topics: list[str] = []

    instantiations = raw.get("instantiations", {})
    use_real = bool(real_map.get("relationship_head", False))
    pairs = instantiations.get(
        "real_pairs" if use_real else "imaginary_pairs", [])
    if not pairs:
        # fallback to the other pool if empty
        pairs = instantiations.get(
            "imaginary_pairs" if use_real else "real_pairs", [])

    # Collect heads/tails
    heads = [p.get("head") for p in pairs if p.get("head")]
    tails = [p.get("tail") for p in pairs if p.get("tail")]

    # topic (tail): pick one
    if tails:
        chosen_topic = random.choice(tails)
        entities["topic"] = chosen_topic
        leftover_topics = [t for t in tails if t != chosen_topic]

    # relationship_head (head): pick learn/update
    if heads:
        unique_heads = list(dict.fromkeys(heads))
        if len(unique_heads) >= 2:
            learn_val, update_val = random.sample(unique_heads, 2)
        else:
            learn_val = update_val = unique_heads[0]
        entities["relationship_head"] = {
            "learn": learn_val, "update": update_val}
        unchosen_heads = [
            h for h in unique_heads if h not in {learn_val, update_val}]

    return entities, unchosen_heads, leftover_topics


def convert_file(
    input_path: Path,
    output_path: Path,
    real_map: dict[str, bool],
    control_topic_real: bool,
    control_target_real: bool,
    train_data_type: str = "all"
) -> Path | None:
    """Convert a single synthetic YAML *input_path* and write the converted
    YAML to *output_path*.

    Returns the *output_path* on success.

    Args:
        input_path (Path): Path to the input synthetic YAML file.
        output_path (Path): Path where the converted YAML file will be written.
        real_map (dict[str, bool]): Mapping of placeholders to whether to use
            real (True) or imaginary (False) values.
        control_topic_real (bool): Whether to use real topics for control problems.
        control_target_real (bool): Whether to use real targets for control answers.
        train_data_type (str): Type of training data to include; currently unused.

    Returns:
        Path: The path to the output YAML file.

    """
    raw = load_yaml(input_path)
    if raw.get('skip', False):
        return None

    # Build entities and collect leftovers
    entities, _leftover_heads_unused, _ = build_entities(raw, real_map)

    # Build the output structure
    topic_category_raw = raw.get("topic_category", "")
    topic_raw = entities.get("topic") or raw.get("topic", "")
    task_id_components = [
        slugify(topic_category_raw), slugify(topic_raw), input_path.stem]
    task_id = "_".join([c for c in task_id_components if c])

    output: dict = {
        "entities": entities,
        "correct_problems": [],
        "control_problems": [],
        "control_answers": [],
        "task_id": task_id,
    }

    sentences = [
        {"sentence": s.get("sentence", ""),
         "version": str(s.get("version", ""))}
        for s in raw.get("sentences", [])
    ]
    scenarios = [
        {"sentence": s.get("scenario", ""),
         "scenario_key": s.get("scenario_key", ""),
         "version": str(s.get("version", ""))}
        for s in raw.get("scenarios", [])
    ]
    if train_data_type == "sentences":
        train_sentences = sentences
    elif train_data_type == "scenarios":
        train_sentences = scenarios
    elif train_data_type == "all":
        train_sentences = sentences + scenarios
    else:
        raise ValueError(
            f"Invalid train_data_type '{train_data_type}'; must be one of 'sentences', 'scenarios', 'all'.")

    if not train_sentences:
        raise ValueError(
            f"No training sentences or scenarios found in file '{input_path}'.")

    output["train_sentences"] = train_sentences

    # Find questions that match the criteria:
    # - Question contains {topic}
    # - Question does NOT contain {relationship_head}
    # - Answer contains {relationship_head}
    matching_questions = []
    for q in raw.get("questions", []):
        question_text = q.get("question", "")
        answer_text = q.get("answer", "")

        has_topic = "{topic}" in question_text
        has_head_in_question = "{relationship_head}" in question_text
        has_head_in_answer = "{relationship_head}" in answer_text

        if has_topic and not has_head_in_question and has_head_in_answer:
            matching_questions.append({
                "question": question_text,
                "answer": answer_text,
                "version": q.get("version", "1")
            })

    if not matching_questions:
        raise SkipFile(
            "No question found with {topic} in question (not {relationship_head}), {relationship_head} in answer; skipping file."
        )

    # Select a single correct problem from matching questions
    target_problem = random.choice(matching_questions)
    output["correct_problems"].append({
        "problem": target_problem["question"],
        "answer": target_problem["answer"],
        "version": "1",
    })

    # Build single control problem using alternative topic (tail)
    inst_pairs = raw.get("instantiations", {})
    desired_pairs_key = "real_pairs" if control_topic_real else "imaginary_pairs"
    topic_source_list = [p.get("tail") for p in inst_pairs.get(
        desired_pairs_key, []) if p.get("tail")]
    main_topic_val = entities.get("topic")
    topic_source_list = [t for t in topic_source_list if t != main_topic_val]

    if topic_source_list:
        control_topic = random.choice(topic_source_list)
        control_question = target_problem["question"].replace(
            "{topic}", control_topic)
        output["control_problems"].append({
            "problem": control_question,
            "answer": target_problem["answer"],
            "version": "1",
        })

    # Build control answers from heads
    desired_pairs_key = "real_pairs" if control_target_real else "imaginary_pairs"
    head_pool = [p.get("head") for p in inst_pairs.get(
        desired_pairs_key, []) if p.get("head")]
    chosen_head_vals = set(entities.get("relationship_head", {}).values()) if isinstance(
        entities.get("relationship_head"), dict) else set()
    control_heads = [h for h in head_pool if h not in chosen_head_vals]
    output["control_answers"] = control_heads

    dump_yaml(output, output_path)
    return output_path


def walk_synthetic_dir(root: Path):
    """Yield all YAML files under *root* recursively."""
    for path in root.rglob("*.yaml"):
        yield path


def main(
    input_root: str | Path,
    output_root: str | Path,
    real_relationship_head: bool = False,
    real_relationship_tail: bool = False,
    real_control_relationship_tail: bool = False,
    real_control_relationship_heads: bool = False,
    train_data_type: str = "sentences",
    seed: int = 0,
):
    """Finalize synthetic YAML files by converting them into inference-task format.

    Behaviour
    ---------
    1. Recursively walks *input_root* collecting every ``*.yaml`` file.
    2. Each file is parsed and passed to :pyfunc:`convert_file` which performs
       schema conversion and writes a new YAML file to *output_root*.
    3. For each input file the resulting output filename is ``{task_id}.yaml``
       where *task_id* is taken from the original ``topic`` field or, if that
       is missing, from the parent directory name.
    4. Prints a summary of how many files were converted.

    Examples
    --------
    Convert using *imaginary* entity values (default):

    python3 -m src.origins.synthetic_data.finalize_yaml_files_for_synthetic_data \
     --input_root data/synthetic_data/synth_dataset_v4/updated \
     --output_root data/synthetic_data/synth_dataset_v4_formatted_imaginary_sentences \
     --real_control_relationship_tail=True --real_control_relationship_heads=True \
     --train_data_type sentences

    Convert using *real* entity values:

    python3 -m src.origins.synthetic_data.finalize_yaml_files_for_synthetic_data \
     --input_root data/synthetic_data/synth_dataset_v4/updated \
     --output_root data/synthetic_data/synth_dataset_v4_formatted_real_sentences \
     --real_relationship_head=True --real_relationship_tail=True --real_control_relationship_tail=True --real_control_relationship_heads=True \
     --train_data_type sentences

    Args:
        input_root (str): Root directory of the synthetic YAML files.
        output_root (str): Directory where converted YAML files will be written.
        real_relationship_head (bool): Use real entries for the relationship_head placeholder.
        real_relationship_tail (bool): Use real entries for the topic (tail) placeholder.
        real_control_relationship_tail (bool): Use real topics for control problems.
        real_control_relationship_heads (bool): Use real relationship_head values for control answers.
        train_data_type (str): 'sentences', 'scenarios', or 'all'.
        seed (int): Random seed for reproducibility.
    """
    random.seed(seed)

    input_root = Path(input_root)
    output_root = Path(output_root)
    real_map = {
        "relationship_head": bool(real_relationship_head),
        "relationship_tail": bool(real_relationship_tail),
    }
    control_topic_real = bool(real_control_relationship_tail)
    control_target_real = bool(real_control_relationship_heads)

    if not input_root.exists():
        raise FileNotFoundError(f"Input path '{input_root}' does not exist.")

    converted_paths = []
    skipped_files = []
    for yaml_path in walk_synthetic_dir(input_root):
        relative = yaml_path.relative_to(input_root)
        out_path = output_root / relative
        try:
            out = convert_file(
                input_path=yaml_path,
                output_path=out_path,
                real_map=real_map,
                control_topic_real=control_topic_real,
                control_target_real=control_target_real,
                train_data_type=train_data_type
            )
            if out is not None:
                raise SkipFile
        except SkipFile as e:
            skipped_files.append(yaml_path)
            print(f"[skip] {yaml_path}: {e}")

    print(f"Converted {len(converted_paths)} files, skipped {len(skipped_files)} files without {{relationship_head}} answer -> {output_root}")


if __name__ == "__main__":
    fire.Fire(main)
