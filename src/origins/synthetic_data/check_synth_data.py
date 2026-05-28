import re
from collections import defaultdict

import yaml
import fire


DEFAULT_IGNORE_WORDS = {
    "bias", "prejudice", "stereotype", "law", "system", "syndrome", "disorder",
    "act", "policy", "decree", "test", "index", "probe", "scan", "monitor",
    "of", "the", "a", "an", "and", "in", "to", "for", "is", "on",
}

# Key used by the generator to attach the on-disk path to an in-memory datapoint
# so the duplicate report can point at the file the user needs to fix.
SOURCE_PATH_KEY = "_source_path"


def find_imaginary_duplicates(
    datapoints: list[dict],
    ignore_words: set[str] | None = None,
) -> dict[str, list[str]]:
    """Return a mapping of duplicated word -> list of locations where it appears.

    Each location is a human-readable string. If a datapoint carries a
    ``_source_path`` key, that path is included so the report points at the file
    that needs editing.

    Same-pair head/tail token reuse is *not* counted as a duplicate (e.g., head
    "Voron" + tail "Voron's Field Unification" is one logical occurrence of
    ``voron``). Each unique pair contributes each of its tokens at most once.
    Duplicates are reported when the same token appears in two different pairs,
    whether those pairs are in the same datapoint or in different datapoints.
    """
    if ignore_words is None:
        ignore_words = DEFAULT_IGNORE_WORDS

    word_locations: dict[str, list[str]] = defaultdict(list)
    for entry in datapoints:
        uid = entry.get("uid", "Unknown UID")
        topic_category = entry.get("topic_category", "unknown")
        source_path = entry.get(SOURCE_PATH_KEY)
        imaginary_pairs = entry.get("instantiations", {}).get("imaginary_pairs", []) or []

        for i, pair in enumerate(imaginary_pairs):
            head_text = pair.get("head", "") or ""
            tail_text = pair.get("tail", "") or ""

            pair_tokens: set[str] = set()
            for text in (head_text, tail_text):
                for word in re.findall(r"\b[a-z]+\b", text.lower()):
                    if word in ignore_words:
                        continue
                    pair_tokens.add(word)

            for word in pair_tokens:
                parts = [topic_category, f"UID {uid}"]
                if source_path:
                    parts.append(source_path)
                parts.extend(
                    [f"Pair {i + 1}", f"HEAD: '{head_text}'", f"TAIL: '{tail_text}'"]
                )
                word_locations[word].append(" | ".join(parts))

    return {word: locs for word, locs in word_locations.items() if len(locs) > 1}


def format_duplicates(duplicates: dict[str, list[str]]) -> str:
    if not duplicates:
        return "Success: zero duplicate fantasy words found across all imaginary pairs."

    lines = [f"Found {len(duplicates)} duplicate words (excluding ignored words):", ""]
    for word, locs in duplicates.items():
        lines.append(f"Word: '{word.upper()}' (Found {len(locs)} times)")
        for loc in locs:
            lines.append(f"  - {loc}")
        lines.append("-" * 60)
    return "\n".join(lines)


def check_imaginary_duplicates(yaml_path: str) -> None:
    """CLI entry point: load a yaml file (list of datapoints) and print duplicates.

    python -m src.origins.synthetic_data.check_synth_data \
        --yaml_path='data/synthetic_data/synth_data_v4/updated_dp.yaml'
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    duplicates = find_imaginary_duplicates(data)
    print(format_duplicates(duplicates))


if __name__ == "__main__":
    fire.Fire(check_imaginary_duplicates)
