from jinja2 import Environment, FileSystemLoader, select_autoescape
from typing import Optional
import re
import yaml
import yaml.scanner
import fire
import os


TEMPLATE_FOLDER = './data/prompt_templates'


def render_template(
        template_name: str,
        user_data: Optional[dict] = None,
        template_folder: Optional[str] = None
) -> str:
    """
    Render a Jinja2 template with user data and automatic whitespace control.

    Args:
        template_name: Name of the template file (e.g., 'task.html')
        user_data: Dictionary of variables to pass to the template
        template_folder: Optional folder path where templates are stored.
                         Defaults to TEMPLATE_FOLDER if not provided.

    Returns:
        Rendered template as string
    """

    if user_data is None:
        user_data = {}

    if template_folder is None:
        template_folder = TEMPLATE_FOLDER

    # Ensure template_dirs is a list
    template_dirs = [os.path.join(template_folder, f)
                     for f in os.listdir(template_folder)] + [template_folder]

    # Create environment with whitespace control
    env = Environment(
        loader=FileSystemLoader(template_dirs),
        autoescape=select_autoescape(['html', 'xml']),
        # Automatic whitespace trimming
        trim_blocks=True,      # Remove newline after a block
        lstrip_blocks=True,    # Remove spaces/tabs at beginning of line
        keep_trailing_newline=False  # Don't keep final newline
    )

    # Load and render template
    template = env.get_template(template_name)
    rendered = template.render(**user_data)

    return rendered


def parse_yaml_response(
    yaml_str: str,
    expected_keys: Optional[list[str]] = None,
    verbose: bool = False
) -> dict:
    """
    Main entry point for parsing LLM YAML responses.

    Args:
        yaml_str: The raw YAML string from the LLM response.
        expected_keys: Optional list of keys that must be present in the output.
        verbose: If True, prints debug information.
    Returns:
        Parsed dictionary from YAML.
    """
    # 1. Clean up Markdown tags
    match = re.search(r"```(?:\w+:?)?(.*?)```", yaml_str, re.DOTALL)
    content = match.group(1).strip() if match else yaml_str.strip()

    # 2. Attempt Primary Parsing (Standard YAML)
    try:
        data = yaml.safe_load(content)

        # VALIDATION CHECK:
        # We only accept the standard parse if it resulted in a dict
        if isinstance(data, dict):
            if expected_keys:
                # Check that every expected key exists and is not None/Empty
                is_valid = all(
                    key in data and data[key] is not None and str(
                        data[key]).strip()
                    for key in expected_keys
                )
                if is_valid:
                    if verbose:
                        print("Standard YAML parsed successfully with all keys.")
                    return data
                elif verbose:
                    print(
                        "Standard YAML parsed but failed validation (missing/empty keys). Triggering fallback...")
            else:
                return data

    except (yaml.YAMLError, yaml.scanner.ScannerError) as e:
        if verbose:
            print(f"Standard YAML parsing failed: {e}. Triggering fallback...")

    if expected_keys is not None:
        return _fallback_parse_multikey(content, expected_keys, verbose)
    else:
        if verbose:
            print("No expected keys provided for fallback parsing. Returning empty dict.")
        return {}


def _fallback_parse_multikey(
    content: str,
    keys: list[str],
    verbose: bool = False
) -> dict:
    """
    Advanced Fallback Parser.
    Attempts to extract values for a list of specific keys using Regex.
    Assumes keys appear in the text as "KeyName: Value".

    Args:
        content: The raw text content to parse.
        keys: List of keys to extract from the content.
        verbose: If True, prints debug information.
    Returns:
        Dictionary of extracted key-value pairs.
    """
    result = {}

    # Sort keys by length (descending) to avoid partial matches causing issues
    # (e.g. matching 'description' inside 'meta_description')
    sorted_keys = sorted(keys, key=len, reverse=True)

    for key in sorted_keys:
        # LOGIC:
        # 1. We look for the key starting a line (^\s*{key}:)
        # 2. We capture everything after it (?P<val>.*?)
        # 3. UNTIL we hit the start of another expected key OR the end of the string.

        # Build a regex group of all OTHER keys to use as a "stop" signal
        other_keys = [re.escape(k) for k in keys if k != key]
        if other_keys:
            # (?= ... ) is a positive lookahead. It stops the capture when it sees the pattern ahead.
            # We look for: Start of line + any other key + colon
            stop_pattern = f"(?=^\\s*(?:{'|'.join(other_keys)}):|\\Z)"
        else:
            # Read until end of string if no other keys
            stop_pattern = r"(?=\Z)"

        # Full Pattern:
        # ^\s* -> Start of line with optional indent
        # {key}:     -> The key we want + colon
        # \s* -> Optional spaces
        # [|>\"]?    -> Optional YAML block indicator or quote
        # \s* -> Optional spaces/newline
        # (?P<val>.*?) -> CAPTURE content non-greedily...
        # {stop_pattern} -> ...until hitting another key or EOF
        pattern = re.compile(
            rf"^\s*{re.escape(key)}:\s*[|>\"]?\s*(?P<val>.*?){stop_pattern}",
            re.DOTALL | re.MULTILINE
        )

        match = pattern.search(content)
        if match:
            raw_value = match.group("val").strip()

            # Cleanup Artifacts:
            # 1. Remove trailing quotes if it looks like a Python string
            if raw_value.endswith('"""'):
                raw_value = raw_value[:-3]
            elif raw_value.endswith('"') and not raw_value.endswith('\\"'):
                raw_value = raw_value[:-1]

            # 2. Unescape quotes
            raw_value = raw_value.replace('\\"', '"').replace("\\'", "'")

            result[key] = raw_value

    if verbose:
        found = list(result.keys())
        missing = [k for k in keys if k not in result]
        print(f"Fallback extraction: Found {found}, Missing {missing}")

    return result


def _main(template_name: str, user_data: Optional[dict] = None):

    if user_data is None:
        user_data = {
            "hidden_coordinates": True,
            "map_str": "<some map>"
        }

    # Example usage
    template = render_template(
        template_name=template_name,
        user_data=user_data
    )
    print(template)


if __name__ == "__main__":
    fire.Fire(_main)
