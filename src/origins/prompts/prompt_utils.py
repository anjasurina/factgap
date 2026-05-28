import os
import re
import random
from jinja2 import Environment, FileSystemLoader, meta, exceptions
from origins.custom_classes import PromptType

from accelerate.logging import get_logger
logger = get_logger(__name__)

TEMPLATE_DIR = "src/origins/prompts/prompt_templates"

def _ensure_response_float(response) -> float | None:
    if isinstance(response, float):
        return response
    elif isinstance(response, int):
        # Convert int to float
        return float(response)
    elif isinstance(response, str):
        try:
            return float(response)
        except ValueError:
            return None
    return None


def _ensure_response_int(response) -> int | None:
    if isinstance(response, int):
        return response
    elif isinstance(response, float) and response is not None:
        # Convert float to int if it's a whole number
        return int(response)
    elif isinstance(response, str) and response.isdigit():
        return int(response)
    
    return None


def _ensure_response_str(response) -> str | None:
    if isinstance(response, str):
        response_c = response.strip()
         # Remove surrounding quotes if they exist
        if (response_c.startswith('"') and response_c.endswith('"')) or \
            (response_c.startswith("'") and response_c.endswith("'")):
            response_c = response_c[1:-1] # Slice off the first and last char
        return response_c
    else:
        return None


def _ensure_response_bool(response) -> bool | None:
    if isinstance(response, str) and len(response) > 2:
        response_cleaned = response.lower().strip()
        
         # Remove surrounding quotes if they exist
        if (response_cleaned.startswith('"') and response_cleaned.endswith('"')) or \
            (response_cleaned.startswith("'") and response_cleaned.endswith("'")):
            response_cleaned = response_cleaned[1:-1] # Slice off the first and last char

        if response_cleaned == "true":
            return True
        elif response_cleaned == "false":
            return False
    elif isinstance(response, bool):
        return response

    return None


def _ensure_response_list(response) -> list | None:
    if isinstance(response, list):
        return response
    elif isinstance(response, str):
        # Attempt to parse the string as a list
        try:
            parsed_list = eval(response)
            if isinstance(parsed_list, list):
                return parsed_list
        except (SyntaxError, NameError):
            pass  # Ignore parsing errors
    
    return None


def _ensure_response_dict(response) -> dict | None:
    if isinstance(response, dict):
        return response
    elif isinstance(response, str):
        # Attempt to parse the string as a dictionary
        try:
            parsed_dict = eval(response)
            if isinstance(parsed_dict, dict):
                return parsed_dict
        except (SyntaxError, NameError):
            pass  # Ignore parsing errors
    
    return None


def ensure_base_response_types(response: dict, key_types: dict[str, str]) -> dict:
    """
    Checks if the keys in the response dictionary match the expected types.

    Args:
        response: A dictionary containing key-value pairs to check.
        key_types: A dictionary where keys are expected keys and values are
                   the expected types (as strings, e.g., "str", "bool").

    Returns:
        A dictionary with keys from the response and boolean values indicating
        whether each key's type matches the expected type.
    """
    result = {}
    for key, expected_type in key_types.items():
        if key in response:
            # Check if the type of the value matches the expected type
            if expected_type == "str":
                result[key] = _ensure_response_str(response.get(key))
            elif expected_type == "bool":
                result[key] = _ensure_response_bool(response.get(key))
            elif expected_type == "int":
                result[key] = _ensure_response_int(response.get(key))
            elif expected_type == "float":
                result[key] = _ensure_response_float(response.get(key))
            elif expected_type == "list":
                result[key] = _ensure_response_list(response.get(key))
            elif expected_type == "dict":
                result[key] = _ensure_response_dict(response.get(key))
            else:
                pass

    return result


def parse_text_in_tags(keys_with_types: dict[str, str], text_to_search: str) -> dict:
    """
    Parses a string to find text enclosed within specified tags.

    Args:
        keys_with_types: A list of strings, where each string is a tag name (e.g., "reasoning").
              The function will look for <key>content</key>.
        text_to_search: The string to parse.

    Returns:
        A dictionary where keys are the input tag names and values are the
        extracted text content. If a tag is not found, its corresponding
        value in the dictionary will be None.
    """
    extracted_data = {}
    for key in keys_with_types:
        # Construct the regex pattern for each key.
        pattern = re.compile(f"<{key}>(.*?)</{key}>", re.DOTALL)

        # Search for the pattern in the text
        match = pattern.search(text_to_search)

        if match:
            extracted_data[key] = match.group(1).strip()
        else:
            extracted_data[key] = None
    
    extracted_data_checked = ensure_base_response_types(extracted_data, keys_with_types)

    return extracted_data_checked


def get_parse_keys_with_types(prompt_type: PromptType, include_reasoning: bool = True) -> dict[str, str]:
    
    """
    Returns a dictionary of keys and their expected types for a given template.

    Args:
        prompt_type: The type of prompt.
        include_reasoning: Whether to include reasoning in the response.

    Returns:
        A dictionary where keys are the expected keys and values are the
        expected types (as strings, e.g., "str", "bool").
    """
    # Define the expected keys and their types for each template
    if prompt_type in [PromptType.DOUBLE_CRITIC, PromptType.DOUBLE_CRITIC_MC]:
        kwt = {
            "response": "bool"
        }
    elif prompt_type in [PromptType.GENERATIVE_FREE]:
        kwt = {
            "answer": "str"
        }
    elif prompt_type in [PromptType.GENERATIVE_MC]:
        kwt = {
            "answer_letter": "str",
        }
    else:
        raise NotImplementedError(
            f"Template '{prompt_type}' is not implemented for key type extraction."
        )
    
    if include_reasoning:
        kwt["reasoning"] = "str"
    
    return kwt


def render_j2_template(data: dict, template_name: str, template_dir: str = TEMPLATE_DIR) -> str:
    """
    Renders a Jinja2 template with the given data after checking for key presence.

    Args:
        data: A dictionary containing key-value pairs for template rendering.
        template_name: The filename of the Jinja2 template (e.g., "my_template.j2").
        template_dir: The directory where templates are stored. 
                      Defaults to TEMPLATE_DIR.

    Returns:
        The rendered template as a string.

    Raises:
        FileNotFoundError: If the template file is not found.
        ValueError: If there are missing keys in the data dictionary that are
                    referenced in the template, or if other Jinja2 errors occur.
    """
    # 1. Set up Jinja2 environment and loader
    # The FileSystemLoader loads templates from the file system.
    if os.getcwd().endswith("/outputs/debug/debug") or "notebooks" in os.getcwd():
        template_dir = "../../../" + template_dir
        
    env = Environment(loader=FileSystemLoader(template_dir),
                       trim_blocks=True,
                       lstrip_blocks=True
                      )
    if env is None or env.loader is None:
        raise ValueError(f"Failed to create Jinja2 environment with template directory: {template_dir}")

    # 2. Load the template source to parse for variables
    try:
        # get_source returns (source, filename, uptodate_func)
        template_source = env.loader.get_source(env, template_name)[0]
    except exceptions.TemplateNotFound:
        raise FileNotFoundError(
            f"Template '{template_name}' not found in directory '{template_dir}'. "
            f"Ensure the directory and file exist."
        )
    except Exception as e:
        # Catch other potential Jinja2 errors during source loading
        raise ValueError(f"Error loading template source for '{template_name}': {e}")

    # 3. Check if all keys referenced in the template are present in the data
    try:
        # Parse the template source into an Abstract Syntax Tree (AST)
        parsed_content = env.parse(template_source)
    except exceptions.TemplateSyntaxError as e:
        raise ValueError(f"Syntax error in template '{template_name}': {e}")

    # Find all undeclared variables in the parsed template
    # These are the variables expected to be in the 'data' dictionary
    template_variables = meta.find_undeclared_variables(parsed_content)

    # Identify any keys that are in template_variables but not in data.keys()
    missing_keys = template_variables - set(data.keys())
    if missing_keys:
        raise ValueError(
            f"Missing keys in input data for template '{template_name}': "
            f"{', '.join(sorted(list(missing_keys)))}"
        )

    # 4. Load the template object for rendering
    try:
        template = env.get_template(template_name)
    except exceptions.TemplateNotFound:
        # This case should ideally be caught by get_source earlier,
        # but it's good practice for robustness.
        raise FileNotFoundError(
             f"Template '{template_name}' not found in directory '{template_dir}' "
             f"during final loading stage."
        )

    # 5. Fill in (render) the template with the provided data
    rendered_string = template.render(data)

    return rendered_string


def randomize_mc_options(answer: str, options: list[str]) -> tuple[list[str], int]:
    """
    Randomizes the order of options and returns the randomized list along with the index of the correct answer.

    Args:
        answer: The correct answer to be included in the options.
        options: A list of options to be randomized.

    Returns:
        A tuple containing the randomized list of options and the index of the correct answer.
    """
    if not isinstance(options, list) or not options:
        raise ValueError("Options must be a non-empty list.")

    # Ensure the answer is in the options
    if answer is None:
        raise ValueError("Answer must not be None.")
    
    answers = list(set(options + [answer]))
    
    # Randomize the order of options
    random.shuffle(answers)
    # Get the index of the correct answer
    correct_index = answers.index(answer)

    return answers, correct_index
