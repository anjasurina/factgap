import unittest
from src.origins.utils.prompt_helper import parse_yaml_response


class TestYamlParser(unittest.TestCase):

    def setUp(self):
        self.expected_keys = ['title', 'snippet', 'date']

        # Basic Valid Case
        self.good_yaml = """```yaml
title: Success
date: 2024-01-01
snippet: |
  Valid content.
```"""

        # Basic Broken Case
        self.broken_yaml = """```yaml
title: Failed Indentation
date: 2024-01-01
snippet: |
Broken indentation line 1.
Broken indentation line 2.
```"""

    def test_parse_valid_yaml(self):
        """Standard valid YAML."""
        result = parse_yaml_response(
            self.good_yaml, expected_keys=self.expected_keys)
        self.assertEqual(result['title'], "Success")

    def test_fallback_on_broken_yaml(self):
        """Broken indentation triggers fallback."""
        result = parse_yaml_response(
            self.broken_yaml, expected_keys=self.expected_keys)
        self.assertIn("Broken indentation line 1", result['snippet'])

    # --- NEW TESTS BELOW ---

    def test_colons_inside_content(self):
        """
        Scenario: The text contains a colon (e.g., 'Note: ...').
        Risk: Regex might think 'Note:' is a new key and cut the content short.
        """
        colon_yaml = """```yaml
title: Colon Test
date: 2024-01-01
snippet: |
This is a sentence: with a colon.
It should not break the parser.
```"""
        # We simulate broken indentation to force regex usage
        broken_colon_yaml = colon_yaml.replace("  ", "")

        result = parse_yaml_response(
            broken_colon_yaml, expected_keys=self.expected_keys)

        self.assertIn("sentence: with a colon", result['snippet'])
        self.assertEqual(result['title'], "Colon Test")

    def test_overlapping_keys(self):
        """
        Scenario: One key is a substring of another (e.g., 'desc' and 'meta_desc').
        Risk: Searching for 'desc' might accidentally match inside 'meta_desc'.
        """
        keys = ['desc', 'meta_desc']
        overlap_yaml = """
meta_desc: This is the long meta description.
desc: This is the short description.
"""
        result = parse_yaml_response(overlap_yaml, expected_keys=keys)

        self.assertEqual(result['meta_desc'],
                         "This is the long meta description.")
        self.assertEqual(result['desc'], "This is the short description.")

    def test_python_triple_quotes(self):
        """
        Scenario: Model outputs Python-style triple quotes (\"\"\") instead of YAML block scalar (|).
        Risk: Standard parser crashes; Regex must strip the quotes.
        """
        triple_quote_yaml = """```yaml
title: Triple Quote Test
date: 2024-01-01
snippet: \"\"\"
This is surrounded by triple quotes.
Common LLM mistake.
\"\"\"
```"""
        result = parse_yaml_response(
            triple_quote_yaml, expected_keys=self.expected_keys)

        self.assertIn("surrounded by triple quotes", result['snippet'])
        # Ensure the actual quotes were stripped from the final value
        self.assertFalse(result['snippet'].strip().startswith('"""'))
        self.assertFalse(result['snippet'].strip().endswith('"""'))

    def test_empty_value_triggers_fallback(self):
        """
        Scenario: Valid YAML syntax, but a required key is empty.
        Risk: Standard parser accepts it. Logic should reject it and try fallback 
              (in case fallback can find the content hidden elsewhere).
        """
        empty_val_yaml = """```yaml
title: Empty Snippet
date: 2024-01-01
snippet: 
```"""
        # If fallback also fails (which it will here), it returns what it can.
        # But we want to ensure standard parser didn't just return {'snippet': None} and exit early.
        # We can check this by setting verbose=True manually or observing behavior.

        # To strictly test the trigger, we can feed it input where standard sees empty,
        # but regex sees content (e.g. mixed formatting).

        mixed_yaml = """```yaml
title: Mixed
date: 2024-01-01
snippet: 
This content looks like YAML key but is unindented text.
```"""
        # Standard YAML sees snippet=None. Regex should see the text below it.
        result = parse_yaml_response(
            mixed_yaml, expected_keys=self.expected_keys)

        self.assertTrue(
            result['snippet'], "Fallback should have captured the unindented text")
        self.assertIn("unindented text", result['snippet'])

    def test_no_markdown_tags(self):
        """
        Scenario: Model forgets the ```yaml tags entirely.
        Risk: Regex might fail to extract content if it relies strictly on markdown groups.
        """
        plain_text = """
title: Plain Text
date: 2024-01-01
snippet: |
No markdown code blocks here.
Just raw text.
"""
        result = parse_yaml_response(
            plain_text, expected_keys=self.expected_keys)
        self.assertEqual(result['title'], "Plain Text")
        self.assertIn("No markdown", result['snippet'])


if __name__ == '__main__':
    unittest.main()
