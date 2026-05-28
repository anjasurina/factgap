from dataclasses import dataclass
from typing import Callable, Any, Optional

import fire

from .inference import LM, LMFactory
from .adapters import OutputAPI, _SupportedProviders
from ..utils.general_utilities import print_c, ColorType
from ..utils.reusable_classes import SystemMessage, AssistantMessage, UserMessage


@dataclass
class TestCase:
    """Single test case definition"""
    name: str
    description: str
    messages: Any  # Can be single message or list of messages
    validate: Callable[[Any], tuple[bool, str]]  # Returns (passed, message)


def validate_non_empty(response) -> tuple[bool, str]:
    """Validate response is not empty"""
    if isinstance(response, Exception):
        return False, str(response)
    if not response or not response[0] or response[0] is None:
        return False, "No response"
    if isinstance(response[0], OutputAPI) and not response[0].completion:
        return False, "Response is empty."
    return True, str(response[0])


def validate_contains(expected: str):
    """Validate response contains expected text"""
    def validator(response) -> tuple[bool, str]:
        if isinstance(response, Exception):
            return False, str(response)
        if not response or not response[0] or response[0] is None:
            return False, "No response from assistant followed by user message."
        if isinstance(response[0], OutputAPI):
            if expected in response[0].completion:
                return True, str(response[0])
            return False, f"Response is not of expected content: {response[0]}"
        return False, f"Response is not of expected type: {type(response[0])}"
    return validator


def validate_two_consecutive_users(response) -> tuple[bool, str]:
    """Validate response to two consecutive user messages"""
    if isinstance(response, Exception):
        return False, str(response)
    if not response or not response[0] or response[0] is None:
        return False, "No response from user message followed by another user message."
    if isinstance(response[0], OutputAPI) and not response[0].completion:
        return False, "Response is empty."
    if isinstance(response[0], OutputAPI):
        content = response[0].completion
        has_11 = '11' in content
        has_22 = '22' in content
        if has_11 and has_22:
            return True, str(response[0])
        elif has_22 and not has_11:
            return False, f"Response only answered last message: {response[0]}"
        elif has_11 and not has_22:
            return False, f"Response only answered first message {response[0]}"
        else:
            return False, f"Response is not of expected content: {response[0]}"
    return True, f"Unexpected response {response[0]}"


def validate_multiple_system_prompts(response) -> tuple[bool, str]:
    """
    Validate multiple system prompt behavior.
    Expected behaviors:
    1. Provider rejects with error (valid)
    2. Response follows both prompts (starts with 'Thus' and ends with 'That is all')
    3. Response ignores one or both prompts (partial compliance)
    """
    # Check if provider rejects multiple system prompts (valid behavior)
    if isinstance(response, Exception):
        error_msg = str(response).lower()
        if any(phrase in error_msg for phrase in ['multiple system', 'system message', 'only one system']):
            return True, f"(provider correctly rejects multiple system prompts): {response}"
        return False, f"with unexpected error: {response}"

    # Check for empty response
    if not response or not response[0] or response[0] is None:
        return False, "No response from multiple system prompts."

    if isinstance(response[0], OutputAPI) and not response[0].completion:
        return False, "Response is empty."

    # Check if response follows both system prompts
    if isinstance(response[0], OutputAPI):
        content = response[0].completion.strip()

        # Check if starts with "Thus" (case-insensitive)
        starts_with_thus = content.lower().startswith('thus')

        # Check if ends with "That is all" (case-insensitive, handle punctuation)
        ends_with_that_is_all = any([
            content.lower().endswith('that is all'),
            content.lower().endswith('that is all.'),
            content.lower().endswith('that is all!'),
        ])

        # Determine compliance level
        if starts_with_thus and ends_with_that_is_all:
            return True, f"(FULL COMPLIANCE - follows both system prompts): {response[0]}"
        elif starts_with_thus and not ends_with_that_is_all:
            return True, f"(PARTIAL COMPLIANCE - follows first system prompt only, ignores second): {response[0]}"
        elif not starts_with_thus and ends_with_that_is_all:
            return True, f"(PARTIAL COMPLIANCE - follows second system prompt only, ignores first): {response[0]}"
        else:
            return True, f"(NO COMPLIANCE - ignores both system prompts but still responds): {response[0]}"

    return False, f"Unexpected response type: {type(response[0])}"


def get_test_cases() -> list[TestCase]:
    """Define all test cases"""
    system_msg = SystemMessage(
        "You answer and propose simple math problems. Always output one!")
    assistant_msg = AssistantMessage(
        "1 + 3 = 4. Now your turn: what is 2 + 7?")
    user_msg = UserMessage(
        "2 + 7 = 9. Now your turn: what is 5 + 6? Respond with the answer only.")

    # Updated system messages for test 7
    system_msg_1 = SystemMessage(
        "You must always start your response with the word 'Thus'.")
    system_msg_2 = SystemMessage(
        "You must always end your response with the exact phrase 'That is all'.")

    return [
        TestCase(
            name="test_1",
            description="System message only",
            messages=system_msg,
            validate=validate_non_empty
        ),
        TestCase(
            name="test_2",
            description="Assistant message only",
            messages=assistant_msg,
            validate=validate_non_empty
        ),
        TestCase(
            name="test_3",
            description="Assistant message followed by user message",
            messages=[assistant_msg, user_msg],
            validate=validate_contains('11')
        ),
        TestCase(
            name="test_4",
            description="System message followed by assistant message",
            messages=[system_msg, assistant_msg],
            validate=validate_non_empty
        ),
        TestCase(
            name="test_5",
            description="System message followed by assistant and user message",
            messages=[system_msg, assistant_msg, user_msg],
            validate=validate_contains('11')
        ),
        TestCase(
            name="test_6",
            description="User message followed by another user message",
            messages=[user_msg, UserMessage(
                "I forget to ask: also solve 19 + 3 please")],
            validate=validate_two_consecutive_users
        ),
        TestCase(
            name="test_7",
            description="Multiple system prompts (system -> user -> system)",
            messages=[system_msg_1, user_msg, system_msg_2],
            validate=validate_multiple_system_prompts
        ),
    ]


def test_response_structure(lm: LM) -> None:
    """Test response structure for a given LM"""
    print_c(f"Testing response [{lm.adapter.provider}/{lm.adapter.model_name}] structure...",
            color=ColorType.BLUE)
    print_c("========================================", color=ColorType.BLUE)

    test_outcomes = {}

    for test_case in get_test_cases():
        # Print test start
        print_c(f"{test_case.name} STARTED", color=ColorType.BLUE)

        # Execute test
        try:
            response = lm(test_case.messages)
        except Exception as e:
            response = e

        # Validate response
        passed, message = test_case.validate(response)

        # Store outcome
        test_outcomes[test_case.name] = {
            "status": passed,
            "description": test_case.description,
            "message": message
        }

        # Print result
        if passed:
            print_c(f"{test_case.name} PASSED:\n{test_case.description}\n-->Test {test_case.name[5:]} passed: {message}",
                    color=ColorType.GREEN)
        else:
            print_c(f"{test_case.name} FAILED:\n{test_case.description}\n-->Test {test_case.name[5:]} failed: {message}",
                    color=ColorType.RED)


def test_all_api_response_structures() -> None:
    """Test all configured provider/model pairs"""
    model_pairs = {
        "azure": ["Meta-Llama-3.1-8B-Instruct", "gpt-4.1", "phi-4"],
        "anthropic": ["claude-sonnet-3.7"],
        "google": ["gemini-2.5-flash"],
    }

    for provider, models in model_pairs.items():
        for model in models:
            print_c(f"Testing response structure for {provider}/{model}...",
                    color=ColorType.BLUE)
            lm = LMFactory(
                provider=provider,  # type: ignore
                model_name=model,
            ).create_instance()
            test_response_structure(lm)
            print_c("========================================",
                    color=ColorType.BLUE)


def main(
    lm_name: Optional[str] = None,
    lm_provider: Optional[_SupportedProviders] = None,
    test_all: bool = False
) -> None:
    """
    Main function to test the response structure of a specified language model.

    Args:
        lm_name (str): The name of the language model to test.
        lm_provider (_SupportedProviders): The provider of the language model.
        test_all (bool): Test all configured models.
    """
    if test_all:
        test_all_api_response_structures()
    elif lm_name:
        lm = LMFactory(
            provider=lm_provider,  # type: ignore
            model_name=lm_name,
        ).create_instance()
        test_response_structure(lm)
    else:
        print_c("Provide either --test_all or --lm_name",
                color=ColorType.RED)


if __name__ == "__main__":    
    fire.Fire(main)
