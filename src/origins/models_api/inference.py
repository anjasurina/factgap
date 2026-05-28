import copy
from uuid import uuid4
from dataclasses import dataclass, field
from typing import Optional, overload, Literal, Sequence
import aiohttp
import asyncio
import nest_asyncio
import logging

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)

from .adapters import (
    OpenAIAdapter, AzureAdapter, AnthropicAdapter, GoogleAdapter,
    CohereAdapter, TogetherAIAdapter, XAIAdapter, AzureFoundryAdapter,
    AzureCognitiveAdapter,
    lookup_model_details, ProviderAPI, _SupportedProviders,
    OutputAPI, InputAPI, LMAdapter, LMConfig,
    infer_model_provider, _ReasoningEffort
)

from ..utils.reusable_classes import (
    Message, UserMessage, AssistantMessage, SystemMessage,
    Conversation)

_MAX_CONCURRENT = 10
_MAX_RETRIES = 6


# Set up logging
# Configure logging to show INFO and above
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s:\n%(message)s'
)
logger = logging.getLogger(__name__)


_InputTypes = str | list[str] | Message | list[Message] | Conversation | list[Conversation]
_OutputTypes = OutputAPI | AssistantMessage | None
_SingleSample = Literal[1]
_MultipleSamples = int  # Represents int > 1


class LM:
    """
    A class to access a Language Model (LM) through an API.

    Attributes:
        adapter (LMAdapter): The adapter to interact with the LM API.
        config (LMConfig): Configuration for the LM, such as temperature, max tokens, etc.
        max_concurrent (int): Maximum number of concurrent requests to the LM API.
        max_retries (int): Maximum number of retries for failed requests.
    Methods:
        get_response: Asynchronously calls the LM API to get a response based on input data.
        _prepare_for_multiple_samples: Prepares conversations for multiple samples by duplicating each conversation.
        _acall: Asynchronously gets responses from the LM based on provided messages or conversations.
        __call__: Synchronous wrapper for the async call, handling event loop logic for both script and notebook environments.
    Usage:
        lm = LM(adapter=OpenAIAdapter(model_name="gpt-3.5-turbo"),
                config=LMConfig(temperature=0.7))
        messages = [Message(role=RoleType.USER, content="Hello, how are you?")]
        responses = lm(input_data=messages, return_as_message=True, num_samples=1)
        for response in responses:
            print(f"Response: {response.content}")
    """

    def __init__(
            self,
            adapter: LMAdapter,
            config: Optional[LMConfig] = None,
            max_concurrent: int = _MAX_CONCURRENT,
            max_retries: int = _MAX_RETRIES):

        self.adapter = adapter
        self.config = config or LMConfig()
        self.max_concurrent = max_concurrent
        self.max_retries = max_retries
        self.unique_id = str(uuid4())
        self.debug_mode = False
        self.remove_thinking_tokens = True

    def __repr__(self):
        return f"LM(adapter={repr(self.adapter)},max_concurrent={self.max_concurrent}, max_retries={self.max_retries})"

    def _remove_thinking_tokens(self, content):
        if (
            self.remove_thinking_tokens and
            isinstance(content, str) and
            "<think>" in content and
            "</think>" in content and
            "deepseek" in self.adapter.model_name.lower()
        ):
            # find index of "</think>" and only keep string after that
            end_index = content.index("</think>")
            content = content[end_index + len("</think>"):].strip()

        return content

    @overload
    async def get_response(
        self,
        *,
        input_api: InputAPI,
        identifier: str | int,
        session: aiohttp.ClientSession,
        semaphore: asyncio.Semaphore,
        return_as_message: Literal[True]
    ) -> AssistantMessage: ...

    @overload
    async def get_response(
        self,
        *,
        input_api: InputAPI,
        identifier: str | int,
        session: aiohttp.ClientSession,
        semaphore: asyncio.Semaphore,
        return_as_message: Literal[False]
    ) -> OutputAPI: ...

    @retry(
        stop=stop_after_attempt(_MAX_RETRIES),
        wait=wait_exponential(multiplier=3, min=1, max=30),
        retry=retry_if_exception_type((
            aiohttp.ClientError,
            aiohttp.ClientResponseError,
            asyncio.TimeoutError
        )),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    async def get_response(
            self,
            *,
            input_api: InputAPI,
            identifier: str | int,
            session: aiohttp.ClientSession,
            semaphore: asyncio.Semaphore,
            return_as_message: bool = False
    ) -> OutputAPI | AssistantMessage:
        """
        Asynchronously call an API to get a response from the model.
        Retries are handled by the @retry decorator.
        """
        async with semaphore:
            async with session.post(input_api.url,
                                    headers=input_api.headers,
                                    json=input_api.payload) as resp:

                if resp.status >= 400:
                    error_details = await resp.text()
                    logger.error(
                        f'HTTP Error {resp.status} for {resp.url}'
                        f'API Response: {error_details}'
                    )

                resp.raise_for_status()
                data = await resp.json()
                data_processed = self.adapter.process_response(data)
                data_processed.identifier = identifier
                data_processed.model_name = self.adapter.model_name
                data_processed.model_provider = self.adapter.provider.value
                data_processed.agent_id = self.unique_id
                data_processed.completion = self._remove_thinking_tokens(
                    data_processed.completion)

                if return_as_message:
                    return data_processed.to_assistant_message()

                return data_processed

    def _prepare_for_multiple_samples(
            self,
            conversations: list[Conversation],
            num_samples: int) -> list[Conversation]:
        """
        Prepare conversations for multiple samples by duplicating each conversation.

        Args:
            conversations (list[Conversation]): List of conversations to duplicate.
            num_samples (int): Number of times to duplicate each conversation.
        Returns:
            list[Conversation]: A new list of conversations, each duplicated num_samples times.
        """
        if num_samples <= 1:
            return conversations

        duplicated_conversations = []
        for conversation in conversations:
            if conversation.identifier is None:
                conversation.identifier = str(uuid4())
            duplicated_conversations.extend(
                [copy.deepcopy(conversation) for _ in range(num_samples)]
            )
        return duplicated_conversations

    def _process_multiple_samples(
            self,
            responses: Sequence[OutputAPI | AssistantMessage | BaseException],
            num_samples: int
    ) -> Sequence[_OutputTypes] | Sequence[Sequence[_OutputTypes]]:
        """
        Process responses for multiple samples.

        Args:
            responses (Sequence[OutputAPI | AssistantMessage]): Responses from the model.
            num_samples (int): Number of samples per input data.
        Returns:
            Sequence[OutputAPI | AssistantMessage] | Sequence[Sequence[OutputAPI | AssistantMessage]]:
                If num_samples == 1, returns a sequence of responses.
                If num_samples > 1, returns a sequence of sequences, where each inner sequence contains
                responses for each sample of the input data.
        """
        processed_responses = []
        if num_samples == 1:
            # check for errors in responses
            for i, response in enumerate(responses):
                if (isinstance(response, OutputAPI) and response.error):
                    print(f"Error in task {i}, response: {response.error}")
                    processed_responses.append(None)
                elif isinstance(response, Exception):
                    print(f"Exception in task {i}, response: {response}")
                    processed_responses.append(None)
                else:
                    processed_responses.append(response)
        else:
            n_batch = []
            for i, response in enumerate(responses):
                if (isinstance(response, OutputAPI) and response.error):
                    print(f"Error in task {i}, response: {response.error}")
                    n_batch.append(None)
                elif isinstance(response, Exception):
                    print(f"Exception in task {i}, response: {response}")
                    n_batch.append(None)
                else:
                    n_batch.append(response)

                if (i + 1) % num_samples == 0:
                    processed_responses.append(n_batch)
                    n_batch = []

        return processed_responses

    def _normalize_input_data(
        self,
        input_data: _InputTypes
    ) -> list[Conversation]:
        """Normalize input data to a list of Conversation objects."""
        conversations: list[Conversation]
        if input_data is None:
            raise ValueError("Input data must be provided.")

        if isinstance(input_data, str):
            # Single string input, treat as a single message
            conversations = [Conversation(messages=[
                UserMessage(content=input_data)])]
        elif isinstance(input_data, Message):
            # Single message input
            conversations = [Conversation(messages=[input_data])]
        elif isinstance(input_data, list):
            if all(isinstance(item, Message) for item in input_data):
                # List of messages
                conversations = [Conversation(
                    messages=input_data)]  # type: ignore
            elif all(isinstance(item, Conversation) for item in input_data):
                # List of conversations
                conversations = input_data  # type: ignore
            elif all(isinstance(item, str) for item in input_data):
                # List of strings, treat each as a single message in its own conversation
                conversations = [
                    Conversation(messages=[UserMessage(content=msg)])
                    for msg in input_data
                ]
            else:
                raise ValueError(
                    "Input data must be a String, list of Strings, Message, list of Messages, Conversation, or list of Conversations.")
        elif isinstance(input_data, Conversation):
            # Single conversation input
            conversations = [input_data]
        else:
            raise ValueError(
                "Input data must be a Message, list of Messages, Conversation, or list of Conversations.")

        if not conversations:
            raise ValueError("No valid input data provided.")

        return conversations

    @overload
    async def _acall(
        self,
        input_data: _InputTypes,
        *,
        num_samples: _SingleSample,
        return_as_message: Literal[True]
    ) -> Sequence[AssistantMessage | None]: ...

    @overload
    async def _acall(
        self,
        input_data: _InputTypes,
        *,
        num_samples: _SingleSample,
        return_as_message: Literal[False],
    ) -> Sequence[OutputAPI | None]: ...

    @overload
    async def _acall(
        self,
        input_data: _InputTypes,
        *,
        num_samples: _MultipleSamples,
        return_as_message: Literal[True]
    ) -> Sequence[Sequence[AssistantMessage | None]]: ...

    @overload
    async def _acall(
        self,
        input_data: _InputTypes,
        *,
        num_samples: _MultipleSamples,
        return_as_message: Literal[False],
    ) -> Sequence[Sequence[OutputAPI | None]]: ...

    async def _acall(
        self,
        input_data: _InputTypes,
        *,
        return_as_message: bool = False,
        num_samples: int = 1
    ) -> (
            Sequence[_OutputTypes] |
            Sequence[Sequence[_OutputTypes]]):
        """Get responses from the model based on provided messages or conversations.

        Args:
            messages (Optional[Message | list[Message]]): A single message or a list of messages to send to the model.
            conversations (Optional[list[Conversation] | Conversation]): A single conversation or a list of conversations to send to the model.
            return_as_message (bool): If True, returns responses as AssistantMessage objects; otherwise, returns OutputAPI objects.
        Returns:
            if num_samples == 1:
                Sequence[OutputAPI | AssistantMessage]: A sequence of responses from the model.
            else:
                Sequence[Sequence[OutputAPI | AssistantMessage]]: A sequence of sequences, where each
                inner sequence contains responses for each sample of the input data.
        Raises:
            ValueError: If input data is not provided or is of an unsupported type.
        Raises:
            RuntimeError: If there is an issue with the event loop (e.g., no running loop).
        """

        # Normalize input data to a list of Conversation objects
        conversations = self._normalize_input_data(input_data)

        # Prepare conversations for multiple samples if needed.
        conversations = self._prepare_for_multiple_samples(
            conversations=conversations,
            num_samples=num_samples
        )

        # If debug mode is enabled, return fake responses for testing
        if self.debug_mode:
            return [
                [AssistantMessage(
                    content=f"Debug response {i + 1} for conversation {conv.identifier}",
                    agent_id=self.unique_id,
                ) for i in range(num_samples)]
                for conv in conversations
            ] if num_samples > 1 else [
                AssistantMessage(
                    content=f"Debug response for conversation {conv.identifier}",
                    agent_id=self.unique_id,
                ) for conv in conversations
            ]

        semaphore = asyncio.Semaphore(_MAX_CONCURRENT)
        async with aiohttp.ClientSession() as session:
            tasks = [
                self.get_response(
                    input_api=self.adapter.prepare_call(
                        messages=self.adapter.format_prompt_iterable(
                            conversation.messages),
                        config=self.config
                    ),
                    identifier=conversation.identifier or str(uuid4()),
                    session=session,
                    semaphore=semaphore,
                    return_as_message=return_as_message
                )
                for conversation in conversations
            ]
            responses = await asyncio.gather(*tasks, return_exceptions=True)

        responses = self._process_multiple_samples(
            responses=responses,
            num_samples=num_samples
        )

        return responses

    @overload
    def __call__(
        self,
        input_data: _InputTypes,
        *,
        num_samples: _SingleSample,
        return_as_message: Literal[True],
    ) -> Sequence[AssistantMessage | None]: ...

    @overload
    def __call__(
        self,
        input_data: _InputTypes,
        *,
        num_samples: _SingleSample,
        return_as_message: Literal[False],
    ) -> Sequence[OutputAPI | None]: ...

    @overload
    def __call__(
        self,
        input_data: _InputTypes,
        *,
        num_samples: _MultipleSamples,
        return_as_message: Literal[True],
    ) -> Sequence[Sequence[AssistantMessage | None]]: ...

    @overload
    def __call__(
        self,
        input_data: _InputTypes,
        *,
        num_samples: _MultipleSamples,
        return_as_message: Literal[False],
    ) -> Sequence[Sequence[OutputAPI | None]]: ...

    @overload
    def __call__(
        self,
        input_data: _InputTypes
    ) -> Sequence[_OutputTypes]: ...

    def __call__(
        self,
        input_data: _InputTypes,
        *,
        num_samples: int = 1,
        return_as_message: bool = False,
    ) -> Sequence[_OutputTypes] | Sequence[Sequence[_OutputTypes]]:
        """
        Synchronous wrapper for async call.
        Handles event loop logic for both script and notebook environments.
        """
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                # In a notebook or running event loop
                nest_asyncio.apply()
                return loop.run_until_complete(
                    self._acall(
                        input_data=input_data,
                        num_samples=num_samples,
                        return_as_message=return_as_message
                    )
                )
        except RuntimeError:
            # No event loop running
            pass

        # Standard script usage
        return asyncio.run(
            self._acall(
                input_data=input_data,
                num_samples=num_samples,
                return_as_message=return_as_message
            )
        )


@dataclass
class LMFactory:
    """
    Factory class to create instances of LM based on the provider and model.
    This class encapsulates the logic for creating LM instances with different adapters.

    Attributes:
        provider (str): The provider to use ('openai' or 'azure').
        model_name (str): The name of the model to use.
        model_builder (Optional[str]): The model builder to use (e.g., 'meta').
        endpoint (Optional[str]): The endpoint for Azure API (required for Azure).
        api_version (str): The API version for Azure.
        config (Optional[LMConfig]): Configuration for the LM.
        max_concurrent (int): Maximum number of concurrent requests.
        max_retries (int): Maximum number of retries for failed requests.

    Usage:
        factory = LMFactory(
            provider='azure', model_name='Meta-Llama-3.1-8B-Instruct',
            model_builder='meta', endpoint='https://trapi.research.microsoft.com/gcr/shared',
            api_version='2024-05-01-preview',
            config=LMConfig(temperature=0.2),
            max_concurrent=10, max_retries=5)
        lm_instance = factory.create_lm_instance()
    """
    model_name: str
    provider: Optional[_SupportedProviders] = None
    model_builder: Optional[str] = None
    endpoint: Optional[str] = None
    api_version: str = "2024-05-01-preview"
    secrets_filepath: str | None = None
    config: LMConfig = field(default_factory=LMConfig)
    max_concurrent: int = _MAX_CONCURRENT
    max_retries: int = _MAX_RETRIES
    skip_model_lookup: bool = False

    def __post_init__(self):
        if self.provider is None:
            self.provider = infer_model_provider(self.model_name)

        OAI_o_models = [
            'o1', 'o1-mini', 'o3', 'o3-mini', 'o4', 'o4-mini',
            'gpt-5.4', 'gpt-5.3', 'gpt-5.2', 'gpt-5.1',
            'gpt-5', 'gpt-5-mini', 'gpt-5-nano']
        if self.model_name in OAI_o_models:
            self.api_version = "2024-12-01-preview"
            self.config.temperature = 1

    def __call__(self):
        """
        Create an LM instance using the factory method.
        This method is a convenience wrapper for create_instance().
        """
        return self.create_instance()

    def create_instance(self) -> LM:
        """
        Factory function to create an LM instance based on the provider.

        Returns:
            LM: An instance of the LM class configured with the specified provider and model.
        Raises:
            NotImplementedError: If the provider is not implemented.
        """

        if self.provider in [
            "openai", "azure", "anthropic", "google", "cohere", "xai",
            "togetherai", "azure_foundry", "azure_cognitive"
        ] and not self.skip_model_lookup:
            model_details = lookup_model_details(
                model_name=self.model_name,
                provider=ProviderAPI[self.provider.upper()],
                deployment_name=self.model_builder)
            self.model_name = model_details.model_deployment_name
            self.model_builder = model_details.model_builder

            # only use reasoning meta parameters if supported by API
            if not model_details.reasoning_support:
                self.config.reasoning_effort = None
                self.config.reasoning_summary = None

        if self.skip_model_lookup and self.model_builder is None:
            # NOTE: this is a dummy workflow where the model is not actually used
            self.model_builder = self.provider

        if self.provider == "openai":
            adapter = OpenAIAdapter(
                model_name=self.model_name,
                secrets_filepath=self.secrets_filepath
            )
        elif self.provider == "azure":
            assert self.model_builder is not None, \
                "model_builder must be specified for Azure provider."
            if self.endpoint is None:
                if self.model_builder == 'openai':
                    self.endpoint = 'https://trapi.research.microsoft.com/msraif/shared'
                else:
                    self.endpoint = 'https://trapi.research.microsoft.com/gcr/shared'
                logger.debug(
                    f"Warning: using default Azure endpoint: {self.endpoint}")
            adapter = AzureAdapter(
                model_name=self.model_name,
                model_builder=self.model_builder,
                endpoint=self.endpoint,
                api_version=self.api_version
            )
        elif self.provider == "anthropic":
            adapter = AnthropicAdapter(
                model_name=self.model_name,
                secrets_filepath=self.secrets_filepath
            )
        elif self.provider == "google":
            adapter = GoogleAdapter(
                model_name=self.model_name,
                secrets_filepath=self.secrets_filepath
            )
        elif self.provider == 'cohere':
            adapter = CohereAdapter(
                model_name=self.model_name,
                secrets_filepath=self.secrets_filepath
            )
        elif self.provider == 'togetherai':
            adapter = TogetherAIAdapter(
                model_name=self.model_name,
                secrets_filepath=self.secrets_filepath
            )
        elif self.provider == 'xai':
            adapter = XAIAdapter(
                model_name=self.model_name,
                secrets_filepath=self.secrets_filepath
            )
        elif self.provider == 'azure_foundry':
            adapter = AzureFoundryAdapter(
                model_name=self.model_name,
                secrets_filepath=self.secrets_filepath
            )
        elif self.provider == 'azure_cognitive':
            adapter = AzureCognitiveAdapter(
                model_name=self.model_name,
                secrets_filepath=self.secrets_filepath
            )
        else:
            raise NotImplementedError(
                f"Provider {self.provider} is not implemented.")

        return LM(
            adapter=adapter,
            config=self.config,
            max_concurrent=self.max_concurrent,
            max_retries=self.max_retries
        )


def example(
        provider: Optional[_SupportedProviders] = None,
        model_name: str = 'gemini-3-flash',
        prompt: str = "Give me three of your favorite movie titles and nothing else.",
        num_samples: int = 1,
        temperature: float = 0.2,
        reasoning_effort: _ReasoningEffort | None = None,
        system_prompt: Optional[str] = None,
        system_prompt_2: Optional[str] = None,
        api_version="2024-05-01-preview",
) -> None:
    """
    Example function to demonstrate how to use the LM class with different providers.

    Args:
        provider (str): The provider to use ('openai' or 'azure').
        model_name (str): The name of the model to use.
        prompt (str): The prompt to send to the model.
        num_samples (int): The number of samples to generate.
        temperature (float): The temperature for the model's responses.
        system_prompt (Optional[str]): The system prompt to use.
    Raises:
        NotImplementedError: If the provider is not implemented.
    Usage:
        To run this example, you can use the command line:
        python -m src.utils.models --num_samples=3 --temperature=0.8
    """

    lm = LMFactory(
        provider=provider,
        model_name=model_name,
        api_version=api_version,
        config=LMConfig(temperature=temperature,
                        reasoning_effort=reasoning_effort),
    ).create_instance()

    messages: list[Message] = []
    if system_prompt is not None:
        system_message = SystemMessage(content=system_prompt)
        messages.append(system_message)
    message = UserMessage(prompt)
    messages.append(message)

    # This is a quick way to test if models allow multiple system prompts
    if system_prompt_2 is not None:
        system_message_2 = SystemMessage(content=system_prompt_2)
        messages.append(system_message_2)

    responses = lm(
        messages,
        return_as_message=False,
        num_samples=num_samples
    )
    for response in responses:
        if isinstance(response, OutputAPI):
            print(f"\nResponse:\n{response.completion}")
            print(
                f"\nInput/Output Tokens: {response.input_tokens}/{response.output_tokens}")
            if response.error:
                print(f"Error: {response.error}")

            print(f"Identifier: {response.identifier}")


# Example usage:
if __name__ == "__main__":
    import fire
    fire.Fire(example)
