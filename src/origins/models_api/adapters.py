"""
This file contains utilities to adapt the inference calls to various providers 
"""
import enum
import yaml
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Any, Literal, TypeAlias
import functools

from ..utils.reusable_classes import (
    Message, AssistantMessage, UserMessage, RoleType, ContentType)

_MODEL_DETAILS_FPATH = "./data/models.yaml"


class ProviderAPI(enum.Enum):
    OPENAI = "openai"
    AZURE = "azure"
    ANTHROPIC = "anthropic"
    COHERE = "cohere"
    GOOGLE = "google"
    XAI = "xai"
    TOGETHERAI = "togetherai"
    AZURE_FOUNDRY = "azure_foundry"
    AZURE_COGNITIVE = "azure_cognitive"


_SupportedProviders: TypeAlias = Literal[
    "openai",
    "anthropic",
    "azure",
    "google",
    "cohere",
    "xai",
    "togetherai",
    "azure_foundry",
    "azure_cognitive"
]

_ReasoningEffort: TypeAlias = Literal['none',
                                      'minimal', 'low', 'medium', 'high']


def infer_model_provider(name: str) -> _SupportedProviders:
    """Incomplete utility to infer the model provider from the model name."""
    _name = name.lower()

    azure_models = [
        'llama-3', 'mistral', 'mixtral', 'phi-', 'deepseek',
    ]
    oai_models = [
        'gpt-5', 'gpt-4.1', 'gpt-4o', 'o3',
        'o4-mini', 'o4', 'o3-mini', 'o3-pro'
    ]
    togetherai_models = [
        'gpt-oss',
        'llama-4',
        'kimi',
        'qwen'
    ]

    if _name.startswith("af_"):
        return "azure_foundry"
    elif _name.startswith("ac_"):
        return "azure_cognitive"
    elif any(model in _name for model in oai_models):
        return "openai"
    elif any(model in _name for model in azure_models):
        return "azure"
    elif "gemini" in _name or "gemma" in _name:
        return "google"
    elif "claude" in _name:
        return "anthropic"
    elif 'command' in _name:
        return "cohere"
    elif any(model in _name for model in togetherai_models):
        return "togetherai"
    elif 'grok' in _name or 'xai' in _name:
        return "xai"
    else:
        raise ValueError(f"Unknown model provider for name: {name}")


def get_token(
    provider: ProviderAPI,
    fpath: str | None = None
) -> str | None:

    if fpath is None:
        fpath = "./secrets.yaml"

    if provider == ProviderAPI.AZURE:
        # Silence Azure Identity logging
        logging.getLogger('azure.identity').setLevel(logging.WARNING)

        scope = "api://trapi/.default"
        credential = get_bearer_token_provider(ChainedTokenCredential(
            AzureCliCredential(),
            ManagedIdentityCredential(),
        ), scope)
        token = credential()
    elif provider == ProviderAPI.OPENAI:
        with open(fpath, "r") as f:
            secrets = yaml.safe_load(f)
            token = secrets["openai"]
    elif provider == ProviderAPI.ANTHROPIC:
        with open(fpath, "r") as f:
            secrets = yaml.safe_load(f)
            token = secrets["anthropic"]
    elif provider == ProviderAPI.GOOGLE:
        with open(fpath, "r") as f:
            secrets = yaml.safe_load(f)
            token = secrets["google"]
    elif provider == ProviderAPI.COHERE:
        with open(fpath, "r") as f:
            secrets = yaml.safe_load(f)
            token = secrets["cohere"]
    elif provider == ProviderAPI.XAI:
        with open(fpath, "r") as f:
            secrets = yaml.safe_load(f)
            token = secrets["xai"]
    elif provider == ProviderAPI.TOGETHERAI:
        with open(fpath, "r") as f:
            secrets = yaml.safe_load(f)
            token = secrets["togetherai"]
    elif provider == ProviderAPI.AZURE_FOUNDRY:
        with open(fpath, "r") as f:
            secrets = yaml.safe_load(f)
            token = secrets["azure_foundry"]
    elif provider == ProviderAPI.AZURE_COGNITIVE:
        with open(fpath, "r") as f:
            secrets = yaml.safe_load(f)
            token = secrets["azure_cognitive"]
    else:
        raise NotImplementedError(
            f"Provider {provider} is not implemented for token retrieval.")

    return token


@dataclass
class InputAPI:
    url: str
    headers: dict[str, str]
    payload: dict[str, Any]


@dataclass
class OutputAPI:
    completion: str
    input_tokens: int
    output_tokens: int
    reasoning_completion: Optional[str] = None
    reasoning_tokens: Optional[int] = None
    error: Optional[str] = None
    identifier: Optional[str | int] = None
    model_name: Optional[str] = None
    model_provider: Optional[str] = None
    agent_id: Optional[str] = None

    def to_assistant_message(self) -> AssistantMessage:
        """
        Converts this OutputAPI instance to an AssistantMessage.
        """
        return AssistantMessage(
            content=self.completion,
            content_type=ContentType.TEXT,
            model_name=self.model_name,
            model_provider=self.model_provider,
            agent_id=self.agent_id,
        )


@dataclass
class _MessageAPI:
    role_field_name: str
    role_field_value: str
    content_field_name: str
    content_field_value: str
    content_field_type_name: Optional[str] = None
    content_field_type_value: Optional[str] = None


@dataclass
class LMConfig:
    temperature: float = 0.2
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    stop: Optional[list[str]] = None
    logit_bias: Optional[dict[str, int]] = None
    seed: Optional[int] = None
    system_instructions: Optional[str] = None
    log_probs: Optional[bool] = None
    reasoning_effort: Optional[_ReasoningEffort | None] = None
    reasoning_summary: Optional[str] = None


@dataclass
class _ModelDetails:
    model_name: str
    model_deployment_name: str
    model_builder: str
    provider: str
    reasoning_support: bool = False
    _model_version: Optional[str] = None
    _agg_request_rate: Optional[int] = None
    _request_rate_window: Optional[int] = None

    @classmethod
    def from_dict(cls, model_name, provider, data: dict[str, Any]) -> "_ModelDetails":
        """
        Create an instance of _ModelDetails from a dictionary.
        """
        return cls(
            model_name=model_name,
            provider=provider,
            model_deployment_name=data["DeploymentName"],
            model_builder=data["ModelBuilder"],
            reasoning_support=data.get("ReasoningSupport", False),
            _model_version=data.get("ModelVersion"),
            _agg_request_rate=data.get("AggregatedRequestRate"),
            _request_rate_window=data.get("RequestRateWindow")
        )

    @functools.cached_property
    def rate_num_requests_per_minute(self) -> Optional[int]:
        if self._agg_request_rate is not None and self._request_rate_window is not None:
            try:
                self._agg_request_rate = int(self._agg_request_rate)
                self._request_rate_window = int(self._request_rate_window)
                window_norm = self._request_rate_window / 60
                return int(self._agg_request_rate / window_norm)
            except:
                return None
        return None


def lookup_model_details(
        model_name,
        provider: ProviderAPI,
        deployment_name: Optional[str] = None) -> _ModelDetails:

    if provider not in [
        ProviderAPI.AZURE,
        ProviderAPI.ANTHROPIC,
        ProviderAPI.GOOGLE,
        ProviderAPI.COHERE,
        ProviderAPI.OPENAI,
        ProviderAPI.XAI,
        ProviderAPI.TOGETHERAI,
        ProviderAPI.AZURE_FOUNDRY,
        ProviderAPI.AZURE_COGNITIVE
    ]:
        raise ValueError(
            f"Provider {provider} is not supported for model details lookup.")

    with open(_MODEL_DETAILS_FPATH, "r") as f:
        model_details = yaml.safe_load(f)

    # look for the model in the model details
    if model_name not in model_details:
        raise ValueError(
            f"Model {model_name} not found in model details file: {_MODEL_DETAILS_FPATH}")

    details_list = model_details[model_name]
    # get model with most
    if deployment_name is not None:
        details_list = [
            details for details in details_list if details["DeploymentName"] == deployment_name]

    details_list_original = [d for d in details_list]
    details_list = [
        details for details in details_list if details["Provider"] == provider.value]

    if not details_list:
        if details_list_original:
            print(f"Found options from other providers: ", details_list_original)
        raise ValueError(
            f"Model {model_name} with deployment {deployment_name} not found in model details file: {_MODEL_DETAILS_FPATH}")

    # take the model with the highest request rate
    detail_objects = [_ModelDetails.from_dict(
        model_name=model_name,
        provider=provider.value,
        data=details) for details in details_list]
    detail_objects.sort(
        key=lambda x: x.rate_num_requests_per_minute or 0, reverse=True)

    return detail_objects[0]


class LMAdapter(ABC):

    def __init__(self, provider: ProviderAPI, model_name: str, **kwargs):
        self.provider = provider
        self.model_name = model_name
        self.token = get_token(self.provider, kwargs.get("secrets_filepath"))

    @abstractmethod
    def prepare_call(
        self,
        messages: list[_MessageAPI],
        config: Optional[LMConfig] = None) -> InputAPI: ...

    @abstractmethod
    def process_response(self, response: dict) -> OutputAPI: ...

    @abstractmethod
    def format_prompt(self, prompt: Message) -> _MessageAPI: ...

    def format_prompt_iterable(self, prompts: list[Message]) -> list[_MessageAPI]:
        """
        Formats a list of InternalPrompt objects into a list of Prompt objects.
        """
        return [self.format_prompt(prompt) for prompt in prompts]

    def __repr__(self):
        return f"{self.__class__.__name__}(provider={self.provider}, name={self.model_name})"


@dataclass
class AzureFoundryAdapter(LMAdapter):
    """
    An adapter for Azure Foundry's API to prepare payloads and process responses.
    """

    def __init__(self, model_name: str, **kwargs):
        super().__init__(provider=ProviderAPI.AZURE_FOUNDRY, model_name=model_name, **kwargs)

    def prepare_call(
        self,
        messages: list[_MessageAPI],
        config: Optional[LMConfig] = None
    ) -> InputAPI:
        url = "https://collaboration-gap-resource.services.ai.azure.com/models/chat/completions"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}"
        }
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    message.role_field_name: message.role_field_value,
                    message.content_field_name: message.content_field_value,
                } for message in messages
            ]
        }

        if config:
            if config.temperature is not None:
                payload["temperature"] = config.temperature
            if config.top_p is not None:
                payload["top_p"] = config.top_p
            if config.max_tokens is not None:
                payload["max_tokens"] = config.max_tokens
            if config.presence_penalty is not None:
                payload["presence_penalty"] = config.presence_penalty
            if config.frequency_penalty is not None:
                payload["frequency_penalty"] = config.frequency_penalty

        return InputAPI(url=url, headers=headers, payload=payload)

    def process_response(self, response: dict) -> OutputAPI:
        return OutputAPI(
            completion=response["choices"][0]["message"]["content"],
            input_tokens=response.get("usage", {}).get("prompt_tokens", 0),
            output_tokens=response.get("usage", {}).get(
                "completion_tokens", 0),
            error=response.get("error")
        )

    def format_prompt(self, prompt: Message) -> _MessageAPI:
        if prompt.role == RoleType.USER:
            role = "user"
        elif prompt.role == RoleType.ASSISTANT:
            role = "assistant"
        elif prompt.role == RoleType.SYSTEM:
            role = "system"
        else:
            raise ValueError(f"Unknown role type: {prompt.role}")

        return _MessageAPI(
            role_field_name="role",
            role_field_value=role,
            content_field_name="content",
            content_field_value=prompt.content
        )


@dataclass
class AzureCognitiveAdapter(LMAdapter):
    """
    An adapter for Azure Cognitive Services API to prepare payloads and process responses.
    """

    def __init__(self, model_name: str, **kwargs):
        super().__init__(provider=ProviderAPI.AZURE_COGNITIVE, model_name=model_name, **kwargs)

    def prepare_call(
        self,
        messages: list[_MessageAPI],
        config: Optional[LMConfig] = None
    ) -> InputAPI:
        url = "https://westus2-tip.api.cognitive.microsoft.com/models/chat/completions?api-version=2024-05-01-preview"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}"
        }
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    message.role_field_name: message.role_field_value,
                    message.content_field_name: message.content_field_value,
                } for message in messages
            ]
        }

        if config:
            if config.temperature is not None:
                payload["temperature"] = config.temperature
            if config.top_p is not None:
                payload["top_p"] = config.top_p
            if config.max_tokens is not None:
                payload["max_tokens"] = config.max_tokens
            if config.presence_penalty is not None:
                payload["presence_penalty"] = config.presence_penalty
            if config.frequency_penalty is not None:
                payload["frequency_penalty"] = config.frequency_penalty

        return InputAPI(url=url, headers=headers, payload=payload)

    def process_response(self, response: dict) -> OutputAPI:
        return OutputAPI(
            completion=response["choices"][0]["message"]["content"],
            input_tokens=response.get("usage", {}).get("prompt_tokens", 0),
            output_tokens=response.get("usage", {}).get(
                "completion_tokens", 0),
            error=response.get("error")
        )

    def format_prompt(self, prompt: Message) -> _MessageAPI:
        if prompt.role == RoleType.USER:
            role = "user"
        elif prompt.role == RoleType.ASSISTANT:
            role = "assistant"
        elif prompt.role == RoleType.SYSTEM:
            role = "system"
        else:
            raise ValueError(f"Unknown role type: {prompt.role}")

        return _MessageAPI(
            role_field_name="role",
            role_field_value=role,
            content_field_name="content",
            content_field_value=prompt.content
        )


@dataclass
class GoogleAdapter(LMAdapter):
    """
    An adapter for Google's API to prepare payloads and process responses.
    """

    def __init__(self, model_name: str, **kwargs):
        super().__init__(
            provider=ProviderAPI.GOOGLE,
            model_name=model_name,
            **kwargs)

    def prepare_call(
        self,
        messages: list[_MessageAPI],
        config: Optional[LMConfig] = None
    ) -> InputAPI:

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.token}"

        headers = {
            "Content-Type": "application/json",
        }

        system_message = None
        system_messages = [
            msg for msg in messages if msg.role_field_value == RoleType.SYSTEM.value]
        if system_messages:

            # at the time of writing, Gemma-3 models do not support system messages
            if "gemma-3" in self.model_name.lower():
                messages = [
                    msg if msg.role_field_value != RoleType.SYSTEM.value
                    else self.format_prompt(
                        UserMessage(content=msg.content_field_value)
                    ) for msg in messages
                ]
            else:
                if len(system_messages) > 1:
                    raise ValueError(
                        "Google API only supports a single system message.")
                system_message = {
                    "parts": [
                        {
                            system_messages[0].content_field_name: system_messages[0].content_field_value
                        }
                    ]
                }
                # take out system messages from messages
                messages = [
                    msg for msg in messages if msg.role_field_value != RoleType.SYSTEM.value]

        if not messages:
            raise ValueError(
                "At least one message is required for Google API.")

        if len(messages) == 1 and messages[0].role_field_value == RoleType.ASSISTANT.value:
            raise ValueError(
                "Google API requires a user message but only an assistant message was provided."
            )

        payload = {
            "contents": [{
                message.role_field_name: message.role_field_value,
                "parts": [
                    {
                        message.content_field_name: message.content_field_value
                    }
                ]
            } for message in messages]
        }

        if system_message is not None:
            payload["system_instruction"] = system_message  # type: ignore

        # Add generation config if provided
        if config:
            generation_config = {}

            if config.temperature is not None:
                generation_config["temperature"] = config.temperature
            if config.top_p is not None:
                generation_config["topP"] = config.top_p  # Note: camelCase
            if config.max_tokens is not None:
                generation_config["maxOutputTokens"] = config.max_tokens
            if config.stop is not None:
                generation_config["stopSequences"] = config.stop if isinstance(
                    config.stop, list) else [config.stop]
            if config.seed is not None:
                generation_config["seed"] = config.seed
            if config.log_probs is not None:
                generation_config["logProbs"] = config.log_probs
            if config.reasoning_effort is not None:
                generation_config["thinkingConfig"] = {
                    "thinkingLevel": config.reasoning_effort}
            if generation_config:
                payload["generationConfig"] = generation_config  # type: ignore

        return InputAPI(url=url, headers=headers, payload=payload)

    def process_response(self, response: dict) -> OutputAPI:
        # Handle error responses
        if "error" in response:
            return OutputAPI(
                completion="",
                input_tokens=0,
                output_tokens=0,
                error=response["error"]
            )

        # Extract the text from the response
        try:
            # Google's response structure
            completion = response["candidates"][0]["content"]["parts"][0]["text"]

            # Get token counts if available
            usage_metadata = response.get("usageMetadata", {})
            input_tokens = usage_metadata.get("promptTokenCount", 0)
            output_tokens = usage_metadata.get("candidatesTokenCount", 0)

        except (KeyError, IndexError) as e:
            return OutputAPI(
                completion="",
                input_tokens=0,
                output_tokens=0,
                error=f"Failed to parse response: {str(e)}"
            )

        return OutputAPI(
            completion=completion,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            error=None
        )

    def format_prompt(self, prompt: Message) -> _MessageAPI:
        # Google uses "user" and "model" as roles
        if prompt.role == RoleType.USER:
            role = "user"
        elif prompt.role == RoleType.ASSISTANT:
            role = "model"  # Google uses "model" instead of "assistant"
        elif prompt.role == RoleType.SYSTEM:
            # Google doesn't have a direct system role
            # Common workaround is to prepend to first user message
            # or use as a user message with instructions
            role = "system"
        else:
            raise ValueError(f"Unknown role type: {prompt.role}")

        return _MessageAPI(
            role_field_name="role",
            role_field_value=role,
            content_field_name="text",  # Google uses "parts" structure
            content_field_value=prompt.content
        )


@dataclass
class OpenAIAdapter(LMAdapter):
    """
    An adapter for OpenAI's API to prepare payloads and process responses.
    """

    def __init__(self, model_name: str, **kwargs):

        super().__init__(provider=ProviderAPI.OPENAI, model_name=model_name, **kwargs)

    def prepare_call(
        self,
        messages: list[_MessageAPI],
        config: Optional[LMConfig] = None
    ) -> InputAPI:
        # url = "https://api.openai.com/v1/chat/completions"  Deprecated

        url = "https://api.openai.com/v1/responses"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}"
        }
        payload = {
            "model": self.model_name,
            "input": [
                {
                    message.role_field_name: message.role_field_value,
                    message.content_field_name: message.content_field_value,
                } for message in messages
            ]
        }

        if config:
            if config.temperature is not None:
                payload["temperature"] = config.temperature
            if config.top_p is not None:
                payload["top_p"] = config.top_p
            if config.max_tokens is not None:
                payload["max_output_tokens"] = config.max_tokens
            if config.reasoning_effort is not None:
                payload["reasoning"] = {"effort": config.reasoning_effort}
            if config.reasoning_summary is not None:
                payload["reasoning"] = payload.get(
                    "reasoning", {})
                payload["reasoning"]["summary"] = config.reasoning_summary

        return InputAPI(url=url, headers=headers, payload=payload)

    def process_response(self, response: dict) -> OutputAPI:

        content_msg = [o for o in response.get(
            "output", []) if o.get("type", "") == "message"]
        if content_msg:
            content_msg = content_msg[0]["content"][0]["text"]
        else:
            content_msg = ""

        reasoning_msgs = [o for o in response.get(
            "output", []) if o.get("type", "") == "reasoning"]
        if reasoning_msgs:
            summaries = [s.get("text")
                         for s in reasoning_msgs[0].get("summary", [])]
            reasoning_msg = "\n\n".join(summaries)
        else:
            reasoning_msg = None

        return OutputAPI(
            completion=content_msg,
            reasoning_completion=reasoning_msg,
            input_tokens=response.get("usage", {}).get("input_tokens", 0),
            output_tokens=response.get("usage", {}).get(
                "output_tokens", 0),
            reasoning_tokens=response.get("usage", {}).get(
                "output_tokens_details", {}).get("reasoning_tokens", 0),
            error=response.get("error")
        )

    def format_prompt(self, prompt: Message) -> _MessageAPI:

        if prompt.role == RoleType.USER:
            role = "user"
        elif prompt.role == RoleType.ASSISTANT:
            role = "assistant"
        elif prompt.role == RoleType.SYSTEM:
            role = "developer"
        elif prompt.role == RoleType.FUNCTION:
            role = "function"
        else:
            raise ValueError(f"Unknown role type: {prompt.role}")

        return _MessageAPI(
            role_field_name="role",
            role_field_value=role,
            content_field_name="content",
            content_field_value=prompt.content
        )


@dataclass
class AzureAdapter(LMAdapter):
    """
    An adapter for Azure's OpenAI API to prepare payloads and process responses.

    see the following link for supported TRAPI models and deployment information:
    https://dev.azure.com/msresearch/TRAPI/_wiki/wikis/TRAPI.wiki/15124/Deployment-Model-Information

    """

    def __init__(
            self,
            model_name: str,
            model_builder: str,
            endpoint: str = "https://trapi.research.microsoft.com/gcr/shared/",
            api_version: str = "2024-05-01-preview",
            **kwargs
    ):

        super().__init__(provider=ProviderAPI.AZURE, model_name=model_name, **kwargs)
        self.model_builder = model_builder
        self.endpoint = endpoint
        self.api_version = api_version

    def prepare_call(
        self,
        messages: list[_MessageAPI],
        config: Optional[LMConfig] = None
    ) -> InputAPI:
        url = f"{self.endpoint}/{self.model_builder}/deployments/{self.model_name}"
        if self.model_builder != "openai":
            url += "/openai"  # odd TRAPI requirement
        url += f"/chat/completions?api-version={self.api_version}"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}"
        }
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    message.role_field_name: message.role_field_value,
                    message.content_field_name: message.content_field_value,
                } for message in messages
            ]
        }

        if config:
            if config.temperature is not None:
                payload["temperature"] = config.temperature
            if config.top_p is not None:
                payload["top_p"] = config.top_p
            if config.max_tokens is not None:
                payload["max_tokens"] = config.max_tokens
            if config.presence_penalty is not None:
                payload["presence_penalty"] = config.presence_penalty
            if config.frequency_penalty is not None:
                payload["frequency_penalty"] = config.frequency_penalty
            if config.stop is not None:
                payload["stop"] = config.stop
            if config.logit_bias is not None:
                payload["logit_bias"] = config.logit_bias
            if config.seed is not None:
                payload["seed"] = config.seed

        return InputAPI(url=url, headers=headers, payload=payload)

    def process_response(self, response: dict) -> OutputAPI:
        return OutputAPI(
            completion=response["choices"][0]["message"]["content"],
            input_tokens=response.get("usage", {}).get("prompt_tokens", 0),
            output_tokens=response.get("usage", {}).get(
                "completion_tokens", 0),
            error=response.get("error"))

    def format_prompt(self, prompt: Message) -> _MessageAPI:
        if prompt.role == RoleType.USER:
            role = "user"
        elif prompt.role == RoleType.ASSISTANT:
            role = "assistant"
        elif prompt.role == RoleType.SYSTEM:
            role = "system"
        elif prompt.role == RoleType.FUNCTION:
            role = "function"
        else:
            raise ValueError(f"Unknown role type: {prompt.role}")

        return _MessageAPI(
            role_field_name="role",
            role_field_value=role,
            content_field_name="content",
            content_field_value=prompt.content
        )


@dataclass
class AnthropicAdapter(LMAdapter):
    """
    An adapter for Anthropic's API to prepare payloads and process responses.
    """

    def __init__(self, model_name: str, **kwargs):
        super().__init__(provider=ProviderAPI.ANTHROPIC, model_name=model_name, **kwargs)

    def prepare_call(
        self,
        messages: list[_MessageAPI],
        config: Optional[LMConfig] = None
    ) -> InputAPI:
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "content-type": "application/json",
            "x-api-key": f"{self.token}",
            "anthropic-version": "2023-06-01"
        }

        system_message = None
        system_messages = [
            msg for msg in messages if msg.role_field_value == RoleType.SYSTEM.value]
        if system_messages:
            if len(system_messages) > 1:
                raise ValueError(
                    "Anthropic API only supports a single system message.")
            system_message = system_messages[0].content_field_value
            messages = [
                msg for msg in messages if msg.role_field_value != RoleType.SYSTEM.value]

        if not messages:
            raise ValueError(
                "At least one message is required for Anthropic API.")
        if len(messages) == 1 and messages[0].role_field_value == RoleType.ASSISTANT.value:
            raise ValueError(
                "Anthropic API requires at least one user message but only an assistant message was provided."
            )
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    message.role_field_name: message.role_field_value,
                    message.content_field_name: message.content_field_value,
                } for message in messages
            ],
            "max_tokens": 2048
        }
        if system_message:
            payload["system"] = system_message

        if config:
            if config.temperature is not None:
                payload["temperature"] = config.temperature
            if config.max_tokens is not None:
                payload["max_tokens"] = config.max_tokens

        return InputAPI(url=url, headers=headers, payload=payload)

    def process_response(self, response: dict) -> OutputAPI:
        completion = response.get("content", [])
        if completion:
            completion = completion[0].get('text', '')
        output = OutputAPI(
            completion=completion,
            input_tokens=response.get('usage', {}).get("input_tokens", 0),
            output_tokens=response.get('usage', {}).get("output_tokens", 0),
            error=response.get("error"))
        return output

    def format_prompt(self, prompt: Message) -> _MessageAPI:
        if prompt.role == RoleType.USER:
            role = "user"
        elif prompt.role == RoleType.ASSISTANT:
            role = "assistant"
        elif prompt.role == RoleType.SYSTEM:
            role = "system"  # Note: Anthropic API does NOT interleave system messages
        else:
            raise ValueError(f"Unknown role type: {prompt.role}")

        return _MessageAPI(
            role_field_name="role",
            role_field_value=role,
            content_field_name="content",
            content_field_value=prompt.content
        )


@dataclass
class CohereAdapter(LMAdapter):
    """
    An adapter for Cohere's API to prepare payloads and process responses.
    """

    def __init__(self, model_name: str, **kwargs):
        super().__init__(provider=ProviderAPI.COHERE, model_name=model_name, **kwargs)

    def prepare_call(
        self,
        messages: list[_MessageAPI],
        config: Optional[LMConfig] = None
    ) -> InputAPI:
        url = "https://api.cohere.ai/v2/chat"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"bearer {self.token}",
            "Accept": "application/json"
        }
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    message.role_field_name: message.role_field_value,
                    message.content_field_name: message.content_field_value,
                } for message in messages
            ]
        }

        if config:
            if config.temperature is not None:
                payload["temperature"] = config.temperature
            if config.max_tokens is not None:
                payload["max_tokens"] = config.max_tokens
            if config.top_p is not None:
                payload["top_p"] = config.top_p

        return InputAPI(url=url, headers=headers, payload=payload)

    def process_response(self, response: dict) -> OutputAPI:
        return OutputAPI(
            completion=response["message"]["content"][0]['text'],
            input_tokens=response.get("usage", {}).get(
                'tokens', {}).get("input_tokens", 0),
            output_tokens=response.get("usage", {}).get(
                'tokens', {}).get("output_tokens", 0),
            error=response.get("error"))

    def format_prompt(self, prompt: Message) -> _MessageAPI:
        if prompt.role == RoleType.USER:
            role = "user"
        elif prompt.role == RoleType.ASSISTANT:
            role = "assistant"
        elif prompt.role == RoleType.SYSTEM:
            role = "system"
        else:
            raise ValueError(f"Unknown role type: {prompt.role}")

        return _MessageAPI(
            role_field_name="role",
            role_field_value=role,
            content_field_name="content",
            content_field_value=prompt.content
        )


@dataclass
class TogetherAIAdapter(LMAdapter):
    """
    An adapter for TogetherAI's API to prepare payloads and process responses.
    """

    def __init__(self, model_name: str, **kwargs):
        super().__init__(provider=ProviderAPI.TOGETHERAI, model_name=model_name, **kwargs)

    def prepare_call(
        self,
        messages: list[_MessageAPI],
        config: Optional[LMConfig] = None
    ) -> InputAPI:
        url = "https://api.together.xyz/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}"
        }
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    message.role_field_name: message.role_field_value,
                    message.content_field_name: message.content_field_value,
                } for message in messages
            ]
        }

        if config:
            if config.temperature is not None:
                payload["temperature"] = config.temperature
            if config.max_tokens is not None:
                payload["max_tokens"] = config.max_tokens

        return InputAPI(url=url, headers=headers, payload=payload)

    def process_response(self, response: dict) -> OutputAPI:
        return OutputAPI(
            completion=response["choices"][0]["message"]["content"],
            input_tokens=response.get("usage", {}).get("prompt_tokens", 0),
            output_tokens=response.get("usage", {}).get(
                "completion_tokens", 0),
            error=response.get("error"))

    def format_prompt(self, prompt: Message) -> _MessageAPI:
        if prompt.role == RoleType.USER:
            role = "user"
        elif prompt.role == RoleType.ASSISTANT:
            role = "assistant"
        elif prompt.role == RoleType.SYSTEM:
            role = "system"
        else:
            raise ValueError(f"Unknown role type: {prompt.role}")

        return _MessageAPI(
            role_field_name="role",
            role_field_value=role,
            content_field_name="content",
            content_field_value=prompt.content
        )


@dataclass
class XAIAdapter(LMAdapter):
    """
    An adapter for Grok's API to prepare payloads and process responses.

    source: https://docs.x.ai/docs/guides/chat
    """

    def __init__(self, model_name: str, **kwargs):
        super().__init__(provider=ProviderAPI.XAI, model_name=model_name, **kwargs)

    def prepare_call(
        self,
        messages: list[_MessageAPI],
        config: Optional[LMConfig] = None
    ) -> InputAPI:
        url = "https://api.x.ai/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}"
        }
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    message.role_field_name: message.role_field_value,
                    message.content_field_name: message.content_field_value,
                } for message in messages
            ]
        }

        if config:
            if config.temperature is not None:
                payload["temperature"] = config.temperature
            if config.max_tokens is not None:
                payload["max_tokens"] = config.max_tokens

        return InputAPI(url=url, headers=headers, payload=payload)

    def process_response(self, response: dict) -> OutputAPI:
        return OutputAPI(
            completion=response["choices"][0]["message"]["content"],
            input_tokens=response.get("usage", {}).get("prompt_tokens", 0),
            output_tokens=response.get("usage", {}).get(
                "completion_tokens", 0),
            error=response.get("error"))

    def format_prompt(self, prompt: Message) -> _MessageAPI:
        if prompt.role == RoleType.USER:
            role = "user"
        elif prompt.role == RoleType.ASSISTANT:
            role = "assistant"
        elif prompt.role == RoleType.SYSTEM:
            role = "system"
        else:
            raise ValueError(f"Unknown role type: {prompt.role}")

        return _MessageAPI(
            role_field_name="role",
            role_field_value=role,
            content_field_name="content",
            content_field_value=prompt.content
        )


def example(
        provider: _SupportedProviders = ProviderAPI.AZURE.value,
        model_name: str = 'Meta-Llama-3.1-8B-Instruct',
        prompt: str = "Give me three of your favorite movie titles and nothing else.",
        num_samples: int = 1,
        temperature: float = 0.2,
) -> None:
    pass


# Example usage:
if __name__ == "__main__":

    import fire
    from azure.identity import (ChainedTokenCredential, AzureCliCredential,
                                ManagedIdentityCredential,
                                get_bearer_token_provider)
    fire.Fire(example)
