import enum
from typing import Any, Optional
from dataclasses import dataclass, field
from .general_utilities import ColorType, print_c


class RoleType(enum.Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    FUNCTION = "function"


class ContentType(enum.Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


@dataclass
class Message:
    content: Any
    content_type: ContentType = ContentType.TEXT
    agent_id: Optional[str] = None
    role: RoleType = field(init=False)
    model_name: Optional[str] = None
    model_provider: Optional[str] = None

    def to_dict(self) -> dict:
        """
        Convert the message to a dictionary format.
        """
        return {
            "content": self.content,
            "content_type": self.content_type.value,
            "agent_id": self.agent_id,
            "role": self.role.value,
            "model_name": self.model_name,
            "model_provider": self.model_provider
        }


@dataclass
class AssistantMessage(Message):
    role: RoleType = field(default=RoleType.ASSISTANT, init=False)


@dataclass
class UserMessage(Message):
    role: RoleType = field(default=RoleType.USER, init=False)


@dataclass
class SystemMessage(Message):
    role: RoleType = field(default=RoleType.SYSTEM, init=False)


@dataclass
class Conversation:
    messages: list[Message]
    metadata: Optional[dict[str, Any]] = None
    identifier: Optional[str | int] = None

    def display(
            self,
            use_agent_ids: bool = False,
            use_model_names: bool = False) -> None:
        """
        Display the conversation in a readable format.
        """
        if self.metadata:
            print_c(f"Metadata: {self.metadata}", ColorType.BLUE)

        if use_agent_ids and use_model_names:
            unique_ids = set(
                [str(msg.model_name) + str(msg.agent_id)
                 for msg in self.messages]
            )
            color_map = {
                id_combo: ColorType.get_color(i) for i, id_combo in enumerate(unique_ids)
            }
        elif use_agent_ids:
            unique_ids = set(
                [str(msg.agent_id) for msg in self.messages])
            color_map = {
                agent_id: ColorType.get_color(i) for i, agent_id in enumerate(unique_ids)
            }
        elif use_model_names:
            unique_models = set(
                [str(msg.model_name) for msg in self.messages])
            color_map = {
                model: ColorType.get_color(i) for i, model in enumerate(unique_models)
            }
        else:
            color_map = {role.value: ColorType.get_color(
                i) for i, role in enumerate(RoleType)}

        if self.identifier is not None:
            print_c(f"Conversation ID: {self.identifier}", ColorType.BLUE)
        for message in self.messages:
            if use_agent_ids and use_model_names:
                id_combo = str(message.model_name) + str(message.agent_id)
                color = color_map.get(id_combo, ColorType.DEFAULT)
                role_prefix = f'{message.model_name} ({message.agent_id})'
            elif use_agent_ids:
                color = color_map.get(str(message.agent_id), ColorType.DEFAULT)
                role_prefix = str(message.agent_id)
            elif use_model_names:
                color = color_map.get(
                    str(message.model_name), ColorType.DEFAULT)
                role_prefix = str(message.model_name)
            else:
                color = color_map.get(message.role.value, ColorType.DEFAULT)
                role_prefix = message.role.value

            print_c(f'{role_prefix}:\n{message.content}\n', color=color)

    def as_transcript(
            self,
            mapping: dict[str, str],
            real_user_agent_id: Optional[str] = None,
            system_display: Optional[str] = "system",
            show_system_message: bool = False,
            show_real_user_message: bool = False
    ) -> str:
        """
        Convert the conversation to a transcript format.

        Args:
            show_system_message (bool): Whether to include system messages in the transcript.

        Returns:
            str: The formatted transcript.
        """
        transcript_lines = []
        for turn, message in enumerate(self.messages):

            if message.role == RoleType.SYSTEM:
                if not show_system_message:
                    continue
                role_display = system_display or "System"
            else:

                agent_id = message.agent_id
                if agent_id is None:
                    raise ValueError(
                        "Message must have an agent_id to be converted to a transcript.")

                if agent_id == real_user_agent_id:
                    if not show_real_user_message:
                        continue
                    role_display = mapping.get(agent_id, "user")
                else:
                    role_display = mapping[agent_id]

            content = message.content if isinstance(
                message.content, str) else str(message.content)
            transcript_lines.append(
                f"[{role_display}, turn {turn}]: {content}\n")

        return "\n".join(transcript_lines)

    def as_dict(self, add_turns=True) -> dict:
        """
        Convert the conversation to a dictionary format.

        Args:
            add_turns (bool): Whether to include turn numbers in the output.

        Returns:
            dict: The conversation as a dictionary.
        """

        if add_turns:
            messages_with_turns = []
            for i, msg in enumerate(self.messages):
                msg_dict = msg.to_dict()
                msg_dict["turn"] = i + 1
                messages_with_turns.append(msg_dict)
            return {
                "messages": messages_with_turns,
                "metadata": self.metadata,
                "identifier": self.identifier
            }
        else:
            return {
                "messages": [msg.to_dict() for msg in self.messages],
                "metadata": self.metadata,
                "identifier": self.identifier
            }


@dataclass
class PrimerConfig:
    primer_file: str
    num_pre_turns: Optional[int] = None
    early_stopping: Optional[bool] = None
    limited_budget: Optional[bool] = None
    same_agent_value: Optional[bool] = None
