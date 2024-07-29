from __future__ import annotations
from typing import Any, Dict, List, Text

from transformers import GPT2TokenizerFast
from tiktoken.model import MODEL_PREFIX_TO_ENCODING
from rasa.engine.graph import ExecutionContext
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.nlu.tokenizers.tokenizer import Token, Tokenizer
from rasa.shared.nlu.training_data.message import Message


@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.MESSAGE_TOKENIZER, is_trainable=False
)
class BPETokenizer(Tokenizer):
    """Creates features for entity extraction."""

    @staticmethod
    def required_packages() -> List[Text]:
        """Any extra python dependencies required for this component to run."""
        return ["tokenizers"]

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        """Returns the component's default config."""
        return {
            # This *must* be added due to the parent class.
            "intent_tokenization_flag": False,
            # This *must* be added due to the parent class.
            "intent_split_symbol": "_",
        }

    def __init__(self, config: Dict[Text, Any]) -> None:
        """Initialize the tokenizer."""
        config = {**self.get_default_config()}
        super().__init__(config)
        self.tokenizer = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")

    @classmethod
    def create(
            cls,
            config: Dict[Text, Any],
            model_storage: ModelStorage,
            resource: Resource,
            execution_context: ExecutionContext,
    ) -> BPETokenizer:
        return cls(config)

    def tokenize(self, message: Message, attribute: Text) -> List[Token]:
        """Tokenizes the text of the provided message."""
        text = message.get(attribute)
        tokens = self.process_text(text)
        return self._apply_token_pattern(tokens)

    def process_text(self, text) -> List[Token]:
        words = self.tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
        tokens = []
        for word, offset in words:
            if word.startswith('Ġ'):
                tokens.append(Token(text=word.replace('Ġ', ''), start=offset[0]+1, end=offset[1]))
            else:
                tokens.append(Token(text=word.replace('Ġ', ''), start=offset[0], end=offset[1]))
        return tokens

    @classmethod
    def validate_config(cls, config: Dict[Text, Any]) -> None:
        """Validates that the component is configured properly."""
        models = set(MODEL_PREFIX_TO_ENCODING.values())
        model = config['model']
        if model not in models:
            raise ValueError(
                f"BPETokenizer invalid model {model}.")
