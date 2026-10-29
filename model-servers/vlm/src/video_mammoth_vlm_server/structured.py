import json
from collections.abc import Sequence

import torch
from jsonschema import Draft202012Validator, SchemaError, ValidationError
from lmformatenforcer import JsonSchemaParser, TokenEnforcer
from lmformatenforcer.tokenenforcer import TokenEnforcerTokenizerData
from pydantic import JsonValue
from referencing import Registry
from referencing.exceptions import Unresolvable
from transformers import LogitsProcessor, PreTrainedTokenizerBase


class StructuredOutputError(RuntimeError):
    """The model finished, but its structured answer was invalid."""


class JSONDecoder:
    """Build the tokenizer lookup once and reuse it across schema requests."""

    def __init__(
        self, tokenizer: PreTrainedTokenizerBase, eos_token_ids: Sequence[int]
    ) -> None:
        self.tokenizer = tokenizer
        self.eos_token_ids = tuple(eos_token_ids)
        special_ids = set(tokenizer.all_special_ids)
        prefix_id = tokenizer.encode("0", add_special_tokens=False)[-1]
        prefix_text = tokenizer.decode([prefix_id])
        tokens = []
        for token_id in range(len(tokenizer)):
            if token_id in special_ids:
                continue
            isolated = tokenizer.decode([token_id])
            contextual = tokenizer.decode([prefix_id, token_id])[len(prefix_text) :]
            tokens.append((token_id, contextual, len(contextual) > len(isolated)))

        self.data = TokenEnforcerTokenizerData(
            tokens, self._decode, list(eos_token_ids), False, len(tokenizer)
        )

    def _decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids).rstrip("\ufffd")

    def constrain(
        self,
        schema: dict[str, JsonValue],
        prompt_length: int,
        end_thinking_id: int | None,
    ) -> "_JSONConstraint":
        try:
            Draft202012Validator.check_schema(schema)
        except SchemaError as cause:
            raise ValueError("Invalid JSON Schema.") from cause
        if schema.get("type") != "object":
            raise ValueError("Structured responses require an object-root schema.")
        # The library can raise generic exceptions for unsupported schema features.
        try:
            parser = JsonSchemaParser(schema)
        except Exception as cause:
            raise ValueError("Unsupported JSON Schema for local decoding.") from cause

        enforcer = TokenEnforcer(self.data, parser)
        parser.config.max_consecutive_whitespaces = 2
        validator = Draft202012Validator(schema, registry=Registry())
        return _JSONConstraint(
            enforcer, validator, self.eos_token_ids, prompt_length, end_thinking_id
        )


class _JSONConstraint(LogitsProcessor):
    def __init__(
        self,
        enforcer: TokenEnforcer,
        validator: Draft202012Validator,
        eos_token_ids: Sequence[int],
        prompt_length: int,
        end_thinking_id: int | None,
    ) -> None:
        self.enforcer = enforcer
        self.validator = validator
        self.eos_token_ids = tuple(eos_token_ids)
        self.prompt_length = prompt_length
        self.end_thinking_id = end_thinking_id
        self.answer_start = prompt_length if end_thinking_id is None else None

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        if self.answer_start is None:
            if (
                input_ids.shape[-1] <= self.prompt_length
                or int(input_ids[0, -1]) != self.end_thinking_id
            ):
                return scores
            self.answer_start = input_ids.shape[-1]

        answer_tokens = input_ids[0, self.answer_start :].tolist()
        allowed = self.enforcer.get_allowed_tokens(answer_tokens).allowed_tokens
        # EOS is legal only after the parser has a complete JSON value.
        ending = [token_id for token_id in self.eos_token_ids if token_id in allowed]
        if ending:
            allowed = ending
        constrained = torch.full_like(scores, -torch.inf)
        constrained[:, allowed] = scores[:, allowed]
        return constrained

    def validate(self, text: str) -> None:
        try:
            self.validator.validate(json.loads(text))
        except (ValueError, ValidationError, Unresolvable) as cause:
            raise StructuredOutputError(
                "The VLM stopped without a valid JSON answer matching the schema."
            ) from cause
