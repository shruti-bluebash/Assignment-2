#!/usr/bin/env python3
"""Structured customer review analysis using PydanticOutputParser."""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Literal

from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, field_validator


DEFAULT_REVIEW = (
    "The phone has a great camera and the battery life is excellent. "
    "However, the device becomes slightly warm during gaming and the price is a bit high."
)


class ReviewAnalysisResponse(BaseModel):
    summary: str = Field(
        description="A brief summary of the customer review with maximum 3 lines"
    )
    positives: list[str] = Field(
        default_factory=list,
        max_length=3,
        description=(
            "A list showing the positives mentioned by the customer in the review "
            "if any - max 3 points"
        ),
    )
    negatives: list[str] = Field(
        default_factory=list,
        max_length=3,
        description=(
            "A list showing the negatives mentioned by the customer in the review "
            "if any - max 3 points"
        ),
    )
    sentiment: Literal["positive", "negative", "neutral"] = Field(
        description=(
            "One word showing the sentiment of the review - "
            "positive, negative or neutral"
        )
    )

    @field_validator("summary")
    @classmethod
    def summary_must_be_max_three_lines(cls, value: str) -> str:
        non_empty_lines = [line for line in value.splitlines() if line.strip()]
        if len(non_empty_lines) > 3:
            raise ValueError("summary must be at most 3 lines")
        return value.strip()


def _build_parser() -> PydanticOutputParser:
    return PydanticOutputParser(pydantic_object=ReviewAnalysisResponse)


def _message_to_text(message: object) -> str:
    if isinstance(message, str):
        return message
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                chunks.append(str(item["text"]))
            else:
                chunks.append(str(item))
        return "\n".join(chunks)
    return str(content)


def _extract_json_candidate(text: str) -> str:
    match = re.search(r"\{[\s\S]*\}", text)
    return match.group(0) if match else text


def _repair_with_llm(
    llm: ChatOpenAI, bad_output: str, parser: PydanticOutputParser
) -> str:
    repair_prompt = f"""
The text below does not match the required schema.
Rewrite it as valid JSON and return JSON only.

Required JSON format:
{parser.get_format_instructions()}

Invalid text:
{bad_output}
"""
    repaired = llm.invoke(repair_prompt)
    return _message_to_text(repaired)


def _parse_with_repair(
    raw_output: str, parser: PydanticOutputParser, llm: ChatOpenAI | None = None
) -> ReviewAnalysisResponse:
    parse_error: Exception | None = None

    for candidate in (raw_output, _extract_json_candidate(raw_output)):
        try:
            return parser.parse(candidate)
        except Exception as exc:  # noqa: BLE001
            parse_error = exc

    if llm is not None:
        repaired_output = _repair_with_llm(llm, raw_output, parser)
        for candidate in (repaired_output, _extract_json_candidate(repaired_output)):
            try:
                return parser.parse(candidate)
            except Exception as exc:  # noqa: BLE001
                parse_error = exc

    raise ValueError("Could not parse a valid ReviewAnalysisResponse") from parse_error


def _build_prompt(review: str, parser: PydanticOutputParser) -> str:
    return f"""
You are an AI assistant that analyzes customer reviews.
Return output strictly in the required JSON schema.

Rules:
- Summary must be concise and no more than 3 lines.
- Positives can have at most 3 bullet points.
- Negatives can have at most 3 bullet points.
- Sentiment must be one word: positive, negative, or neutral.
- Return JSON only (no markdown, no extra text).

Format instructions:
{parser.get_format_instructions()}

Review:
{review}
"""


def _mock_llm_output(review: str) -> str:
    lower = review.lower()

    positives: list[str] = []
    negatives: list[str] = []

    if "camera" in lower:
        positives.append("Great camera quality")
    if "battery" in lower:
        positives.append("Excellent battery life")
    if "warm" in lower or "heating" in lower or "hot" in lower:
        negatives.append("Device becomes warm during gaming")
    if "price" in lower or "expensive" in lower or "cost" in lower:
        negatives.append("Price is slightly high")

    positives = positives[:3]
    negatives = negatives[:3]

    if len(positives) > len(negatives):
        sentiment = "positive"
    elif len(positives) < len(negatives):
        sentiment = "negative"
    else:
        sentiment = "neutral"

    if positives and negatives:
        summary = (
            "The customer highlights useful strengths but also reports notable drawbacks."
        )
    elif positives:
        summary = "The review is largely positive and praises the product's strengths."
    elif negatives:
        summary = "The review is mostly critical and focuses on product issues."
    else:
        summary = "The review is mixed or unclear with limited specific details."

    payload = {
        "summary": summary,
        "positives": positives,
        "negatives": negatives,
        "sentiment": sentiment,
    }

    # Intentionally wrapped in extra text to demonstrate malformed-output handling.
    return f"Analysis result:\n{json.dumps(payload, indent=2)}"


def analyze_review(
    review: str,
    provider: Literal["openai", "mock"] = "openai",
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
) -> tuple[ReviewAnalysisResponse, str]:
    parser = _build_parser()

    if provider == "mock":
        raw_output = _mock_llm_output(review)
        parsed = _parse_with_repair(raw_output, parser, llm=None)
        return parsed, raw_output

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Set it or run with --provider mock."
        )

    llm = ChatOpenAI(model=model, temperature=temperature)
    prompt = _build_prompt(review, parser)
    raw_message = llm.invoke(prompt)
    raw_output = _message_to_text(raw_message)
    parsed = _parse_with_repair(raw_output, parser, llm=llm)
    return parsed, raw_output


def main() -> None:
    cli = argparse.ArgumentParser(
        description="Analyze a customer review into structured JSON."
    )
    cli.add_argument(
        "--review",
        type=str,
        default=DEFAULT_REVIEW,
        help="Customer review text to analyze.",
    )
    cli.add_argument(
        "--provider",
        type=str,
        choices=["openai", "mock"],
        default="mock",
        help="LLM provider: 'openai' (real model) or 'mock' (local demo).",
    )
    cli.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model name when provider=openai.",
    )
    cli.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for provider=openai.",
    )
    cli.add_argument(
        "--show-raw",
        action="store_true",
        help="Print raw model output before parsed JSON.",
    )
    args = cli.parse_args()

    parsed, raw_output = analyze_review(
        review=args.review,
        provider=args.provider,
        model=args.model,
        temperature=args.temperature,
    )

    if args.show_raw:
        print("RAW OUTPUT:")
        print(raw_output)
        print()

    print("STRUCTURED OUTPUT:")
    print(json.dumps(parsed.model_dump(), indent=2))


if __name__ == "__main__":
    main()
