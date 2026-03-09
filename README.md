# Structured Review Analysis using Pydantic Output Parser

This project implements an AI assistant that analyzes customer reviews and returns structured insights using a strict Pydantic schema and a LangChain `PydanticOutputParser`.

## 1) Source code

- Main application: `review_analyzer.py`

## 2) Pydantic schema used

```python
class ReviewAnalysisResponse(BaseModel):
    summary: str = Field(
        description="A brief summary of the customer review with maximum 3 lines"
    )
    positives: list[str] = Field(
        default_factory=list,
        max_length=3,
        description="A list showing the positives mentioned by the customer in the review if any - max 3 points",
    )
    negatives: list[str] = Field(
        default_factory=list,
        max_length=3,
        description="A list showing the negatives mentioned by the customer in the review if any - max 3 points",
    )
    sentiment: Literal["positive", "negative", "neutral"] = Field(
        description="One word showing the sentiment of the review - positive, negative or neutral"
    )
```

## 3) Example inputs and outputs

### Example input

```text
The phone has a great camera and the battery life is excellent.
However, the device becomes slightly warm during gaming and the price is a bit high.
```

### Example run (local mock mode)

```bash
python3 review_analyzer.py --provider mock --show-raw
```

### Example structured output

```json
{
  "summary": "The customer highlights useful strengths but also reports notable drawbacks.",
  "positives": [
    "Great camera quality",
    "Excellent battery life"
  ],
  "negatives": [
    "Device becomes warm during gaming",
    "Price is slightly high"
  ],
  "sentiment": "neutral"
}
```

### Example run with OpenAI

```bash
export OPENAI_API_KEY="your_key_here"
python3 review_analyzer.py --provider openai --model gpt-4o-mini
```

## 4) How the output parser enforces schema

1. `PydanticOutputParser` injects strict formatting instructions into the prompt.
2. The model output is parsed into `ReviewAnalysisResponse`.
3. If the output is malformed, the app:
   - first tries JSON extraction,
   - then asks the LLM to repair output to the required schema,
   - and parses again.
4. Pydantic validation enforces constraints like:
   - required keys,
   - list length limits (`max 3`),
   - `sentiment` restricted to `positive | negative | neutral`,
   - summary line limit.

## Setup

```bash
python3 -m pip install -r requirements.txt
```

## Quick command examples

```bash
# default review in mock mode
python3 review_analyzer.py

# custom review
python3 review_analyzer.py --provider mock --review "Service was fast, but food was cold."
```
# Assignment-2
