"""Small validated records shared by extraction, generation, and downloads."""

from typing import Literal, Annotated
from pydantic import BaseModel, ConfigDict, Field, model_validator

# Type alias to ensure a string is not empty
Nonempty = Annotated[str, Field(min_length=1)]


class Record(BaseModel):
    # Reject unexpected model fields and whitespace-only strings.
    # This ensures strict conformance to the expected JSON schemas.
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class RequirementDraft(Record):
    # A single hardware requirement extracted by the LLM
    statement: str = Field(min_length=1, max_length=800)
    source_quote: str = Field(min_length=1, max_length=1000)
    source_kind: Literal["text", "visual"]
    measurable: bool


class RequirementBatch(Record):
    # Used to parse an array of requirements from a single LLM response
    requirements: list[RequirementDraft]


class Requirement(RequirementDraft):
    # Extends the draft to include system-assigned IDs and page numbers
    id: str
    page: int


class CitationDraft(Record):
    # Links a generated test step back to the source evidence
    reference_id: str
    quote: str = Field(min_length=1)


class TestDraft(Record):
    # The generated testing procedure structure
    title: str = Field(min_length=1)
    status: Literal["draft", "insufficient_information"]
    preconditions: list[Nonempty]
    steps: list[Nonempty]
    pass_criteria: str | None
    fail_criteria: str | None
    missing_information: list[Nonempty]
    citations: list[CitationDraft]

    @model_validator(mode="after")
    def validate_decision(self):
        # A runnable-looking draft must include both criteria and source evidence.
        if self.status == "draft":
            if not self.steps or not self.pass_criteria or not self.fail_criteria or not self.citations:
                raise ValueError("draft requires steps, pass/fail criteria, and citations")
            if self.missing_information:
                raise ValueError("unresolved information requires abstention")
        elif not self.missing_information:
            # If they declare insufficient information, they must state why.
            raise ValueError("insufficient_information requires an explanation")

        # Do not retain a partial operational procedure after abstaining.
        if self.status == "insufficient_information":
            self.steps = []
            self.pass_criteria = self.fail_criteria = None
        return self