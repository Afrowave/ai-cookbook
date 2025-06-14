import logging

from typing import Optional
from datetime import datetime
from ollama import chat
from pydantic import BaseModel, Field

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

ai_model = 'qwen3:14b'


# --------------------------------------------------------------
# Step 1: Define the data models for each stage
# --------------------------------------------------------------


class EventExtraction(BaseModel):
    """
    First LLM call: Extract basic event information
    """

    description: str = Field(description="Raw description of the event")
    is_calendar_event: bool = Field(
        description="Whether this text describes a calendar event"
    )
    confidence_score: float = Field(description="Confidence score between 0 and 1")


class EventDetails(BaseModel):
    """
    Second LLM call: Parse specific event details
    """

    name: str = Field(description="Name of the event")
    date: str = Field(
        description="Date and time of the event. Use ISO 8601 to format this value."
    )
    duration_minutes: int = Field(description="Expected duration in minutes")
    participants: list[str] = Field(description="List of participants")


class EventConfirmation(BaseModel):
    """
    Third LLM call: Generate confirmation message
    """

    confirmation_message: str = Field(
        description="Natural language confirmation message"
    )
    calendar_link: Optional[str] = Field(
        description="Generated calendar link if applicable"
    )


# --------------------------------------------------------------
# Step 2: Define the functions
# --------------------------------------------------------------

# Extract Event Info function
def extract_event_info(user_input: str) -> EventExtraction:
    """
    First LLM call to determine if input is a calendar event
    """

    logger.info("Starting event extraction analysis")
    logger.debug(f"Input text: {user_input}")

    today = datetime.now()
    date_context = f"Today is {today.strftime('%A, %B %d, %Y')}."

    response = chat(
        model=ai_model,  # Replace with the model you are using
        messages=[
            {"role": "system", "content": f"{date_context} Analyze if the text describes a calendar event."},
            {"role": "user", "content": user_input},
        ],
        format=EventExtraction.model_json_schema(),
    )


    result = EventExtraction.model_validate_json(response.message.content)
    logger.info(
        f"Extraction complete - Is calendar event: {result.is_calendar_event}, Confidence: {result.confidence_score:.2f}"
    )
    return result

# Process event details function
def parse_event_details(description: str) -> EventDetails:
    """
    Second LLM call to extract specific event details
    """

    logger.info("Starting event details parsing")

    today = datetime.now()
    date_context = f"Today is {today.strftime('%A, %B %d, %Y')}."

    response = chat(
        model=ai_model,  # Replace with the model you are using
        messages=[
            {"role": "system", "content": f"{date_context} Extract detailed event information. When dates reference 'next Tuesday' or similar relative dates, use this current date as reference."},
            {"role": "user", "content": description},
        ],
        format=EventDetails.model_json_schema(),
    )
    result = EventDetails.model_validate_json(response.message.content)
    logger.info(
        f"Parsed event details - Name: {result.name}, Date: {result.date}, Duration: {result.duration_minutes}min"
    )
    logger.debug(f"Participants: {', '.join(result.participants)}")
    return result


# Create confirmation message function
def generate_confirmation(event_details: EventDetails) -> EventConfirmation:
    """
    Third LLM call to generate a confirmation message
    """

    logger.info("Generating confirmation message")

    response = chat(
        model=ai_model,
        messages=[
            {"role": "system", "content": "Generate a natural confirmation message for the event. Sign of with your name; Susie"},
            {"role": "user", "content": str(event_details.model_dump())},
        ],
        format=EventConfirmation.model_json_schema(),
    )
    result = EventConfirmation.model_validate_json(response.message.content)
    logger.info("Confirmation message generated successfully")
    return result


# --------------------------------------------------------------
# Step 3: Chain the functions together
# --------------------------------------------------------------


def process_calendar_request(user_input: str) -> Optional[EventConfirmation]:
    """
    Main function implementing the prompt chain with gate check
    """

    logger.info("Processing calendar request")
    logger.debug(f"Raw input: {user_input}")

    # First LLM call: Extract basic info
    initial_extraction = extract_event_info(user_input)

    # Gate check: Verify if it's a calendar event with sufficient confidence
    if (
        not initial_extraction.is_calendar_event
        or initial_extraction.confidence_score < 0.7
    ):
        logger.warning(
            f"Gate check failed - is_calendar_event: {initial_extraction.is_calendar_event}, confidence: {initial_extraction.confidence_score:.2f}"
        )
        return None

    logger.info("Gate check passed, proceeding with event processing")

    # Second LLM call: Get detailed event information
    event_details = parse_event_details(initial_extraction.description)

    # Third LLM call: Generate confirmation
    confirmation = generate_confirmation(event_details)

    logger.info("Calendar request processing completed successfully")
    return confirmation


# --------------------------------------------------------------
# Step 4: Test the chain with a valid input
# --------------------------------------------------------------

user_input = "Let's schedule a 1h team meeting next Tuesday at 2pm with Alice and Bob to discuss the project roadmap."

result = process_calendar_request(user_input)
if result:
    print(f"Confirmation: {result.confirmation_message}")
    if result.calendar_link:
        print(f"Calendar Link: {result.calendar_link}")
else:
    print("This doesn't appear to be a calendar event request.")


# --------------------------------------------------------------
# Step 5: Test the chain with an invalid input
# --------------------------------------------------------------

user_input = "Can you send an email to Alice and Bob to discuss the project roadmap?"

result = process_calendar_request(user_input)
if result:
    print(f"Confirmation: {result.confirmation_message}")
    if result.calendar_link:
        print(f"Calendar Link: {result.calendar_link}")
else:
    print("This doesn't appear to be a calendar event request.")
