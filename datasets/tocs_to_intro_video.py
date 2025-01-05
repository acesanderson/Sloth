from Kramer.database.MongoDB_CRUD import get_all_courses_sync
from sloth.training.train_sloth import train_sloth
import html

model_name = "tocs_to_intro_video"

def clean_html_text(text):
    try:
        text = html.unescape(text)  # Unescape HTML entities first
        return str(text)
    except:
        return str(text)

def zap_instructors(description: str, instructor_name: str):
    try:
        return description.replace(instructor_name, "INSTRUCTOR_NAME")
    except:
        return description


def has_first_video_captions(course) -> bool:
    """
    Detects whether a course has captions in the first video entry.
    Returns true if they do, False if they don't.
    Some courses don't have entries, those also return False.
    """
    try:
        if course.sections[0].entries[0].transcript:
            return True
        else:
            return False
    except:
        return False


if __name__ == "__main__":
    courses = get_all_courses_sync()
    # Intputs: TOCs
    inputs = []
    # Outputs: First video of course
    outputs = []
    for course in courses:
        if has_first_video_captions(course):
            inputs.append(course.course_TOC_verbose)
            outputs.append(zap_instructors(course.sections[0].entries[0].transcript, course.metadata['Instructor Name'])) # type:ignore
    assert len(inputs) == len(outputs)

    # Modified prompt template for course descriptions
    course_prompt = """
    Look at this table of contents for a LinkedIn Learning video course, and write the transcript for the first video.

    ### Table of Contents
    {}


    ### Video transcript
    {}"""

    model, tokenizer = train_sloth(
        model_name=model_name,
        prompt_template=course_prompt,
        data={"inputs": inputs, "outputs": outputs},
    )
