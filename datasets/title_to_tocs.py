"""
Training dataset:
- "Look at this description of a LinkedIn Learnign video course:\n<description>{{Course Name}}: {{Course Description}}</description>\nCreate a descriptive Table of Contents for the course." -> course.TOC_verbose
"""

from Kramer.database.MongoDB_CRUD import get_all_courses_sync
from sloth.training.train_sloth import train_sloth
import html


def clean_html_text(text):
    try:
        text = html.unescape(text)  # Unescape HTML entities first
        return str(text)
    except:
        return str(text)


if __name__ == "__main__":
    courses = get_all_courses_sync()
    # Inputs: course titles
    inputs = [clean_html_text(course.metadata["Course Name"]) for course in courses]
    # Outputs: course TOCs
    outputs = [clean_html_text(course.course_TOC_verbose) for course in courses]
    assert len(inputs) == len(outputs)

    # Modified prompt template for course descriptions
    course_prompt = """
    Look at this title of a LinkedIn Learning video course, and create a descriptive Table of Contents for the course.

    ### Description
    {}


    ### Table of Contents:
    {}"""

    model, tokenizer = train_sloth(
        model_name="titles_to_tocs_70b",
        prompt_template=course_prompt,
        data={"inputs": inputs, "outputs": outputs},
    )
