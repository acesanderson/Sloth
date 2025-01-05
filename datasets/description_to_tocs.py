"""
Training dataset:
- "Look at this description of a LinkedIn Learnign video course:\n<description>{{Course Name}}: {{Course Description}}</description>\nCreate a descriptive Table of Contents for the course." -> course.TOC_verbose
"""

from Kramer.database.MongoDB_CRUD import get_all_courses_sync
from sloth.training.train_sloth import train_sloth
import html

# Data Preparation functions
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


if __name__ == "__main__":
    courses = get_all_courses_sync()

    # Inputs
    course_titles = [clean_html_text(course.metadata['Course Name']) for course in courses]
    course_descriptions = [clean_html_text(zap_instructors(course.metadata['Course Description'], course.metadata['Instructor Name'])) for course in courses]
    inputs = [clean_html_text(course.metadata['Course Name']) + ": " + clean_html_text(zap_instructors(course.metadata['Course Description'], course.metadata['Instructor Name'])) for course in courses]

    outputs = [clean_html_text(course.course_TOC_verbose) for course in courses]

    # Inputs: course titles
    assert len(inputs) == len(outputs)

    # Modified prompt template for course descriptions
    course_prompt = """
    Look at this title+description of a LinkedIn Learning video course, and create a descriptive Table of Contents for the course.

    ### Description
    {}


    ### Table of Contents:
    {}"""

    model, tokenizer = train_sloth(
        model_name="descriptions_to_tocs",
        prompt_template=course_prompt,
        data={"inputs": inputs, "outputs": outputs},
    )
