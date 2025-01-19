"""
Training dataset:
- "Look at this description of a LinkedIn Learnign video course:\n<description>{{Course Name}}: {{Course Description}}</description>\nCreate a descriptive Table of Contents for the course." -> course.TOC_verbose
"""

from Kramer.database.MongoDB_CRUD import get_all_courses_sync
import html
import json


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
    dataset = list(zip(inputs, outputs))
    dataset = [{"prompt": dataset[0], "completion": dataset[1]}]
    cutoff = int(0.9 * len(dataset))
    train_set = dataset[:cutoff]
    valid_set = dataset[: len(dataset) - cutoff]
    print(valid_set)
