### Purpose of this project

This is a simple ("for dummies") wrapper around unsloth and related libraries for the purpose of finetuning.

### Current state of project:
- [x] Have the basic unsloth workflow simplified into something I can use.
- [ ] In the middle of finetuning a full model on dataset (course titles + descriptions)
### To do
- [ ] create convenience functions for training and inference
- [ ] do bulk training in a queue:
    - [ ] course titles -> course descriptions
    - [ ] course descriptions -> course TOCs
    - [ ] course descriptions|TOCs -> intro video
    - [ ] course descriptions|TOCs|intro video -> individual video scripts
- [ ] add support for other models (llama3.2, mistral, gemma, phi, etc.)
- [ ] determine which models are cleared for commercial use
- [ ] add more detailed parameters

### Architecture
Sloth/
└── datasets/
      |── course_titles_and_descriptions.py
      |-- course_data_and_tocs.py
      |-- course_data_and_intro_videos.py
      |-- course_data_and_video_scripts.py
├── training/
└── inference/
