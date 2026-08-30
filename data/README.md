---
license: cc-by-nc-4.0
task_categories:
- text-generation
language:
- en
tags:
- Education
pretty_name: Question Anchored Tutoring Dialogues
size_categories:
- 1K<n<10K
configs:
- config_name: anchored-dialogues
  data_files:
  - split: train
    path: anchored-dialogues/train-*
  - split: test
    path: anchored-dialogues/test-*
- config_name: dq-question-metadata
  data_files:
  - split: train
    path: dq-question-metadata/train-*
dataset_info:
- config_name: anchored-dialogues
  features:
  - name: InterventionId
    dtype: int64
  - name: UserId
    dtype: int64
  - name: QuestionId_DQ
    dtype: int64
  - name: MessageSequence
    dtype: int64
  - name: IsTutor
    dtype: int64
  - name: MessageString
    dtype: string
  - name: TalkMovePrediction
    dtype: string
  splits:
  - name: train
    num_bytes: 4674549
    num_examples: 55322
  - name: test
    num_bytes: 1138842
    num_examples: 13395
  download_size: 1734395
  dataset_size: 5813391
- config_name: dq-question-metadata
  features:
  - name: QuestionId_DQ
    dtype: int64
  - name: InterventionId
    dtype: int64
  - name: MetaDataId
    dtype: int64
  - name: Text
    dtype: string
  - name: Sequence
    dtype: int64
  - name: MetaDataTagId
    dtype: int64
  - name: Label
    dtype: string
  splits:
  - name: train
    num_bytes: 1129952
    num_examples: 10857
  download_size: 350214
  dataset_size: 1129952
---

# Question-Anchored-Tutoring-Dialogues-2k

<!-- Provide a quick summary of the dataset. -->

This dataset contains dialogues from math tutoring interventions recorded on Eedi. 

## Dataset Details

### Dataset Description

<!-- Provide a longer summary of what this dataset is. -->
Each dialogue represents a chat-based conversation between a tutor and a student prompted by the student requesting assistance while working on a lesson. Dialogues are accompanied with 2 sources of meta-data:
1. DQ-Question-Metadata: The question the student was working on that prompted the tutoring session.
2. Dialogue-Subjects: The subject, topic(s), and subtopic(s) of the lesson the student was working on.

- **Curated by:** Matthew Zent, Digory Smith, and Simon Woodhead
- **Funded by:** Eedi
- **Language(s) (NLP):** English
- **License:** cc-by-nc-sa 4.0

### Dataset Sources [optional]

<!-- Provide the basic links for the dataset. -->

- **Repository:** [Eedi/Question-Anchored-Tutoring-Dialogues-2k](https://huggingface.co/datasets/Eedi/Question-Anchored-Tutoring-Dialogues-2k)
- **Paper** [IN PROGRESS]
- **Demo:** [Eedi/question-anchored-tutoring-dialogues](https://github.com/Eedi/question-anchored-tutoring-dialogues.git)


## Uses

<!-- Address questions around how the dataset is intended to be used. -->

### Direct Use

<!-- This section describes suitable use cases for the dataset. -->

This dataset is intended for non-commercial research focused on improving learning outcomes. Suitable use cases include, but are not limited to, model training, fine-tuning, and calibration tasks focused on effective dialogue modeling and tutoring interactions in educational contexts.

Researchers are encouraged to explore how dialogue-based interventions support student understanding and to use this data in ways that positively contribute to learning outcomes.

For commercial use, please contact us at hello@eedi.com.

### Out-of-Scope Use

<!-- This section addresses misuse, malicious use, and uses that the dataset will not work well for. -->

Any attempts to identify or re-identify individuals represented in the data are not allowed.

Additionally, use cases that risk adverse outcomes for students or educators, including but not limited to surveillance, profiling, or automated high-stakes decision-making without human oversight, are considered out of scope.

Dataset users are expected to carefully consider the impact of their work and avoid applications that could reinforce bias or inequality in learning environments.

## Dataset Structure

<!-- This section provides a description of the dataset fields, and additional information about the dataset structure such as criteria used to create the splits, relationships between data points, etc. -->

```
Eedi/Question-Anchored-Tutoring-Dialogues-2k/
├── README.md
├── anchored-dialogues/
│   ├── train.csv
│   ├── test.csv
├── dialogue-subjects.csv
├── dq-question-metadata.csv
└── dataset_config.yaml
```

### anchored-dialogues <a name='anchored-dialogues' id='anchored-dialogues'></a>

Each dialogue represents a conversation between a tutor and stutent on Eedi.
        The conversation started when the student asked for help while working on a Diagnostic 
        Question.
    

#### Glossary

| Term | Definition |
| --- | --- |
| InterventionId | A unique identifier for the intervention. |
| TutorId | Unique identifier for a tutor. |
| QuestionId_DQ | The QuestionId of the Diagnostic Question (DQ) that the intervention is anchored to. |
| MessageSequence | The sequence number of the message in the entire dialogue. |
| IsTutor | Defines if a message is from a tutor (1) or a student (0). |
| MessageString | The message content. |
| TalkMovePrediction | GPT-applied labels for prompts by the tutor used to support students’ mathematical thinking, understanding, and communication (see Talk Move Annotations). |

### dq-question-metadata <a name='dq-question-metadata' id='dq-question-metadata'></a>

Each Diagnostic Question (DQ) is presented to students as a single image,
        with the question and answer options embedded within the image. We have extracted the text 
        associated with the question and answer options. Where the question or answer contained an 
        image, the extracted text is a description of the image.

#### Glossary

| Term | Definition |
| --- | --- |
| QuestionId_DQ | The QuestionId of the Diagnostic Question (DQ) that the intervention is anchored to. |
| InterventionId | A unique identifier for the intervention this DQ is anchored to. |
| MetaDataId | Unique metadata identifier. |
| Text | Extracted text for the question. LaTeX is denoted as `\(...\)`. The text for image-based items is a description of the image. |
| Sequence | The order in which the text is presented. |
| MetaDataTagId | An integer value with a 1:1 mapping to the 'Label' column. |
| Label | A label denoting what the extracted text refers to. e.g. Answer B Text. |

### dialogue-subjects <a name='dialogue-subjects' id='dialogue-subjects'></a>

The subjects/topics of tutor-student dialogues.
    
The subjects are organised in a hierarchy, with increasing granularity as the
`SubjectLevel` increases. The `ParentSubjectId` allows us to navigate the
hierarchy.

Dialogues without a subject can occur when a student has asked for help outside of a lesson.
For example when asking for help with non-Eedi homework.

#### Glossary

| Term | Definition |
| --- | --- |
| InterventionId | A unique identifier for the intervention. |
| SubjectId | Unique subject identifier. |
| ParentSubjectId | Unique parent subject identifier. |
| SubjectName | The name of the subject. |
| SubjectLevel | The level of the subject. |
| SubjectType | The type of the subject, one of 'Subject', 'Topic', or 'Subtopic'. |


## Dataset Creation

### Source Data

<!-- This section describes the source data (e.g. news text and headlines, social media posts, translated sentences, ...). -->

Dialogues and all learning content are sourced from the math learning platform [Eedi](https://eedi.com/). Anchored-Dialogues include a column 'TalkMovePrediction' used to facilitate [Data Processing](#data collection and processing). Talk move predictions were sourced from [Moreau-Pernet et al. (2024)](https://dl.acm.org/doi/10.1145/3657604.3664664) (see [Talk Move Annotation](#talk moves))

#### Data Collection and Processing

<!-- This section describes the data collection and processing process such as data selection criteria, filtering and normalization methods, tools and libraries used, etc. -->

Conversation data and associated metadata were sourced from Eedi's internal databases from Nov. 2021 to Feb. 2025.

The dataset was curated through the following sequential filtering and processing steps:

1. **Initial Filtering**  
   Dialogues were selected if they contained at least 20 total messages, with a minimum of 7 messages from both the student and the tutor.

2. **Content Moderation**  
   Conversations were screened using OpenAI’s `omni-moderation-latest` model across the following safety categories:  
   `"sexual"`, `"sexual/minors"`, `"harassment"`, `"harassment/threatening"`, `"hate"`, `"hate/threatening"`, `"illicit"`, `"illicit/violent"`, `"self-harm"`, `"self-harm/intent"`, `"self-harm/instructions"`, `"violence"`, `"violence/graphic"`.

3. **Tutor Consent**  
   Only dialogues involving one of 25 tutors who provided explicit consent for research use were retained.

4. **Downsampling**  
   The dataset was downsampled to 2,000 conversations, with a cap of 1,000 unique questions and no more than 8 conversations per question. Sampling prioritized diversity in dialogue quality and structure by maximizing the TF-IDF score of each conversation's `TalkMovePredictions`.

5. **Manual Review**  
   Conversations and question texts were manually reviewed for potential personally identifiable information (PII) and content appropriateness. A total of 29 conversations were removed due to safety concerns or lack of educational value.

6. **Potential-PII Anonymization**
   Potential-PII was anonymized using [PIIvot](https://github.com/Eedi/PIIvot), which utilizes a hidden-in-plain-sight approach to replace labeled text with surrogates that maintain the context of the dialogue.

   Names, date of births, emails or social handles, urls, locations or addresses, phone numbers, and school names are replaced with realistic surrogates (see [Potential-PII Annotations](#potential-pii) for how labels were applied).
   
7. **Train/Test Splits**
   Train/test splits are obtained using a 0.8/0.2 split on the dialogue key 'InterventionId'.

   
#### Who are the source data producers?

<!-- This section describes the people or systems who originally created the data. It should also include self-reported demographic or identity information for the source data creators if this information is available. -->

This dataset originates from conversations between Eedi tutors and students to support student learning while completing online math lessons. Student demographic information is reported by student whereas platform informormation is by-intervention at the time of each intervention and includes multiple rows per student.

**Overview**
| Statistic                        | Count  |
|----------------------------------|--------|
| Unique students                  | 1,073  |
| Unique tutors                    | 25     |
| Unique Interventions             | 1,971  |
| Total messages                   | 68,717 |

**Student Demographics**
| Category             | Distribution                              |
|----------------------|-------------------------------------------|
| Gender               | Female: 314, Male: 159, Unknown: 600      |
| Income Status        | Low-income: 173, Not low-income: 480, Unknown: 420 |
| Location             | UK: 856, Unknown: 217                     |

**Platform Information**
| Category               | Average (25%, 75%)           |
|------------------------|------------------------------|
| DaysOnPlatform         | 140.5 (9, 210.3)         |
| HistoricQuestionCount  | 244 (25, 228)        |
| HistoricCorrectness    | 0.62 (0.54, 0.72)             |
| AgeAtIntervention      | 12 (11, 13)          |

### Annotations
<!-- If the dataset contains annotations which are not part of the initial data collection, use this section to describe them. -->

#### Talk Moves

To facilitate **downsampling**, talk moves were applied using the GPT-based classification model in [Moreau-Pernet et al. (2024)](https://dl.acm.org/doi/10.1145/3657604.3664664). The model was fine-tuned on conversation transcripts from small-group math tutoring sessions. 

To evaluate the model’s generalizability to 1:1 chat-based math tutoring sessions, the 1st author manually annotated a validation set of 200 tutor utterances sampled through weighted stratified sampling using the orignal codebook from [Moreau-Pernet et al. (2024)](https://dl.acm.org/doi/10.1145/3657604.3664664). The sample distribution was flattened by 0.8 of the original label distribution represented in the full 4k dialogue dataset in order to validate more examples of minority class labels.

An error analysis was conducted on all mismatched labels. **All** instances where the model predicted a `<Getting Student to Relate>` label involved a common type of Eedi's word problems involving a conversation between two fictional students. The original codebook was not designed with this unique edge-case in mind and as such we posit the `<Getting Student to Relate>` label may not generalize this to 1:1 context. Considering this limitation, we present an additional results excluding GSR labels and find similar acceptable metrics to the original paper.

| Label                        | Ratio of 4k Dialogues | Ratio of Validation Set  | Baptiste et al.  | F1 Score (Original) | F1 Score Excl. \<GSR\>   | F1 Score Baptiste et al.     |
|------------------------------|-----------------------|--------------------------|------------------|---------------------|------------------------|------------------------------|
| `<None>`                     | 0.42                  | 0.28                     | 0.73             | 0.8991              | 0.8991                 | 0.96                         |
| `<Keep Together>`            | 0.36                  | 0.20                     | 0.09             | 0.8333              | 0.8750                 | 0.81                         |
| `<Revoicing>`                | 0.12                  | 0.18                     | 0.03             | 0.8986              | 0.8986                 | 0.76                         |
| `<Press for Accuracy>`       | 0.06                  | 0.16                     | 0.13             | 0.7733              | 0.8286                 | 0.88                         |
| `<Getting Student to Relate>`| 0.02                  | 0.08                     | 0.004            | 0.0000              |   --                   | 0.75                         |
| `<Press for Reasoning>`      | 0.002                 | 0.06                     | 0.006            | 0.7857              | 0.9565                 | 0.94                         |
| `<Restating>`                | 0.0003                | 0.04                     | 0.008            | 0.8000              | 0.8000                 | 0.95                         |


#### Potential-PII
<!-- This section describes the annotation process such as annotation tools used in the process, the amount of data annotated, annotation guidelines provided to the annotators, interannotator statistics, annotation validation, etc. -->

To facilitate PII anonymization, both machine and human-annotated labels were applied to the data.

A [labeling codebook](https://eedi.notion.site/Student-Tutor-Dialogue-Annotation-Codebook-Public-1d91272d907280d794d4ffb3843ce1ba) was developed to label an independent set of ~40k student-tutor messages, achieving a minimum macro-averaged F1 score of 0.98 between raters over ~350 dialogues. 4 annotators from this original labeling initiative then applied the codebook to QATD-2k.
 
Machine labels for potential-PII were obtained using the [PIIvot](https://github.com/Eedi/PIIvot) package and a DeBERTa-based NER pipeline fine-tuned on the prior set of ~40k student-tutor messages. The NER model was trained to recognize seven types of entities that may contain Personally Identifiable Information (PII): names, date of births, emails or socials, urls, locations or addresses, phone numbers, and school names.

Disagreements between machine and annotator labels were resolved to determine ground truth labels. Micro-averaged metrics for both PIIvot and Annotator label sets in relation to this ground truth are included below.

| Label Set      | Precision | Recall   | F1 Score  |
|----------------|-----------|----------|-----------|
| **Dialogues**  |           |          |           |
| &nbsp;&nbsp;&nbsp;&nbsp;PIIvot     | 0.984     | 0.984    | 0.984     |
| &nbsp;&nbsp;&nbsp;&nbsp;Annotators | 0.993     | 0.995    | 0.994     |
| **Questions**  |           |          |           |
| &nbsp;&nbsp;&nbsp;&nbsp;PIIvot     | 0.991     | 0.699    | 0.820     |
| &nbsp;&nbsp;&nbsp;&nbsp;Annotators | 0.997     | 0.997    | 0.997     |


We make three important distinctions:

1. **Potential-PII**: Our approach does not attempt to differentiate between real and fictional entities when labeling for potential PII. This simplifies the NER objective by avoiding the need to resolve whether a given name is "real." We instead rely on the anonymization step to enforce dialogue coherence (i.e., ensuring that a name anonymized in the question remains consistent throughout the dialogue). While this is suitable for Eedi's math word problems, it may be less appropriate in domains where specific names carry meaning (e.g., a historical essay).

2. **Performance degradation on LaTeX questions**: The lower performance of the model on question metadata (compared to dialogues) is attributed to domain shift. The NER model was trained on dialogue data and struggles to generalize to LaTeX formated question text.

3. **NER Model release constraints**: Due to the sensitive nature of the labeling task, the fine-tuned NER model used in this process cannot be released publicly due to re-identification concerns. It could potentially be used to detect residual PII in datasets processed using the HIPS anonymization method.

#### Personal and Sensitive Information

<!-- State whether the dataset contains data that might be considered personal, sensitive, or private (e.g., data that reveals addresses, uniquely identifiable names or aliases, racial or ethnic origins, sexual orientations, religious beliefs, political opinions, financial or health data, etc.). If efforts were made to anonymize the data, describe the anonymization process. -->

This dataset contains real student-tutor dialogues. Extensive efforts were made to mitigate privacy risks, but these risks cannot be alleviated completely. Identifying details were removed or anonymized where detected, but some sensitive content may persist. If you have concerns about any content, please contact the first author at [matthew.zent@eedi.co.uk](mailto:matthew.zent@eedi.co.uk).

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

We present aggregate demographic and behavioral statistics to help downstream users assess potential biases in the dataset. Here we note highlight high-impact biases and limitations for downstream use:
1. QATD’s focuses on UK students and intentionally excludes US-based learners. 
2. The data reflects real interactions on the Eedi platform, where tutors sometimes manage multiple students simultaneously, especially during peak hours. 
3. The collection period includes growing public exposure to AI chatbots. As such, there are many instances of students expressing skepticism about whether their tutor was a real person. 
4. Our strong commitment to student privacy limits access to individual-level student details, which may constrain downstream use in student modeling tasks.

## Citation

Matthew Zent, Digory Smith, and Simon Woodhead. 2025. [PIIvot: A Lightweight NLP Anonymization Framework for Question-Anchored Tutoring Dialogues](https://aclanthology.org/2025.emnlp-main.1397/). In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 27479–27488, Suzhou, China. Association for Computational Linguistics.

## Dataset Card Author

Matthew Zent

## Dataset Card Contact

Matthew.Zent@eedi.com