# Datasheet for OsteoBench

This datasheet follows the structure proposed by Gebru et al. (*Datasheets for Datasets*, 2018) and is the sole documentation file for this release. OsteoBench accompanies the OsteoPilot multi-agent framework described in the associated paper ("A Multi Agent AI Assistant for Osteoporosis Research at the Bone Microscale"), accepted at the ECCV 2026 BioImage Computing workshop. The proceedings are not yet available; the full reference will be added on publication.

SR-microCT bone volumes are typically not shareable as open data: acquisitions are institution-specific, reach terabyte scale, and often carry data-sharing restrictions that keep the raw images themselves private. Releasing the underlying scans behind OsteoBench is therefore not possible. What is released instead is the structured task and evaluation layer built on top of them: the questions, input/output contracts, and machine-checkable evaluation profiles, so that other groups working on bone microCT and osteoporosis research can benchmark and advance agentic tools against a realistic version of this workflow even without access to this particular dataset, by pointing the same question set at their own SR-microCT data.

## Motivation

**For what purpose was the dataset created?**
OsteoBench was created to evaluate tool-using LLM agents on multi-step SR-microCT bone image analysis for osteoporosis research, a domain that, unlike many medical-AI benchmarks, requires iterative planning, tool orchestration, and code generation over whole-volume 3D imaging rather than single-image question answering. Most existing medical-agent benchmarks largely assess text-based knowledge (multiple-choice) or image-grounded QA on static, pre-processed inputs, and typically record only final-answer success. This does not capture whether an agent reached its answer through a valid pipeline. OsteoBench instead scores the full execution trace, routing, task scope, tool use, produced artifacts, and final output, so that a plausible answer reached through an invalid path (fabricated data, a skipped preprocessing step, the wrong tool) is not rewarded as correct.

**Who created the dataset?**
Matteo Giannetto\*, Isabella Poles\*, and Marco D. Santambrogio (Politecnico di Milano, Italy), with Eleonora D'Arnese (University of Edinburgh, United Kingdom). \*Equal contribution.

**Who funded it?**
We acknowledge ISCRA for awarding this project access to the LEONARDO supercomputer, owned by the EuroHPC Joint Undertaking and hosted by CINECA (Italy). This work was supported by the Polisocial Award 2022, Politecnico di Milano.

## Composition

**What do the instances represent?**
Each instance is a structured question over SR-microCT bone imaging data: a natural-language `question`, `additional_instructions`, an explicit `output_instructions` contract, and a machine-checkable `evaluation` profile that encodes the expected execution path (which routing decision, which tools, which artifacts) as well as the expected final answer. Most instances reference one or more relative paths to a volume, mask, or dataset directory that provides the concrete input(s) for the question.

**How many instances are there?**
68 questions, held as five JSON files that follow the analysis pipeline end to end:

| File | Family | # questions | Covers |
|---|---|---|---|
| `data_qa.json` | Data inspection | 10 | Stack geometry, dtype, dynamic range, histograms, TIFF to volume intake |
| `preprocessing_qa.json` | Preprocessing | 8 | Normalization, Gaussian filtering, Otsu thresholding, individually and as ordered pipelines |
| `segmentation_qa.json` | Segmentation | 10 | Lacunar mask production from volumes, cross-sample/group comparison |
| `phenotyping_qa.json` | Classification & phenotyping | 25 | Per-lacuna and trabecular-bone morphometry, plus KNN block retrieval/class inference over the reference descriptor library |
| `statistics_qa.json` | Statistics | 15 | Segmentation-quality metrics (Dice/IoU/confusion matrix) and KNN leave-one-out evaluation (accuracy, ROC-AUC, calibration, distance analysis) |

By data type (drives input scope and output granularity):

| Data type | n |
|---|---|
| `single_patient_wsi` | 6 |
| `single_patient_image_volume` | 38 |
| `multiple_patient_image_volume` | 24 |
| `dataset` | 16 |

By execution modality:

| Execution modality | n |
|---|---|
| Specialist tools only | 40 |
| Tools & code sandbox | 28 |

By workflow depth:

| Workflow depth | n |
|---|---|
| Direct task (single delegated step) | 37 |
| Chained workflow (multi-step) | 31 |

By capability theme (multi-label, a question can touch more than one theme, so these do not sum to 68):

| Capability theme | n |
|---|---|
| Image data injection | 24 |
| General data injection | 26 |
| Image processing | 24 |
| Morphometry | 26 |
| Pretrained model use | 40 |
| Modelling | 20 |
| Performance computation | 41 |
| Performance analysis | 16 |

**Does the dataset contain all possible instances, or is it a sample?**
It is a curated set, not an exhaustive enumeration. Questions were authored to span three orthogonal difficulty axes: input scope (single volume, multiple volumes, whole-dataset/leave-one-out), workflow depth (direct vs. chained), and execution modality (tool-only vs. tool-plus-code), over a fixed underlying imaging cohort (see below), not sampled at random from a larger question pool.

**What data does each instance consist of?**

| Field | Type | Description |
|---|---|---|
| `id` | string | Unique question identifier |
| `data_type` | enum | `single_patient_wsi` / `single_patient_image_volume` / `multiple_patient_image_volume` / `dataset` |
| `volume_relative_path` / `mask_relative_path` / `gt_mask_relative_path` / `segmentation_relative_path` / `dataset_relative_path` / `patient_id` | string or list[string] | Input path(s)/identifier(s), relative to the configured workspace root; present only on the field(s) relevant to that question. Where one of these is a list, it enumerates alternative concrete inputs for the same question template, a benchmark run resolves the placeholder to one entry, not all of them at once |
| `question` | string | Natural-language request, with `{field}` placeholders (e.g. `{volume_relative_path}`, `{patient_id}`, `{dataset_relative_path}`, `{sample_id}`) resolved from the item's own fields before the question is shown to the agent |
| `additional_instructions` | string | Extra task framing / constraints |
| `output_instructions` | string | The exact output contract the final answer must satisfy |
| `rationale` | string | Why the question is in the benchmark |
| `capabilities` | list[string] | Pipeline capability tags the question exercises |
| `sample_id` | string or list[string] | Representative sample identifier(s) for the resolved input |
| `evaluation` | object | Five-dimension scoring profile (see below) |

### Dataset aliases and expected layouts

The repository does not distribute biomedical data. To run a question against a
compatible private dataset, resolve the paths below under
`<workspace_root>/data/`. Names such as `<sample_id>` and `<patient_id>` are
placeholders: each benchmark question supplies the concrete identifier it uses.

| Alias | Minimum layout required by the specifications |
|---|---|
| `dataset_a` | One directory per sample, `dataset_a/<sample_id>/`, containing the TIFF stack inspected by the data-ingestion questions. TIFF filenames are not constrained. |
| `dataset_b` | A flat directory of compressed NIfTI volumes: `dataset_b/<sample_id>.nii.gz`. |
| `dataset_c` | One directory per sample: `dataset_c/<sample_id>/<sample_id>_original.nii.gz`. Questions that require a reference mask also expect `dataset_c/<sample_id>/segmentation/<sample_id>_segmented.nii.gz`. |
| `dataset_d` | A flat directory of compressed NIfTI volumes: `dataset_d/<sample_id>.nii.gz`. |
| `dataset_e` | One directory per sample containing a source NIfTI volume: `dataset_e/<sample_id>/<sample_id>_original.nii.gz`. |
| `dataset_f` | Six or more grouping directories, each with `group_1/` and `group_2/` direct child folders. Each group folder contains its input `.nii` or `.nii.gz` files directly; symlinks are allowed. The single-group task uses `group_1/` only. |
| `dataset_g` | One directory per sample with both the source volume and ground truth: `dataset_g/<sample_id>/<sample_id>_original.nii.gz` and `dataset_g/<sample_id>/segmentation/<sample_id>_segmented.nii.gz`. |
| `dataset_h` | One directory per sample with the same source/ground-truth pairing used for cohort evaluation: `<sample_id>/<sample_id>_original.nii.gz` and `<sample_id>/segmentation/<sample_id>_segmented.nii.gz`. |
| `dataset_i` | KNN query blocks only. For every required patient/section, provide `morphometry.csv` and `metadata.json` in `<patient_id>_step1_Z0_z460-530_sectionT1/`, `<patient_id>_step1_Z0_z989-1059_sectionT2/`, and/or `<patient_id>_step1_Z0_z1519-1589_sectionT3/`. The local KNN reference library is separate from these query inputs. |
| `dataset_k` | A prebuilt leave-one-out query set: step-0, section-T2 morphometry CSV files plus `manifest.json`. The manifest must enumerate the query blocks and supply their full `patient_id` and `class_label` values. |

`dataset_j` is intentionally unused: it is neither a required alias nor a
missing public dataset.

*Evaluation profile (dimensions A to E).* Each `evaluation` object encodes the same five scoring dimensions described in the paper's evaluator:

- **`dimension_A_routing`**: which subgraph/route should have handled the question (`expected_primary_route`), which routes are mandatory (`necessary_routes`), and which are disallowed (`forbidden_routes`).
- **`dimension_B_task`**: the expected task framing given to the subgraph (`expected_task`) and the rationale an LLM judge uses to grade whether the delegated task stayed in scope (`rationale_for_llm_judge`).
- **`dimension_C_tools`**: the specific tools that must be called (`expected_tools`, `required_tool_sequence`), tools that must *not* be called (`forbidden_tools`), and whether the code sandbox is required (`requires_sandbox`).
- **`dimension_D_artifacts`**: whether the run must produce registered output files, and of what extension(s)/count (`required_extension_groups`).
- **`dimension_E_final_output`**: how the final answer is checked (`strategy`: exact-match / range-check / sanity-check / monotonicity / format-check / a bespoke Python verifier), the expected output variables and types, the concrete pass/fail `rules`, and a `biological_note` explaining what the expected value means domain-wise.

A run is only credited on dimension E if it also satisfied the routing, task-scope, tool-use, and artifact dimensions: a plausible final number reached via a wrong or skipped step is not scored as correct.

Underlying imaging cohort referenced by these questions (per the paper's experimental setup): SR-microCT scans of the human femoral head from ten donors, evenly split healthy / osteoporotic, 3,600 slices total (200 per volume), averaging 3100×3100 pixels in-plane, acquired at an isotropic voxel edge of 1.6 μm. Sample-level disease-class labels were obtained retrospectively from DEXA/CT studies; lacuna segmentation ground truth was manually annotated by two operators.

**Are there any missing data?**
Yes, by design, several fields are specific to a question's family and are absent otherwise: `gt_mask_relative_path`/`segmentation_relative_path` appear only where a precomputed ground-truth mask is used; `dataset_relative_path` appears only for multi-volume/grouped-cohort questions; `patient_id` appears only for the KNN block-retrieval questions. `id`, `data_type`, `question`, `output_instructions`, `capabilities`, and `evaluation` are populated on every instance.

**Is there a label or target?**
Yes. The `evaluation` object is the full grading target: it specifies the expected routing decision, required/forbidden tools and their order, required output artifacts, and the final-answer verification strategy together with the concrete pass/fail rules. This is richer than a single gold-answer string, by design, it lets a scorer attribute a failure to the specific pipeline stage where it occurred rather than only checking the last output.

## Collection process

**How was the data acquired?**
The underlying SR-microCT volumes were acquired at the Elettra Synchrotron in Trieste, Italy. The 68 questions and evaluation profiles in this release were authored on top of that existing imaging cohort and its derived products (preprocessed volumes, lacunar segmentation masks, per-lacuna morphometry tables, and a KNN reference descriptor library); the raw imaging data itself is external to this release.

**Who was involved in the data-collection process?**
Two biomechanical researchers from Politecnico di Milano, together with the Elettra beamline team and medical experts from San Raffaele Hospital (Milan, Italy), were involved in acquiring and preparing the underlying imaging cohort.

**Were any ethical review processes conducted?**
Yes. The femoral heads were collected with prior authorization from the Ethics Committee of San Raffaele Hospital (Milan, Italy), approved on 13/05/2020 and registered as ClinicalTrials.gov ID [NCT04787679](https://clinicaltrials.gov/study/NCT04787679), and with signed informed consent from the patients.

**Over what timeframe was the data collected?**
2022-2026 (preceding ECCV submission, July 2026).

**How was the question-answer content generated?**
The questions and their evaluation profiles were authored by AI researcher, purpose-built to dynamically test an agentic system's execution path (routing, tool sequencing, artifact production) rather than to probe static domain knowledge. Drafts were shared with the two biomechanical researchers involved in the underlying data collection for domain review of question framing and expected values.

**Has the data been validated for quality?**
Yes, along two complementary axes: (i) the authoring AI researcher validated that each question correctly exercises the intended agent execution dynamics (routing, tool sequencing, artifact production) it was designed to probe; and (ii) domain correctness was validated by grounding each question in the biomechanical researchers' own established analysis pipeline, and by actually running that pipeline with the corresponding state-of-the-art tools adopted for each stage (preprocessing, segmentation, morphometry, statistics) to confirm that the questions and their expected answers/evaluation rules are consistent with real tool output on the underlying data.

## Preprocessing applied for release

The released files differ from the lab-internal working set in the following ways:

- **Merged file layout.** `feature_extraction_qa.json` and `knn_feature_extraction_qa.json` were merged into `phenotyping_qa.json`; `statistics_qa.json` and `statistics_knn_qa.json` were merged into a single `statistics_qa.json`. Question `id`s were kept distinguishable across the two sub-groups within each merged file.
- **Removed machine- and project-identifying paths.** Absolute filesystem paths (revealing the host project/institution directory layout) were removed, including duplicate `*_absolute_path` fields; the corresponding `*_relative_path` fields are used instead, inlined into question prose via the same `{field}` placeholder convention used elsewhere. All `../` path segments were stripped.
- **Anonymized dataset-partition names.** Internal working-folder names for the underlying data partitions (e.g. ad hoc names describing how or when a sample batch was collected) were replaced with neutral codes (`dataset_a` through `dataset_i` and `dataset_k`, documented in the dataset-alias table above) to avoid identity leakage while still preventing path collisions between same-named samples drawn from different underlying partitions.
- **No changes to underlying scientific/numeric content** beyond the corrections listed above; question wording, output contracts, and evaluation rules are otherwise as authored.

## Uses

**Intended uses.**
Evaluating tool-using/agentic AI systems on multi-step SR-microCT bone image-analysis workflows: whether an agent routes a request correctly, selects and sequences the right tools, produces the artifacts it claims to, and reaches a correct, evidence-grounded final answer. Not a training set: there is no train/validation split, and the evaluation profiles are built for scoring an agent's execution trace, not for supervised fine-tuning.

**Out-of-scope uses.**
- The underlying imaging cohort is 10 specimens. Any accuracy, AUC, or significance figure computed against the KNN classification/leave-one-out questions should be read as a pipeline sanity check, not a clinically validated estimate of diagnostic performance.
- OsteoBench does not include the raw SR-microCT volumes (they are not publicly shareable, see the introduction above); it cannot be used to train or evaluate a model directly "out of the box" without independently obtaining a matching or equivalent SR-microCT dataset and pointing the question templates at it.
- Several evaluation profiles assume specific named tools and their behavior (from the accompanying OsteoPilot tool suite); scoring an agent built on a materially different tool interface will require adapting `dimension_C_tools`/`dimension_D_artifacts` expectations rather than reusing them verbatim.
- Must not be used as a clinical decision-support benchmark or as evidence of real-world diagnostic validity.

## Distribution

**How will the dataset be distributed?**
Through this repository, which is the primary distribution channel. It holds everything that is releasable: the 68 question specifications, their input/output contracts, and the machine-checkable evaluation profiles. The underlying SR-microCT images are not publishable, for the reasons given in the introduction above, and are therefore not distributed here or anywhere else. Running the benchmark means pointing these question specifications at a compatible SR-microCT dataset of your own, following the dataset-alias layouts documented above.

## Maintenance

**Who is supporting / maintaining the dataset?**
Matteo Giannetto and Isabella Poles (Politecnico di Milano).

**How can the curator be contacted?**
Matteo Giannetto ([matteo.giannetto@mail.polimi.it](mailto:matteo.giannetto@mail.polimi.it)) and Isabella Poles ([isabella.poles@polimi.it](mailto:isabella.poles@polimi.it)).

**Will the dataset be updated?**
Known candidates for a future revision include: scaling the underlying cohort beyond the current ten specimens, including SR-microCT ring artifacts detection and bone microcrack analysis tools usage.
