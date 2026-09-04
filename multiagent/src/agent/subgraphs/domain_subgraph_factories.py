from __future__ import annotations

from pathlib import Path

from langchain_core.language_models import BaseChatModel

from .generic_subgraph_agent import SubgraphAgent

# import helper builders / tool lists / prompt pieces from wherever needed

REPO_ROOT = Path(__file__).resolve().parents[4]
TOOLS_DIR = REPO_ROOT / "multiagent" / "src" / "agent" / "tools"
CALCULATOR_TOOL_SPEC = (
    "__auto__",
    str(TOOLS_DIR / "calculator_tool.py"),
)
CSV_TABLE_AUDIT_TOOL_SPEC = (
    "__auto__",
    str(TOOLS_DIR / "csv_table_audit_tool.py"),
)


ALL_DOMAIN_TOOL_SPECS: list[tuple[str, str]] = [
    # ── Filesystem / format intake ──────────────────────────────────────────
    CALCULATOR_TOOL_SPEC,
    CSV_TABLE_AUDIT_TOOL_SPEC,
    ("__auto__", str(TOOLS_DIR / "session_tree_tool.py")),
    ("__auto__", str(TOOLS_DIR / "folder_listing_with_sizes_tool.py")),
    ("__auto__", str(TOOLS_DIR / "load_discovered_files_as_artifacts_tool.py")),
    ("__auto__", str(TOOLS_DIR / "volume_metadata_audit_tool.py")),
    ("__auto__", str(TOOLS_DIR / "volume_histogram_tool.py")),
    ("__auto__", str(TOOLS_DIR / "tiff_folder_to_nifti_tool.py")),
    # ── Imaging preprocessing ─────────────────────────────────────────────
    ("__auto__", str(TOOLS_DIR / "image_intensity_preprocessing_tools.py")),
    ("__auto__", str(TOOLS_DIR / "standard_microscopy_preprocessing_tool.py")),
    # ── Segmentation ──────────────────────────────────────────────────────
    ("__auto__", str(TOOLS_DIR / "microscopy_segmentation_tool.py")),
    # ── Quantitative analysis ─────────────────────────────────────────────
    ("__auto__", str(TOOLS_DIR / "lacunae_morphometry_tool.py")),
    ("__auto__", str(TOOLS_DIR / "patient_similarity_hotelling_t2_tool.py")),
    ("__auto__", str(TOOLS_DIR / "segmentation_metrics_tool.py")),
    ("__auto__", str(TOOLS_DIR / "mask_component_analysis_tools.py")),
]


def build_default_subgraphs(
    llm: BaseChatModel,
    strict_tool_loading: bool = False,
) -> list[SubgraphAgent]:
    subgraphs: list[SubgraphAgent] = []

    subgraphs.append(
        make_filesystem_and_format_intake(
            llm,
            strict_tool_loading=strict_tool_loading,
        )
    )

    subgraphs.append(
        make_imaging_preprocessing(
            llm,
            strict_tool_loading=strict_tool_loading,
        )
    )


    subgraphs.append(
        make_quantitative_imaging_analysis(
            llm,
            strict_tool_loading=strict_tool_loading,
        )
    )

    return subgraphs


def make_filesystem_and_format_intake(
    llm: BaseChatModel,
    strict_tool_loading: bool = False,
) -> SubgraphAgent:
    """
    Instantiate a SubgraphAgent specialised for inspecting unknown folders,
    listing files, converting formats, and performing lightweight volume data QA.
    """
    return SubgraphAgent(
        llm,
        key="filesystem_and_format_intake",
        title="Filesystem Discovery and Format Conversion",
        description=(
            "Use this subgraph ONLY to inspect the file system (list folders, check file sizes), "
            "check basic structural geometry (number of slices, bounding dimensions, metadata), "
            "perform lightweight volume data QA such as dtype inspection and dynamic range, "
            "compute adjustable intensity histograms, "
            "load discovered existing files as graph artifacts, and convert a folder of TIFFs "
            "to a NIfTI volume. "
            "Do not route segmentation, preprocessing, lacunae morphometry, segmentation metrics, "
            "registration, or model-training tasks here."
        ),
        tool_specs=[
            CALCULATOR_TOOL_SPEC,
            CSV_TABLE_AUDIT_TOOL_SPEC,
            (
                "__auto__",
                str(TOOLS_DIR / "session_tree_tool.py"),
            ),
            (
                "__auto__",
                str(TOOLS_DIR / "folder_listing_with_sizes_tool.py"),
            ),
            (
                "__auto__",
                str(TOOLS_DIR / "load_discovered_files_as_artifacts_tool.py"),
            ),
            (
                "__auto__",
                str(TOOLS_DIR / "volume_metadata_audit_tool.py"),
            ),
            (
                "__auto__",
                str(TOOLS_DIR / "volume_histogram_tool.py"),
            ),
            (
                "__auto__",
                str(TOOLS_DIR / "tiff_folder_to_nifti_tool.py"),
            ),
        ],
        planner_domain_context=(
            "You are the planner for the Filesystem Discovery and Format Conversion subgraph.\n"
            "This subgraph discovers what is inside folders, reports file sizes, reads basic "
            "dimensional headers (width, height, slice count), performs lightweight image data "
            "QA using available audit tools, registers discovered files as graph artifacts/attachments, "
            "and converts TIFF folders to NIfTI.\n"
            "When a routed task needs files found during discovery to be available to later "
            "subgraphs, plan an artifact-registration step after discovery.\n"
            "Use volume_metadata_audit when the task asks for volume dimensions, dtype/bit-depth, "
            "or dynamic range. "
            "Use volume_intensity_histogram when the task asks for histogram bin edges, counts, "
            "or normalized/relative bin frequencies with adjustable bin parameters. "
            "Use csv_table_audit for CSV shape/column audits and concise per-column or grouped "
            "aggregates such as count, mean, std, min, max, median, and sum. "
            "Use evaluate_math_expression for arithmetic, percentages, ratios, unitless formulas, "
            "or simple statistics; do not calculate numeric results in your own reasoning. "
            "If a task requires segmentation, lacunae morphometry, segmentation quality metrics, "
            "or preprocessing, report it as unsupported for this subgraph."
        ),
        executor_domain_context=(
            "You are the execution agent for Filesystem Discovery and Format Conversion.\n"
            "Use only the capabilities explicitly exposed by the available tools.\n\n"
            "Strict scope rules:\n"
            "- Use session_tree_tool or folder_listing_with_sizes_tool to explore directories.\n"
            "- Use load_discovered_files_as_artifacts to register existing discovered files as "
            "attachments for downstream subgraphs; it must not read or copy their contents.\n"
            "- Use volume_metadata_audit to get structural geometry (dimensions, slice count), "
            "or first-slice dtype/dynamic-range audits.\n"
            "- Use volume_intensity_histogram to compute equally spaced intensity histograms with "
            "adjustable bins, optional ranges, and normalized relative frequencies.\n"
            "- Use csv_table_audit to inspect CSV shape, column names, dtypes, missing counts, "
            "and aggregate CSV columns; do not read CSV contents manually when this tool fits.\n"
            "- Use tiff_folder_to_nifti_tool to convert formats without altering or filling missing data.\n"
            "- Use evaluate_math_expression for every arithmetic expression, percentage, ratio, "
            "formula, or simple statistic; do not compute numeric results yourself.\n"
            "- REJECT instructions for segmentation, preprocessing, lacunae parameter extraction, "
            "segmentation metrics, registration, report generation, or model training."
        ),
        summarizer_domain_context=(
            "You are the final summariser for the Filesystem Discovery and Format Conversion subgraph.\n"
            "Summarise discovered files, registered graph artifacts, file sizes, spatial geometry, "
            "conversion status, histograms, CSV audits/aggregates, and any basic data-QA statistics returned by tools. "
            "When numeric calculations were needed, report calculator results rather than recomputing them. "
            "Do not invent image intensities, dtypes, or pixel-level statistics not returned by a tool."
        ),
        capabilities=[
            "Inspect an active session tree.",
            "List folder contents with file sizes and aggregate size information.",
            "Audit basic dimensional metadata (X, Y, Z bounds) of TIFF-folders and NIfTI files.",
            "Compute lightweight data-QA statistics: dtype and dynamic range.",
            "Compute adjustable intensity histograms with bin edges, counts, and relative frequencies.",
            "Audit CSV files and compute concise per-column or grouped aggregate statistics.",
            "Evaluate safe deterministic arithmetic, formulas, ratios, percentages, and simple statistics.",
            "Register discovered existing files as graph artifacts/attachments for later subgraphs.",
            "Convert existing TIFF slice folders into NIfTI volumes.",
            "Register generated NIfTI outputs as contract-compliant attachments."
        ],
        strict_tool_loading=strict_tool_loading,
    )

def make_imaging_preprocessing(
    llm: BaseChatModel,
    strict_tool_loading: bool = False,
) -> SubgraphAgent:
    """
    Instantiate a SubgraphAgent specialised for SR-microCT imaging preprocessing
    and lacunae segmentation.
    """
    return SubgraphAgent(
        llm,
        key="imaging_manipulation",
        title="Imaging Manipulation",
        description=(
            "Preprocessing and segmentation subgraph for SR-microCT microscopy data. "
            "Supports a fixed preprocessing algorithm on TIFF (.tif/.tiff) slices, "
            "explicit preprocessing operations including min-max intensity scaling, "
            "Gaussian filtering with adjustable sigma, and Otsu thresholding, plus "
            "NIfTI (.nii/.nii.gz) volume lacunae segmentation via a pretrained U-Net model. "
            "Does not perform quantitative analysis, registration, model training, "
            "or report generation."
        ),
        tool_specs=[
            CALCULATOR_TOOL_SPEC,
            CSV_TABLE_AUDIT_TOOL_SPEC,
            (
                "__auto__",
                str(TOOLS_DIR / "image_intensity_preprocessing_tools.py"),
            ),
            (
                "__auto__",
                str(TOOLS_DIR / "standard_microscopy_preprocessing_tool.py"),
            ),
            (
                "__auto__",
                str(TOOLS_DIR / "microscopy_segmentation_tool.py"),
            ),
            (
                "__auto__",
                str(TOOLS_DIR / "session_tree_tool.py"),
            ),
        ],
        planner_domain_context=(
            "You are the planner for an SR-microCT imaging preprocessing and segmentation subgraph.\n"
            "This subgraph operates only on already available TIFF slices or NIfTI volumes.\n"
            "Supported preprocessing operations are min-max intensity normalization, "
            "Gaussian filtering with an explicit sigma value, and Otsu thresholding. "
            "Min-max normalization is linear scaling that remaps the input minimum to "
            "the requested output_min and the input maximum to output_max; it is not "
            "intended for ground truth masks, segmentation masks, or categorical label images.\n"
            "Pipeline integrity is mandatory for ordered biomedical workflows. If the task "
            "specifies preprocessing before segmentation, describe in the normal planning "
            "steps or planner_notes which output is expected to feed the next input. If a "
            "requested step is intentionally used only for QC or not used downstream, explain "
            "that choice with a concrete reason that must be disclosed to the user.\n"
            "If a requested preprocessing workflow contains both supported steps and unsupported "
            "steps, such as morphology, percentile clipping, connected-component analysis, or "
            "ring-artifact detection, mark the task as partially_supported when appropriate. "
            "Plan only the supported tool-backed preprocessing steps, and note that unsupported "
            "concrete computation steps may be candidates for code_sandbox if no other subgraph "
            "advertises them.\n"
            "Use evaluate_math_expression for arithmetic, parameter formulas, percentages, ratios, "
            "or simple statistics; do not calculate numeric results in your own reasoning.\n"
            "Use csv_table_audit for CSV outputs produced by tools when the task needs shape, "
            "columns, dtypes, missing values, or summary aggregates rather than row-level data.\n"
            "Its supported scope is limited to the capabilities exposed by its tools; "
            "do not plan quantitative analysis, registration, model training, visualisation, "
            "or report generation steps."
        ), 
        
        executor_domain_context=( 
            "You are the execution agent of an SR-microCT imaging preprocessing and segmentation subgraph.\n"
            "Use only the capabilities explicitly exposed by the available tools.\n\n"
            "Strict scope rules:\n"
            "- Only use capabilities that are explicitly represented by the available tools.\n"
            "- Use min_max_intensity_normalization only for grayscale intensity images/volumes; "
            "do not use it on ground truth masks, segmentation masks, or categorical labels.\n"
            "- Use gaussian_filter_image for Gaussian smoothing when the task specifies sigma.\n"
            "- Use otsu_threshold_image for Otsu intensity thresholding of grayscale images/volumes: "
            "by default it sets pixels/voxels at or below the Otsu threshold to 0 and preserves "
            "original intensity values above the threshold. Use output_mode='binary_mask' when "
            "the user asks for only the binary Otsu mask.\n"
            "- For ordered pipelines, pass each generated output into the next required step. "
            "Do not call segment_microscopy on the original volume when the requested workflow "
            "requires a preprocessed input. If you must deviate because of tool/model constraints, "
            "stop and leave a clear explanation in the execution history so the final answer can "
            "disclose it.\n"
            "- Use session_tree when you need to inspect the active session tree, a specific "
            "subfolder inside it, or a directory symlink such as the session data link. Use "
            "folder_listing_with_sizes for user-provided folders outside the session tree.\n"
            "- When a segmentation task specifies an exact output filename, suffix, or same-directory "
            "destination, pass an explicit output_path to segment_microscopy; do not rely on the "
            "default timestamped session filename.\n"
            "- Do not invent quantitative analysis, registration, visualisation, "
            "report generation, or model training steps.\n"
            "- Do not invent unsupported preprocessing operations such as ring-artifact detection, "
            "percentile clipping, or morphology unless a matching tool is available.\n"
            "- For partially supported preprocessing workflows, execute only the available "
            "tool-backed steps and leave unsupported steps clearly identified for router handoff.\n"
            "- Do not assume imaging modality details unless they are explicitly present in the context.\n"
            "- Prefer the minimum necessary next action.\n"
            "- Use evaluate_math_expression for every arithmetic expression, percentage, ratio, "
            "formula, or simple statistic; do not compute numeric results yourself.\n"
            "- Use csv_table_audit for CSV audit and aggregate extraction from tool-generated CSV files.\n"
            "- Do not restate prior tool outputs unless needed to decide the next tool call."
        ),
        summarizer_domain_context=(
            "You are the final summariser for an SR-microCT imaging preprocessing and segmentation subgraph.\n"
            "When numeric calculations were needed, report calculator results rather than recomputing them. "
            "When the task requested an ordered preprocessing/segmentation pipeline, report a "
            "concise plain-language pipeline audit and any methodology-changing deviation visible "
            "in the execution history. "
            "Do not invent operations, preprocessing parameters, segmentation assumptions, "
            "missing results, or future actions."
        ),
        capabilities=[
            "Fixed preprocessing algorithm on SR-microCT TIFF slices and NIfTI volumes.",
            "Min-max intensity normalization by linear scaling from input min/max to requested output min/max; not for ground truth masks or label masks.",
            "Gaussian filtering of TIFF/NIfTI image data with adjustable sigma.",
            "Otsu thresholding of grayscale TIFF/NIfTI image data into intensity-preserving thresholded images or binary masks.",
            "Automated lacunae segmentation via pretrained U-Net model.",
            "Session-aware execution over already available files and derived artefacts.",
            "Audit CSV files and compute concise per-column or grouped aggregate statistics.",
            "Evaluate safe deterministic arithmetic, formulas, ratios, percentages, and simple statistics.",
        ],
        strict_tool_loading=strict_tool_loading,
    )
    
    
    
    
def make_quantitative_imaging_analysis(
    llm: BaseChatModel,
    strict_tool_loading: bool = False,
) -> SubgraphAgent:
    """
    Instantiate a SubgraphAgent specialised for quantitative imaging analysis.
    """
    
    return SubgraphAgent(
        llm,
        key="quantitative_imaging_analysis",
        title="Quantitative Imaging Analysis",
        description=(
            "Quantitative analysis subgraph for existing image volumes and segmentation masks. "
            "Supports lacunae morphometric extraction from an original volume plus segmentation, "
            "Hotelling T2 KNN comparison/classification from generated lacunae morphometry CSVs, "
            "connected-component object counting, lacunae counting from 2D or 3D masks, "
            "post-filtering per-lacuna and per-crack volume or area measurement, "
            "bone volume extraction by Otsu thresholding, "
            "and segmentation-quality evaluation against ground truth, including single-pair and "
            "batch workflows. Does not perform preprocessing, segmentation generation, "
            "registration, or model training."
        ),
        tool_specs=[
            CALCULATOR_TOOL_SPEC,
            CSV_TABLE_AUDIT_TOOL_SPEC,
            (
                "__auto__",
                str(TOOLS_DIR / "lacunae_morphometry_tool.py"),
            ),
            (
                "__auto__",
                str(TOOLS_DIR / "patient_similarity_hotelling_t2_tool.py"),
            ),
            (
                "__auto__",
                str(TOOLS_DIR / "segmentation_metrics_tool.py"),
            ),
            (
                "__auto__",
                str(TOOLS_DIR / "mask_component_analysis_tools.py"),
            ),
            (
                "__auto__",
                str(TOOLS_DIR / "session_tree_tool.py"),
            ),
        ],
        planner_domain_context=(
            "You are the planner for a quantitative imaging analysis subgraph.\n"
            "This subgraph operates only on already available images, segmentations, "
            "and related derived files.\n"
            "Use count_connected_components for 2D/3D connected-component counting and "
            "component-size filtering in generic masks when no domain-specific wrapper fits. "
            "Use count_lacunae for lacunae counts, post-filtering per-lacuna voxel counts, "
            "physical lacuna volumes/areas, total lacuna volume/area, and optional CSV attachments "
            "from class-1 lacunar masks. Use count_cracks for the equivalent class-2 crack workflow. "
            "Use calculate_bone_volume_otsu "
            "for bone volume extraction from preprocessed microscopy data using Otsu thresholding; "
            "by default it excludes zero-valued outside-sample background from the Otsu calculation. "
            "When a user gives a voxel volume in mm^3, do not pass that number as a voxel side length. "
            "Convert voxel_volume_mm3 to an isotropic voxel_size_um as "
            "(voxel_volume_mm3 * 1e9) ** (1/3) and pass [side_um, side_um, side_um]. "
            "When a user gives voxel size in mm, convert to micrometers by multiplying by 1000. "
            "Use find_similar_patients_by_block_hotelling_t2 after morphometry extraction when a generated "
            "individual lacunae morphometry CSV should be compared against the default KNN reference library. "
            "For lacunar morphometry feature-extraction tasks, the canonical source of truth is "
            "`calculate_lacunae_parameters`.\n"
            "You must call `calculate_lacunae_parameters` before deriving any lacunar "
            "morphometry values, counts, distributions, percentiles, filters, or summaries.\n"
            "Do not use `count_lacunae` as a substitute for morphometry extraction. "
            "`count_lacunae` counts segmentation connected components and may not match the "
            "canonical morphometry population.\n"
            "Do not compute surface area, volume filtering, or morphometry statistics directly "
            "from the segmentation mask in code_sandbox unless `calculate_lacunae_parameters` "
            "has already succeeded and produced the canonical per-lacuna morphometry CSV.\n"
            "If `calculate_lacunae_parameters` fails, report failure or retry that tool. "
            "Do not replace it with a custom algorithm. "
            "For segmentation-quality evaluation, a predicted mask must be compared against an "
            "independent ground-truth mask. Never plan calculate_segmentation_metrics with the same "
            "file as both prediction_path and ground_truth_path; if no distinct ground-truth mask is "
            "available, mark the task as unsupported or incomplete instead of fabricating a self-pair. "
            "Its supported scope is limited to the capabilities exposed by its tools; "
            "Use evaluate_math_expression for arithmetic, percentages, ratios, formulas, "
            "or simple statistics; do not calculate numeric results in your own reasoning. "
            "Use csv_table_audit for CSV outputs when only shape, columns, missing values, "
            "aggregates such as count, mean, std, min, max, median, and sum, or safe "
            "threshold-based conditional row counts are needed. "
            "do not plan training, visualisation, preprocessing or registration steps "
            "unless those capabilities are explicitly exposed by the available tools."
        ),
        executor_domain_context=(
            "You are the execution agent of a quantitative imaging analysis subgraph.\n"
            "Use only the capabilities explicitly exposed by the available tools.\n\n"
            "Strict scope rules:\n"
            "- Only use capabilities that are explicitly represented by the available tools.\n"
            "- Use count_connected_components for generic object/component counts in binary "
            "or label masks, including min/max component-size filtering, when no domain-specific "
            "wrapper applies.\n"
            "- Use count_lacunae when the requested object count is lacunae in a lacunar mask; "
            "the default class value is 1. Use it for lacuna counts, per-lacuna voxel counts, "
            "physical lacuna volumes/areas, total lacuna volume/area, and optional per-lacuna CSV output.\n"
            "- Use count_cracks when the requested object count is cracks/cricche in a label mask; "
            "the default class value is 2. Use it for crack counts, per-crack voxel counts, "
            "physical crack volumes/areas, total crack volume/area, and optional per-crack CSV output.\n"
            "- Use calculate_bone_volume_otsu when the task asks for bone tissue volume from "
            "an Otsu-derived bone mask; its default Otsu rule excludes zero background voxels.\n"
            "- For physical calibration, voxel_size_um is a side-length vector, not voxel volume. "
            "If the user gives voxel volume in mm^3, convert it to isotropic side length with "
            "(voxel_volume_mm3 * 1e9) ** (1/3) before calling quantitative tools. "
            "If the user gives voxel size in mm, multiply by 1000 to get micrometers. "
            "Use evaluate_math_expression for the conversion when there is any ambiguity.\n"
            "- Use find_similar_patients_by_block_hotelling_t2 when the task asks for Hotelling T2 KNN "
            "block similarity or class inference from an individual lacunae morphometry CSV. For patient-level "
            "leave-one-out, pass exclude_patient_id as the full query patient_id before '_step' "
            "(for example T1_S26), never the prefix before the first underscore.\n"
            "- For lacunar morphometry feature-extraction tasks, the canonical source of truth is "
            "`calculate_lacunae_parameters`.\n"
            "- You must call `calculate_lacunae_parameters` before deriving any lacunar "
            "morphometry values, counts, distributions, percentiles, filters, or summaries.\n"
            "- Do not use `count_lacunae` as a substitute for morphometry extraction. "
            "`count_lacunae` counts segmentation connected components and may not match the "
            "canonical morphometry population.\n"
            "- Do not compute surface area, volume filtering, or morphometry statistics directly "
            "from the segmentation mask in code_sandbox unless `calculate_lacunae_parameters` "
            "has already succeeded and produced the canonical per-lacuna morphometry CSV.\n"
            "- If `calculate_lacunae_parameters` fails, report failure or retry that tool. "
            "Do not replace it with a custom algorithm.\n"
            "- Use calculate_segmentation_metrics only when you have a predicted mask and a distinct "
            "ground-truth mask. Before calling it, verify that prediction_path and ground_truth_path "
            "are not the same path and do not refer to the same file. If the only available candidate "
            "is a self-pair, do not call the metrics tool; report that the evaluation is incomplete "
            "because an independent ground-truth mask is missing.\n"
            "- Do not invent preprocessing, segmentation generation, registration, "
            "visualisation, report generation, or training steps.\n"
            "- Do not assume imaging modality details unless they are explicitly present in the context.\n"
            "- Prefer the minimum necessary next action.\n"
            "- Use evaluate_math_expression for every arithmetic expression, percentage, ratio, "
            "formula, or simple statistic; do not compute numeric results yourself.\n"
            "- Use csv_table_audit to audit and summarise CSV outputs from morphometry, batch metrics, "
            "or component-measurement tools when row-level values are not the final answer. Use its "
            "conditional_counts option for fixed-threshold row counts, such as Lc_Ob > 0.5.\n"
            "- If csv_table_audit has already shown that a needed column is missing or entirely null "
            "(for example area_um2 from a 3D count_lacunae component CSV), do not audit the same CSV "
            "again with the same columns and operations. Treat that as evidence that this CSV cannot "
            "answer the requested metric, then switch to a tool that can compute the metric or leave "
            "the unsupported remainder for router/summarizer handoff.\n"
            "- Do not restate prior tool outputs unless needed to decide the next tool call."
        ),
        summarizer_domain_context=(
            "You are the final summariser for a quantitative imaging analysis subgraph.\n"
            "When numeric calculations were needed, report calculator results rather than recomputing them. "
            "For physical units, keep voxel volume, voxel side length, micrometers, and millimeters distinct; "
            "do not restate voxel_size_um as a voxel volume or vice versa. "
            "For segmentation metrics, do not accept or present self-comparison runs "
            "(prediction and ground truth are the same file) as valid results. If the execution "
            "history shows a self-pair or a metrics tool failure with no later successful correction, "
            "state that the evaluation is invalid or incomplete and do not report its metrics as meaningful. "
            "Do not invent operations, interpretations, modality assumptions, "
            "missing results, or future actions."
        ),
        capabilities=[
            "Quantitative morphometric extraction from existing images and segmentation masks.",
            "Hotelling T2 KNN block similarity and class inference from generated individual morphometry CSVs.",
            "Generic connected component counting in 2D TIFF masks and 3D NIfTI/TIFF masks with component-size filtering.",
            "Lacunae count, per-lacuna voxel counts, physical lacuna volume/area, total lacuna volume/area, and optional CSV attachments from 2D or 3D class-1 masks.",
            "Crack count, per-crack voxel counts, physical crack volume/area, total crack volume/area, and optional CSV attachments from 2D or 3D class-2 masks.",
            "Bone volume extraction by Otsu thresholding with zero-valued background excluded by default.",
            "Segmentation quality evaluation against ground truth for single-pair and batch workflows.",
            "Session-aware execution over already available files and derived artefacts.",
            "Audit CSV files and compute concise per-column or grouped aggregate statistics.",
            "Compute safe CSV conditional row counts for fixed column-threshold predicates.",
            "Evaluate safe deterministic arithmetic, formulas, ratios, percentages, and simple statistics.",
        ],
        strict_tool_loading=strict_tool_loading,

    )
