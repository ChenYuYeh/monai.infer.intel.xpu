
# Overview
Create a clean Python repository for a healthcare AI medical image segmentation POC.

Goal:
Build an end-to-end MONAI-based CT segmentation demo.

Requirements:
1. Use MONAI and PyTorch.
2. Start from the official MONAI spleen 3D segmentation tutorial structure.
3. Refactor notebook-style code into Python modules:
   - src/train.py
   - src/infer.py
   - src/visualize.py
   - src/export_onnx.py
   - src/benchmark.py
4. Add a README that explains:
   - medical imaging task
   - dataset format
   - preprocessing pipeline
   - model architecture
   - inference flow
   - evaluation metric such as Dice score
   - deployment considerations
5. Make the code runnable on a single developer workstation.
6. Add AGENTS.md with setup, test, and verification instructions.

## References:

- Internal link:
https://docs.openvino.ai/2024/notebooks/ct-segmentation-quantize-nncf-with-output.html

- Medical data is from Kits19.
https://github.com/neheller/kits19

```mermaid
flowchart TD
    %% ===============================
    %% Data Sources
    %% ===============================
    A1[Medical Device / Imaging Source<br/>• Ultrasound Probe<br/>• Endoscope<br/>• CT / MRI Scanner<br/>• Surgical Camera]
    A2[PACS / DICOM Server]
    A3["Physiological Sensors<br/>(Vitals, Tracking, Metadata)"]

    %% ===============================
    %% Holoscan Ingestion Layer
    %% ===============================
    B1[Holoscan Input Operators<br/>• Frame Grabber<br/>• DICOM Reader<br/>• Sensor Adapter]

    %% ===============================
    %% Pre‑Processing (GXF Operators)
    %% ===============================
    C1["Pre‑Processing Operators<br/>(GXF)<br/>• Resize / Crop<br/>• Normalization<br/>• Denoising<br/>• GPU Memory Transfer"]

    %% ===============================
    %% AI Inference Paths
    %% ===============================
    D1["PyTorch Inference<br/>(TorchScript)<br/>• CNN / Transformer<br/>• MONAI Models"]

    %% ===============================
    %% Post‑Processing
    %% ===============================
    E1[Post‑Processing Operators<br/>• Thresholding<br/>• Morphology<br/>• Tracking<br/>• Temporal Smoothing]

    %% ===============================
    %% Clara Medical AI Layer
    %% ===============================
    F1[Clara AI Modules<br/>• Segmentation<br/>• Detection<br/>• Registration<br/>• Clinical Workflows]
    F2[Clara Validation & QA<br/>• Model Validation<br/>• Dataset Curation<br/>• Performance Metrics]
    F3[Clara Deployment Services<br/>• Model Lifecycle<br/>• Versioning<br/>• Rollback]

    %% ===============================
    %% Output & Integration
    %% ===============================
    G1[Visualization & UI<br/>• Surgical Overlay<br/>• Live Monitor<br/>• 3D Rendering]
    G2[DICOM Export / PACS<br/>• SR<br/>• RTSTRUCT<br/>• SEG]
    G3[Hospital IT Systems<br/>• EMR / HL7 / FHIR<br/>• Reporting]

    %% ===============================
    %% Data Flow Connections
    %% ===============================
    A1 --> B1
    A2 --> B1
    A3 --> B1

    B1 --> C1

    C1 --> D1

    D1 --> E1

    E1 --> F1
    F1 --> F2
    F2 --> F3

    F1 --> G1
    F1 --> G2
    F3 --> G3
```