# Cvartel Open Source Face SDK

[**Cvartel**](https://www.cvartel.com) Open Source Face SDK is an open-source library that analyzes images to identify human faces, facial landmarks and attributes.

Available for **C++, Python, C#, Java and JavaScript** on **Linux x86-64** and **Windows**.

---

## Try the Online Demo

Explore the SDK directly in your browser — no installation required:

| [**Try Cvartel Face API demo →**](https://www.cvartel.com/demo) |
|---|
| Interactive demo: liveness check, face matching, photo quality (ISO/ICAO), and face attributes (age, gender, emotions) |

> The online demo uses the **commercial-grade Cvartel algorithm**, which delivers the NIST FRTE 1:1 top-tier accuracy shown below. The open-source SDK in this repository contains the same architecture but ships with a different model — see the [Open Source vs Commercial](#open-source-vs-commercial) section for details.

---

## Open Source vs Commercial

The open-source SDK in this repository is a fully functional face recognition toolkit. The **commercial version** adds the high-accuracy Cvartel algorithm benchmarked in NIST FRTE 1:1, plus additional modules, deployment options and licensing terms.

### What you get in the Open Source SDK

| Module | Description |
|---|---|
| **Face Detection** | Detect one or multiple faces on an image with bounding boxes |
| **Face Landmarks (Mesh)** | Estimate face keypoints and facial mesh |
| **Face Recognition** | Build face templates, 1:1 verification and 1:N identification |
| **Liveness Detection** | Detect whether the face belongs to a live person (with confidence and verdict) |
| **Age Estimator** | Estimate the apparent age of a detected face |
| **Gender Estimator** | Estimate gender from a detected face |
| **Emotions Estimator** | Estimate emotions with confidence score |
| **Glasses Detector** | Detect whether the person wears glasses |
| **Mask Detector** | Detect whether the person wears a medical mask |
| **Eyes Openness Estimator** | Estimate openness for each eye separately |

> Open Source SDK ships with the base recognition model included in this repo. Accuracy is suitable for prototyping, research and majority of production use cases.

### What is available in the Commercial Version

| Module / Capability | Description |
|---|---|
| **NIST FRTE 1:1 algorithm** | The high-accuracy Cvartel face verification algorithm with 99.80% TAR (Visa vs border, FMR 1 in 1,000,000) — benchmarked in the official NIST FRTE report |
| **Deepfake Detection** | Detect synthetic and AI-generated face images |
| **Face Image Quality Assessment** | ISO/ICAO-compliant photo quality scoring for KYC and onboarding |
| **White-label Licensing** | Use Cvartel technology under your own brand |
| **Royalty-free Licensing** | Pay once, deploy without per-match fees |
| **In-country Deployment** | On-premise deployment for data-sovereignty and regulatory compliance |

The commercial version is designed for production-grade identity verification, KYC and access control where peak accuracy, additional detectors and a non-AGPL license are required. Suitable for banking, border control, government and enterprise use cases that need a proprietary license or localized in-country deployment.

**Contact:** info@cvartel.com — [request a commercial trial](https://www.cvartel.com/demo)

---

## NIST FRTE 1:1 Results (Commercial Algorithm)

The **commercial Cvartel algorithm** is evaluated in the **NIST FRTE 1:1** (Face Recognition Technology Evaluation, formerly FRVT) report. True match rates at the stated false match rate (FMR) for each scenario:

| Scenario | True Match Rate | FMR |
|---|---|---|
| Visa photo vs border camera | **99.80%** | 1 in 1,000,000 |
| Border vs border | **99.62%** | 1 in 1,000,000 |
| Border vs kiosk | **96.04%** | 1 in 100,000 |

[View the NIST report card →](https://face.nist.gov/frte/reportcards/11/cvartel_001.html)

> ⚠️ The numbers above reflect the **commercial Cvartel algorithm**, not the open-source model shipped in this repository. To evaluate the commercial algorithm in your own pipeline, request a trial at **info@cvartel.com** or try the [online demo](https://www.cvartel.com/demo).

---

## Use Cases

The Open Source SDK can be integrated into the following workflows:

- **Digital Onboarding and KYC** — verify remote users during account opening
- **Banking and Financial Services** — secure remote verification and transaction authentication
- **Public Safety** — proactive surveillance and threat detection
- **Access Control** — authenticate individuals for restricted physical and digital spaces
- **Identity Verification** — match a live selfie against a stored photo ID

---

## Quick Start

### 1. Download models

Download the [archive with models](https://drive.google.com/file/d/162OXlEh_18TLI0denqNysnBAGE3l8F-N/view?usp=drive_link) and unpack it into the `data/models` folder.

> **Note:** For Python, models are downloaded automatically to the package folder, or you can specify `--sdk_path` manually.

### 2. Install Python API

```bash
# Windows
pip3 install http://download.cvartel.com/facesdk/archives/artifacts/wl/windows/face_sdk-1.0.0-py3-none-any.whl

# Linux
pip3 install http://download.cvartel.com/facesdk/archives/artifacts/wl/linux/face_sdk-1.0.0-py3-none-any.whl
```

### 3. Build from source

| Platform | Build instructions |
|---|---|
| Linux x86-64 (C++) | [scripts/linux_x86_64/BUILD_LINUX_X86_64.md](scripts/linux_x86_64/BUILD_LINUX_X86_64.md) |
| Windows x86-64 MSVC 17+ (C++) | [scripts/windows/BUILD_WINDOWS.md](scripts/windows/BUILD_WINDOWS.md) |
| Linux x86-64 (C#) | [scripts/linux_x86_64/BUILD_CSHARP_LINUX_X86_64.md](scripts/linux_x86_64/BUILD_CSHARP_LINUX_X86_64.md) |
| Windows x86-64 (C#) | [scripts/windows/BUILD_CSHARP_WINDOWS.md](scripts/windows/BUILD_CSHARP_WINDOWS.md) |
| Java | Add `-DWITH_JAVA=ON -DJAVA_HOME="/usr/lib/jvm/java-8-openjdk-amd64"` to the cmake command |
| JavaScript | Copy `data/models/face_detector/` to `src/javascript_api/` and start any web-server in `src/javascript_api/` |

### 4. Run samples

#### face_demo

```bash
# Linux
LD_LIBRARY_PATH=../lib ./face_demo --mode detection --input_image <path_to_image>

# Windows
./face_demo.exe --mode detection --input_image <path_to_image>

# Python
python3 face_demo.py --mode detection --input_image <path_to_image>
```

Modes:
- `detection` — detect the face and draw the bounding box
- `landmarks` — estimate face keypoints
- `recognition` — detect a face on each image, build and compare face templates, print matching result and distance

#### estimator_demo

```bash
# Linux
LD_LIBRARY_PATH=../lib ./estimator_demo --mode all --input_image <path_to_image>

# Windows
./estimator_demo.exe --mode all --input_image <path_to_image>

# Python
python3 estimator_demo.py --mode all --input_image <path_to_image>
```

Modes:
- `all` — run all estimators
- `age` — estimate age
- `gender` — estimate gender
- `emotion` — estimate emotions with confidence
- `liveness` — estimate liveness with confidence and verdict
- `mask` — detect medical mask
- `glasses` — detect glasses with confidence
- `eye_openness` — estimate eyes openness with confidence per eye

---

## Project Structure

```
sdk/
├── 3rdparty/      # third-party dependencies
├── data/models/    # base models (empty by default, download separately)
├── img_samples/    # test images
├── include/        # SDK header files
├── samples/        # demo source code
├── scripts/        # build scripts per platform
└── src/            # SDK source code
```

---

## Contributions

Improvements to the SDK are always welcome. Please **fork** the repository and open a pull request.

Feel free to open an issue if you want to:

- report a bug
- submit a fix
- discuss the current state of the repo
- propose a new feature

---

## License

[Affero GPLv3 (AGPL-3.0)](http://www.gnu.org/licenses/agpl-3.0.html)

Commercial-friendly licensing (white-label, royalty-free, in-country deployment) is available. Contact **info@cvartel.com** for details.

---

## Links

- **Website:** <https://www.cvartel.com>
- **Online demo:** <https://www.cvartel.com/demo>
- **NIST FRTE 1:1 report card:** <https://face.nist.gov/frte/reportcards/11/cvartel_001.html>
- **Email:** <info@cvartel.com>
