# Cvartel Open Source SDK

## Available Models
||||  
|--|--|--|  
|Face Detector :heavy_check_mark: |Face Mesh :heavy_check_mark: | Face Recognizer :heavy_check_mark:| 
|Gender Estimator :heavy_check_mark:| Age Estimator :heavy_check_mark:| Emotions Estimator :heavy_check_mark:| 
|Liveness Detector :heavy_check_mark:| Glass and Mask Detectors :heavy_check_mark:|Eyes Openness Estimator :heavy_check_mark:|
| Body Detector :heavy_check_mark:| Person Re-Identificator :heavy_check_mark:| Human Pose Estimator :heavy_check_mark:|


## APIs and Platforms support

|API Support|Linux x86-64| Windows| 
|--|--|--|
|C++            |:heavy_check_mark:|:heavy_check_mark:|
|Python         |:heavy_check_mark:|:heavy_check_mark:|
|C#             |:heavy_check_mark:|:heavy_check_mark:|
|Java           |:heavy_check_mark:|:heavy_check_mark:|
|JavaScript     |:heavy_check_mark:|:heavy_check_mark:|


## Project structure:
* [models](data/models)  - contains base models for SDK submodules. it's empty by default, you should download models by yourself;
* [image samples ](img_samples) - contains test images for running demo examples, also in the doc folder there are images with the result of the work of the modules;
* [samples](samples) - contains demo source code(see: [Sample](#sample));
* [scripts](scripts) - contains scripts for the basic assembly of the SDK(see: [Building the library](#building-the-library));
* [src](src) and [include](include) - contains SDKs source code 

## Prerequisites For Windows
* Visual C++ Redistributable for Visual Studio **2015 or higher**;


## Building the library
### Download models
Download [archive with models](https://drive.google.com/file/d/162OXlEh_18TLI0denqNysnBAGE3l8F-N/view?usp=drive_link) and unpack it in data/models folder
**Note:** Python download needed models to package folder or specific ```--sdk_path```
### C++ Build
Build instructions are available for the following platforms:
* [Linux x86-64](scripts/linux_x86_64/BUILD_LINUX_X86_64.md)
* [Windows x86-64 MSVC_17+](scripts/windows/BUILD_WINDOWS.md)
### Python API
Python API avaialble through wheel packages:
* For Windows
```bash
pip3 install  http://download.cvartel.com/facesdk/archives/artifacts/wl/windows/face_sdk-1.0.0-py3-none-any.whl
```
* For Linux:
```bash
pip3 install  http://download.cvartel.com/facesdk/archives/artifacts/wl/linux/face_sdk-1.0.0-py3-none-any.whl
```
* Alternatively, you can install python_api (located in _build/make-install_ after library building) via the _pip3 install <path_to_python_api>_ command to execute python sample
### Java API
To build Java bindings you should add _-DWITH_JAVA=ON -DJAVA_HOME="/usr/lib/jvm/java-8-openjdk-amd64"_ parameters to cmake comand. To build and execute Java sample you should go to _sdk/samples/java_ and 
```bash
mkdir bin 
javac -sourcepath ../../src/java_api/src/ -d bin com/face_detector_demo/face_detector_demo.java 

LD_LIBRARY_PATH=../../build/make-install/lib java -classpath ./bin com.face_detector_demo.face_detector_demo  path_to_image ../../build/make-install/
```
### C# API
Instructions are available for the following platforms:
* [Linux x86-64](scripts/linux_x86_64/BUILD_CSHARP_LINUX_X86_64.md)
* [Windows x86-64 MSVC_17+](scripts/windows/BUILD_CSHARP_WINDOWS.md)
### JavaScript API
* Copy _data/models/face_detector_ folder to _src/javascript_api_ directory.
* Start any web-server in _src/javascript_api_. For example, run following command in folder:
```bash
python3 -m http.server
```

## Samples

### face_demo
Startup arguments:
* `--mode` - optional, operating mode, default value is "detection", see the list of available modes below
* `--input_image` - required, path to the input image file
* `--input_image2` - required for recognition mode, path to the second input image file
* `--sdk_path` - optional, the path to the installed SDK directory, default value is ".." to launch from the default location _build/make-install/bin_
* `--window` - optional, allows to disable displaying window with results (specify any value except of "yes"), default value is "yes"
* `--output` - optional, allows to disable printing results (points coordinates) in console (specify any value except of "yes"), default value is "yes"

Run the following commands from the _build/make-install/bin_ directory to execute the sample:

* С++ (Windows): 
```bash
./face_demo.exe --mode detection --input_image <path to image>
```
* С++ (Linux): 
```bash
LD_LIBRARY_PATH=../lib ./face_demo --mode detection --input_image <path to image>
```

* Python: 
```bash
python3 face_demo.py --mode detection --input_image <path to image>
```


#### Sample modes
* **detection** - Detects the face on the input image visualize bounding box. 
* **landmarks** - Estimates face keypoints and visualize them.
* **recognition** - Detects a face on each image, builds and compares face patterns. Prints matching result and the distance between templates. Draw the crops of detected faces.  
_**Note:** If there is more than one face detected on the image, sample will process the detection with greatest confidence._


### estimator_demo
Startup arguments:
* `--mode` - optional, operating mode, default value is "all", see the list of available modes below
* `--input_image` - required, path to the input image file
* `--sdk_path` - optional, the path to the installed SDK directory, default value is ".." to launch from the default location _build/make-install/bin_
* `--window` - optional, allows to disable displaying window with results (specify any value except of "yes"), default value is "yes"
* `--output` - optional, allows to disable printing results (points coordinates) in console (specify any value except of "yes"), default value is "yes"

Run the following commands from the _build/make-install/bin_ directory to execute the sample:

* С++ (Windows): 
```bash
./estimator_demo.exe --mode all --input_image <path to image>
```
* С++ (Linux): 
```bash
LD_LIBRARY_PATH=../lib ./estimator_demo --mode all --input_image <path to image>
```

* Python: 
```bash
python3 estimator_demo.py --mode all --input_image <path to image>
```


#### Sample modes
* **all** - Launch all modes.
* **age** - Estimates age. 
* **gender** - Estimates gender.
* **emotion** - Estimates emotions and provides confidence.  
* **liveness** - Estimates liveness also provides confidence and verdict.
* **mask** - Detects medical mask.
* **glasses** - Estimates glasses and provides confidence.
* **eye_openness** - Estimates eyes openness and provides confidence about each eye openness.

### body_demo
Startup arguments:
* `--mode` - optional, operating mode, default value is "detection", see the list of available modes below
* `--input_image` - required, path to the input image file
* `--sdk_path` - optional, the path to the installed SDK directory, default value is ".." to launch from the default location _build/make-install/bin_
* `--output` - optional, allows to disable printing results (points coordinates) in console (specify any value except of "yes"), default value is "yes"

Run the following commands from the _build/make-install/bin_ directory to execute the sample:

* С++ (Windows): 
```bash
./body_demo.exe --mode detection --input_image <path to image>
```
* С++ (Linux): 
```bash
LD_LIBRARY_PATH=../lib ./body_demo --mode detection --input_image <path to image>
```

* Python: 
```bash
python3 body_demo.py --mode detection --input_image <path to image>
```


#### Sample modes
* **detection** - Detects human body on the input image visualize bounding box. .
* **pose** - Estimates skeleton keypoints and visualize them. 
* **reidentification** - Computes template by body crop.