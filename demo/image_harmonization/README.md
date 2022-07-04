## Harmonizer - Image Harmonization Demo
This is an offline demo of Harmonizer Image Harmonization.  
We recommend you try our [online demo](https://zhke.io/#/?harmonizer_demo), which (1) does not require any coding and (2) includes an automatic foreground segmentation feature.


### 1. Requirements
The basic requirements for this demo are:
- Ubuntu System
- Python 3+


### 2. Run Demo
We recommend creating a new conda virtual environment to run this demo, as follow:

1. Clone this repository:
    ```
    git clone https://github.com/ZHKKKe/Harmonizer.git
    cd Harmonizer
    ```

2. Download the pre-trained model `harmonizer.pth` from [this link](https://drive.google.com/file/d/15XGPQHBppaYGnhsP9l7iOGZudXNw1WbA/view?usp=sharing) and put it into the folder `Harmonizer/pretrained/`.


3. Create a conda virtual environment named `harmonizer` (if it doesn't exist) and activate it. Here we use `python=3.8` as an example:
     ```
    conda create -n harmonizer python=3.8
    source activate harmonizer
    ```

4. Install PyTorch and the required python dependencies (please make sure your CUDA version is supported by the PyTorch version installed). In the root path of this repository, run:
    ```
    pip install -r src/requirements.txt
    ```

5. Execute the demo code in the root path of this repository, as:
    ```
    python -m demo.image_harmonization.run \
           --example-path ./demo/image_harmonization/example
    ```
    where `./demo/image_harmonization/example` contains some samples we provided.   
    If you want to test your own samples, please refer to the folder `./demo/image_harmonization/example` to prepare the `composite` image and the corresponding foreground `mask` image.

6. Check the image harmonization results in the folder: `./demo/image_harmonization/example/harmonized`.
