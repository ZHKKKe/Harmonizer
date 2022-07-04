## Enhancer - Image Enhancement Demo
This is an offline demo of Enhancer Image Enhancement.  


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

2. Download the pre-trained model `enhancer.pth` from [this link](https://drive.google.com/file/d/19SsiV6wQ8W4x4Q9XgSPXST1VEnhoGg3R/view?usp=sharing) and put it into the folder `Harmonizer/pretrained/`.


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
    python -m demo.image_enhancement.run \
           --example-path ./demo/image_enhancement/example
    ```
    where `./demo/image_enhancement/example` contains a sample we provided.   
    If you want to test your own samples, please refer to the folder `./demo/image_enhancement/example` to prepare the `original` image.

6. Check the image enhancement results in the folder: `./demo/image_enhancement/example/enhanced`.
