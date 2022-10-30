## Quick Start - Training Harmonizer


1. Download the iHarmony4 dataset and put it in the folder `./harmonizer/dataset/`
2. Pre-process the iHarmony4 dataset for training.  
We provide the processed Hday2night subset as an example in the folder `./harmonizer/dataset/iHarmony4/Hday2night`.  
You should convert other subsets to the same format for training. 
Otherwise, you need to implement new dataset loaders in the file `./harmonizer/data.py` to load datasets with other formats.
3. Run the training script by:
    ```
    cd ./harmonizer
    python -m script.train
    ```
    You can config the training arguments in the script.