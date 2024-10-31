This repository is based on the [SeqDiffuSeq official repository](https://github.com/Yuanhy1997/SeqDiffuSeq/tree/main) for the paper [SeqDiffuSeq: Text Diffusion with Encoder-Decoder Transformers](https://arxiv.org/pdf/2212.10325.pdf)

The repository was slightly modified due to the fact that opacus models should be trained/finetuned sequentially. 

1. The enviroment setup is similar to the original repository. The instruction to setup the enviroment can be found in the [original repository](https://github.com/Yuanhy1997/SeqDiffuSeq/tree/main?tab=readme-ov-file#environment).

2. To train the model from sratch:
    - Data from the SeqDiffuSeq paper can be found [here](https://github.com/Yuanhy1997/SeqDiffuSeq/tree/main?tab=readme-ov-file#preparing-dataset). 
    - Place the data under the path data/\[dataset-name\]/
    - Learning the BPE tokenizer using the following command 
    ```
    python ./tokenizer_utils.py train-byte-level [dataset-name] 10000
    ```
    - To train (non-parallel), create a `ckpts` folder (if not already exist) and run the following lines:
    ```
    bash train_scripts/[training-script].sh 0
    ```

3. To finetune an existing model, run the following command:
    ```
    python train_opacus.py [dataset-name] [model-path] [num-training-samples] [seq-len] [epsilon]
    ```

4. To inference, run:
    ```
    bash inference_scrpts/[inference-script].sh \
    [model-path] \
    [path-to-save-results] \
    [diffusion-path]
    ```

    - Note that the model ends with ".pt" and the diffusion ends with ".npy"
    - Also if you used your own dataset, you need to make your own inference script, just copy from the `inference_scrpts` folder and change the dataset name and the path to your test data