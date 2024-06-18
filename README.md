Welcome to the Riding_LLaMA wiki!

### Instruction to set up LLaMA 
https://github.com/facebookresearch/llama

After download the LLaMA files (e.g. parameters), run the script from the github above ./download.sh

The URL in the email is necessary while running the download.sh 

### How to set up environment  
In the instruction,`In a conda env with PyTorch / CUDA available clone and download this repository`.

A conda environment is a directory that contains a specific collection of conda packages that you have installed. 

#### 1. How to set up conda? 
Conda allows you to create separate environments, each containing their own files, packages, and package dependencies. The contents of each environment do not interact with each other.

The following URL contains how to set up conda . 

https://saturncloud.io/blog/how-to-create-a-conda-environment-with-a-specific-python-version/

In summary, 

`conda create --name myenv python=3.7`

`conda activate myenv`

`python --version`


#### my conda venv name in my ubuntu machine (big box) is `conda_env`

#### 2. Installation of PyTorch and CUDA in conda virtual environment

`conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia`

The following URL (https://www.cherryservers.com/blog/how-to-install-pytorch-ubuntu) has more explanation. 

During this step, python version is need to be upgraded, do following command. 

`conda install python=3.10.6`

The following URL (https://bobbyhadz.com/blog/syntax-error-future-feature-annotations-is-not-defined) has more explanation. 

### How to infer LLaMA

In the instruction, following command should be executed. 

```
torchrun --nproc_per_node 1 example_chat_completion.py \
    --ckpt_dir llama-2-7b-chat/ \
    --tokenizer_path tokenizer.model \
    --max_seq_len 512 --max_batch_size 6
```

The LLaMA directories are following. (llama-2-7b, llama-2-7b-chat, tokenizer.model)
The command should be ran like following. 

<pre>(conda_env) <font color="#8AE234"><b>soojung@soojung-X299-UD4</b></font>:<font color="#729FCF"><b>~/llama</b></font>$ torchrun --nproc_per_node 1 example_chat_completion.py --ckpt_dir /home/soojung/llama-2-7b-chat/ --tokenizer_path /home/soojung/tokenizer.model --max_seq_len 512 --max_batch_size 6</pre>

* if I run the command in my computer, I see memory error.
  
### Use this script to convert parameters to HuggingFace and run the test question

reference : https://ai.meta.com/blog/5-steps-to-getting-started-with-llama-2/

### How to run the python script with prompt to get answer 
1. put script test_script.py in root (~)
2. activate conda environment with command : `(base) soojung@soojung-X299-UD4:~$ conda activate conda_env`
3. When you see the conda_env is activated, then run the script : `(conda_env) soojung@soojung-X299-UD4:~$ python test_script.py`
4. You see from the command line as loading the parameters : Loading checkpoint shards: 100%|█████████████████████████████████████████████████████| 3/3 [00:06<00:00,  2.25s/it]
