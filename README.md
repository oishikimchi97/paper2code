# Paper2Code
Automatically generate the pytorch model code from a paper by utilizing the multi agent framework.
# Repository Architecture
```
├── agent
│   ├── model_creator.py
├── config
│   ├── llm_config.py
│   ├── OAI_CONFIG_LIST.json
├── data
│   ├── data1
│   │   ├── image
│   │   │   └── image1.png
│   │   │   └── ...
│   │   └── script.txt
├── initiate_chat.py
├── output
│   └── data1
│       ├── chat_log.log
├── README.md
└── utils
    └── utils.py
```

# Installation
1. Clone this repository on your machine.
2. Install Autogen. It can be installed from pip:
  ```sh
  pip install pyautogen[llm]
  ```
> **Note**:  
> Autogen requires Python version >= 3.8, < 3.12. You can reference [this site](https://github.com/microsoft/autogen?tab=readme-ov-file#installation).
3. Install other python packages if you need when executing the code.  
Otherwise you can use your own docker image. Check [this site](https://microsoft.github.io/autogen/docs/FAQ/#code-execution)
4. Create `OAI_CONFIG_LIST.json` file in `config/`.  
	Here is the sample of OAI_CONFIG_LIST.json
  ```json
  [
    {
      "model": "gpt-4-vision-preview",
      "api_key": "YOUR_API_KEY"
    },
    {
      "model": "gpt-4-1106-preview",
      "api_key": "YOUR_API_KEY"
    },
  ]
  ```

# Data Convention
In the each data folders, it must include the following file and folder.
1. script.txt  
  This is the script file that contains the text content in the paper.
2. image  
  This is the image folder that contains the image files from the paper.

# Usage
You can initiate the multi agent chatting to create the Pytorch model you want by the following command.
  ```sh
  python initiate_chat.py --data_dir ./data/your_data_folder --output_dir ./output
  ```
