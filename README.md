# SMS-Guard - experiments framework

This project is the framework to run experiments with different SLM against different propmts and SMS tagged datasets.
The main idea is to compare and get to conclusion on what the best SLM and prompt combination is appropiate for the
application: SMS phishing classification running on device (edge AI for privacy preservation)

## Setup
- create a virtual environment. I used python 3.13.2
- Install dependencies from [requirements.txt](requirements.txt):

```bash
pip install -r requirements.txt
```

- If you are going to use the Google API to run experiments instead of ollama:
  1. Create a `.env` file in the project root directory
  2. Add the following line to the `.env` file:
     ```
     GOOGLE_API_KEY=your_api_key_here
     ```
   If not, then install ollama, and pull a model.

- Make sure you a dataset in `datasets` folder (with .py extension)
- Make sure you have a `prompts.py` in the `prompts` folder

## How to run

1. Run experiment with either Google API (web) or Ollama (local)

Running with Google API

```bash
python scr/run_sms_experiment.py
```

Running with ollama
```bash
python scr/run_experiment_ollama.py
```

The results are going to be stored in the `experiment_results` folder as .csv file with timestamp (e.g. small_test_dataset_experiment_results_20250624_095916.csv)

2. analize results and get metric printed in the console
Change line 7 in the file `scr/get_metrics_from_csv.py` to update the name of the last result, for example to:
`CSV_FILENAME = os.path.join('experiment_results', 'small_test_dataset_experiment_results_20250624_095916.csv')`

then run

```bash
python scr/get_metrics_from_csv.py
```

You will get something like this:

| Prompt    | Acc    | FS     | TPR     | TNR    | FPR    | FNR   |
|-----------|--------|--------|---------|--------|--------|--------|
| prompt_01 | 62.79% | 65.22% | 100.00% | 42.86% | 57.14% | 0.00% |
| prompt_02 | 83.72% | 81.08% | 100.00% | 75.00% | 25.00% | 0.00% |


Juan Vargas
vargasjcamilo@gmail.com
