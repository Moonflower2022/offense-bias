# offense-bias

## why

for a paper that was submitted to this [conference](https://deliberation.stanford.edu/ai-agent-good-alignment-safety-impact)

## how
most of it is written by Claude-Sonnet-4 or ChatGPT-o3, which is how i was able to experiment and change courses so fast

some small tweaks were written by me

## what
everything that isnt 
* `datasets/superset.csv`
* `outputs`
* `superset_claude.py`
* `plot_superset_claude.py`
* `plot_superset_claude_paper.py`
is just experimentation

four datasets:
* <https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech>
* <https://huggingface.co/datasets/manueltonneau/english-hate-speech-superset>
  * the one i actually used
* <https://github.com/hate-alert/HateXplain>
* <https://huggingface.co/datasets/badmatr11x/hate-offensive-speech>

three models:
* gemini-1.5-flash
  * i call it `gemini-flash`
* claude-3-5-sonnet-20240620
  * i call it `claude`
* https://huggingface.co/badmatr11x/distilroberta-base-offensive-hateful-speech-text-multiclassification
  * i call it `multi-roberta`

script naming usually goes `the_dataset_the_model.py`
other stuff:
* `cache`
  * (probably) cache only for `superset_claude.py`
* `outputs`
  * outputs of `superset_claude.py`, `plot_superset_claude.py`, and `plot_superset_claude_paper.py`
* `datasets`:
  * files for the datasets (one is missing bc i never downloaded it)
* `bias_analysis*`
  * bias analysis from `badmatr11x_multi-roberta_bias.py`