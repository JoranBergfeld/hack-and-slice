> **Historical plan, superseded where it conflicts with the guided exercise.**
> Use [the current specification (#13)](https://github.com/JoranBergfeld/hack-and-slice/issues/13)
> and the [opening slides and shared concepts](slides/README.md) for implementation and workshop guidance.
> The text below preserves earlier assumptions, not confirmed requirements or measured workshop results.
> In particular, the cost score and two-second rule are speculative, not established event rules;
> there is no required accuracy threshold, leaderboard, scorer, LLM comparison or deployment.
> Participants compare at least two chosen configurations with shared seed 42, split and preprocessing;
> model sizes and the three learning rates are configurable starting suggestions, not a fixed sweep.
> Early stopping and model selection use validation accuracy, not validation loss or cost.
> Training supports local CPU or independently submitted Azure ML jobs with explicit CPU/GPU selection
> in the participant's existing workspace, not automatic shared-cluster provisioning or a required sweep.
> Final test evaluation and demonstration-message prediction happen locally; deployment, cost investigation,
> LLM comparison and pretrained embeddings are optional next steps without provided solutions.
> Timing, accuracy and performance claims below have not been verified for this exercise.

How we build our own intent classifier (Hack & Slice #1)
This is the plan for the network people build on 27 November. It is written as input for a GitHub Copilot session in which you set up the repository yourself, so it explains each choice and the reason behind it: the data split, the tokenizer, every layer, the loss, the optimizer, the learning rate, training length and the Azure setup. Values called starting points are meant to be tuned on the validation set, not taken as final.

The job in numbers
Item	Value
Task	Pick one of 77 intents for a customer message to a bank
Data	Banking77 by PolyAI (CC BY 4.0): 10,003 training and 3,080 test messages
Message length	about 12 words on average (dataset paper); 54 characters on average in the test set
Score	running cost + €0.10 per misrouted message, per 1,000 messages. An answer slower than 2 seconds counts as misrouted
One accuracy point	10 messages per 1,000, so €1.00
Blind guess	1.3% accuracy and a loss of ln(77) = 4.34
Two reference points help set expectations. The paper that introduced the dataset reports test accuracies from 87.2% (BERT used as a fixed encoder) to 93.7% (BERT fine-tuned), with 93.4% for its own USE+ConveRT model. All of those start from large pretrained encoders. One public project reports about 84% with TF-IDF features and logistic regression. Our network learns its word vectors from nothing but the training messages, so measure it against the 84% classic baseline and treat the low 90s as what pretrained models reach.

The shape of the model
text
Copy
message
  1. tokenizer      lowercase, keep words            "my card hasn't arrived"
  2. vocabulary     word to id, 0 = <unk>            [3, 1, 57, 112]
  3. EmbeddingBag   look up 64 numbers per id, average them
  4. Linear 64 to 128, then ReLU, then Dropout 0.3
  5. Linear 128 to 77                                 77 scores (logits)
  6. softmax, only when you need probabilities
This is the same recipe the dataset paper uses for its classifier: one hidden ReLU layer with a softmax over the intents. The difference is the input. The paper feeds the classifier sentence vectors from big pretrained encoders; we learn our own word vectors from 9,002 training messages. That keeps the whole thing small enough to explain on the slides and to train in a minute or two on a CPU.

Data and validation
Get the data from the PolyAI repository (banking_data/train.csv and test.csv, columns text and category) or from Hugging Face (PolyAI/banking77).

Split 10% of the training file off as a validation set, stratified by intent and with a fixed seed. That gives 9,002 messages for training and 1,001 for validation, about 13 per intent. Stratify because a plain random split can leave a rare intent with almost no validation examples, and then its errors are invisible.

Every choice in this document, from the stopping point to the learning rate and the layer sizes, is made on validation. The 3,080 test messages are used once, by the scorer. They are public, so keeping them out of training is a team agreement rather than a technical barrier. Say that out loud at the start.

Check the label counts in the data step. A third-party analysis reports that the training split runs from about 35 to 182 messages per intent, while the test split has exactly 40 per intent. If that holds, a model trained on the raw split leans towards the frequent intents, and a class-weighted loss is worth a try (see the improvement list).

Tokenizer and vocabulary
Lowercase the message, keep every run of letters, digits and apostrophes (the regex [a-z0-9']+) and drop everything else. Messages are short and the intents hinge on keywords such as card, top up, transfer, pending and declined, so a word tokenizer does most of the work. It is two lines of code and runs fast. It keeps "hasn't" as one token.

It has two weak spots. Typos and unseen words ("recieve", a new brand name) all become <unk>, and averaging the word vectors throws away word order. The dataset paper points out that individual words don't always decide the intent: "reverted top-up" and "failed top-up" share most of their words. The confusion pairs after the first run will show how much this costs. Word bigrams, character trigrams or a small BPE vocabulary are the fixes to try, in that order of effort.

Build the vocabulary from the 9,002 training messages only. Keep tokens that occur at least twice, sort by frequency and reserve id 0 for <unk>. In production the model meets words it has never seen, and a vocabulary that includes validation or test words would make the validation score look better than reality. The minimum count of 2 matters because a word seen once gets 64 weights trained on a single example; it mostly memorises that one message.

Save the vocabulary (vocab.json) and the label list (labels.json) next to the weights. Serving must use exactly the same tokenizer code and vocabulary as training. If a message ends up with no tokens at all ("???"), feed it as a single <unk> so the average is defined.

The layers, one by one
Layer	Shape	Why this choice	Try next
EmbeddingBag, mode="mean"	V × 64	Looks up a learned vector for every token and averages them in one step, so a message of any length becomes one vector of 64 numbers without padding. The mean keeps long and short messages on the same scale.	32 or 128 numbers per word
Linear	64 to 128	The hidden layer. Without it the model can only add up evidence per word. With it, it can combine words: card plus arrived points somewhere else than card plus declined. 128 is wider than the 77 outputs, so it doesn't squeeze them.	256, or a second hidden layer if validation agrees
ReLU		max(0, z). Its slope is 1 for every positive input, so gradients don't fade the way they do on the flat ends of a sigmoid, and it is the cheapest non-linearity to compute.	Rarely worth changing at this size
Dropout	p = 0.3	During training it sets 30% of the hidden values to zero at random, so no single neuron can carry a memorised message. model.eval() switches it off for predictions.	0.5 if training accuracy runs far ahead of validation, 0.1 to 0.2 if both are low
Linear	128 to 77	One score, a logit, per intent.	
There is no softmax layer inside the model. nn.CrossEntropyLoss applies log-softmax to the logits itself, in a numerically stable way. Apply softmax only when you need probabilities, for example for a confidence threshold.

The embedding size is a trade-off. The embedding table is by far the largest block of weights, and it is filled from only 9,002 messages, so making it wider mostly adds room to memorise. 64 numbers are enough to separate a few thousand words, and the sweep will tell you whether 32 or 128 does better.

One hidden layer of 128 follows the dataset paper, which gets its results with a single hidden layer. With averaged word vectors as input there is not much structure left for deeper layers to find. Add depth only when the model is clearly too simple, meaning training and validation accuracy are both low.

Weights and compute, for a vocabulary of V words:

Part	Numbers to learn
Embedding table	V × 64, so 192,000 for 3,000 words
Hidden layer	64 × 128 + 128 = 8,320
Output layer	128 × 77 + 77 = 9,933
Total for 3,000 words	210,253, about 0.8 MB as 32-bit floats
One prediction costs about 18,048 multiply-adds (64 × 128 + 128 × 77) plus the average. That takes microseconds, so the 2-second limit is about everything around the model: the HTTP call, loading, JSON.

Test the starting values of the embedding table early. PyTorch fills an EmbeddingBag with values drawn from N(0, 1). A word seen only a handful of times gets few updates, so its vector stays mostly random and adds noise to every message it appears in. Try starting the table small, for example uniform between −0.1 and 0.1, and keep it if validation loss improves.

A sketch of the model, so Copilot has an exact target:

python
Copy
import torch
from torch import nn


class IntentClassifier(nn.Module):
    def __init__(self, vocab_size: int, n_intents: int = 77,
                 embed_dim: int = 64, hidden: int = 128, dropout: float = 0.3):
        super().__init__()
        self.embed = nn.EmbeddingBag(vocab_size, embed_dim, mode="mean")
        self.hidden = nn.Linear(embed_dim, hidden)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, n_intents)

    def forward(self, token_ids: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        x = self.embed(token_ids, offsets)          # (batch, embed_dim)
        x = self.drop(torch.relu(self.hidden(x)))   # (batch, hidden)
        return self.out(x)                          # (batch, n_intents) logits
token_ids holds the ids of all messages in the batch back to back, and offsets says where each message starts. That is how EmbeddingBag handles messages of different lengths without padding.

Loss: cross-entropy
Use nn.CrossEntropyLoss on the logits. It is the standard loss for picking one class out of many. It takes minus the log of the probability the model gave the right intent, so a confident wrong answer costs a lot and a confident right answer costs almost nothing. Its gradient at the output is simply p − y, which the deck walks through.

It also gives you a free sanity check. An untrained network spreads its probability roughly evenly over the 77 intents, so the first loss you see should be close to ln(77) = 4.34. If validation loss isn't clearly below that after one epoch, something is broken, usually the labels or the learning rate.

Label smoothing is worth one experiment: CrossEntropyLoss(label_smoothing=0.1) tells the model the right answer is 90% certain instead of 100%. With look-alike intents such as card_arrival and card_delivery_estimate, that often lowers validation loss and makes the confidence scores more honest, which matters if you route low-confidence messages to the LLM.

Optimizer: Adam
Use torch.optim.Adam with its defaults: learning rate 0.001, betas (0.9, 0.999), eps 1e-8 and no weight decay.

Adam gives every weight its own step size. It divides each update by a running average of that weight's recent gradient size. In this model that matters a lot: the embedding row for "card" gets a gradient in almost every batch, while the row for "courier" only gets one now and then. With plain SGD the rare rows barely move. With Adam their steps stay in proportion, so rare but telling words still learn.

Adam also keeps momentum. The first beta, 0.9, keeps a running direction, so one odd mini-batch doesn't throw the weights around.

The defaults also work out of the box. In a hackathon afternoon, an optimizer that gives a decent first run without tuning is worth a lot. SGD with momentum can match it, but usually only after a learning-rate search. The dataset paper trained its classifier with plain SGD at a learning rate of 0.7 with linear decay and dropout of 0.75, but its inputs were fixed pretrained vectors, so those settings don't carry over to a model that also learns its own embeddings.

If you add weight decay, use AdamW (or Adam with decoupled_weight_decay=True). Plain Adam folds the decay into the gradient, and its per-weight scaling then partly undoes it. Start at 0.01 and compare on validation.

Learning rate: 0.001
It is Adam's default, from the original Adam paper. With Adam, the learning rate is roughly the size of the step each weight takes per update. Steps of about 0.001 are small next to the weights themselves, so training stays stable, and with 141 updates per epoch the model still gets thousands of updates within the epoch budget.

Check it in ten minutes rather than trusting it. Run one or two epochs each at 0.0003, 0.001, 0.003 and 0.01, and pick the largest value whose validation loss still goes down smoothly. While training, two signals tell you to change it:

the loss jumps around or turns into NaN: too high, divide by 10
the loss has barely moved after 3 epochs: too low, multiply by 3
That is the same rule of thumb as on the gradient-descent slide.

A schedule such as ReduceLROnPlateau, which halves the rate when validation loss stalls, usually squeezes out a little more. Keep the baseline on a constant rate so you can see what each change does.

Batch size, epochs and early stopping
Setting	Value	Why
Batch size	64	141 updates per epoch over 9,002 messages. Small enough that the noise between batches helps the model generalise, big enough for fast matrix maths on a CPU.
Shuffle	every epoch	Data in file order can put one intent in a whole batch, and training then swings from intent to intent.
Epochs	up to 30	4,230 updates at most. An epoch of this model takes seconds on a CPU, so the budget is generous.
Early stopping	patience 3, on validation loss	Stop when validation loss hasn't improved for 3 epochs and go back to the best weights. Validation loss starts rising as soon as the model memorises, usually well before epoch 30.
Reproducibility
Fix the seeds for Python, NumPy and PyTorch, and log them with the package versions and the git commit of every run. Small models move a little between seeds, so before calling one setting better than another, compare the average of three seeds.

What to measure
Log these per epoch with MLflow:

training and validation loss
validation accuracy, which drives the score
macro-F1 on validation, which shows whether a few intents are being ignored
the five most confused intent pairs
the latency of a single message, p50 and p95, measured the way the scorer will call it
the estimated cost per 1,000 messages on validation: running cost + €0.10 × misroutes
The sweep and the leaderboard should use the same cost function, so keep it in one place in the code.

Running it on Azure
Piece	Choice	Why
Workspace	One Azure Machine Learning workspace, one experiment per team	Shared data and environment; each team's runs stay together
Compute	CPU cluster with a minimum of 0 nodes, small general-purpose size	You pay only while jobs run. The dataset paper saw no gain from a GPU when training a one-hidden-layer classifier on the full data, because the overhead outweighed the speed-up.
Data	Register train.csv as a data asset; keep test.csv with the scorer	Jobs get a versioned input, and the test set stays out of training jobs
Environment	Python 3.10 or later, PyTorch CPU build, pandas, scikit-learn, mlflow, azureml-mlflow	The same environment on the laptop, in the job and behind the endpoint
Training	Command job that runs the same train.py as the laptop, with every setting as a command-line argument	What works locally works in the cloud, and the sweep can vary the arguments
Tracking	MLflow, built into the workspace	Loss curves and metrics of all runs side by side
Tuning	Sweep job with random sampling and a Bandit early-termination policy	A cheap search over the few settings that matter, and weak trials stop early
Registry	Register the best run's weights, vocab.json and labels.json as one model	The endpoint serves exactly what was validated
Serving	Managed online endpoint with auth_mode: aad_token on one small CPU instance	Callers sign in with Microsoft Entra ID instead of using keys, and scoring rights go through Azure RBAC
A sweep that fits in an afternoon:

learning rate 0.0003, 0.001 or 0.003
embedding size 32, 64 or 128
hidden size 64, 128 or 256
dropout 0.2, 0.3 or 0.5
batch size 32, 64 or 128
primary metric val_cost_per_1000, goal minimise. The training script has to log the metric under exactly that name.
at most 20 trials, 4 at a time
BanditPolicy(slack_factor=0.1, evaluation_interval=1, delay_evaluation=5). Every time the script logs the primary metric counts as one interval, so with one log per epoch the policy starts judging trials after epoch 5.
For the endpoint, score.py loads the weights, vocabulary and labels once in init(), not per request. run() takes {"text": "..."} and returns the intent and its confidence, with the model in eval() mode and inside torch.no_grad(). With Entra ID authentication, callers need a role that allows scoring; the built-in AzureML Data Scientist role does. An online deployment costs money for every hour it exists, busy or idle, so delete it after the event.

The LLM opponent
Block 2 puts each team's network against an LLM they deploy in Foundry. To make that a fair fight:

Send the 77 intent names, short instructions and the message, and ask for exactly one label.
Use structured outputs where the model supports them: a JSON schema whose label field is an enum of the 77 intents, so every answer is a valid label. Count anything else, and any timeout, as a misroute.
Set temperature to 0 where the model allows it, so the same message gets the same answer.
Count the tokens. At 4 characters per token, the 77 names are about 412 tokens and an average test message about 14, so 1,000 messages send roughly 1,000 × (instructions + 426) input tokens. The schema with its 77 values counts as input too. Measure the exact number with tiktoken.
Few-shot examples usually raise accuracy, but they are paid on every call. The score shows whether they pay for themselves.
The hybrid is worth testing: let the network answer when its top probability is above a threshold and send the rest to the LLM. Pick the threshold on validation. An LLM call is worth it when the misroutes it saves, at €0.10 each, outweigh its token cost.

Improvements to try
Change one thing at a time and compare the three-seed average on validation. Roughly in order of effort:

Word bigrams as extra tokens ("not arrived", "top up"), which bring back some word order cheaply.
A class-weighted loss, if the training split turns out to be uneven.
Label smoothing of 0.1.
A smaller starting range for the embedding table.
AdamW with weight decay 0.01.
Character trigrams, for typos.
A wider hidden layer or a second one.
Three models with different seeds, averaging their logits.
A small 1D convolution or GRU over the word vectors, if the confused pairs show that word order matters. More code and slower, so only after the cheap options.
Routing low-confidence messages to the LLM.
Suggested repository layout
text
Copy
hack-and-slice-01/
  README.md
  pyproject.toml
  docs/
    approach.md              # this document
  data/                      # downloaded, not committed
  src/hns/
    data.py                  # download, load, stratified split
    text.py                  # tokenizer and vocabulary
    model.py                 # IntentClassifier
    train.py                 # training loop, early stopping, MLflow logging
    evaluate.py              # accuracy, macro-F1, confused pairs, cost per 1,000
    predict.py               # load the artefacts, classify one message
  azure/
    environment.yml
    compute.yml
    train-job.yml
    sweep-job.yml
    endpoint.yml
    deployment.yml
    score.py                 # init() and run() for the online endpoint
  llm/
    classify_llm.py          # Foundry call with structured output
    count_tokens.py
  scorer/
    run_scorer.py            # one message per request, timed, cost per 1,000
  tests/
    test_text.py
    test_model.py
Build order for the Copilot session
Data: download, stratified split, print the label counts. Test that the split is 9,002 and 1,001 and that every intent appears in both.
Tokenizer and vocabulary. Test that "My card hasn't arrived." becomes my, card, hasn't, arrived, and that an unknown word maps to 0.
Model. Test that a batch of 3 messages gives logits of shape (3, 77).
Training loop on the laptop. The first loss should be near 4.34 and validation loss should fall after the first epoch.
Evaluation: accuracy, macro-F1, confused pairs and cost per 1,000.
Prediction for one message, with its latency.
Azure: environment, compute, command job and MLflow.
The sweep.
Register, deploy and call the endpoint with Entra ID.
The LLM classifier and the token count.
The scorer, which treats both the same way.
Keep this document in the repository as docs/approach.md and point Copilot at it for each milestone, one at a time. That way the generated code follows these choices instead of generic defaults.

Decisions for you
Are pretrained word vectors or sentence encoders allowed in Block 1? This document assumes not, because the point is a network people train themselves. Allowing them changes the challenge.
How is the network's running cost counted: the instance price times the time the scorer needs, or a fixed share of the hourly price?
One shared workspace with an experiment per team, or a workspace per team?
Does every team deploy its own endpoint, or does the scorer run the models in one place?
Is the hybrid of network plus LLM allowed, and is it scored as one system?
Which Foundry models can teams pick, and whose budget pays for the tokens?
Sources
Casanueva et al. (2020), "Efficient Intent Detection with Dual Sentence Encoders", NLP4ConvAI: https://aclanthology.org/2020.nlp4convai-1.5/ (results in Table 3, classifier setup in section 4)
Banking77 data: https://github.com/PolyAI-LDN/task-specific-datasets and https://huggingface.co/datasets/PolyAI/banking77
Label distribution (third-party analysis): https://github.com/Sarvagya4/Banking77/blob/main/Learnings/Banking77_Dataset.md
TF-IDF and logistic regression baseline (third-party): https://github.com/RikiSTM/banking77-intent-router
PyTorch Adam: https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
Azure Machine Learning, authenticating clients for online endpoints: https://learn.microsoft.com/en-us/azure/machine-learning/how-to-authenticate-online-endpoint?view=azureml-api-2
Azure Machine Learning, hyperparameter tuning: https://learn.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters?view=azureml-api-2
3Blue1Brown, "But what is a neural network?": https://www.youtube.com/watch?v=aircAruvnKk
