# Shared concepts and slide flow

This reference describes the concepts and order for the opening presentation. It is not the slides themselves. Participants know Python, Git, and the command line, but do not need prior machine-learning experience. Aim for about 20–30 minutes, using small numerical examples and intuition rather than calculus or derivations.

Start with handwritten digits, then use that mental model to introduce the Banking77 classifier. This is one example of an AI architecture, not a description of every AI system. The guided exercise and its explanations should stand on their own; they can use the concepts below without depending on specific slides or their order.

## 1. Digit classification: architecture

- **Input:** represent an image as numerical pixel values. A network transforms these values into scores for the ten possible digits, 0 through 9.
- **Layer and weights:** a linear layer combines inputs using learned weights and biases. For example, inputs `2` and `-1`, weights `0.5` and `1`, and bias `0.25` produce the weighted sum `0.25`.
- **ReLU:** a nonlinearity that replaces negative values with zero and leaves positive values unchanged. It allows a network of layers to represent more than a single linear transformation.
- **Dropout:** randomly omit some activations during training to reduce reliance on particular activations. It is disabled during evaluation and prediction.
- **Logits and prediction:** the output layer produces one score, or logit, per digit. The highest-scoring digit is the prediction. Logits are not themselves probabilities.

Use the small example to make the arithmetic concrete, not to calculate a full image by hand.

## 2. Digit classification: training and evaluation

- **Labeled examples:** each training image has a correct digit. A batch groups examples for one update; an epoch is one pass through the training examples.
- **Softmax and loss:** softmax converts logits into probabilities. Cross-entropy measures how much probability the model assigns to the correct label: less probability means greater loss. For two equal logits, the correct label has probability `0.5` and loss `-ln(0.5)`, about `0.69`.
- **Gradients and updates:** gradients indicate how changing each learned parameter affects the loss. Automatic differentiation computes them, and an optimizer updates the parameters to reduce the loss. Explain the direction of improvement without deriving backpropagation.
- **Adam and learning rate:** Adam is the optimizer used in the exercise; the learning rate controls the scale of its updates. A learning rate that is too high or too low can hinder learning. Do not derive Adam's formula.
- **Train, validation, and test:** training data is used for parameter updates; validation data guides choices during development; the held-out test set is reserved for final evaluation. Validation accuracy can guide early stopping and comparison of learning rates. Avoid using the test set to choose a model.

PyTorch cross-entropy accepts logits directly; it handles the relevant log-softmax calculation internally. A separate softmax is unnecessary when selecting the highest-scoring label.

## 3. From digit images to message intents

The same broad classifier idea applies, but the input and outputs change:

1. **Tokenization and vocabulary:** split a message into tokens, then map them to numeric IDs from a vocabulary built only from the training split. Unknown tokens use an unknown-token ID. An ID identifies a token; its numeric value is not a measure of meaning.
2. **Learned embedding:** look up a vector of learned numbers for each token. The exercise learns these vectors from scratch rather than using a pretrained encoder.
3. **Mean pooling:** average the token vectors to obtain a fixed-size message vector. For example, averaging `[1, 0]` and `[0, 2]` gives `[0.5, 1]`. Averaging loses word order.
4. **Classifier:** feed the 64-dimensional mean embedding through a 128-unit linear layer, ReLU, dropout, and a final linear layer producing 77 intent logits. The sizes are settings of this exercise, not numbers participants need to memorize.
5. **Prediction:** choose the highest-scoring official intent label for the message.

Pretrained embedding models are an optional next step, not part of the guided exercise.

## 4. Training the text classifier

Messages and intent labels replace images and digit labels. Vocabulary construction uses the training split only; unseen tokens are mapped to the unknown-token ID. The embeddings and classifier weights are learned together through the same loss, gradient, and optimizer concepts introduced for digits. PyTorch provides the layers, automatic differentiation, and Adam; participants do not implement backpropagation themselves.

The exercise compares three learning rates using validation accuracy and uses early stopping. The official held-out test data is for final evaluation, not training or tuning. A demonstration message that shows the trained classifier labeling text is distinct from that test dataset.

## Concepts for challenge explanations

Each challenge's "Why this step?" should explain its purpose in its own words, using consistent language rather than linking to or requiring the presentation:

| Activity | Relevant concepts |
|---|---|
| Prepare data | Labeled messages, train/validation/test separation, training-only vocabulary, and unknown tokens |
| Build the model | Learned embeddings, mean pooling, linear layers, ReLU, dropout, and intent logits |
| Train the model | Batches, cross-entropy, gradients, optimizer updates, epochs, and training-only dropout |
| Compare and evaluate | Learning rate, validation accuracy, early stopping, and the held-out test set |
| Predict an intent | Consistent tokenization and vocabulary, logits, and the highest-scoring label |

These are conceptual correspondences, not a prescribed number or order of challenges. Project-specific terms are defined in the [glossary](../../GLOSSARY.md).
