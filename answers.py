r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

def part1_rnn_hyperparams():
    hypers = dict(
        batch_size=0, seq_len=0,
        h_dim=0, n_layers=0, dropout=0,
        learn_rate=0.0, lr_sched_factor=0.0, lr_sched_patience=0,
    )
    # TODO: Set the hyperparameters to train the model.
    # ====== YOUR CODE: ======
    hypers['batch_size'] = 150
    hypers['seq_len'] = 64
    hypers['h_dim'] = 256
    hypers['n_layers'] = 3
    hypers['dropout'] = 0.4
    hypers['learn_rate'] = 0.001
    hypers['lr_sched_factor'] = 0.5
    hypers['lr_sched_patience'] = 3
    # ========================
    return hypers


def part1_generation_params():
    start_seq = ""
    temperature = .0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return start_seq, temperature


part1_q1 = r"""
**Your answer:
We split the corpus into sequences instead of training on the whole text for a couple of reasons. Training on the whole
corpus could take a lot of memory, so much that it wouldn't fit the memory limits. Also, training on the whole corpus
would make the forward pass and backward pass go through the entire text every time, making it very long, 
which might make are model not converse. 
Since we save the hidden states after every pass, we can split the text to smaller sequences and still take into account
the previous sequences.
**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part1_q2 = r"""
**Your answer:
As explained in the previous question, we keep the history of the previous sequences in the hidden states and therefore 
we can generate text based on previous sequences as well.
**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part1_q3 = r"""
**Your answer:
As we explained before, by keeping the hidden states we don't need to train on the whole text, but we do need to keep
the order of the batches for the hidden states to be relevant. That's why it is important that following sequences are
in the same placement in following batches in order to maintain the logical order of the text.
**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part1_q4 = r"""
Why do we lower the temperature for sampling (compared to the default of  1.0  when training)?
What happens when the temperature is very high and why?
What happens when the temperature is very low and why?
**Your answer:
1. When sampling, we want to generate the correct characters that have the highest confident prediction. Lowering the 
temperature we enlarge the differences between the different values, making the higher predictions even larger and the
lower even smaller, thus we are less likely to make mistakes.
When training, we want to enable our model to change it's predictions so we set the temperature to 1.0 in order to have 
more diversity in our results by using softer distributions.
2. When the temperature is high, softmax will flatten the distribution as we can see in the graph we produced earlier 
in part 1. This might cause the model to generate random characters and make more mistakes.
3. When the temperature is very low, we will sample from less "candidates" making our sampling more conservative. 
Softmax will enhance the next character prediction and we will most likely get the same sentence over and over again.
**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""
# ==============


# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = None


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=0,
        h_dim=0, z_dim=0, x_sigma2=0,
        learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    hypers['batch_size'] = 64
    hypers['h_dim'] = 512
    hypers['z_dim'] = 4
    hypers['x_sigma2'] = 0.5
    hypers['learn_rate'] = 0.001
    hypers['betas'] = (0.5, 0.999)
    # ========================
    return hypers


part2_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============

# ==============
# Part 3 answers

PART3_CUSTOM_DATA_URL = None


def part3_gan_hyperparams():
    hypers = dict(
        batch_size=0, z_dim=0,
        data_label=0, label_noise=0.0,
        discriminator_optimizer=dict(
            type='',  # Any name in nn.optim like SGD, Adam
            lr=0.0,
            # You an add extra args for the optimizer here
        ),
        generator_optimizer=dict(
            type='',  # Any name in nn.optim like SGD, Adam
            lr=0.0,
            # You an add extra args for the optimizer here
        ),
    )
    # TODO: Tweak the hyperparameters to train your GAN.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return hypers


part3_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============


