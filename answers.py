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
    start_seq = "ACT I."
    temperature = 0.4
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
    hypers['h_dim'] = 32
    hypers['z_dim'] = 10
    hypers['x_sigma2'] = 0.005
    hypers['learn_rate'] = 0.002
    hypers['betas'] = (0.5, 0.999)
    # ========================
    return hypers


part2_q1 = r"""
**Your answer:
The $sigma^2$ parameter is the standard deviation of $P(X|Z)$ meaning it allows us to control the relative influence of
the data loss. When using a small STDV we force our model to generate samples that are similar to the original data set
because the relative contribution of the data loss is high. When using a high STDV we allow a wider range of images to 
be generated meaning that we generate new samples but still with the same characteristics as the original data set.**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q2 = r"""
**Your answer:
1. The reconstruction loss is the distance between the original image and the generated image (point wise loss), thus
it controls the probability that the model will generate an image that is similar to the original data set.
The KL divergence loss is the distance between the posterior distribution and the prior distribution, meaning this loss
part ensures that the encoder's approximation of the latent space (i.e the posterior distribution) is
normally distributed (like the prior distribution).

2. By minimizing the VAE loss term and by extent the KL divergence loss, we try to get the posterior distribution as 
close as possible to the prior distribution. Meaning that we work to change the encoders approximation of the latent 
space distribution to be normally distributed.

3. As we saw in the lecture choosing $z$ from the latent space based on the model parameters is not possible. So in
order to sample $z$ we need the latent space to be normally distributed using the reparametrization trick.
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
    hypers['batch_size'] = 32
    hypers['z_dim'] = 128
    hypers['data_label'] = 1
    hypers['label_noise'] = 0.2
    hypers['discriminator_optimizer']['lr'] = 0.0002
    hypers['discriminator_optimizer']['type'] = 'Adam'
    hypers['discriminator_optimizer']['weight_decay'] = 0.02
    hypers['discriminator_optimizer']['betas'] = (0.5, 0.999)
    hypers['generator_optimizer']['lr'] = 0.0002
    hypers['generator_optimizer']['type'] = 'Adam'
    hypers['generator_optimizer']['weight_decay'] = 0.02
    hypers['generator_optimizer']['betas'] = (0.5, 0.999)
    # ========================
    return hypers


part3_q1 = r"""
**Your answer:
During the training of the GAN model we want to maintain the gradient only when we train the generator, and not when we
train the discriminator. This is because when we train the generator we want it to improve based on the samples it
generates and it's gradients. On the other hand the gradients of the samples don't affect the training of the 
discriminator thus we prefer to discard the gradinets for this part of the training process and as a result improve
computation time of the training. So when we sample for the generator we sample with autograd as opposed to when we 
sample for the discriminator.**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:
1. No, we should not stop training solely base on the Generator loss being below a certain threshold. In the training 
the generator and discriminator losses affect each other, for example if we have a bad discriminator the generator will
be able deceive the discriminator even if the generator is bad himself. In this case the generator loss will be low, we
would stop the training but the generated images would not be accurate enough.

2. If the generator loss is decreasing it means it is getting better at producing generated images close to the real 
data set. Although the discriminator loss remains at a constant value, it is also improving because it is still able
to produce the same results with better generated images.  
**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
Compare the results you got when generating images with the VAE to the GAN results. What's the main difference and 
what's causing it?
**Your answer:
We can see from the results of our two models that the VAE generates images that are blurry on the edges but the face in
the middle of the image is clear and the overall images is smooth. On the other hand the GAN generates images that have 
a more defined background, but the overall image is less smooth and has more contrast between adjacent pixels. 
For example the GAN model produces images that show the US flag like in the background as opposed to the VAE model that 
produces images with grayscale background only.

This difference can be explained through the nature of the loss function of each model. The VAE aimes to minimize the 
loss by making the generated image after the "dimension reduction" similar to the original image. This is done by trying
to preserve the features of the main object in the image. On the other hand the GAN loss function expresses the process 
of the generator trying to deceive the discriminator by producing an image similar to a real image. 

**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============


