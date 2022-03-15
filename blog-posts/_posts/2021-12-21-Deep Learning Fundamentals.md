---
published: True
layout: post
title: Deep Learning Fundamentals
description: >
  A brief introduction to basics of deep learning. 
image:  
  https://user-images.githubusercontent.com/37147511/146909130-c82172dd-1a6e-4d6f-b1f8-f50ff7ff9d68.jpg
canonical_url: http://Deep Learning Fundamentals
hide_image: False
accent_color: '#4fb1ba'
accent_image:
  background: 'linear-gradient(to bottom,#193747 0%,#233e4c 30%,#3c929e 50%,#d5d5d4 70%,#cdccc8 100%)'
  overlay:    true

---

Today I am going to talk about the fundamentals of deep learning. 
Let's start with the basics of neural networks. neural networks are called
neural because they are biologically inspired by `neurons` which do all the
computing in our bodies and the kind of mental model of a neuron is that it's a cell that has things coming out of the main part called `dendrites`
and we can think of them as like `receptors of information`. If enough stimulation has been received by the dendrites then the whole neuron does a thing
called `firing` which is basically an electrical impulse
that begins in the cell and then propagates down a long branch called the `axon`
The axon terminates in little branches that are basically adjoining other neurons dendrites.

![biological_neuron](https://user-images.githubusercontent.com/37147511/146909499-44777620-3bd1-4afa-9ad7-f068c7f95c34.jpg)

So it's like a network of these neurons getting stimulated. If they get
stimulated enough they fire and the electrical potential travels
down the long branch and stimulates other neurons in turn

![artificial neuron](https://user-images.githubusercontent.com/37147511/146909130-c82172dd-1a6e-4d6f-b1f8-f50ff7ff9d68.jpg)

Mathematically we can think about this
as a pretty simple function called `perceptron`. It's a pretty
old concept in computing but we can think of the stimulation arriving at the dendrites as basically inputs (x sub not x sub one x sub two and so on) .How much it gets stimulated by that input is determined by
a weight (w sub zero, w sub one, w sub two and so on) and if we sum all
that, we have that sum over i of w sub i x times x sub i and that's really like the neuron getting stimulated by the input. Then there's b which is just the
bias because this is a linear function
we kind of want; a little offset for the y-intercept basically. Then the whole thing is enclosed in some kind of activation function
because the way that a neuron works is it's either fully on or fully off.
If it's stimulated enough it fires and if it's not stimulated enough then it doesn't fire. So An activation function that basically
is a threshold function like if the sum exceeds the threshold
it passes it on otherwise it remains off 

![activation functions](https://user-images.githubusercontent.com/37147511/146917611-72db6967-75e1-40eb-8d32-7f980b79e2e8.jpg)

There are some good activation functions. Classical neural network literature
really used the `sigmoid function` which is a simple function that kind of
squashes everything into no matter what the input is,
the output is going to be between 0 and 1. It kind of
asymptotes at 0 for negative inputs and 1 for positive inputs, and then in between around zero it quickly changes from zero to one.
So that's what we want to see in an activation function. It has a nice
derivative which can also be called the gradient, g prime. But in recent times
people have mostly used the activation function called the
`rectified linear unit`
or also known as a `max function` because it's literally just saying
whatever input comes in if it's above zero then pass it on, it's 1 
and then if it's not above zero or less than zero then don't pass it on, it's zero. The gradient for it has a discontinuity
but that's fine so basically the gradient is one if the input was greater
than zero or it's zero otherwise. and this is part of the innovation actually that really kicked off the deep
learning revolution in 2013, the relu 

![neural network](https://user-images.githubusercontent.com/37147511/146922924-41484698-64c6-4eae-88f7-920547d293da.jpg)

We talked about neurons
individually, we call them perceptrons.
But what makes a neural network? So if you
arrange perceptrons in layers like we do here that's where the the
terminology of networks comes from. Usually there's an input layer
that's whatever our input data. Then
the input layer connects to
a hidden layer. That hidden layer
connects to another hidden layer. There's some number of those and then
finally there's an output layer. Now each one of these perceptrons make up this network which has its own weights and biases. The setting of these
weights and biases determines how the neural network responds to input

This neural network represents
some function y.Here y equals f of x (the input) and w (the setting of all the weights). But what can that function be?

![functions neural net represent](https://user-images.githubusercontent.com/37147511/146925564-6bc47d00-85df-485d-a027-c66e67a2a9f3.jpg) 

Let's look at this function on the left, f of x. Lots of peaks and valleys in
here. Basically this function represents how can we know if there's a neural
network and a choice of weights for it if we have a neural network. We can get more idea about this by looking at universal approximations of functions.

![universal approximation function](https://user-images.githubusercontent.com/37147511/146929542-769f2899-eb75-47c5-b380-08ca7ffd28ba.jpg)

For detailed understanding of this function, explore the idea interactively at [here](http://neuralnetworksanddeeplearning.com/chap4.html).
Neural networks are incredibly general
and theoretically we can represent any function as a neural network.

Now a question pops up that is, what do we use neural networks for? Well, we do for machine learning problems? then another question comes up, what is machine learning problems are? There's three kind of big. A breakdown of all the machine learnings out there have three categories:
- *Supervised learning*
- *Unsupervised learning*
- *Reinforcement learning*

Though there's also `transfer learning, meta learning, imitation learning`  etc., but these are the three big categories.

![types of learnings](https://user-images.githubusercontent.com/37147511/146938651-19182eb6-b1da-401e-8d04-8107052ae214.jpg)

For `unsupervised learning`, we get unlabeled data, X that means X
can be maybe sound clips text strings or images but there's no nothing else
associated with them. It's just the sound clips or the text strings or images. the goal is really to learn the structure of that data so we can
learn X. The reason we want to do it because
we can generate more of these types of data. We can
generate fake sound clips images or reviews
but we can also obtain insights into
what the data might hold. Here in the picture(left) above we see two examples 
1. Here's some fake text, an
amazon review that we can just generate using a neural net. 
2. Here is the concept of clustering. So we have some data.
We don't know anything about it. We don't have labels for it but just
because of how it's structured, we might infer that there
are clusters. Some process gave rise to this data such that
the data in the left here came from one process and the data in right here came from another process

For `supervised learning`, we get both X
and Y. X is the raw input data
and Y is a label for it. We could have an image and a label. 
A label could be **_does it have a cat in it?_** and
The goal is to learn a function that goes from X so an
image to Y a label for it and the goal for that is just to be able
to make predictions. Here in the picture(center) above we see two examples of it
1. If We get an image we can say it's a
cat 
2. If we get a sound clip we might
be able to understand that it's a person speaking the words _hey siri_

In `reinforcement learning`, the goal is to learn **_how to take actions in an environment_**. So there's some kind of agent, maybe a robot, maybe a computer virus or something like that. It can take actions like it can maybe move
forward or it can look somewhere and
because when it takes an action you know reality
provides some kind of input back to it because it's acting in an environment. We can interpret that environment as basically providing a reward or not
to the agent and then changing the state that the agent is in. if a robot
was in a place and then it took a forward move in action. Now it's in another place and maybe there's a reward associated with it or maybe not. Here in the picture(right) above we see two examples of it 
1. If a robot was in a place and then it took a forward move in action. Now it's in another place and maybe there's a reward associated with it or maybe not.
2. We can train game playing agents using reinforcement learning. Here it's the action is place down a piece on the go board and then the reward can be like did we eventually win the game or not and the state is obviously just the
state of the of the go board
