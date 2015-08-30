---
layout: post
title:  "LSTM networks explained"
date:   2015-08-30 20:04:15
---

## Preface
For a long time I've been looking for a good tutorial on implementing LSTM networks.
They seemed to be complicated and I've never done anything with them before.
Quick googling didn't help, as all I've found were some slides.

Fortunately, I took part in Kaggle EEG Competition and thought that it might be fun
to use LSTMs and finally learn how they work. I based my solution and this post's code on [char-rnn](https://github.com/karpathy/char-rnn)
by [Andrej Karpathy](https://karpathy.github.io),
which I highly recommend you to check out.

### RNN misconception
There is one important thing that as I feel
hasn't been emphasized strongly enough (and is the main reason why I couldn't get myself to
do anything with RNNs). There isn't much difference between an RNN and
feedforward network implementation. It's the easiest to implement an RNN just
as a feedforward network with some parts of the input feeding into the middle of the stack,
and a bunch of outputs coming out from there as well. There is no magic internal state
kept in the network. It's provided as a part of the input!

<div class="images">
  <img src="assets/posts/lstm-explained/RNNvsFNN.svg">
  <div class="label">
    The overall structure of RNNs is very similar to that of feedforward networks.
  </div>
</div>

### LSTM refresher
This section is intended only for people who'd like to refresh the theory behind
LSTMs. If you feel comfortable with it, feel free to skip it
([click here](#building-your-own-lstm)).



## Building your own LSTM
The code for this tutorial will be written in Torch7.
**Don't worry if you don't know it**. I'll explain everything, so you'll be able
to implement the same algorithm in your favorite framework.

The network will be implemented as a `nngraph.gModule`, which basically means that we'll define
a computation graph consisting of standard `nn` modules.
We will need the following layers:

* `nn.Identity()` - passes on the input (used as a placeholder for input)
* `nn.Dropout(p)` - standard dropout module (drops with probability `1 - p`)
* `nn.Linear(in, out)` - an affine transform from `in` dimensions to `out` dims
* `nn.Narrow(dim, start, len)` - selects a subvector along `dim` dimension having `len` elements starting from `start` index
* `nn.Sigmoid()` - element-wise sigmoid
* `nn.Tanh()` - element-wise tanh
* `nn.CMulTable(t)` - forwards sum of all tensors in `t`
* `nn.CAddTable(t)` - forwards product of all tensors in `t`

### Inputs

First, let's define the input structure. The array-like objects in lua
are called tables. This network will accept a table of tensors like the one below:


<div class="images">
  <img src="assets/posts/lstm-explained/input_table.svg" alt="Input table structure" style="width: 50%;"/>
</div>

{% highlight lua %}
local inputs = {}
table.insert(inputs, nn.Identity()())   -- network input
for l = 1,L do
  table.insert(inputs, nn.Identity()()) -- c at time t-1 from layer L
  table.insert(inputs, nn.Identity()()) -- h at time t-1 from layer L
end
{% endhighlight %}

Identity modules will just copy whatever we'll provide to the network into the graph.

### Computing gate values

Let's assume that we're constructing layer \\(l\\) and `prev_h` holds
the current layer's output state from the previous time step.
`input` contains a node, which feeds into the layer.

To make our implementation faster we will be applying the transformations for the whole
LSTM layer simultaneously.

{% highlight lua %}
local i2h = nn.Linear(input_size, 4 * rnn_size)(x)    -- input to hidden
local h2h = nn.Linear(rnn_size, 4 * rnn_size)(prev_h) -- hidden to hidden
local preactivations = nn.CAddTable()({i2h, h2h})     -- i2h + h2h
{% endhighlight %}

If you're unfamiliar with `nngraph` it probably seems strange that we're constructing
a module and already calling it once more with a graph node. What actually happens is that the
second call converts the `nn.Module` to `nngraph.gModule` and the argument specifies it's parent in the graph.

`preactivations` module outputs a vector created by a linear transform of input
and previous hidden state. These are raw values which will be used to compute the
gate activations and the cell input. This vector is divided into 4 parts, each
of size `rnn_size`. The first will be used for in gates, second for forget gates,
third for out gates and the last one as a cell input (so the indices of respective gates
and input of a cell number \\(i\\) are
\\(\left\\{i,\ \text{rnn_size}+i,\ 2\cdot\text{rnn_size}+i,\  3\cdot\text{rnn_size}+i\right\\}\\)).

<div class="images">
  <img src="assets/posts/lstm-explained/graph1_full.svg" alt="Input table structure" style="width: 30%;"/>
  <img src="assets/posts/lstm-explained/preactivation_graph.svg" alt="Input table structure" style="width: 40%;"/>
</div>

Next, we have to apply a nonlinearity, but while all the gates use the sigmoid,
we will use a tanh for the input preactivation. Because of this, we will place two `nn.Narow`
modules, which will select appropriate parts of the preactivation vector.

{% highlight lua %}
local pre_sigmoid_chunk = nn.Narrow(2, 1, 3 * rnn_size)(preactivations)
local all_gates = nn.Sigmoid()(pre_sigmoid_chunk)

local in_chunk = nn.Narrow(2, 3 * rnn_size + 1, rnn_size)(preactivations)
local in_transform = nn.Tanh()(in_chunk)
{% endhighlight %}

After the nonlinearities we have to place a couple more `nn.Narrow`s and we have the gates done!

{% highlight lua %}
local in_gate = nn.Narrow(2, 1, rnn_size)(all_gates)
local forget_gate = nn.Narrow(2, rnn_size + 1, rnn_size)(all_gates)
local out_gate = nn.Narrow(2, 2 * rnn_size + 1, rnn_size)(all_gates)
{% endhighlight %}

<div class="images">
  <img src="assets/posts/lstm-explained/graph2_full.svg" alt="Input table structure" style="width: 30%;"/>
  <img src="assets/posts/lstm-explained/gates.svg" alt="Input table structure" style="width: 40%;"/>
</div>

### Cell and hidden state

Having computed the gate values we can now calculate the current cell state. All that's required are just
two `nn.CMulTable` modules (one for \\(f \cdot c_{t-1}^{l}\\) and one for \\(i \cdot x\\)),
and a `nn.CAddTable` to sum them up to a current cell state.

{% highlight lua %}
local c_forget = nn.CMulTable()({forget_gate, prev_c})
local c_input = nn.CMulTable()({in_gate, in_transform})
local next_c = nn.CAddTable()({
  c_forget,
  c_input
})
{% endhighlight %}

It's finally time to implement hidden state calculation. It's the simplest part, because it just
involves applying tanh to current cell state (`nn.Tanh`) and multiplying it with an output gate
(`nn.CMulTable`).

{% highlight lua %}
local c_transform = nn.Tanh()(next_c)
local next_h = nn.CMulTable()({out_gate, c_transform})
{% endhighlight %}

<div class="images">
  <img src="assets/posts/lstm-explained/graph3_full.svg" alt="Input table structure" style="width: 30%;"/>
  <img src="assets/posts/lstm-explained/state_calculation.svg" alt="Input table structure" style="width: 40%;"/>
</div>

### Defining the module

Now, if you want to export the whole graph as a standalone module you can wrap it like that:

{% highlight lua %}
outputs = {}
table.insert(outputs, next_c)
table.insert(outputs, next_h)

return nn.gModule(inputs, outputs)
{% endhighlight %}

## That's it!

That's it. It's quite easy when you understand how to deal with the hidden state.
After connecting several layers just put a regular MLP on top and connect it to last
layer's hidden state and you're done!

Here are some nice papers on RNNs if you're interested:

* [Visualizing and Understanding Recurrent Networks](http://arxiv.org/abs/1506.02078)
* [An Empirical Exploration of Recurrent Network Architectures](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
* [Recurrent Neural Network Regularization](http://arxiv.org/abs/1409.2329)
* [Sequence to Sequence Learning with Neural Networks](http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf)


<script>
  var disqus_identifier = 'lstm-explained';
  var disqus_title = '{{ page.title }}'
  var disqus_url = '{{ page.url | prepend: site.baseurl | prepend: site.url }}'
</script>
