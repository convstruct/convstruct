<br>
<br>
<br>
<br>

<div align="center">

  # The Evaluator
  #### Evaluates, Scores, and Recommends Topologies

</div>

<br>
<br>
<br>
<br>

---

<br>

### Convstruct 1.1 is here :zap: :zap:

1.1 highlights the Evaluator by separating it from the rest of the Convstruct codebase giving it an individual section. </br>
[View full release notes](https://convstruct.org)

<br>

---



<img align="right" width="250" height="250" src="https://i.ibb.co/7XNfsmL/evaluator-icon.png">

<br>
<br>

The Evaluator’s purpose is to learn to search and train a topology in a set amount of time. To do this, the evaluator trains during episodes, using reinforcement learning to choose from a set of actions at checkpoints during the training of topologies. Each action taken leads the evaluator to converge on the best model, using the ground-truth data as a benchmark.

<br>
<br>

---

<br>

The evaluator is a part of the main Convstruct function cs.createEpisode. The function takes the macro architecture fed to the function and returns an episode loop.

[<img align="center" src="https://i.ibb.co/9tPMDNw/eval-1.png" alt="evaluator-banner-1" border="0">](https://arxiv.org/abs/1807.06653)

<br>

---

During the episode loop, after an epoch of training occurs an evaluation of the current topology output compared to the ground truth data takes place. This is built into the cs.Evaluator.getScore function.

<img align="center" src="https://i.ibb.co/znfYVkm/eval-2.png" alt="evaluator-banner-2" border="0">

Along with the evaluation between output and data, a secondary evaluation is done on weights of each layer compared to its previous values. If a topology is bad, a layer will show up as having basically no change between current and previous weight values (scaled against other layers).

<br>
<br>

---

<br>


In the episode loop, the cs.Evaluator.getAction function returns a random chosen action at first, but as training continues, it switches to the Evaluator’s chosen action, using an MLP model.

[<img align="center" src="https://i.ibb.co/ZmQPPd0/eval-2.png" alt="evaluator-banner-3" border="0">](https://arxiv.org/pdf/1701.04968.pdf)


> The rewards provided at every checkpoint are broken down as followed: <br>
+/- Points for having a lower/higher loss than the previous checkpoint <br>
+/- Points for having a higher/lower accuracy than the previous checkpoint <br>
+/- Points for each layer with no* change in weight value between epochs

<br>

---

<br>

For inquiries into the evaluator contact: [hello@convstruct.org](hello@convstruct.org)

<br>
<br>
