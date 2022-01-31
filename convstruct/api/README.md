<br>
<br>
<br>
<br>

<div align="center">

  # The API
  #### Randomizes, Graphs, and Optimizes Topologies

</div>

<br>
<br>
<br>
<br>

---

<br>

### Convstruct 1.1 is here :zap: :zap:

1.1 introduces a new section, API, that makes up the core of import Convstruct. Also introduced is a new way to add custom ops.
[View release notes](https://github.com/convstruct/convstruct/releases/tag/1.1.0)

<br>

---



<img align="right" width="250" height="250" src="https://i.ibb.co/R31qNBS/api-icon.png">

<br>

The API’s purpose is to provide functionality to the main functions that can be used when importing Convstruct. The breakdown of this functionality is three sections; ***cs.core***, ***cs.ops*** and ***cs.util***. To make contributing to Convstruct as simple as possible, cs.ops can be expanded with custom loss functions and optimizers without the affecting the main functions.

<br>
<br>

>The main function cs.createEpisode is provided functionality from Evaluator not API.

<br>

---

<br>

cs.core contains the functionality to get the topologies, by randomizing parameters and setting them into topologies, and the functionality to take those topologies and set them into graphs.

[<img align="center" src="https://i.ibb.co/RPy7yJM/api-banner-1.png" alt="api-banner-1" border="0">](https://github.com/convstruct/convstruct/tree/main/convstruct/evaluator)

<br>

---

cs.ops contains a variety of loss functions and optimizers to be used with topologies. The main function, cs.createOps, requires one of the loss functions and one of the optimizers within cs.ops to create the ops to be fed to tf.session.run.

[<img align="center" src="https://i.ibb.co/8bPp6Z5/api-banner-2.png" alt="api-banner-2" border="0">](https://github.com/convstruct/convstruct/blob/main/convstruct/api/__init__.py)

<br>
<br>

>- createOps Loss function accepted arguments: loss_feed, graph, and gpu.<br>
>- createOps Optimizer function accepted arguments: returned loss from loss function and name.

<br>

---

<br>


cs.util contains several optional functions that can be used to create your Tensorflow models. These functions are optional as they are not essential to running Convstruct. Their purpose is to provide common Tensorflow functions used alongside Convstruct.

<br>

---

<br>

For inquiries into the API contact: [hello@convstruct.org](hello@convstruct.org)

<br>
<br>
