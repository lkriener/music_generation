# Music generation with artificial neural networks

## Group project for *Advanced topics in machine learning* lecture (2019)
#### Benjamin Ellenberger, Nicolas Deperrois, Laura Kriener

### Project description and motivation
The topics for the group project work was chosen in the very beginning of the course.
As it was announced that the course will nearly always use images for demonstrations, examples and excercises, we decided that our project should be on a different form of data.
We decided to work in the broad field of music generation with deep learing.

To narrow down the topic we investigated what the state of the art in this field is.
The most impressive recent results are produced by Google and OpenAI.
The [Google Magenta project](https://magenta.tensorflow.org/) covers a wide range of applications such as harmonization, drum-machines and a music generating network using the transformer network architecture with attention.

An other very recent result in the field of generating music was published by OpenAI. The [MuseNet](https://openai.com/blog/musenet/) uses the recently published GPT2-architecture which is a large-scale transformer network as well. 

The Google and OpenAI approaches as well as other (less famous) approaches have in common, that they employ very complicated network architectures in combination with the use of immense computational resources.

As the required computational power is far out of our reach, we wondered if this level of complexity is really unavoidable.
And so the question **How much can you do with how little?** became the leading theme for our project.
We want to see, what results can be achieved using much simpler network architecutres (i.e. architectures within the scope of the lecture)?
Which aspects of music generation can be achieved and which have to be ignored? For example can you generate a resonable melody line without considering the rhythm?

### **Report**
A detailed description of our approach, the used data sets, achieved results and exemplary usage of our networks can be found in our [Report](Report.ipynb).


### **Presentation**
The final project presentation given in the lecture can be found [here](project_presentation.odp).
