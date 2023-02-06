# NLP702 Deep Learning for Natural Language Processing

* Instructor: Dr. Muhammad Abdul-Mageed

## 1. Course Rationale & Goal
* Catalog Description: This course provides a methodological and an in-depth background on key core Natural Language Processing areas based on deep learning. It builds upon fundamental concepts in Natural Language Processing and assumes familiarization with mathematical and machine learning concepts and programming. 
* Goal: This graduate-level course aims to instill a deeper and thorough understanding of advanced Natural Language Processing methods based on deep learning, to equip students with capabilities of researching, developing, and implementing these methods.
* The course covers the following key core areas: (I) Foundation Models, (II) Representation Learning for NLP, (III) Machine Translation (IV) Multilinguality and Low-Resource NLP, (V) Speech-Language Interface, and (VI) Vision-Language Interface

## 2. Recommended Textbooks
* This advanced course will use research papers. Students may find the following textbooks relevant:
  
  (1) Tunstall, L., von Werra, L., & Wolf, T. (2022). Natural Language Processing with Transformers. O'Reilly Media, Inc., ISBN 978-1098136796.
  
  (2) Ian Goodfellow, Yoshua Bengio, and Aaron Courville. Deep Learning. 2016, MIT Press. ISBN: 9780262035613.
  

* Relevant research papers, technical reports, and surveys for each topic, where needed, will be provided to students. In addition, the following textbooks may be useful:

   (1) Chris Manning et al, Foundation of statistical natural language processing, 1999, MIT Press, ISBN: 0262133601.
   
   (2) Dan Jurafsky and James H. Martin, Speech and Language Processing (3rd edition, draft) https://web.stanford.edu/~jurafsky/slp3/
   
   (3) Aurélien Géron. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems (2nd Edition). 2019, O'Reilly Media, ISBN 9781492032649
   
   (4)François Chollet. Deep Learning with Python. 2017,  Manning Publications Co. , ISBN 9781617294433.
   
   (5)Yue Zhang, Zhiyang Teng. Natural Language Processing: A Machine Learning Perspective, 2021, Cambridge University Press, ISBN 9781108420211.
   
   (6) Jacob Eisenstein. Introduction to Natural Language Processing. 2019, MIT Press, ISBN 9780262042840.

## 3. Course syllabus
| Teaching Week | Topic | Leture | Lab |
| ----  | ------ | ------- | ------- |
| 1 | Course Overview & Refresher on Classical Deep Learning Architectures  | Overview of research directions in key core NLP areas; Refresher on classical neural architectures (e.g., CNNs, RNNs, LSTMs, and GRUs) with example applications | [NumPy](tutorial/intro_to_numpy_pytorch/numpy_tutorial.ipynb); [PyTorch](tutorial/intro_to_numpy_pytorch/pytorch_tutorial.ipynb); [Practice text classification with attention-based RNNs](tutorial/text_classification_attention/rnn_attention_tutorial.ipynb) | 
| 2 | Transformer Variations  | Refresher on the Transformer; Long-document Transformers; Wide Transformers; Understanding the role of attention in the Transformer | [Transformer architecture](tutorial/transformer/transformer_tutorial.ipynb); [Longformer](tutorial/text_classification_Longformer/Longformer.ipynb) | 
| 3 | Foundation Models: Encoder-Only Models  | Self-supervised learning; Encoder-only models (e.g., BERT, RoBERTA, SpanBERT, mBERT); Denoising objectives and contrastive objectives in encoder-only models; Anisotropy in the embeddings of the encoder; Evaluating encoder-only models | [Pretraining of Encoder-only model](tutorial/MLM-pretraining); [Inspecting the model embeddings for anisotropy](tutorial/anisotropy_visualization/Anisotropy_Viz.ipynb) | 
| 4 | Foundation Models: Decoder-Only Models  | Generative Pretraining (GPT); OpenAI GPT series (GPT1, GPT2, and GPT3); Public autoregressive models (e.g., GPT-NeoX, OPT, BLOOM); Evaluating GPT models | [Pretraining of GPT2](tutorial/GPT); [Calculating perplexity](tutorial/perplexity/Perplexity_Tutorial.ipynb) |
| 5 | Foundation Models: Encoder-Decoder Models  | Sequence-to-sequence learning; Encoder-decoder models (e.g., T5, BART, mT5, mBART); Evaluating encoder-decoder models | [Programming exercises using T5](tutorial/T5-fine-tuning) | 
