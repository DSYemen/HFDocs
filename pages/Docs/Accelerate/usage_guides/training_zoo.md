# مثال Zoo

فيما يلي قائمة غير شاملة بالبرامج التعليمية والنصوص البرمجية التي تعرض 🤗 Accelerate

## أمثلة Accelerate الرسمية:

### الأمثلة الأساسية:

تعرض هذه الأمثلة الميزات الأساسية لـ Accelerate وهي نقطة انطلاق رائعة.

- [مثال NLP الأساسي](https://github.com/huggingface/accelerate/blob/main/examples/nlp_example.py)
- [مثال NLP الموزع الأساسي في Jupyter Notebook](https://github.com/huggingface/notebooks/blob/main/examples/accelerate_examples/simple_nlp_example.ipynb)
- [مثال رؤية حاسوبية أساسي](https://github.com/huggingface/accelerate/blob/main/examples/cv_example.py)
- [مثال رؤية حاسوبية موزع أساسي في Jupyter Notebook](https://github.com/huggingface/notebooks/blob/main/examples/accelerate_examples/simple_cv_example.ipynb)
- [استخدام Accelerate في Kaggle](https://www.kaggle.com/code/muellerzr/multi-gpu-and-accelerate)

### أمثلة خاصة بالميزات:

تعرض هذه الأمثلة ميزات محددة يقدمها إطار عمل Accelerate.

- [التجميع التلقائي للذاكرة](https://github.com/huggingface/accelerate/blob/main/examples/by_feature/automatic_gradient_accumulation.py)
- [حالات التثبيت](https://github.com/huggingface/accelerate/blob/main/examples/by_feature/checkpointing.py)
- [التحقق من صحة التعابر](https://github.com/huggingface/accelerate/blob/main/examples/by_feature/cross_validation.py)
- [DeepSpeed](https://github.com/huggingface/accelerate/blob/main/examples/by_feature/deepspeed_with_config_support.py)
- [Fully Sharded Data Parallelism](https://github.com/huggingface/accelerate/blob/main/examples/by_feature/fsdp_with_peak_mem_tracking.py)
- [تجميع التدرجات](https://github.com/huggingface/accelerate/blob/main/examples/by_feature/gradient_accumulation.py)
- [محدد حجم الدفعة الواعية بالذاكرة](https://github.com/huggingface/accelerate/blob/main/examples/by_feature/memory.py)
- [حساب القياس](https://github.com/huggingface/accelerate/blob/main/examples/by_feature/multi_process_metrics.py)
- [استخدام أجهزة التتبع](https://github.com/huggingface/accelerate/blob/main/examples/by_feature/tracking.py)
- [استخدام Megatron-LM](https://github.com/huggingface/accelerate/blob/main/examples/by_feature/megatron_lm_gpt_pretraining.py)

### الأمثلة الكاملة:

تعرض هذه الأمثلة كل ميزة في Accelerate في نفس الوقت الذي تم عرضه في "أمثلة خاصة بالميزات".

- [مثال NLP الكامل](https://github.com/huggingface/accelerate/blob/main/examples/complete_nlp_example.py)
- [مثال الرؤية الحاسوبية الكامل](https://github.com/huggingface/accelerate/blob/main/examples/complete_cv_example.py)
- [مثال رؤية شامل وقابل للتوسيع للغاية يُظهر SLURM و hydra واستخدام إطار العمل القابل للتوسيع للغاية](https://github.com/yuvalkirstain/PickScore)
- [مثال ضبط نموذج اللغة السببية](https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm_no_trainer.py)
- [مثال ضبط نموذج اللغة المُقنعة](https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_mlm_no_trainer.py)
- [مثال الضبط المسبق للكلام](https://github.com/huggingface/transformers/blob/main/examples/pytorch/speech-pretraining/run_wav2vec2_pretraining_no_trainer.py)
- [مثال الضبط الدقيق للترجمة](https://github.com/huggingface/transformers/blob/main/examples/pytorch/translation/run_translation_no_trainer.py)
- [مثال الضبط الدقيق لتصنيف النصوص](https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_glue_no_trainer.py)
- [مثال الضبط الدقيق لتجزئة الصور](https://github.com/huggingface/transformers/blob/main/examples/pytorch/semantic-segmentation/run_semantic_segmentation_no_trainer.py)
- [مثال الضبط الدقيق للإجابة على الأسئلة](https://github.com/huggingface/transformers/blob/main/examples/pytorch/question-answering/run_qa_no_trainer.py)
- [مثال الضبط الدقيق للإجابة على الأسئلة باستخدام البحث الشعاعي](https://github.com/huggingface/transformers/blob/main/examples/pytorch/question-answering/run_qa_beam_search_no_trainer.py)
- [مثال الضبط الدقيق للإجابة على أسئلة الاختيار المتعدد](https://github.com/huggingface/transformers/blob/main/examples/pytorch/multiple-choice/run_swag_no_trainer.py)
- [مثال الضبط الدقيق للتعرف على الكيانات المسماة](https://github.com/huggingface/transformers/blob/main/examples/pytorch/token-classification/run_ner_no_trainer.py)
- [مثال الضبط الدقيق لتصنيف الصور](https://github.com/huggingface/transformers/blob/main/examples/pytorch/image-classification/run_image_classification_no_trainer.py)
- [مثال الضبط الدقيق للتلخيص](https://github.com/huggingface/transformers/blob/main/examples/pytorch/summarization/run_summarization_no_trainer.py)
- [أمثلة شاملة حول كيفية استخدام تكامل AWS SageMaker لـ Accelerate](https://github.com/huggingface/notebooks/blob/main/sagemaker/22_accelerate_sagemaker_examples/README.md)
- [أمثلة Megatron-LM لمهام NLP المختلفة](https://github.com/pacman100/accelerate-megatron-test)

## أمثلة التكامل:

هذه هي البرامج التعليمية من المكتبات التي تدمج مع 🤗 Accelerate:

> هل لا تجد تكامل هنا؟ قم بإنشاء طلب سحب لإضافته!

### Amphion

- [تدريب نماذج Text-to-Speech باستخدام Amphion](https://github.com/open-mmlab/Amphion/blob/main/egs/tts/README.md)
- [تدريب نماذج تحويل صوت الغناء باستخدام Amphion](https://github.com/open-mmlab/Amphion/blob/main/egs/svc/README.md)
- [تدريب Vocoders باستخدام Amphion](https://github.com/open-mmlab/Amphion/blob/main/egs/vocoder/README.md)

### المحفز

- [برنامج تعليمي للتدريب الموزع باستخدام المحفز](https://catalyst-team.github.io/catalyst/tutorials/ddp.html)

### DALLE2-pytorch

- [ضبط دقيق لـ DALLE2](https://github.com/lucidrains/DALLE2-pytorch#usage)

### 🤗 الناشرون

- [أداء الانعكاس النصي باستخدام الناشرون](https://github.com/huggingface/diffusers/tree/main/examples/textual_inversion)
- [تدريب DreamBooth باستخدام الناشرون](https://github.com/huggingface/diffusers/tree/main/examples/dreambooth)

### fastai

- [التدريب الموزع من Jupyter Notebooks باستخدام fastai](https://docs.fast.ai/tutorial.distributed.html)
- [أمثلة التدريب الموزع الأساسية باستخدام fastai](https://docs.fast.ai/examples/distributed_app_examples.html)

### GradsFlow

- [التصنيف التلقائي للصور باستخدام GradsFlow](https://docs.gradsflow.com/en/latest/examples/nbs/01-ImageClassification/)

### imagen-pytorch

- [ضبط دقيق لـ Imagen](https://github.com/lucidrains/imagen-pytorch#usage)

### Kornia

- [ضبط دقيق لنماذج الرؤية باستخدام مدرب Kornia](https://kornia.readthedocs.io/en/latest/get-started/training.html)

### PyTorch Accelerated

- [برنامج تعليمي للبدء السريع للتدريب الموزع باستخدام PyTorch Accelerated](https://pytorch-accelerated.readthedocs.io/en/latest/quickstart.html)

### PyTorch3D

- [أداء التعلم العميق مع بيانات ثلاثية الأبعاد](https://pytorch3d.org/tutorials/)

### Stable-Dreamfusion

- [التدريب باستخدام Stable-Dreamfusion لتحويل النص إلى نموذج ثلاثي الأبعاد](https://colab.research.google.com/drive/1MXT3yfOFvO0ooKEfiUUvTKwUkrrlCHpF?usp=sharing)

### Tez

- [كشف أمراض الأوراق باستخدام Tez و Accelerate](https://www.kaggle.com/code/abhishek/tez-faster-and-easier-training-for-leaf-detection/notebook)

### trlx

- [كيفية تنفيذ مهمة تعلم المشاعر باستخدام trlx](https://github.com/CarperAI/trlx#example-how-to-add-a-task)

### Comfy-UI

- [تمكين استخدام نماذج Stable Diffusion الكبيرة في إعدادات vram المنخفضة باستخدام Accelerate](https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/model_management.py#L291-L296)
## في العلوم

فيما يلي قائمة غير شاملة بالأوراق البحثية التي تستخدم Hugging Face Accelerate.

> إذا لم تجد ورقتك هنا، قم بإنشاء طلب سحب (Pull Request) لإضافتها!

* Yuval Kirstain, Adam Polyak, Uriel Singer, Shahbuland Matiana, Joe Penna, Omer Levy: “Pick-a-Pic: An Open Dataset of User Preferences for Text-to-Image Generation”, 2023; [arXiv:2305.01569](http://arxiv.org/abs/2305.01569).
* Lei Wang, Wanyu Xu, Yihuai Lan, Zhiqiang Hu, Yunshi Lan, Roy Ka-Wei Lee, Ee-Peng Lim: “Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning by Large Language Models”, 2023; [arXiv:2305.04091](http://arxiv.org/abs/2305.04091).
* Arthur Câmara, Claudia Hauff: “Moving Stuff Around: A study on efficiency of moving documents into memory for Neural IR models”, 2022; [arXiv:2205.08343](http://arxiv.org/abs/2205.08343).
* Ying Sheng, Lianmin Zheng, Binhang Yuan, Zhuohan Li, Max Ryabinin, Daniel Y. Fu, Zhiqiang Xie, Beidi Chen, Clark Barrett, Joseph E. Gonzalez, Percy Liang, Christopher Ré, Ion Stoica, Ce Zhang: “High-throughput Generative Inference of Large Language Models with a Single GPU”, 2023; [arXiv:2303.06865](http://arxiv.org/abs/2303.06865).
* Peter Melchior, Yan Liang, ChangHoon Hahn, Andy Goulding: “Autoencoding Galaxy Spectra I: Architecture”, 2022; [arXiv:2211.07890](http://arxiv.org/abs/2211.07890).
* Jiaao Chen, Aston Zhang, Mu Li, Alex Smola, Diyi Yang: “A Cheaper and Better Diffusion Language Model with Soft-Masked Noise”, 2023; [arXiv:2304.04746](http://arxiv.org/abs/2304.04746).
* Ayaan Haque, Matthew Tancik, Alexei A. Efros, Aleksander Holynski, Angjoo Kanazawa: “Instruct-NeRF2NeRF: Editing 3D Scenes with Instructions”, 2023; [arXiv:2303.12789](http://arxiv.org/abs/2303.12789).
* Luke Melas-Kyriazi, Christian Rupprecht, Iro Laina, Andrea Vedaldi: “RealFusion: 360° Reconstruction of Any Object from a Single Image”, 2023; [arXiv:2302.10663](http://arxiv.org/abs/2302.10663).
* Xiaoshi Wu, Keqiang Sun, Feng Zhu, Rui Zhao, Hongsheng Li: “Better Aligning Text-to-Image Models with Human Preference”, 2023; [arXiv:2303.14420](http://arxiv.org/abs/2303.14420).
* Yongliang Shen, Kaitao Song, Xu Tan, Dongsheng Li, Weiming Lu, Yueting Zhuang: “HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in HuggingFace”, 2023; [arXiv:2303.17580](http://arxiv.org/abs/2303.17580).
* Yue Yang, Wenlin Yao, Hongming Zhang, Xiaoyang Wang, Dong Yu, Jianshu Chen: “Z-LaVI: Zero-Shot Language Solver Fueled by Visual Imagination”, 2022; [arXiv:2210.12261](http://arxiv.org/abs/2210.12261).
* Sheng-Yen Chou, Pin-Yu Chen, Tsung-Yi Ho: “How to Backdoor Diffusion Models?”, 2022; [arXiv:2212.05400](http://arxiv.org/abs/2212.05400).
* Junyoung Seo, Wooseok Jang, Min-Seop Kwak, Jaehoon Ko, Hyeonsu Kim, Junho Kim, Jin-Hwa Kim, Jiyoung Lee, Seungryong Kim: “Let 2D Diffusion Model Know 3D-Consistency for Robust Text-to-3D Generation”, 2023; [arXiv:2303.07937](http://arxiv.org/abs/2303.07937).
* Or Patashnik, Daniel Garibi, Idan Azuri, Hadar Averbuch-Elor, Daniel Cohen-Or: “Localizing Object-level Shape Variations with Text-to-Image Diffusion Models”, 2023; [arXiv:2303.11306](http://arxiv.org/abs/2303.11306).
* Dídac Surís, Sachit Menon, Carl Vondrick: “ViperGPT: Visual Inference via Python Execution for Reasoning”, 2023; [arXiv:2303.08128](http://arxiv.org/abs/2303.08128).
* Chenyang Qi, Xiaodong Cun, Yong Zhang, Chenyang Lei, Xintao Wang, Ying Shan, Qifeng Chen: “FateZero: Fusing Attentions for Zero-shot Text-based Video Editing”, 2023; [arXiv:2303.09535](http://arxiv.org/abs/2303.09535).
* Sean Welleck, Jiacheng Liu, Ximing Lu, Hannaneh Hajishirzi, Yejin Choi: “NaturalProver: Grounded Mathematical Proof Generation with Language Models”, 2022; [arXiv:2205.12910](http://arxiv.org/abs/2205.12910).
* Elad Richardson, Gal Metzer, Yuval Alaluf, Raja Giryes, Daniel Cohen-Or: “TEXTure: Text-Guided Texturing of 3D Shapes”, 2023; [arXiv:2302.01721](http://arxiv.org/abs/2302.01721).
* Puijin Cheng, Li Lin, Yijin Huang, Huaqing He, Wenhan Luo, Xiaoying Tang: “Learning Enhancement From Degradation: A Diffusion Model For Fundus Image Enhancement”, 2023; [arXiv:2303.04603](http://arxiv.org/abs/2303.04603).
* Shun Shao, Yftah Ziser, Shay Cohen: “Erasure of Unaligned Attributes from Neural Representations”, 2023; [arXiv:2302.02997](http://arxiv.org/abs/2302.02997).
* Seonghyeon Ye, Hyeonbin Hwang, Sohee Yang, Hyeongu Yun, Yireun Kim, Minjoon Seo: “In-Context Instruction Learning”, 2023; [arXiv:2302.14691](http://arxiv.org/abs/2302.14691).
* Shikun Liu, Linxi Fan, Edward Johns, Zhiding Yu, Chaowei Xiao, Anima Anandkumar: “Prismer: A Vision-Language Model with An Ensemble of Experts”, 2023; [arXiv:2303.02506](http://arxiv.org/abs/2303.02506).
* Haoyu Chen, Zhihua Wang, Yang Yang, Qilin Sun, Kede Ma: “Learning a Deep Color Difference Metric for Photographic Images”, 2023; [arXiv:2303.14964](http://arxiv.org/abs/2303.14964).
* Van-Hoang Le, Hongyu Zhang: “Log Parsing with Prompt-based Few-shot Learning”, 2023; [arXiv:2302.07435](http://arxiv.org/abs/2302.07435).
* Keito Kudo, Yoichi Aoki, Tatsuki Kuribayashi, Ana Brassard, Masashi Yoshikawa, Keisuke Sakaguchi, Kentaro Inui: “Do Deep Neural Networks Capture Compositionality in Arithmetic Reasoning?”, 2023; [arXiv:2302.07866](http://arxiv.org/abs/2302.07866).
* Ruoyao Wang, Peter Jansen, Marc-Alexandre Côté, Prithviraj Ammanabrolu: “Behavior Cloned Transformers are Neurosymbolic Reasoners”, 2022; [arXiv:2210.07382](http://arxiv.org/abs/2210.07382).
* Martin Wessel, Tomáš Horych, Terry Ruas, Akiko Aizawa, Bela Gipp, Timo Spinde: “Introducing MBIB -- the first Media Bias Identification Benchmark Task and Dataset Collection”, 2023; [arXiv:2304.13148](http://arxiv.org/abs/2304.13148). DOI: [https://dx.doi.org/10.1145/3539618.3591882 10.1145/3539618.3591882].
* Hila Chefer, Yuval Alaluf, Yael Vinker, Lior Wolf, Daniel Cohen-Or: “Attend-and-Excite: Attention-Based Semantic Guidance for Text-to-Image Diffusion Models”, 2023; [arXiv:2301.13826](http://arxiv.org/abs/2301.13826).
* Marcio Fonseca, Yftah Ziser, Shay B. Cohen: “Factorizing Content and Budget Decisions in Abstractive Summarization of Long Documents”, 2022; [arXiv:2205.12486](http://arxiv.org/abs/2205.12486).
* Elad Richardson, Gal Metzer, Yuval Alaluf, Raja Giryes, Daniel Cohen-Or: “TEXTure: Text-Guided Texturing of 3D Shapes”, 2023; [arXiv:2302.01721](http://arxiv.org/abs/2302.01721).
* Tianxing He, Jingyu Zhang, Tianle Wang, Sachin Kumar, Kyunghyun Cho, James Glass, Yulia Tsvetkov: “On the Blind Spots of Model-Based Evaluation Metrics for Text Generation”, 2022; [arXiv:2212.10020](http://arxiv.org/abs/2212.10020).
* Ori Ram, Yoav Levine, Itay Dalmedigos, Dor Muhlgay, Amnon Shashua, Kevin Leyton-Brown, Yoav Shoham: “In-Context Retrieval-Augmented Language Models”, 2023; [arXiv:2302.00083](http://arxiv.org/abs/2302.00083).
* Dacheng Li, Rulin Shao, Hongyi Wang, Han Guo, Eric P. Xing, Hao Zhang: “MPCFormer: fast, performant and private Transformer inference with MPC”, 2022; [arXiv:2211.01452](http://arxiv.org/abs/2211.01452).
* Baolin Peng, Michel Galley, Pengcheng He, Chris Brockett, Lars Liden, Elnaz Nouri, Zhou Yu, Bill Dolan, Jianfeng Gao: “GODEL: Large-Scale Pre-Training for Goal-Directed Dialog”, 2022; [arXiv:2206.11309](http://arxiv.org/abs/2206.11309).
* Egil Rønningstad, Erik Velldal, Lilja Øvrelid: “Entity-Level Sentiment Analysis (ELSA): An exploratory task survey”, 2023, Proceedings of the 29th International Conference on Computational Linguistics, 2022, pages 6773-6783; [arXiv:2304.14241](http://arxiv.org/abs/2304.14241).
* Charlie Snell, Ilya Kostrikov, Yi Su, Mengjiao Yang, Sergey Levine: “Offline RL for Natural Language Generation with Implicit Language Q Learning”, 2022; [arXiv:2206.11871](http://arxiv.org/abs/2206.11871).
* Zhiruo Wang, Shuyan Zhou, Daniel Fried, Graham Neubig: “Execution-Based Evaluation for Open-Domain Code Generation”, 2022; [arXiv:2212.10481](http://arxiv.org/abs/2212.10481).
* Minh-Long Luu, Zeyi Huang, Eric P. Xing, Yong Jae Lee, Haohan Wang: “Expeditious Saliency-guided Mix-up through Random Gradient Thresholding”, 2022; [arXiv:2212.04875](http://arxiv.org/abs/2212.04875).
* Jun Hao Liew, Hanshu Yan, Daquan Zhou, Jiashi Feng: “MagicMix: Semantic Mixing with Diffusion Models”, 2022; [arXiv:2210.16056](http://arxiv.org/abs/2210.16056).
* Yaqing Wang, Subhabrata Mukherjee, Xiaodong Liu, Jing Gao, Ahmed Hassan Awadallah, Jianfeng Gao: “LiST: Lite Prompted Self-training Makes Parameter-Efficient Few-shot Learners”, 2021; [arXiv:2110.06274](http://arxiv.org/abs/2110.06274).