# آليات 🤗 Accelerate الداخلية 

داخلياً، يعمل 🤗 Accelerate أولاً عن طريق تحليل بيئة تشغيل النص البرمجي لتحديد نوع الإعداد الموزع المستخدم، وعدد العمليات المختلفة، والتعرف على العملية الحالية. يتم تخزين جميع هذه المعلومات في فئة [`~AcceleratorState`][acceleratorstate]. 

يتم تهيئة هذه الفئة لأول مرة عند إنشاء مثيل لفئة [`~Accelerator`][accelerator]، بالإضافة إلى تنفيذ أي تهيئة محددة يحتاجها الإعداد الموزع الخاص بك. بعد ذلك، يتم مشاركة حالتها بشكل فريد عبر جميع مثيلات فئة [`~state.AcceleratorState`][stateacceleratorstate]. (يمكن القيام بالشيء نفسه مع فئة [`PartialState`][partialstate]، وهي نسخة أكثر بساطة من الفئة الأصلية التي ترث منها الوظائف). 

عند استدعاء الدالة [`~Accelerator.prepare`][acceleratorprepare]، يقوم المكتبة بما يلي: 

- تغليف النموذج (النماذج) الخاص بك في الحاوية المناسبة للإعداد الموزع. 
- تغليف المحسن (المحسنات) الخاص بك في فئة [`~optimizer.AcceleratedOptimizer`][optimizeracceleratedoptimizer]. 
- تغليف المخطط (المخططات) الخاص بك في فئة [`~scheduler.AcceleratedScheduler`][scheduleraacceleratedscheduler]. 
- إنشاء نسخة جديدة من برنامج تحميل البيانات (DataLoader) الخاص بك في فئة [`~data_loader.DataLoaderShard`][dataloaderdataloader] أو [`~data_loader.DataLoaderDispatcher`][dataloaderdispatcherdata]. 

في حين يتم وضع النماذج والمحسنات والمخططات ببساطة في أغلفة، يتم إعادة إنشاء برامج تحميل البيانات. ويرجع ذلك بشكل أساسي إلى أن PyTorch لا يسمح للمستخدم بتغيير `batch_sampler` لبرنامج تحميل البيانات بعد إنشائه، حيث يقوم المكتبة بتقسيم بياناتك بين العمليات عن طريق تغيير `batch_sampler` لإنتاج كل دفعة أخرى `num_processes` (إذا تم تمكينها). 

تُضيف فئة [`~data_loader.DataLoaderShard`][dataloaderdataloader]، التي تُورث من فئة `DataLoader`، الوظائف التالية: 

- تزامن مولد الأرقام العشوائية المناسب لجميع العمليات في كل تكرار جديد، لضمان تنفيذ أي عملية تعشيق (مثل التعشيق) بنفس الطريقة عبر العمليات. 
- وضع الدفعات على الجهاز المناسب قبل إنتاجها (ما لم تختر إلغاء `device_placement=True`). 

تختلف فئة [`~data_loader.DataLoaderDispatcher`][dataloaderdispatcherdata] عن فئة [`~data_loader.DataLoaderShard`][dataloaderdataloader] بأنه عند التكرار خلال `DataLoader`، يتم تقسيم البيانات بدءًا من العملية 0 *ثم* إرسالها إلى كل عملية بدلاً من حدوث ذلك على مستوى مجموعة البيانات. 

سيقوم تزامن مولد الأرقام العشوائية بشكل افتراضي بمزامنة ما يلي: 

- صفة `generator` لمأخذ عينات معين (مثل `RandomSampler` في PyTorch) للإصدارات PyTorch >= 1.6. 
- مولد الأرقام العشوائية الرئيسي في PyTorch <=1.5.1. 

يمكنك اختيار مولد الأرقام العشوائية الذي تريد مزامنته باستخدام وسيط `rng_types` في فئة [`Accelerator`][accelerator] الرئيسية. في PyTorch >= 1.6، يُنصح بالاعتماد على مولد `generator` محلي لتجنب تعيين نفس البذرة في مولد الأرقام العشوائية الرئيسي في جميع العمليات. 

<Tip warning={true}> 

ستؤثر مزامنة مولد الأرقام العشوائية الرئيسي في PyTorch (أو CUDA أو XLA) على أي آثار عشوائية أخرى محتملة في مجموعة البيانات الخاصة بك (مثل التعزيز العشوائي للبيانات) بمعنى أن جميع العمليات ستحصل على نفس الأرقام العشوائية من وحدات PyTorch العشوائية (لذا فستطبق نفس التعزيز العشوائي للبيانات إذا كان يتحكم فيه PyTorch). 

</Tip> 

<Tip> 

يجب تنفيذ الجزء العشوائي من مأخذ العينات المخصص أو المأخذ العشوائي للدفعات أو مجموعة البيانات القابلة للتحديد باستخدام كائن `torch.Generator` محلي (في PyTorch >= 1.6)، انظر `RandomSampler` التقليدي كمثال. 

</Tip> 

للحصول على مزيد من التفاصيل حول المكونات الداخلية، راجع صفحة [الداخليات][internals].

[acceleratorstate]: https://huggingface.co/docs/accelerate/main/en/package_reference/torch_wrappers#acceleratorstate
[accelerator]: https://huggingface.co/docs/accelerate/main/en/package_reference/torch_wrappers#accelerator
[stateacceleratorstate]: https://huggingface.co/docs/accelerate/main/en/package_reference/torch_wrappers#stateacceleratorstate
[partialstate]: https://huggingface.co/docs/accelerate/main/en/package_reference/torch_wrappers#partialstate
[acceleratorprepare]: https://huggingface.co/docs/accelerate/main/en/package_reference/torch_wrappers#acceleratorprepare
[optimizeracceleratedoptimizer]: https://huggingface.co/docs/accelerate/main/en/package_reference/torch_wrappers#optimizeracceleratedoptimizer
[scheduleraacceleratedscheduler]: https://huggingface.co/docs/accelerate/main/en/package_reference/torch_wrappers#scheduleraacceleratedscheduler
[dataloaderdataloader]: https://huggingface.co/docs/accelerate/main/en/package_reference/torch_wrappers#dataloaderdataloader
[dataloaderdispatcherdata]: https://huggingface.co/docs/accelerate/main/en/package_reference/torch_wrappers#dataloaderdispatcherdata
[internals]: https://huggingface.co/docs/accelerate/main/en/package_reference/internals