# المرافق الخاصة بـ Megatron-LM 

[[autodoc]] utils.MegatronLMPlugin  
[[autodoc]] utils.MegatronLMDummyScheduler  
[[autodoc]] utils.MegatronLMDummyDataLoader  

تعد MegatronLMPlugin فئة فرعية من PLUGIN_CLASS، والتي توفر وظائف إضافية محددة لـ Megatron-LM. على وجه التحديد، فإنه يضيف جدولة التعلم المعزز، وتعديل معدل التعلم، وتعديل معامل الانحدار، وإدارة الذاكرة المؤقتة، وتخصيص عملية التدريب.

MegatronLMDummyScheduler هي فئة وهمية تستخدم لتحل محل الجدولة الفعلية عند الحاجة إلى تشغيل الخطوات التدريبية بشكل متكرر. إنه يتجاوز الأسلوب step() لإرجاع القيمة الصحيحة في الوقت المناسب.

MegatronLMDummyDataLoader هي فئة أخرى وهمية، مصممة لتحل محل DataLoader الفعلي. إنه يتجاوز الأساليب __iter__() و__len__() لتقديم الوظائف الصحيحة.

[[autodoc]] utils.AbstractTrainStep  
[[autodoc]] utils.GPTTrainStep  
[[autodoc]] utils.BertTrainStep  
[[autodoc]] utils.T5TrainStep  

تعد AbstractTrainStep فئة أساسية توفر الوظائف المشتركة لخطوات التدريب المختلفة. إنه يحدد الأساليب init() وset_model() وset_optimizer() وzero_grad() وforward_step() وbackward_step() وstep()، والتي يتم تجاوزها بواسطة الفئات الفرعية.

GPTTrainStep وBertTrainStep وT5TrainStep هي فئات فرعية من AbstractTrainStep، مصممة لنماذج GPT وBERT وT5 على التوالي. إنها تتجاوز الأساليب المناسبة من الفئة الأساسية لتنفيذ خطوات التدريب المخصصة لكل نوع من النماذج.

[[autodoc]] utils.avg_losses_across_data_parallel_group  

تعد avg_losses_across_data_parallel_group وظيفة مفيدة لحساب متوسط الخسائر عبر مجموعة من العمليات المتوازية للبيانات. إنه يأخذ قائمة من الخسائر كمدخلات ويعيد متوسط الخسائر، مع مراعاة النطاق الحالي.