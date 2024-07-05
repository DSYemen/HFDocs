# فئات التغليف لفئات بيانات PyTorch ومحسنات وجدولة 

الفئات الداخلية التي يستخدمها Accelerate لإعداد الكائنات للتدريب الموزع عند استدعاء [`~Accelerator.prepare`].

## مجموعات البيانات ومجموعات بيانات التدريب

[[autodoc]] data_loader.prepare_data_loader
[[autodoc]] data_loader.skip_first_batches
[[autodoc]] data_loader.BatchSamplerShard
[[autodoc]] data_loader.IterableDatasetShard
[[autodoc]] data_loader.DataLoaderShard
[[autodoc]] data_loader.DataLoaderDispatcher

## المحسنات

[[autodoc]] optimizer.AcceleratedOptimizer

## الجدولة

[[autodoc]] scheduler.AcceleratedScheduler