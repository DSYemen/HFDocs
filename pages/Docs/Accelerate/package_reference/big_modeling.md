# العمل مع النماذج الكبيرة

## إرسال النماذج ونقل عبء العمل منها

[[autodoc]] big_modeling.init_empty_weights

[[autodoc]] big_modeling.cpu_offload

[[autodoc]] big_modeling.cpu_offload_with_hook

يسمح big_modeling.cpu_offload_with_hook بإدارة النقل التلقائي للعبء من وحدة المعالجة المركزية.

[[autodoc]] big_modeling.disk_offload

[[autodoc]] big_modeling.dispatch_model

[[autodoc]] big_modeling.load_checkpoint_and_dispatch

[[autodoc]] big_modeling.load_checkpoint_in_model

[[autodoc]] utils.infer_auto_device_map

## خطافات النماذج

### فئات الخطاف

[[autodoc]] hooks.ModelHook

[[autodoc]] hooks.AlignDevicesHook

[[autodoc]] hooks.SequentialHook

### إضافة خطافات

[[autodoc]] hooks.add_hook_to_module

[[autodoc]] hooks.attach_execution_device_hook

[[autodoc]] hooks.attach_align_device_hook

[[autodoc]] hooks.attach_align_device_hook_on_blocks

### إزالة الخطافات

[[autodoc]] hooks.remove_hook_from_module

[[autodoc]] hooks.remove_hook_from_submodules