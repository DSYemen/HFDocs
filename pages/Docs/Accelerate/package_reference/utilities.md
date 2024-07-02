# وظائف مساعدة مفيدة

فيما يلي مجموعة متنوعة من وظائف المنفعة التي توفرها 🤗 Accelerate، مصنفة حسب الحالة الاستخدام.

## الثوابت

يتم استخدام الثوابت في جميع أنحاء 🤗 Accelerate للرجوع إليها.

فيما يلي الثوابت المستخدمة عند استخدام [`Accelerator.save_state`]:

- `utils.MODEL_NAME`: `"pytorch_model"`
- `utils.OPTIMIZER_NAME`: `"optimizer"`
- `utils.RNG_STATE_NAME`: `"random_states"`
- `utils.SCALER_NAME`: `"scaler.pt"`
- `utils.SCHEDULER_NAME`: `"scheduler"`

فيما يلي الثوابت المستخدمة عند استخدام [`Accelerator.save_model`]:

- `utils.WEIGHTS_NAME`: `"pytorch_model.bin"`
- `utils.SAFE_WEIGHTS_NAME`: `"model.safetensors"`
- `utils.WEIGHTS_INDEX_NAME`: `"pytorch_model.bin.index.json"`
- `utils.SAFE_WEIGHTS_INDEX_NAME`: `"model.safetensors.index.json"`

## فئات البيانات

هذه هي فئات البيانات الأساسية المستخدمة في جميع أنحاء 🤗 Accelerate ويمكن تمريرها كمعلمات.

### منفصل

هذه هي فئات البيانات المنفصلة المستخدمة للتحقق، مثل نوع نظام الموزع الذي يتم استخدامه:

- [[autodoc]] utils.ComputeEnvironment
- [[autodoc]] utils.DistributedType
- [[autodoc]] utils.DynamoBackend
- [[autodoc]] utils.LoggerType
- [[autodoc]] utils.PrecisionType
- [[autodoc]] utils.RNGType
- [[autodoc]] utils.SageMakerDistributedType

### kwargs

هذه هي الحجج القابلة للتكوين لتفاعلات محددة في جميع أنحاء النظام البيئي PyTorch التي يعالجها Accelerate تحت الغطاء:

- [[autodoc]] utils.AutocastKwargs
- [[autodoc]] utils.DistributedDataParallelKwargs
- [[autodoc]] utils.FP8RecipeKwargs
- [[autodoc]] utils.GradScalerKwargs
- [[autodoc]] utils.InitProcessGroupKwargs
- [[autodoc]] utils.KwargsHandler

## المكونات الإضافية

هذه هي المكونات الإضافية التي يمكن تمريرها إلى كائن [`Accelerator`]. في حين أنها معرفة في مكان آخر في التوثيق، وللراحة، يمكن الاطلاع على جميعها هنا:

- [[autodoc]] utils.DeepSpeedPlugin
- [[autodoc]] utils.FullyShardedDataParallelPlugin
- [[autodoc]] utils.GradientAccumulationPlugin
- [[autodoc]] utils.MegatronLMPlugin
- [[autodoc]] utils.TorchDynamoPlugin

## التكوينات

هذه هي الفئات التي يمكن تكوينها وتمريرها عبر التكامل المناسب:

- [[autodoc]] utils.BnbQuantizationConfig
- [[autodoc]] utils.DataLoaderConfiguration
- [[autodoc]] utils.ProjectConfiguration

## متغيرات البيئة

هذه هي متغيرات البيئة التي يمكن تمكينها لحالات استخدام مختلفة:

- `ACCELERATE_DEBUG_MODE` (`str`): ما إذا كان سيتم تشغيل Accelerate في وضع التصحيح. مزيد من المعلومات متاحة [هنا](../usage_guides/debug.md).

## عمليات البيانات والعمليات

تشمل هذه العمليات عمليات بيانات تحاكي نفس عمليات "التورتش" ولكن يمكن استخدامها في عمليات موزعة.

- [[autodoc]] utils.broadcast
- [[autodoc]] utils.broadcast_object_list
- [[autodoc]] utils.concatenate
- [[autodoc]] utils.convert_outputs_to_fp32
- [[autodoc]] utils.convert_to_fp32
- [[autodoc]] utils.gather
- [[autodoc]] utils.gather_object
- [[autodoc]] utils.listify
- [[autodoc]] utils.pad_across_processes
- [[autodoc]] utils.recursively_apply
- [[autodoc]] utils.reduce
- [[autodoc]] utils.send_to_device
- [[autodoc]] utils.slice_tensors

## فحوصات البيئة

تفحص هذه الوظائف حالة بيئة العمل الحالية، بما في ذلك معلومات حول نظام التشغيل نفسه، وما يمكنه دعمه، وما إذا كانت التبعيات معينة مثبتة.

- [[autodoc]] utils.is_bf16_available
- [[autodoc]] utils.is_ipex_available
- [[autodoc]] utils.is_mps_available
- [[autodoc]] utils.is_npu_available
- [[autodoc]] utils.is_torch_version
- [[autodoc]] utils.is_torch_xla_available
- [[autodoc]] utils.is_xpu_available

## التلاعب بالبيئة

- [[autodoc]] utils.patch_environment
- [[autodoc]] utils.clear_environment
- [[autodoc]] utils.write_basic_config

عند إعداد 🤗 Accelerate لأول مرة، بدلاً من تشغيل `accelerate config`، يمكن استخدام [~utils.write_basic_config] كبديل للتكوين السريع.

- [[autodoc]] utils.set_numa_affinity
- [[autodoc]] utils.environment.override_numa_affinity

## الذاكرة

- [[autodoc]] utils.find_executable_batch_size

## وضع النماذج

ترتبط هذه المرافق بالتفاعل مع نماذج PyTorch:

- [[autodoc]] utils.calculate_maximum_sizes
- [[autodoc]] utils.compute_module_sizes
- [[autodoc]] utils.extract_model_from_parallel
- [[autodoc]] utils.get_balanced_memory
- [[autodoc]] utils.get_max_layer_size
- [[autodoc]] utils.infer_auto_device_map
- [[autodoc]] utils.load_checkpoint_in_model
- [[autodoc]] utils.load_offloaded_weights
- [[autodoc]] utils.load_state_dict
- [[autodoc]] utils.offload_state_dict
- [[autodoc]] utils.retie_parameters
- [[autodoc]] utils.set_module_tensor_to_device
- [[autodoc]] utils.shard_checkpoint

## متوازي

تشمل هذه المرافق العامة التي يجب استخدامها عند العمل بالتوازي:

- [[autodoc]] utils.extract_model_from_parallel
- [[autodoc]] utils.save
- [[autodoc]] utils.wait_for_everyone

## عشوائي

ترتبط هذه المرافق بضبط جميع حالات الأرقام العشوائية وتزامنها:

- [[autodoc]] utils.set_seed
- [[autodoc]] utils.synchronize_rng_state
- [[autodoc]] utils.synchronize_rng_states

## PyTorch XLA

تشمل هذه المرافق المفيدة أثناء استخدام PyTorch مع XLA:

- [[autodoc]] utils.install_xla

## تحميل أوزان النموذج

تشمل هذه المرافق المفيدة لتحميل نقاط التفتيش:

- [[autodoc]] utils.load_checkpoint_in_model

## التكميم

تشمل هذه المرافق المفيدة لكمية نموذج:

- [[autodoc]] utils.load_and_quantize_model