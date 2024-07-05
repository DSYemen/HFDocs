# التثبيت والتهيئة

قبل البدء، ستحتاج إلى إعداد بيئتك، وتثبيت الحزم المناسبة، وتهيئة 🤗 Accelerate. تم اختبار 🤗 Accelerate على **Python 3.8+**.

## تثبيت 🤗 Accelerate

🤗 Accelerate متاح على pypi و conda، وكذلك على GitHub. تتوفر التفاصيل لتثبيته من كل منها أدناه:

### pip

لتثبيت 🤗 Accelerate من pypi، قم بتشغيل ما يلي:

```bash
pip install accelerate
```

### conda

يمكن أيضًا تثبيت 🤗 Accelerate مع conda باستخدام ما يلي:

```bash
conda install -c conda-forge accelerate
```

### المصدر

يتم إضافة ميزات جديدة كل يوم لم يتم إصدارها بعد. لتجربتها بنفسك، قم بالتثبيت من مستودع GitHub:

```bash
pip install git+https://github.com/huggingface/accelerate
```

إذا كنت تعمل على المساهمة في المكتبة أو ترغب في اللعب مع الكود المصدري ورؤية النتائج المباشرة أثناء تشغيل الكود، يمكن تثبيت إصدار قابل للتحرير من نسخة مستنسخة محليًا من المستودع:

```bash
git clone https://github.com/huggingface/accelerate
cd accelerate
pip install -e .
```

## تهيئة 🤗 Accelerate

بعد التثبيت، تحتاج إلى تهيئة 🤗 Accelerate وفقًا لطريقة إعداد النظام الحالي للتدريب. للقيام بذلك، قم بتشغيل ما يلي والإجابة عن الأسئلة المطروحة:

```bash
accelerate config
```

لإنشاء تكوين أساسي لا يتضمن خيارات مثل تكوين DeepSpeed أو التشغيل على وحدات TPU، يمكنك تشغيل ما يلي بسرعة:

```bash
python -c "from accelerate.utils import write_basic_config; write_basic_config(mixed_precision='fp16')"
```

🤗 Accelerate سيقوم تلقائيًا باستخدام العدد الأقصى من وحدات معالجة الرسوميات (GPUs) المتاحة وتعيين وضع الدقة المختلطة.

للتحقق من مظهر تكوينك، قم بتشغيل:

```bash
accelerate env
```

يُظهر الإخراج أدناه مثالًا يصف وحدتي GPU على جهاز واحد دون استخدام الدقة المختلطة:

```bash
- `Accelerate` version: 0.11.0.dev0
- Platform: Linux-5.10.0-15-cloud-amd64-x86_64-with-debian-11.3
- Python version: 3.7.12
- Numpy version: 1.19.5
- PyTorch version (GPU?): 1.12.0+cu102 (True)
- `Accelerate` default config:
        - compute_environment: LOCAL_MACHINE
        - distributed_type: MULTI_GPU
        - mixed_precision: no
        - use_cpu: False
        - num_processes: 2
        - machine_rank: 0
        - num_machines: 1
        - main_process_ip: None
        - main_process_port: None
        - main_training_function: main
        - deepspeed_config: {}
        - fsdp_config: {}
```

