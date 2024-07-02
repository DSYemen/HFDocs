# PEFT
🤗 PEFT (Parameter-Efficient Fine-Tuning) هي مكتبة لتكييف النماذج الكبيرة مسبقة التدريب بكفاءة مع تطبيقات مختلفة دون الحاجة إلى ضبط دقيق لجميع معلمات النموذج لأن ذلك مكلف للغاية. تقوم طرق PEFT بضبط دقيق لعدد صغير فقط من معلمات (إضافية) النموذج - مما يقلل بشكل كبير من التكاليف الحسابية وتكاليف التخزين - مع تحقيق أداء قابل للمقارنة مع نموذج تمت معايرته بشكل كامل. يجعل هذا الأمر أكثر سهولة في تدريب وتخزين نماذج اللغة الكبيرة (LLMs) على أجهزة المستهلك.

تم دمج PEFT مع مكتبات Transformers وDiffusers وAccelerate لتوفير طريقة أسرع وأسهل لتحميل النماذج الكبيرة وتدريبها واستخدامها للاستنتاج.

<div class="mt-10">
<div class="w-full flex flex-col space-y-4 md:space-y-0 md:grid md:grid-cols-2 md:gap-y-4 md:gap-x-5">
<a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="quicktour"
><div class="w-full text-center bg-gradient-to-br from-blue-400 to-blue-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">ابدأ</div>
<p class="text-gray-700">ابدأ من هنا إذا كنت جديدًا على 🤗 PEFT للحصول على نظرة عامة على الميزات الرئيسية للمكتبة، وكيفية تدريب نموذج باستخدام طريقة PEFT.</p>
</a>
<a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="./task_guides/image_classification_lora"
><div class="w-full text-center bg-gradient-to-br from-indigo-400 to-indigo-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">أدلة كيفية الاستخدام</div>
<p class="text-gray-700">أدلة عملية توضح كيفية تطبيق طرق PEFT المختلفة عبر أنواع مختلفة من المهام مثل تصنيف الصور ونمذجة اللغة السببية والتعرف التلقائي على الكلام، وأكثر من ذلك. تعلم كيفية استخدام 🤗 PEFT مع DeepSpeed وFully Sharded Data Parallel scripts.</p>
</a>
<a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="./conceptual_guides/lora"
><div class="w-full text-center bg-gradient-to-br from-pink-400 to-pink-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">أدلة مفاهيمية</div>
<p class="text-gray-700">احصل على فهم نظري أفضل لكيفية مساهمة LoRA ومختلف طرق الإشارة الناعمة في تقليل عدد المعلمات القابلة للتدريب لجعل التدريب أكثر كفاءة.</p>
</a>
<a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="./package_reference/config"
><div class="w-full text-center bg-gradient-to-br from-purple-400 to-purple-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">مرجع</div>
<p class="text-gray-700">الأوصاف الفنية لكيفية عمل فئات 🤗 PEFT والطرق.</p>
</a>
</div>
</div>

<iframe
src="https://stevhliu-peft-methods.hf.space"
frameborder="0"
width="850"
height="620"
></iframe>