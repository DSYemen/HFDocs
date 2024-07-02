# LyCORIS

[LyCORIS](https://hf.co/papers/2309.14859) (Lora beYond Conventional methods, Other Rank adaptation Implementations for Stable diffusion) هي أدوات تفكيك مصفوفة شبيهة بـ LoRA تقوم بتعديل طبقة cross-attention في شبكة UNet. ويورث كل من الأسلوبين [LoHa](loha) و [LoKr](lokr) من فئات `Lycoris` هنا.

## LycorisConfig

[[autodoc]] tuners.lycoris_utils.LycorisConfig

## LycorisLayer

[[autodoc]] tuners.lycoris_utils.LycorisLayer

## LycorisTuner

[[autodoc]] tuners.lycoris_utils.LycorisTuner