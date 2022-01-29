"""Contains StyleGAN2 modules.

StyleGAN2 is from the NVIDIA paper "Analyzing and Improving the Quality of
StyleGAN," available here: https://arxiv.org/pdf/1912.04958.pdf

Some stylistic decisions are made to preserve similarity with the NVIDIA code
base, although it is TFv1.14 and we implement in TFv2. The original code base
can be found here: https://github.com/NVlabs/stylegan2

Additional sources.
LoGANv2 (https://arxiv.org/abs/1909.09974): Discusses use of conditioning
labels.
"""
