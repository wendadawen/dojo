# 资料冲突

- 文本编码器词表规模｜paper/clip_paper.tex Choosing and Scaling a Model 写「a lower-cased byte pair encoding (BPE) representation of the text with a 49,152 vocab size」；同一文件 Model Hyperparameters 的超参表写「Vocabulary size & 49408」｜页面按正文写 49152
- 文本编码器最长长度｜paper/clip_paper.tex Choosing and Scaling a Model 写「the max sequence length was capped at 76」；repo/CLIP/clip/clip.py L205、L215 与 README.md 写 context length = 77｜页面按论文正文写 76
- MNIST 零样本准确率｜paper/clip_paper.tex Limitations 写「CLIP only achieves 88\% accuracy on the handwritten digits of MNIST」；同一文件附录 table:ocr（L1253）写 88.4｜页面按 Limitations 正文写 88
