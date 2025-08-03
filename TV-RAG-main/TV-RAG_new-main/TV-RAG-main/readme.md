# TV-RAG: A Temporal-aware and Visual Entropy-Weighted Framework for Long Video Retrieval and Understanding

## 😮 Highlights
- TV-RAG is the first model to address video RAG tasks from
both temporal and semantic perspectives within a unified
training-free framework. It integrates time-aligned OCR,
ASR, and object detection data, ensuring the relevance of text
queries in multimedia contexts and enhancing long video
comprehension.
- TV-RAG employs a novel temporal-aware and semanticentropy weighting strategy that ensures keyframes are evenly
distributed over time. This reduces redundancy and improves
representativeness, enhancing overall video understanding.
- Extensive experiments on several well-established benchmarks show that TV-RAG achieves state-of-the-art performance and outperforms other models. It can also be used as
a plug-in for other models. 

## 🔨 Usage

This repo is built upon LLaVA-NeXT:

- Step 1: Clone and build LLaVA-NeXT conda environment:

```
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip  # Enable PEP 660 support.
pip install -e ".[train]"
```
Then install the following packages in llava environment:
```
pip install spacy faiss-cpu easyocr ffmpeg-python
pip install torch==2.1.2 torchaudio numpy
python -m spacy download en_core_web_sm
# Optional: pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.0.0/en_core_web_sm-3.0.0.tar.gz
```

- Step 2: Clone and build another conda environment for APE by: 

```

pip3 install -r requirements.txt
python3 -m pip install -e .
```

- Step 3: Copy all the files in `vidrag_pipeline` under the root dir of LLaVA-NeXT;

- Step 4: Copy all the files in `ape_tools` under the `demo` dir of APE;

- Step 5: Opening a service of APE by running the code under `APE/demo`:

```
python demo/ape_service.py
```

- Step 6: You can now run our pipeline build upon LLaVA-Video-7B by:

```
python vidrag_pipeline.py
```

- Note that you can also use our pipeline in any LVLMs by implementing some modifications in `vidrag_pipeline.py`:
```
1. The video-language model you load (line #161).
2. The llava_inference() function, make sure your model supports both inputs with/without video (line #175).
3. The process_video() function may suit your model (line #34).
4. The final prompt may suit your model (line #366).
```
