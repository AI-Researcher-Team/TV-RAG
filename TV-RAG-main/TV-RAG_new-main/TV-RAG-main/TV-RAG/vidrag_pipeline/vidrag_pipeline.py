import requests
from PIL import Image
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, CLIPProcessor, CLIPModel, WhisperForConditionalGeneration, WhisperProcessor
import copy
from decord import VideoReader, cpu
import numpy as np
import json
import re
import time
from tqdm import tqdm
import os
import easyocr
from rag_retriever_dynamic import retrieve_documents_with_dynamic
import re
import ast
import socket
import pickle
from filter_keywords import filter_keywords
from scene_graph import generate_scene_graph_description
import torchaudio, ffmpeg

short_count = 0
short_cor = 0
medium_count = 0
medium_cor  = 0
long_count = 0
long_cor = 0

short_count_1 = 0
short_cor_1 = 0
medium_count_1 = 0
med_cor_1 = 0
long_count_1 = 0
long_cor_1 = 0

total_time = 0
video_count = 0

max_frames_num = 16
clip_model = CLIPModel.from_pretrained("/root/autodl-tmp/Video-RAG-master/LLaVA-NeXT/models/models--openai--clip-vit-large-patch14-336/snapshots/ce19dc912ca5cd21c8a653c79e251e808ccabcd1", torch_dtype=torch.float16, device_map="auto",cache_dir='models')
clip_processor = CLIPProcessor.from_pretrained("/root/autodl-tmp/Video-RAG-master/LLaVA-NeXT/models/models--openai--clip-vit-large-patch14-336/snapshots/ce19dc912ca5cd21c8a653c79e251e808ccabcd1",cache_dir='models')
whisper_model = WhisperForConditionalGeneration.from_pretrained(
    "/root/autodl-tmp/Video-RAG-master/LLaVA-NeXT/models/models--openai--whisper-large/snapshots/4ef9b41f0d4fe232daafdb5f76bb1dd8b23e01d7",
    torch_dtype=torch.float16,
    device_map="auto",cache_dir='models'
)
whisper_processor = WhisperProcessor.from_pretrained("/root/autodl-tmp/Video-RAG-master/LLaVA-NeXT/models/models--openai--whisper-large/snapshots/4ef9b41f0d4fe232daafdb5f76bb1dd8b23e01d7",cache_dir='models')


def select_key_frames(frames, query_text, clip_model, clip_processor, initial_threshold=0.03, max_iter=1, min_frames=32):
    """
    Select key frames from a sequence of frames based on their similarity to a query text using CLIP model.
    
    Args:
        frames: List of input frames (PIL Images or similar)
        query_text: Text query to compare frames against
        clip_model: Loaded CLIP model
        clip_processor: CLIP processor for feature extraction
        initial_threshold: Initial similarity threshold for frame selection
        max_iter: Maximum number of threshold adjustment iterations
        min_frames: Minimum number of frames to return (default: 5)
    
    Returns:
        selected_frames: List of selected frames
        key_indices: Indices of selected frames in the original sequence
    """
    # Extract frame features
    frame_features = []
    for frame in frames:
        inputs = clip_processor(images=frame, return_tensors="pt").to(clip_model.device)
        with torch.no_grad():
            features = clip_model.get_image_features(**inputs)
        frame_features.append(features.cpu())
    frame_features = torch.stack(frame_features, dim=0).squeeze(1)

    # Extract text features
    text_inputs = clip_processor(text=[query_text], return_tensors="pt", padding=True,  truncation=True, max_length=77).to(clip_model.device)
    with torch.no_grad():
        text_features = clip_model.get_text_features(**text_inputs).cpu()

    # Initialize selection variables
    selected_indices = []
    prev_count = 0
    threshold = initial_threshold
    iteration = 0
    
    # Iterative threshold adjustment
    while iteration < max_iter:
        # Calculate similarities
        similarities = torch.mm(frame_features, text_features.T).squeeze(1).numpy()
        
        # Calculate entropy-based weights
        prob = similarities / (similarities.sum() + 1e-9)  # Add small epsilon to avoid division by zero
        entropy = -np.sum(prob * np.log(prob + 1e-9))  # Add small epsilon to avoid log(0)
        time_weights = similarities * entropy / (similarities.sum() + 1e-9)
        
        # Select frames above threshold
        key_indices = [i for i, w in enumerate(time_weights) if w > threshold]
        
        # Stop if no new frames are selected
        if len(key_indices) == prev_count:
            break
            
        selected_indices = key_indices
        prev_count = len(key_indices)
        threshold *= 1.2  # Increase threshold for next iteration
        iteration += 1

    # Ensure we return at least min_frames
    if len(selected_indices) < min_frames:
        # Fallback: select top N frames by similarity
        similarities = torch.mm(frame_features, text_features.T).squeeze(1).numpy()
        top_indices = np.argsort(similarities)[-min_frames:]
        selected_indices = sorted(top_indices.tolist())

    return [frames[i] for i in selected_indices], selected_indices


def process_video(video_path, max_frames_num, fps=1, force_sample=False):
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3))
    vr = VideoReader(video_path, ctx=cpu(),num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps()/fps)
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time = [i/fps for i in frame_idx]
    if len(frame_idx) > max_frames_num or force_sample:
        sample_fps = max_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i/vr.get_avg_fps() for i in frame_idx]
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames, frame_time, video_time

def extract_audio(video_path, audio_path):
    if not os.path.exists(audio_path):
        ffmpeg.input(video_path).output(audio_path, acodec='pcm_s16le', ac=1, ar='16k').run()


def evaluate_answer(question: str, answer: str, context_info: dict, video) -> list[str]:
    critique_prompt = f"""
        Analyze the following Q&A pair based on the provided context and video content:
        - Question: '{text}'
        - Options: {option}
        - Answer: '{answer}'
        - Context: {json.dumps(context_info, indent=2)}
        
        Evaluation Criteria:
        1. Check if the answer fully addresses the question
        2. Verify all key elements from context are included
        3. Assess whether video content supports the answer
        
        If incomplete, generate 1-3 ultra-specific clarification questions (≤12 words each) following these rules:
        - Must start with: "What", "Where", "When", "Which", or "How"
        - Must reference concrete elements from context/video
        - No vague pronouns ("it", "they") - use specific nouns
        - Examples: "Which timestamp shows the error?" or "How many frames were processed?"
        
        Output format: 
        - Return [] for complete answers
        - Return ["specific_question1?"] (specifically 1 questions) for incomplete answers
        """

    critique_response = llava_inference(critique_prompt, video)
    if critique_response:
        return critique_response
    
    return []



def chunk_audio(audio_path, chunk_length_s=30):
    speech, sr = torchaudio.load(audio_path)
    speech = speech.mean(dim=0)  
    speech = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(speech)  
    num_samples_per_chunk = chunk_length_s * 16000 
    chunks = []
    for i in range(0, len(speech), num_samples_per_chunk):
        chunks.append(speech[i:i + num_samples_per_chunk])
    return chunks

def transcribe_chunk(chunk):

    inputs = whisper_processor(chunk, return_tensors="pt")
    inputs["input_features"] = inputs["input_features"].to(whisper_model.device, torch.float16)
    with torch.no_grad():
        predicted_ids = whisper_model.generate(
            inputs["input_features"],
            no_repeat_ngram_size=2,
            early_stopping=True
        )
    transcription = whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription

def get_asr_docs(video_path, audio_path):

    full_transcription = []
    try:
        extract_audio(video_path, audio_path)
    except:
        return full_transcription
    audio_chunks = chunk_audio(audio_path, chunk_length_s=30)
    
    for chunk in audio_chunks:
        transcription = transcribe_chunk(chunk)
        full_transcription.append(transcription)

    return full_transcription

def get_ocr_docs(frames):
    reader = easyocr.Reader(['en']) 
    text_set = []
    ocr_docs = []
    for img in frames:
        ocr_results = reader.readtext(img)
        det_info = ""
        for result in ocr_results:
            text = result[1]
            confidence = result[2]
            if confidence > 0.5 and text not in text_set:
                det_info += f"{text}; "
                text_set.append(text)
        if len(det_info) > 0:
            ocr_docs.append(det_info)

    return ocr_docs

    
def save_frames(frames, file_name):
    file_paths = []
    for i, frame in enumerate(frames):
        img = Image.fromarray(frame)
        file_path = f'restore/{file_name}/frame_{i}.png'
        img.save(file_path)
        file_paths.append(file_path)
    return file_paths
    
def get_det_docs(frames, prompt, file_name):
    prompt = ",".join(prompt)
    frames_path = save_frames(frames, file_name)
    res = []
    if len(frames) > 0:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect(('0.0.0.0', 9999))
        data = (frames_path, prompt)
        client_socket.send(pickle.dumps(data))
        result_data = client_socket.recv(4096)
        try:
            res = pickle.loads(result_data)
        except:
            res = []
    return res

def det_preprocess(det_docs, location, relation, number):

    scene_descriptions = []

    for det_doc_per_frame in det_docs:
        objects = []
        scene_description = ""
        if len(det_doc_per_frame) > 0:
            for obj_id, objs in enumerate(det_doc_per_frame.split(";")):
                obj_name = objs.split(":")[0].strip()
                obj_bbox = objs.split(":")[1].strip()
                obj_bbox = ast.literal_eval(obj_bbox)
                objects.append({"id": obj_id, "label": obj_name, "bbox": obj_bbox})

            scene_description = generate_scene_graph_description(objects, location, relation, number)
        scene_descriptions.append(scene_description)
    
    return scene_descriptions

device = "cuda"
overwrite_config = {}
tokenizer, model, image_processor, max_length = load_pretrained_model(
    "/root/autodl-tmp/Video-RAG-master/LLaVA-NeXT/models/models--lmms-lab--LLaVA-Video-7B-Qwen2/snapshots/013210b3aff822f1558b166d39c1046dd109520f", 
    None, 
    "llava_qwen", 
    torch_dtype="bfloat16", 
    device_map="auto", 
    overwrite_config=overwrite_config) # Add any other thing you want to pass in llava_model_args
model.eval()
conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models

def llava_inference(qs, video):
    if video is not None:
        time_instruciton = f"The video lasts for {video_time:.2f} seconds, and {len(video[0])} frames are uniformly sampled from it. These frames are located at {frame_time}.Please answer the following questions related to this video."
        question = DEFAULT_IMAGE_TOKEN + f"{time_instruciton}\n" + qs
    else:
        question = qs
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    
    if video is not None:
        cont = model.generate(
            input_ids,
            images=video,
            modalities= ["video"],
            do_sample=False,
            temperature=0,
            max_new_tokens=16,
            top_p=1.0,
            num_beams=1
        )
    else:
        cont = model.generate(
            input_ids,
            images=video,
            modalities= ["video"],
            do_sample=False,
            temperature=0,
            max_new_tokens=4096,
        )
    
    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
    return text_outputs,

rep_list = []
rag_threshold = 0.3
clip_threshold = 0.3
beta = 3.0 
USE_OCR = True
USE_ASR = True
USE_DET = True
print(f"---------------OCR{rag_threshold}: {USE_OCR}-----------------")
print(f"---------------ASR{rag_threshold}: {USE_ASR}-----------------")
print(f"---------------DET{beta}-{clip_threshold}: {USE_DET}-----------------")
print(f"---------------Frames: {max_frames_num}-----------------")

import os
def find_mp4(root_folder):
    mp4_filenames = []
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.endswith('.mp4'):
                mp4_filenames.append(filename.replace('.mp4',''))
    return mp4_filenames
    
def find_mp4_files(root_folder):
    mp4_files = []
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.endswith('.mp4'):
                mp4_files.append(os.path.join(dirpath, filename))
    return mp4_files
    
video_path = "video"  # your video path


import pandas as pd

# 读取 Parquet 文件
file_path = 'test.parquet'
df = pd.read_parquet(file_path, engine='fastparquet')
print(f'df is {df}')
names = df['videoID']
questions = df['question_id']

short = 0
medium = 0
long = 0
short_cor = 0
med_cor = 0
long_cor = 0


short_count = 0 
medium_count = 0 
long_count = 0


videos = find_mp4(video_path)
print(f'------the found videos are ------ {videos}')
paths = find_mp4_files(video_path)
print(paths)
for question in questions:
    cols = df[df['question_id']==question]
    name = cols['videoID'].values[0]
    print(f'-----the video is {videos}-----')
    if name in videos:
        duration = cols['duration'].values[0]
        if duration =='short':
            short +=1
            # if short <= 114:
            #     continue
            pass
        if duration == 'medium':
            medium += 1
            pass
        if duration == 'long':
            long +=1
            pass
        text = cols['question'].values[0]
        option = cols['options'].values[0]
        standard_answer = cols['answer'].values[0]

        for path in paths:
            print(f'-----the current path is -----{path}')
            if name in path:
                video_path = path
            else:
                continue
        print(f'--------------the question is {text}--------------')

    else:
        continue
    try:
        frames, frame_time, video_time = process_video(video_path, max_frames_num, 1, force_sample=True)
    except Exception as e:
        print(f"Error processing video {video_path}: {str(e)}")
        print("Skipping this video...")
        continue  # or return None, depending on your context
    print(f'-------starting the process of inference--------')
    raw_video = [f for f in frames]
    video = image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].cuda().bfloat16()
    video = [video]
   
    start_time = time.time()
    USE_DET = True
    USE_OCR = True
    USE_ASR = True
    if USE_DET:
        print(f'--------using the DET tools--------')
        video_tensor = []
        for frame in raw_video:
            processed = clip_processor(images=frame, return_tensors="pt")["pixel_values"].to(clip_model.device, dtype=torch.float16)
            video_tensor.append(processed.squeeze(0))
        video_tensor = torch.stack(video_tensor, dim=0)

    if USE_OCR:
        print(f'--------using the OCR tools--------')
        ocr_docs_total = get_ocr_docs(frames)
    if USE_ASR: 
         print(f'--------using the ASR tools--------')
         if os.path.exists(os.path.join("restore/audio", os.path.basename(video_path).split(".")[0] + ".txt")): 
             with open(os.path.join("restore/audio", os.path.basename(video_path).split(".")[0] + ".txt"), 'r', encoding='utf-8') as f: 
                 asr_docs_total = f.readlines() 
         else: 
             audio_path = os.path.join("restore/audio", os.path.basename(video_path).split(".")[0] + ".wav") 
             asr_docs_total = get_asr_docs(video_path, audio_path) 
             with open(os.path.join("restore/audio", os.path.basename(video_path).split(".")[0] + ".txt"), 'w', encoding='utf-8') as f: 
                 for doc in asr_docs_total: 
                     f.write(doc + '\n') 
    
    print(f'-----# step 0: get cot information--------')
    # step 0: get cot information
    retrieve_pmt_0 = "Question: " + text + '\n' + " ".join(option)
    retrieve_pmt_0 += "\nTo answer the question step by step, you can provide your retrieve request to assist you by the following json format:"
    retrieve_pmt_0 += '''{
        "ASR": Optional[str]. The subtitles of the video that may relavent to the question you want to retrieve, in two sentences. If you no need for this information, please return null.
        "DET": Optional[list]. (The output must include only physical entities, not abstract concepts, less than five entities) All the physical entities and their location related to the question you want to retrieve, not abstract concepts. If you no need for this information, please return null.
        "TYPE": Optional[list]. (The output must be specified as null or a list containing only one or more of the following strings: 'location', 'number', 'relation'. No other values are valid for this field) The information you want to obtain about the detected objects. If you need the object location in the video frame, output "location"; if you need the number of specific object, output "number"; if you need the positional relationship between objects, output "relation". 
    }
    ## Example 1: 
    Question: How many blue balloons are over the long table in the middle of the room at the end of this video? A. 1. B. 2. C. 3. D. 4.
    Your retrieve can be:
    {
        "ASR": "The location and the color of balloons, the number of the blue balloons.",
        "DET": ["blue ballons", "long table"],
        "TYPE": ["relation", "number"]
    }
    ## Example 2: 
    Question: In the lower left corner of the video, what color is the woman wearing on the right side of the man in black clothes? A. Blue. B. White. C. Red. D. Yellow.
    Your retrieve can be:
    {
        "ASR": null,
        "DET": ["the man in black", "woman"],
        "TYPE": ["location", "relation"]
    }
    ## Example 3: 
    Question: In which country is the comedy featured in the video recognized worldwide? A. China. B. UK. C. Germany. D. United States.
    Your retrieve can be:
    {
        "ASR": "The country recognized worldwide for its comedy.",
        "DET": null,
        "TYPE": null
    }
    Note that you don't need to answer the question in this step, so you don't need any infomation about the video of image. You only need to provide your retrieve request (it's optional), and I will help you retrieve the infomation you want. Please provide the json format.'''
    print(f'------- the question is {question}------- ')
    def generate_query_context(question: str) -> str:
        """
        Generates a helpful background context and 2-3 paraphrased variants of the input question
        to enhance query understanding for downstream reasoning.
    
        Args:
            question (str): Original user query.
    
        Returns:
            str: A formatted string containing the generated background context and paraphrased questions.
        """
        prompt = (
            "Given a question, first generate a helpful background context. "
            "Then, provide 2-3 alternative phrasings of the question with similar meaning.\n\n"
            f"Question: {question}\n\n<background>\n"
        )
    
        result = llava_inference(prompt)[0]  # assuming llava_inference returns a tuple
        return result.strip()
    
    
    context = generate_query_context(question)

    qs = ""

    if USE_ASR or USE_DET or USE_OCR:
        print(f'------- the starting the inference is ------- ')
        json_request = llava_inference(retrieve_pmt_0, None)

        # step 1: get docs information
        query = [text]


        torch.cuda.empty_cache()

        # APE fetch
        if USE_DET:
            det_docs = []
            try:
                request_det = json.loads(json_request)["DET"]
                request_det = filter_keywords(request_det)
                clip_text = ["A picture of " + txt for txt in request_det]
                if len(clip_text) == 0:
                    clip_text = ["A picture of object"]
            except:
                request_det = None
                clip_text = ["A picture of object"]

            clip_inputs = clip_processor(text=clip_text, return_tensors="pt", padding=True, truncation=True).to(clip_model.device)
            clip_img_feats = clip_model.get_image_features(video_tensor)
            with torch.no_grad():
                text_features = clip_model.get_text_features(**clip_inputs)
                sim_matrix = (clip_img_feats @ text_features.T).cpu() 
            similarities = sim_matrix.mean(dim=1).numpy()  
            p_all = similarities / (np.sum(similarities) + 1e-8)
            frame_entropies = -p_all * np.log(p_all + 1e-8)  
            weighted_scores = similarities * frame_entropies  
            del clip_inputs, clip_img_feats, text_features
            torch.cuda.empty_cache()
            det_top_idx = [i for i, score in enumerate(weighted_scores) if score > clip_threshold]
                
            if request_det is not None and len(request_det) > 0:
                # process directly
                det_docs = get_det_docs(frames[det_top_idx], request_det, file_name)  

                L, R, N = False, False, False
                try:
                    det_retrieve_info = json.loads(json_request)["TYPE"]
                except:
                    det_retrieve_info = None
                if det_retrieve_info is not None:
                    if "location" in det_retrieve_info:
                        L = True
                    if "relation" in det_retrieve_info:
                        R = True
                    if "number" in det_retrieve_info:
                        N = True
                det_docs = det_preprocess(det_docs, location=L, relation=R, number=N)  # pre-process of APE information


        # OCR fetch
        if USE_OCR:
            try:
                request_det = json.loads(json_request)["DET"]
                request_det = filter_keywords(request_det)
            except:
                request_det = None
            ocr_docs = []
            if len(ocr_docs_total) > 0:
                ocr_query = query.copy()
                if request_det is not None and len(request_det) > 0:
                    ocr_query.extend(request_det)
                # ocr_docs, _ = retrieve_documents_with_dynamic(ocr_docs_total, ocr_query, threshold=rag_threshold)
                ocr_docs, _, _ = retrieve_documents_with_temporal_rankning(ocr_docs_total, ocr_query, threshold=rag_threshold)

        # ASR fetch
        if USE_ASR:
            asr_docs = []
            try:
                request_asr = json.loads(json_request)["ASR"]
            except:
                request_asr = None
            if len(asr_docs_total) > 0:
                asr_query = query.copy()
                if request_asr is not None:
                    asr_query.append(request_asr)
                # asr_docs, _ = retrieve_documents_with_dynamic(asr_docs_total, asr_query, threshold=rag_threshold)
                asr_docs, _, _ = retrieve_documents_with_temporal_rankning(asr_docs_total, asr_query,threshold=rag_threshold)
    
    
    if USE_DET and len(det_docs) > 0:
        for i, info in enumerate(det_docs):
            if len(info) > 0:
                qs += f"Frame {str(det_top_idx[i]+1)}: " + info + "\n"
        if len(qs) > 0:
            qs = f"\nVideo have {str(max_frames_num)} frames in total, the detected objects' information in specific frames: " + qs
    if USE_ASR and len(asr_docs) > 0:
        qs += "\nVideo Automatic Speech Recognition information (given in chronological order of the video): " + " ".join(asr_docs)
    if USE_OCR and len(ocr_docs) > 0:
        qs += "\nVideo OCR information (given in chronological order of the video): " + "; ".join(ocr_docs)
    qs += (
    "\nThe following is background knowledge and reformulated versions of the question "
    "to help you better understand and answer it:\n\n"
    + context.strip() + "\n\n"
    )
    
    qs += (
        "Select the best answer to the following multiple-choice question based on the video "
        "and the information (if given). Respond with only the letter (A, B, C, or D) of the correct option.\n\n"
    )
    qs += f"Question: {text}\n"
    qs += " ".join(option) + "\n"
    qs += "The best answer is:"
    print(f'-----starting the inference -------')
    (result,) = llava_inference(qs, video)
    result = result.strip('.')
    '''
    res = res.replace('(','')
    res = res.replace(')','')
    res = res.replace("'",'')
    res = res.replace('.','')
    res = res.replace(',','')
    '''
    
    model_answer = str(result)
    
    end_time = time.time()
   
    video_count += 1
    total_time += (end_time - start_time)
    avg_total_time = (total_time /  video_count)
    if duration == 'short':
                short_count += 1
               
                if model_answer == standard_answer:
                    short_cor += 1
     
    if duration == 'medium':
                medium_count += 1
              
                if model_answer == standard_answer:
                    med_cor += 1
        
    if duration == 'long':
                long_count += 1
               
                if model_answer == standard_answer:
                    long_cor += 1
        
    short_acc = short_cor / short_count if short_count else 0
    medium_acc = med_cor / medium_count if medium_count else 0
    long_acc = long_cor / long_count if long_count else 0
    total_cor = short_cor + med_cor + long_cor
    total_count = short_count + medium_count + long_count
    overall_acc = (short_cor + med_cor + long_cor) / (short_count + medium_count + long_count) if  (short_count + medium_count + long_count) else 0
    print(f"📊 当前multiagent准确率(w reflection)：short: {short_acc:.2%} ({short_cor}/{short_count}), "
                  f"medium: {medium_acc:.2%} ({med_cor}/{medium_count}), "
                  f"long: {long_acc:.2%} ({long_cor}/{long_count}), "
                  f"overall: {overall_acc:.2%} ({total_cor}/{total_count})",
                  f'------ the current total time is {avg_total_time : .2f}------')
